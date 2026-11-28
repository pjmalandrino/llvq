//! O13: arm v9k8, the KV cache as group-affine int8 (`pers/e1.rs::p4e_k8_edits`),
//! against kv8_ref. One file because every gate here reads the int8 cache.

use super::*;

// ---- O13: arm v9k8, the KV cache as group-affine int8 (pers/e1.rs::p4e_k8_edits) ----
//
// The reference is llvq-cuda/src/kv8_ref.rs: its quantizer, its layout, and
// attn_ref's arithmetic on its values. The kernel text (pers_4e_k8.cuh) runs
// on the persistent side of form 7; the served side carries an independent
// C++ port (host_pers.cpp, ref_k8_quant_row and ref_attn_row_k8), held here
// against the same Rust, so the arm's chain can be compared with a served
// chain on the int8 cache (command 'G' + 512).

/// A side-1 build of form 7 from a copy with one edit (exactly once).
fn mutant_v9(tag: &str, file: &str, from: &str, to: &str) -> PathBuf {
    let root = mutant_root(tag, &[(file, from, to)]);
    build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"])
}

/// Rows of 128 binary16 values that reach every branch of the quantizer: a
/// constant group (scale 0), a minimum of -0, two groups whose scale a true
/// division by 255 would move, exact ties of half a scale, a subnormal scale
/// that clamps at 255; then random rows of several spreads.
fn k8_rows(n: usize) -> Vec<u16> {
    use llvq_cuda::attn_ref::{f2h, fill};
    let mut rows = fill(n * 128, 0x6b38, 1.0);
    for (p, row) in rows.chunks_exact_mut(128).enumerate() {
        let amp = [0.01f32, 0.3, 1.0, 4.0, 60.0][p % 5];
        let r = fill(128, 0x6b38 + p as u64, amp);
        row.copy_from_slice(&r);
    }
    let put = |rows: &mut Vec<u16>, p: usize, g: usize, vals: &[u16]| {
        let o = p * 128 + g * 64;
        rows[o..o + 64].fill(vals[vals.len() - 1]);
        rows[o..o + vals.len()].copy_from_slice(vals);
    };
    put(&mut rows, 0, 0, &[f2h(0.375)]);
    put(&mut rows, 1, 1, &[0x8000, f2h(0.5), f2h(0.25)]);
    put(&mut rows, 2, 0, &[0x94c6, 0x3f46, 0]);
    put(&mut rows, 2, 1, &[0xd558, 0x204f, 0]);
    put(&mut rows, 3, 0, &[f2h(255.0 / 256.0), f2h(2.5 / 256.0), f2h(7.5 / 256.0), f2h(3.25 / 256.0), 0]);
    put(&mut rows, 4, 1, &[f2h((255.0f64 * 1.4 * (-24f64).exp2()) as f32), 0]);
    put(&mut rows, 5, 0, &[0x8000, f2h(0.5), f2h(0.25)]);
    // a range whose scale one ulp of the reciprocal moves (0x2c7f against 0x2c80)
    put(&mut rows, 6, 1, &[0xcc7b, 0x01c6, 0]);
    rows
}

/// The quantizer: the persistent side's fill (pers_kv8_fill, its quantizer
/// shared with the append) and the served side's port both write
/// kv8_ref's bytes, every position of a head, on every branch.
#[test]
fn the_k8_quantizer_is_kv8_ref() {
    use llvq_cuda::kv8_ref;
    let n = 48usize;
    let rows = k8_rows(n);
    let mut c = kv8_ref::Cache8::new(1, n);
    c.fill(&rows, &rows, n, 0, n);
    assert_eq!(c.k, c.v);
    let mut cmd = Cmd::new(b'k').u32(n as u32).0;
    for x in &rows {
        cmd.extend(x.to_le_bytes());
    }
    let mut want = c.k.clone();
    want.extend(&c.v);
    assert_eq!(run(persistent_v9(), &cmd), want, "the kernel's fill against kv8_ref");
    assert_eq!(run(served(), &cmd), want, "the served port against kv8_ref");
    // the planted rows took their branches
    let g = |p: usize, gi: usize| kv8_ref::quant_group(&rows[p * 128 + gi * 64..p * 128 + gi * 64 + 64]);
    assert_eq!(g(0, 0).s, 0);
    assert_eq!(g(1, 1).b, 0);
    assert_eq!((g(2, 0).s, g(2, 1).s), (0x1f4f, 0x355e));
    assert_eq!(g(4, 1).s, 1);
    assert_eq!(g(6, 1).s, 0x2c7f);
    let f = "llvq-cuda/kernels/pers_4e_k8.cuh";
    for (tag, from, to) in [
        ("k8_tie_even", "(__fsub_rn(t, fl) >= 0.5f ? 1u : 0u)", "(__fsub_rn(t, fl) > 0.5f ? 1u : 0u)"),
        ("k8_inv255", "0x3b808081u", "0x3b808082u"),
        ("k8_zero_sign", "    if (mn0 == 0.0f) mn0 = 0.0f;\n", ""),
        ("k8_no_clamp", "return q > 255u ? 255u : q;", "return q & 255u;"),
        ("k8_sb_order", "sb[0] = (u32)s0 | ((u32)b0 << 16);", "sb[0] = (u32)b0 | ((u32)s0 << 16);"),
        ("k8_group_cols", "for (u32 k = 0; k < 4u; ++k) x[k] = h2f(__ldcg(src + lane + 32u * k));", "for (u32 k = 0; k < 4u; ++k) x[k] = h2f(__ldcg(src + lane + 32u * (k ^ 1u)));"),
    ] {
        let b = mutant_v9(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "k8 quantizer mutant {tag} survived");
        eprintln!("k8 quantizer mutant {tag}: killed");
    }
}

/// One 'x' command: q, a binary16 cache quantized into int8 for positions
/// below l, the bytes of every position at or past l set to 0xFF (a scale of
/// NaN: a position past the row that entered the arithmetic would show), and
/// the reference's output and f32 state.
fn k8_attn_cmd(n_kv: u32, groups: u32, cap: u32, l: u32, grid: u32, threads: u32, seed: u64) -> (Vec<u8>, Vec<u16>, Vec<u32>) {
    use llvq_cuda::{attn_ref, kv8_ref};
    let nh = (n_kv * groups) as usize;
    let q = attn_ref::fill(nh * 128, seed, 2.0);
    let mut k = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 1, 1.0);
    let v = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 2, 1.0);
    for p in 0..3usize {
        let (a, b) = (p * 128, (p + 9) * 128);
        if b + 128 <= k.len() / n_kv as usize {
            let row: Vec<u16> = k[a..a + 128].to_vec();
            k[b..b + 128].copy_from_slice(&row);
        }
    }
    let (cap_u, l_u) = (cap as usize, l as usize);
    let mut c8 = kv8_ref::Cache8::new(n_kv as usize, cap_u);
    c8.fill(&k, &v, cap_u, 0, l_u);
    for h in 0..n_kv as usize {
        for p in l_u..cap_u {
            let o = kv8_ref::blk_offset(h, p, cap_u);
            let j = p % 8;
            for buf in [&mut c8.k, &mut c8.v] {
                buf[o + j * 128..o + (j + 1) * 128].fill(0xFF);
                buf[o + kv8_ref::CODES + j * 8..o + kv8_ref::CODES + (j + 1) * 8].fill(0xFF);
            }
        }
    }
    let c = attn_ref::score_scale(128);
    let want = attn_ref::attend(&q, &c8, n_kv as usize, groups as usize, l_u - 1, 1, c, attn_ref::UNIT_SERVED);
    let mut state = Vec::with_capacity(nh * 130);
    for hq in 0..nh {
        let qf: [f32; 128] = std::array::from_fn(|i| attn_ref::h2f(q[hq * 128 + i]));
        let mut total = attn_ref::St::empty();
        for s in 0..l_u.div_ceil(attn_ref::SPLIT) {
            attn_ref::st_merge(&mut total, &attn_ref::split_state(&qf, &c8, hq / groups as usize, s, l_u, c, attn_ref::UNIT_SERVED));
        }
        state.extend(total.a.iter().map(|x| x.to_bits()));
        state.push(total.m.to_bits());
        state.push(total.d.to_bits());
    }
    let mut cmd = Cmd::new(b'x').u32(n_kv).u32(groups).u32(cap).u32(l).u32(grid).u32(threads).f32(c).0;
    for x in &q {
        cmd.extend(x.to_le_bytes());
    }
    cmd.extend(&c8.k);
    cmd.extend(&c8.v);
    (cmd, want, state)
}

/// The arm's attention phase on the persistent grid equals attn_ref's
/// arithmetic on kv8_ref's values bit for bit, on the lengths, groups and
/// grids of `the_4e_attention_phase_is_attn_ref`: inside one block, at unit
/// and split boundaries, across several splits, a last block partly past the
/// row; the split fold leaves its tickets at zero; the served port agrees.
#[test]
fn the_k8_attention_phase_is_kv8_ref() {
    let lens = [(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 4, 192, 150), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300), (3, 5, 2304, 2049)];
    for &(n_kv, groups, cap, l) in &lens {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96), (1, 512)] {
            let (cmd, want, state) = k8_attn_cmd(n_kv, groups, cap, l, grid, threads, 0xb88 + l as u64);
            let got = run(persistent_v9(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (0, 0), "every ticket back at zero");
            assert_eq!(u32s(&got[n + 8..]), state, "the f32 state before the division, l {l}");
        }
        let (cmd, want, _) = k8_attn_cmd(n_kv, groups, cap, l, 3, 64, 0xb88 + l as u64);
        assert_eq!(u16s(&run(served(), &cmd)[..want.len() * 2]), want, "the served port of the k8 reference, l {l}");
    }
    let f = "llvq-cuda/kernels/pers_4e_k8.cuh";
    let wanted = |n_kv: u32, groups: u32, cap: u32, l: u32| {
        let (cmd, want, state) = k8_attn_cmd(n_kv, groups, cap, l, 3, 64, 0xb88);
        let mut w: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.extend([0u32, 0].iter().flat_map(|x| x.to_le_bytes()));
        w.extend(state.iter().flat_map(|x| x.to_le_bytes()));
        (cmd, w)
    };
    let cmds = [wanted(1, 4, 1408, 1300), wanted(2, 5, 512, 257), wanted(2, 4, 64, 5)];
    for (tag, from, to) in [
        ("k8_v_from_k", "sb8 + (PERS_UPS + ts) * PERS_K8_BLK, nv, c, lane);", "sb8 + ts * PERS_K8_BLK, nv, c, lane);"),
        ("k8_group_sb", "gsb = PERS_K8_CODES + (l16 >> 3) * 4u;", "gsb = PERS_K8_CODES + (l16 >> 4) * 4u;"),
        ("k8_last_block", "const bool valid = p0 < lr;", "const bool valid = p0 + PERS_TT <= lr;"),
        ("k8_dequant_f16", "f[i] = __fmaf_rn(s, (float)((w0 >> (8u * i)) & 0xffu), b);", "f[i] = h2f(f2h(__fmaf_rn(s, (float)((w0 >> (8u * i)) & 0xffu), b)));"),
        ("k8_codes_half", "f[4u + i] = __fmaf_rn(s, (float)((w1 >> (8u * i)) & 0xffu), b);", "f[4u + i] = __fmaf_rn(s, (float)((w1 >> (8u * (3u - i))) & 0xffu), b);"),
        ("k8_block_stride", "+ ch,\n                       valid);", "+ ch + (ch == 67u ? 0u : 1u),\n                       valid);"),
        ("k8_unit_order", "for (u32 uu = 0; uu < PERS_UPS; ++uu) {\n                const float* row", "for (u32 uu = PERS_UPS; uu-- > 0u;) {\n                const float* row"),
        ("k8_head_stride", "const unsigned char* vh = vc8 + (unsigned long long)h * (cap / PERS_TT) * PERS_K8_BLK;", "const unsigned char* vh = vc8;"),
    ] {
        let b = mutant_v9(tag, f, from, to);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "k8 attention mutant {tag} survived");
        eprintln!("k8 attention mutant {tag}: killed");
    }
}

/// The arm's append: v9's q heads and v9's K and V rows (command 'Z', the
/// served replay), the K and V rows quantized by kv8_ref at `pos`, nothing
/// else written; the persistent kernel and the served port agree on every
/// form, position in a block and grid.
#[test]
fn the_k8_append_quantizes_v9s_rows() {
    use llvq_cuda::kv8_ref;
    let (nh, nkv, cap) = (8u32, 2u32, 16u32);
    let cmd = |tag: u8, form: u32, pos: u32, grid: u32| Cmd::new(tag).u32(form).u32(nh).u32(nkv).u32(cap).u32(pos).u32(grid).u32(64).u64(0x20 + pos as u64).0;
    let hb = (cap as usize / 8) * kv8_ref::BLK;
    for form in 0..3u32 {
        for (pos, grid) in [(0u32, 2u32), (7, 3), (15, 4), (9, 1)] {
            let got = run(persistent_v9(), &cmd(b'z', form, pos, grid));
            assert_eq!(got, run(served(), &cmd(b'z', form, pos, grid)), "form {form}, pos {pos}: the port");
            let z = run(served(), &cmd(b'Z', form, pos, grid));
            let nq = nh as usize * 128 * 2;
            assert_eq!(got[..nq], z[..nq], "q out, form {form}, pos {pos}");
            let (k16, v16) = (u16s(&z[nq..nq + nkv as usize * cap as usize * 256]), u16s(&z[nq + nkv as usize * cap as usize * 256..]));
            let mut want = vec![0xAAu8; 2 * nkv as usize * hb];
            let (wk, wv) = want.split_at_mut(nkv as usize * hb);
            for h in 0..nkv as usize {
                let o = (h * cap as usize + pos as usize) * 128;
                kv8_ref::put_row(wk, h, pos as usize, cap as usize, &k16[o..o + 128]);
                kv8_ref::put_row(wv, h, pos as usize, cap as usize, &v16[o..o + 128]);
            }
            assert_eq!(got[nq..], want[..], "the int8 caches, form {form}, pos {pos}");
        }
    }
    let f = "llvq-cuda/kernels/pers_4e_k8.cuh";
    let c = cmd(b'z', 1, 7, 3);
    let want = run(served(), &c);
    for (tag, from, to) in [
        ("k8_append_pos", "pers_k8_quant_row(kr, kc8 + h * hstride + blk, pos % PERS_TT, lane);", "pers_k8_quant_row(kr, kc8 + h * hstride + blk, (pos + 1u) % PERS_TT, lane);"),
        ("k8_append_unroped", "pers_head_norm_rope<FORM>(kr, k_alpha, cosr, sinr, eps, kr, lane);", "(void)k_alpha;"),
        ("k8_append_v_head", "pers_k8_quant_row(v_raw + h * 128u, vc8", "pers_k8_quant_row(v_raw, vc8"),
    ] {
        let b = mutant_v9(tag, f, from, to);
        assert!(run_mutant(&b, &c) != Some(want.clone()), "k8 append mutant {tag} survived");
        eprintln!("k8 append mutant {tag}: killed");
    }
}

/// Arm v9k8's token loop is the served chain on the int8 cache (+ 512):
/// both caches quantized by the fill before the first launch, the tokens,
/// the final residual and both layers' int8 caches, on both row variants,
/// each RoPE form, positions past 256 (the split fold over int8 blocks),
/// consecutive launches and the stamp symbol. The int8 chain is not the
/// binary16 one (its residual moves), so the comparison reads the cache.
#[test]
fn the_4e_k8_arm_is_the_int8_served_chain() {
    let b = persistent_e1("v9k8", "e1_v9k8", None);
    for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2), (1, 5, 64, 2), (1, 3, 512, 2), (1, 2, 128, 0)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x9e40 + form as u64, variant | 512);
        let (want, got) = (run(served(), &cmd), run(&b, &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9k8, form {form}, grid {grid}x{threads}, variant {variant}");
    }
    for (pos0, cap, seed) in [(64u32, 160u32, 0x9e50u64), (270, 288, 0x9e52)] {
        let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2 | 512);
        let (want, got) = (run(served(), &cmd), run(&b, &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9k8, positions from {pos0}");
        // the int8 chain is another chain: its final residual differs from the binary16 one's
        let f16 = run(served(), &g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2));
        assert!(want[12..12 + 2 * 2560] != f16[12..12 + 2 * 2560], "the int8 cache moved nothing, positions from {pos0}");
    }
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 2 | 512);
    let (want, plain) = (run(served(), &cmd), run(&b, &cmd));
    assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "v9k8: two consecutive launches");
    let stamped = run(&b, &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 1 | 512));
    assert!(stamped[..plain.len()] == plain[..], "v9k8: the stamped launch moved an output");
    // the arm's own edits: V read from the K cache, and the attention left on v9's binary16 reader
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 3, 7)], 288, u32::MAX, 0x9e52, 2 | 512);
    let want = run(served(), &cmd);
    for (tag, twist) in [
        ("e1_v9k8_v_from_k", ("reinterpret_cast<const unsigned char*>(Y.vc), A->n_heads,", "reinterpret_cast<const unsigned char*>(Y.kc), A->n_heads,")),
        ("e1_v9k8_append_k_in_v", ("reinterpret_cast<unsigned char*>(Y.vc));", "reinterpret_cast<unsigned char*>(Y.kc));")),
    ] {
        let m = persistent_e1("v9k8", tag, Some(twist));
        let got = run_mutant(&m, &cmd);
        assert!(got.is_none_or(|o| o[..o.len() - 8] != want[..want.len() - 8]), "v9k8 mutant {tag} survived");
        eprintln!("v9k8 mutant {tag}: killed");
    }
}
