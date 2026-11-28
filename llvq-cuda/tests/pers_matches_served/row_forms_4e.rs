//! 4E's row forms v4, v6, v7, v8 and v9, and E1's L2 requests on v9. One file
//! because each form is the previous one plus one lever, gated on the same chain.

use super::*;

// ---- 4E v4: v2's row loop inside v3's structure (PERS_4E_ROWS = 2) ---------

/// The module built with row form 2 runs the served chain bit for bit: the
/// q/k/v phase, the token loop (both RoPE forms read, positions past 256,
/// EOS), and the stamped symbol's tokens and inner stamps.
#[test]
fn the_4e_v4_rows_are_the_served_chain() {
    let r = rot(2560, 41);
    let qkv = |grid: u32, threads: u32, seed: u64| {
        Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0
    };
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        assert!(run(served(), &qkv(grid, threads, seed)) == run(persistent_v4(), &qkv(grid, threads, seed)), "v4 q/k/v, grid {grid}");
    }
    let n = 3u32;
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, u32::MAX, 0x4e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v4(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v4, form {form}, grid {grid}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x4e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v4(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v4, positions 270-271");
    // the stamp symbol of v4: the same outputs as the token symbol, stamps that read
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v4(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 2));
    let stamped = run(persistent_v4(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v4: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let r = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v4's stamps");
    assert!(r.subs.iter().all(|p| p.mean.0 > 0.0), "{r:?}");
}

/// v4's driver: each mutant moves the q/k/v rows or the token loop.
#[test]
fn the_4e_v4_mutants_are_killed() {
    let r = rot(2560, 41);
    let v = Cmd::new(b'V').u32(3).u32(96).u64(1).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0;
    let g = g_cmd_variant(0, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x4e40, 2);
    let (want_v, want_g) = (run(served(), &v), run(served(), &g));
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v4_rscale", "            rs = R.rscale[R.r];\n            start(k);", "            rs = R.rscale[R.r ^ 1u];\n            start(k);"),
        ("v4_tail", "epi(k, pers_epilogue(a, rs, R.tail + (unsigned long long)R.r * tail_w,", "epi(k, pers_epilogue(a, rs, R.tail + (unsigned long long)(R.r + 1u) * tail_w,"),
        ("v4_start", "            rs = R.rscale[R.r];\n            start(k);", "            rs = R.rscale[R.r];\n            (void)start;"),
        ("v4_row_order", "    for (u32 k = 0; k < nrows; ++k) {\n        const PersTRow R = row_of(k);", "    for (u32 k = nrows; k-- > 0;) {\n        const PersTRow R = row_of(k);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v4"), &["PERS_4E_ROWS=2u"]);
        let killed = run_mutant(&b, &v).is_none_or(|o| o != want_v) || run_mutant(&b, &g).is_none_or(|o| o != want_g);
        assert!(killed, "v4 mutant {tag} survived");
    }
}


// ---- 4E v6: ln1, ln2 and the down's input in one round trip (PERS_4E_ROWS = 4) --

/// The persistent side with 4E's row form 4 (4E v6: v4's rows, the
/// redundant prologues loaded as vectors, `small` staged in shared memory).
pub(super) fn persistent_v6() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v6", &["PERS_4E_ROWS=4u"]))
}

/// The -0 column of [`the_mix_turns_minus_zero_into_plus_zero_as_the_padding_does`]
/// at odd factor `k`: the last real term of every g >= 1 underflows to -0.
fn mix_minus_zero_cmd(k: u32) -> Vec<u8> {
    let m = 32u32;
    let mut s = vec![0.0f32; (k * m) as usize];
    let mut small = vec![0.0f32; KMAX * KMAX];
    let last = (k - 1) as usize;
    for j in 0..m as usize {
        s[last * m as usize + j] = 1e-30;
        s[j] = if j % 2 == 0 { 0.0 } else { 0.5 };
    }
    for g in 0..k as usize {
        small[g * KMAX + last] = -1e-20;
        small[g * KMAX] = if g == 0 { 1.0 } else { 0.0 };
    }
    Cmd::new(b'M').u32(m).u32(k).f32(1.0).f32s(&s).f32s(&small).0
}

/// The module built with row form 4 runs the served chain bit for bit:
/// - the rotation at every served width through the vector load and the
///   staged mix (k = 5 and 19 unrolled, the other k on the generic column),
///   at 64 threads (several batches, `small` staged in a loop) and 512 (the
///   card's geometry, one batch);
/// - the -0 column through the unrolled mixes;
/// - the token loop (ln1, ln2, a Tetra and an int4 down input), at four
///   geometries, past 256 positions, and through the stamp symbol.
#[test]
fn the_4e_v6_prologues_are_the_served_chain() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 512] {
            let cmd = rot_cmd(n, threads, 0x6e00 + n as u64);
            assert!(run(served(), &cmd) == run(persistent_v6(), &cmd), "v6 rotation, n = {n}, {threads} threads");
        }
    }
    for k in [3u32, 5, 19] {
        let cmd = mix_minus_zero_cmd(k);
        let (a, b) = (run(served(), &cmd), run(persistent_v6(), &cmd));
        assert!(a == b, "v6 mix, k = {k}");
        assert!(u32s(&a).contains(&0), "k = {k}: some output is +0 (the column the padding fixed)");
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x6e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v6(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v6, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x6e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v6(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v6, positions 270-271");
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v6(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x6e21, 2));
    let stamped = run(persistent_v6(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x6e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v6: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let r = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v6's stamps");
    assert!(r.subs.iter().all(|p| p.mean.0 > 0.0), "{r:?}");
}

/// v6's one-trip code: each mutant moves a rotation, the -0 column or the
/// token loop. A mutant that survived would mean the vector path is not the
/// one the commands run.
#[test]
fn the_4e_v6_mutants_are_killed() {
    let mut cmds = vec![rot_cmd(2560, 64, 0x6e01), rot_cmd(9728, 64, 0x6e02), rot_cmd(9728, 512, 0x6e03)];
    cmds.extend([5u32, 19].map(mix_minus_zero_cmd));
    cmds.push(g_cmd_variant(1, 3, 96, &[(5, 3, 7)], 16, u32::MAX, 0x6e60, 2));
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v6_half_order", "for (u32 e = 0; e < 8u; ++e) s[8u * q + e] = ((bits >> e) & 1u) ? -f[e] : f[e];", "for (u32 e = 0; e < 8u; ++e) s[8u * q + (e ^ 1u)] = ((bits >> e) & 1u) ? -f[e] : f[e];"),
        ("v6_sign_word", "if (signbits != nullptr) sw[u] = signbits[q >> 2];", "if (signbits != nullptr) sw[u] = signbits[q >> 3];"),
        ("v6_ln_sign_shift", "const u32 bits = sw[u] >> ((q & 3u) * 8u);\n#pragma unroll\n                for (u32 e = 0; e < 8u; ++e) {", "const u32 bits = sw[u] >> ((q & 3u) * 4u);\n#pragma unroll\n                for (u32 e = 0; e < 8u; ++e) {"),
        ("v6_small_store", "if (i0 < ns) ssm[i0] = r.a0;", "if (i0 < ns) ssm[i0 ^ 1u] = r.a0;"),
        ("v6_small_rest", "for (u32 i = threadIdx.x + 2u * blockDim.x; i < ns; i += blockDim.x) ssm[i] = small[i];", ""),
        ("v6_mix_order", "for (u32 t = 0; t < K; ++t) acc = __fmaf_rn(ssm[g * LLVQ_ROT_KMAX + t], col[t], acc);", "for (u32 t = K; t-- > 0;) acc = __fmaf_rn(ssm[g * LLVQ_ROT_KMAX + t], col[t], acc);"),
        ("v6_mix_fadd", "if (K < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);", ""),
        ("v6_apply_order", "return f2h(__fmul_rn(__fmul_rn(scale, xf), af));", "return f2h(__fmul_rn(scale, __fmul_rn(xf, af)));"),
        ("v6_alpha_vector", "a[u] = av[q];", "a[u] = av[q ^ 1u];"),
        ("v6_nat_copy", "if (nat) xs[i] = hv;", "if (nat) xs[i] = -hv;"),
        ("v6_rms_loader", "[&](u32 col) { return xs[col]; }", "[&](u32 col) { return __fmul_rn(xs[col], 1.0009765625f); }"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v6"), &["PERS_4E_ROWS=4u"]);
        let killed = cmds.iter().zip(&want).any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v6 mutant {tag} survived");
    }
    // the call sites in the MLP segment: the token loop runs them (ln2, a Tetra
    // and an int4 down input), so each mutant moves its tokens or state
    let (g, want_g) = (cmds.last().unwrap(), want.last().unwrap());
    for (tag, from, to) in [
        ("v6_seg_ln2_signs", "pers_ln_1rt<NORMV>(L.x, A->alpha, A->signs_h,", "pers_ln_1rt<NORMV>(L.x, A->alpha, A->signs_i,"),
        ("v6_seg_down_k", "pers_load_1rt(A->act, A->signs_i, v, I, A->small_i, A->k_i, ssm);", "pers_load_1rt(A->act, A->signs_i, v, I, A->small_i, A->k_i - 1u, ssm);"),
        ("v6_seg_widen_signs", "pers_load_1rt(A->act, nullptr, v, I, nullptr, 0u, nullptr);", "pers_load_1rt(A->act, A->signs_i, v, I, nullptr, 0u, nullptr);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, "llvq-cuda/kernels/pers_segment.cu", from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v6"), &["PERS_4E_ROWS=4u"]);
        assert!(run_mutant(&b, g).is_none_or(|o| &o != want_g), "v6 segment mutant {tag} survived");
    }
}

// ---- 4E v7: the launch's constants read during the barriers (PERS_4E_ROWS = 5) --

/// The persistent side with 4E's row form 5 (4E v7: v6, with the argument
/// structs, the CTA's slices, the rotations' sign words and `small` copied
/// into shared memory during the barrier before they are read, the o phase's
/// input widened from vectors, and `small` read as float4 in the mix).
fn persistent_v7() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v7", &["PERS_4E_ROWS=5u"]))
}

/// The module built with row form 5 runs the served chain bit for bit:
/// - the rotation at every served width with its signs and `small` read
///   from the v7 area, and the float4 mix (k = 5 and 19), at 64 and 512
///   threads;
/// - the -0 column through the float4 mixes;
/// - the q/k/v phase and the o phase (its vector widening, x read at the
///   row's start) at several grids;
/// - the token loop (ln1 and ln2 from the staged area, a Tetra and an int4
///   down input, the layer struct staged during the barriers) at four
///   geometries, past 256 positions, over consecutive launches, and through
///   the stamp symbol.
#[test]
fn the_4e_v7_staged_constants_are_the_served_chain() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 512] {
            let cmd = rot_cmd(n, threads, 0x7e00 + n as u64);
            assert!(run(served(), &cmd) == run(persistent_v7(), &cmd), "v7 rotation, n = {n}, {threads} threads");
        }
    }
    for k in [3u32, 5, 19] {
        let cmd = mix_minus_zero_cmd(k);
        let (a, b) = (run(served(), &cmd), run(persistent_v7(), &cmd));
        assert!(a == b, "v7 mix, k = {k}");
        assert!(u32s(&a).contains(&0), "k = {k}: some output is +0 (the column the padding fixed)");
    }
    for (grid, threads) in [(3u32, 96u32), (4, 64), (7, 128)] {
        let w = Cmd::new(b'W').u32(grid).u32(threads).u64(7).0;
        assert!(run(served(), &w) == run(persistent_v7(), &w), "v7 o phase, grid {grid}x{threads}");
    }
    let r = rot(2560, 41);
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        let v = Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0;
        assert!(run(served(), &v) == run(persistent_v7(), &v), "v7 q/k/v, grid {grid}");
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x7e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v7(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x7e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v7(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, positions 270-271");
    // consecutive launches through the same state, one rewinding: each
    // launch stages layer 0 again in its embedding barrier
    let cmd = g_cmd_variant(2, 4, 96, &[(5, 3, 7), (8, 2, 9), (6, 2, 3)], 16, u32::MAX, 0x7e60, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v7(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, three consecutive launches");
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v7(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x7e21, 2));
    let stamped = run(persistent_v7(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x7e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v7: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let rep = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v7's stamps");
    assert!(rep.subs.iter().all(|p| p.mean.0 > 0.0), "{rep:?}");
}

/// v7's staging: each mutant moves a rotation, the o phase or the token loop.
/// A survivor would mean the staged copy is not the one the commands read.
#[test]
fn the_4e_v7_mutants_are_killed() {
    let mut cmds = vec![rot_cmd(2560, 64, 0x7e01), rot_cmd(9728, 512, 0x7e03)];
    cmds.extend([5u32, 19].map(mix_minus_zero_cmd));
    cmds.push(Cmd::new(b'W').u32(3).u32(96).u64(7).0);
    cmds.push(g_cmd_variant(1, 3, 96, &[(5, 3, 7)], 16, u32::MAX, 0x7e70, 2));
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let cuh = "llvq-cuda/kernels/llvq_pers.cuh";
    let e4 = "llvq-cuda/kernels/pers_4e.cu";
    let seg = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, file, from, to) in [
        // the float4 mix: a component, and the padding's + 0 that turns -0 into +0
        ("v7_mix_lane", cuh, "acc = __fmaf_rn(w.x, col[4u * t4], acc);", "acc = __fmaf_rn(w.y, col[4u * t4], acc);"),
        ("v7_mix_fadd", cuh, "acc = K < LLVQ_ROT_KMAX ? __fadd_rn(acc, 0.0f) : acc;", "acc = K < LLVQ_ROT_KMAX ? __fadd_rn(acc, -0.0f) : acc;"),
        // the staged rows of `small` and sign words
        ("v7_stage_small", cuh, "                  k * LLVQ_ROT_KMAX);\n}", "                  (k - 1u) * LLVQ_ROT_KMAX);\n}"),
        ("v7_stage_signs", cuh, "pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SIGNS), signbits, width >> 5);",
         "pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SIGNS), signbits, (width >> 5) - 1u);"),
        // the gap wait's closing barrier: without it a thread reads the copies before they are made
        ("v7_gap_close", cuh, "    __syncthreads();\n    return hdr[PERS_HDR_OK] != 0u;\n}\n\n// n words", "    return hdr[PERS_HDR_OK] != 0u;\n}\n\n// n words"),
        // the o phase: the vector widening's width, and x at the row's start
        ("v7_o_width", cuh, "pers_load_1rt(ctx, nullptr, cs, d_ctx, nullptr, 0u, nullptr);", "pers_load_1rt(ctx, nullptr, cs, d_ctx - 8u, nullptr, 0u, nullptr);"),
        ("v7_o_x", cuh, "const unsigned short xr = lane == 0u ? __ldcg(x + r) : (unsigned short)0u;", "const unsigned short xr = lane == 0u ? __ldcg(x + (r ^ 1u)) : (unsigned short)0u;"),
        // 4E's staging: the args copy, the area's offset, the next layer, its slices, ln1 and ln2 inputs
        ("v7_args_words", e4, "(u32)(sizeof(Pers4eArgs) / 4u));", "(u32)(sizeof(Pers4eArgs) / 8u));"),
        ("v7_area_offset", e4, "if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = v7_off;", "if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = v7_off + 4u;"),
        ("v7_next_layer", e4, "if (l + 1u < A->n_layers) stage_layer(l + 1u);", "if (l + 1u < A->n_layers) stage_layer(l);"),
        ("v7_slice_hi", e4, "sl[2u * blockIdx.x + (threadIdx.x & 1u)];", "sl[2u * blockIdx.x];"),
        ("v7_ln1_stage", e4, "pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);", "pers_v7_stage_rot(area, g->mlp.signs_h, Ag->hidden, g->small_a, g->k_a);"),
        ("v7_ln2_stage", e4, "pers_v7_stage_rot(area, Y.mlp.signs_h, H, Y.mlp.small_h, Y.mlp.k_h);", "pers_v7_stage_rot(area, Y.signs_a, H, Y.mlp.small_h, Y.mlp.k_h);"),
        // the segment: the down's staging, and the staged signs it reads
        ("v7_down_stage", seg, "if (TETRA_DOWN) pers_v7_stage_rot(area, A->signs_i, I, A->small_i, A->k_i);", "if (TETRA_DOWN) pers_v7_stage_rot(area, A->signs_i, I, A->small_h, A->k_i);"),
        ("v7_down_signs", seg, "pers_load_1rt(A->act, v7g, v, I, nullptr, A->k_i, ssm);", "pers_load_1rt(A->act, A->signs_h, v, I, nullptr, A->k_i, ssm);"),
        ("v7_seg_slice", seg, "#define PERS_SEG_DN(i) (IN4E ? v7s[2u + (i)] :", "#define PERS_SEG_DN(i) (IN4E ? v7s[(i)] :"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, file, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v7"), &["PERS_4E_ROWS=5u"]);
        let killed = cmds.iter().zip(&want).any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v7 mutant {tag} survived");
    }
}

// ---- 4E v8: v7 with the attention steps in a cp.async ring (PERS_4E_ROWS = 6) --

/// The persistent side with 4E's row form 6 (4E v8: v7, and the attention
/// phase's steps staged into a ring of three stages, K3a of plan-300). The
/// host's cp.async copies at once, so these tests hold the ring's indexing
/// and barriers; the card's load-time identity holds its asynchrony.
pub(super) fn persistent_v8() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v8", &["PERS_4E_ROWS=6u"]))
}

/// The attention phase of form 6 is attn_ref's on every row length the
/// form-1 test reads (one step, two, four and eight a job; one split and
/// several), at three geometries; and the token loop is the served chain.
#[test]
fn the_4e_v8_attention_ring_is_the_served_chain() {
    let lens = [(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 4, 192, 150), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300), (3, 5, 2304, 2049)];
    for &(n_kv, groups, cap, l) in &lens {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96), (1, 512)] {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, grid, threads, 0xa77 + l as u64);
            let got = run(persistent_v8(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "v8: n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (0, 0), "v8: every ticket back at zero");
            assert_eq!(u32s(&got[n + 8..]), state, "v8: the f32 state before the division, l {l}");
        }
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x8e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v8(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v8, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    // rows of 66 to 130 positions (four steps a job) and past 256 (two splits)
    for (pos0, cap, seed) in [(64u32, 160u32, 0x8e50u64), (125, 160, 0x8e51), (270, 288, 0x8e52)] {
        let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2);
        let (want, got) = (run(served(), &cmd), run(persistent_v8(), &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v8, positions from {pos0}");
    }
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v8(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x8e21, 2));
    let stamped = run(persistent_v8(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x8e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v8: the stamped launch moved an output");
}

/// The ring's indexing and its reader barrier: each mutant moves a row of
/// four or eight steps. (Waits and commits are no-ops on the host; the card's
/// identity gate holds them.)
#[test]
fn the_4e_v8_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(2u32, 4u32, 192u32, 150u32), (1, 4, 1408, 1300)]
        .iter()
        .map(|&(n_kv, groups, cap, l)| {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, 3, 64, 0xa78);
            let mut w: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.extend([0u32, 0].iter().flat_map(|x| x.to_le_bytes()));
            w.extend(state.iter().flat_map(|x| x.to_le_bytes()));
            (cmd, w)
        })
        .collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v8_issue_step", "pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);", "pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING - 1u, nsteps, tpu, l);"),
        ("v8_slot", "uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;", "uint4* const sbuf = stage + (st % 2u) * PERS_STAGE_U4;"),
        ("v8_issue_pos", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + pl % PERS_TT;", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + (pl + 1u) % PERS_TT;"),
        ("v8_first_issues", "for (u32 st = 0; st < PERS_ATTN_RING; ++st)\n            pers_attn_issue(", "for (u32 st = 0; st + 1u < PERS_ATTN_RING; ++st)\n            pers_attn_issue("),
        ("v8_reader_barrier", "            __syncthreads();  // the slot's readers are done: it takes step st + RING\n", ""),
        ("v8_kv_half", "const unsigned short* src = (idx < PERS_STAGE * 16u ? kh : vh);", "const unsigned short* src = (idx <= PERS_STAGE * 16u ? kh : vh);"),
        // the unit states and q after the whole ring, not after one stage
        ("v8_states_after_ring", "smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);", "smem + 16u * PERS_STAGE_U4);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v8"), &["PERS_4E_ROWS=6u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v8 mutant {tag} survived");
    }
}

// ---- 4E v9: v8 with the int4 rows on tiled words (PERS_4E_ROWS = 7) -------

/// The persistent side with 4E's row form 7 (4E v9: v8, and every int4 row of
/// the token loop, v, o and the int4 down, read from words relaid by
/// `pers_q4_tile_index`, K11 of plan-300). The harness relays the 4E side's
/// records as `persist4e_cuda.rs` does on the card; the served side keeps
/// the served words.
pub(super) fn persistent_v9() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v9", &["PERS_4E_ROWS=7u"]))
}

/// The token loop of form 7 is the served chain: both row variants, each
/// RoPE form, positions past 256, consecutive launches, and the stamp symbol.
/// The v rows (d_in 2,560: 2 tiles and a 64-word tail), the o rows (4,096: 4
/// tiles, no tail) and layer 1's int4 down (9,728: 9 tiles and a tail) all
/// run the tiled reader.
#[test]
fn the_4e_v9_int4_tiles_are_the_served_chain() {
    for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2), (1, 5, 64, 2), (1, 3, 512, 2), (1, 2, 128, 0)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x9e40 + form as u64, variant);
        let want = run(served(), &cmd);
        let got = run(persistent_v9(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9, form {form}, grid {grid}x{threads}, variant {variant}: tokens, residual or caches");
    }
    for (pos0, cap, seed) in [(64u32, 160u32, 0x9e50u64), (270, 288, 0x9e52)] {
        let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2);
        let (want, got) = (run(served(), &cmd), run(persistent_v9(), &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9, positions from {pos0}");
    }
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 2);
    let (want, plain) = (run(served(), &cmd), run(persistent_v9(), &cmd));
    assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "v9: two consecutive launches");
    let stamped = run(persistent_v9(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v9: the stamped launch moved an output");
}

/// The tiled reader, its layout, and each int4 site of the token loop: each
/// mutant moves the tokens, the residual or a cache. A site mutant puts that
/// site back on the served reader, which then reads tiled words: it proves
/// the site runs the tiled reader under form 7.
#[test]
fn the_4e_v9_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(1u32, 3u32, 128u32, 2u32, 0x9e60u64), (1, 4, 96, 0, 0x9e61)]
        .iter()
        .map(|&(form, grid, threads, variant, seed)| {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, seed, variant);
            let want = run(persistent_v9(), &cmd);
            (cmd, want)
        })
        .collect();
    let cuh = "llvq-cuda/kernels/llvq_pers.cuh";
    let seg = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, f, from, to) in [
        ("v9_index", cuh, "return (t << 7) + 4u * (k & 31u) + (k >> 5);", "return (t << 7) + 4u * (k & 31u) + ((k >> 5) ^ 1u);"),
        ("v9_first_tile", cuh, "if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * lane);", "if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * (lane ^ 1u));"),
        ("v9_next_tile", cuh, "if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + ((t + 1u) << 7) + 4u * lane);", "if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + (t << 7) + 4u * lane);"),
        ("v9_column", cuh, "const u32 c = ((t << 7) + 32u * s + lane) << 3;", "const u32 c = ((t << 7) + 32u * (3u - s) + lane) << 3;"),
        ("v9_tail", cuh, "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {\n        const u32 p = wrow[wi];", "for (u32 wi = (nt << 7) + lane + 32u; wi < nwords; wi += 32u) {\n        const u32 p = wrow[wi];"),
        ("v9_site_v", cuh, "const float acc = pers_q4_row_4e<true>(P.vw,", "const float acc = pers_q4_row<true>(P.vw,"),
        ("v9_site_o", cuh, ": (unsigned short)0u;\n        const float acc = pers_q4_row_4e<PF>(ow,", ": (unsigned short)0u;\n        const float acc = pers_q4_row<PF>(ow,"),
        ("v9_site_down", seg, "const float acc = pers_q4_row_4e<true>(A->q4w,", "const float acc = pers_q4_row<true>(A->q4w,"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v9 mutant {tag} survived");
    }
}

// ---- E1: where 4E v9 asks L2 for later rows (pers.rs::p4e_e1_edits) -------

/// The host twin of an E1 arm: the sources with the arm's edits, the ones
/// `pers::p4e_e1_unit_text` applies to the card's module, under row form 7.
/// `twist` rewrites one replacement text (a mutant of the arm), or nothing.
pub(super) fn persistent_e1(arm: &str, tag: &str, twist: Option<(&str, &str)>) -> PathBuf {
    persistent_e1_twists(arm, tag, twist.as_slice())
}

/// [`persistent_e1`] with several twists, each of exactly one site in the
/// arm's replacement texts, applied in order (a mutant of two sites).
pub(super) fn persistent_e1_twists(arm: &str, tag: &str, twists: &[(&str, &str)]) -> PathBuf {
    let edits = pers::p4e_e1_edits(arm).unwrap();
    let mut files: Vec<(String, String, String)> =
        edits.into_iter().map(|(f, a, b)| (format!("llvq-cuda/kernels/{f}"), a, b)).collect();
    for &(from, to) in twists {
        let hits: usize = files.iter().map(|e| e.2.matches(from).count()).sum();
        assert_eq!(hits, 1, "E1 mutant {tag}: {hits} sites for {from:?} in the arm's edits, expected exactly one");
        for e in files.iter_mut() {
            e.2 = e.2.replacen(from, to, 1);
        }
    }
    let refs: Vec<(&str, &str, &str)> = files.iter().map(|(f, a, b)| (f.as_str(), a.as_str(), b.as_str())).collect();
    let root = mutant_root(tag, &refs);
    build_side_with(&root, 1, "off", tag, &["PERS_4E_ROWS=7u"])
}

/// Each E1 arm's token loop is the served chain, as v9's is
/// (`the_4e_v9_int4_tiles_are_the_served_chain`'s commands): both row
/// variants, each RoPE form, positions past 256, consecutive launches, and
/// the stamp symbol. On the host a prefetch is nothing, so this proves what
/// the arms add beside their prefetches (mv's header slots, its staging in
/// the barrier gaps, the layer-0 branch, the logits edits) moves no bit.
#[test]
fn the_4e_e1_arms_are_the_served_chain() {
    // every arm that owes v9's bits; v9fl owes a bound on rows (the_4e_rc_flat_arm_*)
    // and the one-chunk arms (v9ep3) run on 142 CTAs: the_4e_rc_one_chunk_arms_on_the_cards_grid
    for arm in pers::P4E_E1_ARMS.into_iter().filter(|a| pers::p4e_e1_owes_bits(a) && !pers::p4e_rc_body(a).is_some_and(pers::rc_body_one_chunk)) {
        let b = persistent_e1(arm, &format!("e1_{arm}"), None);
        for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2), (1, 5, 64, 2), (1, 3, 512, 2), (1, 2, 128, 0)] {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x9e40 + form as u64, variant);
            let (want, got) = (run(served(), &cmd), run(&b, &cmd));
            assert!(got[..got.len() - 8] == want[..want.len() - 8], "{arm}, form {form}, grid {grid}x{threads}, variant {variant}");
        }
        for (pos0, cap, seed) in [(64u32, 160u32, 0x9e50u64), (270, 288, 0x9e52)] {
            let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2);
            let (want, got) = (run(served(), &cmd), run(&b, &cmd));
            assert!(got[..got.len() - 8] == want[..want.len() - 8], "{arm}, positions from {pos0}");
        }
        let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
        let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 2);
        let (want, plain) = (run(served(), &cmd), run(&b, &cmd));
        assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "{arm}: two consecutive launches");
        let stamped = run(&b, &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 1));
        assert!(stamped[..plain.len()] == plain[..], "{arm}: the stamped launch moved an output");
    }
}

/// mv's only stores outside a prefetch are its header slots: a mutant that
/// moves them onto the v7 area's offset (header u32 8) is killed, so the
/// test above sees the slots, and slots 10..25 are free.
#[test]
fn the_4e_e1_mv_slot_mutant_is_killed() {
    let cmd = g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e60, 2);
    let want = run(served(), &cmd);
    let b = persistent_e1("v9mv", "e1_mv_slot_on_v7", Some(("#define PERS_HDR_E1 10u", "#define PERS_HDR_E1 6u")));
    let got = run_mutant(&b, &cmd);
    assert!(got.is_none_or(|o| o[..o.len() - 8] != want[..want.len() - 8]), "E1 mv mutant slot_on_v7 survived");
}

/// The two ft mutants of tree_verify_host.rs on arm v9ft, one site each in
/// the ft text (`persistent_e1` counts it): the full-tile body taken on a
/// tile of 7, and position 3's chain reading position 4's K. Each moves the
/// served chain on a run whose tiles hold 6, 7 and 8 positions (from 5) or
/// 7, 8 and 1 after 33 full ones (from 270, across the split at 256).
#[test]
fn the_4e_v9ft_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e70, 2), g_cmd_variant(1, 3, 128, &[(270, 3, 7)], 288, u32::MAX, 0x9e71, 2)]
        .into_iter()
        .map(|c| {
            let want = run(served(), &c);
            (c, want)
        })
        .collect();
    for (tag, from, to) in [
        ("v9ft_taken_at_7", "    if (nv == PERS_TT) {", "    if (nv + 1u >= PERS_TT) {"),
        ("v9ft_neighbour_k", "for (u32 j = 0; j < PERS_TT; ++j) kw[j] = ks[j * 16u + l16];", "for (u32 j = 0; j < PERS_TT; ++j) kw[j] = ks[(j == 3u ? 4u : j) * 16u + l16];"),
    ] {
        let b = persistent_e1("v9ft", tag, Some((from, to)));
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| o[..o.len().saturating_sub(8)] != w[..w.len() - 8]));
        assert!(killed, "v9ft mutant {tag} survived");
        eprintln!("v9ft mutant {tag}: killed");
    }
}
