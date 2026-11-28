//! 4E's phases (head, layer, attention, norm-rope-append) and its token loop
//! against the served chain. One file because the `g_cmd` family builds every
//! token-loop command these gates and the later row forms send.

use super::*;

/// 4E's head phase: the persistent head rows are the served tv_emb_q4_h
/// logits bit for bit, and the token every CTA reads back after its W4
/// barrier is the served argmax (candle's tie order), planted ties included,
/// on grids that split the vocabulary differently.
#[test]
fn the_4e_head_phase_is_the_served_argmax() {
    for (vocab, plant) in [(3072u32, 0u32), (3072, 1), (4104, 1)] {
        for (grid, threads) in [(3u32, 64u32), (5, 96)] {
            let cmd = Cmd::new(b'H').u32(vocab).u32(256).u32(grid).u32(threads).u32(plant).u64(0x4e + vocab as u64).0;
            let (want, got) = (run(served(), &cmd), run(persistent(), &cmd));
            assert!(u32s(&want[..4])[0] < vocab);
            assert!(want == got, "vocab {vocab}, plant {plant}, grid {grid}x{threads}: token or logits differ");
        }
    }
    let cmd = Cmd::new(b'H').u32(3072).u32(256).u32(3).u32(64).u32(1).u64(0x4e + 3072).0;
    let want = run(served(), &cmd);
    let b = mutant("head_no_w4", "llvq-cuda/kernels/llvq_pers.cuh", "    if (!pers_grid_wait(bar, bar_target, abort_flag)) return 0xffffffffu;\n    if (pers_ld_acquire(nan_flag)", "    (void)bar; (void)bar_target; (void)abort_flag;\n    if (pers_ld_acquire(nan_flag)");
    assert!(run_mutant(&b, &cmd) != Some(want.clone()), "the W4 barrier is not load-bearing");
    let b = mutant("head_key_index", "llvq-cuda/kernels/llvq_pers.cuh", "return ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);", "return ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | br;");
    assert!(run_mutant(&b, &cmd) != Some(want), "the key's bit reversal on the way back survived");
}

/// 4E's embedding, q/k/v and o phases equal the served chain bit for bit:
/// emb_q4_gather; ln1, rot_apply, tv_tetra48_h (as NVVM compiled it) and
/// tv_q4_hh on the unrotated ln1 output; tv_q4_hh then the residual add.
#[test]
fn the_4e_layer_phases_are_the_served_chain() {
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tok, grid) in [(0u32, 3u32), (517, 4), (1023, 5)] {
        let cmd = Cmd::new(b'E').u32(1024).u32(2560).u32(tok).u32(grid).u64(0xe3b).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "embedding, token {tok}, grid {grid}");
    }
    let r = rot(2560, 41);
    let qkv = |grid: u32, threads: u32, seed: u64| {
        Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0
    };
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        assert!(run(served(), &qkv(grid, threads, seed)) == run(persistent(), &qkv(grid, threads, seed)), "q/k/v, grid {grid}");
    }
    for (grid, threads) in [(3u32, 96u32), (5, 64)] {
        let cmd = Cmd::new(b'W').u32(grid).u32(threads).u64(7).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "o, grid {grid}");
    }
    // mutants
    let e = Cmd::new(b'E').u32(1024).u32(2560).u32(517).u32(3).u64(0xe3b).0;
    let b = mutant("embed_group", f, "const u32 g = s0 + (c >> 6);\n        dst[c]", "const u32 g = s0 + (c >> 7);\n        dst[c]");
    assert!(run_mutant(&b, &e) != Some(run(served(), &e)), "the embedding's g64 group survived as g128");
    let v = qkv(3, 96, 1);
    let want = run(served(), &v);
    // the 512-thread path (the row stream) is the one these commands run
    for (tag, from, to) in [
        ("v_reads_rotated", "pers_q4_row_4e<true>(P.vw, P.vsc, P.vbi, xn, rv,", "pers_q4_row_4e<true>(P.vw, P.vsc, P.vbi, ((void)xn, xr), rv,"),
        ("k_offset", "PersTRow{P.kw + (r - P.q_dout) * P.k_stride,", "PersTRow{P.kw + (r - P.q_dout + 1u) * P.k_stride,"),
        ("stream_tail_half", "                    if (tail_w > 8u) t1 = tp[1];\n", ""),
        ("stream_lane_order", "                const u32 j = lane + 32u * i;\n                if (j < nblocks) acc = pers_block(", "                const u32 j = lane + 32u * (nit - 1u - i);\n                if (j < nblocks) acc = pers_block("),
        ("stream_slot", "            if (j2 < nblocks) f1r_load(w2, j2, lo[d], hi[d]);\n            if (++i2 == nit) {", "            if (j2 < nblocks) f1r_load(w2, j2, lo[(d + 1u) % D], hi[(d + 1u) % D]);\n            if (++i2 == nit) {"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &v) != Some(want.clone()), "q/k/v mutant {tag} survived");
    }
    let w = Cmd::new(b'W').u32(3).u32(96).u64(7).0;
    let b = mutant("o_residual", f, "out[r] = exact_hadd(__ldcg(x + r), f2h(acc));", "out[r] = f2h(__fadd_rn(h2f(__ldcg(x + r)), acc));");
    assert!(run_mutant(&b, &w) != Some(run(served(), &w)), "the o residual's binary16 rounding of o survived");
}

pub(super) fn attn_cmd(n_kv: u32, groups: u32, cap: u32, l: u32, grid: u32, threads: u32, seed: u64) -> (Vec<u8>, Vec<u16>, Vec<u32>) {
    use llvq_cuda::attn_ref;
    let nh = (n_kv * groups) as usize;
    let q = attn_ref::fill(nh * 128, seed, 2.0);
    let mut k = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 1, 1.0);
    let v = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 2, 1.0);
    // a planted run of equal keys, so ties of the tile maximum are exercised
    for p in 0..3usize {
        let (a, b) = (p * 128, (p + 9) * 128);
        if b + 128 <= k.len() / n_kv as usize {
            let row: Vec<u16> = k[a..a + 128].to_vec();
            k[b..b + 128].copy_from_slice(&row);
        }
    }
    let c = attn_ref::score_scale(128);
    let kv = attn_ref::Kv { k: &k, v: &v, cap: cap as usize };
    let want = attn_ref::attend(&q, &kv, n_kv as usize, groups as usize, l as usize - 1, 1, c, attn_ref::UNIT_SERVED);
    // the f32 state of each head before the division, built the reference's
    // way: units into splits, splits into the row
    let mut state = Vec::with_capacity(nh * 130);
    for hq in 0..nh {
        let mut qf = [0f32; 128];
        for (i, x) in qf.iter_mut().enumerate() {
            *x = attn_ref::h2f(q[hq * 128 + i]);
        }
        let mut total = attn_ref::St::empty();
        for s in 0..(l as usize).div_ceil(attn_ref::SPLIT) {
            attn_ref::st_merge(&mut total, &attn_ref::split_state(&qf, &kv, hq / groups as usize, s, l as usize, c, attn_ref::UNIT_SERVED));
        }
        state.extend(total.a.iter().map(|x| x.to_bits()));
        state.push(total.m.to_bits());
        state.push(total.d.to_bits());
    }
    let mut cmd = Cmd::new(b'X').u32(n_kv).u32(groups).u32(cap).u32(l).u32(grid).u32(threads).f32(c).0;
    for x in q.iter().chain(&k).chain(&v) {
        cmd.extend(x.to_le_bytes());
    }
    (cmd, want, state)
}

/// 4E's attention phase on the persistent grid equals attn_ref.rs (and so
/// attn_dec.cu) bit for bit: lengths inside one tile, at unit and split
/// boundaries, across several splits; groups of 4 (4B, 8B) and 5 (14B);
/// grids of 1 to 5 CTAs, of 2 to 16 warps (the 16-warp CTA holds every task
/// at once, smaller ones walk them); the split fold through the tickets,
/// which it leaves at zero.
#[test]
fn the_4e_attention_phase_is_attn_ref() {
    let lens = [(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 4, 192, 150), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300), (3, 5, 2304, 2049)];
    for &(n_kv, groups, cap, l) in &lens {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96), (1, 512)] {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, grid, threads, 0xa77 + l as u64);
            let got = run(persistent(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (0, 0), "every ticket back at zero");
            assert_eq!(u32s(&got[n + 8..]), state, "the f32 state before the division, l {l}");
            let port = run(served(), &cmd);
            assert_eq!(u16s(&port[..n]), want, "the C++ port of attn_ref (the 4E test's reference), l {l}");
        }
    }
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let wanted = |n_kv: u32, groups: u32, cap: u32, l: u32| {
        let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, 3, 64, 0xa77);
        let mut w: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.extend([0u32, 0].iter().flat_map(|x| x.to_le_bytes()));
        w.extend(state.iter().flat_map(|x| x.to_le_bytes()));
        (cmd, w)
    };
    let (cmd, want) = wanted(1, 4, 1408, 1300);
    let cases = [
        ("attn_unit_order", "for (u32 uu = 0; uu < PERS_UPS; ++uu) {\n                const float* row", "for (u32 uu = PERS_UPS; uu-- > 0u;) {\n                const float* row"),
        ("attn_carry_fused", "const float dn = __fadd_rn(__fmul_rn(d, carry), ts);", "const float dn = __fmaf_rn(d, carry, ts);"),
        ("attn_tile", "#define PERS_TT 8u", "#define PERS_TT 16u"),
        ("attn_no_fold", "atomicAdd(ticket + h, 1u) == ns - 1u ? 1u : 0u;", "atomicAdd(ticket + h, 1u) == ns ? 1u : 0u;"),
        ("attn_no_reset", "if (threadIdx.x == 0u) ticket[h] = 0u;", "(void)0;"),
        ("attn_stage_pos", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + (pl % PERS_TT);", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + ((pl + 1u) % PERS_TT);"),
        ("attn_step_tile_order", "const u32 sl = (u * tpu + t) * PERS_TT;", "const u32 sl = (u * tpu + (tpu - 1u - t)) * PERS_TT;"),
        ("attn_scale", "sc[j] = __fmul_rn(t, c);", "sc[j] = __fmul_rn(t, __fmul_rn(c, 0.5f));"),
        ("attn_no_syncwarp", "    __syncwarp();  // memory order: every lane's reads of the row precede lane 0's rewrite of m and d\n", ""),
        ("attn_split_chunk", "pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[j]);", "pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[(j + 1u) % PERS_ATTN_CH]);"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "attention mutant {tag} survived");
    }
    // groups of 5: the half-warp past the last head must keep nothing
    let (cmd, want) = wanted(2, 5, 512, 257);
    let b = mutant("attn_spare_half_writes", f, "    if (own) {\n#pragma unroll\n        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];", "    if (true) {\n#pragma unroll\n        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];");
    assert!(run_mutant(&b, &cmd) != Some(want), "attention mutant attn_spare_half_writes survived");
}

/// The fused binary16 FMA of RoPE's candidate forms rounds once from the exact
/// value: against an exact integer reference on 20,000 random finite triples
/// and on planted ties decided by the third term's sign or by evenness.
#[test]
fn the_fused_f16_fma_rounds_once() {
    let cmd = Cmd::new(b'F').u32(20_000).u64(0xf16).0;
    let (want, got) = (run(served(), &cmd), run(persistent(), &cmd));
    assert!(want == got, "the f64 + TwoSum emulation differs from the exact reference");
    let tail: Vec<u16> = u16s(&want)[20_000..20_005].to_vec();
    assert_eq!(tail, vec![0x3e02, 0x3e02, 0x3e01, 0x3e02, 0x3e01], "the planted ties: even, up, down, up, down");
    // An EQUIVALENT mutant, checked as such: for binary16 inputs the f64 sum is
    // exact (an in-range product and the third term span at most 41 bits), so
    // TwoSum's error is always zero and the tie rule never reads it.
    let b = mutant("tie_ignores_err", "llvq-cuda/kernels/llvq_pers.cuh", "(rem == half && (beyond || (err == 0.0 && (q & 1ull))))", "(rem == half && ((void)beyond, (q & 1ull)))");
    assert!(run_mutant(&b, &cmd) == Some(want.clone()), "dropping the TwoSum error changed a result: the exactness argument is wrong");
    let b = mutant("f16_round_down", "llvq-cuda/kernels/llvq_pers.cuh", "if (rem > half || (rem == half && (beyond", "if (rem > half + half || (rem == half && (beyond");
    assert!(run_mutant(&b, &cmd) != Some(want), "a wrong rounding of the f64 value survived");
}

/// 4E's q/k-norm, RoPE and KV append, the three candidate forms, against the
/// served replay (candle's 32-thread norm tree, rope's pairing at i and i +
/// 64, the exact reference for the fused forms): q out, the K and V caches
/// written at `pos` only.
#[test]
fn the_4e_norm_rope_append_phase() {
    let z = |form: u32, pos: u32, grid: u32| Cmd::new(b'Z').u32(form).u32(8).u32(2).u32(16).u32(pos).u32(grid).u32(64).u64(0x20 + pos as u64).0;
    for form in 0..3u32 {
        for (pos, grid) in [(0u32, 2u32), (7, 3), (15, 4)] {
            assert!(run(served(), &z(form, pos, grid)) == run(persistent(), &z(form, pos, grid)), "form {form}, pos {pos}");
        }
    }
    let differ = run(served(), &z(0, 7, 3)) != run(served(), &z(1, 7, 3)) || run(served(), &z(0, 7, 3)) != run(served(), &z(2, 7, 3));
    assert!(differ, "the three forms never differ: the load-time choice would read nothing");
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cmd = z(1, 7, 3);
    let want = run(served(), &cmd);
    for (tag, from, to) in [
        ("rope_pair", "const unsigned short b = exact_rms_apply(scale, hx[j + 2u], alpha[i + 64u]);", "const unsigned short b = exact_rms_apply(scale, hx[j ^ 1u], alpha[i + 64u]);"),
        ("rope_form_swap", "        *o1 = pers_f16_fma_exact(a, c, (unsigned short)(bs ^ 0x8000u), false);", "        *o1 = pers_f16_fma_exact(b, s, ac, true);"),
        ("kv_pos", "kc + ((unsigned long long)h * cap + pos) * 128u, lane);", "kc + ((unsigned long long)h * cap + pos + 1u) * 128u, lane);"),
        ("qk_norm_tree", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "norm/rope mutant {tag} survived");
    }
}

#[allow(clippy::too_many_arguments)]
fn g_cmd(form: u32, grid: u32, threads: u32, n: u32, pos0: u32, cap: u32, tok0: u32, eos: u32, seed: u64) -> Vec<u8> {
    g_cmd_runs(form, grid, threads, &[(pos0, n, tok0)], cap, eos, seed)
}

/// Several launches through the same state, each `(pos0, n, tok0)`, the way
/// `Runtime4e::decode` is called launch after launch.
fn g_cmd_runs(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, eos: u32, seed: u64) -> Vec<u8> {
    g_cmd_variant(form, grid, threads, runs, cap, eos, seed, 0)
}

/// [`g_cmd_runs`] on one variant of the persistent side: 0 the 768-thread
/// token symbols (rows without prefetch), 1 the 512-thread stamp symbols,
/// 2 the 512-thread token symbols (rows with prefetch), 3 the 768-thread
/// stamp symbols. A stamp variant's output carries the stamps after the tail.
#[allow(clippy::too_many_arguments)]
pub(super) fn g_cmd_variant(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, eos: u32, seed: u64, variant: u32) -> Vec<u8> {
    let (pos0, n, tok0) = runs[0];
    let mut c = Cmd::new(b'G').u32(form).u32(grid).u32(threads).u32(n).u32(pos0).u32(cap).u32(tok0).u32(eos).u32(1024).u64(seed);
    for (w, s) in [(2560usize, 51u64), (2560, 52), (9728, 53)] {
        let r = rot(w, s);
        c = c.u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small);
    }
    c = c.u32(runs.len() as u32 - 1);
    for &(p, k, t) in &runs[1..] {
        c = c.u32(p).u32(k).u32(t);
    }
    c = c.u32(variant);
    // + 16: the 8B's Tetra o (its rotation at the context's width) and an
    // untied head table
    if variant & 16 != 0 {
        let r = rot(4096, 54);
        c = c.u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small);
    }
    c.0
}

/// 4E's token loop: N tokens in one cooperative launch equal the served chain
/// run token by token (embedding, two layers of q/k/v, norm/RoPE/append,
/// attention, o, MLP with a Tetra then an int4 down, final norm, head,
/// argmax): the tokens, both layers' KV caches, the final residual. Each
/// RoPE form, both row variants (768 threads without the rows' prefetch, 512
/// with it); an end-of-sequence stops the loop on every CTA; the barrier
/// counter ends at (tokens made) x (2 + 6 x layers) x grid; positions past
/// 256 take attention's split fold, whose tickets end at zero.
#[test]
fn the_4e_token_loop_is_the_served_chain() {
    let n = 3u32;
    let split = |b: &[u8]| -> (Vec<u32>, Vec<u8>, Vec<u32>) {
        let ring = u32s(&b[..4 * n as usize]);
        let tail = u32s(&b[b.len() - 8..]);
        (ring, b[4 * n as usize..b.len() - 8].to_vec(), tail)
    };
    for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, u32::MAX, 0x4e00 + form as u64, variant);
        let (want_ring, want_state, _) = split(&run(served(), &cmd));
        let (ring, state, tail) = split(&run(persistent(), &cmd));
        assert!(want_ring.iter().all(|&t| t < 1024), "the served run made {n} tokens: {want_ring:?}");
        assert_eq!(ring, want_ring, "form {form}, grid {grid}: tokens");
        assert!(state == want_state, "form {form}, grid {grid}: residual or KV caches differ");
        assert_eq!(tail, vec![n * llvq_cuda::pers::n_bar_4e(2) * grid, 0], "barrier count and abort");
        // an end-of-sequence at the second token stops every CTA after it
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, want_ring[1], 0x4e00 + form as u64, variant);
        let (ring, _, tail) = split(&run(persistent(), &cmd));
        assert_eq!(ring, vec![want_ring[0], want_ring[1], u32::MAX], "EOS");
        assert_eq!(tail[0], 2 * llvq_cuda::pers::n_bar_4e(2) * grid, "two tokens' barriers");
    }
    // Past 256 positions: two splits, the tickets' fold, both variants.
    for (variant, threads) in [(2u32, 128u32), (0, 64)] {
        let cmd = g_cmd_variant(1, 3, threads, &[(270, 2, 7)], 288, u32::MAX, 0x4e30, variant);
        let want = run(served(), &cmd);
        let got = run(persistent(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "positions 270-271, variant {variant}: tokens, residual or caches");
        assert_eq!(u32s(&got[got.len() - 8..]), vec![2 * llvq_cuda::pers::n_bar_4e(2) * 3, 0], "barriers, abort and tickets");
    }
    let f = "llvq-cuda/kernels/pers_4e.cu";
    let cmd = g_cmd(0, 3, 128, n, 5, 16, 7, u32::MAX, 0x4e00);
    let want = run(served(), &cmd)[..4 * n as usize].to_vec();
    // the segment's row stream (the 512-thread symbols): the deferred SiLU
    // lookup and the residual loaded at a down row's start
    let cmd512 = g_cmd_variant(2, 4, 96, &[(5, n, 7)], 16, u32::MAX, 0x4e02, 2);
    let want512 = run(served(), &cmd512)[..4 * n as usize].to_vec();
    let fs = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, from, to) in [
        ("silu_deferred", "pend_sv = A->silu[g];", "pend_sv = A->silu[g ^ 1u];"),
        ("down_residual_row", "xr = __ldcg(L.x + dlo + warp + k * nw);", "xr = __ldcg(L.x + dlo + warp + k * nw + 1u);"),
        ("silu_last_pair", "        if (lane == 0u && pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);\n", ""),
    ] {
        let b = mutant(tag, fs, from, to);
        let got = run_mutant(&b, &cmd512).map(|o| o[..4 * n as usize].to_vec());
        assert!(got != Some(want512.clone()), "segment stream mutant {tag} survived");
    }
    for (tag, from, to) in [
        ("4e_pos", "const u32 pos = L.pos0 + t;", "const u32 pos = L.pos0 + t + (t > 0u ? 1u : 0u);"),
        ("4e_ping_pong", "const SegLaunch S{A->x_b, A->x_a,", "const SegLaunch S{A->x_a, A->x_a,"),
        ("4e_attn_len", "A->cap, pos + 1u, A->c,", "A->cap, pos, A->c,"),
    ] {
        let b = mutant(tag, f, from, to);
        let got = run_mutant(&b, &cmd).map(|o| o[..4 * n as usize].to_vec());
        assert!(got != Some(want.clone()), "4E mutant {tag} survived");
    }
}

/// Consecutive decodes through one state, as `Runtime4e::decode` runs them
/// and as fusedrun's 4E mode times them: the same launch again (rewound to
/// the prompt), a launch that continues, then a rewind again. Every launch's
/// tokens, the caches, the residual and the barrier count equal the served
/// chain run the same way. The first card reading (job 6aebcb97) decoded its
/// first launch right and a rewound one wrong.
#[test]
fn consecutive_4e_decodes_keep_their_tokens() {
    let runs = [(5u32, 3u32, 7u32), (5, 3, 7), (8, 2, 9), (5, 3, 7)];
    let made: u32 = runs.iter().map(|r| r.1).sum();
    for (form, grid, threads, seed) in [(0u32, 3u32, 128u32, 0x4e10u64), (1, 4, 96, 0x4e11), (2, 3, 64, 0x4e12)] {
        let cmd = g_cmd_runs(form, grid, threads, &runs, 16, u32::MAX, seed);
        let want = run(served(), &cmd);
        let got = run(persistent(), &cmd);
        let rings = |b: &[u8]| u32s(&b[..4 * made as usize]);
        let (wr, gr) = (rings(&want), rings(&got));
        assert!(wr.iter().all(|&t| t < 1024), "the served runs made every token: {wr:?}");
        assert_eq!(&wr[..3], &wr[3..6], "the served chain rewound gives the same tokens");
        assert_eq!(gr, wr, "form {form}, grid {grid}: the tokens of consecutive launches");
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "form {form}, grid {grid}: residual or caches");
        let tail = u32s(&got[got.len() - 8..]);
        assert_eq!(tail, vec![made * llvq_cuda::pers::n_bar_4e(2) * grid, 0], "barrier count and abort");
    }
    // Teeth: without the clear at launch start, a rewound launch reads the
    // previous launch's key (form 1, grid 4 caught it before the fix).
    let b = mutant_many(
        "4e_slots_kept",
        &[("llvq-cuda/kernels/pers_4e.cu", "        A->slots[0] = 0ull;\n        A->slots[1] = 0ull;\n", "")],
    );
    let cmd = g_cmd_runs(1, 4, 96, &runs, 16, u32::MAX, 0x4e11);
    let want = u32s(&run(served(), &cmd)[..4 * made as usize]);
    let got = run_mutant(&b, &cmd).map(|o| u32s(&o[..4 * made as usize]));
    assert!(got != Some(want), "the slots-kept mutant survived");
}

/// A launch that faults (the NaN flag raised: every CTA leaves at its first
/// head) fails alone once the host resets the counter, both flags and its
/// barrier base, as `pers::p4e_settle` does: the launches after it make the
/// served chain's tokens. Before the reset (audit 2026-11-08 §1.1) every
/// later launch left at its first head too, and the runtime answered 500
/// until a restart. A reset of the flags and the counter that keeps the base
/// deadlocks the next launch on its first barrier, until the watchdog.
#[test]
fn a_faulted_4e_launch_fails_alone_after_the_reset() {
    let runs = [(5u32, 3u32, 7u32), (5, 3, 7), (5, 3, 7), (8, 2, 9)];
    let made: usize = runs.iter().map(|r| r.1 as usize).sum();
    let cmd = |policy: u32| {
        let mut c = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e40, 128);
        c.extend_from_slice(&1u32.to_le_bytes()); // the second launch faults
        c.extend_from_slice(&policy.to_le_bytes());
        c
    };
    let want = u32s(&run(served(), &cmd(0))[..4 * made]);
    assert!(want.iter().all(|&t| t < 1024), "the served runs made every token: {want:?}");
    let rings = |b: &[u8]| {
        let r = u32s(&b[..4 * made]);
        (r[..3].to_vec(), r[3..6].to_vec(), r[6..].to_vec())
    };
    let got = run(persistent(), &cmd(0));
    let (first, faulted, after) = rings(&got);
    assert_eq!(first, want[..3].to_vec(), "the launch before the fault");
    assert_eq!(faulted, vec![u32::MAX; 3], "the faulted launch wrote no token");
    assert_eq!(after, want[6..].to_vec(), "the launches after the reset make the served tokens");
    let tail = u32s(&got[got.len() - 8..]);
    assert_eq!(tail, vec![5 * pers::n_bar_4e(2) * 4, 0], "the counter restarted at 0, no abort, tickets at zero");
    // Teeth: the code before the reset, and a reset that keeps the base.
    for (policy, what) in [(1u32, "no reset"), (3, "flags and counter, base kept")] {
        let (_, _, after) = rings(&run(persistent(), &cmd(policy)));
        assert_ne!(after, want[6..].to_vec(), "policy {policy} ({what}) made the served tokens");
    }
}

/// 4E's phase-stamp symbols (`LLVQ_PERS_4E`'s stamped launch) make the
/// tokens, caches and residual of the token symbols, over two launches
/// through one state, and their stamps read as the barriers they bracket:
/// every CTA's stamps in event order, every release after the wait's last
/// arrival, the phases' walls summing to the token. The harness clock counts
/// calls, so this proves the order of the stamps, never a duration.
#[test]
fn the_4e_phase_stamps_keep_the_tokens_and_bracket_the_waits() {
    use llvq_cuda::pers::{p4e_events, p4e_phase_report};
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let made: usize = runs.iter().map(|r| r.1 as usize).sum();
    let ev = p4e_events(2) as usize;
    for (form, grid, threads, seed, variant) in [(0u32, 3u32, 128u32, 0x4e20u64, 1u32), (1, 4, 96, 0x4e21, 3)] {
        let plain = run(persistent(), &g_cmd_runs(form, grid, threads, &runs, 16, u32::MAX, seed));
        let cmd = g_cmd_variant(form, grid, threads, &runs, 16, u32::MAX, seed, variant);
        let want = run(served(), &cmd);
        assert!(want[..want.len() - 8] == plain[..plain.len() - 8], "form {form}, grid {grid}: the token symbols are the served chain");
        let got = run(persistent(), &cmd);
        let n_st = made * ev * grid as usize;
        assert_eq!(got.len(), plain.len() + 8 * n_st, "the stamps follow the tail");
        assert!(got[..plain.len()] == plain[..], "form {form}, grid {grid}: the stamped launch moved an output");
        let st: Vec<u64> = got[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        let (first, second) = st.split_at(3 * ev * grid as usize);
        let r = p4e_phase_report(first, 3, 2, grid).expect("the first launch's stamps");
        assert_eq!(r.tokens_read, 2);
        assert_eq!(r.grain, 1, "the harness clock steps by one");
        let r = p4e_phase_report(second, 2, 2, grid).expect("the second launch's stamps");
        let walls: f64 = r.phases.iter().map(|p| p.wall.0).sum();
        assert_eq!(walls, r.token.0, "one token read: the phases' walls are the token");
        assert!(r.phases.iter().all(|p| p.crit.0 >= p.mean.0 && p.mean.0 > 0.0), "{r:?}");
        // the inner stamps: every part worked, and the parts are their phase
        assert!(r.subs.iter().all(|p| p.crit.0 >= p.mean.0 && p.mean.0 > 0.0), "{r:?}");
        let parts = |lo: usize, hi: usize| r.subs[lo..hi].iter().map(|p| p.mean.0).sum::<f64>();
        assert!((parts(0, 2) - r.phases[1].mean.0).abs() < 1e-9, "q/k/v parts");
        assert!((parts(2, 7) - r.phases[5].mean.0).abs() < 1e-9, "MLP parts");
    }
    // Teeth. A release left unstamped, and an arrival stamped after the wait
    // instead of before it: the report refuses both.
    let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 1);
    let plain_len = run(persistent(), &g_cmd_runs(1, 4, 96, &runs, 16, u32::MAX, 0x4e21)).len();
    let first_launch = |o: &[u8]| -> Vec<u64> {
        o[plain_len..plain_len + 8 * 3 * ev * 4].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect()
    };
    for (tag, edits) in [
        ("4e_release_unstamped", vec![("llvq-cuda/kernels/pers_4e.cu", "        PERS_STAMP_AT((e) + 1u);", "        PERS_STAMP_AT((e) + 0u);")]),
        (
            "4e_arrival_after_wait",
            vec![
                ("llvq-cuda/kernels/pers_4e.cu", "            PERS_STAMP_AT(e);", "            (void)0;"),
                ("llvq-cuda/kernels/pers_4e.cu", "        PERS_STAMP_AT((e) + 1u);", "        PERS_STAMP_AT(e); PERS_STAMP_AT((e) + 1u);"),
            ],
        ),
    ] {
        let b = mutant_many(tag, &edits);
        let o = run_mutant(&b, &cmd).unwrap_or_else(|| panic!("{tag}: the mutant must still run"));
        let r = p4e_phase_report(&first_launch(&o), 3, 2, 4);
        assert!(r.is_err(), "the {tag} mutant survived: {r:?}");
    }
}
