//! The verify pass (`pers_4ev.cu`): n rows of the decode in one launch, against
//! the served chain, and its launch mirror. One file for the verify kernel.

use super::*;

// ---- The verify pass (pers_4ev.cu): n rows of the decode in one launch -----

/// `PERS_ATTN_SMEM(4)` of form 7 in floats: the smallest staging area a
/// verify launch takes at the 4B's groups. At it the o rows run in groups of
/// 3 and 1 and the down rows one at a time.
const V_ATTN4: u32 = 14_880;
/// The area that takes the down's 9,728 two rows at a time, and every row
/// phase four at a time (`P4V_SSM` and the head's maxima, then the rows).
const V_DOWN2: u32 = 2 * 9728 + 648;
const V_ALL4: u32 = 4 * 9728 + 648;

/// A `'G'` command whose last run is a verify run (variant bit 32): its rows
/// are that run's tok0 then `forced`, on a staging area of `act_floats`.
#[allow(clippy::too_many_arguments)]
fn v_cmd(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, seed: u64, variant: u32, act_floats: u32, forced: &[u32]) -> Vec<u8> {
    assert_eq!(forced.len() + 1, runs.last().expect("a run").1 as usize, "forced tokens: the verify run's rows less one");
    let mut c = g_cmd_variant(form, grid, threads, runs, cap, u32::MAX, seed, variant | 32);
    c.extend(act_floats.to_le_bytes());
    for &t in forced {
        c.extend(t.to_le_bytes());
    }
    c
}

/// The verify launch is the served chain run token by token on the same
/// tokens: each row's argmax, both layers' KV caches (every row appends its
/// position), the last row's residual, and the barrier count of one 4E token
/// whatever the rows. Every RoPE form; a staging area that shares every row
/// phase across the four rows, one that takes the down two rows at a time,
/// one that takes it a row at a time and the o rows 3 + 1; the 8B's Tetra o
/// and untied head; after a 4E launch through the same state; rows across
/// the first split (lengths 255 to 258, the fold of row 2 and 3); one, two
/// and three rows.
#[test]
fn the_4e_verify_rows_are_the_served_chain() {
    let nb = llvq_cuda::pers::n_bar_4e(2);
    let check = |cmd: &[u8], grid: u32, bars: u32, what: &str| {
        let want = run(served(), cmd);
        let got = run(persistent_v9(), cmd);
        assert_eq!(got.len(), want.len(), "{what}: output length");
        let n = got.len() - 8;
        assert!(got[..n] == want[..n], "{what}: tokens, residual or KV caches differ from the served chain");
        assert_eq!(u32s(&got[n..]), vec![bars * grid, 0], "{what}: barrier count, abort and tickets");
        want
    };
    let forced = [11u32, 523, 7];
    for (form, grid, threads, act, variant) in
        [(0u32, 3u32, 128u32, V_ALL4, 0u32), (1, 4, 96, V_DOWN2, 0), (2, 5, 64, V_ATTN4, 0), (1, 3, 128, V_ATTN4, 16), (1, 4, 512, V_DOWN2, 16)]
    {
        let cmd = v_cmd(form, grid, threads, &[(5, 4, 7)], 16, 0xf400 + form as u64, variant, act, &forced);
        let want = check(&cmd, grid, nb, &format!("form {form}, grid {grid}x{threads}, area {act}, variant {variant}"));
        let ring = u32s(&want[..16]);
        assert!(ring.iter().all(|&t| t < 1024), "four argmaxes: {ring:?}");
    }
    // after a 4E launch of 3 tokens, through the same state and counter
    let cmd = v_cmd(1, 4, 96, &[(5, 3, 7), (8, 4, 9)], 16, 0xf410, 2, V_DOWN2, &forced);
    check(&cmd, 4, 4 * nb, "a 4E launch, then a verify launch");
    // rows across the first split: lengths 255, 256, 257, 258
    let cmd = v_cmd(1, 3, 128, &[(254, 4, 7)], 288, 0xf420, 0, V_ATTN4, &forced);
    check(&cmd, 3, nb, "rows across 256");
    let cmd = v_cmd(2, 4, 96, &[(270, 3, 7)], 288, 0xf421, 16, V_ALL4, &forced[..2]);
    check(&cmd, 4, nb, "three rows past 256, the 8B's parts");
    // one row: the verify launch is a 4E token
    let cmd = v_cmd(0, 3, 128, &[(5, 1, 7)], 16, 0xf430, 0, V_ATTN4, &[]);
    check(&cmd, 3, nb, "one row");
    // two rows, the 8B's parts; three rows on an area that shares every
    // phase, the last row of each group on the vector ln
    let cmd = v_cmd(1, 4, 96, &[(5, 2, 7)], 16, 0xf431, 16, V_ATTN4, &forced[..1]);
    check(&cmd, 4, nb, "two rows");
    let cmd = v_cmd(0, 3, 128, &[(5, 3, 7)], 16, 0xf432, 0, V_ALL4, &forced[..2]);
    check(&cmd, 3, nb, "three rows, every phase shared");
    // a stamped launch: the same rows, every CTA's clock at every event
    let cmd = v_cmd(1, 4, 96, &[(5, 4, 7)], 16, 0xf433, 64, V_DOWN2, &forced);
    check(&cmd, 4, nb, "four rows, stamped");
}

/// Each guard of the verify pass has a mutant, killed by a verify command:
/// a row's attention length, its RoPE position, site 1 of the Tetra rows, an
/// int4 row's vector stride, a group's v rows written to its first row, the
/// o residual, the down's rotation, the v rows read on signed values, the
/// vector ln that stages no `small`, the head key's row, the tickets left as
/// they were (rows past 256 fold on a stale count), each phase stamp, the
/// last row's ln scratch on its own row.
#[test]
fn the_4e_verify_mutants_are_killed() {
    let forced = [11u32, 523, 7];
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [
        v_cmd(1, 3, 128, &[(5, 4, 7)], 16, 0xf440, 0, V_ALL4, &forced),
        v_cmd(2, 4, 96, &[(254, 4, 7)], 288, 0xf441, 16, V_ATTN4, &forced),
        v_cmd(1, 3, 128, &[(5, 3, 7)], 16, 0xf442, 64, V_ALL4, &forced[..2]),
    ]
    .into_iter()
    .map(|c| {
        let w = run(persistent_v9(), &c);
        (c, w)
    })
    .collect();
    let f = "llvq-cuda/kernels/pers_4ev.cu";
    for (tag, from, to) in [
        ("v_len", "const u32 l = L.pos0 + r + 1u;", "const u32 l = L.pos0 + 1u;"),
        ("v_rope_pos", "hh = w - r * per, pos = L.pos0 + r;", "hh = w - r * per, pos = L.pos0;"),
        ("v_site1", "acc[r] = __fmaf_rn(__fmul_rn(a, gs), iv, acc[r]);", "acc[r] = __fmaf_rn(a, __fmul_rn(gs, iv), acc[r]);"),
        ("v_q4_stride", "const float d = pers_q4_deq((wv[s] >> (4u * k)) & 0xfu, sb, bb);\n#pragma unroll\n                for (u32 r = 0; r < P4V_KMAX; ++r)\n                    if (r < n) acc[r] = __fmaf_rn(d, xs[r * xst + c + k], acc[r]);", "const float d = pers_q4_deq((wv[s] >> (4u * k)) & 0xfu, sb, bb);\n#pragma unroll\n                for (u32 r = 0; r < P4V_KMAX; ++r)\n                    if (r < n) acc[r] = __fmaf_rn(d, xs[c + k], acc[r]);"),
        ("v_v_row", "L.v_raw[(size_t)(r0 + r) * dkv + rv] = f2h(o[r]);", "L.v_raw[(size_t)r0 * dkv + rv] = f2h(o[r]);"),
        ("v_o_resid", "L.x_b[at] = exact_hadd(__ldcg(L.x_a + at), f2h(o[r]));", "L.x_b[at] = f2h(o[r]);"),
        ("v_down_rot", "pers_rot_finish<true, true>(R + r * I, ssm, I, M.m_i, M.k_i, M.inv_i);", "__syncthreads();"),
        ("v_v_natural", "                flip();\n                for (u32 rv = vlo + warp;", "                for (u32 rv = vlo + warp;"),
        ("v_ln_small", "pers_ln_1rt<0>(x, alpha, signs, w, eps, small, k, ssm, s, scratch, false);", "pers_ln_1rt<0>(x, alpha, signs, w, eps, nullptr, k, ssm, s, scratch, false);"),
        ("v_head_key", "else atomicMax(cmax + r, pers_tie_key(hb, row));", "else atomicMax(cmax + r, pers_tie_key(hb, row ^ 1u));"),
        ("v_tickets", "for (u32 i = threadIdx.x; i < P4V_KMAX * A->n_kv; i += blockDim.x) L.tickets[i] = 0u;", ""),
        ("v_stamp_wait", "        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)nb * G + c0] = pers_now(); \\\n", ""),
        ("v_stamp_start", "    if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[c0] = pers_now();\n", ""),
        ("v_ln_last", "return (r + 2u) * w <= FR ? R + (r + 1u) * w : nullptr;", "return (r + 2u) * w <= FR ? R + r * w : nullptr;"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "verify mutant {tag} survived");
    }
}

/// `pers.rs::P4vLaunch` against `Pers4vLaunch` of the verify kernel: its
/// size, its fields' offsets, and the row bound.
#[test]
fn the_verify_launch_mirror_matches_the_kernel() {
    let o = u32s(&run(persistent_v9(), &Cmd::new(b'P').0));
    assert_eq!(o, vec![pers::P4V_LAUNCH_BYTES as u32, 12, 16, 24, 32, 88, 96, 104, 112, 120, 128, pers::P4V_KMAX]);
    assert_eq!(std::mem::size_of::<pers::P4vLaunch>(), pers::P4V_LAUNCH_BYTES);
    // 4B: every phase of 4 rows fits under the L40S's 101,376 B; the 14B's
    // down does not, and takes the most that fits
    assert_eq!(pers::p4v_act_floats(2560, 9728, 4096, 4, 101_376), Ok(20_576));
    assert_eq!(pers::p4v_act_floats(2560, 9728, 4096, 4, 200_000), Ok(4 * 9728 + 648));
    assert_eq!(pers::p4v_act_floats(1024, 3072, 2048, 2, 101_376), Ok(13_584));
    assert!(pers::p4v_act_floats(5120, 17408, 5120, 5, 60_000).is_err());
}
