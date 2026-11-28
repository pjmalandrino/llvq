//! Route P's step A segment, the row-loop and verify-row microbenchmarks, and
//! plan-300 step 1 (K4's arms, K11's int4 rows, K3b's counters). One file
//! because each is a research kernel held to the served rows, not a 4E form.

use super::*;

// ---- route P, step A ------------------------------------------------------

fn seg_cmd(grid: u32, threads: u32, normv: u32, tetra_down: bool, seed: u64, bar_base: u32) -> Vec<u8> {
    let (h, i) = (2560u32, 9728u32);
    let (rh, ri) = (rot(h as usize, 31), rot(i as usize, 32));
    let gu: Vec<u32> = pers::slices(i, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let dn: Vec<u32> = pers::slices(h, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    Cmd::new(b'S')
        .u32(grid)
        .u32(threads)
        .u32(normv)
        .u32(u32::from(tetra_down))
        .u64(seed)
        .u32(h)
        .u32(i)
        .u32(rh.m)
        .u32(rh.k)
        .u32(ri.m)
        .u32(ri.k)
        .f32(rh.inv)
        .f32(ri.inv)
        .f32(1e-6)
        .u32(bar_base)
        .u32s(&rh.signs)
        .u32s(&ri.signs)
        .f32s(&rh.small)
        .f32s(&ri.small)
        .u32s(&gu)
        .u32s(&dn)
        .0
}

/// The MLP segment in one launch equals the eager chain of eight (Tetra down)
/// or six (int4 down) launches, for both norm variants, and its one barrier
/// moves the counter by one grid.
#[test]
fn the_step_a_segment_is_the_eager_chain() {
    for tetra_down in [true, false] {
        for normv in [0u32, 1] {
            let bar_base = 0xffff_fff0u32; // across the wrap
            let cmd = seg_cmd(3, 128, normv, tetra_down, 0x5e9 + normv as u64, bar_base);
            let want = run(served(), &cmd);
            let got = run(persistent(), &cmd);
            let n = 2 * (2560 + 9728);
            assert!(want[..n] == got[..n], "segment differs (tetra_down {tetra_down}, variant {normv})");
            let t = u32s(&got[n..]);
            assert_eq!(t[1], 0, "the segment aborted");
            assert_eq!(t[0], bar_base.wrapping_add(3), "one barrier, one arrival a CTA");
        }
    }
}

#[test]
fn step_a_mutants_are_killed() {
    let cmd = seg_cmd(3, 96, 0, true, 0x5ea, 0);
    let cmd_q4 = seg_cmd(3, 96, 0, false, 0x5eb, 0);
    let want = run(served(), &cmd);
    let want_q4 = run(served(), &cmd_q4);
    let f = "llvq-cuda/kernels/pers_segment.cu";
    let cases = [
        ("seg_no_barrier", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x, A->abort_flag)) return;", "(void)0;"),
        ("seg_barrier_target", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x, A->abort_flag)) return;", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x + 1u, A->abort_flag)) return;"),
        ("seg_residual_order", "L.out[r] = exact_hadd(__ldcg(L.x + r), f2h(y));", "L.out[r] = f2h(__fadd_rn(h2f(__ldcg(L.x + r)), y));"),
        ("seg_silu_operand", "A->act[r] = exact_silu_mul(A->silu, g, u);", "A->act[r] = exact_silu_mul(A->silu, u, g);"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = run(&b, &cmd) != want || run(&b, &cmd_q4) != want_q4;
        assert!(killed, "segment mutant {tag} survived");
    }
}

/// Grids of 2 to 5 CTAs give the same bits: every interleaving the harness
/// tries is the served chain.
#[test]
fn the_segment_does_not_depend_on_the_grid() {
    let want = run(served(), &seg_cmd(4, 64, 0, false, 1, 0));
    let n = 2 * (2560 + 9728);
    for grid in 2..=5u32 {
        assert!(run(persistent(), &seg_cmd(grid, 64, 0, false, 1, 0))[..n] == want[..n], "grid {grid}");
    }
}

// ---- the row-loop microbenchmark (pers_trows.cu) ---------------------------

// Every tail is 8 or 16 wide, as on the 4B: the stream reads the tail as
// 16-byte chunks (pers_tetra_stream).
const TROWS_SITES: [(u32, u32, u64); 5] = [(64, 2560, 0x7101), (48, 4096, 0x7102), (40, 9728, 0x7103), (16, 128, 0x7104), (23, 2560, 0x7105)];

fn trows_cmd(arm: u32, grid: u32, threads: u32, stamps: u32, sites: &[(u32, u32, u64)]) -> Vec<u8> {
    let mut c = Cmd::new(b'J').u32(arm).u32(grid).u32(threads).u32(stamps).u64(0x7e70).u32(sites.len() as u32);
    for &(o, i, s) in sites {
        c = c.u32(o).u32(i).u64(s);
    }
    c.0
}

/// The four arms that owe the bits (V2, V3, ILP2, V2U) give the served rows on
/// every site, grid and warp count, with and without stamps; the three
/// diagnostics run and move rows, so each changes what it says it changes.
#[test]
fn the_row_loop_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    let rows: u32 = TROWS_SITES.iter().map(|s| s.0).sum();
    assert_eq!(want.len(), 6 * rows as usize, "binary16 rows, then their f32 values");
    for arm in 0..4u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "row-loop arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
    for arm in 4..7u32 {
        let got = run(persistent(), &trows_cmd(arm, 3, 96, 1, &TROWS_SITES));
        assert_eq!(got.len(), want.len());
        assert!(got != want, "diagnostic arm {arm} gives the served rows: it changes nothing");
    }
}

/// K2 (one decode, two activations): its rows on each activation are the
/// served rows on that activation, and its rows on the first are the k = 1
/// arm's (V2) bit for bit, binary16 and f32 before f2h, at every grid.
#[test]
fn the_k2_arm_is_the_k1_arm_on_each_activation() {
    let want = run(served(), &trows_cmd(7, 1, 64, 0, &TROWS_SITES));
    let rows: usize = TROWS_SITES.iter().map(|s| s.0 as usize).sum();
    assert_eq!(want.len(), 12 * rows, "two activations: binary16 rows, then their f32 values");
    let k1 = run(persistent(), &trows_cmd(0, 3, 96, 0, &TROWS_SITES));
    let (h, f) = (2 * rows, 4 * rows);
    // served layout: [y(x), y(x2)] then [f(x), f(x2)]
    assert_eq!(&want[..h], &k1[..h], "the served rows on the first activation are V2's");
    assert_eq!(&want[2 * h..2 * h + f], &k1[h..], "and so are their f32 values");
    assert!(want[..h] != want[h..2 * h], "the two activations must give different rows");
    for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
        let got = run(persistent(), &trows_cmd(7, grid, threads, stamps, &TROWS_SITES));
        assert!(got == want, "K2, grid {grid}x{threads}, stamps {stamps}: rows differ from the k = 1 rows on their activation");
    }
}

#[test]
fn row_loop_mutants_are_killed() {
    let cmds: Vec<Vec<u8>> = (0..4u32).map(|arm| trows_cmd(arm, 3, 96, 0, &TROWS_SITES)).collect();
    let want = run(served(), &cmds[0]);
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let cases = [
        ("ilp2_fold_order", "        acc = pers_fold(acc, a0, q0, h0, gscale, invnorm);\n        if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);",
         "        if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);\n        acc = pers_fold(acc, a0, q0, h0, gscale, invnorm);"),
        ("ilp2_second_x", "const float* x1 = two ? x0 + 32u * 24u : x0;", "const float* x1 = x0;"),
        // Removing the guard is an equivalent mutant: zero words decode to the
        // zero vector, whose fold leaves acc as it is. Tightening it is not.
        ("ilp2_guard", "if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);", "if (j + 64u < nblocks) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);"),
        ("ilp2_prefetch", "if (j + 96u < nblocks) f1r_load(wrow, j + 96u, m1, g1);", "if (j + 64u < nblocks) f1r_load(wrow, j + 64u, m1, g1);"),
        // V2U's guard keeps reads inside the staged activation; for the bits it
        // is equivalent (zero words past the row), so the trip count is mutated.
        ("uniform_trip", "const u32 nit = (nblocks + 31u) >> 5;\n    float acc = 0.0f;\n    u32 lo = 0u, hi16 = 0u;", "const u32 nit = nblocks >> 5;\n    float acc = 0.0f;\n    u32 lo = 0u, hi16 = 0u;"),
        ("cta_slice", "hi = (u32)((unsigned long long)d_out * (c + 1u) / g);", "hi = (u32)((unsigned long long)d_out * (c + 1u) / g) - (c == 0u);"),
        ("warp_stride", "for (u32 r = lo + warp; r < hi; r += nw) {\n                const u32* wrow", "for (u32 r = lo + warp; r < hi; r += nw + (nw > 1u)) {\n                const u32* wrow"),
        ("v3_rows", "            pers_tetra_stream<PERS_STREAM_D>(pers_warp_rows(lo, hi, warp, nw), nb, tw, tab, t_inv, xs, lane, row_of,",
         "            pers_tetra_stream<PERS_STREAM_D>(pers_warp_rows(lo, hi, warp, nw) - (warp == 0u), nb, tw, tab, t_inv, xs, lane, row_of,"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().any(|c| run_mutant(&b, c).is_none_or(|got| got != want));
        assert!(killed, "row-loop mutant {tag} survived");
    }
    // K2, against its own served output (two activations).
    let k2cmd = trows_cmd(7, 3, 96, 0, &TROWS_SITES);
    let k2want = run(served(), &k2cmd);
    let k2cases = [
        ("k2_second_fold", "acc2 = pers_fold(acc2, b, q, hi16, gscale, invnorm);", "acc2 = pers_fold(acc2, a, q, hi16, gscale, invnorm);"),
        ("k2_second_x", "const float* xb = xs2 + j * 24u;", "const float* xb = (xs2 == xs ? xs2 : xs) + j * 24u;"),
        ("k2_staging", "if (ARM == TROWS_K2) pers_stage_f32<24u, false>(x2, xs2, nb, tw);", "if (ARM == TROWS_K2) pers_stage_f32<24u, false>(x2 == x ? x2 : x, xs2, nb, tw);"),
        ("k2_second_tail", "const float v2 = pers_epilogue(acc2, rs[r], tail + (size_t)r * tw, xs2 + nb * 24u, tw);", "const float v2 = pers_epilogue(acc2, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);"),
    ];
    for (tag, from, to) in k2cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &k2cmd).is_none_or(|got| got != k2want), "row-loop mutant {tag} survived");
    }
}


// ---- the verify rows (pers_vrows.cu): B1's levers and k rows a pass --------

// Odd and even block counts, a row of 33 blocks (rotation across many rows of
// one warp), a row under 32 (rotation falls back to v2's row), a tail of 0.
const VROWS_SITES: [(u32, u32, u64); 7] =
    [(64, 2560, 0x7201), (48, 4096, 0x7202), (40, 9728, 0x7203), (16, 128, 0x7204), (23, 2568, 0x7205), (37, 800, 0x7206), (9, 4096, 0x7207)];

fn vrows_cmd(arm: u32, grid: u32, threads: u32, xseed: u64, sites: &[(u32, u32, u64)]) -> Vec<u8> {
    let mut c = Cmd::new(b'D').u32(arm).u32(grid).u32(threads).u64(xseed).u32(sites.len() as u32);
    for &(o, i, s) in sites {
        c = c.u32(o).u32(i).u64(s);
    }
    c.0
}

/// One 'D' output split: rows (binary16) and f32 values, by activation.
struct VOut {
    y: Vec<Vec<u16>>,
    f: Vec<Vec<f32>>,
}
fn vout(b: &[u8], rows: usize) -> VOut {
    assert_eq!(b.len(), 4 * rows * 6, "four activations: binary16 rows, then f32 values");
    let y = u16s(&b[..8 * rows]);
    let f: Vec<f32> = u32s(&b[8 * rows..]).into_iter().map(f32::from_bits).collect();
    VOut { y: (0..4).map(|a| y[a * rows..(a + 1) * rows].to_vec()).collect(), f: (0..4).map(|a| f[a * rows..(a + 1) * rows].to_vec()).collect() }
}
fn vrows_rows() -> usize {
    VROWS_SITES.iter().map(|s| s.0 as usize).sum()
}
const VROWS_GRIDS: [(u32, u32); 4] = [(3, 96), (2, 64), (4, 128), (5, 32)];

/// v2, the relabeled decoder and lane rotation give the served rows on every
/// site, grid and warp count, binary16 and f32 before f2h; K2 gives them on
/// both its activations.
#[test]
fn the_verify_row_arms_that_owe_the_bits_keep_them() {
    let rows = vrows_rows();
    let want = vout(&run(served(), &vrows_cmd(0, 1, 64, 0x7e70, &VROWS_SITES)), rows);
    assert!(want.y[0] != want.y[1], "the activations must give different rows");
    for (arm, k) in [(0u32, 1usize), (3, 1), (4, 1), (1, 2)] {
        for &(grid, threads) in &VROWS_GRIDS {
            let got = vout(&run(persistent(), &vrows_cmd(arm, grid, threads, 0x7e70, &VROWS_SITES)), rows);
            for a in 0..k {
                assert!(got.y[a] == want.y[a], "arm {arm}, grid {grid}x{threads}, activation {a}: binary16 rows differ from tv_tetra48_h");
                assert!(got.f[a].iter().zip(&want.f[a]).all(|(g, w)| g.to_bits() == w.to_bits()), "arm {arm}, grid {grid}x{threads}, activation {a}: f32 values differ");
            }
            for a in k..4 {
                assert!(got.y[a].iter().all(|&v| v == 0xFFFF), "arm {arm} wrote rows on activation {a}, past its K");
            }
        }
    }
    let pword = vout(&run(persistent(), &vrows_cmd(2, 3, 96, 0x7e70, &VROWS_SITES)), rows);
    assert!(pword.y[0] != want.y[0], "pword decodes nothing and must not give the served rows");
}

/// B1's integer dot: I16 at K = 1 is within the activation rounding of the
/// served rows (max error under 1e-3 of the site's rms, f32 before f2h), and
/// at K = 2, 3, 4 each activation's rows are I16 at K = 1's on that activation,
/// bit for bit.
#[test]
#[allow(clippy::needless_range_loop)]
fn the_integer_dot_verify_rows_are_the_one_row_rows() {
    let rows = vrows_rows();
    let k1: Vec<VOut> = (0..4u64).map(|a| vout(&run(persistent(), &vrows_cmd(5, 3, 96, 0x7e70 + a, &VROWS_SITES)), rows)).collect();
    let want = vout(&run(served(), &vrows_cmd(0, 1, 64, 0x7e70, &VROWS_SITES)), rows);
    let mut at = 0usize;
    let mut worst = 0f64;
    for &(o, _, _) in &VROWS_SITES {
        let r = at..at + o as usize;
        for a in 0..4 {
            let w = &want.f[a][r.clone()];
            let g = &k1[a].f[0][r.clone()];
            let rms = (w.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / w.len() as f64).sqrt();
            let e = g.iter().zip(w).map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs()).fold(0f64, f64::max) / rms;
            worst = worst.max(e);
        }
        at += o as usize;
    }
    assert!(worst < 1e-3, "the int16 rows sit {worst:.2e} of the rms from the served rows");
    assert!(worst > 0.0, "the int16 rows equal the served rows: the integer dot is not running");
    for (arm, k) in [(6u32, 2usize), (7, 3), (8, 4)] {
        for &(grid, threads) in &VROWS_GRIDS {
            let got = vout(&run(persistent(), &vrows_cmd(arm, grid, threads, 0x7e70, &VROWS_SITES)), rows);
            for a in 0..k {
                assert!(got.y[a] == k1[a].y[0], "i16 K = {k}, grid {grid}x{threads}, activation {a}: rows differ from K = 1's");
                assert!(got.f[a].iter().zip(&k1[a].f[0]).all(|(g, w)| g.to_bits() == w.to_bits()), "i16 K = {k}, activation {a}: f32 values differ from K = 1's");
            }
        }
    }
}

#[test]
fn verify_row_mutants_are_killed() {
    let rows = vrows_rows();
    let f = "llvq-cuda/kernels/pers_vrows.cu";
    let served_out = run(served(), &vrows_cmd(0, 1, 64, 0x7e70, &VROWS_SITES));
    let want = vout(&served_out, rows);
    // Against the served rows: lean (arm 3) and rot (arm 4).
    let exact = [
        // Masking g (bit 6) off is an equivalent mutant: GB ignores it by construction. Masking a b2 bit is not.
        ("lean_gb_mask", 3u32, "((hi16 >> 9) & 0x7cu)", "((hi16 >> 9) & 0x78u)"),
        ("lean_r1", 3, "const u32 R1 = lo & 0xfffu;", "const u32 R1 = lo & 0x7ffu;"),
        ("lean_r3", 3, "const u32 r3 = (p ^ (R1 >> 11) ^ (mid ? 0u : 1u)) & 1u;", "const u32 r3 = (p ^ (R1 >> 11)) & 1u;"),
        ("lean_word_b3", 3, "(b1 << 24) | (b3 << 25)", "(b1 << 24) | ((b3 ^ b1) << 25)"),
        ("relayout_odd", 3, "d[3u * m + 2u] = (a1 >> 16) | (b1 << 16);", "d[3u * m + 2u] = (a1 >> 16) | (b << 16);"),
        // The padding half is never decoded, so keeping it is not tested; relabeling the last block's half is.
        ("relayout_last", 3, "d[3u * m + 1u] = b | (s[3u * m + 1u] & 0xffff0000u);", "d[3u * m + 1u] = s[3u * m + 1u];"),
        ("rot_end", 4, "if (end <= 32u * s + 32u) {", "if (end <= 32u * s + 31u) {"),
        ("rot_next_row", 4, "f1r_load(words + (size_t)(r0 + k2 * nw) * stride, j2, lo2, hi2);", "f1r_load(words + (size_t)(r0 + k * nw) * stride, j2, lo2, hi2);"),
        ("rot_mine", 4, "const bool mine = k == kc;", "const bool mine = true;"),
    ];
    for (tag, arm, from, to) in exact {
        let b = mutant(tag, f, from, to);
        let killed = VROWS_GRIDS.iter().any(|&(g, th)| run_mutant(&b, &vrows_cmd(arm, g, th, 0x7e70, &VROWS_SITES)).is_none_or(|o| vout(&o, rows).y[0] != want.y[0]));
        assert!(killed, "verify-row mutant {tag} survived");
    }
    // Against I16's own rows (arms 5 and 8): the integer dot and its staging.
    let ref_k1 = run(persistent(), &vrows_cmd(5, 3, 96, 0x7e70, &VROWS_SITES));
    let ref_k4 = run(persistent(), &vrows_cmd(8, 3, 96, 0x7e70, &VROWS_SITES));
    let ints = [
        ("i16_hi_lo", 5u32, "dot = __dp2a_hi((int)v.y, s[2u * i], dot);", "dot = __dp2a_lo((int)v.y, s[2u * i], dot);"),
        ("i16_scale", 5, "__uint_as_float(s16 << 16)), w, acc[a]);", "__uint_as_float(s16 << 15)), w, acc[a]);"),
        ("i16_order", 5, "const int lo = vrows_q16(v[TETRA48_ORDER[2u * k]], inv)", "const int lo = vrows_q16(v[2u * k], inv)"),
        ("i16_round_up", 5, "return (b + 0xffffu) >> 16;", "return b >> 16;"),
        ("i16_stage_x", 8, "f[a][i] = __ldcg(reinterpret_cast<const float4*>(xa[a0 + a] + 24u * b) + i);", "f[a][i] = __ldcg(reinterpret_cast<const float4*>(xa[a0] + 24u * b) + i);"),
        ("i16_area", 8, "const unsigned char* ar = areas + a * abytes;", "const unsigned char* ar = areas + (a >> 1) * abytes;"),
        ("i16_tail", 8, "const float* xt = reinterpret_cast<const float*>(act + a * abytes + abytes - 64u);", "const float* xt = reinterpret_cast<const float*>(act + abytes - 64u);"),
    ];
    for (tag, arm, from, to) in ints {
        let b = mutant(tag, f, from, to);
        let want = if arm == 5 { &ref_k1 } else { &ref_k4 };
        let got = run_mutant(&b, &vrows_cmd(arm, 3, 96, 0x7e70, &VROWS_SITES));
        assert!(got.as_ref() != Some(want), "verify-row mutant {tag} survived");
    }
}

// ---- plan-300 step 1: K4's row-loop arms, K11's int4 rows, K3b's counters ----

/// K4's five arms (NC, NC256, CPA, TILE, BAL: arms 8 to 12) give the served
/// rows on every site, grid and warp count, with and without stamps. The
/// 9,728 site holds three full tiles and 21 blocks left, the 4,096 one tile
/// and 42, the 2,560 none, so TILE runs both of its planes.
#[test]
fn the_step1_row_loop_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    for arm in 8..13u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "K4 arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
}

#[test]
fn step1_row_loop_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    let cases: [(&str, u32, &str, &str); 11] = [
        ("nc_word", 8, "const u32 w1 = trows_ldw<LD>(row + w + 1u);", "const u32 w1 = trows_ldw<LD>(row + w);"),
        ("nc_prefetch", 9, "if (j + 32u < nblocks) trows_load<LD>(wrow, j + 32u, lo2, hi2);", "if (j + 64u < nblocks) trows_load<LD>(wrow, j + 32u, lo2, hi2);"),
        ("cpa_slot", 10, "f1r_load(ring + TROWS_CPA_SLOT * (i % 3u), lane, lo, hi16);", "f1r_load(ring + TROWS_CPA_SLOT * ((i + 1u) % 3u), lane, lo, hi16);"),
        ("cpa_chunk", 10, "if (c + 4u <= row_u32) trows_cp16(", "if (c + 8u <= row_u32) trows_cp16("),
        ("cpa_ahead", 10, "        trows_cpa_issue(ring, wrow, row_u32, i + 2u, nit, lane);\n", "        trows_cpa_issue(ring, wrow, row_u32, i + 3u, nit, lane);\n"),
        ("tile_hi_half", 11, "acc = pers_block(acc, lo[s], (u32)(H >> (16u * s)) & 0xffffu,", "acc = pers_block(acc, lo[s], (u32)(H >> (16u * (s ^ 1u))) & 0xffffu,"),
        ("tile_put_lane", 11, "drow[192u * t + 4u * l + s] = lo;", "drow[192u * t + 4u * s + l] = lo;"),
        ("tile_next", 11, "if (t + 1u < nt) {\n            const u32* nx", "if (t + 2u < nt) {\n            const u32* nx"),
        ("tile_rem_hi", 11, "const unsigned short* rhi = reinterpret_cast<const unsigned short*>(rlo + rem);", "const unsigned short* rhi = reinterpret_cast<const unsigned short*>(rlo + rem + 1u);"),
        // A change of the stride alone is equivalent here (relayout and read
        // share it); pers.rs pins it to the served stride at the 4B's widths.
        ("tile_put_rem_hi", 11, "h[2u * (192u * nt + rem) + rr] = (unsigned short)hi16;", "h[2u * (192u * nt + rem) + (rr ^ 1u)] = (unsigned short)hi16;"),
        ("bal_last_row", 12, "if (r >= hi) break;", "if (r + 1u >= hi) break;"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = [(3u32, 96u32), (2, 64)].iter().any(|&(g, th)| run_mutant(&b, &trows_cmd(arm, g, th, 0, &TROWS_SITES)).is_none_or(|got| got != want));
        assert!(killed, "K4 mutant {tag} survived");
    }
}

/// The row lead's arms (2026-11-07): GB (v2 with a grid barrier after every
/// site) and GBC (the rows claimed across the grid) give the served rows on
/// every site, grid and warp count, with and without stamps; the harness
/// checks that every CTA arrived at the barrier after every site.
#[test]
fn the_rows_grid_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    for arm in [13u32, 14] {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0), (1, 64, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "row-lead arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
}

#[test]
fn rows_grid_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    // GB's row choice has no mutant: without its branch it falls to the
    // uniform loop, which owes the same bits (an equivalent mutant).
    let cases: [(&str, u32, &str, &str); 5] = [
        ("gbc_first", 14, "if (lane == 0u) r = atomicAdd(cur, 1u);", "if (lane == 0u) r = atomicAdd(cur, 1u) + 1u;"),
        ("gbc_next", 14, "if (lane == 0u) rn = atomicAdd(cur, 1u);", "if (lane == 0u) rn = atomicAdd(cur, 2u);"),
        ("gbc_bound", 14, "while (r < d_out) {", "while (r + 1u < d_out) {"),
        ("gbc_site_cursor", 14, "reinterpret_cast<u32*>(const_cast<float*>(x2)) + 2u + s;", "reinterpret_cast<u32*>(const_cast<float*>(x2)) + 2u;"),
        ("gb_barrier", 13, "if (!pers_grid_wait(ctr, (s + 1u) * g, ctr + 1)) return;", "(void)ctr;"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = [(3u32, 96u32), (2, 64)].iter().any(|&(g, th)| run_mutant(&b, &trows_cmd(arm, g, th, 0, &TROWS_SITES)).is_none_or(|got| got != want));
        assert!(killed, "row-lead mutant {tag} survived");
    }
}

// Int4 sites of the 4B's three shapes, cut to a few rows; d_in in whole
// steps of 32 words, d_out a multiple of 8 (tv_q4_hh's blocks). 9,728: 9
// full tiles of 128 words and 2 steps left.
const Q4_SITES: [(u32, u32, u64); 4] = [(16, 2560, 0x7401), (24, 4096, 0x7402), (8, 9728, 0x7403), (40, 2560, 0x7404)];

fn q4rows_cmd(arm: u32, grid: u32, threads: u32, stamps: u32, sites: &[(u32, u32, u64)]) -> Vec<u8> {
    let mut c = Cmd::new(b'y').u32(arm).u32(grid).u32(threads).u32(stamps).u64(0x7e74).u32(sites.len() as u32);
    for &(o, i, s) in sites {
        c = c.u32(o).u32(i).u64(s);
    }
    c.0
}

/// K11: 4E's int4 row (Q4), its read-only loads (NC) and its tile layout
/// (TILE) give tv_q4_hh's rows bit for bit, binary16 and f32 before f2h, on
/// every grid; READ, the diagnostic, owes nothing and moves the rows.
#[test]
fn the_int4_rows_alone_are_the_served_rows() {
    let want = run(served(), &q4rows_cmd(0, 1, 64, 0, &Q4_SITES));
    let rows: usize = Q4_SITES.iter().map(|s| s.0 as usize).sum();
    assert_eq!(want.len(), 6 * rows);
    for arm in 0..3u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (5, 128, 1), (4, 32, 0)] {
            let got = run(persistent(), &q4rows_cmd(arm, grid, threads, stamps, &Q4_SITES));
            assert!(got == want, "int4 arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows differ from tv_q4_hh");
        }
    }
    let got = run(persistent(), &q4rows_cmd(3, 3, 96, 1, &Q4_SITES));
    assert_eq!(got.len(), want.len());
    assert!(got != want, "READ gives the served rows: it computes something");
}

#[test]
fn int4_rows_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_q4rows.cu";
    let want = run(served(), &q4rows_cmd(0, 1, 64, 0, &Q4_SITES));
    let cases: [(&str, u32, &str, &str); 6] = [
        ("q4nc_next", 1, "p2 = trows_ldw<LD>(wq + w0 + wi + 32u);", "p2 = trows_ldw<LD>(wq + w0 + wi + 64u);"),
        ("q4nc_scale", 1, "sb2 = scales[s0 + ((wi + 32u) >> 4)];", "sb2 = scales[s0 + (wi >> 4)];"),
        ("q4t_index", 2, "return (t << 7) + 4u * (k & 31u) + (k >> 5);", "return (t << 7) + 4u * (k >> 5) + (k & 31u);"),
        ("q4t_next", 2, "if (t + 1u < nt) P2 = ", "if (t + 2u < nt) P2 = "),
        ("q4t_group", 2, "const u32 c = ((t << 7) + 32u * s + lane) << 3;", "const u32 c = ((t << 7) + 32u * s + (lane ^ 16u)) << 3;"),
        ("q4t_rest", 2, "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {", "for (u32 wi = (nt << 7) + lane + 32u * (nt > 0u); wi < nwords; wi += 32u) {"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &q4rows_cmd(arm, 3, 96, 0, &Q4_SITES)).is_none_or(|got| got != want), "int4 mutant {tag} survived");
    }
}

fn ctr_cmd(arm: u32, grid: u32, threads: u32, n: u32, work: u32) -> Vec<u8> {
    Cmd::new(b'c').u32(arm).u32(grid).u32(threads).u32(n).u32(work).0
}

/// K3b: every counter kernel runs its hops to the end on the host grid, at
/// grids with and without a short last group, over two launches whose second
/// starts from the first's counters (the epoch). A hop that moved a target
/// would hang and raise the abort flag; a lost arrival would leave a count.
#[test]
fn the_counter_kernels_count_every_hop() {
    for (k, &(name, _)) in pers::CTR_ARMS.iter().enumerate() {
        for &(grid, threads) in &[(12u32, 64u32), (5, 32), (17, 96)] {
            let n = 4u32;
            let got = u32s(&run(persistent(), &ctr_cmd(k as u32, grid, threads, n, 6)));
            let want: Vec<u32> = pers::ctr_expected(name, grid, n).iter().map(|v| 2 * v).collect();
            assert_eq!(got[..got.len() - 1], want[..], "{name}, grid {grid}x{threads}: counts after two launches");
            assert_eq!(got[got.len() - 1], 0, "{name}, grid {grid}x{threads}: abort flag");
        }
    }
}

#[test]
fn counter_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_ctr.cu";
    let cases: [(&str, u32, &str, &str); 4] = [
        ("ctr8_signal", 3, "ctr_red_release(ctr + CTR_LINE * p);", "ctr_red_release(ctr + CTR_LINE * q);"),
        ("rel_target", 2, "            ctr_red_release(ctr);\n            ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);", "            ctr_red_release(ctr);\n            ok = ctr_spin_tight(ctr, (base + i + 2u) * gridDim.x, abort_flag);"),
        ("tight_arrival", 1, "            __threadfence();\n            atomicAdd(ctr, 1u);\n            ok = ctr_spin_tight", "            __threadfence();\n            ok = ctr_spin_tight"),
        ("split_target", 4, "if (threadIdx.x == 0u) ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);", "if (threadIdx.x == 0u) ok = ctr_spin_tight(ctr, (base + i + 2u) * gridDim.x, abort_flag);"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let name = pers::CTR_ARMS[arm as usize].0;
        let want: Vec<u32> = pers::ctr_expected(name, 12, 4).iter().map(|v| 2 * v).chain([0]).collect();
        let killed = run_mutant(&b, &ctr_cmd(arm, 12, 64, 4, 6)).is_none_or(|o| u32s(&o) != want);
        assert!(killed, "counter mutant {tag} survived");
    }
}
