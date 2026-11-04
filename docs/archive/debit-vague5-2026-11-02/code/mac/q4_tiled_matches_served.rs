// DRAFT, never compiled or run. Wave 5 deep dive, lead "mac", 2026-11-02, revised the same day.
//
// Proposed as `llvq-metal/tests/q4_tiled_matches_served.rs`, the file that
// `llvq-llm/kernels/tv_q4_h.metal:162` already names and that does not exist.
// It reuses the fixture and the reference of `llvq-metal/tests/q4_matches_host.rs`
// (copy them, or move them to a shared `tests/common/q4.rs`).
//
// Claims, each an equality of every f32, never a tolerance:
//   1. where the served kernel runs (d_in <= 8192), the tiled kernel returns its
//      bits at tiles 256, 2048 and 8192;
//   2. past the wall (9728, 12288, 17408: the down_proj of the 4B, 8B, 14B), the
//      tiled kernel returns the bits of the host reference that takes its
//      weights from `Int4Matrix::to_f32` and its order from the CUDA original;
//   3. the answer does not depend on Metal's fast math, which the shipped path
//      leaves ON (`fused_metal.rs:397` compiles with `None` options);
//   4. sixteen repetitions at many tiles and many threadgroups return the same
//      bits, which is what a missing barrier would break.
// llvq-tetra checked claim 2 only against the dense reconstruction, at 2.90e-07
// relative (docs/mesures/hf-metal-m2-4b-2026-09-30.txt).
//
// The claims run with fast math ON (`Kernel::new`), the shipped compile. Claim 3
// adds the exact compile. A bench compiled with fast math off over-predicted in
// situ (commit 31a5acb), and a gate compiled only that way would judge a kernel
// the shipped path never builds.
//
// Race mutants. The untiled kernel's barrier mutant "failed five of the seven
// GPU tests and passed two" (q4_matches_host.rs header): a race is a fact about
// this machine, not a guarantee. The rotation's dropped per-row offset lived up
// to 40 rows and died from 48 on this M3 Max (commit a10c92c): below the
// machine's concurrency the threadgroups do not overlap. So the fixtures here
// have 320 to 640 threadgroups of 8 rows and up to 16 tiles, and claim 4
// repeats.
//
// Mutants to plant in the TILED kernel only, partitioned at the next entry
// point (`kernel void tv_q4_metal(` follows it in the file), with a site count
// asserted, because a split on the name alone hits both kernels and voids the
// mutant (hf-metal-m2 journal, controls):
//   - `xs[c - c0 + k]` -> `xs[c + k - 8]`            (tile origin wrong)
//   - `min(c0 + tile_cols, d_in)` -> `c0 + tile_cols` (short last tile overrun)
//   - the FIRST `threadgroup_barrier` removed (tv_q4_h.metal:202): a fast
//     simdgroup refills `xs` while a slow one still reads the previous tile.
//     New with the tiled kernel. A race: killed if any of 16 repetitions of
//     claim 4 fails, kill rate recorded in the mutation journal
//   - the SECOND `threadgroup_barrier` removed (:206): readers race the fill.
//     A race: same rule
//   - lane stride `32u` -> `31u` in the tiled loop
//   - `(c0 >> 3) + lane` -> `lane`                    (every tile restarts at word 0)
// Each must fail at least one assertion below. A race mutant that never fails
// in 16 repetitions widens the fixtures; it is not declared equivalent.
//
// Do NOT plant "buffer 7 unbound" or "tile 0" here: either makes `c0 += 0`
// loop forever, a GPU hang on the operator's machine. Those two are guarded
// structurally by the host (MetalTuning::validate and the unconditional bind)
// and tested through the binding in q4_metal_candle.rs.

#![cfg(target_os = "macos")]

use llvq_artifact::{Int4Matrix, INT4G128_BITS, INT4G128_GROUP};
use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, Kernel};

const SOURCE: &str = include_str!("../../llvq-llm/kernels/tv_q4_h.metal");
const LANES: usize = 32;
const GROUP: usize = 256;
const LIMIT: usize = 32_768;
const REPEATS: usize = 16;

// fixture(seed, d_out, d_in) and reference(m, x): verbatim from q4_matches_host.rs.
// (Omitted here; the draft does not duplicate 90 lines it would copy.) The
// activation must be f16-representable, as the model's is: the int4 arm widens
// an f16 activation on entry (fused_metal.rs:997-1003).
fn fixture(_seed: u64, _d_out: usize, _d_in: usize) -> (Int4Matrix, Vec<f32>) {
    unimplemented!("copy from q4_matches_host.rs::fixture")
}
fn reference(_m: &Int4Matrix, _x: &[f32]) -> Vec<f32> {
    unimplemented!("copy from q4_matches_host.rs::reference")
}

fn words(m: &Int4Matrix) -> Vec<u32> {
    m.packed.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `Kernel::new` is Metal's default compile, fast math ON, which is what
/// candle's `None` options give the shipped path. `new_exact` turns it off.
fn compile(name: &str, fast_math: bool) -> Kernel {
    match fast_math {
        true => Kernel::new(SOURCE, name),
        false => Kernel::new_exact(SOURCE, name),
    }
    .expect("compiles")
}

/// `tv_q4_metal`, the whole activation staged.
fn served(m: &Int4Matrix, x: &[f32], fast_math: bool) -> Vec<f32> {
    assert!(m.d_in * 4 <= LIMIT, "the served kernel only takes d_in <= 8192");
    let k = compile("tv_q4_metal", fast_math);
    let (b_wq, b_sc, b_bi, b_x) = (k.buffer(&words(m)), k.buffer(&m.scales), k.buffer(&m.biases), k.buffer(x));
    let b_y = k.empty::<f32>(m.d_out);
    let (d_in, gpr) = (m.d_in as u32, m.groups_per_row() as u32);
    k.dispatch((m.d_out * LANES) as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_wq), 0);
        enc.set_buffer(1, Some(&b_sc), 0);
        enc.set_buffer(2, Some(&b_bi), 0);
        enc.set_buffer(3, Some(&b_x), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        enc.set_bytes(5, 4, &d_in as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &gpr as *const u32 as *const std::ffi::c_void);
        enc.set_threadgroup_memory_length(0, (m.d_in * 4) as u64);
    });
    unsafe { k.read::<f32>(&b_y, m.d_out) }
}

/// `tv_q4_metal_tiled` at `tile` columns a tile, on an already compiled kernel
/// so repetitions do not recompile.
fn tiled_on(k: &Kernel, m: &Int4Matrix, x: &[f32], tile: usize) -> Vec<f32> {
    assert!(tile > 0 && tile % 256 == 0 && tile * 4 <= LIMIT, "the host refuses tile {tile}");
    let (b_wq, b_sc, b_bi, b_x) = (k.buffer(&words(m)), k.buffer(&m.scales), k.buffer(&m.biases), k.buffer(x));
    let b_y = k.empty::<f32>(m.d_out);
    let (d_in, gpr, tc) = (m.d_in as u32, m.groups_per_row() as u32, tile as u32);
    k.dispatch((m.d_out * LANES) as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_wq), 0);
        enc.set_buffer(1, Some(&b_sc), 0);
        enc.set_buffer(2, Some(&b_bi), 0);
        enc.set_buffer(3, Some(&b_x), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        enc.set_bytes(5, 4, &d_in as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &gpr as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &tc as *const u32 as *const std::ffi::c_void);
        // The tile, not d_in: the point of this kernel.
        enc.set_threadgroup_memory_length(0, (tile * 4) as u64);
    });
    unsafe { k.read::<f32>(&b_y, m.d_out) }
}

fn tiled(m: &Int4Matrix, x: &[f32], tile: usize, fast_math: bool) -> Vec<f32> {
    tiled_on(&compile("tv_q4_metal_tiled", fast_math), m, x, tile)
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

/// Claim 1: the shapes the served kernel can take, v_proj and o_proj of the
/// three sealed sizes.
#[test]
fn tiled_equals_served_where_both_run() {
    for (seed, d_out, d_in) in [(1u64, 1024usize, 2560usize), (2, 2560, 4096), (3, 5120, 5120)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = served(&m, &x, true);
        for tile in [256usize, 2048, 8192] {
            let got = tiled(&m, &x, tile, true);
            assert_eq!(first_difference(&want, &got), None, "{d_out}x{d_in} at tile {tile}");
        }
    }
}

/// Claim 2: past the wall, against the host reference.
#[test]
fn tiled_equals_the_host_reference_past_the_wall() {
    for (seed, d_out, d_in) in [(4u64, 2560usize, 9728usize), (5, 4096, 12288), (6, 5120, 17408)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = reference(&m, &x);
        for tile in [2048usize, 8192] {
            let got = tiled(&m, &x, tile, true);
            assert_eq!(first_difference(&want, &got), None, "{d_out}x{d_in} at tile {tile}");
        }
    }
}

/// The reference is the served kernel's order, so it must also equal the
/// served kernel where that one runs. Without this, claim 2 rests on a
/// reference nobody checked against the shipped arithmetic.
#[test]
fn the_reference_equals_the_served_kernel() {
    let (m, x) = fixture(7, 2560, 4096);
    assert_eq!(first_difference(&reference(&m, &x), &served(&m, &x, true)), None);
}

/// Claim 3: the explicit `fma` and `contract(off)` carry the arithmetic, not
/// the compile option. The twin of q4_matches_host.rs's
/// `the_same_source_gives_the_same_numbers_with_fast_math_either_way`.
#[test]
fn the_tiled_kernel_gives_the_same_numbers_with_fast_math_either_way() {
    for (seed, d_out, d_in, tile) in [(9u64, 2560usize, 9728usize, 2048usize), (10, 2560, 4096, 256)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let fast = tiled(&m, &x, tile, true);
        let exact = tiled(&m, &x, tile, false);
        assert_eq!(first_difference(&fast, &exact), None, "{d_out}x{d_in} tile {tile}: fast math moved the answer");
        assert_eq!(first_difference(&fast, &reference(&m, &x)), None, "{d_out}x{d_in} tile {tile}: and the reference");
    }
}

/// Every tile walks the same words in the same order, so every tile gives the
/// same bits at the widest shape too.
#[test]
fn every_tile_gives_the_same_bits_at_17408() {
    let (m, x) = fixture(8, 5120, 17408);
    let base = tiled(&m, &x, 8192, true);
    for tile in [256usize, 1024, 2048, 4096] {
        assert_eq!(first_difference(&base, &tiled(&m, &x, tile, true)), None, "tile {tile}");
    }
}

/// Claim 4, the barriers. Many tiles (16 at 2560x4096 with tile 256, 9 at
/// 5120x17408 with tile 2048) and many threadgroups (320 and 640), sixteen
/// times each. A missing barrier is a race, so it shows up as a repetition
/// that differs from the reference, not as a stable wrong answer.
#[test]
fn the_tiles_do_not_race_under_repetition() {
    let k = compile("tv_q4_metal_tiled", true);
    for (seed, d_out, d_in, tile) in [(11u64, 2560usize, 4096usize, 256usize), (12, 5120, 17408, 2048)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = reference(&m, &x);
        let mut failures = Vec::new();
        for rep in 0..REPEATS {
            if let Some(d) = first_difference(&want, &tiled_on(&k, &m, &x, tile)) {
                failures.push((rep, d));
            }
        }
        assert!(failures.is_empty(), "{d_out}x{d_in} tile {tile}: {} of {REPEATS} repetitions differ, first {:?}",
                failures.len(), failures.first());
    }
}

// Silence unused imports in the draft.
#[allow(dead_code)]
fn _uses() -> (u8, usize, u16) {
    let _ = SplitMix64::new(0);
    (INT4G128_BITS, INT4G128_GROUP, f16_bits(0.0))
}
