//! `tv_q4_metal_tiled`, the int4 matvec past the 32 KB wall, and the
//! prefetching twins of both int4 kernels, against the served kernel and the
//! host reference. The file `llvq-llm/kernels/tv_q4_h.metal` names.
//!
//! Every sealed file's int4 `down_proj` is past the wall: d_in 9,728 at 4B,
//! 12,288 at 8B, 17,408 at 14B. Until this lot the Metal loader refused them,
//! so no sealed file ran fused on a Mac.
//!
//! Claims, each an equality of every f32, never a tolerance:
//!   1. where the served kernel runs (d_in <= 8,192), the tiled kernel
//!      returns its bits at tiles 256, 2,048 and 8,192;
//!   2. past the wall, the tiled kernel returns the bits of the host
//!      reference, which takes its weights from `Int4Matrix::to_f32` and its
//!      order from the CUDA original; and the reference equals the served
//!      kernel where that one runs;
//!   3. the answer does not depend on Metal's fast math, which the shipped
//!      path leaves on;
//!   4. sixteen repetitions at many tiles and many threadgroups return the
//!      same bits, which is what a missing barrier would break;
//!   5. the prefetching twins (`tv_q4_pf.metal`, the port of 4E's
//!      `pers_q4_row<PREFETCH>`) return the served kernels' bits.
//!
//! Mutation run (2026-11-06, journal `docs/mesures/metal-nuit-2026-11-06.txt`):
//! the mutants are planted in the kernel they name, partitioned at the next
//! entry point, with the site count asserted.
//!
//! Runs on any Mac, needs no model and no artifact.

#![cfg(target_os = "macos")]

use llvq_artifact::{Int4Matrix, INT4G128_BITS, INT4G128_GROUP};
use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, Kernel};

const SOURCE: &str = include_str!("../../llvq-llm/kernels/tv_q4_h.metal");
const SOURCE_PF: &str = concat!(
    include_str!("../../llvq-llm/kernels/tv_q4_h.metal"),
    include_str!("../../llvq-llm/kernels/tv_q4_pf.metal")
);
const LANES: usize = 32;
const GROUP: usize = 256;
const LIMIT: usize = 32_768;
const REPEATS: usize = 16;

/// `q4_matches_host.rs::fixture`, verbatim: random nibbles, one group in
/// seven ungraded, scales spread over two decades. The activation is drawn
/// f16-representable, as the model's is (the int4 arm widens an f16
/// activation on entry).
fn fixture(seed: u64, d_out: usize, d_in: usize) -> (Int4Matrix, Vec<f32>) {
    assert!(d_in.is_multiple_of(INT4G128_GROUP), "the reader refuses a short group");
    let mut rng = SplitMix64::new(seed);
    let gpr = d_in / INT4G128_GROUP;
    let mut packed: Vec<u8> = (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect();
    let mut scales = Vec::with_capacity(d_out * gpr);
    let mut biases = Vec::with_capacity(d_out * gpr);
    for row in 0..d_out {
        for g in 0..gpr {
            let ungraded = (row * gpr + g).is_multiple_of(7);
            let s = match ungraded {
                true => 1.0,
                false => 0.002 + 0.08 * rng.next_f64() as f32,
            };
            scales.push(f16_bits(s));
            biases.push(f16_bits(rng.next_gaussian() as f32 * 0.05));
            if ungraded {
                let lo = (row * d_in + g * INT4G128_GROUP) / 2;
                packed[lo..lo + INT4G128_GROUP / 2].fill(0);
            }
        }
    }
    let m = Int4Matrix {
        name: "model.layers.12.mlp.down_proj.weight".into(),
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed,
        scales,
        biases,
    };
    let x = (0..d_in)
        .map(|_| llvq_metal::f16_to_f64(f16_bits(rng.next_gaussian() as f32)) as f32)
        .collect();
    (m, x)
}

/// `q4_matches_host.rs::reference`, verbatim: lane `l` takes words `l, l + 32,
/// ...`, eight columns a word with `mul_add`, then the shuffle-xor butterfly.
fn reference(m: &Int4Matrix, x: &[f32]) -> Vec<f32> {
    let w = m.to_f32();
    let nwords = m.d_in / 8;
    let mut y = vec![0f32; m.d_out];
    for (row, out) in y.iter_mut().enumerate() {
        let mut lanes = [0f32; LANES];
        for (lane, acc) in lanes.iter_mut().enumerate() {
            let mut wi = lane;
            while wi < nwords {
                let c = wi * 8;
                let wr = &w[row * m.d_in + c..row * m.d_in + c + 8];
                for (wv, xv) in wr.iter().zip(&x[c..c + 8]) {
                    *acc = wv.mul_add(*xv, *acc);
                }
                wi += LANES;
            }
        }
        for k in [16usize, 8, 4, 2, 1] {
            let mut next = [0f32; LANES];
            for (l, n) in next.iter_mut().enumerate() {
                *n = lanes[l] + lanes[l ^ k];
            }
            lanes = next;
        }
        *out = lanes[0];
    }
    y
}

fn words(m: &Int4Matrix) -> Vec<u32> {
    m.packed.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `Kernel::new` is Metal's default compile, fast math on, which is what
/// candle's `None` options give the shipped path. `new_exact` turns it off.
fn compile(source: &str, name: &str, fast_math: bool) -> Kernel {
    match fast_math {
        true => Kernel::new(source, name),
        false => Kernel::new_exact(source, name),
    }
    .expect("compiles")
}

/// An untiled kernel (`tv_q4_metal` or its twin): the whole activation staged.
fn untiled_on(k: &Kernel, m: &Int4Matrix, x: &[f32]) -> Vec<f32> {
    assert!(m.d_in * 4 <= LIMIT, "the untiled kernels only take d_in <= 8192");
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

fn served(m: &Int4Matrix, x: &[f32], fast_math: bool) -> Vec<f32> {
    untiled_on(&compile(SOURCE, "tv_q4_metal", fast_math), m, x)
}

/// A tiled kernel at `tile` columns a tile, on an already compiled kernel so
/// repetitions do not recompile.
fn tiled_on(k: &Kernel, m: &Int4Matrix, x: &[f32], tile: usize) -> Vec<f32> {
    // The host refuses any other tile; a zero would loop forever on the GPU.
    assert!(tile > 0 && tile.is_multiple_of(256) && tile * 4 <= LIMIT, "tile {tile}");
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
        enc.set_threadgroup_memory_length(0, (tile * 4) as u64);
    });
    unsafe { k.read::<f32>(&b_y, m.d_out) }
}

fn tiled(m: &Int4Matrix, x: &[f32], tile: usize, fast_math: bool) -> Vec<f32> {
    tiled_on(&compile(SOURCE, "tv_q4_metal_tiled", fast_math), m, x, tile)
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

fn nonzero(v: &[f32]) {
    assert!(v.iter().any(|y| *y != 0.0), "zeros would pass every equality");
}

/// Claim 1: the shapes the served kernel can take, v_proj and o_proj of the
/// three sealed sizes.
#[test]
fn tiled_equals_served_where_both_run() {
    for (seed, d_out, d_in) in [(1u64, 1024usize, 2560usize), (2, 2560, 4096), (3, 1024, 5120)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = served(&m, &x, true);
        nonzero(&want);
        for tile in [256usize, 2048, 8192] {
            let got = tiled(&m, &x, tile, true);
            assert_eq!(first_difference(&want, &got), None, "{d_out}x{d_in} at tile {tile}");
        }
    }
}

/// Claim 2: past the wall, against the host reference, at the down_proj
/// widths of the three sealed sizes.
#[test]
fn tiled_equals_the_host_reference_past_the_wall() {
    for (seed, d_out, d_in) in [(4u64, 2560usize, 9728usize), (5, 1024, 12288), (6, 512, 17408)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = reference(&m, &x);
        nonzero(&want);
        for tile in [2048usize, 8192] {
            let got = tiled(&m, &x, tile, true);
            assert_eq!(first_difference(&want, &got), None, "{d_out}x{d_in} at tile {tile}");
        }
    }
}

/// Claim 2's anchor: the reference is the served kernel's order, so it equals
/// the served kernel where that one runs.
#[test]
fn the_reference_equals_the_served_kernel() {
    let (m, x) = fixture(7, 2560, 4096);
    assert_eq!(first_difference(&reference(&m, &x), &served(&m, &x, true)), None);
}

/// Claim 3: the explicit `fma` and `contract(off)` carry the arithmetic, not
/// the compile option.
#[test]
fn the_tiled_kernel_gives_the_same_numbers_with_fast_math_either_way() {
    for (seed, d_out, d_in, tile) in [(9u64, 2560usize, 9728usize, 2048usize), (10, 2560, 4096, 256)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let fast = tiled(&m, &x, tile, true);
        let exact = tiled(&m, &x, tile, false);
        assert_eq!(first_difference(&fast, &exact), None, "{d_out}x{d_in} tile {tile}: fast math moved it");
        assert_eq!(first_difference(&fast, &reference(&m, &x)), None, "{d_out}x{d_in} tile {tile}: reference");
    }
}

/// Every tile walks the same words in the same order.
#[test]
fn every_tile_gives_the_same_bits_at_17408() {
    let (m, x) = fixture(8, 1024, 17408);
    let base = tiled(&m, &x, 8192, true);
    for tile in [256usize, 1024, 2048, 4096] {
        assert_eq!(first_difference(&base, &tiled(&m, &x, tile, true)), None, "tile {tile}");
    }
}

/// Claim 4, the barriers: many tiles (16 at 2560x4096 with tile 256, 5 at
/// 2560x9728 with tile 2048) and many threadgroups (320), sixteen times each.
#[test]
fn the_tiles_do_not_race_under_repetition() {
    for (name, source) in [("tv_q4_metal_tiled", SOURCE), ("tv_q4_metal_tiled_p", SOURCE_PF)] {
        let k = compile(source, name, true);
        for (seed, d_out, d_in, tile) in [(11u64, 2560usize, 4096usize, 256usize), (12, 2560, 9728, 2048)] {
            let (m, x) = fixture(seed, d_out, d_in);
            let want = reference(&m, &x);
            let mut failures = Vec::new();
            for rep in 0..REPEATS {
                if let Some(d) = first_difference(&want, &tiled_on(&k, &m, &x, tile)) {
                    failures.push((rep, d));
                }
            }
            assert!(
                failures.is_empty(),
                "{name} {d_out}x{d_in} tile {tile}: {} of {REPEATS} repetitions differ, first {:?}",
                failures.len(),
                failures.first()
            );
        }
    }
}

/// Claim 5, untiled: `tv_q4_metal_p` returns `tv_q4_metal`'s bits at the
/// v_proj and o_proj widths, and at widths where some lanes take no word or
/// one more word than others.
#[test]
fn the_prefetching_untiled_twin_returns_the_served_bits() {
    let k = compile(SOURCE_PF, "tv_q4_metal_p", true);
    for (seed, d_out, d_in) in [(20u64, 1024usize, 2560usize), (21, 2560, 4096), (22, 64, 128), (23, 256, 384), (24, 512, 8192)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = served(&m, &x, true);
        nonzero(&want);
        for rep in 0..4 {
            assert_eq!(first_difference(&want, &untiled_on(&k, &m, &x)), None, "{d_out}x{d_in} rep {rep}");
        }
    }
}

/// Claim 5, tiled: `tv_q4_metal_tiled_p` returns `tv_q4_metal_tiled`'s bits,
/// hence the reference's, at the three down_proj widths and three tiles.
#[test]
fn the_prefetching_tiled_twin_returns_the_served_bits() {
    let k = compile(SOURCE_PF, "tv_q4_metal_tiled_p", true);
    for (seed, d_out, d_in) in [(30u64, 2560usize, 9728usize), (31, 1024, 12288), (32, 512, 17408), (33, 256, 4096)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = reference(&m, &x);
        nonzero(&want);
        for tile in [256usize, 2048, 8192] {
            assert_eq!(first_difference(&want, &tiled_on(&k, &m, &x, tile)), None, "{d_out}x{d_in} tile {tile}");
        }
    }
}
