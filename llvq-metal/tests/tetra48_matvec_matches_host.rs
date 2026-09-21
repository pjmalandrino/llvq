//! The Tetra matvec in MSL, against a host reference that reproduces its
//! summation order exactly.
//!
//! ## Why the reference is written the hard way
//!
//! Floating-point addition is not associative, so "the same matvec" computed
//! in a different order is a different number. A test that summed the blocks
//! in a plain loop would have to accept a tolerance, and a tolerance is what
//! lets a real defect through: a transposed lane partition, a tile boundary
//! off by one, a block counted twice all land within any epsilon a 288-wide
//! dot product would need.
//!
//! So the reference below reproduces the kernel: 32 lanes, each taking blocks
//! `lane, lane + 32, ...` within each tile, then the same `__shfl_xor`
//! butterfly the CUDA `warp_sum` performs and the MSL `warp_sum` mirrors, then
//! the same multiply-then-add tail. The assertion is equality.
//!
//! ## The mutation run
//!
//! Nine mutants of the shader, seven killed: the butterfly stopping at 8
//! lanes, the lane stride at 31 so lanes overlap, the tile starting one block
//! late, `rscale` dropped, the tail reading row 0 for every row, the gain bit
//! forced to zero, `invnorm` dropped, and the tile staging the wrong slice of
//! the activation.
//!
//! TWO DID NOT DIE, and neither is papered over.
//!
//! Widening the tile to `TILE + 1` survives. It double-counts one block and
//! reads 24 floats past the threadgroup allocation, and the answer does not
//! move: Apple returns zero for a threadgroup read past the end, so the
//! doubled block contributes nothing on its first pass. That is hardware
//! behaviour standing in for a bounds check, not coverage. Starting the tile
//! one block LATE, which does not overflow, is killed, so the tile boundary
//! itself is under test.
//!
//! Removing the first `threadgroup_barrier` survives, and a functional test
//! cannot honestly be expected to kill it. It is a RACE: the next tile's fill
//! may overwrite a straggler's staging, which depends on scheduling and not on
//! the inputs. The barrier's necessity is argued from the memory model, the
//! way `matvec.cu` argues it, and is not claimed to be measured here.
//!
//! ## What it does not cover
//!
//! Nothing runs through candle here. The shipped Metal adapter must reach this
//! kernel through `CustomOp1::metal_fwd` and objc2-metal; this file drives it
//! through `llvq-metal`'s own metal-rs host layer, which proves the KERNEL and
//! says nothing about the binding.

#![cfg(target_os = "macos")]

use llvq_artifact::tetra48::{transcode_tetra48, TETRA48_SHELLS};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use llvq_metal::Kernel;
use llvq_search::index::N13;
use llvq_search::tetra::Tetra;

const SOURCE: &str = include_str!("../../llvq-llm/kernels/llvq_tetra48.metal");

/// Must match `LLVQ_TILE_BLOCKS` in the shader.
const TILE: usize = 64;
/// The kernel gives one SIMD-group to a row; Apple's is 32 lanes wide.
const LANES: usize = 32;
/// Threads a threadgroup, so 8 rows each, as the CUDA host uses.
const GROUP: usize = 256;

/// `1/sqrt(16 m)`, entry 0 the origin. The same three lines
/// `bin/planesbench` and `bin/f1rankfloor` build, because it is the same
/// kernel.
fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// The coordinate order the quads come out in.
const ORDER: [usize; DIM] = [
    0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23,
];

struct Fixture {
    d_out: usize,
    nblocks: usize,
    tail_w: usize,
    stream: llvq_artifact::tetra48::Tetra48Blocks,
    gscale: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<HalfF16>,
    x: Vec<f32>,
}

/// A local f16, so the test does not take a dependency for two conversions.
///
/// Both directions are the IEEE ones. Widening is exact, which is what the
/// kernel's `float(half)` does and what the CUDA `h2f` does in software.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct HalfF16(u16);

impl HalfF16 {
    fn from_f32(v: f32) -> Self {
        // Round to nearest even, via the bit layout. Only finite values of
        // modest magnitude reach this, which the fixture guarantees.
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = b & 0x7f_ffff;
        if exp <= 0 {
            return HalfF16(sign);
        }
        if exp >= 31 {
            exp = 30;
            return HalfF16(sign | ((exp as u16) << 10) | 0x3ff);
        }
        let mut m = (mant >> 13) as u16;
        // Round to nearest even on the dropped 13 bits.
        let rest = mant & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                m = 0;
                exp += 1;
            }
        }
        HalfF16(sign | ((exp as u16) << 10) | m)
    }

    fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign);
            }
            // Subnormal: normalise.
            let shift = mant.leading_zeros() - 21;
            let e = 127 - 15 - shift;
            let m = (mant << (shift + 1)) & 0x3ff;
            return f32::from_bits(sign | (e << 23) | (m << 13));
        }
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

fn fixture(seed: u64, d_out: usize, nblocks: usize, tail_w: usize) -> Fixture {
    let mut rng = SplitMix64::new(seed);
    let n = d_out * nblocks;
    let indices: Vec<u64> = (0..n)
        .map(|_| match rng.next().is_multiple_of(6) {
            true => 0,
            false => 1 + rng.next() % (N13.min(1u64 << 47) - 1),
        })
        .collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let stream = transcode_tetra48(&indices, &gains, d_out, nblocks).expect("transcodes");
    Fixture {
        d_out,
        nblocks,
        tail_w,
        stream,
        // Two distinct centroids, so a swapped gain bit moves the answer.
        gscale: [0.625, 1.375],
        rscale: (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect(),
        tail: (0..d_out * tail_w)
            .map(|_| HalfF16::from_f32(rng.next_gaussian() as f32))
            .collect(),
        x: (0..nblocks * DIM + tail_w).map(|_| rng.next_gaussian() as f32).collect(),
    }
}

/// The kernel's arithmetic, on the host, in the kernel's order.
fn reference(f: &Fixture) -> Vec<f32> {
    let tetra = Tetra::new();
    let invnorm = invnorm_table();
    let ntiles = f.nblocks.div_ceil(TILE);
    let mut y = vec![0f32; f.d_out];

    for (row, out) in y.iter_mut().enumerate() {
        let mut lanes = [0f32; LANES];
        for t in 0..ntiles {
            let jlo = t * TILE;
            let jhi = (jlo + TILE).min(f.nblocks);
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let mut j = jlo + lane;
                while j < jhi {
                    let (point, gain) = f.stream.decode_block(&tetra, row, j);
                    let n2: i32 = point.iter().map(|&v| v * v).sum();
                    let m = (((n2 as u32) >> 4) & (TETRA48_SHELLS as u32 - 1)) as usize;
                    // The quad order, with `fma`, exactly as the shader.
                    let xb = &f.x[j * DIM..(j + 1) * DIM];
                    let mut d = 0f32;
                    for i in 0..6 {
                        for k in 0..4 {
                            let idx = ORDER[4 * i + k];
                            d = (point[idx] as f32).mul_add(xb[idx], d);
                        }
                    }
                    *acc += d * f.gscale[gain as usize] * invnorm[m];
                    j += LANES;
                }
            }
        }
        // The `__shfl_xor` butterfly, lane for lane.
        for k in [16usize, 8, 4, 2, 1] {
            let mut next = [0f32; LANES];
            for (l, n) in next.iter_mut().enumerate() {
                *n = lanes[l] + lanes[l ^ k];
            }
            lanes = next;
        }
        // Multiply-then-add, not `fma`: the epilogue's own association.
        let mut tv = 0f32;
        let xt = &f.x[f.nblocks * DIM..];
        // `mul_add` at both sites, because the CUDA twin contracts there
        // and the shader now does too. See the shader's epilogue comment.
        for (i, xi) in xt.iter().enumerate().take(f.tail_w) {
            tv = f.tail[row * f.tail_w + i].to_f32().mul_add(*xi, tv);
        }
        *out = lanes[0].mul_add(f.rscale[row], tv);
    }
    y
}

fn run_on_metal(f: &Fixture) -> Vec<f32> {
    run_with(f, true)
}

/// `exact` chooses whether Metal's fast math is off. The answer must not
/// depend on it; `the_same_source_gives_the_same_numbers_with_fast_math_either_way`
/// is what says so.
fn run_with(f: &Fixture, exact: bool) -> Vec<f32> {
    run_named(f, exact, "tv_tetra48_metal")
}

/// The same, naming the kernel, so the threadgroup-pinned variant is judged
/// by this file's reference rather than by the kernel it is meant to replace.
fn run_named(f: &Fixture, exact: bool, name: &str) -> Vec<f32> {
    let pinned = name.ends_with("_tg");
    let k = match exact {
        true => Kernel::new_exact(SOURCE, name),
        false => Kernel::new(SOURCE, name),
    }
    .expect("the matvec compiles");
    let table = RankTable::build();
    let tr = Trellis::new();
    let invnorm = invnorm_table();

    let b_words = k.buffer(&f.stream.data);
    let b_rows = k.buffer(&table.rows);
    let b_pref = k.buffer(&prefix_bytes(&tr));
    let b_bran = k.buffer(&branch_words(&tr));
    let b_suff = k.buffer(&suffix_bytes(&tr));
    let b_gs = k.buffer(&f.gscale);
    let b_iv = k.buffer(&invnorm);
    let b_rs = k.buffer(&f.rscale);
    // Metal refuses a zero-length buffer and hands back a null pointer, the
    // same wall cudarc puts up. A `d_in` that is a multiple of 24 has no tail,
    // so it gets a one-element dummy the kernel never reads: `tail_w == 0`
    // makes the epilogue's loop empty. The shipped adapter owes this too.
    let dummy = [HalfF16(0)];
    let b_tail = k.buffer(if f.tail.is_empty() { &dummy } else { &f.tail[..] });
    let b_x = k.buffer(&f.x);
    let b_y = k.empty::<f32>(f.d_out);

    let stride = f.stream.stride_u32 as u32;
    let nb = f.nblocks as u32;
    let tw = f.tail_w as u32;
    let tg_bytes = (TILE * DIM * 4) as u64;

    k.dispatch((f.d_out * LANES) as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_words), 0);
        enc.set_bytes(1, 4, &stride as *const u32 as *const std::ffi::c_void);
        enc.set_buffer(2, Some(&b_rows), 0);
        enc.set_buffer(3, Some(&b_pref), 0);
        enc.set_buffer(4, Some(&b_bran), 0);
        enc.set_buffer(5, Some(&b_suff), 0);
        enc.set_buffer(6, Some(&b_gs), 0);
        enc.set_buffer(7, Some(&b_iv), 0);
        enc.set_buffer(8, Some(&b_rs), 0);
        enc.set_buffer(9, Some(&b_tail), 0);
        enc.set_buffer(10, Some(&b_x), 0);
        enc.set_buffer(11, Some(&b_y), 0);
        enc.set_bytes(12, 4, &nb as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(13, 4, &tw as *const u32 as *const std::ffi::c_void);
        enc.set_threadgroup_memory_length(0, tg_bytes);
        if pinned {
            // 4,096 u32 + 1,024 u16 + 128 + 128 = 18,688 B, which with the
            // tile's 6,144 fits Apple's 32,768.
            enc.set_threadgroup_memory_length(1, 4096 * 4);
            enc.set_threadgroup_memory_length(2, 1024 * 2);
            enc.set_threadgroup_memory_length(3, 128);
            enc.set_threadgroup_memory_length(4, 128);
        }
    });

    let out = unsafe { std::slice::from_raw_parts(b_y.contents() as *const f32, f.d_out) };
    out.to_vec()
}

/// One tile exactly: `nblocks == TILE`, so the tile loop runs once and every
/// lane takes two blocks.
#[test]
fn the_matvec_matches_the_host_on_one_tile() {
    let f = fixture(0x7E_48A0, 64, TILE, 8);
    let got = run_on_metal(&f);
    let want = reference(&f);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// Several tiles, and a last one that is PARTIAL: `nblocks` is not a multiple
/// of `TILE`, so `jhi` clamps and some lanes take no block in the last pass.
#[test]
fn the_matvec_matches_the_host_across_a_partial_tile() {
    let f = fixture(0x7E_48A1, 32, 3 * TILE + 17, 5);
    let got = run_on_metal(&f);
    let want = reference(&f);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// No tail at all, which is the case `d_in % 24 == 0` produces and where an
/// epilogue that read one element too many would show.
#[test]
fn the_matvec_matches_the_host_with_no_tail() {
    let f = fixture(0x7E_48A2, 16, TILE + 1, 0);
    let got = run_on_metal(&f);
    let want = reference(&f);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// The result depends on the activation: a kernel that ignored `x` would pass
/// every equality above if the reference ignored it too.
#[test]
fn a_different_activation_gives_a_different_answer() {
    let mut f = fixture(0x7E_48A3, 16, TILE, 4);
    let a = run_on_metal(&f);
    for v in f.x.iter_mut() {
        *v += 1.0;
    }
    let b = run_on_metal(&f);
    assert!(
        a.iter().zip(&b).any(|(p, q)| p != q),
        "the matvec must read the activation"
    );
    let want = reference(&f);
    for (i, (g, w)) in b.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i} after the shift");
    }
}

/// The pragma, not the compile option, is what carries the arithmetic.
///
/// This matters because the SHIPPED path does not use `new_exact`:
/// `llvq-llm/src/fused_metal.rs` compiles the same source through candle with
/// `None` options, which is Metal's default and has fast math ON. If the
/// arithmetic depended on the option, the gate and the shipped path would be
/// two different kernels and this whole file would be judging the wrong one.
///
/// An audit of 2026-09-21 found `new_exact`'s doc claiming fast math off was
/// required. It is not, once `#pragma clang fp contract(off)` is in the
/// source. Turning it off stays as belt and braces, and this test is what
/// makes the belt checkable.
#[test]
fn the_same_source_gives_the_same_numbers_with_fast_math_either_way() {
    let f = fixture(0x7E_48A4, 32, TILE + 9, 6);
    let exact = run_with(&f, true);
    let fast = run_with(&f, false);
    let want = reference(&f);
    for (i, ((a, b), w)) in exact.iter().zip(&fast).zip(&want).enumerate() {
        assert_eq!(a, b, "row {i}: fast math moved the answer, {a} against {b}");
        assert_eq!(a, w, "row {i}: and neither matches the host reference");
    }
}

/// The threadgroup-pinned variant computes what the device-memory one does.
///
/// It exists to be faster, and a faster kernel that moves a number is worth
/// nothing. The decode is duplicated with a second address space, which is
/// exactly the kind of copy that drifts, so it is judged by the same host
/// reference and demanded EQUAL.
#[test]
fn the_pinned_variant_matches_the_host_too() {
    for (d_out, nblocks, tail_w) in [(64usize, TILE, 8usize), (32, 3 * TILE + 17, 5), (16, TILE + 1, 0)] {
        let f = fixture(0x7E_48B0 + d_out as u64, d_out, nblocks, tail_w);
        for name in ["tv_tetra48_metal_tg", "tv_tetra48_metal_ar", "tv_tetra48_metal_lut"] {
            let got = run_named(&f, true, name);
            let want = reference(&f);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{name} d_out={d_out} nblocks={nblocks}: row {i}, {g} against host {w}");
            }
        }
    }
}
