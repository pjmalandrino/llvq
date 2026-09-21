//! The Metal rotation against the CUDA rotation, bit for bit.
//!
//! ## Where the reference comes from, and why it is not this port
//!
//! Two references, neither written from the shader.
//!
//! The first is the CUDA source itself, EXECUTED. `llvq-cuda/tests/host_rotate.cpp`
//! `#include`s `llvq_rot.cuh` and `rotate.cu` verbatim and runs them as
//! ordinary C++ through `host_shim.h`. This file compiles that driver with
//! `clang++`, feeds it the same bytes the shader gets, and requires the same
//! f32. Nothing is transcribed, so the two sides cannot share a transcription
//! mistake. That is the defect an audit of 2026-09-20 found in the Tetra
//! matvec: both host references had been written by reading the shader, so
//! both carried its bug and the gate passed on a wrong kernel.
//!
//! The second is the mathematical definition. `Q = (Q_odd ⊗ H_m) D` with
//! `H_m[j][j'] = (-1)^popcount(j & j') / sqrt(m)`, applied in f64 by a dense
//! sum with no butterfly in it. It shares no code and no algorithm with either
//! GPU kernel. It is what says the CUDA original is the transform the sealed
//! artifact was quantized in, rather than a transform both GPUs agree on.
//!
//! `llvq-quant`'s `Rotation` is the specification the CUDA side is diffed
//! against in `llvq-cuda/tests/rotation_matches_rust.rs`. This crate does not
//! depend on `llvq-quant`, so the chain here runs through the CUDA text
//! instead of reaching for it directly.
//!
//! ## What is exact and what is a tolerance
//!
//! The CUDA comparison is exact equality on f32. Every operation on both sides
//! is an IEEE add, subtract, multiply or fused multiply-add, correctly rounded
//! on an Apple GPU and under `clang++ -ffp-contract=off`. A difference of one
//! bit is a defect, not noise.
//!
//! The definition comparison is a tolerance, since it runs in f64 and the
//! kernels run in f32. It is sized the way `llvq-cuda/tests/rotation_matches_rust.rs`
//! sizes it: a relative norm error and a worst coordinate, because they fail
//! differently.
//!
//! ## The mutation run
//!
//! Sixteen mutants of the shader, fifteen killed every run. Dropped sign flip,
//! sign bitmap read bit-reversed, Walsh-Hadamard addressed flat, butterfly
//! subtraction reversed, one stage short, `1/sqrt(m)` dropped from each of the
//! two output phases, `Q_odd` transposed, `fma` written out as `a * b + c`,
//! the mix output transposed, `col` padded with 1.0f, `x_off` ignored, the
//! batched input using `n` where `row_stride` belongs, the batched output
//! using `row_stride` where `n` belongs, and the in-loop barrier removed.
//!
//! Two of those are worth naming. `fma` written out as `a * b + c` is killed
//! by the CUDA comparison alone, which is the whole reason that comparison is
//! exact: the CUDA spells `__fmaf_rn` and rounds once, the contraction pragma
//! makes the written form round twice, and 27 % of triples differ. `col`
//! padded with 1.0f survives every other assertion and is killed only by
//! `the_mix_ignores_whatever_pads_the_small_block`, which is why that test
//! exists.
//!
//! THE SIXTEENTH DIES HALF THE TIME, and it is not papered over. Removing the
//! barrier between `rot_load` and the first butterfly is killed in 3 runs out
//! of 6, always by `the_work_split_does_not_move_a_bit`. It is a RACE:
//! whether a thread reads a slot before its owner writes depends on
//! scheduling and not on the inputs. The scratch buffer arrives filled with
//! NaN, which is what makes the race loud when it fires, and sweeping the
//! thread count is what makes it fire at all. A gate that kills half the time
//! is not a gate, so this mutant is counted as a survivor and the barrier's
//! necessity is argued from the memory model. The in-loop barrier is killed
//! in 6 runs out of 6.
//!
//! ## What this file does not cover
//!
//! Nothing runs through candle. The shipped Metal adapter must reach these
//! kernels through `CustomOp1::metal_fwd`; this file drives them through
//! `llvq-metal`'s own metal-rs host layer. It proves the KERNEL and says
//! nothing about the binding.
//!
//! Performance is not measured. The staging area is a device buffer rather
//! than threadgroup memory, for the reason the shader's header gives, and
//! whether that costs anything on Apple is a measurement nobody has taken.

#![cfg(target_os = "macos")]

use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, f16_to_f64, Kernel};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::process::{Command, Stdio};
use std::rc::Rc;

const SOURCE: &str = include_str!("../../llvq-llm/kernels/llvq_rot.metal");
const CUDA_HEADER: &str = include_str!("../../llvq-cuda/kernels/llvq_rot.cuh");

/// Must match `LLVQ_ROT_KMAX` on both sides, which
/// `the_kmax_constant_is_the_same_number_in_both_kernels` checks.
const KMAX: usize = 32;

/// `(n, seed)`. The comment is what the shape is in a real model.
///
/// The widths span both branches of `k`, since `k == 1` takes a different
/// phase entirely, both ends of `m`, and the odd factors the served models
/// actually use.
const CASES: [(usize, u64); 8] = [
    (24, 0x1),    // m=8,    k=3   one Leech block, small enough to reason about
    (96, 0x2),    // m=32,   k=3
    (1024, 0x3),  // m=1024, k=1   the no-mix branch, narrow
    (3072, 0x4),  // m=1024, k=3   Qwen3-0.6B intermediate
    (2560, 0x5),  // m=512,  k=5   Qwen3-4B hidden: q/k/v/gate/up input
    (4096, 0x6),  // m=4096, k=1   Qwen3-4B o_proj input, widest power of two
    (9728, 0x7),  // m=512,  k=19  Qwen3-4B down_proj input, widest dense block
    (12288, 0x8), // m=4096, k=3   Qwen3-8B intermediate
];

// ---------------------------------------------------------------------------
// The fixture. Both references and the shader are handed the same bytes.
// ---------------------------------------------------------------------------

struct Fixture {
    n: usize,
    m: usize,
    k: usize,
    /// `1/sqrt(m)`, narrowed once here and never recomputed on a device.
    inv: f32,
    /// The activation, as f16 bit patterns.
    bits: Vec<u16>,
    /// The sign bitmap the kernels read, one `u32` per 32 coordinates.
    signbits: Vec<u32>,
    /// The same signs as `±1`, for the definition reference.
    signs: Vec<f64>,
    /// `Q_odd` padded to `KMAX × KMAX` row-major, as the kernels want it.
    small: Vec<f32>,
    /// `Q_odd` as `k × k` in f64, for the definition reference.
    qodd: Vec<f64>,
}

/// A `k × k` orthonormal matrix by Gram-Schmidt on Gaussian rows.
///
/// This is FIXTURE generation, not reference arithmetic. `Q_odd` is an input
/// table both sides read; how it was drawn changes nothing either kernel
/// computes. It is orthonormal so that
/// `the_rotation_preserves_the_norm` has a property to check.
fn orthonormal(k: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let mut q = vec![0.0f64; k * k];
    for i in 0..k {
        let mut row: Vec<f64> = (0..k).map(|_| rng.next_gaussian()).collect();
        for p in 0..i {
            let prev = &q[p * k..(p + 1) * k];
            let d: f64 = row.iter().zip(prev).map(|(a, b)| a * b).sum();
            for (r, b) in row.iter_mut().zip(prev) {
                *r -= d * b;
            }
        }
        let nrm = row.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(nrm > 1e-9, "degenerate Gram-Schmidt, pick another seed");
        for (slot, r) in q[i * k..(i + 1) * k].iter_mut().zip(row.iter()) {
            *slot = r / nrm;
        }
    }
    q
}

/// `pad` is what fills `small` outside its `k × k` corner.
///
/// Zero is the host contract. Making it an argument is what lets
/// `the_mix_ignores_whatever_pads_the_small_block` show that the kernel does
/// not depend on it.
fn fixture(n: usize, seed: u64, pad: f32) -> Fixture {
    let m = 1usize << n.trailing_zeros();
    let k = n / m;
    assert!(k <= KMAX, "case n={n} has k={k}, past the kernel's cap");

    let mut rng = SplitMix64::new(seed);
    let signs: Vec<f64> = (0..n)
        .map(|_| if rng.next_gaussian() < 0.0 { -1.0 } else { 1.0 })
        .collect();
    // `Q_odd` is the identity when there is no odd factor, and that is a fact
    // about the transform rather than a shortcut. Both kernels take the
    // scale-out branch at `k == 1` and never read `small`, and
    // `llvq-quant/src/rotation.rs::mix` returns early at the same point. A
    // 1×1 Gram-Schmidt draw is ±1, so drawing one here would make the
    // definition reference disagree with all three by a global sign. It did,
    // at n=4096, which is what the reference is for.
    let qodd = match k {
        1 => vec![1.0f64],
        _ => orthonormal(k, &mut rng),
    };

    let mut signbits = vec![0u32; n.div_ceil(32)];
    for (i, &s) in signs.iter().enumerate() {
        if s < 0.0 {
            signbits[i >> 5] |= 1 << (i & 31);
        }
    }

    let mut small = vec![pad; KMAX * KMAX];
    for g in 0..k {
        for t in 0..k {
            small[g * KMAX + t] = qodd[g * k + t] as f32;
        }
    }

    // The activation is drawn in f32 and narrowed to f16, which is what an
    // inference runtime hands over. Both references widen the same bits back,
    // so neither is charged for that rounding.
    let mut arng = SplitMix64::new(seed ^ 0xa5a5_a5a5);
    let bits: Vec<u16> = (0..n).map(|_| f16_bits(arng.next_gaussian() as f32)).collect();

    Fixture {
        n,
        m,
        k,
        inv: (1.0f64 / (m as f64).sqrt()) as f32,
        bits,
        signbits,
        signs,
        small,
        qodd,
    }
}

// ---------------------------------------------------------------------------
// Reference 1: the CUDA kernel text, compiled by clang++ and run.
// ---------------------------------------------------------------------------

/// How many row counts `host_rotate.cpp` runs `rot_apply_rows` at.
const CUDA_ROWS_CASES: usize = 3;

/// Compile `llvq-cuda/tests/host_rotate.cpp` and return the binary.
///
/// The flags are the ones `llvq-cuda/tests/rotation_matches_rust.rs` uses, and
/// `-ffp-contract=off` is the load-bearing one: the kernel spells out
/// `__fmaf_rn` where it wants a fused multiply-add, so a host compiler free to
/// fuse anything else would make a float difference unattributable.
///
/// The output name differs from the CUDA crate's so the two test binaries can
/// run at the same time without overwriting each other.
fn build_cuda_driver() -> std::path::PathBuf {
    let out = std::env::temp_dir().join("llvq_host_rotate_msl");
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../llvq-cuda/tests/host_rotate.cpp");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the CUDA rotation does not compile as host C++");
    out
}

/// `rot_apply` as the CUDA text computes it, at one thread.
///
/// The driver also replays the phases at 7, 32 and 256 threads and appends the
/// `rot_apply_rows` residuals. Those are the CUDA crate's own assertions; what
/// is taken here is the first arm, which is the real kernel run whole.
fn run_cuda(driver: &std::path::Path, f: &Fixture) -> Vec<f32> {
    let mut fx: Vec<u8> = Vec::new();
    fx.extend((f.n as u32).to_le_bytes());
    fx.extend((f.m as u32).to_le_bytes());
    fx.extend((f.k as u32).to_le_bytes());
    fx.extend(f.inv.to_le_bytes());
    for w in &f.signbits {
        fx.extend(w.to_le_bytes());
    }
    for v in &f.small {
        fx.extend(v.to_le_bytes());
    }
    for b in &f.bits {
        fx.extend(b.to_le_bytes());
    }

    let mut child = Command::new(driver)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the rotation driver runs");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&fx)
        .expect("fixture written");
    let out = child.wait_with_output().expect("driver finished");
    assert!(out.status.success(), "driver failed on n={}", f.n);
    assert_eq!(
        out.stdout.len(),
        4 * f.n * 4 + CUDA_ROWS_CASES * 4,
        "driver returned {} bytes for n={}",
        out.stdout.len(),
        f.n
    );
    out.stdout[..4 * f.n]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Reference 2: the definition, in f64, with no butterfly in it.
// ---------------------------------------------------------------------------

/// `Q x` from `Q = (Q_odd ⊗ H_m) D`, summed densely.
///
/// `H_m[j][j'] = (-1)^popcount(j & j') / sqrt(m)` is the Walsh-Hadamard matrix
/// written down rather than computed. A butterfly and this sum agree only if
/// the butterfly's addressing is right, which is what makes this an
/// independent check and not a restatement.
fn reference_from_definition(f: &Fixture) -> Vec<f64> {
    let (n, m, k) = (f.n, f.m, f.k);
    let scale = 1.0f64 / (m as f64).sqrt();

    // D: the sign flip, on the widened activation.
    let d: Vec<f64> = (0..n).map(|i| f16_to_f64(f.bits[i]) * f.signs[i]).collect();

    // H_m within each group.
    let mut h = vec![0.0f64; n];
    for g in 0..k {
        for j in 0..m {
            let mut acc = 0.0f64;
            for (jp, dv) in d[g * m..(g + 1) * m].iter().enumerate() {
                acc += if (j & jp).count_ones() % 2 == 1 { -dv } else { *dv };
            }
            h[g * m + j] = acc * scale;
        }
    }

    // Q_odd across the groups, at each position.
    let mut out = vec![0.0f64; n];
    for j in 0..m {
        for g in 0..k {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += f.qodd[g * k + t] * h[t * m + j];
            }
            out[g * m + j] = acc;
        }
    }
    out
}

/// Relative error in the norm, and the worst coordinate against a typical one.
///
/// Two numbers because they fail differently. A systematic error moves the
/// norm. A single mis-addressed butterfly leaves the norm almost intact and
/// shows only in the maximum. An orthogonal transform preserves the norm, so
/// `‖want‖/sqrt(n)` is what one coordinate is worth.
fn errors(got: &[f32], want: &[f64]) -> (f64, f64) {
    let nrm = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    let dev = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w) * (g as f64 - w))
        .sum::<f64>()
        .sqrt();
    let worst = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w).abs())
        .fold(0.0, f64::max);
    let typical = nrm / (want.len() as f64).sqrt();
    (dev / nrm.max(1e-300), worst / typical.max(1e-300))
}

// ---------------------------------------------------------------------------
// Driving the shader.
// ---------------------------------------------------------------------------

/// A one-element `small`, for the `k == 1` shapes.
///
/// Metal refuses a zero-length buffer and hands back a null pointer. The
/// `k == 1` branch never reads `small`, so a host that has no odd factor still
/// owes a dummy. `a_dummy_small_buffer_is_enough_when_there_is_no_odd_factor`
/// is what pins it.
const DUMMY_SMALL: [f32; 1] = [0.0];

struct Run {
    /// Threads in the single threadgroup. Clamped to what the pipeline takes.
    nthreads: u64,
    /// Metal's fast math off, as `Kernel::new_exact` does.
    exact: bool,
    /// Elements of padding placed before the activation, passed as `x_off`.
    x_off: usize,
    /// Hand the kernel a one-element `small` instead of the padded block.
    dummy_small: bool,
}

impl Default for Run {
    fn default() -> Self {
        Run { nthreads: 256, exact: true, x_off: 0, dummy_small: false }
    }
}

thread_local! {
    /// One pipeline per entry point and fast-math setting, reused.
    ///
    /// Compiling this shader costs 11 s in a debug build and 5 ms in a release
    /// one (*measured*, 2026-09-21). The tests below dispatch hundreds of
    /// times, so compiling per call put one debug test at 91 s and the file at
    /// 69 s, which is not a fast loop. Metal objects are not `Send`, so the
    /// cache is per thread and the harness's parallelism still works.
    ///
    /// Sharing a pipeline changes nothing a test can see. Every dispatch makes
    /// its own command buffer, encoder and buffers.
    static PIPELINES: RefCell<HashMap<(&'static str, bool), Rc<Kernel>>> =
        RefCell::new(HashMap::new());
}

fn compile(name: &'static str, exact: bool) -> Rc<Kernel> {
    PIPELINES.with(|p| {
        Rc::clone(p.borrow_mut().entry((name, exact)).or_insert_with(|| {
            Rc::new(
                match exact {
                    true => Kernel::new_exact(SOURCE, name),
                    false => Kernel::new(SOURCE, name),
                }
                .expect("the rotation shader compiles"),
            )
        }))
    })
}

/// `rot_apply_metal` on one activation.
fn run_metal(f: &Fixture, r: &Run) -> Vec<f32> {
    let kern = compile("rot_apply_metal", r.exact);

    // One threadgroup, always. `Kernel::dispatch` clamps the group size to
    // what the pipeline accepts but leaves the total alone, so asking for more
    // threads than the pipeline takes would silently launch a SECOND
    // threadgroup on the same scratch. Clamping both sides is what keeps the
    // dispatch to one.
    let g = r.nthreads.min(kern.max_threads_per_group());

    // The activation preceded by `x_off` elements of junk, so a kernel that
    // ignored the offset would read the junk.
    let mut padded = vec![f16_bits(-7.5); r.x_off];
    padded.extend_from_slice(&f.bits);

    let b_x = kern.buffer(&padded);
    let b_sign = kern.buffer(&f.signbits);
    let b_small = match r.dummy_small {
        true => kern.buffer(&DUMMY_SMALL),
        false => kern.buffer(&f.small),
    };
    let b_out = kern.empty::<f32>(f.n);
    // The scratch arrives poisoned, which turns two silent failures loud. A
    // slot `rot_load` never covers stays NaN. A stage that reads a slot before
    // the thread that owns it has written turns the answer NaN instead of
    // plausible. The second is a race, so this makes its CONSEQUENCE visible
    // and does not make it happen.
    let b_scr = kern.buffer(&vec![f32::NAN; f.n]);

    let (n, m, k) = (f.n as u32, f.m as u32, f.k as u32);
    let (inv, x_off) = (f.inv, r.x_off as u32);

    kern.dispatch(g, g, |enc| {
        enc.set_buffer(0, Some(&b_x), 0);
        enc.set_buffer(1, Some(&b_sign), 0);
        enc.set_buffer(2, Some(&b_small), 0);
        enc.set_buffer(3, Some(&b_out), 0);
        enc.set_buffer(4, Some(&b_scr), 0);
        enc.set_bytes(5, 4, &n as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &m as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &inv as *const f32 as *const c_void);
        enc.set_bytes(9, 4, &x_off as *const u32 as *const c_void);
    });

    unsafe { kern.read::<f32>(&b_out, f.n) }
}

/// `rot_apply_rows_metal` on a batch whose rows are `row_stride` apart.
fn run_metal_rows(f: &Fixture, batch: &[u16], rows: usize, row_stride: usize) -> Vec<f32> {
    let kern = compile("rot_apply_rows_metal", true);
    let g = 256u64.min(kern.max_threads_per_group());

    let b_x = kern.buffer(batch);
    let b_sign = kern.buffer(&f.signbits);
    let b_small = kern.buffer(&f.small);
    let b_out = kern.empty::<f32>(rows * f.n);
    let b_scr = kern.buffer(&vec![f32::NAN; rows * f.n]);

    let (n, m, k) = (f.n as u32, f.m as u32, f.k as u32);
    let (inv, x_off, stride) = (f.inv, 0u32, row_stride as u32);

    kern.dispatch(rows as u64 * g, g, |enc| {
        enc.set_buffer(0, Some(&b_x), 0);
        enc.set_buffer(1, Some(&b_sign), 0);
        enc.set_buffer(2, Some(&b_small), 0);
        enc.set_buffer(3, Some(&b_out), 0);
        enc.set_buffer(4, Some(&b_scr), 0);
        enc.set_bytes(5, 4, &n as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &m as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &inv as *const f32 as *const c_void);
        enc.set_bytes(9, 4, &x_off as *const u32 as *const c_void);
        enc.set_bytes(10, 4, &stride as *const u32 as *const c_void);
    });

    unsafe { kern.read::<f32>(&b_out, rows * f.n) }
}

// ---------------------------------------------------------------------------
// The gates.
// ---------------------------------------------------------------------------

/// The whole point of the file: the shader and the CUDA text agree on f32.
///
/// Exact equality over eight widths, both branches of `k`. Every operation on
/// both sides is an IEEE add, subtract, multiply or fused multiply-add, so
/// there is no reassociation left to round differently and one differing bit
/// is a defect.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs eight widths, run in release")]
fn the_metal_rotation_is_the_cuda_rotation_bit_for_bit() {
    let driver = build_cuda_driver();
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let want = run_cuda(&driver, &f);
        let got = run_metal(&f, &Run::default());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "n={n} coordinate {i}: metal {g} against cuda {w}"
            );
        }
    }
}

/// The transform is the one the definition describes, not merely the one the
/// CUDA computes.
///
/// This is the reference that shares no code with either kernel. It closes the
/// case where both GPUs agree on a wrong basis, which no diff between them can
/// see.
#[test]
#[cfg_attr(debug_assertions, ignore = "a dense f64 transform at eight widths, run in release")]
fn the_rotation_is_the_transform_the_definition_describes() {
    let mut worst_rel = 0.0f64;
    let mut worst_max = 0.0f64;
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let got = run_metal(&f, &Run::default());
        let want = reference_from_definition(&f);
        let (rel, max) = errors(&got, &want);
        assert!(
            rel < 1e-5 && max < 1e-3,
            "n={n}: relative {rel:.3e}, worst coordinate {max:.3e}"
        );
        worst_rel = worst_rel.max(rel);
        worst_max = worst_max.max(max);
    }
    eprintln!(
        "worst over {} shapes: relative {worst_rel:.3e}, coordinate {worst_max:.3e}",
        CASES.len()
    );
}

/// The work split does not move a bit.
///
/// Every butterfly touches two slots no other pair touches, so a different
/// thread count reorders nothing. A stage where two threads own the same slot
/// is the one plausible addressing bug, and it would show here as a
/// width-dependent answer. 7 is in the list because it divides nothing.
#[test]
fn the_work_split_does_not_move_a_bit() {
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let base = run_metal(&f, &Run { nthreads: 1, ..Run::default() });
        for w in [7u64, 32, 64, 256, 1024] {
            let got = run_metal(&f, &Run { nthreads: w, ..Run::default() });
            assert_eq!(got, base, "n={n}: {w} threads disagrees with 1");
        }
    }
}

/// `rot_apply_rows_metal` row `r` IS `rot_apply_metal` of row `r`.
///
/// The rows are `n + 8` apart, deliberately not `n`. The kernel reads the
/// input at `x_off + r · row_stride` and writes the output at `r · n`, two
/// quantities that are equal in every model this repository serves. A kernel
/// that used one of them for both is right on every real shape and wrong here.
///
/// The rows are distinct, since row `r` is the fixture cyclically shifted by
/// `r`, so reading the wrong row is a wrong answer rather than a lucky one.
/// Exact equality: the two paths run the same helpers on the same values in
/// the same order.
#[test]
fn the_batched_rotation_is_the_single_row_one_row_by_row() {
    const STRIDE_PAD: usize = 8;
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let stride = n + STRIDE_PAD;
        // 64 and 96 are here on purpose, and a review of 2026-09-21 is why.
        //
        // The batched kernel stages each row at `scratch + r * n`. Dropping
        // that offset, so every row stages over its neighbour, SURVIVED a
        // sweep of 1, 3 and 4 rows: below this machine's concurrency the
        // threadgroups do not overlap and the race never starts. Measured on
        // an M3 Max: the mutant lives at 5, 8, 12, 16, 20, 24, 32 and 40 rows
        // and dies at 48, 56, 64 and 128.
        //
        // CUDA cannot have this defect: its staging is a `__shared__` array
        // per block. It is the one thing the port introduces, so it is the one
        // thing the sweep must reach.
        for rows in [1usize, 3, 4, 64, 96] {
            let mut batch = vec![0u16; stride * rows];
            for r in 0..rows {
                for i in 0..n {
                    batch[r * stride + i] = f.bits[(i + r) % n];
                }
            }
            let many = run_metal_rows(&f, &batch, rows, stride);
            for r in 0..rows {
                let shifted = Fixture {
                    bits: (0..n).map(|i| f.bits[(i + r) % n]).collect(),
                    ..fixture(n, seed, 0.0)
                };
                let one = run_metal(&shifted, &Run::default());
                for (i, (a, b)) in many[r * n..(r + 1) * n].iter().zip(&one).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "n={n}, {rows} rows, row {r} coordinate {i}: {a} against {b}"
                    );
                }
            }
        }
    }
}

/// `x_off` is read, and it is read as an element offset.
///
/// The batched kernel derives its own offset from it, so a kernel that ignored
/// it would be wrong on every prefill row but right on the single-row tests.
#[test]
fn the_activation_offset_is_honoured() {
    for (n, seed) in [CASES[0], CASES[2], CASES[4]] {
        let f = fixture(n, seed, 0.0);
        let base = run_metal(&f, &Run::default());
        for off in [1usize, 3, 37] {
            let got = run_metal(&f, &Run { x_off: off, ..Run::default() });
            assert_eq!(got, base, "n={n}: an offset of {off} moved the answer");
        }
    }
}

/// The rotation is orthogonal, so it preserves the norm.
///
/// A property neither reference can be tricked into sharing. A missing scale
/// or a transposed `Q_odd` would satisfy a diff against a reference that made
/// the same mistake, and would fail here.
#[test]
fn the_rotation_preserves_the_norm() {
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let before = f.bits.iter().map(|&h| f16_to_f64(h) * f16_to_f64(h)).sum::<f64>().sqrt();
        let got = run_metal(&f, &Run::default());
        let after = got.iter().map(|&g| g as f64 * g as f64).sum::<f64>().sqrt();
        let rel = (after - before).abs() / before;
        assert!(rel < 1e-5, "n={n}: norm {before:.9} became {after:.9} ({rel:.3e})");
    }
}

/// What pads `small` must not reach the result.
///
/// The mix loops to `LLVQ_ROT_KMAX` and two separate things keep the extra
/// terms harmless: the zeros the host pads `small` with, and the zeros the
/// kernel puts in `col` past `k`. Either alone suffices, so mutating one of
/// them changes nothing. Filling the padding with something else is what makes
/// `col`'s zeros load-bearing.
///
/// The results must stay bit-identical, since `pad · 0` is exactly zero and an
/// `fma` with an exact zero addend changes no bit. The contract is "any finite
/// pad", not "any pad": `NaN · 0` is NaN, and keeping one out is the host's
/// job.
#[test]
fn the_mix_ignores_whatever_pads_the_small_block() {
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        if f.k == 1 {
            continue; // no mix at all on this branch
        }
        let clean = run_metal(&f, &Run::default());
        for pad in [1.0f32, -3.5e12, 7.25e-9] {
            let dirty = run_metal(&fixture(n, seed, pad), &Run::default());
            assert_eq!(clean, dirty, "n={n}: padding `small` with {pad} moved the result");
        }
    }
}

/// A one-element `small` is enough when there is no odd factor.
///
/// Metal refuses a zero-length buffer and returns a null pointer, so a host
/// with `k == 1` and no `Q_odd` to upload still owes a dummy. This is the
/// assertion that the `k == 1` branch really never reads it.
#[test]
fn a_dummy_small_buffer_is_enough_when_there_is_no_odd_factor() {
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        if f.k != 1 {
            continue;
        }
        let full = run_metal(&f, &Run::default());
        let dummy = run_metal(&f, &Run { dummy_small: true, ..Run::default() });
        assert_eq!(full, dummy, "n={n}: the k=1 branch reads `small`");
    }
}

/// The answer depends on the activation.
///
/// A kernel that returned a function of its tables alone would pass every
/// equality above if both references did the same. They do not, and this says
/// so cheaply.
#[test]
fn a_different_activation_gives_a_different_answer() {
    let mut f = fixture(2560, 0x5, 0.0);
    let a = run_metal(&f, &Run::default());
    for b in f.bits.iter_mut() {
        *b = f16_bits(f16_to_f64(*b) as f32 + 1.0);
    }
    let b = run_metal(&f, &Run::default());
    assert!(a.iter().zip(&b).any(|(p, q)| p != q), "the rotation must read the activation");
}

/// The pragma, not the compile option, is what carries the arithmetic.
///
/// The shipped path does not use `new_exact`. A Metal adapter compiles the
/// same source through candle with default options, which is fast math ON. If
/// the arithmetic depended on the option, the gate and the shipped path would
/// be two different kernels.
#[test]
fn the_same_source_gives_the_same_numbers_with_fast_math_either_way() {
    for (n, seed) in CASES {
        let f = fixture(n, seed, 0.0);
        let exact = run_metal(&f, &Run::default());
        let fast = run_metal(&f, &Run { exact: false, ..Run::default() });
        for (i, (a, b)) in exact.iter().zip(&fast).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "n={n} coordinate {i}: fast math moved the answer, {a} against {b}"
            );
        }
    }
}

/// `LLVQ_ROT_KMAX` is the same number in both kernels, and in this file.
///
/// The host pads `small` to `KMAX × KMAX` and both kernels read `KMAX` of
/// them. A drift would read past the block the host uploaded. A comment saying
/// "keep these in sync" is not a mechanism, so this reads both texts.
#[test]
fn the_kmax_constant_is_the_same_number_in_both_kernels() {
    fn kmax_of(src: &str, what: &str) -> usize {
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with("#define LLVQ_ROT_KMAX"))
            .unwrap_or_else(|| panic!("{what} defines LLVQ_ROT_KMAX"));
        line.split_whitespace()
            .nth(2)
            .and_then(|t| t.trim_end_matches('u').parse().ok())
            .unwrap_or_else(|| panic!("{what} gives LLVQ_ROT_KMAX a literal"))
    }
    let msl = kmax_of(SOURCE, "the shader");
    let cuda = kmax_of(CUDA_HEADER, "the CUDA header");
    assert_eq!(msl, cuda, "shader says {msl}, CUDA says {cuda}");
    assert_eq!(msl, KMAX, "shader says {msl}, this test says {KMAX}");
}

/// The contraction pragma is in the source.
///
/// Its effect is checked by the two equality gates above. This checks its
/// presence, which is what a future edit would drop by accident.
#[test]
fn the_shader_forbids_contraction() {
    assert!(
        SOURCE.contains("#pragma clang fp contract(off)"),
        "the shader must forbid `a * b + c` becoming an fma"
    );
}
