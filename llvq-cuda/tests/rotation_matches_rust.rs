//! The CUDA rotation, compiled by `clang++` and diffed against the Rust one.
//!
//! `llvq-quant`'s `Rotation` is the specification: it is what quantized the
//! sealed artifact, so whatever it computes *is* the basis the stored weights
//! live in. This test compiles the kernel text NVRTC will compile, runs it,
//! and requires it to reproduce that transform.
//!
//! ## Why this can run on a Mac, when the matvec cannot
//!
//! `rot_apply` uses shared memory and barriers, which a single-threaded
//! driver cannot reproduce — but it takes its thread identity as *arguments*
//! rather than reading `threadIdx` inside the phases. At one thread the
//! barriers are vacuous and the arithmetic is unchanged, so the host runs the
//! real kernel. The phases are then replayed at 7, 32 and 256 threads and all
//! four results must agree: that is what pins the Walsh–Hadamard addressing,
//! whose one plausible bug is a stage where two threads own the same slot.
//!
//! What is left for the card: occupancy, register pressure, and that the
//! barriers really are where the source says. Everything arithmetic is here.
//!
//! ## The fixture exercises the parameters, not just the shapes
//!
//! Per §5 of `CLAUDE.md`: an assertion that never varies the parameter it
//! covers tests nothing. So the widths span both branches of `k` (`k == 1`
//! takes a different phase entirely), both ends of `m`, and the odd factors
//! the real models actually use — `k = 19` for Qwen3-4B's `down_proj` is the
//! widest dense block in the deliverable, and `n = 4096` is the only shape
//! where the mix is skipped.

use llvq_core::SplitMix64;
use llvq_cuda::{f16_bits, f16_to_f64, ROT_KMAX};
use llvq_quant::rotation::Rotation;
use std::io::Write;
use std::process::{Command, Stdio};

/// The crate's mirror of `LLVQ_ROT_KMAX`, pinned against the kernel text
/// by `the_kmax_constant_matches_the_kernel`.
const KMAX: usize = ROT_KMAX;

/// `(n, seed)`. The comment is what the shape is in a real model.
const CASES: [(usize, u64); 8] = [
    (24, 0x1),      // m=8,    k=3   — one Leech block, small enough to reason about
    (96, 0x2),      // m=32,   k=3
    (1024, 0x3),    // m=1024, k=1   — the no-mix branch, narrow
    (3072, 0x4),    // m=1024, k=3   — Qwen3-0.6B intermediate
    (2560, 0x5),    // m=512,  k=5   — Qwen3-4B hidden: q/k/v/gate/up input
    (4096, 0x6),    // m=4096, k=1   — Qwen3-4B o_proj input, widest power of two
    (9728, 0x7),    // m=512,  k=19  — Qwen3-4B down_proj input, widest dense block
    (12288, 0x8),   // m=4096, k=3   — Qwen3-8B intermediate
];

fn build_driver() -> std::path::PathBuf {
    let out = std::env::temp_dir().join("llvq_host_rotate");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // Same reasoning as `decoder_matches_rust.rs`: NVRTC contracts
            // `a*b+c` into an FMA, so a host compiler free to do it or not at
            // its own discretion would make a float difference impossible to
            // attribute. Contraction off, and the one multiply-add this
            // kernel depends on is written as an explicit `__fmaf_rn`.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_rotate.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the rotation kernel does not compile as host C++");
    out
}

/// `(reference, four kernel results)` for one shape.
///
/// `pad` is what fills `small` outside its `k × k` corner. Zero is the
/// contract; the point of making it an argument is that the kernel must not
/// depend on it — see `the_mix_ignores_whatever_pads_the_small_block`.
fn run_case(
    driver: &std::path::Path,
    n: usize,
    seed: u64,
    pad: f32,
) -> (Vec<f64>, Vec<Vec<f32>>) {
    let rot = Rotation::new(n, seed);
    let (m, k) = (rot.pow2(), rot.odd());
    assert!(k <= KMAX, "case n={n} has k={k}, past the kernel's cap");

    // The activation the kernel will see: drawn in f32, narrowed to f16, and
    // widened again for the reference. Anything else would compare the kernel
    // against a vector it was never given and charge it for the rounding.
    let mut rng = SplitMix64::new(seed ^ 0xa5a5_a5a5);
    let bits: Vec<u16> = (0..n).map(|_| f16_bits(rng.next_gaussian() as f32)).collect();
    let mut want: Vec<f64> = bits.iter().map(|&h| f16_to_f64(h)).collect();
    rot.apply(&mut want);

    let mut fixture: Vec<u8> = Vec::new();
    fixture.extend((n as u32).to_le_bytes());
    fixture.extend((m as u32).to_le_bytes());
    fixture.extend((k as u32).to_le_bytes());
    // `1/√m` narrowed once, here, and handed over — never recomputed on the
    // device, where `sqrtf` and `rsqrtf` would each land somewhere else.
    fixture.extend(((1.0f64 / (m as f64).sqrt()) as f32).to_le_bytes());

    let mut words = vec![0u32; n.div_ceil(32)];
    for (i, &s) in rot.signs().iter().enumerate() {
        if s < 0.0 {
            words[i >> 5] |= 1 << (i & 31);
        }
    }
    for w in &words {
        fixture.extend(w.to_le_bytes());
    }

    // Padded to KMAX × KMAX, because the kernel's inner loops are unrolled to
    // a compile-time bound and must read something at every slot.
    let mut small = vec![pad; KMAX * KMAX];
    for g in 0..k {
        for t in 0..k {
            small[g * KMAX + t] = rot.small()[g * k + t] as f32;
        }
    }
    for v in &small {
        fixture.extend(v.to_le_bytes());
    }
    for b in &bits {
        fixture.extend(b.to_le_bytes());
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
        .write_all(&fixture)
        .expect("fixture written");
    let out = child.wait_with_output().expect("driver finished");
    assert!(out.status.success(), "driver failed on n={n}");
    assert_eq!(
        out.stdout.len(),
        4 * n * 4,
        "driver returned {} bytes for n={n}",
        out.stdout.len()
    );

    let got: Vec<Vec<f32>> = out
        .stdout
        .chunks_exact(n * 4)
        .map(|c| {
            c.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    (want, got)
}

/// Relative error in the norm, and the worst single coordinate measured
/// against a *typical* one.
///
/// Two numbers rather than one because they fail differently: a systematic
/// error (a missing scale, a transposed `Q_odd`) moves the norm, while a
/// single mis-addressed butterfly leaves the norm almost intact and shows up
/// only in the max. An orthogonal transform preserves the norm, so
/// `‖want‖/√n` is what one coordinate is worth and the right denominator.
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

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs eight widths, run in release")]
fn the_cuda_rotation_computes_what_the_rust_rotation_computes() {
    let driver = build_driver();
    let mut worst_rel = 0.0f64;
    let mut worst_max = 0.0f64;

    for (n, seed) in CASES {
        let (want, got) = run_case(&driver, n, seed, 0.0);
        let widths = [1u32, 7, 32, 256];

        for (arm, g) in got.iter().enumerate() {
            let (rel, max) = errors(g, &want);
            assert!(
                rel < 1e-5 && max < 1e-3,
                "n={n} at {} threads: relative {rel:.3e}, worst coordinate {max:.3e}",
                widths[arm]
            );
            worst_rel = worst_rel.max(rel);
            worst_max = worst_max.max(max);
        }

        // The work split must not change the answer. This is the assertion
        // the thread-count sweep exists for: it is bit-exact, because every
        // butterfly touches two slots no other pair touches, so a different
        // width reorders nothing an FMA could round differently.
        for (arm, g) in got.iter().enumerate().skip(1) {
            assert_eq!(
                *g, got[0],
                "n={n}: {} threads disagrees with the kernel at 1",
                widths[arm]
            );
        }
    }

    eprintln!("worst over {} shapes: relative {worst_rel:.3e}, coordinate {worst_max:.3e}", CASES.len());
}

/// The rotation is orthogonal, so it must preserve the norm — a property the
/// diff against Rust cannot check, since a shared misconception would satisfy
/// both. This is the independent one.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn the_cuda_rotation_preserves_the_norm() {
    let driver = build_driver();
    for (n, seed) in CASES {
        // The same draw `run_case` makes, so this is the norm of the vector
        // the kernel is actually handed.
        let mut rng = SplitMix64::new(seed ^ 0xa5a5_a5a5);
        let bits: Vec<u16> = (0..n).map(|_| f16_bits(rng.next_gaussian() as f32)).collect();
        let before = bits
            .iter()
            .map(|&h| f16_to_f64(h) * f16_to_f64(h))
            .sum::<f64>()
            .sqrt();

        let (_, got) = run_case(&driver, n, seed, 0.0);
        let after = got[0].iter().map(|&g| g as f64 * g as f64).sum::<f64>().sqrt();
        let rel = (after - before).abs() / before;
        assert!(rel < 1e-5, "n={n}: norm {before:.9} became {after:.9} ({rel:.3e})");
    }
}

/// What pads `small` must not reach the result.
///
/// This exists because of a surviving mutant. The mix loops to
/// `LLVQ_ROT_KMAX` and two separate things keep the extra terms harmless: the
/// zeros the host pads `small` with, and the zeros the kernel puts in `col`
/// past `k`. Either alone suffices, so mutating one of them changes nothing —
/// `col[t] = 1.0f` instead of `0.0f` passed every assertion above.
///
/// Filling `small`'s padding with something else is the assertion that makes
/// `col`'s zeros load-bearing: the results must stay **bit-identical**, since
/// `pad · 0` is exactly zero and an `fma` with an exact zero addend changes no
/// bit. If `col` ever stopped being zeroed there, this is what would catch it.
///
/// The contract it pins is "any finite pad", not "any pad": `NaN · 0` is NaN,
/// so a NaN in the padding would poison the sum no zero can absorb. That is a
/// real limit of the design and it is the host's job not to hand one over.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn the_mix_ignores_whatever_pads_the_small_block() {
    let driver = build_driver();
    for (n, seed) in CASES {
        let (_, clean) = run_case(&driver, n, seed, 0.0);
        for pad in [1.0f32, -3.5e12, 7.25e-9] {
            let (_, dirty) = run_case(&driver, n, seed, pad);
            assert_eq!(
                clean, dirty,
                "n={n}: padding `small` with {pad} changed the result"
            );
        }
    }
}

/// `ROT_KMAX` and `LLVQ_ROT_KMAX` are the same number in two languages.
///
/// The host pads `small` to `ROT_KMAX × ROT_KMAX` and the kernel reads
/// `LLVQ_ROT_KMAX` of them. If they ever drifted, the kernel would read past
/// the block the host uploaded — on a card, an illegal address in the middle
/// of a billed job; here, silence. A comment saying "keep these in sync" is
/// not a mechanism, so this reads the kernel text and checks.
#[test]
fn the_kmax_constant_matches_the_kernel() {
    let src = include_str!("../kernels/llvq_rot.cuh");
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with("#define LLVQ_ROT_KMAX"))
        .expect("the kernel defines LLVQ_ROT_KMAX");
    let val: usize = line
        .split_whitespace()
        .nth(2)
        .and_then(|t| t.trim_end_matches('u').parse().ok())
        .expect("LLVQ_ROT_KMAX is a literal");
    assert_eq!(val, ROT_KMAX, "kernel says {val}, Rust says {ROT_KMAX}");
}
