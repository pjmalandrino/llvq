//! The batched GEMM of `kernels/tv_batch_gemm.cu`, executed on the CPU.
//!
//! `host_batch_gemm.cpp` runs every CTA of a launch through `host_cta.h` (real
//! barriers, real tickets, a permuted CTA order) with the tensor-core MMA
//! modelled lane by lane from the PTX fragment layout, and prints the kernel's
//! f16 outputs beside a double-precision reference built from the decoder's own
//! values and the worst-case error of the two f16 roundings the kernel makes
//! (the folded weight, the activation).
//!
//! What the bound holds is the indexing: a wrong chunk offset, a fragment
//! layout, a scale on the wrong row, a projection read with another's
//! activation, a slice folded twice are errors of the order of the output.
//! Per hard rule 10 a missing `clang++` fails, naming the tool.

#![cfg(any(target_os = "macos", feature = "cuda"))]

use llvq_core::SplitMix64;
use llvq_search::tetra::Tetra;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

fn probe() -> &'static std::path::Path {
    static EXE: OnceLock<std::path::PathBuf> = OnceLock::new();
    EXE.get_or_init(|| {
        let out = std::env::temp_dir().join(format!("llvq_host_batch_gemm_{}", std::process::id()));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        let res = Command::new("clang++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pthread"])
            .arg(dir.join("host_batch_gemm.cpp"))
            .arg("-o")
            .arg(&out)
            .output()
            .expect("clang++ is on PATH: the batched kernel text is checked by it (rule 10)");
        assert!(res.status.success(), "clang++ refused host_batch_gemm.cpp:\n{}", String::from_utf8_lossy(&res.stderr));
        out
    })
}

fn put<T: Copy>(v: &mut Vec<u8>, xs: &[T], f: impl Fn(T) -> Vec<u8>) {
    for &x in xs {
        v.extend(f(x));
    }
}

fn rnd(rng: &mut SplitMix64) -> f64 {
    (rng.next() >> 11) as f64 / (1u64 << 53) as f64
}

/// One projection of a launch.
enum P {
    /// d_out, nblocks, tail_w, splits
    Tetra(usize, usize, usize, u32),
    /// d_out, d_in, group shift, splits
    Int4(usize, usize, u32, u32),
}

impl P {
    fn d_out(&self) -> usize {
        match *self {
            P::Tetra(d, ..) | P::Int4(d, ..) => d,
        }
    }
}

fn launch(n_rows: usize, projs: &[P], seed: u64) {
    let t = Tetra::new();
    let mut rng = SplitMix64::new(seed);
    let mut f = Vec::new();
    f.extend((n_rows as u32).to_le_bytes());
    f.extend((projs.len() as u32).to_le_bytes());
    put(&mut f, t.rows(), |v: u32| v.to_le_bytes().to_vec());
    for s in 0..64usize {
        f.extend_from_slice(&t.prefixes()[s]);
    }
    for s in 0..64usize {
        for b in 0..16usize {
            let (byte, s16) = t.branches()[s][b];
            f.extend((byte as u16 | (s16 as u16) << 8).to_le_bytes());
        }
    }
    for s in 0..64usize {
        f.extend_from_slice(&t.suffixes()[s]);
    }
    let inv: Vec<f32> = (0..32).map(|m| if m == 0 { 0.0 } else { (1.0 / ((16 * m) as f64).sqrt()) as f32 }).collect();
    put(&mut f, &inv, |v: f32| v.to_le_bytes().to_vec());
    for p in projs {
        match *p {
            P::Tetra(d_out, nblocks, tail_w, splits) => {
                let d_in = nblocks * 24 + tail_w;
                // A row on an 8-byte boundary, as the artifact pads it.
                let rs = (nblocks * 6).div_ceil(8) * 2;
                f.extend(0u32.to_le_bytes());
                for v in [d_out, nblocks, tail_w, rs] {
                    f.extend((v as u32).to_le_bytes());
                }
                f.extend(splits.to_le_bytes());
                put(&mut f, &[0.031f32, 0.047], |v: f32| v.to_le_bytes().to_vec());
                let mut words = vec![0u32; d_out * rs];
                for row in 0..d_out {
                    let mut halves = vec![0u16; rs * 2];
                    for j in 0..nblocks {
                        let w = if row == 0 && j == 0 { 0 } else { rng.next() & ((1u64 << 48) - 1) };
                        for k in 0..3 {
                            halves[3 * j + k] = (w >> (16 * k)) as u16;
                        }
                    }
                    for i in 0..rs {
                        words[row * rs + i] = halves[2 * i] as u32 | (halves[2 * i + 1] as u32) << 16;
                    }
                }
                put(&mut f, &words, |v: u32| v.to_le_bytes().to_vec());
                let rscale: Vec<f32> = (0..d_out).map(|_| (0.5 + rnd(&mut rng)) as f32).collect();
                put(&mut f, &rscale, |v: f32| v.to_le_bytes().to_vec());
                let tail: Vec<u16> = (0..d_out * tail_w).map(|_| half::f16::from_f64(rnd(&mut rng) - 0.5).to_bits()).collect();
                put(&mut f, &tail, |v: u16| v.to_le_bytes().to_vec());
                // Rotated activations over three decades, rounded to f16 as the caller hands them.
                let x: Vec<u16> = (0..n_rows * d_in)
                    .map(|_| half::f16::from_f64((rnd(&mut rng) - 0.5) * 10f64.powf(3.0 * rnd(&mut rng) - 1.5)).to_bits())
                    .collect();
                put(&mut f, &x, |v: u16| v.to_le_bytes().to_vec());
            }
            P::Int4(d_out, d_in, gshift, splits) => {
                let gpr = d_in >> gshift;
                f.extend(1u32.to_le_bytes());
                for v in [d_out as u32, d_in as u32, gshift, splits] {
                    f.extend(v.to_le_bytes());
                }
                let words: Vec<u32> = (0..d_out * d_in / 8).map(|_| rng.next() as u32).collect();
                put(&mut f, &words, |v: u32| v.to_le_bytes().to_vec());
                let sc: Vec<u16> = (0..d_out * gpr).map(|_| half::f16::from_f64(0.002 + 0.01 * rnd(&mut rng)).to_bits()).collect();
                let bi: Vec<u16> = (0..d_out * gpr).map(|_| half::f16::from_f64(-0.08 * rnd(&mut rng)).to_bits()).collect();
                put(&mut f, &sc, |v: u16| v.to_le_bytes().to_vec());
                put(&mut f, &bi, |v: u16| v.to_le_bytes().to_vec());
                let x: Vec<u16> = (0..n_rows * d_in).map(|_| half::f16::from_f64((rnd(&mut rng) - 0.5) * 4.0).to_bits()).collect();
                put(&mut f, &x, |v: u16| v.to_le_bytes().to_vec());
            }
        }
    }
    let mut ch = Command::new(probe()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("probe runs");
    let mut stdin = ch.stdin.take().expect("stdin");
    let w = std::thread::spawn(move || stdin.write_all(&f).expect("fixture written"));
    let out = ch.wait_with_output().expect("probe exits");
    w.join().expect("writer");
    assert!(out.status.success(), "probe failed: {}", String::from_utf8_lossy(&out.stderr));
    let total: usize = projs.iter().map(|p| n_rows * p.d_out() * 10).sum();
    assert_eq!(out.stdout.len(), total, "probe wrote the wrong number of bytes");
    let mut at = 0usize;
    for (i, p) in projs.iter().enumerate() {
        let m = n_rows * p.d_out();
        let rd = |at: usize, k: usize| &out.stdout[at..at + k];
        let y: Vec<f32> = rd(at, 2 * m).chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
        let r: Vec<f32> = rd(at + 2 * m, 4 * m).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let b: Vec<f32> = rd(at + 6 * m, 4 * m).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        at += 10 * m;
        check(&format!("rows {n_rows}, projection {i}"), &y, &r, &b);
    }
}

/// Every output within its rounding bound plus the f16 rounding of the output
/// and a floor of 1e-4 of the RMS output; and the RMS error under 2e-3 of the
/// RMS reference, so that a zero or constant output cannot pass.
fn check(tag: &str, y: &[f32], r: &[f32], b: &[f32]) {
    let rms = (r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    assert!(rms > 0.0, "{tag}: a zero reference proves nothing");
    let mut se = 0.0f64;
    for (i, ((a, r), b)) in y.iter().zip(r).zip(b).enumerate() {
        let e = (*a as f64 - *r as f64).abs();
        se += e * e;
        let tol = *b as f64 + (*r as f64).abs() * 2f64.powi(-11) + 1e-4 * rms;
        assert!(e <= tol, "{tag}: output {i}: kernel {a}, reference {r}, |error| {e:.3e} > {tol:.3e}");
    }
    let rel = (se / y.len() as f64).sqrt() / rms;
    assert!(rel < 2e-3, "{tag}: RMS error {rel:.3e} of the RMS output");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn tetra_projections_match_the_reference_at_every_width() {
    // Each entry (8, 16, 32, 64, 128 rows) and ragged counts; a tail and none;
    // one slice and several; a chunk that ends inside a row's last block pair.
    launch(8, &[P::Tetra(128, 13, 16, 1)], 1);
    launch(13, &[P::Tetra(256, 21, 8, 3)], 2);
    launch(32, &[P::Tetra(128, 37, 0, 4)], 3);
    launch(1, &[P::Tetra(128, 9, 5, 2)], 4);
    launch(47, &[P::Tetra(128, 30, 16, 5)], 5);
    launch(128, &[P::Tetra(128, 11, 8, 2)], 6);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn int4_projections_match_the_reference_for_records_and_the_head() {
    launch(8, &[P::Int4(128, 512, 7, 1)], 11);
    launch(16, &[P::Int4(256, 1024, 7, 3)], 12);
    launch(29, &[P::Int4(128, 640, 6, 2)], 13);
    launch(64, &[P::Int4(128, 256, 6, 1)], 14);
    launch(100, &[P::Int4(128, 384, 7, 3)], 15);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn several_projections_share_one_launch() {
    // q, k (Tetra) and v (int4) of one step, each with its own split.
    launch(12, &[P::Tetra(256, 10, 16, 2), P::Tetra(128, 10, 16, 3), P::Int4(128, 256, 7, 1)], 21);
    // gate and up, one slice each, then four projections at 64 rows.
    launch(5, &[P::Tetra(128, 7, 8, 1), P::Tetra(256, 7, 8, 1)], 22);
    launch(64, &[P::Int4(128, 256, 7, 2), P::Tetra(128, 6, 0, 1), P::Tetra(128, 6, 0, 2), P::Int4(128, 128, 6, 1)], 23);
}
