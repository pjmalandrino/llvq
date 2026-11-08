//! The batched projections of `kernels/tv_batch_mma.cu`, executed on the CPU.
//!
//! `host_batch_mma.cpp` runs every CTA of a launch through `host_cta.h` (real
//! barriers, real tickets, a permuted CTA order) with the tensor-core MMA
//! modelled lane by lane from the PTX fragment layout, and prints the kernel's
//! f16 outputs beside a double-precision reference built from the decoder's own
//! values (`tetra48_decode_f`, pinned bit for bit to the Rust decoder by
//! `llvq-cuda/tests/tetra48_matches_rust.rs`).
//!
//! The comparison is a tolerance, and it is the right one: the kernel is not
//! bit for bit with batch 1 by design (the tensor core's sum order, the split
//! of K). What it must hold is the indexing, and a wrong index (a fragment
//! layout, a staging offset, a scale applied to the wrong row, a slice folded
//! twice) is an error of the order of the output. The bound below is the f16
//! rounding of the output plus a floor relative to the matrix's typical value.
//!
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
        let out = std::env::temp_dir().join(format!("llvq_host_batch_mma_{}", std::process::id()));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        let res = Command::new("clang++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pthread"])
            .arg(dir.join("host_batch_mma.cpp"))
            .arg("-o")
            .arg(&out)
            .output()
            .expect("clang++ is on PATH: the batched kernel text is checked by it (rule 10)");
        assert!(res.status.success(), "clang++ refused host_batch_mma.cpp:\n{}", String::from_utf8_lossy(&res.stderr));
        out
    })
}

fn run(input: &[u8], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ch = Command::new(probe()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("probe runs");
    let mut stdin = ch.stdin.take().expect("stdin");
    let data = input.to_vec();
    let w = std::thread::spawn(move || stdin.write_all(&data).expect("fixture written"));
    let out = ch.wait_with_output().expect("probe exits");
    w.join().expect("writer");
    assert!(out.status.success(), "probe failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout.len(), n * 6, "probe wrote the wrong number of bytes");
    let y = out.stdout[..2 * n].chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
    let r = out.stdout[2 * n..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    (y, r)
}

fn put<T: Copy>(v: &mut Vec<u8>, xs: &[T], f: impl Fn(T) -> Vec<u8>) {
    for &x in xs {
        v.extend(f(x));
    }
}

/// Every output within f16 rounding of the reference, plus a floor at 1e-4 of
/// the RMS output; and, so that a constant or a zero output cannot pass, the
/// RMS error under 1e-3 of the RMS reference.
fn check(tag: &str, y: &[f32], r: &[f32]) {
    let rms = (r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    assert!(rms > 0.0, "{tag}: a zero reference proves nothing");
    let mut se = 0.0f64;
    for (i, (a, b)) in y.iter().zip(r).enumerate() {
        let e = (*a as f64 - *b as f64).abs();
        se += e * e;
        let tol = (*b as f64).abs() * 2f64.powi(-10) + 1e-4 * rms;
        assert!(e <= tol, "{tag}: output {i}: kernel {a}, reference {b}, |error| {e:.3e} > {tol:.3e}");
    }
    let rel = (se / y.len() as f64).sqrt() / rms;
    assert!(rel < 1e-3, "{tag}: RMS error {rel:.3e} of the RMS output");
}

fn rnd(rng: &mut SplitMix64) -> f64 {
    (rng.next() >> 11) as f64 / (1u64 << 53) as f64
}

fn tetra(d_out: usize, nblocks: usize, tail_w: usize, n_rows: usize, splits: u32, seed: u64) {
    let t = Tetra::new();
    let mut rng = SplitMix64::new(seed);
    let d_in = nblocks * 24 + tail_w;
    let rs = (nblocks * 6).div_ceil(4) + 1; // a row on a u32 boundary, one word of slack
    let mut f = Vec::new();
    f.extend(0u32.to_le_bytes());
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
    put(&mut f, &[0.031f32, 0.047], |v: f32| v.to_le_bytes().to_vec());
    let inv: Vec<f32> = (0..32).map(|m| if m == 0 { 0.0 } else { (1.0 / ((16 * m) as f64).sqrt()) as f32 }).collect();
    put(&mut f, &inv, |v: f32| v.to_le_bytes().to_vec());
    for v in [d_out, nblocks, tail_w, n_rows] {
        f.extend((v as u32).to_le_bytes());
    }
    f.extend(splits.to_le_bytes());
    f.extend((rs as u32).to_le_bytes());
    // Words: random 48-bit labels (every value is one), the origin at block 0 of
    // row 0, packed as 16-bit halves, block j at halves 3j..3j+2 of its row.
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
    // Activations over four decades, as a rotated activation spreads.
    let x: Vec<f32> = (0..n_rows * d_in).map(|_| ((rnd(&mut rng) - 0.5) * 10f64.powf(4.0 * rnd(&mut rng) - 2.0)) as f32).collect();
    put(&mut f, &x, |v: f32| v.to_le_bytes().to_vec());
    let (y, r) = run(&f, n_rows * d_out);
    check(&format!("tetra d_out {d_out} blocks {nblocks} tail {tail_w} rows {n_rows} splits {splits}"), &y, &r);
}

fn q4(d_out: usize, d_in: usize, gshift: u32, n_rows: usize, splits: u32, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let gpr = d_in >> gshift;
    let mut f = Vec::new();
    f.extend(1u32.to_le_bytes());
    for v in [d_out as u32, d_in as u32, gshift, n_rows as u32, splits] {
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
    let (y, r) = run(&f, n_rows * d_out);
    check(&format!("int4 d_out {d_out} d_in {d_in} group {} rows {n_rows} splits {splits}", 1 << gshift), &y, &r);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn the_tetra_kernel_matches_the_reference_at_every_width_and_split() {
    // One slice and several; 8, 16 and 32 rows and ragged counts; a tail and none.
    tetra(128, 13, 16, 8, 1, 1);
    tetra(256, 21, 8, 13, 3, 2);
    tetra(128, 37, 0, 32, 4, 3);
    tetra(256, 9, 5, 1, 2, 4);
    tetra(128, 30, 16, 27, 7, 5);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn the_int4_kernel_matches_the_reference_for_records_and_the_head() {
    q4(128, 512, 7, 8, 1, 11);
    q4(256, 1024, 7, 16, 3, 12);
    q4(128, 640, 6, 32, 2, 13);
    q4(128, 256, 6, 5, 1, 14);
    q4(256, 768, 7, 29, 5, 15);
}
