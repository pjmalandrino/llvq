//! # Gate G5 — the calibration seam
//!
//! The Hessian accumulator sits between a validated forward pass and a
//! validated GPTQ core, and until now nothing tested it. That is exactly the
//! shape of the two defects this project has already paid for: a ridge term
//! that no test exercised, and a monotonicity assertion that a no-op
//! satisfied. `H = AᵀA/N` is one line of algebra with an exact reference, so
//! there is no excuse for taking it on faith.

use candle_core::{Device, Tensor};
use llvq_llm::calib::Hessian;

const TOL: f64 = 1e-4; // the accumulator runs in f32, by design

/// `AᵀA/N` against a scalar reference computed from the same numbers.
#[test]
fn hessian_matches_a_direct_computation() {
    let dev = Device::Cpu;
    let (rows, n) = (37usize, 11usize);
    // Deterministic, and deliberately not centred: a bug that subtracts a
    // mean would pass on centred data.
    let a: Vec<f32> = (0..rows * n)
        .map(|i| ((i * 7919 % 101) as f32 / 50.0) - 0.7)
        .collect();

    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    let t = Tensor::from_slice(&a, (1, rows, n), &dev).expect("tensor");
    acc.accumulate(&t).expect("accumulate");
    let got = acc.to_f64().expect("readback");

    for i in 0..n {
        for j in 0..n {
            let want: f64 = (0..rows)
                .map(|r| a[r * n + i] as f64 * a[r * n + j] as f64)
                .sum::<f64>()
                / rows as f64;
            assert!(
                (got[i * n + j] - want).abs() <= TOL * want.abs().max(1.0),
                "H[{i},{j}] = {} vs {want}",
                got[i * n + j]
            );
        }
    }
}

/// Splitting the same rows across several calls must give the same Hessian.
///
/// This is what the real loop does — one call per calibration window — and it
/// is where a per-call rather than per-corpus normalization would hide.
#[test]
fn accumulation_is_independent_of_how_rows_are_batched() {
    let dev = Device::Cpu;
    let (rows, n) = (48usize, 9usize);
    let a: Vec<f32> = (0..rows * n)
        .map(|i| ((i * 31 % 17) as f32 / 4.0) - 2.0)
        .collect();

    let mut one = Hessian::new(n, &dev, rows).expect("alloc");
    one.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let whole = one.to_f64().expect("readback");

    let mut split = Hessian::new(n, &dev, rows).expect("alloc");
    for chunk in a.chunks(12 * n) {
        let r = chunk.len() / n;
        split
            .accumulate(&Tensor::from_slice(chunk, (1, r, n), &dev).expect("t"))
            .expect("acc");
    }
    let pieces = split.to_f64().expect("readback");

    for (k, (w, p)) in whole.iter().zip(pieces.iter()).enumerate() {
        assert!(
            (w - p).abs() <= TOL * w.abs().max(1.0),
            "entry {k}: one shot {w} vs batched {p} — the normalization is \
             per-call where it must be per-corpus"
        );
    }
}

/// The result must be symmetric positive semi-definite, and full rank once
/// there are more rows than columns — otherwise the Cholesky downstream fails
/// for reasons that have nothing to do with the model.
#[test]
fn hessian_is_symmetric_and_factorizable() {
    let dev = Device::Cpu;
    let (rows, n) = (200usize, 16usize);
    let a: Vec<f32> = (0..rows * n)
        .map(|i| (((i * 2654435761usize) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    acc.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let h = acc.to_f64().expect("readback");

    for i in 0..n {
        for j in 0..n {
            assert!(
                (h[i * n + j] - h[j * n + i]).abs() <= 1e-9,
                "H must be symmetric at ({i},{j})"
            );
        }
        assert!(h[i * n + i] >= 0.0, "diagonal {i} must be non-negative");
    }
    llvq_quant::linalg::GptqFactor::new(&h, n, 1e-2)
        .expect("a well-fed Hessian must factor with standard damping");
}

/// A dead input channel — a column that never fires — makes `H` singular.
/// Damping must rescue it, because real calibration sets do contain them.
#[test]
fn damping_rescues_a_dead_input_channel() {
    let dev = Device::Cpu;
    let (rows, n) = (128usize, 12usize);
    let mut a: Vec<f32> = (0..rows * n)
        .map(|i| (((i * 7919) % 97) as f32 / 48.0) - 1.0)
        .collect();
    for r in 0..rows {
        a[r * n + 5] = 0.0; // channel 5 never fires
    }
    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    acc.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let h = acc.to_f64().expect("readback");

    assert!(
        llvq_quant::linalg::GptqFactor::new(&h, n, 0.0).is_err(),
        "without damping a dead channel must be reported, not silently \
         factored — that is the signal the calibration set is wrong"
    );
    llvq_quant::linalg::GptqFactor::new(&h, n, 1e-2)
        .expect("standard damping must make it factorable");
}
