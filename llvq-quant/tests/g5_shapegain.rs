//! # Gate G5 — the gain code
//!
//! The defect this suite exists for: a "shape–gain" quantizer that keeps the
//! block magnitude as a free float is not a 2 bit/weight quantizer, however
//! exactly its direction costs 48 bits per 24 weights. That error survived a
//! full 4B run and a published figure, because nothing asserted that the
//! magnitude was *quantized*. It does now.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::{
    fit_gain_centroids, row_scale, BlockQuantizer, LeechDirection, LeechShapeGain,
};

fn random_matrix(rng: &mut SplitMix64, d_out: usize, d_in: usize) -> Vec<f64> {
    // Rows deliberately at very different scales — that is what a per-row
    // reference is for, and a bug that ignores it shows up here.
    (0..d_out * d_in)
        .map(|k| {
            let row = k / d_in;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

/// The whole point: a reconstructed block's magnitude must be one of the
/// finitely many levels the code can express, not the input's own norm.
#[test]
fn the_gain_is_actually_quantized() {
    let mut rng = SplitMix64::new(0x6_0001);
    let (d_out, d_in) = (8usize, 4 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);
    assert_eq!(centroids.len(), 2, "one gain bit means two levels");

    let mut q = LeechShapeGain::new(centroids.clone());
    assert_eq!(q.gain_bits(), 1);
    let mut out = vec![0.0f64; DIM];
    let mut distinct: Vec<f64> = Vec::new();

    for i in 0..d_out {
        let row = &w[i * d_in..(i + 1) * d_in];
        let rs = row_scale(row);
        q.set_row_scale(rs);
        for b in row.chunks_exact(DIM) {
            q.quantize(b, &mut out);
            let n = out.iter().map(|a| a * a).sum::<f64>().sqrt();
            // The magnitude must be a level times the row scale.
            let rel = n / rs;
            assert!(
                centroids.iter().any(|c| (c - rel).abs() < 1e-9),
                "block magnitude {rel} is not one of the {} levels {centroids:?} \
                 — the gain is not being quantized",
                centroids.len()
            );
            if !distinct.iter().any(|d| (d - rel).abs() < 1e-9) {
                distinct.push(rel);
            }
        }
    }
    assert!(
        distinct.len() > 1,
        "every block landed on the same level; the code is degenerate"
    );
}

/// And the control: the direction-only quantizer does *not* quantize the
/// magnitude. If this ever starts passing the assertion above, the two types
/// have been confused.
#[test]
fn direction_only_keeps_the_magnitude_free() {
    let mut rng = SplitMix64::new(0x6_0002);
    let mut q = LeechDirection::new();
    let mut out = vec![0.0f64; DIM];
    let mut norms = Vec::new();
    for _ in 0..32 {
        let v: Vec<f64> = (0..DIM).map(|_| rng.next_gaussian()).collect();
        q.quantize(&v, &mut out);
        let got = out.iter().map(|a| a * a).sum::<f64>().sqrt();
        let want = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!((got - want).abs() < 1e-9 * want, "magnitude must pass through");
        norms.push(got);
    }
    // 32 free floats, not a handful of levels.
    norms.sort_unstable_by(f64::total_cmp);
    norms.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    assert!(norms.len() > 16, "these should be free magnitudes, got {} levels", norms.len());
}

/// More gain bits must reduce reconstruction error, and zero bits must
/// collapse to a single level — the paper's true "zero gain bits".
#[test]
fn error_decreases_with_gain_bits() {
    let mut rng = SplitMix64::new(0x6_0003);
    let (d_out, d_in) = (6usize, 6 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);

    let err = |k: u32| -> f64 {
        let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, k, 40);
        assert_eq!(centroids.len(), 1usize << k);
        let mut q = LeechShapeGain::new(centroids);
        let mut out = vec![0.0f64; DIM];
        let mut e = 0.0;
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            q.set_row_scale(row_scale(row));
            for b in row.chunks_exact(DIM) {
                q.quantize(b, &mut out);
                e += b.iter().zip(out.iter()).map(|(a, c)| (a - c) * (a - c)).sum::<f64>();
            }
        }
        e
    };

    let (e0, e1, e2) = (err(0), err(1), err(2));
    assert!(e1 < e0, "one gain bit must beat zero: {e1} vs {e0}");
    assert!(e2 < e1, "two must beat one: {e2} vs {e1}");
}

/// The per-row scale is load-bearing: feed a wrong one and the gain code,
/// which only has a handful of levels, stops being able to express the block.
#[test]
fn the_row_scale_is_load_bearing() {
    let mut rng = SplitMix64::new(0x6_0004);
    let (d_out, d_in) = (6usize, 4 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);

    let err = |use_row_scale: bool| -> f64 {
        let mut q = LeechShapeGain::new(centroids.clone());
        let mut out = vec![0.0f64; DIM];
        let mut e = 0.0;
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            q.set_row_scale(if use_row_scale { row_scale(row) } else { 1.0 });
            for b in row.chunks_exact(DIM) {
                q.quantize(b, &mut out);
                e += b.iter().zip(out.iter()).map(|(a, c)| (a - c) * (a - c)).sum::<f64>();
            }
        }
        e
    };
    let (with, without) = (err(true), err(false));
    assert!(
        with < without * 0.5,
        "a per-row reference must matter a lot across rows spanning decades: \
         {with} vs {without}"
    );
}
