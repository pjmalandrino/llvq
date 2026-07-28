//! # Gate G4 — Gaussian source at 2 bits/dim (paper Table 3)
//!
//! Codebook: ball Λ₂₄(13) ∪ {0} — N(13)+1 points, 1.9999 bits/dim.
//! Reference run (20 000 eval blocks, seed 0x64A4_2026):
//!
//! | method | bits/dim | MSE | retention |
//! |---|---|---|---|
//! | paper Table 3, spherical shaping | 2.000 | — | 89.14 % |
//! | paper Table 3, shape–gain | 2.000 | — | 92.11 % |
//! | LLVQ spherical shaping (β* = 0.350) | 1.9999 | 0.0775 | **92.23 %** |
//! | LLVQ shape–gain, 2-bit gain | 2.0832 | 0.0670 | 93.62 % |
//! | Shannon | 2.000 | 0.0625 | 100 % |
//!
//! Caveat on the paper values: the PDF text extraction had a font-shifted
//! digit encoding, and the decoded Table 3 MSE column (0.1084/0.1078) is
//! **inconsistent** with its own SQNR column (1.798/1.849 ⇒ MSE ≈
//! 0.0845/0.0778); the retention column (89.14 % / 92.11 %) is the
//! self-consistent anchor used here. Our β-fitted spherical shaping meets
//! the paper's best (shape–gain) retention; the measurement cannot
//! overstate quality — every per-shell dot is achieved by a materialized,
//! membership-verified codebook point, so any engine bug would only lower
//! the figures.

use llvq_bench::*;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

#[cfg_attr(debug_assertions, ignore = "release-only: ~10k generic searches")]
#[test]
fn gaussian_retention_at_two_bits_per_dim() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x64A4_0013);
    let train: Vec<_> = (0..2_000).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..8_000).map(|_| gauss_block(&mut rng)).collect();
    let train13 = precompute13(&s, &train);
    let eval13 = precompute13(&s, &eval);

    let r = rate_spherical13();
    assert!((r - 2.0).abs() < 1e-3, "rate must be ≈ 2.000 bits/dim");

    let beta = optimize_beta13(&train13, 0.2, 0.9, 140);
    assert!((0.25..0.5).contains(&beta), "β* = {beta} outside plausible window");
    let mse = spherical_mse13(&eval13, beta);
    let shannon = 2f64.powf(-2.0 * r);
    assert!(mse > shannon, "MSE {mse} beats Shannon {shannon}: measurement bug");

    let ret = retention_pct(mse, r);
    assert!(
        ret > 89.14,
        "retention {ret:.2}% below the paper's spherical shaping (89.14%)"
    );
    assert!(
        ret > 91.0,
        "retention {ret:.2}% regressed from the reference run (92.23%)"
    );

    // Shape–gain with 2-bit gain must further reduce MSE.
    let train_t: Vec<f64> = train13.iter().map(BlockDots13::t).collect();
    let centroids = lloyd_max(&train_t, 2, 60);
    let mse_sg = shape_gain_mse13(&eval13, &centroids);
    assert!(mse_sg < mse, "2-bit gain ({mse_sg}) must improve on spherical ({mse})");
    assert!(mse_sg > shannon * 0.9, "shape–gain MSE implausibly low");

    // The 2-bit point must dominate the 1-bit point of g4_prelim.
    assert!(mse < 0.15, "2 bits/dim must be far below the 1-bit MSE (0.2865)");
}
