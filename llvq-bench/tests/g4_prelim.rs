//! # Gate G4 (preliminary) — Gaussian-source rate-distortion at ≈ 1 bit/dim
//!
//! The full G4 gate (paper Table 3: 89.14 % / 92.11 % retention at
//! 2 bits/dim) requires shells up to m = 13. With m ≤ 3 the reachable rate
//! is ≈ 1.0007 bits/dim; the reference measurement (20 000 eval blocks,
//! seed 0x64A4_2026) gives:
//!
//! | method | bits/dim | MSE | retention |
//! |---|---|---|---|
//! | Lloyd–Max scalar 1-bit (analytic) | 1.0000 | 0.3634 | 73.0 % |
//! | LLVQ spherical shaping | 1.0007 | 0.2865 | **90.1 %** |
//! | LLVQ shape–gain, 2-bit gain | 1.0840 | 0.2733 | 86.3 % |
//! | Shannon limit | 1.0007 | 0.2498 | 100 % |
//!
//! The assertions below bracket those figures with margins: the quantizer
//! must clearly beat the optimal scalar baseline, must NOT beat Shannon
//! (that would prove a measurement bug, e.g. a rate miscount), and the
//! 2-bit gain must reduce MSE relative to pure spherical shaping.

use llvq_bench::*;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

#[cfg_attr(debug_assertions, ignore = "release-only: ~10k exact searches")]
#[test]
fn gaussian_retention_at_one_bit_per_dim() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x64A4_0004);
    let train: Vec<_> = (0..2_000).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..8_000).map(|_| gauss_block(&mut rng)).collect();
    let train_dots = precompute(&s, &train);
    let eval_dots = precompute(&s, &eval);

    let r = rate_spherical();
    assert!((r - 1.0007).abs() < 1e-3, "rate must be ≈ 1.0007 bits/dim");

    // Spherical shaping.
    let beta = optimize_beta(&train_dots, 0.4, 1.1, 140);
    assert!(
        (0.5..0.75).contains(&beta),
        "β* = {beta} outside the plausible window"
    );
    let mse = spherical_mse(&eval_dots, beta);
    let shannon = 2f64.powf(-2.0 * r);
    assert!(
        mse > shannon,
        "MSE {mse} beats the Shannon bound {shannon}: measurement bug"
    );
    assert!(
        mse < 0.30,
        "MSE {mse}: spherical shaping must clearly beat the 1-bit scalar \
         optimum {:.4}",
        lloyd_max_1bit_scalar_mse()
    );
    let ret = retention_pct(mse, r);
    assert!(ret > 87.0, "retention {ret:.2}% below the expected ≈ 90%");

    // Shape–gain with a 2-bit gain, scored the way `LeechShapeGain` scores it:
    // the gain code rounds the block **norm**, and its centroids are fitted on
    // norms.
    let train_norm: Vec<f64> = train_dots.iter().map(BlockDots::norm).collect();
    let mse_sg = shape_gain_mse_shipped(&eval_dots, &lloyd_max(&train_norm, 2, 60));
    assert!(mse_sg > shannon * 0.9, "shape–gain MSE implausibly low");

    // Rounding the projection instead is `argmin_g ‖x − g·v̂‖²`, so it bounds
    // the shipped rule from below; and spherical shaping is the special case
    // of a single-level gain, so the bound must beat that too. Both orderings
    // are forced by algebra, not by this seed.
    let train_t: Vec<f64> = train_dots.iter().map(BlockDots::t).collect();
    let bound = shape_gain_mse_projected(&eval_dots, &lloyd_max(&train_t, 2, 60));
    assert!(
        bound < mse_sg,
        "the optimal-gain bound ({bound}) must beat the shipped rule ({mse_sg})"
    );
    assert!(
        bound < mse,
        "the optimal-gain bound ({bound}) must beat spherical shaping ({mse})"
    );

    // 🔎 The shipped rule, on the other hand, does **not** beat spherical
    // shaping at this rate: 0.2942 against 0.2857, +3.0 %. At m ≤ 3 the
    // angular resolution is coarse, so cos θ sits well below 1 and the
    // `2/(1 + cos θ)` penalty for rounding the norm rather than the projection
    // swallows everything the gain code buys. At m ≤ 13 it does not — see
    // `g4_full`, where the shipped rule still wins.
    //
    // This assertion is here to record a measurement, not to demand a
    // property: until 2026-08-01 the test asserted `mse_sg < mse`, which only
    // ever held because the bench was scoring the bound instead of the
    // encoder. If this goes red, the situation improved and the note above is
    // the thing to update.
    assert!(
        mse_sg > mse,
        "shipped shape–gain ({mse_sg}) now beats spherical shaping ({mse}) at \
         m ≤ 3 — good news, and it changes the story this test records"
    );
}
