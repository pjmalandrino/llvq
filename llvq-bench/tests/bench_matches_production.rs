//! The bench must measure the quantizer that ships. Nothing else.
//!
//! ## The gap this closes
//!
//! `llvq-bench` and `bin/lcap` scored shape–gain by rounding the projection
//! `t = max_m d_m/√(16m)`, which minimizes `‖x − g·v̂‖²` at fixed direction.
//! `LeechShapeGain` rounds the block **norm** and fits its centroids on block
//! norms. Same reconstruction, same error formula, different scalar going into
//! the gain code — and the projection version is strictly the better
//! quantizer, by `2/(1 + cos θ)` per block.
//!
//! So gate G4's 92.23 %, the whole G4 table, and every level-cap row in
//! `docs/archive/face-au-4-bits.md` described a codebook that never ran on a model,
//! while format decisions were being taken off those tables. The gap is small
//! — about 2 % of block MSE, 0.7 points of retention — and it is exactly the
//! class of defect this repo has been bitten by three times: a property
//! believed to hold because nobody ever stated it.
//!
//! This file states it. `shape_gain_mse13_shipped` is pinned to
//! `LeechShapeGain::quantize` itself, block by block, so the bench and the
//! encoder cannot drift apart again without a test going red.

use llvq_bench::{
    gauss_block, lloyd_max, precompute13, shape_gain_mse13_projected, shape_gain_mse13_shipped,
    BlockDots13,
};
use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::{BlockQuantizer, LeechShapeGain};
use llvq_search::Searcher;

/// Enough blocks for the aggregate to be stable, few enough to stay quick in
/// debug — the per-block assertion is the strict one anyway.
const N: usize = 600;
const GAIN_BITS: u32 = 1;

#[test]
#[cfg_attr(debug_assertions, ignore = "ball-13 search, release only")]
fn the_bench_scores_what_leech_shape_gain_produces() {
    let mut rng = SplitMix64::new(0x000B_01CE);
    let blocks: Vec<[f64; DIM]> = (0..N).map(|_| gauss_block(&mut rng)).collect();
    let s = Searcher::new();
    let dots = precompute13(&s, &blocks);

    // One centroid set, handed to both sides: this test is about the *rule*,
    // not about whether two Lloyd–Max fits agree.
    let norms: Vec<f64> = dots.iter().map(BlockDots13::norm).collect();
    let centroids = lloyd_max(&norms, GAIN_BITS, 60);

    // Production. `row_scale` stays 1, so the gain sees the raw block norm —
    // the Gaussian source has no rows to normalize against.
    let mut q = LeechShapeGain::new(centroids.clone());
    let mut out = [0.0f64; DIM];
    let mut produced = 0.0f64;
    for (i, x) in blocks.iter().enumerate() {
        q.quantize(x, &mut out);
        let e2: f64 = x.iter().zip(out.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        // Per block, not just in aggregate: a rule that is wrong on a third of
        // the blocks and compensates on the rest would pass an aggregate test.
        let bench_e2 = {
            let d = &dots[i];
            let g = centroids[llvq_bench::nearest_centroid(&centroids, d.norm())];
            d.xx - 2.0 * g * d.t() + g * g
        };
        assert!(
            (e2 - bench_e2).abs() <= 1e-9 * e2.max(1e-12),
            "block {i}: production {e2:e}, bench {bench_e2:e}"
        );
        produced += e2;
    }
    let produced_mse = produced / (DIM * N) as f64;

    let shipped = shape_gain_mse13_shipped(&dots, &centroids);
    assert!(
        (shipped - produced_mse).abs() <= 1e-12 * produced_mse,
        "bench {shipped:e} against the encoder's own output {produced_mse:e}"
    );
}

/// The projection rule is a **bound**, and the bench must not be able to quote
/// it as a configuration again.
///
/// Rounding `t` lands on `argmin_g ‖x − g·v̂‖²`; rounding `‖x‖` overshoots it,
/// since `t = ‖x‖·cos θ ≤ ‖x‖`. So the ordering is not an empirical accident
/// and a violation means one of the two functions is mislabelled.
#[test]
#[cfg_attr(debug_assertions, ignore = "ball-13 search, release only")]
fn the_projection_rule_bounds_the_shipped_one_from_below() {
    let mut rng = SplitMix64::new(0x000B_01CF);
    let blocks: Vec<[f64; DIM]> = (0..N).map(|_| gauss_block(&mut rng)).collect();
    let dots = precompute13(&Searcher::new(), &blocks);

    let norms: Vec<f64> = dots.iter().map(BlockDots13::norm).collect();
    let projections: Vec<f64> = dots.iter().map(BlockDots13::t).collect();

    for k in [0u32, 1, 2] {
        let shipped = shape_gain_mse13_shipped(&dots, &lloyd_max(&norms, k, 60));
        let bound = shape_gain_mse13_projected(&dots, &lloyd_max(&projections, k, 60));
        assert!(
            bound < shipped,
            "{k}-bit gain: bound {bound:e} must sit below the shipped rule {shipped:e}"
        );
        // And the two must not be the *same* number — that is what a copy-paste
        // collapsing one onto the other would produce.
        assert!(
            (shipped - bound) / shipped > 1e-3,
            "{k}-bit gain: the two rules differ by {:.4} %, which is not a \
             substitution of ‖x‖ for ⟨x,v̂⟩",
            100.0 * (shipped - bound) / shipped
        );
    }
}
