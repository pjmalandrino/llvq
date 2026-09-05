//! # The Trio encoder against the bench
//!
//! Two claims of `llvq_search::trio::Encoder`, each against the bench in the
//! same process: at one scale it returns the points of the bench's
//! rank-region codebook (`llvq_bench::f1::rankbook`, the copy of
//! `examples/f1rankbench.rs`), and on the fixed 2,000 evaluation blocks of the
//! F1b seed its retention does not regress. Both are release-only: the bench
//! spends ~20 ms per (block, scale) and the control searches twelve shells.
//!
//! The blocks are the journals': drawn by `gauss_block` from seed
//! `0x0f1b_2026_0904` after 4,000 training draws, handed to the bench as
//! drawn (it reads its input in trio order) and to the production encoder in
//! the natural frame whose trio-order view is that draw.

use llvq_bench::f1::point_to_natural;
use llvq_bench::f1::rank::RankTable;
use llvq_bench::f1::rankbook::{mse_shape_gain, t_ball12, Codebook, RankRegion};
use llvq_bench::{gauss_block, lloyd_max, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::trio::encoder::t_of;
use llvq_search::trio::{Encoder, Scratch, Trio};
use llvq_search::Searcher;

const SEED: u64 = 0x0f1b_2026_0904;
const N_TRAIN: usize = 4_000;
const N_EVAL: usize = 2_000;

/// The training draws (for the gain centroids) and the evaluation draws.
fn blocks() -> (Vec<[f64; DIM]>, Vec<[f64; DIM]>) {
    let mut rng = SplitMix64::new(SEED);
    let train = (0..N_TRAIN).map(|_| gauss_block(&mut rng)).collect();
    let eval = (0..N_EVAL).map(|_| gauss_block(&mut rng)).collect();
    (train, eval)
}

/// The natural-order block whose trio-order view is `raw`.
fn natural_frame(raw: &[f64; DIM], order: &[u32; DIM]) -> [f64; DIM] {
    let mut nat = [0.0f64; DIM];
    for (j, &v) in raw.iter().enumerate() {
        nat[order[j] as usize] = v;
    }
    nat
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The third scale of the yardstick test: `s₀/STRESS`, where the target is
/// far enough out that the shrink-1.0 base is seldom a member and the
/// re-roundings, the later shrinks and the fallbacks decide — the part of
/// the rule the adaptive scales seldom reach (at `s₀` the winning path is
/// nearly always three closed bases, so a mutant of the re-rounding rule
/// survived 4,000 pairs at `s₀` and `RATIO·s₀` alone).
const STRESS: f64 = 2.5;

/// (1) At the two scales of `encode` and at the stress scale, the production
/// encoder returns the bench's point — or a point at the same cost to 1e-9,
/// a tie the bench's own iteration order would break either way. Every pair
/// is checked; the tie count is printed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn encode_at_scale_returns_the_bench_points_on_the_evaluation_blocks() {
    let (_, eval) = blocks();
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let order = *trio.order();
    let cb: Codebook<RankRegion> = Codebook::rank(&RankTable::build());
    let sqrt_dim = (DIM as f64).sqrt();

    let chunk = eval.len().div_ceil(threads());
    let counts: Vec<(usize, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = eval
            .chunks(chunk)
            .map(|blocks| {
                let (enc, cb) = (&enc, &cb);
                s.spawn(move || {
                    let mut sc = Scratch::new();
                    let (mut same, mut ties) = (0usize, 0usize);
                    for raw in blocks {
                        let nat = natural_frame(raw, &order);
                        let s0 = Encoder::ALPHA * raw.iter().map(|v| v * v).sum::<f64>().sqrt() / sqrt_dim;
                        for s in [s0, s0 * Encoder::RATIO, s0 / STRESS] {
                            let bench = point_to_natural(&cb.encode_at_scale(raw, s), &order);
                            let ours = enc.encode_at_scale(&nat, s, &mut sc);
                            if ours.point == bench {
                                same += 1;
                                continue;
                            }
                            let cost = |y: &[i32; DIM]| y.iter().zip(&nat).map(|(&v, &x)| (v as f64 - x / s).powi(2)).sum::<f64>();
                            let (cb, co) = (cost(&bench), cost(&ours.point));
                            assert!((cb - co).abs() <= 1e-9 * cb.max(1.0), "s = {s:.4}: bench {bench:?} at {cb} against {:?} at {co}", ours.point);
                            ties += 1;
                        }
                    }
                    (same, ties)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });
    let (same, ties) = counts.iter().fold((0, 0), |(a, b), &(c, d)| (a + c, b + d));
    assert_eq!(same + ties, 3 * N_EVAL);
    println!("{same} (block, scale) pairs at the bench's point, {ties} at a different point of equal cost");
}

/// (4) Retention on the FIXED 2,000 evaluation blocks with the F1b rule — one
/// gain bit on the norm, Lloyd-Max centroids from the 4,000 training norms —
/// against the ball-12 control in the same process: at least 88.85 %, the
/// prototype's 88.89 less the room a pinning on training blocks may take.
/// A non-regression test on fixed blocks, NOT a quality claim: Gaussian
/// retention is two transpositions away from perplexity (`docs/ROADMAP.md`
/// §2.2 bis), and the control's own 92.00 is the journal's.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn retention_on_the_fixed_blocks_does_not_regress() {
    let (train, eval) = blocks();
    let rate = 48.0 / DIM as f64;
    let norms: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);
    let xx: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();

    let s = Searcher::new();
    let dots = precompute13(&s, &eval);
    let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
    let ctrl = retention_pct(mse_shape_gain(&xx, &t_ctrl, &centroids), rate);
    assert!((ctrl - 92.00).abs() < 0.05, "the ball-12 control reads {ctrl:.2} %, the journal's is 92.00");

    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let order = *trio.order();
    let chunk = eval.len().div_ceil(threads());
    let t_trio: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = eval
            .chunks(chunk)
            .map(|blocks| {
                let enc = &enc;
                s.spawn(move || {
                    let mut sc = Scratch::new();
                    blocks
                        .iter()
                        .map(|raw| {
                            let nat = natural_frame(raw, &order);
                            let code = enc.encode(&nat, &mut sc);
                            assert_eq!(code.t, t_of(&nat, &code.point));
                            code.t
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("thread")).collect()
    });
    let trio_pct = retention_pct(mse_shape_gain(&xx, &t_trio, &centroids), rate);
    println!("ball-12 control {ctrl:.2} %, Trio {trio_pct:.2} %, Δ {:+.2} pp on {N_EVAL} fixed blocks", trio_pct - ctrl);
    assert!(trio_pct >= 88.85, "Trio retention {trio_pct:.2} % is under 88.85 on the fixed blocks (control {ctrl:.2})");
}
