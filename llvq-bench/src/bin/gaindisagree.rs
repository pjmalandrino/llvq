//! Stage 0 of the Tetra gain-decision plan: how often the served norm rule and
//! the Euclidean optimum choose a different gain, and what the norm rule costs
//! when they differ.
//!
//! `cargo run --release -p llvq-bench --bin gaindisagree`
//!
//! ## What it compares
//!
//! A Tetra block decodes to `c_g · row_scale · u`, with `u` the unit direction
//! of the encoded point and `c_g` one of the two fitted centroids
//! (`reconstruct_shape_gain`). The served rule picks the centroid nearest
//! `‖x‖ / row_scale` (`TetraShapeGain::quantize`). The Euclidean optimum over
//! the same two centroids is the one nearest `⟨x,u⟩ / row_scale`, since
//! `‖x − a u‖² = ‖x‖² − 2a⟨x,u⟩ + a²` is minimized by `a* = ⟨x,u⟩`. With two
//! admissible levels the optimum is the oracle, so the second rule's regret is
//! zero by construction and the first rule's regret is exact.
//!
//! `⟨x,u⟩ = ‖x‖ cos θ ≤ ‖x‖`, so the two rules differ exactly when the
//! midpoint of the centroids falls between the two, and the shift is one
//! sided: the Euclidean rule can only move a block down a level.
//!
//! ## What it does not measure
//!
//! The blocks are Gaussian draws, not GPTQ residues. Production hands the
//! quantizer a block already rewritten by the error feedback of every earlier
//! column, against a row scale fitted before that feedback ran. The rotated
//! blocks of a real matrix have kurtosis 3.01 (*measured*,
//! `docs/ROADMAP-QUALITY.md` row 8), which is what justifies a Gaussian source
//! for the angular part of this question and not for the trajectory part.
//!
//! The conditional (Schur) rule is absent here. It needs a real Hessian factor
//! and therefore a model run.
//!
//! ## Protocol
//!
//! Seed, block source and draw order are the F1b ones
//! (`llvq-bench/tests/tetra_encoder.rs`), so the block population is the one
//! the 88.89 % retention figure was read on. Rows carry 106 blocks, the count
//! of full blocks in a 2560-wide Qwen3-4B matrix; the training draws feed
//! `fit_gain_centroids` exactly as `calib.rs` calls it, and the evaluation
//! draws are disjoint. Retention here is a within-protocol control for the
//! paired delta and is not the published 88.89 %, which uses the bench's own
//! frame and gain handling.

use llvq_bench::{gauss_block, lloyd_max, retention_pct};
use llvq_core::{Leech, SplitMix64, DIM};
use llvq_quant::quantizer::{
    fit_gain_centroids, reconstruct_shape_gain, row_scale, BlockCode, BlockQuantizer,
    TetraShapeGain,
};

/// The F1b seed, so the blocks are the journals'.
const SEED: u64 = 0x0f1b_2026_0904;
/// Full blocks in a 2560-wide matrix, the Qwen3-4B attention width.
const BLOCKS_PER_ROW: usize = 106;
const ROWS_TRAIN: usize = 40;
const ROWS_EVAL: usize = 400;
/// The rate a 48-bit word over 24 coordinates is charged at.
const RATE_BITS_PER_DIM: f64 = 2.0;

fn draw_matrix(rng: &mut SplitMix64, rows: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(rows * BLOCKS_PER_ROW * DIM);
    for _ in 0..rows * BLOCKS_PER_ROW {
        w.extend_from_slice(&gauss_block(rng));
    }
    w
}

/// Quantiles of an unsorted sample, at the fractions of `QS`.
const QS: [f64; 9] = [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0];

fn quantiles(v: &mut [f64]) -> [f64; 9] {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    QS.map(|q| {
        let i = ((q * (n - 1) as f64).round() as usize).min(n - 1);
        v[i]
    })
}

fn print_quantiles(label: &str, q: &[f64; 9]) {
    println!(
        "{label:<22} min {:.4}  p1 {:.4}  p10 {:.4}  p25 {:.4}  med {:.4}  p75 {:.4}  p90 {:.4}  p99 {:.4}  max {:.4}",
        q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]
    );
}

/// One evaluated block: the two candidate costs and what each rule picked.
struct Verdict {
    cos: f64,
    relative_norm: f64,
    gain_served: u32,
    gain_euclid: u32,
    err_served: f64,
    err_euclid: f64,
    /// `‖x‖ / row_scale` and `⟨x,u⟩ / row_scale`, the statistics the two rules
    /// compare against the centroids, kept so a refit can be scored on them.
    target_norm: f64,
    target_euclid: f64,
    /// `‖x‖²`, so a refitted centroid pair can be scored without re-encoding.
    nx2: f64,
    /// `⟨x,u⟩`, same reason.
    t: f64,
    row_scale: f64,
}

/// Squared error of reconstructing this block at gain `a = c · row_scale`.
fn err_at(v: &Verdict, c: f64) -> f64 {
    let a = c * v.row_scale;
    v.nx2 - 2.0 * a * v.t + a * a
}

/// Nearest of two centroids to `x`.
fn nearest2(centroids: &[f64], x: f64) -> usize {
    usize::from((centroids[1] - x).abs() < (centroids[0] - x).abs())
}

/// Mean squared error per dimension of one (rule, centroid pair) arm.
///
/// `euclid_rule` picks the level nearest `⟨x,u⟩ / row_scale`, the served rule
/// the level nearest `‖x‖ / row_scale`. The centroids are whatever the caller
/// fitted; the rate is the same 48-bit word either way.
fn arm_mse(verdicts: &[Verdict], centroids: &[f64], euclid_rule: bool) -> f64 {
    let total: f64 = verdicts
        .iter()
        .map(|v| {
            let stat = if euclid_rule {
                v.target_euclid
            } else {
                v.target_norm
            };
            err_at(v, centroids[nearest2(centroids, stat)])
        })
        .sum();
    total / (verdicts.len() * DIM) as f64
}

fn evaluate(x: &[f64], point: &llvq_core::Point, gain_served: u32, centroids: &[f64], rs: f64) -> Option<Verdict> {
    let m = Leech::shell_index(point)?;
    if m == 0 {
        return None;
    }
    let nx2: f64 = x.iter().map(|a| a * a).sum();
    let nx = nx2.sqrt();
    if nx == 0.0 {
        return None;
    }
    // `u = p / √(16m)` is the unit direction the decoder reconstructs along.
    let norm_p = ((16 * m) as f64).sqrt();
    let t: f64 = x
        .iter()
        .zip(point.iter())
        .map(|(&a, &p)| a * p as f64)
        .sum::<f64>()
        / norm_p;
    let err = |g: usize| {
        let a = centroids[g] * rs;
        nx2 - 2.0 * a * t + a * a
    };
    // Two levels, so the Euclidean argmin is the centroid nearest `t / rs`.
    let gain_euclid = if (centroids[0] - t / rs).abs() <= (centroids[1] - t / rs).abs() {
        0u32
    } else {
        1u32
    };
    Some(Verdict {
        cos: t / nx,
        relative_norm: nx / rs,
        gain_served,
        gain_euclid,
        err_served: err(gain_served as usize),
        err_euclid: err(gain_euclid as usize),
        target_norm: nx / rs,
        target_euclid: t / rs,
        nx2,
        t,
        row_scale: rs,
    })
}

/// Encode every block of a matrix and return the two per-block statistics the
/// centroid fits read: `‖x‖ / row_scale` and `⟨x,u⟩ / row_scale`.
fn fit_targets(w: &[f64], d_in: usize, centroids: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut q = TetraShapeGain::new(centroids.to_vec());
    let mut out = vec![0.0f64; DIM];
    let mut by_norm = Vec::new();
    let mut by_euclid = Vec::new();
    for row in w.chunks_exact(d_in) {
        let rs = row_scale(row);
        q.set_row_scale(rs);
        for x in row.chunks_exact(DIM) {
            q.quantize(x, &mut out);
            let code = q.last_code().expect("Tetra emits a code for every block");
            if let Some(v) = evaluate(x, &code.point, code.gain, centroids, rs) {
                by_norm.push(v.target_norm);
                by_euclid.push(v.target_euclid);
            }
        }
    }
    (by_norm, by_euclid)
}

fn main() {
    let mut rng = SplitMix64::new(SEED);
    let train = draw_matrix(&mut rng, ROWS_TRAIN);
    let eval = draw_matrix(&mut rng, ROWS_EVAL);
    let d_in = BLOCKS_PER_ROW * DIM;

    let centroids = fit_gain_centroids(&train, ROWS_TRAIN, d_in, DIM, 1, 40);
    assert_eq!(centroids.len(), 2, "one gain bit");

    println!("=== gaindisagree ===");
    println!("seed            0x{SEED:016x}  (F1b)");
    println!("train           {ROWS_TRAIN} rows x {BLOCKS_PER_ROW} blocks");
    println!("eval            {ROWS_EVAL} rows x {BLOCKS_PER_ROW} blocks");
    println!("centroids       {:.6}  {:.6}", centroids[0], centroids[1]);
    println!(
        "midpoint        {:.6}",
        0.5 * (centroids[0] + centroids[1])
    );

    let t0 = std::time::Instant::now();
    let mut q = TetraShapeGain::new(centroids.clone());
    let mut out = vec![0.0f64; DIM];
    let mut verdicts: Vec<Verdict> = Vec::with_capacity(ROWS_EVAL * BLOCKS_PER_ROW);
    let mut skipped = 0usize;
    // The cost model above is algebra. The decoder is the authority, so every
    // block's two candidate costs are checked against `reconstruct_shape_gain`
    // rebuilding the block exactly as an artifact would.
    let mut worst_gap = 0.0f64;
    let mut alt = vec![0.0f64; DIM];
    for row in eval.chunks_exact(d_in) {
        let rs = row_scale(row);
        q.set_row_scale(rs);
        for x in row.chunks_exact(DIM) {
            q.quantize(x, &mut out);
            let code = q.last_code().expect("Tetra emits a code for every block");
            let Some(v) = evaluate(x, &code.point, code.gain, &centroids, rs) else {
                skipped += 1;
                continue;
            };
            for g in 0..2u32 {
                let decoded = if g == code.gain {
                    &out
                } else {
                    let other = BlockCode { point: code.point, gain: g };
                    reconstruct_shape_gain(&other, &centroids, rs, &mut alt);
                    &alt
                };
                let direct: f64 = x
                    .iter()
                    .zip(decoded.iter())
                    .map(|(&a, &d)| (a - d) * (a - d))
                    .sum();
                worst_gap = worst_gap.max((direct - err_at(&v, centroids[g as usize])).abs());
            }
            verdicts.push(v);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let n = verdicts.len();
    println!("encoded         {n} blocks in {secs:.1} s, {skipped} at the origin");
    println!("decoder check   worst |algebra - reconstruct_shape_gain| over {} costs: {worst_gap:.3e}", 2 * n);
    assert!(
        worst_gap < 1e-9,
        "the cost model disagrees with the decoder by {worst_gap:.3e}"
    );
    println!();

    // ---- the headline: how often do the rules differ, and which way ----
    let disagree: Vec<&Verdict> = verdicts
        .iter()
        .filter(|v| v.gain_served != v.gain_euclid)
        .collect();
    let down = disagree
        .iter()
        .filter(|v| v.gain_euclid < v.gain_served)
        .count();
    println!("--- disagreement ---");
    println!(
        "served vs euclid  {} of {} blocks, {:.3} %",
        disagree.len(),
        n,
        100.0 * disagree.len() as f64 / n as f64
    );
    println!(
        "  euclid picks lower level  {down}   higher level  {}",
        disagree.len() - down
    );
    let occ_served = verdicts.iter().filter(|v| v.gain_served == 1).count();
    let occ_euclid = verdicts.iter().filter(|v| v.gain_euclid == 1).count();
    println!(
        "gain occupancy    served {:.3} % at level 1, euclid {:.3} %",
        100.0 * occ_served as f64 / n as f64,
        100.0 * occ_euclid as f64 / n as f64
    );
    println!();

    // ---- what the disagreement costs ----
    let sum_served: f64 = verdicts.iter().map(|v| v.err_served).sum();
    let sum_euclid: f64 = verdicts.iter().map(|v| v.err_euclid).sum();
    let mse_served = sum_served / (n * DIM) as f64;
    let mse_euclid = sum_euclid / (n * DIM) as f64;
    println!("--- cost of the served rule ---");
    println!("mse/dim           served {mse_served:.6}   euclid {mse_euclid:.6}");
    println!(
        "squared error      served rule wastes {:.4} % of its own total",
        100.0 * (sum_served - sum_euclid) / sum_served
    );
    println!(
        "retention at {RATE_BITS_PER_DIM:.3} b/dim   served {:.4} %   euclid {:.4} %   delta {:+.4} pp",
        retention_pct(mse_served, RATE_BITS_PER_DIM),
        retention_pct(mse_euclid, RATE_BITS_PER_DIM),
        retention_pct(mse_euclid, RATE_BITS_PER_DIM) - retention_pct(mse_served, RATE_BITS_PER_DIM)
    );
    if !disagree.is_empty() {
        let mut reg: Vec<f64> = disagree
            .iter()
            .map(|v| v.err_served - v.err_euclid)
            .collect();
        let mean_reg = reg.iter().sum::<f64>() / reg.len() as f64;
        let qr = quantiles(&mut reg);
        println!("regret on disagreed blocks, squared error per block");
        println!("  mean            {mean_reg:.6}");
        print_quantiles("  quantiles", &qr);
        let mean_block_err = sum_served / n as f64;
        println!(
            "  mean regret is {:.3} % of the mean block error ({mean_block_err:.4})",
            100.0 * mean_reg / mean_block_err
        );
    }
    println!();

    // ---- the 2x2: the rule and the centroids are separate mechanisms ----
    //
    // The encoded direction depends on `x` alone, never on the gain, so the
    // per-block statistics above are valid for any centroid pair and a refit
    // needs no second encoding pass over the evaluation blocks.
    let (train_norm, train_euclid) = fit_targets(&train, d_in, &centroids);
    let refit = lloyd_max(&train_euclid, 1, 40);
    println!("--- centroids ---");
    println!(
        "fitted on norms   {:.6}  {:.6}   (the served fit, {} training blocks)",
        centroids[0],
        centroids[1],
        train_norm.len()
    );
    println!(
        "fitted on <x,u>   {:.6}  {:.6}   (same blocks, same rate, same format)",
        refit[0], refit[1]
    );
    // The refit turns out to be close to a single scalar shrink of the served
    // pair, so the shrink is scored as its own arm: it costs one multiply at
    // fit time, where the refit costs an encoding pass over the matrix.
    let mean_cos_train = {
        let (n, e) = (&train_norm, &train_euclid);
        e.iter().zip(n).map(|(a, b)| a / b).sum::<f64>() / n.len() as f64
    };
    let shrunk: Vec<f64> = centroids.iter().map(|c| c * mean_cos_train).collect();
    println!(
        "shrunk by mean cos {:.6}  {:.6}   (served pair x {mean_cos_train:.6}, one multiply)",
        shrunk[0], shrunk[1]
    );
    let arms = [
        ("served rule, served centroids", arm_mse(&verdicts, &centroids, false)),
        ("euclid rule, served centroids", arm_mse(&verdicts, &centroids, true)),
        ("served rule, refit centroids", arm_mse(&verdicts, &refit, false)),
        ("served rule, shrunk centroids", arm_mse(&verdicts, &shrunk, false)),
        ("euclid rule, refit centroids", arm_mse(&verdicts, &refit, true)),
        ("euclid rule, shrunk centroids", arm_mse(&verdicts, &shrunk, true)),
    ];
    let base = arms[0].1;
    println!();
    println!("--- 2x2, mse per dimension and retention at 2.000 b/dim ---");
    for (name, mse) in arms {
        println!(
            "{name:<32} mse {mse:.6}   retention {:.4} %   delta {:+.4} pp   error {:+.4} %",
            retention_pct(mse, RATE_BITS_PER_DIM),
            retention_pct(mse, RATE_BITS_PER_DIM) - retention_pct(base, RATE_BITS_PER_DIM),
            100.0 * (mse - base) / base
        );
    }
    println!();

    // ---- the geometry that sets the band ----
    let mut cos: Vec<f64> = verdicts.iter().map(|v| v.cos).collect();
    let mut rel: Vec<f64> = verdicts.iter().map(|v| v.relative_norm).collect();
    let qc = quantiles(&mut cos);
    let qn = quantiles(&mut rel);
    println!("--- geometry ---");
    print_quantiles("cos(x, u)", &qc);
    print_quantiles("norm / row_scale", &qn);
    println!(
        "mean cos          {:.6}   so the euclidean target sits {:.3} % below the norm",
        verdicts.iter().map(|v| v.cos).sum::<f64>() / n as f64,
        100.0 * (1.0 - verdicts.iter().map(|v| v.cos).sum::<f64>() / n as f64)
    );
}
