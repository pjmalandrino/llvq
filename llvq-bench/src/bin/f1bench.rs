//! F1b — Gaussian retention of the three-section codebook, against a ball-12
//! control measured in the same process on the same blocks.
//!
//! `cargo run --release -p llvq-bench --bin f1bench -- [n_train] [n_eval]`
//!
//! Protocol fixed by `proofs/preregistration-f1b-2026-09-04.md`, timestamped
//! before the first measurement. **F1b is a measurement, not a gate**: the
//! operator's rule of 2026-09-04 is that a gate reads a fundamental criterion,
//! and Gaussian retention is not one — it is two transpositions away from
//! quality. The number here feeds the signed prediction F1c is read against.
//!
//! ## The control, and why it is not the 92.14% everyone quotes
//!
//! 92.14% is the paper's Table 8, on its unrounded MSE 0.077718. This
//! repository has never measured it: no retention bench calls `set_shell_cap`,
//! and `main.rs` never runs the 1-bit gain arm. So the ball-12 row below is the
//! first in-repo measurement of that configuration, and the two numbers are
//! printed side by side so the harness offset is visible **before** the F1 row
//! is read. Without that, F1's number would be our encoder against theirs.
//!
//! The control is taken by restricting the shell reduction to m ≤ 12 rather
//! than by capping the searcher: `shell_bests` already returns the best point
//! of every shell, so the maximum over shells 2..=12 *is* the ball-12 answer.
//! It also sidesteps the trap an adversarial review named — `BallSearcher`
//! starts at cap 13, and a control that forgets to lower it is a 49-bit
//! codebook wearing a 48-bit label.
//!
//! ## Both sides at 2.000 b/dim, and the same everything else
//!
//! Ball-12 is 47 index bits plus one gain bit; F1 is `[state 8][s₁][s₂][s₃]`
//! summing to 47, plus one gain bit. Both are 48 bits over 24 dimensions.
//! Dividing by the 47-bit field alone would flatter F1 by 1.9 points, and
//! dividing by a fractional rate is the error that cost a retracted 92.24% on
//! 2026-08-04. The gain centroids are fitted on block **norms**, which do not
//! depend on the arm, so both rows share them by construction.

use llvq_bench::f1::Codebook;
use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

/// Shells 2..=12 of the ball, the control's codebook.
const BALL12_SHELLS: usize = 11;

/// `t = ⟨x, v̂⟩` restricted to shells 2..=12.
fn t_ball12(d: &[f64; 12]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

fn mse_shape_gain(xx: &[f64], t: &[f64], centroids: &[f64]) -> f64 {
    xx.iter()
        .zip(t)
        .map(|(&x2, &tv)| {
            let g = centroids[nearest_centroid(centroids, x2.sqrt())];
            x2 - 2.0 * g * tv + g * g
        })
        .sum::<f64>()
        / (DIM * xx.len()) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_train: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4_000);
    let n_eval: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16_000);
    let rate: f64 = 48.0 / DIM as f64;
    assert!((rate - 2.0).abs() < 1e-12, "the word is 48 bits over 24 dimensions");

    // The prereg's fixed seed. Both arms see exactly these blocks.
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let train: Vec<_> = (0..n_train).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..n_eval).map(|_| gauss_block(&mut rng)).collect();
    println!("F1b — {n_train} blocs d'entraînement, {n_eval} d'évaluation, débit {rate:.3} b/dim\n");

    // ---- gain centroids: block norms, so both arms share them ----
    let xx_train: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let xx_eval: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = xx_train.iter().map(|v| v.sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);
    println!("centroïdes de gain (1 bit, sur les normes) : {centroids:?}\n");

    // ---- arm A2: the ball-12 control ----
    let s = Searcher::new();
    let t0 = std::time::Instant::now();
    let dots = precompute13(&s, &eval);
    let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
    let mse_ctrl = mse_shape_gain(&xx_eval, &t_ctrl, &centroids);
    println!(
        "témoin boule-12 + 1 bit de gain : MSE {mse_ctrl:.6}  rétention {:.2} %   ({:.1} s)",
        retention_pct(mse_ctrl, rate),
        t0.elapsed().as_secs_f64()
    );
    println!("  le papier, Table 8, sur sa MSE non arrondie : MSE 0.077718  rétention 92.14 %");
    println!(
        "  écart de banc : {:+.2} point — à lire AVANT la ligne F1\n",
        retention_pct(mse_ctrl, rate) - 92.14
    );

    // ---- arm A4: F1 ----
    // The sweep grid, and the first pilot's whole error. `t` peaks near
    // s = 0.34 and falls off on both sides; the first grid ran 0.600 to 3.706,
    // entirely on the far side of the peak, and above s ≈ 1 the encoder
    // collapses onto the origin where `t` is undefined. Measured in
    // `examples/f1scale.rs`, which exists because a 25-point shortfall against
    // the ceiling argument was a bug and not a result.
    let scales: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();
    for w in [[12u32, 15, 12], [12, 16, 11], [13, 13, 13]] {
        let n = if w == [12, 15, 12] { n_eval } else { n_eval.min(2_000) };
        let t1 = std::time::Instant::now();
        let cb = Codebook::new(w);
        let t_f1: Vec<f64> = eval[..n].iter().map(|x| cb.best_t(x, &scales).0).collect();
        let mse = mse_shape_gain(&xx_eval[..n], &t_f1, &centroids);
        let ret = retention_pct(mse, rate);
        let tag = if n < n_eval { format!(" (sur {n} blocs)") } else { String::new() };
        println!(
            "F1 {}/{}/{}{tag} : MSE {mse:.6}  rétention {ret:.2} %   Δ témoin {:+.2} pp   ({:.1} min)",
            w[0],
            w[1],
            w[2],
            ret - retention_pct(mse_ctrl, rate),
            t1.elapsed().as_secs_f64() / 60.0
        );
    }
}
