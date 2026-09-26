//! Why the spherical retraction is free at 0 gain bits and a lie at 1.
//!
//! `cargo run --release -p llvq-bench --example sphgeom -- [n_eval] [seed]`
//!
//! `docs/mesures/sph-ppl-0.6b-2026-09-22.txt` measures that feeding the GPTQ
//! loop the retracted residual costs +31.9 % of perplexity on `Tetra`, and
//! records that its mechanism was not measured. This is that mechanism, on a
//! Gaussian source, at $0. It decides nothing: the endpoint is already read.
//!
//! One block `x`, one coded direction `u = ŵ/‖ŵ‖`, `t = ⟨x, u⟩ = ‖x‖·cos θ`.
//! The file stores `ŵ = ĝ·u`. Two residuals can be handed to the correction:
//!
//! | residual | formula | radial component |
//! |---|---|---|
//! | published | `x − ĝ·u` | `t − ĝ` |
//! | retracted (`sph`) | `x − ‖x‖·u` | `t − ‖x‖` |
//!
//! Their difference is `(‖x‖ − ĝ)·u`: **purely radial, and exactly zero when
//! the code stores the block's own norm.** A 0-gain-bit direction code has
//! `ĝ = ‖x‖` by construction, so the two residuals are the same vector and
//! the paper's retraction changes nothing to propagate. A coded gain does
//! not, and the gap is what the loop is never told about.
//!
//! Printed here: the distribution of `ĝ/‖x‖` under the production encoder,
//! and that gap's energy against the error the file actually carries.

use llvq_bench::{gauss_block, optimize_beta13, precompute13};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_quant::quantizer::{
    fit_gain_centroids, row_scale, BlockQuantizer, LeechBall, TetraShapeGain,
};
use llvq_search::Searcher;

const SEED: u64 = 0x0f1b_2026_0922;
const N_TRAIN: usize = 4_000;

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

/// `ĝ/‖x‖` binned for plotting: 40 bins over [0.5, 1.5], then the tails.
fn histogram(v: &[f64], label: &str) {
    const LO: f64 = 0.5;
    const HI: f64 = 1.5;
    const N: usize = 40;
    let mut bins = vec![0usize; N];
    let (mut under, mut over) = (0usize, 0usize);
    for &x in v {
        if x < LO {
            under += 1;
        } else if x >= HI {
            over += 1;
        } else {
            bins[((x - LO) / (HI - LO) * N as f64) as usize] += 1;
        }
    }
    println!("HIST {label} lo={LO} hi={HI} n={N} under={under} over={over}");
    for (i, c) in bins.iter().enumerate() {
        println!(
            "  {:.4} {c}",
            LO + (HI - LO) * (i as f64 + 0.5) / N as f64
        );
    }
}

fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (m, sd)
}

fn main() {
    let n_eval: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(SEED);

    let mut rng = SplitMix64::new(seed);
    let n = N_TRAIN + n_eval;
    let blocks: Vec<[f64; DIM]> = (0..n).map(|_| gauss_block(&mut rng)).collect();

    // Production accounting: the blocks are one row, the row scale is its RMS
    // block norm, the two centroids are fitted on the train prefix exactly as
    // `smoke` fits them.
    let flat: Vec<f64> = blocks.iter().flat_map(|b| b.iter().copied()).collect();
    let train = &flat[..N_TRAIN * DIM];
    let centroids = fit_gain_centroids(train, 1, N_TRAIN * DIM, DIM, 1, 40);
    let rs = row_scale(&flat);

    let mut q = TetraShapeGain::new(centroids.clone());
    q.set_row_scale(rs);
    let mut out = [0.0f64; DIM];

    let mut ratio = Vec::with_capacity(n_eval); // ĝ / ‖x‖
    let mut costheta = Vec::with_capacity(n_eval);
    let mut hidden_share = Vec::with_capacity(n_eval); // (‖x‖−ĝ)² / ‖x−ŵ‖²
    let mut e_pub2 = 0.0f64;
    let mut e_sph2 = 0.0f64;
    let mut hidden2 = 0.0f64;
    let mut on_upper = 0usize;

    for x in &blocks[N_TRAIN..] {
        q.quantize(x, &mut out);
        let xn = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let gn = out.iter().map(|a| a * a).sum::<f64>().sqrt();
        if gn == 0.0 {
            continue;
        }
        let t: f64 = x.iter().zip(out.iter()).map(|(a, b)| a * b).sum::<f64>() / gn;
        let ep2: f64 = x
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        // ‖x − ‖x‖u‖² = 2‖x‖(‖x‖ − t)
        let es2 = 2.0 * xn * (xn - t);
        let h2 = (xn - gn) * (xn - gn);

        ratio.push(gn / xn);
        costheta.push(t / xn);
        hidden_share.push(h2 / ep2);
        e_pub2 += ep2;
        e_sph2 += es2;
        hidden2 += h2;
        if q.last_code().expect("Tetra always emits a code").gain == 1 {
            on_upper += 1;
        }
    }

    let m = ratio.len();
    let mut sorted = ratio.clone();
    sorted.sort_by(f64::total_cmp);
    let (mr, sdr) = stats(&ratio);
    let (mc, sdc) = stats(&costheta);
    let (mh, _) = stats(&hidden_share);

    println!("sphgeom — the radial gap the retracted residual hides");
    println!("  source            Gaussian, {m} evaluation blocks, {N_TRAIN} train, seed {seed:#x}");
    println!("  encoder           production TetraShapeGain, 1 gain bit, 2 centroids");
    println!(
        "  centroids         {:.6} / {:.6}   (relative to row scale {rs:.6})",
        centroids[0], centroids[1]
    );
    println!(
        "  upper level        {:.2} % of blocks",
        100.0 * on_upper as f64 / m as f64
    );
    println!();
    println!("THE MAGNITUDE THE FILE STORES, AGAINST THE BLOCK'S OWN NORM");
    println!("  ĝ/‖x‖   mean {mr:.5}   sd {sdr:.5}");
    println!(
        "          p1 {:.4}   p25 {:.4}   p50 {:.4}   p75 {:.4}   p99 {:.4}",
        pct(&sorted, 0.01),
        pct(&sorted, 0.25),
        pct(&sorted, 0.50),
        pct(&sorted, 0.75),
        pct(&sorted, 0.99)
    );
    println!("  cos θ   mean {mc:.5}   sd {sdc:.5}");
    println!("  A 0-gain-bit direction code has ĝ/‖x‖ = 1 exactly, sd 0.");
    println!();
    println!("THE GAP BETWEEN WHAT IS PROPAGATED AND WHAT IS STORED");
    println!("  it is (‖x‖ − ĝ)·u, purely radial, and zero at 0 gain bits");
    println!(
        "  energy of the gap / energy of the stored error   {:.4}   (per-block mean {mh:.4})",
        hidden2 / e_pub2
    );
    println!(
        "  stored error   ‖x − ĝu‖²  total {:.4}   per weight {:.6}",
        e_pub2,
        e_pub2 / (m * DIM) as f64
    );
    println!(
        "  propagated     ‖x − ‖x‖u‖² total {:.4}   per weight {:.6}",
        e_sph2,
        e_sph2 / (m * DIM) as f64
    );
    println!(
        "  the retracted residual is {:+.2} % of the true error energy",
        100.0 * (e_sph2 / e_pub2 - 1.0)
    );

    // The contrast arm: spherical shaping, the ball code the paper's 191.90
    // belongs to. Its magnitude is whatever the nearest ball point's norm is,
    // so nothing holds it near ‖x‖. That is radial *drift*, and it is what the
    // retraction was built to remove. A global β is legitimate here only
    // because the source is homogeneous; a weight matrix needs a row scale.
    let searcher = Searcher::new();
    let beta = optimize_beta13(&precompute13(&searcher, &blocks[..N_TRAIN]), 0.2, 0.9, 140);
    let mut ball = LeechBall::new(beta);
    let mut ball_ratio = Vec::with_capacity(n_eval);
    for x in &blocks[N_TRAIN..] {
        ball.quantize(x, &mut out);
        let xn = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let gn = out.iter().map(|a| a * a).sum::<f64>().sqrt();
        ball_ratio.push(gn / xn);
    }
    let (mb, sdb) = stats(&ball_ratio);
    let mut bs = ball_ratio.clone();
    bs.sort_by(f64::total_cmp);
    println!();
    println!("THE CONTRAST ARM: SPHERICAL SHAPING, THE BALL CODE (beta* = {beta:.4})");
    println!("  its magnitude is the ball point's own norm, tied to nothing");
    println!("  ‖ŵ‖/‖x‖ mean {mb:.5}   sd {sdb:.5}");
    println!(
        "          p1 {:.4}   p25 {:.4}   p50 {:.4}   p75 {:.4}   p99 {:.4}",
        pct(&bs, 0.01),
        pct(&bs, 0.25),
        pct(&bs, 0.50),
        pct(&bs, 0.75),
        pct(&bs, 0.99)
    );
    println!(
        "  zero blocks (the origin is a codeword) {:.2} %",
        100.0 * ball_ratio.iter().filter(|r| **r == 0.0).count() as f64 / ball_ratio.len() as f64
    );
    println!();
    histogram(&ratio, "tetra");
    histogram(&ball_ratio, "ball");
}
