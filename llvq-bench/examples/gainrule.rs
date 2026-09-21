//! Which of the two legal gain levels a Tetra block should take, measured.
//!
//! `cargo run --release -p llvq-bench --example gainrule -- [n_eval] [threads]`
//!
//! Production rounds the block **norm** to a level (`quantizer.rs`,
//! `TetraShapeGain::quantize`: `nearest_level_index(centroids, norm/row_scale)`).
//! The reconstruction is `ĝ·v̂` either way, so the block error is
//! `‖x‖² − 2ĝt + ĝ²` with `t = ⟨x, v̂⟩ = ‖x‖·cos θ`. That is minimized at
//! `ĝ = t`, so rounding the norm overshoots by construction. `lib.rs`
//! already names the two rules and measures their gap for the ball-13
//! codebook; this measures it for **Tetra**, which is the object now, and
//! adds the quantity the warning there turns on: how much the reconstruction
//! shrinks if the rule changes.
//!
//! Four arms on the same blocks, same directions, same 48-bit word:
//!
//! | arm | centroids fitted on | level chosen by |
//! |---|---|---|
//! | A, shipped | norms | norm |
//! | B | norms | projection |
//! | C | projections | projection |
//! | D, bound | free | the continuous optimum `t` |
//!
//! A, B and C write the same file: one label, one gain bit, the same two
//! floats per matrix. Only the encoder's decision differs, so nothing here
//! touches the format, the decoder or the served kernel.
//!
//! Gaussian blocks are the source, as in every retention bench of this
//! repository (`docs/ROADMAP.md` §2.2 bis). A GPTQ residue is not Gaussian
//! and a smaller local error has composed worse three times in this file
//! (design C, `group_scales`, gptq2), so a win here is a reason to run the
//! model A/B, not a result about the model.

use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::tetra::{Encoder, Scratch, Tetra};

const SEED: u64 = 0x0f1b_2026_0904;
const N_TRAIN: usize = 4_000;

/// `‖x − g·v̂‖²` for a block of squared norm `xx` and projection `t`.
fn err(xx: f64, t: f64, g: f64) -> f64 {
    xx - 2.0 * g * t + g * g
}

/// Mean and standard error of a paired difference.
fn paired(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let m = d.iter().sum::<f64>() / n;
    let sd = (d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (m, sd / n.sqrt())
}

fn main() {
    let n_eval: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let threads: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let seed: u64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(SEED);
    let rate = 48.0 / DIM as f64;

    let tetra = Tetra::new();
    let enc = Encoder::new(&tetra);

    // One pass over train + eval, production encoder, threads wide.
    let mut rng = SplitMix64::new(seed);
    let blocks: Vec<[f64; DIM]> = (0..N_TRAIN + n_eval).map(|_| gauss_block(&mut rng)).collect();
    let chunk = blocks.len().div_ceil(threads.max(1));
    let mut coded: Vec<(f64, f64)> = Vec::with_capacity(blocks.len()); // (‖x‖², t)
    std::thread::scope(|sc| {
        let handles: Vec<_> = blocks
            .chunks(chunk)
            .map(|ch| {
                let enc = &enc;
                sc.spawn(move || {
                    let mut scratch = Scratch::new();
                    ch.iter()
                        .map(|x| {
                            let xx: f64 = x.iter().map(|v| v * v).sum();
                            (xx, enc.encode(x, &mut scratch).t)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            coded.extend(h.join().expect("thread"));
        }
    });

    let (train, eval) = coded.split_at(N_TRAIN);
    let norms_tr: Vec<f64> = train.iter().map(|&(xx, _)| xx.sqrt()).collect();
    let projs_tr: Vec<f64> = train.iter().map(|&(_, t)| t).collect();
    let c_norm = lloyd_max(&norms_tr, 1, 60);
    let c_proj = lloyd_max(&projs_tr, 1, 60);

    println!(
        "gainrule — Tetra, production encoder, seed {seed:#x}, {N_TRAIN} training blocks, \
         {n_eval} evaluation blocks, {threads} threads, rate {rate:.3} b/dim\n"
    );
    println!("gain centroids fitted on norms       {c_norm:?}");
    println!("gain centroids fitted on projections {c_proj:?}");

    let cos: Vec<f64> = eval.iter().map(|&(xx, t)| t / xx.sqrt()).collect();
    let mut sorted = cos.clone();
    sorted.sort_by(f64::total_cmp);
    let mean_cos = cos.iter().sum::<f64>() / cos.len() as f64;
    let sd_cos = (cos.iter().map(|c| (c - mean_cos).powi(2)).sum::<f64>()
        / (cos.len() as f64 - 1.0))
        .sqrt();
    println!(
        "\ncos θ between a block and its Tetra direction: mean {mean_cos:.4}, sd {sd_cos:.4}, \
         median {:.4}, 1st percentile {:.4}",
        sorted[sorted.len() / 2],
        sorted[sorted.len() / 100]
    );
    println!(
        "the ratio 2/(1+cos θ) the norm rule pays at the continuous optimum: {:.4}",
        2.0 / (1.0 + mean_cos)
    );

    // Four arms, per block.
    let arm_a: Vec<f64> = eval
        .iter()
        .map(|&(xx, t)| err(xx, t, c_norm[nearest_centroid(&c_norm, xx.sqrt())]))
        .collect();
    let arm_b: Vec<f64> = eval
        .iter()
        .map(|&(xx, t)| err(xx, t, c_norm[nearest_centroid(&c_norm, t)]))
        .collect();
    let arm_c: Vec<f64> = eval
        .iter()
        .map(|&(xx, t)| err(xx, t, c_proj[nearest_centroid(&c_proj, t)]))
        .collect();
    let arm_d: Vec<f64> = eval.iter().map(|&(xx, t)| err(xx, t, t)).collect();
    // E: C's centroids rescaled so the mean reconstructed norm matches A's.
    // It separates the MSE win from the radial drift that comes with it.
    let mean_gain = |cent: &[f64], pick_t: bool| -> f64 {
        eval.iter()
            .map(|&(xx, t)| cent[nearest_centroid(cent, if pick_t { t } else { xx.sqrt() })])
            .sum::<f64>()
            / eval.len() as f64
    };
    let k = mean_gain(&c_norm, false) / mean_gain(&c_proj, true);
    let c_resc: Vec<f64> = c_proj.iter().map(|g| g * k).collect();
    let arm_e: Vec<f64> = eval
        .iter()
        .map(|&(xx, t)| err(xx, t, c_resc[nearest_centroid(&c_proj, t)]))
        .collect();

    let mse = |v: &[f64]| v.iter().sum::<f64>() / (DIM * v.len()) as f64;
    println!(
        "\n{:<34} {:>10} {:>11} {:>24} {:>12}",
        "arm", "MSE", "retention", "paired gap against A", "levels moved"
    );
    let base = mse(&arm_a);
    for (name, v, cent, pick_t) in [
        ("A, shipped: norms, round the norm", &arm_a, &c_norm, false),
        ("B: norm centroids, round t", &arm_b, &c_norm, true),
        ("C: projection centroids, round t", &arm_c, &c_proj, true),
        ("E: C rescaled to A's mean norm", &arm_e, &c_resc, true),
        ("D: free gain, the bound", &arm_d, &c_norm, false),
    ] {
        let m = mse(v);
        let (d, se) = paired(v, &arm_a);
        // A block "moves" when the rule lands on the other of the two levels.
        let moved = eval
            .iter()
            .filter(|&&(xx, t)| {
                nearest_centroid(&c_norm, xx.sqrt())
                    != nearest_centroid(cent, if pick_t { t } else { xx.sqrt() })
            })
            .count();
        print!(
            "{name:<34} {m:10.6} {:10.2}% {:+9.5} +- {:.5} {:10.1}%",
            retention_pct(m, rate),
            d / DIM as f64,
            se / DIM as f64,
            100.0 * moved as f64 / eval.len() as f64
        );
        println!("{}", if m < base { "" } else { "   worse" });
    }
    let _ = base;

    // The radial drift the warning in lib.rs turns on.
    let norm_of = |cent: &[f64], pick_t: bool| -> f64 {
        eval.iter()
            .map(|&(xx, t)| cent[nearest_centroid(cent, if pick_t { t } else { xx.sqrt() })])
            .sum::<f64>()
            / eval.len() as f64
    };
    let n_x = eval.iter().map(|&(xx, _)| xx.sqrt()).sum::<f64>() / eval.len() as f64;
    println!(
        "\nmean block norm {n_x:.4}; mean reconstructed norm: A {:.4}, B {:.4}, C {:.4}, \
         continuous optimum {:.4}",
        norm_of(&c_norm, false),
        norm_of(&c_norm, true),
        norm_of(&c_proj, true),
        eval.iter().map(|&(_, t)| t).sum::<f64>() / eval.len() as f64
    );
    println!(
        "radial drift against A: B {:+.2}%, C {:+.2}%",
        100.0 * (norm_of(&c_norm, true) / norm_of(&c_norm, false) - 1.0),
        100.0 * (norm_of(&c_proj, true) / norm_of(&c_norm, false) - 1.0)
    );

    // The control that decides whether the decision rule matters at all:
    // keep the shipped rule and shrink its centroids by a constant. If the
    // curve reaches C's MSE, the rule bought nothing and the shrink bought
    // everything.
    println!(
        "\nThe shipped rule with its centroids scaled by k, same decision on the norm:\n\
         {:>8} {:>12} {:>11} {:>14}",
        "k", "MSE", "retention", "mean norm"
    );
    let mut best = (f64::INFINITY, 1.0);
    for i in 0..=14 {
        let k = 0.90 + 0.01 * i as f64;
        let cent: Vec<f64> = c_norm.iter().map(|g| g * k).collect();
        let m = eval
            .iter()
            .map(|&(xx, t)| err(xx, t, cent[nearest_centroid(&c_norm, xx.sqrt())]))
            .sum::<f64>()
            / (DIM * eval.len()) as f64;
        let mn = eval
            .iter()
            .map(|&(xx, _)| cent[nearest_centroid(&c_norm, xx.sqrt())])
            .sum::<f64>()
            / eval.len() as f64;
        if m < best.0 {
            best = (m, k);
        }
        println!("{k:8.2} {m:12.6} {:10.2}% {mn:14.4}", retention_pct(m, rate));
    }
    println!(
        "best k {:.2} at MSE {:.6} ({:.2}% retention); arm C reads {:.6} ({:.2}%)",
        best.1,
        best.0,
        retention_pct(best.0, rate),
        mse(&arm_c),
        retention_pct(mse(&arm_c), rate)
    );
}
