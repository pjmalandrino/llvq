//! Where the F1 bench encoder's 240 ms go: the scale sweep, arm by arm.
//!
//! `cargo run --release -p llvq-bench --example f1encscales -- [n_eval] [threads]`
//!
//! ## The question
//!
//! F1c's encoding-cost gate is 656 µs per block per core, and the bench
//! encoder (`llvq_bench::f1::Codebook::best_t`) runs at ~240 ms: 18 scales
//! `0.10·1.14^i`, and at each scale a full per-section search joined through
//! the trellis. A production encoder (ROADMAP §2.2, "Viterbi at fixed scale")
//! has to drop the sweep. This example measures what the sweep is worth:
//!
//! * the cost per block per core with 18, 6, 3 and 1 scales;
//! * the retention with each, against the same ball-12 control as F1b;
//! * which scale wins in the 18-point sweep, so a single scale can be chosen
//!   from the data rather than guessed;
//! * one scale **relative to the block norm** (`s = c·‖x‖`), which is the
//!   form a production encoder would take once the row scale normalizes the
//!   block.
//!
//! Exact 12/15/12 regions (`Prepared`), not the universal-table ones: the
//! membership test differs, the loop structure — and therefore the sweep's
//! share — does not. Nothing here touches a format or a served path.

use llvq_bench::f1::Codebook;
use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

const BALL12_SHELLS: usize = 11;

fn t_ball12(d: &[f64; 12]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Shape-gain MSE from the per-block projections `t = ⟨x, ŷ⟩`, the gain being
/// the centroid nearest to `‖x‖` — the shipped rule, as in `f1rankbench`.
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

/// `best_t` written out, so the winning scale is observable.
fn best_t_with_winner(cb: &Codebook, x: &[f64; 24], scales: &[f64]) -> (f64, usize) {
    let mut best = (f64::NEG_INFINITY, 0usize);
    for (i, &s) in scales.iter().enumerate() {
        let y = cb.encode_at_scale(x, s);
        let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
        let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
        if nn > 0.0 {
            let t = dot / nn.sqrt();
            if t > best.0 {
                best = (t, i);
            }
        }
    }
    best
}

/// One arm: every block encoded over `scale_of(x)`, `threads` ways.
/// Returns the per-block `(t, winner index)` and the wall time.
fn run_arm(
    cb: &Codebook,
    xs: &[[f64; 24]],
    scale_of: &(dyn Fn(&[f64; 24]) -> Vec<f64> + Sync),
    threads: usize,
) -> (Vec<(f64, usize)>, f64) {
    let t0 = std::time::Instant::now();
    let chunk = xs.len().div_ceil(threads);
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs
            .chunks(chunk)
            .map(|c| {
                sc.spawn(move || {
                    c.iter()
                        .map(|x| best_t_with_winner(cb, x, &scale_of(x)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().expect("thread"));
        }
    });
    (out, t0.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_eval: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_train = 4_000usize;
    let rate: f64 = 48.0 / DIM as f64;
    let w = [12u32, 15, 12];

    // Same seed as f1rankbench / bin/f1bench: same blocks as every F1 number.
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let train: Vec<_> = (0..n_train).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..n_eval).map(|_| gauss_block(&mut rng)).collect();
    println!(
        "F1 encodeur, coût du balayage d'échelle — {n_eval} blocs d'évaluation, {threads} fils, débit {rate:.3} b/dim"
    );

    let xx_eval: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);

    // ---- control: ball-12 + 1 gain bit, exact search ----
    let s = Searcher::new();
    let t0 = std::time::Instant::now();
    let dots = precompute13(&s, &eval);
    let ctrl_wall = t0.elapsed().as_secs_f64();
    let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
    let ret_ctrl = retention_pct(mse_shape_gain(&xx_eval, &t_ctrl, &centroids), rate);
    let ctrl_threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(n_eval);
    println!(
        "témoin boule-12 + 1 bit : rétention {ret_ctrl:.2} %   {:.3} ms/bloc/cœur (recherche exacte v1, {ctrl_threads} fils)\n",
        ctrl_wall * ctrl_threads as f64 / n_eval as f64 * 1e3
    );

    let t0 = std::time::Instant::now();
    let cb = Codebook::new(w);
    println!("codebook F1 exact 12/15/12 préparé en {:.1} s", t0.elapsed().as_secs_f64());

    let s18: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();

    // ---- arm 1: the 18-scale sweep, with the winner recorded ----
    let (r18, wall18) = run_arm(&cb, &eval, &|_| s18.clone(), threads);
    let t18: Vec<f64> = r18.iter().map(|r| r.0).collect();
    let ret18 = retention_pct(mse_shape_gain(&xx_eval, &t18, &centroids), rate);
    let per18 = wall18 * threads as f64 / n_eval as f64 * 1e3;
    println!("18 échelles (référence F1b) : rétention {ret18:.2} %   {per18:.1} ms/bloc/cœur");

    let mut hist = vec![0usize; s18.len()];
    for r in &r18 {
        hist[r.1] += 1;
    }
    println!("  échelle gagnante, histogramme (i, s, blocs) :");
    for (i, (&sc, &n)) in s18.iter().zip(&hist).enumerate() {
        if n > 0 {
            println!("    {i:>2}  s = {sc:.4}  {n:>5}  ({:.1} %)", 100.0 * n as f64 / n_eval as f64);
        }
    }
    // The single most frequent winner, and the mean of s_win / ‖x‖.
    let i_star = (0..s18.len()).max_by_key(|&i| hist[i]).expect("18 > 0");
    let c_star: f64 = r18
        .iter()
        .zip(&xx_eval)
        .map(|(r, &x2)| s18[r.1] / x2.sqrt())
        .sum::<f64>()
        / n_eval as f64;
    println!("  échelle modale s* = {:.4} (i = {i_star}) ; ⟨s_gagnante/‖x‖⟩ = {c_star:.5}\n", s18[i_star]);

    // ---- arms 2-4: fewer scales, same grid family ----
    for (label, step) in [("6 échelles (i ≡ i* mod 3)", 3usize), ("3 échelles (i* − 1, i*, i* + 1)", 1)] {
        let sub: Vec<f64> = if step == 3 {
            s18.iter().enumerate().filter(|(i, _)| i % 3 == i_star % 3).map(|(_, &v)| v).collect()
        } else {
            (i_star.saturating_sub(1)..=(i_star + 1).min(s18.len() - 1)).map(|i| s18[i]).collect()
        };
        let (r, wall) = run_arm(&cb, &eval, &|_| sub.clone(), threads);
        let t: Vec<f64> = r.iter().map(|r| r.0).collect();
        let ret = retention_pct(mse_shape_gain(&xx_eval, &t, &centroids), rate);
        println!(
            "{label} : rétention {ret:.2} % (Δ18 {:+.2} pp)   {:.1} ms/bloc/cœur",
            ret - ret18,
            wall * threads as f64 / n_eval as f64 * 1e3
        );
    }

    // ---- arm 5: one fixed scale, the modal one ----
    let one = vec![s18[i_star]];
    let (r1, wall1) = run_arm(&cb, &eval, &|_| one.clone(), threads);
    let t1: Vec<f64> = r1.iter().map(|r| r.0).collect();
    let ret1 = retention_pct(mse_shape_gain(&xx_eval, &t1, &centroids), rate);
    println!(
        "1 échelle fixe s* : rétention {ret1:.2} % (Δ18 {:+.2} pp)   {:.1} ms/bloc/cœur",
        ret1 - ret18,
        wall1 * threads as f64 / n_eval as f64 * 1e3
    );

    // ---- arm 6: one scale relative to the block norm, s = c*·‖x‖ ----
    let (rn, walln) = run_arm(
        &cb,
        &eval,
        &|x| vec![c_star * x.iter().map(|v| v * v).sum::<f64>().sqrt()],
        threads,
    );
    let tn: Vec<f64> = rn.iter().map(|r| r.0).collect();
    let retn = retention_pct(mse_shape_gain(&xx_eval, &tn, &centroids), rate);
    println!(
        "1 échelle relative s = c*·‖x‖ : rétention {retn:.2} % (Δ18 {:+.2} pp)   {:.1} ms/bloc/cœur",
        retn - ret18,
        walln * threads as f64 / n_eval as f64 * 1e3
    );

    // ---- arm 7: three scales around c*·‖x‖ (×1/1.14, ×1, ×1.14) ----
    let (r3n, wall3n) = run_arm(
        &cb,
        &eval,
        &|x| {
            let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            vec![c_star * n / 1.14, c_star * n, c_star * n * 1.14]
        },
        threads,
    );
    let t3n: Vec<f64> = r3n.iter().map(|r| r.0).collect();
    let ret3n = retention_pct(mse_shape_gain(&xx_eval, &t3n, &centroids), rate);
    println!(
        "3 échelles relatives c*·‖x‖·1.14^k, k ∈ {{−1,0,1}} : rétention {ret3n:.2} % (Δ18 {:+.2} pp)   {:.1} ms/bloc/cœur",
        ret3n - ret18,
        wall3n * threads as f64 / n_eval as f64 * 1e3
    );

    println!(
        "\nporte F1c : 0.656 ms/bloc/cœur ; la référence 18 échelles est à {:.0}× la porte, une échelle à {:.0}×",
        per18 / 0.656,
        wall1 * threads as f64 / n_eval as f64 * 1e3 / 0.656
    );
}
