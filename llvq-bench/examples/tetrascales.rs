//! Tetra: `α` and the scale pair of the production encoder, fixed on the
//! TRAINING blocks of the F1b seed — the prototype had fixed them on the
//! evaluation blocks (`docs/mesures/f1-encodeur-prototype-2026-09-05.txt`,
//! "⚠️ α = 0,321 et la paire … sont choisis sur les blocs d'ÉVALUATION").
//!
//! `nice -n 10 cargo run --release -p llvq-bench --example tetrascales -- [threads]`
//!
//! ## The protocol, in the order it runs
//!
//! 1. Seed `0x0f1b_2026_0904`, the FIRST 4,000 blocks of `gauss_block`: the
//!    training set. Gain centroids: Lloyd-Max, 1 bit, on their norms.
//! 2. The bench's rank-region encoder (`llvq_bench::f1::rankbook`) on the
//!    18-point grid `0.10·1.14^i`; per block the winning scale `s_win`
//!    (largest `t = ⟨x, y⟩/‖y‖`), and `α = s_win·√24/‖x‖`. `α*` = the median.
//! 3. Three pairs scored on the SAME training blocks with the F1b rule:
//!    `(α*, 1.14·α*)`, `(α*/1.07, 1.07·α*)`, `(α*/1.14, α*)` — by the bench
//!    encoder, which decides, and by `Encoder::encode_at_scale`, which must
//!    give the same points. The winner is what `Encoder::ALPHA`/`RATIO` pin.
//! 4. Only then the next 2,000 blocks of the stream, the evaluation set, are
//!    read once: the ball-12 control in the same process, and the winner's
//!    retention — the number `tests/tetra_encoder.rs` guards.
//!
//! Blocks are handed to the bench encoder as drawn (it reads its input in
//! trio order) and to the production encoder in the natural frame whose
//! trio-order view is that same draw, so both see the fixed blocks of the
//! journals. Every retention is a Gaussian-source figure at 2.000 b/dim,
//! not a quality claim (`docs/ROADMAP.md` §2.2 bis).

use llvq_bench::f1::point_to_natural;
use llvq_bench::f1::rank::RankTable;
use llvq_bench::f1::rankbook::{mse_shape_gain, t_ball12, Codebook, RankRegion};
use llvq_bench::{gauss_block, lloyd_max, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::tetra::encoder::t_of;
use llvq_search::tetra::{Encoder, Scratch, Tetra};
use llvq_search::Searcher;
use std::time::Instant;

const SEED: u64 = 0x0f1b_2026_0904;
const N_TRAIN: usize = 4_000;
const N_EVAL: usize = 2_000;
const GRID: usize = 18;

/// The natural-order block whose trio-order view is `raw`.
fn natural_frame(raw: &[f64; DIM], order: &[u32; DIM]) -> [f64; DIM] {
    let mut nat = [0.0f64; DIM];
    for (j, &v) in raw.iter().enumerate() {
        nat[order[j] as usize] = v;
    }
    nat
}

/// `f` over the blocks, `threads` wide, one `S` per thread; results in order.
fn par_map<S, T: Send, I: Fn() -> S + Sync, F: Fn(&mut S, usize, &[f64; DIM]) -> T + Sync>(xs: &[[f64; DIM]], threads: usize, init: I, f: F) -> Vec<T> {
    let chunk = xs.len().div_ceil(threads.max(1));
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs
            .chunks(chunk)
            .enumerate()
            .map(|(c, blocks)| {
                let (init, f) = (&init, &f);
                sc.spawn(move || {
                    let mut s = init();
                    blocks.iter().enumerate().map(|(i, x)| f(&mut s, c * chunk + i, x)).collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().expect("thread"));
        }
    });
    out
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}

fn main() {
    let threads: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let t_all = Instant::now();
    let rate = 48.0 / DIM as f64;
    let sqrt_dim = (DIM as f64).sqrt();

    let mut rng = SplitMix64::new(SEED);
    let train: Vec<[f64; DIM]> = (0..N_TRAIN).map(|_| gauss_block(&mut rng)).collect();
    println!("Tetra — α et la paire d'échelles fixés sur les {N_TRAIN} blocs d'ENTRAÎNEMENT (graine {SEED:#x}, les {N_TRAIN} premiers), {threads} fils, débit {rate:.3} b/dim\n");

    let xx: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = xx.iter().map(|v| v.sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);
    println!("centroïdes de gain (1 bit, sur les normes d'entraînement) : {centroids:?}");
    let retention = |ts: &[f64]| retention_pct(mse_shape_gain(&xx, ts, &centroids), rate);
    let err = |i: usize, t: f64| {
        let g = centroids[llvq_bench::nearest_centroid(&centroids, norms[i])];
        xx[i] - 2.0 * g * t + g * g
    };
    let paired = |ts: &[f64], reference: &[f64]| {
        let d: Vec<f64> = (0..N_TRAIN).map(|i| err(i, ts[i]) - err(i, reference[i])).collect();
        let m = d.iter().sum::<f64>() / N_TRAIN as f64;
        let sd = (d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (N_TRAIN as f64 - 1.0)).sqrt();
        (m, sd / (N_TRAIN as f64).sqrt())
    };

    let tetra = Tetra::new();
    let order = *tetra.order();
    let enc = Encoder::new(&tetra);
    let table = RankTable::build();
    let cb: Codebook<RankRegion> = Codebook::rank(&table);

    // ---- 2. the 18-point grid, bench encoder, per-block winning scale ----
    let grid: Vec<f64> = (0..GRID as i32).map(|i| 0.10 * 1.14f64.powi(i)).collect();
    let t0 = Instant::now();
    let tmat: Vec<[f64; GRID]> = par_map(&train, threads, || (), |_, _, x| core::array::from_fn(|k| t_of(x, &cb.encode_at_scale(x, grid[k]))));
    let el = t0.elapsed().as_secs_f64();
    let t_grid: Vec<f64> = tmat.iter().map(|r| r.iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect();
    let win: Vec<usize> = tmat.iter().map(|r| r.iter().enumerate().fold(0usize, |b, (k, &t)| if t > r[b] { k } else { b })).collect();
    let mut hist = [0usize; GRID];
    for &k in &win {
        hist[k] += 1;
    }
    println!(
        "\n(a) grille 18 points 0,10·1,14^i, encodeur du banc : rétention {:.2} %   ({:.1} min, {:.2} ms par (bloc, échelle) par fil)",
        retention(&t_grid),
        el / 60.0,
        1e3 * el * threads as f64 / (N_TRAIN * GRID) as f64
    );
    let row: Vec<String> = hist.iter().map(|&c| format!("{:.1}", 100.0 * c as f64 / N_TRAIN as f64)).collect();
    println!("    échelle gagnante, % des blocs par indice 0..17 : [{}]", row.join(" "));
    let mut alphas: Vec<f64> = (0..N_TRAIN).map(|i| grid[win[i]] * sqrt_dim / norms[i]).collect();
    alphas.sort_unstable_by(f64::total_cmp);
    let alpha_star = median(&alphas);
    println!(
        "    α = s_gagnante·√24/‖x‖ : quartiles {:.3} / {:.3} / {:.3}, déciles {:.3} / {:.3}   →   α* = {alpha_star:.4} (médiane)",
        quantile(&alphas, 0.25),
        alpha_star,
        quantile(&alphas, 0.75),
        quantile(&alphas, 0.10),
        quantile(&alphas, 0.90)
    );

    // ---- 3. the three pairs, on the same training blocks ----
    println!("\npaires d'échelles adaptatives (s = α·‖x‖/√24), notées sur les MÊMES blocs d'entraînement, règle F1b :");
    let pairs: [(&str, f64, f64); 3] = [
        ("(α*, 1,14·α*)", alpha_star, 1.14 * alpha_star),
        ("(α*/1,07, 1,07·α*)", alpha_star / 1.07, 1.07 * alpha_star),
        ("(α*/1,14, α*)", alpha_star / 1.14, alpha_star),
    ];
    let mut best: Option<(usize, f64)> = None;
    for (k, &(label, lo, hi)) in pairs.iter().enumerate() {
        let t0 = Instant::now();
        // The bench decides; the production encoder must return its points.
        let bench: Vec<(f64, [i32; DIM], [i32; DIM])> = par_map(&train, threads, || (), |_, i, x| {
            let (s0, s1) = (lo * norms[i] / sqrt_dim, hi * norms[i] / sqrt_dim);
            let (y0, y1) = (cb.encode_at_scale(x, s0), cb.encode_at_scale(x, s1));
            (t_of(x, &y0).max(t_of(x, &y1)), point_to_natural(&y0, &order), point_to_natural(&y1, &order))
        });
        let el_bench = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let prod: Vec<(f64, [i32; DIM], [i32; DIM])> = par_map(&train, threads, Scratch::new, |sc, i, x| {
            let nat = natural_frame(x, &order);
            let (s0, s1) = (lo * norms[i] / sqrt_dim, hi * norms[i] / sqrt_dim);
            let (a, b) = (enc.encode_at_scale(&nat, s0, sc), enc.encode_at_scale(&nat, s1, sc));
            (a.t.max(b.t), a.point, b.point)
        });
        let el_prod = t0.elapsed().as_secs_f64();
        let t_bench: Vec<f64> = bench.iter().map(|e| e.0).collect();
        let t_prod: Vec<f64> = prod.iter().map(|e| e.0).collect();
        let differ = bench.iter().zip(&prod).filter(|(b, p)| b.1 != p.1 || b.2 != p.2).count();
        let (r_bench, r_prod) = (retention(&t_bench), retention(&t_prod));
        let (m, se) = paired(&t_bench, &t_grid);
        println!(
            "  {label:<22} banc {r_bench:6.2} %   production {r_prod:6.2} %   Δ(a) {:+.2} pp   écart apparié {m:+.4} ± {se:.4}   {differ} blocs à point différent   ({el_bench:.1} s banc, {el_prod:.1} s production)",
            r_bench - retention(&t_grid)
        );
        if best.is_none_or(|(_, r)| r_bench > r) {
            best = Some((k, r_bench));
        }
    }
    let (k, r_win) = best.expect("three pairs");
    let (label, lo, hi) = pairs[k];
    println!("\n  gagnante : {label} à {r_win:.2} %   →   ALPHA = {lo:.4}, RATIO = {:.4}", hi / lo);
    let pinned = (Encoder::ALPHA - lo).abs() < 5e-5 && (Encoder::RATIO - hi / lo).abs() < 5e-5;
    println!(
        "  constantes de l'encodeur : ALPHA = {:.4}, RATIO = {:.4} — {}",
        Encoder::ALPHA,
        Encoder::RATIO,
        if pinned { "égales à la gagnante" } else { "DIFFÉRENTES de la gagnante : à épingler dans encoder.rs" }
    );

    // ---- 4. the evaluation blocks, read once ----
    let eval: Vec<[f64; DIM]> = (0..N_EVAL).map(|_| gauss_block(&mut rng)).collect();
    let xx_eval: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms_eval: Vec<f64> = xx_eval.iter().map(|v| v.sqrt()).collect();
    let ret_eval = |ts: &[f64]| retention_pct(mse_shape_gain(&xx_eval, ts, &centroids), rate);
    println!("\nles {N_EVAL} blocs d'ÉVALUATION (les suivants du flux), lus une fois, centroïdes d'entraînement :");
    let t0 = Instant::now();
    let s = Searcher::new();
    let dots = precompute13(&s, &eval);
    let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
    let r_ctrl = ret_eval(&t_ctrl);
    println!("  témoin boule-12 + 1 bit de gain : rétention {r_ctrl:.2} %   ({:.1} s)", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let t_win: Vec<f64> = par_map(&eval, threads, Scratch::new, |sc, i, x| {
        let nat = natural_frame(x, &order);
        let (s0, s1) = (lo * norms_eval[i] / sqrt_dim, hi * norms_eval[i] / sqrt_dim);
        enc.encode_at_scale(&nat, s0, sc).t.max(enc.encode_at_scale(&nat, s1, sc).t)
    });
    let r_win_eval = ret_eval(&t_win);
    println!("  Tetra, paire gagnante {label:<22} : rétention {r_win_eval:.2} %   Δ témoin {:+.2} pp   ({:.1} s)", r_win_eval - r_ctrl, t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let t_const: Vec<f64> = par_map(&eval, threads, Scratch::new, |sc, _, x| enc.encode(&natural_frame(x, &order), sc).t);
    let r_const = ret_eval(&t_const);
    println!(
        "  Tetra, `Encoder::encode` (ALPHA {:.4}, RATIO {:.4}) : rétention {r_const:.2} %   Δ témoin {:+.2} pp   ({:.1} s){}",
        Encoder::ALPHA,
        Encoder::RATIO,
        r_const - r_ctrl,
        t0.elapsed().as_secs_f64(),
        if pinned { "" } else { "   [constantes provisoires]" }
    );
    println!("\ntotal {:.1} min", t_all.elapsed().as_secs_f64() / 60.0);
}
