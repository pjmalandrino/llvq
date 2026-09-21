//! Does a direction other than the nearest one cost less, once the activations
//! are taken into account?
//!
//! ```text
//! cargo run --release -p llvq-llm --features fast-linalg --bin tetraalt -- <pilot dir>
//! ```
//!
//! ## The question
//!
//! Tetra picks the direction that minimizes Euclidean distance to the block,
//! over both parities and a Viterbi through the three-section trellis. That
//! choice ignores `H`: two errors of the same norm can cost 2.84x one another
//! in the metric the model actually pays (*measured*, p5 to p95 over the 4B's
//! rotated Hessians). So a slightly worse direction whose error lands in a
//! quiet subspace may beat the nearest one.
//!
//! ## What is scored, and why not a raw block of H
//!
//! GPTQ compensates a block's error onto the columns that follow, so the cost
//! of writing `q` at position `start` is not `eᵀH_block e`: it is the
//! conditional cost after the continuous suffix is re-optimized,
//! `‖e U_BB⁻¹‖²` ([`llvq_quant::schur::continuous_lower_bound`]). Scoring the
//! raw block would rank candidates on a quantity the loop does not pay.
//!
//! Everything else is held fixed: the same two gain centroids, the same
//! source-norm gain rule, the same 48-bit word, the same trajectory. Only the
//! direction moves, so a difference is attributable to the direction alone.
//!
//! ## What the candidates are, and what that bounds
//!
//! The encoder's own answer at a range of scales ([`Encoder::encode_at_scale`]),
//! deduplicated by point. These are representable candidates reached through
//! the production rule, not the K best paths of the trellis — a real K-best
//! would need the Viterbi to carry a beam. So the gain measured here is a
//! LOWER BOUND on what a proper K-best could reach, and a negative result is
//! about these candidates rather than about the idea.
//!
//! The current choice is always in the candidate set, so the selection score
//! can only improve or tie. The result is therefore never "does it win" but
//! "by how much, where, and does it survive off the calibration set".

use anyhow::Context;
use llvq_core::DIM;
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{row_scale, BlockCode, BlockQuantizer, TetraShapeGain};
use llvq_quant::schur::{commit, continuous_lower_bound};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// (rows, served rollout, selected rollout, served reserved, selected reserved).
type RowTotals = (usize, f64, f64, f64, f64);

/// Scales the candidate sweep visits, as multiples of the encoder's own.
/// Spread on both sides: shrinking the target moves the winner to a lower
/// shell, stretching it to a higher one, and both are legitimate directions.
const SWEEP: [f64; 15] = [
    0.80, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.09, 1.12, 1.16, 1.20, 1.25, 1.30,
];

fn read_json(p: &Path) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}

fn read_f64le(p: &Path) -> anyhow::Result<Vec<f64>> {
    let b = std::fs::read(p)?;
    anyhow::ensure!(b.len() % 8 == 0, "{}: not a f64 array", p.display());
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

/// One block's verdict: what the served rule picked, and the best the
/// candidate set could do under the conditional cost.
#[derive(Default, Clone, Copy)]
struct Cell {
    blocks: usize,
    changed: usize,
    /// Conditional cost of the served choice, and of the best candidate.
    served: f64,
    best: f64,
    /// The same two, scored on reserved activations instead.
    served_val: f64,
    best_val: f64,
    candidates: usize,
}

/// Mean squared output error on reserved activation rows, for one row of
/// weights against its original.
fn validation_cost(acts: &[f64], original: &[f64], q: &[f64]) -> f64 {
    let n = original.len();
    acts.chunks_exact(n)
        .map(|a| {
            a.iter()
                .zip(original.iter().zip(q))
                .map(|(a, (w, q))| a * (q - w))
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        / (acts.len() / n) as f64
}

fn main() -> anyhow::Result<()> {
    let root = std::env::args()
        .nth(1)
        .context("usage: tetraalt <pilot dir>")?;
    let root = Path::new(&root);
    let k_max: usize = std::env::var("LLVQ_ALT_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    println!("=== tetraalt ===");
    println!("pilot        {}", root.display());
    println!("candidates   up to {k_max} distinct points per block, from {} scales", SWEEP.len());
    println!("score        conditional cost ||e U_BB^-1||^2, not a raw block of H");
    println!("held fixed   gain rule, centroids, word, trajectory\n");

    let encoder = TetraShapeGain::encoder();
    let mut by_cell: HashMap<(String, usize), Cell> = HashMap::new();
    let mut ks: Vec<Cell> = vec![Cell::default(); k_max + 1];
    // (rows, served rollout, selected rollout, served reserved, selected reserved)
    let mut rows_acc: HashMap<(String, usize), RowTotals> = HashMap::new();

    for seed in ["seed-1", "seed-2"] {
        let sdir = root.join(seed);
        if !sdir.is_dir() {
            continue;
        }
        let mut cells: Vec<_> = std::fs::read_dir(&sdir)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("layer-")))
            .collect();
        cells.sort();
        for cell in cells {
            let b = read_json(&cell.join("bundle.json"))?;
            let n = b["width"].as_u64().context("width")? as usize;
            let layer = b["layer"].as_u64().context("layer")? as usize;
            let proj = b["projection"].as_str().context("projection")?.to_string();
            let centroids: Vec<f64> = b["centroids"]
                .as_array()
                .context("centroids")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect();
            let hname = b["hessian"]["name"].as_str().context("hessian name")?;
            let h = read_f64le(&sdir.join("capture").join(hname))?;
            anyhow::ensure!(h.len() == n * n, "{hname}: {} values for width {n}", h.len());
            let factor = GptqFactor::new(&h, n, 1e-2).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let vname = b["validation"]["name"].as_str().context("validation name")?;
            let acts = read_f64le(&sdir.join("capture").join(vname))?;

            let mut rows: Vec<_> = std::fs::read_dir(&cell)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.file_name().is_some_and(|f| f.to_string_lossy().starts_with("row-")))
                .collect();
            rows.sort();
            for rf in rows {
                let d = read_json(&rf)?;
                let d = &d["diagnostic"];
                let original: Vec<f64> = d["original"]
                    .as_array()
                    .context("original")?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect();
                anyhow::ensure!(original.len() == n, "row width {} against {n}", original.len());
                let scale = row_scale(&original).max(f64::MIN_POSITIVE);
                let mut quant = TetraShapeGain::with_encoder(encoder.clone(), centroids.clone());
                quant.set_row_scale(scale);
                // The served trajectory, block by block, exactly as `diagnose_row`
                // walks it: the candidate comparison happens at the state the
                // production loop is actually in.
                let mut working = original.clone();
                let mut scratch = vec![0.0f64; DIM];
                for blk in 0..n / DIM {
                    let start = blk * DIM;
                    let x: [f64; DIM] = working[start..start + DIM]
                        .try_into()
                        .expect("DIM coordinates");
                    quant.quantize(&x, &mut scratch);
                    let served = quant.last_code().context("Tetra emits a code")?;

                    // Candidates: the encoder's own answer at a sweep of scales,
                    // deduplicated, the served point first.
                    let mut pts: Vec<[i32; DIM]> = vec![served.point];
                    let (m, s0) = llvq_search::tetra::Encoder::lower_scale(&x);
                    if m > 0.0 && s0.is_normal() {
                        let mut sc = llvq_search::tetra::Scratch::new();
                        for f in SWEEP {
                            if pts.len() >= k_max {
                                break;
                            }
                            let c = encoder.encode_at_scale(&x, s0 * f, &mut sc);
                            if !pts.contains(&c.point) {
                                pts.push(c.point);
                            }
                        }
                    }

                    // Score every candidate under the SAME gain rule, so only
                    // the direction differs.
                    let mut recon = vec![0.0f64; DIM];
                    let mut best = (f64::INFINITY, 0usize, 0.0f64);
                    let mut served_cost = (0.0f64, 0.0f64);
                    for (i, p) in pts.iter().enumerate() {
                        let code = BlockCode { point: *p, gain: served.gain };
                        quant.reconstruct(&code, scale, &mut recon);
                        let e: Vec<f64> = x.iter().zip(&recon).map(|(a, q)| a - q).collect();
                        let cost = continuous_lower_bound(&factor, start, &e);
                        // The reserved-activation cost of writing this block,
                        // with the rest of the row at its original values.
                        let mut trial = working.clone();
                        trial[start..start + DIM].copy_from_slice(&recon);
                        let vcost = validation_cost(&acts, &original, &trial);
                        if i == 0 {
                            served_cost = (cost, vcost);
                        }
                        if cost < best.0 {
                            best = (cost, i, vcost);
                        }
                    }

                    let key = (proj.clone(), layer);
                    let c = by_cell.entry(key).or_default();
                    c.blocks += 1;
                    c.candidates += pts.len();
                    c.served += served_cost.0;
                    c.best += best.0;
                    c.served_val += served_cost.1;
                    c.best_val += best.2;
                    c.changed += usize::from(best.1 != 0);
                    // The same block scored under every budget K, so the curve
                    // comes from one pass rather than from k_max passes.
                    for (k, e) in ks.iter_mut().enumerate().skip(1) {
                        let lim = pts.len().min(k);
                        let mut bk = (f64::INFINITY, 0usize, 0.0f64);
                        for (i, p) in pts.iter().take(lim).enumerate() {
                            let code = BlockCode { point: *p, gain: served.gain };
                            quant.reconstruct(&code, scale, &mut recon);
                            let e: Vec<f64> = x.iter().zip(&recon).map(|(a, q)| a - q).collect();
                            let cost = continuous_lower_bound(&factor, start, &e);
                            if cost < bk.0 {
                                let mut trial = working.clone();
                                trial[start..start + DIM].copy_from_slice(&recon);
                                bk = (cost, i, validation_cost(&acts, &original, &trial));
                            }
                        }
                        e.blocks += 1;
                        e.served += served_cost.0;
                        e.best += bk.0;
                        e.served_val += served_cost.1;
                        e.best_val += bk.2;
                        e.changed += usize::from(bk.1 != 0);
                    }

                    // Advance on the SERVED choice: this pass measures what a
                    // different direction is worth AT THIS STATE.
                    quant.reconstruct(&served, scale, &mut recon);
                    commit(&factor, &mut working, start, &recon);
                }

                // ---- the question the per-block table cannot answer ----
                //
                // Every number above is measured at a state the SERVED policy
                // reached. Chaining the selector changes the state each later
                // block sees, so a per-block gain can compound, cancel, or
                // reverse. This walks the same row again, selecting at every
                // block and committing what it selected, and compares the two
                // completed rows on the objective the loop minimizes and on
                // reserved activations.
                let mut alt = original.clone();
                let mut altq = TetraShapeGain::with_encoder(encoder.clone(), centroids.clone());
                altq.set_row_scale(scale);
                let mut sc = llvq_search::tetra::Scratch::new();
                for blk in 0..n / DIM {
                    let start = blk * DIM;
                    let x: [f64; DIM] = alt[start..start + DIM].try_into().expect("DIM");
                    altq.quantize(&x, &mut scratch);
                    let base = altq.last_code().context("Tetra emits a code")?;
                    let mut pts: Vec<[i32; DIM]> = vec![base.point];
                    let (m, s0) = llvq_search::tetra::Encoder::lower_scale(&x);
                    if m > 0.0 && s0.is_normal() {
                        for f in SWEEP {
                            if pts.len() >= k_max {
                                break;
                            }
                            let c = encoder.encode_at_scale(&x, s0 * f, &mut sc);
                            if !pts.contains(&c.point) {
                                pts.push(c.point);
                            }
                        }
                    }
                    let mut recon = vec![0.0f64; DIM];
                    let mut keep = (f64::INFINITY, base.point);
                    for p in &pts {
                        let code = BlockCode { point: *p, gain: base.gain };
                        altq.reconstruct(&code, scale, &mut recon);
                        let e: Vec<f64> = x.iter().zip(&recon).map(|(a, q)| a - q).collect();
                        let c = continuous_lower_bound(&factor, start, &e);
                        if c < keep.0 {
                            keep = (c, *p);
                        }
                    }
                    let code = BlockCode { point: keep.1, gain: base.gain };
                    altq.reconstruct(&code, scale, &mut recon);
                    commit(&factor, &mut alt, start, &recon);
                }
                let rl_served = llvq_quant::schur::rollout_loss(&factor, &original, &working);
                let rl_alt = llvq_quant::schur::rollout_loss(&factor, &original, &alt);
                let vl_served = validation_cost(&acts, &original, &working);
                let vl_alt = validation_cost(&acts, &original, &alt);
                let r = rows_acc.entry((proj.clone(), layer)).or_insert((0usize, 0.0, 0.0, 0.0, 0.0));
                r.0 += 1;
                r.1 += rl_served;
                r.2 += rl_alt;
                r.3 += vl_served;
                r.4 += vl_alt;
            }
        }
    }

    let pct = |a: f64, b: f64| 100.0 * (1.0 - a / b);
    println!("--- the curve in K, over every block ---");
    println!("  {:>3} {:>10} {:>12} {:>12}", "K", "changed", "cost gain", "on reserved");
    for (k, c) in ks.iter().enumerate().skip(1) {
        if c.blocks == 0 {
            continue;
        }
        println!(
            "  {k:>3} {:>9.2} % {:>11.3} % {:>11.3} %",
            100.0 * c.changed as f64 / c.blocks as f64,
            pct(c.best, c.served),
            pct(c.best_val, c.served_val)
        );
    }

    println!("\n--- by matrix and depth, at K = {k_max} ---");
    println!("  {:<22} {:>6} {:>9} {:>11} {:>11}", "cell", "blocks", "changed", "cost gain", "reserved");
    let mut keys: Vec<_> = by_cell.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let c = by_cell[&k];
        println!(
            "  {:<16} L{:<4} {:>6} {:>8.2} % {:>10.3} % {:>10.3} %",
            k.0,
            k.1,
            c.blocks,
            100.0 * c.changed as f64 / c.blocks as f64,
            pct(c.best, c.served),
            pct(c.best_val, c.served_val)
        );
    }
    let tot = by_cell.values().fold(Cell::default(), |mut a, c| {
        a.blocks += c.blocks;
        a.changed += c.changed;
        a.candidates += c.candidates;
        a.served += c.served;
        a.best += c.best;
        a.served_val += c.served_val;
        a.best_val += c.best_val;
        a
    });
    println!(
        "\n  total {} blocks, {:.2} candidates a block, {:.2} % changed",
        tot.blocks,
        tot.candidates as f64 / tot.blocks as f64,
        100.0 * tot.changed as f64 / tot.blocks as f64
    );
    println!(
        "  conditional cost gain {:.3} %   on reserved activations {:.3} %",
        pct(tot.best, tot.served),
        pct(tot.best_val, tot.served_val)
    );
    println!(
        "\n  the served choice is in the candidate set, so the per-block selection score\n  \
         cannot get worse. The two columns above do NOT treat the suffix alike —\n  \
         the cost re-optimizes it continuously, the reserved one leaves it compensated —\n  \
         so their ratio is a comparison of two different gains, not a transfer rate."
    );

    println!("\n--- CHAINED: the whole row walked under each policy ---");
    println!("  {:<22} {:>5} {:>13} {:>13}", "cell", "rows", "rollout gain", "reserved gain");
    let mut keys: Vec<_> = rows_acc.keys().cloned().collect();
    keys.sort();
    let (mut ts, mut ta, mut vs, mut va, mut nr) = (0.0, 0.0, 0.0, 0.0, 0usize);
    for k in keys {
        let (n, rs, ra, vsd, vad) = rows_acc[&k];
        println!(
            "  {:<16} L{:<4} {n:>5} {:>12.3} % {:>12.3} %",
            k.0,
            k.1,
            pct(ra, rs),
            pct(vad, vsd)
        );
        nr += n;
        ts += rs;
        ta += ra;
        vs += vsd;
        va += vad;
    }
    println!(
        "  {:<22} {nr:>5} {:>12.3} % {:>12.3} %",
        "all",
        pct(ta, ts),
        pct(va, vs)
    );
    println!(
        "\n  here the selector DRIVES the trajectory, so nothing is guaranteed:\n  \
         a per-block gain can compound, cancel or reverse once chained."
    );
    Ok(())
}
