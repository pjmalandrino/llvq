//! # Does the spherical retraction annul the gain code?
//!
//! `g5_shapegain.rs` asserts that `LeechShapeGain` quantizes the magnitude —
//! but it calls `quantize()` **directly**, so it never exercises the
//! retraction that `quantize_layer` applies right afterwards. The real runs
//! all use `retract: true` (Spherical GPTQ).
//!
//! These tests exercise the composed path, which is the one that ships.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{quantize_layer, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, row_scale, LeechShapeGain};

const D_OUT: usize = 8;
const NBLK: usize = 3;
const D_IN: usize = NBLK * DIM;

fn random_hessian(rng: &mut SplitMix64, n: usize, samples: usize) -> Vec<f64> {
    let a: Vec<f64> = (0..samples * n).map(|_| rng.next_gaussian()).collect();
    let mut h = vec![0.0f64; n * n];
    for s in 0..samples {
        let row = &a[s * n..(s + 1) * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] += row[i] * row[j];
            }
        }
    }
    let inv = 1.0 / samples as f64;
    for v in h.iter_mut() {
        *v *= inv;
    }
    for i in 0..n {
        h[i * n + i] += 1e-3;
    }
    h
}

fn weights(rng: &mut SplitMix64) -> Vec<f64> {
    (0..D_OUT * D_IN)
        .map(|k| {
            let row = k / D_IN;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

fn cfg(retract: bool) -> GptqConfig {
    GptqConfig {
        block: DIM,
        retract,
        group_scales: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    }
}

/// Quantize the same matrix through the full layer loop with two wildly
/// different gain codebooks, and report the largest disagreement.
fn spread_between_codebooks(retract: bool) -> f64 {
    let mut rng = SplitMix64::new(0x6_2E77);
    let base = weights(&mut rng);
    let h = random_hessian(&mut rng, D_IN, 4 * D_IN);
    let factor = GptqFactor::new(&h, D_IN, 1e-2).expect("SPD");

    let run = |centroids: Vec<f64>| -> Vec<f64> {
        let mut w = Weights::new(D_OUT, D_IN, base.clone());
        let mut q = LeechShapeGain::new(centroids);
        quantize_layer(&mut w, &factor, None, &mut q, &cfg(retract));
        w.w
    };

    // Two codebooks that share no level and differ by an order of magnitude.
    let a = run(vec![0.25, 0.75]);
    let b = run(vec![6.0, 19.0]);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// The control: **without** the retraction, the gain codebook obviously
/// changes the reconstruction. If this ever fails, the test itself is broken.
#[test]
fn without_retraction_the_gain_codebook_changes_the_output() {
    let spread = spread_between_codebooks(false);
    assert!(
        spread > 1e-6,
        "two disjoint gain codebooks produced identical weights without any \
         retraction ({spread:e}) — the harness is not exercising the gain at all"
    );
}

/// The real path. `retract: true` is what every run ships with.
#[test]
fn with_retraction_the_gain_codebook_still_changes_the_output() {
    let spread = spread_between_codebooks(true);
    assert!(
        spread > 1e-6,
        "SPHERICAL RETRACTION ANNULS THE GAIN CODE: two disjoint gain \
         codebooks ([0.25,0.75] vs [6,19]) produce bit-identical weights \
         (max |Δ| = {spread:e}). The retraction rescales each block back to \
         its pre-quantization norm, so the chosen level cancels out and the \
         stored magnitude is a free float per block — 16 bits that the rate \
         accounting does not charge."
    );
}

/// The rate a configuration claims must be a rate its reconstruction can
/// actually honour.
///
/// Every earlier defect in this project was a *quality* assertion standing in
/// for a *cost* one. This is the cost one: the number of distinct magnitudes
/// the reconstruction produces must fit in the bits the quantizer charges for
/// them.
#[test]
fn the_claimed_rate_matches_what_the_reconstruction_needs() {
    let mut rng = SplitMix64::new(0x6_B175);
    let base = weights(&mut rng);
    let h = random_hessian(&mut rng, D_IN, 4 * D_IN);
    let factor = GptqFactor::new(&h, D_IN, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, D_IN, DIM, 1, 40);

    // How many distinct magnitudes does a given configuration actually emit,
    // per row, and how many does it claim to be able to express?
    let measure = |q: LeechShapeGain| -> (usize, u32) {
        let claimed = q.block_bits();
        let mut w = Weights::new(D_OUT, D_IN, base.clone());
        let mut q = q;
        quantize_layer(&mut w, &factor, None, &mut q, &cfg(true));
        let mut worst = 0usize;
        for i in 0..D_OUT {
            let mut rel: Vec<f64> = Vec::new();
            let rs = row_scale(&base[i * D_IN..(i + 1) * D_IN]);
            for b in w.w[i * D_IN..(i + 1) * D_IN].chunks_exact(DIM) {
                let r = b.iter().map(|a| a * a).sum::<f64>().sqrt() / rs;
                if !rel.iter().any(|d| (d - r).abs() < 1e-9) {
                    rel.push(r);
                }
            }
            worst = worst.max(rel.len());
        }
        (worst, claimed)
    };

    let cap = 12;
    let (levels, bits) = measure(LeechShapeGain::with_shell_cap(centroids.clone(), cap));
    assert_eq!(bits, 47 + 1, "Λ24(12) index plus one gain bit");
    assert!(
        levels <= centroids.len(),
        "claims {bits} bits/block (47 index + 1 gain) but emits {levels} \
         distinct magnitudes per row — {} would be expressible",
        centroids.len()
    );

    // And the free-magnitude variant must *say* it costs the 16 bits it uses.
    let free = LeechShapeGain::with_shell_cap(centroids.clone(), cap).with_free_magnitude();
    let (levels, bits) = measure(free);
    assert_eq!(
        bits,
        47 + 16,
        "a free per-block magnitude costs an f16, and must be charged as one"
    );
    assert!(
        levels > centroids.len(),
        "the free-magnitude variant should emit more magnitudes than the code \
         has levels ({levels} vs {}); if it does not, the A/B is not \
         exercising anything",
        centroids.len()
    );
}

/// Records every row scale the layer loop announces, and introduces enough
/// error that the loop's feedback actually moves the weights.
struct SpyQuantizer {
    seen: Vec<f64>,
}

impl llvq_quant::quantizer::BlockQuantizer for SpyQuantizer {
    fn block_len(&self) -> usize {
        DIM
    }
    fn set_row_scale(&mut self, scale: f64) {
        self.seen.push(scale);
    }
    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        // Coarse rounding: lossy enough that error feedback rewrites the
        // not-yet-quantized columns, which is what would move a row's scale.
        for (o, &a) in out.iter_mut().zip(v.iter()) {
            *o = (a / 0.05).round() * 0.05;
        }
    }
}

/// The per-row scale is charged as **one f16 per output row** by the rate
/// accounting (`Report::rows * 16`). For that to be honest, the loop must
/// announce one scale per row — not one per (row, block).
#[test]
fn the_row_scale_is_announced_once_per_row_not_once_per_block() {
    let mut rng = SplitMix64::new(0x6_5CA1);
    let base = weights(&mut rng);
    let h = random_hessian(&mut rng, D_IN, 4 * D_IN);
    let factor = GptqFactor::new(&h, D_IN, 1e-2).expect("SPD");

    let mut w = Weights::new(D_OUT, D_IN, base.clone());
    let mut spy = SpyQuantizer { seen: Vec::new() };
    quantize_layer(&mut w, &factor, None, &mut spy, &cfg(true));

    assert_eq!(
        spy.seen.len(),
        D_OUT * NBLK,
        "expected one announcement per (row, block)"
    );
    // Blocks are the outer loop, rows the inner one: seen[b * D_OUT + i].
    for i in 0..D_OUT {
        let first = spy.seen[i];
        for b in 1..NBLK {
            let got = spy.seen[b * D_OUT + i];
            assert!(
                (got - first).abs() <= 1e-12 * first.abs(),
                "row {i}: the scale changed between block 0 ({first}) and \
                 block {b} ({got}). A scale that varies per block cannot be \
                 stored as one f16 per row, so the artifact cannot reproduce \
                 the evaluated weights — and the rate accounting undercharges."
            );
        }
    }
}

/// The property `the_gain_is_actually_quantized` means to assert, checked on
/// the composed path instead of on the bare quantizer: after a real layer
/// pass, a block's magnitude must still be one of the code's finitely many
/// levels.
#[test]
fn block_magnitudes_after_a_layer_pass_are_still_quantized() {
    let mut rng = SplitMix64::new(0x6_1234);
    let base = weights(&mut rng);
    let h = random_hessian(&mut rng, D_IN, 4 * D_IN);
    let factor = GptqFactor::new(&h, D_IN, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, D_IN, DIM, 1, 40);

    let mut w = Weights::new(D_OUT, D_IN, base.clone());
    let mut q = LeechShapeGain::new(centroids.clone());
    quantize_layer(&mut w, &factor, None, &mut q, &cfg(true));

    // Count how many distinct relative magnitudes the blocks actually take.
    // The reference is the scale of the row **as it entered** the loop, which
    // is the one the artifact stores — not the scale of the quantized row.
    let mut rel: Vec<f64> = Vec::new();
    for i in 0..D_OUT {
        let row = &w.w[i * D_IN..(i + 1) * D_IN];
        let rs = row_scale(&base[i * D_IN..(i + 1) * D_IN]);
        for b in row.chunks_exact(DIM) {
            let n = b.iter().map(|a| a * a).sum::<f64>().sqrt();
            let r = n / rs;
            if !rel.iter().any(|d| (d - r).abs() < 1e-9) {
                rel.push(r);
            }
        }
    }
    assert!(
        rel.len() <= centroids.len(),
        "after a layer pass the blocks take {} distinct relative magnitudes, \
         but the gain code has only {} levels — the magnitudes are free \
         floats, not code levels",
        rel.len(),
        centroids.len()
    );
}
