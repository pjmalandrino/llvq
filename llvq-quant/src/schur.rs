//! Gains-only diagnostics at a common GPTQ state. No model or runtime policy changes.
//!
//! The factor includes rotation, shrinkage and damping supplied by the caller.
//! Bounds and rollout losses refer to that regularized metric. Validation uses
//! separate activation rows in the same coordinate basis.

use crate::linalg::GptqFactor;
use crate::quantizer::{row_scale, BlockCode, BlockQuantizer, TetraShapeGain};
use llvq_core::DIM;

/// Conditional cost after eliminating the continuous suffix: ||e U_BB^-1||².
/// The fixed prefix is excluded; add its accumulated cost for a global bound.
pub fn continuous_lower_bound(factor: &GptqFactor, start: usize, residual: &[f64]) -> f64 {
    assert!(!residual.is_empty() && start + residual.len() <= factor.dim());
    let mut v = residual.to_vec();
    factor.solve_block(&mut v, 1, start, residual.len());
    v.iter().map(|x| x * x).sum()
}

/// Commit one block and propagate its actual residual in production loop order.
/// This also works for arbitrary block sizes in the independent algebra tests.
pub fn commit(factor: &GptqFactor, working: &mut [f64], start: usize, q: &[f64]) -> f64 {
    assert_eq!(working.len(), factor.dim());
    assert!(!q.is_empty() && start + q.len() <= working.len());
    let end = start + q.len();
    let mut e: Vec<_> = working[start..end]
        .iter()
        .zip(q)
        .map(|(x, q)| x - q)
        .collect();
    working[start..end].copy_from_slice(q);
    factor.solve_block(&mut e, 1, start, q.len());
    for (k, &v) in e.iter().enumerate() {
        if v != 0.0 {
            for (j, x) in working[end..].iter_mut().enumerate() {
                *x -= v * factor.u(start + k, end + j);
            }
        }
    }
    e.iter().map(|v| v * v).sum()
}

/// Full regularized loss; H = U^-1 U^-T. No inverse is materialized.
pub fn rollout_loss(factor: &GptqFactor, original: &[f64], q: &[f64]) -> f64 {
    assert_eq!(original.len(), factor.dim());
    assert_eq!(q.len(), original.len());
    let e: Vec<_> = original.iter().zip(q).map(|(w, q)| q - w).collect();
    continuous_lower_bound(factor, 0, &e)
}

/// Mean squared projection-output error on reserved activation rows.
pub fn validation_loss(activations: &[f64], original: &[f64], q: &[f64]) -> f64 {
    assert_eq!(original.len(), q.len());
    assert!(!original.is_empty() && !activations.is_empty());
    assert_eq!(activations.len() % original.len(), 0);
    let n = original.len();
    activations
        .chunks_exact(n)
        .map(|a| {
            a.iter()
                .zip(original.iter().zip(q))
                .map(|(a, (w, q))| a * (q - w))
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        / (activations.len() / n) as f64
}

#[derive(Debug, Clone)]
pub struct GainComparison {
    pub block: usize,
    pub source_norm: f64,
    pub projected_gain: f64,
    pub amplitudes: [f64; 2],
    pub codes: [BlockCode; 2],
    pub reconstructed: [[f64; DIM]; 2],
    /// A: source norm; B: post-shape Euclidean projection; C: Schur.
    pub choices: [usize; 3],
    pub euclidean: [f64; 2],
    /// Costs excluding the constant fixed-prefix contribution.
    pub conditional: [f64; 2],
    pub prefix_loss: f64,
}

fn nearest(levels: &[f64; 2], x: f64) -> usize {
    usize::from((x - levels[1]).abs() < (x - levels[0]).abs())
}

/// One direction search via the witness quantizer, then two reconstructions.
/// `quant` must use the source-norm policy and the given fixed scale and levels.
fn compare(
    quant: &mut TetraShapeGain,
    factor: &GptqFactor,
    working: &[f64],
    block: usize,
    levels: [f64; 2],
    scale: f64,
    prefix_loss: f64,
) -> GainComparison {
    let start = block * DIM;
    let x = &working[start..start + DIM];
    let mut witness = [0.0; DIM];
    quant.quantize(x, &mut witness);
    let code = quant.last_code().expect("Tetra always emits a code");
    let codes = [BlockCode { gain: 0, ..code }, BlockCode { gain: 1, ..code }];
    let mut reconstructed = [[0.0; DIM]; 2];
    for g in 0..2 {
        quant.reconstruct(&codes[g], scale, &mut reconstructed[g]);
    }
    let source_norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm = code
        .point
        .iter()
        .map(|&p| (p as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let projected_gain = if norm == 0.0 {
        0.0
    } else {
        x.iter()
            .zip(code.point)
            .map(|(a, p)| a * p as f64)
            .sum::<f64>()
            / norm
    };
    let euclidean = core::array::from_fn(|g| {
        x.iter()
            .zip(reconstructed[g])
            .map(|(a, q)| (a - q).powi(2))
            .sum()
    });
    let conditional = core::array::from_fn(|g| {
        let e: Vec<_> = x.iter().zip(reconstructed[g]).map(|(a, q)| a - q).collect();
        continuous_lower_bound(factor, start, &e)
    });
    GainComparison {
        block,
        source_norm,
        projected_gain,
        amplitudes: levels.map(|g| g * scale),
        codes,
        reconstructed,
        choices: [
            code.gain as usize,
            if norm == 0.0 {
                0
            } else {
                nearest(&levels, projected_gain / scale)
            },
            usize::from(conditional[1] < conditional[0]),
        ],
        euclidean,
        conditional,
        prefix_loss,
    }
}

#[derive(Debug, Clone)]
pub struct Rollout {
    pub gain: usize,
    pub codes: Vec<BlockCode>,
    pub reconstructed: Vec<f64>,
    pub rollout_loss: f64,
    pub continuous_lower_bound: f64,
    pub rollout_excess: f64,
    pub validation_loss: f64,
    pub encoder_calls: usize,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Prefix is committed; suffix is the compensated conditional center.
    pub working: Vec<f64>,
    pub prefix_codes: Vec<BlockCode>,
    pub comparison: GainComparison,
    pub branches: [Rollout; 2],
}

#[derive(Debug, Clone)]
pub struct RowDiagnostic {
    pub original: Vec<f64>,
    pub row_scale: f64,
    pub shadow: Vec<GainComparison>,
    pub snapshots: Vec<Snapshot>,
    pub witness: Vec<f64>,
    pub witness_codes: Vec<BlockCode>,
    pub encoder_calls: usize,
}

/// Walk a source-norm witness, shadow every full block, branch only at `blocks`.
/// Each branch resumes the source-norm policy with its own scratch and last_code.
/// The final partial block stays at its compensated value (KeepExact).
/// No group scales, Design C, candidate-direction search or policy adaptation.
pub fn diagnose_row(
    original: &[f64],
    factor: &GptqFactor,
    levels: [f64; 2],
    validation: &[f64],
    blocks: &[usize],
) -> RowDiagnostic {
    assert_eq!(original.len(), factor.dim());
    assert!(original.len() >= DIM && original.iter().all(|v| v.is_finite()));
    assert!(levels.iter().all(|v| v.is_finite() && *v >= 0.0) && levels[0] <= levels[1]);
    assert!(!validation.is_empty() && validation.len().is_multiple_of(original.len()));
    assert!(validation.iter().all(|v| v.is_finite()));
    let nblocks = original.len() / DIM;
    assert!(blocks.iter().all(|&b| b < nblocks));
    assert!(
        blocks.windows(2).all(|b| b[0] < b[1]),
        "snapshot blocks must be sorted and unique"
    );
    let scale = row_scale(original);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let encoder = TetraShapeGain::encoder();
    let make = || {
        let mut q = TetraShapeGain::with_encoder(encoder.clone(), levels.to_vec());
        q.set_row_scale(scale);
        q
    };
    let mut quant = make();
    let mut working = original.to_vec();
    let mut prefix_codes = Vec::new();
    let mut shadow = Vec::new();
    let mut snapshots = Vec::new();
    let mut prefix_loss = 0.0;
    let mut encoder_calls = nblocks;
    for block in 0..nblocks {
        let c = compare(
            &mut quant,
            factor,
            &working,
            block,
            levels,
            scale,
            prefix_loss,
        );
        if blocks.binary_search(&block).is_ok() {
            let branches = core::array::from_fn(|gain| {
                let mut q = working.clone();
                let mut branch_quant = make();
                let mut codes = prefix_codes.clone();
                codes.push(c.codes[gain]);
                commit(factor, &mut q, block * DIM, &c.reconstructed[gain]);
                for next in block + 1..nblocks {
                    let mut out = [0.0; DIM];
                    branch_quant.quantize(&q[next * DIM..(next + 1) * DIM], &mut out);
                    codes.push(branch_quant.last_code().expect("Tetra code"));
                    commit(factor, &mut q, next * DIM, &out);
                }
                let loss = rollout_loss(factor, original, &q);
                let bound = prefix_loss + c.conditional[gain];
                Rollout {
                    gain,
                    codes,
                    validation_loss: validation_loss(validation, original, &q),
                    reconstructed: q,
                    rollout_loss: loss,
                    continuous_lower_bound: bound,
                    rollout_excess: loss - bound,
                    encoder_calls: nblocks - block - 1,
                }
            });
            encoder_calls += branches.iter().map(|b| b.encoder_calls).sum::<usize>();
            snapshots.push(Snapshot {
                working: working.clone(),
                prefix_codes: prefix_codes.clone(),
                comparison: c.clone(),
                branches,
            });
        }
        let gain = c.choices[0];
        prefix_loss += commit(factor, &mut working, block * DIM, &c.reconstructed[gain]);
        prefix_codes.push(c.codes[gain]);
        shadow.push(c);
    }
    RowDiagnostic {
        original: original.to_vec(),
        row_scale: scale,
        shadow,
        snapshots,
        witness: working,
        witness_codes: prefix_codes,
        encoder_calls,
    }
}
