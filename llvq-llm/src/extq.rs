//! The per-matrix chain of [`crate::calib::quantize_model_guided`], for a
//! `(W, H)` pair that comes from outside the Qwen3 block loop.
//!
//! The block loop does four things per activation and per matrix that do not
//! depend on the model around them:
//!
//! 1. rotate the Hessian into the incoherence basis, `H' = Q H Qᵀ`, then damp
//!    and factor it (`calib.rs`, the `factor once per activation` section);
//! 2. rotate the weight rows, `W' = W Qᵀ`, fit the two gain centroids on `W'`,
//!    freeze the row scales;
//! 3. run the row-parallel lattice quantization with error feedback;
//! 4. round the `KeepExact` tail to f32, un-rotate, narrow to f32: the dense
//!    reconstruction `mmlu` and `ppl` score.
//!
//! This module repeats those steps in the same order with the same calls, and
//! nothing else. It is not a second implementation of the quantizer: every
//! numeric step is a call into `llvq_quant`. `tests/extq_equiv.rs` holds it
//! to bit equality with the block loop on a tiny model, so a drift between
//! the two is a red test, not a silent change of recipe.
//!
//! What it leaves out, because it belongs to the loop and not to a matrix:
//! the capture of `H` (the caller brings it), the sequential advance through
//! quantized blocks, the int4 records, the artifact sink.

use crate::calib::Codebook;
use llvq_quant::gptq::{GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::{GptqFactor, NotPositiveDefinite};
use llvq_quant::quantizer::{
    fit_gain_centroids, reconstruct_shape_gain, row_scale, BlockCode, BlockQuantizer,
    TetraShapeGain,
};
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::Encoder;
use std::sync::Arc;

/// The encoder settings of one run.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub codebook: Codebook,
    pub gptq: GptqConfig,
    /// Hessian damping, relative to `mean(diag H)`.
    pub damping: f64,
    /// Multiplier of the fitted centroids; `1.0` skips the multiply.
    pub gain_scale: f64,
    pub threads: usize,
}

impl Settings {
    /// The sealed Tetra recipe, as `bin/smoke` builds it for
    /// `smoke 64 2048 12 4096 <dev> nogs tetra1 999 rot` with no `LLVQ_DAMPING`,
    /// `LLVQ_H_SHRINK` or `LLVQ_GAIN_SCALE` set (`bin/smoke/`).
    pub fn sealed_recipe(threads: usize) -> Self {
        let codebook = Codebook::Tetra {
            gain_bits: 1,
            post_shape_gain: false,
        };
        Self {
            codebook,
            gptq: GptqConfig {
                block: codebook.block_len(),
                retract: true,
                group_scales: false,
                design_c: false,
                spherical_feedback: false,
                lambda: 1e-2,
                tail: TailPolicy::KeepExact,
            },
            damping: 1e-2,
            gain_scale: 1.0,
            threads,
        }
    }
}

/// A Hessian made ready for the block loop: its factor, in the basis the
/// weights will be quantized in.
pub struct Factored {
    pub n: usize,
    pub factor: GptqFactor,
    pub rotation: Option<Rotation>,
}

/// Step 1. `h` is dense row-major `n × n`, in the natural basis.
///
/// The same two calls as the loop, in the same order: `rotate_hessian`, then
/// `GptqFactor::new(.., damping)`. `h_shrink = 1` is the published path and
/// is skipped there, so it is not offered here.
pub fn factor_hessian(
    mut h: Vec<f64>,
    n: usize,
    rotation_seed: Option<u64>,
    damping: f64,
) -> Result<Factored, NotPositiveDefinite> {
    assert_eq!(h.len(), n * n, "H must be n × n");
    let rotation = rotation_seed.map(|s| Rotation::new(n, s));
    if let Some(q) = &rotation {
        q.rotate_hessian(&mut h);
    }
    let factor = GptqFactor::new(&h, n, damping)?;
    Ok(Factored {
        n,
        factor,
        rotation,
    })
}

/// What one matrix encodes to.
pub struct Encoded {
    /// The dense reconstruction in the natural basis, f32, row-major: the
    /// weights the evaluated model holds after the loop.
    pub recon: Vec<f32>,
    /// One per output row, frozen on the rotated weights before the loop.
    pub row_scales: Vec<f64>,
    /// The two gain centroids, relative to the row scale.
    pub centroids: Vec<f64>,
    /// `d_out × (d_in / 24)` block codes, row-major.
    pub codes: Vec<BlockCode>,
    /// Tail columns in the rotated basis, already narrowed to f32.
    pub tail: Vec<f64>,
}

/// Steps 2 to 4 for one matrix, `w` row-major `d_out × d_in` in the natural
/// basis, against a factor of the Hessian of its input.
///
/// Only the codebooks a sealed file can hold are accepted (Tetra, and the
/// ball shape–gain with a coded gain), because the reconstruction returned
/// here is claimed to be what a decoder rebuilds from the codes, and
/// [`Encoded`] is checked against that decode before it is returned.
pub fn encode_matrix(
    w: &[f32],
    d_out: usize,
    d_in: usize,
    f: &Factored,
    s: &Settings,
    tetra_encoder: &Arc<Encoder>,
) -> anyhow::Result<Encoded> {
    anyhow::ensure!(w.len() == d_out * d_in, "W is not {d_out} × {d_in}");
    anyhow::ensure!(f.n == d_in, "factor of width {} for a matrix of width {d_in}", f.n);
    anyhow::ensure!(
        !s.gptq.group_scales && !s.gptq.design_c,
        "group scales and design C read a dense H; this path keeps only the factor"
    );
    let gain_bits = match s.codebook {
        Codebook::Tetra {
            gain_bits,
            post_shape_gain: false,
        } => gain_bits,
        other => anyhow::bail!("extq encodes the sealed Tetra recipe only, not {other:?}"),
    };
    let cfg = s.gptq;
    let flat: Vec<f64> = w.iter().map(|v| *v as f64).collect();
    let mut weights = Weights::new(d_out, d_in, flat);
    if let Some(q) = &f.rotation {
        q.rotate_weight_rows(&mut weights.w, d_out);
    }
    let mut centroids = fit_gain_centroids(&weights.w, d_out, d_in, cfg.block, gain_bits, 40);
    if s.gain_scale != 1.0 {
        for c in &mut centroids {
            *c *= s.gain_scale;
        }
    }
    let enc = tetra_encoder.clone();
    let levels = centroids.clone();
    let make = move || -> Box<dyn BlockQuantizer> {
        Box::new(TetraShapeGain::with_encoder(enc.clone(), levels.clone()))
    };
    let row_scales: Vec<f64> = (0..d_out)
        .map(|i| row_scale(&weights.w[i * d_in..(i + 1) * d_in]))
        .collect();
    let row_blocks = d_in / cfg.block;
    let mut codes: Vec<Option<BlockCode>> = vec![None; d_out * row_blocks];
    llvq_quant::gptq::quantize_layer_parallel_capturing(
        &mut weights,
        &f.factor,
        None,
        &make,
        &cfg,
        s.threads,
        Some(&mut codes),
    );
    let tail_w = d_in % cfg.block;
    let mut tail = Vec::with_capacity(d_out * tail_w);
    for i in 0..d_out {
        let at = i * d_in + row_blocks * cfg.block;
        for v in weights.w[at..at + tail_w].iter_mut() {
            *v = *v as f32 as f64;
            tail.push(*v);
        }
    }
    let codes = codes
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("a quantized block emitted no code"))?;
    // The decoder's view, block by block, in the rotated basis: the same
    // check `smoke`'s `verify_artifact` makes on every file it writes.
    let mut out = [0.0f64; llvq_core::DIM];
    for i in 0..d_out {
        for p in 0..row_blocks {
            reconstruct_shape_gain(&codes[i * row_blocks + p], &centroids, row_scales[i], &mut out);
            let at = i * d_in + p * cfg.block;
            anyhow::ensure!(
                weights.w[at..at + cfg.block]
                    .iter()
                    .zip(&out)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "row {i}, block {p}: the loop's weights differ from the decode of their code"
            );
        }
    }
    if let Some(q) = &f.rotation {
        q.unrotate_weight_rows(&mut weights.w, d_out);
    }
    let recon: Vec<f32> = weights.w.iter().map(|v| *v as f32).collect();
    Ok(Encoded {
        recon,
        row_scales,
        centroids,
        codes,
        tail,
    })
}

/// `tr(ΔW H ΔWᵀ)` and `tr(W H Wᵀ)` in the natural basis, split by rows over
/// `threads`. `h` is `d_in × d_in`, row-major; any positive scale of `H`
/// cancels in their ratio.
pub fn proxy_traces(
    w: &[f32],
    w_hat: &[f32],
    d_out: usize,
    d_in: usize,
    h: &[f64],
    threads: usize,
) -> (f64, f64) {
    assert_eq!(h.len(), d_in * d_in);
    let threads = threads.max(1).min(d_out);
    let rows_per = d_out.div_ceil(threads);
    let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    let mut d = vec![0.0f64; d_in];
                    let mut x = vec![0.0f64; d_in];
                    for i in t * rows_per..((t + 1) * rows_per).min(d_out) {
                        for k in 0..d_in {
                            x[k] = w[i * d_in + k] as f64;
                            d[k] = w_hat[i * d_in + k] as f64 - x[k];
                        }
                        for a in 0..d_in {
                            let hr = &h[a * d_in..(a + 1) * d_in];
                            let (mut sd, mut sx) = (0.0, 0.0);
                            for b in 0..d_in {
                                sd += hr[b] * d[b];
                                sx += hr[b] * x[b];
                            }
                            num += d[a] * sd;
                            den += x[a] * sx;
                        }
                    }
                    (num, den)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("trace thread")).collect()
    });
    parts.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}
