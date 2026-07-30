//! Layer Hessians, and the sequential block loop that consumes them.
//!
//! ## Why the loop is sequential
//!
//! GPTQ quantizes block `t` against the activations that actually reach it —
//! which means the activations that have already flowed through blocks
//! `0..t` **in their quantized form**. Calibrating every layer against the
//! FP16 model instead is simpler and measurably worse: each layer would be
//! corrected for an input distribution the deployed model never sees.
//!
//! So each block costs two forward passes over the calibration set: one with
//! the original weights to collect `H`, and one with the quantized weights to
//! produce the input of the next block.
//!
//! ## Where the Hessians come from
//!
//! `H = AᵀA/N` is accumulated as a matmul on the device — `n × n` for
//! `n` up to 3072, which is a GEMM, not something to loop in scalar Rust.
//! Accumulation is in f32 (as every GPTQ implementation does) with each
//! window pre-scaled by `1/N` so the running sum stays O(1) instead of
//! growing with the sample count; the conversion to f64 happens once, at the
//! factorization.

use crate::model::{Act, Capture, Qwen3};
use candle_core::{DType, Device, Tensor};
use llvq_quant::gptq::{quantize_layer_parallel, GptqConfig, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{
    fit_gain_centroids, BlockQuantizer, Identity, LeechDirection, LeechShapeGain, ScalarGrid,
};
use llvq_quant::rotation::Rotation;
use std::collections::HashMap;

/// Running `AᵀA/N` for one activation.
pub struct Hessian {
    sum: Tensor,
    scale: f64,
}

impl Hessian {
    pub fn new(width: usize, device: &Device, total_rows: usize) -> candle_core::Result<Self> {
        Ok(Self {
            sum: Tensor::zeros((width, width), DType::F32, device)?,
            scale: 1.0 / total_rows as f64,
        })
    }

    /// `x` is `(batch, seq, width)`.
    pub fn accumulate(&mut self, x: &Tensor) -> candle_core::Result<()> {
        let w = x.dim(x.rank() - 1)?;
        let a = x.reshape(((), w))?.to_dtype(DType::F32)?.contiguous()?;
        let g = (a.t()?.contiguous()?.matmul(&a)? * self.scale)?;
        self.sum = (&self.sum + g)?;
        Ok(())
    }

    /// Dense row-major `n × n`, in f64, ready for [`GptqFactor`].
    pub fn to_f64(&self) -> candle_core::Result<Vec<f64>> {
        Ok(self
            .sum
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| v as f64)
            .collect())
    }
}

/// Collects the four Hessians of a single block.
struct BlockCapture {
    target: usize,
    acc: HashMap<Act, Hessian>,
}

impl Capture for BlockCapture {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == self.target {
            if let Some(h) = self.acc.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

/// What a quantization run reports back.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub matrices: usize,
    pub weights: u64,
    /// Weights that stayed at full precision because they fell in a tail.
    pub tail_weights: u64,
    /// Output rows, each carrying one f16 scale.
    pub rows: u64,
    /// Bits per quantized block, gain included.
    pub block_bits: f64,
    pub seconds: f64,
}

impl Report {
    /// Effective bits per weight — **everything that would be written to
    /// disk**: the block code (index + gain), 16 bits for each full-precision
    /// tail weight, and one f16 scale per output row.
    ///
    /// Counting only the index is how a 2.73 bit/weight run got reported as
    /// 2.07; the magnitude has to be in here.
    pub fn bits_per_weight(&self) -> f64 {
        let quantized = self.weights - self.tail_weights;
        let blocks = quantized as f64 / llvq_core::DIM as f64;
        (blocks * self.block_bits + self.tail_weights as f64 * 16.0 + self.rows as f64 * 16.0)
            / self.weights as f64
    }
}

/// Which codebook to quantize with.
///
/// An enum rather than a factory closure because one of them —
/// [`Codebook::ShapeGain`] — has to be **fitted to each matrix**: its gain
/// levels come from that matrix's own block magnitudes. A closure built once
/// for the whole run cannot do that.
#[derive(Clone, Copy, Debug)]
pub enum Codebook {
    /// Lossless. The control that proves the pipeline is a no-op when the
    /// codebook is.
    Identity,
    /// Round-to-nearest on a fixed grid.
    Grid { step: f64 },
    /// Leech direction with the block magnitude left in full precision.
    /// **Not** a 2 bit/weight code — see `LeechDirection`.
    Direction,
    /// Leech direction plus a `gain_bits`-bit magnitude code, relative to a
    /// per-row scale. The honest 2 bit/weight configuration.
    ///
    /// `max_shell` restricts the direction code: 13 is the full ball (48-bit
    /// index), 12 costs 47 bits and pays for the gain bit at the same rate.
    ShapeGain { gain_bits: u32, max_shell: u32 },
}

impl Codebook {
    /// Bits per 24-weight block, gain included. The per-row scale is one f16
    /// per output row and is counted separately by [`Report`].
    pub fn block_bits(&self) -> f64 {
        match self {
            Codebook::Identity | Codebook::Grid { .. } => 24.0 * 16.0,
            Codebook::Direction => 48.0 + 16.0,
            Codebook::ShapeGain {
                gain_bits,
                max_shell,
            } => (llvq_quant::quantizer::index_bits(*max_shell) + *gain_bits) as f64,
        }
    }
}

/// How to run [`quantize_model`].
pub struct RunConfig {
    pub gptq: GptqConfig,
    /// Hessian damping, relative to `mean(diag H)`.
    pub damping: f64,
    pub codebook: Codebook,
    pub threads: usize,
    /// Quantize only the first `limit` blocks; later ones are left untouched
    /// but still advanced. Diagnostics only — `usize::MAX` for a real run.
    pub limit: usize,
    /// Seed for the incoherence rotation applied to each linear's **input**
    /// basis, or `None` to quantize in the natural basis.
    ///
    /// The rotation is per (block, activation), so the matrices that share an
    /// input — q/k/v, and gate/up — necessarily share it too, which is what
    /// makes rotating the single shared Hessian legitimate.
    pub rotation_seed: Option<u64>,
}

/// Quantize every block of `model` in place, sequentially.
///
/// `hidden` holds one `(1, seq, hidden_size)` tensor per calibration window,
/// entering block 0; it is advanced in place as the loop proceeds.
pub fn quantize_model(
    model: &mut Qwen3,
    hidden: &mut [Tensor],
    run: &RunConfig,
    mut progress: impl FnMut(usize, usize, &str),
) -> anyhow::Result<Report> {
    let RunConfig {
        gptq: cfg,
        damping,
        codebook,
        threads,
        limit,
        rotation_seed,
    } = run;
    let codebook = *codebook;
    let (damping, threads, limit) = (*damping, *threads, *limit);
    let rotation_seed = *rotation_seed;
    let t0 = std::time::Instant::now();
    let mut report = Report::default();
    let device = model.device().clone();
    let nblocks = model.blocks.len();
    let total_rows: usize = hidden.iter().map(|h| h.dim(1).unwrap_or(0)).sum();
    anyhow::ensure!(total_rows > 0, "no calibration data");

    for t in 0..nblocks {
        if t >= limit {
            let mut none = crate::model::NoCapture;
            let mask = model.causal_mask_for(&hidden[0])?;
            for h in hidden.iter_mut() {
                let next = model.blocks[t].forward(&*h, model.rotary(), &mask, t, &mut none)?;
                *h = next;
            }
            continue;
        }
        // ---- pass 1: collect H with the original weights ----
        let mut cap = BlockCapture {
            target: t,
            acc: HashMap::new(),
        };
        for act in Act::ALL {
            let w = act.width(model.config());
            cap.acc.insert(act, Hessian::new(w, &device, total_rows)?);
        }
        let mask = model.causal_mask_for(&hidden[0])?;
        for h in hidden.iter() {
            let _ = model.blocks[t].forward(h, model.rotary(), &mask, t, &mut cap)?;
        }

        // ---- factor once per activation, not once per matrix ----
        let mut factors: HashMap<Act, (GptqFactor, Vec<f64>, Option<Rotation>)> =
            HashMap::new();
        for act in Act::ALL {
            let mut h = cap.acc[&act].to_f64()?;
            let n = act.width(model.config());
            // Quantizing in a rotated basis means the Hessian has to move
            // with it: H' = Q H Qᵀ, since x' = Q x.
            let rot = rotation_seed.map(|s| {
                Rotation::new(n, s ^ ((t as u64) << 32) ^ (act.index() << 16))
            });
            if let Some(q) = &rot {
                q.rotate_hessian(&mut h);
            }
            let f = GptqFactor::new(&h, n, damping)
                .map_err(|e| anyhow::anyhow!("block {t}, {act:?}: {e}"))?;
            factors.insert(act, (f, h, rot));
        }

        // ---- quantize the seven matrices ----
        for act in Act::ALL {
            let (factor, hmat, rot) = &factors[&act];
            for name in act.consumers() {
                let lin = model.blocks[t].linear_mut(name);
                let w = lin.weight();
                let (d_out, d_in) = w.dims2()?;
                let flat: Vec<f64> = w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                let mut weights = Weights::new(d_out, d_in, flat);
                // W' = W Qᵀ, quantize there, then Ŵ = Ŵ' Q — a drop-in
                // replacement that needs no runtime transform.
                if let Some(q) = rot {
                    q.rotate_weight_rows(&mut weights.w, d_out);
                }
                // The gain levels are fitted to *this* matrix, in the basis it
                // will be quantized in.
                let gain = match codebook {
                    Codebook::ShapeGain { gain_bits, .. } => Some(fit_gain_centroids(
                        &weights.w,
                        d_out,
                        d_in,
                        cfg.block,
                        gain_bits,
                        40,
                    )),
                    _ => None,
                };
                let make = move || -> Box<dyn BlockQuantizer> {
                    match codebook {
                        Codebook::Identity => Box::new(Identity { block: cfg.block }),
                        Codebook::Grid { step } => Box::new(ScalarGrid {
                            block: cfg.block,
                            step,
                        }),
                        Codebook::Direction => Box::new(LeechDirection::new()),
                        Codebook::ShapeGain { max_shell, .. } => {
                            Box::new(LeechShapeGain::with_shell_cap(
                                gain.clone().expect("fitted above"),
                                max_shell,
                            ))
                        }
                    }
                };
                quantize_layer_parallel(
                    &mut weights,
                    factor,
                    cfg.group_scales.then_some(hmat.as_slice()),
                    &make,
                    cfg,
                    threads,
                );
                if let Some(q) = rot {
                    q.unrotate_weight_rows(&mut weights.w, d_out);
                }
                let recon: Vec<f32> = weights.w.iter().map(|v| *v as f32).collect();
                let t2 = Tensor::from_vec(recon, (d_out, d_in), &device)?
                    .to_dtype(w.dtype())?;
                *lin = candle_nn::Linear::new(t2, None);

                report.matrices += 1;
                report.weights += (d_out * d_in) as u64;
                report.tail_weights += (d_out * (d_in % cfg.block)) as u64;
                report.rows += d_out as u64;
                report.block_bits = codebook.block_bits();
                progress(t, nblocks, name);
            }
        }

        // ---- pass 2: advance the activations through the quantized block ----
        let mut none = crate::model::NoCapture;
        for h in hidden.iter_mut() {
            let next = model.blocks[t].forward(&*h, model.rotary(), &mask, t, &mut none)?;
            *h = next;
        }
    }

    report.seconds = t0.elapsed().as_secs_f64();
    Ok(report)
}
