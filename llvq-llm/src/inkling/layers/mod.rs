//! The pieces of an Inkling 2 Mini decoder layer, f32, for one sequence.
//!
//! Every function cites the reference line it reproduces: transformers
//! 046bdb8cc2, `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`
//! ("line N" below). Activations are `[T, width]` for one sequence; heads are
//! split as `[T, heads, dim]` and attention runs per head as `[heads, T, dim]`.
//!
//! One code path serves the prompt and the decode step. Each layer visit owns
//! a [`SlotCache`]: its keys and values after their norms, and the last `K - 1`
//! inputs of each of its four convolutions. A prompt runs on an empty slot
//! (the convolution history is then zeros, which is the reference's left
//! padding, lines 316-322 and 444), a step on the slot the prompt filled
//! (`causal_conv1d_update`, lines 286-302, and `cache.update`, line 612).
//!
//! | file | holds |
//! |---|---|
//! | this file | the norms, the clip, the log scale, the block residual, the embedding, `linear` |
//! | `attn` | [`AttnWeights`] in its three holdings, the score bias, the window switch, the forward |
//! | `cache` | [`SlotCache`] with its growth, its rollback marks and its snapshot form |
//! | `conv` | [`ConvState`], the short and the dynamic convolutions |
//! | `decoder` | the message gates and [`LayerWeights`], with its candle and fused forwards |
//! | `moe` | the two expert seams, the routing, the dense expert sum |
//! | `phases` | the wall-time split of a forward, for `inkrun` |
//!
//! Each child imports what it uses by name. This file re-exports only what
//! the rest of the workspace names, plus the types its public signatures carry.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

use super::config::InklingConfig;
use super::loader::NativeCheckpoint;

#[cfg(ink_glue)]
use super::{attn_bias, attn_dense, attn_scores, glue};
use super::snapshot;

mod attn;
mod cache;
mod conv;
mod decoder;
mod moe;

pub use attn::{
    score_bias_host, set_window_scores, window_score_chunks, AttnMats, AttnWeights, CpuAttnProj, KvConv,
    DEFAULT_ATTN_CHUNK,
};
pub use cache::{SlotCache, SlotMark, DEFAULT_KV_SLACK};
pub use conv::{dynamic_conv, short_conv, ConvState};
pub use decoder::{LayerWeights, MessageGate, MessageGates};
pub use moe::{experts_forward, route, ExpertSource, MoeSource, Routing};

/// Wall time of a forward split by part, for the measurement flag of
/// `inkrun`. Off by default: then [`mark`] is one atomic load and changes
/// nothing. On, each mark drains the device and charges the time since the
/// previous mark to the part just done, so the parts sum to the wall time of
/// the marked span, slowed by the drains.
pub mod phases;

/// `x @ w.T` for `x` `[T, in]` and `w` `[out, in]` (`nn.Linear` without bias).
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// `Inkling2LoopUnweightedRMSNorm`, lines 67-76: `x * rsqrt(mean(x^2) + eps)` on the last dim.
pub fn rms_unweighted(x: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?;
    x.broadcast_mul(&ms.sqrt()?.recip()?)
}

/// `Inkling2LoopRMSNorm`, lines 730-745: `weight * (x * rsqrt(mean(x^2) + eps))`.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    rms_unweighted(x, eps)?.broadcast_mul(w)
}

/// `Inkling2LoopL2Clip`, lines 176-187: each vector on the last dim scaled by
/// `min(threshold / (||x|| + 1e-6), 1)`.
pub fn l2_clip(x: &Tensor, threshold: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let f = ((norm + 1e-6)?.recip()? * threshold)?.minimum(1.0)?;
    x.broadcast_mul(&f)
}

/// `Inkling2LoopLogScale`, lines 205-220: `1 + alpha * ln(max((pos + 1) / n_floor, 1))`, as `[T, 1, 1]`.
pub fn log_scale(cfg: &InklingConfig, positions: &[usize], dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = positions
        .iter()
        .map(|&p| {
            let r = ((p + 1) as f32 / cfg.log_scaling_n_floor as f32).max(1.0);
            1.0 + cfg.log_scaling_alpha as f32 * r.ln()
        })
        .collect();
    Tensor::from_vec(v, (positions.len(), 1, 1), dev)
}

/// `Inkling2LoopBlockAttentionResidual.forward`, lines 838-853 (and the
/// message residual of lines 158-173 has the same body): the block states and
/// the partial state stacked, unweighted-normed (keys AND values), one
/// softmax over the stack per query, scaled by `hidden^-0.5`. `queries` is
/// `[nq, H]`; returns `nq` tensors `[T, H]`.
pub fn block_attn_res(queries: &Tensor, states: &[Tensor], partial: &Tensor) -> Result<Vec<Tensor>> {
    let mut all: Vec<Tensor> = states.to_vec();
    all.push(partial.clone());
    let h = partial.dim(D::Minus1)?;
    let s = rms_unweighted(&Tensor::stack(&all, 1)?, 1e-6)?; // [T, n, H], line 839-840
    let t = s.dim(0)?;
    let nq = queries.dim(0)?;
    let q = queries.unsqueeze(0)?.broadcast_as((t, nq, h))?.contiguous()?;
    let logits = (q.matmul(&s.transpose(1, 2)?)? * (h as f64).powf(-0.5))?; // [T, nq, n]
    let w = candle_nn::ops::softmax_last_dim(&logits)?;
    let out = w.matmul(&s)?; // [T, nq, H]
    (0..nq).map(|i| out.i((.., i, ..))?.contiguous()).collect()
}

/// `Inkling2LoopSoftcap`, lines 751-757.
fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    (x / cap)?.tanh()? * cap
}

/// `Inkling2LoopNormedEmbedding`, lines 1240-1246: the embedding row, then an RMS norm with weight.
pub fn embed(ck: &NativeCheckpoint, cfg: &InklingConfig, ids: &[u32], dev: &Device) -> Result<Tensor> {
    let rows = ck.embed_rows(ids, dev)?;
    let w = ck.f32("embed_norm.weight", dev)?;
    rms_norm(&rows, &w, cfg.rms_norm_eps)
}

/// Max absolute difference and the max absolute value of `reference`, both f32 tensors of one shape,
/// compared on the CPU whatever device each lives on.
pub fn max_abs_err(ours: &Tensor, reference: &Tensor) -> Result<(f32, f32)> {
    let ours = ours.to_device(&Device::Cpu)?;
    let reference = reference.to_device(&Device::Cpu)?;
    let d = (ours.to_dtype(DType::F32)? - reference.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?;
    let r = reference.to_dtype(DType::F32)?.abs()?.flatten_all()?.max(0)?;
    Ok((d.to_scalar()?, r.to_scalar()?))
}

#[cfg(test)]
mod tests;
