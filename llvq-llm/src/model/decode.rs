//! Free helpers around a decode: the greedy token, a log-probability, the two result records and
//! the causal mask. Apart from [`Qwen3`] because the binaries and the unit tests call them alone.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

// Read by intra-doc links only; rustc never compiles it, so it cannot go unused.
#[cfg(doc)]
use super::Qwen3;

/// The greedy token of `[1, 1, vocab]` logits, read back to the host.
pub fn argmax_last(logits: &Tensor) -> Result<u32> {
    logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()
}

/// The log-probability of `target` under `[1, 1, vocab]` f32 logits.
pub fn logp_of(logits: &Tensor, target: u32) -> Result<f64> {
    let logp = candle_nn::ops::log_softmax(&logits.i((0, 0))?, D::Minus1)?;
    Ok(logp.i(target as usize)?.to_scalar::<f32>()? as f64)
}

/// What [`Qwen3::decode_nll_paired`] returns for one window.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedDecode {
    /// Summed negative log-likelihood of each arm, in nats.
    pub nll: [f64; 2],
    /// Predictions scored, the window length minus one.
    pub count: usize,
    /// Steps where both arms put the same token on top.
    pub top1_agree: usize,
    /// The largest absolute difference between the two arms' logits, over
    /// every vocabulary entry of every step.
    pub max_dlogit: f32,
    /// Steps below the split whose two logit vectors differ in any bit.
    pub below_differ: usize,
    /// [`Self::nll`] over the steps at the split or later.
    pub tail_nll: [f64; 2],
    /// Steps at the split or later.
    pub tail_count: usize,
    /// [`Self::top1_agree`] over the steps at the split or later.
    pub tail_top1_agree: usize,
}

/// One greedy generation from [`Qwen3::generate_timed`], with its two clocks.
pub struct TimedGeneration {
    pub tokens: Vec<u32>,
    /// From the first prompt token sent to the first new token read back.
    pub prompt_s: f64,
    /// The `tokens.len() - 1` decode steps after the first new token.
    pub decode_s: f64,
}

/// The causal mask at an imposed key width `w >= offset + l` — free-standing
/// so the unit tests can exercise the exact object the model uses, not a
/// re-derivation of it. Columns past `offset + i` are −inf: the causal rule
/// is also the validity rule of a preallocated window's unwritten tail.
pub fn build_causal_mask(
    b: usize,
    l: usize,
    offset: usize,
    w: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    debug_assert!(w >= offset + l);
    // Built on the host in the dtype it is used in. +0 and -inf are exact in
    // every float dtype, so this is bit for bit the f32 mask cast on the
    // device, without its f32 copy: 134 MB at the second chunk of an
    // 8,192-token prompt, freed into the pool right before the attention.
    fn rows<T: Copy>(l: usize, w: usize, offset: usize, zero: T, ninf: T) -> Vec<T> {
        (0..l).flat_map(|i| (0..w).map(move |j| if j <= offset + i { zero } else { ninf })).collect()
    }
    let shape = (b, 1, l, w);
    match dtype {
        DType::F16 => Tensor::from_slice(&rows(l, w, offset, half::f16::ZERO, half::f16::NEG_INFINITY), shape, device),
        DType::BF16 => {
            Tensor::from_slice(&rows(l, w, offset, half::bf16::ZERO, half::bf16::NEG_INFINITY), shape, device)
        }
        _ => Tensor::from_slice(&rows(l, w, offset, 0f32, f32::NEG_INFINITY), shape, device)?.to_dtype(dtype),
    }
}
