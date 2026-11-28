//! A Qwen3 forward pass we control, so the input of every linear layer is
//! observable.
//!
//! GPTQ needs `H = AᵀA/N` where `A` is the *input* to a linear layer. No
//! inference library exposes that: `candle`'s Qwen3 applies its `Linear`s
//! inline with no hook, and wrapping them from outside is not possible
//! because the intermediate tensors never leave the block. So the block
//! forward is written out here, with capture points on the four distinct
//! activations a Qwen3 block produces.
//!
//! **Four, not seven.** `q_proj`, `k_proj` and `v_proj` all consume the same
//! tensor (`input_layernorm(x)`), and `gate_proj`/`up_proj` both consume
//! `post_attention_layernorm(x)`. Only `o_proj` and `down_proj` have inputs
//! of their own. Accumulating one Hessian per *activation* rather than per
//! *matrix* is exact and saves 3/7 of the work.
//!
//! The implementation is validated against `candle_transformers`' own Qwen3
//! (see `bin/oracle.rs`): same weights, same window, logits must agree. That
//! reference is the only thing standing between a subtly wrong RoPE and
//! three weeks of chasing a perplexity that is off by 15 %.
//!
//! Derived from the architecture as implemented in `candle-transformers`
//! (MIT OR Apache-2.0), restructured for observability.
//!
//! | file | holds |
//! |---|---|
//! | `capture` | the activation names, the [`Capture`] trait and its two implementations, the fenced clock |
//! | `rotary` | the rotary position tables |
//! | `cache` | one block's [`KvCache`] and the helpers its growth needs |
//! | `attention` | the unexpanded attention forms, the masked softmax, the `Kernel` dispatch |
//! | `proj` | [`Proj`], [`ProjSource`], the [`Rotated`] activation, the row caps, the grouped forward |
//! | `block` | the embedding, the head, one transformer [`Block`], the batch entry points |
//! | `qwen3` | [`Qwen3`], the whole model |
//! | `decode` | the greedy token, a log-probability, the decode records, the causal mask |
//!
//! Each child imports what it uses by name. This file re-exports only what
//! the rest of the workspace names, plus the types its public signatures carry.

mod capture;
mod rotary;
mod cache;
mod attention;
mod proj;
mod block;
mod qwen3;
mod decode;

pub use attention::{set_softmax_whole_for_measurement, softmax_form};
pub use block::{batch_group, Block, Embed, Head};
pub use cache::{KvCache, KvRoundTrip};
pub use capture::{Act, Capture, NoCapture, PhaseReport};
pub use decode::{argmax_last, logp_of, PairedDecode, TimedGeneration};
pub use proj::{group_forward, row_cap, Proj, ProjSource, Rotated, MAX_PREFILL_ROWS, MAX_ROWS};
pub use qwen3::Qwen3;
pub use rotary::Rotary;

/// The tiny test model, for the host tests of other modules.
#[cfg(test)]
pub(crate) mod tests_support {
    /// Two layers, 8 query heads over 2 KV heads at head dim 128, binary16 on
    /// the CPU: a shape the attention kernel's CPU replay takes.
    pub fn tiny_model_kernel() -> super::Qwen3 {
        super::tests::tiny_model_hd(candle_core::DType::F16, 8, 2, 128)
    }
}

#[cfg(test)]
mod tests;
