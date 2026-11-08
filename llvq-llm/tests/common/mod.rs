//! What several test files of this crate share.
//!
//! `tiny` was copied into seven files until 2026-11-08 (audit batch 10).
//! `tests/session.rs` keeps its own, a vocabulary of 64.

use candle_nn::Activation;
use candle_transformers::models::qwen3::Config;

/// A two-layer Qwen3 small enough to build from random weights in a test.
pub fn tiny() -> Config {
    Config {
        vocab_size: 128,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}
