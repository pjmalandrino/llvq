//! # llvq-llm — Phase 5: the model side
//!
//! Loads a Qwen3 checkpoint, runs a forward pass whose linear-layer inputs
//! are observable (that is the whole point — see [`model`]), and scores
//! perplexity. The quantization itself lives in `llvq-quant`.
//!
//! `unsafe` is allowed here, and only here in the workspace: mapping a
//! multi-gigabyte safetensors file is `unsafe` in `candle` by construction
//! (the file could change underneath the mapping). The mathematical core
//! keeps `forbid(unsafe_code)`.

pub mod artifact;
pub mod artifact2;
pub mod calib;
pub mod corpus;
pub mod embedquant;
pub mod eval;
pub mod fused;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod fused_cuda;
pub mod loader;
pub mod model;
pub mod sealed;
