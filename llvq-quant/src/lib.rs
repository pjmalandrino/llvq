//! # llvq-quant — Phase 5: Spherical GPTQ
//!
//! Layer-wise post-training quantization with Hessian error feedback, on top
//! of the Leech quantizers of [`llvq_search`]. This is the paper's
//! Algorithm 1 (Appendix B) and Algorithm 3 (Appendix I.1); the transcription
//! and the reasoning behind every reading of them live in
//! `docs/llvq-paper-notes.md`.
//!
//! ## What this crate is, and is not
//!
//! It quantizes a weight matrix given a Hessian. It does **not** run a model:
//! obtaining `H = AᵀA/N` requires a forward pass over a calibration corpus
//! (the paper uses 6 100 sequences of DCLM-edu), which is a separate concern
//! and a separate decision — see the open questions below. Everything here is
//! therefore testable, and tested, without a model file.
//!
//! ## Target configuration
//!
//! Appendix I of the paper recommends, and Table 6 supports, **shape–gain
//! with 0 gain bits under Spherical GPTQ**: all 2 bits/weight go to the
//! direction ([`quantizer::LeechDirection`]), the block norm is preserved
//! exactly through the GPTQ loop ([`gptq::GptqConfig::retract`]), and
//! magnitudes are recovered at the end by a closed-form solve in the Hessian
//! metric ([`gptq::GptqConfig::group_scales`]).
//!
//! ## Open questions, deliberately not answered here
//!
//! * **Tail blocks.** A block is 24 input channels, but real layers are not
//!   multiples of 24 (Qwen3-4B: 2560 = 24·106 + 16). The paper says only
//!   "last may be smaller" and never says what quantizes it. Rather than
//!   guess, [`gptq::quantize_layer`] asserts on a mismatched tail so the
//!   decision surfaces at the call site instead of silently degrading a
//!   layer.
//! * **Where `H` comes from.** A Rust forward pass, or a dumped tensor from
//!   an existing runtime. This crate takes `H` as an argument either way.

#![forbid(unsafe_code)]

pub mod gptq;
pub mod linalg;
pub mod quantizer;
pub mod rotation;
