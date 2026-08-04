//! The LLVQ 2-bit artifact format — packed Leech lattice indices, and the
//! decoder that turns them back into weights.
//!
//! ## Why this is its own crate, with no dependencies
//!
//! Reading a quantized model must not require a tensor runtime. Someone who
//! wants to check what a `.llvq` file actually contains — or write a reader in
//! another language, or audit the decoder before trusting a model — should be
//! able to read this crate end to end without auditing candle, tokenizers and
//! four hundred transitive dependencies first. That is the sovereignty
//! argument applied to the format itself.
//!
//! Consequently there is no `anyhow` here either: errors are a small explicit
//! enum. The crate that bridges to candle ([`llvq_llm`]) converts them.
//!
//! ## The format, in one paragraph
//!
//! A file is `"LVQ1"`, a matrix count, then that many matrices. Each matrix
//! carries its name, dimensions, shell cap, gain centroids, one scale per
//! output row, the trailing columns that were left unquantized, and finally a
//! dense bit stream of `(index, gain)` pairs — 47 or 48 bits for the index,
//! `log2(centroids)` for the gain, packed back to back with no padding except
//! at the very end.
//!
//! ## What it does not carry
//!
//! Embeddings, RMSNorm weights and the model config. A `.llvq` is the
//! compressed 90 % of a model, not a self-contained one. On Qwen3-4B the tied
//! embedding is 389 M weights at f16 — 9.7 % of the model, and the reason the
//! end-to-end ratio is ~4.6× rather than the ~7.6× the linear layers alone
//! would suggest.

mod error;
mod format;
pub mod runtime;
mod sealed;

pub use error::Error;
pub use format::{
    decode_matrix, read_all, read_header, read_matrix, read_matrix_raw, split_name, write_matrix,
    write_matrix_raw, ArtifactWriter, Header, QuantizedMatrix, RawMatrix, MAGIC, MAGIC_V1,
    MAGIC_V2,
};
pub use sealed::{
    f16_to_f32, read_blob, read_raw, write_blob, write_raw, Blob, QuantData, RawData, RawTensor,
};

/// Result alias for this crate.
pub type Result<T> = core::result::Result<T, Error>;
