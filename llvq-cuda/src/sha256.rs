//! SHA-256 of a byte string, portable.
//!
//! `gpu.rs` hashes the NVRTC text with it, the pinned-unit tests of `pers.rs`
//! compare against it, and the rotation fixture is written on the Mac and
//! checked on the card with it. Since 2026-11-08 it is the workspace's one
//! implementation, `llvq_artifact::sha256`, re-exported under its old path.

pub use llvq_artifact::sha256::sha256_hex;
