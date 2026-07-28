//! # llvq-core
//!
//! Exact arithmetic core for **Leech Lattice Vector Quantization** (LLVQ,
//! [arXiv:2603.11021](https://arxiv.org/abs/2603.11021), van der Ouderaa,
//! van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026).
//!
//! This crate implements Phase 1 of the implementation plan:
//!
//! * the extended binary Golay code `[24, 12, 8]` ([`golay::Golay`]),
//! * the integer-coordinate construction of the Leech lattice Λ₂₄ from the
//!   Golay code (paper Eq. 4–5, [`leech::Leech`]),
//! * the G1 invariant test-suite (`tests/g1_invariants.rs`), which validates
//!   the implementation against publicly known constants of Λ₂₄ — the
//!   kissing number 196 560, the theta-series coefficients, the Golay weight
//!   distribution — none of which depend on the paper itself.
//!
//! The crate deliberately has **zero dependencies** and forbids `unsafe`:
//! the mathematical core must remain auditable, portable and reproducible.

#![forbid(unsafe_code)]

pub mod golay;
pub mod leech;
pub mod rng;
pub mod shells;

pub use golay::Golay;
pub use leech::{Leech, Point, DIM};
pub use rng::SplitMix64;
