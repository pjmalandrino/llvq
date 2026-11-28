//! Incoherence processing: an orthogonal rotation applied before quantizing.
//!
//! The rotation itself lives in [`llvq_search::rotation`], where the artifact
//! decoder can reach it without depending on this crate; the construction and
//! the reasons for it are documented there. It is re-exported here at its old
//! path so every caller compiles unchanged.

pub use llvq_search::rotation::Rotation;
