//! Loading the sealed artifact **without decoding it** — the shape a fused
//! kernel needs.
//!
//! ## What this is not
//!
//! [`crate::sealed::load`] rebuilds a model by calling `decode_matrix` on
//! every projection, which materializes 3.63 G weights as f16 tensors. That is
//! correct, it is what every published perplexity refers to, and it throws
//! away the entire point of the format: once decoded, the file occupies as
//! much memory as an f16 checkpoint and runs at exactly its speed. The
//! 2026-08-05 miniature run measured that plainly — 42.7 tok/s against 42.8.
//!
//! This module keeps the weights encoded. It transcodes each lattice matrix
//! to the runtime layout the fused matvec reads, the one `LLVQ_FUSED_LAYOUT`
//! names (see [`FusedLayout`]): `Tetra48` for the three served files, which
//! hold Tetra records, and `Planes14` (the default), `Planes12x`, `Slot32` or
//! `Golay70` for ball files. An int4 record is carried as it is stored. It
//! hands the host tables over. Nothing here touches a GPU: this is the
//! portable half, and it is testable without a card.
//!
//! ## The rotation tables, and why they are here
//!
//! The artifact is quantized in a rotated basis, so a kernel reading the
//! stored weights computes `W' x`, not `W x`. The identity it owes is
//!
//! ```text
//! y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//! ```
//!
//! pinned by `llvq-artifact/tests/fused_path_matches_dense.rs`. A kernel
//! cannot rebuild `Q` from its seed — that would mean porting `SplitMix64` and
//! Gram–Schmidt to the device — so the host builds it once per distinct
//! `(width, seed)` and ships three small tables.
//!
//! Distinct pairs are far fewer than matrices: q/k/v share an activation and
//! therefore a rotation, and so do gate/up. An int4 record is stored in the
//! natural basis and has none. The served Qwen3-4B carries 84 of its 252
//! projections as int4, so its 168 lattice projections share **96** rotations
//! (36 for q/k, 36 for gate/up, one for each of the 24 Tetra `down_proj`);
//! an all-lattice 4B has 144. The tables together weigh under a megabyte.

use std::collections::HashMap;
use std::io::Read;

use llvq_artifact::CodeKind;
use llvq_artifact::runtime::{
    transcode, transcode_golay70, transcode_planes12x, transcode_planes14, ClassTable,
    Golay70Blocks, Golay70Table, Layout, Planes12xBlocks, PlanesBlocks, RuntimeBlocks,
    PLANES14_BYTES,
};
use llvq_quant::rotation::Rotation;
use llvq_search::fastdec::FastDecoder;
use llvq_search::Searcher;

mod group;
mod host;
mod kernels;
mod layout;
mod load;
mod transcode;

pub use group::*;
pub use host::*;
pub use kernels::*;
pub use layout::*;
pub use load::*;
pub use transcode::*;

#[cfg(test)]
mod tests;
