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
//!
//! ## The files
//!
//! | file | holds |
//! |---|---|
//! | `layout` | [`FusedLayout`], [`EmbedMode`], the q8 and q4 embedding tables and their footprint |
//! | `host` | one matrix's stream ([`HostStream`], [`FusedMatrix`], [`FusedInt4`]), the side bytes, [`RotationTables`] |
//! | `transcode` | the packers, the Golay70 device tables, [`Transcoder`] |
//! | `kernels` | the kernel text a layout needs, its entry point names, the int4 fixtures |
//! | `group` | the row concatenation ([`FusedGroup`], [`SegPart`]) and [`FusedModel`] |
//! | `load` | [`load`], [`load_with`] and [`FusedSealed`] |
//!
//! Each child imports what it uses by name. This file re-exports only what
//! the rest of the workspace names, plus the types its public signatures carry.

mod group;
mod host;
mod kernels;
mod layout;
mod load;
mod transcode;

pub use group::{
    check_seg_spans, planes_payload_words, segment_matrices, splice_planes14, FusedGroup, FusedModel, SegPart,
};
pub use host::{
    matrix_side_bytes, tail_f16_bits, FusedInt4, FusedMatrix, HostStream, RotKey, RotationTables, ROW_SCALE_BYTES,
    TAIL_BYTES,
};
pub use kernels::{
    check_fuse, check_kinds, emb_kernel_names, int4_synthetic, int4_tileable, int4_wide_unit_names,
    lattice_kind, load_emb_q4_sources, load_int4_sources, load_int4_wide_sources, load_planes_sources,
    matvec_kernel_name, parse_int4_wide_fixture, planes_source_names, rows_kernel_name, seg_kernel_name, serves_kind,
    tetra48_tables, FuseMode, Int4Layout, Int4WideFixture, Tetra48Tables, INT4_HALF_KERNEL_NAME, INT4_KERNEL_NAME,
    INT4_TILE_HALF_KERNEL_NAME, INT4_TILE_KERNEL_NAME, INT4_WIDE_CARD_CASES, INT4_WIDE_FIXTURE, INT4_WIDE_KERNEL_NAME,
    Q4_TILE_CU_EMBED, Q4_WIDE_CU_EMBED,
};
pub use layout::{
    carried_embed_tables, embed_q4, embed_q8, q4_device_bytes, q8_device_bytes, take_embed_tables, take_embed_tables_at,
    EmbedMode, EmbedReport, EmbedTables, FusedLayout, EMBED_GROUP, EMBED_NAME, HEAD_NAME,
};
pub use load::{load, load_with, FusedSealed, PREALLOC_CAP};
pub use transcode::{golay70_gpu_class_table, golay70_gpu_codewords, Transcoder};

#[cfg(test)]
mod tests;
