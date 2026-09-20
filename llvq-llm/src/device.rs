//! The device seam: what the model asks a backend for, stated once.
//!
//! The model holds `Arc<dyn LatticeProj>` and names no device type. One
//! implementation reads `tv_tetra48_h` on a card. Another will read an MSL
//! kernel on Metal. A third is a CPU fake a test builds. They differ in where
//! the weights live and in nothing the model can see.
//!
//! ## Why four traits and not one
//!
//! A `SegGroup` cannot reach `LatticeProj::matvec`, and an embedding table
//! cannot reach `Int4Proj::matvec`. That separation is free here and is the
//! reason for the shape: a single trait with nine methods would let the
//! loader hand a group to a per-projection call and find out on a card.
//! Do not merge them into one activation-agnostic trait later.
//!
//! ## Why `Send + Sync`
//!
//! Measured, not assumed. `ops/check-cuda.sh` type-checked a probe asserting
//! `Send` and `Sync` on `FusedRuntime`, `FusedProj`, `FusedInt4Proj`,
//! `FusedSegProj` and `QuantEmbed`. All five carry both. The bound is stated
//! here rather than discovered by a caller that wants to share a model.

use crate::fused::RotKey;
use candle_core::{Result, Tensor};

/// Rows one launch of the prefill kernel carries, and the tile it was
/// compiled at.
///
/// Portable by construction. `llvq_cuda::tile::Prefill` holds the same three
/// fields and no device type, but `llvq-cuda` is target-gated on Linux and
/// pulls `cudarc` there, so naming it from an unconditional module would drag
/// the driver into a CPU-only build. Ten lines here cost less than that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prefill {
    /// Rows a single launch takes. `1` means there is no prefill kernel.
    pub rows: usize,
    /// Activation blocks one CTA stages, the `TILE_BLOCKS` the unit compiled at.
    pub tile: usize,
    /// Whether an explicit environment override chose `tile`, rather than the
    /// measured row for the card. Provenance, printed and never acted on.
    pub from_env: bool,
}

/// One projection whose weights live on a device, in the encoding that
/// device's kernels read.
pub trait LatticeProj: Send + Sync {
    /// The artifact name, for error messages.
    fn name(&self) -> &str;
    fn d_out(&self) -> usize;
    fn d_in(&self) -> usize;

    /// The rotation these weights were quantized under, `None` in the natural
    /// basis. `rotplan::check_key` compares it against the activation's.
    fn rotation(&self) -> Option<RotKey>;

    /// `x`, `[1, d_in]`, into the basis these weights were quantized in.
    fn prepare(&self, x: &Tensor) -> Result<Tensor>;

    /// [`Self::prepare`] for `rows` activations, `[rows, d_in]` contiguous.
    ///
    /// Never called with `rows == 1`. The caller takes [`Self::prepare`]
    /// there, which keeps every decode step on the code `bin/oracle`
    /// certifies rather than on a grid of one block.
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor>;

    /// `y = W xr`, with the caller's rank preserved through `out_dims`.
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor>;

    /// Rows one launch of [`Self::matvec_rows`] carries.
    ///
    /// `1` means there is no such kernel, and `group_forward` fans out row by
    /// row through [`Self::matvec`]: the same arithmetic, more launches. The
    /// default is `1`, so a first adapter for a new backend writes neither
    /// this nor [`Self::matvec_rows`] and is correct.
    ///
    /// One number on purpose. This replaced a `bool` and a `usize` that could
    /// disagree, and a chunk of eight handed to a kernel compiled at four is
    /// the silent corruption the pair existed to prevent.
    fn rows_per_launch(&self) -> usize {
        1
    }

    /// `y = W X` for `n_rows` prepared rows, one launch.
    ///
    /// Reached only when [`Self::rows_per_launch`] is above one. The default
    /// refuses by name rather than computing the first row and using it for
    /// four.
    fn matvec_rows(&self, _xr: &Tensor, n_rows: usize) -> Result<Tensor> {
        candle_core::bail!(
            "{}: {n_rows} rows asked of a projection that takes one a launch",
            self.name()
        )
    }
}

/// One projection served as affine int4 g128, `v_proj` in the mixed file.
///
/// It behaves like a dense projection in every respect that matters to the
/// caller and like a lattice one in none: the weights are STORED, in the
/// natural basis, so there is no rotation to carry and no key to check.
pub trait Int4Proj: Send + Sync {
    fn name(&self) -> &str;
    fn d_out(&self) -> usize;
    fn d_in(&self) -> usize;

    /// `y = W x`, reading the activation the CALLER handed in.
    ///
    /// `x` and never the prepared tensor, deliberately. These weights are in
    /// the natural basis, so the two are equal today; taking `x` keeps that
    /// true if `prepare` ever stops being the identity here. Do not merge
    /// this with [`LatticeProj::matvec`] into one activation-agnostic method.
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor>;
}

/// The row-concatenation of the projections that share one activation: one
/// launch for the whole group, then one view per part.
pub trait SegGroup: Send + Sync {
    /// The group key, what an error message names.
    fn name(&self) -> &str;
    /// The **total** width, the sum of the parts'.
    fn d_out(&self) -> usize;
    fn d_in(&self) -> usize;
    fn rotation(&self) -> Option<RotKey>;
    /// The artifact name of the part at `rank`.
    fn part_name(&self, rank: usize) -> &str;

    /// One `rot_apply` for the whole group.
    ///
    /// The parts share one `d_in` and one rotation key by construction,
    /// checked at load. A group is one launch already, so it never batches:
    /// `rows_per_launch` has no analogue here and a chunk of more than one
    /// row is refused by the adapter, by name.
    fn prepare(&self, x: &Tensor) -> Result<Tensor>;

    /// `y = W' xr` over the group's total width.
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor>;
}

/// The embedding table, and the `lm_head` that may be the same buffer.
///
/// One trait for both ends because a tied model holds two clones of one
/// `Arc`. `Arc::ptr_eq` on the two is then the tie itself.
pub trait QuantEmbedTable: Send + Sync {
    /// Token ids `(.., l)` to hidden states `(.., l, d)`.
    fn gather(&self, ids: &Tensor) -> Result<Tensor>;
    /// `h . W^T`, logits from hidden states.
    fn project(&self, h: &Tensor) -> Result<Tensor>;
}
