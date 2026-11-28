//! The port of `crate::device`, implemented for Metal: a lattice projection and an int4 projection as the
//! model sees them. One file because it is the whole surface `model/` touches.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

use super::{MetalInt4Proj, MetalRotation, MetalRuntime, MetalTetraProj};

// ---------------------------------------------------------------------------
// The port, implemented. From here `model/` cannot tell a card from a Mac.
// ---------------------------------------------------------------------------

/// One Tetra projection, its runtime and its rotation, as the model sees it.
pub(super) struct MetalLattice {
    pub(super) rt: Arc<MetalRuntime>,
    pub(super) proj: MetalTetraProj,
    pub(super) rot: Option<Arc<MetalRotation>>,
    pub(super) key: Option<crate::fused::RotKey>,
}

impl crate::device::LatticeProj for MetalLattice {
    fn name(&self) -> &str {
        &self.proj.name
    }
    /// The concrete projection, so the served door can read which kernels
    /// the runtime under a loaded model launches ([`tuning_of`]).
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn rotation(&self) -> Option<crate::fused::RotKey> {
        self.key
    }
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        match &self.rot {
            Some(r) => self.rt.rotate(r, &self.proj.name, x),
            // An artifact without a rotation. The CUDA adapter refuses this
            // path by name rather than guess; so does this one.
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered",
                self.proj.name
            ),
        }
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor> {
        match &self.rot {
            Some(r) => self.rt.rotate_rows(r, &self.proj.name, xs, rows),
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered",
                self.proj.name
            ),
        }
    }
    fn matvec(&self, xr: &Tensor, _out_dims: &[usize]) -> Result<Tensor> {
        // Narrowed HERE, not in the kernel.
        //
        // The CUDA twin ends in `f2h` and stores f16; these kernels store f32,
        // which was a deliberate staging decision. The model runs in f16, so
        // the adapter narrows at its boundary instead. candle's conversion is
        // the `half` crate's, which is IEEE round-to-nearest-even, the same
        // rule `f2h` implements. So the two paths agree to the bit, and what
        // differs is only WHERE the rounding happens, plus one f32 buffer's
        // worth of traffic that a later lot can remove.
        self.rt.matvec(&self.proj, xr)?.to_dtype(DType::F16)
    }
    /// 1 on the served tuning: no rows kernel, so the caller fans a chunk out
    /// row by row. [`MetalTuning::prefill_rows`] above 1 hands it chunks for
    /// `tv_tetra48_metal_lutdr`, the same bits a row.
    fn rows_per_launch(&self) -> usize {
        self.rt.prefill_rows()
    }
    fn matvec_rows(&self, xr: &Tensor, n_rows: usize) -> Result<Tensor> {
        self.rt.matvec_rows(&self.proj, xr, n_rows)
    }
}

/// One int4 projection, as the model sees it.
pub(super) struct MetalInt4 {
    pub(super) rt: Arc<MetalRuntime>,
    pub(super) proj: MetalInt4Proj,
}

impl crate::device::Int4Proj for MetalInt4 {
    fn name(&self) -> &str {
        &self.proj.name
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn matvec(&self, x: &Tensor, _out_dims: &[usize]) -> Result<Tensor> {
        // See MetalLattice::matvec: narrowed at the boundary, not in the kernel.
        self.rt.matvec_int4(&self.proj, x)?.to_dtype(DType::F16)
    }
    /// 1 unless [`MetalTuning::q4_rows`]: then the Tetra prefill chunk.
    fn rows_per_launch(&self) -> usize {
        match self.rt.q4_rows {
            true => self.rt.prefill_rows(),
            false => 1,
        }
    }
    fn matvec_rows(&self, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        self.rt.matvec_int4_rows(&self.proj, x, n_rows)
    }
}
