//! The CUDA adapters of the device port (`crate::device`) and the plain views
//! the persistent and batched paths read their buffers through. One file
//! because every type here only forwards to the runtime that owns the buffers.

use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::Tensor;

use crate::fused::{EmbedMode, RotKey};

use super::{DeviceStream, FusedInt4Proj, FusedProj, FusedRuntime, FusedSegProj, QuantEmbed};

// ---------------------------------------------------------------------------
// The CUDA adapters: the device port of `crate::device`, implemented here.
//
// Each owns its runtime AND its buffers, which is the whole shape of the port.
// The model holds `Arc<dyn LatticeProj>` and cannot tell a card from a fake.
// Nothing below computes: every method forwards to the inherent method that
// already shipped, so the served arithmetic is unchanged by construction.
// ---------------------------------------------------------------------------

/// One `Tetra` or `Planes` projection on a card.
pub struct CudaLattice {
    pub(super) rt: std::sync::Arc<FusedRuntime>,
    pub(super) proj: FusedProj,
}

impl crate::device::LatticeProj for CudaLattice {
    fn name(&self) -> &str {
        &self.proj.name
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn rotation(&self) -> Option<RotKey> {
        self.proj.rotation()
    }
    fn prepare(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.rt.rotate(&self.proj, x)
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> candle_core::Result<Tensor> {
        self.rt.rotate_rows(&self.proj, xs, rows)
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> candle_core::Result<Tensor> {
        self.rt.forward_rotated(&self.proj, xr, out_dims)
    }
    /// The runtime's two answers, collapsed into the one number the port
    /// asks for. `has_rows_kernel` false and `prefill_rows` four could once
    /// disagree; `1` here says the same thing the `bool` said, and cannot.
    fn rows_per_launch(&self) -> usize {
        match self.rt.has_rows_kernel() {
            true => self.rt.prefill_rows(),
            false => 1,
        }
    }
    fn matvec_rows(&self, xr: &Tensor, n_rows: usize) -> candle_core::Result<Tensor> {
        self.rt.forward_rotated_rows(&self.proj, xr, n_rows)
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// One projection served as affine int4 g128 on a card.
pub struct CudaInt4 {
    pub(super) rt: std::sync::Arc<FusedRuntime>,
    pub(super) proj: FusedInt4Proj,
}

impl crate::device::Int4Proj for CudaInt4 {
    fn name(&self) -> &str {
        &self.proj.name
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> candle_core::Result<Tensor> {
        self.rt.forward_int4(&self.proj, x, out_dims)
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// ---- read-only device views for 4E (`persist4e_cuda/`) ----
//
// Raw device addresses of what the runtime already holds, so the persistent
// segment reads the very buffers the eager path reads: the same words, row
// scales, tails, gains, rotations and tables. Nothing here allocates or
// writes; the views are valid while the runtime and its projections live,
// which the model guarantees by holding both.

pub(super) fn dev_addr<T>(s: &CudaSlice<T>, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> u64 {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    s.device_ptr(stream).0
}

/// One Tetra48 projection.
#[derive(Clone, Copy, Debug)]
pub struct TetraView {
    pub words: u64,
    pub stride_u32: u32,
    pub nblocks: u32,
    pub tail_w: u32,
    pub d_out: u32,
    pub d_in: u32,
    pub rscale: u64,
    /// binary16, `d_out * tail_w`
    pub tail: u64,
    pub gscale: u64,
    pub rotation: Option<RotKey>,
}

/// One rotation, as `rot_apply` takes it.
#[derive(Clone, Copy, Debug)]
pub struct RotView {
    pub signbits: u64,
    pub small: u64,
    pub n: u32,
    pub m: u32,
    pub k: u32,
    pub inv: f32,
}

/// One int4 g128 projection.
#[derive(Clone, Copy, Debug)]
pub struct Int4View {
    /// The words, in [`Int4View::layout`]'s order.
    pub wq: u64,
    pub scales: u64,
    pub biases: u64,
    pub gpr: u32,
    pub d_out: u32,
    pub d_in: u32,
    /// The file's order, or 4E v9's tiled one: 4E v3 to v8 refuse a tiled
    /// record, v9 reads it in place.
    pub layout: crate::fused::Int4Layout,
}

/// The Tetra decoder tables.
#[derive(Clone, Copy, Debug)]
pub struct TableView {
    pub rows: u64,
    pub pref: u64,
    pub br: u64,
    pub suf: u64,
    pub invnorm: u64,
}

impl CudaLattice {
    pub fn runtime(&self) -> &std::sync::Arc<FusedRuntime> {
        &self.rt
    }
    /// `None` unless the projection is a Tetra48 stream.
    pub fn tetra_view(&self) -> Option<TetraView> {
        let st = self.rt.device.cuda_stream();
        let DeviceStream::Tetra48 { words, stride_u32 } = &self.proj.stream else {
            return None;
        };
        Some(TetraView {
            words: dev_addr(words, &st),
            stride_u32: *stride_u32,
            nblocks: self.proj.nblocks,
            tail_w: self.proj.tail_w,
            d_out: self.proj.d_out as u32,
            d_in: self.proj.d_in as u32,
            rscale: dev_addr(&self.proj.rscale, &st),
            tail: dev_addr(&self.proj.tail, &st),
            gscale: dev_addr(&self.proj.gscale, &st),
            rotation: self.proj.rotation,
        })
    }
}

impl CudaInt4 {
    pub fn runtime(&self) -> &std::sync::Arc<FusedRuntime> {
        &self.rt
    }
    pub fn int4_view(&self) -> Int4View {
        let st = self.rt.device.cuda_stream();
        Int4View {
            wq: dev_addr(&self.proj.wq, &st),
            scales: dev_addr(&self.proj.scales, &st),
            biases: dev_addr(&self.proj.biases, &st),
            gpr: self.proj.gpr,
            d_out: self.proj.d_out as u32,
            d_in: self.proj.d_in as u32,
            layout: self.proj.layout,
        }
    }
}

impl FusedRuntime {
    pub fn rot_view(&self, key: &RotKey) -> Option<RotView> {
        let st = self.device.cuda_stream();
        self.rotations.get(key).map(|r| RotView {
            signbits: dev_addr(&r.signbits, &st),
            small: dev_addr(&r.small, &st),
            n: r.n,
            m: r.m,
            k: r.k,
            inv: r.inv,
        })
    }
    pub fn table_view(&self) -> Option<TableView> {
        let st = self.device.cuda_stream();
        self.tetra_tabs.as_ref().map(|t| TableView {
            rows: dev_addr(&t.rows, &st),
            pref: dev_addr(&t.prefixes, &st),
            br: dev_addr(&t.branches, &st),
            suf: dev_addr(&t.suffixes, &st),
            invnorm: dev_addr(&t.invnorm, &st),
        })
    }
    /// The `#define` this runtime's unit compiled its tile with.
    pub fn tile_define(&self) -> String {
        self.tile.define()
    }
    pub fn cuda_device(&self) -> &candle_core::CudaDevice {
        &self.device
    }
}

/// One row-concatenated group on a card, launched once for every part.
pub(super) struct CudaSeg {
    pub(super) rt: std::sync::Arc<FusedRuntime>,
    pub(super) group: FusedSegProj,
}

impl crate::device::SegGroup for CudaSeg {
    fn name(&self) -> &str {
        &self.group.name
    }
    fn d_out(&self) -> usize {
        self.group.d_out
    }
    fn d_in(&self) -> usize {
        self.group.d_in
    }
    fn rotation(&self) -> Option<RotKey> {
        self.group.rotation()
    }
    fn part_name(&self, rank: usize) -> &str {
        self.group.part_name(rank)
    }
    fn prepare(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.rt.rotate_group(&self.group, x)
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> candle_core::Result<Tensor> {
        self.rt.forward_rotated_seg(&self.group, xr, out_dims)
    }
}

/// The quantized embedding table on a card (int8 or int4, by `LLVQ_EMBED`;
/// int4 in the served configs), and the `lm_head` that may share it.
///
/// `q` stays an `Arc` rather than being owned, because a tied model uploads
/// ONE table and points both ends at it. Two `CudaEmbed` values then share one
/// buffer, which is what the load-time tie check asserts.
pub struct CudaEmbed {
    pub(super) rt: std::sync::Arc<FusedRuntime>,
    pub(super) q: std::sync::Arc<QuantEmbed>,
}

impl crate::device::QuantEmbedTable for CudaEmbed {
    fn gather(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.rt.embed(&self.q, ids)
    }
    fn project(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        self.rt.lm_head(&self.q, h)
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The int4 g64 table's device buffers, for 4E's embedding and head phases.
#[derive(Clone, Copy, Debug)]
pub struct EmbedView {
    pub words: u64,
    pub scales: u64,
    pub biases: u64,
    pub vocab: u32,
    pub d: u32,
    pub gpr: u32,
}

impl CudaEmbed {
    pub fn runtime(&self) -> &std::sync::Arc<FusedRuntime> {
        &self.rt
    }
    /// `None` unless the table is the int4 g64 one (`LLVQ_EMBED=q4`, the three
    /// served configs): the q8 table is another format.
    pub fn q4_view(&self) -> Option<EmbedView> {
        if self.rt.emode != EmbedMode::Q4 || self.q.gpr as usize * crate::fused::EMBED_GROUP != self.q.d {
            return None;
        }
        let st = self.rt.device.cuda_stream();
        Some(EmbedView {
            words: dev_addr(&self.q.words, &st),
            scales: dev_addr(&self.q.scales, &st),
            biases: dev_addr(&self.q.biases, &st),
            vocab: self.q.vocab as u32,
            d: self.q.d as u32,
            gpr: self.q.gpr,
        })
    }
}
