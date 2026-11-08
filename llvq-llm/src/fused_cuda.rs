//! The fused projections, running inside candle.
//!
//! This is where the kernel stops being a bench and starts being inference.
//! It holds the encoded streams on the device — `Planes14` by default,
//! `Planes12x` or `Slot32` under `LLVQ_FUSED_LAYOUT` — and replaces one
//! linear layer with two launches:
//!
//! ```text
//! x (f16) ──rot_apply──▶ x' (f32, rotated basis) ──tv_*_h──▶ y (f16)
//! ```
//!
//! ## Three things it does not do, on purpose
//!
//! * **It does not un-rotate.** The stored weights are `W' = W Qᵀ`, so the
//!   activation carries the `Q`. The identity is pinned on the CPU by
//!   `llvq-artifact/tests/fused_path_matches_dense.rs`; getting it backwards
//!   produces finite, plausible, wrong numbers and no error at all.
//! * **It does not handle more than one token per call.** `tv_slot` is a
//!   matrix–*vector* product: one warp per output row, one activation staged
//!   in shared memory. A prompt of `l` tokens loops `l` times, which is
//!   correct and slower than a GEMM would be — acceptable for generation
//!   (`l = 1` after the prompt), useless for scoring a 2048-token window.
//!   Perplexity therefore keeps the dense path, and `bin/ppl` is untouched.
//! * **It does not keep a dense copy as a fallback.** That would put the 8 GB
//!   back and forfeit the only gain that is not in dispute — 8.04 GB of f16
//!   weights against 3.28 with the projections encoded.
//!
//! ## Why the stream comes from candle
//!
//! `Cuda::on_stream` compiles our module onto candle's own stream. Both would
//! land on the same primary context either way, so the pointers were never in
//! question; what sharing buys is **ordering**. On a separate stream, a
//! `tv_slot` reading an activation candle has not finished writing is a race
//! that reproduces once in a hundred runs, on the card, in a billed job.

use std::collections::HashMap;

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Device, Layout, Shape, Tensor};
use half::f16;

use crate::fused::{
    load_planes_sources, matvec_kernel_name, seg_kernel_name, EmbedMode, FuseMode, FusedGroup,
    FusedLayout, FusedMatrix, FusedModel, HostStream, RotKey, RotationTables, EMBED_GROUP,
};

/// Threads per block for `tv_slot`: 256 = eight rows per block, the shape the
/// bench has always measured.
const THREADS: u32 = 256;
/// Entries in the device class table. The class field is nine bits, so 512
/// are addressable while 384 exist; the tail stays the origin so a truncated
/// or corrupt index cannot address out of bounds.
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;

/// The q8 embedding kernels — appended only under `LLVQ_EMBED=q8`, so the
/// translation unit of both f16-embedding arms stays byte-identical to what
/// every published number compiled.
const EMB_Q8_CU_EMBED: &str = include_str!("../kernels/emb_q8.cu");

/// The q8 embedding kernel source, honouring `LLVQ_KERNEL_DIR` with the same
/// contract as [`crate::fused::load_planes_sources`].
fn load_emb_sources() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((EMB_Q8_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let p = std::path::Path::new(&dir).join("emb_q8.cu");
            let s = std::fs::read_to_string(&p)
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: emb_q8.cu: {e}"))?;
            Ok((s, Some(dir)))
        }
    }
}

/// The rotation tables, on the device.
struct RotBuffers {
    signbits: CudaSlice<u32>,
    small: CudaSlice<f32>,
    n: u32,
    m: u32,
    k: u32,
    inv: f32,
    /// One block, so the only knob is the thread count.
    threads: u32,
    /// `rot_apply` wherever it runs on this card; `rot_apply_dev`
    /// (`rotate_wide.cu`) only where `llvq_cuda::shared::rot_plan` refuses
    /// the width. Decided once, at upload, by `llvq_cuda::wide::rot_kernel`.
    kernel: llvq_cuda::wide::RotKernel,
}

/// The functions of the wide unit (`llvq_rot.cuh` + `rotate_wide.cu`,
/// `tv_q4_h.cu` + `tv_q4_wide.cu`), compiled as a second module on the
/// runtime's stream only when some width needs them. Each `Some` is the
/// authorisation, the `f_int4` pattern: no launch can reach a function the
/// unit never carried.
struct WideFns {
    /// `rot_apply_dev` and `rot_apply_rows_dev`: present when some rotation
    /// is `RotKernel::Device`.
    rot: Option<(CudaFunction, CudaFunction)>,
    /// `tv_q4_hs`, opted in at the widest F16-staged record: present when
    /// some int4 record is `Int4Staging::F16`.
    int4: Option<CudaFunction>,
    /// The module the functions were loaded from, kept alive with them.
    _module: llvq_cuda::gpu::Cuda,
}

/// A matrix's payload on the device, mirroring [`HostStream`] variant for
/// variant. The `Planes14` arm carries **no bases slice at all** — reading or
/// launching with one on the planes path is a compile error, not a bug class —
/// and, symmetrically, only the `Planes12x` arm can name an exception table,
/// so no other layout's launch can be handed one.
enum DeviceStream {
    Slot32 {
        words: CudaSlice<u32>,
        bases: CudaSlice<u32>,
    },
    Planes14 {
        words: CudaSlice<u32>,
    },
    Planes12x {
        words: CudaSlice<u32>,
        exc_idx: CudaSlice<u32>,
        exc_words: CudaSlice<u32>,
        row_exc: CudaSlice<u32>,
    },
    Golay70 {
        words: CudaSlice<u32>,
        exc_idx: CudaSlice<u32>,
        exc_words: CudaSlice<u32>,
        row_exc: CudaSlice<u32>,
    },
    Tetra48 {
        /// The 48-bit words, **row-strided**: the only stream here that is not
        /// flat over the matrix's blocks. `llvq_artifact::tetra48` explains
        /// why — a six-byte record is u32-aligned only every second block.
        words: CudaSlice<u32>,
        /// Words per row. Carried beside the buffer rather than recomputed at
        /// the launch site: the transcoder rounds it to eight bytes so the
        /// last block's read window fits, and a launch that re-derived it from
        /// `nblocks` with a different rounding would read at a shifted phase.
        stride_u32: u32,
    },
}

/// The five constant arrays `tv_tetra48_h` reads, uploaded once and shared by
/// every matrix — the `g70_tabs` pattern, and for the same reason: they are
/// properties of the codebook, not of a projection.
struct TetraTabs {
    rows: CudaSlice<u32>,
    prefixes: CudaSlice<u32>,
    branches: CudaSlice<u16>,
    suffixes: CudaSlice<u32>,
    invnorm: CudaSlice<f32>,
}

/// One projection's weights, on the device.
pub struct FusedProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    nblocks: u32,
    tail_w: u32,
    stream: DeviceStream,
    gscale: CudaSlice<f32>,
    rscale: CudaSlice<f32>,
    /// The `KeepExact` tail as binary16 bits — the precision the dense arm
    /// holds these same columns at. See [`crate::fused::tail_f16_bits`], which
    /// owns the conversion, the argument and the accounting; `load` below
    /// refuses any dtype under which that argument would not hold.
    tail: CudaSlice<u16>,
    rotation: Option<RotKey>,
}

/// One `v_proj` served as affine int4 g128, on the device.
///
/// It shares nothing with [`FusedProj`] and that is the point: an int4 record
/// carries stored weights in the **natural basis** — never GPTQ, never the
/// rotation, never a lattice index — so it has no rotation key, no gain scale,
/// no row scale and no tail. `calib.rs` writes it that way and `tv_q4_h.cu`
/// reads exactly those three arrays.
/// What [`FusedRuntime::new`] hands back: the runtime, and the three lists of
/// projections it uploaded — lattice, segmented, int4. Three because the model
/// indexes all three by the same `(layer, name)` pair and none of them can be
/// derived from another.
type Loaded = (FusedRuntime, Vec<FusedProj>, Vec<FusedSegProj>, Vec<FusedInt4Proj>);

pub struct FusedInt4Proj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// `ceil(d_in / 128)`, the scale/bias pairs a row carries. Passed to the
    /// kernel rather than recomputed there, "so the host and the device cannot
    /// disagree about the rounding up" — the kernel's own words.
    gpr: u32,
    /// The nibble stream as u32 words, little-endian. The disk stream is
    /// bytes, **low nibble first**, at the global flat index `row · d_in + c`;
    /// a lattice index in the same file is packed MSB-first. Two orders in one
    /// file, and reading one with the other's convention gives plausible,
    /// wrong weights.
    wq: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    /// The whole activation staged in shared: `d_in · 4` under F32 staging,
    /// `d_in · 2` under F16 (`tv_q4_hs`). Held here so the launch and the
    /// load-time check against the device limit are one number.
    shared: u32,
    /// F32 (`tv_q4_h` / `tv_q4_hh`, today's kernels) wherever the f32 staging
    /// fits the card; F16 (`tv_q4_hs`) only past it.
    staging: llvq_cuda::wide::Int4Staging,
    /// The order of `wq` on the card: the file's, or 4E v9's tiled one
    /// (`llvq_cuda::pers::q4_tile_words`), written at upload. Decides the
    /// entry point every launch takes, so a tiled record cannot meet a
    /// served-order reader.
    pub layout: crate::fused::Int4Layout,
    pub bytes: u64,
}

impl FusedInt4Proj {
    /// Always `None`: stored in the natural basis, so there is no activation
    /// to carry and nothing to check a key against. `model::Proj` reads this
    /// exactly as it reads a dense projection's.
    pub fn rotation(&self) -> Option<RotKey> {
        None
    }
}

impl FusedProj {
    /// The rotation this matrix was quantized under. Read by `model::Proj` to
    /// tag the activation it prepares, and by nothing else.
    pub fn rotation(&self) -> Option<RotKey> {
        self.rotation
    }
}

/// One fused group's weights, on the device — the row-concatenation of the
/// projections that share an activation.
///
/// A bare `words` field rather than a one-variant [`DeviceStream`]: the
/// guarantee `DeviceStream` buys (a `Planes14` launch cannot be handed a bases
/// array) is bought here by the *type of the struct itself* — a group is
/// `Planes14` by construction and [`upload_group`] refuses anything else, so
/// there is no second shape for this variant to distinguish it from.
pub struct FusedSegProj {
    /// The group key, `"{layer:03}.{act:?}"` — what an error message names.
    pub name: String,
    /// The **total** width: Σ of the parts.
    pub d_out: usize,
    pub d_in: usize,
    nblocks: u32,
    tail_w: u32,
    words: CudaSlice<u32>,
    gscale: CudaSlice<f32>,
    /// One entry a row: where that row's part's pair starts in `gscale`. The
    /// only thing a row concatenation cannot fold away.
    gs_off: CudaSlice<u32>,
    rscale: CudaSlice<f32>,
    tail: CudaSlice<u16>,
    rotation: Option<RotKey>,
    /// One artifact name per part, indexed by rank — what `Proj::site_name`
    /// returns, so an error still names a projection and not a group.
    part_names: Vec<String>,
}

impl FusedSegProj {
    /// The rotation every part of this group was quantized under. Equal across
    /// the parts by construction, re-asserted by `fused::segment_matrices`.
    pub fn rotation(&self) -> Option<RotKey> {
        self.rotation
    }

    /// The artifact name of the part at `rank`.
    pub fn part_name(&self, rank: usize) -> &str {
        self.part_names
            .get(rank)
            .map_or("(part outside the group)", String::as_str)
    }
}

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
    rt: std::sync::Arc<FusedRuntime>,
    proj: FusedProj,
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
    rt: std::sync::Arc<FusedRuntime>,
    proj: FusedInt4Proj,
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

// ---- read-only device views for 4E (`persist4e_cuda.rs`) ----
//
// Raw device addresses of what the runtime already holds, so the persistent
// segment reads the very buffers the eager path reads: the same words, row
// scales, tails, gains, rotations and tables. Nothing here allocates or
// writes; the views are valid while the runtime and its projections live,
// which the model guarantees by holding both.

fn dev_addr<T>(s: &CudaSlice<T>, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> u64 {
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
pub struct CudaSeg {
    rt: std::sync::Arc<FusedRuntime>,
    group: FusedSegProj,
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

/// The q8 embedding table on a card, and the `lm_head` that may share it.
///
/// `q` stays an `Arc` rather than being owned, because a tied model uploads
/// ONE table and points both ends at it. Two `CudaEmbed` values then share one
/// buffer, which is what the load-time tie check asserts.
pub struct CudaEmbed {
    rt: std::sync::Arc<FusedRuntime>,
    q: std::sync::Arc<QuantEmbed>,
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

/// The module, the shared tables, and the device everything lives on.
pub struct FusedRuntime {
    cuda: llvq_cuda::gpu::Cuda,
    /// The tile this module was **compiled** with, resolved from the card
    /// before the source was assembled.
    ///
    /// It lived here as a second `const TILE_BLOCKS = 128`, unlinked from
    /// `llvq_cuda::TILE_BLOCKS` — the defect that constant's own comment warns
    /// against, *"the Metal side carried it for months, TILE_BLOCKS defined
    /// twice, unlinked"*, reintroduced across the crate boundary. Held as one
    /// value because it feeds two things that must never disagree: the
    /// `#define` NVRTC compiled and the shared bytes every launch asks for.
    /// A launch whose shared parameter is smaller than the compiled tile
    /// overruns its staging area with no diagnostic.
    tile: llvq_cuda::tile::Tile,
    f_rot: CudaFunction,
    /// `rot_apply` with the rows in the grid — one launch a chunk instead of
    /// one a row. Not an `Option`: `rotate.cu` is in every layout's source
    /// list, because every layout rotates, so the symbol is in every unit.
    f_rot_rows: CudaFunction,
    /// Whichever entry point [`matvec_kernel_name`] gave for the layout — one
    /// kernel per runtime, chosen with the layout, so a stream and a kernel
    /// of different layouts cannot meet.
    f_matvec: CudaFunction,
    /// `tv_planes_seg_h` — present exactly when the layout can be segmented
    /// **and** the loader was asked to fuse. The `f_emb`/`g70_tabs` pattern: an
    /// `Option` whose `Some` is the *authorisation*, so no launch path can
    /// reach a function the source list never carried.
    f_matvec_seg: Option<CudaFunction>,
    tab: CudaSlice<u32>,
    /// The Golay70 constant tables `(cwtab, gtab)` — the canonical 4096-word
    /// codeword table and the 512-entry `GolayClassRec` table — present
    /// exactly when the layout is `Golay70`, the `f_emb` pattern.
    g70_tabs: Option<(CudaSlice<u32>, CudaSlice<u32>)>,
    /// The Tetra constant tables, present exactly when the layout is
    /// `Tetra48` — the `f_emb` pattern, where the `Some` is the
    /// *authorisation*: no launch path can reach a table the source list never
    /// carried.
    tetra_tabs: Option<TetraTabs>,
    /// Bytes the prefill kernel stages, formed once so the launch and the
    /// check against the card are one number.
    prefill_shared: u32,
    /// The `(rows, tile)` the prefill kernel was COMPILED at. The launch bound
    /// and the `#define`s come from this one value, so a runtime that staged
    /// one pair and launched another cannot exist.
    pub prefill: llvq_cuda::tile::Prefill,
    /// `tv_tetra48_rows_h` — present exactly when the layout carries a prefill
    /// kernel. The decode path never reaches it: at one row the one-row kernel
    /// is what runs, and every decode-time number stays attached to it.
    f_matvec_rows: Option<CudaFunction>,
    /// `tv_q4_h` — present exactly when the file carried an int4 record, which
    /// is when its source was appended to the translation unit. The
    /// `f_emb`/`tetra_tabs` pattern: the `Some` is the authorisation.
    ///
    /// 🕳️ Keyed on the SOURCE being in the unit and never on the layout. The
    /// register report asked for `tv_planes_seg_h` because its condition was
    /// "the layout is not Slot32", and the Tetra unit does not carry that
    /// source: `named symbol not found`, on a rented card, after a full load
    /// (2026-09-10). int4 is orthogonal to the layout — that orthogonality is
    /// what makes a mixed file possible — so the layout could not answer this
    /// question even in principle.
    f_int4: Option<CudaFunction>,
    /// `tv_q4_hh`, the same matvec reading an f16 activation. Present exactly
    /// when `f_int4` is: both entry points live in `tv_q4_h.cu`.
    f_int4_half: Option<CudaFunction>,
    /// `tv_q4_ht` and `tv_q4_hht` (`tv_q4_tile.cu`): present exactly when the
    /// runtime holds its int4 records tiled ([`crate::fused::Int4Layout`]),
    /// which is when that source is in the unit. The `f_int4` pattern.
    f_int4_tile: Option<(CudaFunction, CudaFunction)>,
    /// The kernels for widths past the card's shared-memory ceiling, from a
    /// translation unit of their own. `None` for every model whose widths all
    /// fit, which is every served model: their unit, its printed sha256 and
    /// their kernels do not move.
    wide: Option<WideFns>,
    rotations: HashMap<RotKey, RotBuffers>,
    /// The embedding kernels, `(gather, lm_head matvec)` — present exactly
    /// when the runtime was built with a quantized [`EmbedMode`], which is when
    /// their source was in the translation unit: `emb_q8.cu` under `Q8`,
    /// `emb_q4.cu` under `Q4`.
    f_emb: Option<(CudaFunction, CudaFunction)>,
    /// The mode `f_emb` was built for. [`FusedRuntime::upload_embed`] refuses
    /// a table of any other width, since the kernels compiled one.
    emode: EmbedMode,
    /// Dynamic shared memory the card allows one block **without asking**,
    /// read at startup — `tv_q8_h` stages the whole activation and must be
    /// refused past it. This is the *default* allowance and not the opt-in
    /// ceiling, because `tv_q8_h` is loaded through `func` and never opts in;
    /// the rotation, which does, is bounded in `new` instead and against both
    /// numbers (`llvq_cuda::shared`).
    shared_limit: usize,
    device: candle_core::CudaDevice,
    /// Largest `d_in` any projection takes — the staging bound the rotation
    /// kernel needs in shared memory.
    max_d_in: usize,
}

/// One quantized g64 embedding table (int8 or int4), resident on the device.
///
/// When the model ties its two ends (Qwen3-4B) a single instance serves both
/// the gather at the input and the `lm_head` at the output — which is the
/// point: the −365 MB lot B validated exist only if no f16 copy is ever
/// materialized beside this. When they are untied (Qwen3-8B) there are two
/// instances, one per table, and `EmbedTables::wiring` says which is which.
pub struct QuantEmbed {
    pub vocab: usize,
    pub d: usize,
    gpr: u32,
    /// Packed rows as `u32` words, rows word-aligned (`upload_embed` asserts
    /// the width that makes them so).
    words: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    /// Device bytes: payload + scales + biases.
    pub bytes: u64,
}

mod launch;
mod load;
mod ops;
mod runtime;
mod upload;

use self::launch::*;
pub use self::load::*;
pub use self::upload::*;

/// Re-exported so `fused_cuda::FusedSealed` keeps resolving. The type
/// itself moved to [`crate::fused`] with the device port: a loader's
/// return type cannot live in one backend's module.
pub use crate::fused::FusedSealed;
