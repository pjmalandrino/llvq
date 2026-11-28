//! The fused projections, running inside candle.
//!
//! This is where the kernel stops being a bench and starts being inference.
//! It holds the encoded streams on the device, in the layout
//! `LLVQ_FUSED_LAYOUT` names: `Tetra48` for the three served files,
//! `Planes14` (the default), `Planes12x`, `Slot32` or `Golay70` for ball
//! files. It replaces a lattice linear layer with two launches:
//!
//! ```text
//! x (f16) ──rot_apply──▶ x' (f32, rotated basis) ──tv_*_h──▶ y (f16)
//! ```
//!
//! and an int4 one with a single launch on the natural basis, with no
//! rotation (`tv_q4_*`, see [`FusedInt4Proj`]).
//!
//! ## Three things it does not do, on purpose
//!
//! * **It does not un-rotate.** The stored weights are `W' = W Qᵀ`, so the
//!   activation carries the `Q`. The identity is pinned on the CPU by
//!   `llvq-artifact/tests/fused_path_matches_dense.rs`; getting it backwards
//!   produces finite, plausible, wrong numbers and no error at all.
//! * **It does not loop a prompt through the one-row kernel** where the layout
//!   carries a rows kernel. The one-row kernel (`tv_slot`, `tv_tetra48_h`) is a
//!   matrix–*vector* product: one warp per output row, one activation staged
//!   in shared memory, what a decode token runs. A prompt goes through
//!   [`FusedRuntime::forward_rotated_rows`] instead: the rows kernel
//!   (`tv_tetra48_rows_h`), up to [`FusedRuntime::prefill_rows`] rows a launch,
//!   in chunks `model::group_forward` cuts. A layout with no rows kernel still
//!   takes a prompt row by row. `bin/ppl` keeps the dense path.
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

mod adapters;
mod ext;
mod head;
mod launch;
mod load;
mod moe;
mod ops;
mod runtime;

pub use adapters::*;
pub use ext::*;
pub use head::*;
pub use launch::*;
pub use load::*;
pub use moe::*;
pub use ops::*;
pub use runtime::*;

/// Threads per block for `tv_slot` and for the int4 launches (`tv_q4_h`,
/// `tv_q4_hs`): 256 = eight rows per block, the shape the bench has always
/// measured.
const THREADS: u32 = 256;
/// Entries in the device class table. The class field is nine bits, so 512
/// are addressable while 384 exist; the tail stays the origin so a truncated
/// or corrupt index cannot address out of bounds.
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;

/// The q8 embedding kernels — appended only under `LLVQ_EMBED=q8`, so the
/// translation unit of both f16-embedding arms stays byte-identical to what
/// every published number compiled.
const EMB_Q8_CU_EMBED: &str = include_str!("../../kernels/emb_q8.cu");

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

/// What [`FusedRuntime::new`] hands back: the runtime, and the three lists of
/// projections it uploaded — lattice, segmented, int4. Three because the model
/// indexes all three by the same `(layer, name)` pair and none of them can be
/// derived from another.
type Loaded = (FusedRuntime, Vec<FusedProj>, Vec<FusedSegProj>, Vec<FusedInt4Proj>);

/// One projection served as affine int4 g128, on the device.
///
/// It shares nothing with [`FusedProj`] and that is the point: an int4 record
/// carries stored weights in the **natural basis** — never GPTQ, never the
/// rotation, never a lattice index — so it has no rotation key, no gain scale,
/// no row scale and no tail. `calib.rs` writes it that way and `tv_q4_h.cu`
/// reads exactly those three arrays.
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
