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

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaSlice, LaunchConfig, PushKernelArg,
};
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
    /// The whole activation staged in shared, `d_in · 4` — this kernel does
    /// NOT tile, because its `d_in` is a hidden size. Held here so the launch
    /// and the load-time check against the device limit are one number.
    shared: u32,
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
    prefill: llvq_cuda::tile::Prefill,
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
    rotations: HashMap<RotKey, RotBuffers>,
    /// The q8 embedding kernels, `(gather, lm_head matvec)` — present exactly
    /// when the runtime was built with [`EmbedMode::Q8`], which is when their
    /// source was in the translation unit.
    f_emb: Option<(CudaFunction, CudaFunction)>,
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

impl FusedRuntime {
    /// Upload a loaded model and compile the kernels onto candle's stream.
    ///
    /// `emode` decides whether the q8 embedding kernels join the translation
    /// unit; it must match how the caller intends to build the model, and is
    /// taken here rather than re-read from the environment so a runtime and
    /// its loader cannot resolve the variable twice differently. `fuse` is
    /// taken for exactly the same reason, and decides whether the segmented
    /// entry point is looked up at all.
    pub fn new(
        model: &FusedModel,
        device: &Device,
        emode: EmbedMode,
        fuse: FuseMode,
    ) -> candle_core::Result<Loaded> {
        let dev = device.as_cuda_device()?.clone();
        let stream = dev.cuda_stream();
        // Before the source is assembled, not after: the tile is a `#define`
        // in that text, and `Cuda::device()` is a method on the compiled
        // module. The stream already carries its context, so no probe is
        // needed here — unlike the two benches, which build their source
        // before any context exists.
        let tile = llvq_cuda::tile::resolve(
            llvq_cuda::gpu::compute_cap_of(stream.context()).map_err(candle_core::Error::msg)?,
        );

        // The Slot32 translation unit is bit-identical to what shipped before
        // the layout switch existed — that arm is the comparison and the
        // fallback, and its register allocation must not move because a new
        // layout joined the build. Planes14 appends its three parts (in
        // planesbench's proven order: llvq_planes.cuh needs llvq_slot.cuh,
        // planes.cu needs matvec.cu) plus the half-storing entry point.
        let sources = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_rot.cuh", "rotate.cu"])
            .map_err(candle_core::Error::msg)?;
        let planes = match model.layout {
            FusedLayout::Slot32 => None,
            layout => Some(load_planes_sources(layout).map_err(candle_core::Error::msg)?),
        };
        let emb = match emode {
            EmbedMode::F16 => None,
            EmbedMode::Q8 => Some(load_emb_sources().map_err(candle_core::Error::msg)?),
        };
        // Appended when the FILE carried an int4 record, never when the layout
        // suggests one: int4 is orthogonal to the lattice layout, which is the
        // whole reason a mixed file can be served at all. Its composition
        // contract is `llvq_slot.cuh` then `matvec.cu` — the first two parts of
        // every unit — so it goes last and nothing before it moves.
        let int4 = match model.int4.is_empty() {
            true => None,
            false => Some(crate::fused::load_int4_sources().map_err(candle_core::Error::msg)?),
        };
        // The prefill pair, resolved once and refused here if it does not fit
        // — before NVRTC sees a byte. `LLVQ_PREFILL` is a measurement mode:
        // it moves no bit of any answer, only how many passes over the weight
        // stream a prompt costs.
        let prefill = llvq_cuda::tile::Prefill::resolve().map_err(candle_core::Error::msg)?;
        let defines = format!("{}{}", tile.define(), prefill.defines());
        let mut parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        if let Some((pp, overridden)) = &planes {
            parts.extend(pp.iter().map(String::as_str));
            if let Some(d) = overridden {
                eprintln!("WARNING: {} SOURCES OVERRIDDEN from {d}", model.layout.name());
            }
        }
        if let Some((es, overridden)) = &emb {
            parts.push(es.as_str());
            if let Some(d) = overridden {
                eprintln!("WARNING: emb_q8 SOURCE OVERRIDDEN from {d}");
            }
        }
        if let Some((cu, overridden)) = &int4 {
            parts.push(cu.as_str());
            if let Some(d) = overridden {
                eprintln!("WARNING: tv_q4_h SOURCE OVERRIDDEN from {d}");
            }
        }
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        // On the assembled text, which is the only place where "what NVRTC
        // compiles" and "what this runtime reports" are both visible.
        tile.assert_defined_in(&src.text);
        // The five binaries of `llvq-cuda` print this and the served path never
        // did, while `fused::load_planes_sources` cites "the printed sha256" as
        // the justification of its all-or-nothing override policy. A lot that
        // ADDS a file to the served unit is the one where that stops being a
        // formality: without this line, a run with `LLVQ_KERNEL_DIR` set is
        // traceable by a directory name and nothing else.
        println!(
            "NVRTC source: {} bytes, sha256 {} ({} parts), {}, {}, target {}",
            src.text.len(),
            src.sha256,
            parts.len(),
            tile.provenance(),
            prefill.provenance(),
            // `LLVQ_NVRTC_ARCH` is the one variable below the served door that
            // the config deliberately does not refuse — it names the card, not
            // the object — so it is printed here, where a journal copies from.
            llvq_cuda::gpu::arch()
        );
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;

        // The register report is a contract, not a diagnostic: the block
        // decoders keep their accumulators and `rot_mix` a KMAX-wide column
        // in registers, and a spill costs occupancy without changing a
        // result. Checked on the kernel this runtime will actually launch.
        let matvec_name = matvec_kernel_name(model.layout);
        let mut spill_checked = vec![matvec_name, "rot_apply", "rot_apply_rows"];
        // In the translation unit whenever ITS SOURCE is — read off the one
        // list that decides, never inferred from "the layout is not Slot32".
        //
        // 🕳️ It was `planes.is_some()`, which is true for every layout but
        // Slot32 — including `Tetra48`, whose list shares nothing with the
        // ball ones and carries no `tv_planes_seg_h.cu`. So the served Tetra
        // path compiled, loaded, reported 216 projections and 0.92 GB on the
        // card, and then died asking the driver for a symbol its own unit had
        // never contained: `no kernel tv_planes_seg_h: named symbol not found`
        // (2026-09-10, $0.03, the second card run of the served path). The
        // register report is a contract, and a contract that names a kernel
        // the build does not have is a crash rather than a check.
        if crate::fused::planes_source_names(model.layout).contains(&"tv_planes_seg_h.cu") {
            spill_checked.push("tv_planes_seg_h");
        }
        if emb.is_some() {
            spill_checked.extend(["emb_q8_gather", "tv_q8_h"]);
        }
        // 🚨 The prefill kernel, reported exactly when its source is in the
        // unit — and it is the arm where a spill is most likely and would cost
        // the most. `float acc[TETRA48_ROWS]` is one register a row: four is
        // nothing, sixteen is a sixth of a thread's 64-register budget at full
        // occupancy, and a spill there turns the accumulator into local memory
        // — which is DRAM, on the one kernel whose whole point is to touch
        // DRAM less. The report is what says so before a prompt is timed.
        if let Some(n) = crate::fused::rows_kernel_name(model.layout) {
            spill_checked.push(n);
        }
        // Same rule as the segmented kernel above: reported exactly when its
        // source is in the unit, which here is exactly when `int4` is `Some`.
        if int4.is_some() {
            spill_checked.push(crate::fused::INT4_KERNEL_NAME);
        }
        // Keyed on the same answer the lookup uses. It carries one accumulator
        // a row on top of the one-row kernel's 40 registers, so a spill here
        // is the number that says PREFILL_ROWS is too high — and a spill is a
        // hard stop, not a diagnostic.
        if let Some(n) = crate::fused::rows_kernel_name(model.layout) {
            spill_checked.push(n);
        }
        for name in spill_checked {
            let r = cuda.report(name).map_err(candle_core::Error::msg)?;
            if r.local_bytes != 0 {
                candle_core::bail!("{name}: {} bytes of spill", r.local_bytes);
            }
        }
        let f_matvec = cuda.func(matvec_name).map_err(candle_core::Error::msg)?;
        // The prefill entry point. `Some` is the authorisation, and it is
        // keyed on the LAYOUT naming one — which `fused.rs` pins to the source
        // list, so the pair cannot drift the way `tv_planes_seg_h` did.
        // Formed from the SAME tile the module was compiled at, and refused
        // against the card rather than assumed: this kernel is loaded through
        // `func` with no opt-in, so the default per-block allowance is the
        // bound. At the served tile of 128 four rows is exactly 49,152.
        // The VALUE here, where the tile is known; the CHECK against the card
        // ninety lines down, where `shared_limit` is read from the driver.
        // They were one statement and it named `shared_limit` before that line
        // existed — which this machine could not see, because the file needs
        // nvcc, and which the Space build reported in two errors.
        let prefill_shared = prefill.shared_bytes();
        let f_matvec_rows = match crate::fused::rows_kernel_name(model.layout) {
            Some(n) => Some(cuda.func(n).map_err(candle_core::Error::msg)?),
            None => None,
        };
        let f_int4 = match int4.is_some() {
            true => Some(
                cuda.func(crate::fused::INT4_KERNEL_NAME)
                    .map_err(candle_core::Error::msg)?,
            ),
            false => None,
        };
        // Looked up only when both the layout and the caller allow it: the
        // `Some` is the authorisation, so `forward_rotated_seg` has nothing to
        // fall back on rather than something to check.
        let f_matvec_seg = match (fuse, seg_kernel_name(model.layout)) {
            (FuseMode::On, Some(n)) => Some(cuda.func(n).map_err(candle_core::Error::msg)?),
            _ => None,
        };
        // `f_rot` is loaded further down, once the widest rotation is known:
        // staging past 48 KiB needs an opt-in posed on the *function*, and it
        // has to name the number of bytes. See the shared-memory block below.
        let f_emb = match emb {
            None => None,
            Some(_) => Some((
                cuda.func("emb_q8_gather").map_err(candle_core::Error::msg)?,
                cuda.func("tv_q8_h").map_err(candle_core::Error::msg)?,
            )),
        };

        // The 384-entry class table both sides of the format share, laid out
        // as the kernel's `ClassRec { float vals[5]; u32 len; }`.
        let fd = llvq_search::fastdec::FastDecoder::new();
        let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        for e in 0..TABLE_ENTRIES {
            tab[e * REC_WORDS + REC_WORDS - 1] = 1;
        }
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let norm = ((16 * lv.shell) as f64).sqrt();
            let base = (1 + ci) * REC_WORDS;
            for k in 0..lv.len {
                tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
            }
            tab[base + REC_WORDS - 1] = lv.len as u32;
        }
        let tab = cuda.up_u32(&tab).map_err(candle_core::Error::msg)?;

        // The Golay70 constant tables, uploaded once and shared by every
        // matrix — built from the same `Golay70Table` derivation the
        // transcoder encoded against (`fused::golay70_gpu_class_table`), so
        // encoder and decoder cannot drift apart.
        // The Tetra tables, from the SAME `llvq_search::tetra::Tetra` the
        // encoder used — `crate::fused::tetra48_tables` owns the shapes and is
        // checked on a machine without a card.
        let tetra_tabs = match model.layout {
            FusedLayout::Tetra48 => {
                let tb = crate::fused::tetra48_tables(&llvq_search::tetra::Tetra::new());
                Some(TetraTabs {
                    rows: cuda.up_u32(&tb.rows).map_err(candle_core::Error::msg)?,
                    prefixes: cuda.up_u32(&tb.prefixes).map_err(candle_core::Error::msg)?,
                    branches: cuda.up_u16(&tb.branches).map_err(candle_core::Error::msg)?,
                    suffixes: cuda.up_u32(&tb.suffixes).map_err(candle_core::Error::msg)?,
                    invnorm: cuda.up_f32(&tb.invnorm).map_err(candle_core::Error::msg)?,
                })
            }
            _ => None,
        };

        let g70_tabs = match model.layout {
            FusedLayout::Golay70 => {
                let g70 = llvq_artifact::runtime::Golay70Table::new(&fd);
                let cw = cuda
                    .up_u32(&crate::fused::golay70_gpu_codewords(&g70))
                    .map_err(candle_core::Error::msg)?;
                let gt = cuda
                    .up_u32(&crate::fused::golay70_gpu_class_table(&fd, &g70))
                    .map_err(candle_core::Error::msg)?;
                Some((cw, gt))
            }
            _ => None,
        };

        let dev_report = cuda.device().map_err(candle_core::Error::msg)?;
        // The DEFAULT allowance, and it stays the default on purpose: this
        // bound belongs to `tv_q8_h`, which stages `d` floats and is launched
        // through `func`, with no opt-in. Widening it here would loosen a
        // guard on a kernel that never asked the driver for anything.
        let shared_limit = dev_report.shared_per_block as usize;

        // The prefill staging, refused against the card rather than assumed:
        // that kernel is loaded through `func` with no opt-in, so the DEFAULT
        // per-block allowance is its bound — the same argument the rotation
        // makes just below. At the served tile of 128, four rows is exactly
        // 49,152 B, so this refuses nothing that exists today and names the
        // pair that would stop fitting.
        if crate::fused::rows_kernel_name(model.layout).is_some() && prefill_shared > shared_limit {
            candle_core::bail!(
                "the prefill kernel stages {prefill_shared} B ({} rows at tile {}), \
                 and the card allows {shared_limit}",
                prefill.rows,
                prefill.tile
            );
        }
        let prefill_shared = prefill_shared as u32;

        // The rotation is the one staging that can exceed the default — and
        // comparing it against `shared_limit` is what refused Qwen3-14B on
        // 2026-08-17 (69,632 o wanted, 49,152 offered by default, 101,376
        // available on request). Both bounds now, from `llvq_cuda::shared`,
        // which is where this arithmetic is testable: this file compiles
        // nowhere but inside an image build.
        for t in model.rotations.values() {
            llvq_cuda::shared::rot_plan(
                t.n,
                dev_report.shared_per_block as usize,
                dev_report.shared_per_block_optin as usize,
            )
            .map_err(candle_core::Error::msg)?;
        }
        // Posed on the function, once, for the widest rotation this model has
        // — before any launch, and never per token.
        let rot_bytes = model
            .rotations
            .values()
            .map(|t| llvq_cuda::shared::rot_bytes(t.n))
            .max()
            .unwrap_or(0);
        let f_rot = cuda
            .func_dynamic_shared("rot_apply", rot_bytes as u32)
            .map_err(candle_core::Error::msg)?;
        // Same bound, and it is the same bound for a reason: the rows variant
        // puts one row in a block and the rows in the grid, so its shared
        // memory per block is one row's, unchanged at any chunk length.
        let f_rot_rows = cuda
            .func_dynamic_shared("rot_apply_rows", rot_bytes as u32)
            .map_err(candle_core::Error::msg)?;

        let mut rotations = HashMap::new();
        for (&key, t) in &model.rotations {
            rotations.insert(key, upload_rotation(&cuda, t)?);
        }

        // Over the groups as well: under fusion most of the model's activations
        // are a group's, and a bound taken over the lone projections alone
        // would be a bound over `o_proj` and `down_proj`.
        let max_d_in = model
            .matrices
            .iter()
            .map(|m| m.d_in)
            .chain(model.groups.iter().map(|g| g.d_in))
            .max()
            .unwrap_or(0);
        let mut projs = Vec::with_capacity(model.matrices.len());
        for m in &model.matrices {
            projs.push(upload_matrix(&cuda, m, model.layout)?);
        }
        let mut seg_projs = Vec::with_capacity(model.groups.len());
        for g in &model.groups {
            seg_projs.push(upload_group(&cuda, g, model.layout)?);
        }
        let mut int4_projs = Vec::with_capacity(model.int4.len());
        for q in &model.int4 {
            int4_projs.push(upload_int4(&cuda, q, shared_limit)?);
        }

        Ok((
            Self {
                cuda,
                tile,
                f_rot,
                f_rot_rows,
                f_matvec,
                f_matvec_seg,
                tab,
                tetra_tabs,
                prefill_shared,
                prefill,
                f_matvec_rows,
                f_int4,
                g70_tabs,
                rotations,
                f_emb,
                shared_limit,
                device: dev,
                max_d_in,
            },
            projs,
            seg_projs,
            int4_projs,
        ))
    }

    /// The widest activation any projection takes.
    ///
    /// The rotation kernel stages the whole vector in shared memory, so this
    /// is the bound a caller checks against the card before uploading
    /// anything — `new` refuses past it rather than corrupting.
    pub fn max_d_in(&self) -> usize {
        self.max_d_in
    }

    /// One activation `[1, d_in]` in f16 → the same activation in the rotated
    /// basis, `[1, d_in]` in f32. Half of what `forward` used to do in one
    /// call; `crate::rotplan` says why it was split and why nothing is cached.
    /// The row loop lives in `model::group_forward` now, hence the one row.
    pub fn rotate(&self, proj: &FusedProj, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the projections of a group.",
                proj.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotOp { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for `n_rows` activations, ONE launch.
    ///
    /// `x` is `[n_rows, d_in]` f16 and contiguous — `model::row_block` narrows
    /// it out of one allocation, so its rows are `d_in` apart and no copy was
    /// made to put them there. The result is `[n_rows, d_in]` f32, contiguous,
    /// which is exactly what `forward_rotated_rows` reads.
    ///
    /// That last sentence is the whole point. Before this, a chunk was rotated
    /// a row at a time into `n_rows` separate allocations and then stacked with
    /// `Tensor::cat` — one device copy a row — to rebuild the very shape the
    /// matvec wanted. Now the shape comes out of the rotation.
    pub fn rotate_rows(
        &self,
        proj: &FusedProj,
        x: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{}: {rows} rows of activation for {n_rows}", proj.name);
        }
        if n_rows == 0 {
            candle_core::bail!("{}: a rotation of zero rows", proj.name);
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotRowsOp { rt: self, proj, n_rows };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one activation already in the rotated basis. `xr` is
    /// [`Self::rotate`]'s f32 output; `out_dims` is the *caller's* shape, so
    /// the result keeps the caller's rank, exactly as a `Linear` would.
    /// `y = W x` for an int4 projection, from the activation in its **natural**
    /// basis — there is no rotated form, and asking for one would be asking for
    /// a basis the weights were never quantized in.
    ///
    /// The one conversion in this file: `tv_q4_h` takes `x` in f32, "as in
    /// every projection kernel of this family", while the model carries f16.
    /// The rotated arms get their f32 for free because `rot_apply` produces it;
    /// this arm has no rotation to produce it, so it widens here. Not hidden in
    /// the kernel: the reference this path is checked against —
    /// `RawTensor::to_f32` — is f32 arithmetic, and narrowing the activation
    /// first would make the served row disagree with the row the file decodes
    /// to, which is the whole correctness test.
    pub fn forward_int4(
        &self,
        proj: &FusedInt4Proj,
        x: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        let xf = x.to_dtype(candle_core::DType::F32)?.contiguous()?;
        let op = FusedInt4Op {
            rt: self,
            proj,
            out_shape,
        };
        xf.apply_op1_no_bwd(&op)
    }

    /// Whether this runtime's layout carries a kernel that takes several rows.
    pub fn has_rows_kernel(&self) -> bool {
        self.f_matvec_rows.is_some()
    }

    /// Rows this runtime's prefill kernel was COMPILED to take.
    ///
    /// `model::group_forward` chunks by it. Read from the runtime and not from
    /// `tile::PREFILL_ROWS`, because under `LLVQ_PREFILL` they differ — and a
    /// chunk of eight handed to a kernel compiled at four is exactly the
    /// silent corruption the pair exists to prevent.
    pub fn prefill_rows(&self) -> usize {
        self.prefill.rows
    }

    /// `y = W X` for `n_rows` rotated rows, one launch.
    ///
    /// `xr` is `[n_rows, d_in]` f32, contiguous, and based at offset zero —
    /// [`RotRowsOp`] allocates it, so all three hold by construction. It used
    /// to be built by `Tensor::cat` of `n_rows` single-row rotations, one
    /// device copy each; the rotation produces the shape now.
    pub fn forward_rotated_rows(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill.rows {
            candle_core::bail!(
                "{n_rows} rows for a kernel compiled at {}",
                self.prefill.rows
            );
        }
        let op = FusedRowsOp {
            rt: self,
            proj,
            n_rows,
            out_shape: Shape::from(vec![n_rows, proj.d_out]),
        };
        xr.apply_op1_no_bwd(&op)
    }

    pub fn forward_rotated(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        let op = FusedOp {
            rt: self,
            proj,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for a fused group — one `rot_apply`, one f32
    /// `[1, d_in]` result. The parts share one `d_in` and one rotation key by
    /// construction (`fused::segment_matrices` re-asserts both), so there is
    /// one rotation to do and no choice of which.
    pub fn rotate_group(
        &self,
        g: &FusedSegProj,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != g.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", g.name, g.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the parts of the group.",
                g.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotSegOp { rt: self, group: g };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one fused group. `out_dims` is the caller's shape; the
    /// last axis becomes the group's **total** width, which
    /// `model::group_forward` then narrows back into the parts.
    pub fn forward_rotated_seg(
        &self,
        g: &FusedSegProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        // The only place the absent `CudaFunction` can surface, so it surfaces
        // as a message naming the group rather than as an `unwrap` in the
        // middle of a billed job.
        let f = self.f_matvec_seg.as_ref().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "{}: fused group launched by a runtime built without tv_planes_seg_h \
                 (LLVQ_FUSE=0, or a layout that does not segment)",
                g.name
            ))
        })?;
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = g.d_out;
            Shape::from(d)
        };
        let op = FusedSegOp {
            rt: self,
            f,
            group: g,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// Upload an int8 g64 tensor — the embedding — for the two q8 kernels.
    ///
    /// Every assumption the kernels compile in is asserted here rather than
    /// trusted: 8 bits, group 64, `d % 4 == 0` (rows on word boundaries),
    /// `vocab % 8 == 0` (whole warps, no bounds guard), and the staged
    /// activation within the card's shared memory.
    pub fn upload_embed_q8(
        &self,
        t: &llvq_artifact::RawTensor,
    ) -> candle_core::Result<QuantEmbed> {
        if self.f_emb.is_none() {
            candle_core::bail!("runtime built without the q8 kernels (LLVQ_EMBED=f16)");
        }
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{}: not a quantized tensor", t.name);
        };
        if q.bits != 8 || q.group != EMBED_GROUP {
            candle_core::bail!(
                "{}: int{} g{}, but the kernels hardcode int8 g{EMBED_GROUP}",
                t.name, q.bits, q.group
            );
        }
        if t.dims.len() != 2 {
            candle_core::bail!("{}: dims {:?}, an embedding is 2-D", t.name, t.dims);
        }
        let (vocab, d) = (t.dims[0], t.dims[1]);
        if !d.is_multiple_of(4) {
            candle_core::bail!("{}: d={d} is not a multiple of 4", t.name);
        }
        if !vocab.is_multiple_of(8) {
            candle_core::bail!("{}: vocab={vocab} is not a multiple of 8", t.name);
        }
        if d * 4 > self.shared_limit {
            candle_core::bail!(
                "{}: tv_q8_h asks for {} B of shared memory, the card offers {}",
                t.name, d * 4, self.shared_limit
            );
        }
        let gpr = d.div_ceil(EMBED_GROUP);
        if q.packed.len() != vocab * d
            || q.scales.len() != vocab * gpr
            || q.biases.len() != q.scales.len()
        {
            candle_core::bail!(
                "{}: inconsistent payload ({} bytes, {} scales, {} biases for {vocab}×{d})",
                t.name, q.packed.len(), q.scales.len(), q.biases.len()
            );
        }
        let words: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bytes =
            q.packed.len() as u64 + (q.scales.len() as u64 + q.biases.len() as u64) * 2;
        Ok(QuantEmbed {
            vocab,
            d,
            gpr: gpr as u32,
            words: self.cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
            scales: self.cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
            biases: self.cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
            bytes,
        })
    }

    /// Token ids `(.., l)` in u32 → embeddings `(.., l, d)` in f16, one
    /// gather launch for the whole call, rows dequantized on the device.
    pub fn embed(&self, q: &QuantEmbed, ids: &Tensor) -> candle_core::Result<Tensor> {
        let ids = ids.contiguous()?;
        let mut dims = ids.dims().to_vec();
        dims.push(q.d);
        let op = EmbedOp {
            rt: self,
            q,
            out_shape: Shape::from(dims),
        };
        ids.apply_op1_no_bwd(&op)
    }

    /// `logits = W_q8 · h` for hidden states `(.., d)` in f16 — the tied
    /// `lm_head`, one matvec launch per row, all rows into one buffer.
    pub fn lm_head(&self, q: &QuantEmbed, h: &Tensor) -> candle_core::Result<Tensor> {
        let dims = h.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != q.d {
            candle_core::bail!("lm_head q8 expects d={}, got {d_in}", q.d);
        }
        let h = h.contiguous()?.to_dtype(DType::F16)?;
        let out_shape = {
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = q.vocab;
            Shape::from(d)
        };
        let op = HeadOp {
            rt: self,
            q,
            out_shape,
        };
        h.apply_op1_no_bwd(&op)
    }
}

/// One int8 g64 embedding table, resident on the device.
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
    /// Packed int8 rows as `u32` words (`d % 4 == 0`, rows word-aligned).
    words: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    /// Device bytes: payload + scales + biases.
    pub bytes: u64,
}

/// The gather: token ids in, f16 rows out.
struct EmbedOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for EmbedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-emb-q8-gather"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the q8 gather has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let ids = storage.as_cuda_slice::<u32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous ids"))?;
        let ntok = end - start;
        if ntok == 0 {
            candle_core::bail!("q8 gather: zero tokens");
        }
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(ntok * self.q.d)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc emb: {e}")))?;
        let (f_gather, _) = self.rt.f_emb.as_ref().expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (ntok as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (d, gpr, ids_off) = (self.q.d as u32, self.q.gpr, start as u32);
        let mut b = self.rt.cuda.stream().launch_builder(f_gather);
        b.arg(&self.q.words)
            .arg(&self.q.scales)
            .arg(&self.q.biases)
            .arg(ids)
            .arg(&mut y)
            .arg(&d)
            .arg(&gpr)
            .arg(&ids_off);
        unsafe { b.launch(cfg) }
            .map_err(|e| candle_core::Error::msg(format!("emb_q8_gather: {e}")))?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// The tied `lm_head`: f16 hidden states in, f16 logits out.
struct HeadOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-lmhead-q8"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the q8 lm_head has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let x = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len == 0 || !len.is_multiple_of(self.q.d) {
            candle_core::bail!("lm_head q8: {len} values for d={}", self.q.d);
        }
        let rows = len / self.q.d;
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(rows * self.q.vocab)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc logits: {e}")))?;
        let (_, f_head) = self.rt.f_emb.as_ref().expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (self.q.vocab as u32 * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: (self.q.d * 4) as u32,
        };
        let (d, gpr) = (self.q.d as u32, self.q.gpr);
        for r in 0..rows {
            let x_off = (start + r * self.q.d) as u32;
            let y_off = (r * self.q.vocab) as u32;
            let mut b = self.rt.cuda.stream().launch_builder(f_head);
            b.arg(&self.q.words)
                .arg(&self.q.scales)
                .arg(&self.q.biases)
                .arg(x)
                .arg(&mut y)
                .arg(&d)
                .arg(&gpr)
                .arg(&x_off)
                .arg(&y_off);
            unsafe { b.launch(cfg) }
                .map_err(|e| candle_core::Error::msg(format!("tv_q8_h: {e}")))?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `rot_apply` as a candle op: f16 activation in, f32 rotated activation out.
///
/// The rotated activation used to be a scratch buffer keyed on `d_in` and held
/// between two launches under one lock; it is a tensor now, owned by the group
/// that shares it. `crate::rotplan` carries the argument. Output uninitialised
/// on purpose: `rot_apply` writes every coordinate of `[0, n)`.
struct RotOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
}

/// `rot_apply_rows` as a candle op: `[n_rows, d_in]` f16 in, the same shape in
/// f32 rotated out.
///
/// Deliberately a second op rather than a row count on [`RotOp`]: a chunk of
/// one takes the one-row path — `model::Proj::prepare_rows` short-circuits —
/// so every decode step stays on the kernel `bin/oracle` certifies, and this
/// one is reached only by a prefill.
struct RotRowsOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    n_rows: usize,
}

impl candle_core::CustomOp1 for RotRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply-rows"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation chunk"))?;
        let want = self.proj.d_in * self.n_rows;
        if end - start != want {
            candle_core::bail!(
                "activation of {} values for {} rows of d_in={}",
                end - start,
                self.n_rows,
                self.proj.d_in
            );
        }
        let rot = match self.proj.rotation {
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered, see fused_cuda.rs",
                self.proj.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(want) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot rows: {e}")))?;
        self.rt
            .cuda
            .launch_rot_rows(
                &self.rt.f_rot_rows,
                all,
                &rot.signbits,
                &rot.small,
                &mut xr,
                rot.n,
                rot.m,
                rot.k,
                rot.inv,
                start as u32,
                // The input rows are `d_in` apart, which is `rot.n` today and
                // is not the same statement: `rot.n` describes the transform,
                // `d_in` describes the buffer.
                self.proj.d_in as u32,
                self.n_rows as u32,
                rot.threads,
            )
            .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![self.n_rows, self.proj.d_in]),
        ))
    }
}

impl candle_core::CustomOp1 for RotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        if end - start != self.proj.d_in {
            candle_core::bail!(
                "activation of {} values for d_in={}",
                end - start,
                self.proj.d_in
            );
        }
        let rot = match self.proj.rotation {
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered, see fused_cuda.rs",
                self.proj.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(self.proj.d_in) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot: {e}")))?;
        self.rt
            .cuda
            .launch_rot(
                &self.rt.f_rot, all, &rot.signbits, &rot.small, &mut xr, rot.n, rot.m, rot.k,
                rot.inv, start as u32, rot.threads,
            )
            .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![1, self.proj.d_in]),
        ))
    }
}

/// The `CustomOp1` candle needs to let us at the tensor's storage.
///
/// There is no public way to reach a `Tensor`'s device pointer other than this
/// trait, which is the right design: it keeps the layout in the picture. We
/// require contiguity rather than honouring arbitrary strides — the kernel
/// stages 24 consecutive floats per block and a strided activation would be
/// silently wrong.
///
/// Since lot A4 it reads [`RotOp`]'s f32 output and launches nothing but the
/// matvec: it no longer rotates, and no longer decides anything.
struct FusedOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        // `(start, start + elem_count)` — a *range*, not `(offset, length)`.
        // Taking the second field for a length is silent at offset 0 and wrong
        // everywhere else, which is exactly how it survived the first run and
        // died on the second vector of the prompt.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.proj.d_in {
            candle_core::bail!(
                "rotated activation of {len} values for d_in={}",
                self.proj.d_in
            );
        }
        // The matvec kernels index `x` from the base pointer and take no
        // offset, so a nonzero start would read the wrong slice and return
        // finite, plausible, wrong numbers. Asserted, not assumed.
        if start != 0 {
            candle_core::bail!(
                "{}: rotated activation at offset {start}, while the matvec kernels \
                 index from the base",
                self.proj.name
            );
        }

        // f16, and uninitialised. Two things at once:
        //
        //  * `tv_slot_h` stores halves, so candle no longer needs a conversion
        //    kernel per projection — 252 launches a token, on a decode whose
        //    budget is half launch latency;
        //  * nothing zeroes it: the kernel writes `y[row]` for every row, the
        //    grid is exact and there is no bounds guard.
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.proj.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        let shared = self.rt.tile.shared_bytes();

        // One arm per layout. The Planes14 arm has no bases to pass — the
        // variant carries none — and the Slot32 arm is the exact call that
        // shipped before the switch existed.
        match &self.proj.stream {
            DeviceStream::Slot32 { words, bases } => self
                .rt
                .cuda
                .launch_slot_h(
                    &self.rt.f_matvec,
                    words,
                    bases,
                    &self.rt.tab,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?,
            DeviceStream::Tetra48 { words, stride_u32 } => {
                let t = self.rt.tetra_tabs.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "tetra48 stream without its constant tables: the runtime was built \
                         for another layout",
                    )
                })?;
                launch_tetra48_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    words,
                    *stride_u32,
                    t,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?
            }
            DeviceStream::Planes14 { words } => launch_planes_h(
                &self.rt.cuda,
                &self.rt.f_matvec,
                words,
                &self.rt.tab,
                &self.proj.gscale,
                &self.proj.rscale,
                &self.proj.tail,
                xr,
                &mut y,
                self.proj.nblocks,
                self.proj.tail_w,
                self.proj.d_out as u32,
                THREADS,
                shared,
            )
            .map_err(candle_core::Error::msg)?,
            DeviceStream::Planes12x {
                words,
                exc_idx,
                exc_words,
                row_exc,
            } => launch_planes12x_h(
                &self.rt.cuda,
                &self.rt.f_matvec,
                &[words, exc_idx, exc_words, row_exc],
                &self.rt.tab,
                &self.proj.gscale,
                &self.proj.rscale,
                &self.proj.tail,
                xr,
                &mut y,
                self.proj.nblocks,
                self.proj.tail_w,
                self.proj.d_out as u32,
                THREADS,
                shared,
            )
            .map_err(candle_core::Error::msg)?,
            DeviceStream::Golay70 {
                words,
                exc_idx,
                exc_words,
                row_exc,
            } => {
                let (cwtab, gtab) = self.rt.g70_tabs.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "Golay70 stream without constant tables, a runtime construction bug",
                    )
                })?;
                launch_golay70_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    &[words, exc_idx, exc_words, row_exc],
                    cwtab,
                    gtab,
                    &self.rt.tab,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?
            }
        }

        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `y = W x` for one int4 projection.
///
/// Its own type rather than an arm of [`FusedOp`], for the reason `RotSegOp`
/// is its own: the two carry different borrows and share no field. This one
/// has no rotation, no gain scale, no row scale and no tail to reach for.
struct FusedInt4Op<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedInt4Proj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedInt4Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-int4-matvec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ int4 kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let x = storage.as_cuda_slice::<f32>()?;
        // A range, not an offset and a length — `FusedOp` records what taking
        // the second field for a length costs, and it costs the same here.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.proj.d_in {
            candle_core::bail!(
                "{}: activation of {len} values for d_in={}",
                self.proj.name,
                self.proj.d_in
            );
        }
        // The kernel indexes `x` from the base pointer and takes no offset, so
        // a nonzero start would read the wrong slice and return finite,
        // plausible, wrong numbers.
        if start != 0 {
            candle_core::bail!(
                "{}: activation at offset {start}, and the kernel reads from the base",
                self.proj.name
            );
        }
        let f = self.rt.f_int4.as_ref().ok_or_else(|| {
            candle_core::Error::msg(
                "an int4 projection on a runtime built without `tv_q4_h`: its source \
                 is appended only when the file carried an int4 record",
            )
        })?;
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.proj.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        launch_q4_h(&self.rt.cuda, f, self.proj, x, &mut y, THREADS)
            .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `y = W X` for up to `PREFILL_ROWS` rotated activation rows at once.
///
/// The prefill counterpart of [`FusedOp`]. It exists so a prompt is not
/// decoded once per token: the one-row kernel reads the whole weight stream
/// per row, and a 5-shot MMLU question is several hundred rows.
struct FusedRowsOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    n_rows: usize,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec-rows"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.n_rows * self.proj.d_in {
            candle_core::bail!(
                "{} rows of {} values arrived as {len}",
                self.n_rows,
                self.proj.d_in
            );
        }
        if start != 0 {
            candle_core::bail!("activation at offset {start}, and the kernel reads from the base");
        }
        let f = self.rt.f_matvec_rows.as_ref().ok_or_else(|| {
            candle_core::Error::msg(
                "a prefill launch on a runtime whose layout carries no rows kernel",
            )
        })?;
        let t = self.rt.tetra_tabs.as_ref().ok_or_else(|| {
            candle_core::Error::msg("the rows kernel without the Tetra constant tables")
        })?;
        let words = match &self.proj.stream {
            DeviceStream::Tetra48 { words, .. } => words,
            _ => candle_core::bail!("the rows kernel on a stream that is not Tetra48"),
        };
        let stride_u32 = match &self.proj.stream {
            DeviceStream::Tetra48 { stride_u32, .. } => *stride_u32,
            _ => unreachable!("checked above"),
        };
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(self.n_rows * self.proj.d_out)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        launch_tetra48_rows_h(
            &self.rt.cuda,
            f,
            words,
            stride_u32,
            t,
            &self.proj.gscale,
            &self.proj.rscale,
            &self.proj.tail,
            xr,
            &mut y,
            self.proj.nblocks,
            self.proj.tail_w,
            self.n_rows as u32,
            self.proj.d_in as u32,
            self.proj.d_out as u32,
            THREADS,
            self.rt.prefill_shared,
        )
        .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// [`RotOp`] for a fused group — the same launch, keyed on the group's shared
/// rotation instead of one matrix's.
///
/// A separate type rather than an `enum` field on `RotOp`: the two carry
/// different borrows and nothing else, and a shared struct with two `Option`s
/// would be one `unwrap` away from launching a group's rotation on a lone
/// projection's width.
struct RotSegOp<'a> {
    rt: &'a FusedRuntime,
    group: &'a FusedSegProj,
}

impl candle_core::CustomOp1 for RotSegOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply-seg"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        if end - start != self.group.d_in {
            candle_core::bail!(
                "activation of {} values for d_in={}",
                end - start,
                self.group.d_in
            );
        }
        let rot = match self.group.rotation {
            None => candle_core::bail!(
                "{}: group without rotation, path not covered, see fused_cuda.rs",
                self.group.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(self.group.d_in) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot: {e}")))?;
        self.rt
            .cuda
            .launch_rot(
                &self.rt.f_rot, all, &rot.signbits, &rot.small, &mut xr, rot.n, rot.m, rot.k,
                rot.inv, start as u32, rot.threads,
            )
            .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![1, self.group.d_in]),
        ))
    }
}

/// [`FusedOp`] for a fused group: one launch over the row concatenation, with
/// the same three guards on the activation and the same uninitialised output.
///
/// It carries `f` rather than reading `rt.f_matvec_seg` here, so the absent
/// function is refused in [`FusedRuntime::forward_rotated_seg`] — before any
/// allocation — instead of inside a `CustomOp1` whose error surfaces two frames
/// away from the projection that caused it.
struct FusedSegOp<'a> {
    rt: &'a FusedRuntime,
    f: &'a CudaFunction,
    group: &'a FusedSegProj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedSegOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec-seg"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        // `(start, start + elem_count)` — a *range*, not `(offset, length)`,
        // exactly as in `FusedOp::cuda_fwd`. Taking the second field for a
        // length is silent at offset 0 and wrong everywhere else.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.group.d_in {
            candle_core::bail!(
                "rotated activation of {len} values for d_in={}",
                self.group.d_in
            );
        }
        // The matvec kernels index `x` from the base pointer and take no
        // offset, so a nonzero start would read the wrong slice and return
        // finite, plausible, wrong numbers. Asserted, not assumed.
        if start != 0 {
            candle_core::bail!(
                "{}: rotated activation at offset {start}, while the matvec kernels \
                 index from the base",
                self.group.name
            );
        }

        // f16, and uninitialised — the same argument as `FusedOp`, and it needs
        // restating because segmentation is exactly the shape in which it could
        // stop holding: a segmented matrix is a concatenation **by rows**, rows
        // partition the output, the grid is exact, and every row is *stored*
        // rather than accumulated into. No CTA outside a row's own warp writes
        // that row, so there is nothing to zero. ⚠️ If an exception region is
        // ever added to a segmented layout, this allocation becomes a memset in
        // the same commit.
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.group.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        let shared = self.rt.tile.shared_bytes();

        launch_planes_seg_h(
            &self.rt.cuda,
            self.f,
            &self.group.words,
            &self.rt.tab,
            &self.group.gscale,
            &self.group.gs_off,
            &self.group.rscale,
            &self.group.tail,
            xr,
            &mut y,
            self.group.nblocks,
            self.group.tail_w,
            self.group.d_out as u32,
            THREADS,
            shared,
        )
        .map_err(candle_core::Error::msg)?;

        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

fn upload_rotation(
    cuda: &llvq_cuda::gpu::Cuda,
    t: &RotationTables,
) -> candle_core::Result<RotBuffers> {
    Ok(RotBuffers {
        signbits: cuda.up_u32(&t.signbits).map_err(candle_core::Error::msg)?,
        small: cuda.up_f32(&t.small).map_err(candle_core::Error::msg)?,
        n: t.n as u32,
        m: t.m as u32,
        k: t.k as u32,
        inv: t.inv,
        // The widest block the driver allows, clamped to a warp at the bottom
        // so a narrow width still fills one.
        threads: (t.n as u32).next_power_of_two().clamp(32, 1024),
    })
}

/// The Planes14 twin of `Cuda::launch_slot_h` — `tv_slot_h`'s argument list
/// minus the bases array, which Planes14 does not have. Local to this crate
/// because `llvq-cuda` belongs to another lot; same grid (one warp per row,
/// whole blocks only, no bounds guard in the kernel), same generic `y` so
/// candle's `CudaSlice<half::f16>` can be handed over.
#[allow(clippy::too_many_arguments)]
fn launch_planes_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_h: {e}"))?;
    Ok(())
}

/// `tv_tetra48_h(words, row_stride_u32, rows, prefixes, branches, suffixes,
/// gscale, invnorm, rscale, tail, x, y, nblocks, tail_w)`.
///
/// The grid is `launch_planes_h`'s, unchanged — one warp per row, eight rows a
/// block — so a residency comparison between the two served layouts is like
/// for like. What differs is the second argument: `row_stride_u32`, which
/// Planes14 does not have because it addresses blocks flat. Passing the wrong
/// stride there does not fail; it reads every row after the first at a shifted
/// phase and returns plausible, wrong weights, which is why it travels with
/// the buffer in [`DeviceStream::Tetra48`] rather than being re-derived here.
#[allow(clippy::too_many_arguments)]
/// One int4 record onto the card: three arrays, no transcode.
///
/// The kernel reads these bytes as they sit on disk — that is what
/// distinguishes a stored-weight format from a lattice one — so the only work
/// here is the byte-to-u32 view and four checks the kernel's own header asks
/// the host to make.
fn upload_int4(
    cuda: &llvq_cuda::gpu::Cuda,
    q: &crate::fused::FusedInt4,
    shared_limit: usize,
) -> candle_core::Result<FusedInt4Proj> {
    // The four the kernel names, each because the arithmetic below it breaks
    // silently otherwise rather than faulting.
    if q.group != 128 {
        candle_core::bail!("{}: group {} — `tv_q4_h` hard-codes 128", q.name, q.group);
    }
    if !q.d_in.is_multiple_of(8) {
        candle_core::bail!(
            "{}: d_in {} is not a multiple of 8, so a row does not start on a u32 \\
             and every row after the first would read at a shifted nibble",
            q.name,
            q.d_in
        );
    }
    if !q.d_out.is_multiple_of(THREADS as usize / 32) {
        candle_core::bail!(
            "{}: d_out {} does not fill whole blocks of {} rows, and the kernel \\
             carries no bounds guard",
            q.name,
            q.d_out,
            THREADS / 32
        );
    }
    let gpr = q.d_in.div_ceil(q.group);
    if q.scales.len() != q.d_out * gpr || q.biases.len() != q.d_out * gpr {
        candle_core::bail!(
            "{}: {} scales and {} biases for {} rows × {gpr} groups",
            q.name,
            q.scales.len(),
            q.biases.len(),
            q.d_out
        );
    }
    if q.packed.len() != q.d_out * q.d_in / 2 {
        candle_core::bail!(
            "{}: {} packed bytes for {} × {} nibbles",
            q.name,
            q.packed.len(),
            q.d_out,
            q.d_in
        );
    }
    // The whole activation, not a tile: `d_in` here is a hidden size. Checked
    // against the card rather than assumed — the projection kernels tile
    // because their `d_in` can be an intermediate size, and this one must not
    // inherit a bound it does not share.
    let shared = q.d_in * 4;
    if shared > shared_limit {
        candle_core::bail!(
            "{}: staging {} values needs {shared} B of shared, the card allows {shared_limit}",
            q.name,
            q.d_in
        );
    }
    // Little-endian, so byte `b` of the stream is bits `8b` of the word and
    // nibble `i` stays at `4 · (i % 8)` — which is what `(p >> 4k) & 0xf`
    // reads. The disk order is low-nibble-first; a big-endian view here would
    // transpose every pair of columns and produce plausible, wrong weights.
    let words: Vec<u32> = q
        .packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.len() * 4 != q.packed.len() {
        candle_core::bail!(
            "{}: {} packed bytes is not a whole number of u32 words",
            q.name,
            q.packed.len()
        );
    }
    Ok(FusedInt4Proj {
        name: q.name.clone(),
        d_out: q.d_out,
        d_in: q.d_in,
        gpr: gpr as u32,
        wq: cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
        scales: cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
        biases: cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
        shared: shared as u32,
        bytes: q.bytes,
    })
}

/// `tv_q4_h(wq, scales, biases, x, y, d_in, gpr)` — the served int4 matvec.
///
/// One warp a row and 256-thread blocks, like every projection kernel here,
/// and NO tile: the activation is staged whole, so `shared` comes off the
/// projection rather than off `Tile`.
fn launch_q4_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    p: &FusedInt4Proj,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    threads: u32,
) -> Result<(), String> {
    let d_out = p.d_out as u32;
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: p.shared,
    };
    let d_in = p.d_in as u32;
    let mut b = cuda.stream().launch_builder(f);
    b.arg(&p.wq)
        .arg(&p.scales)
        .arg(&p.biases)
        .arg(x)
        .arg(y)
        .arg(&d_in)
        .arg(&p.gpr);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_h: {e}"))?;
    Ok(())
}

/// `tv_tetra48_rows_h(...)` — the prefill launch, `n_rows` activation rows.
///
/// Same grid as the one-row kernel: one warp an output row, eight rows a
/// block. What changes is the shared request — `PREFILL_ROWS` rows of tile
/// instead of one — and the three sizes the kernel needs to address a batch.
#[allow(clippy::too_many_arguments)]
fn launch_tetra48_rows_h(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    stride_u32: u32,
    t: &TetraTabs,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f16>,
    nblocks: u32,
    tail_w: u32,
    n_rows: u32,
    d_in: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    assert!(
        n_rows as usize <= llvq_cuda::tile::PREFILL_ROWS,
        "{n_rows} rows for a kernel compiled at {}",
        llvq_cuda::tile::PREFILL_ROWS
    );
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words)
        .arg(&stride_u32)
        .arg(&t.rows)
        .arg(&t.prefixes)
        .arg(&t.branches)
        .arg(&t.suffixes)
        .arg(gscale)
        .arg(&t.invnorm)
        .arg(rscale)
        .arg(tail)
        .arg(x)
        .arg(y)
        .arg(&nblocks)
        .arg(&tail_w)
        .arg(&n_rows)
        .arg(&d_in)
        .arg(&d_out);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_rows_h: {e}"))?;
    Ok(())
}

// Fifteen, and every one of them is a pointer the kernel takes: bundling them
// into a struct would put a second description of the argument list beside the
// `extern "C"` one, which is the drift this file spends its comments avoiding.
#[allow(clippy::too_many_arguments)]
fn launch_tetra48_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    stride_u32: u32,
    t: &TetraTabs,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words)
        .arg(&stride_u32)
        .arg(&t.rows)
        .arg(&t.prefixes)
        .arg(&t.branches)
        .arg(&t.suffixes)
        .arg(gscale)
        .arg(&t.invnorm)
        .arg(rscale)
        .arg(tail)
        .arg(x)
        .arg(y)
        .arg(&nblocks)
        .arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_h: {e}"))?;
    Ok(())
}

/// The segmented twin of [`launch_planes_h`] — same grid, one extra array.
///
/// `gs_off` sits between `gscale` and `rscale`, which is `tv_planes_seg_h`'s
/// declaration order. Note what the types buy here, against the remark
/// [`launch_planes12x_h`] makes: `gscale` is `CudaSlice<f32>` and `gs_off` is
/// `CudaSlice<u32>`, so transposing *those two* does not compile. The pair that
/// still could is `gscale`/`rscale`, and that hazard predates this lot.
///
/// The grid is `tv_planes_h`'s, unchanged, over the **total** `d_out`: on the
/// published 4B, q+k+v becomes 768 CTAs where it was 512+128+128, and gate+up
/// 2432 where it was 1216+1216. Same CTAs, one launch.
#[allow(clippy::too_many_arguments)]
fn launch_planes_seg_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    gs_off: &CudaSlice<u32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(gs_off).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_seg_h: {e}"))?;
    Ok(())
}

/// The Planes12x twin of [`launch_planes_h`] — same grid, four stream arrays
/// instead of one.
///
/// 🚨 **Read the grid, because it is the whole design decision.** It is
/// `tv_planes_h`'s grid, unchanged: `d_out·32/threads` CTAs, one warp per
/// output row, and *no* exception region. planesbench's `tv_planes12x` adds
/// `ceil(n_exc/8)` CTAs that `atomicAdd` into a `y` it memsets first; this
/// path instead hands each row its own slice of the exception table
/// (`row_exc`) so the corrections happen inside the row's warp. Consequences,
/// in the order they matter here:
///
///  * **`y` is not zeroed, and must not be.** `FusedOp::cuda_fwd` allocates
///    it uninitialised. That is sound for exactly the reason it is sound for
///    `tv_planes_h` — the grid is exact and every row is *stored*, never
///    accumulated into — and it stays sound here only because no CTA outside
///    a row's own warp writes that row. If anyone ever reintroduces an
///    exception region, the allocation upstream has to become a memset in the
///    same commit.
///  * **no atomic, so no `atomicAdd` on `__half`.** The accumulation is f32
///    from the first block to the last correction; the single narrowing to
///    binary16 is the final store, where candle would have narrowed anyway.
///    An `atomicAdd(__half*)` would have rounded every partial sum instead,
///    which is not the arithmetic the `Planes14` arm this replaces performs.
///
/// `arrays` is `[words, exc_idx, exc_words, row_exc]` in kernel order, taken
/// as one slice so a caller cannot silently transpose two `CudaSlice<u32>`
/// arguments of identical type — the one mistake here that compiles.
#[allow(clippy::too_many_arguments)]
fn launch_planes12x_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    arrays: &[&CudaSlice<u32>; 4],
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(arrays[0]).arg(arrays[1]).arg(arrays[2]).arg(arrays[3])
        .arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes12x_h: {e}"))?;
    Ok(())
}

/// The Golay70 twin of [`launch_planes12x_h`] — same grid (one warp per
/// output row, no exception region, no memset, no atomic: the row-sliced
/// correction design of `tv_planes12x_h`, see its launcher's 🚨 note, which
/// holds here word for word), with the two Golay70 constant tables added
/// between the stream arrays and the shared class table, in the kernel's
/// argument order: `tv_golay70_h(words, exc_idx, exc_words, row_exc, cwtab,
/// gtab, tab, gscale, rscale, tail, x, y, nblocks, tail_w)`.
///
/// `arrays` is `[words, exc_idx, exc_words, row_exc]`, one slice for the same
/// reason as Planes12x: four `CudaSlice<u32>` of identical type, and a silent
/// transposition is the one mistake here that compiles.
#[allow(clippy::too_many_arguments)]
fn launch_golay70_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    arrays: &[&CudaSlice<u32>; 4],
    cwtab: &CudaSlice<u32>,
    gtab: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(arrays[0]).arg(arrays[1]).arg(arrays[2]).arg(arrays[3])
        .arg(cwtab).arg(gtab)
        .arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_golay70_h: {e}"))?;
    Ok(())
}

fn upload_matrix(
    cuda: &llvq_cuda::gpu::Cuda,
    m: &FusedMatrix,
    layout: FusedLayout,
) -> candle_core::Result<FusedProj> {
    if !m.d_out.is_multiple_of(8) {
        // `tv_slot` has no bounds guard: a `return` before `__syncthreads()`
        // deadlocks, and it would break the full-warp mask the reduction
        // relies on. The grid is exact, so the host asserts instead.
        candle_core::bail!("{}: d_out={} is not a multiple of 8", m.name, m.d_out);
    }
    // A stream in the wrong layout would be read by the wrong kernel into
    // finite, plausible, wrong numbers — refused here, matrix by matrix,
    // rather than trusted to have been built consistently.
    let stream = match (&m.stream, layout) {
        (HostStream::Slot32 { words, bases }, FusedLayout::Slot32) => DeviceStream::Slot32 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
            bases: cuda.up_u32(bases).map_err(candle_core::Error::msg)?,
        },
        (HostStream::Tetra48 { words, stride_u32 }, FusedLayout::Tetra48) => {
            DeviceStream::Tetra48 {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                stride_u32: *stride_u32,
            }
        }
        (HostStream::Planes14 { words }, FusedLayout::Planes14) => DeviceStream::Planes14 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
        },
        (
            HostStream::Planes12x { words, exc_idx, exc_words, row_exc },
            FusedLayout::Planes12x,
        ) => {
            // cudarc refuses a zero-length upload. A matrix with no 5-level
            // block still needs a bound pointer, so it gets a one-word dummy
            // the kernel never dereferences: every row's slice is empty, so
            // `planes12x_row_correction`'s loop never runs. `exc_words` is
            // never empty — `pack_plane_bytes` appends the read-window pad —
            // and `row_exc` has `d_out + 1` entries, so only `exc_idx` needs
            // this.
            let dummy = [0u32];
            let idx: &[u32] = if exc_idx.is_empty() { &dummy } else { exc_idx };
            DeviceStream::Planes12x {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                exc_idx: cuda.up_u32(idx).map_err(candle_core::Error::msg)?,
                exc_words: cuda.up_u32(exc_words).map_err(candle_core::Error::msg)?,
                row_exc: cuda.up_u32(row_exc).map_err(candle_core::Error::msg)?,
            }
        }
        (
            HostStream::Golay70 { words, exc_idx, exc_words, row_exc },
            FusedLayout::Golay70,
        ) => {
            // Same zero-length rule as Planes12x: a matrix without any
            // exception block gets a one-word dummy the kernel never
            // dereferences — every row's slice is empty, so
            // `golay70_row_correction`'s loop never runs. `exc_words` is
            // never empty (`pack_plane_bytes` appends the read-window pad)
            // and `row_exc` has `d_out + 1` entries.
            let dummy = [0u32];
            let idx: &[u32] = if exc_idx.is_empty() { &dummy } else { exc_idx };
            DeviceStream::Golay70 {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                exc_idx: cuda.up_u32(idx).map_err(candle_core::Error::msg)?,
                exc_words: cuda.up_u32(exc_words).map_err(candle_core::Error::msg)?,
                row_exc: cuda.up_u32(row_exc).map_err(candle_core::Error::msg)?,
            }
        }
        _ => candle_core::bail!(
            "{}: host stream and runtime layout ({}) disagree",
            m.name,
            layout.name()
        ),
    };
    Ok(FusedProj {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        nblocks: m.nblocks as u32,
        tail_w: m.tail_w as u32,
        stream,
        gscale: cuda.up_f32(&m.gscale).map_err(candle_core::Error::msg)?,
        rscale: cuda.up_f32(&m.rscale).map_err(candle_core::Error::msg)?,
        // Binary16 bits, narrowed on the host by `fused::tail_f16_bits`.
        // cudarc refuses a zero-length upload, so a matrix whose `d_in` is a
        // multiple of 24 gets a one-element dummy the kernel never reads
        // (`tail_w == 0` makes `tail_dot_h`'s loop empty) — the same shape the
        // `Planes12x` exception index uses above.
        tail: cuda
            .up_u16(if m.tail.is_empty() { &[0u16] } else { &m.tail })
            .map_err(candle_core::Error::msg)?,
        rotation: m.rotation,
    })
}

/// Upload one fused group — the row concatenation, plus the offset table that
/// is the only thing it adds.
///
/// Every refusal below fires **before** a byte reaches the card, and every one
/// of them guards a failure that is finite, plausible and wrong rather than a
/// crash. The `gs_off` sweep in particular is the only guard possible on that
/// table: an entry past `gscale` would read arbitrary floats downstream of it
/// without ever leaving an arena allocator's allocation, so nothing on the card
/// would notice. It costs one pass over 25,600 `u32` a group, once, at load.
fn upload_group(
    cuda: &llvq_cuda::gpu::Cuda,
    g: &FusedGroup,
    layout: FusedLayout,
) -> candle_core::Result<FusedSegProj> {
    if seg_kernel_name(layout).is_none() {
        candle_core::bail!(
            "{}: fused group on layout {}, only planes14 segments",
            g.key,
            layout.name()
        );
    }
    // Per part **and** on the total, the `fused::segment_matrices` rule: the
    // total alone would let an individually ragged part through on a lucky sum,
    // and the unfused control arm could then not be launched at all.
    if !g.d_out.is_multiple_of(8) {
        candle_core::bail!("{}: d_out={} is not a multiple of 8", g.key, g.d_out);
    }
    for p in &g.parts {
        if !p.d_out.is_multiple_of(8) {
            candle_core::bail!("{}: {} has d_out={}, not a multiple of 8", g.key, p.name, p.d_out);
        }
    }
    if g.gs_off.len() != g.d_out
        || g.gscale.len() != 2 * g.parts.len()
        || g.rscale.len() != g.d_out
        || g.tail.len() != g.d_out * g.tail_w
    {
        candle_core::bail!(
            "{}: {} gs_off, {} centroids, {} scales, {} tail values for {} rows \
             of {} and {} parts",
            g.key,
            g.gs_off.len(),
            g.gscale.len(),
            g.rscale.len(),
            g.tail.len(),
            g.d_out,
            g.tail_w,
            g.parts.len()
        );
    }
    if let Some(bad) = g.gs_off.iter().position(|&o| o as usize + 1 >= g.gscale.len()) {
        candle_core::bail!(
            "{}: gs_off[{bad}]={} outside the table of {} centroids",
            g.key,
            g.gs_off[bad],
            g.gscale.len()
        );
    }
    let HostStream::Planes14 { words } = &g.stream else {
        candle_core::bail!("{}: group stream that is not Planes14", g.key);
    };
    Ok(FusedSegProj {
        name: g.key.clone(),
        d_out: g.d_out,
        d_in: g.d_in,
        nblocks: g.nblocks as u32,
        tail_w: g.tail_w as u32,
        words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
        gscale: cuda.up_f32(&g.gscale).map_err(candle_core::Error::msg)?,
        gs_off: cuda.up_u32(&g.gs_off).map_err(candle_core::Error::msg)?,
        rscale: cuda.up_f32(&g.rscale).map_err(candle_core::Error::msg)?,
        // Same zero-length rule as `upload_matrix`: cudarc refuses an empty
        // upload, so a group whose `d_in` is a multiple of 24 gets a one-element
        // dummy the kernel never reads (`tail_w == 0` makes `tail_dot_h`'s loop
        // empty).
        tail: cuda
            .up_u16(if g.tail.is_empty() { &[0u16] } else { &g.tail })
            .map_err(candle_core::Error::msg)?,
        rotation: g.rotation,
        part_names: g.parts.iter().map(|p| p.name.clone()).collect(),
    })
}

/// A model rebuilt from a sealed artifact **with its projections still
/// encoded**, plus what it took to do so.
pub struct FusedSealed {
    pub model: crate::model::Qwen3,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: candle_transformers::models::qwen3::Config,
    /// The runtime layout the projections were transcoded to.
    pub layout: FusedLayout,
    /// How the embedding and tied `lm_head` sit on the device.
    pub embed_mode: EmbedMode,
    /// Whether a shared activation is rotated once per group (`LLVQ_ROT_SHARE`).
    pub rot_share: crate::rotplan::RotShare,
    /// `rot_apply` launches one decode token costs. Printed on both arms: a
    /// gate showing identical tokens at 252 launches each proves nothing.
    pub rot_launches: usize,
    /// Whether the projections that share an activation were row-concatenated
    /// into one launch (`LLVQ_FUSE`).
    pub fuse: FuseMode,
    /// Matvec launches one decode token costs — 252 unfused on the published
    /// 4B, 144 fused. Printed on the arm line for the same reason
    /// [`Self::rot_launches`] is: a gate showing identical tokens while both
    /// arms issued 252 matvecs proves the tokens and nothing about the lot.
    pub matvec_launches: usize,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy on the device — the number that decides
    /// whether a model fits, and the one a disk figure must never stand in for.
    pub runtime_bytes: u64,
    /// Bytes the carried tensors occupy on the device: `2 · carried_weights`
    /// under `LLVQ_EMBED=f16`, the int8 payload of **every** embedding table
    /// plus the f16 norms under `q8` — one table when the model ties its two
    /// ends, two when it unties them. `carried_weights · 2` must no longer
    /// stand in for this: that identity is exactly what q8 breaks, by −365 MB
    /// on the tied 4B and −1.17 GB on the untied 8B.
    pub carried_bytes: u64,
}

/// Load a sealed artifact straight onto the fused path.
///
/// The counterpart of [`crate::sealed::load`], and the two must produce the
/// same logits — `bin/fusedrun` is what checks that. What differs is what
/// sits in VRAM: 8.04 GB of f16 there, `runtime_bytes` plus the embedding
/// here.
///
/// ## `dtype` must be F16, and this now says so out loud
///
/// It always was, in fact: [`FusedRuntime::rotate`] casts its input with
/// `to_dtype(DType::F16)` and the kernels *store* halves, so a model built at
/// F32 would have taken f16 tensors out of every projection. What changed on
/// 2026-08-09 is that the assumption acquired a consequence — the `KeepExact`
/// tail is now resident as binary16, and the argument that this costs nothing
/// (see [`crate::fused::tail_f16_bits`]) is *entirely* the fact that the dense
/// arm narrows the same columns to the run's dtype. At F32 that argument is
/// false and the tail would be the one place the fused path is coarser than
/// its reference. Refusing beats carrying a silently weaker claim.
/// [`load`] with the fusion mode named by the caller rather than read from the
/// environment — what lets `bin/fusedrun` run both arms in one process, each
/// dropped before the next loads, so the card holds one arm at a time and the
/// two share a card, a prompt and one NVRTC translation unit.
pub fn load_with(
    path: &str,
    device: &Device,
    dtype: DType,
    fuse: FuseMode,
) -> candle_core::Result<FusedSealed> {
    // The layout, the embedding mode and the hoist come from the environment
    // here — this is the measurement door, and every A/B in `docs/mesures/`
    // turns one of those variables between two processes.
    let layout = FusedLayout::from_env().map_err(candle_core::Error::msg)?;
    let emode = EmbedMode::from_env().map_err(candle_core::Error::msg)?;
    let share = crate::rotplan::RotShare::from_env().map_err(candle_core::Error::msg)?;
    // 🕳️ **F16 here by ALIGNMENT, not by default.** `bin/fusedrun` loads its
    // dense arm with `KvMode::F16` hardcoded (the dense arm of `bin/fusedrun`): its question
    // is the fused kernel, not the cache. The fused arm must take the same
    // one, or the comparison gains a second variable — which is what this
    // workstream spends its time forbidding. The served door takes the value
    // its config names, and answers a different question.
    load_resolved(path, device, dtype, layout, emode, share, fuse, crate::kvq::KvMode::F16, None)
}

/// [`load_with`] with the four choices already decided.
///
/// The served door. `crate::served::Served` reads them from a file and hands
/// them over, so nothing between that file and the transcoder consults the
/// environment — which is what makes the file the authority rather than a
/// suggestion that an unset variable could quietly outvote.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    device: &Device,
    dtype: DType,
    layout: FusedLayout,
    emode: EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: FuseMode,
    kv: crate::kvq::KvMode,
    // `Some("LLVQ_CONFIG")` from the served door, `None` from `load_with`. The
    // three choice lines below print it in place of the variable names, so a
    // served log does not attribute its choices to variables nobody set —
    // which is the ambiguity `crate::served` was written against. In bench
    // mode the variable names stay: they tell an A/B reader which one flipped.
    source: Option<&str>,
) -> candle_core::Result<FusedSealed> {
    use std::sync::Arc;
    let from = |var: &str| source.unwrap_or(var).to_string();

    if dtype != DType::F16 {
        candle_core::bail!(
            "fused path requested in {dtype:?}: it is f16 end to end (activations \
             converted, kernels storing halves, KeepExact tail resident in binary16 to \
             line up with what `sealed::load` narrows to the same dtype). Another dtype \
             would not make the model more accurate, it would make a comparison wrong."
        );
    }

    // The layout and the embedding mode arrive resolved and are printed next
    // to the device bytes they decide — an A/B where the arm has to be
    // inferred from a byte count is not an A/B.
    //
    // Both refusals before the 145 s transcode, not after it: a job that pays
    // for a load and then discovers its two variables are incompatible has
    // spent the money for nothing.
    crate::fused::check_fuse(layout, share, fuse).map_err(candle_core::Error::msg)?;
    let mut model =
        crate::fused::load_with(path, layout, fuse).map_err(candle_core::Error::msg)?;
    // The partition is checked inside `fused::load_with`; these are only its
    // counts, printed because `LLVQ_ROT_SHARE=1` reporting 252 launches — or
    // `LLVQ_FUSE=1` reporting 252 matvecs — would be a lot that did nothing
    // while looking green.
    let rot_launches = crate::rotplan::rot_launches(share, &model.matrices, &model.groups);
    let matvec_launches =
        crate::rotplan::matvec_launches_per_token(&model.matrices, &model.groups, model.int4.len());
    // Every projection the model holds, int4 included: the count sits beside a
    // launch count on the next two lines, and one that omitted 36 of 252 would
    // make the launches per token read as an inconsistency rather than as the
    // fact they are.
    let projections = model.matrices.len()
        + model.groups.iter().map(|g| g.parts.len()).sum::<usize>()
        + model.int4.len();
    println!(
        "shared rotation: {} ({}), {rot_launches} rot_launches/token \
         for {projections} projections",
        share.name(),
        from("LLVQ_ROT_SHARE")
    );
    println!(
        "projection fusion: {} ({}), {matvec_launches} matvec_launches/token \
         for {projections} projections ({} groups + {} lone + {} int4)",
        fuse.name(),
        from("LLVQ_FUSE"),
        model.groups.len(),
        model.matrices.len(),
        model.int4.len()
    );
    // The accounting is named on the line, because since lot A7a this number
    // is deliberately *not* the bench's: the tail is resident at binary16
    // here and at f32 in `planesbench`/`rtbits`, worth 0.075 b/weight on the
    // 4B. A reader comparing 4.729 to a published 4.804 must be able to see
    // from the log itself that they are two residencies, not a regression.
    println!(
        "fused layout: {} ({}), projections {:.2} GB on the card, \
         {:.3} b/weight (INFERENCE accounting: KeepExact tail in binary16; \
         the bench bills its own in f32)",
        layout.name(),
        from("LLVQ_FUSED_LAYOUT"),
        model.runtime_bytes as f64 / 1e9,
        model.runtime_bits_per_weight()
    );

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("tokenizer.json: {e}")))?;

    // Under q8, the embedding tables leave the carried list before any tensor
    // is built: nothing downstream may materialize an f16 copy of them. One
    // table when the ends are tied (4B), two when they are not (8B) — the
    // second is `lm_head.weight`, a different weight that gets a different
    // buffer. Both go through the exact `bin/embedq` arithmetic (same
    // function), or carry the file's own q8 bytes through untouched.
    let embed_tables = match emode {
        EmbedMode::F16 => None,
        EmbedMode::Q8 => Some(
            crate::fused::take_embed_tables(&mut model.raw, config.tie_word_embeddings)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    let (rt, projs, seg_projs, int4_projs) = FusedRuntime::new(&model, device, emode, fuse)?;
    let rt = Arc::new(rt);

    // Index every uploaded projection by the pair `Block::new_with` asks for —
    // from **both** sources. A lone projection yields one `Proj::Fused`; a group
    // yields one `Proj::FusedSeg` per part, all pointing at the *same*
    // `Arc<FusedSegProj>`, which is what `model::SegPlan::of` recognises with
    // `Arc::ptr_eq`. The `claimed != total_sites` check further down then also
    // catches a group one of whose parts the model never claimed.
    let mut by_site: HashMap<(usize, String), crate::model::Proj> = HashMap::new();
    for p in projs {
        let (layer, proj) =
            llvq_artifact::split_name(&p.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, proj),
            crate::model::Proj::Fused { rt: rt.clone(), proj: Arc::new(p) },
        );
    }
    // The int4 records, indexed by the same pair. They are never part of a
    // group: `fused::segment_matrices` reads `model.matrices`, which holds
    // lattice records only, so a `v_proj` served as int4 simply is not offered
    // to the fusion — and `model::group_forward` would refuse a mixed group
    // anyway, at `check_key`, because this projection's key is `None` and a
    // rotated group's is not.
    for q in int4_projs {
        let (layer, proj) = llvq_artifact::split_name(&q.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, proj),
            crate::model::Proj::FusedInt4 { rt: rt.clone(), proj: Arc::new(q) },
        );
    }
    // The row order comes from `fused::segment_matrices`, which read it off
    // `Act::consumers()`; `model::SegPlan::of` re-derives it from these very
    // fields and refuses a group that does not tile. Two places, one table.
    for (g, group) in model.groups.iter().zip(seg_projs) {
        let group = Arc::new(group);
        for part in &g.parts {
            by_site.insert(
                (part.layer, part.proj.clone()),
                crate::model::Proj::FusedSeg {
                    rt: rt.clone(),
                    group: group.clone(),
                    row0: part.row0,
                    d_out: part.d_out,
                    rank: part.rank,
                },
            );
        }
    }

    // Everything still carried — norms, and the embedding in f16 mode — as
    // ordinary tensors. Whatever encoding the file used, the device holds f16
    // here, so each costs `2 · len` bytes on the card.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut carried_bytes = 0u64;
    for t in &model.raw {
        carried_bytes += t.len() as u64 * 2;
        tensors.insert(
            t.name.clone(),
            Tensor::from_vec(t.to_f32(), t.dims.clone(), device)?.to_dtype(dtype)?,
        );
    }

    // The announced footprint, mode by mode, before the first launch. Both
    // modes count **every** embedding table the device holds: with the ends
    // untied there are two, and reporting one of them beside a correct total
    // is how a footprint line ends up contradicting the one below it.
    let quant_embed = match &embed_tables {
        None => {
            let carried = crate::fused::carried_embed_tables(&model.raw);
            println!(
                "{}",
                crate::fused::EmbedReport::new(EmbedMode::F16, &carried).line(&from("LLVQ_EMBED"))
            );
            None
        }
        Some(tables) => {
            let to_upload = tables.buffers();
            let report = crate::fused::EmbedReport::new(EmbedMode::Q8, &to_upload);
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let mut uploaded: Vec<Arc<QuantEmbed>> = Vec::with_capacity(to_upload.len());
            for (t, (_, packed, sb)) in to_upload.iter().zip(&report.tables) {
                let q = rt.upload_embed_q8(t)?;
                // A hard check, not a `debug_assert!`: release is the only
                // profile that runs on a card, and the whole point of this lot
                // is that the printed line cannot contradict the total below
                // it. Comparing what was announced to what was uploaded costs
                // nothing and closes exactly the defect being fixed.
                if q.bytes != *packed + *sb {
                    candle_core::bail!(
                        "carried table {}: {} bytes announced, {} uploaded",
                        t.name, *packed + *sb, q.bytes
                    );
                }
                carried_bytes += q.bytes;
                uploaded.push(Arc::new(q));
            }
            Some((uploaded, tables.wiring()))
        }
    };
    println!(
        "total expected on the card: {:.2} GB (projections {:.2} + carried {:.2})",
        (model.runtime_bytes + carried_bytes) as f64 / 1e9,
        model.runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    // Every site the artifact carries must be claimed; anything left over
    // means a name the loader and the model disagree about, and the model
    // would silently fall back to a `VarBuilder` lookup that cannot succeed.
    //
    // `remove` rather than `get`: a `Proj` is not `Clone` (a `FusedSeg` shares
    // one `Arc` between its parts, and cloning the enum would be a second way
    // to say that), so the map hands each site over exactly once. `total_sites`
    // is therefore read **before** the model claims any of them.
    let total_sites = by_site.len();
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.remove(&(layer, name.to_string())).inspect(|_| {
            claimed += 1;
        })
    };
    // `(ie, ih)` is `(0, 0)` when the ends are tied — two clones of one `Arc`,
    // exactly what shipped — and `(0, 1)` when they are not. The choice is
    // `EmbedTables::wiring`'s, tested on any machine; here it is only indexed.
    // The two indexing expressions below are the only lines of this path no
    // test on a developer machine can reach, and a one-character slip there
    // (`ie` twice) yields a model that runs and lies. This check is their
    // self-verification, on the card, at zero cost.
    if let Some((bufs, (ie, ih))) = &quant_embed {
        if Arc::ptr_eq(&bufs[*ie], &bufs[*ih]) != config.tie_word_embeddings {
            candle_core::bail!(
                "inconsistent q8 wiring: embedding and lm_head {} the same buffer \
                 while tie_word_embeddings = {}",
                if Arc::ptr_eq(&bufs[*ie], &bufs[*ih]) { "share" } else { "do not share" },
                config.tie_word_embeddings
            );
        }
    }
    // The KV mode arrives as an argument. `load_with` pins it to F16 for the
    // alignment its own comment gives; the served door passes what its config
    // names. Either way it is decided above this line, not here.
    //
    // 🚨 These two calls stopped compiling when `KvMode` arrived (KV q8,
    // 2026-08-15): this file is under `cfg(cuda)`, so NO development machine
    // type-checks it, and the breakage only surfaced at the first image build,
    // 255 s of CI, on 2026-08-16. `--features cuda` is not covered by
    // `cargo clippy --all-targets` on a Mac, and it is the same class of blind
    // spot as the `planesbench` wiring of the same day. `ops/check-cuda.sh`
    // closes it in a third of a second.
    let mut qwen = match &quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take, kv)?,
        Some((bufs, (ie, ih))) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8 {
                rt: rt.clone(),
                q: bufs[*ie].clone(),
            },
            crate::model::Head::Q8 {
                rt: rt.clone(),
                q: bufs[*ih].clone(),
            },
            kv,
        )?,
    };
    if claimed != total_sites {
        candle_core::bail!(
            "{claimed} projections claimed by the model out of {total_sites} carried by the \
             file"
        );
    }
    // The one place `LLVQ_ROT_SHARE` reaches a model: every other `Qwen3` keeps
    // `RotShare::Off` and cannot be moved by an exported variable.
    qwen.set_rot_share(share);

    Ok(FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        embed_mode: emode,
        rot_share: share,
        rot_launches,
        fuse,
        matvec_launches,
        quantized_weights: model.quantized_weights,
        // Counted at read time in `fused::load`, embedding included — the q8
        // extraction changes where those weights sit, not how many there are.
        carried_weights: model.carried_weights,
        file_bytes: model.file_bytes,
        runtime_bytes: model.runtime_bytes,
        carried_bytes,
    })
}
