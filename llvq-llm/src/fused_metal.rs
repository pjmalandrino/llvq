//! The Tetra kernel reached through candle, on Metal.
//!
//! The Metal twin of [`crate::fused_cuda`], and the adapter half of the port
//! in [`crate::device`]. The MSL lives in `kernels/llvq_tetra48.metal` and is
//! proved on its own by `llvq-metal/tests/tetra48_matches_rust.rs` and
//! `tetra48_matvec_matches_host.rs`, which drive it through metal-rs. This
//! module proves the BINDING: the same kernel, reached from a candle tensor.
//!
//! ## Why none of `llvq-metal` is reused
//!
//! `llvq-metal` pins `metal = "0.29"` (metal-rs). candle reaches Metal through
//! `objc2-metal`, and the two `Buffer` types are unrelated. A buffer made by
//! one cannot be bound by the other's encoder. Only the MSL text crosses.
//!
//! ## Five things about candle's Metal backend that cost a night if unknown
//!
//! Every one was measured on 2026-09-20, in a scratch crate, before this file
//! was written.
//!
//!   1. A bad MSL source **panics**; it does not return `Err`.
//!      `new_library_with_source` ends in `.unwrap()`. So the compile is
//!      wrapped in `catch_unwind` and the payload is re-raised as the error
//!      text, because `served::load_resolved` owes a refusal by name.
//!   2. Forgetting `set_threadgroup_memory_length` is **not** an error. The
//!      kernel dispatches, completes, and writes zeros. No fault, no warning.
//!   3. The allocator hands your buffers back out as scratch. `new_buffer`
//!      reuses anything at `Arc::strong_count == 1`, so the tables and the
//!      weight stream must be OWNED for the life of the model.
//!   4. Output buffers are pooled, rounded up to a power of two, and not
//!      zeroed. A row the kernel skips returns a previous tensor's bytes,
//!      which look like a plausible activation.
//!   5. candle's own `Kernels` cache cannot hold this library: `Source` is a
//!      closed enum over 15 `include_str!` constants. This module carries its
//!      own cache, and its key includes the injected tile, because a define
//!      that fails to land changes the staging and not the arithmetic.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// The coordinate order the quads come out in. Mirrors `TETRA48_ORDER` in the
/// shader and `TETRA48_ORDER` in `llvq_tetra48.cuh`.
const ORDER: [usize; llvq_core::DIM] = [
    0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23,
];

/// Lanes in an Apple SIMD-group. The kernel gives one to a row.
const LANES: usize = 32;
/// Threads a threadgroup, so eight rows each. The kernel carries no
/// `row < d_out` guard, exactly like the CUDA one, so `d_out` must be a
/// multiple of the rows a group covers. Asserted at upload.
const GROUP: usize = 256;
/// Lanes `tv_tetra48_metal_ilp` gives one output row, and why the served
/// path does NOT use it.
///
/// Eight lanes instead of 32 gives each lane 13 blocks rather than 3.3, which
/// is what the roofline asks for: the 32-lane kernel reaches 22 % of the
/// measured 340 GB/s AND 19 % of the measured 12.5 TFLOPS, and a kernel that
/// saturates neither is waiting on latency.
///
/// It works, and it is refused. The lane count is part of the ARITHMETIC:
/// each lane accumulates a different subset and the butterfly has a different
/// depth, so the answer moves in the last bits. Measured 2026-09-21: the
/// 60-token identity against the dense arm BREAKS at token 57, "dans
/// l'atmosphere" becoming "dans la atmosphere", while the throughput gain in
/// situ is 5 percent, 48.1 to 50.4 tok/s. The isolated bench had promised 15.
///
/// Five percent is not worth the property the whole path is gated on.
#[allow(dead_code)]
const ROW_LANES: usize = 8;
/// Threadgroup memory an M3 Max offers, measured 2026-09-20 through
/// `MTLDevice.maxThreadgroupMemoryLength`.
///
/// A constant rather than a device query on purpose: the kernels are compiled
/// against it, so a machine that offered less has to be refused rather than
/// quietly served a kernel that stages past its limit. Asking the device and
/// adapting would hide exactly that. [`MetalRuntime::new`] and
/// [`MetalRuntime::from_tables`] refuse such a device by name
/// (`check_threadgroup_memory`).
const THREADGROUP_LIMIT: usize = 32_768;

/// Columns `tv_q4_metal_tiled` stages a tile, for an int4 record whose whole
/// activation does not fit in `THREADGROUP_LIMIT`.
///
/// Every sealed file's int4 `down_proj` is past the wall: d_in 9,728 at 4B,
/// 12,288 at 8B, 17,408 at 14B. Records that fit keep `tv_q4_metal` and their
/// exact code path.
///
/// 2,048 and not the 8,192 that would fill the limit, for residency: a tile of
/// 8,192 asks all 32,768 B, so likely one threadgroup a core, while 8 KB leaves
/// room for several. Pinning 18.7 KB of threadgroup memory already capped
/// residency on this M3 Max and cost +8 to +23 % (commit 82c029c). llvq-tetra
/// ships 2,048 as the value its kernels were validated with. The tile moves no
/// bit: a multiple of 256 columns is a multiple of 32 words, so lane `l` walks
/// the untiled kernel's words in the untiled order
/// (`llvq-metal/tests/q4_tiled_matches_served.rs`).
pub const Q4_TILE_COLS: usize = 2048;

/// Load-time options of the Metal path, for measurement harnesses.
///
/// None of them changes a served bit: each kernel a field names is gated
/// equal to the served one, bit for bit. The served door takes
/// [`MetalTuning::default`], which is the served path as it stood before this
/// struct existed, plus the tiled int4 route for the records past the wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTuning {
    /// Columns a tile for the int4 records past the wall.
    pub q4_tile_cols: usize,
    /// The Tetra matvec. `tv_tetra48_metal_lut` is the served one.
    pub tetra_kernel: &'static str,
    /// The int4 row loop: `false` is the served `tv_q4_metal` /
    /// `tv_q4_metal_tiled`, `true` their prefetching twins
    /// (`kernels/tv_q4_pf.metal`).
    pub q4_prefetch: bool,
    /// The half-io twins: the Tetra matvec stores f16 (`_luth`, `_lutdh`) and
    /// the int4 matvecs read and store f16 (`tv_q4_metal_h`,
    /// `tv_q4_metal_tiled_h`), so the adapter's casts around them, 336
    /// launches a token on the sealed 4B, are not issued. The same roundings
    /// in another place, so the same bits.
    pub half_io: bool,
    /// Prompt rows one launch of the Tetra prefill kernel takes
    /// (`tv_tetra48_metal_lutdr`). `1`, the served value, means no rows
    /// kernel: the prefill fans out row by row through the matvec. 2, 4 or 8
    /// decode each block once for that many rows, with every row's bits those
    /// of the matvec narrowed to f16.
    pub prefill_rows: usize,
    /// The int4 records take the prefill chunks too, through
    /// `tv_q4_metal_rows`; requires `prefill_rows` above 1. `false`, the
    /// served value, keeps their row-by-row loop.
    pub q4_rows: bool,
    /// The one-row rotations of width at most 8,192 run `rot_apply_tg_metal`,
    /// which stages the vector in threadgroup memory, instead of
    /// `rot_apply_metal`, which runs every stage through device memory. The
    /// same arithmetic in the same order, so the same bits. `false` is served.
    pub rot_tg: bool,
}

impl Default for MetalTuning {
    fn default() -> Self {
        Self {
            q4_tile_cols: Q4_TILE_COLS,
            tetra_kernel: SERVED_TETRA,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
        }
    }
}

/// Rows the Tetra prefill kernel can take a launch (`TETRA48_PREFILL_MAX`).
pub const TETRA_PREFILL_MAX: usize = 8;

impl MetalTuning {
    /// The tuning a `bin/metalrun` arm names, `lutd+h+pq` for one: a Tetra
    /// matvec, then modifiers. The one reader of that spelling, shared by the
    /// bench and by the served configs that name their Metal kernels
    /// ([`MetalTuning::of_served`]), so an arm a journal timed and a config
    /// that serves it cannot mean two things. `+grp` and `+ker` are attention
    /// forms, not kernels: the caller takes them off first.
    pub fn parse_arm(s: &str) -> Result<Self> {
        let mut parts = s.split('+');
        let kernel = match parts.next().unwrap_or("") {
            "lut" => SERVED_TETRA,
            "lutp" => "tv_tetra48_metal_lutp",
            "lutd" => "tv_tetra48_metal_lutd",
            "lutdp" => "tv_tetra48_metal_lutdp",
            "lutd2" => "tv_tetra48_metal_lutd2",
            "lutd4" => "tv_tetra48_metal_lutd4",
            "lutdt" => "tv_tetra48_metal_lutdt",
            other => candle_core::bail!("arm {s:?}: {other:?} is not lut, lutp, lutd, lutdp, lutd2, lutd4 or lutdt"),
        };
        let mut t = MetalTuning { tetra_kernel: kernel, ..Default::default() };
        for m in parts {
            match m {
                "q4p" => t.q4_prefetch = true,
                "h" => t.half_io = true,
                "p" => t.prefill_rows = TETRA_PREFILL_MAX,
                "r" => t.rot_tg = true,
                "pq" => {
                    t.prefill_rows = TETRA_PREFILL_MAX;
                    t.q4_rows = true;
                }
                c if c.starts_with('t') && c.len() > 1 => {
                    t.q4_tile_cols =
                        c[1..].parse().map_err(|_| candle_core::Error::msg(format!("arm {s:?}: tile {c:?}")))?
                }
                other => candle_core::bail!("arm {s:?}: unknown modifier {other:?}"),
            }
        }
        t.validate()?;
        Ok(t)
    }

    /// The tuning a served config names in `metal_kernels`. `Default` is
    /// [`MetalTuning::default`], what every config before 2026-11-26 served.
    pub fn of_served(k: crate::served::MetalKernels) -> Result<Self> {
        match k.arm() {
            None => Ok(Self::default()),
            Some(arm) => Self::parse_arm(arm),
        }
    }
}

/// What the runtime under a loaded model launches, found through its first
/// Tetra projection; `None` when the model has no Metal Tetra projection.
pub fn tuning_of(m: &crate::model::Qwen3) -> Option<MetalTuning> {
    m.blocks.iter().find_map(|b| {
        [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, &b.gate_proj, &b.up_proj, &b.down_proj].into_iter().find_map(|p| {
            match p {
                crate::model::Proj::Lattice(d) => {
                    d.as_any().and_then(|a| a.downcast_ref::<MetalLattice>()).map(|l| l.rt.tuning())
                }
                _ => None,
            }
        })
    })
}

/// The served Tetra matvec on Metal.
pub const SERVED_TETRA: &str = "tv_tetra48_metal_lut";

impl MetalTuning {
    /// Refuses a bad tile or an unknown kernel when the load starts, whether
    /// or not a record of this file needs it. A zero tile makes the kernel's
    /// `c0 += tile_cols` loop forever: a GPU hang, not an error.
    pub fn validate(&self) -> Result<()> {
        let c = self.q4_tile_cols;
        if c == 0 || !c.is_multiple_of(256) {
            candle_core::bail!(
                "q4 tile_cols {c} is not a positive multiple of 256. A tile of tile_cols/8 words must be a \
                 multiple of 32 or the lanes interleave differently and the f32 moves"
            );
        }
        if c * 4 > THREADGROUP_LIMIT {
            candle_core::bail!(
                "q4 tile_cols {c} stages {} B of threadgroup memory against {THREADGROUP_LIMIT}",
                c * 4
            );
        }
        source_of(self.tetra_kernel)?;
        let k = self.tetra_kernel;
        if k != SERVED_TETRA && !TETRA_ROW_LOOPS.contains(&k) && !TETRA_REFERENCES.contains(&k) {
            candle_core::bail!(
                "{k} is not a 32-lane Tetra matvec with the served arguments and an f32 output: \
                 {SERVED_TETRA}, a row loop of TETRA_ROW_LOOPS or a reference of TETRA_REFERENCES. \
                 The f16 twins come through half-io, the prefill kernel through prefill rows"
            );
        }
        if self.half_io {
            if tetra_half_twin(self.tetra_kernel).is_none() {
                candle_core::bail!(
                    "half-io: {} has no half-store twin; tv_tetra48_metal_lut and tv_tetra48_metal_lutd have one",
                    self.tetra_kernel
                );
            }
            if self.q4_prefetch {
                candle_core::bail!("half-io and the prefetching int4 twins are two arms; name one");
            }
        }
        if !matches!(self.prefill_rows, 1 | 2 | 4 | 8) {
            candle_core::bail!(
                "prefill rows {}: the Tetra prefill kernel takes 2, 4 or 8 rows a launch, and 1 is none",
                self.prefill_rows
            );
        }
        if self.q4_rows && self.prefill_rows == 1 {
            candle_core::bail!("q4 rows: the int4 prefill kernel takes the chunks prefill rows names, and 1 is none");
        }
        Ok(())
    }
}

/// The half-store twin of a Tetra matvec, if it has one.
pub fn tetra_half_twin(name: &str) -> Option<&'static str> {
    match name {
        "tv_tetra48_metal_lut" => Some("tv_tetra48_metal_luth"),
        "tv_tetra48_metal_lutd" => Some("tv_tetra48_metal_lutdh"),
        "tv_tetra48_metal_lutdt" => Some("tv_tetra48_metal_lutdth"),
        _ => None,
    }
}

/// The row-loop twins of the served Tetra matvec (`kernels/llvq_tetra48_rows.metal`).
/// Each has the served kernel's arguments and, by gate, its bits.
pub const TETRA_ROW_LOOPS: [&str; 6] = [
    "tv_tetra48_metal_lutp",
    "tv_tetra48_metal_lutd",
    "tv_tetra48_metal_lutdp",
    "tv_tetra48_metal_lutd2",
    "tv_tetra48_metal_lutd4",
    "tv_tetra48_metal_lutdt",
];

/// The reference variants of the served Tetra matvec (`kernels/llvq_tetra48.metal`):
/// the faithful port and three other decoders of it (`_ar`, `_u2`, `_c3`).
/// Each has the served kernel's arguments, 32 lanes a row, an f32 output, and
/// no threadgroup memory but the staging at index 0 the host sizes.
pub const TETRA_REFERENCES: [&str; 4] =
    ["tv_tetra48_metal", "tv_tetra48_metal_ar", "tv_tetra48_metal_u2", "tv_tetra48_metal_c3"];

/// Whether a Tetra matvec stages the decoder tables in threadgroup memory and
/// loops its simdgroups over the rows (`_lutdt`, `_lutdth`): its own dispatch.
pub fn tetra_loops_rows(name: &str) -> bool {
    matches!(name, "tv_tetra48_metal_lutdt" | "tv_tetra48_metal_lutdth")
}

/// Rows a simdgroup of `_lutdt` takes, on average, from one copy of the tables.
/// The threadgroup count is `ceil(d_out / (simdgroups * this))`.
pub const TETRA_TG_LOOP_ROWS: usize = 4;

/// Threadgroup bytes of `_lutdt`'s five tables, in binding order: rank rows
/// (4,096 u32), branches (1,024 u16), prefixes and suffixes (128 u8 each),
/// the 64 values of `F1R_VAL`.
const TETRA_TG_TABLES: [usize; 5] = [16_384, 2_048, 128, 128, 256];

/// Whether a Tetra matvec reads the activation from device memory, with no
/// threadgroup staging to size.
fn tetra_reads_direct(name: &str) -> bool {
    matches!(
        name,
        "tv_tetra48_metal_lutd"
            | "tv_tetra48_metal_lutdp"
            | "tv_tetra48_metal_lutd2"
            | "tv_tetra48_metal_lutd4"
            | "tv_tetra48_metal_lutdh"
            | "tv_tetra48_metal_lutdt"
            | "tv_tetra48_metal_lutdth"
    )
}

/// Rows a simdgroup of a Tetra matvec carries: 1 for every kernel but the
/// multi-row twins, which keep 32 lanes a row and give a lane R rows.
pub fn tetra_rows_per_simd(name: &str) -> usize {
    match name {
        "tv_tetra48_metal_lutd2" => 2,
        "tv_tetra48_metal_lutd4" => 4,
        _ => 1,
    }
}

/// `None` when the whole activation of an int4 record fits in threadgroup
/// memory, the tile otherwise.
fn int4_tile_for(name: &str, d_in: usize, gpr: usize, tile: usize) -> Result<Option<usize>> {
    // The kernel indexes `s0 = row * gpr` and `g = s0 + c / 128`: a gpr that
    // is not ceil(d_in / 128) reads another row's scales, silently.
    if gpr != d_in.div_ceil(128) {
        candle_core::bail!(
            "{name}: {gpr} groups a row for d_in {d_in}, the kernel wants {}",
            d_in.div_ceil(128)
        );
    }
    if d_in * 4 <= THREADGROUP_LIMIT {
        return Ok(None);
    }
    MetalTuning { q4_tile_cols: tile, ..Default::default() }.validate()?;
    Ok(Some(tile))
}

/// Refuses by name a device that offers less threadgroup memory than
/// [`THREADGROUP_LIMIT`], the bound every staging of this file is sized to.
fn check_threadgroup_memory(dev: &candle_core::MetalDevice) -> Result<()> {
    use objc2_metal::MTLDevice as _;
    let offered = dev.device().as_ref().maxThreadgroupMemoryLength();
    if offered < THREADGROUP_LIMIT {
        candle_core::bail!(
            "the Metal device offers {offered} B of threadgroup memory; the kernels of this file \
             stage up to {THREADGROUP_LIMIT} B"
        );
    }
    Ok(())
}

/// Threads a threadgroup for a launch that asks `want`: `want` when the
/// pipeline takes it, else the largest `want / 2^k` it takes, down to one
/// SIMD-group, else a refusal by name.
///
/// Only for a kernel whose result does not depend on the group: each row is
/// the SIMD-group of its global thread index (`gid >> 5`), and the staging
/// loops over `threads_per_threadgroup`. Halving keeps every threadgroup
/// whole on a grid `want` tiled.
fn group_within(pipe: &ComputePipeline, name: &str, want: usize) -> Result<usize> {
    let max = pipe.max_total_threads_per_threadgroup();
    let mut threads = want;
    while threads > max && threads > LANES {
        threads /= 2;
    }
    if threads > max {
        candle_core::bail!(
            "{name}: the pipeline allows {max} threads a threadgroup, fewer than one SIMD-group of {LANES}"
        );
    }
    Ok(threads)
}

/// `want` threads a threadgroup, refused by name when the pipeline takes
/// fewer: for a kernel whose indexing assumes that count.
fn group_exactly(pipe: &ComputePipeline, name: &str, want: usize) -> Result<usize> {
    let max = pipe.max_total_threads_per_threadgroup();
    if max < want {
        candle_core::bail!("{name}: the launch needs {want} threads a threadgroup, the pipeline allows {max}");
    }
    Ok(want)
}

const SRC_TETRA: &str = include_str!("../kernels/llvq_tetra48.metal");
const SRC_ROT: &str = include_str!("../kernels/llvq_rot.metal");
const SRC_Q4: &str = include_str!("../kernels/tv_q4_h.metal");
const SRC_Q8: &str = include_str!("../kernels/emb_q8.metal");
const SRC_E4: &str = include_str!("../kernels/emb_q4.metal");
/// The row-loop twins of the Tetra matvec: the served unit's text, then the
/// new kernels. Compiled as its own library, so the served pipelines, which
/// compile from `SRC_TETRA` alone, keep their register allocation.
const SRC_TETRA_ROWS: &str =
    concat!(include_str!("../kernels/llvq_tetra48.metal"), include_str!("../kernels/llvq_tetra48_rows.metal"));
/// The prefetching twins of the int4 matvecs, on the same rule.
const SRC_Q4_PF: &str = concat!(include_str!("../kernels/tv_q4_h.metal"), include_str!("../kernels/tv_q4_pf.metal"));

/// Which file carries which entry point.
///
/// Four translation units, not one, for the reason `tetra48_seg.cu` gives on
/// the CUDA side: appending to a shipped unit changes its bytes and can move
/// the register allocation of a kernel no correctness test can see move.
fn source_of(name: &str) -> Result<&'static str> {
    match name {
        "tetra48_probe" | "tv_tetra48_metal" | "tv_tetra48_metal_ar" | "tv_tetra48_metal_arh" | "tv_tetra48_metal_lut" | "tv_tetra48_metal_lutg" | "tv_tetra48_metal_ilp" | "tv_tetra48_metal_u2" | "tv_tetra48_metal_c3" => Ok(SRC_TETRA),
        "rot_apply_metal" | "rot_apply_rows_metal" | "rot_apply_tg_metal" => Ok(SRC_ROT),
        "tv_q4_metal" | "tv_q4_metal_tiled" => Ok(SRC_Q4),
        "tv_tetra48_metal_lutp" | "tv_tetra48_metal_lutd" | "tv_tetra48_metal_lutdp" | "tv_tetra48_metal_lutd2"
        | "tv_tetra48_metal_lutd4" | "tv_tetra48_metal_luth" | "tv_tetra48_metal_lutdh"
        | "tv_tetra48_metal_lutdr" | "tv_tetra48_metal_lutdt" | "tv_tetra48_metal_lutdth" => Ok(SRC_TETRA_ROWS),
        "tv_q4_metal_p" | "tv_q4_metal_tiled_p" | "tv_q4_metal_h" | "tv_q4_metal_tiled_h"
        | "tv_q4_metal_rows" => Ok(SRC_Q4_PF),
        "emb_q8_gather_metal" | "tv_q8_metal" => Ok(SRC_Q8),
        "emb_q4_gather_metal" | "tv_emb_q4_metal" => Ok(SRC_E4),
        "tv_tetra48_metal_lutdrf" => Ok(SRC_TETRA_ROWS_F32),
        "tv_emb_q4_metal_xf" => Ok(SRC_E4_XF),
        "moe_rot_metal" => Ok(SRC_MOE_ROT),
        "moe_tetra_gu_metal" | "moe_tetra_down_metal" | "moe_tetra_down_sg_metal" => Ok(SRC_MOE_TETRA),
        "moe_rows_lutdrf" | "moe_pairs_sum" => Ok(SRC_MOE_ROWS),
        other => candle_core::bail!("no Metal source carries {other}"),
    }
}

/// The pipeline cache, keyed on everything that changes the compiled code.
///
/// The tile is in the key and must stay there: it is injected as a text
/// prelude, and a define that fails to land changes how much threadgroup
/// memory the host must stage while leaving the arithmetic alone. That is the
/// failure a cache keyed on the function name alone produces, silently.
type Key = (&'static str, usize);

/// What the decoder needs, uploaded once and shared by every projection.
///
/// 18,688 bytes: rows 4,096 u32, branches 1,024 u16, prefixes 128 u8,
/// suffixes 128 u8. Held as `Arc<Buffer>` for the reason the header gives:
/// released, they become somebody else's scratch.
pub struct MetalTables {
    rows: Arc<Buffer>,
    prefixes: Arc<Buffer>,
    branches: Arc<Buffer>,
    suffixes: Arc<Buffer>,
    /// `1/sqrt(16 m)`, entry 0 the origin.
    invnorm: Arc<Buffer>,
    /// `suffixes[2 * ((branches[16*s8 + b2] >> 8) & 63) + b3]`, precomposed.
    ///
    /// 64 x 16 x 2 = 2,048 bytes. It holds no new information: it is the two
    /// tables it is built from, composed, so the value is unchanged. What it
    /// removes is the DEPENDENCY, which was the second load waiting on the
    /// first.
    c3_of: Arc<Buffer>,
    /// Host copies, kept only so `cpu_fwd` can serve as the oracle. `None` on
    /// a served model, where the CPU arm refuses instead.
    host: Option<HostTables>,
}

struct HostTables {
    rows: Vec<u32>,
    prefixes: Vec<u8>,
    branches: Vec<u16>,
    suffixes: Vec<u8>,
    invnorm: Vec<f32>,
}

/// One Tetra projection on the device.
pub struct MetalTetraProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    nblocks: usize,
    tail_w: usize,
    row_stride_u32: usize,
    words: Arc<Buffer>,
    gscale: Arc<Buffer>,
    rscale: Arc<Buffer>,
    tail: Arc<Buffer>,
    host: Option<HostProj>,
}

struct HostProj {
    words: Vec<u8>,
    gscale: Vec<f32>,
    rscale: Vec<f32>,
    tail: Vec<u16>,
}

/// The device, the tables and the compiled pipelines.
pub struct MetalRuntime {
    /// The matvec this runtime launches. See `metal_fwd`.
    kernel: &'static str,
    /// Columns a tile for the int4 records past the wall.
    q4_tile_cols: usize,
    /// Whether the int4 matvecs are the prefetching twins.
    q4_prefetch: bool,
    /// Whether the matvecs are the half-io twins ([`MetalTuning::half_io`]).
    half_io: bool,
    /// Prompt rows a Tetra prefill launch ([`MetalTuning::prefill_rows`]).
    prefill_rows: usize,
    /// Whether the int4 records take the prefill chunks ([`MetalTuning::q4_rows`]).
    q4_rows: bool,
    /// Whether one-row rotations that fit stage in threadgroup memory ([`MetalTuning::rot_tg`]).
    rot_tg: bool,
    /// Whether the fused MoE call's `down` runs one simdgroup a slot
    /// (`moe_tetra_down_sg_metal`): the same bits ([`Self::set_moe_down_sg`]).
    moe_down_sg: std::sync::atomic::AtomicBool,
    device: candle_core::MetalDevice,
    tables: MetalTables,
    tile: usize,
    pipelines: RwLock<HashMap<Key, ComputePipeline>>,
}

/// f16 bits widened exactly, which is what the shader's `float(half)` does.
///
/// A transcription of `h2f` in `llvq-cuda/kernels/matvec.cu:44-77`, branch for
/// branch, and it is a transcription because an earlier version here was
/// written from memory and was wrong on 4,094 of the 65,536 patterns: every
/// subnormal had its exponent one too low and its mantissa shifted one too
/// far, and there was no `exp == 31` arm at all, so infinities came back as
/// 65536 and NaN payloads were lost. The gate could not see it, because its
/// fixtures draw from a Gaussian and never produce one.
fn h2f(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mut man = h & 0x3ff;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Normalise: shift until the hidden bit appears, one exponent a shift.
            let mut e = 127 - 15 + 1;
            while man & 0x400 == 0 {
                man <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((man & 0x3ff) << 13)
        }
    } else if exp == 31 {
        // inf / NaN, payload kept.
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// The suffix table composed with the branch table that indexes it.
///
/// `suffixes` is read at `2 * s16 + b3` where `s16` is the high byte of
/// `branches[16 * s8 + b2]`, masked to the 64 Golay states. So the composition
/// is indexed by `(s8, b2, b3)` and holds exactly what the two-step read
/// would have returned.
fn compose_c3(branches: &[u16], suffixes: &[u8]) -> Vec<u8> {
    let mut t = vec![0u8; 64 * 16 * 2];
    for s8 in 0..64usize {
        for b2 in 0..16usize {
            let s16 = ((branches[16 * s8 + b2] >> 8) & 63) as usize;
            for b3 in 0..2usize {
                t[((s8 * 16 + b2) << 1) | b3] = suffixes[2 * s16 + b3];
            }
        }
    }
    t
}


/// The matvec named by `LLVQ_METAL_KERNEL`, or the served one.
fn kernel_from_env() -> Result<&'static str> {
    kernel_named(std::env::var("LLVQ_METAL_KERNEL").ok().as_deref())
}

/// The matvec a value of `LLVQ_METAL_KERNEL` names. An unknown name is refused:
/// falling back to the served kernel would label an A/B arm with a kernel it
/// did not run.
///
/// The names are a subset of what [`MetalTuning::validate`] accepts. `ilp` is
/// not one: `tv_tetra48_metal_ilp` gives a row 8 lanes, so on the 32-lane grid
/// every launch uses it would write past the output.
fn kernel_named(v: Option<&str>) -> Result<&'static str> {
    const SERVED: &str = "tv_tetra48_metal_lut";
    Ok(match v {
        None | Some("lut") => SERVED,
        Some("c3") => "tv_tetra48_metal_c3",
        Some("ar") => "tv_tetra48_metal_ar",
        Some("u2") => "tv_tetra48_metal_u2",
        Some("plain") => "tv_tetra48_metal",
        Some(other) => candle_core::bail!(
            "LLVQ_METAL_KERNEL={other:?} is not a matvec of this file: lut, c3, ar, u2 or plain"
        ),
    })
}

impl MetalRuntime {
    /// Upload the decoder tables and take the device.
    ///
    /// `keep_host` keeps a host copy of everything, which is what lets
    /// [`TetraMatvec::cpu_fwd`] act as the oracle a test compares against. A
    /// served model passes `false` and the CPU arm then refuses by name
    /// instead of silently costing a model its memory twice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &candle_core::Device,
        rows: &[u32],
        prefixes: &[u8],
        branches: &[u16],
        suffixes: &[u8],
        invnorm: &[f32],
        tile: usize,
        keep_host: bool,
    ) -> Result<Self> {
        let dev = match device {
            candle_core::Device::Metal(d) => d.clone(),
            other => candle_core::bail!("the Metal runtime wants a Metal device, got {other:?}"),
        };
        check_threadgroup_memory(&dev)?;
        // 32..=256, not the CUDA contract's 32..=512. The staging is
        // `tile * 24 * 4` bytes against Apple's 32,768 B of threadgroup
        // memory, so 341 blocks is the ceiling and 256 the largest power of
        // two under it. 512 would ask for 49,152 B, pass a validator copied
        // from the wrong platform, and abort the process at dispatch.
        if !tile.is_power_of_two() || !(32..=256).contains(&tile) {
            candle_core::bail!(
                "tile {tile} is not a power of two in 32..=256. Apple stages \
                 {} B for it against a limit of 32768",
                tile * llvq_core::DIM * 4
            );
        }
        let tables = MetalTables {
            rows: dev.new_buffer_with_data(rows)?,
            prefixes: dev.new_buffer_with_data(prefixes)?,
            branches: dev.new_buffer_with_data(branches)?,
            suffixes: dev.new_buffer_with_data(suffixes)?,
            invnorm: dev.new_buffer_with_data(invnorm)?,
            c3_of: dev.new_buffer_with_data(&compose_c3(branches, suffixes))?,
            host: keep_host.then(|| HostTables {
                rows: rows.to_vec(),
                prefixes: prefixes.to_vec(),
                branches: branches.to_vec(),
                suffixes: suffixes.to_vec(),
                invnorm: invnorm.to_vec(),
            }),
        };
        Ok(Self {
            kernel: kernel_from_env()?,
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
            moe_down_sg: std::sync::atomic::AtomicBool::new(MOE_DOWN_SG_DEFAULT),
            device: dev,
            tables,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Upload one projection's stream and its scales.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u8],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}. The kernel \
                 carries no row guard, so a partial threadgroup would compute and STORE \
                 rows past the output"
            );
        }
        let tail_w = d_in - nblocks * llvq_core::DIM;
        if rscale.len() != d_out {
            candle_core::bail!("{name}: {} row scales for {d_out} rows", rscale.len());
        }
        if tail.len() != d_out * tail_w {
            candle_core::bail!("{name}: {} tail values for {d_out}x{tail_w}", tail.len());
        }
        // The stream, which the kernel indexes as `words + row * row_stride_u32`
        // with no bound of its own. `rscale` and `tail` were checked and this
        // was not, which is the asymmetry an audit of 2026-09-21 named.
        let want_stride = llvq_artifact::tetra48::stride_u32(nblocks);
        if row_stride_u32 != want_stride {
            candle_core::bail!(
                "{name}: row stride {row_stride_u32} for {nblocks} blocks, which wants \
                 {want_stride}"
            );
        }
        // The stream is `d_out * row_stride_u32` words, plus AT MOST one guard
        // word. `f1r_load` reads `row[w]` and `row[w + 1]`, so the last block
        // of the last row reaches one word past the table; the writer pads for
        // it. An equality here refused the served 4B by four bytes.
        let want_bytes = d_out * row_stride_u32 * 4;
        if words.len() != want_bytes && words.len() != want_bytes + 4 {
            candle_core::bail!(
                "{name}: {} stream bytes for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_bytes} or {want_bytes} plus one guard word",
                words.len()
            );
        }
        // Metal refuses a zero-length buffer and hands back a null pointer,
        // the same wall cudarc puts up. A `d_in` that is a multiple of 24 has
        // no tail, so it gets a one-element dummy the kernel never reads:
        // `tail_w == 0` makes the epilogue's loop empty.
        let dummy = [0u16];
        let tail_src = if tail.is_empty() { &dummy[..] } else { tail };
        Ok(MetalTetraProj {
            name: name.to_string(),
            d_out,
            d_in,
            nblocks,
            tail_w,
            row_stride_u32,
            words: self.device.new_buffer_with_data(words)?,
            gscale: self.device.new_buffer_with_data(gscale)?,
            rscale: self.device.new_buffer_with_data(rscale)?,
            tail: self.device.new_buffer_with_data(tail_src)?,
            host: self.tables.host.is_some().then(|| HostProj {
                words: words.to_vec(),
                gscale: gscale.to_vec(),
                rscale: rscale.to_vec(),
                tail: tail.to_vec(),
            }),
        })
    }

    /// The pipeline for `name` at this runtime's tile, compiled once.
    fn pipeline(&self, name: &'static str) -> Result<ComputePipeline> {
        let key: Key = (name, self.tile);
        if let Some(p) = self.pipelines.read().expect("no panic held this lock").get(&key) {
            return Ok(p.clone());
        }
        let src = format!("#define LLVQ_TILE_BLOCKS {}u\n{}", self.tile, source_of(name)?);
        let dev = self.device.device().clone();
        // A bad source PANICS rather than returning Err: both
        // `new_library_with_source` and `new_compute_pipeline_state_with_function`
        // end in `.unwrap()`. The panic payload carries Metal's own diagnostics,
        // so it is re-raised as the message instead of being swallowed.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lib = dev.new_library_with_source(&src, None)?;
            let func = lib.get_function(name, None)?;
            dev.new_compute_pipeline_state_with_function(&func)
        }));
        let pipe = match built {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => candle_core::bail!("{name}: Metal refused the source: {e:?}"),
            Err(panic) => {
                let what = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "no diagnostics".into());
                candle_core::bail!("{name}: the MSL did not compile: {what}");
            }
        };
        // The matvec and head kernels give a row one SIMD-group of 32 lanes,
        // find it as `gid >> 5`, reduce across lanes `l ^ 16 .. l ^ 1` and
        // write from lane 0. A pipeline that runs another width would return
        // wrong rows with no error. Checked on every pipeline, before it is
        // cached: a device whose width is not 32 serves none of the path.
        {
            use objc2_metal::MTLComputePipelineState as _;
            let width = pipe.as_ref().threadExecutionWidth();
            if width != LANES {
                candle_core::bail!(
                    "{name}: the pipeline runs SIMD-groups of {width} threads, and the kernels of this \
                     file assume {LANES}"
                );
            }
        }
        self.pipelines
            .write()
            .expect("no panic held this lock")
            .insert(key, pipe.clone());
        Ok(pipe)
    }

    /// `y = W x` for one activation row, through the kernel.
    pub fn matvec(&self, proj: &MetalTetraProj, x: &Tensor) -> Result<Tensor> {
        let op = TetraMatvec { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }

    /// Prompt rows a Tetra prefill launch takes; 1 is no rows kernel.
    pub fn prefill_rows(&self) -> usize {
        self.prefill_rows
    }

    /// `Y = W X` for `n_rows` rotated rows `[n_rows, d_in]` in f32, one launch
    /// of `tv_tetra48_metal_lutdr`, as f16 `[n_rows, d_out]`. Row r is the
    /// matvec of row r narrowed to f16, bit for bit.
    pub fn matvec_rows(&self, proj: &MetalTetraProj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill_rows.max(1) || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for a prefill launch of at most {} (kernel bound {TETRA_PREFILL_MAX})",
                proj.name,
                self.prefill_rows
            );
        }
        x.apply_op1_no_bwd(&TetraRows { rt: self, proj, n_rows })
    }
}

/// One launch of `tv_tetra48_metal_lutdr`, as a candle op.
struct TetraRows<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
    n_rows: usize,
}

impl CustomOp1 for TetraRows<'_> {
    fn name(&self) -> &'static str {
        "tetra48-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "{}: the Tetra prefill kernel runs on Metal only; its reference is the matvec, row by row",
            self.proj.name
        )
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: prefill rows in f32 only, got {:?}", p.name, storage.dtype());
        }
        if !layout.is_contiguous() {
            candle_core::bail!("{}: prefill rows must be contiguous", p.name);
        }
        if layout.dims() != [self.n_rows, p.d_in] {
            candle_core::bail!(
                "{}: prefill activation {:?} for [{}, {}]",
                p.name,
                layout.dims(),
                self.n_rows,
                p.d_in
            );
        }
        if !p.d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("{}: d_out {} is not a multiple of {} rows", p.name, p.d_out, GROUP / LANES);
        }
        let dev = storage.device().clone();
        let pipe = self.rt.pipeline("tv_tetra48_metal_lutdr")?;
        let group = group_within(&pipe, "tv_tetra48_metal_lutdr", GROUP)?;
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F16, "tetra48-rows")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, p.row_stride_u32 as u32);
        set_param(&enc, 2, (&*self.rt.tables.rows, 0usize));
        set_param(&enc, 3, (&*self.rt.tables.prefixes, 0usize));
        set_param(&enc, 4, (&*self.rt.tables.branches, 0usize));
        set_param(&enc, 5, (&*self.rt.tables.suffixes, 0usize));
        set_param(&enc, 6, (&*p.gscale, 0usize));
        set_param(&enc, 7, (&*self.rt.tables.invnorm, 0usize));
        set_param(&enc, 8, (&*p.rscale, 0usize));
        set_param(&enc, 9, (&*p.tail, 0usize));
        set_param(&enc, 10, (storage.buffer(), x_off));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, p.nblocks as u32);
        set_param(&enc, 13, p.tail_w as u32);
        set_param(&enc, 15, self.n_rows as u32);
        set_param(&enc, 16, p.d_out as u32);
        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F16),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

/// One launch of `tv_tetra48_metal`, as a candle op.
struct TetraMatvec<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
}

impl TetraMatvec<'_> {
    /// What both arms owe the caller before either computes anything.
    ///
    /// Added after an audit of 2026-09-21 found `metal_fwd` checking dtype and
    /// contiguity and nothing else, while `cpu_fwd` checked only the dtype.
    /// The CUDA twin refuses all of this by name at `fused_cuda.rs:1491`, with
    /// a comment saying the missing check already cost one run.
    fn check(&self, dtype: DType, layout: &Layout) -> Result<()> {
        let p = self.proj;
        if dtype != DType::F32 {
            candle_core::bail!("{}: f32 only, got {dtype:?}", p.name);
        }
        if !layout.is_contiguous() {
            candle_core::bail!(
                "{}: the activation must be contiguous. The kernel indexes it linearly \
                 from one base and cannot honour a stride",
                p.name
            );
        }
        let dims = layout.dims();
        let d_in = *dims.last().unwrap_or(&0);
        if d_in != p.d_in {
            candle_core::bail!(
                "{}: activation of {d_in} values for d_in={}. A short one reads past the \
                 buffer and returns finite, plausible, wrong numbers",
                p.name,
                p.d_in
            );
        }
        // One vector a launch, like the CUDA twin. The output is allocated at
        // `d_out` elements, so accepting several rows would hand back a tensor
        // whose shape claims more than its storage holds.
        let rows: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: {rows} vectors at once. This kernel is a matvec and takes one; the \
                 row loop belongs to the caller",
                p.name
            );
        }
        Ok(())
    }
}

impl CustomOp1 for TetraMatvec<'_> {
    fn name(&self) -> &'static str {
        "tetra48-matvec"
    }

    /// The oracle, not a stub.
    ///
    /// `CustomOp1` requires a CPU arm, and making it the reference rather than
    /// a `bail!` is what lets a test drive the SAME op on both devices and
    /// demand equality. It reproduces the kernel's summation order exactly:
    /// 32 lanes striding the blocks, the shuffle-xor butterfly, and the
    /// multiply-then-add tail. A plain loop would need a tolerance, and a
    /// tolerance is the size of a real defect here.
    ///
    /// Refuses when the runtime was built without host copies, which is the
    /// served case: a model that kept them would pay for its weights twice.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (ht, hp) = match (&self.rt.tables.host, &self.proj.host) {
            (Some(t), Some(p)) => (t, p),
            _ => candle_core::bail!(
                "{}: this runtime was built without host copies, so its CPU arm is not \
                 available. It is the oracle of a test, never a fallback for a served model",
                self.proj.name
            ),
        };
        self.check(storage.dtype(), layout)?;
        let x = match storage {
            CpuStorage::F32(v) => &v[layout.start_offset()..],
            other => candle_core::bail!("{}: f32 only, got {:?}", self.proj.name, other.dtype()),
        };
        let p = self.proj;
        let tile = self.rt.tile;
        let mut y = vec![0f32; p.d_out];
        let ntiles = p.nblocks.div_ceil(tile);

        for (row, out) in y.iter_mut().enumerate() {
            let mut lanes = [0f32; LANES];
            for t in 0..ntiles {
                let jlo = t * tile;
                let jhi = (jlo + tile).min(p.nblocks);
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut j = jlo + lane;
                    while j < jhi {
                        *acc += host_block(ht, hp, p, row, j, x);
                        j += LANES;
                    }
                }
            }
            for k in [16usize, 8, 4, 2, 1] {
                let mut next = [0f32; LANES];
                for (l, n) in next.iter_mut().enumerate() {
                    *n = lanes[l] + lanes[l ^ k];
                }
                lanes = next;
            }
            let mut tv = 0f32;
            let xt = &x[p.nblocks * llvq_core::DIM..];
            // `mul_add` at both sites: the CUDA twin contracts there under
            // NVRTC's `--fmad=true`, and the shader now spells it out.
            for (i, xi) in xt.iter().enumerate().take(p.tail_w) {
                tv = h2f(hp.tail[row * p.tail_w + i]).mul_add(*xi, tv);
            }
            *out = lanes[0].mul_add(hp.rscale[row], tv);
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((CpuStorage::F32(y), Shape::from(dims)))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.check(storage.dtype(), layout)?;
        let p = self.proj;
        let dev = storage.device().clone();
        // The ARITHMETIC variant, not the faithful port.
        //
        // `tv_tetra48_metal` mirrors the CUDA v3 decoder, which gathers its
        // coordinates with `prmt.b32`, one instruction. Metal has no PRMT, so
        // that gather costs a four-trip emulation called twelve times a block.
        // `tv_tetra48_metal_ar` computes the value instead, which is what the
        // SCALAR CUDA decoder always did, and measures 25 to 34 percent faster.
        // `_lut` then replaces the ten-operation computation with ONE indexed
        // load from 64 constant floats, and takes a further 9 to 16 percent. `_lutg`
        // moves those 64 floats from `constant` to `threadgroup`, worth 9 to
        // 18 more: `constant` is optimised for UNIFORM access and this index
        // varies per lane.
        // Cumulative against the faithful port: 19.85 ms a token to 12.66
        // (`llvq-metal/examples/metalsplit.rs`).
        //
        // Both are proven equal to the same host reference. The faithful port
        // is kept as the reference implementation and as the thing the
        // arithmetic one is diffed against.
        // `LLVQ_METAL_KERNEL` picks the matvec, for A/B only.
        //
        // A MEASUREMENT MODE, never a served setting. It exists because this
        // machine drifts: the same binary measured 49.7 tok/s in the morning
        // and 23.1 after three hours of compiling, so two numbers taken an
        // hour apart compare the thermals and not the kernels. Interleaving
        // two kernels in ONE process is the only honest A/B here.
        //
        // Unset, the served kernel is `_lut` and nothing changes.
        // The half-io twin stores f16 where the served kernel stores f32; the
        // adapter's narrowing cast then has nothing to do.
        let (kname, out_dtype) = match self.rt.half_io {
            false => (self.rt.kernel, DType::F32),
            true => (
                tetra_half_twin(self.rt.kernel).ok_or_else(|| {
                    candle_core::Error::msg(format!("half-io: {} has no half-store twin", self.rt.kernel))
                })?,
                DType::F16,
            ),
        };
        let pipe = self.rt.pipeline(kname)?;
        // The one-launch kernels' group; the row-looping ones size their own
        // below.
        let group = group_within(&pipe, kname, GROUP)?;

        let out = dev.new_buffer(p.d_out, out_dtype, "tetra48-matvec")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);

        // `start_offset()` is in ELEMENTS; the binding wants bytes.
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        let stride = p.row_stride_u32 as u32;
        let nb = p.nblocks as u32;
        let tw = p.tail_w as u32;
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, stride);
        set_param(&enc, 2, (&*self.rt.tables.rows, 0usize));
        set_param(&enc, 3, (&*self.rt.tables.prefixes, 0usize));
        set_param(&enc, 4, (&*self.rt.tables.branches, 0usize));
        set_param(&enc, 5, (&*self.rt.tables.suffixes, 0usize));
        set_param(&enc, 6, (&*p.gscale, 0usize));
        set_param(&enc, 7, (&*self.rt.tables.invnorm, 0usize));
        set_param(&enc, 8, (&*p.rscale, 0usize));
        set_param(&enc, 9, (&*p.tail, 0usize));
        set_param(&enc, 10, (storage.buffer(), x_off));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, nb);
        set_param(&enc, 13, tw);
        set_param(&enc, 14, (&*self.rt.tables.c3_of, 0usize));

        // Not optional, and not an error if omitted: the kernel would run to
        // completion and write zeros. The tile in the length and the tile in
        // the source are the same number by construction, because both come
        // from `self.rt.tile`.
        if !tetra_reads_direct(kname) {
            let staged = self.rt.tile * llvq_core::DIM * DType::F32.size_in_bytes();
            enc.set_threadgroup_memory_length(0, staged);
        }

        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*self.rt.tables.c3_of,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);

        // The table-staging kernels: as many simdgroups a threadgroup as the
        // pipeline allows up to 32, each looping over rows with a guard, and
        // enough threadgroups for about TETRA_TG_LOOP_ROWS rows a simdgroup.
        if tetra_loops_rows(kname) {
            set_param(&enc, 16, p.d_out as u32);
            for (i, len) in TETRA_TG_TABLES.iter().enumerate() {
                enc.set_threadgroup_memory_length(i, *len);
            }
            let threads = (pipe.max_total_threads_per_threadgroup().min(1024) / LANES) * LANES;
            if threads == 0 {
                candle_core::bail!("{kname}: the pipeline allows fewer than {LANES} threads a threadgroup");
            }
            let nsg = threads / LANES;
            let ntg = p.d_out.div_ceil(nsg * TETRA_TG_LOOP_ROWS).max(1);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize { width: ntg, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
            );
            let mut dims = layout.dims().to_vec();
            *dims.last_mut().expect("rank >= 1") = p.d_out;
            return Ok((MetalStorage::new(out, dev, p.d_out, out_dtype), Shape::from(dims)));
        }

        // A multi-row twin covers R rows a simdgroup and carries no row guard,
        // like the served kernel: a d_out that eight simdgroups of R rows do
        // not tile would compute and store rows past the output.
        let rps = tetra_rows_per_simd(self.rt.kernel);
        if !p.d_out.is_multiple_of(rps * GROUP / LANES) {
            candle_core::bail!(
                "{}: d_out {} is not a multiple of {} rows, which {} covers a threadgroup",
                p.name,
                p.d_out,
                rps * GROUP / LANES,
                self.rt.kernel
            );
        }
        let grid = objc2_metal::MTLSize {
            width: p.d_out / rps * LANES,
            height: 1,
            depth: 1,
        };
        let group = objc2_metal::MTLSize {
            width: group,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, group);

        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((
            MetalStorage::new(out, dev, p.d_out, out_dtype),
            Shape::from(dims),
        ))
    }
}

/// One block's contribution, in the shader's order.
fn host_block(
    ht: &HostTables,
    hp: &HostProj,
    p: &MetalTetraProj,
    row: usize,
    j: usize,
    x: &[f32],
) -> f32 {
    let (lo, hi16) = host_load(&hp.words, p.row_stride_u32, row, j);
    let q = host_quads(ht, lo, hi16);
    let xb = &x[j * llvq_core::DIM..(j + 1) * llvq_core::DIM];
    let mut d = 0f32;
    for (i, quad) in q.iter().enumerate() {
        for k in 0..4 {
            let idx = ORDER[4 * i + k];
            let v = ((quad >> (8 * k)) & 0xff) as f32 - 128.0;
            d = v.mul_add(xb[idx], d);
        }
    }
    let n2: i32 = q
        .iter()
        .flat_map(|quad| (0..4).map(move |k| (((quad >> (8 * k)) & 0xff) as i32) - 128))
        .map(|v| v * v)
        .sum();
    let m = (((n2 as u32) >> 4) & 31) as usize;
    let g = ((hi16 >> 15) & 1) as usize;
    d * hp.gscale[g] * ht.invnorm[m]
}

/// `f1r_load`, on the host.
fn host_load(words: &[u8], row_stride_u32: usize, row: usize, j: usize) -> (u32, u32) {
    let base = row * row_stride_u32;
    let w = (3 * j) >> 1;
    let at = |i: usize| -> u32 {
        let o = (base + i) * 4;
        u32::from_le_bytes([words[o], words[o + 1], words[o + 2], words[o + 3]])
    };
    let w0 = at(w);
    let w1 = at(w + 1);
    match j & 1 {
        1 => ((w0 >> 16) | (w1 << 16), w1 >> 16),
        _ => (w0, w1 & 0xffff),
    }
}

/// CUDA's PRMT, default mode. See the shader's `prmt`.
fn prmt(a: u32, b: u32, s: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4 {
        let sel = (s >> (4 * i)) & 0xf;
        let idx = sel & 7;
        let src = if idx < 4 { a } else { b };
        let byte = (src >> (8 * (idx & 3))) & 0xff;
        let v = if sel & 8 != 0 {
            if byte & 0x80 != 0 {
                0xff
            } else {
                0
            }
        } else {
            byte
        };
        out |= v << (8 * i);
    }
    out
}

/// `f1r_v3_quads`, on the host.
fn host_quads(t: &HostTables, lo: u32, hi16: u32) -> [u32; 6] {
    const T: [[u32; 2]; 4] = [
        [0x887c8480, 0x90748c78],
        [0x79857d81, 0x718d7589],
        [0x7a867e82, 0x728e768a],
        [0x877b837f, 0x8f738b77],
    ];
    let p = lo & 1;
    let r = (lo >> 1) & 1;
    let s8 = (lo >> 2) & 63;
    let b1 = (lo >> 8) & 1;
    let i1 = (lo >> 9) & 0x7ff;
    let b2 = (lo >> 20) & 15;
    let i2 = ((lo >> 24) | ((hi16 & 7) << 8)) & 0x7ff;
    let b3 = (hi16 >> 3) & 1;
    let i3 = (hi16 >> 4) & 0x7ff;

    let c1 = t.prefixes[(2 * s8 + b1) as usize] as u32;
    let br = t.branches[(16 * s8 + b2) as usize] as u32;
    let c2 = br & 0xff;
    let s16 = (br >> 8) & 63;
    let c3 = t.suffixes[(2 * s16 + b3) as usize] as u32;

    let row1 = t.rows[(2048 * r + i1) as usize];
    let mid = i2 < 1240;
    let idx2 = if mid { i2 } else { (2048 - 1240) + i2 };
    let row2 = t.rows[idx2 as usize];
    let delta = u32::from(!mid);
    let r3 = (p ^ r ^ delta) & 1;
    let row3 = t.rows[(2048 * r3 + i3) as usize];

    let (c0lo, c0hi) = if p == 1 { (T[1][0], T[1][1]) } else { (T[0][0], T[0][1]) };
    let (c1lo, c1hi) = if p == 1 { (T[3][0], T[3][1]) } else { (T[2][0], T[2][1]) };
    let quad = |sel: u32, c4: u32| -> u32 {
        let m0 = prmt(c0lo, c0hi, sel);
        let m1 = prmt(c1lo, c1hi, sel);
        let k = ((c4.wrapping_mul(0x0020_4081)) & 0x0101_0101).wrapping_mul(0xff);
        (m0 & !k) | (m1 & k)
    };
    [
        quad(row1, c1 & 0xf),
        quad(row1 >> 16, c1 >> 4),
        quad(row2, c2 & 0xf),
        quad(row2 >> 16, c2 >> 4),
        quad(row3, c3 & 0xf),
        quad(row3 >> 16, c3 >> 4),
    ]
}



// ---------------------------------------------------------------------------
// The rotation.
//
// `prepare` owes this: the weights were quantized in a rotated basis, so the
// activation has to arrive in it. The kernel is a Walsh-Hadamard transform
// with a per-coordinate sign flip, then either a scale-out or a small mix.
//
// One thing differs from CUDA and it is not cosmetic. There the staging is a
// `__shared__` array, per block and free. Here it needs `n` floats a row, and
// `n` reaches 9,728 on the served 4B, which is 38,912 B against Apple's 32,768
// of threadgroup memory. So the scratch is a DEVICE buffer and every stage of
// the transform goes through global memory. Nothing about that cost is
// measured; it is the first thing to look at if the Metal throughput
// disappoints.
// ---------------------------------------------------------------------------

/// One rotation, uploaded. Shared by every projection that names its key.
pub struct MetalRotation {
    signbits: Arc<Buffer>,
    small: Arc<Buffer>,
    n: u32,
    m: u32,
    k: u32,
    inv: f32,
    /// Threads a threadgroup, the same rule the CUDA host uses.
    threads: usize,
}

impl MetalRuntime {
    /// Upload one rotation's tables.
    pub fn upload_rotation(&self, t: &crate::fused::RotationTables) -> Result<MetalRotation> {
        // Metal refuses a zero-length buffer. `small` is empty when k == 1,
        // which is the common case: `rot_mix` is not reached then and the
        // kernel never reads it.
        let dummy = [0f32];
        let small = if t.small.is_empty() { &dummy[..] } else { &t.small };
        Ok(MetalRotation {
            signbits: self.device.new_buffer_with_data(&t.signbits)?,
            small: self.device.new_buffer_with_data(small)?,
            n: t.n as u32,
            m: t.m as u32,
            k: t.k as u32,
            inv: t.inv,
            threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
        })
    }

    /// `x` into the basis the weights were quantized in, one vector.
    pub fn rotate(&self, rot: &MetalRotation, name: &str, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{name}: rotation requested for {rows} vectors. The row loop belongs to \
                 the caller, which shares it across a group"
            );
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: 1 })
    }

    /// The same for a chunk of rows, one threadgroup each.
    pub fn rotate_rows(&self, rot: &MetalRotation, name: &str, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{name}: {rows} rows of activation for {n_rows}");
        }
        if n_rows == 0 {
            candle_core::bail!("{name}: a rotation of zero rows");
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: n_rows })
    }
}

struct RotOp<'a> {
    rt: &'a MetalRuntime,
    rot: &'a MetalRotation,
    name: String,
    rows: usize,
}

impl CustomOp1 for RotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-rot"
    }

    /// No CPU path, and deliberately none.
    ///
    /// Unlike the matvec, this op has no oracle to be: the rotation is judged
    /// by `llvq-metal/tests/rot_matches_host.rs` against the CUDA text itself,
    /// executed. A second Rust transcription here would be a third place for
    /// the same mistake to live.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the LLVQ rotation has no CPU path", self.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the rotation reads f16, got {:?}", self.name, storage.dtype());
        }
        let n = self.rot.n as usize;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", self.name)))?;
        if end - start != n * self.rows {
            candle_core::bail!(
                "{}: activation of {} values for {} rows of n={n}",
                self.name,
                end - start,
                self.rows
            );
        }
        let batched = self.rows > 1;
        let dev = storage.device().clone();
        let out = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot")?;
        use candle_metal_kernels::utils::set_param;

        // One row that fits: the vector staged in threadgroup memory, no
        // scratch, the same stages.
        let staged = n * DType::F32.size_in_bytes();
        if self.rt.rot_tg && !batched && staged <= THREADGROUP_LIMIT {
            let pipe = self.rt.pipeline("rot_apply_tg_metal")?;
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (storage.buffer(), 0usize));
            set_param(&enc, 1, (&*self.rot.signbits, 0usize));
            set_param(&enc, 2, (&*self.rot.small, 0usize));
            set_param(&enc, 3, (&*out, 0usize));
            set_param(&enc, 5, self.rot.n);
            set_param(&enc, 6, self.rot.m);
            set_param(&enc, 7, self.rot.k);
            set_param(&enc, 8, self.rot.inv);
            set_param(&enc, 9, start as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [storage.buffer(), &*self.rot.signbits, &*self.rot.small] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            // One threadgroup, never larger than the pipeline takes. The
            // kernel strides every loop by its thread count, so any count
            // gives the same bits.
            let threads = (self.rot.threads.min(pipe.max_total_threads_per_threadgroup()) / LANES) * LANES;
            if threads == 0 {
                candle_core::bail!("rot_apply_tg_metal: the pipeline allows fewer than {LANES} threads a threadgroup");
            }
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
            );
            return Ok((MetalStorage::new(out, dev, n, DType::F32), Shape::from(vec![1, n])));
        }

        let kname = if batched { "rot_apply_rows_metal" } else { "rot_apply_metal" };
        let pipe = self.rt.pipeline(kname)?;
        // The staging, in DEVICE memory. See the section header.
        let scratch = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot-scratch")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        // `x_off` is in ELEMENTS here, not bytes: the kernel indexes
        // `xin[x_off + i]` on a ushort pointer. The Tetra matvec takes a byte
        // offset instead, because it binds at the offset. Two kernels, two
        // conventions, each matched to its own source.
        set_param(&enc, 0, (storage.buffer(), 0usize));
        set_param(&enc, 1, (&*self.rot.signbits, 0usize));
        set_param(&enc, 2, (&*self.rot.small, 0usize));
        set_param(&enc, 3, (&*out, 0usize));
        set_param(&enc, 4, (&*scratch, 0usize));
        set_param(&enc, 5, self.rot.n);
        set_param(&enc, 6, self.rot.m);
        set_param(&enc, 7, self.rot.k);
        set_param(&enc, 8, self.rot.inv);
        set_param(&enc, 9, start as u32);
        if batched {
            set_param(&enc, 10, n as u32);
        }
        for b in [storage.buffer(), &*self.rot.signbits, &*self.rot.small] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        // One threadgroup a row: a group larger than the pipeline takes would
        // split a row over two groups on one scratch. The kernel strides every
        // loop by its thread count, so any count gives the same bits.
        let threads = (self.rot.threads.min(pipe.max_total_threads_per_threadgroup()) / LANES) * LANES;
        if threads == 0 {
            candle_core::bail!("{kname}: the pipeline allows fewer than {LANES} threads a threadgroup");
        }
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * self.rows, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.rows * n, DType::F32),
            Shape::from(vec![self.rows, n]),
        ))
    }
}

// ---------------------------------------------------------------------------
// The int4 arm: `v_proj` on the served mixed file, 36 of the 252 projections.
// ---------------------------------------------------------------------------

/// One projection served as affine int4 g128.
pub struct MetalInt4Proj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    gpr: usize,
    /// `None`: `tv_q4_metal` stages the whole activation.
    /// `Some(c)`: `tv_q4_metal_tiled` stages `c` columns a tile.
    tile_cols: Option<usize>,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalInt4Proj {
    /// The tile this record was routed with: `None` for the whole activation.
    pub fn tile_cols(&self) -> Option<usize> {
        self.tile_cols
    }
}

impl MetalRuntime {
    /// Upload one int4 projection, at this runtime's tile if it is past the
    /// wall.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_int4(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<MetalInt4Proj> {
        self.upload_int4_tiled(name, d_out, d_in, gpr, wq, scales, biases, self.q4_tile_cols)
    }

    /// [`Self::upload_int4`] at an explicit tile, which only a record past
    /// the wall uses. The primitive the binding test drives.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_int4_tiled(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
        tile: usize,
    ) -> Result<MetalInt4Proj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}, and the kernel \
                 carries no row guard"
            );
        }
        if !d_in.is_multiple_of(8) {
            candle_core::bail!("{name}: d_in {d_in} is not a multiple of 8, so a row would \
                                start at a shifted nibble");
        }
        // The whole activation when it fits, `tile` columns at a time past
        // the wall (`tv_q4_metal_tiled`). The tile is checked whether or not
        // this record needs it, so a bad one is refused at the first record.
        MetalTuning { q4_tile_cols: tile, ..Default::default() }.validate()?;
        let tile_cols = int4_tile_for(name, d_in, gpr, tile)?;
        if wq.len() != d_out * d_in / 8 {
            candle_core::bail!("{name}: {} words for {d_out}x{d_in} int4", wq.len());
        }
        if scales.len() != d_out * gpr || biases.len() != d_out * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {d_out}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(MetalInt4Proj {
            name: name.to_string(),
            d_out,
            d_in,
            gpr,
            tile_cols,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        })
    }

    /// `y = W x`, one vector, through `tv_q4_metal`.
    pub fn matvec_int4(&self, proj: &MetalInt4Proj, x: &Tensor) -> Result<Tensor> {
        // Widened here, as `FusedRuntime::forward_int4` does. This arm reads
        // the CALLER's activation, which arrives in the model's f16, while the
        // kernel stages f32. `contiguous` because a narrowed row is a view.
        //
        // The half-io twins read the f16 activation as it arrives and widen
        // it while staging, which is the same exact widening.
        let x = match (self.half_io, x.dtype()) {
            (true, DType::F16) => x.contiguous()?,
            _ => x.to_dtype(DType::F32)?.contiguous()?,
        };
        x.apply_op1_no_bwd(&Int4Op { rt: self, proj })
    }

    /// `Y = W X` for `n_rows` rows `[n_rows, d_in]`, one launch of
    /// `tv_q4_metal_rows`, as f16 `[n_rows, d_out]`. Row r is the int4 matvec
    /// of row r narrowed to f16, bit for bit. Widened to f32 first, as the
    /// served path widens: exact from f16.
    pub fn matvec_int4_rows(&self, proj: &MetalInt4Proj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if !self.q4_rows || n_rows == 0 || n_rows > self.prefill_rows || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for an int4 prefill launch of at most {} (q4 rows {})",
                proj.name,
                self.prefill_rows,
                self.q4_rows
            );
        }
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        x.apply_op1_no_bwd(&Int4Rows { rt: self, proj, n_rows })
    }
}

/// One launch of `tv_q4_metal_rows`, as a candle op.
struct Int4Rows<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalInt4Proj,
    n_rows: usize,
}

impl CustomOp1 for Int4Rows<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 prefill kernel has no CPU path", self.proj.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: int4 prefill rows in f32 only, got {:?}", p.name, storage.dtype());
        }
        if !layout.is_contiguous() || layout.dims() != [self.n_rows, p.d_in] {
            candle_core::bail!(
                "{}: int4 prefill activation {:?} for a contiguous [{}, {}]",
                p.name,
                layout.dims(),
                self.n_rows,
                p.d_in
            );
        }
        let pipe = self.rt.pipeline("tv_q4_metal_rows")?;
        let group = group_within(&pipe, "tv_q4_metal_rows", GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F16, "llvq-metal-q4-rows")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.wq, 0usize));
        set_param(&enc, 1, (&*p.scales, 0usize));
        set_param(&enc, 2, (&*p.biases, 0usize));
        set_param(&enc, 3, (storage.buffer(), layout.start_offset() * DType::F32.size_in_bytes()));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, p.d_in as u32);
        set_param(&enc, 6, p.gpr as u32);
        set_param(&enc, 7, self.n_rows as u32);
        set_param(&enc, 8, p.d_out as u32);
        for b in [&*p.wq, &*p.scales, &*p.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F16),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

struct Int4Op<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalInt4Proj,
}

impl CustomOp1 for Int4Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 kernel has no CPU path", self.proj.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        let half = match (storage.dtype(), self.rt.half_io) {
            (DType::F32, _) => false,
            (DType::F16, true) => true,
            (other, _) => candle_core::bail!(
                "{}: f32, or f16 on the half-io twins, got {other:?} with half-io {}",
                p.name,
                self.rt.half_io
            ),
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", p.name)))?;
        if end - start != p.d_in {
            candle_core::bail!("{}: activation of {} values for d_in={}", p.name, end - start, p.d_in);
        }
        let (kname, staged_cols) = match (p.tile_cols, self.rt.q4_prefetch, half) {
            (None, false, false) => ("tv_q4_metal", p.d_in),
            (None, true, false) => ("tv_q4_metal_p", p.d_in),
            (Some(c), false, false) => ("tv_q4_metal_tiled", c),
            (Some(c), true, false) => ("tv_q4_metal_tiled_p", c),
            (None, false, true) => ("tv_q4_metal_h", p.d_in),
            (Some(c), false, true) => ("tv_q4_metal_tiled_h", c),
            (_, true, true) => candle_core::bail!("{}: half-io and prefetch are two arms", p.name),
        };
        let io = if half { DType::F16 } else { DType::F32 };
        let pipe = self.rt.pipeline(kname)?;
        let group = group_within(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(p.d_out, io, "llvq-metal-q4")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.wq, 0usize));
        set_param(&enc, 1, (&*p.scales, 0usize));
        set_param(&enc, 2, (&*p.biases, 0usize));
        // Bound AT the offset, so the kernel's `x[i]` starts at the row.
        set_param(&enc, 3, (storage.buffer(), start * io.size_in_bytes()));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, p.d_in as u32);
        set_param(&enc, 6, p.gpr as u32);
        if let Some(c) = p.tile_cols {
            // Never left unbound: a zero here loops `c0 += 0` forever.
            set_param(&enc, 7, c as u32);
        }
        // The tile, never d_in, on the tiled path: d_in * 4 is above the
        // limit there. Omitted, the kernel would run and write zeros.
        enc.set_threadgroup_memory_length(0, staged_cols * DType::F32.size_in_bytes());
        for b in [&*p.wq, &*p.scales, &*p.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((MetalStorage::new(out, dev, p.d_out, io), Shape::from(dims)))
    }
}

// ---------------------------------------------------------------------------
// The q8 embedding, and the lm_head that reads the same table.
//
// 413.3 MB of the served 4B's 1.39 GB, so it is not a detail. One table, two
// ends: `gather` for the token lookup and `project` for the logits.
// ---------------------------------------------------------------------------

/// A quantized g64 embedding table on the device, int8 or int4.
pub struct MetalEmbedTable {
    pub name: String,
    pub vocab: usize,
    pub d: usize,
    gpr: usize,
    /// 8 or 4: which pair of kernels reads this table.
    bits: u8,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalRuntime {
    /// Upload one table at `bits` (8 or 4). A tied model uploads ONE and
    /// points both ends at it.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_embed(
        &self,
        name: &str,
        bits: u8,
        vocab: usize,
        d: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<Arc<MetalEmbedTable>> {
        if bits != 8 && bits != 4 {
            candle_core::bail!("{name}: int{bits}, the embedding kernels read int8 or int4");
        }
        let per_word = 32 / bits as usize;
        if !d.is_multiple_of(per_word) {
            candle_core::bail!("{name}: d {d} is not a multiple of {per_word}, so a row would \
                                start inside a word");
        }
        if wq.len() != vocab * d / per_word {
            candle_core::bail!("{name}: {} words for {vocab}x{d} int{bits}", wq.len());
        }
        if scales.len() != vocab * gpr || biases.len() != vocab * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {vocab}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(Arc::new(MetalEmbedTable {
            name: name.to_string(),
            vocab,
            d,
            gpr,
            bits,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        }))
    }
}

/// The table plus the runtime that can launch it.
pub struct MetalEmbed {
    rt: Arc<MetalRuntime>,
    table: Arc<MetalEmbedTable>,
}

impl MetalEmbed {
    pub fn new(rt: Arc<MetalRuntime>, table: Arc<MetalEmbedTable>) -> Self {
        Self { rt, table }
    }

    /// The inner table, so the loader can compare two ends with `Arc::ptr_eq`.
    ///
    /// The tie is NOT observable on the `MetalEmbed`: a tied model wraps one
    /// table in two of them. It IS observable here, which is where
    /// `load_resolved` checks it against `tie_word_embeddings`, exactly as the
    /// CUDA loader does on its own buffers.
    pub fn table(&self) -> &Arc<MetalEmbedTable> {
        &self.table
    }
}

impl crate::device::QuantEmbedTable for MetalEmbed {
    fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::U32)?.contiguous()?;
        ids.apply_op1_no_bwd(&GatherOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let dims = h.dims();
        let d = *dims.last().expect("rank >= 1");
        if d != self.table.d {
            candle_core::bail!("{}: hidden of {d} for d={}", self.table.name, self.table.d);
        }
        let h = h.to_dtype(DType::F16)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }
}

struct GatherOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for GatherOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-gather"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 gather has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::U32 {
            candle_core::bail!("{}: ids must be u32, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous ids", t.name)))?;
        let n_ids = end - start;
        if n_ids == 0 {
            candle_core::bail!("{}: a gather of zero ids", t.name);
        }
        let kname = match t.bits {
            4 => "emb_q4_gather_metal",
            _ => "emb_q8_gather_metal",
        };
        let pipe = self.rt.pipeline(kname)?;
        // Not reducible: the kernel takes its token from the threadgroup's
        // position, so the grid is `n_ids` groups of exactly this many.
        let group = group_exactly(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(n_ids * t.d, DType::F32, "llvq-metal-q8-gather")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*t.wq, 0usize));
        set_param(&enc, 1, (&*t.scales, 0usize));
        set_param(&enc, 2, (&*t.biases, 0usize));
        set_param(&enc, 3, (storage.buffer(), 0usize));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, t.d as u32);
        set_param(&enc, 6, t.gpr as u32);
        // In ELEMENTS: the kernel reads `ids[ids_off + tok]`.
        set_param(&enc, 7, start as u32);
        for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: n_ids * group, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        dims.push(t.d);
        Ok((MetalStorage::new(out, dev, n_ids * t.d, DType::F32), Shape::from(dims)))
    }
}

struct HeadOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-head"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 head has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the head reads f16, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden", t.name)))?;
        let rows = (end - start) / t.d;
        if rows * t.d != end - start {
            candle_core::bail!("{}: {} values is not a whole number of rows of {}",
                               t.name, end - start, t.d);
        }
        let staged = t.d * DType::F32.size_in_bytes();
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!("{}: staging d={} wants {staged} B against {THREADGROUP_LIMIT}",
                               t.name, t.d);
        }
        let kname = match t.bits {
            4 => "tv_emb_q4_metal",
            _ => "tv_q8_metal",
        };
        let pipe = self.rt.pipeline(kname)?;
        let group = group_within(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(rows * t.vocab, DType::F32, "llvq-metal-q8-head")?;
        use candle_metal_kernels::utils::set_param;
        // One launch a row, which is what the CUDA host does: the kernel
        // computes one vocabulary row per SIMD-group and takes one activation.
        for r in 0..rows {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (&*t.wq, 0usize));
            set_param(&enc, 1, (&*t.scales, 0usize));
            set_param(&enc, 2, (&*t.biases, 0usize));
            set_param(&enc, 3, (storage.buffer(), 0usize));
            set_param(&enc, 4, (&*out, 0usize));
            set_param(&enc, 5, t.d as u32);
            set_param(&enc, 6, t.gpr as u32);
            set_param(&enc, 7, (start + r * t.d) as u32);
            set_param(&enc, 8, (r * t.vocab) as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: t.vocab * LANES, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
            );
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.vocab;
        Ok((MetalStorage::new(out, dev, rows * t.vocab, DType::F32), Shape::from(dims)))
    }
}

// ---------------------------------------------------------------------------
// The port, implemented. From here `model.rs` cannot tell a card from a Mac.
// ---------------------------------------------------------------------------

/// One Tetra projection, its runtime and its rotation, as the model sees it.
pub struct MetalLattice {
    rt: Arc<MetalRuntime>,
    proj: MetalTetraProj,
    rot: Option<Arc<MetalRotation>>,
    key: Option<crate::fused::RotKey>,
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
pub struct MetalInt4 {
    rt: Arc<MetalRuntime>,
    proj: MetalInt4Proj,
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

// ---------------------------------------------------------------------------
// The loader. One door, and after it `model.rs` names no backend.
// ---------------------------------------------------------------------------

impl MetalRuntime {
    /// A runtime from the portable table builder, which is where the CUDA
    /// loader gets the same numbers.
    ///
    /// `prefixes` and `suffixes` arrive packed into `u32` because that is the
    /// shape the CUDA upload path wants. The MSL reads them as `uchar*`, and
    /// on a little-endian machine those are the same bytes. Both sides are
    /// Apple silicon or x86, so this is a fact rather than an assumption.
    pub fn from_tables(
        device: &candle_core::Device,
        t: &crate::fused::Tetra48Tables,
        tile: usize,
    ) -> Result<Self> {
        let dev = match device {
            candle_core::Device::Metal(d) => d.clone(),
            other => candle_core::bail!("the Metal runtime wants a Metal device, got {other:?}"),
        };
        check_threadgroup_memory(&dev)?;
        if !tile.is_power_of_two() || !(32..=256).contains(&tile) {
            candle_core::bail!("tile {tile} is not a power of two in 32..=256");
        }
        Ok(Self {
            tables: MetalTables {
                rows: dev.new_buffer_with_data(&t.rows)?,
                prefixes: dev.new_buffer_with_data(&t.prefixes)?,
                branches: dev.new_buffer_with_data(&t.branches)?,
                suffixes: dev.new_buffer_with_data(&t.suffixes)?,
                invnorm: dev.new_buffer_with_data(&t.invnorm)?,
                // `Tetra48Tables` packs the suffix bytes into `u32` for the
                // CUDA upload path; unpacked here because the composition is
                // indexed by byte.
                c3_of: dev.new_buffer_with_data(&compose_c3(
                    &t.branches,
                    &t.suffixes.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
                ))?,
                // A served model keeps no host copy: it would pay for the
                // tables twice and its CPU arm is not a fallback.
                host: None,
            },
            kernel: kernel_from_env()?,
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
            moe_down_sg: std::sync::atomic::AtomicBool::new(MOE_DOWN_SG_DEFAULT),
            device: dev,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Put this runtime on `tuning`: the Tetra matvec, the int4 tile and row
    /// loop. Refused by name before anything is set if the tuning is bad.
    /// Projections already uploaded keep the tile they were routed with.
    pub fn set_tuning(&mut self, tuning: MetalTuning) -> Result<()> {
        tuning.validate()?;
        self.kernel = tuning.tetra_kernel;
        self.q4_tile_cols = tuning.q4_tile_cols;
        self.q4_prefetch = tuning.q4_prefetch;
        self.half_io = tuning.half_io;
        self.prefill_rows = tuning.prefill_rows;
        self.q4_rows = tuning.q4_rows;
        self.rot_tg = tuning.rot_tg;
        Ok(())
    }

    /// The entry points this runtime has compiled, sorted: which kernels a
    /// run actually launched. The twins return the served bits by design, so
    /// no output can say which one ran; this can.
    pub fn compiled_pipelines(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> =
            self.pipelines.read().expect("no panic held this lock").keys().map(|k| k.0).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// What this runtime launches, for a provenance line.
    pub fn tuning(&self) -> MetalTuning {
        MetalTuning {
            q4_tile_cols: self.q4_tile_cols,
            tetra_kernel: self.kernel,
            q4_prefetch: self.q4_prefetch,
            half_io: self.half_io,
            prefill_rows: self.prefill_rows,
            q4_rows: self.q4_rows,
            rot_tg: self.rot_tg,
        }
    }

    /// [`Self::upload`] from the row-strided `u32` stream the reader hands
    /// back, rather than from bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_stream(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u32],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.upload(name, d_out, d_in, nblocks, row_stride_u32, &bytes, gscale, rscale, tail)
    }
}

/// The served object on Metal, loaded from a sealed artifact.
///
/// The twin of `fused_cuda::load_resolved`, and deliberately not a copy of it:
/// everything portable comes from [`crate::fused`], which the CUDA loader uses
/// too. What is written here is the device half, about a hundred lines.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
) -> Result<crate::fused::FusedSealed> {
    load_resolved_tuned(path, device, dtype, layout, emode, share, fuse, kv, origin, None)
}

/// [`load_resolved`] with the runtime put on `tuning` before the first
/// upload. `None` is the served door, byte for byte: the Tetra matvec named
/// by `LLVQ_METAL_KERNEL` or the served one, the served int4 kernels, and
/// [`Q4_TILE_COLS`] for the records past the wall. A measurement harness
/// passes `Some` and names its arms itself.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved_tuned(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
    tuning: Option<MetalTuning>,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();
    if let Some(t) = &tuning {
        t.validate()?;
    }
    if dtype != DType::F16 {
        candle_core::bail!("the Metal fused path serves f16, asked for {dtype:?}");
    }
    if layout != crate::fused::FusedLayout::Tetra48 {
        candle_core::bail!(
            "{}: the Metal path carries the Tetra48 kernels only, asked for {layout:?}",
            from("LLVQ_FUSED_LAYOUT")
        );
    }
    if fuse == crate::fused::FuseMode::On {
        candle_core::bail!(
            "{}: there is no segmented Metal kernel, so a fused group cannot launch",
            from("LLVQ_FUSE")
        );
    }
    crate::fused::check_fuse(layout, share, fuse).map_err(candle_core::Error::msg)?;

    let mut model = crate::fused::load_with(path, layout, fuse).map_err(candle_core::Error::msg)?;
    let rot_launches = crate::rotplan::rot_launches(share, &model.matrices, &model.groups);
    let matvec_launches = model.matrices.len() + model.int4.len() + model.groups.len();
    let quantized_weights = model.quantized_weights;
    let carried_weights = model.carried_weights;
    let (file_bytes, runtime_bytes) = (model.file_bytes, model.runtime_bytes);

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("{path}: config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("{path}: tokenizer.json: {e}")))?;
    crate::fused_cpu::check_token_ids(&tokenizer, &model.raw)
        .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?;

    let embed_tables = match emode.bits() {
        None => None,
        Some(bits) => Some(
            crate::fused::take_embed_tables_at(&mut model.raw, config.tie_word_embeddings, bits)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    // The tile: no measured row for Apple, so the shader's own default.
    // `tuile-l40s-2026-09-20` measured 64 on sm_89 and nothing here.
    let tetra = llvq_search::tetra::Tetra::new();
    let tables = crate::fused::tetra48_tables(&tetra);
    let mut rt = MetalRuntime::from_tables(device, &tables, 64)?;
    if let Some(t) = tuning {
        rt.set_tuning(t)?;
    }
    let rt = Arc::new(rt);

    let mut rotations: HashMap<crate::fused::RotKey, Arc<MetalRotation>> = HashMap::new();
    for (&key, t) in &model.rotations {
        rotations.insert(key, Arc::new(rt.upload_rotation(t)?));
    }

    let mut by_site: HashMap<(usize, String), crate::model::Proj> = HashMap::new();
    for m in &model.matrices {
        let crate::fused::HostStream::Tetra48 { words, stride_u32 } = &m.stream else {
            candle_core::bail!("{}: not a Tetra48 stream on a Tetra48 load", m.name);
        };
        let proj = rt.upload_stream(
            &m.name, m.d_out, m.d_in, m.nblocks, *stride_u32 as usize, words, &m.gscale, &m.rscale, &m.tail,
        )?;
        let rot = match m.rotation {
            None => None,
            Some(k) => Some(
                rotations
                    .get(&k)
                    .ok_or_else(|| candle_core::Error::msg(format!("rotation {k:?} missing")))?
                    .clone(),
            ),
        };
        let (layer, name) = llvq_artifact::split_name(&m.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, name),
            crate::model::Proj::Lattice(Arc::new(MetalLattice {
                rt: rt.clone(),
                proj,
                rot,
                key: m.rotation,
            })),
        );
    }

    let (mut int4_records, mut int4_tiled) = (0usize, 0usize);
    for q in &model.int4 {
        let gpr = q.d_in / q.group;
        let wq: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let proj = rt.upload_int4(&q.name, q.d_out, q.d_in, gpr, &wq, &q.scales, &q.biases)?;
        int4_records += 1;
        int4_tiled += usize::from(proj.tile_cols().is_some());
        let (layer, name) = llvq_artifact::split_name(&q.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, name),
            crate::model::Proj::Int4(Arc::new(MetalInt4 { rt: rt.clone(), proj })),
        );
    }
    let t = rt.tuning();
    println!(
        "metal: tetra matvec {} | int4 {} records, {int4_tiled} through tv_q4_metal_tiled at {} columns a tile, {} \
         untiled | int4 row loop {} | io {} | prefill {} | rotation {}",
        t.tetra_kernel,
        int4_records,
        t.q4_tile_cols,
        int4_records - int4_tiled,
        if t.q4_prefetch { "prefetching (tv_q4_pf.metal)" } else { "served" },
        if t.half_io { "half (in-kernel casts, tv_q4_pf.metal and llvq_tetra48_rows.metal)" } else { "f32 (served)" },
        match t.prefill_rows {
            1 => "row by row (served)".to_string(),
            n => format!(
                "{n} rows a launch, Tetra (tv_tetra48_metal_lutdr){}",
                if t.q4_rows { " and int4 (tv_q4_metal_rows)" } else { ", int4 row by row" }
            ),
        },
        if t.rot_tg { "threadgroup up to 8,192 (rot_apply_tg_metal)" } else { "device (served)" },
    );
    finish(
        path, device, dtype, layout, emode, share, fuse, kv, origin, model, config, tokenizer,
        embed_tables, rt, by_site, rot_launches, matvec_launches, quantized_weights,
        carried_weights, file_bytes, runtime_bytes,
    )
}

/// The half of the load that names no device.
///
/// Carried tensors, the embedding, the `VarBuilder`, the model and the
/// bookkeeping. It is a separate function because it is the part a third
/// backend would reuse verbatim, and because `load_resolved` above is then
/// short enough to read as what it is: an upload loop.
#[allow(clippy::too_many_arguments)]
fn finish(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
    model: crate::fused::FusedModel,
    config: candle_transformers::models::qwen3::Config,
    tokenizer: tokenizers::Tokenizer,
    embed_tables: Option<crate::fused::EmbedTables>,
    rt: Arc<MetalRuntime>,
    mut by_site: HashMap<(usize, String), crate::model::Proj>,
    rot_launches: usize,
    matvec_launches: usize,
    quantized_weights: usize,
    carried_weights: usize,
    file_bytes: u64,
    runtime_bytes: u64,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut carried_bytes = 0u64;
    for t in &model.raw {
        carried_bytes += t.len() as u64 * 2;
        tensors.insert(
            t.name.clone(),
            Tensor::from_vec(t.to_f32(), t.dims.clone(), device)?.to_dtype(dtype)?,
        );
    }

    let quant_embed = match &embed_tables {
        None => {
            let carried = crate::fused::carried_embed_tables(&model.raw);
            println!(
                "{}",
                crate::fused::EmbedReport::new(crate::fused::EmbedMode::F16, &carried)
                    .line(&from("LLVQ_EMBED"))
            );
            None
        }
        Some(tables) => {
            let to_upload = tables.buffers();
            let report = crate::fused::EmbedReport::new(emode, &to_upload);
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let mut uploaded: Vec<Arc<MetalEmbedTable>> = Vec::with_capacity(to_upload.len());
            for t in &to_upload {
                let llvq_artifact::RawData::Quant(q) = &t.data else {
                    candle_core::bail!("{}: not a quantized tensor", t.name);
                };
                if Some(q.bits) != emode.bits() {
                    candle_core::bail!(
                        "{}: int{} where the {} kernels want int{:?}",
                        t.name, q.bits, emode.name(), emode.bits()
                    );
                }
                if t.dims.len() != 2 {
                    candle_core::bail!("{}: dims {:?}, an embedding is 2-D", t.name, t.dims);
                }
                let (vocab, d) = (t.dims[0], t.dims[1]);
                let wq: Vec<u32> = q
                    .packed
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                uploaded.push(rt.upload_embed(
                    &t.name, q.bits, vocab, d, d.div_ceil(q.group), &wq, &q.scales, &q.biases,
                )?);
                carried_bytes += (q.packed.len() + 2 * q.scales.len() + 2 * q.biases.len()) as u64;
            }
            Some((uploaded, tables.wiring()))
        }
    };
    println!(
        "total expected on the device: {:.2} GB (projections {:.2} + carried {:.2})",
        (runtime_bytes + carried_bytes) as f64 / 1e9,
        runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    // The tie, checked on the TABLES and not on the handles. Two `MetalEmbed`
    // values wrap one table when the ends are tied, so `Arc::ptr_eq` on them
    // is false either way. This is the same check `fused_cuda` makes on its
    // own buffers, and `device.rs` says so where the trait is declared.
    if let Some((bufs, (ie, ih))) = &quant_embed {
        let tied = Arc::ptr_eq(&bufs[*ie], &bufs[*ih]);
        if tied != config.tie_word_embeddings {
            candle_core::bail!(
                "inconsistent q8 wiring: embedding and lm_head {} the same table while \
                 tie_word_embeddings = {}",
                if tied { "share" } else { "do not share" },
                config.tie_word_embeddings
            );
        }
    }

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    let total_sites = by_site.len();
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.remove(&(layer, name.to_string())).inspect(|_| {
            claimed += 1;
        })
    };
    let mut qwen = match &quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take, kv)?,
        Some((bufs, (ie, ih))) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8(Arc::new(MetalEmbed::new(rt.clone(), bufs[*ie].clone()))),
            crate::model::Head::Q8(Arc::new(MetalEmbed::new(rt.clone(), bufs[*ih].clone()))),
            kv,
        )?,
    };
    if claimed != total_sites {
        candle_core::bail!(
            "{path}: {claimed} of {total_sites} device projections were claimed by the model. \
             The rest would be served dense without saying so"
        );
    }
    qwen.set_rot_share(share);

    Ok(crate::fused::FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        // `rows` is 1 on the served tuning, which carries no prefill kernel;
        // `tile` is what the shader was compiled at.
        prefill: crate::device::Prefill { rows: rt.prefill_rows(), tile: rt.tile, from_env: false },
        embed_mode: emode,
        rot_share: share,
        rot_launches,
        fuse,
        matvec_launches,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
        carried_bytes,
    })
}

// ---------------------------------------------------------------------------
// Wave 1 of the Inkling 2 Mini throughput plan (`docs/plan-debit-inkling2-mini.md`).
//
// Three additions, none of which touches a served path:
//   - K1: resident data allocated by the Metal device itself, outside
//     candle's buffer pool, with the pool's storage mode.
//   - P1: `tv_tetra48_metal_lutdrf`, the prefill rows kernel storing f32.
//   - D4: `tv_emb_q4_metal_xf`, the int4 g64 head reading an f32 activation.
//
// The two kernels are new text appended to a shipped unit and compiled as a
// library of their own (`source_of`), so the served pipelines keep their
// source bytes and their register allocation.
// ---------------------------------------------------------------------------

/// `tv_tetra48_metal_lutdr` storing f32: the same loop, the same per-row
/// arithmetic, and the served f32 epilogue `tetra48_rows_epilogue` instead of
/// the half one. Row r of a chunk is therefore the f32 the served matvec
/// returns for that row alone. Gated: `tests/inkling_wave1_kernels.rs`.
macro_rules! tetra_rows_f32_msl {
    () => {
        r#"
// ---------------------------------------------------------------------------
// tv_tetra48_metal_lutdrf: tv_tetra48_metal_lutdr with f32 rows (plan item P1).
// ---------------------------------------------------------------------------
kernel void tv_tetra48_metal_lutdrf(const device uint*   words          [[buffer(0)]],
                                    constant uint&       row_stride_u32 [[buffer(1)]],
                                    const device uint*   rows           [[buffer(2)]],
                                    const device uchar*  prefixes       [[buffer(3)]],
                                    const device ushort* branches       [[buffer(4)]],
                                    const device uchar*  suffixes       [[buffer(5)]],
                                    const device float*  gscale         [[buffer(6)]],
                                    const device float*  invnorm        [[buffer(7)]],
                                    const device float*  rscale         [[buffer(8)]],
                                    const device half*   tail           [[buffer(9)]],
                                    const device float*  x              [[buffer(10)]],
                                    device float*        y              [[buffer(11)]],
                                    constant uint&       nblocks        [[buffer(12)]],
                                    constant uint&       tail_w         [[buffer(13)]],
                                    constant uint&       nrows          [[buffer(15)]],
                                    constant uint&       d_out          [[buffer(16)]],
                                    uint gid  [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    uint d_in = nblocks * 24u + tail_w;
    float acc[TETRA48_PREFILL_MAX];
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
            if (r < nrows) {
                const device float* xb = x + r * d_in + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) {
                    a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                }
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue(acc[r], row, lane, rscale, tail, x + r * d_in, y + r * d_out,
                                  nblocks, tail_w);
        }
    }
}
"#
    };
}

/// `tv_emb_q4_metal` reading an f32 activation: the int4 g64 head of a model
/// that runs in f32 (Inkling 2 Mini), so the hidden state is not narrowed to
/// f16 on its way in. The staging, the lane order, the dequantization and the
/// butterfly are `tv_emb_q4_metal`'s text; only the type of `x` changes.
macro_rules! emb_q4_xf_msl {
    () => {
        r#"
// ---------------------------------------------------------------------------
// tv_emb_q4_metal_xf: tv_emb_q4_metal with an f32 activation (plan item D4).
// ---------------------------------------------------------------------------
kernel void tv_emb_q4_metal_xf(const device uint*  wq     [[buffer(0)]],
                               const device half*  scales [[buffer(1)]],
                               const device half*  biases [[buffer(2)]],
                               const device float* x      [[buffer(3)]],
                               device float*       y      [[buffer(4)]],
                               constant uint&      d      [[buffer(5)]],
                               constant uint&      gpr    [[buffer(6)]],
                               constant uint&      x_off  [[buffer(7)]],
                               constant uint&      y_off  [[buffer(8)]],
                               threadgroup float*  xs     [[threadgroup(0)]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint gid  [[thread_position_in_grid]],
                               uint tgs  [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    for (uint i = tid; i < d; i += tgs) {
        xs[i] = x[x_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nwords = d >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 3;
        half sb = scales[s0 + (c >> 6)];
        half bb = biases[s0 + (c >> 6)];
        for (uint k = 0; k < 8u; ++k) {
            acc = fma(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    acc = e4_warp_sum(acc);
    if (lane == 0u) {
        y[y_off + row] = acc;
    }
}
"#
    };
}

const SRC_TETRA_ROWS_F32: &str = concat!(
    include_str!("../kernels/llvq_tetra48.metal"),
    include_str!("../kernels/llvq_tetra48_rows.metal"),
    tetra_rows_f32_msl!()
);
const SRC_E4_XF: &str = concat!(include_str!("../kernels/emb_q4.metal"), emb_q4_xf_msl!());

impl MetalRuntime {
    /// A buffer holding `data`, made by the Metal device itself, in the
    /// storage mode candle's pool uses (`StorageModeShared`), and NOT
    /// registered in that pool (plan item K1).
    ///
    /// candle 0.9.2 registers every `new_buffer_with_data` in its pool, and
    /// walks the pool at every output allocation (`find_available_buffer`)
    /// and at every command-buffer flush (`drop_unused_buffers`). A resident
    /// Inkling layer puts 2,048 buffers there, 49,152 for 24 layers. A buffer
    /// made here is freed when its last `Arc` drops, and the allocator can
    /// never hand it out as scratch (trap 3 of this file's header).
    fn buffer_outside_pool<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        if size == 0 {
            candle_core::bail!("Metal refuses a zero-length buffer");
        }
        let b = self
            .device
            .device()
            .new_buffer_with_data(data.as_ptr().cast(), size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .map_err(|e| candle_core::Error::msg(format!("a buffer of {size} B outside the pool: {e:?}")))?;
        Ok(Arc::new(b))
    }

    /// [`Self::upload_stream`] with every buffer outside candle's pool: the
    /// same checks, the same bytes, another allocator. The resident Inkling
    /// experts use it; no served path does.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_stream_unpooled(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u32],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        // The checks of `upload`, in its order and with its messages.
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}. The kernel \
                 carries no row guard, so a partial threadgroup would compute and STORE \
                 rows past the output"
            );
        }
        if nblocks * llvq_core::DIM > d_in {
            candle_core::bail!("{name}: {nblocks} blocks for d_in {d_in}");
        }
        let tail_w = d_in - nblocks * llvq_core::DIM;
        if rscale.len() != d_out {
            candle_core::bail!("{name}: {} row scales for {d_out} rows", rscale.len());
        }
        if tail.len() != d_out * tail_w {
            candle_core::bail!("{name}: {} tail values for {d_out}x{tail_w}", tail.len());
        }
        let want_stride = llvq_artifact::tetra48::stride_u32(nblocks);
        if row_stride_u32 != want_stride {
            candle_core::bail!(
                "{name}: row stride {row_stride_u32} for {nblocks} blocks, which wants \
                 {want_stride}"
            );
        }
        let want_words = d_out * row_stride_u32;
        if words.len() != want_words && words.len() != want_words + 1 {
            candle_core::bail!(
                "{name}: {} stream words for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_words} or {want_words} plus one guard word",
                words.len()
            );
        }
        let dummy = [0u16];
        let tail_src = if tail.is_empty() { &dummy[..] } else { tail };
        // The words as the reader hands them back: on this little-endian
        // machine the same bytes `upload_stream` builds by hand.
        Ok(MetalTetraProj {
            name: name.to_string(),
            d_out,
            d_in,
            nblocks,
            tail_w,
            row_stride_u32,
            words: self.buffer_outside_pool(words)?,
            gscale: self.buffer_outside_pool(gscale)?,
            rscale: self.buffer_outside_pool(rscale)?,
            tail: self.buffer_outside_pool(tail_src)?,
            host: self.tables.host.is_some().then(|| HostProj {
                words: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
                gscale: gscale.to_vec(),
                rscale: rscale.to_vec(),
                tail: tail.to_vec(),
            }),
        })
    }

    /// [`Self::upload_rotation`] with both tables outside candle's pool.
    pub fn upload_rotation_unpooled(&self, t: &crate::fused::RotationTables) -> Result<MetalRotation> {
        let dummy = [0f32];
        let small = if t.small.is_empty() { &dummy[..] } else { &t.small };
        Ok(MetalRotation {
            signbits: self.buffer_outside_pool(&t.signbits)?,
            small: self.buffer_outside_pool(small)?,
            n: t.n as u32,
            m: t.m as u32,
            k: t.k as u32,
            inv: t.inv,
            threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
        })
    }

    /// `Y = W X` for `n_rows` rotated rows `[n_rows, d_in]` in f32, one launch
    /// of `tv_tetra48_metal_lutdrf`, as f32 `[n_rows, d_out]`. Row r is the
    /// served matvec of row r, bit for bit (plan item P1). Same bounds as
    /// [`Self::matvec_rows`].
    pub fn matvec_rows_f32(&self, proj: &MetalTetraProj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill_rows.max(1) || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for a prefill launch of at most {} (kernel bound {TETRA_PREFILL_MAX})",
                proj.name,
                self.prefill_rows
            );
        }
        x.apply_op1_no_bwd(&TetraRowsF32 { rt: self, proj, n_rows })
    }

    /// Logits `h W^T` of an int4 g64 table for `h` `[rows, d]` in f32, as f32
    /// `[rows, vocab]`: one launch of `tv_emb_q4_metal_xf` a row, the f32
    /// activation staged as it arrives (plan item D4).
    pub fn head_q4_f32(&self, t: &MetalEmbedTable, h: &Tensor) -> Result<Tensor> {
        if t.bits != 4 {
            candle_core::bail!("{}: int{}, the f32 head reads int4", t.name, t.bits);
        }
        let d = *h.dims().last().expect("rank >= 1");
        if d != t.d {
            candle_core::bail!("{}: hidden of {d} for d={}", t.name, t.d);
        }
        let h = h.to_dtype(DType::F32)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadF32Op { rt: self, t })
    }
}

impl MetalTetraProj {
    /// Device bytes of the four buffers (words, gain scales, row scales,
    /// tail): what a report counts.
    pub fn device_bytes(&self) -> u64 {
        (self.words.length() + self.gscale.length() + self.rscale.length() + self.tail.length()) as u64
    }
}

/// One launch of `tv_tetra48_metal_lutdrf`, as a candle op.
struct TetraRowsF32<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
    n_rows: usize,
}

impl CustomOp1 for TetraRowsF32<'_> {
    fn name(&self) -> &'static str {
        "tetra48-rows-f32"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "{}: the f32 prefill kernel runs on Metal only; its reference is the matvec, row by row",
            self.proj.name
        )
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: prefill rows in f32 only, got {:?}", p.name, storage.dtype());
        }
        if !layout.is_contiguous() {
            candle_core::bail!("{}: prefill rows must be contiguous", p.name);
        }
        if layout.dims() != [self.n_rows, p.d_in] {
            candle_core::bail!(
                "{}: prefill activation {:?} for [{}, {}]",
                p.name,
                layout.dims(),
                self.n_rows,
                p.d_in
            );
        }
        if !p.d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("{}: d_out {} is not a multiple of {} rows", p.name, p.d_out, GROUP / LANES);
        }
        let dev = storage.device().clone();
        let pipe = self.rt.pipeline("tv_tetra48_metal_lutdrf")?;
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F32, "tetra48-rows-f32")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, p.row_stride_u32 as u32);
        set_param(&enc, 2, (&*self.rt.tables.rows, 0usize));
        set_param(&enc, 3, (&*self.rt.tables.prefixes, 0usize));
        set_param(&enc, 4, (&*self.rt.tables.branches, 0usize));
        set_param(&enc, 5, (&*self.rt.tables.suffixes, 0usize));
        set_param(&enc, 6, (&*p.gscale, 0usize));
        set_param(&enc, 7, (&*self.rt.tables.invnorm, 0usize));
        set_param(&enc, 8, (&*p.rscale, 0usize));
        set_param(&enc, 9, (&*p.tail, 0usize));
        set_param(&enc, 10, (storage.buffer(), x_off));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, p.nblocks as u32);
        set_param(&enc, 13, p.tail_w as u32);
        set_param(&enc, 15, self.n_rows as u32);
        set_param(&enc, 16, p.d_out as u32);
        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F32),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

/// `tv_emb_q4_metal_xf`, one launch a row, as a candle op.
struct HeadF32Op<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for HeadF32Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4-head-f32"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 f32 head has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: the f32 head reads f32, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden", t.name)))?;
        let rows = (end - start) / t.d;
        if rows == 0 || rows * t.d != end - start {
            candle_core::bail!("{}: {} values is not a whole number of rows of {}", t.name, end - start, t.d);
        }
        // The kernel's own limits: no row guard (8 rows a threadgroup), the
        // activation staged whole, every index a u32.
        if !t.vocab.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("{}: {} rows is not a multiple of {}", t.name, t.vocab, GROUP / LANES);
        }
        let staged = t.d * DType::F32.size_in_bytes();
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!("{}: staging d={} wants {staged} B against {THREADGROUP_LIMIT}", t.name, t.d);
        }
        let reach = [rows * t.vocab, t.vocab * (t.d / 8), t.vocab * t.gpr, start + rows * t.d];
        if reach.iter().any(|&v| v > u32::MAX as usize) || t.vocab * LANES > u32::MAX as usize {
            candle_core::bail!("{}: an index past u32 ({reach:?})", t.name);
        }
        let pipe = self.rt.pipeline("tv_emb_q4_metal_xf")?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(rows * t.vocab, DType::F32, "llvq-metal-q4-head-f32")?;
        use candle_metal_kernels::utils::set_param;
        for r in 0..rows {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (&*t.wq, 0usize));
            set_param(&enc, 1, (&*t.scales, 0usize));
            set_param(&enc, 2, (&*t.biases, 0usize));
            set_param(&enc, 3, (storage.buffer(), 0usize));
            set_param(&enc, 4, (&*out, 0usize));
            set_param(&enc, 5, t.d as u32);
            set_param(&enc, 6, t.gpr as u32);
            set_param(&enc, 7, (start + r * t.d) as u32);
            set_param(&enc, 8, (r * t.vocab) as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: t.vocab * LANES, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
            );
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.vocab;
        Ok((MetalStorage::new(out, dev, rows * t.vocab, DType::F32), Shape::from(dims)))
    }
}

// ---------------------------------------------------------------------------
// Wave 2 of the Inkling 2 Mini throughput plan, item D3: the routed experts
// of one decoded token in four launches.
//
// The layout. Every record of a layer's experts (Tetra48 words, gain scales,
// row scales, tails, rotation sign bits and odd blocks) sits in ONE Metal
// placement heap, outside candle's pool. Two kinds of buffers alias it:
//   - one slab buffer over the whole heap, which the new kernels index with
//     the per-expert offsets of a small table (`MoeExpertOff` in the MSL);
//   - one buffer per record, at the record's offset, which is what a
//     `MetalTetraProj` or a `MetalRotation` holds, so the one-expert-at-a-time
//     path reads the same bytes through the served host ops, unchanged.
// One copy of the bytes, at a 256-byte alignment a record
// (`heapBufferSizeAndAlignWithLength`, measured on the M3 Max).
//
// The four launches, each reading the routed ids and weights from one small
// device buffer (`sel`: ids ascending, then the weights' f32 bits):
//   a. `moe_rot_metal`: the input rotation of the selected experts, one
//      threadgroup an expert, the body of `rot_apply_rows_metal`;
//   b. `moe_tetra_gu_metal`: gate and up of the selected experts, one
//      simdgroup a row, the body of the served `tv_tetra48_metal_lut`;
//   c. `moe_rot_metal` again, on the activation, into the basis of `down`;
//   d. `moe_tetra_down_metal`: down of each selected expert, the same body,
//      each row times its routing weight and summed in ascending expert
//      order in f32, from +0, with the two roundings of the separate
//      `mul` and `add` (the unit compiles under `fp contract(off)`).
// The activation between b and c stays the caller's (candle ops), so its
// bits are candle's.
//
// The new text is appended to a shipped unit and compiled as a library of its
// own (`source_of`), as wave 1 did, so no served pipeline changes source.
// ---------------------------------------------------------------------------

/// The per-expert offset record both new libraries read. 16 `u32`, offsets
/// from the slab base in 4-byte words, tails in 2-byte halves.
macro_rules! moe_off_msl {
    () => {
        r#"
// ---------------------------------------------------------------------------
// Plan item D3: where one expert's records sit in its layer's slab.
// ---------------------------------------------------------------------------
struct MoeExpertOff {
    uint words[3];
    uint rscale[3];
    uint gscale[3];
    uint tail[3];
    uint sign_in;
    uint small_in;
    uint sign_down;
    uint small_down;
};
"#
    };
}

/// `rot_apply_rows_metal` with the sign bits and the odd block of each row
/// taken from the routed expert of that row.
macro_rules! moe_rot_msl {
    () => {
        r#"
// moe_rot_metal: one threadgroup a selected expert, the stages of
// rot_apply_rows_metal. `which` 0 reads the expert's input rotation, 1 its
// down rotation. `row_stride` 0 rotates the same input row for every expert.
kernel void moe_rot_metal(const device ushort*       xin        [[buffer(0)]],
                          const device uint*         slab       [[buffer(1)]],
                          const device float*        slabf      [[buffer(2)]],
                          const device MoeExpertOff* off        [[buffer(3)]],
                          const device uint*         sel        [[buffer(4)]],
                          device float*              xout       [[buffer(5)]],
                          device float*              scratch    [[buffer(6)]],
                          constant uint&             n          [[buffer(7)]],
                          constant uint&             m          [[buffer(8)]],
                          constant uint&             k          [[buffer(9)]],
                          constant float&            inv        [[buffer(10)]],
                          constant uint&             x_off      [[buffer(11)]],
                          constant uint&             row_stride [[buffer(12)]],
                          constant uint&             which      [[buffer(13)]],
                          uint tid      [[thread_position_in_threadgroup]],
                          uint nthreads [[threads_per_threadgroup]],
                          uint r        [[threadgroup_position_in_grid]])
{
    MoeExpertOff o = off[sel[r]];
    const device uint* signbits = slab + (which == 0u ? o.sign_in : o.sign_down);
    const device float* small = slabf + (which == 0u ? o.small_in : o.small_down);
    device float* s = scratch + (ulong)r * (ulong)n;

    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_device);

    for (uint len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        threadgroup_barrier(mem_flags::mem_device);
    }

    device float* out = xout + (ulong)r * (ulong)n;
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}
"#
    };
}

/// Gate and up, then down with the weighted sum: the body of the served
/// `tv_tetra48_metal_lut`, one output row a simdgroup, its pointers taken
/// from the routed expert.
macro_rules! moe_tetra_msl {
    () => {
        r#"
// moe_tetra_gu_metal: gate (mat 0) and up (mat 1) of every selected expert.
// Simdgroup g computes row `row` of matrix `mat` of selected expert `kk`,
// g = (mat * nsel + kk) * d_out + row, and stores it at y[g], so the output
// is [2, nsel, d_out]. d_out is a multiple of 8, so the 8 simdgroups of a
// threadgroup share one expert and one activation row, which is what the
// staging needs.
kernel void moe_tetra_gu_metal(const device uint*         slab           [[buffer(0)]],
                               const device float*        slabf          [[buffer(1)]],
                               const device half*         slabh          [[buffer(2)]],
                               const device MoeExpertOff* off            [[buffer(3)]],
                               const device uint*         sel            [[buffer(4)]],
                               const device uint*         rows           [[buffer(5)]],
                               const device uchar*        prefixes       [[buffer(6)]],
                               const device ushort*       branches       [[buffer(7)]],
                               const device uchar*        suffixes       [[buffer(8)]],
                               const device float*        invnorm        [[buffer(9)]],
                               const device float*        xall           [[buffer(10)]],
                               device float*              y              [[buffer(11)]],
                               constant uint&             row_stride_u32 [[buffer(12)]],
                               constant uint&             nblocks        [[buffer(13)]],
                               constant uint&             tail_w         [[buffer(14)]],
                               constant uint&             d_out          [[buffer(15)]],
                               constant uint&             nsel           [[buffer(16)]],
                               threadgroup float*         xs             [[threadgroup(0)]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint gid  [[thread_position_in_grid]],
                               uint tgs  [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]])
{
    uint g = gid >> 5;
    uint per_mat = nsel * d_out;
    uint mat = g / per_mat;
    uint kk = (g - mat * per_mat) / d_out;
    uint row = g - mat * per_mat - kk * d_out;
    MoeExpertOff o = off[sel[kk]];
    const device uint* words = slab + o.words[mat];
    const device float* gscale = slabf + o.gscale[mat];
    const device float* rscale = slabf + o.rscale[mat];
    const device half* tail = slabh + o.tail[mat];
    const device float* x = xall + kk * (nblocks * 24u + tail_w);

    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_lut(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[g] = fma(acc, rscale[row], tv);
    }
}

/// `+0.0f + p` with the IEEE sign rule spelled in integers: under fast math
/// the compiler may assume no signed zeros and drop a written `0.0f + p`,
/// which would hand back -0 where the separate `add` returns +0.
inline float moe_zero_plus(float p)
{
    return as_type<uint>(p) == 0x80000000u ? 0.0f : p;
}

// moe_tetra_down_metal: down (mat 2) of every selected expert, in ascending
// slot order, each row times its routing weight (the f32 bits at
// sel[nsel + kk]) and added to the running row, from +0. One simdgroup an
// output row; `nsel` is uniform, so every barrier is reached by every thread.
kernel void moe_tetra_down_metal(const device uint*         slab           [[buffer(0)]],
                                 const device float*        slabf          [[buffer(1)]],
                                 const device half*         slabh          [[buffer(2)]],
                                 const device MoeExpertOff* off            [[buffer(3)]],
                                 const device uint*         sel            [[buffer(4)]],
                                 const device uint*         rows           [[buffer(5)]],
                                 const device uchar*        prefixes       [[buffer(6)]],
                                 const device ushort*       branches       [[buffer(7)]],
                                 const device uchar*        suffixes       [[buffer(8)]],
                                 const device float*        invnorm        [[buffer(9)]],
                                 const device float*        xall           [[buffer(10)]],
                                 device float*              y              [[buffer(11)]],
                                 constant uint&             row_stride_u32 [[buffer(12)]],
                                 constant uint&             nblocks        [[buffer(13)]],
                                 constant uint&             tail_w         [[buffer(14)]],
                                 constant uint&             d_out          [[buffer(15)]],
                                 constant uint&             nsel           [[buffer(16)]],
                                 threadgroup float*         xs             [[threadgroup(0)]],
                                 uint tid  [[thread_position_in_threadgroup]],
                                 uint gid  [[thread_position_in_grid]],
                                 uint tgs  [[threads_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    float out = 0.0f;

    for (uint kk = 0u; kk < nsel; ++kk) {
        MoeExpertOff o = off[sel[kk]];
        const device uint* words = slab + o.words[2];
        const device float* gscale = slabf + o.gscale[2];
        const device float* rscale = slabf + o.rscale[2];
        const device half* tail = slabh + o.tail[2];
        const device float* x = xall + kk * (nblocks * 24u + tail_w);
        const device uint* wrow = words + row * row_stride_u32;
        float acc = 0.0f;

        for (uint t = 0u; t < ntiles; ++t) {
            uint jlo = t * LLVQ_TILE_BLOCKS;
            uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
            uint n = (jhi - jlo) * 24u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < n; i += tgs) {
                xs[i] = x[jlo * 24u + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint j = jlo + lane; j < jhi; j += 32u) {
                uint lo, hi16;
                f1r_load(wrow, j, lo, hi16);
                acc += tetra48_dot_lut(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
            }
        }

        acc = warp_sum(acc);
        if (lane == 0u) {
            float tv = 0.0f;
            const device float* xt = x + nblocks * 24u;
            for (uint i = 0u; i < tail_w; ++i) {
                tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
            }
            float yv = fma(acc, rscale[row], tv);
            float p = yv * as_type<float>(sel[nsel + kk]);
            out = kk == 0u ? moe_zero_plus(p) : out + p;
        }
    }
    if (lane == 0u) {
        y[row] = out;
    }
}

/// `tetra48_dot_lut` reading its 24 activations from device memory: the
/// served function's text but the pointer's address space (the twin of
/// `tetra48_dot_lut_dev` in `llvq_tetra48_rows.metal`).
inline float moe_dot_lut_dev(uint lo,
                             uint hi16,
                             F1rTables t,
                             const device float* xb,
                             const device float* gscale,
                             const device float* invnorm)
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float acc = 0.0f;
    float n2 = 0.0f;
    for (uint sec = 0u; sec < 3u; ++sec) {
        uint c = cs[sec];
        uint row = rs[sec];
        for (uint j = 0u; j < 8u; ++j) {
            uint o = b.p + 2u * ((c >> j) & 1u);
            uint rho = (row >> (4u * j)) & 15u;
            float v = F1R_VAL[(o << 4) | rho];
            n2 = fma(v, v, n2);
            acc = fma(v, xb[TETRA48_ORDER[sec * 8u + j]], acc);
        }
    }
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

// moe_tetra_down_sg_metal: moe_tetra_down_metal with the selected experts
// side by side. One threadgroup an output row, `nsel` simdgroups, simdgroup
// kk the row of slot kk's expert: the same lane order and butterfly as the
// served body, the activation read from device memory (the same floats), no
// tile and no barrier in the loop. Each slot's weighted product goes to
// threadgroup memory; lane 0 of simdgroup 0 adds them in ascending slot
// order from +0, the served kernel's sum.
kernel void moe_tetra_down_sg_metal(const device uint*         slab           [[buffer(0)]],
                                    const device float*        slabf          [[buffer(1)]],
                                    const device half*         slabh          [[buffer(2)]],
                                    const device MoeExpertOff* off            [[buffer(3)]],
                                    const device uint*         sel            [[buffer(4)]],
                                    const device uint*         rows           [[buffer(5)]],
                                    const device uchar*        prefixes       [[buffer(6)]],
                                    const device ushort*       branches       [[buffer(7)]],
                                    const device uchar*        suffixes       [[buffer(8)]],
                                    const device float*        invnorm        [[buffer(9)]],
                                    const device float*        xall           [[buffer(10)]],
                                    device float*              y              [[buffer(11)]],
                                    constant uint&             row_stride_u32 [[buffer(12)]],
                                    constant uint&             nblocks        [[buffer(13)]],
                                    constant uint&             tail_w         [[buffer(14)]],
                                    constant uint&             d_out          [[buffer(15)]],
                                    constant uint&             nsel           [[buffer(16)]],
                                    uint row  [[threadgroup_position_in_grid]],
                                    uint kk   [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float part[32];
    F1rTables tab = { rows, prefixes, branches, suffixes };
    MoeExpertOff o = off[sel[kk]];
    const device uint* words = slab + o.words[2];
    const device float* gscale = slabf + o.gscale[2];
    const device float* rscale = slabf + o.rscale[2];
    const device half* tail = slabh + o.tail[2];
    const device float* x = xall + kk * (nblocks * 24u + tail_w);
    const device uint* wrow = words + row * row_stride_u32;
    float acc = 0.0f;
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        acc += moe_dot_lut_dev(lo, hi16, tab, x + j * 24u, gscale, invnorm);
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        float yv = fma(acc, rscale[row], tv);
        part[kk] = yv * as_type<float>(sel[nsel + kk]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk == 0u && lane == 0u) {
        float out = moe_zero_plus(part[0]);
        for (uint i = 1u; i < nsel; ++i) {
            out = out + part[i];
        }
        y[row] = out;
    }
}
"#
    };
}

use candle_core::CustomOp2;

const SRC_MOE_ROT: &str = concat!(include_str!("../kernels/llvq_rot.metal"), moe_off_msl!(), moe_rot_msl!());
const SRC_MOE_TETRA: &str = concat!(include_str!("../kernels/llvq_tetra48.metal"), moe_off_msl!(), moe_tetra_msl!());

/// Words of one `MoeExpertOff`.
const MOE_OFF_WORDS: usize = 16;
/// Routed experts a call can name: the bound of the `sel` buffer.
pub const MOE_MAX_SELECTED: usize = 32;

/// Whether a new runtime runs the fused MoE call's `down` one simdgroup a
/// slot ([`MetalRuntime::set_moe_down_sg`]). On since 2026-11-26: every f32
/// of the one-expert path (`tests/inkling_decode_kernels.rs`), the logits of
/// 128 decode steps byte for byte, and a step 5.22 % shorter, median of 5
/// alternated rounds [4.07, 10.58] (`docs/mesures/inkling-mac-decode-2026-11-26.txt`).
pub const MOE_DOWN_SG_DEFAULT: bool = true;

/// One record of an expert, on the host, as the resident loader reads it.
pub struct MoeHostProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub row_stride_u32: usize,
    pub words: Vec<u32>,
    pub gscale: [f32; 2],
    pub rscale: Vec<f32>,
    pub tail: Vec<u16>,
}

/// One expert on the host: gate, up, down, and its two rotations.
pub struct MoeHostExpert {
    pub projs: [MoeHostProj; 3],
    pub rot_in: crate::fused::RotationTables,
    pub rot_down: crate::fused::RotationTables,
}

/// One expert of a slab, as the one-expert path reads it: aliases of the
/// slab's records.
pub struct MoeSlabExpert {
    pub gate: MetalTetraProj,
    pub up: MetalTetraProj,
    pub down: MetalTetraProj,
    pub rot_in: MetalRotation,
    pub rot_down: MetalRotation,
}

/// What the four launches need to know, the same for every expert of a
/// layer. `None` on a slab whose experts differ, which the fused path then
/// refuses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeShape {
    pub hidden: usize,
    pub inter: usize,
    nb_in: usize,
    nb_down: usize,
    tail_in: usize,
    tail_down: usize,
    stride_in: usize,
    stride_down: usize,
    rot_in: (u32, u32, u32, u32, usize),
    rot_down: (u32, u32, u32, u32, usize),
}

/// Owns a Metal object whose type this crate cannot name (`objc2` is not
/// one of its dependencies), so the object lives as long as its owner.
trait MoeHold: Send + Sync {}
struct MoeHeld<T>(#[allow(dead_code)] T);
// SAFETY: the held heap is never used through this handle, only kept alive;
// Metal objects may be retained and released from any thread.
unsafe impl<T> Send for MoeHeld<T> {}
unsafe impl<T> Sync for MoeHeld<T> {}
impl<T> MoeHold for MoeHeld<T> {}

/// A layer's experts in one placement heap, and the offset table.
pub struct MetalMoeSlab {
    // Held so the heap outlives every alias, whatever the drop order.
    _heap: Box<dyn MoeHold>,
    slab: Arc<Buffer>,
    offsets: Arc<Buffer>,
    experts: usize,
    shape: Option<MoeShape>,
    bytes: u64,
    records: usize,
}

impl MetalMoeSlab {
    /// Experts in the slab.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes of the heap: the records, each at a 256-byte alignment.
    pub fn device_bytes(&self) -> u64 {
        self.bytes
    }

    /// Records the heap holds (each one aliased by one buffer).
    pub fn records(&self) -> usize {
        self.records
    }

    /// The shape the fused launches read, if every expert has it.
    pub fn shape(&self) -> Option<MoeShape> {
        self.shape
    }
}

/// The checks of `upload_stream_unpooled`, in its order, for a record that
/// goes into a slab. Returns the tail width.
fn moe_check_proj(p: &MoeHostProj) -> Result<usize> {
    let name = &p.name;
    let rows_per_group = GROUP / LANES;
    if !p.d_out.is_multiple_of(rows_per_group) {
        candle_core::bail!("{name}: d_out {} is not a multiple of {rows_per_group}", p.d_out);
    }
    if p.nblocks * llvq_core::DIM > p.d_in {
        candle_core::bail!("{name}: {} blocks for d_in {}", p.nblocks, p.d_in);
    }
    let tail_w = p.d_in - p.nblocks * llvq_core::DIM;
    if p.rscale.len() != p.d_out {
        candle_core::bail!("{name}: {} row scales for {} rows", p.rscale.len(), p.d_out);
    }
    if p.tail.len() != p.d_out * tail_w {
        candle_core::bail!("{name}: {} tail values for {}x{tail_w}", p.tail.len(), p.d_out);
    }
    let want_stride = llvq_artifact::tetra48::stride_u32(p.nblocks);
    if p.row_stride_u32 != want_stride {
        candle_core::bail!("{name}: row stride {} for {} blocks, which wants {want_stride}", p.row_stride_u32, p.nblocks);
    }
    let want_words = p.d_out * p.row_stride_u32;
    if p.words.len() != want_words && p.words.len() != want_words + 1 {
        candle_core::bail!("{name}: {} stream words, which wants {want_words} or one more", p.words.len());
    }
    Ok(tail_w)
}

fn moe_rot_params(t: &crate::fused::RotationTables) -> (u32, u32, u32, u32, usize) {
    (t.n as u32, t.m as u32, t.k as u32, t.inv.to_bits(), (t.n as u32).next_power_of_two().clamp(32, 1024) as usize)
}

impl MetalRuntime {
    /// Whether the fused MoE launches reproduce what this runtime launches
    /// one expert at a time: the served matvec (`_lut`, f32 store) and the
    /// device-staged rotation. Any other tuning keeps the one-expert path.
    pub fn moe_fused_eligible(&self) -> bool {
        self.kernel == SERVED_TETRA && !self.half_io && !self.rot_tg
    }

    /// Run the fused MoE call's `down` one simdgroup a slot
    /// (`moe_tetra_down_sg_metal`, `true`) or one simdgroup a row looping
    /// over the slots (`moe_tetra_down_metal`). The same bits either way.
    pub fn set_moe_down_sg(&self, on: bool) {
        self.moe_down_sg.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether [`Self::set_moe_down_sg`] is on.
    pub fn moe_down_sg(&self) -> bool {
        self.moe_down_sg.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Upload a layer's experts into one placement heap outside candle's
    /// pool, and hand back the slab and, per expert, the aliases the
    /// one-expert path reads. Every record passes the checks of
    /// [`Self::upload_stream_unpooled`]; the bytes are the bytes it uploads.
    pub fn upload_moe_layer(&self, experts: &[MoeHostExpert]) -> Result<(MetalMoeSlab, Vec<MoeSlabExpert>)> {
        use objc2_metal::{MTLDevice as _, MTLHeap as _};
        if experts.is_empty() {
            candle_core::bail!("a slab of zero experts");
        }
        let opts = objc2_metal::MTLResourceOptions::StorageModeShared;
        let mtl = self.device.device().as_ref();
        // Byte views of every record, in slab order, with dummies where the
        // one-record path uploads one.
        let dummy_tail = [0u16];
        let dummy_small = [0f32];
        let mut items: Vec<&[u8]> = Vec::with_capacity(experts.len() * 16);
        let mut tails = Vec::with_capacity(experts.len() * 3);
        fn bytes<T>(v: &[T]) -> &[u8] {
            // SAFETY: plain numbers (u32, u16, f32), read as bytes.
            unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
        }
        for e in experts {
            for p in &e.projs {
                tails.push(moe_check_proj(p)?);
            }
            for p in &e.projs {
                items.push(bytes(&p.words));
            }
            for p in &e.projs {
                items.push(bytes(&p.gscale));
            }
            for p in &e.projs {
                items.push(bytes(&p.rscale));
            }
            for p in &e.projs {
                items.push(bytes(if p.tail.is_empty() { &dummy_tail[..] } else { &p.tail[..] }));
            }
            for t in [&e.rot_in, &e.rot_down] {
                if t.signbits.is_empty() {
                    candle_core::bail!("a rotation of width {} with no sign bits", t.n);
                }
                items.push(bytes(&t.signbits));
                items.push(bytes(if t.small.is_empty() { &dummy_small[..] } else { &t.small[..] }));
            }
        }
        // Placement: each record at its heap alignment.
        let mut offs = Vec::with_capacity(items.len());
        let mut cursor = 0usize;
        for it in &items {
            let sa = mtl.heapBufferSizeAndAlignWithLength_options(it.len(), opts);
            let align = sa.align.max(4);
            cursor = cursor.div_ceil(align) * align;
            offs.push(cursor);
            cursor += sa.size.max(it.len());
        }
        let total = mtl.heapBufferSizeAndAlignWithLength_options(cursor, opts).size.max(cursor);
        if total / 4 > u32::MAX as usize {
            candle_core::bail!("a slab of {total} B: its word offsets pass u32");
        }
        let desc = objc2_metal::MTLHeapDescriptor::new();
        desc.setType(objc2_metal::MTLHeapType::Placement);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
        desc.setHazardTrackingMode(objc2_metal::MTLHazardTrackingMode::Untracked);
        desc.setSize(total);
        let heap = mtl
            .newHeapWithDescriptor(&desc)
            .ok_or_else(|| candle_core::Error::msg(format!("Metal refused a placement heap of {total} B")))?;
        let alias = |off: usize, len: usize| -> Result<Arc<Buffer>> {
            // SAFETY: `off` and `len` lie inside the heap, at its alignment.
            let raw = unsafe { heap.newBufferWithLength_options_offset(len, opts, off) }
                .ok_or_else(|| candle_core::Error::msg(format!("a heap buffer of {len} B at {off}")))?;
            Ok(Arc::new(Buffer::new(raw)))
        };
        let slab = alias(0, total)?;
        let base = slab.contents();
        for (it, &off) in items.iter().zip(&offs) {
            // SAFETY: the record fits at `off` by construction of `offs`, and
            // nothing reads the heap yet.
            unsafe { std::ptr::copy_nonoverlapping(it.as_ptr(), base.add(off), it.len()) };
        }
        let keep_host = self.tables.host.is_some();
        let mut table = Vec::with_capacity(experts.len() * MOE_OFF_WORDS);
        let mut out = Vec::with_capacity(experts.len());
        let first_shape = |e: &MoeHostExpert, t: &[usize]| MoeShape {
            hidden: e.projs[0].d_in,
            inter: e.projs[0].d_out,
            nb_in: e.projs[0].nblocks,
            nb_down: e.projs[2].nblocks,
            tail_in: t[0],
            tail_down: t[2],
            stride_in: e.projs[0].row_stride_u32,
            stride_down: e.projs[2].row_stride_u32,
            rot_in: moe_rot_params(&e.rot_in),
            rot_down: moe_rot_params(&e.rot_down),
        };
        let mut shape = Some(first_shape(&experts[0], &tails[0..3]));
        for (ei, e) in experts.iter().enumerate() {
            let o = &offs[ei * 16..ei * 16 + 16];
            let it = &items[ei * 16..ei * 16 + 16];
            let t = &tails[ei * 3..ei * 3 + 3];
            let [g, u, d] = &e.projs;
            let gate_up_alike = g.d_out == u.d_out && g.d_in == u.d_in && g.nblocks == u.nblocks && t[0] == t[1];
            let fits = d.d_in == g.d_out && d.d_out == g.d_in && e.rot_in.n == g.d_in && e.rot_down.n == d.d_in;
            if !gate_up_alike || !fits || shape != Some(first_shape(e, t)) {
                shape = None;
            }
            for w in [o[0] / 4, o[1] / 4, o[2] / 4, o[6] / 4, o[7] / 4, o[8] / 4, o[3] / 4, o[4] / 4, o[5] / 4] {
                table.push(w as u32);
            }
            for &h in &o[9..12] {
                table.push((h / 2) as u32);
            }
            for w in [o[12] / 4, o[13] / 4, o[14] / 4, o[15] / 4] {
                table.push(w as u32);
            }
            let mut projs = Vec::with_capacity(3);
            for (m, p) in e.projs.iter().enumerate() {
                projs.push(MetalTetraProj {
                    name: p.name.clone(),
                    d_out: p.d_out,
                    d_in: p.d_in,
                    nblocks: p.nblocks,
                    tail_w: t[m],
                    row_stride_u32: p.row_stride_u32,
                    words: alias(o[m], it[m].len())?,
                    gscale: alias(o[3 + m], it[3 + m].len())?,
                    rscale: alias(o[6 + m], it[6 + m].len())?,
                    tail: alias(o[9 + m], it[9 + m].len())?,
                    host: keep_host.then(|| HostProj {
                        words: p.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
                        gscale: p.gscale.to_vec(),
                        rscale: p.rscale.clone(),
                        tail: p.tail.clone(),
                    }),
                });
            }
            let rot = |t: &crate::fused::RotationTables, i: usize| -> Result<MetalRotation> {
                Ok(MetalRotation {
                    signbits: alias(o[i], it[i].len())?,
                    small: alias(o[i + 1], it[i + 1].len())?,
                    n: t.n as u32,
                    m: t.m as u32,
                    k: t.k as u32,
                    inv: t.inv,
                    threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
                })
            };
            let rot_in = rot(&e.rot_in, 12)?;
            let rot_down = rot(&e.rot_down, 14)?;
            let down = projs.pop().expect("three");
            let up = projs.pop().expect("three");
            let gate = projs.pop().expect("three");
            out.push(MoeSlabExpert { gate, up, down, rot_in, rot_down });
        }
        let offsets = self.buffer_outside_pool(&table)?;
        Ok((
            MetalMoeSlab {
                _heap: Box::new(MoeHeld(heap)),
                slab,
                offsets,
                experts: experts.len(),
                shape,
                bytes: total as u64,
                records: items.len(),
            },
            out,
        ))
    }

    /// Launch a or c: rotate the selected experts' rows into the basis of
    /// their input rotation (`down == false`, `x` one f16 row `[1, hidden]`
    /// shared by every expert) or of their down rotation (`down == true`, `x`
    /// f16 `[nsel, inter]`, one row an expert). f32 `[nsel, n]`; row r is
    /// what `rotate` returns for slot r's expert alone.
    ///
    /// `sel` is the routing buffer: `nsel` expert ids, ascending and distinct,
    /// then their weights' f32 bits, as u32. The kernel does not bound the
    /// ids: the caller checks them against [`MetalMoeSlab::experts`].
    pub fn moe_rotate(&self, slab: &MetalMoeSlab, down: bool, x: &Tensor, sel: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(sel, &MoeRotOp { rt: self, slab, down })
    }

    /// Launch b: gate and up of the selected experts on their rotated rows
    /// `xr` f32 `[nsel, hidden]`. f32 `[2, nsel, inter]`, gate first; each row
    /// is what `matvec` returns for that expert alone.
    pub fn moe_gate_up(&self, slab: &MetalMoeSlab, xr: &Tensor, sel: &Tensor) -> Result<Tensor> {
        xr.apply_op2_no_bwd(sel, &MoeTetraOp { rt: self, slab, down: false })
    }

    /// Launch d: down of the selected experts on their rotated rows `ar` f32
    /// `[nsel, inter]`, each times its weight, summed in ascending slot order
    /// from +0. f32 `[1, hidden]`.
    pub fn moe_down_sum(&self, slab: &MetalMoeSlab, ar: &Tensor, sel: &Tensor) -> Result<Tensor> {
        ar.apply_op2_no_bwd(sel, &MoeTetraOp { rt: self, slab, down: true })
    }
}

/// The selected count, from the routing buffer `sel` (`2 * nsel` u32).
fn moe_nsel(sel: &MetalStorage, l: &Layout) -> Result<(usize, usize)> {
    if sel.dtype() != DType::U32 {
        candle_core::bail!("moe: the routing buffer is u32, got {:?}", sel.dtype());
    }
    let (start, end) =
        l.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe: a strided routing buffer"))?;
    let len = end - start;
    if len == 0 || !len.is_multiple_of(2) || len / 2 > MOE_MAX_SELECTED {
        candle_core::bail!("moe: a routing buffer of {len} words, which wants 2 x 1..={MOE_MAX_SELECTED}");
    }
    Ok((len / 2, start * 4))
}

fn moe_shape(slab: &MetalMoeSlab) -> Result<MoeShape> {
    slab.shape.ok_or_else(|| candle_core::Error::msg("moe: the experts of this slab differ in shape; no fused path"))
}

struct MoeRotOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    down: bool,
}

impl CustomOp2 for MoeRotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-rot"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe rotation: Metal only; its reference is `rotate`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, sel: &MetalStorage, ls: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        let (nsel, sel_off) = moe_nsel(sel, ls)?;
        if x.dtype() != DType::F16 {
            candle_core::bail!("moe rotation: the rotation reads f16, got {:?}", x.dtype());
        }
        let (start, end) = lx
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("moe rotation: a strided activation"))?;
        let (n, m, k, inv, threads) = if self.down { s.rot_down } else { s.rot_in };
        let n = n as usize;
        let (rows_in, row_stride) = if self.down { (nsel, n) } else { (1, 0) };
        if end - start != rows_in * n {
            candle_core::bail!("moe rotation: {} values for {rows_in} rows of {n}", end - start);
        }
        let dev = x.device().clone();
        let pipe = self.rt.pipeline("moe_rot_metal")?;
        let out = dev.new_buffer(nsel * n, DType::F32, "llvq-moe-rot")?;
        let scratch = dev.new_buffer(nsel * n, DType::F32, "llvq-moe-rot-scratch")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (x.buffer(), 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (sel.buffer(), sel_off));
        set_param(&enc, 5, (&*out, 0usize));
        set_param(&enc, 6, (&*scratch, 0usize));
        set_param(&enc, 7, n as u32);
        set_param(&enc, 8, m);
        set_param(&enc, 9, k);
        set_param(&enc, 10, f32::from_bits(inv));
        set_param(&enc, 11, start as u32);
        set_param(&enc, 12, row_stride as u32);
        set_param(&enc, 13, u32::from(self.down));
        for b in [x.buffer(), &*self.slab.slab, &*self.slab.offsets, sel.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * nsel, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, nsel * n, DType::F32), Shape::from((nsel, n))))
    }
}

struct MoeTetraOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    down: bool,
}

impl CustomOp2 for MoeTetraOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-tetra"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe products: Metal only; their reference is `matvec`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, sel: &MetalStorage, ls: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        let (nsel, sel_off) = moe_nsel(sel, ls)?;
        if x.dtype() != DType::F32 {
            candle_core::bail!("moe products: f32 rows only, got {:?}", x.dtype());
        }
        if !lx.is_contiguous() {
            candle_core::bail!("moe products: the rows must be contiguous");
        }
        let (d_in, d_out, nb, tw, stride) = if self.down {
            (s.inter, s.hidden, s.nb_down, s.tail_down, s.stride_down)
        } else {
            (s.hidden, s.inter, s.nb_in, s.tail_in, s.stride_in)
        };
        if lx.dims() != [nsel, d_in] {
            candle_core::bail!("moe products: rows {:?} for [{nsel}, {d_in}]", lx.dims());
        }
        if !d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("moe products: d_out {d_out} is not a multiple of {} rows", GROUP / LANES);
        }
        let sg = self.down && self.rt.moe_down_sg();
        let (kname, out_len, shape) = if sg {
            ("moe_tetra_down_sg_metal", d_out, Shape::from((1, d_out)))
        } else if self.down {
            ("moe_tetra_down_metal", d_out, Shape::from((1, d_out)))
        } else {
            ("moe_tetra_gu_metal", 2 * nsel * d_out, Shape::from((2, nsel, d_out)))
        };
        let dev = x.device().clone();
        let pipe = self.rt.pipeline(kname)?;
        let out = dev.new_buffer(out_len, DType::F32, "llvq-moe-tetra")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        let t = &self.rt.tables;
        set_param(&enc, 0, (&*self.slab.slab, 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (sel.buffer(), sel_off));
        set_param(&enc, 5, (&*t.rows, 0usize));
        set_param(&enc, 6, (&*t.prefixes, 0usize));
        set_param(&enc, 7, (&*t.branches, 0usize));
        set_param(&enc, 8, (&*t.suffixes, 0usize));
        set_param(&enc, 9, (&*t.invnorm, 0usize));
        set_param(&enc, 10, (x.buffer(), lx.start_offset() * DType::F32.size_in_bytes()));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, stride as u32);
        set_param(&enc, 13, nb as u32);
        set_param(&enc, 14, tw as u32);
        set_param(&enc, 15, d_out as u32);
        set_param(&enc, 16, nsel as u32);
        if !sg {
            enc.set_threadgroup_memory_length(0, self.rt.tile * llvq_core::DIM * DType::F32.size_in_bytes());
        }
        for b in [&*self.slab.slab, &*self.slab.offsets, sel.buffer(), &*t.rows, &*t.prefixes, &*t.branches, &*t.suffixes, &*t.invnorm, x.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        if sg {
            // One threadgroup an output row, one simdgroup a selected slot.
            let tg = nsel * LANES;
            if tg > pipe.max_total_threads_per_threadgroup() {
                candle_core::bail!("moe_tetra_down_sg_metal: {nsel} slots want {tg} threads, the pipeline allows {}",
                    pipe.max_total_threads_per_threadgroup());
            }
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: d_out * tg, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: tg, height: 1, depth: 1 },
            );
        } else {
            let rows = if self.down { d_out } else { 2 * nsel * d_out };
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: rows * LANES, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
            );
        }
        Ok((MetalStorage::new(out, dev, out_len, DType::F32), shape))
    }
}

// ---------------------------------------------------------------------------
// A prompt's rows through the resident experts of a layer in a few launches
// (the Metal twin of `resident_cuda.rs`'s `forward_rows_batched`).
//
// `ResidentExperts::forward_general` runs, for each expert that received
// tokens, a gather, an upload of its ids and weights, a rotation, gate and up
// in chunks of 8 rows through `tv_tetra48_metal_lutdrf`, candle's activation,
// a rotation, down in chunks, a product by the weights and an `index_add`:
// about 35 launches an expert, 4,500 a layer visit for a piece of 512 tokens.
// Here every (token, expert) pair is a row, sorted by expert then token, and
// a visit is:
//   a. `moe_rot_metal` (the one-row path's kernel) over all the pairs, each
//      row taking the input rotation of its own expert;
//   b. `moe_rows_lutdrf`: gate and up of every chunk of at most 8 rows of one
//      expert, the body of `tv_tetra48_metal_lutdrf` with its pointers taken
//      from the slab;
//   c. candle's activation on all the rows, then `moe_rot_metal` again with
//      the down rotations;
//   d. `moe_rows_lutdrf` for down;
//   e. `moe_pairs_sum`: each token's rows times their weights, added in
//      ascending expert order from +0, the `mul` and the `add` rounded apart.
// A row's value does not depend on the other rows of its chunk or launch, the
// chunks are the general path's (consecutive rows of one expert, at most 8),
// and the sum runs the general path's additions in its order. So the values
// are meant to be its own, bit for bit; `tests/inkling_moe_rows.rs` demands
// it. The text is appended to the served unit (its `fp contract(off)`
// covers it) and compiled as a library of its own.
// ---------------------------------------------------------------------------

macro_rules! moe_rows_msl {
    () => {
        r#"
// moe_rows_lutdrf: simdgroup g computes output row `row` of matrix `mat0 + mi`
// for the rows of chunk `ci`: g = (mi * nchunks + ci) * d_out + row. A chunk
// is (expert, first pair row, rows), rows 1..=TETRA48_PREFILL_MAX. d_out is a
// multiple of 8, so the 8 simdgroups of a threadgroup share one chunk and one
// row count. The loop and the epilogue are `tv_tetra48_metal_lutdrf`'s text.
kernel void moe_rows_lutdrf(const device uint*         slab           [[buffer(0)]],
                            const device float*        slabf          [[buffer(1)]],
                            const device half*         slabh          [[buffer(2)]],
                            const device MoeExpertOff* off            [[buffer(3)]],
                            const device uint*         chunks         [[buffer(4)]],
                            const device uint*         rows           [[buffer(5)]],
                            const device uchar*        prefixes       [[buffer(6)]],
                            const device ushort*       branches       [[buffer(7)]],
                            const device uchar*        suffixes       [[buffer(8)]],
                            const device float*        invnorm        [[buffer(9)]],
                            const device float*        xall           [[buffer(10)]],
                            device float*              yall           [[buffer(11)]],
                            constant uint&             row_stride_u32 [[buffer(12)]],
                            constant uint&             nblocks        [[buffer(13)]],
                            constant uint&             tail_w         [[buffer(14)]],
                            constant uint&             d_out          [[buffer(15)]],
                            constant uint&             nchunks        [[buffer(16)]],
                            constant uint&             mat0           [[buffer(17)]],
                            constant uint&             npairs         [[buffer(18)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint gg = gid >> 5;
    uint per_mat = nchunks * d_out;
    uint mi = gg / per_mat;
    uint rem = gg - mi * per_mat;
    uint ci = rem / d_out;
    uint row = rem - ci * d_out;
    uint e = chunks[3u * ci];
    uint first = chunks[3u * ci + 1u];
    uint nrows = chunks[3u * ci + 2u];
    uint mat = mat0 + mi;
    MoeExpertOff o = off[e];
    const device uint* words = slab + o.words[mat];
    const device float* gscale = slabf + o.gscale[mat];
    const device float* rscale = slabf + o.rscale[mat];
    const device half* tail = slabh + o.tail[mat];
    uint d_in = nblocks * 24u + tail_w;
    const device float* x = xall + (ulong)first * d_in;
    device float* y = yall + ((ulong)mi * npairs + first) * d_out;

    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[TETRA48_PREFILL_MAX];
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o8 = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o8 << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
            if (r < nrows) {
                const device float* xb = x + r * d_in + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) {
                    a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                }
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue(acc[r], row, lane, rscale, tail, x + r * d_in, y + r * d_out,
                                  nblocks, tail_w);
        }
    }
}

/// `+0.0f + p` with the IEEE sign rule spelled in integers (see
/// `moe_zero_plus` of the one-row unit).
inline float moe_rows_zero_plus(float p)
{
    return as_type<uint>(p) == 0x80000000u ? 0.0f : p;
}

// moe_pairs_sum: out[t][c] = sum over the token's k rows, in ascending expert
// order, of y[pair][c] * w, from +0. `tp` holds, a token, its k pair rows in
// that order; `tw` their weights.
kernel void moe_pairs_sum(const device float* y    [[buffer(0)]],
                          const device uint*  tp   [[buffer(1)]],
                          const device float* tw   [[buffer(2)]],
                          device float*       out  [[buffer(3)]],
                          constant uint&      k    [[buffer(4)]],
                          constant uint&      h    [[buffer(5)]],
                          constant uint&      t    [[buffer(6)]],
                          uint2 gp [[thread_position_in_grid]])
{
    uint c = gp.x;
    uint tok = gp.y;
    if (c >= h || tok >= t) { return; }
    float acc = 0.0f;
    for (uint kk = 0u; kk < k; ++kk) {
        float p = y[(ulong)tp[tok * k + kk] * h + c] * tw[tok * k + kk];
        acc = kk == 0u ? moe_rows_zero_plus(p) : acc + p;
    }
    out[(ulong)tok * h + c] = acc;
}
"#
    };
}

const SRC_MOE_ROWS: &str = concat!(
    include_str!("../kernels/llvq_tetra48.metal"),
    include_str!("../kernels/llvq_tetra48_rows.metal"),
    moe_off_msl!(),
    moe_rows_msl!()
);

/// Which matrices one launch of `moe_rows_lutdrf` computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeRowsMats {
    /// Gate and up: `[2, P, inter]` from rows `[P, hidden]`.
    GateUp,
    /// Down: `[1, P, hidden]` from rows `[P, inter]`.
    Down,
}

impl MetalRuntime {
    /// Whether [`Self::moe_rows`] and [`Self::moe_rotate_pairs`] reproduce the
    /// general path of `ResidentExperts`: the one-row fused path's conditions,
    /// and the rows kernel in use (`prefill_rows` at least 2).
    pub fn moe_rows_eligible(&self) -> bool {
        self.moe_fused_eligible() && self.prefill_rows >= 2
    }

    /// Rows `x` f16 `[P, n]` into the basis of their own expert's input
    /// rotation (`down == false`, n = hidden) or down rotation (`down ==
    /// true`, n = inter); `pair_expert` u32 `[P]`, the expert of each row,
    /// checked by the caller against the slab. f32 `[P, n]`; row r is what
    /// `rotate` returns for that row and expert alone.
    pub fn moe_rotate_pairs(&self, slab: &MetalMoeSlab, down: bool, x: &Tensor, pair_expert: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(pair_expert, &MoeRotPairsOp { rt: self, slab, down })
    }

    /// Gate and up, or down, of every chunk of `chunks` (u32 `[3 C]`: expert,
    /// first row, rows; rows 1..=`prefill_rows`) on the rotated rows `x` f32
    /// `[P, d_in]`: f32 `[2, P, inter]` or `[1, P, hidden]`. Each row is what
    /// `matvec_rows_f32` returns for it in its chunk on that expert alone.
    pub fn moe_rows(&self, slab: &MetalMoeSlab, mats: MoeRowsMats, x: &Tensor, chunks: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(chunks, &MoeRowsOp { rt: self, slab, mats })
    }

    /// Each token's `k` rows of `y` f32 `[P, h]` (pair rows `tp` u32 `[T k]`,
    /// ascending expert order) times their weights `tw` f32 `[T k]`, added from
    /// +0: f32 `[T, h]`.
    pub fn moe_pairs_sum(&self, y: &Tensor, tp: &Tensor, tw: &Tensor, k: usize) -> Result<Tensor> {
        let (p, h) = y.dims2()?;
        let tk = tp.elem_count();
        if k == 0 || !tk.is_multiple_of(k) || tw.elem_count() != tk || p == 0 {
            candle_core::bail!("moe pairs sum: {tk} pair rows, {} weights, k = {k}, {p} rows", tw.elem_count());
        }
        let op = MoePairsSumOp { rt: self, k, h, t: tk / k };
        let yt = y.contiguous()?;
        let tp = tp.contiguous()?;
        let tw = tw.contiguous()?;
        yt.apply_op3_no_bwd(&tp, &tw, &op)
    }
}

struct MoeRotPairsOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    down: bool,
}

impl CustomOp2 for MoeRotPairsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-rot-pairs"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe pair rotation: Metal only; its reference is `rotate`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, pe: &MetalStorage, lp: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        if x.dtype() != DType::F16 || pe.dtype() != DType::U32 {
            candle_core::bail!("moe pair rotation: f16 rows and u32 experts, got {:?} and {:?}", x.dtype(), pe.dtype());
        }
        let (start, end) =
            lx.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe pair rotation: a strided activation"))?;
        let (pstart, pend) =
            lp.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe pair rotation: a strided expert list"))?;
        let npairs = pend - pstart;
        let (n, m, k, inv, threads) = if self.down { s.rot_down } else { s.rot_in };
        let n = n as usize;
        if npairs == 0 || end - start != npairs * n {
            candle_core::bail!("moe pair rotation: {} values for {npairs} rows of {n}", end - start);
        }
        let dev = x.device().clone();
        let pipe = self.rt.pipeline("moe_rot_metal")?;
        let out = dev.new_buffer(npairs * n, DType::F32, "llvq-moe-rot-pairs")?;
        let scratch = dev.new_buffer(npairs * n, DType::F32, "llvq-moe-rot-pairs-scratch")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (x.buffer(), 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (pe.buffer(), pstart * 4));
        set_param(&enc, 5, (&*out, 0usize));
        set_param(&enc, 6, (&*scratch, 0usize));
        set_param(&enc, 7, n as u32);
        set_param(&enc, 8, m);
        set_param(&enc, 9, k);
        set_param(&enc, 10, f32::from_bits(inv));
        set_param(&enc, 11, start as u32);
        set_param(&enc, 12, n as u32);
        set_param(&enc, 13, u32::from(self.down));
        for b in [x.buffer(), &*self.slab.slab, &*self.slab.offsets, pe.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * npairs, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, npairs * n, DType::F32), Shape::from((npairs, n))))
    }
}

struct MoeRowsOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    mats: MoeRowsMats,
}

impl CustomOp2 for MoeRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe rows: Metal only; their reference is `matvec_rows_f32`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, ch: &MetalStorage, lc: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        if x.dtype() != DType::F32 || ch.dtype() != DType::U32 {
            candle_core::bail!("moe rows: f32 rows and u32 chunks, got {:?} and {:?}", x.dtype(), ch.dtype());
        }
        if !lx.is_contiguous() {
            candle_core::bail!("moe rows: the rows must be contiguous");
        }
        let (cstart, cend) =
            lc.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe rows: a strided chunk table"))?;
        if cend == cstart || !(cend - cstart).is_multiple_of(3) {
            candle_core::bail!("moe rows: a chunk table of {} words", cend - cstart);
        }
        let nchunks = (cend - cstart) / 3;
        let (d_in, d_out, nb, tw, stride, mat0, nmat) = match self.mats {
            MoeRowsMats::GateUp => (s.hidden, s.inter, s.nb_in, s.tail_in, s.stride_in, 0u32, 2usize),
            MoeRowsMats::Down => (s.inter, s.hidden, s.nb_down, s.tail_down, s.stride_down, 2u32, 1usize),
        };
        let dims = lx.dims();
        if dims.len() != 2 || dims[1] != d_in || dims[0] == 0 {
            candle_core::bail!("moe rows: rows {dims:?} for [P, {d_in}]");
        }
        let npairs = dims[0];
        if !d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("moe rows: d_out {d_out} is not a multiple of {} rows", GROUP / LANES);
        }
        let dev = x.device().clone();
        let pipe = self.rt.pipeline("moe_rows_lutdrf")?;
        let out_len = nmat * npairs * d_out;
        let out = dev.new_buffer(out_len, DType::F32, "llvq-moe-rows")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        let t = &self.rt.tables;
        set_param(&enc, 0, (&*self.slab.slab, 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (ch.buffer(), cstart * 4));
        set_param(&enc, 5, (&*t.rows, 0usize));
        set_param(&enc, 6, (&*t.prefixes, 0usize));
        set_param(&enc, 7, (&*t.branches, 0usize));
        set_param(&enc, 8, (&*t.suffixes, 0usize));
        set_param(&enc, 9, (&*t.invnorm, 0usize));
        set_param(&enc, 10, (x.buffer(), lx.start_offset() * DType::F32.size_in_bytes()));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, stride as u32);
        set_param(&enc, 13, nb as u32);
        set_param(&enc, 14, tw as u32);
        set_param(&enc, 15, d_out as u32);
        set_param(&enc, 16, nchunks as u32);
        set_param(&enc, 17, mat0);
        set_param(&enc, 18, npairs as u32);
        for b in [&*self.slab.slab, &*self.slab.offsets, ch.buffer(), &*t.rows, &*t.prefixes, &*t.branches, &*t.suffixes, &*t.invnorm, x.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: nmat * nchunks * d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, out_len, DType::F32), Shape::from((nmat, npairs, d_out))))
    }
}

struct MoePairsSumOp<'a> {
    rt: &'a MetalRuntime,
    k: usize,
    h: usize,
    t: usize,
}

impl candle_core::CustomOp3 for MoePairsSumOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-pairs-sum"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe pairs sum: Metal only; its reference is the general path's index_add")
    }

    fn metal_fwd(
        &self,
        y: &MetalStorage,
        ly: &Layout,
        tp: &MetalStorage,
        lp: &Layout,
        tw: &MetalStorage,
        lw: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if y.dtype() != DType::F32 || tp.dtype() != DType::U32 || tw.dtype() != DType::F32 {
            candle_core::bail!("moe pairs sum: f32 rows, u32 pairs, f32 weights");
        }
        let off = |l: &Layout| l.start_offset() * 4;
        let dev = y.device().clone();
        let pipe = self.rt.pipeline("moe_pairs_sum")?;
        let n = self.t * self.h;
        let out = dev.new_buffer(n, DType::F32, "llvq-moe-pairs-sum")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (y.buffer(), off(ly)));
        set_param(&enc, 1, (tp.buffer(), off(lp)));
        set_param(&enc, 2, (tw.buffer(), off(lw)));
        set_param(&enc, 3, (&*out, 0usize));
        set_param(&enc, 4, self.k as u32);
        set_param(&enc, 5, self.h as u32);
        set_param(&enc, 6, self.t as u32);
        for b in [y.buffer(), tp.buffer(), tw.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: self.h, height: self.t, depth: 1 },
            objc2_metal::MTLSize { width: 32, height: 8, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, n, DType::F32), Shape::from((self.t, self.h))))
    }
}

#[cfg(test)]
mod tests {
    use super::{h2f, kernel_named, MetalTuning, Q4_TILE_COLS, SERVED_TETRA, TETRA_PREFILL_MAX};
    use crate::served::MetalKernels;

    /// The default tuning is the served path of every config before
    /// 2026-11-26, and the Metal configs' `lutd+h+pq` is the bundle arm of
    /// `metal-nuit-2026-11-06` (M2, M8, M10) and nothing else: the matvec
    /// `lutd`, the half-io twins, the Tetra and int4 prefill kernels at 8
    /// rows, the served int4 tile, no prefetch, no threadgroup rotation.
    #[test]
    fn the_served_metal_kernels_are_the_measured_arm() {
        let d = MetalTuning::default();
        assert_eq!(
            d,
            MetalTuning {
                q4_tile_cols: Q4_TILE_COLS,
                tetra_kernel: SERVED_TETRA,
                q4_prefetch: false,
                half_io: false,
                prefill_rows: 1,
                q4_rows: false,
                rot_tg: false,
            }
        );
        assert_eq!(MetalTuning::of_served(MetalKernels::Default).unwrap(), d);
        assert_eq!(MetalTuning::parse_arm("lut").unwrap(), d);
        let t = MetalTuning::of_served(MetalKernels::LutdHPq).unwrap();
        assert_eq!(
            t,
            MetalTuning {
                tetra_kernel: "tv_tetra48_metal_lutd",
                half_io: true,
                prefill_rows: TETRA_PREFILL_MAX,
                q4_rows: true,
                ..d
            }
        );
        for name in MetalKernels::NAMES {
            let k = MetalKernels::parse(Some(name)).unwrap();
            assert_eq!(MetalTuning::of_served(k).unwrap(), MetalTuning::parse_arm(name).unwrap(), "{name}");
        }
        // The bench's other spellings still read, and a form is not a kernel.
        assert!(MetalTuning::parse_arm("lutd+h+pq+r").unwrap().rot_tg);
        assert_eq!(MetalTuning::parse_arm("lut+t8192").unwrap().q4_tile_cols, 8192);
        for bad in ["lutd+grp", "lutd+ker", "lutx", "lutd+zz", "lutd+t", "lutd+t100", "lutp+h"] {
            assert!(MetalTuning::parse_arm(bad).is_err(), "{bad:?} must be refused");
        }
    }

    /// Unset and `lut` serve the served matvec, each A/B name maps to its own
    /// kernel, and an unknown name is refused by name instead of falling back
    /// to the served kernel under a label it did not run.
    #[test]
    fn an_unknown_metal_kernel_is_refused() {
        assert_eq!(kernel_named(None).unwrap(), "tv_tetra48_metal_lut");
        assert_eq!(kernel_named(Some("lut")).unwrap(), "tv_tetra48_metal_lut");
        assert_eq!(kernel_named(Some("plain")).unwrap(), "tv_tetra48_metal");
        assert_eq!(kernel_named(Some("c3")).unwrap(), "tv_tetra48_metal_c3");
        // `ilp` names an 8-lane kernel, which the 32-lane grid would run past
        // the output.
        for bad in ["lutd", "", "LUT", "ilp"] {
            let e = kernel_named(Some(bad)).expect_err(bad).to_string();
            assert!(e.contains("LLVQ_METAL_KERNEL"), "{bad:?}: {e}");
        }
    }

    /// Every name the variable accepts, `validate` accepts too: the variable
    /// does not pass through it.
    #[test]
    fn every_kernel_the_variable_names_passes_validate() {
        for v in [None, Some("lut"), Some("c3"), Some("ar"), Some("u2"), Some("plain")] {
            let tetra_kernel = kernel_named(v).unwrap();
            super::MetalTuning { tetra_kernel, ..Default::default() }
                .validate()
                .unwrap_or_else(|e| panic!("{v:?} names {tetra_kernel}, which validate refuses: {e}"));
        }
    }

    /// The IEEE definition of binary16, in f64, from the fields.
    ///
    /// Not a second copy of `h2f`: this computes the VALUE from the exponent
    /// and the significand, where `h2f` assembles a bit pattern. A shared
    /// mistake would have to be a mistake about IEEE itself.
    fn ieee_half(h: u16) -> Option<f64> {
        let s = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f64;
        match e {
            // Subnormal: no hidden bit, fixed exponent of -14.
            0 => Some(s * (m / 1024.0) * 2f64.powi(-14)),
            // inf / NaN, which the caller checks separately.
            31 => None,
            _ => Some(s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)),
        }
    }

    /// All 65,536 patterns. The subnormal and inf arms were both wrong until
    /// an audit of 2026-09-21, and the gate's Gaussian fixtures never drew one.
    #[test]
    fn h2f_widens_every_binary16_pattern() {
        let mut checked = 0u32;
        for b in 0u32..=0xffff {
            let h = b as u16;
            let got = h2f(h);
            match ieee_half(h) {
                Some(want) => {
                    assert_eq!(
                        got,
                        want as f32,
                        "0x{b:04x}: h2f {got:e} against the IEEE value {want:e}"
                    );
                    checked += 1;
                }
                None => {
                    // exp == 31: infinity when the significand is zero, NaN otherwise.
                    if h & 0x3ff == 0 {
                        assert!(got.is_infinite(), "0x{b:04x} is an infinity");
                        assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "0x{b:04x} sign");
                    } else {
                        assert!(got.is_nan(), "0x{b:04x} is a NaN");
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 65_536, "every pattern is judged");
    }

    /// The three patterns the earlier version got wrong, named.
    #[test]
    fn the_patterns_an_earlier_h2f_got_wrong() {
        // The smallest positive subnormal: 2^-24.
        assert_eq!(h2f(0x0001), 2f32.powi(-24));
        // The largest subnormal: 1023 * 2^-24.
        assert_eq!(h2f(0x03ff), 1023.0 * 2f32.powi(-24));
        // Positive infinity, which an earlier version returned as 65536.
        assert!(h2f(0x7c00).is_infinite() && h2f(0x7c00) > 0.0);
    }
}
