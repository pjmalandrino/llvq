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
/// against it, so a machine that offered less would have to be refused rather
/// than quietly served a kernel that stages past its limit. Asking the device
/// and adapting would hide exactly that.
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
}

impl Default for MetalTuning {
    fn default() -> Self {
        Self {
            q4_tile_cols: Q4_TILE_COLS,
            tetra_kernel: SERVED_TETRA,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
        }
    }
}

/// Rows the Tetra prefill kernel can take a launch (`TETRA48_PREFILL_MAX`).
pub const TETRA_PREFILL_MAX: usize = 8;

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
        if !self.tetra_kernel.starts_with("tv_tetra48_metal") || self.tetra_kernel == "tv_tetra48_metal_ilp" {
            candle_core::bail!(
                "{} is not a 32-lane Tetra matvec with the served arguments",
                self.tetra_kernel
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
        Ok(())
    }
}

/// The half-store twin of a Tetra matvec, if it has one.
pub fn tetra_half_twin(name: &str) -> Option<&'static str> {
    match name {
        "tv_tetra48_metal_lut" => Some("tv_tetra48_metal_luth"),
        "tv_tetra48_metal_lutd" => Some("tv_tetra48_metal_lutdh"),
        _ => None,
    }
}

/// The row-loop twins of the served Tetra matvec (`kernels/llvq_tetra48_rows.metal`).
/// Each has the served kernel's arguments and, by gate, its bits.
pub const TETRA_ROW_LOOPS: [&str; 5] = [
    "tv_tetra48_metal_lutp",
    "tv_tetra48_metal_lutd",
    "tv_tetra48_metal_lutdp",
    "tv_tetra48_metal_lutd2",
    "tv_tetra48_metal_lutd4",
];

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
        "rot_apply_metal" | "rot_apply_rows_metal" => Ok(SRC_ROT),
        "tv_q4_metal" | "tv_q4_metal_tiled" => Ok(SRC_Q4),
        "tv_tetra48_metal_lutp" | "tv_tetra48_metal_lutd" | "tv_tetra48_metal_lutdp" | "tv_tetra48_metal_lutd2"
        | "tv_tetra48_metal_lutd4" | "tv_tetra48_metal_luth" | "tv_tetra48_metal_lutdh"
        | "tv_tetra48_metal_lutdr" => Ok(SRC_TETRA_ROWS),
        "tv_q4_metal_p" | "tv_q4_metal_tiled_p" | "tv_q4_metal_h" | "tv_q4_metal_tiled_h" => Ok(SRC_Q4_PF),
        "emb_q8_gather_metal" | "tv_q8_metal" => Ok(SRC_Q8),
        "emb_q4_gather_metal" | "tv_emb_q4_metal" => Ok(SRC_E4),
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
fn kernel_from_env() -> &'static str {
    const SERVED: &str = "tv_tetra48_metal_lut";
    match std::env::var("LLVQ_METAL_KERNEL").ok().as_deref() {
        None => SERVED,
        Some("lut") => SERVED,
        Some("c3") => "tv_tetra48_metal_c3",
        Some("ar") => "tv_tetra48_metal_ar",
        Some("u2") => "tv_tetra48_metal_u2",
        Some("ilp") => "tv_tetra48_metal_ilp",
        Some("plain") => "tv_tetra48_metal",
        Some(other) => {
            eprintln!("LLVQ_METAL_KERNEL={other:?} is not a matvec of this file; serving {SERVED}");
            SERVED
        }
    }
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
            kernel: kernel_from_env(),
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
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
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
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
            width: GROUP,
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
        let kname = if batched { "rot_apply_rows_metal" } else { "rot_apply_metal" };
        let pipe = self.rt.pipeline(kname)?;
        let dev = storage.device().clone();

        let out = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot")?;
        // The staging, in DEVICE memory. See the section header.
        let scratch = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot-scratch")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
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
        let threads = self.rot.threads;
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
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
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
        let pipe = self.rt.pipeline(match t.bits {
            4 => "emb_q4_gather_metal",
            _ => "emb_q8_gather_metal",
        })?;
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
            objc2_metal::MTLSize { width: n_ids * GROUP, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
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
        let pipe = self.rt.pipeline(match t.bits {
            4 => "tv_emb_q4_metal",
            _ => "tv_q8_metal",
        })?;
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
                objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
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
            kernel: kernel_from_env(),
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
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
         untiled | int4 row loop {} | io {} | prefill {}",
        t.tetra_kernel,
        int4_records,
        t.q4_tile_cols,
        int4_records - int4_tiled,
        if t.q4_prefetch { "prefetching (tv_q4_pf.metal)" } else { "served" },
        if t.half_io { "half (in-kernel casts, tv_q4_pf.metal and llvq_tetra48_rows.metal)" } else { "f32 (served)" },
        match t.prefill_rows {
            1 => "row by row (served)".to_string(),
            n => format!("{n} rows a Tetra launch (tv_tetra48_metal_lutdr)"),
        },
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

#[cfg(test)]
mod tests {
    use super::h2f;

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
