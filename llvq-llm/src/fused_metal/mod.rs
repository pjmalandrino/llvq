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
//!
//! ## The files
//!
//! | file | holds |
//! |---|---|
//! | this file | the kernel texts and their entry points, [`MetalRuntime`], [`MetalTables`], [`MetalTetraProj`] |
//! | `tuning` | [`MetalTuning`], the Tetra kernel names and row loops, the int4 tile, the threadgroup checks |
//! | `runtime` | the runtime's core methods, the Tetra matvec launches, the host oracle |
//! | `rot` | [`MetalRotation`] and the Walsh-Hadamard launch |
//! | `int4` | [`MetalInt4Proj`], its upload and its launches |
//! | `embed` | [`MetalEmbedTable`], the token gather and the lm_head |
//! | `port` | the `crate::device` port: the lattice and int4 projections the model sees |
//! | `load` | [`load_resolved`], the one door from a resolved file to the model's projections |
//! | `wave1` | the resident buffers, the f32 prefill rows kernel, the f32-input int4 head |
//! | `moe` | [`MetalMoeSlab`] and one decoded token through the routed experts |
//! | `moe_rows` | a prompt's rows through the resident experts |
//!
//! Each child imports what it uses by name. This file re-exports only what
//! the rest of the workspace names, plus the types its public signatures carry.

use candle_core::Result;
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

mod tuning;
mod runtime;
mod rot;
mod int4;
mod embed;
mod port;
mod load;
mod wave1;
mod moe;
mod moe_rows;

pub use tuning::{tuning_of, MetalTuning, Q4_TILE_COLS, SERVED_TETRA, TETRA_ROW_LOOPS};
pub use rot::MetalRotation;
pub use int4::MetalInt4Proj;
pub use embed::MetalEmbedTable;
pub use load::{load_resolved, load_resolved_tuned};
pub use moe::{MetalMoeSlab, MoeHostExpert, MoeHostProj, MoeShape, MoeSlabExpert, MOE_DOWN_SG_DEFAULT, MOE_MAX_SELECTED};
pub use moe_rows::MoeRowsMats;
use moe::{SRC_MOE_ROT, SRC_MOE_TETRA};
use moe_rows::SRC_MOE_ROWS;
use wave1::{SRC_E4_XF, SRC_TETRA_ROWS_F32};

#[cfg(test)]
mod tests;

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

const SRC_TETRA: &str = include_str!("../../kernels/llvq_tetra48.metal");
const SRC_ROT: &str = include_str!("../../kernels/llvq_rot.metal");
const SRC_Q4: &str = include_str!("../../kernels/tv_q4_h.metal");
const SRC_Q8: &str = include_str!("../../kernels/emb_q8.metal");
const SRC_E4: &str = include_str!("../../kernels/emb_q4.metal");
/// The row-loop twins of the Tetra matvec: the served unit's text, then the
/// new kernels. Compiled as its own library, so the served pipelines, which
/// compile from `SRC_TETRA` alone, keep their register allocation.
const SRC_TETRA_ROWS: &str =
    concat!(include_str!("../../kernels/llvq_tetra48.metal"), include_str!("../../kernels/llvq_tetra48_rows.metal"));
/// The prefetching twins of the int4 matvecs, on the same rule.
const SRC_Q4_PF: &str = concat!(include_str!("../../kernels/tv_q4_h.metal"), include_str!("../../kernels/tv_q4_pf.metal"));

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
