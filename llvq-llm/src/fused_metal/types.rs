//! The device state the children share: [`MetalRuntime`], [`MetalTables`], [`MetalTetraProj`], their host
//! copies, the pipeline cache key, and the composed table the decoder tables are uploaded with. Types only; their
//! methods live with the launches that use them.

use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Read by intra-doc links only; rustc never compiles it, so it cannot go unused.
#[cfg(doc)]
use super::MetalTuning;

/// The pipeline cache, keyed on everything that changes the compiled code.
///
/// The tile is in the key and must stay there: it is injected as a text
/// prelude, and a define that fails to land changes how much threadgroup
/// memory the host must stage while leaving the arithmetic alone. That is the
/// failure a cache keyed on the function name alone produces, silently.
pub(super) type Key = (&'static str, usize);

/// What the decoder needs, uploaded once and shared by every projection.
///
/// 18,688 bytes: rows 4,096 u32, branches 1,024 u16, prefixes 128 u8,
/// suffixes 128 u8. Held as `Arc<Buffer>` for the reason the header gives:
/// released, they become somebody else's scratch.
pub struct MetalTables {
    pub(super) rows: Arc<Buffer>,
    pub(super) prefixes: Arc<Buffer>,
    pub(super) branches: Arc<Buffer>,
    pub(super) suffixes: Arc<Buffer>,
    /// `1/sqrt(16 m)`, entry 0 the origin.
    pub(super) invnorm: Arc<Buffer>,
    /// `suffixes[2 * ((branches[16*s8 + b2] >> 8) & 63) + b3]`, precomposed.
    ///
    /// 64 x 16 x 2 = 2,048 bytes. It holds no new information: it is the two
    /// tables it is built from, composed, so the value is unchanged. What it
    /// removes is the DEPENDENCY, which was the second load waiting on the
    /// first.
    pub(super) c3_of: Arc<Buffer>,
    /// Host copies, kept only so `cpu_fwd` can serve as the oracle. `None` on
    /// a served model, where the CPU arm refuses instead.
    pub(super) host: Option<HostTables>,
}

pub(super) struct HostTables {
    pub(super) rows: Vec<u32>,
    pub(super) prefixes: Vec<u8>,
    pub(super) branches: Vec<u16>,
    pub(super) suffixes: Vec<u8>,
    pub(super) invnorm: Vec<f32>,
}

/// One Tetra projection on the device.
pub struct MetalTetraProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub(super) nblocks: usize,
    pub(super) tail_w: usize,
    pub(super) row_stride_u32: usize,
    pub(super) words: Arc<Buffer>,
    pub(super) gscale: Arc<Buffer>,
    pub(super) rscale: Arc<Buffer>,
    pub(super) tail: Arc<Buffer>,
    pub(super) host: Option<HostProj>,
}

pub(super) struct HostProj {
    pub(super) words: Vec<u8>,
    pub(super) gscale: Vec<f32>,
    pub(super) rscale: Vec<f32>,
    pub(super) tail: Vec<u16>,
}

/// The device, the tables and the compiled pipelines.
pub struct MetalRuntime {
    /// The matvec this runtime launches. See `metal_fwd`.
    pub(super) kernel: &'static str,
    /// Columns a tile for the int4 records past the wall.
    pub(super) q4_tile_cols: usize,
    /// Whether the int4 matvecs are the prefetching twins.
    pub(super) q4_prefetch: bool,
    /// Whether the matvecs are the half-io twins ([`MetalTuning::half_io`]).
    pub(super) half_io: bool,
    /// Prompt rows a Tetra prefill launch ([`MetalTuning::prefill_rows`]).
    pub(super) prefill_rows: usize,
    /// Whether the int4 records take the prefill chunks ([`MetalTuning::q4_rows`]).
    pub(super) q4_rows: bool,
    /// Whether one-row rotations that fit stage in threadgroup memory ([`MetalTuning::rot_tg`]).
    pub(super) rot_tg: bool,
    /// Whether the fused MoE call's `down` runs one simdgroup a slot
    /// (`moe_tetra_down_sg_metal`): the same bits ([`Self::set_moe_down_sg`]).
    pub(super) moe_down_sg: std::sync::atomic::AtomicBool,
    pub(super) device: candle_core::MetalDevice,
    pub(super) tables: MetalTables,
    pub(super) tile: usize,
    pub(super) pipelines: RwLock<HashMap<Key, ComputePipeline>>,
}

/// The suffix table composed with the branch table that indexes it.
///
/// `suffixes` is read at `2 * s16 + b3` where `s16` is the high byte of
/// `branches[16 * s8 + b2]`, masked to the 64 Golay states. So the composition
/// is indexed by `(s8, b2, b3)` and holds exactly what the two-step read
/// would have returned.
pub(super) fn compose_c3(branches: &[u16], suffixes: &[u8]) -> Vec<u8> {
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
