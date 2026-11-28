//! Wave 2 of the Inkling 2 Mini throughput plan, item D3: the routed experts of a layer in one placement heap,
//! and one decoded token through them in four launches. One file because the slab's layout, its MSL offset
//! record and the launches that index it are one contract.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::launch::{GROUP, LANES};
use super::types::{HostProj, MetalRuntime, MetalTetraProj};
use super::{MetalRotation, SERVED_TETRA};

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
pub(super) use moe_off_msl;

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

pub(super) const SRC_MOE_ROT: &str = concat!(include_str!("../../kernels/llvq_rot.metal"), moe_off_msl!(), moe_rot_msl!());
pub(super) const SRC_MOE_TETRA: &str = concat!(include_str!("../../kernels/llvq_tetra48.metal"), moe_off_msl!(), moe_tetra_msl!());

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
    pub(super) nb_in: usize,
    pub(super) nb_down: usize,
    pub(super) tail_in: usize,
    pub(super) tail_down: usize,
    pub(super) stride_in: usize,
    pub(super) stride_down: usize,
    pub(super) rot_in: (u32, u32, u32, u32, usize),
    pub(super) rot_down: (u32, u32, u32, u32, usize),
}

/// Owns a Metal object whose type this crate cannot name (`objc2` is not
/// one of its dependencies), so the object lives as long as its owner.
trait MoeHold: Send + Sync {}
struct MoeHeld<T>(#[allow(dead_code)] T);
// SAFETY: the field is private and never read, so the only thing another
// thread can do with a `MoeHeld` is drop it, which releases the object. The
// impl is generic, but the one construction (`upload_moe_layer`) holds the
// placement heap, a retained Objective-C object whose release is atomic and
// may run on any thread. A `MoeHeld` of a type whose drop is tied to a
// thread would make this unsound; nothing builds one.
unsafe impl<T> Send for MoeHeld<T> {}
// SAFETY: as for `Send`: a shared `&MoeHeld` gives access to nothing, since
// the field is private and unread.
unsafe impl<T> Sync for MoeHeld<T> {}
impl<T> MoeHold for MoeHeld<T> {}

/// A layer's experts in one placement heap, and the offset table.
pub struct MetalMoeSlab {
    // Held so the heap outlives every alias, whatever the drop order.
    _heap: Box<dyn MoeHold>,
    pub(super) slab: Arc<Buffer>,
    pub(super) offsets: Arc<Buffer>,
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
            // SAFETY: called on `u32`, `u16` and `f32` slices only (the
            // records below), which have no padding, so every one of the
            // `size_of_val(v)` bytes is initialised; the result borrows `v`.
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
            // SAFETY: Metal asks that `off` be a multiple of the alignment
            // `heapBufferSizeAndAlign` gave for this length and that the
            // buffer fit the heap: `offs` places each record at its own
            // alignment and the heap is `total >= cursor` bytes, `alias(0,
            // total)` included. Lifetime: `_heap` in the returned slab holds
            // the heap, but the aliases also leave in the `MoeSlabExpert`s
            // returned beside it; that an alias outliving the slab is sound
            // assumes Metal keeps a heap alive while a buffer placed in it
            // lives, which this crate does not establish.
            let raw = unsafe { heap.newBufferWithLength_options_offset(len, opts, off) }
                .ok_or_else(|| candle_core::Error::msg(format!("a heap buffer of {len} B at {off}")))?;
            Ok(Arc::new(Buffer::new(raw)))
        };
        let slab = alias(0, total)?;
        let base = slab.contents();
        for (it, &off) in items.iter().zip(&offs) {
            // SAFETY: `base` is the CPU address of the whole shared-storage
            // heap (`slab` aliases it from 0 for `total` bytes); `off +
            // it.len() <= cursor <= total` by construction of `offs`; the
            // source is a host slice that cannot overlap device memory; no
            // command buffer has seen the heap yet, so nothing reads it.
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

pub(super) fn moe_shape(slab: &MetalMoeSlab) -> Result<MoeShape> {
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
