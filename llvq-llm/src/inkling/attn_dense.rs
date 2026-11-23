//! The dense products of one decoded attention row of Inkling 2 Mini, on
//! Metal, reading the weights in BF16 as the checkpoint stores them.
//!
//! A visit's attention today runs seven candle GEMMs over weights widened to
//! f32 at load, and about thirty small operations around them
//! (`AttnWeights::forward_chunked` in `layers.rs`). These kernels do the same
//! work for ONE row in three launches, f32 accumulation throughout:
//!
//!   - [`AttnDense::stage_in`], launch 1 (`ad_products`): every product that
//!     reads an attention input, one SIMD-group a weight row, the rows of
//!     `wq`, `wr`, `attn_sink_proj`, `gate_attn` (query input), `wk` and the
//!     key `weight_proj` (key input), `wv` and the value `weight_proj` (value
//!     input), all from one owned BF16 buffer read once, in order.
//!   - [`AttnDense::stage_in`], launch 2 (`ad_finish`): one threadgroup a
//!     head. The query's learned scale, L2 clip and log scale; the relative
//!     states' scale, clip, log scale and their projection to the relative
//!     logits; the sinks; the output gate's sigmoid; the key and value
//!     convolutions (static or dynamic) on device rings, the key's L2 clip and
//!     the value's unweighted RMS norm.
//!   - [`AttnDense::stage_out`] (`ad_out`): `(a * gate) @ wo.T`, one SIMD-group
//!     an output row, the gate product formed as the reference forms it.
//!
//! Exactness. The products sum in a SIMD-group order, not in candle's GEMM
//! order, so every output is class B against `layers.rs` (gated at a
//! relative 1e-5 by `tests/inkling_attn_dense.rs`). Everything after a
//! product is the reference's operations in the reference's order: the
//! elementwise steps one for one (constants narrowed from f64 to f32 as
//! candle's `affine` narrows them, `fp contract(off)`, precise `sqrt`, `exp`
//! and division), and the sums of squares of the clips and the norm in the
//! lane order of candle's CPU `vec_sum` on aarch64 (16 lanes, then a fixed
//! fold). Given products that are exact in f32, the query, key, value, sinks
//! and rings are therefore bit for bit those of the CPU reference; the test
//! builds such inputs to prove it. Two outputs stay class B in every case:
//! the relative logits, whose 16-term product has the matmul's order problem
//! again, and the gate, whose `exp` is Metal's and differs from the host
//! libm's `expf` in the last bit on some inputs.
//!
//! The outputs keep the layouts `forward_chunked` hands to the cache and the
//! score step: the query `[heads, 1, d]`, the key and the value
//! `[kv_heads, 1, d]`, the relative logits `[heads, 1, extent]`, the sinks
//! `[heads, 1, 1]`, the gate `[1, heads * d]`.
//!
//! `layers.rs` runs these kernels for a one-row visit when the loader chose
//! them (`LayerWeights::load_dense`, through [`FusedAttn`]); the key and
//! value rings then live in the slot ([`AttnSlot`]). Same candle traps as
//! `crate::fused_metal` (its header) and `inkling::glue`: the compile runs
//! under `catch_unwind`; the weight and ring buffers are OWNED, so the
//! allocator never hands them out as scratch; every output word is written.
#![cfg(ink_glue)]

use std::sync::{Arc, RwLockReadGuard};

#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_core::backend::BackendStorage;
#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_core::MetalStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape, Storage, Tensor};
#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::{MTLResourceUsage, MTLSize};

#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::CudaStorage;

/// The score kernel of `super::attn_scores`. It has no CUDA twin yet: there
/// the type is never built.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub type RowKernel = super::attn_scores::AttnRow;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub struct RowKernel;

/// A compiled kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
type Pipe = ComputePipeline;
#[cfg(all(target_os = "linux", feature = "cuda"))]
type Pipe = CudaFunction;

/// A buffer this module owns on the device: BF16 weights or an f32 ring.
#[cfg(all(target_os = "macos", feature = "metal"))]
type Buf = Arc<Buffer>;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub enum Buf {
    U16(CudaSlice<u16>),
    F32(CudaSlice<f32>),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn up_u16(device: &Device, v: &[u16]) -> Result<Buf> {
    let Device::Metal(dev) = device else { candle_core::bail!("attn_dense wants a Metal device") };
    dev.new_buffer_with_data(v)
}
#[cfg(all(target_os = "macos", feature = "metal"))]
fn up_f32(device: &Device, v: &[f32]) -> Result<Buf> {
    let Device::Metal(dev) = device else { candle_core::bail!("attn_dense wants a Metal device") };
    dev.new_buffer_with_data(v)
}
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn up_u16(device: &Device, v: &[u16]) -> Result<Buf> {
    let Device::Cuda(dev) = device else { candle_core::bail!("attn_dense wants a CUDA device") };
    Ok(Buf::U16(dev.cuda_stream().clone_htod(v).map_err(|e| candle_core::Error::msg(format!("attn_dense: upload: {e}")))?))
}
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn up_f32(device: &Device, v: &[f32]) -> Result<Buf> {
    let Device::Cuda(dev) = device else { candle_core::bail!("attn_dense wants a CUDA device") };
    Ok(Buf::F32(dev.cuda_stream().clone_htod(v).map_err(|e| candle_core::Error::msg(format!("attn_dense: upload: {e}")))?))
}

use super::config::InklingConfig;
use super::layers::{AttnMats, AttnWeights, KvConv};

/// Threads a threadgroup of the two matrix-vector kernels: 8 SIMD-groups,
/// one weight row each.
const MV_TG: usize = 256;
/// Threads a threadgroup of `ad_finish`.
const FIN_TG: usize = 128;
/// Largest head width `ad_finish` stages in threadgroup memory.
pub const MAX_HD: usize = 256;
/// Largest relative width.
pub const MAX_DREL: usize = 64;
/// Largest convolution kernel.
pub const MAX_TAPS: usize = 8;
/// The epsilon `layers::l2_clip` adds to the norm, fixed whatever the config.
const CLIP_EPS: f64 = 1e-6;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const SRC_CUDA: &str = include_str!("../../kernels/ink_attn.cu");

#[cfg(all(target_os = "macos", feature = "metal"))]
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#define NSG (MV_TG / 32u)
#define MAX_HD 256u
#define MAX_DREL 64u

struct ProdArgs { uint h; uint rows; uint q_rows; uint k_rows; };

struct FinArgs {
    uint nh; uint nkv; uint hd; uint dr; uint ext; uint taps; uint dyn; uint has_tau; uint pos_k; uint pos_v;
    uint off_r; uint off_s; uint off_g; uint off_k; uint off_pk; uint off_v; uint off_pv;
    uint so_rs; uint so_rp; uint so_kc; uint so_vc;
    uint oo_k; uint oo_v; uint oo_rel; uint oo_s; uint oo_g;
    float qk_thr; float rel_thr; float clip_eps; float eps; float inv_hd; float tau; float conv_scale;
};

struct OutArgs { uint din; uint rows; };

static inline float bf_lo(uint w) { return as_type<float>(w << 16); }
static inline float bf_hi(uint w) { return as_type<float>(w & 0xffff0000u); }
static inline float bfv(device const ushort *p, uint i) { return as_type<float>(uint(p[i]) << 16); }

// Eight BF16 weights (one uint4) against eight activations, added in index order.
static inline float dot8(uint4 w, float4 x0, float4 x1, float acc) {
    float p;
    p = bf_lo(w.x) * x0.x; acc = acc + p;
    p = bf_hi(w.x) * x0.y; acc = acc + p;
    p = bf_lo(w.y) * x0.z; acc = acc + p;
    p = bf_hi(w.y) * x0.w; acc = acc + p;
    p = bf_lo(w.z) * x1.x; acc = acc + p;
    p = bf_hi(w.z) * x1.y; acc = acc + p;
    p = bf_lo(w.w) * x1.z; acc = acc + p;
    p = bf_hi(w.w) * x1.w; acc = acc + p;
    return acc;
}

// Launch 1: out[r] = x(r) . w[r], one SIMD-group a row. Rows below q_rows read
// the query input, the next k_rows the key input, the rest the value input.
kernel void ad_products(
    device const float *xq [[buffer(0)]],
    device const float *xk [[buffer(1)]],
    device const float *xv [[buffer(2)]],
    device const uint4 *w [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant ProdArgs &a [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg * NSG + sg;
    if (r >= a.rows) { return; }
    device const float *x = r < a.q_rows ? xq : (r < a.q_rows + a.k_rows ? xk : xv);
    device const packed_float4 *x4 = (device const packed_float4 *)x;
    const uint n8 = a.h / 8u;
    device const uint4 *wr = w + (ulong)r * n8;
    float acc = 0.0f;
    for (uint i = lane; i < n8; i += 32u) {
        acc = dot8(wr[i], float4(x4[2u * i]), float4(x4[2u * i + 1u]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[r] = acc; }
}

// candle's CPU sum over a contiguous last dim on aarch64 (cpu/mod.rs vec_sum
// with the NEON unit): 16 lane sums over the whole 16-blocks, folded as
// vec_reduce folds them, then the leftovers in order.
static float cpu_sum(threadgroup const float *x, uint n) {
    float s[16];
    for (uint l = 0; l < 16u; l++) { s[l] = 0.0f; }
    const uint np = n & ~15u;
    for (uint i = 0; i < np; i += 16u) {
        for (uint l = 0; l < 16u; l++) { s[l] = s[l] + x[i + l]; }
    }
    float v[4];
    for (uint e = 0; e < 4u; e++) {
        float a = s[e] + s[4u + e];
        float b = s[8u + e] + s[12u + e];
        v[e] = a + b;
    }
    float lo = v[0] + v[1];
    float hi = v[2] + v[3];
    float b = lo + hi;
    for (uint i = np; i < n; i++) { b = b + x[i]; }
    return b;
}

// The factor of layers::l2_clip: min(thr * recip(sqrt(sum x^2) + 1e-6), 1),
// each step one candle op (affine = x * mul + add, both rounded).
static float clip_factor(threadgroup const float *sq, uint n, float thr, float ceps) {
    float s = cpu_sum(sq, n);
    float norm = precise::sqrt(s);
    float t = norm * 1.0f;
    t = t + ceps;
    float r = precise::divide(1.0f, t);
    float f = r * thr;
    f = f + 0.0f;
    return min(f, 1.0f);
}

// One channel of layers::short_conv on the ring: taps oldest first, then x;
// the residual and the output scale. x replaces the oldest row.
static float conv_static(device float *ring, device const ushort *wt, uint wo, float x, uint c, uint C, uint k,
                         uint pos, float scale) {
    const uint k1 = k - 1u;
    float acc = 0.0f;
    for (uint t = 0; t < k1; t++) {
        float v = ring[((pos + t) % k1) * C + c];
        float p = v * bfv(wt, wo + c * k + t);
        acc = acc + p;
    }
    float p = x * bfv(wt, wo + c * k + k1);
    acc = acc + p;
    ring[pos * C + c] = x;
    float o = acc + x;
    return o * scale;
}

// One channel of layers::dynamic_conv: tap `lag` back is dw[lag] +
// static_w[lag, c]; lag 0 is x, lag l the l-th newest history row.
static float conv_dyn(device float *ring, device const ushort *wt, uint so, device const float *dw, float x, uint c,
                      uint C, uint k, uint pos, float scale) {
    const uint k1 = k - 1u;
    float acc = 0.0f;
    for (uint lag = 0; lag < k; lag++) {
        float src = lag == 0u ? x : ring[((pos + k1 - lag) % k1) * C + c];
        float wv = dw[lag] + bfv(wt, so + lag * C + c);
        float p = src * wv;
        acc = acc + p;
    }
    ring[pos * C + c] = x;
    float o = acc + x;
    return o * scale;
}

// Launch 2. Threadgroups [0, nh): query head g, its relative states and
// logits, its sink, its gate channels. [nh, nh + nkv): key head. Then value heads.
kernel void ad_finish(
    device const float *P [[buffer(0)]],
    device const ushort *small [[buffer(1)]],
    device float *ring_k [[buffer(2)]],
    device float *ring_v [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant FinArgs &a [[buffer(5)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float X[MAX_HD];
    threadgroup float SQ[MAX_HD];
    threadgroup float R[MAX_DREL];
    threadgroup float RS[MAX_DREL];
    threadgroup float FAC[2];
    const uint hd = a.hd;
    if (g < a.nh) {
        const uint h = g;
        for (uint c = tid; c < hd; c += FIN_TG) {
            float v = P[h * hd + c] * bfv(small, h * hd + c);
            X[c] = v;
            SQ[c] = v * v;
        }
        for (uint j = tid; j < a.dr; j += FIN_TG) {
            float v = P[a.off_r + h * a.dr + j] * bfv(small, a.so_rs + h * a.dr + j);
            R[j] = v;
            RS[j] = v * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            FAC[0] = clip_factor(SQ, hd, a.qk_thr, a.clip_eps);
            FAC[1] = clip_factor(RS, a.dr, a.rel_thr, a.clip_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < hd; c += FIN_TG) {
            float q = X[c] * FAC[0];
            if (a.has_tau != 0u) { q = q * a.tau; }
            out[h * hd + c] = q;
        }
        for (uint j = tid; j < a.dr; j += FIN_TG) {
            float r = R[j] * FAC[1];
            if (a.has_tau != 0u) { r = r * a.tau; }
            R[j] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < a.ext; e += FIN_TG) {
            float acc = 0.0f;
            for (uint j = 0; j < a.dr; j++) {
                float p = R[j] * bfv(small, a.so_rp + j * a.ext + e);
                acc = acc + p;
            }
            out[a.oo_rel + h * a.ext + e] = acc;
        }
        if (tid == 0u) { out[a.oo_s + h] = P[a.off_s + h]; }
        for (uint c = tid; c < hd; c += FIN_TG) {
            float v = P[a.off_g + h * hd + c];
            float z = precise::exp(-v);
            z = z + 1.0f;
            out[a.oo_g + h * hd + c] = precise::divide(1.0f, z);
        }
        return;
    }
    const bool is_k = g < a.nh + a.nkv;
    const uint hh = is_k ? g - a.nh : g - a.nh - a.nkv;
    const uint C = a.nkv * hd;
    device float *ring = is_k ? ring_k : ring_v;
    const uint pos = is_k ? a.pos_k : a.pos_v;
    const uint off = is_k ? a.off_k : a.off_v;
    const uint offp = is_k ? a.off_pk : a.off_pv;
    const uint so = is_k ? a.so_kc : a.so_vc;
    for (uint c = tid; c < hd; c += FIN_TG) {
        const uint ch = hh * hd + c;
        float x = P[off + ch];
        float y = a.dyn != 0u
            ? conv_dyn(ring, small, so, P + offp + hh * a.taps, x, ch, C, a.taps, pos, a.conv_scale)
            : conv_static(ring, small, so, x, ch, C, a.taps, pos, a.conv_scale);
        X[c] = y;
        SQ[c] = y * y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        if (is_k) {
            FAC[0] = clip_factor(SQ, hd, a.qk_thr, a.clip_eps);
        } else {
            // layers::rms_unweighted: recip(sqrt(mean(x^2) + eps)).
            float s = cpu_sum(SQ, hd);
            float ms = s * a.inv_hd;
            ms = ms + 0.0f;
            ms = ms * 1.0f;
            ms = ms + a.eps;
            FAC[0] = precise::divide(1.0f, precise::sqrt(ms));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint oo = is_k ? a.oo_k : a.oo_v;
    for (uint c = tid; c < hd; c += FIN_TG) { out[oo + hh * hd + c] = X[c] * FAC[0]; }
}

// stage_out: out[r] = (a * gate) . wo[r], one SIMD-group a row.
kernel void ad_out(
    device const float *av [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const uint4 *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant OutArgs &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg * NSG + sg;
    if (r >= p.rows) { return; }
    device const packed_float4 *a4 = (device const packed_float4 *)av;
    device const packed_float4 *g4 = (device const packed_float4 *)gate;
    const uint n8 = p.din / 8u;
    device const uint4 *wr = w + (ulong)r * n8;
    float acc = 0.0f;
    for (uint i = lane; i < n8; i += 32u) {
        float4 x0 = float4(a4[2u * i]) * float4(g4[2u * i]);
        float4 x1 = float4(a4[2u * i + 1u]) * float4(g4[2u * i + 1u]);
        acc = dot8(wr[i], x0, x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[r] = acc; }
}

// A ring in logical order, oldest row first: what SlotCache holds.
kernel void ad_ring_state(
    device const float *ring [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint4 &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    const uint c = a.x;
    const uint k1 = a.y - 1u;
    const uint pos = a.z;
    if (i >= k1 * c) { return; }
    out[i] = ring[((pos + i / c) % k1) * c + i % c];
}

// BF16 elements [off, off + n) of an owned buffer widened to f32: exact.
kernel void ad_widen(
    device const ushort *src [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.y) { return; }
    out[i] = as_type<float>(uint(src[a.x + i]) << 16);
}
"#;

/// A constant struct of the MSL as its 4-byte words, in field order.
#[derive(Default)]
struct Words(Vec<u32>);

impl Words {
    fn u(mut self, v: usize) -> Self {
        self.0.push(u32::try_from(v).expect("a kernel argument fits u32"));
        self
    }
    fn f(mut self, v: f32) -> Self {
        self.0.push(v.to_bits());
        self
    }
}

/// The f32 the reference's `Inkling2LoopLogScale` gives position `pos`:
/// the expression of `layers::log_scale`, term for term.
pub fn tau(cfg: &InklingConfig, pos: usize) -> f32 {
    tau_of(cfg.log_scaling_alpha, cfg.log_scaling_n_floor, pos)
}

fn tau_of(alpha: f64, n_floor: f64, pos: usize) -> f32 {
    let r = ((pos + 1) as f32 / n_floor as f32).max(1.0);
    1.0 + alpha as f32 * r.ln()
}

/// The BF16 bits of `t`, row-major. A BF16 tensor gives its bits; an f32
/// tensor must hold BF16 values exactly (what the loader's widening gives),
/// or it is refused, since a narrowing that rounds would change the weights.
pub fn bf16_bits(name: &str, t: &Tensor) -> Result<Vec<u16>> {
    let t = t.to_device(&Device::Cpu)?.flatten_all()?;
    match t.dtype() {
        DType::BF16 => Ok(t.to_vec1::<half::bf16>()?.into_iter().map(|v| v.to_bits()).collect()),
        DType::F32 => {
            let v = t.to_vec1::<f32>()?;
            let mut out = Vec::with_capacity(v.len());
            for (i, x) in v.into_iter().enumerate() {
                let b = half::bf16::from_f32(x);
                if b.to_f32().to_bits() != x.to_bits() {
                    candle_core::bail!("{name}: element {i} = {x:e} is not a BF16 value");
                }
                out.push(b.to_bits());
            }
            Ok(out)
        }
        d => candle_core::bail!("{name}: {d:?}, want BF16 or f32 holding BF16 values"),
    }
}

/// The weights of one attention block as the kernels read them: BF16, in
/// owned device buffers.
pub struct AttnDenseWeights {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub extent: usize,
    pub taps: usize,
    pub hidden: usize,
    pub sliding: bool,
    pub dynamic: bool,
    qk_thr: f32,
    rel_thr: f32,
    eps: f32,
    log_alpha: f64,
    log_floor: f64,
    /// Every row read from an attention input, in launch-1 order.
    rows: Buf,
    /// `wo` `[hidden, heads * head_dim]`.
    wo: Buf,
    /// `q_scale | rel_scale | rel_proj | key conv | value conv`.
    small: Buf,
    lay: Offsets,
}

/// The tensors of one attention block, borrowed, for
/// [`AttnDenseWeights::from_parts`]: the fields of `AttnWeights` it reads.
pub struct DenseParts<'a> {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub sliding: bool,
    pub rel_extent: usize,
    pub wq: &'a Tensor,
    pub wr: &'a Tensor,
    pub sink: &'a Tensor,
    pub gate: &'a Tensor,
    pub wk: &'a Tensor,
    pub wv: &'a Tensor,
    pub wo: &'a Tensor,
    pub q_scale: &'a Tensor,
    pub rel_scale: &'a Tensor,
    pub rel_proj: &'a Tensor,
    pub k_conv: &'a KvConv,
    pub v_conv: &'a KvConv,
}

/// A convolution as BF16 bits: its weights in the kernel's layout, its
/// `weight_proj` rows when dynamic, its taps, and whether it is dynamic.
type ConvBits = (Vec<u16>, Option<Vec<u16>>, usize, bool);

#[derive(Clone, Copy, Debug)]
struct Offsets {
    q_rows: usize,
    k_rows: usize,
    rows: usize,
    off_r: usize,
    off_s: usize,
    off_g: usize,
    off_k: usize,
    off_pk: usize,
    off_v: usize,
    off_pv: usize,
    so_rs: usize,
    so_rp: usize,
    so_kc: usize,
    so_vc: usize,
    oo_k: usize,
    oo_v: usize,
    oo_rel: usize,
    oo_s: usize,
    oo_g: usize,
    out: usize,
}

impl AttnDenseWeights {
    /// The BF16 buffers of `w`, whose tensors are BF16 or f32 widened from
    /// BF16 (any other value is refused). `cfg` gives the thresholds and the
    /// norm epsilon.
    pub fn new(w: &AttnWeights, cfg: &InklingConfig, device: &Device) -> Result<Self> {
        let parts = DenseParts {
            heads: w.heads,
            kv_heads: w.kv_heads,
            head_dim: w.head_dim,
            sliding: w.sliding,
            rel_extent: w.rel_extent,
            wq: &w.wq,
            wr: &w.wr,
            sink: &w.sink,
            gate: &w.gate,
            wk: &w.wk,
            wv: &w.wv,
            wo: &w.wo,
            q_scale: &w.q_scale,
            rel_scale: &w.rel_scale,
            rel_proj: &w.rel_proj,
            k_conv: &w.k_conv,
            v_conv: &w.v_conv,
        };
        Self::from_parts(&parts, cfg, device)
    }

    /// [`Self::new`] from the tensors one by one, in BF16 or in f32 holding
    /// BF16 values, on any device: what the loader hands when no f32 copy of
    /// the matrices is to be kept.
    pub fn from_parts(w: &DenseParts, cfg: &InklingConfig, device: &Device) -> Result<Self> {
        let (nh, nkv, hd) = (w.heads, w.kv_heads, w.head_dim);
        let (hidden, din) = (w.wq.dim(1)?, nh * hd);
        let dr = w.rel_scale.dim(1)?;
        let ext = w.rel_extent;
        let c = nkv * hd;
        if hidden % 8 != 0 || din % 8 != 0 {
            candle_core::bail!("attn_dense: widths {hidden} and {din} must be multiples of 8");
        }
        if hd > MAX_HD || dr > MAX_DREL || dr == 0 {
            candle_core::bail!("attn_dense: head width {hd} (at most {MAX_HD}), relative width {dr} (1 to {MAX_DREL})");
        }
        let shape = |name: &str, t: &Tensor, want: &[usize]| -> Result<()> {
            if t.dims() != want {
                candle_core::bail!("attn_dense: {name} is {:?}, want {want:?}", t.dims());
            }
            Ok(())
        };
        shape("wq", w.wq, &[din, hidden])?;
        shape("wr", w.wr, &[nh * dr, hidden])?;
        shape("sink", w.sink, &[nh, hidden])?;
        shape("gate", w.gate, &[din, hidden])?;
        shape("wk", w.wk, &[c, hidden])?;
        shape("wv", w.wv, &[c, hidden])?;
        shape("wo", w.wo, &[hidden, din])?;
        shape("q_scale", w.q_scale, &[nh, hd])?;
        shape("rel_scale", w.rel_scale, &[nh, dr])?;
        shape("rel_proj", w.rel_proj, &[dr, ext])?;
        // (conv weights in the kernel's layout, weight_proj rows or none, taps)
        let conv = |name: &str, k: &KvConv| -> Result<ConvBits> {
            match k {
                KvConv::Static(cw) => {
                    let (cc, taps) = cw.dims2()?;
                    if cc != c {
                        candle_core::bail!("attn_dense: {name} conv over {cc} channels, want {c}");
                    }
                    Ok((bf16_bits(name, cw)?, None, taps, false))
                }
                KvConv::Dynamic { static_w, weight_proj } => {
                    let (taps, cc) = static_w.dims2()?;
                    if cc != c {
                        candle_core::bail!("attn_dense: {name} static_w over {cc} channels, want {c}");
                    }
                    shape("weight_proj", weight_proj, &[nkv * taps, hidden])?;
                    Ok((bf16_bits(name, static_w)?, Some(bf16_bits(name, weight_proj)?), taps, true))
                }
            }
        };
        let (kc, kp, taps, dynamic) = conv("k conv", w.k_conv)?;
        let (vc, vp, vtaps, vdyn) = conv("v conv", w.v_conv)?;
        if taps != vtaps || dynamic != vdyn || !(2..=MAX_TAPS).contains(&taps) {
            candle_core::bail!("attn_dense: key conv {taps} taps dynamic {dynamic}, value conv {vtaps} taps dynamic {vdyn}");
        }
        let np = if dynamic { nkv * taps } else { 0 };
        let q_rows = 2 * din + nh * dr + nh;
        let k_rows = c + np;
        let off_r = din;
        let off_s = off_r + nh * dr;
        let off_g = off_s + nh;
        let off_k = q_rows;
        let off_pk = off_k + c;
        let off_v = off_pk + np;
        let off_pv = off_v + c;
        let rows_n = off_pv + np;
        let mut rows: Vec<u16> = Vec::with_capacity(rows_n * hidden);
        for (name, t) in [("wq", w.wq), ("wr", w.wr), ("sink", w.sink), ("gate", w.gate), ("wk", w.wk)] {
            rows.extend(bf16_bits(name, t)?);
        }
        if let Some(p) = &kp {
            rows.extend_from_slice(p);
        }
        rows.extend(bf16_bits("wv", w.wv)?);
        if let Some(p) = &vp {
            rows.extend_from_slice(p);
        }
        debug_assert_eq!(rows.len(), rows_n * hidden);
        let so_rs = nh * hd;
        let so_rp = so_rs + nh * dr;
        let so_kc = so_rp + dr * ext;
        let so_vc = so_kc + c * taps;
        let mut small = bf16_bits("q_scale", w.q_scale)?;
        small.extend(bf16_bits("rel_scale", w.rel_scale)?);
        small.extend(bf16_bits("rel_proj", w.rel_proj)?);
        small.extend(kc);
        small.extend(vc);
        let oo_k = din;
        let oo_v = oo_k + c;
        let oo_rel = oo_v + c;
        let oo_s = oo_rel + nh * ext;
        let oo_g = oo_s + nh;
        let lay = Offsets {
            q_rows,
            k_rows,
            rows: rows_n,
            off_r,
            off_s,
            off_g,
            off_k,
            off_pk,
            off_v,
            off_pv,
            so_rs,
            so_rp,
            so_kc,
            so_vc,
            oo_k,
            oo_v,
            oo_rel,
            oo_s,
            oo_g,
            out: oo_g + din,
        };
        Ok(Self {
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            d_rel: dr,
            extent: ext,
            taps,
            hidden,
            sliding: w.sliding,
            dynamic,
            qk_thr: cfg.qk_l2_clip_threshold as f32,
            rel_thr: cfg.rel_l2_clip_threshold as f32,
            eps: cfg.rms_norm_eps as f32,
            log_alpha: cfg.log_scaling_alpha,
            log_floor: cfg.log_scaling_n_floor,
            rows: up_u16(device, &rows)?,
            wo: up_u16(device, &bf16_bits("wo", w.wo)?)?,
            small: up_u16(device, &small)?,
            lay,
        })
    }

    /// Weight bytes one visit reads: every BF16 element once.
    pub fn bytes_read(&self) -> usize {
        let l = &self.lay;
        2 * (l.rows * self.hidden
            + self.hidden * self.heads * self.head_dim
            + l.so_vc
            + self.kv_heads * self.head_dim * self.taps)
    }

    /// Bytes of the seven attention matrices in these buffers (`wq`, `wr`,
    /// `sink`, `gate`, `wk`, `wv`, `wo`), BF16: half their f32 size.
    pub fn matrix_bytes(&self) -> usize {
        let (din, c, h) = (self.heads * self.head_dim, self.kv_heads * self.head_dim, self.hidden);
        2 * ((2 * din + self.heads * self.d_rel + self.heads + 2 * c) * h + h * din)
    }

    /// The convolution width of the key and value rings, `kv_heads * head_dim`.
    pub fn conv_channels(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// A key or value convolution state: `taps - 1` rows of `channels` on the
/// device, read and rewritten in place by `ad_finish`, oldest row replaced.
pub struct KvRing {
    buf: Buf,
    taps: usize,
    c: usize,
    /// The physical row of the oldest input.
    pos: usize,
}

impl KvRing {
    /// A ring from a state `[taps - 1, c]` oldest row first (what `SlotCache`
    /// holds after a prompt), or zeros (an empty slot).
    pub fn new(device: &Device, taps: usize, c: usize, state: Option<&Tensor>) -> Result<Self> {
        if taps < 2 {
            candle_core::bail!("KvRing: kernel size {taps} has no state");
        }
        let host = match state {
            Some(s) => {
                if s.dims() != [taps - 1, c] {
                    candle_core::bail!("KvRing: state {:?} for [{}, {c}]", s.dims(), taps - 1);
                }
                s.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
            }
            None => vec![0f32; (taps - 1) * c],
        };
        Ok(Self { buf: up_f32(device, &host)?, taps, c, pos: 0 })
    }

    /// The state `[taps - 1, c]`, oldest row first, as a new tensor on the
    /// device of `like`.
    pub fn state(&self, k: &AttnDense, like: &Tensor) -> Result<Tensor> {
        let args = Words::default().u(self.c).u(self.taps).u(self.pos).u(0);
        let n = (self.taps - 1) * self.c;
        let op = Launch {
            name: "attn-dense-ring-state",
            pipe: &k.ring_state,
            args: vec![Arg::Owned(&self.buf, false), Arg::Out, Arg::Bytes(args)],
            out_elems: n,
            out_shape: Shape::from((self.taps - 1, self.c)),
            grid: (n.div_ceil(256), 256),
        };
        like.apply_op1_no_bwd(&op)
    }
}

/// The key and value convolution states of one visit.
pub struct KvRings {
    pub k: KvRing,
    pub v: KvRing,
}

impl KvRings {
    /// Both rings for `w`, from the states a prompt left (`[taps - 1, kv_heads * head_dim]`
    /// each, oldest row first) or zeros.
    pub fn new(device: &Device, w: &AttnDenseWeights, k: Option<&Tensor>, v: Option<&Tensor>) -> Result<Self> {
        let c = w.conv_channels();
        Ok(Self { k: KvRing::new(device, w.taps, c, k)?, v: KvRing::new(device, w.taps, c, v)? })
    }
}

/// The attention of one layer through the kernels: the compiled library and
/// the layer's BF16 buffers, the only copy of its seven matrices on the
/// device once the loader hands them here.
pub struct FusedAttn {
    pub kernels: Arc<AttnDense>,
    pub w: AttnDenseWeights,
    /// The score kernel of `super::attn_scores`, when the loader chose it
    /// (`AttnKind::FusedScores`): a one-row visit then reads the keys and
    /// the values where the cache holds them, in one launch.
    pub row: Option<Arc<RowKernel>>,
    dev: Device,
}

impl FusedAttn {
    pub fn new(kernels: Arc<AttnDense>, w: AttnDenseWeights, dev: &Device) -> Self {
        Self { kernels, w, row: None, dev: dev.clone() }
    }

    /// The seven matrices widened to f32 for one several-row call (each
    /// BF16 value widens exactly, so the call reads the values the f32 copies
    /// held). Dropped by the caller after the call.
    pub fn widen(&self) -> Result<AttnMats> {
        let like = Tensor::zeros(1, DType::F32, &self.dev)?;
        let w = &self.w;
        let l = &w.lay;
        let (h, din, c) = (w.hidden, w.heads * w.head_dim, w.kv_heads * w.head_dim);
        let one = |buf: &Buf, first_row: usize, rows: usize, cols: usize| -> Result<Tensor> {
            let n = rows * cols;
            let op = Launch {
                name: "attn-dense-widen",
                pipe: &self.kernels.widen,
                args: vec![Arg::Owned(buf, false), Arg::Out, Arg::Bytes(Words::default().u(first_row * cols).u(n))],
                out_elems: n,
                out_shape: Shape::from((rows, cols)),
                grid: (n.div_ceil(256), 256),
            };
            like.apply_op1_no_bwd(&op)
        };
        Ok(AttnMats {
            wq: one(&w.rows, 0, din, h)?,
            wr: one(&w.rows, l.off_r, w.heads * w.d_rel, h)?,
            sink: one(&w.rows, l.off_s, w.heads, h)?,
            gate: one(&w.rows, l.off_g, din, h)?,
            wk: one(&w.rows, l.off_k, c, h)?,
            wv: one(&w.rows, l.off_v, c, h)?,
            wo: one(&w.wo, 0, h, din)?,
        })
    }
}

/// The key and value convolution states of one cache slot while the kernels
/// step it: its rings and what reads them back.
pub struct AttnSlot {
    kernels: Arc<AttnDense>,
    pub rings: KvRings,
    dev: Device,
}

impl std::fmt::Debug for AttnSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AttnSlot {{ taps: {}, channels: {} }}", self.rings.k.taps, self.rings.k.c)
    }
}

impl AttnSlot {
    /// Rings for `a` from the states `[taps - 1, kv_heads * head_dim]` the
    /// slot holds (oldest row first), or zeros.
    pub fn new(a: &FusedAttn, k: Option<&Tensor>, v: Option<&Tensor>) -> Result<Self> {
        Ok(Self { kernels: a.kernels.clone(), rings: KvRings::new(&a.dev, &a.w, k, v)?, dev: a.dev.clone() })
    }

    /// The key and value states `[taps - 1, C]`, oldest row first, as new tensors.
    pub fn states(&self) -> Result<(Tensor, Tensor)> {
        let like = Tensor::zeros(1, DType::F32, &self.dev)?;
        Ok((self.rings.k.state(&self.kernels, &like)?, self.rings.v.state(&self.kernels, &like)?))
    }
}

/// What [`AttnDense::stage_in`] returns: views of one device buffer, in the
/// layouts `forward_chunked` gives the cache and the score step.
pub struct StageIn {
    /// `[heads, 1, head_dim]`: scaled, clipped, log-scaled in full layers.
    pub q: Tensor,
    /// `[kv_heads, 1, head_dim]`: convolved and clipped.
    pub k: Tensor,
    /// `[kv_heads, 1, head_dim]`: convolved and normed.
    pub v: Tensor,
    /// `[heads, 1, extent]`.
    pub rel_logits: Tensor,
    /// `[heads, 1, 1]`.
    pub sinks: Tensor,
    /// `[1, heads * head_dim]`: `sigmoid(xq @ gate.T)`.
    pub gate: Tensor,
}

/// The compiled kernels.
pub struct AttnDense {
    products: Pipe,
    finish: Pipe,
    out: Pipe,
    ring_state: Pipe,
    widen: Pipe,
    /// The module the functions live in.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    _module: llvq_cuda::gpu::Cuda,
}

const NAMES: [&str; 5] = ["ad_products", "ad_finish", "ad_out", "ad_ring_state", "ad_widen"];

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile(dev: &candle_core::MetalDevice) -> Result<Vec<ComputePipeline>> {
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let raw = dev.device().clone();
    let src = format!("#define MV_TG {MV_TG}u\n#define FIN_TG {FIN_TG}u\n{SRC}");
    let names = NAMES;
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let lib = raw.new_library_with_source(&src, Some(&opts))?;
        names
            .iter()
            .map(|n| raw.new_compute_pipeline_state_with_function(&lib.get_function(n, None)?))
            .collect::<std::result::Result<Vec<_>, _>>()
    }));
    match built {
        Ok(Ok(p)) => Ok(p),
        Ok(Err(e)) => candle_core::bail!("inkling attn_dense: Metal refused the source: {e:?}"),
        Err(panic) => {
            let what = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "no diagnostics".into());
            candle_core::bail!("inkling attn_dense: the MSL did not compile: {what}")
        }
    }
}

impl AttnDense {
    /// Compile the kernels for a CUDA device (`kernels/ink_attn.cu`, NVRTC),
    /// a module of its own on the device's stream.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Cuda(dev) = device else { candle_core::bail!("the attn_dense kernels want a CUDA device") };
        let define = format!("#define MV_TG {MV_TG}u\n#define FIN_TG {FIN_TG}u\n");
        let src = llvq_cuda::gpu::KernelSource::new(&[define.as_str(), SRC_CUDA]);
        println!("NVRTC attn unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let module = llvq_cuda::gpu::Cuda::on_stream(dev.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
        let mut p = NAMES
            .iter()
            .map(|n| module.func(n).map_err(candle_core::Error::msg))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || p.next().expect("five functions");
        Ok(Self { products: next(), finish: next(), out: next(), ring_state: next(), widen: next(), _module: module })
    }

    /// Compile the kernels for a Metal device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Metal(dev) = device else { candle_core::bail!("the attn_dense kernels want a Metal device") };
        let mut p = compile(dev)?.into_iter();
        let mut next = || p.next().expect("five pipelines");
        let k = Self { products: next(), finish: next(), out: next(), ring_state: next(), widen: next() };
        for (name, pipe, tg) in
            [("ad_products", &k.products, MV_TG), ("ad_finish", &k.finish, FIN_TG), ("ad_out", &k.out, MV_TG)]
        {
            let max = pipe.max_total_threads_per_threadgroup();
            if max < tg {
                candle_core::bail!("{name}: the pipeline takes {max} threads a threadgroup, the kernel needs {tg}");
            }
        }
        Ok(k)
    }

    /// Launches 1 and 2 for one row at absolute position `pos` (the log
    /// scale of full layers reads it). `xq`, `xk`, `xv` are `[1, hidden]`,
    /// the same tensor when the visit has one input. Both rings advance one row.
    pub fn stage_in(
        &self,
        w: &AttnDenseWeights,
        xq: &Tensor,
        xk: &Tensor,
        xv: &Tensor,
        pos: usize,
        rings: &mut KvRings,
    ) -> Result<StageIn> {
        let KvRings { k: ring_k, v: ring_v } = rings;
        let h = w.hidden;
        let c = w.conv_channels();
        for (name, r) in [("key", &*ring_k), ("value", &*ring_v)] {
            if r.taps != w.taps || r.c != c {
                candle_core::bail!("stage_in: {name} ring of {} taps over {} for {} over {c}", r.taps, r.c, w.taps);
            }
        }
        let l = &w.lay;
        let pargs = Words::default().u(h).u(l.rows).u(l.q_rows).u(l.k_rows);
        let prod = Launch {
            name: "attn-dense-products",
            pipe: &self.products,
            args: vec![
                Arg::Primary,
                Arg::Tensor(xk, h),
                Arg::Tensor(xv, h),
                Arg::Owned(&w.rows, false),
                Arg::Out,
                Arg::Bytes(pargs),
            ],
            out_elems: l.rows,
            out_shape: Shape::from(l.rows),
            grid: (l.rows.div_ceil(MV_TG / 32), MV_TG),
        };
        if xq.elem_count() != h {
            candle_core::bail!("stage_in: query input of {} elements, want {h}", xq.elem_count());
        }
        let p = xq.apply_op1_no_bwd(&prod)?;
        let (nh, nkv, hd) = (w.heads, w.kv_heads, w.head_dim);
        let conv_scale = if w.dynamic {
            1.0 / ((2 * w.taps + 1) as f64).sqrt()
        } else {
            1.0 / ((w.taps + 1) as f64).sqrt()
        } as f32;
        let fargs = Words::default()
            .u(nh)
            .u(nkv)
            .u(hd)
            .u(w.d_rel)
            .u(w.extent)
            .u(w.taps)
            .u(w.dynamic as usize)
            .u(!w.sliding as usize)
            .u(ring_k.pos)
            .u(ring_v.pos)
            .u(l.off_r)
            .u(l.off_s)
            .u(l.off_g)
            .u(l.off_k)
            .u(l.off_pk)
            .u(l.off_v)
            .u(l.off_pv)
            .u(l.so_rs)
            .u(l.so_rp)
            .u(l.so_kc)
            .u(l.so_vc)
            .u(l.oo_k)
            .u(l.oo_v)
            .u(l.oo_rel)
            .u(l.oo_s)
            .u(l.oo_g)
            .f(w.qk_thr)
            .f(w.rel_thr)
            .f(CLIP_EPS as f32)
            .f(w.eps)
            .f((1.0 / hd as f64) as f32)
            .f(if w.sliding { 1.0 } else { tau_of(w.log_alpha, w.log_floor, pos) })
            .f(conv_scale);
        let fin = Launch {
            name: "attn-dense-finish",
            pipe: &self.finish,
            args: vec![
                Arg::Primary,
                Arg::Owned(&w.small, false),
                Arg::Owned(&ring_k.buf, true),
                Arg::Owned(&ring_v.buf, true),
                Arg::Out,
                Arg::Bytes(fargs),
            ],
            out_elems: l.out,
            out_shape: Shape::from(l.out),
            grid: (nh + 2 * nkv, FIN_TG),
        };
        let o = p.apply_op1_no_bwd(&fin)?;
        ring_k.pos = (ring_k.pos + 1) % (w.taps - 1);
        ring_v.pos = (ring_v.pos + 1) % (w.taps - 1);
        Ok(StageIn {
            q: o.narrow(0, 0, nh * hd)?.reshape((nh, 1, hd))?,
            k: o.narrow(0, l.oo_k, nkv * hd)?.reshape((nkv, 1, hd))?,
            v: o.narrow(0, l.oo_v, nkv * hd)?.reshape((nkv, 1, hd))?,
            rel_logits: o.narrow(0, l.oo_rel, nh * w.extent)?.reshape((nh, 1, w.extent))?,
            sinks: o.narrow(0, l.oo_s, nh)?.reshape((nh, 1, 1))?,
            gate: o.narrow(0, l.oo_g, nh * hd)?.reshape((1, nh * hd))?,
        })
    }

    /// `(a * gate) @ wo.T`, `[1, hidden]`: `a` is the attention output
    /// `[1, heads * head_dim]` of the score step, `gate` that of [`Self::stage_in`].
    pub fn stage_out(&self, w: &AttnDenseWeights, a: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let din = w.heads * w.head_dim;
        if a.elem_count() != din {
            candle_core::bail!("stage_out: attention output of {} elements, want {din}", a.elem_count());
        }
        let args = Words::default().u(din).u(w.hidden);
        let op = Launch {
            name: "attn-dense-out",
            pipe: &self.out,
            args: vec![Arg::Primary, Arg::Tensor(gate, din), Arg::Owned(&w.wo, false), Arg::Out, Arg::Bytes(args)],
            out_elems: w.hidden,
            out_shape: Shape::from((1, w.hidden)),
            grid: (w.hidden.div_ceil(MV_TG / 32), MV_TG),
        };
        a.apply_op1_no_bwd(&op)
    }
}

/// One buffer argument of a launch, in buffer-index order.
// On CUDA a pointer carries no usage, so `Owned`'s flag is read on Metal only.
#[cfg_attr(all(target_os = "linux", feature = "cuda"), allow(dead_code))]
enum Arg<'a> {
    /// The tensor the op is applied to.
    Primary,
    /// Another f32 tensor on the device, contiguous, of this many elements.
    Tensor(&'a Tensor, usize),
    /// A buffer this module owns; `true` if the kernel writes it.
    Owned(&'a Buf, bool),
    /// The output.
    Out,
    /// A small constant struct.
    Bytes(Words),
}

/// One kernel launch as a candle op; `grid` is (threadgroups, threads a group).
struct Launch<'a> {
    name: &'static str,
    pipe: &'a Pipe,
    args: Vec<Arg<'a>>,
    out_elems: usize,
    out_shape: Shape,
    grid: (usize, usize),
}

struct Bound<'a> {
    guard: RwLockReadGuard<'a, Storage>,
    offset: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Bound<'_> {
    fn buffer(&self) -> &Buffer {
        match &*self.guard {
            Storage::Metal(m) => m.buffer(),
            _ => unreachable!("checked at bind"),
        }
    }
}

fn bind<'a>(op: &str, i: usize, t: &'a Tensor, want: usize) -> Result<Bound<'a>> {
    let (guard, layout) = t.storage_and_layout();
    if t.dtype() != DType::F32 {
        candle_core::bail!("{op}: buffer {i} is {:?}, the kernels take f32", t.dtype());
    }
    if !layout.is_contiguous() {
        candle_core::bail!("{op}: buffer {i} is not contiguous");
    }
    if t.elem_count() != want {
        candle_core::bail!("{op}: buffer {i} has {} elements, want {want}", t.elem_count());
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let here = matches!(&*guard, Storage::Metal(_));
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let here = matches!(&*guard, Storage::Cuda(_));
    if !here {
        candle_core::bail!("{op}: buffer {i} is not on the device of the kernels");
    }
    let offset = layout.start_offset() * DType::F32.size_in_bytes();
    Ok(Bound { guard, offset })
}

impl CustomOp1 for Launch<'_> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: GPU only; its CPU reference is the layers.rs functions", self.name)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("{}: the primary input must be contiguous", self.name);
        }
        let dev = storage.device.clone();
        let stream = dev.cuda_stream();
        let f32_addr = |s: &CudaStorage, elems_off: usize| -> Result<u64> {
            Ok(s.as_cuda_slice::<f32>()?.device_ptr(&stream).0 + 4 * elems_off as u64)
        };
        let mut bound: Vec<Option<Bound>> = Vec::with_capacity(self.args.len());
        for (i, a) in self.args.iter().enumerate() {
            bound.push(match a {
                Arg::Tensor(t, want) => Some(bind(self.name, i, t, *want)?),
                _ => None,
            });
        }
        // Uninitialised: every kernel writes every word of its output.
        let out = unsafe { stream.alloc::<f32>(self.out_elems) }
            .map_err(|e| candle_core::Error::msg(format!("{}: alloc of the output: {e}", self.name)))?;
        // Pointers as their device addresses, the constant struct as its words: one kernel parameter each.
        let mut ptrs: Vec<u64> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        for (i, a) in self.args.iter().enumerate() {
            match a {
                Arg::Primary => ptrs.push(f32_addr(storage, layout.start_offset())?),
                Arg::Tensor(..) => {
                    let b = bound[i].as_ref().expect("bound above");
                    let Storage::Cuda(c) = &*b.guard else { unreachable!("checked at bind") };
                    ptrs.push(f32_addr(c, b.offset / 4)?);
                }
                Arg::Owned(b, _) => ptrs.push(match b {
                    Buf::U16(s) => s.device_ptr(&stream).0,
                    Buf::F32(s) => s.device_ptr(&stream).0,
                }),
                Arg::Out => ptrs.push(out.device_ptr(&stream).0),
                Arg::Bytes(w) => words.extend_from_slice(&w.0),
            }
        }
        let (groups, threads) = self.grid;
        let cfg = LaunchConfig { grid_dim: (groups as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b = stream.launch_builder(self.pipe);
        for p in &ptrs {
            b.arg(p);
        }
        for w in &words {
            b.arg(w);
        }
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("{}: {e}", self.name)))?;
        drop(bound);
        Ok((CudaStorage::wrap_cuda_slice(out, dev), self.out_shape.clone()))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("{}: the primary input must be contiguous f32", self.name);
        }
        let dev = storage.device().clone();
        // Bind the other tensors first: their read guards live to the end of the encode.
        let mut bound: Vec<Option<Bound>> = Vec::with_capacity(self.args.len());
        for (i, a) in self.args.iter().enumerate() {
            bound.push(match a {
                Arg::Tensor(t, want) => Some(bind(self.name, i, t, *want)?),
                _ => None,
            });
        }
        let out = dev.new_buffer(self.out_elems, DType::F32, self.name)?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(self.pipe);
        for (i, a) in self.args.iter().enumerate() {
            match a {
                Arg::Primary => {
                    let off = layout.start_offset() * DType::F32.size_in_bytes();
                    enc.set_buffer(i, Some(storage.buffer()), off);
                    enc.use_resource(storage.buffer(), MTLResourceUsage::Read);
                }
                Arg::Tensor(..) => {
                    let b = bound[i].as_ref().expect("bound above");
                    enc.set_buffer(i, Some(b.buffer()), b.offset);
                    enc.use_resource(b.buffer(), MTLResourceUsage::Read);
                }
                Arg::Owned(b, writes) => {
                    let b: &Buffer = b;
                    enc.set_buffer(i, Some(b), 0);
                    let usage =
                        if *writes { MTLResourceUsage::Read | MTLResourceUsage::Write } else { MTLResourceUsage::Read };
                    enc.use_resource(b, usage);
                }
                Arg::Out => {
                    enc.set_buffer(i, Some(&out), 0);
                    enc.use_resource(&*out, MTLResourceUsage::Write);
                }
                Arg::Bytes(w) => enc.set_bytes_directly(i, 4 * w.0.len(), w.0.as_ptr() as *const std::ffi::c_void),
            }
        }
        let (groups, threads) = self.grid;
        enc.dispatch_thread_groups(
            MTLSize { width: groups, height: 1, depth: 1 },
            MTLSize { width: threads, height: 1, depth: 1 },
        );
        drop(enc);
        drop(bound);
        Ok((MetalStorage::new(out, dev, self.out_elems, DType::F32), self.out_shape.clone()))
    }
}
