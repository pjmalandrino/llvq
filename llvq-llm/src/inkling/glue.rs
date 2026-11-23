//! Fused glue kernels for one decoded row of an Inkling 2 Mini layer visit, on Metal.
//!
//! Plan item D7, the part that does not touch the attention cache. A visit
//! today runs its norms, message gates, block residual attention, short
//! convolutions and router as separate candle operations, one GPU launch each.
//! These kernels do the same work, in f32, in four launches:
//!
//!   - [`Glue::pre`]: block residual attention over the block states, the
//!     query, key and value message gates when the visit has messages, and the
//!     input norms, to the normed attention inputs. One launch, one
//!     threadgroup an output (1 or 3).
//!   - [`Glue::post_attn`]: the post-attention short convolution on its ring,
//!     the residual or the block boundary, the block residual attention for
//!     the MLP input, its message gate, the norm, and the router's sigmoid
//!     scores. One launch, one threadgroup.
//!   - [`Glue::top8`]: `score + bias`, the top k with the reference tie rule
//!     (higher value in `f32::total_cmp` order, then lower index), and the
//!     weights normalised as `layers::route` does. One launch.
//!   - [`Glue::post_mlp`]: the post-MLP short convolution on its ring and the
//!     residual. One launch.
//!
//! Exactness, per kernel (gated by `tests/inkling_glue.rs`):
//!   - `post_mlp`, and the residual `post_attn` writes, are the reference's
//!     operations in the reference's order: bit for bit against `short_conv`
//!     and the residual add of `layers.rs`, on the CPU and on Metal. The
//!     library is compiled with `fp contract(off)` for that: a fused
//!     multiply-add would round once where candle rounds twice.
//!   - `top8` is bit for bit against `route` on the same scores, ties and
//!     values one float apart included.
//!   - `pre` and the normed input and scores of `post_attn` are class B: their
//!     sums (the norms, the block attention, the gate and router products) run
//!     in a threadgroup tree, not in candle's GEMM order.
//!
//! Each convolution keeps its state in a [`ConvRing`]: `K - 1` rows on the
//! device that the kernel reads and rewrites in place, oldest row replaced,
//! so no step concatenates.
//!
//! `LayerWeights::forward_fused` (`layers.rs`) calls them for a one-row step
//! when `Inkling::set_glue` chose them; a slot then holds its two rings in a
//! [`FusedRings`]. The inputs are plain tensors, so a change of the layer
//! structs does not reach this file.
//!
//! Same candle traps as `crate::fused_metal` (its header): the compile panics
//! on a bad source, so it runs under `catch_unwind`; the ring buffers are
//! OWNED, so the allocator never hands them out as scratch; every output word
//! is written by the kernel, so no pooled buffer leaks a stale value.
#![cfg(ink_glue)]

use std::sync::RwLockReadGuard;

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

/// A compiled kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
type Pipe = ComputePipeline;
#[cfg(all(target_os = "linux", feature = "cuda"))]
type Pipe = CudaFunction;
/// A buffer this module owns on the device (a ring).
#[cfg(all(target_os = "macos", feature = "metal"))]
type Buf = std::sync::Arc<Buffer>;
#[cfg(all(target_os = "linux", feature = "cuda"))]
type Buf = CudaSlice<f32>;

/// Block states a call may carry (the model holds at most 4).
pub const MAX_STATES: usize = 6;
/// Largest hidden width the kernels stage in threadgroup memory.
pub const MAX_H: usize = 4096;
/// Largest message-gate rank.
pub const MAX_RANK: usize = 256;
/// Largest expert count `top8` ranks.
pub const MAX_EXPERTS: usize = 1024;
/// Threads of the per-row kernels' threadgroup.
const TG: usize = 1024;
/// The epsilon of `block_attn_res`, fixed in `layers.rs` whatever the config's.
const BAR_EPS: f64 = 1e-6;
/// Channels one simdgroup of `glue_gate_apply` handles: the grain of that
/// launch. The result does not depend on it.
const GATE_CHANNELS: usize = 16;
/// Threads a threadgroup of `post_mlp` and of the ring read.
const ELT_GROUP: usize = 256;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const SRC_CUDA: &str = include_str!("../../kernels/ink_glue.cu");

#[cfg(all(target_os = "macos", feature = "metal"))]
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#define NSG (GLUE_TG / 32u)
#define MAX_STATES 6u
#define MAX_H 4096u
#define MAX_RANK 256u
#define MAX_EXPERTS 1024u

struct PreArgs {
    uint h; uint n_states; uint nq; uint has_msg; uint n_norms; uint rank;
    float eps; float bar_eps; float inv_h; float bar_scale;
};

struct PostArgs {
    uint h; uint n_states; uint boundary; uint has_msg; uint rank; uint experts; uint k; uint pos;
    float eps; float bar_eps; float inv_h; float bar_scale; float conv_scale;
};

struct TopArgs { uint experts; uint k; float scale; };

struct ConvArgs { uint h; uint k; uint pos; float conv_scale; };

// The sum of v over the threadgroup, every thread gets it. Fixed tree.
static float tg_sum(float v, threadgroup float *scr, uint lane, uint sg) {
    float s = simd_sum(v);
    if (lane == 0) { scr[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < NSG ? scr[lane] : 0.0f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// 1 / sqrt(mean(x^2) + eps), the factor of rms_unweighted.
template <typename P>
static float inv_rms(P x, uint n, float inv_n, float eps, threadgroup float *scr, uint tid, uint lane, uint sg) {
    float acc = 0.0f;
    for (uint c = tid; c < n; c += GLUE_TG) { float v = x[c]; acc = acc + v * v; }
    float ss = tg_sum(acc, scr, lane, sg);
    float ms = ss * inv_n;
    ms = ms + eps;
    return 1.0f / sqrt(ms);
}

// Block residual attention for one query over n sources (block_attn_res):
// every source unweighted-normed, logits q.s * h^-0.5, softmax, weighted sum.
template <typename S>
static void bar_one(device const float *q, S src, uint n, uint h, float inv_h,
                    float eps, float scale, threadgroup float *X, threadgroup float *scr,
                    uint tid, uint lane, uint sg) {
    float inv[MAX_STATES + 2];
    float w[MAX_STATES + 2];
    for (uint i = 0; i < n; i++) { inv[i] = inv_rms(src[i], h, inv_h, eps, scr, tid, lane, sg); }
    for (uint i = 0; i < n; i++) {
        float acc = 0.0f;
        for (uint c = tid; c < h; c += GLUE_TG) { float s = src[i][c] * inv[i]; acc = acc + q[c] * s; }
        w[i] = tg_sum(acc, scr, lane, sg) * scale;
    }
    float m = w[0];
    for (uint i = 1; i < n; i++) { m = max(m, w[i]); }
    float z = 0.0f;
    for (uint i = 0; i < n; i++) { w[i] = exp(w[i] - m); z = z + w[i]; }
    for (uint i = 0; i < n; i++) { w[i] = w[i] / z; }
    for (uint c = tid; c < h; c += GLUE_TG) {
        float acc = 0.0f;
        for (uint i = 0; i < n; i++) { float s = src[i][c] * inv[i]; acc = acc + w[i] * s; }
        X[c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// MessageGate::forward in place on X: (1 - tanh(F g)) * X + tanh(G g) * m,
// g = down [rms(X), rms(m)].
static void gate_inplace(threadgroup float *X, device const float *m, device const float *down,
                         device const float *forget, device const float *gain, uint h, uint rank,
                         float inv_h, float eps, threadgroup float *G, threadgroup float *scr,
                         uint tid, uint lane, uint sg) {
    float ix = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
    float im = inv_rms(m, h, inv_h, eps, scr, tid, lane, sg);
    for (uint r = sg; r < rank; r += NSG) {
        device const float *row = down + r * 2u * h;
        float acc = 0.0f;
        for (uint c = lane; c < h; c += 32u) { float v = X[c] * ix; acc = acc + row[c] * v; }
        for (uint c = lane; c < h; c += 32u) { float v = m[c] * im; acc = acc + row[h + c] * v; }
        acc = simd_sum(acc);
        if (lane == 0) { G[r] = acc; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = sg; c < h; c += NSG) {
        float af = 0.0f;
        float ag = 0.0f;
        for (uint r = lane; r < rank; r += 32u) {
            af = af + forget[c * rank + r] * G[r];
            ag = ag + gain[c * rank + r] * G[r];
        }
        af = simd_sum(af);
        ag = simd_sum(ag);
        if (lane == 0) {
            float f = tanh(af);
            float g = tanh(ag);
            float keep = 1.0f - f;
            float a = keep * X[c];
            float b = g * m[c];
            X[c] = a + b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void glue_pre(
    device const float *hin [[buffer(0)]],
    device const float *s0 [[buffer(1)]], device const float *s1 [[buffer(2)]],
    device const float *s2 [[buffer(3)]], device const float *s3 [[buffer(4)]],
    device const float *s4 [[buffer(5)]], device const float *s5 [[buffer(6)]],
    device const float *queries [[buffer(7)]],
    device const float *norm0 [[buffer(8)]], device const float *norm1 [[buffer(9)]],
    device const float *norm2 [[buffer(10)]],
    device const float *msg0 [[buffer(11)]], device const float *msg1 [[buffer(12)]],
    device const float *msg2 [[buffer(13)]],
    device const float *down0 [[buffer(14)]], device const float *forget0 [[buffer(15)]],
    device const float *gain0 [[buffer(16)]],
    device const float *down1 [[buffer(17)]], device const float *forget1 [[buffer(18)]],
    device const float *gain1 [[buffer(19)]],
    device const float *down2 [[buffer(20)]], device const float *forget2 [[buffer(21)]],
    device const float *gain2 [[buffer(22)]],
    device float *out [[buffer(23)]],
    constant PreArgs &a [[buffer(24)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float X[MAX_H];
    threadgroup float G[MAX_RANK];
    threadgroup float scr[32];
    const uint h = a.h;
    if (a.nq == 0u) {
        for (uint c = tid; c < h; c += GLUE_TG) { X[c] = hin[c]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        device const float *src[MAX_STATES + 2];
        device const float *st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
        for (uint i = 0; i < a.n_states; i++) { src[i] = st[i]; }
        src[a.n_states] = hin;
        uint qi = a.nq == 3u ? j : 0u;
        bar_one(queries + qi * h, src, a.n_states + 1u, h, a.inv_h, a.bar_eps, a.bar_scale, X, scr, tid, lane, sg);
    }
    if (a.has_msg == 2u) {
        // The gate runs in launches of its own (glue_gate_down, glue_gate_apply,
        // glue_gate_norm): hand them X and the two factors gate_inplace starts with.
        device const float *m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
        float ix = inv_rms(X, h, a.inv_h, a.eps, scr, tid, lane, sg);
        float im = inv_rms(m, h, a.inv_h, a.eps, scr, tid, lane, sg);
        for (uint c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }
        if (tid == 0u) {
            out[3u * h + 2u * j] = ix;
            out[3u * h + 2u * j + 1u] = im;
        }
        return;
    }
    if (a.has_msg != 0u) {
        device const float *m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
        device const float *dw = j == 0u ? down0 : (j == 1u ? down1 : down2);
        device const float *fw = j == 0u ? forget0 : (j == 1u ? forget1 : forget2);
        device const float *gw = j == 0u ? gain0 : (j == 1u ? gain1 : gain2);
        gate_inplace(X, m, dw, fw, gw, h, a.rank, a.inv_h, a.eps, G, scr, tid, lane, sg);
    }
    uint ni = a.n_norms == 3u ? j : 0u;
    device const float *w = ni == 0u ? norm0 : (ni == 1u ? norm1 : norm2);
    float inv = inv_rms(X, h, a.inv_h, a.eps, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) { float v = X[c] * inv; out[j * h + c] = v * w[c]; }
}

// The message queries of a loop visit: block_attn_res of each query over the
// previous pass's block states and its hidden state. One threadgroup a query.
kernel void glue_msg(
    device const float *partial [[buffer(0)]],
    device const float *s0 [[buffer(1)]], device const float *s1 [[buffer(2)]],
    device const float *s2 [[buffer(3)]], device const float *s3 [[buffer(4)]],
    device const float *s4 [[buffer(5)]], device const float *s5 [[buffer(6)]],
    device const float *queries [[buffer(7)]],
    device float *out [[buffer(8)]],
    constant PreArgs &a [[buffer(9)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float X[MAX_H];
    threadgroup float scr[32];
    const uint h = a.h;
    device const float *src[MAX_STATES + 2];
    device const float *st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
    for (uint i = 0; i < a.n_states; i++) { src[i] = st[i]; }
    src[a.n_states] = partial;
    bar_one(queries + j * h, src, a.n_states + 1u, h, a.inv_h, a.bar_eps, a.bar_scale, X, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }
}

// One channel of short_conv on the ring: taps oldest first, then x; the
// residual and the output scale. The newest input replaces the oldest row.
static float conv_ring(device float *ring, device const float *w, float x, uint c, uint h, uint k, uint pos,
                       float scale) {
    uint k1 = k - 1u;
    float acc = 0.0f;
    for (uint t = 0; t < k1; t++) {
        float v = ring[((pos + t) % k1) * h + c];
        float p = v * w[c * k + t];
        acc = acc + p;
    }
    float p = x * w[c * k + k1];
    acc = acc + p;
    ring[pos * h + c] = x;
    float o = acc + x;
    return o * scale;
}

kernel void glue_post_attn(
    device const float *attn [[buffer(0)]],
    device float *ring [[buffer(1)]],
    device const float *conv_w [[buffer(2)]],
    device const float *hin [[buffer(3)]],
    device const float *s0 [[buffer(4)]], device const float *s1 [[buffer(5)]],
    device const float *s2 [[buffer(6)]], device const float *s3 [[buffer(7)]],
    device const float *s4 [[buffer(8)]], device const float *s5 [[buffer(9)]],
    device const float *mlp_q [[buffer(10)]],
    device const float *msg [[buffer(11)]],
    device const float *down [[buffer(12)]], device const float *forget [[buffer(13)]],
    device const float *gain [[buffer(14)]],
    device const float *norm_w [[buffer(15)]],
    device const float *router_w [[buffer(16)]],
    device float *out [[buffer(17)]],
    constant PostArgs &a [[buffer(18)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float X[MAX_H];
    threadgroup float G[MAX_RANK];
    threadgroup float scr[32];
    const uint h = a.h;
    for (uint c = tid; c < h; c += GLUE_TG) {
        float v = conv_ring(ring, conv_w, attn[c], c, h, a.k, a.pos, a.conv_scale);
        out[c] = a.boundary != 0u ? v : hin[c] + v;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    device const float *src[MAX_STATES + 2];
    device const float *st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
    uint n = 0;
    for (uint i = 0; i < a.n_states; i++) { src[n++] = st[i]; }
    if (a.boundary != 0u) { src[n++] = hin; }
    src[n++] = out;
    bar_one(mlp_q, src, n, h, a.inv_h, a.bar_eps, a.bar_scale, X, scr, tid, lane, sg);
    if (a.has_msg == 2u) {
        // The gate, the norm and the scores run in launches of their own.
        float ix = inv_rms(X, h, a.inv_h, a.eps, scr, tid, lane, sg);
        float im = inv_rms(msg, h, a.inv_h, a.eps, scr, tid, lane, sg);
        for (uint c = tid; c < h; c += GLUE_TG) { out[h + c] = X[c]; }
        if (tid == 0u) {
            out[2u * h] = ix;
            out[2u * h + 1u] = im;
        }
        return;
    }
    if (a.has_msg != 0u) {
        gate_inplace(X, msg, down, forget, gain, h, a.rank, a.inv_h, a.eps, G, scr, tid, lane, sg);
    }
    float inv = inv_rms(X, h, a.inv_h, a.eps, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) {
        float v = X[c] * inv;
        v = v * norm_w[c];
        X[c] = v;
        out[h + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = sg; e < a.experts; e += NSG) {
        device const float *row = router_w + e * h;
        float acc = 0.0f;
        for (uint c = lane; c < h; c += 32u) { acc = acc + row[c] * X[c]; }
        acc = simd_sum(acc);
        if (lane == 0) {
            float z = exp(-acc) + 1.0f;
            out[2u * h + e] = 1.0f / z;
        }
    }
}

struct GateArgs { uint h; uint rank; uint n; uint n_norms; float eps; float inv_h; uint per; };

// MessageGate::forward spread over threadgroups, for n gates at once. The
// sums are gate_inplace's, lane for lane; one threadgroup read a gate's
// weights at a fraction of the device's bandwidth.
// xa: X [n, h], then (1 / rms(X), 1 / rms(m)) a gate, as stage A wrote them.
// Step 1: g = down [rms-normed X, rms-normed m], one simdgroup a row. out: [n, rank].
kernel void glue_gate_down(
    device const float *xa [[buffer(0)]],
    device const float *msg0 [[buffer(1)]], device const float *msg1 [[buffer(2)]],
    device const float *msg2 [[buffer(3)]],
    device const float *down0 [[buffer(4)]], device const float *down1 [[buffer(5)]],
    device const float *down2 [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant GateArgs &a [[buffer(8)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint h = a.h;
    const uint j = s / a.rank;
    const uint r = s % a.rank;
    device const float *X = xa + j * h;
    device const float *m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
    device const float *dw = j == 0u ? down0 : (j == 1u ? down1 : down2);
    const float ix = xa[a.n * h + 2u * j];
    const float im = xa[a.n * h + 2u * j + 1u];
    device const float *row = dw + r * 2u * h;
    float acc = 0.0f;
    for (uint c = lane; c < h; c += 32u) { float v = X[c] * ix; acc = acc + row[c] * v; }
    for (uint c = lane; c < h; c += 32u) { float v = m[c] * im; acc = acc + row[h + c] * v; }
    acc = simd_sum(acc);
    if (lane == 0) { out[s] = acc; }
}

// Step 2: (1 - tanh(F g)) * X + tanh(G g) * m, a.per channels a simdgroup, each
// channel with the lanes of gate_inplace. out: [n, h].
kernel void glue_gate_apply(
    device const float *xa [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const float *msg0 [[buffer(2)]], device const float *msg1 [[buffer(3)]],
    device const float *msg2 [[buffer(4)]],
    device const float *forget0 [[buffer(5)]], device const float *forget1 [[buffer(6)]],
    device const float *forget2 [[buffer(7)]],
    device const float *gain0 [[buffer(8)]], device const float *gain1 [[buffer(9)]],
    device const float *gain2 [[buffer(10)]],
    device float *out [[buffer(11)]],
    constant GateArgs &a [[buffer(12)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint h = a.h;
    const uint rank = a.rank;
    const uint groups = (h + a.per - 1u) / a.per;
    const uint j = s / groups;
    const uint c0 = (s % groups) * a.per;
    device const float *m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
    device const float *fw = j == 0u ? forget0 : (j == 1u ? forget1 : forget2);
    device const float *gw = j == 0u ? gain0 : (j == 1u ? gain1 : gain2);
    device const float *G = g + j * rank;
    for (uint c = c0; c < min(c0 + a.per, h); c++) {
        float af = 0.0f;
        float ag = 0.0f;
        for (uint r = lane; r < rank; r += 32u) {
            af = af + fw[c * rank + r] * G[r];
            ag = ag + gw[c * rank + r] * G[r];
        }
        af = simd_sum(af);
        ag = simd_sum(ag);
        if (lane == 0) {
            float f = tanh(af);
            float gg = tanh(ag);
            float keep = 1.0f - f;
            float p = keep * xa[j * h + c];
            float q = gg * m[c];
            out[j * h + c] = p + q;
        }
    }
}

// Step 3: the weighted RMS norm that ends glue_pre and the normed input of
// glue_post_attn, one threadgroup a gate. out: [n, h].
kernel void glue_gate_norm(
    device const float *x [[buffer(0)]],
    device const float *norm0 [[buffer(1)]], device const float *norm1 [[buffer(2)]],
    device const float *norm2 [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant GateArgs &a [[buffer(5)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scr[32];
    const uint h = a.h;
    device const float *X = x + j * h;
    uint ni = a.n_norms == 3u ? j : 0u;
    device const float *w = ni == 0u ? norm0 : (ni == 1u ? norm1 : norm2);
    float inv = inv_rms(X, h, a.inv_h, a.eps, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) { float v = X[c] * inv; out[j * h + c] = v * w[c]; }
}

struct RouterArgs { uint h; uint experts; };

// The router's scores, sigmoid(router_w . xn), one simdgroup an expert: the
// sums of glue_post_attn's last loop, each in a threadgroup of its own.
kernel void glue_router(
    device const float *xn [[buffer(0)]],
    device const float *router_w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant RouterArgs &a [[buffer(3)]],
    uint e [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint h = a.h;
    device const float *row = router_w + e * h;
    float acc = 0.0f;
    for (uint c = lane; c < h; c += 32u) { acc = acc + row[c] * xn[c]; }
    acc = simd_sum(acc);
    if (lane == 0) {
        float z = exp(-acc) + 1.0f;
        out[e] = 1.0f / z;
    }
}

// The total order of f32::total_cmp, as a signed key.
static int total_key(float v) {
    int k = as_type<int>(v);
    k ^= int(uint(k >> 31) >> 1);
    return k;
}

// out: sel [E] | weights [k] | margin | ids [k] as f32.
kernel void glue_top8(
    device const float *scores [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant TopArgs &a [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float SEL[MAX_EXPERTS];
    threadgroup int KEY[MAX_EXPERTS];
    threadgroup uint RANK[MAX_EXPERTS];
    const uint e = a.experts;
    const uint k = a.k;
    for (uint i = tid; i < e; i += GLUE_TG) {
        float v = scores[i] + bias[i];
        SEL[i] = v;
        KEY[i] = total_key(v);
        out[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < e; i += GLUE_TG) {
        int ki = KEY[i];
        uint r = 0;
        for (uint j = 0; j < e; j++) {
            int kj = KEY[j];
            r += (kj > ki || (kj == ki && j < i)) ? 1u : 0u;
        }
        RANK[i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = -0.0f;
        float last = 0.0f;
        float next = 0.0f;
        for (uint i = 0; i < e; i++) {
            uint r = RANK[i];
            if (r < k) { sum = sum + scores[i]; }
            if (r == k - 1u) { last = SEL[i]; }
            if (r == k) { next = SEL[i]; }
        }
        sum = sum + 1e-20f;
        uint n = 0;
        for (uint i = 0; i < e; i++) {
            if (RANK[i] < k) {
                float w = precise::divide(scores[i], sum);
                out[e + n] = w * a.scale;
                out[e + k + 1u + n] = float(i);
                n++;
            }
        }
        out[e + k] = last - next;
    }
}

// The routed experts as the MoE kernels read them: the ids [k], ascending as
// glue_top8 wrote them, then the bits of their weights [k].
kernel void glue_sel(
    device const float *top [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant TopArgs &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    const uint e = a.experts;
    const uint k = a.k;
    if (i >= k) { return; }
    out[i] = uint(top[e + k + 1u + i]);
    out[k + i] = as_type<uint>(top[e + i]);
}

kernel void glue_post_mlp(
    device const float *m [[buffer(0)]],
    device float *ring [[buffer(1)]],
    device const float *conv_w [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant ConvArgs &a [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= a.h) { return; }
    float v = conv_ring(ring, conv_w, m[c], c, a.h, a.k, a.pos, a.conv_scale);
    out[c] = residual[c] + v;
}

// The ring in logical order, oldest row first: what SlotCache holds.
kernel void glue_ring_state(
    device const float *ring [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant ConvArgs &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint k1 = a.k - 1u;
    if (i >= k1 * a.h) { return; }
    uint t = i / a.h;
    uint c = i % a.h;
    out[i] = ring[((a.pos + t) % k1) * a.h + c];
}
"#;

/// A constant struct of the MSL as its 4-byte words, in field order: `u32`
/// fields as they are, `f32` fields by their bits.
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

/// The scale of `short_conv`: `out / sqrt(K + 1)` is candle's `affine` with
/// `mul = 1 / sqrt(K + 1)` in f64, narrowed to f32 by the kernel.
fn conv_scale(k: usize) -> f32 {
    (1.0 / ((k + 1) as f64).sqrt()) as f32
}

/// The weights of one message gate (`layers::MessageGate`).
#[derive(Clone, Copy)]
pub struct GateW<'a> {
    /// `[rank, 2H]`.
    pub down: &'a Tensor,
    /// `[H, rank]`.
    pub forget: &'a Tensor,
    /// `[H, rank]`.
    pub gain: &'a Tensor,
}

/// What [`Glue::pre`] reads: the attention-input half of a visit.
pub struct PreInputs<'a> {
    /// The layer input `[1, H]`.
    pub h: &'a Tensor,
    /// The block states before this layer, each `[1, H]`.
    pub states: &'a [Tensor],
    /// The attention-input queries `[nq, H]`, 1 in a sliding layer and 3 in a
    /// full one; `None` in layer 0, whose input is `h` itself.
    pub queries: Option<&'a Tensor>,
    /// `norms_in`: 1 in a sliding layer, 3 in a full one.
    pub norms: &'a [Tensor],
    /// The query, key and value messages and their gates, in the loop passes after the first.
    pub gates: Option<([&'a Tensor; 3], [GateW<'a>; 3])>,
    pub sliding: bool,
    pub eps: f64,
}

/// What [`Glue::post_attn`] reads.
pub struct PostAttnInputs<'a> {
    /// The attention output `[1, H]`.
    pub attn_out: &'a Tensor,
    /// The layer input `[1, H]` (the block residual).
    pub h: &'a Tensor,
    /// The block states before this layer.
    pub states: &'a [Tensor],
    pub boundary: bool,
    /// `post_attention_convolution` `[H, K]`.
    pub conv_w: &'a Tensor,
    /// `mlp_input` `[1, H]`.
    pub mlp_query: &'a Tensor,
    /// The MLP message and its gate.
    pub gate: Option<(&'a Tensor, GateW<'a>)>,
    /// `post_attention_layernorm` `[H]`.
    pub norm: &'a Tensor,
    /// `router_w` `[E, H]`.
    pub router_w: &'a Tensor,
    pub eps: f64,
}

/// The outputs of [`Glue::post_attn`], views of one device buffer.
pub struct PostAttn {
    /// The residual after attention `[1, H]`: what `post_mlp` adds to.
    pub residual: Tensor,
    /// The normed MLP input `[1, H]`.
    pub xn: Tensor,
    /// `sigmoid(xn @ router_w.T)` `[E]`, before the bias.
    pub scores: Tensor,
}

/// The outputs of [`Glue::top8`], views of one device buffer.
pub struct Top8 {
    /// `score + bias` of every expert `[E]`: what the top k selected on.
    pub sel: Tensor,
    /// The weights of the chosen experts `[k]`, aligned with `ids`.
    pub weights: Tensor,
    /// The k-th minus the (k+1)-th largest `sel` `[1]`.
    pub margin: Tensor,
    /// The chosen experts ascending `[k]`, as f32 (exact below 2^24).
    pub ids: Tensor,
    /// The whole output `[E + 2k + 1]`, `sel | weights | margin | ids`: what
    /// one read back to the host takes.
    pub raw: Tensor,
}

/// What [`Top8::read`] brings back to the host.
pub struct TopRead {
    pub sel: Vec<f32>,
    pub weights: Vec<f32>,
    pub margin: f32,
    pub ids: Vec<u32>,
}

impl Top8 {
    /// Read the output back once and split it: `sel`, the weights, the
    /// margin and the expert ids, for `layers::Routing`.
    pub fn read(&self, k: usize) -> Result<TopRead> {
        let v = self.raw.to_vec1::<f32>()?;
        if v.len() < 2 * k + 2 {
            candle_core::bail!("top8 output of {} values for k = {k}", v.len());
        }
        let e = v.len() - 2 * k - 1;
        let ids = v[e + k + 1..].iter().map(|&x| x as u32).collect();
        Ok(TopRead { sel: v[..e].to_vec(), weights: v[e..e + k].to_vec(), margin: v[e + k], ids })
    }
}

/// The two convolution rings of one layer visit (post-attention and
/// post-MLP) while the fused glue steps it, with what reads them back.
pub struct FusedRings {
    pub attn: ConvRing,
    pub mlp: ConvRing,
    glue: std::sync::Arc<Glue>,
    /// A one-element f32 tensor on the device: the carrier of the read-back launch.
    like: Tensor,
}

impl std::fmt::Debug for FusedRings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FusedRings {{ attn: pos {}, mlp: pos {} }}", self.attn.pos, self.mlp.pos)
    }
}

impl FusedRings {
    /// Rings filled with the states `[k - 1, c]` oldest row first (`None`:
    /// zeros, the state of an empty slot).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        glue: std::sync::Arc<Glue>,
        device: &Device,
        c: usize,
        k_attn: usize,
        attn: Option<&Tensor>,
        k_mlp: usize,
        mlp: Option<&Tensor>,
    ) -> Result<Self> {
        Ok(Self {
            attn: ConvRing::new(device, k_attn, c, attn)?,
            mlp: ConvRing::new(device, k_mlp, c, mlp)?,
            glue,
            like: Tensor::zeros(1, DType::F32, device)?,
        })
    }

    /// The two states `[k - 1, c]`, oldest row first, as new device tensors.
    pub fn states(&self) -> Result<(Tensor, Tensor)> {
        Ok((self.attn.state(&self.glue, &self.like)?, self.mlp.state(&self.glue, &self.like)?))
    }
}

/// A convolution state as a ring of `K - 1` rows of `C` on the device.
pub struct ConvRing {
    buf: Buf,
    k: usize,
    c: usize,
    /// The physical row of the oldest input.
    pos: usize,
}

impl ConvRing {
    /// A ring for a kernel of `k` taps over `c` channels, from a state
    /// `[k - 1, c]` oldest row first (what `SlotCache` holds), or zeros.
    pub fn new(device: &Device, k: usize, c: usize, state: Option<&Tensor>) -> Result<Self> {
        if k < 2 {
            candle_core::bail!("ConvRing: kernel size {k} has no state");
        }
        let host = match state {
            Some(s) => {
                if s.dims() != [k - 1, c] {
                    candle_core::bail!("ConvRing: state {:?} for [{}, {c}]", s.dims(), k - 1);
                }
                s.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
            }
            None => vec![0f32; (k - 1) * c],
        };
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let buf = {
            let Device::Metal(dev) = device else { candle_core::bail!("ConvRing wants a Metal device") };
            dev.new_buffer_with_data(&host)?
        };
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        let buf = {
            let Device::Cuda(dev) = device else { candle_core::bail!("ConvRing wants a CUDA device") };
            dev.cuda_stream().clone_htod(&host).map_err(|e| candle_core::Error::msg(format!("ConvRing: upload: {e}")))?
        };
        Ok(Self { buf, k, c, pos: 0 })
    }

    /// The state `[k - 1, c]`, oldest row first.
    pub fn state(&self, glue: &Glue, like: &Tensor) -> Result<Tensor> {
        let args = Words::default().u(self.c).u(self.k).u(self.pos).f(0.0);
        let n = (self.k - 1) * self.c;
        let op = Launch {
            name: "glue-ring-state",
            pipe: &glue.ring_state,
            args: vec![Arg::Owned(&self.buf, false), Arg::Out, Arg::Bytes(args)],
            out_elems: n,
            out_shape: Shape::from((self.k - 1, self.c)),
            out_dtype: DType::F32,
            grid: Grid::Threads(n),
        };
        like.apply_op1_no_bwd(&op)
    }
}

/// The compiled kernels.
pub struct Glue {
    pre: Pipe,
    post_attn: Pipe,
    top8: Pipe,
    post_mlp: Pipe,
    ring_state: Pipe,
    sel: Pipe,
    msg: Pipe,
    router: Pipe,
    gate_down: Pipe,
    gate_apply: Pipe,
    gate_norm: Pipe,
    /// The module the functions live in.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    _module: llvq_cuda::gpu::Cuda,
}

const NAMES: [&str; 11] = [
    "glue_pre", "glue_post_attn", "glue_top8", "glue_post_mlp", "glue_ring_state", "glue_sel", "glue_msg", "glue_router",
    "glue_gate_down", "glue_gate_apply", "glue_gate_norm",
];

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile(dev: &candle_core::MetalDevice) -> Result<Vec<ComputePipeline>> {
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let raw = dev.device().clone();
    let src = format!("#define GLUE_TG {TG}u\n{SRC}");
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
        Ok(Err(e)) => candle_core::bail!("inkling glue: Metal refused the source: {e:?}"),
        Err(panic) => {
            let what = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "no diagnostics".into());
            candle_core::bail!("inkling glue: the MSL did not compile: {what}")
        }
    }
}

impl Glue {
    /// Compile the kernels for a CUDA device (`kernels/ink_glue.cu`, NVRTC),
    /// a module of its own on the device's stream.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Cuda(dev) = device else { candle_core::bail!("the glue kernels want a CUDA device") };
        let define = format!("#define GLUE_TG {TG}u\n");
        let src = llvq_cuda::gpu::KernelSource::new(&[define.as_str(), SRC_CUDA]);
        println!("NVRTC glue unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let module = llvq_cuda::gpu::Cuda::on_stream(dev.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
        let mut p = NAMES
            .iter()
            .map(|n| module.func(n).map_err(candle_core::Error::msg))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || p.next().expect("eleven functions");
        Ok(Self {
            pre: next(),
            post_attn: next(),
            top8: next(),
            post_mlp: next(),
            ring_state: next(),
            sel: next(),
            msg: next(),
            router: next(),
            gate_down: next(),
            gate_apply: next(),
            gate_norm: next(),
            _module: module,
        })
    }

    /// Compile the kernels for a Metal device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Metal(dev) = device else { candle_core::bail!("the glue kernels want a Metal device") };
        let mut p = compile(dev)?.into_iter();
        let mut next = || p.next().expect("eleven pipelines");
        let g = Self { pre: next(), post_attn: next(), top8: next(), post_mlp: next(), ring_state: next(), sel: next(), msg: next(), router: next(), gate_down: next(), gate_apply: next(), gate_norm: next() };
        for (name, pipe) in
            [("glue_pre", &g.pre), ("glue_post_attn", &g.post_attn), ("glue_top8", &g.top8), ("glue_msg", &g.msg), ("glue_gate_norm", &g.gate_norm)]
        {
            let max = pipe.max_total_threads_per_threadgroup();
            if max < TG {
                candle_core::bail!("{name}: the pipeline takes {max} threads a threadgroup, the kernel needs {TG}");
            }
        }
        Ok(g)
    }

    /// The message inputs of a loop visit, `[nq, H]`: `layers::block_attn_res`
    /// of each of the `nq` rows of `queries` over the previous pass's block
    /// `states` and its hidden state `partial`, each `[1, H]`. One launch, a
    /// threadgroup a query, the sums of `pre`'s block residual.
    pub fn messages(&self, queries: &Tensor, states: &[Tensor], partial: &Tensor) -> Result<Tensor> {
        let h = partial.elem_count();
        check_h(h)?;
        let n_states = states.len();
        if n_states > MAX_STATES {
            candle_core::bail!("messages: {n_states} block states, at most {MAX_STATES}");
        }
        let (nq, qh) = queries.dims2()?;
        if nq == 0 || qh != h {
            candle_core::bail!("messages: queries {:?} for a hidden size of {h}", queries.dims());
        }
        // PreArgs: h, n_states, nq, has_msg, n_norms, rank, eps, bar_eps, inv_h, bar_scale.
        let args = Words::default()
            .u(h)
            .u(n_states)
            .u(nq)
            .u(0)
            .u(0)
            .u(0)
            .f(0.0)
            .f(BAR_EPS as f32)
            .f((1.0 / h as f64) as f32)
            .f((h as f64).powf(-0.5) as f32);
        let mut v: Vec<Arg> = vec![Arg::Primary];
        for i in 0..MAX_STATES {
            v.push(Arg::Tensor(states.get(i).unwrap_or(partial), h));
        }
        v.push(Arg::Tensor(queries, nq * h));
        v.push(Arg::Out);
        v.push(Arg::Bytes(args));
        let op = Launch {
            name: "glue-msg",
            pipe: &self.msg,
            args: v,
            out_elems: nq * h,
            out_shape: Shape::from((nq, h)),
            out_dtype: DType::F32,
            grid: Grid::Groups(nq),
        };
        partial.apply_op1_no_bwd(&op)
    }

    /// The normed attention inputs `[n, H]`: `n` is 1 for a sliding visit
    /// without messages, 3 otherwise (query, key, value rows).
    pub fn pre(&self, a: &PreInputs) -> Result<Tensor> {
        self.pre_with(a, true)
    }

    /// [`Self::pre`] with the message gates spread over threadgroups
    /// (`split_gates`, three more launches: [`Self::gates_split`]) or inside
    /// the kernel's one threadgroup an output. The same bits; `false` is the
    /// reference of the split's gate. A visit without messages is one launch
    /// either way.
    pub fn pre_with(&self, a: &PreInputs, split_gates: bool) -> Result<Tensor> {
        let h = a.h.elem_count();
        check_h(h)?;
        let n_states = a.states.len();
        if n_states > MAX_STATES {
            candle_core::bail!("pre: {n_states} block states, at most {MAX_STATES}");
        }
        let nq = match a.queries {
            Some(q) => q.dims2()?.0,
            None => 0,
        };
        let has_msg = a.gates.is_some();
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if has_msg && !split_gates {
            candle_core::bail!("pre: the CUDA kernels carry the split gates only");
        }
        let (want_q, want_n) = if a.sliding { (1, 1) } else { (3, 3) };
        if nq != 0 && nq != want_q {
            candle_core::bail!("pre: {nq} queries for a layer that takes {want_q}");
        }
        if a.norms.len() != want_n {
            candle_core::bail!("pre: {} norms for a layer that takes {want_n}", a.norms.len());
        }
        let nout = if a.sliding && !has_msg { 1 } else { 3 };
        let rank = match &a.gates {
            Some((_, g)) => gate_rank(g[0].down, h)?,
            None => 0,
        };
        // PreArgs: h, n_states, nq, has_msg, n_norms, rank, eps, bar_eps, inv_h, bar_scale.
        let args = Words::default()
            .u(h)
            .u(n_states)
            .u(nq)
            .u(if has_msg && split_gates { 2 } else { has_msg as usize })
            .u(want_n)
            .u(rank)
            .f(a.eps as f32)
            .f(BAR_EPS as f32)
            .f((1.0 / h as f64) as f32)
            .f((h as f64).powf(-0.5) as f32);
        let mut v: Vec<Arg> = vec![Arg::Primary];
        for i in 0..MAX_STATES {
            v.push(match a.states.get(i) {
                Some(s) => Arg::Tensor(s, h),
                None => Arg::Tensor(a.h, h),
            });
        }
        v.push(match a.queries {
            Some(q) => Arg::Tensor(q, nq * h),
            None => Arg::Tensor(a.h, h),
        });
        for i in 0..3 {
            v.push(Arg::Tensor(&a.norms[i % a.norms.len()], h));
        }
        match &a.gates {
            Some((m, g)) => {
                for mi in m {
                    v.push(Arg::Tensor(mi, h));
                }
                for gi in g {
                    v.push(Arg::Tensor(gi.down, rank * 2 * h));
                    v.push(Arg::Tensor(gi.forget, h * rank));
                    v.push(Arg::Tensor(gi.gain, h * rank));
                }
            }
            None => {
                for _ in 0..12 {
                    v.push(Arg::Tensor(a.h, h));
                }
            }
        }
        v.push(Arg::Out);
        v.push(Arg::Bytes(args));
        if let (Some((m, g)), true) = (&a.gates, split_gates) {
            // Stage A: the rows before their gate, then the two factors a gate starts with.
            let n = nout * h + 2 * nout;
            let op = Launch {
                name: "glue-pre-a",
                pipe: &self.pre,
                args: v,
                out_elems: n,
                out_shape: Shape::from(n),
                out_dtype: DType::F32,
                grid: Grid::Groups(nout),
            };
            let xa = a.h.apply_op1_no_bwd(&op)?;
            let norms: Vec<&Tensor> = a.norms.iter().collect();
            return self.gates_split(&xa, nout, h, rank, m, g, &norms, a.eps);
        }
        let op = Launch {
            name: "glue-pre",
            pipe: &self.pre,
            args: v,
            out_elems: nout * h,
            out_shape: Shape::from((nout, h)),
            out_dtype: DType::F32,
            grid: Grid::Groups(nout),
        };
        a.h.apply_op1_no_bwd(&op)
    }

    /// `n` message gates and the weighted norm after each, spread over
    /// threadgroups: `xa` holds the rows before their gate `[n, H]` and then
    /// `(1 / rms(row), 1 / rms(message))` a gate, as stage A of `pre` or of
    /// `post_attn` wrote them. Three launches: the rank-`rank` projection, one
    /// simdgroup a row of `down`; the gate itself, one simdgroup a channel;
    /// the norm, one threadgroup a gate. The sums are `gate_inplace`'s, lane
    /// for lane. Gate `j` reads `msgs[j]`, `gates[j]` and `norms[j]`, or
    /// `norms[0]` when one norm serves every row. Returns `[n, H]`.
    #[allow(clippy::too_many_arguments)]
    fn gates_split(
        &self,
        xa: &Tensor,
        n: usize,
        h: usize,
        rank: usize,
        msgs: &[&Tensor],
        gates: &[GateW],
        norms: &[&Tensor],
        eps: f64,
    ) -> Result<Tensor> {
        if n == 0 || n > 3 || msgs.len() != n || gates.len() != n || norms.is_empty() || norms.len() > 3 {
            candle_core::bail!("gates: {n} rows, {} messages, {} gates, {} norms", msgs.len(), gates.len(), norms.len());
        }
        if xa.elem_count() != n * h + 2 * n {
            candle_core::bail!("gates: stage A wrote {} values for {n} rows of {h}", xa.elem_count());
        }
        // GateArgs: h, rank, n, n_norms, eps, inv_h, per.
        let args = || {
            Words::default().u(h).u(rank).u(n).u(norms.len()).f(eps as f32).f((1.0 / h as f64) as f32).u(GATE_CHANNELS)
        };
        let at = |i: usize| i.min(n - 1);
        let mut v: Vec<Arg> = vec![Arg::Primary];
        for i in 0..3 {
            v.push(Arg::Tensor(msgs[at(i)], h));
        }
        for i in 0..3 {
            v.push(Arg::Tensor(gates[at(i)].down, rank * 2 * h));
        }
        v.push(Arg::Out);
        v.push(Arg::Bytes(args()));
        let down = Launch {
            name: "glue-gate-down",
            pipe: &self.gate_down,
            args: v,
            out_elems: n * rank,
            out_shape: Shape::from(n * rank),
            out_dtype: DType::F32,
            grid: Grid::Simd(n * rank),
        };
        let g = xa.apply_op1_no_bwd(&down)?;
        let mut v: Vec<Arg> = vec![Arg::Primary, Arg::Tensor(&g, n * rank)];
        for i in 0..3 {
            v.push(Arg::Tensor(msgs[at(i)], h));
        }
        for i in 0..3 {
            v.push(Arg::Tensor(gates[at(i)].forget, h * rank));
        }
        for i in 0..3 {
            v.push(Arg::Tensor(gates[at(i)].gain, h * rank));
        }
        v.push(Arg::Out);
        v.push(Arg::Bytes(args()));
        let apply = Launch {
            name: "glue-gate-apply",
            pipe: &self.gate_apply,
            args: v,
            out_elems: n * h,
            out_shape: Shape::from(n * h),
            out_dtype: DType::F32,
            grid: Grid::Simd(n * h.div_ceil(GATE_CHANNELS)),
        };
        let x = xa.apply_op1_no_bwd(&apply)?;
        let mut v: Vec<Arg> = vec![Arg::Primary];
        for i in 0..3 {
            v.push(Arg::Tensor(norms[i % norms.len()], h));
        }
        v.push(Arg::Out);
        v.push(Arg::Bytes(args()));
        let norm = Launch {
            name: "glue-gate-norm",
            pipe: &self.gate_norm,
            args: v,
            out_elems: n * h,
            out_shape: Shape::from((n, h)),
            out_dtype: DType::F32,
            grid: Grid::Groups(n),
        };
        x.apply_op1_no_bwd(&norm)
    }

    /// The post-attention half up to the router's scores; `ring` is the
    /// post-attention convolution state, advanced by one row.
    pub fn post_attn(&self, a: &PostAttnInputs, ring: &mut ConvRing) -> Result<PostAttn> {
        self.post_attn_with(a, ring, true)
    }

    /// [`Self::post_attn`] with the message gate and the router's scores in
    /// launches of their own (`split`: [`Self::gates_split`], and one
    /// simdgroup an expert for the scores), or inside the kernel's one
    /// threadgroup. The sums are the same, lane for lane, so the bits are;
    /// one threadgroup reads those weights at a fraction of the device's
    /// bandwidth. `false` is the reference of the split's gate.
    pub fn post_attn_with(&self, a: &PostAttnInputs, ring: &mut ConvRing, split: bool) -> Result<PostAttn> {
        let h = a.h.elem_count();
        check_h(h)?;
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if !split {
            candle_core::bail!("post_attn: the CUDA kernels carry the split gate and router only");
        }
        let k = a.conv_w.dims2()?.1;
        check_ring(ring, k, h)?;
        let n_states = a.states.len();
        if n_states > MAX_STATES {
            candle_core::bail!("post_attn: {n_states} block states, at most {MAX_STATES}");
        }
        let experts = a.router_w.dims2()?.0;
        if experts > MAX_EXPERTS {
            candle_core::bail!("post_attn: {experts} experts, at most {MAX_EXPERTS}");
        }
        let rank = match &a.gate {
            Some((_, g)) => gate_rank(g.down, h)?,
            None => 0,
        };
        // PostArgs: h, n_states, boundary, has_msg, rank, experts, k, pos,
        // eps, bar_eps, inv_h, bar_scale, conv_scale.
        let args = Words::default()
            .u(h)
            .u(n_states)
            .u(a.boundary as usize)
            .u(if a.gate.is_some() && split { 2 } else { a.gate.is_some() as usize })
            .u(rank)
            .u(if split { 0 } else { experts })
            .u(k)
            .u(ring.pos)
            .f(a.eps as f32)
            .f(BAR_EPS as f32)
            .f((1.0 / h as f64) as f32)
            .f((h as f64).powf(-0.5) as f32)
            .f(conv_scale(k));
        let mut v: Vec<Arg> =
            vec![Arg::Primary, Arg::Owned(&ring.buf, true), Arg::Tensor(a.conv_w, h * k), Arg::Tensor(a.h, h)];
        for i in 0..MAX_STATES {
            v.push(match a.states.get(i) {
                Some(s) => Arg::Tensor(s, h),
                None => Arg::Tensor(a.h, h),
            });
        }
        v.push(Arg::Tensor(a.mlp_query, h));
        match &a.gate {
            Some((m, g)) => {
                v.push(Arg::Tensor(m, h));
                v.push(Arg::Tensor(g.down, rank * 2 * h));
                v.push(Arg::Tensor(g.forget, h * rank));
                v.push(Arg::Tensor(g.gain, h * rank));
            }
            None => {
                for _ in 0..4 {
                    v.push(Arg::Tensor(a.h, h));
                }
            }
        }
        v.push(Arg::Tensor(a.norm, h));
        v.push(Arg::Tensor(a.router_w, experts * h));
        v.push(Arg::Out);
        v.push(Arg::Bytes(args));
        let gate_split = a.gate.is_some() && split;
        let n = match (split, gate_split) {
            (false, _) => 2 * h + experts,
            (true, false) => 2 * h,
            // stage A: the residual, the row before its gate, the two factors the gate starts with
            (true, true) => 2 * h + 2,
        };
        let op = Launch {
            name: "glue-post-attn",
            pipe: &self.post_attn,
            args: v,
            out_elems: n,
            out_shape: Shape::from(n),
            out_dtype: DType::F32,
            grid: Grid::Groups(1),
        };
        let out = a.attn_out.apply_op1_no_bwd(&op)?;
        ring.pos = (ring.pos + 1) % (k - 1);
        let xn = match (&a.gate, gate_split) {
            (Some((m, g)), true) => {
                let xa = out.narrow(0, h, h + 2)?;
                self.gates_split(&xa, 1, h, rank, &[*m], &[*g], &[a.norm], a.eps)?.reshape(h)?
            }
            _ => out.narrow(0, h, h)?,
        };
        let scores = if split {
            let op = Launch {
                name: "glue-router",
                pipe: &self.router,
                args: vec![
                    Arg::Primary,
                    Arg::Tensor(a.router_w, experts * h),
                    Arg::Out,
                    Arg::Bytes(Words::default().u(h).u(experts)),
                ],
                out_elems: experts,
                out_shape: Shape::from(experts),
                out_dtype: DType::F32,
                grid: Grid::Simd(experts),
            };
            xn.apply_op1_no_bwd(&op)?
        } else {
            out.narrow(0, 2 * h, experts)?
        };
        Ok(PostAttn { residual: out.narrow(0, 0, h)?.reshape((1, h))?, xn: xn.reshape((1, h))?, scores })
    }

    /// `score + bias`, the top `k` and their weights, as `layers::route` on one row.
    pub fn top8(&self, scores: &Tensor, bias: &Tensor, k: usize, scale: f64) -> Result<Top8> {
        let e = scores.elem_count();
        if e > MAX_EXPERTS || k == 0 || k >= e {
            candle_core::bail!("top8: {k} of {e} experts (at most {MAX_EXPERTS}, and k < experts)");
        }
        let args = Words::default().u(e).u(k).f(scale as f32);
        let n = e + 2 * k + 1;
        let op = Launch {
            name: "glue-top8",
            pipe: &self.top8,
            args: vec![Arg::Primary, Arg::Tensor(bias, e), Arg::Out, Arg::Bytes(args)],
            out_elems: n,
            out_shape: Shape::from(n),
            out_dtype: DType::F32,
            grid: Grid::Groups(1),
        };
        let out = scores.apply_op1_no_bwd(&op)?;
        Ok(Top8 {
            sel: out.narrow(0, 0, e)?,
            weights: out.narrow(0, e, k)?,
            margin: out.narrow(0, e + k, 1)?,
            ids: out.narrow(0, e + k + 1, k)?,
            raw: out,
        })
    }

    /// The routing of [`Self::top8`] as the fused MoE kernels read it, without
    /// a read back to the host: `[2k]` u32 on the device, the `k` expert ids
    /// ascending, then the bits of their weights.
    pub fn sel(&self, top: &Top8, k: usize) -> Result<Tensor> {
        let n = top.raw.elem_count();
        if k == 0 || n < 2 * k + 2 {
            candle_core::bail!("sel: a top8 output of {n} values for k = {k}");
        }
        let e = n - 2 * k - 1;
        let op = Launch {
            name: "glue-sel",
            pipe: &self.sel,
            args: vec![Arg::Primary, Arg::Out, Arg::Bytes(Words::default().u(e).u(k).f(0.0))],
            out_elems: 2 * k,
            out_shape: Shape::from(2 * k),
            out_dtype: DType::U32,
            grid: Grid::Threads(k),
        };
        top.raw.apply_op1_no_bwd(&op)
    }

    /// `residual + short_conv(moe)` on the post-MLP ring, `[1, H]`.
    pub fn post_mlp(&self, moe: &Tensor, residual: &Tensor, conv_w: &Tensor, ring: &mut ConvRing) -> Result<Tensor> {
        let h = moe.elem_count();
        let k = conv_w.dims2()?.1;
        check_ring(ring, k, h)?;
        let args = Words::default().u(h).u(k).u(ring.pos).f(conv_scale(k));
        let op = Launch {
            name: "glue-post-mlp",
            pipe: &self.post_mlp,
            args: vec![
                Arg::Primary,
                Arg::Owned(&ring.buf, true),
                Arg::Tensor(conv_w, h * k),
                Arg::Tensor(residual, h),
                Arg::Out,
                Arg::Bytes(args),
            ],
            out_elems: h,
            out_shape: Shape::from((1, h)),
            out_dtype: DType::F32,
            grid: Grid::Threads(h),
        };
        let out = moe.apply_op1_no_bwd(&op)?;
        ring.pos = (ring.pos + 1) % (k - 1);
        Ok(out)
    }
}

fn check_h(h: usize) -> Result<()> {
    if h == 0 || h > MAX_H {
        candle_core::bail!("hidden width {h}: the glue kernels take 1 to {MAX_H}");
    }
    Ok(())
}

fn check_ring(ring: &ConvRing, k: usize, h: usize) -> Result<()> {
    if ring.k != k || ring.c != h {
        candle_core::bail!("ring of {} taps over {} channels for a kernel of {k} over {h}", ring.k, ring.c);
    }
    Ok(())
}

fn gate_rank(down: &Tensor, h: usize) -> Result<usize> {
    let (rank, w) = down.dims2()?;
    if w != 2 * h || rank == 0 || rank > MAX_RANK {
        candle_core::bail!("message gate down {:?}: want [rank <= {MAX_RANK}, {}]", down.dims(), 2 * h);
    }
    Ok(rank)
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

enum Grid {
    /// Threadgroups of `TG` threads.
    Groups(usize),
    /// Threads, in groups of `ELT_GROUP`.
    Threads(usize),
    /// Threadgroups of one simdgroup (32 threads).
    Simd(usize),
}

/// One kernel launch as a candle op.
struct Launch<'a> {
    name: &'static str,
    pipe: &'a Pipe,
    args: Vec<Arg<'a>>,
    out_elems: usize,
    out_shape: Shape,
    out_dtype: DType,
    grid: Grid,
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
        // The output, uninitialised: every kernel writes every word of it.
        enum Out {
            F(CudaSlice<f32>),
            U(CudaSlice<u32>),
        }
        let alloc_err = |e| candle_core::Error::msg(format!("{}: alloc of the output: {e}", self.name));
        let out = match self.out_dtype {
            DType::F32 => Out::F(unsafe { stream.alloc::<f32>(self.out_elems) }.map_err(alloc_err)?),
            DType::U32 => Out::U(unsafe { stream.alloc::<u32>(self.out_elems) }.map_err(alloc_err)?),
            o => candle_core::bail!("{}: an output of {o:?}", self.name),
        };
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
                Arg::Owned(b, _) => ptrs.push(b.device_ptr(&stream).0),
                Arg::Out => ptrs.push(match &out {
                    Out::F(o) => o.device_ptr(&stream).0,
                    Out::U(o) => o.device_ptr(&stream).0,
                }),
                Arg::Bytes(w) => words.extend_from_slice(&w.0),
            }
        }
        let (grid, block) = match self.grid {
            Grid::Groups(n) => (n, TG),
            Grid::Threads(n) => (n.div_ceil(ELT_GROUP), ELT_GROUP),
            Grid::Simd(n) => (n, 32),
        };
        let cfg = LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b = stream.launch_builder(self.pipe);
        for p in &ptrs {
            b.arg(p);
        }
        for w in &words {
            b.arg(w);
        }
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("{}: {e}", self.name)))?;
        drop(bound);
        let storage = match out {
            Out::F(o) => CudaStorage::wrap_cuda_slice(o, dev),
            Out::U(o) => CudaStorage::wrap_cuda_slice(o, dev),
        };
        Ok((storage, self.out_shape.clone()))
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
        let out = dev.new_buffer(self.out_elems, self.out_dtype, self.name)?;
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
        match self.grid {
            Grid::Groups(n) => enc.dispatch_thread_groups(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: TG, height: 1, depth: 1 },
            ),
            Grid::Threads(n) => enc.dispatch_threads(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: ELT_GROUP, height: 1, depth: 1 },
            ),
            Grid::Simd(n) => enc.dispatch_thread_groups(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            ),
        }
        drop(enc);
        drop(bound);
        Ok((MetalStorage::new(out, dev, self.out_elems, self.out_dtype), self.out_shape.clone()))
    }
}
