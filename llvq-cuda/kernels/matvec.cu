// The fused kernel: dequantize and multiply in one pass, never materializing
// a weight. This is the contribution — everything else in this crate exists to
// make it measurable.
//
// A direct port of `tv_slot` / `tv_f16` from `llvq-metal/src/bin/thesis.rs`.
// Same tiling, same two barriers, same block-local pointer, same accumulator
// association. Deliberately the *tiled* form and not `bin/matvec`'s: staging a
// whole row needs 38 KB at d_in = 9728, which would cap the block at 2 per SM
// against 6 — and this kernel lives by slipping into memory-latency bubbles.

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

// 128 blocks = 3072 columns = 12 KB, injected by the host so the staging size
// and the kernel's tiling cannot drift apart. That drift is a real defect the
// Metal side carried for months (`TILE_BLOCKS` defined twice, unlinked).
#ifndef TILE_BLOCKS
#error "the host must define TILE_BLOCKS"
#endif
#define TILE_COLS (TILE_BLOCKS * LLVQ_DIM)

// f16 → f32 in one hardware instruction, without `cuda_fp16.h`.
//
// The runtime image ships no `-dev` package, so `__half2float` is not
// reachable; and a software conversion would be a rigged baseline — it would
// charge the FP16 arm for work the hardware does for free, and flatter the
// ratio this file exists to measure. `cvt.f32.f16` is exactly what
// `__half2float` compiles to.
__device__ __forceinline__ float h2f(unsigned short h)
{
#ifdef LLVQ_HOST_BUILD
    // The host syntax check has no PTX assembler. Correctness of this branch
    // does not matter — nothing executes it; only the surrounding code is
    // being type-checked.
    (void)h;
    return 0.0f;
#else
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
#endif
}

// Sum across the 32 lanes of a warp.
//
// Metal's `simd_sum` and this butterfly sum in different orders, so the two
// backends' outputs are *not* bit-comparable — each is checked against the
// shared f64 reference instead, and two worst errors are published rather
// than one delta.
__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
    return v;
}

// One warp per output row, 256 threads = 8 rows per block.
//
// There is deliberately no `if (row >= d_out) return;`. Two reasons, and both
// are silent failures rather than errors: a `return` before `__syncthreads()`
// deadlocks, and it would invalidate the full-warp mask `0xffffffff` under
// Independent Thread Scheduling. The host asserts `d_out % 8 == 0` instead —
// true of every shape in the 4B and the 8B.
//
// The tail epilogue reads `x` from **global** memory, not from `xs`: under
// tiling the tail columns sit past the last staged window.
extern "C" __global__ void tv_slot(const u32* __restrict__ words,
                                   const u32* __restrict__ bases,
                                   const ClassRec* __restrict__ tab,
                                   const float* __restrict__ gscale,
                                   const float* __restrict__ rscale,
                                   const float* __restrict__ tail,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   u32 nblocks,
                                   u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        // Two barriers, not one: the second orders the fill against the
        // readers, the first stops the next fill from racing a straggler
        // still reading the previous tile. `ntiles` depends only on `nblocks`,
        // so both are uniform across the block — which CUDA requires.
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gscale, b0r + j,
                            xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// `tv_slot` over a row-concatenation of several matrices sharing one input.
//
// ## Why this exists
//
// q/k/v consume the same activation, and so do gate/up. Launched separately,
// `k_proj` and `v_proj` are 1024 rows = 128 blocks against the 852 a L40S
// holds resident — 15 % occupancy — and the 2026-08-05 attribution job
// measured them at 157 GB/s against 469 for a well-filled shape, for a ratio
// against FP16 of 1.06×, i.e. nothing. Concatenating q+k+v by rows launches
// 768 blocks instead of three grids of 512/128/128, and the same geometry
// removes 108 launches per token.
//
// ## The one thing that is not shared: the gain centroids
//
// Every matrix carries its own two centroids, and `slot_dot` ends on
// `gscale[gain]`. They cannot be folded into `rscale` — `gscale` is selected
// by the *block's* gain bit, `rscale` is per row — so the segment has to be
// resolved somewhere. It is resolved here, once per row, by an offset table:
// `gs_off[row]` names where that row's pair starts in `gscale`. The read is
// uniform across the warp (one warp owns one row), so it broadcasts.
//
// Everything else concatenates without a special case: `nblocks` and `tail_w`
// are equal by construction (same `d_in`), `rscale` and `tail` are indexed by
// row already, and the block stream is transcoded from the concatenated
// indices, so its groups of 32 simply straddle the segment boundaries — which
// the addressing has always tolerated, since they already straddle rows.
//
// ⚠️ `tv_slot` is left untouched on purpose. It is the object every published
// millisecond refers to; this is a second kernel measured beside it, not a
// replacement, until a job says the fusion is worth adopting.
extern "C" __global__ void tv_slot_seg(const u32* __restrict__ words,
                                       const u32* __restrict__ bases,
                                       const ClassRec* __restrict__ tab,
                                       const float* __restrict__ gscale,
                                       const u32* __restrict__ gs_off,
                                       const float* __restrict__ rscale,
                                       const float* __restrict__ tail,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       u32 nblocks,
                                       u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    const float* gs = gscale + gs_off[row];
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gs, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// f32 → f16, round to nearest even. The inverse of `h2f`, same reasoning: no
// `cuda_fp16.h` in the runtime image, and `cvt.rn.f16.f32` is the instruction
// `__float2half` compiles to.
__device__ __forceinline__ unsigned short f2h(float f)
{
#ifdef LLVQ_HOST_BUILD
    (void)f;
    return 0;
#else
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#endif
}

// `tv_slot` writing f16 — the shape an inference runtime needs.
//
// The model is f16 end to end, so a f32 result costs one conversion kernel per
// projection: 252 launches a token, on a decode already dominated by launch
// latency. Writing the half directly removes them.
//
// A separate kernel rather than a flag on `tv_slot`, for the same reason
// `tv_slot_seg` is separate: `tv_slot` is the object every published
// millisecond refers to, and its register allocation must not move because an
// inference path wanted a different store.
//
// The accumulation is unchanged — f32 throughout, same association, same
// `warp_sum`. Only the final store narrows, exactly where candle's conversion
// kernel would have narrowed it.
extern "C" __global__ void tv_slot_h(const u32* __restrict__ words,
                                     const u32* __restrict__ bases,
                                     const ClassRec* __restrict__ tab,
                                     const float* __restrict__ gscale,
                                     const float* __restrict__ rscale,
                                     const float* __restrict__ tail,
                                     const float* __restrict__ x,
                                     unsigned short* __restrict__ y,
                                     u32 nblocks,
                                     u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gscale, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = f2h(acc * rscale[row] + tv);
    }
}

// The FP16 witness. Same shape, same tiling, 128-bit loads on both sides.
//
// Loading eight halves per lane rather than one is not a courtesy to the
// baseline: K−1(b) measured the load width to be worth ~5 % **on both arms**
// in Metal, so narrowing one side would raise the ratio for a reason that is
// not a property of the format.
//
// Note what this arm is and is not: `w` holds the LLVQ reconstruction rounded
// to binary16, in the rotated basis — not the model's own FP16 checkpoint.
// Both arms compute the same mathematical product to within rounding. It is a
// baseline of *cost*, not of quality.
extern "C" __global__ void tv_f16(const unsigned short* __restrict__ w,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  u32 d_in)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = 0.0f;

    u32 ntiles = (d_in + TILE_COLS - 1u) / TILE_COLS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 c0 = t * TILE_COLS;
        u32 n  = c0 + TILE_COLS < d_in ? TILE_COLS : d_in - c0;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[c0 + i];
        __syncthreads();

        // 8 halves = one 128-bit load per lane, 4 KB contiguous per warp.
        const uint4* w8 = (const uint4*)(w + row * d_in + c0);
        for (u32 j = lane; j < (n >> 3); j += 32u) {
            uint4 p = w8[j];
            u32 c = j << 3;
            const u32 q[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (u32 k = 0; k < 4; ++k) {
                acc = __fmaf_rn(h2f((unsigned short)(q[k] & 0xffffu)), xs[c + 2 * k], acc);
                acc = __fmaf_rn(h2f((unsigned short)(q[k] >> 16)), xs[c + 2 * k + 1], acc);
            }
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}
