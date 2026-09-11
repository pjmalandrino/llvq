// `x' = Q x` — the activation rotated into the basis the sealed weights live
// in. One block, one activation, everything in shared memory.
//
// ## Why a single block
//
// The Walsh–Hadamard transform is `log₂ m` stages separated by barriers, and
// CUDA has no barrier across blocks. Splitting the vector would mean either a
// second kernel launch per rotation — 288 launches a token instead of 144, on
// a decode already dominated by launch latency — or a grid-wide sync, which
// costs cooperative-launch occupancy for a vector that fits in one block's
// shared memory anyway. At `d_in = 9728`, the widest Qwen3-4B projection, the
// staging is 38 KB against the 48 KB a block gets by default.
//
// The cost of that choice is honest and bounded: one SM out of 142 does the
// work. It is the right trade only because the vector is small — this kernel
// moves 39 KB where the matvec it feeds moves 2.5 GB. Should a wider model
// need it (Qwen3-32B's `down_proj` is 25600 wide, 100 KB, and does not fit),
// the two-kernel split is the fallback, not a bigger block.
//
// ## What the host owes this kernel
//
//   * `n = k · m`, `m` a power of two, `k ≤ LLVQ_ROT_KMAX`
//   * `small` zero-padded to `LLVQ_ROT_KMAX × LLVQ_ROT_KMAX`, row-major
//   * `inv = 1/√m`, computed in f64 and narrowed — *not* recomputed here.
//     `rsqrtf` is an approximation and `sqrtf` is one rounding away from the
//     reference; taking the scalar as an argument removes a whole class of
//     "the last bit differs and nobody knows which side is right".
//   * `n · sizeof(float)` bytes of dynamic shared memory
//
// There is deliberately no bounds guard on `n`: an out-of-range `n` would
// overrun shared memory, so it is a host-side assertion, checked once per
// matrix at load time rather than once per token by every thread.

#ifndef LLVQ_ROT_CUH
#include "llvq_rot.cuh"
#endif

extern "C" __global__ void rot_apply(const unsigned short* __restrict__ xin,
                                     const u32* __restrict__ signbits,
                                     const float* __restrict__ small,
                                     float* __restrict__ xout,
                                     u32 n,
                                     u32 m,
                                     u32 k,
                                     float inv,
                                     u32 x_off)
{
    extern __shared__ float s[];
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;

    rot_load(xin, signbits, s, n, x_off, tid, nthreads);
    __syncthreads();

    // `m` is uniform across the block, so every thread runs the same number of
    // stages and reaches every barrier — the condition CUDA actually requires.
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }

    if (k == 1u) {
        rot_scale_out(s, xout, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, xout, m, k, inv, tid, nthreads);
    }
}

// `X' = Q X` for `n_rows` activations, ONE launch.
//
// ## Why this is not the split the header refuses
//
// The comment above rejects splitting ONE vector across blocks, because the
// Walsh-Hadamard transform has `log2 m` stages separated by barriers and CUDA
// has no barrier across blocks. That reason does not apply here: the rows are
// **independent**. One block does one whole row, barriers and all, exactly the
// work `rot_apply` does; the grid carries the rows. Nothing is shared between
// blocks, so nothing needs a barrier between them, and the shared-memory
// footprint per block is unchanged — the bound `max_d_in` checks at load time
// still holds at any `n_rows`.
//
// ## What it buys, and it is two things
//
// A prefill chunk of `PREFILL_ROWS` rows used to call `rot_apply` once a row:
// 4 launches a rotation site, 576 a chunk on the served 4B at
// `LLVQ_ROT_SHARE=1`. It is now 144.
//
// And the output lands **contiguous**, `[n_rows, n]` row-major, which is
// exactly the shape `tv_*_rows_h` wants. The host used to build that shape
// with `Tensor::cat` — one device copy a row, 864 a chunk, the largest single
// term in the prefill's operation count. Those copies are gone, not moved.
//
// ## What the host owes this kernel, on top of what `rot_apply` asks
//
//   * `grid.x == n_rows` exactly. There is no bounds guard, for the reason the
//     header gives about `n`: an out-of-range block would overrun `xout`, so
//     it is a host-side assertion made once, not a branch every thread pays.
//   * `xout` at least `n_rows * n` floats.
//   * `row_stride` in ELEMENTS, the distance from one input row to the next.
//     It is not `n`: the activation handed over is a view into a larger
//     buffer and its rows are `d_in` apart there, not `n` apart. The two are
//     equal today and writing `n` would be a coincidence, not a definition.
extern "C" __global__ void rot_apply_rows(const unsigned short* __restrict__ xin,
                                          const u32* __restrict__ signbits,
                                          const float* __restrict__ small,
                                          float* __restrict__ xout,
                                          u32 n,
                                          u32 m,
                                          u32 k,
                                          float inv,
                                          u32 x_off,
                                          u32 row_stride)
{
    extern __shared__ float s[];
    const u32 r = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;

    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    __syncthreads();

    // Same stage loop, same uniformity argument: `m` is uniform across the
    // block, so every thread reaches every barrier. `r` is uniform too — it is
    // `blockIdx` — so no barrier here is conditional on anything divergent.
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }

    // `unsigned long long`, not `size_t`: NVRTC has no headers, so `size_t` is
    // not a name here. And not `u32` either — `r * n` fits in 32 bits at every
    // width this model family has, and a cast that is right by coincidence is
    // the kind that stops being right in silence.
    float* __restrict__ out = xout + (unsigned long long)r * (unsigned long long)n;
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}
