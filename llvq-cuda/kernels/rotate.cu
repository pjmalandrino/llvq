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
