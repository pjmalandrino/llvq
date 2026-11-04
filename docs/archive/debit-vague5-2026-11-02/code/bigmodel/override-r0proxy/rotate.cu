// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §5, step V0d). Not part of the
// repository, and NEVER a served text: it exists only to be written by a job
// into a directory that LLVQ_KERNEL_DIR names for `rotbench`, beside an
// UNCHANGED copy of llvq-cuda/kernels/llvq_rot.cuh (rotbench loads exactly
// these two files, rotbench.rs:86; the override is all or nothing,
// llvq-cuda/src/lib.rs:315-335, and the binary prints "SOURCES OVERRIDDEN"
// and the sha256 of the exact NVRTC string).
//
// What it is: rot_apply with its staging moved from `extern __shared__` into
// `xout` itself, in device memory. A proxy of R0 that needs no image rebuild,
// so R0's slowdown against K0 can be read at 9,728 and 12,288 before R0 is
// written. Same helpers, same order, so the output is rot_apply's bits:
// rotbench's printed error columns must be IDENTICAL to the stock run's,
// which is this proxy's own identity check.
//
// Why in place is legal. rot_load and rot_wht_step already take a plain
// `float* s` (llvq_rot.cuh:114-120, 159). In the mix, each thread owns its
// columns j: it reads every s[t*m + j], t < k, into registers (`col`) BEFORE
// it writes any xout[g*m + j], and no other thread touches column j. k == 1 is
// elementwise. __syncthreads() orders global accesses within the block as it
// orders shared ones. What is NOT legal is the served helpers' `__restrict__`
// on xout (llvq_rot.cuh:178-180, 212-215) once s aliases xout: the compiler
// could then move a store above a load. So the two output helpers are copied
// here without it, under other names, and xout loses its qualifier too.
//
// What it does NOT measure fairly. `launch_rot` still requests n * 4 B of
// dynamic shared memory (gpu.rs:813), unused here, and rotbench poses the
// 69,632 B opt-in on the function for its widest case (rotbench.rs:127-128).
// The driver sizes the L1/shared carveout from the shared memory the launch
// and the function ask for, so L1 gets what is left beside it: on sm_89, with
// a 128 KB unified block, probably about 64 KB at 9,728 (38,912 B requested),
// enough for the 38,912 B working set, and about 28 KB at 17,408, not enough
// (estimated; the carveout steps are not read here). The proxy therefore
// OVERSTATES R0's cost at 17,408 and reads fairly only up to about 12,288.
// Real R0 requests 0 B.

#ifndef LLVQ_ROT_CUH
#include "llvq_rot.cuh"
#endif

__device__ __forceinline__ void rot_scale_out_np(const float* s, float* xout, u32 n, float inv,
                                                 u32 tid, u32 nthreads)
{
    for (u32 i = tid; i < n; i += nthreads) xout[i] = s[i] * inv;
}

__device__ __forceinline__ void rot_mix_np(const float* s,
                                           const float* __restrict__ small,
                                           float* xout,
                                           u32 m,
                                           u32 k,
                                           float inv,
                                           u32 tid,
                                           u32 nthreads)
{
    for (u32 j = tid; j < m; j += nthreads) {
        float col[LLVQ_ROT_KMAX];
#pragma unroll
        for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t) col[t] = t < k ? s[t * m + j] * inv : 0.0f;
        for (u32 g = 0; g < k; ++g) {
            float acc = 0.0f;
#pragma unroll
            for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t)
                acc = __fmaf_rn(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
            xout[g * m + j] = acc;
        }
    }
}

extern "C" __global__ void rot_apply(const unsigned short* __restrict__ xin,
                                     const u32* __restrict__ signbits,
                                     const float* __restrict__ small,
                                     float* xout,
                                     u32 n,
                                     u32 m,
                                     u32 k,
                                     float inv,
                                     u32 x_off)
{
    float* s = xout;   // the staging, in device memory
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;

    rot_load(xin, signbits, s, n, x_off, tid, nthreads);
    __syncthreads();
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }
    if (k == 1u) {
        rot_scale_out_np(s, xout, n, inv, tid, nthreads);
    } else {
        rot_mix_np(s, small, xout, m, k, inv, tid, nthreads);
    }
}
