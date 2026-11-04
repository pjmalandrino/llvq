// `x' = Q x` for widths whose staging does not fit one block's shared memory.
//
// `rot_apply` stages the whole vector in f32 shared memory: n * 4 bytes. At the
// Qwen3-32B `down_proj` width, 25,600 = 1,024 * 25, that is 102,400 B, and the
// cards with 99 KiB of opt-in (sm_86, sm_89, sm_120: L40S, RTX 5090, RTX PRO
// 6000) offer 101,376. The odd factor is not the problem: k = 25 fits
// LLVQ_ROT_KMAX. Shared memory is.
//
// `rot_apply_dev` runs the SAME helpers of `llvq_rot.cuh` (rot_load,
// rot_wht_step, rot_scale_out, rot_mix), in the SAME order, with the same
// barriers, on a device `scratch` instead of `extern __shared__`. That is what
// the Metal port already does (`llvq-llm/kernels/llvq_rot.metal`), and it is
// bit-identical to `rot_apply` by construction: every output coordinate is
// fixed by the widening, the sign flip, the butterfly stages in order, one
// multiply by the host's `inv` and an explicit `__fmaf_rn` chain from +0.
// There is no reduction across threads and nothing to contract, so the memory
// space the stages run in changes no bit. `__syncthreads()` orders global
// memory accesses among the threads of one block as it orders shared ones.
// `tests/rotation_matches_rust.rs` checks the equality byte for byte on the
// host at every width the 32B needs, and `bin/rotbench` on a card.
//
// ## Why a translation unit of its own
//
// `llvq_rot.cuh` and `rotate.cu` belong to the served unit, whose sha256 every
// served log prints. Appending this file there would move that hash for the
// 4B, 8B and 14B, which never launch these kernels. So the runtime compiles
// `llvq_rot.cuh` + `rotate_wide.cu` as a second module, and only when some
// rotation of the model is refused by `llvq_cuda::shared::rot_plan`
// (`llvq_cuda::wide::rot_kernel` says `Device`). `llvq_rot.cuh` is read, not
// edited: its comment about 25,600 being 512 * 50 is wrong (it is 1,024 * 25),
// and the correction lives in `docs/format-noyau.md`, because the header is
// part of the served unit.
//
// ## What the host owes, on top of `rot_apply`'s contract
//
//   * `scratch`: device memory distinct from `xout` and `xin`, at least
//     `n_rows * n + (LLVQ_ROT_KMAX - k) * m` floats (`wide::rot_scratch_floats`).
//     The tail keeps any speculated read of `rot_mix`'s unrolled column loop
//     (`t >= k`) inside the allocation; the Metal port clamps that index for
//     the same reason.
//   * zero bytes of dynamic shared memory;
//   * the same grid, block and arguments as `rot_apply` otherwise.
//
// The selection rule is the guarantee that no served object moves: a width
// keeps `rot_apply` whenever `rot_apply` can run it on this card.

#ifndef LLVQ_ROT_CUH
#include "llvq_rot.cuh"
#endif

extern "C" __global__ void rot_apply_dev(const unsigned short* __restrict__ xin,
                                         const u32* __restrict__ signbits,
                                         const float* __restrict__ small,
                                         float* __restrict__ xout,
                                         float* __restrict__ scratch,
                                         u32 n,
                                         u32 m,
                                         u32 k,
                                         float inv,
                                         u32 x_off)
{
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;

    rot_load(xin, signbits, scratch, n, x_off, tid, nthreads);
    __syncthreads();

    // `m` is uniform across the block, so every thread reaches every barrier.
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(scratch, n, len, tid, nthreads);
        __syncthreads();
    }

    if (k == 1u) {
        rot_scale_out(scratch, xout, n, inv, tid, nthreads);
    } else {
        rot_mix(scratch, small, xout, m, k, inv, tid, nthreads);
    }
}

// `rot_apply_rows` with its staging in device memory: `grid.x == n_rows`
// exactly, row `r` stages in `scratch + r * n`, reads its input at
// `x_off + r * row_stride` and writes `xout + r * n`. The rows are independent,
// so no block reads another's scratch.
extern "C" __global__ void rot_apply_rows_dev(const unsigned short* __restrict__ xin,
                                              const u32* __restrict__ signbits,
                                              const float* __restrict__ small,
                                              float* __restrict__ xout,
                                              float* __restrict__ scratch,
                                              u32 n,
                                              u32 m,
                                              u32 k,
                                              float inv,
                                              u32 x_off,
                                              u32 row_stride)
{
    const u32 r = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    // `unsigned long long` for the reason `rot_apply_rows` gives: NVRTC has no
    // `size_t`, and a 32-bit product that is right by coincidence is not a
    // definition.
    float* s = scratch + (unsigned long long)r * (unsigned long long)n;

    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    __syncthreads();

    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }

    float* __restrict__ out = xout + (unsigned long long)r * (unsigned long long)n;
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}
