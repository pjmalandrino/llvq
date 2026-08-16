// tv_e1v: the fused matvec over a row-aligned E1v stream.
//
// Same grid as `tv_planes` (one warp per row, 8 rows per 256-thread block),
// same tiling, same two barriers, same shared staging of the activation and the
// same tail epilogue — so a millisecond delta against that arm can only come
// from the layout.
//
// 🚨 **ONE thing differs, and it is not a detail: the inner loop.** `tv_planes`
// writes
//
//     for (u32 j = jlo + lane; j < jhi; j += 32u)
//
// which lets a lane past the end of the tile leave the loop. E1v cannot: its
// payload offsets come from an **exclusive prefix sum over the warp**, and
// `__shfl_up_sync` at full mask is undefined if a lane is missing. A row's last
// group is partial for every shape of the published 4B, so such lanes exist by
// construction. The loop therefore steps a **uniform** `j`, every lane computes
// its own `jj = j + lane`, and a lane with no block carries `have = false`:
// it still enters `e1v_dot`, still contributes a width of zero to the scan, and
// returns 0.0f without reading a record.
//
// Like `tv_planes`, there is deliberately no `if (row >= d_out) return;`: a
// return before `__syncthreads()` deadlocks and would break the full-warp mask
// of `warp_sum`. The host asserts `d_out % 8 == 0` instead.
//
// `nblocks` is **blocks per row**, which is exactly the `row_blocks` the stream
// was cut on (`llvq_artifact::e1v::transcode_e1v_rows`). The two must be the
// same number or every group boundary moves; `e1v_host.rs` passes one value to
// both and the bench's V0 would fail loudly if they ever diverged.
//
// This file cannot `#include` its dependencies under NVRTC — the loader has no
// filesystem, so the host concatenates the sources. The guards below key on
// macros the earlier parts define, and only resolve from disk under a host
// clang++ syntax check (`bin/cuhcheck`). Order is the caller's contract:
// llvq_slot.cuh, matvec.cu, llvq_e1v.cuh, then this file.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_E1V_CUH
#include "llvq_e1v.cuh"
#endif

extern "C" __global__ void tv_e1v(const u32* __restrict__ data,
                                  const u32* __restrict__ bases,
                                  const E1vRec* __restrict__ tab,
                                  const u32* __restrict__ pay,
                                  const u32* __restrict__ binom,
                                  const u32* __restrict__ golay,
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
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        // `j` is UNIFORM across the warp: every lane runs every iteration, which
        // is what keeps the scan inside `e1v_dot` a full-warp operation.
        for (u32 j = jlo; j < jhi; j += 32u) {
            u32  jj   = j + lane;
            bool have = jj < jhi;
            // The staged slice is only read when `have`; the index is clamped so
            // the pointer itself stays inside `xs` either way.
            const float* xb = xs + (have ? (jj - jlo) : 0u) * LLVQ_DIM;
            acc += e1v_dot(data, bases, tab, pay, binom, golay, gscale,
                           row, jj, nblocks, have, xb);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}
