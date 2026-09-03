// tv_nullk: the FLOOR, the same pass without one byte of weights.
//
// Arm `nullk` of `proofs/preregistration-p4-2026-08-14.md` §2.5: *same grid,
// same k, output written, no weight read*.
//
// ## What it measures, and why it is the only arm that measures it
//
// The CUDA attribution splits the 2.04 ms/token into **latency and occupancy
// 39%, stream 33%, decoding 19%**. The four attempts to get under `Planes14`
// (E3, `Golay70` v2, `e1c14`, E1v) all attacked the 33% by inflating the 19%.
// **Nobody has ever attacked the 39%**, and nobody has ever measured it
// directly: it is a residue, obtained by subtraction in an attribution of
// 2026-08-05.
//
// This kernel is that residue, made observable. It keeps EVERYTHING an LLVQ
// arm does around the decoding:
//
//   * the same grid, one warp per row, 8 rows per block of 256;
//   * the same tiling and the same two `__syncthreads`;
//   * the same staging of the activation in shared memory;
//   * the same `warp_sum` reduction and the same tail epilogue;
//   * the same write of `y`.
//
// and it removes exactly one thing: **reading and decoding the block**.
//
//     t(planes14) − t(nullk)  =  weight traffic + decoding
//     t(nullk)                =  everything else, the 39%
//
// ## The trap, and what defuses it
//
// 🚨 A kernel that reads no weight is a kernel the compiler can empty out. If
// it emptied the tile loop it would remove the staging, which is exactly half
// of what this arm exists to measure, and it would return a flattering floor
// that nothing in the output would report.
//
// So it accumulates the STAGED ACTIVATION itself: `xs[j]` is read from shared
// memory, the result goes into `y`, and the chain global → shared → register
// → global is complete. The staging is load-bearing by construction, not by
// hope.
//
// ⚠️ What this arm can NOT be: checked against the f64 reference. It does not
// compute the model's product and has no standard to be held to. Like `sol`
// in `bin/rankbench` it is an anchor, and what is asked of it is to be
// OBSERVABLE, not correct. The bench treats it that way explicitly.
//
// ## What it does not measure
//
// Neither launch latency alone (108 launches fewer are worth 0.392 ms,
// measured elsewhere) nor occupancy alone. It returns their sum plus the
// staging plus the reduction: a ceiling on what a perfectly free decoding
// would leave, and exactly the quantity the dossier is missing.
//
// Same assembly contract as `planes.cu`: NVRTC has no file system, the host
// concatenates, and the guards below only resolve from disk under a host
// check (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif

extern "C" __global__ void tv_nullk(const float* __restrict__ rscale,
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

        // The loop of `tv_planes`, minus the decoding. Each lane touches the
        // 24 slots of its blocks, same iteration count, same access to shared
        // memory, and opens no weight word.
        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(1.0f, xb[s], acc);
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
