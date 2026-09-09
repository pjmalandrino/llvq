// tv_tetra48_h: the served Tetra matvec, f16 in and f16 out.
//
// `tv_planes_h`'s shell with `tetra48_dot` in place of `planes_dot`, so the
// delta between the two served arms is the layout and nothing else: same grid
// (one warp per row, 8 rows per 256-thread block), same tiling, same two
// barriers, same shared staging, same `warp_sum`, same f16 tail epilogue.
//
// Three things differ from the bench arm `tv_f1r_v3g`, and all three are the
// difference between a floor and a served kernel:
//
//   * the tail arrives as `unsigned short` and goes through `tail_dot_h`,
//     because the served file stores it in f16 (`TAIL_BYTES = 2` since
//     2026-08-09) — the bench staged an f32 tail it had generated itself;
//   * the output is f16, written through `f2h`;
//   * the words are addressed **row-strided**, `words + row · row_stride_u32`,
//     where `tv_planes_h` addresses blocks flat at `row · nblocks + j`. That
//     is not a choice: a 14-byte record at `14·b` is 2-aligned and its read
//     window is a constant of the layout, while a 6-byte record is not, so a
//     Tetra row must start on a u32 boundary or every row after the first
//     reads at a shifted phase. `llvq_artifact::tetra48` builds the stride and
//     asserts the last block's window fits inside it.
//
// The activation `x` stays f32 in shared, exactly as `tv_planes_h` stages it:
// the tile is the same 24·4 bytes a block and the same `TILE_BLOCKS`, so a
// residency comparison between the two served arms is like for like. The tile
// is what the sweep of 2026-09-09 showed decides Tetra's speed — 12,288 B a CTA
// leaves ~28 KB of L1 against a decoder table that wants 18 KiB, and the
// optimum tile is 64 on sm_89 and 32 on sm_120 (`docs/mesures/tile-sweep-2026-09-09.txt`).
//
// Like every arm here there is deliberately no `if (row >= d_out) return;`: a
// return before `__syncthreads()` deadlocks and would break `warp_sum`'s
// full-warp mask. The host asserts `d_out % 8 == 0`.
//
// NVRTC has no filesystem, so the host concatenates the parts and the guards
// below only resolve from disk under a host clang++ syntax check. Order is the
// caller's contract: llvq_slot.cuh, matvec.cu, llvq_f1rank.cuh,
// llvq_f1rank_v3.cuh, llvq_tetra48.cuh, then this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_TETRA48_CUH
#include "../../llvq-cuda/kernels/llvq_tetra48.cuh"
#endif

extern "C" __global__ void tv_tetra48_h(const u32* __restrict__ words,
                                        u32 row_stride_u32,
                                        const u32* __restrict__ rows,
                                        const unsigned char* __restrict__ prefixes,
                                        const unsigned short* __restrict__ branches,
                                        const unsigned char* __restrict__ suffixes,
                                        const float* __restrict__ gscale,
                                        const float* __restrict__ invnorm,
                                        const float* __restrict__ rscale,
                                        const unsigned short* __restrict__ tail,
                                        const float* __restrict__ x,
                                        unsigned short* __restrict__ y,
                                        u32 nblocks,
                                        u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const u32* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot(lo, hi16, tab, xs + (j - jlo) * LLVQ_DIM, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[row] = f2h(acc * rscale[row] + tv);
    }
}
