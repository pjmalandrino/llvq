// tv_f1r_v3g: the served Tetra decode in the floor's geometry.
//
// `tv_f1r_v3` measures a decode whose gain bit is never read, whose points are
// never normalised and whose coordinates are dotted in trio order against an
// activation that is in natural order. It is a floor, and it says so. This
// kernel is the same tile loop with `tetra48_dot` in place of `f1r_dot_v3`, so
// the delta between the two arms is exactly the four things a served kernel
// has to do and the floor does not — and nothing else. Same grid, same two
// barriers, same shared staging, same tail epilogue, same warp reduction.
//
// Two pointers are added, both tiny and both read-only:
//
//   gscale[2]                 the matrix's gain centroids
//   invnorm[TETRA48_SHELLS]   1/sqrt(16 m), entry 0 = 0 for the origin
//
// 128 bytes and 8 bytes of constants against a 16 KiB table already in flight:
// if this arm is slower than `f1r_v3`, it is the six `__dp4a` and the two
// multiplies, not the memory.
//
// Like every arm of this bench there is deliberately no `if (row >= d_out)
// return;`: a return before `__syncthreads()` deadlocks and would break the
// full-warp mask of `warp_sum`. The host asserts `d_out % 8 == 0`.
//
// Order is the caller's contract: llvq_slot.cuh, matvec.cu, llvq_f1rank.cuh,
// llvq_f1rank_v3.cuh, llvq_tetra48.cuh, then this file.

#ifndef LLVQ_TETRA48_CUH
#include "llvq_tetra48.cuh"
#endif

extern "C" __global__ void tv_f1r_v3g(const u32* __restrict__ words,
                                      u32 row_stride_u32,
                                      const u32* __restrict__ rows,
                                      const unsigned char* __restrict__ prefixes,
                                      const unsigned short* __restrict__ branches,
                                      const unsigned char* __restrict__ suffixes,
                                      const float* __restrict__ gscale,
                                      const float* __restrict__ invnorm,
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
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot(lo, hi16, tab, xb, gscale, invnorm);
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

// The 24 served values of the first `n` blocks of one row, for the card-side
// control — `tv_f1r_dump`'s twin, and the same standing: the clang++ harness
// proves the arithmetic on the dev machine, this proves the card runs the same
// arithmetic on the same words.
extern "C" __global__ void tv_tetra48_dump(const u32* __restrict__ words,
                                           u32 row_stride_u32,
                                           const u32* __restrict__ rows,
                                           const unsigned char* __restrict__ prefixes,
                                           const unsigned short* __restrict__ branches,
                                           const unsigned char* __restrict__ suffixes,
                                           const float* __restrict__ gscale,
                                           const float* __restrict__ invnorm,
                                           float* __restrict__ out,
                                           u32 nblocks,
                                           u32 n)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // The dump walks the stream in row-major block order, so `i` past the end
    // of a row rolls into the next one — which is what exercises the row
    // stride, and why the host reproduces the same arithmetic.
    u32 row = i / nblocks;
    u32 j   = i % nblocks;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    u32 lo, hi16;
    f1r_load(words + row * row_stride_u32, j, lo, hi16);
    tetra48_decode_f(lo, hi16, tab, gscale, invnorm, out + i * LLVQ_DIM);
}
