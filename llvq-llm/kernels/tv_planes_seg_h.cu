// tv_planes_seg_h: tv_planes_h over a row-concatenation of projections that
// share one activation — the served path's twin of planes_seg.cu's
// tv_planes_seg.
//
// ## What it is, exactly
//
// The delta against `tv_planes_h` is **one indirection and nothing else**:
//
//     const float* gs = gscale + gs_off[row];
//
// hoisted above the tile loop, and `gs` passed to `planes_dot` where
// `tv_planes_h` passes `gscale`. Every other line is that kernel's, character
// for character — same grid (one warp per row, 8 rows per 256-thread block),
// same tiling, same two barriers, same shared staging, same f32 accumulation,
// same `tail_dot_h` epilogue, same narrowing store. The two kernels are meant
// to be diffable by eye, because the only correctness argument that survives a
// reader is "these are the same kernel plus the segment's centroid offset".
//
// ## Why the offset table exists
//
// Everything a segmented matrix carries concatenates by rows with no special
// case: `rscale` and `tail` are indexed by row already, `nblocks` and `tail_w`
// are equal across the parts by construction (they share `d_in`), and the block
// stream is numbered `row * nblocks + p` in both worlds.
//
// The gain centroids are the exception. They belong to a *matrix*, and
// `planes_dot` ends on `gscale[gain]` where `gain` is the **block's** bit — so
// they cannot be folded into the per-row `rscale`. `gs_off[row]` names where
// that row's pair starts in the concatenated `gscale`. One warp owns one row,
// so the read is warp-uniform and broadcasts rather than gathering.
//
// ## Why the store is not an accumulation
//
// A segmented matrix is a concatenation **by rows**, and rows partition the
// output: segment s owns `y[off_s .. off_s + d_out_s)` and no other segment
// ever addresses it. So `y[row]` is a plain store, as in `tv_planes_h`, and
// deliberately not an `atomicAdd`:
//
//  * an atomic is what two CTAs writing one output element need, which the row
//    partition forbids;
//  * it would cost the arithmetic its determinism, and determinism is exactly
//    what makes this kernel's correctness test lethal — fused and unfused run
//    the same blocks in the same order with the same centroids, nothing is
//    reassociated, so they must agree **bit for bit**. A wrong `gs_off` moves
//    some rows by ~2x and leaves others untouched, which no global tolerance
//    catches but a bit-for-bit comparison does;
//  * on `__half` it would round to binary16 at every partial sum, on an
//    accumulator this family keeps in f32 from end to end.
//
// For the same reason `y` is not zeroed and must not be: the grid is exact and
// every row is written. `fused_cuda.rs` allocates the output uninitialised on
// that ground, with a comment saying so. A kernel that quietly assumed a zeroed
// buffer would be correct in a bench that uses `alloc_zeros` and wrong in the
// model.
//
// ⚠️ If an exception region is ever added to a segmented layout — `Planes12x`
// and `Golay70` have one, which is precisely why they are refused a fused arm —
// the allocation must become a memset **in the same commit**, because those
// CTAs add into rows the row CTAs have already written.
//
// Like every kernel of this family there is deliberately no
// `if (row >= d_out) return;`: a return before `__syncthreads()` deadlocks, and
// it would break the full-warp mask of `warp_sum`. The host asserts
// `d_out % 8 == 0` instead — for a segmented matrix that is a statement about
// the total, and the host asserts it per part as well, so a group cannot pass
// by accident while leaving the unfused control unlaunchable.
//
// ⚠️ **COMPILED HERE, NOT VALIDATED HERE.** `tests/host_planes_seg_h.cpp`
// compiles this file as host C++ in the exact order NVRTC sees, which catches
// every syntax and type error before one costs a billed job — that is all it
// can do. A single-threaded driver reproduces neither `__syncthreads` nor a
// warp shuffle. What is proved on the development machine is the host half:
// the spliced stream and the `gs_off` table, in `tests/fused_segment.rs`. This
// kernel's own correctness is an open claim until a job compares its greedy
// tokens against the unfused arm.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guards
// below only resolve from disk under a host clang++ syntax check. Order is the
// caller's contract: llvq_slot.cuh, matvec.cu (TILE_COLS, warp_sum, f2h,
// tail_dot_h), llvq_planes.cuh (planes_dot), planes.cu, tv_planes_h.cu, then
// this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_PLANES_CUH
#include "../../llvq-cuda/kernels/llvq_planes.cuh"
#endif

extern "C" __global__ void tv_planes_seg_h(const u32* __restrict__ words,
                                           const ClassRec* __restrict__ tab,
                                           const float* __restrict__ gscale,
                                           const u32* __restrict__ gs_off,
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
    u32 b0r  = row * nblocks;
    // Warp-uniform: one warp owns one row, so this broadcasts rather than
    // gathering. Hoisted out of the tile loop for the same reason `b0r` is.
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
            acc += planes_dot(words, tab, gs, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[row] = f2h(acc * rscale[row] + tv);
    }
}
