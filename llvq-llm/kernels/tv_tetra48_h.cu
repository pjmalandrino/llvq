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

// How many activation rows one launch carries. Host-injected, like
// TILE_BLOCKS, and bounded by SHARED MEMORY rather than chosen: the staging is
// `TETRA48_ROWS · TILE_BLOCKS · LLVQ_DIM · 4` bytes, which at the served tile
// of 128 is 49,152 at four rows — exactly the per-block allowance every card
// here reports. Eight would need the tile halved, and the tile is the knob the
// two-card split of 2026-09-09 turned on.
#ifndef TETRA48_ROWS
#define TETRA48_ROWS 4u
#endif

// `tv_tetra48_h` over TETRA48_ROWS activation rows at once — the prefill path.
//
// ## What it is for
//
// The one-row kernel above decodes the whole weight stream once per row, so a
// prompt of N tokens reads it N times: on the served 4B a 5-shot MMLU question
// is several hundred tokens, i.e. 776 GB of re-reads and 200,000 launches, and
// `model::MAX_ROWS` refuses past 256 outright rather than look like a hang.
//
// Here the word is decoded ONCE and applied to R rows, so the stream is read R
// times less. Every quantized-inference stack carries this second kernel and
// for the same reason: prefill and decode are two regimes, one compute-bound
// and one bandwidth-bound.
//
// ## What it does not change
//
// **The answer, bit for bit.** `tetra48_dot_rows` keeps the 24 FMAs of a row
// in their order and keeps `(acc · g) · inv` unhoisted;
// `tests/tetra48_matches_rust.rs` proves both routes agree on the same fixture
// and six mutants are killed there, including the hoisted scales.
//
// And the ONE-row kernel above is untouched. Decode-time numbers stay attached
// to the kernel that produced them; this entry point is additional, never a
// replacement.
//
// ## The two things the host owes it
//
//   * `shared = TETRA48_ROWS · TILE_BLOCKS · LLVQ_DIM · 4`, checked against
//     the card's per-block allowance — this kernel is loaded through `func`,
//     with no opt-in, so the DEFAULT allowance is the bound;
//   * `n_rows <= TETRA48_ROWS`. Rows past it fold onto row 0 for the staging,
//     which keeps the reads in bounds, and are not stored.
//
// Like every arm of the family there is deliberately no early `return`: one
// before `__syncthreads()` deadlocks and would break `warp_sum`'s full-warp
// mask. The host asserts `d_out % 8 == 0`.
extern "C" __global__ void tv_tetra48_rows_h(const u32* __restrict__ words,
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
                                             u32 tail_w,
                                             u32 n_rows,
                                             u32 d_in,
                                             u32 d_out)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const u32* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    // One accumulator a row, in registers. This is what bounds TETRA48_ROWS
    // from the other side: the one-row kernel reports 40 registers against a
    // contract of 64.
    float acc[TETRA48_ROWS];
#pragma unroll
    for (u32 r = 0; r < TETRA48_ROWS; ++r) acc[r] = 0.0f;

    // Rows are separated in shared by a whole tile, so `tetra48_dot_rows`
    // steps by this and never by LLVQ_DIM.
    const u32 xs_stride = TILE_BLOCKS * LLVQ_DIM;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < TETRA48_ROWS; ++r) {
            // A row past `n_rows` reads row 0 again rather than off the end.
            // Its output is computed and discarded, which costs one row of
            // arithmetic on the last launch of a prompt and nothing else.
            const float* __restrict__ xr = x + (u64)(r < n_rows ? r : 0u) * d_in;
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                xs[r * xs_stride + i] = xr[jlo * LLVQ_DIM + i];
            }
        }
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            tetra48_dot_rows<TETRA48_ROWS>(lo, hi16, tab,
                                           xs + (j - jlo) * LLVQ_DIM, xs_stride,
                                           gscale, invnorm, acc);
        }
    }

    // Reduced row by row. The loop is entered by every lane — `warp_sum` is a
    // shuffle and a lane that skipped it would hang the others.
#pragma unroll
    for (u32 r = 0; r < TETRA48_ROWS; ++r) {
        float a = warp_sum(acc[r]);
        if (lane == 0 && r < n_rows) {
            const float* __restrict__ xr = x + (u64)r * d_in;
            float tv = tail_dot_h(tail, xr + nblocks * LLVQ_DIM, row, tail_w);
            y[(u64)r * d_out + row] = f2h(a * rscale[row] + tv);
        }
    }
}

