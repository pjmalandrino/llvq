// tv_tetra48_seg: the fused matvec over a row-concatenation of Tetra
// projections that share one input vector — the Tetra twin of
// `planes_seg.cu`'s `tv_planes_seg`.
//
// ## Why this is its own file
//
// The same reason `planes_seg.cu` is: `tetra48_v3g.cu` is concatenated into
// the *shipped* inference translation unit, and appending a kernel to it would
// change that string's bytes, its sha256, and possibly `tv_f1r_v3g`'s register
// allocation — which reported 40 registers and 0 local bytes, and which no
// correctness test can see move. A separate file leaves the production unit
// untouched.
//
// ## What this fuses on the served object, and what it cannot
//
// `tv_planes_seg`'s header prices the Planes14 fusion at 252 launches to 144,
// because q+k+v and gate+up each collapse to one grid. **That grouping does
// not survive on the served Tetra object.** `configs/qwen3-4b-tetra-q5.json`
// serves `v_proj` as int4 g128, so a q+k+v group would span two code kinds and
// two kernels. What is left is q+k and gate+up:
//
//   per layer   q k o gate up down   = 6 Tetra matrices (v_proj is int4)
//   fused       [q k] o [gate up] down = 4 grids
//   whole model 216 -> 144 Tetra launches, 72 removed, the 36 int4 unchanged
//
// So 72 launches a token, not 108. At the 5.3 us a launch the same bench reads
// off the Planes14 fusion (0.569 ms for 108), that is about **0.38 ms** — on an
// arm that measures 3.423 ms at the served tile of 2026-09-20, so about 11 %.
// That is an arithmetic projection from another layout's measurement, not a
// measurement, and the bench exists to replace it.
//
// ## Why the stream concatenates without transcoding, unlike Slot32
//
// Slot32 addresses through a per-group base and a stride that is the widest
// record among a group's 32 blocks, so regrouping across a segment boundary can
// move its byte total — the 2026-08-05 job had to measure that confounder.
// Planes14 has no bases and a uniform 14-byte stride.
//
// Tetra is uniform in a stronger way: the stream is **row-strided**, 48 bits a
// block, `words + row * row_stride_u32`. Segments that share `d_in` share
// `nblocks` and therefore share `row_stride_u32`, so a row-concatenation is a
// concatenation of the word arrays and nothing is re-encoded. The byte total is
// identically unchanged, by construction rather than by measurement — and the
// bench still prints both totals, because an invariance that is never printed
// is a claim.
//
// ## The one thing that does not concatenate: the gain centroids
//
// Identical to `tv_planes_seg`, and for the same reason. Every matrix carries
// its own two, `tetra48_dot` ends on `gscale[gain]`, and they cannot be folded
// into `rscale` because `gscale` is selected by the block's gain bit while
// `rscale` is per row. It is resolved once per row by `gs_off[row]`. One warp
// owns one row, so that read is warp-uniform and broadcasts.
//
// `invnorm` is NOT per matrix: it is `1/sqrt(16 m)` over the shells, a property
// of the lattice and not of the projection, so it is shared by every segment
// and passed once.
//
// ## How the outputs are combined
//
// They are not. `y[row]` is a plain store, exactly as in `tv_f1r_v3g`, because
// a segmented matrix is a concatenation **by rows** and rows partition the
// output: segment s owns `y[off_s .. off_s + d_out_s)` and no other segment
// addresses it. No `atomicAdd` — an atomic is for two CTAs writing one element,
// which the row partition forbids, and it would cost the arithmetic its
// determinism. Determinism is what makes the fusion's correctness test lethal:
// fused and unfused run the same blocks in the same order with the same
// centroids, nothing is reassociated, so they must agree **bit for bit**, and a
// wrong `gs_off` — which moves some rows by about 2x and leaves others
// untouched — cannot hide behind a tolerance.
//
// No zeroing of `y` either, and the same downstream reason `planes_seg.cu`
// gives: `llvq-llm/src/fused_cuda.rs` allocates the output buffer
// uninitialised on purpose, on the grounds that the kernel writes every row.
// A kernel that quietly assumed a zeroed buffer would be correct in a bench
// that uses `alloc_zeros` and wrong in the model.
//
// Like every arm of this bench there is deliberately no `if (row >= d_out)
// return;`: a return before `__syncthreads()` deadlocks and would break the
// full-warp mask of `warp_sum`. The host asserts `d_out % 8 == 0` — which for a
// segmented matrix is a statement about the total, and the host asserts it per
// segment too so a group cannot pass by accident.
//
// The store stays f32, like `tv_f1r_v3g` and `tv_planes_seg`. The f16-storing
// twin belongs beside `tv_tetra48_h` in the inference crate.
//
// WARNING: COMPILED HERE, NOT VALIDATED HERE. A host clang++ syntax check
// catches every syntax and type error before one costs a billed job; that is
// all it can do. A single-threaded driver reproduces neither `__syncthreads`
// nor a warp shuffle, so the kernel itself is proved only on the card, by the
// bench's bit-exact comparison against `tv_f1r_v3g`. Until a job runs that
// comparison, this kernel's correctness is an open claim.
//
// WARNING: `tv_f1r_v3g` is left untouched on purpose. It is the object every
// published Tetra millisecond refers to; this is a second kernel measured
// beside it, not a replacement, until a job says the fusion is worth adopting.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guard
// below only resolves from disk under a host clang++ syntax check. Order is the
// caller's contract: llvq_slot.cuh, matvec.cu (TILE_BLOCKS, warp_sum),
// llvq_f1rank.cuh, llvq_f1rank_v3.cuh, llvq_tetra48.cuh, then this file.

#ifndef LLVQ_TETRA48_CUH
#include "llvq_tetra48.cuh"
#endif

extern "C" __global__ void tv_tetra48_seg(const u32* __restrict__ words,
                                          u32 row_stride_u32,
                                          const u32* __restrict__ rows,
                                          const unsigned char* __restrict__ prefixes,
                                          const unsigned short* __restrict__ branches,
                                          const unsigned char* __restrict__ suffixes,
                                          const float* __restrict__ gscale,
                                          const u32* __restrict__ gs_off,
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
    // Warp-uniform: one warp owns one row, so this broadcasts rather than
    // gathering. Hoisted out of the tile loop for the same reason `wrow` is.
    const float* gs = gscale + gs_off[row];
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        // Two barriers, not one, for the reason matvec.cu gives: the second
        // orders the fill against the readers, the first stops the next fill
        // from racing a straggler still reading the previous tile. `ntiles`
        // depends only on `nblocks`, so both stay uniform across the block.
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot(lo, hi16, tab, xb, gs, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        // A plain store, not an accumulation: see the header. Rows partition
        // the output, so this address belongs to exactly one warp of one CTA.
        y[row] = acc * rscale[row] + tv;
    }
}
