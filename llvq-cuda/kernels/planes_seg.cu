// tv_planes_seg: the fused matvec over a row-concatenation of projections that
// share one input vector — the Planes14 twin of matvec.cu's tv_slot_seg.
//
// ## Why this is its own file
//
// `planes.cu` is not only the bench's: `llvq-llm/src/fused_cuda.rs`
// concatenates `llvq_planes.cuh` + `planes.cu` + `tv_planes_h.cu` and hands
// that string to NVRTC as the *shipped* inference kernel. Appending a kernel
// to `planes.cu` would change that string — its bytes, its sha256, and
// possibly `tv_planes_h`'s register allocation, which no correctness test can
// see. A separate file leaves the production translation unit untouched and
// lets the wiring lot pull this one in deliberately, the same way `planes12.cu`
// and `golay70.cu` arrived.
//
// ## Why the kernel exists
//
// q/k/v consume the same activation, and so do gate/up. Launched separately,
// `k_proj` and `v_proj` are 1024 rows = 128 blocks against the 852 an L40S
// holds resident — 15 % occupancy — and the 2026-08-05 attribution job measured
// them at 157 GB/s against 469 for a well-filled shape. Row-concatenating q+k+v
// launches 768 blocks instead of three grids of 512/128/128, and the same
// geometry removes 108 launches a token (252 → 144).
//
// Measured on **Slot32**, same card, same protocol
// (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`): 5.835 → 5.032 ms, a saving
// of **0.803 ms**, with the byte total identical to 0.00 %.
//
// ⚠️ That number does not transfer to Planes14 and must not be quoted for it.
// Planes14 already runs the same 252 matrices in 5.079 ms
// (`docs/mesures/c1-planesbench-2026-08-06.txt`), so "the same fraction" and
// "the same milliseconds" are two different predictions, and the delta is not
// one quantity anyway: the 108 removed launches are worth 108 × 3.63 µs ≈
// 0.39 ms whatever the layout (`docs/mesures/a3-graph-2026-08-06.txt`), while
// the occupancy half is a property of the memory pipeline on a stream that is
// 13 % lighter and already runs closer to its floor. Hence a bench arm, and a
// measurement — `bin/planesbench`, section "Fusion".
//
// ## Fusion costs no bytes here, and that is structural rather than measured
//
// Slot32 addresses through a per-group base plus a stride that is the widest
// record among the group's 32 blocks; re-transcoding a concatenated stream
// regroups blocks across the segment boundaries, so its byte total *can* move.
// The Slot32 job had to measure that confounder and print it (it came out at
// −0.00 %). Planes14 has no bases and a uniform 14-byte stride: the stream is
// 14 bytes per block whatever the grouping, so a fusion of matrices sharing
// `d_in` leaves the byte count **identically** unchanged. Proved host-side in
// `tests/planes_segment_matches_unfused.rs`, not measured — and the bench
// still prints both totals, because a claimed invariance that is never printed
// is a claim.
//
// ## The one thing that does not concatenate: the gain centroids
//
// Every matrix carries its own two, and `planes_dot` ends on `gscale[gain]`.
// They cannot be folded into `rscale` — `gscale` is selected by the *block's*
// gain bit, `rscale` is per row — so the segment has to be resolved somewhere.
// It is resolved here, once per row, by an offset table: `gs_off[row]` names
// where that row's pair starts in the concatenated `gscale`. One warp owns one
// row, so the read is warp-uniform and broadcasts.
//
// Everything else concatenates without a special case: `nblocks` and `tail_w`
// are equal by construction (same `d_in`), `rscale` and `tail` are indexed by
// row already, and the block stream is transcoded from the concatenated
// indices — block `row * nblocks + p` in both worlds.
//
// ## How the outputs are combined — the design point, decided here
//
// They are not combined at all. `y[row]` is written once, by a plain store,
// exactly as in `tv_planes`, because a segmented matrix is a concatenation **by
// rows** and rows partition the output: segment s owns
// `y[off_s .. off_s + d_out_s)` and no other segment ever addresses it. Two
// consequences, both deliberate:
//
//  * **no `atomicAdd`, on `float` or on `__half`.** An atomic is what you need
//    when two CTAs write the same output element, which the row partition
//    forbids. Adding one anyway would cost the arithmetic its determinism —
//    atomics commit in whatever order the machine picks — and determinism is
//    exactly what makes the fusion's correctness test lethal: fused and unfused
//    run the same blocks in the same order with the same centroids, nothing is
//    reassociated, so they must agree **bit for bit**, and a wrong `gs_off`
//    (which moves some rows by ~2× and leaves others untouched) cannot hide
//    behind a tolerance. An `atomicAdd` on `__half` would be worse still: it
//    would round to binary16 at every partial sum instead of once at the end,
//    on an accumulator this family keeps in f32 from end to end.
//  * **no zeroing of `y`.** `tv_planes12x` and `tv_golay70` do need it and pay
//    a memset inside their timed arm — but that is because their *exception*
//    CTAs add into rows the row CTAs have already written. That mechanism is an
//    overlay, not a segmentation, and importing its memset here would charge
//    this arm for a synchronisation it does not perform. It also matters
//    downstream: `llvq-llm/src/fused_cuda.rs` allocates the model's output
//    buffer **uninitialised**, on purpose and with a comment saying so, on the
//    grounds that the kernel writes every row and the grid is exact. A kernel
//    that quietly assumed a zeroed buffer would be correct in a bench that uses
//    `alloc_zeros` and wrong in the model — the worst possible split.
//
// The store stays f32, like `tv_planes` and `tv_slot_seg`. The f16-storing twin
// belongs beside `tv_planes_h` in the inference crate, for the reason that file
// gives: a bench kernel's register allocation must not move because a runtime
// wanted a different store.
//
// Like `tv_planes`, there is deliberately no `if (row >= d_out) return;`: a
// return before `__syncthreads()` deadlocks, and it would break the full-warp
// mask of `warp_sum`. The host asserts `d_out % 8 == 0` instead — which for a
// segmented matrix is a statement about the *total*, and the host also asserts
// it per segment so a group cannot pass by accident.
//
// ⚠️ **COMPILED HERE, NOT VALIDATED HERE.** `tests/host_planes.cpp` compiles
// this file as host C++ and catches every syntax and type error before one
// costs a billed job — that is all it can do. A single-threaded driver
// reproduces neither `__syncthreads` nor a warp shuffle, so the kernel itself
// is proved only on the card. What *is* proved on the development machine:
// `planes_dot`, unchanged and executed block by block by that same harness, and
// the host-side segmentation — row order, `gs_off`, the concatenated stream —
// in `tests/planes_segment_matches_unfused.rs`. The kernel's own correctness is
// an open claim until a job runs the bench's bit-exact comparison against
// `tv_planes`.
//
// ⚠️ `tv_planes` is left untouched on purpose. It is the object every published
// millisecond refers to; this is a second kernel measured beside it, not a
// replacement, until a job says the fusion is worth adopting.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guards
// below only resolve from disk under a host clang++ syntax check. Order is the
// caller's contract: llvq_slot.cuh, matvec.cu (TILE_COLS, warp_sum),
// llvq_planes.cuh (planes_dot), planes.cu, then this file.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_PLANES_CUH
#include "llvq_planes.cuh"
#endif

extern "C" __global__ void tv_planes_seg(const u32* __restrict__ words,
                                         const ClassRec* __restrict__ tab,
                                         const float* __restrict__ gscale,
                                         const u32* __restrict__ gs_off,
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
        // Two barriers, not one, for the reason matvec.cu gives: the second
        // orders the fill against the readers, the first stops the next fill
        // from racing a straggler still reading the previous tile. `ntiles`
        // depends only on `nblocks`, so both stay uniform across the block.
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += planes_dot(words, tab, gs, b0r + j, xs + (j - jlo) * LLVQ_DIM);
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
