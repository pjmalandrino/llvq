// planes_occ.cu — A3: the occupancy variants of the fused Planes14 matvec.
//
// Prereg: proofs/preregistration-a2-a3-geometrie-2026-08-31.md §5 (sha256
// 802006c5…, stamped 2026-09-01, before any A2/A3 job). Design note:
// docs/design-a3-occupation-2026-09-01.md. Deviations from the note are
// declared in proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md,
// never by editing the prereg.
//
// ## What is being attacked, in one number
//
// A1 measured the floor of the served geometry — 144 launches, a pass that
// reads no weight at all — at 1.794 ms on the L40S
// (docs/mesures/a1-nullk-252-144-2026-08-31.txt). About 0.54 ms of it is
// per-launch and was A2's pool (CUDA Graphs, measured and adopted at the 4B);
// the ~1.25 ms that remain are staging, barriers, reduction and under-filled
// grids — o_proj and down_proj launch 320 CTAs against the 852 an L40S holds
// resident, and they carry 35 % of the bytes. That residue is the pool this
// file exists to measure, arm by arm, **each arm moving ONE thing**:
//
//   tv_planes_pad     the staged activation at a 28-float stride instead of
//                     24. A lane reads its block's 24 floats from shared
//                     memory; at a 96-byte stride a quarter-warp's 128-bit
//                     loads start on banks 0, 24, 16, 8, 0, … and lanes 0 and
//                     4 collide. 112 bytes starts them on 0, 28, 24, …, 4 —
//                     all 32 banks, no collision. Same grid, same order,
//                     same bits. Not in the design note: found while writing
//                     the multi-row arm, declared in the ÉCARTS file.
//   tv_planes_mr2/4   R rows per warp: one staged tile serves 8R rows, every
//                     `xb` read from shared memory feeds R FMA chains, and
//                     the grid shrinks by R — which is exactly the trade the
//                     note leaves unsigned on paper. Same bits per row.
//   tv_planes_mr2p    mr2 over the padded stride — do the two effects add?
//   tv_planes_pers    a persistent, card-sized grid (resident CTAs × SM)
//                     whose CTAs loop over row groups; a single-tile
//                     activation is staged once per CTA rather than once per
//                     group. Same bits. Design note bras (b1).
//   tv_planes_sk      split-K across CTAs: the grid is the set of (row group,
//                     K slice) pairs, so o_proj and down_proj fill the card.
//                     The partial sums of a group meet in a fixed-order
//                     fixup performed by the LAST CTA to finish — one atomic
//                     ticket per CTA, no atomicAdd on any float. Deterministic
//                     but re-associated, so it is held to the f64 reference at
//                     the bench's 1e-5 rather than bit for bit. This is the
//                     note's bras (c) with the split moved from intra-CTA to
//                     inter-CTA: at g rows per CTA and all of K per CTA, the
//                     L2→shared staging of the activation multiplies by 8/g
//                     — 14.5 GB a token at g = 1 (*calculé*, ÉCARTS É2) —
//                     whereas splitting K across CTAs keeps it invariant.
//   tv_planes_persall bench-only: the 144 sites of a round in ONE launch, a
//                     persistent grid walking a work list of row groups. It
//                     is the ceiling of what removing every launch AND every
//                     wave boundary can buy. The served path cannot use it —
//                     rotation, attention and norms sit between the sites —
//                     so it is measured to bound A2+A3 and never ported
//                     (design note, bras b2).
//
// ## The invariants every arm keeps
//
//  * `planes_dot` is called unchanged, on the same stream words, with the
//    same `gscale`/`gs_off` resolution as `tv_planes_seg`. No arm changes
//    what a weight is worth, and the bytes read from the stream are the
//    reference's, byte for byte.
//  * One row is written once, by one thread, with a plain store. No atomic
//    on any float, anywhere — planes_seg.cu's argument holds here too.
//  * Every arm but `sk` accumulates a row's blocks in the SAME lane order as
//    `tv_planes_seg` (lane, lane+32, … straight across tiles), so its output
//    must be BIT-IDENTICAL to the reference kernel's; the bench asserts it,
//    row by row, before a single timing. `sk` sums per-slice partials in a
//    fixed order: bit-identical only where `nsplit == 1`, f64-checked
//    elsewhere.
//
// ## Compiled here, validated on the card
//
// `tests/host_planes_occ.cpp` compiles this file as host C++ and EXECUTES
// the two scalar helpers it can — `occ_xs_index` and `occ_slice` — against
// the Rust mirrors in `llvq_cuda::occ`. The kernels contain barriers,
// shuffles and one atomic; they are proved by the bench's comparisons and by
// nothing else. Same standing as planes_seg.cu.
//
// Assembly contract: llvq_slot.cuh, matvec.cu, llvq_planes.cuh, planes.cu,
// planes_seg.cu, …, then this file — it needs TILE_BLOCKS and warp_sum
// (matvec.cu) and planes_dot (llvq_planes.cuh). The guards below only
// resolve from disk under a host clang++ check; NVRTC gets the concatenation.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_PLANES_CUH
#include "llvq_planes.cuh"
#endif

// The padded stride, in floats, of one staged block: 28 × 4 = 112 bytes.
// Mirrored by `llvq_cuda::occ::XS_PAD`, pinned by a test there.
#define LLVQ_XS_PAD 28u

// One site of the persistent-global arm, 8 × u64 laid out by
// `llvq_cuda::occ::site_words`:
//   [0] words  [1] gscale  [2] gs_off  [3] rscale  [4] tail  [5] y
//   [6] nblocks | tail_w << 32        [7] group0 | ngroups << 32
#define LLVQ_OCC_SITE_WORDS 8u

// Where staged float `i` of a tile lands in `xs` under stride XS.
template <u32 XS>
__device__ __forceinline__ u32 occ_xs_index(u32 i)
{
    if constexpr (XS == LLVQ_DIM) {
        return i;
    } else {
        return (i / LLVQ_DIM) * XS + (i % LLVQ_DIM);
    }
}

// The K slice [klo, khi) of split `s` among `nsplit`, in blocks. Slices are
// `ceil(nblocks / nsplit)` wide; a trailing slice may be shorter, or empty —
// it then contributes an exact 0.0f and still draws its ticket, which is
// what keeps the fixup's arithmetic uniform across the group.
__device__ __forceinline__ void occ_slice(u32 nblocks, u32 nsplit, u32 s, u32* klo, u32* khi)
{
    u32 per = (nblocks + nsplit - 1u) / nsplit;
    u32 lo  = s * per;
    lo = lo < nblocks ? lo : nblocks;
    u32 hi = lo + per;
    *klo = lo;
    *khi = hi < nblocks ? hi : nblocks;
}

// R rows per warp over the K range [klo, khi), staged tile by tile.
//
// `row0` is the warp's first row and `acc[r]` its r-th accumulator. The
// staging and the two barriers are tv_planes', so `khi - klo` must be
// uniform across the CTA (it is: every argument is). For R = 1, XS = 24,
// klo = 0 and khi = nblocks this is tv_planes_seg's loop, block for block in
// the same lane order — which is what makes the bit-exact check possible.
template <u32 R, u32 XS>
__device__ __forceinline__ void occ_rows_tiles(const u32* __restrict__ words,
                                               const ClassRec* __restrict__ tab,
                                               const float* __restrict__ gs,
                                               const float* __restrict__ x,
                                               float* xs,
                                               u32 row0, u32 nblocks, u32 klo, u32 khi,
                                               float* acc)
{
    u32 lane = threadIdx.x & 31u;
    for (u32 jlo = klo; jlo < khi; jlo += TILE_BLOCKS) {
        u32 jhi = jlo + TILE_BLOCKS < khi ? jlo + TILE_BLOCKS : khi;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        // Two barriers, for the reason matvec.cu gives: the first stops this
        // fill from racing a straggler still reading the previous tile.
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x)
            xs[occ_xs_index<XS>(i)] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * XS;
#pragma unroll
            for (u32 r = 0; r < R; ++r)
                acc[r] += planes_dot(words, tab, gs, (row0 + r) * nblocks + j, xb);
        }
    }
}

// Warp-reduce R accumulators and store their rows: tv_planes' epilogue, R
// times. The tail reads `x` from global memory — the tail columns sit past
// the last staged window — exactly as every fused kernel does.
template <u32 R>
__device__ __forceinline__ void occ_store(const float* __restrict__ rscale,
                                          const float* __restrict__ tail,
                                          const float* __restrict__ x,
                                          float* __restrict__ y,
                                          u32 row0, u32 nblocks, u32 tail_w, float* acc)
{
    u32 lane = threadIdx.x & 31u;
#pragma unroll
    for (u32 r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
    if (lane == 0) {
        u32 tc0 = nblocks * LLVQ_DIM;
#pragma unroll
        for (u32 r = 0; r < R; ++r) {
            u32 row  = row0 + r;
            float tv = 0.0f;
            for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
            y[row] = acc[r] * rscale[row] + tv;
        }
    }
}

// ---- pad / mr2 / mr4 / mr2p : R rows per warp, stride XS -------------------
//
// Grid: `d_out · 32 / (threads · R)` CTAs, exact — no bounds guard, for the
// reason every fused kernel gives (a return before `__syncthreads()`
// deadlocks). The host asserts `d_out % (8R) == 0`; true of the four fused
// shapes of the 4B for R ≤ 4 (`occ::mr_grid`).
template <u32 R, u32 XS>
__device__ __forceinline__ void occ_mr(const u32* __restrict__ words,
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
    u32 row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * R;
    // R divides 8 and every segment is a whole number of 8-row groups
    // (`seg_concat` asserts it per part), so a warp's R rows never straddle
    // a segment and share one centroid pair. Warp-uniform: it broadcasts.
    const float* gs = gscale + gs_off[row0];
    float acc[R];
#pragma unroll
    for (u32 r = 0; r < R; ++r) acc[r] = 0.0f;
    occ_rows_tiles<R, XS>(words, tab, gs, x, xs, row0, nblocks, 0u, nblocks, acc);
    occ_store<R>(rscale, tail, x, y, row0, nblocks, tail_w, acc);
}

#define LLVQ_OCC_SEG_ARGS                                                        \
    const u32* __restrict__ words, const ClassRec* __restrict__ tab,             \
    const float* __restrict__ gscale, const u32* __restrict__ gs_off,            \
    const float* __restrict__ rscale, const float* __restrict__ tail,            \
    const float* __restrict__ x, float* __restrict__ y, u32 nblocks, u32 tail_w
#define LLVQ_OCC_SEG_PASS words, tab, gscale, gs_off, rscale, tail, x, y, nblocks, tail_w

extern "C" __global__ void tv_planes_pad(LLVQ_OCC_SEG_ARGS)
{
    occ_mr<1u, LLVQ_XS_PAD>(LLVQ_OCC_SEG_PASS);
}

extern "C" __global__ void tv_planes_mr2(LLVQ_OCC_SEG_ARGS)
{
    occ_mr<2u, LLVQ_DIM>(LLVQ_OCC_SEG_PASS);
}

extern "C" __global__ void tv_planes_mr4(LLVQ_OCC_SEG_ARGS)
{
    occ_mr<4u, LLVQ_DIM>(LLVQ_OCC_SEG_PASS);
}

extern "C" __global__ void tv_planes_mr2p(LLVQ_OCC_SEG_ARGS)
{
    occ_mr<2u, LLVQ_XS_PAD>(LLVQ_OCC_SEG_PASS);
}

// ---- pers : persistent per site --------------------------------------------
//
// Grid: `min(ngroups, resident CTAs)` where the residency is READ off the
// loaded function and the card (`occ::residency`), never assumed. Each CTA
// walks the groups `blockIdx.x, blockIdx.x + gridDim.x, …`; the walk is
// uniform across the CTA, so the barriers inside stay matched.
template <u32 XS>
__device__ __forceinline__ void occ_pers(LLVQ_OCC_SEG_ARGS, u32 ngroups)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 wi   = threadIdx.x >> 5;
    u32 nw   = blockDim.x >> 5;
    if (nblocks <= TILE_BLOCKS) {
        // One tile: the activation is staged ONCE per CTA and every group
        // this CTA visits reads it. That is the whole persistence dividend on
        // the 2560-wide sites. On o/down the branch below re-stages per tile
        // as tv_planes does, and persistence only saves the CTA turnover.
        u32 n = nblocks * LLVQ_DIM;
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[occ_xs_index<XS>(i)] = x[i];
        __syncthreads();
        for (u32 g = blockIdx.x; g < ngroups; g += gridDim.x) {
            u32 row = g * nw + wi;
            const float* gs = gscale + gs_off[row];
            float acc[1] = {0.0f};
            for (u32 j = lane; j < nblocks; j += 32u)
                acc[0] += planes_dot(words, tab, gs, row * nblocks + j, xs + j * XS);
            occ_store<1u>(rscale, tail, x, y, row, nblocks, tail_w, acc);
        }
    } else {
        for (u32 g = blockIdx.x; g < ngroups; g += gridDim.x) {
            u32 row = g * nw + wi;
            const float* gs = gscale + gs_off[row];
            float acc[1] = {0.0f};
            occ_rows_tiles<1u, XS>(words, tab, gs, x, xs, row, nblocks, 0u, nblocks, acc);
            occ_store<1u>(rscale, tail, x, y, row, nblocks, tail_w, acc);
        }
    }
}

extern "C" __global__ void tv_planes_pers(LLVQ_OCC_SEG_ARGS, u32 ngroups)
{
    occ_pers<LLVQ_DIM>(LLVQ_OCC_SEG_PASS, ngroups);
}

// ---- sk : split-K across CTAs, fixed-order fixup by the last CTA ------------
//
// Grid: `(d_out / 8) · nsplit` CTAs; CTA b handles group `b / nsplit`, slice
// `b % nsplit`, so a group's slices are adjacent in launch order and their
// partials are still in L2 when the last one sums them. `part` holds
// `nsplit · d_out` raw partials (before `rscale`), `done` one ticket
// counter per group — zeroed once at allocation and RESET by the fixup, so
// a launch leaves the counters as it found them and needs no memset.
//
// Memory order is the CUDA "threadFenceReduction" pattern: each writer
// fences its partial before the CTA's ticket is drawn; the last CTA fences
// again, then reads the partials through `volatile` so the loads bypass an
// L1 that may hold stale lines. The sum runs k = 0..nsplit-1 whatever CTA
// performs it: the result does not depend on who finished last.
template <u32 XS>
__device__ __forceinline__ void occ_sk(LLVQ_OCC_SEG_ARGS,
                                       float* __restrict__ part,
                                       u32* __restrict__ done,
                                       u32 nsplit,
                                       u32 d_out)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 wi   = threadIdx.x >> 5;
    u32 nw   = blockDim.x >> 5;
    u32 g    = blockIdx.x / nsplit;
    u32 s    = blockIdx.x - g * nsplit;
    u32 row  = g * nw + wi;
    u32 klo, khi;
    occ_slice(nblocks, nsplit, s, &klo, &khi);
    const float* gs = gscale + gs_off[row];
    float acc[1] = {0.0f};
    occ_rows_tiles<1u, XS>(words, tab, gs, x, xs, row, nblocks, klo, khi, acc);
    if (nsplit == 1u) {
        // No partial, no ticket: this is tv_planes_seg, bit for bit.
        occ_store<1u>(rscale, tail, x, y, row, nblocks, tail_w, acc);
        return;
    }
    float a = warp_sum(acc[0]);
    if (lane == 0) part[s * d_out + row] = a;
    // Release: this CTA's partials are device-visible before its ticket.
    __threadfence();
    u32 ticket = 0u;
    if (threadIdx.x == 0) ticket = atomicAdd(done + g, 1u);
    // The barrier and the broadcast in one: every thread learns whether this
    // CTA drew the group's last ticket.
    int last = __syncthreads_or(threadIdx.x == 0 && ticket == nsplit - 1u);
    if (last) {
        __threadfence();  // acquire: the loads below follow the ticket we saw
        if (lane == 0) {
            const volatile float* p = part;
            float sum = 0.0f;
            for (u32 k = 0; k < nsplit; ++k) sum += p[k * d_out + row];
            u32 tc0  = nblocks * LLVQ_DIM;
            float tv = 0.0f;
            for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
            y[row] = sum * rscale[row] + tv;
        }
        // Self-reset. Every other CTA of this group has drawn its ticket —
        // that is what `last` means — so nothing reads the counter again
        // before the next launch, which is ordered after this one.
        if (threadIdx.x == 0) done[g] = 0u;
    }
}

extern "C" __global__ void tv_planes_sk(LLVQ_OCC_SEG_ARGS,
                                        float* __restrict__ part,
                                        u32* __restrict__ done,
                                        u32 nsplit,
                                        u32 d_out)
{
    occ_sk<LLVQ_DIM>(LLVQ_OCC_SEG_PASS, part, done, nsplit, d_out);
}

// ---- persall : every site of a round in one launch (bench only) ------------
//
// `sites` is the descriptor table, ascending in `group0`; group w belongs to
// the last site whose `group0 <= w`. A CTA walks w = blockIdx.x + k·gridDim.x,
// so its site index only ever advances — one forward scan, no search. The
// activation is the bench's single shared `x`; a single-tile site's staging
// is kept in `xs` across consecutive groups of the SAME width (qkv and
// gate_up both stage 106 blocks of the same x, so it even survives a change
// of site) and re-staged only when the width changes. Every branch below is
// uniform across the CTA.
template <u32 XS>
__device__ __forceinline__ void occ_persall(const u64* __restrict__ sites,
                                            u32 nsites,
                                            const ClassRec* __restrict__ tab,
                                            const float* __restrict__ x,
                                            u32 total_groups)
{
    extern __shared__ float xs[];
    u32 lane   = threadIdx.x & 31u;
    u32 wi     = threadIdx.x >> 5;
    u32 nw     = blockDim.x >> 5;
    u32 si     = 0u;
    u32 staged = 0xffffffffu;  // blocks currently in xs (single-tile sites only)
    for (u32 w = blockIdx.x; w < total_groups; w += gridDim.x) {
        while (si + 1u < nsites
               && w >= (u32)(sites[(si + 1u) * LLVQ_OCC_SITE_WORDS + 7u] & 0xffffffffu))
            ++si;
        const u64* d = sites + si * LLVQ_OCC_SITE_WORDS;
        const u32*   words  = (const u32*)d[0];
        const float* gscale = (const float*)d[1];
        const u32*   gs_off = (const u32*)d[2];
        const float* rscale = (const float*)d[3];
        const float* tail   = (const float*)d[4];
        float*       y      = (float*)d[5];
        u32 nblocks = (u32)(d[6] & 0xffffffffu);
        u32 tail_w  = (u32)(d[6] >> 32);
        u32 group0  = (u32)(d[7] & 0xffffffffu);
        u32 row = (w - group0) * nw + wi;
        const float* gs = gscale + gs_off[row];
        float acc[1] = {0.0f};
        if (nblocks <= TILE_BLOCKS) {
            if (staged != nblocks) {
                u32 n = nblocks * LLVQ_DIM;
                __syncthreads();  // nobody still reads the previous tile
                for (u32 i = threadIdx.x; i < n; i += blockDim.x)
                    xs[occ_xs_index<XS>(i)] = x[i];
                __syncthreads();
                staged = nblocks;
            }
            for (u32 j = lane; j < nblocks; j += 32u)
                acc[0] += planes_dot(words, tab, gs, row * nblocks + j, xs + j * XS);
        } else {
            occ_rows_tiles<1u, XS>(words, tab, gs, x, xs, row, nblocks, 0u, nblocks, acc);
            staged = 0xffffffffu;
        }
        occ_store<1u>(rscale, tail, x, y, row, nblocks, tail_w, acc);
    }
}

extern "C" __global__ void tv_planes_persall(const u64* __restrict__ sites,
                                             u32 nsites,
                                             const ClassRec* __restrict__ tab,
                                             const float* __restrict__ x,
                                             u32 total_groups)
{
    occ_persall<LLVQ_DIM>(sites, nsites, tab, x, total_groups);
}
