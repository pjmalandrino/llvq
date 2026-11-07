// tv_trows_*: 4E's Tetra row loops alone, the row-loop microbenchmark
// (2026-11-05, docs/mesures/pers-4e-rowloop-host-2026-11-05.txt).
//
// Why: the 4E v2 and v3 card readings missed their Tetra-row estimates by 35
// to 50 %, and every in-SM bound read off the SASS (issue, dependency chain,
// shared pipe, words in flight) sits three times or more above the measured
// rate. This kernel times the loops themselves, at 4E's geometry, so the next
// estimate is fitted on measured rates:
//   - one CTA of 512 threads a SM, forced by 4E's own shared staging (header,
//     decoder tables, activation of up to 9,728 values: 57,984 bytes);
//   - each site's rows split over CTAs as 4E splits a phase, [d c / g,
//     d (c + 1) / g), and over warps statically, lo + warp + k x nwarps;
//   - sites one after the other, a CTA barrier between them (the activation is
//     restaged), no grid barrier.
// With `stamps` set, each warp writes its busy cycles a site and the CTA its
// own, and thread 0 writes the SM clock and the grid clock at the launch's
// two ends: the cycles a block iteration, the warps' balance inside a CTA,
// and the SM clock in MHz.
//
// Arms (the first template parameter):
//   V2      pers_tetra_row<24, false, true>, 4E v2's row: one block a step,
//           the next block's words in flight; then pers_epilogue. Owes the bits.
//   V3      pers_tetra_stream, 4E v3's stream. Owes the bits.
//   ILP2    two blocks a step in one straight-line region: their dot products
//           are independent, and are folded in the served order. Owes the bits.
//   V2U     V2 with a trip count uniform over the warp (nit = ceil(nblocks /
//           32)), a lane past the row folding nothing. Owes the bits. It is the
//           control of the three below, which change one input of its loop:
//   L1W     every row reads row 0's words: they stay in L1 and L2; the
//           instructions and every shared access are V2U's.
//   BCAST   every lane decodes lane 0's words (two shuffles): each table read
//           is a broadcast; every lane still loads its own words from DRAM.
//   XBCAST  every lane reads the activation of the step's lane-0 block: the
//           six LDS.128 a block are broadcasts.
//   The last three owe nothing; their rows are never checked.
//   K2      one decode, two activations: each block is decoded once and dotted
//           with the activation `x` and with a second one `x2`, each sum in
//           the served order, so each of its two rows owes the bits of the
//           one-activation path on its own activation. The door to k tokens a
//           pass (speculative verification, the prefill GEMM route): its time
//           against V2's is the cost of a second token as a fraction of a pass.
//           Its two activations take 77,824 shared bytes (96,896 in all).
//
// The K4 arms of plan-300's step 1 (2026-11-07,
// proofs/preregistration-step1-micro-2026-11-07.md). Each changes one thing
// in V2's row and owes its bits:
//   NC      the word loads through the read-only path (`ld.global.nc`).
//   NC256   the same with no L1 allocation and a 256-byte L2 prefetch
//           (`ld.global.nc.L1::no_allocate.L2::256B`, sm_80 and later).
//   CPA     each warp stages the words of its row two steps ahead (32 blocks,
//           192 B a step) by `cp.async` into a ring of three slots in shared
//           memory, and decodes from there. The ring sits where K2 stages its
//           second activation: 16 warps x 576 B = 9,216 B, 67,200 B in all.
//           Every row stride must be a multiple of 16 B (the host checks).
//   TILE    the words relaid once (tv_trows_tile_relayout) so one 128-bit load
//           gives a lane the low words of its blocks l, l+32, l+64, l+96 and
//           one 64-bit load their four high halves; the blocks past the last
//           full tile of 128 sit in plane form. Same bytes a row, same lane to
//           block map, same fold order.
//   BAL     V2's row, the rows of the CTA's slice claimed one at a time from
//           the shared cursor (pers_claim) instead of the static warp split:
//           the warp-balance suspect of pers-rowloop (0.63 at d_in 4,096).
//
// Composition: its own module, the persistent unit followed by this file
// (pers.rs::trows_unit_text), so 4E's module keeps its bytes.

#define TROWS_V2 0u
#define TROWS_V3 1u
#define TROWS_ILP2 2u
#define TROWS_V2U 3u
#define TROWS_L1W 4u
#define TROWS_BCAST 5u
#define TROWS_XBCAST 6u
#define TROWS_K2 7u
#define TROWS_NC 8u
#define TROWS_NC256 9u
#define TROWS_CPA 10u
#define TROWS_TILE 11u
#define TROWS_BAL 12u

// Site words, 10 x u64, laid out by pers.rs::trows_site_words:
//   [0] words  [1] gscale  [2] rscale  [3] tail (binary16)  [4] y (binary16)
//   [5] d_out | nblocks << 32  [6] tail_w | row_stride_u32 << 32
//   [7] the f32 value of each row before f2h, or 0 (the card's launches)
//   [8] K2's rows on x2 (binary16), or 0  [9] their f32 values, or 0
#define TROWS_SITE_WORDS 10u
// Where K2 stages its second activation: after the widest first one.
#define TROWS_X2_AT 9728u

// CPA's ring: three slots of 48 u32 (one step of 32 blocks) a warp.
#define TROWS_CPA_SLOT 48u
#define TROWS_CPA_WARP (3u * TROWS_CPA_SLOT)

// The SM clock (cycles). The host harness counts calls, like pers_now.
__device__ __forceinline__ unsigned long long trows_clock()
{
#ifdef LLVQ_HOST_BUILD
    static unsigned long long t = 0ull;
    return ++t;
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
#endif
}

// ILP2: a lane's blocks j = lane + 32 i in pairs (i, i + 1). Both blocks are
// decoded and dotted in one region with no branch between them, so their
// chains can overlap; the folds then run in i order, each with pers_block's
// arithmetic (a = 0, 24 fmas in TETRA48_ORDER, pers_fold). The pair's words
// are loaded one step ahead. A second block past the row is decoded from zero
// words on the first block's activation and never folded.
__device__ __forceinline__ float trows_row_ilp2(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    float acc = 0.0f;
    u32 l0 = 0u, h0 = 0u, l1 = 0u, h1 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, l0, h0);
    if (lane + 32u < nblocks) f1r_load(wrow, lane + 32u, l1, h1);
    for (u32 j = lane; j < nblocks; j += 64u) {
        u32 m0 = 0u, g0 = 0u, m1 = 0u, g1 = 0u;
        if (j + 64u < nblocks) f1r_load(wrow, j + 64u, m0, g0);
        if (j + 96u < nblocks) f1r_load(wrow, j + 96u, m1, g1);
        const bool two = j + 32u < nblocks;
        const float* x0 = xs + j * 24u;
        const float* x1 = two ? x0 + 32u * 24u : x0;
        u32 q0[6], q1[6];
        f1r_v3_quads(l0, h0, tab, q0);
        f1r_v3_quads(l1, h1, tab, q1);
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
            for (u32 k = 0; k < 4u; ++k) {
                a0 = __fmaf_rn(f1r_v3_float(q0[i], k), x0[TETRA48_ORDER[4u * i + k]], a0);
                a1 = __fmaf_rn(f1r_v3_float(q1[i], k), x1[TETRA48_ORDER[4u * i + k]], a1);
            }
        }
        acc = pers_fold(acc, a0, q0, h0, gscale, invnorm);
        if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);
        l0 = m0;
        h0 = g0;
        l1 = m1;
        h1 = g1;
    }
    return warp_sum(acc);
}

// K2: V2's loop (one block a step, the next block's words in flight) with
// each block decoded once and dotted with two activations, `xs` and `xs2`.
// Each dot is pers_block's arithmetic on its own activation (a = 0, 24 fmas
// in TETRA48_ORDER, pers_fold), so each sum is the one-activation sum.
// The first sum is returned, the second written to `out2`.
__device__ __forceinline__ float trows_row_k2(const u32* wrow, const F1rTables& tab, const float* gscale,
                                              const float* invnorm, const float* xs, const float* xs2, u32 nblocks,
                                              u32 lane, float& out2)
{
    float acc = 0.0f, acc2 = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        const float* xa = xs + j * 24u;
        const float* xb = xs2 + j * 24u;
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
            for (u32 k = 0; k < 4u; ++k) {
                const float w = f1r_v3_float(q[i], k);
                a = __fmaf_rn(w, xa[TETRA48_ORDER[4u * i + k]], a);
                b = __fmaf_rn(w, xb[TETRA48_ORDER[4u * i + k]], b);
            }
        }
        acc = pers_fold(acc, a, q, hi16, gscale, invnorm);
        acc2 = pers_fold(acc2, b, q, hi16, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    out2 = warp_sum(acc2);
    return warp_sum(acc);
}

// The word loads of the K4 arms: 0 plain, 1 the read-only path, 2 the
// read-only path with no L1 allocation and a 256-byte L2 prefetch. The same
// value whatever the path: the words are never written during a launch.
template <u32 LD>
__device__ __forceinline__ u32 trows_ldw(const u32* p)
{
#ifdef LLVQ_HOST_BUILD
    return LD == 1u ? __ldg(p) : *p;
#else
    if (LD == 0u) return *p;
    if (LD == 1u) return __ldg(p);
    u32 v;
    asm("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
#endif
}

// f1r_load with the loads of path LD.
template <u32 LD>
__device__ __forceinline__ void trows_load(const u32* row, u32 j, u32& lo, u32& hi16)
{
    const u32 w = (3u * j) >> 1;
    const u32 w0 = trows_ldw<LD>(row + w);
    const u32 w1 = trows_ldw<LD>(row + w + 1u);
    const u32 odd = j & 1u;
    lo = odd ? ((w0 >> 16) | (w1 << 16)) : w0;
    hi16 = odd ? (w1 >> 16) : (w1 & 0xffffu);
}

// NC, NC256: pers_tetra_row<24, false, true> with the loads of path LD.
template <u32 LD>
__device__ __forceinline__ float trows_row_ld(const u32* wrow, const F1rTables& tab, const float* gscale,
                                              const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    float acc = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) trows_load<LD>(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) trows_load<LD>(wrow, j + 32u, lo2, hi2);
        acc = pers_block(acc, lo, hi16, tab, xs + j * 24u, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    return warp_sum(acc);
}

// cp.async of 16 bytes, global to shared, cached in L2 only. The host build
// copies at once, so its commit and wait are empty: the host checks the ring's
// indexing, never the asynchrony.
__device__ __forceinline__ void trows_cp16(u32* dst, const u32* src)
{
#ifdef LLVQ_HOST_BUILD
    for (u32 i = 0; i < 4u; ++i) dst[i] = src[i];
#else
    unsigned long long s64;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(s64) : "l"(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)s64), "l"(src) : "memory");
#endif
}
__device__ __forceinline__ void trows_cp_commit()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
// At most two groups of this thread still in flight.
__device__ __forceinline__ void trows_cp_wait2()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.wait_group 2;" ::: "memory");
#endif
}

// CPA: step i of the row (bytes [192 i, 192 i + 192)) goes to slot i % 3 of
// the warp's ring, twelve 16-byte chunks on lanes 0..11, a chunk only if it
// lies inside the row's `row_u32` words. Every lane commits one group a step,
// empty or not, so `wait_group 2` means "step i has landed" on every lane.
__device__ __forceinline__ void trows_cpa_issue(u32* ring, const u32* wrow, u32 row_u32, u32 i, u32 nit, u32 lane)
{
    if (i < nit && lane < 12u) {
        const u32 c = TROWS_CPA_SLOT * i + 4u * lane;
        if (c + 4u <= row_u32) trows_cp16(ring + TROWS_CPA_SLOT * (i % 3u) + 4u * lane, wrow + c);
    }
    trows_cp_commit();
}

__device__ __forceinline__ float trows_row_cpa(const u32* wrow, u32 row_u32, u32* ring, const F1rTables& tab,
                                               const float* gscale, const float* invnorm, const float* xs, u32 nblocks,
                                               u32 lane)
{
    const u32 nit = (nblocks + 31u) >> 5;
    float acc = 0.0f;
    trows_cpa_issue(ring, wrow, row_u32, 0u, nit, lane);
    trows_cpa_issue(ring, wrow, row_u32, 1u, nit, lane);
    for (u32 i = 0; i < nit; ++i) {
        // slot (i + 2) % 3 was last read at step i - 1, before its __syncwarp
        trows_cpa_issue(ring, wrow, row_u32, i + 2u, nit, lane);
        trows_cp_wait2();
        __syncwarp();
        const u32 j = 32u * i + lane;
        if (j < nblocks) {
            u32 lo, hi16;
            f1r_load(ring + TROWS_CPA_SLOT * (i % 3u), lane, lo, hi16);
            acc = pers_block(acc, lo, hi16, tab, xs + j * 24u, gscale, invnorm);
        }
        __syncwarp();
    }
    return warp_sum(acc);
}

// TILE's layout, a row of `nblocks` blocks: ntiles = nblocks / 128 tiles of
// 192 u32, each a low plane (lane l's uint4 at u32 4l: the low words of
// blocks 128t + l + 32s, s = 0..3) then a high plane (lane l's 64-bit word at
// u32 128 + 2l: their four high halves, s = 0 in the low 16 bits); then the
// rem = nblocks - 128 ntiles blocks left, a plane of rem u32 low words and a
// plane of rem u16 high halves, the row padded to 16 bytes. 6 bytes a block
// as in the served stream; at the 4B's widths (106, 170 and 405 blocks) the
// stride is the served one.
__device__ __forceinline__ u32 trows_tile_stride(u32 nblocks)
{
    const u32 nt = nblocks >> 7, rem = nblocks - (nt << 7);
    return 192u * nt + ((6u * rem + 15u) / 16u) * 4u;
}

// Block j's low word and high half into a TILE row.
__device__ __forceinline__ void trows_tile_put(u32* drow, u32 nblocks, u32 j, u32 lo, u32 hi16)
{
    const u32 nt = nblocks >> 7, rem = nblocks - (nt << 7);
    unsigned short* h = reinterpret_cast<unsigned short*>(drow);
    if (j < (nt << 7)) {
        const u32 t = j >> 7, k = j & 127u, s = k >> 5, l = k & 31u;
        drow[192u * t + 4u * l + s] = lo;
        h[2u * (192u * t + 128u) + 4u * l + s] = (unsigned short)hi16;
    } else {
        const u32 rr = j - (nt << 7);
        drow[192u * nt + rr] = lo;
        h[2u * (192u * nt + rem) + rr] = (unsigned short)hi16;
    }
}

// The relayout, once at setup: every block of every row of one site, read
// as the served stream reads it (f1r_load) and written into the TILE layout.
// The destination comes zeroed, so the padding is zero.
extern "C" __global__ void tv_trows_tile_relayout(const u32* __restrict__ src, u32 src_stride, u32* __restrict__ dst,
                                                  u32 dst_stride, u32 d_out, u32 nblocks)
{
    const unsigned long long n = (unsigned long long)d_out * nblocks;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const u32 r = (u32)(i / nblocks), j = (u32)(i - (unsigned long long)r * nblocks);
        u32 lo, hi16;
        f1r_load(src + (size_t)r * src_stride, j, lo, hi16);
        trows_tile_put(dst + (size_t)r * dst_stride, nblocks, j, lo, hi16);
    }
}

// TILE: V2's row read from the TILE layout, the next tile (or the next block
// left) in flight while one decodes. Lane l folds blocks l, l+32, ... in
// increasing order, as V2.
__device__ __forceinline__ float trows_row_tile(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    const u32 nt = nblocks >> 7, rem = nblocks - (nt << 7);
    float acc = 0.0f;
    uint4 L{0u, 0u, 0u, 0u};
    unsigned long long H = 0ull;
    if (nt > 0u) {
        L = *reinterpret_cast<const uint4*>(wrow + 4u * lane);
        H = *reinterpret_cast<const unsigned long long*>(wrow + 128u + 2u * lane);
    }
    for (u32 t = 0; t < nt; ++t) {
        uint4 L2{0u, 0u, 0u, 0u};
        unsigned long long H2 = 0ull;
        if (t + 1u < nt) {
            const u32* nx = wrow + 192u * (t + 1u);
            L2 = *reinterpret_cast<const uint4*>(nx + 4u * lane);
            H2 = *reinterpret_cast<const unsigned long long*>(nx + 128u + 2u * lane);
        }
        const u32 lo[4] = {L.x, L.y, L.z, L.w};
        const u32 j0 = (t << 7) + lane;
#pragma unroll
        for (u32 s = 0; s < 4u; ++s)
            acc = pers_block(acc, lo[s], (u32)(H >> (16u * s)) & 0xffffu, tab, xs + (j0 + 32u * s) * 24u, gscale,
                             invnorm);
        L = L2;
        H = H2;
    }
    const u32* rlo = wrow + 192u * nt;
    const unsigned short* rhi = reinterpret_cast<const unsigned short*>(rlo + rem);
    u32 lo = 0u, hi16 = 0u;
    if (lane < rem) {
        lo = rlo[lane];
        hi16 = rhi[lane];
    }
    for (u32 k = lane; k < rem; k += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (k + 32u < rem) {
            lo2 = rlo[k + 32u];
            hi2 = rhi[k + 32u];
        }
        acc = pers_block(acc, lo, hi16, tab, xs + ((nt << 7) + k) * 24u, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    return warp_sum(acc);
}

// V2U and its three diagnostics: V2's loop with a uniform trip count. Every
// lane runs nit steps; a lane whose block j is past the row loads nothing and
// folds nothing, so the shuffles of BCAST see the whole warp.
template <u32 ARM>
__device__ __forceinline__ float trows_row_uniform(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                   const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    const u32 nit = (nblocks + 31u) >> 5;
    float acc = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 i = 0; i < nit; ++i) {
        const u32 j = lane + 32u * i;
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        u32 a = lo, b = hi16;
        if (ARM == TROWS_BCAST) {
            a = __shfl_sync(0xffffffffu, lo, 0);
            b = __shfl_sync(0xffffffffu, hi16, 0);
        }
        const float* xb = xs + (ARM == TROWS_XBCAST ? 32u * i : j) * 24u;
        if (j < nblocks) acc = pers_block(acc, a, b, tab, xb, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    return warp_sum(acc);
}

template <u32 ARM>
__device__ __forceinline__ void trows_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                           const u32* __restrict__ rows, const unsigned char* __restrict__ prefixes,
                                           const unsigned short* __restrict__ branches,
                                           const unsigned char* __restrict__ suffixes,
                                           const float* __restrict__ invnorm, const float* __restrict__ x,
                                           const float* __restrict__ x2, u32* __restrict__ stamps,
                                           unsigned long long* __restrict__ ends)
{
    if (stamps != nullptr && threadIdx.x == 0u) {
        ends[4u * blockIdx.x] = trows_clock();
        ends[4u * blockIdx.x + 1u] = pers_now();
    }
    // 4E's layout (pers_4e.cu): the header, the tables, invnorm, then the
    // activation in natural staging, stride 24.
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* xs = t_inv + 32;
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = prefixes[i];
        t_suf[i] = suffixes[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = branches[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = invnorm[i];
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    for (u32 s = 0; s < nsites; ++s) {
        const unsigned long long* d = sites + (size_t)s * TROWS_SITE_WORDS;
        const u32* words = reinterpret_cast<const u32*>(d[0]);
        const float* gs = reinterpret_cast<const float*>(d[1]);
        const float* rs = reinterpret_cast<const float*>(d[2]);
        const unsigned short* tail = reinterpret_cast<const unsigned short*>(d[3]);
        unsigned short* y = reinterpret_cast<unsigned short*>(d[4]);
        const u32 d_out = (u32)(d[5] & 0xffffffffull), nb = (u32)(d[5] >> 32);
        const u32 tw = (u32)(d[6] & 0xffffffffull), stride = (u32)(d[6] >> 32);
        float* yf = reinterpret_cast<float*>(d[7]);
        unsigned short* y2 = reinterpret_cast<unsigned short*>(d[8]);
        float* yf2 = reinterpret_cast<float*>(d[9]);
        float* xs2 = xs + TROWS_X2_AT;
        pers_stage_f32<24u, false>(x, xs, nb, tw);  // barriers on both sides
        if (ARM == TROWS_K2) pers_stage_f32<24u, false>(x2, xs2, nb, tw);
        const u32 lo = (u32)((unsigned long long)d_out * c / g), hi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        const unsigned long long t0 = trows_clock();
        if (ARM == TROWS_V3) {
            const auto row_of = [&](u32 k) {
                const u32 r = lo + warp + k * nw;
                return PersTRow{words + (size_t)r * stride, gs, rs, tail, r};
            };
            const auto start = [](u32) {};
            const auto epi = [&](u32 k, float v) {
                y[lo + warp + k * nw] = f2h(v);
                if (yf != nullptr) yf[lo + warp + k * nw] = v;
            };
            pers_tetra_stream<PERS_STREAM_D>(pers_warp_rows(lo, hi, warp, nw), nb, tw, tab, t_inv, xs, lane, row_of,
                                             start, epi);
        } else if (ARM == TROWS_K2) {
            for (u32 r = lo + warp; r < hi; r += nw) {
                float acc2;
                const float acc = trows_row_k2(words + (size_t)r * stride, tab, gs, t_inv, xs, xs2, nb, lane, acc2);
                if (lane == 0u) {
                    const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
                    const float v2 = pers_epilogue(acc2, rs[r], tail + (size_t)r * tw, xs2 + nb * 24u, tw);
                    y[r] = f2h(v);
                    y2[r] = f2h(v2);
                    if (yf != nullptr) yf[r] = v;
                    if (yf2 != nullptr) yf2[r] = v2;
                }
            }
        } else if (ARM == TROWS_BAL) {
            pers_claim_reset(lo);
            for (;;) {
                const u32 r = pers_claim(lane, 1u);
                if (r >= hi) break;
                const float acc = pers_tetra_row<24u, false, true>(words + (size_t)r * stride, tab, gs, t_inv, xs, nb, lane);
                if (lane == 0u) {
                    const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
                    y[r] = f2h(v);
                    if (yf != nullptr) yf[r] = v;
                }
            }
        } else {
            u32* ring = reinterpret_cast<u32*>(xs + TROWS_X2_AT) + warp * TROWS_CPA_WARP;
            for (u32 r = lo + warp; r < hi; r += nw) {
                const u32* wrow = words + (size_t)(ARM == TROWS_L1W ? 0u : r) * stride;
                float acc;
                if (ARM == TROWS_V2) {
                    acc = pers_tetra_row<24u, false, true>(wrow, tab, gs, t_inv, xs, nb, lane);
                } else if (ARM == TROWS_NC) {
                    acc = trows_row_ld<1u>(wrow, tab, gs, t_inv, xs, nb, lane);
                } else if (ARM == TROWS_NC256) {
                    acc = trows_row_ld<2u>(wrow, tab, gs, t_inv, xs, nb, lane);
                } else if (ARM == TROWS_CPA) {
                    acc = trows_row_cpa(wrow, stride, ring, tab, gs, t_inv, xs, nb, lane);
                } else if (ARM == TROWS_TILE) {
                    acc = trows_row_tile(wrow, tab, gs, t_inv, xs, nb, lane);
                } else if (ARM == TROWS_ILP2) {
                    acc = trows_row_ilp2(wrow, tab, gs, t_inv, xs, nb, lane);
                } else {
                    acc = trows_row_uniform<ARM>(wrow, tab, gs, t_inv, xs, nb, lane);
                }
                if (lane == 0u) {
                    const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
                    y[r] = f2h(v);
                    if (yf != nullptr) yf[r] = v;
                }
            }
        }
        if (stamps != nullptr && lane == 0u) stamps[((size_t)s * g + c) * (nw + 1u) + warp] = (u32)(trows_clock() - t0);
        __syncthreads();
        if (stamps != nullptr && threadIdx.x == 0u) stamps[((size_t)s * g + c) * (nw + 1u) + nw] = (u32)(trows_clock() - t0);
    }
    if (stamps != nullptr && threadIdx.x == 0u) {
        ends[4u * blockIdx.x + 2u] = trows_clock();
        ends[4u * blockIdx.x + 3u] = pers_now();
    }
}

// MINB: CTAs a SM the register allocation must allow (2 for the occupancy
// arm, which runs V2 at two CTAs a SM on the sites of d_in 4,096 or less).
#define TROWS_KERNEL(sym, ARM_, MINB)                                                                             \
    extern "C" __global__ void __launch_bounds__(512, MINB) sym(                                                        \
        const unsigned long long* __restrict__ sites, u32 nsites, const u32* __restrict__ rows,                   \
        const unsigned char* __restrict__ prefixes, const unsigned short* __restrict__ branches,                  \
        const unsigned char* __restrict__ suffixes, const float* __restrict__ invnorm, const float* __restrict__ x, \
        const float* __restrict__ x2, u32* __restrict__ stamps, unsigned long long* __restrict__ ends)            \
    {                                                                                                             \
        trows_body<ARM_>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x, x2, stamps, ends);        \
    }

TROWS_KERNEL(tv_trows_v2, TROWS_V2, 1)
TROWS_KERNEL(tv_trows_v3, TROWS_V3, 1)
TROWS_KERNEL(tv_trows_ilp2, TROWS_ILP2, 1)
TROWS_KERNEL(tv_trows_v2u, TROWS_V2U, 1)
TROWS_KERNEL(tv_trows_l1w, TROWS_L1W, 1)
TROWS_KERNEL(tv_trows_bcast, TROWS_BCAST, 1)
TROWS_KERNEL(tv_trows_xbcast, TROWS_XBCAST, 1)
TROWS_KERNEL(tv_trows_v2_o2, TROWS_V2, 2)
TROWS_KERNEL(tv_trows_k2, TROWS_K2, 1)
TROWS_KERNEL(tv_trows_nc, TROWS_NC, 1)
TROWS_KERNEL(tv_trows_nc256, TROWS_NC256, 1)
TROWS_KERNEL(tv_trows_cpa, TROWS_CPA, 1)
TROWS_KERNEL(tv_trows_tile, TROWS_TILE, 1)
TROWS_KERNEL(tv_trows_bal, TROWS_BAL, 1)
