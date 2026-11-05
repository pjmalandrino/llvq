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

// Site words, 8 x u64, laid out by pers.rs::trows_site_words:
//   [0] words  [1] gscale  [2] rscale  [3] tail (binary16)  [4] y (binary16)
//   [5] d_out | nblocks << 32  [6] tail_w | row_stride_u32 << 32
//   [7] the f32 value of each row before f2h, or 0 (the card's launches)
#define TROWS_SITE_WORDS 8u

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
                                           u32* __restrict__ stamps, unsigned long long* __restrict__ ends)
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
        pers_stage_f32<24u, false>(x, xs, nb, tw);  // barriers on both sides
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
        } else {
            for (u32 r = lo + warp; r < hi; r += nw) {
                const u32* wrow = words + (size_t)(ARM == TROWS_L1W ? 0u : r) * stride;
                float acc;
                if (ARM == TROWS_V2) {
                    acc = pers_tetra_row<24u, false, true>(wrow, tab, gs, t_inv, xs, nb, lane);
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
        u32* __restrict__ stamps, unsigned long long* __restrict__ ends)                                          \
    {                                                                                                             \
        trows_body<ARM_>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x, stamps, ends);            \
    }

TROWS_KERNEL(tv_trows_v2, TROWS_V2, 1)
TROWS_KERNEL(tv_trows_v3, TROWS_V3, 1)
TROWS_KERNEL(tv_trows_ilp2, TROWS_ILP2, 1)
TROWS_KERNEL(tv_trows_v2u, TROWS_V2U, 1)
TROWS_KERNEL(tv_trows_l1w, TROWS_L1W, 1)
TROWS_KERNEL(tv_trows_bcast, TROWS_BCAST, 1)
TROWS_KERNEL(tv_trows_xbcast, TROWS_XBCAST, 1)
TROWS_KERNEL(tv_trows_v2_o2, TROWS_V2, 2)
