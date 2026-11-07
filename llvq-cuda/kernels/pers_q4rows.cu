// tv_q4rows_*: 4E's int4 rows alone, K11 of plan-300's step 1 (2026-11-07,
// proofs/preregistration-step1-micro-2026-11-07.md).
//
// Why: 4E's int4 rows (v and o of every layer, the downs of layers 12-23: 84
// records, 0.409 GB on the 4B) stream about 587 GB/s inside 4E (pers-4e-v3),
// where FlashInfer streams 700 on the same card. The served `tv_q4_hh` has
// never been timed alone, nor has 4E's row at 4E's geometry. This kernel does
// both, and tries two loads that keep each lane's words and their order:
//   Q4      pers_q4_row<true>, 4E's row (word wi + 32 in flight). Owes the bits.
//   NC      the same with the words through `ld.global.nc.L1::no_allocate.
//           L2::256B` (trows_ldw<2>, pers_trows.cu). Owes the bits.
//   TILE    the words relaid once (tv_q4rows_tile_relayout) so one 128-bit
//           load gives lane l its words 128t + l + 32s, s = 0..3; the words
//           past the last full tile of 128 stay in served order. Same bytes,
//           same lane to word map, same fold order. Owes the bits.
//   READ    diagnostic: the same words, scales and biases read by 128-bit
//           loads (scales and biases by 16-bit loads) and folded by xor, no
//           arithmetic, no row owed. The read ceiling at this geometry.
//
// Geometry: 4E's, one CTA of 512 threads a SM; each site's rows split over
// CTAs as [d c / g, d (c + 1) / g) and over warps statically; the activation,
// binary16, widened into f32 staging as tv_q4_hh does (exact), restaged at
// every site between two CTA barriers. Stamps as in pers_trows.cu.
//
// Composition: the row-loop module (pers.rs::trows_unit_text), after
// pers_trows.cu, whose loads and clock it uses.

#define Q4ROWS_Q4 0u
#define Q4ROWS_NC 1u
#define Q4ROWS_TILE 2u
#define Q4ROWS_READ 3u

// Site words, 6 x u64, laid out by pers.rs::q4rows_site_words:
//   [0] words (served order, or TILE's layout)  [1] scales  [2] biases
//   [3] y (binary16)  [4] d_out | d_in << 32  [5] the f32 value of each row
//   before f2h, or 0 (the card's launches)
#define Q4ROWS_SITE_WORDS 6u

// NC: pers_q4_row<true> with the words through path LD of trows_ldw.
template <u32 LD>
__device__ __forceinline__ float q4rows_row_ld(const u32* wq, const unsigned short* scales,
                                               const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                               u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    u32 p = 0u, sb = 0u, bb = 0u;
    if (lane < nwords) {
        p = trows_ldw<LD>(wq + w0 + lane);
        sb = scales[s0 + (lane >> 4)];
        bb = biases[s0 + (lane >> 4)];
    }
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p2 = 0u, sb2 = 0u, bb2 = 0u;
        if (wi + 32u < nwords) {
            p2 = trows_ldw<LD>(wq + w0 + wi + 32u);
            sb2 = scales[s0 + ((wi + 32u) >> 4)];
            bb2 = biases[s0 + ((wi + 32u) >> 4)];
        }
        const u32 c = wi << 3;
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        p = p2;
        sb = sb2;
        bb = bb2;
    }
    return warp_sum(acc);
}

// TILE's layout of one row of nwords words: word wi < 128 nt (nt = nwords /
// 128) goes to u32 128 t + 4 l + s with t = wi / 128, s = (wi % 128) / 32,
// l = wi % 32; the words after stay where they are.
__device__ __forceinline__ u32 q4rows_tile_index(u32 wi, u32 nwords)
{
    const u32 nt = nwords >> 7;
    if (wi >= (nt << 7)) return wi;
    const u32 t = wi >> 7, k = wi & 127u;
    return (t << 7) + 4u * (k & 31u) + (k >> 5);
}

extern "C" __global__ void tv_q4rows_tile_relayout(const u32* __restrict__ src, u32* __restrict__ dst, u32 d_out,
                                                   u32 nwords)
{
    const unsigned long long n = (unsigned long long)d_out * nwords;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const u32 r = (u32)(i / nwords), wi = (u32)(i - (unsigned long long)r * nwords);
        dst[(size_t)r * nwords + q4rows_tile_index(wi, nwords)] = src[i];
    }
}

// TILE: lane l folds its words wi = l, l + 32, ... in increasing order, each
// with q4_staged_dot's eight products in increasing k, as Q4.
__device__ __forceinline__ float q4rows_row_tile(const u32* wq, const unsigned short* scales,
                                                 const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                                 u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3, nt = nwords >> 7;
    const u32* wrow = wq + (size_t)row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    uint4 P{0u, 0u, 0u, 0u};
    if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * lane);
    for (u32 t = 0; t < nt; ++t) {
        uint4 P2{0u, 0u, 0u, 0u};
        if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + ((t + 1u) << 7) + 4u * lane);
        const u32 w[4] = {P.x, P.y, P.z, P.w};
#pragma unroll
        for (u32 s = 0; s < 4u; ++s) {
            const u32 c = ((t << 7) + 32u * s + lane) << 3;
            const u32 g = s0 + (c >> 7);
            const u32 sb = scales[g], bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k)
                acc = __fmaf_rn(pers_q4_deq((w[s] >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
        P = P2;
    }
    for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {
        const u32 p = wrow[wi];
        const u32 c = wi << 3;
        const u32 g = s0 + (c >> 7);
        const u32 sb = scales[g], bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    return warp_sum(acc);
}

// READ: the row's words as uint4, then its scales and biases, folded by xor.
__device__ __forceinline__ u32 q4rows_row_read(const u32* wq, const unsigned short* scales,
                                               const unsigned short* biases, u32 row, u32 d_in, u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3;
    const uint4* p = reinterpret_cast<const uint4*>(wq + (size_t)row * nwords);
    u32 x = 0u;
    for (u32 k = lane; k < (nwords >> 2); k += 32u) {
        const uint4 v = p[k];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (u32 g = lane; g < gpr; g += 32u) x ^= (u32)scales[row * gpr + g] ^ ((u32)biases[row * gpr + g] << 16);
    for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

template <u32 ARM>
__device__ __forceinline__ void q4rows_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                            const unsigned short* __restrict__ xh, u32* __restrict__ stamps,
                                            unsigned long long* __restrict__ ends)
{
    if (stamps != nullptr && threadIdx.x == 0u) {
        ends[4u * blockIdx.x] = trows_clock();
        ends[4u * blockIdx.x + 1u] = pers_now();
    }
    float* xs = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    for (u32 s = 0; s < nsites; ++s) {
        const unsigned long long* d = sites + (size_t)s * Q4ROWS_SITE_WORDS;
        const u32* wq = reinterpret_cast<const u32*>(d[0]);
        const unsigned short* sc = reinterpret_cast<const unsigned short*>(d[1]);
        const unsigned short* bi = reinterpret_cast<const unsigned short*>(d[2]);
        unsigned short* y = reinterpret_cast<unsigned short*>(d[3]);
        const u32 d_out = (u32)(d[4] & 0xffffffffull), d_in = (u32)(d[4] >> 32);
        float* yf = reinterpret_cast<float*>(d[5]);
        const u32 gpr = d_in >> 7;
        __syncthreads();  // no warp still reads the previous staging
        if (ARM != Q4ROWS_READ)
            for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(__ldcg(xh + i));
        __syncthreads();
        const u32 lo = (u32)((unsigned long long)d_out * c / g), hi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        const unsigned long long t0 = trows_clock();
        for (u32 r = lo + warp; r < hi; r += nw) {
            if (ARM == Q4ROWS_READ) {
                const u32 x = q4rows_row_read(wq, sc, bi, r, d_in, gpr, lane);
                if (lane == 0u) y[r] = (unsigned short)(x & 0x3bffu);
                continue;
            }
            float acc;
            if (ARM == Q4ROWS_Q4) {
                acc = pers_q4_row<true>(wq, sc, bi, xs, r, d_in, gpr, lane);
            } else if (ARM == Q4ROWS_NC) {
                acc = q4rows_row_ld<2u>(wq, sc, bi, xs, r, d_in, gpr, lane);
            } else {
                acc = q4rows_row_tile(wq, sc, bi, xs, r, d_in, gpr, lane);
            }
            if (lane == 0u) {
                y[r] = f2h(acc);
                if (yf != nullptr) yf[r] = acc;
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

#define Q4ROWS_KERNEL(sym, ARM_)                                                                                  \
    extern "C" __global__ void __launch_bounds__(512, 1) sym(const unsigned long long* __restrict__ sites,         \
                                                             u32 nsites, const unsigned short* __restrict__ xh,   \
                                                             u32* __restrict__ stamps,                            \
                                                             unsigned long long* __restrict__ ends)               \
    {                                                                                                             \
        q4rows_body<ARM_>(sites, nsites, xh, stamps, ends);                                                       \
    }

Q4ROWS_KERNEL(tv_q4rows_q4, Q4ROWS_Q4)
Q4ROWS_KERNEL(tv_q4rows_nc, Q4ROWS_NC)
Q4ROWS_KERNEL(tv_q4rows_tile, Q4ROWS_TILE)
Q4ROWS_KERNEL(tv_q4rows_read, Q4ROWS_READ)
