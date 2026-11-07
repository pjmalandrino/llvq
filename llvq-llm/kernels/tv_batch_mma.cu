// tv_batch_mma.cu: the batched projections. Each weight block is decoded ONCE
// into shared memory and multiplied for up to 32 activation rows on the tensor
// cores (mma.sync m16n8k16, f16 in, f32 accumulate).
//
// ## Why it exists
//
// The served kernels decode a block and spend 24 FMAs on it per activation
// row, on the ordinary cores. At batch 1 that is free next to the weight read.
// At B rows the read is shared and the arithmetic is not: the multi-row
// prefill kernel (`tv_tetra48_rows_h`) pays B x 24 FMAs and B x 24 shared loads
// a block, so it gains x1.2 to x1.35 over decoding row by row. Here the decode
// is paid once per block and the B x 24 multiply-adds go to the tensor cores.
//
// ## The arithmetic, and what it changes
//
// A Tetra block is `(sum_j y_j x_j) * g * inv` with `y_j` the lattice
// coordinates, small integers (|y_j| <= 10), so **the weights enter the MMA
// exactly**: an f16 holds every integer up to 2048. The scales stay outside
// the MMA, applied per block in f32 as `(d * g) * inv`, the association of
// `tetra48_dot`. The rotated activation is f32; it enters as two f16 halves,
// `x = hi + lo` (hi = f16(x), lo = f16(x - hi)), two MMAs a step, which keeps
// about 22 bits of it. An int4 record is `sum_c (s q_c + b) x_c = s sum q x +
// b sum x` per group: the nibbles enter exactly, the activation is already
// f16 (exact), one MMA a step, and `sum x` is formed once per unit while
// staging.
//
// What differs from the one-row kernels is the ORDER of the sums: the tensor
// core adds its products in its own order, and the K dimension is split across
// CTAs and folded in split order. So this path is not bit for bit with batch
// 1. It is deterministic (no atomics on values, a fixed fold order), and its
// licence is the quality gate of CLAUDE.md (T8 band), not identity.
//
// ## Layout
//
// A CTA owns 128 output rows (4 warps x 32) and one slice of K (`slice`
// units, gridDim.y slices). A unit is one 24-coordinate Tetra block or 32
// int4 columns; in shared memory it always occupies 32 columns (Tetra pads 8
// zeros), two k16 steps. Per unit, lane r of a warp decodes row r of its warp
// into the warp's A tile (natural coordinate order, 32 halves a row) and its
// two scales; the warp then runs 2 m-tiles x NT n-tiles x 2 k-steps of MMA.
// The activation is staged per CTA, TB units at a time, as f16 pairs with a
// 4-word pad a row so the B-fragment loads hit 32 distinct banks.
//
// Split K: each CTA writes its partial sums to `part[s][n][row]`, draws a
// ticket on its row group, and the last CTA folds the slices in order 0..S-1,
// applies the row scale and the f16 tail (Tetra), and stores f16. One slice
// stores directly. The counter resets itself (planes_occ.cu's occ_sk).
//
// Host: `llvq-llm/tests/host_batch_mma.cpp` executes these kernels through
// `host_cta.h`, with the MMA modelled lane by lane from the PTX fragment
// layout, against a double-precision reference.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_TETRA48_CUH
#include "../../llvq-cuda/kernels/llvq_tetra48.cuh"
#endif

#define BM_ROWS 128u   // output rows a CTA owns
#define BM_WARPS 4u    // 32 rows a warp, two m16 tiles
#define BM_UNIT 32u    // shared columns a unit occupies
#define BM_AS 20u      // u32 words a decoded row occupies: 16 pairs + 4 pad

#ifdef LLVQ_HOST_BUILD
// The warp collectives, provided by the host probe (a barrier per call).
void bm_mma_host(float* d, const u32* a, u32 b0, u32 b1);
#define BM_WARP_SYNC() __syncthreads()
#else
#define BM_WARP_SYNC() __syncwarp()
#endif

// Two floats to one f16x2 word, round to nearest; `lo` in the low half.
__device__ __forceinline__ u32 bm_pack(float lo, float hi)
{
#ifdef LLVQ_HOST_BUILD
    return (u32)f2h(lo) | ((u32)f2h(hi) << 16);
#else
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// d += A B for one m16n8k16 tile. `a` is the A fragment (row major), `b0 b1`
// the B fragment (column major), both f16 pairs; `d` f32.
__device__ __forceinline__ void bm_mma(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1)
{
#ifdef LLVQ_HOST_BUILD
    bm_mma_host(d, a, b0, b1);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Two nibbles to an f16x2 pair of the integers 0..15, exactly: 0x6400 | q is
// the half 1024 + q, and one f16x2 subtraction of 1024 leaves q (Marlin's
// trick, exact since every value is an integer below 2048).
__device__ __forceinline__ u32 bm_nib2(u32 q0, u32 q1)
{
#ifdef LLVQ_HOST_BUILD
    return bm_pack((float)q0, (float)q1);
#else
    u32 v = (q0 | 0x6400u) | ((q1 | 0x6400u) << 16), r;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(0x64006400u));
    return r;
#endif
}

// The tensor-core part of one unit: A from the warp's decoded tile, B from the
// staged activation at unit `b`, PASSES halves of it (1: hi only, 2: hi + lo).
// `d` is zeroed by the caller; the result is the unit's dot, unscaled.
template <unsigned NT, unsigned PASSES>
__device__ __forceinline__ void bm_unit(float (&d)[2][NT][4], const u32* aw, const u32* xh,
                                        const u32* xl, u32 xs, u32 b, u32 gid, u32 tig)
{
#pragma unroll
    for (u32 ks = 0; ks < 2u; ++ks) {
        const u32 xo = b * 16u + ks * 8u + tig;
        u32 bh[NT][2], bl[NT][2];
#pragma unroll
        for (u32 nt = 0; nt < NT; ++nt) {
            const u32 n = nt * 8u + gid;
            bh[nt][0] = xh[n * xs + xo];
            bh[nt][1] = xh[n * xs + xo + 4u];
            if (PASSES == 2u) {
                bl[nt][0] = xl[n * xs + xo];
                bl[nt][1] = xl[n * xs + xo + 4u];
            }
        }
#pragma unroll
        for (u32 mt = 0; mt < 2u; ++mt) {
            const u32* a0 = aw + (mt * 16u + gid) * BM_AS + ks * 8u + tig;
            u32 a[4];
            a[0] = a0[0];
            a[1] = a0[8u * BM_AS];
            a[2] = a0[4];
            a[3] = a0[8u * BM_AS + 4u];
#pragma unroll
            for (u32 nt = 0; nt < NT; ++nt) {
                bm_mma(d[mt][nt], a, bh[nt][0], bh[nt][1]);
                if (PASSES == 2u) bm_mma(d[mt][nt], a, bl[nt][0], bl[nt][1]);
            }
        }
    }
}

// Store the CTA's accumulators: directly when K is not split, else as
// partials, the ticket, and the last CTA's ordered fold. `fin(row, n, sum)`
// turns a row's sum into the stored f16 (row scale and tail for Tetra).
template <unsigned NT, class Fin>
__device__ __forceinline__ void bm_store(const float (&acc)[2][NT][4], u32 row0, u32 gid, u32 tig,
                                         u32 n_rows, u32 d_out, float* __restrict__ part,
                                         u32* __restrict__ ticket, unsigned short* __restrict__ y,
                                         Fin fin)
{
    const u32 S = gridDim.y;
    if (S == 1u) {
#pragma unroll
        for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
            for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
                for (u32 e = 0; e < 4u; ++e) {
                    const u32 row = row0 + mt * 16u + gid + (e >> 1) * 8u;
                    const u32 n = nt * 8u + tig * 2u + (e & 1u);
                    if (n < n_rows) y[(u64)n * d_out + row] = fin(row, n, acc[mt][nt][e]);
                }
        return;
    }
    const u32 s = blockIdx.y;
#pragma unroll
    for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
        for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                const u32 row = row0 + mt * 16u + gid + (e >> 1) * 8u;
                const u32 n = nt * 8u + tig * 2u + (e & 1u);
                if (n < n_rows) part[((u64)s * n_rows + n) * d_out + row] = acc[mt][nt][e];
            }
    __threadfence();
    __syncthreads();
    const u32 t = threadIdx.x;
    unsigned tk = 0u;
    if (t == 0u) tk = atomicAdd(ticket + blockIdx.x, 1u);
    const int last = __syncthreads_or(t == 0u && tk == S - 1u);
    if (!last) return;
    __threadfence();
    const u32 row = blockIdx.x * BM_ROWS + t;  // 128 threads, one row each
    for (u32 n = 0; n < n_rows; ++n) {
        float sum = 0.0f;
        for (u32 k = 0; k < S; ++k) sum += ((const volatile float*)part)[((u64)k * n_rows + n) * d_out + row];
        y[(u64)n * d_out + row] = fin(row, n, sum);
    }
    if (t == 0u) ticket[blockIdx.x] = 0u;
}

// Shared memory, in u32 words: the staged activation (hi, then lo for Tetra),
// the four warps' A tiles, their scales, and the int4 unit sums.
#define BM_TB(nt) (16u / (nt))
#define BM_XS(nt) (BM_TB(nt) * 16u + 4u)

template <unsigned NT>
__device__ __forceinline__ void bm_tetra(const u32* __restrict__ words, u32 row_stride_u32,
                                         const u32* __restrict__ trows,
                                         const unsigned char* __restrict__ prefixes,
                                         const unsigned short* __restrict__ branches,
                                         const unsigned char* __restrict__ suffixes,
                                         const float* __restrict__ gscale,
                                         const float* __restrict__ invnorm,
                                         const float* __restrict__ rscale,
                                         const unsigned short* __restrict__ tail,
                                         const float* __restrict__ x, unsigned short* __restrict__ y,
                                         u32 nblocks, u32 tail_w, u32 n_rows, u32 d_in, u32 d_out,
                                         u32 slice, float* __restrict__ part, u32* __restrict__ ticket)
{
    constexpr u32 N = 8u * NT, TB = BM_TB(NT), XS = BM_XS(NT);
    extern __shared__ u32 bm_smem[];
    u32* xh = bm_smem;
    u32* xl = xh + N * XS;
    u32* as = xl + N * XS;
    float* sc = (float*)(as + BM_WARPS * 32u * BM_AS);

    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 row0 = blockIdx.x * BM_ROWS + warp * 32u;
    const u32* wrow = words + (u64)(row0 + lane) * row_stride_u32;
    u32* aw = as + warp * 32u * BM_AS;
    float* sw = sc + warp * 64u;
    F1rTables tab = { trows, prefixes, branches, suffixes };

    float acc[2][NT][4];
#pragma unroll
    for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
        for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) acc[mt][nt][e] = 0.0f;

    const u32 k0 = blockIdx.y * slice;
    const u32 k1 = k0 + slice < nblocks ? k0 + slice : nblocks;
    for (u32 c0 = k0; c0 < k1; c0 += TB) {
        const u32 cn = k1 - c0 < TB ? k1 - c0 : TB;
        __syncthreads();
        for (u32 i = t; i < N * TB * 16u; i += blockDim.x) {
            const u32 n = i / (TB * 16u), b = (i >> 4) % TB, p = i & 15u;
            float v0 = 0.0f, v1 = 0.0f;
            if (n < n_rows && b < cn && p < 12u) {
                const float* xr = x + (u64)n * d_in + (c0 + b) * LLVQ_DIM + 2u * p;
                v0 = xr[0];
                v1 = xr[1];
            }
            const u32 h = bm_pack(v0, v1);
            xh[n * XS + b * 16u + p] = h;
            xl[n * XS + b * 16u + p] = bm_pack(v0 - h2f((unsigned short)(h & 0xffffu)),
                                               v1 - h2f((unsigned short)(h >> 16)));
        }
        __syncthreads();
        for (u32 b = 0; b < cn; ++b) {
            u32 lo, hi16;
            f1r_load(wrow, c0 + b, lo, hi16);
            u32 q[6];
            f1r_v3_quads(lo, hi16, tab, q);
            const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
            const u32 g = (hi16 >> 15) & 1u;
            float yn[24];
#pragma unroll
            for (u32 i = 0; i < 6u; ++i)
#pragma unroll
                for (u32 j = 0; j < 4u; ++j) yn[TETRA48_ORDER[4u * i + j]] = f1r_v3_float(q[i], j);
            u32* ar = aw + lane * BM_AS;
#pragma unroll
            for (u32 p = 0; p < 12u; ++p) ar[p] = bm_pack(yn[2u * p], yn[2u * p + 1u]);
#pragma unroll
            for (u32 p = 12u; p < 16u; ++p) ar[p] = 0u;
            sw[2u * lane] = gscale[g];
            sw[2u * lane + 1u] = invnorm[m];
            BM_WARP_SYNC();

            float d[2][NT][4];
#pragma unroll
            for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
                for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (u32 e = 0; e < 4u; ++e) d[mt][nt][e] = 0.0f;
            bm_unit<NT, 2u>(d, aw, xh, xl, XS, b, gid, tig);
#pragma unroll
            for (u32 mt = 0; mt < 2u; ++mt) {
                const u32 ra = mt * 16u + gid, rb = ra + 8u;
                const float ga = sw[2u * ra], ia = sw[2u * ra + 1u];
                const float gb = sw[2u * rb], ib = sw[2u * rb + 1u];
#pragma unroll
                for (u32 nt = 0; nt < NT; ++nt) {
                    // `(d * g) * inv`, the association of tetra48_dot.
                    acc[mt][nt][0] += d[mt][nt][0] * ga * ia;
                    acc[mt][nt][1] += d[mt][nt][1] * ga * ia;
                    acc[mt][nt][2] += d[mt][nt][2] * gb * ib;
                    acc[mt][nt][3] += d[mt][nt][3] * gb * ib;
                }
            }
            BM_WARP_SYNC();
        }
    }
    bm_store<NT>(acc, row0, gid, tig, n_rows, d_out, part, ticket, y,
                 [&](u32 row, u32 n, float sum) {
                     const float tv = tail_dot_h(tail, x + (u64)n * d_in + nblocks * LLVQ_DIM, row, tail_w);
                     return f2h(sum * rscale[row] + tv);
                 });
}

// An int4 matrix, row major, eight nibbles a u32 word, column c at nibble c%8
// of word c/8; one f16 scale and bias per row and group of `1 << gshift`
// columns (128 for the int4 records, 64 for the embedding table as lm_head).
template <unsigned NT>
__device__ __forceinline__ void bm_q4(const u32* __restrict__ wq,
                                      const unsigned short* __restrict__ scales,
                                      const unsigned short* __restrict__ biases,
                                      const unsigned short* __restrict__ x,
                                      unsigned short* __restrict__ y, u32 d_in, u32 d_out, u32 gpr,
                                      u32 gshift, u32 n_rows, u32 slice, float* __restrict__ part,
                                      u32* __restrict__ ticket)
{
    constexpr u32 N = 8u * NT, TB = BM_TB(NT), XS = BM_XS(NT);
    extern __shared__ u32 bm_smem[];
    u32* xh = bm_smem;
    float* xsum = (float*)(xh + N * XS);  // [N][TB], where Tetra keeps its lo half
    u32* as = xh + 2u * N * XS;
    float* sc = (float*)(as + BM_WARPS * 32u * BM_AS);

    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 row0 = blockIdx.x * BM_ROWS + warp * 32u;
    const u32 myrow = row0 + lane;
    const u32 nunits = d_in / BM_UNIT;
    const uint4* wrow = (const uint4*)(wq + (u64)myrow * (d_in >> 3));
    u32* aw = as + warp * 32u * BM_AS;
    float* sw = sc + warp * 64u;

    float acc[2][NT][4];
#pragma unroll
    for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
        for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) acc[mt][nt][e] = 0.0f;

    const u32 k0 = blockIdx.y * slice;
    const u32 k1 = k0 + slice < nunits ? k0 + slice : nunits;
    for (u32 c0 = k0; c0 < k1; c0 += TB) {
        const u32 cn = k1 - c0 < TB ? k1 - c0 : TB;
        __syncthreads();
        for (u32 i = t; i < N * TB * 16u; i += blockDim.x) {
            const u32 n = i / (TB * 16u), b = (i >> 4) % TB, p = i & 15u;
            u32 v = 0u;
            if (n < n_rows && b < cn) {
                const u32* xr = (const u32*)(x + (u64)n * d_in + (c0 + b) * BM_UNIT);
                v = xr[p];
            }
            xh[n * XS + b * 16u + p] = v;
        }
        __syncthreads();
        for (u32 i = t; i < N * TB; i += blockDim.x) {
            const u32 n = i / TB, b = i % TB;
            float s = 0.0f;
            for (u32 p = 0; p < 16u; ++p) {
                const u32 v = xh[n * XS + b * 16u + p];
                s += h2f((unsigned short)(v & 0xffffu));
                s += h2f((unsigned short)(v >> 16));
            }
            xsum[n * TB + b] = s;
        }
        __syncthreads();
        for (u32 b = 0; b < cn; ++b) {
            const u32 u = c0 + b;
            const uint4 w = wrow[u];
            const u32 ws[4] = { w.x, w.y, w.z, w.w };
            u32* ar = aw + lane * BM_AS;
#pragma unroll
            for (u32 k = 0; k < 4u; ++k)
#pragma unroll
                for (u32 j = 0; j < 4u; ++j)
                    ar[4u * k + j] = bm_nib2((ws[k] >> (8u * j)) & 0xfu, (ws[k] >> (8u * j + 4u)) & 0xfu);
            const u64 gi = (u64)myrow * gpr + ((u * BM_UNIT) >> gshift);
            sw[2u * lane] = h2f(scales[gi]);
            sw[2u * lane + 1u] = h2f(biases[gi]);
            BM_WARP_SYNC();

            float d[2][NT][4];
#pragma unroll
            for (u32 mt = 0; mt < 2u; ++mt)
#pragma unroll
                for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (u32 e = 0; e < 4u; ++e) d[mt][nt][e] = 0.0f;
            bm_unit<NT, 1u>(d, aw, xh, xh, XS, b, gid, tig);
#pragma unroll
            for (u32 mt = 0; mt < 2u; ++mt) {
                const u32 ra = mt * 16u + gid, rb = ra + 8u;
                const float sa = sw[2u * ra], ba = sw[2u * ra + 1u];
                const float sb = sw[2u * rb], bb = sw[2u * rb + 1u];
#pragma unroll
                for (u32 nt = 0; nt < NT; ++nt) {
                    const u32 n0 = nt * 8u + tig * 2u;
                    const float x0 = xsum[n0 * TB + b], x1 = xsum[(n0 + 1u) * TB + b];
                    acc[mt][nt][0] += d[mt][nt][0] * sa + ba * x0;
                    acc[mt][nt][1] += d[mt][nt][1] * sa + ba * x1;
                    acc[mt][nt][2] += d[mt][nt][2] * sb + bb * x0;
                    acc[mt][nt][3] += d[mt][nt][3] * sb + bb * x1;
                }
            }
            BM_WARP_SYNC();
        }
    }
    bm_store<NT>(acc, row0, gid, tig, n_rows, d_out, part, ticket, y,
                 [](u32, u32, float sum) { return f2h(sum); });
}

#define BM_TETRA_ENTRY(NAME, NT)                                                                    \
    extern "C" __global__ void NAME(const u32* __restrict__ words, u32 row_stride_u32,               \
                                    const u32* __restrict__ trows,                                   \
                                    const unsigned char* __restrict__ prefixes,                      \
                                    const unsigned short* __restrict__ branches,                     \
                                    const unsigned char* __restrict__ suffixes,                      \
                                    const float* __restrict__ gscale,                                \
                                    const float* __restrict__ invnorm,                               \
                                    const float* __restrict__ rscale,                                \
                                    const unsigned short* __restrict__ tail,                         \
                                    const float* __restrict__ x, unsigned short* __restrict__ y,     \
                                    u32 nblocks, u32 tail_w, u32 n_rows, u32 d_in, u32 d_out,        \
                                    u32 slice, float* __restrict__ part, u32* __restrict__ ticket)   \
    {                                                                                                \
        bm_tetra<NT>(words, row_stride_u32, trows, prefixes, branches, suffixes, gscale, invnorm,    \
                     rscale, tail, x, y, nblocks, tail_w, n_rows, d_in, d_out, slice, part, ticket); \
    }

#define BM_Q4_ENTRY(NAME, NT)                                                                       \
    extern "C" __global__ void NAME(const u32* __restrict__ wq,                                      \
                                    const unsigned short* __restrict__ scales,                       \
                                    const unsigned short* __restrict__ biases,                       \
                                    const unsigned short* __restrict__ x,                            \
                                    unsigned short* __restrict__ y, u32 d_in, u32 d_out, u32 gpr,    \
                                    u32 gshift, u32 n_rows, u32 slice, float* __restrict__ part,     \
                                    u32* __restrict__ ticket)                                        \
    {                                                                                                \
        bm_q4<NT>(wq, scales, biases, x, y, d_in, d_out, gpr, gshift, n_rows, slice, part, ticket);  \
    }

BM_TETRA_ENTRY(tv_tetra48_mma_n8, 1u)
BM_TETRA_ENTRY(tv_tetra48_mma_n16, 2u)
BM_TETRA_ENTRY(tv_tetra48_mma_n32, 4u)
BM_Q4_ENTRY(tv_q4_mma_n8, 1u)
BM_Q4_ENTRY(tv_q4_mma_n16, 2u)
BM_Q4_ENTRY(tv_q4_mma_n32, 4u)
