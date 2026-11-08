// Instruction-count probe for docs/recherche-debit-maths.md: the served Tetra block against
// exact variants (trio staging, raw prmt, SoA words) and int8-activation variants (dp4a).
// One warp a row, one block a lane-iteration, `#pragma unroll 1`: the inner loop is one block.
// Compiled only to count instructions (ops/tetra-sass-probe.sh); nothing here is served or run.

__device__ __forceinline__ u32 prmt_raw(u32 a, u32 b, u32 s)
{
    u32 d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

// Served: natural staging, TETRA48_ORDER addresses, f32 FMA chain, fold.
extern "C" __global__ void row_f32(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                   const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                   const float* __restrict__ x, float* __restrict__ y)
{
    __shared__ float xs[24 * 128];
    for (u32 i = threadIdx.x; i < nb * 24u; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * stride;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo, hi16;
        f1r_load(w, j, lo, hi16);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        const float* xb = xs + j * 24u;
        float a = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i)
#pragma unroll
            for (u32 k = 0; k < 4u; ++k) a = __fmaf_rn(f1r_v3_float(q[i], k), xb[TETRA48_ORDER[4u * i + k]], a);
        const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc = __fmaf_rn(__fmul_rn(a, gscale[g]), invnorm[m], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[r] = acc;
}

// Served arithmetic, trio staging (activation pre-permuted, six float4): same bits.
extern "C" __global__ void row_f32_trio(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                        const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                        const float* __restrict__ x, float* __restrict__ y)
{
    __shared__ float4 xs4[7 * 128];
    float* xs = reinterpret_cast<float*>(xs4);
    for (u32 i = threadIdx.x; i < nb * 24u; i += blockDim.x) xs[(i / 24u) * 28u + i % 24u] = x[i];
    __syncthreads();
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * stride;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo, hi16;
        f1r_load(w, j, lo, hi16);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        const float4* x4 = xs4 + j * 7u;
        float a = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
            const float4 v = x4[i];
            a = __fmaf_rn(f1r_v3_float(q[i], 0u), v.x, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 2u), v.z, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 3u), v.w, a);
        }
        const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc = __fmaf_rn(__fmul_rn(a, gscale[g]), invnorm[m], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[r] = acc;
}

// int8 activation in trio order (six u32 a block), one f32 scale a block: the
// lattice values are exact int8 after XOR 0x80, so the dot is six dp4a.
template <bool SIGNPRMT>
__device__ __forceinline__ void quads_i8(u32 lo, u32 hi16, const F1rTables& t, u32 q[6])
{
    if (!SIGNPRMT) {
        f1r_v3_quads(lo, hi16, t, q);
        return;
    }
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    F1rV3Tab v = f1r_v3_tables(b.p);
    const u32 cs[3] = {b.c1, b.c2, b.c3}, rs[3] = {b.row1, b.row2, b.row3};
#pragma unroll
    for (u32 s = 0; s < 3u; ++s) {
#pragma unroll
        for (u32 h = 0; h < 2u; ++h) {
            const u32 sel = h ? rs[s] >> 16 : rs[s];
            const u32 c4 = h ? cs[s] >> 4 : cs[s] & 0xfu;
            const u32 m0 = __byte_perm(v.c0lo, v.c0hi, sel);
            const u32 m1 = __byte_perm(v.c1lo, v.c1hi, sel);
            const u32 k = prmt_raw(c4 * 0x10204080u, 0u, 0xba98u);
            q[2u * s + h] = (m0 & ~k) | (m1 & k);
        }
    }
}

template <bool SIGNPRMT>
__device__ __forceinline__ void row_i8_body(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                            const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                            const u32* __restrict__ xq, const float* __restrict__ xsc,
                                            float* __restrict__ y)
{
    __shared__ uint2 xs2[3 * 512];
    __shared__ float ss[512];
    u32* xs = reinterpret_cast<u32*>(xs2);
    for (u32 i = threadIdx.x; i < nb * 6u; i += blockDim.x) xs[i] = xq[i];
    for (u32 i = threadIdx.x; i < nb; i += blockDim.x) ss[i] = xsc[i];
    __syncthreads();
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * stride;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo, hi16;
        f1r_load(w, j, lo, hi16);
        u32 q[6];
        quads_i8<SIGNPRMT>(lo, hi16, tab, q);
        const uint2* x2 = xs2 + j * 3u;
        int dot = 0, n2 = 0;
#pragma unroll
        for (u32 i = 0; i < 3u; ++i) {
            const uint2 v = x2[i];
            const int s0 = (int)(q[2u * i] ^ 0x80808080u), s1 = (int)(q[2u * i + 1u] ^ 0x80808080u);
            dot = __dp4a(s0, (int)v.x, dot);
            dot = __dp4a(s1, (int)v.y, dot);
            n2 = __dp4a(s0, s0, n2);
            n2 = __dp4a(s1, s1, n2);
        }
        const u32 m = ((u32)n2 >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc = __fmaf_rn((float)dot * ss[j], gscale[g] * invnorm[m], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[r] = acc;
}

extern "C" __global__ void row_i8(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                  const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                  const u32* __restrict__ xq, const float* __restrict__ xsc, float* __restrict__ y)
{
    row_i8_body<false>(words, stride, nb, tab, gscale, invnorm, xq, xsc, y);
}

extern "C" __global__ void row_i8_sp(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                     const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                     const u32* __restrict__ xq, const float* __restrict__ xsc, float* __restrict__ y)
{
    row_i8_body<true>(words, stride, nb, tab, gscale, invnorm, xq, xsc, y);
}

// The decode alone, no dot (values XOR-folded so nothing is elided): the floor
// any dot arithmetic sits on.
extern "C" __global__ void row_dec(const u32* __restrict__ words, u32 stride, u32 nb, F1rTables tab,
                                   const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                   float* __restrict__ y)
{
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * stride;
    u32 h = 0u;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo, hi16;
        f1r_load(w, j, lo, hi16);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        h ^= q[0] ^ q[1] ^ q[2] ^ q[3] ^ q[4] ^ q[5];
        const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc += gscale[g] * invnorm[m];
    }
    acc = warp_sum(acc + (float)h);
    if (lane == 0u) y[r] = acc;
}
// Exact variants: same values, same FMA order. SoA words: lo plane (u32) and hi plane (u16).
__device__ __forceinline__ void quads_sp(u32 lo, u32 hi16, const F1rTables& t, u32 q[6])
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    F1rV3Tab v = f1r_v3_tables(b.p);
    const u32 cs[3] = {b.c1, b.c2, b.c3}, rs[3] = {b.row1, b.row2, b.row3};
#pragma unroll
    for (u32 s = 0; s < 3u; ++s)
#pragma unroll
        for (u32 h = 0; h < 2u; ++h) {
            const u32 sel = h ? rs[s] >> 16 : rs[s];
            const u32 c4 = h ? cs[s] >> 4 : cs[s] & 0xfu;
            const u32 m0 = __byte_perm(v.c0lo, v.c0hi, sel);
            const u32 m1 = __byte_perm(v.c1lo, v.c1hi, sel);
            const u32 k = prmt_raw(c4 * 0x10204080u, 0u, 0xba98u);
            q[2u * s + h] = (m0 & ~k) | (m1 & k);
        }
}

template <bool SP, bool SOA>
__device__ __forceinline__ void row_exact_body(const u32* __restrict__ words, const unsigned short* __restrict__ whi,
                                               u32 stride, u32 nb, F1rTables tab, const float* __restrict__ gscale,
                                               const float* __restrict__ invnorm, const float* __restrict__ x,
                                               float* __restrict__ y)
{
    __shared__ float4 xs4[7 * 128];
    float* xs = reinterpret_cast<float*>(xs4);
    for (u32 i = threadIdx.x; i < nb * 24u; i += blockDim.x) xs[(i / 24u) * 28u + i % 24u] = x[i];
    __syncthreads();
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * stride;
    const unsigned short* wh = whi + r * nb;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo, hi16;
        if (SOA) { lo = w[j]; hi16 = wh[j]; } else f1r_load(w, j, lo, hi16);
        u32 q[6];
        if (SP) quads_sp(lo, hi16, tab, q); else f1r_v3_quads(lo, hi16, tab, q);
        const float4* x4 = xs4 + j * 7u;
        float a = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
            const float4 v = x4[i];
            a = __fmaf_rn(f1r_v3_float(q[i], 0u), v.x, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 2u), v.z, a);
            a = __fmaf_rn(f1r_v3_float(q[i], 3u), v.w, a);
        }
        const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc = __fmaf_rn(__fmul_rn(a, gscale[g]), invnorm[m], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[r] = acc;
}
#define EX(name, SP, SOA) extern "C" __global__ void name(const u32* __restrict__ w, const unsigned short* __restrict__ wh, u32 st, u32 nb, F1rTables t, const float* __restrict__ gs, const float* __restrict__ iv, const float* __restrict__ x, float* __restrict__ y) { row_exact_body<SP, SOA>(w, wh, st, nb, t, gs, iv, x, y); }
EX(row_ex_sp, true, false)
EX(row_ex_soa, false, true)
EX(row_ex_all, true, true)

// int8 with SoA words and raw prmt: the best int8 form.
extern "C" __global__ void row_i8_best(const u32* __restrict__ words, const unsigned short* __restrict__ whi, u32 nb,
                                       F1rTables tab, const float* __restrict__ gscale, const float* __restrict__ invnorm,
                                       const u32* __restrict__ xq, const float* __restrict__ xsc, float* __restrict__ y)
{
    __shared__ uint2 xs2[3 * 512];
    __shared__ float ss[512];
    u32* xs = reinterpret_cast<u32*>(xs2);
    for (u32 i = threadIdx.x; i < nb * 6u; i += blockDim.x) xs[i] = xq[i];
    for (u32 i = threadIdx.x; i < nb; i += blockDim.x) ss[i] = xsc[i];
    __syncthreads();
    const u32 lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const u32* w = words + r * nb;
    const unsigned short* wh = whi + r * nb;
    float acc = 0.0f;
#pragma unroll 1
    for (u32 j = lane; j < nb; j += 32u) {
        const u32 lo = w[j], hi16 = wh[j];
        u32 q[6];
        quads_sp(lo, hi16, tab, q);
        const uint2* x2 = xs2 + j * 3u;
        int dot = 0, n2 = 0;
#pragma unroll
        for (u32 i = 0; i < 3u; ++i) {
            const uint2 v = x2[i];
            const int s0 = (int)(q[2u * i] ^ 0x80808080u), s1 = (int)(q[2u * i + 1u] ^ 0x80808080u);
            dot = __dp4a(s0, (int)v.x, dot);
            dot = __dp4a(s1, (int)v.y, dot);
            n2 = __dp4a(s0, s0, n2);
            n2 = __dp4a(s1, s1, n2);
        }
        const u32 m = ((u32)n2 >> 4) & (TETRA48_SHELLS - 1u);
        const u32 g = (hi16 >> 15) & 1u;
        acc = __fmaf_rn((float)dot * ss[j], gscale[g] * invnorm[m], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[r] = acc;
}
