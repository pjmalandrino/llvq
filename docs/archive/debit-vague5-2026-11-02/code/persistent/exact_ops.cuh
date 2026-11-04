// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, section 3.6).
//
// The candle operations a route-P segment absorbs, reproduced so that the
// output equals candle 0.9.2's kernels bit for bit, without any CUDA header
// (the runtime image has none, NVRTC 12.4 bundles none). Each one is checked
// on the card at load against candle's own kernel on real activations, and
// the segment falls back to the eager candle launch for any op that differs.
//
//   rmsnorm_f16   candle-kernels reduce.cu:187-228, launched by candle-nn
//                 ops.rs:545 with block_size 1024 when n_cols >= 1024, 32 below
//   usilu_f16     unary.cu:58-61: x / (__half(1) + hexp(-x)). NOT emulated:
//                 tabulated at load from candle's own kernel over all 65,536
//                 inputs, so equal by exhaustion.
//   bmul_f16      __hmul on two f16: the exact f32 product (22 significant bits)
//                 rounded once to f16 equals mul.f16 RN.
//   badd_f16      __hadd on two f16: f32 sum then RN to f16 equals add.f16 RN,
//                 because double rounding from p' = 24 to p = 11 is innocuous
//                 when p' >= 2p + 2 (Figueroa's bound for + and x).
//   fast_argmax   reduce.cu:476-523 with 1,024 threads, one block
//                 (cuda_backend/mod.rs:324-331): a 64-bit key reproduces its
//                 tie order.

// ---- f16 helpers (bits in unsigned short, as the served kernels do) -------
// h2f and f2h come from matvec.cu: cvt.f32.f16 and cvt.rn.f16.f32.

__device__ __forceinline__ unsigned short exact_hmul(unsigned short a, unsigned short b)
{
    return f2h(__fmul_rn(h2f(a), h2f(b)));
}

__device__ __forceinline__ unsigned short exact_hadd(unsigned short a, unsigned short b)
{
    return f2h(__fadd_rn(h2f(a), h2f(b)));
}

// SiLU by the table candle itself produced at load (65,536 x 2 B = 128 KiB,
// device memory, read through L2). Input NaN payloads map to whatever candle
// returned for them, by construction.
__device__ __forceinline__ unsigned short exact_silu(const unsigned short* __restrict__ silu_tab,
                                                     unsigned short g)
{
    return __ldg(silu_tab + g);
}

// gate/up epilogue of one row pair: h = silu(g) * u, then stored as f16.
__device__ __forceinline__ unsigned short exact_silu_mul(const unsigned short* silu_tab,
                                                         unsigned short g, unsigned short u)
{
    return exact_hmul(exact_silu(silu_tab, g), u);
}

// ---- RMSNorm for n_cols >= 1024 (candle's block_size 1024 tree) ------------
// Candle: virtual thread t (0..1023) sums cols t, t+1024, t+2048, ... in that
// order with `tmp += xi * xi` (NVVM contracts it under nvcc -O3 with fmad on;
// VARIANT selects the contracted or the split form, chosen at load by the
// parity check), then a xor butterfly 16, 8, 4, 2, 1 inside each virtual warp,
// then s_sum[w] for the 32 virtual warps, then the same butterfly over them.
// Here a CTA of PERS_THREADS threads computes the same 1,024 partials: real
// warp r handles virtual warps r, r + W, r + 2W, ... Every partial, every
// butterfly operand and every pairing is candle's, so the sum is candle's.
template <int VARIANT>
__device__ float exact_rms_scale_1024(const unsigned short* __restrict__ x_ldcg_src,
                                      u32 ncols, float eps, float* s_sum /* 32 floats in shared */)
{
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    for (u32 vw = warp; vw < 32u; vw += nwarps) {
        u32 t = vw * 32u + lane;                       // candle's threadIdx.x
        float tmp = 0.0f;
        for (u32 col = t; col < ncols; col += 1024u) {
            float xi = h2f(__ldcg(x_ldcg_src + col));
            tmp = VARIANT == 0 ? __fmaf_rn(xi, xi, tmp) : __fadd_rn(tmp, __fmul_rn(xi, xi));
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, mask);
        if (lane == 0u) s_sum[vw] = tmp;
    }
    __syncthreads();
    float tot = 0.0f;
    if (warp == 0u) {
        tot = s_sum[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, mask);
        if (lane == 0u) {
            float mean = __fdiv_rn(tot, (float)ncols);      // candle: tmp / ncols, IEEE division
            s_sum[0] = rsqrtf(__fadd_rn(mean, eps));       // candle: rsqrtf(mean + eps), rsqrt.approx.f32
        }
    }
    __syncthreads();
    float scale = s_sum[0];
    __syncthreads();                                    // s_sum is reused by the next call
    return scale;
}

// dst[col] = f16((scale * x[col]) * alpha[col]): two roundings, left to right,
// as candle's `scale * static_cast<float>(x) * a`.
__device__ __forceinline__ unsigned short exact_rms_apply(float scale, unsigned short x, unsigned short alpha)
{
    return f2h(__fmul_rn(__fmul_rn(scale, h2f(x)), h2f(alpha)));
}

// ---- argmax key -------------------------------------------------------------
// Candle scans i = t, t + 1024, ... per thread with a strict `>` (earliest
// index wins inside a thread), then reduces with `shr[tid + s] > shr[tid]`
// (the left operand keeps ties). Between tied threads the winner is the one
// whose index is clear at the lowest differing bit, i.e. the smallest
// bit-reversed thread index. Larger key = candle's winner:
//   key = ord16(v) << 32 | (1023 - bitrev10(i & 1023)) << 22 | (0x3FFFFF - (i >> 10))
// ord16 maps f16 bits to an order-preserving u32 after -0 -> +0. NaN is
// refused: candle's scan treats NaN by position, which no key reproduces, and
// a NaN logit means the step is already lost.
__device__ __forceinline__ unsigned long long exact_argmax_key(unsigned short h, u32 i, u32* nan_flag)
{
    if ((h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0u) atomicExch(nan_flag, 1u);
    if (h == 0x8000u) h = 0u;
    u32 ord = (h & 0x8000u) ? (u32)(~h & 0xffffu) : (u32)(h | 0x8000u);
    u32 t = i & 1023u;
    u32 br = __brev(t) >> 22;                    // 10-bit bit reversal
    unsigned long long key = ((unsigned long long)ord << 32)
                           | ((unsigned long long)(1023u - br) << 22)
                           | (unsigned long long)(0x3FFFFFu - (i >> 10));
    return key;
}

__device__ __forceinline__ u32 exact_argmax_index(unsigned long long key)
{
    u32 br = 1023u - (u32)((key >> 22) & 1023u);
    u32 t = __brev(br) >> 22;
    u32 hi = 0x3FFFFFu - (u32)(key & 0x3FFFFFu);
    return (hi << 10) | t;
}
