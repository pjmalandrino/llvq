// tv_q4_ht / tv_q4_hht: the int4 g128 matvec on tiled words, the layout 4E v9
// reads (pers_q4_tile_index in llvq-cuda/kernels/llvq_pers.cuh).
//
// ## Why it exists
//
// 4E v9 reads its int4 rows (v, o and the int4 downs) from words relaid so one
// 128-bit load gives lane l its words 128 t + l + 32 s, s = 0..3 (K11 of
// docs/plan-300.md, x1.338 over the served row alone). Its measurement form
// kept the served words for the eager path and a tiled copy for 4E: the int4
// bytes twice on the card, 0.385 GB more on the 4B. Under a config whose rows
// are "v9" the loader writes the tiled words alone (fused_cuda.rs,
// `upload_int4`), and the eager path, which prefills and runs the load-time
// checks, reads them through these two entry points.
//
// ## Why it is the served arithmetic, bit for bit
//
// `q4_staged_dot` (tv_q4_h.cu) has lane l fold its words l, l + 32, l + 64,
// ... in increasing order, eight products each in increasing column, then
// `warp_sum`. Inside tile t those words are 128 t + l + 32 s for s = 0..3, in
// that order, and the tiled layout puts them at 128 t + 4 l + s: one uint4 at
// 4 l. The loop below reads that uint4 and folds s = 0..3, so every FMA sees
// the same two operands in the same order. The words past the last full tile
// keep the served order and run the served loop. Only the addresses move.
// `llvq-cuda/tests/pers_matches_served.rs` runs both readers on the host and
// compares the outputs byte for byte; the card checks it again at every load
// on a real record (`check_int4_tile_parity`).
//
// ## What the host asserts
//
// `d_in % 32 == 0` (nwords a multiple of 4, so every row starts 16-byte
// aligned on a 256-byte aligned buffer), as 4E's install does. The rest is
// tv_q4_h's contract: d_out % 8 == 0, one warp a row, 256-thread blocks, the
// activation staged whole in f32.
//
// ## Why a file of its own
//
// Appended to the unit only when the runtime holds tiled words, so the unit
// every other config compiles, and the sha256 its log prints, do not move.
// Composition: llvq_slot.cuh (u32), matvec.cu (h2f, f2h, warp_sum),
// tv_q4_h.cu (q4_deq), then this file.

#ifndef LLVQ_Q4_GROUP
#include "tv_q4_h.cu"
#endif

__device__ __forceinline__ void q4t_staged_dot(const u32* __restrict__ wq,
                                               const unsigned short* __restrict__ scales,
                                               const unsigned short* __restrict__ biases,
                                               const float* xs,
                                               unsigned short* __restrict__ y,
                                               u32 d_in,
                                               u32 gpr)
{
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 nwords = d_in >> 3;
    u32 nt = nwords >> 7;
    const u32* wrow = wq + (unsigned long long)row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 t = 0; t < nt; ++t) {
        const uint4 P = *reinterpret_cast<const uint4*>(wrow + (t << 7) + 4u * lane);
        const u32 w[4] = {P.x, P.y, P.z, P.w};
#pragma unroll
        for (u32 s = 0; s < 4u; ++s) {
            u32 c = ((t << 7) + 32u * s + lane) << 3;
            u32 g  = s0 + (c >> 7);
            u32 sb = scales[g];
            u32 bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k)
                acc = __fmaf_rn(q4_deq((w[s] >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {
        u32 p = wrow[wi];
        u32 c = wi << 3;
        u32 g  = s0 + (c >> 7);
        u32 sb = scales[g];
        u32 bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k)
            acc = __fmaf_rn(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = f2h(acc);
}

// tv_q4_h on tiled words: `x` in f32.
extern "C" __global__ void tv_q4_ht(const u32* __restrict__ wq,
                                    const unsigned short* __restrict__ scales,
                                    const unsigned short* __restrict__ biases,
                                    const float* __restrict__ x,
                                    unsigned short* __restrict__ y,
                                    u32 d_in,
                                    u32 gpr)
{
    extern __shared__ float xs[];
    for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    q4t_staged_dot(wq, scales, biases, xs, y, d_in, gpr);
}

// tv_q4_hh on tiled words: `x` in f16, widened while it is staged.
extern "C" __global__ void tv_q4_hht(const u32* __restrict__ wq,
                                     const unsigned short* __restrict__ scales,
                                     const unsigned short* __restrict__ biases,
                                     const unsigned short* __restrict__ x,
                                     unsigned short* __restrict__ y,
                                     u32 d_in,
                                     u32 gpr)
{
    extern __shared__ float xs[];
    for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(x[i]);
    __syncthreads();
    q4t_staged_dot(wq, scales, biases, xs, y, d_in, gpr);
}
