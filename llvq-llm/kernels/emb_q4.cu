// emb_q4: the carried embedding (and the tied lm_head) as int4 g64, on the
// device.
//
// The payload `bin/embedq q4` writes and the 2026-09-23 census scored on the
// dense path (docs/mesures/embed-q4-swap-2026-09-23.txt): per group of 64
// along the row, one f16 scale and one f16 bias, `w = scale·q + bias`, q in
// [0, 16). It is emb_q8.cu with the weight width halved, and nothing else:
//
//   1. four bits per weight, so a u32 carries eight columns rather than four,
//      and the nibble of column c is `4 * (c & 7)`;
//   2. the group is still the format's 64, so the pair of column c is still
//      `c >> 6`, and a word's eight columns never straddle a group because
//      64 % 8 == 0.
//
// Two kernels read it, under names of their own: `tv_q4_h` already names the
// int4 g128 projection kernel, and the served mixed file puts both sources in
// one translation unit.
//
//  * `emb_q4_gather`: token id → dequantized f16 row, one block per token.
//  * `tv_emb_q4_h`: logits = W_q4 · h, one warp per vocabulary row, the
//    activation staged once in shared memory (d·4 bytes, host-checked).
//
// Packed bytes are read through a u32 pointer. The packer writes nibbles at
// the global flat index, `packed[i / 2] |= q << (4 * (i % 2))` with
// `i = row · d + c` (llvq-llm/src/embedquant.rs), so a row starts on a word
// boundary exactly when `d % 8 == 0`, which the host asserts. Then word `wi`
// of row `r` is `r · (d / 8) + wi`, independent of the row's parity.
//
// No `if (row >= d_out) return;` in tv_emb_q4_h, for emb_q8.cu's reason: a
// return before __syncthreads() deadlocks and would break warp_sum's full
// mask. The host asserts vocab % 8 == 0 (151,936 = 8 · 18,992 on the 4B).
//
// Composition contract (NVRTC has no filesystem, the host concatenates):
// llvq_slot.cuh (u32), matvec.cu (h2f, f2h, warp_sum), then this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif

// Dequantize one weight: the reader's arithmetic (`RawTensor::to_f32`), widen
// scale and bias from f16, one f32 multiply, one f32 add, never contracted.
// On this payload the product is exact anyway: an f16 significand is 11 bits
// and q is 4, so 15 bits fit an f32's 24 and contraction could not move a
// bit. The spelling stays the port of the reader, as in emb_q8.cu.
__device__ __forceinline__ float e4_deq(u32 q, u32 sbits, u32 bbits)
{
    return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q),
                     h2f((unsigned short)bbits));
}

// ids[ids_off + t] → y[t·d .. t·d+d), dequantized and narrowed to f16 exactly
// where the f16 tensor would have held it. gpr = ceil(d / 64).
extern "C" __global__ void emb_q4_gather(const u32* __restrict__ wq,
                                         const unsigned short* __restrict__ scales,
                                         const unsigned short* __restrict__ biases,
                                         const u32* __restrict__ ids,
                                         unsigned short* __restrict__ y,
                                         u32 d, u32 gpr, u32 ids_off)
{
    u32 row = ids[ids_off + blockIdx.x];
    u32 w0  = row * (d >> 3);
    u32 s0  = row * gpr;
    for (u32 c = threadIdx.x; c < d; c += blockDim.x) {
        u32 q = (wq[w0 + (c >> 3)] >> ((c & 7u) * 4u)) & 0xfu;
        u32 g = s0 + (c >> 6);
        y[blockIdx.x * d + c] = f2h(e4_deq(q, scales[g], biases[g]));
    }
}

// One warp per vocabulary row: y[y_off + row] = f16(Σ_c deq(W[row,c])·x[c]).
//
// x is the f16 hidden state as candle holds it, x_off the view's offset,
// widened once into shared. f32 accumulation with explicit __fmaf_rn in
// column order within a lane, one narrowing store: tv_q8_h's shape.
extern "C" __global__ void tv_emb_q4_h(const u32* __restrict__ wq,
                                       const unsigned short* __restrict__ scales,
                                       const unsigned short* __restrict__ biases,
                                       const unsigned short* __restrict__ x,
                                       unsigned short* __restrict__ y,
                                       u32 d, u32 gpr, u32 x_off, u32 y_off)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) xs[i] = h2f(x[x_off + i]);
    __syncthreads();

    u32 nwords = d >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 3;
        // One pair a word: its eight columns share a group.
        u32 sb = scales[s0 + (c >> 6)];
        u32 bb = biases[s0 + (c >> 6)];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k)
            acc = __fmaf_rn(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[y_off + row] = f2h(acc);
}
