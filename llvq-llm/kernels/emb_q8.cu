// emb_q8: the carried embedding (and the tied lm_head) as int8 g64, on the
// device.
//
// The payload is MLX's exact scheme, the one `bin/embedq` writes and lot B
// scored: per group of 64 along the row, one f16 scale and one f16 bias,
// `w = scale·q + bias`, q in [0, 256). Two kernels read it:
//
//  * `emb_q8_gather` — token id → dequantized f16 row, one block per token.
//    A decode step reads exactly one row; a prefill of l tokens is one launch
//    of l blocks. Trivial arithmetic, launched twice per generated token.
//  * `tv_q8_h` — logits = W_q8 · h, one warp per output row like tv_f16 /
//    tv_slot_h: same 256-thread blocks, f32 accumulation, f16 store. The
//    activation is staged once in shared memory (d·4 bytes — 10 KB at
//    d = 2560, host-checked against the device limit; no tiling, unlike the
//    projection kernels whose d_in can be wider).
//
// Composition contract (NVRTC has no filesystem, the host concatenates):
// llvq_slot.cuh (u32), matvec.cu (h2f, f2h, warp_sum, TILE_COLS), then this
// file — after the Planes14 parts when those are present. The guard below
// only resolves from disk under a host clang++ check, where the harness
// pre-defines TILE_COLS and supplies real h2f/f2h (matvec.cu's host stubs
// return 0, which would nullify the executed mirror).
//
// Packed bytes are read through a u32 pointer: the host asserts d % 4 == 0,
// so every row starts on a word boundary, and 64 % 4 == 0 means the four
// weights of one word never straddle a group — one scale/bias pair per word.
//
// Like the rest of the family there is no `if (row >= d_out) return;` in
// tv_q8_h: a return before __syncthreads() deadlocks and would break the
// full-warp mask of warp_sum. The host asserts d_out % 8 == 0 instead
// (151 936 = 8 · 18 992 on the 4B).

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif

// Dequantize one weight: exactly the reader's arithmetic
// (llvq-artifact `RawTensor::to_f32`, line for line): widen scale and bias
// from f16, one f32 multiply, one f32 add — round-to-nearest each, and
// *never* contracted into an fma. __fmul_rn/__fadd_rn are how CUDA spells
// "no contraction"; the CPU mirror in tests/host_embq8.cpp is bit-exact
// against this text because of it.
__device__ __forceinline__ float q8_deq(u32 q, u32 sbits, u32 bbits)
{
    return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q),
                     h2f((unsigned short)bbits));
}

// ids[ids_off + t] → y[t·d .. t·d+d), dequantized and narrowed to f16 exactly
// where the f16 tensor would have held it. gpr = ceil(d / 64), the number of
// scale/bias pairs per row; the group of column c is c >> 6 because the group
// width is the format's constant 64, not a knob.
extern "C" __global__ void emb_q8_gather(const u32* __restrict__ wq,
                                         const unsigned short* __restrict__ scales,
                                         const unsigned short* __restrict__ biases,
                                         const u32* __restrict__ ids,
                                         unsigned short* __restrict__ y,
                                         u32 d, u32 gpr, u32 ids_off)
{
    u32 row = ids[ids_off + blockIdx.x];
    u32 w0  = row * (d >> 2);
    u32 s0  = row * gpr;
    for (u32 c = threadIdx.x; c < d; c += blockDim.x) {
        u32 q = (wq[w0 + (c >> 2)] >> ((c & 3u) * 8u)) & 0xffu;
        u32 g = s0 + (c >> 6);
        y[blockIdx.x * d + c] = f2h(q8_deq(q, scales[g], biases[g]));
    }
}

// One warp per vocabulary row: y[y_off + row] = f16(Σ_c deq(W[row,c])·x[c]).
//
// x is the f16 hidden state as candle holds it (x_off carries the view's
// offset, same convention as rot_apply); it is widened once into shared and
// read from there by every warp of the block. Accumulation is f32 with
// explicit __fmaf_rn, narrowed once at the store — the same arithmetic shape
// as tv_slot_h. y_off lets the host loop rows of a multi-token call into one
// output buffer.
extern "C" __global__ void tv_q8_h(const u32* __restrict__ wq,
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

    u32 w0 = row * (d >> 2);
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < (d >> 2); wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 2;
        u32 sb = scales[s0 + (c >> 6)];
        u32 bb = biases[s0 + (c >> 6)];
        acc = __fmaf_rn(q8_deq(p & 0xffu, sb, bb), xs[c], acc);
        acc = __fmaf_rn(q8_deq((p >> 8) & 0xffu, sb, bb), xs[c + 1u], acc);
        acc = __fmaf_rn(q8_deq((p >> 16) & 0xffu, sb, bb), xs[c + 2u], acc);
        acc = __fmaf_rn(q8_deq(p >> 24, sb, bb), xs[c + 3u], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[y_off + row] = f2h(acc);
}
