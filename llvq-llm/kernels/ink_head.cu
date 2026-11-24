// tv_emb_q4_xf: `tv_emb_q4_h` reading an f32 activation and storing f32, the
// int4 g64 output head of a model that runs in f32 (Inkling 2 Mini), CUDA
// twin of `tv_emb_q4_metal_xf`. The staging, the lane order, the
// dequantization and the reduction are `tv_emb_q4_h`'s; only the types of `x`
// and `y` change, so neither the hidden state nor a logit is narrowed to f16.
//
// Composition: llvq_slot.cuh, matvec.cu, emb_q4.cu (e4_deq), then this file.
// The host owes `d * 4` bytes of shared memory and a row count that fills
// whole blocks.
extern "C" __global__ void tv_emb_q4_xf(const u32* __restrict__ wq,
                                        const unsigned short* __restrict__ scales,
                                        const unsigned short* __restrict__ biases,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        u32 d, u32 gpr, u32 x_off, u32 y_off)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (u32 i = threadIdx.x; i < d; i += blockDim.x) xs[i] = x[x_off + i];
    __syncthreads();

    u32 nwords = d >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 3;
        u32 sb = scales[s0 + (c >> 6)];
        u32 bb = biases[s0 + (c >> 6)];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k)
            acc = __fmaf_rn(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[y_off + row] = acc;
}
