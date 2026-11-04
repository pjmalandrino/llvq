// tv_q4_hs: the int4 g128 matvec with its activation staged as f16 bits.
//
// `tv_q4_h` and `tv_q4_hh` stage the whole activation in f32 shared memory,
// `d_in * 4` bytes. An int4 `down_proj` of Qwen3-32B has d_in = 25,600, so it
// asks 102,400 B, and the cards with 99 KiB of opt-in (sm_86, sm_89, sm_120)
// offer 101,376. This entry point stages the raw f16 bits instead, 51,200 B,
// and widens each one at its use: `h2f(xs16[c + k])` is exactly the float
// `tv_q4_hh` stored in `xs[c + k]` (binary16 to binary32 is exact, and it is
// the same `h2f`). So every FMA sees the same two operands in the same order,
// the warp butterfly is the same `warp_sum`, and the output is `tv_q4_hh`'s,
// bit for bit. `llvq-llm/tests/proj_q4_wide.rs` executes both kernels lane by
// lane on the host and compares the partial sums at every width the 32B needs.
//
// 51,200 B is past the 49,152 B default, so this function takes its OWN
// opt-in, sized over the f16-staged records alone
// (`llvq_cuda::wide::int4_entry_bytes`). One CTA of 8 warps an SM at 25,600.
//
// ## Only where the f32 staging is refused
//
// `llvq_cuda::wide::int4_staging` sends a record here only when its f32
// staging passes the card's ceiling. On the served 4B, 8B and 14B no record
// does, so this unit is never compiled for them and nothing they run moves.
//
// ## The activation, and `x_off`
//
// It takes an f16 activation only: an f32 one would have to be narrowed, which
// changes the arithmetic, so the runtime refuses that pairing by name. It
// takes `x_off` (in elements) so a prefill row narrowed out of a block is read
// where it lies; the served kernels send such a row through an f32 cast and
// `tv_q4_h`, which would hit the wall again at 25,600.
//
// ## Why a unit of its own, and a duplicated body
//
// Composition: `llvq_slot.cuh` (u32), `matvec.cu` (h2f, f2h, warp_sum),
// `tv_q4_h.cu` (q4_deq), then this file. Appending it to the served unit would
// move the sha256 every served log prints. The loop below duplicates
// `q4_staged_dot` rather than templating it, so the served text and its
// register allocation do not move either; the host test is what keeps the
// copy from drifting.

#ifndef LLVQ_Q4_GROUP
#include "tv_q4_h.cu"
#endif

// The lane's partial sum, before the butterfly: `q4_staged_dot`'s loop with
// `xs[c + k]` read as `h2f(xs16[c + k])`. A function of its own so the host
// test can run it for one lane at a time; the kernel below is its only
// caller on the device.
__device__ __forceinline__ float q4hs_lane(const u32* __restrict__ wq,
                                           const unsigned short* __restrict__ scales,
                                           const unsigned short* __restrict__ biases,
                                           const unsigned short* xs16,
                                           u32 d_in,
                                           u32 gpr,
                                           u32 row,
                                           u32 lane)
{
    u32 nwords = d_in >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 3;
        u32 g  = s0 + (c >> 7);
        u32 sb = scales[g];
        u32 bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k)
            acc = __fmaf_rn(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), h2f(xs16[c + k]), acc);
    }
    return acc;
}

// y[row] = f16(Σ_c deq(W[row,c]) · x[x_off + c]), one warp a row, `x` in f16.
// Same grid as `tv_q4_hh`: 256-thread blocks, 8 rows a block, no bounds guard
// (the host asserts d_out % 8 == 0), `d_in * 2` bytes of dynamic shared memory.
extern "C" __global__ void tv_q4_hs(const u32* __restrict__ wq,
                                    const unsigned short* __restrict__ scales,
                                    const unsigned short* __restrict__ biases,
                                    const unsigned short* __restrict__ x,
                                    unsigned short* __restrict__ y,
                                    u32 d_in,
                                    u32 gpr,
                                    u32 x_off)
{
    extern __shared__ unsigned short xs16[];
    for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs16[i] = x[x_off + i];
    __syncthreads();

    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = q4hs_lane(wq, scales, biases, xs16, d_in, gpr, row, lane);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = f2h(acc);
}
