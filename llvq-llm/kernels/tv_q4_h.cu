// tv_q4_h: one projection served as affine int4, group 128, on the device.
//
// The mixed-precision arm of Q5. `v_proj` carries +4.48 pp of MMLU at f16 and
// +3.60 pp at int4 g128 for 2.6% of the weights (*measured*,
// docs/mesures/m2-attribution-4b-2026-09-02.txt and m2b-v4bits-2026-09-02.txt),
// and int4 g128 unfolds to 4.250 b/weight against Planes14's 4.804 — so this
// path costs nothing on either axis. Until it existed, that gain was measured
// on a matrix dequantized to f16 before the matvec: the quality of a *format*,
// not of a served path.
//
// ## What it is
//
// `tv_q8_h` (emb_q8.cu) with three changes and no fourth:
//
//   1. four bits per weight instead of eight, so a u32 carries eight columns
//      rather than four, and the nibble of column c is `4 * (c & 7)`;
//   2. the group is the format's constant 128, so the pair of column c is
//      `c >> 7` — the same "not a knob" argument emb_q8.cu makes for its 64;
//   3. `x` arrives **f32**, as in every projection kernel of this family
//      (`tv_planes_h`, `tv_planes_seg_h`), not f16 as in the vocabulary
//      kernels. Nothing is widened on the way in.
//
// Everything else is that kernel's: 256-thread blocks, one warp per output
// row, the activation staged once in shared, f32 accumulation with explicit
// `__fmaf_rn`, one narrowing store.
//
// ## Why the dequantization is a multiply and an add, and never an fma
//
// `llvq_artifact::RawTensor::to_f32` reads `scale * q + bias` — two roundings.
// Contracting them into one fma would make this kernel disagree with the
// reader in the last bit, on a path whose whole correctness test is that a
// served row equals the row the file decodes to. `__fmul_rn` / `__fadd_rn`
// are how CUDA spells "do not contract", and emb_q8.cu's `q8_deq` makes the
// same choice for the same reason.
//
// ## Addressing, and the two things the host must assert
//
// The packer writes nibbles at the **global** flat index, `packed[i / 2] |=
// q << (4 * (i % 2))` with `i = row * d_in + c` (llvq-llm/src/embedquant.rs).
// Reading that stream through a u32 pointer is only row-aligned when
// `d_in % 8 == 0`; then word `wi` of row `r` is `r * (d_in / 8) + wi` and the
// nibble of column c is `4 * (c & 7)`, with no dependence on the row. And
// `128 % 8 == 0` means the eight weights of one word never straddle a group,
// so one scale/bias pair serves the whole word — emb_q8.cu's argument, at the
// other width.
//
// Like the rest of the family there is deliberately no `if (row >= d_out)
// return;`: a return before `__syncthreads()` deadlocks, and it would break
// the full-warp mask of `warp_sum`. The host asserts `d_out % 8 == 0`.
//
// ⚠️ The activation is staged whole, not tiled: `d_in` here is a hidden size
// (2560 at 4B, 5120 at 14B), so the shared request is 10 to 20 KB. The host
// checks it against the device limit rather than assuming it — the projection
// kernels tile because their `d_in` can be an intermediate size, and this one
// must not silently inherit a bound it does not share.
//
// Composition contract (NVRTC has no filesystem, the host concatenates):
// llvq_slot.cuh (u32), matvec.cu (h2f, f2h, warp_sum), then this file. The
// guard below only resolves from disk under a host clang++ syntax check.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif

// The format's group width. A knob here would be a second source of truth
// against `embedquant::quantize_affine`'s caller; the host refuses any other
// value instead.
#define LLVQ_Q4_GROUP 128u

// Dequantize one weight: `llvq_artifact::RawTensor::to_f32`'s arithmetic,
// line for line — widen scale and bias from f16, one f32 multiply, one f32
// add, round-to-nearest each and never contracted.
__device__ __forceinline__ float q4_deq(u32 q, u32 sbits, u32 bbits)
{
    return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q),
                     h2f((unsigned short)bbits));
}

// y[row] = f16(Σ_c deq(W[row,c]) · x[c]), one warp per row.
//
// `gpr` is ceil(d_in / 128), the number of scale/bias pairs per row; it is
// passed rather than recomputed so the host and the device cannot disagree
// about the rounding up.
extern "C" __global__ void tv_q4_h(const u32* __restrict__ wq,
                                   const unsigned short* __restrict__ scales,
                                   const unsigned short* __restrict__ biases,
                                   const float* __restrict__ x,
                                   unsigned short* __restrict__ y,
                                   u32 d_in,
                                   u32 gpr)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    u32 nwords = d_in >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 3;
        // The eight columns of a word share a group by construction, so the
        // pair is read once and reused, exactly as emb_q8.cu does for four.
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
