// The q4 embedding in Metal Shading Language.
//
// A port of `llvq-llm/kernels/emb_q4.cu`, both of its kernels, and
// `emb_q8.metal` at half the weight width. The payload is the int4 g64 scheme
// `bin/embedq q4` writes and `llvq_artifact::RawTensor::to_f32` reads: per
// group of 64 along the row, one f16 scale and one f16 bias,
// `w = scale*q + bias`, q in [0, 16).
//
//  * `emb_q4_gather_metal`, token id to dequantized row, one threadgroup a
//    token.
//  * `tv_emb_q4_metal`, logits = W_q4 . h, one SIMD-group an output row, the
//    activation staged once in threadgroup memory.
//
// Named apart from `tv_q4_metal` (`tv_q4_h.metal`, the int4 g128 projection),
// which a served mixed file launches beside these.
//
// The gate is `llvq-metal/tests/q4e_matches_host.rs`, equality against
// `RawTensor::to_f32` and against the CUDA original's lane order.
//
// ## Arithmetic, and what the host owes
//
// `emb_q8.metal` explains the pragma and the two roundings of `e4_deq`. On
// this payload the product is exact (11 + 4 significand bits against 24), so
// contracted or not it is the same number; the line stays the port of
// `__fmul_rn` and `__fadd_rn`.
//
// Packed bytes are read through a u32 pointer, nibbles low first at the global
// flat index, so the host asserts d % 8 == 0 and every row starts on a word
// boundary; 64 % 8 == 0 means one scale/bias pair a word. `tv_emb_q4_metal`
// stages d floats (10,240 B at d = 2560) and the host MUST call
// `set_threadgroup_memory_length`. No early return before the barrier; the
// host asserts vocab % 8 == 0. Both kernels store f32, as the q8 pair does.

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

/// Dequantize one weight: exactly the reader's arithmetic.
inline float e4_deq(uint q, half s, half b)
{
    return float(s) * float(q) + float(b);
}

/// ids[ids_off + t] goes to y[t*d .. t*d+d), dequantized. gpr = ceil(d / 64).
kernel void emb_q4_gather_metal(const device uint* wq      [[buffer(0)]],
                                const device half* scales  [[buffer(1)]],
                                const device half* biases  [[buffer(2)]],
                                const device uint* ids     [[buffer(3)]],
                                device float*      y       [[buffer(4)]],
                                constant uint&     d       [[buffer(5)]],
                                constant uint&     gpr     [[buffer(6)]],
                                constant uint&     ids_off [[buffer(7)]],
                                uint tok [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tgs [[threads_per_threadgroup]])
{
    uint row = ids[ids_off + tok];
    uint w0  = row * (d >> 3);
    uint s0  = row * gpr;
    for (uint c = tid; c < d; c += tgs) {
        uint q = (wq[w0 + (c >> 3)] >> ((c & 7u) * 4u)) & 0xfu;
        uint g = s0 + (c >> 6);
        y[tok * d + c] = e4_deq(q, scales[g], biases[g]);
    }
}

/// The butterfly, written out rather than `simd_sum`, for the reason
/// `emb_q8.metal` gives. Named apart from its twins in the other files.
inline float e4_warp_sum(float v)
{
    for (ushort k = 16; k > 0; k >>= 1) {
        v += simd_shuffle_xor(v, k);
    }
    return v;
}

/// One SIMD-group a vocabulary row: y[y_off + row] = sum_c deq(W[row,c])*x[c].
///
/// Lane `l` takes words `l, l + 32, ...`, the eight columns of a word
/// accumulate in order with explicit `fma`, and the lanes meet in the
/// butterfly: `tv_emb_q4_h`'s order, lane for lane.
kernel void tv_emb_q4_metal(const device uint* wq     [[buffer(0)]],
                            const device half* scales [[buffer(1)]],
                            const device half* biases [[buffer(2)]],
                            const device half* x      [[buffer(3)]],
                            device float*      y      [[buffer(4)]],
                            constant uint&     d      [[buffer(5)]],
                            constant uint&     gpr    [[buffer(6)]],
                            constant uint&     x_off  [[buffer(7)]],
                            constant uint&     y_off  [[buffer(8)]],
                            threadgroup float* xs     [[threadgroup(0)]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint gid  [[thread_position_in_grid]],
                            uint tgs  [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    for (uint i = tid; i < d; i += tgs) {
        xs[i] = float(x[x_off + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nwords = d >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 3;
        // One pair a word: its eight columns share a group.
        half sb = scales[s0 + (c >> 6)];
        half bb = biases[s0 + (c >> 6)];
        for (uint k = 0; k < 8u; ++k) {
            acc = fma(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    acc = e4_warp_sum(acc);
    if (lane == 0u) {
        y[y_off + row] = acc;
    }
}
