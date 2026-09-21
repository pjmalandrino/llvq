// The q8 embedding in Metal Shading Language.
//
// A faithful port of `llvq-llm/kernels/emb_q8.cu`, both of its kernels. The
// payload is MLX's scheme, the one `llvq-llm/src/embedquant.rs` writes and
// `llvq_artifact::RawTensor::to_f32` reads: per group of 64 along the row, one
// f16 scale and one f16 bias, `w = scale*q + bias`, q in [0, 256).
//
//  * `emb_q8_gather_metal`, token id to dequantized row, one threadgroup a
//    token. A decode step reads exactly one row; a prefill of l tokens is one
//    dispatch of l threadgroups.
//  * `tv_q8_metal`, logits = W_q8 . h, one SIMD-group an output row, the same
//    shape as `tv_tetra48_metal`. The activation is staged once in threadgroup
//    memory and read from there by every SIMD-group of the group.
//
// The gate is `llvq-metal/tests/q8_matches_host.rs`. It demands equality
// against `llvq_artifact::RawTensor::to_f32`, not a tolerance, because both
// sides compute `scale*q + bias` in f32 and a port that is merely close has
// lost a bit somewhere.
//
// ## The arithmetic is EXACTLY what is written
//
// Metal is clang, and clang contracts `a * b + c` into an `fma` unless told
// not to. `fma` rounds once where the written form rounds twice, so the two
// are different numbers. The pragma below stops the contraction, and it is the
// belt: the shipped path compiles through candle with default options, which
// have fast math on, so a file that relied on the compile option would ship a
// different kernel than the one under test.
//
// This port is easier than the Tetra one, because the CUDA original spells
// every rounding out. `__fmul_rn` and `__fadd_rn` become a plain `*` and `+`
// under the pragma, `__fmaf_rn` becomes `fma` by name. One decision, written
// down, on both sides.
//
// And on q8 the pragma turns out to change NOTHING, which the mutation run
// measured rather than assumed. See `q8_deq`: its product is exact, so
// contracted or not it is the same number. The pragma stays as insurance
// against a future line that is not exact, and because one rule for every MSL
// file in this repository is cheaper than one rule a file.
//
// ## Address spaces and what the host owes this file
//
// `tv_q8_metal` stages d floats in threadgroup memory, 10,240 B at d = 2560.
// An M3 Max offers 32,768 B, measured, so there is no tiling here, unlike the
// projection kernels whose d_in can be wider. The host MUST call
// `set_threadgroup_memory_length`: omitting it is not an error on Apple, the
// kernel runs and writes zeros.
//
// Packed bytes are read through a u32 pointer, so the host asserts d % 4 == 0
// and every row starts on a word boundary. 64 % 4 == 0 means the four weights
// of one word never straddle a group, hence one scale/bias pair a word.
//
// Like the CUDA family there is no `if (row >= d_out) return;` in
// `tv_q8_metal`: a return before a barrier is undefined, and it would break
// the full-width SIMD-group the butterfly assumes. The host asserts
// d_out % 8 == 0 instead, which is 151,936 = 8 * 18,992 on the 4B.

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

/// Dequantize one weight: exactly the reader's arithmetic.
///
/// `llvq_artifact::RawTensor::to_f32` computes `f16_to_f32(scale) * q as f32 +
/// f16_to_f32(bias)`, two roundings, and Rust never contracts. CUDA spells the
/// same two roundings `__fmul_rn` and `__fadd_rn`. The pragma above is what
/// keeps this line two roundings rather than one `fma`, and on this payload
/// the difference is provably nil.
///
/// `float(half)` is exact widening, the same value `h2f` produces in CUDA and
/// `f16_to_f32` produces in Rust. Widening has nothing to round.
///
/// The two roundings are one rounding, and that is provable here. An f16
/// significand is 11 bits, `q` is 8, so the product needs at most 19 and an
/// f32 significand holds 24. The exponents are nowhere near either end: the
/// largest case is 65504 * 255, about 1.7e7. So `float(s) * float(q)` is
/// EXACT, and `fma(float(s), float(q), float(b))` returns the same bits as
/// this line for every input. Checked by exhaustion on 2026-09-21 over all
/// 63,488 finite f16 scales times all 256 byte values, zero products rounded.
/// The mutation run confirms it: replacing this line by that `fma` is the one
/// mutant the gate does not kill, and it cannot, because it is the same
/// function.
///
/// The line stays as written, because it is the port of `__fmul_rn` and
/// `__fadd_rn` and because the proof above holds for q8 alone. A q4 or a q16
/// payload would need it checked again.
inline float q8_deq(uint q, half s, half b)
{
    return float(s) * float(q) + float(b);
}

/// ids[ids_off + t] goes to y[t*d .. t*d+d), dequantized.
///
/// `gpr` is ceil(d / 64), the number of scale/bias pairs a row. The group of
/// column c is `c >> 6` because the group width is the format's constant 64,
/// not a knob.
///
/// The store is f32 where the CUDA twin stores f16 through `f2h`. Deliberate,
/// and a decision of this lot rather than an oversight: `f2h` is
/// round-to-nearest-even and narrowing here without a gate that compares the
/// two widths would be a claim, not a change. `tv_tetra48_metal` already
/// stores f32 for the same reason.
///
/// The column loop strides by the threadgroup's own width, so it covers every
/// column whatever size Metal gave this group. That matters because
/// `dispatch_threads` hands out non-uniform threadgroups.
kernel void emb_q8_gather_metal(const device uint* wq      [[buffer(0)]],
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
    uint w0  = row * (d >> 2);
    uint s0  = row * gpr;
    for (uint c = tid; c < d; c += tgs) {
        uint q = (wq[w0 + (c >> 2)] >> ((c & 3u) * 8u)) & 0xffu;
        uint g = s0 + (c >> 6);
        y[tok * d + c] = q8_deq(q, scales[g], biases[g]);
    }
}

/// The butterfly, written out rather than `simd_sum`.
///
/// `simd_sum` does not specify its reduction order, and floating-point
/// addition is not associative, so a kernel built on it cannot promise the
/// same bits twice across drivers. CUDA's `warp_sum` is an explicit
/// `__shfl_xor_sync` butterfly; this is the same one, lane for lane, which is
/// what lets the gate demand equality against a host reference.
///
/// Named apart from `warp_sum` in `llvq_tetra48.metal`, which is the same
/// function. The two files are separate libraries today, and a host that ever
/// concatenated them would hit a redefinition rather than a choice.
inline float q8_warp_sum(float v)
{
    for (ushort k = 16; k > 0; k >>= 1) {
        v += simd_shuffle_xor(v, k);
    }
    return v;
}

/// One SIMD-group a vocabulary row: y[y_off + row] = sum_c deq(W[row,c])*x[c].
///
/// x is the f16 hidden state as candle holds it, and `x_off` carries the
/// view's offset, the same convention as `rot_apply`. It is widened once into
/// threadgroup memory and read from there by every SIMD-group of the group.
/// Accumulation is f32 with explicit `fma`, mirroring the CUDA `__fmaf_rn`.
/// `y_off` lets the host loop the rows of a multi-token call into one output
/// buffer.
///
/// One barrier, not two, and the difference from `tv_tetra48_metal` is real:
/// the staging happens once here, so no second pass can race a straggler still
/// reading it.
///
/// The store is f32, for the reason `emb_q8_gather_metal` gives.
kernel void tv_q8_metal(const device uint* wq     [[buffer(0)]],
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

    uint w0 = row * (d >> 2);
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < (d >> 2); wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 2;
        // One pair a word: the four columns of a word share a group, because
        // the group width 64 is a multiple of 4.
        half sb = scales[s0 + (c >> 6)];
        half bb = biases[s0 + (c >> 6)];
        acc = fma(q8_deq(p & 0xffu, sb, bb), xs[c], acc);
        acc = fma(q8_deq((p >> 8) & 0xffu, sb, bb), xs[c + 1u], acc);
        acc = fma(q8_deq((p >> 16) & 0xffu, sb, bb), xs[c + 2u], acc);
        acc = fma(q8_deq(p >> 24, sb, bb), xs[c + 3u], acc);
    }
    acc = q8_warp_sum(acc);
    if (lane == 0u) {
        y[y_off + row] = acc;
    }
}
