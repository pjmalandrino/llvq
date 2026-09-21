// The affine int4 matvec in Metal Shading Language.
//
// A port of `llvq-llm/kernels/tv_q4_h.cu`, the mixed-precision arm of Q5. The
// served 4B runs 36 `v_proj` through this kernel and 216 matrices through the
// Leech path (`configs/qwen3-4b-tetra-q5.json`). Read the CUDA file first: it
// carries the format argument, the rate accounting and the addressing proof,
// and none of that is repeated here. This file carries only what changes when
// the target is Apple.
//
// ## What the port has to get right
//
// Each of the following fails silently rather than faulting.
//
//   The nibble order. The packer writes at the GLOBAL flat index,
//   `packed[i / 2] |= q << (4 * (i % 2))` with `i = row * d_in + c`. A word
//   read little-endian therefore holds column `c + k` at bit `4 * k`. Reading
//   the other order transposes every pair of columns and returns weights that
//   load, run, and give numbers. `ops/awq_dequant.py` carries the same hazard
//   under the name `AWQ_REVERSE_ORDER`.
//
//   The group index. One scale and one bias serve 128 columns. Off by one
//   group is a few percent on every weight, which no tolerance would catch.
//
//   The summation order. Floating-point addition is not associative, so the
//   32 lanes, their stride and their butterfly are part of the answer, not an
//   implementation detail.
//
// ## The arithmetic, and why it is spelled out
//
// The CUDA original never leaves a multiply-add to the compiler. It writes
// `__fmul_rn` and `__fadd_rn` where it wants two roundings, and `__fmaf_rn`
// where it wants one. Metal is clang, and clang contracts `a * b + c` into an
// `fma` unless told not to. So this file does both: it carries the pragma
// below, and it splits every two-rounding site into two statements. There is
// no bare `a * b + c` anywhere in it.
//
// ## The store is f32, on purpose
//
// The CUDA kernel ends in `f2h` and writes `unsigned short`. This one writes
// `device float*`, as `tv_tetra48_metal` already does. Narrowing to f16 is a
// separate lot with its own gate, because `f2h` is round-to-nearest-even and
// an untested rounding would be a second defect wearing the first one's
// clothes.
//
// ## Address spaces and the three host duties, none of them written yet
//
// The activation is staged WHOLE, with no tile. `d_in` here is a hidden size,
// 2560 on the served 4B, so the request is 10,240 B against the 32,768 B an
// M3 Max offers (measured). The host owes two things for that to work:
//
//   1. `setThreadgroupMemoryLength` on index 0. Omitting it is not an error.
//      The kernel runs and writes zeros, which is the worst failure available;
//   2. a check of `d_in * 4` against the device limit. The projection kernels
//      tile because their `d_in` can be an intermediate size, and this one
//      must not silently inherit a bound it does not share;
//   3. `d_out` a multiple of the rows a threadgroup covers, because there is
//      deliberately no `row >= d_out` guard below. A partial group would
//      compute rows past the output AND STORE them.
//
// None of the three is written. There is no Metal host for this kernel yet:
// the gate discharges all three itself, and a review of 2026-09-21 asked that
// this be said rather than implied. `fused_metal::upload` already refuses (3)
// for the Tetra kernel and owes the same refusal here.
//
// Metal refuses a zero-length buffer and hands back a null pointer, the same
// wall cudarc puts up. No argument here can be empty: `d_in` is a multiple of
// 8 and positive, so `wq`, `scales`, `biases` and `x` all hold at least one
// element. The dummy the Tetra kernel's tail needs has no counterpart here.

#include <metal_stdlib>
using namespace metal;

// Every multiply-add in this file is exactly what is written.
//
// `fma` rounds once where the written form rounds twice, so the two are
// different numbers. That is invisible in a benchmark and fatal here: the gate
// is equality against `llvq_artifact::Int4Matrix::to_f32`, which rounds twice.
// Turning Metal's fast math off is necessary and not sufficient, because it
// stops the reassociation and leaves the contraction. This stops the
// contraction, and it is the belt the shipped path wears: candle compiles this
// source with default options, which have fast math on.
#pragma clang fp contract(off)

// The format's group width, `llvq_artifact::INT4G128_GROUP`.
//
// A knob here would be a second source of truth against the encoder, so the
// host refuses any other value instead. `llvq_llm::fused` asserts this literal
// against the constant, and `q4_matches_host.rs` asserts it against this file.
#define LLVQ_Q4_GROUP 128u

/// Sum across the 32 lanes of a SIMD group.
///
/// Written out rather than `simd_sum`, which does not specify its reduction
/// order. Floating-point addition is not associative, so a kernel built on
/// `simd_sum` cannot promise the same bits twice across drivers. CUDA's
/// `warp_sum` is an explicit `__shfl_xor_sync` butterfly, and this is the same
/// one lane for lane, which is what lets the gate demand equality.
inline float warp_sum(float v)
{
    for (ushort k = 16; k > 0; k >>= 1) {
        v += simd_shuffle_xor(v, k);
    }
    return v;
}

/// Dequantize one weight: `scale * q + bias`, two roundings, never one.
///
/// This is `llvq_artifact::Int4Matrix::to_f32` line for line, and that is the
/// whole point. The file path and the served path have to agree in the last
/// bit, on a kernel whose correctness test is that a served row equals the row
/// the file decodes to. Contracting the product and the sum into one `fma`
/// would break that agreement while looking like an optimisation.
///
/// `as_type<half>` is the bitcast CUDA's `cvt.f32.f16` performs on the same
/// `unsigned short`. Widening binary16 to binary32 is exact, so there is
/// nothing to round on the way in.
///
/// Measured 2026-09-21: at THIS width the two forms cannot differ. A binary16
/// scale carries 11 significant bits and `q` carries 4, so the product needs
/// 15 and f32 offers 24. Over all 63,488 finite f16 scales and all 16 levels,
/// 1,015,808 products, not one rounds. `fma(s, q, b)` therefore returns the
/// same f32 as `s * q + b`, and the mutation run in `q4_matches_host.rs`
/// could not kill the contracted form. The two roundings stay, because the
/// equality is a property of the operand ranges: a scale stored at f32 would
/// end it, and the kernel must not depend on an accident it does not state.
inline float q4_deq(uint q, ushort sbits, ushort bbits)
{
    float p = float(as_type<half>(sbits)) * float(q);
    return p + float(as_type<half>(bbits));
}

/// y[row] = Σ_c deq(W[row,c]) · x[c], one SIMD group a row.
///
/// `gpr` is ceil(d_in / 128), the scale/bias pairs a row carries. It is passed
/// rather than recomputed so the host and the device cannot disagree about the
/// rounding up.
///
/// There is deliberately no `if (row >= d_out) return;`. The CUDA reason is
/// that a return before `__syncthreads()` deadlocks; the Apple reason is
/// simpler, because `dispatchThreads` launches exactly `d_out * 32` threads
/// and no thread is ever out of range. A partial last threadgroup is legal,
/// and `tgs` reports its real size, so the staging loop below still covers
/// `d_in`.
kernel void tv_q4_metal(const device uint*   wq     [[buffer(0)]],
                        const device ushort* scales [[buffer(1)]],
                        const device ushort* biases [[buffer(2)]],
                        const device float*  x      [[buffer(3)]],
                        device float*        y      [[buffer(4)]],
                        constant uint&       d_in   [[buffer(5)]],
                        constant uint&       gpr    [[buffer(6)]],
                        threadgroup float*   xs     [[threadgroup(0)]],
                        uint tid  [[thread_position_in_threadgroup]],
                        uint gid  [[thread_position_in_grid]],
                        uint tgs  [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    for (uint i = tid; i < d_in; i += tgs) {
        xs[i] = x[i];
    }
    // One barrier, not the two the tiled kernels carry: there is a single
    // fill, so nothing can race a straggler reading a previous tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Eight nibbles a word, which is what makes a row start on a word when
    // `d_in % 8 == 0`. NOBODY asserts that today: there is no Metal host for
    // this kernel, and the gate only ever builds conforming fixtures. Without
    // it every row after the first reads at a shifted nibble.
    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 3;
        // The eight columns of a word share a group by construction, because
        // 128 is a multiple of 8. So the pair is read once and reused. The
        // division is by a compile-time power of two on an unsigned value,
        // which is the CUDA original's `c >> 7` with one source of truth for
        // the 128 instead of two.
        uint g = s0 + c / LLVQ_Q4_GROUP;
        ushort sb = scales[g];
        ushort bb = biases[g];
        for (uint k = 0u; k < 8u; ++k) {
            // `fma` by name, where the CUDA original writes `__fmaf_rn`.
            acc = fma(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        // f32, deliberately. See the header: narrowing is a separate lot.
        y[row] = acc;
    }
}
