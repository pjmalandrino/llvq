// AWQ 4-bit gemv — the deployed 4-bit reference, in our harness.
//
// ============================================================================
// PROVENANCE. This file is a port of `awq/kernels/csrc/quantization/gemv_cuda.cu`
// from https://github.com/mit-han-lab/llm-awq, MIT License, Copyright (c) 2023
// MIT HAN Lab. Retrieved 2026-08-10. Their header, preserved verbatim:
//
//     // Inspired by https://github.com/ankan-ban/llama_cu_awq
//     /*
//     @article{lin2023awq,
//       title={AWQ: Activation-aware Weight Quantization for LLM Compression
//              and Acceleration},
//       author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and
//               Dang, Xingyu and Han, Song},
//       journal={arXiv},
//       year={2023}
//     }
//     */
//
// ============================================================================
// WHY A PORT AND NOT A COPY, AND WHAT EXACTLY CHANGED.
//
// The point of this arm is to time **their** kernel, so the arithmetic is
// untouched: the same dequantisation `scale * (w - zero)`, the same nibble
// walk, the same `float4` load geometry, the same warp reduction, the same
// accumulation order. Reassociating one sum would make this a measurement of
// our rewrite instead of a measurement of AWQ.
//
// Four changes, all of them type names and none of them arithmetic:
//
//   1. `#include <cuda_fp16.h>`, `<stdio.h>`, `<torch/extension.h>` and
//      `"gemv_cuda.h"` are removed. The first is unreachable — the runtime
//      image ships no `-dev` package, the same wall `matvec.cu:22-30` already
//      documents. The last three are host-side plumbing we do not use.
//   2. `half` becomes `unsigned short`, which is what the rest of this
//      workspace calls a binary16 in device code.
//   3. `__half2float` / `__float2half` become `h2f` / `f2h` from `matvec.cu`.
//      Those are `cvt.f32.f16` / `cvt.rn.f16.f32` in inline PTX — *exactly*
//      what the intrinsics compile to — with a software branch under
//      `LLVQ_HOST_BUILD` so the clang++ host harness can run this file.
//   4. `uint32_t` becomes `unsigned int`, so this file needs no `<cstdint>`.
//      Same type on every CUDA target.
//
// And one addition: `extern "C"` on the entry point, because `Cuda::func`
// looks up an unmangled name. Linkage, not codegen.
//
// The `gemv_kernel_g64` variant is NOT ported. Qwen3's AWQ checkpoints are
// `group_size = 128`, and carrying a kernel no arm can reach would put an
// unmeasured `__global__` in the translation unit — the register report would
// grow a line for a kernel nobody runs, which is precisely the drift the
// report exists to catch.
//
// ============================================================================
// WHAT IT EXPECTS, AND WHY THE HOST HAS WORK TO DO.
//
//   inputs           [IC], binary16, read as float4 (8 halves per load)
//   weight           [OC, IC / 8], u32, eight 4-bit nibbles per word
//   zeros            [OC, ceil(IC/group_size / 8) * 8] packed, u32
//   scaling_factors  [OC, ceil(IC/group_size / 8) * 8], binary16
//   outputs          [OC], binary16
//
// ⚠️ An AWQ checkpoint on the Hub stores `scales` and `qzeros` **transposed**
// relative to this: `[IC/group_size, OC]` and `[IC/group_size, OC/8]`. The
// transpose is host-side work, and it is not free to get right — `ops/awq_dequant.py`
// carries the same hazard under the name `AWQ_REVERSE_ORDER`, and its module
// docstring says why it matters: a wrong unpacking "produces weights that
// load, run and give numbers".
//
// ⚠️ `zeros_w` below is *their* expression, and their own comment calls it
// incorrect ("TODO (Haotian): zeros_w is incorrect, after fixing we got
// misaligned address"). It is preserved as-is. We are timing the kernel that
// ships, not the kernel that should have shipped, and changing it would
// silently move the bytes this arm reads.

#define AWQ_PACK_FACTOR 8
#define AWQ_WARP_SIZE 32

// Their tree reduction, unchanged.
__device__ __forceinline__ float awq_warp_reduce_sum(float sum)
{
#pragma unroll
    for (int i = 4; i >= 0; i--) {
        sum += __shfl_down_sync(0xffffffff, sum, 1 << i);
    }
    return sum;
}

__device__ __forceinline__ int awq_make_divisible(int c, int divisor)
{
    return (c + divisor - 1) / divisor;
}

/// GEMV at group_size = 128. Their `gemv_kernel_g128`, verbatim but for the
/// four type substitutions listed at the top of this file.
///
///   inputs: [batch_size, IC] · weight: [OC, IC / 8] · output: [OC]
///   zeros: [OC, IC / group_size / 8] · scaling_factors: [OC, IC / group_size]
///
/// Their note, kept because it is a real trap: one cannot infer `group_size`
/// from the shape of the scaling factors, the second dimension being rounded
/// up to a multiple of PACK_FACTOR.
extern "C" __global__ void awq_gemv_g128(const float4* _inputs,
                                         const unsigned int* weight,
                                         const unsigned int* zeros,
                                         const unsigned short* scaling_factors,
                                         unsigned short* _outputs,
                                         const int IC,
                                         const int OC)
{
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const float4* inputs = _inputs + batch_idx * IC / AWQ_PACK_FACTOR;
    unsigned short* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = awq_make_divisible(IC / group_size, AWQ_PACK_FACTOR);
    const int weight_w = IC / AWQ_PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = awq_make_divisible(IC / group_size, AWQ_PACK_FACTOR);
    // consistent with input shape
    const int sf_w = awq_make_divisible(IC / group_size, AWQ_PACK_FACTOR) * AWQ_PACK_FACTOR;
    // tile size: 4 OC x 1024 IC per iter
    for (int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++) {
        // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
        unsigned int packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
        unsigned int packed_weights[4];
        // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
        *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w
                                                   + packed_group_idx * (AWQ_WARP_SIZE * 4)
                                                   + threadIdx.x * 4));
        // load scaling factors
        // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
        float scaling_factor =
            h2f(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
        float current_zeros = (float)((packed_zeros >> (threadIdx.x / 4 * 4)) & 0xF);
        int inputs_ptr_delta = packed_group_idx * AWQ_WARP_SIZE * 4 + threadIdx.x * 4;
        const float4* inputs_ptr = inputs + inputs_ptr_delta;
        // multiply 32 weights with 32 inputs
#pragma unroll
        for (int ic_0 = 0; ic_0 < 4; ic_0++) {
            // iterate over different packed_weights in this loop
            unsigned int current_packed_weight = packed_weights[ic_0];
            unsigned short packed_inputs[AWQ_PACK_FACTOR];
            // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8
            if (inputs_ptr_delta + ic_0 < IC / AWQ_PACK_FACTOR) {
                *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
#pragma unroll
                for (int ic_1 = 0; ic_1 < AWQ_PACK_FACTOR; ic_1++) {
                    // iterate over 8 numbers packed within each u32 number
                    float current_single_weight_fp = (float)(current_packed_weight & 0xF);
                    float dequantized_weight =
                        scaling_factor * (current_single_weight_fp - current_zeros);
                    psum += dequantized_weight * h2f(packed_inputs[ic_1]);
                    current_packed_weight = current_packed_weight >> 4;
                }
            }
        }
    }
    psum = awq_warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
        outputs[oc_idx] = f2h(psum);
    }
}
