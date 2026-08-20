// QTIP 2-bit trellis GEMV — the shims that give its kernel a reachable name.
//
// ============================================================================
// WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT.
//
// It contains NO upstream code. Every other competitor arm in this directory
// carries its kernel (`awq_gemv.cu` is a port of MIT-licensed sources, header
// preserved). QTIP is **GPL v3**, and this workspace is MIT OR Apache-2.0, so
// its kernel is never committed here: `ops/fetch-qtip.sh` fetches it at job
// time at a pinned commit, verifies two sha256, strips four dead lines, and
// extracts a device-only `qtip_device.cuh`. That directory reaches the bench
// through `LLVQ_KERNEL_DIR`, which for this arm is the *only* path — there is
// no `include_str!` fallback that could silently succeed with nothing to
// measure. Rationale in full: `docs/qtip-provenance.md`.
//
// ============================================================================
// WHY THE SHIMS EXIST.
//
// Upstream declares one kernel, templated on seven compile-time parameters:
//
//     template <uint32_t L, uint32_t S, uint32_t R, uint32_t V,
//               uint32_t M, uint32_t N, uint32_t K>
//     __global__ static void kernel_decompress_matvec(float*, const uint32_t*,
//                                                     const half2*, const half2*)
//
// cudarc resolves kernels BY NAME (`Cuda::func`, `src/gpu.rs:357`), and a
// `__global__ static` template instantiation has no name a caller can spell:
// `static` gives it internal linkage and the mangling is a compiler detail we
// refuse to hard-code. The fetch script therefore rewrites that one
// declaration to `__device__ static`, and the shims below — ours, named,
// `extern "C"` — instantiate it. Nothing else about the kernel changes; the
// `__launch_bounds__(1024, 1)` that upstream put on the template simply moves
// here, onto the functions that are now the entry points.
//
// M and K being template parameters is why there is one shim per projection
// shape rather than one kernel with runtime dimensions.
//
// ============================================================================
// THE PARAMETERS, AND WHY THESE SHAPES ARE ADMISSIBLE.
//
// <L=16, S=9, R=2, V=1, M=d_out, N=1, K=d_in> is upstream's own 2-bit
// configuration (their `test.cu` calls `<16, 9, 2, 1, M, N, K>`): a 16-bit
// shift register, a 9-bit codebook index, 2 bits per weight, and V=1 meaning
// log2 of the two weights decoded per state.
//
// Their static_asserts were checked by hand against the five distinct shapes
// of Qwen3-4B before this file was written, because a violated one is a
// compile error on a rented card:
//
//   shape (M, K)   tileCountM = M/16   tileCountM % 2   tileCountK = K/16   (tileCountK % 128) % 4
//   (4096, 2560)          256                0                160            32 % 4 = 0
//   (1024, 2560)           64                0                160            32 % 4 = 0
//   (2560, 4096)          160                0                256             0 % 4 = 0
//   (9728, 2560)          608                0                160            32 % 4 = 0
//   (2560, 9728)          160                0                608            96 % 4 = 0
//
// All five pass, and M % 16 == 0 and K % 16 == 0 hold throughout. The seven
// projections of the model collapse to these five: k_proj and v_proj share a
// shape, and so do gate_proj and up_proj.
//
// ⚠️ The 64 KiB codebook lives in DYNAMIC shared memory (upstream sizes it
// `1 << (S + 5 + V + 1)`), which is above the 48 KiB per-block default. The
// launcher must opt in — `Cuda::func_dynamic_shared(name, 65536)` — exactly as
// the rotation kernel does. A shim compiles fine without that opt-in and fails
// at launch, so the requirement lives in the host code, not here.
// ============================================================================

// The repository's include convention (see `planes.cu`): under NVRTC the host
// concatenates the sources and there is no filesystem, so the guard is already
// defined by the part concatenated before this one and the include is skipped.
// It resolves only under a host syntax check, where a filesystem exists.
// Order is the caller's contract: qtip_device.cuh, then this file.
#ifndef QTIP_DEVICE_CUH
#include "qtip_device.cuh"
#endif

/// One entry point per projection shape.
///
/// The body is a single instantiation: no work of ours runs on the device, so
/// a difference between this arm and upstream's own numbers cannot come from
/// this file. That is the point of keeping it this thin.
#define QTIP_WRAP(DOUT, DIN)                                                   \
    extern "C" __global__ void __launch_bounds__(1024, 1)                      \
    qtip_mv_##DOUT##x##DIN(float *__restrict__ out,                            \
                           const uint32_t *__restrict__ compressed,            \
                           const half2 *__restrict__ x,                        \
                           const half2 *__restrict__ codebook)                 \
    {                                                                          \
        kernel_decompress_matvec<16, 9, 2, 1, DOUT, 1, DIN>(                   \
            out, compressed, x, codebook);                                     \
    }

QTIP_WRAP(4096, 2560)   // q_proj
QTIP_WRAP(1024, 2560)   // k_proj, v_proj
QTIP_WRAP(2560, 4096)   // o_proj
QTIP_WRAP(9728, 2560)   // gate_proj, up_proj
QTIP_WRAP(2560, 9728)   // down_proj

#undef QTIP_WRAP
