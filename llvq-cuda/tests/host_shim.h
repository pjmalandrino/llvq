// Just enough CUDA to compile the kernels as ordinary C++ on the dev machine.
//
// `slot_dot` contains no warp primitive, no shared memory, no atomic — it is
// scalar arithmetic over registers. So the one thing that cannot be checked
// without a GPU is the *hardware* behaviour, and the one thing that can is the
// part where every bug actually lives: the bit arithmetic. This header buys
// that check for two seconds on a Mac instead of ten minutes and twenty cents
// on a rented card.
//
// The kernels are turned into plain functions and the block/thread indices
// into mutable globals the driver loop sets, so the *same text* that NVRTC
// compiles is the text exercised here. A separate host reimplementation would
// prove nothing: it could share a bug with the kernel, or drift from it.

#ifndef LLVQ_HOST_SHIM_H
#define LLVQ_HOST_SHIM_H

#include <cmath>

#define __device__
#define __global__
#define __forceinline__ inline
#ifndef __restrict__
#define __restrict__
#endif

// Round-to-nearest fused multiply-add. `std::fma` is IEEE-754 correctly
// rounded and single-rounded, which is what `__fmaf_rn` means — so the host
// and the device agree on this operation, and a difference in the final float
// can only come from something else.
static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }

// Enough to *type-check* the fused matvec. These are not semantics: a
// single-threaded driver cannot reproduce a barrier or a warp shuffle, and
// the matvec test below is compile-only for that reason. What it does catch
// is every syntax and type error — which is what a fifty-minute image rebuild
// would otherwise charge for.
#define __shared__
#define LLVQ_HOST_BUILD 1
static inline void __syncthreads() {}
static inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
struct uint4 { unsigned x, y, z, w; };

struct Dim3 { unsigned x, y, z; };
extern Dim3 blockIdx;
extern Dim3 threadIdx;
extern Dim3 blockDim;

#endif  // LLVQ_HOST_SHIM_H
