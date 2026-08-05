// Compile-only harness for the fused matvec.
//
// A single-threaded driver cannot reproduce a barrier or a warp shuffle, so
// this file is never executed — `tv_slot` and `tv_f16` are validated on the
// card. What it does catch is every syntax and type error in the two kernels,
// before they cost a fifty-minute image rebuild. It has already earned that
// once: `preflight.cu` shipped an `#include` that NVRTC could not resolve,
// and only a billed job found it.
#include "host_shim.h"
#include "../kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../kernels/matvec.cu"
// The floor probes, same treatment: they use `warp_sum` and `TILE_BLOCKS` from
// the file above, so they are included after it — the same order the host
// concatenates them in for NVRTC.
#include "../kernels/llvq_floor.cuh"

// The `extern __shared__` both kernels declare. `__shared__` expands to
// nothing under the shim, so this is an ordinary external array.
float xs[TILE_BLOCKS * LLVQ_DIM];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

int main() { return 0; }
