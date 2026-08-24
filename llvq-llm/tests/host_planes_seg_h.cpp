// Compiles the served Planes14 translation unit, segmented arm included.
//
// ## What this file is for
//
// `fused_planes12x.rs::each_layout_names_its_own_translation_unit` checks the
// NVRTC source list with `unit.contains("void tv_planes_seg_h(")` — a
// **substring search**. It proves the file was concatenated; it cannot prove
// the result compiles. An `#include` NVRTC fails to resolve, a type that moved,
// a shim symbol that is not there: none of that is visible to a search, and all
// of it surfaces first in a billed job. `host_matvec.cpp` exists because that
// has happened.
//
// So this harness compiles the unit in the **exact order** `fused_cuda`
// concatenates for the `Planes14` layout — llvq_slot.cuh, matvec.cu,
// llvq_planes.cuh, planes.cu, tv_planes_h.cu, tv_planes_seg_h.cu — minus the
// rotation pair, which shares no symbol with any of it. Every `#ifndef` guard
// therefore takes the same branch here as on the device.
//
// ## Compile-only, and why that is still worth three seconds
//
// `tv_planes_seg_h` is never executed here. A single-threaded driver
// reproduces neither `__syncthreads` nor the `warp_sum` shuffle, and `f2h` is a
// PTX instruction the shim stubs. What running it would produce is not a
// reference, so this file holds no expectations.
//
// What the compiler does check is everything a reader would have to check by
// eye: that `gs_off` is the type `planes_dot`'s caller can offset with, that
// the argument list matches the launcher's, that no identifier drifted when the
// kernel was derived from `tv_planes_h`. `-Werror` with `-Wall -Wextra` turns
// an unused parameter or a narrowing conversion into a failure here rather than
// into a silent difference on the card.
//
// The host half of the lot — the spliced stream and the `gs_off` table, which
// is where the silent failures live — is proved in `tests/fused_segment.rs`,
// not here.

#include "../../llvq-cuda/tests/host_shim.h"

#include "../../llvq-cuda/kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../../llvq-cuda/kernels/matvec.cu"
#include "../../llvq-cuda/kernels/llvq_planes.cuh"
#include "../../llvq-cuda/kernels/planes.cu"
#include "../kernels/tv_planes_h.cu"
#include "../kernels/tv_planes_seg_h.cu"

#include <cstdio>

// The `extern __shared__` array and thread indices the (never-executed)
// kernels reference. `__shared__` expands to nothing under the shim.
float xs[TILE_BLOCKS * LLVQ_DIM];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

// Taking the address of both kernels keeps the linker from discarding them and
// makes "it compiled" mean "both entry points exist with these signatures".
int main() {
    const void* unfused = reinterpret_cast<const void*>(&tv_planes_h);
    const void* fused = reinterpret_cast<const void*>(&tv_planes_seg_h);
    std::printf("%d\n", unfused != nullptr && fused != nullptr && unfused != fused);
    return 0;
}
