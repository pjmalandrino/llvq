// Drives `tail_dot_h` — the KeepExact tail epilogue of the three f16-storing
// entry points — on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against an
// f64 reference and against `half::f16`, so this file holds no expectations —
// it is a harness, not a reference.
//
//   in : u32 d_out, u32 tail_w
//        u16[d_out*tail_w] tail   (binary16 bits, row-major)
//        f32[tail_w]       xt     (the tail columns of the activation)
//   out: f32[d_out] tv
//
// ## What this file proves, and what it cannot
//
// `tail_dot_h` has no barrier, no shuffle and no atomic — it is the `lane == 0`
// epilogue, scalar register arithmetic over a `u32` index. So it is
// **executed** here, exactly as llvq-llm/tests/host_planes12x_h.cpp executes
// `planes12x_row_correction`. Three things follow, and they are the whole
// reason this harness exists:
//
//   * the widening is checked. `h2f`'s host branch used to return 0 (matvec.cu
//     licensed that with "nothing executes it"); this file is what stopped
//     that from being true, so the branch is now the exact software widening
//     and a stub would fail here on the first non-zero tail;
//   * the *indexing* is checked. `tail[row * tail_w + i]` against
//     `xt[i]` is row-major over the tail and offset-relative over the
//     activation, and a transposed or absolutely-indexed variant returns
//     finite, plausible, wrong numbers on the card and nothing else would
//     catch it here;
//   * the association is checked to be multiply-then-add in f32, not f64 and
//     not fma, because the Rust side compares against an f32 replay as well as
//     an f64 reference.
//
// What it cannot prove: that the three **kernels** call it correctly. Their
// bodies — grid, barriers, `warp_sum`, the f16 store — are compile-only here,
// as everywhere on a machine with no card. Including all three below still
// costs two seconds to catch every syntax and type error in the new signature,
// which is the difference between finding a typo here and finding it in a
// billed job.
//
// The translation unit is assembled in the order `fused_cuda` concatenates for
// NVRTC, minus the rotation pair (it shares no symbol with any of this): every
// `#ifndef` guard therefore takes the same branch here as on the device.

#include "../../llvq-cuda/tests/host_shim.h"

#include "../../llvq-cuda/kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../../llvq-cuda/kernels/matvec.cu"
#include "../../llvq-cuda/kernels/llvq_planes.cuh"
#include "../../llvq-cuda/kernels/planes.cu"
#include "../kernels/tv_planes_h.cu"
#include "../../llvq-cuda/kernels/llvq_planes12.cuh"
#include "../kernels/tv_planes12x_h.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

// The `extern __shared__` arrays and thread indices the (never-executed)
// kernels reference. `__shared__` expands to nothing under the shim.
float xs[TILE_BLOCKS * LLVQ_DIM];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "fixture tronquee: %s\n", what);
        std::exit(2);
    }
    return v;
}

int main() {
    std::FILE* in = stdin;
    auto hdr = read_n<unsigned>(in, 2, "en-tete");
    unsigned d_out = hdr[0], tail_w = hdr[1];

    auto tail = read_n<unsigned short>(
        in, static_cast<std::size_t>(d_out) * tail_w, "tail");
    auto xt = read_n<float>(in, tail_w, "xt");

    std::vector<float> tv(d_out, 0.0f);
    for (unsigned row = 0; row < d_out; ++row)
        tv[row] = tail_dot_h(tail.data(), xt.data(), row, tail_w);

    std::fwrite(tv.data(), sizeof(float), tv.size(), stdout);
    return 0;
}
