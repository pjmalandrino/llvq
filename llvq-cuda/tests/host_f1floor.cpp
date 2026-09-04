// Compiles f1floor.cu as host C++, in the order NVRTC sees it.
//
// Same purpose as tests/host_planes.cpp and the other host checks: catch every
// syntax and type error before one costs a billed job, and — for this file in
// particular — before it costs a 40-to-70 minute image rebuild. A single-thread
// driver reproduces neither `__syncthreads` nor a warp shuffle, so the kernels
// are compile-checked, not executed; `f1_mix` and `f1_fold` are scalar register
// arithmetic and ARE executed, against the Rust mirror in `src/f1floor.rs`.
//
//   in : u32 n, then u32[n] seeds
//   out: u32[n] f1_mix(seed), f32[n] f1_fold(seed, mix(seed), mix(mix(seed)))

#include "host_shim.h"
#include "../kernels/llvq_slot.cuh"  // u32, LLVQ_DIM

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE_BLOCKS 128u
#define TILE_COLS (TILE_BLOCKS * LLVQ_DIM)

// `__fmaf_rn` comes from host_shim.h; `warp_sum` is matvec.cu's, which the
// guard below skips because TILE_COLS is already defined — so it is supplied
// here, on the compile-only path (a single thread is not a warp).
static inline float warp_sum(float v) { return v; }

#include "../kernels/f1floor.cu"

// The two `extern __shared__` arrays the kernels declare, and the thread
// indices they read. None is executed here — the kernels are compile-checked
// only — but the linker still wants them.
float xs[TILE_COLS];
unsigned char smem[TILE_COLS * 4 + (64u << 10)];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

int main() {
    unsigned n = 0;
    if (std::fread(&n, 4, 1, stdin) != 1) {
        std::fprintf(stderr, "fixture tronquee: n\n");
        return 2;
    }
    std::vector<unsigned> seeds(n);
    if (n && std::fread(seeds.data(), 4, n, stdin) != n) {
        std::fprintf(stderr, "fixture tronquee: seeds\n");
        return 2;
    }
    std::vector<unsigned> mixed(n);
    std::vector<float> folded(n);
    for (unsigned i = 0; i < n; ++i) {
        unsigned h0 = f1_mix(seeds[i]);
        unsigned h1 = f1_mix(h0);
        unsigned h2 = f1_mix(h1);
        mixed[i] = h0;
        folded[i] = f1_fold(h0, h1, h2);
    }
    std::fwrite(mixed.data(), 4, n, stdout);
    std::fwrite(folded.data(), 4, n, stdout);
    return 0;
}
