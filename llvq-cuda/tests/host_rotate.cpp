// Executable harness for the rotation kernel.
//
// Unlike `host_matvec.cpp`, this one *runs*. `rot_apply` uses shared memory
// and barriers, but it reads its thread identity from arguments the shim can
// set, so a driver at `blockDim.x == 1` executes the same text with vacuous
// barriers and produces the answer the GPU must produce. Nothing is
// reimplemented — the file under test is `#include`d.
//
// Two runs, on purpose:
//
//   * `rot_apply` at one thread — the real kernel, real orchestration, so a
//     phase called in the wrong order or a missing scale shows up here;
//   * the phases replayed at 7, 32 and 256 threads — same sequence, different
//     work split. The Walsh–Hadamard addressing is the one place where a
//     thread-count-dependent bug can hide (a stage that pairs slots two
//     threads both own), and comparing the widths against each other catches
//     it without a GPU. 7 is in there because it divides nothing.
//
// The replay duplicates the kernel's phase order, which is a real risk of
// drift — so the test compares the replay against `rot_apply` as well as
// against Rust. If the duplication ever diverges, that comparison fails.
//
// Fixture on stdin, results on stdout, both little-endian:
//
//   in : u32 n, u32 m, u32 k, f32 inv, u32 words[(n+31)/32],
//        f32 small[KMAX*KMAX], u16 xin[n]
//   out: f32 out[n] × 4   — widths 1, 7, 32, 256 in that order

#include "host_shim.h"

#include "../kernels/llvq_rot.cuh"
#include "../kernels/rotate.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The `extern __shared__` the kernel declares. `__shared__` expands to
// nothing under the shim, so this is an ordinary definition of it. Sized for
// the widest Qwen3-4B projection with room to spare; the fixture is refused
// above it rather than overrunning.
static const unsigned SMAX = 16384u;
float s[SMAX];

Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

// The kernel's phase order, replayed at an arbitrary thread count.
//
// `__syncthreads()` is a no-op here; the barrier is the loop itself, which
// finishes every thread's share of a phase before the next phase starts.
// That is exactly what a barrier means, so this models the device faithfully
// for any width.
static void replay(const unsigned short* xin,
                   const u32* signbits,
                   const float* small,
                   float* xout,
                   u32 n,
                   u32 m,
                   u32 k,
                   float inv,
                   u32 nthreads)
{
    for (u32 t = 0; t < nthreads; ++t) rot_load(xin, signbits, s, n, 0u, t, nthreads);
    for (u32 len = 1u; len < m; len <<= 1)
        for (u32 t = 0; t < nthreads; ++t) rot_wht_step(s, n, len, t, nthreads);
    if (k == 1u) {
        for (u32 t = 0; t < nthreads; ++t) rot_scale_out(s, xout, n, inv, t, nthreads);
    } else {
        for (u32 t = 0; t < nthreads; ++t) rot_mix(s, small, xout, m, k, inv, t, nthreads);
    }
}

template <typename T>
static void take(const std::vector<unsigned char>& buf, size_t& at, T* dst, size_t count)
{
    if (at + count * sizeof(T) > buf.size()) {
        std::fprintf(stderr, "fixture truncated\n");
        std::exit(2);
    }
    __builtin_memcpy(dst, buf.data() + at, count * sizeof(T));
    at += count * sizeof(T);
}

int main()
{
    std::vector<unsigned char> buf;
    unsigned char chunk[65536];
    size_t got;
    while ((got = std::fread(chunk, 1, sizeof chunk, stdin)) > 0)
        buf.insert(buf.end(), chunk, chunk + got);

    size_t at = 0;
    u32 n, m, k;
    float inv;
    take(buf, at, &n, 1);
    take(buf, at, &m, 1);
    take(buf, at, &k, 1);
    take(buf, at, &inv, 1);
    if (n > SMAX || k > LLVQ_ROT_KMAX || m == 0u || n != k * m) {
        std::fprintf(stderr, "fixture out of range: n=%u m=%u k=%u\n", n, m, k);
        return 2;
    }

    std::vector<u32> signbits((n + 31u) / 32u);
    std::vector<float> small(LLVQ_ROT_KMAX * LLVQ_ROT_KMAX);
    std::vector<unsigned short> xin(n);
    take(buf, at, signbits.data(), signbits.size());
    take(buf, at, small.data(), small.size());
    take(buf, at, xin.data(), xin.size());

    std::vector<float> out(n);

    // The real kernel, one thread.
    blockDim.x = 1;
    threadIdx.x = 0;
    rot_apply(xin.data(), signbits.data(), small.data(), out.data(), n, m, k, inv, 0u);
    std::fwrite(out.data(), sizeof(float), n, stdout);

    for (u32 w : {7u, 32u, 256u}) {
        std::fill(out.begin(), out.end(), 0.0f);
        replay(xin.data(), signbits.data(), small.data(), out.data(), n, m, k, inv, w);
        std::fwrite(out.data(), sizeof(float), n, stdout);
    }
    return 0;
}
