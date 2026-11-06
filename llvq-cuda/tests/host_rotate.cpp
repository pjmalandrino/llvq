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
//   * the phases replayed at 7, 32, 256 and 1,024 threads — same sequence,
//     different work split. The Walsh–Hadamard addressing is the one place where a
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
//   out: f32 out[n] × 5   — widths 1, 7, 32, 256, 1024 in that order
//        f32 worst[3]     — rot_apply_rows against rot_apply, at 1, 3 and 4
//                           rows; every one of them must be exactly 0
//        f32 dev[n]       — rot_apply_dev at one thread (rotate_wide.cu)
//        u32 flags[5]     — 0 when equal byte for byte, 1 otherwise:
//                           rot_apply_dev at x_off 13 against rot_apply;
//                           rot_apply_rows_dev at 1, 3 and 4 rows, each row
//                           against rot_apply on that row; and the scratch
//                           past the 4 rows left untouched
//
// The device-staged kernels of `rotate_wide.cu` exist for widths whose f32
// staging passes the card's shared-memory ceiling (25,600 for Qwen3-32B). The
// host has no such ceiling, so `rot_apply` itself runs here at those widths
// and is the reference the new kernels must equal, bit for bit.

#include "host_shim.h"

#include "../kernels/llvq_rot.cuh"
#include "../kernels/rotate.cu"
#include "../kernels/rotate_wide.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The `extern __shared__` the kernel declares. `__shared__` expands to
// nothing under the shim, so this is an ordinary definition of it. Sized for
// the widest width the fixtures carry, 31,744 (k = 31), so the served 14B
// width (17,408) and the Qwen3-32B one (25,600) run here; the fixture is
// refused above it rather than overrunning. It was 16,384 until 2026-11-04.
static const unsigned SMAX = 32768u;
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

    for (u32 w : {7u, 32u, 256u, 1024u}) {
        std::fill(out.begin(), out.end(), 0.0f);
        replay(xin.data(), signbits.data(), small.data(), out.data(), n, m, k, inv, w);
        std::fwrite(out.data(), sizeof(float), n, stdout);
    }

    // ---- rot_apply_rows: the same kernel, the rows in the grid -------------
    //
    // The property, and it is the only one that needs a new fixture: row `r`
    // of a batched rotation is the single-row rotation of row `r` of the
    // input. Both sides are the REAL kernels — `rot_apply_rows` driven at
    // `blockIdx.x = r`, `rot_apply` on that row alone — so this is not a
    // replay and cannot drift from one.
    //
    // 🚨 The input rows are `n + STRIDE_PAD` apart, deliberately NOT `n`. The
    // kernel reads the input at `x_off + r * row_stride` and writes the output
    // at `r * n`, and those are two different quantities that are equal in
    // every model this repository serves. A kernel that used `n` for both, or
    // `row_stride` for both, is right on every real shape and wrong here.
    //
    // The rows are distinct — row `r` is the fixture cyclically shifted by `r`
    // — so reading the wrong row is a wrong answer rather than a lucky one.
    {
        const u32 STRIDE_PAD = 8u;
        const u32 stride = n + STRIDE_PAD;
        for (u32 rows : {1u, 3u, 4u}) {
            std::vector<unsigned short> batch((size_t)stride * rows, 0u);
            for (u32 r = 0; r < rows; ++r)
                for (u32 i = 0; i < n; ++i)
                    batch[(size_t)r * stride + i] = xin[(i + r) % n];

            std::vector<float> many((size_t)n * rows, 0.0f);
            blockDim.x = 1;
            threadIdx.x = 0;
            for (u32 r = 0; r < rows; ++r) {
                blockIdx.x = r;
                rot_apply_rows(batch.data(), signbits.data(), small.data(), many.data(),
                               n, m, k, inv, 0u, stride);
            }
            blockIdx.x = 0;

            float worst = 0.0f;
            std::vector<float> one(n);
            for (u32 r = 0; r < rows; ++r) {
                std::fill(one.begin(), one.end(), 0.0f);
                rot_apply(batch.data() + (size_t)r * stride, signbits.data(), small.data(),
                          one.data(), n, m, k, inv, 0u);
                for (u32 i = 0; i < n; ++i) {
                    float d = many[(size_t)r * n + i] - one[i];
                    worst = std::max(worst, d < 0.0f ? -d : d);
                }
            }
            std::fwrite(&worst, sizeof(float), 1, stdout);
        }
    }

    // ---- rotate_wide.cu: the same helpers, staged in device memory --------
    //
    // `ref` is `rot_apply` at one thread, the first vector written above. The
    // scratch is a separate allocation sized as the runtime sizes it
    // (`wide::rot_scratch_floats`) and filled with a NaN sentinel first, so a
    // kernel that read a slot it had not written would carry the NaN into its
    // output instead of a stale but plausible value.
    {
        std::vector<float> ref(n);
        blockDim.x = 1;
        threadIdx.x = 0;
        blockIdx.x = 0;
        rot_apply(xin.data(), signbits.data(), small.data(), ref.data(), n, m, k, inv, 0u);

        const u32 ROWS_MAX = 4u;
        const size_t tail = (size_t)(LLVQ_ROT_KMAX - k) * m;
        const size_t scratch_len = (size_t)ROWS_MAX * n + tail;
        const u32 SENTINEL = 0x7fc0deadu;
        std::vector<float> scratch(scratch_len);
        auto poison = [&]() {
            for (auto& v : scratch) __builtin_memcpy(&v, &SENTINEL, 4);
        };

        // The real entry point, one thread.
        poison();
        std::vector<float> dev(n, 0.0f);
        rot_apply_dev(xin.data(), signbits.data(), small.data(), dev.data(), scratch.data(),
                      n, m, k, inv, 0u);
        std::fwrite(dev.data(), sizeof(float), n, stdout);

        u32 flags[5];
        // `x_off`: the same activation 13 elements into a larger buffer.
        {
            const u32 OFF = 13u;
            std::vector<unsigned short> shifted(n + OFF, 0x3c00u);
            std::copy(xin.begin(), xin.end(), shifted.begin() + OFF);
            poison();
            std::vector<float> got(n, 0.0f);
            rot_apply_dev(shifted.data(), signbits.data(), small.data(), got.data(),
                          scratch.data(), n, m, k, inv, OFF);
            flags[0] = std::memcmp(got.data(), ref.data(), (size_t)n * 4) == 0 ? 0u : 1u;
        }
        // The rows twin, rows `n + 8` apart for the reason the rows test above
        // gives, each row against `rot_apply` of that row alone.
        {
            const u32 STRIDE_PAD = 8u;
            const u32 stride = n + STRIDE_PAD;
            int at = 1;
            for (u32 rows : {1u, 3u, 4u}) {
                std::vector<unsigned short> batch((size_t)stride * rows, 0u);
                for (u32 r = 0; r < rows; ++r)
                    for (u32 i = 0; i < n; ++i)
                        batch[(size_t)r * stride + i] = xin[(i + r) % n];
                poison();
                std::vector<float> many((size_t)n * rows, 0.0f);
                for (u32 r = 0; r < rows; ++r) {
                    blockIdx.x = r;
                    rot_apply_rows_dev(batch.data(), signbits.data(), small.data(), many.data(),
                                       scratch.data(), n, m, k, inv, 0u, stride);
                }
                blockIdx.x = 0;
                bool same = true;
                std::vector<float> one(n);
                for (u32 r = 0; r < rows; ++r) {
                    rot_apply(batch.data() + (size_t)r * stride, signbits.data(), small.data(),
                              one.data(), n, m, k, inv, 0u);
                    if (std::memcmp(many.data() + (size_t)r * n, one.data(), (size_t)n * 4) != 0)
                        same = false;
                }
                flags[at++] = same ? 0u : 1u;
            }
            // After the 4-row run: nothing past the rows' own staging was
            // written. A row staged at `r * row_stride` instead of `r * n`
            // returns the right answer and lands outside its slot.
            bool clean = true;
            for (size_t i = (size_t)ROWS_MAX * n; i < scratch_len; ++i) {
                u32 b;
                __builtin_memcpy(&b, &scratch[i], 4);
                if (b != SENTINEL) clean = false;
            }
            flags[4] = clean ? 0u : 1u;
        }
        std::fwrite(flags, sizeof(u32), 5, stdout);
    }
    return 0;
}
