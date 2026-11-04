// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §5, step V1). Not part of the repository.
//
// Extension of llvq-cuda/tests/host_rotate.cpp. Two facts about today's file
// drive it:
//   * host_rotate.cpp:45 sizes the shared stand-in at SMAX = 16,384 floats and
//     :104 refuses any n above it, so the host test cannot reach the served
//     14B width (17,408) nor the 32B width (25,600). Raised to 32,768 here.
//   * the CASES of rotation_matches_rust.rs:42-51 stop at 12,288.
//
// What this driver adds, for each fixture (n, m, k, inv, signs, small, xin):
//   ref   = rot_apply at one thread (the existing first output);
//   R0    = rot_apply_dev at one thread, on a malloc'd scratch;
//   R1    = rot_apply_r replayed at 1, 7, 32, 256 and 1,024 threads;
//   R1w   = rot_apply_rw replayed at the same widths, gb in {1, 3, 8};
// and requires every one of them to equal `ref` BIT FOR BIT (memcmp), at every
// width. The Rust side then writes the fixture the card job compares against
// (llvq-cuda/tests/fixtures/rot-hashes.txt), one line per (n, seed):
//     <n> <seed hex> <sha256 of the input> <sha256 of ref>
// where the input bytes are, in order and little-endian, the n f16 bits, the
// sign words, the KMAX*KMAX padded `small` floats and `inv`, and ref's bytes
// are its n f32 outputs. Both hashes use the portable sha256_hex that rotbench
// also uses (moved out of the Linux-only gpu.rs). rotbench COMPILES this file
// in with include_str! (the card image has no repository tree), so the
// fixture must be committed before the image is built; a missing file fails
// the free build, not a billed job. The fixture draw is the one rotbench
// already makes (SplitMix64(seed ^ 0xa5a5_a5a5), Rotation::new(n, seed)), so
// the input hash is the check that the two draws did not drift apart.
//
// Edited files stay outside the served bytes: this driver includes
// rotate_wide.cu, a unit of its own; llvq_rot.cuh and rotate.cu are read, not
// changed (llvq_rot.cuh's wrong comment about 25,600 is corrected in the docs,
// since the file is part of the served unit whose sha256 every log prints).
//
// Replays follow the kernels' phase order with a loop over thread ids per
// phase, which is what a barrier means (host_rotate.cpp:52-57 makes the same
// argument). A replay that drifted from its kernel is caught because R1 and
// R1w are ALSO run through their real entry points at blockDim.x == 1.

#include "host_shim.h"
#include "../kernels/llvq_rot.cuh"
#include "../kernels/rotate.cu"
#include "../kernels/rotate_wide.cu"   // the draft of code/bigmodel/rotate_wide.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const unsigned SMAX = 32768u;   // was 16,384: covers 25,600 and 31,744
float s[SMAX];                          // the `extern __shared__` stand-in

Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

static void replay_r1(const unsigned short* xin, const u32* sb, const float* small, float* out,
                      u32 n, u32 m, u32 k, float inv, u32 nt)
{
    for (u32 t = 0; t < nt; ++t) rot_r_phase1(xin, sb, s, n, 0u, 0u, t, nt);
    u32 lg = 2u;
    for (; (4u << lg) <= m; lg += 2u)
        for (u32 t = 0; t < nt; ++t) rot_r_pass4(s, n, lg, t, nt);
    if ((1u << lg) < m)
        for (u32 t = 0; t < nt; ++t) rot_r_pass2(s, n, lg, t, nt);
    for (u32 t = 0; t < nt; ++t) rot_r_mix(s, small, out, n, m, k, inv, t, nt);
}

static void replay_r1w(const unsigned short* xin, const u32* sb, const float* small, float* out,
                       float* scratch, u32 n, u32 m, u32 k, float inv, u32 gb, u32 nt)
{
    for (u32 g0 = 0; g0 < k; g0 += gb) {
        u32 ng = (k - g0 < gb) ? k - g0 : gb, nb = ng * m, base = g0 * m;
        for (u32 t = 0; t < nt; ++t) rot_r_phase1(xin, sb, s, nb, base, 0u, t, nt);
        u32 lg = 2u;
        for (; (4u << lg) <= m; lg += 2u)
            for (u32 t = 0; t < nt; ++t) rot_r_pass4(s, nb, lg, t, nt);
        if ((1u << lg) < m)
            for (u32 t = 0; t < nt; ++t) rot_r_pass2(s, nb, lg, t, nt);
        for (u32 i = 0; i < nb; ++i) scratch[base + i] = s[i];
    }
    for (u32 t = 0; t < nt; ++t) rot_r_mix(scratch, small, out, n, m, k, inv, t, nt);
}

// main(): read the fixture as host_rotate.cpp does (n, m, k, inv, signs,
// small, xin; refuse n > SMAX or k > KMAX or n != k * m or m < 4), then:
//
//   ref  <- rot_apply(...) at blockDim.x = 1              (write it to stdout)
//   dev  <- rot_apply_dev(..., scratch) at blockDim.x = 1 (memcmp == ref)
//   r1   <- rot_apply_r(...) at blockDim.x = 1            (memcmp == ref)
//   rw   <- rot_apply_rw(..., scratch, gb = 8) at 1       (memcmp == ref)
//   for nt in {1, 7, 32, 256, 1024}: replay_r1, replay_r1w (gb 1, 3, 8)
//                                                         (memcmp == ref)
//   rows twins at 1, 3 and 4 rows with row_stride = n + 8 (memcmp per row)
//
// and print one status byte per comparison, so the Rust test names the
// first (kernel, width, thread count) that differs instead of a bare failure.
//
// Mutants the suite must kill (bigmodel.md §6):
//   M1 rot_r_pass4: s[j0 + 2len] = cd0 - ab0 (subtraction reversed)
//   M2 rot_r_stages: pass2 run before the pass4 loop (stage order)
//   M3 rot_r_mix: KB chosen as 8 for k = 17 (chain cut below k)
//   M4 rot_r_phase1: sign word signbits[gi0 >> 4] (wrong word)
//   M5 rot_apply_rw: base = g0 * gb (batch offset)
//   M6 rot_r_mix_kb: inv applied to acc after the chain instead of to col
//   M7 rows twin: output at r * row_stride instead of r * n
// A missing __syncthreads() in R0, R1 or R1w is invisible here (barriers are
// vacuous at one thread): the card job's hash gate is the only net for it.
