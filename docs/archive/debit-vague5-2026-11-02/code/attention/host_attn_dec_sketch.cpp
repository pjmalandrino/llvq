// DRAFT, never compiled or run. Host probe harness for the attention kernel text, sketched for
// llvq-llm/tests/host_attn_dec.cpp (wave5-deep/attention.md, V1). Not repository code.
//
// Pattern: llvq-llm/tests/host_tv_q4_h.cpp and llvq-cuda/tests/host_tetra48.cpp. The Rust side
// (llvq-llm/tests/attn_dec_host.rs, to write) builds a fixture, compiles this file with
//   clang++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror
// (and once more with -fsanitize=undefined), pipes the fixture on stdin, and diffs every output
// against attn_ref_sketch.rs bit for bit. This file holds no expectations: it is a harness, not a
// reference. A missing clang++ fails and names the tool (hard rule 10).
//
// What is EXECUTED: the shuffle-free functions of the kernel text, which carry all the float
// arithmetic: ex2_fixed, dot8, tree16, tile_max_carry, tile_p, tile_accumulate, st_merge, deq_r1,
// quant_codes. What is only COMPILED: the two device schedules and the kernel body (shuffles,
// barriers, tickets), through compile-only stubs. host_shim.h's __shfl_xor_sync takes three
// arguments and returns its input; the kernel calls the 4-argument form with width 16, so the stub
// below is supplied here, as cuhcheck supplies its own. A 16-lane schedule is never emulated with
// it: the tree is pinned by tree16, executed, and by the card's load-time check against the replay.

#include "../../llvq-cuda/tests/host_shim.h"
#include "../../llvq-cuda/kernels/llvq_slot.cuh"   // u32

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// One rounding each under -ffp-contract=off, as the device intrinsics.
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
// floorf, roundf (half away from zero), fmaxf and fminf come from <cmath>, with the device's
// semantics on every finite input; a +-0 tie in fmaxf may differ, which ex2_fixed absorbs.

// Compile-only: never executed by this harness.
static inline float __shfl_xor_sync(unsigned, float v, int, int) { return v; }
#define __launch_bounds__(...)

// Exact binary16 conversions through clang's _Float16, the choice host_tv_q4_h.cpp made: exact
// widening and round-to-nearest-even narrowing, what cvt.f32.f16 and cvt.rn.f16.f32 do.
static inline float h2f(unsigned short h) { _Float16 v; std::memcpy(&v, &h, 2); return (float)v; }
static inline unsigned short f2h(float f) {
    _Float16 v = (_Float16)f; unsigned short r; std::memcpy(&r, &v, 2); return r;
}

Dim3 blockIdx, threadIdx, blockDim, gridDim;

#include "../../llvq-cuda/kernels/attn_dec.cu"

// ---------------------------------------------------------------- fixture I/O (little-endian)
//   in : u32 n_ex2,  f32[n_ex2]                       ex2_fixed inputs, edges included
//        u32 n_dot,  {f32 q[4][8], u16 k[8]}[n_dot]   dot8<4>
//        u32 n_tree, f32[16][n_tree]                  tree16
//        u32 n_tile, {St<4>, f32 sc[4][8], u32 nv, u16 V[8][128], u32 lane}[n_tile]
//                                                     tile_max_carry, tile_p, tile_accumulate
//        u32 n_merge, {St<4> a, St<4> b}[n_merge]      st_merge
//        u32 n_deq,  {u8 code, u16 s16, u16 mn16}[n_deq]
//        u32 n_q,    {f32 x[4], f32 mn, f32 mx}[n_q]  quant_codes
//   out: the same counts, each function's results in input order, raw bits. The Rust side compares
//        every output bit for bit, except a running maximum m, where +0 and -0 count as equal: the
//        one place where fmaxf on a tie may pick either zero, and every value derived from m is
//        still compared bit for bit.
template <class T> static T rd() { T v; if (std::fread(&v, sizeof v, 1, stdin) != 1) std::exit(2); return v; }
template <class T> static void wr(const T& v) { std::fwrite(&v, sizeof v, 1, stdout); }

int main() {
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) wr(ex2_fixed(rd<float>()));
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        float q[4][DPL]; unsigned short k[DPL]; float part[4];
        for (auto& r : q) for (float& x : r) x = rd<float>();
        for (auto& x : k) x = rd<unsigned short>();
        dot8<4>(q, k, part);
        for (float x : part) wr(x);
    }
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        float v[LANES]; for (float& x : v) x = rd<float>();
        wr(tree16(v));
    }
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        St<4> s; float sc[4][ATTN_TT];
        for (int g = 0; g < 4; ++g) { s.m[g] = rd<float>(); s.d[g] = rd<float>();
                                      for (float& x : s.a[g]) x = rd<float>(); }
        for (auto& r : sc) for (float& x : r) x = rd<float>();
        const int nv = (int)rd<u32>();
        std::vector<unsigned short> V(ATTN_TT * HD);
        for (auto& x : V) x = rd<unsigned short>();
        const int lane = (int)rd<u32>();
        for (int g = 0; g < 4; ++g) {
            float mn, carry, p[ATTN_TT];
            tile_max_carry(s.m[g], sc[g], nv, &mn, &carry);
            for (int j = 0; j < nv; ++j) p[j] = tile_p(sc[g][j], mn);
            tile_accumulate(&s.d[g], s.a[g], p, nv, carry, V.data(), lane);
            s.m[g] = mn;
            wr(mn); wr(carry); wr(s.d[g]); for (float x : s.a[g]) wr(x);
        }
    }
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        St<4> a, b;
        for (St<4>* t : {&a, &b})
            for (int g = 0; g < 4; ++g) { t->m[g] = rd<float>(); t->d[g] = rd<float>();
                                          for (float& x : t->a[g]) x = rd<float>(); }
        st_merge<4>(a, b);
        for (int g = 0; g < 4; ++g) { wr(a.m[g]); wr(a.d[g]); for (float x : a.a[g]) wr(x); }
    }
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        const uint8_t c = rd<uint8_t>(); const unsigned short s16 = rd<unsigned short>();
        const unsigned short mn16 = rd<unsigned short>();
        wr(deq_r1(c, s16, mn16));
    }
    for (u32 i = 0, n = rd<u32>(); i < n; ++i) {
        float x[4]; for (float& v : x) v = rd<float>();
        const float mn = rd<float>(), mx = rd<float>();
        uint8_t code[4]; unsigned short s16, mn16;
        quant_codes(x, mn, mx, code, &s16, &mn16);
        for (uint8_t c : code) wr(c);
        wr(s16); wr(mn16);
    }
    return 0;
}
