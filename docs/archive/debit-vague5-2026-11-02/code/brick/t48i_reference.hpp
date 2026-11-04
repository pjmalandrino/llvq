// DRAFT, never compiled or run. Revised 2026-11-02 after review: the S grid now spans m = 1..26
// (52 nonzero scales), takes E_S over |S| with a signed S_int, and refuses a matrix by name instead
// of asserting; the epilogue handles e_lo + E_S < -126 in two exact steps.
//
// Host C++ reference of the brick's canonical integer semantics (wave5-deep/brick.md §3.1).
// Intended home: llvq-cuda/kernels/llvq_t48i.cuh (host-compilable device functions, through
// tests/host_shim.h) and a Rust twin in llvq-artifact. Every device path (dp4a k = 1, IMMA k >= 2,
// prefill rows, persistent kernel) must return these bits. Build like host_tetra48.cpp:
// clang++ -std=c++17 -O2 -ffp-contract=off.
//
// Conventions:
//   * a "block" is 24 coordinates; a slice is 16 blocks (384 coordinates) from block 0;
//   * the f16 tail columns are not quantized and stay f32;
//   * MMA column k <-> trio position: k = 4s + j -> 8s + j, k = 16 + 4s + j -> 8s + 4 + j (s < 3);
//     k in {12..15, 28..31} is zero in the activation (A carries S_int at 12..15);
//   * trio position p <-> natural coordinate TETRA48_ORDER[p] (llvq_tetra48.cuh:70-73).

#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

namespace t48i {

constexpr int DIM = 24, SLICE_BLOCKS = 16, CAP = 3;  // exponent spread cap, in binades
constexpr int QMAX = 32767;                          // symmetric int16 range

static const unsigned char ORDER[24] = {0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14,
                                        16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23};

// k (0..31) -> trio position, or -1 for the zero columns.
inline int trio_of_k(int k) {
    int hi = k >> 4, s = (k & 15) >> 2, j = k & 3;
    if (s == 3) return -1;
    return 8 * s + 4 * hi + j;
}

// Binary exponent of a positive finite float, exactly, from its bits (subnormals handled).
inline int exp2_of(float a) {
    uint32_t b; std::memcpy(&b, &a, 4);
    int e = int((b >> 23) & 0xff);
    if (e == 0) { int lz = __builtin_clz(b & 0x7fffff) - 9; return -127 - lz; }
    return e - 127;
}

inline float pow2f(int e) {  // exact 2^e for e in [-126, 127]
    assert(e >= -126 && e <= 127);
    uint32_t b = uint32_t(e + 127) << 23; float f; std::memcpy(&f, &b, 4); return f;
}

// One token, quantized. Planes are stored per block as 32 bytes indexed by MMA k here; the device
// stores the same bytes as four 8-byte slots (slot t = k 4t..4t+3, then 16+4t..19+4t), so the same
// record feeds the B fragment (8-byte slot t = k 4t..4t+3, then 16+4t..19+4t) and the dp4a lane
// (eight u32: q0' q1' q2' q3' q4' q5' 0 0 in quad order).
struct QToken {
    std::vector<int8_t> hi;    // nblocks * 32, MMA k order
    std::vector<uint8_t> lo;   // nblocks * 32
    std::vector<int> shift;    // per slice, s = e_slice - e_lo in [0, CAP]
    int e_lo = 0;
    bool nonfinite = false;
};

inline QToken quantize(const float* x, int nblocks) {
    QToken q; q.hi.assign(32 * nblocks, 0); q.lo.assign(32 * nblocks, 0);
    int nslices = (nblocks + SLICE_BLOCKS - 1) / SLICE_BLOCKS;
    std::vector<int> e(nslices, INT32_MIN);
    for (int s = 0; s < nslices; ++s) {
        float amax = 0.0f;
        for (int i = 24 * SLICE_BLOCKS * s; i < 24 * std::min(nblocks, SLICE_BLOCKS * (s + 1)); ++i) {
            if (!std::isfinite(x[i])) q.nonfinite = true;
            amax = std::fmax(amax, std::fabs(x[i]));
        }
        if (amax > 0.0f) e[s] = exp2_of(amax) - 14;  // amax * 2^-e in [2^14, 2^15)
    }
    int e_hi = INT32_MIN;
    for (int v : e) e_hi = std::max(e_hi, v);
    if (e_hi == INT32_MIN) e_hi = 0;                 // all-zero token
    q.e_lo = e_hi - CAP;
    q.shift.resize(nslices);
    for (int s = 0; s < nslices; ++s) {
        int es = std::max(e[s], q.e_lo);             // cap; a zero slice takes e_lo
        q.shift[s] = es - q.e_lo;
        float inv = pow2f(-es);
        for (int b = SLICE_BLOCKS * s; b < std::min(nblocks, SLICE_BLOCKS * (s + 1)); ++b) {
            for (int k = 0; k < 32; ++k) {
                int p = trio_of_k(k);
                if (p < 0) continue;                         // zero column, invariant
                float v = x[24 * b + ORDER[p]] * inv;        // exact: power-of-two scaling
                long r = std::lrint(v);                      // half-even in the default mode
                if (r > QMAX) r = QMAX;
                if (r < -QMAX) r = -QMAX;
                int xi = int(r);
                q.hi[32 * b + k] = int8_t(xi >> 8);          // arithmetic shift, not /256
                q.lo[32 * b + k] = uint8_t(xi & 0xff);
            }
        }
    }
    return q;
}

// Per-matrix scale grid. S_f32[g][m] = round_f32(gscale[g] / sqrt(16 m)) evaluated in f64, for
// g in {0, 1} and m in 1..26 (llvq_tetra48.cuh:26-30 bounds m <= 26); S[g][0] = 0 (the origin).
// That is 52 nonzero scales. E_S is taken over |S|, so a negative gain level (artstat counts them)
// is carried by the sign of S_int, and a zero level (c0 = 0) simply leaves 26 nonzero scales.
// |S_int| < 2^28 keeps the int64 bound of brick.md §3.1 (|I| < 2^62.7 at 1,067 blocks); it holds
// when max(|c0|, |c1|) / min(|c0|, |c1|) <= 3.14. A matrix past it is REFUSED BY NAME at load, never
// asserted: the served files' levels are unread until step S0 (`artstat`), and a release build
// would compile an assert away.
constexpr int M_MAX = 26;
struct SGrid { int32_t s_int[2][M_MAX + 1]; int e_s; };

// Returns false and fills `why` when the grid cannot hold this matrix's levels exactly.
inline bool make_grid(const float gscale[2], const char* name, SGrid& g, std::string& why) {
    g = SGrid{};
    float s[2][M_MAX + 1];
    float amin = INFINITY;
    for (int gi = 0; gi < 2; ++gi)
        for (int m = 0; m <= M_MAX; ++m) {
            s[gi][m] = m == 0 ? 0.0f : float(double(gscale[gi]) / std::sqrt(16.0 * m));
            if (s[gi][m] != 0.0f) amin = std::fmin(amin, std::fabs(s[gi][m]));
        }
    if (!std::isfinite(amin)) { why = std::string(name) + ": both gain levels are zero"; return false; }
    g.e_s = exp2_of(amin) - 23;
    for (int gi = 0; gi < 2; ++gi)
        for (int m = 0; m <= M_MAX; ++m) {
            double v = std::ldexp(double(s[gi][m]), -g.e_s);            // exact: a power-of-two scaling
            if (v != std::floor(v)) { why = std::string(name) + ": a scale is off the 2^E_S grid"; return false; }
            if (std::fabs(v) >= double(1 << 28)) {
                why = std::string(name) + ": gain levels span more than 4 binades with sqrt(26); "
                      "needs the two-accumulator form of brick.md §3.1";
                return false;
            }
            g.s_int[gi][m] = int32_t(v);
        }
    return true;
}

// Block dot from six signed quads (trio order: q[i] byte j <-> trio 4i + j) and one quantized
// block record. Exact: |P| < 2^21.65.
inline int32_t block_dot(const uint32_t q[6], const int8_t* hi32, const uint8_t* lo32) {
    // quad i covers trio 4i..4i+3; in MMA k order that is k = 4(i/2) + 16(i%2) + j.
    int32_t ph = 0, pl = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 4; ++j) {
            int y = int(int8_t(uint8_t(q[i] >> (8 * j))));
            int k = 4 * (i >> 1) + 16 * (i & 1) + j;
            ph += y * int(hi32[k]);
            pl += y * int(lo32[k]);
        }
    return ph * 256 + pl;    // the C-operand chain of the device: (P_hi * 256) + P_lo
                             // (multiplications, not shifts: << on a negative int is UB before C++20)
}

// One output row for one token. `quads(b)` returns the six signed quads of block b and its (g, m).
template <class Quads>
inline uint16_t row_out(int nblocks, Quads quads, const QToken& t, const SGrid& g,
                        float rscale, const uint16_t* tail16, const float* xtail, int tail_w,
                        float (*h2f)(uint16_t), uint16_t (*f2h)(float)) {
    if (t.nonfinite) return 0x7e00;   // NaN policy
    int64_t acc = 0;                  // order-free: any split or order gives the same value
    for (int b = 0; b < nblocks; ++b) {
        uint32_t q[6]; int gi, m;
        quads(b, q, gi, m);
        int32_t p = block_dot(q, &t.hi[32 * b], &t.lo[32 * b]) * (1 << t.shift[b / SLICE_BLOCKS]);
        acc += int64_t(p) * int64_t(g.s_int[gi][m]);
    }
    float u = float(acc);                                   // cvt.rn.f32.s64
    int e = t.e_lo + g.e_s;                                 // device: the same two steps, __fmul_rn each
    float v = e >= -126 ? u * pow2f(e)                      // exact while v stays normal
                        : (u * pow2f(-126)) * pow2f(e + 126);  // two correctly rounded steps, no FTZ
    float tv = 0.0f;
    for (int i = 0; i < tail_w; ++i) tv = std::fma(h2f(tail16[i]), xtail[i], tv);  // __fmaf_rn chain
    return f2h(std::fma(v, rscale, tv));                    // __fmaf_rn, then one rounding to f16
}

}  // namespace t48i
