// Drives the persistent kernels and the served kernels they must equal, on
// the CPU, through host_grid.h. Built twice from this one file:
//
//   -DPERS_SIDE=0  the SERVED side: tv_tetra48_h, tv_q4_hh, rot_apply, a
//                  line-by-line replay of candle's rmsnorm tree, and a replay
//                  of tv_tetra48_h (`served_model`) whose three contractible
//                  sites can be fused or not. Compiled with
//                  -ffp-contract=off, so the served TEXT runs exactly as
//                  written. The replay with nothing fused must equal that text
//                  bit for bit (command 'U'), which proves it has the served
//                  lane order, tiling and reduction; the replay with the three
//                  fusions NVVM made in the served unit's PTX
//                  (docs/mesures/ptx-probe-2026-11-04.txt §2) is the reference
//                  the persistent side is held to. Host clang cannot stand in
//                  for NVVM there: with -ffp-contract=fast it fuses the
//                  epilogue but not `acc += t * invnorm` (measured on the dev
//                  Mac, 2026-11-04: 9 of 16 rows matched the fused model).
//   -DPERS_SIDE=1  the PERSISTENT side: llvq_pers.cuh, tetra48_pers.cu and
//                  pers_segment.cu, also with -ffp-contract=off. Their
//                  arithmetic is spelled with explicit intrinsics.
//
// tests/pers_matches_served.rs runs both on the same command and compares the
// outputs byte for byte. Fixtures are generated here from a seed, by bit
// construction only (no float multiply or add), so both builds generate the
// same bytes whatever their contraction.
//
// stdin : u32 rows[4096], u8 pref[128], u16 br[1024], u8 suf[128], f32 invnorm[32]
//         then one command (u32 tag, then its arguments)
// stdout: the command's outputs, little-endian.

#ifndef PERS_SIDE
#error "build with -DPERS_SIDE=0 (served) or 1 (persistent)"
#endif

#include "host_grid.h"

#define PERS_SPIN_LIMIT (1u << 14)
#define TILE_BLOCKS 64u
#include "../kernels/llvq_slot.cuh"
#include "../kernels/matvec.cu"
#include "../kernels/llvq_f1rank.cuh"
#include "../kernels/llvq_f1rank_v3.cuh"
#include "../kernels/llvq_tetra48.cuh"
#include "../kernels/llvq_rot.cuh"
#if PERS_SIDE == 0
#include "../kernels/rotate.cu"
#include "../kernels/tetra48_seg.cu"
// The served kernels write `f2h(<f32 value>)`. A binary16 output hides most
// association changes (a 1-ulp f32 move survives the narrowing about once in
// 8,192 rows), so the served side records the f32 argument, row by row,
// without touching the expression: the macro only renames the call.
static float* g_capture = nullptr;
static inline unsigned short served_capture_f2h(float v) {
    if (g_capture) g_capture[(blockIdx.x * blockDim.x + threadIdx.x) >> 5] = v;
    return f2h(v);
}
#define f2h(v) served_capture_f2h(v)
#include "../../llvq-llm/kernels/tv_tetra48_h.cu"
#include "../../llvq-llm/kernels/tv_q4_h.cu"
#include "../../llvq-llm/kernels/tv_q4_tile.cu"
#include "../../llvq-llm/kernels/emb_q4.cu"
#undef f2h
#else
#include "../kernels/llvq_pers.cuh"
#include "../kernels/tetra48_pers.cu"
#include "../kernels/pers_segment.cu"
#include "../kernels/pers_4e.cu"
#if PERS_4E_ROWS >= 7
#include "../kernels/pers_4ev.cu"
#endif
#include "../kernels/pers_trows.cu"
#include "../kernels/pers_q4rows.cu"
#include "../kernels/pers_ctr.cu"
#include "../kernels/tetra48_persall.cu"
#include "../../llvq-llm/kernels/tok_argmax.cu"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The served kernels' `extern __shared__` arrays, one CTA at a time.
float xs[32768];
float s[32768];

// ---- I/O -------------------------------------------------------------------
template <typename T>
static std::vector<T> rd(size_t n) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, stdin) != n) {
        std::fprintf(stderr, "host_pers: truncated fixture\n");
        std::exit(2);
    }
    return v;
}
template <typename T>
static T rd1() {
    return rd<T>(1)[0];
}
template <typename T>
static void wr(const std::vector<T>& v) {
    std::fwrite(v.data(), sizeof(T), v.size(), stdout);
}

// ---- fixtures, by bit construction ---------------------------------------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    // binary16 with exponent field in [lo, hi], random sign and mantissa
    unsigned short h16(unsigned lo, unsigned hi) {
        uint64_t r = next();
        unsigned e = lo + (unsigned)((r >> 20) % (hi - lo + 1));
        return (unsigned short)(((r >> 40) & 1u) << 15 | e << 10 | (r & 0x3ffu));
    }
    // binary32 with exponent field in [lo, hi]
    float f32(unsigned lo, unsigned hi, bool sign) {
        uint64_t r = next();
        unsigned e = lo + (unsigned)((r >> 32) % (hi - lo + 1));
        unsigned b = (sign ? (unsigned)((r >> 60) & 1u) << 31 : 0u) | e << 23 | (unsigned)(r & 0x7fffffu);
        return __uint_as_float(b);
    }
};

static std::vector<u32> g_rows;
static std::vector<unsigned char> g_pref, g_suf;
static std::vector<unsigned short> g_br;
static std::vector<float> g_inv;

static F1rTables tables() { return F1rTables{g_rows.data(), g_pref.data(), g_br.data(), g_suf.data()}; }

// One Tetra matrix of the served stream, generated from a seed.
struct TMat {
    u32 d_out, nblocks, tail_w, stride;
    std::vector<u32> words;
    std::vector<float> rscale, gscale;
    std::vector<unsigned short> tail;
};
static TMat tmat(u32 d_out, u32 d_in, uint64_t seed) {
    Rng r(seed);
    TMat m;
    m.d_out = d_out;
    m.nblocks = d_in / 24;
    m.tail_w = d_in % 24;
    m.stride = (6 * m.nblocks + 7) / 8 * 2;
    m.words.resize((size_t)d_out * m.stride);
    for (auto& w : m.words) w = (u32)r.next();
    m.rscale.resize(d_out);
    for (auto& v : m.rscale) v = r.f32(119, 123, false);
    m.gscale = {r.f32(126, 126, false), r.f32(127, 127, false)};
    m.tail.resize((size_t)d_out * m.tail_w);
    for (auto& v : m.tail) v = r.h16(7, 12);
    return m;
}
// The f32 activation a Tetra row reads: the rotated vector, full binary32
// precision. Widened binary16 values would hide site 2: a binary16 tail times
// a binary16 value is exact, so its contraction could not show.
static std::vector<float> act(u32 n, uint64_t seed) {
    Rng r(seed);
    std::vector<float> x(n);
    for (auto& v : x) v = r.f32(122, 128, true);
    return x;
}
static std::vector<unsigned short> act_h(u32 n, uint64_t seed, unsigned lo = 11, unsigned hi = 16) {
    Rng r(seed);
    std::vector<unsigned short> x(n);
    for (auto& v : x) v = r.h16(lo, hi);
    return x;
}

// ---- served side -----------------------------------------------------------
#if PERS_SIDE == 0
static std::vector<unsigned short> served_tetra(const TMat& m, const std::vector<float>& x, float* f32_out = nullptr) {
    std::vector<unsigned short> y(m.d_out);
    const F1rTables t = tables();
    g_capture = f32_out;
    hostgrid::launch_seq(m.d_out / 8, 256, [&] {
        tv_tetra48_h(m.words.data(), m.stride, t.rows, t.prefixes, t.branches, t.suffixes, m.gscale.data(),
                     g_inv.data(), m.rscale.data(), m.tail.data(), x.data(), y.data(), m.nblocks, m.tail_w);
    });
    g_capture = nullptr;
    return y;
}

// tv_tetra48_h, replayed: lane l adds blocks l, l+32, ... (the tile of 64 is
// a multiple of 32, so the order runs straight across tiles), the 24-term
// chain of tetra48_dot, then the xor butterfly, the f16 tail, the epilogue.
// `fused` makes the three contractible sites single roundings, as the served
// unit's PTX has them; unfused, each is a rounded product and a rounded sum,
// as the source reads without contraction.
static std::vector<unsigned short> served_model(const TMat& m, const std::vector<float>& x, float* f32_out, bool fused) {
    const F1rTables t = tables();
    std::vector<unsigned short> y(m.d_out);
    for (u32 row = 0; row < m.d_out; ++row) {
        const u32* wrow = m.words.data() + (size_t)row * m.stride;
        float lanes[32];
        for (u32 l = 0; l < 32; ++l) {
            float acc = 0.0f;
            for (u32 j = l; j < m.nblocks; j += 32u) {
                u32 lo, hi16, q[6];
                f1r_load(wrow, j, lo, hi16);
                f1r_v3_quads(lo, hi16, t, q);
                float ab = 0.0f;
                for (u32 i = 0; i < 6u; ++i)
                    for (u32 k = 0; k < 4u; ++k) ab = __fmaf_rn(f1r_v3_float(q[i], k), x[j * 24u + TETRA48_ORDER[4u * i + k]], ab);
                const u32 mm = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
                const float tt = __fmul_rn(ab, m.gscale[(hi16 >> 15) & 1u]);
                acc = fused ? __fmaf_rn(tt, g_inv[mm], acc) : __fadd_rn(acc, __fmul_rn(tt, g_inv[mm]));
            }
            lanes[l] = acc;
        }
        for (int k = 16; k > 0; k >>= 1) {
            float n[32];
            for (u32 l = 0; l < 32; ++l) n[l] = __fadd_rn(lanes[l], lanes[l ^ (u32)k]);
            for (u32 l = 0; l < 32; ++l) lanes[l] = n[l];
        }
        float tv = 0.0f;
        for (u32 i = 0; i < m.tail_w; ++i) {
            const float h = h2f(m.tail[(size_t)row * m.tail_w + i]), xi = x[m.nblocks * 24u + i];
            tv = fused ? __fmaf_rn(h, xi, tv) : __fadd_rn(tv, __fmul_rn(h, xi));
        }
        const float v = fused ? __fmaf_rn(lanes[0], m.rscale[row], tv) : __fadd_rn(__fmul_rn(lanes[0], m.rscale[row]), tv);
        if (f32_out) f32_out[row] = v;
        y[row] = f2h(v);
    }
    return y;
}

// The reference every comparison uses: the served kernel as NVVM compiled it.
static std::vector<unsigned short> served_ref(const TMat& m, const std::vector<float>& x, float* f32_out = nullptr) {
    return served_model(m, x, f32_out, true);
}

// fma.rn.f16(±a·b, c) by exact integer arithmetic: an independent reference
// for the persistent side's f64 + TwoSum emulation.
static void f16_parts(unsigned short h, long long& mant, int& ex) {
    const int e = (h >> 10) & 0x1f;
    const long long m = h & 0x3ff;
    mant = e ? (m | 0x400) : m;
    ex = e ? e - 25 : -24;
    if (h & 0x8000) mant = -mant;
}
static unsigned short ref_f16_fma(unsigned short a, unsigned short b, unsigned short c, bool neg_ab) {
    long long ma, mb, mc;
    int ea, eb, ec;
    f16_parts(a, ma, ea);
    f16_parts(b, mb, eb);
    f16_parts(c, mc, ec);
    __int128 P = (__int128)ma * mb * (neg_ab ? -1 : 1);
    int ep = ea + eb;
    __int128 C = mc;
    int E = ep < ec ? ep : ec;
    P <<= (ep - E);
    C <<= (ec - E);
    __int128 N = P + C;
    if (N == 0) return 0;  // an exact zero of a sum rounds to +0
    const unsigned short sign = N < 0 ? 0x8000 : 0;
    unsigned __int128 A = N < 0 ? (unsigned __int128)(-N) : (unsigned __int128)N;
    int L = 0;
    for (unsigned __int128 t = A; t; t >>= 1) ++L;
    const int u = L - 1 + E;  // unbiased exponent of the leading bit
    if (u > 15) return sign | 0x7c00;
    int shift = u >= -14 ? L - 11 : -24 - E;
    unsigned __int128 q;
    if (shift <= 0) {
        q = A << (-shift);
    } else {
        q = A >> shift;
        const unsigned __int128 rem = A & ((((unsigned __int128)1) << shift) - 1), half = ((unsigned __int128)1) << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
    }
    if (u < -14) return sign | (unsigned short)q;  // subnormal, or the least normal after a carry
    int e16 = u + 15;
    if (q == 0x800) { q = 0x400; ++e16; }
    if (e16 >= 31) return sign | 0x7c00;
    return sign | (unsigned short)(e16 << 10) | (unsigned short)(q & 0x3ff);
}

static void ref_rope_pair(int form, unsigned short a, unsigned short b, unsigned short c, unsigned short s,
                          unsigned short* o1, unsigned short* o2) {
    auto hm = [](unsigned short x, unsigned short y) { return f2h(__fmul_rn(h2f(x), h2f(y))); };
    const unsigned short ac = hm(a, c), bs = hm(b, s), as = hm(a, s), bc = hm(b, c);
    if (form == 0) {
        *o1 = f2h(__fsub_rn(h2f(ac), h2f(bs)));
        *o2 = f2h(__fadd_rn(h2f(as), h2f(bc)));
    } else if (form == 1) {
        *o1 = ref_f16_fma(a, c, bs ^ 0x8000, false);
        *o2 = ref_f16_fma(a, s, bc, false);
    } else {
        *o1 = ref_f16_fma(b, s, ac, true);
        *o2 = ref_f16_fma(b, c, as, false);
    }
}

// attn_ref.rs, ported line for line (an independent text from the persistent
// phase; tests/pers_matches_served.rs holds it against the Rust original).
static float ref_ex2(float x) {
    if (x != x) return x;
    if (x <= -126.0f) return 0.0f;
    const float fl = std::floor(x);
    const float f = __fsub_rn(x, fl);
    const unsigned cb[6] = {0x3f317217u, 0x3e75feb3u, 0x3d634139u, 0x3c1eab8fu, 0x3aa228dbu, 0x39665cf6u};
    float p = __uint_as_float(cb[5]);
    for (int i = 4; i >= 0; --i) p = std::fma(p, f, __uint_as_float(cb[i]));
    p = std::fma(p, f, 1.0f);
    const int e = (int)fl;
    return __uint_as_float(__float_as_uint(p) + ((unsigned)e << 23));
}
struct RefSt { float m, d, a[128]; };
static void ref_st_empty(RefSt& s) { s.m = -INFINITY; s.d = 0.0f; for (float& v : s.a) v = 0.0f; }
static void ref_st_merge(RefSt& a, const RefSt& b) {
    if (b.m == -INFINITY) return;
    if (a.m == -INFINITY) { a = b; return; }
    const float m = std::max(a.m, b.m), fa = ref_ex2(__fsub_rn(a.m, m)), fb = ref_ex2(__fsub_rn(b.m, m));
    a.d = __fadd_rn(__fmul_rn(a.d, fa), __fmul_rn(b.d, fb));
    for (int i = 0; i < 128; ++i) a.a[i] = __fadd_rn(__fmul_rn(a.a[i], fa), __fmul_rn(b.a[i], fb));
    a.m = m;
}
static void ref_attn_row(const unsigned short* q16, const unsigned short* kc, const unsigned short* vc, u32 l, float c,
                         unsigned short* out) {
    float q[128];
    for (int i = 0; i < 128; ++i) q[i] = h2f(q16[i]);
    RefSt total;
    ref_st_empty(total);
    for (u32 s0 = 0; s0 < l; s0 += 256) {
        RefSt split;
        ref_st_empty(split);
        for (u32 u0 = s0; u0 < s0 + 256 && u0 < l; u0 += 64) {
            RefSt st;
            ref_st_empty(st);
            for (u32 p0 = u0; p0 < u0 + 64 && p0 < l; p0 += 8) {
                const u32 nv = std::min<u32>(8, l - p0);
                float sc[8];
                for (u32 j = 0; j < nv; ++j) {
                    float x[16];
                    for (int ln = 0; ln < 16; ++ln) {
                        float t = 0.0f;
                        for (int i = 0; i < 8; ++i) t = std::fma(q[ln * 8 + i], h2f(kc[(p0 + j) * 128 + ln * 8 + i]), t);
                        x[ln] = t;
                    }
                    for (int off : {8, 4, 2, 1}) {
                        float y[16];
                        for (int ln = 0; ln < 16; ++ln) y[ln] = __fadd_rn(x[ln], x[ln ^ off]);
                        for (int ln = 0; ln < 16; ++ln) x[ln] = y[ln];
                    }
                    sc[j] = __fmul_rn(x[0], c);
                }
                float mt = sc[0];
                for (u32 j = 1; j < nv; ++j) mt = std::max(mt, sc[j]);
                float m_new, carry;
                if (st.m == -INFINITY) { m_new = mt; carry = 0.0f; }
                else { m_new = std::max(st.m, mt); carry = ref_ex2(__fsub_rn(st.m, m_new)); }
                float pj[8], ts = 0.0f;
                for (u32 j = 0; j < nv; ++j) { pj[j] = ref_ex2(__fsub_rn(sc[j], m_new)); ts = __fadd_rn(ts, pj[j]); }
                st.d = __fadd_rn(__fmul_rn(st.d, carry), ts);
                for (int i = 0; i < 128; ++i) {
                    float a = __fmul_rn(st.a[i], carry);
                    for (u32 j = 0; j < nv; ++j) a = std::fma(pj[j], h2f(vc[(p0 + j) * 128 + i]), a);
                    st.a[i] = a;
                }
                st.m = m_new;
            }
            ref_st_merge(split, st);
        }
        ref_st_merge(total, split);
    }
    for (int i = 0; i < 128; ++i) out[i] = f2h(__fdiv_rn(total.a[i], total.d));
}

// candle-kernels 0.9.2 reduce.cu:187-228, one row, BLOCK virtual threads,
// replayed in candle's order. VARIANT 0: `tmp += xi * xi` contracted to an
// fma; 1: not.
static float candle_rms_scale(const unsigned short* x, u32 ncols, float eps, int variant) {
    const u32 block = ncols < 1024u ? 32u : 1024u;
    std::vector<float> tmp(block, 0.0f);
    for (u32 t = 0; t < block; ++t)
        for (u32 col = t; col < ncols; col += block) {
            const float xi = h2f(x[col]);
            tmp[t] = variant == 0 ? std::fma(xi, xi, tmp[t]) : __fadd_rn(tmp[t], __fmul_rn(xi, xi));
        }
    auto butterfly = [](std::vector<float>& v, u32 base) {
        for (int mask = 16; mask > 0; mask >>= 1) {
            float nv[32];
            for (u32 l = 0; l < 32; ++l) nv[l] = __fadd_rn(v[base + l], v[base + (l ^ (u32)mask)]);
            for (u32 l = 0; l < 32; ++l) v[base + l] = nv[l];
        }
    };
    for (u32 w = 0; w < block / 32; ++w) butterfly(tmp, w * 32);
    float total;
    if (block > 32u) {
        std::vector<float> s_sum(32);
        for (u32 w = 0; w < 32; ++w) s_sum[w] = tmp[w * 32];  // lane 0 of each warp
        butterfly(s_sum, 0);
        total = s_sum[0];
    } else {
        total = tmp[0];
    }
    const float mean = __fdiv_rn(total, (float)ncols);
    return rsqrtf(__fadd_rn(mean, eps));
}
#endif

// ---- persistent side: test kernels over the header's functions ------------
#if PERS_SIDE == 1
template <u32 STRIDE, bool TRIO, bool PREFETCH>
static void pers_rows(const TMat& m, const std::vector<float>& x, std::vector<unsigned short>& y,
                      std::vector<float>& dbg, unsigned grid, unsigned threads, uint64_t seed) {
    const F1rTables t = tables();
    const u32 n = m.nblocks * STRIDE + m.tail_w;
    hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 4 * n + 16, seed, [&] {
        float* xsp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
        pers_stage_f32<STRIDE, TRIO>(x.data(), xsp, m.nblocks, m.tail_w);
        const u32 lo = (u32)((uint64_t)m.d_out * blockIdx.x / gridDim.x);
        const u32 hi = (u32)((uint64_t)m.d_out * (blockIdx.x + 1) / gridDim.x);
        pers_claim_reset(lo);
        const u32 lane = threadIdx.x & 31u;
        for (;;) {
            const u32 r = pers_claim(lane, 1u);
            if (r >= hi) break;
            const float acc = pers_tetra_row<STRIDE, TRIO, PREFETCH>(m.words.data() + (size_t)r * m.stride, t,
                                                                     m.gscale.data(), g_inv.data(), xsp, m.nblocks,
                                                                     lane);
            if (lane == 0u) {
                const float v = pers_epilogue(acc, m.rscale[r], m.tail.data() + (size_t)r * m.tail_w,
                                              xsp + m.nblocks * STRIDE, m.tail_w);
                dbg[r] = v;
                y[r] = f2h(v);
            }
        }
        __syncthreads();
    });
}
#endif

int main() {
    g_rows = rd<u32>(4096);
    g_pref = rd<unsigned char>(128);
    g_br = rd<unsigned short>(1024);
    g_suf = rd<unsigned char>(128);
    g_inv = rd<float>(32);
    const u32 tag = rd1<u32>();

    if (tag == 'R') {
        // ROTATION: n, m, k, inv, threads, seed; then u32 signs[n/32], f32 small[KMAX*KMAX]
        const u32 n = rd1<u32>(), m = rd1<u32>(), k = rd1<u32>();
        const float inv = rd1<float>();
        const u32 threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const auto signs = rd<u32>(n / 32);
        const auto small = rd<float>(LLVQ_ROT_KMAX * LLVQ_ROT_KMAX);
        // the whole binary16 range: inputs that span a few binades add
        // exactly inside a lane, and stage order could not show
        const auto xin = act_h(n, seed, 1, 29);
        std::vector<float> out(n);
#if PERS_SIDE == 0
        hostgrid::launch_seq(1, threads, [&] {
            rot_apply(xin.data(), signs.data(), small.data(), out.data(), n, m, k, inv, 0u);
        });
#else
#if PERS_4E_ROWS == 4
        // form 4: the one-round-trip load, `small` staged after the vector
        if (!pers_1rt_ok(xin.data(), n)) {
            std::fprintf(stderr, "host_pers: the rotation's input is not 16-byte aligned\n");
            std::exit(2);
        }
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER + 4 * n + 4 * LLVQ_ROT_KMAX * LLVQ_ROT_KMAX, seed, [&] {
            float* sp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            pers_load_1rt(xin.data(), signs.data(), sp, n, small.data(), k, sp + n);
            pers_rot_finish<true>(sp, sp + n, n, m, k, inv);
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) out[i] = sp[i];
        });
#elif PERS_4E_ROWS >= 5
        // form 5: the sign words and `small` copied into the v7 area first
        // (4E does it during the barrier before), then the vector load reads
        // the signs from there and the mix reads `small` as float4
        if (!pers_1rt_ok(xin.data(), n)) {
            std::fprintf(stderr, "host_pers: the rotation's input is not 16-byte aligned\n");
            std::exit(2);
        }
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER + 4 * n + 4 * PERS_V7_FLOATS, seed, [&] {
            float* sp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            float* area = sp + n;
            pers_v7_stage_rot(area, signs.data(), n, small.data(), k);
            __syncthreads();
            pers_load_1rt(xin.data(), reinterpret_cast<const u32*>(area + PERS_V7_SIGNS), sp, n, nullptr, k,
                          area + PERS_V7_SMALL);
            pers_rot_finish<true, true>(sp, area + PERS_V7_SMALL, n, m, k, inv);
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) out[i] = sp[i];
        });
#else
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER + 4 * n, seed, [&] {
            float* sp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            pers_rot_load(xin.data(), signs.data(), sp, n);
            pers_rot_finish(sp, small.data(), n, m, k, inv);
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) out[i] = sp[i];
        });
#endif
#endif
        wr(out);
    } else if (tag == 'M') {
        // MIX on a given vector: m, k, inv, then f32 s[k*m], f32 small[KMAX*KMAX]
        const u32 m = rd1<u32>(), k = rd1<u32>();
        const float inv = rd1<float>();
        auto sv = rd<float>((size_t)k * m);
        const auto small = rd<float>(LLVQ_ROT_KMAX * LLVQ_ROT_KMAX);
        std::vector<float> out((size_t)k * m);
#if PERS_SIDE == 0
        hostgrid::launch_seq(1, 32, [&] { rot_mix(sv.data(), small.data(), out.data(), m, k, inv, threadIdx.x, 32u); });
#else
        hostgrid::launch_coop(1, 32, PERS_SMEM_HEADER, 0, [&] {
#if PERS_4E_ROWS == 4
            for (u32 j = threadIdx.x; j < m; j += 32u) pers_mix_1rt(sv.data(), small.data(), m, k, inv, j);
#elif PERS_4E_ROWS >= 5
            for (u32 j = threadIdx.x; j < m; j += 32u) pers_mix_1rt<true>(sv.data(), small.data(), m, k, inv, j);
#else
            for (u32 j = threadIdx.x; j < m; j += 32u) pers_mix_column(sv.data(), small.data(), m, k, inv, j);
#endif
        });
        out = sv;
#endif
        wr(out);
    } else if (tag == 'N') {
        // NORM: ncols, eps, variant, threads, seed
        const u32 ncols = rd1<u32>();
        const float eps = rd1<float>();
        const u32 variant = rd1<u32>(), threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const auto x = act_h(ncols, seed, 10, 17);
        const auto alpha = act_h(ncols, seed + 1, 13, 16);
        std::vector<unsigned short> y(ncols);
        std::vector<float> scale(1);
#if PERS_SIDE == 0
        (void)threads;
        scale[0] = candle_rms_scale(x.data(), ncols, eps, (int)variant);
        for (u32 i = 0; i < ncols; ++i) y[i] = f2h(__fmul_rn(__fmul_rn(scale[0], h2f(x[i])), h2f(alpha[i])));
#else
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER, seed, [&] {
            const float sc = variant == 0 ? exact_rms_scale<0>(x.data(), ncols, eps)
                                          : exact_rms_scale<1>(x.data(), ncols, eps);
            for (u32 i = threadIdx.x; i < ncols; i += blockDim.x) y[i] = exact_rms_apply(sc, x[i], alpha[i]);
            if (threadIdx.x == 0) scale[0] = sc;
        });
#endif
        wr(y);
        wr(scale);
    } else if (tag == 'J') {
        // ROW LOOPS (pers_trows.cu): arm, grid, threads, stamps (0/1), x seed, nsites, then per site d_out, d_in,
        // seed. Every site's rows in order, binary16, then their f32 values before f2h; arm 7 (K2) adds every
        // site's rows on the second activation after each block. The served side reads the arm for that only.
        const u32 arm = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), stamp = rd1<u32>();
        const uint64_t xseed = rd1<uint64_t>();
        const u32 nsites = rd1<u32>();
        std::vector<TMat> mats;
        u32 maxd = 0u;
        for (u32 s = 0; s < nsites; ++s) {
            const u32 d_out = rd1<u32>(), d_in = rd1<u32>();
            const uint64_t seed = rd1<uint64_t>();
            mats.push_back(tmat(d_out, d_in, seed));
            maxd = std::max(maxd, d_in);
        }
        const auto x = act(maxd, xseed);
        const auto x2 = act(maxd, xseed + 1u);
        const bool k2 = arm == 7u;
        std::vector<unsigned short> out;
        std::vector<float> outf;
#if PERS_SIDE == 0
        (void)grid;
        (void)threads;
        (void)stamp;
        std::vector<unsigned short> out2;
        std::vector<float> outf2;
        for (const TMat& m : mats) {
            for (int which = 0; which < (k2 ? 2 : 1); ++which) {
                const auto& xv = which ? x2 : x;
                const std::vector<float> xi(xv.begin(), xv.begin() + m.nblocks * 24u + m.tail_w);
                std::vector<float> f(m.d_out);
                const auto y = served_ref(m, xi, f.data());
                (which ? out2 : out).insert((which ? out2 : out).end(), y.begin(), y.end());
                (which ? outf2 : outf).insert((which ? outf2 : outf).end(), f.begin(), f.end());
            }
        }
        out.insert(out.end(), out2.begin(), out2.end());
        outf.insert(outf.end(), outf2.begin(), outf2.end());
#else
        std::vector<std::vector<unsigned short>> ys, ys2;
        std::vector<std::vector<float>> fs, fs2;
        std::vector<unsigned long long> words;
        ys.reserve(mats.size());
        fs.reserve(mats.size());
        ys2.reserve(mats.size());
        fs2.reserve(mats.size());
        for (const TMat& m : mats) {
            ys.emplace_back(m.d_out, 0xFFFFu);
            fs.emplace_back(m.d_out, -1.0f);
            ys2.emplace_back(m.d_out, 0xFFFFu);
            fs2.emplace_back(m.d_out, -1.0f);
            const unsigned long long w[TROWS_SITE_WORDS] = {
                (unsigned long long)(uintptr_t)m.words.data(), (unsigned long long)(uintptr_t)m.gscale.data(),
                (unsigned long long)(uintptr_t)m.rscale.data(), (unsigned long long)(uintptr_t)m.tail.data(),
                (unsigned long long)(uintptr_t)ys.back().data(),
                (unsigned long long)m.d_out | ((unsigned long long)m.nblocks << 32),
                (unsigned long long)m.tail_w | ((unsigned long long)m.stride << 32),
                (unsigned long long)(uintptr_t)fs.back().data(),
                k2 ? (unsigned long long)(uintptr_t)ys2.back().data() : 0ull,
                k2 ? (unsigned long long)(uintptr_t)fs2.back().data() : 0ull};
            words.insert(words.end(), w, w + TROWS_SITE_WORDS);
        }
        // TILE (arm 11) reads the words relaid by tv_trows_tile_relayout, run
        // through the harness like any kernel; its site words point there.
        std::vector<std::vector<u32>> tiled(mats.size());
        if (arm == TROWS_TILE) {
            for (size_t k = 0; k < mats.size(); ++k) {
                const TMat& m = mats[k];
                const u32 ts = trows_tile_stride(m.nblocks);
                tiled[k].assign((size_t)m.d_out * ts, 0u);
                hostgrid::launch_seq(3, 64, [&] {
                    tv_trows_tile_relayout(m.words.data(), m.stride, tiled[k].data(), ts, m.d_out, m.nblocks);
                });
                words[k * TROWS_SITE_WORDS] = (unsigned long long)(uintptr_t)tiled[k].data();
                words[k * TROWS_SITE_WORDS + 6u] = (unsigned long long)m.tail_w | ((unsigned long long)ts << 32);
            }
        }
        std::vector<u32> st((size_t)nsites * grid * (threads / 32u + 1u), 0u);
        std::vector<unsigned long long> ends(4u * grid, 0ull);
        u32* sp = stamp ? st.data() : nullptr;
        unsigned long long* ep = stamp ? ends.data() : nullptr;
        std::vector<u32> ctrs(2u + nsites, 0u);
        const bool cpa = arm == TROWS_CPA;
        const size_t smem = PERS_SMEM_HEADER + 18816u + 4u * (k2 || cpa ? TROWS_X2_AT + maxd : maxd) + 16u +
                            (cpa ? 4u * TROWS_CPA_WARP * (threads / 32u) : 0u);
        hostgrid::launch_coop(grid, threads, smem, xseed, [&] {
            switch (arm) {
            case 0: trows_body<TROWS_V2>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 1: trows_body<TROWS_V3>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 2: trows_body<TROWS_ILP2>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 3: trows_body<TROWS_V2U>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 4: trows_body<TROWS_L1W>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 5: trows_body<TROWS_BCAST>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 6: trows_body<TROWS_XBCAST>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 7: trows_body<TROWS_K2>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 8: trows_body<TROWS_NC>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 9: trows_body<TROWS_NC256>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 10: trows_body<TROWS_CPA>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 11: trows_body<TROWS_TILE>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            case 12: trows_body<TROWS_BAL>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), x2.data(), sp, ep); break;
            // GB and GBC read x2 as their counters: barrier, abort flag, a cursor a site
            case 13: trows_body<TROWS_GB>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), reinterpret_cast<const float*>(ctrs.data()), sp, ep); break;
            case 14: trows_body<TROWS_GBC>(words.data(), nsites, g_rows.data(), g_pref.data(), g_br.data(), g_suf.data(), g_inv.data(), x.data(), reinterpret_cast<const float*>(ctrs.data()), sp, ep); break;
            default: std::abort();
            }
        });
        // GB and GBC: every CTA arrived at the grid barrier after every site
        if ((arm == TROWS_GB || arm == TROWS_GBC) && (ctrs[0] != nsites * grid || ctrs[1] != 0u)) {
            std::fprintf(stderr, "host_pers: grid barrier at %u, %u sites x %u CTAs expected\n", ctrs[0], nsites, grid);
            return 2;
        }
        if (stamp) {
            // every busy warp and every CTA stamped a positive count
            for (size_t i = 0; i < st.size(); ++i)
                if (st[i] == 0u && (i % (threads / 32u + 1u)) == threads / 32u) {
                    std::fprintf(stderr, "host_pers: CTA stamp %zu is zero\n", i);
                    return 2;
                }
            for (u32 c = 0; c < grid; ++c)
                if (ends[4u * c + 2u] <= ends[4u * c] || ends[4u * c + 3u] <= ends[4u * c + 1u]) {
                    std::fprintf(stderr, "host_pers: CTA %u: launch ends out of order\n", c);
                    return 2;
                }
        }
        for (const auto& y : ys) out.insert(out.end(), y.begin(), y.end());
        for (const auto& f : fs) outf.insert(outf.end(), f.begin(), f.end());
        if (k2) {
            for (const auto& y : ys2) out.insert(out.end(), y.begin(), y.end());
            for (const auto& f : fs2) outf.insert(outf.end(), f.begin(), f.end());
        }
#endif
        wr(out);
        wr(outf);
    } else if (tag == 'T') {
        // TETRA ROWS: d_out, d_in, variant (0 natural/24, 1 natural/25, 2 trio/28, 3 trio/28 prefetch),
        // grid, threads, seed
        const u32 d_out = rd1<u32>(), d_in = rd1<u32>(), variant = rd1<u32>(), grid = rd1<u32>(),
                  threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const TMat m = tmat(d_out, d_in, seed);
        const auto x = act(d_in, seed + 7);
        std::vector<unsigned short> y(d_out, 0xFFFFu);
        std::vector<float> dbg(d_out, 0.0f);
#if PERS_SIDE == 0
        (void)variant;
        (void)grid;
        (void)threads;
        y = served_ref(m, x, dbg.data());
#else
        switch (variant) {
        case 0: pers_rows<24u, false, false>(m, x, y, dbg, grid, threads, seed); break;
        case 1: pers_rows<25u, false, false>(m, x, y, dbg, grid, threads, seed); break;
        case 2: pers_rows<28u, true, false>(m, x, y, dbg, grid, threads, seed); break;
        default: pers_rows<28u, true, true>(m, x, y, dbg, grid, threads, seed); break;
        }
#endif
        wr(y);
        wr(dbg);
    } else if (tag == 'U') {
        // The replay with nothing fused against the served text, and the
        // replay fused: y and f32 of each, in that order. d_out, d_in, seed.
        const u32 d_out = rd1<u32>(), d_in = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const TMat m = tmat(d_out, d_in, seed);
        const auto x = act(d_in, seed + 7);
        std::vector<float> f_text(d_out), f_off(d_out), f_on(d_out);
#if PERS_SIDE == 0
        const auto y_text = served_tetra(m, x, f_text.data());
        const auto y_off = served_model(m, x, f_off.data(), false);
        const auto y_on = served_model(m, x, f_on.data(), true);
#else
        std::vector<unsigned short> y_text(d_out), y_off(d_out), y_on(d_out);
#endif
        wr(y_text); wr(f_text); wr(y_off); wr(f_off); wr(y_on); wr(f_on);
    } else if (tag == 'Q') {
        // INT4 ROWS: d_out, d_in, grid, threads, seed
        const u32 d_out = rd1<u32>(), d_in = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        Rng r(seed);
        const u32 gpr = d_in / 128u;
        std::vector<u32> wq((size_t)d_out * d_in / 8);
        for (auto& w : wq) w = (u32)r.next();
        std::vector<unsigned short> sc((size_t)d_out * gpr), bi((size_t)d_out * gpr);
        for (auto& v : sc) v = r.h16(6, 10);
        for (auto& v : bi) v = r.h16(8, 12);
        const auto xh = act_h(d_in, seed + 3);
        std::vector<unsigned short> y(d_out, 0xFFFFu);
        std::vector<float> dbg(d_out, 0.0f);
#if PERS_SIDE == 0
        (void)grid;
        (void)threads;
        g_capture = dbg.data();
        hostgrid::launch_seq(d_out / 8, 256, [&] { tv_q4_hh(wq.data(), sc.data(), bi.data(), xh.data(), y.data(), d_in, gpr); });
        g_capture = nullptr;
#else
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 4 * d_in, seed, [&] {
            float* xsp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xsp[i] = h2f(xh[i]);
            const u32 lo = (u32)((uint64_t)d_out * blockIdx.x / gridDim.x);
            const u32 hi = (u32)((uint64_t)d_out * (blockIdx.x + 1) / gridDim.x);
            pers_claim_reset(lo);
            const u32 lane = threadIdx.x & 31u;
            for (;;) {
                const u32 row = pers_claim(lane, 1u);
                if (row >= hi) break;
                // even rows the plain loop, odd rows the PREFETCH one (4E's)
                const float acc = (row & 1u) ? pers_q4_row<true>(wq.data(), sc.data(), bi.data(), xsp, row, d_in, gpr, lane)
                                             : pers_q4_row<false>(wq.data(), sc.data(), bi.data(), xsp, row, d_in, gpr, lane);
                if (lane == 0u) {
                    dbg[row] = acc;
                    y[row] = f2h(acc);
                }
            }
            __syncthreads();
        });
#endif
        wr(y);
        wr(dbg);
    } else if (tag == 'q') {
        // EAGER INT4 ENTRY POINTS on words the host hands in: d_out, d_in,
        // entry (0 tv_q4_hh, 1 tv_q4_h, 2 tv_q4_hht, 3 tv_q4_ht), seed, then
        // d_out x d_in / 8 words. Scales, biases and x drawn from the seed as
        // 'Q' draws them; x f16, widened exactly for the f32 entries. Served
        // side only: y and each row's f32 before the narrowing.
        const u32 d_out = rd1<u32>(), d_in = rd1<u32>(), entry = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const std::vector<u32> wq = rd<u32>((size_t)d_out * d_in / 8);
        Rng r(seed);
        const u32 gpr = d_in / 128u;
        std::vector<unsigned short> sc((size_t)d_out * gpr), bi((size_t)d_out * gpr);
        for (auto& v : sc) v = r.h16(6, 10);
        for (auto& v : bi) v = r.h16(8, 12);
        const auto xh = act_h(d_in, seed + 3);
        std::vector<float> xf(d_in);
        for (u32 i = 0; i < d_in; ++i) xf[i] = h2f(xh[i]);
        std::vector<unsigned short> y(d_out, 0xFFFFu);
        std::vector<float> dbg(d_out, 0.0f);
#if PERS_SIDE == 0
        g_capture = dbg.data();
        hostgrid::launch_seq(d_out / 8, 256, [&] {
            switch (entry) {
            case 0: tv_q4_hh(wq.data(), sc.data(), bi.data(), xh.data(), y.data(), d_in, gpr); break;
            case 1: tv_q4_h(wq.data(), sc.data(), bi.data(), xf.data(), y.data(), d_in, gpr); break;
            case 2: tv_q4_hht(wq.data(), sc.data(), bi.data(), xh.data(), y.data(), d_in, gpr); break;
            default: tv_q4_ht(wq.data(), sc.data(), bi.data(), xf.data(), y.data(), d_in, gpr); break;
            }
        });
        g_capture = nullptr;
#else
        (void)entry;
#endif
        wr(y);
        wr(dbg);
    } else if (tag == 'i') {
        // TILE INDEX: n, then n widths in words; for each, pers_q4_tile_index
        // of every word. Persistent side only (the served side writes zeros).
        const u32 n = rd1<u32>();
        const std::vector<u32> widths = rd<u32>(n);
        std::vector<u32> o;
        for (u32 nw : widths)
            for (u32 wi = 0; wi < nw; ++wi) {
#if PERS_SIDE == 1
                o.push_back(pers_q4_tile_index(wi, nw));
#else
                o.push_back(0u);
#endif
            }
        wr(o);
    } else if (tag == 'S') {
#include "host_pers_segment.inc"
    } else if (tag == 'G') {
#include "host_pers_4e.inc"
    } else if (tag == 'A') {
        // PERSALL: sites = a fused group of two parts (16 + 24 rows, d_in
        // 2560, two centroid pairs through gs_off) and a single (16 rows,
        // d_in 9728). Served side: tv_tetra48_seg per site; persistent side:
        // tv_tetra48_persall over both, CTAs one at a time (no wait inside).
        const u32 grid = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const u32 variant = rd1<u32>();  // pers::PERSALL_VARIANTS; the served side ignores it
        struct S { TMat m; std::vector<float> gs; std::vector<u32> off; std::vector<float> tail; };
        std::vector<S> sites;
        {
            TMat a = tmat(16, 2560, seed + 1), b = tmat(24, 2560, seed + 2);
            S g{a, {a.gscale[0], a.gscale[1], b.gscale[0], b.gscale[1]}, {}, {}};
            g.m.d_out = 40;
            g.m.words.insert(g.m.words.end(), b.words.begin(), b.words.end());
            g.m.rscale.insert(g.m.rscale.end(), b.rscale.begin(), b.rscale.end());
            g.m.tail.insert(g.m.tail.end(), b.tail.begin(), b.tail.end());
            for (u32 r = 0; r < 40; ++r) g.off.push_back(r < 16 ? 0u : 2u);
            sites.push_back(g);
            TMat c = tmat(16, 9728, seed + 3);
            sites.push_back(S{c, c.gscale, std::vector<u32>(16, 0u), {}});
        }
        for (auto& st : sites) for (auto t : st.m.tail) st.tail.push_back(h2f(t));
        const auto x = act(9728, seed + 4);
        std::vector<float> y(56, 0.0f);
        const F1rTables t = tables();
#if PERS_SIDE == 0
        (void)grid;
        (void)variant;
        u32 at = 0;
        for (auto& st : sites) {
            hostgrid::launch_seq(st.m.d_out / 8, 256, [&] {
                tv_tetra48_seg(st.m.words.data(), st.m.stride, t.rows, t.prefixes, t.branches, t.suffixes, st.gs.data(),
                               st.off.data(), g_inv.data(), st.m.rscale.data(), st.tail.data(), x.data(), y.data() + at,
                               st.m.nblocks, st.m.tail_w);
            });
            at += st.m.d_out;
        }
#else
        std::vector<unsigned long long> words;
        u32 g0 = 0, at = 0;
        for (auto& st : sites) {
            const unsigned long long w[8] = {
                (unsigned long long)(uintptr_t)st.m.words.data(), (unsigned long long)(uintptr_t)st.gs.data(),
                (unsigned long long)(uintptr_t)st.off.data(), (unsigned long long)(uintptr_t)st.m.rscale.data(),
                (unsigned long long)(uintptr_t)st.tail.data(), (unsigned long long)(uintptr_t)(y.data() + at),
                (unsigned long long)st.m.nblocks | (unsigned long long)st.m.tail_w << 16 | (unsigned long long)st.m.stride << 32,
                (unsigned long long)g0 | (unsigned long long)(st.m.d_out / 8) << 32};
            words.insert(words.end(), w, w + 8);
            g0 += st.m.d_out / 8;
            at += st.m.d_out;
        }
        typedef void (*Fn)(const unsigned long long*, u32, const u32*, const unsigned char*, const unsigned short*,
                           const unsigned char*, const float*, const float*, u32);
        const Fn fns[5] = {tv_tetra48_persall, tv_tetra48_persall_pf, tv_tetra48_persall_ts, tv_tetra48_persall_ts_pf,
                           tv_tetra48_persall_ts_pf_l2};
        if (variant >= 5u) { std::fprintf(stderr, "host_pers: persall variant %u\n", variant); return 2; }
        hostgrid::launch_seq(grid, 256, [&] {
            fns[variant](words.data(), 2u, t.rows, t.prefixes, t.branches, t.suffixes, g_inv.data(), x.data(), g0);
        });
#endif
        wr(y);
    } else if (tag == 'H') {
        // HEAD + ARGMAX (4E): vocab rows of an int4 g64 table, d columns.
        // Served: tv_emb_q4_h's binary16 logits, then the argmax by the key of
        // tok_argmax.cu (candle's order). Persistent: pers_head_argmax on a
        // cooperative grid. Writes the token, then every logit. in: vocab, d,
        // grid, threads, plant (0: none, 1: a tie planted at the maximum),
        // seed.
        const u32 vocab = rd1<u32>(), d = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), plant = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        Rng r(seed);
        const u32 gpr = d / 64u;
        std::vector<u32> wq((size_t)vocab * d / 8);
        for (auto& w : wq) w = (u32)r.next();
        std::vector<unsigned short> sc((size_t)vocab * gpr), bi((size_t)vocab * gpr);
        for (auto& v : sc) v = r.h16(6, 9);
        for (auto& v : bi) v = r.h16(7, 10);
        if (plant) {
            // rows 3, 1027 and vocab - 5 get row 0's weights: equal logits
            for (u32 t : {3u, 1027u, vocab - 5u}) {
                for (u32 w = 0; w < d / 8; ++w) wq[(size_t)t * d / 8 + w] = wq[w];
                for (u32 g = 0; g < gpr; ++g) { sc[(size_t)t * gpr + g] = sc[g]; bi[(size_t)t * gpr + g] = bi[g]; }
            }
        }
        const auto xh = act_h(d, seed + 5, 10, 15);
        std::vector<unsigned short> logits(vocab, 0xFFFFu);
        std::vector<u32> token(1, 0xFFFFFFFFu);
#if PERS_SIDE == 0
        (void)grid; (void)threads;
        hostgrid::launch_seq(vocab / 8, 256, [&] { tv_emb_q4_h(wq.data(), sc.data(), bi.data(), xh.data(), logits.data(), d, gpr, 0u, 0u); });
        // the key of tok_argmax.cu, written out here: ord16, bitrev10, the lap
        unsigned long long best = 0ull;
        for (u32 i = 0; i < vocab; ++i) {
            u32 b = logits[i];
            if (b == 0x8000u) b = 0u;
            const u32 ord = (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
            const unsigned long long k = ((unsigned long long)ord << 32) | ((unsigned long long)(1023u - (__brev(i & 1023u) >> 22)) << 22) | (0x3FFFFFu - (i >> 10));
            if (k > best) { best = k; token[0] = i; }
        }
#else
        std::vector<unsigned long long> slot(1, 0ull);
        std::vector<u32> nan(1, 0u), bar(1, 0u), ab(1, 0u), toks(grid, 0u);
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 4 * d + 16, seed, [&] {
            float* xsp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            for (u32 i = threadIdx.x; i < d; i += blockDim.x) xsp[i] = h2f(xh[i]);
            __syncthreads();
            const u32 lo = (u32)((uint64_t)vocab * blockIdx.x / gridDim.x), hi = (u32)((uint64_t)vocab * (blockIdx.x + 1) / gridDim.x);
            const u32 t = pers_head_argmax(wq.data(), sc.data(), bi.data(), xsp, d, gpr, lo, hi, slot.data(), nan.data(),
                                           bar.data(), gridDim.x, ab.data(), logits.data());
            if (threadIdx.x == 0u) toks[blockIdx.x] = t;
        });
        for (u32 c = 1; c < grid; ++c) if (toks[c] != toks[0]) toks[0] = 0xFFFFFFFEu;  // CTAs disagree
        token[0] = toks[0];
#endif
        wr(token);
        wr(logits);
    } else if (tag == 'E') {
        // EMBED (4E): one token's int4 g64 row. Served: emb_q4_gather.
        // Persistent: pers_embed, each CTA writing its slice. in: vocab, d,
        // token, grid, seed. out: u16[d].
        const u32 vocab = rd1<u32>(), d = rd1<u32>(), tok = rd1<u32>(), grid = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        Rng r(seed);
        const u32 gpr = d / 64u;
        std::vector<u32> wq((size_t)vocab * d / 8);
        for (auto& w : wq) w = (u32)r.next();
        std::vector<unsigned short> sc((size_t)vocab * gpr), bi((size_t)vocab * gpr);
        for (auto& v : sc) v = r.h16(6, 9);
        for (auto& v : bi) v = r.h16(7, 10);
        std::vector<unsigned short> y(d, 0xFFFFu);
#if PERS_SIDE == 0
        (void)grid;
        std::vector<u32> ids(1, tok);
        hostgrid::launch_seq(1, 256, [&] { emb_q4_gather(wq.data(), sc.data(), bi.data(), ids.data(), y.data(), d, gpr, 0u); });
#else
        hostgrid::launch_coop(grid, 64, PERS_SMEM_HEADER, seed, [&] {
            const u32 lo = (u32)((uint64_t)d * blockIdx.x / gridDim.x), hi = (u32)((uint64_t)d * (blockIdx.x + 1) / gridDim.x);
            pers_embed(wq.data(), sc.data(), bi.data(), tok, d, gpr, y.data(), lo, hi);
        });
#endif
        wr(y);
    } else if (tag == 'V') {
        // Q/K/V (4E): ln1, the shared q/k rotation, q (4096) and k (1024)
        // Tetra rows, v (1024) int4 rows on the unrotated ln1 output, hidden
        // 2560. Served: the replay of candle's norm, rot_apply, tv_tetra48_h
        // as NVVM compiled it, tv_q4_hh. in: grid, threads, seed, m, k, inv,
        // signs[80], small[1024]. out: u16 q[4096], k[1024], v[1024].
        const u32 grid = rd1<u32>(), threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const u32 m = rd1<u32>(), kk = rd1<u32>();
        const float inv = rd1<float>();
        const auto signs = rd<u32>(80);
        const auto small = rd<float>(1024);
        const u32 H = 2560;
        const auto x = act_h(H, seed + 1, 10, 16);
        const auto alpha = act_h(H, seed + 2, 13, 16);
        const TMat qm = tmat(4096, H, seed + 3), km = tmat(1024, H, seed + 4);
        Rng rv(seed + 5);
        std::vector<u32> vw((size_t)1024 * H / 8);
        for (auto& w : vw) w = (u32)rv.next();
        std::vector<unsigned short> vs(1024 * 20), vb(1024 * 20);
        for (auto& e : vs) e = rv.h16(6, 10);
        for (auto& e : vb) e = rv.h16(8, 12);
        std::vector<unsigned short> q(4096, 0xFFFFu), k(1024, 0xFFFFu), v(1024, 0xFFFFu);
#if PERS_SIDE == 0
        (void)grid; (void)threads;
        const float sc = candle_rms_scale(x.data(), H, 1e-6f, 0);
        std::vector<unsigned short> h0(H);
        for (u32 i = 0; i < H; ++i) h0[i] = f2h(__fmul_rn(__fmul_rn(sc, h2f(x[i])), h2f(alpha[i])));
        std::vector<float> xr(H);
        hostgrid::launch_seq(1, 256, [&] { rot_apply(h0.data(), signs.data(), small.data(), xr.data(), H, m, kk, inv, 0u); });
        q = served_ref(qm, xr);
        k = served_ref(km, xr);
        hostgrid::launch_seq(1024 / 8, 256, [&] { tv_q4_hh(vw.data(), vs.data(), vb.data(), h0.data(), v.data(), H, 20u); });
#else
        const F1rTables t = tables();
        Pers4eQkv P{qm.words.data(), qm.rscale.data(), qm.tail.data(), qm.gscale.data(), qm.stride, 4096u,
                    km.words.data(), km.rscale.data(), km.tail.data(), km.gscale.data(), km.stride, 1024u,
                    vw.data(), vs.data(), vb.data(), 1024u, 20u, qm.nblocks, qm.tail_w, H, q.data(), k.data(), v.data()};
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 8 * H + PERS_4E_SCRATCH, seed, [&] {
            float* xr = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            float* xn = xr + H;
            const float sc = exact_rms_scale<0>(x.data(), H, 1e-6f);
            for (u32 i = threadIdx.x; i < H; i += blockDim.x) {
                const float hv = h2f(exact_rms_apply(sc, x[i], alpha[i]));
                xn[i] = hv;
                xr[i] = ((signs[i >> 5] >> (i & 31u)) & 1u) ? -hv : hv;
            }
            pers_rot_finish(xr, small.data(), H, m, kk, inv);
            pers_4e_qkv_rows(P, t, g_inv.data(), xr, xn, blockIdx.x, gridDim.x, xn + H);
        });
#endif
        wr(q);
        wr(k);
        wr(v);
    } else if (tag == 'W') {
        // O (4E, int4 o as at the 4B and 14B): ctx[4096] -> o rows (2560) ->
        // the residual. Served: tv_q4_hh then badd. in: grid, threads, seed.
        const u32 grid = rd1<u32>(), threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const u32 H = 2560, D = 4096, gpr = D / 128u;
        const auto ctx = act_h(D, seed + 1, 9, 15);
        const auto x = act_h(H, seed + 2, 10, 16);
        Rng r(seed + 3);
        std::vector<u32> ow((size_t)H * D / 8);
        for (auto& w : ow) w = (u32)r.next();
        std::vector<unsigned short> os((size_t)H * gpr), ob((size_t)H * gpr);
        for (auto& e : os) e = r.h16(6, 10);
        for (auto& e : ob) e = r.h16(8, 12);
        std::vector<unsigned short> out(H, 0xFFFFu);
#if PERS_SIDE == 0
        (void)grid; (void)threads;
        std::vector<unsigned short> o(H);
        hostgrid::launch_seq(H / 8, 256, [&] { tv_q4_hh(ow.data(), os.data(), ob.data(), ctx.data(), o.data(), D, gpr); });
        for (u32 i = 0; i < H; ++i) out[i] = f2h(__fadd_rn(h2f(x[i]), h2f(o[i])));
#else
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 4 * D, seed, [&] {
            float* cs = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            const u32 lo = (u32)((uint64_t)H * blockIdx.x / gridDim.x), hi = (u32)((uint64_t)H * (blockIdx.x + 1) / gridDim.x);
            pers_4e_o_int4(ow.data(), os.data(), ob.data(), gpr, ctx.data(), cs, D, x.data(), out.data(), lo, hi);
        });
#endif
        wr(out);
    } else if (tag == 'X') {
        // ATTENTION (4E): one decode row of length l over a cache of `cap`.
        // in: n_kv, groups, cap, l, grid, threads, f32 c, u16 q[n_heads*128],
        // u16 k[n_kv*cap*128], u16 v[same]. out: u16 out[n_heads*128], u32
        // the tickets' sum after the phase (zero: the folds reset them), 0,
        // then f32 state[n_heads*130]. The reference is attn_ref.rs, on the
        // Rust side.
        const u32 n_kv = rd1<u32>(), groups = rd1<u32>(), cap = rd1<u32>(), l = rd1<u32>(), grid = rd1<u32>(),
                  threads = rd1<u32>();
        const float c = rd1<float>();
        const u32 nh = n_kv * groups;
        const auto q = rd<unsigned short>((size_t)nh * 128);
        const auto kc = rd<unsigned short>((size_t)n_kv * cap * 128);
        const auto vc = rd<unsigned short>((size_t)n_kv * cap * 128);
        std::vector<unsigned short> out((size_t)nh * 128, 0xFFFFu);
        std::vector<u32> tail(2, 0u);
        std::vector<float> state((size_t)nh * 130u, 0.0f);
#if PERS_SIDE == 1
        std::vector<float> part((size_t)n_kv * ((cap + 255u) / 256u) * groups * 130u, 0.0f);
        std::vector<u32> ticket(n_kv, 0u);
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + PERS_ATTN_SMEM(groups), 0x5eedu + l, [&] {
            pers_attention<2u>(q.data(), kc.data(), vc.data(), nh, groups, cap, l, c, part.data(), ticket.data(), out.data(),
                           PERS_SMEM_BASE() + PERS_SMEM_HEADER, state.data());
        });
        for (u32 t : ticket) tail[0] += t;
#else
        (void)grid; (void)threads;
        for (u32 hq = 0; hq < nh; ++hq)
            ref_attn_row(q.data() + hq * 128, kc.data() + (size_t)(hq / groups) * cap * 128,
                         vc.data() + (size_t)(hq / groups) * cap * 128, l, c, out.data() + hq * 128);
#endif
        wr(out);
        wr(tail);
        wr(state);
    } else if (tag == 'F') {
        // FUSED f16 FMA: n random triples (all finite patterns) and the
        // planted ties. out: u16[n + ties] per sign, neg then pos.
        const u32 n = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        Rng r(seed);
        std::vector<unsigned short> A, B, C;
        for (u32 i = 0; i < n; ++i) {
            auto fin = [&] { unsigned short h; do { h = (unsigned short)r.next(); } while ((h & 0x7c00u) == 0x7c00u); return h; };
            A.push_back(fin()); B.push_back(fin()); C.push_back(fin());
        }
        // 1.5 x (1 + 2^-10) = 1.5 + 2^-10 + 2^-11: a binary16 tie, decided by
        // the third term's sign or, at zero, by evenness
        for (unsigned short cc : {(unsigned short)0x0000u, (unsigned short)0x0001u, (unsigned short)0x8001u, (unsigned short)0x0400u, (unsigned short)0x8400u}) {
            A.push_back(0x3e00u); B.push_back(0x3c01u); C.push_back(cc);
        }
        std::vector<unsigned short> out;
        for (int neg = 0; neg < 2; ++neg)
            for (size_t i = 0; i < A.size(); ++i) {
#if PERS_SIDE == 0
                out.push_back(ref_f16_fma(A[i], B[i], C[i], neg != 0));
#else
                out.push_back(pers_f16_fma_exact(A[i], B[i], C[i], neg != 0));
#endif
            }
        wr(out);
    } else if (tag == 'Z') {
        // Q/K-NORM, ROPE, KV APPEND (4E). in: form, n_heads, n_kv, cap, pos,
        // grid, threads, seed. out: u16 q_out[n_heads*128], kc, vc [n_kv*cap*128].
        const u32 form = rd1<u32>(), nh = rd1<u32>(), nkv = rd1<u32>(), cap = rd1<u32>(), pos = rd1<u32>(),
                  grid = rd1<u32>(), threads = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
        const auto qr = act_h(nh * 128, seed + 1, 8, 17);
        const auto kr = act_h(nkv * 128, seed + 2, 8, 17);
        const auto vr = act_h(nkv * 128, seed + 3, 8, 17);
        const auto qa = act_h(128, seed + 4, 13, 16), ka = act_h(128, seed + 5, 13, 16);
        const auto co = act_h(64, seed + 6, 4, 14), si = act_h(64, seed + 7, 4, 14);
        std::vector<unsigned short> qo(nh * 128, 0xFFFFu), kc((size_t)nkv * cap * 128, 0xAAAAu), vc((size_t)nkv * cap * 128, 0xAAAAu);
        const float eps = 1e-6f;
#if PERS_SIDE == 0
        (void)grid; (void)threads;
        auto head = [&](const unsigned short* src, const unsigned short* al, unsigned short* dst) {
            const float sc = candle_rms_scale(src, 128, eps, 0);
            for (u32 i = 0; i < 64; ++i) {
                const unsigned short a = f2h(__fmul_rn(__fmul_rn(sc, h2f(src[i])), h2f(al[i])));
                const unsigned short b = f2h(__fmul_rn(__fmul_rn(sc, h2f(src[i + 64])), h2f(al[i + 64])));
                ref_rope_pair((int)form, a, b, co[i], si[i], dst + i, dst + i + 64);
            }
        };
        for (u32 h = 0; h < nh; ++h) head(qr.data() + h * 128, qa.data(), qo.data() + h * 128);
        for (u32 h = 0; h < nkv; ++h) {
            head(kr.data() + h * 128, ka.data(), kc.data() + ((size_t)h * cap + pos) * 128);
            for (u32 i = 0; i < 128; ++i) vc[((size_t)h * cap + pos) * 128 + i] = vr[h * 128 + i];
        }
#else
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER, seed, [&] {
            if (form == 0) pers_4e_norm_rope_append<0>(qr.data(), kr.data(), vr.data(), qa.data(), ka.data(), co.data(), si.data(), eps, nh, nkv, cap, pos, qo.data(), kc.data(), vc.data());
            else if (form == 1) pers_4e_norm_rope_append<1>(qr.data(), kr.data(), vr.data(), qa.data(), ka.data(), co.data(), si.data(), eps, nh, nkv, cap, pos, qo.data(), kc.data(), vc.data());
            else pers_4e_norm_rope_append<2>(qr.data(), kr.data(), vr.data(), qa.data(), ka.data(), co.data(), si.data(), eps, nh, nkv, cap, pos, qo.data(), kc.data(), vc.data());
        });
#endif
        wr(qo);
        wr(kc);
        wr(vc);
    } else if (tag == 'L') {
        // LAYOUT of the structs the host mirrors (pers.rs): sizes and offsets.
        std::vector<u32> o;
#if PERS_SIDE == 1
#define OFF(T, f) o.push_back((u32)__builtin_offsetof(T, f))
        o.push_back((u32)sizeof(SegTetra));
        OFF(SegTetra, words); OFF(SegTetra, rscale); OFF(SegTetra, tail); OFF(SegTetra, gscale);
        OFF(SegTetra, stride_u32); OFF(SegTetra, nblocks); OFF(SegTetra, tail_w); OFF(SegTetra, d_out);
        o.push_back((u32)sizeof(SegArgs));
        OFF(SegArgs, alpha); OFF(SegArgs, signs_h); OFF(SegArgs, small_h); OFF(SegArgs, silu); OFF(SegArgs, act);
        OFF(SegArgs, gate); OFF(SegArgs, up); OFF(SegArgs, down); OFF(SegArgs, signs_i); OFF(SegArgs, small_i);
        OFF(SegArgs, q4w); OFF(SegArgs, q4s); OFF(SegArgs, q4b); OFF(SegArgs, rows); OFF(SegArgs, pref);
        OFF(SegArgs, br); OFF(SegArgs, suf); OFF(SegArgs, invnorm); OFF(SegArgs, slice_gu); OFF(SegArgs, slice_dn);
        OFF(SegArgs, bar); OFF(SegArgs, abort_flag); OFF(SegArgs, hidden); OFF(SegArgs, inter); OFF(SegArgs, m_h);
        OFF(SegArgs, k_h); OFF(SegArgs, m_i); OFF(SegArgs, k_i); OFF(SegArgs, inv_h); OFF(SegArgs, inv_i);
        OFF(SegArgs, eps); OFF(SegArgs, gpr);
        o.push_back((u32)sizeof(SegLaunch));
        OFF(SegLaunch, x); OFF(SegLaunch, out); OFF(SegLaunch, bar_base);
        o.push_back((u32)sizeof(Pers4eQkv));
        OFF(Pers4eQkv, kw); OFF(Pers4eQkv, k_stride); OFF(Pers4eQkv, vw); OFF(Pers4eQkv, v_dout); OFF(Pers4eQkv, nblocks);
        OFF(Pers4eQkv, d_in); OFF(Pers4eQkv, q); OFF(Pers4eQkv, v);
        o.push_back((u32)sizeof(Pers4eLayer));
        OFF(Pers4eLayer, m_a); OFF(Pers4eLayer, inv_a); OFF(Pers4eLayer, qkv); OFF(Pers4eLayer, q_norm); OFF(Pers4eLayer, obi);
        OFF(Pers4eLayer, o_gpr); OFF(Pers4eLayer, tetra_down); OFF(Pers4eLayer, mlp); OFF(Pers4eLayer, ot);
        o.push_back((u32)sizeof(Pers4eArgs));
        OFF(Pers4eArgs, invnorm); OFF(Pers4eArgs, attn_ticket); OFF(Pers4eArgs, n_layers); OFF(Pers4eArgs, cap);
        OFF(Pers4eArgs, eps); OFF(Pers4eArgs, c); OFF(Pers4eArgs, head_w); OFF(Pers4eArgs, head_b);
        o.push_back((u32)sizeof(Pers4eOT));
        OFF(Pers4eOT, t); OFF(Pers4eOT, signs); OFF(Pers4eOT, small); OFF(Pers4eOT, m); OFF(Pers4eOT, k); OFF(Pers4eOT, inv);
        o.push_back((u32)sizeof(Pers4eLaunch));
        OFF(Pers4eLaunch, eos); OFF(Pers4eLaunch, ring); OFF(Pers4eLaunch, stamps);
#undef OFF
#endif
        wr(o);
    } else if (tag == 'P') {
        // The verify pass's launch struct (pers_4ev.cu): its size and the
        // offsets of its fields, for pers.rs::P4vLaunch.
        std::vector<u32> o;
#if PERS_SIDE == 1 && PERS_4E_ROWS >= 7
#define OFF(T, f) o.push_back((u32)offsetof(T, f))
        o.push_back((u32)sizeof(Pers4vLaunch));
        OFF(Pers4vLaunch, act_floats); OFF(Pers4vLaunch, toks); OFF(Pers4vLaunch, ring); OFF(Pers4vLaunch, x_a);
        OFF(Pers4vLaunch, act); OFF(Pers4vLaunch, part); OFF(Pers4vLaunch, tickets); OFF(Pers4vLaunch, slots);
        OFF(Pers4vLaunch, part_stride);
        o.push_back(P4V_KMAX);
#undef OFF
#endif
        wr(o);
    } else if (tag == 'y' || tag == 'c') {
#include "host_pers_step1.inc"
    } else if (tag == 'K') {
        // KEY: the persistent key against tok_argmax.cu's, every binary16
        // pattern at a spread of indices. Writes the number of mismatches.
        std::vector<u32> bad(1, 0u);
#if PERS_SIDE == 1
        const u32 idx[] = {0u, 1u, 2u, 511u, 512u, 1023u, 1024u, 1025u, 4096u + 7u, 151935u, 151936u - 1024u, 77777u};
        for (u32 b = 0; b < 65536u; ++b)
            for (u32 i : idx) {
                if (pers_tie_key(b, i) != tie_key(b, i)) ++bad[0];
                if (pers_is_nan16(b) != is_nan16(b)) ++bad[0];
            }
#endif
        wr(bad);
    } else {
        std::fprintf(stderr, "host_pers: unknown command %u\n", tag);
        return 2;
    }
    std::fprintf(stderr, "host_pers: %llu fiber switches\n", hostgrid::switches);
    return 0;
}
