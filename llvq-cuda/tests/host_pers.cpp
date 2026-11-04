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
#include "../../llvq-llm/kernels/emb_q4.cu"
#undef f2h
#else
#include "../kernels/llvq_pers.cuh"
#include "../kernels/tetra48_pers.cu"
#include "../kernels/pers_segment.cu"
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
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER + 4 * n, seed, [&] {
            float* sp = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            pers_rot_load(xin.data(), signs.data(), sp, n);
            pers_rot_finish(sp, small.data(), n, m, k, inv);
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) out[i] = sp[i];
        });
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
            for (u32 j = threadIdx.x; j < m; j += 32u) pers_mix_column(sv.data(), small.data(), m, k, inv, j);
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
                const float acc = pers_q4_row(wq.data(), sc.data(), bi.data(), xsp, row, d_in, gpr, lane);
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
    } else if (tag == 'B') {
#include "host_pers_bench.inc"
    } else if (tag == 'S') {
#include "host_pers_segment.inc"
    } else if (tag == 'A') {
        // PERSALL: sites = a fused group of two parts (16 + 24 rows, d_in
        // 2560, two centroid pairs through gs_off) and a single (16 rows,
        // d_in 9728). Served side: tv_tetra48_seg per site; persistent side:
        // tv_tetra48_persall over both, CTAs one at a time (no wait inside).
        const u32 grid = rd1<u32>();
        const uint64_t seed = rd1<uint64_t>();
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
        hostgrid::launch_seq(grid, 256, [&] {
            tv_tetra48_persall(words.data(), 2u, t.rows, t.prefixes, t.branches, t.suffixes, g_inv.data(), x.data(), g0);
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
    } else if (tag == 'L') {
        // LAYOUT of the structs the host mirrors (pers.rs): sizes and offsets.
        std::vector<u32> o;
#if PERS_SIDE == 1
#define OFF(T, f) o.push_back((u32)__builtin_offsetof(T, f))
        o.push_back((u32)sizeof(PersArgs));
        OFF(PersArgs, sites); OFF(PersArgs, rows); OFF(PersArgs, pref); OFF(PersArgs, br); OFF(PersArgs, suf);
        OFF(PersArgs, invnorm); OFF(PersArgs, gscale); OFF(PersArgs, rscale); OFF(PersArgs, tail); OFF(PersArgs, x);
        OFF(PersArgs, y_all); OFF(PersArgs, slice_lo); OFF(PersArgs, slice_hi); OFF(PersArgs, bar); OFF(PersArgs, cnts);
        OFF(PersArgs, abort_flag); OFF(PersArgs, resid); OFF(PersArgs, alpha); OFF(PersArgs, signs_h);
        OFF(PersArgs, signs_i); OFF(PersArgs, small_h); OFF(PersArgs, small_i);
        OFF(PersArgs, nsites); OFF(PersArgs, hidden); OFF(PersArgs, inter); OFF(PersArgs, m_h); OFF(PersArgs, k_h);
        OFF(PersArgs, m_i); OFF(PersArgs, k_i); OFF(PersArgs, inv_h); OFF(PersArgs, inv_i); OFF(PersArgs, eps);
        o.push_back((u32)sizeof(PersLaunch));
        OFF(PersLaunch, bar_base); OFF(PersLaunch, epoch); OFF(PersLaunch, cnt_offset); OFF(PersLaunch, dbg);
        OFF(PersLaunch, dbg_norm); OFF(PersLaunch, dbg_rot);
        o.push_back((u32)sizeof(PersSite));
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
#undef OFF
#endif
        wr(o);
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
