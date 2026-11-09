// Executes the batched GEMM of tv_batch_gemm.cu on the CPU, whole CTAs at a
// time through host_cta.h (one fiber a thread, real barriers, real tickets, a
// permuted CTA order), and prints the f16 outputs beside a double-precision
// reference and an error bound.
//
// The MMA is modelled from the PTX fragment layout of
// mma.m16n8k16.row.col.f32.f16.f16.f32, as in host_batch_mma.cpp. cp.async is
// a synchronous copy. What this file establishes is the indexing: the chunk
// and sub-step offsets, the staging of both kinds, the folded scales, the
// tail, several projections in one launch, the split-K fold. The bound is the
// worst case of the two f16 roundings (weight, activation), 2^-10 of
// sum |w x|, which a wrong index exceeds by orders of magnitude.
//
//   in : u32 n_rows, u32 nseg,
//        u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes, f32[32] invnorm,
//        then per projection, u32 kind:
//          0 Tetra: u32 d_out, nblocks, tail_w, row_stride, splits; f32[2] gscale;
//                   u32[d_out*row_stride] words; f32[d_out] rscale; u16[d_out*tail_w] tail;
//                   u16[n_rows*d_in] x (the rotated rows, already f16)
//          1 int4:  u32 d_out, d_in, gshift, splits, tiled; u32[d_out*d_in/8] words (file order),
//                   then when tiled the same words in 4E v9's tiled order, which the kernel reads;
//                   u16[d_out*gpr] scales; u16[d_out*gpr] biases; u16[n_rows*d_in] x
//   out: per projection, u16[n_rows*d_out] y, f32[n_rows*d_out] reference, f32[n_rows*d_out] bound

#define HOSTCTA_MAX_THREADS 256
#include "../../llvq-cuda/tests/host_cta.h"

#undef __shared__
#define __shared__
#define __align__(n)
#define __launch_bounds__(...)

#include <cmath>
#include <cstdlib>
#include <vector>

static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
static inline float __uint_as_float(unsigned i) { float f; __builtin_memcpy(&f, &i, 4); return f; }
static inline unsigned __byte_perm(unsigned a, unsigned b, unsigned s) {
    unsigned long long src = ((unsigned long long)b << 32) | a;
    unsigned d = 0;
    for (unsigned k = 0; k < 4; ++k) {
        unsigned ctl = (s >> (4 * k)) & 0xfu;
        unsigned byte = (unsigned)(src >> (8 * (ctl & 7u))) & 0xffu;
        if (ctl & 8u) byte = (byte & 0x80u) ? 0xffu : 0x00u;
        d |= byte << (8 * k);
    }
    return d;
}
static inline int __dp4a(int a, int b, int c) {
    for (unsigned k = 0; k < 4; ++k) c += (int)(signed char)((unsigned)a >> (8 * k)) * (int)(signed char)((unsigned)b >> (8 * k));
    return c;
}
static inline float __shfl_xor_sync(unsigned, float, int) { std::abort(); }

#define TILE_BLOCKS 64u
#include "../../llvq-cuda/kernels/llvq_slot.cuh"
#include "../../llvq-cuda/kernels/matvec.cu"
#include "../../llvq-cuda/kernels/llvq_f1rank.cuh"
#include "../../llvq-cuda/kernels/llvq_f1rank_v3.cuh"
#include "../../llvq-cuda/kernels/llvq_tetra48.cuh"
#include "../kernels/tv_batch_gemm.cu"

u32 bg_smem[32768];
float xs[1];

static u32 mma_a[256][4], mma_b[256][2];
static float mma_d[256][4];

void bg_mma_host(float* d, const u32* a, u32 b0, u32 b1)
{
    const unsigned t = threadIdx.x;
    for (int i = 0; i < 4; ++i) {
        mma_a[t][i] = a[i];
        mma_d[t][i] = d[i];
    }
    mma_b[t][0] = b0;
    mma_b[t][1] = b1;
    __syncthreads();
    const unsigned w0 = t & ~31u, lane = t & 31u, gid = lane >> 2, tig = lane & 3u;
    float A[16][16], B[16][8];
    auto lo = [](u32 v) { return h2f((unsigned short)(v & 0xffffu)); };
    auto hi = [](u32 v) { return h2f((unsigned short)(v >> 16)); };
    for (unsigned L = 0; L < 32; ++L) {
        const unsigned g = L >> 2, q = L & 3u;
        const u32* aa = mma_a[w0 + L];
        A[g][2 * q] = lo(aa[0]);         A[g][2 * q + 1] = hi(aa[0]);
        A[g + 8][2 * q] = lo(aa[1]);     A[g + 8][2 * q + 1] = hi(aa[1]);
        A[g][2 * q + 8] = lo(aa[2]);     A[g][2 * q + 9] = hi(aa[2]);
        A[g + 8][2 * q + 8] = lo(aa[3]); A[g + 8][2 * q + 9] = hi(aa[3]);
        const u32* bb = mma_b[w0 + L];
        B[2 * q][g] = lo(bb[0]);     B[2 * q + 1][g] = hi(bb[0]);
        B[2 * q + 8][g] = lo(bb[1]); B[2 * q + 9][g] = hi(bb[1]);
    }
    float out[4];
    for (unsigned e = 0; e < 4; ++e) {
        const unsigned r = gid + (e >> 1) * 8u, c = 2u * tig + (e & 1u);
        float s = mma_d[t][e];
        for (unsigned k = 0; k < 16; ++k) s += A[r][k] * B[k][c];
        out[e] = s;
    }
    __syncthreads();
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}

template <typename T>
static std::vector<T> rd(std::size_t n) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, stdin) != n) {
        std::fprintf(stderr, "short fixture\n");
        std::exit(2);
    }
    return v;
}
static u32 rd1() { return rd<u32>(1)[0]; }

struct Proj {
    u32 kind, d_out, d_in, nblocks, tail_w, rs, gshift, splits, tiled = 0;
    std::vector<u32> dev_words;  // what the kernel reads
    std::vector<float> gs, rscale;
    std::vector<u32> words;
    std::vector<unsigned short> tail, sc, bi, xh, y;
    std::vector<float> part;
};

static u64 P(const void* p) { return (u64)(uintptr_t)p; }

int main()
{
    const u32 n_rows = rd1(), nseg = rd1();
    auto trows = rd<u32>(4096);
    auto pre = rd<unsigned char>(128);
    auto br = rd<unsigned short>(1024);
    auto suf = rd<unsigned char>(128);
    auto inv = rd<float>(32);
    std::vector<Proj> ps(nseg);
    for (auto& p : ps) {
        p.kind = rd1();
        if (p.kind == 0) {
            p.d_out = rd1(); p.nblocks = rd1(); p.tail_w = rd1(); p.rs = rd1(); p.splits = rd1();
            p.d_in = p.nblocks * LLVQ_DIM + p.tail_w;
            p.gs = rd<float>(2);
            p.words = rd<u32>((std::size_t)p.d_out * p.rs);
            p.rscale = rd<float>(p.d_out);
            p.tail = rd<unsigned short>((std::size_t)p.d_out * p.tail_w);
            p.xh = rd<unsigned short>((std::size_t)n_rows * p.d_in);
        } else {
            p.d_out = rd1(); p.d_in = rd1(); p.gshift = rd1(); p.splits = rd1(); p.tiled = rd1();
            const u32 gpr = p.d_in >> p.gshift;
            p.words = rd<u32>((std::size_t)p.d_out * p.d_in / 8);
            // Tiled: the loader's own permutation (llvq_cuda::pers::q4_tile_words)
            // follows, and the kernel reads it; the reference reads the file order.
            p.dev_words = p.tiled ? rd<u32>((std::size_t)p.d_out * p.d_in / 8) : p.words;
            p.sc = rd<unsigned short>((std::size_t)p.d_out * gpr);
            p.bi = rd<unsigned short>((std::size_t)p.d_out * gpr);
            p.xh = rd<unsigned short>((std::size_t)n_rows * p.d_in);
        }
    }
    const u32 nt = n_rows <= 8 ? 1 : n_rows <= 16 ? 2 : n_rows <= 32 ? 4 : n_rows <= 64 ? 8 : 16;
    // The K chunk of each entry (BG_ENTRY lines of tv_batch_gemm.cu): the
    // served set, or the two-CTA set when HBG_OCC is set.
    const bool occ = std::getenv("HBG_OCC") != nullptr;
    const u32 kc = occ ? (nt == 1 ? 192 : nt <= 4 ? 96 : 48) : (nt <= 4 ? 192 : 96);

    BgArgs a;
    __builtin_memset(&a, 0, sizeof a);
    a.trows = P(trows.data()); a.pref = P(pre.data()); a.br = P(br.data()); a.suf = P(suf.data()); a.invnorm = P(inv.data());
    a.nseg = nseg; a.n_rows = n_rows;
    u32 cta = 0, tk = 0;
    for (u32 i = 0; i < nseg; ++i) {
        Proj& p = ps[i];
        BgSeg& s = a.seg[i];
        const u32 nk = (p.d_in + kc - 1) / kc;
        const u32 slice = (nk + p.splits - 1) / p.splits;
        const u32 S = (nk + slice - 1) / slice;
        p.y.assign((std::size_t)n_rows * p.d_out, 0xffffu);
        p.part.assign((std::size_t)S * n_rows * p.d_out, NAN);
        s.kind = p.kind; s.d_in = p.d_in; s.d_out = p.d_out; s.nk = nk; s.slice = slice; s.S = S;
        s.cta0 = cta; s.tk0 = tk;
        s.y = P(p.y.data()); s.part = P(p.part.data());
        if (p.kind == 0) {
            s.w = P(p.words.data()); s.a0 = P(p.gs.data()); s.a1 = P(p.rscale.data()); s.tail = P(p.tail.data());
            s.x = P(p.xh.data()); s.stride = p.rs; s.nblocks = p.nblocks; s.tail_w = p.tail_w;
        } else {
            s.w = P(p.dev_words.data()); s.a0 = P(p.sc.data()); s.a1 = P(p.bi.data()); s.x = P(p.xh.data());
            s.tiled = p.tiled;
            s.stride = p.d_in >> p.gshift; s.nblocks = p.gshift;
        }
        cta += p.d_out / BG_ROWS * S;
        tk += p.d_out / BG_ROWS;
    }
    std::vector<u32> ticket(tk, 0u);
    a.ticket = P(ticket.data());
    hostcta::launch(cta, 1, BG_THREADS, 0x2545f491u, [&] {
        switch (nt) {
        case 1: occ ? tv_bgo_n8(a) : tv_bg_n8(a); break;
        case 2: occ ? tv_bgo_n16(a) : tv_bg_n16(a); break;
        case 4: occ ? tv_bgo_n32(a) : tv_bg_n32(a); break;
        case 8: occ ? tv_bgo_n64(a) : tv_bg_n64(a); break;
        default: occ ? tv_bgo_n128(a) : tv_bg_n128(a); break;
        }
    });
    for (u32 t : ticket)
        if (t != 0) { std::fprintf(stderr, "a ticket did not reset\n"); return 3; }

    F1rTables tab = { trows.data(), pre.data(), br.data(), suf.data() };
    for (auto& p : ps) {
        std::vector<float> ref((std::size_t)n_rows * p.d_out), bound(ref.size());
        std::vector<double> w(p.d_in);
        for (u32 row = 0; row < p.d_out; ++row) {
            if (p.kind == 0) {
                for (u32 j = 0; j < p.nblocks; ++j) {
                    u32 lo, hi16;
                    f1r_load(p.words.data() + (std::size_t)row * p.rs, j, lo, hi16);
                    float yv[24];
                    tetra48_decode_f(lo, hi16, tab, p.gs.data(), inv.data(), yv);
                    for (u32 c = 0; c < 24; ++c) w[j * 24 + c] = (double)yv[c] * p.rscale[row];
                }
                for (u32 c = 0; c < p.tail_w; ++c) w[p.nblocks * 24 + c] = h2f(p.tail[(std::size_t)row * p.tail_w + c]);
            } else {
                const u32 gpr = p.d_in >> p.gshift;
                for (u32 c = 0; c < p.d_in; ++c) {
                    const u32 q = (p.words[((std::size_t)row * p.d_in + c) / 8] >> (4 * (c % 8))) & 0xfu;
                    const std::size_t g = (std::size_t)row * gpr + (c >> p.gshift);
                    w[c] = (double)h2f(p.sc[g]) * q + h2f(p.bi[g]);
                }
            }
            for (u32 n = 0; n < n_rows; ++n) {
                double s = 0.0, m = 0.0;
                for (u32 c = 0; c < p.d_in; ++c) {
                    const double x = (double)h2f(p.xh[(std::size_t)n * p.d_in + c]);
                    s += w[c] * x;
                    m += std::fabs(w[c] * x);
                }
                ref[(std::size_t)n * p.d_out + row] = (float)s;
                bound[(std::size_t)n * p.d_out + row] = (float)(m * (std::ldexp(1.0, -10) + std::ldexp(1.0, -16)));
            }
        }
        std::fwrite(p.y.data(), 2, p.y.size(), stdout);
        std::fwrite(ref.data(), 4, ref.size(), stdout);
        std::fwrite(bound.data(), 4, bound.size(), stdout);
    }
    return 0;
}
