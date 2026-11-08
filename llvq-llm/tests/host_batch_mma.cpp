// Executes the batched projections of tv_batch_mma.cu on the CPU, whole CTAs
// at a time through host_cta.h (one fiber a thread, real barriers, real
// tickets), and prints the f16 outputs beside a double-precision reference.
//
// The MMA is modelled from the PTX ISA's fragment layout for
// mma.m16n8k16.row.col.f32.f16.f16.f32: every lane deposits its fragments,
// a barrier, each lane rebuilds the warp's 16x16 A and 16x8 B and computes its
// four outputs, a barrier. The products of two halves are exact in f32; the
// sum is formed in k order, which the card does not promise, so the Rust side
// compares under a tolerance. What this file establishes is the indexing:
// fragment layouts, the staging, the decode into shared memory, the scales,
// the split-K fold. A wrong index is an error of the order of the output, not
// of its last bit.
//
//   in : u32 mode (0 Tetra, 1 int4), then
//        Tetra: u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes,
//               f32[2] gscale, f32[32] invnorm, u32 d_out, nblocks, tail_w, n_rows, splits,
//               row_stride_u32, u32[d_out*row_stride] words, f32[d_out] rscale,
//               u16[d_out*tail_w] tail, f32[n_rows*d_in] x
//        int4:  u32 d_out, d_in, gshift, n_rows, splits, u32[d_out*d_in/8] words,
//               u16[d_out*gpr] scales, u16[d_out*gpr] biases, u16[n_rows*d_in] x
//   out: u16[n_rows*d_out] y (row n at n*d_out), f32[n_rows*d_out] reference

#include "../../llvq-cuda/tests/host_cta.h"

// host_cta.h makes `__shared__` static, which cannot be `extern`: the kernels'
// dynamic allocation is a plain global here, exact since one CTA runs at a time.
#undef __shared__
#define __shared__

#include <cmath>
#include <vector>

// The scalar intrinsics the decoder headers use, as in host_shim.h.
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
// matvec.cu's butterfly names a 3-argument shuffle; nothing executed here calls it.
static inline float __shfl_xor_sync(unsigned, float, int) { std::abort(); }

#define TILE_BLOCKS 64u
#include "../../llvq-cuda/kernels/llvq_slot.cuh"
#include "../../llvq-cuda/kernels/matvec.cu"
#include "../../llvq-cuda/kernels/llvq_f1rank.cuh"
#include "../../llvq-cuda/kernels/llvq_f1rank_v3.cuh"
#include "../../llvq-cuda/kernels/llvq_tetra48.cuh"
#include "../kernels/tv_batch_mma.cu"

u32 bm_smem[16384];
float xs[1];

static u32 mma_a[128][4], mma_b[128][2];
static float mma_d[128][4];

void bm_mma_host(float* d, const u32* a, u32 b0, u32 b1)
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

static unsigned nt_for(u32 n_rows) { return n_rows <= 8 ? 1u : n_rows <= 16 ? 2u : 4u; }

int main()
{
    const u32 mode = rd1();
    if (mode == 0) {
        auto trows = rd<u32>(4096);
        auto pre = rd<unsigned char>(128);
        auto br = rd<unsigned short>(1024);
        auto suf = rd<unsigned char>(128);
        auto gs = rd<float>(2);
        auto inv = rd<float>(32);
        const u32 d_out = rd1(), nblocks = rd1(), tail_w = rd1(), n_rows = rd1(), splits = rd1(), rs = rd1();
        const u32 d_in = nblocks * LLVQ_DIM + tail_w;
        auto words = rd<u32>((std::size_t)d_out * rs);
        auto rscale = rd<float>(d_out);
        auto tail = rd<unsigned short>((std::size_t)d_out * tail_w);
        auto x = rd<float>((std::size_t)n_rows * d_in);
        const u32 slice = (nblocks + splits - 1) / splits;
        const u32 S = (nblocks + slice - 1) / slice;
        std::vector<unsigned short> y((std::size_t)n_rows * d_out, 0xffffu);
        std::vector<float> part((std::size_t)S * n_rows * d_out);
        std::vector<u32> ticket(d_out / BM_ROWS, 0u);
        const unsigned nt = nt_for(n_rows);
        hostcta::launch(d_out / BM_ROWS, S, 128, 0x9e3779b9u, [&] {
            if (nt == 1) tv_tetra48_mma_n8(words.data(), rs, trows.data(), pre.data(), br.data(), suf.data(), gs.data(), inv.data(), rscale.data(), tail.data(), x.data(), y.data(), nblocks, tail_w, n_rows, d_in, d_out, slice, part.data(), ticket.data());
            if (nt == 2) tv_tetra48_mma_n16(words.data(), rs, trows.data(), pre.data(), br.data(), suf.data(), gs.data(), inv.data(), rscale.data(), tail.data(), x.data(), y.data(), nblocks, tail_w, n_rows, d_in, d_out, slice, part.data(), ticket.data());
            if (nt == 4) tv_tetra48_mma_n32(words.data(), rs, trows.data(), pre.data(), br.data(), suf.data(), gs.data(), inv.data(), rscale.data(), tail.data(), x.data(), y.data(), nblocks, tail_w, n_rows, d_in, d_out, slice, part.data(), ticket.data());
        });
        for (u32 t : ticket)
            if (t != 0) { std::fprintf(stderr, "a ticket did not reset\n"); return 3; }
        std::vector<float> ref((std::size_t)n_rows * d_out);
        F1rTables tab = { trows.data(), pre.data(), br.data(), suf.data() };
        for (u32 row = 0; row < d_out; ++row) {
            std::vector<double> w((std::size_t)d_in, 0.0);
            for (u32 j = 0; j < nblocks; ++j) {
                u32 lo, hi16;
                f1r_load(words.data() + (std::size_t)row * rs, j, lo, hi16);
                float yv[24];
                tetra48_decode_f(lo, hi16, tab, gs.data(), inv.data(), yv);
                for (u32 c = 0; c < 24; ++c) w[j * 24 + c] = (double)yv[c] * rscale[row];
            }
            for (u32 c = 0; c < tail_w; ++c) w[nblocks * 24 + c] = h2f(tail[(std::size_t)row * tail_w + c]);
            for (u32 n = 0; n < n_rows; ++n) {
                double s = 0.0;
                for (u32 c = 0; c < d_in; ++c) s += w[c] * x[(std::size_t)n * d_in + c];
                ref[(std::size_t)n * d_out + row] = (float)s;
            }
        }
        std::fwrite(y.data(), 2, y.size(), stdout);
        std::fwrite(ref.data(), 4, ref.size(), stdout);
        return 0;
    }
    const u32 d_out = rd1(), d_in = rd1(), gshift = rd1(), n_rows = rd1(), splits = rd1();
    const u32 gpr = d_in >> gshift;
    auto words = rd<u32>((std::size_t)d_out * d_in / 8);
    auto sc = rd<unsigned short>((std::size_t)d_out * gpr);
    auto bi = rd<unsigned short>((std::size_t)d_out * gpr);
    auto x = rd<unsigned short>((std::size_t)n_rows * d_in);
    const u32 nunits = d_in / BM_UNIT;
    const u32 slice = (nunits + splits - 1) / splits;
    const u32 S = (nunits + slice - 1) / slice;
    std::vector<unsigned short> y((std::size_t)n_rows * d_out, 0xffffu);
    std::vector<float> part((std::size_t)S * n_rows * d_out);
    std::vector<u32> ticket(d_out / BM_ROWS, 0u);
    const unsigned nt = nt_for(n_rows);
    hostcta::launch(d_out / BM_ROWS, S, 128, 0x51ed27u, [&] {
        if (nt == 1) tv_q4_mma_n8(words.data(), sc.data(), bi.data(), x.data(), y.data(), d_in, d_out, gpr, gshift, n_rows, slice, part.data(), ticket.data());
        if (nt == 2) tv_q4_mma_n16(words.data(), sc.data(), bi.data(), x.data(), y.data(), d_in, d_out, gpr, gshift, n_rows, slice, part.data(), ticket.data());
        if (nt == 4) tv_q4_mma_n32(words.data(), sc.data(), bi.data(), x.data(), y.data(), d_in, d_out, gpr, gshift, n_rows, slice, part.data(), ticket.data());
    });
    for (u32 t : ticket)
        if (t != 0) { std::fprintf(stderr, "a ticket did not reset\n"); return 3; }
    std::vector<float> ref((std::size_t)n_rows * d_out);
    for (u32 row = 0; row < d_out; ++row)
        for (u32 n = 0; n < n_rows; ++n) {
            double s = 0.0;
            for (u32 c = 0; c < d_in; ++c) {
                const u32 q = (words[((std::size_t)row * d_in + c) / 8] >> (4 * (c % 8))) & 0xfu;
                const std::size_t g = (std::size_t)row * gpr + (c >> gshift);
                s += ((double)h2f(sc[g]) * q + h2f(bi[g])) * h2f(x[(std::size_t)n * d_in + c]);
            }
            ref[(std::size_t)n * d_out + row] = (float)s;
        }
    std::fwrite(y.data(), 2, y.size(), stdout);
    std::fwrite(ref.data(), 4, ref.size(), stdout);
    return 0;
}
