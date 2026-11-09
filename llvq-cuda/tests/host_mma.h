// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 for one warp of
// host_grid.h, from the PTX fragment layout. Lane (gid, tig) = (lane / 4,
// lane % 4) holds rows gid and gid + 8 of A (a0: row gid, columns 2 tig and
// 2 tig + 1; a1: row gid + 8; a2 and a3: the same 8 columns on), column gid of
// B (b0: rows 2 tig and 2 tig + 1; b1: 8 rows on), and D's rows gid and
// gid + 8 at columns 2 tig and 2 tig + 1 (d0, d1, d2, d3). Each pair is two
// binary16, the lower index in the low half.
//
// Each lane leaves its fragments in slots double-buffered by the parity of
// its warp barrier's generation (as host_grid.h's shuffles), the warp meets
// once, and each lane computes its four outputs: C, then the 16 products in
// increasing k, each product exact in binary32 (two 11-bit significands) and
// added with one rounding. The card's tensor core sums in an order and a
// precision of its own: what this model fixes is which operands meet, so a
// fragment index, a column mix-up or a row read from the wrong slot shows
// here, and an output column depends only on its own column of B, as on the
// card. Include after host_grid.h and matvec.cu (h2f).

#ifndef LLVQ_HOST_MMA_H
#define LLVQ_HOST_MMA_H

#include <vector>

static std::vector<u32> g_mma_a[2], g_mma_b[2];

void p4t_mma_host(float* d, const u32* a, u32 b0, u32 b1)
{
    const size_t n = hostgrid::fibers.size();
    if (g_mma_a[0].size() < 4 * n) {
        for (auto& v : g_mma_a) v.resize(4 * n);
        for (auto& v : g_mma_b) v.resize(2 * n);
    }
    const size_t me = (size_t)hostgrid::current;
    hostgrid::Barrier& bar = hostgrid::warp_bar[hostgrid::gwarp()];
    const unsigned par = bar.gen & 1u;
    for (int i = 0; i < 4; ++i) g_mma_a[par][4 * me + i] = a[i];
    g_mma_b[par][2 * me] = b0;
    g_mma_b[par][2 * me + 1] = b1;
    hostgrid::wait(bar);
    const size_t w0 = me & ~(size_t)31;
    const unsigned lane = (unsigned)(me & 31u), gid = lane >> 2, tig = lane & 3u;
    const auto lo = [](u32 v) { return h2f((unsigned short)(v & 0xffffu)); };
    const auto hi = [](u32 v) { return h2f((unsigned short)(v >> 16)); };
    float Ar[2][16], Bc[2][16];
    for (unsigned L = 0; L < 32; ++L) {
        const unsigned g = L >> 2, q = L & 3u;
        const u32* aa = &g_mma_a[par][4 * (w0 + L)];
        if (g == gid) {
            Ar[0][2 * q] = lo(aa[0]);
            Ar[0][2 * q + 1] = hi(aa[0]);
            Ar[1][2 * q] = lo(aa[1]);
            Ar[1][2 * q + 1] = hi(aa[1]);
            Ar[0][2 * q + 8] = lo(aa[2]);
            Ar[0][2 * q + 9] = hi(aa[2]);
            Ar[1][2 * q + 8] = lo(aa[3]);
            Ar[1][2 * q + 9] = hi(aa[3]);
        }
        if (g == 2 * tig || g == 2 * tig + 1) {
            const u32* bb = &g_mma_b[par][2 * (w0 + L)];
            float* B = Bc[g - 2 * tig];
            B[2 * q] = lo(bb[0]);
            B[2 * q + 1] = hi(bb[0]);
            B[2 * q + 8] = lo(bb[1]);
            B[2 * q + 9] = hi(bb[1]);
        }
    }
    for (unsigned e = 0; e < 4; ++e) {
        const float* ar = Ar[e >> 1];
        const float* bc = Bc[e & 1u];
        float s = d[e];
        for (unsigned k = 0; k < 16; ++k) s = s + ar[k] * bc[k];
        d[e] = s;
    }
}

#endif  // LLVQ_HOST_MMA_H
