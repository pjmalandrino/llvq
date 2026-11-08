// Executes attn_dec_batch.cu's entry on the CPU through host_cta.h and
// requires, sequence by sequence, the bits of the served one-row entry
// (attn_dec_g4_u64 / _g5_u64) called alone on that sequence: same query, same
// fresh key and value written by the kernel, same cache, its own grid.
//
// The batch mixes lengths on purpose: a first position (one split, the
// `single` path at its own grid), a row inside the second split and one in the
// third, so the batched launch runs a grid wider than two of its sequences and
// folds their partials through the ticket while the one-row calls do not.
//
//   in : u32 G, u32 n_kv, u32 cap, u32 B, i32[B] pos, u64 seed
//   out: "MATCH <values>" on stdout and exit 0, or the first difference and exit 1

#include "host_cta.h"
#include "../kernels/attn_dec.cu"
#include "../kernels/attn_dec_batch.cu"

#include <vector>

// host_cta's launcher for a third grid axis: every (x, y, z) CTA in one
// permuted order, so CTAs of different sequences interleave as they may on a
// card, and a ticket or a partial shared across sequences shows.
template <class F>
static void launch3(unsigned gx, unsigned gy, unsigned gz, uint64_t seed, F body) {
    gridDim = {gx, gy, gz};
    blockDim = {ATTN_THREADS, 1, 1};
    std::vector<Dim3> order;
    for (unsigned z = 0; z < gz; ++z)
        for (unsigned y = 0; y < gy; ++y)
            for (unsigned x = 0; x < gx; ++x) order.push_back({x, y, z});
    uint64_t s = seed;
    for (size_t i = order.size(); i > 1; --i) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        std::swap(order[i - 1], order[(size_t)((s * 0x2545f4914f6cdd1dull) % i)]);
    }
    const std::function<void()> fn = body;
    hostcta::body_fn = &fn;
    for (const auto& b : order) {
        blockIdx = b;
        hostcta::run_cta(ATTN_THREADS);
    }
    hostcta::body_fn = nullptr;
}

template <class T>
static T rd() {
    T v;
    if (std::fread(&v, sizeof v, 1, stdin) != 1) {
        std::fprintf(stderr, "host_attn_batch: short read\n");
        std::exit(2);
    }
    return v;
}

int main() {
    const unsigned G = rd<unsigned>(), n_kv = rd<unsigned>(), cap = rd<unsigned>(), B = rd<unsigned>();
    std::vector<int> pos(B);
    for (auto& p : pos) p = rd<int>();
    uint64_t s = rd<uint64_t>();
    auto next = [&]() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        return attn_f2h((float)((double)(z >> 11) / 9007199254740992.0 * 2.0 - 1.0));
    };
    const size_t nq = (size_t)n_kv * G * ATTN_HD, nk = (size_t)n_kv * ATTN_HD, nc = (size_t)n_kv * cap * ATTN_HD;
    std::vector<unsigned short> q(B * nq), kn(B * nk), vn(B * nk), kc(B * nc), vc(B * nc);
    for (auto* v : {&q, &kn, &vn, &kc, &vc})
        for (auto& x : *v) x = next();
    const float c = 0.08838834764831845f * 1.4426950408889634f;

    int maxl = 0;
    for (int p : pos) maxl = p + 1 > maxl ? p + 1 : maxl;
    const unsigned gx = (unsigned)((maxl + ATTN_SPLIT - 1) / ATTN_SPLIT);
    const int ss = (int)gx;

    // The batched launch, its CTAs interleaved across sequences.
    std::vector<unsigned short> bkc = kc, bvc = vc, bout(B * nq, 0xffffu);
    std::vector<float> bpart((size_t)B * n_kv * ss * ATTN_KMAX * G * ATTN_PSTRIDE, 0.0f);
    std::vector<unsigned> bt((size_t)B * n_kv, 0u);
    std::vector<int> posv = pos;
    {
        launch3(gx, n_kv, B, 0x5eedu, [&] {
            if (G == 4) attn_batch_g4_u64(q.data(), kn.data(), vn.data(), bkc.data(), bvc.data(), posv.data(), (int)cap, c, bpart.data(), bt.data(), bout.data(), ss);
            else attn_batch_g5_u64(q.data(), kn.data(), vn.data(), bkc.data(), bvc.data(), posv.data(), (int)cap, c, bpart.data(), bt.data(), bout.data(), ss);
        });
    }
    for (unsigned t : bt)
        if (t != 0) { std::printf("a ticket did not reset\n"); return 1; }

    size_t n = 0;
    for (unsigned z = 0; z < B; ++z) {
        std::vector<unsigned short> skc(kc.begin() + z * nc, kc.begin() + (z + 1) * nc);
        std::vector<unsigned short> svc(vc.begin() + z * nc, vc.begin() + (z + 1) * nc);
        std::vector<unsigned short> sout(nq, 0xffffu);
        const unsigned sg = (unsigned)((pos[z] + 1 + ATTN_SPLIT - 1) / ATTN_SPLIT);
        std::vector<float> spart((size_t)n_kv * sg * ATTN_KMAX * G * ATTN_PSTRIDE, 0.0f);
        std::vector<unsigned> st(n_kv, 0u);
        hostcta::launch(sg, n_kv, ATTN_THREADS, 0xabcu + z, [&] {
            blockIdx.z = 0;
            if (G == 4) attn_dec_g4_u64(q.data() + z * nq, kn.data() + z * nk, vn.data() + z * nk, skc.data(), svc.data(), pos[z], 1, (int)cap, c, spart.data(), st.data(), sout.data(), (int)sg);
            else attn_dec_g5_u64(q.data() + z * nq, kn.data() + z * nk, vn.data() + z * nk, skc.data(), svc.data(), pos[z], 1, (int)cap, c, spart.data(), st.data(), sout.data(), (int)sg);
        });
        for (size_t i = 0; i < nq; ++i, ++n)
            if (sout[i] != bout[z * nq + i]) {
                std::printf("sequence %u value %zu: one-row %04x, batched %04x\n", z, i, sout[i], bout[z * nq + i]);
                return 1;
            }
        for (size_t i = 0; i < nc; ++i)
            if (skc[i] != bkc[z * nc + i] || svc[i] != bvc[z * nc + i]) {
                std::printf("sequence %u cache value %zu differs\n", z, i);
                return 1;
            }
    }
    std::printf("MATCH %zu\n", n);
    return 0;
}
