// Drives attn_dec.cu on the CPU, the whole kernel, through host_cta.h.
//
// A harness, not a reference: it holds no expectation. The Rust side
// (tests/attn_dec_matches_rust.rs) builds every fixture, pipes it on stdin,
// and compares every output against src/attn_ref.rs bit for bit.
//
// Commands on stdin, little-endian, repeated until end of input:
//   u32 1, u32 n, f32[n]                    -> f32[n]   attn_ex2, executed
//   u32 2, u32 G, u32 unit, u32 n_kv, u32 cap, u32 pos0, u32 k, u32 grid_x,
//          u32 s_stride, f32 c, u32 with_new, u64 order_seed,
//          u16 q[n_kv*G][k][128], u16 kc[n_kv][cap][128], u16 vc[...],
//          and when with_new: u16 k_new[n_kv][k][128], u16 v_new[...]
//                                           -> u16 out[n_kv*G][k][128],
//                                              u16 kc, u16 vc after the call,
//                                              u32 tickets[n_kv] after the call
//
// Built by clang++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror
// -pthread, like every host probe here, plus -pthread.

#include "host_cta.h"
#include "../kernels/attn_dec.cu"

template <class T>
static T rd() {
    T v;
    if (std::fread(&v, sizeof v, 1, stdin) != 1) {
        std::fprintf(stderr, "host_attn_dec: short read\n");
        std::exit(2);
    }
    return v;
}

template <class T>
static void rdv(std::vector<T>& v, size_t n) {
    v.resize(n);
    if (n && std::fread(v.data(), sizeof(T), n, stdin) != n) {
        std::fprintf(stderr, "host_attn_dec: short read of %zu values\n", n);
        std::exit(2);
    }
}

template <class T>
static void wrv(const std::vector<T>& v) {
    std::fwrite(v.data(), sizeof(T), v.size(), stdout);
}

using Entry = void (*)(const unsigned short*, const unsigned short*, const unsigned short*,
                       unsigned short*, unsigned short*, int, int, int, float, float*, unsigned*,
                       unsigned short*, int);

static Entry entry(unsigned g, unsigned unit) {
    if (g == 4 && unit == 64) return attn_dec_g4_u64;
    if (g == 5 && unit == 64) return attn_dec_g5_u64;
    if (g == 4 && unit == 32) return attn_dec_g4_u32;
    if (g == 5 && unit == 32) return attn_dec_g5_u32;
    std::fprintf(stderr, "host_attn_dec: no entry point for G=%u unit=%u\n", g, unit);
    std::exit(3);
}

int main() {
    unsigned mode;
    while (std::fread(&mode, 4, 1, stdin) == 1) {
        if (mode == 1) {
            const unsigned n = rd<unsigned>();
            std::vector<float> x;
            rdv(x, n);
            for (float& v : x) v = attn_ex2(v);
            wrv(x);
        } else if (mode == 2) {
            const unsigned G = rd<unsigned>(), unit = rd<unsigned>(), n_kv = rd<unsigned>();
            const unsigned cap = rd<unsigned>(), pos0 = rd<unsigned>(), k = rd<unsigned>();
            const unsigned grid_x = rd<unsigned>(), s_stride = rd<unsigned>();
            const float c = rd<float>();
            const unsigned with_new = rd<unsigned>();
            const uint64_t seed = rd<uint64_t>();
            const size_t nq = (size_t)n_kv * G * k * ATTN_HD;
            const size_t ncache = (size_t)n_kv * cap * ATTN_HD;
            const size_t nnew = (size_t)n_kv * k * ATTN_HD;
            std::vector<unsigned short> q, kc, vc, kn, vn;
            rdv(q, nq);
            rdv(kc, ncache);
            rdv(vc, ncache);
            if (with_new) {
                rdv(kn, nnew);
                rdv(vn, nnew);
            }
            std::vector<float> part((size_t)n_kv * s_stride * ATTN_KMAX * G * ATTN_PSTRIDE, 0.0f);
            std::vector<unsigned> ticket(n_kv, 0u);
            std::vector<unsigned short> out(nq, 0xffffu);  // a NaN pattern: unwritten shows
            const Entry f = entry(G, unit);
            const unsigned short* knp = with_new ? kn.data() : nullptr;
            const unsigned short* vnp = with_new ? vn.data() : nullptr;
            hostcta::launch(grid_x, n_kv, ATTN_THREADS, seed, [&] {
                f(q.data(), knp, vnp, kc.data(), vc.data(), (int)pos0, (int)k, (int)cap, c,
                  part.data(), ticket.data(), out.data(), (int)s_stride);
            });
            wrv(out);
            wrv(kc);
            wrv(vc);
            wrv(ticket);
        } else {
            std::fprintf(stderr, "host_attn_dec: unknown command %u\n", mode);
            return 4;
        }
        std::fflush(stdout);
    }
    return 0;
}
