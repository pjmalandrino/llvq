// DRAFT, never compiled or run. Revised 2026-11-02 after review: the layout model below is the
// emulator's one unverified assumption, so beta0's `preflight` fragment probe (prereg-brick-0.md
// gate 1) checks it on the card, lane by lane, before this file becomes the only host-side oracle
// for the fragment code. The self-cleaning cross-CTA combine is outside one warp: its mutant (the
// last CTA skips the clean) is caught by running the CTA loop twice over one scratch.
//
// Host warp emulator for the brick's fragment code (wave5-deep/brick.md §6 step 1). Today's
// host_shim.h makes barriers and shuffles vacuous and has no ldmatrix or mma, so no warp-level
// kernel runs on the Mac (fused_cuda.rs:2216-2222 says so). Here the 32 lanes of one warp run as
// 32 std::threads over the SAME kernel source; every collective is a rendezvous: each lane posts
// its operands, a barrier, each lane reads its result, a barrier. Slow (two barriers per
// collective) and fine for tests at small shapes. clang++ -std=c++20 -O1 -ffp-contract=off -pthread.
//
// The layouts are CUTLASS's SM80_16x8x32 maps (mma_traits_sm80.hpp): A row g + 8*v1, k 4t + v0 +
// 16*v2; B token g, k 4t + v0 + 16*v1; C (g + 8*(e>>1), 2t + (e&1)). ldmatrix.x4: matrix i takes
// its 8 row addresses from lanes 8i..8i+7, and lane (g, t) receives bytes 4t..4t+3 of row g.

#pragma once
#include <barrier>
#include <cstdint>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace wemu {

struct Warp {
    std::barrier<> bar{32};
    uint32_t post[32][8];                 // operands posted by each lane
    const unsigned char* smem = nullptr;  // emulated shared memory base

    // Run `body(lane)` on 32 threads; body calls the collectives below with its lane id.
    void run(const std::function<void(unsigned)>& body) {
        std::vector<std::thread> ts;
        for (unsigned l = 0; l < 32; ++l) ts.emplace_back(body, l);
        for (auto& t : ts) t.join();
    }

    uint32_t shfl(unsigned lane, uint32_t v, unsigned src) {
        post[lane][0] = v; bar.arrive_and_wait();
        uint32_t r = post[src & 31][0]; bar.arrive_and_wait();
        return r;
    }

    // ldmatrix.sync.aligned.m8n8.x4.shared.b16: `addr` is this lane's row address (bytes).
    void ldmatrix_x4(unsigned lane, uint32_t addr, uint32_t out[4]) {
        post[lane][0] = addr; bar.arrive_and_wait();
        unsigned g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; ++i) {
            uint32_t row_addr = post[8 * i + g][0];
            std::memcpy(&out[i], smem + row_addr + 4 * t, 4);
        }
        bar.arrive_and_wait();
    }

    // mma.sync.aligned.m16n8k32.row.col.s32.{s8,u8}: B_UNSIGNED selects the lo-plane form.
    template <bool B_UNSIGNED>
    void mma_m16n8k32(unsigned lane, const uint32_t a[4], uint32_t b0, uint32_t b1,
                      const int c[4], int d[4]) {
        for (int i = 0; i < 4; ++i) post[lane][i] = a[i];
        post[lane][4] = b0; post[lane][5] = b1;
        bar.arrive_and_wait();
        auto A = [&](int m, int k) -> int {          // reassemble the 16x32 A tile
            int g = m & 7, v1 = m >> 3, v2 = k >> 4, t = (k & 15) >> 2, v0 = k & 3;
            uint32_t w = post[4 * g + t][v1 + 2 * v2];   // a0 (g,lo) a1 (g+8,lo) a2 (g,hi) a3 (g+8,hi)
            return int(int8_t(uint8_t(w >> (8 * v0))));
        };
        auto B = [&](int n, int k) -> int {          // reassemble the 32x8 B tile
            int v1 = k >> 4, t = (k & 15) >> 2, v0 = k & 3;
            uint32_t w = post[4 * n + t][4 + v1];
            uint8_t byte = uint8_t(w >> (8 * v0));
            return B_UNSIGNED ? int(byte) : int(int8_t(byte));
        };
        unsigned g = lane >> 2, t = lane & 3;
        for (int e = 0; e < 4; ++e) {
            int m = g + 8 * (e >> 1), n = 2 * t + (e & 1);
            long long s = c[e];
            for (int k = 0; k < 32; ++k) s += (long long)A(m, k) * B(n, k);
            d[e] = int(s);                            // s32 accumulate; overflow is a test failure
        }
        bar.arrive_and_wait();
    }
};

}  // namespace wemu
