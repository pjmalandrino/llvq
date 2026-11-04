// A CTA executed on the host: one fiber per thread, real barriers, 16-lane
// shuffles, real tickets.
//
// `host_shim.h` turns the CUDA keywords into C++ for ONE thread, which is why
// the probes that include it execute the shuffle-free parts of a kernel and
// only compile the rest. This header goes further for kernels whose point is
// the cooperation: each launched CTA runs as `blockDim.x` cooperative fibers
// (ucontext) on one OS thread. A fiber runs until it reaches `__syncthreads`
// or a shuffle, then yields; the scheduler resumes it once every fiber of its
// CTA (barrier) or of its 16-lane group (shuffle) has arrived. `__shared__`
// becomes `static`, which is exact because one CTA runs at a time, and the
// launcher runs the CTAs in an order the caller permutes, so the "last CTA" of
// a ticket combine is not always the last in grid order.
//
// Fibers rather than OS threads: 128 threads a CTA outnumber the cores, and
// waking them through the kernel cost ten times the work (measured on the dev
// Mac, 2026-11-04: 38 s of system time for 8 s of wall on one 4-CTA launch,
// with a condition variable; worse with spin-and-yield). On arm64 the switch
// is twenty instructions of assembly below; `swapcontext` costs a
// `sigprocmask` system call a switch (2.9 s on the same launch) and stays as
// the fallback on other architectures.
//
// What it checks that a card would not tell you cheaply: every barrier is
// reached by every thread of the CTA, and every shuffle by every lane of the
// group (otherwise the scheduler finds no runnable fiber and aborts naming
// it, where a card would hang); every shuffle names exactly the 16 lanes of
// its group, with width 16; the kernel's indexing, staging and combine compute
// the bits the Rust replay computes.
//
// What it does not model: memory ordering between CTAs that run at the same
// time (they never do here), a real warp's timing, cp.async or any PTX.
// Include it INSTEAD of host_shim.h, never beside it.

#ifndef LLVQ_HOST_CTA_H
#define LLVQ_HOST_CTA_H

#define _XOPEN_SOURCE 600
#include <math.h>
#include <ucontext.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

// ucontext is deprecated on macOS and still shipped; this harness is the one
// place that wants it.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#define LLVQ_HOST_BUILD 1
#define __device__
#define __global__
#define __forceinline__ inline
#ifndef __restrict__
#define __restrict__
#endif
#define __shared__ static

struct Dim3 {
    unsigned x, y, z;
};
struct uint4 {
    unsigned x, y, z, w;
};

// Plain globals: one fiber runs at a time, and the scheduler sets them on
// every switch.
static Dim3 threadIdx;
static Dim3 blockIdx;
static Dim3 gridDim;
static Dim3 blockDim;

static inline int __float_as_int(float f) {
    int i;
    std::memcpy(&i, &f, 4);
    return i;
}
static inline float __int_as_float(int i) {
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

namespace hostcta {

constexpr int kMaxThreads = 128;
constexpr int kGroup = 16;
constexpr size_t kStack = 128 * 1024;

// A barrier is a generation counter. A fiber that arrives last advances it
// and runs on; the others yield until it has moved.
struct Barrier {
    int n = 0;
    int arrived = 0;
    unsigned gen = 0;
};

struct Fiber {
#if defined(__aarch64__)
    void* sp = nullptr;
#else
    ucontext_t ctx;
#endif
    std::vector<char> stack;
    bool done = false;
    Barrier* waiting_on = nullptr;
    unsigned waiting_gen = 0;
};

static Fiber fibers[kMaxThreads];
static int current = -1;
static const std::function<void()>* body_fn = nullptr;
static Barrier block_bar;
static Barrier group_bar[kMaxThreads / kGroup];
// Double-buffered by generation parity: a fiber that runs ahead can write its
// next value before a slower one has read the last, and the parity keeps them
// apart (it cannot get two generations ahead without everyone arriving).
static float shfl_slot[2][kMaxThreads];
static int or_slot[2][kMaxThreads];

#if defined(__aarch64__)
// hc_switch(save, load): push the callee-saved registers (AAPCS64: x19-x30,
// d8-d15; x18 is the platform's and is left alone), store sp to *save, load
// sp from load, pop, return into whatever that stack was switched out from.
extern "C" void hc_switch(void** save, void* load) asm("llvq_hostcta_switch");
asm(R"(
    .text
    .p2align 2
llvq_hostcta_switch:
    sub sp, sp, #160
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mov x9, sp
    str x9, [x0]
    mov sp, x1
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    ldp x29, x30, [sp, #80]
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    add sp, sp, #160
    ret
)");
static void* sched_sp = nullptr;
static inline void to_sched(Fiber& f) { hc_switch(&f.sp, sched_sp); }
static inline void to_fiber(Fiber& f) { hc_switch(&sched_sp, f.sp); }
#else
static ucontext_t sched_ctx;
static inline void to_sched(Fiber& f) { swapcontext(&f.ctx, &sched_ctx); }
static inline void to_fiber(Fiber& f) { swapcontext(&sched_ctx, &f.ctx); }
#endif

static inline unsigned wait(Barrier& b) {
    const unsigned g = b.gen;
    if (++b.arrived == b.n) {
        b.arrived = 0;
        ++b.gen;
        return g;
    }
    Fiber& f = fibers[current];
    f.waiting_on = &b;
    f.waiting_gen = g;
    to_sched(f);
    return g;
}

static void trampoline() {
    (*body_fn)();
    fibers[current].done = true;
    to_sched(fibers[current]);
    std::abort();  // a finished fiber is never resumed
}

}  // namespace hostcta

static inline void __syncthreads() { hostcta::wait(hostcta::block_bar); }

static inline int __syncthreads_or(int p) {
    const unsigned t = threadIdx.x;
    const unsigned par = hostcta::block_bar.gen & 1u;
    hostcta::or_slot[par][t] = p != 0;
    hostcta::wait(hostcta::block_bar);
    int any = 0;
    for (unsigned i = 0; i < blockDim.x; ++i) any |= hostcta::or_slot[par][i];
    return any;
}

static inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }

static inline unsigned atomicAdd(unsigned* a, unsigned v) {
    const unsigned o = *a;
    *a = o + v;
    return o;
}

// Width 16 only, xor offsets 1..8, and the mask must be the caller's own 16
// lanes: anything else is a kernel defect this header refuses by name.
static inline float __shfl_xor_sync(unsigned mask, float v, int off, int width) {
    const unsigned t = threadIdx.x;
    const unsigned want = 0xffffu << (t & 16u);
    if (width != 16 || mask != want || off < 1 || off > 8) {
        std::fprintf(stderr,
                     "host_cta: thread %u shuffles with mask %08x width %d offset %d, "
                     "expected mask %08x width 16 offset 1..8\n",
                     t, mask, width, off, want);
        std::abort();
    }
    hostcta::Barrier& b = hostcta::group_bar[t / hostcta::kGroup];
    const unsigned par = b.gen & 1u;
    hostcta::shfl_slot[par][t] = v;
    hostcta::wait(b);
    return hostcta::shfl_slot[par][(t & ~15u) | ((t & 15u) ^ (unsigned)off)];
}

namespace hostcta {

// Run one CTA to completion: resume runnable fibers round-robin until all are
// done; a pass that resumes none while some are not done is a divergent
// barrier, named and aborted.
static void run_cta(unsigned threads) {
    for (unsigned t = 0; t < threads; ++t) {
        Fiber& f = fibers[t];
        if (f.stack.empty()) f.stack.resize(kStack);
        f.done = false;
        f.waiting_on = nullptr;
#if defined(__aarch64__)
        // A frame hc_switch can pop: zeros, and x30 (lr) = the trampoline,
        // so the first switch "returns" into it with a 16-aligned sp.
        uintptr_t top = ((uintptr_t)(f.stack.data() + f.stack.size())) & ~(uintptr_t)15;
        void** frame = (void**)(top - 160);
        std::memset(frame, 0, 160);
        frame[11] = (void*)&trampoline;  // the x30 slot, [sp, #88]
        f.sp = frame;
#else
        getcontext(&f.ctx);
        f.ctx.uc_stack.ss_sp = f.stack.data();
        f.ctx.uc_stack.ss_size = f.stack.size();
        f.ctx.uc_link = nullptr;
        makecontext(&f.ctx, trampoline, 0);
#endif
    }
    block_bar = Barrier{(int)threads, 0, 0};
    for (unsigned g = 0; g < threads / kGroup; ++g) group_bar[g] = Barrier{kGroup, 0, 0};
    unsigned left = threads;
    while (left > 0) {
        bool progressed = false;
        for (unsigned t = 0; t < threads; ++t) {
            Fiber& f = fibers[t];
            if (f.done) continue;
            if (f.waiting_on && f.waiting_on->gen == f.waiting_gen) continue;
            f.waiting_on = nullptr;
            current = (int)t;
            threadIdx = {t, 0, 0};
            to_fiber(f);
            progressed = true;
            if (f.done) --left;
        }
        if (!progressed) {
            std::fprintf(stderr,
                         "host_cta: CTA (%u, %u): no thread can run, a barrier or a shuffle "
                         "was not reached by every thread it names\n",
                         blockIdx.x, blockIdx.y);
            std::abort();
        }
    }
}

// Launch `body` over a (gx, gy) grid of `threads`-thread CTAs, one CTA at a
// time, in an order permuted by `seed` (0 keeps grid order).
template <class F>
void launch(unsigned gx, unsigned gy, unsigned threads, uint64_t seed, F body) {
    if (threads > (unsigned)kMaxThreads || threads % kGroup != 0) {
        std::fprintf(stderr, "host_cta: %u threads a CTA is not supported\n", threads);
        std::abort();
    }
    gridDim = {gx, gy, 1};
    blockDim = {threads, 1, 1};
    std::vector<std::pair<unsigned, unsigned>> order;
    for (unsigned y = 0; y < gy; ++y)
        for (unsigned x = 0; x < gx; ++x) order.push_back({x, y});
    uint64_t s = seed;
    if (s != 0) {
        for (size_t i = order.size(); i > 1; --i) {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            const size_t j = (size_t)((s * 0x2545f4914f6cdd1dull) % i);
            std::swap(order[i - 1], order[j]);
        }
    }
    const std::function<void()> fn = body;
    body_fn = &fn;
    for (const auto& b : order) {
        blockIdx = {b.first, b.second, 0};
        run_cta(threads);
    }
    body_fn = nullptr;
}

}  // namespace hostcta

#endif  // LLVQ_HOST_CTA_H
