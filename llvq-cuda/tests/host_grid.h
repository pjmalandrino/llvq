// A whole grid executed on the host: every CTA at once, one fiber per thread,
// 32-lane warps, real barriers, real shuffles, real spin-waits.
//
// `host_cta.h` runs one CTA at a time with 16-lane groups, which is right for
// a kernel whose CTAs never wait for each other. A persistent kernel's CTAs
// do: a grid barrier needs every CTA resident, so here every thread of every
// CTA is a fiber of ONE scheduler. A fiber runs until it reaches a barrier, a
// shuffle or a spin (`__nanosleep`), then yields. The scheduler resumes the
// runnable fibers in an order a seed permutes, and with a seed it also leaves
// runnable fibers out of a pass at random, so one warp can run far ahead of
// another: that is how a missing barrier shows up as a skipped row here
// rather than once in a thousand launches on a card.
//
// Shared memory: the persistent kernels declare no static `__shared__`; they
// reach their dynamic buffer through `hostgrid_smem()`, which returns the
// current fiber's CTA buffer. The served kernels this harness also runs (one
// CTA at a time, `launch_seq`) name `extern __shared__ float xs[]`; the driver
// defines those arrays as globals, which is exact when one CTA runs at a time.
//
// What it checks: every barrier is reached by every thread of its CTA, every
// shuffle by every lane of its warp, and a spin that never ends trips the
// kernel's own watchdog (PERS_SPIN_LIMIT, lowered for the host). What it does
// not model: memory ordering (fibers never run at the same time), timing, PTX.
// Include it INSTEAD of host_shim.h, never beside it.

#ifndef LLVQ_HOST_GRID_H
#define LLVQ_HOST_GRID_H

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

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#define LLVQ_HOST_BUILD 1
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#ifndef __restrict__
#define __restrict__
#endif
#define __shared__

struct Dim3 {
    unsigned x, y, z;
};
struct uint4 {
    unsigned x, y, z, w;
};
struct alignas(16) float4 {
    float x, y, z, w;
};

static Dim3 threadIdx;
static Dim3 blockIdx;
static Dim3 gridDim;
static Dim3 blockDim;

// ---- arithmetic intrinsics: semantics, not stand-ins ----------------------
// `noinline` on the rounded single operations: the served side of the host
// test is compiled with -ffp-contract=fast to model NVVM's contraction of the
// served source, and an explicit `__fmul_rn` or `__fadd_rn` must never be
// fused into an fma there, exactly as `mul.rn.f32` and `add.rn.f32` are never
// fused on the card.
__attribute__((noinline, unused)) static float __fmul_rn(float a, float b) { return a * b; }
__attribute__((noinline, unused)) static float __fadd_rn(float a, float b) { return a + b; }
__attribute__((noinline, unused)) static float __fsub_rn(float a, float b) { return a - b; }
__attribute__((noinline, unused)) static float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
// Not the card's rsqrt.approx.f32. Both sides of every host comparison call
// this same function on the same value, so the stand-in cancels; the card's
// load-time check compares the real instruction with candle's.
__attribute__((noinline, unused)) static float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
static inline int __popc(unsigned x) { return __builtin_popcount(x); }
static inline int __clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
static inline int __ffs(int x) { return __builtin_ffs(x); }
static inline unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
static inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
static inline long long __double_as_longlong(double d) { long long i; std::memcpy(&i, &d, 8); return i; }
static inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
static inline unsigned __float_as_uint(float f) { unsigned i; std::memcpy(&i, &f, 4); return i; }
static inline float __uint_as_float(unsigned i) { float f; std::memcpy(&f, &i, 4); return f; }
// PTX prmt.b32 in its default mode (see host_shim.h for the full note).
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
    for (unsigned k = 0; k < 4; ++k) {
        int x = (signed char)((unsigned)a >> (8 * k));
        int y = (signed char)((unsigned)b >> (8 * k));
        c += x * y;
    }
    return c;
}
template <class T>
static inline T __ldcg(const T* p) { return *p; }
template <class T>
static inline T __ldg(const T* p) { return *p; }

// Atomics: fibers never preempt, so a plain read-modify-write is atomic.
static inline unsigned atomicAdd(unsigned* a, unsigned v) { unsigned o = *a; *a = o + v; return o; }
static inline unsigned atomicExch(unsigned* a, unsigned v) { unsigned o = *a; *a = v; return o; }
static inline unsigned long long atomicMax(unsigned long long* a, unsigned long long v) {
    unsigned long long o = *a;
    if (v > o) *a = v;
    return o;
}
static inline unsigned atomicOr(unsigned* a, unsigned v) { unsigned o = *a; *a = o | v; return o; }
static inline unsigned long long atomicExch(unsigned long long* a, unsigned long long v) {
    unsigned long long o = *a;
    *a = v;
    return o;
}
static inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }

namespace hostgrid {

constexpr size_t kStack = 64 * 1024;

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
    unsigned cta = 0;   // linear CTA index
    unsigned tid = 0;   // thread index in its CTA
};

static std::vector<Fiber> fibers;
static int current = -1;
static const std::function<void()>* body_fn = nullptr;
static std::vector<Barrier> cta_bar;
static std::vector<Barrier> warp_bar;
static std::vector<std::vector<unsigned char>> smem;
// Shuffle slots, double-buffered by the parity of the warp barrier's
// generation (a fast lane can write its next value before a slow one has
// read the last; it cannot get two generations ahead).
static std::vector<unsigned long long> slot[2];
static uint64_t rng = 0;
static unsigned threads_per_cta = 0;
static unsigned long long switches = 0;

static inline uint64_t next_rand() {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 0x2545f4914f6cdd1dull;
}

#if defined(__aarch64__)
extern "C" void hg_switch(void** save, void* load) asm("llvq_hostgrid_switch");
asm(R"(
    .text
    .p2align 2
llvq_hostgrid_switch:
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
static inline void to_sched(Fiber& f) { hg_switch(&f.sp, sched_sp); }
static inline void to_fiber(Fiber& f) { hg_switch(&sched_sp, f.sp); }
#else
static ucontext_t sched_ctx;
static inline void to_sched(Fiber& f) { swapcontext(&f.ctx, &sched_ctx); }
static inline void to_fiber(Fiber& f) { swapcontext(&sched_ctx, &f.ctx); }
#endif

static inline void yield() {
    Fiber& f = fibers[current];
    f.waiting_on = nullptr;
    to_sched(f);
}

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

static inline void trampoline() {
    (*body_fn)();
    fibers[current].done = true;
    to_sched(fibers[current]);
    std::abort();
}

static inline unsigned lane() { return threadIdx.x & 31u; }
static inline unsigned gwarp() { return fibers[current].cta * (threads_per_cta / 32u) + (threadIdx.x >> 5); }

template <class T>
static inline T shfl(unsigned mask, T v, unsigned src_lane, const char* what) {
    if (mask != 0xffffffffu) {
        std::fprintf(stderr, "host_grid: %s with mask %08x, only full warps are modelled\n", what, mask);
        std::abort();
    }
    Barrier& b = warp_bar[gwarp()];
    const unsigned par = b.gen & 1u;
    const unsigned me = (unsigned)current;
    unsigned long long bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    slot[par][me] = bits;
    wait(b);
    const unsigned src = (me & ~31u) | (src_lane & 31u);
    T out;
    bits = slot[par][src];
    std::memcpy(&out, &bits, sizeof(T));
    return out;
}

static inline void make_fiber(Fiber& f) {
    if (f.stack.empty()) f.stack.resize(kStack);
    f.done = false;
    f.waiting_on = nullptr;
#if defined(__aarch64__)
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

// Run fibers [first, first + count) to completion.
static inline void run(unsigned first, unsigned count, uint64_t seed) {
    unsigned left = count;
    std::vector<unsigned> order(count);
    for (unsigned i = 0; i < count; ++i) order[i] = first + i;
    while (left > 0) {
        if (seed != 0) {
            for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[next_rand() % i]);
        }
        bool progressed = false;
        for (unsigned idx : order) {
            Fiber& f = fibers[idx];
            if (f.done) continue;
            if (f.waiting_on && f.waiting_on->gen == f.waiting_gen) continue;
            // With a seed, leave a runnable fiber out of this pass one time in
            // three: warps drift apart the way they do on a card.
            if (seed != 0 && progressed && next_rand() % 3 == 0) continue;
            f.waiting_on = nullptr;
            current = (int)idx;
            threadIdx = {f.tid, 0, 0};
            blockIdx = {f.cta % gridDim.x, f.cta / gridDim.x, 0};
            to_fiber(f);
            ++switches;
            progressed = true;
            if (f.done) --left;
        }
        if (!progressed) {
            std::fprintf(stderr, "host_grid: no thread can run, a barrier or a shuffle was not reached by every "
                                 "thread it names\n");
            std::abort();
        }
    }
}

static inline void setup(unsigned ctas, unsigned threads, size_t smem_bytes) {
    if (threads == 0 || threads % 32 != 0 || threads > 1024) {
        std::fprintf(stderr, "host_grid: %u threads a CTA is not supported\n", threads);
        std::abort();
    }
    threads_per_cta = threads;
    blockDim = {threads, 1, 1};
    const unsigned n = ctas * threads;
    if (fibers.size() < n) fibers.resize(n);
    for (unsigned c = 0; c < ctas; ++c)
        for (unsigned t = 0; t < threads; ++t) {
            Fiber& f = fibers[c * threads + t];
            f.cta = c;
            f.tid = t;
        }
    cta_bar.assign(ctas, Barrier{(int)threads, 0, 0});
    warp_bar.assign(ctas * threads / 32, Barrier{32, 0, 0});
    for (auto& s : slot) s.assign(n, 0);
    smem.assign(ctas, std::vector<unsigned char>(smem_bytes + 16, 0xA5));
}

// A cooperative launch: every CTA resident at once.
template <class F>
void launch_coop(unsigned grid, unsigned threads, size_t smem_bytes, uint64_t seed, F body) {
    gridDim = {grid, 1, 1};
    setup(grid, threads, smem_bytes);
    rng = seed ? seed : 1;
    const std::function<void()> fn = body;
    body_fn = &fn;
    for (unsigned i = 0; i < grid * threads; ++i) make_fiber(fibers[i]);
    run(0, grid * threads, seed);
    body_fn = nullptr;
}

// A normal launch: CTAs one after the other, as host_cta.h runs them.
template <class F>
void launch_seq(unsigned grid, unsigned threads, F body) {
    gridDim = {grid, 1, 1};
    setup(1, threads, 0);
    const std::function<void()> fn = body;
    body_fn = &fn;
    for (unsigned c = 0; c < grid; ++c) {
        for (unsigned t = 0; t < threads; ++t) {
            fibers[t].cta = 0;
            make_fiber(fibers[t]);
        }
        cta_bar[0] = Barrier{(int)threads, 0, 0};
        for (auto& w : warp_bar) w = Barrier{32, 0, 0};
        // blockIdx is set from the fiber's cta, so remap: one CTA slot, many ids.
        const unsigned saved = c;
        for (unsigned t = 0; t < threads; ++t) fibers[t].cta = 0;
        gridDim = {grid, 1, 1};
        // run with blockIdx forced to c
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
                blockIdx = {saved, 0, 0};
                to_fiber(f);
                progressed = true;
                if (f.done) --left;
            }
            if (!progressed) {
                std::fprintf(stderr, "host_grid: CTA %u: no thread can run\n", c);
                std::abort();
            }
        }
    }
    body_fn = nullptr;
}

}  // namespace hostgrid

unsigned char* hostgrid_smem() {
    unsigned char* p = hostgrid::smem[hostgrid::fibers[hostgrid::current].cta].data();
    return (unsigned char*)(((uintptr_t)p + 15) & ~(uintptr_t)15);
}

static inline void __syncthreads() { hostgrid::wait(hostgrid::cta_bar[hostgrid::fibers[hostgrid::current].cta]); }
static inline void __syncwarp(unsigned mask = 0xffffffffu) {
    if (mask != 0xffffffffu) std::abort();
    hostgrid::wait(hostgrid::warp_bar[hostgrid::gwarp()]);
}
static inline void __nanosleep(unsigned) { hostgrid::yield(); }
static inline float __shfl_xor_sync(unsigned m, float v, int o, int w = 32) {
    if (w != 32 || o < 1 || o > 16) std::abort();
    return hostgrid::shfl(m, v, hostgrid::lane() ^ (unsigned)o, "__shfl_xor_sync");
}
static inline unsigned __shfl_xor_sync(unsigned m, unsigned v, int o, int w = 32) {
    if (w != 32 || o < 1 || o > 16) std::abort();
    return hostgrid::shfl(m, v, hostgrid::lane() ^ (unsigned)o, "__shfl_xor_sync");
}
static inline unsigned long long __shfl_xor_sync(unsigned m, unsigned long long v, int o, int w = 32) {
    if (w != 32 || o < 1 || o > 16) std::abort();
    return hostgrid::shfl(m, v, hostgrid::lane() ^ (unsigned)o, "__shfl_xor_sync");
}
static inline unsigned __shfl_sync(unsigned m, unsigned v, int src, int w = 32) {
    if (w != 32) std::abort();
    return hostgrid::shfl(m, v, (unsigned)src, "__shfl_sync");
}
static inline float __shfl_sync(unsigned m, float v, int src, int w = 32) {
    if (w != 32) std::abort();
    return hostgrid::shfl(m, v, (unsigned)src, "__shfl_sync");
}
static inline int __any_sync(unsigned m, int p) {
    unsigned any = (unsigned)(p != 0);
    for (int o = 1; o < 32; o <<= 1) any |= __shfl_xor_sync(m, any, o);
    return (int)any;
}

#endif  // LLVQ_HOST_GRID_H
