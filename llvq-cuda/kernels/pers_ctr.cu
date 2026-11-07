// tv_ctr_*: grid barriers against event counters, K3b of plan-300's step 1
// (2026-11-07, proofs/preregistration-step1-micro-2026-11-07.md).
//
// Why: 4E crosses 218 grid barriers a token at 0.79 to 1.13 us each
// (pers-l1-2026-11-04), and its phase walls exceed their mean work by 369 us
// outside attention. Mirage MPK replaces barriers by counters that a consumer
// waits on only for what it reads. Before 4E moves, these kernels time one
// hop of each form alone, at 4E's geometry, cooperatively launched:
//   BAR       pers_grid_wait as 4E calls it: bar.sync, a fence, an atomicAdd,
//             thread 0 spinning with a __nanosleep back-off from 32 to 512 ns,
//             a fence, bar.sync. pers-l1's barrier, the replication anchor.
//   TIGHT     the same barrier, thread 0 polling with no __nanosleep at all.
//   REL       arrival by `red.release.gpu` (no separate fence), polling by
//             `ld.acquire.gpu` with no back-off, then bar.sync.
//   CTR8      REL's arrival and poll on a counter that only 8 CTAs signal: CTA c
//             signals the counter of its group c / 8 and waits on the next
//             group's, as a consumer that reads one head's producers would.
//   SPLIT     REL's arrival, then `work` cycles of independent work on every
//             thread, then the poll: what of the hop the work does not hide.
//   WORK      the same `work` cycles and two bar.sync a hop, no grid wait:
//             SPLIT's control.
// Every kernel takes the same arguments: ctr (one counter a 32-u32 line; CTR8
// uses one line a group), base (hops already made on these counters by
// earlier launches), n hops, work cycles, the abort flag.
//
// Composition: the row-loop module (pers.rs::trows_unit_text), after
// pers_trows.cu, whose clock it uses.

#ifndef CTR_SPIN_LIMIT
#define CTR_SPIN_LIMIT (PERS_SPIN_LIMIT << 6)  // polls with no back-off, then abort
#endif
// Counters sit one 128-byte line apart.
#define CTR_LINE 32u
// CTR8's group.
#define CTR_GROUP 8u

// Nothing between two polls on the card; on the host, the fiber yields.
__device__ __forceinline__ void ctr_pause()
{
#ifdef LLVQ_HOST_BUILD
    __nanosleep(0u);
#endif
}

__device__ __forceinline__ void ctr_red_release(u32* p)
{
#ifdef LLVQ_HOST_BUILD
    __threadfence();
    atomicAdd(p, 1u);
#else
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
#endif
}

// Thread 0 polls with no back-off; 0 when the abort flag is up.
__device__ __forceinline__ int ctr_spin_tight(const u32* ctr, u32 target, u32* abort_flag)
{
    u32 spins = 0u;
    while (!pers_reached(pers_ld_acquire(ctr), target)) {
        ctr_pause();
        if ((++spins & 1023u) == 0u) {
            if (pers_ld_acquire(abort_flag) != 0u) return 0;
            if (spins >= CTR_SPIN_LIMIT) {
                atomicExch(abort_flag, 1u);
                return 0;
            }
        }
    }
    return 1;
}

// Every thread spins `cycles` SM cycles (the host: `cycles` clock calls).
__device__ __forceinline__ void ctr_work(u32 cycles)
{
    const unsigned long long t0 = trows_clock();
    while (trows_clock() - t0 < (unsigned long long)cycles) {
    }
}

// Broadcast thread 0's verdict, as pers_grid_wait does.
__device__ __forceinline__ bool ctr_verdict(int ok)
{
    u32* hdr = pers_hdr();
    if (threadIdx.x == 0u) hdr[PERS_HDR_OK] = (u32)ok;
    __syncthreads();
    return hdr[PERS_HDR_OK] != 0u;
}

extern "C" __global__ void tv_ctr_bar(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    (void)work;
    for (u32 i = 0; i < n; ++i)
        if (!pers_grid_wait(ctr, (base + i + 1u) * gridDim.x, abort_flag)) return;
}

extern "C" __global__ void tv_ctr_tight(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    (void)work;
    for (u32 i = 0; i < n; ++i) {
        __syncthreads();
        int ok = 1;
        if (threadIdx.x == 0u) {
            __threadfence();
            atomicAdd(ctr, 1u);
            ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);
            __threadfence();
        }
        if (!ctr_verdict(ok)) return;
    }
}

extern "C" __global__ void tv_ctr_rel(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    (void)work;
    for (u32 i = 0; i < n; ++i) {
        __syncthreads();
        int ok = 1;
        if (threadIdx.x == 0u) {
            ctr_red_release(ctr);
            ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);
        }
        if (!ctr_verdict(ok)) return;
    }
}

// Members of group p of a grid of g CTAs: 8, the last group fewer.
__device__ __forceinline__ u32 ctr_members(u32 p, u32 g)
{
    const u32 first = p * CTR_GROUP;
    return g - first < CTR_GROUP ? g - first : CTR_GROUP;
}

extern "C" __global__ void tv_ctr_8(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    (void)work;
    const u32 g = gridDim.x, ng = (g + CTR_GROUP - 1u) / CTR_GROUP;
    const u32 p = blockIdx.x / CTR_GROUP, q = (p + 1u) % ng;
    const u32 mq = ctr_members(q, g);
    for (u32 i = 0; i < n; ++i) {
        __syncthreads();
        int ok = 1;
        if (threadIdx.x == 0u) {
            ctr_red_release(ctr + CTR_LINE * p);
            ok = ctr_spin_tight(ctr + CTR_LINE * q, (base + i + 1u) * mq, abort_flag);
        }
        if (!ctr_verdict(ok)) return;
    }
}

extern "C" __global__ void tv_ctr_split(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    for (u32 i = 0; i < n; ++i) {
        __syncthreads();
        if (threadIdx.x == 0u) ctr_red_release(ctr);
        ctr_work(work);
        __syncthreads();
        int ok = 1;
        if (threadIdx.x == 0u) ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);
        if (!ctr_verdict(ok)) return;
    }
}

extern "C" __global__ void tv_ctr_work(u32* ctr, u32 base, u32 n, u32 work, u32* abort_flag)
{
    (void)base;
    (void)abort_flag;
    for (u32 i = 0; i < n; ++i) {
        __syncthreads();
        ctr_work(work);
        __syncthreads();
    }
    // one arrival a launch, so the host can check every CTA ran to the end
    if (threadIdx.x == 0u) atomicAdd(ctr, 1u);
}
