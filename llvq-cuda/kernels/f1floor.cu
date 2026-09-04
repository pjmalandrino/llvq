// tv_f1_*: the FLOOR OF THE F1 DECODER TABLE — the lookups an F1 decoder would
// perform, and nothing else.
//
// Lead F1 stops unfolding the served index: 48 bits written and 48 bits read,
// against 2.16 written and 4.804 served today. Its quality is measured and it
// is fine. The risk moved to the decoder, because decoding becomes
// `label → point` through a table, and that table does not fit in shared
// memory: 67 orbits at the ends and 9 in the middle, six bytes an entry, is
// 3,336 KiB against the card's 101,376 B opt-in — 34× over
// (*computed*, `llvq-bench/examples/f1table.rs`). A model pass over the 4B's
// 3,633,315,840 projection weights is 151.4 M blocks, so **454 M lookups**:
// 14.5 GB of table reads against 0.98 GB of weight reads, fifteen to one.
//
// Three previous format leads died on decode cost and none on quality. E1v
// reached 2.3877 b/weight — better bytes than F1 — and measured 0.25× FP16
// (`docs/mesures/e1v-cuda-2026-08-16.txt`). So this file exists to answer one
// question before a week is spent writing `tv_l3e8`: **does the L2 absorb those
// lookups?**
//
// ## The ladder, and why no arm is read on its own
//
//     Didx  = t(hash3) − t(nullk)     manufacturing the indices, which a real
//                                     F1 kernel gets FREE from the weight word
//                                     this bench does not read
//     D(S)  = t(tab3, S) − t(hash3)   THE TABLE ALONE — hashes and address
//                                     arithmetic cancel exactly
//
// Every arm runs `tv_nullk`'s grid, in `tv_nullk`'s process: one warp per row,
// 256 threads = 8 rows per block, the same tiling, the same two barriers, the
// same 12,288 B staging, the same `warp_sum`, the same tail epilogue, the same
// store. `docs/format-noyau.md` §6 forbids subtracting `nullk` from an arm on
// another grid, and that prohibition is what struck out F1d's first threshold
// on 2026-09-04.
//
// ## Why the footprint is swept instead of the access being modelled
//
// 🚨 A uniform draw over the whole 3,336 KiB is NOT an optimistic floor, and
// calling it one was the error this design corrects. The access is strongly
// skewed: of the 67 end orbits, **two hold 256 of the 512 regions** — half the
// end lookups, a third of all lookups, inside 48 KiB (*computed*, f1table.rs,
// orbit sizes `[128, 128, 4, 4, ...]`). A uniform draw destroys a hot set the
// hardware would have cached for free, so a uniform arm is *pessimistic* by an
// unbounded amount and a kill read off it would be a kill on the mixer.
//
// Uniform access over `S` bytes is monotone in `S`, so sweeping the footprint
// brackets the truth from both sides without inventing a distribution nobody
// has measured:
//
//     D(12 KiB)  ≤  D(F1's real, skewed access)  ≤  D(3,336 KiB)
//
// ## The shared-memory arms, and why they are the point
//
// If a third of lookups fall in 48 KiB, that 48 KiB can be *placed* in shared
// memory rather than left to the cache — and placement is what a competing
// 0.9 GB weight stream cannot evict. This is the only design that might make F1
// fit at all, so it is measured rather than reasoned about:
//
//     24 KiB hot → 36 KiB/block → 2 blocks/SM, covers 17% of lookups
//     48 KiB hot → 60 KiB/block → 1 block/SM,  covers 33%
//     (today: 12 KiB/block, 8 blocks/SM)
//
// ⚠️ Those arms may NOT be subtracted from `tv_nullk`: a 60 KiB/block arm and a
// 12 KiB/block one do not have the same occupancy, and differencing across that
// is §6's prohibition in substance. Each shared arm therefore has its own
// matched anchor, `tv_f1_smem_fill`, which stages the same bytes and performs
// no lookup. `t(smem) − t(smem_fill)` is the only legitimate reading.
//
// ## The trap, and what defuses it
//
// 🚨 A kernel whose loads feed nothing is a kernel the compiler deletes, and it
// would return a flattering floor that nothing in the output would report —
// `nullk.cu` documents the same hazard for its staging. So the loaded words are
// folded into a float that multiplies every one of the 24 slots: the chain
// global → register → `acc` → `y` is complete, and the FMA count is identical
// to `nullk`'s so the arms stay comparable. `bin/f1floorbench` additionally
// asserts that `y` differs from `nullk`'s and reproduces round to round; a
// deleted load shows up as the first check failing.
//
// ⚠️ Like `nullk` and like `sol` in `bin/rankbench`, these arms have no
// reference to be checked against. They compute no product of the model. What
// is asked of them is to be OBSERVABLE, not correct, and the bench says so.
//
// Same assembly contract as `planes.cu` and `nullk.cu`: NVRTC has no file
// system, the host concatenates, and the guard below only resolves from disk
// under a host check (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif

// A 32-bit finaliser, three multiplies and three shifts. Cheap enough that
// `Didx` stays small, and strong enough that consecutive block indices land far
// apart in the table — a weak mixer would give the arm a locality F1 does not
// have, which is the same error in the opposite direction.
__device__ __forceinline__ u32 f1_mix(u32 h)
{
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// The three loaded words folded into a multiplier. Reading only the low byte
// keeps the conversion one instruction and leaves the value dependent on every
// load, so none of the three can be dropped.
__device__ __forceinline__ float f1_fold(u32 a, u32 b, u32 c)
{
    return (float)((a ^ b ^ c) & 0xffu);
}

// ---------------------------------------------------------------------------
// Rung 1 — the indices, no load. Isolates what F1 gets free from its weight word.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1_hash3(const float* __restrict__ rscale,
                                       const float* __restrict__ tail,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       u32 nblocks,
                                       u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
            u32 h0 = f1_mix(row * 0x9e3779b9u + j);
            u32 h1 = f1_mix(h0);
            u32 h2 = f1_mix(h1);
            float f = f1_fold(h0, h1, h2);
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(f, xb[s], acc);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// ---------------------------------------------------------------------------
// Rung 2 — the same, plus three loads. `tab_mask` is `entries − 1`, so the
// footprint is a power of two and the address is one AND.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1_tab3(const float* __restrict__ rscale,
                                      const float* __restrict__ tail,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      const u32* __restrict__ tab,
                                      u32 tab_mask,
                                      u32 nblocks,
                                      u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
            u32 h0 = f1_mix(row * 0x9e3779b9u + j);
            u32 h1 = f1_mix(h0);
            u32 h2 = f1_mix(h1);
            float f = f1_fold(tab[h0 & tab_mask], tab[h1 & tab_mask], tab[h2 & tab_mask]);
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(f, xb[s], acc);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// ---------------------------------------------------------------------------
// The shared-memory pair. `hot_words` u32 are staged after the activation tile
// and a lookup that falls inside the hot range reads shared instead of global.
// `hot_frac` is the share of lookups routed there, in 1/256ths, so the bench can
// sweep coverage without changing the footprint.
//
// ⚠️ Read ONLY as `t(smem) − t(smem_fill)`. The two share a shared-memory
// request, hence an occupancy; `tv_nullk` does not, and differencing across
// that is what §6 forbids.
// ---------------------------------------------------------------------------
#define F1_SMEM_BODY(DO_LOOKUP)                                                                   \
    extern __shared__ unsigned char smem[];                                                       \
    float* xs = (float*)smem;                                                                     \
    u32* hot = (u32*)(smem + TILE_COLS * 4u);                                                     \
    u32 lane = threadIdx.x & 31u;                                                                 \
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                      \
    float acc = 0.0f;                                                                             \
    for (u32 i = threadIdx.x; i < hot_words; i += blockDim.x) hot[i] = tab[i];                     \
    __syncthreads();                                                                              \
    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;                                      \
    for (u32 t = 0; t < ntiles; ++t) {                                                            \
        u32 jlo = t * TILE_BLOCKS;                                                                \
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;                      \
        u32 n   = (jhi - jlo) * LLVQ_DIM;                                                         \
        __syncthreads();                                                                          \
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];          \
        __syncthreads();                                                                          \
        for (u32 j = jlo + lane; j < jhi; j += 32u) {                                             \
            const float* xb = xs + (j - jlo) * LLVQ_DIM;                                          \
            u32 h0 = f1_mix(row * 0x9e3779b9u + j);                                               \
            u32 h1 = f1_mix(h0);                                                                  \
            u32 h2 = f1_mix(h1);                                                                  \
            float f = DO_LOOKUP;                                                                  \
            _Pragma("unroll")                                                                     \
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(f, xb[s], acc);                    \
        }                                                                                         \
    }                                                                                             \
    acc = warp_sum(acc);                                                                          \
    if (lane == 0) {                                                                              \
        float tv = 0.0f;                                                                          \
        u32 tc0 = nblocks * LLVQ_DIM;                                                             \
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];               \
        y[row] = acc * rscale[row] + tv;                                                          \
    }

/// One lookup: shared when the mixed index falls in the hot share, global
/// otherwise. The branch is per lane and data-dependent, exactly as a real
/// hot/cold split would be.
#define F1_PICK(H) (((H) >> 24) < hot_frac ? hot[(H) % hot_words] : tab[(H) & tab_mask])

extern "C" __global__ void tv_f1_smem(const float* __restrict__ rscale,
                                      const float* __restrict__ tail,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      const u32* __restrict__ tab,
                                      u32 tab_mask,
                                      u32 hot_words,
                                      u32 hot_frac,
                                      u32 nblocks,
                                      u32 tail_w)
{
    F1_SMEM_BODY(f1_fold(F1_PICK(h0), F1_PICK(h1), F1_PICK(h2)))
}

/// The matched anchor: stages the same `hot_words`, performs no lookup. Its
/// difference against `tv_f1_smem` is the lookups alone, at that occupancy.
extern "C" __global__ void tv_f1_smem_fill(const float* __restrict__ rscale,
                                           const float* __restrict__ tail,
                                           const float* __restrict__ x,
                                           float* __restrict__ y,
                                           const u32* __restrict__ tab,
                                           u32 tab_mask,
                                           u32 hot_words,
                                           u32 hot_frac,
                                           u32 nblocks,
                                           u32 tail_w)
{
    (void)tab_mask;
    (void)hot_frac;
    F1_SMEM_BODY(f1_fold(h0, h1, h2))
}
