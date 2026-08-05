// The "floor" kernels: what `tv_slot` would cost if it did less.
//
// ## The question they answer
//
// The 2026-08-05 measurement on the published model reads 2.50 GB at
// 431 GB/s where `tv_f16` reads 7.27 GB at 662 on the same card, the same
// stream and the same protocol. The LLVQ arm's DRAM floor is 2.50/0.662 =
// 3.78 ms against 5.805 measured: **~2 ms unaccounted for**. The perf audit
// of the same day lists four candidates whose bounds *overlap* — 8-way bank
// conflicts on the staged activations, unmasked latency, grid underfill,
// launch gaps — and says plainly that an aggregate over 252 matrices cannot
// separate them. That is what these kernels are for.
//
// ## The decomposition
//
// Each stage adds exactly one thing `tv_slot` does, keeping everything else
// identical — same grid, same tiling, same two barriers, same `slot_byte`
// addressing, same five-word window:
//
//     tv_floor_stream   bases + the five payload words
//     tv_floor_tab      + the dispersed `tab[id]` gather
//     tv_floor_xs       + the 24 staged activation reads
//     tv_slot           + the decode itself (masks, selects, 24 FMAs)
//
// Three differences, three attributions:
//
//     tab    − stream  =  the gather                    (audit: ~0.5 ms)
//     xs     − tab     =  the shared-memory access      (audit: 0.3–2.9 ms)
//     slot   − xs      =  the decode ALU                (audit: ~0 ms)
//     stream − 3.78 ms =  the stream's own read pattern (audit: ~0.3 ms)
//
// The last line is the one that decides the format question: if
// `tv_floor_stream` already sits near 5.6 ms, the Slot32 byte stream is the
// ceiling and reclaiming bits comes first; if it sits near 3.8, the read
// pattern is exonerated and the gisement is everything above it.
//
// ## Why anti-elision matters more than the arithmetic
//
// None of these kernels computes anything meaningful, and a compiler that
// proves so deletes the loads — leaving a kernel that measures the launch gap
// and nothing else, which would read as a spectacular floor. Every stage
// therefore folds its loads into the value written to `y`. The arithmetic is
// deliberately trivial (xor, adds): it must not be *free*, but it must also
// not add a dependency chain `tv_slot` does not have.
//
// ⚠️ These are diagnostics, not baselines. Nothing here computes a matvec, so
// no ratio may ever be quoted against them.

#ifndef LLVQ_FLOOR_CUH
#define LLVQ_FLOOR_CUH

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

template <int STAGE>
__device__ __forceinline__ float floor_block(const u32* __restrict__ words,
                                             const u32* __restrict__ bases,
                                             const ClassRec* __restrict__ tab,
                                             u32 b,
                                             const float* xb)
{
    u32 byte = slot_byte(bases, b);
    u32 w = byte >> 2;
    u32 w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
        w3 = words[w + 3], w4 = words[w + 4];

    // Anti-elision. The xor keeps all five loads live at one instruction, and
    // the mask keeps the float small enough that 252 matrices of accumulation
    // stay finite — an inf would be a legal answer the driver computes fast.
    u32 mix = w0 ^ w1 ^ w2 ^ w3 ^ w4;
    float acc = (float)(mix & 0xffffu);

    if constexpr (STAGE >= 2) {
        // The class field, extracted exactly as `slot_fields` extracts it, so
        // the gather lands on the same entry with the same divergence. Not a
        // simplification: `id` drives which cache line is touched, and a
        // uniform id would turn a scattered gather into a broadcast.
        u32 sh = (byte & 3u) * 8u;
        u32 id = (u32)((((u64)w1 << 32) | (u64)w0) >> sh) & 0x1ffu;
        const ClassRec r = tab[id];
        acc += r.vals[0] + (float)r.len;
    }

    if constexpr (STAGE >= 3) {
        // The 24 shared reads, same order and same stride as `slot_dot`.
        // Lane L reads `xs[24L + j]`, and 24L mod 32 takes four values, so
        // this is where the audit predicts an 8-way bank conflict. Summed
        // rather than multiplied: an FMA here would import the decode's ALU
        // cost into a stage meant to isolate the access.
#pragma unroll
        for (u32 i = 0; i < LLVQ_DIM; ++i) acc += xb[i];
    }
    return acc;
}

// One kernel per stage, each a copy of `tv_slot`'s shell.
//
// Copied rather than templated over the body *inside* one kernel because the
// shell is what must not vary: same warp-per-row mapping, same `ntiles`, same
// two barriers, same absence of a bounds guard. A macro keeps the four texts
// from drifting while leaving each one a distinct entry point.
#define LLVQ_FLOOR_KERNEL(NAME, STAGE)                                              \
    extern "C" __global__ void NAME(const u32* __restrict__ words,                  \
                                    const u32* __restrict__ bases,                  \
                                    const ClassRec* __restrict__ tab,               \
                                    const float* __restrict__ x,                    \
                                    float* __restrict__ y,                          \
                                    u32 nblocks)                                    \
    {                                                                               \
        extern __shared__ float xs[];                                               \
        u32 lane = threadIdx.x & 31u;                                               \
        u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                    \
        u32 b0r  = row * nblocks;                                                   \
        float acc = 0.0f;                                                           \
        u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;                    \
        for (u32 t = 0; t < ntiles; ++t) {                                          \
            u32 jlo = t * TILE_BLOCKS;                                              \
            u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;    \
            u32 n   = (jhi - jlo) * LLVQ_DIM;                                       \
            __syncthreads();                                                        \
            for (u32 i = threadIdx.x; i < n; i += blockDim.x)                       \
                xs[i] = x[jlo * LLVQ_DIM + i];                                      \
            __syncthreads();                                                        \
            for (u32 j = jlo + lane; j < jhi; j += 32u)                             \
                acc += floor_block<STAGE>(words, bases, tab, b0r + j,               \
                                          xs + (j - jlo) * LLVQ_DIM);               \
        }                                                                           \
        acc = warp_sum(acc);                                                        \
        if (lane == 0) y[row] = acc;                                                \
    }

LLVQ_FLOOR_KERNEL(tv_floor_stream, 1)
LLVQ_FLOOR_KERNEL(tv_floor_tab, 2)
LLVQ_FLOOR_KERNEL(tv_floor_xs, 3)

#endif  // LLVQ_FLOOR_CUH
