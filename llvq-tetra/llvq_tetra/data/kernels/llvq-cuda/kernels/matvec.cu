// The fused kernel: dequantize and multiply in one pass, never materializing
// a weight. This is the contribution — everything else in this crate exists to
// make it measurable.
//
// A direct port of `tv_slot` / `tv_f16` from `llvq-metal/src/bin/thesis.rs`.
// Same tiling, same two barriers, same block-local pointer, same accumulator
// association. Deliberately the *tiled* form and not `bin/matvec`'s: staging a
// whole row needs 38 KB at d_in = 9728, which would cap the block at 2 per SM
// against 6 — and this kernel lives by slipping into memory-latency bubbles.

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

// 128 blocks = 3072 columns = 12 KB, injected by the host so the staging size
// and the kernel's tiling cannot drift apart. That drift is a real defect the
// Metal side carried for months (`TILE_BLOCKS` defined twice, unlinked).
#ifndef TILE_BLOCKS
#error "the host must define TILE_BLOCKS"
#endif
#define TILE_COLS (TILE_BLOCKS * LLVQ_DIM)

// f16 → f32 in one hardware instruction, without `cuda_fp16.h`.
//
// The runtime image ships no `-dev` package, so `__half2float` is not
// reachable; and a software conversion would be a rigged baseline — it would
// charge the FP16 arm for work the hardware does for free, and flatter the
// ratio this file exists to measure. `cvt.f32.f16` is exactly what
// `__half2float` compiles to.
//
// 🕳️ **The host branch used to return 0, and that stopped being acceptable on
// 2026-08-09.** Its licence was one sentence — "nothing executes it; only the
// surrounding code is being type-checked" — and `tail_dot_h` below broke it:
// the tail epilogue is scalar register arithmetic, so the clang++ harness
// *runs* it, and a stubbed widening would have made that probe pass on any
// tail whatsoever. This is now the same exact software widening `rot_h2f`
// (llvq_rot.cuh) already carries for the same reason, and it costs the device
// nothing: the `#else` arm is what NVRTC compiles, unchanged.
//
// Widening binary16 to binary32 is exact, so a correct software conversion is
// bit-identical to `cvt.f32.f16`. There is nothing to approximate and nothing
// to flatter — the "rigged baseline" argument above is about the *device*
// path, which this branch never reaches.
__device__ __forceinline__ float h2f(unsigned short h)
{
#ifdef LLVQ_HOST_BUILD
    u32 sign = (u32)(h & 0x8000u) << 16;
    u32 exp = (h >> 10) & 0x1fu;
    u32 man = h & 0x3ffu;
    u32 bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;  // ±0
        } else {
            // Subnormal binary16 is normal in binary32: shift the mantissa up
            // until its leading 1 falls off, and charge the exponent for it.
            u32 e = 127u - 15u + 1u;
            while ((man & 0x400u) == 0u) {
                man <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);  // inf / NaN, payload kept
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
    }
    float r;
    __builtin_memcpy(&r, &bits, sizeof r);
    return r;
#else
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
#endif
}

// Sum across the 32 lanes of a warp.
//
// Metal's `simd_sum` and this butterfly sum in different orders, so the two
// backends' outputs are *not* bit-comparable — each is checked against the
// shared f64 reference instead, and two worst errors are published rather
// than one delta.
__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
    return v;
}

// One warp per output row, 256 threads = 8 rows per block.
//
// There is deliberately no `if (row >= d_out) return;`. Two reasons, and both
// are silent failures rather than errors: a `return` before `__syncthreads()`
// deadlocks, and it would invalidate the full-warp mask `0xffffffff` under
// Independent Thread Scheduling. The host asserts `d_out % 8 == 0` instead —
// true of every shape in the 4B and the 8B.
//
// The tail epilogue reads `x` from **global** memory, not from `xs`: under
// tiling the tail columns sit past the last staged window.
extern "C" __global__ void tv_slot(const u32* __restrict__ words,
                                   const u32* __restrict__ bases,
                                   const ClassRec* __restrict__ tab,
                                   const float* __restrict__ gscale,
                                   const float* __restrict__ rscale,
                                   const float* __restrict__ tail,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   u32 nblocks,
                                   u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        // Two barriers, not one: the second orders the fill against the
        // readers, the first stops the next fill from racing a straggler
        // still reading the previous tile. `ntiles` depends only on `nblocks`,
        // so both are uniform across the block — which CUDA requires.
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gscale, b0r + j,
                            xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// `tv_slot` over a row-concatenation of several matrices sharing one input.
//
// ## Why this exists
//
// q/k/v consume the same activation, and so do gate/up. Launched separately,
// `k_proj` and `v_proj` are 1024 rows = 128 blocks against the 852 a L40S
// holds resident — 15 % occupancy — and the 2026-08-05 attribution job
// measured them at 157 GB/s against 469 for a well-filled shape, for a ratio
// against FP16 of 1.06×, i.e. nothing. Concatenating q+k+v by rows launches
// 768 blocks instead of three grids of 512/128/128, and the same geometry
// removes 108 launches per token.
//
// ## The one thing that is not shared: the gain centroids
//
// Every matrix carries its own two centroids, and `slot_dot` ends on
// `gscale[gain]`. They cannot be folded into `rscale` — `gscale` is selected
// by the *block's* gain bit, `rscale` is per row — so the segment has to be
// resolved somewhere. It is resolved here, once per row, by an offset table:
// `gs_off[row]` names where that row's pair starts in `gscale`. The read is
// uniform across the warp (one warp owns one row), so it broadcasts.
//
// Everything else concatenates without a special case: `nblocks` and `tail_w`
// are equal by construction (same `d_in`), `rscale` and `tail` are indexed by
// row already, and the block stream is transcoded from the concatenated
// indices, so its groups of 32 simply straddle the segment boundaries — which
// the addressing has always tolerated, since they already straddle rows.
//
// ⚠️ `tv_slot` is left untouched on purpose. It is the object every published
// millisecond refers to; this is a second kernel measured beside it, not a
// replacement, until a job says the fusion is worth adopting.
extern "C" __global__ void tv_slot_seg(const u32* __restrict__ words,
                                       const u32* __restrict__ bases,
                                       const ClassRec* __restrict__ tab,
                                       const float* __restrict__ gscale,
                                       const u32* __restrict__ gs_off,
                                       const float* __restrict__ rscale,
                                       const float* __restrict__ tail,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       u32 nblocks,
                                       u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    const float* gs = gscale + gs_off[row];
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gs, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}

// f32 → f16, round to nearest even. The inverse of `h2f`, same reasoning: no
// `cuda_fp16.h` in the runtime image, and `cvt.rn.f16.f32` is the instruction
// `__float2half` compiles to.
// 🕳️ **The host branch used to return 0, the same defect `h2f` carried until
// 2026-08-09: fixed there, left here.** Its licence was the same sentence,
// "nothing executes it, only the surrounding code is type-checked". It stopped
// holding on 2026-08-10, when the AWQ arm arrived: its kernel writes its output
// through `f2h`, so a host C++ test of `awq_gemv_g128` would have read zeros and
// gone green on any kernel whatsoever, an empty one included. This is exactly
// the pattern of §5 of CLAUDE.md: a neutral-valued parameter that makes the test
// blind to what it claims to cover.
//
// Round to nearest even is reproduced here in software. Unlike the widening in
// `h2f`, the CONTRACTION is not exact: it rounds, and a badly done rounding
// would pass most tests and fail only at the half-ulps. Hence the three explicit
// cases, overflow to infinity, subnormal, and the even half-case, rather than a
// naive shift.
//
// ⚠️ The `#else` that NVRTC compiles is UNCHANGED. The text of the translation
// unit grows, so its sha256 moves; the PTX does not move. That is precisely why
// the sha256 of the PTX should be printed next to the one of the source: the
// second detects changes the first does not see, and the other way round.
__device__ __forceinline__ unsigned short f2h(float f)
{
#ifdef LLVQ_HOST_BUILD
    u32 bits;
    __builtin_memcpy(&bits, &f, 4);
    u32 sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    u32 man = bits & 0x7fffffu;

    if (((bits >> 23) & 0xffu) == 0xffu) {
        // NaN keeps a non-zero mantissa; infinity loses it.
        return (unsigned short)(sign | 0x7c00u | (man ? 0x0200u : 0u));
    }
    if (exp >= 0x1f) {
        return (unsigned short)(sign | 0x7c00u);  // overflows → ±inf
    }
    if (exp <= 0) {
        // Subnormal binary16, or zero. The implicit bit becomes explicit
        // again, then shift by what the exponent is short of.
        if (exp < -10) {
            return (unsigned short)sign;
        }
        man |= 0x800000u;
        u32 shift = (u32)(14 - exp);
        u32 r = man >> shift;
        u32 rem = man & ((1u << shift) - 1u);
        u32 half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) {
            ++r;
        }
        return (unsigned short)(sign | r);
    }
    u32 r = man >> 13;
    u32 rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        ++r;
        if (r == 0x400u) {  // the mantissa overflowed into the exponent
            r = 0;
            ++exp;
            if (exp >= 0x1f) {
                return (unsigned short)(sign | 0x7c00u);
            }
        }
    }
    return (unsigned short)(sign | ((u32)exp << 10) | r);
#else
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#endif
}

// The `TailPolicy::KeepExact` epilogue of every **f16-storing** entry point,
// with the tail resident as binary16.
//
// ## Why the tail is halves here and floats in the bench arms
//
// A column that is not a multiple of 24 wide leaves `d_in mod 24` columns
// unquantized. The file stores them as `f32` (`calib.rs` narrows them there on
// purpose, so the file and the evaluated model are one object), and the fused
// loader used to carry that width straight onto the card. That was inherited
// storage, never a precision decision: **the arm this path is compared
// against holds those same columns at binary16**, because `sealed::load`
// narrows the whole decoded matrix — tail included — to the run's dtype, and
// `bin/fusedrun` runs both arms at `DType::F16`. Keeping 24 mantissa bits on
// the card bought precision the reference does not have, at 2 bytes a weight.
//
// So this is an *alignment*, not a saving taken out of quality: 16,957,440
// tail weights on the published 4B, 33.9 MB, −0.067 b/param whole-model. The
// rounding it introduces lands ~12× (4B `q/k/v/o/gate/up`) to ~34× (4B
// `down_proj`) under the binary16 rounding `f2h` already applies to `y` two
// lines later — see `llvq_llm::fused::tail_f16_bits`, which owns the host half
// and is where that arithmetic is pinned.
//
// The bench kernels (`tv_slot`, `tv_slot_seg`, `tv_planes`, `tv_planes12x`,
// `tv_golay70`) keep `const float*`, deliberately: they are the objects every
// published millisecond and every published kernel `b/weight` refer to, and
// `rtbits` bills their tail at 32 bits. Moving them would move numbers that
// are already in the dossier. A future `tv_*_seg_h` — the A4 wiring — belongs
// on *this* side of that line and should call this function.
//
// ## What this function is for, beyond deduplication
//
// It contains no barrier, no shuffle and no atomic, so unlike the kernels
// that call it a single-threaded host *can* execute it — which is why the
// epilogue was lifted out of the three call sites at all.
// `llvq-llm/tests/host_tail_h.cpp` runs it against an f64 reference. The
// kernels around it stay compile-only.
__device__ __forceinline__ float tail_dot_h(const unsigned short* __restrict__ tail,
                                            const float* __restrict__ xt,
                                            u32 row,
                                            u32 tail_w)
{
    float tv = 0.0f;
    // Multiply-then-add, not `__fmaf_rn`: this is the association the f32
    // epilogue had, and the only intended change is the stored width of the
    // weight. Widening happens before the multiply, so the product and the
    // sum stay f32 exactly as before.
    for (u32 i = 0; i < tail_w; ++i) tv += h2f(tail[row * tail_w + i]) * xt[i];
    return tv;
}

// `tv_slot` writing f16 — the shape an inference runtime needs.
//
// The model is f16 end to end, so a f32 result costs one conversion kernel per
// projection: 252 launches a token, on a decode already dominated by launch
// latency. Writing the half directly removes them.
//
// A separate kernel rather than a flag on `tv_slot`, for the same reason
// `tv_slot_seg` is separate: `tv_slot` is the object every published
// millisecond refers to, and its register allocation must not move because an
// inference path wanted a different store.
//
// The accumulation is unchanged — f32 throughout, same association, same
// `warp_sum`. Only the final store narrows, exactly where candle's conversion
// kernel would have narrowed it.
extern "C" __global__ void tv_slot_h(const u32* __restrict__ words,
                                     const u32* __restrict__ bases,
                                     const ClassRec* __restrict__ tab,
                                     const float* __restrict__ gscale,
                                     const float* __restrict__ rscale,
                                     const unsigned short* __restrict__ tail,
                                     const float* __restrict__ x,
                                     unsigned short* __restrict__ y,
                                     u32 nblocks,
                                     u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += slot_dot(words, bases, tab, gscale, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        // `x + tc0`, from **global** memory: under tiling the tail columns sit
        // past the last staged window.
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[row] = f2h(acc * rscale[row] + tv);
    }
}

// The FP16 witness. Same shape, same tiling, 128-bit loads on both sides.
//
// Loading eight halves per lane rather than one is not a courtesy to the
// baseline: K−1(b) measured the load width to be worth ~5 % **on both arms**
// in Metal, so narrowing one side would raise the ratio for a reason that is
// not a property of the format.
//
// Note what this arm is and is not: `w` holds the LLVQ reconstruction rounded
// to binary16, in the rotated basis — not the model's own FP16 checkpoint.
// Both arms compute the same mathematical product to within rounding. It is a
// baseline of *cost*, not of quality.
extern "C" __global__ void tv_f16(const unsigned short* __restrict__ w,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  u32 d_in)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = 0.0f;

    u32 ntiles = (d_in + TILE_COLS - 1u) / TILE_COLS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 c0 = t * TILE_COLS;
        u32 n  = c0 + TILE_COLS < d_in ? TILE_COLS : d_in - c0;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[c0 + i];
        __syncthreads();

        // 8 halves = one 128-bit load per lane, 4 KB contiguous per warp.
        const uint4* w8 = (const uint4*)(w + row * d_in + c0);
        for (u32 j = lane; j < (n >> 3); j += 32u) {
            uint4 p = w8[j];
            u32 c = j << 3;
            const u32 q[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (u32 k = 0; k < 4; ++k) {
                acc = __fmaf_rn(h2f((unsigned short)(q[k] & 0xffffu)), xs[c + 2 * k], acc);
                acc = __fmaf_rn(h2f((unsigned short)(q[k] >> 16)), xs[c + 2 * k + 1], acc);
            }
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}
