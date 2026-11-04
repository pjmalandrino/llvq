// llvq_pers.cuh: the device half of the persistent Tetra kernels.
//
// One header for the L2 bench arms (`tetra48_pers.cu`) and for route P's
// segment kernel (`pers_segment.cu`). Design: docs/archive/debit-vague5-
// 2026-11-02/persistent.md §3; host half: `src/pers.rs`.
//
// The contract this header keeps is bit identity with the served path:
//
//   * a Tetra row is the served `tv_tetra48_h` row: lane l adds blocks
//     j = l, l+32, ... in increasing j, the 24 products of a block in
//     TETRA48_ORDER, then `warp_sum`, then the epilogue. The three places
//     the served source leaves to contraction are SPELLED here as the PTX of
//     the served unit shows them (docs/mesures/ptx-probe-2026-11-04.txt §2):
//       site 1  acc = fma(acc_b * gscale[g], invnorm[m], acc)
//       site 2  tv  = fma(h2f(tail[i]), x[i], tv), i increasing
//       site 3  y   = f2h(fma(acc, rscale[row], tv))
//     so no compiler, NVVM or the driver's ptxas, can move them;
//   * an int4 row is `q4_staged_dot`'s row, explicit intrinsics throughout;
//   * the rotation is `rot_apply`'s: the same butterfly stages in the same
//     order and the same per-output chains; only where the values sit
//     (registers, shared) differs;
//   * candle's RMSNorm, f16 add and mul are reproduced op for op, SiLU is a
//     table candle's own kernel filled.
//
// No static `__shared__` anywhere: every CTA-shared scalar lives in the
// header of the dynamic buffer (pers.rs::SMEM_HEADER). That is what lets the
// host harness (tests/host_grid.h) run every CTA of a cooperative grid at
// once, each with its own buffer, where a `static` would be one copy for all.
//
// Composition (NVRTC has no file system, the host concatenates):
// llvq_slot.cuh, matvec.cu, llvq_f1rank.cuh, llvq_f1rank_v3.cuh,
// llvq_tetra48.cuh, llvq_rot.cuh, then this file.

#ifndef LLVQ_PERS_CUH
#define LLVQ_PERS_CUH

#ifndef LLVQ_TETRA48_CUH
#include "llvq_tetra48.cuh"
#endif
#ifndef LLVQ_ROT_CUH
#include "llvq_rot.cuh"
#endif
#ifndef TILE_COLS
#include "matvec.cu"
#endif

#ifndef PERS_SPIN_LIMIT
#define PERS_SPIN_LIMIT (1u << 17)  // about 67 ms at the 512-ns back-off cap, then abort
#endif

// ---------------------------------------------------------------------------
// 0. The dynamic shared buffer and its header (pers.rs::SMEM_HEADER = 256 B).
//    u32 0: the claim cursor; u32 1: a wait verdict broadcast; u32 2: the
//    argmax NaN flag of the CTA; u32 32..63: 32 floats of norm partials.
// ---------------------------------------------------------------------------
#define PERS_SMEM_HEADER 256u
#define PERS_HDR_NEXT 0u
#define PERS_HDR_OK 1u
#define PERS_HDR_SUM 32u

#ifdef LLVQ_HOST_BUILD
// The harness gives each CTA its own buffer and models the spin-wait yield.
unsigned char* hostgrid_smem();
#define PERS_SMEM_BASE() (hostgrid_smem())
#else
extern __shared__ float4 pers_smem4[];
#define PERS_SMEM_BASE() (reinterpret_cast<unsigned char*>(pers_smem4))
#endif

__device__ __forceinline__ u32* pers_hdr() { return reinterpret_cast<u32*>(PERS_SMEM_BASE()); }
__device__ __forceinline__ float* pers_sum() { return reinterpret_cast<float*>(PERS_SMEM_BASE()) + PERS_HDR_SUM; }

// ---------------------------------------------------------------------------
// 1. Synchronization. One monotonic barrier counter for the life of the
//    runtime; targets are host bases plus (ordinal + 1) * gridDim.x, compared
//    wrap-safe. The grid-sync pattern of cooperative groups: bar.sync, one
//    thread fences and arrives, spins with back-off, fences, bar.sync. Every
//    CTA calls each ordinal exactly once, with rows or without.
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 pers_ld_acquire(const u32* p)
{
#ifdef LLVQ_HOST_BUILD
    return *reinterpret_cast<const volatile u32*>(p);
#else
    u32 v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#endif
}

__device__ __forceinline__ bool pers_reached(u32 v, u32 target)
{
    return (int)(v - target) >= 0;  // valid while fewer than 2^31 arrivals separate them
}

// Thread 0 spins until `*ctr` reaches `target`; false when the abort flag is
// up (this CTA's watchdog or another's). The caller broadcasts.
__device__ __forceinline__ int pers_spin(const u32* ctr, u32 target, u32* abort_flag)
{
    u32 ns = 32u, spins = 0u;
    while (!pers_reached(pers_ld_acquire(ctr), target)) {
        __nanosleep(ns);
        if (ns < 512u) ns <<= 1;
        if ((++spins & 255u) == 0u && pers_ld_acquire(abort_flag) != 0u) return 0;
        if (spins >= PERS_SPIN_LIMIT) {
            atomicExch(abort_flag, 1u);
            return 0;
        }
    }
    return 1;
}

__device__ __forceinline__ bool pers_grid_wait(u32* bar, u32 target, u32* abort_flag)
{
    u32* hdr = pers_hdr();
    __syncthreads();  // the CTA's stores precede thread 0's arrival
    if (threadIdx.x == 0u) {
        __threadfence();  // cumulative: orders the CTA's stores before the arrival
        atomicAdd(bar, 1u);
        const int ok = pers_spin(bar, target, abort_flag);
        __threadfence();
        hdr[PERS_HDR_OK] = (u32)ok;
    }
    __syncthreads();
    // No barrier after the read: thread 0 overwrites the verdict only in the
    // next wait, after that wait's opening barrier, which every thread
    // reaches after reading this one.
    return hdr[PERS_HDR_OK] != 0u;
}

// Producer side of a counter, after a __syncthreads() that ordered the
// CTA's stores before thread 0.
__device__ __forceinline__ void pers_signal(u32* cnt)
{
    if (threadIdx.x == 0u) {
        __threadfence();
        atomicAdd(cnt, 1u);
    }
}

// Consumer side: every thread returns once `contributors` signals of this
// epoch have arrived.
__device__ __forceinline__ bool pers_wait_count(const u32* cnt, u32 target, u32* abort_flag)
{
    u32* hdr = pers_hdr();
    __syncthreads();
    if (threadIdx.x == 0u) {
        const int ok = pers_spin(cnt, target, abort_flag);
        __threadfence();
        hdr[PERS_HDR_OK] = (u32)ok;
    }
    __syncthreads();
    return hdr[PERS_HDR_OK] != 0u;
}

// ---------------------------------------------------------------------------
// 2. Work claiming inside a CTA. The CTA's row slice is static (host table);
//    warps take the next row from the shared cursor. A row's bits do not
//    depend on which warp computes it: the lane-to-block map and `warp_sum`
//    are fixed. The caller sets the cursor under a barrier and ends every site
//    with __syncthreads() before it is set again, so no warp still claiming
//    site s draws an index of site s + 1 (review finding 4).
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 pers_claim(u32 lane, u32 step)
{
    u32 r = 0u;
    if (lane == 0u) r = atomicAdd(pers_hdr() + PERS_HDR_NEXT, step);
    return __shfl_sync(0xffffffffu, r, 0);
}

__device__ __forceinline__ void pers_claim_reset(u32 lo)
{
    __syncthreads();  // every warp has left the previous claim loop
    if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_NEXT] = lo;
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 3. Staging. Natural coordinate c of block b lands at b*STRIDE + c, or at
//    b*STRIDE + PERS_INV_ORDER[c] under TRIO, so the trio dot reads slots
//    0..23 in sequence as six aligned float4. Tail columns follow the last
//    block contiguously. Same values as the served staging; `__ldcg` because
//    route P reads activations written in the same launch, and the read-only
//    path is not coherent with them.
// ---------------------------------------------------------------------------
__device__ static constexpr unsigned char PERS_INV_ORDER[24] = {
    0, 1, 2, 3, 4, 16, 8, 5, 17, 18, 6, 9, 7, 10, 11, 19, 12, 13, 14, 15, 20, 21, 22, 23,
};  // the inverse of TETRA48_ORDER; pers.rs::tests reads both literals

template <u32 STRIDE, bool TRIO>
__device__ __forceinline__ u32 pers_xs_index(u32 i, u32 nblocks)
{
    const u32 nmain = nblocks * LLVQ_DIM;
    if (i >= nmain) return nblocks * STRIDE + (i - nmain);
    const u32 b = i / LLVQ_DIM, c = i - b * LLVQ_DIM;
    return b * STRIDE + (TRIO ? (u32)PERS_INV_ORDER[c] : c);
}

template <u32 STRIDE, bool TRIO>
__device__ __forceinline__ void pers_stage_f32(const float* x, float* xs, u32 nblocks, u32 tail_w)
{
    __syncthreads();  // no warp still reads the previous staging
    const u32 n = nblocks * LLVQ_DIM + tail_w;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[pers_xs_index<STRIDE, TRIO>(i, nblocks)] = __ldcg(x + i);
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 4. The Tetra row.
// ---------------------------------------------------------------------------
// Site 1 of the served PTX: `t = acc_b * gscale[g]` (mul.f32, whose only
// consumer is an fma multiplicand), then `fma(t, invnorm[m], acc)`.
__device__ __forceinline__ float pers_fold(float acc, float acc_b, u32 q[6], u32 hi16, const float* gscale,
                                           const float* invnorm)
{
    const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    const u32 g = (hi16 >> 15) & 1u;
    return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);
}

// `tetra48_dot` in natural staging: the same 24 __fmaf_rn on the same values.
__device__ __forceinline__ float pers_block(float acc, u32 lo, u32 hi16, const F1rTables& t, const float* xb,
                                            const float* gscale, const float* invnorm)
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    float a = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) a = __fmaf_rn(f1r_v3_float(q[i], j), xb[TETRA48_ORDER[4u * i + j]], a);
    }
    return pers_fold(acc, a, q, hi16, gscale, invnorm);
}

// The same in trio staging: slot 4i + j holds natural coordinate
// TETRA48_ORDER[4i + j], read as component j of the i-th float4. Explicit
// float4 makes the six 128-bit loads a property of the source, given a
// 16-byte aligned block: the buffer is float4-declared and 28 floats is 112 B.
__device__ __forceinline__ float pers_block_trio(float acc, u32 lo, u32 hi16, const F1rTables& t, const float* xb,
                                                 const float* gscale, const float* invnorm)
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    float a = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
        const float4 v = x4[i];
        a = __fmaf_rn(f1r_v3_float(q[i], 0u), v.x, a);
        a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, a);
        a = __fmaf_rn(f1r_v3_float(q[i], 2u), v.z, a);
        a = __fmaf_rn(f1r_v3_float(q[i], 3u), v.w, a);
    }
    return pers_fold(acc, a, q, hi16, gscale, invnorm);
}

// One row's lane partial over all its blocks, reduced: the served lane order
// (`tv_tetra48_h.cu:68-82` at any tile that is a multiple of 32).
template <u32 STRIDE, bool TRIO, bool PREFETCH>
__device__ __forceinline__ float pers_tetra_row(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    float acc = 0.0f;
    if (!PREFETCH) {
        for (u32 j = lane; j < nblocks; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc = TRIO ? pers_block_trio(acc, lo, hi16, tab, xs + j * STRIDE, gscale, invnorm)
                       : pers_block(acc, lo, hi16, tab, xs + j * STRIDE, gscale, invnorm);
        }
    } else {
        // Block j + 32's words are in flight while block j decodes: twice the
        // bytes in flight per lane, the same accumulation order.
        u32 lo = 0u, hi16 = 0u;
        if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
        for (u32 j = lane; j < nblocks; j += 32u) {
            u32 lo2 = 0u, hi2 = 0u;
            if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
            acc = TRIO ? pers_block_trio(acc, lo, hi16, tab, xs + j * STRIDE, gscale, invnorm)
                       : pers_block(acc, lo, hi16, tab, xs + j * STRIDE, gscale, invnorm);
            lo = lo2;
            hi16 = hi2;
        }
    }
    return warp_sum(acc);
}

// Sites 2 and 3: the f16 tail, then the epilogue before `f2h`. `xt` is the
// tail of the activation, wherever it is staged: same values.
__device__ __forceinline__ float pers_epilogue(float acc, float rscale, const unsigned short* tail_row,
                                               const float* xt, u32 tail_w)
{
    float tv = 0.0f;
    for (u32 i = 0; i < tail_w; ++i) tv = __fmaf_rn(h2f(tail_row[i]), xt[i], tv);
    return __fmaf_rn(acc, rscale, tv);
}

// ---------------------------------------------------------------------------
// 5. The int4 row: `q4_staged_dot` (tv_q4_h.cu) with the row as a parameter.
//    Explicit intrinsics only, so contraction cannot move anything.
// ---------------------------------------------------------------------------
// The product of an 11-bit scale and a 4-bit code is exact in binary32, so
// even a fused form would round the same sum; the explicit form is kept
// because it is the served text.
__device__ __forceinline__ float pers_q4_deq(u32 q, u32 sbits, u32 bbits)
{
    return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q), h2f((unsigned short)bbits));
}

__device__ __forceinline__ float pers_q4_row(const u32* wq, const unsigned short* scales,
                                             const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                             u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        const u32 p = wq[w0 + wi];
        const u32 c = wi << 3;
        const u32 g = s0 + (c >> 7);
        const u32 sb = scales[g];
        const u32 bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    return warp_sum(acc);
}

// ---------------------------------------------------------------------------
// 6. The rotation, one CTA, in place in `s` (n floats of shared memory).
//    `rot_apply` (rotate.cu, llvq_rot.cuh) runs three phases over the whole
//    vector: load with signs; the Walsh-Hadamard stages len = 1 .. m/2 in
//    increasing order, each pair (j, j+len) becoming (a + b, a - b); then the
//    mix of the k groups, or the scale when k = 1.
//
//    Here the stages below 512 run in registers: one warp a 512-chunk, lane l
//    holding s[16l .. 16l+15], stages 1..8 inside the lane and 16..256 as
//    shfl_xor 1..16. The stages from 512 to m/2 (only when m > 512: 1,024 at
//    the 14B, 4,096 at the 8B) run in shared memory with rot_wht_step. Every
//    output of every stage is the same two operations on the same two values
//    as in rot_apply, and the stages run in the same order, so the bits are
//    rot_apply's whatever thread holds the value.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pers_wht512(float v[16], u32 lane)
{
#pragma unroll
    for (u32 len = 1u; len < 16u; len <<= 1) {
#pragma unroll
        for (u32 i = 0u; i < 16u; ++i) {
            if ((i & len) == 0u) {
                const float a = v[i], b = v[i + len];
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
    }
#pragma unroll
    for (u32 sh = 1u; sh < 32u; sh <<= 1) {
        const bool upper = (lane & sh) != 0u;
#pragma unroll
        for (u32 i = 0u; i < 16u; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], sh);
            v[i] = upper ? (p - v[i]) : (v[i] + p);  // upper: a - b with a the partner's (lower) value
        }
    }
}

// The mix of one column j, in place. rot_mix runs KMAX terms; terms t >= k
// are fma(+0, +0, acc) because the host pads `small` with +0.0f and col with
// 0.0f, and that equals acc + (+0) whatever acc is (-0 becomes +0, every
// other value is kept). Applied once or KMAX - k times it is the same, hence
// the single __fadd_rn below. It matters only when the last real term leaves
// acc at -0, which an fma can do when a product underflows below half the
// least subnormal: tests/pers_matches_served.rs builds that column.
__device__ __forceinline__ void pers_mix_column(float* s, const float* small, u32 m, u32 k, float inv, u32 j)
{
    float col[LLVQ_ROT_KMAX];
#pragma unroll
    for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t) col[t] = t < k ? s[t * m + j] * inv : 0.0f;
    for (u32 g = 0; g < k; ++g) {
        float acc = 0.0f;
#pragma unroll
        for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t)
            if (t < k) acc = __fmaf_rn(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
        if (k < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);
        s[g * m + j] = acc;
    }
}

// Load `n` binary16 values with their signs (rot_load), from global memory.
__device__ __forceinline__ void pers_rot_load(const unsigned short* xin, const u32* signbits, float* s, u32 n)
{
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h2f(__ldcg(xin + i));
        s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;
    }
}

// Phases 2 and 3 on `s`, already loaded and signed; ends with a barrier.
__device__ __forceinline__ void pers_rot_finish(float* s, const float* small, u32 n, u32 m, u32 k, float inv)
{
    __syncthreads();
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, nwarps = blockDim.x >> 5;
    for (u32 c = warp; c < n / 512u; c += nwarps) {
        float v[16];
        float* p = s + c * 512u + 16u * lane;
#pragma unroll
        for (u32 i = 0; i < 16u; ++i) v[i] = p[i];
        pers_wht512(v, lane);
#pragma unroll
        for (u32 i = 0; i < 16u; ++i) p[i] = v[i];
    }
    __syncthreads();
    for (u32 len = 512u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, threadIdx.x, blockDim.x);
        __syncthreads();
    }
    if (k == 1u) {
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) s[i] = s[i] * inv;
    } else {
        for (u32 j = threadIdx.x; j < m; j += blockDim.x) pers_mix_column(s, small, m, k, inv, j);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 7. Candle's operations (candle-kernels 0.9.2), reproduced.
//
//    rmsnorm_f16   reduce.cu:187-228, block 1,024 when n_cols >= 1,024 and 32
//                  below (candle-nn ops.rs:545). Thread t sums columns t,
//                  t + B, ... with `tmp += xi * xi`, then the xor butterfly
//                  16..1, the 32 warp partials, the butterfly again,
//                  `tmp / ncols`, `rsqrtf(mean + eps)`, and `T(scale * x * a)`,
//                  left to right. Whether nvcc contracts `tmp += xi * xi` does
//                  not matter: xi is a widened binary16, 11 significant bits,
//                  so xi * xi has at most 22 and is exact in binary32, and
//                  fma(xi, xi, tmp) rounds the same sum as tmp + xi * xi.
//                  VARIANT keeps both spellings so the host test proves it on
//                  every column it draws (the_norm_is_candles_tree).
//    badd_f16      `__hadd`: the f32 sum rounded to binary16 equals add.f16 RN,
//                  double rounding 24 -> 11 bits being innocuous (24 >= 2*11+2).
//    bmul_f16      `__hmul`: the f32 product of two binary16 is exact (22 bits),
//                  one rounding.
//    usilu_f16     `x / (__half(1) + hexp(-x))`: NOT reproduced. NVRTC 12.4 ships
//                  no cuda_fp16.h, so `hexp` cannot be recompiled. A 65,536-entry
//                  table is filled at load by candle's own kernel over every
//                  binary16 pattern: equal by exhaustion.
//    fast_argmax   reduce.cu:476-523: the 64-bit key of tok_argmax.cu.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned short exact_hadd(unsigned short a, unsigned short b)
{
    return f2h(__fadd_rn(h2f(a), h2f(b)));
}

__device__ __forceinline__ unsigned short exact_hmul(unsigned short a, unsigned short b)
{
    return f2h(__fmul_rn(h2f(a), h2f(b)));
}

__device__ __forceinline__ unsigned short exact_silu_mul(const unsigned short* silu_tab, unsigned short g,
                                                         unsigned short u)
{
    return exact_hmul(silu_tab[g], u);
}

// The scale of candle's rmsnorm over x[0..ncols), every thread of the CTA
// gets it. The CTA's warps play candle's BLOCK virtual threads: real warp r
// takes virtual warps r, r + W, ... Every partial, every butterfly operand
// and every pairing is candle's. Ends with a barrier.
template <int VARIANT>
__device__ float exact_rms_scale(const unsigned short* x, u32 ncols, float eps)
{
    float* s_sum = pers_sum();
    const u32 block = ncols < 1024u ? 32u : 1024u;
    const u32 nvw = block / 32u;
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    __syncthreads();  // s_sum is free
    for (u32 vw = warp; vw < nvw; vw += nwarps) {
        const u32 t = vw * 32u + lane;  // candle's threadIdx.x
        float tmp = 0.0f;
        for (u32 col = t; col < ncols; col += block) {
            const float xi = h2f(__ldcg(x + col));
            tmp = VARIANT == 0 ? __fmaf_rn(xi, xi, tmp) : __fadd_rn(tmp, __fmul_rn(xi, xi));
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));
        if (lane == 0u) s_sum[vw] = tmp;
    }
    __syncthreads();
    if (warp == 0u) {
        float tot = s_sum[block > 32u ? lane : 0u];
        if (block > 32u) {
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) tot = __fadd_rn(tot, __shfl_xor_sync(0xffffffffu, tot, mask));
        }
        const float mean = __fdiv_rn(tot, (float)ncols);
        const float scale = rsqrtf(__fadd_rn(mean, eps));
        __syncwarp();
        if (lane == 0u) s_sum[0] = scale;
    }
    __syncthreads();
    const float scale = s_sum[0];
    __syncthreads();  // s_sum is reused by the next call
    return scale;
}

// dst = f16((scale * x) * alpha), two roundings, left to right.
__device__ __forceinline__ unsigned short exact_rms_apply(float scale, unsigned short x, unsigned short alpha)
{
    return f2h(__fmul_rn(__fmul_rn(scale, h2f(x)), h2f(alpha)));
}

// tok_argmax.cu's key: ord16(v) << 32 | (1023 - bitrev10(i & 1023)) << 22 |
// (0x3FFFFF - (i >> 10)). The larger key is candle's winner; a NaN is refused
// by the caller.
__device__ __forceinline__ u32 pers_ord16(u32 b)
{
    b &= 0xFFFFu;
    if (b == 0x8000u) b = 0u;
    return (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ unsigned long long pers_tie_key(u32 bits, u32 i)
{
    return ((unsigned long long)pers_ord16(bits) << 32) | ((unsigned long long)(1023u - (__brev(i & 1023u) >> 22)) << 22)
           | (unsigned long long)(0x3FFFFFu - (i >> 10));
}

__device__ __forceinline__ bool pers_is_nan16(u32 b)
{
    return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u;
}

// ---------------------------------------------------------------------------
// 8. 4E's head phase: the tied int4 g64 head rows (tv_emb_q4_h's row, with the
//    row as a parameter), each row's argmax key, the CTA's maximum through a
//    shared 64-bit slot, one global atomicMax, the W4 barrier, then the token
//    every CTA reads back. The served path writes binary16 logits with
//    tv_emb_q4_h, casts and argmaxes them with candle; the key reproduces
//    candle's order on binary16 values (tok_argmax.cu, argmax_key.rs), and
//    the maximum of unique keys does not depend on the merge order, so the
//    token is the served token. A NaN logit raises the flag and the caller
//    falls back, as tok_argmax does.
// ---------------------------------------------------------------------------
#define PERS_HDR_KEY 4u  // u32 4..5 of the header: the CTA's 64-bit maximum

__device__ __forceinline__ float pers_q4g64_row(const u32* wq, const unsigned short* scales,
                                                const unsigned short* biases, const float* xs, u32 row, u32 d,
                                                u32 gpr, u32 lane)
{
    const u32 nwords = d >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        const u32 p = wq[w0 + wi];
        const u32 c = wi << 3;
        const u32 sb = scales[s0 + (c >> 6)];
        const u32 bb = biases[s0 + (c >> 6)];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    return warp_sum(acc);
}

// Rows [lo, hi) of the head on this CTA, `xs` the staged f32 hidden vector.
// Returns the token on every thread of every CTA once the W4 barrier is
// passed, or 0xffffffff when a NaN logit was seen or the wait aborted.
__device__ __forceinline__ u32 pers_head_argmax(const u32* wq, const unsigned short* scales,
                                                const unsigned short* biases, const float* xs, u32 d, u32 gpr,
                                                u32 lo, u32 hi, unsigned long long* slot, u32* nan_flag, u32* bar,
                                                u32 bar_target, u32* abort_flag, unsigned short* logits)
{
    u32* hdr = pers_hdr();
    unsigned long long* cta_max = reinterpret_cast<unsigned long long*>(hdr + PERS_HDR_KEY);
    const u32 lane = threadIdx.x & 31u;
    if (threadIdx.x == 0u) *cta_max = 0ull;
    pers_claim_reset(lo);
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        const float acc = pers_q4g64_row(wq, scales, biases, xs, r, d, gpr, lane);
        if (lane == 0u) {
            const unsigned short h = f2h(acc);
            if (logits) logits[r] = h;
            if (pers_is_nan16(h)) atomicExch(nan_flag, 1u);
            else atomicMax(cta_max, pers_tie_key(h, r));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) atomicMax(slot, *cta_max);
    if (!pers_grid_wait(bar, bar_target, abort_flag)) return 0xffffffffu;
    if (pers_ld_acquire(nan_flag) != 0u) return 0xffffffffu;
    const unsigned long long k = __ldcg(slot);
    const u32 br = 1023u - (u32)((k >> 22) & 0x3FFu);
    return ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);
}

// ---------------------------------------------------------------------------
// 9. 4E's per-layer phases around attention (persistent.md §3.9). Each writes
//    what the served path writes, bit for bit, and is reused by the token
//    loop: the embedding row, then ln1 + the q/k rotation + the q, k and v
//    rows, then (after attention) the o rows and the residual.
// ---------------------------------------------------------------------------

// emb_q4_gather's row (emb_q4.cu): column c of token `tok`'s int4 g64 row,
// rounded to binary16. Writes dst[c] for c in [lo, hi), any thread mapping.
__device__ __forceinline__ void pers_embed(const u32* wq, const unsigned short* scales, const unsigned short* biases,
                                           u32 tok, u32 d, u32 gpr, unsigned short* dst, u32 lo, u32 hi)
{
    const u32 w0 = tok * (d >> 3), s0 = tok * gpr;
    for (u32 c = lo + threadIdx.x; c < hi; c += blockDim.x) {
        const u32 q = (wq[w0 + (c >> 3)] >> ((c & 7u) * 4u)) & 0xfu;
        const u32 g = s0 + (c >> 6);
        dst[c] = f2h(pers_q4_deq(q, scales[g], biases[g]));
    }
}

// One Tetra row of a projection whose staged input is `xs` (natural, stride
// 24), through the epilogue, as a binary16.
__device__ __forceinline__ unsigned short pers_tetra_out(const u32* words, u32 stride_u32, u32 nblocks, u32 tail_w,
                                                         const float* rscale, const unsigned short* tail,
                                                         const float* gscale, const F1rTables& tab,
                                                         const float* invnorm, const float* xs, u32 r, u32 lane)
{
    const float acc = pers_tetra_row<24u, false, false>(words + r * stride_u32, tab, gscale, invnorm, xs, nblocks, lane);
    return f2h(pers_epilogue(acc, rscale[r], tail + r * tail_w, xs + nblocks * 24u, tail_w));
}

// The q/k/v phase on one CTA. `xr` holds ln1(x) rotated (q and k read it),
// `xn` ln1(x) widened in the natural basis (the int4 v reads it), both f32 in
// shared memory. Rows [lo, hi) of the concatenation q | k | v are this CTA's;
// q and k are Tetra under one rotation, v int4 g128, as at the 4B and 14B.
struct Pers4eQkv {
    const u32* qw; const float* qrs; const unsigned short* qtl; const float* qgs; u32 q_stride, q_dout;
    const u32* kw; const float* krs; const unsigned short* ktl; const float* kgs; u32 k_stride, k_dout;
    const u32* vw; const unsigned short* vsc; const unsigned short* vbi; u32 v_dout, v_gpr;
    u32 nblocks, tail_w, d_in;
    unsigned short *q, *k, *v;  // outputs, binary16
};

__device__ __forceinline__ void pers_4e_qkv_rows(const Pers4eQkv& P, const F1rTables& tab, const float* invnorm,
                                                 const float* xr, const float* xn, u32 lo, u32 hi)
{
    const u32 lane = threadIdx.x & 31u;
    pers_claim_reset(lo);
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        if (r < P.q_dout) {
            const unsigned short y = pers_tetra_out(P.qw, P.q_stride, P.nblocks, P.tail_w, P.qrs, P.qtl, P.qgs, tab,
                                                    invnorm, xr, r, lane);
            if (lane == 0u) P.q[r] = y;
        } else if (r < P.q_dout + P.k_dout) {
            const u32 rk = r - P.q_dout;
            const unsigned short y = pers_tetra_out(P.kw, P.k_stride, P.nblocks, P.tail_w, P.krs, P.ktl, P.kgs, tab,
                                                    invnorm, xr, rk, lane);
            if (lane == 0u) P.k[rk] = y;
        } else {
            const u32 rv = r - P.q_dout - P.k_dout;
            const float acc = pers_q4_row(P.vw, P.vsc, P.vbi, xn, rv, P.d_in, P.v_gpr, lane);
            if (lane == 0u) P.v[rv] = f2h(acc);
        }
    }
    __syncthreads();
}

// The o phase: int4 g128 rows over the attention output `ctx` (binary16,
// widened into `cs`), then out[r] = x[r] + o[r] as badd_f16 does.
__device__ __forceinline__ void pers_4e_o_int4(const u32* ow, const unsigned short* osc, const unsigned short* obi,
                                               u32 gpr, const unsigned short* ctx, float* cs, u32 d_ctx,
                                               const unsigned short* x, unsigned short* out, u32 lo, u32 hi)
{
    __syncthreads();
    for (u32 i = threadIdx.x; i < d_ctx; i += blockDim.x) cs[i] = h2f(__ldcg(ctx + i));
    const u32 lane = threadIdx.x & 31u;
    pers_claim_reset(lo);
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        const float acc = pers_q4_row(ow, osc, obi, cs, r, d_ctx, gpr, lane);
        if (lane == 0u) out[r] = exact_hadd(__ldcg(x + r), f2h(acc));
    }
    __syncthreads();
}

#endif  // LLVQ_PERS_CUH
