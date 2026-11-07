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

// The grid-wide clock of 4E's phase stamps (pers_4e.cu, STAMP = true): the
// `%globaltimer` register, nanoseconds, one clock for every SM. The host
// harness counts calls instead, which keeps the order of events and nothing
// else: a stamp there says what came before what, never how long.
__device__ __forceinline__ unsigned long long pers_now()
{
#ifdef LLVQ_HOST_BUILD
    static unsigned long long t = 0ull;
    return ++t;
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

// Ask L2 for the 128-byte lines of [p, p + bytes), spread over the CTA's
// threads: `prefetch.global.L2`, no register written, no value read, so no
// bit of any result can depend on it. 4E's CTAs issue it for the rows of a
// later phase while their own phase leaves the DRAM idle (route P's rule D,
// pers-l1-2026-11-04). The host harness has no cache: a no-op there.
__device__ __forceinline__ void pers_l2_prefetch(const void* p, unsigned long long bytes)
{
#ifdef LLVQ_HOST_BUILD
    (void)p;
    (void)bytes;
#else
    if (bytes == 0ull) return;
    const unsigned long long a0 = (unsigned long long)p & ~127ull, a1 = (unsigned long long)p + bytes;
    for (unsigned long long a = a0 + 128ull * threadIdx.x; a < a1; a += 128ull * blockDim.x)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
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

// PREFETCH (4E): word wi + 32 and its scale and bias are in flight while
// word wi is used; the same words, scales and products, accumulated in the
// same order, so the same bits.
template <bool PREFETCH = false>
__device__ __forceinline__ float pers_q4_row(const u32* wq, const unsigned short* scales,
                                             const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                             u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    if (!PREFETCH) {
        for (u32 wi = lane; wi < nwords; wi += 32u) {
            const u32 p = wq[w0 + wi];
            const u32 c = wi << 3;
            const u32 g = s0 + (c >> 7);
            const u32 sb = scales[g];
            const u32 bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    } else {
        u32 p = 0u, sb = 0u, bb = 0u;
        if (lane < nwords) {
            p = wq[w0 + lane];
            sb = scales[s0 + (lane >> 4)];
            bb = biases[s0 + (lane >> 4)];
        }
        for (u32 wi = lane; wi < nwords; wi += 32u) {
            u32 p2 = 0u, sb2 = 0u, bb2 = 0u;
            if (wi + 32u < nwords) {
                p2 = wq[w0 + wi + 32u];
                sb2 = scales[s0 + ((wi + 32u) >> 4)];
                bb2 = biases[s0 + ((wi + 32u) >> 4)];
            }
            const u32 c = wi << 3;
#pragma unroll
            for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
            p = p2;
            sb = sb2;
            bb = bb2;
        }
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

// Form 4 (4E v6): the same column at a compile-time K, `ssm` the k rows of
// `small` staged in shared memory. Every g is its own chain, unrolled, so the
// 19 chains of the 4B's down input interleave; each chain is pers_mix_column's:
// from +0, t ascending, then the single + 0 when K < KMAX.
template <u32 K>
__device__ __forceinline__ void pers_mix_column_k(float* s, const float* ssm, u32 m, float inv, u32 j)
{
    float col[K];
#pragma unroll
    for (u32 t = 0; t < K; ++t) col[t] = s[t * m + j] * inv;
#pragma unroll
    for (u32 g = 0; g < K; ++g) {
        float acc = 0.0f;
#pragma unroll
        for (u32 t = 0; t < K; ++t) acc = __fmaf_rn(ssm[g * LLVQ_ROT_KMAX + t], col[t], acc);
        if (K < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);
        s[g * m + j] = acc;
    }
}

// Form 5 (4E v7): pers_mix_column_k with each row of `small` read as float4
// (ssm 16-byte aligned, rows of KMAX = 32 floats): 5 loads a chain at K = 19
// instead of 19. Each chain is pers_mix_column_k's term for term: the same
// value of `small` multiplies the same col[t], t ascending, from +0, then the
// single + 0 when K < KMAX. Lanes past K in the last float4 are never used.
template <u32 K>
__device__ __forceinline__ void pers_mix_column_k4(float* s, const float* ssm, u32 m, float inv, u32 j)
{
    float col[K];
#pragma unroll
    for (u32 t = 0; t < K; ++t) col[t] = s[t * m + j] * inv;
#pragma unroll
    for (u32 g = 0; g < K; ++g) {
        const float4* r4 = reinterpret_cast<const float4*>(ssm + g * LLVQ_ROT_KMAX);
        float acc = 0.0f;
#pragma unroll
        for (u32 t4 = 0; t4 < (K + 3u) / 4u; ++t4) {
            const float4 w = r4[t4];
            acc = __fmaf_rn(w.x, col[4u * t4], acc);
            if (4u * t4 + 1u < K) acc = __fmaf_rn(w.y, col[4u * t4 + 1u], acc);
            if (4u * t4 + 2u < K) acc = __fmaf_rn(w.z, col[4u * t4 + 2u], acc);
            if (4u * t4 + 3u < K) acc = __fmaf_rn(w.w, col[4u * t4 + 3u], acc);
        }
        acc = K < LLVQ_ROT_KMAX ? __fadd_rn(acc, 0.0f) : acc;
        s[g * m + j] = acc;
    }
}

// The 4B's two odd factors (5 at 2,560, 19 at 9,728) unrolled; any other k
// runs pers_mix_column on the staged rows. V4: the float4 reads of form 5.
template <bool V4 = false>
__device__ __forceinline__ void pers_mix_1rt(float* s, const float* ssm, u32 m, u32 k, float inv, u32 j)
{
    if (V4 && k == 5u) pers_mix_column_k4<5u>(s, ssm, m, inv, j);
    else if (V4 && k == 19u) pers_mix_column_k4<19u>(s, ssm, m, inv, j);
    else if (k == 5u) pers_mix_column_k<5u>(s, ssm, m, inv, j);
    else if (k == 19u) pers_mix_column_k<19u>(s, ssm, m, inv, j);
    else pers_mix_column(s, ssm, m, k, inv, j);
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
// S1RT (form 4): `small` is the k rows staged in shared memory, and the mix
// runs pers_mix_1rt. V4 (form 5): with its float4 reads.
template <bool S1RT = false, bool V4 = false>
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
        for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
            if (S1RT) pers_mix_1rt<V4>(s, small, m, k, inv, j);
            else pers_mix_column(s, small, m, k, inv, j);
        }
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
// LD(col) gives candle's xi, the widened binary16 of column col.
template <int VARIANT, class LD>
__device__ __forceinline__ float exact_rms_scale_by(LD ld, u32 ncols, float eps)
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
            const float xi = ld(col);
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

template <int VARIANT>
__device__ float exact_rms_scale(const unsigned short* x, u32 ncols, float eps)
{
    return exact_rms_scale_by<VARIANT>([&](u32 col) { return h2f(__ldcg(x + col)); }, ncols, eps);
}

// dst = f16((scale * x) * alpha), two roundings, left to right.
__device__ __forceinline__ unsigned short exact_rms_apply(float scale, unsigned short x, unsigned short alpha)
{
    return f2h(__fmul_rn(__fmul_rn(scale, h2f(x)), h2f(alpha)));
}

// ---------------------------------------------------------------------------
// 7b. Form 4 (4E v6): the parts every CTA computes on the whole vector (ln1,
//     ln2, the down's input) with each thread's global loads issued before
//     their first use. The scalar loops above load one element and use it
//     before the next load issues: 15 serial round trips a ln, 38 a Tetra
//     down input, 19 an int4 one, each on lines all the grid's CTAs read at
//     once (docs/mesures/shaker-4e-rule-c-host-2026-11-05.txt). Here the
//     binary16 vectors come as 16-byte loads, with the sign words and the k
//     rows of `small` (staged in shared memory for the mix) in the same batch.
//     The values and the operations on them are the scalar path's.
// ---------------------------------------------------------------------------
// The 8 binary16 of a 16-byte chunk, in memory order.
__device__ __forceinline__ void pers_unpack8(const uint4 w, float f[8])
{
    f[0] = h2f((unsigned short)(w.x & 0xffffu));
    f[1] = h2f((unsigned short)(w.x >> 16));
    f[2] = h2f((unsigned short)(w.y & 0xffffu));
    f[3] = h2f((unsigned short)(w.y >> 16));
    f[4] = h2f((unsigned short)(w.z & 0xffffu));
    f[5] = h2f((unsigned short)(w.z >> 16));
    f[6] = h2f((unsigned short)(w.w & 0xffffu));
    f[7] = h2f((unsigned short)(w.w >> 16));
}

#define PERS_1RT_V 3u  // vectors a thread a batch: the 4B's 1,216 at 9,728 in one batch at 512 threads

// The vector path's conditions: p 16-byte aligned, n a multiple of 8. The
// callers keep the scalar path otherwise.
__device__ __forceinline__ bool pers_1rt_ok(const void* p, u32 n)
{
    return ((reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull) && ((n & 7u) == 0u);
}

// x widened, as rms_apply reads it after h2f: the same value.
__device__ __forceinline__ unsigned short exact_rms_apply_f(float scale, float xf, float af)
{
    return f2h(__fmul_rn(__fmul_rn(scale, xf), af));
}

// The k rows of `small` (k x KMAX floats): two a thread loaded by
// pers_small_load, stored into ssm by pers_small_store once the caller's
// other loads are issued; a CTA under 512 threads loops for the rest.
struct PersSmall {
    float a0, a1;
};
__device__ __forceinline__ PersSmall pers_small_load(const float* small, u32 k)
{
    const u32 ns = k * LLVQ_ROT_KMAX, i0 = threadIdx.x, i1 = threadIdx.x + blockDim.x;
    PersSmall r{0.0f, 0.0f};
    if (small != nullptr && i0 < ns) r.a0 = small[i0];
    if (small != nullptr && i1 < ns) r.a1 = small[i1];
    return r;
}
__device__ __forceinline__ void pers_small_store(PersSmall r, const float* small, u32 k, float* ssm)
{
    if (small == nullptr) return;
    const u32 ns = k * LLVQ_ROT_KMAX, i0 = threadIdx.x, i1 = threadIdx.x + blockDim.x;
    if (i0 < ns) ssm[i0] = r.a0;
    if (i1 < ns) ssm[i1] = r.a1;
    for (u32 i = threadIdx.x + 2u * blockDim.x; i < ns; i += blockDim.x) ssm[i] = small[i];
}

// pers_rot_load (signbits non-null), or the int4 widening (null), of n
// binary16 values into s; with `small` non-null its k rows go to ssm in the
// same batch. Vector q of a batch is thread (q - q0) mod B's. No barrier:
// pers_rot_finish starts with one, the widening's caller adds one.
__device__ __forceinline__ void pers_load_1rt(const unsigned short* xin, const u32* signbits, float* s, u32 n,
                                              const float* small, u32 k, float* ssm)
{
    const uint4* xv = reinterpret_cast<const uint4*>(xin);
    const u32 nv = n >> 3, B = blockDim.x;
    const PersSmall sm = pers_small_load(small, k);
    for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
        uint4 w[PERS_1RT_V];
        u32 sw[PERS_1RT_V];
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            sw[u] = 0u;
            if (q < nv) {
                w[u] = __ldcg(xv + q);
                if (signbits != nullptr) sw[u] = signbits[q >> 2];
            }
        }
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            if (q < nv) {
                float f[8];
                pers_unpack8(w[u], f);
                const u32 bits = sw[u] >> ((q & 3u) * 8u);
#pragma unroll
                for (u32 e = 0; e < 8u; ++e) s[8u * q + e] = ((bits >> e) & 1u) ? -f[e] : f[e];
            }
        }
    }
    pers_small_store(sm, small, k, ssm);
}

// ln1 or ln2: x and alpha as vectors and the k rows of `small` in one batch;
// x widened into xs (n floats of shared memory), alpha widened into out;
// candle's scale replayed from xs; then, by the thread that loaded them,
// out[i] = the signed widened binary16 of the rmsnorm and, when `nat`,
// xs[i] = the unsigned one (ln1's natural copy for the int4 v rows). The
// sign words are read again after the scale, from L1.
template <int VARIANT>
__device__ __forceinline__ void pers_ln_1rt(const unsigned short* x, const unsigned short* alpha, const u32* signbits,
                                            u32 n, float eps, const float* small, u32 k, float* ssm, float* out,
                                            float* xs, bool nat)
{
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* av = reinterpret_cast<const uint4*>(alpha);
    const u32 nv = n >> 3, B = blockDim.x;
    const PersSmall sm = pers_small_load(small, k);
    for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
        uint4 w[PERS_1RT_V], a[PERS_1RT_V];
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            if (q < nv) {
                w[u] = __ldcg(xv + q);
                a[u] = av[q];
            }
        }
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            if (q < nv) {
                float f[8], g[8];
                pers_unpack8(w[u], f);
                pers_unpack8(a[u], g);
#pragma unroll
                for (u32 e = 0; e < 8u; ++e) {
                    xs[8u * q + e] = f[e];
                    out[8u * q + e] = g[e];
                }
            }
        }
    }
    pers_small_store(sm, small, k, ssm);
    const float scale = exact_rms_scale_by<VARIANT>([&](u32 col) { return xs[col]; }, n, eps);
    for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
        u32 sw[PERS_1RT_V];
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            sw[u] = q < nv ? signbits[q >> 2] : 0u;
        }
#pragma unroll
        for (u32 u = 0; u < PERS_1RT_V; ++u) {
            const u32 q = q0 + threadIdx.x + u * B;
            if (q < nv) {
                const u32 bits = sw[u] >> ((q & 3u) * 8u);
#pragma unroll
                for (u32 e = 0; e < 8u; ++e) {
                    const u32 i = 8u * q + e;
                    const float hv = h2f(exact_rms_apply_f(scale, xs[i], out[i]));
                    if (nat) xs[i] = hv;
                    out[i] = ((bits >> e) & 1u) ? -hv : hv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7c. Form 5 (4E v7): what a CTA reads right after a grid barrier, read
//     during it. After every barrier v6 still reads, from L2 and on lines all
//     142 CTAs ask for at once: the fields of the layer's argument struct
//     (pointers and sizes, a dependent load before the first data load can
//     issue), the CTA's row slices (a second dependent load), the rotation's
//     sign words (ln reads them again after the scale) and the k rows of
//     `small`. Each is a constant of the launch. Form 5 copies them into the
//     CTA's shared memory between its arrival at the previous barrier and its
//     release (pers_grid_wait_gap), so after the release the prologue loads
//     only the activation. Where a value is read from changes, never which
//     value or what is done with it: every bit is form 4's.
//
//     The area sits after the larger of the row vector and attention's stage,
//     so attention never overwrites it; its offset from `v`, in floats, is in
//     header u32 PERS_HDR_V7 (the MLP segment does not know the head groups).
// ---------------------------------------------------------------------------
#define PERS_HDR_V7 8u           // header u32 8: the v7 area's offset from v, in floats
#define PERS_V7_SMALL 0u         // the k rows of `small`, KMAX x KMAX floats
#define PERS_V7_ARGS 1024u       // the launch's Pers4eArgs, 256 bytes
#define PERS_V7_LAYER 1088u      // the current layer's Pers4eLayer, 640 bytes
#define PERS_V7_SLICE 1248u      // the CTA's gate/up and down slices: lo, hi, lo, hi
#define PERS_V7_SIGNS 1264u      // a rotation's sign words
#define PERS_V7_SIGNS_MAX 576u   // up to width 18,432: the 14B's inter is 17,408
#define PERS_V7_FLOATS 1840u     // 7,360 bytes, PERS_4E_SCRATCH of form 5

// pers_grid_wait with STAGE run by every thread between thread 0's arrival
// and its spin. STAGE may only copy constants of the launch (nothing a CTA
// writes during it), so no fence is owed for them: the arrival's ordering is
// pers_grid_wait's, and the closing barrier makes the copies visible to the
// whole CTA, as it makes the verdict.
template <class Stage>
__device__ __forceinline__ bool pers_grid_wait_gap(u32* bar, u32 target, u32* abort_flag, Stage stage)
{
    u32* hdr = pers_hdr();
    __syncthreads();  // the CTA's stores precede thread 0's arrival
    if (threadIdx.x == 0u) {
        __threadfence();
        atomicAdd(bar, 1u);
    }
    stage();
    if (threadIdx.x == 0u) {
        const int ok = pers_spin(bar, target, abort_flag);
        __threadfence();
        hdr[PERS_HDR_OK] = (u32)ok;
    }
    __syncthreads();
    return hdr[PERS_HDR_OK] != 0u;
}

// n words of a constant, spread over the CTA's threads. No barrier.
__device__ __forceinline__ void pers_copy_u32(u32* dst, const u32* src, u32 n)
{
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// A rotation's sign words (width / 32) and its k rows of `small` (k x KMAX
// floats), into the v7 area. No barrier.
__device__ __forceinline__ void pers_v7_stage_rot(float* area, const u32* signbits, u32 width, const float* small,
                                                  u32 k)
{
    pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SIGNS), signbits, width >> 5);
    pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SMALL), reinterpret_cast<const u32*>(small),
                  k * LLVQ_ROT_KMAX);
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

// PREFETCH (4E's 512-thread symbols): word wi + 32 in flight while word wi
// is used (pers_q4_row's PREFETCH), the same products in the same order.
template <bool PREFETCH = true>
__device__ __forceinline__ float pers_q4g64_row(const u32* wq, const unsigned short* scales,
                                                const unsigned short* biases, const float* xs, u32 row, u32 d,
                                                u32 gpr, u32 lane)
{
    const u32 nwords = d >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    if (!PREFETCH) {
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
    u32 p = 0u, sb = 0u, bb = 0u;
    if (lane < nwords) {
        p = wq[w0 + lane];
        sb = scales[s0 + (lane >> 3)];
        bb = biases[s0 + (lane >> 3)];
    }
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p2 = 0u, sb2 = 0u, bb2 = 0u;
        if (wi + 32u < nwords) {
            p2 = wq[w0 + wi + 32u];
            sb2 = scales[s0 + ((wi + 32u) >> 3)];
            bb2 = biases[s0 + ((wi + 32u) >> 3)];
        }
        const u32 c = wi << 3;
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        p = p2;
        sb = sb2;
        bb = bb2;
    }
    return warp_sum(acc);
}

// Rows [lo, hi) of the head on this CTA, `xs` the staged f32 hidden vector.
// Returns the token on every thread of every CTA once the W4 barrier is
// passed, or 0xffffffff when a NaN logit was seen or the wait aborted.
template <bool PF = true>
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
        const float acc = pers_q4g64_row<PF>(wq, scales, biases, xs, r, d, gpr, lane);
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
template <bool PF = true>
__device__ __forceinline__ unsigned short pers_tetra_out(const u32* words, u32 stride_u32, u32 nblocks, u32 tail_w,
                                                         const float* rscale, const unsigned short* tail,
                                                         const float* gscale, const F1rTables& tab,
                                                         const float* invnorm, const float* xs, u32 r, u32 lane)
{
    const float acc = pers_tetra_row<24u, false, PF>(words + r * stride_u32, tab, gscale, invnorm, xs, nblocks, lane);
    return f2h(pers_epilogue(acc, rscale[r], tail + r * tail_w, xs + nblocks * 24u, tail_w));
}

// The Tetra row stream (4E's 512-thread symbols). The card's phase table of
// 4E v2 (pers-4e-v2-2026-11-05) puts a d_in 2,560 row at about 0.69 us an
// iteration and 3.9 us of fixed cost (*estimated* from its q/k/v and MLP
// phases): the row's first words, then lane 0's rscale and tail, and in
// gate/up the SiLU table, each a dependent round trip once the blocks are
// summed. Here a warp runs a static list of rows, nrows of them, and the
// lane's blocks of the whole list form one sequence: block j = lane + 32 i
// of row k is item k x nit + i, nit = ceil(nblocks / 32) for every lane, and
// the items D ahead are always in flight, across row boundaries. Lane 0
// loads a row's rscale and tail (and START's extras) when the row's first
// item is consumed, nit - 1 items before its epilogue. Per lane and per row
// the blocks are summed in the served order (j increasing), then warp_sum,
// then sites 2 and 3 as pers_epilogue spells them: the same bits. A lane
// whose block j is past the row computes nothing; the item count is the
// same on every lane, so warp_sum stays convergent.
//
// ROW(k) gives row k's words, the matrix's gscale, rscale and tail bases and
// its row index; START(k) runs on lane 0 at the row's first item (extra
// loads); EPI(k, y) on lane 0 with the row's f32 value before f2h. The tail
// is read as 16-byte chunks: tail_w a multiple of 8, at most 16, and each
// row's tail 16-byte aligned (the host checks the shape).
#ifndef PERS_STREAM_D
#define PERS_STREAM_D 4u  // items in flight a lane
#endif
struct PersTRow {
    const u32* w;             // the row's words
    const float* gscale;      // the matrix's two centroid scales
    const float* rscale;      // the matrix's row scales
    const unsigned short* tail;  // the matrix's tails, tail_w a row
    u32 r;                    // the row in its matrix
};

template <u32 D, class Row, class Start, class Epi>
__device__ __forceinline__ void pers_tetra_stream(u32 nrows, u32 nblocks, u32 tail_w, const F1rTables& tab,
                                                  const float* invnorm, const float* xs, u32 lane, Row row_of,
                                                  Start start, Epi epi)
{
    const u32 nit = (nblocks + 31u) >> 5;
    // (k, i): the item consumed, row k, iteration i; (k2, i2): the next item
    // to load, w2 its row's words. Both advance by counting, never dividing:
    // an integer division an item cost about 160 instructions a block more
    // than the plain row loop (devel SASS of the first draft).
    u32 k = 0u, i = 0u, k2 = 0u, i2 = 0u;
    const u32* w2 = nrows ? row_of(0u).w : nullptr;
    u32 lo[D], hi[D];
#pragma unroll
    for (u32 d = 0; d < D; ++d) {
        lo[d] = 0u;
        hi[d] = 0u;
        if (k2 < nrows) {
            const u32 j2 = lane + 32u * i2;
            if (j2 < nblocks) f1r_load(w2, j2, lo[d], hi[d]);
            if (++i2 == nit) {
                i2 = 0u;
                if (++k2 < nrows) w2 = row_of(k2).w;
            }
        }
    }
    const float* gs = nullptr;  // the consumed row's matrix scales
    float acc = 0.0f, rs = 0.0f;
    uint4 t0, t1;
    t0.x = t0.y = t0.z = t0.w = 0u;
    t1 = t0;
    while (k < nrows) {  // uniform over the warp
#pragma unroll
        for (u32 d = 0; d < D; ++d) {
            if (k < nrows) {
                if (i == 0u) {
                    const PersTRow R = row_of(k);
                    gs = R.gscale;
                    if (lane == 0u) {
                        rs = R.rscale[R.r];
                        const uint4* tp = reinterpret_cast<const uint4*>(R.tail + (unsigned long long)R.r * tail_w);
                        t0 = tp[0];
                        if (tail_w > 8u) t1 = tp[1];
                        start(k);
                    }
                }
                const u32 j = lane + 32u * i;
                if (j < nblocks) acc = pers_block(acc, lo[d], hi[d], tab, xs + j * 24u, gs, invnorm);
                lo[d] = 0u;
                hi[d] = 0u;
                if (k2 < nrows) {
                    const u32 j2 = lane + 32u * i2;
                    if (j2 < nblocks) f1r_load(w2, j2, lo[d], hi[d]);
                    if (++i2 == nit) {
                        i2 = 0u;
                        if (++k2 < nrows) w2 = row_of(k2).w;
                    }
                }
                if (i == nit - 1u) {
                    const float a = warp_sum(acc);
                    acc = 0.0f;
                    if (lane == 0u) {
                        // pers_epilogue: site 2, the tail in order, then site 3
                        const u32 tw[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
                        const float* xt = xs + nblocks * 24u;
                        float tv = 0.0f;
#pragma unroll
                        for (u32 q = 0; q < 16u; ++q)
                            if (q < tail_w) tv = __fmaf_rn(h2f((unsigned short)(tw[q >> 1] >> (16u * (q & 1u)))), xt[q], tv);
                        epi(k, __fmaf_rn(a, rs, tv));
                    }
                    i = 0u;
                    ++k;
                } else {
                    ++i;
                }
            }
        }
    }
}

// The row form of 4E's 512-thread symbols, chosen when the module is built
// (persist4e_cuda.rs prepends the define): 1, the stream above (4E v3, the
// default and what every served launch has run); 2, 4E v4, v2's row loop
// (pers_tetra_row with the next block's words in flight) on the same static
// rows, row by row. The row-loop card reading timed v2's loop alone at x1.18
// over the stream's (pers-rowloop-2026-11-05). 3, 4E v5: form 2, with the
// CTA's last rows (its row count modulo the warps) split block by block over
// every warp (pers_4e_rows_cta). 4, 4E v6: form 2's rows, with ln1, ln2 and
// the down's input in one round trip a thread (section 7b). 5, 4E v7: form 4
// with the launch's constants read during the barriers (section 7c) and the o
// phase's input widened from vectors. 6, 4E v8: form 5 with attention's
// steps in a cp.async ring. 7, 4E v9: form 6 with the int4 rows on tiled
// words. All keep every bit.
#ifndef PERS_4E_ROWS
#define PERS_4E_ROWS 1u
#endif
// Form 3's shared scratch, after the activation staging: the split rows'
// fold inputs, and gate/up's binary16 values before their product.
#if PERS_4E_ROWS == 3
#define PERS_4E_SCRATCH 16384u
#elif PERS_4E_ROWS == 4
#define PERS_4E_SCRATCH 4096u  // the k rows of `small` staged by section 7b
#elif PERS_4E_ROWS >= 5
#define PERS_4E_SCRATCH 7360u  // the v7 area of section 7c, PERS_V7_FLOATS x 4
#else
#define PERS_4E_SCRATCH 0u
#endif

// Form 7 (4E v9, K11 of docs/plan-300.md): v8, with 4E's int4 rows (v, o and
// the int4 downs) read from words relaid once at install, so one 128-bit
// load gives lane l its words 128 t + l + 32 s, s = 0..3 (step1-micro's q4t,
// x1.338 over pers_q4_row alone). Lane l still folds its words l, l + 32, ...
// in increasing order, eight products each in increasing k, then the same
// warp_sum: pers_q4_row's bits. The words past the last full tile of 128 keep
// the served order. pers_q4_tile_index is the layout and
// pers_4e_q4_tile_relayout (pers_4e.cu) writes it; nwords must be a multiple
// of 4 and every row 16-byte aligned (persist4e_cuda.rs checks both).
__device__ __forceinline__ u32 pers_q4_tile_index(u32 wi, u32 nwords)
{
    const u32 nt = nwords >> 7;
    if (wi >= (nt << 7)) return wi;
    const u32 t = wi >> 7, k = wi & 127u;
    return (t << 7) + 4u * (k & 31u) + (k >> 5);
}

__device__ __forceinline__ float pers_q4_row_tile(const u32* wq, const unsigned short* scales,
                                                  const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                                  u32 gpr, u32 lane)
{
    const u32 nwords = d_in >> 3, nt = nwords >> 7;
    const u32* wrow = wq + (size_t)row * nwords;
    const u32 s0 = row * gpr;
    float acc = 0.0f;
    uint4 P{0u, 0u, 0u, 0u};
    if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * lane);
    for (u32 t = 0; t < nt; ++t) {
        uint4 P2{0u, 0u, 0u, 0u};
        if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + ((t + 1u) << 7) + 4u * lane);
        const u32 w[4] = {P.x, P.y, P.z, P.w};
#pragma unroll
        for (u32 s = 0; s < 4u; ++s) {
            const u32 c = ((t << 7) + 32u * s + lane) << 3;
            const u32 g = (c >> 7) + s0;  // the group of word c / 8
            const u32 sb = scales[g], bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((w[s] >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
        P = P2;
    }
    for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {
        const u32 p = wrow[wi];
        const u32 c = wi << 3;
        const u32 g = (c >> 7) + s0;  // the group of word c / 8
        const u32 sb = scales[g], bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) acc = __fmaf_rn(pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
    }
    return warp_sum(acc);
}

// 4E's int4 row: form 7's tiled words, else pers_q4_row on the served words.
template <bool PF>
__device__ __forceinline__ float pers_q4_row_4e(const u32* wq, const unsigned short* scales,
                                                const unsigned short* biases, const float* xs, u32 row, u32 d_in,
                                                u32 gpr, u32 lane)
{
#if PERS_4E_ROWS >= 7
    return pers_q4_row_tile(wq, scales, biases, xs, row, d_in, gpr, lane);
#else
    return pers_q4_row<PF>(wq, scales, biases, xs, row, d_in, gpr, lane);
#endif
}

// Form 2: the stream's interface, one row after the other. Lane 0 reads the
// row's rscale and runs START before the blocks, so their loads are in flight
// while the row decodes; the epilogue is pers_epilogue's, the stream's own.
template <class Row, class Start, class Epi>
__device__ __forceinline__ void pers_tetra_rows_static(u32 nrows, u32 nblocks, u32 tail_w, const F1rTables& tab,
                                                       const float* invnorm, const float* xs, u32 lane, Row row_of,
                                                       Start start, Epi epi)
{
    for (u32 k = 0; k < nrows; ++k) {
        const PersTRow R = row_of(k);
        float rs = 0.0f;
        if (lane == 0u) {
            rs = R.rscale[R.r];
            start(k);
        }
        const float a = pers_tetra_row<24u, false, true>(R.w, tab, R.gscale, invnorm, xs, nblocks, lane);
        if (lane == 0u) epi(k, pers_epilogue(a, rs, R.tail + (unsigned long long)R.r * tail_w, xs + nblocks * 24u, tail_w));
    }
}

// Form 3: the rows of one CTA, idx in [0, n), every warp of it. The full
// rounds (n rounded down to the warps) run as form 2, row idx on warp idx mod
// nwarps. The rest, at most nwarps - 1 rows, is split into (row, lane step)
// units dealt to every warp: a unit decodes block j = lane + 32 i of its row
// and keeps pers_fold's two factors, t = a x gscale[g] and invnorm[m], in the
// scratch. After a CTA barrier the row's owner (warp row mod nwarps) folds
// them in i order, fma(t, inv, acc) from 0 on the blocks of the row, which is
// pers_tetra_row's sum term for term, then warp_sum and the epilogue. When
// the units do not fit the scratch, the rest runs as form 2 too.
// ROW_AT(idx) gives the row; EPI(idx, y) on lane 0 with the f32 value.
template <class RowAt, class Epi>
__device__ __forceinline__ void pers_4e_rows_cta(u32 n, u32 nblocks, u32 tail_w, const F1rTables& tab,
                                                 const float* invnorm, const float* xs, RowAt row_at, Epi epi,
                                                 float* scratch, u32 scratch_floats)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 nit = (nblocks + 31u) >> 5;
    u32 full = n / nw * nw;
    u32 rem = n - full;
    if (2u * 32u * rem * nit > scratch_floats) {
        full = n;
        rem = 0u;
    }
    const auto whole = [&](u32 idx) {
        const PersTRow R = row_at(idx);
        const float a = pers_tetra_row<24u, false, true>(R.w, tab, R.gscale, invnorm, xs, nblocks, lane);
        if (lane == 0u) epi(idx, pers_epilogue(a, R.rscale[R.r], R.tail + (unsigned long long)R.r * tail_w,
                                               xs + nblocks * 24u, tail_w));
    };
    for (u32 idx = warp; idx < full; idx += nw) whole(idx);
    if (rem == 0u) return;
    float* st = scratch;                    // t, [unit][lane]
    float* sv = scratch + 32u * rem * nit;  // invnorm[m], [unit][lane]
    const u32 units = rem * nit;
    for (u32 u = warp; u < units; u += nw) {
        const u32 rr = u / nit, i = u - rr * nit;
        const PersTRow R = row_at(full + rr);
        const u32 j = lane + 32u * i;
        float tv = 0.0f, iv = 0.0f;
        if (j < nblocks) {
            u32 lo, hi16, q[6];
            f1r_load(R.w, j, lo, hi16);
            f1r_v3_quads(lo, hi16, tab, q);
            const float* xb = xs + j * 24u;
            float a = 0.0f;
#pragma unroll
            for (u32 b = 0; b < 6u; ++b) {
#pragma unroll
                for (u32 k = 0; k < 4u; ++k) a = __fmaf_rn(f1r_v3_float(q[b], k), xb[TETRA48_ORDER[4u * b + k]], a);
            }
            const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
            tv = __fmul_rn(a, R.gscale[(hi16 >> 15) & 1u]);
            iv = invnorm[m];
        }
        st[u * 32u + lane] = tv;
        sv[u * 32u + lane] = iv;
    }
    __syncthreads();
    for (u32 rr = warp; rr < rem; rr += nw) {
        float acc = 0.0f;
        for (u32 i = 0; i < nit; ++i) {
            const u32 u = rr * nit + i;
            if (lane + 32u * i < nblocks) acc = __fmaf_rn(st[u * 32u + lane], sv[u * 32u + lane], acc);
        }
        acc = warp_sum(acc);
        if (lane == 0u) {
            const PersTRow R = row_at(full + rr);
            epi(full + rr, pers_epilogue(acc, R.rscale[R.r], R.tail + (unsigned long long)R.r * tail_w,
                                         xs + nblocks * 24u, tail_w));
        }
    }
}

// 4E's row driver: the form the module was built with.
template <class Row, class Start, class Epi>
__device__ __forceinline__ void pers_4e_rows(u32 nrows, u32 nblocks, u32 tail_w, const F1rTables& tab,
                                             const float* invnorm, const float* xs, u32 lane, Row row_of, Start start,
                                             Epi epi)
{
#if PERS_4E_ROWS >= 2
    pers_tetra_rows_static(nrows, nblocks, tail_w, tab, invnorm, xs, lane, row_of, start, epi);
#else
    pers_tetra_stream<PERS_STREAM_D>(nrows, nblocks, tail_w, tab, invnorm, xs, lane, row_of, start, epi);
#endif
}

// Rows a warp takes from a CTA's slice [lo, hi) under the stream's static
// assignment: lo + warp, lo + warp + nwarps, ...
__device__ __forceinline__ u32 pers_warp_rows(u32 lo, u32 hi, u32 warp, u32 nwarps)
{
    return hi > lo + warp ? (hi - lo - warp + nwarps - 1u) / nwarps : 0u;
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

// CTA c of g. PF (the stream): the CTA takes its share of the q | k rows,
// [qk c / g, qk (c + 1) / g), and its share of the v rows, so every CTA holds
// the same mix of Tetra and int4 rows (a contiguous slice of q | k | v left
// the int4 v rows to the last CTAs, the slowest once the Tetra rows stream).
// Otherwise the contiguous slice [n c / g, n (c + 1) / g) of q | k | v.
template <bool PF = true>
__device__ __forceinline__ void pers_4e_qkv_rows(const Pers4eQkv& P, const F1rTables& tab, const float* invnorm,
                                                 const float* xr, const float* xn, u32 c, u32 g,
                                                 float* scratch = nullptr)
{
    const u32 lane = threadIdx.x & 31u;
    if (PF) {
        // The CTA's q | k rows as one list, then the int4 v rows by warp.
        const u32 warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
        const u32 qk = P.q_dout + P.k_dout;
        const u32 lo = qk * c / g, thi = qk * (c + 1u) / g;
        const auto qk_row = [&](u32 r) {
            return r < P.q_dout ? PersTRow{P.qw + r * P.q_stride, P.qgs, P.qrs, P.qtl, r}
                                : PersTRow{P.kw + (r - P.q_dout) * P.k_stride, P.kgs, P.krs, P.ktl, r - P.q_dout};
        };
        const auto qk_out = [&](u32 r, float y) {
            if (r < P.q_dout) P.q[r] = f2h(y);
            else P.k[r - P.q_dout] = f2h(y);
        };
#if PERS_4E_ROWS == 3
        // form 3: every warp of the CTA on its list, the last rows split
        const auto row_at = [&](u32 idx) { return qk_row(lo + idx); };
        const auto epi = [&](u32 idx, float y) { qk_out(lo + idx, y); };
        pers_4e_rows_cta(thi - lo, P.nblocks, P.tail_w, tab, invnorm, xr, row_at, epi, scratch, PERS_4E_SCRATCH / 4u);
#else
        (void)scratch;
        const auto row_of = [&](u32 k) { return qk_row(lo + warp + k * nw); };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) { qk_out(lo + warp + k * nw, y); };
        pers_4e_rows(pers_warp_rows(lo, thi, warp, nw), P.nblocks, P.tail_w, tab, invnorm, xr, lane, row_of, start,
                     epi);
#endif
        const u32 vlo = P.v_dout * c / g, vhi = P.v_dout * (c + 1u) / g;
        for (u32 rv = vlo + warp; rv < vhi; rv += nw) {
            const float acc = pers_q4_row_4e<true>(P.vw, P.vsc, P.vbi, xn, rv, P.d_in, P.v_gpr, lane);
            if (lane == 0u) P.v[rv] = f2h(acc);
        }
        __syncthreads();
        return;
    }
#if PERS_4E_ROWS != 3
    (void)scratch;
#endif
    const u32 n = P.q_dout + P.k_dout + P.v_dout;
    const u32 lo = n * c / g, hi = n * (c + 1u) / g;
    pers_claim_reset(lo);
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        if (r < P.q_dout) {
            const unsigned short y = pers_tetra_out<PF>(P.qw, P.q_stride, P.nblocks, P.tail_w, P.qrs, P.qtl, P.qgs, tab,
                                                        invnorm, xr, r, lane);
            if (lane == 0u) P.q[r] = y;
        } else if (r < P.q_dout + P.k_dout) {
            const u32 rk = r - P.q_dout;
            const unsigned short y = pers_tetra_out<PF>(P.kw, P.k_stride, P.nblocks, P.tail_w, P.krs, P.ktl, P.kgs, tab,
                                                        invnorm, xr, rk, lane);
            if (lane == 0u) P.k[rk] = y;
        } else {
            const u32 rv = r - P.q_dout - P.k_dout;
            const float acc = pers_q4_row_4e<PF>(P.vw, P.vsc, P.vbi, xn, rv, P.d_in, P.v_gpr, lane);
            if (lane == 0u) P.v[rv] = f2h(acc);
        }
    }
    __syncthreads();
}

// The o phase: int4 g128 rows over the attention output `ctx` (binary16,
// widened into `cs`), then out[r] = x[r] + o[r] as badd_f16 does.
template <bool PF = true>
__device__ __forceinline__ void pers_4e_o_int4(const u32* ow, const unsigned short* osc, const unsigned short* obi,
                                               u32 gpr, const unsigned short* ctx, float* cs, u32 d_ctx,
                                               const unsigned short* x, unsigned short* out, u32 lo, u32 hi)
{
    __syncthreads();
#if PERS_4E_ROWS >= 5
    // form 5: ctx widened from 16-byte vectors, all issued before the first
    // use (the scalar loop is d_ctx / 512 serial round trips), and lane 0
    // asks for x[r] before the row's blocks. The same values, the same rows.
    if (pers_1rt_ok(ctx, d_ctx)) {
        pers_load_1rt(ctx, nullptr, cs, d_ctx, nullptr, 0u, nullptr);
    } else
#endif
    for (u32 i = threadIdx.x; i < d_ctx; i += blockDim.x) cs[i] = h2f(__ldcg(ctx + i));
    const u32 lane = threadIdx.x & 31u;
    pers_claim_reset(lo);  // its opening barrier orders the widening before every row
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
#if PERS_4E_ROWS >= 5
        const unsigned short xr = lane == 0u ? __ldcg(x + r) : (unsigned short)0u;
        const float acc = pers_q4_row_4e<PF>(ow, osc, obi, cs, r, d_ctx, gpr, lane);
        if (lane == 0u) out[r] = exact_hadd(xr, f2h(acc));
#else
        const float acc = pers_q4_row_4e<PF>(ow, osc, obi, cs, r, d_ctx, gpr, lane);
        if (lane == 0u) out[r] = exact_hadd(__ldcg(x + r), f2h(acc));
#endif
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 10. 4E's attention phase: one decode row, the canonical arithmetic of
//     llvq-cuda/src/attn_ref.rs (and of attn_dec.cu, proven equal to it by
//     the attention lead), re-mapped onto the persistent grid. The bits of a
//     row depend on its length alone: 8-position tiles aligned to absolute
//     positions, 64-position units, 256-position splits, units folded into
//     their split left to right, splits into the row left to right, every
//     float operation an explicit RN intrinsic. Under T8: identity is owed to
//     the "kernel" attention form, not to candle's cuBLAS path.
//
//     The mapping is attn_dec.cu's, on a CTA of the persistent grid: a job
//     is one (split, KV head), job j on CTA j mod grid. At step st the CTA
//     stages tile st of the split's four units, K then V, 32 positions, 16 KB,
//     into shared memory, once for every query head of the KV head, with the
//     next step's loads already in flight in registers. A task is one (unit,
//     pair of query heads), one warp, one head a half-warp, so the 16-lane
//     butterfly (xor 8, 4, 2, 1) stays inside each half and every shuffle is
//     a full-warp one. A task's state (m, d, a[128]) lives in shared memory
//     between steps, so any number of warps covers the tasks. After the steps
//     the units fold into the split state, one thread a (head, dimension).
//     A row of one split divides and writes; otherwise the split state goes
//     to global scratch and the CTA draws a ticket on its KV head's counter
//     (attn_dec.cu's occ_sk pattern): the last one folds the splits in order,
//     8 a round trip, writes the output and resets the counter. No grid
//     barrier inside the phase: the one after it orders the output.
//
//     The first version gave one warp each (query head, unit), walked the
//     unit's positions through dependent 2-byte loads, and folded the units
//     of a head serially after a grid barrier: 2,355 us a token at a 5-token
//     prompt and 13,273 us at 8,192 on the card (pers-4e-r3-2026-11-05).
// ---------------------------------------------------------------------------
#define PERS_HD 128u
#define PERS_TT 8u
#define PERS_UNIT 64u
#define PERS_SPLIT 256u
#define PERS_UPS 4u                                       // units a split
#define PERS_PSTRIDE 130u
#define PERS_STAGE (PERS_UPS * PERS_TT)                   // positions a step stages: 32
#define PERS_STAGE_U4 (2u * PERS_STAGE * (PERS_HD / 8u))  // its 16-byte chunks, K then V: 1,024
#define PERS_ATTN_CH 8u                                   // splits a round trip of the split fold
#define PERS_NEG_INF __int_as_float((int)0xff800000u)
#define PERS_HDR_TICKET 6u                                // header u32 6: this CTA drew the last ticket

// Form 6 (4E v8, K3a of docs/plan-300.md): the steps staged by cp.async into
// a ring of PERS_ATTN_RING stages, so up to three steps' loads are in flight
// while a step's tiles run. Below form 6, one stage filled from registers.
#if PERS_4E_ROWS >= 6
#define PERS_ATTN_RING 3u
#else
#define PERS_ATTN_RING 1u
#endif

// Shared bytes the phase needs at `groups` query heads a KV head: the
// stages, the unit states [UPS][groups][PSTRIDE], q [groups][128] widened.
#define PERS_ATTN_SMEM(groups) (16u * PERS_STAGE_U4 * PERS_ATTN_RING + 4u * PERS_UPS * (groups) * PERS_PSTRIDE + 4u * (groups) * PERS_HD)

// attn_ref.rs::ex2_fixed / attn_dec.cu::attn_ex2, the fixed polynomial.
__device__ __forceinline__ float pers_ex2(float x)
{
    if (x != x) return x;
    if (!(x > -126.0f)) return 0.0f;
    const float fl = floorf(x);
    const float f = __fsub_rn(x, fl);
    float p = __uint_as_float(0x39665cf6u);
    p = __fmaf_rn(p, f, __uint_as_float(0x3aa228dbu));
    p = __fmaf_rn(p, f, __uint_as_float(0x3c1eab8fu));
    p = __fmaf_rn(p, f, __uint_as_float(0x3d634139u));
    p = __fmaf_rn(p, f, __uint_as_float(0x3e75feb3u));
    p = __fmaf_rn(p, f, __uint_as_float(0x3f317217u));
    p = __fmaf_rn(p, f, 1.0f);
    const int e = (int)fl;
    return __int_as_float((int)((unsigned)__float_as_int(p) + ((unsigned)e << 23)));
}

// attn_ref.rs::st_merge on one value: b (later) into a (earlier). m and d are
// the same scalar operations on every dimension, so every thread that folds a
// dimension computes the same m and d.
__device__ __forceinline__ void pers_attn_merge1(float& am, float& ad, float& aa, float bm, float bd, float ba)
{
    if (bm == PERS_NEG_INF) return;
    if (am == PERS_NEG_INF) {
        am = bm;
        ad = bd;
        aa = ba;
        return;
    }
    const float m = fmaxf(am, bm);
    const float fa = pers_ex2(__fsub_rn(am, m)), fb = pers_ex2(__fsub_rn(bm, m));
    ad = __fadd_rn(__fmul_rn(ad, fa), __fmul_rn(bd, fb));
    aa = __fadd_rn(__fmul_rn(aa, fa), __fmul_rn(ba, fb));
    am = m;
}

// Chunks [r0, r0 + PF x nthreads) of step st, this thread's share: chunk idx
// is tensor idx / 512 (K first), staged position pl = (idx mod 512) / 16,
// bytes 16 (idx mod 16) of its row. A step holds TPU tiles of each of 4 / TPU
// units: pl = (unit x TPU + t) x 8 + offset, tile st x TPU + t of the unit
// (TPU 1: one tile of each of the 4 units, attn_dec.cu's step). Past the
// row: zeros, never used.
template <u32 PF>
__device__ __forceinline__ void pers_attn_load(uint4 pf[PF], const unsigned short* kh, const unsigned short* vh,
                                               u32 base, u32 st, u32 tpu, u32 lr, u32 r0)
{
#pragma unroll
    for (u32 i = 0; i < PF; ++i) {
        const u32 idx = r0 + i * blockDim.x + threadIdx.x;
        uint4 z;
        z.x = 0u; z.y = 0u; z.z = 0u; z.w = 0u;
        if (idx < PERS_STAGE_U4) {
            const u32 rem = idx % (PERS_STAGE * 16u), pl = rem >> 4;
            const u32 tl = pl / PERS_TT, u = tl / tpu, t = tl - u * tpu;
            const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + (pl % PERS_TT);
            if (p < lr) z = __ldcg(reinterpret_cast<const uint4*>((idx < PERS_STAGE * 16u ? kh : vh) + (unsigned long long)p * PERS_HD) + (rem & 15u));
        }
        pf[i] = z;
    }
}

// Form 6: chunk idx of step st copied global to shared by cp.async, cached in
// L2 only (the cache was written in this launch, as __ldcg reads it). A
// chunk past the row is zero-filled (source size 0, nothing read). The host
// build copies at once: its cp.async completes before the wait that follows.
__device__ __forceinline__ void pers_cp16z(uint4* dst, const uint4* src, bool valid)
{
#ifdef LLVQ_HOST_BUILD
    uint4 z;
    z.x = 0u; z.y = 0u; z.z = 0u; z.w = 0u;
    *dst = valid ? *src : z;
#else
    unsigned long long d;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(d) : "l"(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"((unsigned)d), "l"(src), "r"(valid ? 16u : 0u)
                 : "memory");
#endif
}

__device__ __forceinline__ void pers_cp_commit()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Every group but the PERS_ATTN_RING - 1 newest is complete, for this
// thread's copies (a CTA barrier after it makes them everyone's).
__device__ __forceinline__ void pers_cp_wait_ring()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.wait_group %0;" ::"n"(PERS_ATTN_RING - 1u) : "memory");
#endif
}

__device__ __forceinline__ void pers_cp_wait_all()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Step st of a job into `stage` (one ring slot), every chunk pers_attn_load
// would load, the same address map; then one commit, empty past the last
// step, so the count of groups a thread has committed stays uniform.
__device__ __forceinline__ void pers_attn_issue(uint4* stage, const unsigned short* kh, const unsigned short* vh,
                                                u32 base, u32 st, u32 nsteps, u32 tpu, u32 lr)
{
    if (st < nsteps) {
        for (u32 idx = threadIdx.x; idx < PERS_STAGE_U4; idx += blockDim.x) {
            const u32 rem = idx % (PERS_STAGE * 16u), pl = rem >> 4;
            const u32 tl = pl / PERS_TT, u = tl / tpu, t = tl - u * tpu;
            const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + pl % PERS_TT;
            const unsigned short* src = (idx < PERS_STAGE * 16u ? kh : vh);
            const bool valid = p < lr;
            pers_cp16z(stage + idx,
                       reinterpret_cast<const uint4*>(src + (unsigned long long)(valid ? p : 0u) * PERS_HD) + (rem & 15u),
                       valid);
        }
    }
    pers_cp_commit();
}

template <u32 PF>
__device__ __forceinline__ void pers_attn_store(const uint4 pf[PF], uint4* stage, u32 r0)
{
#pragma unroll
    for (u32 i = 0; i < PF; ++i) {
        const u32 idx = r0 + i * blockDim.x + threadIdx.x;
        if (idx < PERS_STAGE_U4) stage[idx] = pf[i];
    }
}

// One tile of one task: nv positions staged at ks / vs (16 chunks a
// position), this lane's 8 dimensions of its query head (qrow, widened in
// shared memory), its state in shared memory (row: a[128], m, d). `own`
// false: a half-warp past the last head, which computes from a fresh state
// and writes nothing (it never reads a row another half writes). Both halves
// of the warp call it with the same nv. attn_dec.cu's tile, op for op; the
// accumulators are read only once the scores are made, so q and the
// accumulators are never live together. Lane 0 of the half rewrites m and d
// at the end, which the other lanes read: a __syncwarp separates every read
// from that write (without it the host harness, which runs a lane to its next
// shuffle or barrier, caught a lane reading the new m).
__device__ __forceinline__ void pers_attn_tile(const float* qrow, float* row, bool own, const uint4* ks,
                                               const uint4* vs, u32 nv, float c, u32 lane)
{
    const u32 l16 = lane & 15u, dl = l16 * 8u;
    __syncwarp();  // a previous tile of this task (same step) wrote the row: its writes precede these reads
    float sc[PERS_TT];
    {
        float qf[8];
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) qf[i] = qrow[dl + i];
#pragma unroll
        for (u32 j = 0; j < PERS_TT; ++j) {
            sc[j] = 0.0f;
            if (j < nv) {
                float kf[8];
                pers_unpack8(ks[j * 16u + l16], kf);
                float t = 0.0f;
#pragma unroll
                for (u32 i = 0; i < 8u; ++i) t = __fmaf_rn(qf[i], kf[i], t);
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off));
                sc[j] = __fmul_rn(t, c);
            }
        }
    }
    const float m = own ? row[PERS_HD] : PERS_NEG_INF, d = own ? row[PERS_HD + 1u] : 0.0f;
    float mt = sc[0];
#pragma unroll
    for (u32 j = 1; j < PERS_TT; ++j)
        if (j < nv) mt = fmaxf(mt, sc[j]);
    const bool fresh = m == PERS_NEG_INF;
    const float mn = fresh ? mt : fmaxf(m, mt);
    const float carry = fresh ? 0.0f : pers_ex2(__fsub_rn(m, mn));
    float ts = 0.0f;
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) {
        if (j < nv) {
            sc[j] = pers_ex2(__fsub_rn(sc[j], mn));
            ts = __fadd_rn(ts, sc[j]);
        }
    }
    const float dn = __fadd_rn(__fmul_rn(d, carry), ts);
    float a[8];
#pragma unroll
    for (u32 i = 0; i < 8u; ++i) a[i] = __fmul_rn(own ? row[dl + i] : 0.0f, carry);
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) {
        if (j < nv) {
            float vf[8];
            pers_unpack8(vs[j * 16u + l16], vf);
#pragma unroll
            for (u32 i = 0; i < 8u; ++i) a[i] = __fmaf_rn(sc[j], vf[i], a[i]);
        }
    }
    __syncwarp();  // memory order: every lane's reads of the row precede lane 0's rewrite of m and d
    if (own) {
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];
        if (l16 == 0u) {
            row[PERS_HD] = mn;
            row[PERS_HD + 1u] = dn;
        }
    }
}

// The phase. q: [n_heads][128] binary16 after norm and RoPE; kc, vc:
// [n_kv][cap][128] holding positions 0..l-1 (the new one appended); part:
// [n_kv][ceil(cap / 256)][groups][PSTRIDE] floats of split states; ticket:
// [n_kv] counters, zero at rest and left at zero; out: [n_heads][128]
// binary16; smem: PERS_ATTN_SMEM(groups) bytes of this CTA's shared memory,
// 16-byte aligned. `dbg_state`, nullable: [n_heads][PSTRIDE], each head's
// (a, m, d) before the division. Every thread of every CTA calls it; no
// grid barrier inside, so the caller's next grid wait orders `out`.
//
// Form 6 (PERS_ATTN_RING 3): the steps go global to shared by cp.async into
// three stages, steps st + 1 and st + 2 in flight while step st's tiles run,
// and PF is not used. The staged values, the tiles and the folds are the
// same, so the bits are the same.
//
// PF: the 16-byte chunks a thread holds. When PF x blockDim covers the step's
// 1,024 chunks (PF 2 at 512 threads), the next step's loads are in flight
// during the tiles; otherwise (PF 1 at 768, whose registers cannot hold
// more, or the host's small CTAs) each step is staged in rounds before its
// tiles. The staged values are the same either way.
template <u32 PF>
__device__ __forceinline__ void pers_attention(const unsigned short* q, const unsigned short* kc,
                                               const unsigned short* vc, u32 n_heads, u32 groups, u32 cap, u32 l,
                                               float c, float* part, u32* ticket, unsigned short* out,
                                               unsigned char* smem, float* dbg_state)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 n_kv = n_heads / groups;
    const u32 ns = (l + PERS_SPLIT - 1u) / PERS_SPLIT, s_stride = (cap + PERS_SPLIT - 1u) / PERS_SPLIT;
    const u32 nhp = (groups + 1u) / 2u, ntasks = PERS_UPS * nhp;
    const u32 rounds = (PERS_STAGE_U4 + PF * blockDim.x - 1u) / (PF * blockDim.x);
    uint4* stage = reinterpret_cast<uint4*>(smem);
    float* ust = reinterpret_cast<float*>(smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);  // [UPS][groups][PSTRIDE]
    float* qs = ust + PERS_UPS * groups * PERS_PSTRIDE;                 // [groups][128]
    u32* hdr = pers_hdr();
    for (u32 job = blockIdx.x; job < ns * n_kv; job += gridDim.x) {
        const u32 s = job / n_kv, h = job - s * n_kv;
        const u32 base = s * PERS_SPLIT;
        // A split's units past the row are empty: with nu of them holding
        // positions (1, 2, or 3-4), a step stages TPU = 4, 2 or 1 tiles of
        // each of 4 / TPU units, so a row of 64 positions or less takes 2
        // steps instead of 8. Each unit still runs its tiles in order.
        const u32 nu = (l - base + PERS_UNIT - 1u) / PERS_UNIT;
        const u32 tpu = nu >= 3u ? 1u : (nu == 2u ? 2u : 4u);
        const u32 tiles_raw = (l - base + PERS_TT - 1u) / PERS_TT;
        const u32 tiles = tiles_raw < PERS_UNIT / PERS_TT ? tiles_raw : PERS_UNIT / PERS_TT;
        const u32 nsteps = (tiles + tpu - 1u) / tpu;
        const unsigned short* kh = kc + (unsigned long long)h * cap * PERS_HD;
        const unsigned short* vh = vc + (unsigned long long)h * cap * PERS_HD;
#if PERS_ATTN_RING > 1
        // form 6: the first steps' copies, one commit each, before the states
        // and q are written
        for (u32 st = 0; st < PERS_ATTN_RING; ++st)
            pers_attn_issue(stage + st * PERS_STAGE_U4, kh, vh, base, st, nsteps, tpu, l);
        (void)rounds;
#else
        uint4 pf[PF];
        if (rounds == 1u) pers_attn_load<PF>(pf, kh, vh, base, 0u, tpu, l, 0u);
#endif
        for (u32 i = threadIdx.x; i < PERS_UPS * groups * PERS_PSTRIDE; i += blockDim.x) {
            const u32 k = i % PERS_PSTRIDE;
            ust[i] = k == PERS_HD ? PERS_NEG_INF : 0.0f;
        }
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) qs[i] = h2f(__ldcg(q + h * groups * PERS_HD + i));
        for (u32 st = 0; st < nsteps; ++st) {
#if PERS_ATTN_RING > 1
            uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;
            pers_cp_wait_ring();  // this thread's copies of step st are done
            __syncthreads();      // everyone's: the step is staged, the states and q are written
#else
            uint4* const sbuf = stage;
            __syncthreads();  // the previous step's readers are done, the states and q are written
            if (rounds == 1u) {
                pers_attn_store<PF>(pf, stage, 0u);
            } else {
                for (u32 r = 0; r < rounds; ++r) {
                    pers_attn_load<PF>(pf, kh, vh, base, st, tpu, l, r * PF * blockDim.x);
                    pers_attn_store<PF>(pf, stage, r * PF * blockDim.x);
                }
            }
            __syncthreads();  // the step is staged
            if (rounds == 1u && st + 1u < nsteps) pers_attn_load<PF>(pf, kh, vh, base, st + 1u, tpu, l, 0u);
#endif
            for (u32 task = warp; task < ntasks; task += nwarps) {
                const u32 u = task / nhp, hp = task - u * nhp;
                if (u * tpu >= PERS_UPS) continue;  // a unit this step does not stage (uniform)
                const u32 g = 2u * hp + (lane >> 4);
                // A half past the last head computes on a real q from a fresh
                // state and keeps nothing.
                const bool own = g < groups;
                const u32 ge = own ? g : groups - 1u;
                for (u32 t = 0; t < tpu; ++t) {
                    const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
                    if (p0 >= l || (st * tpu + t) * PERS_TT >= PERS_UNIT) break;  // uniform over the warp: unit u
                    const u32 nv = l - p0 < PERS_TT ? l - p0 : PERS_TT;
                    const u32 sl = (u * tpu + t) * PERS_TT;
                    pers_attn_tile(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own, sbuf + sl * 16u,
                                   sbuf + (PERS_STAGE + sl) * 16u, nv, c, lane);
                }
            }
#if PERS_ATTN_RING > 1
            __syncthreads();  // the slot's readers are done: it takes step st + RING
            pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);
#endif
        }
#if PERS_ATTN_RING > 1
        pers_cp_wait_all();  // no copy may land in the stages once they are reused
#endif
        __syncthreads();  // every unit state is written
        // The units, folded into the split state in unit order.
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) {
            const u32 g = i / PERS_HD, k = i - g * PERS_HD;
            float sm = PERS_NEG_INF, sd = 0.0f, sa = 0.0f;
            for (u32 uu = 0; uu < PERS_UPS; ++uu) {
                const float* row = ust + (uu * groups + g) * PERS_PSTRIDE;
                pers_attn_merge1(sm, sd, sa, row[PERS_HD], row[PERS_HD + 1u], row[k]);
            }
            const u32 hq = h * groups + g;
            if (ns == 1u) {
                // The row's only split: folding it into an empty state is a copy.
                out[hq * PERS_HD + k] = f2h(__fdiv_rn(sa, sd));
                if (dbg_state) {
                    dbg_state[hq * PERS_PSTRIDE + k] = sa;
                    if (k == 0u) {
                        dbg_state[hq * PERS_PSTRIDE + PERS_HD] = sm;
                        dbg_state[hq * PERS_PSTRIDE + PERS_HD + 1u] = sd;
                    }
                }
            } else {
                float* P = part + (((unsigned long long)h * s_stride + s) * groups + g) * PERS_PSTRIDE;
                P[k] = sa;
                if (k == 0u) {
                    P[PERS_HD] = sm;
                    P[PERS_HD + 1u] = sd;
                }
            }
        }
        if (ns > 1u) {
            // One ticket a job on its KV head's counter; the last folds.
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0u) hdr[PERS_HDR_TICKET] = atomicAdd(ticket + h, 1u) == ns - 1u ? 1u : 0u;
            __syncthreads();
            if (hdr[PERS_HDR_TICKET] != 0u) {
                __threadfence();  // acquire: the states read below follow the ticket drawn
                // The split fold, chunk by chunk, attn_dec.cu's combine: every
                // load of a chunk issued before any is used (one round trip a
                // chunk), its maxima and sums through shared memory (the stage,
                // free after the steps), each thread's accumulators in
                // registers. Per head the splits in ascending order. The loop
                // over (head, dimension) is uniform, so its barriers are too.
                float* cmd = reinterpret_cast<float*>(stage);  // [CH][groups][2]: m, d
                for (u32 b0 = 0; b0 < groups * PERS_HD; b0 += blockDim.x) {
                    const u32 i = b0 + threadIdx.x;
                    const bool act = i < groups * PERS_HD;
                    const u32 g = act ? i / PERS_HD : 0u, k = act ? i - g * PERS_HD : 0u;
                    float am = PERS_NEG_INF, ad = 0.0f, aa = 0.0f;
                    for (u32 c0 = 0; c0 < ns; c0 += PERS_ATTN_CH) {
                        __syncthreads();  // the previous chunk's readers are done
                        for (u32 t = threadIdx.x; t < 2u * PERS_ATTN_CH * groups; t += blockDim.x) {
                            const u32 j = t / (2u * groups), gg = (t >> 1) % groups;
                            float val = 0.0f;
                            if (c0 + j < ns)
                                val = __ldcg(part + (((unsigned long long)h * s_stride + c0 + j) * groups + gg) * PERS_PSTRIDE + PERS_HD + (t & 1u));
                            cmd[t] = val;
                        }
                        float ba[PERS_ATTN_CH];
#pragma unroll
                        for (u32 j = 0; j < PERS_ATTN_CH; ++j) {
                            ba[j] = 0.0f;
                            if (act && c0 + j < ns)
                                ba[j] = __ldcg(part + (((unsigned long long)h * s_stride + c0 + j) * groups + g) * PERS_PSTRIDE + k);
                        }
                        __syncthreads();  // the chunk's maxima and sums are in shared memory
                        if (act) {
#pragma unroll
                            for (u32 j = 0; j < PERS_ATTN_CH; ++j)
                                if (c0 + j < ns)
                                    pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[j]);
                        }
                    }
                    if (act) {
                        const u32 hq = h * groups + g;
                        out[hq * PERS_HD + k] = f2h(__fdiv_rn(aa, ad));
                        if (dbg_state) {
                            dbg_state[hq * PERS_PSTRIDE + k] = aa;
                            if (k == 0u) {
                                dbg_state[hq * PERS_PSTRIDE + PERS_HD] = am;
                                dbg_state[hq * PERS_PSTRIDE + PERS_HD + 1u] = ad;
                            }
                        }
                    }
                }
                if (threadIdx.x == 0u) ticket[h] = 0u;  // the next layer's draws follow a grid barrier
            }
        }
        __syncthreads();  // the next job restages the shared buffer and the header
    }
}

// ---------------------------------------------------------------------------
// 11. 4E's q/k-norm, RoPE and KV append, one decode row at position `pos`.
//     candle runs, per head of 128: rmsnorm_f16 with its 32-thread block
//     (exact_rms_scale with ncols = 128), then rotary_emb::rope (reduce.cu
//     rope, non-interleaved): o1 = a*c - b*s, o2 = a*s + b*c on the pair
//     (i, i + 64), in __half operators. Those compile to mul.f16 and sub.f16
//     or add.f16, which the driver's ptxas may fuse into fma.rn.f16, either
//     product. FORM picks one: 0 none fused, 1 the first product fused, 2 the
//     second. Which one candle's card code is can only be read on the card:
//     route P's load-time check runs candle's rope and keeps the form that
//     matches, or keeps candle's launch. A fused form rounds once from the
//     exact value: the 22-bit product is exact in f64, and so is its sum with
//     a binary16 term (an in-range product and the term span at most 41 bits,
//     under f64's 53), so one rounding of the f64 sum to binary16 is the
//     single rounding fma.rn.f16 makes. TwoSum's error is kept as a guard, and
//     the host test shows it is always zero on binary16 inputs.
// ---------------------------------------------------------------------------

// RN to binary16 of the exact value s + err, |err| below half an ulp of s in
// f64 (TwoSum's guarantee): err only matters when s sits on a binary16 tie.
__device__ __forceinline__ unsigned short pers_f64_to_f16(double s, double err)
{
    const unsigned long long b = (unsigned long long)__double_as_longlong(s);
    const u32 sign = (u32)(b >> 48) & 0x8000u;
    const int e = (int)((b >> 52) & 0x7ffu);
    const unsigned long long m = b & 0xFFFFFFFFFFFFFull;
    if (e == 0x7ff) return (unsigned short)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0) return (unsigned short)sign;  // f64 subnormal: far below the least binary16
    const unsigned long long sig = m | (1ull << 52);
    int e16 = e - 1023 + 15;
    if (e16 >= 31) return (unsigned short)(sign | 0x7c00u);
    const int shift = e16 <= 0 ? 1051 - e : 42;
    if (shift >= 64) return (unsigned short)sign;
    unsigned long long q = sig >> shift;
    const unsigned long long rem = sig & ((1ull << shift) - 1ull), half = 1ull << (shift - 1);
    // the exact value is larger in magnitude than s when err has s's sign
    const bool beyond = err != 0.0 && ((err > 0.0) == (s > 0.0));
    if (rem > half || (rem == half && (beyond || (err == 0.0 && (q & 1ull))))) ++q;
    if (e16 <= 0) return (unsigned short)(sign | (u32)q);  // q = 0x400 is the least normal, encoded alike
    if (q == 0x800ull) {
        q = 0x400ull;
        if (++e16 >= 31) return (unsigned short)(sign | 0x7c00u);
    }
    return (unsigned short)(sign | ((u32)e16 << 10) | (u32)(q & 0x3ffull));
}

// fma.rn.f16(±a·b, c): one rounding of the exact value.
__device__ __forceinline__ unsigned short pers_f16_fma_exact(unsigned short a, unsigned short b, unsigned short c,
                                                             bool neg_ab)
{
    const double x = (neg_ab ? -1.0 : 1.0) * ((double)h2f(a) * (double)h2f(b));  // exact: 22 bits
    const double y = (double)h2f(c);
    const double s = x + y;
    const double bb = s - x;
    const double err = (x - (s - bb)) + (y - bb);  // TwoSum: s + err == x + y exactly
    return pers_f64_to_f16(s, err);
}

template <int FORM>
__device__ __forceinline__ void pers_rope_pair(unsigned short a, unsigned short b, unsigned short c, unsigned short s,
                                               unsigned short* o1, unsigned short* o2)
{
    const unsigned short ac = exact_hmul(a, c), bs = exact_hmul(b, s), as = exact_hmul(a, s), bc = exact_hmul(b, c);
    if (FORM == 0) {
        *o1 = f2h(__fsub_rn(h2f(ac), h2f(bs)));
        *o2 = exact_hadd(as, bc);
    } else if (FORM == 1) {
        *o1 = pers_f16_fma_exact(a, c, (unsigned short)(bs ^ 0x8000u), false);
        *o2 = pers_f16_fma_exact(a, s, bc, false);
    } else {
        *o1 = pers_f16_fma_exact(b, s, ac, true);
        *o2 = pers_f16_fma_exact(b, c, as, false);
    }
}

// One head of 128 in place-free form: src (binary16, this head's raw row),
// alpha (the norm's 128 weights), cos/sin (64 entries at this position), dst.
// One warp; candle's 32-thread tree for the norm. Lane l reads columns l,
// l + 32, l + 64, l + 96 once: the norm sums them in that order (candle's
// thread order), and RoPE's pairs (i, i + 64) at i = l and l + 32 are the
// same four values, so they stay in registers between the two.
template <int FORM>
__device__ __forceinline__ void pers_head_norm_rope(const unsigned short* src, const unsigned short* alpha,
                                                    const unsigned short* cosr, const unsigned short* sinr, float eps,
                                                    unsigned short* dst, u32 lane)
{
    unsigned short hx[4];
#pragma unroll
    for (u32 j = 0; j < 4u; ++j) hx[j] = __ldcg(src + lane + 32u * j);
    float tmp = 0.0f;
#pragma unroll
    for (u32 j = 0; j < 4u; ++j) {
        const float xi = h2f(hx[j]);
        tmp = __fmaf_rn(xi, xi, tmp);
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));
    const float scale = rsqrtf(__fadd_rn(__fdiv_rn(tmp, 128.0f), eps));
#pragma unroll
    for (u32 j = 0; j < 2u; ++j) {
        const u32 i = lane + 32u * j;
        const unsigned short a = exact_rms_apply(scale, hx[j], alpha[i]);
        const unsigned short b = exact_rms_apply(scale, hx[j + 2u], alpha[i + 64u]);
        unsigned short o1, o2;
        pers_rope_pair<FORM>(a, b, cosr[i], sinr[i], &o1, &o2);
        dst[i] = o1;
        dst[i + 64u] = o2;
    }
}

// The phase: q heads normed and roped into q_out; k heads normed and roped
// into the K cache at `pos`; v copied into the V cache at `pos`. One warp a
// head, head w on CTA w mod grid: the 48 heads of the 4B on 48 SMs, one warp
// each, where the first version put 16 on each of CTAs 0 to 2 and their f64
// RoPE arithmetic shared one SM's FP64 units (189 us a token on CTA 0, 5.7 on
// the mean CTA, pers-4e-r3-2026-11-05). q_raw [n_heads][128], k_raw and v_raw
// [n_kv][128].
template <int FORM>
__device__ __forceinline__ void pers_4e_norm_rope_append(const unsigned short* q_raw, const unsigned short* k_raw,
                                                         const unsigned short* v_raw, const unsigned short* q_alpha,
                                                         const unsigned short* k_alpha, const unsigned short* cosr,
                                                         const unsigned short* sinr, float eps, u32 n_heads, u32 n_kv,
                                                         u32 cap, u32 pos, unsigned short* q_out, unsigned short* kc,
                                                         unsigned short* vc)
{
    const u32 lane = threadIdx.x & 31u;
    const u32 nwarps = blockDim.x >> 5;
    const u32 total = gridDim.x * nwarps;
    for (u32 w = blockIdx.x + gridDim.x * (threadIdx.x >> 5); w < n_heads + 2u * n_kv; w += total) {
        if (w < n_heads) {
            pers_head_norm_rope<FORM>(q_raw + w * 128u, q_alpha, cosr, sinr, eps, q_out + w * 128u, lane);
        } else if (w < n_heads + n_kv) {
            const u32 h = w - n_heads;
            pers_head_norm_rope<FORM>(k_raw + h * 128u, k_alpha, cosr, sinr, eps,
                                      kc + ((unsigned long long)h * cap + pos) * 128u, lane);
        } else {
            const u32 h = w - n_heads - n_kv;
            for (u32 i = lane; i < 128u; i += 32u) vc[((unsigned long long)h * cap + pos) * 128u + i] = __ldcg(v_raw + h * 128u + i);
        }
    }
}

#endif  // LLVQ_PERS_CUH
