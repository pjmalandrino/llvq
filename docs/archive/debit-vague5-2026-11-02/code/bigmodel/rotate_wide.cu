// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// operator's review of lead "bigmodel" (wave5-deep/bigmodel.md §3), revised
// the same day after review. Not part of the repository. Every claim of exactness below is an argument to be
// checked by the host test and the card job of prereg-bigmodel-2, not a result.
//
// rotate_wide.cu: three entry points (plus their rows twins) that return the
// bits of rot_apply (llvq-cuda/kernels/rotate.cu:38-67) without its one
// limitation that matters here, the whole vector staged in shared memory
// (n * 4 bytes, 102,400 at n = 25,600 against 101,376 opt-in on sm_89/sm_120).
//
// Composition contract, as rotate.cu: llvq_rot.cuh first (u32, rot_h2f,
// rot_load, rot_wht_step, rot_scale_out, rot_mix, LLVQ_ROT_KMAX), then this
// file. At serve time it is a unit of its own (llvq_rot.cuh, rotate_wide.cu),
// loaded only when some width maps to Device, so the served unit's text and
// the sha256 every served log prints (fused_cuda.rs:553) do not move. The
// bench unit of rotbench is llvq_rot.cuh, rotate.cu, rotate_wide.cu, so K0
// and the new kernels are timed side by side; embedded_source() needs a
// "rotate_wide.cu" arm. When R1 is served (after V3 and the operator's
// served-path ruling), the served unit's sha256 moves then, by decision.
// llvq_rot.cuh itself is not edited, not even its wrong comment at :46-48
// about 25,600: that correction goes to the docs (format-noyau §8, ETAT).
//
// The exactness contract every kernel here honours (bigmodel.md §3.1): each
// output coordinate is
//   (1) f16 -> f32 widening (exact), (2) sign flip (exact),
//   (3) butterfly stages len = 1, 2, 4, ..., m/2 IN THAT ORDER, each mapping
//       (a, b) -> (a + b, a - b) with a the LOWER index, f32 round-to-nearest,
//   (4) col = y * inv (one f32 multiply, inv narrowed on the host),
//   (5) acc = fma(small[g][t], col[t], acc) for t ascending from acc = +0.0f.
// There is no reduction across threads, and no a*b+c pattern other than the
// explicit __fmaf_rn (the col product feeds a multiplicand, never an addend),
// so NVRTC's default --fmad=true has nothing to contract. Any thread mapping
// and any memory space that applies these operations to the same operands
// returns the same bits.

#ifndef LLVQ_ROT_CUH
#include "llvq_rot.cuh"
#endif

// ===========================================================================
// R0. rot_apply with its staging in DEVICE memory: the minimal wall lift.
//
// The Metal port already does exactly this (llvq_rot.metal:28-43 and its
// entry point), and llvq-metal/tests/rot_matches_host.rs gates it bit for bit
// against the CUDA text. Here the same helpers run on a global `scratch`
// instead of `extern __shared__`. __syncthreads() orders global as well as
// shared accesses among the threads of one block (CUDA programming guide),
// which is the only scope this kernel shares anything in.
//
// Host owes, on top of rot_apply's contract:
//   * scratch: device memory, distinct from xout, at least
//       n_rows * n + (LLVQ_ROT_KMAX - k) * m floats.
//     The tail term keeps any speculated read of rot_mix's unrolled col loop
//     (t >= k, llvq_rot.cuh:224) inside the allocation. NVRTC predicates those
//     loads in practice; the sizing removes the question instead of trusting
//     the compiler (the Metal port clamps the index for the same reason,
//     llvq_rot.metal rot_mix).
//   * dynamic shared memory: 0.
//   * selection: only where llvq_cuda::shared::rot_plan refuses the width, so
//     no served width of the 4B, 8B or 14B changes kernel.
// ===========================================================================
extern "C" __global__ void rot_apply_dev(const unsigned short* __restrict__ xin,
                                         const u32* __restrict__ signbits,
                                         const float* __restrict__ small,
                                         float* __restrict__ xout,
                                         float* __restrict__ scratch,
                                         u32 n,
                                         u32 m,
                                         u32 k,
                                         float inv,
                                         u32 x_off)
{
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    rot_load(xin, signbits, scratch, n, x_off, tid, nthreads);
    __syncthreads();
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(scratch, n, len, tid, nthreads);
        __syncthreads();
    }
    if (k == 1u) {
        rot_scale_out(scratch, xout, n, inv, tid, nthreads);
    } else {
        rot_mix(scratch, small, xout, m, k, inv, tid, nthreads);
    }
}

// grid.x == n_rows exactly; row r uses scratch + r * n, xout + r * n.
extern "C" __global__ void rot_apply_rows_dev(const unsigned short* __restrict__ xin,
                                              const u32* __restrict__ signbits,
                                              const float* __restrict__ small,
                                              float* __restrict__ xout,
                                              float* __restrict__ scratch,
                                              u32 n,
                                              u32 m,
                                              u32 k,
                                              float inv,
                                              u32 x_off,
                                              u32 row_stride)
{
    const u32 r = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    float* s = scratch + (unsigned long long)r * (unsigned long long)n;
    float* out = xout + (unsigned long long)r * (unsigned long long)n;
    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    __syncthreads();
    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}

// ===========================================================================
// R1. The fast single-CTA rotation, for widths whose n * 4 fits one block.
//
// Three changes against rot_apply, none to the arithmetic:
//   * stages len = 1 and 2 in registers, four consecutive coordinates a
//     thread (one float4 of shared memory, conflict-free store);
//   * the remaining stages two at a time ("radix-4" passes): a thread owns the
//     quad j0, j0+len, j0+2len, j0+3len and applies stage len then stage 2len
//     to it, which are exactly the four pairs those two stages touch there;
//     addressing by shifts, no runtime division (rot_wht_step divides by a
//     runtime len, llvq_rot.cuh:163, which is most of its instruction count);
//   * the mix on every thread, and its chain cut at KB in {8, 24, 32} >= k
//     instead of LLVQ_ROT_KMAX = 32 (proof in bigmodel.md §3.1: the dropped
//     terms are fma(0, 0, acc) = acc, and acc is never -0).
// Barriers: 1 + ceil((log2 m - 2) / 2), i.e. 5 at m = 512 and m = 1024, 6 at
// m = 4096, against log2(m) + 1 = 10, 11 and 13 for rot_apply.
// Host owes: m >= 4 (every real width has m >= 512), n * 4 bytes of dynamic
// shared memory (unchanged), x_off % 4 == 0 only if the float4 load path is
// enabled (the scalar path below has no alignment requirement).
// ===========================================================================

// Phase 1 for the coordinates [base, base + nb): widen, flip, stages 1 and 2.
// `s` receives coordinate base + i at s[i]. base is a multiple of m >= 4, so a
// quad never straddles a group and its four sign bits sit in one word.
__device__ __forceinline__ void rot_r_phase1(const unsigned short* __restrict__ xin,
                                             const u32* __restrict__ signbits,
                                             float* s,
                                             u32 nb,
                                             u32 base,
                                             u32 x_off,
                                             u32 tid,
                                             u32 nthreads)
{
    for (u32 c = tid; c < (nb >> 2); c += nthreads) {
        u32 i0 = c << 2;
        u32 gi0 = base + i0;
        u32 w = signbits[gi0 >> 5];
        float v[4];
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) {
            float x = rot_h2f(xin[x_off + gi0 + e]);
            v[e] = ((w >> ((gi0 + e) & 31u)) & 1u) ? -x : x;
        }
        // stage len = 1: pairs (0,1) and (2,3); lower index keeps a + b
        float a0 = v[0] + v[1], a1 = v[0] - v[1];
        float a2 = v[2] + v[3], a3 = v[2] - v[3];
        // stage len = 2: pairs (0,2) and (1,3)
        s[i0 + 0u] = a0 + a2;
        s[i0 + 2u] = a0 - a2;
        s[i0 + 1u] = a1 + a3;
        s[i0 + 3u] = a1 - a3;
        // On the card these four stores are one float4 store (16-byte aligned
        // because i0 % 4 == 0 and the extern shared array is aligned); written
        // scalar here so the host build needs no float4 type.
    }
}

// Stages len = 2^lg and 2len, over nb coordinates made of whole groups of m.
// Valid when 4 * len <= m. The group term distributes out of the quad index
// exactly as in rot_wht_step's comment (m is a multiple of 4 * len).
__device__ __forceinline__ void rot_r_pass4(float* s, u32 nb, u32 lg, u32 tid, u32 nthreads)
{
    u32 len = 1u << lg;
    for (u32 q = tid; q < (nb >> 2); q += nthreads) {
        u32 j0 = ((q >> lg) << (lg + 2u)) | (q & (len - 1u));
        float a = s[j0];
        float b = s[j0 + len];
        float c = s[j0 + 2u * len];
        float d = s[j0 + 3u * len];
        float ab0 = a + b, ab1 = a - b;   // stage len, pair (j0, j0+len)
        float cd0 = c + d, cd1 = c - d;   // stage len, pair (j0+2len, j0+3len)
        s[j0] = ab0 + cd0;                // stage 2len, pair (j0, j0+2len)
        s[j0 + 2u * len] = ab0 - cd0;
        s[j0 + len] = ab1 + cd1;          // stage 2len, pair (j0+len, j0+3len)
        s[j0 + 3u * len] = ab1 - cd1;
    }
}

// The single stage left when log2(m) - 2 is odd (len = m/2).
__device__ __forceinline__ void rot_r_pass2(float* s, u32 nb, u32 lg, u32 tid, u32 nthreads)
{
    u32 len = 1u << lg;
    for (u32 p = tid; p < (nb >> 1); p += nthreads) {
        u32 j = ((p >> lg) << (lg + 1u)) | (p & (len - 1u));
        float a = s[j];
        float b = s[j + len];
        s[j] = a + b;
        s[j + len] = a - b;
    }
}

// Stages 4 .. m/2 over nb coordinates, barriers included. Every thread runs
// the same number of passes (m is uniform), so every thread reaches every
// barrier. Returns nothing; leaves the unscaled transform in s.
__device__ __forceinline__ void rot_r_stages(float* s, u32 nb, u32 m, u32 tid, u32 nthreads)
{
    u32 lg = 2u;
    for (; (4u << lg) <= m; lg += 2u) {
        rot_r_pass4(s, nb, lg, tid, nthreads);
        __syncthreads();
    }
    if ((1u << lg) < m) {
        rot_r_pass2(s, nb, lg, tid, nthreads);
        __syncthreads();
    }
}

// One block of mix outputs with the chain cut at KB (k <= KB <= KMAX).
// Work item w = (slice, column): P threads share a column when the block is
// wider than m (P = 2 at m = 512 and 1,024 threads), each taking a contiguous
// slice of the k outputs, so slices are warp-uniform and `small` reads are
// broadcasts. `s` may be shared (R1) or global (R1w): reads only.
template <unsigned KB>
__device__ __forceinline__ void rot_r_mix_kb(const float* s,
                                             const float* __restrict__ small,
                                             float* __restrict__ xout,
                                             u32 m,
                                             u32 k,
                                             float inv,
                                             u32 tid,
                                             u32 nthreads)
{
    u32 P = nthreads > m ? nthreads / m : 1u;   // m and nthreads are powers of two
    u32 per = (k + P - 1u) / P;
    for (u32 w = tid; w < m * P; w += nthreads) {
        u32 j = w & (m - 1u);
        u32 sl = w / m;                          // a shift in the real code
        float col[KB];
#pragma unroll
        for (u32 t = 0; t < KB; ++t) col[t] = t < k ? s[t * m + j] * inv : 0.0f;
        u32 g0 = sl * per;
        u32 g1 = (g0 + per < k) ? g0 + per : k;
        for (u32 g = g0; g < g1; ++g) {
            float acc = 0.0f;
#pragma unroll
            for (u32 t = 0; t < KB; ++t)
                acc = __fmaf_rn(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
            xout[g * m + j] = acc;
        }
    }
}

__device__ __forceinline__ void rot_r_mix(const float* s,
                                          const float* __restrict__ small,
                                          float* __restrict__ xout,
                                          u32 n,
                                          u32 m,
                                          u32 k,
                                          float inv,
                                          u32 tid,
                                          u32 nthreads)
{
    // k == 1: rot_apply's own branch (rot_scale_out), not a 1x1 mix: the Rust
    // reference skips Q_odd entirely when k == 1 (rotation.rs:123-125).
    if (k == 1u) rot_scale_out(s, xout, n, inv, tid, nthreads);
    else if (k <= 8u) rot_r_mix_kb<8u>(s, small, xout, m, k, inv, tid, nthreads);
    else if (k <= 24u) rot_r_mix_kb<24u>(s, small, xout, m, k, inv, tid, nthreads);
    else rot_r_mix_kb<32u>(s, small, xout, m, k, inv, tid, nthreads);
}

extern "C" __global__ void rot_apply_r(const unsigned short* __restrict__ xin,
                                       const u32* __restrict__ signbits,
                                       const float* __restrict__ small,
                                       float* __restrict__ xout,
                                       u32 n,
                                       u32 m,
                                       u32 k,
                                       float inv,
                                       u32 x_off)
{
    extern __shared__ float s[];
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    rot_r_phase1(xin, signbits, s, n, 0u, x_off, tid, nthreads);
    __syncthreads();
    rot_r_stages(s, n, m, tid, nthreads);
    rot_r_mix(s, small, xout, n, m, k, inv, tid, nthreads);
}

// ===========================================================================
// R1w. R1 for widths whose n * 4 exceeds the block ceiling (the 32B down
// site, 25,600): groups pass through shared memory gb at a time (gb * m * 4
// bytes; at m = 1,024, gb = 12 is 48 KiB, on the default bound, so 3 batches
// at 25,600; 6 barriers a batch: phase 1, 4 passes, the scratch copy), each
// finished group is written to `scratch` in device memory, and the mix reads
// the scratch. One launch, one CTA, no inter-block protocol.
// Host owes: scratch of at least n + (32 - k) * m floats (see R0), distinct
// from xout; gb >= 1 with gb * m * 4 <= the function's dynamic shared size.
// One whole group must fit (m * 4 bytes): a width whose single group exceeds
// the block's shared memory (none in Qwen3 or Qwen3.5) goes to R0 instead.
// ===========================================================================
extern "C" __global__ void rot_apply_rw(const unsigned short* __restrict__ xin,
                                        const u32* __restrict__ signbits,
                                        const float* __restrict__ small,
                                        float* __restrict__ xout,
                                        float* __restrict__ scratch,
                                        u32 n,
                                        u32 m,
                                        u32 k,
                                        float inv,
                                        u32 x_off,
                                        u32 gb)
{
    extern __shared__ float s[];   // gb * m floats
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    for (u32 g0 = 0u; g0 < k; g0 += gb) {
        u32 ng = (k - g0 < gb) ? k - g0 : gb;
        u32 nb = ng * m;
        u32 base = g0 * m;
        rot_r_phase1(xin, signbits, s, nb, base, x_off, tid, nthreads);
        __syncthreads();
        rot_r_stages(s, nb, m, tid, nthreads);
        for (u32 i = tid; i < nb; i += nthreads) scratch[base + i] = s[i];
        // The next batch overwrites s, and the mix reads scratch: one barrier
        // covers both (global writes of the block are visible after it).
        __syncthreads();
    }
    rot_r_mix(scratch, small, xout, n, m, k, inv, tid, nthreads);
}

// Rows twins (rot_apply_rows_r, rot_apply_rows_rw) follow rot_apply_rows:
// r = blockIdx.x, x_off + r * row_stride on the input, xout + r * n on the
// output, and for R1w scratch + r * (n + (32 - k) * m). Not written out here.

// ===========================================================================
// Option R2 (not first): k CTAs, one per group, then the LAST CTA to finish
// does the mix. One launch, no spin-wait. Kept as a fallback if R1w measures
// slow at 25,600. Protocol, per row r (tickets[r] zeroed at load):
//
//   ...write this group's unscaled transform to scratch...
//   __threadfence();                       // release: this thread's writes
//   __syncthreads();
//   __shared__ u32 last;
//   if (threadIdx.x == 0) last = (atomicAdd(&tickets[r], 1u) == k - 1u);
//   __syncthreads();
//   if (!last) return;                     // uniform over the CTA
//   __threadfence();                       // acquire side
//   ...mix, reading scratch with __ldcg (L2, never a stale L1 line)...
//   if (threadIdx.x == 0) tickets[r] = 0u; // re-armed; the kernel boundary orders it
//
// Same per-element operations, so the same bits, whichever CTA arrives last.
// Contract: one stream, one in-flight launch per tickets array.
// ===========================================================================
