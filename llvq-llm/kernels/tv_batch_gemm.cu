// tv_batch_gemm.cu: the batched projections as a dequantizing GEMM. Each
// weight block is decoded ONCE per CTA into f16 weights with every scale
// folded in, and multiplied for up to 128 activation rows on the tensor cores
// (mma.sync m16n8k16, f16 in, f32 accumulate). Several projections that read
// rows of the same step go in ONE launch (q, k, v; gate, up).
//
// ## What changed from tv_batch_mma.cu, and why
//
// The first batched kernel (batch-tetra-2026-11-07) gave each lane one weight
// row and read that row's 6-byte block straight from global memory at every
// unit: 32 lanes, 32 rows, 32 sectors a load, and the decode, the shared store
// and the MMA all waited on it. It read 54 GB/s at B = 32. Here:
//
//   * the weight words and the activation of a K chunk are copied into shared
//     memory by cp.async, a row's words contiguous and coalesced, two stages
//     deep, so chunk c + 1 is in flight while chunk c is decoded and
//     multiplied;
//   * the scales are folded into the weight before the MMA: a Tetra weight is
//     `f16(y * (g * inv) * rscale)` and an int4 weight `f16(s * q + b)`. The
//     K dimension is then one plain sum over d_in, with no per-block epilogue,
//     no 24 -> 32 padding (two blocks fill three k16 steps) and the f16 tail of
//     a Tetra row read as two more columns of A;
//   * the activation enters the MMA once, in f16 (the first kernel carried it
//     in two halves, two MMAs a step). An f16 weight and an f16 activation are
//     what every f16 inference stack multiplies. The caller hands the rotated
//     rows of a Tetra projection already rounded to f16, so every CTA reads
//     two bytes a value: each CTA re-reads the activation of its K slice, and
//     at 128 rows in f32 with 64-row tiles that re-read asked the L2 for about
//     16 bytes a cycle a SM, more than it delivers (batch-l40s-2026-11-08).
//
// The weight rounding (2^-11 relative) and the activation rounding are far
// below the quantization step of a 2.7-bit weight. This path is not bit for
// bit with batch 1 and does not try to be: its licence is the T8 band of
// CLAUDE.md, read by the gate of preregistration-batch-l40s.
//
// ## Layout
//
// A CTA owns 128 output rows (8 warps x 16) of one projection and one slice of
// its K chunks. Grid x enumerates the CTAs of every projection of the launch
// (`BgSeg.cta0` is a projection's first). A chunk is KC columns; inside it a
// warp runs sub-steps of 48 columns: lane l decodes row l / 2, half l % 2 of
// the sub-step (one Tetra block, or three int4 words), into the warp's A tile,
// then the warp runs 3 k16 steps x NT n8 tiles of MMA. The activation is f16
// for both kinds, staged as f16 pairs.
//
// Split K: as tv_batch_mma.cu, partials [S][n][d_out] per projection, a
// ticket per row tile, the last CTA folds the slices in order 0..S-1 and
// resets the ticket. One slice stores directly. Deterministic: no atomic on a
// value, a fixed fold order.
//
// Host: llvq-llm/tests/host_batch_gemm.cpp runs these kernels through
// host_cta.h with the MMA modelled from the PTX fragment layout and cp.async
// as a synchronous copy.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_TETRA48_CUH
#include "../../llvq-cuda/kernels/llvq_tetra48.cuh"
#endif

#define BG_ROWS 128u    // output rows a CTA owns
#define BG_WARPS 8u     // 16 rows a warp
#define BG_THREADS 256u
#define BG_SUB 48u      // columns a sub-step decodes: two Tetra blocks, six int4 words
#define BG_AS 28u       // u32 words of a decoded row in a warp's A tile: 24 + 4 pad
#define BG_MAXSEG 4u

// One projection of a launch. Field order is the Rust `#[repr(C)]` mirror's
// (`batch_cuda.rs`, `BgSeg`): eight-byte fields first, no implicit padding.
struct BgSeg {
    unsigned long long w;     // Tetra: words, row-strided; int4: words, d_in / 8 a row
    unsigned long long a0;    // Tetra: gscale f32[2]; int4: scales f16 [d_out][gpr]
    unsigned long long a1;    // Tetra: rscale f32[d_out]; int4: biases f16 [d_out][gpr]
    unsigned long long tail;  // Tetra: tail f16 [d_out][tail_w]; int4: unused
    unsigned long long x;     // f16 [n][d_in]: the rotated rows (Tetra), the natural ones (int4)
    unsigned long long y;     // f16 [n][d_out]
    unsigned long long part;  // f32 [S][n][d_out] when S > 1
    u32 kind;                 // 0 Tetra, 1 int4
    u32 stride;               // Tetra: row stride in u32; int4: groups a row
    u32 nblocks;              // Tetra: blocks a row; int4: log2 of the group
    u32 tail_w;               // Tetra: tail columns; int4: 0
    u32 d_in, d_out;
    u32 nk;                   // chunks of KC columns over d_in (Tetra: and the tail)
    u32 slice;                // chunks a slice
    u32 S;                    // slices
    u32 cta0;                 // first CTA of this projection in the grid
    u32 tk0;                  // first ticket of this projection
    u32 pad;
};

struct BgArgs {
    BgSeg seg[BG_MAXSEG];
    unsigned long long trows, pref, br, suf, invnorm;  // Tetra decoder tables
    unsigned long long ticket;                         // u32, zero at rest
    u32 nseg, n_rows;
};

#ifdef LLVQ_HOST_BUILD
void bg_mma_host(float* d, const u32* a, u32 b0, u32 b1);
#define BG_WARP_SYNC() __syncthreads()
#else
#define BG_WARP_SYNC() __syncwarp()
#endif

__device__ __forceinline__ u32 bg_pack(float lo, float hi)
{
#ifdef LLVQ_HOST_BUILD
    return (u32)f2h(lo) | ((u32)f2h(hi) << 16);
#else
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

__device__ __forceinline__ void bg_mma(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1)
{
#ifdef LLVQ_HOST_BUILD
    bg_mma_host(d, a, b0, b1);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// cp.async of `bytes` (4, 8 or 16) from global to shared; `ok` false writes
// zeros and reads nothing.
template <unsigned BYTES>
__device__ __forceinline__ void bg_cp(void* dst, const void* src, bool ok)
{
#ifdef LLVQ_HOST_BUILD
    if (ok) __builtin_memcpy(dst, src, BYTES);
    else __builtin_memset(dst, 0, BYTES);
#else
    const u32 d = (u32)__cvta_generic_to_shared(dst);
    const u32 n = ok ? BYTES : 0u;
    if (BYTES == 16u)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(d), "l"(src), "n"(BYTES), "r"(n));
#endif
}

__device__ __forceinline__ void bg_commit()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <unsigned PENDING>
__device__ __forceinline__ void bg_wait()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
#endif
}

// Shared words a stage holds: the activation of N rows, then the weight words
// of BG_ROWS rows. The activation is f16 pairs with a row of KC / 2 + 4 words,
// which puts a B fragment's 32 lanes on 32 banks. Weight rows: KC / 16 words (Tetra) rounded up to 2 mod 4, so the 16
// rows a warp decodes sit on 16 distinct even banks; KC / 8 + 2 (int4).
#define BG_XW(kind, kc) ((kc) / 2u + 4u)
#define BG_WW(kind, kc) ((kind) == 0u ? ((kc) / 16u + 1u) / 4u * 4u + 2u : (kc) / 8u + 2u)
#define BG_STAGE(kind, n, kc) ((n) * BG_XW(kind, kc) + BG_ROWS * BG_WW(kind, kc))

template <unsigned NT, unsigned KC, unsigned KIND>
__device__ __forceinline__ void bg_issue(const BgSeg& s, u32* stage, u32 kc, u32 n_rows)
{
    constexpr u32 N = 8u * NT, XW = BG_XW(KIND, KC), WW = BG_WW(KIND, KC);
    const u32 t = threadIdx.x;
    const u32 c0 = kc * KC;
    u32* xs = stage;
    u32* ws = stage + N * XW;
    // The activation: 16-byte pieces of 8 halves.
    constexpr u32 PER = 8u;
    constexpr u32 PIECES = KC / PER;
    for (u32 i = t; i < N * PIECES; i += BG_THREADS) {
        const u32 n = i / PIECES, p = i % PIECES, col = c0 + p * PER;
        const bool ok = n < n_rows && col < s.d_in;
        const char* src = (const char*)s.x + ((unsigned long long)(ok ? n : 0u) * s.d_in + (ok ? col : 0u)) * 2u;
        bg_cp<16u>(xs + n * XW + p * 4u, src, ok);
    }
    // The weight words of the CTA's rows.
    const u32 lb = blockIdx.x - s.cta0, tiles = s.d_out / BG_ROWS;
    const u32 wrow0 = (lb % tiles) * BG_ROWS;
    if (KIND == 0u) {
        constexpr u32 WC = KC / 16u;  // words of a chunk's blocks
        const u32 w0 = kc * WC;
        for (u32 i = t; i < BG_ROWS * WC; i += BG_THREADS) {
            const u32 r = i / WC, k = i % WC;
            const bool ok = w0 + k < s.stride;
            const u32* src = (const u32*)s.w + (unsigned long long)(wrow0 + r) * s.stride + (ok ? w0 + k : 0u);
            bg_cp<4u>(ws + r * WW + k, src, ok);
        }
    } else {
        constexpr u32 WC = KC / 16u;  // 8-byte pieces: two words, 16 columns
        const u32 nw = s.d_in / 8u, w0 = kc * (KC / 8u);
        for (u32 i = t; i < BG_ROWS * WC; i += BG_THREADS) {
            const u32 r = i / WC, k = i % WC;
            const bool ok = w0 + 2u * k < nw;
            const u32* src = (const u32*)s.w + (unsigned long long)(wrow0 + r) * nw + (ok ? w0 + 2u * k : 0u);
            bg_cp<8u>(ws + r * WW + 2u * k, src, ok);
        }
    }
}

// One sub-step's A tile: lane l writes row l / 2, columns 24 (l % 2) .. +24 of
// the sub-step, as twelve f16 pairs.
template <unsigned KC, unsigned KIND>
__device__ __forceinline__ void bg_decode(const BgSeg& s, const BgArgs& a, const u32* ws, u32* aw,
                                          u32 kc, u32 sub, u32 wrow, u32 lane)
{
    constexpr u32 WW = BG_WW(KIND, KC);
    const u32 r = lane >> 1, h = lane & 1u, row = wrow + r;
    const u32* wr = ws + (wrow % BG_ROWS + r) * WW;
    u32 p[12];
    if (KIND == 0u) {
        const u32 jl = 2u * sub + h;               // block inside the chunk (chunks start on even blocks)
        const u32 j = kc * (KC / 24u) + jl;        // block of the row
        if (j < s.nblocks) {
            u32 lo, hi16;
            f1r_load(wr, jl, lo, hi16);
            F1rTables tab = { (const u32*)a.trows, (const unsigned char*)a.pref,
                              (const unsigned short*)a.br, (const unsigned char*)a.suf };
            float yv[24];
            tetra48_decode_f(lo, hi16, tab, (const float*)s.a0, (const float*)a.invnorm, yv);
            const float rs = ((const float*)s.a1)[row];
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) p[i] = bg_pack(yv[2u * i] * rs, yv[2u * i + 1u] * rs);
        } else if (j == s.nblocks && s.tail_w != 0u) {
            const unsigned short* tr = (const unsigned short*)s.tail + (unsigned long long)row * s.tail_w;
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) {
                const u32 c = 2u * i;
                const u32 v0 = c < s.tail_w ? tr[c] : 0u, v1 = c + 1u < s.tail_w ? tr[c + 1u] : 0u;
                p[i] = v0 | (v1 << 16);
            }
        } else {
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) p[i] = 0u;
        }
    } else {
        const u32 col0 = kc * KC + sub * BG_SUB + h * 24u;  // first column of the lane's three words
        const unsigned short* sc = (const unsigned short*)s.a0 + (unsigned long long)row * s.stride;
        const unsigned short* bi = (const unsigned short*)s.a1 + (unsigned long long)row * s.stride;
#pragma unroll
        for (u32 k = 0; k < 3u; ++k) {
            const u32 col = col0 + 8u * k;
            if (col < s.d_in) {
                const u32 wv = wr[sub * 6u + h * 3u + k];
                const u32 g = col >> s.nblocks;
                const float sv = h2f(sc[g]), bv = h2f(bi[g]);
#pragma unroll
                for (u32 e = 0; e < 4u; ++e) {
                    const float q0 = (float)((wv >> (8u * e)) & 0xfu), q1 = (float)((wv >> (8u * e + 4u)) & 0xfu);
                    p[4u * k + e] = bg_pack(sv * q0 + bv, sv * q1 + bv);
                }
            } else {
#pragma unroll
                for (u32 e = 0; e < 4u; ++e) p[4u * k + e] = 0u;
            }
        }
    }
    u32* ar = aw + r * BG_AS + h * 12u;
#pragma unroll
    for (u32 i = 0; i < 12u; ++i) ar[i] = p[i];
}

template <unsigned NT, unsigned KC, unsigned KIND>
__device__ __forceinline__ void bg_cta(const BgArgs& a, const BgSeg& s)
{
    constexpr u32 N = 8u * NT, XW = BG_XW(KIND, KC), STAGE = BG_STAGE(KIND, N, KC);
    extern __shared__ __align__(16) u32 bg_smem[];
    // Stage i at bg_smem + i * STAGE: an offset, not an array of two pointers,
    // which a runtime index would put in local memory.
    u32* as = bg_smem + 2u * STAGE;

    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 lb = blockIdx.x - s.cta0, tiles = s.d_out / BG_ROWS;
    const u32 tile = lb % tiles, sl = lb / tiles;
    const u32 wrow = tile * BG_ROWS + warp * 16u;  // the warp's first output row
    u32* aw = as + warp * 16u * BG_AS;
    const u32 n_rows = a.n_rows;

    float acc[NT][4];
#pragma unroll
    for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;

    const u32 k0 = sl * s.slice;
    const u32 k1 = k0 + s.slice < s.nk ? k0 + s.slice : s.nk;
    if (k0 < k1) {
        bg_issue<NT, KC, KIND>(s, bg_smem, k0, n_rows);
        bg_commit();
    }
    for (u32 kc = k0; kc < k1; ++kc) {
        const u32 cur = (kc - k0) & 1u;
        if (kc + 1u < k1) {
            bg_issue<NT, KC, KIND>(s, bg_smem + (cur ^ 1u) * STAGE, kc + 1u, n_rows);
            bg_commit();
            bg_wait<1u>();
        } else {
            bg_wait<0u>();
        }
        __syncthreads();
        const u32* xs = bg_smem + cur * STAGE;
        const u32* ws = xs + N * XW;
#pragma unroll 1
        for (u32 sub = 0; sub < KC / BG_SUB; ++sub) {
            bg_decode<KC, KIND>(s, a, ws, aw, kc, sub, wrow, lane);
            BG_WARP_SYNC();
#pragma unroll
            for (u32 ks = 0; ks < 3u; ++ks) {
                const u32* a0p = aw + gid * BG_AS + ks * 8u + tig;
                u32 af[4];
                af[0] = a0p[0];
                af[1] = a0p[8u * BG_AS];
                af[2] = a0p[4];
                af[3] = a0p[8u * BG_AS + 4u];
                const u32 k = sub * BG_SUB + ks * 16u + 2u * tig;  // column inside the chunk
#pragma unroll
                for (u32 nt = 0; nt < NT; ++nt) {
                    const u32 n = nt * 8u + gid;
                    const u32* xr = xs + n * XW + (k >> 1);
                    bg_mma(acc[nt], af, xr[0], xr[4]);
                }
            }
            BG_WARP_SYNC();
        }
        __syncthreads();
    }

    // Store: directly, or the partials, the ticket and the ordered fold.
    unsigned short* y = (unsigned short*)s.y;
    if (s.S == 1u) {
#pragma unroll
        for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + tig * 2u + (e & 1u);
                if (n < n_rows) y[(unsigned long long)n * s.d_out + row] = f2h(acc[nt][e]);
            }
        return;
    }
    float* part = (float*)s.part;
#pragma unroll
    for (u32 nt = 0; nt < NT; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) {
            const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + tig * 2u + (e & 1u);
            if (n < n_rows) part[((unsigned long long)sl * n_rows + n) * s.d_out + row] = acc[nt][e];
        }
    __threadfence();
    __syncthreads();
    u32* ticket = (u32*)a.ticket + s.tk0 + tile;
    unsigned tk = 0u;
    if (t == 0u) tk = atomicAdd(ticket, 1u);
    const int last = __syncthreads_or(t == 0u && tk == s.S - 1u);
    if (!last) return;
    __threadfence();
    const u32 row = tile * BG_ROWS + (t & (BG_ROWS - 1u));
    for (u32 n = t / BG_ROWS; n < n_rows; n += BG_THREADS / BG_ROWS) {
        float sum = 0.0f;
        for (u32 k = 0; k < s.S; ++k)
            sum += ((const volatile float*)part)[((unsigned long long)k * n_rows + n) * s.d_out + row];
        y[(unsigned long long)n * s.d_out + row] = f2h(sum);
    }
    if (t == 0u) *ticket = 0u;
}

// The projection a CTA belongs to, selected without indexing the parameter
// array at a runtime index (which would copy it to local memory).
__device__ __forceinline__ BgSeg bg_seg(const BgArgs& a)
{
    BgSeg s = a.seg[0];
#pragma unroll
    for (u32 i = 1; i < BG_MAXSEG; ++i)
        if (i < a.nseg && blockIdx.x >= a.seg[i].cta0) s = a.seg[i];
    return s;
}

// Shared bytes an entry asks for: two stages of the wider kind, the A tiles.
#define BG_SMEM_WORDS(n, kc)                                                                   \
    (2u * (BG_STAGE(0u, n, kc) > BG_STAGE(1u, n, kc) ? BG_STAGE(0u, n, kc) : BG_STAGE(1u, n, kc)) \
     + BG_WARPS * 16u * BG_AS)

#define BG_ENTRY(NAME, NT, KC)                                                    \
    extern "C" __global__ void __launch_bounds__(BG_THREADS) NAME(const BgArgs a)       \
    {                                                                             \
        const BgSeg s = bg_seg(a);                                                \
        if (s.kind == 0u) bg_cta<NT, KC, 0u>(a, s);                               \
        else bg_cta<NT, KC, 1u>(a, s);                                            \
    }

// Rows a launch carries, and the K chunk of each: two stages stay near 30 to
// 60 KB of shared memory.
BG_ENTRY(tv_bg_n8, 1u, 192u)
BG_ENTRY(tv_bg_n16, 2u, 192u)
BG_ENTRY(tv_bg_n32, 4u, 192u)
BG_ENTRY(tv_bg_n64, 8u, 96u)
BG_ENTRY(tv_bg_n128, 16u, 96u)
