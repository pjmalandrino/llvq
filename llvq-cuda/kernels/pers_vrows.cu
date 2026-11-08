// tv_vrows_*: the verify rows of a speculative round and the levers of B1, at
// 4E's geometry (2026-11-08, proofs/preregistration-spec-micro-2026-11-08.md,
// docs/recherche-debit-maths.md §8 and §11).
//
// Why: a round of k drafted tokens is checked by one pass over the weights
// with k + 1 rows. Its cost is the first gate of every speculative route at
// batch 1, and spec-gate (2026-11-07) read 1.95 greedy tokens for 2 rows. The
// study attributes that to the kernel, and projects about 0.30 ms an extra row
// at 4B with B1's integer dot. This file times the Tetra part of it alone.
//
// Geometry: pers_trows.cu's. One CTA of 512 threads a SM, each site's rows
// split over CTAs as 4E splits a phase and over warps statically, the sites one
// after the other with a CTA barrier between them, no grid barrier.
//
// Arms (the first template parameter; K = activations a pass):
//   V2     pers_tetra_row<24, false, true>, 4E v2's row. Owes the bits. The
//          control every speed is formed against.
//   K2     pers_trows.cu's K2 (one decode, two f32 activations). Owes the bits
//          on each activation. Replicates shaker-4e's 0.378.
//   PWORD  the words read with V2's loads and prefetch, no decode: the stream
//          rate of Tetra words at this geometry. Owes nothing.
//   LEAN   the relabeled word (tv_vrows_lean_relayout): its fields are
//          contiguous, the first rank row's index is one 12-bit field, and the
//          three Golay pattern bytes come from two tables by linearity,
//          c = GA[s8] ^ GB[b1, b3, b2] (tv_vrows_lean_tables). Same point, so
//          the f32 dot and the fold are V2's: owes the bits.
//   ROT    V2's decode and dot with lane rotation: the warp's rows form one
//          stream of items, item t on lane t mod 32, so no lane idles at the end
//          of a row. Row k's block j sits on physical lane (j + o_k) mod 32,
//          o_k = (k nblocks) mod 32, so each logical lane keeps its blocks and
//          their order. warp_sum needs no shuffle back: its XOR butterfly pairs
//          residue classes (c, c + 2^k) mod 2^(k+1) at every level, a set that a
//          cyclic shift of the lanes maps onto itself, so the rotated sum is
//          the same tree up to commutation, the same bits on every lane. Owes
//          the bits. Rows under 32 blocks run V2's row.
//   I16<K> LEAN's decode with B1's dot: each activation rounded to int16 a
//          block of 24 at staging (a bf16 scale rounded up, so |q| ≤ 32,767),
//          the dot in exact integers through twelve dp2a, one decode for K
//          activations. Owes no served bit; each activation's rows owe the bits
//          of I16<1> on that activation, whatever K.
//
// Composition: pers.rs::vrows_unit_text, the row-loop unit followed by this
// file, its own NVRTC module.

#define VROWS_V2 0u
#define VROWS_K2 1u
#define VROWS_PWORD 2u
#define VROWS_LEAN 3u
#define VROWS_ROT 4u
#define VROWS_I16 5u

#define VROWS_KMAX 4u

// Site words, 14 x u64, laid out by pers.rs::vrows_site_words:
//   [0] words  [1] gscale  [2] rscale  [3] tail (binary16)
//   [4..7] the rows on activation 0..3 (binary16), 0 past the arm's K
//   [8] d_out | nblocks << 32  [9] tail_w | row_stride_u32 << 32
//   [10..13] their f32 values before f2h, or 0 (the card's launches)
#define VROWS_SITE_WORDS 14u

// Shared bytes after the header: rows (16,384), prefixes (128), branches
// (2,048), suffixes (128), GA (256), GB (512), invnorm (128).
#define VROWS_TABLE_BYTES 19584u
// K2's second f32 activation, after the widest first one (pers_trows.cu).
#define VROWS_X2_AT 9728u

struct VLeanTab {
    const u32* rows;  // the served 4,096 rank rows
    const u32* ga;    // 64: the pattern bytes of (s8, 0, 0, 0), c1 | c2 << 8 | c3 << 16
    const u32* gb;    // 128: those of (0, b1, b2, b3), index b1 | b3 << 1 | b2 << 2 | g << 6, g ignored
};

// ---------------------------------------------------------------------------
// The relabeled word. A bijection of the 48 bits of llvq_f1rank.cuh's word:
//   lo   bits 0..11  R1 = r << 11 | i1   (the first rank row's index)
//        bits 12..22 i2      bit 23 p      bit 24 b1      bit 25 b3
//        bits 26..31 s8
//   hi16 bits 0..10  i3      bits 11..14 b2      bit 15 g (where pers_fold reads it)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void vrows_lean_word(u32 lo, u32 hi16, u32& nlo, u32& nhi)
{
    const u32 p = lo & 1u, r = (lo >> 1) & 1u, s8 = (lo >> 2) & 63u, b1 = (lo >> 8) & 1u;
    const u32 i1 = (lo >> 9) & 0x7ffu, b2 = (lo >> 20) & 15u;
    const u32 i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    const u32 b3 = (hi16 >> 3) & 1u, i3 = (hi16 >> 4) & 0x7ffu, g = (hi16 >> 15) & 1u;
    nlo = (r << 11) | i1 | (i2 << 12) | (p << 23) | (b1 << 24) | (b3 << 25) | (s8 << 26);
    nhi = i3 | (b2 << 11) | (g << 15);
}

// The three pattern bytes of a Golay path, by the trellis (f1r_v3_fetch's chain).
__device__ __forceinline__ u32 vrows_golay(const unsigned char* pref, const unsigned short* br, const unsigned char* suf,
                                           u32 s8, u32 b1, u32 b2, u32 b3)
{
    const u32 c1 = pref[2u * s8 + b1];
    const u32 m = br[16u * s8 + b2];
    const u32 c3 = suf[2u * ((m >> 8) & 63u) + b3];
    return c1 | ((m & 0xffu) << 8) | (c3 << 16);
}

// GA and GB, and the count of the 4,096 paths where GA ^ GB is not the trellis
// (0: the map is linear and the tables are exact). One thread.
extern "C" __global__ void tv_vrows_lean_tables(const unsigned char* __restrict__ pref,
                                                const unsigned short* __restrict__ br,
                                                const unsigned char* __restrict__ suf, u32* __restrict__ ga,
                                                u32* __restrict__ gb, u32* __restrict__ bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    for (u32 s8 = 0; s8 < 64u; ++s8) ga[s8] = vrows_golay(pref, br, suf, s8, 0u, 0u, 0u);
    for (u32 i = 0; i < 128u; ++i) gb[i] = vrows_golay(pref, br, suf, 0u, i & 1u, (i >> 2) & 15u, (i >> 1) & 1u);
    u32 n = 0u;
    for (u32 s8 = 0; s8 < 64u; ++s8)
        for (u32 i = 0; i < 64u; ++i)
            n += (ga[s8] ^ gb[i]) != vrows_golay(pref, br, suf, s8, i & 1u, (i >> 2) & 15u, (i >> 1) & 1u);
    bad[0] = n;
}

// The relayout, once at setup: every block of every row, read as the served
// stream reads it and written relabeled at the same place. A thread owns a
// pair of blocks, three u32, so no two threads write one word. A row with an
// odd count keeps the padding half of its last word.
extern "C" __global__ void tv_vrows_lean_relayout(const u32* __restrict__ src, u32 stride, u32* __restrict__ dst,
                                                  u32 d_out, u32 nblocks)
{
    const u32 npairs = (nblocks + 1u) >> 1;
    const unsigned long long n = (unsigned long long)d_out * npairs;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const u32 r = (u32)(i / npairs), m = (u32)(i - (unsigned long long)r * npairs);
        const u32* s = src + (size_t)r * stride;
        u32* d = dst + (size_t)r * stride;
        u32 lo, hi16, a, b;
        f1r_load(s, 2u * m, lo, hi16);
        vrows_lean_word(lo, hi16, a, b);
        d[3u * m] = a;
        if (2u * m + 1u < nblocks) {
            u32 lo1, hi1, a1, b1;
            f1r_load(s, 2u * m + 1u, lo1, hi1);
            vrows_lean_word(lo1, hi1, a1, b1);
            d[3u * m + 1u] = b | (a1 << 16);
            d[3u * m + 2u] = (a1 >> 16) | (b1 << 16);
        } else {
            d[3u * m + 1u] = b | (s[3u * m + 1u] & 0xffff0000u);
        }
    }
}

// The six quads of a relabeled word: f1r_v3_quads's values, from two table
// reads and one XOR where the served chain reads three dependent tables.
__device__ __forceinline__ void vrows_lean_quads(u32 lo, u32 hi16, const VLeanTab& t, u32 q[6])
{
    const u32 R1 = lo & 0xfffu;
    const u32 i2 = (lo >> 12) & 0x7ffu;
    const u32 p = (lo >> 23) & 1u;
    const u32 c = t.ga[lo >> 26] ^ t.gb[((lo >> 24) & 3u) | ((hi16 >> 9) & 0x7cu)];
    const bool mid = i2 < F1R_N0_MIXED;
    const u32 idx2 = mid ? i2 : i2 + (2048u - F1R_N0_MIXED);
    const u32 r3 = (p ^ (R1 >> 11) ^ (mid ? 0u : 1u)) & 1u;
    const u32 row1 = t.rows[R1], row2 = t.rows[idx2], row3 = t.rows[(r3 << 11) | (hi16 & 0x7ffu)];
    const F1rV3Tab v = f1r_v3_tables(p);
    f1r_v3_section(v, c & 0xffu, row1, q[0], q[1]);
    f1r_v3_section(v, (c >> 8) & 0xffu, row2, q[2], q[3]);
    f1r_v3_section(v, c >> 16, row3, q[4], q[5]);
}

// pers_block with the relabeled decode: the same 24 fmas and the same fold.
__device__ __forceinline__ float vrows_lean_block(float acc, u32 lo, u32 hi16, const VLeanTab& t, const float* xb,
                                                  const float* gscale, const float* invnorm)
{
    u32 q[6];
    vrows_lean_quads(lo, hi16, t, q);
    float a = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) a = __fmaf_rn(f1r_v3_float(q[i], j), xb[TETRA48_ORDER[4u * i + j]], a);
    }
    return pers_fold(acc, a, q, hi16, gscale, invnorm);
}

__device__ __forceinline__ float vrows_row_lean(const u32* wrow, const VLeanTab& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
{
    float acc = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        acc = vrows_lean_block(acc, lo, hi16, tab, xs + j * 24u, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    return warp_sum(acc);
}

// PWORD: V2's loads and prefetch, the words folded by XOR and nothing decoded.
__device__ __forceinline__ float vrows_row_pword(const u32* wrow, u32 nblocks, u32 lane)
{
    u32 acc = 0u;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        acc ^= lo ^ (hi16 << 7);
        lo = lo2;
        hi16 = hi2;
    }
    return warp_sum((float)(acc & 0xffffu));
}

// ROT: the `nr` rows lo_row + warp + k nw, k < nr, of one warp as one stream.
// At step s lane P holds item t = 32 s + P, block j of row k with t = k nb + j.
// Every row has at least 32 blocks, so a step meets at most two rows: the
// oldest unfinished one, kc, whose partials are in `cur`, and kc + 1 in `nxt`.
// Row kc's last item falls in step e = floor(((kc + 1) nb - 1) / 32); after it
// every lane's partial of row kc is final, physical lane P holding logical
// lane (P - (kc nb)) mod 32's, and warp_sum reduces them as they lie (above).
__device__ __forceinline__ void vrows_rot_rows(const u32* words, u32 stride, u32 r0, u32 nr, u32 nw,
                                               const F1rTables& tab, const float* gs, const float* inv, const float* xs,
                                               u32 nb, u32 tw, u32 lane, const float* rs, const unsigned short* tail,
                                               unsigned short* y, float* yf)
{
    const u32 total = nr * nb, nsteps = (total + 31u) >> 5;
    u32 k = 0u, j = lane;
    u32 lo = 0u, hi16 = 0u;
    if (lane < total) f1r_load(words + (size_t)r0 * stride, j, lo, hi16);
    float cur = 0.0f, nxt = 0.0f;
    u32 kc = 0u, end = nb;  // warp-uniform: the oldest unfinished row and its end item
    for (u32 s = 0; s < nsteps; ++s) {
        const u32 t = 32u * s + lane;
        u32 k2 = k, j2 = j + 32u;
        if (j2 >= nb) {
            j2 -= nb;
            ++k2;
        }
        u32 lo2 = 0u, hi2 = 0u;
        if (t + 32u < total) f1r_load(words + (size_t)(r0 + k2 * nw) * stride, j2, lo2, hi2);
        if (t < total) {
            const bool mine = k == kc;
            const float r = pers_block(mine ? cur : nxt, lo, hi16, tab, xs + j * 24u, gs, inv);
            if (mine) cur = r;
            else nxt = r;
        }
        if (end <= 32u * s + 32u) {
            const float v = warp_sum(cur);
            if (lane == 0u) {
                const u32 row = r0 + kc * nw;
                const float out = pers_epilogue(v, rs[row], tail + (size_t)row * tw, xs + nb * 24u, tw);
                y[row] = f2h(out);
                if (yf != nullptr) yf[row] = out;
            }
            cur = nxt;
            nxt = 0.0f;
            ++kc;
            end += nb;
        }
        k = k2;
        j = j2;
        lo = lo2;
        hi16 = hi2;
    }
}

// ---------------------------------------------------------------------------
// B1's activation: int16 a block of 24, trio order, a bf16 scale.
// An activation's area, at nb blocks: nb x 48 bytes of int16 (slot k of block
// b holds natural coordinate TETRA48_ORDER[k]), nb bf16 scales, padded to 16
// bytes, then 64 bytes for the f32 tail (at most 16 values).
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 vrows_area_bytes(u32 nb) { return ((50u * nb + 15u) & ~15u) + 64u; }

// The bf16 at or above a non-negative finite f32.
__device__ __forceinline__ u32 vrows_bf16_up(float s)
{
    const u32 b = __float_as_uint(s);
    return (b + 0xffffu) >> 16;
}

__device__ __forceinline__ int vrows_q16(float v, float inv)
{
    const int q = __float2int_rn(__fmul_rn(v, inv));
    return q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
}

// K activations into their areas; no barrier (the caller's). A thread a block
// of every activation, two activations at a time: their twelve 128-bit loads
// first, in one straight-line region so they are in flight together (four at
// a time would hold 96 registers of loads), then per activation the block's
// max, its scale, 24 roundings and three 128-bit stores. The tails stay f32.
__device__ __forceinline__ void vrows_stage_block(const float4* f, unsigned char* area, u32 nb, u32 b)
{
    float v[24];
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
        v[4u * i] = f[i].x;
        v[4u * i + 1u] = f[i].y;
        v[4u * i + 2u] = f[i].z;
        v[4u * i + 3u] = f[i].w;
    }
    float amax = 0.0f;
#pragma unroll
    for (u32 c = 0; c < 24u; ++c) amax = fmaxf(amax, fabsf(v[c]));
    const u32 s16 = vrows_bf16_up(__fmul_rn(amax, 1.0f / 32767.0f));
    const float s = __uint_as_float(s16 << 16);
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    u32 w[12];
#pragma unroll
    for (u32 k = 0; k < 12u; ++k) {
        const int lo = vrows_q16(v[TETRA48_ORDER[2u * k]], inv), hi = vrows_q16(v[TETRA48_ORDER[2u * k + 1u]], inv);
        w[k] = ((u32)lo & 0xffffu) | ((u32)hi << 16);
    }
    uint4* xq = reinterpret_cast<uint4*>(area);
#pragma unroll
    for (u32 i = 0; i < 3u; ++i) xq[3u * b + i] = uint4{w[4u * i], w[4u * i + 1u], w[4u * i + 2u], w[4u * i + 3u]};
    reinterpret_cast<unsigned short*>(area + 48u * nb)[b] = (unsigned short)s16;
}

template <u32 K>
__device__ __forceinline__ void vrows_stage_i16(const float* const* xa, unsigned char* act, u32 abytes, u32 nb, u32 tw)
{
    for (u32 b = threadIdx.x; b < nb; b += blockDim.x) {
#pragma unroll
        for (u32 a0 = 0; a0 < K; a0 += 2u) {
            const u32 n = K - a0 < 2u ? K - a0 : 2u;
            float4 f[2][6];
#pragma unroll
            for (u32 a = 0; a < n; ++a)
#pragma unroll
                for (u32 i = 0; i < 6u; ++i) f[a][i] = __ldcg(reinterpret_cast<const float4*>(xa[a0 + a] + 24u * b) + i);
#pragma unroll
            for (u32 a = 0; a < n; ++a) vrows_stage_block(f[a], act + (a0 + a) * abytes, nb, b);
        }
    }
#pragma unroll
    for (u32 a = 0; a < K; ++a) {
        float* xt = reinterpret_cast<float*>(act + a * abytes + abytes - 64u);
        for (u32 i = threadIdx.x; i < tw; i += blockDim.x) xt[i] = __ldcg(xa[a] + 24u * nb + i);
    }
}

// One block against K activations: the decode, the signed quads and the norm
// once; per activation twelve dp2a, a scale and one fold.
template <u32 K>
__device__ __forceinline__ void vrows_i16_block(float acc[K], u32 lo, u32 hi16, const VLeanTab& t,
                                                const unsigned char* areas, u32 abytes, u32 nb, u32 j,
                                                const float* gscale, const float* invnorm)
{
    u32 q[6];
    vrows_lean_quads(lo, hi16, t, q);
    int s[6];
    int n2 = 0;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
        s[i] = (int)(q[i] ^ 0x80808080u);
        n2 = __dp4a(s[i], s[i], n2);
    }
    const u32 m = ((u32)n2 >> 4) & (TETRA48_SHELLS - 1u);
    const float w = __fmul_rn(gscale[(hi16 >> 15) & 1u], invnorm[m]);
#pragma unroll
    for (u32 a = 0; a < K; ++a) {
        const unsigned char* ar = areas + a * abytes;
        const uint4* x8 = reinterpret_cast<const uint4*>(ar) + 3u * j;
        int dot = 0;
#pragma unroll
        for (u32 i = 0; i < 3u; ++i) {
            const uint4 v = x8[i];
            dot = __dp2a_lo((int)v.x, s[2u * i], dot);
            dot = __dp2a_hi((int)v.y, s[2u * i], dot);
            dot = __dp2a_lo((int)v.z, s[2u * i + 1u], dot);
            dot = __dp2a_hi((int)v.w, s[2u * i + 1u], dot);
        }
        const u32 s16 = reinterpret_cast<const unsigned short*>(ar + 48u * nb)[j];
        acc[a] = __fmaf_rn(__fmul_rn((float)dot, __uint_as_float(s16 << 16)), w, acc[a]);
    }
}

template <u32 K>
__device__ __forceinline__ void vrows_row_i16(float out[K], const u32* wrow, const VLeanTab& tab, const float* gscale,
                                              const float* invnorm, const unsigned char* areas, u32 abytes,
                                              u32 nblocks, u32 lane)
{
    float acc[K];
#pragma unroll
    for (u32 a = 0; a < K; ++a) acc[a] = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        vrows_i16_block<K>(acc, lo, hi16, tab, areas, abytes, nblocks, j, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
#pragma unroll
    for (u32 a = 0; a < K; ++a) out[a] = warp_sum(acc[a]);
}

template <u32 ARM, u32 K>
__device__ __forceinline__ void vrows_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                           const u32* __restrict__ rows, const unsigned char* __restrict__ prefixes,
                                           const unsigned short* __restrict__ branches,
                                           const unsigned char* __restrict__ suffixes, const u32* __restrict__ ga,
                                           const u32* __restrict__ gb, const float* __restrict__ invnorm,
                                           const float* __restrict__ x0, const float* __restrict__ x1,
                                           const float* __restrict__ x2, const float* __restrict__ x3)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    u32* t_ga = reinterpret_cast<u32*>(t_suf + 128);
    u32* t_gb = t_ga + 64;
    float* t_inv = reinterpret_cast<float*>(t_gb + 128);
    unsigned char* act = base + PERS_SMEM_HEADER + VROWS_TABLE_BYTES;
    float* xs = reinterpret_cast<float*>(act);
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = prefixes[i];
        t_suf[i] = suffixes[i];
        t_gb[i] = gb[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = branches[i];
    for (u32 i = threadIdx.x; i < 64u; i += blockDim.x) t_ga[i] = ga[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = invnorm[i];
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const VLeanTab lt{t_rows, t_ga, t_gb};
    const float* xa[VROWS_KMAX] = {x0, x1, x2, x3};
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    for (u32 s = 0; s < nsites; ++s) {
        const unsigned long long* d = sites + (size_t)s * VROWS_SITE_WORDS;
        const u32* words = reinterpret_cast<const u32*>(d[0]);
        const float* gs = reinterpret_cast<const float*>(d[1]);
        const float* rs = reinterpret_cast<const float*>(d[2]);
        const unsigned short* tail = reinterpret_cast<const unsigned short*>(d[3]);
        const u32 d_out = (u32)(d[8] & 0xffffffffull), nb = (u32)(d[8] >> 32);
        const u32 tw = (u32)(d[9] & 0xffffffffull), stride = (u32)(d[9] >> 32);
        unsigned short* y = reinterpret_cast<unsigned short*>(d[4]);
        float* yf = reinterpret_cast<float*>(d[10]);
        const u32 abytes = vrows_area_bytes(nb);
        if (ARM == VROWS_I16) {
            __syncthreads();  // no warp still reads the previous staging
            vrows_stage_i16<K>(xa, act, abytes, nb, tw);
            __syncthreads();
        } else {
            pers_stage_f32<24u, false>(x0, xs, nb, tw);  // barriers on both sides
            if (ARM == VROWS_K2) pers_stage_f32<24u, false>(x1, xs + VROWS_X2_AT, nb, tw);
        }
        const u32 lo = (u32)((unsigned long long)d_out * c / g), hi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        if (ARM == VROWS_ROT && nb >= 32u) {
            vrows_rot_rows(words, stride, lo + warp, pers_warp_rows(lo, hi, warp, nw), nw, tab, gs, t_inv, xs, nb, tw,
                           lane, rs, tail, y, yf);
        } else {
            for (u32 r = lo + warp; r < hi; r += nw) {
                const u32* wrow = words + (size_t)r * stride;
                const unsigned short* trow = tail + (size_t)r * tw;
                if (ARM == VROWS_I16) {
                    float acc[K];
                    vrows_row_i16<K>(acc, wrow, lt, gs, t_inv, act, abytes, nb, lane);
                    if (lane == 0u) {
#pragma unroll
                        for (u32 a = 0; a < K; ++a) {
                            const float* xt = reinterpret_cast<const float*>(act + a * abytes + abytes - 64u);
                            const float v = pers_epilogue(acc[a], rs[r], trow, xt, tw);
                            reinterpret_cast<unsigned short*>(d[4u + a])[r] = f2h(v);
                            if (d[10u + a] != 0ull) reinterpret_cast<float*>(d[10u + a])[r] = v;
                        }
                    }
                } else if (ARM == VROWS_K2) {
                    float acc2;
                    const float acc = trows_row_k2(wrow, tab, gs, t_inv, xs, xs + VROWS_X2_AT, nb, lane, acc2);
                    if (lane == 0u) {
                        const float v = pers_epilogue(acc, rs[r], trow, xs + nb * 24u, tw);
                        const float v2 = pers_epilogue(acc2, rs[r], trow, xs + VROWS_X2_AT + nb * 24u, tw);
                        y[r] = f2h(v);
                        reinterpret_cast<unsigned short*>(d[5])[r] = f2h(v2);
                        if (yf != nullptr) yf[r] = v;
                        if (d[11] != 0ull) reinterpret_cast<float*>(d[11])[r] = v2;
                    }
                } else if (ARM == VROWS_PWORD) {
                    const float v = vrows_row_pword(wrow, nb, lane);
                    if (lane == 0u) {
                        y[r] = f2h(v);
                        if (yf != nullptr) yf[r] = v;
                    }
                } else {
                    const float acc = ARM == VROWS_LEAN
                                          ? vrows_row_lean(wrow, lt, gs, t_inv, xs, nb, lane)
                                          : pers_tetra_row<24u, false, true>(wrow, tab, gs, t_inv, xs, nb, lane);
                    if (lane == 0u) {
                        const float v = pers_epilogue(acc, rs[r], trow, xs + nb * 24u, tw);
                        y[r] = f2h(v);
                        if (yf != nullptr) yf[r] = v;
                    }
                }
            }
        }
        __syncthreads();
    }
}

#define VROWS_KERNEL(sym, ARM_, K_)                                                                                \
    extern "C" __global__ void __launch_bounds__(512, 1) sym(                                                      \
        const unsigned long long* __restrict__ sites, u32 nsites, const u32* __restrict__ rows,                    \
        const unsigned char* __restrict__ prefixes, const unsigned short* __restrict__ branches,                   \
        const unsigned char* __restrict__ suffixes, const u32* __restrict__ ga, const u32* __restrict__ gb,        \
        const float* __restrict__ invnorm, const float* __restrict__ x0, const float* __restrict__ x1,             \
        const float* __restrict__ x2, const float* __restrict__ x3)                                                \
    {                                                                                                              \
        vrows_body<ARM_, K_>(sites, nsites, rows, prefixes, branches, suffixes, ga, gb, invnorm, x0, x1, x2, x3);  \
    }

VROWS_KERNEL(tv_vrows_v2, VROWS_V2, 1u)
VROWS_KERNEL(tv_vrows_k2, VROWS_K2, 2u)
VROWS_KERNEL(tv_vrows_pword, VROWS_PWORD, 1u)
VROWS_KERNEL(tv_vrows_lean, VROWS_LEAN, 1u)
VROWS_KERNEL(tv_vrows_rot, VROWS_ROT, 1u)
VROWS_KERNEL(tv_vrows_i16k1, VROWS_I16, 1u)
VROWS_KERNEL(tv_vrows_i16k2, VROWS_I16, 2u)
VROWS_KERNEL(tv_vrows_i16k3, VROWS_I16, 3u)
VROWS_KERNEL(tv_vrows_i16k4, VROWS_I16, 4u)
