// pers_4et.cu: the tree verify pass (docs/plan-verify-tree.md, stage S0).
//
// One cooperative launch verifies a tree of R <= P4T_RMAX drafted rows. Row r
// is the token toks[r] at position pos0 + depth[r]. Its ancestors, the root
// first, are path[r][0 .. depth[r]), and path[r][depth[r]] = r. Rows sit in
// any slot order: the root need not be row 0, a parent need not precede its
// child. Each row's argmax is written. Each row's K and V go to the tree
// buffer (tk, tv, a slot per row), not to the cache, since siblings share a
// position. The next launch moves the accepted rows into the cache in its
// first phase: slot mv[i] of the previous launch to position move_pos0 + i.
//
// The contract is identity, not 4E's bits. Row r's results (its argmax, its
// logits, its residual, its K and V) depend on row r and its ancestors alone:
// not on R, not on its slot, not on the other rows, not on which CTA runs
// what. A speculative decode then emits the greedy decode of this pass at
// R = 1, token for token (plan section 6). Against 4E the logits move by the
// arithmetic below, which the plan bounds at 1e-2 relative RMS on the host
// fixture and at +-0.12 % paired perplexity on the card (S2).
//
// The arithmetic:
//
//   * Projections (q, k, v, o, gate, up, down, the head) on the tensor cores,
//     tv_batch_gemm.cu's form: each block decoded once into binary16 weights
//     with its scales folded in, a Tetra weight f16(v x ((gscale[g] x
//     invnorm[m]) x rscale[row])), its tail as two more binary16 columns, an
//     int4 weight f16(s q + b); the activation binary16; mma.m16n8k16 with f32
//     accumulation, the k16 steps in increasing column order. A projection's
//     d_in is cut into S slices of 48-column sub-steps, S fixed by the host
//     for each projection, never by R; each slice's f32 partial is folded in
//     slice order 0 .. S - 1 by the warp that draws the tile's last ticket.
//     Row r of the output is column r of the MMA's B and D: no other column
//     enters it.
//   * Prologues (ln1, ln2, the down's input, the final norm, the Tetra o's
//     rotation): a row on one CTA, row r on CTA r mod grid, with 4E's scalar
//     arithmetic (candle's RMSNorm, rot_apply's rotation) in f32, then rounded
//     to binary16 for the MMA. 4E has every CTA compute them over the whole
//     vector; here each runs once a row, behind one more grid wait.
//   * Attention: 4E's (llvq_pers.cuh section 10), the row's keys in logical
//     order. Position p < pos0 comes from the cache, position pos0 + d from
//     the tree buffer's slot path[r][d]. Its tiles, units and splits are
//     functions of the length pos0 + depth[r] + 1 alone, so a row equals the
//     same row decoded later as a root, once its ancestors were moved.
//   * The head: int4 g64 rows on the tensor cores, binary16 logits, 4E's
//     argmax key a row, the CTA's maxima then one global atomicMax a row.
//
// Waits: 1 (embedding and move) + 9 a layer (+ 1 a layer with a Tetra o) +
// 2 (final norm, head). n_bar_4et below.
//
// Shared memory, after the 256-byte header and 4E's tables (18,816 bytes):
// an area of area_floats floats, which each phase reuses. Prologue: the row in
// f32. Projection: the warps' A tiles, then two cp.async stages of the
// activation chunk (R rows x 96 columns, binary16) and the unit's weight
// words. Attention: 4E's stages and states. The CTA's argmax maxima sit in
// the area's last 2 x P4T_RMAX floats. p4t_area_floats gives the size.
//
// Host: llvq-cuda/tests/host_pers_tree.inc runs it through host_grid.h with
// the MMA modelled from the PTX fragment layout and cp.async as a synchronous
// copy (command 'Y' of host_pers.cpp). Built only with PERS_4E_ROWS 7, the
// v9 module (it reads the int4 rows on 4E v9's tiled words), after
// pers_4e.cu. Not in any served module.
//
// Tree-tiled weights (job N1, arm a2): a module that defines P4T_TILED 1
// emits the pers_4et_tiled_* symbols, which read every projection's weights
// from a copy laid out by p4t_tile_words: for a 16-row tile and a 48-column
// sub-step, its rows' words contiguous (Tetra 48 u32: row r's two blocks at
// 3 r, the row layout's 48-bit packing; int4 96 u32: row r's 6 logical words
// at 6 r), so a warp's chunk is one stream copied in 16-byte cp.async.cg
// pieces. Same words, same decode, same MMA: the tiled pass's bits are the
// untiled pass's at the same slices. Without P4T_TILED the module emits the
// untiled symbols only, whose code is S1's.

#ifndef P4T_TILED
#define P4T_TILED 0
#endif

#ifndef LLVQ_PERS_CUH
#include "llvq_pers.cuh"
#endif

#if PERS_4E_ROWS < 7
#error "pers_4et.cu reads the int4 rows on tiled words and attention's ring: build it with PERS_4E_ROWS 7 (4E v9)"
#endif

#define P4T_RMAX 32u               // rows a launch
#define P4T_SUB 48u                // columns a sub-step: two Tetra blocks, six int4 words
#define P4T_CH 2u                  // sub-steps a staged chunk
#define P4T_KC (P4T_SUB * P4T_CH)  // columns a staged chunk: 96
#define P4T_AS 28u                 // u32 words of a decoded row in a warp's A tile: 24 + 4 pad
#define P4T_XW (P4T_KC / 2u + 4u)  // u32 words of a staged activation row (binary16 pairs), 16-byte multiple
#define P4T_WWT 6u                 // u32 words of a staged Tetra row: 3 a sub-step
#define P4T_WWQ 14u                // u32 words of a staged int4 row: 6 a sub-step, 2 pad
#define P4T_STORE 0u               // epilogues: y = f16(sum)
#define P4T_RESID 1u               // y = res + f16(sum), binary16 add
#define P4T_HEAD 2u                // logit f16(sum), the row's argmax key

struct Pers4tLaunch {
    u32 bar_base;                // arrivals before this launch
    u32 pos0;                    // the root's position
    u32 n_rows;                  // R, 1..P4T_RMAX
    u32 area_floats;             // the shared area after the tables, floats, even
    u32 n_move;                  // rows of the previous launch moved into the cache
    u32 move_pos0;               // the cache position of the first
    u32 part_stride;             // attention's split states: floats a row
    u32 ldx;                     // xin's row stride in halves: >= every projection's d_in, a multiple of 8
    u32 split_qkv, split_o;      // d_in slices of each projection phase, fixed for the model
    u32 split_gu, split_dn;
    u32 split_head, pad;
    const u32* toks;             // [R]
    const u32* depth;            // [R] row r sits at position pos0 + depth[r]
    const unsigned char* path;   // [R][P4T_RMAX] path[r][d]: row r's ancestor at depth d; path[r][depth[r]] = r
    const u32* mv;               // [n_move] the previous launch's slots, root first
    u32* ring;                   // [R] each row's argmax
    unsigned short* logits;      // null, or [R][vocab]: the binary16 logits (the host check)
    unsigned short* x_a;         // [R][hidden] the residual, ping
    unsigned short* x_b;         // [R][hidden] pong
    unsigned short* xin;         // [R][ldx] the binary16 input of the next projection phase
    unsigned short* xn;          // [R][hidden] ln1 in the natural basis: the int4 v rows' input
    unsigned short* q_raw;       // [R][n_heads x 128]
    unsigned short* k_raw;       // [R][n_kv x 128]
    unsigned short* v_raw;       // [R][n_kv x 128]
    unsigned short* q_out;       // [R][n_heads x 128] after norm and RoPE
    unsigned short* ctx;         // [R][n_heads x 128]
    unsigned short* g_raw;       // [R][inter] the gate rows' outputs
    unsigned short* u_raw;       // [R][inter] the up rows' outputs
    unsigned short* tk;          // [layers][P4T_RMAX][n_kv x 128] each slot's K
    unsigned short* tv;          // [layers][P4T_RMAX][n_kv x 128] each slot's V
    float* part;                 // [R][part_stride] attention's split states
    float* fpart;                // the projections' slice partials, [S][R][d_out] each projection with S > 1
    u32* tickets;                // [P4T_RMAX][n_kv] attention's split fold, zero at rest
    u32* ftickets;               // [d_out / 16] each projection of a phase: the slice fold's, zero at rest
    unsigned long long* slots;   // [P4T_RMAX] argmax keys, cleared at launch start
    unsigned long long* stamps;  // null, or [1 + 2 n_bar_4et][grid]: each CTA's clock at its start (event 0),
                                 // on reaching wait e (event 2e - 1) and on leaving it (event 2e)
};

// The waits of one launch, for the host's barrier base.
__device__ __forceinline__ u32 n_bar_4et(u32 n_layers, u32 n_tetra_o)
{
    return 3u + 9u * n_layers + n_tetra_o;
}

// Floats of the projection phases' share of the area: the A tiles, two
// stages of rs staged rows (8 a tile of rows, up to 32).
#define P4T_MMA_FLOATS(nw, rs) ((nw) * 16u * P4T_AS + 2u * ((rs) * P4T_XW + 16u * (nw) * P4T_WWQ))

// The area a launch needs at `nw` warps a CTA, `rs` staged rows: the widest
// row a prologue holds, attention's share, the projections', then the maxima.
__device__ __forceinline__ u32 p4t_area_floats(u32 nw, u32 rs, u32 widest, u32 groups)
{
    u32 f = widest;
    const u32 attn = (PERS_ATTN_SMEM(groups) + 3u) / 4u;
    if (attn > f) f = attn;
    if (P4T_MMA_FLOATS(nw, rs) > f) f = P4T_MMA_FLOATS(nw, rs);
    return f + 2u * P4T_RMAX;
}

// ---- the projection phases ---------------------------------------------------

// One projection of a phase.
struct P4tSeg {
    const u32* w;                // Tetra: words, row-strided; int4: d_in / 8 words a row (tiled or served order)
    const float* gscale;         // Tetra [2]
    const float* rscale;         // Tetra [d_out]
    const unsigned short* tail;  // Tetra [d_out][tail_w]
    const unsigned short* sc;    // int4 [d_out][gpr]
    const unsigned short* bi;    // int4 [d_out][gpr]
    const unsigned short* x;     // binary16 [R][ldx]
    unsigned short* y;           // binary16 [R][ldy]
    const unsigned short* res;   // P4T_RESID: binary16 [R][ldy]
    float* part;                 // S > 1: [S][R][d_out]
    u32* ticket;                 // S > 1: [d_out / 16]
    u32 kind;                    // 0 Tetra, 1 int4
    u32 tiled;                   // int4: the words in 4E v9's tiled order (pers_q4_tile_index)
    u32 stride;                  // Tetra: u32 words a row
    u32 nblocks, tail_w;         // Tetra
    u32 gshift, gpr;             // int4: log2 of the group, groups a row
    u32 d_in, d_out, ldx, ldy;
    u32 S, epi;
};

__device__ __forceinline__ P4tSeg p4t_seg_tetra(const u32* w, const float* gs, const float* rs,
                                                const unsigned short* tail, u32 stride, u32 nblocks, u32 tail_w,
                                                u32 d_out)
{
    P4tSeg g{};
    g.kind = 0u;
    g.w = w;
    g.gscale = gs;
    g.rscale = rs;
    g.tail = tail;
    g.stride = stride;
    g.nblocks = nblocks;
    g.tail_w = tail_w;
    g.d_in = nblocks * 24u + tail_w;
    g.d_out = d_out;
    g.S = 1u;
    return g;
}

__device__ __forceinline__ P4tSeg p4t_seg_q4(const u32* w, const unsigned short* sc, const unsigned short* bi,
                                             u32 d_in, u32 d_out, u32 gshift, u32 tiled)
{
    P4tSeg g{};
    g.kind = 1u;
    g.w = w;
    g.sc = sc;
    g.bi = bi;
    g.tiled = tiled;
    g.gshift = gshift;
    g.gpr = d_in >> gshift;
    g.d_in = d_in;
    g.d_out = d_out;
    g.S = 1u;
    return g;
}

// The input, the output, the epilogue and the slices of a projection.
__device__ __forceinline__ void p4t_io(P4tSeg& g, const unsigned short* x, u32 ldx, unsigned short* y, u32 ldy,
                                       const unsigned short* res, u32 epi, u32 S)
{
    g.x = x;
    g.ldx = ldx;
    g.y = y;
    g.ldy = ldy;
    g.res = res;
    g.epi = epi;
    g.S = S;
}

// The partials and tickets of a phase's projections, one after the other.
__device__ __forceinline__ void p4t_place(P4tSeg& g, const Pers4tLaunch& L, unsigned long long& pofs, u32& tofs)
{
    g.part = L.fpart + pofs;
    g.ticket = L.ftickets + tofs;
    if (g.S > 1u) pofs += (unsigned long long)g.S * L.n_rows * g.d_out;
    tofs += g.d_out >> 4;
}

#ifdef LLVQ_HOST_BUILD
void p4t_mma_host(float* d, const u32* a, u32 b0, u32 b1);
#endif

__device__ __forceinline__ void p4t_mma(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1)
{
#ifdef LLVQ_HOST_BUILD
    p4t_mma_host(d, a, b0, b1);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Two floats as a binary16 pair, lo in the low half, each rounded to nearest.
__device__ __forceinline__ u32 p4t_pack(float lo, float hi)
{
#ifdef LLVQ_HOST_BUILD
    return (u32)f2h(lo) | ((u32)f2h(hi) << 16);
#else
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// cp.async of BYTES (4 or 16) global to shared; `ok` false writes zeros and
// reads nothing. The host copies at once.
template <unsigned BYTES>
__device__ __forceinline__ void p4t_cp(void* dst, const void* src, bool ok)
{
#ifdef LLVQ_HOST_BUILD
    if (ok) __builtin_memcpy(dst, src, BYTES);
    else __builtin_memset(dst, 0, BYTES);
#else
    const u32 d = (u32)__cvta_generic_to_shared(dst);
    const u32 n = ok ? BYTES : 0u;
    if (BYTES == 16u)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(d), "l"(src), "n"(BYTES), "r"(n)
                     : "memory");
#endif
}

template <unsigned PENDING>
__device__ __forceinline__ void p4t_cp_wait()
{
#ifndef LLVQ_HOST_BUILD
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING) : "memory");
#endif
}

// The tiles of 16 rows a projection has, and the CTA units over them: a unit
// is nw tiles (one a warp) of one slice.
__device__ __forceinline__ u32 p4t_nblk(const P4tSeg& g, u32 nw)
{
    return ((g.d_out >> 4) + nw - 1u) / nw;
}

// u32 words of one tile's sub-step in the tree-tiled copy: Tetra 16 rows x 3,
// int4 16 rows x 6.
#define P4T_TSW 48u
#define P4T_TSQ 96u

// Word i of the tree-tiled copy of a projection: tile T = i / (nsub SW), its
// sub-step s, element e. Tetra (kind 0): row 16 T + e / 3, word 3 s + e % 3
// of the row layout (stride u32 a row); int4 (kind 1): row 16 T + e / 6,
// logical word 6 s + e % 6 of the row's `stride` words, read at its tiled
// place when `served_tiled` (4E v9's pers_q4_tile_index). Words past a row's
// end are zero. The host twin builds its copies through this function, the
// card through p4t_tile_words below.
__device__ __forceinline__ u32 p4t_tile_word(const u32* src, u32 kind, u32 stride, u32 nsub, u32 served_tiled,
                                             unsigned long long i)
{
    const u32 SW = kind == 0u ? P4T_TSW : P4T_TSQ;
    const u32 e = (u32)(i % SW);
    const unsigned long long ts = i / SW;
    const u32 s = (u32)(ts % nsub), T = (u32)(ts / nsub);
    const u32 per = kind == 0u ? 3u : 6u;
    const u32 r = e / per, wi = per * s + e % per;
    const u32 row = 16u * T + r;
    if (wi >= stride) return 0u;
    const u32 phys = (kind != 0u && served_tiled != 0u) ? pers_q4_tile_index(wi, stride) : wi;
    return src[(unsigned long long)row * stride + phys];
}

// Sub-steps [sub0, sub0 + cnt) into a stage: the activation of the R rows
// (16-byte pieces of 8 halves, rows past R and columns past d_in zero), then
// the weight words of the unit's rows from urow0, in the row's logical word
// order (an int4 word read at its tiled place). TILED: the weights from the
// tree-tiled copy, each warp's tile a contiguous run of cnt sub-steps in
// 16-byte pieces, staged in the copy's order at the warp's own place.
template <bool TILED>
__device__ __forceinline__ void p4t_issue(const P4tSeg& g, u32* stage, u32 sub0, u32 cnt, u32 urow0, u32 R, u32 RS)
{
    const u32 t = threadIdx.x, T = blockDim.x, U = 16u * (T >> 5);
    u32* xs = stage;
    u32* ws = stage + RS * P4T_XW;
    const u32 PP = P4T_KC / 8u;
    for (u32 i = t; i < RS * PP; i += T) {
        const u32 n = i / PP, p = i - n * PP, col = sub0 * P4T_SUB + 8u * p;
        const bool ok = n < R && p < 6u * cnt && col < g.d_in;
        const unsigned short* src = g.x + (unsigned long long)(ok ? n : 0u) * g.ldx + (ok ? col : 0u);
        p4t_cp<16u>(xs + n * P4T_XW + 4u * p, src, ok);
    }
    if (TILED) {
        const u32 SW = g.kind == 0u ? P4T_TSW : P4T_TSQ, PC = SW / 4u;
        const u32 WS = 16u * (g.kind == 0u ? P4T_WWT : P4T_WWQ);
        const u32 tsub = (g.d_in + P4T_SUB - 1u) / P4T_SUB, ntiles = g.d_out >> 4, tile0 = urow0 >> 4;
        for (u32 i = t; i < (U >> 4) * P4T_CH * PC; i += T) {
            const u32 w = i / (P4T_CH * PC), rem = i - w * (P4T_CH * PC), ks = rem / PC, p = rem - ks * PC;
            const u32 tile = tile0 + w;
            const bool ok = tile < ntiles && ks < cnt;
            const u32* src = g.w + ((unsigned long long)(ok ? tile : 0u) * tsub + (ok ? sub0 + ks : 0u)) * SW + 4u * p;
            p4t_cp<16u>(ws + w * WS + ks * SW + 4u * p, src, ok);
        }
    } else if (g.kind == 0u) {
        const u32 WC = 3u * P4T_CH;
        for (u32 i = t; i < U * WC; i += T) {
            const u32 r = i / WC, k = i - r * WC, row = urow0 + r, wi = 3u * sub0 + k;
            const bool ok = row < g.d_out && k < 3u * cnt && wi < g.stride;
            p4t_cp<4u>(ws + r * P4T_WWT + k, g.w + (unsigned long long)(ok ? row : 0u) * g.stride + (ok ? wi : 0u), ok);
        }
    } else {
        const u32 WC = 6u * P4T_CH, nw = g.d_in >> 3;
        for (u32 i = t; i < U * WC; i += T) {
            const u32 r = i / WC, k = i - r * WC, row = urow0 + r, wi = 6u * sub0 + k;
            const bool ok = row < g.d_out && k < 6u * cnt && wi < nw;
            const u32 phys = g.tiled ? pers_q4_tile_index(wi, nw) : wi;
            p4t_cp<4u>(ws + r * P4T_WWQ + k, g.w + (unsigned long long)(ok ? row : 0u) * nw + (ok ? phys : 0u), ok);
        }
    }
    pers_cp_commit();
}

// One sub-step of the warp's A tile: lane l decodes row l / 2, half l % 2 of
// the 48 columns (one Tetra block or three int4 words), twelve binary16
// pairs in column order. `wsw` is the warp's 16 staged rows, `ks` the
// sub-step's index in its chunk. TILED: `wsw` is the warp's tile in the
// tree-tiled order, sub-step ks at P4T_TSW ks (Tetra) or P4T_TSQ ks (int4).
template <bool TILED>
__device__ __forceinline__ void p4t_decode(const P4tSeg& g, const F1rTables& tab, const float* t_inv, const u32* wsw,
                                           u32* aw, u32 sub, u32 ks, u32 wrow, u32 lane)
{
    const u32 r = lane >> 1, h = lane & 1u, row = wrow + r;
    u32 p[12];
    if (g.kind == 0u) {
        const u32 j = 2u * sub + h;  // the row's block
        if (j < g.nblocks) {
            u32 lo, hi16;
            f1r_load(TILED ? wsw + P4T_TSW * ks + 3u * r : wsw + r * P4T_WWT + 3u * ks, h, lo, hi16);
            u32 q[6];
            f1r_v3_quads(lo, hi16, tab, q);
            const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
            const float s = __fmul_rn(__fmul_rn(g.gscale[(hi16 >> 15) & 1u], t_inv[m]), g.rscale[row]);
            float y[24];
#pragma unroll
            for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
                for (u32 k = 0; k < 4u; ++k) y[TETRA48_ORDER[4u * i + k]] = __fmul_rn(f1r_v3_float(q[i], k), s);
            }
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) p[i] = p4t_pack(y[2u * i], y[2u * i + 1u]);
        } else if (j == g.nblocks && g.tail_w != 0u) {
            const unsigned short* tr = g.tail + (unsigned long long)row * g.tail_w;
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) {
                const u32 c = 2u * i;
                const u32 v0 = c < g.tail_w ? tr[c] : 0u, v1 = c + 1u < g.tail_w ? tr[c + 1u] : 0u;
                p[i] = v0 | (v1 << 16);
            }
        } else {
#pragma unroll
            for (u32 i = 0; i < 12u; ++i) p[i] = 0u;
        }
    } else {
        const u32 col0 = sub * P4T_SUB + 24u * h;
        const unsigned short* sc = g.sc + (unsigned long long)row * g.gpr;
        const unsigned short* bi = g.bi + (unsigned long long)row * g.gpr;
#pragma unroll
        for (u32 k = 0; k < 3u; ++k) {
            const u32 col = col0 + 8u * k;
            if (col < g.d_in) {
                const u32 wv = TILED ? wsw[P4T_TSQ * ks + 6u * r + 3u * h + k] : wsw[r * P4T_WWQ + 6u * ks + 3u * h + k];
                const u32 gi = col >> g.gshift;
                const u32 sb = sc[gi], bb = bi[gi];
#pragma unroll
                for (u32 e = 0; e < 4u; ++e)
                    p[4u * k + e] = p4t_pack(pers_q4_deq((wv >> (8u * e)) & 0xfu, sb, bb),
                                             pers_q4_deq((wv >> (8u * e + 4u)) & 0xfu, sb, bb));
            } else {
#pragma unroll
                for (u32 e = 0; e < 4u; ++e) p[4u * k + e] = 0u;
            }
        }
    }
    u32* ar = aw + r * P4T_AS + 12u * h;
#pragma unroll
    for (u32 i = 0; i < 12u; ++i) ar[i] = p[i];
}

// Row n's output at output row `row`, the epilogue of its projection.
__device__ __forceinline__ void p4t_epi(const P4tSeg& g, const Pers4tLaunch& L, const Pers4eArgs* A, u32 n, u32 row,
                                        float v, unsigned long long* cmax)
{
    const unsigned short h = f2h(v);
    if (g.epi == P4T_HEAD) {
        if (L.logits != nullptr) L.logits[(unsigned long long)n * A->vocab + row] = h;
        if (pers_is_nan16(h)) atomicExch(A->nan_flag, 1u);
        else atomicMax(cmax + n, pers_tie_key(h, row));
    } else if (g.epi == P4T_RESID) {
        g.y[(unsigned long long)n * g.ldy + row] = exact_hadd(__ldcg(g.res + (unsigned long long)n * g.ldy + row), h);
    } else {
        g.y[(unsigned long long)n * g.ldy + row] = h;
    }
}

// The warp's accumulators of slice sl: stored through the epilogue when the
// projection has one slice; otherwise stored as the slice's partial, and the
// warp that draws the tile's last ticket folds the S partials in slice order
// and runs the epilogue. D's element e of n8 tile nt is output row wrow + gid
// + 8 (e / 2), activation row 8 nt + 2 tig + e % 2.
__device__ __forceinline__ void p4t_store(const P4tSeg& g, const Pers4tLaunch& L, const Pers4eArgs* A, u32 sl,
                                          const float (&acc)[4][4], u32 wrow, u32 lane, unsigned long long* cmax)
{
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, gid = lane >> 2, tig = lane & 3u;
    if (g.S == 1u) {
#pragma unroll
        for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);
                if (nt < NT && n < R) p4t_epi(g, L, A, n, row, acc[nt][e], cmax);
            }
        return;
    }
#pragma unroll
    for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) {
            const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);
            if (nt < NT && n < R) g.part[((unsigned long long)sl * R + n) * g.d_out + row] = acc[nt][e];
        }
    __threadfence();  // each lane's partials before the warp's ticket
    __syncwarp();
    u32 last = 0u;
    if (lane == 0u) last = atomicAdd(g.ticket + (wrow >> 4), 1u) == g.S - 1u ? 1u : 0u;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last == 0u) return;
    __threadfence();
    for (u32 i = lane; i < 16u * R; i += 32u) {
        const u32 n = i >> 4, row = wrow + (i & 15u);
        float v = __ldcg(g.part + (unsigned long long)n * g.d_out + row);
        for (u32 k = 1u; k < g.S; ++k) v = __fadd_rn(v, __ldcg(g.part + ((unsigned long long)k * R + n) * g.d_out + row));
        p4t_epi(g, L, A, n, row, v, cmax);
    }
    __syncwarp();
    if (lane == 0u) g.ticket[wrow >> 4] = 0u;
}

// One unit: tiles [blk nw, blk nw + nw) of g (one a warp; a warp past the
// last tile stages with the others and computes nothing), slice sl. The
// slice's sub-steps run in chunks of P4T_CH through two stages, chunk c + 1
// in flight while chunk c is decoded and multiplied.
template <bool TILED>
__device__ __forceinline__ void p4t_unit(const P4tSeg& g, u32 sl, u32 blk, const Pers4tLaunch& L,
                                         const Pers4eArgs* A, const F1rTables& tab, const float* t_inv, u32* area,
                                         unsigned long long* cmax)
{
    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, NW = blockDim.x >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, RS = 8u * NT;
    const u32 WW = g.kind == 0u ? P4T_WWT : P4T_WWQ;
    const u32 STAGE = RS * P4T_XW + 16u * NW * WW;
    u32* aw = area + warp * 16u * P4T_AS;
    u32* st0 = area + NW * 16u * P4T_AS;
    const u32 tile = blk * NW + warp;
    const bool has = tile < (g.d_out >> 4);
    const u32 urow0 = blk * NW * 16u, wrow = tile * 16u;
    const u32 nsub = (g.d_in + P4T_SUB - 1u) / P4T_SUB;
    const u32 sub_lo = sl * nsub / g.S, sub_hi = (sl + 1u) * nsub / g.S;
    const u32 nch = (sub_hi - sub_lo + P4T_CH - 1u) / P4T_CH;
    float acc[4][4];
#pragma unroll
    for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;
    if (nch > 0u) p4t_issue<TILED>(g, st0, sub_lo, sub_hi - sub_lo < P4T_CH ? sub_hi - sub_lo : P4T_CH, urow0, R, RS);
    for (u32 c = 0; c < nch; ++c) {
        const u32 sub0 = sub_lo + c * P4T_CH;
        const u32 cnt = sub_hi - sub0 < P4T_CH ? sub_hi - sub0 : P4T_CH;
        const u32* xs = st0 + (c & 1u) * STAGE;
        if (c + 1u < nch) {
            const u32 s1 = sub0 + P4T_CH;
            p4t_issue<TILED>(g, st0 + ((c + 1u) & 1u) * STAGE, s1, sub_hi - s1 < P4T_CH ? sub_hi - s1 : P4T_CH, urow0, R, RS);
            p4t_cp_wait<1u>();
        } else {
            p4t_cp_wait<0u>();
        }
        __syncthreads();
        const u32* ws = xs + RS * P4T_XW + warp * 16u * WW;
        for (u32 ks = 0; ks < cnt; ++ks) {
            if (has) p4t_decode<TILED>(g, tab, t_inv, ws, aw, sub0 + ks, ks, wrow, lane);
            __syncwarp();
            if (has) {
#pragma unroll
                for (u32 kk = 0; kk < 3u; ++kk) {
                    const u32* a0p = aw + gid * P4T_AS + kk * 8u + tig;
                    const u32 af[4] = {a0p[0], a0p[8u * P4T_AS], a0p[4], a0p[8u * P4T_AS + 4u]};
                    const u32 kw = (ks * P4T_SUB + kk * 16u + 2u * tig) >> 1;
#pragma unroll
                    for (u32 nt = 0; nt < 4u; ++nt) {
                        if (nt < NT) {
                            const u32* xr = xs + (nt * 8u + gid) * P4T_XW + kw;
                            p4t_mma(acc[nt], af, xr[0], xr[4]);
                        }
                    }
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);
}

// A projection phase over up to three projections: the units (projection,
// block of tiles, slice) dealt over the grid, unit u on CTA u mod grid. The
// slices of a block are neighbouring units, so they run on neighbouring CTAs
// at the same time and their tickets arrive in no fixed order: only the fold
// itself fixes the order of the sum. Dealt slice after slice, the last slice
// always arrived last, and the host gate could not see a fold that followed
// the arrivals (tree-verify-s0-host-2026-11-09, mutant fold_arrival).
template <bool TILED>
__device__ __forceinline__ void p4t_phase(const P4tSeg& s0, const P4tSeg& s1, const P4tSeg& s2, u32 nseg,
                                          const Pers4tLaunch& L, const Pers4eArgs* A, const F1rTables& tab,
                                          const float* t_inv, u32* area, unsigned long long* cmax)
{
    const u32 NW = blockDim.x >> 5;
    const u32 n0 = p4t_nblk(s0, NW) * s0.S;
    const u32 n1 = nseg > 1u ? p4t_nblk(s1, NW) * s1.S : 0u;
    const u32 n2 = nseg > 2u ? p4t_nblk(s2, NW) * s2.S : 0u;
    for (u32 u = blockIdx.x; u < n0 + n1 + n2; u += gridDim.x) {
        const u32 i = u < n0 ? 0u : (u < n0 + n1 ? 1u : 2u);
        const u32 lu = i == 0u ? u : (i == 1u ? u - n0 : u - n0 - n1);
        const P4tSeg& g = i == 0u ? s0 : (i == 1u ? s1 : s2);
        p4t_unit<TILED>(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);
    }
}

// ---- the prologues -------------------------------------------------------------

// candle's RMSNorm of row x (binary16, written in this launch) with alpha:
// `nat` (non-null) gets the binary16 results, S their f32 values, each
// sign-flipped where `signs` has its bit (null: none). Ends with a barrier.
__device__ __forceinline__ void p4t_ln(const unsigned short* x, const unsigned short* alpha, const u32* signs,
                                       float eps, u32 w, float* S, unsigned short* nat)
{
    const float sc = exact_rms_scale<0>(x, w, eps);
    for (u32 i = threadIdx.x; i < w; i += blockDim.x) {
        const unsigned short h = exact_rms_apply(sc, __ldcg(x + i), alpha[i]);
        if (nat != nullptr) nat[i] = h;
        const float f = h2f(h);
        S[i] = (signs != nullptr && ((signs[i >> 5] >> (i & 31u)) & 1u)) ? -f : f;
    }
    __syncthreads();
}

// S rounded to binary16 into dst; ends with a barrier (S is reused).
__device__ __forceinline__ void p4t_out(const float* S, u32 w, unsigned short* dst)
{
    for (u32 i = threadIdx.x; i < w; i += blockDim.x) dst[i] = f2h(S[i]);
    __syncthreads();
}

// ---- attention ------------------------------------------------------------------

// Step st of a job into a ring slot: pers_attn_issue's chunks, position p of
// the row from the cache below pos0 and from the tree buffer's slot
// path_r[p - pos0] from pos0 on (tkh, tvh: the layer's buffer at head h).
__device__ __forceinline__ void p4t_attn_issue(uint4* stage, const unsigned short* kh, const unsigned short* vh,
                                               const unsigned short* tkh, const unsigned short* tvh, u32 slot_stride,
                                               const unsigned char* path_r, u32 pos0, u32 base, u32 st, u32 nsteps,
                                               u32 tpu, u32 lr)
{
    if (st < nsteps) {
        for (u32 idx = threadIdx.x; idx < PERS_STAGE_U4; idx += blockDim.x) {
            const u32 rem = idx % (PERS_STAGE * 16u), pl = rem >> 4;
            const u32 tl = pl / PERS_TT, u = tl / tpu, t = tl - u * tpu;
            const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + pl % PERS_TT;
            const bool isk = idx < PERS_STAGE * 16u;
            const bool valid = p < lr;
            const unsigned short* src = kh;
            if (valid && p < pos0) src = (isk ? kh : vh) + (unsigned long long)p * PERS_HD;
            else if (valid) src = (isk ? tkh : tvh) + (unsigned long long)path_r[p - pos0] * slot_stride;
            pers_cp16z(stage + idx, reinterpret_cast<const uint4*>(src) + (rem & 15u), valid);
        }
    }
    pers_cp_commit();
}

// Attention for every row: pers_4ev.cu's job body (4E's, form 6's ring),
// the jobs of row 0, then row 1, ... dealt over the grid. Row r's length is
// pos0 + depth[r] + 1, its keys in logical order; it reads q_out and writes
// ctx, its split states and its tickets at its own offsets.
__device__ __forceinline__ void p4t_attention(const Pers4eArgs* A, const Pers4tLaunch& L, const unsigned short* kc,
                                              const unsigned short* vc, const unsigned short* tk_l,
                                              const unsigned short* tv_l, unsigned char* smem)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 n_heads = A->n_heads, groups = A->groups, cap = A->cap, n_kv = A->n_kv;
    const float c = A->c;
    const u32 s_stride = (cap + PERS_SPLIT - 1u) / PERS_SPLIT;
    const u32 nhp = (groups + 1u) / 2u, ntasks = PERS_UPS * nhp;
    uint4* stage = reinterpret_cast<uint4*>(smem);
    float* ust = reinterpret_cast<float*>(smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);  // [UPS][groups][PSTRIDE]
    float* qs = ust + PERS_UPS * groups * PERS_PSTRIDE;                                  // [groups][128]
    u32* hdr = pers_hdr();
    u32 total = 0u;
    for (u32 r = 0; r < L.n_rows; ++r) total += n_kv * ((L.pos0 + L.depth[r] + 1u + PERS_SPLIT - 1u) / PERS_SPLIT);
    for (u32 job = blockIdx.x; job < total; job += gridDim.x) {
        u32 r = 0u, j = job;
        for (;;) {
            const u32 nj = n_kv * ((L.pos0 + L.depth[r] + 1u + PERS_SPLIT - 1u) / PERS_SPLIT);
            if (j < nj) break;
            j -= nj;
            ++r;
        }
        const u32 l = L.pos0 + L.depth[r] + 1u;
        const u32 ns = (l + PERS_SPLIT - 1u) / PERS_SPLIT;
        const u32 s = j / n_kv, h = j - s * n_kv;
        const unsigned short* q = L.q_out + (size_t)r * n_heads * PERS_HD;
        unsigned short* out = L.ctx + (size_t)r * n_heads * PERS_HD;
        float* part = L.part + (size_t)r * L.part_stride;
        u32* ticket = L.tickets + r * n_kv;
        const unsigned char* path_r = L.path + r * P4T_RMAX;
        const u32 base = s * PERS_SPLIT;
        const u32 nu = (l - base + PERS_UNIT - 1u) / PERS_UNIT;
        const u32 tpu = nu >= 3u ? 1u : (nu == 2u ? 2u : 4u);
        const u32 tiles_raw = (l - base + PERS_TT - 1u) / PERS_TT;
        const u32 tiles = tiles_raw < PERS_UNIT / PERS_TT ? tiles_raw : PERS_UNIT / PERS_TT;
        const u32 nsteps = (tiles + tpu - 1u) / tpu;
        const unsigned short* kh = kc + (unsigned long long)h * cap * PERS_HD;
        const unsigned short* vh = vc + (unsigned long long)h * cap * PERS_HD;
        const unsigned short* tkh = tk_l + h * PERS_HD;
        const unsigned short* tvh = tv_l + h * PERS_HD;
        for (u32 st = 0; st < PERS_ATTN_RING; ++st)
            p4t_attn_issue(stage + st * PERS_STAGE_U4, kh, vh, tkh, tvh, n_kv * PERS_HD, path_r, L.pos0, base, st,
                           nsteps, tpu, l);
        for (u32 i = threadIdx.x; i < PERS_UPS * groups * PERS_PSTRIDE; i += blockDim.x) {
            const u32 k = i % PERS_PSTRIDE;
            ust[i] = k == PERS_HD ? PERS_NEG_INF : 0.0f;
        }
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) qs[i] = h2f(__ldcg(q + h * groups * PERS_HD + i));
        for (u32 st = 0; st < nsteps; ++st) {
            uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;
            pers_cp_wait_ring();
            __syncthreads();
            for (u32 task = warp; task < ntasks; task += nwarps) {
                const u32 u = task / nhp, hp = task - u * nhp;
                if (u * tpu >= PERS_UPS) continue;
                const u32 g = 2u * hp + (lane >> 4);
                const bool own = g < groups;
                const u32 ge = own ? g : groups - 1u;
                for (u32 t = 0; t < tpu; ++t) {
                    const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
                    if (p0 >= l || (st * tpu + t) * PERS_TT >= PERS_UNIT) break;
                    const u32 nv = l - p0 < PERS_TT ? l - p0 : PERS_TT;
                    const u32 sl = (u * tpu + t) * PERS_TT;
                    pers_attn_tile(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own, sbuf + sl * 16u,
                                   sbuf + (PERS_STAGE + sl) * 16u, nv, c, lane);
                }
            }
            __syncthreads();
            p4t_attn_issue(sbuf, kh, vh, tkh, tvh, n_kv * PERS_HD, path_r, L.pos0, base, st + PERS_ATTN_RING, nsteps,
                           tpu, l);
        }
        pers_cp_wait_all();
        __syncthreads();
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) {
            const u32 g = i / PERS_HD, k = i - g * PERS_HD;
            float sm = PERS_NEG_INF, sd = 0.0f, sa = 0.0f;
            for (u32 uu = 0; uu < PERS_UPS; ++uu) {
                const float* row = ust + (uu * groups + g) * PERS_PSTRIDE;
                pers_attn_merge1(sm, sd, sa, row[PERS_HD], row[PERS_HD + 1u], row[k]);
            }
            const u32 hq = h * groups + g;
            if (ns == 1u) {
                out[hq * PERS_HD + k] = f2h(__fdiv_rn(sa, sd));
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
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0u) hdr[PERS_HDR_TICKET] = atomicAdd(ticket + h, 1u) == ns - 1u ? 1u : 0u;
            __syncthreads();
            if (hdr[PERS_HDR_TICKET] != 0u) {
                __threadfence();
                float* cmd = reinterpret_cast<float*>(stage);
                for (u32 b0 = 0; b0 < groups * PERS_HD; b0 += blockDim.x) {
                    const u32 i = b0 + threadIdx.x;
                    const bool act = i < groups * PERS_HD;
                    const u32 g = act ? i / PERS_HD : 0u, k = act ? i - g * PERS_HD : 0u;
                    float am = PERS_NEG_INF, ad = 0.0f, aa = 0.0f;
                    for (u32 c0 = 0; c0 < ns; c0 += PERS_ATTN_CH) {
                        __syncthreads();
                        for (u32 t = threadIdx.x; t < 2u * PERS_ATTN_CH * groups; t += blockDim.x) {
                            const u32 jj = t / (2u * groups), gg = (t >> 1) % groups;
                            float val = 0.0f;
                            if (c0 + jj < ns)
                                val = __ldcg(part + (((unsigned long long)h * s_stride + c0 + jj) * groups + gg) * PERS_PSTRIDE + PERS_HD + (t & 1u));
                            cmd[t] = val;
                        }
                        float ba[PERS_ATTN_CH];
#pragma unroll
                        for (u32 jj = 0; jj < PERS_ATTN_CH; ++jj) {
                            ba[jj] = 0.0f;
                            if (act && c0 + jj < ns)
                                ba[jj] = __ldcg(part + (((unsigned long long)h * s_stride + c0 + jj) * groups + g) * PERS_PSTRIDE + k);
                        }
                        __syncthreads();
                        if (act) {
#pragma unroll
                            for (u32 jj = 0; jj < PERS_ATTN_CH; ++jj)
                                if (c0 + jj < ns)
                                    pers_attn_merge1(am, ad, aa, cmd[(jj * groups + g) * 2u], cmd[(jj * groups + g) * 2u + 1u], ba[jj]);
                        }
                    }
                    if (act) out[(h * groups + g) * PERS_HD + k] = f2h(__fdiv_rn(aa, ad));
                }
                if (threadIdx.x == 0u) ticket[h] = 0u;
            }
        }
        __syncthreads();
    }
}

// ---- the pass -------------------------------------------------------------------

template <int FORM, bool TILED>
__device__ __forceinline__ void pers_4et_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* S = t_inv + 32;  // the area, L.area_floats floats
    u32* area = reinterpret_cast<u32*>(S);
    unsigned long long* cmax = reinterpret_cast<unsigned long long*>(S + (L.area_floats - 2u * P4T_RMAX));
    const u32 H = A->hidden, G = gridDim.x, c0 = blockIdx.x, R = L.n_rows, NL = A->n_layers;
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 dq = A->n_heads * 128u, dkv = A->n_kv * 128u;
    // Refused before any wait, by every CTA alike: a row count out of range,
    // an area smaller than this launch needs.
    {
        const u32 I = A->layers[0].mlp.inter;
        u32 widest = H > I ? H : I;
        if (dq > widest) widest = dq;
        if (R == 0u || R > P4T_RMAX || (L.area_floats & 1u) != 0u ||
            L.area_floats < p4t_area_floats(nw, 8u * ((R + 7u) >> 3), widest, A->groups)) {
            if (c0 == 0u && threadIdx.x == 0u) atomicExch(A->abort_flag, 1u);
            return;
        }
    }
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = A->pref[i];
        t_suf[i] = A->suf[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
    __syncthreads();
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    u32 nb = 0u;
#define P4T_WAIT()                                                                                   \
    do {                                                                                             \
        if (L.stamps != nullptr) {                                                                   \
            __syncthreads();                                                                         \
            if (threadIdx.x == 0u) L.stamps[(size_t)(2u * nb + 1u) * G + c0] = pers_now();           \
        }                                                                                            \
        if (!pers_grid_wait(A->bar, L.bar_base + (++nb) * G, A->abort_flag)) return;                 \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)(2u * nb) * G + c0] = pers_now(); \
    } while (0)
    // Phase stamps (a measurement launch, L.stamps non-null): event 0 at the
    // CTA's start; for its e-th wait, event 2e - 1 once the whole CTA reached
    // it (its work of the phase done) and event 2e on leaving it, thread 0 of
    // each CTA. Nothing a row computes reads the clock.
    if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[c0] = pers_now();
    // The argmax keys and every ticket start at zero; the first wait's fence
    // orders these stores before any CTA's atomics.
    if (c0 == 0u) {
        for (u32 i = threadIdx.x; i < P4T_RMAX; i += blockDim.x) L.slots[i] = 0ull;
        for (u32 i = threadIdx.x; i < P4T_RMAX * A->n_kv; i += blockDim.x) L.tickets[i] = 0u;
    }
    // The previous launch's accepted rows into the cache, every layer.
    for (u32 it = c0; it < L.n_move * NL; it += G) {
        const u32 i = it / NL, l = it - i * NL, pos = L.move_pos0 + i;
        const Pers4eLayer& Y = A->layers[l];
        const unsigned short* sk = L.tk + ((unsigned long long)l * P4T_RMAX + L.mv[i]) * dkv;
        const unsigned short* sv = L.tv + ((unsigned long long)l * P4T_RMAX + L.mv[i]) * dkv;
        for (u32 e = threadIdx.x; e < dkv; e += blockDim.x) {
            const unsigned long long at = ((unsigned long long)(e >> 7) * A->cap + pos) * 128u + (e & 127u);
            Y.kc[at] = __ldcg(sk + e);
            Y.vc[at] = __ldcg(sv + e);
        }
    }
    for (u32 r = 0; r < R; ++r)
        pers_embed(A->emb_w, A->emb_s, A->emb_b, L.toks[r], H, A->emb_gpr, L.x_a + (size_t)r * H, H * c0 / G,
                   H * (c0 + 1u) / G);
    P4T_WAIT();
    for (u32 l = 0; l < NL; ++l) {
        const Pers4eLayer& Y = A->layers[l];
        const Pers4eQkv& N = Y.qkv;
        const SegArgs& M = Y.mlp;
        const u32 I = M.inter;
        unsigned short* tk_l = L.tk + (unsigned long long)l * P4T_RMAX * dkv;
        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;
        // ln1: the natural values for the int4 v rows, the signed and
        // rotated ones for the Tetra q and k rows
        for (u32 r = c0; r < R; r += G) {
            p4t_ln(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, A->eps, H, S, L.xn + (size_t)r * H);
            pers_rot_finish(S, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
        }
        P4T_WAIT();
        {
            P4tSeg sq = p4t_seg_tetra(N.qw, N.qgs, N.qrs, N.qtl, N.q_stride, N.nblocks, N.tail_w, N.q_dout);
            P4tSeg sk = p4t_seg_tetra(N.kw, N.kgs, N.krs, N.ktl, N.k_stride, N.nblocks, N.tail_w, N.k_dout);
            P4tSeg sv = p4t_seg_q4(N.vw, N.vsc, N.vbi, N.d_in, N.v_dout, 7u, 1u);
            p4t_io(sq, L.xin, L.ldx, L.q_raw, dq, nullptr, P4T_STORE, L.split_qkv);
            p4t_io(sk, L.xin, L.ldx, L.k_raw, dkv, nullptr, P4T_STORE, L.split_qkv);
            p4t_io(sv, L.xn, H, L.v_raw, dkv, nullptr, P4T_STORE, L.split_qkv);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sq, L, po, to);
            p4t_place(sk, L, po, to);
            p4t_place(sv, L, po, to);
            p4t_phase<TILED>(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // q/k-norm and RoPE at the row's position; K and V into its slot
        {
            const u32 per = A->n_heads + 2u * A->n_kv;
            for (u32 w = c0 + G * warp; w < R * per; w += G * nw) {
                const u32 r = w / per, hh = w - r * per, pos = L.pos0 + L.depth[r];
                const unsigned short* cr = A->cosr + pos * 64u;
                const unsigned short* sr = A->sinr + pos * 64u;
                if (hh < A->n_heads) {
                    pers_head_norm_rope<FORM>(L.q_raw + (size_t)r * dq + hh * 128u, Y.q_norm, cr, sr, A->eps,
                                              L.q_out + (size_t)r * dq + hh * 128u, lane);
                } else if (hh < A->n_heads + A->n_kv) {
                    const u32 h = hh - A->n_heads;
                    pers_head_norm_rope<FORM>(L.k_raw + (size_t)r * dkv + h * 128u, Y.k_norm, cr, sr, A->eps,
                                              tk_l + (size_t)r * dkv + h * 128u, lane);
                } else {
                    const u32 h = hh - A->n_heads - A->n_kv;
                    for (u32 i = lane; i < 128u; i += 32u)
                        tv_l[(size_t)r * dkv + h * 128u + i] = __ldcg(L.v_raw + (size_t)r * dkv + h * 128u + i);
                }
            }
        }
        P4T_WAIT();
        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));
        P4T_WAIT();
        // o and the residual: x_b = x_a + o(ctx)
        {
            P4tSeg so;
            if (Y.ot != nullptr) {
                const Pers4eOT& O = *Y.ot;
                for (u32 r = c0; r < R; r += G) {
                    __syncthreads();
                    pers_rot_load(L.ctx + (size_t)r * dq, O.signs, S, dq);
                    pers_rot_finish(S, O.small, dq, O.m, O.k, O.inv);
                    p4t_out(S, dq, L.xin + (size_t)r * L.ldx);
                }
                P4T_WAIT();
                so = p4t_seg_tetra(O.t.words, O.t.gscale, O.t.rscale, O.t.tail, O.t.stride_u32, O.t.nblocks,
                                   O.t.tail_w, O.t.d_out);
                p4t_io(so, L.xin, L.ldx, L.x_b, H, L.x_a, P4T_RESID, L.split_o);
            } else {
                so = p4t_seg_q4(Y.ow, Y.osc, Y.obi, dq, H, 7u, 1u);
                p4t_io(so, L.ctx, dq, L.x_b, H, L.x_a, P4T_RESID, L.split_o);
            }
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(so, L, po, to);
            p4t_phase<TILED>(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // ln2, signed and rotated: the gate and up rows' input
        for (u32 r = c0; r < R; r += G) {
            p4t_ln(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.eps, H, S, nullptr);
            pers_rot_finish(S, M.small_h, H, M.m_h, M.k_h, M.inv_h);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
        }
        P4T_WAIT();
        {
            P4tSeg sg = p4t_seg_tetra(M.gate.words, M.gate.gscale, M.gate.rscale, M.gate.tail, M.gate.stride_u32,
                                      M.gate.nblocks, M.gate.tail_w, M.gate.d_out);
            P4tSeg su = p4t_seg_tetra(M.up.words, M.up.gscale, M.up.rscale, M.up.tail, M.up.stride_u32, M.up.nblocks,
                                      M.up.tail_w, M.up.d_out);
            p4t_io(sg, L.xin, L.ldx, L.g_raw, I, nullptr, P4T_STORE, L.split_gu);
            p4t_io(su, L.xin, L.ldx, L.u_raw, I, nullptr, P4T_STORE, L.split_gu);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sg, L, po, to);
            p4t_place(su, L, po, to);
            p4t_phase<TILED>(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // the down's input: silu(g) u, signed and rotated for a Tetra down,
        // natural for an int4 one
        for (u32 r = c0; r < R; r += G) {
            const unsigned short* gr = L.g_raw + (size_t)r * I;
            const unsigned short* ur = L.u_raw + (size_t)r * I;
            unsigned short* dst = L.xin + (size_t)r * L.ldx;
            for (u32 i = threadIdx.x; i < I; i += blockDim.x) {
                const unsigned short a = exact_silu_mul(M.silu, __ldcg(gr + i), __ldcg(ur + i));
                if (Y.tetra_down) {
                    const float f = h2f(a);
                    S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;
                } else {
                    dst[i] = a;
                }
            }
            if (Y.tetra_down) {
                pers_rot_finish(S, M.small_i, I, M.m_i, M.k_i, M.inv_i);
                p4t_out(S, I, dst);
            } else {
                __syncthreads();
            }
        }
        P4T_WAIT();
        // the down and the residual: x_a = x_b + down
        {
            P4tSeg sd = Y.tetra_down
                            ? p4t_seg_tetra(M.down.words, M.down.gscale, M.down.rscale, M.down.tail,
                                            M.down.stride_u32, M.down.nblocks, M.down.tail_w, M.down.d_out)
                            : p4t_seg_q4(M.q4w, M.q4s, M.q4b, I, H, 7u, 1u);
            p4t_io(sd, L.xin, L.ldx, L.x_a, H, L.x_b, P4T_RESID, L.split_dn);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sd, L, po, to);
            p4t_phase<TILED>(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
    }
    // the final norm, natural: the head's input
    for (u32 r = c0; r < R; r += G)
        p4t_ln(L.x_a + (size_t)r * H, A->final_norm, nullptr, A->eps, H, S, L.xin + (size_t)r * L.ldx);
    P4T_WAIT();
    // the head: each row's argmax key through the CTA's maxima
    {
        for (u32 i = threadIdx.x; i < P4T_RMAX; i += blockDim.x) cmax[i] = 0ull;
        __syncthreads();
        P4tSeg sh = p4t_seg_q4(A->head_w, A->head_s, A->head_b, H, A->vocab, 6u, 0u);
        p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);
        unsigned long long po = 0ull;
        u32 to = 0u;
        p4t_place(sh, L, po, to);
        p4t_phase<TILED>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
        __syncthreads();
        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);
    }
    P4T_WAIT();
    if (c0 == 0u && threadIdx.x < R) {
        if (pers_ld_acquire(A->nan_flag) != 0u) return;  // the host reads the flag
        const unsigned long long k = __ldcg(L.slots + threadIdx.x);
        const u32 br = 1023u - (u32)((k >> 22) & 0x3FFu);
        L.ring[threadIdx.x] = ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);
    }
#undef P4T_WAIT
}

#define PERS_4ET(sym, FORM_, TILED_, THREADS_)                                                           \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,      \
                                                                   const Pers4tLaunch L)                 \
    {                                                                                                    \
        pers_4et_body<FORM_, TILED_>(A, L);                                                              \
    }

#if !P4T_TILED || defined(LLVQ_HOST_BUILD)
PERS_4ET(pers_4et_rows_f0_t512, 0, false, 512)
PERS_4ET(pers_4et_rows_f1_t512, 1, false, 512)
PERS_4ET(pers_4et_rows_f2_t512, 2, false, 512)
#endif

#if P4T_TILED || defined(LLVQ_HOST_BUILD)
// The tiled stage of a warp holds its chunk in the place the untiled one had.
static_assert(P4T_CH * P4T_TSW <= 16u * P4T_WWT && P4T_CH * P4T_TSQ <= 16u * P4T_WWQ, "a tiled chunk outgrows the stage");
PERS_4ET(pers_4et_tiled_f0_t512, 0, true, 512)
PERS_4ET(pers_4et_tiled_f1_t512, 1, true, 512)
PERS_4ET(pers_4et_tiled_f2_t512, 2, true, 512)

// The tree-tiled copy of a projection, built once on the card: n = tiles x
// nsub x SW words (p4t_tile_word).
extern "C" __global__ void p4t_tile_words(const u32* src, u32* dst, u32 kind, u32 stride, u32 nsub, u32 served_tiled,
                                          unsigned long long n)
{
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        dst[i] = p4t_tile_word(src, kind, stride, nsub, served_tiled, i);
}
#endif
