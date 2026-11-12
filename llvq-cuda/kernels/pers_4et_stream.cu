// pers_4et_stream.cu: arm a3 of the tree verify (job N2), the stream unit.
//
// Appended after pers_4et.cu, which it leaves byte for byte as it is, so the
// modules of arms a0, a1 and a2 keep their text and their sha256. A module
// that defines P4T_STREAM 1 (pers.rs, p4t_stream_unit_text: also P4T_TILED 1,
// for p4t_tile_words) emits the pers_4et_stream_* symbols. They read the
// tree-tiled copies a2 reads, with a2's arithmetic: binary16 weights with
// folded scales decoded into the warp's A tile, the binary16 activation,
// mma.m16n8k16, f32 accumulation in increasing column order, p4t_store and
// the slice fold. Under the same slices their bits are a2's.
//
// What changes is the unit's control flow. a2 runs a slice in chunks of two
// 48-column sub-steps, each chunk staged by the whole CTA (activation and
// words through cp.async), then a wait and two __syncthreads. Here:
//
//   1. The CTA stages the slice's columns of the R activation rows once, in
//      16-byte cp.async.cg pieces (rows past R and columns past d_in zero),
//      one wait, one __syncthreads. No weight word is staged.
//   2. Each warp then runs alone over the slice's sub-steps: every lane reads
//      its own words straight from its tile's stream in the tree-tiled copy
//      (Tetra: words 3 r + h and 3 r + h + 1 of the sub-step's 48, which are
//      f1r_load's two loads at block h of tile row r; int4: words 6 r + 3 h +
//      k of its 96), decodes into the warp's A tile, __syncwarp, the three
//      k16 MMA steps for each n8 tile of rows with B read from the staged
//      slice, __syncwarp. No __syncthreads and no cp.async in that loop.
//   3. The words of sub-step s + 1 are loaded into registers before
//      sub-step s decodes, one sub-step ahead, plain loads: 4E v9's row loop
//      (pers_tetra_row<24, false, true>, the next block's words in flight
//      while a block decodes; pers_q4_row_tile, the next 128-word tile's
//      uint4 in flight). A lane's int4 words sit at 12-byte offsets, so they
//      are three 4-byte loads, not 4E's one 16-byte load.
//   4. One __syncthreads at the unit's end, before the next unit restages.
//
// Shared memory: the launch's area is a2's (p4t_area_floats, the same
// symbol bytes). It holds the warps' A tiles, then one staged slice of RS
// rows of 24 ns + 4 u32 (ns the slice's sub-steps; 4 pad words keep the B
// reads free of bank conflicts and every row 16-byte aligned), then the
// argmax maxima. p4t_stream_cap gives the longest slice that fits at RS =
// P4T_RMAX, a function of the area and the warps, never of R; the host plan
// (pers.rs, p4t_splits_stream) keeps every slice within it, and a launch
// whose slice exceeds it is refused through the abort flag.

#ifndef P4T_STREAM
#define P4T_STREAM 0
#endif

#if P4T_STREAM || defined(LLVQ_HOST_BUILD)

// The sub-steps a slice may hold at most: an area of `area_floats` floats,
// `nw` warps' A tiles, P4T_RMAX staged rows of 24 ns + 4 words, the maxima.
__device__ __forceinline__ u32 p4t_stream_cap(u32 area_floats, u32 nw)
{
    const u32 fixed = nw * 16u * P4T_AS + 2u * P4T_RMAX;
    if (area_floats < fixed) return 0u;
    const u32 per = (area_floats - fixed) / P4T_RMAX;
    return per < 4u ? 0u : (per - 4u) / 24u;
}

// The slice's ns sub-steps from sub_lo of the R activation rows into xs,
// row n at n XS, 16-byte pieces of 8 halves; rows past R and columns past
// d_in zero.
__device__ __forceinline__ void p4t_stream_stage(const P4tSeg& g, u32* xs, u32 sub_lo, u32 ns, u32 R, u32 RS, u32 XS)
{
    const u32 t = threadIdx.x, T = blockDim.x, PP = 6u * ns;
    for (u32 i = t; i < RS * PP; i += T) {
        const u32 n = i / PP, p = i - n * PP, col = sub_lo * P4T_SUB + 8u * p;
        const bool ok = n < R && col < g.d_in;
        const unsigned short* src = g.x + (unsigned long long)(ok ? n : 0u) * g.ldx + (ok ? col : 0u);
        p4t_cp<16u>(xs + n * XS + 4u * p, src, ok);
    }
    pers_cp_commit();
}

// A lane's words of one sub-step of its tile's stream, `wp` already at the
// lane's offset: Tetra two (f1r_load's), int4 three.
__device__ __forceinline__ void p4t_stream_load(u32 kind, const u32* wp, u32& c0, u32& c1, u32& c2)
{
    c0 = wp[0];
    c1 = wp[1];
    c2 = kind == 0u ? 0u : wp[2];
}

// p4t_decode's arithmetic on a lane's words in registers: lane l decodes row
// l / 2, half l % 2 of sub-step `sub` into the warp's A tile.
__device__ __forceinline__ void p4t_decode_stream(const P4tSeg& g, const F1rTables& tab, const float* t_inv, u32 c0,
                                                  u32 c1, u32 c2, u32* aw, u32 sub, u32 wrow, u32 lane)
{
    const u32 r = lane >> 1, h = lane & 1u, row = wrow + r;
    u32 p[12];
    if (g.kind == 0u) {
        const u32 j = 2u * sub + h;  // the row's block
        if (j < g.nblocks) {
            // f1r_load at block h of the row's three words, c0 and c1 its two loads
            const u32 lo = h ? ((c0 >> 16) | (c1 << 16)) : c0;
            const u32 hi16 = h ? (c1 >> 16) : (c1 & 0xffffu);
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
                const u32 wv = k == 0u ? c0 : (k == 1u ? c1 : c2);
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

// One unit: tiles [blk nw, blk nw + nw) of g (one a warp; a warp past the
// last tile stages with the others and computes nothing), slice sl, staged
// once and streamed by each warp alone.
__device__ __forceinline__ void p4t_unit_stream(const P4tSeg& g, u32 sl, u32 blk, const Pers4tLaunch& L,
                                                const Pers4eArgs* A, const F1rTables& tab, const float* t_inv,
                                                u32* area, unsigned long long* cmax)
{
    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, NW = blockDim.x >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, RS = 8u * NT;
    const u32 nsub = (g.d_in + P4T_SUB - 1u) / P4T_SUB;
    const u32 sub_lo = sl * nsub / g.S, sub_hi = (sl + 1u) * nsub / g.S, ns = sub_hi - sub_lo;
    if (ns > p4t_stream_cap(L.area_floats, NW)) {
        // a plan the area cannot stage: refused, every CTA alike
        if (t == 0u) atomicExch(A->abort_flag, 1u);
        return;
    }
    const u32 XS = 24u * ns + 4u;
    u32* aw = area + warp * 16u * P4T_AS;
    u32* xs = area + NW * 16u * P4T_AS;
    const u32 tile = blk * NW + warp;
    const bool has = tile < (g.d_out >> 4);
    const u32 wrow = tile * 16u;
    float acc[4][4];
#pragma unroll
    for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;
    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);
    p4t_cp_wait<0u>();
    __syncthreads();
    if (has && ns > 0u) {
        const u32 SW = g.kind == 0u ? P4T_TSW : P4T_TSQ;
        const u32 r = lane >> 1, h = lane & 1u;
        const u32* wp = g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);
        u32 c0, c1, c2;
        p4t_stream_load(g.kind, wp, c0, c1, c2);
        for (u32 s = 0; s < ns; ++s) {
            u32 n0 = 0u, n1 = 0u, n2 = 0u;
            if (s + 1u < ns) p4t_stream_load(g.kind, wp + SW, n0, n1, n2);
            p4t_decode_stream(g, tab, t_inv, c0, c1, c2, aw, sub_lo + s, wrow, lane);
            __syncwarp();
#pragma unroll
            for (u32 kk = 0; kk < 3u; ++kk) {
                const u32* a0p = aw + gid * P4T_AS + kk * 8u + tig;
                const u32 af[4] = {a0p[0], a0p[8u * P4T_AS], a0p[4], a0p[8u * P4T_AS + 4u]};
                const u32 kw = 24u * s + 8u * kk + tig;
#pragma unroll
                for (u32 nt = 0; nt < 4u; ++nt) {
                    if (nt < NT) {
                        const u32* xr = xs + (nt * 8u + gid) * XS + kw;
                        p4t_mma(acc[nt], af, xr[0], xr[4]);
                    }
                }
            }
            __syncwarp();
            wp += SW;
            c0 = n0;
            c1 = n1;
            c2 = n2;
        }
    }
    __syncthreads();
    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);
}

// p4t_phase's deal, the stream unit.
__device__ __forceinline__ void p4t_phase_stream(const P4tSeg& s0, const P4tSeg& s1, const P4tSeg& s2, u32 nseg,
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
        p4t_unit_stream(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);
    }
}

// ---- the pass: pers_4et.cu's pers_4et_body, its projection phases on the
// stream unit. The text below is that body with three substitutions
// (template <int FORM, bool TILED> -> template <int FORM>, pers_4et_body ->
// pers_4et_stream_body, p4t_phase<TILED>( -> p4t_phase_stream( ), and
// pers.rs's test the_stream_body_is_the_tree_body pins it so.

template <int FORM>
__device__ __forceinline__ void pers_4et_stream_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
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
            p4t_phase_stream(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_stream(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_stream(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_stream(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
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
        p4t_phase_stream(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
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

#define PERS_4ET_STREAM(sym, FORM_, THREADS_)                                                           \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_stream_body<FORM_>(A, L);                                                              \
    }

PERS_4ET_STREAM(pers_4et_stream_f0_t512, 0, 512)
PERS_4ET_STREAM(pers_4et_stream_f1_t512, 1, 512)
PERS_4ET_STREAM(pers_4et_stream_f2_t512, 2, 512)

#endif
