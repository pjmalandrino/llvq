// pers_4et_a10rc.cu: arms a10r8 and a10r16 of the tree verify, a10 compiled
// for at most 8 or 16 rows a launch. Bit for bit with a10 at the rows they
// allow. Appended after a10's files, which keep their bytes.
//
// What changes is live state only: a unit's accumulators are RC n8 tiles of
// four floats (RC = 1 or 2, against a10's 4), the MMA loop and the store run
// over those RC tiles, and a launch above 8 RC rows is refused through the
// abort flag. The staged activation rows were already 8 x ceil(R / 8), the
// launch's area and plan are a10's, the slice cap is a10's (computed at
// P4T_RMAX rows), so the slices do not change. The unit, the deal and the
// pass are a9's and a10's with the substitutions pers.rs's test
// the_row_cap_passes_are_a10s pins. A module defines P4T_A10RC as 8 or 16
// and gets that arm's symbols only.

#ifndef P4T_A10RC
#define P4T_A10RC 0
#endif

#if P4T_A10RC || defined(LLVQ_HOST_BUILD)

// p4t_store_a9 (and p4t_store's S = 1 branch) over RC n8 tiles.
template <u32 RC>
__device__ __forceinline__ void p4t_store_rc(const P4tSeg& g, const Pers4tLaunch& L, const Pers4eArgs* A, u32 sl,
                                             const float (&acc)[RC][4], u32 wrow, u32 lane, unsigned long long* cmax)
{
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, gid = lane >> 2, tig = lane & 3u;
    if (g.S == 1u) {
#pragma unroll
        for (u32 nt = 0; nt < RC; ++nt)
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);
                if (nt < NT && n < R) p4t_epi(g, L, A, n, row, acc[nt][e], cmax);
            }
        return;
    }
#pragma unroll
    for (u32 nt = 0; nt < RC; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) {
            const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);
            if (nt < NT && n < R) g.part[((unsigned long long)sl * R + n) * g.d_out + row] = acc[nt][e];
        }
}

// ---- the served words read in place (arm a12ip, step M1 of the memory axis) ----
//
// The stream unit's lane reads, for each sub-step, its own row's words: the
// tree-tiled copy only puts a tile's 16 rows of a sub-step side by side. So
// the unit can read the words where 4E and the eager path hold them, the
// row layout (Tetra, g.stride words a row) and 4E v9's int4 order (d_in / 8
// words a row, pers_q4_tile_index when g.tiled, the file's order for the
// head), and the copies are not needed. Each read returns the word
// p4t_tile_word writes at the lane's offset of the copy: the same words into
// the same decode, so the same bits. Under IP = false the unit is a10r16's,
// unchanged.

// u32 words of a row in the served order.
__device__ __forceinline__ u32 p4t_ip_stride(const P4tSeg& g)
{
    return g.kind == 0u ? g.stride : (g.d_in >> 3);
}

// Row `row` of g in the served order.
__device__ __forceinline__ const u32* p4t_ip_row(const P4tSeg& g, u32 row)
{
    return g.w + (unsigned long long)row * p4t_ip_stride(g);
}

// Logical word wi of a row read in place: zero past the row's end, an int4
// word at its 4E v9 place when g.tiled (p4t_tile_word's rule).
__device__ __forceinline__ u32 p4t_ip_word(const P4tSeg& g, const u32* rowp, u32 sw, u32 wi)
{
    if (wi >= sw) return 0u;
    return rowp[(g.kind != 0u && g.tiled != 0u) ? pers_q4_tile_index(wi, sw) : wi];
}

// A lane's words of sub-step sa, half h of its row: what p4t_stream_load reads
// at the lane's offset of the copy. Tetra: logical words 3 sa + h and
// 3 sa + h + 1; int4: 6 sa + 3 h + k, k < 3.
__device__ __forceinline__ void p4t_ip_load(const P4tSeg& g, const u32* rowp, u32 sa, u32 h, u32& c0, u32& c1, u32& c2)
{
    const u32 sw = p4t_ip_stride(g);
    const u32 w0 = g.kind == 0u ? 3u * sa + h : 6u * sa + 3u * h;
    c0 = p4t_ip_word(g, rowp, sw, w0);
    c1 = p4t_ip_word(g, rowp, sw, w0 + 1u);
    c2 = g.kind == 0u ? 0u : p4t_ip_word(g, rowp, sw, w0 + 2u);
}

// ---- a9's unit and deal over RC tiles, a10's pass ----

// One unit: tiles [blk nw, blk nw + nw) of g (one a warp; a warp past the
// last tile stages with the others and computes nothing), slice sl, staged
// once and streamed by each warp alone. IP: the words read in place, from
// the served order (p4t_ip_load), instead of the tree-tiled copy.
template <u32 RC, bool IP = false>
__device__ __forceinline__ void p4t_unit_rc(const P4tSeg& g, u32 sl, u32 blk, const Pers4tLaunch& L,
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
    float acc[RC][4];
#pragma unroll
    for (u32 nt = 0; nt < RC; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;
    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);
    p4t_cp_wait<0u>();
    __syncthreads();
    if (has && ns > 0u) {
        const u32 SW = g.kind == 0u ? P4T_TSW : P4T_TSQ;
        const u32 r = lane >> 1, h = lane & 1u;
        const u32* wp = IP ? p4t_ip_row(g, wrow + r)
                           : g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);
        u32 c0, c1, c2;
        if (IP) p4t_ip_load(g, wp, sub_lo, h, c0, c1, c2);
        else p4t_stream_load(g.kind, wp, c0, c1, c2);
        for (u32 s = 0; s < ns; ++s) {
            u32 n0 = 0u, n1 = 0u, n2 = 0u;
            if (IP) {
                if (s + 1u < ns) p4t_ip_load(g, wp, sub_lo + s + 1u, h, n0, n1, n2);
            } else if (s + 1u < ns) {
                p4t_stream_load(g.kind, wp + SW, n0, n1, n2);
            }
            p4t_decode_stream(g, tab, t_inv, c0, c1, c2, aw, sub_lo + s, wrow, lane);
            __syncwarp();
#pragma unroll
            for (u32 kk = 0; kk < 3u; ++kk) {
                const u32* a0p = aw + gid * P4T_AS + kk * 8u + tig;
                const u32 af[4] = {a0p[0], a0p[8u * P4T_AS], a0p[4], a0p[8u * P4T_AS + 4u]};
                const u32 kw = 24u * s + 8u * kk + tig;
#pragma unroll
                for (u32 nt = 0; nt < RC; ++nt) {
                    if (nt < NT) {
                        const u32* xr = xs + (nt * 8u + gid) * XS + kw;
                        p4t_mma(acc[nt], af, xr[0], xr[4]);
                    }
                }
            }
            __syncwarp();
            if (!IP) wp += SW;
            c0 = n0;
            c1 = n1;
            c2 = n2;
        }
    }
    __syncthreads();
    if (has) p4t_store_rc<RC>(g, L, A, sl, acc, wrow, lane, cmax);
}

// p4t_phase's deal, the stream unit.
template <u32 RC, bool IP = false>
__device__ __forceinline__ void p4t_phase_rc(const P4tSeg& s0, const P4tSeg& s1, const P4tSeg& s2, u32 nseg,
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
        p4t_unit_rc<RC, IP>(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);
    }
}

template <int FORM, u32 RC>
__device__ __forceinline__ void pers_4et_rc_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
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
        if (R == 0u || R > 8u * RC || (L.area_floats & 1u) != 0u ||
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
#define P4T_WAIT_GAP(...)                                                                            \
    do {                                                                                             \
        if (L.stamps != nullptr) {                                                                   \
            __syncthreads();                                                                         \
            if (threadIdx.x == 0u) L.stamps[(size_t)(2u * nb + 1u) * G + c0] = pers_now();           \
        }                                                                                            \
        if (!pers_grid_wait_gap(A->bar, L.bar_base + (++nb) * G, A->abort_flag, [&] { __VA_ARGS__; })) return; \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)(2u * nb) * G + c0] = pers_now(); \
    } while (0)
    // a10: the staged rotation area (null: the prologues keep a9's path)
    float* const v7a10 = p4t_v7_a10(S, L.area_floats, 2u * H > A->layers[0].mlp.inter ? 2u * H : A->layers[0].mlp.inter);
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
    P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, A->layers[0].signs_a, H, A->layers[0].small_a, A->layers[0].k_a));
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
            p4t_ln_rot_a10(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, Y.small_a, A->eps, H, Y.m_a, Y.k_a, Y.inv_a, S, v7a10,
                           L.xn + (size_t)r * H, L.xin + (size_t)r * L.ldx);
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
            p4t_phase_rc<RC>(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
            P4T_WAIT();
            p4t_fold_a9(sq, sk, sv, 3u, L, A, cmax);
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
            p4t_phase_rc<RC>(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
            P4T_WAIT();
            p4t_fold_a9(so, so, so, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, M.signs_h, H, M.small_h, M.k_h));
        // ln2, signed and rotated: the gate and up rows' input
        for (u32 r = c0; r < R; r += G) {
            p4t_ln_rot_a10(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.small_h, M.eps, H, M.m_h, M.k_h, M.inv_h, S, v7a10,
                           nullptr, L.xin + (size_t)r * L.ldx);
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
            p4t_phase_rc<RC>(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
            P4T_WAIT();
            p4t_fold_gu_a10(sg, su, M, L);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr && Y.tetra_down) pers_v7_stage_rot(v7a10, M.signs_i, I, M.small_i, M.k_i));
        // the down's input: silu(g) u, signed and rotated for a Tetra down,
        // natural for an int4 one
        for (u32 r = c0; r < R; r += G) p4t_dn_in_a10(M, Y.tetra_down, L.xin + (size_t)r * L.ldx, S, I, v7a10);
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
            p4t_phase_rc<RC>(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
            P4T_WAIT();
            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) {
            if (l + 1u < NL) pers_v7_stage_rot(v7a10, A->layers[l + 1u].signs_a, H, A->layers[l + 1u].small_a, A->layers[l + 1u].k_a);
            else p4t_v7_zero_signs_a10(v7a10, H);
        });
    }
    // the final norm, natural: the head's input
    for (u32 r = c0; r < R; r += G)
        p4t_ln_final_a10(L.x_a + (size_t)r * H, A->final_norm, A->eps, H, S, v7a10, L.xin + (size_t)r * L.ldx);
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
        p4t_phase_rc<RC>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
        __syncthreads();
        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);
        P4T_WAIT();
        p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);
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
#undef P4T_WAIT_GAP
#undef P4T_WAIT
}

#define PERS_4ET_RC(sym, FORM_, RC_, THREADS_)                                                           \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_rc_body<FORM_, RC_>(A, L);                                                             \
    }

#if P4T_A10RC == 8 || defined(LLVQ_HOST_BUILD)
PERS_4ET_RC(pers_4et_a10r8_f0_t512, 0, 1u, 512)
PERS_4ET_RC(pers_4et_a10r8_f1_t512, 1, 1u, 512)
PERS_4ET_RC(pers_4et_a10r8_f2_t512, 2, 1u, 512)
#endif
#if P4T_A10RC == 16 || defined(LLVQ_HOST_BUILD)
PERS_4ET_RC(pers_4et_a10r16_f0_t512, 0, 2u, 512)
PERS_4ET_RC(pers_4et_a10r16_f1_t512, 1, 2u, 512)
PERS_4ET_RC(pers_4et_a10r16_f2_t512, 2, 2u, 512)
#endif

#endif
