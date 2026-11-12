// pers_4et_a5.cu: arm a5 of the tree verify, a4 with the L2 prefetch of the
// next row phase's tile streams.
//
// Appended after pers_4et.cu, pers_4et_stream.cu and pers_4et_a4.cu, which
// keep their bytes (the a0..a4 module texts do not move). A module that
// defines P4T_A5 1 (pers.rs, p4t_a5_unit_text) emits pers_4et_a5_*: a4's
// pass (its unit, fold, prologues and F3 parts) with three prefetch points
// on 4E v9's schedule (pers_4e.cu, pers_segment.cu):
//
//   * at the layer's start, before ln1: the q/k/v phase (4E: its q/k/v rows
//     before ln1 and the rotation);
//   * once the CTA's attention jobs are done, before the wait: the o and
//     gate/up phases (4E: o and gate/up rows around attention, before it on
//     a CTA with no attention job, after it otherwise; here after it on
//     every CTA, the tree's attention deal being per row);
//   * once the CTA's gate/up units are done, before the wait: the down
//     phase (4E: its down rows while the slowest CTA finishes gate/up).
//
// What a CTA asks is what it will itself run: the units u = blockIdx.x mod
// grid of the target phase, p4t_phase's deal, and for each unit its warps'
// tile streams over the slice's sub-steps in the tree-tiled copy, warp w
// tile w, the lanes over the 128-byte lines. `prefetch.global.L2` with no
// eviction hint, as 4E's pers_l2_prefetch: no register written, no value
// read, so no bit can depend on it. The host twin has no cache: there each
// range's first and last words are read into a volatile sink, so a range
// outside the copies faults (the twin's only view of the addresses).

#ifndef P4T_A5
#define P4T_A5 0
#endif

#if P4T_A5 || defined(LLVQ_HOST_BUILD)

// The 128-byte lines of [p, p + bytes) asked of L2 by a warp's lanes.
__device__ __forceinline__ void p4t_pf_lines(const u32* p, unsigned long long bytes, u32 lane)
{
    if (bytes == 0ull) return;
#ifdef LLVQ_HOST_BUILD
    volatile u32 sink = p[0] ^ p[bytes / 4ull - 1ull];
    (void)sink;
    (void)lane;
#else
    const unsigned long long a0 = (unsigned long long)p & ~127ull, a1 = (unsigned long long)p + bytes;
    for (unsigned long long a = a0 + 128ull * lane; a < a1; a += 128ull * 32ull)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
#endif
}

// The tile streams of the units this CTA will run in a phase of up to three
// projections (p4t_phase's deal), warp w the tile w of each unit.
__device__ __forceinline__ void p4t_pf_phase(const P4tSeg& s0, const P4tSeg& s1, const P4tSeg& s2, u32 nseg)
{
    const u32 NW = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const u32 n0 = p4t_nblk(s0, NW) * s0.S;
    const u32 n1 = nseg > 1u ? p4t_nblk(s1, NW) * s1.S : 0u;
    const u32 n2 = nseg > 2u ? p4t_nblk(s2, NW) * s2.S : 0u;
    for (u32 u = blockIdx.x; u < n0 + n1 + n2; u += gridDim.x) {
        const u32 i = u < n0 ? 0u : (u < n0 + n1 ? 1u : 2u);
        const u32 lu = i == 0u ? u : (i == 1u ? u - n0 : u - n0 - n1);
        const P4tSeg& g = i == 0u ? s0 : (i == 1u ? s1 : s2);
        const u32 sl = lu % g.S, blk = lu / g.S, tile = blk * NW + warp;
        if (tile >= (g.d_out >> 4)) continue;
        const u32 nsub = (g.d_in + P4T_SUB - 1u) / P4T_SUB, SW = g.kind == 0u ? P4T_TSW : P4T_TSQ;
        const u32 sub_lo = sl * nsub / g.S, sub_hi = (sl + 1u) * nsub / g.S;
        p4t_pf_lines(g.w + ((unsigned long long)tile * nsub + sub_lo) * SW, 4ull * (sub_hi - sub_lo) * SW, lane);
    }
}

// The q/k/v phase's streams (the body's segments, its slices).
__device__ __forceinline__ void p4t_pf_qkv(const Pers4eLayer& Y, const Pers4tLaunch& L)
{
    const Pers4eQkv& N = Y.qkv;
    P4tSeg sq = p4t_seg_tetra(N.qw, N.qgs, N.qrs, N.qtl, N.q_stride, N.nblocks, N.tail_w, N.q_dout);
    P4tSeg sk = p4t_seg_tetra(N.kw, N.kgs, N.krs, N.ktl, N.k_stride, N.nblocks, N.tail_w, N.k_dout);
    P4tSeg sv = p4t_seg_q4(N.vw, N.vsc, N.vbi, N.d_in, N.v_dout, 7u, 1u);
    sq.S = sk.S = sv.S = L.split_qkv;
    p4t_pf_phase(sq, sk, sv, 3u);
}

// The o and gate/up phases' streams.
__device__ __forceinline__ void p4t_pf_o_gu(const Pers4eLayer& Y, const Pers4tLaunch& L, u32 H, u32 dq)
{
    P4tSeg so;
    if (Y.ot != nullptr) {
        const Pers4eOT& O = *Y.ot;
        so = p4t_seg_tetra(O.t.words, O.t.gscale, O.t.rscale, O.t.tail, O.t.stride_u32, O.t.nblocks, O.t.tail_w,
                           O.t.d_out);
    } else {
        so = p4t_seg_q4(Y.ow, Y.osc, Y.obi, dq, H, 7u, 1u);
    }
    so.S = L.split_o;
    p4t_pf_phase(so, so, so, 1u);
    const SegArgs& M = Y.mlp;
    P4tSeg sg = p4t_seg_tetra(M.gate.words, M.gate.gscale, M.gate.rscale, M.gate.tail, M.gate.stride_u32,
                              M.gate.nblocks, M.gate.tail_w, M.gate.d_out);
    P4tSeg su = p4t_seg_tetra(M.up.words, M.up.gscale, M.up.rscale, M.up.tail, M.up.stride_u32, M.up.nblocks,
                              M.up.tail_w, M.up.d_out);
    sg.S = su.S = L.split_gu;
    p4t_pf_phase(sg, su, su, 2u);
}

// The down phase's streams.
__device__ __forceinline__ void p4t_pf_down(const Pers4eLayer& Y, const Pers4tLaunch& L, u32 H)
{
    const SegArgs& M = Y.mlp;
    P4tSeg sd = Y.tetra_down ? p4t_seg_tetra(M.down.words, M.down.gscale, M.down.rscale, M.down.tail,
                                             M.down.stride_u32, M.down.nblocks, M.down.tail_w, M.down.d_out)
                             : p4t_seg_q4(M.q4w, M.q4s, M.q4b, M.inter, H, 7u, 1u);
    sd.S = L.split_dn;
    p4t_pf_phase(sd, sd, sd, 1u);
}

// ---- the pass: pers_4et_a4.cu's, with the three prefetch points ----

template <int FORM>
__device__ __forceinline__ void pers_4et_a5_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
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
        p4t_pf_qkv(Y, L);
        // ln1: the natural values for the int4 v rows, the signed and
        // rotated ones for the Tetra q and k rows
        for (u32 r = c0; r < R; r += G) {
            p4t_ln_a4(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, A->eps, H, S, L.xn + (size_t)r * H);
            pers_rot_finish(S, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            p4t_out_a4(S, H, L.xin + (size_t)r * L.ldx);
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
            p4t_phase_a4(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
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
        p4t_pf_o_gu(Y, L, H, dq);
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
                    p4t_out_a4(S, dq, L.xin + (size_t)r * L.ldx);
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
            p4t_phase_a4(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // ln2, signed and rotated: the gate and up rows' input
        for (u32 r = c0; r < R; r += G) {
            p4t_ln_a4(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.eps, H, S, nullptr);
            pers_rot_finish(S, M.small_h, H, M.m_h, M.k_h, M.inv_h);
            p4t_out_a4(S, H, L.xin + (size_t)r * L.ldx);
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
            p4t_phase_a4(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
            p4t_pf_down(Y, L, H);
        }
        P4T_WAIT();
        // the down's input: silu(g) u, signed and rotated for a Tetra down,
        // natural for an int4 one
        for (u32 r = c0; r < R; r += G) {
            const unsigned short* gr = L.g_raw + (size_t)r * I;
            const unsigned short* ur = L.u_raw + (size_t)r * I;
            unsigned short* dst = L.xin + (size_t)r * L.ldx;
            p4t_dn_in_a4(M, Y.tetra_down, gr, ur, dst, S, I);
            if (Y.tetra_down) {
                pers_rot_finish(S, M.small_i, I, M.m_i, M.k_i, M.inv_i);
                p4t_out_a4(S, I, dst);
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
            p4t_phase_a4(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
    }
    // the final norm, natural: the head's input
    for (u32 r = c0; r < R; r += G)
        p4t_ln_a4(L.x_a + (size_t)r * H, A->final_norm, nullptr, A->eps, H, S, L.xin + (size_t)r * L.ldx);
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
        p4t_phase_a4(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
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

#define PERS_4ET_A5(sym, FORM_, THREADS_)                                                               \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a5_body<FORM_>(A, L);                                                                  \
    }

PERS_4ET_A5(pers_4et_a5_f0_t512, 0, 512)
PERS_4ET_A5(pers_4et_a5_f1_t512, 1, 512)
PERS_4ET_A5(pers_4et_a5_f2_t512, 2, 512)

#endif
