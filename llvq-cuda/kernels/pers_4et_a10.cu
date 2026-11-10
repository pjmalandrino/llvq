// pers_4et_a10.cu: arm a10 of the tree verify, a9 (pers_4et_a9.cu) with 4E
// v9's prologue path. Bit for bit with a9. Appended after a9's files, which
// keep their bytes.
//
//   * ln1, ln2 and the final norm on the row's owner CTA through
//     pers_ln_1rt (16-byte loads, PERS_1RT_V vectors a thread issued before
//     first use), the rotation through pers_rot_finish<true, true> (the
//     k rows of `small` read from shared memory, 4E's unrolled float4 mix),
//     the binary16 write-out in 16-byte stores. Element for element p4t_ln,
//     the tree's pers_rot_finish and p4t_out: the same candle scale, the same
//     apply, the same WHT stages and mix chains (pers_mix_column_k4 is
//     pers_mix_column term for term, llvq_pers.cuh section 6).
//   * The rotation's sign words and k rows of `small` are staged into a
//     1,840-float area (4E's v7 layout, PERS_V7_*) at the end of the
//     launch's area, just before the argmax maxima, by every CTA during the
//     grid wait that precedes the prologue (pers_grid_wait_gap, 4E's
//     stage_layer): ln1 of layer l + 1 during layer l's last wait (layer 0
//     during the embedding's), ln2 during the o fold's, the down's during
//     the gate/up fold's, zero sign words for the final norm during the last
//     layer's. The area must hold the widest prologue row (and twice the
//     hidden width for ln's natural copy) below it; otherwise the prologues
//     keep a9's path.
//   * silu(gate) x up leaves the down's prologue: a9's gate/up fold step
//     deals the (row, quarter) pairs so the thread that folds a gate quarter
//     folds the same rows' up quarter, then writes exact_silu_mul of the
//     four elements into xin (the down's input row). An int4 down has no
//     prologue left; a Tetra down's prologue is pers_load_1rt of that row
//     with the staged signs, the rotation, the write-out in place. With S = 1
//     on gate/up (the phase stored g and u directly) the same step reads them
//     back from g_raw and u_raw.
//
// The pass is pers_4et_a9.cu's with the substitutions pers.rs's test
// the_a10_body_is_the_a9_body pins.

#ifndef P4T_A10
#define P4T_A10 0
#endif

#if P4T_A10 || defined(LLVQ_HOST_BUILD)

// The staged rotation area: PERS_V7_FLOATS floats, 16-byte aligned, below
// the argmax maxima; null when the area cannot hold it above `need` floats.
__device__ __forceinline__ float* p4t_v7_a10(float* S, u32 area_floats, u32 need)
{
    if (area_floats < 2u * P4T_RMAX + PERS_V7_FLOATS + 4u) return nullptr;
    const u32 off = (area_floats - 2u * P4T_RMAX - PERS_V7_FLOATS) & ~3u;
    return off >= need ? S + off : nullptr;
}

// Zero sign words for the final norm (no rotation), PERS_V7 layout.
__device__ __forceinline__ void p4t_v7_zero_signs_a10(float* v7, u32 width)
{
    u32* sw = reinterpret_cast<u32*>(v7 + PERS_V7_SIGNS);
    for (u32 i = threadIdx.x; i < (width >> 5); i += blockDim.x) sw[i] = 0u;
}

// S (n floats) rounded to binary16 into dst, 8 halves a 16-byte store.
// Ends with a barrier.
__device__ __forceinline__ void p4t_out_v_a10(const float* S, u32 n, unsigned short* dst)
{
    uint4* dv = reinterpret_cast<uint4*>(dst);
    for (u32 q = threadIdx.x; q < (n >> 3); q += blockDim.x) {
        uint4 w;
        w.x = (u32)f2h(S[8u * q]) | ((u32)f2h(S[8u * q + 1u]) << 16);
        w.y = (u32)f2h(S[8u * q + 2u]) | ((u32)f2h(S[8u * q + 3u]) << 16);
        w.z = (u32)f2h(S[8u * q + 4u]) | ((u32)f2h(S[8u * q + 5u]) << 16);
        w.w = (u32)f2h(S[8u * q + 6u]) | ((u32)f2h(S[8u * q + 7u]) << 16);
        dv[q] = w;
    }
    __syncthreads();
}

// A rotated ln (ln1 with nat, ln2 without) on the staged rows: p4t_ln, the
// tree's rotation and p4t_out, 4E's way. v7 null or a row not aligned: a9's
// three calls.
__device__ __forceinline__ void p4t_ln_rot_a10(const unsigned short* x, const unsigned short* alpha, const u32* signs,
                                               const float* small, float eps, u32 H, u32 m, u32 k, float inv, float* S,
                                               const float* v7, unsigned short* nat, unsigned short* out)
{
    if (v7 != nullptr && pers_1rt_ok(x, H) && pers_1rt_ok(alpha, H) && pers_1rt_ok(out, H) &&
        (nat == nullptr || pers_1rt_ok(nat, H))) {
        float* xs = S + H;
        pers_ln_1rt<0>(x, alpha, reinterpret_cast<const u32*>(v7 + PERS_V7_SIGNS), H, eps, nullptr, k,
                       const_cast<float*>(v7 + PERS_V7_SMALL), S, xs, nat != nullptr);
        if (nat != nullptr) p4t_out_v_a10(xs, H, nat);
        pers_rot_finish<true, true>(S, v7 + PERS_V7_SMALL, H, m, k, inv);
        p4t_out_v_a10(S, H, out);
        return;
    }
    p4t_ln(x, alpha, signs, eps, H, S, nat);
    pers_rot_finish(S, small, H, m, k, inv);
    p4t_out(S, H, out);
}

// The final norm (natural, no rotation) on staged zero sign words: p4t_ln
// with nat = dst. v7 null or a row not aligned: p4t_ln.
__device__ __forceinline__ void p4t_ln_final_a10(const unsigned short* x, const unsigned short* alpha, float eps, u32 H,
                                                 float* S, const float* v7, unsigned short* dst)
{
    if (v7 != nullptr && pers_1rt_ok(x, H) && pers_1rt_ok(alpha, H) && pers_1rt_ok(dst, H)) {
        pers_ln_1rt<0>(x, alpha, reinterpret_cast<const u32*>(v7 + PERS_V7_SIGNS), H, eps, nullptr, 0u, nullptr, S,
                       S + H, true);
        __syncthreads();
        p4t_out_v_a10(S + H, H, dst);
        return;
    }
    p4t_ln(x, alpha, nullptr, eps, H, S, dst);
}

// The gate/up fold step with the down's input: pairs (row n, quarter) of the
// inter width over the grid's threads; the gate and up quarters folded (or
// read back when S = 1), their binary16 values f2h(sum) as a9 stores them,
// then exact_silu_mul of the four elements into xin row n.
__device__ __forceinline__ void p4t_fold_gu_a10(const P4tSeg& sg, const P4tSeg& su, const SegArgs& M,
                                                const Pers4tLaunch& L)
{
    const u32 R = L.n_rows, nq = sg.d_out >> 2, np = nq * R;
    const bool vec = sg.S > 1u && ((reinterpret_cast<unsigned long long>(sg.part) & 15ull) == 0ull) &&
                     ((reinterpret_cast<unsigned long long>(su.part) & 15ull) == 0ull) && (sg.d_out & 3u) == 0u;
    const u32 T = gridDim.x * blockDim.x;
    for (u32 p = blockIdx.x * blockDim.x + threadIdx.x; p < np; p += T) {
        const u32 n = p / nq, row0 = 4u * (p - n * nq);
        unsigned short gh[4], uh[4];
        if (sg.S == 1u) {
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                gh[e] = __ldcg(sg.y + (unsigned long long)n * sg.ldy + row0 + e);
                uh[e] = __ldcg(su.y + (unsigned long long)n * su.ldy + row0 + e);
            }
        } else {
            float vg[4], vu[4];
            if (vec) {
                const float4 g0 = __ldcg(reinterpret_cast<const float4*>(sg.part + (unsigned long long)n * sg.d_out + row0));
                const float4 u0 = __ldcg(reinterpret_cast<const float4*>(su.part + (unsigned long long)n * su.d_out + row0));
                vg[0] = g0.x; vg[1] = g0.y; vg[2] = g0.z; vg[3] = g0.w;
                vu[0] = u0.x; vu[1] = u0.y; vu[2] = u0.z; vu[3] = u0.w;
                for (u32 k = 1u; k < sg.S; ++k) {
                    const float4 gw = __ldcg(reinterpret_cast<const float4*>(sg.part + ((unsigned long long)k * R + n) * sg.d_out + row0));
                    const float4 uw = __ldcg(reinterpret_cast<const float4*>(su.part + ((unsigned long long)k * R + n) * su.d_out + row0));
                    vg[0] = __fadd_rn(vg[0], gw.x); vg[1] = __fadd_rn(vg[1], gw.y);
                    vg[2] = __fadd_rn(vg[2], gw.z); vg[3] = __fadd_rn(vg[3], gw.w);
                    vu[0] = __fadd_rn(vu[0], uw.x); vu[1] = __fadd_rn(vu[1], uw.y);
                    vu[2] = __fadd_rn(vu[2], uw.z); vu[3] = __fadd_rn(vu[3], uw.w);
                }
            } else {
#pragma unroll
                for (u32 e = 0; e < 4u; ++e) {
                    float a = __ldcg(sg.part + (unsigned long long)n * sg.d_out + row0 + e);
                    float b = __ldcg(su.part + (unsigned long long)n * su.d_out + row0 + e);
                    for (u32 k = 1u; k < sg.S; ++k) {
                        a = __fadd_rn(a, __ldcg(sg.part + ((unsigned long long)k * R + n) * sg.d_out + row0 + e));
                        b = __fadd_rn(b, __ldcg(su.part + ((unsigned long long)k * R + n) * su.d_out + row0 + e));
                    }
                    vg[e] = a;
                    vu[e] = b;
                }
            }
#pragma unroll
            for (u32 e = 0; e < 4u; ++e) {
                gh[e] = f2h(vg[e]);
                uh[e] = f2h(vu[e]);
            }
        }
        unsigned short* dst = L.xin + (unsigned long long)n * L.ldx + row0;
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) dst[e] = exact_silu_mul(M.silu, gh[e], uh[e]);
    }
}

// The down's prologue on the owner CTA: an int4 down has none (its input is
// in xin); a Tetra down loads its row of silu(g) u from xin with the staged
// signs, rotates it with the staged rows and writes it back in place.
__device__ __forceinline__ void p4t_dn_in_a10(const SegArgs& M, bool tetra, unsigned short* dst, float* S, u32 I,
                                              const float* v7)
{
    if (!tetra) return;
    if (v7 != nullptr && pers_1rt_ok(dst, I)) {
        pers_load_1rt(dst, reinterpret_cast<const u32*>(v7 + PERS_V7_SIGNS), S, I, nullptr, M.k_i,
                      const_cast<float*>(v7 + PERS_V7_SMALL));
        pers_rot_finish<true, true>(S, v7 + PERS_V7_SMALL, I, M.m_i, M.k_i, M.inv_i);
        p4t_out_v_a10(S, I, dst);
        return;
    }
    for (u32 i = threadIdx.x; i < I; i += blockDim.x) {
        const float f = h2f(__ldcg(dst + i));
        S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;
    }
    pers_rot_finish(S, M.small_i, I, M.m_i, M.k_i, M.inv_i);
    p4t_out(S, I, dst);
}

// ---- a10: a9's pass with 4E's prologues ----

template <int FORM>
__device__ __forceinline__ void pers_4et_a10_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
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
            p4t_phase_a9(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_a9(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_a9(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_a9(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
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
        p4t_phase_a9(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
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

#define PERS_4ET_A10(sym, FORM_, THREADS_)                                                              \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a10_body<FORM_>(A, L);                                                                 \
    }

PERS_4ET_A10(pers_4et_a10_f0_t512, 0, 512)
PERS_4ET_A10(pers_4et_a10_f1_t512, 1, 512)
PERS_4ET_A10(pers_4et_a10_f2_t512, 2, 512)

#endif
