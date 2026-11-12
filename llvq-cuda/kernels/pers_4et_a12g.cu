// pers_4et_a12g.cu: arm a12ipg, D1 of docs/mesures/large-prefill-contexte-
// 2026-11-11.txt section 3: a12ip whose attention reads a split of the cache
// once for a group of up to P4G_RG rows, where a12ip reads it once a row.
// Bit for bit with a12ip on everything a row leaves (argmax, logits,
// residual, K, V, taps). Appended after a12ip's text, which keeps its bytes.
//
// Why a group can share a staged split. In a12ip's attention (p4t_attention,
// pers_4et.cu) a job is (row r, split s, KV head h); it stages the split's
// positions step by step, K then V, position p from the cache below pos0 and
// from the tree buffer's slot path[r][p - pos0] from pos0 on, and folds them
// into the row's unit states with pers_attn_tile. Every quantity of the job
// is a function of the row's length l = pos0 + depth[r] + 1 and of s: the
// units nu, tiles a step tpu, tiles, steps, each tile's count nv. A split
// that ends at or below pos0, (s + 1) x 256 <= pos0, has l - base > 256 for
// every row of the launch, so nu >= 5, tpu = 1, tiles = 8, nsteps = 8, nv = 8
// at every tile, and every position comes from the cache: its staged steps
// are the same bytes for every row. There are nsh = pos0 / 256 such splits.
//
// The jobs here. First the shared ones, (group of rows, split s < nsh, KV
// head h), the group fastest so the jobs that read the same split run on
// neighbouring CTAs; group g is rows [P4G_RG g, P4G_RG g + P4G_RG) of the
// launch. A shared job stages each step once and runs, for each row of its
// group, a12ip's tasks (unit, head pair) on that row's own q and unit states:
// pers_attn_tile on the same staged tile, in the same step and tile order,
// so each state receives the same operations in the same order as in a12ip's
// job for that row. Then the per-row ones, (row, split s >= nsh, h), a12ip's
// job unchanged (the split that holds pos0, the tree's positions, and the
// last, partial split). After the steps each row of the job merges its units,
// writes its split state and draws its ticket on its own counter, as a12ip:
// the fold of the splits runs in split order whichever job draws the last
// ticket, so its bits do not depend on which job a split came from.
//
// The ring holds P4G_RING = 2 stages, not 3: the stages are copies, so their
// depth moves no bit (form 6 against form 5 changed nothing either). Shared
// memory of the phase: 2 stages (32,768 B) and, a row, its unit states and
// q (8,320 + 2,048 B at 4 query heads a KV head): 74,240 B at P4G_RG = 4,
// against a12ip's 59,520 B. With the 19,072 B of header and tables the CTA
// takes 93,568 B of the 101,376 B an sm_89 CTA may have; P4G_RG = 5 at two
// stages needs 103,936 B, and 4 at three stages 109,696 B. The area grows,
// the arm's other phases use what they used (the projections' share, the
// prologue rows, the maxima at the area's end). No buffer on the card moves.

#ifndef P4T_A12IPG
#define P4T_A12IPG 0
#endif

#if P4T_A12IPG || defined(LLVQ_HOST_BUILD)

#define P4G_RG 4u    // rows a shared job
#define P4G_RING 2u  // attention's stages

// Shared bytes of the grouped phase at `groups` query heads a KV head.
#define P4G_ATTN_SMEM(groups) \
    (16u * PERS_STAGE_U4 * P4G_RING + P4G_RG * (4u * PERS_UPS * (groups) * PERS_PSTRIDE + 4u * (groups) * PERS_HD))

// a12ip's area, or the grouped attention's with the maxima if larger.
__device__ __forceinline__ u32 p4g_area_floats(u32 nw, u32 rs, u32 widest, u32 groups)
{
    const u32 f = p4t_area_floats(nw, rs, widest, groups);
    const u32 g = (P4G_ATTN_SMEM(groups) + 3u) / 4u + 2u * P4T_RMAX;
    return g > f ? g : f;
}

// The splits every row of the launch reads whole from the cache.
__device__ __forceinline__ u32 p4g_shared_splits(u32 pos0)
{
    return pos0 / PERS_SPLIT;
}

// Attention for every row, the shared splits a group of rows a job.
__device__ __forceinline__ void p4g_attention(const Pers4eArgs* A, const Pers4tLaunch& L, const unsigned short* kc,
                                              const unsigned short* vc, const unsigned short* tk_l,
                                              const unsigned short* tv_l, unsigned char* smem)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 n_heads = A->n_heads, groups = A->groups, cap = A->cap, n_kv = A->n_kv;
    const float c = A->c;
    const u32 s_stride = (cap + PERS_SPLIT - 1u) / PERS_SPLIT;
    const u32 nhp = (groups + 1u) / 2u, tpr = PERS_UPS * nhp;  // a row's tasks
    const u32 ust_row = PERS_UPS * groups * PERS_PSTRIDE, qs_row = groups * PERS_HD;
    uint4* stage = reinterpret_cast<uint4*>(smem);
    float* ust0 = reinterpret_cast<float*>(smem + 16u * PERS_STAGE_U4 * P4G_RING);  // [RG][UPS][groups][PSTRIDE]
    float* qs0 = ust0 + P4G_RG * ust_row;                                          // [RG][groups][128]
    u32* hdr = pers_hdr();
    const u32 R = L.n_rows;
    const u32 nsh = p4g_shared_splits(L.pos0);
    const u32 ngrp = (R + P4G_RG - 1u) / P4G_RG;
    const u32 nshared = nsh * n_kv * ngrp;
    u32 total = nshared;
    for (u32 r = 0; r < R; ++r) total += n_kv * ((L.pos0 + L.depth[r] + 1u + PERS_SPLIT - 1u) / PERS_SPLIT - nsh);
    for (u32 job = blockIdx.x; job < total; job += gridDim.x) {
        u32 r0, nr, s, h;
        if (job < nshared) {
            const u32 gi = job % ngrp, sh = job / ngrp;
            s = sh / n_kv;
            h = sh - s * n_kv;
            r0 = gi * P4G_RG;
            nr = R - r0 < P4G_RG ? R - r0 : P4G_RG;
        } else {
            u32 r = 0u, j = job - nshared;
            for (;;) {
                const u32 nj = n_kv * ((L.pos0 + L.depth[r] + 1u + PERS_SPLIT - 1u) / PERS_SPLIT - nsh);
                if (j < nj) break;
                j -= nj;
                ++r;
            }
            s = nsh + j / n_kv;
            h = j % n_kv;
            r0 = r;
            nr = 1u;
        }
        // The job's geometry from its first row's length; on a shared split
        // every row's gives the same (see the head of this file).
        const u32 l = L.pos0 + L.depth[r0] + 1u;
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
        const unsigned char* path_r = L.path + r0 * P4T_RMAX;
        for (u32 st = 0; st < P4G_RING; ++st)
            p4t_attn_issue(stage + st * PERS_STAGE_U4, kh, vh, tkh, tvh, n_kv * PERS_HD, path_r, L.pos0, base, st, nsteps,
                           tpu, l);
        for (u32 i = threadIdx.x; i < nr * ust_row; i += blockDim.x) {
            const u32 k = (i % ust_row) % PERS_PSTRIDE;
            ust0[i] = k == PERS_HD ? PERS_NEG_INF : 0.0f;
        }
        for (u32 i = threadIdx.x; i < nr * qs_row; i += blockDim.x) {
            const u32 ri = i / qs_row, e = i - ri * qs_row;
            qs0[i] = h2f(__ldcg(L.q_out + (size_t)(r0 + ri) * n_heads * PERS_HD + h * groups * PERS_HD + e));
        }
        for (u32 st = 0; st < nsteps; ++st) {
            uint4* const sbuf = stage + (st % P4G_RING) * PERS_STAGE_U4;
            p4t_cp_wait<P4G_RING - 1u>();
            __syncthreads();
            for (u32 task = warp; task < nr * tpr; task += nwarps) {
                const u32 ri = task / tpr, tt = task - ri * tpr;
                const u32 u = tt / nhp, hp = tt - u * nhp;
                if (u * tpu >= PERS_UPS) continue;
                const u32 g = 2u * hp + (lane >> 4);
                const bool own = g < groups;
                const u32 ge = own ? g : groups - 1u;
                const float* qrow = qs0 + ri * qs_row + ge * PERS_HD;
                float* urow = ust0 + ri * ust_row + (u * groups + ge) * PERS_PSTRIDE;
                for (u32 t = 0; t < tpu; ++t) {
                    const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
                    if (p0 >= l || (st * tpu + t) * PERS_TT >= PERS_UNIT) break;
                    const u32 nv = l - p0 < PERS_TT ? l - p0 : PERS_TT;
                    const u32 sl = (u * tpu + t) * PERS_TT;
                    pers_attn_tile(qrow, urow, own, sbuf + sl * 16u, sbuf + (PERS_STAGE + sl) * 16u, nv, c, lane);
                }
            }
            __syncthreads();
            p4t_attn_issue(sbuf, kh, vh, tkh, tvh, n_kv * PERS_HD, path_r, L.pos0, base, st + P4G_RING, nsteps, tpu, l);
        }
        p4t_cp_wait<0u>();
        __syncthreads();
        // each row of the job: a12ip's end of a job on the row's own states
        for (u32 ri = 0; ri < nr; ++ri) {
            const u32 r = r0 + ri;
            const u32 lr = L.pos0 + L.depth[r] + 1u;
            const u32 ns = (lr + PERS_SPLIT - 1u) / PERS_SPLIT;
            const float* ust = ust0 + ri * ust_row;
            unsigned short* out = L.ctx + (size_t)r * n_heads * PERS_HD;
            float* part = L.part + (size_t)r * L.part_stride;
            u32* ticket = L.tickets + r * n_kv;
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
}

// ---- a12ipg: a12ip's pass (pers_4et_a12_body at RC 2, IP) with the grouped attention ----

template <int FORM, u32 RC, bool IP>
__device__ __forceinline__ void pers_4et_a12g_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
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
            L.area_floats < p4g_area_floats(nw, 8u * ((R + 7u) >> 3), widest, A->groups)) {
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
        p4t_taps_a12(L, H, R, l, c0, G);
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
            p4t_phase_rc<RC, IP>(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
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
        p4g_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));
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
            p4t_phase_rc<RC, IP>(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_rc<RC, IP>(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
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
            p4t_phase_rc<RC, IP>(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
            P4T_WAIT();
            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) {
            if (l + 1u < NL) pers_v7_stage_rot(v7a10, A->layers[l + 1u].signs_a, H, A->layers[l + 1u].small_a, A->layers[l + 1u].k_a);
            else p4t_v7_zero_signs_a10(v7a10, H);
        });
    }
    p4t_taps_a12(L, H, R, NL, c0, G);
    // A launch with L.split_head == 0 (the chain prefill's passes before the
    // prompt's last, docs/mesures/large-prefill-contexte-2026-11-11.txt) skips
    // the final norm and the head, every wait kept: no logits, no argmax (the
    // ring reads u32 max), the rows' residuals, K, V and taps untouched.
    const bool head_on = L.split_head != 0u;
    // the final norm, natural: the head's input
    if (head_on)
        for (u32 r = c0; r < R; r += G)
            p4t_ln_final_a10(L.x_a + (size_t)r * H, A->final_norm, A->eps, H, S, v7a10, L.xin + (size_t)r * L.ldx);
    P4T_WAIT();
    // the head: each row's argmax key through the CTA's maxima
    {
        for (u32 i = threadIdx.x; i < P4T_RMAX; i += blockDim.x) cmax[i] = 0ull;
        __syncthreads();
        P4tSeg sh = p4t_seg_q4(A->head_w, A->head_s, A->head_b, H, A->vocab, 6u, 0u);
        if (head_on) {
            p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sh, L, po, to);
            p4t_phase_rc<RC, IP>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
        }
        __syncthreads();
        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);
        P4T_WAIT();
        if (head_on) p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);
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

#define PERS_4ET_A12IPG(sym, FORM_, THREADS_)                                                           \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a12g_body<FORM_, 2u, true>(A, L);                                                      \
    }

PERS_4ET_A12IPG(pers_4et_a12ipg_f0_t384, 0, 384)
PERS_4ET_A12IPG(pers_4et_a12ipg_f1_t384, 1, 384)
PERS_4ET_A12IPG(pers_4et_a12ipg_f2_t384, 2, 384)

#endif
