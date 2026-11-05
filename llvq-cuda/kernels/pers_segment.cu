// pers_segment.cu: route P, step A. One cooperative launch per layer runs the
// MLP half of a decode step, from `ln2` to the residual add after down_proj,
// and keeps every bit of the served path.
//
// Design: docs/archive/debit-vague5-2026-11-02/persistent.md §3.1 (step A),
// §3.5-3.7. Host half: src/pers.rs (step_a_plan, slices, Ledger).
//
// The served path this replaces, one decode row, f16 throughout, 8 launches
// for a Tetra down and 6 for an int4 one (Block::finish, model.rs):
//   h0  = rmsnorm_f16(x, ln2)                    candle, block 1,024
//   xr  = rot_apply(h0)                          f32, shared by gate and up
//   g   = tv_tetra48_h(gate, xr); u = tv_tetra48_h(up, xr)
//   a   = bmul_f16(usilu_f16(g), u)              candle, two launches
//   dn  = tv_tetra48_h(down, rot_apply(a))       Tetra down
//       | tv_q4_hh(down, a)                      int4 down, natural basis
//   out = badd_f16(x, dn)                        candle
//
// Here, one launch:
//   1. every CTA computes candle's RMSNorm of x and the rotation of the result
//      into its own shared memory (redundant work, no wait);
//   2. every CTA takes a static slice of the gate/up rows; a warp computes gate
//      row r then up row r, and lane 0 writes a[r] = silu(g) * u to global;
//   3. one grid barrier: every a[r] is written;
//   4. every CTA reads a whole (`__ldcg`, written in this launch), rotates it
//      (Tetra down) or widens it (int4 down), takes its slice of the down rows,
//      and writes out[r] = x[r] + dn[r].
// No CTA reads `out`, and `x` is never written: the residual buffers are the
// caller's two tensors. The launch boundary orders the next layer.
//
// Arithmetic: llvq_pers.cuh. Each op is the served op on the same values:
// the Tetra rows and the rotation by construction, RMSNorm by the replay of
// candle's tree (VARIANT picked at load against candle), SiLU by candle's own
// table, the f16 mul and add exactly through f32.
//
// Composition: as tetra48_pers.cu (llvq_pers.cuh first), its own module.

#ifndef LLVQ_PERS_CUH
#include "llvq_pers.cuh"
#endif

struct SegTetra {
    const u32* words;              // row-strided stream
    const float* rscale;           // d_out
    const unsigned short* tail;    // d_out * tail_w, binary16
    const float* gscale;           // 2
    u32 stride_u32, nblocks, tail_w, d_out;
};

// One layer's constants, uploaded once (persist_cuda.rs). Pointers first,
// then 4-byte fields: 27 pointers (3 SegTetra of 4 each inside), 360 bytes.
struct SegArgs {
    const unsigned short* alpha;   // ln2 weights, binary16 [hidden]
    const u32* signs_h;            // gate/up rotation
    const float* small_h;          // KMAX x KMAX, +0.0f padded
    const unsigned short* silu;    // 65,536 entries, candle's usilu_f16 of every pattern
    unsigned short* act;           // [inter] the gated activation, global scratch
    SegTetra gate, up, down;       // down used when the kernel is the Tetra one
    const u32* signs_i;            // down rotation (Tetra down)
    const float* small_i;
    const u32* q4w;                // int4 down: words, scales, biases (int4 kernel)
    const unsigned short* q4s;
    const unsigned short* q4b;
    const u32* rows;               // decoder tables, global
    const unsigned char* pref;
    const unsigned short* br;
    const unsigned char* suf;
    const float* invnorm;
    const u32* slice_gu;           // [2 * grid]: lo, hi of each CTA's gate/up rows
    const u32* slice_dn;           // [2 * grid]: the same for the down rows
    u32* bar;
    u32* abort_flag;
    u32 hidden, inter;
    u32 m_h, k_h, m_i, k_i;
    float inv_h, inv_i, eps;
    u32 gpr;                       // int4 groups per row
};

// What changes with every call, passed by value: the two residual tensors
// candle allocated for this step, and the barrier counter's base
// (pers.rs::Ledger, one barrier a launch). 24 bytes.
struct SegLaunch {
    const unsigned short* x;       // the residual entering ln2, binary16 [hidden]
    unsigned short* out;           // [hidden] x + down(act)
    u32 bar_base;
    u32 pad;
};

// Smem: 256-B header | tables 18,816 B | one vector of max(hidden, inter)
// floats, the rotation buffer and the staging at once (natural stride 24, the
// served layout: the rotated vector IS the staged vector).
//
// IN4E (4E's call, never route P's launches): the tables are already in this
// CTA's shared memory, staged once at the start of the launch, and nothing
// in 4E writes them, so the segment does not restage them every layer; and
// the CTA asks L2 for its slice of the down rows before it waits for the
// activation. PF: the Tetra and int4 rows keep the next block or word in
// flight (PREFETCH, the same order of accumulation). None of it moves a bit.
template <int NORMV, bool TETRA_DOWN, bool IN4E = false, bool PF = false>
// `sub` (4E's stamp symbols only, null otherwise): four stamps inside the
// segment, each by thread 0 once the whole CTA is past the point, at
// sub[k x stride] for k = 0..3: ln2 and its rotation done, the gate/up rows
// done (the inner wait's arrival), the inner wait released, the down's input
// ready (its rotation, or the int4 widening).
__device__ __forceinline__ void pers_segment_body(const SegArgs* __restrict__ A, const SegLaunch L,
                                                  unsigned long long* sub = nullptr, u32 stride = 0u)
{
#define PERS_SEG_SUB(k)                                                  \
    do {                                                                 \
        if (sub != nullptr) {                                            \
            __syncthreads();                                             \
            if (threadIdx.x == 0u) sub[(size_t)(k) * stride] = pers_now(); \
        }                                                                \
    } while (0)
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* v = t_inv + 32;
    if (!IN4E) {
        for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
        for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
            t_pref[i] = A->pref[i];
            t_suf[i] = A->suf[i];
        }
        for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
        for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
    }
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const u32 lane = threadIdx.x & 31u;
    const u32 H = A->hidden, I = A->inter;

    // 1. ln2 and the gate/up rotation, in this CTA's shared memory.
#if PERS_4E_ROWS == 4
    // form 4 (4E only): one round trip; x staged in v + H, `small` after the
    // row vector, the k rows of the down's rotation there later
    float* ssm = v + (I > 2u * H ? I : 2u * H);
    if (IN4E && pers_1rt_ok(L.x, H) && pers_1rt_ok(A->alpha, H)) {
        pers_ln_1rt<NORMV>(L.x, A->alpha, A->signs_h, H, A->eps, A->small_h, A->k_h, ssm, v, v + H, false);
        pers_rot_finish<true>(v, ssm, H, A->m_h, A->k_h, A->inv_h);
    } else
#endif
    {
        const float scale = exact_rms_scale<NORMV>(L.x, H, A->eps);
        for (u32 i = threadIdx.x; i < H; i += blockDim.x) {
            const float h = h2f(exact_rms_apply(scale, __ldcg(L.x + i), A->alpha[i]));
            v[i] = ((A->signs_h[i >> 5] >> (i & 31u)) & 1u) ? -h : h;
        }
        pers_rot_finish(v, A->small_h, H, A->m_h, A->k_h, A->inv_h);
    }
    PERS_SEG_SUB(0u);

    // 2. gate and up, row pairs, the SiLU-mul epilogue.
#if PERS_4E_ROWS == 3
    // form 3: the CTA's gate/up rows as one list, gate of pair p at 2p, up at
    // 2p + 1; their binary16 values in the first 2 KB of the scratch (1,024
    // values: 512 pairs a CTA, 69 on the 4B's 142 CTAs), the products after a
    // barrier (exact_hmul(silu[g], u), the stream's own product). A CTA with
    // more pairs takes form 2's path below.
    float* scratch = v + (I > 2u * H ? I : 2u * H);
    const u32 gu_pairs = A->slice_gu[2u * blockIdx.x + 1u] - A->slice_gu[2u * blockIdx.x];
    if (IN4E && PF && gu_pairs <= 512u) {
        const u32 glo = A->slice_gu[2u * blockIdx.x], ghi = A->slice_gu[2u * blockIdx.x + 1u];
        unsigned short* sgu = reinterpret_cast<unsigned short*>(scratch);
        float* fold = scratch + 512u;  // after 1,024 binary16 values
        const auto row_at = [&](u32 idx) {
            const u32 r = glo + (idx >> 1);
            const SegTetra& M = (idx & 1u) ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto epi = [&](u32 idx, float y) { sgu[idx] = f2h(y); };
        pers_4e_rows_cta(2u * (ghi - glo), A->gate.nblocks, A->gate.tail_w, tab, t_inv, v, row_at, epi, fold,
                         PERS_4E_SCRATCH / 4u - 512u);
        __syncthreads();
        for (u32 p = threadIdx.x; p < ghi - glo; p += blockDim.x) A->act[glo + p] = exact_hmul(A->silu[sgu[2u * p]], sgu[2u * p + 1u]);
    } else
#endif
    if (IN4E && PF) {
        // The stream (llvq_pers.cuh): this CTA's pairs static by warp, gate
        // then up as consecutive rows. Lane 0 keeps g; at up it issues the
        // SiLU table load and stores the PREVIOUS pair's activation, so the
        // table's round trip runs under the next pair's blocks. The same g, u,
        // table entry and product: the same act.
        const u32 warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
        const u32 glo = A->slice_gu[2u * blockIdx.x], ghi = A->slice_gu[2u * blockIdx.x + 1u];
        const u32 npairs = pers_warp_rows(glo, ghi, warp, nw);
        unsigned short g = 0u, pend_sv = 0u, pend_u = 0u;
        u32 pend_r = 0xffffffffu;
        const auto row_of = [&](u32 k) {
            const u32 r = glo + warp + (k >> 1) * nw;
            const SegTetra& M = (k & 1u) ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) {
            if ((k & 1u) == 0u) {
                g = f2h(y);
                return;
            }
            if (pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
            pend_sv = A->silu[g];
            pend_u = f2h(y);
            pend_r = glo + warp + (k >> 1) * nw;
        };
        pers_4e_rows(2u * npairs, A->gate.nblocks, A->gate.tail_w, tab, t_inv, v, lane, row_of, start, epi);
        if (lane == 0u && pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
    } else {
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        pers_claim_reset(A->slice_gu[2u * blockIdx.x]);
        const u32 hi = A->slice_gu[2u * blockIdx.x + 1u];
        for (;;) {
            const u32 r = pers_claim(lane, 1u);
            if (r >= hi) break;
            const float ag = pers_tetra_row<24u, false, PF>(A->gate.words + r * A->gate.stride_u32, tab,
                                                               A->gate.gscale, t_inv, v, nb, lane);
            const float au = pers_tetra_row<24u, false, PF>(A->up.words + r * A->up.stride_u32, tab,
                                                               A->up.gscale, t_inv, v, nb, lane);
            if (lane == 0u) {
                const unsigned short g = f2h(pers_epilogue(ag, A->gate.rscale[r], A->gate.tail + r * tw, v + nb * 24u, tw));
                const unsigned short u = f2h(pers_epilogue(au, A->up.rscale[r], A->up.tail + r * tw, v + nb * 24u, tw));
                A->act[r] = exact_silu_mul(A->silu, g, u);
            }
        }
    }
    (void)lane;

    // 3. every row of the activation is written. 4E: this CTA's down rows are
    // asked of L2 first, while the slowest CTA finishes its gate/up rows.
    if (IN4E) {
        const u32 dlo = A->slice_dn[2u * blockIdx.x], dhi = A->slice_dn[2u * blockIdx.x + 1u];
        if (TETRA_DOWN) {
            pers_l2_prefetch(A->down.words + (unsigned long long)dlo * A->down.stride_u32,
                             4ull * (dhi - dlo) * A->down.stride_u32);
        } else {
            pers_l2_prefetch(A->q4w + (unsigned long long)dlo * (I >> 3), 4ull * (dhi - dlo) * (I >> 3));
        }
    }
    PERS_SEG_SUB(1u);
    if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x, A->abort_flag)) return;
    PERS_SEG_SUB(2u);

    // 4. down, then the residual.
#if PERS_4E_ROWS == 4
    // form 4 (4E only): the activation as vectors, the signs and the k rows of
    // `small` in the same batch
    if (IN4E && pers_1rt_ok(A->act, I)) {
        if (TETRA_DOWN) {
            pers_load_1rt(A->act, A->signs_i, v, I, A->small_i, A->k_i, ssm);
            pers_rot_finish<true>(v, ssm, I, A->m_i, A->k_i, A->inv_i);
        } else {
            pers_load_1rt(A->act, nullptr, v, I, nullptr, 0u, nullptr);
            __syncthreads();
        }
    } else
#endif
    if (TETRA_DOWN) {
        pers_rot_load(A->act, A->signs_i, v, I);
        pers_rot_finish(v, A->small_i, I, A->m_i, A->k_i, A->inv_i);
    } else {
        for (u32 i = threadIdx.x; i < I; i += blockDim.x) v[i] = h2f(__ldcg(A->act + i));
        __syncthreads();
    }
    PERS_SEG_SUB(3u);
    if (IN4E && PF) {
        // The stream for a Tetra down; the int4 down static by warp with its
        // word prefetch. Lane 0 loads x[r] when the row starts.
        const u32 warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
        const u32 dlo = A->slice_dn[2u * blockIdx.x], dhi = A->slice_dn[2u * blockIdx.x + 1u];
        if (TETRA_DOWN) {
#if PERS_4E_ROWS == 3
            {
                const auto row_at3 = [&](u32 idx) {
                    const u32 r = dlo + idx;
                    return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
                };
                const auto epi3 = [&](u32 idx, float y) { L.out[dlo + idx] = exact_hadd(__ldcg(L.x + dlo + idx), f2h(y)); };
                pers_4e_rows_cta(dhi - dlo, A->down.nblocks, A->down.tail_w, tab, t_inv, v, row_at3, epi3, scratch,
                                 PERS_4E_SCRATCH / 4u);
                (void)warp;
                (void)nw;
                return;
            }
#endif
            unsigned short xr = 0u;
            const auto row_of = [&](u32 k) {
                const u32 r = dlo + warp + k * nw;
                return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
            };
            const auto start = [&](u32 k) { xr = __ldcg(L.x + dlo + warp + k * nw); };
            const auto epi = [&](u32 k, float y) { L.out[dlo + warp + k * nw] = exact_hadd(xr, f2h(y)); };
            pers_4e_rows(pers_warp_rows(dlo, dhi, warp, nw), A->down.nblocks, A->down.tail_w, tab, t_inv, v, lane,
                         row_of, start, epi);
        } else {
            for (u32 r = dlo + warp; r < dhi; r += nw) {
                const float acc = pers_q4_row<true>(A->q4w, A->q4s, A->q4b, v, r, I, A->gpr, lane);
                if (lane == 0u) L.out[r] = exact_hadd(__ldcg(L.x + r), f2h(acc));
            }
        }
        return;
    }
    pers_claim_reset(A->slice_dn[2u * blockIdx.x]);
    const u32 hi = A->slice_dn[2u * blockIdx.x + 1u];
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        float acc;
        if (TETRA_DOWN) {
            acc = pers_tetra_row<24u, false, PF>(A->down.words + r * A->down.stride_u32, tab, A->down.gscale, t_inv,
                                                     v, A->down.nblocks, lane);
        } else {
            acc = pers_q4_row<PF>(A->q4w, A->q4s, A->q4b, v, r, I, A->gpr, lane);
        }
        if (lane == 0u) {
            const float y = TETRA_DOWN ? pers_epilogue(acc, A->down.rscale[r], A->down.tail + r * A->down.tail_w,
                                                       v + A->down.nblocks * 24u, A->down.tail_w)
                                       : acc;
            L.out[r] = exact_hadd(__ldcg(L.x + r), f2h(y));
        }
    }
#undef PERS_SEG_SUB
}

#define PERS_SEG(sym, NORMV_, TD_)                                                                   \
    extern "C" __global__ void __launch_bounds__(768, 1) sym(const SegArgs* __restrict__ A, const SegLaunch L) \
    {                                                                                               \
        pers_segment_body<NORMV_, TD_>(A, L);                                                       \
    }

PERS_SEG(pers_seg_tetra_v0, 0, true)
PERS_SEG(pers_seg_tetra_v1, 1, true)
PERS_SEG(pers_seg_q4_v0, 0, false)
PERS_SEG(pers_seg_q4_v1, 1, false)
