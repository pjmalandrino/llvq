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

struct SegArgs {
    const unsigned short* x;       // the residual entering ln2, binary16 [hidden]
    const unsigned short* alpha;   // ln2 weights
    const u32* signs_h;            // gate/up rotation
    const float* small_h;          // KMAX x KMAX, +0.0f padded
    const unsigned short* silu;    // 65,536 entries, candle's usilu_f16 of every pattern
    unsigned short* act;           // [inter] the gated activation, global scratch
    unsigned short* out;           // [hidden] x + down(act)
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
    u32 bar_base;                  // pers.rs::Ledger, one barrier a launch
    u32 hidden, inter;
    u32 m_h, k_h, m_i, k_i;
    float inv_h, inv_i, eps;
    u32 gpr;                       // int4 groups per row
};

// Smem: 256-B header | tables 18,816 B | one vector of max(hidden, inter)
// floats, the rotation buffer and the staging at once (natural stride 24, the
// served layout: the rotated vector IS the staged vector).
template <int NORMV, bool TETRA_DOWN>
__device__ __forceinline__ void pers_segment_body(const SegArgs* __restrict__ A)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* v = t_inv + 32;
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = A->pref[i];
        t_suf[i] = A->suf[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const u32 lane = threadIdx.x & 31u;
    const u32 H = A->hidden, I = A->inter;

    // 1. ln2 and the gate/up rotation, in this CTA's shared memory.
    const float scale = exact_rms_scale<NORMV>(A->x, H, A->eps);
    for (u32 i = threadIdx.x; i < H; i += blockDim.x) {
        const float h = h2f(exact_rms_apply(scale, __ldcg(A->x + i), A->alpha[i]));
        v[i] = ((A->signs_h[i >> 5] >> (i & 31u)) & 1u) ? -h : h;
    }
    pers_rot_finish(v, A->small_h, H, A->m_h, A->k_h, A->inv_h);

    // 2. gate and up, row pairs, the SiLU-mul epilogue.
    {
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        pers_claim_reset(A->slice_gu[2u * blockIdx.x]);
        const u32 hi = A->slice_gu[2u * blockIdx.x + 1u];
        for (;;) {
            const u32 r = pers_claim(lane, 1u);
            if (r >= hi) break;
            const float ag = pers_tetra_row<24u, false, false>(A->gate.words + r * A->gate.stride_u32, tab,
                                                               A->gate.gscale, t_inv, v, nb, lane);
            const float au = pers_tetra_row<24u, false, false>(A->up.words + r * A->up.stride_u32, tab,
                                                               A->up.gscale, t_inv, v, nb, lane);
            if (lane == 0u) {
                const unsigned short g = f2h(pers_epilogue(ag, A->gate.rscale[r], A->gate.tail + r * tw, v + nb * 24u, tw));
                const unsigned short u = f2h(pers_epilogue(au, A->up.rscale[r], A->up.tail + r * tw, v + nb * 24u, tw));
                A->act[r] = exact_silu_mul(A->silu, g, u);
            }
        }
    }

    // 3. every row of the activation is written.
    if (!pers_grid_wait(A->bar, A->bar_base + gridDim.x, A->abort_flag)) return;

    // 4. down, then the residual.
    if (TETRA_DOWN) {
        pers_rot_load(A->act, A->signs_i, v, I);
        pers_rot_finish(v, A->small_i, I, A->m_i, A->k_i, A->inv_i);
    } else {
        for (u32 i = threadIdx.x; i < I; i += blockDim.x) v[i] = h2f(__ldcg(A->act + i));
        __syncthreads();
    }
    pers_claim_reset(A->slice_dn[2u * blockIdx.x]);
    const u32 hi = A->slice_dn[2u * blockIdx.x + 1u];
    for (;;) {
        const u32 r = pers_claim(lane, 1u);
        if (r >= hi) break;
        float acc;
        if (TETRA_DOWN) {
            acc = pers_tetra_row<24u, false, false>(A->down.words + r * A->down.stride_u32, tab, A->down.gscale, t_inv,
                                                     v, A->down.nblocks, lane);
        } else {
            acc = pers_q4_row(A->q4w, A->q4s, A->q4b, v, r, I, A->gpr, lane);
        }
        if (lane == 0u) {
            const float y = TETRA_DOWN ? pers_epilogue(acc, A->down.rscale[r], A->down.tail + r * A->down.tail_w,
                                                       v + A->down.nblocks * 24u, A->down.tail_w)
                                       : acc;
            A->out[r] = exact_hadd(A->x[r], f2h(y));
        }
    }
}

#define PERS_SEG(sym, NORMV_, TD_)                                                              \
    extern "C" __global__ void __launch_bounds__(768, 1) sym(const SegArgs* __restrict__ A)  \
    {                                                                                          \
        pers_segment_body<NORMV_, TD_>(A);                                                     \
    }

PERS_SEG(pers_seg_tetra_v0, 0, true)
PERS_SEG(pers_seg_tetra_v1, 1, true)
PERS_SEG(pers_seg_q4_v0, 0, false)
PERS_SEG(pers_seg_q4_v1, 1, false)
