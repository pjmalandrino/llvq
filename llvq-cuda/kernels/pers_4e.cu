// pers_4e.cu: 4E, the whole decode token in one cooperative launch, N tokens
// a launch (persistent.md §3.9). It chains the phases of llvq_pers.cuh, each
// proven bit for bit on the host against the served chain
// (tests/pers_matches_served.rs): the embedding, then per layer the q/k/v
// rows, q/k-norm + RoPE + the KV append, attention (attn_ref.rs's arithmetic,
// under T8), the o rows and the residual, the MLP segment (route P step A),
// then the final norm, the head and the argmax, and the next token.
//
// Waits: one grid barrier at every phase boundary, 2 + 6 x layers a token
// (pers.rs::n_bar_4e): the attention phase has none inside (its split fold
// is a per-KV-head ticket, llvq_pers.cuh section 10), so the count does not
// depend on the position. That is more than the design's 97 at 4B, whose
// finer counters are an optimization, never a correctness requirement. Every
// CTA runs the same phase list and arrives at every barrier, so the count is
// the same on every CTA.
//
// Version 2 (2026-11-05, after the card's phase table, pers-4e-r3): the
// attention phase on attn_dec.cu's mapping, the norm/RoPE heads one a CTA,
// every row with its next block or word in flight, the decoder tables staged
// once a launch, and L2 prefetches of a later phase's rows by CTAs whose own
// phase leaves the DRAM idle. Each keeps every bit (tests/pers_matches_served.rs).
//
// Scope of this version: the 4B and 14B shapes (v and o int4, q and k Tetra
// under one rotation, a Tetra or int4 down per layer, groups of 4 or 5); an
// 8B Tetra o is refused by the host. RoPE's form is a template parameter
// (pers_rope_pair): the card's load-time check picks it against candle.
//
// Composition: the persistent module (pers.rs::PERS_UNIT_PARTS), after
// pers_segment.cu, whose SegArgs it embeds.

#ifndef LLVQ_PERS_CUH
#include "llvq_pers.cuh"
#endif

struct Pers4eLayer {
    const unsigned short* ln1;          // [hidden]
    const u32* signs_a;                 // the q/k rotation
    const float* small_a;
    u32 m_a, k_a;
    float inv_a;
    u32 pad0;
    Pers4eQkv qkv;                      // q, k Tetra; v int4; outputs: the shared q/k/v scratch
    const unsigned short* q_norm;       // [128]
    const unsigned short* k_norm;       // [128]
    unsigned short* kc;                 // [n_kv][cap][128]
    unsigned short* vc;
    const u32* ow;                      // int4 g128 o: [hidden][n_heads*128]
    const unsigned short* osc;
    const unsigned short* obi;
    u32 o_gpr, tetra_down;              // tetra_down: this layer's MLP takes the Tetra down branch
    SegArgs mlp;                        // the layer's MLP segment, step A's own struct
};

struct Pers4eArgs {
    const Pers4eLayer* layers;
    const u32* emb_w;                   // tied int4 g64 table, [vocab][hidden]
    const unsigned short* emb_s;
    const unsigned short* emb_b;
    const unsigned short* final_norm;   // [hidden]
    const unsigned short* cosr;         // [max_pos][64]
    const unsigned short* sinr;
    unsigned short* x_a;                // residual ping-pong, [hidden]
    unsigned short* x_b;
    unsigned short* q_out;              // [n_heads][128] after norm and RoPE
    unsigned short* ctx;                // [n_heads][128]
    float* attn_scratch;                // >= [n_kv][ceil(cap / 256)][groups][130]: split states
    u32* bar;
    u32* abort_flag;
    unsigned long long* slots;          // [2] argmax keys, alternated by position parity; cleared at launch start
    u32* nan_flag;
    const u32* rows;                    // decoder tables
    const unsigned char* pref;
    const unsigned short* br;
    const unsigned char* suf;
    const float* invnorm;
    u32* attn_ticket;                   // [n_kv] the split fold's counters, zero at rest
    u32 n_layers, vocab, hidden, emb_gpr;
    u32 n_heads, n_kv, groups, cap;
    float eps, c;                       // c: 1/sqrt(128) x log2(e), attn_ref::score_scale
};

struct Pers4eLaunch {
    u32 bar_base;                       // arrivals before this launch (pers.rs::Ledger)
    u32 pos0;                           // the position of the first token of this launch
    u32 n_tokens;
    u32 tok0;                           // the token this launch starts from
    u32 eos;
    u32 pad;
    u32* ring;                          // [n_tokens] the tokens produced; the host reads it after the launch
    unsigned long long* stamps;         // STAMP symbols only: [n_tokens][3 + 10 x layers][grid], else unread
};

// Phase stamps (STAMP = true, a measurement symbol; the token symbols are
// STAMP = false and carry none of it). Thread 0 of every CTA writes the grid
// clock at each grid wait: once the whole CTA has finished its phase (after a
// __syncthreads, so the stamp is the CTA's last store, not thread 0's), and
// once the wait has released it. Per token, event e of CTA c lands at
// stamps[(t x E + e) x grid + c], E = 3 + 10 x layers (pers.rs::p4e_events):
//   0, 1               the embedding: arrival, release
//   b = 2 + 15l        layer l, in the order they happen:
//     b                q/k/v: ln1 and its rotation done (thread 0 after a
//                      CTA barrier; no grid wait)
//     b + 1, b + 2     q/k/v: arrival, release
//     b + 3 .. b + 8   norm/RoPE/append, attention, o: arrival, release
//     b + 9 .. b + 12  MLP, inside (pers_segment_body's `sub`): ln2 and its
//                      rotation done, gate/up rows done, inner wait released,
//                      the down's input ready
//     b + 13, b + 14   MLP: arrival, release
//   2 + 15 x layers    the head's W4 released: the token's end
// pers.rs::p4e_phase_report turns them into time per phase.
template <int FORM, bool STAMP, bool PF>
__device__ __forceinline__ void pers_4e_body(const Pers4eArgs* __restrict__ A, const Pers4eLaunch L)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* v = t_inv + 32;  // max(inter, 2 x hidden) floats
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = A->pref[i];
        t_suf[i] = A->suf[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
#if PERS_4E_ROWS >= 5
    // form 5: the launch's arguments read from this CTA's copy (llvq_pers.cuh
    // section 7c); each layer's struct, its slices and its rotations' sign
    // words and `small` copied during the barrier before them
    static_assert(sizeof(Pers4eArgs) <= 4u * (PERS_V7_LAYER - PERS_V7_ARGS), "Pers4eArgs fits the v7 area");
    static_assert(sizeof(Pers4eLayer) <= 4u * (PERS_V7_SLICE - PERS_V7_LAYER), "Pers4eLayer fits the v7 area");
    static_assert(sizeof(Pers4eArgs) % 4u == 0u && sizeof(Pers4eLayer) % 4u == 0u, "copied as words");
    const Pers4eArgs* const Ag = A;
    const u32 v7_vec = Ag->layers[0].mlp.inter > 2u * Ag->hidden ? Ag->layers[0].mlp.inter : 2u * Ag->hidden;
    const u32 v7_off = v7_vec > PERS_ATTN_SMEM(Ag->groups) / 4u ? v7_vec : PERS_ATTN_SMEM(Ag->groups) / 4u;
    float* const area = v + v7_off;
    if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = v7_off;
    pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_ARGS), reinterpret_cast<const u32*>(Ag),
                  (u32)(sizeof(Pers4eArgs) / 4u));
    __syncthreads();
    A = reinterpret_cast<const Pers4eArgs*>(area + PERS_V7_ARGS);
    // Layer l's struct, its CTA slices and ln1's rotation inputs: the staging
    // of the barrier before the layer (no barrier inside).
    const auto stage_layer = [&](u32 l) {
        const Pers4eLayer* g = Ag->layers + l;
        pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_LAYER), reinterpret_cast<const u32*>(g),
                      (u32)(sizeof(Pers4eLayer) / 4u));
        if (threadIdx.x < 4u) {
            const u32* sl = threadIdx.x < 2u ? g->mlp.slice_gu : g->mlp.slice_dn;
            reinterpret_cast<u32*>(area + PERS_V7_SLICE)[threadIdx.x] = sl[2u * blockIdx.x + (threadIdx.x & 1u)];
        }
        pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);
    };
#endif
    __syncthreads();
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const u32 H = A->hidden, G = gridDim.x, c0 = blockIdx.x;
    const u32 nqkv = A->n_heads * 128u + 2u * A->n_kv * 128u;
    u32 nb = 0u;  // barriers passed in this launch
    const u32 E = 3u + 15u * A->n_layers;
#define PERS_BAR() (L.bar_base + (++nb) * G)
#define PERS_STAMP_AT(e)                                                   \
    do {                                                                   \
        if (STAMP && threadIdx.x == 0u) sp[(size_t)(e) * G] = pers_now(); \
    } while (0)
#define PERS_WAIT(e)                                                 \
    do {                                                             \
        if (STAMP) {                                                 \
            __syncthreads();                                         \
            PERS_STAMP_AT(e);                                        \
        }                                                            \
        if (!pers_grid_wait(A->bar, PERS_BAR(), A->abort_flag)) return; \
        PERS_STAMP_AT((e) + 1u);                                     \
    } while (0)
    // Form 5's waits: PERS_WAIT with the stage (a lambda, the variadic
    // arguments so its commas pass) run during the wait (section 7c).
#define PERS_WAIT_GAP(e, ...)                                                             \
    do {                                                                                  \
        if (STAMP) {                                                                      \
            __syncthreads();                                                              \
            PERS_STAMP_AT((e));                                                           \
        }                                                                                 \
        if (!pers_grid_wait_gap(A->bar, PERS_BAR(), A->abort_flag, __VA_ARGS__)) return; \
        PERS_STAMP_AT(1u + (e));                                                          \
    } while (0)
    // Both argmax slots start at zero in every launch. The slot of a position
    // is its parity, and a launch may start at any position: a rewound decode
    // (the measurement's, or a regenerated answer) would otherwise read the
    // key left by the previous launch's last position (the first card reading,
    // job 6aebcb97). CTA 0 clears them before the first barrier, whose fence
    // orders the stores before any CTA's atomicMax.
    // The split fold's tickets too: a launch that aborted may have left one
    // drawn (the fold resets them itself otherwise).
    if (c0 == 0u && threadIdx.x == 0u) {
        A->slots[0] = 0ull;
        A->slots[1] = 0ull;
    }
    if (c0 == 0u)
        for (u32 i = threadIdx.x; i < A->n_kv; i += blockDim.x) A->attn_ticket[i] = 0u;
    u32 tok = L.tok0;
    for (u32 t = 0; t < L.n_tokens; ++t) {
        const u32 pos = L.pos0 + t;
        // this token's stamps of this CTA: event e at sp[e x grid]
        unsigned long long* sp = STAMP ? L.stamps + (size_t)t * E * G + c0 : nullptr;
        // embedding: this CTA's slice of the residual
        pers_embed(A->emb_w, A->emb_s, A->emb_b, tok, H, A->emb_gpr, A->x_a, H * c0 / G, H * (c0 + 1u) / G);
#if PERS_4E_ROWS >= 5
        PERS_WAIT_GAP(0u, [&] { stage_layer(0u); });
#else
        PERS_WAIT(0u);
#endif
        for (u32 l = 0; l < A->n_layers; ++l) {
#if PERS_4E_ROWS >= 5
            const Pers4eLayer& Y = *reinterpret_cast<const Pers4eLayer*>(area + PERS_V7_LAYER);
#else
            const Pers4eLayer& Y = A->layers[l];
#endif
            const u32 e0 = 2u + 15u * l;  // this layer's first stamp event
            // L2 prefetch: this CTA's q/k/v rows, asked for before ln1 and
            // the rotation every CTA computes first (the rows pers_4e_qkv_rows
            // gives the CTA: its share of q | k and of v under PF).
            if (PF) {
                const Pers4eQkv& N = Y.qkv;
                const u32 qk = N.q_dout + N.k_dout, lo = qk * c0 / G, hi = qk * (c0 + 1u) / G;
                if (lo < N.q_dout) pers_l2_prefetch(N.qw + (unsigned long long)lo * N.q_stride, 4ull * ((hi < N.q_dout ? hi : N.q_dout) - lo) * N.q_stride);
                if (hi > N.q_dout) {
                    const u32 a = lo > N.q_dout ? lo - N.q_dout : 0u, b = hi - N.q_dout;
                    pers_l2_prefetch(N.kw + (unsigned long long)a * N.k_stride, 4ull * (b - a) * N.k_stride);
                }
                const u32 vlo = N.v_dout * c0 / G, vhi = N.v_dout * (c0 + 1u) / G;
                pers_l2_prefetch(N.vw + (unsigned long long)vlo * (N.d_in >> 3), 4ull * (vhi - vlo) * (N.d_in >> 3));
            } else {
                const Pers4eQkv& N = Y.qkv;
                const u32 lo = nqkv * c0 / G, hi = nqkv * (c0 + 1u) / G;
                const u32 qe = N.q_dout, ke = N.q_dout + N.k_dout;
                if (lo < qe) pers_l2_prefetch(N.qw + (unsigned long long)lo * N.q_stride, 4ull * ((hi < qe ? hi : qe) - lo) * N.q_stride);
                if (hi > qe && lo < ke) {
                    const u32 a = lo > qe ? lo - qe : 0u, b = (hi < ke ? hi : ke) - qe;
                    pers_l2_prefetch(N.kw + (unsigned long long)a * N.k_stride, 4ull * (b - a) * N.k_stride);
                }
                if (hi > ke) {
                    const u32 a = lo > ke ? lo - ke : 0u, b = hi - ke;
                    pers_l2_prefetch(N.vw + (unsigned long long)a * (N.d_in >> 3), 4ull * (b - a) * (N.d_in >> 3));
                }
            }
            // ln1, the q/k rotation, the q/k/v rows
            float* xr = v;
            float* xn = v + H;
#if PERS_4E_ROWS == 4
            // form 4: one round trip; `small` staged after the row vector
            float* ssm = v + (Y.mlp.inter > 2u * H ? Y.mlp.inter : 2u * H);
            if (pers_1rt_ok(A->x_a, H) && pers_1rt_ok(Y.ln1, H)) {
                pers_ln_1rt<0>(A->x_a, Y.ln1, Y.signs_a, H, A->eps, Y.small_a, Y.k_a, ssm, xr, xn, true);
                pers_rot_finish<true>(xr, ssm, H, Y.m_a, Y.k_a, Y.inv_a);
            } else
#elif PERS_4E_ROWS >= 5
            // form 5: the sign words and `small` were staged during the
            // barrier before this layer; x and ln1 are the only loads left
            float* ssm = area + PERS_V7_SMALL;
            if (pers_1rt_ok(A->x_a, H) && pers_1rt_ok(Y.ln1, H)) {
                pers_ln_1rt<0>(A->x_a, Y.ln1, reinterpret_cast<const u32*>(area + PERS_V7_SIGNS), H, A->eps, nullptr,
                               Y.k_a, ssm, xr, xn, true);
                pers_rot_finish<true, true>(xr, ssm, H, Y.m_a, Y.k_a, Y.inv_a);
            } else
#endif
            {
                const float sc = exact_rms_scale<0>(A->x_a, H, A->eps);
                for (u32 i = threadIdx.x; i < H; i += blockDim.x) {
                    const float hv = h2f(exact_rms_apply(sc, __ldcg(A->x_a + i), Y.ln1[i]));
                    xn[i] = hv;
                    xr[i] = ((Y.signs_a[i >> 5] >> (i & 31u)) & 1u) ? -hv : hv;
                }
                pers_rot_finish(xr, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            }
            if (STAMP) {
                __syncthreads();
                PERS_STAMP_AT(e0);
            }
            pers_4e_qkv_rows<PF>(Y.qkv, tab, t_inv, xr, xn, c0, G,
                                 v + (Y.mlp.inter > 2u * H ? Y.mlp.inter : 2u * H));
            PERS_WAIT(e0 + 1u);
            pers_4e_norm_rope_append<FORM>(Y.qkv.q, Y.qkv.k, Y.qkv.v, Y.q_norm, Y.k_norm, A->cosr + pos * 64u,
                                           A->sinr + pos * 64u, A->eps, A->n_heads, A->n_kv, A->cap, pos, A->q_out,
                                           Y.kc, Y.vc);
            PERS_WAIT(e0 + 3u);
            // L2 prefetch: this CTA's o rows and gate/up rows, the next two
            // row phases. A CTA with no attention job asks at once; a CTA with
            // one asks after it, so the prefetches never delay attention's
            // own loads (v2's card table: 8 CTAs carried the phase at a short
            // context, 737 us against a mean of 260).
            const u32 attn_jobs = ((pos + 1u + 255u) / 256u) * A->n_kv;
            const auto prefetch_next = [&] {
                const u32 olo = H * c0 / G, ohi = H * (c0 + 1u) / G, dctx = A->n_heads * 128u;
                pers_l2_prefetch(Y.ow + (unsigned long long)olo * (dctx >> 3), 4ull * (ohi - olo) * (dctx >> 3));
                const u32 glo = Y.mlp.slice_gu[2u * c0], ghi = Y.mlp.slice_gu[2u * c0 + 1u];
                pers_l2_prefetch(Y.mlp.gate.words + (unsigned long long)glo * Y.mlp.gate.stride_u32,
                                 4ull * (ghi - glo) * Y.mlp.gate.stride_u32);
                pers_l2_prefetch(Y.mlp.up.words + (unsigned long long)glo * Y.mlp.up.stride_u32,
                                 4ull * (ghi - glo) * Y.mlp.up.stride_u32);
            };
            if (c0 >= attn_jobs) prefetch_next();
            pers_attention<PF ? 2u : 1u>(A->q_out, Y.kc, Y.vc, A->n_heads, A->groups, A->cap, pos + 1u, A->c,
                                         A->attn_scratch, A->attn_ticket, A->ctx, reinterpret_cast<unsigned char*>(v),
                                         nullptr);
            if (c0 < attn_jobs) prefetch_next();
            PERS_WAIT(e0 + 5u);
            pers_4e_o_int4<PF>(Y.ow, Y.osc, Y.obi, Y.o_gpr, A->ctx, v, A->n_heads * 128u, A->x_a, A->x_b, H * c0 / G,
                           H * (c0 + 1u) / G);
#if PERS_4E_ROWS >= 5
            // ln2's rotation inputs, during the o phase's barrier
            PERS_WAIT_GAP(e0 + 7u, [&] { pers_v7_stage_rot(area, Y.mlp.signs_h, H, Y.mlp.small_h, Y.mlp.k_h); });
#else
            PERS_WAIT(e0 + 7u);
#endif
            // the MLP segment: its own barrier is the next ordinal
            const SegLaunch S{A->x_b, A->x_a, L.bar_base + nb * G, 0u};
            ++nb;
            unsigned long long* sub = STAMP ? sp + (size_t)(e0 + 9u) * G : nullptr;
            if (Y.tetra_down) pers_segment_body<0, true, true, PF>(&Y.mlp, S, sub, G);
            else pers_segment_body<0, false, true, PF>(&Y.mlp, S, sub, G);
#if PERS_4E_ROWS >= 5
            // the next layer's struct and ln1 inputs, during this layer's last
            // barrier (nothing of this layer is read after the arrival)
            PERS_WAIT_GAP(e0 + 13u, [&] {
                if (l + 1u < A->n_layers) stage_layer(l + 1u);
            });
#else
            PERS_WAIT(e0 + 13u);
#endif
        }
        // the final norm, the head, the argmax (its own barrier is W4)
        const float sc = exact_rms_scale<0>(A->x_a, H, A->eps);
        for (u32 i = threadIdx.x; i < H; i += blockDim.x) v[i] = h2f(exact_rms_apply(sc, __ldcg(A->x_a + i), A->final_norm[i]));
        __syncthreads();
        const u32 next = pers_head_argmax<PF>(A->emb_w, A->emb_s, A->emb_b, v, H, A->emb_gpr, A->vocab * c0 / G,
                                          A->vocab * (c0 + 1u) / G, A->slots + (pos & 1u), A->nan_flag, A->bar,
                                          PERS_BAR(), A->abort_flag, nullptr);
        if (next == 0xffffffffu) return;  // a NaN logit or an abort: the host reads the flags
        PERS_STAMP_AT(E - 1u);
        if (c0 == 0u && threadIdx.x == 0u) {
            L.ring[t] = next;
            // the next position's slot, cleared for the next token of this
            // launch; this one is still being read
            A->slots[(pos + 1u) & 1u] = 0ull;
        }
        tok = next;
        if (tok == L.eos) return;
    }
#undef PERS_WAIT
#undef PERS_WAIT_GAP
#undef PERS_STAMP_AT
#undef PERS_BAR
}

// One symbol a (RoPE form, block size). PF, the rows' next block or word in
// flight, costs registers: under the devel ptxas 12.4 the 512-thread symbols
// hold 128 registers and no local memory with it, and the 768-thread ones
// (85 registers a thread at most) spilled 176 B with it. So the 768-thread
// symbols hide the load latency with 24 warps a SM instead, and the
// 512-thread ones with the prefetch; both keep every bit, and the card times
// whichever loads without local memory (persist4e_cuda.rs).
#define PERS_4E(sym, FORM_, THREADS_, STAMP_, PF_)                                                       \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,      \
                                                                   const Pers4eLaunch L)                 \
    {                                                                                                    \
        pers_4e_body<FORM_, STAMP_, PF_>(A, L);                                                          \
    }

PERS_4E(pers_4e_tokens_f0, 0, 768, false, false)
PERS_4E(pers_4e_tokens_f1, 1, 768, false, false)
PERS_4E(pers_4e_tokens_f2, 2, 768, false, false)
PERS_4E(pers_4e_tokens_f0_t512, 0, 512, false, true)
PERS_4E(pers_4e_tokens_f1_t512, 1, 512, false, true)
PERS_4E(pers_4e_tokens_f2_t512, 2, 512, false, true)
// The phase-stamp symbols (LLVQ_PERS_4E's stamped launch), one a geometry:
// the same tokens or the run is void.
PERS_4E(pers_4e_stamps_f0_t512, 0, 512, true, true)
PERS_4E(pers_4e_stamps_f1_t512, 1, 512, true, true)
PERS_4E(pers_4e_stamps_f2_t512, 2, 512, true, true)
PERS_4E(pers_4e_stamps_f0_t768, 0, 768, true, false)
PERS_4E(pers_4e_stamps_f1_t768, 1, 768, true, false)
PERS_4E(pers_4e_stamps_f2_t768, 2, 768, true, false)

// The RoPE load check (persist4e_cuda.rs): each form applied to n quads
// (a, b, cos, sin) the host took from candle's own rope call on the card,
// whose outputs it compares bit for bit. The form that matches is the one
// the token loop runs; none matching keeps 4E off.
template <int FORM>
__device__ __forceinline__ void pers_rope_probe_body(const unsigned short* a, const unsigned short* b,
                                                     const unsigned short* c, const unsigned short* s,
                                                     unsigned short* o1, unsigned short* o2, u32 n)
{
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        pers_rope_pair<FORM>(a[i], b[i], c[i], s[i], o1 + i, o2 + i);
}

#define PERS_ROPE_PROBE(sym, FORM_)                                                                      \
    extern "C" __global__ void sym(const unsigned short* a, const unsigned short* b, const unsigned short* c, \
                                   const unsigned short* s, unsigned short* o1, unsigned short* o2, u32 n)   \
    {                                                                                                    \
        pers_rope_probe_body<FORM_>(a, b, c, s, o1, o2, n);                                              \
    }

PERS_ROPE_PROBE(pers_rope_probe_f0, 0)
PERS_ROPE_PROBE(pers_rope_probe_f1, 1)
PERS_ROPE_PROBE(pers_rope_probe_f2, 2)

#if PERS_4E_ROWS >= 7
// Form 7's words (pers_q4_tile_index), written once at install from the
// served record into a buffer of the same size: the bits move, none changes.
extern "C" __global__ void pers_4e_q4_tile_relayout(const u32* __restrict__ src, u32* __restrict__ dst, u32 d_out,
                                                    u32 nwords)
{
    const unsigned long long n = (unsigned long long)d_out * nwords;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const u32 r = (u32)(i / nwords), wi = (u32)(i - (unsigned long long)r * nwords);
        dst[(size_t)r * nwords + pers_q4_tile_index(wi, nwords)] = src[i];
    }
}
#endif
