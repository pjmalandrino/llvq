// tetra48_pers.cu: the persistent Tetra bench arms (step L2) and the barrier
// microbenchmarks (step L1).
//
// Design: docs/archive/debit-vague5-2026-11-02/persistent.md §3, §5. Host
// half: src/pers.rs (arm table, site list, slices, barrier ledger). Prereg:
// proofs/preregistration-pers-l2-*.md (arms), -l1-* (microbenchmarks).
//
// One launch walks the 252 sites of the bare Tetra file in served order, one
// CTA per SM, `__launch_bounds__(T, 1)`, launched cooperatively after the
// occupancy check. One template body; each arm is one instantiation behind
// its own extern "C" symbol, so the unit never changes with the selection.
//
//   arm            symbol                 MODE THREADS STRIDE TRIO PREFETCH WAITS PROLOGUE
//   pnull          tv_pers_null            0    768     28    yes    no      0      no
//   pword          tv_pers_word            1    768     28    yes    no      0      no
//   pdec_l1        tv_pers_dec_l1          4    768     28    yes    no      0      no
//   pdec_smem      tv_pers_dec_smem        5    768     28    yes    no      0      no
//   pl1            tv_pers_l1              2    768     24    no     no      0      no
//   pl1_t28        tv_pers_l1_t28          2    768     28    yes    no      0      no
//   psmem          tv_pers_smem            3    768     24    no     no      0      no
//   psmem_s25      tv_pers_smem_s25        3    768     25    no     no      0      no
//   psmem_t28      tv_pers_smem_t28        3    768     28    yes    no      0      no
//   psmem_t28_u2   tv_pers_smem_t28_u2     3    768     28    yes    yes     0      no
//   pt28_512       tv_pers_t28_512         3    512     28    yes    no      0      no
//   pt28_1024      tv_pers_t28_1024        3   1024     28    yes    no      0      no
//   pchain_w       tv_pers_chain_w         3    768     28    yes    no      1      no
//   pchain_w2      tv_pers_chain_w2        3    768     28    yes    no      2      no
//   pchain_wp      tv_pers_chain_wp        3    768     28    yes    no      1      yes
//
// MODE: 0 staging and reduction only, 1 every word read and XOR-folded, 2 full
// decode with the tables read in place (L1), 3 full decode with the tables
// copied to shared memory once a launch, 4 and 5 the decode of modes 2 and 3
// without the dot: the 24 values built and XOR-folded as bits, the shell and
// both scales read, no activation read and no FMA (the decode arms of
// docs/mesures/pers-sass-2026-11-05.txt). WAITS: 0 none; 1 the served 4B
// schedule (95 barrier ordinals, 48 counter waits every CTA's signal turns
// into grid-wide waits, 2 trailing barriers: 145 grid-wide syncs); 2 the same
// with one more grid barrier after each sync.
//
// Every arm runs the same rows: every site ends with __syncthreads() before
// the claim cursor is set again (pers_claim_reset), in every mode, so a warp
// still claiming site s never draws an index of site s + 1. The draft's
// waitless arms reset the cursor without that barrier and skipped rows.
//
// Composition: llvq_slot.cuh, matvec.cu, llvq_f1rank.cuh, llvq_f1rank_v3.cuh,
// llvq_tetra48.cuh, llvq_rot.cuh, llvq_pers.cuh, then this file. Its own NVRTC
// module: the bench unit and the served unit keep their bytes.

#ifndef LLVQ_PERS_CUH
#include "llvq_pers.cuh"
#endif

// The device site record, pers.rs::site_record, 8 u32.
struct PersSite {
    u32 words_lo, words_hi;  // row-strided word stream
    u32 d_out;
    u32 nb_tw;               // nblocks | tail_w << 16
    u32 row0_in_y;           // the site's first row in y_all
    u32 stride_u32;          // tetra48::stride_u32(nblocks)
    u32 wait;                // bits 0-1: 0 none, 1 grid barrier, 2 counter; producer site << 2
    u32 flags;               // bit 0: signal after; bits 1-2: prologue 0 none, 1 norm + rot, 2 rot;
                             // bits 8-31: layer
};

// Uploaded once; its address is every arm's first argument. Pointers first,
// then 4-byte fields, so the host mirror (persbench.rs) packs it without
// padding rules: 22 pointers, 10 u32, 3 f32: 228 bytes, 232 with the tail pad.
struct PersArgs {
    const PersSite* sites;
    const u32* rows;                       // decoder tables, global copies
    const unsigned char* pref;
    const unsigned short* br;
    const unsigned char* suf;
    const float* invnorm;                  // 32 floats
    const float* const* gscale;            // per site, 2 floats
    const float* const* rscale;            // per site, d_out floats
    const unsigned short* const* tail;     // per site, d_out * tail_w binary16
    const float* x;                        // the bench activation, f32, width >= 9728
    unsigned short* y_all;                 // every site's rows; poisoned before an identity launch
    const u32* slice_lo;                   // [nsites * grid], pers.rs::slice_tables
    const u32* slice_hi;
    u32* bar;                              // monotonic barrier counter, life of the runtime
    u32* cnts;                             // counters; each arm's region from PersLaunch
    u32* abort_flag;
    // prologue inputs (pchain_wp): a fixed binary16 residual, RMSNorm weights,
    // the two rotations' sign bitmaps and padded odd blocks
    const unsigned short* resid;           // width >= inter
    const unsigned short* alpha;           // width hidden
    const u32* signs_h;
    const u32* signs_i;
    const float* small_h;                  // KMAX x KMAX, +0.0f padded
    const float* small_i;
    u32 nsites;
    u32 hidden, inter;                     // prologue widths
    u32 m_h, k_h, m_i, k_i;                // their rotation plans (pers.rs::rot_mk)
    u32 pad0, pad1, pad2;
    float inv_h, inv_i;                    // 1/sqrt(m) as the host passes rot_apply
    float eps;
};

// What changes from one launch to the next, passed by value so a timed launch
// needs no copy: the barrier and counter state (pers.rs::Ledger) and the
// identity launch's debug outputs (null on timed launches, so the timed code
// and the checked code are one symbol). 40 bytes.
struct PersLaunch {
    u32 bar_base;                          // arrivals before this launch
    u32 epoch;                             // this arm's earlier signalling launches
    u32 cnt_offset;                        // this arm's counter region
    u32 pad;
    float* dbg;                            // the f32 value before f2h, [rows]
    unsigned short* dbg_norm;              // CTA 0's normed vectors, [n_norm][hidden]
    float* dbg_rot;                        // CTA 0's rotated vectors, [n_prol][inter]
};

__device__ __forceinline__ bool pers_bench_barrier(const PersArgs* A, const PersLaunch& L, u32& nbar)
{
    const u32 t = L.bar_base + (nbar + 1u) * gridDim.x;
    nbar += 1u;
    return pers_grid_wait(A->bar, t, A->abort_flag);
}

// The route-P prologue of the chained bench arm: candle's RMSNorm of the
// residual (kind 1), then the rotation; or the rotation alone (kind 2). In
// place in `s`; the staging that follows overwrites it, so the rows still
// read the bench `x` and still owe identity. CTA 0 copies its vectors out
// when the debug pointers are set (identity launch only).
__device__ void pers_bench_prologue(const PersArgs* A, const PersLaunch& L, u32 kind, u32 slot, u32 norm_slot, float* s)
{
    const bool norm = kind == 1u;
    const u32 n = norm ? A->hidden : A->inter;
    const u32 m = norm ? A->m_h : A->m_i;
    const u32 k = norm ? A->k_h : A->k_i;
    const u32* signs = norm ? A->signs_h : A->signs_i;
    const float* small = norm ? A->small_h : A->small_i;
    const float inv = norm ? A->inv_h : A->inv_i;
    __syncthreads();  // every warp has left the previous site: s is free
    if (norm) {
        const float scale = exact_rms_scale<0>(A->resid, n, A->eps);
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned short h = exact_rms_apply(scale, __ldcg(A->resid + i), A->alpha[i]);
            if (L.dbg_norm && blockIdx.x == 0u) L.dbg_norm[norm_slot * n + i] = h;
            const float v = h2f(h);
            s[i] = ((signs[i >> 5] >> (i & 31u)) & 1u) ? -v : v;
        }
    } else {
        pers_rot_load(A->resid, signs, s, n);
    }
    pers_rot_finish(s, small, n, m, k, inv);
    if (L.dbg_rot && blockIdx.x == 0u)
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) L.dbg_rot[slot * A->inter + i] = s[i];
}

template <int MODE, u32 THREADS, u32 STRIDE, bool TRIO, bool PREFETCH, int WAITS, bool PROLOGUE>
__device__ __forceinline__ void pers_body(const PersArgs* __restrict__ A, const PersLaunch L)
{
    unsigned char* base = PERS_SMEM_BASE();
    const u32 lane = threadIdx.x & 31u;

    // Tables: 18,816 B, a multiple of 16, after the 256-B header, so the
    // staging starts 16-B aligned. MODE 2 reads them in place through L1.
    F1rTables tab;
    const float* invnorm;
    float* xs;
    if (MODE == 3 || MODE == 5) {
        u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
        unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
        unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
        unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
        float* t_inv = reinterpret_cast<float*>(t_suf + 128);
        for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
        for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
            t_pref[i] = A->pref[i];
            t_suf[i] = A->suf[i];
        }
        for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
        for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
        __syncthreads();
        tab = F1rTables{t_rows, t_pref, t_br, t_suf};
        invnorm = t_inv;
        xs = t_inv + 32;
    } else {
        tab = F1rTables{A->rows, A->pref, A->br, A->suf};
        invnorm = A->invnorm;
        xs = reinterpret_cast<float*>(base + PERS_SMEM_HEADER);
    }

    u32 nbar = 0u;  // grid barriers passed in this launch, pers.rs::n_bar
    u32 nprol = 0u, nnorm = 0u;
    u32 staged_din = 0xffffffffu;
    u32 staged_layer = 0xffffffffu;
    const u32 cnt_target = (L.epoch + 1u) * gridDim.x;
    u32* cnts = A->cnts + L.cnt_offset;
    for (u32 s = 0; s < A->nsites; ++s) {
        const PersSite st = A->sites[s];
        if (WAITS >= 1) {
            const u32 kind = st.wait & 3u, producer = st.wait >> 2;
            if (kind == 1u) {
                if (!pers_bench_barrier(A, L, nbar)) return;
                if (WAITS == 2 && !pers_bench_barrier(A, L, nbar)) return;
            } else if (kind == 2u) {
                if (!pers_wait_count(cnts + producer, cnt_target, A->abort_flag)) return;
                if (WAITS == 2 && !pers_bench_barrier(A, L, nbar)) return;
            }
        }
        const u32 nblocks = st.nb_tw & 0xffffu, tail_w = st.nb_tw >> 16;
        const u32 d_in = nblocks * LLVQ_DIM + tail_w;
        // q/k/v and gate/up share one staging, as in the eager path; a new
        // layer always restages (its prologue reads a new residual).
        const u32 layer = st.flags >> 8;
        if (d_in != staged_din || layer != staged_layer) {
            const u32 pk = (st.flags >> 1) & 3u;
            if (PROLOGUE && pk != 0u) {
                pers_bench_prologue(A, L, pk, nprol, nnorm, xs);
                nprol += 1u;
                nnorm += pk == 1u ? 1u : 0u;
            }
            pers_stage_f32<STRIDE, TRIO>(A->x, xs, nblocks, tail_w);
            staged_din = d_in;
            staged_layer = layer;
        }
        const u32* words = reinterpret_cast<const u32*>(((unsigned long long)st.words_hi << 32) | st.words_lo);
        const float* gs = A->gscale[s];
        const float* rs = A->rscale[s];
        const unsigned short* tl = A->tail[s];
        pers_claim_reset(A->slice_lo[s * gridDim.x + blockIdx.x]);
        const u32 hi = A->slice_hi[s * gridDim.x + blockIdx.x];
        for (;;) {
            const u32 r = pers_claim(lane, 1u);
            if (r >= hi) break;
            const u32 yrow = st.row0_in_y + r;
            if (MODE == 0) {
                float a = 0.0f;
                for (u32 j = lane; j < nblocks; j += 32u) a = __fmaf_rn(1.0f, xs[j * STRIDE], a);
                a = warp_sum(a);
                if (lane == 0u) A->y_all[yrow] = f2h(a);
            } else if (MODE == 1) {
                const u32* wrow = words + r * st.stride_u32;
                u32 a = 0u;
                for (u32 j = lane; j < nblocks; j += 32u) {
                    u32 lo, hi16;
                    f1r_load(wrow, j, lo, hi16);
                    a ^= lo ^ (hi16 << 7);
                }
#pragma unroll
                for (int m = 16; m > 0; m >>= 1) a ^= __shfl_xor_sync(0xffffffffu, a, m);
                if (lane == 0u) A->y_all[yrow] = (unsigned short)((a ^ (a >> 16)) & 0x7fffu);
            } else if (MODE >= 4) {
                // tetra48_dot's decode with its dot replaced by an XOR of the
                // values' bits: same loads, same table chain, same float
                // construction, same shell and scales.
                const u32* wrow = words + r * st.stride_u32;
                u32 a = 0u;
                for (u32 j = lane; j < nblocks; j += 32u) {
                    u32 lo, hi16;
                    f1r_load(wrow, j, lo, hi16);
                    u32 q[6];
                    f1r_v3_quads(lo, hi16, tab, q);
#pragma unroll
                    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
                        for (u32 k = 0; k < 4u; ++k) a ^= __float_as_uint(f1r_v3_float(q[i], k));
                    }
                    const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
                    a ^= __float_as_uint(gs[(hi16 >> 15) & 1u] * invnorm[m]);
                }
#pragma unroll
                for (int m = 16; m > 0; m >>= 1) a ^= __shfl_xor_sync(0xffffffffu, a, m);
                if (lane == 0u) A->y_all[yrow] = (unsigned short)((a ^ (a >> 16)) & 0x7fffu);
            } else {
                const float acc = pers_tetra_row<STRIDE, TRIO, PREFETCH>(words + r * st.stride_u32, tab, gs, invnorm,
                                                                         xs, nblocks, lane);
                if (lane == 0u) {
                    const float v = pers_epilogue(acc, rs[r], tl + r * tail_w, xs + nblocks * STRIDE, tail_w);
                    if (L.dbg) L.dbg[yrow] = v;
                    A->y_all[yrow] = f2h(v);
                }
            }
        }
        __syncthreads();  // the CTA's rows of site s precede its signal and the next reset
        if (WAITS >= 1 && (st.flags & 1u)) pers_signal(cnts + s);
    }
    if (WAITS >= 1) {  // the final norm's W1 and the argmax combine (W4) of the served schedule
        for (u32 t = 0; t < 2u; ++t) {
            if (!pers_bench_barrier(A, L, nbar)) return;
            if (WAITS == 2 && !pers_bench_barrier(A, L, nbar)) return;
        }
    }
}

#define PERS_ARM(sym, threads, MODE_, STRIDE_, TRIO_, PF_, WAITS_, PROL_)                         \
    extern "C" __global__ void __launch_bounds__(threads, 1) sym(const PersArgs* __restrict__ A,  \
                                                                 const PersLaunch L)               \
    {                                                                                              \
        pers_body<MODE_, threads, STRIDE_, TRIO_, PF_, WAITS_, PROL_>(A, L);                      \
    }

// One line an arm; pers.rs::tests::the_table_is_the_kernel_text reads them.
PERS_ARM(tv_pers_null, 768u, 0, 28u, true, false, 0, false)
PERS_ARM(tv_pers_word, 768u, 1, 28u, true, false, 0, false)
PERS_ARM(tv_pers_dec_l1, 768u, 4, 28u, true, false, 0, false)
PERS_ARM(tv_pers_dec_smem, 768u, 5, 28u, true, false, 0, false)
PERS_ARM(tv_pers_l1, 768u, 2, 24u, false, false, 0, false)
PERS_ARM(tv_pers_l1_t28, 768u, 2, 28u, true, false, 0, false)
PERS_ARM(tv_pers_smem, 768u, 3, 24u, false, false, 0, false)
PERS_ARM(tv_pers_smem_s25, 768u, 3, 25u, false, false, 0, false)
PERS_ARM(tv_pers_smem_t28, 768u, 3, 28u, true, false, 0, false)
PERS_ARM(tv_pers_smem_t28_u2, 768u, 3, 28u, true, true, 0, false)
PERS_ARM(tv_pers_t28_512, 512u, 3, 28u, true, false, 0, false)
PERS_ARM(tv_pers_t28_1024, 1024u, 3, 28u, true, false, 0, false)
PERS_ARM(tv_pers_chain_w, 768u, 3, 28u, true, false, 1, false)
PERS_ARM(tv_pers_chain_w2, 768u, 3, 28u, true, false, 2, false)
PERS_ARM(tv_pers_chain_wp, 768u, 3, 28u, true, false, 1, true)

// ---------------------------------------------------------------------------
// Microbenchmarks (step L1). Launched cooperatively only: co-residency is the
// driver's promise, never ours. Timed by events around one launch of n hops.
// Each needs PERS_SMEM_HEADER bytes of dynamic shared memory.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_pers_bar(u32* bar, u32 bar_base, u32 n, u32* abort_flag)
{
    for (u32 i = 0; i < n; ++i)
        if (!pers_grid_wait(bar, bar_base + (i + 1u) * gridDim.x, abort_flag)) return;
}

// A ring of R CTAs: CTA c waits for its turn on one counter, then passes it
// on. One hop is one dependent flag latency, CTA to CTA.
extern "C" __global__ void tv_pers_ring(u32* flag, u32 base, u32 n, u32* abort_flag)
{
    const u32 c = blockIdx.x, R = gridDim.x;
    for (u32 h = 0; h < n; ++h) {
        if (!pers_wait_count(flag, base + h * R + c, abort_flag)) return;
        pers_signal(flag);
    }
}

extern "C" __global__ void tv_pers_empty(u32 unused)
{
    (void)unused;
}
