// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, sections 3 and 5, step L2),
// revised the same day after an adversarial review. Nothing here has met
// NVRTC, a card or a host test; every claim of bit identity below is an
// argument, checked on the card only by the row-identity pass the bench runs
// before timing (prereg-persistent-1.md, gate 7).
//
// What the review changed in this file:
//   * a work-claiming race is closed: every site now ends with __syncthreads()
//     in every mode, before thread 0 resets the claim cursor (finding 4);
//   * every arm is its own extern "C" symbol over ONE template body, with no
//     macro that changes with the selection (finding 13);
//   * pchain_w is exactly psmem_t28's body plus the waits; signals are sent
//     only after the 48 sites someone waits on; the two trailing barriers of
//     the served schedule (final norm W1, argmax W4) exist (findings 6, 13);
//   * a doubled-wait arm (pchain_w2) gives the marginal cost of a barrier
//     with no arrival skew (finding 6);
//   * stride 25 (scalar loads, no conflict) and trio order at stride 28 with
//     EXPLICIT float4 loads replace the use-order vectorization guess
//     (finding 10); the F2-trellis arms are dropped (no code, no decision rides
//     on them, f1-rang-variantes-2026-09-05 measured f1r_v2 at 5.625 ms
//     against 3.875 for v3);
//   * a register prefetch arm, the "unroll x2" of
//     audit-perf-noyau-cuda-2026-08-05.md:116-122, never measured (finding 15);
//   * the f32 value before f2h goes to a nullable debug buffer, so a moved
//     association is judged against f64 at 1e-5 (finding 3).
//
// Arms and symbols (pers.rs::PERS_ARMS mirrors this table; registration order
// is dispatch order):
//
//   arm            symbol                 MODE THREADS STRIDE TRIO PREFETCH WAITS PROLOGUE owes identity
//   pnull          tv_pers_null            0    768     28    yes    no      0      no      no
//   pword          tv_pers_word            1    768     28    yes    no      0      no      no
//   pl1            tv_pers_l1              2    768     24    no     no      0      no      yes
//   psmem          tv_pers_smem            3    768     24    no     no      0      no      yes
//   psmem_s25      tv_pers_smem_s25        3    768     25    no     no      0      no      yes
//   psmem_t28      tv_pers_smem_t28        3    768     28    yes    no      0      no      yes
//   psmem_t28_u2   tv_pers_smem_t28_u2     3    768     28    yes    yes     0      no      yes
//   pt28_512       tv_pers_t28_512         3    512     28    yes    no      0      no      yes
//   pt28_1024      tv_pers_t28_1024        3   1024     28    yes    no      0      no      yes
//   pchain_w       tv_pers_chain_w         3    768     28    yes    no      1      no      yes
//   pchain_w2      tv_pers_chain_w2        3    768     28    yes    no      2      no      yes
//   pchain_wp      tv_pers_chain_wp        3    768     28    yes    no      1      yes     yes
//
// MODE: 0 staging and reduction only, 1 every word read and XOR-folded,
// 2 full decode with the tables read in place (L1), 3 full decode with the
// tables copied to shared memory once per launch.
// WAITS: 0 none; 1 the served 4B schedule laid on the bare file (95 barrier
// ordinals in the site list, 48 counter waits whose producers are signalled by
// every CTA, 2 trailing barriers: 145 grid-wide syncs, pers.rs::N_GRID_SYNCS);
// 2 the same with each sync followed by one more grid barrier (290).
//
// Composition contract (NVRTC has no file system, the host concatenates):
//   llvq_slot.cuh, matvec.cu (warp_sum, h2f, f2h, tail_dot_h), llvq_f1rank.cuh,
//   llvq_f1rank_v3.cuh, llvq_tetra48.cuh (tetra48_dot, TETRA48_ORDER,
//   tetra48_n2), llvq_rot.cuh (LLVQ_ROT_KMAX), exact_ops.cuh, then this file.
//   No CUDA header: the runtime image has none (llvq_slot.cuh:18-20) and
//   NVRTC 12.4 bundles none. float4 is an NVRTC built-in (awq_gemv.cu uses it).
//   Its own NVRTC module: the bench unit and the served unit keep their bytes.

#ifndef PERS_SPIN_LIMIT
#define PERS_SPIN_LIMIT (1u << 17)  // about 67 ms at the 512-ns cap of the back-off, then abort
#endif

// ---------------------------------------------------------------------------
// 0. Host-built descriptors (pers.rs::site_words, pers.rs::PersArgs layout).
// ---------------------------------------------------------------------------
struct PersSite {             // 8 u32 a site, uploaded once
    u32 words_lo, words_hi;   // device address of the site's row-strided word stream
    u32 d_out;
    u32 nb_tw;                // nblocks | tail_w << 16
    u32 row0_in_y;            // the site's first row in y_all
    u32 stride_u32;           // tetra48::stride_u32(nblocks), asserted host-side
    u32 wait;                 // bits 0-1: 0 none, 1 grid barrier, 2 counter; producer site << 2
    u32 flags;                // bit 0: signal after the site; bits 1-2: prologue 0 none, 1 norm + rot 2560, 2 rot 9728
};

struct PersArgs {             // one device struct; its address is every arm's only argument
    const PersSite* sites;
    u32 nsites, prologue_slots;           // prologue_slots: rows of dbg_rot (96 on the 4B pattern)
    const u32* rows;                      // decoder tables, global copies (llvq_f1rank.cuh)
    const unsigned char* pref;
    const unsigned short* br;
    const unsigned char* suf;
    const float* invnorm;                 // 32 floats, shared by every site
    const float* const* gscale;           // per site, 2 floats
    const float* const* rscale;           // per site, d_out floats
    const unsigned short* const* tail;    // per site, d_out * tail_w binary16
    const float* x;                       // the bench activation, f32, width >= 9728
    unsigned short* y_all;                // every site's rows; poisoned to 0xFFFF before an identity launch
    float* dbg;                           // nullable: v = acc*rscale + tv before f2h, identity launch only
    const u32* slice_lo;                  // [nsites * grid], pers.rs::slices
    const u32* slice_hi;
    u32* bar;                             // monotonic grid-barrier counter, life of the runtime
    u32* cnts;                            // one monotonic counter per site
    u32* abort_flag;
    u32 bar_base;                         // arrivals before this launch; the host adds n_bar * grid after it
    u32 launch_idx;                       // launches of this arm before this one: counter targets
    const unsigned short* resid;          // prologue inputs: a fixed binary16 residual, width 9728
    const unsigned short* alpha;          // RMSNorm weights, binary16, width 2560
    const u32* signs2560;                 // sign bitmaps, rot_load's convention
    const u32* signs9728;
    const float* small2560;               // Q_odd, k x KMAX, zero-padded with +0.0f (fused.rs:1259)
    const float* small9728;
    float inv512;                         // 1/sqrt(512) as the host passes it to rot_apply
    float eps;
    unsigned short* dbg_norm;             // nullable: CTA 0's normed vectors, [72][2560]
    float* dbg_rot;                       // nullable: CTA 0's rotated vectors, [96][9728]
};

// ---------------------------------------------------------------------------
// 1. Synchronization. One monotonic barrier counter; targets are host bases
//    plus (ordinal + 1) * gridDim.x, compared wrap-safe. Cooperative groups'
//    grid-sync pattern: bar.sync, one thread fences and arrives, spins with
//    back-off, fences, bar.sync. Every CTA calls each ordinal exactly once,
//    with rows or without (the miscount pitfall plan A cites).
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 pers_ld_acquire(const u32* p)
{
    u32 v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ bool pers_reached(u32 v, u32 target)
{
    return (int)(v - target) >= 0;  // valid while fewer than 2^31 arrivals separate them
}

// Returns false when the abort flag is up (this CTA's watchdog or another's).
__device__ bool pers_grid_wait_raw(u32* bar, u32 target, u32* abort_flag)
{
    __shared__ int s_ok;
    __syncthreads();                         // the CTA's stores precede thread 0's arrival
    if (threadIdx.x == 0) {
        __threadfence();                     // cumulative: orders the CTA's stores before the arrival
        atomicAdd(bar, 1u);
        u32 ns = 32u, spins = 0u;
        int ok = 1;
        while (!pers_reached(pers_ld_acquire(bar), target)) {
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
            if ((++spins & 255u) == 0u && pers_ld_acquire(abort_flag) != 0u) { ok = 0; break; }
            if (spins >= PERS_SPIN_LIMIT) { atomicExch(abort_flag, 1u); ok = 0; break; }
        }
        __threadfence();
        s_ok = ok;
    }
    __syncthreads();
    return s_ok != 0;
}

__device__ __forceinline__ bool pers_grid_wait(const PersArgs* __restrict__ A, u32 ordinal)
{
    return pers_grid_wait_raw(A->bar, A->bar_base + (ordinal + 1u) * gridDim.x, A->abort_flag);
}

// Producer side of a counter. Called right after the end-of-site
// __syncthreads(), which already ordered the CTA's stores before thread 0.
__device__ __forceinline__ void pers_signal_after_barrier(u32* cnt)
{
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(cnt, 1u); }
}

// Consumer side. In the bench every CTA signals every waited site, so the
// target is (launch_idx + 1) * grid and the wait behaves as a grid barrier:
// that is why pers.rs counts it among the 145 grid-wide syncs.
__device__ bool pers_wait_count(const u32* cnt, u32 target, u32* abort_flag)
{
    __shared__ int s_ok2;
    if (threadIdx.x == 0) {
        u32 ns = 32u, spins = 0u;
        int ok = 1;
        while (!pers_reached(pers_ld_acquire(cnt), target)) {
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
            if ((++spins & 255u) == 0u && pers_ld_acquire(abort_flag) != 0u) { ok = 0; break; }
            if (spins >= PERS_SPIN_LIMIT) { atomicExch(abort_flag, 1u); ok = 0; break; }
        }
        __threadfence();
        s_ok2 = ok;
    }
    __syncthreads();
    return s_ok2 != 0;
}

// ---------------------------------------------------------------------------
// 2. Work claiming inside a CTA. The CTA's row slice is static (host table);
//    warps take the next row from a shared cursor. A row's bits do not depend
//    on which warp computes it: the lane-to-block map and warp_sum are fixed.
//    The cursor is reset by thread 0 only after the end-of-site barrier, so no
//    warp still claiming site s can draw an index of site s + 1.
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 pers_claim(u32* s_next, u32 lane)
{
    u32 r = 0u;
    if (lane == 0u) r = atomicAdd(s_next, 1u);
    return __shfl_sync(0xffffffffu, r, 0);
}

// ---------------------------------------------------------------------------
// 3. Staging. Natural coordinate c of block b lands at b*STRIDE + c, or at
//    b*STRIDE + PERS_INV_ORDER[c] under TRIO, so the trio dot reads slots
//    0..23 in sequence as six aligned float4. Tail columns follow the last
//    block contiguously. Same values as the reference staging, read with
//    __ldcg because route P reads activations written in the same launch.
// ---------------------------------------------------------------------------
__device__ static constexpr unsigned char PERS_INV_ORDER[24] = {
    0, 1, 2, 3, 4, 16, 8, 5, 17, 18, 6, 9, 7, 10, 11, 19, 12, 13, 14, 15, 20, 21, 22, 23,
};  // the inverse of TETRA48_ORDER (llvq_tetra48.cuh:70-73); pers.rs::tests checks it

template <u32 STRIDE, bool TRIO>
__device__ __forceinline__ void pers_stage_x(const float* __restrict__ x, float* xs, u32 nblocks, u32 tail_w)
{
    __syncthreads();  // no warp still reads the previous staging, nor the prologue's vector
    const u32 nmain = nblocks * LLVQ_DIM;
    for (u32 i = threadIdx.x; i < nmain + tail_w; i += blockDim.x) {
        u32 dst;
        if (i < nmain) {
            u32 b = i / LLVQ_DIM, c = i - b * LLVQ_DIM;
            dst = b * STRIDE + (TRIO ? (u32)PERS_INV_ORDER[c] : c);
        } else {
            dst = nblocks * STRIDE + (i - nmain);
        }
        xs[dst] = __ldcg(x + i);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 4. The block dot in trio order: tetra48_dot (llvq_tetra48.cuh:111-133) with
//    xb[TETRA48_ORDER[4i + j]] read as component j of the i-th float4 of the
//    trio-staged block. Same 24 __fmaf_rn in the same order on the same
//    values, same (acc*g)*inv association: bit-identical by construction.
//    Explicit float4 makes the six LDS.128 a property of the source rather
//    than of the load-store vectorizer, provided the base is 16-B aligned:
//    the extern shared array is declared float4 and every offset is a
//    multiple of 112 bytes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float tetra48_dot_trio4(u32 lo, u32 hi16, const F1rTables& t,
                                                   const float* __restrict__ xb_trio,
                                                   const float* __restrict__ gscale,
                                                   const float* __restrict__ invnorm)
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    const float4* x4 = reinterpret_cast<const float4*>(xb_trio);
    float acc = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
        const float4 v = x4[i];
        acc = __fmaf_rn(f1r_v3_float(q[i], 0u), v.x, acc);
        acc = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, acc);
        acc = __fmaf_rn(f1r_v3_float(q[i], 2u), v.z, acc);
        acc = __fmaf_rn(f1r_v3_float(q[i], 3u), v.w, acc);
    }
    u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    u32 g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];  // (acc*g)*inv, never acc*(g*inv)
}

template <bool TRIO>
__device__ __forceinline__ float pers_dot(u32 lo, u32 hi16, const F1rTables& t, const float* xb,
                                          const float* __restrict__ gscale, const float* __restrict__ invnorm)
{
    if (TRIO) return tetra48_dot_trio4(lo, hi16, t, xb, gscale, invnorm);
    return tetra48_dot(lo, hi16, t, xb, gscale, invnorm);
}

// One Tetra row's lane partials, reduced. Lane l adds blocks j = l, l+32, ...
// in increasing j, as tv_tetra48_h does at any tile that is a multiple of 32
// (tv_tetra48_h.cu:68-82). `acc += dot` keeps the served expression shape: it
// is one of the three contractible sites (persistent.md 3.7), and L1a reads
// what NVVM made of it before anyone relies on the shape.
template <u32 STRIDE, bool TRIO, bool PREFETCH>
__device__ __forceinline__ float pers_row_acc(const u32* __restrict__ wrow, const F1rTables& tab,
                                              const float* __restrict__ gscale,
                                              const float* __restrict__ invnorm, const float* xs,
                                              u32 nblocks, u32 lane)
{
    float acc = 0.0f;
    if (!PREFETCH) {
        for (u32 j = lane; j < nblocks; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += pers_dot<TRIO>(lo, hi16, tab, xs + j * STRIDE, gscale, invnorm);
        }
    } else {
        // Block j + 32's words are in flight while block j decodes: twice the
        // bytes in flight per lane at the same accumulation order.
        u32 lo = 0u, hi16 = 0u;
        if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
        for (u32 j = lane; j < nblocks; j += 32u) {
            u32 lo2 = 0u, hi2 = 0u;
            if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
            acc += pers_dot<TRIO>(lo, hi16, tab, xs + j * STRIDE, gscale, invnorm);
            lo = lo2;
            hi16 = hi2;
        }
    }
    return warp_sum(acc);
}

// ---------------------------------------------------------------------------
// 5. The route-P prologue, one CTA: candle's RMSNorm (exact_ops.cuh), then
//    rot_apply's three phases (llvq_rot.cuh:114-233, rotate.cu:38-67) with the
//    WHT in registers and the mix without padding. Works in place in `s`
//    (the staging buffer, free at this point, 16-B aligned); the staging that
//    follows overwrites it, so the rows still read the bench x and still owe
//    identity. CTA 0 copies its normed and rotated vectors out when the debug
//    pointers are set (identity launch only); the bench compares them with the
//    L0 host replay of candle's tree and with rot_apply on the card.
// ---------------------------------------------------------------------------
// Lane l holds v[0..15] = s[16l .. 16l+15] of one 512-group. Stages len = 1, 2,
// 4, 8 inside the lane, then len = 16 .. 256 as shfl_xor 1 .. 16, increasing,
// as rot_wht_step walks them. Lower element a + b, upper a - b with a the
// lower value: rot_wht_step's two operations, so the same bits.
__device__ __forceinline__ void pers_wht512(float v[16], u32 lane)
{
#pragma unroll
    for (u32 len = 1u; len < 16u; len <<= 1) {
#pragma unroll
        for (u32 i = 0u; i < 16u; ++i) {
            if ((i & len) == 0u) {
                float a = v[i], b = v[i + len];
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
    }
#pragma unroll
    for (u32 sh = 1u; sh < 32u; sh <<= 1) {
        bool upper = (lane & sh) != 0u;
#pragma unroll
        for (u32 i = 0u; i < 16u; ++i) {
            float p = __shfl_xor_sync(0xffffffffu, v[i], sh);
            v[i] = upper ? (p - v[i]) : (v[i] + p);  // upper: a - b with a = the partner's (lower) value
        }
    }
}

// rot_mix for one column j, in place. The KMAX - k padded terms of rot_mix are
// fma(+0, +0, acc), equal to one acc + (+0) under round to nearest (the only
// effect is -0 -> +0, after which acc + (+0) is the identity), and `small` is
// padded with +0.0f (fused.rs:1259). Predicated full unrolling keeps col in
// registers; k > 1 at both widths here (5 and 19 groups of 512).
__device__ __forceinline__ void pers_mix_column(float* s, const float* __restrict__ small, u32 m, u32 k,
                                                float inv, u32 j)
{
    float col[LLVQ_ROT_KMAX];
#pragma unroll
    for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t) col[t] = t < k ? s[t * m + j] * inv : 0.0f;
    for (u32 g = 0; g < k; ++g) {
        float acc = 0.0f;
#pragma unroll
        for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t)
            if (t < k) acc = __fmaf_rn(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
        if (k < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);
        s[g * m + j] = acc;
    }
}

// kind 1: norm + rotation at n = 2,560 (k = 5); kind 2: rotation at 9,728 (k = 19).
// `slot` numbers the prologue within the launch, for the debug copies.
template <u32 THREADS>
__device__ void pers_prologue(const PersArgs* __restrict__ A, u32 kind, u32 slot, u32 norm_slot,
                              float* s, float* s_sum)
{
    const u32 n = kind == 1u ? 2560u : 9728u;
    const u32 k = n / 512u;
    const u32* signs = kind == 1u ? A->signs2560 : A->signs9728;
    const float* small = kind == 1u ? A->small2560 : A->small9728;
    __syncthreads();  // every warp has left the previous site: s is free
    float scale = 0.0f;
    if (kind == 1u) scale = exact_rms_scale_1024<0>(A->resid, n, A->eps, s_sum);
    for (u32 i = threadIdx.x; i < n; i += THREADS) {
        unsigned short h = A->resid[i];
        if (kind == 1u) {
            h = exact_rms_apply(scale, h, A->alpha[i]);
            if (A->dbg_norm && blockIdx.x == 0u) A->dbg_norm[norm_slot * 2560u + i] = h;
        }
        float v = h2f(h);
        s[i] = ((signs[i >> 5] >> (i & 31u)) & 1u) ? -v : v;  // rot_load
    }
    __syncthreads();
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (u32 g = warp; g < k; g += THREADS / 32u) {
        float v[16];
#pragma unroll
        for (u32 i = 0; i < 16u; ++i) v[i] = s[g * 512u + 16u * lane + i];
        pers_wht512(v, lane);
#pragma unroll
        for (u32 i = 0; i < 16u; ++i) s[g * 512u + 16u * lane + i] = v[i];
    }
    __syncthreads();
    for (u32 j = threadIdx.x; j < 512u; j += THREADS) pers_mix_column(s, small, 512u, k, A->inv512, j);
    __syncthreads();
    if (A->dbg_rot && blockIdx.x == 0u)
        for (u32 i = threadIdx.x; i < n; i += THREADS) A->dbg_rot[slot * 9728u + i] = s[i];
}

// ---------------------------------------------------------------------------
// 6. The persistent bench body: one launch walks the 252 sites in served
//    order. One template; each arm is one instantiation behind its own symbol.
// ---------------------------------------------------------------------------
template <int MODE, u32 THREADS, u32 STRIDE, bool TRIO, bool PREFETCH, int WAITS, bool PROLOGUE>
__device__ __forceinline__ void pers_body(const PersArgs* __restrict__ A)
{
    extern __shared__ float4 pers_smem4[];  // 16-B aligned base
    float* smem = reinterpret_cast<float*>(pers_smem4);
    __shared__ u32 s_next;
    __shared__ float s_sum[32];
    const u32 lane = threadIdx.x & 31u;

    // Tables: 16,384 + 128 + 2,048 + 128 B, then invnorm 128 B: 18,816 B, the
    // sizes banc-252-brut/sweep-64.txt prints, a multiple of 16, so xs starts
    // 16-B aligned. MODE 2 reads them in place through L1, as today.
    F1rTables tab;
    const float* invnorm;
    float* xs;
    if (MODE == 3) {
        u32* t_rows = reinterpret_cast<u32*>(smem);
        unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
        unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
        unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
        float* t_inv = reinterpret_cast<float*>(t_suf + 128);
        for (u32 i = threadIdx.x; i < 4096u; i += THREADS) t_rows[i] = A->rows[i];
        for (u32 i = threadIdx.x; i < 128u; i += THREADS) { t_pref[i] = A->pref[i]; t_suf[i] = A->suf[i]; }
        for (u32 i = threadIdx.x; i < 1024u; i += THREADS) t_br[i] = A->br[i];
        for (u32 i = threadIdx.x; i < 32u; i += THREADS) t_inv[i] = A->invnorm[i];
        __syncthreads();
        tab = F1rTables{ t_rows, t_pref, t_br, t_suf };
        invnorm = t_inv;
        xs = t_inv + 32;
    } else {
        tab = F1rTables{ A->rows, A->pref, A->br, A->suf };
        invnorm = A->invnorm;
        xs = smem;
    }

    u32 nbar = 0u;           // grid barriers passed in this launch, pers.rs::n_bar
    u32 nprol = 0u, nnorm = 0u;
    u32 staged_din = 0xffffffffu;
    for (u32 s = 0; s < A->nsites; ++s) {
        const PersSite st = A->sites[s];
        if (WAITS >= 1) {
            const u32 kind = st.wait & 3u, producer = st.wait >> 2;
            if (kind == 1u) {
                if (!pers_grid_wait(A, nbar++)) return;
                if (WAITS == 2 && !pers_grid_wait(A, nbar++)) return;
            } else if (kind == 2u) {
                if (!pers_wait_count(A->cnts + producer, (A->launch_idx + 1u) * gridDim.x, A->abort_flag)) return;
                if (WAITS == 2 && !pers_grid_wait(A, nbar++)) return;
            }
        }
        const u32 nblocks = st.nb_tw & 0xffffu, tail_w = st.nb_tw >> 16;
        const u32 d_in = nblocks * LLVQ_DIM + tail_w;
        if (d_in != staged_din) {  // q/k/v and gate/up share one staging, as in the eager path
            const u32 pk = (st.flags >> 1) & 3u;
            if (PROLOGUE && pk != 0u) {
                pers_prologue<THREADS>(A, pk, nprol, nnorm, xs, s_sum);
                nprol += 1u;
                nnorm += pk == 1u ? 1u : 0u;
            }
            pers_stage_x<STRIDE, TRIO>(A->x, xs, nblocks, tail_w);
            staged_din = d_in;
        }
        const u32* words = reinterpret_cast<const u32*>(((unsigned long long)st.words_hi << 32) | st.words_lo);
        const float* gs = A->gscale[s];
        const float* rs = A->rscale[s];
        const unsigned short* tl = A->tail[s];
        if (threadIdx.x == 0u) s_next = A->slice_lo[s * gridDim.x + blockIdx.x];
        __syncthreads();
        const u32 hi = A->slice_hi[s * gridDim.x + blockIdx.x];
        for (;;) {
            const u32 r = pers_claim(&s_next, lane);
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
                if (lane == 0u) A->y_all[yrow] = (unsigned short)(a ^ (a >> 16));
            } else {
                const float acc = pers_row_acc<STRIDE, TRIO, PREFETCH>(words + r * st.stride_u32, tab, gs,
                                                                       invnorm, xs, nblocks, lane);
                if (lane == 0u) {
                    // tv_tetra48_h.cu:86-87: tail_dot_h, then f2h(acc*rscale + tv). The
                    // tail activation is read from the staged copy: same values.
                    const float tv = tail_dot_h(tl, xs + nblocks * STRIDE, r, tail_w);
                    const float v = acc * rs[r] + tv;
                    if (A->dbg) A->dbg[yrow] = v;
                    A->y_all[yrow] = f2h(v);
                }
            }
        }
        // REVIEW FIX (race): every warp has left site s before thread 0 resets
        // s_next for site s + 1, in every mode, so the waitless arms do the
        // same rows as the chained ones and as the reference.
        __syncthreads();
        if (WAITS >= 1 && (st.flags & 1u)) pers_signal_after_barrier(A->cnts + s);
    }
    if (WAITS >= 1) {  // the final norm's W1 and the argmax combine (W4) of the served schedule
        for (u32 t = 0; t < 2u; ++t) {
            if (!pers_grid_wait(A, nbar++)) return;
            if (WAITS == 2 && !pers_grid_wait(A, nbar++)) return;
        }
    }
}

#define PERS_ARM(sym, threads, MODE_, STRIDE_, TRIO_, PF_, WAITS_, PROL_)                         \
    extern "C" __global__ void __launch_bounds__(threads, 1) sym(const PersArgs* __restrict__ A)  \
    {                                                                                              \
        pers_body<MODE_, threads, STRIDE_, TRIO_, PF_, WAITS_, PROL_>(A);                         \
    }

PERS_ARM(tv_pers_null,        768u,  0, 28u, true,  false, 0, false)
PERS_ARM(tv_pers_word,        768u,  1, 28u, true,  false, 0, false)
PERS_ARM(tv_pers_l1,          768u,  2, 24u, false, false, 0, false)
PERS_ARM(tv_pers_smem,        768u,  3, 24u, false, false, 0, false)
PERS_ARM(tv_pers_smem_s25,    768u,  3, 25u, false, false, 0, false)
PERS_ARM(tv_pers_smem_t28,    768u,  3, 28u, true,  false, 0, false)
PERS_ARM(tv_pers_smem_t28_u2, 768u,  3, 28u, true,  true,  0, false)
PERS_ARM(tv_pers_t28_512,     512u,  3, 28u, true,  false, 0, false)
PERS_ARM(tv_pers_t28_1024,   1024u,  3, 28u, true,  false, 0, false)
PERS_ARM(tv_pers_chain_w,     768u,  3, 28u, true,  false, 1, false)
PERS_ARM(tv_pers_chain_w2,    768u,  3, 28u, true,  false, 2, false)
PERS_ARM(tv_pers_chain_wp,    768u,  3, 28u, true,  false, 1, true)

// Shared bytes a launch, host side (pers.rs::shared_bytes): tables 18,816 B
// for MODE 3, plus the widest staging, nblocks * STRIDE + tail_w floats at
// d_in = 9,728: 38,912 B (24), 40,532 B (25), 45,392 B (28). The largest arm
// takes 64,208 B plus 132 B of statics: above 48 KiB, so the host sets
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (cudarc set_attribute,
// core.rs:2442), under the 101,376 B opt-in (format-noyau.md section 8).

// ---------------------------------------------------------------------------
// 7. Microbenchmarks (prereg-persistent-0.md, the first paid step). Launched
//    cooperatively only: co-residency is the driver's promise, never ours.
//    Timed by events around one launch of n hops.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_pers_bar(u32* bar, u32 bar_base, u32 n, u32* abort_flag)
{
    for (u32 i = 0; i < n; ++i)
        if (!pers_grid_wait_raw(bar, bar_base + (i + 1u) * gridDim.x, abort_flag)) return;
}

// A ring of R CTAs (R = 16): CTA c waits for its turn on one counter, then
// passes it on. One hop is one dependent flag latency, CTA to CTA.
extern "C" __global__ void tv_pers_ring(u32* flag, u32 base, u32 n, u32* abort_flag)
{
    const u32 c = blockIdx.x, R = gridDim.x;
    for (u32 h = 0; h < n; ++h) {
        if (!pers_wait_count(flag, base + h * R + c, abort_flag)) return;
        if (threadIdx.x == 0) { __threadfence(); atomicAdd(flag, 1u); }
        __syncthreads();
    }
}

extern "C" __global__ void tv_pers_empty(u32 unused) { (void)unused; }
