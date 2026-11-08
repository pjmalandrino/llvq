// pers_4ev.cu: 4E's verify pass. One cooperative launch runs n rows of the
// decode (n <= P4V_KMAX), row r the token toks[r] at position pos0 + r, and
// writes each row's argmax: the target half of a greedy speculative decode
// (docs/plan-speculation.md). Row r owes the bits of the 4E token at that
// position, which owe the served chain's (pers_4e.cu): the same embedding,
// the same KV it appends, the same logits, the same argmax.
//
// Where the rows share work: every Tetra and int4 row phase (q/k/v, o,
// gate/up, down, the head) decodes a weight once and multiplies it into the
// staged vectors of every row of a group, one accumulator a row, each in the
// served lane order (pers_tetra_row, pers_q4_row_tile, pers_q4g64_row) term for
// term. A group is as many rows as the CTA's staging area holds at the
// phase's width (`act_floats` / width, the host sizes the area): at the 4B,
// four rows at 2,560 and 4,096, two at the down's 9,728. Everything else runs
// row after row inside the phase with the single-row code: the norms, the
// rotations, RoPE and the append, and attention, whose jobs of every row are
// dealt over the whole grid (row r's length is pos0 + r + 1, and a row's bits
// depend on its length alone, llvq_pers.cuh section 10).
//
// The parts it runs are the forms every 4E row form is proven equal to: the
// scalar norm and rotation prologues (pers_rot_finish without staged `small`),
// the tiled int4 rows of form 7, attention's cp.async ring of form 6. So it
// is built only with PERS_4E_ROWS 7, the v9 module, after pers_4e.cu (its
// Pers4eArgs and Pers4eLayer), and appended to the module only when the
// runtime installs the speculative decode: the served module's text does not
// move otherwise.
//
// Waits: the same 2 + 6 x layers as one 4E token (pers.rs::n_bar_4e),
// whatever n, so the host advances the barrier base by that once a launch.

#ifndef LLVQ_PERS_CUH
#include "llvq_pers.cuh"
#endif

#if PERS_4E_ROWS < 7
#error "pers_4ev.cu reads the int4 rows on tiled words and attention's ring: build it with PERS_4E_ROWS 7 (4E v9)"
#endif

#ifndef P4V_KMAX
#define P4V_KMAX 4u
#endif
// The staging area: the k rows of a rotation's `small` (k x KMAX floats, k at
// most 20: 19 is the largest odd factor of the served widths), then the
// group's rows, then the head's maxima in its last 2 x P4V_KMAX floats.
#define P4V_SSM 640u

struct Pers4vLaunch {
    u32 bar_base;               // arrivals before this launch (the 4E ledger's)
    u32 pos0;                   // the position of row 0
    u32 n_rows;                 // 1..P4V_KMAX
    u32 act_floats;             // the staging area after the tables, floats, even
    const u32* toks;            // [n_rows] the rows' tokens
    u32* ring;                  // [n_rows] each row's argmax
    unsigned short* x_a;        // [KMAX][hidden] the residual, ping
    unsigned short* x_b;        // [KMAX][hidden] pong
    unsigned short* q_raw;      // [KMAX][n_heads x 128]
    unsigned short* k_raw;      // [KMAX][n_kv x 128]
    unsigned short* v_raw;      // [KMAX][n_kv x 128]
    unsigned short* q_out;      // [KMAX][n_heads x 128] after norm and RoPE
    unsigned short* ctx;        // [KMAX][n_heads x 128]
    unsigned short* act;        // [KMAX][inter]
    float* part;                // [KMAX][part_stride] attention's split states
    u32* tickets;               // [KMAX][n_kv] the split fold's counters, zero at rest
    unsigned long long* slots;  // [KMAX] argmax keys, cleared at launch start
    u32 part_stride;            // floats a row: n_kv x ceil(cap / 256) x groups x 130
    u32 pad;
    unsigned long long* stamps; // null, or [1 + waits][grid]: each CTA's clock at its start and after each wait
};

// Rows of a group at width w: what `floats` holds, at least one.
__device__ __forceinline__ u32 p4v_group(u32 n, u32 floats, u32 w)
{
    const u32 g = floats / w;
    return g == 0u ? 1u : (g < n ? g : n);
}

// One Tetra row against n staged vectors (vector r at xs + r x xst, natural
// stride 24, its tail after the blocks). Per vector: the lane's blocks
// j = lane + 32 i in increasing j, the 24 products in TETRA48_ORDER from 0,
// site 1 (fma(a x gscale[g], invnorm[m], acc)), then warp_sum. That is
// pers_tetra_row<24, false, *>'s sum term for term; the block's words and
// its 24 weights are read and decoded once for every vector.
__device__ __forceinline__ void p4v_tetra_row(const u32* wrow, const F1rTables& tab, const float* gscale,
                                              const float* invnorm, const float* xs, u32 xst, u32 n, u32 nblocks,
                                              u32 lane, float out[P4V_KMAX])
{
    float acc[P4V_KMAX];
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r) acc[r] = 0.0f;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nblocks; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        float w[24];
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
            for (u32 k = 0; k < 4u; ++k) w[4u * i + k] = f1r_v3_float(q[i], k);
        }
        const u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
        const float gs = gscale[(hi16 >> 15) & 1u], iv = invnorm[m];
#pragma unroll
        for (u32 r = 0; r < P4V_KMAX; ++r) {
            if (r < n) {
                const float* xb = xs + r * xst + j * 24u;
                float a = 0.0f;
#pragma unroll
                for (u32 t = 0; t < 24u; ++t) a = __fmaf_rn(w[t], xb[TETRA48_ORDER[t]], a);
                acc[r] = __fmaf_rn(__fmul_rn(a, gs), iv, acc[r]);
            }
        }
        lo = lo2;
        hi16 = hi2;
    }
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r)
        if (r < n) out[r] = warp_sum(acc[r]);
}

// One int4 g128 row on form 7's tiled words against n staged vectors:
// pers_q4_row_tile's words, scales and products in its order, one
// accumulator a vector, each word dequantized once.
__device__ __forceinline__ void p4v_q4_row(const u32* wq, const unsigned short* scales, const unsigned short* biases,
                                           const float* xs, u32 xst, u32 n, u32 row, u32 d_in, u32 gpr, u32 lane,
                                           float out[P4V_KMAX])
{
    const u32 nwords = d_in >> 3, nt = nwords >> 7;
    const u32* wrow = wq + (size_t)row * nwords;
    const u32 s0 = row * gpr;
    float acc[P4V_KMAX];
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r) acc[r] = 0.0f;
    uint4 P{0u, 0u, 0u, 0u};
    if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * lane);
    for (u32 t = 0; t < nt; ++t) {
        uint4 P2{0u, 0u, 0u, 0u};
        if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + ((t + 1u) << 7) + 4u * lane);
        const u32 wv[4] = {P.x, P.y, P.z, P.w};
#pragma unroll
        for (u32 s = 0; s < 4u; ++s) {
            const u32 c = ((t << 7) + 32u * s + lane) << 3;
            const u32 g = (c >> 7) + s0;
            const u32 sb = scales[g], bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k) {
                const float d = pers_q4_deq((wv[s] >> (4u * k)) & 0xfu, sb, bb);
#pragma unroll
                for (u32 r = 0; r < P4V_KMAX; ++r)
                    if (r < n) acc[r] = __fmaf_rn(d, xs[r * xst + c + k], acc[r]);
            }
        }
        P = P2;
    }
    for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {
        const u32 p = wrow[wi];
        const u32 c = wi << 3;
        const u32 g = (c >> 7) + s0;
        const u32 sb = scales[g], bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) {
            const float d = pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb);
#pragma unroll
            for (u32 r = 0; r < P4V_KMAX; ++r)
                if (r < n) acc[r] = __fmaf_rn(d, xs[r * xst + c + k], acc[r]);
        }
    }
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r)
        if (r < n) out[r] = warp_sum(acc[r]);
}

// One head row (int4 g64, the served word order) against n staged vectors:
// pers_q4g64_row's products in its order, one accumulator a vector.
__device__ __forceinline__ void p4v_q4g64_row(const u32* wq, const unsigned short* scales,
                                              const unsigned short* biases, const float* xs, u32 xst, u32 n, u32 row,
                                              u32 d, u32 gpr, u32 lane, float out[P4V_KMAX])
{
    const u32 nwords = d >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;
    float acc[P4V_KMAX];
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r) acc[r] = 0.0f;
    u32 p = 0u, sb = 0u, bb = 0u;
    if (lane < nwords) {
        p = wq[w0 + lane];
        sb = scales[s0 + (lane >> 3)];
        bb = biases[s0 + (lane >> 3)];
    }
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p2 = 0u, sb2 = 0u, bb2 = 0u;
        if (wi + 32u < nwords) {
            p2 = wq[w0 + wi + 32u];
            sb2 = scales[s0 + ((wi + 32u) >> 3)];
            bb2 = biases[s0 + ((wi + 32u) >> 3)];
        }
        const u32 c = wi << 3;
#pragma unroll
        for (u32 k = 0; k < 8u; ++k) {
            const float dq = pers_q4_deq((p >> (4u * k)) & 0xfu, sb, bb);
#pragma unroll
            for (u32 r = 0; r < P4V_KMAX; ++r)
                if (r < n) acc[r] = __fmaf_rn(dq, xs[r * xst + c + k], acc[r]);
        }
        p = p2;
        sb = sb2;
        bb = bb2;
    }
#pragma unroll
    for (u32 r = 0; r < P4V_KMAX; ++r)
        if (r < n) out[r] = warp_sum(acc[r]);
}

// candle's RMSNorm of row x (binary16, global, written in this launch) into
// s as f32 values of the binary16 results, each sign-flipped where `signs`
// has its bit (null: none). Ends with a barrier. The scalar path every 4E
// form falls back on.
__device__ __forceinline__ void p4v_norm(const unsigned short* x, const unsigned short* alpha, const u32* signs,
                                         float eps, u32 w, float* s)
{
    const float sc = exact_rms_scale<0>(x, w, eps);
    for (u32 i = threadIdx.x; i < w; i += blockDim.x) {
        const float h = h2f(exact_rms_apply(sc, __ldcg(x + i), alpha[i]));
        s[i] = (signs != nullptr && ((signs[i >> 5] >> (i & 31u)) & 1u)) ? -h : h;
    }
    __syncthreads();
}

// ln of row x into s, signed by `signs`, and the k rows of `small` staged in
// ssm for pers_rot_finish<true, true>: pers_ln_1rt (form 4's one round trip)
// when `scratch` (w free floats) is given and the inputs are 16-byte aligned,
// else the scalar path and the staging after it. The same values either way
// (every 4E form is one of the two). No trailing barrier on the vector path:
// pers_rot_finish starts with one, other callers add one.
__device__ __forceinline__ void p4v_ln(const unsigned short* x, const unsigned short* alpha, const u32* signs,
                                       const float* small, u32 k, u32 w, float eps, float* s, float* scratch, float* ssm)
{
    if (scratch != nullptr && pers_1rt_ok(x, w) && pers_1rt_ok(alpha, w)) {
        pers_ln_1rt<0>(x, alpha, signs, w, eps, small, k, ssm, s, scratch, false);
    } else {
        p4v_norm(x, alpha, signs, eps, w, s);
        pers_small_store(pers_small_load(small, k), small, k, ssm);
    }
}

// A binary16 row of global memory into s: signed for a rotation, its `small`
// staged in ssm (signs non-null), or widened (null). pers_load_1rt when
// aligned, else the scalar loads. No trailing barrier (see p4v_ln).
__device__ __forceinline__ void p4v_load(const unsigned short* x, const u32* signs, const float* small, u32 k, u32 w,
                                         float* s, float* ssm)
{
    if (pers_1rt_ok(x, w)) {
        pers_load_1rt(x, signs, s, w, signs != nullptr ? small : nullptr, k, ssm);
    } else if (signs != nullptr) {
        pers_rot_load(x, signs, s, w);
        pers_small_store(pers_small_load(small, k), small, k, ssm);
    } else {
        for (u32 i = threadIdx.x; i < w; i += blockDim.x) s[i] = h2f(__ldcg(x + i));
    }
}

// Attention for every row of the launch: pers_attention's job body on form
// 6's ring, the jobs of row 0, then row 1, ... dealt over the grid. Row r
// reads q_out and writes ctx, its split states and its tickets at its own
// offsets; its length is pos0 + r + 1.
__device__ __forceinline__ void p4v_attention(const Pers4eArgs* A, const Pers4vLaunch& L, const unsigned short* kc,
                                              const unsigned short* vc, unsigned char* smem)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 n_heads = A->n_heads, groups = A->groups, cap = A->cap, n_kv = A->n_kv;
    const float c = A->c;
    const u32 s_stride = (cap + PERS_SPLIT - 1u) / PERS_SPLIT;
    const u32 nhp = (groups + 1u) / 2u, ntasks = PERS_UPS * nhp;
    uint4* stage = reinterpret_cast<uint4*>(smem);
    float* ust = reinterpret_cast<float*>(smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);  // [UPS][groups][PSTRIDE]
    float* qs = ust + PERS_UPS * groups * PERS_PSTRIDE;                                  // [groups][128]
    u32* hdr = pers_hdr();
    u32 total = 0u;
    for (u32 r = 0; r < L.n_rows; ++r) total += n_kv * ((L.pos0 + r + 1u + PERS_SPLIT - 1u) / PERS_SPLIT);
    for (u32 job = blockIdx.x; job < total; job += gridDim.x) {
        u32 r = 0u, j = job;
        for (;;) {
            const u32 nj = n_kv * ((L.pos0 + r + 1u + PERS_SPLIT - 1u) / PERS_SPLIT);
            if (j < nj) break;
            j -= nj;
            ++r;
        }
        const u32 l = L.pos0 + r + 1u;
        const u32 ns = (l + PERS_SPLIT - 1u) / PERS_SPLIT;
        const u32 s = j / n_kv, h = j - s * n_kv;
        const unsigned short* q = L.q_out + (size_t)r * n_heads * PERS_HD;
        unsigned short* out = L.ctx + (size_t)r * n_heads * PERS_HD;
        float* part = L.part + (size_t)r * L.part_stride;
        u32* ticket = L.tickets + r * n_kv;
        const u32 base = s * PERS_SPLIT;
        const u32 nu = (l - base + PERS_UNIT - 1u) / PERS_UNIT;
        const u32 tpu = nu >= 3u ? 1u : (nu == 2u ? 2u : 4u);
        const u32 tiles_raw = (l - base + PERS_TT - 1u) / PERS_TT;
        const u32 tiles = tiles_raw < PERS_UNIT / PERS_TT ? tiles_raw : PERS_UNIT / PERS_TT;
        const u32 nsteps = (tiles + tpu - 1u) / tpu;
        const unsigned short* kh = kc + (unsigned long long)h * cap * PERS_HD;
        const unsigned short* vh = vc + (unsigned long long)h * cap * PERS_HD;
        for (u32 st = 0; st < PERS_ATTN_RING; ++st)
            pers_attn_issue(stage + st * PERS_STAGE_U4, kh, vh, base, st, nsteps, tpu, l);
        for (u32 i = threadIdx.x; i < PERS_UPS * groups * PERS_PSTRIDE; i += blockDim.x) {
            const u32 k = i % PERS_PSTRIDE;
            ust[i] = k == PERS_HD ? PERS_NEG_INF : 0.0f;
        }
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) qs[i] = h2f(__ldcg(q + h * groups * PERS_HD + i));
        for (u32 st = 0; st < nsteps; ++st) {
            uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;
            pers_cp_wait_ring();
            __syncthreads();
            for (u32 task = warp; task < ntasks; task += nwarps) {
                const u32 u = task / nhp, hp = task - u * nhp;
                if (u * tpu >= PERS_UPS) continue;
                const u32 g = 2u * hp + (lane >> 4);
                const bool own = g < groups;
                const u32 ge = own ? g : groups - 1u;
                for (u32 t = 0; t < tpu; ++t) {
                    const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
                    if (p0 >= l || (st * tpu + t) * PERS_TT >= PERS_UNIT) break;
                    const u32 nv = l - p0 < PERS_TT ? l - p0 : PERS_TT;
                    const u32 sl = (u * tpu + t) * PERS_TT;
                    pers_attn_tile(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own, sbuf + sl * 16u,
                                   sbuf + (PERS_STAGE + sl) * 16u, nv, c, lane);
                }
            }
            __syncthreads();
            pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);
        }
        pers_cp_wait_all();
        __syncthreads();
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

template <int FORM>
__device__ __forceinline__ void pers_4ev_body(const Pers4eArgs* __restrict__ A, const Pers4vLaunch L)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* S = t_inv + 32;  // the staging area, L.act_floats floats; attention's stage in its phase
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = A->pref[i];
        t_suf[i] = A->suf[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
    __syncthreads();
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    const u32 H = A->hidden, G = gridDim.x, c0 = blockIdx.x, n = L.n_rows, F = L.act_floats;
    float* const ssm = S;            // the k rows of `small`, 16-byte aligned
    float* const R = S + P4V_SSM;    // the group's rows
    const u32 FR = F - P4V_SSM - 2u * P4V_KMAX;
    // The ln of row r uses row r + 1's place as its scratch; the last row
    // of a group too when the area holds one more row, else the scalar path
    // (the same values either way, p4v_ln).
    const auto scratch_of = [&](u32 r, u32 w) { return (r + 2u) * w <= FR ? R + (r + 1u) * w : nullptr; };
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 dq = A->n_heads * 128u, dkv = A->n_kv * 128u;
    u32 nb = 0u;
#define P4V_WAIT()                                                                          \
    do {                                                                                    \
        if (!pers_grid_wait(A->bar, L.bar_base + (++nb) * G, A->abort_flag)) return;        \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)nb * G + c0] = pers_now(); \
    } while (0)
    // Phase stamps (a measurement launch, L.stamps non-null): event 0 at the
    // CTA's start, event e after its e-th wait, thread 0 of each CTA once the
    // wait returned. Phase e lasts from event e - 1 to event e. The clock is
    // pers_now's; nothing a row computes reads it.
    if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[c0] = pers_now();
    // The argmax keys and the tickets start at zero; the first barrier's
    // fence orders these stores before any CTA's atomics.
    if (c0 == 0u) {
        for (u32 i = threadIdx.x; i < P4V_KMAX; i += blockDim.x) L.slots[i] = 0ull;
        for (u32 i = threadIdx.x; i < P4V_KMAX * A->n_kv; i += blockDim.x) L.tickets[i] = 0u;
    }
    for (u32 r = 0; r < n; ++r)
        pers_embed(A->emb_w, A->emb_s, A->emb_b, L.toks[r], H, A->emb_gpr, L.x_a + (size_t)r * H, H * c0 / G,
                   H * (c0 + 1u) / G);
    P4V_WAIT();
    for (u32 l = 0; l < A->n_layers; ++l) {
        const Pers4eLayer& Y = A->layers[l];
        const Pers4eQkv& N = Y.qkv;
        // q/k/v: ln1 of each row of the group (natural basis), the int4 v
        // rows, then each row signed and rotated in place, the Tetra q|k rows.
        {
            const u32 gq = p4v_group(n, FR, H);
            const u32 qk = N.q_dout + N.k_dout, lo = qk * c0 / G, hi = qk * (c0 + 1u) / G;
            const u32 vlo = N.v_dout * c0 / G, vhi = N.v_dout * (c0 + 1u) / G;
            for (u32 r0 = 0; r0 < n; r0 += gq) {
                const u32 g = n - r0 < gq ? n - r0 : gq;
                // ln1 of each row, signed; the k rows of `small` staged
                for (u32 r = 0; r < g; ++r)
                    p4v_ln(L.x_a + (size_t)(r0 + r) * H, Y.ln1, Y.signs_a, Y.small_a, Y.k_a, H, A->eps, R + r * H,
                           scratch_of(r, H), ssm);
                // the natural values for the int4 v rows: the signs undone
                // (negation is exact), then put back for the rotation
                const auto flip = [&] {
                    __syncthreads();
                    for (u32 i = threadIdx.x; i < g * H; i += blockDim.x) {
                        const u32 c = i % H;
                        if ((Y.signs_a[c >> 5] >> (c & 31u)) & 1u) R[i] = -R[i];
                    }
                    __syncthreads();
                };
                flip();
                for (u32 rv = vlo + warp; rv < vhi; rv += nw) {
                    float o[P4V_KMAX];
                    p4v_q4_row(N.vw, N.vsc, N.vbi, R, H, g, rv, N.d_in, N.v_gpr, lane, o);
                    if (lane == 0u)
                        for (u32 r = 0; r < g; ++r) L.v_raw[(size_t)(r0 + r) * dkv + rv] = f2h(o[r]);
                }
                flip();
                for (u32 r = 0; r < g; ++r) pers_rot_finish<true, true>(R + r * H, ssm, H, Y.m_a, Y.k_a, Y.inv_a);
                for (u32 rr = lo + warp; rr < hi; rr += nw) {
                    const bool isq = rr < N.q_dout;
                    const u32 row = isq ? rr : rr - N.q_dout;
                    const u32* wrow = isq ? N.qw + (size_t)row * N.q_stride : N.kw + (size_t)row * N.k_stride;
                    float o[P4V_KMAX];
                    p4v_tetra_row(wrow, tab, isq ? N.qgs : N.kgs, t_inv, R, H, g, N.nblocks, lane, o);
                    if (lane == 0u) {
                        const float* rs = isq ? N.qrs : N.krs;
                        const unsigned short* tl = (isq ? N.qtl : N.ktl) + (size_t)row * N.tail_w;
                        unsigned short* dst = isq ? L.q_raw + row : L.k_raw + row;
                        const u32 ds = isq ? dq : dkv;
                        for (u32 r = 0; r < g; ++r)
                            dst[(size_t)(r0 + r) * ds] = f2h(pers_epilogue(o[r], rs[row], tl, R + r * H + N.nblocks * 24u, N.tail_w));
                    }
                }
                __syncthreads();
            }
        }
        P4V_WAIT();
        // q/k-norm, RoPE, the KV append: (row, head) items over every warp
        {
            const u32 per = A->n_heads + 2u * A->n_kv;
            for (u32 w = c0 + G * warp; w < n * per; w += G * nw) {
                const u32 r = w / per, hh = w - r * per, pos = L.pos0 + r;
                const unsigned short* cr = A->cosr + pos * 64u;
                const unsigned short* sr = A->sinr + pos * 64u;
                if (hh < A->n_heads) {
                    pers_head_norm_rope<FORM>(L.q_raw + (size_t)r * dq + hh * 128u, Y.q_norm, cr, sr, A->eps,
                                              L.q_out + (size_t)r * dq + hh * 128u, lane);
                } else if (hh < A->n_heads + A->n_kv) {
                    const u32 h = hh - A->n_heads;
                    pers_head_norm_rope<FORM>(L.k_raw + (size_t)r * dkv + h * 128u, Y.k_norm, cr, sr, A->eps,
                                              Y.kc + ((unsigned long long)h * A->cap + pos) * 128u, lane);
                } else {
                    const u32 h = hh - A->n_heads - A->n_kv;
                    for (u32 i = lane; i < 128u; i += 32u)
                        Y.vc[((unsigned long long)h * A->cap + pos) * 128u + i] = __ldcg(L.v_raw + (size_t)r * dkv + h * 128u + i);
                }
            }
        }
        P4V_WAIT();
        p4v_attention(A, L, Y.kc, Y.vc, reinterpret_cast<unsigned char*>(S));
        P4V_WAIT();
        // o and the residual: x_b = x_a + o(ctx)
        {
            const u32 olo = H * c0 / G, ohi = H * (c0 + 1u) / G;
            const u32 go = p4v_group(n, FR, dq);
            for (u32 r0 = 0; r0 < n; r0 += go) {
                const u32 g = n - r0 < go ? n - r0 : go;
                if (Y.ot != nullptr) {
                    const Pers4eOT& O = *Y.ot;
                    for (u32 r = 0; r < g; ++r) {
                        p4v_load(L.ctx + (size_t)(r0 + r) * dq, O.signs, O.small, O.k, dq, R + r * dq, ssm);
                        pers_rot_finish<true, true>(R + r * dq, ssm, dq, O.m, O.k, O.inv);
                    }
                    for (u32 row = olo + warp; row < ohi; row += nw) {
                        float o[P4V_KMAX];
                        p4v_tetra_row(O.t.words + (size_t)row * O.t.stride_u32, tab, O.t.gscale, t_inv, R, dq, g,
                                      O.t.nblocks, lane, o);
                        if (lane == 0u)
                            for (u32 r = 0; r < g; ++r) {
                                const size_t at = (size_t)(r0 + r) * H + row;
                                const float y = pers_epilogue(o[r], O.t.rscale[row], O.t.tail + (size_t)row * O.t.tail_w,
                                                              R + r * dq + O.t.nblocks * 24u, O.t.tail_w);
                                L.x_b[at] = exact_hadd(__ldcg(L.x_a + at), f2h(y));
                            }
                    }
                } else {
                    for (u32 r = 0; r < g; ++r) p4v_load(L.ctx + (size_t)(r0 + r) * dq, nullptr, nullptr, 0u, dq, R + r * dq, ssm);
                    __syncthreads();
                    for (u32 row = olo + warp; row < ohi; row += nw) {
                        float o[P4V_KMAX];
                        p4v_q4_row(Y.ow, Y.osc, Y.obi, R, dq, g, row, dq, Y.o_gpr, lane, o);
                        if (lane == 0u)
                            for (u32 r = 0; r < g; ++r) {
                                const size_t at = (size_t)(r0 + r) * H + row;
                                L.x_b[at] = exact_hadd(__ldcg(L.x_a + at), f2h(o[r]));
                            }
                    }
                }
                __syncthreads();
            }
        }
        P4V_WAIT();
        // MLP gate/up: ln2 of each row signed and rotated, then act = silu(g) u
        const SegArgs& M = Y.mlp;
        const u32 I = M.inter;
        {
            const u32 glo = M.slice_gu[2u * c0], ghi = M.slice_gu[2u * c0 + 1u];
            const u32 gg = p4v_group(n, FR, H);
            const u32 tw = M.gate.tail_w, nbk = M.gate.nblocks;
            for (u32 r0 = 0; r0 < n; r0 += gg) {
                const u32 g = n - r0 < gg ? n - r0 : gg;
                for (u32 r = 0; r < g; ++r) {
                    p4v_ln(L.x_b + (size_t)(r0 + r) * H, M.alpha, M.signs_h, M.small_h, M.k_h, H, M.eps, R + r * H,
                           scratch_of(r, H), ssm);
                    pers_rot_finish<true, true>(R + r * H, ssm, H, M.m_h, M.k_h, M.inv_h);
                }
                for (u32 row = glo + warp; row < ghi; row += nw) {
                    float og[P4V_KMAX], ou[P4V_KMAX];
                    p4v_tetra_row(M.gate.words + (size_t)row * M.gate.stride_u32, tab, M.gate.gscale, t_inv, R, H, g, nbk, lane, og);
                    p4v_tetra_row(M.up.words + (size_t)row * M.up.stride_u32, tab, M.up.gscale, t_inv, R, H, g, nbk, lane, ou);
                    if (lane == 0u)
                        for (u32 r = 0; r < g; ++r) {
                            const float* xt = R + r * H + nbk * 24u;
                            const unsigned short gv = f2h(pers_epilogue(og[r], M.gate.rscale[row], M.gate.tail + (size_t)row * tw, xt, tw));
                            const unsigned short uv = f2h(pers_epilogue(ou[r], M.up.rscale[row], M.up.tail + (size_t)row * tw, xt, tw));
                            L.act[(size_t)(r0 + r) * I + row] = exact_silu_mul(M.silu, gv, uv);
                        }
                }
                __syncthreads();
            }
        }
        P4V_WAIT();
        // MLP down and the residual: x_a = x_b + down(act)
        {
            const u32 dlo = M.slice_dn[2u * c0], dhi = M.slice_dn[2u * c0 + 1u];
            const u32 gd = p4v_group(n, FR, I);
            for (u32 r0 = 0; r0 < n; r0 += gd) {
                const u32 g = n - r0 < gd ? n - r0 : gd;
                for (u32 r = 0; r < g; ++r) {
                    if (Y.tetra_down) {
                        p4v_load(L.act + (size_t)(r0 + r) * I, M.signs_i, M.small_i, M.k_i, I, R + r * I, ssm);
                        pers_rot_finish<true, true>(R + r * I, ssm, I, M.m_i, M.k_i, M.inv_i);
                    } else {
                        p4v_load(L.act + (size_t)(r0 + r) * I, nullptr, nullptr, 0u, I, R + r * I, ssm);
                    }
                }
                __syncthreads();
                for (u32 row = dlo + warp; row < dhi; row += nw) {
                    float o[P4V_KMAX];
                    if (Y.tetra_down)
                        p4v_tetra_row(M.down.words + (size_t)row * M.down.stride_u32, tab, M.down.gscale, t_inv, R, I, g,
                                      M.down.nblocks, lane, o);
                    else
                        p4v_q4_row(M.q4w, M.q4s, M.q4b, R, I, g, row, I, M.gpr, lane, o);
                    if (lane == 0u)
                        for (u32 r = 0; r < g; ++r) {
                            const size_t at = (size_t)(r0 + r) * H + row;
                            const float y = Y.tetra_down
                                                ? pers_epilogue(o[r], M.down.rscale[row], M.down.tail + (size_t)row * M.down.tail_w,
                                                                R + r * I + M.down.nblocks * 24u, M.down.tail_w)
                                                : o[r];
                            L.x_a[at] = exact_hadd(__ldcg(L.x_b + at), f2h(y));
                        }
                }
                __syncthreads();
            }
        }
        P4V_WAIT();
    }
    // The final norm, the head, each row's argmax key; the CTA's maxima at
    // the end of the staging area (8-byte aligned: F is even and the area
    // starts 8-byte aligned).
    {
        unsigned long long* cmax = reinterpret_cast<unsigned long long*>(S + (F - 2u * P4V_KMAX));
        const u32 hlo = A->vocab * c0 / G, hhi = A->vocab * (c0 + 1u) / G;
        const u32 gh = p4v_group(n, FR, H);
        for (u32 r0 = 0; r0 < n; r0 += gh) {
            const u32 g = n - r0 < gh ? n - r0 : gh;
            for (u32 r = 0; r < g; ++r) p4v_norm(L.x_a + (size_t)(r0 + r) * H, A->final_norm, nullptr, A->eps, H, R + r * H);
            if (threadIdx.x < P4V_KMAX) cmax[threadIdx.x] = 0ull;
            __syncthreads();
            for (u32 row = hlo + warp; row < hhi; row += nw) {
                float o[P4V_KMAX];
                p4v_q4g64_row(A->head_w, A->head_s, A->head_b, R, H, g, row, H, A->emb_gpr, lane, o);
                if (lane == 0u)
                    for (u32 r = 0; r < g; ++r) {
                        const unsigned short hb = f2h(o[r]);
                        if (pers_is_nan16(hb)) atomicExch(A->nan_flag, 1u);
                        else atomicMax(cmax + r, pers_tie_key(hb, row));
                    }
            }
            __syncthreads();
            if (threadIdx.x < g) atomicMax(L.slots + r0 + threadIdx.x, cmax[threadIdx.x]);
            __syncthreads();
        }
    }
    P4V_WAIT();
    if (c0 == 0u && threadIdx.x < n) {
        if (pers_ld_acquire(A->nan_flag) != 0u) return;  // the host reads the flag
        const unsigned long long k = __ldcg(L.slots + threadIdx.x);
        const u32 br = 1023u - (u32)((k >> 22) & 0x3FFu);
        L.ring[threadIdx.x] = ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);
    }
#undef P4V_WAIT
}

#define PERS_4EV(sym, FORM_, THREADS_)                                                                   \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,      \
                                                                   const Pers4vLaunch L)                 \
    {                                                                                                    \
        pers_4ev_body<FORM_>(A, L);                                                                      \
    }

PERS_4EV(pers_4ev_rows_f0, 0, 768)
PERS_4EV(pers_4ev_rows_f1, 1, 768)
PERS_4EV(pers_4ev_rows_f2, 2, 768)
PERS_4EV(pers_4ev_rows_f0_t512, 0, 512)
PERS_4EV(pers_4ev_rows_f1_t512, 1, 512)
PERS_4EV(pers_4ev_rows_f2_t512, 2, 512)
