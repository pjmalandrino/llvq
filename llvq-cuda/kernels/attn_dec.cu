// attn_dec.cu: split-KV grouped decode and verify attention for Qwen3
// (head_dim 128) on a `Grow` KV store, one launch per layer.
//
// SELF-CONTAINED. This file is its own NVRTC translation unit: it includes
// nothing and is concatenated with nothing, so the projections' unit keeps
// its bytes and its printed sha256 whatever this file does. f16 is carried as
// `unsigned short` and converted by `attn_h2f` / `attn_f2h` below (the runtime
// image has no cuda_fp16.h).
//
// THE ARITHMETIC IS CANONICAL (bit-relevant). It is stated once in Rust,
// `llvq-cuda/src/attn_ref.rs`, and this text must return its bits:
//   - 16 lanes own one (row, query head); lane l owns dims 8l..8l+7 and its
//     partial dot is an fma chain over them from +0;
//   - the score is the xor butterfly 8, 4, 2, 1 over the 16 partials, then
//     one RN product by c = scale x log2(e), rounded to f32 on the host;
//   - a tile is ATTN_TT = 8 positions aligned to absolute positions: tile
//     maximum, carry ex2(m_old - m_new), p = ex2(s - m_new), their sum as a
//     left fold from +0, d <- d*carry + ts, each accumulator <- acc*carry then
//     an fma chain over the tile's positions in order; positions at or past
//     the row's length do no arithmetic and are never multiplied;
//   - a unit is UNIT positions (template parameter, 64 served, 32 for the
//     bench), tiles in order; a split is 256 positions aligned to multiples
//     of 256, units folded left to right; a row folds its splits left to
//     right; the merge is two RN products and one RN sum per term;
//   - the output is f16 round-to-nearest of acc / d (IEEE division);
//   - exp2 is attn_ex2, a fixed polynomial (coefficients in attn_ref.rs).
// Every float operation below is an explicit RN intrinsic, so NVRTC's
// `--fmad=true` default cannot contract anything. Which tiles, units and
// splits a row visits depends on its length alone, and the combine order on
// split indices alone: a row's bits do not depend on the number of rows in the
// call, on the grid, or on which CTA finishes last.
//
// THE MAPPING IS NOT (bit-neutral). Grid (gridDim.x, n_kv): one CTA per
// (split, KV head), 128 threads = 8 groups of 16 lanes. A split holds
// UPS = 256 / UNIT units. Group w takes unit w % UPS and the query heads
// [g0, g0 + nh) of its KV head, g0 = (w / UPS) * GH. All groups stage the
// split's tiles together: at step st the CTA loads tile st of every unit of
// the split, K and V, 32 positions, 16 KB, into shared memory, with the next
// step's loads already in flight in registers. The unit states are then
// folded in order by the whole CTA, and the split state goes to a partials
// buffer. Every launched CTA draws one ticket on its KV head's counter, early
// exits included; the last one folds the splits of every row in order, writes
// the f16 output and resets the counter. That is planes_occ.cu's occ_sk
// pattern: fence, barrier, atomic, barrier-plus-OR, acquire fence, volatile
// reads, self-reset. A grid wider than the work stays correct.
//
// Verify (k > 1): rows are processed one after the other by the same CTA,
// with the same calls, hence the same bits as k separate decode calls. Fresh
// rows: when `k_new` is not null, the CTA that owns the split of position
// pos0 + r copies row r into the cache before reading it, so no position is
// written by one CTA and read by another in the same launch. When it is
// null, the cache already holds them (the model's append does).
//
// Who compiles this text: NVRTC at load on the card; clang++ on the Mac
// through `llvq-llm/tests/host_attn_dec.cpp`, which EXECUTES the whole kernel
// with 128 host threads per CTA (real barriers, 16-lane shuffles) and diffs
// it against attn_ref.rs; and bin/cuhcheck, which parses it.

#define ATTN_HD 128
#define ATTN_LANES 16
#define ATTN_DPL 8
#define ATTN_TT 8
#define ATTN_SPLIT 256
#define ATTN_KMAX 8
#define ATTN_THREADS 128
#define ATTN_GROUPS 8
#define ATTN_PSTRIDE 130

// ---------------------------------------------------------------- RN operations
#ifndef LLVQ_HOST_BUILD
__device__ __forceinline__ float rn_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float rn_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float rn_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float rn_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float rn_fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float attn_h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short attn_f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#define ATTN_BOUNDS __launch_bounds__(ATTN_THREADS, 2)
#else
// Host: one rounding each, provided the probe compiles with -ffp-contract=off
// (it does, and so does every host probe of this repository). _Float16 is
// clang's IEEE binary16: exact widening, round-to-nearest-even narrowing.
static inline float rn_add(float a, float b) { return a + b; }
static inline float rn_sub(float a, float b) { return a - b; }
static inline float rn_mul(float a, float b) { return a * b; }
static inline float rn_div(float a, float b) { return a / b; }
static inline float rn_fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
static inline float attn_h2f(unsigned short h) {
    _Float16 v;
    __builtin_memcpy(&v, &h, 2);
    return (float)v;
}
static inline unsigned short attn_f2h(float f) {
    _Float16 v = (_Float16)f;
    unsigned short r;
    __builtin_memcpy(&r, &v, 2);
    return r;
}
#define ATTN_BOUNDS
#endif

#define ATTN_NEG_INF __int_as_float((int)0xff800000u)

// ---------------------------------------------------------------- exp2
// 2^x for x <= 0. NaN propagates; x <= -126 and -inf give +0; then floor, an
// exact fraction, six fmas (minimax coefficients, attn_ref.rs EX2_COEF_BITS),
// and the integer part added to the exponent field in UNSIGNED arithmetic: a
// left shift of a negative int is undefined in C++17, and clang++ -O2
// compiles this text on the Mac.
__device__ __forceinline__ float attn_ex2(float x) {
    if (x != x) return x;
    if (!(x > -126.0f)) return 0.0f;
    const float fl = floorf(x);
    const float f = rn_sub(x, fl);
    float p = 0.000219691399f;
    p = rn_fma(p, f, 0.0012371795f);
    p = rn_fma(p, f, 0.00968445744f);
    p = rn_fma(p, f, 0.055482123f);
    p = rn_fma(p, f, 0.240229413f);
    p = rn_fma(p, f, 0.693147123f);
    p = rn_fma(p, f, 1.0f);
    const int e = (int)fl;
    return __int_as_float((int)((unsigned)__float_as_int(p) + ((unsigned)e << 23)));
}

// The merge coefficients of b (later) into a (earlier): 0 when b is empty
// (keep a), 1 when a is empty (take b), 2 otherwise with m, fa, fb set.
__device__ __forceinline__ int attn_merge_coef(float am, float bm, float* m, float* fa, float* fb) {
    if (bm == ATTN_NEG_INF) return 0;
    if (am == ATTN_NEG_INF) return 1;
    *m = fmaxf(am, bm);
    *fa = attn_ex2(rn_sub(am, *m));
    *fb = attn_ex2(rn_sub(bm, *m));
    return 2;
}

// The 16-lane butterfly: every lane of the group ends with the tree of the
// 16 partials. `mask16` names the group's own 16 lanes, so the two groups of
// a warp may diverge (one inactive tile) without a shuffle naming a lane that
// does not take part.
__device__ __forceinline__ float attn_tree16(float t, unsigned mask16) {
    t = rn_add(t, __shfl_xor_sync(mask16, t, 8, 16));
    t = rn_add(t, __shfl_xor_sync(mask16, t, 4, 16));
    t = rn_add(t, __shfl_xor_sync(mask16, t, 2, 16));
    t = rn_add(t, __shfl_xor_sync(mask16, t, 1, 16));
    return t;
}

// The 8 halves of a 16-byte chunk, in memory order.
__device__ __forceinline__ void attn_unpack8(const uint4 w, float (&f)[ATTN_DPL]) {
    f[0] = attn_h2f((unsigned short)(w.x & 0xffffu));
    f[1] = attn_h2f((unsigned short)(w.x >> 16));
    f[2] = attn_h2f((unsigned short)(w.y & 0xffffu));
    f[3] = attn_h2f((unsigned short)(w.y >> 16));
    f[4] = attn_h2f((unsigned short)(w.z & 0xffffu));
    f[5] = attn_h2f((unsigned short)(w.z >> 16));
    f[6] = attn_h2f((unsigned short)(w.w & 0xffffu));
    f[7] = attn_h2f((unsigned short)(w.w >> 16));
}

// One step stages STAGE = (256 / UNIT) x TT positions, K then V, as 16-byte
// chunks: chunk idx = i*128 + t of thread t is tensor idx / (STAGE*16) (K
// first), staged position pl = (idx mod STAGE*16) / 16, bytes 16*(idx mod 16)
// of its row. Staged position pl belongs to unit pl / TT, tile st, offset
// pl % TT. Positions at or past `lr` are zero-filled and never used.
template <int UNIT, int STAGE, int NLD>
__device__ __forceinline__ void attn_load_step(uint4 (&pf)[NLD], const unsigned short* kc,
                                               const unsigned short* vc, long long head_base,
                                               int base, int st, int lr, int t) {
#pragma unroll
    for (int i = 0; i < NLD; ++i) {
        const int idx = i * ATTN_THREADS + t;
        const int tens = idx / (STAGE * 16);
        const int rem = idx % (STAGE * 16);
        const int pl = rem >> 4;
        const int ch = rem & 15;
        const int p = base + (pl / ATTN_TT) * UNIT + st * ATTN_TT + (pl % ATTN_TT);
        uint4 z;
        z.x = 0u; z.y = 0u; z.z = 0u; z.w = 0u;
        if (p < lr) {
            const unsigned short* src = (tens ? vc : kc) + (head_base + (long long)p) * ATTN_HD;
            z = ((const uint4*)src)[ch];
        }
        pf[i] = z;
    }
}

template <int G, int UNIT>
__device__ __forceinline__ void attn_dec_body(
    const unsigned short* __restrict__ q,      // [n_kv*G][k][HD], after norm and RoPE
    const unsigned short* __restrict__ k_new,  // [n_kv][k][HD], or null
    const unsigned short* __restrict__ v_new,  // [n_kv][k][HD], or null
    unsigned short* kc,                        // cache K [n_kv][cap][HD]
    unsigned short* vc,                        // cache V [n_kv][cap][HD]
    int pos0, int k, int cap, float c,
    float* part,                               // [n_kv][s_stride][KMAX][G][PSTRIDE]
    unsigned* ticket,                          // [n_kv], zero at rest
    unsigned short* out,                       // [n_kv*G][k][HD]
    int s_stride) {
    constexpr int UPS = ATTN_SPLIT / UNIT;     // units per split: 4 or 8
    constexpr int GPU = ATTN_GROUPS / UPS;     // groups per unit: 2 or 1
    constexpr int GH = (G + GPU - 1) / GPU;    // query heads a group holds at most
    constexpr int STEPS = UNIT / ATTN_TT;      // tiles per unit: 8 or 4
    constexpr int STAGE = UPS * ATTN_TT;       // positions a step stages: 32 or 64
    constexpr int NLD = STAGE * 2 * (ATTN_HD / 8) / ATTN_THREADS;  // chunks a thread loads
    constexpr int STAGE_U4 = 2 * STAGE * (ATTN_HD / 8);
    constexpr int UST_U4 = (UPS * G * ATTN_PSTRIDE + 3) / 4;
    constexpr int SMEM_U4 = STAGE_U4 > UST_U4 ? STAGE_U4 : UST_U4;
    static_assert(UPS * GPU == ATTN_GROUPS, "every group has a unit");
    static_assert(NLD * ATTN_THREADS == STAGE_U4, "the loads cover the stage exactly");

    // One buffer, two lives: the staged tiles during the steps, the unit
    // states after them (a barrier separates the two). 16 KB at UNIT 64,
    // 32 KB at UNIT 32, under the 48 KB of static shared memory.
    __shared__ uint4 smem[SMEM_U4];
    uint4 (*stage)[STAGE][ATTN_HD / 8] = reinterpret_cast<uint4 (*)[STAGE][ATTN_HD / 8]>(smem);
    float (*ust)[G][ATTN_PSTRIDE] = reinterpret_cast<float (*)[G][ATTN_PSTRIDE]>(smem);

    const int s = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int t = (int)threadIdx.x;
    const int w = t >> 4;
    const int lane = t & 15;
    const unsigned mask16 = 0xffffu << (t & 16);
    const int u = w % UPS;
    const int g0 = (w / UPS) * GH;
    const int nh = (G - g0 < GH) ? (G - g0) : GH;
    const int lmax = pos0 + k;
    const int base = s * ATTN_SPLIT;
    const long long head_base = (long long)h * cap;

    if (base < lmax) {
        if (k_new != 0) {
            for (int r = 0; r < k; ++r) {
                const int p = pos0 + r;
                if (p >= base && p < base + ATTN_SPLIT) {
                    const long long src = ((long long)h * k + r) * ATTN_HD + t;
                    kc[(head_base + p) * ATTN_HD + t] = k_new[src];
                    vc[(head_base + p) * ATTN_HD + t] = v_new[src];
                }
            }
            __syncthreads();
        }
        for (int r = 0; r < k; ++r) {
            const int lr = pos0 + r + 1;
            if (base >= lr) continue;
            const int nsteps_raw = (lr - base + ATTN_TT - 1) / ATTN_TT;
            const int nsteps = nsteps_raw < STEPS ? nsteps_raw : STEPS;

            float qf[GH][ATTN_DPL];
            float m[GH], dd[GH], acc[GH][ATTN_DPL];
#pragma unroll
            for (int j = 0; j < GH; ++j) {
                m[j] = ATTN_NEG_INF;
                dd[j] = 0.0f;
#pragma unroll
                for (int i = 0; i < ATTN_DPL; ++i) {
                    acc[j][i] = 0.0f;
                    qf[j][i] = 0.0f;
                }
                if (j < nh) {
                    const unsigned short* qr =
                        q + (((long long)(h * G + g0 + j)) * k + r) * ATTN_HD + lane * ATTN_DPL;
#pragma unroll
                    for (int i = 0; i < ATTN_DPL; ++i) qf[j][i] = attn_h2f(qr[i]);
                }
            }

            uint4 pf[NLD];
            attn_load_step<UNIT, STAGE, NLD>(pf, kc, vc, head_base, base, 0, lr, t);
            for (int st = 0; st < nsteps; ++st) {
                __syncthreads();  // the previous step's readers are done
#pragma unroll
                for (int i = 0; i < NLD; ++i) {
                    const int idx = i * ATTN_THREADS + t;
                    stage[idx / (STAGE * 16)][(idx % (STAGE * 16)) >> 4][idx & 15] = pf[i];
                }
                __syncthreads();  // the step is staged
                if (st + 1 < nsteps)
                    attn_load_step<UNIT, STAGE, NLD>(pf, kc, vc, head_base, base, st + 1, lr, t);

                const int p0 = base + u * UNIT + st * ATTN_TT;
                if (p0 < lr) {  // uniform over the group
                    const int nv = (lr - p0 < ATTN_TT) ? (lr - p0) : ATTN_TT;
                    float sc[GH][ATTN_TT];
#pragma unroll
                    for (int j = 0; j < ATTN_TT; ++j) {
#pragma unroll
                        for (int jj = 0; jj < GH; ++jj) sc[jj][j] = 0.0f;
                        if (j < nv) {
                            float kf[ATTN_DPL];
                            attn_unpack8(stage[0][u * ATTN_TT + j][lane], kf);
#pragma unroll
                            for (int jj = 0; jj < GH; ++jj) {
                                if (jj < nh) {
                                    float d8 = 0.0f;
#pragma unroll
                                    for (int i = 0; i < ATTN_DPL; ++i) d8 = rn_fma(qf[jj][i], kf[i], d8);
                                    sc[jj][j] = rn_mul(attn_tree16(d8, mask16), c);
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int jj = 0; jj < GH; ++jj) {
                        if (jj < nh) {
                            float mt = sc[jj][0];
#pragma unroll
                            for (int j = 1; j < ATTN_TT; ++j)
                                if (j < nv) mt = fmaxf(mt, sc[jj][j]);
                            const bool fresh = (m[jj] == ATTN_NEG_INF);
                            const float mn = fresh ? mt : fmaxf(m[jj], mt);
                            const float carry = fresh ? 0.0f : attn_ex2(rn_sub(m[jj], mn));
                            float p[ATTN_TT];
                            float ts = 0.0f;
#pragma unroll
                            for (int j = 0; j < ATTN_TT; ++j) {
                                p[j] = 0.0f;
                                if (j < nv) {
                                    p[j] = attn_ex2(rn_sub(sc[jj][j], mn));
                                    ts = rn_add(ts, p[j]);
                                }
                            }
                            dd[jj] = rn_add(rn_mul(dd[jj], carry), ts);
#pragma unroll
                            for (int i = 0; i < ATTN_DPL; ++i) acc[jj][i] = rn_mul(acc[jj][i], carry);
#pragma unroll
                            for (int j = 0; j < ATTN_TT; ++j) {
                                if (j < nv) {
                                    float vf[ATTN_DPL];
                                    attn_unpack8(stage[1][u * ATTN_TT + j][lane], vf);
#pragma unroll
                                    for (int i = 0; i < ATTN_DPL; ++i)
                                        acc[jj][i] = rn_fma(p[j], vf[i], acc[jj][i]);
                                }
                            }
                            m[jj] = mn;
                        }
                    }
                }
            }

            // Unit states, then the split state, folded in unit order. The
            // states overwrite the stage, so every reader of it is done first.
            __syncthreads();
#pragma unroll
            for (int jj = 0; jj < GH; ++jj) {
                if (jj < nh) {
#pragma unroll
                    for (int i = 0; i < ATTN_DPL; ++i) ust[u][g0 + jj][lane * ATTN_DPL + i] = acc[jj][i];
                    if (lane == 0) {
                        ust[u][g0 + jj][ATTN_HD] = m[jj];
                        ust[u][g0 + jj][ATTN_HD + 1] = dd[jj];
                    }
                }
            }
            __syncthreads();
            for (int g = 0; g < G; ++g) {
                float sm = ATTN_NEG_INF, sd = 0.0f, sa = 0.0f;
                for (int uu = 0; uu < UPS; ++uu) {
                    const float bm = ust[uu][g][ATTN_HD];
                    float mm = 0.0f, fa = 0.0f, fb = 0.0f;
                    const int how = attn_merge_coef(sm, bm, &mm, &fa, &fb);
                    if (how == 1) {
                        sm = bm;
                        sd = ust[uu][g][ATTN_HD + 1];
                        sa = ust[uu][g][t];
                    } else if (how == 2) {
                        sd = rn_add(rn_mul(sd, fa), rn_mul(ust[uu][g][ATTN_HD + 1], fb));
                        sa = rn_add(rn_mul(sa, fa), rn_mul(ust[uu][g][t], fb));
                        sm = mm;
                    }
                }
                float* P = part + ((((long long)h * s_stride + s) * ATTN_KMAX + r) * G + g) * ATTN_PSTRIDE;
                P[t] = sa;
                if (t == 0) {
                    P[ATTN_HD] = sm;
                    P[ATTN_HD + 1] = sd;
                }
            }
            __syncthreads();  // ust is reused by the next row
        }
    }

    // One ticket per launched CTA, active or not (occ_sk).
    __threadfence();
    __syncthreads();
    unsigned tk = 0u;
    if (t == 0) tk = atomicAdd(ticket + h, 1u);
    const int last = __syncthreads_or(t == 0 && tk == gridDim.x - 1u);
    if (!last) return;
    __threadfence();  // acquire: the reads below follow the ticket we drew

    for (int r = 0; r < k; ++r) {
        const int lr = pos0 + r + 1;
        const int ns = (lr + ATTN_SPLIT - 1) / ATTN_SPLIT;
        for (int g = 0; g < G; ++g) {
            float am = ATTN_NEG_INF, ad = 0.0f, aa = 0.0f;
            for (int ss = 0; ss < ns; ++ss) {
                const volatile float* P =
                    part + ((((long long)h * s_stride + ss) * ATTN_KMAX + r) * G + g) * ATTN_PSTRIDE;
                const float bm = P[ATTN_HD];
                float mm = 0.0f, fa = 0.0f, fb = 0.0f;
                const int how = attn_merge_coef(am, bm, &mm, &fa, &fb);
                if (how == 1) {
                    am = bm;
                    ad = P[ATTN_HD + 1];
                    aa = P[t];
                } else if (how == 2) {
                    ad = rn_add(rn_mul(ad, fa), rn_mul(P[ATTN_HD + 1], fb));
                    aa = rn_add(rn_mul(aa, fa), rn_mul(P[t], fb));
                    am = mm;
                }
            }
            out[(((long long)(h * G + g)) * k + r) * ATTN_HD + t] = attn_f2h(rn_div(aa, ad));
        }
    }
    if (t == 0) ticket[h] = 0u;  // self-reset; the next launch is ordered after this one
}

#define ATTN_ENTRY(NAME, G, UNIT)                                                              \
    extern "C" __global__ void ATTN_BOUNDS NAME(                                               \
        const unsigned short* q, const unsigned short* k_new, const unsigned short* v_new,    \
        unsigned short* kc, unsigned short* vc, int pos0, int k, int cap, float c,            \
        float* part, unsigned* ticket, unsigned short* out, int s_stride) {                    \
        attn_dec_body<G, UNIT>(q, k_new, v_new, kc, vc, pos0, k, cap, c, part, ticket, out,    \
                               s_stride);                                                      \
    }

ATTN_ENTRY(attn_dec_g4_u64, 4, 64)
ATTN_ENTRY(attn_dec_g5_u64, 5, 64)
ATTN_ENTRY(attn_dec_g4_u32, 4, 32)
ATTN_ENTRY(attn_dec_g5_u32, 5, 32)
