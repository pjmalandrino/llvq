// DRAFT, never compiled or run. Design sketch for wave5-deep/attention.md, section 3, revision 2.
// Not repository code. Every name, layout and constant here is a proposal. Proposed home:
// llvq-cuda/kernels/attn_dec.cu, with an entry in embedded_source and in bin/cuhcheck's units.
//
// Who compiles this text first, once it is written into the repository:
//   1. clang++ on the Mac, through a host probe (llvq-llm/tests/host_attn_dec.cpp with
//      llvq-cuda/tests/host_shim.h, `-std=c++17 -O2 -ffp-contract=off`), at $0. The probe EXECUTES
//      every shuffle-free function below and diffs it against the Rust replay bit for bit
//      (sketch: host_attn_dec_sketch.cpp, attn_ref_sketch.rs).
//   2. NVRTC at load, on the card. The image build never compiles kernel text: ops/Dockerfile.cuda
//      has no nvcc step, and kernels are include_str! sources handed to NVRTC at load.
// No cuda_fp16.h: the runtime image does not carry it (llvq-cuda/kernels/matvec.cu, f2h). f16 is
// stored as unsigned short and converted by h2f / f2h, which this file takes from matvec.cu (device:
// cvt instructions; host: exact software branches under LLVQ_HOST_BUILD).
//
// Split-KV grouped decode and verify attention for Qwen3 (head_dim 128) on a Grow(256) store.
//
// CANONICAL ARITHMETIC (bit-relevant). Frozen after the kernel-only bench V4a, before the T8 control V6:
//   - lanes: 16 lanes x 8 dims per (row, query head); each partial dot is an fma chain over the
//     lane's 8 dims, from +0 (dot8);
//   - score: the xor 8, 4, 2, 1 tree over the 16 partials (tree16), then one RN product by
//     c = scale * log2(e) rounded to f32 on the host;
//   - tile: ATTN_TT positions aligned to absolute positions: tile max, carry, p = ex2(s - m),
//     ts summed in position order, d = d * carry + ts, acc chains in position order (tile_*);
//   - unit: ATTN_UNIT positions (tiles in order); split: SPLIT positions (units folded left to
//     right); row: splits folded left to right; the merge is st_merge;
//   - exp2 is ex2_fixed; the output is f2h(acc / d) with an IEEE division.
// Which units, tiles and splits a row visits depends on its length alone, so a row's bits do not
// depend on the number of rows in the call, the grid, or which CTA finishes last.
//
// MAPPING (bit-neutral, chosen on the bench, no T8):
//   - head-to-thread: all G heads on one 16-lane group (below), or one head per 16-lane group
//     (FlashInfer's bdy = G); the tree of each score is the same either way;
//   - tree16 schedule: allreduce (every lane ends with every score; the sketch's st_tile_allreduce)
//     or reduce-scatter (each lane ends with 8G/16 scores, same tree, p computed once per score and
//     broadcast through shared memory); the second issues about 40 % fewer instructions
//     (calc/attention/attention_v2.py, section 4, *estimated*);
//   - CTA per split (units of a split folded in the CTA) or CTA per unit (unit states folded in the
//     combine with two-level tickets: per split, then per head, in the same nested order);
//   - stage depth and size, CTAs per SM, whether k > 1 widens tiles to f32 once per CTA.
// Changing TT, UNIT, SPLIT, the tree, the tile order or the exp2 coefficients changes bits.

#include <stdint.h>
// Assembled after llvq_slot.cuh (u32) and matvec.cu (h2f, f2h), as tv_q4_h.cu is.

#define HD 128
#define LANES 16
#define DPL 8            // dims per lane = HD / LANES
#ifndef ATTN_TT
#define ATTN_TT 8        // positions per tile; V4a may time 4 and 8 before the freeze
#endif
#ifndef ATTN_UNIT
#define ATTN_UNIT 64     // positions per unit; V4a may time 32 and 64 before the freeze
#endif
#define SPLIT 256        // positions per split
#define KMAX 8           // rows per call: decode k = 1, verify k <= 8
#define THREADS 128
#define NEG_INF __int_as_float((int)0xff800000u)

// ---------------------------------------------------------------- exp2, exactly specified
// Expected domain x <= 0 (a score minus a running maximum). Coefficients are PLACEHOLDERS (Taylor,
// about 1e-5 relative error at f -> 1, far above the 3 ulp target): freeze them once, here and in
// the Rust replay, from a minimax fit of 2^f on [0, 1).
#define C1 0.693147182f
#define C2 0.240226507f
#define C3 0.0555041087f
#define C4 0.00961812911f
#define C5 0.00133335581f
#define C6 0.000154035304f
__device__ __forceinline__ float ex2_fixed(float x) {
    if (x != x) return x;                       // NaN propagates; masking it to 0 would hide a bug
    if (!(x > -126.0f)) return 0.0f;            // -inf and underflow flush to +0, on both sides
    const float fl = floorf(x);                 // exact
    const float f = __fsub_rn(x, fl);           // exact for |x| < 2^23, f in [0, 1); -0 gives +0
    float p = C6;
    p = __fmaf_rn(p, f, C5); p = __fmaf_rn(p, f, C4); p = __fmaf_rn(p, f, C3);
    p = __fmaf_rn(p, f, C2); p = __fmaf_rn(p, f, C1); p = __fmaf_rn(p, f, 1.0f);
    const int e = (int)fl;                      // in [-126, 0]
    // Unsigned arithmetic: a left shift of a negative int is undefined in C++17, and the host probe
    // compiles this text with clang++ -O2. Wrapping mod 2^32 subtracts |e| from the exponent field.
    return __int_as_float((int)((unsigned)__float_as_int(p) + ((unsigned)e << 23)));
}

// ---------------------------------------------------------------- one online-softmax state
template <int G> struct St { float m[G], d[G], a[G][DPL]; };

template <int G> __device__ __forceinline__ void st_empty(St<G>& s) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
        s.m[g] = NEG_INF; s.d[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) s.a[g][i] = 0.0f;
    }
}

// Merge b (later positions) into a (earlier positions). Emptiness depends on the row length only,
// so the same branches run in decode and in verify. Executed by the host probe.
template <int G> __device__ __forceinline__ void st_merge(St<G>& a, const St<G>& b) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (b.m[g] == NEG_INF) continue;                       // empty right operand
        if (a.m[g] == NEG_INF) {                               // empty left operand: copy
            a.m[g] = b.m[g]; a.d[g] = b.d[g];
#pragma unroll
            for (int i = 0; i < DPL; ++i) a.a[g][i] = b.a[g][i];
            continue;
        }
        const float m = fmaxf(a.m[g], b.m[g]);                 // exact; a +-0 tie is harmless:
        const float fa = ex2_fixed(__fsub_rn(a.m[g], m));      // ex2(+0) = ex2(-0) = 1
        const float fb = ex2_fixed(__fsub_rn(b.m[g], m));
        a.d[g] = __fadd_rn(__fmul_rn(a.d[g], fa), __fmul_rn(b.d[g], fb));
#pragma unroll
        for (int i = 0; i < DPL; ++i)
            a.a[g][i] = __fadd_rn(__fmul_rn(a.a[g][i], fa), __fmul_rn(b.a[g][i], fb));
        a.m[g] = m;
    }
}

// ---------------------------------------------------------------- scores, shuffle-free parts
// One lane's partials for G heads at one position: an fma chain over its 8 dims from +0.
// q[g][i] is f32 widened from f16 (exact); krow8 points at this lane's 8 halves of the K row.
template <int G> __device__ __forceinline__ void dot8(const float (&q)[G][DPL],
                                                      const unsigned short* krow8, float (&part)[G]) {
    float kf[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) kf[i] = h2f(krow8[i]);
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) t = __fmaf_rn(q[g][i], kf[i], t);
        part[g] = t;
    }
}

// The canonical tree over the 16 lane partials of one score: offsets 8, 4, 2, 1, each level a
// pairwise RN sum. RN addition commutes, so every lane of a butterfly ends with these bits. This is
// the host mirror the probe executes; the device computes the same tree with shuffles below.
__device__ __forceinline__ float tree16(const float (&v)[LANES]) {
    float a[LANES];
#pragma unroll
    for (int l = 0; l < LANES; ++l) a[l] = v[l];
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        float b[LANES];
#pragma unroll
        for (int l = 0; l < LANES; ++l) b[l] = __fadd_rn(a[l], a[l ^ off]);
#pragma unroll
        for (int l = 0; l < LANES; ++l) a[l] = b[l];
    }
    return a[0];
}

// ---------------------------------------------------------------- one tile, shuffle-free parts
// The tile's new maximum and carry for head g. sc[j]: the tile's scores (tree16 x c); nv: valid
// positions (positions at or past the row's length L_r are not valid and never read).
__device__ __forceinline__ void tile_max_carry(float m_old, const float* sc, int nv,
                                               float* m_new, float* carry) {
    float mt = sc[0];
    for (int j = 1; j < nv; ++j) mt = fmaxf(mt, sc[j]);        // exact, order-free
    const bool fresh = (m_old == NEG_INF);
    *m_new = fresh ? mt : fmaxf(m_old, mt);
    *carry = fresh ? 0.0f : ex2_fixed(__fsub_rn(m_old, *m_new));
}

__device__ __forceinline__ float tile_p(float s, float m_new) {
    return ex2_fixed(__fsub_rn(s, m_new));
}

// d, ts and this lane's 8 accumulators for head g, given the tile's p values and carry. ts is a
// left fold in position order from +0. Vt: the tile's V rows (f16 bits), row stride HD.
// Positions j >= nv are skipped, never multiplied: a stale V row may hold inf or NaN.
__device__ __forceinline__ void tile_accumulate(float* d, float (&acc)[DPL], const float* p, int nv,
                                                float carry, const unsigned short* Vt, int lane16) {
    float ts = 0.0f;
    for (int j = 0; j < nv; ++j) ts = __fadd_rn(ts, p[j]);
    *d = __fadd_rn(__fmul_rn(*d, carry), ts);
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
        float a = __fmul_rn(acc[i], carry);
        for (int j = 0; j < nv; ++j) a = __fmaf_rn(p[j], h2f(Vt[j * HD + lane16 * DPL + i]), a);
        acc[i] = a;
    }
}

// ---------------------------------------------------------------- device-only schedules
// The host probe COMPILES these and the kernel below with compile-only stubs (cuhcheck's SUPPLEMENT
// pattern: a 4-argument __shfl_xor_sync that returns its argument) and never EXECUTES them. Only
// inline PTX (cp.async) sits under #ifndef LLVQ_HOST_BUILD.
// Schedule A, allreduce: every lane of the 16-lane group ends with tree16 of the partials.
__device__ __forceinline__ float bfly_allreduce16(float t, unsigned mask16) {
    t = __fadd_rn(t, __shfl_xor_sync(mask16, t, 8, 16));
    t = __fadd_rn(t, __shfl_xor_sync(mask16, t, 4, 16));
    t = __fadd_rn(t, __shfl_xor_sync(mask16, t, 2, 16));
    t = __fadd_rn(t, __shfl_xor_sync(mask16, t, 1, 16));
    return t;
}

// Schedule B, reduce-scatter: the same tree, recursive halving. N values per lane at entry (the
// tile's TT x G partials, padded to a multiple of 16: G = 5 pads 40 to 48). At level `off` a lane
// keeps the half selected by (lane16 & off) and adds the partner's copy of that half, so a kept
// value at lane l is v_l + v_(l^off), the tree's own pair. After four levels a lane holds N/16
// complete scores, at indices fixed by its lane bits; the kernel scales them by c, computes their
// p with tile_p once, and broadcasts p and the per-head carries through shared memory.
template <int N> __device__ __forceinline__ void bfly_reduce_scatter16(float (&v)[N], int lane16,
                                                                     unsigned mask16) {
    int n = N;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        const int half = n / 2;
        const bool up = (lane16 & off) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = up ? v[i] : v[i + half];
            const float keep = up ? v[i + half] : v[i];
            v[i] = __fadd_rn(keep, __shfl_xor_sync(mask16, send, off, 16));
        }
        n = half;
    }
}

// ---------------------------------------------------------------- the int8 store, R1 semantics
// One (kv head, position) row: 128 codes (u8) plus 4 halves (s0, mn0, s1, mn1), group of 64.
// The value is kvq.rs's quantize_dequantize, bit for bit:
//   graded (s > 0): f2h((code * s) + mn), two RN roundings (candle runs broadcast_mul and
//                   broadcast_add as two kernels; its later `* graded + mn * ungraded` adds a
//                   signed zero, which leaves every value unchanged);
//   ungraded:       mn when mn != 0, and +0 when mn is +-0 (the round trip computes
//                   (q*s + mn) * 0 + mn * 1, and +0 + -0 is +0 in RN).
// The tile loader writes these values into the same shared tile the f16 path reads.
__device__ __forceinline__ unsigned short deq_r1(uint8_t code, unsigned short s16, unsigned short mn16) {
    const float s = h2f(s16), mn = h2f(mn16);
    if (!(s > 0.0f)) return (mn == 0.0f) ? (unsigned short)0u : mn16;
    return f2h(__fadd_rn(__fmul_rn((float)code, s), mn));
}

// Fresh-row quantizer, kvq.rs semantics, one lane's 4 of the group's 64 values. mn and mx come
// from an exact fminf / fmaxf butterfly (device only); they may differ from candle's reduction in
// the sign of a zero, which changes the stored mn16 bits and no dequantized value. Identity checks
// therefore compare dequantized values, never raw coefficient bytes. Executed by the host probe.
#define INV255 0.0039215688593685627f   // (1.0 / 255.0) rounded to f32, as candle's affine sees it
__device__ __forceinline__ void quant_codes(const float (&x)[4], float mn, float mx, uint8_t (&code)[4],
                                            unsigned short* s16, unsigned short* mn16) {
    *s16 = f2h(__fmul_rn(__fsub_rn(mx, mn), INV255));
    *mn16 = f2h(mn);                                       // exact: mn is an f16 value
    const float s = h2f(*s16), mnr = h2f(*mn16);
    const float safe = (s > 0.0f) ? s : 1.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float qf = roundf(__fdiv_rn(__fsub_rn(x[i], mnr), safe));   // roundf: half away from zero
        qf = fminf(fmaxf(qf, 0.0f), 255.0f);
        code[i] = (uint8_t)qf;
    }
}

// ---------------------------------------------------------------- the kernel, f16 store
// Grid: (gridDim.x, n_kv). gridDim.x is the host's ceil((pos0 + k) / SPLIT) in eager mode, or any
// larger frozen value (s_max = cap / SPLIT). EVERY launched CTA draws a ticket, early exits
// included, so the last arrival is ticket gridDim.x - 1 whatever the device length: a frozen grid
// stays correct. The pattern, its barrier and its volatile reads are planes_occ.cu's occ_sk, where
// an empty slice also draws its ticket. A graph-captured or persistent design would still need
// per-layer tickets or an epoch; neither is built or claimed here.
//
// Shared memory (G = 4, schedule A): staging 2 stages x 32 positions x 512 B = 32 KB (k = 1,
// 4 units x 8 positions a stage) or 3 x 8 x 512 B = 12 KB (k > 1); unit-state exchange
// 4 x G x (HD + 2) x 4 B = 8.3 KB; about 41 KB at k = 1, so 2 CTAs per SM in 102,400 B.
// Registers: estimated 130 (k = 1), 170 (k > 1), 200 at G = 5. The driver reports num_regs and
// local_bytes at load (llvq-cuda/src/gpu.rs, report); the bench prints them for every entry point.
template <int G>
__device__ void attn_split(const unsigned short* __restrict__ q,      // [k][n_heads][HD], post norm, RoPE
                           const unsigned short* __restrict__ k_new,  // [k][n_kv][HD], or null when the
                           const unsigned short* __restrict__ v_new,  //   prologue kernel wrote the rows
                           unsigned short* __restrict__ kc,           // cache K [n_kv][cap][HD]
                           unsigned short* __restrict__ vc,           // cache V [n_kv][cap][HD]
                           const uint32_t* __restrict__ pos_dev,      // cached length before the call
                           int k, int cap, int n_kv, float c,
                           float* __restrict__ part,                  // [n_kv][s_max][KMAX][G][HD + 2]
                           u32* __restrict__ ticket,                  // [n_kv], zero at rest
                           unsigned short* __restrict__ out,          // [k][n_heads][HD]
                           int s_max) {
    const int s = blockIdx.x, h = blockIdx.y;
    const int pos0 = (int)*(const volatile uint32_t*)pos_dev;
    const int lmax = pos0 + k;
    const bool active = s * SPLIT < lmax;

    if (active) {
        // 1. Fresh rows owned by this split, when k_new is not null: write them to the cache and
        //    stage them from k_new / v_new, so no CTA reads a position another CTA writes in this
        //    launch. When the operator's fused prologue (audit step 2) has already written them,
        //    this step is skipped and every position comes from the cache, written by an earlier
        //    launch on the same stream.
        // 2. Query registers: row r = group in the k > 1 path, row 0 in k = 1, f32 from f16.
        // 3. Stage tiles with cp.async (inline PTX, zero-filled past lmax). For each row r whose
        //    length L_r = pos0 + r + 1 reaches this split: per unit (ascending), per tile
        //    (ascending): dot8 per position, the tree16 schedule, x c, tile_max_carry, tile_p,
        //    tile_accumulate; st_merge of the unit into the split state.
        // 4. Write the split state of each such row and head to part[h][s][r][g].
    }
    // 5. Release, then one ticket per CTA (occ_sk): fence, barrier, atomic, barrier-plus-OR.
    __threadfence();
    __syncthreads();
    u32 tk = 0u;
    if (threadIdx.x == 0) tk = atomicAdd(ticket + h, 1u);
    const int last = __syncthreads_or(threadIdx.x == 0 && tk == gridDim.x - 1u);
    if (!last) return;
    __threadfence();                       // acquire: the loads below follow the ticket we saw
    // 6. For each row r: S_r = ceil(L_r / SPLIT) from pos_dev, on the device; fold splits
    //    0 .. S_r - 1 with st_merge, reading part through `const volatile float*` (no stale L1
    //    line); out = f2h(__fdiv_rn(acc, d)). Which CTA folds depends on timing; what it computes
    //    does not.
    if (threadIdx.x == 0) ticket[h] = 0u;  // self-reset; the next launch is ordered after this one
}

extern "C" __global__ void __launch_bounds__(THREADS, 2)
attn_dec_f16_g4(const unsigned short* q, const unsigned short* k_new, const unsigned short* v_new,
                unsigned short* kc, unsigned short* vc, const uint32_t* pos_dev, int k, int cap,
                int n_kv, float c, float* part, u32* ticket, unsigned short* out, int s_max) {
    attn_split<4>(q, k_new, v_new, kc, vc, pos_dev, k, cap, n_kv, c, part, ticket, out, s_max);
}

extern "C" __global__ void __launch_bounds__(THREADS, 2)
attn_dec_f16_g5(const unsigned short* q, const unsigned short* k_new, const unsigned short* v_new,
                unsigned short* kc, unsigned short* vc, const uint32_t* pos_dev, int k, int cap,
                int n_kv, float c, float* part, u32* ticket, unsigned short* out, int s_max) {
    attn_split<5>(q, k_new, v_new, kc, vc, pos_dev, k, cap, n_kv, c, part, ticket, out, s_max);
}

// The int8 entry points differ only in the staging: codes and coefficients are copied with
// cp.async into an int8 stage, a warp pass applies deq_r1 into the f16 tile, and steps 3-6 are
// attn_split's. Fresh rows are quantized by quant_codes before step 1, and the tile gets their
// deq_r1 values, never their raw f16, so a row reads back what the cache holds.
//
// The q and k norms, RoPE and the KV write belong to the operator's branch (audit step 2). The
// bit-keeping design for that kernel, its traps and its check are in attention.md section 3.4.
