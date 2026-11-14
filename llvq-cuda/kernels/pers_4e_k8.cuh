// pers_4e_k8.cuh: 4E's research arm v9k8 (docs/CERVEAU.md O13), the KV cache
// held as group-affine int8. Never part of a served module:
// pers.rs::p4e_e1_edits("v9k8") inserts this text at the head of pers_4e.cu
// in that arm's module only, and puts the two calls below in place of v9's
// append and attention; the served v9 module's text does not move.
//
// The cache (llvq-cuda/src/kv8_ref.rs, the reference): per KV head, one block
// of PERS_K8_BLK = 1,088 bytes for each 8 positions aligned to absolute
// positions, the 8 rows of 128 codes first (position j at 128 j), then the 8
// rows of scale and bias (position j at 1,024 + 8 j: scale 0, bias 0, scale 1,
// bias 1, binary16; group g holds dimensions 64 g .. 64 g + 63). 136 bytes a
// position and head against binary16's 256. The cache of `cap` positions
// (a multiple of 8) is [n_kv][cap / 8][1,088] bytes.
//
// One quantizer (pers_k8_quant_row) writes every position: the append of a
// decode step and the fill after a prefill call it on the same binary16 row
// and write the same bytes. Its operations are kvq's (llvq-llm/src/kvq.rs),
// in order, each an explicit round to nearest even: minimum and maximum
// (exact; a zero read as +0), scale = f2h((max - min) x 1/255), bias =
// f2h(min) (exact), code = 0 if the scale is 0, else t = (x - min) / scale,
// floor(t) + (t - floor(t) >= 0.5), at most 255.
//
// The value the attention reads: fma(scale, code, bias) in f32, scale and
// bias widened from binary16, not rounded back (kv8_ref::dequant). Past that
// widening the attention is section 10's, op for op: the same steps, units,
// splits, folds and tickets, so the bits of a row still depend on its length
// alone. A step stages 4 K blocks then 4 V blocks, 544 chunks of 16 bytes
// (8,704 bytes) where v9 stages 1,024 (16 KB), in the same ring of
// PERS_ATTN_RING stages, which keeps its size so 4E's shared layout does not
// move. A lane reads 8 bytes of codes and 4 of scale and bias a position
// where v9 reads 16 bytes of binary16.

#ifndef LLVQ_PERS_K8_CUH
#define LLVQ_PERS_K8_CUH

#if PERS_ATTN_RING < 2
#error "pers_4e_k8.cuh: the int8 attention is staged by cp.async (row form 6 or later)"
#endif

#define PERS_K8_CODES 1024u                                   // bytes of a block's codes
#define PERS_K8_BLK 1088u                                     // bytes of one block of 8 positions
#define PERS_K8_BLK_U4 68u                                    // its 16-byte chunks
#define PERS_K8_HALF_U4 (PERS_UPS * PERS_K8_BLK_U4)           // a step's K chunks: 272
#define PERS_K8_STAGE_U4 (2u * PERS_K8_HALF_U4)               // K then V: 544
#define PERS_K8_INV255 __uint_as_float(0x3b808081u)           // (float)(1.0 / 255.0), candle's divisor

static_assert(PERS_K8_STAGE_U4 <= PERS_STAGE_U4, "a k8 step fits v9's stage");
static_assert(PERS_K8_CODES == PERS_TT * PERS_HD, "a block holds one tile");

// The code of one value (kv8_ref::code).
__device__ __forceinline__ u32 pers_k8_code(float x, float mn, float s)
{
    const float t = __fdiv_rn(__fsub_rn(x, mn), s);
    const float fl = floorf(t);
    const u32 q = (u32)fl + (__fsub_rn(t, fl) >= 0.5f ? 1u : 0u);
    return q > 255u ? 255u : q;
}

// One row of 128 binary16 values (src, global) quantized into position j of
// its block (blk, global). One warp: lane l holds dimensions l, l + 32 (group
// 0) and l + 64, l + 96 (group 1). The minimum and maximum are exact, so the
// shuffle order cannot move them; every lane forms the same scale and bias.
__device__ __forceinline__ void pers_k8_quant_row(const unsigned short* src, unsigned char* blk, u32 j, u32 lane)
{
    float x[4];
#pragma unroll
    for (u32 k = 0; k < 4u; ++k) x[k] = h2f(__ldcg(src + lane + 32u * k));
    float mn0 = fminf(x[0], x[1]), mx0 = fmaxf(x[0], x[1]);
    float mn1 = fminf(x[2], x[3]), mx1 = fmaxf(x[2], x[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mn0 = fminf(mn0, __shfl_xor_sync(0xffffffffu, mn0, off));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
        mn1 = fminf(mn1, __shfl_xor_sync(0xffffffffu, mn1, off));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    // a zero read as +0, so the bytes do not depend on which zero the order kept
    if (mn0 == 0.0f) mn0 = 0.0f;
    if (mx0 == 0.0f) mx0 = 0.0f;
    if (mn1 == 0.0f) mn1 = 0.0f;
    if (mx1 == 0.0f) mx1 = 0.0f;
    const unsigned short s0 = f2h(__fmul_rn(__fsub_rn(mx0, mn0), PERS_K8_INV255)), b0 = f2h(mn0);
    const unsigned short s1 = f2h(__fmul_rn(__fsub_rn(mx1, mn1), PERS_K8_INV255)), b1 = f2h(mn1);
    const float sf0 = h2f(s0), sf1 = h2f(s1);
    unsigned char* row = blk + j * PERS_HD;
#pragma unroll
    for (u32 k = 0; k < 4u; ++k) {
        const float s = k < 2u ? sf0 : sf1, mn = k < 2u ? mn0 : mn1;
        row[lane + 32u * k] = (unsigned char)(s == 0.0f ? 0u : pers_k8_code(x[k], mn, s));
    }
    if (lane == 0u) {
        u32* sb = reinterpret_cast<u32*>(blk + PERS_K8_CODES + j * 8u);
        sb[0] = (u32)s0 | ((u32)b0 << 16);
        sb[1] = (u32)s1 | ((u32)b1 << 16);
    }
}

// The fill: positions [p_lo, p_hi) of a binary16 cache ([n_kv][cap16][128],
// written by the prefill) quantized into the int8 caches ([n_kv][cap8 / 8]
// blocks). One warp a (position, head, K or V); a block of whole warps.
extern "C" __global__ void pers_kv8_fill(const unsigned short* __restrict__ k16, const unsigned short* __restrict__ v16,
                                         unsigned char* __restrict__ k8, unsigned char* __restrict__ v8, u32 n_kv,
                                         u32 cap16, u32 cap8, u32 p_lo, u32 p_hi)
{
    const u32 lane = threadIdx.x & 31u;
    const u32 n = (p_hi - p_lo) * n_kv * 2u;
    const u32 stride = (gridDim.x * blockDim.x) >> 5;
    for (u32 w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < n; w += stride) {
        const u32 kv = w & 1u, r = w >> 1, h = r % n_kv, p = p_lo + r / n_kv;
        const unsigned short* src = (kv ? v16 : k16) + ((unsigned long long)h * cap16 + p) * PERS_HD;
        unsigned char* blk = (kv ? v8 : k8) + ((unsigned long long)h * (cap8 / PERS_TT) + p / PERS_TT) * PERS_K8_BLK;
        pers_k8_quant_row(src, blk, p % PERS_TT, lane);
    }
}

// Section 11's phase with the int8 append: q heads as v9; a k head normed and
// roped in place in k_raw (pers_head_norm_rope reads its four columns before
// it writes them, and writes only those), then quantized at `pos`; a v head
// quantized from v_raw. kc8, vc8: the int8 caches of `cap` positions.
template <int FORM>
__device__ __forceinline__ void pers_4e_norm_rope_append_k8(const unsigned short* q_raw, unsigned short* k_raw,
                                                            const unsigned short* v_raw, const unsigned short* q_alpha,
                                                            const unsigned short* k_alpha, const unsigned short* cosr,
                                                            const unsigned short* sinr, float eps, u32 n_heads,
                                                            u32 n_kv, u32 cap, u32 pos, unsigned short* q_out,
                                                            unsigned char* kc8, unsigned char* vc8)
{
    const u32 lane = threadIdx.x & 31u;
    const u32 nwarps = blockDim.x >> 5;
    const u32 total = gridDim.x * nwarps;
    const unsigned long long blk = (unsigned long long)(pos / PERS_TT) * PERS_K8_BLK;
    const unsigned long long hstride = (unsigned long long)(cap / PERS_TT) * PERS_K8_BLK;
    for (u32 w = blockIdx.x + gridDim.x * (threadIdx.x >> 5); w < n_heads + 2u * n_kv; w += total) {
        if (w < n_heads) {
            pers_head_norm_rope<FORM>(q_raw + w * 128u, q_alpha, cosr, sinr, eps, q_out + w * 128u, lane);
        } else if (w < n_heads + n_kv) {
            const u32 h = w - n_heads;
            unsigned short* kr = k_raw + h * 128u;
            pers_head_norm_rope<FORM>(kr, k_alpha, cosr, sinr, eps, kr, lane);
            __syncwarp();
            pers_k8_quant_row(kr, kc8 + h * hstride + blk, pos % PERS_TT, lane);
        } else {
            const u32 h = w - n_heads - n_kv;
            pers_k8_quant_row(v_raw + h * 128u, vc8 + h * hstride + blk, pos % PERS_TT, lane);
        }
    }
}

// This lane's 8 values of one position: 8 codes at `codes`, its group's
// scale and bias (one u32) at `sb`, each fma(scale, code, bias).
__device__ __forceinline__ void pers_k8_unpack8(const unsigned char* codes, const unsigned char* sb, float f[8])
{
    const u32* w = reinterpret_cast<const u32*>(codes);
    const u32 w0 = w[0], w1 = w[1];
    const u32 p = *reinterpret_cast<const u32*>(sb);
    const float s = h2f((unsigned short)(p & 0xffffu)), b = h2f((unsigned short)(p >> 16));
#pragma unroll
    for (u32 i = 0; i < 4u; ++i) {
        f[i] = __fmaf_rn(s, (float)((w0 >> (8u * i)) & 0xffu), b);
        f[4u + i] = __fmaf_rn(s, (float)((w1 >> (8u * i)) & 0xffu), b);
    }
}

// pers_attn_tile on one block of K and one of V staged in shared memory
// (kb, vb): the same operations, each position's values from
// pers_k8_unpack8 in place of pers_unpack8.
__device__ __forceinline__ void pers_attn_tile_k8(const float* qrow, float* row, bool own, const unsigned char* kb,
                                                  const unsigned char* vb, u32 nv, float c, u32 lane)
{
    const u32 l16 = lane & 15u, dl = l16 * 8u, gsb = PERS_K8_CODES + (l16 >> 3) * 4u;
    __syncwarp();  // a previous tile of this task (same step) wrote the row: its writes precede these reads
    float sc[PERS_TT];
    {
        float qf[8];
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) qf[i] = qrow[dl + i];
#pragma unroll
        for (u32 j = 0; j < PERS_TT; ++j) {
            sc[j] = 0.0f;
            if (j < nv) {
                float kf[8];
                pers_k8_unpack8(kb + j * PERS_HD + dl, kb + gsb + j * 8u, kf);
                float t = 0.0f;
#pragma unroll
                for (u32 i = 0; i < 8u; ++i) t = __fmaf_rn(qf[i], kf[i], t);
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off));
                sc[j] = __fmul_rn(t, c);
            }
        }
    }
    const float m = own ? row[PERS_HD] : PERS_NEG_INF, d = own ? row[PERS_HD + 1u] : 0.0f;
    float mt = sc[0];
#pragma unroll
    for (u32 j = 1; j < PERS_TT; ++j)
        if (j < nv) mt = fmaxf(mt, sc[j]);
    const bool fresh = m == PERS_NEG_INF;
    const float mn = fresh ? mt : fmaxf(m, mt);
    const float carry = fresh ? 0.0f : pers_ex2(__fsub_rn(m, mn));
    float ts = 0.0f;
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) {
        if (j < nv) {
            sc[j] = pers_ex2(__fsub_rn(sc[j], mn));
            ts = __fadd_rn(ts, sc[j]);
        }
    }
    const float dn = __fadd_rn(__fmul_rn(d, carry), ts);
    float a[8];
#pragma unroll
    for (u32 i = 0; i < 8u; ++i) a[i] = __fmul_rn(own ? row[dl + i] : 0.0f, carry);
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) {
        if (j < nv) {
            float vf[8];
            pers_k8_unpack8(vb + j * PERS_HD + dl, vb + gsb + j * 8u, vf);
#pragma unroll
            for (u32 i = 0; i < 8u; ++i) a[i] = __fmaf_rn(sc[j], vf[i], a[i]);
        }
    }
    __syncwarp();  // memory order: every lane's reads of the row precede lane 0's rewrite of m and d
    if (own) {
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];
        if (l16 == 0u) {
            row[PERS_HD] = mn;
            row[PERS_HD + 1u] = dn;
        }
    }
}

// pers_attn_issue for the int8 cache: step st's blocks into `stage`, K block
// of slot ts (tile t of unit u, ts = u x TPU + t) at chunk 68 ts, V at
// 272 + 68 ts. A block whose first position is past the row is zero-filled;
// a block holding the row's last position is copied whole, its positions
// past the row never read. Then one commit, empty past the last step.
__device__ __forceinline__ void pers_attn_issue_k8(uint4* stage, const unsigned char* kh, const unsigned char* vh,
                                                   u32 base, u32 st, u32 nsteps, u32 tpu, u32 lr)
{
    if (st < nsteps) {
        for (u32 idx = threadIdx.x; idx < PERS_K8_STAGE_U4; idx += blockDim.x) {
            const u32 rem = idx % PERS_K8_HALF_U4, ts = rem / PERS_K8_BLK_U4, ch = rem - ts * PERS_K8_BLK_U4;
            const u32 u = ts / tpu, t = ts - u * tpu;
            const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
            const unsigned char* src = idx < PERS_K8_HALF_U4 ? kh : vh;
            const bool valid = p0 < lr;
            pers_cp16z(stage + idx,
                       reinterpret_cast<const uint4*>(src + (unsigned long long)(valid ? p0 / PERS_TT : 0u) * PERS_K8_BLK) + ch,
                       valid);
        }
    }
    pers_cp_commit();
}

// pers_attention's form 6 on the int8 cache: kc8, vc8 [n_kv][cap / 8][1,088]
// bytes holding positions 0..l-1 (the new one appended); everything else as
// pers_attention, its arguments, scratch, tickets and shared layout.
template <u32 PF>
__device__ __forceinline__ void pers_attention_k8(const unsigned short* q, const unsigned char* kc8,
                                                  const unsigned char* vc8, u32 n_heads, u32 groups, u32 cap, u32 l,
                                                  float c, float* part, u32* ticket, unsigned short* out,
                                                  unsigned char* smem, float* dbg_state)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 n_kv = n_heads / groups;
    const u32 ns = (l + PERS_SPLIT - 1u) / PERS_SPLIT, s_stride = (cap + PERS_SPLIT - 1u) / PERS_SPLIT;
    const u32 nhp = (groups + 1u) / 2u, ntasks = PERS_UPS * nhp;
    uint4* stage = reinterpret_cast<uint4*>(smem);
    float* ust = reinterpret_cast<float*>(smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);  // [UPS][groups][PSTRIDE]
    float* qs = ust + PERS_UPS * groups * PERS_PSTRIDE;                 // [groups][128]
    u32* hdr = pers_hdr();
    for (u32 job = blockIdx.x; job < ns * n_kv; job += gridDim.x) {
        const u32 s = job / n_kv, h = job - s * n_kv;
        const u32 base = s * PERS_SPLIT;
        const u32 nu = (l - base + PERS_UNIT - 1u) / PERS_UNIT;
        const u32 tpu = nu >= 3u ? 1u : (nu == 2u ? 2u : 4u);
        const u32 tiles_raw = (l - base + PERS_TT - 1u) / PERS_TT;
        const u32 tiles = tiles_raw < PERS_UNIT / PERS_TT ? tiles_raw : PERS_UNIT / PERS_TT;
        const u32 nsteps = (tiles + tpu - 1u) / tpu;
        const unsigned char* kh = kc8 + (unsigned long long)h * (cap / PERS_TT) * PERS_K8_BLK;
        const unsigned char* vh = vc8 + (unsigned long long)h * (cap / PERS_TT) * PERS_K8_BLK;
        for (u32 st = 0; st < PERS_ATTN_RING; ++st)
            pers_attn_issue_k8(stage + st * PERS_STAGE_U4, kh, vh, base, st, nsteps, tpu, l);
        for (u32 i = threadIdx.x; i < PERS_UPS * groups * PERS_PSTRIDE; i += blockDim.x) {
            const u32 k = i % PERS_PSTRIDE;
            ust[i] = k == PERS_HD ? PERS_NEG_INF : 0.0f;
        }
        for (u32 i = threadIdx.x; i < groups * PERS_HD; i += blockDim.x) qs[i] = h2f(__ldcg(q + h * groups * PERS_HD + i));
        for (u32 st = 0; st < nsteps; ++st) {
            uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;
            pers_cp_wait_ring();  // this thread's copies of step st are done
            __syncthreads();      // everyone's: the step is staged, the states and q are written
            const unsigned char* const sb8 = reinterpret_cast<const unsigned char*>(sbuf);
            for (u32 task = warp; task < ntasks; task += nwarps) {
                const u32 u = task / nhp, hp = task - u * nhp;
                if (u * tpu >= PERS_UPS) continue;  // a unit this step does not stage (uniform)
                const u32 g = 2u * hp + (lane >> 4);
                const bool own = g < groups;
                const u32 ge = own ? g : groups - 1u;
                for (u32 t = 0; t < tpu; ++t) {
                    const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;
                    if (p0 >= l || (st * tpu + t) * PERS_TT >= PERS_UNIT) break;  // uniform over the warp: unit u
                    const u32 nv = l - p0 < PERS_TT ? l - p0 : PERS_TT;
                    const u32 ts = u * tpu + t;
                    pers_attn_tile_k8(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own,
                                      sb8 + ts * PERS_K8_BLK, sb8 + (PERS_UPS + ts) * PERS_K8_BLK, nv, c, lane);
                }
            }
            __syncthreads();  // the slot's readers are done: it takes step st + RING
            pers_attn_issue_k8(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);
        }
        pers_cp_wait_all();  // no copy may land in the stages once they are reused
        __syncthreads();  // every unit state is written
        // The units, folded into the split state in unit order.
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
                if (dbg_state) {
                    dbg_state[hq * PERS_PSTRIDE + k] = sa;
                    if (k == 0u) {
                        dbg_state[hq * PERS_PSTRIDE + PERS_HD] = sm;
                        dbg_state[hq * PERS_PSTRIDE + PERS_HD + 1u] = sd;
                    }
                }
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
                __threadfence();  // acquire: the states read below follow the ticket drawn
                float* cmd = reinterpret_cast<float*>(stage);  // [CH][groups][2]: m, d
                for (u32 b0 = 0; b0 < groups * PERS_HD; b0 += blockDim.x) {
                    const u32 i = b0 + threadIdx.x;
                    const bool act = i < groups * PERS_HD;
                    const u32 g = act ? i / PERS_HD : 0u, k = act ? i - g * PERS_HD : 0u;
                    float am = PERS_NEG_INF, ad = 0.0f, aa = 0.0f;
                    for (u32 c0 = 0; c0 < ns; c0 += PERS_ATTN_CH) {
                        __syncthreads();  // the previous chunk's readers are done
                        for (u32 t = threadIdx.x; t < 2u * PERS_ATTN_CH * groups; t += blockDim.x) {
                            const u32 j = t / (2u * groups), gg = (t >> 1) % groups;
                            float val = 0.0f;
                            if (c0 + j < ns)
                                val = __ldcg(part + (((unsigned long long)h * s_stride + c0 + j) * groups + gg) * PERS_PSTRIDE + PERS_HD + (t & 1u));
                            cmd[t] = val;
                        }
                        float ba[PERS_ATTN_CH];
#pragma unroll
                        for (u32 j = 0; j < PERS_ATTN_CH; ++j) {
                            ba[j] = 0.0f;
                            if (act && c0 + j < ns)
                                ba[j] = __ldcg(part + (((unsigned long long)h * s_stride + c0 + j) * groups + g) * PERS_PSTRIDE + k);
                        }
                        __syncthreads();  // the chunk's maxima and sums are in shared memory
                        if (act) {
#pragma unroll
                            for (u32 j = 0; j < PERS_ATTN_CH; ++j)
                                if (c0 + j < ns)
                                    pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[j]);
                        }
                    }
                    if (act) {
                        const u32 hq = h * groups + g;
                        out[hq * PERS_HD + k] = f2h(__fdiv_rn(aa, ad));
                        if (dbg_state) {
                            dbg_state[hq * PERS_PSTRIDE + k] = aa;
                            if (k == 0u) {
                                dbg_state[hq * PERS_PSTRIDE + PERS_HD] = am;
                                dbg_state[hq * PERS_PSTRIDE + PERS_HD + 1u] = ad;
                            }
                        }
                    }
                }
                if (threadIdx.x == 0u) ticket[h] = 0u;  // the next layer's draws follow a grid barrier
            }
        }
        __syncthreads();  // the next job restages the shared buffer and the header
    }
}

#endif  // LLVQ_PERS_K8_CUH
