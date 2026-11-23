// ink_glue: the fused glue of one decoded row of an Inkling 2 Mini layer
// visit, the CUDA twin of the Metal kernels of `inkling/glue.rs` (its header
// says what each kernel does and which outputs are bit for bit).
//
// The host (`glue.rs`, one `Launch` for both backends) passes every buffer as
// a pointer in the order the Metal kernel binds them, then the words of the
// kernel's constant struct, one scalar parameter a word. A threadgroup of the
// Metal kernels is a block here (GLUE_TG threads, 32 for the one-warp kernels,
// 256 for the elementwise ones), a simdgroup is a warp.
//
// What is carried: the split forms only. The message gates run in
// `glue_gate_down`, `glue_gate_apply` and `glue_gate_norm`, the router's
// scores in `glue_router`; `glue_pre` and `glue_post_attn` refuse nothing but
// write garbage for `has_msg == 1`, which the host never sends on CUDA.
//
// Exactness. The convolutions and the residual adds are written with
// `__fmul_rn` and `__fadd_rn`, which NVRTC never contracts: candle's
// operations in candle's order. `glue_top8` only adds and divides. The norms,
// the block attention, the gate and router products sum in a block or warp
// tree, not in candle's order: class B, as on Metal.

typedef unsigned int u32;

#define NSG (GLUE_TG / 32u)
#define MAX_STATES 6u
#define MAX_H 4096u
#define MAX_EXPERTS 1024u

__device__ __forceinline__ float wsum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The sum of v over the block, every thread gets it. Fixed tree.
__device__ __forceinline__ float tg_sum(float v, float* scr, u32 lane, u32 sg)
{
    float s = wsum(v);
    if (lane == 0u) { scr[sg] = s; }
    __syncthreads();
    float t = lane < NSG ? scr[lane] : 0.0f;
    t = wsum(t);
    __syncthreads();
    return t;
}

// 1 / sqrt(mean(x^2) + eps), the factor of rms_unweighted.
__device__ __forceinline__ float inv_rms(const float* x, u32 n, float inv_n, float eps, float* scr, u32 tid, u32 lane, u32 sg)
{
    float acc = 0.0f;
    for (u32 c = tid; c < n; c += GLUE_TG) { float v = x[c]; acc = acc + v * v; }
    float ss = tg_sum(acc, scr, lane, sg);
    float ms = ss * inv_n;
    ms = ms + eps;
    return 1.0f / sqrtf(ms);
}

// Block residual attention for one query over n sources (block_attn_res).
__device__ __forceinline__ void bar_one(const float* q, const float* const* src, u32 n, u32 h, float inv_h,
                                        float eps, float scale, float* X, float* scr, u32 tid, u32 lane, u32 sg)
{
    float inv[MAX_STATES + 2];
    float w[MAX_STATES + 2];
    for (u32 i = 0; i < n; i++) { inv[i] = inv_rms(src[i], h, inv_h, eps, scr, tid, lane, sg); }
    for (u32 i = 0; i < n; i++) {
        float acc = 0.0f;
        for (u32 c = tid; c < h; c += GLUE_TG) { float s = src[i][c] * inv[i]; acc = acc + q[c] * s; }
        w[i] = tg_sum(acc, scr, lane, sg) * scale;
    }
    float m = w[0];
    for (u32 i = 1; i < n; i++) { m = fmaxf(m, w[i]); }
    float z = 0.0f;
    for (u32 i = 0; i < n; i++) { w[i] = expf(w[i] - m); z = z + w[i]; }
    for (u32 i = 0; i < n; i++) { w[i] = w[i] / z; }
    for (u32 c = tid; c < h; c += GLUE_TG) {
        float acc = 0.0f;
        for (u32 i = 0; i < n; i++) { float s = src[i][c] * inv[i]; acc = acc + w[i] * s; }
        X[c] = acc;
    }
    __syncthreads();
}

extern "C" __global__ void glue_pre(
    const float* hin,
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, const float* s5,
    const float* queries,
    const float* norm0, const float* norm1, const float* norm2,
    const float* msg0, const float* msg1, const float* msg2,
    const float* down0, const float* forget0, const float* gain0,
    const float* down1, const float* forget1, const float* gain1,
    const float* down2, const float* forget2, const float* gain2,
    float* out,
    u32 h, u32 n_states, u32 nq, u32 has_msg, u32 n_norms, u32 rank,
    float eps, float bar_eps, float inv_h, float bar_scale)
{
    __shared__ float X[MAX_H];
    __shared__ float scr[32];
    const u32 j = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5;
    if (nq == 0u) {
        for (u32 c = tid; c < h; c += GLUE_TG) { X[c] = hin[c]; }
        __syncthreads();
    } else {
        const float* src[MAX_STATES + 2];
        const float* st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
        for (u32 i = 0; i < n_states; i++) { src[i] = st[i]; }
        src[n_states] = hin;
        u32 qi = nq == 3u ? j : 0u;
        bar_one(queries + qi * h, src, n_states + 1u, h, inv_h, bar_eps, bar_scale, X, scr, tid, lane, sg);
    }
    if (has_msg != 0u) {
        // The gate runs in launches of its own: hand them X and the two factors it starts with.
        const float* m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
        float ix = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
        float im = inv_rms(m, h, inv_h, eps, scr, tid, lane, sg);
        for (u32 c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }
        if (tid == 0u) {
            out[3u * h + 2u * j] = ix;
            out[3u * h + 2u * j + 1u] = im;
        }
        return;
    }
    u32 ni = n_norms == 3u ? j : 0u;
    const float* w = ni == 0u ? norm0 : (ni == 1u ? norm1 : norm2);
    float inv = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
    for (u32 c = tid; c < h; c += GLUE_TG) { float v = X[c] * inv; out[j * h + c] = v * w[c]; }
}

// The message queries of a loop visit, one block a query.
extern "C" __global__ void glue_msg(
    const float* partial,
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, const float* s5,
    const float* queries,
    float* out,
    u32 h, u32 n_states, u32 nq, u32 has_msg, u32 n_norms, u32 rank,
    float eps, float bar_eps, float inv_h, float bar_scale)
{
    __shared__ float X[MAX_H];
    __shared__ float scr[32];
    const u32 j = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5;
    const float* src[MAX_STATES + 2];
    const float* st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
    for (u32 i = 0; i < n_states; i++) { src[i] = st[i]; }
    src[n_states] = partial;
    bar_one(queries + j * h, src, n_states + 1u, h, inv_h, bar_eps, bar_scale, X, scr, tid, lane, sg);
    for (u32 c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }
}

// One channel of short_conv on the ring: taps oldest first, then x; the
// residual and the output scale. The newest input replaces the oldest row.
// Every product and sum is a rounded f32 operation, as candle's are.
__device__ __forceinline__ float conv_ring(float* ring, const float* w, float x, u32 c, u32 h, u32 k, u32 pos, float scale)
{
    u32 k1 = k - 1u;
    float acc = 0.0f;
    for (u32 t = 0; t < k1; t++) {
        float v = ring[((pos + t) % k1) * h + c];
        acc = __fadd_rn(acc, __fmul_rn(v, w[c * k + t]));
    }
    acc = __fadd_rn(acc, __fmul_rn(x, w[c * k + k1]));
    ring[pos * h + c] = x;
    float o = __fadd_rn(acc, x);
    return __fmul_rn(o, scale);
}

// out: the residual [h], then the row before its gate [h] and (1 / rms(row),
// 1 / rms(message)) when the visit has a message, or the normed MLP input [h].
extern "C" __global__ void glue_post_attn(
    const float* attn,
    float* ring,
    const float* conv_w,
    const float* hin,
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, const float* s5,
    const float* mlp_q,
    const float* msg,
    const float* down, const float* forget, const float* gain,
    const float* norm_w,
    const float* router_w,
    float* out,
    u32 h, u32 n_states, u32 boundary, u32 has_msg, u32 rank, u32 experts, u32 k, u32 pos,
    float eps, float bar_eps, float inv_h, float bar_scale, float conv_scale)
{
    __shared__ float X[MAX_H];
    __shared__ float scr[32];
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5;
    for (u32 c = tid; c < h; c += GLUE_TG) {
        float v = conv_ring(ring, conv_w, attn[c], c, h, k, pos, conv_scale);
        out[c] = boundary != 0u ? v : __fadd_rn(hin[c], v);
    }
    __syncthreads();
    const float* src[MAX_STATES + 2];
    const float* st[MAX_STATES] = { s0, s1, s2, s3, s4, s5 };
    u32 n = 0;
    for (u32 i = 0; i < n_states; i++) { src[n++] = st[i]; }
    if (boundary != 0u) { src[n++] = hin; }
    src[n++] = out;
    bar_one(mlp_q, src, n, h, inv_h, bar_eps, bar_scale, X, scr, tid, lane, sg);
    if (has_msg != 0u) {
        float ix = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
        float im = inv_rms(msg, h, inv_h, eps, scr, tid, lane, sg);
        for (u32 c = tid; c < h; c += GLUE_TG) { out[h + c] = X[c]; }
        if (tid == 0u) {
            out[2u * h] = ix;
            out[2u * h + 1u] = im;
        }
        return;
    }
    float inv = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
    for (u32 c = tid; c < h; c += GLUE_TG) {
        float v = X[c] * inv;
        out[h + c] = v * norm_w[c];
    }
}

// MessageGate::forward spread over blocks, for n gates at once.
// xa: X [n, h], then (1 / rms(X), 1 / rms(m)) a gate, as stage A wrote them.
// Step 1: g = down [rms-normed X, rms-normed m], one warp a row. out: [n, rank].
extern "C" __global__ void glue_gate_down(
    const float* xa,
    const float* msg0, const float* msg1, const float* msg2,
    const float* down0, const float* down1, const float* down2,
    float* out,
    u32 h, u32 rank, u32 n, u32 n_norms, float eps, float inv_h, u32 per)
{
    const u32 s = blockIdx.x;
    const u32 lane = threadIdx.x;
    const u32 j = s / rank;
    const u32 r = s % rank;
    const float* X = xa + j * h;
    const float* m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
    const float* dw = j == 0u ? down0 : (j == 1u ? down1 : down2);
    const float ix = xa[n * h + 2u * j];
    const float im = xa[n * h + 2u * j + 1u];
    const float* row = dw + r * 2u * h;
    float acc = 0.0f;
    for (u32 c = lane; c < h; c += 32u) { float v = X[c] * ix; acc = acc + row[c] * v; }
    for (u32 c = lane; c < h; c += 32u) { float v = m[c] * im; acc = acc + row[h + c] * v; }
    acc = wsum(acc);
    if (lane == 0u) { out[s] = acc; }
}

// Step 2: (1 - tanh(F g)) * X + tanh(G g) * m, `per` channels a warp. out: [n, h].
extern "C" __global__ void glue_gate_apply(
    const float* xa,
    const float* g,
    const float* msg0, const float* msg1, const float* msg2,
    const float* forget0, const float* forget1, const float* forget2,
    const float* gain0, const float* gain1, const float* gain2,
    float* out,
    u32 h, u32 rank, u32 n, u32 n_norms, float eps, float inv_h, u32 per)
{
    const u32 s = blockIdx.x;
    const u32 lane = threadIdx.x;
    const u32 groups = (h + per - 1u) / per;
    const u32 j = s / groups;
    const u32 c0 = (s % groups) * per;
    const u32 c1 = c0 + per < h ? c0 + per : h;
    const float* m = j == 0u ? msg0 : (j == 1u ? msg1 : msg2);
    const float* fw = j == 0u ? forget0 : (j == 1u ? forget1 : forget2);
    const float* gw = j == 0u ? gain0 : (j == 1u ? gain1 : gain2);
    const float* G = g + j * rank;
    for (u32 c = c0; c < c1; c++) {
        float af = 0.0f;
        float ag = 0.0f;
        for (u32 r = lane; r < rank; r += 32u) {
            af = af + fw[c * rank + r] * G[r];
            ag = ag + gw[c * rank + r] * G[r];
        }
        af = wsum(af);
        ag = wsum(ag);
        if (lane == 0u) {
            float f = tanhf(af);
            float gg = tanhf(ag);
            float keep = 1.0f - f;
            float p = keep * xa[j * h + c];
            float q = gg * m[c];
            out[j * h + c] = p + q;
        }
    }
}

// Step 3: the weighted RMS norm, one block a gate. out: [n, h].
extern "C" __global__ void glue_gate_norm(
    const float* x,
    const float* norm0, const float* norm1, const float* norm2,
    float* out,
    u32 h, u32 rank, u32 n, u32 n_norms, float eps, float inv_h, u32 per)
{
    __shared__ float scr[32];
    const u32 j = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5;
    const float* X = x + j * h;
    u32 ni = n_norms == 3u ? j : 0u;
    const float* w = ni == 0u ? norm0 : (ni == 1u ? norm1 : norm2);
    float inv = inv_rms(X, h, inv_h, eps, scr, tid, lane, sg);
    for (u32 c = tid; c < h; c += GLUE_TG) { float v = X[c] * inv; out[j * h + c] = v * w[c]; }
}

// The router's scores, sigmoid(router_w . xn), one warp an expert.
extern "C" __global__ void glue_router(
    const float* xn,
    const float* router_w,
    float* out,
    u32 h, u32 experts)
{
    const u32 e = blockIdx.x;
    const u32 lane = threadIdx.x;
    const float* row = router_w + e * h;
    float acc = 0.0f;
    for (u32 c = lane; c < h; c += 32u) { acc = acc + row[c] * xn[c]; }
    acc = wsum(acc);
    if (lane == 0u) {
        float z = expf(-acc) + 1.0f;
        out[e] = 1.0f / z;
    }
}

// The total order of f32::total_cmp, as a signed key.
__device__ __forceinline__ int total_key(float v)
{
    int k = __float_as_int(v);
    k ^= (int)((u32)(k >> 31) >> 1);
    return k;
}

// out: sel [E] | weights [k] | margin | ids [k] as f32.
extern "C" __global__ void glue_top8(
    const float* scores,
    const float* bias,
    float* out,
    u32 e, u32 k, float scale)
{
    __shared__ float SEL[MAX_EXPERTS];
    __shared__ int KEY[MAX_EXPERTS];
    __shared__ u32 RANK[MAX_EXPERTS];
    const u32 tid = threadIdx.x;
    for (u32 i = tid; i < e; i += GLUE_TG) {
        float v = __fadd_rn(scores[i], bias[i]);
        SEL[i] = v;
        KEY[i] = total_key(v);
        out[i] = v;
    }
    __syncthreads();
    for (u32 i = tid; i < e; i += GLUE_TG) {
        int ki = KEY[i];
        u32 r = 0;
        for (u32 j = 0; j < e; j++) {
            int kj = KEY[j];
            r += (kj > ki || (kj == ki && j < i)) ? 1u : 0u;
        }
        RANK[i] = r;
    }
    __syncthreads();
    if (tid == 0u) {
        float sum = -0.0f;
        float last = 0.0f;
        float next = 0.0f;
        for (u32 i = 0; i < e; i++) {
            u32 r = RANK[i];
            if (r < k) { sum = __fadd_rn(sum, scores[i]); }
            if (r == k - 1u) { last = SEL[i]; }
            if (r == k) { next = SEL[i]; }
        }
        sum = __fadd_rn(sum, 1e-20f);
        u32 n = 0;
        for (u32 i = 0; i < e; i++) {
            if (RANK[i] < k) {
                float w = __fdiv_rn(scores[i], sum);
                out[e + n] = __fmul_rn(w, scale);
                out[e + k + 1u + n] = (float)i;
                n++;
            }
        }
        out[e + k] = last - next;
    }
}

// The routed experts as the MoE kernels read them: the ids [k], ascending,
// then the bits of their weights [k].
extern "C" __global__ void glue_sel(
    const float* top,
    u32* out,
    u32 e, u32 k, float scale)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) { return; }
    out[i] = (u32)top[e + k + 1u + i];
    out[k + i] = __float_as_uint(top[e + i]);
}

extern "C" __global__ void glue_post_mlp(
    const float* m,
    float* ring,
    const float* conv_w,
    const float* residual,
    float* out,
    u32 h, u32 k, u32 pos, float conv_scale)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= h) { return; }
    float v = conv_ring(ring, conv_w, m[c], c, h, k, pos, conv_scale);
    out[c] = __fadd_rn(residual[c], v);
}

// The ring in logical order, oldest row first: what SlotCache holds.
extern "C" __global__ void glue_ring_state(
    const float* ring,
    float* out,
    u32 h, u32 k, u32 pos, float conv_scale)
{
    const u32 k1 = k - 1u;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k1 * h) { return; }
    u32 t = i / h;
    u32 c = i % h;
    out[i] = ring[((pos + t) % k1) * h + c];
}
