// attn_row: the scores of one decoded attention row over the cache, in one
// launch, the CUDA twin of the Metal kernel of `inkling/attn_scores.rs` (its
// header says what it reads). One block a query head, ATTN_TG threads; the
// keys and the values are read in place through their strides. Class B
// against candle's operations: the sums run in a block tree.
//
// The host passes the buffers in the Metal kernel's order, then the words of
// its constant struct, one scalar parameter a word.

typedef unsigned int u32;

#define NSG (ATTN_TG / 32u)
#define MAX_HD 128u
#define COLS (MAX_HD / 32u)

__device__ __forceinline__ float wsum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float wmax(float v)
{
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

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

__device__ __forceinline__ float tg_max(float v, float* scr, u32 lane, u32 sg)
{
    float s = wmax(v);
    if (lane == 0u) { scr[sg] = s; }
    __syncthreads();
    float t = lane < NSG ? scr[lane] : -3.402823466e+38f;
    t = wmax(t);
    __syncthreads();
    return t;
}

// out: the result [nh, hd] | scratch [nh, nvis] (the scores, then their exponentials).
extern "C" __global__ void attn_row(
    const float* q,
    const float* rel,
    const float* sinks,
    const float* kb,
    const float* vb,
    float* out,
    u32 nh, u32 nkv, u32 hd, u32 extent,
    u32 pos, u32 lo, u32 nvis,
    u32 ks_h, u32 ks_p, u32 ks_d,
    u32 vs_h, u32 vs_p, u32 vs_d,
    float scale)
{
    __shared__ float Q[MAX_HD];
    __shared__ float SCR[NSG];
    __shared__ float P[NSG * MAX_HD];
    const u32 h = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5;
    const u32 g = h / (nh / nkv);
    for (u32 c = tid; c < hd; c += ATTN_TG) { Q[c] = q[h * hd + c]; }
    __syncthreads();

    float* E = out + nh * hd + h * nvis;
    const float* kh = kb + (unsigned long long)g * ks_h;
    const float* vh = vb + (unsigned long long)g * vs_h;
    const float* rh = rel + h * extent;

    // Scores: one position a thread, the dot in order over the head dimension.
    float mloc = -3.402823466e+38f;
    for (u32 i = tid; i < nvis; i += ATTN_TG) {
        u32 j = lo + i;
        const float* kp = kh + (unsigned long long)j * ks_p;
        float acc = 0.0f;
        for (u32 c = 0; c < hd; c++) { acc = __fmaf_rn(Q[c], kp[c * ks_d], acc); }
        float s = acc * scale;
        u32 d = pos - j;
        float b = d < extent ? rh[d] : 0.0f;
        s = s + b;
        E[i] = s;
        mloc = fmaxf(mloc, s);
    }
    const float sink = sinks[h];
    float m = tg_max(mloc, SCR, lane, sg);
    m = fmaxf(m, sink);

    // Exponentials and their sum, the sink's term last.
    float zloc = 0.0f;
    for (u32 i = tid; i < nvis; i += ATTN_TG) {
        float e = expf(E[i] - m);
        E[i] = e;
        zloc = zloc + e;
    }
    float z = tg_sum(zloc, SCR, lane, sg);
    z = z + expf(sink - m);

    // Weighted sum of the values: a warp a position, a lane four columns.
    float acc[COLS];
    for (u32 r = 0; r < COLS; r++) { acc[r] = 0.0f; }
    for (u32 i = sg; i < nvis; i += NSG) {
        float p = E[i] / z;
        const float* vp = vh + (unsigned long long)(lo + i) * vs_p;
        for (u32 r = 0; r < COLS; r++) {
            u32 c = lane + 32u * r;
            if (c < hd) { acc[r] = __fmaf_rn(p, vp[c * vs_d], acc[r]); }
        }
    }
    for (u32 r = 0; r < COLS; r++) {
        u32 c = lane + 32u * r;
        if (c < hd) { P[sg * MAX_HD + c] = acc[r]; }
    }
    __syncthreads();
    for (u32 c = tid; c < hd; c += ATTN_TG) {
        float t = 0.0f;
        for (u32 s = 0; s < NSG; s++) { t = t + P[s * MAX_HD + c]; }
        out[h * hd + c] = t;
    }
}
