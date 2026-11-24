// ink_attn: the dense products of one decoded attention row of Inkling 2
// Mini, reading the weights in BF16, the CUDA twin of the Metal kernels of
// `inkling/attn_dense.rs` (its header says what each launch does).
//
// The host (`attn_dense.rs`, one `Launch` for both backends) passes every
// buffer as a pointer in the order the Metal kernel binds them, then the words
// of the kernel's constant struct, one scalar parameter a word. A threadgroup
// is a block (MV_TG or FIN_TG threads), a simdgroup a warp.
//
// Exactness. Every output is class B against candle's operations on CUDA: the
// products sum in a warp order, and NVRTC may fuse a multiply into the add
// that follows it. The steps keep the reference's order.

typedef unsigned int u32;
typedef unsigned short u16;

#define NSG (MV_TG / 32u)
#define MAX_HD 256u
#define MAX_DREL 64u

__device__ __forceinline__ float wsum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bfv(const u16* p, u32 i) { return __uint_as_float(((u32)p[i]) << 16); }

// Eight BF16 weights (four words) against eight activations, added in index
// order. The words come as one 128-bit load: a row starts on a 16-byte
// boundary (its length is a multiple of 8 weights), and one wide load replaces
// four narrow ones, the lesson of `tv_q4_tile.cu`. The activations stay narrow
// loads: a view of a larger buffer may start anywhere. Same values, same order.
__device__ __forceinline__ float dot8(uint4 w, const float* x, float acc)
{
    acc = acc + bf_lo(w.x) * x[0];
    acc = acc + bf_hi(w.x) * x[1];
    acc = acc + bf_lo(w.y) * x[2];
    acc = acc + bf_hi(w.y) * x[3];
    acc = acc + bf_lo(w.z) * x[4];
    acc = acc + bf_hi(w.z) * x[5];
    acc = acc + bf_lo(w.w) * x[6];
    acc = acc + bf_hi(w.w) * x[7];
    return acc;
}

// Launch 1: out[r] = x(r) . w[r], one warp a row. Rows below q_rows read the
// query input, the next k_rows the key input, the rest the value input.
extern "C" __global__ void ad_products(
    const float* xq, const float* xk, const float* xv,
    const u32* w,
    float* out,
    u32 h, u32 rows, u32 q_rows, u32 k_rows)
{
    const u32 lane = threadIdx.x & 31u;
    const u32 r = blockIdx.x * NSG + (threadIdx.x >> 5);
    if (r >= rows) { return; }
    const float* x = r < q_rows ? xq : (r < q_rows + k_rows ? xk : xv);
    const u32 n8 = h / 8u;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (unsigned long long)r * n8 * 4u);
    float acc = 0.0f;
    for (u32 i = lane; i < n8; i += 32u) {
        acc = dot8(wr[i], x + 8u * i, acc);
    }
    acc = wsum(acc);
    if (lane == 0u) { out[r] = acc; }
}

// candle's CPU sum over a contiguous last dim (16 lane sums, a fixed fold,
// then the leftovers in order): the order the Metal kernel keeps.
__device__ __forceinline__ float cpu_sum(const float* x, u32 n)
{
    float s[16];
    for (u32 l = 0; l < 16u; l++) { s[l] = 0.0f; }
    const u32 np = n & ~15u;
    for (u32 i = 0; i < np; i += 16u) {
        for (u32 l = 0; l < 16u; l++) { s[l] = s[l] + x[i + l]; }
    }
    float v[4];
    for (u32 e = 0; e < 4u; e++) {
        float a = s[e] + s[4u + e];
        float b = s[8u + e] + s[12u + e];
        v[e] = a + b;
    }
    float lo = v[0] + v[1];
    float hi = v[2] + v[3];
    float b = lo + hi;
    for (u32 i = np; i < n; i++) { b = b + x[i]; }
    return b;
}

// The factor of layers::l2_clip: min(thr * recip(sqrt(sum x^2) + 1e-6), 1).
__device__ __forceinline__ float clip_factor(const float* sq, u32 n, float thr, float ceps)
{
    float s = cpu_sum(sq, n);
    float t = sqrtf(s) + ceps;
    float f = (1.0f / t) * thr;
    return fminf(f, 1.0f);
}

// One channel of layers::short_conv on the ring: taps oldest first, then x;
// the residual and the output scale. x replaces the oldest row.
__device__ __forceinline__ float conv_static(float* ring, const u16* wt, u32 wo, float x, u32 c, u32 C, u32 k,
                                             u32 pos, float scale)
{
    const u32 k1 = k - 1u;
    float acc = 0.0f;
    for (u32 t = 0; t < k1; t++) {
        float v = ring[((pos + t) % k1) * C + c];
        acc = acc + v * bfv(wt, wo + c * k + t);
    }
    acc = acc + x * bfv(wt, wo + c * k + k1);
    ring[pos * C + c] = x;
    float o = acc + x;
    return o * scale;
}

// One channel of layers::dynamic_conv: tap `lag` back is dw[lag] +
// static_w[lag, c]; lag 0 is x, lag l the l-th newest history row.
__device__ __forceinline__ float conv_dyn(float* ring, const u16* wt, u32 so, const float* dw, float x, u32 c,
                                          u32 C, u32 k, u32 pos, float scale)
{
    const u32 k1 = k - 1u;
    float acc = 0.0f;
    for (u32 lag = 0; lag < k; lag++) {
        float src = lag == 0u ? x : ring[((pos + k1 - lag) % k1) * C + c];
        float wv = dw[lag] + bfv(wt, so + lag * C + c);
        acc = acc + src * wv;
    }
    ring[pos * C + c] = x;
    float o = acc + x;
    return o * scale;
}

// Launch 2. Blocks [0, nh): query head g, its relative states and logits, its
// sink, its gate channels. [nh, nh + nkv): key head. Then value heads.
extern "C" __global__ void ad_finish(
    const float* P,
    const u16* small,
    float* ring_k,
    float* ring_v,
    float* out,
    u32 nh, u32 nkv, u32 hd, u32 dr, u32 ext, u32 taps, u32 dyn, u32 has_tau, u32 pos_k, u32 pos_v,
    u32 off_r, u32 off_s, u32 off_g, u32 off_k, u32 off_pk, u32 off_v, u32 off_pv,
    u32 so_rs, u32 so_rp, u32 so_kc, u32 so_vc,
    u32 oo_k, u32 oo_v, u32 oo_rel, u32 oo_s, u32 oo_g,
    float qk_thr, float rel_thr, float clip_eps, float eps, float inv_hd, float tau, float conv_scale)
{
    __shared__ float X[MAX_HD];
    __shared__ float SQ[MAX_HD];
    __shared__ float R[MAX_DREL];
    __shared__ float RS[MAX_DREL];
    __shared__ float FAC[2];
    const u32 g = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (g < nh) {
        const u32 h = g;
        for (u32 c = tid; c < hd; c += FIN_TG) {
            float v = P[h * hd + c] * bfv(small, h * hd + c);
            X[c] = v;
            SQ[c] = v * v;
        }
        for (u32 j = tid; j < dr; j += FIN_TG) {
            float v = P[off_r + h * dr + j] * bfv(small, so_rs + h * dr + j);
            R[j] = v;
            RS[j] = v * v;
        }
        __syncthreads();
        if (tid == 0u) {
            FAC[0] = clip_factor(SQ, hd, qk_thr, clip_eps);
            FAC[1] = clip_factor(RS, dr, rel_thr, clip_eps);
        }
        __syncthreads();
        for (u32 c = tid; c < hd; c += FIN_TG) {
            float q = X[c] * FAC[0];
            if (has_tau != 0u) { q = q * tau; }
            out[h * hd + c] = q;
        }
        for (u32 j = tid; j < dr; j += FIN_TG) {
            float r = R[j] * FAC[1];
            if (has_tau != 0u) { r = r * tau; }
            R[j] = r;
        }
        __syncthreads();
        for (u32 e = tid; e < ext; e += FIN_TG) {
            float acc = 0.0f;
            for (u32 j = 0; j < dr; j++) {
                acc = acc + R[j] * bfv(small, so_rp + j * ext + e);
            }
            out[oo_rel + h * ext + e] = acc;
        }
        if (tid == 0u) { out[oo_s + h] = P[off_s + h]; }
        for (u32 c = tid; c < hd; c += FIN_TG) {
            float v = P[off_g + h * hd + c];
            float z = expf(-v) + 1.0f;
            out[oo_g + h * hd + c] = 1.0f / z;
        }
        return;
    }
    const bool is_k = g < nh + nkv;
    const u32 hh = is_k ? g - nh : g - nh - nkv;
    const u32 C = nkv * hd;
    float* ring = is_k ? ring_k : ring_v;
    const u32 pos = is_k ? pos_k : pos_v;
    const u32 off = is_k ? off_k : off_v;
    const u32 offp = is_k ? off_pk : off_pv;
    const u32 so = is_k ? so_kc : so_vc;
    for (u32 c = tid; c < hd; c += FIN_TG) {
        const u32 ch = hh * hd + c;
        float x = P[off + ch];
        float y = dyn != 0u
            ? conv_dyn(ring, small, so, P + offp + hh * taps, x, ch, C, taps, pos, conv_scale)
            : conv_static(ring, small, so, x, ch, C, taps, pos, conv_scale);
        X[c] = y;
        SQ[c] = y * y;
    }
    __syncthreads();
    if (tid == 0u) {
        if (is_k) {
            FAC[0] = clip_factor(SQ, hd, qk_thr, clip_eps);
        } else {
            // layers::rms_unweighted: recip(sqrt(mean(x^2) + eps)).
            float ms = cpu_sum(SQ, hd) * inv_hd + eps;
            FAC[0] = 1.0f / sqrtf(ms);
        }
    }
    __syncthreads();
    const u32 oo = is_k ? oo_k : oo_v;
    for (u32 c = tid; c < hd; c += FIN_TG) { out[oo + hh * hd + c] = X[c] * FAC[0]; }
}

// stage_out: out[r] = (a * gate) . wo[r], one warp a row.
extern "C" __global__ void ad_out(
    const float* av,
    const float* gate,
    const u32* w,
    float* out,
    u32 din, u32 rows)
{
    const u32 lane = threadIdx.x & 31u;
    const u32 r = blockIdx.x * NSG + (threadIdx.x >> 5);
    if (r >= rows) { return; }
    const u32 n8 = din / 8u;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (unsigned long long)r * n8 * 4u);
    float acc = 0.0f;
    for (u32 i = lane; i < n8; i += 32u) {
        float x[8];
        for (u32 t = 0; t < 8u; t++) { x[t] = av[8u * i + t] * gate[8u * i + t]; }
        acc = dot8(wr[i], x, acc);
    }
    acc = wsum(acc);
    if (lane == 0u) { out[r] = acc; }
}

// A ring in logical order, oldest row first: what SlotCache holds.
extern "C" __global__ void ad_ring_state(
    const float* ring,
    float* out,
    u32 c, u32 taps, u32 pos, u32 unused)
{
    const u32 k1 = taps - 1u;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k1 * c) { return; }
    out[i] = ring[((pos + i / c) % k1) * c + i % c];
}

// BF16 elements [off, off + n) of an owned buffer widened to f32: exact.
extern "C" __global__ void ad_widen(
    const u16* src,
    float* out,
    u32 off, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    out[i] = __uint_as_float(((u32)src[off + i]) << 16);
}
