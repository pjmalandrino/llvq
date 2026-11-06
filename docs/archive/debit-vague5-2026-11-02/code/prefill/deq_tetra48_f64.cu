// DRAFT, never compiled or run. A sketch for the GEMM route of the prefill, arithmetic C-dense
// (wave5-deep/prefill.md §3.2), written read-only against HEAD 06e1602.
//
// deq_tetra48_f64_h: one Tetra48 record -> f16 [d_out, d_in] in the NATURAL basis, equal bit for
// bit to what the dense arm holds: llvq_artifact::decode_matrix (format.rs:1510-1532, f64 decode,
// f64 un-rotation, narrowed to f32) followed by candle's f32 -> f16 cast on the card (cast.cu,
// `out[i] = inp[i]`, round to nearest even). Reference outside Rust: llvq_tetra/dequant.py.
//
// Exactness rules (each one is a mutant the host replay must kill):
//   * every f64 op is an intrinsic: NVRTC compiles with --fmad=true, and Rust never contracts;
//   * sqrt and division correctly rounded (__dsqrt_rn, __ddiv_rn), as Rust's f64 ops are;
//   * the mix accumulates from 0.0 over t ascending, one rounded product then one rounded add;
//   * the WHT runs len = 1, 2, 4, ... then multiplies by s = 1 / sqrt(m) computed once;
//   * the narrowing is TWO roundings, f64 -> f32 (__double2float_rn) -> f16 (the unit's own `f2h`,
//     round to nearest even, matvec.cu:239), never a direct f64 -> f16 conversion (one rounding).
//
// Composition contract as for the served unit: llvq_slot.cuh, matvec.cu (h2f, f2h),
// llvq_f1rank.cuh, llvq_f1rank_v3.cuh, llvq_tetra48.cuh, then this file.

#ifndef DEQ_MAX_SHARED_D
// Rows whose f64 image fits the opt-in budget go through shared memory: 12,672 * 8 <= 101,376.
#define DEQ_MAX_SHARED_D 12672u
#endif

__device__ __forceinline__ double d_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double d_add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double d_sub(double a, double b) { return __dsub_rn(a, b); }

// One CTA (256 threads) per output row. `v` is the row's f64 image: shared memory when
// d_in <= DEQ_MAX_SHARED_D, else `gscratch + blockIdx.x * d_in` (an L2-resident slice; the host
// sizes it for the resident CTAs and launches in waves, or one row per CTA over d_out).
extern "C" __global__ void deq_tetra48_f64_h(const u32* __restrict__ words,
                                             u32 row_stride_u32,
                                             const u32* __restrict__ rows,
                                             const unsigned char* __restrict__ prefixes,
                                             const unsigned short* __restrict__ branches,
                                             const unsigned char* __restrict__ suffixes,
                                             const double* __restrict__ centroids,   // [2], f64, from the file
                                             const double* __restrict__ row_scales,  // [d_out], f64, from the file
                                             const float* __restrict__ tail,         // [d_out * tail_w], f32, from the file
                                             const double* __restrict__ signs,       // [d_in], +-1.0
                                             const double* __restrict__ small,       // [k * k], row-major, f64
                                             double* __restrict__ gscratch,
                                             unsigned short* __restrict__ w16,       // [d_out, d_in]
                                             u32 nblocks, u32 tail_w, u32 d_in,
                                             u32 m, u32 k)                           // d_in = m * k, m a power of two
{
    extern __shared__ double vs[];
    const u32 row = blockIdx.x;
    double* v = (d_in <= DEQ_MAX_SHARED_D) ? vs : gscratch + (u64)row * d_in;
    const u32* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    const double rs = row_scales[row];

    // 1. Decode in the rotated basis: reconstruct_shape_gain (quantizer.rs:962-976).
    //    picked = centroids[g] * row_scale; scale = picked / sqrt(16 m); out = p * scale.
    for (u32 j = threadIdx.x; j < nblocks; j += blockDim.x) {
        u32 lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        u32 q[6];
        f1r_v3_quads(lo, hi16, tab, q);
        u32 msh = tetra48_n2(q) >> 4;          // exact: n2 is a multiple of 16 on the lattice
        u32 g = (hi16 >> 15) & 1u;
        double* out = v + j * 24u;
        if (msh == 0u) {
            // The origin: Rust's `out.fill(0.0)`, i.e. +0.0 everywhere.
#pragma unroll
            for (u32 e = 0; e < 24u; ++e) out[e] = 0.0;
        } else {
            double picked = d_mul(centroids[g], rs);
            double scale = __ddiv_rn(picked, __dsqrt_rn((double)(16u * msh)));
#pragma unroll
            for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
                for (u32 jj = 0; jj < 4u; ++jj) {
                    // f1r_v3_float is an exact small integer; TETRA48_ORDER maps trio -> natural.
                    out[TETRA48_ORDER[4u * i + jj]] = d_mul((double)f1r_v3_float(q[i], jj), scale);
                }
            }
        }
    }
    // 2. The tail, stored f32 in the file and widened exactly.
    for (u32 t = threadIdx.x; t < tail_w; t += blockDim.x)
        v[nblocks * 24u + t] = (double)tail[(u64)row * tail_w + t];
    __syncthreads();

    // 3. Mix by Q^T across the k groups (rotation.rs:122-145, transpose = true):
    //    v[g*m + j] = sum_t small[t*k + g] * v[t*m + j], acc from 0.0, t ascending.
    //    One thread owns position j for all k groups, so the update is in place and race-free.
    if (k > 1u) {
        for (u32 j = threadIdx.x; j < m; j += blockDim.x) {
            double col[32];                    // k <= 25 up to the 32B's 25,600 = 1,024 * 25
            for (u32 t = 0; t < k; ++t) col[t] = v[t * m + j];
            for (u32 g = 0; g < k; ++g) {
                double acc = 0.0;
                for (u32 t = 0; t < k; ++t) acc = d_add(acc, d_mul(small[t * k + g], col[t]));
                v[g * m + j] = acc;
            }
        }
        __syncthreads();
    }

    // 4. WHT on each group of m (rotation.rs:192-212): len doubles from 1; each output is one
    //    add or one subtract of the same two values; then one global scale.
    for (u32 len = 1u; len < m; len <<= 1) {
        u32 half_pairs = (m >> 1) * k;
        for (u32 p = threadIdx.x; p < half_pairs; p += blockDim.x) {
            u32 g = p / (m >> 1);
            u32 qq = p % (m >> 1);
            u32 i = (qq / len) * (len << 1) + (qq % len);
            double* base = v + g * m;
            double a = base[i], b = base[i + len];
            base[i] = d_add(a, b);
            base[i + len] = d_sub(a, b);
        }
        __syncthreads();
    }
    const double s = __ddiv_rn(1.0, __dsqrt_rn((double)m));

    // 5. Scale, signs, narrow twice, store. The signs multiply is exact (+-1.0).
    for (u32 c = threadIdx.x; c < d_in; c += blockDim.x) {
        double x = d_mul(d_mul(v[c], s), signs[c]);
        w16[(u64)row * d_in + c] = f2h(__double2float_rn(x));   // f2h: matvec.cu:239, f32 -> f16, round to nearest even
    }
}

// The int4 records (v_proj, o_proj, down_proj 12-23 on the 4B) need no f64: the dense arm
// widens them with RawTensor::to_f32, `scale * q + bias` in f32, two roundings. q4_deq
// (tv_q4_h.cu:74-78) is that arithmetic; one more rounding to f16 makes the dense arm's tensor.
extern "C" __global__ void deq_q4_h(const u32* __restrict__ wq,
                                    const unsigned short* __restrict__ scales,
                                    const unsigned short* __restrict__ biases,
                                    unsigned short* __restrict__ w16,
                                    u32 d_in, u32 gpr, u32 d_out)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)d_out * d_in) return;    // no barrier in this kernel, so an early return is safe
    u32 row = (u32)(i / d_in), c = (u32)(i % d_in);
    u32 p = wq[(u64)row * (d_in >> 3) + (c >> 3)];
    u32 g = row * gpr + (c >> 7);
    float w = q4_deq((p >> (4u * (c & 7u))) & 0xfu, scales[g], biases[g]);
    w16[i] = f2h(w);
}

// Cost, estimated (prefill.md §3.2): 81 G FP64 instructions for a full 4B pass, 0.11 s at the
// 1/64 rate's peak, 0.15-0.23 s in practice; 0.5-0.9 s at 14B. If that passes 20 % of TTFT,
// the rotated-basis arithmetic C-rot16 replaces it (a third arithmetic, its own T8).
