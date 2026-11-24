// moe_tetra: the routed experts of one decoded token of a mixture-of-experts
// layer, in four launches (plan item D3 of the Inkling 2 Mini throughput plan,
// CUDA twin of the Metal kernels of `fused_metal.rs`).
//
// Before these kernels the CUDA path took the experts one at a time: a
// rotation, a gate matvec, an up matvec, the activation, a rotation and a down
// matvec per routed expert, eight experts a call. Here one launch does a step
// for every routed expert:
//   a. `moe_rot`: the input rotation of the routed experts, one block an
//      expert, the body of `rot_apply_rows`;
//   b. `moe_tetra_gu`: gate and up of the routed experts, one warp a row, the
//      body of `tv_tetra48_h`;
//   c. `moe_rot` again, on the activation, into the basis of `down`;
//   d. `moe_tetra_down`: down of each routed expert, each row times its
//      routing weight and summed in ascending slot order.
// The activation between b and c stays the caller's.
//
// Exactness. The values are meant to be those of the one-expert path, bit for
// bit: the same helper bodies in the same order, each matvec row rounded to
// binary16 as `tv_tetra48_h` stores it and widened back as the host's cast
// did, and the weighted sum written with `__fmul_rn` and `__fadd_rn`, which
// NVRTC never contracts, starting from +0 as `index_add` into zeros did.
//
// The records are not copied: an expert's buffers stay where the one-expert
// path uploaded them, and `MoePtr` carries their device addresses.
//
// NVRTC has no filesystem: the host appends this file after llvq_slot.cuh,
// matvec.cu, llvq_rot.cuh, rotate.cu and the Tetra parts.

struct MoePtr {
    const u32* words[3];
    const float* rscale[3];
    const float* gscale[3];
    const unsigned short* tail[3];
    const u32* sign_in;
    const float* small_in;
    const u32* sign_down;
    const float* small_down;
};

// One block a routed expert. `which` 0 reads the expert's input rotation, 1
// its down rotation. `row_stride` 0 rotates the same input row for every
// expert. `sel`: the routed expert ids, ascending, then the weights' bits.
// The host owes it `grid.x == nsel` and `n * 4` bytes of shared memory.
extern "C" __global__ void moe_rot(const unsigned short* __restrict__ xin,
                                   const MoePtr* __restrict__ ptr,
                                   const u32* __restrict__ sel,
                                   float* __restrict__ xout,
                                   u32 n,
                                   u32 m,
                                   u32 k,
                                   float inv,
                                   u32 x_off,
                                   u32 row_stride,
                                   u32 which)
{
    extern __shared__ float s[];
    const u32 r = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 nthreads = blockDim.x;
    const MoePtr* o = ptr + sel[r];
    const u32* signbits = which == 0u ? o->sign_in : o->sign_down;
    const float* small = which == 0u ? o->small_in : o->small_down;

    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    __syncthreads();

    for (u32 len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        __syncthreads();
    }

    float* __restrict__ out = xout + (unsigned long long)r * (unsigned long long)n;
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}

// Gate (mat 0) and up (mat 1) of every routed expert. Warp g computes row
// `row` of matrix `mat` of slot `kk`, g = (mat * nsel + kk) * d_out + row,
// and stores it at y[g]: the output is [2, nsel, d_out] f32, each value the
// binary16 `tv_tetra48_h` stores, widened. d_out is a multiple of the rows of
// a block, so the warps of a block share one expert and one activation row.
// `xall`: the rotated rows, [nsel, d_in].
extern "C" __global__ void moe_tetra_gu(const MoePtr* __restrict__ ptr,
                                        const u32* __restrict__ sel,
                                        u32 row_stride_u32,
                                        const u32* __restrict__ rows,
                                        const unsigned char* __restrict__ prefixes,
                                        const unsigned short* __restrict__ branches,
                                        const unsigned char* __restrict__ suffixes,
                                        const float* __restrict__ invnorm,
                                        const float* __restrict__ xall,
                                        float* __restrict__ y,
                                        u32 nblocks,
                                        u32 tail_w,
                                        u32 d_out,
                                        u32 nsel)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 per_mat = nsel * d_out;
    u32 mat = g / per_mat;
    u32 kk = (g - mat * per_mat) / d_out;
    u32 row = g - mat * per_mat - kk * d_out;
    const MoePtr* o = ptr + sel[kk];
    const u32* wrow = o->words[mat] + row * row_stride_u32;
    const float* gscale = o->gscale[mat];
    const float* rscale = o->rscale[mat];
    const unsigned short* tail = o->tail[mat];
    const float* x = xall + (unsigned long long)kk * (nblocks * LLVQ_DIM + tail_w);
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot(lo, hi16, tab, xs + (j - jlo) * LLVQ_DIM, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[g] = rot_h2f(f2h(acc * rscale[row] + tv));
    }
}

// Down (mat 2) of every routed expert, in ascending slot order: each row,
// rounded to binary16 as `tv_tetra48_h` stores it, times its routing weight
// (the f32 bits at sel[nsel + kk]) and added to the running row from +0. One
// warp an output row; `nsel` is uniform, so every barrier is reached by every
// thread. `xall`: the rotated activations, [nsel, d_in]. `y`: [d_out] f32.
extern "C" __global__ void moe_tetra_down(const MoePtr* __restrict__ ptr,
                                          const u32* __restrict__ sel,
                                          u32 row_stride_u32,
                                          const u32* __restrict__ rows,
                                          const unsigned char* __restrict__ prefixes,
                                          const unsigned short* __restrict__ branches,
                                          const unsigned char* __restrict__ suffixes,
                                          const float* __restrict__ invnorm,
                                          const float* __restrict__ xall,
                                          float* __restrict__ y,
                                          u32 nblocks,
                                          u32 tail_w,
                                          u32 nsel)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    float out = 0.0f;

    for (u32 kk = 0; kk < nsel; ++kk) {
        const MoePtr* o = ptr + sel[kk];
        const u32* wrow = o->words[2] + row * row_stride_u32;
        const float* gscale = o->gscale[2];
        const float* rscale = o->rscale[2];
        const unsigned short* tail = o->tail[2];
        const float* x = xall + (unsigned long long)kk * (nblocks * LLVQ_DIM + tail_w);
        float acc = 0.0f;

        for (u32 t = 0; t < ntiles; ++t) {
            u32 jlo = t * TILE_BLOCKS;
            u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
            u32 n   = (jhi - jlo) * LLVQ_DIM;
            __syncthreads();
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
            __syncthreads();

            for (u32 j = jlo + lane; j < jhi; j += 32u) {
                u32 lo, hi16;
                f1r_load(wrow, j, lo, hi16);
                acc += tetra48_dot(lo, hi16, tab, xs + (j - jlo) * LLVQ_DIM, gscale, invnorm);
            }
        }

        acc = warp_sum(acc);
        if (lane == 0) {
            float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
            float yv = rot_h2f(f2h(acc * rscale[row] + tv));
            float w = __uint_as_float(sel[nsel + kk]);
            // +0 + p then the running sum, each a rounded f32 operation.
            out = __fadd_rn(out, __fmul_rn(yv, w));
        }
    }
    if (lane == 0) {
        y[row] = out;
    }
}

// The activation between launches b and c, in one launch: for the gate rows
// then the up rows of `gu` ([2, n] f32), softcap(gate) * sigmoid(gate) *
// softcap(up) as binary16, the type the down rotation reads. The steps are
// the reference's (lines 751-771), in its order; candle runs them as eleven
// launches a call. Not gated bit for bit against candle's CUDA operations:
// `tanhf` and `expf` here are NVRTC's.
extern "C" __global__ void moe_act(const float* __restrict__ gu,
                                   unsigned short* __restrict__ out,
                                   u32 n, float inv_cap_g, float cap_g, float inv_cap_u, float cap_u)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    float g = gu[i];
    float u = gu[n + i];
    float sg = __fmul_rn(tanhf(__fmul_rn(g, inv_cap_g)), cap_g);
    float s = 1.0f / (1.0f + expf(-g));
    float su = __fmul_rn(tanhf(__fmul_rn(u, inv_cap_u)), cap_u);
    out[i] = f2h(__fmul_rn(__fmul_rn(sg, s), su));
}

// A prompt's rows through one matrix kind of every expert, in one launch
// (the prefill twin of the launches above, 2026-11-24). Before it the rows of
// a prompt went one expert and one chunk at a time: 970,955 launches for a
// prompt of 2,000 tokens.
//
// `chunks`: three u32 a chunk, the expert, the first pair and the count of
// pairs (1..=TETRA48_ROWS); a chunk's pairs are consecutive rows of `xall`
// and share its expert. Warp g computes output row `row` of matrix
// `mat0 + mi` for chunk `c`, g = (mi * nchunks + c) * d_out + row, and
// stores y[(mi * npairs + pair) * d_out + row] as binary16: the output is
// [nmat, npairs, d_out]. d_out is a multiple of the rows of a block, so the
// warps of a block share one chunk and one staging.
//
// Exactness: the body is `tv_tetra48_rows_h`'s, which is `tv_tetra48_h` row
// by row whatever the chunking; the values are those of the path that
// launches `tv_tetra48_rows_h` an expert and a chunk at a time, bit for bit.
// The host owes it `TETRA48_ROWS * TETRA48_TILE * LLVQ_DIM * 4` bytes of
// shared memory.
extern "C" __global__ void moe_tetra_rows(const MoePtr* __restrict__ ptr,
                                          const u32* __restrict__ chunks,
                                          u32 row_stride_u32,
                                          const u32* __restrict__ rows,
                                          const unsigned char* __restrict__ prefixes,
                                          const unsigned short* __restrict__ branches,
                                          const unsigned char* __restrict__ suffixes,
                                          const float* __restrict__ invnorm,
                                          const float* __restrict__ xall,
                                          unsigned short* __restrict__ y,
                                          u32 nblocks,
                                          u32 tail_w,
                                          u32 d_out,
                                          u32 nchunks,
                                          u32 npairs,
                                          u32 mat0)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 per_mat = nchunks * d_out;
    u32 mi = g / per_mat;
    u32 c = (g - mi * per_mat) / d_out;
    u32 row = g - mi * per_mat - c * d_out;
    u32 mat = mat0 + mi;
    const MoePtr* o = ptr + chunks[3u * c];
    u32 p0 = chunks[3u * c + 1u];
    u32 n_rows = chunks[3u * c + 2u];
    u32 d_in = nblocks * LLVQ_DIM + tail_w;
    const u32* wrow = o->words[mat] + row * row_stride_u32;
    const float* gscale = o->gscale[mat];
    const float* rscale = o->rscale[mat];
    const unsigned short* tail = o->tail[mat];
    const float* x = xall + (unsigned long long)p0 * d_in;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[TETRA48_ROWS];
#pragma unroll
    for (u32 r = 0; r < TETRA48_ROWS; ++r) acc[r] = 0.0f;

    const u32 xs_stride = TETRA48_TILE * LLVQ_DIM;
    u32 ntiles = (nblocks + TETRA48_TILE - 1u) / TETRA48_TILE;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TETRA48_TILE;
        u32 jhi = jlo + TETRA48_TILE < nblocks ? jlo + TETRA48_TILE : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < TETRA48_ROWS; ++r) {
            // A row past `n_rows` reads row 0 again rather than another chunk.
            const float* __restrict__ xr = x + (unsigned long long)(r < n_rows ? r : 0u) * d_in;
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                xs[r * xs_stride + i] = xr[jlo * LLVQ_DIM + i];
            }
        }
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            tetra48_dot_rows<TETRA48_ROWS>(lo, hi16, tab,
                                           xs + (j - jlo) * LLVQ_DIM, xs_stride,
                                           gscale, invnorm, acc);
        }
    }

#pragma unroll
    for (u32 r = 0; r < TETRA48_ROWS; ++r) {
        float a = warp_sum(acc[r]);
        if (lane == 0 && r < n_rows) {
            const float* __restrict__ xr = x + (unsigned long long)r * d_in;
            float tv = tail_dot_h(tail, xr + nblocks * LLVQ_DIM, row, tail_w);
            y[((unsigned long long)mi * npairs + p0 + r) * d_out + row] = f2h(a * rscale[row] + tv);
        }
    }
}

// Each token's weighted rows of a prompt's down output, summed in one launch
// (2026-11-24). Before it the host ran a multiply, then K row gathers and K
// adds through candle.
//
// `y`: the binary16 output of `moe_tetra_rows` for down, [npairs, d]. `side`:
// 2 * ntok * k words. The first ntok * k give, for token t and rank r, the
// pair holding the token's r-th expert in ascending expert order, at
// side[t * k + r]. The next ntok * k give that pair's routing weight as f32
// bits, at the same place. One thread a (token, column); the output is
// [ntok, d] f32.
//
// Exactness. The host path widened `y` to f32 (exact), multiplied each row
// by its weight (one rounded f32 product), and added the products rank by
// rank into a row of +0. Here: the same widening (`rot_h2f`), the same
// product (`__fmul_rn`), the same sums in the same order from +0
// (`__fadd_rn`). NVRTC never contracts the two into an FMA. No barrier, so
// the early return is safe. The host owes it ntok * d + blockDim.x below 2^32.
extern "C" __global__ void moe_wsum(const unsigned short* __restrict__ y,
                                    const u32* __restrict__ side,
                                    float* __restrict__ out,
                                    u32 ntok,
                                    u32 d,
                                    u32 k)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntok * d) { return; }
    const u32 t = i / d;
    const u32 col = i - t * d;
    const u32* __restrict__ pairs = side + t * k;
    const u32* __restrict__ wbits = side + ntok * k + t * k;
    float acc = 0.0f;
    for (u32 r = 0; r < k; ++r) {
        const u32 p = pairs[r];
        const float v = rot_h2f(y[(unsigned long long)p * d + col]);
        acc = __fadd_rn(acc, __fmul_rn(v, __uint_as_float(wbits[r])));
    }
    out[(unsigned long long)t * d + col] = acc;
}
