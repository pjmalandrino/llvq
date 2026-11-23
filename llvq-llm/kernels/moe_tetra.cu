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
