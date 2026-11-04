// tv_tetra48_persall: A3's `tv_planes_persall` (planes_occ.cu) with
// `tetra48_dot` in place of `planes_dot` (step L1 of the persistent lead,
// persistent.md §5; prereg proofs/preregistration-pers-l1-*.md).
//
// The 144 sites of one fused bench round (72 q+k+v and gate+up groups, 72
// single o and down matrices of the bare Tetra file) in ONE launch: a
// card-sized grid (resident CTAs x SMs, occ::pers_grid) walks a work list of
// 8-row groups at normal occupancy, with no dependency between sites. Like
// A3's persall it is a BOUND, never a port: on the served path rotation,
// attention and norms sit between the sites.
//
// Invariants, as in planes_occ.cu: the stream words, `gscale + gs_off[row]`
// and the f32 tail of tv_tetra48_seg; one row written once by one thread with
// a plain store; tv_tetra48_seg's tile loop and lane order; the same source
// expressions (`acc += tetra48_dot(...)`, `tv += t * x`, `acc * rscale + tv`),
// so NVVM contracts them as it does there. A single matrix is a site whose
// `gs_off` is all zeros. Bit identity with tv_tetra48_seg is owed.
//
// Composition: in the persistent module (llvq_pers.cuh's unit), never in the
// bench unit, whose bytes the published numbers rest on. It needs TILE_BLOCKS,
// warp_sum, F1rTables, f1r_load and tetra48_dot.

#ifndef LLVQ_TETRA48_CUH
#include "llvq_tetra48.cuh"
#endif

// Site words, 8 x u64 laid out by occ.rs::tetra_site_words:
//   [0] words  [1] gscale  [2] gs_off  [3] rscale  [4] tail (f32)  [5] y
//   [6] nblocks | tail_w << 16 | row_stride_u32 << 32
//   [7] group0 | ngroups << 32
#define LLVQ_TOCC_SITE_WORDS 8u

extern "C" __global__ void tv_tetra48_persall(const unsigned long long* __restrict__ sites,
                                              u32 nsites,
                                              const u32* __restrict__ rows,
                                              const unsigned char* __restrict__ prefixes,
                                              const unsigned short* __restrict__ branches,
                                              const unsigned char* __restrict__ suffixes,
                                              const float* __restrict__ invnorm,
                                              const float* __restrict__ x,
                                              u32 total_groups)
{
    extern __shared__ float xs[];
    const u32 lane = threadIdx.x & 31u;
    const u32 wi = threadIdx.x >> 5;
    const u32 nw = blockDim.x >> 5;
    const F1rTables tab = {rows, prefixes, branches, suffixes};
    u32 si = 0u;
    for (u32 w = blockIdx.x; w < total_groups; w += gridDim.x) {
        // Sites ascend in group0 and w only grows: one forward scan, uniform
        // across the CTA, so the barriers below stay uniform too.
        while (si + 1u < nsites && w >= (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;
        const unsigned long long* d = sites + si * LLVQ_TOCC_SITE_WORDS;
        const u32* words = (const u32*)d[0];
        const float* gscale = (const float*)d[1];
        const u32* gs_off = (const u32*)d[2];
        const float* rscale = (const float*)d[3];
        const float* tail = (const float*)d[4];
        float* y = (float*)d[5];
        const u32 nblocks = (u32)(d[6] & 0xffffull);
        const u32 tail_w = (u32)((d[6] >> 16) & 0xffffull);
        const u32 stride = (u32)(d[6] >> 32);
        const u32 group0 = (u32)(d[7] & 0xffffffffull);
        const u32 row = (w - group0) * nw + wi;
        const u32* wrow = words + row * stride;
        const float* gs = gscale + gs_off[row];  // warp-uniform, as tv_tetra48_seg
        float acc = 0.0f;

        const u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
        for (u32 t = 0; t < ntiles; ++t) {
            const u32 jlo = t * TILE_BLOCKS;
            const u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
            const u32 n = (jhi - jlo) * LLVQ_DIM;
            __syncthreads();
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
            __syncthreads();
            for (u32 j = jlo + lane; j < jhi; j += 32u) {
                const float* xb = xs + (j - jlo) * LLVQ_DIM;
                u32 lo, hi16;
                f1r_load(wrow, j, lo, hi16);
                acc += tetra48_dot(lo, hi16, tab, xb, gs, invnorm);
            }
        }

        acc = warp_sum(acc);
        if (lane == 0) {
            float tv = 0.0f;
            const u32 tc0 = nblocks * LLVQ_DIM;
            for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
            y[row] = acc * rscale[row] + tv;  // tetra48_seg.cu, the same expression
        }
    }
}
