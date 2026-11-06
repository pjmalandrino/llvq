// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, section 5, step L1;
// prereg-persistent-0.md). Added after the adversarial review, which proposed
// it as the cheap first reading of U1 and U3 (about $0.3 and a few days of
// code, against two to three weeks for the full persistent arm set of L2).
//
// tv_tetra48_persall: A3's `tv_planes_persall` (planes_occ.cu:376-442) with
// `tetra48_dot` in place of `planes_dot`. The 144 fused sites of one bench
// round in ONE launch, a card-sized grid (resident CTAs x SMs, occ::pers_grid)
// walking a work list of 8-row groups, normal occupancy, no dependency between
// sites. Like A3's persall it is a BOUND, never a port: rotation, attention and
// norms sit between the sites on the served path (a3-occupation-banc §"le fait
// neuf"). What it adds to the record:
//   * the first on-card bit-identity reading of a re-gridded Tetra kernel
//     (U3, f32 family): every row against tv_tetra48_seg, which is itself
//     checked bit for bit against tv_f1r_v3g per part in section A4, once the
//     A4 stream is built from the Tetra file (seg_source_fix.rs);
//   * the first one-launch Tetra stream rate (U1's anchor at full occupancy),
//     against the fused 144-launch geometry in the same rounds: the A3 grammar,
//     where beating the 252-launch geometry would only re-measure the fusion
//     (preregistration-a2-a3-geometrie-2026-08-31.md:117-122).
//
// Invariants, as in planes_occ.cu: the same stream words, the same gscale /
// gs_off resolution as tv_tetra48_seg, one row written once by one thread with
// a plain store, the lane order of tv_tetra48_seg (lane, lane+32, ... straight
// across tiles), the same epilogue expression. Bit identity is owed.
//
// Composition: after tetra48_seg.cu in the bench unit (it needs TILE_BLOCKS,
// warp_sum, F1rTables, f1r_load, tetra48_dot, and occ_xs_index's convention
// is not used: the stride stays 24). Appended last, so no fragment of an arm
// that carries a published number moves.

// Site words, 8 x u64 laid out by occ.rs::tetra_site_words (to write):
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
    const F1rTables tab = { rows, prefixes, branches, suffixes };
    u32 si = 0u;
    for (u32 w = blockIdx.x; w < total_groups; w += gridDim.x) {
        // Sites ascend in group0, and w only grows: one forward scan.
        while (si + 1u < nsites
               && w >= (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull))
            ++si;
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
        const float* gs = gscale + gs_off[row];  // warp-uniform, as tv_tetra48_seg.cu:129
        float acc = 0.0f;

        // tv_tetra48_seg's tile loop, verbatim in its arithmetic: every site
        // of the 4B has nblocks > TILE_BLOCKS at the served tile 64, so every
        // group restages its tiles and the two barriers stay where they are.
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
            y[row] = acc * rscale[row] + tv;  // tetra48_seg.cu:161, the same expression
        }
    }
}

// Host side, to write in planesbench beside the A3 section (a sketch, not
// code): a new flag LLVQ_TETRA_OCC=1, refused unless LLVQ_SEG_TETRA=1 (its
// reference) and tetra48 is selected; the site table from the fused Tetra
// groups in the order section A4 builds them; grid = occ::pers_grid(groups,
// residency read from the card for this symbol); correctness first, every row
// of every site against tv_tetra48_seg, A3's check_site rule (owed equality;
// a miss with the f64 reference within 1e-5 is reported as a moved
// association and the arm is still timed; above, the arm is invalidated and
// never timed, the job goes on); then the three Tetra arms interleaved in the
// same 7 rounds, 2 discarded: tetra48 (252 launches), tetra48seg (144),
// tpersall (1), ratios formed round by round.
