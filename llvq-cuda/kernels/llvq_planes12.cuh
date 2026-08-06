// The Planes12x block decoder — M2, the sparse-overlay layout: a 12-byte main
// stream every block of which decodes through at most 4 magnitude levels,
// plus a per-matrix exception table carrying the exact 14-byte Planes14
// record of every 5-level block.
//
// Main-stream record, from bit 0, LSB-first within a byte, bytes ascending —
// same convention as Planes14/Slot32:
//
//     [ class : 9 ][ gain : 1 ][ smask : 24 ][ plane0 : 24 ][ plane1 : 24 ][ 0 : 14 ]
//
// The level of slot j is `plane0[j] | plane1[j]<<1` — two planes, a 4-leaf
// tree. Blocks whose class has <= 4 levels are stored exactly; a 5-level
// block stores its best L <= 4 direction (the lswap swap, gain copied) and
// its exact record lives in the exception table. The row pass therefore
// computes an approximation on ~3.4 % of blocks, and the correction pass —
// same launch, extra CTAs — adds (exact - approx)·x·rscale back per
// exception, making y exact. The CPU reference is
// `Planes12xBlocks::decode_block` (exact) / `decode_approx_block` (row pass).
//
// ## The read window
//
// The byte offset 12·b is always a multiple of 4: the record is exactly the
// three aligned words `words[3b .. 3b+3]`, zero shift, no straddle. The +4
// byte tail padding the host appends is for symmetry with the other streams
// (and forgiving of a trailing partial word), not a correctness need here.
//
// ## No dynamic indexing
//
// As in llvq_planes.cuh: values come from a predicated tree over the plane
// bits, never `vals[idx]` — a dynamically indexed array is local memory on
// the hottest path (`local_size_bytes() == 0` is the detector).

#ifndef LLVQ_PLANES12_CUH
#define LLVQ_PLANES12_CUH

#ifndef LLVQ_PLANES_CUH
#include "llvq_planes.cuh"
#endif

#define LLVQ_PLANES12_STRIDE 12u

struct Planes12Fields {
    u32 id, gain, smask, p0, p1;
};

// The three-word aligned window, and every field off it. No variable shift
// anywhere: the offsets are constants of the layout.
__device__ __forceinline__ Planes12Fields planes12_fields(const u32* __restrict__ words,
                                                          u32 b)
{
    u32 w  = 3u * b;  // byte 12·b, always word-aligned
    u32 w0 = words[w], w1 = words[w + 1], w2 = words[w + 2];
    u64 lo = ((u64)w1 << 32) | (u64)w0;

    Planes12Fields f;
    f.id    = w0 & 0x1ffu;
    f.gain  = (w0 >> 9) & 1u;
    f.smask = (u32)(lo >> 10) & 0xffffffu;
    f.p0    = (u32)(lo >> 34) & 0xffffffu;
    f.p1    = (u32)((lo >> 58) | ((u64)w2 << 6)) & 0xffffffu;
    return f;
}

// One main-stream block's contribution to a dot product — planes_dot with the
// third plane gone: the value tree has exactly 4 leaves, matching the layout's
// level cap. Same four FMA chains, same accumulator selection, same final
// parenthesisation, same gscale epilogue as the other arms.
__device__ __forceinline__ float planes12_dot(const u32* __restrict__ words,
                                              const ClassRec* __restrict__ tab,
                                              const float* __restrict__ gscale,
                                              u32 b,
                                              const float* xb)
{
    Planes12Fields f = planes12_fields(words, b);
    const ClassRec r = tab[f.id];
    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2], v3 = r.vals[3];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (u32 i = 0; i < LLVQ_DIM; i += 4) {
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 j  = i + k;
            u32 bj = 1u << j;
            float v = (f.p1 & bj) ? ((f.p0 & bj) ? v3 : v2)
                                  : ((f.p0 & bj) ? v1 : v0);
            v = (f.smask & bj) ? -v : v;
            if (k == 0)      d0 = __fmaf_rn(v, xb[j], d0);
            else if (k == 1) d1 = __fmaf_rn(v, xb[j], d1);
            else if (k == 2) d2 = __fmaf_rn(v, xb[j], d2);
            else             d3 = __fmaf_rn(v, xb[j], d3);
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

// ---------------------------------------------------------------------------
// The correction pass's pieces — factored out of tv_planes12x so the host
// harness (tests/host_planes12.cpp) executes the *same text* the kernel runs,
// lane by lane; warp_sum and atomicAdd are the only parts it cannot.
// ---------------------------------------------------------------------------

// Where exception e lives: its matrix-wide block index, its output row, and
// its block-column within the row.
struct Planes12xExc {
    u32 b, row, col;
};

__device__ __forceinline__ Planes12xExc planes12x_locate(const u32* __restrict__ exc_idx,
                                                         u32 e,
                                                         u32 nblocks)
{
    Planes12xExc l;
    l.b   = exc_idx[e];
    l.row = l.b / nblocks;
    l.col = l.b - l.row * nblocks;
    return l;
}

// One lane's two products for exception e: the exact record's slot value
// (3-plane tree, from the exception stream) and the approximate slot value
// (2-plane tree, re-read from the main stream), each times x[lane]. Lanes
// 24..31 contribute zero. Signs applied here; gains and rscale are applied
// after the warp reduction, in planes12x_combine.
__device__ __forceinline__ void planes12x_lane_terms(const PlanesFields& fe,
                                                     const Planes12Fields& fa,
                                                     const ClassRec& re,
                                                     const ClassRec& ra,
                                                     u32 lane,
                                                     const float* __restrict__ xb,
                                                     float* ve,
                                                     float* va)
{
    *ve = 0.0f;
    *va = 0.0f;
    if (lane < LLVQ_DIM) {
        u32 bj = 1u << lane;
        float elo = (fe.p1 & bj) ? ((fe.p0 & bj) ? re.vals[3] : re.vals[2])
                                 : ((fe.p0 & bj) ? re.vals[1] : re.vals[0]);
        float ev  = (fe.p2 & bj) ? re.vals[4] : elo;
        ev = (fe.smask & bj) ? -ev : ev;
        float av = (fa.p1 & bj) ? ((fa.p0 & bj) ? ra.vals[3] : ra.vals[2])
                                : ((fa.p0 & bj) ? ra.vals[1] : ra.vals[0]);
        av = (fa.smask & bj) ? -av : av;
        float xv = xb[lane];
        *ve = ev * xv;
        *va = av * xv;
    }
}

// The value the correction adds to y[row]: exact contribution minus the
// approximate one the row pass already counted, each under its own gain
// centroid (equal by construction — the swap copies the gain — but read from
// its own record, so a transcoder that stopped copying would be *visible*
// here, not silently absorbed), times the row scale.
__device__ __forceinline__ float planes12x_combine(float ve,
                                                   float va,
                                                   const float* __restrict__ gscale,
                                                   u32 ge,
                                                   u32 ga,
                                                   float rs)
{
    return (ve * gscale[ge] - va * gscale[ga]) * rs;
}

#endif  // LLVQ_PLANES12_CUH
