// pers_4e_e1.cuh: arm mv of experiment E1 (docs/mesures/large-4e-budget-2026-11-11.txt,
// cut 1). Never part of a served module: pers.rs::p4e_e1_unit_text inserts it
// at the head of pers_segment.cu's text for the mv arm only, and the served
// v9 module's text does not move (its digest is pinned in pers.rs).
//
// In v9 every CTA asks L2 for the next o and gate/up rows during the
// attention phase (pers_4e.cu, prefetch_next). In mv no rows are prefetched
// there: each CTA asks L2 for its share of layer l + 1's q/k/v rows and o
// rows once its gate/up rows of layer l start (pers_segment_body, right after
// ln2), where the DRAM idles; gate/up rows are no longer prefetched. The
// shares are v9's own: the q | k and v slices of pers_4e_qkv_rows<true> and
// the o slice [H c / G, H (c + 1) / G).
//
// The (address, bytes) pairs of layer l + 1 are computed by one thread during
// the barrier before layer l (stage_layer's gap, which already reads layer l's
// struct from global memory) and kept in header u32 10..25, which nothing else
// in the 4E module uses (llvq_pers.cuh section 0 and section 7c name the used
// ones: 0, 1, 4..6, 8, 32..63). prefetch.global.L2 writes no register: no
// value of any phase can depend on these lines.
// e1-pf begin
#define PERS_HDR_E1 10u  // header u32 10..25: four (address, bytes) pairs, u64 each

__device__ __forceinline__ unsigned long long* pers_e1_slots()
{
    return reinterpret_cast<unsigned long long*>(pers_hdr() + PERS_HDR_E1);
}

// Layer l's q, k, v and o shares of this CTA into the slots, or empty
// ranges when l is past the last layer. One thread of warp 1 (thread 0
// arrives at the barrier and spins); the gap's closing barrier publishes them.
template <class Lay>
__device__ __forceinline__ void pers_e1_stage_next(const Lay* layers, u32 l, u32 n_layers, u32 H, u32 n_heads)
{
    if (threadIdx.x != (blockDim.x > 32u ? 32u : 0u)) return;
    unsigned long long* s = pers_e1_slots();
    for (u32 i = 0; i < 8u; ++i) s[i] = 0ull;
    if (l >= n_layers) return;
    const Lay& Y = layers[l];
    const auto& N = Y.qkv;
    const u32 G = gridDim.x, c0 = blockIdx.x;
    const u32 qk = N.q_dout + N.k_dout, lo = qk * c0 / G, hi = qk * (c0 + 1u) / G;
    if (lo < N.q_dout) {
        s[0] = reinterpret_cast<unsigned long long>(N.qw + (unsigned long long)lo * N.q_stride);
        s[1] = 4ull * ((hi < N.q_dout ? hi : N.q_dout) - lo) * N.q_stride;
    }
    if (hi > N.q_dout) {
        const u32 a = lo > N.q_dout ? lo - N.q_dout : 0u, b = hi - N.q_dout;
        s[2] = reinterpret_cast<unsigned long long>(N.kw + (unsigned long long)a * N.k_stride);
        s[3] = 4ull * (b - a) * N.k_stride;
    }
    const u32 vlo = N.v_dout * c0 / G, vhi = N.v_dout * (c0 + 1u) / G;
    s[4] = reinterpret_cast<unsigned long long>(N.vw + (unsigned long long)vlo * (N.d_in >> 3));
    s[5] = 4ull * (vhi - vlo) * (N.d_in >> 3);
    const u32 olo = H * c0 / G, ohi = H * (c0 + 1u) / G, dctx = n_heads * 128u;
    if (Y.ot != nullptr) {
        s[6] = reinterpret_cast<unsigned long long>(Y.ot->t.words + (unsigned long long)olo * Y.ot->t.stride_u32);
        s[7] = 4ull * (ohi - olo) * Y.ot->t.stride_u32;
    } else {
        s[6] = reinterpret_cast<unsigned long long>(Y.ow + (unsigned long long)olo * (dctx >> 3));
        s[7] = 4ull * (ohi - olo) * (dctx >> 3);
    }
}

// The staged ranges, asked of L2 by every thread of the CTA.
__device__ __forceinline__ void pers_e1_prefetch_staged()
{
    const unsigned long long* s = pers_e1_slots();
    for (u32 i = 0; i < 4u; ++i) pers_l2_prefetch(reinterpret_cast<const void*>(s[2u * i]), s[2u * i + 1u]);
}

// Layer 0's o share (its q/k/v share is v9's own prefetch before ln1).
template <class Lay>
__device__ __forceinline__ void pers_e1_prefetch_o(const Lay& Y, u32 H, u32 n_heads)
{
    const u32 G = gridDim.x, c0 = blockIdx.x;
    const u32 olo = H * c0 / G, ohi = H * (c0 + 1u) / G, dctx = n_heads * 128u;
    if (Y.ot != nullptr)
        pers_l2_prefetch(Y.ot->t.words + (unsigned long long)olo * Y.ot->t.stride_u32, 4ull * (ohi - olo) * Y.ot->t.stride_u32);
    else
        pers_l2_prefetch(Y.ow + (unsigned long long)olo * (dctx >> 3), 4ull * (ohi - olo) * (dctx >> 3));
}
// e1-pf end
