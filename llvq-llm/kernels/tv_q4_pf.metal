
// ===========================================================================
// tv_q4_pf.metal: prefetching twins of `tv_q4_metal` and `tv_q4_metal_tiled`.
//
// Never compiled alone. The host compiles it appended to `tv_q4_h.metal`
// (`SRC_Q4_PF` in `fused_metal.rs`), as a library of its own, so the served
// pipelines keep their bytes. The served unit's `fp contract(off)` covers
// this text, and the llvq-tetra wheel's checked copy of `tv_q4_h.metal` is
// untouched.
//
// The port of 4E's int4 row loop (`pers_q4_row<PREFETCH>`,
// `llvq-cuda/kernels/llvq_pers.cuh`): word wi + 32 and its scale and bias are
// loaded while word wi is used. The same words, scales and products,
// accumulated in the same order, so the same bits. Gated by
// `llvq-metal/tests/q4_tiled_matches_served.rs`, every f32 equal.
// ===========================================================================

/// One word's eight products into `acc`, in the served order.
inline float q4_word(float acc, uint p, ushort sb, ushort bb, const threadgroup float* xw)
{
    for (uint k = 0u; k < 8u; ++k) {
        acc = fma(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xw[k], acc);
    }
    return acc;
}

kernel void tv_q4_metal_p(const device uint*   wq     [[buffer(0)]],
                          const device ushort* scales [[buffer(1)]],
                          const device ushort* biases [[buffer(2)]],
                          const device float*  x      [[buffer(3)]],
                          device float*        y      [[buffer(4)]],
                          constant uint&       d_in   [[buffer(5)]],
                          constant uint&       gpr    [[buffer(6)]],
                          threadgroup float*   xs     [[threadgroup(0)]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint gid  [[thread_position_in_grid]],
                          uint tgs  [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    // The lane's first word and its pair, issued before the fill.
    uint p = 0u;
    ushort sb = 0, bb = 0;
    if (lane < nwords) {
        uint g = s0 + (lane << 3) / LLVQ_Q4_GROUP;
        p = wq[w0 + lane];
        sb = scales[g];
        bb = biases[g];
    }
    for (uint i = tid; i < d_in; i += tgs) {
        xs[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p2 = 0u;
        ushort sb2 = 0, bb2 = 0;
        if (wi + 32u < nwords) {
            uint g2 = s0 + ((wi + 32u) << 3) / LLVQ_Q4_GROUP;
            p2 = wq[w0 + wi + 32u];
            sb2 = scales[g2];
            bb2 = biases[g2];
        }
        acc = q4_word(acc, p, sb, bb, xs + (wi << 3));
        p = p2;
        sb = sb2;
        bb = bb2;
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        y[row] = acc;
    }
}

kernel void tv_q4_metal_tiled_p(const device uint*   wq        [[buffer(0)]],
                                const device ushort* scales    [[buffer(1)]],
                                const device ushort* biases    [[buffer(2)]],
                                const device float*  x         [[buffer(3)]],
                                device float*        y         [[buffer(4)]],
                                constant uint&       d_in      [[buffer(5)]],
                                constant uint&       gpr       [[buffer(6)]],
                                constant uint&       tile_cols [[buffer(7)]],
                                threadgroup float*   xs        [[threadgroup(0)]],
                                uint tid  [[thread_position_in_threadgroup]],
                                uint gid  [[thread_position_in_grid]],
                                uint tgs  [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    uint p = 0u;
    ushort sb = 0, bb = 0;
    if (lane < nwords) {
        uint g = s0 + (lane << 3) / LLVQ_Q4_GROUP;
        p = wq[w0 + lane];
        sb = scales[g];
        bb = biases[g];
    }
    float acc = 0.0f;

    for (uint c0 = 0u; c0 < d_in; c0 += tile_cols) {
        uint chi = min(c0 + tile_cols, d_in);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < chi - c0; i += tgs) {
            xs[i] = x[c0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // tile_cols is a multiple of 256, so the lane's words run on across
        // tiles: wi + 32 may be the first word of the next tile.
        for (uint wi = (c0 >> 3) + lane; wi < (chi >> 3); wi += 32u) {
            uint p2 = 0u;
            ushort sb2 = 0, bb2 = 0;
            if (wi + 32u < nwords) {
                uint g2 = s0 + ((wi + 32u) << 3) / LLVQ_Q4_GROUP;
                p2 = wq[w0 + wi + 32u];
                sb2 = scales[g2];
                bb2 = biases[g2];
            }
            acc = q4_word(acc, p, sb, bb, xs + ((wi << 3) - c0));
            p = p2;
            sb = sb2;
            bb = bb2;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        y[row] = acc;
    }
}

// ===========================================================================
// Half-io twins: `tv_q4_metal_h` and `tv_q4_metal_tiled_h`.
//
// The served int4 matvecs read an f32 activation and store an f32 result, so
// the model pays two candle launches a projection around them: a widening
// cast of its f16 activation, and a narrowing cast of the result. 84 int4
// records a token, 168 launches that carry no arithmetic.
//
// These read the f16 activation and widen it while staging, then narrow the
// result in the kernel. Both conversions are the ones the casts did:
// `float(half)` is exact, and `half(float)` rounds to nearest even, which is
// candle's `static_cast<half>` in `cast.metal`. Everything between is the
// served text, the same words in the same lane order. Gated by
// `llvq-llm/tests/q4_metal_half_io.rs` against the served kernel plus the two
// casts, every f16 equal.
// ===========================================================================

kernel void tv_q4_metal_h(const device uint*   wq     [[buffer(0)]],
                          const device ushort* scales [[buffer(1)]],
                          const device ushort* biases [[buffer(2)]],
                          const device half*   x      [[buffer(3)]],
                          device half*         y      [[buffer(4)]],
                          constant uint&       d_in   [[buffer(5)]],
                          constant uint&       gpr    [[buffer(6)]],
                          threadgroup float*   xs     [[threadgroup(0)]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint gid  [[thread_position_in_grid]],
                          uint tgs  [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    for (uint i = tid; i < d_in; i += tgs) {
        xs[i] = float(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 3;
        uint g = s0 + c / LLVQ_Q4_GROUP;
        ushort sb = scales[g];
        ushort bb = biases[g];
        for (uint k = 0u; k < 8u; ++k) {
            acc = fma(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        y[row] = half(acc);
    }
}

kernel void tv_q4_metal_tiled_h(const device uint*   wq        [[buffer(0)]],
                                const device ushort* scales    [[buffer(1)]],
                                const device ushort* biases    [[buffer(2)]],
                                const device half*   x         [[buffer(3)]],
                                device half*         y         [[buffer(4)]],
                                constant uint&       d_in      [[buffer(5)]],
                                constant uint&       gpr       [[buffer(6)]],
                                constant uint&       tile_cols [[buffer(7)]],
                                threadgroup float*   xs        [[threadgroup(0)]],
                                uint tid  [[thread_position_in_threadgroup]],
                                uint gid  [[thread_position_in_grid]],
                                uint tgs  [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;

    for (uint c0 = 0u; c0 < d_in; c0 += tile_cols) {
        uint chi = min(c0 + tile_cols, d_in);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < chi - c0; i += tgs) {
            xs[i] = float(x[c0 + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint wi = (c0 >> 3) + lane; wi < (chi >> 3); wi += 32u) {
            uint p = wq[w0 + wi];
            uint c = wi << 3;
            uint g = s0 + c / LLVQ_Q4_GROUP;
            ushort sb = scales[g];
            ushort bb = biases[g];
            for (uint k = 0u; k < 8u; ++k) {
                acc = fma(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c - c0 + k], acc);
            }
        }
    }
    acc = warp_sum(acc);
    if (lane == 0u) {
        y[row] = half(acc);
    }
}
