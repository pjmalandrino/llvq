// DRAFT, never compiled or run. A sketch for the rows path 1a (wave5-deep/prefill.md §3.3),
// written read-only against HEAD 06e1602.
//
// tv_q4_rows_hh: `tv_q4_hh` (llvq-llm/kernels/tv_q4_h.cu:137-149) over Q4_ROWS activation rows,
// with the activation staged in column tiles so that the 14B's down_proj (d_in 17,408) fits.
//
// ## What must not change: each row's bits
//
// Per output row and per activation row, the accumulator sees exactly the FMA sequence of the
// one-row kernel:
//   * lane `l` visits words wi = l, l + 32, l + 64, ... in increasing order. A tile spans
//     Q4_TILE_COLS columns = Q4_TILE_COLS / 8 words, a multiple of 32, so each tile starts on a
//     word index that is a multiple of 32 and the lane's sub-sequence inside it is the global one;
//   * inside a word, k = 0..7, `acc = __fmaf_rn(q4_deq(...), x[c + k], acc)`; q4_deq is computed
//     once per weight (it is a pure function with explicit roundings) and reused for every row;
//   * the activation is widened from f16 while staged, exactly as tv_q4_hh does: the same floats;
//   * warp_sum per row, then one f2h, as before.
// So a row's output equals tv_q4_hh's on that row, bit for bit, by construction. The load-time
// check (the check_int4_half_parity pattern, fused_cuda.rs:2224) is what proves it on the card,
// because NVRTC compiles with --fmad=true and only the explicit intrinsics are pinned.
//
// ## Shared budget, f32 staging (computed, prefill.md §3.3)
//   default 49,152 B:  R = 4 -> T = 3,072;  R = 8 -> T = 1,536;  R = 16 -> T = 768
//   opt-in 101,376 B:  R = 4 -> T = 6,144;  R = 8 -> T = 3,072;  R = 16 -> T = 1,536
//   untiled, 8 rows:   311,296 B at d_in 9,728; 557,056 B at 17,408 (does not fit)
//
// ## Grid
// blockDim 256 (8 warps, one output row each, as every kernel of the family);
// gridDim.x = d_out / 8; gridDim.y = ceil(n_rows / Q4_ROWS). One launch per projection per
// prefill chunk, instead of one per row plus a cast for every row at an offset.
// No early return before a barrier; the host asserts d_out % 8 == 0 and d_in % 8 == 0.
//
// Composition contract: llvq_slot.cuh (u32), matvec.cu (h2f, f2h, warp_sum), tv_q4_h.cu
// (q4_deq, LLVQ_Q4_GROUP), then this file. Q4_ROWS and Q4_TILE_COLS are host-injected, and the
// host refuses a pair whose Q4_ROWS * Q4_TILE_COLS * 4 exceeds the budget it asks for.

#ifndef Q4_ROWS
#define Q4_ROWS 8u
#endif
#ifndef Q4_TILE_COLS
#define Q4_TILE_COLS 1536u
#endif

extern "C" __global__ void tv_q4_rows_hh(const u32* __restrict__ wq,
                                         const unsigned short* __restrict__ scales,
                                         const unsigned short* __restrict__ biases,
                                         const unsigned short* __restrict__ x,   // [n_rows, d_in] f16, row-major
                                         unsigned short* __restrict__ y,         // [n_rows, d_out] f16, written in place
                                         u32 d_in,
                                         u32 gpr,
                                         u32 n_rows,
                                         u32 d_out)
{
    extern __shared__ float xs[];                // Q4_ROWS * Q4_TILE_COLS floats
    const u32 lane = threadIdx.x & 31u;
    const u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // weight row = output column
    const u32 r0   = blockIdx.y * Q4_ROWS;                           // first activation row
    const u32 nwords = d_in >> 3;
    const u32 w0 = row * nwords;
    const u32 s0 = row * gpr;

    float acc[Q4_ROWS];
#pragma unroll
    for (u32 r = 0; r < Q4_ROWS; ++r) acc[r] = 0.0f;

    for (u32 c0 = 0; c0 < d_in; c0 += Q4_TILE_COLS) {
        const u32 nc = (d_in - c0 < Q4_TILE_COLS) ? d_in - c0 : Q4_TILE_COLS;
        __syncthreads();
#pragma unroll
        for (u32 r = 0; r < Q4_ROWS; ++r) {
            // A row past n_rows re-reads row r0: in bounds, computed, never stored.
            const u32 rr = (r0 + r < n_rows) ? r0 + r : r0;
            const unsigned short* xr = x + (u64)rr * d_in + c0;
            for (u32 i = threadIdx.x; i < nc; i += blockDim.x) xs[r * Q4_TILE_COLS + i] = h2f(xr[i]);
        }
        __syncthreads();

        const u32 wlo = c0 >> 3;                  // a multiple of 32, since Q4_TILE_COLS % 256 == 0
        const u32 whi = (c0 + nc) >> 3;
        for (u32 wi = wlo + lane; wi < whi; wi += 32u) {
            const u32 p = wq[w0 + wi];
            const u32 c = wi << 3;
            const u32 g = s0 + (c >> 7);          // eight columns of a word share one g128 group
            const u32 sb = scales[g];
            const u32 bb = biases[g];
#pragma unroll
            for (u32 kk = 0; kk < 8u; ++kk) {
                const float w = q4_deq((p >> (4u * kk)) & 0xfu, sb, bb);
                const u32 off = (c - c0) + kk;
#pragma unroll
                for (u32 r = 0; r < Q4_ROWS; ++r)
                    acc[r] = __fmaf_rn(w, xs[r * Q4_TILE_COLS + off], acc[r]);
            }
        }
    }

#pragma unroll
    for (u32 r = 0; r < Q4_ROWS; ++r) {
        const float a = warp_sum(acc[r]);         // every lane enters: warp_sum is a full-warp shuffle
        if (lane == 0u && r0 + r < n_rows) y[(u64)(r0 + r) * d_out + row] = f2h(a);
    }
}

// Note on shared-memory traffic (why 1a has a floor): each weight costs Q4_ROWS 4-byte shared
// reads; at 0.77 G int4 weights on the 4B that is 3.08 GB per prompt token, 0.067 ms at 45.8 TB/s.
// The Tetra rows kernel has the same structure at 2.86 G weights, 11.45 GB per token: the floor of
// the SIMT rows path, 0.32-0.57 ms per token (computed, prefill.md §3.3).
