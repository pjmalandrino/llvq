// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §3.6), revised the same day after
// review. Not part of the repository.
//
// The second 32B wall: tv_q4_h / tv_q4_hh stage the whole activation in f32
// shared memory (llvq-llm/kernels/tv_q4_h.cu:125-127, 145-147), so an int4
// down_proj at d_in = 25,600 asks for 102,400 B (fused.rs:734-736) against
// the 101,376 B opt-in of sm_89 and sm_120. Two exact ways out, both keeping
// every FMA operand of tv_q4_hh:
//
//   tv_q4_hs  the activation staged as f16 BITS (2 bytes a column) and widened
//             at the point of use. h2f is exact and is the very conversion
//             tv_q4_hh applies while staging, so each FMA sees the same float.
//             51,200 B at 25,600: PAST the 49,152 B default, so this entry
//             point needs its OWN opt-in (func_dynamic_shared at its own size;
//             the journal note "sous le DÉFAUT" in
//             rot-partagee-14b-2026-08-17.txt:92 is wrong by 2,048 B).
//             One CTA of 8 warps per SM at 25,600, as the 14B's int4 down at
//             17,408 today.
//   tv_q4_ht  f32 tiles of TQ columns (TQ a multiple of 256), each lane
//             walking its own residue class of words in increasing order, so
//             its FMA chain is the same chain; warp_sum unchanged. Any d_in,
//             and the tile sets the occupancy (TQ = 6,144 -> 24 KB -> 4 CTAs
//             per SM). Costs two barriers a tile.
//
// Both take x_off (elements), so a prefill row narrowed out of a block can be
// read where it lies instead of going through the f32 cast and tv_q4_h
// (fused_cuda.rs:941-951), which would hit the wall again at 25,600. An f32
// ACTIVATION (a non-f16 dtype, or the `+cast` arm of LLVQ_TRIM_AB) at an
// F16-staged record cannot use either as written: narrowing it to f16 would
// change the arithmetic. Such a record takes an f32-input copy of tv_q4_ht
// (same chain, `xt[i] = x32[x_off + c0 + i]`) or refuses that activation by
// name. The choice is the implementer's; silently narrowing is not.
//
// THE LOAD PATH MUST CHANGE TOO, or the 32B fails at load with these kernels
// present: today one size, max d_in * 4 over ALL int4 records, is posed on
// both tv_q4_h and tv_q4_hh (fused_cuda.rs:641-660), and upload_int4 stores
// and checks d_in * 4 per record (:2162-2173, :2198). See selection.rs,
// `int4_entry_bytes`: one size per entry point, and the staged size per record.
//
// Composition: a unit of its own, llvq_slot.cuh, matvec.cu (h2f, f2h,
// warp_sum), tv_q4_h.cu (q4_deq, LLVQ_Q4_GROUP), then this file, loaded only
// when some record is F16-staged or under LLVQ_INT4_WIDE_CHECK. Appending this
// file to the served unit would move its printed sha256; a unit of its own
// does not.
//
// The body below duplicates q4_staged_dot rather than templating it, so the
// served tv_q4_h / tv_q4_hh text (and its register allocation) does not move.
// The cost of duplication is drift, and what catches drift is a CARD check:
// the host pattern (llvq-llm/tests/host_tv_q4_h.cpp:15-19) compiles the
// kernel and executes only q4_deq and a hand-written mirror of the lane
// order, never the kernel itself. The card checks, bytes of every row:
//   * at every load with an F16-staged record: hs against hh on the first int4
//     record (both fit), and hs against ht on the F16-staged record;
//   * V2 (L40S), under LLVQ_INT4_WIDE_CHECK=1 at the bench door: synthetic
//     256 x 25,344 (101,376 B in f32, the widest where all three fit) hs, ht,
//     hh pairwise, and synthetic 256 x 25,600 hs against ht;
//   * V5 (A100, opt-in 166,912 B): hs against hh at 25,600 directly, the only
//     card in reach where the f32-staged kernel runs at the 32B width.

// tv_q4_hs: f16 staging.
extern "C" __global__ void tv_q4_hs(const u32* __restrict__ wq,
                                    const unsigned short* __restrict__ scales,
                                    const unsigned short* __restrict__ biases,
                                    const unsigned short* __restrict__ x,
                                    unsigned short* __restrict__ y,
                                    u32 d_in,
                                    u32 gpr,
                                    u32 x_off)
{
    extern __shared__ unsigned short xs16[];   // d_in halves
    for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs16[i] = x[x_off + i];
    __syncthreads();

    u32 lane = threadIdx.x & 31u;
    u32 row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 nwords = d_in >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 wi = lane; wi < nwords; wi += 32u) {
        u32 p = wq[w0 + wi];
        u32 c = wi << 3;
        u32 g = s0 + (c >> 7);
        u32 sb = scales[g];
        u32 bb = biases[g];
#pragma unroll
        for (u32 k = 0; k < 8u; ++k)
            acc = __fmaf_rn(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), h2f(xs16[c + k]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = f2h(acc);
}

// tv_q4_ht: f32 tiles. TQ is a compile-time constant of the unit (the host
// checks TQ % 256 == 0 and TQ * 4 <= the function's dynamic shared size).
#ifndef LLVQ_Q4_TQ
#define LLVQ_Q4_TQ 6144u
#endif
extern "C" __global__ void tv_q4_ht(const u32* __restrict__ wq,
                                    const unsigned short* __restrict__ scales,
                                    const unsigned short* __restrict__ biases,
                                    const unsigned short* __restrict__ x,
                                    unsigned short* __restrict__ y,
                                    u32 d_in,
                                    u32 gpr,
                                    u32 x_off)
{
    extern __shared__ float xt[];   // LLVQ_Q4_TQ floats
    u32 lane = threadIdx.x & 31u;
    u32 row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 nwords = d_in >> 3;
    u32 w0 = row * nwords;
    u32 s0 = row * gpr;
    float acc = 0.0f;
    for (u32 c0 = 0u; c0 < d_in; c0 += LLVQ_Q4_TQ) {
        u32 tc = (d_in - c0 < LLVQ_Q4_TQ) ? d_in - c0 : LLVQ_Q4_TQ;
        __syncthreads();   // the previous tile is fully consumed
        for (u32 i = threadIdx.x; i < tc; i += blockDim.x) xt[i] = h2f(x[x_off + c0 + i]);
        __syncthreads();
        // c0 is a multiple of 256 columns = 32 words, so this lane's first word
        // in the tile is w_lo + lane, and its words stay the residue class
        // lane mod 32 in increasing order: the same sequence as q4_staged_dot.
        u32 w_lo = c0 >> 3;
        u32 w_hi = (c0 + tc) >> 3;
        for (u32 wi = w_lo + lane; wi < w_hi; wi += 32u) {
            u32 p = wq[w0 + wi];
            u32 c = wi << 3;
            u32 g = s0 + (c >> 7);
            u32 sb = scales[g];
            u32 bb = biases[g];
#pragma unroll
            for (u32 k = 0; k < 8u; ++k)
                acc = __fmaf_rn(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), xt[c - c0 + k], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = f2h(acc);
}
// Note on the barriers of tv_q4_ht: every warp of the block runs the same
// number of tiles (d_in is uniform), so every thread reaches every barrier;
// the family's "no early return" rule (tv_q4_h.cu:47-49) still holds.
