// DRAFT, never compiled or run. A sketch for the prefill attention kernel d2 (wave5-deep/prefill.md
// §3.4), written read-only against HEAD 06e1602. Added in revision 2 after the adversarial review:
// folding in the same order is not enough for equal bits, because NVRTC compiles with
// --fmad=true and two separately compiled kernels may contract the fold, or pick another exp.
//
// The contract: ONE header, included by BOTH the prefill kernel (tv_attn_prefill_h) and the
// split-KV decode kernel of stage 3, holding every floating-point operation that touches a
// row's partials after its scores exist. Every multiply and add is an explicit round-to-nearest
// intrinsic, so no contraction can differ between the two compilation units; one exp
// implementation; one division. The per-tile scores and the per-tile PV products come from the
// same mma.sync.m16n8k16 sequence in both kernels (the decode kernel pads its row into a 16-row
// tile): that the tensor core accumulates an output element the same way whatever row of the
// tile it sits in is an ASSUMPTION, and the card test "prefill row = decode row" checks it.
//
// Tiles: 64 keys, aligned to ABSOLUTE positions (tile t holds keys [64t, 64t+64)). A row at
// absolute position p processes tiles t = 0 .. p/64, the same set in a prefill chunk of any size
// and in a decode step. Partials are folded in ascending t, sequentially, never as a tree of
// per-split folds: fold(fold(t0,t1),fold(t2,t3)) is not fold(t0,t1,t2,t3) in floating point.
//
// Host replay (clang++ -ffp-contract=off): the same functions over f32 inputs reproduce the device
// fold bit for bit given the per-tile partials the device printed. Mutants the replay and the card
// test must kill: the fold written as `l * a + lt * b` (contractible); `expf` or `__expf` in one
// kernel and `attn_exp` in the other; tiles aligned to the chunk start instead of position 0;
// a split-KV combine that folds per-split partials; `__frcp_rn` times O instead of `__fdiv_rn`.

#ifndef ATTN_FOLD_CUH
#define ATTN_FOLD_CUH

#define ATTN_TILE 64u
#define ATTN_LOG2E 1.4426950408889634f

// The only exp of the attention. exp2f without --use_fast_math is the accurate library routine,
// deterministic for a given input; the prescale is an explicit rounded multiply. exp2f(0) = 1
// and exp2f(-inf) = 0 exactly, which the fold relies on.
__device__ __forceinline__ float attn_exp(float x) {
    return exp2f(__fmul_rn(x, ATTN_LOG2E));
}

// The running state of one query row: max, normalizer, and its share of the output columns.
struct AttnRowState {
    float m;      // starts at -INFINITY
    float l;      // starts at 0
};

// Per-tile normalizer of one row: the lane's own fragment values summed in register order, then
// the quad reduced by a fixed xor tree (each row of an m16n8 C fragment lives in one quad, two
// values per lane per n8 tile, so 16 values per lane for a 64-key tile). Every row runs the same
// sequence whatever its position in the MMA tile. `p` holds attn_exp(s - m_t) in f32.
__device__ __forceinline__ float attn_tile_sum(const float p[16]) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc = __fadd_rn(acc, p[i]);
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    return acc;
}

// Fold one tile's partials (m_t, l_t, O_t) into the row state. `o` and `o_t` are this lane's
// output columns of the row (the same columns in both kernels: the PV fragment layout is the same
// mma shape). m_t is finite: a processed tile always holds the key at position 64t <= p.
__device__ __forceinline__ void attn_fold(AttnRowState& st, float* o, float m_t, float l_t,
                                          const float* o_t, int ncols) {
    const float m_new = fmaxf(st.m, m_t);                 // exact, order-free
    const float a = attn_exp(__fsub_rn(st.m, m_new));     // 0 on the first tile (m = -inf)
    const float b = attn_exp(__fsub_rn(m_t, m_new));
    st.l = __fadd_rn(__fmul_rn(st.l, a), __fmul_rn(l_t, b));
    for (int c = 0; c < ncols; ++c) o[c] = __fadd_rn(__fmul_rn(o[c], a), __fmul_rn(o_t[c], b));
    st.m = m_new;
}

// The row's output, one correctly rounded division per column, then f16 (round to nearest even,
// the unit's own f2h, matvec.cu:239).
__device__ __forceinline__ float attn_finish(float o, float l) {
    return __fdiv_rn(o, l);
}

#endif // ATTN_FOLD_CUH
