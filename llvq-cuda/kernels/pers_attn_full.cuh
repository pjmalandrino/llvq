// pers_attn_full.cuh: the attention tile's full-tile path (docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt).
// Never part of a served module: pers.rs::pers_ft_apply inserts it before
// pers_attn_tile in llvq_pers.cuh's text, and adds at the head of
// pers_attn_tile a branch that takes it when nv = PERS_TT, for the ft arms
// only (tree a12ipft, a12ipgft, a12ipg32ft; 4E v9ft). The module texts of
// a12ip, a12ipg, a12ipg32 and 4E v9 do not move (pinned in pers.rs).
//
// pers_attn_tile guards each of its 8 positions with `if (j < nv)`, so the
// compiler keeps one position's chain (load, 8 dependent fma, 4 dependent
// shuffle-add) behind the previous one's. On a full tile no guard is needed.
// This body is pers_attn_tile at nv = PERS_TT with the guards removed and the
// 8 positions advanced together step by step: each score, each ex2 and each
// PV sum has the same operands in the same order, so the bits are the same.
// - score j: t_j = fma(q_i, k_ji, t_j) for i = 0..7 from 0, then
//   t_j += shfl_xor(t_j, off) for off = 8, 4, 2, 1, then t_j x c: as before;
// - the tile max, sc[0] then sc[1..7] in order: as before;
// - ex2 of each (pers_ex2_sel, the same value as pers_ex2), then ts summed
//   j = 0..7 in order: as before;
// - a_i = fma(p_j, v_ji, a_i) for j = 0..7 in order, from a_i x carry: as before.
// Only the order in which independent chains are issued differs.
//
// pers_ex2 returns early on a NaN and below -126, so each call is a branch
// region of its own and the 9 calls of a tile run one after the other.
// pers_ex2_sel computes the same value with selects: on x > -126 it runs
// pers_ex2's operations on x itself, in the same order; otherwise it runs them
// on 0 (no conversion of a NaN or an infinity to int) and returns x on a NaN,
// 0 below -126, as pers_ex2 does.
// ft begin
__device__ __forceinline__ float pers_ex2_sel(float x)
{
    const bool in = x > -126.0f;  // false on a NaN
    const float xs = in ? x : 0.0f;
    const float fl = floorf(xs);
    const float f = __fsub_rn(xs, fl);
    float p = __uint_as_float(0x39665cf6u);
    p = __fmaf_rn(p, f, __uint_as_float(0x3aa228dbu));
    p = __fmaf_rn(p, f, __uint_as_float(0x3c1eab8fu));
    p = __fmaf_rn(p, f, __uint_as_float(0x3d634139u));
    p = __fmaf_rn(p, f, __uint_as_float(0x3e75feb3u));
    p = __fmaf_rn(p, f, __uint_as_float(0x3f317217u));
    p = __fmaf_rn(p, f, 1.0f);
    const int e = (int)fl;
    const float r = __int_as_float((int)((unsigned)__float_as_int(p) + ((unsigned)e << 23)));
    return x != x ? x : (in ? r : 0.0f);
}

__device__ __forceinline__ void pers_attn_tile_full(const float* qrow, float* row, bool own, const uint4* ks,
                                                    const uint4* vs, float c, u32 lane)
{
    const u32 l16 = lane & 15u, dl = l16 * 8u;
    __syncwarp();  // a previous tile of this task (same step) wrote the row: its writes precede these reads
    float sc[PERS_TT];
    {
        float qf[8];
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) qf[i] = qrow[dl + i];
        uint4 kw[PERS_TT];
#pragma unroll
        for (u32 j = 0; j < PERS_TT; ++j) kw[j] = ks[j * 16u + l16];
        float t[PERS_TT];
#pragma unroll
        for (u32 j = 0; j < PERS_TT; ++j) t[j] = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) {
#pragma unroll
            for (u32 j = 0; j < PERS_TT; ++j) {
                float kf[8];
                pers_unpack8(kw[j], kf);
                t[j] = __fmaf_rn(qf[i], kf[i], t[j]);
            }
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
            for (u32 j = 0; j < PERS_TT; ++j) t[j] = __fadd_rn(t[j], __shfl_xor_sync(0xffffffffu, t[j], off));
        }
#pragma unroll
        for (u32 j = 0; j < PERS_TT; ++j) sc[j] = __fmul_rn(t[j], c);
    }
    const float m = own ? row[PERS_HD] : PERS_NEG_INF, d = own ? row[PERS_HD + 1u] : 0.0f;
    float mt = sc[0];
#pragma unroll
    for (u32 j = 1; j < PERS_TT; ++j) mt = fmaxf(mt, sc[j]);
    const bool fresh = m == PERS_NEG_INF;
    const float mn = fresh ? mt : fmaxf(m, mt);
    const float carry = fresh ? 0.0f : pers_ex2_sel(__fsub_rn(m, mn));
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) sc[j] = pers_ex2_sel(__fsub_rn(sc[j], mn));
    float ts = 0.0f;
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) ts = __fadd_rn(ts, sc[j]);
    const float dn = __fadd_rn(__fmul_rn(d, carry), ts);
    float a[8];
#pragma unroll
    for (u32 i = 0; i < 8u; ++i) a[i] = __fmul_rn(own ? row[dl + i] : 0.0f, carry);
#pragma unroll
    for (u32 j = 0; j < PERS_TT; ++j) {
        float vf[8];
        pers_unpack8(vs[j * 16u + l16], vf);
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) a[i] = __fmaf_rn(sc[j], vf[i], a[i]);
    }
    __syncwarp();  // memory order: every lane's reads of the row precede lane 0's rewrite of m and d
    if (own) {
#pragma unroll
        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];
        if (l16 == 0u) {
            row[PERS_HD] = mn;
            row[PERS_HD + 1u] = dn;
        }
    }
}
// ft end
