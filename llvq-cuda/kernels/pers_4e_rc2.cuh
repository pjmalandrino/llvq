// pers_4e_rc2.cuh: lighter variants of the two research row bodies (pers_4e_rc.cuh), step R6b of the decode-speed
// research phase. The first card run (docs/mesures/recherche-debit-jeton-2026-11-13.txt) refused both bodies: under
// the driver's ptxas (580.178.04) 4E's timed token symbol held 128 registers and 8 B of local memory with either body,
// where v9 holds 128 and 0 B. Each variant sheds live values a different way, keeps what its body owes, and is a
// further E1 arm (pers.rs::p4e_rc_edits, bodies 3 to 6), so the card's load check (fusedrun LLVQ_PERS_4E=check:...)
// can say which one the driver's allocator takes.
//
//   body 3, ep2  (v9ep2)  endpass with NO block in flight in its first pass (pers_tetra_row<24, false, false>): two
//                         words a lane fewer live across the block loop. The same sum, term for term: the bits.
//   body 4, ep3  (v9ep3)  endpass in ONE chunk (the CTA's rows fit the 2,048 slots: no chunk loop, so neither its
//                         bounds nor the slot offset live across the row loop). The bits.
//   body 5, fl2  (v9fl2)  flat in ONE chunk (rows x runs <= 2,048): no chunk loop. The same association as flat.
//   body 6, fl3  (v9fl3)  fl2 with each run's three quads loaded as its blocks consume them (blocks 0-1 from quad 0,
//                         2-4 once quad 1 is in, 5-7 once quad 2 is): at most five words of the run live, not twelve.
//                         The same association as flat.
// Bodies 5 and 6 fold every run and every row in flat's order: their f32 rows equal flat's bit for bit, and so owe
// flat's bound of 1e-5 on rows. Bodies 4 to 6 trap if the CTA's slice does not fit one chunk; pers.rs::p4e_rc_layout
// refuses such a shape at install (the 4B on 142 CTAs fits; the 8B's flat gate/up does not).
//
// Never part of a served module. Inserted after pers_4e_rc.cuh's text in the modules of bodies 3 to 6 only; the
// arms' call sites name pers_rc2_rows and pers_rc2_work_floats where bodies 1 and 2 name pers_rc_rows and
// pers_rc_work_floats. tests/host_pers.cpp includes it on the persistent side (command 'e').
#ifndef PERS_4E_RC2_CUH
#define PERS_4E_RC2_CUH

// The CTA's slice must fit one chunk of slots.
__device__ __forceinline__ void pers_rc2_one_chunk(u32 needed)
{
    if (needed > PERS_RC_SLOTS) PERS_RC_TRAP();
}

// ep2: pers_rc_endpass with the first pass's row on its plain loop (no next block in flight).
template <class RowAt, class Epi>
__device__ __forceinline__ void pers_rc2_endpass_np(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                                    const float* xs, RowAt row_at, Epi epi, float* slots)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (u32 c0 = 0; c0 < n; c0 += PERS_RC_SLOTS) {
        const u32 c1 = n - c0 < PERS_RC_SLOTS ? n : c0 + PERS_RC_SLOTS;
        for (u32 i = c0 + warp; i < c1; i += nw) {
            const PersTRow R = row_at(i);
            const float a = pers_tetra_row<24u, false, false>(R.w, tab, R.gscale, invnorm, xs, nb, lane);
            if (lane == 0u) slots[i - c0] = a;
        }
        __syncthreads();
        for (u32 i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
            const PersTRow R = row_at(i);
            epi(i, pers_epilogue(slots[i - c0], R.rscale[R.r], R.tail + (unsigned long long)R.r * tw, xs + nb * 24u, tw));
        }
        __syncthreads();
    }
}

// ep3: endpass in one chunk, the next block in flight as v9.
template <class RowAt, class Epi>
__device__ __forceinline__ void pers_rc2_endpass_one(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                                     const float* xs, RowAt row_at, Epi epi, float* slots)
{
    pers_rc2_one_chunk(n);
    const u32 lane = threadIdx.x & 31u;
    for (u32 i = threadIdx.x >> 5; i < n; i += blockDim.x >> 5) {
        const PersTRow R = row_at(i);
        const float a = pers_tetra_row<24u, false, true>(R.w, tab, R.gscale, invnorm, xs, nb, lane);
        if (lane == 0u) slots[i] = a;
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        const PersTRow R = row_at(i);
        epi(i, pers_epilogue(slots[i], R.rscale[R.r], R.tail + (unsigned long long)R.r * tw, xs + nb * 24u, tw));
    }
    __syncthreads();
}

// One block of a run from words w0, w1, w2 at the block's word offset a = 3 i / 2: f1r_load's selection
// (rowc_run_block's, pers_rc_run_fold's), for an even block (lo = w0) or an odd one (lo straddles w0 and w1).
__device__ __forceinline__ float pers_rc2_even(float acc, u32 w0, u32 w1, u32 b, const F1rTables& tab, const float* xp,
                                               const float* gs, const float* invnorm)
{
    return pers_block(acc, w0, w1 & 0xffffu, tab, xp + pers_rc_xat(b), gs, invnorm);
}
__device__ __forceinline__ float pers_rc2_odd(float acc, u32 w0, u32 w1, u32 b, const F1rTables& tab, const float* xp,
                                              const float* gs, const float* invnorm)
{
    return pers_block(acc, (w0 >> 16) | (w1 << 16), w1 >> 16, tab, xp + pers_rc_xat(b), gs, invnorm);
}

// fl3's run fold: the same blocks, in the same order, from 0.0f, as pers_rc_run_fold; the three quads loaded as the
// blocks need them. Block i of the run reads words 3 i / 2 and the next: blocks 0, 1 words 0-2 (quad 0); block 2
// words 3-4; blocks 3, 4 words 4-7 (quad 1); block 5 words 7-8; blocks 6, 7 words 9-11 (quad 2). The empty asm with a
// memory clobber keeps the compiler from hoisting the next quad's load above the blocks before it. A row off 16 bytes
// takes pers_rc_run_words' u32 reader and pers_rc_run_fold.
__device__ __forceinline__ float pers_rc2_run_fold_seq(const u32* p, u32 g, u32 nq, u32 nwd, u32 nb, const F1rTables& tab,
                                                       const float* xp, const float* gs, const float* invnorm)
{
    if ((reinterpret_cast<unsigned long long>(p) & 15ull) != 0ull) {
        u32 w[12];
        pers_rc_run_words(p, g, nq, nwd, w);
        return pers_rc_run_fold(w, g, nb, tab, xp, gs, invnorm);
    }
    const uint4* q = reinterpret_cast<const uint4*>(p) + 3u * g;
    const u32 b0 = 8u * g;
    float acc = 0.0f;
    uint4 a = {0u, 0u, 0u, 0u};
    if (3u * g < nq) a = q[0];
    if (b0 < nb) acc = pers_rc2_even(acc, a.x, a.y, b0, tab, xp, gs, invnorm);
    if (b0 + 1u < nb) acc = pers_rc2_odd(acc, a.y, a.z, b0 + 1u, tab, xp, gs, invnorm);
    asm volatile("" ::: "memory");
    uint4 c = {0u, 0u, 0u, 0u};
    if (3u * g + 1u < nq) c = q[1];
    if (b0 + 2u < nb) acc = pers_rc2_even(acc, a.w, c.x, b0 + 2u, tab, xp, gs, invnorm);
    if (b0 + 3u < nb) acc = pers_rc2_odd(acc, c.x, c.y, b0 + 3u, tab, xp, gs, invnorm);
    if (b0 + 4u < nb) acc = pers_rc2_even(acc, c.z, c.w, b0 + 4u, tab, xp, gs, invnorm);
    asm volatile("" ::: "memory");
    uint4 e = {0u, 0u, 0u, 0u};
    if (3u * g + 2u < nq) e = q[2];
    if (b0 + 5u < nb) acc = pers_rc2_odd(acc, c.w, e.x, b0 + 5u, tab, xp, gs, invnorm);
    if (b0 + 6u < nb) acc = pers_rc2_even(acc, e.y, e.z, b0 + 6u, tab, xp, gs, invnorm);
    if (b0 + 7u < nb) acc = pers_rc2_odd(acc, e.z, e.w, b0 + 7u, tab, xp, gs, invnorm);
    return acc;
}

// fl2 (SEQ false) and fl3 (SEQ true): flat in one chunk. Slot u is written by the one thread that folds unit u and
// read after the barrier by the one thread of its row; the row's runs are summed in run order: flat's sum.
template <bool SEQ, class RowAt, class Epi>
__device__ __forceinline__ void pers_rc2_flat_one(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                                  const float* xp, RowAt row_at, Epi epi, float* slots)
{
    const u32 G = (nb + 7u) >> 3;
    const u32 nq = (6u * nb + 15u) >> 4, nwd = (6u * nb + 3u) >> 2;
    pers_rc2_one_chunk(n * G);
    for (u32 u = threadIdx.x; u < n * G; u += blockDim.x) {
        const u32 rr = u / G, g = u - rr * G;
        const PersTRow Rw = row_at(rr);
        if (SEQ) {
            slots[u] = pers_rc2_run_fold_seq(Rw.w, g, nq, nwd, nb, tab, xp, Rw.gscale, invnorm);
        } else {
            u32 w[12];
            pers_rc_run_words(Rw.w, g, nq, nwd, w);
            slots[u] = pers_rc_run_fold(w, g, nb, tab, xp, Rw.gscale, invnorm);
        }
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        float a = 0.0f;
        for (u32 g = 0; g < G; ++g) a = __fadd_rn(a, slots[i * G + g]);
        const PersTRow Rw = row_at(i);
        epi(i, pers_epilogue(a, Rw.rscale[Rw.r], Rw.tail + (unsigned long long)Rw.r * tw, xp + pers_rc_xat(nb), tw));
    }
    __syncthreads();
}

// Floats a variant uses from `work` on (pers_rc_work_floats' rule): flat's padded copy unless in place, the slots.
template <u32 ARM>
__device__ __forceinline__ u32 pers_rc2_work_floats(u32 nb, u32 tw, bool in_place)
{
    return (ARM >= 5u && !in_place ? pers_rc_padded_floats(nb, tw) : 0u) + PERS_RC_SLOTS;
}

// The row form of a variant's module: ARM 3 ep2, 4 ep3, 5 fl2, 6 fl3. Same contract as pers_rc_rows.
template <u32 ARM, class RowAt, class Epi>
__device__ __forceinline__ void pers_rc2_rows(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                              float* xs, RowAt row_at, Epi epi, float* work, bool in_place)
{
    static_assert(ARM >= 3u && ARM <= 6u, "PERS_RC_ARM: 3 ep2, 4 ep3, 5 fl2, 6 fl3");
    pers_rc_guard(work + pers_rc2_work_floats<ARM>(nb, tw, in_place));
    if (ARM == 3u) {
        pers_rc2_endpass_np(n, nb, tw, tab, invnorm, xs, row_at, epi, work);
        return;
    }
    if (ARM == 4u) {
        pers_rc2_endpass_one(n, nb, tw, tab, invnorm, xs, row_at, epi, work);
        return;
    }
    float* const xp = in_place ? xs : work;
    pers_rc_repad(xs, xp, nb, tw);
    pers_rc2_flat_one<ARM == 6u>(n, nb, tw, tab, invnorm, xp, row_at, epi,
                                 in_place ? work : work + pers_rc_padded_floats(nb, tw));
}

#endif  // PERS_4E_RC2_CUH
