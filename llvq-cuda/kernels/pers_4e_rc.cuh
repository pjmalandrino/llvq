// pers_4e_rc.cuh: the two row bodies of the decode-speed research phase (R3, rowbench's `endpass` and `flat`,
// docs/mesures/recherche-debit-corps-2026-11-13.txt) as row forms of 4E v9, for the token-level measurement (R6).
//
// Never part of a served module. pers.rs::p4e_e1_edits inserts this text into llvq_pers.cuh, before
// `struct Pers4eQkv`, in the modules of the E1 arms "v9ep" and "v9fl" only, with `#define PERS_RC_ARM` naming the
// body, and rewrites 4E's Tetra row phases (q/k, gate/up, the Tetra down, the 8B's Tetra o) to call
// pers_rc_rows<PERS_RC_ARM> on the CTA's whole slice instead of pers_4e_rows on each warp's list. The served v9
// module does not move. tests/host_pers.cpp includes this file on the persistent side (command 'e', the row gate);
// the include guard makes that include a no-op in an arm's twin, whose llvq_pers.cuh already holds the text.
//
// The interface of both bodies is pers_4e_rows_cta's: the CTA's rows idx in [0, n), ROW_AT(idx) gives the row
// (PersTRow), EPI(idx, y) receives the row's f32 value before f2h, on any thread. Every thread of the CTA calls it;
// it holds barriers; `xs` is the activation staged by the phase (natural stride 24, the tail after the last block).
//
//   endpass (PERS_RC_ARM 1, owes the BITS): 4E v9's row (pers_tetra_row with the next block in flight, a warp a row,
//     row idx on warp idx mod nwarps as v9 deals them), lane 0 writing the row's sum to a slot; after a barrier a
//     thread a row does pers_epilogue on the slot and EPI. v9's arithmetic, value for value.
//   flat (PERS_RC_ARM 2, owes the BOUND 1e-5 on rows): the activation restaged with one float of padding every 8
//     blocks; the CTA's rows as one stream of runs (a run is 8 consecutive blocks, 48 bytes, three aligned 128-bit
//     loads when the row is on 16 bytes), unit u = run u mod G of row u / G, a thread a unit, each run's fold to its
//     own slot; after a barrier a thread a row sums its G slots in run order and does pers_epilogue. A reassociated
//     sum: rowbench's RowcFlat, whose rows read within 1.7e-6 of the served rows on the card.
//
// Shared memory. Both bodies use only the part of 4E's dynamic buffer that the row phases leave free: after the
// widest row-phase vector and before the v7 area, the region the attention phase's stage occupies between row
// phases. No byte is added to 4E's launch. The caller passes `work`, the first free float; pers_rc_rows checks the
// last float it uses against the v7 area's offset (header u32 8) and traps past it; pers.rs::p4e_rc_layout refuses a
// shape where it would, at install, by name.
#ifndef PERS_4E_RC_CUH
#define PERS_4E_RC_CUH

#ifndef PERS_RC_SLOTS
#define PERS_RC_SLOTS 2048u  // f32 slots: rows (endpass) or runs (flat) a chunk; 8,192 B, rowbench's extra region
#endif
#define PERS_RC_RB 4u  // values a thread a batch of the in-place restaging

#ifdef LLVQ_HOST_BUILD
#define PERS_RC_TRAP() std::abort()
#else
#define PERS_RC_TRAP() __trap()
#endif

// 4E's row vector `v` (pers_4e_body): after the header and the decoder tables.
__device__ __forceinline__ float* pers_rc_v()
{
    return reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER + 18816u);
}

// The last float a body touches must lie at or before the v7 area (form 5's launch constants).
__device__ __forceinline__ void pers_rc_guard(const float* end)
{
    if (end > pers_rc_v() + pers_hdr()[PERS_HDR_V7]) PERS_RC_TRAP();
}

// The padded staging: block b at 24 b + b / 8, the tail at 24 nb + nb / 8; its length rounded to 16 bytes.
__device__ __forceinline__ u32 pers_rc_xat(u32 b)
{
    return 24u * b + (b >> 3);
}
__device__ __forceinline__ u32 pers_rc_padded_floats(u32 nb, u32 tw)
{
    return (pers_rc_xat(nb) + tw + 3u) & ~3u;
}

// The natural staging at xs copied into the padded one at xp, xp >= xs (xp == xs: in place). A destination never
// lies below its source, so in place the values move from the top down, PERS_RC_RB x blockDim a batch: every thread
// reads its values of the batch, a barrier, then writes them; a write lands on sources of its own batch (read before
// the barrier) or of higher batches (read before an earlier barrier). Disjoint (xp at or past xs + n): one pass.
// Barriers on both sides.
__device__ __forceinline__ void pers_rc_repad(const float* xs, float* xp, u32 nb, u32 tw)
{
    const u32 n = 24u * nb + tw;
    const auto dst = [&](u32 i) {
        const u32 b = i / 24u;
        return i + ((b < nb ? b : nb) >> 3);
    };
    __syncthreads();
    if (xp >= xs + n) {
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xp[dst(i)] = xs[i];
        __syncthreads();
        return;
    }
    const u32 batch = PERS_RC_RB * blockDim.x;
    for (u32 top = n; top > 0u;) {
        const u32 s = top > batch ? top - batch : 0u;
        float t[PERS_RC_RB];
#pragma unroll
        for (u32 j = 0; j < PERS_RC_RB; ++j) {
            const u32 i = s + threadIdx.x + j * blockDim.x;
            t[j] = i < top ? xs[i] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (u32 j = 0; j < PERS_RC_RB; ++j) {
            const u32 i = s + threadIdx.x + j * blockDim.x;
            if (i < top) xp[dst(i)] = t[j];
        }
        top = s;
    }
    __syncthreads();
}

// endpass. Chunks of PERS_RC_SLOTS rows; slots[idx - c0] is written by lane 0 of the one warp that sums row idx.
template <class RowAt, class Epi>
__device__ __forceinline__ void pers_rc_endpass(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                                const float* xs, RowAt row_at, Epi epi, float* slots)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (u32 c0 = 0; c0 < n; c0 += PERS_RC_SLOTS) {
        const u32 c1 = n - c0 < PERS_RC_SLOTS ? n : c0 + PERS_RC_SLOTS;
        for (u32 i = c0 + warp; i < c1; i += nw) {
            const PersTRow R = row_at(i);
            const float a = pers_tetra_row<24u, false, true>(R.w, tab, R.gscale, invnorm, xs, nb, lane);
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

// The 12 words of run g of a row: three aligned 128-bit loads when the row is on 16 bytes, else 12 u32 loads (the
// same words). Units past the row's blocks (nq quads, nwd u32) read as zero: they are never decoded.
__device__ __forceinline__ void pers_rc_run_words(const u32* p, u32 g, u32 nq, u32 nwd, u32 (&w)[12])
{
    if ((reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull) {
#pragma unroll
        for (u32 k = 0; k < 3u; ++k) {
            const u32 q = 3u * g + k;
            uint4 v = {0u, 0u, 0u, 0u};
            if (q < nq) v = reinterpret_cast<const uint4*>(p)[q];
            w[4u * k] = v.x;
            w[4u * k + 1u] = v.y;
            w[4u * k + 2u] = v.z;
            w[4u * k + 3u] = v.w;
        }
    } else {
#pragma unroll
        for (u32 k = 0; k < 12u; ++k) {
            const u32 i = 12u * g + k;
            w[k] = i < nwd ? p[i] : 0u;
        }
    }
}

// Run g's blocks folded from 0.0f in block order, each as pers_block folds it, on the padded staging.
__device__ __forceinline__ float pers_rc_run_fold(const u32 (&w)[12], u32 g, u32 nb, const F1rTables& tab,
                                                  const float* xp, const float* gs, const float* invnorm)
{
    float acc = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 8u; ++i) {
        const u32 b = 8u * g + i;
        if (b < nb) {
            const u32 a = (3u * i) >> 1;
            const u32 lo = (i & 1u) ? ((w[a] >> 16) | (w[a + 1u] << 16)) : w[a];
            const u32 hi16 = (i & 1u) ? (w[a + 1u] >> 16) : (w[a + 1u] & 0xffffu);
            acc = pers_block(acc, lo, hi16, tab, xp + pers_rc_xat(b), gs, invnorm);
        }
    }
    return acc;
}

// flat. Chunks of R = PERS_RC_SLOTS / G rows; slot u of a chunk is written by the one thread that folds unit u and
// read, after the barrier, by the one thread of its row; the barrier that ends the chunk orders the next chunk's
// writes. No atomic, no shuffle: a row's sum has one order, runs 0 to G - 1.
template <class RowAt, class Epi>
__device__ __forceinline__ void pers_rc_flat(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                             const float* xp, RowAt row_at, Epi epi, float* slots)
{
    const u32 G = (nb + 7u) >> 3, R = PERS_RC_SLOTS / G;
    const u32 nq = (6u * nb + 15u) >> 4, nwd = (6u * nb + 3u) >> 2;
    for (u32 c0 = 0; c0 < n; c0 += R) {
        const u32 c1 = n - c0 < R ? n : c0 + R;
        const u32 nu = (c1 - c0) * G;
        for (u32 u = threadIdx.x; u < nu; u += blockDim.x) {
            const u32 rr = u / G, g = u - rr * G;
            const PersTRow Rw = row_at(c0 + rr);
            u32 w[12];
            pers_rc_run_words(Rw.w, g, nq, nwd, w);
            slots[u] = pers_rc_run_fold(w, g, nb, tab, xp, Rw.gscale, invnorm);
        }
        __syncthreads();
        for (u32 i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
            float a = 0.0f;
            for (u32 g = 0; g < G; ++g) a = __fadd_rn(a, slots[(i - c0) * G + g]);
            const PersTRow Rw = row_at(i);
            epi(i, pers_epilogue(a, Rw.rscale[Rw.r], Rw.tail + (unsigned long long)Rw.r * tw, xp + pers_rc_xat(nb), tw));
        }
        __syncthreads();
    }
}

// Floats a body uses from `work` on: the slots, and before them flat's padded copy unless it restages in place.
template <u32 ARM>
__device__ __forceinline__ u32 pers_rc_work_floats(u32 nb, u32 tw, bool in_place)
{
    return (ARM == 2u && !in_place ? pers_rc_padded_floats(nb, tw) : 0u) + PERS_RC_SLOTS;
}

// The row form the arm's module runs: ARM 1 endpass, 2 flat. `work` is the first free float of 4E's buffer; with
// `in_place` flat restages xs where it lies (the caller has put `work` at or past xs + pers_rc_padded_floats),
// otherwise into `work`, its slots after.
template <u32 ARM, class RowAt, class Epi>
__device__ __forceinline__ void pers_rc_rows(u32 n, u32 nb, u32 tw, const F1rTables& tab, const float* invnorm,
                                             float* xs, RowAt row_at, Epi epi, float* work, bool in_place)
{
    static_assert(ARM == 1u || ARM == 2u, "PERS_RC_ARM: 1 endpass, 2 flat");
    pers_rc_guard(work + pers_rc_work_floats<ARM>(nb, tw, in_place));
    if (ARM == 1u) {
        pers_rc_endpass(n, nb, tw, tab, invnorm, xs, row_at, epi, work);
        return;
    }
    float* const xp = in_place ? xs : work;
    pers_rc_repad(xs, xp, nb, tw);
    pers_rc_flat(n, nb, tw, tab, invnorm, xp, row_at, epi, in_place ? work : work + pers_rc_padded_floats(nb, tw));
}

#endif  // PERS_4E_RC_CUH
