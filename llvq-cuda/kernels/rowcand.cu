// rowcand.cu: the row programs of the decode-speed research phase (MODE RECHERCHE, step R2,
// docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt), and the bench bodies that run them at 4E's geometry.
//
// A CANDIDATE is a struct with one static member function:
//
//   static __device__ __forceinline__ float row(const u32* wrow, const F1rTables& tab, const float* gscale,
//                                               const float* invnorm, const float* xs, u32 nblocks, u32 lane);
//
// `wrow` is the row's words (the served Tetra48 stream, or the candidate's own relaid copy, see RELAY), `tab`
// the decoder tables in shared memory, `gscale` the matrix's two centroid scales, `invnorm` 1/sqrt(16 m) in
// shared memory, `xs` the rotated activation staged in natural order (block b at xs + 24 b, the tail after the
// last block). It returns the row's sum over its `nblocks` blocks, each block scaled as pers_fold scales it,
// BEFORE the tail and the row scale, valid on lane 0 at least. The body applies the served epilogue
// (pers_epilogue: the binary16 tail, then the row scale) and writes y in binary16, and its f32 value before f2h
// when the site table asks for it.
//
// REGISTRATION is one line of ROWC_ARMS below:
//
//   X(name, Struct, OWES, bound, RELAY, BPI)
//
//   name    the arm's name; the kernels are tv_rowc_<name> (the 4E row geometry over a list of sites),
//           tv_rowc_gu_<name> (4E's gate/up phase: pairs of rows, gate then up) and tv_rowc_relay_<name>.
//   OWES    ROWC_BITS: y and its f32 value equal the served row's bit for bit;
//           ROWC_BOUND: |f32 - served f32| / rms(served f32 of the site) <= bound on every row (a reassociated
//           sum); ROWC_NONE: a diagnostic that owes nothing (its rate is printed as such, never as a candidate).
//   bound   the relative bound of ROWC_BOUND, a decimal literal; 0 otherwise.
//   RELAY   0: the arm reads the served stream. 1: Struct also has
//             static __device__ void relay(const u32* src_row, u32* dst_row, u32 nblocks, u32 stride_u32);
//           which rewrites one row of served words into the arm's layout, in the SAME stride (no more VRAM);
//           the host runs tv_rowc_relay_<name> once at load and the arm reads the copy.
//   BPI     blocks one lane folds in one iteration of the struct's inner loop, to turn the SASS loop count into
//           instructions a block of 24 weights (ops/rowcand-sass.sh).
//
// The host twin is this text: tests/host_pers.cpp compiles it on the Mac (command 'W') and the stage-1 gate
// (tests/rowcand_gate.rs) runs every registered arm against the served tv_tetra48_h on recorded words. Nothing
// else is written by hand for a new candidate.
//
// Composition: its own module, the row-loop unit (pers.rs::trows_unit_text), then bwprobe.cu, then this file
// (rowcand.rs::rowc_unit_text). No served module changes a byte.

#define ROWC_BITS 0u
#define ROWC_BOUND 1u
#define ROWC_NONE 2u

// ---- the candidates -------------------------------------------------------------------------------------------

// The served row: 4E v9's Tetra row (pers_tetra_rows_static), v2's row with the next block's words in flight.
// tests/pers_matches_served.rs proves it equal to tv_tetra48_h bit for bit; the gate proves it again here.
struct RowcServed {
    static __device__ __forceinline__ float row(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
    {
        return pers_tetra_row<24u, false, true>(wrow, tab, gscale, invnorm, xs, nblocks, lane);
    }
};

// The same row with no word in flight (pers_tetra_row's plain loop): the served bits, half the bytes in flight.
struct RowcServedNp {
    static __device__ __forceinline__ float row(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
    {
        return pers_tetra_row<24u, false, false>(wrow, tab, gscale, invnorm, xs, nblocks, lane);
    }
};

// Two accumulators a lane: blocks lane + 64 i on the first, lane + 32 + 64 i on the second, summed before the
// butterfly. Two independent fold chains, a reassociated sum: it owes the bound, not the bits.
struct RowcAcc2 {
    static __device__ __forceinline__ float row(const u32* wrow, const F1rTables& tab, const float* gscale,
                                                const float* invnorm, const float* xs, u32 nblocks, u32 lane)
    {
        float a0 = 0.0f, a1 = 0.0f;
        u32 lo = 0u, hi16 = 0u;
        if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
        for (u32 j = lane; j < nblocks; j += 64u) {
            u32 lo1 = 0u, hi1 = 0u, lo2 = 0u, hi2 = 0u;
            if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo1, hi1);
            if (j + 64u < nblocks) f1r_load(wrow, j + 64u, lo2, hi2);
            a0 = pers_block(a0, lo, hi16, tab, xs + j * 24u, gscale, invnorm);
            if (j + 32u < nblocks) a1 = pers_block(a1, lo1, hi1, tab, xs + (j + 32u) * 24u, gscale, invnorm);
            lo = lo2;
            hi16 = hi2;
        }
        return warp_sum(__fadd_rn(a0, a1));
    }
};

// pword: the row's words read as the served row reads them, folded by xor, nothing decoded. A diagnostic: the
// rate of the Tetra stream at this geometry with no decode, the Tetra twin of q4read.
struct RowcPword {
    static __device__ __forceinline__ float row(const u32* wrow, const F1rTables&, const float*, const float*,
                                                const float*, u32 nblocks, u32 lane)
    {
        u32 h = 0u;
        u32 lo = 0u, hi16 = 0u;
        if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);
        for (u32 j = lane; j < nblocks; j += 32u) {
            u32 lo2 = 0u, hi2 = 0u;
            if (j + 32u < nblocks) f1r_load(wrow, j + 32u, lo2, hi2);
            h ^= lo ^ (hi16 << 7);
            lo = lo2;
            hi16 = hi2;
        }
        return warp_sum((float)(h & 0xffu));
    }
};

// ---- the registry: ONE line a candidate ----------------------------------------------------------------------
#define ROWC_ARMS(X)                                       \
    X(served, RowcServed, ROWC_BITS, 0, 0, 1)              \
    X(served_np, RowcServedNp, ROWC_BITS, 0, 0, 1)         \
    X(acc2, RowcAcc2, ROWC_BOUND, 1e-5, 0, 2)              \
    X(pword, RowcPword, ROWC_NONE, 0, 0, 1)

// ---- the bench bodies -----------------------------------------------------------------------------------------

// Site words: TROWS_SITE_WORDS u64, laid out by rowcand.rs::rowc_site_words (pers_trows.cu's layout):
//   [0] words  [1] gscale  [2] rscale  [3] tail (binary16)  [4] y (binary16)
//   [5] d_out | nblocks << 32  [6] tail_w | row_stride_u32 << 32  [7] the f32 value of each row, or 0
//   [8], [9] unused (0)

// 4E's shared layout (pers_4e.cu, pers_trows.cu): the header, the decoder tables, invnorm, then the activation
// in natural staging, stride 24. Returns the activation's base.
__device__ __forceinline__ float* rowc_stage_tables(const u32* __restrict__ rows,
                                                    const unsigned char* __restrict__ prefixes,
                                                    const unsigned short* __restrict__ branches,
                                                    const unsigned char* __restrict__ suffixes,
                                                    const float* __restrict__ invnorm, F1rTables& tab, float*& t_inv)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    t_inv = reinterpret_cast<float*>(t_suf + 128);
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = prefixes[i];
        t_suf[i] = suffixes[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = branches[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = invnorm[i];
    tab = F1rTables{t_rows, t_pref, t_br, t_suf};
    return t_inv + 32;
}

// One site's rows on one warp: rows lo + warp, lo + warp + nw, ... of the CTA's slice, as 4E's static form.
template <class C>
__device__ __forceinline__ void rowc_site_rows(const unsigned long long* d, const F1rTables& tab, const float* t_inv,
                                               const float* xs, u32 lo, u32 hi, u32 warp, u32 nw, u32 lane)
{
    const u32* words = reinterpret_cast<const u32*>(d[0]);
    const float* gs = reinterpret_cast<const float*>(d[1]);
    const float* rs = reinterpret_cast<const float*>(d[2]);
    const unsigned short* tail = reinterpret_cast<const unsigned short*>(d[3]);
    unsigned short* y = reinterpret_cast<unsigned short*>(d[4]);
    const u32 nb = (u32)(d[5] >> 32);
    const u32 tw = (u32)(d[6] & 0xffffffffull), stride = (u32)(d[6] >> 32);
    float* yf = reinterpret_cast<float*>(d[7]);
    for (u32 r = lo + warp; r < hi; r += nw) {
        const float acc = C::row(words + (size_t)r * stride, tab, gs, t_inv, xs, nb, lane);
        if (lane == 0u) {
            const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
            y[r] = f2h(v);
            if (yf != nullptr) yf[r] = v;
        }
    }
}

// tv_rowc_<name>: the sites one after the other, each split over the CTAs as 4E splits a phase,
// [d c / g, d (c + 1) / g), and over the warps statically; the activation restaged at every site.
template <class C>
__device__ __forceinline__ void rowc_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                          const u32* __restrict__ rows, const unsigned char* __restrict__ prefixes,
                                          const unsigned short* __restrict__ branches,
                                          const unsigned char* __restrict__ suffixes,
                                          const float* __restrict__ invnorm, const float* __restrict__ x)
{
    F1rTables tab;
    float* t_inv;
    float* xs = rowc_stage_tables(rows, prefixes, branches, suffixes, invnorm, tab, t_inv);
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    for (u32 s = 0; s < nsites; ++s) {
        const unsigned long long* d = sites + (size_t)s * TROWS_SITE_WORDS;
        const u32 d_out = (u32)(d[5] & 0xffffffffull), nb = (u32)(d[5] >> 32);
        const u32 tw = (u32)(d[6] & 0xffffffffull);
        pers_stage_f32<24u, false>(x, xs, nb, tw);  // barriers on both sides
        const u32 lo = (u32)((unsigned long long)d_out * c / g), hi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        rowc_site_rows<C>(d, tab, t_inv, xs, lo, hi, warp, nw, lane);
    }
}

// tv_rowc_gu_<name>: 4E's gate/up phase alone (pers_segment.cu, IN4E && PF, form 7). The sites come in pairs,
// gate then up of one layer, same shape. The CTA takes the pairs [d c / g, d (c + 1) / g); warp w takes pairs
// glo + w, glo + w + nw, ..., the gate row then the up row of each, both on one staged activation.
template <class C>
__device__ __forceinline__ void rowc_gu_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                             const u32* __restrict__ rows, const unsigned char* __restrict__ prefixes,
                                             const unsigned short* __restrict__ branches,
                                             const unsigned char* __restrict__ suffixes,
                                             const float* __restrict__ invnorm, const float* __restrict__ x)
{
    F1rTables tab;
    float* t_inv;
    float* xs = rowc_stage_tables(rows, prefixes, branches, suffixes, invnorm, tab, t_inv);
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    for (u32 s = 0; s + 1u < nsites; s += 2u) {
        const unsigned long long* dg = sites + (size_t)s * TROWS_SITE_WORDS;
        const unsigned long long* du = dg + TROWS_SITE_WORDS;
        const u32 d_out = (u32)(dg[5] & 0xffffffffull), nb = (u32)(dg[5] >> 32);
        const u32 tw = (u32)(dg[6] & 0xffffffffull);
        pers_stage_f32<24u, false>(x, xs, nb, tw);
        const u32 glo = (u32)((unsigned long long)d_out * c / g), ghi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        for (u32 p = glo + warp; p < ghi; p += nw) {
            rowc_site_rows<C>(dg, tab, t_inv, xs, p, p + 1u, 0u, 1u, lane);
            rowc_site_rows<C>(du, tab, t_inv, xs, p, p + 1u, 0u, 1u, lane);
        }
    }
}

// tv_rowc_relay_<name>: a RELAY arm's rows rewritten once at load, in the served stride; an arm with RELAY 0
// gets a plain copy (never launched by the host).
template <class C, u32 RELAY>
struct RowcRelay {
    static __device__ __forceinline__ void run(const u32* __restrict__ src, u32* __restrict__ dst, u32 stride,
                                               u32 d_out, u32)
    {
        const unsigned long long n = (unsigned long long)d_out * stride;
        for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (unsigned long long)gridDim.x * blockDim.x)
            dst[i] = src[i];
    }
};
template <class C>
struct RowcRelay<C, 1u> {
    static __device__ __forceinline__ void run(const u32* __restrict__ src, u32* __restrict__ dst, u32 stride,
                                               u32 d_out, u32 nblocks)
    {
        for (u32 r = blockIdx.x * blockDim.x + threadIdx.x; r < d_out; r += gridDim.x * blockDim.x)
            C::relay(src + (size_t)r * stride, dst + (size_t)r * stride, nblocks, stride);
    }
};

#define ROWC_SITE_ARGS                                                                                         \
    const unsigned long long *__restrict__ sites, u32 nsites, const u32 *__restrict__ rows,                    \
        const unsigned char *__restrict__ prefixes, const unsigned short *__restrict__ branches,               \
        const unsigned char *__restrict__ suffixes, const float *__restrict__ invnorm, const float *__restrict__ x

#define ROWC_KERNELS(name, S, OWES, BOUND, RELAY, BPI)                                                         \
    extern "C" __global__ void __launch_bounds__(512, 1) tv_rowc_##name(ROWC_SITE_ARGS)                        \
    {                                                                                                          \
        rowc_body<S>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x);                           \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(512, 1) tv_rowc_gu_##name(ROWC_SITE_ARGS)                     \
    {                                                                                                          \
        rowc_gu_body<S>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x);                        \
    }                                                                                                          \
    extern "C" __global__ void tv_rowc_relay_##name(const u32* __restrict__ src, u32* __restrict__ dst,         \
                                                    u32 stride, u32 d_out, u32 nblocks)                        \
    {                                                                                                          \
        RowcRelay<S, RELAY>::run(src, dst, stride, d_out, nblocks);                                            \
    }

ROWC_ARMS(ROWC_KERNELS)

// ---- the read diagnostics (recherche-debit-r1-recadrage-2026-11-12, H0 to H5) ---------------------------------
//
// pword reads the sealed file's Tetra words at 493 GB/s where the bare probe reads 751. Each arm below changes ONE
// thing of pword's read, so the bench can say how much of the gap each hypothesis of the reframing note holds:
//
//   pw_chk      pword itself, its xor fold replaced by the read check (below): the base of the others
//   pw_noside   H1, no side stream: no binary16 tail, no row scale read, no y store (the row's reduction is kept)
//   pw_flat     H2, no row: the CTA's slice of the site, [d c / g, d (c + 1) / g) rows, read as one flat stream of
//               6-byte units by all its threads, no row boundary, no per-row warp reduction; the epilogue of each
//               row (tail, scale, store) is kept, in its own loop after the read
//   pw_u4       H3, the load: each row's words read by aligned 128-bit loads, one a lane, the next in flight as
//               pword keeps its next block in flight
//   pw_nostage  H4, the site: the activation staged ONCE at the widest site, not at every site (pword stages the
//               decoder tables once a launch already; only the activation and its two barriers are per site)
//   pw_grid     H5, the schedule: pw_flat with the probe's schedule, every thread of the GRID striding the WHOLE site
//               (bwprobe.cu's grid stride), no split of the site over the CTAs; the epilogue as pw_flat
//   pw_all      the stack: pw_grid's schedule, 128-bit loads, no per-site staging, no side stream
//   pw_empty    H0: the same launch over the same site list, reading only the sites' descriptors
//
// THE READ CHECK. Every arm but pw_empty adds every u32 of the words it read into a u64 (pword's 48-bit units add
// their bytes with each half-word at its weight in its u32, rowd_unit_sum), keeps the sum a thread over the whole
// launch, and writes it with one more u64 (`sink`, what keeps an unused value alive) at out[2 t], out[2 t + 1], t
// the thread's index in the grid. `out` is the site table's word [8]. The host adds the 2 x grid x threads slots
// and compares with its own sum of the words: every word read once, or the arm is refused. pw_empty adds d_out of
// every site it visits; the host expects grid x threads x the sum of d_out. The row padding is zero (transcode).
//
// REGISTRATION: X(name, Policy, hypothesis, base), `base` the arm it differs from by one thing.

#define ROWD_ROWS 0u  // rows over warps, as pword
#define ROWD_FLAT 1u  // the CTA's slice as one stream of its threads
#define ROWD_GRID 2u  // the whole site as one stream of the grid's threads
#define ROWD_L48 0u   // f1r_load: two u32 a 48-bit block, as the served row
#define ROWD_U4 1u    // one 128-bit load a lane, 16-byte aligned

struct RowdPwChk {
    static constexpr u32 SCHED = ROWD_ROWS, LOAD = ROWD_L48;
    static constexpr bool STAGE = true, EPI = true, EMPTY = false;
};
struct RowdNoSide {
    static constexpr u32 SCHED = ROWD_ROWS, LOAD = ROWD_L48;
    static constexpr bool STAGE = true, EPI = false, EMPTY = false;
};
struct RowdFlat {
    static constexpr u32 SCHED = ROWD_FLAT, LOAD = ROWD_L48;
    static constexpr bool STAGE = true, EPI = true, EMPTY = false;
};
struct RowdU4 {
    static constexpr u32 SCHED = ROWD_ROWS, LOAD = ROWD_U4;
    static constexpr bool STAGE = true, EPI = true, EMPTY = false;
};
struct RowdNoStage {
    static constexpr u32 SCHED = ROWD_ROWS, LOAD = ROWD_L48;
    static constexpr bool STAGE = false, EPI = true, EMPTY = false;
};
struct RowdGrid {
    static constexpr u32 SCHED = ROWD_GRID, LOAD = ROWD_L48;
    static constexpr bool STAGE = true, EPI = true, EMPTY = false;
};
struct RowdAll {
    static constexpr u32 SCHED = ROWD_GRID, LOAD = ROWD_U4;
    static constexpr bool STAGE = false, EPI = false, EMPTY = false;
};
struct RowdEmpty {
    static constexpr u32 SCHED = ROWD_ROWS, LOAD = ROWD_L48;
    static constexpr bool STAGE = false, EPI = false, EMPTY = true;
};

#define ROWD_ARMS(X)                          \
    X(pw_chk, RowdPwChk, pword, pword)        \
    X(pw_noside, RowdNoSide, H1, pw_chk)      \
    X(pw_flat, RowdFlat, H2, pw_chk)          \
    X(pw_u4, RowdU4, H3, pw_chk)              \
    X(pw_nostage, RowdNoStage, H4, pw_chk)    \
    X(pw_grid, RowdGrid, H5, pw_flat)         \
    X(pw_all, RowdAll, stack, pw_chk)         \
    X(pw_empty, RowdEmpty, H0, pw_chk)

// The bytes of one 48-bit unit f1r_load gave, each half-word at its weight in its u32: an even unit holds word w and
// the low half of w + 1, an odd unit the high half of w and word w + 1 (llvq_f1rank.cuh, f1r_load).
__device__ __forceinline__ unsigned long long rowd_unit_sum(u32 lo, u32 hi16, u32 odd)
{
    return odd ? (unsigned long long)(lo << 16) + (unsigned long long)((lo >> 16) | (hi16 << 16))
               : (unsigned long long)lo + (unsigned long long)hi16;
}

// One row's blocks, as pword reads them (lane j, j + 32, ..., the next block in flight), summed.
__device__ __forceinline__ unsigned long long rowd_row48(const u32* wrow, u32 nb, u32 lane)
{
    unsigned long long s = 0ull;
    u32 lo = 0u, hi16 = 0u;
    if (lane < nb) f1r_load(wrow, lane, lo, hi16);
    for (u32 j = lane; j < nb; j += 32u) {
        u32 lo2 = 0u, hi2 = 0u;
        if (j + 32u < nb) f1r_load(wrow, j + 32u, lo2, hi2);
        s += rowd_unit_sum(lo, hi16, j & 1u);
        lo = lo2;
        hi16 = hi2;
    }
    return s;
}

// n u32 from p (n even) as 48-bit units, thread t of T: unit u is read by thread u mod T, the next in flight. Units
// come in pairs over three words, so none reaches past p + n; the 0 to 2 words after the last whole pair are read
// as u32 by threads 0 and 1.
__device__ __forceinline__ unsigned long long rowd_flat48(const u32* p, unsigned long long n, unsigned long long t,
                                                          unsigned long long T)
{
    const unsigned long long nu = 2ull * (n / 3ull);
    unsigned long long s = 0ull;
    u32 lo = 0u, hi16 = 0u;
    if (t < nu) f1r_load(p, (u32)t, lo, hi16);
    for (unsigned long long u = t; u < nu; u += T) {
        u32 lo2 = 0u, hi2 = 0u;
        if (u + T < nu) f1r_load(p, (u32)(u + T), lo2, hi2);
        s += rowd_unit_sum(lo, hi16, (u32)(u & 1ull));
        lo = lo2;
        hi16 = hi2;
    }
    const unsigned long long whole = 3ull * (n / 3ull);
    if (t < n - whole) s += p[whole + t];
    return s;
}

// n u32 from p by 128-bit loads, thread t of T, the next load in flight: the 0 or 2 words before the first 16-byte
// boundary and the 0 to 3 after the last whole quad as u32 (the row stride is even, so p is 8-byte aligned).
__device__ __forceinline__ unsigned long long rowd_read_u4(const u32* p, unsigned long long n, unsigned long long t,
                                                           unsigned long long T)
{
    const unsigned long long mis = (reinterpret_cast<unsigned long long>(p) & 15ull) >> 2;
    const unsigned long long head = mis == 0ull ? 0ull : (4ull - mis < n ? 4ull - mis : n);
    unsigned long long s = 0ull;
    if (t < head) s += p[t];
    const uint4* q = reinterpret_cast<const uint4*>(p + head);
    const unsigned long long nq = (n - head) >> 2;
    uint4 v = {0u, 0u, 0u, 0u};
    if (t < nq) v = q[t];
    for (unsigned long long i = t; i < nq; i += T) {
        uint4 v2 = {0u, 0u, 0u, 0u};
        if (i + T < nq) v2 = q[i + T];
        s += (unsigned long long)v.x + v.y + v.z + v.w;
        v = v2;
    }
    const unsigned long long rest = (n - head) & 3ull;
    if (t < rest) s += p[head + 4ull * nq + t];
    return s;
}

// One row as pword runs it: the read, the warp's reduction of a value the read made, the epilogue on lane 0.
template <class P>
__device__ __forceinline__ void rowd_row(const unsigned long long* d, const float* xs, u32 r, u32 lane,
                                         unsigned long long& chk, unsigned long long& sink)
{
    const u32* words = reinterpret_cast<const u32*>(d[0]);
    const u32 nb = (u32)(d[5] >> 32);
    const u32 tw = (u32)(d[6] & 0xffffffffull), stride = (u32)(d[6] >> 32);
    const u32* wrow = words + (size_t)r * stride;
    const unsigned long long s = P::LOAD == ROWD_U4 ? rowd_read_u4(wrow, stride, lane, 32u) : rowd_row48(wrow, nb, lane);
    chk += s;
    const float acc = warp_sum((float)(u32)(s & 0xffull));
    if (P::EPI) {
        if (lane == 0u) {
            const float* rs = reinterpret_cast<const float*>(d[2]);
            const unsigned short* tail = reinterpret_cast<const unsigned short*>(d[3]);
            unsigned short* y = reinterpret_cast<unsigned short*>(d[4]);
            float* yf = reinterpret_cast<float*>(d[7]);
            const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
            y[r] = f2h(v);
            if (yf != nullptr) yf[r] = v;
        }
    } else {
        sink += __float_as_uint(acc);
    }
}

// The epilogue of rows [lo, hi) of site d on warp `warp` of nw, with no row read: pw_flat's and pw_grid's side
// streams, the same loads and stores as pword's epilogue.
__device__ __forceinline__ void rowd_epilogues(const unsigned long long* d, const float* xs, u32 lo, u32 hi, u32 warp,
                                               u32 nw, u32 lane, float acc)
{
    const float* rs = reinterpret_cast<const float*>(d[2]);
    const unsigned short* tail = reinterpret_cast<const unsigned short*>(d[3]);
    unsigned short* y = reinterpret_cast<unsigned short*>(d[4]);
    float* yf = reinterpret_cast<float*>(d[7]);
    const u32 nb = (u32)(d[5] >> 32);
    const u32 tw = (u32)(d[6] & 0xffffffffull);
    if (lane != 0u) return;
    for (u32 r = lo + warp; r < hi; r += nw) {
        const float v = pers_epilogue(acc, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);
        y[r] = f2h(v);
        if (yf != nullptr) yf[r] = v;
    }
}

// tv_rowd_<name> (GU false) and tv_rowd_gu_<name> (GU true, pairs of sites as rowc_gu_body): pword's bodies with
// the policy's one change.
template <class P, bool GU>
__device__ __forceinline__ void rowd_body(const unsigned long long* __restrict__ sites, u32 nsites,
                                          const u32* __restrict__ rows, const unsigned char* __restrict__ prefixes,
                                          const unsigned short* __restrict__ branches,
                                          const unsigned char* __restrict__ suffixes,
                                          const float* __restrict__ invnorm, const float* __restrict__ x)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 c = blockIdx.x, g = gridDim.x;
    const unsigned long long gt = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long GT = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long* out = reinterpret_cast<unsigned long long*>(sites[8]);
    const u32 step = GU ? 2u : 1u;
    unsigned long long chk = 0ull, sink = 0ull;
    if (P::EMPTY) {
        for (u32 s = 0; s + step <= nsites; s += step)
            for (u32 k = 0; k < step; ++k) chk += sites[(size_t)(s + k) * TROWS_SITE_WORDS + 5] & 0xffffffffull;
        out[2ull * gt] = chk;
        out[2ull * gt + 1ull] = sink;
        return;
    }
    F1rTables tab;
    float* t_inv;
    float* xs = rowc_stage_tables(rows, prefixes, branches, suffixes, invnorm, tab, t_inv);
    (void)tab;
    (void)t_inv;
    if (!P::STAGE) {
        // once, at the widest site: every site's epilogue then reads staged values, at its own offsets
        u32 dmax = 0u;
        for (u32 s = 0; s < nsites; ++s) {
            const unsigned long long* d = sites + (size_t)s * TROWS_SITE_WORDS;
            const u32 din = (u32)(d[5] >> 32) * 24u + (u32)(d[6] & 0xffffffffull);
            dmax = din > dmax ? din : dmax;
        }
        pers_stage_f32<24u, false>(x, xs, dmax / 24u, dmax % 24u);
    }
    for (u32 s = 0; s + step <= nsites; s += step) {
        const unsigned long long* d = sites + (size_t)s * TROWS_SITE_WORDS;
        const u32 d_out = (u32)(d[5] & 0xffffffffull), nb = (u32)(d[5] >> 32);
        const u32 tw = (u32)(d[6] & 0xffffffffull);
        if (P::STAGE) pers_stage_f32<24u, false>(x, xs, nb, tw);  // barriers on both sides
        const u32 lo = (u32)((unsigned long long)d_out * c / g), hi = (u32)((unsigned long long)d_out * (c + 1u) / g);
        if (P::SCHED == ROWD_ROWS) {
            for (u32 r = lo + warp; r < hi; r += nw)
                for (u32 k = 0; k < step; ++k) rowd_row<P>(d + (size_t)k * TROWS_SITE_WORDS, xs, r, lane, chk, sink);
        } else {
            for (u32 k = 0; k < step; ++k) {
                const unsigned long long* dk = d + (size_t)k * TROWS_SITE_WORDS;
                const u32* words = reinterpret_cast<const u32*>(dk[0]);
                const unsigned long long stride = dk[6] >> 32;
                const bool flat = P::SCHED == ROWD_FLAT;
                const u32* p = flat ? words + lo * stride : words;
                const unsigned long long n = flat ? (unsigned long long)(hi - lo) * stride : d_out * stride;
                const unsigned long long t = flat ? (unsigned long long)threadIdx.x : gt;
                const unsigned long long T = flat ? (unsigned long long)blockDim.x : GT;
                chk += P::LOAD == ROWD_U4 ? rowd_read_u4(p, n, t, T) : rowd_flat48(p, n, t, T);
            }
            if (P::EPI)
                for (u32 k = 0; k < step; ++k)
                    rowd_epilogues(d + (size_t)k * TROWS_SITE_WORDS, xs, lo, hi, warp, nw, lane, (float)(u32)(chk & 0xffull));
        }
    }
    out[2ull * gt] = chk;
    out[2ull * gt + 1ull] = sink;
}

#define ROWD_KERNELS(name, P, H, BASE)                                                                         \
    extern "C" __global__ void __launch_bounds__(512, 1) tv_rowd_##name(ROWC_SITE_ARGS)                        \
    {                                                                                                          \
        rowd_body<P, false>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x);                    \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(512, 1) tv_rowd_gu_##name(ROWC_SITE_ARGS)                     \
    {                                                                                                          \
        rowd_body<P, true>(sites, nsites, rows, prefixes, branches, suffixes, invnorm, x);                     \
    }

ROWD_ARMS(ROWD_KERNELS)

#ifdef LLVQ_HOST_BUILD
// The host twin's dispatch, in registration order: the same bodies, called by tests/host_pers.cpp ('W').
typedef void (*RowcBodyFn)(const unsigned long long*, u32, const u32*, const unsigned char*, const unsigned short*,
                           const unsigned char*, const float*, const float*);
typedef void (*RowcRelayFn)(const u32*, u32*, u32, u32, u32);
#define ROWC_HOST_BODY(name, S, OWES, BOUND, RELAY, BPI) &rowc_body<S>,
#define ROWC_HOST_GU(name, S, OWES, BOUND, RELAY, BPI) &rowc_gu_body<S>,
#define ROWC_HOST_RELAY(name, S, OWES, BOUND, RELAY, BPI) &RowcRelay<S, RELAY>::run,
#define ROWC_HOST_RELAID(name, S, OWES, BOUND, RELAY, BPI) RELAY,
static RowcBodyFn const ROWC_HOST_BODIES[] = {ROWC_ARMS(ROWC_HOST_BODY)};
static RowcBodyFn const ROWC_HOST_GU_BODIES[] = {ROWC_ARMS(ROWC_HOST_GU)};
static RowcRelayFn const ROWC_HOST_RELAYS[] = {ROWC_ARMS(ROWC_HOST_RELAY)};
static const u32 ROWC_HOST_RELAID[] = {ROWC_ARMS(ROWC_HOST_RELAID)};
#define ROWC_HOST_COUNT (sizeof(ROWC_HOST_BODIES) / sizeof(ROWC_HOST_BODIES[0]))
#define ROWD_HOST_BODY(name, P, H, BASE) &rowd_body<P, false>,
#define ROWD_HOST_GU(name, P, H, BASE) &rowd_body<P, true>,
static RowcBodyFn const ROWD_HOST_BODIES[] = {ROWD_ARMS(ROWD_HOST_BODY)};
static RowcBodyFn const ROWD_HOST_GU_BODIES[] = {ROWD_ARMS(ROWD_HOST_GU)};
#define ROWD_HOST_COUNT (sizeof(ROWD_HOST_BODIES) / sizeof(ROWD_HOST_BODIES[0]))
#endif
