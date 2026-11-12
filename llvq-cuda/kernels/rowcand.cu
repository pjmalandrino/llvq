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
#endif
