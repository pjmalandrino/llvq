// DRAFT, never compiled or run. Revised 2026-11-02 after review: the cross-CTA combine is now
// self-cleaning (no memset launch), the decoder tables are read through L1 as in the served
// kernel, and beta0's trivial decoder is sketched beside the real one.
//
// Device-side sketch of the brick's two back ends (wave5-deep/brick.md §3.2-3.4): the warp
// tile with its XOR swizzle, ldmatrix, the two int8 MMAs per k-step, the S_int carried in the
// A record's pad bytes, the int64 accumulation and the order-free cross-CTA combine. Shapes and
// fragment maps follow CUTLASS SM80_16x8x32_S32S8S8S32_TN (A: rows g, g+8; k 4t..4t+3 and
// 16+4t..19+4t; B: token g; C: (g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1)). NVRTC target compute_89
// (compute_80 for the A100). Every float operation is an _rn intrinsic: NVRTC runs --fmad=true.

typedef unsigned int u32;
typedef unsigned long long u64;

// ---- PTX wrappers -------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(u32 a[4], u32 smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem_addr));
}
__device__ __forceinline__ void mma_s8s8(int d[4], const u32 a[4], u32 b0, u32 b1, const int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void mma_s8u8(int d[4], const u32 a[4], u32 b0, u32 b1, const int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// signed weights x unsigned lo plane: the intrinsic __dp4a has no mixed form.
__device__ __forceinline__ int dp4a_su(u32 a_s8x4, u32 b_u8x4, int c) {
    int d; asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_s8x4), "r"(b_u8x4), "r"(c)); return d;
}
__device__ __forceinline__ long long mad_wide(int a, int b, long long c) {
    long long d; asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(d) : "r"(a), "r"(b), "l"(c)); return d;
}

// ---- warp tile ------------------------------------------------------------------------------
// 32 records of 32 B per round: record l = row (l & 15), block (l >> 4) of the round.
// Half h of record l lives at 16-byte chunk 2l + (h ^ ((l >> 2) & 1)): conflict-free for the two
// STS.128 of the decode and for ldmatrix.x4 (8 wavefronts per round instead of 16, enumerated).
__device__ __forceinline__ u32 chunk_of(u32 l, u32 h) { return 2u * l + (h ^ ((l >> 2) & 1u)); }

// The decoding lane writes its record: [q0 q2 q4 S | q1 q3 q5 0], q signed quads (biased tables
// XOR 0x80808080, or signed immediates). S = S_int[g][m] of the block, carried in k = 12..15,
// where every activation slot is zero (invariant; its mutant is a nonzero B slot 3).
__device__ __forceinline__ void store_record(char* tile, u32 lane, const u32 q[6], u32 s_int) {
    uint4 h0 = make_uint4(q[0], q[2], q[4], s_int);
    uint4 h1 = make_uint4(q[1], q[3], q[5], 0u);
    *reinterpret_cast<uint4*>(tile + 16u * chunk_of(lane, 0u)) = h0;
    *reinterpret_cast<uint4*>(tile + 16u * chunk_of(lane, 1u)) = h1;
}

// Row address this lane hands to ldmatrix.x4 for block jb (0 or 1) of the round: lanes 0-7 rows
// 0-7 half 0, 8-15 rows 8-15 half 0, 16-23 rows 0-7 half 1, 24-31 rows 8-15 half 1.
__device__ __forceinline__ u32 ldm_addr(u32 tile_s, u32 lane, u32 jb) {
    u32 row = lane & 15u, h = lane >> 4;
    u32 l = row + 16u * jb;
    return tile_s + 16u * chunk_of(l, h);
}

// ---- one k-step of the IMMA back end ---------------------------------------------------------
// NT n8 tiles (k <= 8 * NT tokens). Bhi/Blo point at this block's B records in shared memory,
// layout [plane][token][32 B]; thread (g, t) reads the 8-byte slot t of token 8n + g: one LDS.64,
// 2 wavefronts per warp instruction. sh[n][c] is the slice shift of token 8n + 2t + c.
template <int NT>
__device__ __forceinline__ void kstep(u32 tile_s, u32 lane, u32 jb,
                                      const char* Bhi, const char* Blo, const u32 sh[NT][2],
                                      long long acc[NT][4]) {
    u32 a[4];
    ldmatrix_x4(a, ldm_addr(tile_s, lane, jb));
    const int src = (lane & ~3u) | 3u;                         // lane t = 3 holds the pad bytes
    const int s_g  = (int)__shfl_sync(0xffffffffu, a[0], src);  // S_int of row g
    const int s_g8 = (int)__shfl_sync(0xffffffffu, a[1], src);  // S_int of row g + 8
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const u32 g = lane >> 2, t = lane & 3u;
        uint2 bh = *reinterpret_cast<const uint2*>(Bhi + 32u * (8u * n + g) + 8u * t);
        uint2 bl = *reinterpret_cast<const uint2*>(Blo + 32u * (8u * n + g) + 8u * t);
        int z[4] = {0, 0, 0, 0}, d[4];
        mma_s8s8(d, a, bh.x, bh.y, z);                             // P_hi
#pragma unroll
        for (int e = 0; e < 4; ++e) d[e] *= 256;                   // C operand of the lo MMA
        mma_s8u8(d, a, bl.x, bl.y, d);                             // P = 256 P_hi + P_lo, exact
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[n][e] = mad_wide(d[e] * (1 << sh[n][e & 1]), e < 2 ? s_g : s_g8, acc[n][e]);
    }
}

// ---- dp4a back end, one row-block, R tokens --------------------------------------------------
// The lane owns (row, block); hi and lo point at the token's 32-byte records of this block, read as
// eight u32 in quad order (the same buffer as the B fragments).
template <int R>
__device__ __forceinline__ void dp4a_block(const u32 q[6], int s_int, const u32* hi, const u32* lo,
                                           u32 tok_stride_u32, const u32 sh[R], long long acc[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const u32* h = hi + r * tok_stride_u32;
        const u32* l = lo + r * tok_stride_u32;
        int p = 0;
#pragma unroll
        for (int i = 0; i < 6; ++i) p = __dp4a((int)q[i], (int)h[i], p);
        p *= 256;
#pragma unroll
        for (int i = 0; i < 6; ++i) p = dp4a_su(q[i], l[i], p);
        acc[r] = mad_wide(p * (1 << sh[r]), s_int, acc[r]);
    }
}

// ---- decoder input ---------------------------------------------------------------------------
// The served v3 decoder runs unchanged (the 2026-08-05 setting "decodage gele jusqu'au SASS" holds);
// its tables stay in global memory and are read through L1, as tv_tetra48_h does. Staging them in
// shared memory per CTA would move 1.36 GB a pass over 69,120 CTAs (calc/brick/geometry_budget.out).
// Its biased bytes (val + 128) become signed int8 quads by one XOR at use:
__device__ __forceinline__ u32 to_signed_quad(u32 biased) { return biased ^ 0x80808080u; }

// beta0 only (prereg-brick-0.md): the trivial decoder. Byte i of the 48-bit word, replicated into
// quad i, S = 1. It costs a few PRMT where the real decoder costs ~55 lane-ops and 6 gathers, so the
// arm times the geometry, the stream, the MMAs and the combine, not the decode. Integer, so the host
// replays it exactly.
__device__ __forceinline__ void trivial_decode(u32 lo32, u32 hi16, u32 q[6]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = __byte_perm(lo32, 0u, (u32)(i * 0x1111));  // byte i x4
    q[4] = __byte_perm(hi16, 0u, 0x0000u);
    q[5] = __byte_perm(hi16, 0u, 0x1111u);
}

// ---- combine and epilogue --------------------------------------------------------------------
// In-CTA: warps add their int64 partials into shared slots in any order (integer, associative).
// Across CTAs, self-cleaning, so no memset launch joins the 1,410 launches a token:
//   1. every CTA of a row group adds its partials with red.global.add.u64 into `gacc`, a scratch that
//      persists across launches and is zeroed ONCE at allocation;
//   2. __threadfence(), then one thread takes a ticket on the row group's counter;
//   3. the CTA that draws Cx - 1 is last: it reads the sums through L2 (ld.global.cg, never L1), runs
//      the epilogue, then stores 0 to every slot it read and to the counter.
// The next launch, ordered by the stream, starts from zeros. A missed reset changes the next launch's
// bits, so the load-time check (two launches, equal bits) and beta0's gate 3 (round r = round 1) see it;
// the host emulator carries the mutant "last CTA skips the clean". Arrival order cannot move a bit.
__device__ __forceinline__ void cross_cta_add(long long* gacc, long long v) {
    atomicAdd(reinterpret_cast<u64*>(gacc), (u64)v);   // two's complement wraps exactly
}

// Called by every thread of the CTA after its partials are added. Returns true in the last CTA.
__device__ __forceinline__ bool last_cta(unsigned* counter, unsigned cx) {
    __shared__ unsigned ticket;
    __threadfence();                                   // this CTA's red.adds before its ticket
    __syncthreads();
    if (threadIdx.x == 0) ticket = atomicAdd(counter, 1u);
    __syncthreads();
    return ticket == cx - 1u;
}

// In the last CTA: thread t owns slots t, t + blockDim.x, ...; read, epilogue, then clean.
__device__ __forceinline__ long long take_and_clear(long long* gacc_slot) {
    long long v = __ldcg(gacc_slot);                   // through L2: the other CTAs' atomics live there
    *gacc_slot = 0;                                     // self-cleaning
    return v;
}
__device__ __forceinline__ void reset_counter(unsigned* counter) {
    __syncthreads();
    if (threadIdx.x == 0) *counter = 0u;               // stream order publishes it to the next launch
}

// One output, one thread, fixed order. e = e_lo(token) + e_s(matrix); pow2 built from bits.
__device__ __forceinline__ float epilogue(long long I, int e, float rscale,
                                          const unsigned short* tail, const float* xtail, u32 tail_w,
                                          float (*h2f)(unsigned short)) {
    float u = __ll2float_rn(I);                                   // cvt.rn.f32.s64
    float v = __fmul_rn(u, __uint_as_float((u32)(e + 127) << 23)); // exact, e in [-126, 127]
    float tv = 0.0f;
    for (u32 i = 0; i < tail_w; ++i) tv = __fmaf_rn(h2f(tail[i]), xtail[i], tv);
    return __fmaf_rn(v, rscale, tv);                              // then f2h by the caller
}
