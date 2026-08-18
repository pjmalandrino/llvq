// The E1v block decoder — the CNS's combinatorial stream, row-aligned.
//
// Arm `e1v` of `llvq-cuda/src/arms.rs`. Format: `llvq_artifact::e1v`, the cut
// produced by `transcode_e1v_rows` and **only** that one.
//
// ## What this decodes, and why it is not another Planes
//
// Every other layout in this crate stores a per-slot level index and a sign,
// and its decoder is a select. E1v stores **combinatorial ranks**: a Golay
// codeword index, one rank per kind of two arrangements, and two sign fields.
// Rebuilding a block therefore costs a codeword lookup, two binomial walks, the
// parity repair and three sign rules — the work `llvq-search/src/cns.rs`
// specifies and `llvq-metal/shaders/binomial_block_flat.metal` already runs on
// another GPU.
//
// What it buys is size: **2,3983 b/poids noyau** on the sealed 4B, measured on
// written bytes (`llvq-artifact/tests/e1v_format.rs`), against 4,804 for the
// served `Planes14`. Whether the walks cost less than the bytes they save is
// the bench's question and not this file's.
//
// ## The addressing, which is the whole difference from every other arm here
//
// A group is 32 blocks and **never straddles a row** — X3 showed the served
// matvec puts one warp per row, so a group cut in file order would be read by
// two warps and neither would read one group
// (`docs/mesures/x3-alignement-warp-2026-08-15.txt`). The last group of a row
// is therefore **partial**, holding `k = min(32, nblocks − 32·g)` blocks.
//
//     bases[g]                    first word of group g
//     bit 10·l                    block l's header: class 9 | gain 1
//     bit 10·k + Σ widths[<l]     block l's payload, width from its class
//
// `k` is derived, never stored. A lane finds its payload by an **exclusive
// prefix sum over the group's 32 payload widths** — the one operation no other
// decoder in this crate performs.
//
// 🚨 **The scan makes every lane load-bearing, and that constrains the caller.**
// `__shfl_up_sync` at full mask requires all 32 lanes to participate, so the
// serving loop may NOT be `planes.cu`'s `for (j = jlo + lane; j < jhi; j += 32)`
// — that lets the lanes past the end exit early, and a partial group has such
// lanes by construction. Every lane must reach `e1v_dot` and carry a predicate
// instead — the `have` argument.
//
// ## No dynamic indexing
//
// Same rule as `llvq_planes.cuh` and `llvq_e1c.cuh`, and it is the reason this
// file transcribes the **flat** Metal arm rather than the one P1c measured.
// `binomial_block.metal` keeps `kinds_on[24]` and `kinds_off[24]` indexed **by
// slot**, i.e. by data: on this hardware that is local memory on the hottest
// path, and `local_size_bytes() == 0` is a contract this crate refuses to
// break. The flat arm deposits each kind straight from its `taken` mask and
// recovers both sign ranks by popcount. One array survives — `tk`, the per-kind
// masks — and it is indexed by a **loop counter** over a compile-time bound,
// so `#pragma unroll` keeps it in registers.
//
// ⚠️ Consequence to state rather than discover: the Metal arm P1c times uses
// the OTHER body, and on Metal the flat one measured **24 % slower**
// (0,8346 against 0,6704 ns/bloc, P1b É1). A CUDA figure from this file is not
// comparable to P1c's, and no journal may subtract them.
//
// ## What has been checked, and where
//
// Three checks, on a machine with no card, and each says less than the next:
//
//   * it **parses** — `bin/cuhcheck`, every unit of this directory;
//   * `e1v_decode_at` is **executed**, this very text, against the Rust decoder
//     on every class plus the origin — `tests/host_e1v.cpp` and
//     `tests/e1v_decoder_matches_rust.rs`, the pattern the four other
//     `*_decoder_matches_rust.rs` established;
//   * a Rust **mirror** re-derives the addressing, scan included, and checks it
//     slot by slot — `llvq-artifact/tests/e1v_cuda_mirror.rs`.
//
// 🚨 What none of them touches is the **warp scan**. `__shfl_up_sync` has no
// host semantics — `tests/host_shim.h` says so of every shuffle in this crate —
// so `e1v_scan_exclusive` and `e1v_dot` are compile-only here, and the mirror's
// serial prefix sum is a statement about the value, not about the primitive.
// Occupancy, register allocation, spill and races are the card's to answer,
// which is what `preflight` is for.

#ifndef LLVQ_E1V_CUH
#define LLVQ_E1V_CUH

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

// Blocks per addressing group — the warp, and the reason rows are aligned.
#define LLVQ_E1V_GROUP 32u
// Header bits per block: class 9 + gain 1, the same split every record of this
// repo uses.
#define LLVQ_E1V_HDR_BITS 10u
// The origin's class field: all nine bits set, a value no class takes.
#define LLVQ_E1V_ORIGIN_ID 511u
// Kinds an arrangement may have. Mirrors `MAX_KINDS` in `llvq-search`.
#define LLVQ_E1V_KINDS 8u
// Row stride of the transposed binomial table, `bt[c·25 + p] = C(p, c)`.
#define LLVQ_E1V_BINOM_STRIDE 25u

// Mirror of `llvq_metal::p1host::GpuBlockRec`, 108 bytes, alignment 4 — the
// same host record the Metal arms read, uploaded unchanged. The host asserts
// the size; a mirror that drifted would decode plausible wrong weights.
struct E1vRec {
    float vals_on[LLVQ_E1V_KINDS];   // support values (even) / class values (odd),
                                     // ALREADY divided by sqrt(16·m). On the odd
                                     // coset the SIGN BIT carries bit 1 of the
                                     // integer |value| — the forced-sign rule.
    float vals_off[LLVQ_E1V_KINDS];  // off-support values; the zero run's is 0.0f,
                                     // and that zero is what marks it
    unsigned char cnt_on[LLVQ_E1V_KINDS];
    unsigned char cnt_off[LLVQ_E1V_KINDS];
    unsigned char wb_on[LLVQ_E1V_KINDS];   // ceil(log2 radix) per kind, 0 for the last
    unsigned char wb_off[LLVQ_E1V_KINDS];
    unsigned char k_on, k_off, w, geom;    // geom: p_req | odd<<1
    u32 golay_base;
    unsigned char wb_golay, wb_sw, wb_sf, pad;
};

// ---------------------------------------------------------------------------
// Reading the stream
// ---------------------------------------------------------------------------

// A field of at most 32 bits at an arbitrary bit offset: two words, because a
// window that starts anywhere in a word ends inside the next one.
__device__ __forceinline__ u32 e1v_read_small(const u32* __restrict__ data,
                                              u64 bit,
                                              u32 width)
{
    u32 w  = (u32)(bit >> 5);
    u32 sh = (u32)(bit & 31u);
    u64 two = (u64)data[w] | ((u64)data[w + 1u] << 32);
    return (u32)((two >> sh) & (((u64)1 << width) - 1));
}

// The payload window: 96 bits from `bit`, shifted so `bit` becomes bit 0 of the
// pair `(lo, hi)`. THREE words — the widest payload is 46 bits (class 323: a
// 56-bit record less its 10-bit header) and 46 bits from offset 31 end in the
// third. 96 − 31 = 65 bits are readable at every offset, and the host asserts
// no class asks for more.
//
// The stream buffer carries two words of padding past the last group so this
// read cannot fault. Per the repo's convention that padding is not counted in
// an arm's bytes.
__device__ __forceinline__ void e1v_window(const u32* __restrict__ data,
                                           u64 bit,
                                           u64& lo,
                                           u32& hi)
{
    u32 w  = (u32)(bit >> 5);
    u32 sh = (u32)(bit & 31u);
    lo = (u64)data[w] | ((u64)data[w + 1u] << 32);
    hi = data[w + 2u];
    // A shift by 64 is undefined, so the aligned case is named rather than
    // reached by arithmetic.
    if (sh != 0u) {
        lo = (lo >> sh) | ((u64)hi << (64u - sh));
        hi = hi >> sh;
    }
}

// Read `width` bits at `off` of the window, without moving a cursor.
// Transcribed from `binomial_block_flat.metal`'s `peek`, minus its `10 +`
// header term: E1v's header is in the group prefix, not in the payload.
//
// 🚨 **And that missing term is exactly why `off == 0` needs its own branch
// here while the Metal original needs none.** That file says so in its own
// comment — *"`off` is always at least 10 — the header precedes every field —
// so the `64 - off` shift below is always defined"* — and the guarantee is
// true there and FALSE here: E1v's first payload field sits at offset zero, so
// `(u64)hi << (64 - 0)` is a shift by 64, which is undefined in C++ and which
// NVIDIA hardware answers by shifting modulo 64, i.e. by not shifting at all.
// The garbage would land in the low bits of the very first field read — the
// Golay index — on every block of every class whose codeword bucket is wider
// than one entry.
//
// A transcription that carried the body across without carrying the *reason*
// for its guard would have reproduced this silently.
__device__ __forceinline__ u32 e1v_peek(u64 lo, u32 hi, u32 off, u32 width) {
    if (width == 0u) return 0u;
    u64 v;
    if (off == 0u) {
        v = lo;
    } else if (off >= 64u) {
        v = (u64)hi >> (off - 64u);
    } else {
        v = (lo >> off) | ((u64)hi << (64u - off));
    }
    return (u32)(v & (((u64)1 << width) - 1));
}

// The exclusive prefix sum over the warp's 32 payload widths — the operation
// that makes this arm what it is.
//
// 🚨 Every lane must call it, including a partial group's empty lanes, which
// contribute a width of zero. `__shfl_up_sync` at full mask with a lane missing
// is undefined; that is why the serving loop cannot let lanes exit early.
__device__ __forceinline__ u32 e1v_scan_exclusive(u32 width) {
    u32 acc = width;
#pragma unroll
    for (u32 d = 1u; d < LLVQ_E1V_GROUP; d <<= 1) {
        u32 n = __shfl_up_sync(0xffffffffu, acc, d);
        if ((threadIdx.x & 31u) >= d) acc += n;
    }
    return acc - width; // inclusive → exclusive
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

// One kind's slots, by colex unranking over the free positions, walked in SLOT
// space. **Byte-identical in arithmetic to `binomial_block_flat.metal`'s
// `walk_kind`** — same snapshot discipline, same transposed row, same
// compare-and-subtract, no division. Returns the mask of slots this kind takes.
__device__ __forceinline__ u32 e1v_walk_kind(u32 rank,
                                             u32 cnt,
                                             u32 freem,
                                             const u32* __restrict__ binom)
{
    const u32* row = binom + cnt * LLVQ_E1V_BINOM_STRIDE;
    u32 taken = 0u;
    u32 p = __popc(freem);
    u32 m = freem;
    while (m != 0u) {
        u32 s = 31u - __clz(m);
        m ^= 1u << s;
        --p;
        u32 b = row[p];
        bool hit = (rank >= b) && (cnt != 0u);
        u32 one = hit ? 1u : 0u;
        rank  -= hit ? b : 0u;
        row   -= one * LLVQ_E1V_BINOM_STRIDE;
        cnt   -= one;
        taken |= hit ? (1u << s) : 0u;
    }
    return taken;
}

// ---------------------------------------------------------------------------
// One block's contribution to a dot product
// ---------------------------------------------------------------------------
//
// Four independent FMA chains and the same `gscale` epilogue as `planes_dot`,
// so the arms' outputs are comparable against the shared f64 reference. They
// are **not** asserted bit-equal to each other: two kernels, and the compiler
// may contract differently.
// 🚨 **The decode, split out of `e1v_dot` so it can be EXECUTED on a machine
// with no card.** It contains no warp primitive, no shared memory and no
// atomic — it is bit arithmetic over registers, which is where every bug this
// repository has actually had lives. `tests/host_e1v.cpp` runs *this text*
// against the Rust decoder; the scan around it stays compile-only, exactly as
// `host_shim.h` says of every barrier and shuffle in this crate.
//
// Everything the lane has already learnt is passed in: its group's base bit,
// the group's occupancy, its own header fields and its payload offset. Those
// are the four things the scan produces, and none of them is recomputed here.
__device__ __forceinline__ float e1v_decode_at(const u32* __restrict__ data,
                                               const E1vRec* __restrict__ tab,
                                               const u32* __restrict__ binom,
                                               const u32* __restrict__ golay,
                                               const float* __restrict__ gscale,
                                               u64 gbit,
                                               u32 len,
                                               u32 id,
                                               u32 gain,
                                               u32 off,
                                               const float* xb)
{
    u64 lo; u32 hi;
    e1v_window(data, gbit + (u64)(LLVQ_E1V_HDR_BITS * len) + (u64)off, lo, hi);

    // The stream's id convention is not the table's: 0..382 for a class and 511
    // for the origin, against entry 0 for the origin and 1+ci for class ci.
    const E1vRec r = tab[(id == LLVQ_E1V_ORIGIN_ID) ? 0u : id + 1u];
    u32  w     = r.w;
    bool odd   = ((r.geom >> 1) & 1u) != 0u;
    u32  p_req = r.geom & 1u;

    // ---- the two sign fields, peeked ahead -------------------------------
    u32 pre = (u32)r.wb_golay;
#pragma unroll
    for (u32 k = 0u; k < LLVQ_E1V_KINDS; ++k) pre += (u32)r.wb_on[k] + (u32)r.wb_off[k];
    u32 r_sw = e1v_peek(lo, hi, pre, r.wb_sw);
    u32 r_sf = e1v_peek(lo, hi, pre + (u32)r.wb_sw, r.wb_sf);
    // The parity repair is the XOR of every free support bit, and `s_w` is
    // exactly 2^(w−1), so `r_sw` holds those bits and nothing else: the
    // accumulator the slot-indexed arm carries across the loop is one popcount.
    u32 par = (u32)__popc(r_sw) & 1u;

    u32 cur = 0u;                                   // bit cursor into the payload
    u32 gi  = e1v_peek(lo, hi, cur, r.wb_golay); cur += (u32)r.wb_golay;
    u32 cw  = golay[r.golay_base + gi] & 0xffffffu;

    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;

    // ---- the support (all 24 slots on the odd coset) ----------------------
    {
        u32 freem = odd ? 0xffffffu : cw;
#pragma unroll
        for (u32 k = 0u; k < LLVQ_E1V_KINDS; ++k) {
            if (k >= r.k_on) break;
            u32 taken;
            if (k + 1u == (u32)r.k_on) {
                taken = freem;
                freem = 0u;
            } else {
                u32 rank = e1v_peek(lo, hi, cur, r.wb_on[k]); cur += (u32)r.wb_on[k];
                u32 c = r.cnt_on[k];
                if (c == 0u) continue;
                taken = e1v_walk_kind(rank, c, freem, binom);
                freem ^= taken;
            }
            float v   = r.vals_on[k];
            float mag = fabsf(v);
            u32   flag = (u32)__float_as_int(v) >> 31;  // bit 1 of |value|, odd coset
            u32 m = taken;
            while (m != 0u) {
                u32 s = (u32)__ffs((int)m) - 1u;
                m &= m - 1u;
                u32 neg;
                if (odd) {
                    neg = ((cw >> s) & 1u) ^ flag;
                } else {
                    u32 s_on = (u32)__popc(cw & ((1u << s) - 1u));
                    neg = (s_on + 1u == w) ? (par ^ p_req) : ((r_sw >> s_on) & 1u);
                }
                float wv = __int_as_float(__float_as_int(mag) ^ (int)(neg << 31));
                float xv = xb[s];
                if ((s & 3u) == 0u) dot0 = fmaf(wv, xv, dot0);
                else if ((s & 3u) == 1u) dot1 = fmaf(wv, xv, dot1);
                else if ((s & 3u) == 2u) dot2 = fmaf(wv, xv, dot2);
                else dot3 = fmaf(wv, xv, dot3);
            }
        }
    }

    // ---- the off-support, even coset only ---------------------------------
    //
    // Two loops, and the split is the whole point: the first walks every kind
    // and unions the nonzero ones into `nz`, the second deposits. A single loop
    // cannot, because a slot's sign rank counts the nonzero slots BEFORE it and
    // a later kind can own one of those.
    if (!odd) {
        u32 tk[LLVQ_E1V_KINDS];
        u32 nz = 0u;
        u32 freem = (~cw) & 0xffffffu;
#pragma unroll
        for (u32 k = 0u; k < LLVQ_E1V_KINDS; ++k) tk[k] = 0u;
#pragma unroll
        for (u32 k = 0u; k < LLVQ_E1V_KINDS; ++k) {
            if (k >= r.k_off) break;
            u32 taken;
            if (k + 1u == (u32)r.k_off) {
                taken = freem;
                freem = 0u;
            } else {
                u32 rank = e1v_peek(lo, hi, cur, r.wb_off[k]); cur += (u32)r.wb_off[k];
                u32 c = r.cnt_off[k];
                if (c == 0u) continue;
                taken = e1v_walk_kind(rank, c, freem, binom);
                freem ^= taken;
            }
            tk[k] = taken;
            // The zero run is marked by its VALUE, never by its index.
            if (r.vals_off[k] != 0.0f) nz |= taken;
        }
#pragma unroll
        for (u32 k = 0u; k < LLVQ_E1V_KINDS; ++k) {
            if (k >= r.k_off) break;
            float v = r.vals_off[k];
            if (v == 0.0f) continue;    // the zero run deposits nothing
            u32 m = tk[k];
            while (m != 0u) {
                u32 s = (u32)__ffs((int)m) - 1u;
                m &= m - 1u;
                u32 fbit = (u32)__popc(nz & ((1u << s) - 1u));
                u32 neg  = (r_sf >> fbit) & 1u;
                float wv = __int_as_float(__float_as_int(v) ^ (int)(neg << 31));
                float xv = xb[s];
                if ((s & 3u) == 0u) dot0 = fmaf(wv, xv, dot0);
                else if ((s & 3u) == 1u) dot1 = fmaf(wv, xv, dot1);
                else if ((s & 3u) == 2u) dot2 = fmaf(wv, xv, dot2);
                else dot3 = fmaf(wv, xv, dot3);
            }
        }
    }

    return ((dot0 + dot1) + (dot2 + dot3)) * gscale[gain];
}

// One block's contribution to a dot product: the addressing, then the decode.
//
// `row`/`j` are the block's row and its index within that row — the two
// quantities the serving kernel already holds (`planes.cu`: `b0r = row·nblocks`,
// `j` the loop variable), so no divide is reintroduced here.
//
// `have` is the lane's predicate: false for a lane past the end of a partial
// group. Such a lane still calls this function and still joins the scan with a
// width of zero — that is the whole reason the predicate exists rather than an
// early `continue` in the caller.
__device__ __forceinline__ float e1v_dot(const u32* __restrict__ data,
                                         const u32* __restrict__ bases,
                                         const E1vRec* __restrict__ tab,
                                         const u32* __restrict__ pay,
                                         const u32* __restrict__ binom,
                                         const u32* __restrict__ golay,
                                         const float* __restrict__ gscale,
                                         u32 row,
                                         u32 j,
                                         u32 row_blocks,
                                         bool have,
                                         const float* xb)
{
    u32 per_row = (row_blocks + LLVQ_E1V_GROUP - 1u) / LLVQ_E1V_GROUP;
    u32 gl      = j / LLVQ_E1V_GROUP;          // group within the row
    u32 lane    = j - gl * LLVQ_E1V_GROUP;
    u32 len     = row_blocks - gl * LLVQ_E1V_GROUP;
    if (len > LLVQ_E1V_GROUP) len = LLVQ_E1V_GROUP;
    u64 gbit = (u64)bases[row * per_row + gl] * 32;

    // The header, at its fixed stride — the only field readable before anything
    // else is known. An absent lane reads nothing and declares a zero width, so
    // the scan stays a full-warp operation.
    u32 hdr  = have ? e1v_read_small(data, gbit + LLVQ_E1V_HDR_BITS * lane,
                                     LLVQ_E1V_HDR_BITS)
                    : 0u;
    u32 id   = have ? (hdr & 0x1ffu) : LLVQ_E1V_ORIGIN_ID;
    u32 gain = hdr >> 9;

    u32 off = e1v_scan_exclusive(have ? pay[id] : 0u);
    if (!have) return 0.0f;

    return e1v_decode_at(data, tab, binom, golay, gscale, gbit, len, id, gain, off, xb);
}

#endif // LLVQ_E1V_CUH
