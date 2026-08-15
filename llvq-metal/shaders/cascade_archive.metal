// =============================================================================
// cascade_archive.metal — the `cascade-archive` arm of P1
//
// Pre-registration: proofs/preregistration-p1-2026-08-13.md (§2, arm 3).
// CPU reference:    llvq-search/src/fastdec.rs::unrank_multiset (:167), which
//                   is `FastDecoder::decode`'s own inner loop, made public for
//                   exactly this reason (§2, the API debt, resolved as
//                   "export, do not duplicate").
//
// This is **the incumbent**, ported to MSL as it is: the recurrence
// `M' = M·c_j/n` with a real 64-bit division, a `continue` on a spent kind, a
// `break` when the slot is decided, and an outer trip count of `out.len()` —
// 24 on the odd coset, `w` then `24 − w` on the even one. Nothing here is
// uniformised. That is the point: `cascade_uniform.metal` differs from this
// file **only** in the shape of the work, reads the same table and the same
// bits, so the gap between the two arms measures the shape and nothing else.
//
// -----------------------------------------------------------------------------
// 1. WHY THIS ARM IS NOT DECORATION — it can close the whole line
// -----------------------------------------------------------------------------
// §4.3 of the pre-registration: **if this arm returns ≤ 2,0 ns/block — the
// threshold imposed on the uniformised cascade — then E1v is stillborn.** The
// archive exists, it is proven on 150 681 600 blocks, it is *smaller*
// (2,19 b/weight kernel-side against 2,3709 for E1v), and it would have just
// cleared the bar set for a decoder that would still have to be written,
// proven and transcoded to. This is the only arm of the five that can make the
// other four pointless, and it was nearly written as a control.
//
// -----------------------------------------------------------------------------
// 2. WHAT IT SHARES WITH THE UNIFORMISED ARM, AND WHAT IT MUST NOT
// -----------------------------------------------------------------------------
// Shared, deliberately: the `CascadeRec` table (`p1host::cascade_records`), the
// `ends` table and its nine-step class search, the Golay codewords, the field
// order of the mixed-radix peel, and the three sign rules. A difference in any
// of those would make the two arms incomparable, and the journal would be
// reporting the cost of a table layout rather than the cost of a loop.
//
// NOT shared, and this is the arm:
//   * the peel divides for real (`/`, `%`) instead of multiplying by a magic
//     reciprocal — `magic_on` / `magic_off` / `dflags` are ignored here;
//   * the candidate loop `break`s as soon as the slot is decided and
//     `continue`s past a kind with count zero, so its trip count is data;
//   * the even coset is two loops of `w` and `24 − w` steps, not one loop of 24
//     with a state swap;
//   * `counts` is a register array indexed by the chosen kind. On a GPU that is
//     spilled to thread-local memory, and that spill is part of what this arm
//     costs. `cascade_uniform` packs the eight counts into one `ulong` to avoid
//     it (its §1(d)); doing the same here would be uniformising the incumbent,
//     which would delete the comparison.
//
// No timing, no threshold, no claim. The nanoseconds are the bench's business.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint DIM        = 24;
constant uint MAX_KINDS  = 8;
constant uint N_CLASSES  = 383;
constant uint CLASS_TOP  = 256;   // p1host::CLASS_PAD / 2, the search's top bit

// Byte-for-byte the mirror of `p1host::GpuCascadeRec`, 88 bytes. Declared here
// rather than shared with the other shader because each `.metal` file is
// compiled on its own; the Rust side asserts the size once, for both.
struct CascadeRec {
    ulong offset;
    ulong d_on;
    ulong d_off;
    ulong magic_on;    // unused by this arm — see §2
    ulong magic_off;   // unused by this arm
    ulong counts_a;
    ulong counts_b;
    ulong vals_a;
    ulong vals_b;
    float inv_norm;
    uint  golay_base;
    uint  geom;
    uint  dflags;
};

// ---------------------------------------------------------------------------
// The incumbent's unranking, in MSL. `unrank_fast` (fastdec.rs:171) line for
// line: one multiply and one divide per candidate, exact because `n` divides
// `m·c_j` at every step.
//
// `counts` is a thread-local array, written back after each slot. That is the
// dynamic register indexing §2 says not to remove.
// ---------------------------------------------------------------------------
static inline void unrank_archive(ulong rank,
                                  thread uchar* counts,
                                  uint k,
                                  ulong m0,
                                  thread uchar* out,
                                  uint n_slots)
{
    ulong m = m0;
    ulong n = ulong(n_slots);
    for (uint slot = 0u; slot < n_slots; ++slot) {
        for (uint j = 0u; j < k; ++j) {
            ulong c = ulong(counts[j]);
            if (c == 0ul) continue;
            ulong mj = m * c / n;
            if (rank < mj) {
                out[slot] = uchar(j);
                counts[j] -= 1;
                m = mj;
                break;
            }
            rank -= mj;
        }
        n -= 1ul;
    }
}

// One block: archive index in, dot product out. Same contract as
// `cascade_uniform`'s `cascade_block`, same output, same scale.
static inline float archive_block(ulong idx,
                                  uint  gain,
                                  device const ulong*      ends,
                                  device const CascadeRec* classes,
                                  device const uint*       golay,
                                  constant float*          gscale,
                                  threadgroup const float* xs)
{
    // The class search is the uniform one on purpose: it is not what this arm
    // is about, and giving the two arms different searches would put a second
    // difference into a comparison meant to isolate one.
    bool  live = (idx != 0ul);
    ulong i    = live ? (idx - 1ul) : 0ul;

    uint ci = 0u;
    for (uint b = CLASS_TOP; b != 0u; b >>= 1) {
        ci = (ends[ci + b - 1u] <= i) ? (ci + b) : ci;
    }
    ci = min(ci, N_CLASSES - 1u);
    device const CascadeRec& r = classes[ci];

    ulong local = i - r.offset;
    uint  geom  = r.geom;
    uint  w     = geom & 0xffu;
    uint  p_req = (geom >> 8) & 1u;
    bool  odd   = ((r.dflags >> 18) & 1u) != 0u;

    float dot = 0.0f;

    if (odd) {
        // One 24-slot problem, indexed by the coordinate itself.
        ulong r_arr = local % r.d_on;
        ulong gi    = local / r.d_on;
        uint  cw    = golay[gi] & 0xffffffu;

        uchar counts[MAX_KINDS];
        for (uint j = 0u; j < MAX_KINDS; ++j) counts[j] = uchar((r.counts_a >> (8u * j)) & 0xfful);
        uchar kinds[DIM];
        unrank_archive(r_arr, counts, MAX_KINDS, r.d_on, kinds, DIM);

        for (uint s = 0u; s < DIM; ++s) {
            uint v   = uint((r.vals_a >> (8u * uint(kinds[s]))) & 0xfful);
            uint neg = ((cw >> s) & 1u) ^ ((v >> 1) & 1u);
            float wv = as_type<float>(as_type<uint>(float(v)) ^ (neg << 31u));
            dot = fma(wv, xs[s], dot);
        }
    } else {
        // Four fields peeled off the local rank, in the archive's order.
        uint sf_shift = (geom >> 16) & 0xffu;
        uint sw_shift = (geom >> 24) & 0xffu;
        uint r_sf = uint(local & ((1ul << sf_shift) - 1ul));
        local >>= sf_shift;
        uint r_sw = uint(local & ((1ul << sw_shift) - 1ul));
        local >>= sw_shift;
        ulong r_off = local % r.d_off;
        local /= r.d_off;
        ulong r_on  = local % r.d_on;
        ulong gi    = local / r.d_on;
        uint  cw    = golay[r.golay_base + uint(gi)] & 0xffffffu;

        uchar on_counts[MAX_KINDS], off_counts[MAX_KINDS];
        for (uint j = 0u; j < MAX_KINDS; ++j) {
            on_counts[j]  = uchar((r.counts_a >> (8u * j)) & 0xfful);
            off_counts[j] = uchar((r.counts_b >> (8u * j)) & 0xfful);
        }
        uchar on_kinds[DIM], off_kinds[DIM];
        unrank_archive(r_on,  on_counts,  MAX_KINDS, r.d_on,  on_kinds,  w);
        unrank_archive(r_off, off_counts, MAX_KINDS, r.d_off, off_kinds, DIM - w);

        // The zero run is the last off-support kind, and it is marked by its
        // value byte being zero — the same convention `cascade_uniform` reads.
        uint n_off = 0u;
        for (uint j = 0u; j < MAX_KINDS; ++j) {
            if (((r.counts_b >> (8u * j)) & 0xfful) != 0ul) n_off = j + 1u;
        }
        uint zero_kind = n_off - 1u;

        uint s_on = 0u, s_off = 0u, par = 0u, fbit = 0u;
        for (uint s = 0u; s < DIM; ++s) {
            float wv = 0.0f;
            if ((cw >> s) & 1u) {
                uint v = uint((r.vals_a >> (8u * uint(on_kinds[s_on]))) & 0xfful);
                uint bfree = (r_sw >> s_on) & 1u;
                uint neg;
                if (s_on + 1u == w) {
                    neg = par ^ p_req;
                } else {
                    neg = bfree;
                    par ^= bfree;
                }
                wv = as_type<float>(as_type<uint>(float(v)) ^ (neg << 31u));
                s_on += 1u;
            } else {
                uint kd = uint(off_kinds[s_off]);
                if (kd != zero_kind) {
                    uint v   = uint((r.vals_b >> (8u * kd)) & 0xfful);
                    uint neg = (r_sf >> fbit) & 1u;
                    wv = as_type<float>(as_type<uint>(float(v)) ^ (neg << 31u));
                    fbit += 1u;
                }
                s_off += 1u;
            }
            dot = fma(wv, xs[s], dot);
        }
    }

    float scale = r.inv_norm * gscale[gain];
    return dot * (live ? scale : 0.0f);
}

// ---------------------------------------------------------------------------
// The arm. Same signature as `cascade_uniform` minus the `DivTab` it does not
// read, so the two are bound the same way and fed the same buffers.
//
//   buffer(0) idx      device const ulong*      (idx << 1) | gain, per block
//   buffer(1) ends     device const ulong*      cumulative class ends, 512 padded
//   buffer(2) classes  device const CascadeRec* 383 records
//   buffer(3) golay    device const uint*       the 4096 codewords
//   buffer(4) gscale   constant float*          gain centroids
//   buffer(5) x        device const float*      24 activations
//   buffer(6) out      device float*            one float per block
//
// ⚠️ The gain rides in bit 0 of the index word here, where `cascade_uniform`
// takes it from a separate byte buffer. That is a difference in the *feed*, not
// in the arm, and it exists because §4.2 of the bench plan fixes one 8-byte
// word per block for both rank arms — the same buffer, so the comparison
// between them is bit-for-bit fair. The bench binds this word to both.
// ---------------------------------------------------------------------------
kernel void cascade_archive(device const ulong*      idx     [[buffer(0)]],
                            device const ulong*      ends    [[buffer(1)]],
                            device const CascadeRec* classes [[buffer(2)]],
                            device const uint*       golay   [[buffer(3)]],
                            constant float*          gscale  [[buffer(4)]],
                            device const float*      x       [[buffer(5)]],
                            device float*            out     [[buffer(6)]],
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong word = idx[gid];
    out[gid] = archive_block(word >> 1, uint(word & 1ul), ends, classes, golay, gscale, xs);
}
