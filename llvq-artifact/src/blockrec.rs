//! **The per-class record a CNS block decoder needs — on either GPU.**
//!
//! A decoder of the E1v stream, or of any of P1's block arms, needs more than
//! the level table [`crate::runtime::ClassTable`] carries: the off-support
//! kinds, the Golay bucket, the parity requirement, and the width of every
//! field of the record. That is what lives here.
//!
//! ## Why it is in this crate rather than beside a shader
//!
//! 🕳️ It was in `llvq_metal::p1host`, which is the crate that talks to Metal —
//! and `llvq-cuda` cannot see that crate. The day a second GPU needed the same
//! table, the choice was to duplicate it or to move it, and this repository has
//! already paid for a duplicated constant twice (`TILE_BLOCKS` in
//! `bin/planesbench` alone, and before that `TILE_BLOCKS` "defined twice,
//! unlinked" on the Metal side, which `matvec.cu` still complains about in its
//! own header).
//!
//! This crate is the right home for the same reason it already hosts
//! [`crate::runtime::ClassTable`]: it is the format crate, it has no external
//! dependency, and both GPU crates depend on it. `llvq_metal::p1host`
//! re-exports every item below, so the Metal side reads exactly as it did.
//!
//! ## Two mirrors, one record
//!
//! [`GpuBlockRec`] is a byte-for-byte mirror of **two** declarations that must
//! not drift from it: `struct BlockRec` in `llvq-metal/shaders/
//! binomial_block.metal` (and its flat twin, and `e1v_flux.metal`), and
//! `struct E1vRec` in `llvq-cuda/kernels/llvq_e1v.cuh`. The size is asserted at
//! build time here; the field order is asserted by each side's V0.

use crate::e1v::{E1V_GROUP, E1V_ORIGIN_ID};
use llvq_core::{Golay, DIM};
use llvq_search::cns::cns_layout;
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::binom;

pub const BINOM_STRIDE: usize = DIM + 1;

///
/// The transpose is what lets the shader hold `c` fixed and walk `p` downward
/// at a stride of one word, decrementing the row pointer by 25 when a unit is
/// placed. The zeros for `c > p` are load-bearing, not a guard: they are how a
/// kind that must take every remaining position gets placed.
///
/// Narrowed from the crate's one `u64` table with a checked cast rather than
/// re-derived — the largest entry is `C(24,12) = 2 704 156`.
pub fn binom_table() -> Vec<u32> {
    let mut t = vec![0u32; BINOM_STRIDE * BINOM_STRIDE];
    for c in 0..BINOM_STRIDE {
        for p in 0..BINOM_STRIDE {
            t[c * BINOM_STRIDE + p] =
                u32::try_from(binom(p, c)).expect("C(p,c) fits u32 for p <= 24");
        }
    }
    t
}

/// Entries of the block table: the origin, then one per class — the id
/// convention of every table in this workspace.
pub const BLOCK_ENTRIES: usize = 384;

/// `ceil(log2 x)` for `x >= 1`.
///
/// Public because `llvq_metal::p1host` builds the cascade and walk tables from
/// the same widths, and a second copy of this is a second thing to get wrong.
#[inline]
pub fn ceil_log2(x: u64) -> u32 {
    assert!(x >= 1, "log2 of zero");
    if x == 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}
/// Start of each Hamming weight's bucket in [`Golay::codewords`].
///
/// `of_weight(w)` is a slice of that flat table; the shader is handed the flat
/// table and an offset, so the offset is recovered here rather than assumed.
pub fn golay_offsets(golay: &Golay) -> [u32; 25] {
    let flat = golay.codewords();
    let mut off = [0u32; 25];
    for (w, o) in off.iter_mut().enumerate() {
        *o = flat
            .iter()
            .position(|c| c.count_ones() as usize == w)
            .map_or(0, |p| p as u32);
    }
    off
}
/// Mirror of `binomial_block.metal`'s `struct BlockRec`, 108 bytes.
///
/// One per class plus the origin at entry 0, the id convention of every table
/// in this crate. It carries what a **block** decode needs and the walk arm's
/// record does not: the off-support kinds, the Golay bucket, the parity
/// requirement, and the widths of the three fields that are not per-kind.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuBlockRec {
    /// Support values (even) or the class's values (odd), **already divided by**
    /// `sqrt(16·m)`.
    ///
    /// 🚨 On the **odd** coset the float's **sign bit carries bit 1 of the
    /// integer |value|**, which is the whole forced-sign rule: a value ≡ 3
    /// (mod 4) is positive on the support and negative off it, and bit 1 of an
    /// odd |value| is set exactly when it is ≡ 3. The shader recovers the
    /// magnitude with `fabs` and the flag with a shift, so the rule costs one
    /// XOR. Pinned by `the_odd_sign_flag_rides_in_the_sign_bit`.
    pub vals_on: [f32; MAX_KINDS],
    /// Off-support values; the zero run's is `0.0`, and that zero is what marks
    /// it — never its index. Reconstructing the index from counts is what
    /// failed V0 on 883 blocks in P1.
    pub vals_off: [f32; MAX_KINDS],
    pub cnt_on: [u8; MAX_KINDS],
    pub cnt_off: [u8; MAX_KINDS],
    /// `⌈log₂ radix⌉` per kind; 0 for the last, whose radix is 1.
    pub wb_on: [u8; MAX_KINDS],
    pub wb_off: [u8; MAX_KINDS],
    pub k_on: u8,
    pub k_off: u8,
    /// Support weight; 24 on the odd coset, whose single walk spans all slots.
    pub w: u8,
    /// `p_req | odd << 1`.
    pub geom: u8,
    /// Offset of the class's weight bucket in the flat Golay table.
    pub golay_base: u32,
    pub wb_golay: u8,
    pub wb_sw: u8,
    pub wb_sf: u8,
    pub pad: u8,
}

const _: () = assert!(std::mem::size_of::<GpuBlockRec>() == 108);
const _: () = assert!(std::mem::align_of::<GpuBlockRec>() == 4);

/// The 384 block records — entry 0 the origin, entry `1 + ci` class `ci`.
///
/// Everything is read from [`FastDecoder::cascade_class`], [`FastDecoder::levels`]
/// and [`llvq_search::cns::cns_layout`] — the same three sources the CPU
/// reference reads, so there is no fourth derivation to drift from.
pub fn block_records(fd: &FastDecoder, golay: &Golay) -> Vec<GpuBlockRec> {
    let goff = golay_offsets(golay);
    let mut out = Vec::with_capacity(BLOCK_ENTRIES);
    // The origin: no kind, no field, and a dot of zero whatever the gain.
    out.push(GpuBlockRec {
        k_on: 1,
        k_off: 0,
        w: DIM as u8,
        geom: 0b10, // odd, so the sign path never reads `vals_off`
        ..GpuBlockRec::default()
    });

    for ci in 0..fd.n_classes() {
        let c = fd.cascade_class(ci);
        let lay = cns_layout(fd, golay, ci);
        let shell = fd.levels(ci).shell;
        let norm = ((16 * shell) as f64).sqrt();
        let mut r = GpuBlockRec {
            k_on: lay.n_on.max(1) as u8,
            k_off: lay.n_off as u8,
            w: if c.odd { DIM as u8 } else { c.w as u8 },
            geom: (c.p_req as u8) | (u8::from(c.odd) << 1),
            golay_base: if c.odd { 0 } else { goff[c.w as usize] },
            wb_golay: ceil_log2(lay.golay_radix) as u8,
            wb_sw: ceil_log2(lay.s_w) as u8,
            wb_sf: ceil_log2(lay.s_f) as u8,
            ..GpuBlockRec::default()
        };
        for j in 0..lay.n_on {
            r.wb_on[j] = ceil_log2(lay.on_radices[j]) as u8;
        }
        for j in 0..lay.n_off {
            r.wb_off[j] = ceil_log2(lay.off_radices[j]) as u8;
        }
        if c.odd {
            for j in 0..c.n_kinds {
                r.cnt_on[j] = c.counts[j];
                let v = (c.vals[j] as f64 / norm) as f32;
                // The forced-sign flag rides in the sign bit; see the field's
                // doc. `vals[j]` is odd on this coset, so bit 1 is the whole
                // rule.
                let flag = (c.vals[j] as u32 >> 1) & 1;
                r.vals_on[j] = f32::from_bits(v.to_bits() | (flag << 31));
            }
        } else {
            for j in 0..c.n_word {
                r.cnt_on[j] = c.word_counts[j];
                r.vals_on[j] = (c.word_vals[j] as f64 / norm) as f32;
            }
            for j in 0..c.n_off {
                r.cnt_off[j] = c.off_counts[j];
                // `free_vals` holds the nonzero magnitudes; the zero run's
                // entry stays 0.0, which is what marks it.
                r.vals_off[j] = (c.free_vals[j] as f64 / norm) as f32;
            }
        }
        out.push(r);
    }
    out
}
/// Entries of the payload-width table — one per value the 9-bit class field can
/// hold, so the kernel indexes it with the header itself and never with a
/// converted id.
pub const E1V_PAY_ENTRIES: usize = 512;

/// The origin id, as a table index. The format states it once, in [`crate::e1v`].
const ORIGIN_IDX: usize = E1V_ORIGIN_ID as usize;

/// Payload bits of every E1v record, indexed by the **stream's own class
/// field**: `ci` for a class, [`E1V_ORIGIN_ID`] for the origin, zero for the
/// values no class takes.
///
/// Two things make this table load-bearing rather than a convenience, and both
/// are asserted here.
///
/// * It is what the **warp-scan** sums, so it decides where every *later* lane
///   of the group starts reading. A width off by one bit does not corrupt one
///   field, it desynchronises the rest of the group.
/// * It must equal, entry for entry, what the **cursor** consumes —
///   `wb_golay + Σ wb_on + Σ wb_off + wb_sw + wb_sf` of the very record the
///   shader decodes with. The scan and the cursor are two different arithmetics
///   over the same widths, and nothing else ties them together: a record whose
///   fields summed to less than its declared width would decode correctly and
///   still move its neighbours.
///
/// The first quantity comes from [`cns_layout`] — the writer's own source, via
/// `llvq_artifact::e1v::payload_bits` — and the second from [`GpuBlockRec`],
/// which is built from the class table. Neither reads the other.
pub fn e1v_payload_bits(fd: &FastDecoder, golay: &Golay, recs: &[GpuBlockRec]) -> Vec<u32> {
    use llvq_search::cns::HEADER_BITS;

    assert_eq!(
        recs.len(),
        BLOCK_ENTRIES,
        "the block table carries the origin plus one record per class"
    );
    let mut pay = vec![0u32; E1V_PAY_ENTRIES];

    // The origin's record is the header alone (P5 §2.2). Its entry is zero, and
    // that zero has to be the *table's* zero rather than the one every unused
    // slot already carries — so the record it names is checked to spend nothing.
    let o = &recs[0];
    let origin_fields = u32::from(o.wb_golay)
        + o.wb_on.iter().map(|&b| u32::from(b)).sum::<u32>()
        + o.wb_off.iter().map(|&b| u32::from(b)).sum::<u32>()
        + u32::from(o.wb_sw)
        + u32::from(o.wb_sf);
    assert_eq!(
        origin_fields, 0,
        "the origin's record must read no field at all: its whole record is the header"
    );

    for ci in 0..fd.n_classes() {
        let lay = cns_layout(fd, golay, ci);
        let bits = u32::try_from(lay.bits() - HEADER_BITS).expect("a record fits u32 bits");

        // What the shader's cursor will actually consume, field by field.
        let r = &recs[1 + ci];
        let widths = std::iter::once(r.wb_golay)
            .chain(r.wb_on.iter().take(r.k_on as usize).copied())
            .chain(r.wb_off.iter().take(r.k_off as usize).copied())
            .chain([r.wb_sw, r.wb_sf]);
        let mut sum = 0u32;
        for w in widths {
            // `take` returns a `uint`: a field wider than 32 bits would be
            // silently truncated rather than refused.
            assert!(w <= 32, "class {ci}: a field is {w} bits, and `take` returns 32");
            sum += u32::from(w);
        }
        assert_eq!(
            bits, sum,
            "class {ci}: the scan allocates {bits} bits and the cursor consumes {sum} — \
             the group would desynchronise from this lane on"
        );
        // The window the shader loads is 96 bits and its start can be misaligned
        // by up to 31, so 65 bits are guaranteed readable. The widest record of
        // the cap-13 codebook is 56.
        assert!(
            bits <= 65,
            "class {ci}: a record of {bits} bits does not fit the 96-bit window \
             at every offset"
        );
        pay[ci] = bits;
    }

    // Values 383..511 are ids no class takes; they stay zero, and no block ever
    // reads them. The origin's does, and it is zero by the same construction.
    pay[ORIGIN_IDX] = 0;
    pay
}

/// The E1v fixture — the three things a draw from the published 4B cannot
/// reach, and one it can only reach by accident.
///
/// 1. **The origin.** The file carries zero origin block, so no draw ever sets
///    the header to 511. That id is the one whose payload is empty, whose table
///    entry is 0 rather than `1 + ci`, and which contributes nothing to the
///    warp-scan — three special cases in one value, none of them exercised by
///    2^24 real blocks.
/// 2. **Every class, at both ends.** A class boundary off by one is invisible
///    from any interior index.
/// 3. **The classes the file never uses** — every shell-13 class among them,
///    out of reach of any prefix of any matrix, however long.
/// 4. **One whole group of the widest class**: the largest prefix sum the scan
///    can ever be handed. The draw's groups are mixtures, and a mixture never
///    reaches an extreme.
///
/// Padded with the origin, whose dot is zero whatever its gain, so the padding
/// cannot flatter the worst error of a real entry. `group` is the caller's
/// threadgroup size; the result is a whole number of them **and** a whole
/// number of E1v groups.
///
/// It lives here rather than in either bin for the reason the etalons do:
/// `bin/p1v0` proves the arm exact on it and `bin/rankbench` re-proves it
/// before timing, and two copies of a fixture drift into two coverages.
pub fn e1v_fixture(fd: &FastDecoder, pay: &[u32], group: usize) -> (Vec<u64>, Vec<u8>) {
    let mut idx = vec![0u64]; // the origin, first
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        idx.push(first);
        idx.push(last);
        idx.push(first + (last - first) / 2);
    }
    idx.push(0); // and one mid-stream, so it is not forever lane 0

    // Aligned first, so the block below is exactly ONE group rather than a
    // widest tail and a widest head.
    while !idx.len().is_multiple_of(E1V_GROUP) {
        idx.push(0);
    }
    let widest = (0..fd.n_classes())
        .max_by_key(|&ci| pay[ci])
        .expect("le codebook n'est pas vide");
    let (first, last) = fd.class_range(widest);
    for j in 0..E1V_GROUP {
        idx.push(if j % 2 == 0 { first } else { last });
    }

    assert!(
        group.is_multiple_of(E1V_GROUP),
        "un threadgroup de {group} ne contient pas un nombre entier de groupes E1v"
    );
    while !idx.len().is_multiple_of(group) {
        idx.push(0);
    }
    let gains = (0..idx.len()).map(|i| (i % 2) as u8).collect();
    (idx, gains)
}
#[cfg(test)]
mod tests {
    use super::*;
    use llvq_search::rankdec::binom;

    /// The binomial table is the crate's own, transposed — not a second
    /// derivation.
    #[test]
    fn the_binomial_table_is_the_transpose() {
        let t = binom_table();
        for c in 0..BINOM_STRIDE {
            for p in 0..BINOM_STRIDE {
                assert_eq!(t[c * BINOM_STRIDE + p] as u64, binom(p, c), "c={c} p={p}");
            }
        }
    }
}
