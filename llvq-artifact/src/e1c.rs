//! `E1c` — the bit-sliced twin of [`PlanesBlocks`] and [`Planes12xBlocks`],
//! barreau E1c of `docs/archive/pistes-format-vram-2026-08-05.md` and item X1/X2 of
//! `docs/archive/spec-memoire-extreme-2026-08-12.md`.
//!
//! ## What it is, in one sentence
//!
//! Exactly the `Planes` content — same class ids, same gains, same signs,
//! same per-slot levels, same exception table — with the per-slot bits
//! **transposed over a group of 32 blocks** so that the record's padding
//! disappears: `Planes14` spends 6 bits per block closing its 14-byte
//! stride and `Planes12x` spends 14, while a group of 32 here is a whole
//! number of 32-bit words with nothing left over.
//!
//! ```text
//! Planes14   112 bits/block (106 of payload + 6 of padding)   4.6667 b/w
//! E1c14      106 bits/block, zero padding                     4.4167 b/w
//! Planes12x   96 bits/block ( 82 of payload + 14 of padding)  4.0000 b/w
//! E1c12       82 bits/block, zero padding                     3.4167 b/w
//! ```
//!
//! The saving is the padding and nothing else, which is why this module is a
//! **repack** rather than an encoder: [`E1c14Blocks::from_planes14`] takes an
//! already-transcoded stream and moves its bits. No lattice search, no
//! `Indexer`, no `ClassTable` — and therefore none of the 404 s
//! `transcode_planes12x` spends on `nearest_angular` per 5-level block, which
//! `Planes12x` already paid and this layout inherits.
//!
//! ## Why a group of 32
//!
//! Because that is the CUDA warp. The transposition puts bit `l` of a word at
//! lane `l`: every lane of the warp reads the **same word address**, so the
//! load is a broadcast out of L1 rather than 32 scattered strided reads, and
//! a lane's own bit is `(word >> lane) & 1` — one shift, no offset
//! arithmetic. That is the whole argument for the layout, and it is the half
//! that no arithmetic here can prove: whether the extra loads (82 or 106 per
//! group instead of 3 or 4 per lane) cost less than the bytes they save is a
//! question for the bench, not for this file. See the spec's X3 criteria.
//!
//! ## The word map
//!
//! A group is `E1C_HEADER_WORDS` header words followed by the per-slot region,
//! **slot-major**:
//!
//! ```text
//! word  0 ..  9   headers, 10 bits per block, block l at bit 10·l
//! word 10 + j·F + f   slot j, field f, bit l = block l's bit
//!                     F = 1 + planes   (smask, then plane 0 .. plane n-1)
//! ```
//!
//! Slot-major, not field-major, and the reason is the kernel: a lane walking
//! slots `0..24` reads a slot's whole field set from **contiguous** words —
//! four for `E1c14`, i.e. one 128-bit load, and three for `E1c12`. Field-major
//! would scatter those four words 24 apart and turn one load into four.
//!
//! ## Padding of the last group
//!
//! `n_blocks` is not a multiple of 32, so the trailing group is completed with
//! **origin blocks** — class 0, no gain, all planes and signs zero. Every bit
//! of the padding is therefore zero, canonicity comes for free, and the
//! decoder needs no special case: the padded lanes decode to the zero point,
//! exactly as an origin block anywhere else in the stream.
//!
//! ## Endianness
//!
//! Words are stored little-endian in [`E1c14Blocks::data`], so a device
//! reading the bytes as `const u32*` on a little-endian host sees the words
//! this module wrote. That is the one place in a layout port where a mistake
//! passes every CPU test and breaks only on the GPU — the same warning
//! `llvq_slot.cuh` carries, restated because the trap is identical.

use crate::runtime::{
    ClassTable, Planes12xBlocks, PlanesBlocks, PLANES12X_BYTES, PLANES12X_PAD_BIT,
    PLANES12X_PLANE_BIT, PLANES14_BYTES, PLANES14_PAD_BIT, PLANES14_PLANE_BIT,
    PLANES14_SMASK_BIT,
};
use llvq_core::{Point, DIM};

/// Blocks per transposed group — the CUDA warp, and the reason the layout
/// exists. Also exactly the width of the word a bit is transposed into, which
/// is what makes the header region come out whole (see [`header_words`]).
pub const E1C_GROUP: usize = 32;

/// Bits of the header a block carries: class 9 + gain 1.
///
/// Derived from the frozen `Planes` offsets rather than written as `10`: the
/// sign mask starts where the header ends, so if either field ever moves this
/// constant moves with it instead of silently disagreeing.
pub const E1C_HEADER_BITS: u64 = PLANES14_SMASK_BIT;

/// Words the header region of one group occupies.
///
/// `E1C_HEADER_BITS · E1C_GROUP / 32` — a whole number **because** the group
/// is the word width, not by luck of the constants: the headers of 32 blocks
/// at 10 bits each are 320 bits, ten words, with no bit left over. Stated as
/// a division rather than as `10` so that a group of 64 or a wider header
/// would still give the right answer.
pub const fn header_words() -> usize {
    (E1C_HEADER_BITS as usize * E1C_GROUP) / 32
}

/// Words one group of 32 blocks occupies, for a record with `planes` level
/// bit-planes: the header region, then `1 + planes` words per slot (the sign
/// mask and the planes), 24 slots.
///
/// `planes = 3` gives 106 words — one per payload bit of a `Planes14` record;
/// `planes = 2` gives 82, one per payload bit of a `Planes12x` record. That
/// identity is the layout's whole claim and it is pinned by
/// [`group_words_equal_the_planes_payload_bits`].
pub const fn group_words(planes: usize) -> usize {
    header_words() + DIM * (1 + planes)
}

/// Level bit-planes of the two variants.
pub const E1C14_PLANES: usize = PLANES14_PLANE_BIT.len();
pub const E1C12_PLANES: usize = PLANES12X_PLANE_BIT.len();

/// The fields of one block, in either variant's record. `planes[k]` beyond
/// the variant's plane count is zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Fields {
    id: u32,
    gain: u32,
    smask: u32,
    planes: [u32; 3],
}

impl Fields {
    /// The header as the 10-bit word the transposed layout stores.
    fn header(&self) -> u32 {
        self.id | (self.gain << 9)
    }

    /// Field `f` of the slot-major region: 0 = sign mask, `1 + k` = plane k.
    fn field(&self, f: usize) -> u32 {
        if f == 0 {
            self.smask
        } else {
            self.planes[f - 1]
        }
    }

    fn set_field(&mut self, f: usize, v: u32) {
        if f == 0 {
            self.smask = v;
        } else {
            self.planes[f - 1] = v;
        }
    }
}

/// Read one `Planes` record out of a byte stream, without a class table.
///
/// Pure bit extraction against the frozen offsets — the table is what turns a
/// level into a value, and a repack never needs that. Lethal on a nonzero
/// padding field, because canonicity is part of the source format and a
/// stream that violates it would repack into something this module could not
/// invert.
fn read_planes_record(data: &[u8], b: usize, bytes: usize, planes: usize, pad_bit: u64) -> Fields {
    let mut buf = [0u8; 16];
    buf[..bytes].copy_from_slice(&data[b * bytes..(b + 1) * bytes]);
    let r = u128::from_le_bytes(buf);
    let pad = r >> pad_bit;
    assert_eq!(
        pad, 0,
        "block {b}: padding past bit {pad_bit} is {pad:#x}, not zero"
    );
    let mut f = Fields {
        id: (r & 0x1ff) as u32,
        gain: ((r >> 9) & 1) as u32,
        smask: ((r >> PLANES14_SMASK_BIT) & 0xff_ffff) as u32,
        planes: [0; 3],
    };
    for k in 0..planes {
        // Both variants put plane 0 straight after the sign mask and pack the
        // planes contiguously, so the offset is derivable; asserting it
        // against the frozen table keeps that from being an assumption.
        let off = PLANES14_SMASK_BIT + DIM as u64 * (1 + k as u64);
        f.planes[k] = ((r >> off) & 0xff_ffff) as u32;
    }
    f
}

/// Write one `Planes` record into a byte stream — the inverse of
/// [`read_planes_record`], used by the `to_planes*` round trips.
fn write_planes_record(data: &mut [u8], b: usize, bytes: usize, planes: usize, f: &Fields) {
    let mut r = u128::from(f.header());
    r |= u128::from(f.smask) << PLANES14_SMASK_BIT;
    for k in 0..planes {
        let off = PLANES14_SMASK_BIT + DIM as u64 * (1 + k as u64);
        r |= u128::from(f.planes[k]) << off;
    }
    data[b * bytes..(b + 1) * bytes].copy_from_slice(&r.to_le_bytes()[..bytes]);
}

/// Transpose `n_blocks` records into groups of 32 words, and flatten to
/// little-endian bytes.
///
/// The trailing group is completed with origin blocks: `read` is simply not
/// called past `n_blocks`, and the words stay zero.
fn transpose(n_blocks: usize, planes: usize, read: impl Fn(usize) -> Fields) -> Vec<u8> {
    let gw = group_words(planes);
    let groups = n_blocks.div_ceil(E1C_GROUP);
    let mut words = vec![0u32; groups * gw];
    for b in 0..n_blocks {
        let (g, l) = (b / E1C_GROUP, b % E1C_GROUP);
        let base = g * gw;
        let f = read(b);
        // Header: 10 bits at bit 10·l of the header region, which straddles a
        // word boundary for most lanes — hence the 64-bit staging window.
        let bit = E1C_HEADER_BITS * l as u64;
        let (w, sh) = ((bit / 32) as usize, bit % 32);
        let v = u64::from(f.header()) << sh;
        words[base + w] |= v as u32;
        if sh + E1C_HEADER_BITS > 32 {
            words[base + w + 1] |= (v >> 32) as u32;
        }
        // Per-slot region, slot-major: one bit per (slot, field), at lane l.
        for j in 0..DIM {
            for f_i in 0..=planes {
                let bit = (f.field(f_i) >> j) & 1;
                words[base + header_words() + j * (1 + planes) + f_i] |= bit << l;
            }
        }
    }
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Read block `b`'s fields back out of a transposed stream — the inverse of
/// [`transpose`], and the function every decoder here goes through.
fn untranspose(data: &[u8], b: usize, planes: usize) -> Fields {
    let gw = group_words(planes);
    let (g, l) = (b / E1C_GROUP, b % E1C_GROUP);
    let base = g * gw;
    let word = |i: usize| -> u32 {
        let o = (base + i) * 4;
        u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
    };
    let bit = E1C_HEADER_BITS * l as u64;
    let (w, sh) = ((bit / 32) as usize, bit % 32);
    let mut win = u64::from(word(w));
    if sh + E1C_HEADER_BITS > 32 {
        win |= u64::from(word(w + 1)) << 32;
    }
    let hdr = ((win >> sh) & ((1 << E1C_HEADER_BITS) - 1)) as u32;
    let mut f = Fields {
        id: hdr & 0x1ff,
        gain: hdr >> 9,
        ..Default::default()
    };
    for j in 0..DIM {
        for f_i in 0..=planes {
            let bit = (word(header_words() + j * (1 + planes) + f_i) >> l) & 1;
            f.set_field(f_i, f.field(f_i) | (bit << j));
        }
    }
    f
}

/// Turn fields into a lattice point, the way every `Planes` decoder does.
///
/// Lethal on a level at or past the class's `len`: a transposition bug that
/// scrambled plane bits would otherwise produce plausible weights, which is
/// precisely the failure mode this crate refuses everywhere else.
fn point_of(f: &Fields, table: &ClassTable, planes: usize, b: usize) -> (Point, u32) {
    let rec = table.record(f.id as usize);
    if f.id == 0 {
        return ([0; DIM], f.gain);
    }
    let mut p = [0i32; DIM];
    for (i, pi) in p.iter_mut().enumerate() {
        let mut lvl = 0u32;
        for k in 0..planes {
            lvl |= ((f.planes[k] >> i) & 1) << k;
        }
        assert!(
            (lvl as usize) < rec.len as usize,
            "block {b}, slot {i}: level {lvl} outside class {} (len {})",
            f.id,
            rec.len
        );
        let v = rec.values[lvl as usize];
        *pi = if (f.smask >> i) & 1 == 1 { -v } else { v };
    }
    (p, f.gain)
}

// ---------------------------------------------------------------------------
// E1c14 — the no-cap variant: same content as Planes14, zero padding
// ---------------------------------------------------------------------------

/// `Planes14` transposed: 106 words per group of 32 blocks, **4.4167 b/w**
/// against 4.6667, at strictly identical decoded content.
///
/// The variant with no quality question at all — no level cap, no exception
/// table, a pure bijection of the production stream. If the bench likes it,
/// it is a free 0.25 b/w; if it does not, the transposition idea is dead and
/// [`E1c12Blocks`] need not be measured.
pub struct E1c14Blocks {
    pub n_blocks: usize,
    /// Transposed words, little-endian, `group_words(3)·4` bytes per group.
    pub data: Vec<u8>,
}

impl E1c14Blocks {
    /// Repack a [`PlanesBlocks`] stream. Pure bit movement — no table, no
    /// search, no possibility of a different decoded value.
    pub fn from_planes14(p: &PlanesBlocks) -> Self {
        assert_eq!(
            p.data.len(),
            p.n_blocks * PLANES14_BYTES,
            "Planes14 stream is not {PLANES14_BYTES} bytes per block"
        );
        let data = transpose(p.n_blocks, E1C14_PLANES, |b| {
            read_planes_record(&p.data, b, PLANES14_BYTES, E1C14_PLANES, PLANES14_PAD_BIT)
        });
        Self { n_blocks: p.n_blocks, data }
    }

    /// The inverse repack. Exists to make the bijection testable **without a
    /// class table**: a decoded-value comparison needs a table and a real
    /// artifact, while `from ∘ to == identity` on raw bytes is checkable on
    /// any machine, including one that has never seen a model.
    pub fn to_planes14(&self) -> PlanesBlocks {
        let mut data = vec![0u8; self.n_blocks * PLANES14_BYTES];
        for b in 0..self.n_blocks {
            let f = untranspose(&self.data, b, E1C14_PLANES);
            write_planes_record(&mut data, b, PLANES14_BYTES, E1C14_PLANES, &f);
        }
        PlanesBlocks { n_blocks: self.n_blocks, data }
    }

    /// Bits per weight, padding of the trailing group included — so a stream
    /// of one block honestly reports a whole group, not 106/24.
    pub fn bits_per_weight(&self) -> f64 {
        (self.data.len() as u64 * 8) as f64 / (self.n_blocks * DIM) as f64
    }

    /// Decode block `b` — must equal [`PlanesBlocks::decode_block`] on the
    /// same input, bit for bit.
    pub fn decode_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        assert_eq!(table.gain_bits(), 1, "E1c freezes the gain field at one bit");
        point_of(&untranspose(&self.data, b, E1C14_PLANES), table, E1C14_PLANES, b)
    }
}

// ---------------------------------------------------------------------------
// E1c12 — the capped variant: Planes12x transposed, same exception table
// ---------------------------------------------------------------------------

/// `Planes12x` transposed: 82 words per group, **3.4167 b/w** of main stream
/// against 4.0000, plus the *same* exception table, carried verbatim.
///
/// The exception records stay `Planes14`-shaped and un-transposed on purpose:
/// they are read by a correction pass that touches ~3.4 % of blocks in no
/// particular order, where a group transposition buys nothing and costs a
/// gather. Their cost is therefore identical to `Planes12x`'s, which is what
/// makes the two layouts' difference exactly the main-stream padding.
pub struct E1c12Blocks {
    pub n_blocks: usize,
    /// Transposed main stream, little-endian, `group_words(2)·4` bytes per
    /// group.
    pub data: Vec<u8>,
    /// Matrix-wide block index of each exception, strictly ascending —
    /// [`Planes12xBlocks::exc_idx`] unchanged.
    pub exc_idx: Vec<u32>,
    /// One `PLANES14_BYTES` record per exception, the **exact** block —
    /// [`Planes12xBlocks::exc_data`] unchanged.
    pub exc_data: Vec<u8>,
}

impl E1c12Blocks {
    /// Repack a [`Planes12xBlocks`] stream, exceptions carried through.
    pub fn from_planes12x(p: &Planes12xBlocks) -> Self {
        assert_eq!(
            p.data.len(),
            p.n_blocks * PLANES12X_BYTES,
            "Planes12x main stream is not {PLANES12X_BYTES} bytes per block"
        );
        let data = transpose(p.n_blocks, E1C12_PLANES, |b| {
            read_planes_record(&p.data, b, PLANES12X_BYTES, E1C12_PLANES, PLANES12X_PAD_BIT)
        });
        Self {
            n_blocks: p.n_blocks,
            data,
            exc_idx: p.exc_idx.clone(),
            exc_data: p.exc_data.clone(),
        }
    }

    /// The inverse repack — main stream only, exceptions cloned back.
    pub fn to_planes12x(&self) -> Planes12xBlocks {
        let mut data = vec![0u8; self.n_blocks * PLANES12X_BYTES];
        for b in 0..self.n_blocks {
            let f = untranspose(&self.data, b, E1C12_PLANES);
            write_planes_record(&mut data, b, PLANES12X_BYTES, E1C12_PLANES, &f);
        }
        Planes12xBlocks {
            n_blocks: self.n_blocks,
            data,
            exc_idx: self.exc_idx.clone(),
            exc_data: self.exc_data.clone(),
        }
    }

    pub fn n_exceptions(&self) -> usize {
        self.exc_idx.len()
    }

    /// Bits per weight, exception table included — the number the spec's
    /// 3.4167 + 0.20 prediction is to be checked against.
    pub fn bits_per_weight(&self) -> f64 {
        let bits = self.data.len() as u64 * 8
            + self.exc_idx.len() as u64 * 32
            + self.exc_data.len() as u64 * 8;
        bits as f64 / (self.n_blocks * DIM) as f64
    }

    /// What the **main stream** says about block `b` — exact for ≤ 4-level
    /// blocks, the swapped representative for exceptions, exactly as
    /// [`Planes12xBlocks::decode_approx_block`].
    pub fn decode_approx_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        point_of(&untranspose(&self.data, b, E1C12_PLANES), table, E1C12_PLANES, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::PLANES12X_SMASK_BIT;
    use llvq_core::SplitMix64;

    /// `read_planes_record` reads **both** variants against the `Planes14`
    /// offsets, which is only legitimate because the two records share their
    /// header and place plane 0 at the same bit. Left implicit, that is an
    /// assumption; asserted here, it is a contract — and if `Planes12x` ever
    /// moved a field, this fires instead of the repack quietly reading
    /// garbage out of the right-looking place.
    #[test]
    fn both_source_records_share_their_header_geometry() {
        assert_eq!(PLANES12X_SMASK_BIT, PLANES14_SMASK_BIT);
        assert_eq!(PLANES12X_PLANE_BIT[0], PLANES14_PLANE_BIT[0]);
        for (k, (&a, &b)) in PLANES12X_PLANE_BIT
            .iter()
            .zip(PLANES14_PLANE_BIT.iter())
            .enumerate()
        {
            assert_eq!(a, b, "plan {k} n'est pas au même bit dans les deux formats");
        }
    }

    /// Deterministic valid records: every field masked to its width and the
    /// padding left zero, which is what the source formats guarantee.
    fn random_stream(n: usize, bytes: usize, planes: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data = vec![0u8; n * bytes];
        for b in 0..n {
            let mut f = Fields {
                id: (rng.next() % 384) as u32,
                gain: (rng.next() & 1) as u32,
                smask: (rng.next() & 0xff_ffff) as u32,
                planes: [0; 3],
            };
            for k in 0..planes {
                f.planes[k] = (rng.next() & 0xff_ffff) as u32;
            }
            write_planes_record(&mut data, b, bytes, planes, &f);
        }
        data
    }

    /// **The layout's whole claim**: a group of 32 blocks is a whole number of
    /// words, and that number equals the source record's *payload* bits — so
    /// the transposition costs exactly the padding and nothing else.
    ///
    /// Pinned against the frozen `PLANES*_PAD_BIT` rather than against 106 and
    /// 82, so a format change lands here instead of drifting.
    #[test]
    fn group_words_equal_the_planes_payload_bits() {
        assert_eq!(header_words(), 10, "10 bits × 32 blocs = 320 bits = 10 mots");
        assert_eq!(group_words(E1C14_PLANES), PLANES14_PAD_BIT as usize);
        assert_eq!(group_words(E1C12_PLANES), PLANES12X_PAD_BIT as usize);
        assert_eq!(group_words(E1C14_PLANES), 106);
        assert_eq!(group_words(E1C12_PLANES), 82);
        // And the saving is the padding, stated as bits per block.
        assert_eq!(PLANES14_BYTES * 8 - group_words(E1C14_PLANES), 6);
        assert_eq!(PLANES12X_BYTES * 8 - group_words(E1C12_PLANES), 14);
    }

    /// The payload rates the spec predicts, from the geometry alone.
    #[test]
    fn payload_rates_are_the_predicted_ones() {
        let r14 = group_words(E1C14_PLANES) as f64 / DIM as f64;
        let r12 = group_words(E1C12_PLANES) as f64 / DIM as f64;
        assert!((r14 - 4.416_666).abs() < 1e-5, "E1c14 : {r14}");
        assert!((r12 - 3.416_666).abs() < 1e-5, "E1c12 : {r12}");
        // Strictly under the layouts they replace, on both variants.
        assert!(r14 < PLANES14_BYTES as f64 * 8.0 / DIM as f64);
        assert!(r12 < PLANES12X_BYTES as f64 * 8.0 / DIM as f64);
    }

    /// Round trip on both variants, over several groups **and a partial one**
    /// — the trailing group is where an off-by-one in the padding would live.
    #[test]
    fn repack_is_a_bijection_on_both_variants() {
        for n in [1, 31, 32, 33, 100] {
            let src = random_stream(n, PLANES14_BYTES, E1C14_PLANES, 0x1c14 + n as u64);
            let p = PlanesBlocks { n_blocks: n, data: src.clone() };
            let back = E1c14Blocks::from_planes14(&p).to_planes14();
            assert_eq!(back.data, src, "E1c14 round trip, n = {n}");

            let src = random_stream(n, PLANES12X_BYTES, E1C12_PLANES, 0x1c12 + n as u64);
            let p = Planes12xBlocks {
                n_blocks: n,
                data: src.clone(),
                exc_idx: vec![],
                exc_data: vec![],
            };
            let back = E1c12Blocks::from_planes12x(&p).to_planes12x();
            assert_eq!(back.data, src, "E1c12 round trip, n = {n}");
        }
    }

    /// A bijection test alone is **not enough**, and this is the reason: any
    /// permutation of the word map round-trips perfectly through its own
    /// inverse. Swapping slot-major for field-major, or lane for slot, would
    /// leave the test above green and produce a stream no kernel could read.
    ///
    /// So: set exactly one bit in one field of one block, and demand it land
    /// at the one word and the one bit the word map specifies.
    #[test]
    fn the_word_map_is_slot_major_and_lane_indexed() {
        // Block 5 of the first group, slot 7, plane 1 (field 2), nothing else.
        const B: usize = 5;
        const J: usize = 7;
        const F: usize = 2;
        let mut data = vec![0u8; 40 * PLANES14_BYTES];
        let mut f = Fields::default();
        f.planes[F - 1] = 1 << J;
        write_planes_record(&mut data, B, PLANES14_BYTES, E1C14_PLANES, &f);
        let e = E1c14Blocks::from_planes14(&PlanesBlocks { n_blocks: 40, data });

        let expect = header_words() + J * (1 + E1C14_PLANES) + F;
        let words: Vec<u32> = e
            .data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let set: Vec<usize> = words
            .iter()
            .enumerate()
            .filter(|(_, &w)| w != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(set, vec![expect], "un seul mot non nul, et c'est {expect}");
        assert_eq!(words[expect], 1 << B, "le bit du bloc est à la lane {B}");
        // Slot-major is the claim: slot J's fields are contiguous, so the
        // next slot's field 0 is exactly `1 + planes` words further on.
        assert_eq!(
            header_words() + (J + 1) * (1 + E1C14_PLANES),
            expect + (1 + E1C14_PLANES) - F,
        );
    }

    /// Headers straddle word boundaries for most lanes (10 bits into 32), so
    /// the staging window is load-bearing. Walk every lane of a group with a
    /// distinct id and demand each comes back.
    #[test]
    fn every_lane_header_survives_the_straddle() {
        let n = E1C_GROUP;
        let mut data = vec![0u8; n * PLANES14_BYTES];
        for b in 0..n {
            let f = Fields {
                id: (b as u32 * 11 + 1) & 0x1ff,
                gain: (b & 1) as u32,
                ..Default::default()
            };
            write_planes_record(&mut data, b, PLANES14_BYTES, E1C14_PLANES, &f);
        }
        let e = E1c14Blocks::from_planes14(&PlanesBlocks { n_blocks: n, data });
        for b in 0..n {
            let f = untranspose(&e.data, b, E1C14_PLANES);
            assert_eq!(f.id, (b as u32 * 11 + 1) & 0x1ff, "lane {b}");
            assert_eq!(f.gain, (b & 1) as u32, "lane {b}");
        }
    }

    /// The trailing group is padded with **origin** blocks, so every padding
    /// bit is zero — canonicity for free, and a decoder with no special case.
    #[test]
    fn the_trailing_group_is_padded_with_zeros() {
        let n = 33; // one full group plus a single lane
        let data = random_stream(n, PLANES14_BYTES, E1C14_PLANES, 0x9AD_5EED);
        let e = E1c14Blocks::from_planes14(&PlanesBlocks { n_blocks: n, data });
        assert_eq!(e.data.len(), 2 * group_words(E1C14_PLANES) * 4);
        // Lanes 1..32 of the second group were never written.
        for b in 1..E1C_GROUP {
            let f = untranspose(&e.data, n - 1 + b, E1C14_PLANES);
            assert_eq!(f, Fields::default(), "lane de bourrage {b} non nulle");
        }
    }

    /// A source stream whose padding is not zero is not a `Planes` stream, and
    /// repacking it would silently drop bits the inverse could not restore.
    #[test]
    #[should_panic(expected = "padding past bit")]
    fn a_noncanonical_source_record_is_refused() {
        let mut data = vec![0u8; PLANES14_BYTES];
        data[PLANES14_BYTES - 1] = 0x80; // bit 111, inside the 6 pad bits
        let _ = E1c14Blocks::from_planes14(&PlanesBlocks { n_blocks: 1, data });
    }

    /// `bits_per_weight` must charge the trailing group's padding, or a small
    /// matrix would report a rate it does not pay.
    #[test]
    fn the_rate_charges_the_trailing_group() {
        let one = E1c14Blocks {
            n_blocks: 1,
            data: vec![0u8; group_words(E1C14_PLANES) * 4],
        };
        // One block alone pays a whole group: 106 words for 24 weights.
        assert!(one.bits_per_weight() > 100.0);
        // A full group pays the nominal rate exactly.
        let full = E1c14Blocks {
            n_blocks: E1C_GROUP,
            data: vec![0u8; group_words(E1C14_PLANES) * 4],
        };
        assert!((full.bits_per_weight() - 106.0 / 24.0).abs() < 1e-12);
    }
}
