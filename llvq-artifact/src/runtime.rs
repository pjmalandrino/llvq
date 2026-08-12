//! The runtime block layout, and the transcoder that fills it from a `.llvq`.
//!
//! The archive stores a permutation rank: optimal in bits, 8.27 ns/block to
//! open on a GPU — 106× the floor. The kernel needs a layout it can decode in
//! the shadow of its own memory traffic, and `llvq-metal/decode` measured
//! nested masks there at 0.11 ns/block. The two formats meet exactly once,
//! here, at load time: rank in, masks out, paid once per model.
//!
//! ## The payload, frozen by measurement (`rtbits`, 150.7 M real blocks)
//!
//! Per block, bit-packed **LSB-first** (bit `i` of the stream is
//! `data[i/8] >> (i%8) & 1` — the order a shader reading little-endian words
//! gets for free):
//!
//! ```text
//! [class : 9 bits][gain : g bits][signs : nz bits][nested masks]
//! ```
//!
//! * **class** — 0 for the origin block, else `1 +` the class's rank in the
//!   v1 layout (shells ascending, even before odd). 384 values, 9 bits.
//! * **gain** — the block's gain level rank, `g` bits as the matrix declares.
//! * **signs** — one bit per **nonzero** coordinate, slot order, 1 = negative.
//!   Zero slots carry none: a decoder walking slots in order keeps a running
//!   nonzero count anyway, so the sign index is free.
//! * **masks** — levels in the canonical order (descending count, ties by
//!   descending |value|); mask `k` covers the slots levels `< k` left free,
//!   one bit per remaining slot in slot order; the last level is implicit.
//!   Widths and level values come from [`ClassTable`], 384 constant entries.
//!
//! The alternatives died measured: positional +0.8 bits/weight everywhere, a
//! u16 offset per block 3.59 b/w against grouped's 3.35, fixed-128 nibbles
//! 5.33 against fixed-96's 4.00.
//!
//! ## Two addressings, one payload
//!
//! * [`Layout::Fixed96`] — every block spans 96 bits (12 bytes, three aligned
//!   u32). The **exact** worst case over the whole cap-13 class table is 74
//!   bits (class 238, shell 12), so 96 holds every possible block, forever.
//!   4.000 b/w, no indirection.
//! * [`Layout::Grouped32`] — blocks in groups of 32 (one SIMD group); every
//!   lane of a group reads the group's max payload width rounded up to a
//!   byte; one u32 byte base per group, strides implied by consecutive bases.
//!   3.355 b/w measured, at the price of unaligned loads.
//!
//! Which one the kernel wants is a GPU question; the transcoder produces
//! both, and the `llvq-metal` bench on real blocks settles it.
//!
//! ## What this module does *not* do
//!
//! Scales, centroids, rotation seeds and tails stay beside the blocks
//! ([`crate::RawMatrix`] carries them); they are per-row or per-matrix and
//! read once per row, not per block. This module owns the per-block stream
//! only — the part whose format the kernel is married to.

use crate::{Error, Result};
use llvq_core::{Golay, Point, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};
// Planes12x only: the L = 5 swap re-encodes a block's direction, which needs
// the exact nearest-neighbour machinery `lswap` uses — nothing external.
use llvq_search::generic::BallSearcher;
use llvq_search::index::Indexer;
use llvq_search::Searcher;

/// Bits of the class field: 383 classes plus the origin fit in 9.
pub const CLASS_BITS: u32 = 9;
/// Bytes per block of the fixed layout.
pub const FIXED96_BYTES: usize = 12;
/// Lanes per group of the grouped layout.
pub const GROUP: usize = 32;
/// Bits of the in-group position field of [`Layout::Sorted32`].
pub const POS_BITS: u32 = 5;

/// What the decoder knows about a class before reading any block bits.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassRecord {
    /// Level |values|, canonical order; slots ≥ `len` unused.
    pub values: [i32; MAX_LEVELS],
    /// Coordinates per level; sums to 24 over `len` slots.
    pub counts: [u8; MAX_LEVELS],
    pub len: u8,
    /// Nonzero coordinates — the sign field's width.
    pub nonzero: u8,
    /// Total payload width in bits, class and gain fields included —
    /// nested-mask arrangement ([`Layout::Fixed96`] / [`Layout::Grouped32`]).
    pub width: u16,
    /// Same, for the slot-space arrangement of [`Layout::Flat32`].
    pub width_flat: u16,
    /// Same, for [`Layout::Slot32`] — the nz-bit sign field traded for the
    /// fixed 24-bit mask. One source for both the stride computation and
    /// the encoder's width assert, so they cannot drift apart silently.
    pub width_slot: u16,
}

/// The 384-entry constant table both sides of the format share.
///
/// Entry 0 is the origin (one level, value 0, no signs, no masks); entry
/// `1 + ci` is class `ci` of the v1 layout.
pub struct ClassTable {
    recs: Vec<ClassRecord>,
    gain_bits: u32,
}

impl ClassTable {
    pub fn new(fd: &FastDecoder, gain_bits: u32) -> Self {
        let mut recs = Vec::with_capacity(fd.n_classes() + 1);
        recs.push(ClassRecord {
            values: [0; MAX_LEVELS],
            counts: {
                let mut c = [0u8; MAX_LEVELS];
                c[0] = DIM as u8;
                c
            },
            len: 1,
            nonzero: 0,
            width: (CLASS_BITS + gain_bits) as u16,
            width_flat: (CLASS_BITS + gain_bits) as u16,
            width_slot: (CLASS_BITS + gain_bits) as u16,
        });
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let mut mask_bits = 0u32;
            let mut left = DIM as u32;
            for i in 0..lv.len.saturating_sub(1) {
                mask_bits += left;
                left -= lv.counts[i] as u32;
            }
            let common = CLASS_BITS + gain_bits + lv.nonzero as u32;
            recs.push(ClassRecord {
                values: lv.values,
                counts: lv.counts,
                len: lv.len as u8,
                nonzero: lv.nonzero,
                width: (common + mask_bits) as u16,
                width_flat: (common + DIM as u32 * (lv.len as u32 - 1)) as u16,
                width_slot: (common - lv.nonzero as u32
                    + DIM as u32 * lv.len as u32) as u16,
            });
        }
        Self { recs, gain_bits }
    }

    pub fn gain_bits(&self) -> u32 {
        self.gain_bits
    }

    pub fn record(&self, id: usize) -> &ClassRecord {
        &self.recs[id]
    }

    pub fn n_entries(&self) -> usize {
        self.recs.len()
    }

    /// Widest payload any entry can produce — must fit the fixed layout.
    pub fn worst_width(&self) -> u32 {
        self.recs.iter().map(|r| r.width as u32).max().unwrap_or(0)
    }

    /// Widest [`Layout::Flat32`] payload — 130 bits on the cap-13 table.
    pub fn worst_width_flat(&self) -> u32 {
        self.recs.iter().map(|r| r.width_flat as u32).max().unwrap_or(0)
    }

    /// Widest [`Layout::Slot32`] payload — 130 bits on the cap-13 table with
    /// a 1-bit gain, since `width_slot = 9 + gain_bits + 24·L`.
    ///
    /// This is the bound the fused kernel's read window rests on. `slot_dot`
    /// loads five consecutive `u32` from `byte >> 2`, so every record must
    /// satisfy `8·(byte & 3) + worst_width_slot() <= 160`; with a byte shift
    /// of at most 24 that is `24 + 130 = 154`, six bits of margin.
    ///
    /// [`Self::worst_width_flat`] does **not** imply it. `width_slot −
    /// width_flat = 24 − nonzero >= 0`, and the two coincide here only
    /// because the widest flat record happens to be an odd 5-level class,
    /// where `nonzero == 24`. Cap the shell differently and the coincidence
    /// breaks without breaking a single existing assertion — which is why
    /// this accessor exists rather than reusing the flat one.
    pub fn worst_width_slot(&self) -> u32 {
        self.recs.iter().map(|r| r.width_slot as u32).max().unwrap_or(0)
    }
}

/// How blocks are addressed in the stream — and, for [`Layout::Flat32`],
/// how the payload itself is arranged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 96 bits per block, constant stride, aligned. Nested-mask payload.
    Fixed96,
    /// Byte-rounded max-width stride per group of [`GROUP`] blocks, one u32
    /// byte base per group. Nested-mask payload.
    Grouped32,
    /// Grouped addressing with a **kernel-shaped** payload, bought with
    /// bits: the fused matvec showed the nested masks' cost is fundamental —
    /// resolving one slot takes prefix popcounts in rank space, ~156 µs of
    /// ALU on a layer whose structural floor is 49 µs. This layout removes
    /// that wall:
    ///
    /// ```text
    /// [class : 9][gain : g][signs : nz, level-major][slot masks : 24 × (L−1)]
    /// ```
    ///
    /// * masks are in **slot space**, one 24-bit word per level `1..L`;
    ///   level 0 — the most populated — is the complement of their union.
    ///   No popcounts, no rank walk: a level's slots come straight off its
    ///   word with `ctz`, and zero slots are never touched at all.
    /// * signs are reordered **level-major** (levels in canonical order,
    ///   slots ascending within a level, the zero level skipped), so a
    ///   per-level walk consumes them sequentially — the running counter
    ///   the slot-order signs forced is gone.
    ///
    /// Worst case 130 bits (odd 5-level class, 24 signs), so this payload
    /// does not fit `Fixed96` and is grouped-only.
    Flat32,
    /// [`Layout::Flat32`] plus divergence control: within each group of 32,
    /// blocks are **sorted by class id**, and each payload carries its
    /// original in-group position so the consumer can put the dot product
    /// back on the right output row:
    ///
    /// ```text
    /// [class : 9][gain : g][pos : 5][signs][slot masks]
    /// ```
    ///
    /// The 32 lanes of a SIMD group then decode runs of equal classes:
    /// aligned level loops, uniform table loads. Costs 5 bits per block and
    /// a per-group permutation; the kernel pays one atomic add per row per
    /// group instead of a per-row `simd_sum`. Worst case 135 bits.
    ///
    /// ⚠️ The adversarial audit showed the sort itself is a null lever: the
    /// 32 lanes of a SIMD group execute the same block set whatever their
    /// order inside it. Kept for the measurement record.
    Sorted32,
    /// The audit's structural breach: signs as a 24-bit **slot mask**, so
    /// the whole payload sits at fixed offsets:
    ///
    /// ```text
    /// [class : 9][gain : g][sign mask : 24][slot masks : 24 × (L−1)]
    /// ```
    ///
    /// Width is a function of the level count alone — 34 + 24(L−1), worst
    /// case 130 bits, unchanged. The decoder needs no `nz`, no zero-level
    /// index, no sign bases and no per-level walks: a fixed 24-slot loop,
    /// level by four slot-mask tests (absent masks are zero, level 0 is the
    /// default), sign by one bit of the mask — zero divergence, no serial
    /// state, fully unrollable. Sign bits of zero slots are written 0, so
    /// transcodes stay byte-deterministic. Costs 24−nz bits over `Flat32`
    /// (nothing on odd-coset blocks); grouped-only, like the others.
    Slot32,
}

/// A transcoded block stream, plus what it takes to address it.
pub struct RuntimeBlocks {
    pub layout: Layout,
    pub n_blocks: usize,
    pub gain_bits: u32,
    /// The bit-packed blocks, LSB-first within each byte.
    pub data: Vec<u8>,
    /// `Grouped32` only: byte offset of each group's first block, plus one
    /// final entry at the end of the stream. The stride of group `g` is
    /// `(bases[g+1] - bases[g]) / 32`, exact because a trailing partial
    /// group is padded to all 32 lanes.
    pub bases: Vec<u32>,
}

impl RuntimeBlocks {
    /// Bits the stream spends per weight, addressing included.
    pub fn bits_per_weight(&self) -> f64 {
        let bits = self.data.len() as u64 * 8 + self.bases.len() as u64 * 32;
        bits as f64 / (self.n_blocks * DIM) as f64
    }

    /// Decode block `b` back to its lattice point and gain rank.
    ///
    /// This is the reference the GPU kernel is checked against, and the other
    /// half of the transcoder's round-trip proof: for every index,
    /// `decode_block(transcode(idx)) == Indexer::decode(idx)`, bit for bit.
    pub fn decode_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        let bit0 = match self.layout {
            Layout::Fixed96 => b as u64 * (FIXED96_BYTES as u64 * 8),
            Layout::Grouped32 | Layout::Flat32 | Layout::Sorted32 | Layout::Slot32 => {
                let g = b / GROUP;
                let stride = (self.bases[g + 1] - self.bases[g]) as u64 / GROUP as u64;
                (self.bases[g] as u64 + (b % GROUP) as u64 * stride) * 8
            }
        };
        // Sorted32 first: the slot at position `b` holds a *different*,
        // sorted block — reading its header as if it were block `b` (and
        // especially taking its `id == 0` early return) would be wrong.
        if self.layout == Layout::Sorted32 {
            // The block for position `b` sits in whichever slot of its group
            // carries that position. Scan the real slots — O(32), and this
            // is the reference decoder, not the kernel.
            let g = b / GROUP;
            let want = (b % GROUP) as u64;
            let stride = (self.bases[g + 1] - self.bases[g]) as u64 / GROUP as u64;
            let real = GROUP.min(self.n_blocks - g * GROUP);
            for slot in 0..real {
                let at = (self.bases[g] as u64 + slot as u64 * stride) * 8;
                let mut c = BitCursor::new(&self.data, at);
                let sid = c.read(CLASS_BITS) as usize;
                let sgain = c.read(self.gain_bits) as u32;
                let spos = c.read(POS_BITS);
                if spos == want {
                    let srec = table.record(sid);
                    if sid == 0 {
                        return ([0; DIM], sgain);
                    }
                    return (decode_flat(&mut c, srec), sgain);
                }
            }
            unreachable!("position {want} missing from group {g}");
        }

        let mut cur = BitCursor::new(&self.data, bit0);
        let id = cur.read(CLASS_BITS) as usize;
        let gain = cur.read(self.gain_bits) as u32;
        let rec = table.record(id);
        if id == 0 {
            return ([0; DIM], gain);
        }
        if self.layout == Layout::Flat32 {
            return (decode_flat(&mut cur, rec), gain);
        }
        if self.layout == Layout::Slot32 {
            let smask = cur.read(DIM as u32) as u32;
            let nlev = rec.len as usize;
            let mut masks = [0u32; MAX_LEVELS];
            for m in masks.iter_mut().take(nlev).skip(1) {
                *m = cur.read(DIM as u32) as u32;
            }
            let mut p = [0i32; DIM];
            for (i, pi) in p.iter_mut().enumerate() {
                let b = 1u32 << i;
                // Absent masks are zero; level 0 is the default.
                let mut v = rec.values[0];
                for (mk, &vk) in masks.iter().zip(&rec.values).take(nlev).skip(1) {
                    if mk & b != 0 {
                        v = vk;
                    }
                }
                *pi = if smask & b != 0 { -v } else { v };
            }
            return (p, gain);
        }

        let signs = cur.read(rec.nonzero as u32);
        let nlev = rec.len as usize;
        let mut masks = [0u32; MAX_LEVELS - 1];
        let mut left = DIM as u32;
        for (k, m) in masks.iter_mut().enumerate().take(nlev - 1) {
            *m = cur.read(left) as u32;
            left -= rec.counts[k] as u32;
        }

        // Walk the slots once; `rank[k]` is the running index among the slots
        // levels < k left free, `nz` the running sign index.
        let mut p = [0i32; DIM];
        let mut rank = [0u32; MAX_LEVELS];
        let mut nz = 0u32;
        for pi in p.iter_mut() {
            let mut level = nlev - 1;
            for k in 0..nlev - 1 {
                let hit = masks[k] >> rank[k] & 1 == 1;
                rank[k] += 1;
                if hit {
                    level = k;
                    break;
                }
            }
            let v = rec.values[level];
            if v != 0 {
                *pi = if signs >> nz & 1 == 1 { -v } else { v };
                nz += 1;
            }
        }
        (p, gain)
    }
}

// ---------------------------------------------------------------------------
// Planes14 — E1a of docs/archive/pistes-format-vram-2026-08-05.md
// ---------------------------------------------------------------------------

/// Bytes per block of the [`PlanesBlocks`] layout — a frozen constant of the
/// format, not a computed stride.
pub const PLANES14_BYTES: usize = 14;
/// First bit of the sign mask in a Planes14 record.
pub const PLANES14_SMASK_BIT: u64 = 10;
/// First bit of each of the three level bit-planes.
pub const PLANES14_PLANE_BIT: [u64; 3] = [34, 58, 82];
/// First padding bit; bits `106..112` must be written zero.
pub const PLANES14_PAD_BIT: u64 = 106;

/// The `Planes14` block stream: one record per block, **uniform 14-byte
/// stride**, no base table. Bit offsets are fixed for every class:
///
/// ```text
/// [class : 9][gain : 1][sign mask : 24][plane0 : 24][plane1 : 24][plane2 : 24][0 : 6]
/// ```
///
/// The level of slot `j` is `p0[j] | p1[j]<<1 | p2[j]<<2` — an explicit
/// 3-bit index into the *same* [`ClassRecord`] values [`Layout::Slot32`]
/// uses (`len <= 5`, so 3 bits always suffice). Level 0 is no longer
/// implicit: a slot at level 0 simply has its three plane bits zero, so an
/// `L = 1` block (the origin included) carries three all-zero planes. The
/// sign mask keeps Slot32's exact semantics: bit `j` is the sign of slot
/// `j`, and zero slots have their bit **written 0** so equal inputs
/// transcode to equal bytes.
///
/// The payload is a bijection of the Slot32 content — same class id, same
/// gain, same signs, same per-slot levels — re-arranged so a decoder needs
/// neither a base table nor per-class widths: `offset = 14·b`, always.
/// 112 bits over 24 weights is 4.6667 b/w, between `Fixed96`'s 4.000 and
/// `Slot32`'s byte-rounded group strides.
///
/// The gain field is **frozen at bit 9, one bit wide** — the offsets above
/// are the format, so a table with `gain_bits != 1` is refused outright
/// rather than silently shifting every field.
pub struct PlanesBlocks {
    pub n_blocks: usize,
    /// The bit-packed records, LSB-first within each byte, `14·n_blocks`
    /// bytes exactly.
    pub data: Vec<u8>,
}

impl PlanesBlocks {
    /// Bits the stream spends per weight — addressing included, which for
    /// this layout is nothing: `112 / 24` exactly.
    pub fn bits_per_weight(&self) -> f64 {
        (self.data.len() as u64 * 8) as f64 / (self.n_blocks * DIM) as f64
    }

    /// Decode block `b` back to its lattice point and gain rank — the CPU
    /// reference every GPU reading of these bytes is checked against, same
    /// signature as [`RuntimeBlocks::decode_block`].
    pub fn decode_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        assert_eq!(
            table.gain_bits(),
            1,
            "Planes14 freezes the gain field at bit 9, one bit wide"
        );
        let bit0 = b as u64 * (PLANES14_BYTES as u64 * 8);
        let mut cur = BitCursor::new(&self.data, bit0);
        let id = cur.read(CLASS_BITS) as usize;
        let gain = cur.read(1) as u32;
        let rec = table.record(id);
        if id == 0 {
            return ([0; DIM], gain);
        }
        let smask = cur.read(DIM as u32) as u32;
        let p0 = cur.read(DIM as u32) as u32;
        let p1 = cur.read(DIM as u32) as u32;
        let p2 = cur.read(DIM as u32) as u32;
        let mut p = [0i32; DIM];
        for (i, pi) in p.iter_mut().enumerate() {
            let lvl = (p0 >> i & 1) | (p1 >> i & 1) << 1 | (p2 >> i & 1) << 2;
            assert!(
                (lvl as usize) < rec.len as usize,
                "block {b}, slot {i}: level {lvl} outside class {id} (len {})",
                rec.len
            );
            let v = rec.values[lvl as usize];
            *pi = if smask >> i & 1 == 1 { -v } else { v };
        }
        (p, gain)
    }
}

/// Transcode raw `(index, gain)` codes into a [`PlanesBlocks`] stream.
///
/// Same contract as [`transcode`]: one fast decode per block, an
/// out-of-range index is an error, and the result decodes bit-for-bit to
/// what `Indexer::decode` gives. Padding bits `106..112` of every record
/// are written zero — canonicity is part of the format, as it is for
/// [`Layout::Slot32`]'s zero-slot sign bits.
pub fn transcode_planes14(
    fd: &FastDecoder,
    table: &ClassTable,
    indices: &[u64],
    gains: &[u32],
) -> Result<PlanesBlocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert_eq!(
        table.gain_bits(),
        1,
        "Planes14 freezes the gain field at bit 9, one bit wide"
    );
    let n = indices.len();
    let mut data = vec![0u8; n * PLANES14_BYTES];
    for (b, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
        assert!(gain < 2, "block {b}: gain {gain} overflows the 1-bit field");
        let id = class_id(fd, idx)?;
        let rec = *table.record(id);
        let bit0 = b as u64 * (PLANES14_BYTES as u64 * 8);
        let mut w = BitSink::new(&mut data, bit0);
        w.push(id as u64, CLASS_BITS);
        w.push(gain as u64, 1);
        if id == 0 {
            // Origin: every slot is at level 0 = value 0, so the sign mask
            // and all three planes are zero — the pre-zeroed record is
            // already exactly right.
            continue;
        }
        let p = fd.decode(idx).expect("class_id validated the index");
        // Level of each slot: |value| matched against the class's levels —
        // distinct by construction, so the match is unambiguous. Same
        // level indices Slot32 encodes through its masks.
        let mut level = [0u8; DIM];
        for (i, &v) in p.iter().enumerate() {
            let a = v.abs();
            level[i] = rec.values[..rec.len as usize]
                .iter()
                .position(|&u| u == a)
                .expect("decoded value belongs to its class") as u8;
        }
        // Sign mask in slot space — zero slots contribute 0 bits, written 0.
        let mut smask = 0u64;
        for (i, &v) in p.iter().enumerate() {
            if v < 0 {
                smask |= 1 << i;
            }
        }
        w.push(smask, DIM as u32);
        // The three bit-planes of the per-slot 3-bit level.
        for plane in 0..3u8 {
            let mut m = 0u64;
            for (i, &l) in level.iter().enumerate() {
                if l >> plane & 1 == 1 {
                    m |= 1 << i;
                }
            }
            w.push(m, DIM as u32);
        }
        // Nothing writes past bit 105 and the buffer was pre-zeroed, so
        // bits 106..112 are zero by construction; the assert makes the
        // frozen geometry a checked fact rather than a comment.
        assert_eq!(
            w.pos - bit0,
            PLANES14_PAD_BIT,
            "class {id}: Planes14 record is not 106 payload bits"
        );
    }
    Ok(PlanesBlocks { n_blocks: n, data })
}

/// Transcode one matrix's raw `(index, gain)` codes into a runtime stream.
///
/// Cost is one fast decode per block (243 ns): ~37 s single-core for a 4B,
/// ~3 s across cores when the caller splits by matrix.
pub fn transcode(
    fd: &FastDecoder,
    table: &ClassTable,
    indices: &[u64],
    gains: &[u32],
    layout: Layout,
) -> Result<RuntimeBlocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert!(
        table.worst_width() <= FIXED96_BYTES as u32 * 8,
        "class table exceeds the fixed layout"
    );
    let n = indices.len();
    let mut out = RuntimeBlocks {
        layout,
        n_blocks: n,
        gain_bits: table.gain_bits(),
        data: Vec::new(),
        bases: Vec::new(),
    };

    let sorted = layout == Layout::Sorted32;
    let width_of = |rec: &ClassRecord| match layout {
        Layout::Fixed96 | Layout::Grouped32 => rec.width as u32,
        Layout::Flat32 => rec.width_flat as u32,
        Layout::Sorted32 => rec.width_flat as u32 + POS_BITS,
        Layout::Slot32 => rec.width_slot as u32,
    };
    match layout {
        Layout::Fixed96 => {
            out.data = vec![0u8; n * FIXED96_BYTES];
            for (b, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
                let bit0 = b as u64 * (FIXED96_BYTES as u64 * 8);
                encode_block(fd, table, idx, gain, &mut out.data, bit0, layout, None)?;
            }
        }
        Layout::Grouped32 | Layout::Flat32 | Layout::Sorted32 | Layout::Slot32 => {
            let ngroups = n.div_ceil(GROUP);
            out.bases = Vec::with_capacity(ngroups + 1);
            let mut base = 0u32;
            for g in 0..ngroups {
                out.bases.push(base);
                let blocks = &indices[g * GROUP..(g * GROUP + GROUP).min(n)];
                // Stride from the class table alone — no decode needed.
                let mut width = 0u32;
                for &idx in blocks {
                    let id = class_id(fd, idx)?;
                    width = width.max(width_of(table.record(id)));
                }
                let stride = width.div_ceil(8);
                // A partial trailing group still pays all 32 lanes, so the
                // base difference always divides by 32.
                out.data.resize(base as usize + GROUP * stride as usize, 0);
                // Sorted32: slot l holds the block whose class id ranks
                // l-th in the group (stable), and says where it came from.
                let mut order: Vec<usize> = (0..blocks.len()).collect();
                if sorted {
                    let ids: Vec<usize> = blocks
                        .iter()
                        .map(|&idx| class_id(fd, idx))
                        .collect::<Result<_>>()?;
                    order.sort_by_key(|&l| ids[l]);
                }
                for (slot, &l) in order.iter().enumerate() {
                    let bit0 = (base as u64 + slot as u64 * stride as u64) * 8;
                    let pos = if sorted { Some(l as u32) } else { None };
                    encode_block(
                        fd, table, blocks[l], gains[g * GROUP + l],
                        &mut out.data, bit0, layout, pos,
                    )?;
                }
                base += GROUP as u32 * stride;
            }
            out.bases.push(base);
        }
    }
    Ok(out)
}

/// Decode a [`Layout::Flat32`] payload after its class and gain fields.
fn decode_flat(cur: &mut BitCursor, rec: &ClassRecord) -> Point {
    let nlev = rec.len as usize;
    let signs = cur.read(rec.nonzero as u32);
    let mut masks = [0u32; MAX_LEVELS];
    let mut union = 0u32;
    for m in masks.iter_mut().take(nlev).skip(1) {
        *m = cur.read(DIM as u32) as u32;
        union |= *m;
    }
    masks[0] = !union & 0xff_ffff;

    let mut p = [0i32; DIM];
    let mut sbase = 0u32;
    for (k, &mk) in masks.iter().enumerate().take(nlev) {
        let v = rec.values[k];
        if v == 0 {
            continue; // zero slots stay zero and carry no signs
        }
        let mut lm = mk;
        let mut t = 0u32;
        while lm != 0 {
            let i = lm.trailing_zeros() as usize;
            lm &= lm - 1;
            p[i] = if signs >> (sbase + t) & 1 == 1 { -v } else { v };
            t += 1;
        }
        sbase += rec.counts[k] as u32;
    }
    p
}

/// Class field value for an index: 0 for the origin, `1 + ci` otherwise.
fn class_id(fd: &FastDecoder, idx: u64) -> Result<usize> {
    match fd.class_of(idx) {
        Some(ci) => Ok(ci + 1),
        None if idx == 0 => Ok(0),
        None => Err(Error::IndexOutOfRange {
            name: String::new(),
            index: idx,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_block(
    fd: &FastDecoder,
    table: &ClassTable,
    idx: u64,
    gain: u32,
    data: &mut [u8],
    bit0: u64,
    layout: Layout,
    pos: Option<u32>,
) -> Result<()> {
    let flat = layout == Layout::Flat32 || layout == Layout::Sorted32;
    let id = class_id(fd, idx)?;
    let rec = *table.record(id);
    let mut w = BitSink::new(data, bit0);
    w.push(id as u64, CLASS_BITS);
    w.push(gain as u64, table.gain_bits);
    let extra = if let Some(p) = pos {
        w.push(p as u64, POS_BITS);
        POS_BITS as u64
    } else {
        0
    };
    if id == 0 {
        return Ok(());
    }

    let p = fd.decode(idx).expect("class_id validated the index");
    // Level of each slot: |value| matched against the class's levels —
    // distinct by construction, so the match is unambiguous.
    let mut level = [0u8; DIM];
    for (i, &v) in p.iter().enumerate() {
        let a = v.abs();
        level[i] = rec.values[..rec.len as usize]
            .iter()
            .position(|&u| u == a)
            .expect("decoded value belongs to its class") as u8;
    }
    let nlev = rec.len as usize;

    if layout == Layout::Slot32 {
        // Sign mask in slot space — zero slots contribute 0 bits, written 0.
        let mut smask = 0u64;
        for (i, &v) in p.iter().enumerate() {
            if v < 0 {
                smask |= 1 << i;
            }
        }
        w.push(smask, DIM as u32);
        // Slot-space masks for levels 1.., level 0 implicit — as Flat32.
        for k in 1..nlev {
            let mut mask = 0u64;
            for (i, &l) in level.iter().enumerate() {
                if l as usize == k {
                    mask |= 1 << i;
                }
            }
            w.push(mask, DIM as u32);
        }
        assert_eq!(
            w.pos - bit0,
            rec.width_slot as u64 + extra,
            "class {id}: encoded slot width disagrees with the table"
        );
        return Ok(());
    }

    if flat {
        // Signs level-major: levels in canonical order, slots ascending
        // within a level, the zero level skipped.
        let mut signs = 0u64;
        let mut nz = 0u32;
        for k in 0..nlev {
            if rec.values[k] == 0 {
                continue;
            }
            for (i, &l) in level.iter().enumerate() {
                if l as usize == k {
                    if p[i] < 0 {
                        signs |= 1 << nz;
                    }
                    nz += 1;
                }
            }
        }
        debug_assert_eq!(nz, rec.nonzero as u32);
        w.push(signs, nz);

        // Slot-space masks for levels 1.., level 0 implicit.
        for k in 1..nlev {
            let mut mask = 0u64;
            for (i, &l) in level.iter().enumerate() {
                if l as usize == k {
                    mask |= 1 << i;
                }
            }
            w.push(mask, DIM as u32);
        }
        assert_eq!(
            w.pos - bit0,
            rec.width_flat as u64 + extra,
            "class {id}: encoded flat width disagrees with the table"
        );
        return Ok(());
    }

    // Signs of the nonzero slots, slot order.
    let mut signs = 0u64;
    let mut nz = 0u32;
    for &v in &p {
        if v != 0 {
            if v < 0 {
                signs |= 1 << nz;
            }
            nz += 1;
        }
    }
    debug_assert_eq!(nz, rec.nonzero as u32);
    w.push(signs, nz);

    // Nested masks, mirroring the decoder's running-rank walk.
    let mut left = DIM as u32;
    for k in 0..nlev - 1 {
        let mut mask = 0u64;
        let mut r = 0u32;
        for &l in &level {
            if l as usize >= k {
                if l as usize == k {
                    mask |= 1 << r;
                }
                r += 1;
            }
        }
        debug_assert_eq!(r, left);
        w.push(mask, left);
        left -= rec.counts[k] as u32;
    }
    // Hard assert, not debug: the release-mode sweep is the only place every
    // class passes through, and a width table that disagrees with what was
    // actually written corrupts grouped strides silently.
    assert_eq!(
        w.pos - bit0,
        rec.width as u64 + extra,
        "class {id}: encoded width disagrees with the table"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Planes12x — M2: the sparse-overlay layout (main stream + exact exceptions)
// ---------------------------------------------------------------------------

/// Bytes per block of the [`Planes12xBlocks`] **main stream** — a frozen
/// constant of the format. `12·b` is always a multiple of 4, so every record
/// is an aligned three-word window with a zero bit shift.
pub const PLANES12X_BYTES: usize = 12;
/// First bit of the sign mask in a Planes12x main-stream record.
pub const PLANES12X_SMASK_BIT: u64 = 10;
/// First bit of each of the two level bit-planes.
pub const PLANES12X_PLANE_BIT: [u64; 2] = [34, 58];
/// First padding bit; bits `82..96` must be written zero.
pub const PLANES12X_PAD_BIT: u64 = 82;
/// Bytes one exception costs: a `u32` block index plus a full
/// [`PLANES14_BYTES`] record of the exact block — 144 bits.
pub const PLANES12X_EXC_BYTES: usize = 4 + PLANES14_BYTES;
/// The magnitude-level cap of the main stream: two bit-planes address levels
/// `0..4`, so a block whose class holds 5 distinct magnitudes cannot be
/// stored exactly and becomes an exception.
pub const PLANES12X_LEVEL_CAP: usize = 4;
/// The ball the L = 5 swap searches — the published file is `leech1c12`, so
/// every representative's index stays inside the stream's 47-bit width.
pub const PLANES12X_SHELL_CAP: u32 = 12;

/// The `Planes12x` block stream: a **12-byte main stream** holding every
/// block at ≤ 4 magnitude levels, plus a sorted **exception table** carrying
/// the exact [`PLANES14_BYTES`] record of every 5-level block.
///
/// Main-stream record, bit offsets fixed for every class:
///
/// ```text
/// [class : 9][gain : 1][sign mask : 24][plane0 : 24][plane1 : 24][0 : 14]
/// ```
///
/// The level of slot `j` is `p0[j] | p1[j]<<1` — two planes, four leaves.
/// Blocks whose class has ≤ 4 levels are stored **exactly** (same semantics
/// as [`PlanesBlocks`], the third plane dropped because it is identically
/// zero). A 5-level block stores its **best L ≤ 4 direction** instead — the
/// `lswap` machinery: `nearest_angular` over the m ≤ [`PLANES12X_SHELL_CAP`]
/// ball with a level cap of [`PLANES12X_LEVEL_CAP`] — with the gain bit
/// **copied**, and contributes one entry to the exception table:
///
/// * `exc_idx[e]` — the block's index in the matrix, strictly ascending;
/// * `exc_data[e·14 .. e·14+14]` — the exact block's Planes14 record.
///
/// [`Self::decode_block`] resolves exceptions first, so the reconstruction
/// is **identical** to [`PlanesBlocks`] / [`Layout::Slot32`], bit for bit —
/// the approximation exists only in the main stream, where a consumer that
/// applies the exceptions (the fused kernel's correction pass) cancels it
/// exactly. Payload: `96/24 = 4` b/w plus `144/24 = 6` b/w × the L = 5
/// fraction — ~4.20 b/w on the published 4B (3.3824 % of blocks).
pub struct Planes12xBlocks {
    pub n_blocks: usize,
    /// The bit-packed main stream, LSB-first within each byte,
    /// `12·n_blocks` bytes exactly.
    pub data: Vec<u8>,
    /// Matrix-wide block index of each exception, strictly ascending.
    pub exc_idx: Vec<u32>,
    /// One [`PLANES14_BYTES`] record per exception — the **exact** block.
    pub exc_data: Vec<u8>,
}

impl Planes12xBlocks {
    /// Exceptions carried — the number of 5-level blocks of the input.
    pub fn n_exceptions(&self) -> usize {
        self.exc_idx.len()
    }

    /// Bits the layout spends per weight, exception table included:
    /// `(96·n + 144·n_exc) / (24·n)` exactly.
    pub fn bits_per_weight(&self) -> f64 {
        let bits = self.data.len() as u64 * 8
            + self.exc_idx.len() as u64 * 32
            + self.exc_data.len() as u64 * 8;
        bits as f64 / (self.n_blocks * DIM) as f64
    }

    /// Decode block `b` to its **exact** lattice point and gain rank —
    /// exceptions resolved from their Planes14 record, so the result equals
    /// [`PlanesBlocks::decode_block`] on the same input, bit for bit. This
    /// is the CPU reference the GPU overlay (main kernel + correction pass)
    /// is checked against.
    pub fn decode_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        assert_eq!(
            table.gain_bits(),
            1,
            "Planes12x freezes the gain field at bit 9, one bit wide"
        );
        if let Ok(e) = self.exc_idx.binary_search(&(b as u32)) {
            let rec = &self.exc_data[e * PLANES14_BYTES..(e + 1) * PLANES14_BYTES];
            let (p, gain, id) = decode_planes14_record(rec, table, b);
            // An exception exists *because* its class has 5 levels; anything
            // else in the table is a transcoder bug, not a valid stream.
            assert_eq!(
                table.record(id).len,
                5,
                "block {b}: exception record is not a 5-level class"
            );
            return (p, gain);
        }
        self.decode_approx_block(table, b)
    }

    /// Decode what the **main stream** says about block `b` — exact for
    /// ≤ 4-level blocks, the swapped L ≤ 4 representative for exceptions.
    /// This is what the GPU's row pass computes before the correction pass
    /// subtracts it back out on exception blocks.
    pub fn decode_approx_block(&self, table: &ClassTable, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        let bit0 = b as u64 * (PLANES12X_BYTES as u64 * 8);
        let mut cur = BitCursor::new(&self.data, bit0);
        let id = cur.read(CLASS_BITS) as usize;
        let gain = cur.read(1) as u32;
        let rec = table.record(id);
        if id == 0 {
            assert!(
                cur.is_zero(PLANES12X_BYTES as u32 * 8 - CLASS_BITS - 1),
                "block {b}: origin record carries bits past the header"
            );
            return ([0; DIM], gain);
        }
        let smask = cur.read(DIM as u32) as u32;
        let p0 = cur.read(DIM as u32) as u32;
        let p1 = cur.read(DIM as u32) as u32;
        assert_eq!(cur.read(14), 0, "block {b}: padding bits 82..96 must be zero");
        let mut p = [0i32; DIM];
        for (i, pi) in p.iter_mut().enumerate() {
            let lvl = (p0 >> i & 1) | (p1 >> i & 1) << 1;
            assert!(
                (lvl as usize) < rec.len as usize,
                "block {b}, slot {i}: level {lvl} outside class {id} (len {})",
                rec.len
            );
            let v = rec.values[lvl as usize];
            *pi = if smask >> i & 1 == 1 { -v } else { v };
        }
        (p, gain)
    }
}

/// Decode one [`PLANES14_BYTES`] record from a byte slice — the exception
/// table's entry format, identical to a [`PlanesBlocks`] record. Returns the
/// class-field value too, so the caller can assert exception invariants.
///
/// Lethal on purpose, like every decoder here: a level at or past the
/// class's `len`, or a bit among the six trailing zeros, panics rather than
/// producing plausible weights.
fn decode_planes14_record(rec_bytes: &[u8], table: &ClassTable, b: usize) -> (Point, u32, usize) {
    let mut buf = [0u8; 16];
    buf[..PLANES14_BYTES].copy_from_slice(rec_bytes);
    let r = u128::from_le_bytes(buf);
    assert_eq!(
        r >> PLANES14_PAD_BIT,
        0,
        "block {b}: exception padding bits 106..112 must be zero"
    );
    let id = (r & 0x1ff) as usize;
    let gain = ((r >> 9) & 1) as u32;
    let rec = table.record(id);
    if id == 0 {
        assert_eq!(r >> 10, 0, "block {b}: origin record carries payload bits");
        return ([0; DIM], gain, id);
    }
    let smask = ((r >> PLANES14_SMASK_BIT) & 0xff_ffff) as u32;
    let p0 = ((r >> PLANES14_PLANE_BIT[0]) & 0xff_ffff) as u32;
    let p1 = ((r >> PLANES14_PLANE_BIT[1]) & 0xff_ffff) as u32;
    let p2 = ((r >> PLANES14_PLANE_BIT[2]) & 0xff_ffff) as u32;
    let mut p = [0i32; DIM];
    for (j, pj) in p.iter_mut().enumerate() {
        let lvl = (p0 >> j & 1) | (p1 >> j & 1) << 1 | (p2 >> j & 1) << 2;
        assert!(
            (lvl as usize) < rec.len as usize,
            "block {b}, slot {j}: level {lvl} outside class {id} (len {})",
            rec.len
        );
        let v = rec.values[lvl as usize];
        *pj = if smask >> j & 1 == 1 { -v } else { v };
    }
    (p, gain, id)
}

/// Encode one bit-plane record — class, 1-bit gain, slot-space sign mask,
/// `nplanes` level bit-planes — the shared shape of a Planes14 record
/// (3 planes) and a Planes12x main-stream record (2 planes). The buffer must
/// be pre-zeroed; padding canonicity follows, as in [`transcode_planes14`].
fn encode_plane_record(
    fd: &FastDecoder,
    table: &ClassTable,
    idx: u64,
    gain: u32,
    nplanes: u32,
    data: &mut [u8],
    bit0: u64,
) -> Result<()> {
    assert!(gain < 2, "gain {gain} overflows the 1-bit field");
    let id = class_id(fd, idx)?;
    let rec = *table.record(id);
    assert!(
        u32::from(rec.len) <= 1 << nplanes,
        "class {id} has {} levels: {nplanes} bit-planes cannot hold it",
        rec.len
    );
    let mut w = BitSink::new(data, bit0);
    w.push(id as u64, CLASS_BITS);
    w.push(u64::from(gain), 1);
    if id == 0 {
        // Origin: every slot at level 0 = value 0; the pre-zeroed record is
        // already exactly right past the header.
        return Ok(());
    }
    let p = fd.decode(idx).expect("class_id validated the index");
    // Level of each slot: |value| matched against the class's levels —
    // distinct by construction, so the match is unambiguous.
    let mut level = [0u8; DIM];
    for (i, &v) in p.iter().enumerate() {
        let a = v.abs();
        level[i] = rec.values[..rec.len as usize]
            .iter()
            .position(|&u| u == a)
            .expect("decoded value belongs to its class") as u8;
    }
    // Sign mask in slot space — zero slots contribute 0 bits, written 0.
    let mut smask = 0u64;
    for (i, &v) in p.iter().enumerate() {
        if v < 0 {
            smask |= 1 << i;
        }
    }
    w.push(smask, DIM as u32);
    // The bit-planes of the per-slot level index.
    for plane in 0..nplanes as u8 {
        let mut m = 0u64;
        for (i, &l) in level.iter().enumerate() {
            if l >> plane & 1 == 1 {
                m |= 1 << i;
            }
        }
        w.push(m, DIM as u32);
    }
    assert_eq!(
        w.pos - bit0,
        10 + u64::from(DIM as u32) * (1 + u64::from(nplanes)),
        "class {id}: record is not header + {} 24-bit fields",
        1 + nplanes
    );
    Ok(())
}

/// One transcoding chunk's output: main-stream bytes, exception indices
/// (matrix-wide, ascending) and exception records.
type Planes12xChunk = (Vec<u8>, Vec<u32>, Vec<u8>);

/// Transcode one chunk of blocks; `b0` is the matrix-wide index of the first.
fn planes12x_chunk(
    fd: &FastDecoder,
    table: &ClassTable,
    s: &Searcher,
    ix: &Indexer,
    indices: &[u64],
    gains: &[u32],
    b0: usize,
) -> Result<Planes12xChunk> {
    let mut ball = BallSearcher::with_level_cap(PLANES12X_LEVEL_CAP);
    ball.set_shell_cap(PLANES12X_SHELL_CAP);
    let mut data = vec![0u8; indices.len() * PLANES12X_BYTES];
    let mut exc_idx: Vec<u32> = Vec::new();
    let mut exc_data: Vec<u8> = Vec::new();
    for (l, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
        let id = class_id(fd, idx)?;
        let bit0 = l as u64 * (PLANES12X_BYTES as u64 * 8);
        if usize::from(table.record(id).len) <= PLANES12X_LEVEL_CAP {
            // Exact: two planes hold levels 0..4.
            encode_plane_record(fd, table, idx, gain, 2, &mut data, bit0)?;
            continue;
        }
        // L = 5: the exact block goes to the exception table (3 planes),
        // the main stream takes its best L ≤ 4 direction — the `lswap`
        // machinery, block for block — with the gain bit copied.
        exc_idx.push((b0 + l) as u32);
        let e0 = exc_data.len();
        exc_data.resize(e0 + PLANES14_BYTES, 0);
        encode_plane_record(fd, table, idx, gain, 3, &mut exc_data, e0 as u64 * 8)?;
        let p = fd.decode(idx).expect("class_id validated the index");
        let x: [f64; DIM] = core::array::from_fn(|i| f64::from(p[i]));
        let f = ball.nearest_angular(s, &x);
        assert!(
            (2..=PLANES12X_SHELL_CAP).contains(&f.shell),
            "block {}: representative left the m ≤ {PLANES12X_SHELL_CAP} ball",
            b0 + l
        );
        let q = ix.encode(&f.point).expect("searched point is a codebook member");
        assert_eq!(
            fd.decode(q).as_ref(),
            Some(&f.point),
            "block {}: index round-trip diverged",
            b0 + l
        );
        // `encode_plane_record` at 2 planes re-asserts the level cap: a
        // 5-level representative would die here, not decode approximately.
        encode_plane_record(fd, table, q, gain, 2, &mut data, bit0)?;
    }
    Ok((data, exc_idx, exc_data))
}

/// Transcode raw `(index, gain)` codes into a [`Planes12xBlocks`] stream.
///
/// Same contract as [`transcode_planes14`] — one fast decode per block, an
/// out-of-range index is an error, canonical zero padding — plus one
/// `nearest_angular` search per 5-level block (the expensive part: ~0.7 ms
/// per exception per core, threaded across all cores here). The result's
/// [`Planes12xBlocks::decode_block`] equals the archive decoder bit for bit
/// on **every** block; only [`Planes12xBlocks::decode_approx_block`] sees
/// the swapped directions.
pub fn transcode_planes12x(
    fd: &FastDecoder,
    table: &ClassTable,
    s: &Searcher,
    indices: &[u64],
    gains: &[u32],
) -> Result<Planes12xBlocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert_eq!(
        table.gain_bits(),
        1,
        "Planes12x freezes the gain field at bit 9, one bit wide"
    );
    let n = indices.len();
    assert!(
        u32::try_from(n).is_ok(),
        "{n} blocks: exception indices must fit u32"
    );
    let ix = Indexer::new();
    let nthreads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let chunk = n.div_ceil(nthreads).max(1);
    let outs: Vec<Result<Planes12xChunk>> = std::thread::scope(|sc| {
        let handles: Vec<_> = indices
            .chunks(chunk)
            .zip(gains.chunks(chunk))
            .enumerate()
            .map(|(t, (ic, gc))| {
                let ix = &ix;
                sc.spawn(move || planes12x_chunk(fd, table, s, ix, ic, gc, t * chunk))
            })
            .collect();
        handles
            .into_iter()
            // Re-raise a chunk's panic with its original message: the
            // lethal asserts above are part of the API's contract, and a
            // wrapped `Any` would hide which one fired.
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let mut out = Planes12xBlocks {
        n_blocks: n,
        data: Vec::with_capacity(n * PLANES12X_BYTES),
        exc_idx: Vec::new(),
        exc_data: Vec::new(),
    };
    for r in outs {
        let (d, ei, ed) = r?;
        out.data.extend_from_slice(&d);
        out.exc_idx.extend_from_slice(&ei);
        out.exc_data.extend_from_slice(&ed);
    }
    debug_assert!(out.exc_idx.windows(2).all(|w| w[0] < w[1]));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Golay70 — E2 of docs/archive/pistes-format-vram-2026-08-05.md
// ---------------------------------------------------------------------------

/// Bytes per block of the [`Golay70Blocks`] **main stream** — a frozen
/// constant of the format. 72 bits per record, 70 of payload.
pub const GOLAY70_BYTES: usize = 9;
/// Bits of the Golay field: the rank of a codeword among all 4096.
pub const GOLAY70_RANK_BITS: u32 = 12;
/// First bit of the Golay rank in a Golay70 main-stream record.
pub const GOLAY70_RANK_BIT: u64 = 10;
/// First bit of the A plane (refinement bit / level low bit).
pub const GOLAY70_A_BIT: u64 = 22;
/// First bit of the B plane (sign mask / level high bit).
pub const GOLAY70_B_BIT: u64 = 46;
/// First padding bit; bits `70..72` must be written zero.
pub const GOLAY70_PAD_BIT: u64 = 70;
/// Bytes one exception costs: a `u32` block index plus a full
/// [`PLANES14_BYTES`] record of the exact block — 144 bits.
pub const GOLAY70_EXC_BYTES: usize = 4 + PLANES14_BYTES;

/// What the Golay70 decoder knows about a class before reading any block
/// bits — the arm's own table, deliberately separate from [`ClassRecord`]:
/// this layout resolves magnitudes through the mod-4 structure the Golay
/// stage exposes, not through level masks.
///
/// * **Odd class** (`odd`): `values[0..len]` are the class's distinct
///   |values| in the canonical level order; bit `k` of `flags` is set when
///   `values[k] ≡ 3 (mod 4)` — the flag the computed-sign rule compares the
///   codeword bit against.
/// * **Even class**: `pairs[r]` holds the ≤ 2 distinct |values| of mod-4
///   residue `2r` (index 0 = residue 0 with zero included, 1 = residue 2),
///   in canonical level order, single-value residues **duplicated** so the
///   kernel's `pairs[r][a]` load is branchless; `distinct[r]` keeps the true
///   count, which is the transcoder's canonical-A rule and the decoder's
///   lethality.
/// * `exception` — the class cannot ride the main stream: an even class
///   with more than 2 distinct |values| in some residue (one refinement bit
///   cannot pick among 3), or any 5-level class of either coset (the level
///   no longer fits 2 direct bits). Odd classes at L ≤ 4 are **never**
///   exceptions — their level is 2 direct bits, residues play no role.
#[derive(Clone, Copy, Debug, Default)]
pub struct Golay70Class {
    pub odd: bool,
    pub exception: bool,
    pub len: u8,
    /// Odd side: distinct |values|, canonical order; slots ≥ `len` unused.
    pub values: [i8; 4],
    /// Odd side: bit `k` set iff `values[k] ≡ 3 (mod 4)`.
    pub flags: u8,
    /// Even side: `pairs[r][a]`, the |values| of residue `2r`.
    pub pairs: [[i8; 2]; 2],
    /// Even side: distinct |values| per residue, before duplication.
    pub distinct: [u8; 2],
}

// The kernel keeps this table resident; 384 entries must stay small. The
// assert freezes the budget the spec allots (≤ 32 bytes per class).
const _: () = assert!(core::mem::size_of::<Golay70Class>() <= 32);

/// The Golay70 arm's constant tables: one [`Golay70Class`] per
/// [`ClassTable`] entry (entry 0 is the origin), plus the canonical 4096-word
/// Golay codeword table the 12-bit rank field indexes.
pub struct Golay70Table {
    classes: Vec<Golay70Class>,
    golay: Golay,
}

impl Golay70Table {
    pub fn new(fd: &FastDecoder) -> Self {
        let mut classes = Vec::with_capacity(fd.n_classes() + 1);
        // The origin: even coset, one level, value 0 in residue 0. Its main
        // record is all-zero past the header, so none of this is ever read,
        // but the entry stays consistent with the class it describes.
        classes.push(Golay70Class {
            len: 1,
            pairs: [[0, 0], [0, 0]],
            distinct: [1, 0],
            ..Default::default()
        });
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let mut cls = Golay70Class {
                odd: lv.odd,
                len: lv.len as u8,
                ..Default::default()
            };
            // Distinct |values| per mod-4 residue — class values are
            // distinct by construction, so counting slots counts values.
            let mut residues = [0u8; 4];
            for &v in &lv.values[..lv.len] {
                assert_eq!(
                    v.rem_euclid(2),
                    i32::from(lv.odd),
                    "class {ci}: |value| {v} contradicts the coset"
                );
                residues[(v % 4) as usize] += 1;
            }
            cls.exception =
                lv.len == 5 || (!lv.odd && residues.iter().any(|&n| n > 2));
            if cls.exception {
                // Exception classes never decode through this table: their
                // blocks live in the exception stream as Planes14 records.
                classes.push(cls);
                continue;
            }
            if lv.odd {
                for (k, &v) in lv.values[..lv.len].iter().enumerate() {
                    cls.values[k] = i8::try_from(v).expect("|value| fits i8");
                    if v % 4 == 3 {
                        cls.flags |= 1 << k;
                    }
                }
            } else {
                // Residue pairs in canonical level order; duplicate a
                // single-value residue so `pairs[r][a]` is total.
                for &v in &lv.values[..lv.len] {
                    let r = usize::from(v % 4 == 2);
                    let d = usize::from(cls.distinct[r]);
                    cls.pairs[r][d] = i8::try_from(v).expect("|value| fits i8");
                    cls.distinct[r] += 1;
                }
                for r in 0..2 {
                    if cls.distinct[r] == 1 {
                        cls.pairs[r][1] = cls.pairs[r][0];
                    }
                }
            }
            classes.push(cls);
        }
        Self {
            classes,
            golay: Golay::new(),
        }
    }

    /// Class entry `id` — [`ClassTable`]'s numbering (0 = origin).
    pub fn class(&self, id: usize) -> &Golay70Class {
        &self.classes[id]
    }

    pub fn n_entries(&self) -> usize {
        self.classes.len()
    }

    /// The canonical codeword table the 12-bit rank field indexes.
    pub fn golay(&self) -> &Golay {
        &self.golay
    }
}

/// The `Golay70` block stream: a **9-byte main stream** resolving magnitudes
/// through the Golay codeword every v1 index already carries, plus a sorted
/// **exception table** with the exact [`PLANES14_BYTES`] record of every
/// block the 70-bit record cannot hold.
///
/// Main-stream record, bit offsets fixed for every class, LSB-first:
///
/// ```text
/// [class : 9][gain : 1][golay : 12][A : 24][B : 24][0 : 2]
/// ```
///
/// * **golay** — the codeword's rank in the canonical table of all 4096
///   ([`Golay::codewords`], weight-major ascending — the v1 order). For an
///   **even** class the codeword is `{i : |x_i| ≡ 2 (mod 4)}`; for an
///   **odd** class it is `{i : x_i ≡ 3 (mod 4)}` — both are the membership
///   planes of §4 of the project notes, codewords by construction.
/// * **Even class**: the codeword bit fixes the slot's mod-4 residue; bit
///   `i` of **A** picks which of the residue's ≤ 2 |values| the slot holds
///   ([`Golay70Class::pairs`]; written 0 on single-value residues); bit `i`
///   of **B** is the sign, [`Layout::Slot32`]'s exact semantics — zero slots
///   written 0.
/// * **Odd class**: **A** and **B** are two level bit-planes — the slot's
///   level is `A_i | B_i << 1`, a direct index into
///   [`Golay70Class::values`]. Signs are **computed, not stored**:
///   `x_i` is positive iff the codeword bit equals the level's mod-4 flag —
///   the "signs carry no information" rule the v1 index rests on.
///
/// Blocks of exception classes ([`Golay70Class::exception`]) put the
/// **origin record** in the main stream — class 0, gain copied, zero
/// contribution — and their exact block in the exception table, so a GPU
/// correction pass adds the exact contribution with no approximation to
/// cancel. [`Self::decode_block`] resolves exceptions first; the
/// reconstruction is **identical** to [`PlanesBlocks`] / [`Layout::Slot32`],
/// bit for bit.
///
/// The main stream is padded to a word boundary plus 4 bytes, all zero: the
/// kernel reads a record as three u32 words from `(9b) >> 2` (last bit read
/// `≤ 93 < 96`), so the final block's window must exist.
///
/// Payload: `72/24 = 3` b/w plus `144/24 = 6` b/w × the exception fraction —
/// 3.4461 b/w on the sealed 4B (7.4357 % of blocks, the classhist census).
pub struct Golay70Blocks {
    pub n_blocks: usize,
    /// The bit-packed main stream, LSB-first within each byte:
    /// `9·n_blocks` bytes rounded up to a word, plus one zero word.
    pub data: Vec<u8>,
    /// Matrix-wide block index of each exception, strictly ascending.
    pub exc_idx: Vec<u32>,
    /// One [`PLANES14_BYTES`] record per exception — the **exact** block.
    pub exc_data: Vec<u8>,
}

impl Golay70Blocks {
    /// Exceptions carried — blocks whose class the 70-bit record cannot hold.
    pub fn n_exceptions(&self) -> usize {
        self.exc_idx.len()
    }

    /// Bits the layout spends per weight, exception table and tail padding
    /// included: `(72·n + pad) + 144·n_exc` over `24·n`.
    pub fn bits_per_weight(&self) -> f64 {
        let bits = self.data.len() as u64 * 8
            + self.exc_idx.len() as u64 * 32
            + self.exc_data.len() as u64 * 8;
        bits as f64 / (self.n_blocks * DIM) as f64
    }

    /// Decode block `b` to its **exact** lattice point and gain rank —
    /// exceptions resolved from their Planes14 record, so the result equals
    /// [`PlanesBlocks::decode_block`] on the same input, bit for bit. This
    /// is the CPU reference the GPU (main kernel + correction pass) is
    /// checked against.
    pub fn decode_block(&self, table: &ClassTable, g70: &Golay70Table, b: usize) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        assert_eq!(
            table.gain_bits(),
            1,
            "Golay70 freezes the gain field at bit 9, one bit wide"
        );
        if let Ok(e) = self.exc_idx.binary_search(&(b as u32)) {
            let rec = &self.exc_data[e * PLANES14_BYTES..(e + 1) * PLANES14_BYTES];
            let (p, gain, id) = decode_planes14_record(rec, table, b);
            // An exception exists *because* its class cannot ride the main
            // stream; anything else in the table is a transcoder bug.
            assert!(
                g70.class(id).exception,
                "block {b}: exception record's class {id} is not an exception class"
            );
            return (p, gain);
        }
        self.decode_main_block(table, g70, b)
    }

    /// Decode what the **main stream** says about block `b` — exact for
    /// blocks of non-exception classes, the zero-contribution origin record
    /// for exceptions. This is what the GPU's row pass computes before the
    /// correction pass adds the exact exception contributions.
    pub fn decode_main_block(
        &self,
        table: &ClassTable,
        g70: &Golay70Table,
        b: usize,
    ) -> (Point, u32) {
        assert!(b < self.n_blocks, "block {b} of {}", self.n_blocks);
        assert_eq!(
            table.gain_bits(),
            1,
            "Golay70 freezes the gain field at bit 9, one bit wide"
        );
        let bit0 = b as u64 * (GOLAY70_BYTES as u64 * 8);
        let mut cur = BitCursor::new(&self.data, bit0);
        let id = cur.read(CLASS_BITS) as usize;
        let gain = cur.read(1) as u32;
        if id == 0 {
            assert_eq!(
                cur.read(62),
                0,
                "block {b}: origin record carries bits past the header"
            );
            return ([0; DIM], gain);
        }
        let cls = g70.class(id);
        // A populated main record of an exception class is a transcoder bug:
        // exceptions must leave the origin record behind, nothing else.
        assert!(
            !cls.exception,
            "block {b}: main stream carries exception class {id}"
        );
        let grank = cur.read(GOLAY70_RANK_BITS) as usize;
        let c = g70.golay().codewords()[grank];
        let a = cur.read(DIM as u32) as u32;
        let bm = cur.read(DIM as u32) as u32;
        assert_eq!(cur.read(2), 0, "block {b}: padding bits 70..72 must be zero");
        let mut p = [0i32; DIM];
        if cls.odd {
            for (i, pi) in p.iter_mut().enumerate() {
                let lvl = ((a >> i & 1) | (bm >> i & 1) << 1) as usize;
                assert!(
                    lvl < cls.len as usize,
                    "block {b}, slot {i}: level {lvl} outside class {id} (len {})",
                    cls.len
                );
                let v = i32::from(cls.values[lvl]);
                let flag = u32::from(cls.flags >> lvl & 1);
                // Signs carry no information: positive iff the codeword bit
                // equals the value's mod-4 flag (1 when value ≡ 3 mod 4).
                *pi = if c >> i & 1 == flag { v } else { -v };
            }
        } else {
            for (i, pi) in p.iter_mut().enumerate() {
                let r = (c >> i & 1) as usize;
                assert!(
                    cls.distinct[r] > 0,
                    "block {b}, slot {i}: codeword puts the slot in an empty residue of class {id}"
                );
                let ai = (a >> i & 1) as usize;
                assert!(
                    cls.distinct[r] == 2 || ai == 0,
                    "block {b}, slot {i}: non-canonical A bit on a single-value residue"
                );
                let v = i32::from(cls.pairs[r][ai]);
                let neg = bm >> i & 1 == 1;
                assert!(
                    v != 0 || !neg,
                    "block {b}, slot {i}: sign bit set on a zero slot"
                );
                *pi = if neg { -v } else { v };
            }
        }
        (p, gain)
    }
}

/// Transcode raw `(index, gain)` codes into a [`Golay70Blocks`] stream.
///
/// Same contract as [`transcode_planes14`] — one fast decode per block, an
/// out-of-range index is an error, canonical zero padding — plus one Golay
/// rank lookup per block. No re-encoding search: unlike
/// [`transcode_planes12x`], the main stream never approximates — exception
/// blocks contribute **zero** there and exactly through the table, so the
/// result's [`Golay70Blocks::decode_block`] equals the archive decoder bit
/// for bit on every block by construction.
pub fn transcode_golay70(
    fd: &FastDecoder,
    table: &ClassTable,
    g70: &Golay70Table,
    indices: &[u64],
    gains: &[u32],
) -> Result<Golay70Blocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert_eq!(
        table.gain_bits(),
        1,
        "Golay70 freezes the gain field at bit 9, one bit wide"
    );
    let n = indices.len();
    assert!(
        u32::try_from(n).is_ok(),
        "{n} blocks: exception indices must fit u32"
    );
    // Main stream plus the frozen tail: word-aligned, one extra zero word,
    // so the kernel's three-word window exists for the last block.
    let mut data = vec![0u8; (n * GOLAY70_BYTES).div_ceil(4) * 4 + 4];
    let mut exc_idx: Vec<u32> = Vec::new();
    let mut exc_data: Vec<u8> = Vec::new();
    for (b, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
        assert!(gain < 2, "block {b}: gain {gain} overflows the 1-bit field");
        let id = class_id(fd, idx)?;
        let bit0 = b as u64 * (GOLAY70_BYTES as u64 * 8);
        let mut w = BitSink::new(&mut data, bit0);
        let cls = *g70.class(id);
        if id == 0 || cls.exception {
            // Origin, or an exception leaving the zero-contribution origin
            // record behind: class 0, gain copied, all other bits zero.
            w.push(0, CLASS_BITS);
            w.push(u64::from(gain), 1);
            if id == 0 {
                continue;
            }
            exc_idx.push(b as u32);
            let e0 = exc_data.len();
            exc_data.resize(e0 + PLANES14_BYTES, 0);
            encode_plane_record(fd, table, idx, gain, 3, &mut exc_data, e0 as u64 * 8)?;
            continue;
        }
        w.push(id as u64, CLASS_BITS);
        w.push(u64::from(gain), 1);
        let p = fd.decode(idx).expect("class_id validated the index");
        // The membership plane — a Golay codeword by construction. Even
        // coset: |x_i| ≡ 2 (mod 4) (−2 ≡ 2, so signed and absolute agree);
        // odd coset: x_i ≡ 3 (mod 4), the signed rule of the v1 index.
        let residue = if cls.odd { 3 } else { 2 };
        let mut c = 0u32;
        for (i, &v) in p.iter().enumerate() {
            if v.rem_euclid(4) == residue {
                c |= 1 << i;
            }
        }
        let grank = g70
            .golay()
            .rank(c)
            .expect("membership plane of a member is a codeword");
        w.push(grank as u64, GOLAY70_RANK_BITS);
        let (mut a, mut bm) = (0u64, 0u64);
        if cls.odd {
            for (i, &v) in p.iter().enumerate() {
                let av = i8::try_from(v.abs()).expect("|value| fits i8");
                let lvl = cls.values[..cls.len as usize]
                    .iter()
                    .position(|&u| u == av)
                    .expect("decoded value belongs to its class") as u64;
                a |= (lvl & 1) << i;
                bm |= (lvl >> 1 & 1) << i;
            }
        } else {
            for (i, &v) in p.iter().enumerate() {
                let av = i8::try_from(v.abs()).expect("|value| fits i8");
                let r = usize::from(v.rem_euclid(4) == 2);
                // Canonical A: the value's position among the residue's
                // *distinct* values — 0 on single-value residues, and a
                // value outside its residue pair is lethal, not canonical.
                let ai = cls.pairs[r][..usize::from(cls.distinct[r])]
                    .iter()
                    .position(|&u| u == av)
                    .expect("decoded value belongs to its residue pair")
                    as u64;
                a |= ai << i;
                if v < 0 {
                    bm |= 1 << i;
                }
            }
        }
        w.push(a, DIM as u32);
        w.push(bm, DIM as u32);
        assert_eq!(
            w.pos - bit0,
            GOLAY70_PAD_BIT,
            "class {id}: Golay70 record is not 70 payload bits"
        );
    }
    Ok(Golay70Blocks {
        n_blocks: n,
        data,
        exc_idx,
        exc_data,
    })
}

/// LSB-first bit writer over a pre-zeroed slice.
struct BitSink<'a> {
    data: &'a mut [u8],
    pos: u64,
}

impl<'a> BitSink<'a> {
    fn new(data: &'a mut [u8], pos: u64) -> Self {
        Self { data, pos }
    }

    fn push(&mut self, value: u64, width: u32) {
        debug_assert!(width == 64 || value < 1u64 << width, "value overflows width");
        for i in 0..width as u64 {
            if value >> i & 1 == 1 {
                let bit = self.pos + i;
                self.data[(bit / 8) as usize] |= 1 << (bit % 8);
            }
        }
        self.pos += width as u64;
    }
}

/// LSB-first bit reader.
struct BitCursor<'a> {
    data: &'a [u8],
    pos: u64,
}

impl<'a> BitCursor<'a> {
    fn new(data: &'a [u8], pos: u64) -> Self {
        Self { data, pos }
    }

    fn read(&mut self, width: u32) -> u64 {
        assert!(width <= 64, "BitCursor::read({width}): a u64 holds 64 bits");
        let mut out = 0u64;
        for i in 0..width as u64 {
            let bit = self.pos + i;
            out |= ((self.data[(bit / 8) as usize] >> (bit % 8)) as u64 & 1) << i;
        }
        self.pos += width as u64;
        out
    }

    /// Are the next `width` bits all zero? Advances the cursor either way.
    ///
    /// Exists because a canonicity check is the one place a field wider than
    /// a `u64` is natural — `Planes12x`'s origin record has 86 bits that must
    /// all be zero — and [`Self::read`] cannot express it. It used to be
    /// written `read(86)`, which shifts a `u64` by up to 85: a panic under
    /// `debug_assertions` and a masked shift in release.
    ///
    /// ⚠️ The release behaviour happened to be *harmless* — the accumulator
    /// is OR-ed, so aliased bits could not cancel and the `== 0` test still
    /// answered correctly — which is exactly why it survived: the assertion
    /// gave the right verdict for the wrong reason, and only in the profile
    /// the sealed sweeps run in. Found by an `e1c_format` fixture that puts
    /// the origin mid-stream and runs in debug.
    fn is_zero(&mut self, width: u32) -> bool {
        let mut all = true;
        let mut left = width;
        while left > 0 {
            let take = left.min(64);
            all &= self.read(take) == 0;
            left -= take;
        }
        all
    }
}

#[cfg(test)]
mod bitcursor_tests {
    use super::*;

    /// Regression: `Planes12x`'s origin canonicity check reads 86 bits, which
    /// no `u64` holds. It was written `read(86)` — a panic in debug, a masked
    /// shift in release — and survived because the sweeps that exercise the
    /// origin run in release, where OR-ing aliased bits still answered `!= 0`
    /// correctly. This pins both halves of the fix.
    #[test]
    fn is_zero_sees_bits_past_the_first_word() {
        // 96 bits, all zero but one — swept across every position, including
        // the ones a 64-bit read cannot reach.
        for bit in 0..96u64 {
            let mut data = [0u8; 12];
            data[(bit / 8) as usize] |= 1 << (bit % 8);
            assert!(
                !BitCursor::new(&data, 0).is_zero(96),
                "bit {bit} manqué par is_zero"
            );
        }
        assert!(BitCursor::new(&[0u8; 12], 0).is_zero(96));
        // And it advances the cursor by exactly what it consumed, so a caller
        // can keep reading after it.
        let mut cur = BitCursor::new(&[0u8; 12], 0);
        assert!(cur.is_zero(86));
        assert_eq!(cur.pos, 86);
    }

    /// `read` must refuse a width it cannot represent rather than silently
    /// masking the shift — the defect above, made impossible to reintroduce.
    #[test]
    #[should_panic(expected = "a u64 holds 64 bits")]
    fn read_refuses_more_than_a_word() {
        BitCursor::new(&[0u8; 16], 0).read(65);
    }
}
