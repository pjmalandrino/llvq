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
use llvq_core::{Point, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};

/// Bits of the class field: 383 classes plus the origin fit in 9.
pub const CLASS_BITS: u32 = 9;
/// Bytes per block of the fixed layout.
pub const FIXED96_BYTES: usize = 12;
/// Lanes per group of the grouped layout.
pub const GROUP: usize = 32;

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
    /// Total payload width in bits, class and gain fields included.
    pub width: u16,
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
        });
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let mut mask_bits = 0u32;
            let mut left = DIM as u32;
            for i in 0..lv.len.saturating_sub(1) {
                mask_bits += left;
                left -= lv.counts[i] as u32;
            }
            recs.push(ClassRecord {
                values: lv.values,
                counts: lv.counts,
                len: lv.len as u8,
                nonzero: lv.nonzero,
                width: (CLASS_BITS + gain_bits + lv.nonzero as u32 + mask_bits) as u16,
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
}

/// How blocks are addressed in the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 96 bits per block, constant stride, aligned.
    Fixed96,
    /// Byte-rounded max-width stride per group of [`GROUP`] blocks, one u32
    /// byte base per group.
    Grouped32,
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
            Layout::Grouped32 => {
                let g = b / GROUP;
                let stride = (self.bases[g + 1] - self.bases[g]) as u64 / GROUP as u64;
                (self.bases[g] as u64 + (b % GROUP) as u64 * stride) * 8
            }
        };
        let mut cur = BitCursor::new(&self.data, bit0);
        let id = cur.read(CLASS_BITS) as usize;
        let gain = cur.read(self.gain_bits) as u32;
        let rec = table.record(id);
        if id == 0 {
            return ([0; DIM], gain);
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

    match layout {
        Layout::Fixed96 => {
            out.data = vec![0u8; n * FIXED96_BYTES];
            for (b, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
                let bit0 = b as u64 * (FIXED96_BYTES as u64 * 8);
                encode_block(fd, table, idx, gain, &mut out.data, bit0)?;
            }
        }
        Layout::Grouped32 => {
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
                    width = width.max(table.record(id).width as u32);
                }
                let stride = width.div_ceil(8);
                // A partial trailing group still pays all 32 lanes, so the
                // base difference always divides by 32.
                out.data.resize(base as usize + GROUP * stride as usize, 0);
                for (l, (&idx, &gain)) in
                    blocks.iter().zip(&gains[g * GROUP..]).enumerate()
                {
                    let bit0 = (base as u64 + l as u64 * stride as u64) * 8;
                    encode_block(fd, table, idx, gain, &mut out.data, bit0)?;
                }
                base += GROUP as u32 * stride;
            }
            out.bases.push(base);
        }
    }
    Ok(out)
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

fn encode_block(
    fd: &FastDecoder,
    table: &ClassTable,
    idx: u64,
    gain: u32,
    data: &mut [u8],
    bit0: u64,
) -> Result<()> {
    let id = class_id(fd, idx)?;
    let rec = *table.record(id);
    let mut w = BitSink::new(data, bit0);
    w.push(id as u64, CLASS_BITS);
    w.push(gain as u64, table.gain_bits);
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
    let nlev = rec.len as usize;
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
        rec.width as u64,
        "class {id}: encoded width disagrees with the table"
    );
    Ok(())
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
        let mut out = 0u64;
        for i in 0..width as u64 {
            let bit = self.pos + i;
            out |= ((self.data[(bit / 8) as usize] >> (bit % 8)) as u64 & 1) << i;
        }
        self.pos += width as u64;
        out
    }
}
