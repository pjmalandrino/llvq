//! **`E1v` — the byte stream the CNS's fields go into.**
//!
//! `proofs/preregistration-p5-2026-08-14.md`. [`llvq_search::cns`] produces
//! *fields*; C4 times a transcode *to a buffer*, and a buffer needs an
//! addressing. This module is that addressing, and nothing else: it invents no
//! field, no width and no order.
//!
//! ## The group, and why the headers come first
//!
//! Blocks are grouped **32 at a time**, in file order. Within a group:
//!
//! ```text
//!   [32 headers, 10 bits each, fixed stride]  [rank payloads, variable]  [pad to a word]
//! ```
//!
//! The headers are first and at a **fixed** stride because a lane cannot find
//! its own record otherwise: a record's width depends on its class, its class
//! lives in its header, and a header at a variable offset would have to be found
//! before it could be read. Fixed-stride headers break that circle — every lane
//! reads its own class at `10·l`, and only then does the group's **warp-scan**
//! (a prefix sum over the 32 widths) say where its payload begins.
//!
//! One `u32` per group holds where that group's payload starts, in words. Same
//! shape as `Grouped32`'s and `Planes12x`'s base tables, deliberately: a layout
//! that invented a third convention would be priced against the others while
//! paying a different addressing bill.
//!
//! ## The accounting, and the one number it must reproduce
//!
//! A group costs `32 + ceil(Σ widths / 32) × 32` bits — the base word, then the
//! payload rounded to a whole word. That model was **validated before it was
//! used**: fed the per-stage widths it reproduces the published 52.869
//! (class-major) and 53.332 (file order) to 5e-3, and their 2.3709 b/weight to
//! 5e-5 (`llvq-artifact/tests/p5_cns_addressing.rs`). Fed the CNS's per-kind
//! widths — the ones a division-free decoder needs, amendment É0 — it gives
//! **53.7370** and **2.3877**.
//!
//! ## What this module does NOT claim
//!
//! It does not claim to be fast, and it does not claim to be the served path.
//! C4 asks one question — is transcoding to it within 2× of `Planes14`? — and
//! the answer is a wall-clock, measured in `bin/e1vbench`, not predicted here.

use crate::Result;
use llvq_core::{Golay, DIM};
use llvq_search::cns::{cns_encode, cns_layout, lg_ceil, CnsLayout, CnsRecord, HEADER_BITS};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};

/// Blocks per addressing group — the warp.
pub const E1V_GROUP: usize = 32;

/// Bits of the class field. The gain takes the tenth bit of the header.
pub const E1V_CLASS_BITS: u32 = 9;

/// ## Two cuts of the same stream, and only one of them can be served
///
/// The groups above are cut **in file order**, 32 blocks at a time. X3
/// (`docs/mesures/x3-alignement-warp-2026-08-15.txt`) showed that the served
/// matvec cannot read that cut: it puts one warp per output **row**
/// (`planes.cu`, `b0r = row · nblocks`, `j = jlo + lane`), so lane `l` handles
/// group rank `(row·nblocks + lane) mod 32` — a rotation that vanishes only
/// when `nblocks ≡ 0 (mod 32)`. On the five shapes of the published 4B it never
/// does: **0 aligned blocks out of 150,681,600**. Every warp would straddle two
/// groups, read two base words and scan two header regions.
///
/// So there is a second cut, [`transcode_e1v_rows`], where **a group never
/// straddles a row**: the last group of each row is *partial*. It is the only
/// one a warp can read, and it is what a CUDA arm must be measured on.
///
/// 🔎 **A partial group is cheap here, and that is not a detail — it is the
/// whole reason E1v survives this where `E1c14` died.** An `E1c` group costs
/// `24·(1+planes)` words whatever the occupancy, so a group of 10 blocks costs
/// the price of 32 and alignment can only be bought by padding rows out to a
/// multiple of 32 blocks: **+15.47%**, which made `E1c14` bigger than the
/// layout it replaces. An `E1v` group costs one base word plus the sum of its
/// records, rounded to a word, so a partial group costs what its records cost.
/// The variable width — everything E1v pays for in decode complexity — is
/// exactly what saves it.
///
/// A partial group of `k` blocks writes `k` headers, not 32, so its payloads
/// begin at `10·k` rather than at 320. `k` is **derived, never stored**:
/// `min(32, nblocks − 32·g)`, which is what the kernel has in hand.
///
/// The `E1v` stream: fixed-stride headers, warp-scanned payloads, one base word
/// per group.
pub struct E1vBlocks {
    pub n_blocks: usize,
    /// Bit-packed groups, LSB-first within each byte. Every group starts on a
    /// word boundary.
    pub data: Vec<u8>,
    /// Word offset of each group's start in `data`.
    pub bases: Vec<u32>,
    /// Blocks per row when the stream is **row-aligned**; `None` in file order.
    ///
    /// One scalar per matrix, and deliberately not a per-group table: the
    /// served kernel has `nblocks` and derives everything else, so a host that
    /// consulted a stored table would be proving a property the kernel cannot
    /// use. It costs no bits and [`Self::bits_per_weight`] counts none for it.
    pub row_blocks: Option<usize>,
}

impl E1vBlocks {
    /// Bits the stream spends per weight — **addressing included**, base words
    /// and padding counted, which is what makes this comparable to the other
    /// layouts' `bits_per_weight`.
    pub fn bits_per_weight(&self) -> f64 {
        (self.data.len() as u64 * 8 + self.bases.len() as u64 * 32) as f64
            / (self.n_blocks * DIM) as f64
    }
}

/// Read `width` bits at `bit`, LSB-first — the reader every test uses to walk
/// the map by hand rather than through the writer's own arithmetic.
pub fn read_bits(data: &[u8], bit: u64, width: u32) -> u64 {
    debug_assert!(width <= 64);
    let mut v = 0u64;
    for i in 0..u64::from(width) {
        let p = bit + i;
        if (data[(p / 8) as usize] >> (p % 8)) & 1 == 1 {
            v |= 1 << i;
        }
    }
    v
}

fn write_bits(data: &mut [u8], bit: u64, value: u64, width: u32) {
    for i in 0..u64::from(width) {
        if (value >> i) & 1 == 1 {
            let p = bit + i;
            data[(p / 8) as usize] |= 1 << (p % 8);
        }
    }
}

/// Width in bits of every field of a record, in the order they are written.
///
/// The header is **not** included: it lives at the group's fixed-stride prefix,
/// not in the payload.
fn payload_fields(l: &CnsLayout) -> ([u32; 2 + 2 * MAX_KINDS], usize) {
    let mut f = [0u32; 2 + 2 * MAX_KINDS];
    let mut n = 0;
    let push = |f: &mut [u32; 2 + 2 * MAX_KINDS], n: &mut usize, w: u32| {
        f[*n] = w;
        *n += 1;
    };
    push(&mut f, &mut n, lg_ceil(u128::from(l.golay_radix)));
    for &r in l.on_radices.iter().take(l.n_on) {
        push(&mut f, &mut n, lg_ceil(u128::from(r)));
    }
    for &r in l.off_radices.iter().take(l.n_off) {
        push(&mut f, &mut n, lg_ceil(u128::from(r)));
    }
    push(&mut f, &mut n, lg_ceil(u128::from(l.s_w)));
    push(&mut f, &mut n, lg_ceil(u128::from(l.s_f)));
    (f, n)
}

/// The field values of a record, in the same order [`payload_fields`] gives
/// their widths.
fn payload_values(l: &CnsLayout, r: &CnsRecord) -> ([u64; 2 + 2 * MAX_KINDS], usize) {
    let mut v = [0u64; 2 + 2 * MAX_KINDS];
    let mut n = 0;
    v[n] = u64::from(r.golay);
    n += 1;
    for j in 0..l.n_on {
        v[n] = r.on[j];
        n += 1;
    }
    for j in 0..l.n_off {
        v[n] = r.off[j];
        n += 1;
    }
    v[n] = r.sw;
    n += 1;
    v[n] = r.sf;
    n += 1;
    (v, n)
}

/// Payload width of one block, header excluded.
fn payload_bits(l: &CnsLayout) -> u64 {
    l.bits() - HEADER_BITS
}

/// **The transcoder C4 times.** `(indices, gains)` in, one byte buffer out.
///
/// Allocation is inside on purpose: C4's `T` is *from array to buffer,
/// allocation included, disk I/O excluded*, and a version that took a
/// pre-allocated buffer would be timing a different thing than `Planes14`'s
/// transcoder, which allocates its own.
pub fn transcode_e1v(
    fd: &FastDecoder,
    golay: &Golay,
    indices: &[u64],
    gains: &[u32],
) -> Result<E1vBlocks> {
    assert_eq!(
        indices.len() % E1V_GROUP,
        0,
        "E1v addresses whole groups of {E1V_GROUP}; the caller pads or splits"
    );
    transcode_groups(fd, golay, indices, gains, None)
}

/// **The servable cut**: the same stream, cut so that a group never straddles a
/// row.
///
/// `indices` is one matrix in row-major order and `row_blocks` its blocks per
/// row (`d_in / 24`). Each row becomes `ceil(row_blocks / 32)` groups, the last
/// of them partial — see the type's own documentation for why a partial group
/// is cheap here and ruinous for `E1c`.
///
/// This is the cut a CUDA arm must be measured on. Measuring the file-order cut
/// on a warp-per-row matvec would price a misalignment and call it a layout —
/// the mistake X3 refused to let `E1c14` be judged by.
pub fn transcode_e1v_rows(
    fd: &FastDecoder,
    golay: &Golay,
    indices: &[u64],
    gains: &[u32],
    row_blocks: usize,
) -> Result<E1vBlocks> {
    assert!(row_blocks > 0, "a row holds at least one block");
    assert_eq!(
        indices.len() % row_blocks,
        0,
        "a row-aligned stream is cut into whole rows of {row_blocks} blocks"
    );
    transcode_groups(fd, golay, indices, gains, Some(row_blocks))
}

/// How many blocks each group holds, in stream order — the only thing the two
/// cuts disagree about.
fn group_lens(n: usize, row_blocks: Option<usize>) -> Vec<usize> {
    match row_blocks {
        None => vec![E1V_GROUP; n / E1V_GROUP],
        Some(rb) => {
            let per_row = rb.div_ceil(E1V_GROUP);
            let mut v = Vec::with_capacity((n / rb) * per_row);
            for _ in 0..n / rb {
                for g in 0..per_row {
                    v.push(E1V_GROUP.min(rb - g * E1V_GROUP));
                }
            }
            v
        }
    }
}

fn transcode_groups(
    fd: &FastDecoder,
    golay: &Golay,
    indices: &[u64],
    gains: &[u32],
    row_blocks: Option<usize>,
) -> Result<E1vBlocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    let n = indices.len();
    let lens = group_lens(n, row_blocks);
    debug_assert_eq!(lens.iter().sum::<usize>(), n, "the groups partition the blocks");

    // One layout per class, built once. The origin has none: its record is the
    // header alone (P5 §2.2).
    let layouts: Vec<CnsLayout> = (0..fd.n_classes())
        .map(|ci| cns_layout(fd, golay, ci))
        .collect();

    // Pass 1 — the sizes, so the buffer is allocated once rather than grown.
    let mut recs: Vec<CnsRecord> = Vec::with_capacity(n);
    let mut total_words = 0u32;
    let mut bases = Vec::with_capacity(lens.len());
    let mut first = 0usize;
    for &len in &lens {
        // A partial group writes `len` headers, not 32: the header region is
        // what the payloads start after, so a group that reserved 32 of them
        // would leave holes the scan does not account for.
        let mut bits = HEADER_BITS * len as u64;
        for b in first..first + len {
            let gain = u8::try_from(gains[b]).expect("one gain bit");
            let rec = cns_encode(fd, indices[b], gain)
                .ok_or(crate::Error::IndexOutOfRange {
                    name: "e1v".into(),
                    index: indices[b],
                })?;
            if let Some(ci) = rec.class {
                bits += payload_bits(&layouts[ci]);
            }
            recs.push(rec);
        }
        let words = u32::try_from(bits.div_ceil(32)).expect("group fits u32 words");
        bases.push(total_words);
        total_words += words;
        first += len;
    }

    // Pass 2 — the bits.
    let mut data = vec![0u8; total_words as usize * 4];
    let mut first = 0usize;
    for (g, &len) in lens.iter().enumerate() {
        let base = u64::from(bases[g]) * 32;
        let mut cursor = base + HEADER_BITS * len as u64;
        for l in 0..len {
            let rec = &recs[first + l];
            // The header, at its fixed stride — the only field a lane can read
            // before it knows anything.
            let id = rec.class.map_or(u64::from((1u32 << E1V_CLASS_BITS) - 1), |ci| ci as u64);
            write_bits(&mut data, base + HEADER_BITS * l as u64, id, E1V_CLASS_BITS);
            write_bits(
                &mut data,
                base + HEADER_BITS * l as u64 + u64::from(E1V_CLASS_BITS),
                u64::from(rec.gain),
                1,
            );
            let Some(ci) = rec.class else { continue };
            let (widths, nf) = payload_fields(&layouts[ci]);
            let (values, nv) = payload_values(&layouts[ci], rec);
            debug_assert_eq!(nf, nv);
            for i in 0..nf {
                write_bits(&mut data, cursor, values[i], widths[i]);
                cursor += u64::from(widths[i]);
            }
        }
        first += len;
    }

    Ok(E1vBlocks {
        n_blocks: n,
        data,
        bases,
        row_blocks,
    })
}

/// The origin's class field: all nine bits set, a value no real class takes
/// (the codebook has 383).
pub const E1V_ORIGIN_ID: u64 = (1 << E1V_CLASS_BITS) - 1;

impl E1vBlocks {
    /// Where block `b` lives: `(group, lane, blocks in that group)`.
    ///
    /// Derived from `row_blocks` alone — **never** from a stored per-group
    /// table — because that is all a kernel has: it holds `nblocks`, its lane
    /// index and its row, and everything else is arithmetic. A host reader that
    /// consulted a table would prove a property no kernel can use.
    pub fn locate(&self, b: usize) -> (usize, usize, usize) {
        assert!(b < self.n_blocks, "block {b} is past the stream");
        match self.row_blocks {
            None => (b / E1V_GROUP, b % E1V_GROUP, E1V_GROUP),
            Some(rb) => {
                let per_row = rb.div_ceil(E1V_GROUP);
                let (row, j) = (b / rb, b % rb);
                let g = j / E1V_GROUP;
                (
                    row * per_row + g,
                    j % E1V_GROUP,
                    E1V_GROUP.min(rb - g * E1V_GROUP),
                )
            }
        }
    }

    /// Read block `b` back out of the bytes and decode it.
    ///
    /// The **only** reader of this stream, and it walks the map the way the
    /// spec describes it — fixed-stride header, then a prefix sum over the
    /// group's widths — rather than remembering where the writer put things.
    pub fn decode_block(
        &self,
        fd: &FastDecoder,
        golay: &Golay,
        layouts: &[CnsLayout],
        b: usize,
    ) -> ([i32; DIM], u8) {
        let (g, l, len) = self.locate(b);
        let base = u64::from(self.bases[g]) * 32;

        // The warp-scan: every lane before this one contributes its payload
        // width, and each of those widths is read from that lane's own header.
        let mut cursor = base + HEADER_BITS * len as u64;
        let mut here = None;
        for i in 0..=l {
            let hb = base + HEADER_BITS * i as u64;
            let id = read_bits(&self.data, hb, E1V_CLASS_BITS);
            let gain = read_bits(&self.data, hb + u64::from(E1V_CLASS_BITS), 1) as u8;
            let layout = (id != E1V_ORIGIN_ID).then(|| &layouts[id as usize]);
            if i == l {
                here = Some((layout, gain, cursor));
                break;
            }
            if let Some(lay) = layout {
                cursor += payload_bits(lay);
            }
        }
        let (layout, gain, mut cur) = here.expect("the loop reaches l");
        let Some(lay) = layout else {
            return ([0i32; DIM], gain);
        };

        let (widths, nf) = payload_fields(lay);
        let mut rec = CnsRecord {
            class: None,
            gain,
            ..CnsRecord::default()
        };
        let field = |w: u32, cur: &mut u64| -> u64 {
            let v = read_bits(&self.data, *cur, w);
            *cur += u64::from(w);
            v
        };
        let mut i = 0;
        rec.golay = field(widths[i], &mut cur) as u32;
        i += 1;
        for j in 0..lay.n_on {
            rec.on[j] = field(widths[i], &mut cur);
            i += 1;
        }
        for j in 0..lay.n_off {
            rec.off[j] = field(widths[i], &mut cur);
            i += 1;
        }
        rec.sw = field(widths[i], &mut cur);
        i += 1;
        rec.sf = field(widths[i], &mut cur);
        debug_assert_eq!(i + 1, nf);

        // The class id is recovered from the header, not carried through.
        let hb = base + HEADER_BITS * l as u64;
        rec.class = Some(read_bits(&self.data, hb, E1V_CLASS_BITS) as usize);
        (llvq_search::cns::cns_decode(fd, golay, &rec), gain)
    }
}
