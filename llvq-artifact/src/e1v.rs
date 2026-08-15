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
//! used**: fed the per-stage widths it reproduces the published 52,869
//! (class-major) and 53,332 (file order) to 5e-3, and their 2,3709 b/weight to
//! 5e-5 (`llvq-artifact/tests/p5_cns_addressing.rs`). Fed the CNS's per-kind
//! widths — the ones a division-free decoder needs, amendment É0 — it gives
//! **53,7370** and **2,3877**.
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

/// The `E1v` stream: fixed-stride headers, warp-scanned payloads, one base word
/// per group.
pub struct E1vBlocks {
    pub n_blocks: usize,
    /// Bit-packed groups, LSB-first within each byte. Every group starts on a
    /// word boundary.
    pub data: Vec<u8>,
    /// Word offset of each group's start in `data`.
    pub bases: Vec<u32>,
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
/// Allocation is inside on purpose: C4's `T` is *du tableau au tampon,
/// allocation comprise, I/O disque exclue*, and a version that took a
/// pre-allocated buffer would be timing a different thing than `Planes14`'s
/// transcoder, which allocates its own.
pub fn transcode_e1v(
    fd: &FastDecoder,
    golay: &Golay,
    indices: &[u64],
    gains: &[u32],
) -> Result<E1vBlocks> {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert_eq!(
        indices.len() % E1V_GROUP,
        0,
        "E1v addresses whole groups of {E1V_GROUP}; the caller pads or splits"
    );
    let n = indices.len();
    let n_groups = n / E1V_GROUP;

    // One layout per class, built once. The origin has none: its record is the
    // header alone (P5 §2.2).
    let layouts: Vec<CnsLayout> = (0..fd.n_classes())
        .map(|ci| cns_layout(fd, golay, ci))
        .collect();

    // Pass 1 — the sizes, so the buffer is allocated once rather than grown.
    let mut recs: Vec<CnsRecord> = Vec::with_capacity(n);
    let mut group_words: Vec<u32> = Vec::with_capacity(n_groups);
    let mut total_words = 0u32;
    let mut bases = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut bits = HEADER_BITS * E1V_GROUP as u64;
        for b in g * E1V_GROUP..(g + 1) * E1V_GROUP {
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
        group_words.push(words);
        total_words += words;
    }

    // Pass 2 — the bits.
    let mut data = vec![0u8; total_words as usize * 4];
    for g in 0..n_groups {
        let base = u64::from(bases[g]) * 32;
        let mut cursor = base + HEADER_BITS * E1V_GROUP as u64;
        for l in 0..E1V_GROUP {
            let rec = &recs[g * E1V_GROUP + l];
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
    }

    Ok(E1vBlocks {
        n_blocks: n,
        data,
        bases,
    })
}

/// The origin's class field: all nine bits set, a value no real class takes
/// (the codebook has 383).
pub const E1V_ORIGIN_ID: u64 = (1 << E1V_CLASS_BITS) - 1;

impl E1vBlocks {
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
        let (g, l) = (b / E1V_GROUP, b % E1V_GROUP);
        let base = u64::from(self.bases[g]) * 32;

        // The warp-scan: every lane before this one contributes its payload
        // width, and each of those widths is read from that lane's own header.
        let mut cursor = base + HEADER_BITS * E1V_GROUP as u64;
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
