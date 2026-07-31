//! # Gate G6 — bit packing
//!
//! A 47-bit index stored in a 48-bit slot wastes exactly the bit the shell cap
//! was trying to win, so the artifact packs values back to back. That makes
//! the bit order a **format decision**: a reader that disagrees about it
//! returns different weights rather than failing, which is the worst kind of
//! bug. These tests pin it.

use llvq_core::SplitMix64;
use llvq_search::pack::{BitReader, BitWriter};

/// The byte layout, spelled out on a case small enough to read by hand.
///
/// Three 3-bit values 0b101, 0b011, 0b110 must produce the bit string
/// 101 011 110, i.e. 0b10101111 0b0xxxxxxx with the tail zero-padded.
#[test]
fn the_bit_order_is_most_significant_first() {
    let mut w = BitWriter::new();
    w.push(0b101, 3);
    w.push(0b011, 3);
    w.push(0b110, 3);
    assert_eq!(w.bit_len(), 9);
    let bytes = w.finish();
    assert_eq!(bytes.len(), 2, "9 bits occupy 2 bytes");
    assert_eq!(bytes[0], 0b1010_1111, "first byte: 101 011 11");
    assert_eq!(bytes[1], 0b0000_0000, "second: trailing 0, then zero padding");
}

#[test]
fn round_trips_at_every_width() {
    for width in 1..=64u32 {
        let mut w = BitWriter::new();
        let hi = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let vals = [0, 1, hi, hi / 2, hi / 3];
        for v in vals {
            w.push(v, width);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for v in vals {
            assert_eq!(r.read(width), v, "width {width}, value {v}");
        }
    }
}

/// The real case: 48-bit indices packed dense, with a gain bit interleaved.
#[test]
fn round_trips_an_index_stream() {
    let mut rng = SplitMix64::new(0x6_9AC1);
    const N: usize = 4096;
    const IDX: u32 = 47;
    let idx: Vec<u64> = (0..N)
        .map(|_| (rng.next() >> (64 - IDX)) % llvq_search::index::N13)
        .collect();
    let gains: Vec<u64> = (0..N).map(|_| rng.next() & 1).collect();

    let mut w = BitWriter::with_capacity(N as u64 * (IDX + 1) as u64);
    for (i, g) in idx.iter().zip(gains.iter()) {
        w.push(*i, IDX);
        w.push(*g, 1);
    }
    assert_eq!(w.bit_len(), N as u64 * 48);
    let bytes = w.finish();
    assert_eq!(
        bytes.len(),
        N * 6,
        "48 bits per block is exactly 6 bytes — no padding waste"
    );

    let mut r = BitReader::new(&bytes);
    for (i, g) in idx.iter().zip(gains.iter()) {
        assert_eq!(r.read(IDX), *i);
        assert_eq!(r.read(1), *g);
    }
    assert_eq!(r.bit_pos(), N as u64 * 48, "the stream is fully consumed");
}

/// Mixed widths in one stream, which is what a real header does.
#[test]
fn round_trips_mixed_widths() {
    let mut rng = SplitMix64::new(0x6_9AC2);
    let plan: Vec<(u64, u32)> = (0..2000)
        .map(|_| {
            let w = 1 + (rng.next() % 64) as u32;
            let hi = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
            (rng.next() & hi, w)
        })
        .collect();

    let mut bw = BitWriter::new();
    for (v, w) in &plan {
        bw.push(*v, *w);
    }
    let total: u64 = plan.iter().map(|(_, w)| *w as u64).sum();
    assert_eq!(bw.bit_len(), total);
    let bytes = bw.finish();
    assert_eq!(bytes.len() as u64, total.div_ceil(8), "no wasted bytes");

    let mut r = BitReader::new(&bytes);
    for (v, w) in &plan {
        assert_eq!(r.read(*w), *v, "width {w}");
    }
}

/// A value wider than its slot must be rejected, not truncated: a silently
/// truncated index is a *valid* index for a different lattice point, so the
/// artifact would decode to plausible, wrong weights.
#[test]
#[should_panic(expected = "does not fit")]
fn an_oversized_value_panics_rather_than_truncating() {
    let mut w = BitWriter::new();
    w.push(0b1_0000, 4);
}

#[test]
#[should_panic(expected = "past the end")]
fn reading_past_the_end_panics() {
    let mut w = BitWriter::new();
    w.push(1, 4);
    let bytes = w.finish();
    let mut r = BitReader::new(&bytes);
    r.read(9);
}

/// Zero-width pushes are no-ops — the zero-gain-bit configuration writes no
/// gain field at all, and must not shift the stream by a single bit.
#[test]
fn zero_width_is_a_no_op() {
    let mut w = BitWriter::new();
    w.push(0, 0);
    w.push(0b1011, 4);
    w.push(0, 0);
    w.push(0b0110, 4);
    assert_eq!(w.bit_len(), 8);
    let bytes = w.finish();
    assert_eq!(bytes, vec![0b1011_0110]);
}
