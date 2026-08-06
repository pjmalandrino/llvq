//! The `Planes14` layout's contract, proved three independent ways:
//!
//! 1. **Against the archive decoder** — `decode_block(transcode(idx))` must
//!    equal `Indexer::decode(idx)` bit for bit, over the same adversarial
//!    sweep the other layouts pass (every class boundary, 200 k uniform
//!    draws, the origin mid-stream).
//! 2. **Against `Slot32`** — the layout claims to be a bijection of the
//!    Slot32 content; every block must reconstruct identically through both.
//! 3. **Against the spec's frozen offsets** — an in-test re-reading of the
//!    raw bits at the published offsets (class 0..9, gain 9, smask 10..34,
//!    planes 34/58/82, padding 106..112) rebuilds every point without
//!    calling `decode_block` at all. An encoder and decoder sharing a
//!    permuted-plane bug would pass 1 and 2 and die here — the same reason
//!    `the_bit_order_is_lsb_first` exists for the other layouts.
//!
//! Plus the two canonicity facts round-trips cannot see (padding bits and
//! zero-slot sign bits are written 0), and the byte accounting the whole
//! layout exists for: 14 bytes per block, no bases, 4.6667 b/w.

use llvq_artifact::runtime::{
    transcode, transcode_planes14, ClassTable, Layout, PLANES14_BYTES, PLANES14_PAD_BIT,
    PLANES14_PLANE_BIT, PLANES14_SMASK_BIT,
};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};

/// LSB-first bit read straight off a byte slice — the spec's definition of
/// the stream, restated independently of the implementation under test.
fn read_bits(data: &[u8], bit0: u64, width: u32) -> u64 {
    let mut v = 0u64;
    for i in 0..width as u64 {
        let b = bit0 + i;
        v |= ((data[(b / 8) as usize] >> (b % 8)) as u64 & 1) << i;
    }
    v
}

/// Class boundaries + uniform draws + the origin mid-stream — the same
/// adversarial block set the other layouts sweep — decoded through Planes14,
/// through Slot32, and through the archive decoder, all three equal.
#[test]
#[cfg_attr(debug_assertions, ignore = "full-space sweep, run in release")]
fn planes14_matches_slot32_and_the_archive_everywhere() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);

    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    let mut rng = SplitMix64::new(0x14_F0FF);
    indices.extend((0..200_000).map(|_| 1 + rng.next() % N13));
    indices.push(0); // the origin block, mid-stream
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();

    let planes = transcode_planes14(&fd, &table, &indices, &gains).expect("transcodes");
    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    assert_eq!(planes.n_blocks, indices.len());
    for b in (0..indices.len()).rev() {
        let (pp, gp) = planes.decode_block(&table, b);
        let (ps, gs) = slot.decode_block(&table, b);
        let want = ix.decode(indices[b]).expect("valid");
        assert_eq!(pp, want, "block {b}: Planes14 differs from the archive");
        assert_eq!(pp, ps, "block {b}: Planes14 differs from Slot32");
        assert_eq!(gp, gains[b], "block {b}: gain differs");
        assert_eq!(gs, gains[b], "block {b}: Slot32 gain differs");
    }
}

/// The accounting the layout exists for: 14 bytes per block **exactly**, for
/// any block count — no group padding, no base table — and therefore
/// 112/24 = 4.6667 bits per weight, an exact quotient, not a measurement.
#[test]
fn fourteen_bytes_per_block_exactly() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x14_5EED);
    // 1 block, group-sized counts, and counts that Slot32 would pad.
    for n in [1usize, 31, 32, 33, 45, 1000] {
        let indices: Vec<u64> = (0..n).map(|_| 1 + rng.next() % N13).collect();
        let gains = vec![0u32; n];
        let rt = transcode_planes14(&fd, &table, &indices, &gains).expect("ok");
        assert_eq!(rt.data.len(), n * PLANES14_BYTES, "n = {n}: not 14 B/block");
        assert_eq!(rt.data.len(), n * 14, "PLANES14_BYTES itself drifted");
        assert_eq!(
            rt.bits_per_weight(),
            112.0 / 24.0,
            "n = {n}: payload accounting is not 4.6667 b/w"
        );
    }
}

/// The offsets are the format. Every class-boundary block is re-read at the
/// spec's fixed bit positions — class, gain, smask, the three planes, the
/// padding — and the point is rebuilt by a second implementation that never
/// touches `decode_block`. This is the test that kills a *consistent*
/// encoder/decoder bug (planes swapped on both sides, smask shifted on both
/// sides), which every round-trip would happily pass.
#[test]
fn planes14_offsets_are_frozen() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);

    // The hand-computed pin first: the origin with gain 1 is 14 bytes, the
    // gain bit lands at stream bit 9 — LSB-first, byte 1, bit 1 — and
    // everything else is zero.
    let rt = transcode_planes14(&fd, &table, &[0], &[1]).expect("ok");
    assert_eq!(rt.data.len(), PLANES14_BYTES);
    assert_eq!(rt.data[0], 0, "class field 0 leaves byte 0 empty");
    assert_eq!(rt.data[1], 0b0000_0010, "gain bit lands at byte 1, bit 1");
    assert!(rt.data[2..].iter().all(|&b| b == 0), "origin planes are zero");

    // Every class, both ends: rebuild from raw bits at the frozen offsets.
    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    let mut rng = SplitMix64::new(0x14_ABCD);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let rt = transcode_planes14(&fd, &table, &indices, &gains).expect("ok");

    for (b, &idx) in indices.iter().enumerate() {
        let bit0 = b as u64 * (PLANES14_BYTES as u64 * 8);
        let id = read_bits(&rt.data, bit0, 9) as usize;
        let gain = read_bits(&rt.data, bit0 + 9, 1) as u32;
        let smask = read_bits(&rt.data, bit0 + PLANES14_SMASK_BIT, 24) as u32;
        let planes: Vec<u32> = PLANES14_PLANE_BIT
            .iter()
            .map(|&off| read_bits(&rt.data, bit0 + off, 24) as u32)
            .collect();
        assert_eq!(
            read_bits(&rt.data, bit0 + PLANES14_PAD_BIT, 6),
            0,
            "block {b}: padding bits 106..112 are not zero"
        );
        assert_eq!(gain, gains[b], "block {b}: gain bit at offset 9");

        let want = ix.decode(idx).expect("valid");
        let rec = table.record(id);
        let mut got = [0i32; DIM];
        for (i, gi) in got.iter_mut().enumerate() {
            let lvl = (planes[0] >> i & 1)
                | (planes[1] >> i & 1) << 1
                | (planes[2] >> i & 1) << 2;
            assert!((lvl as usize) < rec.len as usize, "block {b}, slot {i}");
            let v = rec.values[lvl as usize];
            *gi = if smask >> i & 1 == 1 { -v } else { v };
        }
        assert_eq!(got, want, "block {b}: raw bits at the spec offsets");
    }
}

/// The two canonicity facts a round-trip cannot see, because the decoder
/// ignores padding and −0 = 0: padding bits are written zero, and zero
/// slots carry a 0 sign bit. Equal inputs must transcode to equal bytes.
#[test]
fn planes14_padding_and_zero_slots_are_canonical() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);
    // Class boundaries cover every class — including every zero pattern —
    // plus the origin, whose record must be all-zero past the gain bit.
    let mut indices: Vec<u64> = (0..fd.n_classes())
        .map(|ci| fd.class_range(ci).0)
        .collect();
    indices.push(0);
    let gains = vec![0u32; indices.len()];
    let rt = transcode_planes14(&fd, &table, &indices, &gains).expect("ok");
    for (b, &idx) in indices.iter().enumerate() {
        let bit0 = b as u64 * (PLANES14_BYTES as u64 * 8);
        assert_eq!(
            read_bits(&rt.data, bit0 + PLANES14_PAD_BIT, 6),
            0,
            "block {b}: padding carries bits"
        );
        let smask = read_bits(&rt.data, bit0 + PLANES14_SMASK_BIT, 24) as u32;
        let p = if idx == 0 {
            [0i32; DIM]
        } else {
            ix.decode(idx).expect("valid")
        };
        for (i, &v) in p.iter().enumerate() {
            if v == 0 {
                assert_eq!(
                    smask >> i & 1,
                    0,
                    "block {b}: zero slot {i} carries a sign bit"
                );
            }
        }
    }
}

/// The gain field is bit 9, one bit, frozen — a wider table must be refused
/// loudly, not accommodated by silently shifting every offset.
#[test]
#[should_panic(expected = "freezes the gain field")]
fn planes14_refuses_a_two_bit_gain_table() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 2);
    let _ = transcode_planes14(&fd, &table, &[1], &[0]);
}

/// A gain rank that overflows the 1-bit field must be refused too — pushed
/// as-is it would corrupt the sign mask's first bit.
#[test]
#[should_panic(expected = "overflows the 1-bit field")]
fn planes14_refuses_an_overflowing_gain_value() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let _ = transcode_planes14(&fd, &table, &[1], &[2]);
}

/// An index outside the ball must refuse to transcode — same contract as the
/// other layouts: a corrupt archive becomes an error, never plausible weights.
#[test]
fn planes14_refuses_an_out_of_range_index() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let r = transcode_planes14(&fd, &table, &[N13 + 1], &[0]);
    assert!(r.is_err(), "an index past N13 must not transcode");
}

/// The integral proof on the sealed 4B artifact: every block of every matrix,
/// decoded through Planes14 and through Slot32, must equal the archive
/// decoder bit for bit. Skipped (loudly) when the file is not on this
/// machine; ignored in debug like every heavy sweep in this repo.
#[test]
#[cfg_attr(debug_assertions, ignore = "full 150 M-block artifact, run in release")]
fn the_sealed_artifact_decodes_identically_in_both_layouts() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let path = match std::env::var_os("HOME") {
        Some(h) => std::path::Path::new(&h).join("llvq-q4b.llvq"),
        None => {
            eprintln!("SKIP: no $HOME to find the sealed artifact");
            return;
        }
    };
    if !path.exists() {
        eprintln!("SKIP: {} not present on this machine", path.display());
        return;
    }

    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Work units aligned to the Slot32 group so chunked transcodes see
    // whole groups; alignment is a convenience, not a correctness need —
    // only decoded points are compared, never chunk bytes.
    const CHUNK: usize = 32 * 4096;
    let verified = AtomicU64::new(0);

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert!(m.shell_cap <= 13, "{}: cap {}", m.name, m.shell_cap);
        assert_eq!(m.centroids.len(), 2, "{}: not a 1-bit-gain matrix", m.name);
        let nchunks = m.indices.len().div_ceil(CHUNK);
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let lo = c * CHUNK;
                    let hi = ((c + 1) * CHUNK).min(m.indices.len());
                    let idxs = &m.indices[lo..hi];
                    let gns = &m.gains[lo..hi];
                    let planes =
                        transcode_planes14(&fd, &table, idxs, gns).expect("transcodes");
                    let slot = transcode(&fd, &table, idxs, gns, Layout::Slot32)
                        .expect("transcodes");
                    for (b, &idx) in idxs.iter().enumerate() {
                        let want = match fd.decode(idx) {
                            Some(p) => p,
                            None => {
                                assert_eq!(idx, 0, "{}: index outside the ball", m.name);
                                [0i32; DIM]
                            }
                        };
                        let (pp, gp) = planes.decode_block(&table, b);
                        let (ps, gs) = slot.decode_block(&table, b);
                        assert_eq!(pp, want, "{} block {}: Planes14", m.name, lo + b);
                        assert_eq!(ps, want, "{} block {}: Slot32", m.name, lo + b);
                        assert_eq!(gp, gns[b], "{} block {}: gain", m.name, lo + b);
                        assert_eq!(gs, gns[b], "{} block {}: gain", m.name, lo + b);
                    }
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                });
            }
        });
    }
    println!(
        "sealed sweep: {} blocks verified identical in Planes14 and Slot32",
        verified.load(Ordering::Relaxed)
    );
}
