//! The `Planes12x` sparse-overlay layout's contract:
//!
//! 1. **The overlay is exact.** `decode_block` — main stream with exceptions
//!    resolved — must equal `Planes14`, `Slot32` and the archive decoder bit
//!    for bit on every block of an adversarial sweep. The approximation
//!    exists only in `decode_approx_block`, and only on 5-level blocks.
//! 2. **The main stream is a capped stand-in.** Every main-stream record
//!    decodes through a ≤ 4-level class; on exception blocks the gain bit is
//!    copied and only the direction moves.
//! 3. **The offsets are the format.** Raw bits at the frozen positions
//!    (class 0..9, gain 9, smask 10..34, planes 34/58, padding 82..96; the
//!    exception records at Planes14's published offsets) rebuild every point
//!    without touching `decode_block` — the lock a *consistent*
//!    encoder/decoder bug cannot pass.
//! 4. **The accounting is a formula, not a measurement**: `96·n + 144·n_exc`
//!    bits over `24·n` weights, and on the sealed 4B the exception count is
//!    the `rtbits` census: 5 096 688 exactly, for ~4.20 b/w.

use llvq_artifact::runtime::{
    transcode, transcode_planes12x, transcode_planes14, ClassTable, Layout, PLANES12X_BYTES,
    PLANES12X_EXC_BYTES, PLANES12X_LEVEL_CAP, PLANES12X_PAD_BIT, PLANES12X_PLANE_BIT,
    PLANES12X_SHELL_CAP, PLANES12X_SMASK_BIT, PLANES14_BYTES, PLANES14_PAD_BIT,
    PLANES14_PLANE_BIT, PLANES14_SMASK_BIT,
};
use llvq_core::{Leech, Point, SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};
use llvq_search::Searcher;

mod common;

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

/// Distinct |values| of a lattice point, zero included — `lswap`'s
/// definition, restated so the cap check is independent of the transcoder.
fn distinct_magnitudes(p: &Point) -> usize {
    let mut seen = [false; 16];
    for &v in p {
        seen[v.unsigned_abs() as usize] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

/// Both ends of every class (every zero pattern, every level count), a
/// bounded uniform draw over the cap-13 ball, and the origin mid-stream.
/// 2 000 draws, not 200 000: every 5-level block costs a real
/// `nearest_angular` search, and this fixture already covers all 383 classes.
fn fixture_indices(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    indices.extend((0..2_000).map(|_| 1 + rng.next() % N13));
    indices.push(0);
    indices
}

/// Lock 1: the overlay reconstruction equals Planes14, Slot32 and the
/// archive decoder everywhere; the exception list is exactly the 5-level
/// blocks, strictly ascending; the accounting is the exact formula.
#[test]
#[cfg_attr(debug_assertions, ignore = "searches every L = 5 block, run in release")]
fn planes12x_overlay_matches_planes14_and_the_archive_everywhere() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);

    let mut rng = SplitMix64::new(0x12_F0FF);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let p12 = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("transcodes");
    let p14 = transcode_planes14(&fd, &table, &indices, &gains).expect("transcodes");
    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    assert_eq!(p12.n_blocks, n);
    assert_eq!(p12.data.len(), n * PLANES12X_BYTES, "12 bytes per block, exactly");

    // The exception list is the 5-level blocks, in order, nothing else.
    let want_exc: Vec<u32> = indices
        .iter()
        .enumerate()
        .filter(|&(_, &idx)| {
            fd.class_of(idx)
                .is_some_and(|ci| fd.levels(ci).len > PLANES12X_LEVEL_CAP)
        })
        .map(|(b, _)| b as u32)
        .collect();
    assert!(!want_exc.is_empty(), "fixture must exercise the overlay");
    assert_eq!(p12.exc_idx, want_exc, "exceptions must be exactly the L = 5 blocks");
    assert_eq!(p12.exc_data.len(), want_exc.len() * PLANES14_BYTES);

    // The exact formula the layout exists for.
    let want_bw = (96.0 * n as f64 + (PLANES12X_EXC_BYTES * 8) as f64 * want_exc.len() as f64)
        / (24.0 * n as f64);
    assert_eq!(p12.bits_per_weight(), want_bw, "accounting drifted from the formula");

    for b in (0..n).rev() {
        let (pp, gp) = p12.decode_block(&table, b);
        let (p4, g4) = p14.decode_block(&table, b);
        let (ps, gs) = slot.decode_block(&table, b);
        let want = ix.decode(indices[b]).expect("valid");
        assert_eq!(pp, want, "block {b}: Planes12x differs from the archive");
        assert_eq!(pp, p4, "block {b}: Planes12x differs from Planes14");
        assert_eq!(pp, ps, "block {b}: Planes12x differs from Slot32");
        assert_eq!(gp, gains[b], "block {b}: gain differs");
        assert_eq!((g4, gs), (gains[b], gains[b]), "block {b}: witness gains");
    }
}

/// Lock 2: what the main stream carries. Exact on ≤ 4-level blocks; on
/// exceptions a ≤ 4-magnitude direction inside the m ≤ 12 ball with the gain
/// bit copied — the block the GPU row pass computes before the correction.
#[test]
#[cfg_attr(debug_assertions, ignore = "searches every L = 5 block, run in release")]
fn planes12x_main_stream_is_the_capped_representative() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);

    let mut rng = SplitMix64::new(0x12_ABCD);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let p12 = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("transcodes");

    for (b, &idx) in indices.iter().enumerate() {
        let (ap, ag) = p12.decode_approx_block(&table, b);
        assert!(
            distinct_magnitudes(&ap) <= PLANES12X_LEVEL_CAP,
            "block {b}: main stream holds a {}-magnitude point",
            distinct_magnitudes(&ap)
        );
        assert_eq!(ag, gains[b], "block {b}: main-stream gain");
        let exact = p12.decode_block(&table, b);
        if p12.exc_idx.binary_search(&(b as u32)).is_err() {
            assert_eq!((ap, ag), exact, "block {b}: a non-exception must be exact");
            continue;
        }
        // An exception: the direction moved, the norm code did not, and the
        // representative stayed inside the published file's ball.
        assert_ne!(ap, exact.0, "block {b}: L = 5 exception left unswapped");
        assert_eq!(ag, exact.1, "block {b}: swap must copy the gain");
        let shell =
            Leech::shell_index(&ap).expect("representative is a lattice point") as u32;
        assert!(
            (2..=PLANES12X_SHELL_CAP).contains(&shell),
            "block {b}: representative on shell {shell} (source index {idx})"
        );
    }
}

/// Lock 3: the offsets are the format — main-stream records *and* exception
/// records re-read at the spec's fixed bit positions by a second
/// implementation that never calls `decode_block`.
#[test]
#[cfg_attr(debug_assertions, ignore = "searches every L = 5 block, run in release")]
fn planes12x_offsets_are_frozen() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);

    // The hand-computed pin first: the origin with gain 1 is 12 bytes, the
    // gain bit lands at stream bit 9 — LSB-first, byte 1, bit 1 — and
    // everything else is zero.
    let rt = transcode_planes12x(&fd, &table, &s, &[0], &[1]).expect("ok");
    assert_eq!(rt.data.len(), PLANES12X_BYTES);
    assert_eq!(rt.data[0], 0, "class field 0 leaves byte 0 empty");
    assert_eq!(rt.data[1], 0b0000_0010, "gain bit lands at byte 1, bit 1");
    assert!(rt.data[2..].iter().all(|&b| b == 0), "origin record is zero");
    assert!(rt.exc_idx.is_empty(), "the origin is not an exception");

    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    let mut rng = SplitMix64::new(0x12_5EED);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let rt = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("ok");

    let mut ei = 0usize;
    for (b, &idx) in indices.iter().enumerate() {
        let bit0 = b as u64 * (PLANES12X_BYTES as u64 * 8);
        let id = read_bits(&rt.data, bit0, 9) as usize;
        let gain = read_bits(&rt.data, bit0 + 9, 1) as u32;
        let smask = read_bits(&rt.data, bit0 + PLANES12X_SMASK_BIT, 24) as u32;
        let planes: Vec<u32> = PLANES12X_PLANE_BIT
            .iter()
            .map(|&off| read_bits(&rt.data, bit0 + off, 24) as u32)
            .collect();
        assert_eq!(
            read_bits(&rt.data, bit0 + PLANES12X_PAD_BIT, 14),
            0,
            "block {b}: padding bits 82..96 are not zero"
        );
        assert_eq!(gain, gains[b], "block {b}: gain bit at offset 9");

        // Rebuild the main-stream point from raw bits alone.
        let rec = table.record(id);
        let mut got = [0i32; DIM];
        for (i, gi) in got.iter_mut().enumerate() {
            let lvl = (planes[0] >> i & 1) | (planes[1] >> i & 1) << 1;
            assert!((lvl as usize) < rec.len as usize, "block {b}, slot {i}");
            let v = rec.values[lvl as usize];
            *gi = if smask >> i & 1 == 1 { -v } else { v };
        }

        let want = ix.decode(idx).expect("valid");
        let is_l5 = fd
            .class_of(idx)
            .is_some_and(|ci| fd.levels(ci).len > PLANES12X_LEVEL_CAP);
        if !is_l5 {
            assert_eq!(got, want, "block {b}: raw bits at the main-stream offsets");
            continue;
        }
        // An exception: the exact point sits in the exception record, at
        // Planes14's frozen offsets, under the raw index `b`.
        assert_eq!(rt.exc_idx[ei], b as u32, "exception list order");
        let r0 = ei as u64 * (PLANES14_BYTES as u64 * 8);
        let eid = read_bits(&rt.exc_data, r0, 9) as usize;
        let egain = read_bits(&rt.exc_data, r0 + 9, 1) as u32;
        let esmask = read_bits(&rt.exc_data, r0 + PLANES14_SMASK_BIT, 24) as u32;
        let ep: Vec<u32> = PLANES14_PLANE_BIT
            .iter()
            .map(|&off| read_bits(&rt.exc_data, r0 + off, 24) as u32)
            .collect();
        assert_eq!(
            read_bits(&rt.exc_data, r0 + PLANES14_PAD_BIT, 6),
            0,
            "exception {ei}: padding bits 106..112 are not zero"
        );
        assert_eq!(egain, gains[b], "exception {ei}: gain bit");
        let erec = table.record(eid);
        let mut egot = [0i32; DIM];
        for (i, gi) in egot.iter_mut().enumerate() {
            let lvl = (ep[0] >> i & 1) | (ep[1] >> i & 1) << 1 | (ep[2] >> i & 1) << 2;
            assert!((lvl as usize) < erec.len as usize, "exception {ei}, slot {i}");
            let v = erec.values[lvl as usize];
            *gi = if esmask >> i & 1 == 1 { -v } else { v };
        }
        assert_eq!(egot, want, "exception {ei}: raw bits at the Planes14 offsets");
        ei += 1;
    }
    assert_eq!(ei, rt.exc_idx.len(), "every exception was visited");
}

/// The gain field is bit 9, one bit, frozen — a wider table must be refused
/// loudly, not accommodated by silently shifting every offset.
#[test]
#[should_panic(expected = "freezes the gain field")]
fn planes12x_refuses_a_two_bit_gain_table() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 2);
    let _ = transcode_planes12x(&fd, &table, &s, &[1], &[0]);
}

/// A gain rank that overflows the 1-bit field must be refused too — pushed
/// as-is it would corrupt the sign mask's first bit.
#[test]
#[should_panic(expected = "overflows the 1-bit field")]
fn planes12x_refuses_an_overflowing_gain_value() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let _ = transcode_planes12x(&fd, &table, &s, &[1], &[2]);
}

/// An index outside the ball must refuse to transcode — same contract as the
/// other layouts: a corrupt archive becomes an error, never plausible weights.
#[test]
fn planes12x_refuses_an_out_of_range_index() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let r = transcode_planes12x(&fd, &table, &s, &[N13 + 1], &[0]);
    assert!(r.is_err(), "an index past N13 must not transcode");
}

/// The integral proof on the sealed 4B artifact: every block of every matrix
/// reconstructs bit-identically through the Planes12x overlay and through
/// Planes14, and the exception count is the `rtbits` census of L = 5 blocks:
/// **5 096 688** exactly. ⚠️ This runs one `nearest_angular` search per L = 5
/// block — the longest test in this crate by two orders of magnitude, ~574 s
/// of all-core work, the same bill `lswap` paid. Out of the default run for
/// that reason; asked for explicitly, a missing archive is an error, never a
/// green skip (see `common::sealed_artifact_path`).
#[test]
#[ignore = "~10 min of searches over the sealed archive, which must be present (--include-ignored)"]
fn the_sealed_artifact_planes12x_overlay_is_exact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const EXPECTED_EXCEPTIONS: u64 = 5_096_688;

    let path = common::sealed_artifact_path();

    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Chunked so the transcode-and-compare working set stays bounded; the
    // layout has no group structure, so chunk boundaries change nothing.
    // `transcode_planes12x` threads internally per call; the outer loop
    // feeds it chunks to keep the search load continuous across matrices.
    const CHUNK: usize = 32 * 4096;
    let verified = AtomicU64::new(0);
    let exceptions = AtomicU64::new(0);
    let payload_bits = AtomicU64::new(0);

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(m.shell_cap, 12, "{}: the sealed file is leech1c12", m.name);
        assert_eq!(m.centroids.len(), 2, "{}: not a 1-bit-gain matrix", m.name);
        let nchunks = m.indices.len().div_ceil(CHUNK);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nthreads {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let lo = c * CHUNK;
                    let hi = ((c + 1) * CHUNK).min(m.indices.len());
                    let idxs = &m.indices[lo..hi];
                    let gns = &m.gains[lo..hi];
                    let p12 =
                        transcode_planes12x(&fd, &table, &s, idxs, gns).expect("transcodes");
                    let p14 = transcode_planes14(&fd, &table, idxs, gns).expect("transcodes");
                    for (b, &idx) in idxs.iter().enumerate() {
                        let want = match fd.decode(idx) {
                            Some(p) => p,
                            None => {
                                assert_eq!(idx, 0, "{}: index outside the ball", m.name);
                                [0i32; DIM]
                            }
                        };
                        let (pp, gp) = p12.decode_block(&table, b);
                        let (p4, g4) = p14.decode_block(&table, b);
                        assert_eq!(pp, want, "{} block {}: Planes12x", m.name, lo + b);
                        assert_eq!(p4, want, "{} block {}: Planes14", m.name, lo + b);
                        assert_eq!(gp, gns[b], "{} block {}: gain", m.name, lo + b);
                        assert_eq!(g4, gns[b], "{} block {}: gain", m.name, lo + b);
                    }
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                    exceptions.fetch_add(p12.exc_idx.len() as u64, Ordering::Relaxed);
                    payload_bits.fetch_add(
                        p12.data.len() as u64 * 8
                            + p12.exc_idx.len() as u64 * 32
                            + p12.exc_data.len() as u64 * 8,
                        Ordering::Relaxed,
                    );
                });
            }
        });
    }
    let nb = verified.load(Ordering::Relaxed);
    let ne = exceptions.load(Ordering::Relaxed);
    let bw = payload_bits.load(Ordering::Relaxed) as f64 / (nb * DIM as u64) as f64;
    println!(
        "sealed sweep: {nb} blocks exact through the overlay, {ne} exceptions, \
         {bw:.4} b/w payload"
    );
    assert_eq!(
        ne, EXPECTED_EXCEPTIONS,
        "exception count disagrees with the rtbits census"
    );
}
