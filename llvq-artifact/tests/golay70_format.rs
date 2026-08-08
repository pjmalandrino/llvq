//! The `Golay70` layout's contract:
//!
//! 1. **The overlay is exact.** `decode_block` — main stream with exceptions
//!    resolved — must equal `Planes14`, `Slot32` and the archive decoder bit
//!    for bit on every block of an adversarial sweep covering all four cases:
//!    conforming even class, violating even class (→ exception), odd class at
//!    L ≤ 4 *including* the residue-violating ones the B5 histogram counted
//!    (never exceptions here), and 5-level classes of both cosets
//!    (→ exceptions). The origin rides along mid-stream.
//! 2. **The main stream zeroes its exceptions.** An exception block's main
//!    record is the origin — class 0, gain copied, zero contribution — so
//!    the correction pass adds the exact contribution with nothing to cancel.
//! 3. **The offsets are the format.** Raw bits at the frozen positions
//!    (class 0..9, gain 9, golay rank 10..22, A 22..46, B 46..70, padding
//!    70..72; exception records at Planes14's published offsets) rebuild
//!    every point through an *independent* restatement of the decode rules —
//!    the lock a consistent encoder/decoder bug cannot pass. The rank field
//!    is pinned against `Golay::rank` of the independently recomputed
//!    membership plane.
//! 4. **The accounting is a formula, not a measurement**: `72·n` bits of
//!    main stream (plus the frozen word-aligned tail) and `144` per
//!    exception; on the sealed 4B the exception count is the classhist
//!    census: **11 204 181** exactly, for ~3.45 b/w.

use llvq_artifact::runtime::{
    transcode, transcode_golay70, transcode_planes14, ClassTable, Golay70Table, Layout,
    GOLAY70_A_BIT, GOLAY70_BYTES, GOLAY70_B_BIT, GOLAY70_EXC_BYTES, GOLAY70_PAD_BIT,
    GOLAY70_RANK_BIT, GOLAY70_RANK_BITS, PLANES14_BYTES, PLANES14_PAD_BIT, PLANES14_PLANE_BIT,
    PLANES14_SMASK_BIT,
};
use llvq_core::{Golay, Point, SplitMix64, DIM};
use llvq_search::fastdec::{ClassLevels, FastDecoder};
use llvq_search::index::{Indexer, N13};

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

/// Distinct |values| per mod-4 residue — the E2 rule's ingredient, restated
/// from `ClassLevels` alone so the exception census is independent of
/// `Golay70Table`.
fn residue_profile(lv: &ClassLevels) -> [u32; 4] {
    let mut r = [0u32; 4];
    for &v in &lv.values[..lv.len] {
        r[(v % 4) as usize] += 1;
    }
    r
}

/// The E2 exception rule, restated: a 5-level class of either coset, or an
/// even class with more than 2 distinct |values| in some residue. An odd
/// class at L ≤ 4 is never an exception, whatever its residue profile.
fn is_exception(lv: &ClassLevels) -> bool {
    lv.len == 5 || (!lv.odd && residue_profile(lv).iter().any(|&n| n > 2))
}

/// The frozen main-stream size: `9·n` rounded up to a word plus one word.
fn main_stream_bytes(n: usize) -> usize {
    (n * GOLAY70_BYTES).div_ceil(4) * 4 + 4
}

/// Both ends of every class (every residue profile, every level count), a
/// uniform draw over the cap-13 ball, and the origin mid-stream.
fn fixture_indices(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    indices.extend((0..20_000).map(|_| 1 + rng.next() % N13));
    indices.push(0);
    indices
}

/// Lock 1: the overlay reconstruction equals Planes14, Slot32 and the
/// archive decoder everywhere; the exception list is exactly the E2 rule's
/// blocks, strictly ascending; the accounting is the exact formula; and the
/// fixture provably exercises all four adversarial cases.
#[test]
#[cfg_attr(debug_assertions, ignore = "decodes every class boundary, run in release")]
fn golay70_overlay_matches_planes14_and_the_archive_everywhere() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);

    let mut rng = SplitMix64::new(0x70_F0FF);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let e2 = transcode_golay70(&fd, &table, &g70, &indices, &gains).expect("transcodes");
    let p14 = transcode_planes14(&fd, &table, &indices, &gains).expect("transcodes");
    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    assert_eq!(e2.n_blocks, n);
    assert_eq!(e2.data.len(), main_stream_bytes(n), "9 bytes per block plus the frozen tail");

    // The exception list is the E2 rule's blocks, in order, nothing else —
    // computed here from `ClassLevels` alone, never from `Golay70Table`.
    let mut n_even_ok = 0usize; // conforming even class
    let mut n_even_viol4 = 0usize; // even, L ≤ 4, residue violation → exception
    let mut n_odd_viol4 = 0usize; // odd, L ≤ 4, residue "violation" → NOT exception
    let mut n_l5_even = 0usize;
    let mut n_l5_odd = 0usize;
    let mut want_exc: Vec<u32> = Vec::new();
    for (b, &idx) in indices.iter().enumerate() {
        let Some(ci) = fd.class_of(idx) else { continue };
        let lv = fd.levels(ci);
        let viol = residue_profile(lv).iter().any(|&c| c > 2);
        match (lv.odd, lv.len == 5, viol) {
            (false, false, false) => n_even_ok += 1,
            (false, false, true) => n_even_viol4 += 1,
            (true, false, true) => n_odd_viol4 += 1,
            (false, true, _) => n_l5_even += 1,
            (true, true, _) => n_l5_odd += 1,
            _ => {}
        }
        if is_exception(lv) {
            want_exc.push(b as u32);
        }
    }
    assert!(n_even_ok > 0, "fixture must hold conforming even blocks");
    assert!(n_even_viol4 > 0, "fixture must hold violating even L <= 4 blocks");
    assert!(n_odd_viol4 > 0, "fixture must hold residue-violating odd L <= 4 blocks");
    assert!(n_l5_even > 0 && n_l5_odd > 0, "fixture must hold L = 5 blocks of both cosets");
    assert_eq!(e2.exc_idx, want_exc, "exceptions must be exactly the E2 rule's blocks");
    assert_eq!(e2.exc_data.len(), want_exc.len() * PLANES14_BYTES);
    // The residue-violating odd classes the B5 histogram counted are carried
    // by the main stream, not the exception table — the rule's key clause.
    for (b, &idx) in indices.iter().enumerate() {
        if let Some(ci) = fd.class_of(idx) {
            let lv = fd.levels(ci);
            if lv.odd && lv.len < 5 {
                assert!(
                    e2.exc_idx.binary_search(&(b as u32)).is_err(),
                    "block {b}: an odd L <= 4 class must never be an exception"
                );
            }
        }
    }

    // The exact formula the layout exists for.
    let want_bw = ((main_stream_bytes(n) * 8) as f64
        + (GOLAY70_EXC_BYTES * 8) as f64 * want_exc.len() as f64)
        / (24.0 * n as f64);
    assert_eq!(e2.bits_per_weight(), want_bw, "accounting drifted from the formula");

    for b in (0..n).rev() {
        let (pe, ge) = e2.decode_block(&table, &g70, b);
        let (p4, g4) = p14.decode_block(&table, b);
        let (ps, gs) = slot.decode_block(&table, b);
        let want = ix.decode(indices[b]).expect("valid");
        assert_eq!(pe, want, "block {b}: Golay70 differs from the archive");
        assert_eq!(pe, p4, "block {b}: Golay70 differs from Planes14");
        assert_eq!(pe, ps, "block {b}: Golay70 differs from Slot32");
        assert_eq!(ge, gains[b], "block {b}: gain differs");
        assert_eq!((g4, gs), (gains[b], gains[b]), "block {b}: witness gains");
    }
}

/// Lock 2: what the main stream carries. Exact on non-exception blocks; on
/// exceptions the zero-contribution origin record with the gain copied — the
/// block the GPU row pass computes before the correction pass.
#[test]
#[cfg_attr(debug_assertions, ignore = "decodes every class boundary, run in release")]
fn golay70_main_stream_is_origin_on_exceptions() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);

    let mut rng = SplitMix64::new(0x70_ABCD);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let e2 = transcode_golay70(&fd, &table, &g70, &indices, &gains).expect("transcodes");

    assert!(!e2.exc_idx.is_empty(), "fixture must exercise the overlay");
    for (b, &gain) in gains.iter().enumerate() {
        let (mp, mg) = e2.decode_main_block(&table, &g70, b);
        assert_eq!(mg, gain, "block {b}: main-stream gain");
        let exact = e2.decode_block(&table, &g70, b);
        if e2.exc_idx.binary_search(&(b as u32)).is_err() {
            assert_eq!((mp, mg), exact, "block {b}: a non-exception must be exact");
        } else {
            assert_eq!(mp, [0i32; DIM], "block {b}: exception left a nonzero main record");
            assert_ne!(exact.0, [0i32; DIM], "block {b}: exception decodes to a real point");
        }
    }
}

/// Independent restatement of the main-record decode rules, from a class's
/// `ClassLevels` and the raw fields alone — never through `Golay70Table`.
fn rebuild_from_raw(lv: &ClassLevels, c: u32, a: u32, bm: u32) -> Point {
    let mut p = [0i32; DIM];
    if lv.odd {
        for (i, pi) in p.iter_mut().enumerate() {
            let lvl = ((a >> i & 1) | (bm >> i & 1) << 1) as usize;
            assert!(lvl < lv.len, "level outside the class");
            let v = lv.values[lvl];
            let flag = u32::from(v % 4 == 3);
            *pi = if c >> i & 1 == flag { v } else { -v };
        }
    } else {
        // Residue pairs in canonical level order, rebuilt from scratch.
        let mut pairs = [[0i32; 2]; 2];
        let mut nd = [0usize; 2];
        for &v in &lv.values[..lv.len] {
            let r = usize::from(v % 4 == 2);
            pairs[r][nd[r]] = v;
            nd[r] += 1;
        }
        for (i, pi) in p.iter_mut().enumerate() {
            let r = (c >> i & 1) as usize;
            let ai = (a >> i & 1) as usize;
            assert!(ai < nd[r].max(1), "A bit outside the residue pair");
            let v = pairs[r][ai];
            *pi = if bm >> i & 1 == 1 { -v } else { v };
        }
    }
    p
}

/// Lock 3: the offsets are the format — main-stream records *and* exception
/// records re-read at the spec's fixed bit positions by a second
/// implementation that never calls `decode_block`, and the rank field pinned
/// against `Golay::rank` of the independently recomputed membership plane.
#[test]
#[cfg_attr(debug_assertions, ignore = "decodes every class boundary, run in release")]
fn golay70_offsets_are_frozen() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);
    let golay = Golay::new();

    // The hand-computed pin first: the origin with gain 1 is 9 bytes plus
    // the frozen 16-byte-total tail, the gain bit lands at stream bit 9 —
    // LSB-first, byte 1, bit 1 — and everything else is zero.
    let rt = transcode_golay70(&fd, &table, &g70, &[0], &[1]).expect("ok");
    assert_eq!(rt.data.len(), 16, "9 bytes rounded to a word, plus one word");
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
    let mut rng = SplitMix64::new(0x70_5EED);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let rt = transcode_golay70(&fd, &table, &g70, &indices, &gains).expect("ok");

    let mut ei = 0usize;
    for (b, &idx) in indices.iter().enumerate() {
        let bit0 = b as u64 * (GOLAY70_BYTES as u64 * 8);
        let id = read_bits(&rt.data, bit0, 9) as usize;
        let gain = read_bits(&rt.data, bit0 + 9, 1) as u32;
        let grank = read_bits(&rt.data, bit0 + GOLAY70_RANK_BIT, GOLAY70_RANK_BITS) as usize;
        let a = read_bits(&rt.data, bit0 + GOLAY70_A_BIT, 24) as u32;
        let bm = read_bits(&rt.data, bit0 + GOLAY70_B_BIT, 24) as u32;
        assert_eq!(
            read_bits(&rt.data, bit0 + GOLAY70_PAD_BIT, 2),
            0,
            "block {b}: padding bits 70..72 are not zero"
        );
        assert_eq!(gain, gains[b], "block {b}: gain bit at offset 9");

        let want = ix.decode(idx).expect("valid");
        let ci = fd.class_of(idx).expect("class boundaries are never the origin");
        let lv = fd.levels(ci);
        if !is_exception(lv) {
            // The membership plane, recomputed from the archive point, must
            // be exactly the codeword the rank field names — the pin that
            // survives a consistently shifted encoder/decoder pair.
            let residue = if lv.odd { 3 } else { 2 };
            let mut c = 0u32;
            for (i, &v) in want.iter().enumerate() {
                if v.rem_euclid(4) == residue {
                    c |= 1 << i;
                }
            }
            assert_eq!(
                golay.rank(c),
                Some(grank),
                "block {b}: rank field disagrees with the canonical rank"
            );
            let got = rebuild_from_raw(lv, golay.codewords()[grank], a, bm);
            assert_eq!(got, want, "block {b}: raw bits at the main-stream offsets");
            continue;
        }
        // An exception: the main record is the origin with the gain copied,
        // and the exact point sits in the exception record at Planes14's
        // frozen offsets, under the raw index `b`.
        assert_eq!(id, 0, "block {b}: exception main record must be the origin");
        assert_eq!((grank, a, bm), (0, 0, 0), "block {b}: origin record must be zero");
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
fn golay70_refuses_a_two_bit_gain_table() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 2);
    let g70 = Golay70Table::new(&fd);
    let _ = transcode_golay70(&fd, &table, &g70, &[1], &[0]);
}

/// A gain rank that overflows the 1-bit field must be refused too — pushed
/// as-is it would corrupt the Golay rank's first bit.
#[test]
#[should_panic(expected = "overflows the 1-bit field")]
fn golay70_refuses_an_overflowing_gain_value() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);
    let _ = transcode_golay70(&fd, &table, &g70, &[1], &[2]);
}

/// An index outside the ball must refuse to transcode — same contract as the
/// other layouts: a corrupt archive becomes an error, never plausible weights.
#[test]
fn golay70_refuses_an_out_of_range_index() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);
    let r = transcode_golay70(&fd, &table, &g70, &[N13 + 1], &[0]);
    assert!(r.is_err(), "an index past N13 must not transcode");
}

/// The integral proof on the sealed 4B artifact: every block of every matrix
/// reconstructs bit-identically through the Golay70 overlay and through
/// Planes14, and the exception count is the classhist census of the E2 rule:
/// **11 204 181** exactly. No per-exception search here — this sweep is
/// decode-bound, ~50 s over a 981 MB file — but that is still three orders of
/// magnitude past the rest of this crate, so it is out of the default run;
/// asked for explicitly, a missing archive is an error, never a green skip
/// (see `common::sealed_artifact_path`).
#[test]
#[ignore = "~50 s over the sealed 150 M-block archive, which must be present (--include-ignored)"]
fn the_sealed_artifact_golay70_overlay_is_exact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const EXPECTED_EXCEPTIONS: u64 = 11_204_181;

    let path = common::sealed_artifact_path();

    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let g70 = Golay70Table::new(&fd);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Chunked so the transcode-and-compare working set stays bounded; the
    // layout has no group structure, so chunk boundaries change nothing in
    // the main stream, and exception indices are chunk-local by contract.
    const CHUNK: usize = 32 * 4096;
    let verified = AtomicU64::new(0);
    let exceptions = AtomicU64::new(0);

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
                    let e2 = transcode_golay70(&fd, &table, &g70, idxs, gns)
                        .expect("transcodes");
                    let p14 = transcode_planes14(&fd, &table, idxs, gns).expect("transcodes");
                    for (b, &idx) in idxs.iter().enumerate() {
                        let want = match fd.decode(idx) {
                            Some(p) => p,
                            None => {
                                assert_eq!(idx, 0, "{}: index outside the ball", m.name);
                                [0i32; DIM]
                            }
                        };
                        let (pe, ge) = e2.decode_block(&table, &g70, b);
                        let (p4, g4) = p14.decode_block(&table, b);
                        assert_eq!(pe, want, "{} block {}: Golay70", m.name, lo + b);
                        assert_eq!(p4, want, "{} block {}: Planes14", m.name, lo + b);
                        assert_eq!(ge, gns[b], "{} block {}: gain", m.name, lo + b);
                        assert_eq!(g4, gns[b], "{} block {}: gain", m.name, lo + b);
                    }
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                    exceptions.fetch_add(e2.exc_idx.len() as u64, Ordering::Relaxed);
                });
            }
        });
    }
    let nb = verified.load(Ordering::Relaxed);
    let ne = exceptions.load(Ordering::Relaxed);
    // The payload as the spec accounts it — 72 bits per block, 144 per
    // exception — a formula over the two counts, not a sum of buffers.
    let bw = (72.0 * nb as f64 + 144.0 * ne as f64) / (nb * DIM as u64) as f64;
    println!("sealed sweep: {nb} blocks exact through the overlay, {ne} exceptions, {bw:.4} b/w payload");
    assert_eq!(
        ne, EXPECTED_EXCEPTIONS,
        "exception count disagrees with the classhist census"
    );
}
