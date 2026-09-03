//! The `E1c` transposed layouts' contract — item X1/X2 of
//! `docs/archive/spec-memoire-extreme-2026-08-12.md`.
//!
//! 1. **The repack is exact.** `E1c14Blocks::decode_block` must equal
//!    `PlanesBlocks::decode_block` and the archive decoder bit for bit, and
//!    `E1c12Blocks::decode_approx_block` must equal
//!    `Planes12xBlocks::decode_approx_block` — same swapped representatives
//!    on exception blocks, same exact records everywhere else.
//! 2. **The word map is the format.** Raw words read at the frozen positions
//!    (`header_words()` header words, then slot-major `1 + planes` words per
//!    slot, lane `l` at bit `l`) rebuild every field without touching the
//!    module's own reader — the lock a *consistent* pack/unpack bug cannot
//!    pass, and the reason the unit tests' bijection check is not enough.
//! 3. **The accounting is a formula.** `group_words(p)·32` bits per group of
//!    32 blocks, exactly, plus `Planes12x`'s own exception table unchanged.
//!    On the sealed 4B that is 4.4167 and 3.6196 b/w of payload.
//!
//! The sealed sweep is `#[ignore]`d: it reads a 981 MB archive that is not in
//! the repo, and it fails loudly rather than skipping green when the file is
//! absent (see `common::sealed_artifact_path`).

use llvq_artifact::e1c::{
    group_words, E1c12Blocks, E1c14Blocks, E1C12_PLANES, E1C14_PLANES, E1C_GROUP,
};
use llvq_artifact::runtime::{
    transcode_planes12x, transcode_planes14, ClassTable, PLANES12X_BYTES, PLANES14_BYTES,
};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::Searcher;

mod common;

/// Read word `i` of group `g` straight out of the byte stream, little-endian
/// — the spec's definition of the layout, restated independently of the
/// module under test.
fn word(data: &[u8], planes: usize, g: usize, i: usize) -> u32 {
    let o = (g * group_words(planes) + i) * 4;
    u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
}

/// Rebuild block `b`'s fields from the raw words, walking the word map by
/// hand: header bits `10·l` of the header region, then slot-major fields at
/// `header_words() + j·(1+planes) + f`, lane `l` at bit `l`.
///
/// Deliberately duplicates `untranspose`'s job. A test that called the
/// module's own reader would pass on any self-consistent permutation of the
/// map — the equivalent-mutant trap `CLAUDE.md` §5 is about, and one this
/// layout is unusually exposed to because a transposition is its own family
/// of near-misses.
fn fields_by_hand(data: &[u8], planes: usize, b: usize) -> (u32, u32, u32, [u32; 3]) {
    let (g, l) = (b / E1C_GROUP, b % E1C_GROUP);
    let bit = 10 * l;
    let (w, sh) = (bit / 32, bit % 32);
    let mut win = u64::from(word(data, planes, g, w));
    if sh + 10 > 32 {
        win |= u64::from(word(data, planes, g, w + 1)) << 32;
    }
    let hdr = ((win >> sh) & 0x3ff) as u32;
    let mut smask = 0u32;
    let mut pl = [0u32; 3];
    for j in 0..DIM {
        for f in 0..=planes {
            let bit = (word(data, planes, g, group_words(planes) - DIM * (1 + planes) + j * (1 + planes) + f)
                >> l)
                & 1;
            if f == 0 {
                smask |= bit << j;
            } else {
                pl[f - 1] |= bit << j;
            }
        }
    }
    (hdr & 0x1ff, hdr >> 9, smask, pl)
}

/// Every class at both ends of its range, plus a bounded uniform draw and the
/// origin mid-stream — the `planes12x_format` fixture, restated. 2,000 draws:
/// every 5-level block costs a real `nearest_angular` search in
/// `transcode_planes12x`, and this already covers all 383 classes.
fn fixture_indices(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut v: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
    }
    v.push(0);
    while v.len() < 2_000 {
        let idx = rng.next() % llvq_search::index::N13;
        if fd.class_of(idx).is_some() {
            v.push(idx);
        }
    }
    v
}

/// The repack decodes to exactly what the source layout decodes to, on a
/// fixture that touches every class — and the raw words say what the word map
/// says they say.
#[test]
fn the_repack_is_exact_and_the_word_map_is_the_format() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0xE1C_5EED);
    let idxs = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = (0..idxs.len()).map(|i| (i % 2) as u32).collect();

    let p14 = transcode_planes14(&fd, &table, &idxs, &gains).expect("transcodes");
    let e14 = E1c14Blocks::from_planes14(&p14);
    let p12 = transcode_planes12x(&fd, &table, &s, &idxs, &gains).expect("transcodes");
    let e12 = E1c12Blocks::from_planes12x(&p12);

    // Sizes are the formula, not a measurement.
    let groups = idxs.len().div_ceil(E1C_GROUP);
    assert_eq!(e14.data.len(), groups * group_words(E1C14_PLANES) * 4);
    assert_eq!(e12.data.len(), groups * group_words(E1C12_PLANES) * 4);
    // And strictly smaller than the streams they replace.
    assert!(e14.data.len() < p14.data.len().next_multiple_of(PLANES14_BYTES * E1C_GROUP));
    assert!(e12.data.len() < idxs.len() * PLANES12X_BYTES);

    for (b, &idx) in idxs.iter().enumerate() {
        let want = fd.decode(idx).unwrap_or([0i32; DIM]);
        // 1 — the repack is exact against the archive and against Planes14.
        let (pe, ge) = e14.decode_block(&table, b);
        let (p4, g4) = p14.decode_block(&table, b);
        assert_eq!(pe, want, "block {b}: E1c14 against the archive");
        assert_eq!(pe, p4, "block {b}: E1c14 against Planes14");
        assert_eq!((ge, g4), (gains[b], gains[b]), "block {b}: gain");
        // The main stream of the capped variant matches its source's main
        // stream — swapped representatives included, which is the only place
        // the two differ from the archive.
        assert_eq!(
            e12.decode_approx_block(&table, b),
            p12.decode_approx_block(&table, b),
            "block {b}: E1c12 main stream against Planes12x"
        );

        // 2 — the word map, walked by hand off the raw bytes.
        let (id, gain, smask, pl) = fields_by_hand(&e14.data, E1C14_PLANES, b);
        assert_eq!(gain, gains[b], "block {b}: gain bit by hand");
        let rec = table.record(id as usize);
        if id != 0 {
            for (i, &wi) in want.iter().enumerate() {
                let lvl = (0..E1C14_PLANES).fold(0u32, |a, k| a | (((pl[k] >> i) & 1) << k));
                let v = rec.values[lvl as usize];
                let got = if (smask >> i) & 1 == 1 { -v } else { v };
                assert_eq!(got, wi, "block {b}, slot {i}: raw words");
            }
        }
    }
    // 3 — the exception table is carried verbatim: same indices, same bytes.
    assert_eq!(e12.exc_idx, p12.exc_idx, "exception table moved");
    assert_eq!(e12.exc_data, p12.exc_data, "exception records modified");
}

/// The inverse repack restores the source stream byte for byte, on a real
/// transcoded fixture rather than on random bits — so a field the module
/// happens to drop *and* re-derive consistently would still be caught by the
/// hand-walked map above, and a field it drops outright by this.
#[test]
fn the_inverse_repack_restores_the_source_bytes() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x1_1FEED);
    let idxs = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = (0..idxs.len()).map(|i| (i % 2) as u32).collect();
    let p14 = transcode_planes14(&fd, &table, &idxs, &gains).expect("transcodes");
    let back = E1c14Blocks::from_planes14(&p14).to_planes14();
    assert_eq!(back.data, p14.data, "E1c14 round trip is not identical");
    assert_eq!(back.n_blocks, p14.n_blocks);
}

/// **The sealed sweep.** Every block of the published Qwen3-4B through both
/// variants, against the archive decoder — 150,681,600 blocks, the same
/// standard `Planes14`, `Planes12x` and `Golay70` are each held to.
///
/// This is the run item X1/X2 of the spec exists to make possible on a Mac,
/// before a single dollar of card: if the repack is exact here, the bench arm
/// is a pure speed question.
///
/// `#[ignore]`d unconditionally — it needs an archive that is not in the repo
/// and costs minutes even in release. Absent file = loud failure, never a
/// green skip (see `common::sealed_artifact_path`).
#[test]
#[ignore = "reads the sealed 4B archive; minutes even in release"]
fn the_sealed_artifact_e1c_repack_is_exact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    // Chunks are whole groups, so a chunk boundary is never inside one — the
    // one place this layout differs from the flat ones, and the one thing a
    // chunked sweep could get wrong.
    const CHUNK: usize = E1C_GROUP * 4096;
    let verified = AtomicU64::new(0);
    let e14_bits = AtomicU64::new(0);
    let e12_bits = AtomicU64::new(0);
    let p14_bits = AtomicU64::new(0);
    let p12_bits = AtomicU64::new(0);

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(m.shell_cap, 12, "{}: the sealed file is leech1c12", m.name);
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
                    let (idxs, gns) = (&m.indices[lo..hi], &m.gains[lo..hi]);
                    let p14 = transcode_planes14(&fd, &table, idxs, gns).expect("transcodes");
                    let p12 =
                        transcode_planes12x(&fd, &table, &s, idxs, gns).expect("transcodes");
                    let e14 = E1c14Blocks::from_planes14(&p14);
                    let e12 = E1c12Blocks::from_planes12x(&p12);
                    for (b, &idx) in idxs.iter().enumerate() {
                        let want = fd.decode(idx).unwrap_or_else(|| {
                            assert_eq!(idx, 0, "{}: index outside the ball", m.name);
                            [0i32; DIM]
                        });
                        let (pe, ge) = e14.decode_block(&table, b);
                        assert_eq!(pe, want, "{} block {}: E1c14", m.name, lo + b);
                        assert_eq!(ge, gns[b], "{} block {}: gain", m.name, lo + b);
                        assert_eq!(
                            e12.decode_approx_block(&table, b),
                            p12.decode_approx_block(&table, b),
                            "{} block {}: E1c12 main stream",
                            m.name,
                            lo + b
                        );
                    }
                    // Byte counts, so the rate printed at the end is the sum of
                    // what was actually built rather than a formula re-applied.
                    e14_bits.fetch_add(e14.data.len() as u64 * 8, Ordering::Relaxed);
                    e12_bits.fetch_add(
                        e12.data.len() as u64 * 8
                            + e12.exc_idx.len() as u64 * 32
                            + e12.exc_data.len() as u64 * 8,
                        Ordering::Relaxed,
                    );
                    p14_bits.fetch_add(p14.data.len() as u64 * 8, Ordering::Relaxed);
                    p12_bits.fetch_add(
                        p12.data.len() as u64 * 8
                            + p12.exc_idx.len() as u64 * 32
                            + p12.exc_data.len() as u64 * 8,
                        Ordering::Relaxed,
                    );
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                });
            }
        });
    }

    let n = verified.load(Ordering::Relaxed);
    let bpw = |b: &AtomicU64| b.load(Ordering::Relaxed) as f64 / (n * DIM as u64) as f64;
    // The chunk size is a whole number of groups and every matrix's block
    // count divides by it or leaves one partial chunk, so the payload rates
    // land on the formula. Stated as an assertion because a drift here would
    // mean the trailing group was mis-sized.
    assert!(
        (bpw(&e14_bits) - group_words(E1C14_PLANES) as f64 / DIM as f64).abs() < 1e-3,
        "E1c14: {:.4} b/weight",
        bpw(&e14_bits)
    );
    eprintln!(
        "E1c over {n} blocks — payload b/weight:\n  \
         Planes14 {:.4} → E1c14 {:.4}  ({:+.4})\n  \
         Planes12x {:.4} → E1c12 {:.4}  ({:+.4})",
        bpw(&p14_bits),
        bpw(&e14_bits),
        bpw(&e14_bits) - bpw(&p14_bits),
        bpw(&p12_bits),
        bpw(&e12_bits),
        bpw(&e12_bits) - bpw(&p12_bits),
    );
    assert_eq!(n, 150_681_600, "the published 4B has 150,681,600 blocks");
}
