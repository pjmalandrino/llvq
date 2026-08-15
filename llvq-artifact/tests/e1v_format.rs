//! **`E1v`'s contract** — the stream round-trips, and the word map is the format.
//!
//! Two statements, and the second is the one a self-consistent bug can pass:
//!
//! 1. **The stream round-trips.** Pack, read back through
//!    [`E1vBlocks::decode_block`], and the point must equal
//!    `FastDecoder::decode`'s — on a fixture touching **every** class plus the
//!    origin, and on the whole sealed 4B.
//! 2. **The word map is the format.** The header fields are rebuilt from the raw
//!    bytes at the positions the spec fixes — group base word, then `10·l` —
//!    without going through the module's own reader. A pack/unpack pair that
//!    agreed on a *wrong* map would pass (1) and fail this.
//!
//! The sealed sweep is `#[ignore]`d: it reads a 981 MB archive that is not in
//! the repo, and it fails loudly rather than skipping green when the file is
//! absent (see `common::sealed_artifact_path`).

use llvq_artifact::e1v::{read_bits, transcode_e1v, E1V_CLASS_BITS, E1V_GROUP, E1V_ORIGIN_ID};
use llvq_core::{Golay, SplitMix64, DIM};
use llvq_search::cns::{cns_layout, CnsLayout, HEADER_BITS};
use llvq_search::fastdec::FastDecoder;

mod common;

/// The addressed width the addressing test computes for the per-kind
/// accounting, and the tolerance §4 uses on a bits/block mean.
const ADDRESSED_FO: f64 = 53.7370;
const TOL_BITS: f64 = 5e-3;

fn layouts(fd: &FastDecoder, golay: &Golay) -> Vec<CnsLayout> {
    (0..fd.n_classes())
        .map(|ci| cns_layout(fd, golay, ci))
        .collect()
}

/// Every class at both ends and in the middle, the origin, and a bounded draw —
/// padded to whole groups of 32 with the origin, whose record is the header
/// alone and cannot flatter anything.
fn fixture(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut v = vec![0u64];
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
        v.push(first + (last - first) / 2);
        v.push(first + rng.next() % (last - first + 1));
    }
    while !v.len().is_multiple_of(E1V_GROUP) {
        v.push(0);
    }
    v
}

#[test]
fn the_e1v_stream_round_trips_and_the_word_map_is_the_format() {
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays = layouts(&fd, &golay);
    let mut rng = SplitMix64::new(0x0000_E1F0_5EED);
    let idxs = fixture(&fd, &mut rng);
    let gains: Vec<u32> = (0..idxs.len()).map(|i| (i % 2) as u32).collect();

    let s = transcode_e1v(&fd, &golay, &idxs, &gains).expect("transcodes");
    assert_eq!(s.n_blocks, idxs.len());
    assert_eq!(s.bases.len(), idxs.len() / E1V_GROUP);
    // Every group starts on a word boundary, and the bases are strictly
    // increasing: a base table that repeated an offset would overlay two groups
    // and still round-trip whichever was written last.
    assert!(
        s.bases.windows(2).all(|w| w[0] < w[1]),
        "les mots de base doivent croître strictement"
    );

    for (b, &idx) in idxs.iter().enumerate() {
        // 1 — the round trip.
        let (got, gain) = s.decode_block(&fd, &golay, &lays, b);
        let want = fd.decode(idx).expect("dans la boule");
        assert_eq!(got, want, "bloc {b}, index {idx}");
        assert_eq!(u32::from(gain), gains[b], "bloc {b}: gain");

        // 2 — the word map, walked by hand. The header of lane `l` of group `g`
        // sits at `bases[g]·32 + 10·l`, and nothing about that position depends
        // on any other lane.
        let (g, l) = (b / E1V_GROUP, b % E1V_GROUP);
        let hb = u64::from(s.bases[g]) * 32 + HEADER_BITS * l as u64;
        let id = read_bits(&s.data, hb, E1V_CLASS_BITS);
        let want_id = fd
            .class_of(idx)
            .map_or(E1V_ORIGIN_ID, |ci| ci as u64);
        assert_eq!(id, want_id, "bloc {b}: la classe n'est pas où le format la met");
        assert_eq!(
            read_bits(&s.data, hb + u64::from(E1V_CLASS_BITS), 1),
            u64::from(gains[b]),
            "bloc {b}: le gain n'est pas où le format le met"
        );
    }
}

/// **The sealed sweep.** Every block of the published Qwen3-4B, packed and read
/// back — the statement `p5_cns_sweep` cannot make, since it never writes a
/// byte.
#[test]
#[ignore = "reads the sealed 4B archive; minutes even in release"]
fn the_sealed_artifact_e1v_stream_is_exact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays = layouts(&fd, &golay);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    // Whole groups per chunk: a chunk boundary inside a group would give each
    // half its own base word and change the addressing — the one thing a
    // chunked sweep of this layout could get wrong.
    const CHUNK: usize = E1V_GROUP * 4096;
    let verified = AtomicU64::new(0);
    let bits = AtomicU64::new(0);

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
                    let s = transcode_e1v(&fd, &golay, &m.indices[lo..hi], &m.gains[lo..hi])
                        .expect("transcodes");
                    for (b, &idx) in m.indices[lo..hi].iter().enumerate() {
                        let (got, gain) = s.decode_block(&fd, &golay, &lays, b);
                        let want = fd.decode(idx).expect("dans la boule");
                        assert_eq!(got, want, "{} bloc {}", m.name, lo + b);
                        assert_eq!(u32::from(gain), m.gains[lo + b], "{} bloc {}: gain", m.name, lo + b);
                    }
                    bits.fetch_add(
                        s.data.len() as u64 * 8 + s.bases.len() as u64 * 32,
                        Ordering::Relaxed,
                    );
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                });
            }
        });
    }

    let n = verified.load(Ordering::Relaxed);
    let per_block = bits.load(Ordering::Relaxed) as f64 / n as f64;
    assert_eq!(n, 150_681_600, "le 4B publié fait 150 681 600 blocs");
    eprintln!(
        "E1v flux — {n} blocs, {} matrices\n  \
         aller-retour contre FastDecoder::decode : 0 écart\n  \
         largeur adressée mesurée sur les OCTETS ÉCRITS : {per_block:.4} bits/bloc\n  \
         ({:.4} b/poids payload) — à comparer au {ADDRESSED_FO} que le modèle d'adressage\n  \
         calculait sans écrire un octet.",
        h.matrices,
        per_block / DIM as f64
    );
    // 🚨 The bytes must agree with the model that priced them. A packer that
    // wrote a different number of bits than the accounting predicted would make
    // every b/weight in the dossier a statement about nothing.
    assert!(
        (per_block - ADDRESSED_FO).abs() < TOL_BITS,
        "les octets écrits rendent {per_block:.4} bits/bloc, le modèle en prédisait \
         {ADDRESSED_FO} — la comptabilité et l'empaqueteur ne décrivent pas le même flux"
    );
}
