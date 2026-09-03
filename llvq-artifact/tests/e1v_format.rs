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
        "base words must strictly increase"
    );

    for (b, &idx) in idxs.iter().enumerate() {
        // 1 — the round trip.
        let (got, gain) = s.decode_block(&fd, &golay, &lays, b);
        let want = fd.decode(idx).expect("in the ball");
        assert_eq!(got, want, "block {b}, index {idx}");
        assert_eq!(u32::from(gain), gains[b], "block {b}: gain");

        // 2 — the word map, walked by hand. The header of lane `l` of group `g`
        // sits at `bases[g]·32 + 10·l`, and nothing about that position depends
        // on any other lane.
        let (g, l) = (b / E1V_GROUP, b % E1V_GROUP);
        let hb = u64::from(s.bases[g]) * 32 + HEADER_BITS * l as u64;
        let id = read_bits(&s.data, hb, E1V_CLASS_BITS);
        let want_id = fd
            .class_of(idx)
            .map_or(E1V_ORIGIN_ID, |ci| ci as u64);
        assert_eq!(id, want_id, "block {b}: the class is not where the format puts it");
        assert_eq!(
            read_bits(&s.data, hb + u64::from(E1V_CLASS_BITS), 1),
            u64::from(gains[b]),
            "block {b}: the gain is not where the format puts it"
        );
    }
}

/// **The servable cut.** A group never straddles a row, the last group of a row
/// is partial, and everything still round-trips.
///
/// Three statements, and the third is the one that would let a plausible bug
/// through:
///
/// 1. **The round trip closes** on every shape, including the two degenerate
///    ones — a row narrower than a group, and a row that is an exact multiple
///    of one and therefore has no partial group at all.
/// 2. **The word map is the format**, walked by hand from the raw bytes: the
///    header of lane `l` sits at `bases[g]·32 + 10·l`, and `g` is derived from
///    `row_blocks` the way a kernel derives it.
/// 3. **A partial group's payloads start at `10·k`, not at 320.** This is the
///    whole difference between the two cuts, and nothing else notices it: a
///    writer that reserved 32 header slots in a partial group would round-trip
///    perfectly through its own reader while wasting `10·(32−k)` bits per row
///    — turning the +0.48% this cut costs into something several times worse,
///    silently, in the one number E1v is defended on.
#[test]
fn the_row_aligned_cut_never_straddles_a_row() {
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays = layouts(&fd, &golay);
    let mut rng = SplitMix64::new(0x0000_A116_5EED);
    let pool = fixture(&fd, &mut rng);

    // The three shapes of the published 4B, plus a row narrower than a group,
    // plus one that divides exactly — the case with no partial group, where a
    // reader that special-cased the tail would still pass.
    for row_blocks in [106usize, 170, 405, 7, 64] {
        let rows = 5usize;
        let n = rows * row_blocks;
        let idxs: Vec<u64> = (0..n).map(|i| pool[i % pool.len()]).collect();
        let gains: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();

        let s = llvq_artifact::e1v::transcode_e1v_rows(&fd, &golay, &idxs, &gains, row_blocks)
            .expect("transcodes");
        let per_row = row_blocks.div_ceil(E1V_GROUP);
        assert_eq!(s.bases.len(), rows * per_row, "{row_blocks}: groups per row");
        assert!(
            s.bases.windows(2).all(|w| w[0] < w[1]),
            "{row_blocks}: base words must strictly increase"
        );

        for (b, &idx) in idxs.iter().enumerate() {
            // 1 — the round trip.
            let (got, gain) = s.decode_block(&fd, &golay, &lays, b);
            assert_eq!(got, fd.decode(idx).expect("in the ball"), "{row_blocks}: block {b}");
            assert_eq!(u32::from(gain), gains[b], "{row_blocks}: block {b} gain");

            // 2 — the word map, and the group a kernel would compute.
            let (g, l, len) = s.locate(b);
            assert_eq!(g, (b / row_blocks) * per_row + (b % row_blocks) / E1V_GROUP);
            assert_eq!(len, E1V_GROUP.min(row_blocks - (b % row_blocks) / E1V_GROUP * E1V_GROUP));
            let hb = u64::from(s.bases[g]) * 32 + HEADER_BITS * l as u64;
            let want_id = fd.class_of(idx).map_or(E1V_ORIGIN_ID, |ci| ci as u64);
            assert_eq!(
                read_bits(&s.data, hb, E1V_CLASS_BITS),
                want_id,
                "{row_blocks}: block {b}, the class is not where the format puts it"
            );
        }

        // 3 — a partial group reserves `len` headers, not 32. Read straight out
        // of the bytes: the first payload field of the group's lane 0 must sit
        // at `10·len`, so the bits just below it are the last header and not a
        // hole. The origin has no payload, so a group of origins cannot make
        // this statement — the pool is class-bearing by construction.
        let tail_len = row_blocks % E1V_GROUP;
        if tail_len != 0 {
            let g = per_row - 1; // the first row's partial group
            let base = u64::from(s.bases[g]) * 32;
            let b0 = (per_row - 1) * E1V_GROUP;
            let id = read_bits(&s.data, base + HEADER_BITS * (tail_len as u64 - 1), E1V_CLASS_BITS);
            let want = fd
                .class_of(idxs[b0 + tail_len - 1])
                .map_or(E1V_ORIGIN_ID, |ci| ci as u64);
            assert_eq!(
                id, want,
                "{row_blocks}: the last header of a partial group is not at 10·(k−1)"
            );
        }
    }
}

/// The kernel accounting's three shape counts, mirroring `Shapes` in
/// `llvq-bench/src/bin/rtbits.rs` and `Mat` in `planesbench.rs`.
///
/// ⚠️ Its denominator is **every weight of the matrix**, `d_out · d_in`, tail
/// columns included — not `24 · blocks`. Swapping the two is what turns 4.804
/// into 4.827, and it is the mistake the dossier's errata are made of.
#[derive(Default)]
struct Shapes {
    weights: u64,
    tail_weights: u64,
    rows: u64,
}

impl Shapes {
    fn push(&mut self, d_out: usize, d_in: usize) {
        self.weights += (d_out * d_in) as u64;
        self.tail_weights += (d_out * (d_in % DIM)) as u64;
        self.rows += d_out as u64;
    }

    /// Stream plus the `f32` tail block and the `f32` row scales every LLVQ arm
    /// uploads, over every weight.
    fn kernel_bpw(&self, stream_bits: u64) -> f64 {
        (stream_bits + (self.tail_weights + self.rows) * 32) as f64 / self.weights as f64
    }
}

/// **The +0.48% is measured here, on bytes, for the first time.**
///
/// The two cuts of the same stream over the whole sealed 4B, in **one run, one
/// file and one accounting** — so the ratio between them is formed the way the
/// house rules demand, and not from two numbers that never coexisted.
///
/// What this upgrades: the servable cut's cost was a **counting model**
/// (`docs/mesures/x3-alignement-warp-2026-08-15.txt`, then the evening's
/// passation), of the same family as the identity that buried E3 and `e1c14`.
/// Those identities are sound, and they still had never been confronted with a
/// packer. Here they are.
///
/// It also carries the aligned cut's own round trip over all 150,681,600
/// blocks: a cut is not a re-encoding, but a partial group is a code path the
/// file-order sweep never takes.
#[test]
#[ignore = "reads the sealed 4B archive; transcodes it twice, minutes even in release"]
fn the_row_aligned_cut_is_measured_on_the_sealed_artifact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays = layouts(&fd, &golay);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (fo_bits, al_bits) = (AtomicU64::new(0), AtomicU64::new(0));
    let verified = AtomicU64::new(0);
    let mut shapes = Shapes::default();
    let mut shapes_seen: Vec<(usize, usize)> = Vec::new();

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        let row_blocks = m.d_in / DIM;
        assert_eq!(
            m.indices.len(),
            m.d_out * row_blocks,
            "{}: the matrix is not d_out rows of d_in/24 blocks",
            m.name
        );
        shapes.push(m.d_out, m.d_in);
        if !shapes_seen.contains(&(row_blocks, m.d_out)) {
            shapes_seen.push((row_blocks, m.d_out));
        }

        // 🚨 Two cuts, two chunkings, and NEITHER may be approximated. A chunk
        // boundary inside a row would give the aligned cut a group it never
        // has; a boundary inside a group would do the same to the file-order
        // one. Both are therefore cut on their own boundary, and both partition
        // the matrix exactly — no chunk's bits are ever scaled to stand for
        // blocks it did not hold.
        let rows_per_chunk = m.d_out.div_ceil(nthreads).max(1);
        let nchunks = m.d_out.div_ceil(rows_per_chunk);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nthreads {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let lo = c * rows_per_chunk * row_blocks;
                    let hi = (((c + 1) * rows_per_chunk).min(m.d_out)) * row_blocks;
                    let al = llvq_artifact::e1v::transcode_e1v_rows(
                        &fd,
                        &golay,
                        &m.indices[lo..hi],
                        &m.gains[lo..hi],
                        row_blocks,
                    )
                    .expect("transcodes");
                    for (b, &idx) in m.indices[lo..hi].iter().enumerate() {
                        let (got, gain) = al.decode_block(&fd, &golay, &lays, b);
                        assert_eq!(got, fd.decode(idx).expect("in the ball"), "{} block {}", m.name, lo + b);
                        assert_eq!(u32::from(gain), m.gains[lo + b], "{} block {}: gain", m.name, lo + b);
                    }
                    al_bits.fetch_add(
                        al.data.len() as u64 * 8 + al.bases.len() as u64 * 32,
                        Ordering::Relaxed,
                    );
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                });
            }
        });

        // The file-order cut over the SAME blocks, chunked on ITS boundary.
        // Its round trip is the other test's statement; what is needed here is
        // its bits, in the same run, so the ratio is not formed across two.
        assert_eq!(
            m.indices.len() % E1V_GROUP,
            0,
            "{}: {} blocks are not a whole number of groups — the file-order cut \
             would not partition the matrix",
            m.name,
            m.indices.len()
        );
        const FO_CHUNK: usize = E1V_GROUP * 4096;
        let nfo = m.indices.len().div_ceil(FO_CHUNK);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nthreads {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nfo {
                        break;
                    }
                    let lo = c * FO_CHUNK;
                    let hi = ((c + 1) * FO_CHUNK).min(m.indices.len());
                    let fo = transcode_e1v(&fd, &golay, &m.indices[lo..hi], &m.gains[lo..hi])
                        .expect("transcodes");
                    fo_bits.fetch_add(
                        fo.data.len() as u64 * 8 + fo.bases.len() as u64 * 32,
                        Ordering::Relaxed,
                    );
                });
            }
        });
    }

    let n = verified.load(Ordering::Relaxed);
    assert_eq!(n, 150_681_600, "the published 4B has 150,681,600 blocks");
    let (fo_b, al_b) = (fo_bits.load(Ordering::Relaxed), al_bits.load(Ordering::Relaxed));
    let (fo_per, al_per) = (fo_b as f64 / n as f64, al_b as f64 / n as f64);
    let (fo_k, al_k) = (shapes.kernel_bpw(fo_b), shapes.kernel_bpw(al_b));
    let overhead = (al_b as f64 / fo_b as f64 - 1.0) * 100.0;

    eprintln!(
        "E1v — the two cuts of the SAME stream, one run, one accounting\n  \
         {n} blocks, {} matrices, {} distinct shapes (blocks/row × rows): {:?}\n\n  \
         file order         {fo_per:.4} bits/block   {fo_k:.4} b/weight kernel\n  \
         ROW-ALIGNED        {al_per:.4} bits/block   {al_k:.4} b/weight kernel\n  \
         alignment overhead: {overhead:+.3}%\n\n  \
         X3's counting model announced 2.3877 → 2.3983 and +0.48% WITHOUT writing a \
         byte; these\n  two lines are weighed on the bytes written. WARNING: the overhead \
         reads on the bits, not on\n  the quotient of the two b/weight rounded to four \
         decimals, which would give 0.44%.",
        h.matrices,
        shapes_seen.len(),
        shapes_seen
    );

    // The bound the mechanism implies, and it is the one that matters: aligning
    // by PADDING rows out to a multiple of 32 blocks — the only remedy `E1c`
    // had — costs +15.47% on these shapes. A partial group must cost a base
    // word and a rounding, nothing more. Anything above a per cent means the
    // writer is padding somewhere it should be cutting.
    assert!(
        overhead > 0.0 && overhead < 1.0,
        "the measured alignment overhead is {overhead:+.3}%: outside the regime a partial \
         group implies (one base word and one rounding per row). Above that, the writer \
         pads where it should cut — the mechanism that buried e1c14 at +15.47%."
    );
    assert!(
        (fo_per - ADDRESSED_FO).abs() < TOL_BITS,
        "the file-order cut gives {fo_per:.4} bits/block against the published {ADDRESSED_FO}"
    );
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
                        let want = fd.decode(idx).expect("in the ball");
                        assert_eq!(got, want, "{} block {}", m.name, lo + b);
                        assert_eq!(u32::from(gain), m.gains[lo + b], "{} block {}: gain", m.name, lo + b);
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
    assert_eq!(n, 150_681_600, "the published 4B has 150,681,600 blocks");
    eprintln!(
        "E1v stream — {n} blocks, {} matrices\n  \
         round trip against FastDecoder::decode: 0 deviation\n  \
         addressed width measured on the BYTES WRITTEN: {per_block:.4} bits/block\n  \
         ({:.4} b/weight payload) — to compare with the {ADDRESSED_FO} the addressing model\n  \
         computed without writing a byte.",
        h.matrices,
        per_block / DIM as f64
    );
    // 🚨 The bytes must agree with the model that priced them. A packer that
    // wrote a different number of bits than the accounting predicted would make
    // every b/weight in the dossier a statement about nothing.
    assert!(
        (per_block - ADDRESSED_FO).abs() < TOL_BITS,
        "the bytes written give {per_block:.4} bits/block, the model predicted \
         {ADDRESSED_FO} — the accounting and the packer do not describe the same stream"
    );
}
