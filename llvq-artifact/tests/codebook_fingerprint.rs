//! # The codebook fingerprint — the field that says what an index *means*
//!
//! A matrix record has always carried the shell cap and the centroid count, so
//! the *width* of an index and of a gain are in the file. Nothing in the file
//! said which map turns an index into a lattice point. Change the Golay
//! codeword order, the class enumeration order or a mixed-radix composition
//! between the build that writes and the build that reads, and every index is
//! still in range, every decoded point is still a lattice point, the file
//! opens, the model runs — and the weights are wrong. §4 of the project notes
//! calls this the v1 stability contract; until now the contract was prose.
//!
//! These tests hold three separate claims, and each needs its own kind of
//! evidence:
//!
//! 1. **The map has not moved.** `the_index_map_is_the_published_one` pins the
//!    fingerprint to a constant. This is the only protection an *already
//!    published* artifact can have — it was written before the field existed,
//!    so nothing in its bytes can defend it. A pinned constant can.
//! 2. **The fingerprint is a function of the map.** Pinning a number proves
//!    nothing if the number would not move;
//!    `the_fingerprint_moves_with_every_ingredient_of_the_map` restates the
//!    digest from the same public inputs and perturbs each of them in turn.
//! 3. **A disagreeing file is refused.** `a_file_from_another_codebook_is_refused`
//!    and `a_hand_rolled_v4_header_is_refused`.
//!
//! Claim 3 is tested by writing the stored fingerprint of another codebook
//! rather than by rebuilding the codebook itself, which is not something a
//! test can do — `GEN_POLY` and the enumeration orders are compile-time facts
//! of `llvq-core`/`llvq-search`. The substitution is exact because the field
//! *is* a pure function of those facts, and claim 2 is what establishes that
//! a different codebook gives a different value.

mod common;

use llvq_artifact::{
    codebook_fingerprint, decode_matrix, read_all, read_header, ArtifactWriter, Error,
    QuantizedMatrix, MAGIC, MAGIC_V1, MAGIC_V2, MAGIC_V3, VERSION,
};
use llvq_core::{Point, SplitMix64, DIM};
use llvq_quant::quantizer::BlockCode;
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};
use llvq_search::index::{Indexer, N13};

/// The fingerprint of format v1 as this branch defines it.
///
/// If this constant fails, **do not update it** without deciding, explicitly,
/// which of two things happened:
///
/// * the index map changed — in which case every `.llvq` ever written,
///   including the Qwen3-4B artifact published on Hugging Face and cited by
///   the paper, has just become undecodable by this build, and the change has
///   to be reverted or the files reissued;
/// * only what is *hashed* changed (a probe position, the domain tag) — a
///   deliberate edit to `src/codebook.rs`, and then the new value is correct.
///
/// The first case is the whole reason the constant is here. The failure
/// message alone does not distinguish them; a human has to.
const PUBLISHED_FINGERPRINT: u64 = 0x338f_420f_1186_6319;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A small matrix whose codes are real lattice points, drawn by decoding
/// random indices — the codebook is the only source of valid points.
fn synthetic(ix: &Indexer, rng: &mut SplitMix64, name: &str) -> QuantizedMatrix {
    let (d_out, d_in, shell_cap, n_gain) = (3usize, 2 * DIM + 8, 12u32, 2usize);
    let nblocks = d_in / DIM;
    let hi = llvq_quant::quantizer::index_bits(shell_cap);
    let codes: Vec<BlockCode> = (0..d_out * nblocks)
        .map(|_| loop {
            let i = rng.next() % (1u64 << hi);
            if let Some(point) = ix.decode(i) {
                match llvq_core::Leech::shell_index(&point) {
                    Some(m) if m <= u64::from(shell_cap) => {
                        return BlockCode {
                            point,
                            gain: (rng.next() % n_gain as u64) as u32,
                        }
                    }
                    _ if point == [0; DIM] => return BlockCode { point, gain: 0 },
                    _ => {}
                }
            }
        })
        .collect();
    QuantizedMatrix {
        name: name.to_string(),
        d_out,
        d_in,
        codes,
        row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
        centroids: (0..n_gain).map(|k| 0.5 + 0.4 * k as f64).collect(),
        rotation_seed: Some(0x1234),
        shell_cap,
        tail: (0..d_out * (d_in % DIM))
            .map(|_| f64::from(rng.next_gaussian() as f32))
            .collect(),
    }
}

/// A one-matrix `LVQ4` file, and the weights it should decode to.
fn v4_file() -> (Vec<u8>, Vec<f32>) {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0xC0DE_B00C);
    let m = synthetic(&ix, &mut rng, "model.layers.0.mlp.up_proj.weight");
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut buf, 1).expect("header");
        w.push(&m).expect("write");
        w.finish().expect("flush");
    }
    (buf, decode_matrix(&m))
}

// ---------------------------------------------------------------------------
// 1 — the map has not moved
// ---------------------------------------------------------------------------

#[test]
fn the_index_map_is_the_published_one() {
    assert_eq!(
        codebook_fingerprint(),
        PUBLISHED_FINGERPRINT,
        "the v1 index map changed: files written by any earlier build — the \
         published Qwen3-4B artifact among them — no longer decode to what \
         they meant. Read the note on PUBLISHED_FINGERPRINT before touching \
         this constant."
    );
}

// ---------------------------------------------------------------------------
// 2 — the fingerprint is a function of the map
// ---------------------------------------------------------------------------

/// Everything `codebook_fingerprint` digests, laid out so a test can perturb
/// one piece at a time. Restated from the module's spec rather than shared
/// with it: a yardstick that imports the thing it measures measures nothing.
#[derive(Clone)]
struct Ingredients {
    domain: Vec<u8>,
    dim: u64,
    n13: u64,
    gen_poly: u64,
    words: Vec<u32>,
    classes: Vec<ClassIngredient>,
    edges: Vec<(u64, Option<Point>)>,
}

#[derive(Clone)]
struct ClassIngredient {
    lo: u64,
    hi: u64,
    len: u64,
    nonzero: u64,
    odd: u64,
    shell: u64,
    values: [i32; MAX_LEVELS],
    counts: [u8; MAX_LEVELS],
    probes: Vec<(u64, Option<Point>)>,
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Collect the ingredients as the shipped module would see them.
fn truth() -> Ingredients {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let classes = (0..fd.n_classes())
        .map(|ci| {
            let (lo, hi) = fd.class_range(ci);
            let lv = fd.levels(ci);
            let card = hi - lo + 1;
            let mut probes: Vec<(u64, Option<Point>)> = [(0u64, 4u64), (1, 4), (2, 4), (3, 4)]
                .iter()
                .map(|&(num, den)| lo + card * num / den)
                .chain(std::iter::once(hi))
                .chain((0..2).map(|k| lo + mix(ci as u64 ^ mix(k)) % card))
                .map(|idx| (idx, ix.decode(idx)))
                .collect();
            probes.shrink_to_fit();
            ClassIngredient {
                lo,
                hi,
                len: lv.len as u64,
                nonzero: u64::from(lv.nonzero),
                odd: u64::from(lv.odd),
                shell: u64::from(lv.shell),
                values: lv.values,
                counts: lv.counts,
                probes,
            }
        })
        .collect();
    Ingredients {
        domain: b"llvq codebook fingerprint v1".to_vec(),
        dim: DIM as u64,
        n13: N13,
        gen_poly: u64::from(llvq_core::golay::GEN_POLY),
        words: llvq_core::Golay::new().codewords().to_vec(),
        classes,
        edges: [0, N13 + 1].iter().map(|&i| (i, ix.decode(i))).collect(),
    }
}

fn byte(h: &mut u64, b: u8) {
    *h ^= u64::from(b);
    *h = h.wrapping_mul(0x0000_0100_0000_01b3);
}
fn bytes(h: &mut u64, bs: &[u8]) {
    bs.iter().for_each(|&b| byte(h, b));
}
fn word(h: &mut u64, v: u64) {
    bytes(h, &v.to_le_bytes());
}
fn point(h: &mut u64, idx: u64, p: &Option<Point>) {
    word(h, idx);
    match p {
        Some(p) => {
            byte(h, 1);
            p.iter().for_each(|v| bytes(h, &v.to_le_bytes()));
        }
        None => byte(h, 0),
    }
}

/// FNV-1a over the ingredients, in the order the module documents.
fn digest(g: &Ingredients) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    bytes(&mut h, &g.domain);
    word(&mut h, g.dim);
    word(&mut h, g.n13);
    word(&mut h, g.gen_poly);
    word(&mut h, g.words.len() as u64);
    for &w in &g.words {
        word(&mut h, u64::from(w));
    }
    word(&mut h, g.classes.len() as u64);
    for c in &g.classes {
        word(&mut h, c.lo);
        word(&mut h, c.hi);
        word(&mut h, c.len);
        word(&mut h, c.nonzero);
        word(&mut h, c.odd);
        word(&mut h, c.shell);
        for k in 0..MAX_LEVELS {
            bytes(&mut h, &c.values[k].to_le_bytes());
            byte(&mut h, c.counts[k]);
        }
        for (idx, p) in &c.probes {
            point(&mut h, *idx, p);
        }
    }
    for (idx, p) in &g.edges {
        point(&mut h, *idx, p);
    }
    h
}

/// The restatement must agree with the shipped digest, and then every
/// ingredient of it must be load-bearing.
///
/// This is the half of the argument the stored field rests on. Without it,
/// `a_file_from_another_codebook_is_refused` would only prove that the reader
/// compares two `u64`s — not that the `u64` says anything about the codebook.
/// The perturbations below are exactly the ways §4 says format v1 can move:
/// the Golay generator and codeword order, the class enumeration order, a
/// class's magnitude structure, and the composition that turns an index into
/// coordinates.
#[test]
#[cfg_attr(debug_assertions, ignore = "383 classes × 7 decodes, twice — release only")]
fn the_fingerprint_moves_with_every_ingredient_of_the_map() {
    let base = truth();
    assert_eq!(
        digest(&base),
        codebook_fingerprint(),
        "the independent restatement disagrees with src/codebook.rs — one of \
         the two changed without the other"
    );

    let mut perturbations: Vec<(&str, Ingredients)> = Vec::new();

    let mut g = base.clone();
    g.domain.push(b'!');
    perturbations.push(("domain tag", g));

    let mut g = base.clone();
    g.dim += 1;
    perturbations.push(("block dimension", g));

    let mut g = base.clone();
    g.n13 += 1;
    perturbations.push(("codebook cardinality", g));

    let mut g = base.clone();
    g.gen_poly = u64::from(llvq_core::golay::GEN_POLY_RECIPROCAL);
    perturbations.push(("Golay generator", g));

    // Two codewords transposed — a reordering that leaves the *code* intact
    // and every index in range. Nothing else in a file could see it.
    let mut g = base.clone();
    g.words.swap(1, 2);
    perturbations.push(("Golay codeword order", g));

    // Two classes transposed: the class enumeration order of §4.
    let mut g = base.clone();
    g.classes.swap(3, 4);
    perturbations.push(("class enumeration order", g));

    // A shell boundary moved by one index.
    let mut g = base.clone();
    g.classes[7].lo += 1;
    perturbations.push(("class offset", g));

    // A class's magnitude structure — what ClassTable and every runtime
    // layout key on.
    let mut g = base.clone();
    g.classes[11].values[0] += 2;
    perturbations.push(("class level value", g));

    let mut g = base.clone();
    g.classes[11].counts[0] += 1;
    perturbations.push(("class level count", g));

    let mut g = base.clone();
    g.classes[13].odd ^= 1;
    perturbations.push(("class coset", g));

    // The composition itself: same index, different coordinates. This is the
    // mixed-radix change no table of parameters would catch.
    let mut g = base.clone();
    let p = g.classes[17].probes[3]
        .1
        .as_mut()
        .expect("a class probe decodes");
    p.swap(0, 1);
    perturbations.push(("coordinate arrangement", g));

    // A sign flipped on one probed point — the sign convention, which carries
    // no bits in the odd coset and is therefore pure convention. A zero
    // coordinate would make the flip a no-op and the perturbation a lie, so
    // the first nonzero one is picked rather than a fixed slot.
    let mut g = base.clone();
    let p = g.classes[19].probes[0].1.as_mut().expect("decodes");
    let nz = p.iter().position(|&v| v != 0).expect("a nonzero coordinate");
    p[nz] *= -1;
    perturbations.push(("sign convention", g));

    // The edge behaviour: the origin, and the first index past the ball.
    let mut g = base.clone();
    g.edges[1].1 = Some([1; DIM]);
    perturbations.push(("the index past the ball now decodes", g));

    let want = digest(&base);
    for (what, g) in perturbations {
        assert_ne!(
            digest(&g),
            want,
            "{what} can change without moving the fingerprint — the field \
             does not cover it, and a file written before the change would be \
             accepted after it"
        );
    }
}

// ---------------------------------------------------------------------------
// 3 — a disagreeing file is refused
// ---------------------------------------------------------------------------

/// **The test the item exists for.** A file whose header names a codebook
/// other than this build's must be refused, at the header, before a single
/// index is decoded.
#[test]
fn a_file_from_another_codebook_is_refused() {
    let (good, want) = v4_file();

    // The control: unpatched, it reads, and the weights come back.
    let head = read_header(&mut std::io::Cursor::new(&good)).expect("a fresh file must open");
    assert_eq!(head.version, VERSION);
    assert_eq!(head.codebook, Some(codebook_fingerprint()));
    assert_eq!(
        decode_matrix(&read_all(&mut std::io::Cursor::new(&good)).expect("read")[0]),
        want
    );

    // The header is magic(4) + count(4) + fingerprint(8); patch the field to
    // what another codebook's map would have produced.
    let stored = u64::from_le_bytes(good[8..16].try_into().unwrap());
    assert_eq!(stored, codebook_fingerprint(), "the field is where we think");

    for other in [stored ^ 1, stored.wrapping_add(0xDEAD_BEEF), 0, u64::MAX] {
        if other == stored {
            continue;
        }
        let mut bad = good.clone();
        bad[8..16].copy_from_slice(&other.to_le_bytes());

        match read_header(&mut std::io::Cursor::new(&bad)) {
            Err(Error::CodebookMismatch { stored: s, computed }) => {
                assert_eq!(s, other);
                assert_eq!(computed, codebook_fingerprint());
            }
            Err(e) => panic!("wrong refusal for fingerprint {other:#x}: {e}"),
            Ok(_) => panic!(
                "a file written against codebook {other:#x} was accepted by a \
                 build whose codebook is {:#x} — its indices decode to valid \
                 lattice points that are not the ones written, and nothing \
                 downstream can tell",
                codebook_fingerprint()
            ),
        }

        // And the whole-file path, not just the header, since that is what a
        // caller actually invokes.
        assert!(
            read_all(&mut std::io::Cursor::new(&bad)).is_err(),
            "read_all must refuse it too"
        );
    }
}

/// A `LVQ4` magic over a header written as `magic + count` — what hand-rolling
/// the two writes produces — must be refused rather than read as if the first
/// eight bytes of the matrix record were a fingerprint.
///
/// `llvq-bench/src/bin/lswap.rs` rebuilds a header exactly that way. This
/// pins the failure as *loud*: a stale writer cannot produce a file that
/// silently reads as something else.
#[test]
fn a_hand_rolled_v4_header_is_refused() {
    let (good, _) = v4_file();
    let mut bad: Vec<u8> = Vec::new();
    bad.extend_from_slice(MAGIC);
    bad.extend_from_slice(&good[4..8]); // the count
    bad.extend_from_slice(&good[16..]); // straight to the matrix record
    assert!(
        matches!(
            read_header(&mut std::io::Cursor::new(&bad)),
            Err(Error::CodebookMismatch { .. })
        ),
        "a v4 header missing its fingerprint must be refused, not read with \
         the matrix record's first eight bytes standing in for the field"
    );
}

// ---------------------------------------------------------------------------
// 4 — and the old files still open
// ---------------------------------------------------------------------------

/// Every legacy version must still read, with no fingerprint demanded.
///
/// Not negotiable: the Qwen3-4B artifact published on Hugging Face and cited
/// in the paper's availability section is `LVQ2`, and it cannot grow a field
/// retroactively. The matrix records are identical across versions, so a
/// legacy file is a `LVQ4` file minus eight header bytes — which is exactly
/// how the fixtures below are built, and exactly why the version bump is
/// cheap.
#[test]
fn legacy_headers_still_read() {
    let (v4, want) = v4_file();

    for (version, magic) in [(1u32, MAGIC_V1), (2, MAGIC_V2), (3, MAGIC_V3)] {
        let mut old: Vec<u8> = Vec::new();
        old.extend_from_slice(magic);
        old.extend_from_slice(&v4[4..8]);
        old.extend_from_slice(&v4[16..]);

        let head = read_header(&mut std::io::Cursor::new(&old))
            .unwrap_or_else(|e| panic!("a v{version} file must still open: {e}"));
        assert_eq!(head.version, version);
        assert_eq!(head.matrices, 1);
        assert_eq!(
            head.codebook, None,
            "a v{version} file has no fingerprint to report"
        );
        assert_eq!(head.is_self_contained(), version >= 2);

        let got = read_all(&mut std::io::Cursor::new(&old)).expect("legacy matrices");
        assert_eq!(got.len(), 1);
        assert_eq!(
            decode_matrix(&got[0]),
            want,
            "a v{version} file must decode to the same weights as the v4 one"
        );
    }
}

/// `write_header` must round-trip at every version, and refuse the ones that
/// do not exist.
///
/// This is the entry point a tool rewriting a file's matrix section is meant
/// to call instead of hand-rolling `magic + count`
/// (`llvq-bench/src/bin/lswap.rs` does exactly that). The geometry it hides is
/// no longer constant across versions, which is the whole reason it exists.
#[test]
fn write_header_round_trips_at_every_version() {
    for version in 1..=VERSION {
        let mut buf: Vec<u8> = Vec::new();
        llvq_artifact::write_header(&mut buf, version, 42).expect("write");
        assert_eq!(
            buf.len(),
            if version >= llvq_artifact::FIRST_FINGERPRINTED_VERSION {
                16
            } else {
                8
            },
            "v{version} header geometry"
        );
        let head = read_header(&mut std::io::Cursor::new(&buf)).expect("read back");
        assert_eq!(head.version, version);
        assert_eq!(head.matrices, 42);
        assert_eq!(
            head.codebook,
            (version >= llvq_artifact::FIRST_FINGERPRINTED_VERSION)
                .then(codebook_fingerprint)
        );
    }
    for bad in [0, VERSION + 1] {
        assert!(
            matches!(
                llvq_artifact::write_header(&mut Vec::new(), bad, 1),
                Err(Error::UnknownVersion { .. })
            ),
            "version {bad} must be refused rather than written with some \
             neighbouring magic"
        );
    }
}

/// The real published archive, header only — the claim that matters stated
/// against the bytes that are actually online rather than a fixture.
///
/// `#[ignore]`d because it needs the file; see `tests/common/mod.rs` for why
/// that is a declaration rather than a silent pass.
#[test]
#[ignore = "needs the sealed 4B archive on this machine"]
fn the_published_archive_still_opens() {
    let path = common::sealed_artifact_path();
    let mut r = std::io::BufReader::new(std::fs::File::open(&path).expect("open the archive"));
    let head = read_header(&mut r).unwrap_or_else(|e| {
        panic!(
            "the published archive no longer opens: {e}\n\nThis is the file \
             the paper points at. A reader that refuses it is a worse defect \
             than the one the fingerprint closes."
        )
    });
    assert!(
        (1..=3).contains(&head.version),
        "the published archive is a legacy version, not v{}",
        head.version
    );
    assert_eq!(
        head.codebook, None,
        "a pre-v4 archive cannot carry a fingerprint"
    );
    assert!(head.matrices > 0, "an archive with no matrices");

    // And its first matrix still decodes — the header check is not the point
    // if the body no longer reads.
    let ix = Indexer::new();
    let m = llvq_artifact::read_matrix(&mut r, &ix).expect("the first matrix must decode");
    assert_eq!(m.codes.len(), m.d_out * (m.d_in / DIM));
}
