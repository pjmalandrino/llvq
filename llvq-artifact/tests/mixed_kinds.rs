//! # A code kind per matrix — the file Q5 will write
//!
//! Step C of the Tetra plan (`docs/ROADMAP.md` §2.2 quater, step 3's
//! follow-up). Q5 serves `v_proj` in int4 g128 beside Tetra matrices (§2.3),
//! so *which map* stopped being a fact about a file the moment that was
//! adopted. From v5 every record carries its own [`CodeKind`], the header
//! keeps a default for [`ArtifactWriter::push`], and it gains a [`KindSet`]:
//! the kinds the file declares it may hold.
//!
//! Four claims, each with its own evidence:
//!
//! 1. **A mixed file round-trips.** Ball and Tetra records in one file decode
//!    to their own weights, through the map each record names, and a raw
//!    passthrough reproduces the bytes (`g6_format` holds the byte-identity;
//!    here it is the decode).
//! 2. **The declaration is enforced.** The header precedes the records and
//!    cannot be revised, so a push of an undeclared kind is refused rather
//!    than written under a header that would lie about it.
//! 3. **Both gates are real, and neither is the other.** The header's set
//!    stops a tool before it reads anything; the record's own kind stops it
//!    at the record, which is what a file whose header under-reports its
//!    kinds would otherwise slip past.
//! 4. **Tag 2 is `Int4G128`.** The number reserved for `int4 g128` is now a
//!    kind this build writes; the record itself is `int4_format.rs`.
//!
//! In-memory bytes only: no sealed artifact, no `#[ignore]`, always in the
//! fast loop.

use llvq_artifact::runtime::{require_ball, require_ball_kinds, transcode_planes14_for_kind, ClassTable, Layout};
use llvq_artifact::{
    decode_matrix, read_all, read_header, read_matrix_raw, write_header_kinds, ArtifactWriter,
    CodeKind, Codebooks, Error, KindSet, QuantizedMatrix, DEFAULT_VERSION, FIRST_KINDED_VERSION,
    RESERVED_INT4G128_TAG, TETRA_SHELL_CAP,
};
use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::BlockCode;
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::Indexer;
use llvq_search::tetra::{Tetra, LABEL_MASK};

/// A Ball matrix at cap 12: points drawn by decoding indices, the only source
/// of valid ones.
fn ball_matrix(ix: &Indexer, rng: &mut SplitMix64, name: &str, d_out: usize, d_in: usize) -> QuantizedMatrix {
    let codes: Vec<BlockCode> = (0..d_out * (d_in / DIM))
        .map(|_| loop {
            let i = rng.next() % (1u64 << 47);
            if let Some(point) = ix.decode(i) {
                let shell = llvq_core::Leech::shell_index(&point);
                if shell.is_none_or(|m| m <= 12) {
                    return BlockCode { point, gain: (rng.next() & 1) as u32 };
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
        centroids: vec![0.6, 1.3],
        rotation_seed: Some(0x5EED),
        shell_cap: 12,
        tail: (0..d_out * (d_in % DIM)).map(|_| rng.next_gaussian() as f32 as f64).collect(),
    }
}

/// A Tetra matrix: labels through the map itself, one gain bit, the sentinel
/// cap.
fn tetra_matrix(tetra: &Tetra, rng: &mut SplitMix64, name: &str, d_out: usize, d_in: usize) -> QuantizedMatrix {
    let codes: Vec<BlockCode> = (0..d_out * (d_in / DIM))
        .map(|_| BlockCode {
            point: tetra.decode(rng.next() & LABEL_MASK),
            gain: (rng.next() & 1) as u32,
        })
        .collect();
    QuantizedMatrix {
        name: name.to_string(),
        d_out,
        d_in,
        codes,
        row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
        centroids: vec![0.7, 1.1],
        rotation_seed: Some(0x7210),
        shell_cap: TETRA_SHELL_CAP,
        tail: (0..d_out * (d_in % DIM)).map(|_| rng.next_gaussian() as f32 as f64).collect(),
    }
}

/// The two matrices of every mixed fixture here, and the file that holds
/// them: a Ball `v_proj` and a Tetra `down_proj`, in that order.
fn mixed_file() -> (Vec<u8>, [QuantizedMatrix; 2]) {
    let ix = Indexer::new();
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0xC0DE_11ED);
    let mats = [
        ball_matrix(&ix, &mut rng, "model.layers.0.self_attn.v_proj.weight", 4, 3 * DIM + 8),
        tetra_matrix(&tetra, &mut rng, "model.layers.0.mlp.down_proj.weight", 3, 2 * DIM),
    ];
    let both = KindSet::BALL.with(CodeKind::Tetra);
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::with_kinds(&mut buf, FIRST_KINDED_VERSION, 2, CodeKind::Tetra, both)
            .expect("header");
        w.push_kind(&mats[0], CodeKind::Ball).expect("the Ball record");
        w.push(&mats[1]).expect("the Tetra record, through the default");
        assert_eq!(w.kinds_used(), both);
        w.finish().expect("flush");
    }
    (buf, mats)
}

/// Byte offset of record `i`'s kind tag, walking the two records of
/// [`mixed_file`] by hand — the header, then name length, name, `d_out`,
/// `d_in`, shell cap.
fn kind_tag_at(file: &[u8], names: [&str; 2], i: usize) -> usize {
    const V5_HEADER: usize = 32;
    let mut at = V5_HEADER;
    for (k, name) in names.iter().enumerate() {
        assert_eq!(
            u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize,
            name.len(),
            "record {k} does not start where the walk says"
        );
        if k == i {
            return at + 4 + name.len() + 4 + 4 + 4;
        }
        // name length + name + d_out + d_in + cap + kind + centroid count +
        // seed + rotation flag, then the payload the record declares.
        let mut p = at + 4 + name.len() + 4 + 4 + 4 + 4;
        let n_cent = u32::from_le_bytes(file[p..p + 4].try_into().unwrap()) as usize;
        let d_out = u32::from_le_bytes(file[at + 4 + name.len()..at + 8 + name.len()].try_into().unwrap()) as usize;
        let d_in = u32::from_le_bytes(file[at + 8 + name.len()..at + 12 + name.len()].try_into().unwrap()) as usize;
        p += 4 + 8 + 4 + 8 * n_cent + 8 * d_out + 4 * d_out * (d_in % DIM);
        let n_bytes = u64::from_le_bytes(file[p..p + 8].try_into().unwrap()) as usize;
        at = p + 8 + n_bytes;
    }
    unreachable!("record {i} of two")
}

// ---------------------------------------------------------------------------
// 1 — a mixed file round-trips
// ---------------------------------------------------------------------------

/// Ball and Tetra in one file: each record decodes through the map it names,
/// and the header's default decides nothing about the other one.
#[test]
fn a_mixed_file_round_trips() {
    let (file, mats) = mixed_file();
    let head = read_header(&mut &file[..]).expect("a mixed file opens");
    assert_eq!(head.version, FIRST_KINDED_VERSION);
    assert_eq!(head.default_kind(), CodeKind::Tetra, "the default is one of the two");
    assert_eq!(head.kinds(), KindSet::BALL.with(CodeKind::Tetra));
    assert!(!head.is_ball_only());
    assert_eq!(head.kinds().to_string(), "Ball+Tetra");

    let got = read_all(&mut &file[..]).expect("read");
    assert_eq!(got.len(), 2);
    for (g, m) in got.iter().zip(&mats) {
        assert_eq!(g.name, m.name);
        assert_eq!(g.codes, m.codes, "{}: the points moved", m.name);
        assert_eq!(decode_matrix(g), decode_matrix(m), "{}: the weights moved", m.name);
    }

    // And undecoded: each record says what it is.
    let mut r = std::io::Cursor::new(&file);
    let head = read_header(&mut r).expect("header");
    let kinds: Vec<CodeKind> = (0..head.matrices)
        .map(|_| read_matrix_raw(&mut r, head.version).expect("record").kind)
        .collect();
    assert_eq!(kinds, vec![CodeKind::Ball, CodeKind::Tetra]);
}

/// A mixed file copied record by record keeps each record's kind.
///
/// `g6_format::raw_passthrough_is_byte_identical_at_v5` pins the bytes; this
/// pins what the bytes mean, which is the half a passthrough that took the
/// kind from the writer's default would get wrong — it would rewrite the Ball
/// record as Tetra, or the Tetra one as Ball, and the copy would still be a
/// well-formed file.
#[test]
fn a_mixed_passthrough_keeps_each_record_kind() {
    let (file, mats) = mixed_file();
    let mut r = std::io::Cursor::new(&file);
    let head = read_header(&mut r).expect("header");
    let mut copied: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::with_kinds(
            &mut copied,
            head.version,
            head.matrices,
            head.default_kind(),
            head.kinds(),
        )
        .expect("header");
        for _ in 0..head.matrices {
            w.push_raw(&read_matrix_raw(&mut r, head.version).expect("record")).expect("copy");
        }
        assert_eq!(w.kinds_used(), head.kinds(), "the copy used both kinds");
        w.finish().expect("flush");
    }
    let mut r = std::io::Cursor::new(&copied);
    read_header(&mut r).expect("header");
    let kinds: Vec<CodeKind> = (0..2)
        .map(|_| read_matrix_raw(&mut r, FIRST_KINDED_VERSION).expect("record").kind)
        .collect();
    assert_eq!(kinds, vec![CodeKind::Ball, CodeKind::Tetra], "a record changed map in the copy");
    let got = read_all(&mut &copied[..]).expect("the copy decodes");
    for (g, m) in got.iter().zip(&mats) {
        assert_eq!(decode_matrix(g), decode_matrix(m), "{}: the copy moved the weights", m.name);
    }
}

/// A file of one kind is unchanged by all of this: the same default, a set of
/// one, and `push` writing that kind.
#[test]
fn a_single_kind_file_declares_one_kind() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x_C0DE_0001);
    let m = ball_matrix(&ix, &mut rng, "model.layers.0.mlp.up_proj.weight", 2, 2 * DIM);
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut buf, 1).expect("header");
        assert_eq!((w.default_kind(), w.kinds()), (CodeKind::Ball, KindSet::BALL));
        w.push(&m).expect("write");
        assert_eq!(w.kinds_used(), KindSet::BALL);
        w.finish().expect("flush");
    }
    let head = read_header(&mut &buf[..]).expect("opens");
    // A v4 file: no kind field anywhere, and Ball is what it can only be.
    assert_eq!(head.version, DEFAULT_VERSION);
    assert_eq!(head.kinds(), KindSet::BALL);
    assert!(head.is_ball_only());
    let mut r = std::io::Cursor::new(&buf);
    read_header(&mut r).expect("header");
    let raw = read_matrix_raw(&mut r, head.version).expect("record");
    assert_eq!(raw.kind, CodeKind::Ball, "a legacy record is a Ball record");
}

/// A Ball-only file never builds Tetra's tables, and a Tetra-only file never
/// builds the ball's. The maps are lazy because neither is cheap — 383
/// classes enumerated one way, a 16 KiB table and a trellis re-derived and
/// asserted the other.
#[test]
fn a_single_kind_file_builds_only_its_own_map() {
    let cbs = Codebooks::new();
    assert!(!cbs.is_built(CodeKind::Ball) && !cbs.is_built(CodeKind::Tetra));
    cbs.get(CodeKind::Ball).expect("Ball has a map");
    assert!(cbs.is_built(CodeKind::Ball), "the map asked for is built");
    assert!(!cbs.is_built(CodeKind::Tetra), "the other one is not");

    let cbs = Codebooks::new();
    cbs.get(CodeKind::Tetra).expect("Tetra has a map");
    assert!(cbs.is_built(CodeKind::Tetra));
    assert!(!cbs.is_built(CodeKind::Ball));
}

// ---------------------------------------------------------------------------
// 2 — the declaration is enforced
// ---------------------------------------------------------------------------

/// A push of a kind the header never declared is refused, by name.
///
/// This is the mutant the whole design rests on: the header is written before
/// the first matrix and cannot be revised, so if `push_kind` did not check
/// the declared set, a Tetra record would land in a file whose header says
/// Ball — and every refusal that reads that header would wave it through.
#[test]
fn a_push_of_an_undeclared_kind_is_refused() {
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0x_C0DE_0002);
    let m = tetra_matrix(&tetra, &mut rng, "model.layers.0.mlp.gate_proj.weight", 2, 2 * DIM);

    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArtifactWriter::with_version_kind(&mut buf, FIRST_KINDED_VERSION, 1, CodeKind::Ball)
        .expect("a v5 Ball writer");
    match w.push_kind(&m, CodeKind::Tetra) {
        Err(Error::KindNotDeclared { name, kind, declared }) => {
            assert_eq!(name, m.name);
            assert_eq!(kind, CodeKind::Tetra);
            assert_eq!(declared, KindSet::BALL);
        }
        other => panic!("expected KindNotDeclared, got {:?}", other.err()),
    }
    assert_eq!(w.matrices, 0, "a refused push must not count as a record");
    assert_eq!(w.kinds_used(), KindSet::empty(), "nor mark the kind used");
    // Nothing of the record reached the sink: the file is its 32-byte header
    // and the two zero counts `finish` closes with, and not one byte of the
    // refused record.
    w.finish().expect("flush");
    assert_eq!(buf.len(), 32 + 8, "the refused record left bytes behind");

    // The same push under a writer that declared both kinds is written.
    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArtifactWriter::with_kinds(
        &mut buf,
        FIRST_KINDED_VERSION,
        1,
        CodeKind::Ball,
        KindSet::BALL.with(CodeKind::Tetra),
    )
    .expect("a mixed writer");
    w.push_kind(&m, CodeKind::Tetra).expect("declared, so written");
    assert_eq!(w.kinds_used(), KindSet::of(CodeKind::Tetra));
    w.finish().expect("flush");
    assert_eq!(decode_matrix(&read_all(&mut &buf[..]).expect("read")[0]), decode_matrix(&m));

    // And a raw passthrough goes through the same gate.
    let mut r = std::io::Cursor::new(&buf);
    let head = read_header(&mut r).expect("header");
    let raw = read_matrix_raw(&mut r, head.version).expect("record");
    let mut ball_only = ArtifactWriter::with_version_kind(Vec::new(), FIRST_KINDED_VERSION, 1, CodeKind::Ball)
        .expect("a v5 Ball writer");
    assert!(
        matches!(ball_only.push_raw(&raw), Err(Error::KindNotDeclared { .. })),
        "push_raw must read the record's kind, not the writer's default"
    );
}

/// A header whose default is outside its own declared set is refused at the
/// writer, and a Tetra kind below v5 is refused whichever entry it comes
/// through.
#[test]
fn a_header_that_contradicts_itself_is_refused() {
    assert!(matches!(
        write_header_kinds(&mut Vec::new(), FIRST_KINDED_VERSION, 1, CodeKind::Tetra, KindSet::BALL),
        Err(Error::Inconsistent { .. })
    ));
    assert!(matches!(
        write_header_kinds(&mut Vec::new(), FIRST_KINDED_VERSION, 1, CodeKind::Ball, KindSet::empty()),
        Err(Error::Inconsistent { .. })
    ));
    for version in 1..FIRST_KINDED_VERSION {
        match write_header_kinds(
            &mut Vec::new(),
            version,
            1,
            CodeKind::Ball,
            KindSet::BALL.with(CodeKind::Tetra),
        ) {
            Err(Error::Inconsistent { name, detail }) => {
                assert_eq!(name, "header");
                assert!(detail.contains("Ball+Tetra"), "detail: {detail}");
                assert!(detail.contains(&format!("v{version}")), "detail: {detail}");
            }
            other => panic!("v{version}: expected Inconsistent, got {:?}", other.err()),
        }
    }
    // Ball alone is what every version can say.
    write_header_kinds(&mut Vec::new(), DEFAULT_VERSION, 1, CodeKind::Ball, KindSet::BALL)
        .expect("a Ball-only v4 header");
}

// ---------------------------------------------------------------------------
// 3 — two gates, neither of them the other
// ---------------------------------------------------------------------------

/// The header's set refuses a mixed file before a record is read.
#[test]
fn the_header_set_refuses_a_file_with_one_tetra_matrix() {
    let (file, _) = mixed_file();
    let head = read_header(&mut &file[..]).expect("opens");
    match require_ball_kinds(head.kinds(), "planes14") {
        Err(Error::Inconsistent { name, detail }) => {
            assert_eq!(name, "planes14");
            assert_eq!(detail, "no runtime layout for Tetra before F1d");
        }
        other => panic!("expected the Tetra refusal, got {:?}", other.err()),
    }
    // The default alone would have waved a Ball-defaulted mixed file through:
    // this is why the refusals read the set.
    let both = KindSet::BALL.with(CodeKind::Tetra);
    require_ball(CodeKind::Ball, "planes14").expect("the default of such a file passes");
    assert!(require_ball_kinds(both, "planes14").is_err(), "the set does not");
    // Ball alone passes, at every entry.
    require_ball_kinds(KindSet::BALL, "planes14").expect("a Ball-only file transcodes");
}

/// A file whose header under-reports its kinds — a Ball-only header over a
/// record that says Tetra — is stopped at the record.
///
/// The header's set is a declaration, checked when the file is written; a
/// file that was not written by this crate can still lie about it. The
/// record's own kind is what survives that, and the mutant it kills is a
/// reader that returns `Ball` for every record instead of reading the field:
/// the forged record would then transcode into a coherent stream of some
/// other file's blocks, and the only symptom would be a wrong number.
#[test]
fn a_ball_header_over_a_tetra_record_is_refused_at_the_record() {
    let (mut file, mats) = mixed_file();
    let names = ["model.layers.0.self_attn.v_proj.weight", "model.layers.0.mlp.down_proj.weight"];
    // Forge it: the header declares Ball alone, while record 1 keeps its Tetra
    // tag. No writer of this crate produces such a file.
    file[24..28].copy_from_slice(&CodeKind::Ball.tag().to_le_bytes());
    file[28..32].copy_from_slice(&KindSet::BALL.bits().to_le_bytes());
    let head = read_header(&mut &file[..]).expect("the forged header is self-consistent");
    require_ball_kinds(head.kinds(), "planes14").expect("and passes the header gate");

    let mut r = std::io::Cursor::new(&file);
    read_header(&mut r).expect("header");
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    for (i, want) in [CodeKind::Ball, CodeKind::Tetra].into_iter().enumerate() {
        let m = read_matrix_raw(&mut r, FIRST_KINDED_VERSION).expect("record");
        assert_eq!(m.kind, want, "record {i}: the kind must come from the record");
        let gains: Vec<u32> = m.gains.clone();
        let out = transcode_planes14_for_kind(m.kind, &fd, &table, &m.indices, &gains);
        match want {
            CodeKind::Ball => {
                out.expect("a Ball record transcodes");
            }
            CodeKind::Tetra => match out {
                Err(Error::Inconsistent { detail, .. }) => {
                    assert_eq!(detail, "no runtime layout for Tetra before F1d")
                }
                other => panic!("a forged Tetra record reached the layout: {:?}", other.err()),
            },
            CodeKind::Int4G128 => unreachable!("this file holds no int4 record"),
        }
    }
    // The bytes are otherwise untouched: the Ball record still decodes to its
    // own weights, so what was refused is the record and not the file.
    let got = read_all(&mut &file[..]).expect("the forged file still reads");
    assert_eq!(decode_matrix(&got[0]), decode_matrix(&mats[0]));
    // The record's tag is where the walk says it is, and it still says Tetra:
    // the forgery was in the header and nowhere else.
    let at = kind_tag_at(&file, names, 1);
    assert_eq!(u32::from_le_bytes(file[at..at + 4].try_into().unwrap()), CodeKind::Tetra.tag());
    assert_eq!(kind_tag_at(&file, names, 0), 32 + 4 + names[0].len() + 4 + 4 + 4);
    // Every layout, not only Planes14 — one call each through the generic
    // entry, which is what `fusedrun` and the benches reach.
    for layout in [Layout::Fixed96, Layout::Grouped32, Layout::Flat32, Layout::Sorted32, Layout::Slot32] {
        assert!(
            llvq_artifact::runtime::transcode_for_kind(CodeKind::Tetra, &fd, &table, &[0], &[0], layout).is_err(),
            "{layout:?} accepted a Tetra record"
        );
    }
}

// ---------------------------------------------------------------------------
// 4 — the third kind
// ---------------------------------------------------------------------------

/// Tag 2 is `Int4G128`, and the promise the number carried is kept.
///
/// Reserving it cost nothing and bought the one thing a format cannot
/// retrofit: that a file written while the tag was reserved and one written
/// now cannot disagree about what a 2 meant. The record itself is
/// `int4_format.rs`; what is checked here is that the number, the kind and
/// the header's set moved together.
#[test]
fn the_int4_tag_is_the_int4_kind() {
    assert_eq!(RESERVED_INT4G128_TAG, 2);
    assert_eq!(CodeKind::Int4G128.tag(), RESERVED_INT4G128_TAG);
    assert_eq!(CodeKind::from_tag(RESERVED_INT4G128_TAG).expect("tag 2"), CodeKind::Int4G128);
    assert_eq!(CodeKind::ALL.len(), 3);
    // The bit opens in the set by the same arithmetic, `1 << tag`.
    let set = KindSet::from_bits(1 << RESERVED_INT4G128_TAG).expect("bit 2");
    assert!(set.contains(CodeKind::Int4G128));
    assert_eq!(set.to_string(), "Int4G128");
    assert_eq!(
        KindSet::from_bits(0b110).expect("Tetra+Int4G128").to_string(),
        "Tetra+Int4G128"
    );
    // Tag 3 is the lowest one this build cannot name, and the message says 3.
    match CodeKind::from_tag(3) {
        Err(Error::UnknownCodeKind { tag }) => assert_eq!(tag, 3),
        other => panic!("expected UnknownCodeKind, got {:?}", other.map(|k| k.name())),
    }
    match KindSet::from_bits(1 << 3) {
        Err(Error::UnknownCodeKind { tag }) => assert_eq!(tag, 3),
        other => panic!("expected UnknownCodeKind, got {:?}", other.map(|s| s.to_string())),
    }
    assert!(CodeKind::from_tag(3).unwrap_err().to_string().contains("Int4G128"));
    // And the known bits still parse, alone and together.
    assert_eq!(KindSet::from_bits(0b01).expect("Ball"), KindSet::BALL);
    assert_eq!(
        KindSet::from_bits(0b11).expect("both"),
        KindSet::BALL.with(CodeKind::Tetra)
    );
    assert_eq!(KindSet::from_bits(0).expect("empty"), KindSet::empty());
    assert_eq!(KindSet::empty().to_string(), "none");
}
