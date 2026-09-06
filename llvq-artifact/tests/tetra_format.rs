//! # Format v5 — a Tetra file, from the word on disk to the refusals
//!
//! Step 3 of the Tetra plan (`docs/ROADMAP.md` §2.2 quater): the file says
//! which map its indices belong to, and everything that reads a record or
//! builds a VRAM layout consults that before any width. Three claims, each
//! with its own evidence:
//!
//! 1. **The word on disk is the word.** `BitWriter` → `BitReader` →
//!    `Tetra::decode(idx | gain << 47)` returns the intended point on 10⁵
//!    words; a whole Tetra matrix survives `write_matrix_with` /
//!    `read_matrix_with` and `read_all`; the gain bit sits at bit 47 and
//!    nowhere else.
//! 2. **The kind is load-bearing.** The same record read as Ball is a
//!    different matrix or an error; a Tetra writer below v5 is refused; a Tetra
//!    matrix with the wrong sentinel cap or the wrong gain width is refused
//!    before a byte is written.
//! 3. **Nothing downstream pretends.** Every runtime transcoder and table
//!    builder has a `*_for_kind` twin that returns `Error::Inconsistent` with
//!    "no runtime layout for Tetra before F1d" on a Tetra header, and the same
//!    stream as its original on a Ball one.
//!
//! The yardstick for the reconstruction — `decode_matrix` against the
//! quantizer's own `reconstruct` — closes the factoring of
//! `reconstruct_shape_gain` out of `LeechShapeGain`.

use llvq_artifact::blockrec::{block_records, block_records_for_kind};
use llvq_artifact::e1v::{transcode_e1v, transcode_e1v_for_kind, transcode_e1v_rows, transcode_e1v_rows_for_kind};
use llvq_artifact::runtime::{
    require_ball, transcode, transcode_for_kind, transcode_golay70, transcode_golay70_for_kind,
    transcode_planes12x, transcode_planes12x_for_kind, transcode_planes14, transcode_planes14_for_kind,
    ClassTable, Golay70Table, Layout,
};
use llvq_artifact::{
    decode_matrix, read_all, read_header, read_matrix, read_matrix_raw, read_matrix_with,
    write_matrix_raw, write_matrix_with, ArtifactWriter, CodeKind, Codebook, Codebooks, Error,
    KindSet, QuantizedMatrix, RawMatrix, DEFAULT_VERSION, FIRST_KINDED_VERSION, MAGIC_V5,
    TETRA_SHELL_CAP,
};
use llvq_core::{Golay, SplitMix64, DIM};
use llvq_quant::quantizer::{BlockCode, LeechShapeGain};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};
use llvq_search::pack::{BitReader, BitWriter};
use llvq_search::tetra::{Tetra, LABEL_BITS, LABEL_MASK};
use llvq_search::Searcher;

const REFUSAL: &str = "no runtime layout for Tetra before F1d";

/// A Tetra matrix: codes drawn as random labels through the map itself — the
/// map is the only source of valid points — with one gain bit and the
/// sentinel cap.
fn synthetic_tetra(tetra: &Tetra, rng: &mut SplitMix64, name: &str, d_out: usize, d_in: usize) -> QuantizedMatrix {
    let nblocks = d_in / DIM;
    let codes: Vec<BlockCode> = (0..d_out * nblocks)
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

/// A one-matrix v5 Tetra file, and the matrix it holds.
fn tetra_file(tetra: &Tetra, seed: u64) -> (Vec<u8>, QuantizedMatrix) {
    let mut rng = SplitMix64::new(seed);
    let m = synthetic_tetra(tetra, &mut rng, "model.layers.0.mlp.up_proj.weight", 4, 3 * DIM + 8);
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::with_kind(&mut buf, CodeKind::Tetra, 1).expect("header");
        w.push(&m).expect("write");
        w.finish().expect("flush");
    }
    (buf, m)
}

// ---------------------------------------------------------------------------
// 1 — the word on disk
// ---------------------------------------------------------------------------

/// `BitWriter::push(label, 47); push(gain, 1)`, MSB-first, read back and
/// reassembled as `idx | gain << 47`: the intended point, 10⁵ times, gain
/// bit included. The disk is big-endian across the six bytes; the card's
/// little-endian word is F1d's transcoder, not this crate's, and this test
/// pins only what this crate writes.
#[test]
fn the_disk_word_decodes_to_the_intended_point() {
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0x7210_0003_0001);
    let n = 100_000usize;
    let words: Vec<u64> = (0..n).map(|_| rng.next() & ((1u64 << 48) - 1)).collect();
    let mut bw = BitWriter::with_capacity(n as u64 * 48);
    for &w in &words {
        bw.push(w & LABEL_MASK, LABEL_BITS);
        bw.push(w >> LABEL_BITS, 1);
    }
    let bytes = bw.finish();
    assert_eq!(bytes.len(), n * 6, "48 bits a block, no padding but the last byte's");
    let mut br = BitReader::new(&bytes);
    for (b, &w) in words.iter().enumerate() {
        let idx = br.read(LABEL_BITS);
        let gain = br.read(1) as u32;
        assert_eq!(idx, w & LABEL_MASK, "block {b}: label");
        assert_eq!(u64::from(gain), w >> LABEL_BITS, "block {b}: gain bit");
        let word = idx | u64::from(gain) << LABEL_BITS;
        assert_eq!(word, w, "block {b}: the word is not reassembled");
        assert_eq!(tetra.decode(word), tetra.decode(w), "block {b}: the point moved");
    }
}

/// The gain bit sits at bit 47 of the word and nowhere else: put at bit 0 it
/// lands on `p`, and every gain-1 block whose `p` is 0 decodes to a point of
/// the other parity. This is the mutant `Codebook::decode` is pinned
/// against — through `cb.decode` itself, on every label, not only by count.
#[test]
fn the_gain_bit_sits_at_bit_47_on_disk() {
    let tetra = Tetra::new();
    let cb = Codebook::new(CodeKind::Tetra);
    let mut rng = SplitMix64::new(0x7210_0003_0002);
    let mut flipped = 0usize;
    for _ in 0..2_000 {
        let label = rng.next() & LABEL_MASK;
        for gain in 0..2u32 {
            let want = tetra.decode(label | u64::from(gain) << LABEL_BITS);
            assert_eq!(cb.decode(label, gain), Some(want));
            assert_eq!(want, tetra.decode(label), "bit 47 is opaque to the decoder");
            if gain == 1 && label & 1 == 0 {
                let wrong = tetra.decode(label | 1);
                assert_ne!(wrong, want, "p flipped and the point did not move");
                assert_ne!(cb.decode(label, gain), Some(wrong), "the gain bit landed on p");
                flipped += 1;
            }
        }
    }
    assert!(flipped > 800, "half the labels have p = 0; {flipped} were checked");
}

/// A whole Tetra matrix through `write_matrix_with` and back through
/// `read_matrix_with`: codes, gains, scales, tail and the decoded weights.
#[test]
fn a_tetra_matrix_survives_a_round_trip() {
    let tetra = Tetra::new();
    let cb = Codebook::new(CodeKind::Tetra);
    let cbs = Codebooks::new();
    let mut rng = SplitMix64::new(0x7210_0003_0003);
    for (d_out, d_in) in [(4usize, 3 * DIM), (3, 100), (5, 5 * DIM)] {
        let m = synthetic_tetra(&tetra, &mut rng, "model.layers.0.self_attn.q_proj.weight", d_out, d_in);
        let mut bytes: Vec<u8> = Vec::new();
        let bits = write_matrix_with(&mut bytes, FIRST_KINDED_VERSION, &cb, &m).expect("write");
        assert_eq!(bits, m.bits(), "48 bits a block is what the accounting says");
        let got = read_matrix_with(&mut &bytes[..], FIRST_KINDED_VERSION, &cbs).expect("read");
        assert_eq!(got.codes, m.codes, "codes differ for {d_out}×{d_in}");
        assert_eq!(got.shell_cap, TETRA_SHELL_CAP);
        assert_eq!(got.row_scales, m.row_scales);
        assert_eq!(got.tail, m.tail);
        assert_eq!(decode_matrix(&got), decode_matrix(&m));

        // The raw view holds the labels and the gain bits, nothing decoded.
        let raw = read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION).expect("raw");
        assert_eq!(raw.kind, CodeKind::Tetra, "the record says what it is");
        for (b, (code, (&idx, &gain))) in m.codes.iter().zip(raw.indices.iter().zip(&raw.gains)).enumerate() {
            assert_eq!(idx, tetra.encode(&code.point).expect("a Tetra point"), "block {b}: label");
            assert_eq!(gain, code.gain, "block {b}: gain");
            assert!(idx < 1 << LABEL_BITS, "block {b}: the gain bit leaked into the label");
        }
    }
}

/// The whole file: `with_kind(Tetra)` writes a v5 header whose kind is Tetra,
/// `read_header` reports it, `read_all` decodes through it.
#[test]
fn a_v5_tetra_file_reads_end_to_end() {
    let tetra = Tetra::new();
    let (buf, m) = tetra_file(&tetra, 0x7210_0003_0004);
    assert_eq!(&buf[..4], MAGIC_V5);
    let head = read_header(&mut &buf[..]).expect("a fresh v5 file must open");
    assert_eq!(head.version, FIRST_KINDED_VERSION);
    assert_eq!(head.default_kind(), CodeKind::Tetra);
    assert_eq!(head.kinds(), KindSet::of(CodeKind::Tetra), "one kind declared, its own");
    assert!(!head.is_ball_only());
    assert_eq!(head.codebook, Some(llvq_artifact::codebook_fingerprint()));
    assert_eq!(head.tetra, Some(llvq_artifact::tetra_fingerprint()));
    assert!(head.is_self_contained(), "a v5 file is a sealed-shape file");

    let got = read_all(&mut &buf[..]).expect("read");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].codes, m.codes);
    assert_eq!(decode_matrix(&got[0]), decode_matrix(&m));
}

// ---------------------------------------------------------------------------
// 2 — the kind is load-bearing
// ---------------------------------------------------------------------------

/// The same Tetra record read as a Ball record — what a reader that ignored
/// the kind would do — is not the same matrix: some label is past the ball
/// and refused, or the points differ. Never silently equal.
///
/// Two refusals, one behind the other. [`read_matrix`] never gets that far:
/// the record says Tetra and it reads the v1 ball, so it stops by name. What
/// would have happened had it not is the second half — the very same labels,
/// which the sentinel cap keeps 47 bits wide either way, put through the
/// `Indexer` by hand.
#[test]
fn a_tetra_record_read_as_ball_is_not_the_same_matrix() {
    let tetra = Tetra::new();
    let cb = Codebook::new(CodeKind::Tetra);
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x7210_0003_0005);
    let m = synthetic_tetra(&tetra, &mut rng, "model.layers.0.mlp.gate_proj.weight", 8, 4 * DIM);
    let mut bytes: Vec<u8> = Vec::new();
    write_matrix_with(&mut bytes, FIRST_KINDED_VERSION, &cb, &m).expect("write");

    // The Ball entry point refuses the record by name, before a label goes
    // through the wrong map.
    match read_matrix(&mut &bytes[..], FIRST_KINDED_VERSION, &ix) {
        Err(Error::WrongCodeKind { want, got, name }) => {
            assert_eq!((want, got), (CodeKind::Ball, CodeKind::Tetra));
            assert_eq!(name, m.name);
        }
        other => panic!("expected WrongCodeKind, got {:?}", other.err()),
    }

    // The widths agree (the sentinel cap is what makes them), so the labels
    // are the same 47-bit words on either reading...
    let raw = read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION).expect("raw");
    assert_eq!(raw.kind, CodeKind::Tetra);
    for (b, &idx) in raw.indices.iter().enumerate() {
        assert_eq!(idx, tetra.encode(&m.codes[b].point).expect("a Tetra point"));
    }

    // ...and the other map makes a different matrix of them: a label past the
    // ball, or a point that is not the one written. Never silently equal.
    let mut differ = 0usize;
    let mut refused = 0usize;
    for (b, &idx) in raw.indices.iter().enumerate() {
        match ix.decode(idx) {
            None => refused += 1,
            Some(p) => differ += usize::from(p != m.codes[b].point),
        }
    }
    assert_eq!(
        refused + differ,
        raw.indices.len(),
        "{} of {} Tetra labels decoded to their own point through the ball",
        raw.indices.len() - refused - differ,
        raw.indices.len()
    );
}

/// A Tetra writer needs v5: below it the header has nowhere to say what its
/// records are, and `with_kind` picks v5 on its own.
#[test]
fn a_tetra_writer_below_v5_is_refused() {
    for version in 1..FIRST_KINDED_VERSION {
        match ArtifactWriter::with_version_kind(Vec::new(), version, 1, CodeKind::Tetra) {
            Err(Error::Inconsistent { name, detail }) => {
                assert_eq!(name, "header");
                assert!(detail.contains(&format!("v{version}")), "detail: {detail}");
                assert!(detail.contains("Tetra"), "detail: {detail}");
            }
            Err(e) => panic!("v{version}: wrong refusal {e}"),
            Ok(_) => panic!("a Tetra writer at v{version} was built: its records would read as Ball"),
        }
        assert!(
            matches!(
                llvq_artifact::write_header_kind(&mut Vec::new(), version, 1, CodeKind::Tetra),
                Err(Error::Inconsistent { .. })
            ),
            "write_header_kind must refuse v{version} for Tetra too"
        );
    }
    let w = ArtifactWriter::with_kind(Vec::new(), CodeKind::Tetra, 0).expect("v5");
    assert_eq!(w.default_kind(), CodeKind::Tetra);
    assert_eq!(w.kinds(), KindSet::of(CodeKind::Tetra));
    assert_eq!(w.kinds_used(), KindSet::empty(), "nothing pushed, nothing used");
    assert_eq!(CodeKind::Tetra.default_version(), FIRST_KINDED_VERSION);
    assert_eq!(CodeKind::Ball.default_version(), DEFAULT_VERSION);
    // And a v5 Ball writer is a thing: the kind is a field, not a version.
    let mut buf: Vec<u8> = Vec::new();
    ArtifactWriter::with_version_kind(&mut buf, FIRST_KINDED_VERSION, 0, CodeKind::Ball)
        .expect("v5 Ball")
        .finish()
        .expect("flush");
    let head = read_header(&mut &buf[..]).expect("opens");
    assert_eq!((head.version, head.default_kind()), (FIRST_KINDED_VERSION, CodeKind::Ball));
}

/// A Tetra matrix must carry the sentinel cap and one gain bit; anything else
/// is refused before a byte is written, by both writers.
#[test]
fn a_tetra_matrix_with_the_wrong_cap_or_gain_width_is_refused() {
    let tetra = Tetra::new();
    let cb = Codebook::new(CodeKind::Tetra);
    let mut rng = SplitMix64::new(0x7210_0003_0006);
    let base = synthetic_tetra(&tetra, &mut rng, "model.layers.0.mlp.down_proj.weight", 2, 2 * DIM);

    for cap in [0u32, 11, 13, 0xFFFF] {
        let m = QuantizedMatrix { shell_cap: cap, ..clone_of(&base) };
        match write_matrix_with(&mut Vec::new(), FIRST_KINDED_VERSION, &cb, &m) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("shell cap {cap}")), "detail: {detail}")
            }
            other => panic!("cap {cap}: expected Inconsistent, got {:?}", other.err()),
        }
    }
    for n_cent in [1usize, 3, 4] {
        let m = QuantizedMatrix {
            centroids: (0..n_cent).map(|k| 0.5 + 0.2 * k as f64).collect(),
            codes: base.codes.iter().map(|c| BlockCode { gain: c.gain.min(n_cent as u32 - 1), ..*c }).collect(),
            ..clone_of(&base)
        };
        match write_matrix_with(&mut Vec::new(), FIRST_KINDED_VERSION, &cb, &m) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("{n_cent} centroids")), "detail: {detail}")
            }
            other => panic!("{n_cent} centroids: expected Inconsistent, got {:?}", other.err()),
        }
    }

    // The raw writer under a Tetra kind: same two invariants.
    let raw = RawMatrix {
        name: base.name.clone(),
        d_out: base.d_out,
        d_in: base.d_in,
        kind: CodeKind::Tetra,
        indices: base.codes.iter().map(|c| tetra.encode(&c.point).expect("Tetra point")).collect(),
        gains: base.codes.iter().map(|c| c.gain).collect(),
        row_scales: base.row_scales.clone(),
        centroids: base.centroids.clone(),
        rotation_seed: base.rotation_seed,
        shell_cap: 13,
        tail: base.tail.clone(),
    };
    assert!(
        matches!(write_matrix_raw(&mut Vec::new(), FIRST_KINDED_VERSION, &raw), Err(Error::Inconsistent { .. })),
        "a raw Tetra record with cap 13 must be refused"
    );
    let raw = RawMatrix { shell_cap: TETRA_SHELL_CAP, centroids: vec![1.0], ..raw };
    assert!(
        matches!(write_matrix_raw(&mut Vec::new(), FIRST_KINDED_VERSION, &raw), Err(Error::Inconsistent { .. })),
        "a raw Tetra record with one centroid must be refused"
    );
    let raw = RawMatrix { centroids: base.centroids.clone(), ..raw };
    write_matrix_raw(&mut Vec::new(), FIRST_KINDED_VERSION, &raw).expect("the honest record writes");
}

/// A point the Tetra map has no word for is `PointOutsideCodebook`, not a
/// truncated or neighbouring label: a Tetra point with one coordinate's parity
/// flipped is off the label set by construction.
#[test]
fn a_point_off_the_tetra_label_set_is_refused_by_the_writer() {
    let tetra = Tetra::new();
    let cb = Codebook::new(CodeKind::Tetra);
    let mut rng = SplitMix64::new(0x7210_0003_0007);
    let mut m = synthetic_tetra(&tetra, &mut rng, "m", 2, DIM);
    m.codes[1].point[5] += 1;
    assert_eq!(cb.encode(&m.codes[1].point), None);
    match write_matrix_with(&mut Vec::new(), FIRST_KINDED_VERSION, &cb, &m) {
        Err(Error::PointOutsideCodebook { name }) => assert_eq!(name, "m"),
        other => panic!("expected PointOutsideCodebook, got {:?}", other.err()),
    }
}

/// `Codebook::Ball` is the `Indexer`, word for word — the enum adds a kind,
/// not a map.
#[test]
fn the_ball_codebook_is_the_indexer() {
    let cb = Codebook::new(CodeKind::Ball);
    let ix = Indexer::new();
    assert_eq!(cb.kind(), CodeKind::Ball);
    let mut rng = SplitMix64::new(0x7210_0003_0008);
    for _ in 0..2_000 {
        let idx = rng.next() % (N13 + 2);
        let p = ix.decode(idx);
        assert_eq!(cb.decode(idx, 1), p, "the gain is not part of a ball index");
        if let Some(p) = p {
            assert_eq!(cb.encode(&p), Some(idx));
        }
    }
    assert_eq!(cb.decode(N13 + 1, 0), None, "the first index past the ball");
}

// ---------------------------------------------------------------------------
// 3 — nothing downstream pretends
// ---------------------------------------------------------------------------

fn is_refusal(r: &Result<(), Error>) -> bool {
    matches!(r, Err(Error::Inconsistent { detail, .. }) if detail == REFUSAL)
}

/// The gate itself: `Ok` for Ball, the named `Inconsistent` for Tetra.
#[test]
fn the_gate_refuses_tetra_by_name() {
    require_ball(CodeKind::Ball, "anything").expect("Ball passes");
    match require_ball(CodeKind::Tetra, "Planes14") {
        Err(Error::Inconsistent { name, detail }) => {
            assert_eq!(name, "Planes14");
            assert_eq!(detail, REFUSAL);
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
    assert!(is_refusal(&require_ball(CodeKind::Tetra, "x")));
    let msg = require_ball(CodeKind::Tetra, "ClassTable").unwrap_err().to_string();
    assert_eq!(msg, format!("ClassTable: {REFUSAL}"));
}

/// Every runtime transcoder and table builder, through its `*_for_kind`
/// twin: the refusal on Tetra, and the very same stream as the original on
/// Ball — the twin adds a gate, not a layout.
#[test]
fn every_runtime_transcoder_refuses_a_tetra_header() {
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x7210_0003_0009);
    // 64 blocks: two E1v groups, four Planes12x rows of 16.
    let n = 64usize;
    let indices: Vec<u64> = (0..n).map(|_| 1 + rng.next() % N13).collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let _ = &ix;

    // Tables.
    assert!(ClassTable::for_kind(CodeKind::Tetra, &fd, 1).is_err());
    let table = ClassTable::for_kind(CodeKind::Ball, &fd, 1).expect("Ball table");
    let plain = ClassTable::new(&fd, 1);
    assert_eq!(table.n_entries(), plain.n_entries());
    for e in 0..table.n_entries() {
        assert_eq!(table.record(e).width, plain.record(e).width, "entry {e}");
    }
    assert!(Golay70Table::for_kind(CodeKind::Tetra, &fd).is_err());
    let g70 = Golay70Table::for_kind(CodeKind::Ball, &fd).expect("Ball g70");
    assert_eq!(g70.n_entries(), Golay70Table::new(&fd).n_entries());
    assert!(block_records_for_kind(CodeKind::Tetra, &fd, &golay).is_err());
    let recs = block_records_for_kind(CodeKind::Ball, &fd, &golay).expect("Ball records");
    assert_eq!(recs.len(), block_records(&fd, &golay).len());

    let refused = |r: Result<(), Error>, what: &str| {
        assert!(is_refusal(&r), "{what}: expected the named refusal, got {:?}", r.err());
    };

    // The five class layouts.
    for layout in [Layout::Fixed96, Layout::Grouped32, Layout::Flat32, Layout::Sorted32, Layout::Slot32] {
        refused(
            transcode_for_kind(CodeKind::Tetra, &fd, &table, &indices, &gains, layout).map(|_| ()),
            &format!("{layout:?}"),
        );
        let a = transcode_for_kind(CodeKind::Ball, &fd, &table, &indices, &gains, layout).expect("Ball");
        let b = transcode(&fd, &table, &indices, &gains, layout).expect("plain");
        assert_eq!((a.data, a.bases), (b.data, b.bases), "{layout:?}: the twin changed the stream");
    }
    // Planes14.
    refused(transcode_planes14_for_kind(CodeKind::Tetra, &fd, &table, &indices, &gains).map(|_| ()), "Planes14");
    assert_eq!(
        transcode_planes14_for_kind(CodeKind::Ball, &fd, &table, &indices, &gains).expect("Ball").data,
        transcode_planes14(&fd, &table, &indices, &gains).expect("plain").data
    );
    // Planes12x.
    let s = Searcher::new();
    refused(
        transcode_planes12x_for_kind(CodeKind::Tetra, &fd, &table, &s, &indices, &gains).map(|_| ()),
        "Planes12x",
    );
    let a = transcode_planes12x_for_kind(CodeKind::Ball, &fd, &table, &s, &indices, &gains).expect("Ball");
    let b = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("plain");
    assert_eq!((a.data, a.exc_idx, a.exc_data), (b.data, b.exc_idx, b.exc_data));
    // Golay70.
    refused(
        transcode_golay70_for_kind(CodeKind::Tetra, &fd, &table, &g70, &indices, &gains).map(|_| ()),
        "Golay70",
    );
    let a = transcode_golay70_for_kind(CodeKind::Ball, &fd, &table, &g70, &indices, &gains).expect("Ball");
    let b = transcode_golay70(&fd, &table, &g70, &indices, &gains).expect("plain");
    assert_eq!((a.data, a.exc_idx, a.exc_data), (b.data, b.exc_idx, b.exc_data));
    // E1v, both cuts.
    refused(transcode_e1v_for_kind(CodeKind::Tetra, &fd, &golay, &indices, &gains).map(|_| ()), "E1v");
    assert_eq!(
        transcode_e1v_for_kind(CodeKind::Ball, &fd, &golay, &indices, &gains).expect("Ball").data,
        transcode_e1v(&fd, &golay, &indices, &gains).expect("plain").data
    );
    refused(
        transcode_e1v_rows_for_kind(CodeKind::Tetra, &fd, &golay, &indices, &gains, 16).map(|_| ()),
        "E1v rows",
    );
    assert_eq!(
        transcode_e1v_rows_for_kind(CodeKind::Ball, &fd, &golay, &indices, &gains, 16).expect("Ball").data,
        transcode_e1v_rows(&fd, &golay, &indices, &gains, 16).expect("plain").data
    );
}

// ---------------------------------------------------------------------------
// The reconstruction, factored and pinned
// ---------------------------------------------------------------------------

/// `decode_matrix` no longer builds a `LeechShapeGain` per matrix; the free
/// function it calls must be the quantizer's own reconstruction, bit for
/// bit, on a Ball matrix and on a Tetra one.
#[test]
fn decode_matrix_is_the_quantizers_reconstruction() {
    let tetra = Tetra::new();
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x7210_0003_000a);
    let mut mats = vec![synthetic_tetra(&tetra, &mut rng, "tetra", 3, 2 * DIM + 8)];
    // A Ball matrix at cap 12 with an origin block, a tail and no rotation.
    let mut ball = synthetic_tetra(&tetra, &mut rng, "ball", 3, 2 * DIM + 8);
    ball.rotation_seed = None;
    for c in &mut ball.codes {
        c.point = loop {
            let i = rng.next() % (1u64 << 47);
            if let Some(p) = ix.decode(i) {
                if llvq_core::Leech::shell_index(&p).is_some_and(|m| m <= 12) {
                    break p;
                }
            }
        };
    }
    ball.codes[0].point = [0; DIM];
    mats.push(ball);

    for m in &mats {
        let q = LeechShapeGain::with_shell_cap(m.centroids.clone(), TETRA_SHELL_CAP);
        let nblocks = m.d_in / DIM;
        let tail_w = m.d_in % DIM;
        let mut w = vec![0.0f64; m.d_out * m.d_in];
        let mut block = [0.0f64; DIM];
        for i in 0..m.d_out {
            for p in 0..nblocks {
                q.reconstruct(&m.codes[i * nblocks + p], m.row_scales[i], &mut block);
                w[i * m.d_in + p * DIM..i * m.d_in + (p + 1) * DIM].copy_from_slice(&block);
            }
            if tail_w > 0 {
                let at = i * m.d_in + nblocks * DIM;
                w[at..at + tail_w].copy_from_slice(&m.tail[i * tail_w..(i + 1) * tail_w]);
            }
        }
        if let Some(seed) = m.rotation_seed {
            llvq_quant::rotation::Rotation::new(m.d_in, seed).unrotate_weight_rows(&mut w, m.d_out);
        }
        let want: Vec<f32> = w.into_iter().map(|v| v as f32).collect();
        assert_eq!(decode_matrix(m), want, "{}: the factored reconstruction drifted", m.name);
    }
}

fn clone_of(m: &QuantizedMatrix) -> QuantizedMatrix {
    QuantizedMatrix {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        codes: m.codes.clone(),
        row_scales: m.row_scales.clone(),
        centroids: m.centroids.clone(),
        rotation_seed: m.rotation_seed,
        shell_cap: m.shell_cap,
        tail: m.tail.clone(),
    }
}
