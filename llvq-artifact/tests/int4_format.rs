//! # The `int4 g128` record — the second half of Q5's mixed file
//!
//! `v_proj` served as affine int4 beside Tetra matrices in one `.llvq` v5
//! (`docs/ROADMAP.md` §2.3, operator decision of 2026-09-06). The record
//! stores its weights instead of indexing them, which makes its failure modes
//! different in kind from a lattice record's: there is no map to be wrong
//! about, and everything that can go wrong is a field read in the wrong place
//! or an axis read along the wrong dimension.
//!
//! Five claims, each with its own evidence:
//!
//! 1. **The payload round-trips**, field by field and value by value.
//! 2. **The rate is 4.250 b/weight**, exactly, and that is the number the
//!    cost tables quote.
//! 3. **The groups run along `d_in`**, provable on a non-square matrix.
//! 4. **Neither reading crosses over.** An int4 record read as Ball or as
//!    Tetra is refused by the shell-cap sentinel; a lattice record read as
//!    int4 contradicts itself in three independent fields.
//! 5. **The version argument is load-bearing.** A v5 int4 record read as v4
//!    is not the same record and must not decode.
//!
//! In-memory bytes only: no sealed artifact, no `#[ignore]`, always in the
//! fast loop.

use llvq_artifact::{
    read_matrix_raw, read_record, write_matrix_int4, ArtifactWriter, CodeKind, Error, Int4Matrix,
    KindSet, QuantizedMatrix, Record, FIRST_KINDED_VERSION, INT4G128_BITS, INT4G128_GROUP,
    INT4G128_SHELL_CAP, RESERVED_INT4G128_TAG, TETRA_SHELL_CAP,
};
use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::BlockCode;
use llvq_search::tetra::{Tetra, LABEL_MASK};

/// An int4 record with pseudo-random nibbles, scales and biases. The values
/// are arbitrary on purpose: this file tests the *container*, and
/// `llvq-llm/tests/int4_wiring.rs` tests that the numbers in it are the ones
/// the quantizer chose.
fn int4_matrix(name: &str, d_out: usize, d_in: usize) -> Int4Matrix {
    let mut rng = SplitMix64::new(0x1234_5678_9abc_def0);
    let groups = d_out * (d_in / INT4G128_GROUP);
    Int4Matrix {
        name: name.to_string(),
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed: (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect(),
        // Bit patterns of small positive f16s: any pattern decodes, but a NaN
        // would make the value comparison below vacuous.
        scales: (0..groups).map(|_| 0x3000 | (rng.next() as u16 & 0x03ff)).collect(),
        biases: (0..groups).map(|_| 0xb000 | (rng.next() as u16 & 0x03ff)).collect(),
    }
}

fn record_bytes(m: &Int4Matrix) -> Vec<u8> {
    let mut out = Vec::new();
    write_matrix_int4(&mut out, FIRST_KINDED_VERSION, m).expect("write");
    out
}

/// Byte offset of the record's kind tag: name length, name, d_out, d_in,
/// shell cap.
fn kind_at(m: &Int4Matrix) -> usize {
    4 + m.name.len() + 4 + 4 + 4
}

/// Byte offset of the centroid count, which follows the kind tag.
fn centroids_at(m: &Int4Matrix) -> usize {
    kind_at(m) + 4
}

/// Byte offset of the rotation flag: centroid count (u32), seed (u64).
fn rot_flag_at(m: &Int4Matrix) -> usize {
    centroids_at(m) + 4 + 8
}

/// Byte offset of the payload length, which follows the rotation flag.
fn payload_len_at(m: &Int4Matrix) -> usize {
    rot_flag_at(m) + 4
}

#[test]
fn int4_record_round_trips() {
    let m = int4_matrix("model.layers.0.self_attn.v_proj.weight", 3, 256);
    let bytes = record_bytes(&m);
    let Record::Int4(back) = read_record(&mut &bytes[..], FIRST_KINDED_VERSION).expect("read")
    else {
        panic!("an int4 record came back as a lattice one");
    };
    assert_eq!(back.name, m.name);
    assert_eq!((back.d_out, back.d_in), (m.d_out, m.d_in));
    assert_eq!(back.bits, INT4G128_BITS);
    assert_eq!(back.group, INT4G128_GROUP);
    // Field by field, so a payload whose scales and biases were swapped — or
    // whose packed length and group count were — fails here and not on some
    // aggregate that both orders satisfy.
    assert_eq!(back.packed, m.packed);
    assert_eq!(back.scales, m.scales);
    assert_eq!(back.biases, m.biases);
    // And bit for bit on the way out, which is what a reader actually gets.
    let (a, b) = (m.to_f32(), back.to_f32());
    assert_eq!(a.len(), m.d_out * m.d_in);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "weight {i}");
    }
}

#[test]
fn an_int4_record_costs_four_and_a_quarter_bits() {
    for (d_out, d_in) in [(1usize, 128usize), (3, 256), (7, 1024)] {
        let m = int4_matrix("m", d_out, d_in);
        let n = (d_out * d_in) as u64;
        // 4 for the nibble, 32 per group of 128 for the f16 scale and the f16
        // bias: 4.250, and the quarter is where AWQ's packed zero puts it at
        // 4.15625. The difference is declared, and against us.
        assert_eq!(m.bits() * 4, n * 17, "{d_out}×{d_in}: not exactly 4.250 b/weight");
        assert_eq!(m.bits() as f64 / n as f64, 4.25);
        // And the number the **writer** hands back, which is what
        // `ArtifactWriter::payload_bits` accumulates and what every rate line
        // downstream divides. Asserting `Int4Matrix::bits()` alone leaves the
        // writer free to return the nibbles and drop the scales and biases:
        // 4.000 instead of 4.250, under-declared and in our favour, which is
        // the shape of the 2.73-reported-as-2.07 accounting error.
        let mut out = Vec::new();
        let written = write_matrix_int4(&mut out, FIRST_KINDED_VERSION, &m).expect("write");
        assert_eq!(written, m.bits(), "{d_out}×{d_in}: the writer's own count");
        assert_eq!(written * 4, n * 17);
    }
}

/// The writer's total, through the entry a mixed run actually uses.
///
/// `push_int4` adds the record's bits to `payload_bits` and `finish` hands the
/// sum back; `bin/smoke` divides that sum by
/// `llvq_llm::calib::Report::quantized_weights()`. Nothing between the two is
/// asserted anywhere else.
#[test]
fn the_writers_payload_counts_each_int4_record_in_full() {
    let a = int4_matrix("model.layers.0.self_attn.v_proj.weight", 3, 256);
    let b = int4_matrix("model.layers.1.self_attn.v_proj.weight", 5, 128);
    let mut out = Vec::new();
    let mut w = ArtifactWriter::with_kinds(
        &mut out,
        FIRST_KINDED_VERSION,
        2,
        CodeKind::Ball,
        KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
    )
    .expect("header");
    w.push_int4(&a).expect("push a");
    w.push_int4(&b).expect("push b");
    let bits = w.finish().expect("finish");
    assert_eq!(bits, a.bits() + b.bits());
    let n = ((a.d_out * a.d_in) + (b.d_out * b.d_in)) as u64;
    assert_eq!(bits * 4, n * 17, "the file's int4 half is not at 4.250");
}

/// `to_f32` against literal values, in this crate, without the other path.
///
/// Every other test here compares `to_f32` to `to_f32`, or zeroes the biases
/// and fills the nibbles with `0xf` — under which a decoder that dropped the
/// bias **and** read the high nibble first still passes. Measured: both
/// mutations applied together leave all 22 test binaries of this crate green.
/// The only net was `the_two_int4_paths_agree_bit_for_bit`, in another crate,
/// and a file written by a future external encoder would read back inverted
/// with nothing here to say so.
#[test]
fn to_f32_is_scale_times_the_low_nibble_first_plus_the_bias() {
    let mut packed = vec![0u8; 128 / 2];
    // Low nibble 1, high nibble 2: the two are different on purpose, and they
    // are not the two halves of a palindrome.
    packed[0] = 0x21;
    let m = Int4Matrix {
        name: "m".into(),
        d_out: 1,
        d_in: 128,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed,
        // f16 1.0 and f16 0.5, so scale·q + bias is exact in binary and the
        // assertion needs no tolerance.
        scales: vec![0x3c00],
        biases: vec![0x3800],
    };
    let w = m.to_f32();
    assert_eq!(w.len(), 128);
    assert_eq!(w[0], 1.5, "q = 1 must come from the LOW nibble of packed[0]");
    assert_eq!(w[1], 2.5, "q = 2 must come from the HIGH nibble of packed[0]");
    // q = 0, so this weight is the bias and nothing else: drop the bias term
    // and it reads 0.0.
    assert_eq!(w[2], 0.5, "scale·q + bias, not scale·q");
}

#[test]
fn the_group_axis_runs_along_d_in() {
    // Two rows, two groups each. Row 0 is scaled a hundred times row 1, so a
    // reader that grouped along `d_out` — or computed groups-per-row from
    // `d_out` — would mix the two dynamics and the assertion below would fail.
    // A square matrix would hide it entirely.
    let (d_out, d_in) = (2usize, 256usize);
    let mut m = int4_matrix("m", d_out, d_in);
    assert_eq!(m.scales.len(), d_out * (d_in / INT4G128_GROUP));
    assert_eq!(m.scales.len(), 4);
    // f16 0x5000 = 32, 0x2400 = 0.0625: three orders apart.
    m.scales = vec![0x5000, 0x5000, 0x2400, 0x2400];
    m.biases = vec![0; 4];
    m.packed = vec![0xff; d_out * d_in / 2]; // every q = 15
    let w = m.to_f32();
    let row0 = w[0..d_in].iter().fold(0f32, |a, b| a.max(*b));
    let row1 = w[d_in..].iter().fold(0f32, |a, b| a.max(*b));
    assert!(row0 > 100.0 * row1, "row0 {row0}, row1 {row1}");
    assert_eq!(m.groups_per_row(), 2);
}

#[test]
fn the_sentinel_cap_is_illegal_for_both_lattice_kinds() {
    // Trivial on purpose, and that is its reason to exist: the whole argument
    // of the layout is that an int4 head cannot be read as a lattice head, and
    // the argument rests entirely on this one value.
    const { assert!(INT4G128_SHELL_CAP > llvq_search::classes::MAX_SHELL) };
    assert_ne!(INT4G128_SHELL_CAP, TETRA_SHELL_CAP);
    assert_eq!(INT4G128_GROUP, 128);
    assert_eq!(INT4G128_BITS, 4);
}

#[test]
fn the_int4_tag_is_the_int4_kind() {
    assert_eq!(RESERVED_INT4G128_TAG, 2);
    assert_eq!(CodeKind::Int4G128.tag(), RESERVED_INT4G128_TAG);
    assert_eq!(CodeKind::from_tag(2).expect("tag 2"), CodeKind::Int4G128);
    assert!(CodeKind::ALL.contains(&CodeKind::Int4G128));
    // The bit opens in the header's set at the same time, and by the same
    // arithmetic — `1 << tag`, never a second constant.
    let set = KindSet::from_bits(1 << RESERVED_INT4G128_TAG).expect("bit 2");
    assert!(set.contains(CodeKind::Int4G128));
    assert!(!set.is_ball_only());
    // Tag 3 is the lowest unknown one now, and the message says 3.
    match CodeKind::from_tag(3) {
        Err(Error::UnknownCodeKind { tag }) => assert_eq!(tag, 3),
        other => panic!("expected UnknownCodeKind, got {:?}", other.map(|k| k.name())),
    }
    assert!(CodeKind::from_tag(3).unwrap_err().to_string().contains('3'));
    // An int4 file is a v5 file: there is nowhere in a v4 record to say so.
    assert_eq!(CodeKind::Int4G128.default_version(), FIRST_KINDED_VERSION);
}

#[test]
fn an_int4_record_read_as_ball_is_refused() {
    let m = int4_matrix("model.layers.0.self_attn.v_proj.weight", 2, 256);
    let good = record_bytes(&m);
    let at = kind_at(&m);
    // Ball: `index_width` refuses a cap above the supported ball before it
    // enumerates a single class. Tetra: the cap must be exactly 12.
    for (tag, kind) in [(CodeKind::Ball, "Ball"), (CodeKind::Tetra, "Tetra")] {
        let mut bytes = good.clone();
        bytes[at..at + 4].copy_from_slice(&tag.tag().to_le_bytes());
        match read_record(&mut &bytes[..], FIRST_KINDED_VERSION) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains("shell cap"), "{kind}: {detail}")
            }
            other => panic!("{kind}: expected Inconsistent, got {:?}", other.map(|_| "a record")),
        }
    }
}

#[test]
fn a_lattice_record_read_as_int4_is_refused() {
    // A Ball record with a rotation seed — the shipped 4B's shape — so all
    // three fields the int4 reader checks disagree at once. Each is removed in
    // turn, and the next one has to catch it: one field could be a
    // coincidence, three cannot.
    let mut rng = SplitMix64::new(7);
    let ix = llvq_search::index::Indexer::new();
    let codes: Vec<BlockCode> = (0..2 * (48 / DIM))
        .map(|_| loop {
            let i = rng.next() % (1u64 << 47);
            if let Some(point) = ix.decode(i) {
                if llvq_core::Leech::shell_index(&point).is_none_or(|s| s <= 12) {
                    return BlockCode { point, gain: 0 };
                }
            }
        })
        .collect();
    let ball = QuantizedMatrix {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        d_out: 2,
        d_in: 48,
        codes,
        row_scales: vec![1.0, 2.0],
        centroids: vec![0.5, 1.5],
        rotation_seed: Some(0x5EED),
        shell_cap: 12,
        tail: vec![],
    };
    let mut good = Vec::new();
    llvq_artifact::write_matrix_with(
        &mut good,
        FIRST_KINDED_VERSION,
        &llvq_artifact::Codebook::new(CodeKind::Ball).expect("ball"),
        &ball,
    )
    .expect("write");
    let name_len = ball.name.len();
    let kind_at = 4 + name_len + 12;
    let cap_at = 4 + name_len + 8;
    let cent_at = kind_at + 4;
    let rot_flag_at = cent_at + 4 + 8;

    let mut bytes = good.clone();
    bytes[kind_at..kind_at + 4].copy_from_slice(&CodeKind::Int4G128.tag().to_le_bytes());
    let detail = |b: &[u8]| match read_record(&mut &b[..], FIRST_KINDED_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => detail,
        other => panic!("expected Inconsistent, got {:?}", other.map(|_| "a record")),
    };
    assert!(detail(&bytes).contains("shell cap"), "the cap must be the first contradiction");
    // Take the cap away, and the centroid count still contradicts.
    bytes[cap_at..cap_at + 4].copy_from_slice(&INT4G128_SHELL_CAP.to_le_bytes());
    let d = detail(&bytes);
    assert!(d.contains("centroids"), "{d}");
    // Take that away too, and the rotation flag still does.
    bytes[cent_at..cent_at + 4].copy_from_slice(&0u32.to_le_bytes());
    let d = detail(&bytes);
    assert!(d.contains("natural basis"), "{d}");
    assert_eq!(u32::from_le_bytes(bytes[rot_flag_at..rot_flag_at + 4].try_into().unwrap()), 1);
}

#[test]
fn an_int4_record_refuses_a_rotation_seed() {
    // The record decodes with no transform at all, so a seed here would not be
    // undone: the weights would come out plausible and wrong, which is the one
    // failure this format spends its bytes to avoid.
    let m = int4_matrix("v", 2, 128);
    let mut bytes = record_bytes(&m);
    let at = rot_flag_at(&m);
    assert_eq!(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()), 0);
    bytes[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    match read_record(&mut &bytes[..], FIRST_KINDED_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("natural basis"), "{detail}")
        }
        other => panic!("expected Inconsistent, got {:?}", other.map(|_| "a record")),
    }
}

#[test]
fn an_int4_record_refuses_a_group_that_is_not_128() {
    // Three distinct refusals, at the writer. A `div_ceil` for the groups per
    // row would make the last group of a non-multiple `d_in` short, and the
    // 4.250 b/weight of the cost table would be wrong on it.
    let mut m = int4_matrix("v", 2, 256);
    m.group = 64;
    let e = write_matrix_int4(&mut Vec::new(), FIRST_KINDED_VERSION, &m).unwrap_err();
    assert!(e.to_string().contains("group 64"), "{e}");
    m.group = 0;
    let e = write_matrix_int4(&mut Vec::new(), FIRST_KINDED_VERSION, &m).unwrap_err();
    assert!(e.to_string().contains("group 0"), "{e}");
    // A `d_in` that is not a multiple of the group, with the group correct.
    let mut m = int4_matrix("v", 2, 256);
    m.d_in = 100;
    m.packed = vec![0; 2 * 100 / 2];
    let e = write_matrix_int4(&mut Vec::new(), FIRST_KINDED_VERSION, &m).unwrap_err();
    assert!(e.to_string().contains("not a multiple"), "{e}");
    // And the same refusal on the reading side, from a record whose stored
    // `d_in` was corrupted to a non-multiple.
    let good = int4_matrix("v", 2, 256);
    let mut bytes = record_bytes(&good);
    let d_in_at = 4 + good.name.len() + 4;
    bytes[d_in_at..d_in_at + 4].copy_from_slice(&100u32.to_le_bytes());
    match read_record(&mut &bytes[..], FIRST_KINDED_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => assert!(
            detail.contains("not a multiple") || detail.contains("packed bytes"),
            "{detail}"
        ),
        other => panic!("expected Inconsistent, got {:?}", other.map(|_| "a record")),
    }
}

#[test]
fn a_v5_int4_record_read_as_v4_is_not_the_same_record() {
    // At v4 there is no kind tag: the 2 is read as a centroid count, every
    // field behind it slides four bytes, and the payload length is taken from
    // the nibbles. The twin of `a_v5_record_read_as_v4_is_not_the_same_record`
    // in `hostile_files.rs` — a `version` argument that changed nothing here
    // would be dead code that misreads every record of the other version.
    let m = int4_matrix("model.layers.0.self_attn.v_proj.weight", 2, 256);
    let bytes = record_bytes(&m);
    assert!(
        read_record(&mut &bytes[..], 4).is_err(),
        "v5 int4 bytes read as v4 produced a record"
    );
    assert!(read_matrix_raw(&mut &bytes[..], 4).is_err());
    // And a v4 int4 record cannot be written in the first place.
    let e = write_matrix_int4(&mut Vec::new(), 4, &m).unwrap_err();
    assert!(e.to_string().contains("v5"), "{e}");
}

#[test]
fn an_int4_record_is_refused_by_every_lattice_entry() {
    let m = int4_matrix("v", 2, 256);
    let bytes = record_bytes(&m);
    // The one refusal that covers the benches, the five Metal binaries, the
    // two CUDA ones, `fused.rs` and `export` without any of them being
    // touched.
    match read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION) {
        Err(Error::NotALatticeRecord { name, kind }) => {
            assert_eq!(name, m.name);
            assert_eq!(kind, CodeKind::Int4G128);
        }
        other => panic!("expected NotALatticeRecord, got {:?}", other.map(|_| "a record")),
    }
    // A hand-built `RawMatrix` wearing the int4 kind cannot be written either:
    // that would be a lattice head over int4 bytes.
    let raw = llvq_artifact::RawMatrix {
        name: "v".into(),
        d_out: 1,
        d_in: DIM,
        kind: CodeKind::Int4G128,
        indices: vec![0],
        gains: vec![0],
        row_scales: vec![1.0],
        centroids: vec![0.5, 1.5],
        rotation_seed: None,
        shell_cap: 12,
        tail: vec![],
    };
    assert!(matches!(
        llvq_artifact::write_matrix_raw(&mut Vec::new(), FIRST_KINDED_VERSION, &raw),
        Err(Error::NotALatticeRecord { .. })
    ));
    // And there is no map to hand out for it.
    assert!(matches!(
        llvq_artifact::Codebook::new(CodeKind::Int4G128),
        Err(Error::NoCodebookForKind { .. })
    ));
    let cbs = llvq_artifact::Codebooks::new();
    assert!(matches!(
        cbs.get(CodeKind::Int4G128),
        Err(Error::NoCodebookForKind { .. })
    ));
    assert!(!cbs.is_built(CodeKind::Int4G128));
}

#[test]
fn the_int4_head_carries_the_shared_fields_and_nothing_else() {
    // The head is the lattice head up to the kind tag, then three fields
    // pinned to zero and the payload. Written out here field by field because
    // the whole argument of the layout is where the payload length sits: four
    // bytes out and a reader takes its length from the nibbles.
    let m = int4_matrix("v_proj", 2, 256);
    let bytes = record_bytes(&m);
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    assert_eq!(u32_at(0), m.name.len() as u32);
    assert_eq!(u32_at(4 + m.name.len()), 2);
    assert_eq!(u32_at(4 + m.name.len() + 4), 256);
    assert_eq!(u32_at(4 + m.name.len() + 8), INT4G128_SHELL_CAP);
    assert_eq!(u32_at(kind_at(&m)), CodeKind::Int4G128.tag());
    assert_eq!(u32_at(centroids_at(&m)), 0, "no centroids");
    assert_eq!(u64_at(centroids_at(&m) + 4), 0, "no rotation seed");
    assert_eq!(u32_at(rot_flag_at(&m)), 0, "the natural basis");
    let groups = m.d_out * (m.d_in / INT4G128_GROUP);
    let payload = u64_at(payload_len_at(&m));
    assert_eq!(payload, (16 + m.d_out * m.d_in / 2 + 8 + 4 * groups) as u64);
    assert_eq!(bytes.len(), payload_len_at(&m) + 8 + payload as usize);
    // Nothing between the head and the payload: no row scales, no tail.
    assert_eq!(payload_len_at(&m), kind_at(&m) + 4 + 4 + 8 + 4);
}

#[test]
fn a_mixed_file_round_trips_three_kinds() {
    // default = Tetra, set = {Tetra, Int4G128}: the shape Q5 writes.
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0xC0FFEE);
    let t0 = tetra_matrix(&tetra, &mut rng, "model.layers.0.self_attn.q_proj.weight", 2, 2 * DIM);
    let i0 = int4_matrix("model.layers.0.self_attn.v_proj.weight", 2, 256);
    let t1 = tetra_matrix(&tetra, &mut rng, "model.layers.0.self_attn.o_proj.weight", 1, 3 * DIM);

    let mut file = Vec::new();
    {
        let mut w = ArtifactWriter::with_kinds(
            &mut file,
            FIRST_KINDED_VERSION,
            3,
            CodeKind::Tetra,
            KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128),
        )
        .expect("header");
        w.push(&t0).expect("tetra 0");
        w.push_int4(&i0).expect("int4");
        w.push(&t1).expect("tetra 1");
        w.finish().expect("finish");
    }

    let mut r = &file[..];
    let head = llvq_artifact::read_header(&mut r).expect("header");
    assert_eq!(head.default_kind(), CodeKind::Tetra);
    assert!(head.kinds().contains(CodeKind::Int4G128));
    let want = [CodeKind::Tetra, CodeKind::Int4G128, CodeKind::Tetra];
    let mut records = Vec::new();
    for (i, k) in want.iter().enumerate() {
        let rec = read_record(&mut r, head.version).expect("record");
        assert_eq!(rec.kind(), *k, "record {i}: the kind comes from the record, not the header");
        records.push(rec);
    }
    // The int4 record's weights are its own, not the default kind's.
    let Record::Int4(back) = &records[1] else { panic!("record 1 is not int4") };
    let (a, b) = (i0.to_f32(), back.to_f32());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "weight {i}");
    }
    // The Tetra records still decode to their own points.
    let cbs = llvq_artifact::Codebooks::new();
    let Record::Lattice(raw) = &records[0] else { panic!("record 0 is not lattice") };
    assert_eq!(raw.indices.len(), t0.codes.len());
    let cb = cbs.get(CodeKind::Tetra).expect("tetra map");
    for (j, (&idx, &g)) in raw.indices.iter().zip(&raw.gains).enumerate() {
        assert_eq!(cb.decode(idx, g), Some(t0.codes[j].point), "block {j}");
    }

    // `read_all` refuses the whole file at the header rather than in its
    // middle: the message has to send the reader to `read_record`.
    match llvq_artifact::read_all(&mut &file[..]) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("read_record"), "{detail}")
        }
        other => panic!("read_all accepted a mixed file: {:?}", other.map(|v| v.len())),
    }

    // ---- a passthrough is byte-identical ----
    let mut again = Vec::new();
    {
        let mut w = ArtifactWriter::with_kinds(
            &mut again,
            FIRST_KINDED_VERSION,
            3,
            CodeKind::Tetra,
            KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128),
        )
        .expect("header");
        for rec in &records {
            w.push_record(rec).expect("push");
        }
        w.finish().expect("finish");
    }
    assert_eq!(again, file, "a passthrough of a mixed file must not move a byte");
}

#[test]
fn push_of_an_undeclared_int4_is_refused() {
    // The header is on the disk before the first record and cannot be told
    // about a kind afterwards. A push that skipped `declare` would produce a
    // file whose own `KindSet` lies, and `require_ball_kinds` would wave it
    // through.
    let mut file = Vec::new();
    let mut w =
        ArtifactWriter::with_kinds(&mut file, FIRST_KINDED_VERSION, 1, CodeKind::Tetra, KindSet::of(CodeKind::Tetra))
            .expect("header");
    let m = int4_matrix("v", 1, 128);
    match w.push_int4(&m) {
        Err(Error::KindNotDeclared { kind, declared, .. }) => {
            assert_eq!(kind, CodeKind::Int4G128);
            assert_eq!(declared, KindSet::of(CodeKind::Tetra));
        }
        other => panic!("expected KindNotDeclared, got {:?}", other.err()),
    }
}

#[test]
fn the_header_set_refuses_a_file_with_one_int4_matrix() {
    let set = KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128);
    let e = llvq_artifact::runtime::require_ball_kinds(set, "Planes14").unwrap_err();
    // The set is walked in tag order, so Tetra is named first here. The point
    // is that the set is walked at all: a check that read only the default
    // kind would pass a file whose `v_proj` records are int4.
    assert!(e.to_string().contains("Tetra"), "{e}");
    let only = KindSet::of(CodeKind::Int4G128);
    let e = llvq_artifact::runtime::require_ball_kinds(only, "Planes14").unwrap_err();
    assert!(e.to_string().contains("Int4G128"), "{e}");
}

#[test]
fn require_ball_names_int4_and_not_tetra() {
    // A bras added by copy-paste would send an operator to F1d's Tetra
    // transcoder for a record that needs a kernel reading stored weights.
    let t = match llvq_artifact::runtime::require_ball(CodeKind::Tetra, "Planes14") {
        Err(Error::Inconsistent { detail, .. }) => detail,
        other => panic!("Tetra accepted: {:?}", other.err()),
    };
    let i = match llvq_artifact::runtime::require_ball(CodeKind::Int4G128, "Planes14") {
        Err(Error::Inconsistent { detail, .. }) => detail,
        other => panic!("Int4G128 accepted: {:?}", other.err()),
    };
    assert_ne!(t, i);
    assert!(i.contains("Int4G128"), "{i}");
    assert!(!i.contains("Tetra"), "{i}");
    assert!(llvq_artifact::runtime::require_ball(CodeKind::Ball, "Planes14").is_ok());
}

/// A Tetra matrix: labels through the map itself, one gain bit, the sentinel
/// cap.
fn tetra_matrix(
    tetra: &Tetra,
    rng: &mut SplitMix64,
    name: &str,
    d_out: usize,
    d_in: usize,
) -> QuantizedMatrix {
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
        centroids: vec![0.7, 1.4],
        rotation_seed: None,
        shell_cap: TETRA_SHELL_CAP,
        tail: (0..d_out * (d_in % DIM)).map(|_| rng.next_gaussian() as f32 as f64).collect(),
    }
}
