//! The reader against files that lie.
//!
//! Every length, count and tag a `.llvq` file carries is a claim, not a fact.
//! The failure mode this suite polices is the reader trusting one of them
//! before the bytes back it up: a corrupted length that aborts the process on
//! OOM, a self-consistent-but-short code stream that panics inside
//! `BitReader`, a wild shell cap that trips an assert in llvq-search. All of
//! those must surface as the *named* [`Error`] variant — a reader that
//! panics on hostile input cannot claim to be auditable, and a reader that
//! aborts cannot even be caught.
//!
//! These tests run against in-memory bytes only: no sealed artifact, no
//! `#[ignore]`, always in the fast loop — including debug, where the
//! overflow checks live.

use llvq_artifact::{
    read_blob, read_header, read_matrix_raw, read_raw, write_header, write_header_kinds,
    write_matrix_raw, write_matrix_with, write_raw, CodeKind, Codebook, Error, KindSet, QuantData,
    QuantizedMatrix, RawData, RawMatrix, RawTensor, DEFAULT_VERSION, FIRST_KINDED_VERSION,
    RESERVED_INT4G128_TAG, TETRA_SHELL_CAP, VERSION,
};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::pack::BitWriter;

/// A minimal valid matrix: one row, two 24-blocks, cap 12 (47-bit indices),
/// one centroid (0 gain bits), no tail.
fn small_matrix() -> RawMatrix {
    RawMatrix {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        d_out: 1,
        d_in: 48,
        kind: CodeKind::Ball,
        indices: vec![3, 5],
        gains: vec![0, 0],
        row_scales: vec![1.0],
        centroids: vec![1.0],
        rotation_seed: None,
        shell_cap: 12,
        tail: vec![],
    }
}

/// The same shape as a Tetra record: the sentinel cap, two centroids (the one
/// gain bit), two 47-bit labels. Raw records are unvalidated labels, so no
/// map is needed to build one.
fn small_tetra_matrix() -> RawMatrix {
    RawMatrix {
        kind: CodeKind::Tetra,
        gains: vec![0, 1],
        centroids: vec![0.7, 1.1],
        shell_cap: TETRA_SHELL_CAP,
        ..small_matrix()
    }
}

/// A v4 Ball record: no kind field, the shape every file before v5 has.
fn matrix_bytes(m: &RawMatrix) -> Vec<u8> {
    let mut out = Vec::new();
    write_matrix_raw(&mut out, DEFAULT_VERSION, m).expect("a valid matrix must serialize");
    out
}

/// A v5 record, kind tag included — `m.kind` is what it says.
fn v5_matrix_bytes(m: &RawMatrix) -> Vec<u8> {
    let mut out = Vec::new();
    write_matrix_raw(&mut out, FIRST_KINDED_VERSION, m).expect("a valid record must serialize");
    out
}

/// Byte offset of the shell cap field: name length (4) + name + d_out (4) +
/// d_in (4).
fn shell_cap_at(m: &RawMatrix) -> usize {
    4 + m.name.len() + 4 + 4
}

/// Byte offset of a v5 record's kind tag: straight after the shell cap.
fn kind_at(m: &RawMatrix) -> usize {
    shell_cap_at(m) + 4
}

#[test]
fn wrong_magic_is_refused_by_name() {
    let bytes = [0xDEu8, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0];
    match read_header(&mut &bytes[..]) {
        Err(Error::NotAnArtifact { got }) => assert_eq!(got, [0xDE, 0xAD, 0xBE, 0xEF]),
        other => panic!("expected NotAnArtifact, got {:?}", other.err()),
    }
}

#[test]
fn an_empty_file_is_truncated_at_the_magic() {
    match read_header(&mut &[][..]) {
        Err(Error::Truncated { reading: "magic" }) => {}
        other => panic!("expected Truncated at the magic, got {:?}", other.err()),
    }
}

#[test]
fn a_header_cut_before_its_count_is_truncated() {
    let mut bytes = Vec::new();
    write_header(&mut bytes, VERSION, 7).unwrap();
    bytes.truncate(6); // magic + half the count
    match read_header(&mut &bytes[..]) {
        Err(Error::Truncated {
            reading: "matrix count",
        }) => {}
        other => panic!("expected Truncated at the count, got {:?}", other.err()),
    }
}

#[test]
fn a_matrix_cut_mid_name_is_truncated_not_a_panic() {
    let mut bytes = matrix_bytes(&small_matrix());
    bytes.truncate(9); // 4 bytes of name length + 5 bytes of a longer name
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Truncated { reading: "name" }) => {}
        other => panic!("expected Truncated in the name, got {:?}", other.err()),
    }
}

#[test]
fn a_lying_name_length_cannot_abort_the_process() {
    // Name length u32::MAX over a 3-byte body: the reader must report the
    // missing bytes, not reserve 4 GiB on the field's say-so.
    let mut bytes = u32::MAX.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"abc");
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Truncated { reading: "name" }) => {}
        other => panic!("expected Truncated, got {:?}", other.err()),
    }
}

#[test]
fn a_bad_utf8_name_is_bad_name() {
    let mut bytes = 2u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0xFF, 0xFE]);
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::BadName) => {}
        other => panic!("expected BadName, got {:?}", other.err()),
    }
}

#[test]
fn a_wild_shell_cap_is_refused_not_a_panic() {
    // Beyond the supported ball, `index_bits` would assert inside
    // llvq-search's class enumeration — a corrupt field must stay an Err.
    let mut bytes = matrix_bytes(&small_matrix());
    // name length (4) + name (38) + d_out (4) + d_in (4) → shell cap.
    let at = 4 + small_matrix().name.len() + 4 + 4;
    bytes[at..at + 4].copy_from_slice(&0xFFFFu32.to_le_bytes());
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("shell cap 65535"), "detail: {detail}")
        }
        other => panic!("expected Inconsistent on the cap, got {:?}", other.err()),
    }
}

#[test]
fn a_lying_code_length_is_truncated_not_a_panic() {
    // The code-length field claims u64::MAX; the stream holds 12 bytes.
    let m = small_matrix();
    let mut bytes = matrix_bytes(&m);
    let code_bytes = (2 * 47u64).div_ceil(8) as usize; // 2 blocks × 47 bits
    let at = bytes.len() - code_bytes - 8;
    bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Truncated {
            reading: "code stream",
        }) => {}
        other => panic!("expected Truncated on the code stream, got {:?}", other.err()),
    }
}

#[test]
fn a_short_code_stream_is_truncated_not_a_panic() {
    // The stream is consistent with its own length field but holds fewer
    // bits than the dimensions promise — the case that used to blow the
    // `BitReader` assert instead of returning an error.
    let m = small_matrix();
    let mut bytes = matrix_bytes(&m);
    let code_bytes = (2 * 47u64).div_ceil(8) as usize;
    let at = bytes.len() - code_bytes - 8;
    bytes[at..at + 8].copy_from_slice(&1u64.to_le_bytes());
    bytes.truncate(at + 8 + 1); // one byte of codes for two 47-bit blocks
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Truncated {
            reading: "code stream",
        }) => {}
        other => panic!("expected Truncated on the code stream, got {:?}", other.err()),
    }
}

#[test]
fn an_index_too_wide_for_its_cap_is_refused_by_the_writer() {
    let mut m = small_matrix();
    m.indices[1] = 1u64 << 47; // one past the widest 47-bit index
    let mut out = Vec::new();
    match write_matrix_raw(&mut out, DEFAULT_VERSION, &m) {
        Err(Error::IndexTooWide { index, bits, .. }) => {
            assert_eq!(index, 1u64 << 47);
            assert_eq!(bits, 47);
        }
        other => panic!("expected IndexTooWide, got {:?}", other.err()),
    }
}

/// `N(12)`, the points of `Λ₂₄(12)` with the origin left out: the sum of the
/// theta series of Λ₂₄ over shells 2 to 12, the same sum that gives the
/// paper's `N(13)` one shell further.
const N12: u64 = 111_043_117_458_000;

/// The code stream of `small_matrix`'s shape: two 47-bit indices and no gain
/// bit, packed MSB-first as the writer packs them.
fn two_indices_at_cap_12(a: u64, b: u64) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.push(a, 47);
    bw.push(b, 47);
    bw.finish()
}

#[test]
fn a_ball_index_past_its_cap_is_refused_on_read() {
    // N(12) is the last point of Λ₂₄(12) and N(12) + 1 the first of shell 13.
    // Both fit 47 bits, so the width alone cannot tell the second from a
    // point of the codebook.
    let m = RawMatrix { indices: vec![N12, 5], ..small_matrix() };
    let mut bytes = matrix_bytes(&m);
    let back = read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION).expect("N(12) is in the cap-12 ball");
    assert_eq!(back.indices, m.indices);

    // The code stream is the last bytes of the record: rewrite its first index.
    let honest = two_indices_at_cap_12(N12, 5);
    let at = bytes.len() - honest.len();
    assert_eq!(bytes[at..], honest[..], "the code stream is not where this test looks");
    bytes[at..].copy_from_slice(&two_indices_at_cap_12(N12 + 1, 5));
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::IndexOutOfRange { name, index }) => {
            assert_eq!(name, m.name);
            assert_eq!(index, N12 + 1);
        }
        other => panic!("expected IndexOutOfRange, got {:?}", other.err()),
    }
    // And the raw writer does not put it in a file in the first place.
    let past = RawMatrix { indices: vec![N12 + 1, 5], ..small_matrix() };
    match write_matrix_raw(&mut Vec::new(), DEFAULT_VERSION, &past) {
        Err(Error::IndexOutOfRange { index, .. }) => assert_eq!(index, N12 + 1),
        other => panic!("raw writer: expected IndexOutOfRange, got {:?}", other.err()),
    }
    // Nor the point writer, handed the shell-13 point itself. The map pins
    // the constant: N(12) is on shell 12, N(12) + 1 on shell 13.
    let ix = llvq_search::index::Indexer::new();
    let last = ix.decode(N12).expect("N(12) decodes");
    let point = ix.decode(N12 + 1).expect("N(12) + 1 decodes");
    assert_eq!(llvq_core::Leech::shell_index(&last), Some(12));
    assert_eq!(llvq_core::Leech::shell_index(&point), Some(13));
    let cb = Codebook::new(CodeKind::Ball).expect("Ball has a map");
    let q = QuantizedMatrix {
        codes: vec![BlockCode { point, gain: 0 }; 4],
        ..tailed_points(32)
    };
    match write_matrix_with(&mut Vec::new(), DEFAULT_VERSION, &cb, &q) {
        Err(Error::IndexOutOfRange { index, .. }) => assert_eq!(index, N12 + 1),
        other => panic!("point writer: expected IndexOutOfRange, got {:?}", other.err()),
    }
}

#[test]
fn a_ball_record_of_cap_1_holds_the_origin_only() {
    // Λ₂₄(1) has no point, so the cap-1 codebook is the origin alone. Its
    // field is still one bit wide, and index 1 would decode to the first
    // point of shell 2.
    let m = RawMatrix { indices: vec![0, 0], shell_cap: 1, ..small_matrix() };
    let mut bytes = matrix_bytes(&m);
    read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION).expect("the origin is the cap-1 codebook");
    // Two 1-bit indices in the last byte, MSB-first: 0b1000_0000 is 1 then 0.
    let last = bytes.len() - 1;
    assert_eq!(bytes[last], 0, "the code stream is not where this test looks");
    bytes[last] = 0b1000_0000;
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::IndexOutOfRange { index, .. }) => assert_eq!(index, 1),
        other => panic!("expected IndexOutOfRange, got {:?}", other.err()),
    }
}

#[test]
fn an_unknown_raw_tensor_tag_is_refused_before_anything_else() {
    // Tag 99 (known: 0 = f16, 1 = quant), followed by garbage the reader
    // must never reach — past an unknown tag every byte means nothing.
    let mut bytes = 99u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0u8; 32]);
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::BadRawEncoding { tag: 99 }) => {}
        other => panic!("expected BadRawEncoding, got {:?}", other.err()),
    }
}

#[test]
fn a_lying_raw_tensor_length_cannot_abort_the_process() {
    // An f16 tensor claiming u64::MAX values: the byte size would overflow
    // u64 — data no file could hold, reported as Truncated rather than
    // wrapped into a small, silently wrong allocation.
    let mut bytes = 0u32.to_le_bytes().to_vec(); // tag f16
    bytes.extend_from_slice(&1u32.to_le_bytes()); // name length
    bytes.push(b't');
    bytes.extend_from_slice(&1u32.to_le_bytes()); // rank
    bytes.extend_from_slice(&4u64.to_le_bytes()); // dims[0]
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // value count
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::Truncated {
            reading: "raw tensor data",
        }) => {}
        other => panic!("expected Truncated on the data, got {:?}", other.err()),
    }
}

#[test]
fn a_lying_blob_length_cannot_abort_the_process() {
    let mut bytes = 1u32.to_le_bytes().to_vec(); // blob name length
    bytes.push(b'b');
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // blob length
    match read_blob(&mut &bytes[..]) {
        Err(Error::Truncated { reading: "blob" }) => {}
        other => panic!("expected Truncated on the blob, got {:?}", other.err()),
    }
}

#[test]
fn an_unknown_code_kind_is_refused_by_name() {
    // A v5 header whose kind tag names nothing this build knows: a newer
    // writer or a corrupted field, refused at the header — past an unknown
    // kind no record has a width.
    let mut good = Vec::new();
    write_header(&mut good, FIRST_KINDED_VERSION, 3).unwrap();
    assert_eq!(good.len(), 32, "magic + count + two fingerprints + kind + kinds present");
    // 3 and not 2: tag 2 is `Int4G128` since Q5's writer exists, and leaving
    // it here would leave a test whose name says "unknown" passing on a kind
    // this build knows.
    for tag in [RESERVED_INT4G128_TAG + 1, 7, u32::MAX] {
        let mut bytes = good.clone();
        bytes[24..28].copy_from_slice(&tag.to_le_bytes());
        match read_header(&mut &bytes[..]) {
            Err(Error::UnknownCodeKind { tag: t }) => assert_eq!(t, tag),
            other => panic!("tag {tag}: expected UnknownCodeKind, got {:?}", other.err()),
        }
    }
    // And every known tag reads back as itself, default and set both.
    for kind in CodeKind::ALL {
        let mut bytes = good.clone();
        bytes[24..28].copy_from_slice(&kind.tag().to_le_bytes());
        bytes[28..32].copy_from_slice(&KindSet::of(kind).bits().to_le_bytes());
        let head = read_header(&mut &bytes[..]).expect("a known kind");
        assert_eq!(head.default_kind(), kind);
        assert_eq!(head.kinds(), KindSet::of(kind));
    }
}

/// The `kinds_present` mask is refused bit by bit, and the reserved int4 bit
/// is one of them.
///
/// Masking an unknown bit off instead would be the silent failure this whole
/// suite exists against: a file with one int4 g128 matrix in it would present
/// as Ball-only and walk straight past `require_ball_kinds`.
#[test]
fn an_unknown_kind_in_the_present_mask_is_refused_by_name() {
    let mut good = Vec::new();
    write_header(&mut good, FIRST_KINDED_VERSION, 3).unwrap();
    // The lowest bit this build cannot name is now 3: bit 2 is `Int4G128`.
    const UNKNOWN_BIT: u32 = RESERVED_INT4G128_TAG + 1;
    for (mask, want) in [
        (1u32 << UNKNOWN_BIT, UNKNOWN_BIT),
        (1 | 1 << UNKNOWN_BIT, UNKNOWN_BIT),
        (1 << 31, 31),
        (u32::MAX, UNKNOWN_BIT),
    ] {
        let mut bytes = good.clone();
        bytes[28..32].copy_from_slice(&mask.to_le_bytes());
        match read_header(&mut &bytes[..]) {
            Err(Error::UnknownCodeKind { tag }) => {
                assert_eq!(tag, want, "mask {mask:#x}: the lowest unknown bit is named")
            }
            other => panic!("mask {mask:#x}: expected UnknownCodeKind, got {:?}", other.err()),
        }
    }
    // A mask that knows every bit but not the default is a header describing
    // no file: two fields that disagree, and which one is right decides how
    // every record is read.
    let mut bytes = good.clone();
    bytes[24..28].copy_from_slice(&CodeKind::Tetra.tag().to_le_bytes());
    match read_header(&mut &bytes[..]) {
        Err(Error::Inconsistent { name, detail }) => {
            assert_eq!(name, "header");
            assert!(detail.contains("Tetra"), "detail: {detail}");
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
    // And the writer refuses to produce one.
    assert!(matches!(
        write_header_kinds(
            &mut Vec::new(),
            FIRST_KINDED_VERSION,
            1,
            CodeKind::Tetra,
            KindSet::BALL
        ),
        Err(Error::Inconsistent { .. })
    ));
}

#[test]
fn a_v5_header_cut_short_is_truncated_at_the_named_field() {
    let mut bytes = Vec::new();
    write_header(&mut bytes, FIRST_KINDED_VERSION, 3).unwrap();
    for (len, field) in [
        (12usize, "codebook fingerprint"),
        (20, "tetra fingerprint"),
        (26, "code kind"),
        (30, "code kinds present"),
    ] {
        let mut cut = bytes.clone();
        cut.truncate(len);
        match read_header(&mut &cut[..]) {
            Err(Error::Truncated { reading }) => assert_eq!(reading, field, "cut at {len}"),
            other => panic!("cut at {len}: expected Truncated at the {field}, got {:?}", other.err()),
        }
    }
}

#[test]
fn a_wild_shell_cap_on_a_tetra_record_is_refused_not_a_panic() {
    // The Tetra reader takes its width from the kind, never from the field —
    // but the field is still checked, and a value that is not the sentinel
    // is a record this crate never wrote. Neither 65535 (past the ball,
    // where the Ball path would have asserted) nor 13 (a perfectly good
    // Ball cap, 48 bits wide) may be read.
    let m = small_tetra_matrix();
    let good = v5_matrix_bytes(&m);
    let at = shell_cap_at(&m);
    for cap in [0xFFFFu32, 13, 11, 0] {
        let mut bytes = good.clone();
        bytes[at..at + 4].copy_from_slice(&cap.to_le_bytes());
        match read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("shell cap {cap}")), "cap {cap}: detail {detail}");
                assert!(detail.contains("Tetra"), "cap {cap}: the refusal names the kind: {detail}");
            }
            other => panic!("cap {cap}: expected Inconsistent, got {:?}", other.err()),
        }
    }
    // The same bytes with the record's kind tag flipped to Ball are a cap-12
    // Ball record: the field means what it always meant on that path, and it
    // is the tag — not the cap — that decided which reading applies.
    let mut as_ball = good.clone();
    as_ball[kind_at(&m)..kind_at(&m) + 4].copy_from_slice(&CodeKind::Ball.tag().to_le_bytes());
    let back = read_matrix_raw(&mut &as_ball[..], FIRST_KINDED_VERSION)
        .expect("47-bit words read as a cap-12 Ball record");
    assert_eq!(back.kind, CodeKind::Ball);
    assert_eq!(back.indices, m.indices, "the width did not move: 47 bits either way");
}

/// A record's kind tag is refused on the same terms as the header's.
#[test]
fn an_unknown_kind_tag_on_a_record_is_refused_by_name() {
    let m = small_tetra_matrix();
    let good = v5_matrix_bytes(&m);
    let at = kind_at(&m);
    for tag in [RESERVED_INT4G128_TAG + 1, 9, u32::MAX] {
        let mut bytes = good.clone();
        bytes[at..at + 4].copy_from_slice(&tag.to_le_bytes());
        match read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION) {
            Err(Error::UnknownCodeKind { tag: t }) => assert_eq!(t, tag),
            other => panic!("tag {tag}: expected UnknownCodeKind, got {:?}", other.err()),
        }
    }
    // Tag 2 is a known kind now, and its refusal is a different one: the
    // lattice reader stops at `NotALatticeRecord` before it derives a width.
    // Left as an "unknown tag" case this would still fail, and for the wrong
    // reason.
    let mut bytes = good.clone();
    bytes[at..at + 4].copy_from_slice(&RESERVED_INT4G128_TAG.to_le_bytes());
    match read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION) {
        Err(Error::NotALatticeRecord { name, kind }) => {
            assert_eq!(name, m.name);
            assert_eq!(kind, CodeKind::Int4G128);
        }
        other => panic!("tag 2: expected NotALatticeRecord, got {:?}", other.err()),
    }
    // And read as the record it claims to be, it contradicts itself: a Tetra
    // record carries shell cap 12 where an int4 one carries u32::MAX.
    match llvq_artifact::read_record(&mut &bytes[..], FIRST_KINDED_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("shell cap"), "{detail}")
        }
        other => panic!("tag 2: expected Inconsistent, got {:?}", other.map(|_| "a record")),
    }
}

/// A v5 record read as v4 takes its kind tag for a centroid count, and a v4
/// record read as v5 takes its centroid count for a kind. Neither may pass.
///
/// This is `an_untagged_v2_raw_tensor_still_reads`' argument one field down:
/// a `version` argument that changed nothing would be dead code, and dead
/// code here silently misreads every record of the other version.
#[test]
fn a_v5_record_read_as_v4_is_not_the_same_record() {
    let m = small_tetra_matrix();
    let v5 = v5_matrix_bytes(&m);
    match read_matrix_raw(&mut &v5[..], DEFAULT_VERSION) {
        Err(_) => {}
        Ok(bad) => assert_ne!(
            (bad.centroids.len(), bad.kind),
            (m.centroids.len(), m.kind),
            "a v5 record read as v4 must not come back as itself"
        ),
    }
    // The Ball record of the same shape, the other way round: its centroid
    // count (1) is read as a kind tag (Tetra), and the widths that follow are
    // taken from the wrong map.
    let v4 = matrix_bytes(&small_matrix());
    match read_matrix_raw(&mut &v4[..], FIRST_KINDED_VERSION) {
        Err(_) => {}
        Ok(bad) => assert_ne!(
            (bad.centroids.len(), bad.kind),
            (1usize, CodeKind::Ball),
            "a v4 record read as v5 must not come back as itself"
        ),
    }
}

#[test]
fn a_tetra_record_with_a_wide_gain_field_is_refused_not_a_panic() {
    // Centroid count patched to 4: two gain bits, a 49-bit block. The Tetra
    // word has one gain bit at bit 47; refused before the (now short)
    // centroid list is even read.
    let m = small_tetra_matrix();
    let mut bytes = v5_matrix_bytes(&m);
    let at = kind_at(&m) + 4;
    for n in [4u32, 1, 3] {
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&n.to_le_bytes());
        match read_matrix_raw(&mut &b[..], FIRST_KINDED_VERSION) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("{n} centroids")), "{n}: detail {detail}")
            }
            other => panic!("{n} centroids: expected Inconsistent, got {:?}", other.err()),
        }
    }
    // The writer refuses the same record before a byte goes out.
    let wide = RawMatrix { centroids: vec![0.5, 0.7, 0.9, 1.1], ..small_tetra_matrix() };
    assert!(matches!(
        write_matrix_raw(&mut bytes, FIRST_KINDED_VERSION, &wide),
        Err(Error::Inconsistent { .. })
    ));
}

#[test]
fn a_hostile_tetra_record_still_round_trips_when_honest() {
    let m = small_tetra_matrix();
    let bytes = v5_matrix_bytes(&m);
    // A v5 Tetra record and a v5 Ball record of the same shape differ in
    // exactly four bytes — the kind tag — and nowhere else: the widths are
    // the same 47 + 1, which is what the sentinel cap is for.
    let ball = v5_matrix_bytes(&RawMatrix { kind: CodeKind::Ball, ..small_tetra_matrix() });
    let at = kind_at(&m);
    assert_eq!(bytes.len(), ball.len(), "the two kinds changed the record shape");
    assert_eq!(bytes[..at], ball[..at], "the record head before the tag moved");
    assert_eq!(bytes[at + 4..], ball[at + 4..], "the record body after the tag moved");
    assert_ne!(bytes[at..at + 4], ball[at..at + 4], "the tag is not written");

    let back = read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION).expect("an honest record must read");
    assert_eq!(back.kind, CodeKind::Tetra);
    assert_eq!(back.indices, m.indices);
    assert_eq!(back.gains, m.gains);
    assert_eq!(back.centroids, m.centroids);
    assert_eq!(back.shell_cap, TETRA_SHELL_CAP);
    // A label past 47 bits is refused on the Tetra path as on the Ball one.
    let wide = RawMatrix { indices: vec![3, 1u64 << 47], ..small_tetra_matrix() };
    match write_matrix_raw(&mut Vec::new(), FIRST_KINDED_VERSION, &wide) {
        Err(Error::IndexTooWide { index, bits, .. }) => assert_eq!((index, bits), (1u64 << 47, 47)),
        other => panic!("expected IndexTooWide, got {:?}", other.err()),
    }
    // And a Tetra record cannot be written into a v4 file at all: there is
    // nowhere to put its tag, and it would read back as Ball.
    match write_matrix_raw(&mut Vec::new(), DEFAULT_VERSION, &small_tetra_matrix()) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("Tetra"), "detail: {detail}");
            assert!(detail.contains("v5"), "detail: {detail}");
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn a_hostile_matrix_still_round_trips_when_honest() {
    // The hardening must not have bent the happy path: the same small
    // matrix, written and read back, field for field.
    let m = small_matrix();
    let bytes = matrix_bytes(&m);
    let back = read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION).expect("an honest file must read");
    assert_eq!(back.name, m.name);
    assert_eq!(back.kind, CodeKind::Ball, "a v4 record is a Ball record");
    assert_eq!(back.d_out, m.d_out);
    assert_eq!(back.d_in, m.d_in);
    assert_eq!(back.indices, m.indices);
    assert_eq!(back.gains, m.gains);
    assert_eq!(back.row_scales, m.row_scales);
    assert_eq!(back.centroids, m.centroids);
    assert_eq!(back.shell_cap, m.shell_cap);
    assert_eq!(back.rotation_seed, m.rotation_seed);
    assert!(back.tail.is_empty());
}

// ---------------------------------------------------------------------------
// Gain ranks and tails: what a lattice record's fields must agree on
// ---------------------------------------------------------------------------

/// Byte offset of a v4 record's centroid count: straight after the shell cap.
fn n_cent_at(m: &RawMatrix) -> usize {
    shell_cap_at(m) + 4
}

/// Byte offset of a v4 record's first centroid: after the count (4), the
/// rotation seed (8) and the rotation flag (4).
fn centroids_at(m: &RawMatrix) -> usize {
    n_cent_at(m) + 4 + 8 + 4
}

/// `bytes` with the record's centroid count set to `n` and the centroids from
/// rank `n` on cut out: a record that agrees with itself on every length.
fn with_centroids_cut_to(m: &RawMatrix, bytes: &[u8], n: usize) -> Vec<u8> {
    let (at, first) = (n_cent_at(m), centroids_at(m));
    let mut out = bytes[..at].to_vec();
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&bytes[at + 4..first + 8 * n]);
    out.extend_from_slice(&bytes[first + 8 * m.centroids.len()..]);
    out
}

#[test]
fn a_ball_gain_past_its_centroids_is_refused_on_read_not_a_panic() {
    // Written with four centroids, so the gain field is two bits wide and
    // gain 3 is a real rank. Cut to three, the field keeps its two bits
    // (ceil(log2 3) = 2) and gain 3 names no centroid: `decode_matrix` would
    // index past the list.
    let m = RawMatrix {
        gains: vec![3, 0],
        centroids: vec![0.5, 0.7, 0.9, 1.1],
        ..small_matrix()
    };
    let good = matrix_bytes(&m);
    let bytes = with_centroids_cut_to(&m, &good, 3);
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Inconsistent { name, detail }) => {
            assert_eq!(name, m.name);
            assert!(detail.contains("gain rank 3 with 3 centroids"), "detail: {detail}");
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
    // The four-centroid original is a valid record and still reads.
    let back = read_matrix_raw(&mut &good[..], DEFAULT_VERSION).expect("gain 3 of four centroids");
    assert_eq!(back.gains, m.gains);
    assert_eq!(back.centroids, m.centroids);
}

#[test]
fn a_ball_record_with_blocks_and_no_centroid_is_refused_on_read() {
    // One centroid is a zero-bit gain field; none leaves the same field and
    // nothing for rank 0 to name.
    let m = small_matrix();
    let bytes = with_centroids_cut_to(&m, &matrix_bytes(&m), 0);
    match read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("no gain centroid for 2 blocks"), "detail: {detail}")
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn a_gain_past_the_centroids_is_refused_by_both_writers() {
    // The raw writer: three centroids make a two-bit field that holds 3.
    let three = RawMatrix {
        gains: vec![0, 3],
        centroids: vec![0.5, 0.9, 1.3],
        ..small_matrix()
    };
    let mut out = Vec::new();
    match write_matrix_raw(&mut out, DEFAULT_VERSION, &three) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("gain rank 3 with 3 centroids"), "detail: {detail}")
        }
        other => panic!("three centroids: expected Inconsistent, got {:?}", other.err()),
    }
    assert!(out.is_empty(), "{} bytes written before the refusal", out.len());
    // No centroid: a zero-bit field, and rank 0 names nothing.
    let none = RawMatrix {
        centroids: vec![],
        ..small_matrix()
    };
    match write_matrix_raw(&mut Vec::new(), DEFAULT_VERSION, &none) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("gain rank 0 with 0 centroids"), "detail: {detail}")
        }
        other => panic!("no centroid: expected Inconsistent, got {:?}", other.err()),
    }
    // The point writer: an origin block with gain 2 under two centroids,
    // refused before the record head goes out.
    let cb = Codebook::new(CodeKind::Ball).expect("Ball has a map");
    let mut q = tailed_points(32);
    q.centroids = vec![0.7, 1.1];
    q.codes[3].gain = 2;
    let mut out = Vec::new();
    match write_matrix_with(&mut out, DEFAULT_VERSION, &cb, &q) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("gain rank 2 with 2 centroids"), "detail: {detail}")
        }
        other => panic!("point writer: expected Inconsistent, got {:?}", other.err()),
    }
    assert!(out.is_empty(), "{} bytes written before the refusal", out.len());
}

/// Two rows of two blocks and 16 tail columns, the `2560 mod 24` of the 4B:
/// 32 tail values is the count.
fn tailed_matrix(tail: usize) -> RawMatrix {
    RawMatrix {
        d_out: 2,
        d_in: 2 * DIM + 16,
        indices: vec![3, 5, 7, 9],
        gains: vec![0; 4],
        row_scales: vec![1.0, 1.0],
        tail: vec![0.25; tail],
        ..small_matrix()
    }
}

/// The same shape as points, for the writer that encodes them: four origin
/// blocks, one centroid.
fn tailed_points(tail: usize) -> QuantizedMatrix {
    QuantizedMatrix {
        name: "model.layers.0.mlp.up_proj.weight".into(),
        d_out: 2,
        d_in: 2 * DIM + 16,
        codes: vec![BlockCode { point: [0; DIM], gain: 0 }; 4],
        row_scales: vec![1.0, 1.0],
        centroids: vec![1.0],
        rotation_seed: None,
        shell_cap: 12,
        tail: vec![0.25; tail],
    }
}

#[test]
fn a_tail_count_off_the_dimensions_is_refused_by_both_writers() {
    let cb = Codebook::new(CodeKind::Ball).expect("Ball has a map");
    for n in [33usize, 31, 0] {
        let mut out = Vec::new();
        match write_matrix_raw(&mut out, DEFAULT_VERSION, &tailed_matrix(n)) {
            Err(Error::Inconsistent { detail, .. }) => assert!(
                detail.contains(&format!("{n} tail values")) && detail.contains("want 32"),
                "{n}, raw writer: {detail}"
            ),
            other => panic!("{n} tail values, raw writer: expected Inconsistent, got {:?}", other.err()),
        }
        assert!(out.is_empty(), "{n}: {} bytes written before the refusal", out.len());
        let mut out = Vec::new();
        match write_matrix_with(&mut out, DEFAULT_VERSION, &cb, &tailed_points(n)) {
            Err(Error::Inconsistent { detail, .. }) => assert!(
                detail.contains(&format!("{n} tail values")) && detail.contains("want 32"),
                "{n}, point writer: {detail}"
            ),
            other => panic!("{n} tail values, point writer: expected Inconsistent, got {:?}", other.err()),
        }
        assert!(out.is_empty(), "{n}: {} bytes written before the refusal", out.len());
    }
    // 32 is the count, and both writers take it.
    write_matrix_raw(&mut Vec::new(), DEFAULT_VERSION, &tailed_matrix(32)).expect("raw writer, 32");
    write_matrix_with(&mut Vec::new(), DEFAULT_VERSION, &cb, &tailed_points(32)).expect("point writer, 32");
}

#[test]
fn a_4b_shaped_record_and_its_tail_still_round_trip() {
    // 2560 = 24 · 106 + 16: the tail the served 4B carries on every row of
    // its d_in = 2560 projections.
    let (d_out, d_in) = (2usize, 2560usize);
    assert_eq!((d_in / DIM, d_in % DIM), (106, 16));
    let nblocks = d_out * (d_in / DIM);
    let m = RawMatrix {
        d_out,
        d_in,
        indices: (0..nblocks as u64).collect(),
        gains: vec![0; nblocks],
        row_scales: vec![1.0; d_out],
        tail: (0..d_out * 16).map(|i| i as f64 * 0.25).collect(),
        ..small_matrix()
    };
    let bytes = matrix_bytes(&m);
    let back = read_matrix_raw(&mut &bytes[..], DEFAULT_VERSION).expect("a 4B-shaped record must read");
    assert_eq!(back.indices, m.indices);
    assert_eq!(back.gains, m.gains);
    assert_eq!(back.tail, m.tail);
    // The point writer takes the same shape.
    let cb = Codebook::new(CodeKind::Ball).expect("Ball has a map");
    let q = QuantizedMatrix {
        d_out,
        d_in,
        codes: vec![BlockCode { point: [0; DIM], gain: 0 }; nblocks],
        row_scales: vec![1.0; d_out],
        tail: m.tail.clone(),
        ..tailed_points(0)
    };
    write_matrix_with(&mut Vec::new(), DEFAULT_VERSION, &cb, &q).expect("a 4B-shaped matrix must write");
}

// ---------------------------------------------------------------------------
// Raw tensors: read_raw checks what write_raw checks
// ---------------------------------------------------------------------------

/// A 4-bit raw tensor, two rows of five values in groups of two: the shape
/// whose short last group a wrong group count gets wrong.
fn small_quant_tensor() -> RawTensor {
    RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![2, 5],
        data: RawData::Quant(QuantData {
            bits: 4,
            group: 2,
            packed: vec![0x10, 0x32, 0x54, 0x76, 0x98],
            scales: vec![0x3C00; 6],
            biases: vec![0x0000; 6],
        }),
    }
}

fn raw_bytes(t: &RawTensor) -> Vec<u8> {
    let mut out = Vec::new();
    write_raw(&mut out, t).expect("a valid tensor must serialize");
    out
}

/// Byte offset of a v3 raw tensor's first dimension: tag (4), name length
/// (4), name, rank (4).
fn dims_at(t: &RawTensor) -> usize {
    4 + 4 + t.name.len() + 4
}

/// Byte offset of a v3 quant record's width field: straight after the dims.
fn quant_width_at(t: &RawTensor) -> usize {
    dims_at(t) + 8 * t.dims.len()
}

#[test]
fn a_raw_quant_width_is_checked_as_stored_not_narrowed() {
    // 0x104 narrowed to a byte is 4, and 2 is no width this format stores:
    // read as int4, both would decode plausible, wrong values.
    let t = small_quant_tensor();
    let good = raw_bytes(&t);
    let at = quant_width_at(&t);
    assert_eq!(good[at..at + 4], 4u32.to_le_bytes(), "the width field is not where this test looks");
    for width in [0x104u32, 2, 0] {
        let mut bytes = good.clone();
        bytes[at..at + 4].copy_from_slice(&width.to_le_bytes());
        match read_raw(&mut &bytes[..], 3) {
            Err(Error::Inconsistent { name, detail }) => {
                assert_eq!(name, t.name);
                assert!(detail.contains(&format!("unsupported quant width {width}")), "{width}: {detail}");
            }
            other => panic!("width {width}: expected Inconsistent, got {:?}", other.err()),
        }
    }
}

#[test]
fn a_raw_quant_group_of_zero_is_refused_not_a_division_by_zero() {
    let t = small_quant_tensor();
    let mut bytes = raw_bytes(&t);
    let at = quant_width_at(&t) + 4;
    assert_eq!(bytes[at..at + 4], 2u32.to_le_bytes(), "the group field is not where this test looks");
    bytes[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::Inconsistent { detail, .. }) => assert!(detail.contains("quant group 0"), "detail: {detail}"),
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
    // The writer refuses the same tensor instead of dividing by zero.
    let mut zero = small_quant_tensor();
    if let RawData::Quant(q) = &mut zero.data {
        q.group = 0;
    }
    match write_raw(&mut Vec::new(), &zero) {
        Err(Error::Inconsistent { detail, .. }) => assert!(detail.contains("quant group 0"), "detail: {detail}"),
        other => panic!("writer: expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn a_raw_quant_payload_short_of_its_dims_is_refused_not_a_panic() {
    // One packed byte fewer, and the length field saying so: a record that
    // agrees with itself and holds 8 of the 10 values its dims name.
    let t = small_quant_tensor();
    let good = raw_bytes(&t);
    let len_at = quant_width_at(&t) + 4 + 4;
    let packed = 5usize;
    assert_eq!(good[len_at..len_at + 8], (packed as u64).to_le_bytes(), "the packed length is not where this test looks");
    let mut bytes = good[..len_at].to_vec();
    bytes.extend_from_slice(&(packed as u64 - 1).to_le_bytes());
    bytes.extend_from_slice(&good[len_at + 8..len_at + 8 + packed - 1]);
    bytes.extend_from_slice(&good[len_at + 8 + packed..]);
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("4 packed bytes") && detail.contains("want 5"), "detail: {detail}")
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn raw_dims_whose_product_overflows_are_refused_by_name() {
    // 2^40 × 2^40 values: no usize holds the count, and a wrapped product
    // would be compared against the payload as if it were one.
    let t = small_quant_tensor();
    let mut bytes = raw_bytes(&t);
    let at = dims_at(&t);
    for k in 0..t.dims.len() {
        bytes[at + 8 * k..at + 8 * k + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    }
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("more values than a usize holds"), "detail: {detail}")
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn a_raw_f16_count_that_disagrees_with_its_dims_is_refused() {
    let t = RawTensor {
        name: "model.norm.weight".into(),
        dims: vec![2, 3],
        data: RawData::F16(vec![0x3C00; 6]),
    };
    let mut bytes = raw_bytes(&t);
    let at = dims_at(&t) + 8; // dims[1]
    bytes[at..at + 8].copy_from_slice(&4u64.to_le_bytes());
    match read_raw(&mut &bytes[..], 3) {
        Err(Error::Inconsistent { detail, .. }) => {
            assert!(detail.contains("6 values for dims [2, 4]"), "detail: {detail}")
        }
        other => panic!("expected Inconsistent, got {:?}", other.err()),
    }
}

#[test]
fn honest_raw_tensors_still_read_byte_for_byte() {
    // int4 with a short last group, int8 with one group per row, f16, and a
    // rank-0 scalar: each reads back, decodes the same, and rewrites to the
    // same bytes.
    let q8 = RawTensor {
        name: "lm_head.weight".into(),
        dims: vec![2, 3],
        data: RawData::Quant(QuantData {
            bits: 8,
            group: 64,
            packed: vec![0, 1, 2, 3, 4, 5],
            scales: vec![0x3C00; 2],
            biases: vec![0x0000; 2],
        }),
    };
    let f16 = RawTensor {
        name: "model.norm.weight".into(),
        dims: vec![3],
        data: RawData::F16(vec![0x3C00, 0xC000, 0x7BFF]),
    };
    let scalar = RawTensor {
        name: "s".into(),
        dims: vec![],
        data: RawData::F16(vec![0x3C00]),
    };
    for t in [small_quant_tensor(), q8, f16, scalar] {
        let bytes = raw_bytes(&t);
        let back = read_raw(&mut &bytes[..], 3).expect("an honest tensor must read");
        assert_eq!(back.name, t.name);
        assert_eq!(back.dims, t.dims);
        assert_eq!(back.to_f32(), t.to_f32(), "{}", t.name);
        assert_eq!(raw_bytes(&back), bytes, "{}: the rewrite moved a byte", t.name);
    }
}

// ---------------------------------------------------------------------------
// The rotation a record's d_in sizes
// ---------------------------------------------------------------------------

/// A rotated Tetra record with no rows: nothing in the code stream backs its
/// `d_in`, so the field is the only thing that sizes the rotation a decoder
/// builds from it.
fn rotated_empty(d_in: usize) -> RawMatrix {
    RawMatrix {
        name: "model.layers.0.mlp.down_proj.weight".into(),
        d_out: 0,
        d_in,
        indices: vec![],
        gains: vec![],
        row_scales: vec![],
        rotation_seed: Some(7),
        tail: vec![],
        ..small_tetra_matrix()
    }
}

/// Byte offset of the d_in field: name length (4) + name + d_out (4).
fn d_in_at(m: &RawMatrix) -> usize {
    4 + m.name.len() + 4
}

#[test]
fn a_rotated_d_in_no_decoder_can_build_is_refused_on_read() {
    // Written honestly at d_in = 48, then the field is rewritten. The three
    // first values are the audit's: a panic in `Rotation::new`, an 8 TB
    // Gram-Schmidt, 74 s for zero rows. The last is a power of two whose sign
    // table alone would be 16 GB.
    let m = rotated_empty(48);
    let good = v5_matrix_bytes(&m);
    let at = d_in_at(&m);
    for d_in in [0u32, 1_000_001, 2001, 1 << 31] {
        let mut bytes = good.clone();
        bytes[at..at + 4].copy_from_slice(&d_in.to_le_bytes());
        match read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("d_in {d_in}")), "{d_in}: {detail}")
            }
            other => panic!("d_in {d_in}: expected Inconsistent, got {:?}", other.err()),
        }
    }
    // The same record unrotated reads: the bound is the rotation's, not the
    // record's.
    let plain = v5_matrix_bytes(&RawMatrix {
        rotation_seed: None,
        ..rotated_empty(48)
    });
    for d_in in [0u32, 2001] {
        let mut bytes = plain.clone();
        bytes[at..at + 4].copy_from_slice(&d_in.to_le_bytes());
        read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION).expect("no rotation, nothing to size");
    }
}

#[test]
fn a_rotated_d_in_no_decoder_can_build_is_refused_by_both_writers() {
    let cb = Codebook::new(CodeKind::Tetra).expect("Tetra has a map");
    for d_in in [0usize, 2001, 1 << 21] {
        let mut out = Vec::new();
        match write_matrix_raw(&mut out, FIRST_KINDED_VERSION, &rotated_empty(d_in)) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("d_in {d_in}")), "{d_in}, raw writer: {detail}")
            }
            other => panic!("d_in {d_in}, raw writer: expected Inconsistent, got {:?}", other.err()),
        }
        assert!(out.is_empty(), "{d_in}: {} bytes written before the refusal", out.len());
        let q = QuantizedMatrix {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            d_out: 0,
            d_in,
            codes: vec![],
            row_scales: vec![],
            centroids: vec![0.7, 1.1],
            rotation_seed: Some(7),
            shell_cap: TETRA_SHELL_CAP,
            tail: vec![],
        };
        let mut out = Vec::new();
        match write_matrix_with(&mut out, FIRST_KINDED_VERSION, &cb, &q) {
            Err(Error::Inconsistent { detail, .. }) => {
                assert!(detail.contains(&format!("d_in {d_in}")), "{d_in}, point writer: {detail}")
            }
            other => panic!("d_in {d_in}, point writer: expected Inconsistent, got {:?}", other.err()),
        }
        assert!(out.is_empty(), "{d_in}: {} bytes written before the refusal", out.len());
    }
}

#[test]
fn every_width_the_product_rotates_still_round_trips() {
    // Qwen3 0.6B, 4B, 8B, 14B and 32B hidden, attention-output and MLP widths
    // (odd factors 1, 3, 5, 17, 19, 25), Inkling 2 Mini's 1152 and 2304 (9),
    // and the bound's own edges: odd factor 255 and the widest power of two.
    for d_in in [
        1024usize, 3072, 2560, 9728, 4096, 12288, 5120, 17408, 8192, 25600, 1152, 2304,
        255 * 128,
        1 << 20,
    ] {
        let m = rotated_empty(d_in);
        let bytes = v5_matrix_bytes(&m);
        let back = read_matrix_raw(&mut &bytes[..], FIRST_KINDED_VERSION).expect("a real width must read");
        assert_eq!((back.d_in, back.rotation_seed), (d_in, Some(7)));
    }
}

#[test]
fn a_matrix_with_no_rows_decodes_without_building_a_rotation() {
    // In memory, past every reader: `Rotation::new(0, _)` asserts, and with no
    // row there is nothing for a rotation to act on.
    let q = QuantizedMatrix {
        name: "model.layers.0.mlp.down_proj.weight".into(),
        d_out: 0,
        d_in: 0,
        codes: vec![],
        row_scales: vec![],
        centroids: vec![0.7, 1.1],
        rotation_seed: Some(7),
        shell_cap: TETRA_SHELL_CAP,
        tail: vec![],
    };
    assert!(llvq_artifact::decode_matrix(&q).is_empty());
}
