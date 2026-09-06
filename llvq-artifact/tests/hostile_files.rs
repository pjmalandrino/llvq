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
    write_matrix_raw, CodeKind, Error, KindSet, RawMatrix, DEFAULT_VERSION, FIRST_KINDED_VERSION,
    RESERVED_INT4G128_TAG, TETRA_SHELL_CAP, VERSION,
};

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
