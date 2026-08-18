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
    read_blob, read_header, read_matrix_raw, read_raw, write_header, write_matrix_raw, Error,
    RawMatrix, VERSION,
};

/// A minimal valid matrix: one row, two 24-blocks, cap 12 (47-bit indices),
/// one centroid (0 gain bits), no tail.
fn small_matrix() -> RawMatrix {
    RawMatrix {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        d_out: 1,
        d_in: 48,
        indices: vec![3, 5],
        gains: vec![0, 0],
        row_scales: vec![1.0],
        centroids: vec![1.0],
        rotation_seed: None,
        shell_cap: 12,
        tail: vec![],
    }
}

fn matrix_bytes(m: &RawMatrix) -> Vec<u8> {
    let mut out = Vec::new();
    write_matrix_raw(&mut out, m).expect("a valid matrix must serialize");
    out
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
    match read_matrix_raw(&mut &bytes[..]) {
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
    match read_matrix_raw(&mut &bytes[..]) {
        Err(Error::Truncated { reading: "name" }) => {}
        other => panic!("expected Truncated, got {:?}", other.err()),
    }
}

#[test]
fn a_bad_utf8_name_is_bad_name() {
    let mut bytes = 2u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0xFF, 0xFE]);
    match read_matrix_raw(&mut &bytes[..]) {
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
    match read_matrix_raw(&mut &bytes[..]) {
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
    match read_matrix_raw(&mut &bytes[..]) {
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
    match read_matrix_raw(&mut &bytes[..]) {
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
    match write_matrix_raw(&mut out, &m) {
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
fn a_hostile_matrix_still_round_trips_when_honest() {
    // The hardening must not have bent the happy path: the same small
    // matrix, written and read back, field for field.
    let m = small_matrix();
    let bytes = matrix_bytes(&m);
    let back = read_matrix_raw(&mut &bytes[..]).expect("an honest file must read");
    assert_eq!(back.name, m.name);
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
