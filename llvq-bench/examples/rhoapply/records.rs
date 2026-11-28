//! The byte-level helpers every walk of this bench shares: the record reader
//! that skips int4 records, and the self-tested SHA-256 that hashes the files.

use super::*;

/// Read one record of a v5 file, returning `None` for an int4 g128 one.
///
/// ## Why every walk in this file needs it
///
/// The served 4B is **mixed**: 216 `Tetra` records and 36 `v_proj` in int4
/// g128 (`configs/qwen3-4b-tetra-q5.json`). `read_matrix_raw` refuses an int4
/// record by name — `Error::NotALatticeRecord`, `format.rs` — so before this
/// helper existed every walk here died at the third record of the served file
/// and left a truncated output behind it. `read_record` is the only entry
/// point that can walk a mixed file.
///
/// `None` and not an error, because an int4 record is not a failure and not a
/// lattice record either: it is group-affine in the **natural** basis, it has
/// no `row_scales`, and there is nothing for ρ to multiply. That is also the
/// safety property L01 asks for — `v_proj` is never scaled, and it is never
/// scaled *by construction* rather than by a name check that a renamed
/// projection would defeat.
pub(super) fn next_raw(r: &mut impl Read, version: u32) -> Result<Option<RawMatrix>, String> {
    match llvq_artifact::read_record(r, version).map_err(|e| e.to_string())? {
        llvq_artifact::Record::Lattice(m) => Ok(Some(m)),
        llvq_artifact::Record::Int4(_) => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// SHA-256, because a core crate has no dependencies
// ---------------------------------------------------------------------------

/// Self-test of the workspace's SHA-256 against published vectors at startup:
/// a hash function that is wrong in the same way on both files would report a
/// false idempotence, which is exactly the control it serves.
pub(super) fn sha256_selftest() {
    let mut a = Sha256::new();
    a.update(b"");
    assert_eq!(
        a.finish(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let mut b = Sha256::new();
    b.update(b"abc");
    assert_eq!(
        b.finish(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A message longer than one block, fed in awkward pieces, to exercise the
    // buffering path that a one-shot vector never reaches.
    let mut c = Sha256::new();
    for chunk in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".chunks(7) {
        c.update(chunk);
    }
    assert_eq!(
        c.finish(),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

pub(super) fn sha256_file(path: &Path) -> std::io::Result<(String, u64)> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((h.finish(), n))
}

/// A sink that hashes and counts on the way through, so a rewrite pays one
/// pass instead of two.
pub(super) struct HashWriter<W: Write> {
    pub(super) inner: W,
    pub(super) hash: Sha256,
    pub(super) bytes: u64,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
