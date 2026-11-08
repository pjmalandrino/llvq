//! SHA-256 digests of a sealed artifact's fields.
//!
//! The hash itself is `llvq_artifact::sha256`, the workspace's one
//! implementation, re-exported here; this module adds the typed helpers.
//!
//! ## What it is for
//!
//! One number per field of a sealed artifact, so an independent reader can
//! claim it rebuilt that field bit for bit. The convention the hashes are
//! taken over is a format fact and is written down once, in
//! [`crate::hfpack`]: little-endian bit patterns, in the order the field
//! stores them.

pub use llvq_artifact::sha256::{sha256_file, sha256_hex, Sha256};

/// SHA-256 over the little-endian bit patterns of `v`, 8 bytes each.
pub fn sha256_f64(v: &[f64]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 8);
        for x in chunk {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}

/// SHA-256 over the little-endian bit patterns of `v` narrowed to f32.
///
/// The artifact widens a stored f32 tail to f64 on read; narrowing it back is
/// exact, and hashing the f32 patterns is what lets the check compare against
/// the four bytes the file holds.
pub fn sha256_f64_as_f32(v: &[f64]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for x in chunk {
            bytes.extend_from_slice(&(*x as f32).to_bits().to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}

/// SHA-256 over the little-endian bit patterns of `v`, 2 bytes each.
pub fn sha256_u16(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 2);
        for x in chunk {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}

/// SHA-256 over `v` little-endian, 4 bytes each.
pub fn sha256_u32(v: &[u32]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for x in chunk {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}

/// SHA-256 over `v` little-endian, 8 bytes each.
pub fn sha256_u64(v: &[u64]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 8);
        for x in chunk {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The typed helpers are the byte convention of the format, so they are
    /// pinned against it by hand rather than against themselves.
    #[test]
    fn the_typed_helpers_hash_little_endian_patterns() {
        assert_eq!(sha256_f64(&[1.0f64]), sha256_hex(&1.0f64.to_bits().to_le_bytes()));
        assert_eq!(sha256_u16(&[0x3C00]), sha256_hex(&[0x00, 0x3C]));
        assert_eq!(sha256_u32(&[1]), sha256_hex(&[1, 0, 0, 0]));
        assert_eq!(sha256_u64(&[1]), sha256_hex(&[1, 0, 0, 0, 0, 0, 0, 0]));
        // A tail read as f64 hashes as the f32 the file stores.
        let x = 0.5f32;
        assert_eq!(
            sha256_f64_as_f32(&[x as f64]),
            sha256_hex(&x.to_bits().to_le_bytes())
        );
        // Chunking inside the helpers must not show either.
        let v: Vec<f64> = (0..200_000).map(|i| i as f64 * 0.25).collect();
        let mut h = Sha256::new();
        for x in &v {
            h.update(&x.to_bits().to_le_bytes());
        }
        assert_eq!(sha256_f64(&v), h.finish());
    }
}
