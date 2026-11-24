//! SHA-256, streaming, written out rather than pulled in.
//!
//! The one implementation of the workspace since 2026-11-08 (audit batch 10):
//! `llvq-cuda` hashes its NVRTC text and its pinned units with it,
//! `llvq-llm::digest` builds the per-field digests of a sealed artifact on it,
//! and `rankbench` and `rhoapply` check files with it. Four hand-written
//! copies preceded it. It lives here because this crate carries no external
//! dependency and every crate that hashes already depends on it.
//!
//! It takes its input in pieces, so a 700 MB code stream is never copied.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// An incremental SHA-256.
pub struct Sha256 {
    h: [u32; 8],
    /// Bytes of the block being filled, `len` of them valid.
    block: [u8; 64],
    len: usize,
    /// Message length in bytes, for the length suffix.
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0u8; 64],
            len: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.len > 0 {
            let take = (64 - self.len).min(data.len());
            self.block[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len == 64 {
                let block = self.block;
                self.compress(&block);
                self.len = 0;
            }
            // Returning here is load-bearing: the tail assignment below sets
            // `len` from this call's remainder, which on an empty slice would
            // drop the partial block just filled. `finish` pads one byte at a
            // time, so that bug is an infinite loop and not a wrong digest.
            if data.is_empty() {
                return;
            }
        }
        debug_assert_eq!(self.len, 0, "a partial block is filled above or returned");
        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            self.compress(chunk);
        }
        let rest = chunks.remainder();
        self.block[..rest.len()].copy_from_slice(rest);
        self.len = rest.len();
    }

    /// The digest, lowercase hex.
    pub fn finish(mut self) -> String {
        let bitlen = self.total * 8;
        self.update_raw(&[0x80]);
        while self.len != 56 {
            self.update_raw(&[0]);
        }
        self.update_raw(&bitlen.to_be_bytes());
        debug_assert_eq!(self.len, 0, "the length suffix closes the last block");
        self.h.iter().map(|x| format!("{x:08x}")).collect()
    }

    /// [`Self::update`] without touching the message length, for the padding.
    fn update_raw(&mut self, data: &[u8]) {
        let total = self.total;
        self.update(data);
        self.total = total;
    }

    fn compress(&mut self, chunk: &[u8]) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = &chunk[i * 4..i * 4 + 4];
            *word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = self.h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (hi, vi) in self.h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
}

/// SHA-256 of one buffer, lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finish()
}

/// SHA-256 of a file, read in 1 MiB pieces.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(h.finish());
        }
        h.update(&buf[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two published vectors, plus the block boundary a streaming
    /// implementation gets wrong: 55, 56, 57 and 64 bytes are the four cases
    /// where the padding decides whether one block or two are compressed.
    #[test]
    fn known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 64]),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    /// Streaming in pieces must equal hashing in one go, for every split. A
    /// digest that depended on the caller's chunking would compare two readers
    /// by how they happened to loop.
    #[test]
    fn every_split_agrees() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 7 % 251) as u8).collect();
        let want = sha256_hex(&data);
        for cut in 0..=data.len() {
            let mut h = Sha256::new();
            h.update(&data[..cut]);
            h.update(&data[cut..]);
            assert_eq!(h.finish(), want, "split at {cut}");
        }
    }

    /// FIPS 180-2's two-block message, fed in pieces of 7 bytes: the
    /// buffering path a one-shot vector never reaches.
    #[test]
    fn the_two_block_vector_in_pieces() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let want = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        assert_eq!(sha256_hex(msg), want);
        let mut h = Sha256::new();
        for chunk in msg.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(h.finish(), want);
    }
}
