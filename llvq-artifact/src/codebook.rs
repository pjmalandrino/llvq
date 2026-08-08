//! A fingerprint of the v1 index mapping — the one thing a `.llvq` file
//! cannot carry and cannot do without.
//!
//! ## The hole this closes
//!
//! A matrix record stores its shell cap and its centroid count, so the *width*
//! of an index and the *width* of a gain are both in the file. What is not in
//! the file, and cannot be, is what an index **means**: the Golay generator and
//! codeword order, the class enumeration order, and the mixed-radix
//! composition inside a class (§4 of the project notes, and the stability
//! contract on [`llvq_search::index`]). Those live in code.
//!
//! So a reader whose codebook disagrees with the writer's does not fail. Every
//! index is still in range, every point it decodes to is still a lattice point,
//! every row scale still applies — the file opens, the model runs, and the
//! logits are plausible and wrong. That is the failure mode this format has to
//! refuse, and until now nothing did.
//!
//! ## What is fingerprinted, and why it is the map rather than its inputs
//!
//! Hashing the parameters that *determine* the map would only cover the ones
//! we remembered to list; the composition order, for one, is control flow, not
//! data. So [`codebook_fingerprint`] digests the **mapping itself**: the class
//! layout as [`FastDecoder`] reports it, plus [`Indexer::decode`] evaluated at
//! deterministic probe positions inside every class. Change the Golay order,
//! reorder the classes, swap two mixed-radix digits, move a shell boundary —
//! the probes move, and the fingerprint with them.
//!
//! The whole Golay codeword table is folded in as well. A single transposition
//! deep inside one weight bucket could, in principle, be missed by every probe;
//! the table catches it exactly, and costs 4096 words to hash.
//!
//! ## What it buys, and what it does not
//!
//! It buys two different things, and only one of them involves a file:
//!
//! * **Pinned in the test suite** (`the_index_map_is_the_published_one`), it
//!   makes §4's stability contract machine-checked. Any change to the index
//!   layout fails a test *before* a file is written or read — which is the
//!   only protection the artifacts already published can have, since they were
//!   written before this field existed.
//! * **Stored in the file** from `LVQ4` on, it catches the case the test
//!   cannot see: a file written by a *different build* — a fork, an older
//!   checkout, a third-party writer.
//!
//! It is a consistency check, not a checksum: it says nothing about whether
//! the payload bytes are intact, and it is not meant to.

use llvq_core::{Golay, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};
use llvq_search::index::{Indexer, N13};
use std::sync::OnceLock;

/// Domain tag — so this digest can never collide with another use of FNV in
/// the tree, and so that changing what is hashed is itself a visible change.
const DOMAIN: &[u8] = b"llvq codebook fingerprint v1";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Probe positions inside a class, as fractions of its span. The first digit
/// of the mixed-radix composition is the most significant one, so fractions
/// spread the probes across codewords; the class's last index maxes every
/// digit at once.
const PROBE_FRACTIONS: [(u64, u64); 4] = [(0, 4), (1, 4), (2, 4), (3, 4)];

/// Extra probes at scrambled positions, which is what actually exercises the
/// *low* digits — arrangement ranks and sign patterns — that the fractions
/// above barely move.
const SCRAMBLED_PROBES: u64 = 2;

/// FNV-1a, 64 bits. Hand-rolled: this crate has no dependencies, and a reader
/// auditing the format should not have to take a hash crate on faith either.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn byte(&mut self, b: u8) {
        self.0 ^= u64::from(b);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    fn bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.byte(b);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    fn i32(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }
}

/// Deterministic 64-bit mixer for the scrambled probe positions.
///
/// Written out here rather than taken from [`llvq_core::SplitMix64`] on
/// purpose: the fingerprint must be a function of the codebook and of nothing
/// else, and borrowing a general-purpose RNG would make a change to *it* look
/// like a change to the index format.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Fold one index and the point it decodes to into the digest.
fn probe(h: &mut Fnv, ix: &Indexer, idx: u64) {
    h.u64(idx);
    match ix.decode(idx) {
        Some(p) => {
            h.byte(1);
            for v in p {
                h.i32(v);
            }
        }
        // An index the codebook refuses is a fact about the map too, and one
        // that moves if a shell boundary does.
        None => h.byte(0),
    }
}

fn compute() -> u64 {
    let mut h = Fnv::new();
    h.bytes(DOMAIN);
    h.u64(DIM as u64);
    h.u64(N13);
    h.u64(u64::from(llvq_core::golay::GEN_POLY));

    let golay = Golay::new();
    let words = golay.codewords();
    h.u64(words.len() as u64);
    for &w in words {
        h.u64(u64::from(w));
    }

    let fd = FastDecoder::new();
    let ix = Indexer::new();
    h.u64(fd.n_classes() as u64);
    for ci in 0..fd.n_classes() {
        let (lo, hi) = fd.class_range(ci);
        h.u64(lo);
        h.u64(hi);

        let lv = fd.levels(ci);
        h.u64(lv.len as u64);
        h.u64(u64::from(lv.nonzero));
        h.u64(u64::from(lv.odd));
        h.u64(u64::from(lv.shell));
        for k in 0..MAX_LEVELS {
            h.i32(lv.values[k]);
            h.byte(lv.counts[k]);
        }

        let card = hi - lo + 1;
        for (num, den) in PROBE_FRACTIONS {
            probe(&mut h, &ix, lo + card * num / den);
        }
        probe(&mut h, &ix, hi);
        for k in 0..SCRAMBLED_PROBES {
            probe(&mut h, &ix, lo + mix(ci as u64 ^ mix(k)) % card);
        }
    }

    // The origin, and the first index past the ball: the two edges every
    // reader has to agree on before any class matters.
    probe(&mut h, &ix, 0);
    probe(&mut h, &ix, N13 + 1);
    h.0
}

/// Fingerprint of the codebook **this build** uses to interpret indices.
///
/// Computed once per process (a few milliseconds: one [`Indexer`], one
/// [`FastDecoder`], and roughly seven decodes per class), then cached.
pub fn codebook_fingerprint() -> u64 {
    static FINGERPRINT: OnceLock<u64> = OnceLock::new();
    *FINGERPRINT.get_or_init(compute)
}
