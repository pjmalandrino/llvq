//! The 48-bit Tetra word: ten fields, cut apart and joined back.
//!
//! Bit 0 is the least significant bit of the `u64`. A block is six bytes and
//! the word is their little-endian reading — which is NOT the order the
//! `.llvq` stream of `pack.rs` writes (most-significant bit first); the
//! transcoder between the two is step 3's, and it is what a v5 fingerprint
//! hashes. Nothing here knows about bytes.
//!
//! ```text
//!   bit 0        p      parity shared by the 24 coordinates
//!   bit 1        r      k-parity class of section 1
//!   bits 2..7    s8     Golay state at cut 8 (0..63)
//!   bit 8        b1     which of the two prefix bytes of s8
//!   bits 9..19   i1     row of section 1 in class r             (11 bits)
//!   bits 20..23  b2     branch out of s8 (0..15)
//!   bits 24..34  i2     row of section 2 in the mixed order     (11 bits)
//!   bit 35       b3     which of the two suffix bytes of s16
//!   bits 36..46  i3     row of section 3 in class p ⊕ r ⊕ δ     (11 bits)
//!   bit 47       gain   the gain bit: carried by the word, never decoded
//! ```
//!
//! Every width is a power of two and the fields tile bits 0..48 exactly, so
//! any 48-bit word is a label: the decode is a bijection from the 2⁴⁷ labels
//! onto 2⁴⁷ lattice points, with the gain bit alongside. [`Fields::join`]
//! refuses a field that does not fit its width rather than truncating it — a
//! truncated index is a valid index for a different point.

use super::{LABEL_BITS, WORD_BITS};

/// A field: `(low bit, width)`.
pub type Slot = (u32, u32);

pub const P: Slot = (0, 1);
pub const R: Slot = (1, 1);
pub const S8: Slot = (2, 6);
pub const B1: Slot = (8, 1);
pub const I1: Slot = (9, 11);
pub const B2: Slot = (20, 4);
pub const I2: Slot = (24, 11);
pub const B3: Slot = (35, 1);
pub const I3: Slot = (36, 11);
pub const GAIN: Slot = (47, 1);

/// The ten fields in the order they are laid, for tests and tools; the
/// slots above are the single source [`Fields`] reads and writes through.
pub const LAYOUT: [(&str, Slot); 10] = [
    ("p", P),
    ("r", R),
    ("s8", S8),
    ("b1", B1),
    ("i1", I1),
    ("b2", B2),
    ("i2", I2),
    ("b3", B3),
    ("i3", I3),
    ("gain", GAIN),
];

/// Bits 0..47 of a word: the label, the gain bit masked off.
pub const LABEL_MASK: u64 = (1 << LABEL_BITS) - 1;

/// Bits 0..48 of a word: what a six-byte block holds.
pub const WORD_MASK: u64 = (1 << WORD_BITS) - 1;

#[inline]
fn get(word: u64, (lo, width): Slot) -> u64 {
    (word >> lo) & ((1u64 << width) - 1)
}

#[inline]
fn put(name: &str, (lo, width): Slot, value: u64) -> u64 {
    assert!(value >> width == 0, "field {name} = {value} does not fit in {width} bits");
    value << lo
}

/// The ten fields of a word, each in the narrowest unsigned type that holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fields {
    pub p: u8,
    pub r: u8,
    pub s8: u8,
    pub b1: u8,
    pub i1: u16,
    pub b2: u8,
    pub i2: u16,
    pub b3: u8,
    pub i3: u16,
    pub gain: u8,
}

impl Fields {
    /// Cut a word into its fields. Bits at or above 48 are ignored.
    pub fn split(word: u64) -> Self {
        Self {
            p: get(word, P) as u8,
            r: get(word, R) as u8,
            s8: get(word, S8) as u8,
            b1: get(word, B1) as u8,
            i1: get(word, I1) as u16,
            b2: get(word, B2) as u8,
            i2: get(word, I2) as u16,
            b3: get(word, B3) as u8,
            i3: get(word, I3) as u16,
            gain: get(word, GAIN) as u8,
        }
    }

    /// The word. Panics on a field wider than its slot.
    pub fn join(&self) -> u64 {
        put("p", P, self.p as u64)
            | put("r", R, self.r as u64)
            | put("s8", S8, self.s8 as u64)
            | put("b1", B1, self.b1 as u64)
            | put("i1", I1, self.i1 as u64)
            | put("b2", B2, self.b2 as u64)
            | put("i2", I2, self.i2 as u64)
            | put("b3", B3, self.b3 as u64)
            | put("i3", I3, self.i3 as u64)
            | put("gain", GAIN, self.gain as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_core::SplitMix64;

    /// The fields tile bits 0..48 once each, in order; the label is the low 47.
    #[test]
    fn the_layout_tiles_forty_eight_bits_in_order() {
        let mut next = 0u32;
        for (name, (lo, width)) in LAYOUT {
            assert_eq!(lo, next, "{name} does not start where the previous field ends");
            assert!(width > 0, "{name} is empty");
            next = lo + width;
        }
        assert_eq!(next, WORD_BITS);
        assert_eq!(GAIN.0, LABEL_BITS, "the gain bit is not the bit after the label");
        assert_eq!(LABEL_MASK, WORD_MASK >> 1);
        let f = Fields::split(u64::MAX);
        assert_eq!((f.s8, f.i1, f.b2, f.i2, f.i3), (63, 2047, 15, 2047, 2047), "field widths");
        assert_eq!((f.p, f.r, f.b1, f.b3, f.gain), (1, 1, 1, 1, 1));
    }

    /// `join ∘ split` is the identity on the low 48 bits and drops the rest;
    /// `split ∘ join` is the identity on fields.
    #[test]
    fn split_and_join_are_inverse() {
        let mut rng = SplitMix64::new(0x7210_0905_0001);
        for _ in 0..10_000 {
            let w = rng.next();
            let f = Fields::split(w);
            assert_eq!(f.join(), w & WORD_MASK, "{w:#018x}");
            assert_eq!(Fields::split(f.join()), f, "{w:#018x}");
        }
        // Each field alone lands at its own slot and nowhere else.
        for (name, slot) in LAYOUT {
            let one = Fields::split(1u64 << slot.0);
            assert_eq!(one.join(), 1u64 << slot.0, "{name}: the low bit of the slot moved");
            assert_eq!(get(one.join(), slot), 1, "{name}");
        }
    }

    /// A field past its width is a different point, not a truncated one.
    #[test]
    #[should_panic(expected = "field s8 = 64 does not fit in 6 bits")]
    fn join_refuses_a_field_wider_than_its_slot() {
        let _ = Fields { s8: 64, ..Fields::default() }.join();
    }
}
