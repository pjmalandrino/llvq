//! Tetra — Λ₂₄ on a 48-bit word of three sections, the format of roadmap lead
//! F1 (`docs/ROADMAP.md` §2.2 quater, step 0).
//!
//! The served v1 index is 47 bits into a ball with no table, unfolded to
//! 4.804 b/weight in VRAM. Tetra reorders the 24 coordinates by three disjoint
//! octads of the Golay code ([`TRIO`]) so that the code is a three-section
//! trellis with 64 states at each cut; a block is then `p`, a path through
//! that trellis, and three 11-bit rows of one universal 16 KiB table
//! ([`Tetra::rows`]) — read from VRAM as it is written on disk.
//!
//! A point of the integer embedding is `y_j = p + 2·c_j + 4·k_j` with `p` the
//! shared parity, `c` a Golay codeword and `Σk ≡ p (mod 2)`. In trio order
//! the sections are the three bytes of `c`; each section is stored as the
//! rank vector of its eight coordinates in their residue-class progressions
//! ([`val`]), and the `Σk` constraint is threaded as one bit: section 1's
//! class `r`, section 2's class `δ` (the mixed row it names), section 3 in
//! class `p ⊕ r ⊕ δ`. The word layout is in [`Fields`]; the decode is
//!
//! ```text
//!   c1 = prefixes[s8][b1]      (c2, s16) = branches[s8][b2]     c3 = suffixes[s16][b3]
//!   row1 = rows[2048·r + i1]
//!   row2 = i2 < N0 ? rows[i2] : rows[2048 + i2 − N0]           δ = [i2 ≥ N0]
//!   row3 = rows[2048·(p ⊕ r ⊕ δ) + i3]
//!   y[8k + j] = val(p + 2·bit_j(c_k), nibble_j(row_k))         then y → natural order
//! ```
//!
//! `llvq_bench::f1::rank` is the independent yardstick this module is pinned
//! to (`llvq-bench/tests/tetra_yardstick.rs`): it was written first, measured
//! (−0.6 pp of retention against exact F1, 2026-09-05) and reproduced on the
//! card; this module re-derives the same objects from `llvq_core::Golay` and
//! the definitions above, with its own construction, and agrees with it word
//! for word. The card's decoders (`llvq-cuda/kernels/llvq_f1rank*.cuh`) read
//! the same word and the same table.
//!
//! ## What is pinned at construction
//!
//! [`Tetra::new`] refuses to exist unless: the trio is three disjoint octads
//! of the code; each cut carries 64 states and the middle 1,024 edges; the
//! mixed split is [`N0_MIXED`]; no row exceeds rank 4; the three closed-form
//! bounds are [`CLASS_BOUNDS`] and [`MIXED_BOUND`]; and the trellis is the
//! F₂-linear map [`LINEAR_COLUMNS`]. All counted, none trusted.
//!
//! ## Coordinate orders
//!
//! [`Tetra::decode`] and [`Tetra::encode`] speak the repository's NATURAL order
//! — what `Leech::contains`, GPTQ and the artifact use. Tetra order exists
//! only inside the word; [`Tetra::decode_trio_order`] exposes it for the
//! yardstick. The origin is word 0: `p = 0`, state 0, prefix/middle/suffix
//! bytes 0, and the all-zero rank vector, which is row 0 of class 0.
//!
//! ## What `encode` refuses
//!
//! `None` for anything that does not land exactly on a label: mixed
//! coordinate parities, a mod-4 pattern that is not a Golay codeword, a
//! coordinate past rank 7, a rank vector outside its class set (past the
//! boundary cut included), a middle row outside the mixed set, and a block
//! whose `Σk ≢ p` — that last one surfaces as section 3's class not being
//! `p ⊕ r ⊕ δ`. The gain bit of an encoded word is 0. This strictness is
//! what makes a v5 fingerprint mean something.

pub mod encoder;
mod rank;
mod trellis;
mod word;

pub use encoder::{Encoder, Scratch, TetraCode};
pub use rank::{cost, pack, rank_class, rank_of, unpack, val, Bound, CLASS_BOUNDS, MAX_RANK, MIXED_BOUND};
pub use trellis::{linear_input, BRANCHES, EDGES, GOLAY_STATES};
pub use word::{Fields, LABEL_MASK, LAYOUT, WORD_MASK};

use llvq_core::DIM;

/// The trio: three disjoint octads covering the 24 coordinates, found by
/// `llvq-bench/src/bin/f1count.rs` as the first among the 759 and pinned.
/// Octad 0 fills trio positions 0..8, octad 1 8..16, octad 2 16..24.
pub const TRIO: [u32; 3] = [0x0000_149f, 0x000f_6840, 0x00f0_8320];

/// Bits of a block: `[p 1][r 1][s8 6][b1 1][i1 11][b2 4][i2 11][b3 1][i3 11][gain 1]`.
pub const WORD_BITS: u32 = 48;

/// Bits 0..47 name the point; bit 47 is the gain bit, opaque to this module.
pub const LABEL_BITS: u32 = 47;

/// Coordinates per section.
pub const SECTION: usize = 8;

/// Rows of the table: two classes.
pub const ROWS: usize = 2 * CLASS_ROWS;

/// Rows per k-parity class.
pub const CLASS_ROWS: usize = 2048;

/// Class-0 rows among the 2,048 lowest-cost overall — the split of the
/// middle section's index. A counted fact, asserted by the builder.
pub const N0_MIXED: usize = 1240;

/// Input bits of the F₂ map: `s8` (6), `b1` (1), `b2` (4), `b3` (1).
pub const LINEAR_IN_BITS: u32 = 12;

/// The trellis as twelve F₂ columns: column `i` is `c1 | c2 << 8 | c3 << 16`
/// contributed by bit `i` of [`linear_input`]; no constant term. Derived on
/// 2026-09-05, equal to `llvq_bench::f1::rank::LINEAR_COLUMNS`, carried as
/// immediates by the card's `v2` decoder; re-derived and asserted by
/// [`Tetra::new`].
pub const LINEAR_COLUMNS: [u32; LINEAR_IN_BITS as usize] = [
    0x2d002e, 0x3a005a, 0x740033, 0x03061e, 0x050963, 0x090578, // s8, bits 0..6
    0x0000ff, // b1: the complementary prefix
    0x2d1d00, 0x3a2b00, 0x744700, 0x638e00, // b2, bits 0..4
    0xff0000, // b3: the complementary suffix
];

/// The word map: the trellis, the table, and their inverses.
pub struct Tetra {
    trellis: trellis::Trellis,
    table: rank::Table,
}

impl Tetra {
    /// Build from `llvq_core::Golay`; panics on any invariant of the module
    /// doc that does not hold.
    pub fn new() -> Self {
        Self { trellis: trellis::Trellis::new(), table: rank::Table::build() }
    }

    /// `order()[j]`: the natural coordinate at trio position `j`.
    pub fn order(&self) -> &[u32; DIM] {
        &self.trellis.order
    }

    /// The 16 KiB table, class 0 then class 1, each in `(cost, ρ)` order.
    pub fn rows(&self) -> &[u32; ROWS] {
        &self.table.rows
    }

    /// The two prefix bytes of each state at cut 8.
    pub fn prefixes(&self) -> &[[u8; 2]; GOLAY_STATES] {
        &self.trellis.prefixes
    }

    /// The two suffix bytes of each state at cut 16.
    pub fn suffixes(&self) -> &[[u8; 2]; GOLAY_STATES] {
        &self.trellis.suffixes
    }

    /// `branches()[s8][b2] = (middle byte, s16)`.
    pub fn branches(&self) -> &[[(u8, u8); BRANCHES]; GOLAY_STATES] {
        &self.trellis.branches
    }

    /// `(c1, c2, c3)` of a word by the F₂ columns, no table.
    pub fn patterns(&self, word: u64) -> (u8, u8, u8) {
        let f = Fields::split(word);
        let c = self.trellis.linear_path(f.s8 as u32, f.b1 as u32, f.b2 as u32, f.b3 as u32);
        (c as u8, (c >> 8) as u8, (c >> 16) as u8)
    }

    /// `(class, index in the class)` of a rank vector, if it is a row — what
    /// an end section's `(r, i)` are.
    pub fn class_index(&self, rho: &[u32; SECTION]) -> Option<(u32, u16)> {
        self.table.class_index(rho)
    }

    /// `(δ, i2)` of a rank vector, if it is one of the 2,048 lowest overall.
    pub fn mixed_index(&self, rho: &[u32; SECTION]) -> Option<(u32, u16)> {
        self.table.mixed_index(rho)
    }

    /// The point of a word, in trio order. Bit 47 and above are ignored.
    pub fn decode_trio_order(&self, word: u64) -> [i32; DIM] {
        let f = Fields::split(word);
        let (s8, i2) = (f.s8 as usize, f.i2 as usize);
        let c1 = self.trellis.prefixes[s8][f.b1 as usize];
        let (c2, s16) = self.trellis.branches[s8][f.b2 as usize];
        let c3 = self.trellis.suffixes[s16 as usize][f.b3 as usize];

        let rows = &self.table.rows;
        let row1 = rows[CLASS_ROWS * f.r as usize + f.i1 as usize];
        let (row2, delta) = if i2 < N0_MIXED { (rows[i2], 0) } else { (rows[CLASS_ROWS + i2 - N0_MIXED], 1) };
        let r3 = (f.p ^ f.r ^ delta) & 1;
        let row3 = rows[CLASS_ROWS * r3 as usize + f.i3 as usize];

        let mut y = [0i32; DIM];
        for (k, (c, row)) in [(c1, row1), (c2, row2), (c3, row3)].into_iter().enumerate() {
            for (j, out) in y[SECTION * k..SECTION * (k + 1)].iter_mut().enumerate() {
                *out = val(f.p as u32 + 2 * ((c >> j) & 1) as u32, (row >> (4 * j)) & 15);
            }
        }
        y
    }

    /// The point of a word, in natural order.
    pub fn decode(&self, word: u64) -> [i32; DIM] {
        let tetra = self.decode_trio_order(word);
        let mut natural = [0i32; DIM];
        for (j, &v) in tetra.iter().enumerate() {
            natural[self.trellis.order[j] as usize] = v;
        }
        natural
    }

    /// The word of a point given in natural order, gain bit 0; `None` unless
    /// the point is exactly a Tetra codeword (module doc).
    pub fn encode(&self, point: &[i32; DIM]) -> Option<u64> {
        let p = point[0].rem_euclid(2);
        if point.iter().any(|&v| v.rem_euclid(2) != p) {
            return None;
        }
        let y: [i32; DIM] = core::array::from_fn(|j| point[self.trellis.order[j] as usize]);
        // c_j = ((y_j − p) / 2) mod 2; the difference is even, so the shift
        // halves it exactly, and it cannot overflow: i32::MIN is even.
        let c = y.iter().enumerate().fold(0u32, |acc, (j, &v)| acc | (((v - p) >> 1).rem_euclid(2) as u32) << j);
        let (c1, c2, c3) = (c as u8, (c >> 8) as u8, (c >> 16) as u8);
        let (s8, b1) = self.trellis.state_of_prefix(c1)?;
        let (b2, s16) = self.trellis.branch_of(s8, c2)?;
        let b3 = self.trellis.suffix_of(s16, c3)?;

        let ranks = |k: usize, byte: u8| -> Option<[u32; SECTION]> {
            let mut rho = [0u32; SECTION];
            for (j, r) in rho.iter_mut().enumerate() {
                *r = rank_of(p as u32 + 2 * ((byte >> j) & 1) as u32, y[SECTION * k + j])?;
            }
            Some(rho)
        };
        let (r, i1) = self.table.class_index(&ranks(0, c1)?)?;
        let (delta, i2) = self.table.mixed_index(&ranks(1, c2)?)?;
        let (r3, i3) = self.table.class_index(&ranks(2, c3)?)?;
        // Σk ≡ p over the block: section 3 owes what the first two did not pay.
        if r3 != (p as u32 ^ r ^ delta) & 1 {
            return None;
        }
        Some(Fields { p: p as u8, r: r as u8, s8, b1, i1, b2, i2, b3, i3, gain: 0 }.join())
    }
}

impl Default for Tetra {
    fn default() -> Self {
        Self::new()
    }
}
