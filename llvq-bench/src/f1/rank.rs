//! The universal 16 KiB decoder table, and the 48-bit word it decodes.
//!
//! This is the reference `llvq-cuda/kernels/llvq_f1rank.cuh` must reproduce
//! bit for bit; `llvq-cuda/tests/f1rank_matches_rust.rs` runs that header on
//! the development machine against [`decode_word`]. Everything the kernel reads
//! is built here: the table by [`RankTable::build`], the three small byte tables
//! by [`prefix_bytes`], [`branch_words`] and [`suffix_bytes`]. What was measured
//! with the private copy of this construction in `examples/f1rankbench.rs`
//! (−0.6 pp of retention against exact F1, 2026-09-05) is what this module
//! serves; the example now imports it, so the two cannot drift.
//!
//! ## The object
//!
//! Λ₂₄ in trio order is a three-section code: sections of 8 coordinates with
//! `y_j = p + 2·c_j + 4·k_j`, `p ∈ {0,1}` shared by the block, `c` the section's
//! Golay pattern byte, `k ∈ Z⁸`. A section is stored not as its point but as
//! its **rank vector** `ρ ∈ {0..7}⁸`: coordinate `j` is the `ρ_j`-th value of
//! the progression `o_j + 4Z` listed outward from zero, `o_j = p + 2·c_j`:
//!
//! ```text
//!   o = 0 :  0, +4, −4, +8, −8, …
//!   o = 2 : +2, −2, +6, −6, +10, …
//!   o = 1 : +1, −3, +5, −7, +9, …
//!   o = 3 : −1, +3, −5, +7, −9, …
//! ```
//!
//! That is [`val`]. The parity of `Σk` — the bit the trellis threads from one
//! section to the next — is a property of `ρ` alone, [`rank_class`], so ONE
//! table serves every pattern of every state. It is one array of 4,096 rows of
//! `u32`, eight ranks of 4 bits, rank `j` at bits `4j..4j+4`:
//!
//! ```text
//!   rows    0..2048 : the 2,048 lowest-cost rank vectors of class 0,
//!                     sorted by (cost, ρ lexicographic) ascending
//!   rows 2048..4096 : the 2,048 lowest-cost of class 1, same order
//!   cost(ρ) = Σ_j (2ρ_j + 1)²
//! ```
//!
//! The end sections read the 2,048 rows of their class. The middle section
//! reads the 2,048 lowest-cost rows OVERALL, which under this order are class-0
//! rows `0..N0` followed by class-1 rows `0..2048−N0`, `N0 = 1240`
//! ([`N0_MIXED`], asserted by the builder). 16 KiB, read by every one of the
//! 528 section regions.
//!
//! ## The word (bit 0 least significant; a block is 6 bytes, little-endian)
//!
//! ```text
//!   bit 0        p
//!   bit 1        r        k-parity class of section 1
//!   bits 2..7    s8       Golay state at cut 8 (0..63)
//!   bit 8        b1       which of the 2 prefix bytes of s8
//!   bits 9..19   i1       row of section 1 in class r            (11 bits)
//!   bits 20..23  b2       branch out of s8 (0..15)
//!   bits 24..34  i2       row of section 2 in the MIXED order    (11 bits)
//!   bit 35       b3       which of the 2 suffix bytes of s16
//!   bits 36..46  i3       row of section 3 in class r3           (11 bits)
//!   bit 47       g        gain bit — read by the served kernel, IGNORED here
//! ```
//!
//! Every field range is a power of two, so any 48-bit word is a valid label
//! and the decode is a bijection onto 2⁴⁷ lattice points per gain value.
//!
//! ## Decode
//!
//! ```text
//!   c1  = prefixes[2·s8 + b1]                  br  = branches[16·s8 + b2]
//!   c2  = br & 0xff ;  s16 = br >> 8           c3  = suffixes[2·s16 + b3]
//!   row1 = table[2048·r + i1]
//!   row2 = i2 < N0 ? table[i2] : table[2048 + i2 − N0] ;  δ = [i2 ≥ N0]
//!   r3   = (p ^ r ^ δ) & 1 ;  row3 = table[2048·r3 + i3]
//!   y[8k + j] = val(p + 2·((c_k >> j) & 1), (row_k >> 4j) & 15)
//! ```
//!
//! `r3` is what closes the block: `Σk ≡ r + δ + (p ^ r ^ δ) ≡ p (mod 2)`,
//! the third constraint of `Leech::contains`. Output in trio order, `|y| ≤ 10`
//! on this table (rank ≤ 4 everywhere), so it fits the kernel's `i8`.

use super::{Trellis, BRANCHES, GOLAY_STATES, SECTION};

/// Rows per parity class.
pub const CLASS_ROWS: usize = 2048;

/// Rows of the whole table: two classes.
pub const ROWS: usize = 2 * CLASS_ROWS;

/// Class-0 rows among the 2,048 lowest-cost overall. A counted fact, not a
/// parameter: [`RankTable::build`] refuses a table where the count differs.
pub const N0_MIXED: usize = 1240;

/// Largest rank the table holds. Asserted by the builder because
/// [`rank_class`] is only exact below 5 — see its comment.
pub const MAX_RANK: u32 = 4;

/// Bits of the word, in the order they are laid: `(name, low bit, width)`.
/// Written once, read by [`split`] and pinned contiguous by a test.
pub const LAYOUT: [(&str, u32, u32); 10] = [
    ("p", 0, 1),
    ("r", 1, 1),
    ("s8", 2, 6),
    ("b1", 8, 1),
    ("i1", 9, 11),
    ("b2", 20, 4),
    ("i2", 24, 11),
    ("b3", 35, 1),
    ("i3", 36, 11),
    ("g", 47, 1),
];

/// Bits of a block on disk and in VRAM.
pub const WORD_BITS: u32 = 48;

/// The `rho`-th value of the progression `o + 4Z`, listed outward from zero.
///
/// Verbatim the `val` of `examples/f1rankbench.rs` that produced the measured
/// retention; the sign alternation is part of the format and the kernel copies
/// it, so it is not to be re-derived. `o = 0` is the only progression that
/// contains zero, hence its extra case: rank 0 is the origin, and ranks 1, 2
/// are ±4, not ±0.
pub fn val(o: u32, rho: u32) -> i32 {
    match o {
        0 => {
            if rho == 0 {
                0
            } else {
                let m = 4 * rho.div_ceil(2) as i32;
                if rho.is_multiple_of(2) { -m } else { m }
            }
        }
        2 => {
            let m = (2 + 4 * (rho / 2)) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        1 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        3 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { -m } else { m }
        }
        _ => unreachable!("o = {o} is not p + 2c"),
    }
}

/// Rank of `y` in the progression `o + 4Z`, or `None` if `y` is not in it or
/// lies beyond rank 7. The inverse of [`val`], by search — eight candidates.
pub fn rank_of(o: u32, y: i32) -> Option<u32> {
    if y.rem_euclid(4) != o as i32 {
        return None;
    }
    (0..8u32).find(|&r| val(o, r) == y)
}

/// Parity of `Σk` for a rank vector: `Σ_j [ρ_j ∈ {1, 2}] mod 2`.
///
/// Pattern-independent, which is the whole point of the universal table — but
/// only because every section byte has even weight. Under `o = 3` the odd
/// ranks are `{0, 3, 4, 7}`, the complement of the other three progressions'
/// `{1, 2, 5, 6}`; the flip cancels over a section because a Golay codeword
/// meets an octad in an even number of coordinates. And the formula names
/// `{1, 2}` rather than `{1, 2, 5, 6}` because ranks 5 and 6 never reach the
/// table (`MAX_RANK`); it is the example's formula verbatim, so the class
/// lists are the ones that were measured.
pub fn rank_class(rho: &[u32; SECTION]) -> u32 {
    rho.iter().filter(|&&r| r == 1 || r == 2).count() as u32 & 1
}

/// `Σ_j (2ρ_j + 1)²`: the second moment of the rank vector under the odd
/// progressions, and the order the table is sorted by.
pub fn cost(rho: &[u32; SECTION]) -> i64 {
    rho.iter().map(|&x| (2 * x as i64 + 1).pow(2)).sum()
}

/// Eight ranks of 4 bits, rank `j` at bits `4j..4j+4`.
pub fn pack(rho: &[u32; SECTION]) -> u32 {
    rho.iter().enumerate().fold(0u32, |a, (j, &r)| a | (r << (4 * j)))
}

/// Undo [`pack`].
pub fn unpack(row: u32) -> [u32; SECTION] {
    core::array::from_fn(|j| (row >> (4 * j)) & 15)
}

/// `n_bits` of `word` starting at `lo_bit`. Bits at or above 48 are never
/// named by [`LAYOUT`], so they cannot reach a decode.
pub fn field(word: u64, lo_bit: u32, n_bits: u32) -> u32 {
    ((word >> lo_bit) & ((1u64 << n_bits) - 1)) as u32
}

/// The ten fields of a word, as [`LAYOUT`] names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word {
    pub p: u32,
    pub r: u32,
    pub s8: u32,
    pub b1: u32,
    pub i1: u32,
    pub b2: u32,
    pub i2: u32,
    pub b3: u32,
    pub i3: u32,
    pub g: u32,
}

/// Cut a word into its fields.
pub fn split(word: u64) -> Word {
    let f = |i: usize| field(word, LAYOUT[i].1, LAYOUT[i].2);
    Word {
        p: f(0),
        r: f(1),
        s8: f(2),
        b1: f(3),
        i1: f(4),
        b2: f(5),
        i2: f(6),
        b3: f(7),
        i3: f(8),
        g: f(9),
    }
}

/// The table, as the kernel holds it.
pub struct RankTable {
    /// 4,096 packed rank vectors: class 0 in `0..2048`, class 1 after, each
    /// half sorted by `(cost, ρ)` ascending.
    pub rows: Vec<u32>,
    /// Class-0 rows among the 2,048 lowest-cost overall; equals [`N0_MIXED`].
    pub n0_mixed: usize,
}

impl RankTable {
    /// Enumerate rank vectors under a cost cap, doubling the cap until both
    /// classes hold 2,048 rows strictly below it — so the ranking is final and
    /// no vector the cap hid could have outranked a kept one, ties on the
    /// boundary cost included (they are enumerated in full and broken by `ρ`).
    ///
    /// Moved from `examples/f1rankbench.rs` without a change to the rule; the
    /// membership sets that example builds are these rows.
    pub fn build() -> Self {
        fn walk(j: usize, acc: i64, cap: i64, rho: &mut [u32; SECTION], out: &mut Vec<([u32; SECTION], i64)>) {
            if j == SECTION {
                out.push((*rho, acc));
                return;
            }
            for r in 0..8u32 {
                let c = (2 * r as i64 + 1).pow(2);
                if acc + c > cap {
                    break;
                }
                rho[j] = r;
                walk(j + 1, acc + c, cap, rho, out);
            }
        }
        let mut cap = 64i64;
        loop {
            let mut all = Vec::new();
            walk(0, 0, cap, &mut [0; SECTION], &mut all);
            all.sort_by_key(|&(r, c)| (c, r));
            let c0: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 0).take(CLASS_ROWS).map(|&(r, _)| r).collect();
            let c1: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 1).take(CLASS_ROWS).map(|&(r, _)| r).collect();
            if c0.len() == CLASS_ROWS && c1.len() == CLASS_ROWS && all.len() >= ROWS {
                let kept = all.iter().take(ROWS).map(|&(_, c)| c).max().expect("4096 rows");
                if kept < cap && cost(&c0[CLASS_ROWS - 1]) < cap && cost(&c1[CLASS_ROWS - 1]) < cap {
                    let n0_mixed = all.iter().take(CLASS_ROWS).filter(|(r, _)| rank_class(r) == 0).count();
                    assert_eq!(n0_mixed, N0_MIXED, "the mixed split is not the counted one");
                    let rows: Vec<u32> = c0.iter().chain(&c1).map(pack).collect();
                    assert_eq!(rows.len(), ROWS);
                    let max_rank = rows.iter().flat_map(|&w| unpack(w)).max().expect("rows");
                    assert_eq!(max_rank, MAX_RANK, "a rank the class formula does not cover reached the table");
                    return Self { rows, n0_mixed };
                }
            }
            cap *= 2;
        }
    }

    /// The 2,048 rows an end section of k-parity `r` may address, in order.
    pub fn class_rows(&self, r: u32) -> &[u32] {
        let lo = CLASS_ROWS * r as usize;
        &self.rows[lo..lo + CLASS_ROWS]
    }

    /// The 2,048 rows the middle section may address: the lowest-cost overall,
    /// in the order `i2` walks them (class 0 first, then class 1).
    pub fn mixed_rows(&self) -> Vec<u32> {
        let mut v = self.rows[..self.n0_mixed].to_vec();
        v.extend_from_slice(&self.rows[CLASS_ROWS..CLASS_ROWS + CLASS_ROWS - self.n0_mixed]);
        v
    }
}

/// `prefixes[2·s8 + b1]`: the kernel's flat copy of `Trellis::prefixes`.
pub fn prefix_bytes(tr: &Trellis) -> [u8; 2 * GOLAY_STATES] {
    core::array::from_fn(|i| tr.prefixes[i / 2][i % 2])
}

/// `suffixes[2·s16 + b3]`: the kernel's flat copy of `Trellis::suffixes`.
pub fn suffix_bytes(tr: &Trellis) -> [u8; 2 * GOLAY_STATES] {
    core::array::from_fn(|i| tr.suffixes[i / 2][i % 2])
}

/// `branches[16·s8 + b2]`, low byte the middle pattern, high byte `s16`: the
/// kernel's flat copy of `Trellis::branches`, whose lists are sorted by byte.
pub fn branch_words(tr: &Trellis) -> [u16; BRANCHES * GOLAY_STATES] {
    core::array::from_fn(|i| {
        let (byte, s16) = tr.branches[i / BRANCHES][i % BRANCHES];
        byte as u16 | (s16 as u16) << 8
    })
}

/// One section: eight coordinates from a pattern byte and a packed row.
fn section(p: u32, c: u8, row: u32) -> [i32; SECTION] {
    core::array::from_fn(|j| val(p + 2 * ((c >> j) & 1) as u32, (row >> (4 * j)) & 15))
}

/// The decode, exactly as the module header states it. Reads the trellis in
/// its own representation rather than through the flat tables, so the harness
/// that feeds those tables to the kernel compares two independent paths.
pub fn decode_word(word: u64, table: &RankTable, tr: &Trellis) -> [i32; 24] {
    let w = split(word);
    let c1 = tr.prefixes[w.s8 as usize][w.b1 as usize];
    let (c2, s16) = tr.branches[w.s8 as usize][w.b2 as usize];
    let c3 = tr.suffixes[s16 as usize][w.b3 as usize];

    let row1 = table.rows[CLASS_ROWS * w.r as usize + w.i1 as usize];
    let (row2, delta) = if (w.i2 as usize) < N0_MIXED {
        (table.rows[w.i2 as usize], 0)
    } else {
        (table.rows[CLASS_ROWS + w.i2 as usize - N0_MIXED], 1)
    };
    let r3 = (w.p ^ w.r ^ delta) & 1;
    let row3 = table.rows[CLASS_ROWS * r3 as usize + w.i3 as usize];

    let mut y = [0i32; 24];
    y[..8].copy_from_slice(&section(w.p, c1, row1));
    y[8..16].copy_from_slice(&section(w.p, c2, row2));
    y[16..].copy_from_slice(&section(w.p, c3, row3));
    y
}

/// Input bits of [`TrellisLinear`]: `s8` at bits 0..6, `b1` at bit 6, `b2`
/// at bits 7..11, `b3` at bit 11 — the four trellis fields of a word in the
/// order the word lays them, with the row indices between them left out.
pub const LINEAR_IN_BITS: u32 = 12;

/// The columns of [`TrellisLinear`] on the trellis of [`TRIO`](super::TRIO),
/// derived on 2026-09-05 (`examples/f1linear.rs`): a counted fact like
/// [`N0_MIXED`], not a parameter — [`TrellisLinear::derive`] is checked
/// against it by a test, and `llvq-cuda/kernels/llvq_f1rank_v2.cuh` carries
/// the same twelve numbers as immediates. Column `i` is `c1 | c2 << 8 |
/// c3 << 16` contributed by input bit `i`; the constant term is zero.
pub const LINEAR_COLUMNS: [u32; LINEAR_IN_BITS as usize] = [
    0x2d002e, 0x3a005a, 0x740033, 0x03061e, 0x050963, 0x090578, // s8, bits 0..6
    0x0000ff, // b1: the complementary prefix
    0x2d1d00, 0x3a2b00, 0x744700, 0x638e00, // b2, bits 0..4
    0xff0000, // b3: the complementary suffix
];

/// Pack the four trellis fields of a word into the input of [`TrellisLinear`].
pub fn linear_input(s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    (s8 & 63) | (b1 & 1) << 6 | (b2 & 15) << 7 | (b3 & 1) << 11
}

/// The three pattern bytes of one path through the trellis, packed
/// `c1 | c2 << 8 | c3 << 16`, read from the trellis's own tables — the truth
/// [`TrellisLinear::patterns`] is compared to.
pub fn trellis_patterns(tr: &Trellis, s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    let c1 = tr.prefixes[s8 as usize][b1 as usize];
    let (c2, s16) = tr.branches[s8 as usize][b2 as usize];
    let c3 = tr.suffixes[s16 as usize][b3 as usize];
    c1 as u32 | (c2 as u32) << 8 | (c3 as u32) << 16
}

/// An affine model `f(x) = ⊕_i x_i·columns[i] ⊕ k` of a function on `n_in`
/// bits, fitted at zero and at the unit vectors, and the number of inputs on
/// which the model and the function disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineFit {
    /// `f(0)`.
    pub k: u32,
    /// `f(e_i) ⊕ f(0)`, one per input bit.
    pub columns: Vec<u32>,
    /// Inputs, out of `2^n_in`, where `f` is not the model. Zero means `f`
    /// is affine over F₂; zero with `k == 0` means linear.
    pub violations: usize,
}

/// Fit `f` on `n_in` input bits and compare the fit to `f` on EVERY input.
/// Exhaustive on purpose: an affine map is determined by `n_in + 1` values,
/// so any disagreement anywhere is a disproof, and the domains here are at
/// most 4,096 wide.
pub fn affine_fit(n_in: u32, f: impl Fn(u32) -> u32) -> AffineFit {
    let k = f(0);
    let columns: Vec<u32> = (0..n_in).map(|i| f(1 << i) ^ k).collect();
    let model = |x: u32| (0..n_in).filter(|&i| x >> i & 1 == 1).fold(k, |a, i| a ^ columns[i as usize]);
    let violations = (0..1u32 << n_in).filter(|&x| f(x) != model(x)).count();
    AffineFit { k, columns, violations }
}

/// The trellis as ONE F₂-linear map, `(s8, b1, b2, b3) ↦ (c1, c2, c3)`.
///
/// The three lookups of the decode — `prefixes[2·s8 + b1]`, `branches[16·s8 +
/// b2]` giving `(c2, s16)`, `suffixes[2·s16 + b3]` — are linear in the cosets
/// by construction, and it is a fact of the numbering, not of the algebra,
/// that they are linear in the NUMBERS the word stores: `Trellis::new` sorts
/// the canonical (smallest) coset representatives and numbers them in that
/// order, and the smallest representatives of a subspace's cosets form a
/// subspace whose numeric order is a linear coordinate (each coset's minimum
/// is its member with zeros at the subspace's pivot bits, which is a linear
/// reduction); the sixteen middle bytes of a state are a coset of one fixed
/// 4-dimensional subspace (`examples/f1mids.rs`) and its sorted order is again
/// linear. [`derive`](Self::derive) does not trust the argument: it fits the
/// model and checks it on all 4,096 inputs, and `s16` is composed away, so a
/// decoder needs twelve 24-bit columns and no table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisLinear {
    /// `columns[i]`: `c1 | c2 << 8 | c3 << 16` contributed by input bit `i`
    /// of [`linear_input`].
    pub columns: [u32; LINEAR_IN_BITS as usize],
}

impl TrellisLinear {
    /// Fit the composed map on `tr` and verify it on every one of the 4,096
    /// inputs; `None` if the trellis is not linear (affine with a nonzero
    /// constant included) under its numbering.
    pub fn derive(tr: &Trellis) -> Option<Self> {
        let fit = affine_fit(LINEAR_IN_BITS, |x| trellis_patterns(tr, x & 63, x >> 6 & 1, x >> 7 & 15, x >> 11 & 1));
        if fit.violations != 0 || fit.k != 0 {
            return None;
        }
        Some(Self { columns: fit.columns.try_into().expect("LINEAR_IN_BITS columns") })
    }

    /// `c1 | c2 << 8 | c3 << 16` for the four trellis fields, by algebra.
    pub fn patterns(&self, s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
        let x = linear_input(s8, b1, b2, b3);
        self.columns.iter().enumerate().filter(|&(i, _)| x >> i & 1 == 1).fold(0, |a, (_, &c)| a ^ c)
    }
}

/// [`decode_word`] with the trellis replaced by [`TrellisLinear`]: the same
/// rows, the same [`val`], the pattern bytes by algebra. Written out rather
/// than sharing a body with [`decode_word`], so the two paths stay
/// independent and a test can compare them; this is what
/// `llvq-cuda/kernels/llvq_f1rank_v2.cuh` mirrors.
pub fn decode_word_linear(word: u64, table: &RankTable, lin: &TrellisLinear) -> [i32; 24] {
    let w = split(word);
    let cc = lin.patterns(w.s8, w.b1, w.b2, w.b3);

    let row1 = table.rows[CLASS_ROWS * w.r as usize + w.i1 as usize];
    let (row2, delta) = if (w.i2 as usize) < N0_MIXED {
        (table.rows[w.i2 as usize], 0)
    } else {
        (table.rows[CLASS_ROWS + w.i2 as usize - N0_MIXED], 1)
    };
    let r3 = (w.p ^ w.r ^ delta) & 1;
    let row3 = table.rows[CLASS_ROWS * r3 as usize + w.i3 as usize];

    let mut y = [0i32; 24];
    y[..8].copy_from_slice(&section(w.p, (cc & 0xff) as u8, row1));
    y[8..16].copy_from_slice(&section(w.p, (cc >> 8 & 0xff) as u8, row2));
    y[16..].copy_from_slice(&section(w.p, (cc >> 16 & 0xff) as u8, row3));
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f1::{point_to_natural, SectionSet};
    use llvq_core::SplitMix64;
    use std::collections::HashSet;

    /// A 48-bit word: the label and its gain bit, nothing above.
    fn word(rng: &mut SplitMix64) -> u64 {
        rng.next() & ((1u64 << WORD_BITS) - 1)
    }

    /// `Σk mod 2` of a section read under its pattern, from the point alone.
    fn k_parity(p: u32, c: u8, y: &[i32]) -> u32 {
        (0..SECTION)
            .map(|j| {
                let base = p as i32 + 2 * ((c >> j) & 1) as i32;
                (y[j] - base).div_euclid(4).rem_euclid(2) as u32
            })
            .fold(0, |a, b| a ^ b)
    }

    /// The rank vector a decoded section came from, recovered through
    /// [`rank_of`] — the inverse path, which `decode_word` never takes.
    fn rho_of(p: u32, c: u8, y: &[i32]) -> [u32; SECTION] {
        core::array::from_fn(|j| rank_of(p + 2 * ((c >> j) & 1) as u32, y[j]).expect("a decoded value has a rank"))
    }

    /// (d) The four progressions, as the format states them.
    #[test]
    fn val_reproduces_the_four_progressions() {
        const WANT: [(u32, [i32; 5]); 4] = [
            (0, [0, 4, -4, 8, -8]),
            (2, [2, -2, 6, -6, 10]),
            (1, [1, -3, 5, -7, 9]),
            (3, [-1, 3, -5, 7, -9]),
        ];
        for (o, seq) in WANT {
            for (rho, &y) in seq.iter().enumerate() {
                assert_eq!(val(o, rho as u32), y, "o={o} ρ={rho}");
            }
        }
        // Beyond the listed prefix: still the residue class, still outward,
        // still injective — a progression that repeated or skipped a value
        // would make two ranks one point or leave a lattice point unreachable.
        for o in 0..4u32 {
            let mut seen = HashSet::new();
            for rho in 0..8u32 {
                let y = val(o, rho);
                assert_eq!(y.rem_euclid(4), o as i32, "o={o} ρ={rho}: not in o + 4Z");
                assert!(seen.insert(y), "o={o}: ρ={rho} repeats a value");
                assert_eq!(rank_of(o, y), Some(rho), "o={o}: rank_of does not invert val");
                if rho > 0 {
                    assert!(y.abs() >= val(o, rho - 1).abs(), "o={o}: ρ={rho} moves inward");
                }
            }
        }
    }

    /// (c) Two classes of the 2,048 lowest-cost vectors, each in (cost, ρ)
    /// order, class 0 then class 1, and the counted mixed split.
    ///
    /// "Lowest" is checked by brute force, not trusted from the builder: every
    /// `ρ ∈ {0..4}⁸` is enumerated flat (390,625 vectors; a rank of 5 already
    /// costs 121, above any row), sorted by the same key, and the first 2,048
    /// of each class must be the rows exactly — sequence equality, so an order
    /// slip and a skipped vector fail alike.
    #[test]
    fn the_rows_are_the_lowest_cost_of_each_class_in_cost_then_lex_order() {
        let t = RankTable::build();
        assert_eq!(t.rows.len(), ROWS);
        assert_eq!(t.n0_mixed, N0_MIXED);
        assert_eq!(t.n0_mixed, 1240);

        let mut all: Vec<[u32; SECTION]> = Vec::with_capacity(5usize.pow(8));
        for code in 0..5u32.pow(8) {
            let mut c = code;
            let mut rho = [0u32; SECTION];
            for slot in rho.iter_mut() {
                *slot = c % 5;
                c /= 5;
            }
            all.push(rho);
        }
        all.sort_by_key(|r| (cost(r), *r));
        for r in 0..2u32 {
            let want: Vec<u32> = all.iter().filter(|v| rank_class(v) == r).take(CLASS_ROWS).map(pack).collect();
            assert_eq!(t.class_rows(r), &want[..], "class {r}: not the 2,048 lowest in (cost, ρ) order");
        }
        // The mixed 2,048 are the head of the merged order, and `i2` walks
        // them class 0 first: the same rows, at the positions `decode_word`
        // reads them from.
        let head: HashSet<u32> = all.iter().take(CLASS_ROWS).map(pack).collect();
        let mixed = t.mixed_rows();
        assert_eq!(mixed.len(), CLASS_ROWS);
        assert_eq!(mixed.iter().copied().collect::<HashSet<_>>(), head, "the mixed rows are not the 2,048 lowest overall");
        assert_eq!(mixed.iter().filter(|&&w| rank_class(&unpack(w)) == 0).count(), N0_MIXED);
        // The class rows never exceed MAX_RANK, which is what makes the
        // `{1, 2}` class formula exact on them; and 4,096 distinct rows.
        assert!(t.rows.iter().all(|&w| unpack(w).iter().all(|&x| x <= MAX_RANK)));
        assert_eq!(t.rows.iter().copied().collect::<HashSet<_>>().len(), ROWS, "duplicate rows");
    }

    /// The word layout tiles bits 0..48 exactly once, in order.
    #[test]
    fn the_layout_is_contiguous_and_forty_eight_bits_wide() {
        let mut next = 0u32;
        for (name, lo, n) in LAYOUT {
            assert_eq!(lo, next, "{name} does not start where the previous field ends");
            assert!(n > 0);
            next = lo + n;
        }
        assert_eq!(next, WORD_BITS);
        let w = split(u64::MAX);
        assert_eq!((w.s8, w.i1, w.b2, w.i2, w.i3), (63, 2047, 15, 2047, 2047), "field widths");
        assert_eq!(field(0xdead_beef_cafe_f00d, 8, 16), 0xf00d >> 8 | (0xcafe & 0xff) << 8);
    }

    /// The flat tables the kernel reads are the trellis, entry for entry.
    #[test]
    fn the_flat_tables_are_the_trellis() {
        let tr = Trellis::new();
        let (pre, suf, br) = (prefix_bytes(&tr), suffix_bytes(&tr), branch_words(&tr));
        for s in 0..GOLAY_STATES {
            for b in 0..2 {
                assert_eq!(pre[2 * s + b], tr.prefixes[s][b]);
                assert_eq!(suf[2 * s + b], tr.suffixes[s][b]);
            }
            for b in 0..BRANCHES {
                let (byte, s16) = tr.branches[s][b];
                assert_eq!((br[16 * s + b] & 0xff) as u8, byte);
                assert_eq!((br[16 * s + b] >> 8) as u8, s16);
                assert!((s16 as usize) < GOLAY_STATES);
            }
        }
    }

    /// (a) Random words land in Λ₂₄ — the lattice's own membership test, in
    /// its own coordinate order. This is the check a wrong `r3`, a swapped
    /// class block or a wrong sign in `val` cannot survive.
    #[test]
    fn two_thousand_random_words_decode_into_the_leech_lattice() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let leech = llvq_core::Leech::new();
        let mut rng = SplitMix64::new(0xf12a_2026_0905);
        for _ in 0..2_000 {
            let w = word(&mut rng);
            let y = decode_word(w, &t, &tr);
            assert!(y.iter().all(|v| v.abs() <= 10), "{w:#014x}: a coordinate outside ±10 — {y:?}");
            let natural = point_to_natural(&y, &tr.code.order);
            assert!(leech.contains(&natural), "{w:#014x} decodes outside Λ₂₄: {y:?}");
        }
    }

    /// (b) The word is a bijection: distinct words, distinct points. Twenty
    /// thousand draws from 2⁴⁷ cannot collide by chance, so any repeat is a
    /// field read twice or a row reachable two ways.
    #[test]
    fn twenty_thousand_random_words_give_twenty_thousand_distinct_points() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let mut rng = SplitMix64::new(0xf12b_2026_0905);
        let mut seen = HashSet::with_capacity(20_000);
        for _ in 0..20_000 {
            // The gain bit is not part of the point, so it is held at zero
            // here: two words differing only there decode alike by design.
            let w = word(&mut rng) & !(1u64 << 47);
            assert!(seen.insert(decode_word(w, &t, &tr)), "{w:#014x} repeats an earlier point");
        }
        assert_eq!(seen.len(), 20_000);
    }

    /// (e) Each decoded section is a member of the section set the word
    /// names, with the k-parity the trellis requires: `r` for section 1, free
    /// for section 2, and for section 3 the parity that closes the block —
    /// computed here from section 2's ACTUAL `Σk`, not from the decoder's `δ`,
    /// so a wrong `r3` formula is caught by this test alone.
    #[test]
    fn each_decoded_section_is_a_member_of_its_section_set_with_the_right_parity() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let mut rng = SplitMix64::new(0xf12e_2026_0905);
        for _ in 0..500 {
            let wd = word(&mut rng);
            let w = split(wd);
            let y = decode_word(wd, &t, &tr);
            let c1 = tr.prefixes[w.s8 as usize][w.b1 as usize];
            let (c2, s16) = tr.branches[w.s8 as usize][w.b2 as usize];
            let c3 = tr.suffixes[s16 as usize][w.b3 as usize];
            let delta = k_parity(w.p, c2, &y[8..16]);
            // The closing parity, from the block's own constraint `Σk ≡ p`:
            // section 3 owes what sections 1 and 2 did not pay. Written
            // differently from the decoder's line on purpose, so a textual
            // mutation of one cannot silently reach the other.
            let r3 = (w.p + w.r + delta) % 2;
            for (k, (c, par)) in [(c1, Some(w.r)), (c2, None), (c3, Some(r3))].into_iter().enumerate() {
                let set = SectionSet { patterns: vec![c], p: w.p, k_parity: par };
                let sec: [i32; SECTION] = y[8 * k..8 * k + 8].try_into().expect("8");
                assert!(set.contains(&sec), "{wd:#014x} section {k}: {sec:?} is not in its set (pattern {c:#04x}, parity {par:?})");
            }
            // And the middle's parity is the class of the row the word named,
            // which is what the decoder's `δ` claims.
            assert_eq!(delta, u32::from(w.i2 as usize >= N0_MIXED), "{wd:#014x}: δ is not the middle row's class");
        }
    }

    /// The middle index walks exactly the 2,048 lowest-cost rows, each once —
    /// the N0 split of the decoder, checked against the table's own sets. An
    /// off-by-one on N0 keeps every point in Λ₂₄ and every word distinct; only
    /// the set of rows reached shows it.
    #[test]
    fn the_middle_index_reaches_the_mixed_rows_exactly_once() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let want: HashSet<u32> = t.mixed_rows().into_iter().collect();
        for p in 0..2u64 {
            let mut reached = HashSet::with_capacity(CLASS_ROWS);
            for i2 in 0..CLASS_ROWS as u64 {
                let wd = p | (5 << 2) | (3 << 20) | (i2 << 24);
                let w = split(wd);
                let (c2, _) = tr.branches[w.s8 as usize][w.b2 as usize];
                let y = decode_word(wd, &t, &tr);
                let row = pack(&rho_of(w.p, c2, &y[8..16]));
                assert!(reached.insert(row), "p={p} i2={i2}: row {row:#010x} reached twice");
            }
            assert_eq!(reached, want, "p={p}: the middle index does not reach the mixed rows");
        }
    }

    /// The decision of 2026-09-05 (`examples/f1linear.rs`), pinned: each of
    /// the three trellis maps, as the decoder reads them, is LINEAR over F₂ in
    /// the bits of the numbers the word stores — `(s8, b1) → c1` on its 128
    /// inputs, `(s8, b2) → (c2, s16)` on its 1,024, `(s16, b3) → c3` on its
    /// 128 — with a zero constant term. The fit is exhaustive, so a numbering
    /// that broke linearity anywhere would fail here, and no relabelling of
    /// the word is needed for an algebraic decoder.
    #[test]
    fn the_three_trellis_maps_are_linear_under_the_current_numbering() {
        let tr = Trellis::new();
        let m1 = affine_fit(7, |x| tr.prefixes[(x & 63) as usize][(x >> 6 & 1) as usize] as u32);
        let m2 = affine_fit(10, |x| {
            let (c2, s16) = tr.branches[(x & 63) as usize][(x >> 6 & 15) as usize];
            c2 as u32 | (s16 as u32) << 8
        });
        let m3 = affine_fit(7, |x| tr.suffixes[(x & 63) as usize][(x >> 6 & 1) as usize] as u32);
        for (name, m) in [("(s8, b1) → c1", &m1), ("(s8, b2) → (c2, s16)", &m2), ("(s16, b3) → c3", &m3)] {
            assert_eq!(m.violations, 0, "{name} is not affine: {} inputs off the model", m.violations);
            assert_eq!(m.k, 0, "{name} has a constant term");
        }
        // The complementary prefix and suffix are the all-ones columns.
        assert_eq!(m1.columns[6], 0xff);
        assert_eq!(m3.columns[6], 0xff);
    }

    /// The composed map derives, its columns are the pinned ones, and it
    /// reproduces the trellis's pattern bytes on all 64 × 2 × 16 × 2 paths.
    #[test]
    fn the_linear_columns_are_the_derived_ones_and_reproduce_every_path() {
        let tr = Trellis::new();
        let lin = TrellisLinear::derive(&tr).expect("the trellis is linear under its numbering");
        assert_eq!(lin.columns, LINEAR_COLUMNS, "the derived columns are not the pinned ones");
        for s8 in 0..GOLAY_STATES as u32 {
            for b1 in 0..2 {
                for b2 in 0..BRANCHES as u32 {
                    for b3 in 0..2 {
                        assert_eq!(
                            lin.patterns(s8, b1, b2, b3),
                            trellis_patterns(&tr, s8, b1, b2, b3),
                            "s8={s8} b1={b1} b2={b2} b3={b3}: the algebra and the tables disagree"
                        );
                    }
                }
            }
        }
        // The input packing is a bijection onto 12 bits, field by field.
        assert_eq!(linear_input(63, 1, 15, 1), (1 << LINEAR_IN_BITS) - 1);
        assert_eq!(linear_input(0, 1, 0, 0), 1 << 6);
        assert_eq!(linear_input(0, 0, 1, 0), 1 << 7);
        assert_eq!(linear_input(0, 0, 0, 1), 1 << 11);
    }

    /// The algebraic decode is the table decode, on random words and under
    /// every single-bit flip of them — so a column mistaken by one bit, which
    /// only some words would reveal, is reached from 200 directions.
    #[test]
    fn decode_word_linear_is_decode_word() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let lin = TrellisLinear::derive(&tr).expect("linear");
        let mut rng = SplitMix64::new(0xf12c_2026_0905);
        for _ in 0..200 {
            let wd = word(&mut rng);
            assert_eq!(decode_word_linear(wd, &t, &lin), decode_word(wd, &t, &tr), "{wd:#014x}");
            for bit in 0..WORD_BITS {
                let f = wd ^ (1u64 << bit);
                assert_eq!(decode_word_linear(f, &t, &lin), decode_word(f, &t, &tr), "{f:#014x} (bit {bit} of {wd:#014x})");
            }
        }
    }

    /// `affine_fit` tells a linear map from a non-linear one: XOR of two
    /// inputs passes with the expected columns; AND of them is caught with
    /// exactly the one violation it has.
    #[test]
    fn affine_fit_detects_non_linearity() {
        let xor = affine_fit(2, |x| (x & 1) ^ (x >> 1 & 1));
        assert_eq!((xor.k, &xor.columns[..], xor.violations), (0, &[1, 1][..], 0));
        let and = affine_fit(2, |x| (x & 1) & (x >> 1 & 1));
        assert_eq!((and.k, &and.columns[..], and.violations), (0, &[0, 0][..], 1));
        let affine = affine_fit(3, |x| x ^ 5);
        assert_eq!((affine.k, affine.violations), (5, 0));
    }

    /// Every bit below 47 moves the point, for every word; bit 47 never does.
    /// A field read one bit narrow, one bit shifted, or the gain bit leaking
    /// into an index would all fail here.
    #[test]
    fn every_label_bit_moves_the_point_and_the_gain_bit_does_not() {
        let (t, tr) = (RankTable::build(), Trellis::new());
        let mut rng = SplitMix64::new(0xf12f_2026_0905);
        for _ in 0..200 {
            let wd = word(&mut rng);
            let y = decode_word(wd, &t, &tr);
            for bit in 0..WORD_BITS {
                let moved = decode_word(wd ^ (1u64 << bit), &t, &tr) != y;
                assert_eq!(moved, bit < 47, "{wd:#014x}: flipping bit {bit} {}", if moved { "moved the point" } else { "changed nothing" });
            }
        }
    }
}
