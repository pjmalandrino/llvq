//! The Golay code in trio order as a three-section trellis, and the inverse
//! maps the encoder walks it back with.
//!
//! In trio order a codeword is `c1 | c2 << 8 | c3 << 16`: a prefix byte on
//! coordinates 0..8, a middle byte on 8..16, a suffix byte on 16..24. Two
//! codewords share the state at a cut when they differ by an element of
//! `V = L_past ⊕ L_future`, the span of the codewords supported entirely on
//! one side of the cut; `|V| = 64` at both cuts, so each carries 4096/64 = 64
//! states. A state is NUMBERED by the rank, in ascending integer order, of
//! its coset's smallest member — a format decision, since the word stores the
//! number, and the one under which the three maps are linear over F₂
//! ([`super::LINEAR_COLUMNS`]).
//!
//! The smallest member is computed by reduction against a fully reduced
//! echelon basis of `V` rather than by enumerating the coset: every basis
//! vector is zero at every other basis vector's pivot (its highest bit), so
//! clearing the pivot bits of `c` is order-independent and leaves the unique
//! member of `c ⊕ V` with no pivot bit set — the minimum, because any other
//! member differs from it first at a pivot, where it holds the 1. A test
//! checks that against the 64-element enumeration anyway.
//!
//! Per Golay state: two prefix bytes (complementary), sixteen branches
//! `(middle byte, s16)` with distinct bytes, two suffix bytes (complementary);
//! `64 × 2 × 16 × 2 = 4096`, each factor asserted on its own so no product
//! can compensate for a wrong term. Λ₂₄ adds two bits to this state — the
//! block parity `p` and the running parity of `Σk` — and they live in the
//! word, not in these tables.

use super::{LINEAR_COLUMNS, LINEAR_IN_BITS, TRIO};
use llvq_core::Golay;

/// Golay trellis states at each cut.
pub const GOLAY_STATES: usize = 64;

/// Middle-section branches leaving one state.
pub const BRANCHES: usize = 16;

/// Edges of the middle section.
pub const EDGES: usize = GOLAY_STATES * BRANCHES;

/// Not a prefix byte.
const NONE: u8 = u8::MAX;

/// The trellis, its inverse, and the F₂ columns it reduces to.
pub(super) struct Trellis {
    /// `order[j]`: the natural coordinate that sits at trio position `j`.
    pub order: [u32; 24],
    /// The two prefix bytes of each state at cut 8, ascending.
    pub prefixes: [[u8; 2]; GOLAY_STATES],
    /// The two suffix bytes of each state at cut 16, ascending.
    pub suffixes: [[u8; 2]; GOLAY_STATES],
    /// `branches[s8][b2] = (middle byte, s16)`, ascending by byte.
    pub branches: [[(u8, u8); BRANCHES]; GOLAY_STATES],
    /// Prefix byte → `2·s8 + b1`, or `NONE`: the 128 prefix bytes are distinct.
    prefix_of: [u8; 256],
    /// The twelve F₂ columns of `(s8, b1, b2, b3) ↦ c1 | c2 << 8 | c3 << 16`.
    pub columns: [u32; LINEAR_IN_BITS as usize],
}

/// The trio order: octad 0's coordinates ascending, then octad 1's, then
/// octad 2's. Refuses to exist unless the trio is three disjoint octads of
/// the code covering the 24 — a wrong constant would still permute, still
/// decode, and quantize a different lattice.
fn tetra_order(g: &Golay) -> [u32; 24] {
    let [a, b, c] = TRIO;
    for w in TRIO {
        assert!(g.contains(w), "{w:#010x} is not a Golay codeword");
        assert_eq!(w.count_ones(), 8, "{w:#010x} is not an octad");
    }
    assert_eq!(a & b | a & c | b & c, 0, "the octads of the trio overlap");
    assert_eq!(a | b | c, 0x00ff_ffff, "the trio does not cover the 24 coordinates");
    let mut order = [0u32; 24];
    let mut slots = order.iter_mut();
    for octad in TRIO {
        for i in (0..24).filter(|&i| octad >> i & 1 == 1) {
            *slots.next().expect("24 slots") = i;
        }
    }
    order
}

/// Bit `j` of the result is bit `order[j]` of `w`.
fn permute(w: u32, order: &[u32; 24]) -> u32 {
    order.iter().enumerate().fold(0, |acc, (j, &i)| acc | (w >> i & 1) << j)
}

/// The pivot of a nonzero vector: its highest set bit.
fn pivot(b: u32) -> u32 {
    1 << (31 - b.leading_zeros())
}

/// A fully reduced echelon basis of the span of `vectors`: distinct pivots,
/// and every basis vector zero at every other's pivot.
fn echelon(vectors: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let mut basis: Vec<u32> = Vec::new();
    for v in vectors {
        let v = reduce(v, &basis);
        if v == 0 {
            continue;
        }
        // The new pivot was no one's pivot, so the older vectors may hold
        // it; clearing it there keeps their own pivots, which are higher.
        for b in basis.iter_mut() {
            if *b & pivot(v) != 0 {
                *b ^= v;
            }
        }
        basis.push(v);
    }
    basis
}

/// The smallest member of `w ⊕ span(basis)`, `basis` being fully reduced.
fn reduce(mut w: u32, basis: &[u32]) -> u32 {
    for &b in basis {
        if w & pivot(b) != 0 {
            w ^= b;
        }
    }
    w
}

/// `(s8, b1, b2, b3)` as the F₂ input: `s8` at bits 0..6, `b1` at bit 6,
/// `b2` at bits 7..11, `b3` at bit 11 — the packing the card's `v2` decoder
/// carries as immediates.
pub fn linear_input(s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    (s8 & 63) | (b1 & 1) << 6 | (b2 & 15) << 7 | (b3 & 1) << 11
}

impl Trellis {
    pub(super) fn new() -> Self {
        let g = Golay::new();
        let order = tetra_order(&g);
        let words: Vec<u32> = g.codewords().iter().map(|&w| permute(w, &order)).collect();

        // The state of every codeword at one cut.
        let states = |cut: u32| -> Vec<u8> {
            let low = (1u32 << cut) - 1;
            let basis = echelon(words.iter().copied().filter(|&w| w & low == 0 || w & !low == 0));
            assert_eq!(basis.len(), 6, "cut {cut}: |L_past ⊕ L_future| is 2^{}, not 64", basis.len());
            let reps: Vec<u32> = words.iter().map(|&w| reduce(w, &basis)).collect();
            let mut distinct = reps.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), GOLAY_STATES, "cut {cut} carries {} states", distinct.len());
            reps.iter().map(|r| distinct.binary_search(r).expect("its own representative") as u8).collect()
        };
        let (s8, s16) = (states(8), states(16));

        let mut pre: Vec<Vec<u8>> = vec![Vec::new(); GOLAY_STATES];
        let mut suf: Vec<Vec<u8>> = vec![Vec::new(); GOLAY_STATES];
        let mut edges: Vec<Vec<(u8, u8)>> = vec![Vec::new(); GOLAY_STATES];
        for (i, &w) in words.iter().enumerate() {
            pre[s8[i] as usize].push(w as u8);
            suf[s16[i] as usize].push((w >> 16) as u8);
            edges[s8[i] as usize].push(((w >> 8) as u8, s16[i]));
        }

        let mut prefixes = [[0u8; 2]; GOLAY_STATES];
        let mut suffixes = [[0u8; 2]; GOLAY_STATES];
        let mut branches = [[(0u8, 0u8); BRANCHES]; GOLAY_STATES];
        let mut prefix_of = [NONE; 256];
        for s in 0..GOLAY_STATES {
            for (name, v, out) in [("prefixes", &mut pre[s], &mut prefixes[s]), ("suffixes", &mut suf[s], &mut suffixes[s])] {
                v.sort_unstable();
                v.dedup();
                assert_eq!(v.len(), 2, "state {s} has {} {name}", v.len());
                // Complementary: the section's point set is a coset of 4·E₈,
                // not of 4·D₈.
                assert_eq!(v[0] ^ v[1], 0xff, "the {name} of state {s} are not complementary");
                out.copy_from_slice(v);
            }
            for (b1, &byte) in prefixes[s].iter().enumerate() {
                assert_eq!(prefix_of[byte as usize], NONE, "prefix byte {byte:#04x} opens two states");
                prefix_of[byte as usize] = (2 * s + b1) as u8;
            }
            let e = &mut edges[s];
            e.sort_unstable();
            e.dedup();
            assert_eq!(e.len(), BRANCHES, "state {s} has {} branches", e.len());
            // Single-valued: one middle byte out of one state reaches one
            // state, else the label would not determine the path.
            assert!(e.windows(2).all(|w| w[0].0 != w[1].0), "state {s}: a middle byte reaches two states");
            branches[s].copy_from_slice(e);
        }
        assert_eq!(edges.iter().map(Vec::len).sum::<usize>(), EDGES, "edges");

        let mut t = Self { order, prefixes, suffixes, branches, prefix_of, columns: [0; LINEAR_IN_BITS as usize] };
        t.columns = t.derive_columns();
        t
    }

    /// `c1 | c2 << 8 | c3 << 16` of one path, read from the tables.
    pub(super) fn path(&self, s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
        let c1 = self.prefixes[s8 as usize][b1 as usize];
        let (c2, s16) = self.branches[s8 as usize][b2 as usize];
        let c3 = self.suffixes[s16 as usize][b3 as usize];
        c1 as u32 | (c2 as u32) << 8 | (c3 as u32) << 16
    }

    /// The same, by the F₂ columns and no table.
    pub(super) fn linear_path(&self, s8: u32, b1: u32, b2: u32, b3: u32) -> u32 {
        let x = linear_input(s8, b1, b2, b3);
        self.columns.iter().enumerate().filter(|&(i, _)| x >> i & 1 == 1).fold(0, |a, (_, &c)| a ^ c)
    }

    /// Fit the composed map at the unit inputs, check it on all 4,096 against
    /// the tables, and against the pinned columns — a numbering that broke
    /// linearity anywhere fails here.
    fn derive_columns(&self) -> [u32; LINEAR_IN_BITS as usize] {
        let f = |x: u32| self.path(x & 63, x >> 6 & 1, x >> 7 & 15, x >> 11 & 1);
        assert_eq!(f(0), 0, "the trellis map has a constant term");
        let columns: [u32; LINEAR_IN_BITS as usize] = core::array::from_fn(|i| f(1 << i));
        for x in 0..1u32 << LINEAR_IN_BITS {
            let by_columns = columns.iter().enumerate().filter(|&(i, _)| x >> i & 1 == 1).fold(0, |a, (_, &c)| a ^ c);
            assert_eq!(by_columns, f(x), "input {x:#05x}: the trellis is not linear under its numbering");
        }
        assert_eq!(columns, LINEAR_COLUMNS, "the derived columns are not the pinned ones");
        columns
    }

    /// `(s8, b1)` of a prefix byte, if it is one.
    pub(super) fn state_of_prefix(&self, c1: u8) -> Option<(u8, u8)> {
        let i = self.prefix_of[c1 as usize];
        (i != NONE).then_some((i / 2, i % 2))
    }

    /// `(b2, s16)` of a middle byte out of `s8`, if the edge exists.
    pub(super) fn branch_of(&self, s8: u8, c2: u8) -> Option<(u8, u8)> {
        let e = &self.branches[s8 as usize];
        e.binary_search_by_key(&c2, |&(byte, _)| byte).ok().map(|b2| (b2 as u8, e[b2].1))
    }

    /// `b3` of a suffix byte into `s16`, if it is one of the state's two.
    pub(super) fn suffix_of(&self, s16: u8, c3: u8) -> Option<u8> {
        self.suffixes[s16 as usize].iter().position(|&b| b == c3).map(|b3| b3 as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Bit `order[j]` of the result is bit `j` of `w`.
    fn unpermute(w: u32, order: &[u32; 24]) -> u32 {
        order.iter().enumerate().fold(0, |acc, (j, &i)| acc | (w >> j & 1) << i)
    }

    /// The reduction is the coset minimum: checked against the 64-element
    /// span at both cuts, for all 4,096 codewords. This is what ties the
    /// numbering here to the numbering the yardstick was measured under.
    #[test]
    fn echelon_reduction_is_the_coset_minimum() {
        let g = Golay::new();
        let order = tetra_order(&g);
        let words: Vec<u32> = g.codewords().iter().map(|&w| permute(w, &order)).collect();
        for cut in [8u32, 16] {
            let low = (1u32 << cut) - 1;
            let side: Vec<u32> = words.iter().copied().filter(|&w| w & low == 0 || w & !low == 0).collect();
            let basis = echelon(side.iter().copied());
            let mut span = vec![0u32];
            for &b in &basis {
                let grown: Vec<u32> = span.iter().map(|&x| x ^ b).collect();
                span.extend(grown);
            }
            span.sort_unstable();
            span.dedup();
            assert_eq!(span.len(), 64, "cut {cut}");
            assert!(side.iter().all(|s| span.binary_search(s).is_ok()), "cut {cut}: the span misses a one-sided codeword");
            for &w in &words {
                let min = span.iter().map(|&x| w ^ x).min().expect("64");
                assert_eq!(reduce(w, &basis), min, "cut {cut} word {w:#08x}");
            }
        }
    }

    /// Walking the trellis rebuilds the permuted code exactly, and the
    /// inverse maps invert every entry; 128 distinct bytes at each section.
    #[test]
    fn the_walk_is_the_code_and_the_inverse_maps_invert_it() {
        let t = Trellis::new();
        let g = Golay::new();
        let mut walked = HashSet::new();
        for s8 in 0..GOLAY_STATES {
            for (b1, &c1) in t.prefixes[s8].iter().enumerate() {
                assert_eq!(t.state_of_prefix(c1), Some((s8 as u8, b1 as u8)));
                for (b2, &(c2, s16)) in t.branches[s8].iter().enumerate() {
                    assert_eq!(t.branch_of(s8 as u8, c2), Some((b2 as u8, s16)));
                    for (b3, &c3) in t.suffixes[s16 as usize].iter().enumerate() {
                        assert_eq!(t.suffix_of(s16, c3), Some(b3 as u8));
                        let w = c1 as u32 | (c2 as u32) << 8 | (c3 as u32) << 16;
                        assert!(g.contains(unpermute(w, &t.order)), "{w:#08x} is not a codeword");
                        assert!(walked.insert(w), "{w:#08x} is reached twice");
                    }
                }
            }
        }
        assert_eq!(walked.len(), 4096);
        let bytes = |xs: Vec<u8>| xs.into_iter().collect::<HashSet<u8>>().len();
        assert_eq!(bytes(t.prefixes.iter().flatten().copied().collect()), 128, "prefix bytes");
        assert_eq!(bytes(t.suffixes.iter().flatten().copied().collect()), 128, "suffix bytes");
        assert_eq!(bytes(t.branches.iter().flatten().map(|&(b, _)| b).collect()), 128, "middle bytes");
        // What is not a byte of the section is refused.
        let prefixes: HashSet<u8> = t.prefixes.iter().flatten().copied().collect();
        for c in 0..=255u8 {
            assert_eq!(t.state_of_prefix(c).is_some(), prefixes.contains(&c), "{c:#04x}");
            assert_eq!(t.branch_of(3, c).is_some(), t.branches[3].iter().any(|&(b, _)| b == c), "{c:#04x}");
            assert_eq!(t.suffix_of(3, c).is_some(), t.suffixes[3].contains(&c), "{c:#04x}");
        }
    }
}
