//! The universal rank table: how a section of eight coordinates becomes an
//! 11-bit row, and the closed form of the three sets a row can belong to.
//!
//! A section is `y_j = o_j + 4·k_j` with `o_j = p + 2·c_j ∈ {0, 1, 2, 3}` fixed
//! by the block parity and the section's Golay byte. The table stores no
//! points; it stores **rank vectors** `ρ ∈ {0..7}⁸`, coordinate `j` being the
//! `ρ_j`-th member of the residue class `o_j + 4Z` listed outward from zero,
//! the positive member of a tied pair first ([`val`]):
//!
//! ```text
//!   o = 0 :  0, +4, −4, +8, −8, +12, −12, +16
//!   o = 2 : +2, −2, +6, −6, +10, −10, +14, −14
//!   o = 1 : +1, −3, +5, −7, +9, −11, +13, −15
//!   o = 3 : −1, +3, −5, +7, −9, +11, −13, +15
//! ```
//!
//! The sign alternation is part of the format — the card's decoders copy
//! it — and it is stated here as the listing rule, not as four formulas, so
//! the four progressions are one definition.
//!
//! What makes ONE table serve every pattern of every state is that the bit
//! the trellis threads between sections — the parity of `Σk` — is a function
//! of `ρ` alone over a section ([`rank_class`]): under `o ∈ {0, 1, 2}` the odd
//! `k` sit at ranks `{1, 2, 5, 6}`, under `o = 3` at their complement, and a
//! Golay codeword meets each octad of the trio in an even number of
//! coordinates, so the flips cancel over the eight.
//!
//! ## The rows
//!
//! `cost(ρ) = Σ_j (2ρ_j + 1)²` orders the vectors; ties are broken by `ρ`
//! lexicographically, `ρ_0` most significant. The table is 4,096 packed rows
//! of `u32`, rank `j` at bits `4j..4j+4`:
//!
//! ```text
//!   rows    0..2048   the 2,048 lowest-cost vectors of class 0, ascending
//!   rows 2048..4096   the 2,048 lowest-cost vectors of class 1, ascending
//! ```
//!
//! An end section addresses the 2,048 rows of its class. The middle section
//! addresses the 2,048 lowest-cost rows OVERALL, which in this order are
//! class-0 rows `0..N0` then class-1 rows `0..2048 − N0`, `N0 = 1,240`
//! ([`N0_MIXED`]); the class of the row it lands on is the `δ` the last
//! section closes against.
//!
//! ## The closed form
//!
//! The rows of a class are a prefix of a total order, so membership is one
//! comparison against the last row kept: `cost(ρ) < C`, or `cost(ρ) = C` and
//! `ρ ≤ cut` lexicographically ([`Bound`]). [`Table::build`] derives the three
//! bounds from the rows and asserts them equal to the pinned [`CLASS_BOUNDS`]
//! and [`MIXED_BOUND`] — the numbers the encoder of step 1 compares against,
//! and the ones `docs/mesures/f1-encodeur-prototype-2026-09-05.txt` counted.

use super::{CLASS_ROWS, N0_MIXED, ROWS, SECTION};

/// Largest rank any row holds. Asserted by the builder: it is what makes the
/// kernels' `|y| ≤ 10` and the `{1, 2}` reading of [`rank_class`] exact.
pub const MAX_RANK: u32 = 4;

/// Every vector of cost below this is enumerated when the table is built. It
/// is a bound the builder checks, not a parameter: the last row of each set
/// must cost strictly less, so that nothing left out could have outranked a
/// row kept. `(2·5 + 1)² + 7 = 128`, so rank 5 never enters.
const COST_CAP: u32 = 128;

/// `VALUES[o][ρ]`: the four progressions, built from the listing rule.
const VALUES: [[i32; 8]; 4] = progressions();

/// Walk the magnitudes outward from zero; at each, emit `+m` then `−m` when
/// they lie in `o + 4Z` (`m = 0` once). Eight per residue.
const fn progressions() -> [[i32; 8]; 4] {
    let mut t = [[0i32; 8]; 4];
    let mut o = 0usize;
    while o < 4 {
        let (mut n, mut m) = (0usize, 0i32);
        while n < 8 {
            if m % 4 == o as i32 {
                t[o][n] = m;
                n += 1;
            }
            if m > 0 && n < 8 && (-m).rem_euclid(4) == o as i32 {
                t[o][n] = -m;
                n += 1;
            }
            m += 1;
        }
        o += 1;
    }
    t
}

/// The `rho`-th value of `o + 4Z` listed outward from zero, positive first.
pub fn val(o: u32, rho: u32) -> i32 {
    assert!(o < 4, "o = {o} is not p + 2c");
    assert!(rho < 8, "rank {rho} is past the progression");
    VALUES[o as usize][rho as usize]
}

/// Rank of `y` in `o + 4Z`, or `None` if `y` is not in it or lies past rank 7.
pub fn rank_of(o: u32, y: i32) -> Option<u32> {
    assert!(o < 4, "o = {o} is not p + 2c");
    VALUES[o as usize].iter().position(|&v| v == y).map(|r| r as u32)
}

/// Parity of `Σk` over a section, from its ranks alone: the parity of
/// `k = val(0, ρ_j) / 4` summed over the eight — the reading under `o = 0`,
/// which is also the reading under `o = 1` and `o = 2`, and the complement of
/// the reading under `o = 3` on each coordinate; the complements come in
/// even number per section (module doc), so this is the block's own `Σk`.
pub fn rank_class(rho: &[u32; SECTION]) -> u32 {
    rho.iter().map(|&r| (val(0, r) / 4).rem_euclid(2) as u32).fold(0, |a, b| a ^ b)
}

/// `Σ_j (2ρ_j + 1)²`: the order of the table.
pub fn cost(rho: &[u32; SECTION]) -> u32 {
    rho.iter().map(|&r| (2 * r + 1).pow(2)).sum()
}

/// Eight ranks of 4 bits, rank `j` at bits `4j..4j+4`.
pub fn pack(rho: &[u32; SECTION]) -> u32 {
    rho.iter().enumerate().fold(0u32, |a, (j, &r)| a | (r & 15) << (4 * j))
}

/// Undo [`pack`].
pub fn unpack(row: u32) -> [u32; SECTION] {
    core::array::from_fn(|j| (row >> (4 * j)) & 15)
}

/// A set of rank vectors closed downward in `(cost, ρ)` order: everything
/// strictly cheaper than `cost`, plus the vectors at `cost` up to `cut`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bound {
    pub cost: u32,
    pub cut: [u32; SECTION],
}

impl Bound {
    /// Membership by one comparison. Says nothing about the class: an end
    /// section also needs `rank_class(ρ) == r`.
    pub fn contains(&self, rho: &[u32; SECTION]) -> bool {
        let c = cost(rho);
        c < self.cost || (c == self.cost && *rho <= self.cut)
    }
}

/// The last row of each class, pinned; derived from the rows and asserted
/// by [`Table::build`].
pub const CLASS_BOUNDS: [Bound; 2] = [
    Bound { cost: 88, cut: [1, 1, 0, 3, 0, 0, 1, 1] },
    Bound { cost: 96, cut: [0, 0, 2, 2, 2, 0, 1, 1] },
];

/// The 2,048th vector overall, pinned: the middle section's set.
pub const MIXED_BOUND: Bound = Bound { cost: 72, cut: [3, 0, 0, 0, 1, 0, 1, 0] };

/// The table and its inverse.
pub(super) struct Table {
    /// The 16 KiB the card reads.
    pub rows: [u32; ROWS],
    /// `(packed row, position in rows)`, sorted by row — the inverse the
    /// encoder looks a rank vector up in.
    index: Vec<(u32, u16)>,
}

/// Every `ρ` with `cost(ρ) < COST_CAP`, by depth-first extension: a rank
/// that overshoots the cap ends its loop, since costs rise with rank and the
/// coordinates still open cost at least 1 each.
fn under_cap(j: usize, spent: u32, rho: &mut [u32; SECTION], out: &mut Vec<[u32; SECTION]>) {
    if j == SECTION {
        out.push(*rho);
        return;
    }
    let open = (SECTION - 1 - j) as u32;
    let mut r = 0u32;
    while spent + (2 * r + 1).pow(2) + open < COST_CAP {
        rho[j] = r;
        under_cap(j + 1, spent + (2 * r + 1).pow(2), rho, out);
        r += 1;
    }
}

impl Table {
    /// Enumerate under the cap, sort by `(cost, ρ)`, split by class, keep
    /// 2,048 of each; derive the bounds and the mixed split and check them
    /// against the pinned constants. Every count is asserted on its own.
    pub(super) fn build() -> Self {
        let mut all: Vec<[u32; SECTION]> = Vec::new();
        under_cap(0, 0, &mut [0; SECTION], &mut all);
        all.sort_unstable_by_key(|r| (cost(r), *r));

        let mut classes: [Vec<[u32; SECTION]>; 2] = [Vec::new(), Vec::new()];
        for r in &all {
            let c = classes[rank_class(r) as usize].len();
            if c < CLASS_ROWS {
                classes[rank_class(r) as usize].push(*r);
            }
        }
        for (c, kept) in classes.iter().enumerate() {
            assert_eq!(kept.len(), CLASS_ROWS, "class {c}: {} rows under cost {COST_CAP}", kept.len());
            let last = kept[CLASS_ROWS - 1];
            // Strictly below the cap: then every vector not enumerated costs
            // more than the last row kept, and the prefix is final.
            assert!(cost(&last) < COST_CAP, "class {c}: the cap {COST_CAP} cuts the ranking");
            let derived = Bound { cost: cost(&last), cut: last };
            assert_eq!(derived, CLASS_BOUNDS[c], "class {c}: the closed form is not the pinned one");
        }

        let n0 = all[..CLASS_ROWS].iter().filter(|r| rank_class(r) == 0).count();
        assert_eq!(n0, N0_MIXED, "the mixed split is not the counted one");
        let last = all[CLASS_ROWS - 1];
        assert!(cost(&last) < COST_CAP, "the cap {COST_CAP} cuts the mixed ranking");
        assert_eq!(Bound { cost: cost(&last), cut: last }, MIXED_BOUND, "the mixed closed form is not the pinned one");
        // The mixed order is the closed form, row by row: class-0 rows below
        // N0 and class-1 rows below 2048 − N0 are exactly the members.
        for (c, kept) in classes.iter().enumerate() {
            let n = if c == 0 { N0_MIXED } else { CLASS_ROWS - N0_MIXED };
            for (i, r) in kept.iter().enumerate() {
                assert_eq!(MIXED_BOUND.contains(r), i < n, "class {c} row {i}: mixed membership and the split disagree");
            }
        }

        let mut rows = [0u32; ROWS];
        for (c, kept) in classes.iter().enumerate() {
            for (i, r) in kept.iter().enumerate() {
                rows[CLASS_ROWS * c + i] = pack(r);
            }
        }
        let max_rank = rows.iter().flat_map(|&w| unpack(w)).max().expect("rows");
        assert_eq!(max_rank, MAX_RANK, "a rank past {MAX_RANK} reached the table");
        for (i, &w) in rows.iter().enumerate() {
            assert_eq!(rank_class(&unpack(w)) as usize, i / CLASS_ROWS, "row {i} sits in the wrong class block");
        }

        let mut index: Vec<(u32, u16)> = rows.iter().enumerate().map(|(i, &w)| (w, i as u16)).collect();
        index.sort_unstable();
        assert!(index.windows(2).all(|w| w[0].0 != w[1].0), "duplicate rows");
        Self { rows, index }
    }

    /// Position of a rank vector in `rows`, if it is one.
    fn position(&self, rho: &[u32; SECTION]) -> Option<usize> {
        // `pack` keeps 4 bits per rank: a rank of 16 would alias row 0.
        if rho.iter().any(|&r| r > MAX_RANK) {
            return None;
        }
        let key = pack(rho);
        self.index.binary_search_by_key(&key, |&(w, _)| w).ok().map(|i| self.index[i].1 as usize)
    }

    /// `(class, index within the class)` of a rank vector, if it is a row.
    pub(super) fn class_index(&self, rho: &[u32; SECTION]) -> Option<(u32, u16)> {
        self.position(rho).map(|p| ((p / CLASS_ROWS) as u32, (p % CLASS_ROWS) as u16))
    }

    /// `(δ, i2)` of a rank vector in the mixed order, if it is one of the
    /// 2,048 lowest overall: class-0 rows keep their index, class-1 rows
    /// follow at `N0 + index`.
    pub(super) fn mixed_index(&self, rho: &[u32; SECTION]) -> Option<(u32, u16)> {
        let (class, i) = self.class_index(rho)?;
        let (i, n) = (i as usize, if class == 0 { N0_MIXED } else { CLASS_ROWS - N0_MIXED });
        (i < n).then(|| (class, (i + class as usize * N0_MIXED) as u16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_core::SplitMix64;

    /// Every `ρ ∈ {0..4}⁸`, flat — 390,625 vectors, every row among them.
    fn all_to_four() -> Vec<[u32; SECTION]> {
        (0..5u32.pow(8))
            .map(|code| {
                let mut c = code;
                core::array::from_fn(|_| {
                    let r = c % 5;
                    c /= 5;
                    r
                })
            })
            .collect()
    }

    /// The four progressions, as the format states them, and their shape:
    /// in the residue class, outward, injective, inverted by `rank_of`.
    #[test]
    fn val_reproduces_the_four_progressions() {
        const WANT: [[i32; 8]; 4] = [
            [0, 4, -4, 8, -8, 12, -12, 16],
            [1, -3, 5, -7, 9, -11, 13, -15],
            [2, -2, 6, -6, 10, -10, 14, -14],
            [-1, 3, -5, 7, -9, 11, -13, 15],
        ];
        for (o, seq) in WANT.iter().enumerate() {
            for (rho, &y) in seq.iter().enumerate() {
                assert_eq!(val(o as u32, rho as u32), y, "o={o} ρ={rho}");
                assert_eq!(rank_of(o as u32, y), Some(rho as u32), "o={o}: rank_of does not invert val");
                assert_eq!(y.rem_euclid(4), o as i32, "o={o} ρ={rho}: not in o + 4Z");
                if rho > 0 {
                    assert!(y.abs() >= seq[rho - 1].abs(), "o={o}: ρ={rho} moves inward");
                }
            }
            // |y| ≥ 17 is past rank 7 in every class; the class's own members there must still be refused.
            for y in (17i32..=20).flat_map(|m| [m, -m]).filter(|y| y.rem_euclid(4) == o as i32) {
                assert_eq!(rank_of(o as u32, y), None, "o={o}: {y} past rank 7 has a rank");
            }
            assert_eq!(rank_of(o as u32, seq[0] + 1), None, "o={o}: a value off the class has a rank");
        }
    }

    /// `rank_class` is the block's own `Σk` parity: read the section back
    /// through `val` under every offset byte of even weight, with `p` free.
    /// The odd-weight control shows the even weight is load-bearing.
    #[test]
    fn rank_class_is_the_k_parity_under_every_even_weight_byte() {
        let mut rng = SplitMix64::new(0x7210_0905_0002);
        let k_parity = |p: u32, c: u8, rho: &[u32; SECTION]| -> u32 {
            (0..SECTION)
                .map(|j| {
                    let o = p + 2 * ((c >> j) & 1) as u32;
                    ((val(o, rho[j]) - o as i32) / 4).rem_euclid(2) as u32
                })
                .fold(0, |a, b| a ^ b)
        };
        let (mut even, mut odd_flipped) = (0usize, 0usize);
        for _ in 0..20_000 {
            let rho: [u32; SECTION] = core::array::from_fn(|_| (rng.next() % 8) as u32);
            let (p, c) = ((rng.next() & 1) as u32, rng.next() as u8);
            if c.count_ones() % 2 == 0 {
                assert_eq!(k_parity(p, c, &rho), rank_class(&rho), "p={p} c={c:#04x} ρ={rho:?}");
                even += 1;
            } else if p == 1 {
                // One o = 3 coordinate too many: the reading flips.
                assert_ne!(k_parity(p, c, &rho), rank_class(&rho), "p={p} c={c:#04x} ρ={rho:?}");
                odd_flipped += 1;
            }
        }
        assert!(even > 5_000 && odd_flipped > 2_000, "the sample covered {even} even and {odd_flipped} odd bytes");
        // The journal's identity: (2ρ+1)² ≡ 9 (mod 16) iff k is odd, so the
        // class is whether the cost is a multiple of 16.
        for rho in all_to_four() {
            assert_eq!(rank_class(&rho), u32::from(cost(&rho).is_multiple_of(16)), "{rho:?}");
        }
    }

    /// The rows are the 2,048 lowest of each class in `(cost, ρ)` order —
    /// sequence equality against a flat enumeration sorted by the same key,
    /// so a skipped vector and an order slip fail alike.
    #[test]
    fn the_rows_are_the_lowest_of_each_class_in_order() {
        let t = Table::build();
        let mut all = all_to_four();
        all.sort_unstable_by_key(|r| (cost(r), *r));
        for c in 0..2u32 {
            let want: Vec<u32> = all.iter().filter(|r| rank_class(r) == c).take(CLASS_ROWS).map(pack).collect();
            assert_eq!(&t.rows[CLASS_ROWS * c as usize..CLASS_ROWS * (c as usize + 1)], &want[..], "class {c}");
        }
        // And the mixed order walks the 2,048 lowest overall, class 0 first.
        let head: Vec<u32> = all.iter().take(CLASS_ROWS).map(pack).collect();
        let mut walked = vec![0u32; CLASS_ROWS];
        let mut reached = 0;
        for &w in &head {
            let (delta, i2) = t.mixed_index(&unpack(w)).expect("a head row is mixed");
            assert_eq!(delta, rank_class(&unpack(w)));
            walked[i2 as usize] = w;
            reached += 1;
        }
        assert_eq!(reached, CLASS_ROWS);
        let mut sorted = walked.clone();
        sorted.sort_unstable();
        let mut head_sorted = head;
        head_sorted.sort_unstable();
        assert_eq!(sorted, head_sorted, "i2 does not reach every head row once");
        assert_eq!(walked[..N0_MIXED].iter().filter(|&&w| rank_class(&unpack(w)) == 0).count(), N0_MIXED);
    }

    /// The closed form and the table agree on all 390,625 vectors up to rank
    /// 4, and on the journal's counts under and on each boundary cost; a
    /// vector with a rank of 5 is refused by both.
    #[test]
    fn the_closed_form_is_the_table() {
        let t = Table::build();
        let mut under = [0usize; 3];
        let mut on = [0usize; 3];
        for rho in all_to_four() {
            let c = rank_class(&rho);
            let member = CLASS_BOUNDS[c as usize].contains(&rho);
            assert_eq!(t.class_index(&rho).map(|(cls, _)| cls), member.then_some(c), "{rho:?}");
            assert_eq!(t.mixed_index(&rho).is_some(), MIXED_BOUND.contains(&rho), "{rho:?}");
            for (k, b) in [(c as usize, CLASS_BOUNDS[c as usize]), (2, MIXED_BOUND)] {
                if cost(&rho) < b.cost {
                    under[k] += 1;
                } else if cost(&rho) == b.cost && rho <= b.cut {
                    on[k] += 1;
                }
            }
        }
        // docs/mesures/f1-encodeur-prototype-2026-09-05.txt:45-47.
        assert_eq!((under, on), ([1256, 1816, 1307], [792, 232, 741]));
        let five = [5, 0, 0, 0, 0, 0, 0, 0];
        assert!(t.class_index(&five).is_none() && !CLASS_BOUNDS[rank_class(&five) as usize].contains(&five));
        // A rank of 16 packs to the nibble of rank 0: both lookups must
        // refuse it rather than answer with the origin's row.
        let sixteen = [16, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!((t.class_index(&sixteen), t.mixed_index(&sixteen)), (None, None));
        // The index is total on the rows: every packed row finds its position.
        for (i, &w) in t.rows.iter().enumerate() {
            assert_eq!(t.class_index(&unpack(w)), Some(((i / CLASS_ROWS) as u32, (i % CLASS_ROWS) as u16)));
        }
    }
}
