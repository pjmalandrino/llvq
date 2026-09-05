//! # Trio — the word map, pinned from both sides
//!
//! The worst failure of a codebook is the silent one: a word that decodes to
//! a plausible point that is not the one encoded, or a point that encodes to
//! a word another point already owns. These tests pin the bijection from
//! both sides — `encode ∘ decode` on random labels, `decode` into Λ₂₄ and
//! onto distinct points — and the refusals of `encode`, which are what a
//! fingerprint over this map will rest on. The agreement with the bench's
//! decoder is in `llvq-bench/tests/trio_yardstick.rs`, the only place both
//! crates are visible.

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::trio::{
    cost, pack, rank_class, unpack, val, Fields, Trio, CLASS_BOUNDS, CLASS_ROWS, LABEL_MASK, LINEAR_COLUMNS, MIXED_BOUND,
    N0_MIXED, ROWS, SECTION, WORD_BITS,
};

/// A 48-bit word: the label and its gain bit, nothing above.
fn word(rng: &mut SplitMix64) -> u64 {
    rng.next() & ((1u64 << WORD_BITS) - 1)
}

/// A point in trio order from `p`, the three pattern bytes and the three
/// rank vectors — the definition written forwards, then moved to natural
/// order the same way `Trio::decode` does. Used to build points the
/// decoder never produces.
fn assemble(t: &Trio, p: u32, bytes: [u8; 3], rhos: [[u32; SECTION]; 3]) -> [i32; DIM] {
    let mut natural = [0i32; DIM];
    for k in 0..3 {
        for j in 0..SECTION {
            let o = p + 2 * ((bytes[k] >> j) & 1) as u32;
            natural[t.order()[SECTION * k + j] as usize] = val(o, rhos[k][j]);
        }
    }
    natural
}

/// (2) `encode(decode(w))` gives back the label of `w` — a million words in
/// release, a hundred thousand in debug.
#[test]
fn every_label_round_trips_through_its_point() {
    const N: usize = if cfg!(debug_assertions) { 100_000 } else { 1_000_000 };
    let t = Trio::new();
    let mut rng = SplitMix64::new(0x7210_2026_0905_0002);
    for _ in 0..N {
        let w = word(&mut rng);
        let y = t.decode(w);
        assert_eq!(t.encode(&y), Some(w & LABEL_MASK), "{w:#014x} → {y:?}");
    }
    // The gain bit is not part of the point: both settings decode alike and
    // encode to the label with the bit clear.
    let w = word(&mut rng) & LABEL_MASK;
    assert_eq!(t.decode(w), t.decode(w | 1 << 47));
    assert_eq!(t.encode(&t.decode(w | 1 << 47)), Some(w));
}

/// (3) Random words land in Λ₂₄ in natural order, and distinct words land on
/// distinct points. Twenty thousand draws from 2⁴⁷ do not collide by chance.
#[test]
fn decoded_words_are_distinct_leech_points() {
    let t = Trio::new();
    let leech = Leech::new();
    let mut rng = SplitMix64::new(0x7210_2026_0905_0003);
    let mut seen = std::collections::HashSet::with_capacity(20_000);
    for _ in 0..20_000 {
        let w = word(&mut rng) & LABEL_MASK;
        let y = t.decode(w);
        assert!(y.iter().all(|v| v.abs() <= 10), "{w:#014x}: a coordinate outside ±10 — {y:?}");
        assert!(leech.contains(&y), "{w:#014x} decodes outside Λ₂₄: {y:?}");
        assert!(seen.insert(y), "{w:#014x} repeats an earlier point");
    }
    assert_eq!(seen.len(), 20_000);
}

/// (4) The origin is word 0, pinned in both directions.
#[test]
fn the_origin_is_word_zero() {
    let t = Trio::new();
    assert_eq!(t.decode(0), [0; DIM]);
    assert_eq!(t.encode(&[0; DIM]), Some(0));
    let f = Fields::split(0);
    assert_eq!((f.p, f.r, f.s8, f.b1, f.i1, f.b2, f.i2, f.b3, f.i3), (0, 0, 0, 0, 0, 0, 0, 0, 0));
    assert_eq!(t.prefixes()[0][0], 0, "state 0 does not carry the zero prefix");
    assert_eq!(unpack(t.rows()[0]), [0; SECTION], "row 0 of class 0 is not the zero rank vector");
}

/// (4) `encode` refuses a point that is not exactly a Trio codeword: mixed
/// parities, a pattern that is not Golay, a block with `Σk ≢ p`, a rank past
/// the table. Each probe is one edit of a valid point; each is also checked
/// against `Leech::contains` so the reason for the refusal is named.
#[test]
fn encode_refuses_what_is_not_a_trio_point() {
    let t = Trio::new();
    let leech = Leech::new();
    let mut rng = SplitMix64::new(0x7210_2026_0905_0004);
    for _ in 0..500 {
        let w = word(&mut rng) & LABEL_MASK;
        let y = t.decode(w);
        let j = (rng.next() % DIM as u64) as usize;

        // A coordinate of the other parity.
        let mut odd = y;
        odd[j] += 1;
        assert!(!leech.contains(&odd));
        assert_eq!(t.encode(&odd), None, "{w:#014x}: a flipped parity encodes");

        // The mod-4 pattern with one bit flipped is not a codeword.
        let mut pattern = y;
        pattern[j] += 2;
        assert!(!leech.contains(&pattern));
        assert_eq!(t.encode(&pattern), None, "{w:#014x}: a non-Golay pattern encodes");

        // k_j += 1: the parity, pattern and residues all hold, only Σk ≢ p.
        // The section's rank vector may or may not be a row of the other
        // class, so this is refused either by the row lookup or by r3.
        let mut ksum = y;
        ksum[j] += 4;
        assert!(!leech.contains(&ksum));
        assert_eq!(t.encode(&ksum), None, "{w:#014x}: Σk ≢ p encodes");

        // k_j += 2 keeps the point in Λ₂₄ and pushes coordinate j two ranks
        // outward, which no row holds once the section's cost passes its
        // bound — always, from rank 2 on: (2·4+1)² = 81 alone is under 96,
        // so probe with a doubled step to make the refusal unconditional.
        let mut far = y;
        far[j] += 16;
        assert!(leech.contains(&far), "{w:#014x}: +16 left Λ₂₄");
        assert_eq!(t.encode(&far), None, "{w:#014x}: a coordinate past the table encodes");
    }
}

/// (4) The boundary shell is cut where the rows say: on each of the three
/// sets the last vector kept encodes and its lexicographic successor at the
/// same cost does not — both points being in Λ₂₄, so the refusal is the
/// region's and not the lattice's.
#[test]
fn the_boundary_shell_is_cut_at_the_pinned_vector() {
    let t = Trio::new();
    let leech = Leech::new();
    // Every ρ ∈ {0..4}⁸ at the boundary cost of a bound, sorted; the first one
    // past the cut is the probe.
    let successor = |bound_cost: u32, cut: [u32; SECTION], class: Option<u32>| -> [u32; SECTION] {
        let mut on_shell: Vec<[u32; SECTION]> = (0..5u32.pow(8))
            .map(|code| {
                let mut c = code;
                core::array::from_fn(|_| {
                    let r = c % 5;
                    c /= 5;
                    r
                })
            })
            .filter(|r| cost(r) == bound_cost && r > &cut && class.is_none_or(|k| rank_class(r) == k))
            .collect();
        on_shell.sort_unstable();
        *on_shell.first().expect("the shell continues past the cut")
    };
    let zero = [0u32; SECTION];
    let p = 1;
    let (c1, (c2, s16)) = (t.prefixes()[5][1], t.branches()[5][9]);
    let c3 = t.suffixes()[s16 as usize][0];
    let bytes = [c1, c2, c3];

    // End sections: class r's cut, with the middle at row 0 (δ = 0) and the
    // other end at row 0 of the class that closes the block.
    for r in 0..2u32 {
        let b = CLASS_BOUNDS[r as usize];
        let past = successor(b.cost, b.cut, Some(r));
        let close = |rho1: [u32; SECTION]| {
            let r3 = (p ^ r) & 1;
            let rho3 = unpack(t.rows()[CLASS_ROWS * r3 as usize]);
            assemble(&t, p, bytes, [rho1, zero, rho3])
        };
        let (kept, refused) = (close(b.cut), close(past));
        assert!(leech.contains(&kept) && leech.contains(&refused), "class {r}: a probe left Λ₂₄");
        assert!(t.encode(&kept).is_some(), "class {r}: the cut {:?} does not encode", b.cut);
        assert_eq!(t.encode(&refused), None, "class {r}: {past:?} past the cut encodes");
        assert_eq!(t.class_index(&b.cut), Some((r, (CLASS_ROWS - 1) as u16)), "class {r}: the cut is not the last row");
    }

    // The middle: the mixed cut, whose class fixes section 3's.
    let past = successor(MIXED_BOUND.cost, MIXED_BOUND.cut, None);
    for (rho2, want) in [(MIXED_BOUND.cut, true), (past, false)] {
        let delta = rank_class(&rho2);
        let rho3 = unpack(t.rows()[CLASS_ROWS * ((p ^ delta) & 1) as usize]);
        let y = assemble(&t, p, bytes, [zero, rho2, rho3]);
        assert!(leech.contains(&y), "middle {rho2:?}: the probe left Λ₂₄");
        assert_eq!(t.encode(&y).is_some(), want, "middle {rho2:?}");
    }
    // The mixed cut is a class-0 vector (two ranks in {1, 2}), so it is the
    // last class-0 row the mixed order reaches, i2 = N0 − 1 — not i2 = 2047.
    assert_eq!(t.mixed_index(&MIXED_BOUND.cut), Some((0, (N0_MIXED - 1) as u16)));
}

/// (5) The columns re-derived from the trellis tables are the pinned ones,
/// and `patterns` by algebra equals the tables on all 64 × 2 × 16 × 2 paths,
/// whatever the other fields hold.
#[test]
fn the_linear_columns_are_the_trellis() {
    let t = Trio::new();
    let by_tables = |s8: usize, b1: usize, b2: usize, b3: usize| -> (u8, u8, u8) {
        let (c2, s16) = t.branches()[s8][b2];
        (t.prefixes()[s8][b1], c2, t.suffixes()[s16 as usize][b3])
    };
    let mut rng = SplitMix64::new(0x7210_2026_0905_0005);
    let mut derived = [0u32; 12];
    for (i, col) in derived.iter_mut().enumerate() {
        let x = 1u32 << i;
        let (c1, c2, c3) = by_tables((x & 63) as usize, (x >> 6 & 1) as usize, (x >> 7 & 15) as usize, (x >> 11 & 1) as usize);
        *col = c1 as u32 | (c2 as u32) << 8 | (c3 as u32) << 16;
    }
    assert_eq!(derived, LINEAR_COLUMNS, "the columns fitted on the tables are not the pinned ones");
    assert_eq!(by_tables(0, 0, 0, 0), (0, 0, 0), "the map has a constant term");
    for s8 in 0..64 {
        for b1 in 0..2 {
            for b2 in 0..16 {
                for b3 in 0..2 {
                    let noise = word(&mut rng);
                    let w = Fields { s8: s8 as u8, b1: b1 as u8, b2: b2 as u8, b3: b3 as u8, ..Fields::split(noise) }.join();
                    assert_eq!(t.patterns(w), by_tables(s8, b1, b2, b3), "s8={s8} b1={b1} b2={b2} b3={b3}");
                }
            }
        }
    }
}

/// (6) The `(C, cut)` triples read off the rows are the literals of the doc:
/// the last row of each class block, and the largest key the mixed order
/// reaches; and the block boundaries of the table are where they should be.
#[test]
fn the_closed_form_bounds_are_the_last_rows() {
    let t = Trio::new();
    let key = |row: u32| (cost(&unpack(row)), unpack(row));
    let rows = t.rows();
    assert_eq!(rows.len(), ROWS);
    for (r, b) in CLASS_BOUNDS.iter().enumerate() {
        let last = rows[CLASS_ROWS * (r + 1) - 1];
        assert_eq!(key(last), (b.cost, b.cut), "class {r}");
        assert_eq!((b.cost, b.cut), ([88, 96][r], [[1, 1, 0, 3, 0, 0, 1, 1], [0, 0, 2, 2, 2, 0, 1, 1]][r]), "class {r} literal");
        // Sorted by (cost, ρ) within the block, every row of the class.
        let block = &rows[CLASS_ROWS * r..CLASS_ROWS * (r + 1)];
        assert!(block.windows(2).all(|w| key(w[0]) < key(w[1])), "class {r}: the block is not in (cost, ρ) order");
        assert!(block.iter().all(|&w| rank_class(&unpack(w)) == r as u32), "class {r}: a row of the other class");
    }
    let mixed: Vec<u32> = rows[..N0_MIXED].iter().chain(&rows[CLASS_ROWS..2 * CLASS_ROWS - N0_MIXED]).copied().collect();
    let top = mixed.iter().map(|&w| key(w)).max().expect("2048 rows");
    assert_eq!(top, (MIXED_BOUND.cost, MIXED_BOUND.cut));
    assert_eq!((MIXED_BOUND.cost, MIXED_BOUND.cut), (72, [3, 0, 0, 0, 1, 0, 1, 0]), "mixed literal");
    // The first rows left out of the mixed order on each side are past it.
    assert!(key(rows[N0_MIXED]) > top && key(rows[2 * CLASS_ROWS - N0_MIXED]) > top);
    assert_eq!(rank_class(&MIXED_BOUND.cut), 0, "the mixed cut is a class-1 vector");
    assert_eq!(pack(&MIXED_BOUND.cut), rows[N0_MIXED - 1], "the mixed cut is not the last class-0 row reached");
}

/// Every bit below 47 moves the point of every word, bit 47 never does: a
/// field read one bit narrow, shifted, or the gain bit leaking into an index
/// would fail here.
#[test]
fn every_label_bit_moves_the_point_and_the_gain_bit_does_not() {
    let t = Trio::new();
    let mut rng = SplitMix64::new(0x7210_2026_0905_0006);
    for _ in 0..200 {
        let w = word(&mut rng);
        let y = t.decode(w);
        for bit in 0..WORD_BITS {
            let moved = t.decode(w ^ (1u64 << bit)) != y;
            assert_eq!(moved, bit < 47, "{w:#014x}: bit {bit} {}", if moved { "moved the point" } else { "changed nothing" });
        }
    }
}
