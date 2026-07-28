//! # Gate G1 — invariants of the Golay code and the Leech lattice
//!
//! Every constant asserted here is public mathematics (Conway & Sloane,
//! *Sphere Packings, Lattices and Groups*; OEIS A008408), independent of the
//! LLVQ paper. The kissing number 196 560 and |Shell(3)| = 16 773 120 are
//! reproduced by exhaustive enumeration in which every counted vector is
//! individually verified by the membership predicate.
//!
//! Completeness arguments (why the enumerated shapes are the only ones) are
//! given inline: a shell fixes `Σxᵢ² = 16m` with uniform parity, so the
//! multiset of |coordinates| solves a small partition problem over squares.
//!
//! ## Mutation coverage
//!
//! An adversarial audit (2026-07) showed by mutation testing that the shell
//! enumerations alone do NOT exercise the Golay stage of `Leech::contains`:
//! they build their mod-4 words from genuine codewords, and every wrong-word
//! candidate they generate happens to also violate the sum congruence
//! (doubled Golay weights are ≡ 0 mod 8, so a word at Hamming distance 1–2
//! from a codeword shifts the sum by ±2/±6). `golay_stage_is_load_bearing`
//! exists precisely to close that hole: its probes satisfy parity AND the
//! sum congruence and are rejected *only* by the Golay stage — deleting that
//! stage makes this file fail, not pass.

use llvq_core::rng::SplitMix64;
use llvq_core::{Golay, Leech, Point, DIM};
use llvq_core::leech::THETA;

// ---------------------------------------------------------------------------
// Golay code invariants
// ---------------------------------------------------------------------------

#[test]
fn golay_weight_distribution() {
    let g = Golay::new();
    assert_eq!(g.codewords().len(), 4096);
    let expected: &[(usize, usize)] =
        &[(0, 1), (8, 759), (12, 2576), (16, 759), (24, 1)];
    for w in 0..=24 {
        let want = expected
            .iter()
            .find(|&&(wt, _)| wt == w)
            .map_or(0, |&(_, n)| n);
        assert_eq!(
            g.of_weight(w).len(),
            want,
            "weight-{w} codeword count mismatch"
        );
    }
}

/// The 4096 stored codewords are pairwise distinct — asserted directly
/// (strict ordering within each weight bucket) rather than inferred from
/// linearity: a non-linear encoder bug that duplicated one octad and
/// dropped another would pass every other test in this file.
#[test]
fn golay_codewords_distinct() {
    let g = Golay::new();
    assert_eq!(g.codewords().len(), 4096);
    let mut total = 0;
    for w in 0..=24 {
        let bucket = g.of_weight(w);
        assert!(
            bucket.windows(2).all(|p| p[0] < p[1]),
            "weight-{w} bucket not strictly increasing"
        );
        total += bucket.len();
    }
    assert_eq!(total, 4096);
}

#[test]
fn golay_min_distance_is_8() {
    // Linear code: min distance = min nonzero weight.
    let g = Golay::new();
    let min = g
        .codewords()
        .iter()
        .filter(|&&c| c != 0)
        .map(|c| c.count_ones())
        .min()
        .unwrap();
    assert_eq!(min, 8);
}

#[test]
fn golay_is_self_dual_and_doubly_even() {
    let g = Golay::new();
    // Doubly even: all weights ≡ 0 (mod 4).
    assert!(g.codewords().iter().all(|c| c.count_ones() % 4 == 0));
    // Self-dual: pairwise even intersections. By bilinearity it suffices to
    // check a spanning set; we check all pairs among the 12 encoded unit
    // messages (rows of a generator matrix).
    let rows: Vec<u32> = (0..12)
        .map(|i| {
            let c = llvq_core::golay::clmul(1u64 << i, llvq_core::golay::GEN_POLY as u64) as u32;
            c | ((c.count_ones() & 1) << 23)
        })
        .collect();
    for a in &rows {
        for b in &rows {
            assert_eq!((a & b).count_ones() % 2, 0);
        }
    }
    // Complement closure: 1^24 is a codeword, so c ↦ !c maps the code to itself.
    assert!(g.contains(0xFF_FFFF));
}

// ---------------------------------------------------------------------------
// Leech lattice: kissing number 196 560 (Shell 2, ‖x‖² = 32)
// ---------------------------------------------------------------------------
//
// Completeness of the shapes for ‖x‖² = 32:
//   odd parity:  24·1 + 8a + 24b = 32 with a = #{|xᵢ|=3}, b = #{|xᵢ|=5}
//                → 8a + 24b = 8 → (a,b) = (1,0): shape (±3, ±1²³).
//   even parity: 4a + 16b = 32 with a = #{|xᵢ|=2}, b = #{|xᵢ|=4} (6² = 36 > 32)
//                → (8,0): (±2⁸); (4,1): (±4, ±2⁴) — its Golay word has
//                weight 4, absent from the code, hence empty (asserted);
//                (0,2): (±4²).

/// Shape (±2⁸ 0¹⁶): support must be a Golay octad (the mod-4 word is the
/// support itself, sign-independent); enumerate all octads × 2⁸ signs.
fn count_shape_2e8(l: &Leech) -> u64 {
    let mut n = 0u64;
    for &oct in l.golay().of_weight(8) {
        let pos: Vec<usize> = (0..DIM).filter(|&i| oct >> i & 1 == 1).collect();
        for s in 0u32..256 {
            let mut x: Point = [0; DIM];
            for (j, &p) in pos.iter().enumerate() {
                x[p] = if s >> j & 1 == 1 { -2 } else { 2 };
            }
            if l.contains(&x) {
                n += 1;
            }
        }
    }
    n
}

/// Shape (±4² 0²²): all C(24,2) placements × 4 sign patterns.
fn count_shape_4sq(l: &Leech) -> u64 {
    let mut n = 0u64;
    for i in 0..DIM {
        for j in (i + 1)..DIM {
            for s in 0u32..4 {
                let mut x: Point = [0; DIM];
                x[i] = if s & 1 == 1 { -4 } else { 4 };
                x[j] = if s & 2 == 2 { -4 } else { 4 };
                if l.contains(&x) {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Shape (±3, ±1²³): for a member, the codeword is exactly
/// `{i : xᵢ ≡ 3 (mod 4)}` = {coords equal to −1 or +3}. Enumerating
/// (codeword c, position p of the |3| coordinate, sign of that coordinate)
/// with the remaining coordinates forced (−1 on c, +1 off c) covers every
/// candidate exactly once; the predicate rejects inconsistent combinations
/// (min distance 8 forbids two codewords differing in one bit, so no
/// candidate is reachable from two distinct triples).
fn count_shape_3_1s(l: &Leech) -> u64 {
    let mut n = 0u64;
    for &c in l.golay().codewords() {
        for p in 0..DIM {
            for sign3 in [3i32, -3] {
                let mut x: Point = [0; DIM];
                for (i, xi) in x.iter_mut().enumerate() {
                    *xi = if c >> i & 1 == 1 { -1 } else { 1 };
                }
                x[p] = sign3;
                if l.contains(&x) {
                    n += 1;
                }
            }
        }
    }
    n
}

#[test]
fn leech_kissing_number_196560() {
    let l = Leech::new();
    // The (±4, ±2⁴) shape is empty because no weight-4 Golay codeword exists.
    assert!(l.golay().of_weight(4).is_empty());

    let c2 = count_shape_2e8(&l);
    let c4 = count_shape_4sq(&l);
    let c3 = count_shape_3_1s(&l);
    // Classical decomposition: 759·2⁷ + C(24,2)·4 + 24·4096.
    assert_eq!(c2, 97_152, "(±2⁸) class");
    assert_eq!(c4, 1_104, "(±4²) class");
    assert_eq!(c3, 98_304, "(±3, ±1²³) class");
    assert_eq!(c2 + c4 + c3, 196_560, "kissing number of Λ24");
    // Pin the exported constant to the enumerated count.
    assert_eq!(THETA[0], (2, c2 + c4 + c3));
}

// ---------------------------------------------------------------------------
// The Golay stage of `contains` must be load-bearing
// ---------------------------------------------------------------------------

/// Probes that satisfy parity AND the sum congruence and are rejected ONLY
/// by the Golay stage (see the mutation-coverage note in the file header).
/// Each probe re-asserts its own sum congruence so that a future edit cannot
/// silently turn it back into a sum-rejected candidate.
#[test]
fn golay_stage_is_load_bearing() {
    let l = Leech::new();
    let sum = |x: &Point| x.iter().map(|&v| v as i64).sum::<i64>().rem_euclid(8);

    // Even probe 1: (2,2,2,2,0²⁰) — word weight 4 ∉ Golay, sum 8 ≡ 0 (mod 8).
    let mut x: Point = [0; DIM];
    for xi in x.iter_mut().take(4) {
        *xi = 2;
    }
    assert_eq!(sum(&x), 0);
    assert!(!l.contains(&x), "(2⁴) must be Golay-rejected");

    // Even probe 2: eight 2s on a weight-8 support that is NOT an octad
    // (sum 16 ≡ 0 mod 8). Find the first such support deterministically.
    let support = (0u32..(1 << 24))
        .find(|&b| b.count_ones() == 8 && !l.golay().contains(b))
        .expect("some weight-8 non-codeword must exist");
    let mut y: Point = [0; DIM];
    for (i, yi) in y.iter_mut().enumerate() {
        if support >> i & 1 == 1 {
            *yi = 2;
        }
    }
    assert_eq!(sum(&y), 0);
    assert!(!l.contains(&y), "non-octad (2⁸) must be Golay-rejected");

    // Odd probe 1: 1²⁴ with two −1 — word weight 2 ∉ Golay,
    // sum 24 − 4 = 20 ≡ 4 (mod 8).
    let mut z: Point = [1; DIM];
    z[0] = -1;
    z[1] = -1;
    assert_eq!(sum(&z), 4);
    assert!(!l.contains(&z), "two-(−1)s probe must be Golay-rejected");

    // Odd probe 2: 1²⁴ with four −1 and one +5 — word weight 4 ∉ Golay,
    // sum 24 − 8 + 4 = 20 ≡ 4 (mod 8).
    let mut w: Point = [1; DIM];
    for wi in w.iter_mut().take(4) {
        *wi = -1;
    }
    w[4] = 5;
    assert_eq!(sum(&w), 4);
    assert!(!l.contains(&w), "weight-4-word odd probe must be Golay-rejected");
}

// ---------------------------------------------------------------------------
// Shell 4 (‖x‖² = 64): class-level spot checks
// ---------------------------------------------------------------------------

#[test]
fn leech_shell4_spot_checks() {
    let l = Leech::new();

    // (±8, 0²³): word 0 ∈ Golay, sum ±8 ≡ 0 — all 48 candidates are members.
    let mut c8 = 0u64;
    for i in 0..DIM {
        for v in [8i32, -8] {
            let mut x: Point = [0; DIM];
            x[i] = v;
            assert_eq!(Leech::shell_index(&x), Some(4));
            if l.contains(&x) {
                c8 += 1;
            }
        }
    }
    assert_eq!(c8, 48, "(±8) class of Shell(4)");

    // (±4⁴, 0²⁰): word 0 ∈ Golay; every sign flip changes the sum by 8, so
    // all 2⁴ patterns pass → C(24,4)·16 = 170 016 (matches paper Table 2).
    let mut c44 = 0u64;
    for i in 0..DIM {
        for j in (i + 1)..DIM {
            for k in (j + 1)..DIM {
                for m in (k + 1)..DIM {
                    for s in 0u32..16 {
                        let mut x: Point = [0; DIM];
                        for (b, &pos) in [i, j, k, m].iter().enumerate() {
                            x[pos] = if s >> b & 1 == 1 { -4 } else { 4 };
                        }
                        if l.contains(&x) {
                            c44 += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(c44, 170_016, "(±4⁴) class of Shell(4) = C(24,4)·16");
}

// ---------------------------------------------------------------------------
// Leech lattice: no nonzero vector below ‖x‖² = 32
// ---------------------------------------------------------------------------

#[test]
fn leech_min_norm_is_32() {
    let l = Leech::new();

    // Even candidates with ‖x‖² < 32: shapes (±2^a) a ∈ 1..=7 and
    // (±4, ±2^a) a ∈ 0..=3. For a ≥ 1 the Golay word is the ±2 support of
    // weight a ∈ 1..=7 — no such codeword exists:
    for w in 1..=7 {
        assert!(l.golay().of_weight(w).is_empty());
    }
    // Remaining even candidate: a single ±4 (word = 0 ∈ Golay) — must be
    // rejected by the sum constraint (±4 ≢ 0 mod 8).
    for i in 0..DIM {
        for v in [4i32, -4] {
            let mut x: Point = [0; DIM];
            x[i] = v;
            assert!(!l.contains(&x), "±4·e_{i} must not be in Λ24");
        }
    }
    // Odd candidates with ‖x‖² < 32: only (±1²⁴) (norm 24). A member's
    // codeword is its (−1)-support, so only 4096 candidates can pass the
    // Golay stage; all must fail the sum constraint (24 − 2w ≢ 4 mod 8 for
    // w ∈ {0, 8, 12, 16, 24}).
    for &c in l.golay().codewords() {
        let mut x: Point = [0; DIM];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = if c >> i & 1 == 1 { -1 } else { 1 };
        }
        assert!(!l.contains(&x), "norm-24 vector must not be in Λ24");
    }
}

// ---------------------------------------------------------------------------
// Leech lattice: group structure and known members
// ---------------------------------------------------------------------------

#[test]
fn leech_additive_closure() {
    let l = Leech::new();
    let mut rng = SplitMix64::new(0x115C_2026);
    for _ in 0..10_000 {
        let a = l.random_point(&mut rng, 2);
        let b = l.random_point(&mut rng, 2);
        assert!(l.contains(&a));
        assert!(l.contains(&b));
        assert!(l.contains(&Leech::add(&a, &b)), "closure under +");
        assert!(l.contains(&Leech::neg(&a)), "closure under negation");
        assert!(
            l.contains(&Leech::add(&a, &Leech::neg(&b))),
            "closure under −"
        );
    }
}

#[test]
fn leech_known_members_and_non_members() {
    let l = Leech::new();
    let e = |i: usize, v: i32| {
        let mut x: Point = [0; DIM];
        x[i] = v;
        x
    };
    // 8·e_i ∈ Λ (shell 4 class (±8, 0²³), 48 vectors).
    assert!(l.contains(&e(0, 8)));
    // 4·e_i ∉ Λ (fails the sum constraint).
    assert!(!l.contains(&e(0, 4)));
    // (4, 4, 0…) ∈ Λ; (4, −4, 0…) ∈ Λ.
    let mut x: Point = [0; DIM];
    x[0] = 4;
    x[1] = 4;
    assert!(l.contains(&x));
    x[1] = -4;
    assert!(l.contains(&x));
    // (2, 2, 0…) ∉ Λ (weight-2 word not in Golay).
    let mut y: Point = [0; DIM];
    y[0] = 2;
    y[1] = 2;
    assert!(!l.contains(&y));
    // Zero vector ∈ Λ.
    assert!(l.contains(&[0; DIM]));
    // Mixed parity ∉ Λ.
    let mut z: Point = [1; DIM];
    z[0] = 2;
    assert!(!l.contains(&z));
}

#[test]
fn leech_scaling_doubles_shell_index() {
    // 2x has norm 4·16m = 16·(4m): doubling a shell-2 vector lands on a
    // shell-8 radius and must stay in the lattice.
    let l = Leech::new();
    let mut x: Point = [0; DIM];
    x[0] = 4;
    x[1] = 4;
    assert_eq!(Leech::shell_index(&x), Some(2));
    let xx = Leech::add(&x, &x);
    assert_eq!(Leech::shell_index(&xx), Some(8));
    assert!(l.contains(&xx));
}

// ---------------------------------------------------------------------------
// Streaming shell enumerator (`Leech::for_each_shell_point`)
// ---------------------------------------------------------------------------
//
// The enumerator constructs members directly (no predicate filtering), so it
// is a THIRD implementation of the shell structure — cross-checked here
// against the theta series, the membership predicate, and (for Shell 2,
// where memory allows) pairwise distinctness.

#[test]
fn shell_enumerator_shell2() {
    let l = Leech::new();
    let mut seen = std::collections::HashSet::new();
    let mut n = 0u64;
    let ok = l.for_each_shell_point(2, |p| {
        assert!(l.contains(p), "enumerator emitted a non-member");
        assert_eq!(Leech::shell_index(p), Some(2));
        assert!(seen.insert(*p), "enumerator emitted a duplicate");
        n += 1;
    });
    assert!(ok);
    assert_eq!(n, THETA[0].1);
    assert!(!l.for_each_shell_point(4, |_| {}), "shell 4 must be unsupported");
}

#[cfg_attr(debug_assertions, ignore = "release-only: 16.7M membership checks")]
#[test]
fn shell_enumerator_shell3() {
    let l = Leech::new();
    let mut n = 0u64;
    let ok = l.for_each_shell_point(3, |p| {
        assert!(l.contains(p), "enumerator emitted a non-member");
        assert_eq!(Leech::shell_index(p), Some(3));
        n += 1;
    });
    assert!(ok);
    assert_eq!(n, THETA[1].1);
}

// ---------------------------------------------------------------------------
// Shell 3 (‖x‖² = 48): |Shell(3)| = 16 773 120 — release builds only
// ---------------------------------------------------------------------------
//
// Shape completeness for ‖x‖² = 48:
//   odd:  24 + 8a + 24b = 48 → (a,b) ∈ {(3,0), (0,1)}:
//         (±3³, ±1²¹) and (±5, ±1²³).
//   even: 4a + 16b + 36c = 48 →
//         c=0: (12,0): (±2¹²); (8,1): (±4, ±2⁸); (4,2): (±4², ±2⁴) — Golay
//              word weight 4, empty; (0,3): (±4³) — word 0 but sum ≡ 4 mod 8,
//              empty (asserted by enumeration below);
//         c=1: (3,0): (±6, ±2³) — word weight 4, empty.

#[cfg_attr(debug_assertions, ignore = "release-only: ~25M membership checks")]
#[test]
fn leech_shell3_cardinality() {
    let l = Leech::new();
    let mut total = 0u64;

    // (±2¹²): support must be a Golay dodecad.
    for &dodecad in l.golay().of_weight(12) {
        let pos: Vec<usize> = (0..DIM).filter(|&i| dodecad >> i & 1 == 1).collect();
        for s in 0u32..4096 {
            let mut x: Point = [0; DIM];
            for (j, &p) in pos.iter().enumerate() {
                x[p] = if s >> j & 1 == 1 { -2 } else { 2 };
            }
            if l.contains(&x) {
                total += 1;
            }
        }
    }
    assert_eq!(total, 5_275_648, "(±2¹²) class = 2576·2¹¹");

    // (±4, ±2⁸): octad support for the 2s, the ±4 outside the octad.
    // On-octad positions are SKIPPED (`continue` below), not predicate-
    // rejected: a ±4 there would overwrite a ±2 and give norm 44 ∉ 16Z —
    // a candidate of the wrong shape — so skipping cannot lose any member.
    let mut c48 = 0u64;
    for &oct in l.golay().of_weight(8) {
        let pos: Vec<usize> = (0..DIM).filter(|&i| oct >> i & 1 == 1).collect();
        for p4 in 0..DIM {
            for s in 0u32..512 {
                let mut x: Point = [0; DIM];
                for (j, &p) in pos.iter().enumerate() {
                    x[p] = if s >> j & 1 == 1 { -2 } else { 2 };
                }
                if pos.contains(&p4) {
                    continue; // overwritten coordinate would leave norm ≠ 48
                }
                x[p4] = if s >> 8 & 1 == 1 { -4 } else { 4 };
                if l.contains(&x) {
                    c48 += 1;
                }
            }
        }
    }
    assert_eq!(c48, 3_108_864, "(±4, ±2⁸) class = 759·16·2·2⁷");
    total += c48;

    // (±4³): word 0 ∈ Golay but every sign pattern violates the sum rule.
    let mut c43 = 0u64;
    for i in 0..DIM {
        for j in (i + 1)..DIM {
            for k in (j + 1)..DIM {
                for s in 0u32..8 {
                    let mut x: Point = [0; DIM];
                    x[i] = if s & 1 == 1 { -4 } else { 4 };
                    x[j] = if s & 2 == 2 { -4 } else { 4 };
                    x[k] = if s & 4 == 4 { -4 } else { 4 };
                    if l.contains(&x) {
                        c43 += 1;
                    }
                }
            }
        }
    }
    assert_eq!(c43, 0, "(±4³) must be empty");

    // (±5, ±1²³): same parametrization as shape (±3, ±1²³).
    let mut c5 = 0u64;
    for &c in l.golay().codewords() {
        for p in 0..DIM {
            for sign5 in [5i32, -5] {
                let mut x: Point = [0; DIM];
                for (i, xi) in x.iter_mut().enumerate() {
                    *xi = if c >> i & 1 == 1 { -1 } else { 1 };
                }
                x[p] = sign5;
                if l.contains(&x) {
                    c5 += 1;
                }
            }
        }
    }
    assert_eq!(c5, 98_304, "(±5, ±1²³) class = 24·4096");
    total += c5;

    // (±3³, ±1²¹): codeword c + a 3-subset T; signs forced by congruences
    // (+3 on T∩c, −3 on T∖c, −1 on c∖T, +1 elsewhere).
    let mut c33 = 0u64;
    for &c in l.golay().codewords() {
        for i in 0..DIM {
            for j in (i + 1)..DIM {
                for k in (j + 1)..DIM {
                    let mut x: Point = [0; DIM];
                    for (t, xt) in x.iter_mut().enumerate() {
                        *xt = if c >> t & 1 == 1 { -1 } else { 1 };
                    }
                    for &t in &[i, j, k] {
                        x[t] = if c >> t & 1 == 1 { 3 } else { -3 };
                    }
                    if l.contains(&x) {
                        c33 += 1;
                    }
                }
            }
        }
    }
    assert_eq!(c33, 8_290_304, "(±3³, ±1²¹) class = 4096·C(24,3)");
    total += c33;

    assert_eq!(total, 16_773_120, "|Shell(3)| — theta series coefficient");
    // Pin the exported constant to the enumerated count.
    assert_eq!(THETA[1], (3, total));
}
