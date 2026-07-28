//! # Gate G2b — the generic class engine (shells m ≤ 13)
//!
//! Three independent validations:
//!
//! 1. **Consistency with the fast path**: on shells 2 and 3 the generic
//!    engine must reproduce the scores of the hand-derived closed forms —
//!    which were themselves validated against brute-force enumeration in
//!    `g2_search.rs`. This transitively pins the generic engine to the
//!    ground truth on the shells where ground truth is enumerable.
//! 2. **DP reference for the even parity repair**: the greedy
//!    sacrifice-and-repack rule is checked against an exhaustive assignment
//!    DP (positions × value-multiset × sign parity) on random inputs, per
//!    (class, codeword) — including the classes of high shells that no
//!    enumeration can reach.
//! 3. **Membership**: every materialized winner must pass `Leech::contains`
//!    on its declared shell radius with a consistent dot product.

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::generic::{BallSearcher, NSHELLS};
use llvq_search::{Searcher, Workspace};

fn gauss_vec(rng: &mut SplitMix64, scale: f64) -> [f64; DIM] {
    core::array::from_fn(|_| scale * rng.next_gaussian())
}

fn dot(x: &[f64; DIM], v: &llvq_core::Point) -> f64 {
    x.iter().zip(v.iter()).map(|(a, &b)| a * b as f64).sum()
}

const TOL: f64 = 1e-9;

#[test]
fn generic_matches_fast_path_on_shells_2_and_3() {
    let s = Searcher::new();
    let mut ball = BallSearcher::new();
    let mut ws = Workspace::new();
    let mut rng = SplitMix64::new(0x2B_0001);
    let n = if cfg!(debug_assertions) { 10 } else { 200 };
    for _ in 0..n {
        let x = gauss_vec(&mut rng, 1.0);
        let bests = ball.shell_bests(&s, &mut ws, &x);
        let f2 = s.nearest_in_shell2_with(&mut ws, &x);
        let f3 = s.nearest_in_shell3_with(&mut ws, &x);
        assert!(
            (bests[0].dot - f2.dot).abs() <= TOL,
            "shell 2: generic {} vs fast {}",
            bests[0].dot,
            f2.dot
        );
        assert!(
            (bests[1].dot - f3.dot).abs() <= TOL,
            "shell 3: generic {} vs fast {}",
            bests[1].dot,
            f3.dot
        );
    }
}

#[test]
fn winners_are_members_with_consistent_dots() {
    let s = Searcher::new();
    let l = Leech::new();
    let mut ball = BallSearcher::new();
    let mut ws = Workspace::new();
    let mut rng = SplitMix64::new(0x2B_0002);
    let n = if cfg!(debug_assertions) { 5 } else { 60 };
    for q in 0..n {
        // Vary the scale so high shells win sometimes (large ‖x‖ pulls the
        // angular-per-shell argmax toward wider leaders).
        let scale = 0.5 + 0.25 * (q % 8) as f64;
        let x = gauss_vec(&mut rng, scale);
        for f in ball.shell_bests(&s, &mut ws, &x) {
            assert!(l.contains(&f.point), "shell {} winner not in Λ24", f.shell);
            assert_eq!(
                Leech::shell_index(&f.point),
                Some(f.shell as u64),
                "winner on wrong shell radius"
            );
            assert!(
                (f.dot - dot(&x, &f.point)).abs() <= TOL,
                "shell {}: reported dot {} vs recomputed {}",
                f.shell,
                f.dot,
                dot(&x, &f.point)
            );
        }
    }
    let _ = NSHELLS;
}

// ---------------------------------------------------------------------------
// DP reference for the even-class on-support maximizer
// ---------------------------------------------------------------------------
//
// Exhaustive assignment DP: assign word values (multiset) to support slots
// sorted by |x| descending, choosing a sign for each, tracking minus-sign
// parity. State: (slot index, remaining counts per kind, parity). This is
// exponential-free and exact; the greedy closed form must match it.

fn dp_on_support(values: &[(f64, usize)], sup: &[f64], p_req: u32) -> f64 {
    // memo over (slot, counts…, parity) — counts encoded in a small radix.
    fn rec(
        slot: usize,
        counts: &mut Vec<usize>,
        parity: u32,
        values: &[(f64, usize)],
        sup: &[f64],
        p_req: u32,
        memo: &mut std::collections::HashMap<(usize, Vec<usize>, u32), f64>,
    ) -> f64 {
        if slot == sup.len() {
            return if parity == p_req { 0.0 } else { f64::NEG_INFINITY };
        }
        let key = (slot, counts.clone(), parity);
        if let Some(&v) = memo.get(&key) {
            return v;
        }
        let mut best = f64::NEG_INFINITY;
        for k in 0..values.len() {
            if counts[k] == 0 {
                continue;
            }
            counts[k] -= 1;
            let v = values[k].0;
            // sign +
            let r =
                v * sup[slot] + rec(slot + 1, counts, parity, values, sup, p_req, memo);
            // sign −
            let r2 = -v * sup[slot]
                + rec(slot + 1, counts, parity ^ 1, values, sup, p_req, memo);
            counts[k] += 1;
            best = best.max(r).max(r2);
        }
        memo.insert(key, best);
        best
    }
    let mut counts: Vec<usize> = values.iter().map(|&(_, n)| n).collect();
    let mut memo = std::collections::HashMap::new();
    rec(0, &mut counts, 0, values, sup, p_req, &mut memo)
}

/// Greedy closed form under test (mirrors generic.rs): pair values desc with
/// |x| desc, signs matched; on parity mismatch, sacrifice one value of some
/// kind at the min-|x| slot and repack.
fn greedy_on_support(values_desc: &[f64], kinds: &[(f64, usize)], sup_desc: &[f64], matched_parity: u32, p_req: u32) -> f64 {
    let w = sup_desc.len();
    let base: f64 = values_desc.iter().zip(sup_desc).map(|(v, a)| v * a).sum();
    if matched_parity == p_req {
        return base;
    }
    let mut bestrep = f64::NEG_INFINITY;
    let mut d = 0.0;
    // kinds as (value, last index) with last increasing; walk j downward.
    let mut k_iter = kinds.len();
    let mut j = w;
    while j > 0 {
        j -= 1;
        while k_iter > 0 && kinds[k_iter - 1].1 == j {
            let u = kinds[k_iter - 1].0;
            let cand = base - u * sup_desc[j] + d - u * sup_desc[w - 1];
            bestrep = bestrep.max(cand);
            k_iter -= 1;
        }
        if j > 0 {
            d += values_desc[j] * (sup_desc[j - 1] - sup_desc[j]);
        }
    }
    bestrep
}

#[test]
fn even_repair_matches_dp_reference() {
    let mut rng = SplitMix64::new(0x2B_0003);
    // Representative word-value multisets from real classes of shells
    // 2..13, including multi-kind ones where the repair is subtle.
    let cases: &[&[(f64, usize)]] = &[
        &[(2.0, 8)],
        &[(2.0, 12)],
        &[(2.0, 16)],
        &[(6.0, 1), (2.0, 7)],
        &[(6.0, 2), (2.0, 6)],
        &[(6.0, 1), (2.0, 11)],
        &[(10.0, 1), (2.0, 7)],
        &[(6.0, 3), (2.0, 5)],
        &[(10.0, 1), (6.0, 1), (2.0, 6)],
    ];
    for &vals in cases {
        let w: usize = vals.iter().map(|&(_, n)| n).sum();
        for trial in 0..40 {
            let mut sup: Vec<f64> = (0..w).map(|_| rng.next_gaussian().abs()).collect();
            sup.sort_unstable_by(|a, b| b.total_cmp(a));
            // Both parities of the requirement; matched parity 0 (all-plus
            // baseline, since the DP works on |x| slots directly).
            for p_req in [0u32, 1] {
                let dp = dp_on_support(vals, &sup, p_req);
                let mut values_desc = Vec::new();
                let mut kinds = Vec::new();
                for &(v, n) in vals {
                    for _ in 0..n {
                        values_desc.push(v);
                    }
                    kinds.push((v, values_desc.len() - 1));
                }
                let greedy = greedy_on_support(&values_desc, &kinds, &sup, 0, p_req);
                assert!(
                    (dp - greedy).abs() <= 1e-9,
                    "case {vals:?} trial {trial} p_req {p_req}: dp {dp} vs greedy {greedy}"
                );
            }
        }
    }
}
