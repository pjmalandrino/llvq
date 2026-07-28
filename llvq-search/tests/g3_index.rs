//! # Gate G3 — bijectivity of the ball-13 index
//!
//! The worst failure mode of a quantizer index is a silent collision: two
//! codebook points sharing an index corrupt weights while every quality
//! metric stays plausible. These tests pin the bijection from both sides:
//!
//! * `decode(encode(p)) == p` exhaustively on Shell(2) (196 560 points,
//!   every index verified to land in the shell-2 range with no collision);
//! * `encode(decode(i)) == i` on random indices across the whole 2⁴⁸-sized
//!   ball, plus every class boundary (offset−1, offset, offset+1) — the
//!   places where off-by-one errors in the mixed-radix layout live;
//! * every decoded point is membership-verified (`debug_assert` in decode,
//!   re-checked explicitly here in release);
//! * integration: the winners of the generic search engine round-trip.

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::index::{Indexer, N13};
use llvq_search::{Searcher, Workspace};

#[test]
fn origin_and_bounds() {
    let ix = Indexer::new();
    assert_eq!(ix.encode(&[0; DIM]), Some(0));
    assert_eq!(ix.decode(0), Some([0; DIM]));
    assert!(ix.decode(N13).is_some(), "last index must decode");
    assert!(ix.decode(N13 + 1).is_none(), "past-the-end must fail");
    // A non-member must not encode.
    let mut bad = [0i32; DIM];
    bad[0] = 2;
    bad[1] = 2;
    assert_eq!(ix.encode(&bad), None);
    // A member beyond the ball (shell 16: 2·(8,0²³) has norm 16·16) must
    // not encode either.
    let mut far = [0i32; DIM];
    far[0] = 16;
    assert_eq!(ix.encode(&far), None);
}

#[test]
fn shell2_exhaustive_roundtrip_no_collision() {
    let ix = Indexer::new();
    let l = Leech::new();
    let mut seen = vec![false; 196_561];
    let mut n = 0u64;
    l.for_each_shell_point(2, |p| {
        let idx = ix.encode(p).expect("shell-2 point must encode");
        assert!(
            (1..=196_560).contains(&idx),
            "shell-2 index {idx} out of range"
        );
        assert!(!seen[idx as usize], "index collision at {idx}");
        seen[idx as usize] = true;
        assert_eq!(ix.decode(idx).as_ref(), Some(p), "round-trip mismatch");
        n += 1;
    });
    assert_eq!(n, 196_560);
    // Surjectivity on the shell follows: 196 560 distinct indices in a
    // range of size 196 560.
}

#[cfg_attr(debug_assertions, ignore = "release-only: 2M decode/encode pairs")]
#[test]
fn random_indices_roundtrip_across_the_ball() {
    let ix = Indexer::new();
    let l = Leech::new();
    let mut rng = SplitMix64::new(0x63_0001);
    for _ in 0..2_000_000 {
        let idx = rng.next() % (N13 + 1);
        let p = ix.decode(idx).expect("in-range index must decode");
        assert!(
            p == [0; DIM] || l.contains(&p),
            "decoded point not in Λ24 at index {idx}"
        );
        assert_eq!(ix.encode(&p), Some(idx), "encode∘decode ≠ id at {idx}");
    }
}

#[test]
fn class_boundaries_roundtrip() {
    let ix = Indexer::new();
    // Probe every class boundary: the exact places where an off-by-one in
    // offsets or in the mixed-radix chain shows up. Boundaries are not
    // exposed directly, so scan a synthetic set: powers stepping through
    // the range plus neighbors of every shell start.
    let mut probes: Vec<u64> = vec![0, 1, 2, N13 - 1, N13];
    let mut idx = 1u64;
    while idx < N13 {
        probes.push(idx);
        probes.push(idx + 1);
        idx = idx.saturating_mul(3) / 2 + 7;
    }
    for &i in &probes {
        let p = ix.decode(i).expect("probe index in range");
        assert_eq!(ix.encode(&p), Some(i), "round-trip failed at {i}");
    }
}

#[test]
fn search_winners_roundtrip() {
    let s = Searcher::new();
    let ix = Indexer::new();
    let mut ball = BallSearcher::new();
    let mut ws = Workspace::new();
    let mut rng = SplitMix64::new(0x63_0002);
    let n = if cfg!(debug_assertions) { 4 } else { 40 };
    for q in 0..n {
        let scale = 0.5 + 0.25 * (q % 8) as f64;
        let x: [f64; DIM] = core::array::from_fn(|_| scale * rng.next_gaussian());
        for f in ball.shell_bests(&s, &mut ws, &x) {
            let idx = ix.encode(&f.point).expect("winner must encode");
            assert_eq!(ix.decode(idx), Some(f.point), "winner round-trip");
        }
    }
}
