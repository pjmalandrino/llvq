//! # Phase 2c — the fast engine must equal the naive one, exactly
//!
//! [`crate::generic`] is optimized for throughput (bounds, sorted early
//! exits, prefix sums, incremental sorts). Every one of those tricks is an
//! opportunity to prune a winner by mistake. This suite pins the fast engine
//! to [`llvq_search::generic_ref`], which computes the same maxima with no
//! tricks at all — and which leaves the even parity repair *unconstrained*
//! (every slot tried), so it can only ever score higher, never lower.
//!
//! Equality is required to the last ulp modulo summation order (1e-9), on
//! all twelve shells, across scales where different shells win.

use llvq_core::{SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::generic_ref::RefBallSearcher;
use llvq_search::Searcher;

fn gauss_vec(rng: &mut SplitMix64, scale: f64) -> [f64; DIM] {
    core::array::from_fn(|_| scale * rng.next_gaussian())
}

const TOL: f64 = 1e-9;

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "reference engine is O(seconds)/query; run in release"
)]
fn fast_engine_matches_naive_reference() {
    let s = Searcher::new();
    let mut fast = BallSearcher::new();
    let slow = RefBallSearcher::new();
    let mut rng = SplitMix64::new(0x2C_0001);

    // The count is set by mutation testing, not by taste: the even parity
    // repair only bites when a *multi-kind* word class wins on a codeword
    // whose sign parity is wrong, and the suffix-carry term inside it only
    // bites when the sacrificed kind is not the last run. Eight queries let
    // a zeroed carry survive; forty do not.
    for q in 0..40 {
        // Sweep the scale: the winning shell moves with ‖x‖, so this
        // exercises the pruning at both ends of the ball.
        let scale = 0.4 + 0.3 * (q % 6) as f64;
        let x = gauss_vec(&mut rng, scale);
        let a = fast.shell_bests(&s, &x);
        let b = slow.shell_dots(&s, &x);
        for m in 0..12 {
            assert!(
                (a[m].dot - b[m]).abs() <= TOL,
                "query {q} (scale {scale}), shell {}: fast {} vs reference {}",
                m + 2,
                a[m].dot,
                b[m]
            );
        }
    }
}

/// Degenerate inputs where sorts tie and the pruning bounds are equalities:
/// all-equal coordinates, zeros, and a single spike.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "reference engine is O(seconds)/query; run in release"
)]
fn fast_engine_matches_reference_on_degenerate_inputs() {
    let s = Searcher::new();
    let mut fast = BallSearcher::new();
    let slow = RefBallSearcher::new();

    let mut cases: Vec<[f64; DIM]> = vec![
        [1.0; DIM],
        core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 }),
        core::array::from_fn(|i| if i == 0 { 5.0 } else { 0.0 }),
        core::array::from_fn(|i| if i < 8 { 1.0 } else { 0.0 }),
    ];
    // Half the coordinates equal, half distinct — ties inside one sort only.
    cases.push(core::array::from_fn(|i| {
        if i < 12 {
            0.7
        } else {
            0.1 * (i as f64 - 11.0)
        }
    }));

    for (q, x) in cases.iter().enumerate() {
        let a = fast.shell_bests(&s, x);
        let b = slow.shell_dots(&s, x);
        for m in 0..12 {
            assert!(
                (a[m].dot - b[m]).abs() <= TOL,
                "degenerate case {q}, shell {}: fast {} vs reference {}",
                m + 2,
                a[m].dot,
                b[m]
            );
        }
    }
}

/// The encoder path prunes across shells with a single β-objective
/// threshold, and skips whole Golay-weight groups. It must still land on
/// exactly the same codeword as ranking the twelve per-shell maxima by hand
/// — which `fast_engine_matches_naive_reference` pins to the naive engine.
#[test]
fn nearest_scaled_matches_the_per_shell_pass() {
    let s = Searcher::new();
    let leech = llvq_core::Leech::new();
    let mut ball = BallSearcher::new();
    let mut rng = SplitMix64::new(0x2C_0002);

    // β well below the optimum makes the origin win; well above it pushes
    // the winner to the outer shells. Both ends must behave.
    for &beta in &[0.05, 0.2, 0.35, 0.5, 0.9, 1.6] {
        let key = |dot: f64, shell: u32| 2.0 * beta * dot - beta * beta * (16 * shell) as f64;
        for q in 0..24 {
            let scale = 0.4 + 0.3 * (q % 6) as f64;
            let x = gauss_vec(&mut rng, scale);
            let f = ball.nearest_scaled(&s, &x, beta);
            // The origin is a codeword too, and its key is exactly 0.
            let want = ball
                .shell_bests(&s, &x)
                .iter()
                .map(|b| key(b.dot, b.shell))
                .fold(0.0, f64::max);
            assert!(
                (key(f.dot, f.shell) - want).abs() <= TOL,
                "β {beta}, query {q}: encoder key {} vs per-shell best {want}",
                key(f.dot, f.shell)
            );
            if f.shell == 0 {
                assert_eq!(f.point, [0; DIM], "origin winner must be the zero point");
                assert_eq!(f.dot, 0.0);
            } else {
                assert!(leech.contains(&f.point), "winner not in Λ24");
                assert_eq!(llvq_core::Leech::shell_index(&f.point), Some(f.shell as u64));
                let recomputed: f64 =
                    x.iter().zip(f.point.iter()).map(|(a, &b)| a * b as f64).sum();
                assert!((f.dot - recomputed).abs() <= TOL);
            }
        }
    }
}

/// The shape quantizer `Q_dir` of the paper's Algorithm 1: the angular
/// objective has no origin candidate and its bound is `B/‖v‖` rather than an
/// affine function of the dot, so it exercises the pruning differently from
/// the Euclidean path. It must still land on the codeword that ranking the
/// twelve per-shell maxima by `dot/√(16m)` picks by hand.
#[test]
fn nearest_angular_matches_the_per_shell_pass() {
    let s = Searcher::new();
    let leech = llvq_core::Leech::new();
    let mut ball = BallSearcher::new();
    let mut rng = SplitMix64::new(0x2C_0003);

    for q in 0..40 {
        // The angular winner is scale-invariant in x, so sweep the shape of
        // the block instead: near-uniform blocks and spiky ones select very
        // different shells.
        let mut x = gauss_vec(&mut rng, 1.0);
        match q % 4 {
            1 => x[0] *= 6.0,
            2 => x.iter_mut().for_each(|v| *v = v.signum() * v.abs().sqrt()),
            3 => x.iter_mut().take(20).for_each(|v| *v *= 0.05),
            _ => {}
        }
        let f = ball.nearest_angular(&s, &x);
        let want = ball
            .shell_bests(&s, &x)
            .iter()
            .map(|b| b.dot / ((16 * b.shell) as f64).sqrt())
            .fold(f64::NEG_INFINITY, f64::max);
        let got = f.dot / ((16 * f.shell) as f64).sqrt();
        assert!(
            (got - want).abs() <= TOL,
            "query {q}: angular encoder {got} vs per-shell best {want}"
        );
        assert!(f.shell >= 2, "a direction is always required, got shell 0");
        assert!(leech.contains(&f.point), "winner not in Λ24");
        assert_eq!(llvq_core::Leech::shell_index(&f.point), Some(f.shell as u64));
        let recomputed: f64 = x.iter().zip(f.point.iter()).map(|(a, &b)| a * b as f64).sum();
        assert!((f.dot - recomputed).abs() <= TOL);
    }

    // Scale invariance: the shape quantizer must ignore ‖x‖ entirely.
    let x = gauss_vec(&mut rng, 1.0);
    let a = ball.nearest_angular(&s, &x);
    let scaled: [f64; DIM] = core::array::from_fn(|i| 37.5 * x[i]);
    let b = ball.nearest_angular(&s, &scaled);
    assert_eq!(a.point, b.point, "Q_dir must be invariant to positive scaling");
}

/// Capping the shell must restrict the codebook and nothing else: the winner
/// has to be the best point of `Λ₂₄(cap)`, which is exactly the per-shell
/// maximum restricted to `2..=cap`.
#[test]
fn shell_cap_restricts_the_codebook_exactly() {
    let s = Searcher::new();
    let leech = llvq_core::Leech::new();
    let mut ball = BallSearcher::new();
    let mut rng = SplitMix64::new(0x2C_0004);

    for cap in [2u32, 5, 12, 13] {
        ball.set_shell_cap(cap);
        assert_eq!(ball.shell_cap(), cap);
        for q in 0..12 {
            let x = gauss_vec(&mut rng, 0.4 + 0.3 * (q % 6) as f64);
            let f = ball.nearest_angular(&s, &x);
            assert!(f.shell <= cap, "returned shell {} above cap {cap}", f.shell);
            assert!(leech.contains(&f.point));

            // Ground truth: the per-shell maxima, ranked over 2..=cap only.
            let mut want = f64::NEG_INFINITY;
            for b in ball.shell_bests(&s, &x) {
                if b.shell <= cap {
                    want = want.max(b.dot / ((16 * b.shell) as f64).sqrt());
                }
            }
            let got = f.dot / ((16 * f.shell) as f64).sqrt();
            assert!(
                (got - want).abs() <= TOL,
                "cap {cap}, query {q}: capped encoder {got} vs restricted best {want}"
            );
        }
    }
    // And the cap must survive: an uncapped searcher may still reach shell 13.
    ball.set_shell_cap(13);
    let mut reached = false;
    for q in 0..40 {
        let x = gauss_vec(&mut rng, 0.5 + 0.2 * (q % 9) as f64);
        if ball.nearest_angular(&s, &x).shell > 12 {
            reached = true;
            break;
        }
    }
    assert!(reached, "shell 13 unreachable even uncapped — the cap leaked");
}
