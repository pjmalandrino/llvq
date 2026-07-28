//! # Gate G2 — exactness of the closed-form search vs brute force
//!
//! The fast search ranks per-class closed-form maxima; the reference
//! enumerates every point of the shell (via `Leech::for_each_shell_point`,
//! itself pinned to the theta series by the G1 suite) and takes the argmax
//! directly. The two share only the Golay tables, so agreement on random
//! Gaussian queries validates the closed forms and the repair arguments.
//!
//! Tolerances: both sides accumulate in f64 with different summation orders,
//! so scores are compared within 1e-9 (queries are O(1)-scaled Gaussians;
//! exact ties have probability ~0 under the seeded sampler).

use llvq_core::{Leech, Point, SplitMix64, DIM};
use llvq_search::{Found, Metric, Searcher};

fn gauss(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next() >> 11) as f64 + 1.0) / 9_007_199_254_740_993.0; // (0,1]
    let u2 = (rng.next() >> 11) as f64 / 9_007_199_254_740_992.0;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn gauss_vec(rng: &mut SplitMix64, scale: f64) -> [f64; DIM] {
    core::array::from_fn(|_| scale * gauss(rng))
}

fn dot(x: &[f64; DIM], v: &Point) -> f64 {
    x.iter().zip(v.iter()).map(|(a, &b)| a * b as f64).sum()
}

/// Brute-force argmax of ⟨x, v⟩ over Shell(m).
fn brute_max_dot(l: &Leech, m: u32, x: &[f64; DIM]) -> (f64, Point) {
    let mut best = f64::NEG_INFINITY;
    let mut best_p: Point = [0; DIM];
    let ok = l.for_each_shell_point(m, |p| {
        let d = dot(x, p);
        if d > best {
            best = d;
            best_p = *p;
        }
    });
    assert!(ok, "shell {m} unsupported");
    (best, best_p)
}

const TOL: f64 = 1e-9;

#[test]
fn shell2_exact_vs_brute_force() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x5EED_0002);
    let queries = if cfg!(debug_assertions) { 8 } else { 300 };
    for _ in 0..queries {
        let x = gauss_vec(&mut rng, 1.0);
        let fast = s.nearest_in_shell2(&x);
        let (brute, _) = brute_max_dot(s.leech(), 2, &x);
        assert!(
            (fast.dot - brute).abs() <= TOL,
            "shell-2 mismatch: fast {} vs brute {brute}",
            fast.dot
        );
        assert!((fast.dot - dot(&x, &fast.point)).abs() <= TOL);
        assert!(s.leech().contains(&fast.point));
    }
}

#[cfg_attr(debug_assertions, ignore = "release-only: streams 16.7M points per query")]
#[test]
fn shell3_and_m3_exact_vs_brute_force() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x5EED_0003);
    for q in 0..12 {
        // Alternate scales so both shells get to win under both metrics.
        let scale = if q % 2 == 0 { 1.0 } else { 0.3 };
        let x = gauss_vec(&mut rng, scale);

        let fast3 = s.nearest_in_shell3(&x);
        let (brute3, _) = brute_max_dot(s.leech(), 3, &x);
        assert!(
            (fast3.dot - brute3).abs() <= TOL,
            "shell-3 mismatch: fast {} vs brute {brute3}",
            fast3.dot
        );

        let (brute2, _) = brute_max_dot(s.leech(), 2, &x);

        // Euclidean reference across the union.
        let fast_e = s.nearest_m3(&x, Metric::Euclidean);
        let brute_e_score = (2.0 * brute2 - 32.0).max(2.0 * brute3 - 48.0);
        assert!(
            ((2.0 * fast_e.dot - (16 * fast_e.shell) as f64) - brute_e_score).abs() <= TOL,
            "euclidean mismatch"
        );

        // Angular reference across the union.
        let fast_a = s.nearest_m3(&x, Metric::Angular);
        let brute_a_score = (brute2 / 32f64.sqrt()).max(brute3 / 48f64.sqrt());
        assert!(
            ((fast_a.dot / ((16 * fast_a.shell) as f64).sqrt()) - brute_a_score).abs() <= TOL,
            "angular mismatch"
        );
    }
}

/// §3.1 of the paper: across shells, Euclidean and angular rankings no
/// longer coincide. With small-magnitude queries the Euclidean metric is
/// pulled toward the smaller-norm shell while the angular metric ignores
/// norms — the two must disagree on at least some queries.
#[test]
fn metrics_disagree_on_some_queries() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x5EED_0004);
    let queries = if cfg!(debug_assertions) { 40 } else { 200 };
    let mut disagreements = 0u32;
    let mut shell2_wins_e = 0u32;
    for _ in 0..queries {
        let x = gauss_vec(&mut rng, 0.25);
        let e = s.nearest_m3(&x, Metric::Euclidean);
        let a = s.nearest_m3(&x, Metric::Angular);
        if e.point != a.point {
            disagreements += 1;
        }
        if e.shell == 2 {
            shell2_wins_e += 1;
        }
    }
    assert!(
        disagreements > 0,
        "Euclidean and angular rankings never disagreed on {queries} queries"
    );
    assert!(shell2_wins_e > 0, "Euclidean never preferred the small shell");
}

/// Throughput report for the G2 gate (target: ≥ 1e5 blocks/s/core for the
/// final encoder; this asserts only a conservative floor and prints the
/// measured figure — run with `--nocapture` to see it).
#[cfg_attr(debug_assertions, ignore = "release-only: timing")]
#[test]
fn throughput_report() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x5EED_0005);
    let queries: Vec<[f64; DIM]> = (0..2_000).map(|_| gauss_vec(&mut rng, 1.0)).collect();
    let mut ws = llvq_search::Workspace::new();
    let t0 = std::time::Instant::now();
    let mut acc = 0.0f64;
    for x in &queries {
        acc += s.nearest_m3_with(&mut ws, x, Metric::Angular).dot;
    }
    let dt = t0.elapsed();
    let qps = queries.len() as f64 / dt.as_secs_f64();
    println!("m3 search throughput: {qps:.0} queries/s/core (acc {acc:.3})");
    assert!(qps > 1_000.0, "throughput collapsed: {qps:.0}/s");
}

/// Every result is a valid lattice point on the declared shell.
#[test]
fn results_are_members() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x5EED_0006);
    for _ in 0..if cfg!(debug_assertions) { 20 } else { 200 } {
        let x = gauss_vec(&mut rng, 1.0);
        for f in [
            s.nearest_in_shell2(&x),
            s.nearest_in_shell3(&x),
            s.nearest_m3(&x, Metric::Euclidean),
            s.nearest_m3(&x, Metric::Angular),
        ] {
            assert!(s.leech().contains(&f.point));
            assert_eq!(Leech::shell_index(&f.point), Some(f.shell as u64));
        }
    }
    let _: fn(&Found, &[f64; DIM]) -> f64 = Found::dist2;
}
