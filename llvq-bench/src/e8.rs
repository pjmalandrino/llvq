//! E8, for the arbitration of 2026-09-18: enumeration, exact nearest
//! neighbour, and the invariants that say this code is the lattice.
//!
//! ## Why this exists at all
//!
//! `docs/arbitrage-e8-2026-09-18.md` asks whether the 24-dimensional block
//! needs the Leech lattice. The comparison the paper offers is confounded: its
//! E8 row is cubic shaping against a Leech row with shape-gain. Answering it
//! needs an E8 that can be held to the same accounting, and this is that E8.
//!
//! ## Doubled coordinates, and why
//!
//! E8 in the even coordinate system is the integer vectors with even sum,
//! together with the half-integer vectors with even sum. Two cases, two
//! predicates, and half-integers in a float are an invitation to a rounding
//! bug in a test that is supposed to be exact.
//!
//! So a point is stored as `y = 2x`, an `i32` vector, and the whole lattice
//! becomes **one** predicate:
//!
//! ```text
//!   y in E8  <=>  all y_i have the same parity  and  sum(y_i) = 0 mod 4
//! ```
//!
//! Integer `x` gives all-even `y` and `sum(y)/2 = sum(x)` even, hence
//! `sum(y) = 0 mod 4`. Half-integer `x` gives all-odd `y` and the same
//! condition. Norms follow as `|x|^2 = |y|^2 / 4`, so the minimal norm 2 is
//! `|y|^2 = 8`, an integer, and every invariant below is an integer identity.
//!
//! ## What is verified here, and what is not
//!
//! Verified: the kissing number 240, the theta coefficients `240 sigma_3(n)`
//! for n = 1 to 5, the minimal norm, membership of every enumerated point, and
//! that the fast decoder agrees with brute force. Not verified here: the
//! normalized second moment, which is a measurement and lives in `bin/e8bench`.

/// Dimension of E8. The 24-dimensional block holds three of these.
pub const E8DIM: usize = 8;

/// A lattice point in doubled coordinates: `y = 2x`.
pub type Y = [i32; E8DIM];

/// The single membership predicate of the doubled representation.
pub fn in_e8(y: &Y) -> bool {
    let p = y[0].rem_euclid(2);
    y.iter().all(|c| c.rem_euclid(2) == p) && y.iter().sum::<i32>().rem_euclid(4) == 0
}

/// `|y|^2`, four times the squared norm of the point it represents.
pub fn qnorm(y: &Y) -> i32 {
    y.iter().map(|c| c * c).sum()
}

/// Every non-zero point of E8 with `|x|^2 <= cap`, that is `|y|^2 <= 4 cap`.
///
/// Depth-first over one parity at a time, pruned by the norm left. The origin
/// is excluded: a shape-gain code has no direction for it, and `Tetra`
/// reconstructs it as zero rather than coding it.
pub fn enumerate(cap: i32) -> Vec<Y> {
    let budget = 4 * cap;
    let mut out = Vec::new();
    for parity in [0i32, 1] {
        let mut y = [0i32; E8DIM];
        walk(0, budget, parity, &mut y, &mut out);
    }
    out.retain(|y| qnorm(y) > 0);
    out
}

fn walk(i: usize, left: i32, parity: i32, y: &mut Y, out: &mut Vec<Y>) {
    if i == E8DIM {
        if y.iter().sum::<i32>().rem_euclid(4) == 0 {
            out.push(*y);
        }
        return;
    }
    // The coordinate runs over its parity class inside the norm still
    // available. `c * c <= left` is the only bound needed: the tail can
    // always be filled with the smallest value of its parity class, which is
    // 0 for even and 1 for odd, and the odd case is charged below.
    let tail = (E8DIM - i - 1) as i32 * parity; // each remaining odd slot costs at least 1
    let mut c = -isqrt(left);
    while c <= isqrt(left) {
        if c.rem_euclid(2) == parity && c * c + tail <= left {
            y[i] = c;
            walk(i + 1, left - c * c, parity, y, out);
        }
        c += 1;
    }
    y[i] = 0;
}

fn isqrt(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut r = (n as f64).sqrt() as i32;
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    while r * r > n {
        r -= 1;
    }
    r
}

/// Nearest point of the **infinite** E8, by the Conway and Sloane route:
/// nearest in `D8`, nearest in `D8 + s` with `s = (1/2, ..., 1/2)`, keep the
/// closer. Returned in doubled coordinates.
///
/// `x` is the point to quantize, in ordinary coordinates.
pub fn nearest(x: &[f64; E8DIM]) -> Y {
    let a = nearest_d8(x);
    let shifted: [f64; E8DIM] = std::array::from_fn(|i| x[i] - 0.5);
    let b0 = nearest_d8(&shifted);
    let b: Y = std::array::from_fn(|i| b0[i] + 1); // + 1/2 in doubled units
    if dist2(x, &a) <= dist2(x, &b) { a } else { b }
}

/// Nearest point of `D8`, the integer vectors of even sum, in doubled units.
fn nearest_d8(x: &[f64; E8DIM]) -> Y {
    let mut r = [0i32; E8DIM];
    let mut worst = 0usize;
    let mut worst_err = -1.0f64;
    for i in 0..E8DIM {
        r[i] = round_half_away(x[i]);
        let e = (x[i] - r[i] as f64).abs();
        if e > worst_err {
            worst_err = e;
            worst = i;
        }
    }
    if r.iter().sum::<i32>().rem_euclid(2) != 0 {
        // Flip the coordinate that was rounded least confidently, the other
        // way. This is the exact repair: any other single change costs more,
        // and the parity defect needs an odd number of changes.
        r[worst] += if x[worst] >= r[worst] as f64 { 1 } else { -1 };
    }
    r.iter_mut().for_each(|c| *c *= 2);
    r
}

fn round_half_away(v: f64) -> i32 {
    if v >= 0.0 { (v + 0.5).floor() as i32 } else { -((-v + 0.5).floor() as i32) }
}

/// Squared distance from `x` to the point `y` holds in doubled coordinates.
pub fn dist2(x: &[f64; E8DIM], y: &Y) -> f64 {
    (0..E8DIM)
        .map(|i| {
            let d = x[i] - y[i] as f64 / 2.0;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigma3(n: i32) -> i32 {
        (1..=n).filter(|d| n % d == 0).map(|d| d * d * d).sum()
    }

    #[test]
    fn e8_kissing_number_240() {
        let shell: Vec<Y> = enumerate(2).into_iter().filter(|y| qnorm(y) == 8).collect();
        assert_eq!(shell.len(), 240, "kissing number of E8");
        // 112 of them are integer vectors, 128 half-integer. The split is the
        // reason the doubled representation exists, so it is asserted too.
        let even = shell.iter().filter(|y| y[0].rem_euclid(2) == 0).count();
        assert_eq!((even, shell.len() - even), (112, 128), "integer/half-integer split");
    }

    #[test]
    fn e8_theta_matches_240_sigma3() {
        let all = enumerate(10);
        for n in 1..=5i32 {
            let want = 240 * sigma3(n);
            let got = all.iter().filter(|y| qnorm(y) == 8 * n).count() as i32;
            assert_eq!(got, want, "norm {} shell of E8", 2 * n);
        }
        assert_eq!(all.len(), 56_880, "cumulative count to norm 10");
    }

    #[test]
    fn e8_minimum_norm_is_two() {
        let m = enumerate(10).iter().map(qnorm).min().unwrap();
        assert_eq!(m, 8, "minimum |y|^2, that is |x|^2 = 2");
    }

    #[test]
    fn e8_membership_of_every_enumerated_point() {
        assert!(enumerate(10).iter().all(in_e8), "an enumerated point outside E8");
    }

    #[test]
    fn e8_rejects_near_misses() {
        // Same parity, wrong sum: (1,1,0,0,0,0,0,0) doubled is integer x with
        // odd sum, which is D8's complement.
        assert!(!in_e8(&[2, 0, 0, 0, 0, 0, 0, 0]));
        // Mixed parity.
        assert!(!in_e8(&[1, 2, 1, 1, 1, 1, 1, 1]));
        // The two smallest real points, for contrast.
        assert!(in_e8(&[2, 2, 0, 0, 0, 0, 0, 0]));
        assert!(in_e8(&[1, 1, 1, 1, 1, 1, 1, 1]));
    }

    #[test]
    fn e8_decoder_agrees_with_brute_force() {
        // Brute force over a codebook wide enough to contain the answer for
        // points drawn well inside it. The decoder is exact for the infinite
        // lattice, so any disagreement is a bug in the decoder or a point
        // whose nearest neighbour left the capped set; the draw keeps |x|
        // small enough that it cannot.
        let book = enumerate(6);
        let mut rng = 0x2026_0918u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 11) as f64 / (1u64 << 53) as f64) * 1.2 - 0.6
        };
        for _ in 0..2000 {
            let x: [f64; E8DIM] = std::array::from_fn(|_| next());
            let fast = nearest(&x);
            let slow = book
                .iter()
                .chain(std::iter::once(&[0i32; E8DIM]))
                .min_by(|a, b| dist2(&x, a).total_cmp(&dist2(&x, b)))
                .copied()
                .unwrap();
            assert!(
                (dist2(&x, &fast) - dist2(&x, &slow)).abs() < 1e-12,
                "decoder missed: x {x:?} fast {fast:?} slow {slow:?}"
            );
        }
    }

    #[test]
    fn e8_decoder_returns_lattice_points() {
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 11) as f64 / (1u64 << 53) as f64) * 20.0 - 10.0
        };
        for _ in 0..5000 {
            let x: [f64; E8DIM] = std::array::from_fn(|_| next());
            assert!(in_e8(&nearest(&x)), "decoder left the lattice");
        }
    }

    #[test]
    fn e8_gram_determinant_is_one() {
        // A standard basis of E8 in the even coordinate system, doubled.
        let rows: [[f64; E8DIM]; E8DIM] = [
            [2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for r in &rows {
            let y: Y = std::array::from_fn(|i| r[i] as i32);
            assert!(in_e8(&y), "basis row outside E8: {r:?}");
        }
        // Undouble, then |det| of the basis is the covolume, which is 1.
        let mut m = rows.map(|r| r.map(|v| v / 2.0));
        let mut det = 1.0f64;
        for c in 0..E8DIM {
            let p = (c..E8DIM).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs())).unwrap();
            if p != c {
                m.swap(p, c);
                det = -det;
            }
            det *= m[c][c];
            let pivot = m[c];
            for row in m.iter_mut().skip(c + 1) {
                let f = row[c] / pivot[c];
                for (mk, pk) in row.iter_mut().zip(pivot.iter()).skip(c) {
                    *mk -= f * pk;
                }
            }
        }
        assert!((det.abs() - 1.0).abs() < 1e-9, "covolume of E8 is 1, got {}", det.abs());
    }
}
