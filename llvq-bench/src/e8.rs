//! E8, for the arbitration of 2026-09-18: enumeration, exact nearest
//! neighbour, and the invariants that say this code is the lattice.
//!
//! ## Why this exists at all
//!
//! `docs/archive/arbitrage-e8-2026-09-18.md` asks whether the 24-dimensional block
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

/// E8 cubed as a block quantizer, so the GPTQ loop can run it unchanged.
///
/// The trait's own documentation says the loop is meant to be testable against
/// codebooks that have nothing to do with the Leech lattice. Stage 1 of the E8
/// arbitration needs exactly that: the same sequential encode with the same
/// error feedback, the same gain levels and the same row scale, with only the
/// direction codebook changed.
///
/// ## The shape-gain semantics are `TetraShapeGain`'s, deliberately
///
/// The level comes from the block norm relative to the row, snapped to the
/// matrix's fitted centroids, and the output is the unit direction times that
/// amplitude. That is the served rule (`quantizer.rs:558-562`), not the
/// projection rule, so the arms differ in the codebook alone.
///
/// ## Why the maximum over the product codebook is exact
///
/// Maximizing `cos` means maximizing `<x, p> / |p|`. Over a product of three
/// E8 codes, `|p|^2` depends only on which shell each sub-block lands in, so
/// for a fixed triple of shells the three inner products maximize
/// independently. Scanning each sub-block once per shell and then comparing
/// the shell triples is therefore exact, not a heuristic.
pub struct E8Cubed {
    book: Vec<[f64; E8DIM]>,
    shell_of: Vec<usize>,
    shell_q: Vec<f64>,
    centroids: Vec<f64>,
    row_scale: f64,
}

impl E8Cubed {
    /// Every non-zero point with `|x|^2 <= cap`, grouped by shell.
    pub fn new(cap: i32, centroids: Vec<f64>) -> Self {
        assert!(centroids.len() >= 2, "a gain code needs at least two levels");
        assert!(centroids.windows(2).all(|c| c[0] <= c[1]), "centroids must be sorted");
        let pts = enumerate(cap);
        let mut shells: Vec<i32> = pts.iter().map(qnorm).collect();
        shells.sort_unstable();
        shells.dedup();
        let shell_of = pts
            .iter()
            .map(|y| shells.iter().position(|&s| s == qnorm(y)).expect("shell present"))
            .collect();
        Self {
            book: pts.iter().map(|y| std::array::from_fn(|i| y[i] as f64)).collect(),
            shell_of,
            shell_q: shells.iter().map(|&s| s as f64).collect(),
            centroids,
            row_scale: 1.0,
        }
    }

    /// Points in the codebook, and the bits an integer index costs.
    pub fn size(&self) -> (usize, u32) {
        (self.book.len(), (self.book.len() as f64).log2().ceil() as u32)
    }
}

impl llvq_quant::quantizer::BlockQuantizer for E8Cubed {
    fn block_len(&self) -> usize {
        3 * E8DIM
    }

    fn set_row_scale(&mut self, scale: f64) {
        self.row_scale = if scale > 0.0 { scale } else { 1.0 };
    }

    /// The output already sits on the level's sphere, so the loop must leave
    /// it alone. Same reasoning as `TetraShapeGain::retraction_target`: a
    /// target of `norm_before` would hand the magnitude back as a free float
    /// and cancel the gain code.
    fn retraction_target(&self, _norm_before: f64) -> Option<f64> {
        None
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        assert_eq!(v.len(), 3 * E8DIM, "block must be 24 weights");
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm == 0.0 {
            out.fill(0.0);
            return;
        }
        let ns = self.shell_q.len();
        let mut dot = vec![f64::NEG_INFINITY; 3 * ns];
        let mut arg = vec![0usize; 3 * ns];
        for (j, chunk) in v.chunks_exact(E8DIM).enumerate() {
            for (i, p) in self.book.iter().enumerate() {
                let d: f64 = (0..E8DIM).map(|k| p[k] * chunk[k]).sum();
                let s = j * ns + self.shell_of[i];
                if d > dot[s] {
                    dot[s] = d;
                    arg[s] = i;
                }
            }
        }
        let mut best = f64::NEG_INFINITY;
        let mut pick = [0usize; 3];
        for a in 0..ns {
            for b in 0..ns {
                for c in 0..ns {
                    let s = dot[a] + dot[ns + b] + dot[2 * ns + c];
                    if s <= 0.0 {
                        continue;
                    }
                    let q = (self.shell_q[a] + self.shell_q[b] + self.shell_q[c]).sqrt();
                    if s / q > best {
                        best = s / q;
                        pick = [arg[a], arg[ns + b], arg[2 * ns + c]];
                    }
                }
            }
        }
        // The gain level, by the served rule: the block norm relative to the
        // row, snapped to the matrix's centroids.
        let g = norm / self.row_scale;
        let level = self
            .centroids
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - g).abs().total_cmp(&(*b - g).abs()))
            .map(|(i, _)| i)
            .expect("centroids are not empty");
        let picked = self.centroids[level] * self.row_scale;
        let pn = pick
            .iter()
            .map(|&i| self.book[i].iter().map(|p| p * p).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        for (j, &i) in pick.iter().enumerate() {
            for k in 0..E8DIM {
                out[j * E8DIM + k] = self.book[i][k] * picked / pn;
            }
        }
    }
}

#[cfg(test)]
mod quantizer_tests {
    use super::*;
    use llvq_quant::quantizer::BlockQuantizer;

    fn q(cap: i32) -> E8Cubed {
        E8Cubed::new(cap, vec![1.0, 2.0])
    }

    #[test]
    fn e8cubed_reconstructs_a_codebook_block_exactly() {
        let mut e = q(8);
        let book = enumerate(8);
        let p: Vec<f64> = [book[0], book[7], book[19]]
            .iter()
            .flat_map(|y| y.iter().map(|&c| c as f64).collect::<Vec<_>>())
            .collect();
        let pn = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x: Vec<f64> = p.iter().map(|v| v / pn * 2.0).collect();
        e.set_row_scale(1.0); // so level 1, centroid 2.0, is the picked one
        let mut out = vec![0.0; 24];
        e.quantize(&x, &mut out);
        let err: f64 = x.iter().zip(&out).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!(err < 1e-20, "a codebook block must come back exactly, got {err:e}");
    }

    #[test]
    fn e8cubed_zero_block_is_zero() {
        let mut e = q(6);
        let mut out = vec![9.0; 24];
        e.quantize(&[0.0; 24], &mut out);
        assert!(out.iter().all(|v| *v == 0.0), "a zero block must stay zero");
    }

    #[test]
    fn e8cubed_output_sits_on_the_level_sphere() {
        let mut e = q(8);
        e.set_row_scale(0.5);
        let mut rng = 0x5eedu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for _ in 0..200 {
            let x: Vec<f64> = (0..24).map(|_| next()).collect();
            let mut out = vec![0.0; 24];
            e.quantize(&x, &mut out);
            let n = out.iter().map(|v| v * v).sum::<f64>().sqrt();
            let want = [0.5f64, 1.0];
            assert!(
                want.iter().any(|w| (n - w).abs() < 1e-12),
                "output norm {n} is on neither level sphere"
            );
        }
    }

    #[test]
    fn e8cubed_beats_a_single_shell_on_angle() {
        // The whole point of a union of shells: more directions. A cap of 8
        // must never be angularly worse than a cap of 2 on the same block.
        let (mut wide, mut narrow) = (q(8), q(2));
        wide.set_row_scale(1.0);
        narrow.set_row_scale(1.0);
        let mut rng = 0xbeefu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut better = 0;
        for _ in 0..200 {
            let x: Vec<f64> = (0..24).map(|_| next()).collect();
            let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut a = vec![0.0; 24];
            let mut b = vec![0.0; 24];
            wide.quantize(&x, &mut a);
            narrow.quantize(&x, &mut b);
            let cos = |o: &[f64]| {
                let d: f64 = x.iter().zip(o).map(|(p, q_)| p * q_).sum();
                let n: f64 = o.iter().map(|v| v * v).sum::<f64>().sqrt();
                d / (nx * n)
            };
            let (ca, cb) = (cos(&a), cos(&b));
            assert!(ca >= cb - 1e-12, "a wider cap lost on angle: {ca} against {cb}");
            if ca > cb + 1e-12 {
                better += 1;
            }
        }
        assert!(better > 100, "a wider cap should win often, won {better} of 200");
    }
}
