//! Integer-coordinate construction of the Leech lattice Λ₂₄.
//!
//! Following LLVQ (arXiv:2603.11021, §2.3, Eq. 4–5) we work in the integer
//! embedding `√8·Λ₂₄ ⊂ Z²⁴`: a vector `x ∈ Z²⁴` belongs to the (scaled)
//! Leech lattice iff
//!
//! | | even coset | odd coset |
//! |---|---|---|
//! | (i) parity | `xᵢ ≡ 0 (mod 2)` ∀i | `xᵢ ≡ 1 (mod 2)` ∀i |
//! | (ii) Golay | `{i : xᵢ ≡ 2 (mod 4)} ∈ G₂₄` | `{i : xᵢ ≡ 3 (mod 4)} ∈ G₂₄` |
//! | (iii) sum | `Σxᵢ ≡ 0 (mod 8)` | `Σxᵢ ≡ 4 (mod 8)` |
//!
//! With the `1/√8` normalization the lattice is even and unimodular; the
//! `m`-th shell `{v : ‖v‖² = 2m}` corresponds to `‖x‖² = 16m` in the
//! integer embedding. The minimal shell is `m = 2` (`‖x‖² = 32`) with
//! 196,560 vectors — the kissing number of Λ₂₄ — which the G1 test-suite
//! reproduces by exhaustive, membership-verified enumeration.
//!
//! ## Exactness domain
//!
//! Every function in this module is exact over the **entire** `[i32; 24]`
//! domain: `norm2` accumulates in `i128` (24·(2³¹)² < 2⁶⁷), `shell_index`
//! returns a `u64` quotient that always fits, and `add`/`neg` use checked
//! arithmetic — a coordinate overflow panics instead of silently returning
//! a *different*, valid-looking lattice point (an `i32` wraparound is a
//! congruence mod 2³², which all three membership constraints — mod 2, 4
//! and 8 — would fail to detect).

use crate::golay::Golay;
use crate::rng::SplitMix64;

/// Ambient dimension of the Leech lattice.
pub const DIM: usize = 24;

/// A point of the integer embedding `√8·Λ₂₄` (or a candidate for membership).
pub type Point = [i32; DIM];

/// Theta-series coefficients of Λ₂₄: `(m, |Shell(m)|)` for the first shells.
/// `Shell(m) = {v ∈ Λ₂₄ : ‖v‖² = 2m}`. Source: OEIS A008408 / paper Table 1.
/// Entries for m = 2 and m = 3 are pinned by exhaustive enumeration in the
/// G1 suite; m = 4 gets class-level spot checks there.
pub const THETA: [(u32, u64); 4] = [
    (2, 196_560),
    (3, 16_773_120),
    (4, 398_034_000),
    (5, 4_629_381_120),
];

/// Cumulative shell count `N(13) = Σ_{m=2..13} |Shell(m)|` at exactly
/// 2.000 bits/dim (paper Table 1), i.e. the codebook size of the
/// 2 bit-per-weight regime. Convention: **the origin is excluded** (the
/// paper's `N(M)` sums shells from m = 2); the closed ball `‖x‖² ≤ 26`
/// contains one point more, 280,974,212,784,721.
pub const N_SHELL_13_CUMULATIVE: u64 = 280_974_212_784_720;

/// The Leech lattice, backed by the extended Golay code.
pub struct Leech {
    golay: Golay,
}

impl Leech {
    /// Build the lattice handle (constructs the Golay code tables once).
    pub fn new() -> Self {
        Self {
            golay: Golay::new(),
        }
    }

    /// The underlying Golay code.
    pub fn golay(&self) -> &Golay {
        &self.golay
    }

    /// Exact membership test for the integer embedding `√8·Λ₂₄`
    /// (Eq. 4–5 of the paper; see module docs for the three constraints).
    pub fn contains(&self, x: &Point) -> bool {
        // (i) uniform parity. Early return here also guarantees that in the
        // odd case every coordinate is odd, so `v - p` below cannot overflow
        // even for v = i32::MIN (which is even).
        let p = x[0].rem_euclid(2);
        if x.iter().any(|&v| v.rem_euclid(2) != p) {
            return false;
        }
        // (ii) the mod-4 pattern must be a Golay codeword.
        // (v - p) is even, so the arithmetic shift is an exact halving.
        let mut word = 0u32;
        for (i, &v) in x.iter().enumerate() {
            if ((v - p) >> 1).rem_euclid(2) == 1 {
                word |= 1 << i;
            }
        }
        if !self.golay.contains(word) {
            return false;
        }
        // (iii) global sum congruence: 0 (mod 8) even, 4 (mod 8) odd.
        // Sum of 24 i32 values fits i64 (|Σ| < 24·2³¹ < 2³⁶).
        let s: i64 = x.iter().map(|&v| v as i64).sum();
        s.rem_euclid(8) == 4 * p as i64
    }

    /// Draw a random lattice point for tests and benches.
    ///
    /// Any `x` with `xᵢ = p + 2cᵢ + 4kᵢ` satisfies (i) and (ii) by
    /// construction; the sum is `24p + 2·wt(c) + 4·Σk ≡ 4·Σk (mod 8)`
    /// because `24p ≡ 0` and doubled Golay weights (2·{0,8,12,16,24}) are
    /// all ≡ 0 (mod 8). Constraint (iii) therefore reduces to
    /// `Σk ≡ p (mod 2)`, fixed up by nudging `k₀` — which skews the
    /// distribution of `x[0]` slightly: fine for validity-oriented tests,
    /// do not use where exact uniformity matters.
    pub fn random_point(&self, rng: &mut SplitMix64, spread: i32) -> Point {
        debug_assert!(spread >= 1);
        let p = (rng.next() & 1) as i32;
        let c = self.golay.codewords()[(rng.next() % 4096) as usize];
        let mut x = [0i32; DIM];
        let mut ksum = 0i64;
        let width = (2 * spread + 1) as u64;
        for (i, xi) in x.iter_mut().enumerate() {
            let k = (rng.next() % width) as i32 - spread;
            ksum += k as i64;
            *xi = p + 2 * ((c >> i) & 1) as i32 + 4 * k;
        }
        if (ksum - p as i64).rem_euclid(2) != 0 {
            x[0] += 4; // k₀ += 1 flips the parity of Σk.
        }
        debug_assert!(self.contains(&x));
        x
    }

    /// Squared Euclidean norm in the integer embedding.
    /// Exact for every `[i32; 24]`: accumulated in `i128`
    /// (24·(2³¹)² = 3·2⁶⁵ < 2¹²⁷).
    pub fn norm2(x: &Point) -> i128 {
        x.iter().map(|&v| (v as i128) * (v as i128)).sum()
    }

    /// Shell index `m` such that `‖x‖² = 16m`, if `x` lies on a shell
    /// radius (necessary but not sufficient for membership).
    /// The quotient always fits: `‖x‖²/16 ≤ 24·2⁶²/16 < 2⁶³`.
    pub fn shell_index(x: &Point) -> Option<u64> {
        let n = Self::norm2(x);
        (n % 16 == 0).then_some((n / 16) as u64)
    }

    /// Component-wise sum (the lattice is an additive group).
    ///
    /// # Panics
    /// Panics on `i32` coordinate overflow: a silent wraparound is a
    /// congruence mod 2³² that the membership constraints cannot detect,
    /// so it would return a *wrong* group element that still passes
    /// [`Leech::contains`].
    pub fn add(a: &Point, b: &Point) -> Point {
        let mut out = [0i32; DIM];
        for i in 0..DIM {
            out[i] = a[i]
                .checked_add(b[i])
                .expect("Leech::add: coordinate overflow");
        }
        out
    }

    /// Component-wise negation.
    ///
    /// # Panics
    /// Panics on `i32::MIN` coordinates (whose negation is unrepresentable;
    /// wrapping would make `neg` a non-involution at the boundary).
    pub fn neg(a: &Point) -> Point {
        let mut out = [0i32; DIM];
        for i in 0..DIM {
            out[i] = a[i]
                .checked_neg()
                .expect("Leech::neg: coordinate overflow");
        }
        out
    }
}

impl Default for Leech {
    fn default() -> Self {
        Self::new()
    }
}
