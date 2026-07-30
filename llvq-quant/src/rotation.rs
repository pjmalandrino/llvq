//! Incoherence processing: an orthogonal rotation applied before quantizing.
//!
//! Lattice quantizers assume their input looks roughly isotropic. Real weight
//! rows do not — they carry outlier channels that a 24-dimensional codebook
//! spends most of its resolution on. Rotating the input basis by a random
//! orthogonal `Q` spreads those outliers across coordinates, which is what
//! QuIP#, QuaRot and the LLVQ paper all mean by *incoherence processing*.
//!
//! ## Why this costs nothing at measurement time
//!
//! For `y = W x`, inserting `QᵀQ = I` gives `y = (W Qᵀ)(Q x)`. Quantizing in
//! the rotated basis and rotating **back** yields a drop-in replacement:
//!
//! ```text
//! W' = W Qᵀ        H' = Q H Qᵀ        Ŵ = quantize(W', H') · Q
//! ```
//!
//! `Ŵ` plugs straight into the unmodified model, so the quality effect of the
//! rotation is measurable without touching the runtime. A real deployment is
//! different: it would store the rotated indices and apply `Q` to the
//! activations online, which is the latency the paper's Appendix I is trying
//! to avoid. What we measure here is the quality ceiling of that transform.
//!
//! ## Construction
//!
//! `Q = (Q_odd ⊗ H_m) D` with `n = k·m`, `m` the largest power of two dividing
//! `n`, `D` a random sign flip, and `H_m` the Walsh–Hadamard transform scaled
//! to be orthogonal. The paper's models are not all power-of-two wide
//! (Qwen3-0.6B: 1024 and 2048 are, 3072 = 3·1024 is not), and a Kronecker
//! factor handles the odd part for any width. `Q_odd` is a small dense
//! orthogonal matrix from Gram–Schmidt; the scaled Walsh–Hadamard transform
//! is symmetric and its own inverse, which keeps the transpose cheap.

use llvq_core::SplitMix64;

/// A fixed random orthogonal transform of `R^n`.
pub struct Rotation {
    n: usize,
    /// Power-of-two factor, handled by the fast transform.
    m: usize,
    /// Odd factor, handled by a dense `k × k` block.
    k: usize,
    signs: Vec<f64>,
    /// `k × k`, row-major, orthogonal.
    small: Vec<f64>,
}

impl Rotation {
    /// Build the rotation for dimension `n`, deterministically from `seed`.
    pub fn new(n: usize, seed: u64) -> Self {
        assert!(n > 0, "dimension must be positive");
        let m = 1usize << n.trailing_zeros();
        let k = n / m;
        let mut rng = SplitMix64::new(seed);
        let signs: Vec<f64> = (0..n)
            .map(|_| if rng.next_gaussian() < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let small = orthonormal(k, &mut rng);
        Self {
            n,
            m,
            k,
            signs,
            small,
        }
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// `v ← Q v`.
    pub fn apply(&self, v: &mut [f64]) {
        assert_eq!(v.len(), self.n);
        for (x, s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }
        for g in 0..self.k {
            wht(&mut v[g * self.m..(g + 1) * self.m]);
        }
        self.mix(v, false);
    }

    /// `v ← Qᵀ v`.
    pub fn apply_transpose(&self, v: &mut [f64]) {
        assert_eq!(v.len(), self.n);
        self.mix(v, true);
        for g in 0..self.k {
            wht(&mut v[g * self.m..(g + 1) * self.m]);
        }
        for (x, s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }
    }

    /// Apply `Q_odd` (or its transpose) across the `k` groups, for each of the
    /// `m` positions within a group.
    fn mix(&self, v: &mut [f64], transpose: bool) {
        if self.k == 1 {
            return;
        }
        let mut col = vec![0.0f64; self.k];
        for j in 0..self.m {
            for (g, c) in col.iter_mut().enumerate() {
                *c = v[g * self.m + j];
            }
            for g in 0..self.k {
                let mut acc = 0.0;
                for (t, c) in col.iter().enumerate() {
                    // Row-major `small`: Q[g][t], or Qᵀ[g][t] = Q[t][g].
                    let q = if transpose {
                        self.small[t * self.k + g]
                    } else {
                        self.small[g * self.k + t]
                    };
                    acc += q * c;
                }
                v[g * self.m + j] = acc;
            }
        }
    }

    /// `W ← W Qᵀ` for a `d_out × n` row-major matrix — i.e. `Q` applied to
    /// every row read as a column vector.
    pub fn rotate_weight_rows(&self, w: &mut [f64], d_out: usize) {
        assert_eq!(w.len(), d_out * self.n);
        for i in 0..d_out {
            self.apply(&mut w[i * self.n..(i + 1) * self.n]);
        }
    }

    /// The inverse of [`Self::rotate_weight_rows`]: `W ← W Q`.
    pub fn unrotate_weight_rows(&self, w: &mut [f64], d_out: usize) {
        assert_eq!(w.len(), d_out * self.n);
        for i in 0..d_out {
            self.apply_transpose(&mut w[i * self.n..(i + 1) * self.n]);
        }
    }

    /// `H ← Q H Qᵀ` for a symmetric `n × n` row-major matrix.
    ///
    /// Two passes of the fast transform (rows, then columns) rather than two
    /// dense products: `O(n² log n)` instead of `O(n³)`, which matters when
    /// `n = 3072` and this runs once per activation per block.
    pub fn rotate_hessian(&self, h: &mut [f64]) {
        assert_eq!(h.len(), self.n * self.n);
        for i in 0..self.n {
            self.apply(&mut h[i * self.n..(i + 1) * self.n]);
        }
        // Now apply Q to the columns, i.e. Q (H Qᵀ).
        let mut col = vec![0.0f64; self.n];
        for j in 0..self.n {
            for (i, c) in col.iter_mut().enumerate() {
                *c = h[i * self.n + j];
            }
            self.apply(&mut col);
            for (i, c) in col.iter().enumerate() {
                h[i * self.n + j] = *c;
            }
        }
    }
}

/// In-place Walsh–Hadamard transform, scaled to be orthogonal.
///
/// The scaled transform is symmetric and involutive, so the same routine
/// serves for `H` and `Hᵀ = H⁻¹`.
fn wht(v: &mut [f64]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let (a, b) = (v[j], v[j + len]);
                v[j] = a + b;
                v[j + len] = a - b;
            }
            i += len << 1;
        }
        len <<= 1;
    }
    let s = 1.0 / (n as f64).sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// A `k × k` orthonormal matrix by Gram–Schmidt on Gaussian columns.
fn orthonormal(k: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let mut q = vec![0.0f64; k * k];
    for i in 0..k {
        let mut row: Vec<f64> = (0..k).map(|_| rng.next_gaussian()).collect();
        for p in 0..i {
            let prev = &q[p * k..(p + 1) * k];
            let d: f64 = row.iter().zip(prev).map(|(a, b)| a * b).sum();
            for (r, b) in row.iter_mut().zip(prev) {
                *r -= d * b;
            }
        }
        let nrm = row.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(nrm > 1e-9, "degenerate Gram–Schmidt; retry with another seed");
        for (slot, r) in q[i * k..(i + 1) * k].iter_mut().zip(row.iter()) {
            *slot = r / nrm;
        }
    }
    q
}
