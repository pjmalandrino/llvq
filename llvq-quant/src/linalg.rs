//! Dense symmetric linear algebra for the GPTQ correction.
//!
//! Everything here serves one purpose: turn a layer Hessian `H = AᵀA/N` into
//! the upper-triangular factor `U` with `H⁻¹ = Uᵀ U`, which is what makes the
//! error-feedback update of the paper's Algorithm 1 a triangular solve rather
//! than a sequence of matrix inversions.
//!
//! Two implementations, one contract. The default is written out here — a
//! Cholesky, a triangular inverse and a triangular product — so the crate
//! stays dependency-free and readable end to end, which is the point of an
//! auditable core. It runs at about **1 G multiply-add/s**: fine up to a
//! 0.6B model, hopeless at 4B, where `d_in = 9728` makes the four cubic
//! passes ~615 G multiply-adds *per layer*.
//!
//! With `--features fast-linalg`, [`GptqFactor::new`] calls `faer` instead:
//! blocked, vectorized and threaded, same mathematics to the last rounding.
//! The tests assert *identities* (`L Lᵀ = H`, `(Uᵀ U) H = I`), not the
//! behaviour of one implementation, so they hold both paths — and
//! `both_factorizations_agree` requires the two to produce the same factor.
//!
//! Matrices are dense row-major `n × n` in a flat `Vec<f64>`.

/// The Hessian was not positive definite even after damping — almost always
/// a calibration problem (dead input channel, too few samples) rather than a
/// numerical one, so it is reported rather than silently patched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotPositiveDefinite {
    /// Pivot index where the factorization failed.
    pub at: usize,
}

impl core::fmt::Display for NotPositiveDefinite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Hessian is not positive definite at pivot {} — check the \
             calibration set (dead input channel? too few sequences?) or \
             raise the damping",
            self.at
        )
    }
}

impl std::error::Error for NotPositiveDefinite {}

#[inline]
fn at(n: usize, i: usize, j: usize) -> usize {
    i * n + j
}

/// `H + λ·mean(diag H)·I`, as a fresh buffer.
fn damped(h: &[f64], n: usize, damping: f64) -> Vec<f64> {
    assert_eq!(h.len(), n * n, "Hessian must be n × n");
    assert!(damping >= 0.0, "damping must be non-negative");
    let mut work = h.to_vec();
    if damping > 0.0 {
        let mean_diag: f64 = (0..n).map(|i| h[at(n, i, i)]).sum::<f64>() / n as f64;
        let d = damping * mean_diag;
        for i in 0..n {
            work[at(n, i, i)] += d;
        }
    }
    work
}

/// `U` with `H⁻¹ = Uᵀ U`, by way of `H = L Lᵀ` and a second Cholesky.
fn reference_factor(mut work: Vec<f64>, n: usize) -> Result<Vec<f64>, NotPositiveDefinite> {
    // H = L Lᵀ  ⇒  H⁻¹ = L⁻ᵀ L⁻¹.
    cholesky_lower(&mut work, n)?;
    invert_lower(&mut work, n);
    let mut hinv = vec![0.0; n * n];
    lower_transpose_product(&work, n, &mut hinv);
    // H⁻¹ = L₂ L₂ᵀ  ⇒  U = L₂ᵀ gives H⁻¹ = Uᵀ U.
    cholesky_lower(&mut hinv, n)?;
    let mut u = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            u[at(n, i, j)] = hinv[at(n, j, i)];
        }
    }
    Ok(u)
}

/// The same factorization on top of `faer`.
#[cfg(feature = "fast-linalg")]
mod fast {
    use super::{at, NotPositiveDefinite};
    use faer::linalg::solvers::DenseSolveCore;
    use faer::{Mat, Side};

    pub fn factor(h: &[f64], n: usize) -> Result<Vec<f64>, NotPositiveDefinite> {
        let a = Mat::<f64>::from_fn(n, n, |i, j| h[at(n, i, j)]);
        let llt = a.llt(Side::Lower).map_err(|_| NotPositiveDefinite { at: 0 })?;
        let hinv = llt.inverse();
        let llt2 = hinv
            .llt(Side::Lower)
            .map_err(|_| NotPositiveDefinite { at: 0 })?;
        let l2 = llt2.L();
        // U = L₂ᵀ, so H⁻¹ = L₂ L₂ᵀ = Uᵀ U with U upper triangular.
        let mut u = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                u[at(n, i, j)] = l2[(j, i)];
            }
        }
        Ok(u)
    }
}

/// In-place lower Cholesky: on success `a` holds `L` with `A = L Lᵀ`, and
/// the strict upper triangle is zeroed.
pub fn cholesky_lower(a: &mut [f64], n: usize) -> Result<(), NotPositiveDefinite> {
    debug_assert_eq!(a.len(), n * n);
    for j in 0..n {
        let mut d = a[at(n, j, j)];
        for k in 0..j {
            d -= a[at(n, j, k)] * a[at(n, j, k)];
        }
        // Also rejects NaN, which a Hessian built from a corrupted
        // activation dump will happily contain.
        if !(d.is_finite() && d > 0.0) {
            return Err(NotPositiveDefinite { at: j });
        }
        let ljj = d.sqrt();
        a[at(n, j, j)] = ljj;
        let inv = 1.0 / ljj;
        for i in j + 1..n {
            let mut s = a[at(n, i, j)];
            for k in 0..j {
                s -= a[at(n, i, k)] * a[at(n, j, k)];
            }
            a[at(n, i, j)] = s * inv;
        }
        for k in j + 1..n {
            a[at(n, j, k)] = 0.0;
        }
    }
    Ok(())
}

/// In-place inverse of a lower-triangular matrix with non-zero diagonal.
pub fn invert_lower(l: &mut [f64], n: usize) {
    debug_assert_eq!(l.len(), n * n);
    for j in 0..n {
        let inv = 1.0 / l[at(n, j, j)];
        l[at(n, j, j)] = inv;
        for i in j + 1..n {
            // Xᵢⱼ = −(Σ_{k=j..i−1} Lᵢₖ Xₖⱼ) / Lᵢᵢ
            let mut s = 0.0;
            for k in j..i {
                s += l[at(n, i, k)] * l[at(n, k, j)];
            }
            l[at(n, i, j)] = -s / l[at(n, i, i)];
        }
    }
}

/// `Mᵀ M` for a lower-triangular `M`, written into `out` (symmetric).
///
/// With `M = L⁻¹` this is exactly `L⁻ᵀ L⁻¹ = (L Lᵀ)⁻¹ = A⁻¹`.
pub fn lower_transpose_product(m: &[f64], n: usize, out: &mut [f64]) {
    debug_assert_eq!(m.len(), n * n);
    debug_assert_eq!(out.len(), n * n);
    out.fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            // Both columns of Mᵀ are supported on rows ≥ max(i, j) = i.
            let mut s = 0.0;
            for k in i..n {
                s += m[at(n, k, i)] * m[at(n, k, j)];
            }
            out[at(n, i, j)] = s;
            out[at(n, j, i)] = s;
        }
    }
}

/// The GPTQ factor of a layer: upper-triangular `U` with `H⁻¹ = Uᵀ U`.
///
/// Left-to-right column processing is what makes this factor the right one:
/// for a column block `Q` followed by the untouched columns `R`, the
/// analytic correction `ΔW_R = −ΔW_Q H_QQ⁻¹ H_QR` of Appendix F.2 rewrites as
/// `(E_Q U_QQ⁻¹) U_QR` — one small triangular solve and one product, with no
/// inverse ever formed.
pub struct GptqFactor {
    n: usize,
    u: Vec<f64>,
}

impl GptqFactor {
    /// Damp with `λ·mean(diag H)` on the diagonal, then factor.
    ///
    /// Damping is not optional in practice: layer Hessians of real models are
    /// routinely rank-deficient (input channels that never fire on the
    /// calibration set), and `λ = 0.01` is the value GPTQ and its descendants
    /// have converged on.
    pub fn new(h: &[f64], n: usize, damping: f64) -> Result<Self, NotPositiveDefinite> {
        let work = damped(h, n, damping);
        #[cfg(feature = "fast-linalg")]
        let u = fast::factor(&work, n)?;
        #[cfg(not(feature = "fast-linalg"))]
        let u = reference_factor(work, n)?;
        Ok(Self { n, u })
    }

    /// The dependency-free factorization, always available so the two paths
    /// can be compared.
    pub fn new_reference(h: &[f64], n: usize, damping: f64) -> Result<Self, NotPositiveDefinite> {
        let work = damped(h, n, damping);
        Ok(Self {
            n,
            u: reference_factor(work, n)?,
        })
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// `U[i][j]`, zero below the diagonal.
    #[inline]
    pub fn u(&self, i: usize, j: usize) -> f64 {
        self.u[at(self.n, i, j)]
    }

    /// Solve `X · U_QQ = E` for `X`, in place on `e` (`rows × b`, row-major),
    /// where `U_QQ` is the diagonal block of `U` on columns `s..s+b`.
    ///
    /// `U_QQ` is upper triangular, so the columns of `X` resolve
    /// left-to-right by forward substitution.
    pub fn solve_block(&self, e: &mut [f64], rows: usize, s: usize, b: usize) {
        debug_assert_eq!(e.len(), rows * b);
        for j in 0..b {
            let ujj = self.u(s + j, s + j);
            debug_assert!(ujj != 0.0, "U has a zero pivot");
            let inv = 1.0 / ujj;
            for r in 0..rows {
                let mut acc = e[r * b + j];
                for k in 0..j {
                    acc -= e[r * b + k] * self.u(s + k, s + j);
                }
                e[r * b + j] = acc * inv;
            }
        }
    }
}
