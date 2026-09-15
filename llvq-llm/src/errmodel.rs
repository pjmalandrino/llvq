//! A surrogate for the model's own loss, so calibration choices can be ranked
//! without re-evaluating the model for each one.
//!
//! ## Why this module exists, and why it is not `calib`
//!
//! GPTQ calibrates one matrix at a time against `tr(E H Eᵀ)`, the squared error
//! its own output takes on the calibration activations. That objective is
//! exact, cheap and **measurably not the one that matters**. Three arms of
//! 2026-09-15 read the same knob — a multiplier on the fitted gain centroids —
//! against three objectives and got three answers (*measured*,
//! `docs/mesures/gain-scale-0.6b-2026-09-15.txt`):
//!
//! | objective | optimal multiplier |
//! |---|---|
//! | Euclidean error on the weights | 0.906 |
//! | `tr(E H Eᵀ)`, what GPTQ minimizes | 0.999 |
//! | the model's perplexity | 1.02 |
//!
//! The middle row is a closed form, verified against direct evaluation to
//! 1.6e-14, so the disagreement is not an estimation artefact. The layer
//! objective conditions on inputs that are already degraded, and cannot see
//! that a systematic error composes across layers. A calibration map built on
//! it reads "nothing to correct" everywhere and misses what the model shows.
//!
//! So this module models the endpoint directly, and keeps the layer objective
//! out. It lives beside `calib` rather than inside it for that reason: `calib`
//! answers "what does this matrix cost", this answers "what does the model
//! cost", and conflating them is the mistake above.
//!
//! ## The surrogate
//!
//! Let `sₖ` scale the reconstruction of matrix `k` and `δₖ = sₖ − 1`. Around
//! `s = 1` the model's loss is
//!
//! ```text
//! L(s) ≈ L(1) + Σₖ gₖ·δₖ + ½ Σₖ hₖ·δₖ²
//! ```
//!
//! with `gₖ` and `hₖ` measured by central differences, two evaluations per
//! matrix. Once they are known, **any** vector of scales is predicted with no
//! further evaluation, which is the point: a chain of candidate calibrations is
//! ranked for the cost of the probes, not for the cost of the candidates.
//!
//! Its per-matrix optimum is closed form, `δₖ* = −gₖ/hₖ` where `hₖ > 0`.
//!
//! ## What it deliberately leaves out, and how that is caught
//!
//! Cross terms `hₖₗ` are not measured: there are `n²` of them and each costs an
//! evaluation. The surrogate therefore assumes matrices perturb the loss
//! independently, which is an assumption and not a theorem. [`Surrogate::
//! predict`] is exact for one matrix at a time by construction, so the
//! assumption is only ever tested by a *combination* — which is what
//! [`Residual`] is for. A combination whose measured loss falls outside the
//! predicted one is the signal that the diagonal model is insufficient, and it
//! is reported rather than smoothed over.

/// One probe point: which matrix was scaled, by how much, and what the model's
/// loss read there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    /// Index of the scaled matrix in the run's own ordering.
    pub matrix: usize,
    /// `δ = s − 1`. Signed, and the two probes of a matrix straddle zero.
    pub delta: f64,
    /// The model's loss at that point. Any scalar the caller wants modelled —
    /// NLL is the natural one; perplexity is not, being its exponential.
    pub loss: f64,
}

/// Central-difference estimates for one matrix, and the step they came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sensitivity {
    pub matrix: usize,
    /// `∂L/∂δ` at `δ = 0`.
    pub gradient: f64,
    /// `∂²L/∂δ²` at `δ = 0`.
    pub curvature: f64,
    /// The `ε` the pair was measured at, kept because a gradient without its
    /// step is not reproducible and cannot be re-differenced later.
    pub step: f64,
}

impl Sensitivity {
    /// The per-matrix optimum of the quadratic, `δ* = −g/h`, when the
    /// curvature is positive. `None` on a flat or concave direction, where the
    /// quadratic has no interior minimum and extrapolating it would invent one.
    pub fn optimum(&self) -> Option<f64> {
        (self.curvature > 0.0).then(|| -self.gradient / self.curvature)
    }

    /// Predicted loss change of moving this matrix alone to its optimum. Never
    /// positive: it is `−g²/(2h)`.
    pub fn best_gain(&self) -> f64 {
        match self.optimum() {
            Some(d) => self.gradient * d + 0.5 * self.curvature * d * d,
            None => 0.0,
        }
    }
}

/// Why a pair of probes could not be turned into a sensitivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeError {
    /// The two probes are not `+ε` and `−ε` of one matrix.
    NotCentred,
    /// A probe carries a non-finite loss, which no amount of arithmetic fixes.
    NotFinite,
    /// `ε = 0`: the difference quotient does not exist.
    ZeroStep,
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCentred => write!(f, "the two probes are not ±ε of one matrix"),
            Self::NotFinite => write!(f, "a probe loss is not finite"),
            Self::ZeroStep => write!(f, "the probe step is zero"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Central differences from the baseline and one `(−ε, +ε)` pair.
///
/// Both estimates are second-order accurate in `ε`; the pair is required to be
/// symmetric because a one-sided difference would carry a first-order error
/// into the curvature, which is the term the whole map turns on.
pub fn differentiate(base: f64, minus: Probe, plus: Probe) -> Result<Sensitivity, ProbeError> {
    if !base.is_finite() || !minus.loss.is_finite() || !plus.loss.is_finite() {
        return Err(ProbeError::NotFinite);
    }
    if minus.matrix != plus.matrix || minus.delta != -plus.delta {
        return Err(ProbeError::NotCentred);
    }
    let eps = plus.delta;
    if eps == 0.0 {
        return Err(ProbeError::ZeroStep);
    }
    Ok(Sensitivity {
        matrix: plus.matrix,
        gradient: (plus.loss - minus.loss) / (2.0 * eps),
        curvature: (plus.loss - 2.0 * base + minus.loss) / (eps * eps),
        step: eps.abs(),
    })
}

/// The quadratic model of the model's loss over per-matrix scales.
#[derive(Debug, Clone, PartialEq)]
pub struct Surrogate {
    /// `L(1)`, the loss of the run as it stands.
    pub base: f64,
    /// One entry per probed matrix, in ascending matrix index.
    pub terms: Vec<Sensitivity>,
}

impl Surrogate {
    /// Builds from measured sensitivities, sorted and checked for duplicates.
    pub fn new(base: f64, mut terms: Vec<Sensitivity>) -> Result<Self, String> {
        terms.sort_by_key(|t| t.matrix);
        if terms.windows(2).any(|w| w[0].matrix == w[1].matrix) {
            return Err("two sensitivities for one matrix".into());
        }
        if !base.is_finite() || terms.iter().any(|t| !t.gradient.is_finite() || !t.curvature.is_finite()) {
            return Err("a non-finite term would poison every prediction".into());
        }
        Ok(Self { base, terms })
    }

    /// Predicted loss at the given deltas, keyed by matrix index. Matrices the
    /// map does not carry contribute nothing, which is the honest default: an
    /// unprobed matrix has no measured sensitivity, and guessing one would be
    /// the map inventing its own content.
    pub fn predict(&self, deltas: &[(usize, f64)]) -> f64 {
        let mut out = self.base;
        for &(matrix, delta) in deltas {
            if let Some(t) = self.terms.iter().find(|t| t.matrix == matrix) {
                out += t.gradient * delta + 0.5 * t.curvature * delta * delta;
            }
        }
        out
    }

    /// The scale vector the surrogate believes is best, one entry per matrix
    /// whose curvature admits an interior minimum.
    pub fn argmin(&self) -> Vec<(usize, f64)> {
        self.terms
            .iter()
            .filter_map(|t| t.optimum().map(|d| (t.matrix, d)))
            .collect()
    }

    /// Predicted loss change at [`Surrogate::argmin`], summed over matrices.
    /// It is what the map claims is available, and the number a real run has to
    /// be held against.
    pub fn claimed_gain(&self) -> f64 {
        self.terms.iter().map(Sensitivity::best_gain).sum()
    }

    /// Matrices ordered by what moving each alone would buy, most first. This
    /// is the calibration map's actual output: where the error is worth
    /// attacking, and by how much.
    pub fn ranked(&self) -> Vec<Sensitivity> {
        let mut v = self.terms.clone();
        v.sort_by(|a, b| a.best_gain().total_cmp(&b.best_gain()));
        v
    }
}

/// One held-out test of the surrogate: what it predicted, what the model did.
///
/// A surrogate that is never confronted with a combination it did not see is a
/// curve fitted to its own probes. This is the confrontation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Residual {
    pub predicted: f64,
    pub measured: f64,
    /// `L(1)`, so the error can be read against the size of the move rather
    /// than against the absolute loss, which is dominated by the model itself.
    pub base: f64,
}

impl Residual {
    pub fn absolute(&self) -> f64 {
        self.measured - self.predicted
    }

    /// Error as a fraction of the loss change the surrogate predicted. This is
    /// the number that decides whether the diagonal model is usable: an error
    /// of 5 % of a move is a working map, an error of 200 % is a map that got
    /// the sign of its own prediction wrong.
    ///
    /// `None` when the predicted move is zero, where a ratio says nothing.
    pub fn relative_to_move(&self) -> Option<f64> {
        let move_size = self.predicted - self.base;
        (move_size != 0.0).then(|| (self.measured - self.predicted) / move_size.abs())
    }

    /// Whether prediction and measurement agree on the *direction* of the move.
    /// The weakest useful claim a surrogate can make, and the one a ranking
    /// needs: a map that cannot tell an improvement from a regression ranks
    /// nothing, whatever its magnitudes look like.
    pub fn agrees_in_sign(&self) -> bool {
        (self.predicted - self.base).signum() == (self.measured - self.base).signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An exactly quadratic loss, which the surrogate must reproduce to the
    /// last bit: `L = base + Σ g·δ + ½ Σ h·δ²`.
    fn quadratic(base: f64, coeffs: &[(f64, f64)], deltas: &[(usize, f64)]) -> f64 {
        let mut out = base;
        for &(k, d) in deltas {
            let (g, h) = coeffs[k];
            out += g * d + 0.5 * h * d * d;
        }
        out
    }

    fn probe_pair(base: f64, coeffs: &[(f64, f64)], k: usize, eps: f64) -> (Probe, Probe) {
        (
            Probe { matrix: k, delta: -eps, loss: quadratic(base, coeffs, &[(k, -eps)]) },
            Probe { matrix: k, delta: eps, loss: quadratic(base, coeffs, &[(k, eps)]) },
        )
    }

    #[test]
    fn central_differences_are_exact_on_a_quadratic() {
        // Central differences carry no truncation error on a quadratic, so any
        // gap here is arithmetic, not method. That is what makes this a test of
        // the formula rather than of the step size.
        let base = 3.5;
        let coeffs = [(0.25, 2.0), (-1.5, 0.5), (0.0, 4.0)];
        for (k, &(g, h)) in coeffs.iter().enumerate() {
            for eps in [1e-1, 1e-2, 1e-3] {
                let (minus, plus) = probe_pair(base, &coeffs, k, eps);
                let s = differentiate(base, minus, plus).expect("well-formed pair");
                assert!((s.gradient - g).abs() < 1e-9, "g at eps={eps}: {} vs {g}", s.gradient);
                assert!((s.curvature - h).abs() < 1e-6, "h at eps={eps}: {} vs {h}", s.curvature);
                assert_eq!(s.matrix, k);
                assert_eq!(s.step, eps);
            }
        }
    }

    #[test]
    fn surrogate_reproduces_the_function_it_models() {
        let base = 10.0;
        let coeffs = [(0.5, 3.0), (-2.0, 8.0)];
        let terms: Vec<_> = (0..coeffs.len())
            .map(|k| {
                let (m, p) = probe_pair(base, &coeffs, k, 1e-2);
                differentiate(base, m, p).unwrap()
            })
            .collect();
        let s = Surrogate::new(base, terms).unwrap();
        // Single moves and a combination: on a separable quadratic the
        // surrogate is the function, so both must agree.
        for deltas in [
            vec![(0usize, 0.3)],
            vec![(1usize, -0.2)],
            vec![(0usize, 0.3), (1usize, -0.2)],
        ] {
            let want = quadratic(base, &coeffs, &deltas);
            assert!(
                (s.predict(&deltas) - want).abs() < 1e-6,
                "{:?}: {} vs {want}",
                deltas,
                s.predict(&deltas)
            );
        }
    }

    #[test]
    fn optimum_is_the_minimum_and_the_gain_is_negative() {
        let base = 1.0;
        let coeffs = [(0.5, 3.0), (-2.0, 8.0)];
        let terms: Vec<_> = (0..coeffs.len())
            .map(|k| {
                let (m, p) = probe_pair(base, &coeffs, k, 1e-2);
                differentiate(base, m, p).unwrap()
            })
            .collect();
        let s = Surrogate::new(base, terms).unwrap();
        for t in &s.terms {
            let d = t.optimum().expect("positive curvature");
            let (g, h) = coeffs[t.matrix];
            assert!((d - (-g / h)).abs() < 1e-6);
            // Nothing beats the optimum, on either side of it.
            let at = quadratic(base, &coeffs, &[(t.matrix, d)]);
            for probe in [d - 0.05, d + 0.05] {
                assert!(quadratic(base, &coeffs, &[(t.matrix, probe)]) > at);
            }
            assert!(t.best_gain() < 0.0, "a gain that raises the loss is not a gain");
        }
        assert!(s.claimed_gain() < 0.0);
    }

    #[test]
    fn a_flat_or_concave_direction_has_no_optimum() {
        // Extrapolating `−g/h` through a non-positive curvature invents a
        // minimum that is not there, and would send a real run to an edge.
        for h in [0.0, -1.0] {
            let t = Sensitivity { matrix: 0, gradient: 1.0, curvature: h, step: 1e-2 };
            assert_eq!(t.optimum(), None);
            assert_eq!(t.best_gain(), 0.0);
        }
    }

    #[test]
    fn malformed_probe_pairs_are_refused_by_name() {
        let ok = Probe { matrix: 0, delta: 0.01, loss: 1.0 };
        assert_eq!(
            differentiate(1.0, Probe { matrix: 1, delta: -0.01, loss: 1.0 }, ok),
            Err(ProbeError::NotCentred)
        );
        assert_eq!(
            differentiate(1.0, Probe { matrix: 0, delta: -0.02, loss: 1.0 }, ok),
            Err(ProbeError::NotCentred)
        );
        assert_eq!(
            differentiate(
                1.0,
                Probe { matrix: 0, delta: 0.0, loss: 1.0 },
                Probe { matrix: 0, delta: 0.0, loss: 1.0 }
            ),
            Err(ProbeError::ZeroStep)
        );
        assert_eq!(
            differentiate(f64::NAN, Probe { matrix: 0, delta: -0.01, loss: 1.0 }, ok),
            Err(ProbeError::NotFinite)
        );
    }

    #[test]
    fn duplicate_matrices_are_refused() {
        let t = Sensitivity { matrix: 2, gradient: 1.0, curvature: 1.0, step: 1e-2 };
        assert!(Surrogate::new(0.0, vec![t, t]).is_err());
        assert!(Surrogate::new(f64::INFINITY, vec![t]).is_err());
    }

    #[test]
    fn unprobed_matrices_contribute_nothing() {
        let t = Sensitivity { matrix: 0, gradient: 1.0, curvature: 2.0, step: 1e-2 };
        let s = Surrogate::new(5.0, vec![t]).unwrap();
        assert_eq!(s.predict(&[(7, 0.5)]), 5.0);
        assert_eq!(s.predict(&[(0, 0.0), (7, 9.9)]), 5.0);
    }

    #[test]
    fn ranking_puts_the_biggest_available_gain_first() {
        let terms = vec![
            Sensitivity { matrix: 0, gradient: 0.1, curvature: 1.0, step: 1e-2 },
            Sensitivity { matrix: 1, gradient: 2.0, curvature: 1.0, step: 1e-2 },
            Sensitivity { matrix: 2, gradient: 0.0, curvature: 1.0, step: 1e-2 },
        ];
        let s = Surrogate::new(0.0, terms).unwrap();
        let order: Vec<_> = s.ranked().iter().map(|t| t.matrix).collect();
        assert_eq!(order, vec![1, 0, 2], "most negative gain first, flat last");
    }

    #[test]
    fn residual_reads_error_against_the_move_and_not_the_loss() {
        // The absolute loss is dominated by the model; only the move is the
        // surrogate's claim, so that is what its error is measured against.
        let r = Residual { base: 100.0, predicted: 99.0, measured: 99.1 };
        assert!((r.absolute() - 0.1).abs() < 1e-12);
        assert!((r.relative_to_move().unwrap() - 0.1).abs() < 1e-12);
        assert!(r.agrees_in_sign());

        let wrong = Residual { base: 100.0, predicted: 99.0, measured: 100.5 };
        assert!(!wrong.agrees_in_sign(), "an improvement predicted, a regression measured");
        assert_eq!(Residual { base: 1.0, predicted: 1.0, measured: 1.0 }.relative_to_move(), None);
    }
}
