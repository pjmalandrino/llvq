//! One positive temperature between a quantized model and its dense teacher.
//!
//! ## The question this answers, and the one it does not
//!
//! A gain-rescaling map fitted on the model's own NLL improved perplexity
//! 17.4 % and lost 3.06 pp of MMLU. The dumps located the NLL half of that:
//! the corrected arm's option logits collapse 32.5 %, and shrinking logits
//! improves a proper scoring rule (*measured*,
//! `docs/mesures/errmap-mmlu-4b-2026-09-16.txt`).
//!
//! That invites a different objective — match the dense model's output
//! distribution — and immediately raises the objection that decides whether
//! the objective is worth fitting: **how much of the gap to the dense model is
//! a single number?** A positive rescaling of logits cannot move an argmax, so
//! every nat of divergence a lone temperature removes is a nat that was never
//! going to move an answer. If one scalar closes most of the gap, a
//! several-hundred-parameter fit against the same objective will spend most of
//! its freedom on ground that accuracy cannot see.
//!
//! This module measures that fraction. It does **not** measure whether the
//! remainder is worth fitting, and it does not establish what causes the
//! accuracy loss: that a rescaling leaves accuracy unchanged is true by
//! construction, not evidence. All that survives from the control above is
//! narrower — the accuracy loss lives somewhere in the part of the logit
//! movement a scale cannot express — and "somewhere" is a large space.
//!
//! ## The objective
//!
//! With `p` the dense model's distribution at one position, `z` the quantized
//! model's logits there and `β = 1/T` the inverse temperature,
//!
//! ```text
//! KL(p ‖ softmax(β z)) = Σ p log p − β⟨p, z⟩ + logsumexp(β z)
//! ```
//!
//! The first two terms are affine in `β`, and `logsumexp` is convex, so **the
//! divergence is convex in `β`** — the minimum a search finds is the global
//! one, and a bracketing triple can be refined by its parabola. It is *not*
//! convex in `T`, which is why everything below is parameterized by `β`.
//!
//! The decomposition is what makes the fit cheap: `Σ p log p` and `⟨p, z⟩` do
//! not depend on `β`, so two scalars per position are accumulated once and
//! only `logsumexp(β z)` is re-evaluated per candidate. Fitting the
//! temperature therefore costs **no additional forward pass** — the model runs
//! once and the search reads its stored logits.

/// Position-summed statistics that do not depend on the temperature.
///
/// Both fields are sums over positions, not means: windows are accumulated
/// independently and the division happens once, in [`Totals::divergence`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Totals {
    /// `Σᵢ H(pᵢ)`, the dense model's entropy. Non-negative.
    pub entropy: f64,
    /// `Σᵢ ⟨pᵢ, zᵢ⟩`, the dense distribution against the quantized logits.
    pub dot: f64,
    /// How many positions were accumulated.
    pub positions: usize,
}

impl Totals {
    /// Accumulate one position's statistics.
    pub fn add(&mut self, entropy: f64, dot: f64) {
        self.entropy += entropy;
        self.dot += dot;
        self.positions += 1;
    }

    /// Accumulate a whole window, already summed over its positions.
    pub fn add_window(&mut self, entropy: f64, dot: f64, positions: usize) {
        self.entropy += entropy;
        self.dot += dot;
        self.positions += positions;
    }

    /// Mean `KL(p_dense ‖ softmax(β z))` over the accumulated positions, nats.
    ///
    /// `log_partition` is `Σᵢ logsumexp(β zᵢ)` at this same `β`, which is the
    /// only term the caller must re-evaluate per candidate.
    pub fn divergence(&self, beta: f64, log_partition: f64) -> f64 {
        if self.positions == 0 {
            return f64::NAN;
        }
        (-self.entropy - beta * self.dot + log_partition) / self.positions as f64
    }
}

/// One position's `(entropy, dot)`, from a dense distribution and quantized
/// logits over the same vocabulary.
///
/// `teacher` must be a probability vector; it is used as given, not
/// renormalized, so that a caller feeding an unnormalized vector gets a wrong
/// answer loudly rather than a right-looking one.
pub fn statistics(teacher: &[f64], student: &[f64]) -> (f64, f64) {
    assert_eq!(teacher.len(), student.len(), "teacher and student vocabularies differ");
    let mut entropy = 0.0;
    let mut dot = 0.0;
    for (&p, &z) in teacher.iter().zip(student) {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        dot += p * z;
    }
    (entropy, dot)
}

/// `logsumexp(β z)`, shifted by the maximum so that large `β` does not overflow.
pub fn log_partition(student: &[f64], beta: f64) -> f64 {
    let mut top = f64::NEG_INFINITY;
    for &z in student {
        let v = beta * z;
        if v > top {
            top = v;
        }
    }
    if !top.is_finite() {
        return top;
    }
    let mut acc = 0.0;
    for &z in student {
        acc += (beta * z - top).exp();
    }
    top + acc.ln()
}

/// `KL(p ‖ softmax(β z))` at one position, computed straight from the
/// definition.
///
/// The reference the decomposed path is tested against. Never used in the
/// measurement itself: it needs the full vocabulary at every candidate `β`,
/// which is the cost [`Totals`] exists to avoid.
pub fn divergence_direct(teacher: &[f64], student: &[f64], beta: f64) -> f64 {
    assert_eq!(teacher.len(), student.len(), "teacher and student vocabularies differ");
    let lse = log_partition(student, beta);
    let mut acc = 0.0;
    for (&p, &z) in teacher.iter().zip(student) {
        if p > 0.0 {
            acc += p * (p.ln() - (beta * z - lse));
        }
    }
    acc
}

/// The vertex of the parabola through three points, or `None` when they are
/// collinear or degenerate.
///
/// Used to refine a bracketing triple once the search has narrowed: the
/// divergence is convex in `β`, so near the minimum it is well approximated by
/// its parabola and one interpolation is worth several more evaluations.
pub fn parabola_vertex(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> Option<f64> {
    let (x0, y0) = p0;
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let d0 = x1 - x0;
    let d2 = x1 - x2;
    let num = d0 * d0 * (y1 - y2) - d2 * d2 * (y1 - y0);
    let den = d0 * (y1 - y2) - d2 * (y1 - y0);
    if den.abs() < f64::EPSILON || !num.is_finite() || !den.is_finite() {
        return None;
    }
    let v = x1 - 0.5 * num / den;
    v.is_finite().then_some(v)
}

/// What a temperature fit found, and what it is worth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    /// The inverse temperature that minimizes the divergence.
    pub beta: f64,
    /// Mean divergence there, nats per position.
    pub divergence: f64,
    /// Mean divergence at `β = 1`, the untouched model.
    pub baseline: f64,
    /// Whether the search stopped against an end of its bracket, in which case
    /// `beta` is a boundary and not a minimum.
    pub at_bound: bool,
}

impl Fit {
    /// `T = 1/β`, the temperature itself.
    pub fn temperature(&self) -> f64 {
        1.0 / self.beta
    }

    /// The fraction of the baseline divergence a single temperature removes.
    ///
    /// Negative when the fit is evaluated on a split it was not fitted on and
    /// does not transfer — which is a result, so it is reported rather than
    /// clamped.
    pub fn recovered(&self) -> f64 {
        if self.baseline == 0.0 {
            return f64::NAN;
        }
        (self.baseline - self.divergence) / self.baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(z: &[f64], beta: f64) -> Vec<f64> {
        let lse = log_partition(z, beta);
        z.iter().map(|&v| (beta * v - lse).exp()).collect()
    }

    /// A deterministic pseudo-vocabulary, so the tests read the same numbers
    /// on every machine.
    fn logits(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) * 6.0 - 3.0
            })
            .collect()
    }

    #[test]
    fn decomposed_divergence_matches_the_definition() {
        let z = logits(64, 7);
        let p = softmax(&logits(64, 11), 1.0);
        for &beta in &[0.4, 0.8, 1.0, 1.3, 2.5] {
            let (e, d) = statistics(&p, &z);
            let mut t = Totals::default();
            t.add(e, d);
            let via = t.divergence(beta, log_partition(&z, beta));
            let direct = divergence_direct(&p, &z, beta);
            assert!(
                (via - direct).abs() < 1e-12,
                "beta {beta}: decomposed {via} against direct {direct}"
            );
        }
    }

    /// Two positions, not one: the single-position tests cannot tell a mean
    /// from a sum, and the mean is what compares a fit split against a
    /// validation split of a different size.
    #[test]
    fn the_divergence_is_a_mean_over_positions() {
        let (z0, z1) = (logits(64, 41), logits(64, 43));
        let p0 = softmax(&logits(64, 47), 1.0);
        let p1 = softmax(&logits(64, 53), 1.0);
        let beta = 0.83;
        let mut t = Totals::default();
        for (p, z) in [(&p0, &z0), (&p1, &z1)] {
            let (e, d) = statistics(p, z);
            t.add(e, d);
        }
        let lse = log_partition(&z0, beta) + log_partition(&z1, beta);
        let via = t.divergence(beta, lse);
        let want = 0.5
            * (divergence_direct(&p0, &z0, beta) + divergence_direct(&p1, &z1, beta));
        assert_eq!(t.positions, 2);
        assert!((via - want).abs() < 1e-12, "mean read as {via}, want {want}");
        // And it is a MEAN: the sum of the two is twice that, so a harness
        // that forgot to divide would read here.
        assert!((via - 2.0 * want).abs() > 1e-6, "a sum is passing as a mean");
    }

    #[test]
    fn a_model_that_matches_its_teacher_has_no_divergence() {
        let z = logits(32, 3);
        let p = softmax(&z, 1.0);
        assert!(divergence_direct(&p, &z, 1.0).abs() < 1e-12);
    }

    #[test]
    fn divergence_is_never_negative() {
        let z = logits(48, 5);
        let p = softmax(&logits(48, 9), 1.0);
        for &beta in &[0.1, 0.5, 1.0, 2.0, 8.0] {
            assert!(divergence_direct(&p, &z, beta) >= -1e-12);
        }
    }

    /// The planted case: a teacher that IS the student at a known temperature.
    /// The minimum must land on that temperature and nowhere else — the test
    /// that a sign error or a missing factor cannot pass.
    #[test]
    fn a_planted_temperature_is_recovered() {
        let z = logits(128, 13);
        for &t0 in &[0.6, 1.0, 1.7] {
            let p = softmax(&z, 1.0 / t0);
            let best = (1..=4000)
                .map(|i| i as f64 * 0.001)
                .min_by(|&a, &b| {
                    divergence_direct(&p, &z, a)
                        .partial_cmp(&divergence_direct(&p, &z, b))
                        .unwrap()
                })
                .unwrap();
            assert!(
                (best - 1.0 / t0).abs() < 2e-3,
                "planted T {t0} (beta {:.4}) recovered as beta {best:.4}",
                1.0 / t0
            );
            assert!(divergence_direct(&p, &z, 1.0 / t0).abs() < 1e-12);
        }
    }

    /// Convexity in `β` is what licenses the parabolic refinement and makes a
    /// bracketed minimum the global one. Checked as a non-negative second
    /// difference rather than asserted in the doc comment.
    #[test]
    fn divergence_is_convex_in_beta_not_in_temperature() {
        let z = logits(96, 17);
        let p = softmax(&logits(96, 23), 1.0);
        let h = 0.01;
        let mut convex_in_beta = true;
        for i in 1..200 {
            let b = 0.1 + i as f64 * h;
            let second = divergence_direct(&p, &z, b + h) - 2.0 * divergence_direct(&p, &z, b)
                + divergence_direct(&p, &z, b - h);
            convex_in_beta &= second >= -1e-9;
        }
        assert!(convex_in_beta, "the divergence is not convex in beta");
    }

    /// The control the whole measurement rests on, stated as a test: no
    /// positive temperature moves a single argmax. A fit that changed an
    /// answer would mean the harness is not doing what it claims.
    #[test]
    fn no_positive_temperature_moves_an_argmax() {
        let z = logits(256, 29);
        let top = |beta: f64| {
            let s = softmax(&z, beta);
            s.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let reference = top(1.0);
        for &beta in &[0.05, 0.3, 1.0, 3.0, 20.0] {
            assert_eq!(top(beta), reference, "beta {beta} moved the argmax");
        }
    }

    #[test]
    fn the_parabola_vertex_is_exact_on_a_parabola() {
        // 3(x − 0.7)² + 2, minimum at 0.7.
        let f = |x: f64| 3.0 * (x - 0.7) * (x - 0.7) + 2.0;
        let v = parabola_vertex((0.2, f(0.2)), (0.6, f(0.6)), (1.4, f(1.4))).unwrap();
        assert!((v - 0.7).abs() < 1e-9, "vertex read as {v}");
    }

    #[test]
    fn collinear_points_have_no_vertex() {
        assert!(parabola_vertex((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)).is_none());
    }

    #[test]
    fn a_fit_reports_what_a_temperature_bought() {
        let f = Fit { beta: 0.8, divergence: 0.25, baseline: 1.0, at_bound: false };
        assert!((f.temperature() - 1.25).abs() < 1e-12);
        assert!((f.recovered() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn a_fit_that_does_not_transfer_reports_a_negative_fraction() {
        let f = Fit { beta: 1.2, divergence: 1.4, baseline: 1.0, at_bound: false };
        assert!(f.recovered() < 0.0, "a worse divergence must read as negative");
    }

    #[test]
    fn totals_sum_over_windows() {
        let mut a = Totals::default();
        a.add(1.0, 2.0);
        a.add(3.0, 4.0);
        let mut b = Totals::default();
        b.add_window(4.0, 6.0, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn log_partition_survives_a_large_beta() {
        let z = logits(64, 31);
        let v = log_partition(&z, 400.0);
        let top = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(v.is_finite(), "logsumexp overflowed");
        assert!((v - 400.0 * top).abs() < 1.0, "logsumexp lost its maximum");
    }
}
