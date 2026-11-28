//! The closed form checked on real rows against a grid of rho, and the
//! statistics the report prints: quantiles, moments, the variance levels.

use super::*;

// ---------------------------------------------------------------------------
// The closed form, checked against the vectors it claims to minimise
// ---------------------------------------------------------------------------

/// Cross-check `ρ̃ = ⟨w − T, B⟩ / ⟨B, B⟩` on real rows, without using the
/// accumulators that produced it.
///
/// For each probed row the objective `‖w − (ρ·B + T)‖²` is summed **from the
/// three vectors** on a grid of ρ, and two things are required:
///
/// * the grid minimum sits on the grid point nearest `ρ̃`;
/// * the parabola through three points taken *away* from `ρ̃` (centred on
///   ρ = 1, so the check cannot pass by cancellation) has its vertex at `ρ̃`
///   to better than `1e-9` relative.
///
/// Returns the worst relative vertex error over the probed rows.
pub(super) fn grid_check(
    name: &str,
    w: &[f32],
    rr: &[f32],
    tt: &[f32],
    d_in: usize,
    probes: &[usize],
) -> Result<f64, String> {
    // `‖w − (ρ·B + T)‖²` for one row, summed straight from the vectors.
    let direct = |i: usize, rho: f64| -> f64 {
        let mut s = 0.0f64;
        for k in i * d_in..(i + 1) * d_in {
            let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr[k]), f64::from(tt[k]));
            let d = wk - tk - rho * (rk - tk);
            s += d * d;
        }
        s
    };
    let mut worst = 0.0f64;
    println!("  {name}, {} rows probed", probes.len());
    println!(
        "    {:>8} {:>12} {:>12} {:>12} {:>12}",
        "row", "ρ̃ closed", "ρ̃ vertex", "rel err", "grid argmin"
    );
    for &i in probes {
        let (mut ub, mut bb) = (0.0f64, 0.0f64);
        for k in i * d_in..(i + 1) * d_in {
            let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr[k]), f64::from(tt[k]));
            ub += (wk - tk) * (rk - tk);
            bb += (rk - tk) * (rk - tk);
        }
        let rho_tilde = ub / bb;

        // Vertex of the parabola through ρ = 1 − h, 1, 1 + h. Centred on 1 and
        // not on ρ̃: at ρ̃ the outer difference is exactly zero and the test
        // would pass on any input.
        let h = 0.05f64;
        let (jm, j0, jp) = (direct(i, 1.0 - h), direct(i, 1.0), direct(i, 1.0 + h));
        let vertex = 1.0 - h * (jp - jm) / (2.0 * (jp - 2.0 * j0 + jm));
        let rel = (vertex - rho_tilde).abs() / rho_tilde.abs();
        worst = worst.max(rel);

        // The grid: 21 points of step 0.01 around ρ̃. Its minimum must be the
        // point nearest ρ̃, which is index 10 by construction.
        let step = 0.01f64;
        let mut best = (0usize, f64::INFINITY);
        for g in 0..21usize {
            let rho = rho_tilde + (g as f64 - 10.0) * step;
            let j = direct(i, rho);
            if j < best.1 {
                best = (g, j);
            }
        }
        println!("    {i:>8} {rho_tilde:>12.8} {vertex:>12.8} {rel:>12.3e} {:>12}", best.0);
        if best.0 != 10 {
            return Err(format!(
                "{name} row {i}: the grid minimum is at index {} and not at ρ̃ (index 10). \
                 The closed form does not minimise the objective it is derived from.",
                best.0
            ));
        }
        if rel > 1e-9 {
            return Err(format!(
                "{name} row {i}: the vertex of the measured parabola is {vertex:.12} and the \
                 closed form gives {rho_tilde:.12}, {rel:.3e} relative — beyond 1e-9."
            ));
        }
    }
    println!("    worst relative error {worst:.3e}, under 1e-9. ρ̃ = ⟨w−T, B⟩/⟨B, B⟩ confirmed.\n");
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

pub(super) fn quantiles(sorted: &[f64], q: f64) -> f64 {
    let x = q * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    sorted[lo] + (x - lo as f64) * (sorted[hi] - sorted[lo])
}

pub(super) fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
    (m, s)
}

/// Pearson correlation of two series of the same length.
pub(super) fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    sxy / (sxx * syy).sqrt()
}

/// Median of a copy, so the caller keeps its order.
pub(super) fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    quantiles(&s, 0.5)
}

/// Everything one row contributes. `u = w − T` and `B = r − T` are the
/// vectors the row scale actually moves (the tail `T` is out of its reach);
/// `w` and `r` are the ones the prereg's ρ is defined on. Six numbers per row
/// are enough to give every ρ, every cosine and every `‖ΔW‖²` below exactly,
/// with no second decode.
#[derive(Clone, Copy)]
pub(super) struct RowStat {
    pub(super) uu: f64,
    pub(super) ub: f64,
    pub(super) bb: f64,
    pub(super) num: f64,
    pub(super) den: f64,
    pub(super) ww: f64,
}

impl RowStat {
    /// The prereg's ρ, §2 verbatim: `⟨w, r⟩ / ⟨r, r⟩`.
    pub(super) fn rho(self) -> f64 {
        self.num / self.den
    }

    /// The minimiser of `‖ΔW‖²` over the parameter that actually moves.
    pub(super) fn rho_tilde(self) -> f64 {
        self.ub / self.bb
    }

    /// `‖ΔW‖²` for this row when its row scale is multiplied by ρ.
    pub(super) fn j(self, rho: f64) -> f64 {
        self.uu - 2.0 * rho * self.ub + rho * rho * self.bb
    }

    /// The angle factor of the shrink: `cos ∠(w, r)`. This is the quantity M0
    /// simulated (0.960740 on gaussian blocks); here it is measured, and it
    /// carries whatever angle GPTQ's compensation opened between `w` and the
    /// block the encoder was actually given.
    pub(super) fn cos(self) -> f64 {
        self.num / (self.ww * self.den).sqrt()
    }

    /// The length factor of the shrink: `‖w‖/‖r‖`. M0 could not see it — its
    /// scales were taken on the encoded blocks — and `ρ_i = L_i · c_i` holds
    /// row by row, exactly.
    pub(super) fn len_ratio(self) -> f64 {
        (self.ww / self.den).sqrt()
    }
}

/// Which rows of the global table belong to one matrix, and what it is.
pub(super) struct MatStat {
    pub(super) name: String,
    pub(super) proj: String,
    pub(super) layer: usize,
    pub(super) first: usize,
    pub(super) len: usize,
}

impl MatStat {
    pub(super) fn rows<'a>(&self, all: &'a [RowStat]) -> &'a [RowStat] {
        &all[self.first..self.first + self.len]
    }
    pub(super) fn slice<'a>(&self, all: &'a [f64]) -> &'a [f64] {
        &all[self.first..self.first + self.len]
    }
}

/// Three-level decomposition of the variance of one number per row: between
/// projection types, between matrices of the same type, and inside a matrix.
/// Group sizes are unequal, so every level is weighted by its count; the three
/// then add up to the total exactly, which is printed rather than asserted.
pub(super) fn variance_levels(values: &[f64], mats: &[MatStat]) -> (f64, f64, f64, f64) {
    let n = values.len() as f64;
    let grand = values.iter().sum::<f64>() / n;
    let total = values.iter().map(|v| (v - grand) * (v - grand)).sum::<f64>() / n;

    let per_mat: Vec<(f64, f64, f64)> = mats
        .iter()
        .map(|m| {
            let v = m.slice(values);
            let c = v.len() as f64;
            let mean = v.iter().sum::<f64>() / c;
            let ss = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
            (c, mean, ss)
        })
        .collect();

    let mut type_n: HashMap<&str, f64> = HashMap::new();
    let mut type_s: HashMap<&str, f64> = HashMap::new();
    for (m, (c, mean, _)) in mats.iter().zip(&per_mat) {
        *type_n.entry(m.proj.as_str()).or_insert(0.0) += c;
        *type_s.entry(m.proj.as_str()).or_insert(0.0) += c * mean;
    }
    let between_type = type_n
        .iter()
        .map(|(k, c)| {
            let mt = type_s[k] / c;
            c * (mt - grand) * (mt - grand)
        })
        .sum::<f64>()
        / n;
    let between_mat = mats
        .iter()
        .zip(&per_mat)
        .map(|(m, (c, mean, _))| {
            let mt = type_s[m.proj.as_str()] / type_n[m.proj.as_str()];
            c * (mean - mt) * (mean - mt)
        })
        .sum::<f64>()
        / n;
    let within = per_mat.iter().map(|(_, _, ss)| ss).sum::<f64>() / n;
    (between_type, between_mat, within, total)
}
