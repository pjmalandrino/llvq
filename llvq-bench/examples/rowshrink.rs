//! M0: is the row axis of chantier 14 a constant in disguise?
//!
//! Prereg `proofs/preregistration-m0-echelles-2026-09-07.md`, timestamped
//! before this file ran. The bench measures and prints; it decides nothing.
//!
//! `nice -n 10 cargo run --release -p llvq-bench --example rowshrink -- [threads]`
//!
//! ## The fact it starts from
//!
//! The shipped encoder picks the gain level on the block **norm**
//! (`llvq-quant/src/quantizer.rs:777`, `nearest_level_index(centroids,
//! norm / row_scale)`), while the paper picks it by least squares against the
//! direction it kept (`docs/llvq-paper-notes.md:195`, `β* = q(w)ᵀw / q(w)ᵀq(w)`,
//! that is `‖x‖·cos θ`). Row scales do not absorb the difference: `gptq.rs:245`
//! fixes them before the block loop, at the RMS of the original row.
//!
//! ## What it computes
//!
//! For block `p` of a simulated row of scale `s`, kept point `y_p`, unit
//! direction `û_p = y_p/‖y_p‖` and kept gain centroid `c_p`:
//!
//! ```text
//!   a_p   = ‖x_p‖ / s                     what the encoder quantizes today
//!   τ_p   = ⟨x_p, û_p⟩ / s                what the paper would quantize
//!   served reconstruction = c_p · s · û_p
//!   J(t)  = Σ_p ‖x_p − t·c_p·s·û_p‖²
//!   ρ*    = Σ_p c_p τ_p / Σ_p c_p²        (prereg §3, closed form)
//!   R     = (J(1) − J(ρ*)) / J(1)
//! ```
//!
//! `J(t) = Σ_p (‖x_p‖² − 2t·c_p·s·⟨x_p, û_p⟩ + t²·c_p²·s²)` is quadratic in
//! `t`, so its minimizer is `Σ_p c_p s² τ_p / Σ_p c_p² s²`. That equals the
//! prereg's `ρ*` when every row shares one scale, and differs from it only
//! through the spread of `s` across rows; both are printed. At the prereg's
//! `ρ*` the drop has the closed form `J(1) − J(ρ*) = s²·Σc²·(1 − ρ*)²` on a
//! single row, which is what `Row::j` evaluates row by row and what
//! `brute_force` re-evaluates on the 24 coordinates of each block.
//!
//! `⟨x, û⟩` is read off `TetraCode::t`, which is already `⟨x, y⟩/‖y‖`
//! (`llvq_search::tetra::encoder::t_of`): it must not be divided by `‖y‖`
//! twice.
//!
//! ## Which of the four numbers are algebra
//!
//! Three of them are, and the bench prints the residual of each identity next
//! to the number it forces.
//!
//! `τ_p = a_p·cos θ_p`, so `ρ*` factors exactly into `G · C` with
//! `G = Σ c·a / Σ c²` and `C = Σ c·τ / Σ c·a`, the `c·a`-weighted mean of
//! `cos θ`. A Lloyd–Max centroid is the mean of its cell, so `Σ_p c_p(a_p − c_p)
//! = 0` cell by cell on the population the centroids were fitted on. In the
//! served arm that population is `a`, so `G = 1` and the pooled `ρ*` is a
//! weighted mean of `cos θ`, a property of the Tetra lattice and not of any
//! row. In the paper arm the fitted population is `τ`, so `Σ c·τ = Σ c²` and
//! both `ρ* = 1` and `R = 0` are the fixed point rather than a measurement.
//!
//! `R` follows too: `J(1) − J(ρ*) = (1 − ρ*)²·Σ s²c²` up to the gap between
//! `ρ*` and the `s²`-weighted argmin, and `Σ s²c²/(24·n) ≈ E‖x‖²/24 = 1`
//! because the gain code reproduces the block norm, so `R ≈ (1 − ρ*)²/MSE`.
//!
//! What is left as measurement is the retention of each arm, the spread of the
//! per-row `ρ*`, and `cos θ` itself.
//!
//! ## What `σ(ρ*)` is, and what it is not
//!
//! `σ(ρ*)` is the standard deviation of `G·C` across rows, and the bench
//! prints the two channels separately because they are not the same quantity.
//! `C` is the angular channel the prereg §6 modelled, at `sd(1 − cos)/√n`.
//! `G` is the residue of the 1 bit gain code against the row's own sample of
//! block norms. Three witnesses date the total: `σ` with the gain code, `σ`
//! with a free gain (`c = a`, no code), and `σ` on a shuffled block order.
//!
//! ## Two limits this bench carries by construction
//!
//! Row scales here are computed on the same blocks the bench encodes.
//! Production fixes them on the original row before the GPTQ loop
//! (`llvq-quant/src/gptq.rs:245`) and then quantizes residuals whose norms
//! have drifted. The scale disagreement is therefore absent from the
//! simulation, and `ρ*` prices the `1/cos θ` radial bias alone.
//!
//! The blocks are i.i.d. Gaussian. Rows are exchangeable, so `σ(ρ*)` is a
//! sampling statistic in `1/√(row length)`; the sweep at the end reads that
//! law off the bench itself.

#![forbid(unsafe_code)]

use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::{Leech, SplitMix64};
use llvq_quant::quantizer::{fit_gain_centroids, reconstruct_shape_gain, row_scale, BlockQuantizer, TetraShapeGain};
use llvq_search::tetra::{Encoder, Scratch, Tetra};
use std::time::Instant;

/// Prereg §3.
const SEED: u64 = 0x0f1b_2026_0907;
const N_BLOCKS: usize = 200_000;
/// The two `d_in` of the 4B, in blocks of 24: 2560/24 and 9728/24.
const ROW_SIZES: [usize; 2] = [106, 405];
/// One gain bit, as Tetra ships it; 40 Lloyd iterations, as production fits.
const GAIN_BITS: u32 = 1;
const LLOYD_ITERS: usize = 40;
/// Tetra spends 48 bits per block of 24 weights.
const BLOCK_BITS: f64 = 48.0;
/// The Gaussian retention the F1b journal pins for this encoder.
const RETENTION_REF: f64 = 88.89;
/// Rows of the 106 grouping re-encoded with their points, for brute force.
const BRUTE_ROWS: usize = 40;
/// Blocks the reconstruction identity is checked on.
const IDENTITY_BLOCKS: usize = 256;
/// Row lengths the sweep reads the law of `σ` on, the two prereg sizes included.
const SWEEP: [usize; 9] = [6, 12, 24, 53, 106, 200, 405, 800, 1600];
/// Seeds of the shuffle witness, distinct from the data seed. A null control
/// measured once is one draw, and its own spread is what makes it readable.
const SHUFFLE_SEEDS: [u64; 5] = [0x5eed_0f1b_2026_0907, 0x5eed_0002, 0x5eed_0003, 0x5eed_0004, 0x5eed_0005];
/// The two thresholds the prereg §5 states in units of `1 − ρ*`.
const GATE_LO: f64 = 0.15;
const GATE_HI: f64 = 0.40;

/// `f` over `0..n`, `threads` wide, one `S` per thread; results in order.
fn par_map<S, T: Send, I: Fn() -> S + Sync, F: Fn(&mut S, usize) -> T + Sync>(n: usize, threads: usize, init: I, f: F) -> Vec<T> {
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let mut out = Vec::with_capacity(n);
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let (init, f) = (&init, &f);
                let end = (start + chunk).min(n);
                sc.spawn(move || {
                    let mut s = init();
                    (start..end).map(|i| f(&mut s, i)).collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().expect("thread"));
        }
    });
    out
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation, `n − 1` at the denominator.
fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// Sample covariance, `n − 1` at the denominator.
fn cov(u: &[f64], v: &[f64]) -> f64 {
    let (mu, mv) = (mean(u), mean(v));
    u.iter().zip(v).map(|(x, y)| (x - mu) * (y - mv)).sum::<f64>() / (u.len() as f64 - 1.0)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

/// Relative gap, or the absolute one when the reference is zero.
fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        (a - b).abs()
    } else {
        (a - b).abs() / b.abs()
    }
}

/// One simulated row, reduced to the five sums every published number needs.
#[derive(Clone, Copy)]
struct Row {
    /// `Σ_p c_p τ_p`.
    a: f64,
    /// `Σ_p c_p a_p`, the gain channel's numerator.
    ca: f64,
    /// `Σ_p c_p²`.
    b: f64,
    /// `Σ_p ‖x_p‖²`.
    xx: f64,
    /// The row scale, `llvq_quant::quantizer::row_scale`.
    s: f64,
}

impl Row {
    /// `J(t)` restricted to this row.
    fn j(&self, t: f64) -> f64 {
        self.xx - 2.0 * t * self.s * self.s * self.a + t * t * self.s * self.s * self.b
    }

    /// The row's own `ρ*`; the scale cancels, so the prereg form is exact here.
    fn rho(&self) -> f64 {
        self.a / self.b
    }

    /// `G = Σ c·a / Σ c²`, the gain code's residue on this row's norms.
    fn g(&self) -> f64 {
        self.ca / self.b
    }

    /// `C = Σ c·τ / Σ c·a`, the `c·a`-weighted mean of `cos θ` on this row.
    fn wcos(&self) -> f64 {
        self.a / self.ca
    }
}

/// The rows of one grouping, given the kept centroid of every block.
fn rows_of(xx: &[f64], tdot: &[f64], c: &[f64], scales: &[f64], row_blocks: usize) -> Vec<Row> {
    scales
        .iter()
        .enumerate()
        .map(|(r, &s)| {
            let rg = r * row_blocks..(r + 1) * row_blocks;
            let (mut a, mut ca, mut b, mut sxx) = (0.0, 0.0, 0.0, 0.0);
            for ((&x2, &td), &cp) in xx[rg.clone()].iter().zip(&tdot[rg.clone()]).zip(&c[rg]) {
                a += cp * td / s;
                ca += cp * x2.sqrt() / s;
                b += cp * cp;
                sxx += x2;
            }
            Row { a, ca, b, xx: sxx, s }
        })
        .collect()
}

/// Standard deviation of the per-row `ρ*` of a grouping.
fn sigma_of(xx: &[f64], tdot: &[f64], c: &[f64], scales: &[f64], row_blocks: usize) -> f64 {
    let rows = rows_of(xx, tdot, c, scales, row_blocks);
    sd(&rows.iter().map(Row::rho).collect::<Vec<_>>())
}

/// One grouping of the blocks into simulated rows, with its gain code.
struct Grouping {
    row_blocks: usize,
    n_rows: usize,
    /// Blocks used, `n_rows · row_blocks`; the tail is dropped.
    n: usize,
    scales: Vec<f64>,
    /// `a_p = ‖x_p‖ / s`, the population the served centroids are fitted on.
    a: Vec<f64>,
    centroids: Vec<f64>,
    /// The centroid the served rule keeps for each block.
    c: Vec<f64>,
}

impl Grouping {
    /// The gain population and code, once the scales are known.
    fn finish(row_blocks: usize, scales: Vec<f64>, xx: &[f64], centroids: Option<Vec<f64>>) -> Self {
        let n_rows = scales.len();
        let n = n_rows * row_blocks;
        let a: Vec<f64> = (0..n).map(|i| xx[i].sqrt() / scales[i / row_blocks]).collect();
        let centroids = centroids.unwrap_or_else(|| lloyd_max(&a, GAIN_BITS, LLOYD_ITERS));
        let c = a.iter().map(|&g| centroids[nearest_centroid(&centroids, g)]).collect();
        Self {
            row_blocks,
            n_rows,
            n,
            scales,
            a,
            centroids,
            c,
        }
    }

    /// The production path: `row_scale` on the weights, `fit_gain_centroids`
    /// over the whole matrix.
    fn production(flat: &[f64], xx: &[f64], row_blocks: usize) -> Self {
        let n_rows = xx.len() / row_blocks;
        let d_in = row_blocks * DIM;
        let scales: Vec<f64> = (0..n_rows).map(|r| row_scale(&flat[r * d_in..(r + 1) * d_in])).collect();
        let centroids = fit_gain_centroids(&flat[..n_rows * d_in], n_rows, d_in, DIM, GAIN_BITS, LLOYD_ITERS);
        Self::finish(row_blocks, scales, xx, Some(centroids))
    }

    /// The same grouping from the block norms alone: `row_scale` is the RMS
    /// block norm, so `xx` carries everything it reads. Used by the witnesses,
    /// which regroup blocks without rebuilding their coordinates.
    fn from_xx(xx: &[f64], row_blocks: usize) -> Self {
        let n_rows = xx.len() / row_blocks;
        let scales: Vec<f64> = (0..n_rows)
            .map(|r| (xx[r * row_blocks..(r + 1) * row_blocks].iter().sum::<f64>() / row_blocks as f64).sqrt())
            .collect();
        Self::finish(row_blocks, scales, xx, None)
    }
}

/// What every arm of one grouping is read against.
struct Ctx {
    row_blocks: usize,
    /// Unweighted mean of `cos θ` over all blocks.
    mean_cos: f64,
    /// `sd(1 − cos θ) = sd(cos θ)` over all blocks, the prereg §6 input.
    sd_cos: f64,
}

/// The published triple of one arm, plus what reads it back against the file.
struct Arm {
    /// `ρ*` in the prereg's form, over every block of the grouping.
    rho: f64,
    /// The exact minimizer of the global `J`, `s²`-weighted.
    rho_argmin: f64,
    /// `R = (J(1) − J(ρ*)) / J(1)`, as a fraction.
    r: f64,
    /// Standard deviation of the per-row `ρ*`.
    sigma: f64,
    /// Per-row `ρ*`, sorted.
    rho_rows: Vec<f64>,
    /// Per-row `R`, sorted, as fractions.
    r_rows: Vec<f64>,
    /// Retention at `t = 1`, the served reconstruction.
    retention: f64,
    /// Retention at `t = ρ*`, same codes and same bits.
    retention_shrunk: f64,
    /// `J(1)` per weight, the arm's mean squared error.
    mse: f64,
    /// Pooled `G = Σ c·a / Σ c²`; 1 when the centroids were fitted on `a`.
    g_pool: f64,
    /// Pooled `C = Σ c·τ / Σ c·a`, the `c·a`-weighted mean of `cos θ`.
    c_pool: f64,
    /// Mean and spread of the two channels across rows.
    g_mean: f64,
    g_sd: f64,
    c_mean: f64,
    c_sd: f64,
    /// Covariance of the two channels across rows.
    gc_cov: f64,
    /// True when the level was picked on `τ`, which forces `ρ* = 1`.
    on_tau: bool,
}

fn summarize(rows: &[Row], n_blocks: usize, rate: f64, on_tau: bool) -> Arm {
    let rho = rows.iter().map(|r| r.a).sum::<f64>() / rows.iter().map(|r| r.b).sum::<f64>();
    let w_a: f64 = rows.iter().map(|r| r.s * r.s * r.a).sum();
    let w_b: f64 = rows.iter().map(|r| r.s * r.s * r.b).sum();
    let j1: f64 = rows.iter().map(|r| r.j(1.0)).sum();
    let jr: f64 = rows.iter().map(|r| r.j(rho)).sum();
    let mut rho_rows: Vec<f64> = rows.iter().map(Row::rho).collect();
    let mut r_rows: Vec<f64> = rows.iter().map(|r| (r.j(1.0) - r.j(r.rho())) / r.j(1.0)).collect();
    let g_rows: Vec<f64> = rows.iter().map(Row::g).collect();
    let c_rows: Vec<f64> = rows.iter().map(Row::wcos).collect();
    let sigma = sd(&rho_rows);
    rho_rows.sort_unstable_by(f64::total_cmp);
    r_rows.sort_unstable_by(f64::total_cmp);
    let per_weight = (DIM * n_blocks) as f64;
    Arm {
        rho,
        rho_argmin: w_a / w_b,
        r: (j1 - jr) / j1,
        sigma,
        rho_rows,
        r_rows,
        retention: retention_pct(j1 / per_weight, rate),
        retention_shrunk: retention_pct(jr / per_weight, rate),
        mse: j1 / per_weight,
        g_pool: rows.iter().map(|r| r.ca).sum::<f64>() / rows.iter().map(|r| r.b).sum::<f64>(),
        c_pool: rows.iter().map(|r| r.a).sum::<f64>() / rows.iter().map(|r| r.ca).sum::<f64>(),
        g_mean: mean(&g_rows),
        g_sd: sd(&g_rows),
        c_mean: mean(&c_rows),
        c_sd: sd(&c_rows),
        gc_cov: cov(&g_rows, &c_rows),
        on_tau,
    }
}

fn report(label: &str, arm: &Arm, ctx: &Ctx) {
    let n = arm.rho_rows.len() as f64;
    println!("  {label}");
    println!(
        "    rho*                {:.6}          rows: mean {:.6}, deciles {:.6} / {:.6}, min {:.6}, max {:.6}",
        arm.rho,
        mean(&arm.rho_rows),
        quantile(&arm.rho_rows, 0.10),
        quantile(&arm.rho_rows, 0.90),
        arm.rho_rows[0],
        arm.rho_rows[arm.rho_rows.len() - 1]
    );
    println!("    exact argmin of J   {:.6}          gap to rho* {:+.2e} relative", arm.rho_argmin, (arm.rho_argmin - arm.rho) / arm.rho);
    println!(
        "    R                   {:.4} %          rows: quartiles {:.4} / {:.4} / {:.4} %",
        100.0 * arm.r,
        100.0 * quantile(&arm.r_rows, 0.25),
        100.0 * quantile(&arm.r_rows, 0.50),
        100.0 * quantile(&arm.r_rows, 0.75)
    );
    println!("    sigma(rho*)         {:.6}          the sd of the per-row rho* above", arm.sigma);
    println!("      standard error of sigma itself       {:.6}   sigma / sqrt(2*(rows - 1))", arm.sigma / (2.0 * (n - 1.0)).sqrt());
    println!("      standard error of the row mean       {:.6}   sigma / sqrt(rows), the error bar of the published rho*", arm.sigma / n.sqrt());
    // The thresholds of §5 are stated in units of `1 − ρ*`. The paper arm's
    // `ρ*` is 1 by construction, so the ratio is not defined there.
    if (1.0 - arm.rho).abs() > 1e-3 {
        println!(
            "      sigma / (1 - rho*)                   {:.4}     the prereg §5 reads this against {GATE_LO:.2} and {GATE_HI:.2}; this bench applies neither",
            arm.sigma / (1.0 - arm.rho)
        );
    } else {
        println!("      sigma / (1 - rho*)                   undefined, rho* = 1");
    }
    println!("    retention           {:.2} % at t = 1     {:.2} % at t = rho*   ({:+.2} pp)", arm.retention, arm.retention_shrunk, arm.retention_shrunk - arm.retention);

    println!("    identities behind the numbers above, computed and not measured");
    if arm.on_tau {
        println!("      rho* = 1 and R = 0 are this arm's Lloyd fixed point, not a measurement: a centroid is");
        println!("      the mean of its cell of tau, so sum c(tau - c) = 0 cell by cell, so sum c*tau = sum c^2.");
        println!("      residual sum(c*tau)/sum(c^2) - 1 = {:+.1e}, the convergence of 40 Lloyd iterations", arm.rho - 1.0);
        println!("      what this arm prices is its retention and the spread of its per-row rho*, nothing else");
    } else {
        println!(
            "      rho* = G * C = {:.10} * {:.6}, exact factorisation, residual {:+.1e}",
            arm.g_pool,
            arm.c_pool,
            arm.g_pool * arm.c_pool - arm.rho
        );
        println!(
            "      G - 1 = {:+.1e}: the Lloyd fixed point on a, sum c(a - c) = 0 cell by cell, so rho* is a weighted mean of cos(theta)",
            arm.g_pool - 1.0
        );
        println!(
            "      C = {:.6} against the unweighted mean cos(theta) {:.6}, apart by {:.1e}",
            arm.c_pool,
            ctx.mean_cos,
            (arm.c_pool - ctx.mean_cos).abs()
        );
        let r_pred = (1.0 - arm.rho).powi(2) / arm.mse;
        println!(
            "      R against (1 - rho*)^2 / MSE: {:.4} % and {:.4} %, apart by {:.2} % relative (MSE {:.6} per weight)",
            100.0 * arm.r,
            100.0 * r_pred,
            100.0 * rel(r_pred, arm.r),
            arm.mse
        );
    }

    let (tg, tc) = ((arm.g_sd * arm.c_mean).powi(2), (arm.c_sd * arm.g_mean).powi(2));
    let tx = 2.0 * arm.g_mean * arm.c_mean * arm.gc_cov;
    let v = arm.sigma * arm.sigma;
    println!("    what sigma is made of, per row: rho* = G * C with G = sum(c*a)/sum(c^2), C = the c*a-weighted cos(theta)");
    println!("      G, the gain code residue on the row's norms   mean {:.6}   sd {:.6}   variance share {:+7.1} %", arm.g_mean, arm.g_sd, 100.0 * tg / v);
    println!("      C, the angular channel                        mean {:.6}   sd {:.6}   variance share {:+7.1} %", arm.c_mean, arm.c_sd, 100.0 * tc / v);
    println!("      covariance of G and C                                              {:+.2e}   variance share {:+7.1} %", arm.gc_cov, 100.0 * tx / v);
    println!("      the three shares sum to {:.1} % of sigma^2, first order", 100.0 * (tg + tc + tx) / v);
    println!(
        "      the prereg §6 motif, sd(1 - cos)/sqrt(row blocks) = {:.6} against sigma {:.6}, factor {:.2}   [sd(1 - cos) = {:.6}]",
        ctx.sd_cos / (ctx.row_blocks as f64).sqrt(),
        arm.sigma,
        arm.sigma * (ctx.row_blocks as f64).sqrt() / ctx.sd_cos,
        ctx.sd_cos
    );
}

/// The identity every number below rests on: the served reconstruction of a
/// block is `c_p · s · û_p`, and `‖y_p‖² = 16·shell`.
///
/// Checked against `TetraShapeGain::quantize` and against
/// `reconstruct_shape_gain`, the routine an artifact decoder runs.
fn check_reconstruction(flat: &[f64], centroids: &[f64], s: f64) {
    let mut q = TetraShapeGain::new(centroids.to_vec());
    q.set_row_scale(s);
    let enc = Encoder::new(&Tetra::new());
    let mut scratch = Scratch::new();
    let (mut worst_shell, mut worst_rec, mut worst_dec, mut worst_t) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut out = [0.0f64; DIM];
    let mut dec = [0.0f64; DIM];
    for blk in flat.chunks_exact(DIM).take(IDENTITY_BLOCKS) {
        let x: &[f64; DIM] = blk.try_into().expect("block of 24");
        q.quantize(blk, &mut out);
        let code = q.last_code().expect("a code per block");
        let m = Leech::shell_index(&code.point).expect("a point on a shell");
        let nn: f64 = code.point.iter().map(|&p| (p as f64) * (p as f64)).sum();
        worst_shell = worst_shell.max((nn - (16 * m) as f64).abs());
        let g = centroids[code.gain as usize] * s;
        reconstruct_shape_gain(&code, centroids, s, &mut dec);
        for ((&o, &d), &p) in out.iter().zip(dec.iter()).zip(code.point.iter()) {
            let want = g * p as f64 / nn.sqrt();
            worst_rec = worst_rec.max((o - want).abs() / g);
            worst_dec = worst_dec.max((d - o).abs() / g);
        }
        // The fast path used below must return the encoder's own point and t.
        let fast = enc.encode(x, &mut scratch);
        assert_eq!(fast.point, code.point, "the fast path left the encoder's point");
        let dot: f64 = x.iter().zip(&code.point).map(|(&a, &p)| a * p as f64).sum();
        worst_t = worst_t.max((fast.t - dot / nn.sqrt()).abs() / fast.t.abs());
    }
    println!("Reconstruction identity, {IDENTITY_BLOCKS} blocks, row scale {s:.4}");
    println!("  ||y||^2 - 16*shell                        max {worst_shell:.1e}");
    println!("  quantize() vs c*s*u, relative to c*s       max {worst_rec:.2e}");
    println!("  reconstruct_shape_gain() vs quantize()     max {worst_dec:.2e}");
    println!("  TetraCode::t vs <x, y>/||y||, relative     max {worst_t:.2e}");
    assert!(worst_shell == 0.0 && worst_rec < 1e-15 && worst_dec == 0.0 && worst_t < 1e-14, "the served reconstruction is not c_p*s*u_p: the bench rests on a false identity");
    println!("  the served reconstruction is c_p * s * u_p, to machine precision\n");
}

/// The closed form of `J` against `J` evaluated on the 24 coordinates.
fn brute_force(flat: &[f64], xx: &[f64], tdot: &[f64], g: &Grouping, threads: usize) {
    let row_blocks = g.row_blocks;
    let n = BRUTE_ROWS * row_blocks;
    let enc = Encoder::new(&Tetra::new());
    let points = par_map(n, threads, Scratch::new, |sc, i| {
        let x: &[f64; DIM] = flat[i * DIM..(i + 1) * DIM].try_into().expect("block of 24");
        enc.encode(x, sc).point
    });
    let (c, scales) = (&g.c, &g.scales);
    let rows = rows_of(&xx[..n], &tdot[..n], &c[..n], &scales[..BRUTE_ROWS], row_blocks);
    let closed = |t: f64| rows.iter().map(|r| r.j(t)).sum::<f64>();
    let direct = |t: f64| {
        (0..n)
            .map(|i| {
                let s = scales[i / row_blocks];
                let gain = t * c[i] * s;
                let nn: f64 = points[i].iter().map(|&p| (p as f64) * (p as f64)).sum::<f64>().sqrt();
                flat[i * DIM..(i + 1) * DIM].iter().zip(&points[i]).map(|(&v, &p)| (v - gain * p as f64 / nn).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
    };
    let rho = rows.iter().map(|r| r.a).sum::<f64>() / rows.iter().map(|r| r.b).sum::<f64>();
    let grid: Vec<f64> = (0..61).map(|i| 0.90 + 0.0025 * i as f64).collect();
    let mut worst = (0.0f64, f64::NAN);
    for &t in grid.iter().chain([rho, 1.0].iter()) {
        let (a, b) = (closed(t), direct(t));
        let gap = (a - b).abs() / b;
        if gap > worst.0 {
            worst = (gap, t);
        }
    }
    let best = grid.iter().copied().fold((f64::INFINITY, f64::NAN), |acc, t| {
        let v = direct(t);
        if v < acc.0 {
            (v, t)
        } else {
            acc
        }
    });
    println!("Closed form against direct evaluation, {BRUTE_ROWS} rows of {row_blocks} blocks ({n} blocks)");
    println!("  J(1)      closed {:.9e}   direct {:.9e}", closed(1.0), direct(1.0));
    println!("  J(rho*)   closed {:.9e}   direct {:.9e}   rho* {rho:.6}", closed(rho), direct(rho));
    println!("  worst relative gap over the 63 values of t   {:.2e} at t = {:.4}", worst.0, worst.1);
    println!("  grid minimum of the direct J at t = {:.4}, J = {:.9e}; direct J(rho*) = {:.9e}", best.1, best.0, direct(rho));
    if worst.0 > 1e-9 {
        println!("  DISAGREEMENT: the closed form and the direct evaluation differ by more than 1e-9 relative");
    } else {
        println!("  the two agree to better than 1e-9 relative");
    }
    if direct(rho) > best.0 {
        println!("  DISAGREEMENT: a grid point beats rho*, so rho* is not the minimizer");
    }
    println!();
}

/// The three witnesses that date `σ`: a free gain, a shuffled block order, and
/// the length law. None of them is one of the prereg's four numbers.
fn witnesses(g: &Grouping, xx: &[f64], tdot: &[f64], arm: &Arm) {
    let (n, row_blocks) = (g.n, g.row_blocks);
    // Free gain: c = a, the same rule with the gain code removed.
    let sigma_free = sigma_of(&xx[..n], &tdot[..n], &g.a, &g.scales, row_blocks);
    // Shuffled order: the rows are rebuilt on a permutation of the blocks, one
    // grouping per seed, so the control carries the spread of its own estimator.
    let mut shuf: Vec<f64> = SHUFFLE_SEEDS
        .iter()
        .map(|&seed| {
            let mut rng = SplitMix64::new(seed);
            let mut idx: Vec<usize> = (0..xx.len()).collect();
            for i in (1..idx.len()).rev() {
                idx.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            let xs: Vec<f64> = idx.iter().map(|&i| xx[i]).collect();
            let ts: Vec<f64> = idx.iter().map(|&i| tdot[i]).collect();
            let gs = Grouping::from_xx(&xs, row_blocks);
            sigma_of(&xs[..gs.n], &ts[..gs.n], &gs.c, &gs.scales, row_blocks)
        })
        .collect();
    shuf.sort_unstable_by(f64::total_cmp);
    // The block-norm shortcut the two witnesses above rely on.
    let plain = Grouping::from_xx(xx, row_blocks);
    let scale_gap = plain.scales.iter().zip(&g.scales).map(|(&a, &b)| rel(a, b)).fold(0.0f64, f64::max);
    let cent_gap = plain.centroids.iter().zip(&g.centroids).map(|(&a, &b)| (a - b).abs()).fold(0.0f64, f64::max);

    let sqn = (row_blocks as f64).sqrt();
    let cross = |k: f64| (arm.sigma * sqn / (k * (1.0 - arm.rho))).powi(2).ceil();
    println!("  witnesses on sigma, outside the prereg's four numbers");
    println!("    served gain code, 1 bit                      {:.6}", arm.sigma);
    println!("    free gain, c = a, the code removed           {:.6}   served over free {:.2}", sigma_free, arm.sigma / sigma_free);
    println!(
        "    blocks shuffled before grouping              {:.6}   median of {} permutations, range {:.6} to {:.6}; i.i.d. blocks, so rows are exchangeable",
        shuf[shuf.len() / 2],
        shuf.len(),
        shuf[0],
        shuf[shuf.len() - 1]
    );
    println!("    sigma * sqrt(row blocks)                     {:.6}", arm.sigma * sqn);
    println!(
        "    under that 1/sqrt(n) law sigma meets {GATE_LO:.2}*(1 - rho*) at {} blocks per row and {GATE_HI:.2}*(1 - rho*) at {}",
        cross(GATE_LO),
        cross(GATE_HI)
    );
    println!("    row_scale on the weights vs the RMS block norm   max relative {scale_gap:.1e}");
    println!("    fit_gain_centroids vs lloyd_max on a             max absolute {cent_gap:.1e}");
}

/// `σ` against row length, on the served rule, one grouping per length.
fn sweep(xx: &[f64], tdot: &[f64], sd_cos: f64) {
    println!("Row length sweep, served rule, one grouping per length, outside the prereg's four numbers");
    println!("  blocks/row    rows    rho*       sigma      sigma*sqrt(n)   sigma/(1-rho*)   sd(1-cos)/sqrt(n)");
    let mut scaled = Vec::with_capacity(SWEEP.len());
    for row_blocks in SWEEP {
        let g = Grouping::from_xx(xx, row_blocks);
        let rows = rows_of(&xx[..g.n], &tdot[..g.n], &g.c, &g.scales, row_blocks);
        let rho = rows.iter().map(|r| r.a).sum::<f64>() / rows.iter().map(|r| r.b).sum::<f64>();
        let sigma = sd(&rows.iter().map(Row::rho).collect::<Vec<_>>());
        let sqn = (row_blocks as f64).sqrt();
        scaled.push(sigma * sqn);
        println!(
            "  {:<10}  {:<6}  {:.6}   {:.6}   {:.6}        {:.4}           {:.6}",
            row_blocks,
            g.n_rows,
            rho,
            sigma,
            sigma * sqn,
            sigma / (1.0 - rho),
            sd_cos / sqn
        );
    }
    let (lo, hi) = scaled.iter().fold((f64::INFINITY, 0.0f64), |a, &v| (a.0.min(v), a.1.max(v)));
    println!(
        "  sigma * sqrt(n) spans {lo:.6} to {hi:.6}, a spread of {:.1} % over a factor {} in row length",
        100.0 * (hi - lo) / lo,
        SWEEP[SWEEP.len() - 1] / SWEEP[0]
    );
    println!();
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get().saturating_sub(4)).unwrap_or(1).max(1));
    let t_all = Instant::now();
    let rate = BLOCK_BITS / DIM as f64;

    println!("M0, the row axis of chantier 14, on simulated blocks");
    println!("prereg proofs/preregistration-m0-echelles-2026-09-07.md; the decision rule is its §5, and this bench applies none of it\n");
    println!("Configuration");
    println!("  seed                {SEED:#014x}, llvq_bench::gauss_block, i.i.d. N(0,1)");
    println!("  blocks              {N_BLOCKS} of {DIM} coordinates");
    println!("  row sizes           {} and {} blocks (d_in = 2560 and 9728)", ROW_SIZES[0], ROW_SIZES[1]);
    println!("  encoder             llvq_search::tetra::Encoder, ALPHA {:.4}, RATIO {:.4}", Encoder::ALPHA, Encoder::RATIO);
    println!("  gain code           {GAIN_BITS} bit, fit_gain_centroids over the whole matrix, {LLOYD_ITERS} Lloyd iterations");
    println!("  rate                {rate:.3} b/dim");
    println!("  threads             {threads}\n");

    let mut rng = SplitMix64::new(SEED);
    let mut flat: Vec<f64> = Vec::with_capacity(N_BLOCKS * DIM);
    for _ in 0..N_BLOCKS {
        flat.extend_from_slice(&gauss_block(&mut rng));
    }

    // The identity the whole bench rests on, before anything is measured.
    check_reconstruction(&flat, &[0.85, 1.10], 24.0f64.sqrt());

    // One encoding pass for every grouping: the direction and `t` do not see
    // the row scale, only the gain level does.
    let t0 = Instant::now();
    let enc = Encoder::new(&Tetra::new());
    let td: Vec<(f64, f64)> = par_map(N_BLOCKS, threads, Scratch::new, |sc, i| {
        let x: &[f64; DIM] = flat[i * DIM..(i + 1) * DIM].try_into().expect("block of 24");
        (x.iter().map(|v| v * v).sum::<f64>(), enc.encode(x, sc).t)
    });
    let xx: Vec<f64> = td.iter().map(|e| e.0).collect();
    let tdot: Vec<f64> = td.iter().map(|e| e.1).collect();
    let el = t0.elapsed().as_secs_f64();
    let cos: Vec<f64> = xx.iter().zip(&tdot).map(|(&x2, &t)| t / x2.sqrt()).collect();
    let (mean_cos, sd_cos) = (mean(&cos), sd(&cos));
    println!("Encoding, {N_BLOCKS} blocks in {:.1} s ({:.0} us per block per thread)", el, 1e6 * el * threads as f64 / N_BLOCKS as f64);
    println!("  mean cos(theta) {mean_cos:.6}, so mean delta = 1 - cos(theta) = {:.6}", 1.0 - mean_cos);
    println!("  sd(1 - cos(theta)) = sd(cos(theta)) = {sd_cos:.6}, the input of the prereg §6 motif\n");

    for row_blocks in ROW_SIZES {
        let g = Grouping::production(&flat, &xx, row_blocks);
        let (n, n_rows) = (g.n, g.n_rows);
        let ctx = Ctx { row_blocks, mean_cos, sd_cos };

        // Served arm: the level is picked on a_p = ||x_p|| / s, and the
        // centroids are fitted once for the whole matrix, as in production.
        // Paper arm: the level is picked on tau_p, and the centroids are
        // refitted on that same population. `lloyd_max` is `fit_gain_centroids`
        // minus the norms it builds itself.
        let tau: Vec<f64> = (0..n).map(|i| tdot[i] / g.scales[i / row_blocks]).collect();
        let paper = lloyd_max(&tau, GAIN_BITS, LLOYD_ITERS);
        let c_paper: Vec<f64> = tau.iter().map(|&v| paper[nearest_centroid(&paper, v)]).collect();

        println!("Rows of {row_blocks} blocks: {n_rows} rows, {n} blocks used, {} dropped", N_BLOCKS - n);
        println!("  row scale {:.4} +/- {:.4}   gain centroids: served {:.5?}, paper {:.5?}", mean(&g.scales), sd(&g.scales), g.centroids, paper);
        let arm_served = summarize(&rows_of(&xx[..n], &tdot[..n], &g.c, &g.scales, row_blocks), n, rate, false);
        let arm_paper = summarize(&rows_of(&xx[..n], &tdot[..n], &c_paper, &g.scales, row_blocks), n, rate, true);
        report("served, level on a_p = ||x||/s  (the three published numbers)", &arm_served, &ctx);
        report("paper variant, level on tau_p, centroids refitted (fourth number)", &arm_paper, &ctx);
        witnesses(&g, &xx, &tdot, &arm_served);

        let by_rule = arm_paper.retention - arm_served.retention;
        let by_const = arm_served.retention_shrunk - arm_served.retention;
        println!("  the paper's rule is worth {by_rule:+.4} pp of retention here ({:.2} % against {:.2} %), for no bit and no format field", arm_paper.retention, arm_served.retention);
        if by_rule.abs() > 1e-9 {
            println!(
                "    of which one global constant rho* returns {by_const:+.4} pp, {:.1} % of it; picking the level per block adds the remaining {:+.4} pp",
                100.0 * by_const / by_rule,
                by_rule - by_const
            );
        }
        let drift = arm_served.retention - RETENTION_REF;
        if drift.abs() > 0.5 {
            println!("  RETENTION MISMATCH: {:.2} % against the {RETENTION_REF} % of the F1b journal, {drift:+.2} pp. The bench pipeline is wrong, do not read the numbers above", arm_served.retention);
        } else {
            println!("  retention cross-check: {drift:+.2} pp from the {RETENTION_REF} % of the F1b journal, within 0.5 pp");
        }
        println!();

        if row_blocks == ROW_SIZES[0] {
            brute_force(&flat, &xx, &tdot, &g, threads);
        }
    }

    sweep(&xx, &tdot, sd_cos);

    println!("Limits this bench carries by construction");
    println!("  Row scales here are taken on the same blocks the bench encodes. Production fixes them on");
    println!("  the original row before the GPTQ loop (llvq-quant/src/gptq.rs:245), then quantizes residuals");
    println!("  whose norms have drifted. That scale disagreement is absent here, so rho* prices the radial");
    println!("  bias 1/cos(theta) alone, and a_p is matched to the fitted centroids by construction.");
    println!("  The blocks are i.i.d. Gaussian, so rows are exchangeable: sigma(rho*) is a sampling statistic");
    println!("  of row length, and the sweep above reads its 1/sqrt(n) law off the bench itself.");
    println!();

    println!("total {:.1} min", t_all.elapsed().as_secs_f64() / 60.0);
}
