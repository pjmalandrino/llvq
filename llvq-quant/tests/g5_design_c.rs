//! # Design C — Algorithm 3 to the letter, made sealable
//!
//! `docs/retraction-et-gain.md` names three coherent readings of Spherical
//! GPTQ composed with a gain code. Design C is the paper's own: the
//! retraction keeps every block on its **exact** pre-quantization norm during
//! the loop (Eq. 17 as written), and the magnitudes are resolved once, at the
//! end of the layer, by the closed-form per-row scale solve of Algorithm 3
//! lines 11–18. The paper stops there — at free scales no block code can
//! express. Our design C adds one step the paper does not have: each block is
//! re-projected onto the gain grid and rebuilt exactly as a decoder would, so
//! the result still costs `47+k` bits per block and seals bit for bit.
//!
//! The load-bearing test is `design_c_matches_the_composed_reference`: it
//! rebuilds the whole path out of independently-checkable pieces — the
//! free-magnitude loop that already ships, an `m × m` solve done with a
//! Gauss–Jordan written in this file, and the public decoder — and demands
//! bit equality on every full block.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{
    proxy_loss, quantize_layer_capturing, quantize_layer_parallel_capturing, GptqConfig,
    TailPolicy, Weights,
};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, row_scale, BlockCode, LeechShapeGain};

const D_OUT: usize = 6;
const NBLK: usize = 3;
const TAIL: usize = 8;
const D_IN: usize = NBLK * DIM + TAIL;

// ---------------------------------------------------------------------------
// Independent references and fixtures
// ---------------------------------------------------------------------------

/// Gauss–Jordan solve of `A x = b`, deliberately unrelated to `solve_spd`'s
/// Cholesky. The re-projection snaps magnitudes onto a *discrete* grid, so
/// the two solvers' last-ulp disagreements cannot leak into the full blocks —
/// which is what lets a bit-exact assertion coexist with an independent
/// solver.
fn gauss_solve(a: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&r, &s| a[r * n + col].abs().total_cmp(&a[s * n + col].abs()))
            .expect("non-empty");
        assert!(a[piv * n + col].abs() > 1e-12, "singular system");
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            x.swap(col, piv);
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
        }
        x[col] /= d;
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= f * a[col * n + k];
            }
            x[r] -= f * x[col];
        }
    }
    x
}

fn random_hessian(rng: &mut SplitMix64, n: usize, samples: usize) -> Vec<f64> {
    let a: Vec<f64> = (0..samples * n).map(|_| rng.next_gaussian()).collect();
    let mut h = vec![0.0f64; n * n];
    for s in 0..samples {
        let row = &a[s * n..(s + 1) * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] += row[i] * row[j];
            }
        }
    }
    let inv = 1.0 / samples as f64;
    for v in h.iter_mut() {
        *v *= inv;
    }
    for i in 0..n {
        h[i * n + i] += 1e-3;
    }
    h
}

/// Rows spanning four orders of magnitude, like a real weight matrix — which
/// is what makes the per-row scale and the relative ridge load-bearing.
fn weights(rng: &mut SplitMix64) -> Vec<f64> {
    (0..D_OUT * D_IN)
        .map(|k| {
            let row = k / D_IN;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

fn cfg_c() -> GptqConfig {
    GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: true,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    }
}

fn cfg_default() -> GptqConfig {
    GptqConfig {
        design_c: false,
        ..cfg_c()
    }
}

struct Fixture {
    base: Vec<f64>,
    h: Vec<f64>,
    factor: GptqFactor,
    centroids: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let mut rng = SplitMix64::new(seed);
    let base = weights(&mut rng);
    let h = random_hessian(&mut rng, D_IN, 4 * D_IN);
    let factor = GptqFactor::new(&h, D_IN, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, D_IN, DIM, 1, 40);
    Fixture {
        base,
        h,
        factor,
        centroids,
    }
}

fn run_design_c(fx: &Fixture) -> (Vec<f64>, Vec<Option<BlockCode>>) {
    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut q = LeechShapeGain::new(fx.centroids.clone());
    let mut codes = vec![None; D_OUT * NBLK];
    quantize_layer_capturing(
        &mut w,
        &fx.factor,
        Some(&fx.h),
        &mut q,
        &cfg_c(),
        Some(&mut codes),
    );
    (w.w, codes)
}

fn nearest(levels: &[f64], g: f64) -> usize {
    let mut best = (f64::INFINITY, 0usize);
    for (i, &c) in levels.iter().enumerate() {
        let d = (g - c).abs();
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

// ---------------------------------------------------------------------------
// The default path is untouched
// ---------------------------------------------------------------------------

/// With both end-of-layer flags off, the Hessian argument must not be read at
/// all: passing it and not passing it must give bit-identical weights and
/// codes. Any leakage of the design C machinery into the published path —
/// the path every shipped artifact came from — would show up here first.
#[test]
fn the_default_path_never_reads_the_hessian() {
    let fx = fixture(0xC_0001);
    let run = |h: Option<&[f64]>| {
        let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
        let mut q = LeechShapeGain::new(fx.centroids.clone());
        let mut codes = vec![None; D_OUT * NBLK];
        quantize_layer_capturing(&mut w, &fx.factor, h, &mut q, &cfg_default(), Some(&mut codes));
        (w.w, codes)
    };
    let (w_none, c_none) = run(None);
    let (w_some, c_some) = run(Some(&fx.h));
    for (a, b) in w_none.iter().zip(w_some.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "default path read the Hessian");
    }
    assert_eq!(c_none, c_some);
}

// ---------------------------------------------------------------------------
// Design C is exactly its three declared steps
// ---------------------------------------------------------------------------

/// Rebuild design C out of parts that are each verifiable on their own:
///
/// 1. the free-magnitude loop (`with_free_magnitude`, the design B path that
///    every pre-2026-07-31 run exercised and `g5_retraction.rs` pins),
/// 2. the per-row solve of Algorithm 3 lines 11–18, redone here with an
///    independent Gauss–Jordan on an `M`/`r` built from scratch,
/// 3. the snap of each block magnitude to the nearest gain level, rebuilt
///    through the public decoder `LeechShapeGain::reconstruct`.
///
/// Full blocks must match **bit for bit** — the snap is discrete, so the two
/// solvers' fp differences cannot reach them. The tail keeps the solver's
/// continuous scale, so it gets a tolerance instead.
///
/// What each mutant does to this test:
/// - retraction still pinning magnitudes in design C → step 1 diverges, the
///   propagated error differs, points and gains move;
/// - solve short-circuited to `s = 1` → step 2 diverges, gains move;
/// - re-projection skipped → step 3 diverges, blocks are off-grid;
/// - residual formed on the uncompensated `W` → the loop itself diverges.
#[test]
fn design_c_matches_the_composed_reference() {
    let fx = fixture(0xC_0002);
    let (got_w, got_codes) = run_design_c(&fx);

    // Step 1 — the free-magnitude loop, exactly as it ships.
    let mut w1 = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut q1 = LeechShapeGain::new(fx.centroids.clone()).with_free_magnitude();
    let mut codes1 = vec![None; D_OUT * NBLK];
    quantize_layer_capturing(
        &mut w1,
        &fx.factor,
        None,
        &mut q1,
        &cfg_default(),
        Some(&mut codes1),
    );

    // Steps 2 and 3, per row.
    let decoder = LeechShapeGain::new(fx.centroids.clone());
    let m = D_IN.div_ceil(DIM); // 3 full blocks + the 8-wide tail group
    let bounds: Vec<(usize, usize)> = (0..m)
        .map(|p| (p * DIM, ((p + 1) * DIM).min(D_IN)))
        .collect();
    for i in 0..D_OUT {
        let orig = &fx.base[i * D_IN..(i + 1) * D_IN];
        let wq = &mut w1.w[i * D_IN..(i + 1) * D_IN];
        let rscale = row_scale(orig);

        // M[p,q] = g_p H g_qᵀ, r[p] = g_p H worigᵀ, ridge relative to the
        // mean diagonal — Algorithm 3 lines 12–14.
        let mut gh = vec![0.0f64; m * D_IN];
        for (p, &(ps, pe)) in bounds.iter().enumerate() {
            for (c, &g) in wq.iter().enumerate().take(pe).skip(ps) {
                for j in 0..D_IN {
                    gh[p * D_IN + j] += g * fx.h[c * D_IN + j];
                }
            }
        }
        let mut mat = vec![0.0f64; m * m];
        let mut rhs = vec![0.0f64; m];
        for p in 0..m {
            let ghp = &gh[p * D_IN..(p + 1) * D_IN];
            rhs[p] = ghp.iter().zip(orig.iter()).map(|(a, b)| a * b).sum();
            for (q, &(qs, qe)) in bounds.iter().enumerate() {
                mat[p * m + q] = (qs..qe).map(|c| ghp[c] * wq[c]).sum();
            }
        }
        let mean_diag: f64 = (0..m).map(|p| mat[p * m + p]).sum::<f64>() / m as f64;
        let ridge = 1e-2 * mean_diag;
        for p in 0..m {
            mat[p * m + p] += ridge;
        }
        let s = gauss_solve(&mat, m, &rhs);
        for (p, &(ps, pe)) in bounds.iter().enumerate() {
            for cell in wq[ps..pe].iter_mut() {
                *cell *= s[p];
            }
        }

        // Step 3 — snap each full block, and rebuild it as the decoder does.
        for p in 0..NBLK {
            let code = codes1[i * NBLK + p].expect("shape–gain always emits a code");
            let blk = &mut wq[p * DIM..(p + 1) * DIM];
            let norm = blk.iter().map(|a| a * a).sum::<f64>().sqrt();
            let level = nearest(&fx.centroids, norm / rscale);
            let expected_code = BlockCode {
                point: code.point,
                gain: level as u32,
            };
            let mut expected = [0.0f64; DIM];
            decoder.reconstruct(&expected_code, rscale, &mut expected);

            let got_blk = &got_w[i * D_IN + p * DIM..i * D_IN + (p + 1) * DIM];
            for (k, (g, e)) in got_blk.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "row {i} block {p} weight {k}: design C gives {g:e}, the \
                     composed reference {e:e}"
                );
            }
            assert_eq!(
                got_codes[i * NBLK + p],
                Some(expected_code),
                "row {i} block {p}: code disagrees with the composed reference"
            );
        }

        // The tail keeps the solve's continuous scale (it is stored verbatim,
        // so it needs no grid); only the solver differs here.
        let got_tail = &got_w[i * D_IN + NBLK * DIM..(i + 1) * D_IN];
        let ref_tail = &wq[NBLK * DIM..];
        for (g, e) in got_tail.iter().zip(ref_tail.iter()) {
            let denom = e.abs().max(1e-30);
            assert!(
                ((g - e) / denom).abs() < 1e-8,
                "tail disagrees beyond solver tolerance: {g:e} vs {e:e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sealability
// ---------------------------------------------------------------------------

/// The property that separates design C from the `group_scales` fragment:
/// after the layer loop, the solve **and** the re-projection, every full
/// block must still be exactly what its code decodes to — same demand as
/// `verify_artifact` makes of a shipped file. A skipped or approximate
/// re-projection fails here on the first block.
#[test]
fn design_c_result_is_sealable_bit_for_bit() {
    let fx = fixture(0xC_0003);
    let (w, codes) = run_design_c(&fx);
    let decoder = LeechShapeGain::new(fx.centroids.clone());
    let mut out = [0.0f64; DIM];
    for i in 0..D_OUT {
        let rscale = row_scale(&fx.base[i * D_IN..(i + 1) * D_IN]);
        for p in 0..NBLK {
            let code = codes[i * NBLK + p].expect("every full block has a code");
            decoder.reconstruct(&code, rscale, &mut out);
            let blk = &w[i * D_IN + p * DIM..i * D_IN + (p + 1) * DIM];
            for (k, (g, e)) in blk.iter().zip(out.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "row {i} block {p} weight {k}: stored {g:e}, decodes to \
                     {e:e} — the result is not sealable"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scale invariance of the closed-form solve
// ---------------------------------------------------------------------------

/// Multiplying the weights by a constant must not change a single decision —
/// lattice points or gain ranks — for any λ. The gain code is relative to the
/// per-row scale, the angular search is scale-free, `M` and `r` both scale
/// with the square of the magnitude, and the ridge is **relative** to
/// `mean(diag M)`; an absolute ridge (the Phase 5 group-scales bug, pinned by
/// `group_scale_ridge_is_scale_invariant`) breaks this immediately. λ = 0 is
/// deliberately not in the sweep — a ridge tested only at its neutral value
/// is not tested (CLAUDE.md §5).
#[test]
fn design_c_decisions_are_scale_invariant() {
    let fx = fixture(0xC_0004);
    for lambda in [1e-3, 1e-2, 1e-1] {
        let run = |scale: f64| {
            let scaled: Vec<f64> = fx.base.iter().map(|v| v * scale).collect();
            let centroids = fit_gain_centroids(&scaled, D_OUT, D_IN, DIM, 1, 40);
            let mut w = Weights::new(D_OUT, D_IN, scaled);
            let mut q = LeechShapeGain::new(centroids);
            let mut codes = vec![None; D_OUT * NBLK];
            let cfg = GptqConfig {
                lambda,
                ..cfg_c()
            };
            quantize_layer_capturing(&mut w, &fx.factor, Some(&fx.h), &mut q, &cfg, Some(&mut codes));
            codes
        };
        let one = run(1.0);
        let eight = run(8.0);
        assert_eq!(
            one, eight,
            "λ = {lambda:e}: scaling the weights by 8 changed a decision"
        );
    }
}

// ---------------------------------------------------------------------------
// The point of the exercise: the proxy must drop, strictly
// ---------------------------------------------------------------------------

/// On this fixture, design C must **strictly** beat the published path
/// (retract-to-level, design A) on the proxy `Tr(ΔW H ΔWᵀ)`. Strict, because
/// a `≤` would accept a no-op design C — the same non-strict-monotonicity
/// hole that let the group-scales mutant survive in Phase 5.
///
/// ⚠️ One synthetic fixture, the local proxy, nothing more: `group_scales`
/// also won locally on 3 blocks and destroyed 28 (CLAUDE.md §5). This test
/// asserts the mechanism does what it claims where it claims it; only a full
/// model run says whether that survives composition across 28 layers.
#[test]
fn design_c_strictly_lowers_the_proxy_below_the_published_path() {
    let fx = fixture(0xC_0005);
    let run = |cfg: &GptqConfig| {
        let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
        let mut q = LeechShapeGain::new(fx.centroids.clone());
        quantize_layer_capturing(&mut w, &fx.factor, Some(&fx.h), &mut q, cfg, None);
        let delta: Vec<f64> = w.w.iter().zip(fx.base.iter()).map(|(a, b)| a - b).collect();
        proxy_loss(&delta, D_OUT, D_IN, &fx.h)
    };
    let published = run(&cfg_default());
    let design_c = run(&cfg_c());
    assert!(
        design_c < published,
        "design C does not strictly improve the proxy on this fixture: \
         {design_c:.6e} vs {published:.6e}"
    );
}

// ---------------------------------------------------------------------------
// Row split stays exact, guards stay loud
// ---------------------------------------------------------------------------

/// Design C's solve and re-projection are per row, like everything else in
/// the loop — so the parallel row split must stay **exact**, codes included,
/// as `parallel_matches_serial_exactly` demands of the default path.
#[test]
fn parallel_design_c_matches_serial_exactly() {
    let fx = fixture(0xC_0006);
    let (serial_w, serial_codes) = run_design_c(&fx);

    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut codes = vec![None; D_OUT * NBLK];
    let centroids = fx.centroids.clone();
    let make = move || -> Box<dyn llvq_quant::quantizer::BlockQuantizer> {
        Box::new(LeechShapeGain::new(centroids.clone()))
    };
    quantize_layer_parallel_capturing(
        &mut w,
        &fx.factor,
        Some(&fx.h),
        &make,
        &cfg_c(),
        3,
        Some(&mut codes),
    );
    for (a, b) in w.w.iter().zip(serial_w.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel design C diverged");
    }
    assert_eq!(codes, serial_codes);
}

/// Design C without the Hessian cannot run its solve; that must be a loud
/// failure, not a silently skipped step.
#[test]
#[should_panic(expected = "closed-form scale solve needs")]
fn design_c_without_hessian_is_refused() {
    let fx = fixture(0xC_0007);
    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut q = LeechShapeGain::new(fx.centroids.clone());
    quantize_layer_capturing(&mut w, &fx.factor, None, &mut q, &cfg_c(), None);
}

/// `group_scales` is the unsealable fragment of the same algorithm; running
/// both would solve twice and mean nothing.
#[test]
#[should_panic(expected = "group_scales is the unsealable fragment")]
fn design_c_and_group_scales_are_mutually_exclusive() {
    let fx = fixture(0xC_0008);
    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut q = LeechShapeGain::new(fx.centroids.clone());
    let cfg = GptqConfig {
        group_scales: true,
        ..cfg_c()
    };
    quantize_layer_capturing(&mut w, &fx.factor, Some(&fx.h), &mut q, &cfg, None);
}

/// The closed-form solve may pick a **negative** scale for a block; the
/// content then points along `−point`. Reprojection must follow that
/// orientation — Λ₂₄ is centrally symmetric, so the flipped point is exactly
/// as representable — rather than silently snapping the block back onto the
/// old side of the sphere. A round-trip test cannot see this defect: both
/// orientations seal bit-exactly. This probe hands `reproject` a negated
/// reconstruction and demands the flipped code.
#[test]
fn reproject_follows_a_negative_scale() {
    use llvq_quant::quantizer::BlockQuantizer;
    let fx = fixture(0xC_0009);
    let mut q = LeechShapeGain::new(fx.centroids.clone());
    let mut rng = SplitMix64::new(0xC_0009);
    let v: Vec<f64> = (0..DIM).map(|_| rng.next_f64() - 0.5).collect();
    q.set_row_scale(row_scale(&v));

    let mut rec = vec![0.0; DIM];
    q.quantize(&v, &mut rec);
    let code = q.last_code().expect("shape–gain always emits a code");
    let norm = rec.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!(norm > 0.0, "degenerate probe block");

    // A negative solve scale: the block is the reconstruction, negated.
    let mut blk: Vec<f64> = rec.iter().map(|a| -a).collect();
    let new = q
        .reproject(&code, norm, &mut blk)
        .expect("shape–gain reprojects");

    for (a, b) in new.point.iter().zip(code.point.iter()) {
        assert_eq!(*a, -*b, "reproject kept the old orientation");
    }
    let dot: f64 = blk
        .iter()
        .zip(rec.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        dot < 0.0,
        "the reprojected block must lie on the solve's side of the sphere"
    );
    // And the flipped point must still be a sealable lattice point: the
    // reconstruction path only accepts indexable points, so rebuilding
    // through the same routine is the proof.
    let mut back = vec![0.0; DIM];
    q.reconstruct(&new, row_scale(&v), &mut back);
    let align: f64 = back.iter().zip(blk.iter()).map(|(a, b)| a * b).sum();
    assert!(align > 0.0, "reconstruction disagrees with the emitted code");
}
