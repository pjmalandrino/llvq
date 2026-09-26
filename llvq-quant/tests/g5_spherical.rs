//! # Spherical GPTQ on Tetra: the feedback is angular, the file is unchanged
//!
//! `GptqConfig::spherical_feedback` forms the residual the Hessian correction
//! propagates against the block retracted to its exact pre-quantization norm
//! (Algorithm 3 line 5), while the block the layer stores stays on the gain
//! grid the decoder can rebuild. These tests pin four things: the flag is off
//! on the published path; on, it changes the layer through the feedback and
//! nothing else; the stored blocks still seal bit for bit; and the columns
//! after a block are compensated for the angular error alone.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{
    quantize_layer, quantize_layer_capturing, quantize_layer_parallel_capturing, GptqConfig,
    TailPolicy, Weights,
};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{
    fit_gain_centroids, reconstruct_shape_gain, row_scale, BlockCode, BlockQuantizer,
    LeechDirection, TetraShapeGain,
};

const D_OUT: usize = 6;
const NBLK: usize = 3;
const TAIL: usize = 8;
const D_IN: usize = NBLK * DIM + TAIL;

// ---------------------------------------------------------------------------
// Fixtures, deliberately the same shape as g5_design_c's
// ---------------------------------------------------------------------------

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

/// Rows spanning four orders of magnitude, like a real weight matrix.
fn weights(rng: &mut SplitMix64) -> Vec<f64> {
    (0..D_OUT * D_IN)
        .map(|k| {
            let row = k / D_IN;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

fn cfg(spherical_feedback: bool) -> GptqConfig {
    GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: false,
        spherical_feedback,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    }
}

struct Fixture {
    base: Vec<f64>,
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
        factor,
        centroids,
    }
}

fn run_tetra(fx: &Fixture, c: &GptqConfig) -> (Vec<f64>, Vec<Option<BlockCode>>) {
    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut q = TetraShapeGain::new(fx.centroids.clone());
    let mut codes = vec![None; D_OUT * NBLK];
    quantize_layer_capturing(&mut w, &fx.factor, None, &mut q, c, Some(&mut codes));
    (w.w, codes)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// The published path
// ---------------------------------------------------------------------------

/// The flag is off by default, and off it is the published loop to the bit.
#[test]
fn off_by_default_and_off_is_the_published_path() {
    assert!(!GptqConfig::default().spherical_feedback);
    let fx = fixture(0x5_0100);
    let (a, _) = run_tetra(&fx, &cfg(false));
    let published = GptqConfig {
        tail: TailPolicy::KeepExact,
        ..GptqConfig::default()
    };
    let (b, _) = run_tetra(&fx, &published);
    assert_eq!(a, b, "cfg(false) must be GptqConfig::default() on this loop");
}

// ---------------------------------------------------------------------------
// What the flag changes, and what it does not
// ---------------------------------------------------------------------------

/// On Tetra the two arms must differ — otherwise the flag is dead code — and
/// they must differ *only through the feedback*: the first block is chosen
/// before any feedback exists, so its stored values are identical in both
/// arms, and the divergence starts at block 1.
#[test]
fn the_flag_changes_the_layer_through_the_feedback_only() {
    let fx = fixture(0x5_0101);
    let (off, _) = run_tetra(&fx, &cfg(false));
    let (on, _) = run_tetra(&fx, &cfg(true));
    assert_ne!(off, on, "spherical feedback left the layer untouched");
    let mut first_block_identical = true;
    let mut later_differs = false;
    for i in 0..D_OUT {
        let r = i * D_IN;
        if off[r..r + DIM] != on[r..r + DIM] {
            first_block_identical = false;
        }
        if off[r + DIM..r + D_IN] != on[r + DIM..r + D_IN] {
            later_differs = true;
        }
    }
    assert!(
        first_block_identical,
        "block 0 sees no feedback, so the flag must not touch it"
    );
    assert!(later_differs, "the divergence has to show up after block 0");
}

/// The stored blocks are still exactly what the decoder rebuilds from the
/// captured codes: the flag never moves a block off the gain grid.
#[test]
fn stored_blocks_still_seal_bit_for_bit() {
    let fx = fixture(0x5_0102);
    let (w, codes) = run_tetra(&fx, &cfg(true));
    let mut out = [0.0f64; DIM];
    let mut checked = 0;
    for i in 0..D_OUT {
        let rs = row_scale(&fx.base[i * D_IN..(i + 1) * D_IN]);
        for p in 0..NBLK {
            let code = codes[i * NBLK + p].expect("Tetra always emits a code");
            reconstruct_shape_gain(&code, &fx.centroids, rs, &mut out);
            let stored = &w[i * D_IN + p * DIM..i * D_IN + (p + 1) * DIM];
            assert_eq!(
                stored,
                &out[..],
                "row {i} block {p}: the stored block is not what its code decodes to"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, D_OUT * NBLK);
}

/// The whole path rebuilt from independent pieces: the Tetra quantizer called
/// directly, the residual formed against `q · ‖w‖/‖q‖`, and the GPTQ factor's
/// own solve and `u` for the propagation. Bit equality on every stored value,
/// tail included.
#[test]
fn spherical_feedback_matches_an_independent_reference_on_tetra() {
    let fx = fixture(0x5_0103);
    let (got, _) = run_tetra(&fx, &cfg(true));

    let mut w = fx.base.clone();
    let mut q = TetraShapeGain::new(fx.centroids.clone());
    let mut qbuf = [0.0f64; DIM];
    let row_scales: Vec<f64> = (0..D_OUT)
        .map(|i| row_scale(&fx.base[i * D_IN..(i + 1) * D_IN]))
        .collect();
    for p in 0..NBLK {
        let s = p * DIM;
        let mut e = vec![0.0f64; D_OUT * DIM];
        for i in 0..D_OUT {
            let row = &mut w[i * D_IN..(i + 1) * D_IN];
            q.set_row_scale(row_scales[i]);
            let before = norm(&row[s..s + DIM]);
            q.quantize(&row[s..s + DIM], &mut qbuf);
            let k = before / norm(&qbuf);
            for j in 0..DIM {
                e[i * DIM + j] = row[s + j] - qbuf[j] * k;
                row[s + j] = qbuf[j];
            }
        }
        let r0 = s + DIM;
        fx.factor.solve_block(&mut e, D_OUT, s, DIM);
        for i in 0..D_OUT {
            let row = &mut w[i * D_IN..(i + 1) * D_IN];
            for k in 0..DIM {
                let x = e[i * DIM + k];
                if x == 0.0 {
                    continue;
                }
                for (j, cell) in row[r0..].iter_mut().enumerate() {
                    *cell -= x * fx.factor.u(s + k, r0 + j);
                }
            }
        }
    }
    assert_eq!(got, w, "the loop and the reference disagree");
}

/// Rows never interact, so the row split is exact with the flag on too.
#[test]
fn parallel_matches_serial_with_spherical_feedback() {
    let fx = fixture(0x5_0104);
    let (serial, serial_codes) = run_tetra(&fx, &cfg(true));
    let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
    let mut codes = vec![None; D_OUT * NBLK];
    let centroids = fx.centroids.clone();
    quantize_layer_parallel_capturing(
        &mut w,
        &fx.factor,
        None,
        &move || Box::new(TetraShapeGain::new(centroids.clone())),
        &cfg(true),
        4,
        Some(&mut codes),
    );
    assert_eq!(serial, w.w);
    assert_eq!(serial_codes, codes);
}

/// A norm-preserving quantizer already feeds back angular error only, so the
/// flag must be a no-op on it up to the rounding of `‖w‖/‖q‖ ≈ 1`.
#[test]
fn norm_preserving_direction_code_makes_the_flag_inert() {
    let fx = fixture(0x5_0105);
    let mut arms = Vec::new();
    for on in [false, true] {
        let mut w = Weights::new(D_OUT, D_IN, fx.base.clone());
        quantize_layer(&mut w, &fx.factor, None, &mut LeechDirection::new(), &cfg(on));
        arms.push(w.w);
    }
    let scale = norm(&fx.base);
    let d = arms[0]
        .iter()
        .zip(arms[1].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        d <= 1e-12 * scale,
        "on a norm-preserving code the flag moved the layer by {d}"
    );
}

// ---------------------------------------------------------------------------
// The load-bearing property: the columns are compensated for the angular
// error alone
// ---------------------------------------------------------------------------

/// Gauss–Jordan solve of `A X = B`, `B` and `X` row-major `n × m`
/// (element `(j, i)` at `[j * m + i]`), unrelated to the crate's Cholesky.
fn gauss_solve(a: &[f64], n: usize, b: &[f64], m: usize) -> Vec<f64> {
    let mut aug = vec![0.0f64; n * (n + m)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + m) + j] = a[i * n + j];
        }
        for c in 0..m {
            aug[i * (n + m) + n + c] = b[i * m + c];
        }
    }
    let w = n + m;
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&x, &y| aug[x * w + col].abs().total_cmp(&aug[y * w + col].abs()))
            .unwrap();
        if piv != col {
            for j in 0..w {
                aug.swap(col * w + j, piv * w + j);
            }
        }
        let p = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] /= p;
        }
        for i in 0..n {
            if i != col {
                let f = aug[i * w + col];
                if f != 0.0 {
                    for j in 0..w {
                        aug[i * w + j] -= f * aug[col * w + j];
                    }
                }
            }
        }
    }
    let mut x = vec![0.0f64; n * m];
    for i in 0..n {
        for c in 0..m {
            x[i * m + c] = aug[i * w + n + c];
        }
    }
    x
}

/// A coarse grid on the first block that stores its output *as is* (no
/// retraction: `retraction_target` is `None`, like a gain code), exact
/// afterwards. With the flag, the later columns must be compensated for
/// `E_ang = w − q·‖w‖/‖q‖` and not for the stored error `w − q`. The two
/// differ by the radial part, which this grid makes large on purpose.
#[test]
fn later_columns_are_compensated_for_the_angular_error_only() {
    let mut rng = SplitMix64::new(0x5_0106);
    let (d_out, d_in, b) = (7usize, 16usize, 4usize);
    let h = random_hessian(&mut rng, d_in, 64);
    let f = GptqFactor::new(&h, d_in, 0.0).expect("SPD");
    let w0: Vec<f64> = (0..d_out * d_in).map(|_| rng.next_gaussian()).collect();

    const STEP: f64 = 0.75;
    struct FirstBlockOnly {
        b: usize,
        seen: usize,
        rows: usize,
    }
    impl BlockQuantizer for FirstBlockOnly {
        fn block_len(&self) -> usize {
            self.b
        }
        fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
            if self.seen < self.rows {
                for (o, &a) in out.iter_mut().zip(v.iter()) {
                    *o = (a / STEP).round() * STEP;
                }
            } else {
                out.copy_from_slice(v);
            }
            self.seen += 1;
        }
        fn retraction_target(&self, _norm_before: f64) -> Option<f64> {
            None
        }
    }

    let c = GptqConfig {
        block: b,
        retract: true,
        group_scales: false,
        design_c: false,
        spherical_feedback: true,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    let mut weights = Weights::new(d_out, d_in, w0.clone());
    let mut q = FirstBlockOnly {
        b,
        seen: 0,
        rows: d_out,
    };
    quantize_layer(&mut weights, &f, None, &mut q, &c);

    // The stored block is the grid output, unretracted — the file's block.
    let mut e_ang = vec![0.0f64; d_out * b];
    let mut e_stored = vec![0.0f64; d_out * b];
    for i in 0..d_out {
        let orig = &w0[i * d_in..i * d_in + b];
        let kept = &weights.w[i * d_in..i * d_in + b];
        let raw: Vec<f64> = orig.iter().map(|a| (a / STEP).round() * STEP).collect();
        assert_eq!(kept, &raw[..], "row {i}: the stored block left the grid");
        let k = norm(orig) / norm(&raw);
        assert!(
            (k - 1.0).abs() > 1e-3,
            "row {i}: no radial error on this draw, the test would hold vacuously"
        );
        for j in 0..b {
            e_ang[i * b + j] = orig[j] - raw[j] * k;
            e_stored[i * b + j] = orig[j] - raw[j];
        }
    }

    // Reference: E H_QR H_RR⁻¹ for a given E, solved densely.
    let r = d_in - b;
    let mut hrr = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            hrr[i * r + j] = h[(b + i) * d_in + (b + j)];
        }
    }
    let compensation = |e: &[f64]| -> Vec<f64> {
        let mut yt = vec![0.0f64; r * d_out];
        for i in 0..d_out {
            for j in 0..r {
                yt[j * d_out + i] = (0..b).map(|k| e[i * b + k] * h[k * d_in + (b + j)]).sum();
            }
        }
        gauss_solve(&hrr, r, &yt, d_out)
    };
    let want_ang = compensation(&e_ang);
    let want_stored = compensation(&e_stored);

    let mut max_stored_gap = 0.0f64;
    for i in 0..d_out {
        for j in 0..r {
            let got = weights.w[i * d_in + (b + j)] - w0[i * d_in + (b + j)];
            let want = want_ang[j * d_out + i];
            assert!(
                (got - want).abs() <= 1e-7,
                "row {i}, col {}: compensated for {got}, the angular error asks {want}",
                b + j
            );
            max_stored_gap = max_stored_gap.max((got - want_stored[j * d_out + i]).abs());
        }
    }
    // And it is not the published compensation: the radial part is visibly
    // left out, or the flag was a no-op on this draw.
    assert!(
        max_stored_gap > 1e-3,
        "the columns were compensated for the stored error too ({max_stored_gap})"
    );
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "without `retract`")]
fn spherical_feedback_without_retraction_is_refused() {
    let fx = fixture(0x5_0107);
    let c = GptqConfig {
        retract: false,
        ..cfg(true)
    };
    run_tetra(&fx, &c);
}

#[test]
#[should_panic(expected = "one variable")]
fn spherical_feedback_with_design_c_is_refused() {
    let fx = fixture(0x5_0108);
    let c = GptqConfig {
        design_c: true,
        ..cfg(true)
    };
    run_tetra(&fx, &c);
}
