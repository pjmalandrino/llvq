//! # Gate G5, part 1 — the GPTQ machinery, pinned without a model
//!
//! Everything a layer-wise quantizer does before it ever sees a checkpoint is
//! algebra with exact identities attached to it. This suite asserts those
//! identities directly, so a wrong sign or a transposed block fails here
//! rather than three hours into a Qwen3-4B run.
//!
//! The load-bearing test is `correction_is_the_analytic_minimizer`: it
//! rebuilds the paper's Appendix F.2 correction `−ΔW_Q H_QR H_RR⁻¹` with an
//! independent dense solver (Gauss–Jordan, written in this file) and demands
//! that the Cholesky-factor path produce exactly it.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{bits_per_weight, proxy_loss, quantize_layer, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::{cholesky_lower, invert_lower, GptqFactor};
use llvq_quant::quantizer::{BlockQuantizer, Identity, LeechDirection, ScalarGrid};

const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Independent references (no crate code)
// ---------------------------------------------------------------------------

/// Gauss–Jordan solve of `A X = B`, both dense row-major. Deliberately naive
/// and unrelated to anything in `linalg`.
fn gauss_solve(a: &[f64], n: usize, b: &[f64], m: usize) -> Vec<f64> {
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
            for k in 0..m {
                x.swap(col * m + k, piv * m + k);
            }
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
        }
        for k in 0..m {
            x[col * m + k] /= d;
        }
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
            for k in 0..m {
                x[r * m + k] -= f * x[col * m + k];
            }
        }
    }
    x
}

/// A random SPD Hessian `AᵀA/N + εI`, the shape a calibration pass produces.
fn random_hessian(rng: &mut SplitMix64, n: usize, samples: usize, eps: f64) -> Vec<f64> {
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
        h[i * n + i] += eps;
    }
    h
}

fn random_weights(rng: &mut SplitMix64, d_out: usize, d_in: usize) -> Vec<f64> {
    (0..d_out * d_in).map(|_| rng.next_gaussian()).collect()
}

// ---------------------------------------------------------------------------
// The factorization
// ---------------------------------------------------------------------------

#[test]
fn cholesky_and_triangular_inverse_are_exact() {
    let mut rng = SplitMix64::new(0x5_0001);
    let n = 20;
    let h = random_hessian(&mut rng, n, 60, 1e-3);

    let mut l = h.clone();
    cholesky_lower(&mut l, n).expect("SPD");
    for i in 0..n {
        for j in 0..n {
            let recon: f64 = (0..n).map(|k| l[i * n + k] * l[j * n + k]).sum();
            assert!((recon - h[i * n + j]).abs() <= TOL, "L Lᵀ ≠ H at ({i},{j})");
            if j > i {
                assert_eq!(l[i * n + j], 0.0, "upper triangle must be zeroed");
            }
        }
    }

    let mut linv = l.clone();
    invert_lower(&mut linv, n);
    for i in 0..n {
        for j in 0..n {
            let e: f64 = (0..n).map(|k| l[i * n + k] * linv[k * n + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((e - want).abs() <= TOL, "L·L⁻¹ ≠ I at ({i},{j})");
        }
    }
}

#[test]
fn gptq_factor_satisfies_its_defining_identity() {
    let mut rng = SplitMix64::new(0x5_0002);
    let n = 18;
    let h = random_hessian(&mut rng, n, 50, 1e-3);
    let f = GptqFactor::new(&h, n, 0.0).expect("SPD");

    // U must be upper triangular with Uᵀ U = H⁻¹, i.e. (Uᵀ U) H = I.
    for i in 0..n {
        for j in 0..i {
            assert_eq!(f.u(i, j), 0.0, "U must be upper triangular at ({i},{j})");
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                let hinv_ik: f64 = (0..n).map(|t| f.u(t, i) * f.u(t, k)).sum();
                acc += hinv_ik * h[k * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((acc - want).abs() <= 1e-7, "(Uᵀ U) H ≠ I at ({i},{j})");
        }
    }
}

#[test]
fn damping_is_applied_relative_to_the_mean_diagonal() {
    let mut rng = SplitMix64::new(0x5_0003);
    let n = 12;
    let h = random_hessian(&mut rng, n, 40, 1e-3);
    let mean_diag: f64 = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let lambda = 0.05;

    let damped = GptqFactor::new(&h, n, lambda).expect("SPD");
    let mut explicit = h.clone();
    for i in 0..n {
        explicit[i * n + i] += lambda * mean_diag;
    }
    let reference = GptqFactor::new(&explicit, n, 0.0).expect("SPD");

    for i in 0..n {
        for j in 0..n {
            assert!((damped.u(i, j) - reference.u(i, j)).abs() <= TOL);
        }
    }
}

// ---------------------------------------------------------------------------
// The correction itself — the load-bearing test
// ---------------------------------------------------------------------------

/// After the first block, the update written onto the remaining columns must
/// equal the analytic minimizer of the local proxy `Tr(ΔW H ΔWᵀ)`.
///
/// With `E = W̃_Q − Ŵ_Q` the introduced error is `ΔW_Q = −E`, and Appendix F.2
/// gives `ΔW*_R = −ΔW_Q H_QR H_RR⁻¹ = +E H_QR H_RR⁻¹`. The implementation
/// never forms an inverse; it applies `−(E U_QQ⁻¹) U_QR`. Those must agree to
/// round-off.
#[test]
fn correction_is_the_analytic_minimizer() {
    let mut rng = SplitMix64::new(0x5_0004);
    let (d_out, d_in, b) = (7, 16, 4);
    let h = random_hessian(&mut rng, d_in, 64, 1e-2);
    let f = GptqFactor::new(&h, d_in, 0.0).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);

    let mut weights = Weights::new(d_out, d_in, w0.clone());
    let cfg = GptqConfig {
        block: b,
        retract: false,
        group_scales: false,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    // A one-block run: stop the loop after the first block by quantizing only
    // it. Emulated by running the full loop with a quantizer that is exact
    // everywhere except the first block.
    // `quantize` is called once per (block, row), so "the first block" is the
    // first `d_out` calls — not the first call.
    struct FirstBlockOnly {
        b: usize,
        seen: usize,
        rows: usize,
        step: f64,
    }
    impl BlockQuantizer for FirstBlockOnly {
        fn block_len(&self) -> usize {
            self.b
        }
        fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
            if self.seen < self.rows {
                for (o, &a) in out.iter_mut().zip(v.iter()) {
                    *o = (a / self.step).round() * self.step;
                }
            } else {
                out.copy_from_slice(v);
            }
            self.seen += 1;
        }
    }
    let mut q = FirstBlockOnly {
        b,
        seen: 0,
        rows: d_out,
        step: 0.75,
    };
    quantize_layer(&mut weights, &f, None, &mut q, &cfg);

    // E from the first block, recomputed from the original weights.
    let mut e = vec![0.0f64; d_out * b];
    for i in 0..d_out {
        for k in 0..b {
            let orig = w0[i * d_in + k];
            e[i * b + k] = orig - (orig / 0.75).round() * 0.75;
        }
    }

    // Reference: E H_QR H_RR⁻¹, solved densely.
    let r = d_in - b;
    let mut hqr = vec![0.0f64; b * r];
    for i in 0..b {
        for j in 0..r {
            hqr[i * r + j] = h[i * d_in + (b + j)];
        }
    }
    let mut hrr = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            hrr[i * r + j] = h[(b + i) * d_in + (b + j)];
        }
    }
    // Y = E H_QR  (d_out × r), then solve X H_RR = Y ⇔ H_RR Xᵀ = Yᵀ.
    let mut yt = vec![0.0f64; r * d_out];
    for i in 0..d_out {
        for j in 0..r {
            let v: f64 = (0..b).map(|k| e[i * b + k] * hqr[k * r + j]).sum();
            yt[j * d_out + i] = v;
        }
    }
    let xt = gauss_solve(&hrr, r, &yt, d_out);

    // The remaining columns were only touched by this one correction (later
    // blocks were quantized exactly), so W̃_R − W_R must equal +E H_QR H_RR⁻¹.
    for i in 0..d_out {
        for j in 0..r {
            let got = weights.w[i * d_in + (b + j)] - w0[i * d_in + (b + j)];
            let want = xt[j * d_out + i];
            assert!(
                (got - want).abs() <= 1e-7,
                "row {i}, col {}: propagated {got} vs analytic {want}",
                b + j
            );
        }
    }
}

#[test]
fn identity_quantizer_leaves_the_layer_untouched() {
    let mut rng = SplitMix64::new(0x5_0005);
    let (d_out, d_in, b) = (6, 20, 5);
    let h = random_hessian(&mut rng, d_in, 70, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);
    let mut weights = Weights::new(d_out, d_in, w0.clone());
    let cfg = GptqConfig {
        block: b,
        retract: true,
        group_scales: false,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    quantize_layer(&mut weights, &f, None, &mut Identity { block: b }, &cfg);
    for (got, want) in weights.w.iter().zip(w0.iter()) {
        assert!(
            (got - want).abs() <= TOL,
            "a lossless quantizer must be a no-op: {got} vs {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// What the algorithm is for
// ---------------------------------------------------------------------------

/// Error feedback must actually buy something: on the objective it optimizes,
/// GPTQ has to beat plain round-to-nearest with the same codebook.
#[test]
fn gptq_beats_round_to_nearest_on_the_proxy() {
    let mut rng = SplitMix64::new(0x5_0006);
    let (d_out, d_in, b) = (12, 24, 4);
    let step = 0.6;
    let mut wins = 0;
    let trials = 12;
    for _ in 0..trials {
        let h = random_hessian(&mut rng, d_in, 96, 1e-2);
        let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
        let w0 = random_weights(&mut rng, d_out, d_in);

        let mut weights = Weights::new(d_out, d_in, w0.clone());
        let cfg = GptqConfig {
            block: b,
            retract: false,
            group_scales: false,
            lambda: 0.0,
            tail: TailPolicy::Reject,
        };
        quantize_layer(
            &mut weights,
            &f,
            None,
            &mut ScalarGrid { block: b, step },
            &cfg,
        );
        let d_gptq: Vec<f64> = weights
            .w
            .iter()
            .zip(w0.iter())
            .map(|(a, b)| a - b)
            .collect();

        let d_rtn: Vec<f64> = w0
            .iter()
            .map(|a| (a / step).round() * step - a)
            .collect();

        let l_gptq = proxy_loss(&d_gptq, d_out, d_in, &h);
        let l_rtn = proxy_loss(&d_rtn, d_out, d_in, &h);
        if l_gptq < l_rtn {
            wins += 1;
        }
    }
    assert_eq!(
        wins, trials,
        "GPTQ lost to round-to-nearest on {} of {trials} random layers",
        trials - wins
    );
}

/// The spherical retraction must leave every quantized row block on the
/// sphere of its pre-quantization norm — that is the entire point of
/// Spherical GPTQ, and the property the Hadamard-free result rests on.
#[test]
fn retraction_preserves_every_row_block_norm() {
    let mut rng = SplitMix64::new(0x5_0007);
    let (d_out, d_in, b) = (5, 15, 3);
    let h = random_hessian(&mut rng, d_in, 60, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);

    // A quantizer that mangles magnitudes badly on purpose.
    struct Shrink {
        b: usize,
    }
    impl BlockQuantizer for Shrink {
        fn block_len(&self) -> usize {
            self.b
        }
        fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
            for (o, &a) in out.iter_mut().zip(v.iter()) {
                *o = 0.017 * a + 0.3;
            }
        }
    }

    // Norms are taken against the *compensated* weights the loop actually
    // quantizes, so replay the loop block by block to know them.
    let mut weights = Weights::new(d_out, d_in, w0.clone());
    let cfg = GptqConfig {
        block: b,
        retract: true,
        group_scales: false,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    let expected = {
        // Re-run the same loop with a norm-recording pass-through wrapper.
        struct Record<'a> {
            b: usize,
            inner: Shrink,
            norms: &'a mut Vec<f64>,
        }
        impl BlockQuantizer for Record<'_> {
            fn block_len(&self) -> usize {
                self.b
            }
            fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
                self.norms.push(v.iter().map(|a| a * a).sum::<f64>().sqrt());
                self.inner.quantize(v, out);
            }
        }
        let mut norms = Vec::new();
        let mut probe = Weights::new(d_out, d_in, w0.clone());
        let mut rec = Record {
            b,
            inner: Shrink { b },
            norms: &mut norms,
        };
        quantize_layer(&mut probe, &f, None, &mut rec, &cfg);
        norms
    };

    quantize_layer(&mut weights, &f, None, &mut Shrink { b }, &cfg);

    let nblocks = d_in / b;
    let mut k = 0;
    for s in 0..nblocks {
        for i in 0..d_out {
            let block = &weights.w[i * d_in + s * b..i * d_in + (s + 1) * b];
            let got = block.iter().map(|a| a * a).sum::<f64>().sqrt();
            assert!(
                (got - expected[k]).abs() <= 1e-9,
                "block {s}, row {i}: norm {got} vs {} before quantization",
                expected[k]
            );
            k += 1;
        }
    }
}

/// With the directions already exact, the closed-form group scales must come
/// out at exactly one — otherwise the solve is not computing what its
/// derivation says.
#[test]
fn group_scales_are_unity_when_the_directions_are_exact() {
    let mut rng = SplitMix64::new(0x5_0008);
    let (d_out, d_in, b) = (4, 12, 3);
    let h = random_hessian(&mut rng, d_in, 80, 1e-1);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);
    let mut weights = Weights::new(d_out, d_in, w0.clone());
    let cfg = GptqConfig {
        block: b,
        retract: true,
        group_scales: true,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    quantize_layer(
        &mut weights,
        &f,
        Some(&h),
        &mut Identity { block: b },
        &cfg,
    );
    for (got, want) in weights.w.iter().zip(w0.iter()) {
        assert!(
            (got - want).abs() <= 1e-6,
            "scales should stay at 1: {got} vs {want}"
        );
    }
}

/// And with lossy directions, the refinement must **strictly** lower the
/// objective it solves.
///
/// Strictness is the point, not pedantry: the right-hand side
/// `r[p] = g_p H_{Q_p,:} w_{i,:}ᵀ` is built against the *original* weights.
/// Build it against the quantized ones instead and the system becomes
/// `M s = M·1`, whose solution is `s = 1` — a refinement that does nothing,
/// which a "must not increase" assertion would happily accept. Mutation
/// testing found exactly that hole.
#[test]
fn group_scales_strictly_lower_the_proxy() {
    let mut rng = SplitMix64::new(0x5_0009);
    let (d_out, d_in, b) = (8, 18, 3);
    for _ in 0..6 {
        let h = random_hessian(&mut rng, d_in, 90, 1e-2);
        let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
        let w0 = random_weights(&mut rng, d_out, d_in);

        let loss = |gs: bool| {
            let mut w = Weights::new(d_out, d_in, w0.clone());
            let cfg = GptqConfig {
                block: b,
                retract: true,
                group_scales: gs,
                lambda: 0.0,
                tail: TailPolicy::Reject,
            };
            quantize_layer(
                &mut w,
                &f,
                Some(&h),
                &mut ScalarGrid { block: b, step: 0.9 },
                &cfg,
            );
            let d: Vec<f64> = w.w.iter().zip(w0.iter()).map(|(a, b)| a - b).collect();
            proxy_loss(&d, d_out, d_in, &h)
        };
        let (without, with) = (loss(false), loss(true));
        assert!(
            with < without * (1.0 - 1e-6),
            "group scales must strictly improve the proxy: {with} vs {without}"
        );
    }
}

// ---------------------------------------------------------------------------
// The Leech direction quantizer, as the GPTQ loop sees it
// ---------------------------------------------------------------------------

#[test]
fn leech_direction_is_norm_preserving_and_lands_on_the_lattice() {
    let leech = llvq_core::Leech::new();
    let mut q = LeechDirection::new();
    let mut rng = SplitMix64::new(0x5_000A);
    let mut out = vec![0.0f64; DIM];
    for t in 0..64 {
        let scale = 0.01 * (1 + t % 7) as f64;
        let v: Vec<f64> = (0..DIM).map(|_| scale * rng.next_gaussian()).collect();
        q.quantize(&v, &mut out);

        let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let no = out.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            (nv - no).abs() <= 1e-9 * nv.max(1.0),
            "0 gain bits means the norm is carried exactly: {no} vs {nv}"
        );

        // The reconstruction is `point · ‖v‖/√(16m)`, so the lattice point
        // comes back by trying each shell radius and demanding integrality
        // *and* membership — a scaled non-lattice vector passes neither.
        let mut shell = None;
        for m in 2..=13u32 {
            let c = ((16 * m) as f64).sqrt() / nv;
            let pt: llvq_core::Point =
                core::array::from_fn(|i| (out[i] * c).round() as i32);
            let integral = (0..DIM).all(|i| (out[i] * c - pt[i] as f64).abs() < 1e-6);
            if integral
                && llvq_core::Leech::shell_index(&pt) == Some(m as u64)
                && leech.contains(&pt)
            {
                shell = Some(m);
                break;
            }
        }
        assert!(
            shell.is_some(),
            "trial {t}: the direction is not a Λ24 point on any shell m ≤ 13"
        );
    }
}

/// A block of zeros has no direction; it must survive rather than produce
/// NaNs — real weight matrices do contain dead blocks.
#[test]
fn zero_block_is_handled() {
    let mut q = LeechDirection::new();
    let mut out = vec![9.9f64; DIM];
    q.quantize(&[0.0; DIM], &mut out);
    assert!(out.iter().all(|v| *v == 0.0), "zero block must stay zero");
}

/// End to end in the configuration the paper recommends: 24-wide blocks, the
/// Leech direction code with 0 gain bits, spherical retraction on. Against
/// the same codebook applied without error feedback, Spherical GPTQ has to
/// win on the proxy — if it does not, nothing downstream will work either.
#[test]
fn spherical_gptq_beats_feedback_free_leech_on_the_proxy() {
    let mut rng = SplitMix64::new(0x5_000B);
    let (d_out, d_in) = (8, 2 * DIM);
    let mut wins = 0;
    let trials = 4;
    for _ in 0..trials {
        let h = random_hessian(&mut rng, d_in, 4 * d_in, 1e-2);
        let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
        let w0 = random_weights(&mut rng, d_out, d_in);

        let cfg = GptqConfig {
            block: DIM,
            retract: true,
            group_scales: false,
            lambda: 0.0,
            tail: TailPolicy::Reject,
        };
        let mut w = Weights::new(d_out, d_in, w0.clone());
        quantize_layer(&mut w, &f, None, &mut LeechDirection::new(), &cfg);
        let d_gptq: Vec<f64> = w.w.iter().zip(w0.iter()).map(|(a, b)| a - b).collect();

        // Same codebook, no error feedback: quantize each block of the
        // original weights independently.
        let mut q = LeechDirection::new();
        let mut plain = w0.clone();
        let mut out = vec![0.0f64; DIM];
        for i in 0..d_out {
            for blk in 0..d_in / DIM {
                let s = i * d_in + blk * DIM;
                q.quantize(&w0[s..s + DIM], &mut out);
                plain[s..s + DIM].copy_from_slice(&out);
            }
        }
        let d_plain: Vec<f64> = plain.iter().zip(w0.iter()).map(|(a, b)| a - b).collect();

        if proxy_loss(&d_gptq, d_out, d_in, &h) < proxy_loss(&d_plain, d_out, d_in, &h) {
            wins += 1;
        }
    }
    assert_eq!(
        wins, trials,
        "Spherical GPTQ lost to feedback-free quantization on {} of {trials} layers",
        trials - wins
    );
}

/// The tail policy must be an explicit choice, and under `KeepExact` the
/// trailing columns must never be forced onto the quantization grid.
///
/// They are still *modified* — they receive the error feedback of every
/// earlier block, which is the point: being unquantized they can absorb that
/// compensation exactly. So the assertion is about the grid, not about
/// equality with the original weights.
#[test]
fn keep_exact_tail_is_never_snapped_to_the_grid() {
    let mut rng = SplitMix64::new(0x5_000C);
    let (d_out, d_in, b) = (5, 14, 4); // 14 = 4·3 + 2
    let step = 0.7;
    let h = random_hessian(&mut rng, d_in, 60, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);
    let mut w = Weights::new(d_out, d_in, w0.clone());
    let cfg = GptqConfig {
        block: b,
        retract: false,
        group_scales: false,
        lambda: 0.0,
        tail: TailPolicy::KeepExact,
    };
    quantize_layer(&mut w, &f, None, &mut ScalarGrid { block: b, step }, &cfg);

    let on_grid = |v: f64| ((v / step) - (v / step).round()).abs() < 1e-9;
    for i in 0..d_out {
        for j in 0..12 {
            assert!(
                on_grid(w.w[i * d_in + j]),
                "column {j} of row {i} should sit on the grid, got {}",
                w.w[i * d_in + j]
            );
        }
        assert!(
            (12..d_in).any(|j| !on_grid(w.w[i * d_in + j])),
            "row {i}: the tail was snapped to the grid — it must stay free"
        );
        assert!(
            (12..d_in).any(|j| w.w[i * d_in + j] != w0[i * d_in + j]),
            "row {i}: the tail must still receive the earlier blocks' compensation"
        );
    }
    // 3 blocks of 4 at 48 bits, 2 columns at 16 → (3·48 + 2·16)/14.
    let bpw = bits_per_weight(d_in, b, 48, 16);
    assert!((bpw - (3.0 * 48.0 + 2.0 * 16.0) / 14.0).abs() < 1e-12);
}

/// And `Reject` must actually reject, rather than quietly skipping.
#[test]
#[should_panic(expected = "TailPolicy")]
fn reject_tail_panics() {
    let mut rng = SplitMix64::new(0x5_000D);
    let (d_out, d_in, b) = (3, 10, 4);
    let h = random_hessian(&mut rng, d_in, 40, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);
    let mut w = Weights::new(d_out, d_in, w0);
    let cfg = GptqConfig {
        block: b,
        retract: false,
        group_scales: false,
        lambda: 0.0,
        tail: TailPolicy::Reject,
    };
    quantize_layer(&mut w, &f, None, &mut ScalarGrid { block: b, step: 0.7 }, &cfg);
}

/// Splitting by row must be exact, not merely close: rows never interact in
/// GPTQ, so the parallel path has to reproduce the serial one bit for bit.
/// Anything less would mean a row is reading state it should not see.
#[test]
fn parallel_matches_serial_exactly() {
    let mut rng = SplitMix64::new(0x5_000E);
    let (d_out, d_in, b) = (23, 20, 4); // row count deliberately not a multiple
    let h = random_hessian(&mut rng, d_in, 80, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let w0 = random_weights(&mut rng, d_out, d_in);
    let cfg = GptqConfig {
        block: b,
        retract: true,
        group_scales: true,
        lambda: 1e-3,
        tail: TailPolicy::KeepExact,
    };

    let mut serial = Weights::new(d_out, d_in, w0.clone());
    quantize_layer(
        &mut serial,
        &f,
        Some(&h),
        &mut ScalarGrid { block: b, step: 0.8 },
        &cfg,
    );

    for threads in [2usize, 3, 8, 64] {
        let mut par = Weights::new(d_out, d_in, w0.clone());
        llvq_quant::gptq::quantize_layer_parallel(
            &mut par,
            &f,
            Some(&h),
            &(|| Box::new(ScalarGrid { block: b, step: 0.8 }) as Box<dyn BlockQuantizer>),
            &cfg,
            threads,
        );
        assert_eq!(
            par.w, serial.w,
            "{threads} threads diverged from the serial result"
        );
    }
}

/// The group-scale ridge must be relative to the system it damps.
///
/// Real LLM weights are ~1e-2, so `M[p,q] = g_p H g_q` is orders of magnitude
/// smaller than for the unit-scale random data the other tests use. An
/// *absolute* `λ` then dwarfs `M`, the solve degenerates to `s ≈ r/λ`, and the
/// blocks get scaled toward zero — which on a real model shows up as a
/// perplexity in the millions, not as a failed assertion.
///
/// This test pins the scale-invariance directly: multiplying the weights by a
/// constant must not change what the refinement decides, for any λ.
#[test]
fn group_scale_ridge_is_scale_invariant() {
    let mut rng = SplitMix64::new(0x5_000F);
    let (d_out, d_in, b) = (4, 48, 24);
    let h = random_hessian(&mut rng, d_in, 200, 1e-2);
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let base = random_weights(&mut rng, d_out, d_in);

    let run = |scale: f64, lambda: f64| -> Vec<f64> {
        let w0: Vec<f64> = base.iter().map(|v| v * scale).collect();
        let mut w = Weights::new(d_out, d_in, w0.clone());
        let cfg = GptqConfig {
            block: b,
            retract: true,
            group_scales: true,
            lambda,
            tail: TailPolicy::KeepExact,
        };
        quantize_layer(
            &mut w,
            &f,
            Some(&h),
            &mut ScalarGrid {
                block: b,
                // grid proportional to the data, so the codebook is the same
                // problem at both scales
                step: 0.8 * scale,
            },
            &cfg,
        );
        // Report the reconstruction relative to its own scale.
        w.w.iter().map(|v| v / scale).collect()
    };

    for &lambda in &[0.0, 1e-4, 1e-2] {
        let unit = run(1.0, lambda);
        let tiny = run(0.02, lambda); // the magnitude of real LLM weights
        let worst = unit
            .iter()
            .zip(tiny.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-6,
            "λ = {lambda}: the refinement is not scale-invariant \
             (max relative divergence {worst:.3e}) — the ridge is absolute \
             where it must be relative to mean(diag M)"
        );
    }
}

/// The two factorization paths must agree.
///
/// `fast-linalg` swaps four hand-written cubic loops for faer's blocked ones.
/// The identities asserted elsewhere hold for both, but only a direct
/// comparison catches a transposition or a triangle-convention mismatch
/// between them — the kind of error that produces a *valid* factor of the
/// wrong thing.
#[test]
#[cfg(feature = "fast-linalg")]
fn both_factorizations_agree() {
    let mut rng = SplitMix64::new(0x5_0010);
    for &n in &[7usize, 24, 64, 129] {
        let h = random_hessian(&mut rng, n, 3 * n, 1e-2);
        let fast = GptqFactor::new(&h, n, 1e-2).expect("SPD");
        let slow = GptqFactor::new_reference(&h, n, 1e-2).expect("SPD");
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                worst = worst.max((fast.u(i, j) - slow.u(i, j)).abs());
                scale = scale.max(slow.u(i, j).abs());
            }
        }
        assert!(
            worst <= 1e-8 * scale.max(1.0),
            "n = {n}: faer and the reference disagree by {worst:.3e} (scale {scale:.3e})"
        );
    }
}
