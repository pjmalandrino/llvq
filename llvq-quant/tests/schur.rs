#![forbid(unsafe_code)]

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{quantize_layer_capturing, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{BlockQuantizer, TetraShapeGain};
use llvq_quant::schur::{
    commit, continuous_lower_bound, diagnose_row, rollout_loss, validation_loss,
};

// Independent Gaussian elimination with partial pivoting, not the GPTQ factor.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        let d = a[k * n + k];
        assert!(d.abs() > 1e-14);
        for j in k..n {
            a[k * n + j] /= d;
        }
        b[k] /= d;
        for i in 0..n {
            if i == k {
                continue;
            }
            let m = a[i * n + k];
            for j in k..n {
                a[i * n + j] -= m * a[k * n + j];
            }
            b[i] -= m * b[k];
        }
    }
    b
}

fn quadratic(h: &[f64], e: &[f64]) -> f64 {
    let n = e.len();
    (0..n)
        .map(|i| (0..n).map(|j| e[i] * h[i * n + j] * e[j]).sum::<f64>())
        .sum()
}

fn spd(n: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(123);
    let a: Vec<_> = (0..n * n).map(|_| rng.next_gaussian()).collect();
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n)
                .map(|k| a[k * n + i] * a[k * n + j] / n as f64)
                .sum::<f64>()
                + if i == j { 0.7 } else { 0.0 }
        })
        .collect()
}

fn near(a: f64, b: f64) {
    // Absolute floor for exact zeros; relative tolerance above unit scale.
    assert!(
        (a - b).abs() <= 2e-10 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}

// Minimum with a fixed prefix, plus its minimizing continuation.
fn conditional(h: &[f64], w: &[f64], fixed: &[f64]) -> (f64, Vec<f64>) {
    let n = w.len();
    let f = fixed.len();
    let r = n - f;
    let a: Vec<_> = (f..n)
        .flat_map(|i| (f..n).map(move |j| h[i * n + j]))
        .collect();
    let b: Vec<_> = (f..n)
        .map(|i| {
            -(0..f)
                .map(|j| h[i * n + j] * (fixed[j] - w[j]))
                .sum::<f64>()
        })
        .collect();
    let e = if r == 0 { vec![] } else { solve(a, b) };
    let q: Vec<_> = fixed
        .iter()
        .copied()
        .chain((0..r).map(|i| w[f + i] + e[i]))
        .collect();
    let error: Vec<_> = q.iter().zip(w).map(|(q, w)| q - w).collect();
    (quadratic(h, &error), q)
}

#[test]
fn schur_matches_independent_elimination_with_nonempty_prefix_and_damping() {
    let n = 9;
    let mut h = spd(n);
    let damping = 0.03;
    let ridge = damping * (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let factor = GptqFactor::new(&h, n, damping).unwrap();
    for i in 0..n {
        h[i * n + i] += ridge;
    }
    let w: Vec<_> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let mut work = w.clone();
    let prefix = [-0.8, 0.4];
    let cf = commit(&factor, &mut work, 0, &prefix);
    let (expected_cf, center) = conditional(&h, &w, &prefix);
    near(cf, expected_cf);
    for i in 2..n {
        near(work[i], center[i]);
    }
    let candidate = [0.2, -0.9, 0.5];
    let e: Vec<_> = work[2..5]
        .iter()
        .zip(candidate)
        .map(|(x, q)| x - q)
        .collect();
    let d0 = continuous_lower_bound(&factor, 2, &e);
    let fixed: Vec<_> = prefix.into_iter().chain(candidate).collect();
    let (minimum, optimal) = conditional(&h, &w, &fixed);
    near(cf + d0, minimum);
    commit(&factor, &mut work, 2, &candidate);
    for i in 0..n {
        near(work[i], optimal[i]);
    }
    near(rollout_loss(&factor, &w, &work), minimum);
    // Complete the square at an arbitrary suffix, not just the optimum.
    let mut q = work.clone();
    for (i, x) in q[5..].iter_mut().enumerate() {
        *x += 0.13 * (i + 1) as f64;
    }
    let err: Vec<_> = q.iter().zip(&w).map(|(q, w)| q - w).collect();
    let delta: Vec<_> = q[5..].iter().zip(&work[5..]).map(|(a, b)| a - b).collect();
    let hrr: Vec<_> = (5..n)
        .flat_map(|i| {
            let h = &h;
            (5..n).map(move |j| h[i * n + j])
        })
        .collect();
    near(quadratic(&h, &err), cf + d0 + quadratic(&hrr, &delta));
}

#[test]
fn exact_tiny_suffix_gap_and_nested_horizons() {
    let h = spd(5);
    let factor = GptqFactor::new(&h, 5, 0.0).unwrap();
    let w = [0.1, -0.8, 0.6, 0.4, -0.3];
    let fixed = [0.7, -0.2];
    let (d0, center) = conditional(&h, &w, &fixed);
    let mut bounds = vec![d0];
    // Last coordinate is a free KeepExact tail. Enumerate only coordinates 2,3.
    for k in 1..=2 {
        let mut best = f64::INFINITY;
        for mask in 0..(1 << k) {
            let mut prefix = fixed.to_vec();
            for j in 0..k {
                prefix.push(if mask & (1 << j) == 0 { -0.5 } else { 0.8 });
            }
            best = best.min(conditional(&h, &w, &prefix).0);
        }
        bounds.push(best);
    }
    assert!(bounds.windows(2).all(|x| x[1] + 1e-12 >= x[0]));
    assert!(bounds[2] > bounds[0] + 1e-6);
    let mut gap = f64::INFINITY;
    for a in [-0.5, 0.8] {
        for b in [-0.5, 0.8] {
            let (loss, q) = conditional(&h, &w, &[fixed[0], fixed[1], a, b]);
            let mut suffix_error = vec![0.0; 5];
            for i in 2..5 {
                suffix_error[i] = q[i] - center[i];
            }
            near(loss, d0 + quadratic(&h, &suffix_error));
            gap = gap.min(quadratic(&h, &suffix_error));
        }
    }
    near(bounds[2] - d0, gap);
    let mut work = w.to_vec();
    commit(&factor, &mut work, 0, &fixed);
    near(rollout_loss(&factor, &w, &work), bounds[0]);
}

#[test]
fn diagonal_has_no_feedback_and_isotropic_rankings_agree() {
    let n = 2 * DIM + 3;
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = 2.5;
    }
    let factor = GptqFactor::new(&h, n, 0.0).unwrap();
    let w: Vec<_> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut work = w.clone();
    commit(&factor, &mut work, 0, &[0.5; DIM]);
    assert_eq!(&work[DIM..], &w[DIM..]);
    let d = diagnose_row(&w, &factor, [0.7, 1.05], &w, &[0, 1]);
    for c in &d.shadow {
        for g in 0..2 {
            near(c.conditional[g], 2.5 * c.euclidean[g]);
        }
        assert_eq!(c.choices[1], c.choices[2]);
    }
}

#[test]
fn witness_and_every_branch_reproduce_production_gptq_with_exact_tail() {
    let n = 2 * DIM + 3;
    let h = spd(n);
    let factor = GptqFactor::new(&h, n, 0.0).unwrap();
    let w: Vec<_> = (0..n).map(|i| (i as f64 * 1.3).sin()).collect();
    let levels = [0.7, 1.05];
    let d = diagnose_row(&w, &factor, levels, &w, &[0, 1]);
    let mut weights = Weights::new(1, n, w.clone());
    let mut quant = TetraShapeGain::new(levels.to_vec());
    let mut codes = vec![None; n / DIM];
    quantize_layer_capturing(
        &mut weights,
        &factor,
        None,
        &mut quant,
        &GptqConfig {
            tail: TailPolicy::KeepExact,
            ..Default::default()
        },
        Some(&mut codes),
    );
    assert_eq!(d.witness, weights.w);
    assert_eq!(
        d.witness_codes,
        codes.into_iter().map(Option::unwrap).collect::<Vec<_>>()
    );
    assert_ne!(&d.witness[2 * DIM..], &w[2 * DIM..]);
    let shadow_only = diagnose_row(&w, &factor, levels, &w, &[]);
    assert_eq!(shadow_only.witness, d.witness);
    assert_eq!(d.encoder_calls, 4); // two shadow calls plus two suffix calls
    for snapshot in &d.snapshots {
        let c = &snapshot.comparison;
        for g in (0..2).rev() {
            // Reverse evaluation order, freshly allocated quantizer and row.
            let mut q = snapshot.working.clone();
            let mut quant = TetraShapeGain::new(levels.to_vec());
            quant.set_row_scale(d.row_scale);
            commit(&factor, &mut q, c.block * DIM, &c.reconstructed[g]);
            if c.block == 0 {
                let mut second = [0.0; DIM];
                quant.quantize(&q[DIM..2 * DIM], &mut second);
                commit(&factor, &mut q, DIM, &second);
            }
            let branch = &snapshot.branches[g];
            assert_eq!(q, branch.reconstructed);
            let error: Vec<_> = q.iter().zip(&w).map(|(q, w)| q - w).collect();
            near(branch.rollout_loss, quadratic(&h, &error));
            assert!(branch.rollout_excess >= -1e-10);
            near(
                branch.continuous_lower_bound,
                c.prefix_loss + c.conditional[g],
            );
            if c.block == 1 {
                near(branch.rollout_excess, 0.0);
            }
            near(branch.validation_loss, validation_loss(&w, &w, &q));
            for (b, code) in branch.codes.iter().enumerate() {
                let mut decoded = [0.0; DIM];
                quant.reconstruct(code, d.row_scale, &mut decoded);
                assert_eq!(&q[b * DIM..(b + 1) * DIM], &decoded);
            }
        }
    }
}

#[test]
fn constructed_post_gain_crossing_scale_ties_zero_and_threshold() {
    let mut rng = SplitMix64::new(99);
    let mut x: Vec<_> = (0..DIM).map(|_| rng.next_gaussian()).collect();
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for a in &mut x {
        *a /= n;
    }
    let encoder = TetraShapeGain::encoder();
    let mut probe = TetraShapeGain::with_encoder(encoder.clone(), vec![0.7, 1.1]);
    let mut tmp = [0.0; DIM];
    probe.quantize(&x, &mut tmp);
    let point = probe.last_code().unwrap().point;
    let pn = point
        .iter()
        .map(|&p| (p as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let projected = x.iter().zip(point).map(|(x, p)| x * p as f64).sum::<f64>() / pn;
    assert!(projected < 1.0 - 1e-6);
    // Construct a midpoint strictly between projection and norm: crossing guaranteed.
    let mid = (projected + 1.0) / 2.0;
    let levels = vec![mid - 0.1, mid + 0.1];
    for scale in [0.125, 1.0, 7.0] {
        let input: Vec<_> = x.iter().map(|v| v * scale).collect();
        let mut old = TetraShapeGain::with_encoder(encoder.clone(), levels.clone());
        let mut post =
            TetraShapeGain::with_encoder(encoder.clone(), levels.clone()).with_post_shape_gain();
        old.set_row_scale(scale);
        post.set_row_scale(scale);
        old.quantize(&input, &mut tmp);
        let old_code = old.last_code().unwrap();
        post.quantize(&input, &mut tmp);
        let new_code = post.last_code().unwrap();
        assert_eq!((old_code.gain, new_code.gain), (1, 0));
        assert_eq!(old_code.point, new_code.point);
        let mut losses = [0.0; 2];
        for (g, loss) in losses.iter_mut().enumerate() {
            let mut out = [0.0; DIM];
            post.reconstruct(
                &llvq_quant::quantizer::BlockCode {
                    gain: g as u32,
                    ..new_code
                },
                scale,
                &mut out,
            );
            *loss = out.iter().zip(&input).map(|(q, x)| (q - x).powi(2)).sum();
            assert!(out.iter().all(|v| v.is_finite()));
        }
        assert!(losses[0] < losses[1]);
    }
    for delta in [-1e-9, 0.0, 1e-9] {
        let mut q = TetraShapeGain::with_encoder(
            encoder.clone(),
            vec![projected - 0.1 + delta, projected + 0.1 + delta],
        )
        .with_post_shape_gain();
        q.quantize(&x, &mut tmp);
        let c = q.last_code().unwrap();
        let selected: f64 = x.iter().zip(tmp).map(|(x, q)| (x - q).powi(2)).sum();
        for g in 0..2 {
            q.reconstruct(
                &llvq_quant::quantizer::BlockCode { gain: g, ..c },
                1.0,
                &mut tmp,
            );
            let err: f64 = x.iter().zip(tmp).map(|(x, q)| (x - q).powi(2)).sum();
            assert!(selected <= err + 1e-12);
        }
    }
    let mut equal = TetraShapeGain::with_encoder(encoder, vec![0.9, 0.9]).with_post_shape_gain();
    equal.quantize(&x, &mut tmp);
    assert_eq!(equal.last_code().unwrap().gain, 0);
    equal.quantize(&[0.0; DIM], &mut tmp);
    assert_eq!(tmp, [0.0; DIM]);
    assert_eq!(equal.last_code().unwrap().point, [0; DIM]);
}

#[test]
fn anisotropic_metric_changes_the_gain_ranking_and_predicts_full_block_loss() {
    let mut rng = SplitMix64::new(99);
    let mut x: Vec<_> = (0..DIM).map(|_| rng.next_gaussian()).collect();
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for a in &mut x {
        *a /= norm;
    }
    let mut quant = TetraShapeGain::new(vec![0.7, 1.1]);
    let mut out = [0.0; DIM];
    quant.quantize(&x, &mut out);
    let p = quant.last_code().unwrap().point;
    let pn = p.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let u: Vec<_> = p.iter().map(|&v| v as f64 / pn).collect();
    let projected = x.iter().zip(&u).map(|(x, u)| x * u).sum::<f64>();
    let r: Vec<_> = x.iter().zip(&u).map(|(x, u)| x - projected * u).collect();
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let v: Vec<_> = u.iter().zip(&r).map(|(u, r)| u + r / rn).collect();
    let h: Vec<_> = (0..DIM * DIM)
        .map(|ij| {
            let (i, j) = (ij / DIM, ij % DIM);
            20.0 * v[i] * v[j] + if i == j { 1.0 } else { 0.0 }
        })
        .collect();
    let mid = (projected + 1.0) / 2.0;
    let factor = GptqFactor::new(&h, DIM, 0.0).unwrap();
    let d = diagnose_row(&x, &factor, [mid - 0.1, mid + 0.1], &x, &[0]);
    assert_eq!(d.shadow[0].choices, [1, 0, 1]);
    let branches = &d.snapshots[0].branches;
    assert!(branches[1].rollout_loss < branches[0].rollout_loss);
    for b in branches {
        near(b.rollout_excess, 0.0);
    }
}

#[test]
fn reserved_output_error_has_an_independent_scalar_reference() {
    // e = [1,-2,3], outputs: 1-4=-3 and -2+3=1. Mean square = 5.
    near(
        validation_loss(
            &[1.0, 2.0, 0.0, 0.0, 1.0, 1.0],
            &[0.2, 0.3, 0.4],
            &[1.2, -1.7, 3.4],
        ),
        5.0,
    );
}

#[test]
fn disabled_post_mode_matches_the_parent_norm_rule_and_refuses_reprojection() {
    use llvq_quant::quantizer::{reconstruct_shape_gain, BlockCode};
    use llvq_search::tetra::{Encoder, Scratch, Tetra};
    let enc = Encoder::new(&Tetra::new());
    let mut scratch = Scratch::new();
    let levels = [0.6, 1.15];
    let mut q = TetraShapeGain::new(levels.to_vec());
    let mut rng = SplitMix64::new(188);
    for scale in [0.1, 1.0, 3.0] {
        q.set_row_scale(scale);
        let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian() * scale);
        let point = enc.encode(&x, &mut scratch).point;
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
        let gain = u32::from((norm - levels[1]).abs() < (norm - levels[0]).abs());
        let expected = BlockCode { point, gain };
        let mut want = [0.0; DIM];
        reconstruct_shape_gain(&expected, &levels, scale, &mut want);
        let mut got = [0.0; DIM];
        q.quantize(&x, &mut got);
        assert_eq!(q.last_code(), Some(expected));
        assert_eq!(got, want);
    }
    let post = q.with_post_shape_gain();
    let code = BlockCode {
        point: [0; DIM],
        gain: 0,
    };
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        post.reproject(&code, 1.0, &mut [0.0; DIM]);
    }))
    .is_err());
}

#[test]
fn production_post_gain_propagates_its_own_error_into_the_second_block() {
    struct Recording {
        quant: TetraShapeGain,
        inputs: Vec<Vec<f64>>,
        outputs: Vec<Vec<f64>>,
    }
    impl BlockQuantizer for Recording {
        fn block_len(&self) -> usize {
            DIM
        }
        fn set_row_scale(&mut self, scale: f64) {
            self.quant.set_row_scale(scale);
        }
        fn retraction_target(&self, _: f64) -> Option<f64> {
            None
        }
        fn quantize(&mut self, x: &[f64], out: &mut [f64]) {
            self.inputs.push(x.to_vec());
            self.quant.quantize(x, out);
            self.outputs.push(out.to_vec());
        }
    }
    let mut rng = SplitMix64::new(99);
    let mut x: Vec<_> = (0..DIM).map(|_| rng.next_gaussian()).collect();
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for a in &mut x {
        *a /= norm;
    }
    let mut probe = TetraShapeGain::new(vec![0.8, 1.1]);
    let mut out = [0.0; DIM];
    probe.quantize(&x, &mut out);
    let p = probe.last_code().unwrap().point;
    let pn = p.iter().map(|&p| (p as f64).powi(2)).sum::<f64>().sqrt();
    let projection = x.iter().zip(p).map(|(x, p)| x * p as f64).sum::<f64>() / pn;
    let midpoint = (1.0 + projection) / 2.0;
    let levels = vec![midpoint - 0.1, midpoint + 0.1];
    let n = 2 * DIM;
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = 1.0;
    }
    for i in 0..DIM {
        h[i * n + i + DIM] = 0.4;
        h[(i + DIM) * n + i] = 0.4;
    }
    let factor = GptqFactor::new(&h, n, 0.0).unwrap();
    let w: Vec<_> = x.iter().chain(&x).copied().collect();
    let mut q = Recording {
        quant: TetraShapeGain::new(levels.clone()).with_post_shape_gain(),
        inputs: vec![],
        outputs: vec![],
    };
    let mut weights = Weights::new(1, n, w.clone());
    llvq_quant::gptq::quantize_layer(&mut weights, &factor, None, &mut q, &GptqConfig::default());
    assert_eq!(q.inputs[0], x);
    let mut norm_quant = TetraShapeGain::new(levels);
    norm_quant.set_row_scale(llvq_quant::quantizer::row_scale(&w));
    norm_quant.quantize(&x, &mut out);
    assert_ne!(q.outputs[0], out);
    // Independent conditional solve fixes the first reconstructed block.
    let (_, expected) = conditional(&h, &w, &q.outputs[0]);
    for (actual, expected) in q.inputs[1].iter().zip(&expected[DIM..]) {
        near(*actual, *expected);
    }
    let (_, wrong) = conditional(&h, &w, &out);
    assert!(q.inputs[1]
        .iter()
        .zip(&wrong[DIM..])
        .any(|(a, b)| (a - b).abs() > 1e-6));
}
