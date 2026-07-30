//! # Gate G5 — incoherence rotation
//!
//! A rotation that is not exactly orthogonal silently changes the function the
//! layer computes, and the damage would look like "2 bits is lossy" rather
//! than "the transform is wrong". So orthogonality is asserted directly, on
//! the widths the real models actually use — including the ones that are not
//! powers of two.

use llvq_core::SplitMix64;
use llvq_quant::rotation::Rotation;

/// Widths of Qwen3-0.6B (1024, 2048, 3072) plus awkward shapes: an odd
/// dimension, a prime, and a bare power of two.
const DIMS: &[usize] = &[1, 3, 7, 16, 24, 96, 1024, 2048, 3072];

#[test]
fn rotation_is_orthogonal_on_every_width() {
    for &n in DIMS {
        let q = Rotation::new(n, 0x9_0001 + n as u64);
        // Qᵀ Q = I, column by column.
        for j in 0..n.min(64) {
            let mut e = vec![0.0f64; n];
            e[j] = 1.0;
            q.apply(&mut e);
            let norm: f64 = e.iter().map(|a| a * a).sum();
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "n = {n}: column {j} has norm² {norm}, must be 1"
            );
            for i in 0..j {
                let mut f = vec![0.0f64; n];
                f[i] = 1.0;
                q.apply(&mut f);
                let dot: f64 = e.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-9,
                    "n = {n}: columns {i} and {j} are not orthogonal (dot {dot})"
                );
            }
        }
    }
}

#[test]
fn transpose_undoes_the_rotation() {
    let mut rng = SplitMix64::new(0x9_0002);
    for &n in DIMS {
        let q = Rotation::new(n, 0x9_0003 + n as u64);
        let v0: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mut v = v0.clone();
        q.apply(&mut v);
        q.apply_transpose(&mut v);
        let worst = v0
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "n = {n}: QᵀQ ≠ I, worst coordinate {worst:.3e}");
        // And it must actually rotate — a transform that does nothing would
        // pass every test above.
        let mut moved = v0.clone();
        q.apply(&mut moved);
        let delta = v0
            .iter()
            .zip(moved.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        if n > 1 {
            assert!(delta > 1e-6, "n = {n}: the rotation is the identity");
        }
    }
}

/// `rotate_weight_rows` must implement `W Qᵀ` exactly, i.e. leave `W x`
/// unchanged once the input is rotated too. This is the identity the whole
/// scheme rests on.
#[test]
fn rotating_weights_and_inputs_preserves_the_layer() {
    let mut rng = SplitMix64::new(0x9_0004);
    let (d_out, n) = (5usize, 96usize);
    let q = Rotation::new(n, 0x9_0005);
    let w: Vec<f64> = (0..d_out * n).map(|_| rng.next_gaussian()).collect();
    let x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

    let y: Vec<f64> = (0..d_out)
        .map(|i| (0..n).map(|j| w[i * n + j] * x[j]).sum())
        .collect();

    let mut wr = w.clone();
    q.rotate_weight_rows(&mut wr, d_out);
    let mut xr = x.clone();
    q.apply(&mut xr);
    let yr: Vec<f64> = (0..d_out)
        .map(|i| (0..n).map(|j| wr[i * n + j] * xr[j]).sum())
        .collect();

    for (a, b) in y.iter().zip(yr.iter()) {
        assert!((a - b).abs() < 1e-9, "(W Qᵀ)(Q x) ≠ W x: {a} vs {b}");
    }

    // And rotating back must restore the weights bit-for-bit-ish.
    q.unrotate_weight_rows(&mut wr, d_out);
    let worst = w
        .iter()
        .zip(wr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-9, "unrotate did not restore W, worst {worst:.3e}");
}

/// `rotate_hessian` must equal `Q H Qᵀ` computed densely.
#[test]
fn hessian_rotation_matches_a_dense_product() {
    let mut rng = SplitMix64::new(0x9_0006);
    let n = 24usize;
    let q = Rotation::new(n, 0x9_0007);

    // A symmetric positive definite H.
    let a: Vec<f64> = (0..(4 * n) * n).map(|_| rng.next_gaussian()).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..4 * n {
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] += a[r * n + i] * a[r * n + j];
            }
        }
    }

    // Dense reference: build Q explicitly, then Q H Qᵀ.
    let mut qm = vec![0.0f64; n * n];
    for j in 0..n {
        let mut e = vec![0.0f64; n];
        e[j] = 1.0;
        q.apply(&mut e);
        for i in 0..n {
            qm[i * n + j] = e[i];
        }
    }
    let mut want = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..n {
                for r in 0..n {
                    acc += qm[i * n + p] * h[p * n + r] * qm[j * n + r];
                }
            }
            want[i * n + j] = acc;
        }
    }

    let mut got = h.clone();
    q.rotate_hessian(&mut got);
    let worst = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let scale = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        worst < 1e-9 * scale.max(1.0),
        "Q H Qᵀ mismatch: worst {worst:.3e} (scale {scale:.3e})"
    );
}
