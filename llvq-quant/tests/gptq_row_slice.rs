//! # The GPTQ propagation reads `U` by row slice, and encodes the same bits
//!
//! The propagation onto the untouched columns reads `U[s + k][r0..]` as one
//! slice per `k` instead of one bounds-checked `u(i, j)` per cell. Each cell
//! still takes one multiply and one subtract per `k`, in increasing `k`, so
//! the layer must come out bit for bit as before. `quantize_layer_indexed`
//! below is the loop of `quantize_layer_capturing` at `925930d`, copied for
//! the configurations it covers (no group scales, no design C, no spherical
//! feedback, tail kept exact), with the propagation still indexed cell by
//! cell. The sealed files come out of this loop: one different bit is a
//! different file.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{quantize_layer, quantize_layer_parallel, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{row_scale, BlockQuantizer, LeechDirection, ScalarGrid};
use std::time::Instant;

/// `quantize_layer_capturing` at `925930d` for `retract` on or off, the tail
/// kept exact, nothing else enabled: the propagation indexes `U` per cell.
fn quantize_layer_indexed(
    weights: &mut Weights,
    factor: &GptqFactor,
    quant: &mut dyn BlockQuantizer,
    block: usize,
    retract: bool,
) {
    let (d_out, d_in) = (weights.d_out, weights.d_in);
    let mut qbuf = vec![0.0f64; block];
    let mut e = vec![0.0f64; d_out * block];
    let row_scales: Vec<f64> = (0..d_out)
        .map(|i| row_scale(&weights.w[i * d_in..(i + 1) * d_in]))
        .collect();

    let mut s = 0usize;
    while s < d_in {
        let b = block.min(d_in - s);
        if b != quant.block_len() {
            break; // TailPolicy::KeepExact
        }
        for i in 0..d_out {
            let row = &mut weights.w[i * d_in..(i + 1) * d_in];
            quant.set_row_scale(row_scales[i]);
            let v = &row[s..s + b];
            let norm_before = if retract {
                v.iter().map(|a| a * a).sum::<f64>().sqrt()
            } else {
                0.0
            };
            quant.quantize(v, &mut qbuf[..b]);
            if retract {
                if let Some(target) = quant.retraction_target(norm_before) {
                    let n = qbuf[..b].iter().map(|a| a * a).sum::<f64>().sqrt();
                    if n > 0.0 {
                        let k = target / n;
                        for q in qbuf[..b].iter_mut() {
                            *q *= k;
                        }
                    }
                }
            }
            for k in 0..b {
                e[i * b + k] = row[s + k] - qbuf[k];
                row[s + k] = qbuf[k];
            }
        }

        let r0 = s + b;
        if r0 < d_in {
            let ecur = &mut e[..d_out * b];
            factor.solve_block(ecur, d_out, s, b);
            for i in 0..d_out {
                let row = &mut weights.w[i * d_in..(i + 1) * d_in];
                for k in 0..b {
                    let x = ecur[i * b + k];
                    if x == 0.0 {
                        continue;
                    }
                    for (j, cell) in row[r0..].iter_mut().enumerate() {
                        *cell -= x * factor.u(s + k, r0 + j);
                    }
                }
            }
        }

        s = r0;
    }
}

fn cfg(retract: bool) -> GptqConfig {
    GptqConfig {
        block: DIM,
        retract,
        group_scales: false,
        design_c: false,
        spherical_feedback: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    }
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
    h
}

/// Rows over four orders of magnitude, one row of zeros (every residual is
/// zero, so every `k` is skipped) and one row already on the scalar grid.
fn weights(rng: &mut SplitMix64, d_out: usize, d_in: usize) -> Vec<f64> {
    let mut w: Vec<f64> = (0..d_out * d_in)
        .map(|k| 10f64.powi((k / d_in) as i32 % 4 - 2) * rng.next_gaussian())
        .collect();
    w[..d_in].fill(0.0);
    for (j, x) in w[d_in..2 * d_in].iter_mut().enumerate() {
        *x = 0.05 * (j as f64 - 40.0);
    }
    w
}

fn bits(w: &[f64]) -> Vec<u64> {
    w.iter().map(|x| x.to_bits()).collect()
}

/// Both factorizations, `faer`'s under `fast-linalg` and the in-house one.
fn factors(h: &[f64], n: usize) -> [(&'static str, GptqFactor); 2] {
    [
        ("GptqFactor::new", GptqFactor::new(h, n, 1e-2).expect("SPD")),
        ("GptqFactor::new_reference", GptqFactor::new_reference(h, n, 1e-2).expect("SPD")),
    ]
}

#[test]
fn a_row_of_u_is_the_entries_u_returns() {
    let mut rng = SplitMix64::new(0x0A04_0001);
    for n in [1usize, 7, 24, 61] {
        let h = random_hessian(&mut rng, n, 3 * n + 4);
        for (name, f) in factors(&h, n) {
            for i in 0..n {
                let row = f.u_row(i);
                assert_eq!(row.len(), n, "{name}, n {n}: row {i}");
                for (j, x) in row.iter().enumerate() {
                    assert_eq!(x.to_bits(), f.u(i, j).to_bits(), "{name}, n {n}: ({i}, {j})");
                }
            }
        }
    }
}

/// The layer, sliced against indexed, on a scalar grid: retraction on and
/// off, with and without a tail, on both factors.
#[test]
fn the_sliced_propagation_encodes_the_indexed_bits_on_a_scalar_grid() {
    // 6 blocks and a 7-wide tail; 8 blocks and no tail; one block, nothing
    // to propagate onto; a layer narrower than a block.
    let shapes: [(u64, usize, usize); 4] =
        [(1, 9, 6 * DIM + 7), (2, 5, 8 * DIM), (3, 4, DIM), (4, 3, 16)];
    for (seed, d_out, d_in) in shapes {
        let mut rng = SplitMix64::new(0x0A04_0100 + seed);
        let base = weights(&mut rng, d_out, d_in);
        let h = random_hessian(&mut rng, d_in, 2 * d_in + 8);
        for (name, f) in factors(&h, d_in) {
            for retract in [false, true] {
                for step in [0.05, 0.5] {
                    let mut want = Weights::new(d_out, d_in, base.clone());
                    let mut q = ScalarGrid { block: DIM, step };
                    quantize_layer_indexed(&mut want, &f, &mut q, DIM, retract);

                    let mut got = Weights::new(d_out, d_in, base.clone());
                    let mut q = ScalarGrid { block: DIM, step };
                    quantize_layer(&mut got, &f, None, &mut q, &cfg(retract));
                    assert_eq!(
                        bits(&got.w),
                        bits(&want.w),
                        "{name}, d_in {d_in}, retract {retract}, step {step}"
                    );
                }
            }
        }
    }
}

/// The same on the Leech direction code, whose retraction is not a no-op on
/// the residual, and across threads.
#[test]
fn the_sliced_propagation_encodes_the_indexed_bits_on_leech_and_in_parallel() {
    let (d_out, d_in) = (6, 3 * DIM + 8);
    let mut rng = SplitMix64::new(0x0A04_0200);
    let base = weights(&mut rng, d_out, d_in);
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    for (name, f) in factors(&h, d_in) {
        let mut want = Weights::new(d_out, d_in, base.clone());
        quantize_layer_indexed(&mut want, &f, &mut LeechDirection::new(), DIM, true);
        let mut got = Weights::new(d_out, d_in, base.clone());
        quantize_layer(&mut got, &f, None, &mut LeechDirection::new(), &cfg(true));
        assert_eq!(bits(&got.w), bits(&want.w), "{name}, Leech direction");

        let mut want = Weights::new(d_out, d_in, base.clone());
        let mut q = ScalarGrid { block: DIM, step: 0.05 };
        quantize_layer_indexed(&mut want, &f, &mut q, DIM, false);
        let mut got = Weights::new(d_out, d_in, base.clone());
        quantize_layer_parallel(
            &mut got,
            &f,
            None,
            &|| Box::new(ScalarGrid { block: DIM, step: 0.05 }),
            &cfg(false),
            4,
        );
        assert_eq!(bits(&got.w), bits(&want.w), "{name}, four threads");
    }
}

/// Times `quantize_layer` against the indexed loop on a 512 × 4096 layer
/// with a scalar grid, so that the time is the propagation's, interleaved
/// round by round. Prints the median per layer over the rounds and the range
/// the rounds formed. An instrument, run by hand in release with
/// `--features fast-linalg` (the factor of a 4096-wide Hessian otherwise
/// takes minutes): `--ignored --nocapture`.
#[test]
#[ignore]
fn the_sliced_propagation_times_against_the_indexed_one() {
    const ROUNDS: usize = 5;
    let (d_out, d_in): (usize, usize) = (512, 4096);
    let mut rng = SplitMix64::new(0x0A04_0300);
    let base: Vec<f64> = (0..d_out * d_in).map(|_| 0.02 * rng.next_gaussian()).collect();
    // A Kac-Murdock-Szego matrix, 0.5^|i-j|: positive definite, built in n².
    let h: Vec<f64> = (0..d_in * d_in)
        .map(|k| 0.5f64.powi((k / d_in).abs_diff(k % d_in) as i32))
        .collect();
    let f = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let (mut old_s, mut new_s) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        let mut want = Weights::new(d_out, d_in, base.clone());
        let start = Instant::now();
        let mut q = ScalarGrid { block: DIM, step: 0.01 };
        quantize_layer_indexed(&mut want, &f, &mut q, DIM, false);
        old_s.push(start.elapsed().as_secs_f64());
        let mut got = Weights::new(d_out, d_in, base.clone());
        let start = Instant::now();
        quantize_layer(&mut got, &f, None, &mut ScalarGrid { block: DIM, step: 0.01 }, &cfg(false));
        new_s.push(start.elapsed().as_secs_f64());
        assert_eq!(bits(&got.w), bits(&want.w), "round {round}");
    }
    for (arm, s) in [("indexed", &mut old_s), ("sliced", &mut new_s)] {
        s.sort_by(f64::total_cmp);
        println!(
            "512 x 4096, scalar grid, one thread: {arm} median {:.3} s per layer, \
             range {:.3}..{:.3} over {ROUNDS} rounds",
            s[ROUNDS / 2],
            s[0],
            s[ROUNDS - 1]
        );
    }
}
