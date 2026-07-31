//! # Gate G6, step 1 — the codes have to be enough
//!
//! Before any file format, one property has to hold: the codes a layer emits
//! must reconstruct **exactly** the weights that were evaluated. Not to a
//! tolerance — bit for bit. Anything less means the artifact is a different
//! model from the one the perplexity was measured on, and no amount of
//! careful serialization afterwards can fix that.
//!
//! This is deliberately the *only* thing tested here. The bit packing, the
//! header and the 48-bit index are plumbing behind this property.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::gptq::{
    quantize_layer_capturing, GptqConfig, TailPolicy, Weights,
};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, row_scale, BlockCode, LeechShapeGain};

const D_OUT: usize = 6;

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

fn weights(rng: &mut SplitMix64, d_in: usize) -> Vec<f64> {
    (0..D_OUT * d_in)
        .map(|k| {
            let row = k / d_in;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

/// Quantize a layer while capturing codes, then rebuild it from the codes
/// alone and demand bit equality.
fn round_trip(d_in: usize, cap: u32, gain_bits: u32, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let base = weights(&mut rng, d_in);
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    let factor = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, d_in, DIM, gain_bits, 40);

    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };

    let nblocks = d_in / DIM;
    let mut codes: Vec<Option<BlockCode>> = vec![None; D_OUT * nblocks];
    let mut w = Weights::new(D_OUT, d_in, base.clone());
    let mut q = LeechShapeGain::with_shell_cap(centroids.clone(), cap);
    quantize_layer_capturing(
        &mut w,
        &factor,
        None,
        &mut q,
        &cfg,
        Some(codes.as_mut_slice()),
    );

    // Every quantized block must have produced a code.
    assert!(
        codes.iter().all(|c| c.is_some()),
        "a quantized block emitted no code"
    );

    // The decoder knows only: the codes, the per-row scale, and the matrix's
    // gain centroids. It does **not** know the original weights.
    let scales: Vec<f64> = (0..D_OUT)
        .map(|i| row_scale(&base[i * d_in..(i + 1) * d_in]))
        .collect();

    let mut block = vec![0.0f64; DIM];
    for i in 0..D_OUT {
        for p in 0..nblocks {
            let code = codes[i * nblocks + p].expect("checked above");
            q.reconstruct(&code, scales[i], &mut block);
            let want = &w.w[i * d_in + p * DIM..i * d_in + (p + 1) * DIM];
            for (k, (&got, &exp)) in block.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "row {i}, block {p}, coord {k}: decoded {got:e} but the \
                     evaluated weight is {exp:e} (Δ = {:e}). The artifact would \
                     be a different model from the one measured.",
                    got - exp
                );
            }
        }
    }
}

#[test]
fn codes_reconstruct_the_layer_bit_for_bit() {
    // Width a multiple of 24: no tail, every column is coded.
    round_trip(4 * DIM, 13, 1, 0x6_A001);
}

#[test]
fn codes_reconstruct_bit_for_bit_under_a_shell_cap() {
    round_trip(4 * DIM, 12, 1, 0x6_A002);
}

#[test]
fn codes_reconstruct_bit_for_bit_with_zero_gain_bits() {
    // One level for the whole tensor — the paper's true "zero gain bits".
    round_trip(3 * DIM, 12, 0, 0x6_A003);
}

#[test]
fn codes_reconstruct_bit_for_bit_beside_a_tail() {
    // 100 = 24·4 + 4: the tail stays exact and is stored verbatim, but the
    // four coded blocks must still round-trip.
    round_trip(100, 12, 1, 0x6_A004);
}

/// Splitting by rows must split the codes the same way.
///
/// The weights are cut with `rows_per * d_in` and the codes with
/// `rows_per * nblocks`. If those two ever disagree, one row silently inherits
/// another's codes — the weights would still be right, the artifact wrong, and
/// only a round-trip would notice. This is the code-side twin of
/// `parallel_matches_serial_exactly`.
#[test]
fn parallel_capture_matches_serial_capture() {
    let d_in = 5 * DIM;
    // Deliberately not a multiple of the thread count, so the last chunk is
    // short and the two slicings can drift apart.
    let d_out = 7usize;
    let mut rng = SplitMix64::new(0x6_A101);
    let base: Vec<f64> = (0..d_out * d_in)
        .map(|k| 10f64.powi((k / d_in) as i32 % 4 - 2) * rng.next_gaussian())
        .collect();
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    let factor = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, d_out, d_in, DIM, 1, 40);
    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    let nblocks = d_in / DIM;

    let run = |threads: usize| -> (Vec<f64>, Vec<Option<BlockCode>>) {
        let mut w = Weights::new(d_out, d_in, base.clone());
        let mut codes = vec![None; d_out * nblocks];
        let cs = centroids.clone();
        let make = move || -> Box<dyn llvq_quant::quantizer::BlockQuantizer> {
            Box::new(LeechShapeGain::with_shell_cap(cs.clone(), 12))
        };
        llvq_quant::gptq::quantize_layer_parallel_capturing(
            &mut w,
            &factor,
            None,
            &make,
            &cfg,
            threads,
            Some(codes.as_mut_slice()),
        );
        (w.w, codes)
    };

    let (w1, c1) = run(1);
    for threads in [2usize, 3, 4, 8] {
        let (wn, cn) = run(threads);
        assert_eq!(w1, wn, "{threads} threads changed the weights");
        assert_eq!(
            c1, cn,
            "{threads} threads changed the codes — the row split and the code \
             split have drifted apart"
        );
    }
}

/// The free-magnitude variant is *not* codeable, and the round-trip is what
/// says so. Without this, a future change could quietly reintroduce the
/// defect of 2026-07-31 and the artifact would silently drift from the model.
#[test]
fn the_free_magnitude_variant_cannot_round_trip() {
    let d_in = 4 * DIM;
    let mut rng = SplitMix64::new(0x6_A005);
    let base = weights(&mut rng, d_in);
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    let factor = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, d_in, DIM, 1, 40);
    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };

    let nblocks = d_in / DIM;
    let mut codes: Vec<Option<BlockCode>> = vec![None; D_OUT * nblocks];
    let mut w = Weights::new(D_OUT, d_in, base.clone());
    let mut q = LeechShapeGain::with_shell_cap(centroids.clone(), 12).with_free_magnitude();
    quantize_layer_capturing(
        &mut w,
        &factor,
        None,
        &mut q,
        &cfg,
        Some(codes.as_mut_slice()),
    );

    let scales: Vec<f64> = (0..D_OUT)
        .map(|i| row_scale(&base[i * d_in..(i + 1) * d_in]))
        .collect();
    let mut block = vec![0.0f64; DIM];
    let mut mismatches = 0usize;
    for i in 0..D_OUT {
        for p in 0..nblocks {
            let code = codes[i * nblocks + p].expect("emitted");
            q.reconstruct(&code, scales[i], &mut block);
            let want = &w.w[i * d_in + p * DIM..i * d_in + (p + 1) * DIM];
            if block
                .iter()
                .zip(want.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                mismatches += 1;
            }
        }
    }
    assert!(
        mismatches > 0,
        "with a free per-block magnitude the codes cannot describe the stored \
         weights, so the round-trip must fail — if it passes, the retraction \
         is no longer restoring the block norm and this variant has silently \
         become the honest one"
    );
}
