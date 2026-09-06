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
use llvq_quant::quantizer::{
    fit_gain_centroids, reconstruct_shape_gain, row_scale, BlockCode, BlockQuantizer,
    LeechShapeGain, TetraShapeGain,
};
use llvq_search::tetra::Tetra;

const D_OUT: usize = 6;

/// The two 48-bit direction codes the round trip has to hold for: the exact
/// ball, and the Tetra word map. They share their gain code, their per-row
/// scale and — the point of the exercise — their reconstruction, which is the
/// single [`reconstruct_shape_gain`] both this test and `llvq_artifact`'s
/// `decode_matrix` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// The ball at the given shell cap: 13 for the full ball, 12 or 11 for
    /// the arms that buy gain bits with index bits.
    Ball(u32),
    /// Tetra, which has one gain bit and no cap to choose.
    Tetra,
}

impl Arm {
    fn make(self, centroids: Vec<f64>) -> Box<dyn BlockQuantizer> {
        match self {
            Arm::Ball(cap) => Box::new(LeechShapeGain::with_shell_cap(centroids, cap)),
            Arm::Tetra => Box::new(TetraShapeGain::new(centroids)),
        }
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
fn round_trip(d_in: usize, arm: Arm, gain_bits: u32, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let base = weights(&mut rng, d_in);
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    let factor = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, d_in, DIM, gain_bits, 40);

    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };

    let nblocks = d_in / DIM;
    let mut codes: Vec<Option<BlockCode>> = vec![None; D_OUT * nblocks];
    let mut w = Weights::new(D_OUT, d_in, base.clone());
    let mut q = arm.make(centroids.clone());
    quantize_layer_capturing(
        &mut w,
        &factor,
        None,
        q.as_mut(),
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

    // A Tetra matrix goes to disk as a 48-bit word and comes back as a point,
    // so the decoder this test stands in for is `Tetra::decode ∘ Tetra::encode`
    // and not the identity. Running the codes through it here is what pins
    // the writer's own refusal: `encode` returns `None` for anything the map
    // has no word for.
    let tetra = (arm == Arm::Tetra).then(Tetra::new);

    let mut block = vec![0.0f64; DIM];
    for i in 0..D_OUT {
        for p in 0..nblocks {
            let mut code = codes[i * nblocks + p].expect("checked above");
            if let Some(tetra) = &tetra {
                let word = tetra
                    .encode(&code.point)
                    .unwrap_or_else(|| panic!("row {i}, block {p}: the map has no word for {:?}", code.point));
                code.point = tetra.decode(word);
            }
            reconstruct_shape_gain(&code, &centroids, scales[i], &mut block);
            let want = &w.w[i * d_in + p * DIM..i * d_in + (p + 1) * DIM];
            for (k, (&got, &exp)) in block.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "{arm:?}, row {i}, block {p}, coord {k}: decoded {got:e} but the \
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
    round_trip(4 * DIM, Arm::Ball(13), 1, 0x6_A001);
}

#[test]
fn codes_reconstruct_bit_for_bit_under_a_shell_cap() {
    round_trip(4 * DIM, Arm::Ball(12), 1, 0x6_A002);
}

#[test]
fn codes_reconstruct_bit_for_bit_with_zero_gain_bits() {
    // One level for the whole tensor — the paper's true "zero gain bits".
    round_trip(3 * DIM, Arm::Ball(12), 0, 0x6_A003);
}

#[test]
fn codes_reconstruct_bit_for_bit_with_two_gain_bits() {
    // The third arm of the 48-bit split: Λ24(11) costs 46 index bits, so two
    // bits are left for the gain. Four levels had never been exercised through
    // the GPTQ loop — only through the file format and the runtime layouts.
    round_trip(4 * DIM, Arm::Ball(11), 2, 0x6_A006);
}

#[test]
fn codes_reconstruct_bit_for_bit_beside_a_tail() {
    // 100 = 24·4 + 4: the tail stays exact and is stored verbatim, but the
    // four coded blocks must still round-trip.
    round_trip(100, Arm::Ball(12), 1, 0x6_A004);
}

#[test]
fn codes_reconstruct_the_layer_bit_for_bit_on_tetra() {
    // The step-2 gate: same property, same loop, the Tetra word map instead of
    // the ball. The reconstruction is the same function on both sides, so what
    // this actually exercises is the encoder's point — its order, its
    // membership of the map, and the level the gain code picked for it.
    round_trip(4 * DIM, Arm::Tetra, 1, 0x6_A007);
}

#[test]
fn codes_reconstruct_bit_for_bit_on_tetra_beside_a_tail() {
    // 100 = 24·4 + 4, as for the ball: the four coded blocks round-trip and
    // the tail is stored verbatim.
    round_trip(100, Arm::Tetra, 1, 0x6_A008);
}

/// Both 48-bit arms take the same three-block layer through the loop and
/// leave a finite, comparable residual.
///
/// This is a sanity check on the *pair*, not a quality claim: Tetra's rule is
/// not the exact nearest neighbour, so its residual is expected to be the
/// larger of the two, and the number that decides anything is a perplexity.
/// What would fail here is an arm that returns NaN, or one whose residual is
/// so far off the other's that the direction code is not being used at all —
/// the shape a wrongly-ordered point, or a dropped row scale, takes.
#[test]
fn both_arms_leave_finite_residuals_on_three_blocks() {
    let d_in = 3 * DIM;
    let mut rng = SplitMix64::new(0x6_A009);
    let base = weights(&mut rng, d_in);
    let h = random_hessian(&mut rng, d_in, 4 * d_in);
    let factor = GptqFactor::new(&h, d_in, 1e-2).expect("SPD");
    let centroids = fit_gain_centroids(&base, D_OUT, d_in, DIM, 1, 40);
    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };

    let residual = |arm: Arm| -> f64 {
        let mut w = Weights::new(D_OUT, d_in, base.clone());
        let mut q = arm.make(centroids.clone());
        quantize_layer_capturing(&mut w, &factor, None, q.as_mut(), &cfg, None);
        assert!(w.w.iter().all(|v| v.is_finite()), "{arm:?} produced a non-finite weight");
        w.w.iter().zip(base.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
    };
    let energy: f64 = base.iter().map(|a| a * a).sum();
    let (ball, tetra) = (residual(Arm::Ball(12)), residual(Arm::Tetra));
    println!("3 blocks × {D_OUT} rows: ball-12 residual {ball:.6e}, Tetra {tetra:.6e}, energy {energy:.6e}");
    for (arm, r) in [("ball-12", ball), ("Tetra", tetra)] {
        assert!(r.is_finite() && r > 0.0, "{arm}: residual {r} is not a finite loss");
        assert!(r < 0.5 * energy, "{arm}: residual {r:.3e} against {energy:.3e} of signal — the direction code is not being used");
    }
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
        design_c: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    let nblocks = d_in / DIM;

    // One Tetra encoder for the whole test: `make_quantizer` runs once per
    // thread and per call, and the encoder's tables cost ~10 ms to build.
    // Sharing them is also what `llvq-llm` will do on a real layer.
    let shared = TetraShapeGain::encoder();

    let run = |arm: Arm, threads: usize| -> (Vec<f64>, Vec<Option<BlockCode>>) {
        let mut w = Weights::new(d_out, d_in, base.clone());
        let mut codes = vec![None; d_out * nblocks];
        let cs = centroids.clone();
        let shared = shared.clone();
        let make = move || -> Box<dyn BlockQuantizer> {
            match arm {
                Arm::Ball(cap) => Box::new(LeechShapeGain::with_shell_cap(cs.clone(), cap)),
                Arm::Tetra => Box::new(TetraShapeGain::with_encoder(shared.clone(), cs.clone())),
            }
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

    for arm in [Arm::Ball(12), Arm::Tetra] {
        let (w1, c1) = run(arm, 1);
        for threads in [2usize, 3, 4, 8] {
            let (wn, cn) = run(arm, threads);
            assert_eq!(w1, wn, "{arm:?}: {threads} threads changed the weights");
            assert_eq!(
                c1, cn,
                "{arm:?}: {threads} threads changed the codes — the row split and \
                 the code split have drifted apart"
            );
        }
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
        design_c: false,
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
