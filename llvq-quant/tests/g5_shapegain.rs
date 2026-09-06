//! # Gate G5 — the gain code
//!
//! The defect this suite exists for: a "shape–gain" quantizer that keeps the
//! block magnitude as a free float is not a 2 bit/weight quantizer, however
//! exactly its direction costs 48 bits per 24 weights. That error survived a
//! full 4B run and a published figure, because nothing asserted that the
//! magnitude was *quantized*. It does now.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::{
    fit_gain_centroids, row_scale, BlockCode, BlockQuantizer, LeechDirection, LeechShapeGain,
    TrioShapeGain,
};
use llvq_search::trio::Trio;

/// The two 48-bit shape–gain arms: the exact ball at `cap = 12`, and the Trio
/// word map. They spend the same budget on the same 24 weights — 47 bits of
/// direction and one of gain — and differ only in how the direction is found
/// and written down, so every property of the *gain* code has to hold on
/// both. The tests below are written once and run twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Ball12,
    Trio,
}

const ARMS: [Arm; 2] = [Arm::Ball12, Arm::Trio];

impl Arm {
    fn make(self, centroids: Vec<f64>) -> Box<dyn BlockQuantizer> {
        match self {
            Arm::Ball12 => Box::new(LeechShapeGain::with_shell_cap(centroids, 12)),
            Arm::Trio => Box::new(TrioShapeGain::new(centroids)),
        }
    }
}

fn random_matrix(rng: &mut SplitMix64, d_out: usize, d_in: usize) -> Vec<f64> {
    // Rows deliberately at very different scales — that is what a per-row
    // reference is for, and a bug that ignores it shows up here.
    (0..d_out * d_in)
        .map(|k| {
            let row = k / d_in;
            let amp = 10f64.powi(row as i32 % 4 - 2);
            amp * rng.next_gaussian()
        })
        .collect()
}

/// The whole point: a reconstructed block's magnitude must be one of the
/// finitely many levels the code can express, not the input's own norm.
#[test]
fn the_gain_is_actually_quantized() {
    let mut rng = SplitMix64::new(0x6_0001);
    let (d_out, d_in) = (8usize, 4 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);
    assert_eq!(centroids.len(), 2, "one gain bit means two levels");

    for arm in ARMS {
        let mut q = arm.make(centroids.clone());
        let mut out = vec![0.0f64; DIM];
        let mut distinct: Vec<f64> = Vec::new();

        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            let rs = row_scale(row);
            q.set_row_scale(rs);
            for b in row.chunks_exact(DIM) {
                q.quantize(b, &mut out);
                let n = out.iter().map(|a| a * a).sum::<f64>().sqrt();
                // The magnitude must be a level times the row scale.
                let rel = n / rs;
                assert!(
                    centroids.iter().any(|c| (c - rel).abs() < 1e-9),
                    "{arm:?}: block magnitude {rel} is not one of the {} levels \
                     {centroids:?} — the gain is not being quantized",
                    centroids.len()
                );
                // And it must be the level **nearest** the block's own
                // relative norm. Without this the gain code could be
                // systematically wrong — every level swapped for the other —
                // and stay invisible: `quantize` and `reconstruct` would
                // agree, so the round trip of `g6_artifact` seals bit for bit
                // on a model that spends its one gain bit backwards.
                let want = b.iter().map(|a| a * a).sum::<f64>().sqrt() / rs;
                let picked = centroids
                    .iter()
                    .copied()
                    .min_by(|a, c| (want - a).abs().total_cmp(&(want - c).abs()))
                    .expect("two levels");
                assert!(
                    (rel - picked).abs() < 1e-9,
                    "{arm:?}: block of relative norm {want} was put on level {rel}, \
                     but {picked} is nearer among {centroids:?}"
                );
                if !distinct.iter().any(|d| (d - rel).abs() < 1e-9) {
                    distinct.push(rel);
                }
            }
        }
        assert!(
            distinct.len() > 1,
            "{arm:?}: every block landed on the same level; the code is degenerate"
        );
    }
    assert_eq!(LeechShapeGain::new(centroids.clone()).gain_bits(), 1);
    assert_eq!(TrioShapeGain::new(centroids).gain_bits(), 1);
}

/// And the control: the direction-only quantizer does *not* quantize the
/// magnitude. If this ever starts passing the assertion above, the two types
/// have been confused.
#[test]
fn direction_only_keeps_the_magnitude_free() {
    let mut rng = SplitMix64::new(0x6_0002);
    let mut q = LeechDirection::new();
    let mut out = vec![0.0f64; DIM];
    let mut norms = Vec::new();
    for _ in 0..32 {
        let v: Vec<f64> = (0..DIM).map(|_| rng.next_gaussian()).collect();
        q.quantize(&v, &mut out);
        let got = out.iter().map(|a| a * a).sum::<f64>().sqrt();
        let want = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!((got - want).abs() < 1e-9 * want, "magnitude must pass through");
        norms.push(got);
    }
    // 32 free floats, not a handful of levels.
    norms.sort_unstable_by(f64::total_cmp);
    norms.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    assert!(norms.len() > 16, "these should be free magnitudes, got {} levels", norms.len());
}

/// More gain bits must reduce reconstruction error, and zero bits must
/// collapse to a single level — the paper's true "zero gain bits".
#[test]
fn error_decreases_with_gain_bits() {
    let mut rng = SplitMix64::new(0x6_0003);
    let (d_out, d_in) = (6usize, 6 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);

    let err = |k: u32| -> f64 {
        let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, k, 40);
        assert_eq!(centroids.len(), 1usize << k);
        let mut q = LeechShapeGain::new(centroids);
        let mut out = vec![0.0f64; DIM];
        let mut e = 0.0;
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            q.set_row_scale(row_scale(row));
            for b in row.chunks_exact(DIM) {
                q.quantize(b, &mut out);
                e += b.iter().zip(out.iter()).map(|(a, c)| (a - c) * (a - c)).sum::<f64>();
            }
        }
        e
    };

    let (e0, e1, e2) = (err(0), err(1), err(2));
    assert!(e1 < e0, "one gain bit must beat zero: {e1} vs {e0}");
    assert!(e2 < e1, "two must beat one: {e2} vs {e1}");
}

/// The per-row scale is load-bearing: feed a wrong one and the gain code,
/// which only has a handful of levels, stops being able to express the block.
#[test]
fn the_row_scale_is_load_bearing() {
    let mut rng = SplitMix64::new(0x6_0004);
    let (d_out, d_in) = (6usize, 4 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);

    let err = |arm: Arm, use_row_scale: bool| -> f64 {
        let mut q = arm.make(centroids.clone());
        let mut out = vec![0.0f64; DIM];
        let mut e = 0.0;
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            q.set_row_scale(if use_row_scale { row_scale(row) } else { 1.0 });
            for b in row.chunks_exact(DIM) {
                q.quantize(b, &mut out);
                e += b.iter().zip(out.iter()).map(|(a, c)| (a - c) * (a - c)).sum::<f64>();
            }
        }
        e
    };
    for arm in ARMS {
        let (with, without) = (err(arm, true), err(arm, false));
        assert!(
            with < without * 0.5,
            "{arm:?}: a per-row reference must matter a lot across rows spanning \
             decades: {with} vs {without}"
        );
    }
}

/// The index width must match the ball actually searched — that one bit is
/// the whole reason to cap at shell 12.
#[test]
fn index_width_follows_the_shell_cap() {
    use llvq_quant::quantizer::index_bits;
    assert_eq!(index_bits(13), 48, "the full ball needs 48 bits");
    assert_eq!(index_bits(12), 47, "Λ24(12) fits in 47 — that pays the gain bit");
    assert_eq!(index_bits(11), 46, "Λ24(11) fits in 46 — that pays a second gain bit");
    assert_eq!(index_bits(10), 44, "Λ24(10) fits in 44 — that pays a fourth gain bit");
    // No ball costs 45 bits: the width jumps 44 → 46, which is why the ladder
    // has no 3-gain-bit rung and the source paper's Table 8 skips it too.
    assert!(
        !(2..=13u32).any(|c| index_bits(c) == 45),
        "a 45-bit ball would add a rung to the direction↔gain ladder"
    );
    // The four arms of the direction↔gain split must land on the same 48-bit
    // budget, or the A/B is not at constant rate and compares nothing.
    for (cap, gain_bits) in [(13u32, 0u32), (12, 1), (11, 2), (10, 4)] {
        assert_eq!(
            index_bits(cap) + gain_bits,
            48,
            "cap {cap} with {gain_bits} gain bits is not a 48-bit block"
        );
    }
    // Monotone, and never wider than the full ball.
    for c in 2..=13u32 {
        assert!(index_bits(c) <= 48);
        if c > 2 {
            assert!(index_bits(c) >= index_bits(c - 1));
        }
    }
}

/// And a capped quantizer must never emit a direction from a higher shell.
#[test]
fn capped_quantizer_stays_inside_its_ball() {
    let leech = llvq_core::Leech::new();
    let mut rng = SplitMix64::new(0x6_0005);
    let (d_out, d_in) = (4usize, 8 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);
    let mut q = LeechShapeGain::with_shell_cap(centroids, 12);

    let mut out = vec![0.0f64; DIM];
    for i in 0..d_out {
        let row = &w[i * d_in..(i + 1) * d_in];
        q.set_row_scale(row_scale(row));
        for b in row.chunks_exact(DIM) {
            q.quantize(b, &mut out);
            let n = out.iter().map(|a| a * a).sum::<f64>().sqrt();
            if n == 0.0 {
                continue;
            }
            // Recover the shell: reconstruction = point · n/√(16m).
            let mut shell = None;
            for m in 2..=13u32 {
                let c = ((16 * m) as f64).sqrt() / n;
                let pt: llvq_core::Point =
                    core::array::from_fn(|k| (out[k] * c).round() as i32);
                if (0..DIM).all(|k| (out[k] * c - pt[k] as f64).abs() < 1e-6)
                    && llvq_core::Leech::shell_index(&pt) == Some(m as u64)
                    && leech.contains(&pt)
                {
                    shell = Some(m);
                    break;
                }
            }
            let m = shell.expect("reconstruction must come from Λ24");
            assert!(m <= 12, "cap 12 violated: direction on shell {m}");
        }
    }
}

/// Every direction a `TrioShapeGain` emits is a **word of the map, in
/// natural order** — the property the whole format rests on and the one a
/// magnitude test cannot see.
///
/// Three things at once, because they fail differently. `Leech::contains`
/// says the point is a lattice point at all. `Trio::encode` says the map has
/// a word for it: it is the writer's own call, and it returns `None` for a
/// point whose sections leave their row sets. `Trio::decode` of that word
/// says the round trip closes.
///
/// **The mutant this is here for** is the coordinate order. `Encoder`
/// permutes `x` into trio order, works there, and permutes back through
/// `order()`; drop that last permutation and the point is still a Λ₂₄ point
/// of the same norm — every magnitude assertion above still passes, the GPTQ
/// residual barely moves — but it is the wrong point and `Trio::encode`
/// refuses it. Applying `order` where its inverse belongs is caught the same
/// way (`order` is not an involution).
/// `TrioShapeGain::reconstruct` rebuilds the block from the code ALONE, and
/// this pins it against an independent formula rather than against itself.
///
/// It exists because two mutants of that method survived the whole workspace
/// on 2026-09-06: dropping the row scale, and reading the neighbouring
/// centroid. `reconstruct` is what `reproject` calls, so those mutants live
/// on the design-C path; the ball twin was killed by a design-C test and the
/// Trio one was covered by nothing. The check here is deliberately NOT a
/// round trip: `quantize` writing and `reconstruct` reading the same wrong
/// convention would agree with each other and prove nothing.
#[test]
fn trio_reconstruct_is_the_centroid_the_row_scale_and_the_direction() {
    let mut rng = SplitMix64::new(0x7_0007);
    let (d_out, d_in) = (5usize, 3 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);
    assert_eq!(centroids.len(), 2, "one gain bit");
    let q = TrioShapeGain::new(centroids.clone());
    let trio = Trio::new();

    // Codes built here, not captured from `quantize`: the point is any word
    // of the map, and the gain any level, so that a reconstruction reading
    // the wrong level or forgetting the scale has nowhere to hide.
    let mut out = vec![0.0f64; DIM];
    for k in 0..64u64 {
        let word = SplitMix64::new(0xC0DE_0000 + k).next() & ((1u64 << 47) - 1);
        let point = trio.decode(word);
        let norm = (point.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt();
        if norm == 0.0 {
            continue; // the origin carries no direction
        }
        for gain in 0..2u32 {
            for &row_scale in &[1.0f64, 0.375, 17.25] {
                let code = BlockCode { point, gain };
                q.reconstruct(&code, row_scale, &mut out);
                for (j, &got) in out.iter().enumerate() {
                    let want = centroids[gain as usize] * row_scale * point[j] as f64 / norm;
                    // Not bit equality: the reference divides by `√(16m)` read
                    // from the shell index where this recomputes `‖y‖` from
                    // the coordinates. The two are the same number and round
                    // to within an ulp. Both mutants this test exists for move
                    // the result by a factor, not by an ulp.
                    let tol = 1e-12 * want.abs().max(1e-9);
                    assert!(
                        (got - want).abs() <= tol,
                        "word {word:#x}, gain {gain}, row scale {row_scale}, coordinate {j}: \
                         {got} against {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_trio_direction_is_a_word_of_the_map_in_natural_order() {
    let mut rng = SplitMix64::new(0x6_0006);
    let (d_out, d_in) = (6usize, 4 * DIM);
    let w = random_matrix(&mut rng, d_out, d_in);
    let centroids = fit_gain_centroids(&w, d_out, d_in, DIM, 1, 40);
    let trio = Trio::new();
    let leech = llvq_core::Leech::new();
    let mut q = TrioShapeGain::new(centroids);
    let mut out = vec![0.0f64; DIM];

    let mut blocks = 0usize;
    for i in 0..d_out {
        let row = &w[i * d_in..(i + 1) * d_in];
        q.set_row_scale(row_scale(row));
        for b in row.chunks_exact(DIM) {
            q.quantize(b, &mut out);
            let code = q.last_code().expect("a Trio block always emits a code");
            assert!(
                leech.contains(&code.point),
                "row {i} block {blocks}: {:?} is not in Λ24",
                code.point
            );
            let word = trio
                .encode(&code.point)
                .unwrap_or_else(|| panic!("row {i} block {blocks}: the map has no word for {:?} — the point is not in natural order", code.point));
            assert_eq!(trio.decode(word), code.point, "the word does not decode back");
            assert_eq!(word >> 47, 0, "the encoder must leave the gain bit to the gain code");
            blocks += 1;
        }
    }
    assert_eq!(blocks, d_out * (d_in / DIM));
}

/// A Trio block is 48 bits — 47 of label, one of gain — and the type refuses
/// any other shape of gain code at construction, where it is still a
/// caller's mistake and not a file `llvq-artifact` cannot write.
#[test]
fn a_trio_block_is_forty_seven_bits_of_label_and_one_of_gain() {
    let q = TrioShapeGain::new(vec![0.5, 1.5]);
    assert_eq!(q.gain_bits(), 1);
    assert_eq!(q.block_bits(), 48, "47 + 1, the budget of the served ball arm");
    assert_eq!(q.block_len(), DIM);
    assert_eq!(
        q.block_bits(),
        llvq_quant::quantizer::index_bits(12) + 1,
        "Trio and the capped ball must cost the same, or the A/B compares nothing"
    );
    // The retraction is a no-op: the block is already on the level's sphere.
    assert_eq!(q.retraction_target(1.234), None);
}

#[test]
#[should_panic(expected = "the gain code has two levels")]
fn a_trio_quantizer_refuses_a_gain_code_the_word_cannot_carry() {
    // Four levels is two gain bits; the word has room for one.
    let _ = TrioShapeGain::new(vec![0.25, 0.75, 1.25, 1.75]);
}
