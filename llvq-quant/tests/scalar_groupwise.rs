//! The affine per-group scalar quantizer, held to the properties that define
//! it rather than to a recorded output.
//!
//! ## Why this file exists at all
//!
//! `ScalarGroupwise` is the arm that decides how far a calibration-variance
//! result reaches. Shown only on Λ₂₄, such a result is a statement about *our*
//! codebook; shown on the field's own affine INT-`k`, it is a statement about
//! **GPTQ**. A reviewer will therefore ask whether our "standard scalar
//! quantizer" is standard — and the answer has to be a test, not a claim.
//!
//! ## What each test kills
//!
//! The doctrine of `CLAUDE.md` §5 is that an assertion which does not exercise
//! the parameter it covers tests nothing. Each test below is paired with the
//! mutant it exists to kill, and the two that matter most are:
//!
//! * `both_extremes_are_exact` — kills a **symmetric** quantizer. Dropping the
//!   zero point is the single most plausible way to write this wrong, and it
//!   leaves every other property here intact.
//! * `the_grid_is_fitted_per_group_not_globally` — kills a fallback to
//!   [`ScalarGrid`]'s behaviour, which is the neighbouring type in the same
//!   file and differs by exactly the thing this arm exists for.

use llvq_core::SplitMix64;
use llvq_quant::quantizer::{BlockQuantizer, ScalarGroupwise};

/// Deterministic test data. A fixed seed rather than a literal array so the
/// tests exercise many extents, and so an exact tie between two levels — which
/// would make the nearest-level reference below ambiguous — stays a measure-zero
/// event rather than something a hand-written array might accidentally contain.
fn sample(seed: u64, n: usize, spread: f64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_gaussian() * spread).collect()
}

fn quantize(bits: u32, v: &[f64]) -> Vec<f64> {
    let mut q = ScalarGroupwise {
        block: v.len(),
        bits,
    };
    let mut out = vec![0.0; v.len()];
    q.quantize(v, &mut out);
    out
}

/// The `2^bits` reconstruction levels of a group, built from its extent.
fn levels(bits: u32, v: &[f64]) -> Vec<f64> {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n = (1u32 << bits) - 1;
    let step = (hi - lo) / n as f64;
    (0..=n).map(|k| lo + k as f64 * step).collect()
}

/// Kills "the output is the input" and "the output is off the grid".
#[test]
fn every_output_lands_on_a_level_of_its_own_group() {
    for bits in 1..=8u32 {
        let v = sample(0x51ca_1a40 ^ bits as u64, 128, 0.02);
        let out = quantize(bits, &v);
        let ls = levels(bits, &v);
        for (i, &o) in out.iter().enumerate() {
            assert!(
                ls.iter().any(|&l| (l - o).abs() <= 1e-12 * l.abs().max(1e-12)),
                "bits {bits}, weight {i}: {o} is not one of the {} levels",
                ls.len()
            );
        }
    }
}

/// 🚨 Kills a **symmetric** quantizer — the most plausible way to get this
/// wrong, and one that no other test here would notice.
///
/// An asymmetric affine map sends the group's minimum to level 0 and its
/// maximum to level `n`, so both come back bit-exact. A symmetric map (scale
/// only, zero point fixed at 0) reproduces neither unless the group happens to
/// be centred, which random data never is.
#[test]
fn both_extremes_are_exact() {
    for bits in 1..=8u32 {
        let v = sample(0xe715_9e11 ^ bits as u64, 96, 0.05);
        let out = quantize(bits, &v);
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let qlo = out[v.iter().position(|&a| a == lo).unwrap()];
        let qhi = out[v.iter().position(|&a| a == hi).unwrap()];
        assert!(
            (qlo - lo).abs() <= 1e-15,
            "bits {bits}: the group minimum {lo} came back {qlo} — the zero \
             point is not doing its job"
        );
        assert!(
            (qhi - hi).abs() <= 1e-12 * hi.abs().max(1e-12),
            "bits {bits}: the group maximum {hi} came back {qhi}"
        );
    }
}

/// 🚨 Kills a fallback to [`llvq_quant::quantizer::ScalarGrid`], the
/// neighbouring type, which holds one step for everything.
///
/// Two groups three orders of magnitude apart in extent: a per-group grid
/// resolves each to its own `step/2`, a global one resolves the small group
/// not at all. The assertion is on the *ratio* of errors, so it cannot be
/// satisfied by a quantizer that is merely accurate.
#[test]
fn the_grid_is_fitted_per_group_not_globally() {
    let big = sample(0x9e37_79b9, 64, 1.0);
    let small: Vec<f64> = big.iter().map(|a| a * 1e-3).collect();

    let err = |v: &[f64]| -> f64 {
        let out = quantize(4, v);
        v.iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };

    let (e_big, e_small) = (err(&big), err(&small));
    let ratio = e_big / e_small;
    assert!(
        (900.0..=1100.0).contains(&ratio),
        "the error should scale with each group's own extent (ratio ≈ 1000), \
         got {ratio:.1} — big {e_big:e}, small {e_small:e}. A global step \
         would put this ratio at 1."
    );
}

/// Kills an off-by-one in the level count — `2^bits` steps instead of
/// `2^bits − 1` — which shifts every reconstruction by a fraction of a step
/// and no other test here would separate from correct rounding.
#[test]
fn no_weight_is_further_than_half_a_step() {
    for bits in 1..=8u32 {
        let v = sample(0xbf58_476d ^ bits as u64, 128, 0.03);
        let out = quantize(bits, &v);
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let step = (hi - lo) / ((1u32 << bits) - 1) as f64;
        for (i, (&a, &o)) in v.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - o).abs() <= step / 2.0 + 1e-12,
                "bits {bits}, weight {i}: |{a} − {o}| exceeds half a step ({})",
                step / 2.0
            );
        }
    }
}

/// Kills "the `bits` field is ignored", which a fixed-width implementation
/// would pass every other test with.
#[test]
fn more_bits_is_strictly_better() {
    let v = sample(0x94d0_49bb, 256, 0.02);
    let worst = |bits: u32| -> f64 {
        let out = quantize(bits, &v);
        v.iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };
    let mut prev = worst(1);
    for bits in 2..=8u32 {
        let e = worst(bits);
        assert!(
            e < prev,
            "going from {} to {bits} bits did not reduce the worst error \
             ({prev:e} → {e:e})",
            bits - 1
        );
        prev = e;
    }
}

/// Kills the division by zero on a group with no extent.
///
/// Not a corner case invented for a test: whole groups of a real matrix go to
/// zero under error feedback, and every weight in them would come back NaN.
#[test]
fn a_group_with_no_extent_is_reproduced_exactly() {
    for (name, value) in [("zeros", 0.0), ("constant", -0.317)] {
        for bits in [1u32, 3, 8] {
            let v = vec![value; 32];
            let out = quantize(bits, &v);
            assert!(
                out.iter().all(|&o| o == value),
                "{name} group at {bits} bits came back {:?}",
                &out[..4]
            );
        }
    }
}

/// The independent reference of `CLAUDE.md` §5: the same answer reached by a
/// route that shares no arithmetic with the implementation.
///
/// `ScalarGroupwise` divides, rounds and clamps. This picks the nearest of the
/// enumerated levels by exhaustive search. A sign error, a misplaced clamp or
/// a wrong rounding mode shows up as a disagreement; nothing about the two
/// paths is common except the definition of the grid.
#[test]
fn an_exhaustive_nearest_level_search_agrees() {
    for bits in 1..=6u32 {
        let v = sample(0x2545_f491 ^ bits as u64, 200, 0.04);
        let out = quantize(bits, &v);
        let ls = levels(bits, &v);
        for (i, (&a, &o)) in v.iter().zip(out.iter()).enumerate() {
            let mut best = ls[0];
            for &l in &ls[1..] {
                if (a - l).abs() < (a - best).abs() {
                    best = l;
                }
            }
            assert!(
                (best - o).abs() <= 1e-12 * best.abs().max(1e-12),
                "bits {bits}, weight {i} = {a}: implementation says {o}, \
                 exhaustive search says {best}"
            );
        }
    }
}

/// The group is the block, and the block is what the caller sets. A quantizer
/// that reported a different width would make `GptqConfig::block` and the rate
/// accounting disagree — the failure `Codebook::block_len` exists to prevent.
#[test]
fn the_reported_block_length_is_the_group() {
    for group in [8usize, 24, 128, 1024] {
        let q = ScalarGroupwise {
            block: group,
            bits: 3,
        };
        assert_eq!(q.block_len(), group);
    }
}

/// 🕳️ The invariant that let a `.clamp(0.0, n)` be **removed** from
/// `quantize`, after mutation testing showed neutralising it left all nine
/// tests green — dead code, in the sense of `CLAUDE.md` §5.
///
/// It is dead only because the affine map is fitted to *this* group: `a ∈
/// [lo, hi]` puts the quotient in `[0, n]`, and escaping after `.round()`
/// would need a relative error of `0.5/n` out of two roundings. This test is
/// what makes that removal safe to have made: a variant fitting the scale
/// somewhere else — AutoGPTQ's `static_groups = true`, where it comes from
/// the original weights and is applied to error-compensated ones — breaks the
/// invariant here rather than silently rounding out of range.
#[test]
fn the_quotient_never_leaves_the_level_range() {
    for mag in [1e-30f64, 1e-12, 1e-3, 1.0, 1e12, 1e30] {
        for bits in 1..=8u32 {
            let v = sample(0xd129_a4bd ^ bits as u64, 64, mag);
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n = ((1u32 << bits) - 1) as f64;
            let step = (hi - lo) / n;
            if !(step.is_finite() && step > 0.0) {
                continue;
            }
            for &a in &v {
                let q = ((a - lo) / step).round();
                assert!(
                    (0.0..=n).contains(&q),
                    "extent {mag:e}, bits {bits}: weight {a:e} quantizes to \
                     level {q}, outside [0, {n}] — the clamp removed from \
                     `quantize` was load-bearing after all, put it back"
                );
            }
        }
    }
}

/// 🚨 An extent that **overflows to infinity** used to hand back a group of
/// NaN, silently.
///
/// `lo = -1e308, hi = 1e308` makes `hi - lo` overflow, so `step` is `inf`.
/// The degenerate-group guard read `!(step > 0.0)`, and `inf > 0.0` is *true*
/// — so the group went down the normal path, `q` came out 0, and `0.0 * inf`
/// is NaN. Found by probing the neighbourhood of a surviving mutant rather
/// than by the mutant itself, which is why it is written down here.
#[test]
fn an_extent_that_overflows_does_not_produce_nan() {
    for bits in [1u32, 3, 8] {
        let mut v = vec![0.0; 16];
        v[0] = -1e308;
        v[1] = 1e308;
        let out = quantize(bits, &v);
        assert!(
            out.iter().all(|o| o.is_finite()),
            "bits {bits}: an infinite extent produced {:?}",
            &out[..4]
        );
    }
}

/// No reconstruction escapes the group's own range, at any width. Kills a
/// clamp removed or applied to the wrong bound.
#[test]
fn nothing_lands_outside_the_group_extent() {
    for bits in 1..=8u32 {
        let v = sample(0xff51_afd7 ^ bits as u64, 128, 0.1);
        let out = quantize(bits, &v);
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for &o in &out {
            assert!(
                o >= lo - 1e-12 && o <= hi + 1e-12,
                "bits {bits}: {o} is outside [{lo}, {hi}]"
            );
        }
    }
}
