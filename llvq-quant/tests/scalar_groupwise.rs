//! The affine per-group scalar quantizer, held against **the upstream source
//! it transcribes** and against the properties that define it.
//!
//! ## Why this file exists at all
//!
//! `ScalarGroupwise` is the arm that decides how far a calibration-variance
//! result reaches. Shown only on Λ₂₄, such a result is a statement about *our*
//! codebook; shown on the field's own affine INT-`k`, it is a statement about
//! **GPTQ**. A reviewer will therefore ask whether our "standard scalar
//! quantizer" is standard — and the answer has to be a test, not a claim.
//!
//! ## 🕳️ What the cross-check caught
//!
//! The first version of this quantizer was written from memory, passed nine
//! property tests, and was **wrong in three ways** — extended range, integer
//! zero point, degenerate-group handling. Every one of them moves the grid,
//! and no property test written without the source in hand could have found
//! them, because each was internally consistent. That is the whole argument
//! for `the_transcription_matches_upstream` below: properties pin a
//! quantizer, only a transcription pins *which* quantizer.

use llvq_core::SplitMix64;
use llvq_quant::quantizer::{BlockQuantizer, ScalarGroupwise};

/// Deterministic test data. A fixed seed rather than a literal array, so the
/// tests exercise many extents rather than one hand-picked shape.
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

/// `find_params` of the upstream quantizer, `sym = false`, transcribed.
///
/// Returns `(scale, zero, maxq)`. Kept separate from the reference `quantize`
/// below because upstream keeps them separate, and because three tests need
/// the grid without needing the reconstruction.
fn upstream_params(bits: u32, x: &[f64]) -> (f64, f64, f64) {
    let maxq = ((1u32 << bits) - 1) as f64;
    // `tmp = torch.zeros(...)`, then `minimum(x.min(1)[0], tmp)` and
    // `maximum(x.max(1)[0], tmp)` — the range is extended to include zero.
    let mut xmin = x.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let mut xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0);
    // `tmp = (xmin == 0) & (xmax == 0); xmin[tmp] = -1; xmax[tmp] = +1`
    if xmin == 0.0 && xmax == 0.0 {
        xmin = -1.0;
        xmax = 1.0;
    }
    let scale = (xmax - xmin) / maxq;
    // `self.zero = torch.round(-xmin / self.scale)` — an integer.
    let zero = (-xmin / scale).round_ties_even();
    (scale, zero, maxq)
}

/// The `2^bits` reconstruction levels of a group: `scale·(k − zero)`.
fn levels(bits: u32, x: &[f64]) -> Vec<f64> {
    let (scale, zero, maxq) = upstream_params(bits, x);
    (0..=(maxq as u32)).map(|k| scale * (k as f64 - zero)).collect()
}

/// 🚨 The cross-check the whole arm rests on: an independent transcription of
/// upstream's two functions, and bit agreement with the implementation.
///
/// Source: `AutoGPTQ/AutoGPTQ@main:auto_gptq/quantization/quantizer.py`,
/// fetched 2026-08-26, sha256
/// `2e0b4588cfc5bd250c8a635697ee1a1d59d65741bf1d4e3a18ce2b79befe2a5d`. The
/// relevant lines, `sym = false, mse = false`:
///
/// ```text
/// tmp  = torch.zeros(x.shape[0])
/// xmin = torch.minimum(x.min(1)[0], tmp)
/// xmax = torch.maximum(x.max(1)[0], tmp)
/// tmp  = (xmin == 0) & (xmax == 0); xmin[tmp] = -1; xmax[tmp] = +1
/// scale = (xmax - xmin) / maxq
/// zero  = torch.round(-xmin / scale)
/// q     = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
/// return  scale * (q - zero)
/// ```
///
/// `torch.round` breaks ties to even, which is `round_ties_even` and *not*
/// `f64::round`; getting that wrong would show up here on tie inputs only,
/// which is exactly the class of bug a transcription claim has to exclude.
#[test]
fn the_transcription_matches_upstream() {
    for bits in 1..=8u32 {
        for spread in [1e-4f64, 0.02, 1.0, 50.0] {
            for shift in [-3.0f64, -0.4, 0.0, 0.4, 3.0] {
                // The shift walks the group across zero, so the extended-range
                // branch is exercised from "entirely negative" to "entirely
                // positive" rather than only near-centred.
                let v: Vec<f64> = sample(0x0141_5926 ^ bits as u64, 96, spread)
                    .into_iter()
                    .map(|a| a + shift * spread)
                    .collect();
                let (scale, zero, maxq) = upstream_params(bits, &v);
                let want: Vec<f64> = v
                    .iter()
                    .map(|&a| {
                        let q = ((a / scale).round_ties_even() + zero).clamp(0.0, maxq);
                        scale * (q - zero)
                    })
                    .collect();
                let got = quantize(bits, &v);
                for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
                    assert_eq!(
                        w.to_bits(),
                        g.to_bits(),
                        "bits {bits}, spread {spread:e}, shift {shift}: weight \
                         {i} = {} — upstream says {w}, we say {g}",
                        v[i]
                    );
                }
            }
        }
    }
}

/// 🚨 Kills the from-memory version, which used the group's raw extent.
///
/// Upstream extends the range to include zero, so an entirely positive group
/// gets `step = xmax/maxq` — coarser than `(xmax − xmin)/maxq` would be. The
/// difference is not subtle on a group far from zero: it is the whole offset.
#[test]
fn the_range_is_extended_to_include_zero() {
    // Entirely positive, and tight: raw extent 0.02, extended extent 1.01.
    let v: Vec<f64> = sample(0x7f4a_7c15, 64, 0.005)
        .into_iter()
        .map(|a| a + 1.0)
        .collect();
    let (scale, _, maxq) = upstream_params(4, &v);
    let xmax = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xmin = v.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (scale - xmax / maxq).abs() < 1e-12,
        "the step should be xmax/maxq = {}, got {scale}",
        xmax / maxq
    );
    assert!(
        scale > 10.0 * (xmax - xmin) / maxq,
        "a raw-extent step would be {} — the two must not be confusable here",
        (xmax - xmin) / maxq
    );
}

/// 🚨 The defining consequence of an **integer** zero point: `0.0` is on the
/// grid of every group, exactly.
///
/// The from-memory version used a float offset, which reproduced both extremes
/// exactly and put zero on the grid only by accident. Deployed formats want
/// the opposite trade, because zero is what padding and sparsity feed in.
#[test]
fn zero_is_exactly_representable_in_every_group() {
    for bits in 1..=8u32 {
        for shift in [-2.0f64, -0.3, 0.0, 0.3, 2.0] {
            let mut v: Vec<f64> = sample(0xa5a5_1234 ^ bits as u64, 48, 0.01)
                .into_iter()
                .map(|a| a + shift * 0.01)
                .collect();
            // Put an exact zero in the group and require it back exactly.
            v[7] = 0.0;
            let out = quantize(bits, &v);
            assert_eq!(
                out[7].to_bits(),
                0.0f64.to_bits(),
                "bits {bits}, shift {shift}: an exact zero came back {}",
                out[7]
            );
        }
    }
}

/// 🕳️ The clamp is not dead, but it is not reached by ordinary data either:
/// probing 38.4 M weights over all eight widths fired it zero times.
///
/// It guards **exact ties**. With `t = −xmin/scale`, the top of the range is
/// `round(maxq − t) + round(t)`, which equals `maxq` for every non-half-integer
/// `t`; at `t = 1.5, maxq = 7`, round-half-to-even makes it `6 + 2 = 8`. The
/// group below is built to land exactly there — `xmin = −3, xmax = 11`, so
/// `scale = 2` and `t = 1.5` — and without the clamp the reconstruction would
/// leave the representable grid.
#[test]
fn the_clamp_keeps_a_tie_group_on_the_grid() {
    let v = vec![-3.0, 0.0, 4.0, 7.0, 11.0];
    let (scale, zero, maxq) = upstream_params(3, &v);
    assert!(
        (scale - 2.0).abs() < 1e-12 && (zero - 2.0).abs() < 1e-12,
        "the constructed tie case drifted: scale {scale}, zero {zero} \
         (expected 2 and 2)"
    );
    // The tie itself: this is what would overflow `maxq` unclamped.
    let raw = (11.0f64 / scale).round_ties_even() + zero;
    assert!(
        raw > maxq,
        "this group no longer exercises the clamp: raw level {raw} <= {maxq}"
    );

    let out = quantize(3, &v);
    let top = scale * (maxq - zero);
    let bottom = scale * (0.0 - zero);
    for (i, &o) in out.iter().enumerate() {
        assert!(
            o >= bottom - 1e-12 && o <= top + 1e-12,
            "weight {i} = {} reconstructed to {o}, outside the representable \
             grid [{bottom}, {top}] — the clamp was load-bearing here",
            v[i]
        );
    }
}

/// 🚨 `torch.round` breaks ties **to even**; `f64::round` breaks them away
/// from zero. Swapping them is invisible on random data — exact ties are
/// measure-zero — so a transcription claim needs a group built to hit one.
///
/// 🕳️ Mutation testing found this gap: replacing `round_ties_even` with
/// `round` in the implementation left all eleven tests green, because none of
/// them contained a tie the two modes disagree on. They agree at `5.5` (both
/// give 6) and differ at `0.5` (0 against 1) — the tie has to sit at an
/// *even* half-integer.
///
/// The group below puts `−xmin/scale` at exactly `0.5`: with `xmin = −1` and
/// `xmax = 13` at three bits, `scale = 2`. Ties-to-even makes `zero = 0`,
/// away-from-zero makes it `1`, and every reconstruction in the group moves.
#[test]
fn the_rounding_mode_is_ties_to_even() {
    let v = vec![-1.0, 0.0, 5.0, 9.0, 13.0];
    let (scale, zero, _) = upstream_params(3, &v);
    assert!(
        (scale - 2.0).abs() < 1e-12,
        "the constructed tie case drifted: scale {scale} (expected 2)"
    );
    assert!(
        (-1.0f64 / scale).abs() == 0.5,
        "this group no longer sits on a tie"
    );
    assert_eq!(
        zero, 0.0,
        "torch.round breaks ties to even, so round(0.5) is 0 — getting 1 here \
         means the implementation uses f64::round and the transcription claim \
         is false"
    );
    // And the difference is observable in the output, not just in `zero`:
    // with `zero = 1` every level shifts by one step.
    let out = quantize(3, &v);
    assert_eq!(
        out[0].to_bits(),
        0.0f64.to_bits(),
        "with ties-to-even, −1 clamps to level 0 and reconstructs to 0; got {}",
        out[0]
    );
}

/// Kills a fallback to [`llvq_quant::quantizer::ScalarGrid`], the neighbouring
/// type, which holds one step for everything.
///
/// Two groups three orders of magnitude apart in extent: a per-group grid
/// resolves each to its own step, a global one resolves the small group not at
/// all. The assertion is on the *ratio* of errors, so it cannot be satisfied
/// by a quantizer that is merely accurate.
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

    let ratio = err(&big) / err(&small);
    assert!(
        (900.0..=1100.0).contains(&ratio),
        "the error should scale with each group's own extent (ratio ≈ 1000), \
         got {ratio:.1}. A global step would put this ratio at 1."
    );
}

/// Kills an off-by-one in the level count and a truncation in place of
/// rounding: absent a tie, the reconstruction is `scale·round(x/scale)`, so
/// nothing sits further than half a step from its weight.
#[test]
fn no_weight_is_further_than_half_a_step() {
    for bits in 1..=8u32 {
        let v = sample(0xbf58_476d ^ bits as u64, 128, 0.03);
        let out = quantize(bits, &v);
        let (scale, _, _) = upstream_params(bits, &v);
        for (i, (&a, &o)) in v.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - o).abs() <= scale / 2.0 + 1e-12,
                "bits {bits}, weight {i}: |{a} − {o}| exceeds half a step ({})",
                scale / 2.0
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

/// The degenerate cases, both of which run on real matrices.
///
/// An all-zero group takes upstream's `[-1, +1]` branch and still comes back
/// zero; a constant group has one extreme at zero by the range extension, and
/// its value lands on a grid endpoint exactly.
#[test]
fn degenerate_groups_come_back_exactly() {
    for bits in [1u32, 3, 8] {
        let zeros = vec![0.0f64; 32];
        assert!(
            quantize(bits, &zeros).iter().all(|&o| o == 0.0),
            "bits {bits}: an all-zero group did not come back zero"
        );
        for c in [-0.317f64, 0.05] {
            let v = vec![c; 32];
            let out = quantize(bits, &v);
            assert!(
                out.iter().all(|&o| (o - c).abs() <= 1e-15 * c.abs()),
                "bits {bits}: a constant group at {c} came back {:?}",
                &out[..3]
            );
        }
    }
}

/// 🚨 The one deliberate divergence from upstream, and why it is there.
///
/// `xmin = −1e308, xmax = 1e308` overflows `xmax − xmin` to infinity. Upstream
/// then computes `0.0 * inf` and returns a group of NaN, silently. Found by
/// probing the neighbourhood of a surviving mutant rather than by the mutant
/// itself, which is why it is written down.
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

/// An exhaustive nearest-level search over the enumerated grid agrees —
/// including where the clamp bites, since clamping to `[0, maxq]` *is*
/// "restrict to the levels that exist".
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
/// reporting a different width would make `GptqConfig::block` and the rate
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
