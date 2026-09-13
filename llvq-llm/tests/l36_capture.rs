//! # L36 — the capture-only pass
//!
//! The instrument of `docs/ROADMAP-QUALITY.md`: one pass over the calibration
//! set with the weights left alone, keeping the dense Hessian the encoder
//! throws away. It yields zero MMLU points by construction and decides eight
//! rows of the table, so the only thing that can go wrong with it is that it
//! captures the wrong matrix — and a dump of the wrong matrix reads exactly
//! like a dump of the right one.
//!
//! Hence what is asserted here. Not "the pass runs": that the `H` it hands to
//! a sink is the `H` the encoder would have factored, in the basis it says it
//! is in, with the shrink applied where the encoder applies it. Five mutants
//! were run against them and all five die here; none would show in a dump.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::calib::{
    capture_model_hessians, effective_rotation_seed, CaptureConfig, CapturedHessian, HBasis,
    Hessian, HessianSink,
};
use llvq_llm::model::{Act, Capture, Qwen3};
use llvq_quant::rotation::Rotation;
use std::collections::HashMap;

/// The accumulator runs in f32 and the readback widens to f64: two paths that
/// perform the *same* operations in the *same* order must agree exactly, and
/// the tests that compare two such paths use 0.0. This tolerance is for the
/// one comparison that reorders arithmetic — the rotation.
const TOL: f64 = 1e-9;

fn tiny() -> Config {
    Config {
        vocab_size: 128,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

/// Four calibration windows of 32 rows. Deterministic and not centred: a bug
/// that subtracted a mean would pass on centred data.
fn windows(dev: &Device) -> Vec<Tensor> {
    let c = tiny();
    (0..4)
        .map(|w| {
            let v: Vec<f32> = (0..32 * c.hidden_size)
                .map(|i| (((i + w * 31) * 7919 % 101) as f32 / 50.0) - 0.7)
                .collect();
            Tensor::from_slice(&v, (1, 32, c.hidden_size), dev).expect("window")
        })
        .collect()
}

/// One emission, flattened: block, activation, basis, declared seed, width, `H`.
type Seen = (usize, Act, HBasis, Option<u64>, usize, Vec<f64>);

/// Everything a sink was handed, kept whole so a test can interrogate it.
#[derive(Default)]
struct Recorder {
    seen: Vec<Seen>,
    means: Vec<Option<Vec<f64>>>,
    norms: Vec<Option<Vec<f32>>>,
}

impl HessianSink for Recorder {
    fn push_hessian(&mut self, c: CapturedHessian<'_>) -> anyhow::Result<()> {
        self.seen
            .push((c.block, c.act, c.basis, c.rotation_seed, c.n, c.h.to_vec()));
        self.means.push(c.mean.map(|m| m.to_vec()));
        self.norms.push(c.token_norms.map(|n| n.to_vec()));
        Ok(())
    }
}

/// Collects one block's Hessians the long way, for the tests to compare against.
struct Ref {
    target: usize,
    acc: HashMap<Act, Hessian>,
}

impl Capture for Ref {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == self.target {
            if let Some(h) = self.acc.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

/// `map` is threaded in and never created here: `VarBuilder::from_varmap`
/// initializes a missing tensor at random, so a helper that made its own
/// `VarMap` would hand every caller a *different* model. Two paths compared
/// entry for entry have to be two paths over one set of weights.
fn capture(map: &VarMap, cfg: CaptureConfig) -> Recorder {
    let dev = Device::Cpu;
    let model = fresh(map, &dev);
    let mut hidden = windows(&dev);
    let mut rec = Recorder::default();
    capture_model_hessians(&model, &mut hidden, &cfg, &mut rec, |_, _| {})
        .expect("the capture pass runs");
    rec
}

fn plain() -> CaptureConfig {
    CaptureConfig {
        h_shrink: 1.0,
        rotation_seed: None,
        emit_natural: false,
    }
}

/// The reference `H` of block 0, accumulated by hand over the same windows.
fn reference_block0(dev: &Device, map: &VarMap) -> HashMap<Act, Vec<f64>> {
    let model = fresh(map, dev);
    let hidden = windows(dev);
    let total_rows: usize = hidden.iter().map(|h| h.dim(1).unwrap()).sum();
    let mut r = Ref {
        target: 0,
        acc: HashMap::new(),
    };
    for act in Act::ALL {
        let w = act.width(model.config());
        r.acc
            .insert(act, Hessian::new(w, dev, total_rows).expect("alloc"));
    }
    let mask = model.causal_mask_for(&hidden[0]).expect("mask");
    for h in hidden.iter() {
        model.blocks[0]
            .forward(h, model.rotary(), &mask, 0, &mut r)
            .expect("forward");
    }
    r.acc
        .into_iter()
        .map(|(a, h)| (a, h.to_f64().expect("readback")))
        .collect()
}

/// **The test the instrument exists for.** What the sink receives for block 0
/// is the matrix the encoder's pass 1 accumulates, entry for entry.
///
/// Exact equality and not a tolerance: both sides run the same f32 GEMM over
/// the same rows in the same order, so any difference at all is a different
/// computation, not rounding.
#[test]
fn the_captured_hessian_is_the_one_the_encoder_would_factor() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let want = reference_block0(&dev, &map);
    let rec = capture(&map, plain());

    for act in Act::ALL {
        let (_, _, basis, seed, n, got) = rec
            .seen
            .iter()
            .find(|(b, a, ..)| *b == 0 && *a == act)
            .unwrap_or_else(|| panic!("block 0 emitted nothing for {act:?}"));
        assert_eq!(*basis, HBasis::Natural, "{act:?}: unrotated run, unrotated H");
        assert_eq!(*seed, None, "{act:?}: no rotation, no seed");
        assert_eq!(*n, act.width(&tiny()), "{act:?}: width");
        let w = &want[&act];
        assert_eq!(got.len(), w.len(), "{act:?}: size");
        for (i, (g, r)) in got.iter().zip(w).enumerate() {
            assert_eq!(g, r, "{act:?}[{i}]: {g} against the reference {r}");
        }
    }
}

/// One emission per activation per block, and no block is skipped.
#[test]
fn every_block_and_every_activation_is_emitted_exactly_once() {
    let rec = capture(&VarMap::new(), plain());
    let n = tiny().num_hidden_layers;
    assert_eq!(rec.seen.len(), n * Act::ALL.len(), "emission count");
    for b in 0..n {
        for act in Act::ALL {
            let k = rec
                .seen
                .iter()
                .filter(|(bb, aa, ..)| *bb == b && *aa == act)
                .count();
            assert_eq!(k, 1, "block {b}, {act:?}: emitted {k} times");
        }
    }
}

/// The rotated emission is `Q H Qᵀ` for the seed it declares — and the seed it
/// declares is the one the encoder derives, not the base seed.
///
/// A mutation that passes the base seed straight through instead of
/// [`effective_rotation_seed`] produces a plausible symmetric matrix with the
/// same trace. Only this assertion sees it.
#[test]
fn the_rotated_emission_is_the_encoders_basis_and_names_its_seed() {
    const BASE: u64 = 0xa5a5_1234;
    let dev = Device::Cpu;
    let map = VarMap::new();
    let natural = reference_block0(&dev, &map);
    let rec = capture(
        &map,
        CaptureConfig {
            rotation_seed: Some(BASE),
            ..plain()
        },
    );

    for act in Act::ALL {
        let (_, _, basis, seed, n, got) = rec
            .seen
            .iter()
            .find(|(b, a, ..)| *b == 0 && *a == act)
            .expect("block 0 emission");
        assert_eq!(*basis, HBasis::Rotated);
        let want_seed = effective_rotation_seed(BASE, 0, act);
        assert_eq!(*seed, Some(want_seed), "{act:?}: declared seed");

        let mut want = natural[&act].clone();
        Rotation::new(*n, want_seed).rotate_hessian(&mut want);
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= TOL * r.abs().max(1.0),
                "{act:?}[{i}]: {g} against Q H Qᵀ = {r}"
            );
        }
    }
}

/// The shrink is applied **before** the rotation, where the encoder applies it.
///
/// `shrink(Q H Qᵀ)` and `Q shrink(H) Qᵀ` are different matrices — the first
/// shrinks off-diagonals of the rotated matrix, the second rotates an already
/// shrunk one — and both are symmetric with a plausible diagonal. Swapping the
/// two lines in the capture loop is the mutation this kills.
#[test]
fn the_shrink_lands_before_the_rotation() {
    const BASE: u64 = 0x51_1234;
    const RHO: f64 = 0.5;
    let dev = Device::Cpu;
    let map = VarMap::new();
    let natural = reference_block0(&dev, &map);
    let rec = capture(
        &map,
        CaptureConfig {
            h_shrink: RHO,
            rotation_seed: Some(BASE),
            emit_natural: false,
        },
    );

    for act in Act::ALL {
        let (_, _, _, _, n, got) = rec
            .seen
            .iter()
            .find(|(b, a, ..)| *b == 0 && *a == act)
            .expect("block 0 emission");
        let mut correct = natural[&act].clone();
        llvq_llm::calib::shrink_off_diagonal(&mut correct, *n, RHO);
        Rotation::new(*n, effective_rotation_seed(BASE, 0, act)).rotate_hessian(&mut correct);

        let mut swapped = natural[&act].clone();
        Rotation::new(*n, effective_rotation_seed(BASE, 0, act)).rotate_hessian(&mut swapped);
        llvq_llm::calib::shrink_off_diagonal(&mut swapped, *n, RHO);

        let d_correct: f64 = got.iter().zip(&correct).map(|(a, b)| (a - b).abs()).sum();
        let d_swapped: f64 = got.iter().zip(&swapped).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            d_correct <= TOL * correct.len() as f64,
            "{act:?}: the capture does not match shrink-then-rotate ({d_correct})"
        );
        assert!(
            d_swapped > 1e-6,
            "{act:?}: the two orders coincide here, so this test proves nothing \
             — pick a rho and a width that separate them"
        );
    }
}

/// `emit_natural` gives both bases, tagged apart, for the same activation.
#[test]
fn both_bases_can_be_emitted_and_they_are_distinguishable() {
    let rec = capture(
        &VarMap::new(),
        CaptureConfig {
            rotation_seed: Some(7),
            emit_natural: true,
            ..plain()
        },
    );
    assert_eq!(rec.seen.len(), tiny().num_hidden_layers * Act::ALL.len() * 2);
    let nat: Vec<_> = rec
        .seen
        .iter()
        .filter(|(b, a, ba, ..)| *b == 0 && *a == Act::Attn && *ba == HBasis::Natural)
        .collect();
    let rot: Vec<_> = rec
        .seen
        .iter()
        .filter(|(b, a, ba, ..)| *b == 0 && *a == Act::Attn && *ba == HBasis::Rotated)
        .collect();
    assert_eq!((nat.len(), rot.len()), (1, 1));
    assert_eq!(nat[0].3, None, "a natural emission carries no seed");
    assert!(rot[0].3.is_some(), "a rotated emission must name its seed");
    assert_ne!(nat[0].5, rot[0].5, "the two bases must differ");
    // The trace is rotation invariant: a cheap check that the rotated matrix
    // is a rotation of *this* matrix and not of some other one.
    let n = nat[0].4;
    let tr = |h: &Vec<f64>| (0..n).map(|i| h[i * n + i]).sum::<f64>();
    let (a, b) = (tr(&nat[0].5), tr(&rot[0].5));
    assert!((a - b).abs() <= TOL * a.abs().max(1.0), "trace {a} against {b}");
}

/// The first moment is captured, and it is `E[x]` — not a column sum of `H`,
/// which is what a wrong implementation would reach for.
///
/// L25's whole row depends on this: `b = ΔW·E[x]` is not derivable from
/// `AᵀA/N`, so if this is wrong the row is silently undecided.
#[test]
fn the_first_moment_is_captured_and_is_not_derivable_from_the_hessian() {
    let rec = capture(&VarMap::new(), plain());
    let i = rec
        .seen
        .iter()
        .position(|(b, a, ..)| *b == 0 && *a == Act::Attn)
        .expect("block 0 attn");
    let mean = rec.means[i].as_ref().expect("E[x] must be captured");
    let (n, h) = (rec.seen[i].4, &rec.seen[i].5);
    assert_eq!(mean.len(), n);
    assert!(
        mean.iter().any(|v| v.abs() > 1e-6),
        "the windows are not centred, so E[x] must not be zero"
    );
    // Were `mean` a row sum of H it would be a quadratic in x; it is linear.
    let row0: f64 = (0..n).map(|j| h[j]).sum();
    assert!(
        (mean[0] - row0).abs() > 1e-9,
        "E[x][0] coincides with a row sum of H: the moment is not being captured"
    );

    // Jensen: `‖E[x]‖² ≤ E[‖x‖²] = tr(H)`. It is the one invariant that ties
    // the first moment to the second on the same scale, and it is what catches
    // an accumulator that forgot its `1/N` — `Σx` is `N` times too large and
    // breaks the bound by `N²`, while passing every shape and sign check.
    let trace: f64 = (0..n).map(|j| h[j * n + j]).sum();
    let m2: f64 = mean.iter().map(|v| v * v).sum();
    assert!(
        m2 <= trace * (1.0 + 1e-6),
        "‖E[x]‖² = {m2} exceeds tr(H) = {trace}: the moment is not normalized by N"
    );
}

/// One `‖x‖²` per calibration row, in stream order, matching the rows fed in.
#[test]
fn the_per_token_norms_cover_every_calibration_row() {
    let dev = Device::Cpu;
    let rows: usize = windows(&dev).iter().map(|h| h.dim(1).unwrap()).sum();
    let rec = capture(&VarMap::new(), plain());
    let i = rec
        .seen
        .iter()
        .position(|(b, a, ..)| *b == 0 && *a == Act::Attn)
        .expect("block 0 attn");
    let norms = rec.norms[i].as_ref().expect("token norms must be captured");
    assert_eq!(norms.len(), rows, "one norm per calibration row");

    // The identity L05 actually reads, and the only assertion here with teeth:
    // `tr(H) = Σᵢ (1/N) Σₜ xₜ[i]² = (1/N) Σₜ ‖xₜ‖²`. A capture that accumulated
    // `Σx` instead of `Σx²` is non-negative, non-zero and the right length —
    // it passes every shape check and fails this one.
    let (n, h) = (rec.seen[i].4, &rec.seen[i].5);
    let trace: f64 = (0..n).map(|j| h[j * n + j]).sum();
    let from_norms: f64 = norms.iter().map(|v| *v as f64).sum::<f64>() / rows as f64;
    assert!(
        (from_norms - trace).abs() <= 1e-5 * trace.abs().max(1.0),
        "Σ‖x‖²/N = {from_norms} against tr(H) = {trace}: these are not squared norms"
    );
    assert!(trace > 0.0, "a trace of zero would make the identity vacuous");
}

/// The encoding path is untouched: `Hessian::new` carries no moments, so no
/// run can acquire them by accident and change what it writes.
#[test]
fn the_served_accumulator_carries_no_moments() {
    let dev = Device::Cpu;
    let mut h = Hessian::new(8, &dev, 4).expect("alloc");
    let x = Tensor::from_slice(&[1.0f32; 32], (1, 4, 8), &dev).expect("x");
    h.accumulate(&x).expect("accumulate");
    assert!(h.mean_f64().expect("readback").is_none());
    assert!(h.token_norms().is_none());
}

/// A capture advances the hidden states, so block 1 sees block 0's output.
///
/// The pass collapses the encoder's two forwards into one precisely because
/// the weights do not move. If it forgot to write the output back, every
/// block after the first would be calibrated on the embedding.
#[test]
fn the_hidden_states_advance_through_the_blocks() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let model = fresh(&map, &dev);
    let before = windows(&dev);
    let mut hidden = windows(&dev);
    let mut rec = Recorder::default();
    capture_model_hessians(&model, &mut hidden, &plain(), &mut rec, |_, _| {}).expect("runs");

    let d: f32 = (&hidden[0] - &before[0])
        .expect("sub")
        .abs()
        .expect("abs")
        .sum_all()
        .expect("sum")
        .to_scalar()
        .expect("scalar");
    assert!(d > 1e-6, "the hidden states were not advanced: {d}");

    // And the two blocks must see different activations.
    let h0 = &rec
        .seen
        .iter()
        .find(|(b, a, ..)| *b == 0 && *a == Act::Attn)
        .expect("block 0")
        .5;
    let h1 = &rec
        .seen
        .iter()
        .find(|(b, a, ..)| *b == 1 && *a == Act::Attn)
        .expect("block 1")
        .5;
    assert_ne!(h0, h1, "both blocks captured the same H");
}

/// A rho outside the unit interval is refused before a forward pass is spent.
#[test]
fn an_illegal_shrink_is_refused_up_front() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let model = fresh(&map, &dev);
    let mut hidden = windows(&dev);
    let mut rec = Recorder::default();
    let e = capture_model_hessians(
        &model,
        &mut hidden,
        &CaptureConfig {
            h_shrink: 1.5,
            ..plain()
        },
        &mut rec,
        |_, _| {},
    )
    .expect_err("rho = 1.5 must be refused");
    assert!(format!("{e}").contains("1.5"), "the error names the value: {e}");
    assert!(rec.seen.is_empty(), "nothing was captured before the refusal");
}
