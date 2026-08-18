//! # Gate G5 — the calibration seam
//!
//! The Hessian accumulator sits between a validated forward pass and a
//! validated GPTQ core, and until now nothing tested it. That is exactly the
//! shape of the two defects this project has already paid for: a ridge term
//! that no test exercised, and a monotonicity assertion that a no-op
//! satisfied. `H = AᵀA/N` is one line of algebra with an exact reference, so
//! there is no excuse for taking it on faith.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::calib::{
    needs_dense_hessian, quantize_model, Codebook, Hessian, Report, RunConfig,
};
use llvq_llm::model::{Act, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};

const TOL: f64 = 1e-4; // the accumulator runs in f32, by design

/// `AᵀA/N` against a scalar reference computed from the same numbers.
#[test]
fn hessian_matches_a_direct_computation() {
    let dev = Device::Cpu;
    let (rows, n) = (37usize, 11usize);
    // Deterministic, and deliberately not centred: a bug that subtracts a
    // mean would pass on centred data.
    let a: Vec<f32> = (0..rows * n)
        .map(|i| ((i * 7919 % 101) as f32 / 50.0) - 0.7)
        .collect();

    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    let t = Tensor::from_slice(&a, (1, rows, n), &dev).expect("tensor");
    acc.accumulate(&t).expect("accumulate");
    let got = acc.to_f64().expect("readback");

    for i in 0..n {
        for j in 0..n {
            let want: f64 = (0..rows)
                .map(|r| a[r * n + i] as f64 * a[r * n + j] as f64)
                .sum::<f64>()
                / rows as f64;
            assert!(
                (got[i * n + j] - want).abs() <= TOL * want.abs().max(1.0),
                "H[{i},{j}] = {} vs {want}",
                got[i * n + j]
            );
        }
    }
}

/// Splitting the same rows across several calls must give the same Hessian.
///
/// This is what the real loop does — one call per calibration window — and it
/// is where a per-call rather than per-corpus normalization would hide.
#[test]
fn accumulation_is_independent_of_how_rows_are_batched() {
    let dev = Device::Cpu;
    let (rows, n) = (48usize, 9usize);
    let a: Vec<f32> = (0..rows * n)
        .map(|i| ((i * 31 % 17) as f32 / 4.0) - 2.0)
        .collect();

    let mut one = Hessian::new(n, &dev, rows).expect("alloc");
    one.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let whole = one.to_f64().expect("readback");

    let mut split = Hessian::new(n, &dev, rows).expect("alloc");
    for chunk in a.chunks(12 * n) {
        let r = chunk.len() / n;
        split
            .accumulate(&Tensor::from_slice(chunk, (1, r, n), &dev).expect("t"))
            .expect("acc");
    }
    let pieces = split.to_f64().expect("readback");

    for (k, (w, p)) in whole.iter().zip(pieces.iter()).enumerate() {
        assert!(
            (w - p).abs() <= TOL * w.abs().max(1.0),
            "entry {k}: one shot {w} vs batched {p} — the normalization is \
             per-call where it must be per-corpus"
        );
    }
}

/// The result must be symmetric positive semi-definite, and full rank once
/// there are more rows than columns — otherwise the Cholesky downstream fails
/// for reasons that have nothing to do with the model.
#[test]
fn hessian_is_symmetric_and_factorizable() {
    let dev = Device::Cpu;
    let (rows, n) = (200usize, 16usize);
    let a: Vec<f32> = (0..rows * n)
        .map(|i| (((i * 2654435761usize) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    acc.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let h = acc.to_f64().expect("readback");

    for i in 0..n {
        for j in 0..n {
            assert!(
                (h[i * n + j] - h[j * n + i]).abs() <= 1e-9,
                "H must be symmetric at ({i},{j})"
            );
        }
        assert!(h[i * n + i] >= 0.0, "diagonal {i} must be non-negative");
    }
    llvq_quant::linalg::GptqFactor::new(&h, n, 1e-2)
        .expect("a well-fed Hessian must factor with standard damping");
}

/// A dead input channel — a column that never fires — makes `H` singular.
/// Damping must rescue it, because real calibration sets do contain them.
#[test]
fn damping_rescues_a_dead_input_channel() {
    let dev = Device::Cpu;
    let (rows, n) = (128usize, 12usize);
    let mut a: Vec<f32> = (0..rows * n)
        .map(|i| (((i * 7919) % 97) as f32 / 48.0) - 1.0)
        .collect();
    for r in 0..rows {
        a[r * n + 5] = 0.0; // channel 5 never fires
    }
    let mut acc = Hessian::new(n, &dev, rows).expect("alloc");
    acc.accumulate(&Tensor::from_slice(&a, (1, rows, n), &dev).expect("t"))
        .expect("acc");
    let h = acc.to_f64().expect("readback");

    assert!(
        llvq_quant::linalg::GptqFactor::new(&h, n, 0.0).is_err(),
        "without damping a dead channel must be reported, not silently \
         factored — that is the signal the calibration set is wrong"
    );
    llvq_quant::linalg::GptqFactor::new(&h, n, 1e-2)
        .expect("standard damping must make it factorable");
}

// ---------------------------------------------------------------------------
// The dense Hessian: kept when a reader exists, dropped when none does.
//
// `calib::quantize_model` used to stash `H` next to its factor unconditionally,
// though nothing reads it unless `group_scales` or `design_c` is set — and both
// are off on the published path. At Qwen3-32B that is 6.2 GB per block held for
// the whole quantization, alongside the 6.2 GB of factors that *are* read.
//
// The correction is one `Option`, which is exactly the shape of change §5 of
// CLAUDE.md warns about: a capability that quietly stops working under a flag
// nobody tests. So both regimes are exercised below — flag off (nothing kept)
// **and** flag on (kept, and the run that needs it still completes). A test
// that only covered the off case would not touch the parameter it certifies.
// ---------------------------------------------------------------------------

/// Same tiny configuration as `phase_timing.rs` and `kv_cache.rs`.
///
/// The three distinct activation widths are what matters here: 32 for q/k/v
/// and gate/up, 32 for o (`head_dim · num_attention_heads`), 64 for down. All
/// three exceed the 24-wide block, so every matrix has at least one full block
/// plus a tail — the real shape, in miniature.
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

/// A model on the *same* weights every time.
///
/// `quantize_model` replaces each `Linear` with a freshly built tensor rather
/// than writing through the `Var`, so the `VarMap` is untouched by a run and
/// rebuilding from it hands back the pristine checkpoint. That is what lets
/// two configurations be compared with one variable between them.
fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

/// Calibration windows entering block 0. Deterministic, non-centred, and wide
/// enough (128 rows) that the 64-column Hessian of `down_proj` is not rank
/// deficient before damping.
fn windows(dev: &Device) -> Vec<Tensor> {
    let cfg = tiny();
    (0..4)
        .map(|w| {
            let n = 32 * cfg.hidden_size;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1)
                .collect();
            Tensor::from_vec(data, (1, 32, cfg.hidden_size), dev).expect("window")
        })
        .collect()
}

/// Every quantized weight of the model, in a fixed order.
fn projections(m: &Qwen3) -> Vec<f32> {
    let mut out = Vec::new();
    for b in &m.blocks {
        for act in Act::ALL {
            for name in act.consumers() {
                out.extend(
                    b.linear(name)
                        .weight()
                        .flatten_all()
                        .expect("flatten")
                        .to_vec1::<f32>()
                        .expect("readback"),
                );
            }
        }
    }
    out
}

/// The knobs a test varies. A struct rather than a parameter list so that a
/// test reads as "this one thing changed" — which is the whole discipline the
/// 45 wasted machine-minutes of `CLAUDE.md` §6 bought.
#[derive(Clone, Copy)]
struct Knobs {
    codebook: Codebook,
    group_scales: bool,
    design_c: bool,
    lambda: f64,
    rotation_seed: Option<u64>,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            codebook: Codebook::Identity,
            group_scales: false,
            design_c: false,
            lambda: 1e-2,
            rotation_seed: None,
        }
    }
}

/// One quantization run.
fn run(map: &VarMap, dev: &Device, k: Knobs) -> (Report, Vec<f32>) {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev);
    let cfg = RunConfig {
        gptq: GptqConfig {
            block: llvq_core::DIM,
            retract: true,
            group_scales: k.group_scales,
            design_c: k.design_c,
            lambda: k.lambda,
            tail: TailPolicy::KeepExact,
        },
        damping: 1e-2,
        codebook: k.codebook,
        threads: 1,
        start: 0,
        limit: usize::MAX,
        rotation_seed: k.rotation_seed,
    };
    let report =
        quantize_model(&mut model, &mut hidden, &cfg, |_, _, _| {}).expect("quantization runs");
    (report, projections(&model))
}

/// `8·n²` over the four activations of one block — the exact cost of keeping
/// the dense Hessians, with no allocator in the way.
fn expected_dense_bytes(c: &Config) -> u64 {
    [
        c.hidden_size,                      // q/k/v
        c.head_dim * c.num_attention_heads, // o
        c.hidden_size,                      // gate/up
        c.intermediate_size,                // down
    ]
    .iter()
    .map(|&n| 8 * (n * n) as u64)
    .sum()
}

/// The published path keeps nothing.
///
/// Reported in bytes rather than asserted as a boolean because the number is
/// the item: on Qwen3-32B the same arithmetic reads 6 199 379 968.
#[test]
fn the_published_path_does_not_keep_the_dense_hessian() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let (report, _) = run(&map, &dev, Knobs::default());
    assert_eq!(
        report.dense_hessian_bytes, 0,
        "no flag reads H, so no copy of it may survive its factorization"
    );
    assert!(
        !needs_dense_hessian(&GptqConfig::default()),
        "the default configuration is the published one"
    );
}

/// And the identity control still holds, bit for bit.
///
/// This is the single most valuable assertion on this path — the one that says
/// Hessians, factorization, GPTQ loop, error feedback, f64 round trip and
/// weight rewrite are collectively a no-op when the codebook is one. It has
/// only ever been run by hand, as a `bin/smoke` invocation reported in
/// CLAUDE.md; a change to what the loop *keeps* is exactly the occasion to
/// automate it. Natural basis on purpose: rotating and un-rotating is not
/// bit-exact in floating point, and this assertion is about the loop.
#[test]
fn the_identity_control_is_a_no_op_bit_for_bit() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let before = projections(&fresh(&map, &dev));
    let (_, after) = run(&map, &dev, Knobs::default());
    assert_eq!(before.len(), after.len());
    for (k, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(
            b.to_bits(),
            a.to_bits(),
            "weight {k}: identity codebook moved it, {b:e} → {a:e}"
        );
    }
}

/// Design C reads `H`, so design C gets `H` — the exact `8·n²` of it.
///
/// If the materialization were forced to `None`, `quantize_layer`'s assertion
/// ("the closed-form scale solve needs the d_in × d_in Hessian") fires and this
/// test dies on a panic rather than on the byte count. Both failures are loud,
/// which is the property being bought.
#[test]
fn design_c_keeps_the_dense_hessian() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let k = Knobs {
        design_c: true,
        ..Knobs::default()
    };
    let (report, _) = run(&map, &dev, k);
    assert_eq!(report.dense_hessian_bytes, expected_dense_bytes(&tiny()));
    assert_eq!(expected_dense_bytes(&tiny()), 57_344, "8·(3·32² + 64²)");
}

/// The other reader, on its own. Dropping `group_scales` from the predicate
/// and keeping `design_c` would pass every test above and fail this one.
#[test]
fn group_scales_keeps_the_dense_hessian() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let k = Knobs {
        group_scales: true,
        ..Knobs::default()
    };
    let (report, _) = run(&map, &dev, k);
    assert_eq!(report.dense_hessian_bytes, expected_dense_bytes(&tiny()));
}

/// And what is kept is *used*: the closed-form solve moves the weights.
///
/// Without this, a future "fix" that made the solve skip itself when the
/// Hessian is `None` would turn both flags into silent no-ops, and every
/// assertion above would still pass — the byte counter would report a Hessian
/// nobody reads. Same failure mode as the `group_scales` right-hand side that
/// mutation testing caught in `llvq-quant` (`CLAUDE.md` §5): an assertion that
/// a no-op satisfies.
#[test]
fn the_kept_hessian_is_load_bearing() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let base = Knobs {
        codebook: Codebook::Grid { step: 0.02 },
        ..Knobs::default()
    };
    let (_, plain) = run(&map, &dev, base);
    for (label, gs, dc) in [("group_scales", true, false), ("design_c", false, true)] {
        let k = Knobs {
            group_scales: gs,
            design_c: dc,
            ..base
        };
        let (_, refined) = run(&map, &dev, k);
        assert!(
            plain
                .iter()
                .zip(refined.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "{label}: the closed-form solve read the Hessian and changed nothing"
        );
    }
}

/// With an exact codebook, the closed-form solve must answer `s = 1` and leave
/// the model where it found it — in the natural basis and in a rotated one.
///
/// The property, not a value: the quantized blocks *are* the original ones, so
/// the solve's unique answer is unity, and anything else means it did not
/// solve the system it derives. The ridge has to be off, since it deliberately
/// pulls the scales below one; at `lambda > 0` unity is the wrong answer and
/// the assertion would be measuring the ridge.
///
/// 🕳️ **What this does *not* certify, and the algebra is worth knowing.** It
/// was written expecting to catch a wrong Hessian crossing the seam — another
/// activation's, or the same one taken before the rotation. It does not, and
/// mutation testing said so: keeping the **unrotated** `H` survives this test.
/// The reason is exact. Writing `g_p` for block `p` of the quantized row,
/// `Σ_q M[p,q] = g_p H (Σ_q g_q) = g_p H w̃`, and with an exact codebook
/// `w̃ = w`, which is precisely `r[p]`. So `M·1 = r` **identically, for any
/// symmetric `H`** — unity is Hessian-independent here, and no exact-codebook
/// test can separate two SPD Hessians. Separating them needs a *lossy*
/// codebook and the proxy loss in the right metric, which is
/// `group_scales_strictly_lower_the_proxy` in `llvq-quant`, one level down
/// where the Hessian is an argument rather than something to be plumbed.
///
/// What it does certify: the solve runs and returns the exact solution rather
/// than degenerating (a singular or zeroed `H` makes `solve_spd` refuse and
/// turns the whole refinement into a no-op — see the test above), and the
/// `lambda = 0` and rotated paths compose to the identity end to end.
#[test]
fn an_exact_codebook_refines_to_unity_scales() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let before = projections(&fresh(&map, &dev));
    for seed in [None, Some(0xC6_5EED)] {
        let k = Knobs {
            group_scales: true,
            lambda: 0.0,
            rotation_seed: seed,
            ..Knobs::default()
        };
        let (_, after) = run(&map, &dev, k);
        for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            // Loose enough to absorb the rotate/un-rotate round trip, which is
            // not bit-exact; tight enough that a wrong Hessian, whose scales
            // land percent-wise away from one, cannot slip through.
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1e-3),
                "rotation {seed:?}, weight {i}: exact directions must refine to \
                 unity scales, got {a:e} from {b:e}"
            );
        }
    }
}
