//! The KV cache, pinned on a model small enough to build in a test.
//!
//! Until `bin/gbench` existed the cache was on no measured path: perplexity
//! and MMLU both go through `hidden()`, which is offset 0 over an empty
//! cache, so a broken cache would have changed no published number. A
//! throughput figure changes that — the cache is now upstream of a number the
//! campaign will print, and it had no automated test at all.
//!
//! What makes a cache bug dangerous is that it does not crash and does not
//! trip a threshold: it shifts a RoPE position or drops a mask entry, and the
//! model goes on producing fluent, plausible, *different* text. So the test
//! is an equality, not a tolerance — and it runs on random weights, because
//! the property is arithmetic and has nothing to do with what the weights
//! mean.
//!
//! Two decode steps minimum. Below that `generate` returns straight after the
//! prefill: `append` never sees a non-empty cache, `causal_mask_offset` never
//! sees a nonzero offset, and the test would compare a code path with itself.
//! That is the failure mode CLAUDE.md §5 documents three times.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_core::SplitMix64;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_llm::rotplan::RotShare;

/// Small enough to build in milliseconds, wide enough to exercise everything
/// that matters: grouped-query attention (4 heads over 2 kv heads), more than
/// one layer, and a head_dim the per-head RMSNorm actually normalises over.
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

fn model(dev: &Device) -> (Qwen3, VarMap) {
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, dev);
    let m = Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds");
    (m, map)
}

/// The same tiny model, with weights that do not depend on a random seed —
/// so a *token sequence* can be pinned rather than a shape.
///
/// `swap_kv` writes `v_proj`'s weights into `k_proj` and vice versa. It exists
/// to prove the golden below is sensitive to exactly the restructuring mistake
/// lot A4 could make: on Qwen3-4B `k_proj` and `v_proj` have the *same*
/// `d_out`, so permuting them inside `forward_cached` type-checks, reshapes
/// cleanly, and changes nothing observable except the answer.
fn deterministic(dev: &Device, swap_kv: bool) -> Qwen3 {
    let cfg = tiny();
    let mut rng = SplitMix64::new(0xA4_C0DE);
    let mut t = HashMap::new();
    let put = |t: &mut HashMap<String, Tensor>, name: &str, r: usize, c: usize, rng: &mut SplitMix64| {
        let v: Vec<f32> = (0..r * c).map(|_| (rng.next_gaussian() * 0.3) as f32).collect();
        t.insert(name.to_string(), Tensor::from_vec(v, (r, c), dev).expect("tensor"));
    };
    let put1 = |t: &mut HashMap<String, Tensor>, name: &str, n: usize| {
        t.insert(
            name.to_string(),
            Tensor::ones(n, DType::F32, dev).expect("ones"),
        );
    };
    let (h, i, nh, nkv, hd) = (
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    put(&mut t, "model.embed_tokens.weight", cfg.vocab_size, h, &mut rng);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        put(&mut t, &format!("{p}.self_attn.q_proj.weight"), nh * hd, h, &mut rng);
        let (kn, vn) = if swap_kv {
            ("v_proj", "k_proj")
        } else {
            ("k_proj", "v_proj")
        };
        put(&mut t, &format!("{p}.self_attn.{kn}.weight"), nkv * hd, h, &mut rng);
        put(&mut t, &format!("{p}.self_attn.{vn}.weight"), nkv * hd, h, &mut rng);
        put(&mut t, &format!("{p}.self_attn.o_proj.weight"), h, nh * hd, &mut rng);
        put(&mut t, &format!("{p}.mlp.gate_proj.weight"), i, h, &mut rng);
        put(&mut t, &format!("{p}.mlp.up_proj.weight"), i, h, &mut rng);
        put(&mut t, &format!("{p}.mlp.down_proj.weight"), h, i, &mut rng);
        put1(&mut t, &format!("{p}.self_attn.q_norm.weight"), hd);
        put1(&mut t, &format!("{p}.self_attn.k_norm.weight"), hd);
        put1(&mut t, &format!("{p}.input_layernorm.weight"), h);
        put1(&mut t, &format!("{p}.post_attention_layernorm.weight"), h);
    }
    put1(&mut t, "model.norm.weight", h);
    let vb = VarBuilder::from_tensors(t, DType::F32, dev);
    Qwen3::new(&cfg, vb, llvq_llm::kvq::KvMode::F16).expect("deterministic model builds")
}

/// **T7.** The restructured block issues the same calls, in the same order.
///
/// Lot A4 rewrote the middle of `Block::forward_cached`: q/k/v and gate/up now
/// go through `model::group_forward` instead of three and two `Proj::forward`
/// calls. On a dense model that must be a *no-op*, because `forward_cached` is
/// the function `bin/oracle` certifies at `max |Δhidden| = 0` and every Hessian
/// in this project is built on it.
///
/// Three assertions, and each covers what the others cannot:
///
///  1. **the golden token sequence** — the only thing that pins the *order* of
///     q, k and v. A permutation of k and v is invisible to every shape in the
///     tiny config (both are `n_kv · head_dim` wide) and to every tolerance;
///  2. **the swap actually moves it** — otherwise the golden would be a
///     constant that happens to be true, and the test would be certifying its
///     own insensitivity. This is the §5 discipline applied to a golden:
///     a fixture that no mutation can move tests nothing;
///  3. **both `RotShare` modes agree**, exactly. On a dense model
///     `group_forward` takes its no-rotation branch and the mode must not
///     reach the arithmetic at all.
///
/// The no-op-ness of `Proj::prepare` on a dense projection is pinned beside it,
/// in `model.rs`'s own test module, where `Rotated`'s private fields are
/// reachable.
#[test]
fn the_restructured_block_is_the_same_calls_in_the_same_order() {
    let dev = Device::Cpu;
    let m = deterministic(&dev, false);
    let prompt = [3u32, 1, 4, 1, 5, 9, 2, 6];

    // (1) The golden. Recompute with `cargo test -- --nocapture` and read the
    // panic message if a legitimate change to the toy config moves it; do not
    // "refresh" it to make a red test green without saying why.
    const GOLDEN: [u32; 8] = [107, 90, 2, 35, 35, 35, 35, 35];
    let got = m.generate(&prompt, GOLDEN.len(), &mut NoCapture).expect("generate");
    assert_eq!(
        got.as_slice(),
        &GOLDEN,
        "la restructuration de forward_cached a changé ce que le modèle dit"
    );

    // (2) The golden is sensitive to a k/v permutation.
    let swapped = deterministic(&dev, true)
        .generate(&prompt, GOLDEN.len(), &mut NoCapture)
        .expect("generate");
    assert_ne!(
        swapped.as_slice(),
        &GOLDEN,
        "échanger k et v ne change pas la sortie : le golden ne prouve rien sur l'ordre"
    );

    // (3) The mode does not reach a dense model.
    for share in [RotShare::Off, RotShare::On] {
        let mut m = deterministic(&dev, false);
        m.set_rot_share(share);
        assert_eq!(
            m.generate(&prompt, GOLDEN.len(), &mut NoCapture).expect("generate"),
            GOLDEN.to_vec(),
            "{share:?} a changé un modèle dense"
        );
    }
}

/// Greedy decoding is deterministic, so the cached and uncached paths must
/// agree on **every token**, not merely on most of them.
///
/// The two paths differ in the shape of every GEMM they issue — `(1,1,h)`
/// against `(1,L,h)` — so their last bits can differ, and an equality on
/// logits would be flaky. An equality on argmax is not: a cache bug that
/// only moved the last bit could not shift a discrete decision either, and
/// one that shifts a position or drops a mask entry changes the tokens
/// immediately.
#[test]
fn the_cache_does_not_change_what_the_model_says() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    for prompt in [vec![7u32], vec![3, 1, 4, 1, 5], (0..20).collect::<Vec<u32>>()] {
        for n_new in [2usize, 3, 7] {
            let cached = m.generate(&prompt, n_new, &mut NoCapture).expect("cached");
            let plain = m
                .generate_uncached(&prompt, n_new, &mut NoCapture)
                .expect("uncached");
            assert_eq!(
                cached.len(),
                n_new,
                "prompt {}, n_new {n_new}: wrong count",
                prompt.len()
            );
            assert_eq!(
                cached,
                plain,
                "prompt {}, n_new {n_new}: the cache changed the answer",
                prompt.len()
            );
        }
    }
}

/// The hidden states themselves, at a nonzero offset.
///
/// `generate` only ever calls `hidden_cached` with `l = 1`, so the `i` term of
/// `j <= offset + i` is exercised by nothing above — a mask that read
/// `j <= offset` would be correct on every path this repository runs today
/// and wrong the moment a prompt is fed in chunks. This feeds one in two
/// chunks and requires the second chunk's hidden states to equal the tail of
/// a single full-sequence pass.
#[test]
fn a_split_prefill_matches_a_whole_one() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    let ids: Vec<u32> = (0..12).collect();
    let (head, tail) = ids.split_at(5);

    let whole = m
        .hidden(
            &Tensor::from_slice(&ids, (1, ids.len()), &dev).unwrap(),
            &mut NoCapture,
        )
        .expect("whole pass");

    let mut caches = m.fresh_caches();
    m.hidden_cached(
        &Tensor::from_slice(head, (1, head.len()), &dev).unwrap(),
        0,
        &mut caches,
        &mut NoCapture,
    )
    .expect("first chunk");
    let second = m
        .hidden_cached(
            &Tensor::from_slice(tail, (1, tail.len()), &dev).unwrap(),
            head.len(),
            &mut caches,
            &mut NoCapture,
        )
        .expect("second chunk");

    let want = whole.narrow(1, head.len(), tail.len()).unwrap();
    let d = (second - want)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    // Not an exact equality: the two runs issue different GEMM shapes, so the
    // last bits legitimately differ. 1e-4 on f32 activations of order 1 is
    // three orders of magnitude below anything a shifted position or a
    // dropped mask entry would produce.
    assert!(d < 1e-4, "split prefill differs from a whole one by {d:e}");
}

/// `max_new = 0` used to run until RoPE walked off the end of its table —
/// 40 960 decode steps and an opaque error — because the stop test sits after
/// the push and `out.len() == 0` is never true. The witness returned `[]`
/// immediately, so the two paths this repository declares equivalent diverged
/// at the very first edge case.
#[test]
fn zero_new_tokens_returns_nothing_on_both_paths() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    assert!(m.generate(&[1, 2, 3], 0, &mut NoCapture).unwrap().is_empty());
    assert!(m
        .generate_uncached(&[1, 2, 3], 0, &mut NoCapture)
        .unwrap()
        .is_empty());
}

// ---------------------------------------------------------------------------
// Le contrôle positif du §3.6 de proofs/preregistration-p3-2026-08-14.md.
// ---------------------------------------------------------------------------

/// Qwen3-4B's KV geometry, which is what the quantizer groups over: 36 layers,
/// 8 KV heads, head_dim 128. Everything else is shrunk — `hidden_size` and
/// `intermediate_size` play no part in how a cache is grouped, and keeping them
/// at 4096 would buy nothing but seconds.
fn kv_geometry_of_4b() -> Config {
    Config {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        head_dim: 128,
        attention_bias: false,
        num_key_value_heads: 8,
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

/// **The gate of P3.** `KvMode::Q8` must change what a block computes.
///
/// Without it, an unwired arm reports Δppl = 0.000 %, ΔMMLU = 0.00 pp and
/// 1.00× throughput — and `LLVQ_VERIFY_CACHE=1` goes green too, since both of
/// its arms would then be f16. Three greens that say nothing. This test is the
/// only thing standing between that and a published "P3 acquis".
///
/// It asserts the *presence* of an effect, not its size: a quantizer that
/// changed nothing would be either unwired or a no-op, and both are the same
/// failure from here.
#[test]
fn q8_actually_changes_what_a_block_computes() {
    let dev = Device::Cpu;
    let cfg = kv_geometry_of_4b();

    // Same weights on both arms — a VarMap built once, read twice.
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    let f16_arm = Qwen3::new(&cfg, vb.clone(), llvq_llm::kvq::KvMode::F16).expect("builds");
    let q8_arm = Qwen3::new(&cfg, vb, llvq_llm::kvq::KvMode::Q8).expect("builds");

    let ids = Tensor::from_vec(vec![3u32, 17, 42, 8, 61, 5], (1, 6), &dev).expect("ids");
    let a = f16_arm
        .logits(&ids, &mut NoCapture)
        .expect("f16 arm runs");
    let b = q8_arm.logits(&ids, &mut NoCapture).expect("q8 arm runs");

    let d = (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
        .expect("same shape")
        .abs()
        .expect("abs")
        .max_all()
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");

    assert!(
        d > 0.0,
        "KvMode::Q8 produced bit-identical logits to F16 on the 4B KV geometry \
         (36 × 8 × 128). The arm is NOT wired: a run in this state would report \
         Δppl = 0, ΔMMLU = 0 and 1.00× throughput, and every one of those greens \
         would be empty. Vérifier les deux sites de construction de KvCache \
         (model.rs : Block::forward et fresh_caches)."
    );
}
