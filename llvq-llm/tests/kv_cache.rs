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

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::model::{NoCapture, Qwen3};

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
    let m = Qwen3::new(&tiny(), vb).expect("tiny model builds");
    (m, map)
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
