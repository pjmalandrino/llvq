//! The fenced, phase-attributed decode (`Qwen3::generate_phased`), pinned.
//!
//! This path is diagnostics-only, but a diagnostic that lies is worse than
//! none: an attribution table with a fence missing between two phases charges
//! one phase's kernels to the next, and every conclusion drawn from it is
//! wrong in a way no reader can detect. So the tests here are about the two
//! properties the timings themselves cannot certify:
//!
//!   1. the instrumented decode says exactly what the published decode says
//!      (same tokens — otherwise it profiles a different computation);
//!   2. every phase boundary really issues a device fence (counted, because
//!      on a fast host a missing sync changes no timing that a threshold
//!      could catch).
//!
//! On the fence count as a mutation detector: `model::fence` is two lines,
//! `synchronize` then `count += 1`, on purpose. Dropping a `fence` call at
//! any boundary changes the count and fails the test below. The residual
//! mutant — keeping the count but stripping the `synchronize` *inside* the
//! helper — is undetectable on the CPU backend, where synchronize is a no-op
//! anyway; the pairing inside the two-line helper is what the count
//! certifies, and that reasoning is documented at the helper.

use candle_core::{DType, Device};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::model::{time_phases_enabled, NoCapture, PhaseReport, Qwen3};

/// Same tiny configuration as `kv_cache.rs`: grouped-query attention, two
/// layers, a real head_dim — everything the phased loop walks through.
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

/// The phased decode must be the published decode with fences in it — not a
/// reimplementation that can drift. Token-for-token equality is the same
/// discipline as `kv_cache.rs`: a discrete decision cannot move unless the
/// computation changed.
#[test]
fn the_fences_do_not_change_what_the_model_says() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    for prompt in [vec![7u32], vec![3, 1, 4, 1, 5]] {
        for n_new in [1usize, 2, 7] {
            let plain = m.generate(&prompt, n_new, &mut NoCapture).expect("plain");
            let (phased, _) = m.generate_phased(&prompt, n_new).expect("phased");
            assert_eq!(
                phased,
                plain,
                "prompt {}, n_new {n_new}: the instrumented decode profiles a \
                 different computation",
                prompt.len()
            );
        }
    }
}

/// Every phase boundary is fenced, by count.
///
/// A decode of `n` tokens issues: one fence closing the prefill, then per
/// token a fence after lm_head and one after argmax, plus — for every token
/// but the last — one after embed and one after blocks+norm. That is
/// `1 + 2n + 2(n − 1) = 4n − 1`, and each phase's sample count follows.
/// Omitting the sync at any single boundary breaks one of these equalities.
#[test]
fn every_phase_boundary_is_fenced() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    for n_new in [1usize, 2, 5] {
        let (tokens, r) = m.generate_phased(&[3, 1, 4], n_new).expect("phased");
        assert_eq!(tokens.len(), n_new);
        assert_eq!(r.fences, 4 * n_new - 1, "n_new {n_new}: fence count");
        assert_eq!(r.head_ms.len(), n_new, "n_new {n_new}: lm_head samples");
        assert_eq!(r.rest_ms.len(), n_new, "n_new {n_new}: argmax samples");
        assert_eq!(r.embed_ms.len(), n_new - 1, "n_new {n_new}: embed samples");
        assert_eq!(r.blocks_ms.len(), n_new - 1, "n_new {n_new}: block samples");
    }
}

/// `max_new = 0` returns empty and fences nothing — same edge the published
/// `generate` had to learn the hard way (see its comment).
#[test]
fn zero_tokens_means_zero_fences() {
    let dev = Device::Cpu;
    let (m, _map) = model(&dev);
    let (tokens, r) = m.generate_phased(&[1, 2, 3], 0).expect("phased");
    assert!(tokens.is_empty());
    assert_eq!(r.fences, 0);
}

/// The env gate: exactly `"1"` and nothing else. `bin/fusedrun`'s published
/// output must be byte-identical whenever this returns false, so a gate that
/// drifted to "any non-empty value" would silently change the protocol for
/// users exporting unrelated values.
#[test]
fn the_gate_accepts_exactly_one() {
    assert!(time_phases_enabled(Some("1")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("yes"), Some("2")] {
        assert!(!time_phases_enabled(v), "gate opened on {v:?}");
    }
}

/// Median and range: odd count takes the middle sample, even count the mean
/// of the two central ones, and the range is `[min, max]` — an implementation
/// that forgot to sort, or swapped the bounds, fails on the shuffled input.
#[test]
fn stats_are_median_and_range() {
    let (med, lo, hi) = PhaseReport::stats(&[5.0, 1.0, 3.0]);
    assert_eq!((med, lo, hi), (3.0, 1.0, 5.0));
    let (med, lo, hi) = PhaseReport::stats(&[4.0, 1.0, 2.0, 8.0]);
    assert_eq!((med, lo, hi), (3.0, 1.0, 8.0));
    let (med, lo, hi) = PhaseReport::stats(&[2.5]);
    assert_eq!((med, lo, hi), (2.5, 2.5, 2.5));
}
