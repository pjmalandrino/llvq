//! `Session`, pinned against the loop `repl.rs` ran before 2026-11-06.
//!
//! The REPL's loop moved into `llvq_llm::session` so the runtime and the REPL
//! run one loop. A move is only a move if it feeds the same ids at the same
//! positions in the same calls, so the old eager loop is copied here verbatim
//! (`old_turn`) and the new one must return its tokens and leave its cache.
//!
//! The second half pins prefix reuse, which the old loop did not have: a
//! conversation replayed from scratch every turn, as a stateless API sends it,
//! must give the ids the stateful conversation gave.
//!
//! Random weights, on the CPU: the properties are arithmetic. Under the `Cat`
//! store and the `Grow` store, the two stores a served config names.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::kvq::{KvMode, KvStore};
use llvq_llm::model::{KvCache, NoCapture, Qwen3};
use llvq_llm::repl::Sampler;
use llvq_llm::served_decode::Decoder;
use llvq_llm::session::{common_prefix, Finish, Flow, Generate, Session};

fn tiny() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 256,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn model(store: KvStore) -> (Qwen3, VarMap) {
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
    let mut m = Qwen3::new(&tiny(), vb, KvMode::F16).expect("tiny model builds");
    m.set_kv_store(store);
    (m, map)
}

const STORES: [KvStore; 2] = [KvStore::Cat, KvStore::Grow(16)];

/// The eager branch of `repl::run_with` as it stood at f6d719b, line for
/// line, minus the printing: prefill the turn in one call, then pick, stop,
/// push, cap, feed. Returns the answer and the new offset after the close.
#[allow(clippy::too_many_arguments)]
fn old_turn(
    qwen: &Qwen3,
    caches: &mut [KvCache],
    mut offset: usize,
    ids: &[u32],
    sampler: &mut Sampler,
    max_new: usize,
    stops: &[u32],
    close: &[u32],
) -> (Vec<u32>, usize) {
    let dev = qwen.device();
    let mut h = qwen
        .hidden_cached(&Tensor::from_slice(ids, (1, ids.len()), dev).unwrap(), offset, caches, &mut NoCapture)
        .unwrap();
    offset += ids.len();
    let mut n = 0usize;
    let mut said = Vec::new();
    loop {
        let logits = qwen.logits_last(&h).unwrap().i((0, 0)).unwrap();
        let next = sampler.pick(&logits).unwrap();
        if stops.contains(&next) {
            break;
        }
        said.push(next);
        n += 1;
        if n >= max_new {
            break;
        }
        h = qwen
            .hidden_cached(&Tensor::from_slice(&[next], (1, 1), dev).unwrap(), offset, caches, &mut NoCapture)
            .unwrap();
        offset += 1;
    }
    qwen.hidden_cached(&Tensor::from_slice(close, (1, close.len()), dev).unwrap(), offset, caches, &mut NoCapture)
        .unwrap();
    offset += close.len();
    (said, offset)
}

/// The next-token logits after feeding `probe` on top of a cache: two caches
/// that hold the same positions give the same row, bit for bit.
fn probe_logits(qwen: &Qwen3, caches: &mut [KvCache], offset: usize, probe: u32) -> Vec<f32> {
    let h = qwen
        .hidden_cached(&Tensor::from_slice(&[probe], (1, 1), qwen.device()).unwrap(), offset, caches, &mut NoCapture)
        .unwrap();
    qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn sampler(temp: f32, seed: u64) -> Sampler {
    Sampler { temp, top_p: 0.9, state: seed }
}

/// A stop token the tiny model actually writes now and then, so both the
/// stop branch and the cap branch run across the turns below.
const STOPS: [u32; 2] = [7, 11];
const CLOSE: [u32; 2] = [3, 4];

#[test]
fn the_session_feeds_what_the_old_repl_loop_fed() {
    for store in STORES {
        for temp in [0.0f32, 0.9] {
            let (qwen, _map) = model(store);
            let turns: [&[u32]; 3] = [&[1, 2, 5, 9, 13], &[1, 21, 22, 23], &[1, 30, 31, 32, 33, 34, 35]];

            let mut old_caches = qwen.fresh_caches();
            let mut old_offset = 0;
            let mut s_old = sampler(temp, 42);

            let mut session = Session::new(&qwen);
            let mut s_new = sampler(temp, 42);
            let mut dec = Decoder::eager();

            let mut finishes = Vec::new();
            for (i, ids) in turns.iter().enumerate() {
                let max_new = 6 + i;
                let (said, off) =
                    old_turn(&qwen, &mut old_caches, old_offset, ids, &mut s_old, max_new, &STOPS, &CLOSE);
                old_offset = off;

                let h = session.feed(&qwen, ids).unwrap();
                let a = session
                    .answer(
                        &qwen,
                        &mut dec,
                        h,
                        Generate { sampler: &mut s_new, max_new, stops: &STOPS },
                        &mut |_| Flow::Continue,
                    )
                    .unwrap();
                session.feed(&qwen, &CLOSE).unwrap();
                assert_eq!(a.said, said, "store {store:?}, temp {temp}, turn {i}: the answers differ");
                assert_eq!(session.len(), old_offset, "store {store:?}, temp {temp}, turn {i}: positions held");
                finishes.push(a.finish);
            }
            // Same cache content, not just the same count: one more position
            // on each side reads the same logits.
            let a = probe_logits(&qwen, &mut old_caches, old_offset, 17);
            let h = session.feed(&qwen, &[17]).unwrap();
            let b = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a, b, "store {store:?}, temp {temp}: the caches hold different positions");
            assert!(finishes.iter().all(|f| *f != Finish::Cancelled));
        }
    }
}

#[test]
fn the_cap_leaves_the_last_token_out_of_the_cache() {
    let (qwen, _map) = model(KvStore::Cat);
    let mut session = Session::new(&qwen);
    let prompt = [1u32, 2, 3];
    let h = session.feed(&qwen, &prompt).unwrap();
    let mut s = sampler(0.0, 0);
    let a = session
        .answer(&qwen, &mut Decoder::eager(), h, Generate { sampler: &mut s, max_new: 4, stops: &[] }, &mut |_| {
            Flow::Continue
        })
        .unwrap();
    assert_eq!(a.finish, Finish::Length);
    assert_eq!(a.said.len(), 4);
    assert_eq!(session.ids(), [&prompt[..], &a.said[..3]].concat().as_slice());
}

#[test]
fn a_cancel_stops_within_one_token_and_keeps_the_cache_consistent() {
    let (qwen, _map) = model(KvStore::Grow(16));
    let mut session = Session::new(&qwen);
    let h = session.feed(&qwen, &[1, 2, 3]).unwrap();
    let mut s = sampler(0.0, 0);
    let mut calls = 0;
    let a = session
        .answer(&qwen, &mut Decoder::eager(), h, Generate { sampler: &mut s, max_new: 50, stops: &[] }, &mut |said| {
            calls += 1;
            match said.len() {
                3 => Flow::Cancel,
                _ => Flow::Continue,
            }
        })
        .unwrap();
    assert_eq!(a.finish, Finish::Cancelled);
    assert_eq!(a.said.len(), 3, "the cancel must land on the token it was asked at");
    assert_eq!(calls, 3);
    // The third token was never fed.
    assert_eq!(session.len(), 3 + 2);
    assert_eq!(&session.ids()[3..], &a.said[..2]);
}

/// Greedy, two turns, stateful: the REPL's way.
fn stateful(qwen: &Qwen3, turns: &[Vec<u32>], max_new: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut session = Session::new(qwen);
    let mut dec = Decoder::eager();
    let mut answers = Vec::new();
    for t in turns {
        let h = session.feed(qwen, t).unwrap();
        let mut s = sampler(0.0, 0);
        let a = session
            .answer(qwen, &mut dec, h, Generate { sampler: &mut s, max_new, stops: &STOPS }, &mut |_| Flow::Continue)
            .unwrap();
        session.feed(qwen, &CLOSE).unwrap();
        answers.push(a.said);
    }
    (answers, session.ids().to_vec())
}

#[test]
fn a_stateless_replay_gives_the_stateful_ids_and_reads_only_the_new_tokens() {
    for store in STORES {
        let (qwen, _map) = model(store);
        let turns = vec![vec![1u32, 2, 5, 9], vec![1u32, 40, 41, 42, 43]];
        let (want, want_ids) = stateful(&qwen, &turns, 8);

        // Stateless: every request is the whole history, as an HTTP client
        // sends it, rebuilt from the ids the cache held after the last turn.
        let mut session = Session::new(&qwen);
        let mut dec = Decoder::eager();
        let mut history: Vec<u32> = Vec::new();
        for (i, t) in turns.iter().enumerate() {
            history.extend_from_slice(t);
            let keep = session.reuse_prefix(&history).unwrap();
            let before = history.len() - keep;
            if i > 0 {
                assert_eq!(keep, history.len() - t.len(), "store {store:?}: turn {i} re-read the history");
            }
            let h = session.feed(&qwen, &history[keep..]).unwrap();
            let mut s = sampler(0.0, 0);
            let a = session
                .answer(&qwen, &mut dec, h, Generate { sampler: &mut s, max_new: 8, stops: &STOPS }, &mut |_| {
                    Flow::Continue
                })
                .unwrap();
            session.feed(&qwen, &CLOSE).unwrap();
            assert_eq!(a.said, want[i], "store {store:?}: turn {i} answered differently ({before} fed)");
            history = session.ids().to_vec();
        }
        assert_eq!(session.ids(), want_ids.as_slice());
    }
}

#[test]
fn a_divergent_history_is_cut_back_to_the_common_prefix() {
    for store in STORES {
        let (qwen, _map) = model(store);
        let mut session = Session::new(&qwen);
        session.feed(&qwen, &[1, 2, 3, 4, 5, 6]).unwrap();
        // The same prompt twice: everything but the last token is kept, so
        // the next token has a hidden state to come from.
        assert_eq!(session.reuse_prefix(&[1, 2, 3, 4, 5, 6]).unwrap(), 5);
        session.feed(&qwen, &[6]).unwrap();
        assert_eq!(session.reuse_prefix(&[1, 2, 9, 9]).unwrap(), 2);
        assert_eq!(session.ids(), &[1, 2]);
        // A cut cache feeds like a fresh one fed the same prefix.
        let h = session.feed(&qwen, &[9, 9]).unwrap();
        let a = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut fresh = Session::new(&qwen);
        let h = fresh.feed(&qwen, &[1, 2, 9, 9]).unwrap();
        let b = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "store {store:?}: a truncated cache kept a stale position");
        assert_eq!(session.reuse_prefix(&[]).unwrap(), 0);
        assert!(session.is_empty());
    }
}

#[test]
fn common_prefix_counts_matching_heads_only() {
    assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_prefix(&[], &[1]), 0);
    assert_eq!(common_prefix(&[5], &[6]), 0);
}

#[test]
fn zero_tokens_asked_answers_nothing_and_feeds_nothing() {
    let (qwen, _map) = model(KvStore::Cat);
    let mut session = Session::new(&qwen);
    let h = session.feed(&qwen, &[1, 2]).unwrap();
    let mut s = sampler(0.0, 0);
    let a = session
        .answer(&qwen, &mut Decoder::eager(), h, Generate { sampler: &mut s, max_new: 0, stops: &[] }, &mut |_| {
            panic!("no token, no callback")
        })
        .unwrap();
    assert!(a.said.is_empty());
    assert_eq!(session.len(), 2);
}
