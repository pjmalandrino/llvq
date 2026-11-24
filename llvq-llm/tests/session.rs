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
//!
//! The persistent (4E) branch runs only on a card, so its accounting is
//! pinned here through `Session::answer_launches` with a launch that stands
//! in for 4E (`stand_in_launch`): the ids held and the cache lengths after
//! each launch, a stop, a cancel, and a launch that fails after it moved the
//! caches.

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
        // A cut cache feeds like a fresh one fed the same prefix, to rounding.
        // Not bit for bit: the kept positions were computed in a matmul of
        // six rows, the fresh ones in other shapes, and x86-64 Linux rounds
        // those apart by about 1e-6 (CI, 2026-11-07). A stale position puts
        // whole tokens back in the attention and moves the logits by far more
        // than the bound (a mutant that skips the cut fails it).
        let h = session.feed(&qwen, &[9, 9]).unwrap();
        let a = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut fresh = Session::new(&qwen);
        let h = fresh.feed(&qwen, &[1, 2, 9, 9]).unwrap();
        let b = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(d <= 1e-4, "store {store:?}: a truncated cache kept a stale position, max |dlogit| {d:e}");
        assert_eq!(session.reuse_prefix(&[]).unwrap(), 0);
        assert!(session.is_empty());
        // Cut to nothing, every cache holds nothing. Under `Cat` the count is
        // the buffers' length, and `KvCache::len` stays at zero whatever they
        // hold, so the ids alone say nothing about the cache.
        for (l, c) in session.caches().iter().enumerate() {
            let held = c.held().unwrap().map_or(0, |(k, _)| k.dim(2).unwrap());
            assert_eq!(held, 0, "store {store:?}: layer {l} kept {held} positions after a cut to zero");
        }
        let h = session.feed(&qwen, &[1, 2, 9, 9]).unwrap();
        let e = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = e.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(d <= 1e-4, "store {store:?}: a cache cut to zero kept a stale position, max |dlogit| {d:e}");
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

/// Under a config that names no chain, `feed_with` is `feed`: the same
/// hidden states, the same ids held, and `answer_fed` the same answer as
/// `answer`, on both stores.
#[test]
fn feed_with_the_eager_prefill_is_feed() {
    use llvq_llm::session::Fed;
    for store in STORES {
        let (qwen, _map) = model(store);
        let ids: Vec<u32> = vec![3, 9, 27, 17, 51, 41];
        let (mut a, mut b) = (Session::new(&qwen), Session::new(&qwen));
        let ha = a.feed(&qwen, &ids).unwrap();
        let fed = b.feed_with(&qwen, &mut Decoder::eager(), &ids).unwrap();
        let Fed::Hidden(hb) = fed else { panic!("store {store:?}: an eager config fed a first token") };
        let va = ha.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = hb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(va, vb, "store {store:?}: the same hidden states");
        assert_eq!(a.ids(), b.ids());
        let (mut sa, mut sb) = (sampler(0.0, 1), sampler(0.0, 1));
        let ga = Generate { sampler: &mut sa, max_new: 9, stops: &[] };
        let ans_a = a.answer(&qwen, &mut Decoder::eager(), ha, ga, &mut |_| Flow::Continue).unwrap();
        let gb = Generate { sampler: &mut sb, max_new: 9, stops: &[] };
        let ans_b = b.answer_fed(&qwen, &mut Decoder::eager(), Fed::Hidden(hb), gb, &mut |_| Flow::Continue).unwrap();
        assert_eq!(ans_a, ans_b, "store {store:?}");
        assert_eq!(a.ids(), b.ids(), "store {store:?}: the same cache after the answer");
    }
}

/// A first token from the chain is greedy and hands over to 4E: beside the
/// eager decode it is refused by name, and nothing is said. A decoder whose
/// config names no chain refuses the chain prefill itself.
#[test]
fn a_chain_first_token_needs_the_persistent_decode() {
    use llvq_llm::session::Fed;
    let (qwen, _map) = model(KvStore::Grow(16));
    let mut session = Session::new(&qwen);
    session.feed(&qwen, &[1, 2, 3]).unwrap();
    let mut s = sampler(0.0, 0);
    let e = session
        .answer_fed(&qwen, &mut Decoder::eager(), Fed::First(5), Generate { sampler: &mut s, max_new: 4, stops: &[] }, &mut |_| {
            panic!("refused before any token")
        })
        .unwrap_err();
    assert!(e.to_string().contains("hands over to 4E"), "{e}");
    assert_eq!(session.len(), 3, "nothing fed");
    let mut caches = qwen.fresh_caches();
    let e = Decoder::eager().chain_prefill(&qwen, &mut caches, 0, &[1, 2]).unwrap_err();
    assert!(e.to_string().contains("names the eager prefill"), "{e}");
    assert!(!Decoder::eager().has_chain());
}

/// One 4E launch, stood in for on the host. It checks what 4E's `prepare`
/// checks, that every cache holds the position the launch starts at, and
/// grows the buffers to hold the launch as 4E does. It writes nothing in
/// them: the property is the accounting, not the arithmetic. Feeding position
/// `p` yields `100 + p`, or ends the launch on a stop token when `p` is
/// `stop_at`; the tokens come back cut before it, as `continue_greedy` cuts.
fn stand_in_launch(
    caches: &mut [KvCache],
    offset: usize,
    k: usize,
    stop_at: Option<usize>,
) -> anyhow::Result<(Vec<u32>, bool)> {
    for c in caches.iter_mut() {
        anyhow::ensure!(c.len() == offset, "a cache holds {} positions, the launch starts at {offset}", c.len());
        c.reserve(offset + k)?;
    }
    let mut made = Vec::new();
    for p in offset..offset + k {
        if stop_at == Some(p) {
            return Ok((made, true));
        }
        made.push(100 + p as u32);
    }
    Ok((made, false))
}

fn assert_caches_hold_the_ids(s: &Session, what: &str) {
    for (l, c) in s.caches().iter().enumerate() {
        assert_eq!(c.len(), s.len(), "{what}: layer {l} holds {} positions for {} ids", c.len(), s.len());
    }
}

/// The persistent branch over several launches and two turns: each launch
/// starts where the last one stopped, `ids` lists every token fed (the
/// answer but its last token on a cap, all of it on a stop), and every cache
/// holds `ids.len()` positions. An eager feed between the turns appends where
/// the launches left the caches.
#[test]
fn the_persistent_branch_feeds_what_its_launches_wrote() {
    let (qwen, _map) = model(KvStore::Grow(16));
    let mut session = Session::new(&qwen);
    let prompt = [1u32, 2, 3, 4, 5];
    session.feed(&qwen, &prompt).unwrap();

    // Turn 1, cut by its cap: the first token, then launches of 4, 4 and 2.
    let mut log = Vec::new();
    let a = session
        .answer_launches(9, 11, &STOPS, 4, &mut |_| Flow::Continue, |caches, last, offset, k| {
            log.push((offset, last, k));
            stand_in_launch(caches, offset, k, None)
        })
        .unwrap();
    let want: Vec<u32> = std::iter::once(9).chain(105..=114).collect();
    assert_eq!(a.finish, Finish::Length);
    assert_eq!(a.said, want);
    assert_eq!(log, [(5usize, 9u32, 4usize), (9, 108, 4), (13, 112, 2)], "where each launch started, and on what");
    assert_eq!(session.ids(), [&prompt[..], &a.said[..10]].concat().as_slice(), "the last token is not fed");
    assert_caches_hold_the_ids(&session, "turn 1");

    // Between the turns, the eager path appends where 4E stopped.
    session.feed(&qwen, &CLOSE).unwrap();
    session.feed(&qwen, &[1, 40, 41]).unwrap();
    let before = session.ids().to_vec();
    assert_eq!(before.len(), 15 + 2 + 3);
    assert_caches_hold_the_ids(&session, "after the eager feeds");

    // Turn 2, ended by a stop token on the second launch: every token said
    // was fed, the one before the stop included.
    let mut log = Vec::new();
    let mut calls = 0;
    let a = session
        .answer_launches(
            20,
            8,
            &STOPS,
            4,
            &mut |_| {
                calls += 1;
                Flow::Continue
            },
            |caches, last, offset, k| {
                log.push((offset, last, k));
                stand_in_launch(caches, offset, k, Some(25))
            },
        )
        .unwrap();
    let want: Vec<u32> = std::iter::once(20).chain(120..=124).collect();
    assert_eq!(a.finish, Finish::Stop);
    assert_eq!(a.said, want);
    assert_eq!(log, [(20usize, 20u32, 4usize), (24, 123, 3)]);
    assert_eq!(calls, 3, "the first token, the first launch, the stopped launch");
    assert_eq!(session.ids(), [&before[..], &a.said[..]].concat().as_slice());
    assert_caches_hold_the_ids(&session, "turn 2");
}

/// The persistent branch's other ends: a cancel leaves the last token made
/// out of the cache, a stop on a launch's first token feeds only the token
/// before it, a first token that is a stop or a cap of one launches nothing.
#[test]
fn the_persistent_branch_cuts_where_it_stops_or_is_cancelled() {
    let (qwen, _map) = model(KvStore::Grow(16));
    let prompt = [1u32, 2, 3];
    let fresh = || {
        let mut s = Session::new(&qwen);
        s.feed(&qwen, &prompt).unwrap();
        s
    };

    let mut s = fresh();
    let cancel_at_five = &mut |said: &[u32]| match said.len() {
        5 => Flow::Cancel,
        _ => Flow::Continue,
    };
    let a = s
        .answer_launches(9, 20, &STOPS, 4, cancel_at_five, |caches, _, offset, k| stand_in_launch(caches, offset, k, None))
        .unwrap();
    assert_eq!(a.finish, Finish::Cancelled);
    assert_eq!(a.said, [9u32, 103, 104, 105, 106]);
    assert_eq!(s.ids(), [&prompt[..], &a.said[..4]].concat().as_slice(), "cancel: the last token made is not fed");
    assert_caches_hold_the_ids(&s, "cancel");

    let mut s = fresh();
    let a = s
        .answer_launches(9, 20, &STOPS, 4, &mut |_| Flow::Continue, |caches, _, offset, k| {
            stand_in_launch(caches, offset, k, Some(offset))
        })
        .unwrap();
    assert_eq!(a.finish, Finish::Stop);
    assert_eq!(a.said, [9u32]);
    assert_eq!(s.ids(), [1u32, 2, 3, 9].as_slice(), "a stop first in a launch: only the token fed before it");
    assert_caches_hold_the_ids(&s, "stop on the first launch token");

    for (first, max_new, finish) in [(STOPS[0], 20, Finish::Stop), (9, 1, Finish::Length)] {
        let mut s = fresh();
        let a = s
            .answer_launches(first, max_new, &STOPS, 4, &mut |_| Flow::Continue, |_, _, _, _| {
                panic!("first {first}, cap {max_new}: no launch")
            })
            .unwrap();
        assert_eq!(a.finish, finish);
        assert_eq!(a.said.len(), usize::from(finish == Finish::Length));
        assert_eq!(s.ids(), prompt.as_slice(), "first {first}, cap {max_new}: nothing fed");
        assert_caches_hold_the_ids(&s, "no launch");
    }
}

/// A launch that fails after it moved the caches, as a 4E launch of the
/// chain prefill or of the decode does on its watchdog or on a NaN: no id is
/// added and the caches run ahead. The next request that keeps every id held
/// cuts them back, so its first launch starts where the ids stop.
#[test]
fn a_failed_launch_leaves_caches_the_next_request_cuts_back() {
    let (qwen, _map) = model(KvStore::Grow(16));
    let mut session = Session::new(&qwen);
    let prompt: Vec<u32> = (1..=10).collect();
    session.feed(&qwen, &prompt).unwrap();
    let e = session
        .answer_launches(9, 8, &[], 4, &mut |_| Flow::Continue, |caches, _, offset, _| {
            for c in caches.iter_mut() {
                c.reserve(offset + 4)?;
                c.set_len(offset + 2)?;
            }
            anyhow::bail!("a barrier aborted on its watchdog")
        })
        .unwrap_err();
    assert!(e.to_string().contains("watchdog"), "{e}");
    assert_eq!(session.len(), 10, "a failed launch adds no id");
    assert!(session.caches().iter().all(|c| c.len() == 12), "the launch moved the caches");

    // The same conversation one token longer keeps every id held.
    let next: Vec<u32> = prompt.iter().copied().chain([30]).collect();
    assert_eq!(session.reuse_prefix(&next).unwrap(), 10);
    assert_caches_hold_the_ids(&session, "after reuse_prefix");

    // And the cut cache reads like a fresh one fed the same ids, to rounding
    // (see `a_divergent_history_is_cut_back_to_the_common_prefix`).
    let h = session.feed(&qwen, &[30]).unwrap();
    let a = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut fresh = Session::new(&qwen);
    let h = fresh.feed(&qwen, &next).unwrap();
    let b = qwen.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(d <= 1e-4, "the cut cache kept a stale position, max |dlogit| {d:e}");
}
