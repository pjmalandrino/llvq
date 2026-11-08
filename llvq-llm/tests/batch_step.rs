//! A batched step returns each sequence's own decode: B prompts of different
//! lengths, fed together through `Qwen3::generate_batch` with the candle
//! reference attention, against `Qwen3::generate` on each prompt alone.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;
use llvq_core::SplitMix64;
use llvq_llm::batch::{BatchProjs, CandleBatch};
use llvq_llm::model::{NoCapture, Qwen3};

fn cfg() -> Config {
    Config {
        vocab_size: 128,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn model(dev: &Device) -> Qwen3 {
    let c = cfg();
    let mut rng = SplitMix64::new(0xBA7C);
    let mut t = HashMap::new();
    let mut put = |name: String, r: usize, k: usize| {
        let v: Vec<f32> = (0..r * k).map(|_| (rng.next_gaussian() * 0.4) as f32).collect();
        t.insert(name, Tensor::from_vec(v, (r, k), dev).expect("tensor"));
    };
    let (h, i, nh, nkv, hd) = (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
    put("model.embed_tokens.weight".into(), c.vocab_size, h);
    for l in 0..c.num_hidden_layers {
        let p = format!("model.layers.{l}");
        put(format!("{p}.self_attn.q_proj.weight"), nh * hd, h);
        put(format!("{p}.self_attn.k_proj.weight"), nkv * hd, h);
        put(format!("{p}.self_attn.v_proj.weight"), nkv * hd, h);
        put(format!("{p}.self_attn.o_proj.weight"), h, nh * hd);
        put(format!("{p}.mlp.gate_proj.weight"), i, h);
        put(format!("{p}.mlp.up_proj.weight"), i, h);
        put(format!("{p}.mlp.down_proj.weight"), h, i);
    }
    let mut ones = |name: String, n: usize| {
        t.insert(name, Tensor::ones(n, DType::F32, dev).expect("ones"));
    };
    for l in 0..c.num_hidden_layers {
        let p = format!("model.layers.{l}");
        ones(format!("{p}.self_attn.q_norm.weight"), hd);
        ones(format!("{p}.self_attn.k_norm.weight"), hd);
        ones(format!("{p}.input_layernorm.weight"), h);
        ones(format!("{p}.post_attention_layernorm.weight"), h);
    }
    ones("model.norm.weight".into(), h);
    let vb = VarBuilder::from_tensors(t, DType::F32, dev);
    Qwen3::new(&c, vb, llvq_llm::kvq::KvMode::F16).expect("tiny model")
}

#[test]
fn a_batch_decodes_each_sequence_as_it_decodes_alone() {
    let dev = Device::Cpu;
    let m = model(&dev);
    let prompts: Vec<Vec<u32>> = vec![vec![5, 17, 99, 3, 40], vec![77], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], vec![64, 64]];
    let alone: Vec<Vec<u32>> = prompts.iter().map(|p| m.generate(p, 20, &mut NoCapture).expect("generate")).collect();
    let batch = m.generate_batch(&prompts, 20, &CandleBatch, BatchProjs::Rows, false, &|_, _| false).expect("batch");
    assert_eq!(batch, alone);
    let pre = m.generate_batch(&prompts, 20, &CandleBatch, BatchProjs::Rows, true, &|_, _| false).expect("prefilled batch");
    assert_eq!(pre, alone, "prefilled through the batch-1 path");
    // Distinct sequences, so a batch that mixed rows up could not pass.
    assert!(alone.windows(2).all(|w| w[0] != w[1]), "{alone:?}");
}

#[test]
fn a_sequence_that_stops_leaves_the_others_untouched() {
    let dev = Device::Cpu;
    let m = model(&dev);
    let prompts: Vec<Vec<u32>> = vec![vec![9, 8, 7], vec![30, 31]];
    let alone: Vec<Vec<u32>> = prompts.iter().map(|p| m.generate(p, 12, &mut NoCapture).expect("generate")).collect();
    let batch = m.generate_batch(&prompts, 12, &CandleBatch, BatchProjs::Rows, true, &|i, g| i == 0 && g.len() == 4).expect("batch");
    assert_eq!(batch[0], alone[0][..4].to_vec());
    assert_eq!(batch[1], alone[1]);
}

#[test]
fn dense_projections_take_their_linear_in_the_tensor_core_arm() {
    let dev = Device::Cpu;
    let m = model(&dev);
    let e = m.generate_batch(&[vec![1, 2]], 2, &CandleBatch, BatchProjs::Mma, false, &|_, _| false);
    assert!(e.is_ok(), "dense projections take their Linear in either arm: {e:?}");
}

#[test]
fn the_batched_window_score_is_the_full_window_score() {
    let dev = Device::Cpu;
    let m = model(&dev);
    let w: Vec<Vec<u32>> = vec![(0..24u32).map(|i| (i * 7 + 3) % 128).collect(), (0..24u32).map(|i| (i * 13 + 1) % 128).collect()];
    let refs: Vec<&[u32]> = w.iter().map(Vec::as_slice).collect();
    let got = m.batch_window_nll(&refs, &CandleBatch, BatchProjs::Rows).expect("batched nll");
    for (i, win) in w.iter().enumerate() {
        let (n, c) = m.window_nll(win, &mut NoCapture).expect("window nll");
        assert_eq!(got[i].1, c);
        assert!((got[i].0 - n).abs() < 1e-3 * n.abs().max(1.0), "window {i}: batched {} full {n}", got[i].0);
    }
}

#[test]
fn sequences_that_join_and_leave_a_pool_decode_as_they_decode_alone() {
    // Three slots, five requests: each joins when a slot frees, the others
    // keep stepping, and every answer is the one `generate` gives alone.
    let dev = Device::Cpu;
    let m = model(&dev);
    let reqs: Vec<(Vec<u32>, usize)> = vec![
        (vec![5, 17, 99, 3, 40], 9),
        (vec![77], 4),
        (vec![1, 2, 3, 4, 5, 6, 7, 8], 12),
        (vec![64, 64], 6),
        (vec![9, 8, 7], 7),
    ];
    let alone: Vec<Vec<u32>> = reqs.iter().map(|(p, n)| m.generate(p, *n, &mut NoCapture).expect("generate")).collect();
    let mut pool = llvq_llm::batch::Pool::new(&m, 3, 32).expect("pool");
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); reqs.len()];
    // (request, slot, next token to feed)
    let mut live: Vec<(usize, usize, u32)> = Vec::new();
    let mut next_req = 0usize;
    while next_req < reqs.len() || !live.is_empty() {
        while next_req < reqs.len() && pool.free() > 0 {
            let (slot, first) = pool.admit(&m, &reqs[next_req].0).expect("admit").expect("a free slot");
            out[next_req].push(first);
            live.push((next_req, slot, first));
            next_req += 1;
        }
        live.retain(|&(r, slot, _)| {
            let done = out[r].len() >= reqs[r].1;
            if done {
                pool.release(slot);
            }
            !done
        });
        if live.is_empty() {
            continue;
        }
        let step: Vec<(usize, u32)> = live.iter().map(|&(_, s, t)| (s, t)).collect();
        let picks = pool.step(&m, &step, &CandleBatch, BatchProjs::Rows).expect("step");
        for (l, p) in live.iter_mut().zip(picks) {
            out[l.0].push(p);
            l.2 = p;
        }
    }
    assert_eq!(out, alone);
}
