//! The cache state of `inkling::snapshot`, written and loaded back (plan
//! item P7, the prefix cache of `llvq-runtime`): a prompt read in pieces,
//! cut at a piece boundary, its state written to bytes and loaded into a new
//! cache, then the rest of the prompt and greedy steps, gives every logit of
//! the read without the cut, bit for bit.
//!
//! The tiny model is the one of `tests/inkling_wave3_attn.rs` (12 layers, the
//! loop of 6 to 11 run three times, window 4, extent 6). Every arm runs the
//! long-prompt settings the runtime serves: pieces, chunked scores, bounded
//! sliding slots. On the CPU with candle's attention and glue; on Metal with
//! the served stack (fused glue, `fused+scores` attention), and once with a
//! state written while the rings of a one-row step hold the convolution
//! states.
//!
//! Tiny tests are not heavy jobs.

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::ExpertSource;
use llvq_llm::inkling::{AttnKind, Inkling, InklingCache, LayerExperts, LoadOptions};
use llvq_llm::lm::CausalLm;

/// splitmix64 mapped to `[-a, a)`, as in `tests/inkling_wave1.rs`.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, a: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * u - 1.0) * a
    }

    fn tensor(&mut self, shape: &[usize], a: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.uniform(a)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).expect("host tensor")
    }

    /// The same values stored in BF16, as the real file stores the attention.
    fn bf16(&mut self, shape: &[usize], a: f32) -> Tensor {
        self.tensor(shape, a).to_dtype(DType::BF16).expect("bf16")
    }

    fn norm(&mut self, n: usize) -> Tensor {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + self.uniform(0.1)).collect();
        Tensor::from_vec(v, n, &Device::Cpu).expect("host tensor")
    }
}

fn tiny_config_json() -> serde_json::Value {
    serde_json::json!({
        "hidden_size": 48, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
        "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 8,
        "swa_head_dim": 8, "swa_num_attention_heads": 6, "swa_num_key_value_heads": 2,
        "sliding_window_size": 4, "d_rel": 4, "rel_extent": 6,
        "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.5,
        "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
    })
}

fn write_tiny() -> (PathBuf, PathBuf, InklingConfig) {
    let dir = std::env::temp_dir().join(format!(
        "inkling_prefix_state_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("config.json");
    std::fs::write(&cfg_path, tiny_config_json().to_string()).unwrap();
    let cfg = InklingConfig::from_file(&cfg_path).unwrap();
    let mut g = Gen(0x5eed_1a71);
    let h = cfg.hidden_size;
    let k = cfg.sconv_kernel_size;
    let rank = cfg.loop_config.gate_rank;
    let mut t: Vec<(String, Tensor)> = vec![];
    let mut put = |name: String, x: Tensor| t.push((format!("model.llm.{name}"), x));
    for l in 0..cfg.num_hidden_layers {
        let p = |s: &str| format!("layers.{l}.{s}");
        let sliding = cfg.is_sliding(l);
        let (nh, nkv, hd) = cfg.attn_shape(l);
        let ext = cfg.layer_rel_extent(l);
        let wa = 1.0 / (h as f32).sqrt();
        put(p("attn.wq_du.weight"), g.bf16(&[nh * hd, h], wa));
        put(p("attn.q_l2_clip.scale"), g.bf16(&[nh, hd], 1.0));
        put(p("attn.wk_dv.weight"), g.bf16(&[nkv * hd, h], wa));
        put(p("attn.wv_dv.weight"), g.bf16(&[nkv * hd, h], wa));
        for which in ["k", "v"] {
            if sliding {
                put(p(&format!("attn.{which}_sconv.weight")), g.bf16(&[nkv * hd, 1, k], 0.5));
            } else {
                put(p(&format!("attn.{which}_sconv.static_w")), g.bf16(&[k, nkv * hd], 0.5));
                put(p(&format!("attn.{which}_sconv.weight_proj")), g.bf16(&[nkv * k, h], wa));
            }
        }
        put(p("attn.gate_attn.weight"), g.bf16(&[nh * hd, h], wa));
        put(p("attn.wo_ud.weight"), g.bf16(&[h, nh * hd], 1.0 / ((nh * hd) as f32).sqrt()));
        put(p("attn.attn_sink_proj.weight"), g.bf16(&[nh, h], wa));
        put(p("attn.wr_du.weight"), g.bf16(&[nh * cfg.d_rel, h], wa));
        put(p("attn.rel_l2_clip.scale"), g.bf16(&[nh, cfg.d_rel], 1.0));
        put(p("attn.rel_logits_proj.proj"), g.bf16(&[cfg.d_rel, ext], 1.0));
        if l > 0 {
            put(format!("attn_res.attn_queries.{}", l - 1), g.tensor(&[if sliding { 1 } else { 3 }, h], 1.0));
        }
        if sliding {
            put(p("attn_norm.weight"), g.norm(h));
        } else {
            for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                put(p(&format!("{n}.weight")), g.norm(h));
            }
        }
        if cfg.is_loop_layer(l) {
            for which in ["attn", "k", "v", "mlp"] {
                put(p(&format!("controllers.{which}.down_weight")), g.tensor(&[rank, 2 * h], 0.2));
                put(p(&format!("controllers.{which}.forget_weight")), g.tensor(&[h, rank], 0.3));
                put(p(&format!("controllers.{which}.gain_weight")), g.tensor(&[h, rank], 0.3));
            }
        }
        put(format!("attn_res.mlp_queries.{l}"), g.tensor(&[h], 1.0));
        put(p("mlp_norm.weight"), g.norm(h));
        put(p("attn_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
        put(p("mlp_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
        put(p("mlp.gate.weight"), g.tensor(&[cfg.n_routed_experts, h], wa));
        put(p("mlp.gate.bias"), g.tensor(&[cfg.n_routed_experts], 0.05));
    }
    let n_loop = cfg.loop_config.core_end - cfg.loop_config.core_start;
    put("middle_cycle_cross_pass_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
    put("middle_cycle_cross_pass_kv_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
    put("attn_res.output_query".into(), g.tensor(&[h], 1.0));
    put("norm.weight".into(), g.norm(h));
    put("embed_norm.weight".into(), g.norm(h));
    put("embed.weight".into(), g.tensor(&[cfg.vocab_size, h], 1.0).to_dtype(DType::BF16).unwrap());
    put("unembed.weight".into(), g.tensor(&[cfg.vocab_size, h], 1.0).to_dtype(DType::BF16).unwrap());
    let ck_path = dir.join("model.safetensors");
    let map: std::collections::HashMap<String, Tensor> = t.into_iter().collect();
    candle_core::safetensors::save(&map, &ck_path).unwrap();
    (cfg_path, ck_path, cfg)
}

struct DenseExperts(Vec<(Tensor, Tensor)>);

impl ExpertSource for DenseExperts {
    fn count(&self) -> usize {
        self.0.len()
    }
    fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok(self.0[e].clone())
    }
}

/// The tiny model with the attention `attn`, and the same experts whatever `attn`.
fn tiny_model(dev: &Device, attn: AttnKind) -> anyhow::Result<Inkling> {
    let (cfg_path, ck_path, cfg) = write_tiny();
    let loaded = Inkling::load_with_attn(&cfg_path, &ck_path, dev, &LoadOptions::default(), attn);
    let _ = std::fs::remove_dir_all(cfg_path.parent().unwrap());
    let mut m = loaded?;
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let mut g = Gen(0xe4be_7500);
    for l in 0..cfg.num_hidden_layers {
        let ex = (0..cfg.n_routed_experts)
            .map(|_| {
                let gu = g.tensor(&[2 * i, h], 1.0 / (h as f32).sqrt()).to_device(dev).unwrap();
                let down = g.tensor(&[h, i], 1.0 / (i as f32).sqrt()).to_device(dev).unwrap();
                (gu, down)
            })
            .collect();
        m.set_experts(l, LayerExperts::Source(Box::new(DenseExperts(ex))));
    }
    Ok(m)
}

fn prompt(n: usize) -> Vec<u32> {
    let mut g = Gen(0x7070);
    (0..n).map(|_| (g.next() % 60) as u32).collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn argmax(b: &[u32]) -> u32 {
    let v: Vec<f32> = b.iter().map(|&x| f32::from_bits(x)).collect();
    (0..v.len()).fold(0, |best, i| if v[i] > v[best] { i } else { best }) as u32
}

/// The served long-prompt settings at the tiny model's scale: pieces of 16
/// tokens, scores in chunks of 8 rows, sliding slots bounded with a slack of 2.
const PIECE: usize = 16;

fn long_prompt_settings(m: &mut Inkling) {
    m.set_prefill_piece(PIECE);
    m.set_attn_chunk(8);
    m.set_kv_window(Some(2));
}

/// Every position's logits of `ids` read in pieces on `cache`, as bits.
fn read(m: &Inkling, ids: &[u32], cache: &mut InklingCache) -> Vec<Vec<u32>> {
    let l = m.forward_pieces(ids, cache, true).unwrap();
    (0..l.dim(0).unwrap()).map(|r| bits(&l.narrow(0, r, 1).unwrap())).collect()
}

/// `n` greedy steps after `last`, as bits.
fn steps(m: &Inkling, cache: &mut InklingCache, last: &[u32], n: usize) -> Vec<Vec<u32>> {
    let mut rows = vec![];
    let mut prev = last.to_vec();
    for _ in 0..n {
        let pos = cache.len();
        let l = CausalLm::step(m, argmax(&prev), pos, cache).unwrap();
        prev = bits(&l);
        rows.push(prev.clone());
    }
    rows
}

fn state(c: &InklingCache) -> Vec<u8> {
    let mut v = Vec::new();
    c.write_state(&mut v).unwrap();
    v
}

/// The prompt read whole, against the prompt read to `b`, the state written,
/// loaded into a new cache, then the rest read: every row and 8 steps, bit for
/// bit, and the loaded cache writes the bytes it was loaded from.
fn cut_matches_whole(m: &Inkling, ids: &[u32], b: usize) {
    let mut whole = m.new_cache();
    let mut want = read(m, ids, &mut whole);
    let last = want.last().unwrap().clone();
    want.extend(steps(m, &mut whole, &last, 8));

    let mut first = m.new_cache();
    let mut got = read(m, &ids[..b], &mut first);
    let saved = state(&first);
    drop(first);
    let mut loaded = m.read_cache(&mut saved.as_slice()).unwrap();
    assert_eq!(state(&loaded), saved, "the loaded cache does not write the state it was loaded from");
    assert_eq!(loaded.len(), b);
    assert_eq!(loaded.tokens(), &ids[..b]);
    got.extend(read(m, &ids[b..], &mut loaded));
    let last = got.last().unwrap().clone();
    got.extend(steps(m, &mut loaded, &last, 8));

    let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("cut at {b} of {}: {differing} of {} rows differ, state {} bytes", ids.len(), want.len(), saved.len());
    assert_eq!(got.len(), want.len());
    assert_eq!(differing, 0, "cut at {b}: {differing} of {} rows differ from the read without the cut", want.len());
    let bounded = (0..m.n_slots()).filter(|&i| loaded.slots()[i].is_bounded()).count();
    assert!(bounded > 0, "no bounded slot: the arm does not test the bound");
    assert!(loaded.slots().iter().any(|s| s.base() > 0), "no slot dropped a position before the cut");
}

#[test]
fn a_loaded_state_reads_on_bit_for_bit_cpu() {
    let mut m = tiny_model(&Device::Cpu, AttnKind::Candle).unwrap();
    long_prompt_settings(&mut m);
    let ids = prompt(70);
    for b in [32, 48, 64] {
        cut_matches_whole(&m, &ids, b);
    }
}

/// A state is the model's settings' own: a cache saved with full sliding
/// slots is refused by a model that bounds them, by name; a damaged state is
/// refused and nothing is returned.
#[test]
fn a_state_of_other_settings_or_damaged_is_refused() {
    let mut m = tiny_model(&Device::Cpu, AttnKind::Candle).unwrap();
    m.set_prefill_piece(PIECE);
    let ids = prompt(40);
    let mut c = m.new_cache();
    m.forward_pieces(&ids, &mut c, false).unwrap();
    let full = state(&c);
    m.set_kv_window(Some(2));
    let e = m.read_cache(&mut full.as_slice()).unwrap_err().to_string();
    assert!(e.contains("bound"), "{e}");
    let mut c = m.new_cache();
    m.forward_pieces(&ids, &mut c, false).unwrap();
    let good = state(&c);
    let e = m.read_cache(&mut &good[..good.len() - 3]).unwrap_err().to_string();
    assert!(e.contains("cache state"), "{e}");
    let mut bad = good.clone();
    bad[0] ^= 1;
    let e = m.read_cache(&mut bad.as_slice()).unwrap_err().to_string();
    assert!(e.contains("magic"), "{e}");
    assert!(m.read_cache(&mut good.as_slice()).is_ok());
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal {
    use super::*;
    use llvq_llm::inkling::GlueKind;

    fn served(dev: &Device) -> Inkling {
        let mut m = tiny_model(dev, AttnKind::FusedScores).unwrap();
        m.set_glue(GlueKind::Fused).unwrap();
        long_prompt_settings(&mut m);
        m
    }

    #[test]
    fn a_loaded_state_reads_on_bit_for_bit_metal() {
        let dev = Device::new_metal(0).unwrap();
        let m = served(&dev);
        let ids = prompt(70);
        for b in [32, 48] {
            cut_matches_whole(&m, &ids, b);
        }
    }

    /// A state written while the fused glue's and the fused attention's rings
    /// hold the convolution states (after one-row steps), loaded into a new
    /// cache: the next steps are those of the cache that never left, bit for bit.
    #[test]
    fn a_state_written_over_live_rings_metal() {
        let dev = Device::new_metal(0).unwrap();
        let m = served(&dev);
        let ids = prompt(40);
        let mut c = m.new_cache();
        let rows = read(&m, &ids, &mut c);
        let after = steps(&m, &mut c, rows.last().unwrap(), 3);
        assert!(c.slots().iter().all(|s| s.has_attn_rings()), "a step left a slot without key and value rings");
        assert!(c.slots().iter().all(|s| s.has_rings()), "a step left a slot without the glue's rings");
        let saved = state(&c);
        let mut loaded = m.read_cache(&mut saved.as_slice()).unwrap();
        assert!(loaded.slots().iter().all(|s| !s.has_rings() && !s.has_attn_rings()));
        let last = after.last().unwrap();
        let want = steps(&m, &mut c, last, 6);
        let got = steps(&m, &mut loaded, last, 6);
        let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("state over live rings: {differing} of 6 steps differ");
        assert_eq!(differing, 0, "{differing} of 6 steps differ after a state written over live rings");
    }
}
