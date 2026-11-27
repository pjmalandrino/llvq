//! Gate D-i of prereg `proofs/preregistration-inkling-fenetre-2026-11-27.md`:
//! a prompt's sliding layers score, chunk by chunk, only the keys the chunk's
//! rows can see (`layers::set_window_scores`), against the path that scores
//! every key the slot holds and masks the rest.
//!
//! The tiny model is the one of `tests/inkling_wave3_attn.rs` (12 layers,
//! window 4, extent 6, full layers among them), from the same generator. The
//! prompt of 70 tokens is read under several settings that cross the window,
//! the slack and a piece boundary, with bounded slots whose first key held is
//! no longer position 0; then 16 steps are fed the reference's own tokens.
//! Every row of the arm is within 1e-4 of the reference scaled by the
//! reference's largest logit, every argmax is the same, and the narrowed path
//! ran (its chunk counter moved). On Metal under the served attention and
//! glue, and on the CPU under the candle attention, where the steps go through
//! the same narrowed path too.
//!
//! The switch is one per process: the tests take [`LOCK`] so that no two
//! flip it at once. Tiny tests are not heavy jobs.

use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::{self, ExpertSource};
use llvq_llm::inkling::{AttnKind, Inkling, InklingCache, LayerExperts, LoadOptions};
use llvq_llm::lm::CausalLm;

static LOCK: Mutex<()> = Mutex::new(());

/// splitmix64 mapped to `[-a, a)`, as in `tests/inkling_wave3_attn.rs`.
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
        "inkling_window_{}_{:?}",
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

/// The 70-token prompt of `tests/inkling_wave1.rs`.
fn prompt() -> Vec<u32> {
    let mut g = Gen(0x7070);
    let _short: Vec<u64> = (0..8).map(|_| g.next()).collect();
    (0..70).map(|_| (g.next() % 60) as u32).collect()
}

fn row(logits: &Tensor) -> Vec<f32> {
    logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

const STEPS: usize = 16;

/// Every position of the prompt read in pieces, then `STEPS` steps fed
/// `feed` (else this arm's own greedy tokens): the rows and the tokens fed.
fn run(m: &Inkling, p: &[u32], feed: Option<&[u32]>) -> (Vec<Vec<f32>>, Vec<u32>, InklingCache) {
    let mut cache = CausalLm::new_cache(m);
    let logits = m.forward_pieces(p, &mut cache, true).unwrap();
    let mut rows: Vec<Vec<f32>> = (0..p.len()).map(|r| row(&logits.narrow(0, r, 1).unwrap())).collect();
    let mut fed = vec![];
    for i in 0..STEPS {
        let next = feed.map_or_else(|| argmax(rows.last().unwrap()), |f| f[i]);
        fed.push(next);
        rows.push(row(&CausalLm::step(m, next, p.len() + i, &mut cache).unwrap()));
    }
    (rows, fed, cache)
}

/// Settings `(piece, chunk, slack)`: one call over full slots, the runtime's
/// long-prompt settings scaled down, and two that put the chunk and piece
/// boundaries at other phases of the window and the slack.
const SETTINGS: [(usize, usize, Option<usize>); 5] =
    [(0, 0, None), (16, 8, Some(2)), (16, 5, Some(0)), (7, 3, Some(1)), (0, 9, None)];

/// Arm on against arm off on `m`, each setting in turn: the worst relative
/// gap over every row, and the assertions of the gate.
fn arm_tracks_reference(m: &mut Inkling, what: &str) {
    let p = prompt();
    let w = m.cfg.sliding_window_size;
    for (piece, chunk, slack) in SETTINGS {
        m.set_prefill_piece(piece);
        m.set_attn_chunk(chunk);
        m.set_kv_window(slack);
        layers::set_window_scores(false);
        let (want, fed, _) = run(m, &p, None);
        layers::set_window_scores(true);
        let before = layers::window_score_chunks();
        let (got, _, cache) = run(m, &p, Some(&fed));
        let chunks = layers::window_score_chunks() - before;
        layers::set_window_scores(false);
        let mut worst = 0f32;
        let mut same = 0;
        let mut exact = 0;
        for (a, b) in got.iter().zip(&want) {
            let scale = b.iter().fold(0f32, |s, x| s.max(x.abs()));
            let gap = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
            worst = worst.max(gap);
            same += usize::from(argmax(a) == argmax(b));
            exact += usize::from(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        println!(
            "{what} piece {piece} chunk {chunk} slack {slack:?}: worst relative gap {worst:e}; argmax equal {same} \
             of {}; bit for bit {exact}; narrowed chunks {chunks}",
            want.len()
        );
        assert!(worst < 1e-4, "{what} piece {piece} chunk {chunk} slack {slack:?}: a row is off by {worst:e}");
        assert_eq!(same, want.len(), "{what} piece {piece} chunk {chunk} slack {slack:?}: an argmax moved");
        assert!(chunks > 0, "{what}: the narrowed path did not run");
        if slack.is_some() {
            let bounded = m
                .schedule()
                .into_iter()
                .filter(|&(l, pass)| m.cfg.is_sliding(l) && cache.slots()[m.slot(l, pass)].base() > 0)
                .count();
            assert!(bounded > 0, "{what}: no sliding slot dropped a position (window {w})");
        }
    }
}

/// On the CPU, the candle attention and glue: the prompt and the steps both
/// run through `scores`.
#[test]
fn window_scores_track_full_scores_cpu() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = tiny_model(&Device::Cpu, AttnKind::Candle).unwrap();
    arm_tracks_reference(&mut m, "cpu");
}

/// On Metal, the served attention (`AttnKind::FusedScores`: a step goes
/// through the one-row kernel, untouched) and the fused glue.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn window_scores_track_full_scores_metal() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("Metal device");
    let mut m = tiny_model(&dev, AttnKind::FusedScores).unwrap();
    m.set_glue(llvq_llm::inkling::GlueKind::Fused).unwrap();
    arm_tracks_reference(&mut m, "metal");
}

