//! Wave 3 wiring (lot 4, agent A): the attention products of a one-row
//! visit through `inkling::attn_dense` inside `Inkling::forward`, chosen at
//! load by `LoadOptions::attn`.
//!
//! The tiny model is the one of `tests/inkling_wave2_wiring.rs` (12 layers,
//! loop of 6 to 11 run three times, window 4, extent 6), from the same fixed
//! generator, with every `attn.*` tensor stored in BF16 as the real file
//! stores it. On it, under both glues:
//!   - the prompt row is bit for bit the candle attention's (several rows
//!     widen the BF16 matrices to f32 for the call, exactly);
//!   - the fused attention's steps track the candle attention's within f32
//!     rounding (class B), on a short and a long prompt, and from an empty
//!     cache;
//!   - with the fused attention, a mark taken while the key and value rings
//!     hold the states, a prompt that hands them back, a rollback and a clone
//!     give the logits, bit for bit, of the same run without the detour;
//!   - the model loaded with the fused attention holds no f32 copy of the
//!     seven attention matrices;
//!   - the fused attention is refused off Metal.
//!
//! Tiny tests are not heavy jobs.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::ExpertSource;
use llvq_llm::inkling::{AttnKind, GlueKind, Inkling, InklingCache, LayerExperts, LoadOptions};
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
        "inkling_wave3_{}_{:?}",
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

/// The two prompts of `tests/inkling_wave1.rs`: 8 tokens, and 70.
fn prompts() -> Vec<Vec<u32>> {
    let mut g = Gen(0x7070);
    let mut p = |n: usize| (0..n).map(|_| (g.next() % 60) as u32).collect::<Vec<u32>>();
    vec![p(8), p(70)]
}

fn row(logits: &Tensor) -> Vec<f32> {
    logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    row(t).iter().map(|x| x.to_bits()).collect()
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

fn metal() -> Device {
    Device::new_metal(0).expect("Metal device")
}

/// Prefill then `STEPS` steps fed `feed` (the candle arm's greedy tokens when
/// given, else this arm's own): the logits rows and the tokens fed.
fn rows_fed(m: &Inkling, prompt: &[u32], feed: Option<&[u32]>) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut cache = CausalLm::new_cache(m);
    let mut rows = vec![row(&CausalLm::prefill(m, prompt, &mut cache).unwrap())];
    let mut fed = vec![];
    for i in 0..STEPS {
        let next = match feed {
            Some(f) => f[i],
            None => argmax(rows.last().unwrap()),
        };
        fed.push(next);
        rows.push(row(&CausalLm::step(m, next, prompt.len() + i, &mut cache).unwrap()));
    }
    (rows, fed)
}

/// Per row, the largest gap over the largest logit of `want`.
fn gaps(got: &[Vec<f32>], want: &[Vec<f32>]) -> Vec<f32> {
    got.iter()
        .zip(want)
        .map(|(a, b)| {
            let scale = b.iter().fold(0f32, |s, x| s.max(x.abs()));
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
        })
        .collect()
}

/// The fused attention's steps after a prompt track the candle attention's,
/// under both glues: the prefill row is bit for bit (several rows run
/// candle's attention on matrices widened exactly), every step within 1e-4 of
/// the largest logit, and some step not bit for bit (the kernels ran).
#[test]
fn fused_attn_tracks_candle_attn_metal() {
    let dev = metal();
    let mut cand = tiny_model(&dev, AttnKind::Candle).unwrap();
    let mut fused = tiny_model(&dev, AttnKind::Fused).unwrap();
    assert_eq!(fused.attn(), AttnKind::Fused);
    for glue in [GlueKind::Candle, GlueKind::Fused] {
        cand.set_glue(glue).unwrap();
        fused.set_glue(glue).unwrap();
        for p in prompts() {
            let (want, fed) = rows_fed(&cand, &p, None);
            let (got, _) = rows_fed(&fused, &p, Some(&fed));
            let g = gaps(&got, &want);
            let same = got.iter().zip(&want).filter(|(a, b)| argmax(a) == argmax(b)).count();
            println!("glue {} prompt {}: relative gaps {g:?}; argmax equal {same} of {}", glue.name(), p.len(), g.len());
            assert_eq!(g[0], 0.0, "glue {} prompt {}: the prefill row is not the candle attention's", glue.name(), p.len());
            let worst = g.iter().copied().fold(0f32, f32::max);
            assert!(worst < 1e-4, "glue {} prompt {}: a step is off by {worst:e}", glue.name(), p.len());
            assert!(g[1..].iter().any(|&x| x > 0.0), "glue {} prompt {}: no step differs: the kernels did not run", glue.name(), p.len());
        }
    }
}

/// A prompt of one token runs the fused attention from empty slots (zero rings).
#[test]
fn fused_attn_from_an_empty_cache_metal() {
    let dev = metal();
    let cand = tiny_model(&dev, AttnKind::Candle).unwrap();
    let fused = tiny_model(&dev, AttnKind::Fused).unwrap();
    let p = &prompts()[0][..1];
    let (want, fed) = rows_fed(&cand, p, None);
    let (got, _) = rows_fed(&fused, p, Some(&fed));
    let g = gaps(&got, &want);
    println!("one-token prompt: relative gaps {g:?}");
    let worst = g.iter().copied().fold(0f32, f32::max);
    assert!(worst < 1e-4, "one-token prompt: off by {worst:e}");
    assert!(g[0] > 0.0, "the one-row prefill did not run the fused attention");
}

/// Every position of `ids` on `cache`, then `steps` greedy steps: the bits of each row.
fn all_positions_then_steps(m: &Inkling, ids: &[u32], cache: &mut InklingCache, steps: usize) -> Vec<Vec<u32>> {
    let out = m.forward(ids, cache, false, true).unwrap();
    let n = out.logits.dim(0).unwrap();
    let mut rows: Vec<Vec<u32>> = (0..n).map(|r| bits(&out.logits.narrow(0, r, 1).unwrap())).collect();
    let mut last = row(&out.logits.narrow(0, n - 1, 1).unwrap());
    for _ in 0..steps {
        let pos = cache.len();
        let l = CausalLm::step(m, argmax(&last), pos, cache).unwrap();
        rows.push(bits(&l));
        last = row(&l);
    }
    rows
}

/// `n` greedy steps from `last`: the bits of each row.
fn steps_from(m: &Inkling, cache: &mut InklingCache, mut last: Vec<f32>, n: usize) -> Vec<Vec<u32>> {
    let mut rows = vec![];
    for _ in 0..n {
        let pos = cache.len();
        let l = CausalLm::step(m, argmax(&last), pos, cache).unwrap();
        rows.push(bits(&l));
        last = row(&l);
    }
    rows
}

/// With the fused attention, under both glues: prefill(a) and a step (the key
/// and value rings now hold the states), mark, prefill(x) (the rings hand
/// back), a step (new rings), rollback, then every position of b and 4 steps,
/// against the same without x; a clone taken while the rings hold the states
/// gives the same rows; a rollback straight into steps (rings refilled from
/// the mark) gives the rows of steps that never left.
#[test]
fn fused_attn_rings_mark_rollback_clone_metal() {
    let dev = metal();
    let mut m = tiny_model(&dev, AttnKind::Fused).unwrap();
    for glue in [GlueKind::Candle, GlueKind::Fused] {
        m.set_glue(glue).unwrap();
        let p = &prompts()[1];
        let (a, b) = (&p[..30], &p[30..55]);
        let x: Vec<u32> = p[55..].iter().rev().copied().collect();

        let mut c1 = CausalLm::new_cache(&m);
        CausalLm::prefill(&m, a, &mut c1).unwrap();
        CausalLm::step(&m, 9, a.len(), &mut c1).unwrap();
        assert!(c1.slots().iter().all(|s| s.has_attn_rings()), "a step left a slot without key and value rings");
        let mark = CausalLm::mark(&m, &c1);
        CausalLm::prefill(&m, &x, &mut c1).unwrap();
        assert!(c1.slots().iter().all(|s| !s.has_attn_rings()), "a prompt ran over rings it did not settle");
        let pos = c1.len();
        CausalLm::step(&m, 7, pos, &mut c1).unwrap();
        CausalLm::rollback(&m, &mut c1, &mark).unwrap();
        let after = all_positions_then_steps(&m, b, &mut c1, 4);

        let mut c2 = CausalLm::new_cache(&m);
        CausalLm::prefill(&m, a, &mut c2).unwrap();
        CausalLm::step(&m, 9, a.len(), &mut c2).unwrap();
        let mut c3 = c2.clone();
        let want = all_positions_then_steps(&m, b, &mut c2, 4);
        let differing = after.iter().zip(&want).filter(|(u, v)| u != v).count();
        println!("glue {}: after rollback {differing} of {} rows differ", glue.name(), want.len());
        assert_eq!(differing, 0, "glue {}: {differing} of {} rows differ after the rollback", glue.name(), want.len());
        assert_eq!(all_positions_then_steps(&m, b, &mut c3, 4), want, "glue {}: a clone shares state", glue.name());

        let mut c4 = CausalLm::new_cache(&m);
        let l0 = row(&CausalLm::prefill(&m, a, &mut c4).unwrap());
        let l1 = row(&CausalLm::step(&m, argmax(&l0), a.len(), &mut c4).unwrap());
        let mark4 = CausalLm::mark(&m, &c4);
        steps_from(&m, &mut c4, l1.clone(), 3);
        CausalLm::rollback(&m, &mut c4, &mark4).unwrap();
        let back = steps_from(&m, &mut c4, l1.clone(), 5);
        let mut c5 = CausalLm::new_cache(&m);
        CausalLm::prefill(&m, a, &mut c5).unwrap();
        CausalLm::step(&m, argmax(&l0), a.len(), &mut c5).unwrap();
        let straight = steps_from(&m, &mut c5, l1, 5);
        let differing = back.iter().zip(&straight).filter(|(u, v)| u != v).count();
        assert_eq!(differing, 0, "glue {}: {differing} of 5 steps differ after a rollback into rings", glue.name());
    }
}

/// The fused attention holds the seven matrices once, in BF16: no f32 copy
/// of them on the device; the candle attention holds them in f32.
#[test]
fn fused_attn_holds_no_f32_matrices_metal() {
    let dev = metal();
    let cand = tiny_model(&dev, AttnKind::Candle).unwrap();
    let fused = tiny_model(&dev, AttnKind::Fused).unwrap();
    let (cf, cb) = cand.attention_matrix_bytes();
    let (ff, fb) = fused.attention_matrix_bytes();
    println!("attention matrices: candle {cf} B f32 + {cb} B BF16; fused {ff} B f32 + {fb} B BF16");
    assert!(cf > 0 && cb == 0, "candle attention: {cf} B f32, {cb} B BF16");
    assert_eq!(ff, 0, "the fused attention kept {ff} B of f32 matrices");
    assert_eq!(2 * fb, cf, "the BF16 copy is not half the f32 one");
}

#[test]
fn fused_attn_is_refused_on_the_cpu() {
    assert!(tiny_model(&Device::Cpu, AttnKind::Fused).is_err(), "the fused attention was accepted on the CPU");
}
