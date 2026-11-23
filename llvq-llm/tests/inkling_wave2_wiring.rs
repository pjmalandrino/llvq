//! Wave 2 wiring of `docs/plan-debit-inkling2-mini.md` (lot 4, agent X): the
//! fused glue kernels inside `Inkling::forward`, and the int4 head chosen at
//! load.
//!
//! The tiny model is the one of `tests/inkling_wave1.rs` (12 layers, loop of
//! 6 to 11 run three times, window 4, extent 6), rebuilt here from the same
//! fixed generator. On it:
//!   - the fused glue tracks the candle glue step by step after a prompt,
//!     within f32 rounding (class B: its sums run in another order), on a
//!     short and a long prompt; a ring filled from the wrong prompt state is
//!     off by far more;
//!   - with the fused glue, a mark taken while the rings hold the states, a
//!     prompt that hands them back, a rollback and a clone give the logits,
//!     bit for bit, of the same run without the detour;
//!   - the fused glue is refused off Metal.
//!
//! Tiny tests are not heavy jobs. The int4 head test loads the real model
//! and is `#[ignore]`d: run it under the guard's lock.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::ExpertSource;
use llvq_llm::inkling::{GlueKind, Inkling, InklingCache, LayerExperts};
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
        "inkling_wave2_{}_{:?}",
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
        put(p("attn.wq_du.weight"), g.tensor(&[nh * hd, h], wa));
        put(p("attn.q_l2_clip.scale"), g.tensor(&[nh, hd], 1.0));
        put(p("attn.wk_dv.weight"), g.tensor(&[nkv * hd, h], wa));
        put(p("attn.wv_dv.weight"), g.tensor(&[nkv * hd, h], wa));
        for which in ["k", "v"] {
            if sliding {
                put(p(&format!("attn.{which}_sconv.weight")), g.tensor(&[nkv * hd, 1, k], 0.5));
            } else {
                put(p(&format!("attn.{which}_sconv.static_w")), g.tensor(&[k, nkv * hd], 0.5));
                put(p(&format!("attn.{which}_sconv.weight_proj")), g.tensor(&[nkv * k, h], wa));
            }
        }
        put(p("attn.gate_attn.weight"), g.tensor(&[nh * hd, h], wa));
        put(p("attn.wo_ud.weight"), g.tensor(&[h, nh * hd], 1.0 / ((nh * hd) as f32).sqrt()));
        put(p("attn.attn_sink_proj.weight"), g.tensor(&[nh, h], wa));
        put(p("attn.wr_du.weight"), g.tensor(&[nh * cfg.d_rel, h], wa));
        put(p("attn.rel_l2_clip.scale"), g.tensor(&[nh, cfg.d_rel], 1.0));
        put(p("attn.rel_logits_proj.proj"), g.tensor(&[cfg.d_rel, ext], 1.0));
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

fn tiny_model(dev: &Device) -> Inkling {
    let (cfg_path, ck_path, cfg) = write_tiny();
    let mut m = Inkling::load(&cfg_path, &ck_path, dev).unwrap();
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
    let _ = std::fs::remove_dir_all(cfg_path.parent().unwrap());
    m
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

/// The fused glue against the candle glue on the same tokens: the largest
/// gap of each row over the largest logit, and the greedy tokens of each arm.
fn fused_against_candle(m: &mut Inkling, prompt: &[u32]) -> (Vec<f32>, usize) {
    m.set_glue(GlueKind::Candle).unwrap();
    let (want, fed) = rows_fed(m, prompt, None);
    m.set_glue(GlueKind::Fused).unwrap();
    let (got, _) = rows_fed(m, prompt, Some(&fed));
    m.set_glue(GlueKind::Candle).unwrap();
    let gaps: Vec<f32> = got
        .iter()
        .zip(&want)
        .map(|(a, b)| {
            let scale = b.iter().fold(0f32, |s, x| s.max(x.abs()));
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
        })
        .collect();
    let same_argmax = got.iter().zip(&want).filter(|(a, b)| argmax(a) == argmax(b)).count();
    (gaps, same_argmax)
}

/// The fused glue's steps after a prompt track the candle glue's: the
/// prefill row is the same computation (several rows run candle's glue), so
/// bit for bit; every step after it within 1e-4 of the largest logit. A ring
/// filled from the wrong state moves a step by far more than that.
#[test]
fn fused_steps_track_candle_steps_metal() {
    let dev = metal();
    let mut m = tiny_model(&dev);
    for p in prompts() {
        let (gaps, same) = fused_against_candle(&mut m, &p);
        println!("prompt {}: relative gaps {gaps:?}; argmax equal {same} of {}", p.len(), gaps.len());
        assert_eq!(gaps[0], 0.0, "prompt {}: the prefill row is not the candle glue's", p.len());
        let worst = gaps.iter().copied().fold(0f32, f32::max);
        assert!(worst < 1e-4, "prompt {}: a fused step is off by {worst:e} of the largest logit", p.len());
        assert!(gaps[1..].iter().any(|&g| g > 0.0), "prompt {}: no step differs at all: the fused glue did not run", p.len());
    }
}

/// A prompt of one token runs the fused glue from empty slots (zero rings).
#[test]
fn fused_from_an_empty_cache_metal() {
    let dev = metal();
    let mut m = tiny_model(&dev);
    let pr = prompts();
    let (gaps, _) = fused_against_candle(&mut m, &pr[0][..1]);
    println!("one-token prompt: relative gaps {gaps:?}");
    let worst = gaps.iter().copied().fold(0f32, f32::max);
    assert!(worst < 1e-4, "one-token prompt: off by {worst:e}");
    assert!(gaps[0] > 0.0, "the one-row prefill did not run the fused glue");
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

/// With the fused glue on both arms: prefill(a) and a fused step (the
/// rings now hold the states), mark, prefill(x) (the rings hand back), a
/// fused step (new rings), rollback, then every position of b and 4 fused
/// steps, against the same without x; a clone taken while the rings hold the
/// states gives the same rows; a rollback straight into fused steps (rings
/// refilled from the mark) gives the rows of steps that never left.
#[test]
fn fused_rings_mark_rollback_clone_metal() {
    let dev = metal();
    let mut m = tiny_model(&dev);
    m.set_glue(GlueKind::Fused).unwrap();
    let p = &prompts()[1];
    let (a, b) = (&p[..30], &p[30..55]);
    let x: Vec<u32> = p[55..].iter().rev().copied().collect();

    let mut c1 = CausalLm::new_cache(&m);
    CausalLm::prefill(&m, a, &mut c1).unwrap();
    CausalLm::step(&m, 9, a.len(), &mut c1).unwrap();
    assert!(c1.slots().iter().all(|s| s.has_rings()), "a fused step left a slot without rings");
    let mark = CausalLm::mark(&m, &c1);
    CausalLm::prefill(&m, &x, &mut c1).unwrap();
    assert!(c1.slots().iter().all(|s| !s.has_rings()), "a prompt ran over rings it did not settle");
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
    assert_eq!(differing, 0, "{differing} of {} rows differ after the rollback", want.len());
    assert_eq!(all_positions_then_steps(&m, b, &mut c3, 4), want, "a clone shares state with its original");

    // rollback straight into fused steps
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
    assert_eq!(differing, 0, "{differing} of 5 fused steps differ after a rollback into rings");
}

/// With the routing handed to the MoE on the device (the default) a source
/// that has no such path, as these dense experts, is called through the
/// routing read back: the logits are those of the switch off, bit for bit.
#[test]
fn routing_on_device_falls_back_to_the_host_metal() {
    let dev = metal();
    let mut m = tiny_model(&dev);
    m.set_glue(GlueKind::Fused).unwrap();
    assert!(m.routing_on_device(), "on by default");
    for p in prompts() {
        let (on, fed) = rows_fed(&m, &p, None);
        m.set_routing_on_device(false);
        let (off, _) = rows_fed(&m, &p, Some(&fed));
        m.set_routing_on_device(true);
        assert_eq!(on, off, "prompt {}: routing on the device against read back", p.len());
    }
}

#[test]
fn fused_glue_is_refused_on_the_cpu() {
    let mut m = tiny_model(&Device::Cpu);
    assert!(m.set_glue(GlueKind::Fused).is_err(), "the fused glue was accepted on the CPU");
    assert_eq!(m.glue(), GlueKind::Candle);
}

// ---- the int4 head chosen at load, on the real model ----

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini")
}

fn need(p: PathBuf) -> PathBuf {
    assert!(p.exists(), "missing input {}: the gate cannot run without it", p.display());
    p
}

fn window(n: usize) -> Vec<u32> {
    let p = need(data_dir().join("lot2-d/wikitext_w0_2048.json"));
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    let ids: Vec<u32> = j["ids"].as_array().expect("ids").iter().map(|v| v.as_u64().unwrap() as u32).collect();
    ids[..n].to_vec()
}

fn prompt32() -> Vec<u32> {
    let p = need(data_dir().join("lot2-u/prompt32.json"));
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    j["prompt"].as_array().expect("prompt").iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

type Calls = std::sync::Arc<std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>>;

/// The model's own head behind a recorder: what it was asked and answered.
struct Recorder {
    inner: Box<dyn llvq_llm::inkling::head::OutputProjection>,
    calls: Calls,
}

impl llvq_llm::inkling::head::OutputProjection for Recorder {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    fn device_bytes(&self) -> u64 {
        self.inner.device_bytes()
    }
    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let y = self.inner.project(h)?;
        self.calls.lock().unwrap().push((row(h), row(&y)));
        Ok(y)
    }
}

/// D4 through `Inkling::load_with`: the model loaded with the int4 table
/// holds no BF16 rows; on the 32- and 600-token prompts, prefill and 16
/// greedy steps, every logits row is the softcap of what its head answered,
/// bit for bit, and that answer is the plain f32 product of the same table
/// decoded: argmax equal, largest relative gap under 1e-5 (K's gate).
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn int4_head_chosen_at_load() {
    let dev = metal();
    let d = data_dir();
    let table = need(d.join("lot1-b/tables/unembed.q4g64.llvqraw"));
    let opts = llvq_llm::inkling::LoadOptions { int4_head: Some(table.clone()) };
    let mut m = Inkling::load_with(
        &need(d.join("textonly/config.json")),
        &need(d.join("native/model-00001-of-00001.safetensors")),
        &dev,
        &opts,
    )
    .expect("load");
    assert!(!m.has_bf16_head(), "the BF16 rows were loaded beside the int4 head");
    let rt = llvq_llm::inkling::resident_runtime(&dev, llvq_llm::inkling::DEFAULT_PREFILL_ROWS).unwrap();
    let resident = m.make_resident(&need(d.join("layers")), rt).expect("resident");
    assert_eq!(resident.len(), 24, "24 resident layers");
    let calls: Calls = Default::default();
    let inner = m.out_proj.take().expect("the int4 head is set at load");
    m.out_proj = Some(Box::new(Recorder { inner, calls: calls.clone() }));
    let cap = m.cfg.final_logit_softcapping;
    let (mut softcap_bits_equal, mut rows) = (0usize, 0usize);
    for ids in [prompt32(), window(600)] {
        let mut cache = m.new_cache();
        let mut l = m.forward(&ids, &mut cache, false, false).unwrap().logits;
        for s in 0..=STEPS {
            let (_, y) = calls.lock().unwrap().last().cloned().expect("one head call a forward");
            let yt = Tensor::from_vec(y, (1, l.dim(1).unwrap()), &dev).unwrap();
            let want = ((yt / cap).unwrap().tanh().unwrap() * cap).unwrap();
            softcap_bits_equal += usize::from(bits(&want) == bits(&l));
            rows += 1;
            if s == STEPS {
                break;
            }
            let next = argmax(&row(&l));
            l = m.forward(&[next], &mut cache, false, false).unwrap().logits;
        }
    }
    let calls = std::mem::take(&mut *calls.lock().unwrap());
    assert_eq!(calls.len(), rows, "one head call a forward");
    let raw = llvq_llm::inkseal::read_raw_file(&table).expect("table");
    let (vocab, hidden) = (m.cfg.unpadded_vocab_size, m.cfg.hidden_size);
    let w = raw.to_f32();
    let w = Tensor::from_slice(&w[..vocab * hidden], (vocab, hidden), &Device::Cpu).unwrap();
    let (mut worst, mut agree) = (0f64, 0usize);
    for (h, y) in &calls {
        let x = Tensor::from_slice(h, (1, hidden), &Device::Cpu).unwrap();
        let plain = row(&x.matmul(&w.t().unwrap()).unwrap());
        let scale = plain.iter().fold(0f64, |s, v| s.max(v.abs() as f64));
        let gap = y.iter().zip(&plain).map(|(a, b)| (a - b).abs() as f64).fold(0f64, f64::max) / scale;
        worst = worst.max(gap);
        agree += usize::from(argmax(y) == argmax(&plain));
    }
    let j = serde_json::json!({"rows": rows, "softcap_bits_equal": softcap_bits_equal,
        "argmax_equal": agree, "worst_relative_gap": worst});
    println!("int4 head chosen at load: {j}");
    let out = d.join("lot4-x/runs/int4_head_chosen_at_load.json");
    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    assert_eq!(softcap_bits_equal, rows, "a logits row is not the softcap of its head's answer");
    assert_eq!(agree, rows, "argmax against the plain product");
    assert!(worst < 1e-5, "relative gap {worst:.3e}");
}
