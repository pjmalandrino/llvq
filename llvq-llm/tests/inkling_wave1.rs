//! Wave 1 of `docs/plan-debit-inkling2-mini.md`, gated on a tiny whole model.
//!
//! The tiny model is a real `Inkling`: a 12-layer config with the loop of
//! layers 6 to 11 run three times, a sliding window of 4 and a relative extent
//! of 6, written as a native-named safetensors file and read by
//! `Inkling::load`. Its weights come from a fixed integer generator, so two
//! builds of this file make the same model bit for bit. Every layer takes its
//! experts from a dense source through `Inkling::set_experts`, which goes
//! through `layers::experts_forward`, the native experts' path.
//!
//! The class A gate: the f32 logits of a prefill and of 16 greedy steps, on a
//! short and a long prompt, equal bit for bit those the untouched code wrote.
//! `capture_before_*` writes that golden file and is run once, from a build
//! of the code before any wave 1 edit (`#[ignore]`); `tiny_matches_before_*`
//! reads it and fails naming it when it is missing.

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::ExpertSource;
use llvq_llm::inkling::{Inkling, LayerExperts};
use llvq_llm::lm::CausalLm;

fn data_dir() -> PathBuf {
    match std::env::var("INKLING_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini"),
    }
}

fn golden(device: &str) -> PathBuf {
    data_dir().join("lot3-e/golden").join(format!("wave1_tiny_{device}.f32"))
}

/// A fixed generator: splitmix64, mapped to `[-a, a)`.
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
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        (2.0 * u - 1.0) * a
    }

    fn tensor(&mut self, shape: &[usize], a: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.uniform(a)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).expect("host tensor")
    }

    /// A norm weight near 1.
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

/// Write the tiny config and checkpoint under a fresh directory; returns their paths.
fn write_tiny() -> (PathBuf, PathBuf, InklingConfig) {
    let dir = std::env::temp_dir().join(format!(
        "inkling_wave1_{}_{:?}",
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

/// Dense experts of one layer, on the model's device.
struct DenseExperts(Vec<(Tensor, Tensor)>);

impl ExpertSource for DenseExperts {
    fn count(&self) -> usize {
        self.0.len()
    }
    fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok(self.0[e].clone())
    }
}

/// The tiny model on `dev`, every layer on its dense experts.
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

/// The two gate prompts: 8 tokens, and 70 (far past the window of 4 and the extent of 6).
fn prompts() -> Vec<Vec<u32>> {
    let mut g = Gen(0x7070);
    let mut p = |n: usize| (0..n).map(|_| (g.next() % 60) as u32).collect::<Vec<u32>>();
    vec![p(8), p(70)]
}

const STEPS: usize = 16;

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

/// Prefill then `STEPS` greedy steps through `CausalLm`: `1 + STEPS` rows.
fn greedy_rows(m: &Inkling, prompt: &[u32]) -> Vec<Vec<f32>> {
    let mut cache = CausalLm::new_cache(m);
    let mut rows = vec![row(&CausalLm::prefill(m, prompt, &mut cache).unwrap())];
    for i in 0..STEPS {
        let next = argmax(rows.last().unwrap());
        rows.push(row(&CausalLm::step(m, next, prompt.len() + i, &mut cache).unwrap()));
    }
    rows
}

fn all_rows(dev: &Device) -> Vec<u8> {
    let m = tiny_model(dev);
    let mut out = vec![];
    for p in prompts() {
        for r in greedy_rows(&m, &p) {
            assert!(r.iter().all(|x| x.is_finite()), "non-finite logits");
            out.extend(r.iter().flat_map(|x| x.to_le_bytes()));
        }
    }
    out
}

fn capture(dev: &Device, name: &str) {
    let path = golden(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    assert!(!path.exists(), "{} exists: the golden is written once, from the untouched code", path.display());
    let bytes = all_rows(dev);
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
}

/// Bit for bit against the golden; on a difference, the first differing row
/// and the largest absolute gap.
fn compare(dev: &Device, name: &str) {
    let path = golden(name);
    assert!(path.exists(), "missing golden {}: run capture_before_{name} on the untouched code", path.display());
    let want = std::fs::read(&path).unwrap();
    let got = all_rows(dev);
    assert_eq!(got.len(), want.len(), "golden {} has another size", path.display());
    if got != want {
        let f = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
        let (a, b) = (f(&got), f(&want));
        let n_diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let first = a.iter().zip(&b).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap();
        let gap = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        panic!("{n_diff} logits differ from {}; first at row {}, largest gap {gap:e}", path.display(), first / 60);
    }
}

#[test]
#[ignore = "writes the golden; run once on the code before wave 1"]
fn capture_before_cpu() {
    capture(&Device::Cpu, "cpu");
}

#[test]
fn tiny_matches_before_cpu() {
    compare(&Device::Cpu, "cpu");
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU and writes the golden; run once on the code before wave 1, under the guard's lock"]
fn capture_before_metal() {
    capture(&Device::new_metal(0).unwrap(), "metal");
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn tiny_matches_before_metal() {
    compare(&Device::new_metal(0).unwrap(), "metal");
}

fn bits(t: &Tensor) -> Vec<u32> {
    row(t).iter().map(|x| x.to_bits()).collect()
}

/// Every position's logits of `ids` on `cache`, then `steps` greedy steps: the bits of each row.
fn all_positions_then_steps(
    m: &Inkling,
    ids: &[u32],
    cache: &mut llvq_llm::inkling::InklingCache,
    steps: usize,
) -> Vec<Vec<u32>> {
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

/// P2: the 70-token prompt with its scores in chunks of 16 query rows gives
/// every position's logits, and the 4 steps after, bit for bit those of one
/// score tensor. Returns the number of differing rows.
fn chunked_vs_one_call(dev: &Device) -> usize {
    let mut m = tiny_model(dev);
    let p = &prompts()[1];
    m.set_attn_chunk(0);
    let one = all_positions_then_steps(&m, p, &mut CausalLm::new_cache(&m), 4);
    m.set_attn_chunk(16);
    let chunked = all_positions_then_steps(&m, p, &mut CausalLm::new_cache(&m), 4);
    assert_eq!(one.len(), chunked.len());
    one.iter().zip(&chunked).filter(|(a, b)| a != b).count()
}

#[test]
fn chunked_prompt_matches_one_call_cpu() {
    let n = chunked_vs_one_call(&Device::Cpu);
    assert_eq!(n, 0, "{n} rows differ between chunks of 16 and one score tensor");
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn chunked_prompt_matches_one_call_metal() {
    let n = chunked_vs_one_call(&Device::new_metal(0).unwrap());
    assert_eq!(n, 0, "{n} rows differ between chunks of 16 and one score tensor");
}

/// P5: prefill(a), mark, prefill(x), rollback, then every position of `b`
/// and 4 steps, against prefill(a) then the same; and the refusals.
fn rollback_gate(dev: &Device) {
    let m = tiny_model(dev);
    let p = &prompts()[1];
    let (a, b) = (&p[..30], &p[30..55]);
    let x: Vec<u32> = p[55..].iter().rev().copied().collect();

    // prefill(a) then one step: the mark holds convolution states in their ring form,
    // and the prefill of b after the rollback reads them
    let mut c1 = CausalLm::new_cache(&m);
    CausalLm::prefill(&m, a, &mut c1).unwrap();
    CausalLm::step(&m, 9, a.len(), &mut c1).unwrap();
    let mark = CausalLm::mark(&m, &c1);
    CausalLm::prefill(&m, &x, &mut c1).unwrap();
    // a step too, so every convolution state sits in its ring form
    let pos = c1.len();
    CausalLm::step(&m, 7, pos, &mut c1).unwrap();
    CausalLm::rollback(&m, &mut c1, &mark).unwrap();
    assert_eq!(c1.len(), a.len() + 1);
    let after = all_positions_then_steps(&m, b, &mut c1, 4);

    let mut c2 = CausalLm::new_cache(&m);
    CausalLm::prefill(&m, a, &mut c2).unwrap();
    CausalLm::step(&m, 9, a.len(), &mut c2).unwrap();
    let mut c3 = c2.clone();
    let want = all_positions_then_steps(&m, b, &mut c2, 4);
    let differing = after.iter().zip(&want).filter(|(u, v)| u != v).count();
    assert_eq!(differing, 0, "{differing} of {} rows differ after the rollback", want.len());

    // a clone is its own cache: the original moved on, the clone gives the same rows
    assert_eq!(all_positions_then_steps(&m, b, &mut c3, 4), want, "a clone shares state with its original");

    // a mark that is not a prefix is refused and changes nothing
    let mut c4 = CausalLm::new_cache(&m);
    CausalLm::prefill(&m, &x, &mut c4).unwrap();
    let held = c4.len();
    assert!(CausalLm::rollback(&m, &mut c4, &mark).is_err(), "a mark of other tokens was accepted");
    assert_eq!(c4.len(), held);
    let early = CausalLm::mark(&m, &CausalLm::new_cache(&m));
    CausalLm::rollback(&m, &mut c4, &early).unwrap();
    assert_eq!(c4.len(), 0);
    assert!(CausalLm::rollback(&m, &mut c4, &mark).is_err(), "a mark longer than the cache was accepted");
}

#[test]
fn rollback_restores_the_cache_cpu() {
    rollback_gate(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn rollback_restores_the_cache_metal() {
    rollback_gate(&Device::new_metal(0).unwrap());
}

/// Not a gate: a and b in two prefills against a then b in one call. The
/// experts batch their rows by size, so the two may round apart.
#[test]
fn two_prefills_against_one_call_cpu() {
    let m = tiny_model(&Device::Cpu);
    let p = &prompts()[1];
    let mut c1 = CausalLm::new_cache(&m);
    CausalLm::prefill(&m, &p[..30], &mut c1).unwrap();
    let two = bits(&CausalLm::prefill(&m, &p[30..55], &mut c1).unwrap());
    let one = bits(&CausalLm::prefill(&m, &p[..55], &mut CausalLm::new_cache(&m)).unwrap());
    println!("two prefills against one call, last logits bit for bit: {}", two == one);
}

/// D5: a cache that grows its key and value buffers in the middle of the
/// steps (250 positions, a grain of 256) gives the logits, bit for bit, of a
/// cache that reserved 512 positions first and never grows. Returns the
/// number of differing rows.
fn growth_vs_reserved(dev: &Device) -> usize {
    let m = tiny_model(dev);
    let mut g = Gen(0x2500);
    let p: Vec<u32> = (0..250).map(|_| (g.next() % 60) as u32).collect();
    let run = |reserve: usize| {
        let mut c = CausalLm::new_cache(&m);
        c.reserve(reserve).unwrap();
        let mut last = row(&CausalLm::prefill(&m, &p, &mut c).unwrap());
        let mut rows = vec![last.iter().map(|x| x.to_bits()).collect::<Vec<u32>>()];
        for i in 0..12 {
            let l = CausalLm::step(&m, argmax(&last), p.len() + i, &mut c).unwrap();
            rows.push(bits(&l));
            last = row(&l);
        }
        (rows, c.slots()[0].capacity())
    };
    let (grown, cap_grown) = run(0);
    let (reserved, cap_reserved) = run(512);
    assert!(cap_grown > 256 && cap_reserved == 512, "capacities {cap_grown} and {cap_reserved}");
    grown.iter().zip(&reserved).filter(|(a, b)| a != b).count()
}

#[test]
fn growth_matches_reserved_cpu() {
    let n = growth_vs_reserved(&Device::Cpu);
    assert_eq!(n, 0, "{n} rows differ between a growing and a reserved cache");
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn growth_matches_reserved_metal() {
    let n = growth_vs_reserved(&Device::new_metal(0).unwrap());
    assert_eq!(n, 0, "{n} rows differ between a growing and a reserved cache");
}
