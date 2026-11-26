//! Wave 2 of the Inkling 2 Mini throughput plan (`docs/plan-debit-inkling2-mini.md`),
//! item D3: the 8 routed experts of one decoded token in a few launches.
//!
//! The class A gate: the f32 logits of the prefill and of 16 greedy decode
//! steps, on a 32-token prompt and on a 640-token prompt, written raw to
//! `D/lot4-m/logits/<binary>_<device>_<prompt>_<arm>.f32` (one row of 200,058
//! f32 a step, prefill first) with a JSON beside it. The same tests are
//! compiled once from the code before D3 and once after, and the files must be
//! equal byte for byte (`cmp`). The binary's own file name tells the builds
//! apart, so no knob is needed.
//!
//! Every test that loads the real model is `#[ignore]`d: run it under the
//! guard's lock, one at a time.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use llvq_llm::inkling::resident::{self, ResidentExperts};
use llvq_llm::inkling::{Inkling, LayerExperts};

const DECODE_STEPS: usize = 16;
const LONG_PROMPT: usize = 640;

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini")
}

fn need(p: PathBuf) -> PathBuf {
    assert!(p.exists(), "missing input {}: the gate cannot run without it", p.display());
    p
}

fn out_dir(sub: &str) -> PathBuf {
    let d = data_dir().join("lot4-m").join(sub);
    std::fs::create_dir_all(&d).expect("output directory");
    d
}

/// The file name of this test binary: the tag of every output.
fn tag() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".into())
}

/// The 32-token prompt of the resident gates (`lot2-d/ref_full.safetensors`).
fn prompt32() -> Vec<u32> {
    let dump = need(data_dir().join("lot2-d/ref_full.safetensors"));
    let refs = candle_core::safetensors::load(&dump, &Device::Cpu).expect("ref_full");
    refs["ids"].to_dtype(DType::I64).unwrap().to_vec1::<i64>().unwrap().iter().map(|&i| i as u32).collect()
}

/// The first `n` tokens of WikiText-2 test window 0 (`lot2-d/wikitext_w0_2048.json`).
fn window(n: usize) -> Vec<u32> {
    let p = need(data_dir().join("lot2-d/wikitext_w0_2048.json"));
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    let ids: Vec<u32> = j["ids"].as_array().expect("ids").iter().map(|v| v.as_u64().unwrap() as u32).collect();
    assert!(ids.len() >= n, "{} ids for {n} asked", ids.len());
    ids[..n].to_vec()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn allocated(dev: &Device) -> u64 {
    match dev {
        Device::Metal(m) => {
            dev.synchronize().unwrap();
            m.device().current_allocated_size() as u64
        }
        _ => 0,
    }
}

/// Which MoE path a resident layer takes at one row: the one-expert path
/// (`forward_one_row`) or the four fused launches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Old,
    New,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Old => "old",
            Arm::New => "new",
        }
    }
}

fn set_arm(r: &mut ResidentExperts, arm: Arm) {
    r.set_fused(arm == Arm::New);
    if arm == Arm::New {
        assert!(r.fused_available(), "layer {}: no fused path", r.layer);
    }
}

/// The model, with every verified layer resident on Metal and on `arm`.
fn model(dev: &Device, arm: Option<Arm>) -> (Inkling, usize, u64) {
    let d = data_dir();
    let mut m = Inkling::load(
        &need(d.join("textonly/config.json")),
        &need(d.join("native/model-00001-of-00001.safetensors")),
        dev,
    )
    .expect("load");
    let mut n = 0;
    if let Some(arm) = arm {
        let rt = resident::served_runtime(dev).expect("runtime");
        let dir = need(d.join("layers"));
        for l in resident::verified_layers(&dir, m.cfg.num_hidden_layers) {
            let mut r = ResidentExperts::load(rt.clone(), dev, &dir.join(format!("L{l:02}")), l, &m.cfg).expect("layer");
            set_arm(&mut r, arm);
            m.set_experts(l, LayerExperts::Source(Box::new(r)));
            n += 1;
        }
    }
    let bytes = allocated(dev);
    println!("resident layers: {n}, device allocated after load: {bytes} B");
    (m, n, bytes)
}

/// Prefill, then `DECODE_STEPS` greedy steps; every logits row to one file.
fn capture(m: &Inkling, dev: &Device, label: &str, prompt: &[u32], extra: serde_json::Value) {
    let mut cache = m.new_cache();
    let mut bytes: Vec<u8> = Vec::new();
    let mut tokens = vec![];
    let mut secs = vec![];
    let mut launches = vec![];
    let t = Instant::now();
    m.take_launches();
    let mut v = to_vec(&m.forward(prompt, &mut cache, false, false).expect("prefill").logits);
    let prefill_s = t.elapsed().as_secs_f64();
    let prefill_launches = m.take_launches();
    for step in 0..=DECODE_STEPS {
        bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        let tok = argmax(&v);
        tokens.push(tok);
        if step == DECODE_STEPS {
            break;
        }
        let t = Instant::now();
        v = to_vec(&m.forward(&[tok], &mut cache, false, false).expect("step").logits);
        secs.push(t.elapsed().as_secs_f64());
        launches.push(m.take_launches());
    }
    let dname = if dev.is_metal() { "metal" } else { "cpu" };
    let base = format!("{}_{dname}_{label}", tag());
    std::fs::write(out_dir("logits").join(format!("{base}.f32")), &bytes).expect("write logits");
    let j = serde_json::json!({
        "binary": tag(), "device": dname, "prompt": label, "prompt_tokens": prompt.len(),
        "rows": DECODE_STEPS + 1, "vocab": v.len(), "tokens": tokens,
        "seconds_prefill": prefill_s, "seconds_steps": secs, "median_step": median(secs.clone()),
        "lattice_launches_prefill": prefill_launches, "lattice_launches_steps": launches, "extra": extra,
    });
    std::fs::write(out_dir("logits").join(format!("{base}.json")), serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{j}");
}

fn capture_metal(arm: Arm) {
    let dev = Device::new_metal(0).expect("Metal");
    let (m, n, bytes) = model(&dev, Some(arm));
    let extra = serde_json::json!({"arm": arm.label(), "resident_layers": n, "allocated_after_load": bytes});
    capture(&m, &dev, &format!("p32_{}", arm.label()), &prompt32(), extra.clone());
    capture(&m, &dev, &format!("p640_{}", arm.label()), &window(LONG_PROMPT), extra);
}

/// The class A capture on Metal, every verified layer resident, old path.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn capture_logits_metal_old() {
    capture_metal(Arm::Old);
}

/// The same capture on the four fused launches.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn capture_logits_metal_new() {
    capture_metal(Arm::New);
}

/// The same capture through the loader the model serves with
/// (`Inkling::make_resident`), on each layer's default path.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn capture_logits_metal_default() {
    let dev = Device::new_metal(0).expect("Metal");
    let (mut m, _, _) = model(&dev, None);
    let rt = resident::served_runtime(&dev).expect("runtime");
    let n = m.make_resident(&need(data_dir().join("layers")), rt).expect("resident").len();
    let bytes = allocated(&dev);
    let extra = serde_json::json!({"arm": "default", "fused_default": resident::FUSED_DEFAULT,
        "resident_layers": n, "allocated_after_load": bytes});
    capture(&m, &dev, "p32_default", &prompt32(), extra.clone());
    capture(&m, &dev, "p640_default", &window(LONG_PROMPT), extra);
}

/// The class A capture on the CPU: native experts, nothing resident. D3
/// does not reach this path; the capture shows the change leaves it alone.
#[test]
#[ignore = "loads the real model on the CPU; run under the guard's lock"]
fn capture_logits_cpu() {
    let dev = Device::Cpu;
    let (m, _, _) = model(&dev, None);
    capture(&m, &dev, "p32", &prompt32(), serde_json::json!({}));
    capture(&m, &dev, "p640", &window(LONG_PROMPT), serde_json::json!({}));
}

/// Decode timing: the 32-token prompt, then 8 timed decode steps (each ends
/// on reading its logits back), every verified layer resident on `arm`. Run
/// the arms alternately in separate processes.
fn decode_timing(arm: Arm) {
    let dev = Device::new_metal(0).expect("Metal");
    let t = Instant::now();
    let (m, n, bytes) = model(&dev, Some(arm));
    let load_s = t.elapsed().as_secs_f64();
    let prompt = prompt32();
    let mut cache = m.new_cache();
    let mut v = to_vec(&m.forward(&prompt, &mut cache, false, false).unwrap().logits);
    m.take_launches();
    let (mut secs, mut tokens, mut launches) = (vec![], vec![], vec![]);
    for _ in 0..8 {
        let tok = argmax(&v);
        tokens.push(tok);
        let t = Instant::now();
        v = to_vec(&m.forward(&[tok], &mut cache, false, false).unwrap().logits);
        secs.push(t.elapsed().as_secs_f64());
        launches.push(m.take_launches());
    }
    let j = serde_json::json!({"arm": arm.label(), "binary": tag(), "resident_layers": n,
        "allocated_after_load": bytes, "seconds_load": load_s, "seconds_steps": secs,
        "median_step": median(secs.clone()), "lattice_launches_steps": launches, "tokens": tokens});
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let p = out_dir("runs").join(format!("decode_{}_{}-{ts}.json", arm.label(), tag()));
    std::fs::write(p, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{j}");
}

#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn m_decode_timing_old() {
    decode_timing(Arm::Old);
}

#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn m_decode_timing_new() {
    decode_timing(Arm::New);
}

// ---------------------------------------------------------------------------
// Synthetic gates: random records at the model's shapes on the GPU, no model.
// Each test takes the lock below, so two tests never upload at once.
// ---------------------------------------------------------------------------

use std::sync::{Arc, Mutex};

use llvq_artifact::tetra48::transcode_tetra48;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::RotationTables;
use llvq_llm::fused_metal::{MetalRuntime, MetalTuning, MoeHostExpert, MoeHostProj};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::Routing;
use llvq_llm::inkling::resident::ResidentExpert;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// A runtime on the bench's rank tables, as `tests/inkling_wave1_kernels.rs` builds it.
fn runtime(dev: &Device) -> MetalRuntime {
    use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
    let table = RankTable::build();
    let tr = llvq_bench::f1::Trellis::new();
    MetalRuntime::new(
        dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        resident::SERVED_TILE,
        false,
    )
    .expect("the runtime builds")
}

fn config() -> InklingConfig {
    InklingConfig::from_file(&need(data_dir().join("textonly/config.json"))).expect("config")
}

/// One synthetic Tetra48 record, `d_out x (nblocks * 24 + tail_w)`.
fn rec(name: String, seed: u64, d_out: usize, nblocks: usize, tail_w: usize) -> MoeHostProj {
    let mut rng = SplitMix64::new(seed);
    let n = d_out * nblocks;
    let indices: Vec<u64> = (0..n)
        .map(|_| match rng.next().is_multiple_of(6) {
            true => 0,
            false => 1 + rng.next() % (llvq_search::index::N13.min(1u64 << 47) - 1),
        })
        .collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let s = transcode_tetra48(&indices, &gains, d_out, nblocks).expect("transcodes");
    let words = s.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    // Row scales of the size the expert records carry, so the activations stay in range.
    let rscale = (0..d_out).map(|_| (0.002 + 0.004 * rng.next_gaussian().abs()) as f32).collect();
    let tail = (0..d_out * tail_w).map(|_| half::f16::from_f32(0.05 * rng.next_gaussian() as f32).to_bits()).collect();
    let g0 = 0.5 + 0.25 * (rng.next() % 4) as f32;
    MoeHostProj {
        name,
        d_out,
        d_in: nblocks * DIM + tail_w,
        nblocks,
        row_stride_u32: s.stride_u32,
        words,
        gscale: [g0, g0 + 0.75],
        rscale,
        tail,
    }
}

/// `n` synthetic experts, gate and up `inter x hidden`, down `hidden x inter`.
/// `tail` adds that many tail columns to every record (the experts of the
/// model have none).
fn hosts(n: usize, hidden: usize, inter: usize, tail: usize, salt: u64) -> Vec<MoeHostExpert> {
    (0..n)
        .map(|e| {
            let s = salt + 8 * e as u64;
            let mk = |m: &str, seed: u64, d_out: usize, d_in: usize| {
                rec(format!("{e}.{m}"), seed, d_out, (d_in - tail) / DIM, tail)
            };
            MoeHostExpert {
                projs: [mk("gate", s, inter, hidden), mk("up", s + 1, inter, hidden), mk("down", s + 2, hidden, inter)],
                rot_in: RotationTables::build(hidden, s + 3).unwrap(),
                rot_down: RotationTables::build(inter, s + 4).unwrap(),
            }
        })
        .collect()
}

/// `t` tokens, each routed to `k` distinct experts of `n` in a random order.
fn routing(seed: u64, t: usize, n: usize, k: usize) -> Routing {
    let mut rng = SplitMix64::new(seed);
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for _ in 0..t {
        let mut pool: Vec<u32> = (0..n as u32).collect();
        let mut idx = vec![];
        for _ in 0..k {
            idx.push(pool.remove((rng.next() % pool.len() as u64) as usize));
        }
        r.weights.push((0..k).map(|_| (0.1 + rng.next_gaussian().abs()) as f32).collect());
        r.indices.push(idx);
        r.margin.push(0.0);
        r.scores.push(vec![]);
    }
    r
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_gaussian() as f32).collect()
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

/// One row through the fused path and through the one-expert path on the
/// same layer, for `count` routings: every f32 equal. Returns the routings
/// compared.
fn fused_against_one_row(layer: &ResidentExperts, cfg: &InklingConfig, dev: &Device, seed: u64, count: usize) -> usize {
    let n = layer.count();
    for i in 0..count as u64 {
        let r = routing(seed + i, 1, n, cfg.num_experts_per_tok);
        let x = Tensor::from_slice(&values(seed ^ (i << 8), cfg.hidden_size), (1, cfg.hidden_size), dev).unwrap();
        let old = to_vec(&layer.forward_one_row(&x, &r, cfg).unwrap().expect("distinct experts"));
        let new = to_vec(&layer.forward_one_row_fused(&x, &r, cfg).unwrap().expect("the fused path runs"));
        assert!(old.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&new, &old), None, "routing {i} {:?}: fused against one row", r.indices[0]);
    }
    count
}

/// D3 at the model's shapes: 128 experts in one slab, 24 random routings of
/// 8 experts, the fused path bit for bit with `forward_one_row`; the switch
/// sends `forward` to the fused path and back; a repeated expert and an
/// expert past the layer are refused as the one-expert path refuses them.
#[test]
fn d3_fused_is_one_row_at_model_shapes() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xD300);
    let mut layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
    drop(hs);
    assert!(layer.fused_available(), "a uniform slab on the served tuning has a fused path");
    let done = fused_against_one_row(&layer, &cfg, &dev, 0xD3A0, 24);
    println!("D3 synthetic, 128 experts at {}x{}: {done} routings bit for bit", cfg.hidden_size, cfg.intermediate_size);

    let r = routing(0xD3F0, 1, layer.count(), cfg.num_experts_per_tok);
    let x = Tensor::from_slice(&values(0xD3F1, cfg.hidden_size), (1, cfg.hidden_size), &dev).unwrap();
    layer.set_fused(false);
    layer.take_launches();
    let a = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let la = layer.take_launches();
    layer.set_fused(true);
    let b = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let lb = layer.take_launches();
    assert_eq!(first_difference(&b, &a), None, "forward, switch on against off");
    assert_eq!((la, lb), (40, 4), "lattice launches a call, one-expert path then fused");
    let down = if rt.moe_down_sg() { "moe_tetra_down_sg_metal" } else { "moe_tetra_down_metal" };
    assert!(rt.compiled_pipelines().contains(&down), "the fused kernels ran");

    let mut rep = r.clone();
    rep.indices[0][1] = rep.indices[0][0];
    assert!(layer.forward_one_row_fused(&x, &rep, &cfg).unwrap().is_none(), "a repeated expert is not fused");
    let general = to_vec(&layer.forward_general(&x, &rep, &cfg).unwrap());
    assert_eq!(first_difference(&to_vec(&layer.forward(&x, &rep, &cfg).unwrap()), &general), None, "repeated expert");
    let mut past = r.clone();
    past.indices[0][3] = layer.count() as u32;
    assert!(layer.forward_one_row_fused(&x, &past, &cfg).is_err(), "an expert past the layer is refused");
    let mut tuned = runtime(&dev);
    tuned.set_tuning(MetalTuning { rot_tg: true, ..MetalTuning::default() }).unwrap();
    layer.set_runtime(Arc::new(tuned));
    assert!(!layer.fused_available(), "another rotation kernel: no fused path");
    assert!(layer.forward_one_row_fused(&x, &r, &cfg).unwrap().is_none(), "and no fused call");
}

/// D9, first half: the routing handed to the MoE on the device. On a slab
/// layer of 128 synthetic experts at the model's shapes, for 16 score rows:
/// `Glue::sel` after `Glue::top8` holds, read back, the ids ascending and the
/// bits of the weights that `Top8::read` gives the host; and
/// `forward_one_row_routed_on_device` on it is bit for bit
/// `forward_one_row_fused` on the routing read back. A buffer of another
/// type, another expert count and the fused switch off send the caller back
/// to the host path.
#[test]
fn d9_routing_on_device_is_the_host_routing() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let (n, k) = (cfg.n_routed_experts, cfg.num_experts_per_tok);
    let hs = hosts(n, cfg.hidden_size, cfg.intermediate_size, 0, 0xD900);
    let mut layer = ResidentExperts::from_host(rt, &dev, 6, &hs).expect("slab layer");
    drop(hs);
    let glue = llvq_llm::inkling::glue::Glue::new(&dev).expect("glue kernels");
    let mut last = None;
    for i in 0..16u64 {
        let sig: Vec<f32> = values(0xD910 + i, n).iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let bias: Vec<f32> = values(0xD950 + i, n).iter().map(|v| 0.01 * v).collect();
        let scores = Tensor::from_slice(&sig, n, &dev).unwrap();
        let bias = Tensor::from_slice(&bias, n, &dev).unwrap();
        let top = glue.top8(&scores, &bias, k, cfg.route_scale).unwrap();
        let r = top.read(k).unwrap();
        assert!(r.ids.windows(2).all(|p| p[0] < p[1]), "row {i}: ids {:?} not ascending", r.ids);
        let sel = glue.sel(&top, k).unwrap();
        let mut want = r.ids.clone();
        want.extend(r.weights.iter().map(|w| w.to_bits()));
        assert_eq!(sel.to_vec1::<u32>().unwrap(), want, "row {i}: the device routing is not the host's");
        let routing =
            Routing { indices: vec![r.ids.clone()], weights: vec![r.weights.clone()], margin: vec![r.margin], scores: vec![vec![]] };
        let x = Tensor::from_slice(&values(0xD990 + i, cfg.hidden_size), (1, cfg.hidden_size), &dev).unwrap();
        let host = to_vec(&layer.forward_one_row_fused(&x, &routing, &cfg).unwrap().expect("the fused path runs"));
        let device = layer.forward_one_row_routed_on_device(&x, &sel, n, &cfg).unwrap().expect("the device path runs");
        assert!(host.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&to_vec(&device), &host), None, "row {i} {:?}: device routing against host routing", r.ids);
        last = Some((x, sel));
    }
    let (x, sel) = last.expect("16 rows");
    assert!(layer.forward_one_row_routed_on_device(&x, &sel, n + 1, &cfg).unwrap().is_none(), "another expert count");
    let as_f32 = sel.to_dtype(DType::F32).unwrap();
    assert!(layer.forward_one_row_routed_on_device(&x, &as_f32, n, &cfg).unwrap().is_none(), "a buffer that is not u32");
    layer.set_fused(false);
    assert!(layer.forward_one_row_routed_on_device(&x, &sel, n, &cfg).unwrap().is_none(), "the fused switch off");
}

/// The activation of the fused path in one launch (`inkling::moe_act`)
/// against candle's operations, inside the MoE: on a slab layer of 128
/// synthetic experts at the model's shapes, 16 routings, the output of
/// `forward_one_row_fused` with the launch is bit for bit its output without.
#[test]
fn fused_activation_launch_is_candle_activation() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xAC70);
    let mut layer = ResidentExperts::from_host(rt, &dev, 6, &hs).expect("slab layer");
    drop(hs);
    assert!(layer.fused_act_active(), "the activation kernel compiled and is on by default");
    for i in 0..16u64 {
        let r = routing(0xAC80 + i, 1, layer.count(), cfg.num_experts_per_tok);
        let x = Tensor::from_slice(&values(0xAC90 + i, cfg.hidden_size), (1, cfg.hidden_size), &dev).unwrap();
        layer.set_fused_act(false);
        assert!(!layer.fused_act_active());
        let candle = to_vec(&layer.forward_one_row_fused(&x, &r, &cfg).unwrap().expect("the fused path runs"));
        layer.set_fused_act(true);
        let kernel = to_vec(&layer.forward_one_row_fused(&x, &r, &cfg).unwrap().expect("the fused path runs"));
        assert!(candle.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&kernel, &candle), None, "routing {i} {:?}: one launch against candle's operations", r.indices[0]);
    }
}

/// D3 on small layers: 12 experts at two other shapes, one with tails, so the
/// offsets of every record kind, the tail loop and the odd-block mix with
/// another k are exercised; 1 to 12 experts a call.
#[test]
fn d3_fused_is_one_row_on_small_layers() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let base = config();
    let rt = Arc::new(runtime(&dev));
    for (hidden, inter, tail, salt) in [(768usize, 384usize, 0usize, 0xE100u64), (1024, 400, 16, 0xE200)] {
        let mut cfg = base.clone();
        cfg.hidden_size = hidden;
        cfg.intermediate_size = inter;
        let hs = hosts(12, hidden, inter, tail, salt);
        let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
        assert!(layer.fused_available(), "{hidden}x{inter}");
        for k in [1usize, 2, 5, 8, 12] {
            cfg.num_experts_per_tok = k;
            fused_against_one_row(&layer, &cfg, &dev, salt + 16 * k as u64, 3);
        }
    }
}

/// The slab's record buffers are the bytes of `upload_stream_unpooled`: the
/// one-expert path on aliases into the heap and on standalone K1 buffers
/// returns the same bits, and the general path at 5 rows too.
#[test]
fn d3_slab_aliases_are_the_unpooled_upload() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let hs = hosts(12, cfg.hidden_size, cfg.intermediate_size, 0, 0xF100);
    let slab = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
    let up = |p: &MoeHostProj| {
        rt.upload_stream_unpooled(&p.name, p.d_out, p.d_in, p.nblocks, p.row_stride_u32, &p.words, &p.gscale, &p.rscale, &p.tail)
            .unwrap()
    };
    let plain: Vec<ResidentExpert> = hs
        .iter()
        .map(|h| ResidentExpert {
            gate: up(&h.projs[0]),
            up: up(&h.projs[1]),
            down: up(&h.projs[2]),
            rot_in: rt.upload_rotation_unpooled(&h.rot_in).unwrap(),
            rot_down: rt.upload_rotation_unpooled(&h.rot_down).unwrap(),
        })
        .collect();
    let plain = ResidentExperts::from_parts(rt.clone(), &dev, 6, plain);
    for seed in 0..4u64 {
        let r = routing(0xF1A0 + seed, 1, 12, cfg.num_experts_per_tok);
        let x = Tensor::from_slice(&values(seed, cfg.hidden_size), (1, cfg.hidden_size), &dev).unwrap();
        let a = to_vec(&slab.forward_one_row(&x, &r, &cfg).unwrap().unwrap());
        let b = to_vec(&plain.forward_one_row(&x, &r, &cfg).unwrap().unwrap());
        assert!(b.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&a, &b), None, "routing {seed}: slab aliases against standalone buffers");
    }
    let r = routing(0xF1B0, 5, 12, cfg.num_experts_per_tok);
    let x = Tensor::from_slice(&values(0xF1B1, 5 * cfg.hidden_size), (5, cfg.hidden_size), &dev).unwrap();
    let a = to_vec(&slab.forward_general(&x, &r, &cfg).unwrap());
    let b = to_vec(&plain.forward_general(&x, &r, &cfg).unwrap());
    assert_eq!(first_difference(&a, &b), None, "5 rows: slab aliases against standalone buffers");
    println!("slab: {} B for 12 experts, {} B of records", slab.stats.slab_bytes, hs.iter().map(|h| {
        h.projs.iter().map(|p| 4 * p.words.len() + 8 + 4 * p.rscale.len() + 2 * p.tail.len().max(1)).sum::<usize>()
            + [&h.rot_in, &h.rot_down].iter().map(|t| 4 * t.signbits.len() + 4 * t.small.len().max(1)).sum::<usize>()
    }).sum::<usize>());
}

/// One layer's resident experts behind a lock, so a loaded model can switch
/// arm between steps.
struct Shared(Arc<std::sync::RwLock<ResidentExperts>>);

impl llvq_llm::inkling::layers::MoeSource for Shared {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> candle_core::Result<Tensor> {
        self.0.read().expect("no panic held this lock").forward(x, routing, cfg)
    }
    fn take_launches(&self) -> usize {
        self.0.read().expect("no panic held this lock").take_launches()
    }
}

/// The MoE calls of a decode step timed one by one (the model drains the
/// device around each call), the two arms alternated step by step in one
/// process: 32-token prompt, then 8 steps an arm. Since the arms are bit for
/// bit, both read the same tokens.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn m_moe_call_timing() {
    let dev = Device::new_metal(0).expect("Metal");
    let (mut m, _, _) = model(&dev, None);
    let rt = resident::served_runtime(&dev).expect("runtime");
    let dir = need(data_dir().join("layers"));
    let mut layers = vec![];
    for l in resident::verified_layers(&dir, m.cfg.num_hidden_layers) {
        let r = ResidentExperts::load(rt.clone(), &dev, &dir.join(format!("L{l:02}")), l, &m.cfg).expect("layer");
        let a = Arc::new(std::sync::RwLock::new(r));
        m.set_experts(l, LayerExperts::Source(Box::new(Shared(a.clone()))));
        layers.push(a);
    }
    let mut cache = m.new_cache();
    let mut v = to_vec(&m.forward(&prompt32(), &mut cache, false, false).unwrap().logits);
    let (mut calls, mut steps) = ([vec![], vec![]], [vec![], vec![]]);
    for step in 0..16 {
        let arm = step % 2;
        for l in &layers {
            l.write().unwrap().set_fused(arm == 1);
        }
        m.set_moe_timing(true);
        let t = Instant::now();
        v = to_vec(&m.forward(&[argmax(&v)], &mut cache, false, false).unwrap().logits);
        steps[arm].push(t.elapsed().as_secs_f64());
        let c = m.take_moe_timing();
        m.set_moe_timing(false);
        calls[arm].push(c.iter().map(|p| p.1).sum::<f64>());
        if step < 2 {
            println!("arm {arm}: {} MoE calls", c.len());
        }
        if step >= 14 {
            let mut per: Vec<f64> = c.iter().map(|p| p.1).collect();
            per.sort_by(|a, b| a.total_cmp(b));
            println!("arm {arm}: median call {:.3} ms", 1e3 * per[per.len() / 2]);
        }
    }
    let j = serde_json::json!({"binary": tag(), "arms": ["old", "new"],
        "moe_seconds_a_step": calls, "step_seconds_timed": steps,
        "median_moe_old": median(calls[0].clone()), "median_moe_new": median(calls[1].clone()),
        "median_step_old": median(steps[0].clone()), "median_step_new": median(steps[1].clone())});
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    std::fs::write(out_dir("runs").join(format!("moe_calls_{}-{ts}.json", tag())), serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{j}");
}
