//! Wave 1 of the Inkling 2 Mini throughput plan (`docs/plan-debit-inkling2-mini.md`):
//! the class A gate of the whole model, and the decode timings.
//!
//! The class A gate: the f32 logits of the prefill and of 16 greedy decode
//! steps, on a 32-token prompt and on a 640-token prompt, written raw to
//! `D/lot3-k/logits/<binary>_<device>_<prompt>.f32` (one row of 200,058 f32
//! a step, prefill first) with a JSON beside it. The same tests are compiled
//! once from the code before wave 1 and once after, and the two files must be
//! equal byte for byte (`cmp`). The binary's own file name tells the two
//! builds apart, so no knob is needed.
//!
//! Every test that loads the real model is `#[ignore]`d: run it under the
//! guard's lock, one at a time.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use std::sync::{Arc, Mutex, RwLock};

use candle_core::{DType, Device, Tensor};
use llvq_artifact::tetra48::transcode_tetra48;
use llvq_artifact::{QuantData, RawData, RawTensor};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::RotationTables;
use llvq_llm::fused_metal::{MetalRuntime, MetalTetraProj, MetalTuning};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::head::{OutputProjection, Q4MetalHead};
use llvq_llm::inkling::layers::{MoeSource, Routing};
use llvq_llm::inkling::resident::{ResidentExpert, ResidentExperts};
use llvq_llm::inkling::{resident, Inkling, LayerExperts};

const DECODE_STEPS: usize = 16;
const LONG_PROMPT: usize = 640;

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini")
}

fn need(p: PathBuf) -> PathBuf {
    assert!(p.exists(), "missing input {}: the gate cannot run without it", p.display());
    p
}

fn out_dir() -> PathBuf {
    let d = data_dir().join("lot3-k/logits");
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

fn model(dev: &Device, resident_layers: bool) -> Inkling {
    let d = data_dir();
    let mut m = Inkling::load(
        &need(d.join("textonly/config.json")),
        &need(d.join("native/model-00001-of-00001.safetensors")),
        dev,
    )
    .expect("load");
    if resident_layers {
        let rt = resident::served_runtime(dev).expect("runtime");
        let layers = m.make_resident(&need(d.join("layers")), rt).expect("resident");
        println!("resident layers: {}", layers.len());
    }
    m
}

/// Prefill, then `DECODE_STEPS` greedy steps; every logits row to one file.
fn capture(m: &Inkling, dev: &Device, label: &str, prompt: &[u32]) {
    let mut cache = m.new_cache();
    let mut bytes: Vec<u8> = Vec::new();
    let mut tokens = vec![];
    let mut secs = vec![];
    let t = Instant::now();
    let mut row = m.forward(prompt, &mut cache, false, false).expect("prefill").logits;
    let mut v = row.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let prefill_s = t.elapsed().as_secs_f64();
    for step in 0..=DECODE_STEPS {
        bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        let tok = argmax(&v);
        tokens.push(tok);
        if step == DECODE_STEPS {
            break;
        }
        let t = Instant::now();
        row = m.forward(&[tok], &mut cache, false, false).expect("step").logits;
        v = row.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        secs.push(t.elapsed().as_secs_f64());
    }
    let dname = if dev.is_metal() { "metal" } else { "cpu" };
    let base = format!("{}_{dname}_{label}", tag());
    std::fs::write(out_dir().join(format!("{base}.f32")), &bytes).expect("write logits");
    let j = serde_json::json!({
        "binary": tag(), "device": dname, "prompt": label, "prompt_tokens": prompt.len(),
        "rows": DECODE_STEPS + 1, "vocab": v.len(), "tokens": tokens,
        "seconds_prefill": prefill_s, "seconds_steps": secs,
    });
    std::fs::write(out_dir().join(format!("{base}.json")), serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{j}");
}

/// The class A capture on Metal, every verified layer resident.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn capture_logits_metal() {
    let dev = Device::new_metal(0).expect("Metal");
    let m = model(&dev, true);
    capture(&m, &dev, "p32", &prompt32());
    capture(&m, &dev, "p640", &window(LONG_PROMPT));
}

/// The class A capture on the CPU: native experts, nothing resident.
#[test]
#[ignore = "loads the real model on the CPU; run under the guard's lock"]
fn capture_logits_cpu() {
    let dev = Device::Cpu;
    let m = model(&dev, false);
    capture(&m, &dev, "p32", &prompt32());
    capture(&m, &dev, "p640", &window(LONG_PROMPT));
}

// ---------------------------------------------------------------------------
// Synthetic gates: tiny records on the GPU, no model. Each test takes the
// lock below, because the pool check reads the device's allocated size and
// another test allocating at the same time would move it.
// ---------------------------------------------------------------------------

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

/// A runtime on the bench's rank tables, with host copies when `host`.
fn runtime(dev: &Device, host: bool) -> MetalRuntime {
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
        host,
    )
    .expect("the runtime builds")
}

const GSCALE: [f32; 2] = [0.625, 1.375];

/// One synthetic Tetra48 record.
struct Rec {
    d_out: usize,
    d_in: usize,
    nblocks: usize,
    stride: usize,
    words: Vec<u32>,
    rscale: Vec<f32>,
    tail: Vec<u16>,
}

fn rec(seed: u64, d_out: usize, nblocks: usize, tail_w: usize) -> Rec {
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
    Rec { d_out, d_in: nblocks * DIM + tail_w, nblocks, stride: s.stride_u32, words, rscale, tail }
}

fn pooled(rt: &MetalRuntime, r: &Rec) -> MetalTetraProj {
    rt.upload_stream("000.gate_proj", r.d_out, r.d_in, r.nblocks, r.stride, &r.words, &GSCALE, &r.rscale, &r.tail)
        .expect("pooled upload")
}

fn unpooled(rt: &MetalRuntime, r: &Rec) -> MetalTetraProj {
    rt.upload_stream_unpooled("000.gate_proj", r.d_out, r.d_in, r.nblocks, r.stride, &r.words, &GSCALE, &r.rscale, &r.tail)
        .expect("unpooled upload")
}

/// `n * d` values, f16-representable when `f16`, as the rotation hands them over.
fn values(seed: u64, n: usize, f16: bool) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let v = rng.next_gaussian() as f32;
            if f16 {
                half::f16::from_f32(v).to_f32()
            } else {
                v
            }
        })
        .collect()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

fn metal_of(dev: &Device) -> candle_core::MetalDevice {
    match dev {
        Device::Metal(m) => m.clone(),
        _ => panic!("a Metal device"),
    }
}

/// K1: a record and a rotation uploaded outside the pool give the bits of the
/// pooled upload and of the CPU oracle; dropped, an unpooled record frees its
/// memory at once, while a pooled one stays allocated in the pool.
#[test]
fn k1_unpooled_upload_matches_pooled_and_stays_out_of_the_pool() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let rt = runtime(&dev, true);
    for (seed, d_out, nb, tw) in [(0xA1u64, 1152usize, 96usize, 0usize), (0xA2, 2304, 48, 0), (0xA3, 1024, 106, 16)] {
        let r = rec(seed, d_out, nb, tw);
        let (p, u) = (pooled(&rt, &r), unpooled(&rt, &r));
        let x = values(seed ^ 1, r.d_in, true);
        let xm = Tensor::from_slice(&x, (1, r.d_in), &dev).unwrap();
        let xc = Tensor::from_slice(&x, (1, r.d_in), &Device::Cpu).unwrap();
        let (a, b) = (to_vec(&rt.matvec(&p, &xm).unwrap()), to_vec(&rt.matvec(&u, &xm).unwrap()));
        let c = to_vec(&rt.matvec(&u, &xc).expect("the oracle reads the unpooled host copy"));
        assert!(a.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&b, &a), None, "{d_out}x{nb}+{tw}: unpooled against pooled");
        assert_eq!(first_difference(&b, &c), None, "{d_out}x{nb}+{tw}: unpooled against the oracle");
    }
    for (n, seed) in [(2304usize, 11u64), (1152, 12)] {
        let t = RotationTables::build(n, seed).unwrap();
        let (rp, ru) = (rt.upload_rotation(&t).unwrap(), rt.upload_rotation_unpooled(&t).unwrap());
        let x = Tensor::from_slice(&values(seed, 5 * n, false), (5, n), &dev).unwrap();
        let one = x.narrow(0, 0, 1).unwrap().contiguous().unwrap();
        let (a, b) = (to_vec(&rt.rotate(&rp, "r", &one).unwrap()), to_vec(&rt.rotate(&ru, "r", &one).unwrap()));
        assert_eq!(first_difference(&b, &a), None, "rotation {n}, one row");
        let (a, b) = (to_vec(&rt.rotate_rows(&rp, "r", &x, 5).unwrap()), to_vec(&rt.rotate_rows(&ru, "r", &x, 5).unwrap()));
        assert_eq!(first_difference(&b, &a), None, "rotation {n}, five rows");
    }
    // Pool membership, read on the device's allocated size, with no command
    // in between, so no flush can empty the pool.
    let m = metal_of(&dev);
    dev.synchronize().unwrap();
    let r = rec(0xA4, 2304, 48, 0);
    let u = unpooled(&rt, &r);
    let bytes = u.device_bytes() as usize;
    let a0 = m.device().current_allocated_size();
    drop(u);
    let a1 = m.device().current_allocated_size();
    // Metal frees the large buffers by the page and keeps the few small ones
    // (8 B of gain scales) in a shared slab, so the freed size is checked
    // against 90 % of the record (measured: 671,744 of 672,778 B).
    let freed = a0 as i64 - a1 as i64;
    assert!(freed >= (bytes as i64) * 9 / 10, "unpooled: dropping {bytes} B freed {freed} B; the pool still holds it");
    let p = pooled(&rt, &r);
    let b0 = m.device().current_allocated_size();
    drop(p);
    let b1 = m.device().current_allocated_size();
    assert_eq!(b0, b1, "pooled: candle's pool keeps a dropped buffer until its next flush");
}

/// P1: every row of the f32 rows kernel is the served matvec's f32 for that
/// row alone, at 1 to 8 rows and three shapes; the f16 rows kernel of the
/// served tuning still narrows the same values.
#[test]
fn p1_f32_rows_are_the_matvec_rows() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let mut rt = runtime(&dev, false);
    rt.set_tuning(MetalTuning { prefill_rows: 8, ..MetalTuning::default() }).unwrap();
    for (seed, d_out, nb, tw) in [(0xB1u64, 1152usize, 96usize, 0usize), (0xB2, 2304, 48, 0), (0xB3, 1024, 106, 16)] {
        let r = rec(seed, d_out, nb, tw);
        let p = unpooled(&rt, &r);
        for n in 1..=8usize {
            let x = values(seed ^ n as u64, n * r.d_in, true);
            let xm = Tensor::from_slice(&x, (n, r.d_in), &dev).unwrap();
            let y = to_vec(&rt.matvec_rows_f32(&p, &xm, n).unwrap());
            assert!(y.iter().any(|v| *v != 0.0), "zeros would pass every equality");
            for row in 0..n {
                let one = xm.narrow(0, row, 1).unwrap().contiguous().unwrap();
                let want = to_vec(&rt.matvec(&p, &one).unwrap());
                assert_eq!(
                    first_difference(&y[row * d_out..(row + 1) * d_out], &want),
                    None,
                    "{d_out}x{nb}+{tw}, {n} rows, row {row}"
                );
            }
            if n == 8 {
                let h = rt.matvec_rows(&p, &xm, n).unwrap();
                assert_eq!(h.dtype(), DType::F16, "the served f16 rows kernel is unchanged");
                let narrowed: Vec<f32> = y.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
                assert_eq!(first_difference(&to_vec(&h), &narrowed), None, "{d_out}: f16 rows are the f32 rows narrowed");
            }
        }
    }
    assert!(rt.compiled_pipelines().contains(&"tv_tetra48_metal_lutdrf"), "the f32 rows kernel ran");
}

fn config() -> InklingConfig {
    InklingConfig::from_file(&need(data_dir().join("textonly/config.json"))).expect("config")
}

/// `n` synthetic experts at the model's shapes (gate and up 1,152 x 2,304,
/// down 2,304 x 1,152), uploaded outside the pool, as one layer.
fn synthetic_layer(rt: Arc<MetalRuntime>, dev: &Device, cfg: &InklingConfig, n: usize) -> ResidentExperts {
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let experts = (0..n)
        .map(|e| {
            let s = 0xC000 + 8 * e as u64;
            let up = |seed: u64, d_out: usize, d_in: usize| unpooled(&rt, &rec(seed, d_out, d_in / DIM, 0));
            ResidentExpert {
                gate: up(s, i, h),
                up: up(s + 1, i, h),
                down: up(s + 2, h, i),
                rot_in: rt.upload_rotation_unpooled(&RotationTables::build(h, s + 3).unwrap()).unwrap(),
                rot_down: rt.upload_rotation_unpooled(&RotationTables::build(i, s + 4).unwrap()).unwrap(),
            }
        })
        .collect();
    ResidentExperts::from_parts(rt, dev, 6, experts)
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

/// D1: at one row the MoE skips the gather, the scatter and all but one
/// upload, and returns the general path's bits; an expert named twice falls
/// back to the general path.
#[test]
fn d1_one_row_moe_is_the_general_path() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let layer = synthetic_layer(Arc::new(runtime(&dev, false)), &dev, &cfg, 12);
    for seed in 0..4u64 {
        let r = routing(0xD1 + seed, 1, 12, cfg.num_experts_per_tok);
        let x = Tensor::from_slice(&values(seed, cfg.hidden_size, false), (1, cfg.hidden_size), &dev).unwrap();
        let fast = layer.forward_one_row(&x, &r, &cfg).unwrap().expect("distinct experts take the one-row path");
        let general = layer.forward_general(&x, &r, &cfg).unwrap();
        let (a, b) = (to_vec(&fast), to_vec(&general));
        assert!(b.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        assert_eq!(first_difference(&a, &b), None, "routing {seed}: one-row path against the general path");
        assert_eq!(first_difference(&to_vec(&layer.forward(&x, &r, &cfg).unwrap()), &b), None, "forward at one row");
    }
    let mut r = routing(0xD9, 1, 12, cfg.num_experts_per_tok);
    r.indices[0][1] = r.indices[0][0];
    let x = Tensor::from_slice(&values(9, cfg.hidden_size, false), (1, cfg.hidden_size), &dev).unwrap();
    assert!(layer.forward_one_row(&x, &r, &cfg).unwrap().is_none(), "an expert named twice is the general path's");
    let (a, b) = (to_vec(&layer.forward(&x, &r, &cfg).unwrap()), to_vec(&layer.forward_general(&x, &r, &cfg).unwrap()));
    assert_eq!(first_difference(&a, &b), None, "forward with a repeated expert");
}

/// P1 in the MoE: 13 tokens through the layer with 8 rows a launch (f32 rows
/// kernel) and with 1 row a launch (the matvec), the same bits.
#[test]
fn p1_moe_rows8_is_rows1() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt1 = Arc::new(runtime(&dev, false));
    let mut rt8 = runtime(&dev, false);
    rt8.set_tuning(MetalTuning { prefill_rows: 8, ..MetalTuning::default() }).unwrap();
    let rt8 = Arc::new(rt8);
    let mut layer = synthetic_layer(rt1.clone(), &dev, &cfg, 12);
    let t = 13;
    let r = routing(0xE1, t, 12, cfg.num_experts_per_tok);
    let x = Tensor::from_slice(&values(0xE2, t * cfg.hidden_size, false), (t, cfg.hidden_size), &dev).unwrap();
    let one = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let l1 = layer.take_launches();
    layer.set_runtime(rt8.clone());
    let eight = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let l8 = layer.take_launches();
    assert!(one.iter().any(|v| *v != 0.0), "zeros would pass every equality");
    assert_eq!(first_difference(&eight, &one), None, "8 rows a launch against 1");
    assert!(l8 < l1, "{l8} launches at 8 rows against {l1} at 1");
    assert!(rt8.compiled_pipelines().contains(&"tv_tetra48_metal_lutdrf"), "the f32 rows kernel ran");
}

/// A synthetic int4 g64 table, `rows x d`.
fn table(seed: u64, rows: usize, d: usize) -> RawTensor {
    let mut rng = SplitMix64::new(seed);
    let gpr = d.div_ceil(64);
    let packed = (0..rows * d / 2).map(|_| (rng.next() & 0xff) as u8).collect();
    let scales = (0..rows * gpr).map(|_| half::f16::from_f32((0.01 + 0.04 * rng.next_gaussian().abs()) as f32).to_bits()).collect();
    let biases = (0..rows * gpr).map(|_| half::f16::from_f32((0.2 * rng.next_gaussian()) as f32).to_bits()).collect();
    RawTensor {
        name: "model.unembed.weight".into(),
        dims: vec![rows, d],
        data: RawData::Quant(QuantData { bits: 4, group: 64, packed, scales, biases }),
    }
}

/// `tv_emb_q4_metal_xf` on the host, in its order: lane l takes words l,
/// l + 32, ..., eight columns a word with `fma`, then the butterfly.
fn head_oracle(q: &QuantData, d: usize, row: usize, x: &[f32]) -> f32 {
    let (nwords, gpr) = (d / 8, d.div_ceil(64));
    let mut lanes = [0f32; 32];
    for (l, acc) in lanes.iter_mut().enumerate() {
        let mut wi = l;
        while wi < nwords {
            let o = row * d / 2 + wi * 4;
            let p = u32::from_le_bytes([q.packed[o], q.packed[o + 1], q.packed[o + 2], q.packed[o + 3]]);
            let c = wi * 8;
            let s = llvq_artifact::f16_to_f32(q.scales[row * gpr + c / 64]);
            let b = llvq_artifact::f16_to_f32(q.biases[row * gpr + c / 64]);
            for k in 0..8 {
                let w = s * ((p >> (4 * k)) & 0xf) as f32 + b;
                *acc = w.mul_add(x[c + k], *acc);
            }
            wi += 32;
        }
    }
    for k in [16usize, 8, 4, 2, 1] {
        let mut next = [0f32; 32];
        for (l, n) in next.iter_mut().enumerate() {
            *n = lanes[l] + lanes[l ^ k];
        }
        lanes = next;
    }
    lanes[0]
}

/// `max |a - b| / max |b|` and whether every row's argmax agrees.
fn compare_rows(a: &[f32], b: &[f32], rows: usize) -> (f64, bool) {
    let v = b.len() / rows;
    let scale = b.iter().fold(0f64, |m, x| m.max(x.abs() as f64));
    let gap = a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs() as f64));
    let agree = (0..rows).all(|r| argmax(&a[r * v..(r + 1) * v]) == argmax(&b[r * v..(r + 1) * v]));
    (gap / scale, agree)
}

/// The plain f32 product `h W^T` of the decoded table's first `vocab` rows.
fn plain_product(t: &RawTensor, vocab: usize, h: &[f32], rows: usize) -> Vec<f32> {
    let d = t.dims[1];
    let w = t.to_f32();
    let w = Tensor::from_slice(&w[..vocab * d], (vocab, d), &Device::Cpu).unwrap();
    let x = Tensor::from_slice(h, (rows, d), &Device::Cpu).unwrap();
    to_vec(&x.matmul(&w.t().unwrap()).unwrap())
}

/// D4: the int4 head on Metal equals its host oracle bit for bit, sits at f32
/// rounding from the plain product of the decoded table, and refuses a table
/// it cannot read.
#[test]
fn d4_q4_head_matches_its_oracle_and_the_plain_product() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let rt = Arc::new(runtime(&dev, false));
    let (rows, d, vocab, t_rows) = (1032usize, 2304usize, 1002usize, 3usize);
    let t = table(0xF1, rows, d);
    let head = Q4MetalHead::from_raw(rt.clone(), &t, vocab, d).expect("the head loads");
    assert_eq!(head.vocab(), vocab);
    let h = values(0xF2, t_rows * d, false);
    let hm = Tensor::from_slice(&h, (t_rows, d), &dev).unwrap();
    let y = head.project(&hm).unwrap();
    assert_eq!(y.dims(), [t_rows, vocab]);
    assert_eq!(y.dtype(), DType::F32);
    let y = to_vec(&y);
    let RawData::Quant(q) = &t.data else { unreachable!() };
    let oracle: Vec<f32> =
        (0..t_rows).flat_map(|r| (0..vocab).map(move |v| (r, v))).map(|(r, v)| head_oracle(q, d, v, &h[r * d..(r + 1) * d])).collect();
    assert_eq!(first_difference(&y, &oracle), None, "the kernel against its host oracle");
    let plain = plain_product(&t, vocab, &h, t_rows);
    let (gap, agree) = compare_rows(&y, &plain, t_rows);
    println!("synthetic head: largest relative gap to the plain product {gap:.3e}, argmax agree {agree}");
    assert!(agree, "argmax against the plain product");
    assert!(gap < 1e-5, "relative gap {gap:.3e} to the plain product");

    assert!(Q4MetalHead::from_raw(rt.clone(), &t, 1030, d).is_ok(), "1,030 rows pad to 1,032, which the table has");
    assert!(Q4MetalHead::from_raw(rt.clone(), &t, 1033, d).is_err(), "1,033 rows pad to 1,040, past the table");
    assert!(Q4MetalHead::from_raw(rt.clone(), &t, vocab, 2560).is_err(), "a row of another width");
    let mut g128 = table(0xF3, rows, d);
    if let RawData::Quant(q) = &mut g128.data {
        q.group = 128;
    }
    assert!(Q4MetalHead::from_raw(rt.clone(), &g128, vocab, d).is_err(), "groups of 128");
    let mut i8 = table(0xF4, rows, d);
    if let RawData::Quant(q) = &mut i8.data {
        q.bits = 8;
    }
    assert!(Q4MetalHead::from_raw(rt, &i8, vocab, d).is_err(), "an int8 table");
}

// ---------------------------------------------------------------------------
// Real-model gates and timings (ignored: run under the guard's lock).
// ---------------------------------------------------------------------------

/// One layer's resident experts behind a lock, so the same buffers can be
/// run under another runtime.
struct Shared(Arc<RwLock<ResidentExperts>>);

impl MoeSource for Shared {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> candle_core::Result<Tensor> {
        self.0.read().expect("no panic held this lock").forward(x, routing, cfg)
    }
    fn take_launches(&self) -> usize {
        self.0.read().expect("no panic held this lock").take_launches()
    }
}

/// Every verified layer resident, pooled or not, each behind a [`Shared`].
fn resident_shared(m: &mut Inkling, dev: &Device, rt: Arc<MetalRuntime>, pooled: bool) -> Vec<Arc<RwLock<ResidentExperts>>> {
    let dir = need(data_dir().join("layers"));
    let mut out = vec![];
    for l in resident::verified_layers(&dir, m.cfg.num_hidden_layers) {
        let ld = dir.join(format!("L{l:02}"));
        let r = if pooled {
            ResidentExperts::load_pooled(rt.clone(), dev, &ld, l, &m.cfg)
        } else {
            ResidentExperts::load(rt.clone(), dev, &ld, l, &m.cfg)
        }
        .expect("resident layer");
        let a = Arc::new(RwLock::new(r));
        m.set_experts(l, LayerExperts::Source(Box::new(Shared(a.clone()))));
        out.push(a);
    }
    out
}

fn write_out(name: &str, j: &serde_json::Value) {
    let d = data_dir().join("lot3-k/runs");
    std::fs::create_dir_all(&d).unwrap();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    std::fs::write(d.join(format!("{name}-{ts}.json")), serde_json::to_string_pretty(j).unwrap()).unwrap();
    println!("{j}");
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// K1 timing: the 32-token prompt, then 8 timed decode steps (each ends on
/// reading its logits back), every verified layer resident in or out of the
/// pool. Run the two tests alternately in separate processes: the pool's
/// cost is the pool's content, so the two arms cannot share one.
fn decode_timing(pooled: bool) {
    let dev = Device::new_metal(0).expect("Metal");
    let mut m = model(&dev, false);
    let t = Instant::now();
    let layers = resident_shared(&mut m, &dev, resident::served_runtime(&dev).unwrap(), pooled);
    let load_s = t.elapsed().as_secs_f64();
    let buffers: usize = layers.iter().map(|l| l.read().unwrap().stats.buffers).sum();
    let prompt = prompt32();
    let mut cache = m.new_cache();
    let mut v = to_vec(&m.forward(&prompt, &mut cache, false, false).unwrap().logits);
    let mut secs = vec![];
    let mut tokens = vec![];
    for _ in 0..8 {
        let tok = argmax(&v);
        tokens.push(tok);
        let t = Instant::now();
        v = to_vec(&m.forward(&[tok], &mut cache, false, false).unwrap().logits);
        secs.push(t.elapsed().as_secs_f64());
    }
    let arm = if pooled { "pooled" } else { "unpooled" };
    write_out(
        &format!("k1_decode_{arm}"),
        &serde_json::json!({"arm": arm, "binary": tag(), "resident_layers": layers.len(), "expert_buffers": buffers,
            "seconds_load_experts": load_s, "seconds_steps": secs, "median_step": median(secs.clone()), "tokens": tokens}),
    );
}

#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn k1_decode_timing_pooled() {
    decode_timing(true);
}

#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn k1_decode_timing_unpooled() {
    decode_timing(false);
}

/// P1 on the model: the logits with 8 rows a launch against 1 row a launch,
/// on every position of the 32-token prompt and on the last position of the
/// 640- and 2,048-token prompts, bit for bit; the 2,048-token prefill timed
/// three times each way, interleaved.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn p1_model_rows8_matches_rows1() {
    let dev = Device::new_metal(0).expect("Metal");
    let mut m = model(&dev, false);
    let rt1 = resident::served_runtime(&dev).unwrap();
    let tables = llvq_llm::fused::tetra48_tables(&llvq_search::tetra::Tetra::new());
    let mut rt8 = MetalRuntime::from_tables(&dev, &tables, resident::SERVED_TILE).unwrap();
    rt8.set_tuning(MetalTuning { prefill_rows: 8, ..MetalTuning::default() }).unwrap();
    let rt8 = Arc::new(rt8);
    let layers = resident_shared(&mut m, &dev, rt1.clone(), false);
    let set = |rt: &Arc<MetalRuntime>| {
        for l in &layers {
            l.write().unwrap().set_runtime(rt.clone());
        }
    };
    let run = |ids: &[u32], all: bool| -> (Vec<f32>, f64, usize) {
        dev.synchronize().unwrap();
        let t = Instant::now();
        let v = to_vec(&m.forward(ids, &mut m.new_cache(), false, all).unwrap().logits);
        (v, t.elapsed().as_secs_f64(), m.take_launches())
    };
    let mut j = serde_json::Map::new();
    for (label, ids, all) in [("p32_all", prompt32(), true), ("p640_last", window(LONG_PROMPT), false)] {
        set(&rt1);
        let (a, ta, la) = run(&ids, all);
        set(&rt8);
        let (b, tb, lb) = run(&ids, all);
        let diff = first_difference(&b, &a);
        println!("{label}: rows 8 against rows 1: first difference {diff:?}; {ta:.2} s / {tb:.2} s; launches {la} / {lb}");
        j.insert(label.into(), serde_json::json!({"bit_equal": diff.is_none(), "first_difference": format!("{diff:?}"),
            "values": a.len(), "seconds_rows1": ta, "seconds_rows8": tb, "launches_rows1": la, "launches_rows8": lb}));
    }
    let ids = window(2048);
    let (mut t1, mut t8, mut equal) = (vec![], vec![], true);
    let mut reference: Option<Vec<f32>> = None;
    for _ in 0..3 {
        for (rt, times) in [(&rt1, &mut t1), (&rt8, &mut t8)] {
            set(rt);
            let (v, s, _) = run(&ids, false);
            times.push(s);
            match &reference {
                None => reference = Some(v),
                Some(r) => equal &= first_difference(&v, r).is_none(),
            }
        }
    }
    println!("2,048 tokens: rows 1 {t1:?} s, rows 8 {t8:?} s; last logits equal in all six rounds: {equal}");
    j.insert("p2048_last".into(), serde_json::json!({"bit_equal_all_rounds": equal, "seconds_rows1": t1,
        "seconds_rows8": t8, "median_rows1": median(t1.clone()), "median_rows8": median(t8.clone())}));
    let j = serde_json::Value::Object(j);
    write_out("p1_model_rows8_vs_rows1", &j);
    assert!(j["p32_all"]["bit_equal"].as_bool().unwrap(), "32-token logits differ");
    assert!(j["p640_last"]["bit_equal"].as_bool().unwrap(), "640-token logits differ");
    assert!(equal, "2,048-token logits differ");
}

/// One head call: the hidden state, the logits, the seconds.
type HeadCall = (Vec<f32>, Vec<f32>, f64);

/// The int4 head behind a recorder: what it was asked, what it answered,
/// and how long it took (device drained on both sides).
struct Recorder {
    inner: Q4MetalHead,
    dev: Device,
    calls: Arc<Mutex<Vec<HeadCall>>>,
}

impl OutputProjection for Recorder {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    fn device_bytes(&self) -> u64 {
        self.inner.device_bytes()
    }
    fn project(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        self.dev.synchronize()?;
        let t = Instant::now();
        let y = self.inner.project(h)?;
        self.dev.synchronize()?;
        let s = t.elapsed().as_secs_f64();
        self.calls.lock().unwrap().push((to_vec(h), to_vec(&y), s));
        Ok(y)
    }
}

/// D4 on the model: prefill and 16 greedy steps through the int4 head on
/// both prompts; at every step the head's logits against the plain f32
/// product of the same table decoded (argmax equal, largest relative gap at
/// f32 rounding); the head's time against the BF16 head's; device memory.
#[test]
#[ignore = "loads the real model on the GPU; run under the guard's lock"]
fn d4_model_q4_head() {
    let dev = Device::new_metal(0).expect("Metal");
    let m_dev = metal_of(&dev);
    let mut m = model(&dev, false);
    let rt = resident::served_runtime(&dev).unwrap();
    let _layers = resident_shared(&mut m, &dev, rt.clone(), false);
    dev.synchronize().unwrap();
    let a0 = m_dev.device().current_allocated_size();
    let path = need(data_dir().join("lot1-b/tables/unembed.q4g64.llvqraw"));
    let raw = llvq_llm::inkseal::read_raw_file(&path).expect("table");
    let (vocab, hidden) = (m.cfg.unpadded_vocab_size, m.cfg.hidden_size);
    let head = Q4MetalHead::from_raw(rt, &raw, vocab, hidden).expect("head");
    dev.synchronize().unwrap();
    let a1 = m_dev.device().current_allocated_size();
    let bytes = head.device_bytes();
    let log = Arc::new(Mutex::new(vec![]));
    m.out_proj = Some(Box::new(Recorder { inner: head, dev: dev.clone(), calls: log.clone() }));
    let mut steps = 0;
    for ids in [prompt32(), window(LONG_PROMPT)] {
        let mut cache = m.new_cache();
        let mut v = to_vec(&m.forward(&ids, &mut cache, false, false).unwrap().logits);
        for _ in 0..DECODE_STEPS {
            v = to_vec(&m.forward(&[argmax(&v)], &mut cache, false, false).unwrap().logits);
        }
        steps += DECODE_STEPS + 1;
    }
    m.out_proj = None;
    let calls = std::mem::take(&mut *log.lock().unwrap());
    assert_eq!(calls.len(), steps, "one head call a step");
    let w = raw.to_f32();
    let w = Tensor::from_slice(&w[..vocab * hidden], (vocab, hidden), &Device::Cpu).unwrap();
    let (mut worst, mut agree) = (0f64, 0usize);
    for (h, y, _) in &calls {
        let x = Tensor::from_slice(h, (1, hidden), &Device::Cpu).unwrap();
        let plain = to_vec(&x.matmul(&w.t().unwrap()).unwrap());
        let (gap, same) = compare_rows(y, &plain, 1);
        worst = worst.max(gap);
        agree += usize::from(same);
    }
    // The BF16 head, the chunked widening of `Inkling::head`, on the same states.
    let unembed = m.checkpoint().raw("unembed.weight", &dev).unwrap().narrow(0, 0, vocab).unwrap();
    let mut bf16 = vec![];
    for (h, _, _) in calls.iter().take(8) {
        let x = Tensor::from_slice(h, (1, hidden), &dev).unwrap();
        dev.synchronize().unwrap();
        let t = Instant::now();
        let mut parts = vec![];
        let mut start = 0;
        while start < vocab {
            let n = 16384.min(vocab - start);
            let wc = unembed.narrow(0, start, n).unwrap().to_dtype(DType::F32).unwrap();
            parts.push(x.matmul(&wc.t().unwrap()).unwrap());
            start += n;
        }
        let _ = Tensor::cat(&parts, 1).unwrap();
        dev.synchronize().unwrap();
        bf16.push(t.elapsed().as_secs_f64());
    }
    let q4: Vec<f64> = calls.iter().map(|c| c.2).collect();
    let j = serde_json::json!({
        "steps": calls.len(), "argmax_equal": agree, "worst_relative_gap": worst,
        "seconds_q4_head": q4, "median_q4_head": median(q4.clone()),
        "seconds_bf16_head": bf16, "median_bf16_head": median(bf16.clone()),
        "allocated_before_head": a0, "allocated_after_head": a1, "head_device_bytes": bytes,
    });
    write_out("d4_model_q4_head", &j);
    assert_eq!(agree, calls.len(), "argmax against the plain product");
    assert!(worst < 1e-5, "relative gap {worst:.3e}");
}
