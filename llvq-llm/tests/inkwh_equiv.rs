//! `inkseal::run_wh` (the `(W, H)` driver of `bin/inkwh`) against the block
//! loop, the way `tests/extq_equiv.rs` holds `extq` to it.
//!
//! The route is the same: a two-block tiny Qwen3, the loop run on block 0
//! only, whose `H` is the one a plain capture of block 0's input
//! accumulates. Here the matrices and the Hessians go through **files**, as
//! the driver reads them: `W` as f32, `H` as the unnormalised sum `H · S`
//! with its weight sum `S` in the manifest (`S` a power of two, so `H · S / S`
//! is `H` to the bit), one manifest per activation, every consumer of that
//! activation in one group. The claim is that the record each matrix leaves
//! is byte for byte the record the loop's sink writes for it, and its
//! reconstruction the weight the loop leaves in the model, bit for bit; for
//! Tetra and for int4 g128 alike.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::calib::{effective_rotation_seed, quantize_model_capturing, Hessian, MatrixSink, RunConfig};
use llvq_llm::extq::Settings;
use llvq_llm::model::{Act, Capture, Qwen3};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const ROT: u64 = 0x11_0FEED;
/// The weight sum the files carry: a power of two, so dividing it back out
/// restores the loop's `H` exactly.
const WSUM: f64 = 128.0;

fn tiny(hidden: usize, inter: usize) -> Config {
    Config {
        vocab_size: 128,
        hidden_size: hidden,
        intermediate_size: inter,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: hidden / 4,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn fresh(c: &Config, map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(c, vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

fn windows(c: &Config, dev: &Device) -> Vec<Tensor> {
    (0..4)
        .map(|w| {
            let n = 32 * c.hidden_size;
            let data: Vec<f32> = (0..n).map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1).collect();
            Tensor::from_vec(data, (1, 32, c.hidden_size), dev).expect("window")
        })
        .collect()
}

/// What the loop's sink receives, as the bytes of the record a file sink
/// writes (`ArtifactWriter::push` / `push_int4`, format v5).
#[derive(Default)]
struct Bytes {
    got: HashMap<String, Vec<u8>>,
}

impl MatrixSink for Bytes {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        let tb = llvq_artifact::Codebook::new(llvq_artifact::CodeKind::Tetra)?;
        let mut b = Vec::new();
        llvq_artifact::write_matrix_with(&mut b, llvq_artifact::VERSION, &tb, &m)?;
        self.got.insert(m.name.clone(), b);
        Ok(())
    }
    fn push_int4(&mut self, m: llvq_artifact::Int4Matrix) -> anyhow::Result<()> {
        let mut b = Vec::new();
        llvq_artifact::write_matrix_int4(&mut b, llvq_artifact::VERSION, &m)?;
        self.got.insert(m.name.clone(), b);
        Ok(())
    }
}

struct Ref {
    acc: HashMap<Act, Hessian>,
}

impl Capture for Ref {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == 0 {
            if let Some(h) = self.acc.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

fn weight(model: &Qwen3, name: &str) -> (usize, usize, Vec<f32>) {
    let w = model.blocks[0].linear(name).weight();
    let (o, i) = w.dims2().unwrap();
    (o, i, w.flatten_all().unwrap().to_vec1::<f32>().unwrap())
}

struct Case {
    c: Config,
    orig: HashMap<&'static str, (usize, usize, Vec<f32>)>,
    hess: HashMap<Act, Vec<f64>>,
    looped: HashMap<&'static str, Vec<f32>>,
    sink: Bytes,
}

fn case(hidden: usize, inter: usize, threads: usize, int4: &[&str]) -> Case {
    let dev = Device::Cpu;
    let c = tiny(hidden, inter);
    let map = VarMap::new();
    let model = fresh(&c, &map, &dev);
    let hs = windows(&c, &dev);
    let rows: usize = hs.iter().map(|h| h.dim(1).unwrap()).sum();
    let mut r = Ref { acc: HashMap::new() };
    for act in Act::ALL {
        r.acc.insert(act, Hessian::new(act.width(&c), &dev, rows).unwrap());
    }
    let mask = model.causal_mask_for(&hs[0]).unwrap();
    for h in &hs {
        model.blocks[0].forward(h, model.rotary(), &mask, 0, &mut r).unwrap();
    }
    let hess = r.acc.into_iter().map(|(a, h)| (a, h.to_f64().unwrap())).collect();
    let mut orig = HashMap::new();
    for act in Act::ALL {
        for name in act.consumers() {
            orig.insert(*name, weight(&model, name));
        }
    }
    let mut model = fresh(&c, &map, &dev);
    let mut hs = windows(&c, &dev);
    let s = Settings::sealed_recipe(threads);
    let cfg = RunConfig {
        sequential_block: false,
        h_shrink: 1.0,
        gain_scale: s.gain_scale,
        int4_types: int4.iter().map(|t| t.to_string()).collect(),
        gptq: s.gptq,
        damping: s.damping,
        codebook: s.codebook,
        threads,
        start: 0,
        limit: 1,
        rotation_seed: Some(ROT),
    };
    let mut sink = Bytes::default();
    quantize_model_capturing(&mut model, &mut hs, &cfg, |_, _, _| {}, Some(&mut sink)).expect("the loop runs");
    let mut looped = HashMap::new();
    for act in Act::ALL {
        for name in act.consumers() {
            looped.insert(*name, weight(&model, name).2);
        }
    }
    Case { c, orig, hess, looped, sink }
}

fn put<T: Copy>(path: &Path, v: &[T], f: impl Fn(T) -> Vec<u8>) {
    let b: Vec<u8> = v.iter().flat_map(|x| f(*x)).collect();
    std::fs::write(path, b).unwrap();
}

/// Writes the files of every activation of block 0, runs the driver on each
/// manifest, and returns the matrices whose record or reconstruction differs
/// from the loop's in any byte, with the kind each was given.
fn mismatches(k: &Case, tag: &str, int4: &[&str], threads: usize, seed_of: impl Fn(Act) -> u64) -> Vec<String> {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inkwh").join(tag);
    let _ = std::fs::remove_dir_all(&root);
    let mut bad = Vec::new();
    for act in Act::ALL {
        let n = act.width(&k.c);
        let dir = root.join(format!("{act:?}"));
        std::fs::create_dir_all(&dir).unwrap();
        let h: Vec<f64> = k.hess[&act].iter().map(|v| v * WSUM).collect();
        put(&dir.join("h.f64"), &h, |v: f64| v.to_le_bytes().to_vec());
        let mut mats = Vec::new();
        for name in act.consumers() {
            let (o, i, w) = &k.orig[name];
            assert_eq!(*i, n);
            let file = format!("{}.f32", name.replace('.', "_"));
            put(&dir.join(&file), w, |v: f32| v.to_le_bytes().to_vec());
            let short = name.rsplit('.').next().unwrap();
            let kind = if int4.contains(&short) { "int4" } else { "tetra" };
            mats.push(json!({"name": llvq_llm::artifact::key(0, name), "w": file, "d_out": o, "kind": kind}));
        }
        let manifest = json!({
            "n": n, "h": "h.f64", "weight_sum": WSUM, "tokens": 128,
            "rotation_seed": seed_of(act), "matrices": mats,
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        let out = dir.join("out");
        llvq_llm::inkseal::run_wh(&dir.join("manifest.json"), &out, threads).expect("driver runs");
        for name in act.consumers() {
            let key = llvq_llm::artifact::key(0, name);
            let rec = std::fs::read(out.join(format!("{key}.llvqrec"))).unwrap();
            let recon: Vec<f32> = std::fs::read(out.join(format!("{key}.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let same_rec = k.sink.got.get(&key) == Some(&rec);
            let same_w = recon.len() == k.looped[name].len()
                && recon.iter().zip(&k.looped[name]).all(|(a, b)| a.to_bits() == b.to_bits());
            if !(same_rec && same_w) {
                bad.push(format!("{name} (record {same_rec}, weights {same_w})"));
            }
        }
    }
    bad
}

fn loop_seed(act: Act) -> u64 {
    effective_rotation_seed(ROT, 0, act)
}

#[test]
fn the_driver_matches_the_block_loop_byte_for_byte_tetra() {
    // 48/96: whole Leech blocks; 32/64: widths with a tail (32 = 24 + 8).
    for (hidden, inter, threads) in [(48, 96, 1), (48, 96, 3), (32, 64, 2)] {
        let k = case(hidden, inter, threads, &[]);
        assert!(k.looped["mlp.gate_proj"] != k.orig["mlp.gate_proj"].2, "the loop left gate_proj untouched");
        let bad = mismatches(&k, &format!("tetra-{hidden}-{threads}"), &[], threads, loop_seed);
        assert!(bad.is_empty(), "{hidden}/{inter}, {threads} threads: {bad:?}");
    }
}

#[test]
fn the_driver_matches_the_block_loop_byte_for_byte_int4() {
    // Widths that are multiples of 128, as an int4 g128 record needs; v_proj
    // and down_proj go int4, the rest of their groups stay Tetra.
    let int4 = ["v_proj", "down_proj"];
    let k = case(128, 256, 2, &int4);
    let bad = mismatches(&k, "int4", &int4, 2, loop_seed);
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn a_shifted_seed_breaks_the_equality() {
    let k = case(48, 96, 2, &[]);
    let bad = mismatches(&k, "seed", &[], 2, |a| loop_seed(a) ^ 1);
    assert_eq!(bad.len(), 7, "a shifted seed reproduced the loop on {} matrices", 7 - bad.len());
}

#[test]
fn an_int4_width_off_the_group_is_refused_by_name() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inkwh").join("refuse");
    std::fs::create_dir_all(&dir).unwrap();
    put(&dir.join("w.f32"), &vec![0.5f32; 4 * 96], |v: f32| v.to_le_bytes().to_vec());
    let m = json!({"n": 96, "matrices": [{"name": "x", "w": "w.f32", "d_out": 4, "kind": "int4"}]});
    std::fs::write(dir.join("manifest.json"), m.to_string()).unwrap();
    let e = llvq_llm::inkseal::run_wh(&dir.join("manifest.json"), &dir.join("out"), 1).unwrap_err();
    assert!(format!("{e:#}").contains("not a multiple of the int4 group"), "{e:#}");
}
