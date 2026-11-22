//! `llvq_llm::extq` against the block loop it repeats.
//!
//! The claim `bin/extq` rests on: for one `(W, H)`, the external chain
//! writes the reconstruction `calib::quantize_model_capturing` writes, bit
//! for bit, under the same rotation and settings. The route is the cheapest
//! real one: a two-block tiny Qwen3, the loop run on block 0 only (whose
//! input is the unquantized calibration set, so its `H` is the one a plain
//! capture accumulates), and the same `H`, captured the way
//! `tests/l36_capture.rs` captures it, handed to the external chain.
//!
//! Two negative controls follow: one setting moved (the damping, the
//! rotation seed) must break the equality, or the equality proves nothing.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::calib::{
    effective_rotation_seed, quantize_model_capturing, Codebook, Hessian, MatrixSink, RunConfig,
};
use llvq_llm::extq::{encode_matrix, factor_hessian, Settings};
use llvq_llm::model::{Act, Capture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};
use llvq_quant::quantizer::TetraShapeGain;
use std::collections::HashMap;

const ROT: u64 = 0x11_0FEED;

/// Widths 48 and 96: whole Leech blocks, as the external shapes are (2304
/// and 1152 are both multiples of 24), so the comparison covers the code
/// path and not only a tail. A second config with tails is run as well.
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
            let data: Vec<f32> = (0..n)
                .map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1)
                .collect();
            Tensor::from_vec(data, (1, 32, c.hidden_size), dev).expect("window")
        })
        .collect()
}

fn tetra() -> Codebook {
    Codebook::Tetra {
        gain_bits: 1,
        post_shape_gain: false,
    }
}

fn run_config(threads: usize) -> RunConfig {
    let s = Settings::sealed_recipe(threads);
    RunConfig {
        sequential_block: false,
        h_shrink: 1.0,
        gain_scale: s.gain_scale,
        int4_types: Vec::new(),
        gptq: s.gptq,
        damping: s.damping,
        codebook: s.codebook,
        threads,
        start: 0,
        limit: 1,
        rotation_seed: Some(ROT),
    }
}

/// Keeps what the sink receives, keyed by matrix name.
#[derive(Default)]
struct Keep {
    got: HashMap<String, (Vec<f64>, Vec<f64>)>,
}

impl MatrixSink for Keep {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        self.got.insert(m.name.clone(), (m.row_scales, m.centroids));
        Ok(())
    }
    fn push_int4(&mut self, _: llvq_artifact::Int4Matrix) -> anyhow::Result<()> {
        unreachable!("no int4 type on this run")
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
    map: VarMap,
    /// Block 0's original weights, its natural-basis `H` per activation.
    orig: HashMap<&'static str, (usize, usize, Vec<f32>)>,
    hess: HashMap<Act, Vec<f64>>,
    /// What the loop left in block 0, and what its sink received.
    looped: HashMap<&'static str, Vec<f32>>,
    sink: Keep,
}

fn case(hidden: usize, inter: usize, threads: usize) -> Case {
    let dev = Device::Cpu;
    let c = tiny(hidden, inter);
    let map = VarMap::new();
    let model = fresh(&c, &map, &dev);
    let hidden_states = windows(&c, &dev);
    let total_rows: usize = hidden_states.iter().map(|h| h.dim(1).unwrap()).sum();
    let mut r = Ref { acc: HashMap::new() };
    for act in Act::ALL {
        r.acc.insert(act, Hessian::new(act.width(&c), &dev, total_rows).unwrap());
    }
    let mask = model.causal_mask_for(&hidden_states[0]).unwrap();
    for h in &hidden_states {
        model.blocks[0].forward(h, model.rotary(), &mask, 0, &mut r).unwrap();
    }
    let hess: HashMap<Act, Vec<f64>> =
        r.acc.into_iter().map(|(a, h)| (a, h.to_f64().unwrap())).collect();
    let mut orig = HashMap::new();
    for act in Act::ALL {
        for name in act.consumers() {
            orig.insert(*name, weight(&model, name));
        }
    }
    let mut model = fresh(&c, &map, &dev);
    let mut hs = windows(&c, &dev);
    let mut sink = Keep::default();
    quantize_model_capturing(&mut model, &mut hs, &run_config(threads), |_, _, _| {}, Some(&mut sink))
        .expect("the loop runs");
    let mut looped = HashMap::new();
    for act in Act::ALL {
        for name in act.consumers() {
            looped.insert(*name, weight(&model, name).2);
        }
    }
    Case {
        c,
        map,
        orig,
        hess,
        looped,
        sink,
    }
}

/// Runs the external chain on every matrix of block 0 and returns the number
/// of matrices whose reconstruction differs from the loop's in any bit.
fn mismatches(k: &Case, s: &Settings, seed_of: impl Fn(Act) -> u64) -> usize {
    let _ = (&k.c, &k.map);
    let enc = TetraShapeGain::encoder();
    let mut bad = 0;
    for act in Act::ALL {
        let n = act.width(&k.c);
        let f = factor_hessian(k.hess[&act].clone(), n, Some(seed_of(act)), s.damping).unwrap();
        for name in act.consumers() {
            let (o, i, w) = &k.orig[name];
            let e = encode_matrix(w, *o, *i, &f, s, &enc).expect("external chain");
            let same = e.recon.iter().zip(&k.looped[name]).all(|(a, b)| a.to_bits() == b.to_bits());
            let key = llvq_llm::artifact::key(0, name);
            let (rs, cs) = &k.sink.got[&key];
            let side = e.row_scales == *rs && e.centroids == *cs;
            if !(same && side) {
                bad += 1;
            }
        }
    }
    bad
}

fn loop_seed(act: Act) -> u64 {
    effective_rotation_seed(ROT, 0, act)
}

#[test]
fn external_chain_matches_the_block_loop_bit_for_bit() {
    for (hidden, inter, threads) in [(48, 96, 1), (48, 96, 3), (32, 64, 2)] {
        let k = case(hidden, inter, threads);
        // The loop really changed the weights, or equality is vacuous.
        let moved = k.looped["mlp.gate_proj"] != k.orig["mlp.gate_proj"].2;
        assert!(moved, "the loop left gate_proj untouched");
        let s = Settings::sealed_recipe(threads);
        let bad = mismatches(&k, &s, loop_seed);
        assert_eq!(bad, 0, "{hidden}/{inter}, {threads} threads: {bad} of 7 matrices differ");
    }
}

#[test]
fn a_different_damping_breaks_the_equality() {
    let k = case(48, 96, 2);
    let mut s = Settings::sealed_recipe(2);
    s.damping = 2e-2;
    let bad = mismatches(&k, &s, loop_seed);
    assert!(bad > 0, "damping 2e-2 reproduced the 1e-2 loop on all 7 matrices");
    eprintln!("damping 2e-2: {bad} of 7 matrices differ");
}

#[test]
fn a_different_rotation_seed_breaks_the_equality() {
    let k = case(48, 96, 2);
    let s = Settings::sealed_recipe(2);
    let bad = mismatches(&k, &s, |a| loop_seed(a) ^ 1);
    assert_eq!(bad, 7, "a shifted seed reproduced the loop on {} matrices", 7 - bad);
}

/// Synthetic sanity at an Inkling width: i.i.d. Gaussian `W`, `H = I`, the
/// rotation on. With `H = I` the error feedback has nothing to carry, so the
/// relative MSE is the code's own Gaussian distortion, which
/// `docs/mesures/innovation-tetra-v2-forme-2026-11-11.txt` line 54-55 gives
/// as 0.0851 per dimension for Tetra with the served 1-bit gain.
#[test]
fn gaussian_w_identity_h_gives_the_documented_tetra_distortion() {
    let (d_out, d_in) = (192, 2304);
    let mut z = 0x1234_5678_u64;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let w: Vec<f32> = (0..d_out * d_in)
        .map(|_| {
            let (u, v) = (next().max(1e-300), next());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect();
    let mut h = vec![0.0f64; d_in * d_in];
    for i in 0..d_in {
        h[i * d_in + i] = 1.0;
    }
    let s = Settings::sealed_recipe(4);
    let f = factor_hessian(h.clone(), d_in, Some(ROT), s.damping).unwrap();
    let e = encode_matrix(&w, d_out, d_in, &f, &s, &TetraShapeGain::encoder()).unwrap();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in w.iter().zip(&e.recon) {
        num += ((*b - *a) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    let rel = num / den;
    let (pn, pd) = llvq_llm::extq::proxy_traces(&w, &e.recon, d_out, d_in, &h, 4);
    let bpw = (d_out * (d_in / 24) * 48 + d_out * 32) as f64 / (d_out * d_in) as f64;
    eprintln!("gaussian H=I: rel MSE {rel:.5}, proxy_rel {:.5}, kernel b/w {bpw:.5}", pn / pd);
    assert!((pn / pd - rel).abs() < 1e-9, "with H = I the proxy is the MSE");
    assert!((0.080..0.090).contains(&rel), "relative MSE {rel} far from the documented 0.0851");
}

/// The settings are the sealed recipe's, field by field, so a later edit of
/// `Settings::sealed_recipe` cannot drift silently.
#[test]
fn sealed_recipe_is_tetra1_nogs_rot_damping_1e2() {
    let s = Settings::sealed_recipe(4);
    assert!(matches!(s.codebook, Codebook::Tetra { gain_bits: 1, post_shape_gain: false }));
    assert_eq!(s.codebook.block_bits(), tetra().block_bits());
    assert_eq!(s.damping, 1e-2);
    assert_eq!(s.gain_scale, 1.0);
    let g: GptqConfig = s.gptq;
    assert!(g.retract && !g.group_scales && !g.design_c && !g.spherical_feedback);
    assert_eq!(g.block, 24);
    assert_eq!(g.lambda, 1e-2);
    assert_eq!(g.tail, TailPolicy::KeepExact);
}
