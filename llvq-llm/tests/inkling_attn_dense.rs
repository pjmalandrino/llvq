//! The attention products of `inkling::attn_dense` against the functions of
//! `inkling/layers/`, run on the CPU and on Metal.
//!
//! The reference for `stage_in` is the statement sequence of
//! `AttnWeights::forward_chunked` up to the score step, written here with the
//! public functions it calls (`linear`, `l2_clip`, `log_scale`,
//! `rms_unweighted`, `short_conv`, `dynamic_conv`); the reference for
//! `stage_out` is its last line. The end-to-end test runs
//! `AttnWeights::forward` itself against `stage_in`, a copy of its score
//! statements, and `stage_out`.
//!
//! Two kinds of inputs. Random BF16 weights and normal activations: every
//! output is class B (the products sum in another order), held to a relative
//! 1e-5. Products exact in f32 (activations and weights in {-1, 0, 1},
//! scales powers of two): the outputs after the products must then be bit for
//! bit those of the CPU reference, which proves the elementwise steps, the
//! sums of squares and the rings follow the reference's order.
//!
//! Tiny and synthetic: not a heavy job. `attn_dense_timing` is ignored by
//! default and run by name.
#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, Result, Tensor, D};
use llvq_llm::inkling::attn_dense::{tau, AttnDense, AttnDenseWeights, KvRings, StageIn};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::{
    dynamic_conv, l2_clip, linear, log_scale, rms_unweighted, short_conv, AttnWeights, ConvState, KvConv, SlotCache,
};

#[derive(Clone, Copy, Debug)]
struct Dims {
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    dr: usize,
    ext: usize,
    taps: usize,
}

const REAL: Dims = Dims { h: 2304, nh: 24, nkv: 8, hd: 128, dr: 16, ext: 1024, taps: 4 };
const TINY: Dims = Dims { h: 48, nh: 4, nkv: 2, hd: 8, dr: 4, ext: 16, taps: 4 };

/// The class B bound of the brief.
const B_REL: f64 = 1e-5;

/// The real config, with the relative width of `d` (the only field the
/// reference reads that the tiny shapes change).
fn cfg_for(d: Dims) -> InklingConfig {
    let mut c = cfg();
    c.d_rel = d.dr;
    c
}

fn cfg() -> InklingConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024, "unpadded_vocab_size": 200058,
        "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
        "swa_head_dim": 128, "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024, "d_rel": 16, "rel_extent": 1024,
        "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 128, "num_experts_per_tok": 8, "intermediate_size": 1152,
        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}
    }))
    .expect("a config")
}

/// xorshift64*, Box-Muller: the same inputs on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unif(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unif(), self.unif());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
    /// Normal values rounded to BF16, held in f32 (the loader's widening).
    fn bf(&mut self, n: usize, mean: f32, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| half::bf16::from_f32(mean + sigma * self.normal()).to_f32()).collect()
    }
    /// Values in {-1, 0, 1}, nonzero with probability `p`.
    fn tern(&mut self, n: usize, p: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                if self.unif() < p {
                    if self.next() & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                }
            })
            .collect()
    }
    /// Powers of two from 1/16 to 4.
    fn pow2(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 2f32.powi((self.next() % 7) as i32 - 4)).collect()
    }
}

/// Host values of one attention block and its inputs.
struct Case {
    d: Dims,
    sliding: bool,
    dynamic: bool,
    /// name -> (values, shape)
    w: Vec<(&'static str, Vec<f32>, Vec<usize>)>,
    /// Three steps of (xq, xk, xv); `xk`/`xv` are `None` when the visit has one input.
    xs: Vec<[Vec<f32>; 3]>,
    one_input: bool,
    /// The key and value states a prompt left, `[taps - 1, C]`.
    k_state: Option<Vec<f32>>,
    v_state: Option<Vec<f32>>,
    /// The attention outputs fed to `stage_out`, one a step.
    a: Vec<Vec<f32>>,
}

impl Case {
    /// Random BF16 weights at the scales the real model's activations see, or
    /// (`exact`) products exact in f32.
    fn new(seed: u64, d: Dims, sliding: bool, dynamic: bool, one_input: bool, state: bool, exact: bool) -> Self {
        let mut r = Rng(seed | 1);
        let (h, nh, nkv, hd, dr, ext, k) = (d.h, d.nh, d.nkv, d.hd, d.dr, d.ext, d.taps);
        let (din, c) = (nh * hd, nkv * hd);
        let s = 1.0 / (h as f32).sqrt();
        let mat = |r: &mut Rng, n: usize, sigma: f32| if exact { r.tern(n, 0.5) } else { r.bf(n, 0.0, sigma) };
        let mut w: Vec<(&'static str, Vec<f32>, Vec<usize>)> = vec![
            ("wq", mat(&mut r, din * h, s), vec![din, h]),
            ("wr", mat(&mut r, nh * dr * h, s), vec![nh * dr, h]),
            ("sink", mat(&mut r, nh * h, s), vec![nh, h]),
            ("gate", mat(&mut r, din * h, s), vec![din, h]),
            ("wk", mat(&mut r, c * h, 2.0 * s), vec![c, h]),
            ("wv", mat(&mut r, c * h, s), vec![c, h]),
            ("wo", r.bf(h * din, 0.0, 1.0 / (din as f32).sqrt()), vec![h, din]),
        ];
        let scale = |r: &mut Rng, n: usize, mean: f32| if exact { r.pow2(n) } else { r.bf(n, mean, 0.5) };
        w.push(("q_scale", scale(&mut r, nh * hd, 2.0), vec![nh, hd]));
        w.push(("rel_scale", scale(&mut r, nh * dr, 2.5), vec![nh, dr]));
        w.push(("rel_proj", r.bf(dr * ext, 0.0, 0.25), vec![dr, ext]));
        for which in ["k", "v"] {
            if dynamic {
                w.push((if which == "k" { "k_static" } else { "v_static" }, r.bf(k * c, 0.0, 0.5), vec![k, c]));
                let p = if exact { r.tern(nkv * k * h, 0.5) } else { r.bf(nkv * k * h, 0.0, 0.3 * s) };
                w.push((if which == "k" { "k_proj" } else { "v_proj" }, p, vec![nkv * k, h]));
            } else {
                w.push((if which == "k" { "k_conv" } else { "v_conv" }, r.bf(c * k, 0.0, 0.5), vec![c, k]));
            }
        }
        let act = |r: &mut Rng| if exact { r.tern(h, 32.0 / h as f64) } else { r.bf(h, 0.0, 1.0) };
        let xs = (0..3)
            .map(|_| {
                let q = act(&mut r);
                if one_input {
                    [q.clone(), q.clone(), q]
                } else {
                    [q, act(&mut r), act(&mut r)]
                }
            })
            .collect();
        let st = |r: &mut Rng| state.then(|| r.bf((k - 1) * c, 0.0, 1.0));
        let k_state = st(&mut r);
        let v_state = st(&mut r);
        let a = (0..3).map(|_| r.bf(din, 0.0, 0.3)).collect();
        Case { d, sliding, dynamic, w, xs, one_input, k_state, v_state, a }
    }

    fn get(&self, name: &str, dev: &Device) -> Tensor {
        let (_, v, s) = self.w.iter().find(|e| e.0 == name).unwrap_or_else(|| panic!("no weight {name}"));
        Tensor::from_vec(v.clone(), s.as_slice(), dev).unwrap()
    }

    fn weights(&self, dev: &Device) -> AttnWeights {
        let conv = |which: &str| {
            if self.dynamic {
                KvConv::Dynamic {
                    static_w: self.get(&format!("{which}_static"), dev),
                    weight_proj: self.get(&format!("{which}_proj"), dev),
                }
            } else {
                KvConv::Static(self.get(&format!("{which}_conv"), dev))
            }
        };
        AttnWeights {
            heads: self.d.nh,
            kv_heads: self.d.nkv,
            head_dim: self.d.hd,
            sliding: self.sliding,
            rel_extent: self.d.ext,
            wq: self.get("wq", dev),
            q_scale: self.get("q_scale", dev),
            wk: self.get("wk", dev),
            k_conv: conv("k"),
            wv: self.get("wv", dev),
            v_conv: conv("v"),
            gate: self.get("gate", dev),
            wo: self.get("wo", dev),
            sink: self.get("sink", dev),
            wr: self.get("wr", dev),
            rel_scale: self.get("rel_scale", dev),
            rel_proj: self.get("rel_proj", dev),
            dense: None,
            cpu: None,
        }
    }

    fn x(&self, step: usize, i: usize, dev: &Device) -> Tensor {
        Tensor::from_vec(self.xs[step][i].clone(), (1, self.d.h), dev).unwrap()
    }

    fn state(&self, which: usize, dev: &Device) -> Option<Tensor> {
        let c = self.d.nkv * self.d.hd;
        let s = if which == 0 { &self.k_state } else { &self.v_state };
        s.as_ref().map(|v| Tensor::from_vec(v.clone(), (self.d.taps - 1, c), dev).unwrap())
    }

    fn slot(&self, dev: &Device) -> SlotCache {
        let mut slot = SlotCache::default();
        if let Some(s) = self.state(0, dev) {
            slot.conv_k = ConvState::Block(s);
        }
        if let Some(s) = self.state(1, dev) {
            slot.conv_v = ConvState::Block(s);
        }
        slot
    }
}

/// `forward_chunked`'s statements from the inputs to the score step, one row
/// at position `pos`: (q, k, v, rel_logits, sinks, gate) in its layouts.
fn reference_in(
    cfg: &InklingConfig,
    w: &AttnWeights,
    xq: &Tensor,
    xk: &Tensor,
    xv: &Tensor,
    slot: &mut SlotCache,
    pos: usize,
) -> Result<[Tensor; 6]> {
    let dev = xq.device();
    let t = 1;
    let (nh, nkv, hd) = (w.heads, w.kv_heads, w.head_dim);
    let tau = if w.sliding { None } else { Some(log_scale(cfg, &[pos], dev)?) };
    let q = linear(xq, &w.wq)?.reshape((t, nh, hd))?.broadcast_mul(&w.q_scale)?;
    let mut q = l2_clip(&q, cfg.qk_l2_clip_threshold)?;
    if let Some(tau) = &tau {
        q = q.broadcast_mul(tau)?;
    }
    let q = q.transpose(0, 1)?.contiguous()?;
    let kv = |x: &Tensor, wt: &Tensor, conv: &KvConv, state: &mut ConvState| -> Result<Tensor> {
        let s = linear(x, wt)?;
        match conv {
            KvConv::Static(cw) => short_conv(&s, cw, state),
            KvConv::Dynamic { static_w, weight_proj } => dynamic_conv(&s, x, static_w, weight_proj, nkv, state),
        }
    };
    let k = kv(xk, &w.wk, &w.k_conv, &mut slot.conv_k)?.reshape((t, nkv, hd))?;
    let k = l2_clip(&k, cfg.qk_l2_clip_threshold)?.transpose(0, 1)?.contiguous()?;
    let v = kv(xv, &w.wv, &w.v_conv, &mut slot.conv_v)?.reshape((t, nkv, hd))?;
    let v = rms_unweighted(&v, cfg.rms_norm_eps)?.transpose(0, 1)?.contiguous()?;
    let r = linear(xq, &w.wr)?.reshape((t, nh, cfg.d_rel))?.broadcast_mul(&w.rel_scale)?;
    let mut r = l2_clip(&r, cfg.rel_l2_clip_threshold)?;
    if let Some(tau) = &tau {
        r = r.broadcast_mul(tau)?;
    }
    let rel = r.transpose(0, 1)?.contiguous()?.broadcast_matmul(&w.rel_proj)?;
    let sinks = linear(xq, &w.sink)?.t()?.unsqueeze(2)?;
    let gate = candle_nn::ops::sigmoid(&linear(xq, &w.gate)?)?;
    Ok([q, k, v, rel, sinks, gate])
}

/// `forward_chunked`'s last line.
fn reference_out(w: &AttnWeights, a: &Tensor, gate: &Tensor) -> Result<Tensor> {
    linear(&a.mul(gate)?, &w.wo)
}

fn parts(s: &StageIn) -> [&Tensor; 6] {
    [&s.q, &s.k, &s.v, &s.rel_logits, &s.sinks, &s.gate]
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// (bit-equal elements, total, largest |a - b| / largest |b|).
fn compare(a: &Tensor, b: &Tensor) -> (usize, usize, f64) {
    assert_eq!(a.dims(), b.dims(), "layouts differ");
    let (a, b) = (host(a), host(b));
    let same = a.iter().zip(&b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
    let s = b.iter().map(|y| y.abs() as f64).fold(0.0, f64::max);
    (same, a.len(), if s == 0.0 { d } else { d / s })
}

/// The `[taps - 1, C]` rows a `ConvState` holds, oldest first.
fn conv_rows(s: &ConvState) -> Tensor {
    match s {
        ConvState::Block(t) => t.clone(),
        ConvState::Ring { rows, head } => {
            let n = rows.len();
            let v: Vec<Tensor> = (0..n).map(|j| rows[(head + j) % n].clone()).collect();
            Tensor::cat(&v, 0).unwrap()
        }
        ConvState::Empty => panic!("a state after a step is never empty"),
    }
}

fn metal() -> Device {
    Device::new_metal(0).expect("a Metal device")
}

/// Query heads whose clip fires on step 0 in the CPU reference, of all
/// heads: a case is known to exercise both branches of the clip.
fn clip_fired(cfg: &InklingConfig, c: &Case, step: usize) -> (usize, usize) {
    let w = c.weights(&Device::Cpu);
    let xq = c.x(step, 0, &Device::Cpu);
    let q = linear(&xq, &w.wq).unwrap().reshape((c.d.nh, c.d.hd)).unwrap().broadcast_mul(&w.q_scale).unwrap();
    let n = q.sqr().unwrap().sum_keepdim(D::Minus1).unwrap().sqrt().unwrap();
    let fired = host(&n).iter().filter(|&&v| v as f64 + 1e-6 > cfg.qk_l2_clip_threshold).count();
    (fired, c.d.nh)
}

/// One case through three steps. Returns per output the worst (equal, total,
/// gap) against the CPU and the Metal reference, and the ring states'.
fn run_case(cfg: &InklingConfig, k: &AttnDense, c: &Case, pos0: usize) -> Result<Vec<[(usize, usize, f64); 2]>> {
    let mtl = metal();
    let (w_cpu, w_mtl) = (c.weights(&Device::Cpu), c.weights(&mtl));
    let dw = AttnDenseWeights::new(&w_mtl, cfg, &mtl)?;
    let mut rings = KvRings::new(&mtl, &dw, c.state(0, &mtl).as_ref(), c.state(1, &mtl).as_ref())?;
    let (mut slot_cpu, mut slot_mtl) = (c.slot(&Device::Cpu), c.slot(&mtl));
    // q k v rel sinks gate out ring_k ring_v
    let mut worst = vec![[(usize::MAX, 0usize, 0f64); 2]; 9];
    let mut fold = |i: usize, j: usize, r: (usize, usize, f64)| {
        let e = &mut worst[i][j];
        *e = (e.0.min(r.0), r.1, e.2.max(r.2));
    };
    for step in 0..3 {
        let pos = pos0 + step;
        let x = |dev: &Device| [c.x(step, 0, dev), c.x(step, 1, dev), c.x(step, 2, dev)];
        let (xc, xm) = (x(&Device::Cpu), x(&mtl));
        let rc = reference_in(cfg, &w_cpu, &xc[0], &xc[1], &xc[2], &mut slot_cpu, pos)?;
        let rm = reference_in(cfg, &w_mtl, &xm[0], &xm[1], &xm[2], &mut slot_mtl, pos)?;
        let (xk, xv) = if c.one_input { (&xm[0], &xm[0]) } else { (&xm[1], &xm[2]) };
        let got = k.stage_in(&dw, &xm[0], xk, xv, pos, &mut rings)?;
        for (i, g) in parts(&got).into_iter().enumerate() {
            fold(i, 0, compare(g, &rc[i]));
            fold(i, 1, compare(g, &rm[i]));
        }
        // stage_out on the kernel's own gate, against the reference line on that gate.
        let a_m = Tensor::from_vec(c.a[step].clone(), (1, c.d.nh * c.d.hd), &mtl)?;
        let out = k.stage_out(&dw, &a_m, &got.gate)?;
        let gate_c = got.gate.to_device(&Device::Cpu)?;
        fold(6, 0, compare(&out, &reference_out(&w_cpu, &a_m.to_device(&Device::Cpu)?, &gate_c)?));
        fold(6, 1, compare(&out, &reference_out(&w_mtl, &a_m, &got.gate)?));
        let like = &xm[0];
        fold(7, 0, compare(&rings.k.state(k, like)?, &conv_rows(&slot_cpu.conv_k)));
        fold(7, 1, compare(&rings.k.state(k, like)?, &conv_rows(&slot_mtl.conv_k)));
        fold(8, 0, compare(&rings.v.state(k, like)?, &conv_rows(&slot_cpu.conv_v)));
        fold(8, 1, compare(&rings.v.state(k, like)?, &conv_rows(&slot_mtl.conv_v)));
    }
    Ok(worst)
}

const ROW_NAMES: [&str; 9] = ["q", "k", "v", "rel_logits", "sinks", "gate", "stage_out", "ring_k", "ring_v"];

fn print_case(label: &str, w: &[[(usize, usize, f64); 2]]) {
    for (i, r) in w.iter().enumerate() {
        println!(
            "{label} {:>10}: vs cpu {}/{} equal (gap {:.3e}), vs metal {}/{} equal (gap {:.3e})",
            ROW_NAMES[i], r[0].0, r[0].1, r[0].2, r[1].0, r[1].1, r[1].2
        );
    }
}

/// (sliding, dynamic, one input, prompt state, first position): the four
/// shapes a visit takes, at a position past the log-scale floor for full layers.
const SHAPES: [(&str, bool, bool, bool, bool, usize); 4] = [
    ("sliding static one-input", true, false, true, true, 5),
    ("sliding static three-inputs", true, false, false, false, 1_500),
    ("full dynamic three-inputs", false, true, false, true, 9_000),
    ("full dynamic one-input", false, true, true, false, 20_000),
];

// ---------- class B: random BF16 weights ----------

/// Every output within the class B bound of both references, at the real and
/// the tiny shapes, in the four shapes of a visit, three ring steps.
#[test]
fn stage_in_and_out_against_layers() -> Result<()> {
    let k = AttnDense::new(&metal())?;
    for d in [REAL, TINY] {
        let cfg = cfg_for(d);
        for (i, (name, sliding, dynamic, one, state, pos)) in SHAPES.iter().enumerate() {
            let c = Case::new(0x5EED + 31 * i as u64 + d.h as u64, d, *sliding, *dynamic, *one, *state, false);
            let (fired, heads) = clip_fired(&cfg, &c, 0);
            let w = run_case(&cfg, &k, &c, *pos)?;
            let label = format!("random h {} {name} (query clip fired in {fired}/{heads} heads)", d.h);
            print_case(&label, &w);
            for (j, r) in w.iter().enumerate() {
                assert!(r[0].2 <= B_REL, "{label}: {} gap {:e} against the CPU", ROW_NAMES[j], r[0].2);
                assert!(r[1].2 <= B_REL, "{label}: {} gap {:e} against Metal", ROW_NAMES[j], r[1].2);
            }
        }
    }
    Ok(())
}

// ---------- class A after the products: exact products ----------

/// With products exact in f32, the query, key, value, sinks and both rings
/// equal the CPU reference bit for bit. Three outputs are held to the class B
/// bound instead: the relative logits (a 16-term matmul), the gate (Metal's
/// precise `exp` and the host libm's `expf` differ in the last bit on some
/// inputs) and `stage_out` (its gate operand is that gate). The Metal
/// reference is printed, not asserted: its reductions run in a tree.
#[test]
fn exact_products_give_the_cpu_reference_bit_for_bit() -> Result<()> {
    let k = AttnDense::new(&metal())?;
    for d in [REAL, TINY] {
        let cfg = cfg_for(d);
        for (i, (name, sliding, dynamic, one, state, pos)) in SHAPES.iter().enumerate() {
            let c = Case::new(0xE7AC + 17 * i as u64 + d.h as u64, d, *sliding, *dynamic, *one, *state, true);
            let (fired, heads) = clip_fired(&cfg, &c, 0);
            let w = run_case(&cfg, &k, &c, *pos)?;
            let label = format!("exact h {} {name} (query clip fired in {fired}/{heads} heads)", d.h);
            print_case(&label, &w);
            for (j, r) in w.iter().enumerate() {
                // rel_logits: a 16-term matmul; gate: exp; stage_out: reads that gate.
                if j == 3 || j == 5 || j == 6 {
                    assert!(r[0].2 <= B_REL, "{label}: {} gap {:e}", ROW_NAMES[j], r[0].2);
                    continue;
                }
                assert_eq!(r[0].0, r[0].1, "{label}: {} differs from the CPU reference", ROW_NAMES[j]);
            }
        }
    }
    Ok(())
}

/// `stage_out` on exact operands: `a` small integers, the gate powers of two,
/// `wo` in {-1, 0, 1}. Bit for bit against both references.
#[test]
fn stage_out_exact() -> Result<()> {
    let mtl = metal();
    let k = AttnDense::new(&mtl)?;
    for d in [REAL, TINY] {
        let cfg = cfg_for(d);
        let mut c = Case::new(0x07 + d.h as u64, d, true, false, true, true, true);
        let din = d.nh * d.hd;
        let mut r = Rng(0xA0A0 + d.h as u64);
        let wo = r.tern(d.h * din, 0.5);
        c.w.iter_mut().find(|e| e.0 == "wo").expect("wo").1 = wo;
        let (w_cpu, w_mtl) = (c.weights(&Device::Cpu), c.weights(&mtl));
        let dw = AttnDenseWeights::new(&w_mtl, &cfg, &mtl)?;
        let a: Vec<f32> = (0..din).map(|_| ((r.next() % 9) as f32) - 4.0).collect();
        let g = r.pow2(din);
        let a_m = Tensor::from_vec(a.clone(), (1, din), &mtl)?;
        let g_m = Tensor::from_vec(g.clone(), (1, din), &mtl)?;
        let got = k.stage_out(&dw, &a_m, &g_m)?;
        let want_c = reference_out(&w_cpu, &Tensor::from_vec(a, (1, din), &Device::Cpu)?, &Tensor::from_vec(g, (1, din), &Device::Cpu)?)?;
        let want_m = reference_out(&w_mtl, &a_m, &g_m)?;
        let (sc, n, gc) = compare(&got, &want_c);
        let (sm, _, gm) = compare(&got, &want_m);
        println!("stage_out exact h {}: vs cpu {sc}/{n} (gap {gc:e}), vs metal {sm}/{n} (gap {gm:e})", d.h);
        assert_eq!(sc, n, "stage_out differs from the CPU reference");
        assert_eq!(sm, n, "stage_out differs from the Metal reference");
    }
    Ok(())
}

// ---------- end to end: AttnWeights::forward ----------

#[allow(clippy::too_many_arguments)]
/// The score step of `forward_chunked` for one row over keys and values
/// `[kv_heads, S, d]`, copied statement for statement (with `score_bias`).
fn scores(q: &Tensor, k: &Tensor, v: &Tensor, rel: &Tensor, sinks: &Tensor, offset: usize, ext: usize, window: Option<usize>) -> Result<Tensor> {
    let (nh, _, hd) = q.dims3()?;
    let (nkv, s, _) = k.dims3()?;
    let rep = nh / nkv;
    let rk = |x: &Tensor| -> Result<Tensor> {
        let (kv, t, dd) = x.dims3()?;
        x.unsqueeze(1)?.broadcast_as((kv, rep, t, dd))?.reshape((kv * rep, t, dd))
    };
    let kt = rk(&k.transpose(1, 2)?.contiguous()?)?.contiguous()?;
    let v = rk(v)?.contiguous()?;
    let dev = q.device();
    let (mut idx, mut keep, mut mask) = (vec![], vec![], vec![]);
    for j in 0..s {
        let dd = offset as i64 - j as i64;
        idx.push(dd.clamp(0, ext as i64 - 1) as u32);
        keep.push(if dd >= 0 && (dd as usize) < ext { 1f32 } else { 0f32 });
        let visible = dd >= 0 && window.is_none_or(|w| (dd as usize) < w);
        mask.push(if visible { 0f32 } else { f32::NEG_INFINITY });
    }
    let idx = Tensor::from_vec(idx, (1, 1, s), dev)?.broadcast_as((nh, 1, s))?.contiguous()?;
    let bias = rel.contiguous()?.gather(&idx, 2)?.broadcast_mul(&Tensor::from_vec(keep, (1, 1, s), dev)?)?;
    let bias = bias.broadcast_add(&Tensor::from_vec(mask, (1, 1, s), dev)?)?;
    let sc = (q.matmul(&kt)? * (hd as f64).powf(-0.5))?.add(&bias)?;
    let probs = candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&sc, sinks], 2)?.contiguous()?)?.narrow(2, 0, s)?;
    probs.contiguous()?.matmul(&v)?.transpose(0, 1)?.reshape((1, nh * hd))
}

/// `AttnWeights::forward` on the CPU and on Metal, three decode steps from an
/// empty cache with prompt convolution states, against `stage_in`, the score
/// statements above and `stage_out`. Class B, within the bound.
#[test]
fn end_to_end_against_forward() -> Result<()> {
    let mtl = metal();
    let k = AttnDense::new(&mtl)?;
    for d in [REAL, TINY] {
        let cfg = cfg_for(d);
        for (i, (name, sliding, dynamic, one, state, _)) in SHAPES.iter().enumerate() {
            let c = Case::new(0xE2E + 7 * i as u64 + d.h as u64, d, *sliding, *dynamic, *one, *state, false);
            let (w_cpu, w_mtl) = (c.weights(&Device::Cpu), c.weights(&mtl));
            let dw = AttnDenseWeights::new(&w_mtl, &cfg, &mtl)?;
            let mut rings = KvRings::new(&mtl, &dw, c.state(0, &mtl).as_ref(), c.state(1, &mtl).as_ref())?;
            let (mut slot_cpu, mut slot_mtl) = (c.slot(&Device::Cpu), c.slot(&mtl));
            let (mut ks, mut vs) = (vec![], vec![]);
            let window = if *sliding { Some(cfg.sliding_window_size) } else { None };
            for step in 0..3 {
                let x = |dev: &Device| [c.x(step, 0, dev), c.x(step, 1, dev), c.x(step, 2, dev)];
                let (xc, xm) = (x(&Device::Cpu), x(&mtl));
                let want_c = w_cpu.forward(&cfg, &xc[0], &xc[1], &xc[2], &mut slot_cpu)?;
                let want_m = w_mtl.forward(&cfg, &xm[0], &xm[1], &xm[2], &mut slot_mtl)?;
                let (xk, xv) = if c.one_input { (&xm[0], &xm[0]) } else { (&xm[1], &xm[2]) };
                let s = k.stage_in(&dw, &xm[0], xk, xv, step, &mut rings)?;
                ks.push(s.k.clone());
                vs.push(s.v.clone());
                let a = scores(&s.q, &Tensor::cat(&ks, 1)?, &Tensor::cat(&vs, 1)?, &s.rel_logits, &s.sinks, step, d.ext, window)?;
                let got = k.stage_out(&dw, &a, &s.gate)?;
                let (sc, n, gc) = compare(&got, &want_c);
                let (sm, _, gm) = compare(&got, &want_m);
                let (_, _, gcm) = compare(&want_m, &want_c);
                println!("e2e h {} {name} step {step}: vs cpu {sc}/{n} (gap {gc:.3e}), vs metal {sm}/{n} (gap {gm:.3e}); metal vs cpu gap {gcm:.3e}", d.h);
                assert!(gc <= B_REL && gm <= B_REL, "e2e {name} step {step}: gap {gc:e} / {gm:e}");
            }
        }
    }
    Ok(())
}

/// The log scale the kernel takes equals `layers::log_scale`'s, bit for bit.
#[test]
fn tau_is_log_scale() -> Result<()> {
    let cfg = cfg();
    for pos in [0usize, 5, 8_190, 8_191, 8_192, 9_000, 20_000, 1_000_000] {
        let want = log_scale(&cfg, &[pos], &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?[0];
        assert_eq!(tau(&cfg, pos).to_bits(), want.to_bits(), "position {pos}");
    }
    Ok(())
}

/// A weight that is not a BF16 value is refused, not rounded.
#[test]
fn non_bf16_weights_are_refused() -> Result<()> {
    let cfg = cfg_for(TINY);
    let mtl = metal();
    let mut c = Case::new(3, TINY, true, false, true, false, false);
    c.w.iter_mut().find(|e| e.0 == "wq").expect("wq").1[7] = 1.0 + f32::EPSILON;
    let err = AttnDenseWeights::new(&c.weights(&mtl), &cfg, &mtl).err().expect("refused");
    println!("refusal: {err}");
    assert!(err.to_string().contains("not a BF16 value"));
    Ok(())
}

// ---------- launches and time ----------

/// candle's path for the products of a visit (the statements of
/// `reference_in` and `reference_out`, f32 weights) against the three
/// launches, 1,000 visits, 5 interleaved rounds, a sliding and a full layer.
#[test]
#[ignore = "timing; run by name"]
fn attn_dense_timing() -> Result<()> {
    let cfg = cfg();
    let mtl = metal();
    let k = AttnDense::new(&mtl)?;
    let visits = 1000;
    let rounds = 5;
    for (name, sliding, dynamic, one) in [("sliding static, one input", true, false, true), ("full dynamic, three inputs", false, true, false)] {
        let c = Case::new(0x71A3, REAL, sliding, dynamic, one, true, false);
        let w = c.weights(&mtl);
        let dw = AttnDenseWeights::new(&w, &cfg, &mtl)?;
        let xm = [c.x(0, 0, &mtl), c.x(0, 1, &mtl), c.x(0, 2, &mtl)];
        let (xk, xv) = if one { (&xm[0], &xm[0]) } else { (&xm[1], &xm[2]) };
        let a = Tensor::from_vec(c.a[0].clone(), (1, REAL.nh * REAL.hd), &mtl)?;
        let mut times: [Vec<f64>; 2] = [vec![], vec![]];
        for round in 0..rounds {
            let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for arm in order {
                let mut slot = c.slot(&mtl);
                let mut rings = KvRings::new(&mtl, &dw, c.state(0, &mtl).as_ref(), c.state(1, &mtl).as_ref())?;
                mtl.synchronize()?;
                let t0 = std::time::Instant::now();
                let mut last = None;
                for v in 0..visits {
                    let pos = 9_000 + v;
                    last = Some(if arm == 0 {
                        let r = reference_in(&cfg, &w, &xm[0], xk, xv, &mut slot, pos)?;
                        (reference_out(&w, &a, &r[5])?, r[0].clone())
                    } else {
                        let s = k.stage_in(&dw, &xm[0], xk, xv, pos, &mut rings)?;
                        (k.stage_out(&dw, &a, &s.gate)?, s.q)
                    });
                }
                let (o, q) = last.expect("a visit");
                let _ = (host(&o), host(&q));
                mtl.synchronize()?;
                times[arm].push(t0.elapsed().as_secs_f64());
            }
        }
        for (arm, t) in ["candle, f32 weights", "attn_dense, 3 launches"].iter().zip(times.iter_mut()) {
            t.sort_by(f64::total_cmp);
            println!(
                "timing {name} / {arm}: {visits} visits, median {:.1} ms [{:.1}, {:.1}] over {rounds} rounds, {:.1} us a visit",
                1e3 * t[t.len() / 2],
                1e3 * t[0],
                1e3 * t[t.len() - 1],
                1e6 * t[t.len() / 2] / visits as f64
            );
        }
        let f32_bytes: usize = w_elems(&w) * 4;
        println!(
            "bytes a visit {name}: candle reads {:.2} MB of f32 weights, attn_dense {:.2} MB of BF16",
            f32_bytes as f64 / 1e6,
            dw.bytes_read() as f64 / 1e6
        );
    }
    Ok(())
}

fn w_elems(w: &AttnWeights) -> usize {
    let mut n = [&w.wq, &w.q_scale, &w.wk, &w.wv, &w.gate, &w.wo, &w.sink, &w.wr, &w.rel_scale, &w.rel_proj]
        .iter()
        .map(|t| t.elem_count())
        .sum::<usize>();
    for c in [&w.k_conv, &w.v_conv] {
        n += match c {
            KvConv::Static(t) => t.elem_count(),
            KvConv::Dynamic { static_w, weight_proj } => static_w.elem_count() + weight_proj.elem_count(),
        };
    }
    n
}
