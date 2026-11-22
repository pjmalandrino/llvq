//! The fused glue kernels of `inkling::glue` against the functions of
//! `inkling/layers.rs`, run on the CPU and on Metal.
//!
//! Synthetic inputs at the real shapes (hidden 2,304, gate rank 64, 128
//! experts, top 8, 4 taps) and at a tiny config (hidden 48, rank 4, 8 experts,
//! top 2). Bit for bit where the kernel keeps the reference's order of
//! operations (`post_mlp`, the residual of `post_attn`, `top8`), and the
//! largest relative gap where it does not (`pre`, the normed input and the
//! scores of `post_attn`), which makes those class B.
//!
//! Tiny and synthetic: not a heavy job. `glue_timing` is ignored by default
//! and run by name.
#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, IndexOp, Result, Tensor};
use llvq_llm::inkling::glue::{ConvRing, GateW, Glue, PostAttnInputs, PreInputs};
use llvq_llm::inkling::layers::{block_attn_res, rms_norm, route, short_conv, ConvState, MessageGate};

const EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
struct Dims {
    h: usize,
    rank: usize,
    experts: usize,
    k: usize,
    top_k: usize,
}

const REAL: Dims = Dims { h: 2304, rank: 64, experts: 128, k: 4, top_k: 8 };
const TINY: Dims = Dims { h: 48, rank: 4, experts: 8, k: 4, top_k: 2 };

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
    fn t(&mut self, shape: &[usize], mean: f32, sigma: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| mean + sigma * self.normal()).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }
}

struct Gate {
    down: Tensor,
    forget: Tensor,
    gain: Tensor,
}

impl Gate {
    fn new(r: &mut Rng, d: Dims) -> Self {
        Gate {
            down: r.t(&[d.rank, 2 * d.h], 0.0, 1.0 / (2.0 * d.h as f32).sqrt()),
            forget: r.t(&[d.h, d.rank], 0.0, 1.0 / (d.rank as f32).sqrt()),
            gain: r.t(&[d.h, d.rank], 0.0, 1.0 / (d.rank as f32).sqrt()),
        }
    }
    fn to(&self, dev: &Device) -> Gate {
        Gate { down: mv(&self.down, dev), forget: mv(&self.forget, dev), gain: mv(&self.gain, dev) }
    }
    fn layer(&self) -> MessageGate {
        MessageGate { down: self.down.clone(), forget: self.forget.clone(), gain: self.gain.clone() }
    }
    fn w(&self) -> GateW<'_> {
        GateW { down: &self.down, forget: &self.forget, gain: &self.gain }
    }
}

fn mv(t: &Tensor, dev: &Device) -> Tensor {
    t.to_device(dev).unwrap()
}

/// A convolution state as `SlotCache` holds it: a prompt's block `[K - 1, C]`, or empty (zeros).
fn conv_state(s: &Option<Tensor>) -> ConvState {
    match s {
        Some(t) => ConvState::Block(t.clone()),
        None => ConvState::Empty,
    }
}

/// The state after a step as one block `[K - 1, C]`, oldest row first.
fn state_block(s: &ConvState, k: usize) -> Tensor {
    s.block(k).unwrap().expect("a state after a step")
}

/// One visit's inputs and weights, everything but the attention and the experts.
struct Visit {
    sliding: bool,
    boundary: bool,
    h: Tensor,
    states: Vec<Tensor>,
    queries: Option<Tensor>,
    norms: Vec<Tensor>,
    msgs: Option<Vec<Tensor>>,
    gates: Option<Vec<Gate>>,
    attn_out: Tensor,
    moe: Tensor,
    conv_attn_w: Tensor,
    conv_mlp_w: Tensor,
    conv_attn_state: Option<Tensor>,
    conv_mlp_state: Option<Tensor>,
    mlp_q: Tensor,
    post_norm: Tensor,
    router_w: Tensor,
    router_b: Tensor,
}

impl Visit {
    #[allow(clippy::too_many_arguments)]
    fn new(seed: u64, d: Dims, sliding: bool, queries: bool, msgs: bool, n_states: usize, boundary: bool, state: bool) -> Self {
        let mut r = Rng(seed | 1);
        let h = d.h;
        let nq = if sliding { 1 } else { 3 };
        Visit {
            sliding,
            boundary,
            h: r.t(&[1, h], 0.0, 1.0),
            states: (0..n_states).map(|i| r.t(&[1, h], 0.0, 0.5 + i as f32)).collect(),
            queries: queries.then(|| r.t(&[nq, h], 0.0, 1.0)),
            norms: (0..nq).map(|_| r.t(&[h], 1.0, 0.1)).collect(),
            msgs: msgs.then(|| (0..4).map(|_| r.t(&[1, h], 0.0, 1.0)).collect()),
            gates: msgs.then(|| (0..4).map(|_| Gate::new(&mut r, d)).collect()),
            attn_out: r.t(&[1, h], 0.0, 1.0),
            moe: r.t(&[1, h], 0.0, 1.0),
            conv_attn_w: r.t(&[h, d.k], 0.0, 0.5),
            conv_mlp_w: r.t(&[h, d.k], 0.0, 0.5),
            conv_attn_state: state.then(|| r.t(&[d.k - 1, h], 0.0, 1.0)),
            conv_mlp_state: state.then(|| r.t(&[d.k - 1, h], 0.0, 1.0)),
            mlp_q: r.t(&[1, h], 0.0, 1.0),
            post_norm: r.t(&[h], 1.0, 0.1),
            router_w: r.t(&[d.experts, h], 0.0, 1.0 / (h as f32).sqrt()),
            router_b: r.t(&[d.experts], 0.0, 0.05),
        }
    }

    fn to(&self, dev: &Device) -> Visit {
        let o = |t: &Option<Tensor>| t.as_ref().map(|t| mv(t, dev));
        let v = |t: &[Tensor]| t.iter().map(|t| mv(t, dev)).collect::<Vec<_>>();
        Visit {
            sliding: self.sliding,
            boundary: self.boundary,
            h: mv(&self.h, dev),
            states: v(&self.states),
            queries: o(&self.queries),
            norms: v(&self.norms),
            msgs: self.msgs.as_ref().map(|m| v(m)),
            gates: self.gates.as_ref().map(|g| g.iter().map(|g| g.to(dev)).collect()),
            attn_out: mv(&self.attn_out, dev),
            moe: mv(&self.moe, dev),
            conv_attn_w: mv(&self.conv_attn_w, dev),
            conv_mlp_w: mv(&self.conv_mlp_w, dev),
            conv_attn_state: o(&self.conv_attn_state),
            conv_mlp_state: o(&self.conv_mlp_state),
            mlp_q: mv(&self.mlp_q, dev),
            post_norm: mv(&self.post_norm, dev),
            router_w: mv(&self.router_w, dev),
            router_b: mv(&self.router_b, dev),
        }
    }

    /// `LayerWeights::forward` lines 592-635 up to the attention call: the normed inputs.
    fn ref_pre(&self) -> Result<Vec<Tensor>> {
        let gates: Option<Vec<MessageGate>> = self.gates.as_ref().map(|g| g.iter().map(Gate::layer).collect());
        let m = self.msgs.as_ref();
        if self.sliding {
            let x = match &self.queries {
                Some(q) => block_attn_res(q, &self.states, &self.h)?.remove(0),
                None => self.h.clone(),
            };
            let w = &self.norms[0];
            Ok(match (&gates, m) {
                (Some(g), Some(m)) => vec![
                    rms_norm(&g[0].forward(&x, &m[0], EPS)?, w, EPS)?,
                    rms_norm(&g[1].forward(&x, &m[1], EPS)?, w, EPS)?,
                    rms_norm(&g[2].forward(&x, &m[2], EPS)?, w, EPS)?,
                ],
                _ => vec![rms_norm(&x, w, EPS)?],
            })
        } else {
            let mut x = match &self.queries {
                Some(q) => block_attn_res(q, &self.states, &self.h)?,
                None => vec![self.h.clone(), self.h.clone(), self.h.clone()],
            };
            if let (Some(g), Some(m)) = (&gates, m) {
                for i in 0..3 {
                    x[i] = g[i].forward(&x[i], &m[i], EPS)?;
                }
            }
            (0..3).map(|i| rms_norm(&x[i], &self.norms[i], EPS)).collect()
        }
    }

    fn pre_inputs(&self) -> PreInputs<'_> {
        PreInputs {
            h: &self.h,
            states: &self.states,
            queries: self.queries.as_ref(),
            norms: &self.norms,
            gates: match (&self.msgs, &self.gates) {
                (Some(m), Some(g)) => Some(([&m[0], &m[1], &m[2]], [g[0].w(), g[1].w(), g[2].w()])),
                _ => None,
            },
            sliding: self.sliding,
            eps: EPS,
        }
    }

    /// Lines 636-652: the residual, the normed MLP input, the sigmoid scores,
    /// the routing, and the post-attention convolution state after the step.
    fn ref_post_attn(&self) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut st = conv_state(&self.conv_attn_state);
        let a = short_conv(&self.attn_out, &self.conv_attn_w, &mut st)?;
        let mut states = self.states.clone();
        let residual = if self.boundary {
            states.push(self.h.clone());
            a
        } else {
            (&self.h + a)?
        };
        let mut m_in = block_attn_res(&self.mlp_q, &states, &residual)?.remove(0);
        if let (Some(g), Some(m)) = (&self.gates, &self.msgs) {
            m_in = g[3].layer().forward(&m_in, &m[3], EPS)?;
        }
        let xn = rms_norm(&m_in, &self.post_norm, EPS)?;
        let scores = candle_nn::ops::sigmoid(&xn.matmul(&self.router_w.t()?)?)?.flatten_all()?;
        Ok((residual, xn, scores, state_block(&st, self.conv_attn_w.dim(1)?)))
    }

    fn post_attn_inputs(&self) -> PostAttnInputs<'_> {
        PostAttnInputs {
            attn_out: &self.attn_out,
            h: &self.h,
            states: &self.states,
            boundary: self.boundary,
            conv_w: &self.conv_attn_w,
            mlp_query: &self.mlp_q,
            gate: match (&self.msgs, &self.gates) {
                (Some(m), Some(g)) => Some((&m[3], g[3].w())),
                _ => None,
            },
            norm: &self.post_norm,
            router_w: &self.router_w,
            eps: EPS,
        }
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// (bit-equal elements, total, largest |a - b| / largest |b|).
fn compare(a: &Tensor, b: &Tensor) -> (usize, usize, f64) {
    let (a, b) = (host(a), host(b));
    assert_eq!(a.len(), b.len());
    let same = a.iter().zip(&b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
    let s = b.iter().map(|y| y.abs() as f64).fold(0.0, f64::max);
    (same, a.len(), d / s)
}

fn metal() -> Device {
    Device::new_metal(0).expect("a Metal device")
}

/// The class B bound the pre and post-attention gates hold to. The CPU and
/// Metal references disagree with each other at the same order of size.
const B_REL: f64 = 1e-5;

// ---------- post_mlp: class A ----------

/// `residual + short_conv(moe)` through the kernel against the reference on
/// the CPU and on Metal, bit for bit, three steps on the ring, from a prompt
/// state and from none; the ring's state after each step equals `SlotCache`'s.
#[test]
fn post_mlp_is_bit_exact() -> Result<()> {
    let dev = metal();
    let glue = Glue::new(&dev)?;
    for d in [REAL, TINY] {
        for with_state in [true, false] {
            let v = Visit::new(0xA11 + d.h as u64, d, true, true, false, 0, false, with_state);
            let mut r = Rng(0xBEEF ^ d.h as u64);
            let moes: Vec<Tensor> = (0..3).map(|_| r.t(&[1, d.h], 0.0, 1.0)).collect();
            let gm = v.to(&dev);
            let mut ring = ConvRing::new(&dev, d.k, d.h, gm.conv_mlp_state.as_ref())?;
            let mut st_cpu = conv_state(&v.conv_mlp_state);
            let mut st_mtl = conv_state(&gm.conv_mlp_state);
            for (step, m) in moes.iter().enumerate() {
                let want_cpu = (&v.h + short_conv(m, &v.conv_mlp_w, &mut st_cpu)?)?;
                let mm = mv(m, &dev);
                let want_mtl = (&gm.h + short_conv(&mm, &gm.conv_mlp_w, &mut st_mtl)?)?;
                let got = glue.post_mlp(&mm, &gm.h, &gm.conv_mlp_w, &mut ring)?;
                let (sc, n, gc) = compare(&got, &want_cpu);
                let (sm, _, gmx) = compare(&got, &want_mtl);
                let ring_st = ring.state(&glue, &gm.h)?;
                let (ss, ns, _) = compare(&ring_st, &state_block(&st_cpu, d.k));
                println!(
                    "post_mlp h {} state {with_state} step {step}: vs cpu {sc}/{n} equal (gap {gc:e}), vs metal {sm}/{n} (gap {gmx:e}), ring {ss}/{ns}",
                    d.h
                );
                assert_eq!(sc, n, "post_mlp differs from the CPU reference");
                assert_eq!(sm, n, "post_mlp differs from the Metal reference");
                assert_eq!(ss, ns, "the ring's state differs from SlotCache's");
            }
        }
    }
    Ok(())
}

// ---------- top8: exact, ties built on purpose ----------

/// `route` on one row whose scores are `sigmoid(logits)`: a `[1, 1]` input of
/// 1 against a `[E, 1]` gate, so `linear` returns the logits exactly. The
/// kernel takes the same sigmoid scores (candle's CPU sigmoid, the op `route`
/// runs) and the bias, and must return the same selection values, experts,
/// weights and margin, bit for bit.
fn top8_case(glue: &Glue, dev: &Device, name: &str, logits: &[f32], bias: &[f32], k: usize) -> Result<()> {
    let e = logits.len();
    let x = Tensor::from_vec(vec![1f32], (1, 1), &Device::Cpu)?;
    let w = Tensor::from_vec(logits.to_vec(), (e, 1), &Device::Cpu)?;
    let b = Tensor::from_vec(bias.to_vec(), e, &Device::Cpu)?;
    let want = route(&x, &w, &b, k, 8.0)?;
    let s = candle_nn::ops::sigmoid(&Tensor::from_vec(logits.to_vec(), e, &Device::Cpu)?)?;
    let got = glue.top8(&mv(&s, dev), &mv(&b, dev), k, 8.0)?;
    let sel = host(&got.sel);
    let ids: Vec<u32> = host(&got.ids).iter().map(|&v| v as u32).collect();
    let wts = host(&got.weights);
    let margin = host(&got.margin)[0];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    println!(
        "top8 {name}: ids {:?} vs {:?}, margin {margin:e} vs {:e}",
        ids, want.indices[0], want.margin[0]
    );
    assert_eq!(bits(&sel), bits(&want.scores[0]), "{name}: selection values");
    assert_eq!(ids, want.indices[0], "{name}: experts");
    assert_eq!(bits(&wts), bits(&want.weights[0]), "{name}: weights");
    assert_eq!(margin.to_bits(), want.margin[0].to_bits(), "{name}: margin");
    Ok(())
}

#[test]
fn top8_matches_route_with_ties() -> Result<()> {
    let dev = metal();
    let glue = Glue::new(&dev)?;
    let e = 128;
    let ulp = 2f32.powi(-24); // one float in [0.5, 1)
    let mut r = Rng(0x7041);
    // random, real shape
    let lg: Vec<f32> = (0..e).map(|_| r.normal()).collect();
    let bs: Vec<f32> = (0..e).map(|_| 0.05 * r.normal()).collect();
    top8_case(&glue, &dev, "random", &lg, &bs, 8)?;
    // all equal: the 8 lowest indices, margin 0
    top8_case(&glue, &dev, "all-equal", &vec![0.0; e], &vec![0.0; e], 8)?;
    // sigmoid(0) = 0.5, so sel = 0.5 + m * ulp exactly: values one float apart.
    // A tie across the 8th and 9th places, at shuffled indices.
    let mut m = vec![0i32; e];
    for (i, mi) in m.iter_mut().enumerate() {
        *mi = (i as i32 * 37) % 23 - 11; // many repeated levels
    }
    let bias: Vec<f32> = m.iter().map(|&mi| mi as f32 * ulp).collect();
    top8_case(&glue, &dev, "levels-one-ulp", &vec![0.0; e], &bias, 8)?;
    // exactly 7 above, then 3 tied at the boundary, the rest one ulp below
    let mut bias = vec![-ulp; e];
    for i in [100, 3, 57, 77, 12, 90, 64] {
        bias[i] = 2.0 * ulp;
    }
    for i in [101, 5, 40] {
        bias[i] = 0.0;
    }
    top8_case(&glue, &dev, "boundary-tie", &vec![0.0; e], &bias, 8)?;
    // the 8th and 9th one float apart, the 9th at a lower index
    let mut bias = vec![-4.0 * ulp; e];
    for i in [120, 110, 100, 90, 80, 70, 60] {
        bias[i] = 3.0 * ulp;
    }
    bias[50] = ulp;
    bias[2] = 0.0;
    top8_case(&glue, &dev, "one-ulp-boundary", &vec![0.0; e], &bias, 8)?;
    // negative selection values, and an exact zero (0.5 + -0.5)
    let mut bias: Vec<f32> = (0..e).map(|i| -1.0 - (i % 5) as f32 * 2f32.powi(-23)).collect();
    bias[9] = -0.5;
    top8_case(&glue, &dev, "negative-and-zero", &vec![0.0; e], &bias, 8)?;
    // distinct scores whose sums tie after the bias
    let lg: Vec<f32> = (0..e).map(|i| (i % 16) as f32 * 0.125 - 1.0).collect();
    let bs: Vec<f32> = (0..e).map(|i| -((i % 16) as f32) * 0.01).collect();
    top8_case(&glue, &dev, "score-bias-mix", &lg, &bs, 8)?;
    // tiny config: 2 of 8
    let lg: Vec<f32> = (0..8).map(|_| r.normal()).collect();
    top8_case(&glue, &dev, "tiny", &lg, &[0.0; 8], 2)?;
    top8_case(&glue, &dev, "tiny-tie", &[0.0; 8], &[0.0, ulp, 0.0, ulp, 0.0, 0.0, ulp, 0.0], 2)?;
    Ok(())
}

// ---------- pre: class B ----------

fn pre_cases(d: Dims) -> Vec<(&'static str, Visit)> {
    let s = d.h as u64;
    vec![
        ("sliding layer0", Visit::new(s ^ 1, d, true, false, false, 0, true, true)),
        ("sliding q", Visit::new(s ^ 2, d, true, true, false, 2, false, true)),
        ("sliding q msgs", Visit::new(s ^ 3, d, true, true, true, 2, false, true)),
        ("full q", Visit::new(s ^ 4, d, false, true, false, 1, false, true)),
        ("full q msgs", Visit::new(s ^ 5, d, false, true, true, 4, false, true)),
    ]
}

#[test]
fn pre_against_layers() -> Result<()> {
    let dev = metal();
    let glue = Glue::new(&dev)?;
    for d in [REAL, TINY] {
        for (name, v) in pre_cases(d) {
            let want_cpu = v.ref_pre()?;
            let g = v.to(&dev);
            let want_mtl = g.ref_pre()?;
            let got = glue.pre(&g.pre_inputs())?;
            assert_eq!(got.dims(), &[want_cpu.len(), d.h], "{name}: outputs");
            for (j, (wc, wm)) in want_cpu.iter().zip(&want_mtl).enumerate() {
                let row = got.i(j)?;
                let (sc, n, gc) = compare(&row, wc);
                let (sm, _, gm) = compare(&row, wm);
                let (_, _, gref) = compare(wm, wc);
                println!(
                    "pre h {} {name} out {j}: vs cpu {sc}/{n} equal gap {gc:.3e}; vs metal {sm}/{n} gap {gm:.3e}; metal ref vs cpu ref {gref:.3e}",
                    d.h
                );
                assert!(gc <= B_REL && gm <= B_REL, "{name} out {j}: gap {gc:e} / {gm:e} over {B_REL:e}");
            }
        }
    }
    Ok(())
}

// ---------- post_attn: residual class A, the rest class B ----------

#[test]
fn post_attn_against_layers() -> Result<()> {
    let dev = metal();
    let glue = Glue::new(&dev)?;
    for d in [REAL, TINY] {
        let s = d.h as u64;
        let cases = [
            ("sliding", Visit::new(s ^ 11, d, true, true, false, 2, false, true)),
            ("boundary", Visit::new(s ^ 12, d, true, true, false, 2, true, true)),
            ("boundary no state", Visit::new(s ^ 13, d, true, true, false, 0, true, false)),
            ("full msgs", Visit::new(s ^ 14, d, false, true, true, 3, false, true)),
            ("full msgs boundary", Visit::new(s ^ 15, d, false, true, true, 3, true, true)),
        ];
        for (name, v) in cases {
            let (res_c, xn_c, sc_c, st_c) = v.ref_post_attn()?;
            let g = v.to(&dev);
            let (res_m, xn_m, sc_m, _) = g.ref_post_attn()?;
            let mut ring = ConvRing::new(&dev, d.k, d.h, g.conv_attn_state.as_ref())?;
            let out = glue.post_attn(&g.post_attn_inputs(), &mut ring)?;
            let (rc, n, _) = compare(&out.residual, &res_c);
            let (rm, _, _) = compare(&out.residual, &res_m);
            let (rs, ns, _) = compare(&ring.state(&glue, &g.h)?, &st_c);
            let (_, _, xc) = compare(&out.xn, &xn_c);
            let (_, _, xm) = compare(&out.xn, &xn_m);
            let (_, _, xr) = compare(&xn_m, &xn_c);
            let (_, ne, scg) = compare(&out.scores, &sc_c);
            let (_, _, smg) = compare(&out.scores, &sc_m);
            // routing on the kernel's own scores, against route on the CPU reference's input
            let want = route(&xn_c, &v.router_w, &v.router_b, d.top_k, 8.0)?;
            let top = glue.top8(&out.scores, &g.router_b, d.top_k, 8.0)?;
            let ids: Vec<u32> = host(&top.ids).iter().map(|&x| x as u32).collect();
            let flips = ids.iter().filter(|i| !want.indices[0].contains(i)).count();
            println!(
                "post_attn h {} {name}: residual vs cpu {rc}/{n} vs metal {rm}/{n}, ring {rs}/{ns}; xn gap cpu {xc:.3e} metal {xm:.3e} (refs {xr:.3e}); scores [{ne}] gap cpu {scg:.3e} metal {smg:.3e}; experts flipped {flips} at margin {:.3e}",
                d.h, want.margin[0]
            );
            assert_eq!(rc, n, "{name}: residual differs from the CPU reference");
            assert_eq!(rm, n, "{name}: residual differs from the Metal reference");
            assert_eq!(rs, ns, "{name}: ring state");
            assert!(xc <= B_REL && xm <= B_REL, "{name}: xn gap {xc:e} / {xm:e}");
            assert!(scg <= B_REL && smg <= B_REL, "{name}: scores gap {scg:e} / {smg:e}");
            assert!(flips == 0 || want.margin[0] < 1e-4, "{name}: {flips} experts flipped at margin {}", want.margin[0]);
        }
    }
    Ok(())
}

// ---------- timing: candle glue against the kernels ----------

/// The glue of one visit through the layers.rs functions on Metal, as
/// `LayerWeights::forward` runs it (attention and experts replaced by fixed
/// tensors): returns the layer output.
fn candle_visit(v: &Visit, d: Dims, st_a: &mut ConvState, st_m: &mut ConvState) -> Result<Tensor> {
    let x = v.ref_pre()?;
    let _ = &x;
    let a = short_conv(&v.attn_out, &v.conv_attn_w, st_a)?;
    let mut states = v.states.clone();
    let residual = if v.boundary {
        states.push(v.h.clone());
        a
    } else {
        (&v.h + a)?
    };
    let mut m_in = block_attn_res(&v.mlp_q, &states, &residual)?.remove(0);
    if let (Some(g), Some(m)) = (&v.gates, &v.msgs) {
        m_in = g[3].layer().forward(&m_in, &m[3], EPS)?;
    }
    let xn = rms_norm(&m_in, &v.post_norm, EPS)?;
    let routing = route(&xn, &v.router_w, &v.router_b, d.top_k, 8.0)?;
    let _ = routing;
    let m = short_conv(&v.moe, &v.conv_mlp_w, st_m)?;
    residual + m
}

fn kernel_visit(glue: &Glue, v: &Visit, d: Dims, ra: &mut ConvRing, rm: &mut ConvRing, read_ids: bool) -> Result<Tensor> {
    let x = glue.pre(&v.pre_inputs())?;
    let _ = &x;
    let p = glue.post_attn(&v.post_attn_inputs(), ra)?;
    let top = glue.top8(&p.scores, &v.router_b, d.top_k, 8.0)?;
    if read_ids {
        let _ = top.ids.to_vec1::<f32>()?;
    }
    glue.post_mlp(&v.moe, &p.residual, &v.conv_mlp_w, rm)
}

#[test]
#[ignore = "timing; run by name"]
fn glue_timing() -> Result<()> {
    let dev = metal();
    let glue = Glue::new(&dev)?;
    let d = REAL;
    let visits = 1000;
    let rounds = 5;
    for (name, v) in [
        ("sliding, no messages", Visit::new(21, d, true, true, false, 2, false, true)),
        ("full, messages", Visit::new(22, d, false, true, true, 3, false, true)),
    ] {
        let g = v.to(&dev);
        let mut times: [Vec<f64>; 3] = [vec![], vec![], vec![]];
        for round in 0..rounds {
            let order: [usize; 3] = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
            for arm in order {
                let mut st_a = conv_state(&g.conv_attn_state);
                let mut st_m = conv_state(&g.conv_mlp_state);
                let mut ra = ConvRing::new(&dev, d.k, d.h, g.conv_attn_state.as_ref())?;
                let mut rm = ConvRing::new(&dev, d.k, d.h, g.conv_mlp_state.as_ref())?;
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                let mut last = None;
                for _ in 0..visits {
                    last = Some(match arm {
                        0 => candle_visit(&g, d, &mut st_a, &mut st_m)?,
                        1 => kernel_visit(&glue, &g, d, &mut ra, &mut rm, false)?,
                        _ => kernel_visit(&glue, &g, d, &mut ra, &mut rm, true)?,
                    });
                }
                let _ = host(&last.expect("a visit"));
                dev.synchronize()?;
                times[arm].push(t0.elapsed().as_secs_f64());
            }
        }
        let arms = ["candle glue", "kernels, ids on device", "kernels, ids read each visit"];
        for (a, t) in arms.iter().zip(times.iter_mut()) {
            t.sort_by(f64::total_cmp);
            println!(
                "timing {name} / {a}: {visits} visits, median {:.1} ms [{:.1}, {:.1}] over {rounds} rounds, {:.1} us a visit",
                1e3 * t[t.len() / 2],
                1e3 * t[0],
                1e3 * t[t.len() - 1],
                1e6 * t[t.len() / 2] / visits as f64
            );
        }
    }
    Ok(())
}
