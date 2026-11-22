//! `inkling::attn_scores` against the attention forward of `inkling/layers.rs`.
//!
//! The reference is `AttnWeights::forward` itself, on one decoded row, with
//! its output gate saturated to exactly 1 and its output projection the
//! identity on the first `heads * hd` rows, so its output IS the attention
//! result before the gate, bit for bit. The kernel reads the query, the
//! relative logits and the sinks computed by the same candle operations the
//! forward runs, and the slot's keys and values:
//!   - on Metal, in place, through `SlotCache::keys_values` after the forward
//!     appended the row;
//!   - against the CPU forward, from the CPU slot's values uploaded.
//!
//! Real attention shapes (24 query heads, 8 key-value heads, head dim 128,
//! extent 1,024, window 1,024; hidden 3,072 so the identity projection
//! exists; the hidden width never reaches the attention past its products)
//! and the tiny config (4 heads over 2, head dim 8, window 4, extent 6).
//! Class B bound: 1e-5 of the largest reference value.
//!
//! Tiny and synthetic: not a heavy job. `attn_row_timing` is ignored by
//! default and run by name.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use llvq_llm::inkling::attn_scores::{bytes_read, visible, AttnRow, RowInputs};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::{l2_clip, linear, log_scale, AttnWeights, KvConv, SlotCache};

/// Class B bound, relative to the largest reference value.
const BOUND: f32 = 1e-5;

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
    fn v(&mut self, n: usize, mean: f32, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| mean + sigma * self.normal()).collect()
    }
    fn t(&mut self, shape: &[usize], mean: f32, sigma: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(self.v(n, mean, sigma), shape, &Device::Cpu).unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
}

const REAL: Dims = Dims { hidden: 3072, nh: 24, nkv: 8, hd: 128 };
const TINY: Dims = Dims { hidden: 48, nh: 4, nkv: 2, hd: 8 };

/// The real config's attention values (`textonly/config.json`), or the tiny
/// config of the `layers.rs` tests.
fn config(real: bool) -> InklingConfig {
    let (h, nh, nkv, hd, win, d_rel, ext) = if real { (2304, 24, 8, 128, 1024, 16, 1024) } else { (48, 4, 2, 8, 4, 4, 6) };
    let v = serde_json::json!({
        "hidden_size": h, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
        "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
        "swa_head_dim": hd, "swa_num_attention_heads": nh, "swa_num_key_value_heads": nkv,
        "sliding_window_size": win, "d_rel": d_rel, "rel_extent": ext,
        "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 24,
        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
        "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
    });
    // One file a call: the tests run in parallel and each removes its own.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("inkling_attn_{}_{}_{n}.json", real, std::process::id()));
    std::fs::write(&path, v.to_string()).unwrap();
    let c = InklingConfig::from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    c
}

/// Random attention weights, gate saturated, output projection `[I; 0]`.
fn weights(rng: &mut Rng, cfg: &InklingConfig, d: Dims, sliding: bool) -> AttnWeights {
    let (h, nh, nkv, hd) = (d.hidden, d.nh, d.nkv, d.hd);
    let extent = if sliding { cfg.sliding_window_size } else { cfg.rel_extent };
    let c = nkv * hd;
    let conv = |rng: &mut Rng| {
        if sliding {
            KvConv::Static(rng.t(&[c, 4], 0.0, 0.5))
        } else {
            KvConv::Dynamic { static_w: rng.t(&[4, c], 0.0, 0.5), weight_proj: rng.t(&[nkv * 4, h], 0.0, 0.01) }
        }
    };
    let wq = rng.t(&[nh * hd, h], 0.0, 0.05);
    let q_scale = rng.t(&[nh, hd], 1.0, 0.2);
    let wk = rng.t(&[c, h], 0.0, 0.05);
    let k_conv = conv(rng);
    let wv = rng.t(&[c, h], 0.0, 0.05);
    let v_conv = conv(rng);
    let mut wo = vec![0f32; h * nh * hd];
    for i in 0..nh * hd {
        wo[i * nh * hd + i] = 1.0;
    }
    AttnWeights {
        heads: nh,
        kv_heads: nkv,
        head_dim: hd,
        sliding,
        rel_extent: extent,
        wq,
        q_scale,
        wk,
        k_conv,
        wv,
        v_conv,
        gate: Tensor::full(300.0 / h as f32, (nh * hd, h), &Device::Cpu).unwrap(),
        wo: Tensor::from_vec(wo, (h, nh * hd), &Device::Cpu).unwrap(),
        sink: rng.t(&[nh, h], 0.0, 0.02),
        wr: rng.t(&[nh * cfg.d_rel, h], 0.0, 0.05),
        rel_scale: rng.t(&[nh, cfg.d_rel], 1.0, 0.2),
        rel_proj: rng.t(&[cfg.d_rel, extent], 0.0, 0.3),
        dense: None,
    }
}

fn to_dev(w: &AttnWeights, dev: &Device) -> AttnWeights {
    let m = |t: &Tensor| t.to_device(dev).unwrap();
    AttnWeights {
        heads: w.heads,
        kv_heads: w.kv_heads,
        head_dim: w.head_dim,
        sliding: w.sliding,
        rel_extent: w.rel_extent,
        wq: m(&w.wq),
        q_scale: m(&w.q_scale),
        wk: m(&w.wk),
        k_conv: match &w.k_conv {
            KvConv::Static(t) => KvConv::Static(m(t)),
            KvConv::Dynamic { static_w, weight_proj } => {
                KvConv::Dynamic { static_w: m(static_w), weight_proj: m(weight_proj) }
            }
        },
        wv: m(&w.wv),
        v_conv: match &w.v_conv {
            KvConv::Static(t) => KvConv::Static(m(t)),
            KvConv::Dynamic { static_w, weight_proj } => {
                KvConv::Dynamic { static_w: m(static_w), weight_proj: m(weight_proj) }
            }
        },
        gate: m(&w.gate),
        wo: m(&w.wo),
        sink: m(&w.sink),
        wr: m(&w.wr),
        rel_scale: m(&w.rel_scale),
        rel_proj: m(&w.rel_proj),
        dense: None,
    }
}

/// How the key and value rows are drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Draw {
    Random,
    /// Every key and value row the same: every score past the relative
    /// extent ties exactly.
    Ties,
    /// The same row times `1 + 2^-22 u`, `u` uniform in [0, 1): ties broken
    /// in the last bits.
    NearTies,
}

/// `[n, h]` query, key and value inputs.
fn inputs(rng: &mut Rng, n: usize, h: usize, draw: Draw) -> (Tensor, Tensor, Tensor) {
    let xq: Vec<f32> = (0..n * h).map(|_| 0.5 + rng.unif() as f32).collect();
    let rows = |rng: &mut Rng| -> Vec<f32> {
        match draw {
            Draw::Random => rng.v(n * h, 0.0, 1.0),
            Draw::Ties | Draw::NearTies => {
                let base = rng.v(h, 0.0, 1.0);
                let mut out = Vec::with_capacity(n * h);
                for _ in 0..n {
                    let f = if draw == Draw::NearTies { 1.0 + rng.unif() as f32 * 2f32.powi(-22) } else { 1.0 };
                    out.extend(base.iter().map(|b| b * f));
                }
                out
            }
        }
    };
    let xk = rows(rng);
    let xv = rows(rng);
    let t = |v: Vec<f32>| Tensor::from_vec(v, (n, h), &Device::Cpu).unwrap();
    (t(xq), t(xk), t(xv))
}

/// The query `[nh, hd]`, relative logits `[nh, extent]` and sinks `[nh]` of
/// one row, by the operations `AttnWeights::forward_chunked` runs.
fn row_inputs(w: &AttnWeights, cfg: &InklingConfig, xq: &Tensor, pos: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (nh, hd) = (w.heads, w.head_dim);
    let tau = if w.sliding { None } else { Some(log_scale(cfg, &[pos], xq.device())?) };
    let q = linear(xq, &w.wq)?.reshape((1, nh, hd))?.broadcast_mul(&w.q_scale)?;
    let mut q = l2_clip(&q, cfg.qk_l2_clip_threshold)?;
    if let Some(tau) = &tau {
        q = q.broadcast_mul(tau)?;
    }
    let q = q.transpose(0, 1)?.contiguous()?;
    let r = linear(xq, &w.wr)?.reshape((1, nh, cfg.d_rel))?.broadcast_mul(&w.rel_scale)?;
    let mut r = l2_clip(&r, cfg.rel_l2_clip_threshold)?;
    if let Some(tau) = &tau {
        r = r.broadcast_mul(tau)?;
    }
    let rel = r.transpose(0, 1)?.contiguous()?.broadcast_matmul(&w.rel_proj)?;
    let sinks = linear(xq, &w.sink)?.t()?.contiguous()?;
    Ok((q.reshape((nh, hd))?, rel.reshape((nh, w.rel_extent))?, sinks.reshape(nh)?))
}

/// One compared row.
#[derive(Debug)]
struct Row {
    held: usize,
    max_abs: f32,
    max_ref: f32,
}

impl Row {
    fn rel(&self) -> f32 {
        self.max_abs / self.max_ref
    }
}

/// Which forward the kernel is compared with.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Against {
    /// The CPU forward; the kernel reads the CPU slot's values, uploaded.
    Cpu,
    /// The Metal forward; the kernel reads the Metal slot in place.
    Metal,
}

/// Fill one slot row by row through `forward_chunked` and, at each held count
/// of `targets`, compare one decoded row of the forward with the kernel.
fn sweep(real: bool, sliding: bool, targets: &[usize], draw: Draw, against: Against, seed: u64) -> Result<Vec<Row>> {
    let metal = Device::new_metal(0)?;
    let kernel = AttnRow::new(&metal)?;
    let cfg = config(real);
    let d = if real { REAL } else { TINY };
    let mut rng = Rng(seed);
    let w0 = weights(&mut rng, &cfg, d, sliding);
    let n = *targets.iter().max().expect("targets");
    let (xq, xk, xv) = inputs(&mut rng, n, d.hidden, draw);
    let dev = match against {
        Against::Cpu => Device::Cpu,
        Against::Metal => metal.clone(),
    };
    let w = to_dev(&w0, &dev);
    let (xq, xk, xv) = (xq.to_device(&dev)?, xk.to_device(&dev)?, xv.to_device(&dev)?);
    let window = if sliding { Some(cfg.sliding_window_size) } else { None };
    let mut cache = SlotCache::default();
    let mut out = Vec::new();
    for &s in targets {
        let row = s - 1;
        let len = cache.len();
        if row > len {
            let k = row - len;
            w.forward_chunked(&cfg, &xq.narrow(0, len, k)?, &xk.narrow(0, len, k)?, &xv.narrow(0, len, k)?, &mut cache, 512)?;
        }
        assert_eq!(cache.len(), row);
        let mut slot = cache.clone();
        let (q1, k1, v1) = (xq.narrow(0, row, 1)?, xk.narrow(0, row, 1)?, xv.narrow(0, row, 1)?);
        let gate = candle_nn::ops::sigmoid(&linear(&q1, &w.gate)?)?.flatten_all()?.min(0)?.to_scalar::<f32>()?;
        assert_eq!(gate, 1.0, "the output gate is not saturated");
        let reference = w.forward(&cfg, &q1, &k1, &v1, &mut slot)?.narrow(1, 0, d.nh * d.hd)?;
        assert_eq!(slot.len(), s);
        let (q, rel, sinks) = row_inputs(&w, &cfg, &q1, row)?;
        let (keys, values) = slot.keys_values()?.expect("a filled slot");
        let up = |t: &Tensor| t.to_device(&metal).unwrap();
        let got = match against {
            Against::Metal => kernel.forward(&RowInputs {
                q: &q,
                rel: &rel,
                sinks: &sinks,
                keys: &keys,
                values: &values,
                pos: row,
                window,
            })?,
            Against::Cpu => kernel.forward(&RowInputs {
                q: &up(&q),
                rel: &up(&rel),
                sinks: &up(&sinks),
                keys: &up(&keys),
                values: &up(&values),
                pos: row,
                window,
            })?,
        };
        let (max_abs, max_ref) = llvq_llm::inkling::layers::max_abs_err(&got, &reference)?;
        out.push(Row { held: s, max_abs, max_ref });
    }
    Ok(out)
}

/// Run a sweep, print every row, and fail past the class B bound.
fn gate(name: &str, real: bool, sliding: bool, targets: &[usize], draw: Draw, against: Against) -> Result<f32> {
    let rows = sweep(real, sliding, targets, draw, against, 0x5EED_0000 + targets.len() as u64)?;
    let mut worst = 0f32;
    for r in &rows {
        println!(
            "{name} {} {:?} {:?} held {:>5}: max |diff| {:.3e}, max |ref| {:.3e}, relative {:.3e}",
            if sliding { "sliding" } else { "full" },
            draw,
            against,
            r.held,
            r.max_abs,
            r.max_ref,
            r.rel()
        );
        assert!(r.max_abs.is_finite(), "{name}: not finite at {} held", r.held);
        worst = worst.max(r.rel());
    }
    println!("{name} worst relative {worst:.3e}");
    assert!(worst <= BOUND, "{name}: relative gap {worst:.3e} past the class B bound {BOUND:e}");
    Ok(worst)
}

const REAL_HELD: [usize; 7] = [1, 33, 1024, 1025, 1026, 1500, 3000];
const TINY_HELD: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20];

#[test]
fn visible_boundaries() {
    assert_eq!(visible(0, None), (0, 1));
    assert_eq!(visible(2999, None), (0, 3000));
    assert_eq!(visible(0, Some(1024)), (0, 1));
    assert_eq!(visible(1022, Some(1024)), (0, 1023));
    assert_eq!(visible(1023, Some(1024)), (0, 1024));
    assert_eq!(visible(1024, Some(1024)), (1, 1024));
    assert_eq!(visible(1025, Some(1024)), (2, 1024));
    assert_eq!(visible(4, Some(4)), (1, 4));
}

#[test]
fn attn_row_matches_forward_real_metal() -> Result<()> {
    gate("real-metal", true, true, &REAL_HELD, Draw::Random, Against::Metal)?;
    gate("real-metal", true, false, &REAL_HELD, Draw::Random, Against::Metal)?;
    Ok(())
}

#[test]
fn attn_row_matches_forward_real_cpu() -> Result<()> {
    gate("real-cpu", true, true, &REAL_HELD, Draw::Random, Against::Cpu)?;
    gate("real-cpu", true, false, &REAL_HELD, Draw::Random, Against::Cpu)?;
    Ok(())
}

#[test]
fn attn_row_matches_forward_tiny() -> Result<()> {
    for against in [Against::Cpu, Against::Metal] {
        gate("tiny", false, true, &TINY_HELD, Draw::Random, against)?;
        gate("tiny", false, false, &TINY_HELD, Draw::Random, against)?;
    }
    Ok(())
}

/// Scores that tie exactly (every position past the relative extent of a
/// full layer, keys and values all equal) and that differ in their last bits.
#[test]
fn attn_row_near_ties() -> Result<()> {
    for against in [Against::Cpu, Against::Metal] {
        for draw in [Draw::Ties, Draw::NearTies] {
            gate("ties", true, false, &[1025, 3000], draw, against)?;
            gate("ties", true, true, &[1500], draw, against)?;
        }
    }
    Ok(())
}

/// `[kv, T, d]` to `[kv * rep, T, d]`, as `layers.rs` does.
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    let (kv, t, d) = x.dims3()?;
    x.unsqueeze(1)?.broadcast_as((kv, rep, t, d))?.reshape((kv * rep, t, d))
}

/// `score_bias` of `layers.rs`, copied for the timing (the original is private).
fn score_bias(rel: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Tensor> {
    let (heads, t, _) = rel.dims3()?;
    let dev = rel.device();
    let mut idx = Vec::with_capacity(t * s);
    let mut mask = Vec::with_capacity(t * s);
    let mut keep = Vec::with_capacity(t * s);
    for i in 0..t {
        for j in 0..s {
            let d = (offset + i) as i64 - j as i64;
            idx.push(d.clamp(0, extent as i64 - 1) as u32);
            keep.push(if d >= 0 && (d as usize) < extent { 1f32 } else { 0f32 });
            let visible = d >= 0 && window.is_none_or(|w| (d as usize) < w);
            mask.push(if visible { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    let idx = Tensor::from_vec(idx, (1, t, s), dev)?.broadcast_as((heads, t, s))?.contiguous()?;
    let keep = Tensor::from_vec(keep, (1, t, s), dev)?;
    let mask = Tensor::from_vec(mask, (1, t, s), dev)?;
    let bias = rel.contiguous()?.gather(&idx, 2)?.broadcast_mul(&keep)?;
    bias.broadcast_add(&mask)
}

/// Bytes candle's path from the scores to the weighted sum reads and writes a
/// visit, `s` positions held (*computed* op by op, f32): the two head repeats
/// (read the view, write the copy), the two products (read the copies), and
/// about 20 passes over `heads x s` score-sized tensors.
fn candle_bytes(nh: usize, nkv: usize, hd: usize, s: usize) -> usize {
    let repeats = 2 * 4 * s * hd * (nkv + nh);
    let products = 2 * 4 * s * hd * nh;
    let scores = 20 * 4 * nh * s;
    repeats + products + scores
}

/// Candle's path against the kernel, 1,000 visits each, 5 interleaved rounds,
/// at 32, 1,024 and 9,000 positions held, sliding and full.
#[test]
#[ignore]
fn attn_row_timing() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let kernel = AttnRow::new(&dev)?;
    let d = REAL;
    let (nh, nkv, hd, extent) = (d.nh, d.nkv, d.hd, 1024);
    let mut rng = Rng(0x7131_4E00);
    let visits = 1000;
    struct Case {
        held: usize,
        window: Option<usize>,
        kt: Tensor,
        v: Tensor,
        q: Tensor,
        rel: Tensor,
        sinks: Tensor,
        t: [Vec<f64>; 2],
    }
    let mut cases = Vec::new();
    for held in [32usize, 1024, 9000] {
        for window in [Some(1024usize), None] {
            let cap = held.div_ceil(256) * 256 + 256;
            let kb = rng.t(&[nkv, hd, cap], 0.0, 0.1).to_device(&dev)?;
            let vb = rng.t(&[nkv, cap, hd], 0.0, 1.0).to_device(&dev)?;
            cases.push(Case {
                held,
                window,
                kt: kb.narrow(2, 0, held)?,
                v: vb.narrow(1, 0, held)?,
                q: rng.t(&[nh, 1, hd], 0.0, 1.0).to_device(&dev)?,
                rel: rng.t(&[nh, 1, extent], 0.0, 1.0).to_device(&dev)?,
                sinks: rng.t(&[nh, 1, 1], 0.0, 1.0).to_device(&dev)?,
                t: [Vec::new(), Vec::new()],
            });
        }
    }
    let candle_arm = |c: &Case| -> Result<Tensor> {
        let s = c.held;
        let rep = nh / nkv;
        let kt = repeat_kv(&c.kt, rep)?.contiguous()?;
        let v = repeat_kv(&c.v, rep)?.contiguous()?;
        let bias = score_bias(&c.rel, s - 1, s, extent, c.window)?;
        let scores = (c.q.matmul(&kt)? * (hd as f64).powf(-0.5))?.add(&bias)?;
        let probs =
            candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &c.sinks], 2)?.contiguous()?)?.narrow(2, 0, s)?;
        probs.contiguous()?.matmul(&v)?.transpose(0, 1)?.reshape((1, nh * hd))
    };
    let kernel_arm = |c: &Case| -> Result<Tensor> {
        let keys = c.kt.transpose(1, 2)?;
        kernel.forward(&RowInputs {
            q: &c.q,
            rel: &c.rel,
            sinks: &c.sinks,
            keys: &keys,
            values: &c.v,
            pos: c.held - 1,
            window: c.window,
        })
    };
    // Both arms once, and their gap, before timing.
    for c in &cases {
        let (a, b) = (candle_arm(c)?, kernel_arm(c)?);
        let (m, r) = llvq_llm::inkling::layers::max_abs_err(&b, &a)?;
        println!("held {:>5} {:?}: kernel against candle's path, relative {:.3e}", c.held, c.window, m / r);
    }
    for round in 0..5 {
        for c in cases.iter_mut() {
            for arm in 0..2 {
                dev.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..visits {
                    let o = if arm == 0 { candle_arm(c)? } else { kernel_arm(c)? };
                    drop(o);
                }
                dev.synchronize()?;
                c.t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / visits as f64);
            }
        }
        println!("round {round} done");
    }
    let med = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    println!("| held | layer | candle us/visit median [range] | kernel us/visit median [range] | candle bytes | kernel bytes logical / unique |");
    for c in &cases {
        let (cm, clo, chi) = med(&c.t[0]);
        let (km, klo, khi) = med(&c.t[1]);
        let n = visible(c.held - 1, c.window).1;
        let (bl, bu) = bytes_read(nh, nkv, hd, extent, n);
        println!(
            "| {} | {} | {:.1} [{:.1}, {:.1}] | {:.1} [{:.1}, {:.1}] | {} | {} / {} |",
            c.held,
            if c.window.is_some() { "sliding" } else { "full" },
            cm,
            clo,
            chi,
            km,
            klo,
            khi,
            candle_bytes(nh, nkv, hd, c.held),
            bl,
            bu
        );
    }
    Ok(())
}
