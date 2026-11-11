//! Per-operation timing of a DFlash round (`dflashcheck ... profile`).
//!
//! Every operation of every stage runs alone, the device synchronised after
//! it: 5 discarded, then 30 timed, median and range. Each stage is then timed
//! whole without the inner synchronisations. Measurement code only: nothing
//! here is on the served path, and nothing changes what the drafter drafts.
//!
//! An operation is one candle call. Where candle materialises a copy that the
//! code does not name (a `reshape` of a non-contiguous tensor, a
//! `contiguous`), the line says so; where a view costs nothing (`narrow`,
//! `transpose`, `unsqueeze`) it gets no line.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

use super::{readout_chunked, DFlashDrafter, DeviceReadout, DraftContext, TargetVocab, BLOCK, DRAFTS, MASK_ID, TREE_TOPK};

const WARM: usize = 5;
const TIMED: usize = 30;

/// Shapes and dtypes, `[a, b] F16`, with `strided` for a non-contiguous view.
pub fn sh(ts: &[&Tensor]) -> String {
    ts.iter()
        .map(|t| format!("{:?} {:?}{}", t.dims(), t.dtype(), if t.is_contiguous() { "" } else { " strided" }))
        .collect::<Vec<_>>()
        .join(" x ")
}

pub struct Profiler {
    device: Device,
    /// (stage, name, shapes, median ms, min, max)
    pub lines: Vec<(String, String, String, f64, f64, f64)>,
}

impl Profiler {
    pub fn new(device: &Device) -> Self {
        Self { device: device.clone(), lines: Vec::new() }
    }

    /// Time `f` alone: WARM discarded, TIMED kept, a synchronisation after each call.
    pub fn op<T>(&mut self, stage: &str, name: &str, shapes: String, mut f: impl FnMut() -> Result<T>) -> Result<T> {
        let mut t = Vec::with_capacity(TIMED);
        let mut last = None;
        for i in 0..WARM + TIMED {
            self.device.synchronize()?;
            let t0 = Instant::now();
            let r = f()?;
            self.device.synchronize()?;
            if i >= WARM {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            last = Some(r);
        }
        t.sort_by(f64::total_cmp);
        self.lines.push((stage.into(), name.into(), shapes, t[t.len() / 2], t[0], t[t.len() - 1]));
        Ok(last.expect("timed at least once"))
    }

    fn sum(&self, stage: &str) -> f64 {
        self.lines.iter().filter(|l| l.0 == stage).map(|l| l.3).sum()
    }

    /// Print a stage's lines, their sum, and `whole` (the stage without inner synchronisations).
    fn print(&self, stage: &str, whole: Option<(f64, f64, f64)>) {
        for l in self.lines.iter().filter(|l| l.0 == stage) {
            println!("  {:<10} {:<46} {:>8.3} ms [{:.3}-{:.3}]  {}", l.0, l.1, l.3, l.4, l.5, l.2);
        }
        print!("  {stage:<10} SUM of the lines above {:>32.3} ms", self.sum(stage));
        if let Some((m, lo, hi)) = whole {
            print!(";  whole, one synchronisation at the end: {m:.3} ms [{lo:.3}-{hi:.3}]");
        }
        println!();
    }

    /// A whole stage, synchronised only at the end.
    fn whole(&self, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64, f64)> {
        let mut t = Vec::with_capacity(TIMED);
        for i in 0..WARM + TIMED {
            self.device.synchronize()?;
            let t0 = Instant::now();
            f()?;
            self.device.synchronize()?;
            if i >= WARM {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        t.sort_by(f64::total_cmp);
        Ok((t[t.len() / 2], t[0], t[t.len() - 1]))
    }
}

/// The readout before e3bbaee (commit 1db1acb, `readout`), as it was: lse by
/// max, sub, exp, sum, log, add; an `arange` and a -inf tensor made a call;
/// four argmax-gather passes with three masks; three transfers.
pub fn readout_pre_e3bbaee(logits: &Tensor) -> Result<super::DraftOut> {
    let z = logits.to_dtype(DType::F32)?;
    let (rows, vocab) = z.dims2()?;
    let mx = z.max_keepdim(D::Minus1)?;
    let lse = (z.broadcast_sub(&mx)?.exp()?.sum_keepdim(D::Minus1)?.log()? + &mx)?;
    let ar = Tensor::arange(0u32, vocab as u32, z.device())?.unsqueeze(0)?;
    let ninf = Tensor::full(f32::NEG_INFINITY, (rows, vocab), z.device())?;
    let mut work = z.clone();
    let mut ids = Vec::with_capacity(TREE_TOPK);
    let mut vals = Vec::with_capacity(TREE_TOPK);
    for k in 0..TREE_TOPK {
        let i = work.argmax_keepdim(D::Minus1)?;
        vals.push(work.gather(&i, D::Minus1)?);
        if k + 1 < TREE_TOPK {
            work = ar.broadcast_eq(&i)?.where_cond(&ninf, &work)?;
        }
        ids.push(i);
    }
    let ids = Tensor::cat(&ids, 1)?.to_vec2::<u32>()?;
    let vals = Tensor::cat(&vals, 1)?.broadcast_sub(&lse)?.to_vec2::<f32>()?;
    let lse = lse.squeeze(1)?.to_vec1::<f32>()?;
    let mut out = super::DraftOut { chain: Vec::new(), top_ids: Vec::new(), top_logp: Vec::new(), lse };
    for r in 0..rows {
        let mut a = [0u32; TREE_TOPK];
        let mut b = [0f32; TREE_TOPK];
        a.copy_from_slice(&ids[r]);
        b.copy_from_slice(&vals[r]);
        out.chain.push(a[0]);
        out.top_ids.push(a);
        out.top_logp.push(b);
    }
    Ok(out)
}

/// One block layer, every operation timed alone under `stage`.
fn layer_ops(p: &mut Profiler, stage: &str, d: &DFlashDrafter, ctx: &DraftContext, li: usize, h: &Tensor) -> Result<Tensor> {
    let cfg = &d.cfg;
    let (hq, kvh, dh) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let rep = hq / kvh;
    let m = ctx.len;
    let all = hq + 2 * kvh;
    let l = &d.layers[li];
    let hn = p.op(stage, "rms_norm ln1", sh(&[h]), || Ok(d.rms(h, &l.ln1)?))?;
    let qkv = p.op(stage, "matmul qkv", sh(&[&hn, &l.qkv]), || Ok(super::lin(&hn, &l.qkv)?))?;
    let qkv = qkv.reshape((BLOCK, all, dh))?;
    let nq = p.op(stage, "rms_norm (unit) q/k/v heads", sh(&[&qkv]), || Ok(d.rms(&qkv, &d.ones)?))?;
    let nq = p.op(stage, "broadcast_mul qkv_norm", sh(&[&nq, &l.qkv_norm]), || Ok(nq.broadcast_mul(&l.qkv_norm)?))?;
    let nt = p.op(stage, "transpose + contiguous (copy)", sh(&[&nq]), || Ok(nq.transpose(0, 1)?.contiguous()?))?;
    let qk = nt.narrow(0, 0, hq + kvh)?.unsqueeze(0)?;
    let r = p.op(stage, "rope q and k", sh(&[&qk]), || Ok(super::DFlashDrafter::rope(ctx, &qk, m)?.squeeze(0)?))?;
    let q = r.narrow(0, 0, hq)?.reshape((kvh, rep * BLOCK, dh))?;
    let kc = ctx.k.narrow(0, li * kvh, kvh)?;
    let vc = ctx.v.narrow(0, li * kvh, kvh)?;
    let kb = r.narrow(0, hq, kvh)?;
    p.op(stage, "slice_set k into the cache", sh(&[&kb]), || Ok(kc.slice_set(&kb, 1, m)?))?;
    let vsrc = qkv.narrow(1, hq + kvh, kvh)?.transpose(0, 1)?;
    let vb = p.op(stage, "v transpose + contiguous (copy)", sh(&[&vsrc]), || Ok(vsrc.contiguous()?))?;
    p.op(stage, "slice_set v into the cache", sh(&[&vb]), || Ok(vc.slice_set(&vb, 1, m)?))?;
    let kc = kc.narrow(1, 0, m + BLOCK)?;
    let vc = vc.narrow(1, 0, m + BLOCK)?;
    let kt = kc.t()?;
    let s = p.op(stage, "matmul scores q k^T", sh(&[&q, &kt]), || Ok(q.matmul(&kt)?))?;
    let pr = p.op(stage, "softmax_last_dim scores", sh(&[&s]), || Ok(candle_nn::ops::softmax_last_dim(&s)?))?;
    let att = p.op(stage, "matmul p v", sh(&[&pr, &vc]), || Ok(pr.matmul(&vc)?))?;
    let at = att.reshape((hq, BLOCK, dh))?.transpose(0, 1)?;
    let att = p.op(stage, "transpose + reshape (copy)", sh(&[&at]), || Ok(at.reshape((BLOCK, hq * dh))?))?;
    let o = p.op(stage, "matmul o", sh(&[&att, &l.o]), || Ok(super::lin(&att, &l.o)?))?;
    let h = p.op(stage, "add residual", sh(&[h, &o]), || Ok((h + &o)?))?;
    let m2 = p.op(stage, "rms_norm ln2", sh(&[&h]), || Ok(d.rms(&h, &l.ln2)?))?;
    let gu = p.op(stage, "matmul gate_up", sh(&[&m2, &l.gate_up]), || Ok(super::lin(&m2, &l.gate_up)?))?;
    let f = cfg.intermediate;
    let g = gu.narrow(1, 0, f)?;
    let u = gu.narrow(1, f, f)?;
    let sg = p.op(stage, "silu gate", sh(&[&g]), || Ok(candle_nn::ops::silu(&g)?))?;
    let a = p.op(stage, "mul silu(gate) up", sh(&[&sg, &u]), || Ok((&sg * &u)?))?;
    let dn = p.op(stage, "matmul down", sh(&[&a, &l.down]), || Ok(super::lin(&a, &l.down)?))?;
    let h = p.op(stage, "add residual", sh(&[&h, &dn]), || Ok((&h + &dn)?))?;
    Ok(h)
}

/// Every stage of a round at context `ctx0`, operation by operation.
/// `new_taps` are host binary16 bits [n][5 x hidden], as the loop hands them.
pub fn run(d: &DFlashDrafter, vocab: &dyn TargetVocab, ctx0_taps: &Tensor, new_taps: &[u16], n: usize) -> Result<()> {
    let dev = d.device().clone();
    let mut p = Profiler::new(&dev);
    let w = new_taps.len() / n;
    println!(
        "PROFILE on {dev:?}, drafter {:?}: every operation alone, a device synchronisation after each, median of {TIMED} after {WARM}. \
         {}",
        d.dtype,
        if dev.is_cuda() { "" } else { "NOT A CUDA DEVICE: these times say nothing about the card; they only show that the mode runs." }
    );
    let mut ctx = d.new_context(ctx0_taps.dim(0)? + 64)?;
    d.extend(&mut ctx, ctx0_taps)?;
    let len0 = ctx.len;
    println!("context {len0} positions, extension of {n}");

    // EXTENSION, as DFlashSpec runs it from the loop's host bits
    let st = "extend";
    let taps = p.op(st, "host u16 -> f16, from_vec (upload)", format!("[{n}, {w}] host u16"), || {
        let bits: Vec<half::f16> = new_taps.iter().map(|&b| half::f16::from_bits(b)).collect();
        Ok(Tensor::from_vec(bits, (n, w), &dev)?)
    })?;
    let taps = p.op(st, "to_dtype f32 (adapter)", sh(&[&taps]), || Ok(taps.to_dtype(DType::F32)?))?;
    let fcout = p.op(st, "matmul fc (f32)", sh(&[&taps, &d.fc]), || Ok(super::lin(&taps, &d.fc)?))?;
    let c = p.op(st, "rms_norm hidden_norm (f32)", sh(&[&fcout]), || Ok(d.rms(&fcout, &d.hidden_norm)?))?;
    let c = p.op(st, "to_dtype drafter dtype", sh(&[&c]), || Ok(c.to_dtype(d.dtype)?))?;
    let heads = d.cfg.layers * d.cfg.kv_heads;
    let dh = d.cfg.head_dim;
    let k = p.op(st, "matmul k_all", sh(&[&c, &d.k_all]), || Ok(super::lin(&c, &d.k_all)?))?;
    let k = k.reshape((n, heads, dh))?;
    let k = p.op(st, "rms_norm (unit) k", sh(&[&k]), || Ok(d.rms(&k, &d.ones)?))?;
    let k = p.op(st, "broadcast_mul k_norm_all", sh(&[&k, &d.k_norm_all]), || Ok(k.broadcast_mul(&d.k_norm_all)?))?;
    let kt = p.op(st, "transpose + contiguous (copy)", sh(&[&k]), || Ok(k.transpose(0, 1)?.contiguous()?.unsqueeze(0)?))?;
    let k = p.op(st, "rope k", sh(&[&kt]), || Ok(DFlashDrafter::rope(&ctx, &kt, len0)?.squeeze(0)?))?;
    let v = p.op(st, "matmul v_all", sh(&[&c, &d.v_all]), || Ok(super::lin(&c, &d.v_all)?))?;
    let vt = v.reshape((n, heads, dh))?.transpose(0, 1)?;
    let v = p.op(st, "v transpose + contiguous (copy)", sh(&[&vt]), || Ok(vt.contiguous()?))?;
    p.op(st, "slice_set k", sh(&[&k]), || Ok(ctx.k.slice_set(&k, 1, len0)?))?;
    p.op(st, "slice_set v", sh(&[&v]), || Ok(ctx.v.slice_set(&v, 1, len0)?))?;
    let whole = p.whole(|| {
        let bits: Vec<half::f16> = new_taps.iter().map(|&b| half::f16::from_bits(b)).collect();
        let t = Tensor::from_vec(bits, (n, w), &dev)?.to_dtype(DType::F32)?;
        ctx.truncate(len0);
        d.extend(&mut ctx, &t)?;
        Ok(())
    })?;
    ctx.truncate(len0);
    p.print(st, Some(whole));

    // BLOCK
    let st = "block";
    let mut ids = vec![MASK_ID; BLOCK];
    ids[0] = 151643;
    let idt = p.op(st, "from_vec block ids (upload)", format!("[{BLOCK}] host u32"), || Ok(Tensor::from_vec(ids.clone(), BLOCK, &dev)?))?;
    let h0 = p.op(st, "embed (index_select)", sh(&[&idt]), || Ok(vocab.embed(&idt)?))?;
    let h0 = p.op(st, "to_dtype drafter dtype (no-op if equal)", sh(&[&h0]), || Ok(h0.to_dtype(d.dtype)?))?;
    let mut h = h0.clone();
    let mut layer_sums = Vec::new();
    for li in 0..d.cfg.layers {
        let tag = format!("layer{li}");
        h = layer_ops(&mut p, &tag, d, &ctx, li, &h)?;
        layer_sums.push(p.sum(&tag));
    }
    let hl = h.narrow(0, 1, DRAFTS)?;
    let hid = p.op(st, "rms_norm final (rows 1..15)", sh(&[&hl]), || Ok(d.rms(&hl, &d.norm)?))?;
    let whole_block = p.whole(|| {
        d.block_hidden(&ctx, 151643, vocab)?;
        Ok(())
    })?;
    p.print("layer0", None);
    println!(
        "  block      per layer, sum of its lines: {}; five layers {:.3} ms",
        layer_sums.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", "),
        layer_sums.iter().sum::<f64>()
    );
    p.print(st, None);
    println!(
        "  block      SUM block lines + five layers {:>27.3} ms;  whole, one synchronisation at the end: {:.3} ms [{:.3}-{:.3}]",
        p.sum(st) + layer_sums.iter().sum::<f64>(),
        whole_block.0,
        whole_block.1,
        whole_block.2
    );

    // HEAD (the vocabulary's logits: for the dense table, the matmul and the cast to f32)
    let st = "head";
    let logits = p.op(st, "vocab.logits (matmul + to_dtype f32)", sh(&[&hid]), || Ok(vocab.logits(&hid)?))?;
    let whole = p.whole(|| {
        vocab.logits(&hid)?;
        Ok(())
    })?;
    p.print(st, Some(whole));
    let z = logits.to_dtype(DType::F32)?;
    let z16 = z.to_dtype(DType::F16)?;
    let (rows, vocab_n) = z.dims2()?;

    // READOUT, device (DeviceReadout::run, unrolled)
    let st = "rd-device";
    let dr = DeviceReadout::new(rows, vocab_n, &dev)?;
    let z_ = p.op(st, "to_dtype f32 (no-op on f32 logits)", sh(&[&z]), || Ok(z.to_dtype(DType::F32)?))?;
    let mut work = z_.clone();
    let mut idv = Vec::new();
    for k in 0..TREE_TOPK {
        let i = p.op(st, &format!("argmax_keepdim #{k}"), sh(&[&work]), || Ok(work.argmax_keepdim(D::Minus1)?))?;
        if k + 1 < TREE_TOPK {
            let eq = p.op(st, &format!("broadcast_eq arange #{k}"), sh(&[&dr.arange, &i]), || Ok(dr.arange.broadcast_eq(&i)?))?;
            work = p.op(st, &format!("where_cond -inf #{k}"), sh(&[&eq, &dr.ninf, &work]), || Ok(eq.where_cond(&dr.ninf, &work)?))?;
        }
        idv.push(i);
    }
    let ids4 = p.op(st, "cat ids", sh(&[&idv[0]]), || Ok(Tensor::cat(&idv, 1)?))?;
    let zv = p.op(st, "gather logits", sh(&[&z_, &ids4]), || Ok(z_.gather(&ids4, 1)?))?;
    let sm = p.op(st, "softmax_last_dim (full vocabulary)", sh(&[&z_]), || Ok(candle_nn::ops::softmax_last_dim(&z_)?))?;
    let pv = p.op(st, "gather probabilities", sh(&[&sm, &ids4]), || Ok(sm.gather(&ids4, 1)?))?;
    let idf = p.op(st, "to_dtype ids u32 -> f32", sh(&[&ids4]), || Ok(ids4.to_dtype(DType::F32)?))?;
    let packed = p.op(st, "cat packed", sh(&[&idf, &zv, &pv]), || Ok(Tensor::cat(&[&idf, &zv, &pv], 1)?))?;
    p.op(st, "to_vec2 (device-to-host)", sh(&[&packed]), || Ok(packed.to_vec2::<f32>()?))?;
    let whole = p.whole(|| {
        dr.run(&z)?;
        Ok(())
    })?;
    p.print(st, Some(whole));

    // READOUT, chunked (readout_chunked, unrolled)
    let st = "rd-chunked";
    let c = 128usize;
    let nch = vocab_n / c;
    let zc = z.reshape((rows, nch, c))?;
    let cm = p.op(st, "max over chunks of 128", sh(&[&zc]), || Ok(zc.max(D::Minus1)?))?;
    let cmax = p.op(st, "to_vec2 chunk maxima (device-to-host)", sh(&[&cm]), || Ok(cm.to_vec2::<f32>()?))?;
    let pick = p.op(st, "host: 4 best chunks a row", format!("[{rows}, {nch}] host"), || {
        let mut pick = Vec::with_capacity(rows * TREE_TOPK * c);
        for row in &cmax {
            let mut best: Vec<u32> = Vec::new();
            for _ in 0..TREE_TOPK {
                let mut b: Option<(usize, f32)> = None;
                for (i, &x) in row.iter().enumerate() {
                    if !best.contains(&(i as u32)) && b.is_none_or(|(_, y)| x > y) {
                        b = Some((i, x));
                    }
                }
                best.push(b.expect("a chunk").0 as u32);
            }
            for &ch in &best {
                pick.extend(std::iter::repeat_n(ch, c));
            }
        }
        Ok(pick)
    })?;
    let idx = p.op(st, "from_vec gather index (upload)", format!("[{rows}, {TREE_TOPK}, {c}] host u32"), || {
        Ok(Tensor::from_vec(pick.clone(), (rows, TREE_TOPK, c), &dev)?)
    })?;
    let sm = p.op(st, "softmax_last_dim (full vocabulary)", sh(&[&z]), || Ok(candle_nn::ops::softmax_last_dim(&z)?))?;
    let smc = sm.reshape((rows, nch, c))?;
    let gz = p.op(st, "gather logits", sh(&[&zc, &idx]), || Ok(zc.gather(&idx, 1)?))?;
    let gp = p.op(st, "gather probabilities", sh(&[&smc, &idx]), || Ok(smc.gather(&idx, 1)?))?;
    let both = p.op(st, "cat", sh(&[&gz, &gp]), || Ok(Tensor::cat(&[&gz, &gp], 1)?))?;
    p.op(st, "to_vec3 (device-to-host)", sh(&[&both]), || Ok(both.to_vec3::<f32>()?))?;
    let whole = p.whole(|| {
        readout_chunked(&z)?;
        Ok(())
    })?;
    p.print(st, Some(whole));

    // CANDIDATES on the same logits
    let st = "cand-a";
    p.op(st, "max_keepdim (f32)", sh(&[&z]), || Ok(z.max_keepdim(D::Minus1)?))?;
    let am = p.op(st, "argmax_keepdim (f32)", sh(&[&z]), || Ok(z.argmax_keepdim(D::Minus1)?))?;
    p.op(st, "to_vec2 argmax (device-to-host)", sh(&[&am]), || Ok(am.to_vec2::<u32>()?))?;
    p.op(st, "max_keepdim (f16)", sh(&[&z16]), || Ok(z16.max_keepdim(D::Minus1)?))?;
    p.op(st, "argmax_keepdim (f16)", sh(&[&z16]), || Ok(z16.argmax_keepdim(D::Minus1)?))?;
    let whole = p.whole(|| {
        let m = z.max_keepdim(D::Minus1)?;
        let a = z.argmax_keepdim(D::Minus1)?;
        Tensor::cat(&[&a.to_dtype(DType::F32)?, &m], 1)?.to_vec2::<f32>()?;
        Ok(())
    })?;
    p.print(st, Some(whole));

    let st = "cand-b";
    let ninf16 = Tensor::full(f32::NEG_INFINITY, (rows, vocab_n), &dev)?.to_dtype(DType::F16)?;
    let mut work = z16.clone();
    let mut idv = Vec::new();
    for k in 0..TREE_TOPK {
        let i = p.op(st, &format!("argmax_keepdim f16 #{k}"), sh(&[&work]), || Ok(work.argmax_keepdim(D::Minus1)?))?;
        if k + 1 < TREE_TOPK {
            let eq = p.op(st, &format!("broadcast_eq arange #{k}"), sh(&[&dr.arange, &i]), || Ok(dr.arange.broadcast_eq(&i)?))?;
            work = p.op(st, &format!("where_cond -inf f16 #{k}"), sh(&[&eq, &ninf16, &work]), || Ok(eq.where_cond(&ninf16, &work)?))?;
        }
        idv.push(i);
    }
    let ids4 = Tensor::cat(&idv, 1)?;
    p.op(st, "gather logits f16", sh(&[&z16, &ids4]), || Ok(z16.gather(&ids4, 1)?))?;
    for (label, zz) in [("f32", &z), ("f16 (sum may overflow past 65,504)", &z16)] {
        let mx = p.op(st, &format!("lse {label}: max_keepdim"), sh(&[zz]), || Ok(zz.max_keepdim(D::Minus1)?))?;
        let sb = p.op(st, &format!("lse {label}: broadcast_sub"), sh(&[zz, &mx]), || Ok(zz.broadcast_sub(&mx)?))?;
        let ex = p.op(st, &format!("lse {label}: exp"), sh(&[&sb]), || Ok(sb.exp()?))?;
        let su = p.op(st, &format!("lse {label}: sum_keepdim"), sh(&[&ex]), || Ok(ex.sum_keepdim(D::Minus1)?))?;
        let lg = p.op(st, &format!("lse {label}: log"), sh(&[&su]), || Ok(su.log()?))?;
        p.op(st, &format!("lse {label}: add"), sh(&[&lg, &mx]), || Ok((&lg + &mx)?))?;
    }
    let whole = p.whole(|| {
        let mut work = z16.clone();
        let mut idv = Vec::new();
        for k in 0..TREE_TOPK {
            let i = work.argmax_keepdim(D::Minus1)?;
            if k + 1 < TREE_TOPK {
                work = dr.arange.broadcast_eq(&i)?.where_cond(&ninf16, &work)?;
            }
            idv.push(i);
        }
        let ids4 = Tensor::cat(&idv, 1)?;
        let zv = z.gather(&ids4, 1)?;
        let mx = z.max_keepdim(D::Minus1)?;
        let lse = (z.broadcast_sub(&mx)?.exp()?.sum_keepdim(D::Minus1)?.log()? + &mx)?;
        Tensor::cat(&[&ids4.to_dtype(DType::F32)?, &zv, &lse], 1)?.to_vec2::<f32>()?;
        Ok(())
    })?;
    p.print(st, Some(whole));
    println!("  cand-b     whole = top 4 on the f16 logits, lse on f32 (max, sub, exp, sum, log, add), one transfer");

    let st = "cand-c";
    let whole = p.whole(|| {
        readout_pre_e3bbaee(&z)?;
        Ok(())
    })?;
    p.op(st, "arange vocab (made each call)", format!("[{vocab_n}] u32"), || Ok(Tensor::arange(0u32, vocab_n as u32, &dev)?))?;
    p.op(st, "full -inf (made each call)", format!("[{rows}, {vocab_n}] F32"), || Ok(Tensor::full(f32::NEG_INFINITY, (rows, vocab_n), &dev)?))?;
    p.print(st, Some(whole));
    println!("  cand-c     whole = the readout of 1db1acb..e3bbaee^ as it was (lse by six operations, 4 argmax + gather, 3 masks, 3 transfers)");
    Ok(())
}
