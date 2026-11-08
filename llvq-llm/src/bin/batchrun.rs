//! Batched decode on a card: B sequences a step, aggregate throughput, and the
//! tokens each sequence gets against the batch-1 path.
//!
//!   LLVQ_CONFIG=<served config> batchrun <sealed> door  <B,B,..> <rows,mma,gemm> [tokens] [rounds]
//!   LLVQ_CONFIG=<served config> batchrun <sealed> check [mma|gemm]
//!   LLVQ_CONFIG=<served config> batchrun <sealed> gsm8k <B> <rows|mma|gemm> <limit>
//!   LLVQ_CONFIG=<served config> batchrun <sealed> ppl <ctx> <windows> [mma|gemm]
//!   LLVQ_CONFIG=<served config> batchrun <sealed> kernels <B,B,..> [arm] [reps]
//!
//! `door`: B copies of the door prompt ("The capital of France is", raw ids),
//! `tokens` greedy tokens each (default 256), the prompt fed through the batch
//! one token a step, wall clock around the whole generation, one warm-up then
//! `rounds` rounds (default 3), every (B, arm) once a round. Prints the
//! aggregate rate, its median and range, steps per second, and for every row
//! whether its tokens equal `LLVQ_BATCH_REF` (the door's fingerprint).
//!
//! `check`: layer 0's seven projections and the head through the tensor-core
//! kernels of the arm (default `gemm`) against the multi-row path, on random
//! activations, at every row count the arm's entries take; under `gemm` also
//! q, k, v and gate, up as the step launches them, one launch a group: the
//! relative RMS difference.
//!
//! `gsm8k`: the pilot problems of `bin/gsm8k` (`gsm8k::select(.., limit)`),
//! each prompt prefilled through the batch-1 path, the decode batched B at a
//! time; dump at `LLVQ_GSM8K_DUMP` in `bin/gsm8k`'s format.
//!
//! `ppl`: the first `windows` windows of `ctx` tokens of the raw WikiText-2
//! test split (as `bin/ppl` cuts them), every position scored by a decode row,
//! all windows in one batch, under the multi-row path and the arm (default
//! `gemm`); the paired difference window by window, pooled, and its 95 %
//! interval (paired t over windows).
//!
//! `kernels`: the projections of one step alone, B random rows, every layer's
//! four groups (q,k,v; o; gate,up; down) and the head, as the step launches
//! them (rotation and conversion included), timed per group over `reps` steps
//! (default 10) after two warm-ups: where the step's time goes.
//!
//! Every `LLVQ_*` variable but `LLVQ_CONFIG`, `LLVQ_BATCH_REF`,
//! `LLVQ_GSM8K_DUMP` and `LLVQ_DATASET_REV` is refused.

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    run::main()
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("batchrun runs on a CUDA build on Linux: the batched step's kernels are CUDA only")
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
mod run {
    use std::io::Write;
    use std::time::Instant;

    use candle_core::{DType, Tensor};
    use llvq_llm::batch::{BatchDevice, BatchProjs};
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    use llvq_llm::gsm8k::{self, Row};
    use llvq_llm::model::{Proj, Qwen3};

    const DOOR: [u32; 5] = [785, 6722, 315, 9625, 374];
    const ALLOWED: [&str; 4] = ["LLVQ_CONFIG", "LLVQ_BATCH_REF", "LLVQ_GSM8K_DUMP", "LLVQ_DATASET_REV"];

    fn median(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let n = s.len();
        if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
    }

    pub fn main() -> anyhow::Result<()> {
        for (k, _) in std::env::vars() {
            anyhow::ensure!(!k.starts_with("LLVQ_") || ALLOWED.contains(&k.as_str()), "refused: {k} is set; batchrun reads {ALLOWED:?} only");
        }
        let a: Vec<String> = std::env::args().skip(1).collect();
        let path = a.first().cloned().ok_or_else(|| anyhow::anyhow!("usage: batchrun <sealed> door|check|gsm8k ..."))?;
        let mode = a.get(1).map(String::as_str).unwrap_or("door");
        let cfg = llvq_llm::served::Served::from_env()
            .map_err(anyhow::Error::msg)?
            .ok_or_else(|| anyhow::anyhow!("batchrun serves a config: set LLVQ_CONFIG"))?;
        println!("{}", cfg.provenance());
        let device = candle_core::Device::new_cuda(0)?;
        let t0 = Instant::now();
        let f = llvq_llm::served::load_resolved(&path, &device, DType::F16, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv, cfg.int4_layout(), Some("LLVQ_CONFIG"))?;
        let (mut model, tok) = (f.model, f.tokenizer);
        cfg.apply(&mut model, &device).map_err(anyhow::Error::msg)?;
        let (_, groups, _) = model.attn_geometry();
        let rt = llvq_llm::batch_cuda::runtime_of(&model)?;
        let dev = llvq_llm::batch_cuda::CudaBatch::new(rt, groups)?;
        for l in &dev.provenance {
            println!("{l}");
        }
        println!("loaded in {:.1} s, attention {} on {}", t0.elapsed().as_secs_f64(), model.attn_heads().label(), model.kv_store().label());
        match mode {
            "door" => door(&model, &dev, &a[2..]),
            "check" => check(&model, &dev, &a[2..]),
            "gsm8k" => gsm(&model, &dev, &tok, &a[2..]),
            "ppl" => ppl(&model, &dev, &tok, &a[2..]),
            "kernels" => kernels(&model, &dev, &a[2..]),
            m => anyhow::bail!("mode {m:?}: door, check, gsm8k, ppl or kernels"),
        }
    }

    fn door(model: &Qwen3, dev: &dyn BatchDevice, a: &[String]) -> anyhow::Result<()> {
        let bs: Vec<usize> = a.first().map(String::as_str).unwrap_or("1,4,8,16,32").split(',').map(str::parse).collect::<Result<_, _>>()?;
        let arms: Vec<BatchProjs> = a.get(1).map(String::as_str).unwrap_or("rows,mma").split(',').map(BatchProjs::parse).collect::<Result<_, _>>().map_err(anyhow::Error::msg)?;
        let n: usize = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(256);
        let rounds: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3);
        // A B past an arm's widest entry skips that arm; every B must fit one.
        for &b in &bs {
            anyhow::ensure!(b >= 1 && arms.iter().any(|&a| b <= dev.max_rows(a)), "B = {b}: no arm takes it");
        }
        let reference = std::env::var("LLVQ_BATCH_REF").ok();
        let mut rates: Vec<Vec<f64>> = vec![Vec::new(); bs.len() * arms.len()];
        let mut ident: Vec<String> = vec![String::new(); bs.len() * arms.len()];
        for r in 0..=rounds {
            for (ib, &b) in bs.iter().enumerate() {
                for (ia, &arm) in arms.iter().enumerate() {
                    if b > dev.max_rows(arm) {
                        continue;
                    }
                    let prompts = vec![DOOR.to_vec(); b];
                    model.device().synchronize()?;
                    let t = Instant::now();
                    let out = model.generate_batch(&prompts, n, dev, arm, false, &|_, _| false)?;
                    model.device().synchronize()?;
                    let dt = t.elapsed().as_secs_f64();
                    let rate = (b * n) as f64 / dt;
                    let k = ib * arms.len() + ia;
                    if r == 0 {
                        let fps: Vec<String> = out.iter().map(|o| format!("{:016x}", llvq_llm::eval::token_fingerprint(o))).collect();
                        let same = match &reference {
                            Some(want) => fps.iter().filter(|f| *f == want).count(),
                            None => fps.iter().filter(|f| **f == fps[0]).count(),
                        };
                        let first_diff: Vec<String> = out
                            .iter()
                            .filter(|o| reference.as_deref().is_some_and(|w| format!("{:016x}", llvq_llm::eval::token_fingerprint(o)) != w))
                            .take(1)
                            .map(|o| format!(" row tokens {:?}", &o[..o.len().min(24)]))
                            .collect();
                        ident[k] = format!("{same} of {b} rows equal {}{}", reference.as_deref().unwrap_or("row 0"), first_diff.concat());
                        println!("  B={b:>2} {:<4} warm-up {rate:8.1} tok/s; tokens: {}; row 0 {}", arm.name(), ident[k], fps[0]);
                    } else {
                        rates[k].push(rate);
                        println!("  round {r} B={b:>2} {:<4} {rate:8.1} tok/s aggregate ({dt:.3} s)", arm.name());
                    }
                }
            }
        }
        println!("\nbatched door: {n} tokens a sequence after the 5-token prompt fed through the batch, prompt and decode, {rounds} rounds");
        println!("{:>3} {:<4} {:>16} {:>22} {:>12} {:>14}  tokens", "B", "arm", "tok/s aggregate", "range", "steps/s", "tok/s a seq");
        for (ib, &b) in bs.iter().enumerate() {
            for (ia, &arm) in arms.iter().enumerate() {
                let k = ib * arms.len() + ia;
                let v = &rates[k];
                if v.is_empty() {
                    continue;
                }
                let m = median(v);
                let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = v.iter().cloned().fold(0.0, f64::max);
                println!("{b:>3} {:<4} {m:>16.1} [{mn:>9.1}-{mx:>9.1}] {:>12.2} {:>14.1}  {}", arm.name(), m / b as f64 * ((n + DOOR.len() - 1) as f64 / n as f64), m / b as f64, ident[k]);
            }
        }
        Ok(())
    }

    fn kernels(model: &Qwen3, dev: &dyn BatchDevice, a: &[String]) -> anyhow::Result<()> {
        let bs: Vec<usize> = a.first().map(String::as_str).unwrap_or("1,8,32,128").split(',').map(str::parse).collect::<Result<_, _>>()?;
        let arm = BatchProjs::parse(a.get(1).map(String::as_str).unwrap_or("gemm")).map_err(anyhow::Error::msg)?;
        let reps: usize = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(10);
        let c = model.config();
        let (hid, qd, inter) = (c.hidden_size, c.num_attention_heads * c.head_dim, c.intermediate_size);
        let share = model.blocks[0].rot_share();
        println!("projections alone, arm {}, {} layers, {reps} steps a figure, ms a step", arm.name(), model.blocks.len());
        println!("{:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "B", "q,k,v", "o", "gate,up", "down", "head", "total");
        for &b in &bs {
            anyhow::ensure!(b >= 1 && b <= dev.max_rows(arm), "B = {b} past the arm's {}", dev.max_rows(arm));
            let r = |d: usize| -> anyhow::Result<Tensor> { Ok(Tensor::randn(0f32, 1f32, (b, 1, d), model.device())?.to_dtype(DType::F16)?) };
            let (xh, xq, xi) = (r(hid)?, r(qd)?, r(inter)?);
            let mut ms = [0f64; 5];
            for (g, m) in ms.iter_mut().enumerate() {
                let once = || -> anyhow::Result<()> {
                    if g == 4 {
                        model.batch_head(&xh, dev, arm)?;
                        return Ok(());
                    }
                    for blk in &model.blocks {
                        let (ps, x): (Vec<&Proj>, &Tensor) = match g {
                            0 => (vec![&blk.q_proj, &blk.k_proj, &blk.v_proj], &xh),
                            1 => (vec![&blk.o_proj], &xq),
                            2 => (vec![&blk.gate_proj, &blk.up_proj], &xh),
                            _ => (vec![&blk.down_proj], &xi),
                        };
                        llvq_llm::model::batch_group(&ps, x, share, dev, arm)?;
                    }
                    Ok(())
                };
                once()?;
                once()?;
                model.device().synchronize()?;
                let t = Instant::now();
                for _ in 0..reps {
                    once()?;
                }
                model.device().synchronize()?;
                *m = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            }
            let tot: f64 = ms.iter().sum();
            println!("{b:>4} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {tot:>9.3}", ms[0], ms[1], ms[2], ms[3], ms[4]);
        }
        Ok(())
    }

    fn rel(a: &Tensor, b: &Tensor) -> anyhow::Result<(f64, f64)> {
        let a: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (mut se, mut sr, mut mx) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            let e = (*x as f64 - *y as f64).abs();
            se += e * e;
            sr += (*y as f64).powi(2);
            mx = mx.max(e);
        }
        let rms = (sr / b.len() as f64).sqrt();
        Ok(((se / b.len() as f64).sqrt() / rms, mx / rms))
    }

    fn check(model: &Qwen3, dev: &dyn BatchDevice, a: &[String]) -> anyhow::Result<()> {
        let arm = BatchProjs::parse(a.first().map(String::as_str).unwrap_or("gemm")).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(arm != BatchProjs::Rows, "check compares a tensor-core arm with rows");
        let counts: &[usize] = match arm {
            BatchProjs::Gemm | BatchProjs::GemmOcc => &[1, 8, 13, 32, 64, 128],
            _ => &[1, 8, 16, 32],
        };
        let blk = &model.blocks[0];
        let named: [(&str, &Proj); 7] = [
            ("q_proj", &blk.q_proj), ("k_proj", &blk.k_proj), ("v_proj", &blk.v_proj), ("o_proj", &blk.o_proj),
            ("gate_proj", &blk.gate_proj), ("up_proj", &blk.up_proj), ("down_proj", &blk.down_proj),
        ];
        let mut worst = 0f64;
        for &n in counts {
            for (name, p) in named {
                let d_in = match p {
                    Proj::Lattice(d) => d.d_in(),
                    Proj::Int4(d) => d.d_in(),
                    _ => anyhow::bail!("{name}: neither lattice nor int4"),
                };
                let x = (Tensor::randn(0f32, 1f32, (n, 1, d_in), model.device())?.to_dtype(DType::F16))?;
                let share = llvq_llm::rotplan::RotShare::Off;
                let a = llvq_llm::model::batch_group(&[p], &x, share, dev, arm)?;
                let b = llvq_llm::model::batch_group(&[p], &x, share, dev, BatchProjs::Rows)?;
                let (r, m) = rel(&a[0], &b[0])?;
                worst = worst.max(r);
                let kind = if matches!(p, Proj::Int4(_)) { "int4" } else { "tetra" };
                println!("check {name:<9} {kind:<5} rows {n:>2}: relative RMS {r:.3e}, max {m:.3e} of the RMS");
            }
            let h = Tensor::randn(0f32, 1f32, (n, 1, model.config().hidden_size), model.device())?.to_dtype(DType::F16)?;
            let a = model.batch_head(&h, dev, arm)?;
            let b = model.batch_head(&h, dev, BatchProjs::Rows)?;
            let (r, m) = rel(&a, &b)?;
            worst = worst.max(r);
            println!("check lm_head   int4  rows {n:>2}: relative RMS {r:.3e}, max {m:.3e} of the RMS");
            if arm.is_gemm() {
                // The groups as the step launches them, under the served share.
                let share = blk.rot_share();
                let groups: [(&str, Vec<&Proj>, usize); 2] = [
                    ("q,k,v", vec![&blk.q_proj, &blk.k_proj, &blk.v_proj], model.config().hidden_size),
                    ("gate,up", vec![&blk.gate_proj, &blk.up_proj], model.config().hidden_size),
                ];
                for (gname, ps, d) in groups {
                    let x = Tensor::randn(0f32, 1f32, (n, 1, d), model.device())?.to_dtype(DType::F16)?;
                    let a = llvq_llm::model::batch_group(&ps, &x, share, dev, arm)?;
                    let b = llvq_llm::model::batch_group(&ps, &x, share, dev, BatchProjs::Rows)?;
                    for (i, (ya, yb)) in a.iter().zip(&b).enumerate() {
                        let (r, m) = rel(ya, yb)?;
                        worst = worst.max(r);
                        println!("check {gname:<9} part {i} rows {n:>3}: relative RMS {r:.3e}, max {m:.3e} of the RMS");
                    }
                }
            }
        }
        println!("check: worst relative RMS {worst:.3e} (gate 1e-2)");
        anyhow::ensure!(worst < 1e-2, "the tensor-core kernels disagree with the multi-row path: {worst:.3e}");
        Ok(())
    }

    fn gsm(model: &Qwen3, dev: &dyn BatchDevice, tok: &tokenizers::Tokenizer, a: &[String]) -> anyhow::Result<()> {
        let b: usize = a.first().ok_or_else(|| anyhow::anyhow!("gsm8k <B> <arm> <limit>"))?.parse()?;
        let arm = BatchProjs::parse(a.get(1).map(String::as_str).unwrap_or("mma")).map_err(anyhow::Error::msg)?;
        let limit: usize = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(50);
        let max_new = 1024usize;
        let marks = Marks::resolve(tok)?;
        let stops = marks.stops();
        let (items, revision) = llvq_llm::corpus::gsm8k_split("test")?;
        let picked = gsm8k::select(items.len(), limit);
        let mut prompts: Vec<Vec<u32>> = Vec::new();
        for &i in &picked {
            let mut ids = turn(tok, &marks, "user", &gsm8k::user_body(&items[i].question))?;
            ids.extend(open_assistant(tok, &marks, false)?);
            prompts.push(ids);
        }
        eprintln!("GSM8K test: {} problems, scoring {}, revision {}, batch {b}, arm {}", items.len(), picked.len(), revision.as_deref().unwrap_or("?"), arm.name());
        let mut dump = match std::env::var("LLVQ_GSM8K_DUMP") {
            Ok(p) if !p.is_empty() => {
                let mut w = std::io::BufWriter::new(std::fs::File::create(&p)?);
                writeln!(w, "{}", serde_json::json!({gsm8k::DUMP_TAG: gsm8k::DUMP_VERSION, "model": "batchrun", "arithmetic": format!("batched {} B={b}", arm.name()), "max_new": max_new, "think": false, "questions": picked.len(), "revision": revision, "instruction": gsm8k::INSTRUCTION}))?;
                Some(w)
            }
            _ => None,
        };
        let t = Instant::now();
        let (mut right, mut gen) = (0usize, 0usize);
        let mut stream: Vec<u32> = Vec::new();
        for (c, chunk) in picked.chunks(b).zip(prompts.chunks(b)) {
            let outs = model.generate_batch(chunk, max_new, dev, arm, true, &|_, g: &[u32]| g.last().is_some_and(|t| stops.contains(t)))?;
            for ((&i, ids), mut said) in c.iter().zip(chunk).zip(outs) {
                let stop = if said.last().is_some_and(|t| stops.contains(t)) { said.pop(); "eos" } else { "cap" };
                gen += said.len();
                stream.extend_from_slice(ids);
                let completion = tok.decode(&said, false).map_err(|e| anyhow::anyhow!("{e}"))?;
                let gold = gsm8k::gold(&items[i].answer).ok_or_else(|| anyhow::anyhow!("row {i}: no gold"))?;
                let (source, extracted) = gsm8k::extract(&completion);
                let correct = gsm8k::is_correct(extracted.as_deref(), &gold);
                right += correct as usize;
                let row = Row {
                    index: i,
                    qhash: format!("{:016x}", llvq_llm::eval::token_fingerprint(ids)),
                    n_prompt: ids.len(),
                    n_gen: said.len(),
                    stop: stop.to_string(),
                    gold,
                    extracted,
                    source: source.name().to_string(),
                    correct,
                    prefill_s: 0.0,
                    decode_s: 0.0,
                    completion,
                };
                if let Some(w) = dump.as_mut() {
                    writeln!(w, "{}", serde_json::to_string(&row)?)?;
                }
            }
        }
        let wall = t.elapsed().as_secs_f64();
        let fp = llvq_llm::eval::token_fingerprint(&stream);
        if let Some(w) = dump.as_mut() {
            writeln!(w, "{}", serde_json::json!({"end": true, "fingerprint": format!("{fp:016x}"), "questions": picked.len()}))?;
            w.flush()?;
        }
        println!("GSM8K batched ({}, B={b}): {right}/{} correct, {gen} tokens generated in {wall:.1} s ({:.1} tok/s of kept tokens), fingerprint {fp:016x}", arm.name(), picked.len(), gen as f64 / wall);
        Ok(())
    }

    fn ppl(model: &Qwen3, dev: &dyn BatchDevice, tok: &tokenizers::Tokenizer, a: &[String]) -> anyhow::Result<()> {
        let ctx: usize = a.first().map(|s| s.parse()).transpose()?.unwrap_or(2048);
        let nwin: usize = a.get(1).map(|s| s.parse()).transpose()?.unwrap_or(16);
        let arm = BatchProjs::parse(a.get(2).map(String::as_str).unwrap_or("gemm")).map_err(anyhow::Error::msg)?;
        anyhow::ensure!((1..=dev.max_rows(arm)).contains(&nwin), "windows in 1..={}", dev.max_rows(arm));
        let text = llvq_llm::corpus::wikitext2_test()?;
        let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        anyhow::ensure!(ids.len() >= nwin * ctx, "corpus shorter than {nwin} windows of {ctx}");
        let fp = llvq_llm::eval::token_fingerprint(&ids[..nwin * ctx]);
        let wins: Vec<&[u32]> = (0..nwin).map(|w| &ids[w * ctx..(w + 1) * ctx]).collect();
        let t = Instant::now();
        let r = model.batch_window_nll(&wins, dev, BatchProjs::Rows)?;
        let tr = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let m = model.batch_window_nll(&wins, dev, arm)?;
        let tm = t.elapsed().as_secs_f64();
        let mut d = Vec::with_capacity(nwin);
        for w in 0..nwin {
            let (pr, pm) = ((r[w].0 / r[w].1 as f64).exp(), (m[w].0 / m[w].1 as f64).exp());
            let dd = 100.0 * (pm / pr - 1.0);
            d.push(dd);
            println!("  window {:>2}: ppl rows {pr:.4}, {} {pm:.4}, {dd:+.4} %", w + 1, arm.name());
        }
        let (nr, nm, c): (f64, f64, usize) = (r.iter().map(|x| x.0).sum(), m.iter().map(|x| x.0).sum(), r.iter().map(|x| x.1).sum());
        let (pr, pm) = ((nr / c as f64).exp(), (nm / c as f64).exp());
        let mean = d.iter().sum::<f64>() / nwin as f64;
        let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nwin.max(2) - 1) as f64).sqrt();
        // t quantile 0.975 for the common window counts, else 1.96.
        let tq = match nwin { 8 => 2.365, 12 => 2.201, 16 => 2.131, 24 => 2.069, 32 => 2.040, _ => 1.96 };
        let half = tq * sd / (nwin as f64).sqrt();
        let pooled = 100.0 * (pm / pr - 1.0);
        let an = arm.name();
        println!("batched ppl: WikiText-2 test, {nwin} windows of {ctx}, {c} positions, tokens {fp:016x}; rows {tr:.1} s, {an} {tm:.1} s");
        println!("ppl rows {pr:.4} (decode path, multi-row kernels), {an} {pm:.4}: pooled {pooled:+.4} %, window mean {mean:+.4} % [{:+.4} ; {:+.4}]", mean - half, mean + half);
        println!("T8 band: pooled inside +-0.12 %: {}", if pooled.abs() <= 0.12 { "yes" } else { "NO" });
        Ok(())
    }
}
