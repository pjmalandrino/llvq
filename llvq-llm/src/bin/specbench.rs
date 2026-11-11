//! The speculative decode loop end to end on the card (`llvq_cuda::spec_loop`
//! around the tree verify pass): for each prompt, N greedy tokens by 4E and
//! by the loop with each drafter asked, token equality asserted over all N,
//! and the loop's time split into the verify launches, the drafter calls and
//! the rest of the host.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> specbench [--prompts <file.jsonl>] [--runs <k>] <sealed> <arm> <drafters> [N] [drafter_dir] [f16|f32]`
//!
//! * `--prompts`: a JSONL prompt file (`llvq_llm::spec_prompts`, the house
//!   set is `ops/data/spec-prompts-v1.jsonl`), each through the same chat
//!   template, thinking off; absent, the four built-in prompts below;
//! * `--runs`: timed runs a prompt; by default 1 discarded and 3 timed for
//!   four prompts or fewer, 1 timed and none discarded above;
//! * `<arm>`: the verify pass, `a12` (taps on) or `a10t384r16` (no taps), with
//!   an optional plan suffix `@q.o.g.d.h` (else the stream plan at its warps);
//! * `<drafters>`: comma-separated, `null` (R = 1 decoding) and `oracle:K`
//!   (K = 0 .. 14: K correct tokens of 4E's greedy continuation then a wrong
//!   one, a pass of K + 2 rows accepting K);
//!   and `dflash:chain` / `dflash:tree` (z-lab/Qwen3-4B-DFlash-b16 at
//!   b74e3a32, `llvq_llm::dflash_spec`: the argmax chain of 15, or the tree
//!   of 15 nodes; it reads the arm's taps, so an arm without them leaves it
//!   skipping every round);
//! * `N`: tokens a prompt at most, 1 to 1,024, default 64; generation stops
//!   at `<|im_end|>`, and the counts are of the tokens actually generated;
//! * `drafter_dir`: the drafter's snapshot (default: the local Hugging Face
//!   cache); `f16` (default) or `f32`: the drafter's dtype on the card. Its
//!   head and embedding are the sealed file's own int4 table dequantized once
//!   to a dense table in that dtype (`dflash::vocab_from_sealed`).
//!
//! The reference is 4E's greedy decode in chained launches of at most its
//! `max_tokens` (`Runtime4e::decode_chunks`); a control line on the first
//! prompt compares one 64-token launch with two chained 32-token ones.
//!
//! Per prompt and drafter: equality with 4E over every generated token (a
//! mismatch prints its position, both tokens and the pass's logits for them
//! there, and the run exits non-zero at the end), accepted drafted tokens a
//! pass, passes, the walls (verify launches, drafter calls, the rest of the
//! host), tok/s end to end against 4E's greedy tok/s. Per drafter, a config
//! line, then aggregates per kind and overall: tokens, equal prompts, accepted
//! a pass (mean and histogram 0..15), tok/s as total tokens over total wall,
//! the ratio to greedy likewise, and the median ratio over prompts with its
//! range. For the oracle at K with four prompts or fewer, the check: greedy x
//! (K + 1) / c_(K + 2), c_R a tree launch of R chained rows against one 4E
//! token, timed here.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        run()
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        anyhow::bail!("specbench: the verify pass is a cooperative CUDA launch; this build has no CUDA backend")
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const ROUNDS: usize = 3;

/// The loop's verifier on the card: one tree launch of an installed arm.
#[cfg(all(target_os = "linux", feature = "cuda"))]
struct CardVerifier<'a> {
    rt: &'a mut llvq_llm::persist4e_cuda::Runtime4e,
    caches: &'a mut [llvq_llm::model::KvCache],
    arm: usize,
    taps: bool,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
impl llvq_cuda::spec_loop::Verifier for CardVerifier<'_> {
    fn verify(&mut self, l: &llvq_cuda::spec_loop::VerifyLaunch) -> Result<llvq_cuda::spec_loop::VerifyOut, String> {
        let o = self
            .rt
            .tree_moved(self.arm, self.caches, l.pos0, l.toks, l.depth, l.path, l.mv, l.move_pos0, false, false, self.taps)
            .map_err(|e| e.to_string())?;
        Ok(llvq_cuda::spec_loop::VerifyOut { ring: o.ring, taps: o.taps })
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run() -> anyhow::Result<()> {
    use llvq_cuda::spec_loop::{run as spec_run, Drafter, NullDrafter, OracleDrafter};
    use llvq_llm::boundary::median_range;
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    use llvq_llm::persist4e_cuda::{TreeForm, TreePlan};
    let usage = "usage: LLVQ_CONFIG=<served v9 config> specbench [--prompts <file.jsonl>] [--runs <k>] <sealed> <a12|a10t384r16>[@q.o.g.d.h] <null,oracle:K,dflash:chain,dflash:tree,...> [N <= 1024] [drafter_dir] [f16|f32]";
    // flags first: --prompts <file.jsonl>, --runs <timed runs a prompt>
    let mut a: Vec<String> = Vec::new();
    let (mut prompts_file, mut runs): (Option<String>, Option<usize>) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--prompts" => prompts_file = Some(it.next().ok_or_else(|| anyhow::anyhow!("--prompts <file.jsonl>; {usage}"))?),
            "--runs" => runs = Some(it.next().ok_or_else(|| anyhow::anyhow!("--runs <k>; {usage}"))?.parse().map_err(|e| anyhow::anyhow!("--runs: {e}"))?),
            _ => a.push(x),
        }
    }
    let path = a.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let arm_s = a.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let drafters_s = a.get(2).ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let n_tok: usize = a.get(3).map_or(Ok(64), |s| s.parse()).map_err(|e| anyhow::anyhow!("N: {e}; {usage}"))?;
    anyhow::ensure!((1..=1024).contains(&n_tok), "N {n_tok}: 1 to 1,024 tokens a prompt; {usage}");
    let (base, plan) = match arm_s.split_once('@') {
        None => (arm_s.as_str(), TreePlan::Stream),
        Some((b, p)) => {
            let v: Vec<u32> = p.split('.').map(|x| x.parse::<u32>()).collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("plan {p}: {e}; {usage}"))?;
            let arr: [u32; 5] = v.try_into().map_err(|_| anyhow::anyhow!("plan {p}: five slice counts; {usage}"))?;
            (b, TreePlan::Explicit(arr))
        }
    };
    let form = match base {
        "a12" => TreeForm::A12,
        "a10t384r16" => TreeForm::A10t384r16,
        other => anyhow::bail!("specbench: arm {other:?} not served here; {usage}"),
    };
    let taps = form == TreeForm::A12;
    // drafters
    #[derive(Clone, Copy)]
    enum Kind {
        Null,
        Oracle(usize),
        DFlash(llvq_llm::dflash_spec::Shape),
    }
    let mut drafters: Vec<(String, Kind)> = Vec::new();
    for d in drafters_s.split(',') {
        match d.split_once(':') {
            None if d == "null" => drafters.push(("null".into(), Kind::Null)),
            Some(("oracle", k)) => {
                let k: usize = k.parse().map_err(|e| anyhow::anyhow!("oracle:{k}: {e}; {usage}"))?;
                anyhow::ensure!(k + 2 <= 16, "oracle:{k}: a pass of {} rows, the arm carries 16", k + 2);
                drafters.push((d.to_string(), Kind::Oracle(k)));
            }
            Some(("dflash", "chain")) => drafters.push((d.to_string(), Kind::DFlash(llvq_llm::dflash_spec::Shape::Chain(15)))),
            Some(("dflash", "tree")) => drafters.push((d.to_string(), Kind::DFlash(llvq_llm::dflash_spec::Shape::Tree(15)))),
            _ => anyhow::bail!("specbench: unknown drafter {d:?}; {usage}"),
        }
    }
    let want_dflash = drafters.iter().any(|(_, k)| matches!(k, Kind::DFlash(_)));
    let ddtype = match a.get(5).map(String::as_str).unwrap_or("f16") {
        "f16" => candle_core::DType::F16,
        "f32" => candle_core::DType::F32,
        o => anyhow::bail!("specbench: drafter dtype {o}: f16 or f32; {usage}"),
    };
    let cfg = llvq_llm::served::Served::from_env()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("LLVQ_CONFIG names the served config; {usage}"))?;
    anyhow::ensure!(
        cfg.decode == llvq_llm::served::Decode::Persistent && cfg.rows.form() == 7,
        "{}: the config must name the persistent decode with rows v9",
        cfg.path.display()
    );
    println!("target: {}", cfg.provenance());
    let device = candle_core::Device::new_cuda(0)?;
    let t_load = Instant::now();
    let mut f = llvq_llm::fused_cuda::load_resolved(
        &path,
        &device,
        candle_core::DType::F16,
        cfg.layout,
        cfg.embed,
        cfg.rot_share,
        cfg.fuse,
        cfg.kv,
        cfg.int4_layout(),
        Some("specbench"),
    )?;
    cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut dec = cfg.decoder(&f.model, &device)?;
    for line in dec.lines() {
        println!("{line}");
    }
    let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("specbench: the config built no 4E runtime"))?;
    let before = rt.lines.len();
    rt.install_tree(&[(arm_s.as_str(), plan, form)])?;
    for line in &rt.lines[before..] {
        println!("{line}");
    }
    let arm0 = rt.tree_arms().into_iter().next().ok_or_else(|| anyhow::anyhow!("specbench: no arm installed"))?;
    anyhow::ensure!(arm0.void.is_none(), "specbench: arm {} is void: {:?}", arm0.name, arm0.void);
    println!("arm {}: {:?} kernel, slices {:?}, {} threads, row cap {}; taps {}; N {n_tok}", arm0.name, arm0.form, arm0.splits, arm0.threads, arm0.form.rmax(), if taps { "on" } else { "off" });
    println!("loaded in {:.1} s", t_load.elapsed().as_secs_f64());
    let tok = f.tokenizer.clone();
    let marks = Marks::resolve(&tok)?;
    let model = &f.model;
    let hidden = model.config().hidden_size;
    // the drafter, once: its weights from the snapshot, its table the sealed file's own
    let mut dspec = if want_dflash {
        let ddir = match a.get(4) {
            Some(d) => std::path::PathBuf::from(d),
            None => llvq_llm::dflash::cached_snapshot()
                .ok_or_else(|| anyhow::anyhow!("specbench: {}@{} is not in the local cache; give its directory", llvq_llm::dflash::REPO, llvq_llm::dflash::REVISION))?,
        };
        let t0 = Instant::now();
        let drafter = llvq_llm::dflash::DFlashDrafter::load(&ddir, &device, ddtype)?;
        let vocab = llvq_llm::dflash::vocab_from_sealed(std::path::Path::new(&path), &device, ddtype)?;
        println!(
            "drafter {} ({:.1} MB, {ddtype:?}), table {:?} from the sealed file, loaded in {:.1} s",
            ddir.display(),
            drafter.weight_bytes() as f64 / 1e6,
            vocab.table.dims(),
            t0.elapsed().as_secs_f64()
        );
        let mut sp = llvq_llm::dflash_spec::DFlashSpec::new(drafter, Box::new(vocab), llvq_llm::dflash_spec::Shape::Chain(15))?;
        sp.timing = true;
        Some(sp)
    } else {
        None
    };
    let tap_layers = [2usize, 10, 18, 26, 34];
    let set = |c: &mut [llvq_llm::model::KvCache], n: usize| -> anyhow::Result<()> {
        for x in c.iter_mut() {
            x.set_len(n)?;
        }
        Ok(())
    };
    let eos = tok.token_to_id("<|im_end|>").ok_or_else(|| anyhow::anyhow!("specbench: the tokenizer has no <|im_end|>"))?;
    // the thinking switch as this template encodes it, once
    let opening = open_assistant(&tok, &marks, false)?;
    println!(
        "template: the assistant opening with thinking off is {opening:?}; the HF tokenizer on the rendered Qwen3 template (the served door's) \
         gives <think>, 271, </think>, 271 (151667 271 151668 271), not 198 198; eos {eos} (<|im_end|>)"
    );
    // the prompts: (id, kind, ids)
    let prompts: Vec<(String, String, Vec<u32>)> = match &prompts_file {
        Some(file) => {
            let text = std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
            let recs = llvq_llm::spec_prompts::parse(&text).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
            let mut v = Vec::with_capacity(recs.len());
            for r in recs {
                let mut ids = Vec::new();
                for (role, content) in &r.messages {
                    ids.extend(turn(&tok, &marks, role, content)?);
                }
                ids.extend(open_assistant(&tok, &marks, false)?);
                v.push((r.id, r.kind, ids));
            }
            v
        }
        None => {
            let chat = |body: &str| -> anyhow::Result<Vec<u32>> {
                let mut ids = turn(&tok, &marks, "user", body)?;
                ids.extend(open_assistant(&tok, &marks, false)?);
                Ok(ids)
            };
            vec![
                ("chat".into(), "chat".into(), chat(CHAT)?),
                ("rag".into(), "rag".into(), chat(&format!("{RAG}\n\n{SUMMARY_TEXT}"))?),
                ("code".into(), "code".into(), chat(CODE)?),
                ("math".into(), "math".into(), chat(MATH)?),
            ]
        }
    };
    let many = prompts.len() > 4;
    let (discard, timed) = match runs {
        Some(k) => (usize::from(!many), k.max(1)),
        None if many => (0, 1),
        None => (1, ROUNDS),
    };
    println!("prompts: {} from {}; {discard} discarded and {timed} timed runs a prompt", prompts.len(), prompts_file.as_deref().unwrap_or("the built-in four"));
    // one row a prompt and drafter
    struct Row {
        kind: String,
        drafter: usize,
        tokens: usize,
        equal: bool,
        accepted: Vec<usize>,
        wall: f64,
        verify: f64,
        draft: f64,
        greedy_tokens: usize,
        greedy_wall: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut mismatches = 0usize;
    let mut control_done = false;
    for (pid, kind, ids) in &prompts {
        let l = ids.len();
        let mut caches = model.fresh_caches();
        // the prompt's taps, from the eager prefill (to rounding of the pass's own)
        let (first, prompt_taps) = if taps {
            model.prefill_taps(ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut caches, &tap_layers)?
        } else {
            (model.prefill_next_in(ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut caches)?, Vec::new())
        };
        model.device().synchronize()?;
        // control, once: one 64-token 4E launch against two chained 32-token ones
        if !control_done {
            set(&mut caches, l)?;
            let one = rt.decode_chunks(&mut caches, l, first, 64, u32::MAX, 64)?;
            set(&mut caches, l)?;
            let two = rt.decode_chunks(&mut caches, l, first, 64, u32::MAX, 32)?;
            println!(
                "control: one 64-token 4E launch against two chained 32-token launches on {pid}: {}",
                if one == two { "the same 64 tokens".to_string() } else { format!("DIFFER at {:?}", one.iter().zip(&two).position(|(a, b)| a != b)) }
            );
            control_done = true;
        }
        // 4E greedy, chained launches, stopped at eos
        let mut want = Vec::new();
        let mut t4e = Vec::new();
        for run in 0..discard + timed {
            set(&mut caches, l)?;
            rt.sync()?;
            let t0 = Instant::now();
            want = rt.decode_chunks(&mut caches, l, first, n_tok, eos, 0)?;
            rt.sync()?;
            if run >= discard {
                t4e.push(t0.elapsed().as_secs_f64());
            }
        }
        let greedy_wall = t4e.iter().sum::<f64>() / t4e.len() as f64;
        println!("\n== {pid} ({kind}): prompt {l} tokens; 4E greedy {} tokens, {:.1} tok/s", want.len(), want.len() as f64 / greedy_wall);
        for (di, (dname, kind_d)) in drafters.iter().enumerate() {
            let kind_d = *kind_d;
            let (mut wall, mut verify, mut draft, mut accepted, mut ntok, mut equal) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0usize, true);
            let (mut d_ext, mut d_block, mut d_head, mut d_read) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut skipped, mut last_error) = (0usize, None::<String>);
            for run in 0..discard + timed {
                set(&mut caches, l)?;
                let mut null = NullDrafter;
                let mut oracle;
                let d: &mut dyn Drafter = match kind_d {
                    Kind::Null => &mut null,
                    Kind::Oracle(k) => {
                        oracle = OracleDrafter { prompt_len: l, truth: std::iter::once(first).chain(want.iter().copied()).collect(), correct: k, vocab: model.config().vocab_size as u32 };
                        &mut oracle
                    }
                    Kind::DFlash(shape) => {
                        let sp = dspec.as_mut().ok_or_else(|| anyhow::anyhow!("specbench: no drafter loaded"))?;
                        sp.shape = shape;
                        sp.reset()?;
                        sp
                    }
                };
                rt.sync()?;
                let o = {
                    let mut v = CardVerifier { rt: &mut *rt, caches: &mut caches, arm: 0, taps };
                    spec_run(&mut v, d, ids, first, &prompt_taps, hidden, n_tok, Some(eos)).map_err(anyhow::Error::msg)?
                };
                if let (Kind::DFlash(_), Some(sp)) = (kind_d, dspec.as_ref()) {
                    if run >= discard {
                        d_ext.push(sp.t_extend.iter().sum::<f64>() * 1e3);
                        d_block.push(sp.t_block.iter().sum::<f64>() * 1e3);
                        d_head.push(sp.t_head.iter().sum::<f64>() * 1e3);
                        d_read.push(sp.t_readout.iter().sum::<f64>() * 1e3);
                    }
                    skipped = skipped.max(sp.skipped);
                    if sp.last_error.is_some() {
                        last_error = sp.last_error.clone();
                    }
                }
                if o.tokens != want && equal {
                    equal = false;
                    mismatches += 1;
                    let p = o.tokens.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(o.tokens.len().min(want.len()));
                    let (got, exp) = (o.tokens.get(p).copied(), want.get(p).copied());
                    // the pass's logit margin there: the row of position q = l + p over the loop's cache
                    let seq: Vec<u32> = ids.iter().copied().chain(std::iter::once(first)).chain(o.tokens.iter().copied()).collect();
                    let q = l + p;
                    let c = caches[0].len();
                    let margin = match (got, exp) {
                        (Some(g), Some(e)) if q < seq.len() && (c >= q || q - c < 16) => {
                            let start = c.min(q);
                            set(&mut caches, start)?;
                            let toks: Vec<u32> = seq[start..=q].to_vec();
                            let r = toks.len();
                            let depth: Vec<u32> = (0..r as u32).collect();
                            let path: Vec<[u8; 32]> = (0..r)
                                .map(|i| {
                                    let mut pp = [0u8; 32];
                                    for (dd, x) in pp.iter_mut().enumerate().take(i + 1) {
                                        *x = dd as u8;
                                    }
                                    pp
                                })
                                .collect();
                            let out = rt.tree(0, &mut caches, start, &toks, &depth, &path, true, false)?;
                            let vv = out.logits.len() / r;
                            let row = &out.logits[(r - 1) * vv..r * vv];
                            let lg = |t: u32| half::f16::from_bits(row[t as usize]).to_f32();
                            format!("logit {} for the loop's {g}, {} for 4E's {e}: margin {:.4}", lg(g), lg(e), lg(g) - lg(e))
                        }
                        _ => "margin not computed (a sequence ended first, or the position is past the rows a pass carries)".to_string(),
                    };
                    println!("  {dname:>12}: MISMATCH at generated token {p}: the loop {got:?}, 4E {exp:?}; {margin}");
                }
                if run >= discard {
                    wall.push(o.t_total);
                    verify.push(o.t_verify.iter().sum::<f64>());
                    draft.push(o.t_draft.iter().sum::<f64>());
                    accepted.extend(o.accepted.iter().copied());
                    ntok = o.tokens.len();
                }
            }
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
            let (w, tv, td) = (mean(&wall), mean(&verify), mean(&draft));
            let passes = accepted.len() as f64 / timed as f64;
            let acc_mean = accepted.iter().sum::<usize>() as f64 / accepted.len().max(1) as f64;
            println!(
                "  {dname:>12}: {} tokens{}; {acc_mean:.2} drafted tokens accepted a pass, {passes:.0} passes; verify {:.1} ms, drafter {:.1} ms, rest of the host {:.1} ms; {:.1} tok/s against 4E's {:.1} (x{:.3})",
                ntok,
                if equal { ", equal to 4E's" } else { ", NOT equal to 4E's" },
                tv * 1e3,
                td * 1e3,
                (w - tv - td) * 1e3,
                ntok as f64 / w,
                want.len() as f64 / greedy_wall,
                (ntok as f64 / w) / (want.len() as f64 / greedy_wall)
            );
            if !d_ext.is_empty() {
                println!(
                    "  {dname:>12}: drafter wall, ms: context extension {:.1}, block {:.1}, head {:.1}, readout {:.1}; skipped rounds {skipped}{}",
                    mean(&d_ext),
                    mean(&d_block),
                    mean(&d_head),
                    mean(&d_read),
                    last_error.as_ref().map_or(String::new(), |e| format!("; last_error: {e}"))
                );
            }
            if let (Kind::Oracle(k), false) = (kind_d, many) {
                let r = k + 2;
                let c = c_of(&mut *rt, &mut caches, l, first, &want, r)?;
                let g = want.len() as f64 / greedy_wall;
                println!("  {dname:>12}: check, greedy x (K + 1) / c_{r} = {g:.1} x {} / {c:.3} = {:.1} tok/s, measured {:.1}", k + 1, g * (k as f64 + 1.0) / c, ntok as f64 / w);
            }
            rows.push(Row {
                kind: kind.clone(),
                drafter: di,
                tokens: ntok,
                equal,
                accepted,
                wall: w,
                verify: tv,
                draft: td,
                greedy_tokens: want.len(),
                greedy_wall,
            });
        }
    }
    // aggregates, per drafter: per kind, then overall
    println!("\naggregates (tok/s = total tokens over total wall; x greedy likewise; median over prompts with its range)");
    for (di, (dname, kind_d)) in drafters.iter().enumerate() {
        let (rev, dtype_s, shape_s) = match kind_d {
            Kind::DFlash(s) => (llvq_llm::dflash::REVISION.to_string(), format!("{ddtype:?}"), format!("{s:?}")),
            Kind::Oracle(k) => ("-".into(), "-".into(), format!("oracle chain of {} ({k} correct)", k + 1)),
            Kind::Null => ("-".into(), "-".into(), "none (R = 1)".into()),
        };
        println!(
            "config {dname}: arm {} ({:?}), plan {:?}, {} threads, row cap {}, drafter revision {rev}, dtype {dtype_s}, shape {shape_s}, N {n_tok}",
            arm0.name,
            arm0.form,
            arm0.splits,
            arm0.threads,
            arm0.form.rmax()
        );
        let mut groups: Vec<&str> = llvq_llm::spec_prompts::KINDS.iter().copied().filter(|k| rows.iter().any(|r| r.drafter == di && r.kind == *k)).collect();
        groups.push("all");
        for g in groups {
            let sel: Vec<&Row> = rows.iter().filter(|r| r.drafter == di && (g == "all" || r.kind == g)).collect();
            let tokens: usize = sel.iter().map(|r| r.tokens).sum();
            let wall: f64 = sel.iter().map(|r| r.wall).sum();
            let gtok: usize = sel.iter().map(|r| r.greedy_tokens).sum();
            let gwall: f64 = sel.iter().map(|r| r.greedy_wall).sum();
            let acc: Vec<usize> = sel.iter().flat_map(|r| r.accepted.iter().copied()).collect();
            let hist = llvq_llm::spec_prompts::histogram(&acc);
            let ratios: Vec<f64> = sel.iter().map(|r| (r.tokens as f64 / r.wall) / (r.greedy_tokens as f64 / r.greedy_wall)).collect();
            let (rm, rlo, rhi) = median_range(&ratios);
            println!(
                "  {dname:>12} {g:>5}: {} prompts, {tokens} tokens, {} of {} equal; {:.2} accepted a pass, histogram 0..15 {hist:?}; {:.1} tok/s, greedy {:.1}, x{:.3}; per prompt x{rm:.3} [{rlo:.3}-{rhi:.3}]; verify {:.0} %, drafter {:.0} % of the wall",
                sel.len(),
                sel.iter().filter(|r| r.equal).count(),
                sel.len(),
                acc.iter().sum::<usize>() as f64 / acc.len().max(1) as f64,
                tokens as f64 / wall,
                gtok as f64 / gwall,
                (tokens as f64 / wall) / (gtok as f64 / gwall),
                100.0 * sel.iter().map(|r| r.verify).sum::<f64>() / wall,
                100.0 * sel.iter().map(|r| r.draft).sum::<f64>() / wall
            );
        }
    }
    anyhow::ensure!(mismatches == 0, "specbench: {mismatches} prompt and drafter pairs whose tokens differ from 4E's greedy");
    Ok(())
}

/// One 4E token and a tree launch of `r` chained rows, round by round: the
/// median ratio c_r, for the oracle's check.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn c_of(
    rt: &mut llvq_llm::persist4e_cuda::Runtime4e,
    caches: &mut [llvq_llm::model::KvCache],
    l: usize,
    first: u32,
    want: &[u32],
    r: usize,
) -> anyhow::Result<f64> {
    let set = |c: &mut [llvq_llm::model::KvCache], n: usize| -> anyhow::Result<()> {
        for x in c.iter_mut() {
            x.set_len(n)?;
        }
        Ok(())
    };
    let nodes: Vec<u32> = std::iter::once(first).chain(want.iter().copied()).take(r).collect();
    let r = nodes.len();
    let depth: Vec<u32> = (0..r as u32).collect();
    let path: Vec<[u8; 32]> = (0..r)
        .map(|i| {
            let mut p = [0u8; 32];
            for (d, x) in p.iter_mut().enumerate().take(i + 1) {
                *x = d as u8;
            }
            p
        })
        .collect();
    let mut ratios = Vec::new();
    for round in 0..=ROUNDS {
        set(caches, l)?;
        rt.sync()?;
        let t0 = Instant::now();
        rt.decode(caches, l, first, 1, u32::MAX)?;
        rt.sync()?;
        let w4 = t0.elapsed().as_secs_f64();
        set(caches, l)?;
        rt.sync()?;
        let t1 = Instant::now();
        rt.tree(0, caches, l, &nodes, &depth, &path, false, false)?;
        rt.sync()?;
        if round > 0 {
            ratios.push(t1.elapsed().as_secs_f64() / w4);
        }
    }
    Ok(llvq_llm::boundary::median_range(&ratios).0)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const CHAT: &str = "I am planning a three-day trip to Lisbon in late autumn with my partner. We like walking, \
small museums and local food, and we do not want to rent a car. Could you suggest a plan for each day and a few \
practical tips?";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const CODE: &str = "Write a Python function that takes a list of intervals as pairs of integers and returns the \
list with every overlapping interval merged, sorted by start. Explain the complexity and give two test cases.";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const MATH: &str = "A baker makes 12 trays of rolls each morning, with 18 rolls on a tray. She sells 3 rolls for \
2 euros and sells everything except 25 rolls, which she gives away. How much money does she take in? Explain each \
step.";
// specrun's (2026-11-07), as in vtreebench, not quoted from any source.
#[cfg(all(target_os = "linux", feature = "cuda"))]
const SUMMARY_TEXT: &str = "The town council met on Tuesday evening to discuss the future of the old railway \
depot on the east side of the river. The building has stood empty for eleven years, since the freight line was \
closed and the last maintenance crews moved to the regional hub. Three proposals were presented. The first, from a \
housing cooperative, would convert the main hall into forty affordable flats and keep the brick facade and the \
iron roof trusses. The second, from a group of local businesses, would turn the depot into a covered market with \
space for about sixty stalls, a small brewery and a weekend food hall. The third, from the regional transport \
authority, would demolish the building and use the site as a bus interchange with a park-and-ride car park for \
three hundred vehicles. Residents who spoke at the meeting were divided. Several argued that the town has a severe \
shortage of affordable homes and that young families are leaving because rents have risen by almost a third in five \
years. Others said the market would bring visitors and jobs to a part of town that has few shops. A smaller group \
supported the interchange, pointing out that the nearest station is two kilometres away and that traffic in the \
centre is getting worse. The council's engineer reported that the roof needs urgent repairs whichever option is \
chosen, at an estimated cost of 1.2 million, and that the soil near the old fuel tanks must be cleaned before any \
housing can be built. The mayor said no decision would be taken before a public consultation in the spring, and \
asked each group to submit a detailed budget and timetable by the end of February.";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const RAG: &str = "Answer from the text below only. What did the council's engineer report, and what did the mayor \
decide? Quote the sentences of the text that support your answer.";
