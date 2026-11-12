//! The speculative decode loop end to end on the card (`llvq_cuda::spec_loop`
//! around the tree verify pass): for each prompt, N greedy tokens by 4E, by
//! the pass alone (the loop with the null drafter) and by the loop with each
//! drafter asked, every drafter's tokens asserted equal to the pass's own,
//! the pass's agreement with 4E reported, and the loop's time split into the verify launches, the drafter calls and
//! the rest of the host.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> specbench [--prompts <file.jsonl>] [--runs <k>] [--vocab dense|q4] [--readout device|chunked] [--dump <file.jsonl>] <sealed> <arm> <drafters> [N] [drafter_dir] [f16|f32]`
//!
//! * `--prompts`: a JSONL prompt file (`llvq_llm::spec_prompts`, the house
//!   set is `ops/data/spec-prompts-v1.jsonl`), each through the same chat
//!   template, thinking off; absent, the four built-in prompts below;
//! * `--vocab`: the drafter's embedding and head, `dense` (default: the sealed
//!   table dequantized once to a dense copy on the card) or `q4` (the
//!   target's own int4 g64 table through its gather and the batched head,
//!   `llvq_llm::dflash_q4`, no copy);
//! * `--readout`: the drafter's readout, `device` (default: on the card, one
//!   small transfer at the end of the round) or `chunked` (per-chunk maxima,
//!   `DFlashSpec::chunked_readout`);
//! * `--dump`: one JSON line a prompt and decoder (`4e`, `pass`, each
//!   drafter's last run): id, kind, decoder, token ids and decoded text, for
//!   grading offline;
//! * `--runs`: timed runs a prompt; by default 1 discarded and 3 timed for
//!   four prompts or fewer, 1 timed and none discarded above;
//! * `<arm>`: the verify pass, `a12` (taps on), `a12ip` (a12 on the words 4E
//!   holds, read in place: no tree-tiled copy, the weights once on the card,
//!   step M1 of the memory axis) or `a10t384r16` (no taps), with an optional
//!   plan suffix `@q.o.g.d.h` (else the stream plan at its warps). After the
//!   install the card's memory in use is printed (cuMemGetInfo), so two runs
//!   of the same prompts read what the copies cost;
//! * `<drafters>`: comma-separated, `null` (R = 1 decoding) and `oracle:K`
//!   (K = 0 .. 14: K correct tokens of the pass's greedy continuation then a wrong
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
//! The hard gate is self-consistency: every drafter's tokens equal the
//! pass's own greedy (the null drafter, run once a prompt untimed as the
//! reference) over every generated token; a mismatch prints its position,
//! both tokens and the pass's logits for them there, and the run exits
//! non-zero at the end. Agreement with 4E is reported, not a gate: the
//! pass's arithmetic is not 4E's (S0: logits within 3.5e-3 relative RMS),
//! so a near tie can flip. Per prompt, the first divergence (or none), both
//! tokens, the pass's logits and margin, the tokens before it; in aggregate,
//! prompts in full agreement and tokens before the first divergence. Past a
//! divergence the texts differ, and tok/s and acceptance there are measured
//! on the pass's own text. Both 4E and the pass break an exact f16 tie with
//! the same key (`pers_tie_key`, candle's order), row by row.
//!
//! Per prompt and drafter: accepted drafted tokens a
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
    let usage = "usage: LLVQ_CONFIG=<served v9 config> specbench [--prompts <file.jsonl>] [--runs <k>] [--vocab dense|q4] [--readout device|chunked] [--dump <file.jsonl>] <sealed> <a12|a12ip|a10t384r16>[@q.o.g.d.h] <null,oracle:K,dflash:chain,dflash:tree,...> [N <= 1024] [drafter_dir] [f16|f32]";
    // flags first: --prompts <file.jsonl>, --runs <timed runs a prompt>
    let mut a: Vec<String> = Vec::new();
    let (mut prompts_file, mut runs): (Option<String>, Option<usize>) = (None, None);
    // the drafter's vocabulary: dense (the sealed table dequantized to a copy)
    // until q4 (the target's own int4 table) passes on the card
    let mut vocab_q4 = false;
    let mut dump_file: Option<String> = None;
    // the drafter's readout: on the device (default) or the chunked one
    let mut chunked = false;
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--prompts" => prompts_file = Some(it.next().ok_or_else(|| anyhow::anyhow!("--prompts <file.jsonl>; {usage}"))?),
            "--runs" => runs = Some(it.next().ok_or_else(|| anyhow::anyhow!("--runs <k>; {usage}"))?.parse::<usize>().map_err(|e| anyhow::anyhow!("--runs: {e}"))?),
            "--dump" => dump_file = Some(it.next().ok_or_else(|| anyhow::anyhow!("--dump <file.jsonl>; {usage}"))?),
            "--readout" => {
                chunked = match it.next().as_deref() {
                    Some("device") => false,
                    Some("chunked") => true,
                    o => anyhow::bail!("--readout {o:?}: device or chunked; {usage}"),
                }
            }
            "--vocab" => {
                vocab_q4 = match it.next().as_deref() {
                    Some("dense") => false,
                    Some("q4") => true,
                    o => anyhow::bail!("--vocab {o:?}: dense or q4; {usage}"),
                }
            }
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
        "a12ip" => TreeForm::A12ip,
        "a10t384r16" => TreeForm::A10t384r16,
        other => anyhow::bail!("specbench: arm {other:?} not served here; {usage}"),
    };
    let taps = matches!(form, TreeForm::A12 | TreeForm::A12ip);
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
    {
        let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
        println!(
            "card memory: {:.3} GB used of {:.3} GB after the target's load, its 4E install and the verify pass's install, before the drafter (cuMemGetInfo)",
            (total - free) as f64 / 1e9,
            total as f64 / 1e9
        );
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
        let (vocab, vline): (Box<dyn llvq_llm::dflash::TargetVocab>, String) = if vocab_q4 {
            let cfg_m = model.config();
            let v = llvq_llm::dflash_q4::Q4Vocab::new(model, cfg_m.num_attention_heads / cfg_m.num_key_value_heads)?;
            for line in v.provenance() {
                println!("{line}");
            }
            let line = format!(
                "vocab q4: the target's int4 g64 table ({:.1} MB, already resident), gather emb_q4_gather, head tv_q4_mma; no dense copy",
                v.table_bytes as f64 / 1e6
            );
            (Box::new(v), line)
        } else {
            let v = llvq_llm::dflash::vocab_from_sealed(std::path::Path::new(&path), &device, ddtype)?;
            let (rows, cols) = v.table.dims2()?;
            let line = format!(
                "vocab dense: table {rows}x{cols} {ddtype:?} from the sealed file, {:.1} MB of VRAM beside the target's",
                (rows * cols * ddtype.size_in_bytes()) as f64 / 1e6
            );
            (Box::new(v), line)
        };
        println!(
            "drafter {} ({:.1} MB, {ddtype:?}), loaded in {:.1} s; {vline}",
            ddir.display(),
            drafter.weight_bytes() as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        );
        let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
        println!(
            "card memory: {:.3} GB used of {:.3} GB after the drafter's load, before the first prompt (cuMemGetInfo)",
            (total - free) as f64 / 1e9,
            total as f64 / 1e9
        );
        let mut sp =llvq_llm::dflash_spec::DFlashSpec::new(drafter, vocab, llvq_llm::dflash_spec::Shape::Chain(15))?;
        sp.timing = true;
        sp.chunked_readout = chunked;
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
    // the pass's greedy against 4E's, a prompt: (kind, tokens before the first
    // divergence or all, pass tokens, divergence margin if any)
    struct Agree {
        kind: String,
        before: usize,
        generated: usize,
        margin: Option<String>,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut agree: Vec<Agree> = Vec::new();
    let mut mismatches = 0usize;
    let mut control_done = false;
    let mut dump_w = match &dump_file {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| anyhow::anyhow!("{p}: {e}"))?)),
        None => None,
    };
    let mut dump = |pid: &str, kind: &str, decoder: &str, toks: &[u32]| -> anyhow::Result<()> {
        if let Some(w) = dump_w.as_mut() {
            use std::io::Write;
            let text = tok.decode(toks, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            let line = serde_json::json!({ "id": pid, "kind": kind, "decoder": decoder, "tokens": toks, "text": text });
            writeln!(w, "{line}")?;
        }
        Ok(())
    };
    println!(
        "equality: every drafter against the pass's own greedy (the null drafter, R = 1), a hard gate; \
         the pass's greedy against 4E's greedy, reported only. Past a divergence from 4E the texts differ, \
         and tok/s and acceptance there are measured on the pass's own text"
    );
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
        dump(pid.as_str(), kind.as_str(), "4e", &want)?;
        // the pass's own greedy: the loop with the null drafter, untimed, the
        // reference every drafter must equal
        set(&mut caches, l)?;
        let pass = {
            let mut v = CardVerifier { rt: &mut *rt, caches: &mut caches, arm: 0, taps };
            let mut null = NullDrafter;
            spec_run(&mut v, &mut null, ids, first, &prompt_taps, hidden, n_tok, Some(eos)).map_err(anyhow::Error::msg)?.tokens
        };
        dump(pid.as_str(), kind.as_str(), "pass", &pass)?;
        // agreement with 4E, reported: the caches hold the pass's text, equal
        // to 4E's before the first divergence
        let div = pass.iter().zip(&want).position(|(a, b)| a != b).or(if pass.len() != want.len() { Some(pass.len().min(want.len())) } else { None });
        match div {
            None => {
                println!("  4E agreement: the pass's greedy equals 4E's over all {} tokens", pass.len());
                agree.push(Agree { kind: kind.clone(), before: pass.len(), generated: pass.len(), margin: None });
            }
            Some(p) => {
                let (a, b) = (pass.get(p).copied(), want.get(p).copied());
                let m = margin_at(&mut *rt, &mut caches, ids, first, &pass, p, a, b)?;
                println!("  4E agreement: first divergence at generated token {p} ({p} tokens equal before it): the pass {a:?}, 4E {b:?}; {m}");
                agree.push(Agree { kind: kind.clone(), before: p, generated: pass.len(), margin: Some(m) });
            }
        }
        for (di, (dname, kind_d)) in drafters.iter().enumerate() {
            let kind_d = *kind_d;
            let (mut wall, mut verify, mut draft, mut accepted, mut ntok, mut equal) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0usize, true);
            let (mut d_ext, mut d_block, mut d_head, mut d_read) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut skipped, mut last_error) = (0usize, None::<String>);
            let mut last = Vec::new();
            for run in 0..discard + timed {
                set(&mut caches, l)?;
                let mut null = NullDrafter;
                let mut oracle;
                let d: &mut dyn Drafter = match kind_d {
                    Kind::Null => &mut null,
                    Kind::Oracle(k) => {
                        oracle = OracleDrafter { prompt_len: l, truth: std::iter::once(first).chain(pass.iter().copied()).collect(), correct: k, vocab: model.config().vocab_size as u32 };
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
                if o.tokens != pass && equal {
                    equal = false;
                    mismatches += 1;
                    let p = o.tokens.iter().zip(&pass).position(|(a, b)| a != b).unwrap_or(o.tokens.len().min(pass.len()));
                    let (got, exp) = (o.tokens.get(p).copied(), pass.get(p).copied());
                    let m = margin_at(&mut *rt, &mut caches, ids, first, &o.tokens, p, got, exp)?;
                    println!("  {dname:>12}: MISMATCH with the pass's own greedy at generated token {p}: the loop {got:?}, the null drafter {exp:?}; {m}");
                }
                if run >= discard {
                    wall.push(o.t_total);
                    verify.push(o.t_verify.iter().sum::<f64>());
                    draft.push(o.t_draft.iter().sum::<f64>());
                    accepted.extend(o.accepted.iter().copied());
                    ntok = o.tokens.len();
                }
                last = o.tokens;
            }
            dump(pid.as_str(), kind.as_str(), dname.as_str(), &last)?;
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
            let (w, tv, td) = (mean(&wall), mean(&verify), mean(&draft));
            let passes = accepted.len() as f64 / timed as f64;
            let acc_mean = accepted.iter().sum::<usize>() as f64 / accepted.len().max(1) as f64;
            println!(
                "  {dname:>12}: {} tokens{}; {acc_mean:.2} drafted tokens accepted a pass, {passes:.0} passes; verify {:.1} ms, drafter {:.1} ms, rest of the host {:.1} ms; {:.1} tok/s against 4E's {:.1} (x{:.3})",
                ntok,
                if equal { ", equal to the pass's greedy" } else { ", NOT equal to the pass's greedy" },
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
                let c = c_of(&mut *rt, &mut caches, l, first, &pass, r)?;
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
    if let Some(mut w) = dump_w {
        use std::io::Write;
        w.flush()?;
        println!("\ndump: {}", dump_file.as_deref().unwrap_or(""));
    }
    // agreement with 4E: reported, not a gate
    println!("\nagreement of the pass's greedy with 4E's greedy (reported, not a gate)");
    let mut groups_a: Vec<&str> = llvq_llm::spec_prompts::KINDS.iter().copied().filter(|k| agree.iter().any(|a| a.kind == *k)).collect();
    groups_a.push("all");
    for g in groups_a {
        let sel: Vec<&Agree> = agree.iter().filter(|a| g == "all" || a.kind == g).collect();
        let full = sel.iter().filter(|a| a.margin.is_none()).count();
        let before: usize = sel.iter().map(|a| a.before).sum();
        let generated: usize = sel.iter().map(|a| a.generated).sum();
        println!("  {g:>5}: {full} of {} prompts in full agreement; {before} of {generated} tokens before the first divergence", sel.len());
    }
    for (a, (pid, _, _)) in agree.iter().zip(&prompts) {
        if let Some(m) = &a.margin {
            println!("  {pid}: at {}: {m}", a.before);
        }
    }
    // aggregates, per drafter: per kind, then overall
    println!("\naggregates (tok/s = total tokens over total wall; x greedy likewise, against 4E's greedy on its own text; median over prompts with its range)");
    for (di, (dname, kind_d)) in drafters.iter().enumerate() {
        let (rev, dtype_s, shape_s) = match kind_d {
            Kind::DFlash(s) => (llvq_llm::dflash::REVISION.to_string(), format!("{ddtype:?}, vocab {}, readout {}", if vocab_q4 { "q4" } else { "dense" }, if chunked { "chunked" } else { "device" }), format!("{s:?}")),
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
                "  {dname:>12} {g:>5}: {} prompts, {tokens} tokens, {} of {} equal to the pass's greedy; {:.2} accepted a pass, histogram 0..15 {hist:?}; {:.1} tok/s, greedy {:.1}, x{:.3}; per prompt x{rm:.3} [{rlo:.3}-{rhi:.3}]; verify {:.0} %, drafter {:.0} % of the wall",
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
    anyhow::ensure!(mismatches == 0, "specbench: {mismatches} prompt and drafter pairs whose tokens differ from the pass's own greedy (the null drafter)");
    Ok(())
}

/// The pass's logits at generated token `p` of `toks` for the tokens `got`
/// and `exp`, and their margin. The caches hold the run that made `toks`
/// (positions before `p` are `toks`' own); the row of position `l + p` is
/// recomputed by one tree launch of chained rows from the caches' length, at
/// most 16 rows.
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn margin_at(
    rt: &mut llvq_llm::persist4e_cuda::Runtime4e,
    caches: &mut [llvq_llm::model::KvCache],
    ids: &[u32],
    first: u32,
    toks: &[u32],
    p: usize,
    got: Option<u32>,
    exp: Option<u32>,
) -> anyhow::Result<String> {
    let l = ids.len();
    let seq: Vec<u32> = ids.iter().copied().chain(std::iter::once(first)).chain(toks.iter().copied()).collect();
    let q = l + p;
    let c = caches[0].len();
    let (Some(g), Some(e)) = (got, exp) else {
        return Ok("margin not computed: one sequence ended first".to_string());
    };
    if q >= seq.len() || (c < q && q - c >= 16) {
        return Ok(format!("margin not computed: position {q} is past the rows a pass carries from the cache's {c}"));
    }
    let start = c.min(q);
    for x in caches.iter_mut() {
        x.set_len(start)?;
    }
    let rows: Vec<u32> = seq[start..=q].to_vec();
    let r = rows.len();
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
    let out = rt.tree(0, caches, start, &rows, &depth, &path, true, false)?;
    let vv = out.logits.len() / r;
    let row = &out.logits[(r - 1) * vv..r * vv];
    let lg = |t: u32| half::f16::from_bits(row[t as usize]).to_f32();
    Ok(format!("the pass's logits: {} for {g}, {} for {e}, margin {:.4}", lg(g), lg(e), lg(g) - lg(e)))
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
