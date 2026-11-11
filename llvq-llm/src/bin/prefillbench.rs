//! The first experiment of docs/mesures/large-prefill-contexte-2026-11-11.txt:
//! the prompt prefilled through the tree verify pass read in place (arm
//! a12ip, no weight copy, no drafter) as a chain of R known tokens a pass,
//! then handed over to the served 4E kernel for the decode; against the
//! served eager prefill, in its own process (`--form eager`), so each form's
//! card memory is its own.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> prefillbench --form eager|chain --long <file.jsonl> [--lengths 1024,4096,8191]
//! [--new <n>] [--prompts <file.jsonl>] [--dump <texts.jsonl>] [--pair-texts <texts.jsonl>] [--ppl <ctx>,<windows>]
//! [--ppl-dump <nll.jsonl>] [--pair-ppl <nll.jsonl>] [--arm a12ip|a12ipg|a12ipg32@q.o.g.d.h[/R]]
//! [--against a12ip|a12ipg|a12ipg32@q.o.g.d.h[/R]]... [--dump-against <texts.jsonl>]... [--r <rows>] <sealed>`
//!
//! * an arm's rows a pass: its `/R` suffix, else `--r` (default 16); refused
//!   by name above the arm's row cap (`TreeForm::rmax`: 16 for a12ip and
//!   a12ipg, 32 for a12ipg32);
//! * `--against` (D1 of the study, the chain form only, repeatable): more arms
//!   installed beside `--arm` in the same process, their words the same. The
//!   first `--against` is the reference. The arms are timed round by round
//!   (the reference first in each round, then the other `--against` arms in
//!   order, `--arm` last), each other arm's identity gate against the
//!   reference replaces the R gates (at every length, each arm at its own R,
//!   the last row's logits, its argmax and every layer's K and V of the
//!   prompt, bit for bit), each arm's phase split is printed, the texts are
//!   made by every arm and each paired with the reference's in the process
//!   (`--dump-against`, once for each `--against` in order, writes theirs);
//!   `--pair-texts` and `--ppl` are refused beside it. The marks carry the
//!   arm's name (`time-<l>-<arm>`, one window a round; `phases-<arm>`);
//!
//! * `--long`: one long prompt (`ops/data/spec-long-8k-v1.jsonl`), through the
//!   chat template, thinking off; a shorter length keeps the template and the
//!   first content tokens, so every length is exact;
//! * timing, per length in increasing order: 1 discarded and 3 timed rounds of
//!   the prefill then `--new` tokens (default 256) of 4E greedy with no stop:
//!   the time to the first token, the prefill until 4E can take over (the
//!   chain's hand-over launch included), the decode tok/s after the first
//!   token ((new - 1) / decode wall, as fusedrun's LLVQ_CONTEXTS), tokens
//!   identical across rounds or refused; the card's memory after the rounds;
//! * chain only, the identity gates: at the first length, R = 16 against R = 1
//!   (the pass's one-row path: a pass a token, each moving the previous) on
//!   the last row's logits, its argmax and every layer's K and V of the
//!   prompt, bit for bit; at the second length, R = 8 against R = 16 the same;
//!   then the phase split of the last three passes of the longest prompt
//!   (`pers::p4t_phases_a9`, stamped launches, untimed);
//! * `--prompts`: the house prompts (`ops/data/spec-prompts-v1.jsonl`), each
//!   prefilled by the form then `--new` tokens of 4E greedy; with the long
//!   prompts' first timed round, one line a prompt in `--dump` (the tokens to
//!   the first `<|im_end|>` included). `--pair-texts` reads the other form's
//!   dump: identical texts, and each first divergence with the pass's logit
//!   margin there (the chain form only);
//! * `--ppl <ctx>,<windows>`: WikiText-2 test windows 0.. of `ctx` tokens,
//!   each window's mean NLL: the eager form by `Qwen3::window_nll` (the
//!   published prefill path), the chain form by chain passes with the head
//!   and logits on every row; `--ppl-dump` writes them, `--pair-ppl` pairs
//!   with the other form's (`paired_ppl_pct`, fusedrun's interval);
//! * `mark <tag> begin|end <unix s>` lines bound each phase, for the job's
//!   card-memory sampler.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        run()
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        anyhow::bail!("prefillbench: the tree pass and 4E are cooperative CUDA launches; this build has no CUDA backend")
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const ROUNDS: usize = 3;

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn mark(tag: &str, what: &str) {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    println!("mark {tag} {what} {t:.3}");
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn card_memory(at: &str) -> anyhow::Result<()> {
    let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info().map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
    println!("card memory: {:.3} GB used of {:.3} GB {at} (cuMemGetInfo)", (total - free) as f64 / 1e9, total as f64 / 1e9);
    Ok(())
}

/// Every layer's cache, empty, its Grow buffers made (the tree pass and 4E
/// write in place): one zero position appended, then the length set to 0.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn empty_caches(model: &llvq_llm::model::Qwen3) -> anyhow::Result<Vec<llvq_llm::model::KvCache>> {
    let cfg = model.config();
    let z = candle_core::Tensor::zeros((1, cfg.num_key_value_heads, 1, cfg.head_dim), model.dtype(), model.device())?;
    let mut c = model.fresh_caches();
    for x in c.iter_mut() {
        x.restore(&z, &z)?;
        x.set_len(0)?;
        anyhow::ensure!(x.grow_buffers().is_some(), "prefillbench: the served store is not Grow");
    }
    Ok(c)
}

/// What one chain prefill leaves.
#[cfg(all(target_os = "linux", feature = "cuda"))]
struct Chain {
    first: u32,
    /// The last row's logits when asked, binary16.
    last_logits: Vec<u16>,
    /// Seconds to the first token, and to the hand-over's end.
    ttft: f64,
    prefill: f64,
}

/// The chain prefill of `ids` into `caches` (empty, buffers made), R rows a
/// pass, the head on the last pass only, then the hand-over launch: the
/// caches hold the prompt and 4E can decode `first` at `ids.len()`.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn chain_prefill(
    rt: &mut llvq_llm::persist4e_cuda::Runtime4e,
    arm: usize,
    caches: &mut [llvq_llm::model::KvCache],
    ids: &[u32],
    r: usize,
    last_logits: bool,
    t0: Instant,
) -> anyhow::Result<Chain> {
    use llvq_cuda::spec_loop::{chain_handover, chain_launches, ChainFault};
    let (ls, mv, mp) = chain_launches(ids, 0, r, ChainFault::None).map_err(anyhow::Error::msg)?;
    let n = ls.len();
    let mut out = None;
    for (i, l) in ls.iter().enumerate() {
        let head = i + 1 == n;
        let o = rt.tree_chain(arm, caches, l.pos0, &l.toks, &l.mv, l.move_pos0, head, head && last_logits, false)?;
        if head {
            out = Some(o);
        }
    }
    let o = out.ok_or_else(|| anyhow::anyhow!("prefillbench: no pass"))?;
    let first = *o.ring.last().ok_or_else(|| anyhow::anyhow!("prefillbench: the last pass wrote no token"))?;
    let ttft = t0.elapsed().as_secs_f64();
    let vv = if last_logits { o.logits.len() / o.ring.len() } else { 0 };
    let last = if last_logits { o.logits[o.logits.len() - vv..].to_vec() } else { Vec::new() };
    let h = chain_handover(first, &mv, mp);
    rt.tree_chain(arm, caches, h.pos0, &h.toks, &h.mv, h.move_pos0, false, false, false)?;
    rt.sync()?;
    Ok(Chain { first, last_logits: last, ttft, prefill: t0.elapsed().as_secs_f64() })
}

/// Per layer, an FNV-1a hash of the K then V bits of positions `0..l`.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn kv_hashes(caches: &[llvq_llm::model::KvCache], l: usize) -> anyhow::Result<Vec<u64>> {
    let mut out = Vec::with_capacity(caches.len());
    for c in caches {
        let (k, v) = c.grow_buffers().ok_or_else(|| anyhow::anyhow!("prefillbench: a cache with no Grow buffer"))?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for t in [k, v] {
            let bits: Vec<half::f16> = t.narrow(2, 0, l)?.contiguous()?.flatten_all()?.to_vec1()?;
            for x in bits {
                for b in x.to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        out.push(h);
    }
    Ok(out)
}

/// The paired relative perplexity difference of `b` against `a`, per cent,
/// and its 95 % interval over the windows (fusedrun's `paired_ppl_pct`).
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn paired_ppl_pct(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let sd = (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let h = llvq_llm::eval::t95(d.len().saturating_sub(1)) * sd / n.sqrt();
    let pct = |x: f64| x.exp_m1() * 100.0;
    (pct(m), pct(m - h), pct(m + h))
}

/// The NLL of `target` under binary16 logits `row`, in f64.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn nll_of(row: &[u16], target: u32) -> f64 {
    let v: Vec<f64> = row.iter().map(|&b| half::f16::from_bits(b).to_f64()).collect();
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = v.iter().map(|x| (x - m).exp()).sum();
    m + s.ln() - v[target as usize]
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run() -> anyhow::Result<()> {
    use llvq_llm::boundary::median_range;
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    use llvq_llm::persist4e_cuda::{TreeForm, TreePlan};
    let usage = "usage: LLVQ_CONFIG=<served v9 config> prefillbench --form eager|chain --long <file.jsonl> [--lengths 1024,4096,8191] [--new <n>] [--prompts <file.jsonl>] [--dump <texts.jsonl>] [--pair-texts <texts.jsonl>] [--ppl <ctx>,<windows>] [--ppl-dump <nll.jsonl>] [--pair-ppl <nll.jsonl>] [--arm a12ip|a12ipg|a12ipg32@q.o.g.d.h[/R]] [--against a12ip|a12ipg|a12ipg32@q.o.g.d.h[/R]]... [--dump-against <texts.jsonl>]... [--r <rows>] <sealed>";
    let mut a: Vec<String> = Vec::new();
    let mut form_s: Option<String> = None;
    let (mut long_file, mut prompts_file, mut dump_file, mut pair_texts) = (None::<String>, None::<String>, None::<String>, None::<String>);
    let (mut ppl, mut ppl_dump, mut pair_ppl) = (None::<(usize, usize)>, None::<String>, None::<String>);
    let mut lengths: Vec<usize> = vec![1024, 4096, 8191];
    let (mut n_new, mut r_rows) = (256usize, 16usize);
    let mut arm_s = "a12ip@8.10.8.20.6".to_string();
    let (mut against_s, mut dump_against) = (Vec::<String>::new(), Vec::<String>::new());
    let mut it = std::env::args().skip(1);
    let need = |x: Option<String>, f: &str| x.ok_or_else(|| anyhow::anyhow!("{f} needs a value; {usage}"));
    while let Some(x) = it.next() {
        match x.as_str() {
            "--form" => form_s = Some(need(it.next(), "--form")?),
            "--long" => long_file = Some(need(it.next(), "--long")?),
            "--prompts" => prompts_file = Some(need(it.next(), "--prompts")?),
            "--dump" => dump_file = Some(need(it.next(), "--dump")?),
            "--pair-texts" => pair_texts = Some(need(it.next(), "--pair-texts")?),
            "--ppl-dump" => ppl_dump = Some(need(it.next(), "--ppl-dump")?),
            "--pair-ppl" => pair_ppl = Some(need(it.next(), "--pair-ppl")?),
            "--arm" => arm_s = need(it.next(), "--arm")?,
            "--against" => against_s.push(need(it.next(), "--against")?),
            "--dump-against" => dump_against.push(need(it.next(), "--dump-against")?),
            "--new" => n_new = need(it.next(), "--new")?.parse().map_err(|e| anyhow::anyhow!("--new: {e}"))?,
            "--r" => r_rows = need(it.next(), "--r")?.parse().map_err(|e| anyhow::anyhow!("--r: {e}"))?,
            "--lengths" => {
                lengths = need(it.next(), "--lengths")?.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("--lengths: {e}"))?
            }
            "--ppl" => {
                let v: Vec<usize> = need(it.next(), "--ppl")?.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("--ppl: {e}"))?;
                anyhow::ensure!(v.len() == 2 && v[0] >= 2 && v[1] >= 2, "--ppl <ctx>,<windows>: two windows of two tokens at least");
                ppl = Some((v[0], v[1]));
            }
            _ => a.push(x),
        }
    }
    let chain = match form_s.as_deref() {
        Some("eager") => false,
        Some("chain") => true,
        o => anyhow::bail!("--form {o:?}: eager or chain; {usage}"),
    };
    let path = a.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let long_file = long_file.ok_or_else(|| anyhow::anyhow!("--long <file.jsonl>; {usage}"))?;
    anyhow::ensure!(n_new >= 2, "--new {n_new}: the decode is timed after the first token");
    anyhow::ensure!((1..=32).contains(&r_rows), "--r {r_rows}: a chain pass carries 1 to 32 rows");
    anyhow::ensure!(!lengths.is_empty() && lengths.windows(2).all(|w| w[0] < w[1]), "--lengths: increasing");
    anyhow::ensure!(chain || (pair_texts.is_none() && pair_ppl.is_none()),"--pair-*: the chain form pairs against the eager form's dumps");
    // an arm by name: a12ip, a12ipg (D1, a12ip with the grouped attention) or
    // a12ipg32 (a12ipg at 32 rows a launch); nothing else. Its rows a pass:
    // the `/R` suffix, else --r, refused above the arm's row cap.
    let parse_arm = |s: &str| -> anyhow::Result<(TreeForm, TreePlan, usize)> {
        let (s, r) = match s.split_once('/') {
            None => (s, r_rows),
            Some((a, r)) => (a, r.parse::<usize>().map_err(|e| anyhow::anyhow!("arm {s}: rows {r}: {e}"))?),
        };
        let (base, plan) = match s.split_once('@') {
            None => (s, TreePlan::Stream),
            Some((b, p)) => {
                let v: Vec<u32> = p.split('.').map(|x| x.parse::<u32>()).collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("plan {p}: {e}"))?;
                (b, TreePlan::Explicit(v.try_into().map_err(|_| anyhow::anyhow!("plan {p}: five slice counts"))?))
            }
        };
        let form = match base {
            "a12ip" => TreeForm::A12ip,
            "a12ipg" => TreeForm::A12ipg,
            "a12ipg32" => TreeForm::A12ipg32,
            _ => anyhow::bail!("prefillbench: arm {base:?}; the chain prefill runs a12ip, a12ipg or a12ipg32 (the served words read in place)"),
        };
        anyhow::ensure!(
            (1..=form.rmax() as usize).contains(&r),
            "prefillbench: arm {s} carries 1 to {} rows a launch (its kernel's row cap), {r} asked",
            form.rmax()
        );
        Ok((form, plan, r))
    };
    // the installed arms, in the runtime's order: --arm, then each --against
    let mut arms_p = vec![parse_arm(&arm_s)?];
    for s in &against_s {
        arms_p.push(parse_arm(s)?);
    }
    let arm_names: Vec<String> = std::iter::once(&arm_s).chain(&against_s).map(|s| s.split('/').next().unwrap_or("").to_string()).collect();
    let arm_r: Vec<usize> = arms_p.iter().map(|a| a.2).collect();
    let bases: Vec<&str> = arm_names.iter().map(|n| n.split('@').next().unwrap_or("")).collect();
    anyhow::ensure!(
        against_s.is_empty() || (chain && pair_texts.is_none() && ppl.is_none() && (1..bases.len()).all(|i| !bases[..i].contains(&bases[i]))),
        "--against: the chain form, arms of distinct kernels, without --pair-texts and --ppl"
    );
    anyhow::ensure!(dump_against.len() <= against_s.len(), "--dump-against: one for each --against at most");
    let cfg = llvq_llm::served::Served::from_env()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("LLVQ_CONFIG names the served config; {usage}"))?;
    anyhow::ensure!(
        cfg.decode == llvq_llm::served::Decode::Persistent && cfg.rows.form() == 7,
        "{}: the config must name the persistent decode with rows v9",
        cfg.path.display()
    );
    println!("target: {}", cfg.provenance());
    println!(
        "form {}: {}",
        if chain { "chain" } else { "eager" },
        if chain {
            format!(
                "the prompt through {} as chains of known tokens, the head on the last pass only, a one-row hand-over launch moving the last chunk, then 4E",
                arm_names.iter().zip(&arm_r).map(|(n, r)| format!("{n} at {r} rows a pass")).collect::<Vec<_>>().join(" and through ")
            )
        } else {
            format!("the served eager prefill (Qwen3::prefill_next_in, chunks of {}), then 4E", llvq_llm::model::MAX_PREFILL_ROWS)
        }
    );
    let device = candle_core::Device::new_cuda(0)?;
    let t_load = Instant::now();
    mark("load", "begin");
    let mut f = llvq_llm::fused_cuda::load_resolved(&path, &device, candle_core::DType::F16, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv, cfg.int4_layout(), Some("prefillbench"))?;
    cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut dec = cfg.decoder(&f.model, &device)?;
    for line in dec.lines() {
        println!("{line}");
    }
    card_memory("after the target's load and its 4E install")?;
    if chain {
        let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("prefillbench: the config built no 4E runtime"))?;
        let before = rt.lines.len();
        let specs: Vec<_> = arm_names.iter().zip(&arms_p).map(|(n, &(f, p, _))| (n.as_str(), p, f)).collect();
        rt.install_tree(&specs)?;
        for line in &rt.lines[before..] {
            println!("{line}");
        }
        let installed = rt.tree_arms();
        anyhow::ensure!(installed.len() == specs.len(), "prefillbench: {} arms installed, {} asked", installed.len(), specs.len());
        for a in &installed {
            anyhow::ensure!(a.void.is_none(), "prefillbench: arm {} is void: {:?}", a.name, a.void);
            println!("arm {}: {:?} kernel, slices {:?}, {} threads, area {} floats, {} B shared, row cap {}", a.name, a.form, a.splits, a.threads, a.area, a.smem, a.form.rmax());
        }
        for (n, r) in arm_names.iter().zip(&arm_r) {
            println!("arm {n}: {r} rows a pass");
        }
        card_memory("after the verify pass's install")?;
    }
    mark("load", "end");
    println!("loaded in {:.1} s", t_load.elapsed().as_secs_f64());
    let model = &f.model;
    let tok = f.tokenizer.clone();
    let marks = Marks::resolve(&tok)?;
    let eos = tok.token_to_id("<|im_end|>").ok_or_else(|| anyhow::anyhow!("prefillbench: the tokenizer has no <|im_end|>"))?;
    let opening = open_assistant(&tok, &marks, false)?;
    // the long prompt, and its cuts
    let recs = llvq_llm::spec_prompts::parse(&std::fs::read_to_string(&long_file).map_err(|e| anyhow::anyhow!("{long_file}: {e}"))?).map_err(|e| anyhow::anyhow!("{long_file}: {e}"))?;
    anyhow::ensure!(recs.len() == 1 && recs[0].messages.len() == 1 && recs[0].messages[0].0 == "user", "{long_file}: one record of one user message");
    let content = tok.encode(recs[0].messages[0].1.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let empty = turn(&tok, &marks, "user", "")?;
    let head: Vec<u32> = empty[..empty.len() - 2].to_vec();
    let tail: Vec<u32> = empty[empty.len() - 2..].iter().copied().chain(opening.iter().copied()).collect();
    let full: Vec<u32> = head.iter().chain(&content).chain(&tail).copied().collect();
    {
        let mut want = turn(&tok, &marks, "user", &recs[0].messages[0].1)?;
        want.extend(&opening);
        anyhow::ensure!(want == full, "prefillbench: the long prompt's cut does not rebuild its template");
    }
    let mut longs: Vec<(String, Vec<u32>)> = Vec::new();
    for &n in &lengths {
        anyhow::ensure!(n > head.len() + tail.len() && n <= full.len(), "length {n}: the long prompt holds {} tokens", full.len());
        let k = n - head.len() - tail.len();
        let ids: Vec<u32> = head.iter().chain(&content[..k]).chain(&tail).copied().collect();
        longs.push((format!("{}-{n}", recs[0].id), ids));
    }
    println!("long prompts: {} ({} tokens in full), cut to {:?} tokens (template kept, the first content tokens)", recs[0].id, full.len(), lengths);
    // one generation: the form's prefill into fresh caches, then n_new - 1 tokens of 4E with no stop
    struct Gen {
        tokens: Vec<u32>,
        ttft: f64,
        prefill: f64,
        decode: f64,
    }
    let generate = |dec: &mut llvq_llm::served_decode::Decoder, arm: usize, ids: &[u32], n_new: usize| -> anyhow::Result<(Gen, Vec<llvq_llm::model::KvCache>)> {
        let l = ids.len();
        model.device().synchronize()?;
        let t0 = Instant::now();
        let (mut caches, first, ttft, prefill) = if chain {
            let mut c = empty_caches(model)?;
            let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("no 4E runtime"))?;
            let ch = chain_prefill(rt, arm, &mut c, ids, arm_r[arm], false, t0)?;
            (c, ch.first, ch.ttft, ch.prefill)
        } else {
            let mut c = model.fresh_caches();
            let first = model.prefill_next_in(ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut c)?;
            model.device().synchronize()?;
            let s = t0.elapsed().as_secs_f64();
            (c, first, s, s)
        };
        let t1 = Instant::now();
        let (rest, _) = dec.continue_greedy(model, &mut caches, first, l, n_new - 1, &[])?;
        let decode = t1.elapsed().as_secs_f64();
        let mut tokens = vec![first];
        tokens.extend(rest);
        Ok((Gen { tokens, ttft, prefill, decode }, caches))
    };
    let to_eos = |t: &[u32]| -> Vec<u32> {
        match t.iter().position(|&x| x == eos) {
            Some(p) => t[..=p].to_vec(),
            None => t.to_vec(),
        }
    };
    // ---- timing ----
    println!("\ntiming: {n_new} tokens after each prompt (4E greedy, no stop), 1 discarded + {ROUNDS} timed rounds; median [range] over the timed rounds");
    println!("  prompt   first token, s           prefill to hand-over, s    decode tok/s after the first token");
    // texts[arm]: each installed arm's (the eager form's in texts[0])
    let n_arms = arm_names.len();
    // the reference: the first --against (arm 1), or the only arm
    let refk = if n_arms > 1 { 1 } else { 0 };
    // each round runs the reference first, then the other --against arms, then --arm
    let order: Vec<usize> = (1..n_arms).chain(std::iter::once(0)).collect();
    // (id, kind, prompt ids, tokens to the first <|im_end|>)
    type Text = (String, String, Vec<u32>, Vec<u32>);
    let mut texts: Vec<Vec<Text>> = vec![Vec::new(); n_arms];
    for (id, ids) in &longs {
        let l = ids.len();
        let (mut ttft, mut pre, mut rate) = (vec![Vec::new(); n_arms], vec![Vec::new(); n_arms], vec![Vec::new(); n_arms]);
        let mut reference: Vec<Option<Vec<u32>>> = vec![None; n_arms];
        if n_arms == 1 {
            mark(&format!("time-{l}"), "begin");
        }
        for round in 0..=ROUNDS {
            for &k in &order {
                let tag = format!("time-{l}-{}", arm_names[k].split('@').next().unwrap_or(""));
                if n_arms > 1 {
                    mark(&tag, "begin");
                }
                let (g, caches) = generate(&mut dec, k, ids, n_new)?;
                drop(caches);
                if n_arms > 1 {
                    mark(&tag, "end");
                }
                match &reference[k] {
                    None => reference[k] = Some(g.tokens.clone()),
                    Some(r) => anyhow::ensure!(*r == g.tokens, "prompt {l}, arm {}, round {round}: the tokens differ from round 0's", arm_names[k]),
                }
                if round > 0 {
                    ttft[k].push(g.ttft);
                    pre[k].push(g.prefill);
                    rate[k].push((n_new - 1) as f64 / g.decode);
                }
            }
        }
        if n_arms == 1 {
            mark(&format!("time-{l}"), "end");
        }
        for k in 0..n_arms {
            let (a1, a2, a3) = median_range(&ttft[k]);
            let (b1, b2, b3) = median_range(&pre[k]);
            let (c1, c2, c3) = median_range(&rate[k]);
            let who = if n_arms > 1 { format!("   {}", arm_names[k]) } else { String::new() };
            println!("  {l:>6}   {a1:7.3} [{a2:.3}-{a3:.3}]    {b1:7.3} [{b2:.3}-{b3:.3}]       {c1:7.1} [{c2:.1}-{c3:.1}]{who}");
            if n_arms > 1 {
                println!(
                    "  {l:>6}   rounds of {}: first token {}",
                    arm_names[k],
                    ttft[k].iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" ")
                );
            }
            texts[k].push((id.clone(), "long".into(), ids.clone(), to_eos(reference[k].as_ref().expect("rounds ran"))));
        }
        for k in (0..n_arms).filter(|&k| n_arms > 1 && k != refk) {
            let ratios: Vec<f64> = ttft[refk].iter().zip(&ttft[k]).map(|(b, a)| b / a).collect();
            let (m, lo, hi) = median_range(&ratios);
            println!(
                "  {l:>6}   first token of {} over {}, round by round: x{m:.3} [{lo:.3}-{hi:.3}]; the {n_new} tokens {}",
                arm_names[refk],
                arm_names[k],
                if reference[k] == reference[refk] { "identical" } else { "DIFFER" }
            );
        }
        card_memory(&format!("after the {l}-token rounds"))?;
    }
    // ---- identity gates and the phase split (chain only) ----
    if chain {
        mark("identity", "begin");
        // (arm, R) against (arm, R): the last row's argmax and logits, every layer's K and V
        let gate = |dec: &mut llvq_llm::served_decode::Decoder, ids: &[u32], (ka, ra): (usize, usize), (kb, rb): (usize, usize)| -> anyhow::Result<bool> {
            let l = ids.len();
            let mut res = Vec::new();
            for (k, r) in [(ka, ra), (kb, rb)] {
                let mut c = empty_caches(model)?;
                let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("no 4E runtime"))?;
                let ch = chain_prefill(rt, k, &mut c, ids, r, true, Instant::now())?;
                let h = kv_hashes(&c, l)?;
                println!("  {} R = {r:>2} at {l}: prefill {:.3} s, first token {}", arm_names[k], ch.prefill, ch.first);
                res.push((ch.first, ch.last_logits, h));
            }
            let (x, y) = (&res[0], &res[1]);
            let lg_same = x.1 == y.1;
            let n_lg = x.1.iter().zip(&y.1).filter(|(p, q)| p != q).count();
            let kv_bad: Vec<usize> = x.2.iter().zip(&y.2).enumerate().filter(|(_, (p, q))| p != q).map(|(i, _)| i).collect();
            let ok = x.0 == y.0 && lg_same && kv_bad.is_empty();
            let (na, nb) = if ka == kb { (String::new(), String::new()) } else { (format!("{} ", arm_names[ka]), format!("{} ", arm_names[kb])) };
            println!(
                "  identity at {l}, {na}R = {ra} against {nb}R = {rb}: argmax {} ({} and {}); last row's logits {} ({n_lg} of {} differ); K and V of {} layers: {}  => {}",
                if x.0 == y.0 { "equal" } else { "DIFFER" },
                x.0,
                y.0,
                if lg_same { "bit for bit" } else { "DIFFER" },
                x.1.len(),
                x.2.len(),
                if kv_bad.is_empty() { "bit for bit".to_string() } else { format!("DIFFER at layers {kv_bad:?}") },
                if ok { "HOLDS" } else { "FAILS" }
            );
            Ok(ok)
        };
        // (arm, length, holds)
        let mut arm_gates: Vec<(usize, usize, bool)> = Vec::new();
        let r0 = arm_r[0];
        let (g1, g1b) = if n_arms > 1 {
            for k in (0..n_arms).filter(|&k| k != refk) {
                println!(
                    "\nidentity: {} at R = {} against {} at R = {} at every length (FNV-1a of each layer's K and V bits over the prompt)",
                    arm_names[k], arm_r[k], arm_names[refk], arm_r[refk]
                );
                for (_, ids) in &longs {
                    arm_gates.push((k, ids.len(), gate(&mut dec, ids, (k, arm_r[k]), (refk, arm_r[refk]))?));
                }
            }
            (true, true)
        } else {
            println!("\nidentity: the chain prefill against the pass's one-row path (FNV-1a of each layer's K and V bits over the prompt)");
            let g1 = gate(&mut dec, &longs[0].1, (0, r0), (0, 1))?;
            let g1b = if longs.len() > 1 { gate(&mut dec, &longs[1].1, (0, 8), (0, r0))? } else { true };
            (g1, g1b)
        };
        mark("identity", "end");
        // the phase split of the longest prompt's last three passes, each arm
        mark("phases", "begin");
        for (arm, name) in arm_names.iter().enumerate() {
            let ptag = format!("phases-{}", bases[arm]);
            mark(&ptag, "begin");
            println!("  phases of {name} at R = {}:", arm_r[arm]);
            use llvq_cuda::pers::{n_bar_a9, p4t_phases_a9, P4T_PHASES_A9};
            use llvq_cuda::spec_loop::{chain_launches, ChainFault};
            let ids = &longs.last().expect("a length").1;
            let mut c = empty_caches(model)?;
            let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("no 4E runtime"))?;
            let (layers, grid) = rt.stamp_shape();
            let n_ev = (1 + 2 * n_bar_a9(layers, 0)) as usize * grid as usize;
            let (ls, _, _) = chain_launches(ids, 0, arm_r[arm], ChainFault::None).map_err(anyhow::Error::msg)?;
            let n = ls.len();
            let mut per: Vec<Vec<f64>> = vec![Vec::new(); P4T_PHASES_A9.len()];
            let mut walls = Vec::new();
            for (i, l) in ls.iter().enumerate() {
                let stamped = i + 3 >= n;
                let head = i + 1 == n;
                let t = Instant::now();
                let o = rt.tree_chain(arm, &mut c, l.pos0, &l.toks, &l.mv, l.move_pos0, head, false, stamped)?;
                if stamped {
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(o.stamps.len() >= n_ev, "{} stamps, the phases need {n_ev}", o.stamps.len());
                    let ph = p4t_phases_a9(&o.stamps[..n_ev], layers, grid).map_err(anyhow::Error::msg)?;
                    let total: f64 = ph.iter().map(|x| x[0]).sum();
                    let attn: f64 = ph[5][0];
                    println!(
                        "  pass {i} of {n} (pos0 {}, {} rows, head {}): wall {:.3} ms host, phases {:.3} ms, attn {:.3} ms ({:.1} %), the rest {:.3} ms",
                        l.pos0,
                        l.toks.len(),
                        if head { "on" } else { "off" },
                        walls.last().expect("pushed"),
                        total / 1e6,
                        attn / 1e6,
                        100.0 * attn / total,
                        (total - attn) / 1e6
                    );
                    for (k, x) in ph.iter().enumerate() {
                        per[k].push(x[0] / 1e6);
                    }
                }
            }
            println!("  phases of the longest prompt's last 3 passes, ms each (summed over the layers, grid clock):");
            for (k, name) in P4T_PHASES_A9.iter().enumerate() {
                println!("    {name:>11}  {}", per[k].iter().map(|x| format!("{x:8.3}")).collect::<Vec<_>>().join(" "));
            }
            let shares: Vec<f64> = (0..per[5].len()).map(|j| per[5][j] / per.iter().map(|p| p[j]).sum::<f64>()).collect();
            let (m, lo, hi) = median_range(&shares);
            println!("  attention share of a pass at {}: median {:.1} % [{:.1}-{:.1}] over the 3 stamped passes", ids.len(), 100.0 * m, 100.0 * lo, 100.0 * hi);
            mark(&ptag, "end");
        }
        mark("phases", "end");
        if n_arms > 1 {
            for k in (0..n_arms).filter(|&k| k != refk) {
                println!(
                    "identity gates, {} at R = {} against {} at R = {}: {}",
                    arm_names[k],
                    arm_r[k],
                    arm_names[refk],
                    arm_r[refk],
                    arm_gates.iter().filter(|g| g.0 == k).map(|(_, l, ok)| format!("{l} {}", if *ok { "HOLDS" } else { "FAILS" })).collect::<Vec<_>>().join(", ")
                );
            }
        } else {
            println!("identity gates: R = {r0} against R = 1 at {}: {}; R = 8 against R = {r0} at {}: {}", longs[0].1.len(), if g1 { "HOLDS" } else { "FAILS" }, longs.get(1).map_or(0, |x| x.1.len()), if g1b { "HOLDS" } else { "FAILS" });
        }
    }
    // ---- texts: the house prompts ----
    if let Some(file) = &prompts_file {
        mark("texts", "begin");
        let recs = llvq_llm::spec_prompts::parse(&std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("{file}: {e}"))?).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
        let (mut pre, mut rate) = (vec![Vec::new(); n_arms], vec![Vec::new(); n_arms]);
        for r in recs {
            let mut ids = Vec::new();
            for (role, c) in &r.messages {
                ids.extend(turn(&tok, &marks, role, c)?);
            }
            ids.extend(&opening);
            for &k in &order {
                let (g, _) = generate(&mut dec, k, &ids, n_new)?;
                pre[k].push(g.prefill);
                rate[k].push((n_new - 1) as f64 / g.decode);
                texts[k].push((r.id.clone(), r.kind.clone(), ids.clone(), to_eos(&g.tokens)));
            }
        }
        mark("texts", "end");
        for k in 0..n_arms {
            let (p1, p2, p3) = median_range(&pre[k]);
            let (q1, q2, q3) = median_range(&rate[k]);
            let who = if chain { format!("chain ({})", arm_names[k]) } else { "eager".to_string() };
            println!("\nhouse prompts: {} through the {who} prefill; prefill {p1:.3} s [{p2:.3}-{p3:.3}], decode {q1:.1} tok/s [{q2:.1}-{q3:.1}], median [range] over prompts", pre[k].len());
        }
    }
    let dumps: Vec<(usize, &String)> = dump_file.iter().map(|p| (0usize, p)).chain(dump_against.iter().enumerate().map(|(i, p)| (i + 1, p))).collect();
    for (k, p) in dumps {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| anyhow::anyhow!("{p}: {e}"))?);
        for (id, kind, ids, t) in &texts[k] {
            let l = ids.len();
            let text = tok.decode(t, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            let arm = if chain { arm_names[k].as_str() } else { "" };
            writeln!(w, "{}", serde_json::json!({ "id": id, "kind": kind, "form": if chain { "chain" } else { "eager" }, "arm": arm, "prompt_len": l, "tokens": t, "text": text }))?;
        }
        w.flush()?;
        println!("dump: {p}, {} texts", texts[k].len());
    }
    // each arm's texts paired with the reference's in the process: identical, or the first divergence
    for k in (0..n_arms).filter(|&k| n_arms > 1 && k != refk && !texts[k].is_empty()) {
        let mut same = 0usize;
        for ((id, kind, _, a), (id_b, _, _, b)) in texts[k].iter().zip(&texts[refk]) {
            anyhow::ensure!(id == id_b, "prefillbench: the arms' texts are out of order ({id} and {id_b})");
            match a.iter().zip(b).position(|(x, y)| x != y).or(if a.len() != b.len() { Some(a.len().min(b.len())) } else { None }) {
                None => same += 1,
                Some(q) => println!("  {id} ({kind}): first divergence at generated token {q} of {} and {}: {} {:?}, {} {:?}", a.len(), b.len(), arm_names[k], a.get(q), arm_names[refk], b.get(q)),
            }
        }
        println!("texts identical through {} and {}: {same} of {}", arm_names[k], arm_names[refk], texts[k].len());
    }
    if let Some(p) = &pair_texts {
        mark("pair-texts", "begin");
        let other: Vec<serde_json::Value> = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{p}: {e}"))?.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?;
        println!("\ntexts after the chain prefill against the eager prefill's ({p}); near-tie: the pass's margin between the two tokens at most 0.25 in logit");
        let (mut same, mut total, mut ties) = (0usize, 0usize, 0usize);
        for (id, kind, ids, t) in &texts[0] {
            let Some(o) = other.iter().find(|v| v["id"].as_str() == Some(id.as_str())) else {
                println!("  {id}: absent from the eager dump");
                continue;
            };
            let e: Vec<u32> = o["tokens"].as_array().map(|v| v.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default();
            total += 1;
            let div = t.iter().zip(&e).position(|(x, y)| x != y).or(if t.len() != e.len() { Some(t.len().min(e.len())) } else { None });
            match div {
                None => same += 1,
                Some(q) => {
                    let (a, b) = (t.get(q).copied(), e.get(q).copied());
                    // the pass's logits for both tokens at that position, on the chain's own prefix
                    let m = match (a, b, ids.is_empty()) {
                        (Some(a), Some(b), false) => {
                            let seq: Vec<u32> = ids.iter().chain(t.iter()).copied().collect();
                            let l = ids.len();
                            let mut c = empty_caches(model)?;
                            let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("no 4E runtime"))?;
                            // the row at position l + q - 1 predicts generated token q
                            let upto = l + q;
                            let ch = chain_prefill(rt, 0, &mut c, &seq[..upto], arm_r[0], true, Instant::now())?;
                            let lg = |x: u32| half::f16::from_bits(ch.last_logits[x as usize]).to_f32();
                            let mm = lg(a) - lg(b);
                            if mm.abs() <= 0.25 {
                                ties += 1;
                            }
                            format!("the pass's logits {} for {a} (chain), {} for {b} (eager), margin {mm:.4}{}", lg(a), lg(b), if mm.abs() <= 0.25 { ", a near-tie" } else { ", NOT a near-tie" })
                        }
                        _ => "margin not computed: one text ended first".to_string(),
                    };
                    println!("  {id} ({kind}): first divergence at generated token {q} of {} and {}: chain {a:?}, eager {b:?}; {m}", t.len(), e.len());
                }
            }
        }
        mark("pair-texts", "end");
        println!("texts identical after the chain and the eager prefill: {same} of {total}; divergences on a near-tie: {ties} of {}", total - same);
    }
    // ---- perplexity ----
    if let Some((ctx, windows)) = ppl {
        mark("ppl", "begin");
        let text = llvq_llm::corpus::wikitext2_test()?;
        let wid = f.tokenizer.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        let nwin = (wid.len() / ctx).min(windows);
        anyhow::ensure!(nwin >= 2, "--ppl: {} tokens hold {nwin} windows of {ctx}", wid.len());
        let span = &wid[..nwin * ctx];
        println!("\nperplexity: wikitext2 test, {nwin} windows of {ctx} from window 0 (tokens {:016x}), {}", llvq_llm::eval::token_fingerprint(span), if chain {
            format!("chain passes of {}, the head and logits on every row (binary16 logits, NLL in f64)", arm_r[0])
        } else {
            "Qwen3::window_nll, the published prefill path".to_string()
        });
        let mut nll = Vec::with_capacity(nwin);
        let t0 = Instant::now();
        for w in 0..nwin {
            let win = &span[w * ctx..(w + 1) * ctx];
            let m = if chain {
                use llvq_cuda::spec_loop::{chain_launches, ChainFault};
                let mut c = empty_caches(model)?;
                let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("no 4E runtime"))?;
                let (ls, _, _) = chain_launches(win, 0, arm_r[0], ChainFault::None).map_err(anyhow::Error::msg)?;
                let mut s = 0.0f64;
                for l in &ls {
                    let o = rt.tree_chain(0, &mut c, l.pos0, &l.toks, &l.mv, l.move_pos0, true, true, false)?;
                    let vv = o.logits.len() / l.toks.len();
                    for k in 0..l.toks.len() {
                        let p = l.pos0 + k;
                        if p + 1 < ctx {
                            s += nll_of(&o.logits[k * vv..(k + 1) * vv], win[p + 1]);
                        }
                    }
                }
                s / (ctx - 1) as f64
            } else {
                let (n, c) = model.window_nll(win, &mut llvq_llm::model::NoCapture)?;
                n / c as f64
            };
            nll.push(m);
            println!("  window {:>3}/{nwin}: mean nll {m:.9} ({:.0} s)", w + 1, t0.elapsed().as_secs_f64());
        }
        mark("ppl", "end");
        let mean = nll.iter().sum::<f64>() / nll.len() as f64;
        println!("  perplexity {:.4} (exp of the mean of the window means)", mean.exp());
        if let Some(p) = &ppl_dump {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| anyhow::anyhow!("{p}: {e}"))?);
            for (i, x) in nll.iter().enumerate() {
                writeln!(w, "{}", serde_json::json!({ "window": i, "ctx": ctx, "nll": x, "form": if chain { "chain" } else { "eager" } }))?;
            }
            w.flush()?;
        }
        if let Some(p) = &pair_ppl {
            let other: Vec<f64> = std::fs::read_to_string(p)
                .map_err(|e| anyhow::anyhow!("{p}: {e}"))?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str::<serde_json::Value>(l).map(|v| v["nll"].as_f64().unwrap_or(f64::NAN)))
                .collect::<Result<_, _>>()?;
            anyhow::ensure!(other.len() == nll.len(), "{p}: {} windows against {}", other.len(), nll.len());
            let (m, lo, hi) = paired_ppl_pct(&other, &nll);
            println!(
                "  paired perplexity, chain against eager over {} windows: {m:+.4} % [{lo:+.4}, {hi:+.4}] (95 %, t); band +-0.12 %: {}",
                nll.len(),
                if lo >= -0.12 && hi <= 0.12 { "INSIDE" } else { "OUTSIDE" }
            );
        }
    }
    card_memory("at the end")?;
    Ok(())
}
