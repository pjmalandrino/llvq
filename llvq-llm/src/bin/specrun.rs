//! The speculative decode's gate on the card: a sealed 0.6B drafts, the served
//! target checks with 4E's verify pass, and every token must be the target's
//! greedy token (`docs/plan-speculation.md`).
//!
//! Usage:
//!   `LLVQ_CONFIG=<target config> specrun <target sealed> <draft config> <draft sealed> [ks] [what]`
//!   `LLVQ_CONFIG=<target config> specrun <target sealed> lookup [ks] [what]`
//!
//! `lookup` drafts by prompt lookup (`speculative::lookup_draft`): n-grams of
//! the prompt and of the text so far, no drafter model, no memory. A round
//! with no match verifies the pending token alone.
//!
//! - `ks`: the draft lengths to run, comma-separated, each below
//!   `pers::P4V_KMAX` (default `1,2,3`).
//! - `what`: `all` (default), `load` (the install and the card memory only),
//!   `door` (the four prompts) or `gsm8k` (the 50-problem pilot).
//!
//! Both configs must name the persistent decode at row form 7 (v9). The
//! target's verify pass is installed beside its 4E decode.
//!
//! Prompts (`door`): `chat`, `code`, `summary`, `edit` (a function to return
//! changed, most of it copied) and `rag` (a question to answer from a passage,
//! quoting it), each one user turn through
//! the chat template with reasoning off, then the house prompt of the served
//! door ("The capital of France is", raw, the prompt every served tok/s ran),
//! which loops and so flatters any drafter: it is printed, never the verdict.
//! 256 new tokens each, cut at the chat's stop tokens. For each: the target's
//! greedy 4E decode makes the reference tokens; each `k` must make the same
//! ones in every round or the run is void. Then 1 discarded and 5 timed
//! rounds, greedy then each `k` in turn, timed as the door times a generate
//! (prompt and decode, the drafter's prefill included in the speculative
//! arm), the decode alone beside it, the ratio to greedy round by round.
//!
//! `gsm8k`: the 50 problems `bin/gsm8k` scores at limit 50 (the served pilots'),
//! zero-shot, reasoning off, 1,024 new tokens at most, cut at the stop tokens.
//! Greedy 4E, then each `k`: every completion must be the greedy one, token
//! for token. Decode tok/s summed over the 50 problems, and the correct count.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        run()
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        anyhow::bail!("specrun: the speculative decode runs 4E, one cooperative CUDA launch; this build has no CUDA backend")
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const ROUNDS_TIMED: usize = 5;
#[cfg(all(target_os = "linux", feature = "cuda"))]
const N_NEW: usize = 256;

#[cfg(all(target_os = "linux", feature = "cuda"))]
struct Loaded {
    f: llvq_llm::fused::FusedSealed,
    dec: llvq_llm::served_decode::Decoder,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn card_used() -> anyhow::Result<f64> {
    let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
    Ok((total - free) as f64 / 1e9)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn load(cfg: &llvq_llm::served::Served, path: &str, device: &candle_core::Device, what: &str) -> anyhow::Result<Loaded> {
    anyhow::ensure!(
        cfg.decode == llvq_llm::served::Decode::Persistent && cfg.rows.form() == 7,
        "{}: the {what} must name the persistent decode with rows v9",
        cfg.path.display()
    );
    println!("{what}: {}", cfg.provenance());
    let mut f = llvq_llm::fused_cuda::load_resolved(
        path,
        device,
        candle_core::DType::F16,
        cfg.layout,
        cfg.embed,
        cfg.rot_share,
        cfg.fuse,
        cfg.kv,
        cfg.int4_layout(),
        Some("specrun"),
    )?;
    cfg.apply(&mut f.model, device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let dec = cfg.decoder(&f.model, device)?;
    Ok(Loaded { f, dec })
}

/// The greedy reference and the speculative arms, one prompt.
#[cfg(all(target_os = "linux", feature = "cuda"))]
struct Arm {
    tokens: Vec<u32>,
    door_s: f64,
    decode_s: f64,
    stats: llvq_llm::speculative::SpecStats,
}

/// One generate: the target's prefill (and the drafter's when `k` is some),
/// then 4E greedy or the speculative loop.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn generate(t: &mut Loaded, mut d: Option<&mut Loaded>, ids: &[u32], max_new: usize, stops: &[u32], k: Option<usize>) -> anyhow::Result<Arm> {
    let chunk = llvq_llm::model::MAX_PREFILL_ROWS;
    let t0 = Instant::now();
    let mut caches = t.f.model.fresh_caches();
    let first = t.f.model.prefill_next_in(ids, chunk, &mut caches)?;
    let mut dcaches = Vec::new();
    if let (Some(_), Some(d)) = (k, d.as_mut()) {
        dcaches = d.f.model.fresh_caches();
        let _ = d.f.model.prefill_next_in(ids, chunk, &mut dcaches)?;
    }
    t.f.model.device().synchronize()?;
    let t1 = Instant::now();
    let mut tokens = Vec::with_capacity(max_new);
    let mut stats = llvq_llm::speculative::SpecStats::default();
    if !stops.contains(&first) {
        tokens.push(first);
        let (rest, _) = match k {
            None => {
                let (r, s) = t.dec.continue_greedy(&t.f.model, &mut caches, first, ids.len(), max_new - 1, stops)?;
                (r, s)
            }
            Some(k) => {
                let (r, s, st) = match d {
                    Some(d) => t.dec.continue_speculative(&mut caches, &mut d.dec, &mut dcaches, first, ids.len(), max_new - 1, stops, k)?,
                    None => t.dec.continue_lookup(&mut caches, ids, first, max_new - 1, stops, k)?,
                };
                stats = st;
                (r, s)
            }
        };
        tokens.extend(rest);
    }
    t.f.model.device().synchronize()?;
    Ok(Arm { tokens, door_s: t0.elapsed().as_secs_f64(), decode_s: t1.elapsed().as_secs_f64(), stats })
}

/// Where a speculative arm's time went: the mean verify pass and the mean
/// drafter launch in ms, and the verify pass in greedy tokens (`greedy_rate`
/// the greedy arm's decode tok/s on the same prompts): what one pass of k + 1
/// rows costs against one 4E token.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn cost_line(s: &llvq_llm::speculative::SpecStats, greedy_rate: f64) -> String {
    let v = s.verify_ns as f64 / 1e6 / s.rounds.max(1) as f64;
    let d = s.draft_ns as f64 / 1e6 / s.draft_launches.max(1) as f64;
    format!("verify {v:.3} ms a pass (x{:.3} a greedy token), drafter {d:.3} ms a launch", v * greedy_rate / 1e3)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(a.iter().zip(b).take_while(|(x, y)| x == y).count())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run() -> anyhow::Result<()> {
    use llvq_llm::boundary::median_range;
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: LLVQ_CONFIG=<target config> specrun <target sealed> <draft config> <draft sealed> [ks] [all|load|door|gsm8k]\n       \
                 LLVQ_CONFIG=<target config> specrun <target sealed> lookup [ks] [all|load|door|gsm8k]";
    let lookup = a.get(1).map(String::as_str) == Some("lookup");
    // Positional arguments after the drafter: one fewer with prompt lookup.
    let rest = if lookup { 2 } else { 3 };
    let (tpath, dcfg_path, dpath) = match (a.first(), a.get(1), a.get(2)) {
        (Some(t), Some(_), _) if lookup => (t.clone(), String::new(), String::new()),
        (Some(t), Some(c), Some(d)) => (t.clone(), c.clone(), d.clone()),
        _ => anyhow::bail!("{usage}"),
    };
    let ks: Vec<usize> = a
        .get(rest)
        .map_or("1,2,3", String::as_str)
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|_| anyhow::anyhow!("ks: {s:?} is not a count")))
        .collect::<anyhow::Result<_>>()?;
    let kmax = llvq_cuda::pers::P4V_KMAX as usize;
    anyhow::ensure!(!ks.is_empty() && ks.iter().all(|&k| (1..kmax).contains(&k)), "ks {ks:?}: each in 1..{}", kmax - 1);
    let what = a.get(rest + 1).map_or("all", String::as_str);
    anyhow::ensure!(["all", "load", "door", "gsm8k"].contains(&what), "{usage}");

    let tcfg = llvq_llm::served::Served::from_env()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("LLVQ_CONFIG names the target's served config; {usage}"))?;
    let device = candle_core::Device::new_cuda(0)?;

    let t_load = Instant::now();
    let mut t = load(&tcfg, &tpath, &device, "target")?;
    for line in t.dec.lines() {
        println!("{line}");
    }
    let used_target = card_used()?;
    println!("card memory: {used_target:.3} GB used after the target's load and its 4E install (cuMemGetInfo)");
    let mut d = if lookup {
        println!("drafter: prompt lookup, n-grams {} to {} of the prompt and the text so far; no model", llvq_llm::speculative::LOOKUP_MAX_N, llvq_llm::speculative::LOOKUP_MIN_N);
        None
    } else {
        let dcfg = llvq_llm::served::Served::read(std::path::Path::new(&dcfg_path)).map_err(|e| anyhow::anyhow!("{e}"))?;
        let d = load(&dcfg, &dpath, &device, "drafter")?;
        for line in d.dec.lines() {
            println!("drafter {line}");
        }
        Some(d)
    };
    let before = t.dec.lines().len();
    t.dec.install_verify()?;
    for line in &t.dec.lines()[before..] {
        println!("{line}");
    }
    let used_all = card_used()?;
    println!(
        "card memory: {used_all:.3} GB used after the drafter's load, its 4E install and the target's verify pass (cuMemGetInfo): {:+.3} GB",
        used_all - used_target
    );
    println!(
        "weights: target {:.3} GB, drafter {:.3} GB (runtime and carried bytes)",
        (t.f.runtime_bytes + t.f.carried_bytes) as f64 / 1e9,
        d.as_ref().map_or(0.0, |d| (d.f.runtime_bytes + d.f.carried_bytes) as f64 / 1e9)
    );
    println!("loaded in {:.1} s; draft lengths {ks:?}", t_load.elapsed().as_secs_f64());
    if what == "load" {
        return Ok(());
    }
    let tok = t.f.tokenizer.clone();
    let marks = Marks::resolve(&tok)?;
    let stops = marks.stops().to_vec();
    let chat = |body: &str| -> anyhow::Result<Vec<u32>> {
        let mut ids = turn(&tok, &marks, "user", body)?;
        ids.extend(open_assistant(&tok, &marks, false)?);
        Ok(ids)
    };

    if what == "all" || what == "door" {
        let prompts: Vec<(&str, Vec<u32>, Vec<u32>)> = vec![
            ("chat", chat(CHAT)?, stops.clone()),
            ("code", chat(CODE)?, stops.clone()),
            ("summary", chat(&format!("{SUMMARY}\n\n{SUMMARY_TEXT}"))?, stops.clone()),
            ("edit", chat(&format!("{EDIT}\n\n{EDIT_CODE}"))?, stops.clone()),
            ("rag", chat(&format!("{RAG}\n\n{SUMMARY_TEXT}"))?, stops.clone()),
            ("door", tok.encode(DOOR, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec(), Vec::new()),
        ];
        for (name, ids, st) in &prompts {
            let want = generate(&mut t, d.as_mut(), ids, N_NEW, st, None)?.tokens;
            println!(
                "\n== {name}: {} prompt tokens, greedy reference {} tokens, fingerprint {:016x}",
                ids.len(),
                want.len(),
                llvq_llm::eval::token_fingerprint(&want)
            );
            let arms: Vec<Option<usize>> = std::iter::once(None).chain(ks.iter().map(|&k| Some(k))).collect();
            let mut door = vec![Vec::new(); arms.len()];
            let mut dec = vec![Vec::new(); arms.len()];
            let mut stats = vec![llvq_llm::speculative::SpecStats::default(); arms.len()];
            for round in 0..=ROUNDS_TIMED {
                for (i, &k) in arms.iter().enumerate() {
                    let g = generate(&mut t, d.as_mut(), ids, N_NEW, st, k)?;
                    if let Some(at) = first_divergence(&want, &g.tokens) {
                        anyhow::bail!(
                            "{name}, round {round}, {}: diverges from the greedy reference at token {at} ({} against {} tokens); the run is void",
                            k.map_or("greedy".to_string(), |k| format!("k = {k}")),
                            g.tokens.len(),
                            want.len()
                        );
                    }
                    if round > 0 {
                        door[i].push(g.tokens.len() as f64 / g.door_s);
                        dec[i].push(g.tokens.len().saturating_sub(1) as f64 / g.decode_s);
                        stats[i] = g.stats;
                    }
                }
            }
            println!("{name}: every arm made the greedy tokens in every round, {} of {}", want.len(), want.len());
            let (greedy_dm, _, _) = median_range(&dec[0]);
            for (i, &k) in arms.iter().enumerate() {
                let (m, lo, hi) = median_range(&door[i]);
                let (dm, dlo, dhi) = median_range(&dec[i]);
                let label = k.map_or("greedy".to_string(), |k| format!("k = {k}  "));
                let mut line = format!("  {name} {label}: door {m:7.1} tok/s [{lo:.1}-{hi:.1}], decode {dm:7.1} [{dlo:.1}-{dhi:.1}]");
                if i > 0 {
                    let r: Vec<f64> = door[i].iter().zip(&door[0]).map(|(a, b)| a / b).collect();
                    let rd: Vec<f64> = dec[i].iter().zip(&dec[0]).map(|(a, b)| a / b).collect();
                    let (rm, rlo, rhi) = median_range(&r);
                    let (qm, qlo, qhi) = median_range(&rd);
                    let s = stats[i];
                    line += &format!(
                        ", / greedy round by round: door x{rm:.3} [{rlo:.3}-{rhi:.3}], decode x{qm:.3} [{qlo:.3}-{qhi:.3}]; acceptance {:.3} ({} of {}), {:.2} tokens a verify, {} catch-ups; {}",
                        s.acceptance(),
                        s.accepted,
                        s.drafted,
                        s.tokens_per_round(),
                        s.catchups,
                        cost_line(&s, greedy_dm)
                    );
                }
                println!("{line}");
            }
            println!("  text: {}", tok.decode(&want, true).unwrap_or_default().replace('\n', " "));
        }
    }

    if what == "all" || what == "gsm8k" {
        let (items, revision) = llvq_llm::corpus::gsm8k_split("test")?;
        let picked = llvq_llm::gsm8k::select(items.len(), 50);
        println!("\n== gsm8k: {} problems of {}, revision {}, max_new 1024, reasoning off", picked.len(), items.len(), revision.as_deref().unwrap_or("not from the Hub cache"));
        let prompts: Vec<Vec<u32>> = picked.iter().map(|&i| chat(&llvq_llm::gsm8k::user_body(&items[i].question))).collect::<anyhow::Result<_>>()?;
        let arms: Vec<Option<usize>> = std::iter::once(None).chain(ks.iter().map(|&k| Some(k))).collect();
        let mut greedy: Vec<Vec<u32>> = Vec::new();
        let mut greedy_rate = 0.0f64;
        for &k in &arms {
            let (mut n_tok, mut dec_s, mut correct) = (0usize, 0.0f64, 0usize);
            let mut st = llvq_llm::speculative::SpecStats::default();
            let mut all = Vec::with_capacity(prompts.len());
            for (j, ids) in prompts.iter().enumerate() {
                let g = generate(&mut t, d.as_mut(), ids, 1024, &stops, k)?;
                if k.is_some() {
                    if let Some(at) = first_divergence(&greedy[j], &g.tokens) {
                        anyhow::bail!("gsm8k problem {}, k = {}: diverges from the greedy completion at token {at}; the run is void", picked[j], k.unwrap_or(0));
                    }
                }
                n_tok += g.tokens.len().saturating_sub(1);
                dec_s += g.decode_s;
                st.rounds += g.stats.rounds;
                st.drafted += g.stats.drafted;
                st.accepted += g.stats.accepted;
                st.catchups += g.stats.catchups;
                st.draft_launches += g.stats.draft_launches;
                st.draft_ns += g.stats.draft_ns;
                st.verify_ns += g.stats.verify_ns;
                let text = tok.decode(&g.tokens, true).unwrap_or_default();
                let gold = llvq_llm::gsm8k::gold(&items[picked[j]].answer).unwrap_or_default();
                let (_, got) = llvq_llm::gsm8k::extract(&text);
                if llvq_llm::gsm8k::is_correct(got.as_deref(), &gold) {
                    correct += 1;
                }
                all.push(g.tokens);
            }
            let label = k.map_or("greedy".to_string(), |k| format!("k = {k}"));
            if k.is_none() {
                greedy_rate = n_tok as f64 / dec_s;
            }
            let mut line = format!("  gsm8k {label}: {correct} of {} correct, decode {:.1} tok/s over {n_tok} tokens", prompts.len(), n_tok as f64 / dec_s);
            if k.is_some() {
                line += &format!(
                    "; completions identical to greedy {} of {}; acceptance {:.3} ({} of {}), {:.2} tokens a verify; {}",
                    prompts.len(),
                    prompts.len(),
                    st.acceptance(),
                    st.accepted,
                    st.drafted,
                    st.tokens_per_round(),
                    cost_line(&st, greedy_rate)
                );
            }
            println!("{line}");
            if k.is_none() {
                let flat: Vec<u32> = all.iter().flatten().copied().collect();
                println!("  gsm8k greedy completions fingerprint {:016x}", llvq_llm::eval::token_fingerprint(&flat));
                greedy = all;
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const DOOR: &str = "The capital of France is";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const CHAT: &str = "I am planning a three-day trip to Lisbon in late autumn with my partner. We like walking, \
small museums and local food, and we do not want to rent a car. Could you suggest a plan for each day and a few \
practical tips?";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const CODE: &str = "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs \
and returns the merged list of overlapping intervals, sorted by start. Include a docstring, type hints, and a few \
unit tests with pytest.";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const SUMMARY: &str = "Summarize the following text in five short bullet points.";
// Written for this gate (2026-11-07), not quoted from any source.
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
const EDIT: &str = "Rename the variable `buf` to `pending` everywhere in this function and add type hints to its \
signature. Return the whole function, changed, and nothing else.";
// Written for this gate (2026-11-08), not quoted from any source.
#[cfg(all(target_os = "linux", feature = "cuda"))]
const EDIT_CODE: &str = "```python
def batch_lines(lines, size, sep):
    out = []
    buf = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if len(buf) >= size:
            out.append(sep.join(buf))
            buf = []
    if buf:
        out.append(sep.join(buf))
    return out
```";
#[cfg(all(target_os = "linux", feature = "cuda"))]
const RAG: &str = "Answer from the text below only. What did the council's engineer report, and what did the mayor \
decide? Quote the sentences of the text that support your answer.";
