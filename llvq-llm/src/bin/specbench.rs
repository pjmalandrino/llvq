//! The speculative decode loop end to end on the card (`llvq_cuda::spec_loop`
//! around the tree verify pass): for each prompt, N greedy tokens by 4E and
//! by the loop with each drafter asked, token equality asserted over all N,
//! and the loop's time split into the verify launches, the drafter calls and
//! the rest of the host.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> specbench <sealed> <arm> <drafters> [N]`
//!
//! * `<arm>`: the verify pass, `a12` (taps on) or `a10t384r16` (no taps), with
//!   an optional plan suffix `@q.o.g.d.h` (else the stream plan at its warps);
//! * `<drafters>`: comma-separated, `null` (R = 1 decoding) and `oracle:K`
//!   (K = 0 .. 14: K correct tokens of 4E's greedy continuation then a wrong
//!   one, a pass of K + 2 rows accepting K);
//! * `N`: tokens a prompt, default 64.
//!
//! Prompts: specrun's chat prompt and the rag prompt of vtreebench, and two
//! written here (code and grade-school arithmetic); the DFlash journal's
//! prompt ids are not in this repository.
//!
//! Per prompt and drafter, over 1 discarded and 3 timed runs: accepted drafted
//! tokens a pass, passes, the walls (verify launches, drafter calls, the rest
//! of the host), tok/s end to end and with the drafter's time excluded,
//! against 4E's greedy tok/s on the same prompt (medians, ranges). For the
//! oracle at K, the check: greedy x (K + 1) / c_(K + 2), c_R a tree launch of
//! R chained rows against one 4E token, timed here.

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
    let usage = "usage: LLVQ_CONFIG=<served v9 config> specbench <sealed> <a12|a10t384r16>[@q.o.g.d.h] <null,oracle:K,...> [N]";
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let arm_s = a.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let drafters_s = a.get(2).ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let n_tok: usize = a.get(3).map_or(Ok(64), |s| s.parse()).map_err(|e| anyhow::anyhow!("N: {e}; {usage}"))?;
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
    // drafters: (name, oracle K or None)
    let mut drafters: Vec<(String, Option<usize>)> = Vec::new();
    for d in drafters_s.split(',') {
        match d.split_once(':') {
            None if d == "null" => drafters.push(("null".into(), None)),
            Some(("oracle", k)) => {
                let k: usize = k.parse().map_err(|e| anyhow::anyhow!("oracle:{k}: {e}; {usage}"))?;
                anyhow::ensure!(k + 2 <= 16, "oracle:{k}: a pass of {} rows, the arm carries 16", k + 2);
                drafters.push((d.to_string(), Some(k)));
            }
            _ => anyhow::bail!("specbench: unknown drafter {d:?}; {usage}"),
        }
    }
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
    println!("arm {}: {:?} kernel, slices {:?}, {} threads; taps {}; N {n_tok}", arm0.name, arm0.form, arm0.splits, arm0.threads, if taps { "on" } else { "off" });
    println!("loaded in {:.1} s", t_load.elapsed().as_secs_f64());
    let tok = f.tokenizer.clone();
    let marks = Marks::resolve(&tok)?;
    let chat = |body: &str| -> anyhow::Result<Vec<u32>> {
        let mut ids = turn(&tok, &marks, "user", body)?;
        ids.extend(open_assistant(&tok, &marks, false)?);
        Ok(ids)
    };
    let model = &f.model;
    let hidden = model.config().hidden_size;
    let tap_layers = [2usize, 10, 18, 26, 34];
    let set = |c: &mut [llvq_llm::model::KvCache], n: usize| -> anyhow::Result<()> {
        for x in c.iter_mut() {
            x.set_len(n)?;
        }
        Ok(())
    };
    let prompts = [("chat", chat(CHAT)?), ("rag", chat(&format!("{RAG}\n\n{SUMMARY_TEXT}"))?), ("code", chat(CODE)?), ("math", chat(MATH)?)];
    let mut summary = Vec::new();
    for (pname, ids) in &prompts {
        let l = ids.len();
        let mut caches = model.fresh_caches();
        // the prompt's taps, from the eager prefill (to rounding of the pass's own)
        let (first, prompt_taps) = if taps {
            model.prefill_taps(ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut caches, &tap_layers)?
        } else {
            (model.prefill_next_in(ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut caches)?, Vec::new())
        };
        model.device().synchronize()?;
        // 4E greedy, timed (1 discarded, ROUNDS timed)
        let mut want = Vec::new();
        let mut t4e = Vec::new();
        for round in 0..=ROUNDS {
            set(&mut caches, l)?;
            rt.sync()?;
            let t0 = Instant::now();
            want = rt.decode(&mut caches, l, first, n_tok as u32, u32::MAX)?;
            rt.sync()?;
            if round > 0 {
                t4e.push(t0.elapsed().as_secs_f64());
            }
        }
        anyhow::ensure!(want.len() == n_tok, "{pname}: 4E made {} of {n_tok} tokens", want.len());
        let (g_med, g_lo, g_hi) = median_range(&t4e.iter().map(|t| n_tok as f64 / t).collect::<Vec<_>>());
        println!("\n== {pname}: prompt {l} tokens; 4E greedy {g_med:.1} tok/s [{g_lo:.1}-{g_hi:.1}] over {n_tok} tokens");
        // one 4E token and tree launches of R chained rows, for the oracle check
        let c_of = |rt: &mut llvq_llm::persist4e_cuda::Runtime4e, caches: &mut [llvq_llm::model::KvCache], r: usize| -> anyhow::Result<f64> {
            let nodes: Vec<u32> = std::iter::once(first).chain(want.iter().copied()).take(r).collect();
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
            Ok(median_range(&ratios).0)
        };
        for (dname, k) in &drafters {
            let mut acc_runs = Vec::new();
            let (mut tps, mut tps_nodraft, mut verify_ms, mut draft_ms, mut host_ms, mut passes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for round in 0..=ROUNDS {
                set(&mut caches, l)?;
                let mut d: Box<dyn Drafter> = match k {
                    None => Box::new(NullDrafter),
                    Some(k) => Box::new(OracleDrafter {
                        prompt_len: l,
                        truth: std::iter::once(first).chain(want.iter().copied()).collect(),
                        correct: *k,
                        vocab: model.config().vocab_size as u32,
                    }),
                };
                rt.sync()?;
                let o = {
                    let mut v = CardVerifier { rt: &mut *rt, caches: &mut caches, arm: 0, taps };
                    spec_run(&mut v, d.as_mut(), ids, first, &prompt_taps, hidden, n_tok, None).map_err(anyhow::Error::msg)?
                };
                anyhow::ensure!(o.tokens == want, "{pname} {dname}: the loop's tokens differ from 4E's greedy (first at {:?})", o.tokens.iter().zip(&want).position(|(a, b)| a != b));
                if round == 0 {
                    continue;
                }
                let tv: f64 = o.t_verify.iter().sum();
                let td: f64 = o.t_draft.iter().sum();
                tps.push(n_tok as f64 / o.t_total);
                tps_nodraft.push(n_tok as f64 / (o.t_total - td));
                verify_ms.push(tv * 1e3);
                draft_ms.push(td * 1e3);
                host_ms.push((o.t_total - tv - td) * 1e3);
                passes.push(o.accepted.len() as f64);
                acc_runs.push(o.accepted.iter().sum::<usize>() as f64 / o.accepted.len() as f64);
            }
            let m = |v: &Vec<f64>| median_range(v);
            let (a_m, ..) = m(&acc_runs);
            let (t_m, t_lo, t_hi) = m(&tps);
            let (n_m, n_lo, n_hi) = m(&tps_nodraft);
            println!(
                "  {dname:>9}: tokens equal to 4E's over {n_tok}; {a_m:.2} drafted tokens accepted a pass, {:.0} passes; verify {:.1} ms, drafter {:.2} ms, rest of the host {:.1} ms; {t_m:.1} tok/s [{t_lo:.1}-{t_hi:.1}], {n_m:.1} without the drafter [{n_lo:.1}-{n_hi:.1}]; x{:.3} of greedy",
                m(&passes).0,
                m(&verify_ms).0,
                m(&draft_ms).0,
                m(&host_ms).0,
                t_m / g_med
            );
            if let Some(k) = k {
                let r = k + 2;
                let c = c_of(&mut *rt, &mut caches, r)?;
                println!(
                    "  {dname:>9}: check, greedy x (K + 1) / c_{r} = {g_med:.1} x {} / {c:.3} = {:.1} tok/s, measured {t_m:.1} without the drafter {n_m:.1}",
                    k + 1,
                    g_med * (*k as f64 + 1.0) / c
                );
            }
            summary.push(format!("{pname} {dname}: {t_m:.1} tok/s, x{:.3} of 4E greedy {g_med:.1}", t_m / g_med));
        }
    }
    println!("\nsummary");
    for s in &summary {
        println!("  {s}");
    }
    Ok(())
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
