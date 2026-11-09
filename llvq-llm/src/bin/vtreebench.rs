//! Stage S1 of the tree verify (`docs/plan-verify-tree.md`): what a tree
//! verify pass of R rows costs beside 4E's one-token launch, on the card.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> vtreebench <sealed>`
//!
//! The served object loads as `specrun` loads it (4E v9 under its config),
//! then the tree pass installs beside it (`Runtime4e::install_tree`, its own
//! module). At two contexts, the chat and rag prompts of `specrun`:
//!
//! 1. 4E decodes 64 greedy tokens from the prompt.
//! 2. `vt1` against 4E: the tree pass of one row at each of those 64
//!    positions, its token the one 4E read there, the caches as 4E left
//!    them; its argmax against 4E's next token. 4E writes no logits on the
//!    card, so the argmax is the check (the RMS bound is the host's, S0).
//! 3. Identity: a tree of 32 rows, each node's parent node (i - 1) / 4 (the
//!    root, 4 children, 16, then 11: depths 0 to 3), tokens of the greedy
//!    text. Passes of its first R nodes, R = 1, 2, 4, 8, 16, 32, then 16
//!    and 32 nodes in a shuffled slot order. Every row's binary16 logits and
//!    argmax must equal its node's in the 32-row pass, byte for byte.
//! 4. Timing: 2 discarded and 5 timed rounds; each round runs one 4E token
//!    launch then `vt1`, `vt2`, `vt4`, `vt8`, `vt16`, `vt32` on the same
//!    tree's first R nodes, host wall from the launch to the ring read, as
//!    `specrun` times its arms. c_R = t(vt R) / t(4e), formed round by round,
//!    median and range.
//! 5. Phases: `vt1` and `vt16` stamped, 5 launches each, the CTAs' median
//!    event clock, each phase summed over the layers (`pers::p4t_phases`).
//!
//! Every launch's caches are the prompt's: the tree pass reads them and does
//! not write them.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        run()
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        anyhow::bail!("vtreebench: the tree verify is a cooperative CUDA launch; this build has no CUDA backend")
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const DISCARD: usize = 2;
#[cfg(all(target_os = "linux", feature = "cuda"))]
const TIMED: usize = 5;
#[cfg(all(target_os = "linux", feature = "cuda"))]
const ROWS: [usize; 6] = [1, 2, 4, 8, 16, 32];
#[cfg(all(target_os = "linux", feature = "cuda"))]
const N_REF: u32 = 64;

/// The 32-node tree: node i's parent (i - 1) / 4; returns (depth, path) of
/// `nodes` in that slot order, every path in slots.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn tree_rows(nodes: &[usize]) -> (Vec<u32>, Vec<[u8; 32]>) {
    let up = |i: usize| -> Vec<usize> {
        let mut p = vec![i];
        while *p.last().expect("non-empty") != 0 {
            let x = *p.last().expect("non-empty");
            p.push((x - 1) / 4);
        }
        p.reverse();
        p
    };
    let slot = |n: usize| nodes.iter().position(|&m| m == n).expect("an ancestor-closed set of nodes") as u8;
    let mut depth = Vec::with_capacity(nodes.len());
    let mut path = Vec::with_capacity(nodes.len());
    for &n in nodes {
        let p = up(n);
        depth.push((p.len() - 1) as u32);
        let mut row = [0u8; 32];
        for (d, &a) in p.iter().enumerate() {
            row[d] = slot(a);
        }
        path.push(row);
    }
    (depth, path)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run() -> anyhow::Result<()> {
    use llvq_cuda::pers::{p4e_phase_report, p4e_report_lines, p4t_phases, P4T_PHASES};
    use llvq_llm::boundary::median_range;
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: LLVQ_CONFIG=<served v9 config> vtreebench <sealed>";
    let path = a.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
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
        Some("vtreebench"),
    )?;
    cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut dec = cfg.decoder(&f.model, &device)?;
    for line in dec.lines() {
        println!("{line}");
    }
    let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("vtreebench: the config built no 4E runtime"))?;
    let before = rt.lines.len();
    rt.install_tree()?;
    for line in &rt.lines[before..] {
        println!("{line}");
    }
    println!("loaded in {:.1} s", t_load.elapsed().as_secs_f64());
    let tok = f.tokenizer.clone();
    let marks = Marks::resolve(&tok)?;
    let chat = |body: &str| -> anyhow::Result<Vec<u32>> {
        let mut ids = turn(&tok, &marks, "user", body)?;
        ids.extend(open_assistant(&tok, &marks, false)?);
        Ok(ids)
    };
    let model = &f.model;
    let (layers, grid) = rt.stamp_shape();
    let mut c16 = Vec::new();
    for (name, ids) in [("chat", chat(CHAT)?), ("rag", chat(&format!("{RAG}\n\n{SUMMARY_TEXT}"))?)] {
        let l = ids.len();
        let mut caches = model.fresh_caches();
        let first = model.prefill_next_in(&ids, llvq_llm::model::MAX_PREFILL_ROWS, &mut caches)?;
        model.device().synchronize()?;
        let want = rt.decode(&mut caches, l, first, N_REF, u32::MAX)?;
        anyhow::ensure!(want.len() == N_REF as usize, "{name}: 4E made {} of {N_REF} tokens", want.len());
        let set = |c: &mut [llvq_llm::model::KvCache], n: usize| -> anyhow::Result<()> {
            for x in c.iter_mut() {
                x.set_len(n)?;
            }
            Ok(())
        };
        println!("\n== tree, {name}: context {l}");
        // 2. vt1 against 4E along 4E's greedy text
        let mut same = 0usize;
        let mut first_miss = None;
        for i in 0..N_REF as usize {
            let t_in = if i == 0 { first } else { want[i - 1] };
            set(&mut caches, l + i)?;
            let out = rt.tree(&mut caches, l + i, &[t_in], &[0], &[[0u8; 32]], false, false)?;
            if out.ring[0] == want[i] {
                same += 1;
            } else if first_miss.is_none() {
                first_miss = Some((i, out.ring[0], want[i]));
            }
        }
        println!(
            "  vt1 against 4E: the same argmax at {same} of {N_REF} positions{}",
            first_miss.map_or(String::new(), |(i, a, b)| format!(" (first difference at position {}: {a} against 4E's {b})", l + i))
        );
        set(&mut caches, l)?;
        // 3. identity across R and slots
        let node_tok = |i: usize| if i == 0 { first } else { want[(7 * i) % N_REF as usize] };
        let full: Vec<usize> = (0..32).collect();
        let (d32, p32) = tree_rows(&full);
        let toks32: Vec<u32> = full.iter().map(|&n| node_tok(n)).collect();
        let base = rt.tree(&mut caches, l, &toks32, &d32, &p32, true, false)?;
        let v = base.logits.len() / 32;
        let mut shuffled16: Vec<usize> = (0..16).collect();
        shuffled16.reverse();
        let mut shuffled32: Vec<usize> = (0..32).map(|i| (i * 13 + 5) % 32).collect();
        shuffled32.sort_by_key(|&i| (i * 7 + 3) % 32);
        let mut sets: Vec<Vec<usize>> = ROWS.iter().map(|&r| (0..r).collect()).collect();
        sets.push(shuffled16);
        sets.push(shuffled32);
        let (mut rows_cmp, mut bad) = (0usize, Vec::new());
        for nodes in &sets {
            let (d, p) = tree_rows(nodes);
            let ts: Vec<u32> = nodes.iter().map(|&n| node_tok(n)).collect();
            let o = rt.tree(&mut caches, l, &ts, &d, &p, true, false)?;
            for (s, &n) in nodes.iter().enumerate() {
                rows_cmp += 1;
                if o.ring[s] != base.ring[n] || o.logits[s * v..(s + 1) * v] != base.logits[n * v..(n + 1) * v] {
                    bad.push((nodes.len(), s, n));
                }
            }
        }
        if bad.is_empty() {
            println!("  identity: every row equals its node in the 32-row pass, logits and argmax, {rows_cmp} rows of {} passes", sets.len());
        } else {
            println!("  identity FAILS: {} of {rows_cmp} rows differ, first (rows, slot, node) {:?}", bad.len(), bad[0]);
        }
        // 4. timing, rounds interleaved
        let mut one = Vec::new();
        let mut walls: Vec<Vec<f64>> = vec![Vec::new(); ROWS.len()];
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); ROWS.len()];
        let tree_of = |r: usize| {
            let nodes: Vec<usize> = (0..r).collect();
            let (d, p) = tree_rows(&nodes);
            let ts: Vec<u32> = nodes.iter().map(|&n| node_tok(n)).collect();
            (ts, d, p)
        };
        let trees: Vec<_> = ROWS.iter().map(|&r| tree_of(r)).collect();
        let mut moved = 0usize;
        for round in 0..DISCARD + TIMED {
            set(&mut caches, l)?;
            rt.sync()?;
            let t0 = Instant::now();
            let made = rt.decode(&mut caches, l, first, 1, u32::MAX)?;
            rt.sync()?;
            let w4 = t0.elapsed().as_secs_f64() * 1e6;
            anyhow::ensure!(made[..] == want[..1], "{name}: a one-token 4E launch moved its token");
            set(&mut caches, l)?;
            for (k, (ts, d, p)) in trees.iter().enumerate() {
                rt.sync()?;
                let t0 = Instant::now();
                let o = rt.tree(&mut caches, l, ts, d, p, false, false)?;
                rt.sync()?;
                let w = t0.elapsed().as_secs_f64() * 1e6;
                if o.ring != base.ring[..ts.len()] {
                    moved += 1;
                }
                if round >= DISCARD {
                    walls[k].push(w);
                    ratios[k].push(w / w4);
                }
            }
            if round >= DISCARD {
                one.push(w4);
            }
        }
        let (om, olo, ohi) = median_range(&one);
        println!("  4E, one token a launch: {om:.1} us [{olo:.1}-{ohi:.1}] (host wall, launch to ring read)");
        println!("  timed passes whose argmaxes differ from the 32-row pass's: {moved} of {}", (DISCARD + TIMED) * ROWS.len());
        println!("  rows   wall us, median [range]     c_R = t(vt R) / t(4e), round by round, median [range]");
        for (k, &r) in ROWS.iter().enumerate() {
            let (wm, wlo, whi) = median_range(&walls[k]);
            let (cm, clo, chi) = median_range(&ratios[k]);
            println!("  {r:4}   {wm:8.1} [{wlo:.1}-{whi:.1}]   c_{r} {cm:.3} [{clo:.3}-{chi:.3}]");
            if r == 16 {
                c16.push((name, cm));
            }
        }
        let (c1, _, _) = median_range(&ratios[0]);
        let (c8, _, _) = median_range(&ratios[3]);
        let (c16m, _, _) = median_range(&ratios[4]);
        println!(
            "  an extra row: (c_16 - c_1) / 15 = {:.4}, (c_8 - c_1) / 7 = {:.4} of a 4E token; c_2 {:.3}",
            (c16m - c1) / 15.0,
            (c8 - c1) / 7.0,
            median_range(&ratios[1]).0
        );
        // 5. phases
        if rt.can_stamp() {
            set(&mut caches, l)?;
            let (made, stamps) = rt.decode_stamped(&mut caches, l, first, 1, u32::MAX)?;
            anyhow::ensure!(made[..] == want[..1], "{name}: the stamped 4E launch moved its token");
            let r = p4e_phase_report(&stamps, 1, layers, grid).map_err(anyhow::Error::msg)?;
            println!("  4E stamped, one token:");
            for line in p4e_report_lines(&r) {
                println!("{line}");
            }
        }
        println!("  tree, each phase summed over the layers, us, median over {TIMED} launches (event clocks, the CTAs' median):");
        println!("    rows{}", P4T_PHASES.iter().map(|p| format!("{p:>8}")).collect::<String>());
        for r in [1usize, 16] {
            let (ts, d, p) = tree_of(r);
            let mut per = vec![Vec::new(); P4T_PHASES.len()];
            for _ in 0..TIMED {
                set(&mut caches, l)?;
                let o = rt.tree(&mut caches, l, &ts, &d, &p, false, true)?;
                let ph = p4t_phases(&o.stamps, layers, grid).map_err(anyhow::Error::msg)?;
                for (acc, x) in per.iter_mut().zip(ph) {
                    acc.push(x / 1e3);
                }
            }
            println!("    {r:4}{}", per.iter().map(|v| format!("{:8.1}", median_range(v).0)).collect::<String>());
        }
        set(&mut caches, l)?;
    }
    let worst = c16.iter().map(|c| c.1).fold(f64::MIN, f64::max);
    println!("\nc_16: {}; the decision reads the larger, {worst:.3}", c16.iter().map(|(n, c)| format!("{n} {c:.3}")).collect::<Vec<_>>().join(", "));
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const CHAT: &str = "I am planning a three-day trip to Lisbon in late autumn with my partner. We like walking, \
small museums and local food, and we do not want to rent a car. Could you suggest a plan for each day and a few \
practical tips?";
// specrun's (2026-11-07), not quoted from any source.
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
