//! Stages S1, N1 and N2 of the tree verify (`docs/plan-verify-tree.md`):
//! what a tree verify pass of R rows costs beside 4E's one-token launch, on
//! the card, for each arm of the pass.
//!
//! Usage: `LLVQ_CONFIG=<served v9 config> vtreebench <sealed> [arms]`, arms a
//! comma-separated subset of `a0,a1,a2,a3,a4` (default `a0,a1,a2`):
//!
//! * a0: S1's kernel and S1's slices (`pers::p4t_splits`);
//! * a1: S1's kernel, the balanced slices q/k/v 11, o 14, gate/up 7, down
//!   14, head 1 (`pers::p4t_splits_balanced` keeps them where q/k/v and
//!   gate/up fill 90 % of the grid);
//! * a2: the tiled kernel on tree-tiled copies (`P4T_TILED`), at a1's slices,
//!   or at a3's when a3 is asked too, so that a2 against a3 isolates the
//!   unit's control flow;
//! * a3: the stream unit (`P4T_STREAM`, `pers_4et_stream.cu`) on the same
//!   copies, at the stream plan (`pers::p4t_splits_stream`). a2 is its guard:
//!   a3's 32-row pass must equal a2's bit for bit, else a3's numbers are
//!   void;
//! * a4: a3 with the slice fold's loads batched and the prologues on 16-byte
//!   vectors (`P4T_A4`, `pers_4et_a4.cu`), at the stream plan; a3 is its
//!   guard, bit for bit.
//!
//! An arm whose module does not build is void and skipped; the others run.
//! An unknown arm is refused before anything loads.
//!
//! The served object loads as `specrun` loads it (4E v9 under its config),
//! then the tree pass installs beside it (`Runtime4e::install_tree`, its own
//! modules). At the chat context for every arm, then at the rag context for
//! the arm with the lowest chat c_16:
//!
//! 1. 4E decodes 64 greedy tokens from the prompt.
//! 2. For each arm, `vt1` against 4E: the tree pass of one row at each of
//!    those 64 positions, its token the one 4E read there; its argmax
//!    against 4E's next token.
//! 3. For each arm, identity: a tree of 32 rows (node i's parent (i - 1) /
//!    4), passes of its first R nodes, R = 1, 2, 8, 16, then 16 and 32 nodes
//!    in shuffled slot orders: every row's binary16 logits and argmax equal
//!    its node's in the arm's 32-row pass. Across arms: a2's 32-row pass
//!    equals a1's bit for bit (same slices, same arithmetic); a0's argmaxes
//!    against a1's (another fold).
//! 4. Timing: 2 discarded and 5 timed rounds; each round one 4E token
//!    launch, then every arm's `vt1`, `vt2`, `vt8`, `vt16`, host wall from
//!    the launch to the ring read. c_R = t(vt R) / t(4e) round by round,
//!    median and range.
//! 5. Phases: 4E stamped over two tokens; each arm's `vt1` and `vt16`
//!    stamped, 5 launches each: per phase the wall (the CTAs' median clocks)
//!    and the spread of a CTA's busy time across the CTAs (min, median,
//!    max), each summed over the layers (`pers::p4t_phases`).
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
const ROWS: [usize; 4] = [1, 2, 8, 16];
#[cfg(all(target_os = "linux", feature = "cuda"))]
const N_REF: u32 = 64;
/// a1's and a2's proposed slices at the 4B (job N1's prereg).
#[cfg(all(target_os = "linux", feature = "cuda"))]
const BALANCED: [u32; 5] = [11, 14, 7, 14, 1];

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
    use llvq_cuda::pers::{p4e_phase_report, p4e_report_lines, p4t_parts, p4t_phases, P4T_PHASES, P4T_ROW_PHASES};
    use llvq_llm::boundary::median_range;
    use llvq_llm::chatfmt::{open_assistant, turn, Marks};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: LLVQ_CONFIG=<served v9 config> vtreebench <sealed> [a0,a1,a2,a3,a4]";
    let path = a.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?.clone();
    let want_arms: Vec<String> = a.get(1).map_or("a0,a1,a2", |s| s.as_str()).split(',').map(str::to_string).collect();
    use llvq_llm::persist4e_cuda::{TreeForm, TreePlan};
    let with_a3 = want_arms.iter().any(|w| w == "a3" || w == "a4");
    let mut specs: Vec<llvq_llm::persist4e_cuda::TreeArmSpec> = Vec::new();
    for w in &want_arms {
        if specs.iter().any(|s| s.0 == w.as_str()) {
            anyhow::bail!("vtreebench: arm {w} asked twice; {usage}");
        }
        specs.push(match w.as_str() {
            "a0" => ("a0", TreePlan::S1, TreeForm::Rows),
            "a1" => ("a1", TreePlan::Balanced(BALANCED), TreeForm::Rows),
            "a2" if with_a3 => ("a2", TreePlan::Stream, TreeForm::Tiled),
            "a2" => ("a2", TreePlan::Balanced(BALANCED), TreeForm::Tiled),
            "a3" => ("a3", TreePlan::Stream, TreeForm::Stream),
            "a4" => ("a4", TreePlan::Stream, TreeForm::A4),
            other => anyhow::bail!("vtreebench: unknown arm {other:?}; {usage}"),
        });
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
        Some("vtreebench"),
    )?;
    cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut dec = cfg.decoder(&f.model, &device)?;
    for line in dec.lines() {
        println!("{line}");
    }
    let rt = dec.runtime_mut().ok_or_else(|| anyhow::anyhow!("vtreebench: the config built no 4E runtime"))?;
    let before = rt.lines.len();
    rt.install_tree(&specs)?;
    for line in &rt.lines[before..] {
        println!("{line}");
    }
    let arms = rt.tree_arms();
    let na = arms.len();
    for a in &arms {
        println!(
            "arm {}: {:?} kernel, {:?} plan, slices {:?}{}",
            a.name,
            a.form,
            a.plan,
            a.splits,
            a.void.as_ref().map_or(String::new(), |v| format!(", VOID: {v}"))
        );
    }
    anyhow::ensure!(arms.iter().any(|a| a.void.is_none()), "vtreebench: every arm is void");
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
    let mut c16_chat: Vec<f64> = vec![f64::MAX; na];
    let mut summary: Vec<String> = Vec::new();
    for (name, ids) in [("chat", chat(CHAT)?), ("rag", chat(&format!("{RAG}\n\n{SUMMARY_TEXT}"))?)] {
        // chat: every arm; rag: the arm with the lowest chat c_16 only
        let live: Vec<usize> = (0..na).filter(|&k| arms[k].void.is_none()).collect();
        let run_arms: Vec<usize> = if name == "chat" {
            live
        } else {
            let best = live.iter().copied().min_by(|&x, &y| c16_chat[x].total_cmp(&c16_chat[y])).expect("an arm");
            vec![best]
        };
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
        println!(
            "\n== tree, {name}: context {l}, arms {}",
            run_arms.iter().map(|&k| arms[k].name.as_str()).collect::<Vec<_>>().join(", ")
        );
        let node_tok = |i: usize| if i == 0 { first } else { want[(7 * i) % N_REF as usize] };
        let full: Vec<usize> = (0..32).collect();
        let (d32, p32) = tree_rows(&full);
        let toks32: Vec<u32> = full.iter().map(|&n| node_tok(n)).collect();
        let mut bases: Vec<Option<llvq_llm::persist4e_cuda::TreeOut>> = (0..na).map(|_| None).collect();
        for &k in &run_arms {
            let an = &arms[k].name;
            // 2. vt1 against 4E along 4E's greedy text
            let mut same = 0usize;
            let mut first_miss = None;
            for i in 0..N_REF as usize {
                let t_in = if i == 0 { first } else { want[i - 1] };
                set(&mut caches, l + i)?;
                let out = rt.tree(k, &mut caches, l + i, &[t_in], &[0], &[[0u8; 32]], false, false)?;
                if out.ring[0] == want[i] {
                    same += 1;
                } else if first_miss.is_none() {
                    first_miss = Some((i, out.ring[0], want[i]));
                }
            }
            println!(
                "  {an} vt1 against 4E: the same argmax at {same} of {N_REF} positions{}",
                first_miss.map_or(String::new(), |(i, a, b)| format!(" (first difference at position {}: {a} against 4E's {b})", l + i))
            );
            set(&mut caches, l)?;
            // 3. identity across R and slots
            let base = rt.tree(k, &mut caches, l, &toks32, &d32, &p32, true, false)?;
            let v = base.logits.len() / 32;
            let shuffled16: Vec<usize> = (0..16).rev().collect();
            let mut shuffled32: Vec<usize> = (0..32).map(|i| (i * 13 + 5) % 32).collect();
            shuffled32.sort_by_key(|&i| (i * 7 + 3) % 32);
            let mut sets: Vec<Vec<usize>> = ROWS.iter().map(|&r| (0..r).collect()).collect();
            sets.push(shuffled16);
            sets.push(shuffled32);
            let (mut rows_cmp, mut bad) = (0usize, Vec::new());
            for nodes in &sets {
                let (d, p) = tree_rows(nodes);
                let ts: Vec<u32> = nodes.iter().map(|&n| node_tok(n)).collect();
                let o = rt.tree(k, &mut caches, l, &ts, &d, &p, true, false)?;
                for (s, &n) in nodes.iter().enumerate() {
                    rows_cmp += 1;
                    if o.ring[s] != base.ring[n] || o.logits[s * v..(s + 1) * v] != base.logits[n * v..(n + 1) * v] {
                        bad.push((nodes.len(), s, n));
                    }
                }
            }
            if bad.is_empty() {
                println!("  {an} identity: every row equals its node in the 32-row pass, logits and argmax, {rows_cmp} rows of {} passes", sets.len());
            } else {
                println!("  {an} identity FAILS: {} of {rows_cmp} rows differ, first (rows, slot, node) {:?}", bad.len(), bad[0]);
            }
            bases[k] = Some(base);
        }
        // across arms: same slices and arithmetic owe each other their bits
        for x in 0..na {
            for y in x + 1..na {
                if let (Some(bx), Some(by)) = (&bases[x], &bases[y]) {
                    let same_tok = bx.ring.iter().zip(&by.ring).filter(|(p, q)| p == q).count();
                    if arms[x].splits == arms[y].splits {
                        let ok = bx.ring == by.ring && bx.logits == by.logits;
                        println!(
                            "  {} against {} (same slices): the 32-row pass {} bit for bit, argmax at {same_tok} of 32 rows",
                            arms[y].name,
                            arms[x].name,
                            if ok { "equal" } else { "DIFFERS" }
                        );
                    } else {
                        println!("  {} against {} (other slices): the same argmax at {same_tok} of 32 rows", arms[y].name, arms[x].name);
                    }
                }
            }
        }
        // the guards: a3 against a2, a4 against a3, at the same plan, bit for bit
        let at = |n: &str| arms.iter().position(|a| a.name == n);
        for (arm, guard) in [("a3", "a2"), ("a4", "a3")] {
            if let (Some(ig), Some(ia)) = (at(guard), at(arm)) {
                if let (Some(bg), Some(ba)) = (&bases[ig], &bases[ia]) {
                    let ok = arms[ig].splits == arms[ia].splits && bg.ring == ba.ring && bg.logits == ba.logits;
                    println!(
                        "  {arm} guard against {guard} (slices {:?} and {:?}): {}",
                        arms[ig].splits,
                        arms[ia].splits,
                        if ok { "PASSES, the 32-row pass equal bit for bit".to_string() } else { format!("FAILS: {arm}'s numbers are void") }
                    );
                } else if name == "chat" {
                    println!("  {arm} guard against {guard}: not run (an arm is void), {arm}'s numbers are void");
                }
            }
        }
        // 4. timing, rounds interleaved
        let mut one = Vec::new();
        let mut ratios: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); ROWS.len()]; na];
        let mut walls: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); ROWS.len()]; na];
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
            for &k in &run_arms {
                let base = bases[k].as_ref().expect("identity ran");
                for (j, (ts, d, p)) in trees.iter().enumerate() {
                    rt.sync()?;
                    let t0 = Instant::now();
                    let o = rt.tree(k, &mut caches, l, ts, d, p, false, false)?;
                    rt.sync()?;
                    let w = t0.elapsed().as_secs_f64() * 1e6;
                    if o.ring != base.ring[..ts.len()] {
                        moved += 1;
                    }
                    if round >= DISCARD {
                        walls[k][j].push(w);
                        ratios[k][j].push(w / w4);
                    }
                }
            }
            if round >= DISCARD {
                one.push(w4);
            }
        }
        let (om, olo, ohi) = median_range(&one);
        println!("  4E, one token a launch: {om:.1} us [{olo:.1}-{ohi:.1}] (host wall, launch to ring read)");
        println!("  timed passes whose argmaxes differ from their arm's 32-row pass: {moved} of {}", (DISCARD + TIMED) * ROWS.len() * run_arms.len());
        for &k in &run_arms {
            let an = &arms[k].name;
            println!("  {an}  rows   wall us, median [range]     c_R = t(vt R) / t(4e), round by round, median [range]");
            for (j, &r) in ROWS.iter().enumerate() {
                let (wm, wlo, whi) = median_range(&walls[k][j]);
                let (cm, clo, chi) = median_range(&ratios[k][j]);
                println!("  {an}  {r:4}   {wm:8.1} [{wlo:.1}-{whi:.1}]   c_{r} {cm:.3} [{clo:.3}-{chi:.3}]");
            }
            let c = |j: usize| median_range(&ratios[k][j]).0;
            println!(
                "  {an} an extra row: (c_16 - c_1) / 15 = {:.4}, (c_8 - c_1) / 7 = {:.4} of a 4E token; c_2 {:.3}",
                (c(3) - c(0)) / 15.0,
                (c(2) - c(0)) / 7.0,
                c(1)
            );
            if name == "chat" {
                c16_chat[k] = c(3);
            }
            summary.push(format!("{name} {an}: c_1 {:.3}, c_2 {:.3}, c_8 {:.3}, c_16 {:.3}", c(0), c(1), c(2), c(3)));
        }
        // 5. phases
        if rt.can_stamp() {
            set(&mut caches, l)?;
            let (made, stamps) = rt.decode_stamped(&mut caches, l, first, 2, u32::MAX)?;
            anyhow::ensure!(made[..] == want[..2], "{name}: the stamped 4E launch moved its tokens");
            let r = p4e_phase_report(&stamps, 2, layers, grid).map_err(anyhow::Error::msg)?;
            println!("  4E stamped, two tokens:");
            for line in p4e_report_lines(&r) {
                println!("{line}");
            }
        }
        for &k in &run_arms {
            let an = &arms[k].name;
            for r in [1usize, 16] {
                let (ts, d, p) = tree_of(r);
                let mut per = vec![[Vec::new(), Vec::new(), Vec::new(), Vec::new()]; P4T_PHASES.len()];
                let mut parts = vec![[Vec::new(), Vec::new(), Vec::new(), Vec::new()]; P4T_ROW_PHASES.len()];
                let n_ev = (1 + 2 * llvq_cuda::pers::n_bar_4et(layers, 0)) as usize * grid as usize;
                for _ in 0..TIMED {
                    set(&mut caches, l)?;
                    let o = rt.tree(k, &mut caches, l, &ts, &d, &p, false, true)?;
                    anyhow::ensure!(o.stamps.len() >= n_ev, "{an}: {} stamps, the phases need {n_ev}", o.stamps.len());
                    let ph = p4t_phases(&o.stamps[..n_ev], layers, grid).map_err(anyhow::Error::msg)?;
                    if o.stamps.len() > n_ev {
                        // an a4 launch's F3 parts, each CTA's sum over the layers, median over the CTAs
                        let pp = p4t_parts(&o.stamps[n_ev..], grid).map_err(anyhow::Error::msg)?;
                        for (acc, x) in parts.iter_mut().zip(pp) {
                            for (a, v) in acc.iter_mut().zip(x) {
                                a.push(v / 1e3);
                            }
                        }
                    }
                    for (acc, x) in per.iter_mut().zip(ph) {
                        for (a, v) in acc.iter_mut().zip(x) {
                            a.push(v / 1e3);
                        }
                    }
                }
                let m = |v: &Vec<f64>| median_range(v).0;
                let total: f64 = per.iter().map(|x| m(&x[0])).sum();
                let rows_ph: f64 = ["qkv", "o", "gateup", "down", "head"]
                    .iter()
                    .map(|n| m(&per[P4T_PHASES.iter().position(|p| p == n).expect("a phase")][0]))
                    .sum();
                println!(
                    "  {an} vt{r} phases, us summed over the layers, median of {TIMED} stamped launches: wall {total:.1}, row phases {rows_ph:.1} ({:.1} %)",
                    100.0 * rows_ph / total
                );
                println!("  {an} vt{r}   phase      wall   busy min   busy med   busy max");
                for (i, ph) in P4T_PHASES.iter().enumerate() {
                    println!(
                        "  {an} vt{r}   {ph:>6} {:9.1} {:10.1} {:10.1} {:10.1}",
                        m(&per[i][0]),
                        m(&per[i][1]),
                        m(&per[i][2]),
                        m(&per[i][3])
                    );
                }
                if !parts[0][0].is_empty() {
                    println!("  {an} vt{r} parts of the row phases, us summed over the layers, median over the CTAs, median of {TIMED} launches");
                    println!("  {an} vt{r}   phase      setup    staging      loops      store");
                    for (i, ph) in P4T_ROW_PHASES.iter().enumerate() {
                        println!(
                            "  {an} vt{r}   {ph:>6} {:10.1} {:10.1} {:10.1} {:10.1}",
                            m(&parts[i][0]),
                            m(&parts[i][1]),
                            m(&parts[i][2]),
                            m(&parts[i][3])
                        );
                    }
                }
            }
        }
        set(&mut caches, l)?;
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
