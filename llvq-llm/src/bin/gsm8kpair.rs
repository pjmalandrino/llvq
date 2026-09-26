//! Two GSM8K dumps, paired problem by problem.
//!
//! ```text
//! cargo run --release -p llvq-llm --bin gsm8kpair -- <a.jsonl> <b.jsonl>
//! ```
//!
//! Every row of both dumps is graded again by `llvq_llm::gsm8k::regrade` before
//! pairing, so one grader scores every arm, whichever engine generated it.
//!
//! Refuses two dumps that do not hold the same problems in the same prompt
//! tokens, and a dump without its trailer. Prints both accuracies, the
//! discordant counts, the paired difference A − B with its 95 % interval, and
//! McNemar's exact p. The interval carries no finite-population correction: a
//! census of the test split is read as a draw from the problems a model could
//! be asked, the reading every census interval of this project takes.

use llvq_llm::gsm8k::{self, Dump};

/// Read a dump and grade every row again with the current rules. The count of
/// rows the re-grade moved is printed: zero on a dump `bin/gsm8k` wrote under
/// these rules, every row on one `ops/gsm8k_vllm.py` wrote ungraded.
fn read(path: &str) -> anyhow::Result<Dump> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
    let mut d = gsm8k::parse_dump(&text, path)?;
    let mut moved = 0usize;
    for r in &mut d.rows {
        moved += usize::from(gsm8k::regrade(r)?);
    }
    println!("{path}: {moved} of {} rows re-graded", d.rows.len());
    Ok(d)
}

fn describe(tag: &str, d: &Dump) {
    let gens: Vec<usize> = d.rows.iter().map(|r| r.n_gen).collect();
    let capped = d.rows.iter().filter(|r| r.stop == "cap").count();
    let right = d.rows.iter().filter(|r| r.correct).count();
    println!(
        "{tag} = {} | {} | device {} | dtype {} | kv {} | max_new {} | reasoning {}",
        d.field("model"),
        d.field("arithmetic"),
        d.field("device"),
        d.field("dtype"),
        d.field("kv"),
        d.field("max_new"),
        if d.header.get("think").and_then(|v| v.as_bool()) == Some(true) {
            "on"
        } else {
            "off"
        }
    );
    println!(
        "    {right}/{} right, mean {:.1} generated tokens, {capped} stopped at the cap, tokens {}",
        d.rows.len(),
        gens.iter().sum::<usize>() as f64 / gens.len().max(1) as f64,
        d.fingerprint
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 2,
        "usage: gsm8kpair <a.jsonl> <b.jsonl>"
    );
    let (a, b) = (read(&args[0])?, read(&args[1])?);
    anyhow::ensure!(
        a.fingerprint == b.fingerprint,
        "run fingerprints differ ({} against {}): the two arms were not asked the same prompt \
         stream",
        a.fingerprint,
        b.fingerprint
    );
    let p = gsm8k::pair(&a, &b)?;
    describe("A", &a);
    describe("B", &b);
    let pct = |x: usize| 100.0 * x as f64 / p.n.max(1) as f64;
    println!("\n{} problems, same prompts problem by problem", p.n);
    println!("  accuracy A = {:.2} %, B = {:.2} %", pct(p.right_a), pct(p.right_b));
    println!(
        "  right in A only {}, in B only {} ({:.1} % discordant)",
        p.only_a,
        p.only_b,
        pct(p.only_a + p.only_b)
    );
    println!(
        "  A − B = {:+.2} pp, 95 % [{:+.2} ; {:+.2}], McNemar exact p = {:.3e}",
        100.0 * p.delta.0,
        100.0 * p.delta.1,
        100.0 * p.delta.2,
        p.p_mcnemar
    );
    Ok(())
}
