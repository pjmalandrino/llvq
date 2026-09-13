//! L36's consumer: the statistics a captured Hessian answers on its own.
//!
//! `nice -n 10 cargo run --release -p llvq-bench --example hstats -- <hcapture-dir>`
//!
//! Reads the dump `llvq-llm --bin hcapture` writes — `meta.csv` plus flat
//! little-endian `diag-*.f64`, `mean-*.f64` and `norms-*.f32` — and prints the
//! three statistics that need **nothing but the dump**. It measures and prints;
//! it decides nothing, and it moves no row of any table by itself.
//!
//! ## What it answers, and what it does not
//!
//! Of the eight rows L36 was built for, this file answers **three**:
//!
//! | row | statistic | section |
//! |---|---|---|
//! | L05 | share of `tr(H)` carried by the first four token positions, and by the top 1% of tokens by norm | 2 |
//! | L14 | spread of `diag(QᵀHQ)` within each block of 24, against the sweep order | 3 |
//! | L19 | the same diagonal, read on the tail columns the `KeepExact` policy leaves | 3 |
//!
//! The other five — L02, L23, L24, L25, L27 — need `ΔW = W − Ŵ`, that is the
//! **checkpoint** beside the artifact. `llvq-bench` depends on the four
//! workspace crates and nothing else (`Cargo.toml`), and the checkpoint reader
//! lives in `llvq-llm` behind `hf-hub`. Those five belong in an `llvq-llm`
//! example and are not attempted here: a section that silently answered half
//! its row would be worse than an absent one.
//!
//! Two more rows the survey attributed to L36 are not answered by any
//! statistic on a Hessian: **L06** asks for a paired re-encode and **L15** for
//! kurtosis on `f1recdump` blocks. See
//! `docs/mesures/l36-capture-2026-09-13.txt`.
//!
//! ## The basis matters and is checked
//!
//! `H` and `QᵀHQ` have the same shape, the same symmetry and the same trace.
//! Section 3 is meaningful only in the **rotated** basis — it asks about the
//! blocks of 24 the encoder quantizes — so it refuses a natural-basis dump by
//! name rather than printing a plausible number about the wrong matrix.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One row of `meta.csv`: an emission, and where to find its vectors.
#[derive(Debug, Clone)]
struct Emission {
    block: usize,
    act: String,
    n: usize,
    basis: String,
    rotation_seed: Option<u64>,
    trace: f64,
    offdiag: f64,
    ntokens: usize,
}

impl Emission {
    fn tag(&self) -> String {
        format!("{}-{}-{}", self.block, self.act, self.basis)
    }
}

fn read_f64(p: &Path) -> Result<Vec<f64>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if b.len() % 8 != 0 {
        return Err(format!("{}: {} bytes is not a whole f64 count", p.display(), b.len()));
    }
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect())
}

fn read_f32(p: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if b.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole f32 count", p.display(), b.len()));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect())
}

fn parse_meta(dir: &Path) -> Result<(String, Vec<Emission>), String> {
    let path = dir.join("meta.csv");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\nThis example reads an `hcapture` dump. Produce one with:\n  \
             LLVQ_CALIB=c4 cargo run --release -p llvq-llm --features metal \\\n    \
             --bin hcapture -- <sealed.bin> <n_calib> <calib_len> metal <dir>",
            path.display()
        )
    })?;
    let mut header = String::new();
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix("# ") {
            header = h.to_string();
            continue;
        }
        if line.starts_with("block,") || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 10 {
            return Err(format!("meta.csv: {} columns in {line:?}, expected 10", c.len()));
        }
        let num = |i: usize, what: &str| -> Result<f64, String> {
            c[i].parse::<f64>()
                .map_err(|e| format!("meta.csv: {what} {:?}: {e}", c[i]))
        };
        out.push(Emission {
            block: num(0, "block")? as usize,
            act: c[1].to_string(),
            n: num(2, "n")? as usize,
            basis: c[3].to_string(),
            rotation_seed: if c[4].is_empty() {
                None
            } else {
                Some(c[4].parse().map_err(|e| format!("meta.csv: seed: {e}"))?)
            },
            trace: num(5, "trace")?,
            offdiag: num(6, "offdiag")?,
            ntokens: num(8, "ntokens")? as usize,
        });
    }
    if out.is_empty() {
        return Err(format!("{}: no emission rows", path.display()));
    }
    Ok((header, out))
}

/// Median, and the quantile a caller names, of an already-sortable sample.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

// ---------------------------------------------------------------------------
// Section 1 — provenance
// ---------------------------------------------------------------------------

fn section_provenance(header: &str, em: &[Emission]) {
    println!("SECTION 1 — PROVENANCE");
    println!("{}", "-".repeat(74));
    println!("  {header}");
    let blocks: std::collections::BTreeSet<usize> = em.iter().map(|e| e.block).collect();
    let mut widths: BTreeMap<&str, usize> = BTreeMap::new();
    let mut bases: BTreeMap<&str, usize> = BTreeMap::new();
    for e in em {
        widths.insert(e.act.as_str(), e.n);
        *bases.entry(e.basis.as_str()).or_insert(0) += 1;
    }
    println!(
        "  {} emissions · {} blocks · {} activations · {} tokens per activation",
        em.len(),
        blocks.len(),
        widths.len(),
        em[0].ntokens
    );
    print!("  widths:");
    for (a, n) in &widths {
        print!(" {a}={n}");
    }
    println!();
    print!("  bases: ");
    for (b, k) in &bases {
        print!(" {b}×{k}");
    }
    println!();
    // Every block must carry every activation, or a later per-block statistic
    // silently averages over a different set of matrices per block.
    let expect = blocks.len() * widths.len();
    if em.len() != expect {
        println!(
            "  ⚠️  {} emissions for {} blocks × {} activations = {expect}: the dump is incomplete",
            em.len(),
            blocks.len(),
            widths.len()
        );
    }
    let seeds: std::collections::BTreeSet<Option<u64>> =
        em.iter().map(|e| e.rotation_seed).collect();
    println!(
        "  {} distinct rotation seeds over {} emissions{}",
        seeds.len(),
        em.len(),
        if seeds.len() == 1 && seeds.contains(&None) {
            " (unrotated dump)"
        } else {
            ""
        }
    );
    println!();
}

// ---------------------------------------------------------------------------
// Section 2 — L05: where the trace comes from
// ---------------------------------------------------------------------------

/// `tr(H) = Σₜ ‖xₜ‖² / N`, so a share of the trace is a share of the token
/// norms. The row asks two questions of it: how much the first four positions
/// carry — the attention-sink hypothesis — and how much the heaviest 1% of
/// tokens carry, which is what a row filter in `Hessian::accumulate` would cut.
fn section_trace(dir: &Path, em: &[Emission]) -> Result<(), String> {
    println!("SECTION 2 — L05: WHAT CARRIES tr(H)");
    println!("{}", "-".repeat(74));
    println!("  The sink-mask decision rests on the first column. `hcapture` writes one");
    println!("  ‖x‖² per calibration row, so these are shares of the trace itself, not");
    println!("  of a proxy.\n");
    println!(
        "  {:<6} {:<8} {:>10} {:>10} {:>10} {:>12}",
        "block", "act", "pos 0..3", "top 1%", "top 0.1%", "tr(H)"
    );

    // Per activation, the spread over blocks is what bounds the decision; a
    // pooled mean over 36 blocks would hide a sink that lives in block 0 only.
    let mut per_act: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut shown = 0usize;
    for e in em {
        let p = dir.join(format!("norms-{}.f32", e.tag()));
        if !p.exists() {
            continue;
        }
        let v = read_f32(&p)?;
        if v.is_empty() {
            continue;
        }
        let total: f64 = v.iter().map(|x| *x as f64).sum();
        if total <= 0.0 {
            continue;
        }
        let head: f64 = v.iter().take(4).map(|x| *x as f64).sum();
        let mut s: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        s.sort_by(|a, b| b.partial_cmp(a).expect("no NaN in a squared norm"));
        let take = |frac: f64| -> f64 {
            let k = ((v.len() as f64 * frac).ceil() as usize).max(1);
            s.iter().take(k).sum::<f64>() / total
        };
        let (p4, t1, t01) = (head / total, take(0.01), take(0.001));
        per_act
            .entry(e.act.clone())
            .or_default()
            .push((p4, t1));
        if shown < 8 {
            println!(
                "  {:<6} {:<8} {:>9.2}% {:>9.2}% {:>9.2}% {:>12.4e}",
                e.block,
                e.act,
                100.0 * p4,
                100.0 * t1,
                100.0 * t01,
                e.trace
            );
            shown += 1;
        }
    }
    if per_act.is_empty() {
        println!("  no `norms-*.f32` in the dump: this section needs `Hessian::with_moments`");
        println!();
        return Ok(());
    }
    println!("  … (first 8 emissions shown)\n");
    println!("  Over all blocks, per activation — median [min; max], never a pooled mean:");
    println!(
        "  {:<10} {:>26} {:>26}",
        "act", "positions 0..3", "top 1% of tokens"
    );
    for (act, v) in &per_act {
        let mut a: Vec<f64> = v.iter().map(|x| x.0).collect();
        let mut b: Vec<f64> = v.iter().map(|x| x.1).collect();
        a.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
        b.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
        println!(
            "  {:<10} {:>10.2}% [{:5.2}; {:5.2}] {:>10.2}% [{:5.2}; {:5.2}]",
            act,
            100.0 * quantile(&a, 0.5),
            100.0 * a[0],
            100.0 * a[a.len() - 1],
            100.0 * quantile(&b, 0.5),
            100.0 * b[0],
            100.0 * b[b.len() - 1],
        );
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Section 3 — L14 and L19: the rotated diagonal, per block of 24
// ---------------------------------------------------------------------------

/// What one activation's diagonal accumulates over the blocks.
#[derive(Default)]
struct DiagStats {
    /// max/min within each block of 24.
    ratios: Vec<f64>,
    /// Coefficient of variation within each block of 24.
    cvs: Vec<f64>,
    /// Mean of the tail columns against the matrix's median column.
    tails: Vec<f64>,
    /// Whole blocks of 24, and the remainder the `KeepExact` tail holds.
    full: usize,
    tail: usize,
}

/// The encoder sweeps blocks of 24 columns left to right and never justified
/// that order by a measurement (L14); the `KeepExact` tail is the remainder
/// modulo 24 of the **rotated** basis, and whether any salience survives the
/// rotation there is L19. Both read one column: `diag(QᵀHQ)`.
fn section_diagonal(dir: &Path, em: &[Emission]) -> Result<(), String> {
    println!("SECTION 3 — L14 AND L19: diag(QᵀHQ) PER BLOCK OF 24");
    println!("{}", "-".repeat(74));

    let rotated: Vec<&Emission> = em.iter().filter(|e| e.basis == "rotated").collect();
    if rotated.is_empty() {
        println!("  REFUSED: this dump carries no rotated emission.");
        println!("  Both rows ask about the blocks of 24 the encoder quantizes, which exist");
        println!("  only in the rotated basis. A natural-basis diagonal would print a");
        println!("  plausible number about a different matrix. Re-run `hcapture` without");
        println!("  LLVQ_ROT=norot.\n");
        return Ok(());
    }
    println!("  Within each block of 24 columns: the ratio max/min of the diagonal, and");
    println!("  its coefficient of variation. A flat diagonal is a sweep order that cannot");
    println!("  matter (L14) and a tail with no salience to re-qualify (L19).\n");
    println!(
        "  {:<10} {:>7} {:>8} {:>10} {:>10} {:>10} {:>9}",
        "act", "blocks", "tail", "med max/min", "p90", "med CV", "tail/med"
    );

    let mut per_act: BTreeMap<String, DiagStats> = BTreeMap::new();
    for e in &rotated {
        let p = dir.join(format!("diag-{}.f64", e.tag()));
        if !p.exists() {
            continue;
        }
        let d = read_f64(&p)?;
        if d.len() != e.n {
            return Err(format!(
                "{}: {} values for a width of {}",
                p.display(),
                d.len(),
                e.n
            ));
        }
        let full = e.n / 24;
        let tail = e.n % 24;
        let entry = per_act.entry(e.act.clone()).or_insert_with(|| DiagStats {
            full,
            tail,
            ..DiagStats::default()
        });
        for b in 0..full {
            let g = &d[b * 24..(b + 1) * 24];
            let mn = g.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = g.iter().sum::<f64>() / 24.0;
            if mn > 0.0 && mean > 0.0 {
                entry.ratios.push(mx / mn);
                let var = g.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 24.0;
                entry.cvs.push(var.sqrt() / mean);
            }
        }
        // L19: the tail columns against the median column of the same matrix.
        if tail > 0 {
            let mut all = d.clone();
            all.sort_by(|a, b| a.partial_cmp(b).expect("no NaN on a Hessian diagonal"));
            let med = quantile(&all, 0.5);
            if med > 0.0 {
                let t: f64 = d[e.n - tail..].iter().sum::<f64>() / tail as f64;
                entry.tails.push(t / med);
            }
        }
    }
    if per_act.is_empty() {
        println!("  no `diag-*.f64` in the dump\n");
        return Ok(());
    }
    for (act, d) in &per_act {
        let mut r = d.ratios.clone();
        let mut c = d.cvs.clone();
        let mut t = d.tails.clone();
        r.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        c.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "  {:<10} {:>7} {:>8} {:>10.2} {:>10.2} {:>10.3} {:>9}",
            act,
            d.full,
            d.tail,
            quantile(&r, 0.5),
            quantile(&r, 0.9),
            quantile(&c, 0.5),
            if t.is_empty() {
                "—".to_string()
            } else {
                format!("{:.3}", quantile(&t, 0.5))
            }
        );
    }
    println!();
    println!("  Reading it: a median max/min near 1 and a CV near 0 say the rotation has");
    println!("  flattened the diagonal, so neither the sweep order nor the tail's identity");
    println!("  carries information — L14 and L19 both close at zero. A tail/median far");
    println!("  from 1 says the remainder columns are not a random sample of the matrix,");
    println!("  which is the only thing that would make L19 playable.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Section 4 — the off-diagonal mass, as a caveat on everything above
// ---------------------------------------------------------------------------

fn section_offdiag(em: &[Emission]) {
    println!("SECTION 4 — HOW MUCH OF H THE DIAGONAL IS NOT");
    println!("{}", "-".repeat(74));
    println!("  Sections 2 and 3 read the diagonal. This says what they are ignoring:");
    println!("  ‖offdiag(H)‖_F against ‖diag(H)‖_F, per activation, over blocks.\n");
    let mut per_act: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for e in em {
        if e.trace > 0.0 {
            // `offdiag` is the Frobenius norm of the off-diagonal part;
            // `trace` is the sum of the diagonal. The ratio below is therefore
            // indicative, not a norm ratio — it is labelled as such.
            per_act.entry(e.act.as_str()).or_default().push(e.offdiag / e.trace);
        }
    }
    println!("  {:<10} {:>14} {:>14} {:>14}", "act", "median", "min", "max");
    for (act, v) in &per_act {
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "  {:<10} {:>14.4} {:>14.4} {:>14.4}",
            act,
            quantile(&s, 0.5),
            s[0],
            s[s.len() - 1]
        );
    }
    println!("\n  ‖offdiag‖_F / tr(H). Not a norm ratio — the denominator is a sum, not a");
    println!("  norm — so read it as an ordering across activations, not as a fraction.");
    println!();
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hstats: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 1 {
        return Err(
            "usage: hstats <hcapture-dir>\n\
             Reads the dump `llvq-llm --bin hcapture` writes and prints the statistics\n\
             that need nothing but it: L05 (section 2), L14 and L19 (section 3).\n\
             L02, L23, L24, L25 and L27 need ΔW and therefore the checkpoint; they are\n\
             not attempted here."
                .to_string(),
        );
    }
    let dir = PathBuf::from(&a[0]);
    let (header, em) = parse_meta(&dir)?;

    println!("\n{}", "=".repeat(74));
    println!("hstats — L36's consumer, on {}", dir.display());
    println!("{}", "=".repeat(74));
    println!("It measures and prints. It decides nothing, and three of L36's eight rows");
    println!("are all it can reach without the checkpoint.\n");

    section_provenance(&header, &em);
    section_trace(&dir, &em)?;
    section_diagonal(&dir, &em)?;
    section_offdiag(&em);

    println!("{}", "=".repeat(74));
    println!("NOT ANSWERED HERE");
    println!("{}", "=".repeat(74));
    println!("  L02, L23, L24, L25, L27  need ΔW = W − Ŵ, so the checkpoint beside the");
    println!("                           artifact. `llvq-bench` cannot read a checkpoint.");
    println!("  L06                      asks for a paired re-encode, not a statistic.");
    println!("  L15                      asks for kurtosis on `f1recdump` blocks, which");
    println!("                           carry no Hessian.");
    println!();
    Ok(())
}
