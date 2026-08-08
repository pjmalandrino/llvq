//! Paired comparison of two MMLU runs, from the dumps `bin/mmlu` writes.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --bin mmlupair -- <a.csv> <b.csv> [more…] [flags]`
//!
//! Flags: `--draws N` (default 10 000) · `--seed 0x…` · `--intersect` · `--no-fpc`.
//! With more than two dumps, every later one is compared against the **first**,
//! which is the shape of the campaigns: one baseline, several quantized arms.
//!
//! ## Why this binary exists
//!
//! Every MMLU error bar this project has published — 55.59 ± 1.35, 70.04 ±
//! 1.25, 65.52 ± 1.31 — is the sampling error of **one arm in isolation**. But
//! the arms answer the *same* questions: `mmlu.rs::select` seeds its shuffle
//! with the subject name's length and nothing else, so the sample does not
//! depend on the model, the device or the dtype. That makes the data
//! **paired**, and comparing two paired proportions with two independent bars
//! is both the wrong test and the conservative one — it can call a real gap
//! "not significant".
//!
//! Two claims of the dossier rest on differences that were never tested this
//! way: −0.28 pp between AWQ and f16 at 4B ("4-bit is indistinguishable from
//! f16"), and −3.07 pp at 8B ("4-bit starts to pay"). Neither is a measurement
//! until a paired interval is put on it.
//!
//! ## The two statistics, and why both are printed
//!
//! **McNemar** is the exact test for paired binary outcomes, and it is exactly
//! the right test for the *pooled* rate: only the discordant pairs carry
//! information, and under the null they split like a fair coin. But the number
//! this project publishes is the **stratified micro** — 57 subject rates
//! re-weighted by their true population, because MMLU's test split runs from
//! 100 questions (`abstract_algebra`) to 1 534 (`professional_law`). McNemar
//! knows nothing about those weights, so it answers a question adjacent to the
//! published one, not the published one.
//!
//! **The paired stratified bootstrap** answers the published one: resample
//! questions inside each subject, recompute both stratified micros, keep the
//! difference. Both are printed, labelled, and they are allowed to disagree —
//! a disagreement between them is a statement about where the difference sits
//! in the subject profile, which is information rather than noise.
//!
//! ## No new dependency
//!
//! An exact binomial tail is a sum of binomial coefficients in log space, and a
//! bootstrap is a loop over a seeded PRNG the workspace already ships
//! (`llvq_core::SplitMix64`). Both are here in full, for the same reason the
//! rest of the workspace is: a number nobody can audit is worth what an
//! un-auditable number is worth.

use llvq_core::SplitMix64;
use std::collections::{BTreeMap, BTreeSet};

/// Must be the first line of a dump. See `mmlu.rs::DUMP_VERSION`.
const DUMP_VERSION: &str = "# llvq-mmlu-dump v1";

/// Default bootstrap seed. Fixed, printed with the result, overridable — an
/// interval that moves between two runs of the *analysis* is not a measurement
/// of anything.
const DEFAULT_SEED: u64 = 0xB007_5EED;

/// The columns this analysis cannot do without. The dump carries more (the
/// four logits); they are read past on purpose, so that adding a column never
/// breaks a reader.
const REQUIRED: [&str; 7] = [
    "subject",
    "index",
    "population",
    "qhash",
    "answer",
    "pick",
    "correct",
];

// ---------------------------------------------------------------- parsing --

/// One scored question.
#[derive(Clone, Copy, Debug)]
struct Row {
    /// Size of the subject in the *test* split — the stratum weight.
    population: usize,
    /// Fingerprint of this question's prompt tokens.
    qhash: u64,
    correct: bool,
}

#[derive(Debug)]
struct Dump {
    path: String,
    label: String,
    dtype: String,
    limit: String,
    fingerprint: u64,
    /// `(subject, parquet index)` → row. A `BTreeMap` because every downstream
    /// number — the join order, the bootstrap draws — has to be reproducible,
    /// and a hash map's iteration order is not.
    rows: BTreeMap<(String, usize), Row>,
}

impl Dump {
    fn read(path: &str) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
        Self::parse(&text, path)
    }

    /// Header block, column line, rows, trailer — in that order.
    ///
    /// Columns are resolved **by name**. The writer is in `bin/mmlu` and the
    /// reader is here; two binaries cannot share a struct, and a positional
    /// contract between them would break silently the day someone inserts a
    /// column.
    fn parse(text: &str, path: &str) -> anyhow::Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let first = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("{path} is empty"))?;
        anyhow::ensure!(
            first.trim_end() == DUMP_VERSION,
            "{path} does not open with {DUMP_VERSION:?} — not an MMLU dump, or a \
             version this build does not read (got {first:?})"
        );

        let (mut label, mut dtype, mut limit) = (String::new(), String::new(), String::new());
        let mut columns: Option<Vec<&str>> = None;
        let mut rows: BTreeMap<(String, usize), Row> = BTreeMap::new();
        let mut trailer: Option<(u64, usize)> = None;

        for line in lines {
            let line = line.trim_end();
            if let Some(meta) = line.strip_prefix("# ") {
                if let Some(end) = meta.strip_prefix("end ") {
                    trailer = Some(parse_trailer(end, path)?);
                } else if let Some((k, v)) = meta.split_once('=') {
                    match k {
                        "model" => label = v.to_string(),
                        "dtype" => dtype = v.to_string(),
                        "limit" => limit = v.to_string(),
                        _ => {}
                    }
                }
                continue;
            }
            anyhow::ensure!(
                trailer.is_none(),
                "{path} has data after its trailer — two runs concatenated?"
            );
            let Some(cols) = columns.as_ref() else {
                let head: Vec<&str> = line.split(',').collect();
                for name in REQUIRED {
                    anyhow::ensure!(
                        head.contains(&name),
                        "{path} has no {name:?} column. A dump without it cannot feed \
                         a paired test — re-run `bin/mmlu` with a build that writes it."
                    );
                }
                columns = Some(head);
                continue;
            };
            let (key, row) = parse_row(line, cols, path)?;
            anyhow::ensure!(
                rows.insert(key.clone(), row).is_none(),
                "{path} scores {}[{}] twice",
                key.0,
                key.1
            );
        }

        // The trailer is a completion marker, not decoration: a census runs for
        // hours against a 30-minute platform default, and a job killed at the
        // timeout leaves a file that parses cleanly and is short by however
        // many subjects never ran. Refusing it here is the difference between
        // a missing result and a wrong one.
        let (fingerprint, questions) = trailer.ok_or_else(|| {
            anyhow::anyhow!(
                "{path} has no `# end` trailer — the run did not finish (timeout?), \
                 so the dump is a prefix of an unknown fraction of the split"
            )
        })?;
        anyhow::ensure!(
            questions == rows.len(),
            "{path} announces {questions} questions and carries {} — truncated or edited",
            rows.len()
        );
        Ok(Self {
            path: path.to_string(),
            label,
            dtype,
            limit,
            fingerprint,
            rows,
        })
    }

    fn subject_population(&self) -> BTreeMap<&str, usize> {
        let mut m = BTreeMap::new();
        for ((subject, _), row) in &self.rows {
            m.insert(subject.as_str(), row.population);
        }
        m
    }
}

fn parse_trailer(end: &str, path: &str) -> anyhow::Result<(u64, usize)> {
    let (mut fp, mut n) = (None, None);
    for field in end.split_whitespace() {
        match field.split_once('=') {
            Some(("fingerprint", v)) => fp = Some(u64::from_str_radix(v, 16)?),
            Some(("questions", v)) => n = Some(v.parse::<usize>()?),
            _ => {}
        }
    }
    match (fp, n) {
        (Some(fp), Some(n)) => Ok((fp, n)),
        _ => Err(anyhow::anyhow!("{path}: malformed trailer {end:?}")),
    }
}

fn parse_row(
    line: &str,
    columns: &[&str],
    path: &str,
) -> anyhow::Result<((String, usize), Row)> {
    let fields: Vec<&str> = line.split(',').collect();
    anyhow::ensure!(
        fields.len() == columns.len(),
        "{path}: {} fields for {} columns in {line:?}",
        fields.len(),
        columns.len()
    );
    let get = |name: &str| -> &str {
        fields[columns
            .iter()
            .position(|c| *c == name)
            .expect("presence checked when the header was read")]
    };
    let answer: usize = get("answer").parse()?;
    let pick: usize = get("pick").parse()?;
    let correct = match get("correct") {
        "0" => false,
        "1" => true,
        other => anyhow::bail!("{path}: correct={other:?}, expected 0 or 1"),
    };
    // A derived column read back as authoritative has to be checked against
    // what derives it, or a corrupted file scores as a plausible one.
    anyhow::ensure!(
        correct == (pick == answer),
        "{path}: correct={} contradicts pick={pick} / answer={answer} in {line:?}",
        u8::from(correct)
    );
    Ok((
        (get("subject").to_string(), get("index").parse()?),
        Row {
            population: get("population").parse()?,
            qhash: u64::from_str_radix(get("qhash"), 16)?,
            correct,
        },
    ))
}

// ----------------------------------------------------------------- paring --

/// One subject, paired.
#[derive(Clone, Debug)]
struct Stratum {
    subject: String,
    /// Questions the subject holds in the test split — the stratum weight, and
    /// the finite population correction's denominator.
    population: usize,
    /// Per question: `+1` where A is right and B wrong, `−1` where B is right
    /// and A wrong, `0` on the concordant pairs. **The sign convention of the
    /// whole tool is A − B**, stated once here.
    d: Vec<i8>,
    a_right: usize,
    b_right: usize,
}

impl Stratum {
    fn n(&self) -> usize {
        self.d.len()
    }
    fn mean_d(&self) -> f64 {
        if self.d.is_empty() {
            return 0.0;
        }
        self.d.iter().map(|&x| x as f64).sum::<f64>() / self.n() as f64
    }
}

/// Join two dumps question by question.
///
/// `intersect` waives the *run-level* refusal — the only legitimate use is a
/// census against a `limit=40` run, which share their 2 280 questions but
/// cannot share a run fingerprint because the census also scored 11 762
/// others. It never waives the per-question check below.
fn pair(a: &Dump, b: &Dump, intersect: bool) -> anyhow::Result<Vec<Stratum>> {
    // The headline gate. Two arms are comparable because they were asked the
    // same questions in the same words; that used to be established by reading
    // the code, and this is where it becomes a check on the data.
    anyhow::ensure!(
        a.fingerprint == b.fingerprint || intersect,
        "token fingerprints differ: {} has {:016x}, {} has {:016x}. These two runs \
         were not asked the same questions, and pairing them would compare two \
         different exams. Pass --intersect only if one is a census of the other.",
        a.path,
        a.fingerprint,
        b.path,
        b.fingerprint
    );
    let keys_a: BTreeSet<&(String, usize)> = a.rows.keys().collect();
    let keys_b: BTreeSet<&(String, usize)> = b.rows.keys().collect();
    let common: Vec<&(String, usize)> = keys_a.intersection(&keys_b).copied().collect();
    anyhow::ensure!(
        !common.is_empty(),
        "{} and {} have no question in common",
        a.path,
        b.path
    );
    anyhow::ensure!(
        intersect || (keys_a.len() == common.len() && keys_b.len() == common.len()),
        "question sets differ: {} of {} in {}, {} in {} — pass --intersect to score \
         the overlap, and expect the stratum weights to be renormalized over it",
        common.len(),
        keys_a.len(),
        a.path,
        keys_b.len(),
        b.path
    );

    let (pa, pb) = (a.subject_population(), b.subject_population());
    let mut strata: BTreeMap<&str, Stratum> = BTreeMap::new();
    for key in common {
        let (ra, rb) = (a.rows[key], b.rows[key]);
        // Per question, and never waived: same key is not the same question if
        // the prompt tokenized differently. The sealed artifact carries its own
        // `tokenizer.json`, and it is *not* byte-identical to the checkpoint's
        // — that is a measured fact of this project, not a hypothesis.
        anyhow::ensure!(
            ra.qhash == rb.qhash,
            "{}[{}] was asked in different words: {:016x} in {}, {:016x} in {}",
            key.0,
            key.1,
            ra.qhash,
            a.path,
            rb.qhash,
            b.path
        );
        anyhow::ensure!(
            ra.population == rb.population,
            "{} holds {} questions in {} and {} in {} — two different splits",
            key.0,
            ra.population,
            a.path,
            rb.population,
            b.path
        );
        let s = strata.entry(&key.0).or_insert_with(|| Stratum {
            subject: key.0.clone(),
            population: pa[key.0.as_str()].max(pb[key.0.as_str()]),
            d: Vec::new(),
            a_right: 0,
            b_right: 0,
        });
        s.d.push(match (ra.correct, rb.correct) {
            (true, false) => 1,
            (false, true) => -1,
            _ => 0,
        });
        s.a_right += usize::from(ra.correct);
        s.b_right += usize::from(rb.correct);
    }
    Ok(strata.into_values().collect())
}

// -------------------------------------------------------------- statistics --

/// Discordant pairs: `(A right & B wrong, A wrong & B right)`.
fn discordant(strata: &[Stratum]) -> (usize, usize) {
    strata.iter().fold((0, 0), |(b, c), s| {
        (
            b + s.d.iter().filter(|&&x| x > 0).count(),
            c + s.d.iter().filter(|&&x| x < 0).count(),
        )
    })
}

/// McNemar's **exact** test: the two-sided binomial p-value of `min(b, c)`
/// successes out of `b + c` fair coin flips.
///
/// The concordant pairs — the questions both arms got right, and the ones both
/// got wrong — carry no information about a difference and do not appear. That
/// is the whole content of the test, and it is why a paired comparison is
/// sharper than two independent bars: on MMLU the arms agree on the large
/// majority of questions, and every one of those agreements is variance the
/// unpaired test still pays for.
///
/// The χ² form is not used: it is an approximation of this, and it misbehaves
/// exactly where the campaign sits — a handful of discordant pairs.
fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        // No discordant pair at all is not evidence of equality, it is no
        // evidence at all.
        return 1.0;
    }
    // Log-factorials by prefix sum. `n` is at most the whole test split
    // (14 042), so the table costs microseconds and keeps C(n, k) in log space,
    // where a 14 000-trial binomial coefficient does not overflow.
    let mut ln_fact = vec![0.0f64; n + 1];
    for k in 1..=n {
        ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
    }
    let ln2 = std::f64::consts::LN_2;
    let tail: f64 = (0..=b.min(c))
        .map(|k| (ln_fact[n] - ln_fact[k] - ln_fact[n - k] - n as f64 * ln2).exp())
        .sum();
    // Symmetric null, so doubling one tail is exact rather than a convention.
    (2.0 * tail).min(1.0)
}

/// How the 57 subject rates are combined.
#[derive(Clone, Copy, PartialEq)]
enum Weighting {
    /// `wₛ = Nₛ / ΣN` — one weight per *question of the test split*. The
    /// published statistic.
    Stratified,
    /// `wₛ = nₛ / Σn` — one weight per *question asked*. Under a uniform limit
    /// this is algebraically the unweighted mean of the subject rates, i.e. the
    /// macro average, and it is what McNemar tests.
    Pooled,
}

fn weights(strata: &[Stratum], mode: Weighting) -> Vec<f64> {
    let raw: Vec<f64> = strata
        .iter()
        .map(|s| match mode {
            Weighting::Stratified => s.population as f64,
            Weighting::Pooled => s.n() as f64,
        })
        .collect();
    let total: f64 = raw.iter().sum();
    if total == 0.0 {
        return raw;
    }
    raw.iter().map(|w| w / total).collect()
}

/// `Σ wₛ · (right / n)ₛ` for one arm.
fn rate(strata: &[Stratum], w: &[f64], right: impl Fn(&Stratum) -> usize) -> f64 {
    strata
        .iter()
        .zip(w)
        .filter(|(s, _)| s.n() > 0)
        .map(|(s, w)| w * right(s) as f64 / s.n() as f64)
        .sum()
}

/// `Σ wₛ · mean(d)ₛ` — the difference A − B, in the same weighting as [`rate`].
fn delta(strata: &[Stratum], w: &[f64]) -> f64 {
    strata.iter().zip(w).map(|(s, w)| w * s.mean_d()).sum()
}

struct Boot {
    lo: f64,
    hi: f64,
    se: f64,
}

/// Paired stratified bootstrap of the difference.
///
/// Resampling happens **inside** each subject, on the per-question differences
/// — never on the raw arms. That is what makes it paired: a question both arms
/// answered correctly contributes exactly zero variance, as it should, whereas
/// resampling the two arms independently would re-inject the whole
/// between-question variance that the shared question set already cancels.
///
/// `fpc` applies the finite population correction the same way `mmlu.rs`'s
/// `micro_stderr` does, by shrinking each stratum's bootstrap deviation by
/// `√(1 − nₛ/Nₛ)`. It matters for one reason: under a census the interval must
/// close to a point. The questions *are* the population then, the difference is
/// exact, and an interval around an exact number would be a lie — the same
/// stance the harness already takes when it prints ±0.00 on a full run.
fn bootstrap(strata: &[Stratum], w: &[f64], fpc: bool, draws: usize, seed: u64) -> Boot {
    let observed: Vec<f64> = strata.iter().map(Stratum::mean_d).collect();
    let shrink: Vec<f64> = strata
        .iter()
        .map(|s| {
            if !fpc || s.population == 0 {
                1.0
            } else {
                (1.0 - (s.n() as f64 / s.population as f64)).max(0.0).sqrt()
            }
        })
        .collect();
    let mut rng = SplitMix64::new(seed);
    let mut draw: Vec<f64> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut acc = 0.0;
        for (k, s) in strata.iter().enumerate() {
            let n = s.n();
            if n == 0 {
                continue;
            }
            // Modulo bias over a 64-bit stream with n ≤ 1 534 is below 1e-16 —
            // fifteen orders of magnitude under the Monte-Carlo error of
            // 10 000 draws.
            let mut sum = 0.0;
            for _ in 0..n {
                sum += s.d[(rng.next() % n as u64) as usize] as f64;
            }
            let m = sum / n as f64;
            acc += w[k] * (observed[k] + shrink[k] * (m - observed[k]));
        }
        draw.push(acc);
    }
    draw.sort_by(|x, y| x.total_cmp(y));
    let mean = draw.iter().sum::<f64>() / draws.max(1) as f64;
    let var = if draws > 1 {
        draw.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (draws - 1) as f64
    } else {
        0.0
    };
    Boot {
        lo: percentile(&draw, 0.025),
        hi: percentile(&draw, 0.975),
        se: var.sqrt(),
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ------------------------------------------------------------------- main --

struct Options {
    draws: usize,
    seed: u64,
    intersect: bool,
    fpc: bool,
}

fn parse_seed(s: &str) -> anyhow::Result<u64> {
    match s.strip_prefix("0x") {
        Some(hex) => Ok(u64::from_str_radix(hex, 16)?),
        None => Ok(s.parse()?),
    }
}

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut paths: Vec<String> = Vec::new();
    let mut opt = Options {
        draws: 10_000,
        seed: DEFAULT_SEED,
        intersect: false,
        fpc: true,
    };
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--draws" => {
                opt.draws = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--draws needs a count"))?
                    .parse()?
            }
            "--seed" => {
                opt.seed = parse_seed(it.next().ok_or_else(|| anyhow::anyhow!("--seed needs a value"))?)?
            }
            "--intersect" => opt.intersect = true,
            "--no-fpc" => opt.fpc = false,
            flag if flag.starts_with("--") => anyhow::bail!("unknown flag {flag}"),
            path => paths.push(path.to_string()),
        }
    }
    anyhow::ensure!(
        paths.len() >= 2,
        "give at least two dumps written by `LLVQ_MMLU_DUMP=… bin/mmlu`.\n\
         Usage: mmlupair <a.csv> <b.csv> [c.csv…] [--draws N] [--seed 0x…] \
         [--intersect] [--no-fpc]"
    );

    let reference = Dump::read(&paths[0])?;
    for path in &paths[1..] {
        let other = Dump::read(path)?;
        report(&reference, &other, &opt)?;
    }
    Ok(())
}

fn report(a: &Dump, b: &Dump, opt: &Options) -> anyhow::Result<()> {
    let strata = pair(a, b, opt.intersect)?;
    let n: usize = strata.iter().map(Stratum::n).sum();

    println!("\n{}", "=".repeat(72));
    println!("A = {}\n    {}", a.label, a.path);
    println!("B = {}\n    {}", b.label, b.path);
    println!(
        "{n} questions appariées, {} matières · dtype {} / {} · limit {} / {}",
        strata.len(),
        a.dtype,
        b.dtype,
        a.limit,
        b.limit
    );
    println!(
        "empreinte de run : {:016x} / {:016x}{}",
        a.fingerprint,
        b.fingerprint,
        if a.fingerprint == b.fingerprint {
            " (identique)"
        } else {
            " ⚠️ DIFFÉRENTE — appariement forcé sur l'intersection"
        }
    );
    if a.dtype != b.dtype {
        println!("⚠️ dtypes différents : la comparaison porte sur deux objets, pas un.");
    }

    let (ws, wp) = (
        weights(&strata, Weighting::Stratified),
        weights(&strata, Weighting::Pooled),
    );
    let (ds, dp) = (delta(&strata, &ws), delta(&strata, &wp));
    println!("\ntaux (Δ = A − B, en points de pourcentage)");
    println!(
        "  micro stratifié     A {:6.2} %   B {:6.2} %   Δ = {:+.2} pp   ← le chiffre publié",
        100.0 * rate(&strata, &ws, |s| s.a_right),
        100.0 * rate(&strata, &ws, |s| s.b_right),
        100.0 * ds
    );
    println!(
        "  non pondéré (poolé) A {:6.2} %   B {:6.2} %   Δ = {:+.2} pp   ← ce que McNemar teste",
        100.0 * rate(&strata, &wp, |s| s.a_right),
        100.0 * rate(&strata, &wp, |s| s.b_right),
        100.0 * dp
    );

    let (nb, nc) = discordant(&strata);
    let p = mcnemar_exact(nb, nc);
    let z = if nb + nc > 0 {
        (nb as f64 - nc as f64) / ((nb + nc) as f64).sqrt()
    } else {
        0.0
    };
    println!("\nMcNemar exact (non pondéré)");
    println!(
        "  A✓B✗ = {nb}   A✗B✓ = {nc}   concordantes = {}   ({:.1} % des questions sont \
         discordantes)",
        n - nb - nc,
        100.0 * (nb + nc) as f64 / n.max(1) as f64
    );
    // A tiny p-value printed as `0.0000` is a wrong number, not a rounded one:
    // an exact test never returns zero, and the campaign's real gaps sit many
    // orders of magnitude below 1e-4.
    if p < 1e-4 {
        println!("  p = {p:.3e}   (z normal = {z:+.2})");
    } else {
        println!("  p = {p:.4}   (z normal = {z:+.2})");
    }

    let boot = bootstrap(&strata, &ws, opt.fpc, opt.draws, opt.seed);
    let flat = bootstrap(&strata, &wp, opt.fpc, opt.draws, opt.seed);
    println!(
        "\nbootstrap apparié stratifié par matière ({} tirages, graine {:#x}{})",
        opt.draws,
        opt.seed,
        if opt.fpc {
            ", correction de population finie"
        } else {
            ", SANS correction de population finie"
        }
    );
    println!(
        "  Δ micro stratifié = {:+.2} pp   IC95 [{:+.2} ; {:+.2}]   SE {:.2} pp",
        100.0 * ds,
        100.0 * boot.lo,
        100.0 * boot.hi,
        100.0 * boot.se
    );
    println!(
        "  contrôle non pondéré = {:+.2} pp   IC95 [{:+.2} ; {:+.2}]   SE {:.2} pp",
        100.0 * dp,
        100.0 * flat.lo,
        100.0 * flat.hi,
        100.0 * flat.se
    );
    println!(
        "  → {}",
        if boot.lo <= 0.0 && boot.hi >= 0.0 {
            "l'IC contient zéro : l'écart n'est pas résolu par ce protocole."
        } else {
            "l'IC exclut zéro : l'écart est résolu par ce protocole."
        }
    );

    // The subject profile, because the aggregate hides the mechanism: this
    // project's own 2-bit arm sits at chance on abstract algebra while holding
    // above 80 % on law, and a single number cannot show that.
    let mut movers: Vec<(f64, &Stratum)> = strata
        .iter()
        .zip(&ws)
        .map(|(s, w)| (w * s.mean_d(), s))
        .collect();
    movers.sort_by(|x, y| y.0.abs().total_cmp(&x.0.abs()));
    println!("\nmatières qui portent l'écart (contribution au Δ stratifié)");
    for (contribution, s) in movers.iter().take(5) {
        println!(
            "  {:<40}{:+6.1} pp × poids {:4.1} % = {:+.3} pp",
            s.subject.replace('_', " "),
            100.0 * s.mean_d(),
            100.0 * s.population as f64 / strata.iter().map(|x| x.population).sum::<usize>() as f64,
            100.0 * contribution
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stratum with `plus` questions only A gets right, `minus` only B, and
    /// `both` the arms agree on (correct for both).
    fn stratum(subject: &str, population: usize, plus: usize, minus: usize, both: usize) -> Stratum {
        let mut d = vec![1i8; plus];
        d.extend(std::iter::repeat_n(-1i8, minus));
        d.extend(std::iter::repeat_n(0i8, both));
        Stratum {
            subject: subject.to_string(),
            population,
            d,
            a_right: plus + both,
            b_right: minus + both,
        }
    }

    fn boot_of(strata: &[Stratum], mode: Weighting, fpc: bool) -> Boot {
        bootstrap(strata, &weights(strata, mode), fpc, 4_000, DEFAULT_SEED)
    }

    // ------------------------------------------------------------ McNemar --

    /// Hand-computable cases. `p = 2·Σ_{k≤min(b,c)} C(n,k) / 2ⁿ`:
    /// 10 discordant pairs 9–1 give `2·(1+10)/1024 = 0.021484375` exactly, and
    /// 6–0 give `2/64 = 0.03125`. Both are written out rather than compared to
    /// another implementation, because "agrees with itself" is what a wrong
    /// implementation also does.
    #[test]
    fn mcnemar_matches_the_hand_computation() {
        assert!((mcnemar_exact(9, 1) - 0.021_484_375).abs() < 1e-12);
        assert!((mcnemar_exact(6, 0) - 0.031_25).abs() < 1e-12);
        // 5–5: Σ_{k≤5} C(10,k) = 638, and 2·638/1024 > 1, so the two-sided
        // p-value saturates rather than exceeding 1.
        assert_eq!(mcnemar_exact(5, 5), 1.0);
        // A single discordant pair can never be evidence: 2·(1/2) = 1.
        assert_eq!(mcnemar_exact(1, 0), 1.0);
        // No discordance is no evidence, not proof of equality.
        assert_eq!(mcnemar_exact(0, 0), 1.0);
    }

    /// The test must not see the concordant pairs. Two arms that agree on 20
    /// questions and two that agree on 20 000 have the same evidence about a
    /// difference — that is precisely the variance an unpaired bar keeps paying
    /// for, and if this assertion ever fails the tool has stopped being paired.
    #[test]
    fn mcnemar_ignores_the_concordant_pairs() {
        let thin = vec![stratum("s", 1_000, 9, 1, 20)];
        let fat = vec![stratum("s", 1_000, 9, 1, 20_000)];
        assert_eq!(discordant(&thin), (9, 1));
        assert_eq!(discordant(&fat), (9, 1));
        let (p_thin, p_fat) = {
            let (b, c) = discordant(&thin);
            let (b2, c2) = discordant(&fat);
            (mcnemar_exact(b, c), mcnemar_exact(b2, c2))
        };
        assert_eq!(p_thin, p_fat);
        assert!(p_thin < 0.05, "9 against 1 is significant: {p_thin}");
    }

    /// Swapping the arms must leave the p-value alone and flip the difference.
    /// The p-value is a two-sided statement about a magnitude; the delta is the
    /// direction, and the two must not be confused — a tool that reports "B is
    /// better" when B is worse is worse than no tool.
    #[test]
    fn swapping_the_arms_flips_the_sign_and_not_the_p_value() {
        let ab = vec![stratum("s", 1_000, 30, 10, 200)];
        let ba = vec![stratum("s", 1_000, 10, 30, 200)];
        let (b, c) = discordant(&ab);
        let (b2, c2) = discordant(&ba);
        assert_eq!((b, c), (30, 10));
        assert_eq!((b2, c2), (10, 30));
        assert_eq!(mcnemar_exact(b, c), mcnemar_exact(b2, c2));

        let (w1, w2) = (
            weights(&ab, Weighting::Stratified),
            weights(&ba, Weighting::Stratified),
        );
        let (d1, d2) = (delta(&ab, &w1), delta(&ba, &w2));
        assert!(d1 > 0.0, "A wins 30 to 10, so Δ = A − B must be positive");
        assert!((d1 + d2).abs() < 1e-12, "{d1} and {d2} must be opposite");
    }

    // ---------------------------------------------------------- bootstrap --

    /// Two arms that trade questions evenly are not two different arms. The
    /// interval must contain zero — and it must have positive width, or the
    /// test would also pass on a bootstrap that resamples nothing.
    #[test]
    fn evenly_traded_questions_leave_zero_inside_the_interval() {
        let strata = vec![stratum("s", 1_000, 40, 40, 200)];
        let b = boot_of(&strata, Weighting::Stratified, true);
        assert!(b.lo < 0.0 && b.hi > 0.0, "[{}, {}]", b.lo, b.hi);
        assert!(b.se > 0.0, "a sample of 280 out of 1 000 has sampling error");
    }

    /// And an arm that is strictly better must come out as strictly better:
    /// 60 questions won, none lost, over 280.
    #[test]
    fn a_strictly_better_arm_excludes_zero() {
        let strata = vec![stratum("s", 1_000, 60, 0, 220)];
        let b = boot_of(&strata, Weighting::Stratified, true);
        assert!(b.lo > 0.0, "IC95 [{}, {}] should sit above zero", b.lo, b.hi);
    }

    /// **The stratification test.** Two subjects of the real MMLU shape — 100
    /// questions and 1 500 — sampled to the same depth, with A winning the
    /// small one by as much as it loses the large one.
    ///
    /// Pooled, the two cancel: Δ = 0, interval straddling zero, "no
    /// difference". Stratified, the large subject carries 15 times the weight:
    /// Δ = (100·(+0.5) + 1500·(−0.5))/1600 = −0.4375, and the interval excludes
    /// zero. **The two estimators disagree on the sign, the magnitude and the
    /// conclusion**, which is the point: a test where stratified and unweighted
    /// coincide proves nothing about the stratification, exactly as §5 of
    /// `CLAUDE.md` says of any parameter only ever exercised at its neutral
    /// value.
    #[test]
    fn stratification_flips_the_conclusion() {
        let strata = vec![
            stratum("abstract_algebra", 100, 30, 10, 0),
            stratum("professional_law", 1_500, 10, 30, 0),
        ];
        let (ws, wp) = (
            weights(&strata, Weighting::Stratified),
            weights(&strata, Weighting::Pooled),
        );
        let (ds, dp) = (delta(&strata, &ws), delta(&strata, &wp));
        assert!(dp.abs() < 1e-12, "pooled must cancel exactly, got {dp}");
        assert!((ds + 0.437_5).abs() < 1e-12, "stratified must be −0.4375, got {ds}");

        let (bs, bp) = (
            boot_of(&strata, Weighting::Stratified, true),
            boot_of(&strata, Weighting::Pooled, true),
        );
        assert!(bs.hi < 0.0, "stratified IC95 [{}, {}] must exclude zero", bs.lo, bs.hi);
        assert!(
            bp.lo < 0.0 && bp.hi > 0.0,
            "pooled IC95 [{}, {}] must contain zero",
            bp.lo,
            bp.hi
        );
    }

    /// A census is not a sample. When every question of every subject has been
    /// scored, the difference is exact and the sampling interval must close to
    /// a point — the same stance `mmlu.rs::micro_stderr` already takes when it
    /// prints ±0.00 on a full run. Without the correction the same data yields
    /// a wide interval, which is the assertion that makes the correction
    /// load-bearing rather than decorative.
    #[test]
    fn a_census_has_no_sampling_interval() {
        let strata = vec![stratum("s", 300, 60, 20, 220)];
        assert_eq!(strata[0].n(), strata[0].population, "this is a census");
        let with = boot_of(&strata, Weighting::Stratified, true);
        // Not `== 0.0`: every draw lands on the same value, but the mean of
        // 4 000 identical doubles is that value to within an ulp, so the
        // variance is fp residue (~1e-15) rather than an exact zero. The bound
        // is twelve orders of magnitude below the interval the same data yields
        // without the correction, which is the assertion right below.
        assert!(with.se < 1e-12, "se = {}", with.se);
        assert!(with.hi - with.lo < 1e-12, "[{}, {}] must be a point", with.lo, with.hi);
        let without = boot_of(&strata, Weighting::Stratified, false);
        assert!(without.hi - without.lo > 0.02, "{} wide", without.hi - without.lo);
    }

    /// Same seed, same interval — or the tool reports a different number every
    /// time it is asked the same question, which is the failure mode the whole
    /// campaign is built to avoid. A different seed must move it, or the seed
    /// is not wired to the draws at all.
    #[test]
    fn the_bootstrap_is_deterministic_and_the_seed_is_wired() {
        let strata = vec![stratum("s", 1_000, 40, 30, 210)];
        let w = weights(&strata, Weighting::Stratified);
        let a = bootstrap(&strata, &w, true, 2_000, 7);
        let b = bootstrap(&strata, &w, true, 2_000, 7);
        assert_eq!((a.lo, a.hi, a.se), (b.lo, b.hi, b.se));
        let c = bootstrap(&strata, &w, true, 2_000, 8);
        assert_ne!((a.lo, a.hi), (c.lo, c.hi), "the seed must reach the draws");
        // Different seeds are still the same measurement: both must bracket the
        // observed difference.
        let d = delta(&strata, &w);
        for b in [&a, &c] {
            assert!(b.lo <= d && d <= b.hi, "[{}, {}] misses {d}", b.lo, b.hi);
        }
    }

    // ------------------------------------------------------------ parsing --

    fn dump_text(fingerprint: &str, rows: &[(&str, usize, usize, &str, usize, usize)]) -> String {
        let mut s = format!("{DUMP_VERSION}\n# model=test\n# dtype=f16\n# limit=40\n");
        s.push_str("subject,index,population,qhash,answer,pick,correct\n");
        for (subject, index, population, qhash, answer, pick) in rows {
            s.push_str(&format!(
                "{subject},{index},{population},{qhash},{answer},{pick},{}\n",
                u8::from(answer == pick)
            ));
        }
        s.push_str(&format!(
            "# end fingerprint={fingerprint} questions={}\n",
            rows.len()
        ));
        s
    }

    fn two_rows(pick_second: usize) -> String {
        dump_text(
            "65dcd53655e8bfa5",
            &[
                ("law", 0, 1_534, "aaaa", 1, 1),
                ("law", 1, 1_534, "bbbb", 2, pick_second),
            ],
        )
    }

    #[test]
    fn a_dump_round_trips() {
        let d = Dump::parse(&two_rows(3), "a.csv").unwrap();
        assert_eq!(d.rows.len(), 2);
        assert_eq!(d.fingerprint, 0x65dc_d536_55e8_bfa5);
        assert_eq!(d.dtype, "f16");
        assert!(d.rows[&("law".to_string(), 0)].correct);
        assert!(!d.rows[&("law".to_string(), 1)].correct);
        let strata = pair(&d, &Dump::parse(&two_rows(2), "b.csv").unwrap(), false).unwrap();
        assert_eq!(discordant(&strata), (0, 1), "B got the second one right");
    }

    /// **The refusal.** Two runs that were not asked the same questions cannot
    /// be paired, and the only signal that says so before the numbers look
    /// plausible is the fingerprint. It is also the one an operator will be
    /// tempted to ignore, so it is an error and not a warning.
    #[test]
    fn dumps_of_different_exams_are_refused() {
        let a = Dump::parse(&two_rows(3), "a.csv").unwrap();
        let other = two_rows(3).replace("65dcd53655e8bfa5", "0123456789abcdef");
        let b = Dump::parse(&other, "b.csv").unwrap();
        let err = pair(&a, &b, false).unwrap_err().to_string();
        assert!(err.contains("fingerprint"), "{err}");
        // …and --intersect is the documented, explicit escape hatch, for the
        // one legitimate case: a census against a sampled run.
        assert!(pair(&a, &b, true).is_ok());
    }

    /// Equal fingerprints are necessary, not sufficient: they say the token
    /// streams matched, and a hand-edited or concatenated dump can still cover
    /// a different set of questions. The set is therefore checked on its own,
    /// and `--intersect` is what turns the refusal into a scored overlap —
    /// which is the census-against-`limit=40` case, and the only one.
    #[test]
    fn a_different_question_set_is_refused_even_at_equal_fingerprints() {
        let a = Dump::parse(
            &dump_text(
                "65dcd53655e8bfa5",
                &[
                    ("law", 0, 1_534, "aaaa", 1, 1),
                    ("law", 1, 1_534, "bbbb", 2, 3),
                    ("law", 2, 1_534, "cccc", 0, 0),
                ],
            ),
            "a.csv",
        )
        .unwrap();
        let b = Dump::parse(&two_rows(2), "b.csv").unwrap();
        assert_eq!(a.fingerprint, b.fingerprint, "the cheap gate cannot see this");
        let err = pair(&a, &b, false).unwrap_err().to_string();
        assert!(err.contains("question sets differ"), "{err}");
        let strata = pair(&a, &b, true).unwrap();
        assert_eq!(strata.iter().map(Stratum::n).sum::<usize>(), 2, "the overlap");
    }

    /// The per-question hash is *not* waivable. Same question id, different
    /// prompt tokens, means the two arms read different text — which is not
    /// hypothetical here: the sealed artifact ships its own `tokenizer.json`
    /// and it is not byte-identical to the checkpoint's.
    #[test]
    fn a_question_asked_in_different_words_is_refused_even_with_intersect() {
        let a = Dump::parse(&two_rows(3), "a.csv").unwrap();
        let b = Dump::parse(&two_rows(3).replace(",bbbb,", ",cccc,"), "b.csv").unwrap();
        for intersect in [false, true] {
            let err = pair(&a, &b, intersect).unwrap_err().to_string();
            assert!(err.contains("different words"), "{err}");
        }
    }

    /// A run killed at a platform timeout leaves a file that parses and scores.
    /// The trailer is what makes that a refusal instead of a short answer.
    #[test]
    fn a_dump_without_its_trailer_is_refused() {
        let text = two_rows(3);
        let truncated: String = text
            .lines()
            .filter(|l| !l.starts_with("# end"))
            .map(|l| format!("{l}\n"))
            .collect();
        let err = Dump::parse(&truncated, "a.csv").unwrap_err().to_string();
        assert!(err.contains("trailer"), "{err}");

        // And a trailer that disagrees with the body is worse than none.
        let lied = text.replace("questions=2", "questions=3");
        assert!(Dump::parse(&lied, "a.csv").is_err());
    }

    /// The stratum weight has to come from the file. A dump missing
    /// `population` cannot produce the published statistic, and silently
    /// falling back to the pooled rate is the macro/micro confusion all over
    /// again — so it is refused at the header.
    #[test]
    fn a_dump_without_the_stratum_weight_is_refused() {
        let text = two_rows(3)
            .replace("subject,index,population,qhash", "subject,index,qhash")
            .replace(",1534,", ",");
        let err = Dump::parse(&text, "a.csv").unwrap_err().to_string();
        assert!(err.contains("population"), "{err}");
    }

    /// `correct` is derived from `pick` and `answer`. A file where they
    /// disagree has been edited or corrupted, and it must not score.
    #[test]
    fn a_self_contradictory_row_is_refused() {
        let text = two_rows(3).replace("law,0,1534,aaaa,1,1,1", "law,0,1534,aaaa,1,1,0");
        let err = Dump::parse(&text, "a.csv").unwrap_err().to_string();
        assert!(err.contains("contradicts"), "{err}");
    }
}
