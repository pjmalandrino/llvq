//! GSM8K scoring: the instruction, the sample, the extraction of an answer
//! from a free-text completion, the grading, and the paired statistics.
//!
//! Everything here is pure, so the whole grading chain is tested on the Mac
//! without a model. `bin/gsm8k` runs the generation around it and
//! `bin/gsm8kpair` reads two of its dumps.
//!
//! ## The protocol
//!
//! * zero-shot, Qwen3 chat template (`crate::chatfmt`), reasoning block
//!   pre-filled empty unless `LLVQ_GSM8K_THINK=1`;
//! * the user turn is the problem, a newline, and [`INSTRUCTION`], the prompt
//!   Qwen publishes for its own math evaluations;
//! * greedy decode, stop on `<|im_end|>` or `<|endoftext|>`, cap at `max_new`;
//! * the answer is the last number inside the last `\boxed{…}`; with no box,
//!   the last number of the completion;
//! * the gold is the number after the last `####` of the reference solution;
//! * a problem is right when both reduce to the same decimal, exactly: no
//!   float, no tolerance.
//!
//! The dump keeps the raw completion, so a different extraction rule can be
//! applied later to the same generations without paying for them twice.

use llvq_core::SplitMix64;

/// The instruction appended to every problem.
pub const INSTRUCTION: &str =
    "Please reason step by step, and put your final answer within \\boxed{}.";

/// The user turn's body: the problem, a newline, the instruction.
pub fn user_body(question: &str) -> String {
    format!("{}\n{INSTRUCTION}", question.trim())
}

/// Seed of the sample. Fixed, so every arm scores the same problems.
pub const SAMPLE_SEED: u64 = 0x0065_736D_386B;

/// Which problems a run scores, in index order.
///
/// `limit >= n` is the census, every problem in parquet order. Below that, a
/// seeded Fisher–Yates over the indices, the first `limit`, sorted back into
/// index order. The seed is a constant, so the sample is the same for every
/// arm by construction, and a smaller sample is a subset of a larger one.
pub fn select(n: usize, limit: usize) -> Vec<usize> {
    if limit >= n {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(SAMPLE_SEED);
    for i in (1..n).rev() {
        idx.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    idx.truncate(limit);
    idx.sort_unstable();
    idx
}

/// Content of the LAST `\boxed{…}`, braces balanced.
///
/// `None` when there is no box, and when the last one never closes: a
/// completion cut by the cap inside its box has not given an answer, and
/// reading a half-written number out of it would score a truncation.
pub fn last_boxed(s: &str) -> Option<&str> {
    const KEY: &str = "\\boxed{";
    let start = s.rfind(KEY)? + KEY.len();
    let mut depth = 1usize;
    for (i, ch) in s[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every number written in `s`, in order, spelled as in the text: an optional
/// minus, digits with commas allowed only between digits, an optional decimal
/// part. `"1,000"` is one number, `"3, 4"` two, `"-5"` negative.
pub fn numbers(s: &str) -> Vec<&str> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = if i > 0 && b[i - 1] == b'-' { i - 1 } else { i };
        let mut j = i;
        while j < b.len()
            && (b[j].is_ascii_digit()
                || (b[j] == b',' && j + 1 < b.len() && b[j + 1].is_ascii_digit()))
        {
            j += 1;
        }
        if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
            j += 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
        }
        out.push(&s[start..j]);
        i = j;
    }
    out
}

/// A number reduced to one spelling: no grouping commas, no leading zeros, no
/// trailing fractional zeros, no sign on zero. `"1,000.50"` → `"1000.5"`,
/// `"-0.0"` → `"0"`. `None` when `raw` is not a plain decimal.
pub fn canonical(raw: &str) -> Option<String> {
    let t: String = raw.trim().chars().filter(|&c| c != ',').collect();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.as_str()),
    };
    let (int, frac) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    if !int.bytes().all(|c| c.is_ascii_digit()) || !frac.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let int = int.trim_start_matches('0');
    let int = if int.is_empty() { "0" } else { int };
    let frac = frac.trim_end_matches('0');
    let zero = int == "0" && frac.is_empty();
    let mut out = String::new();
    if neg && !zero {
        out.push('-');
    }
    out.push_str(int);
    if !frac.is_empty() {
        out.push('.');
        out.push_str(frac);
    }
    Some(out)
}

/// The gold of a reference solution: the number after its last `####`.
pub fn gold(answer: &str) -> Option<String> {
    let (_, tail) = answer.rsplit_once("####")?;
    canonical(numbers(tail).first()?)
}

/// Where an extracted answer came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The last number inside the last `\boxed{…}`.
    Boxed,
    /// No closed box: the last number of the whole completion.
    LastNumber,
    /// Nothing that reads as a number.
    None,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Boxed => "boxed",
            Source::LastNumber => "last",
            Source::None => "none",
        }
    }
}

/// The answer read out of a completion, canonical, and where it came from.
///
/// A `\frac{a}{b}` box is read as the integer `a / b` when `b` divides `a`,
/// and as no number otherwise: every GSM8K gold is an integer, so a fraction
/// that is not one cannot be right, and its numerator must not be read as the
/// answer.
pub fn extract(completion: &str) -> (Source, Option<String>) {
    if let Some(inner) = last_boxed(completion) {
        if let Some(v) = frac_value(inner) {
            return (Source::Boxed, v);
        }
        if let Some(n) = numbers(inner).last() {
            return (Source::Boxed, canonical(n));
        }
    }
    match numbers(completion).last() {
        Some(n) => (Source::LastNumber, canonical(n)),
        None => (Source::None, None),
    }
}

/// `Some(Some(q))` for `\frac{a}{b}` (or `\dfrac`, `\tfrac`) with integer `a`,
/// `b` and `b | a`; `Some(None)` for a fraction that is not an integer;
/// `None` when `inner` holds no fraction.
fn frac_value(inner: &str) -> Option<Option<String>> {
    let key = ["\\dfrac{", "\\tfrac{", "\\frac{"]
        .iter()
        .find_map(|k| inner.find(k).map(|p| p + k.len()))?;
    let rest = &inner[key..];
    let (num, rest) = rest.split_once('}')?;
    let den = rest.strip_prefix('{')?.split_once('}')?.0;
    let parse = |s: &str| -> Option<i128> {
        let c = canonical(s)?;
        if c.contains('.') {
            None
        } else {
            c.parse().ok()
        }
    };
    let (Some(a), Some(b)) = (parse(num), parse(den)) else {
        return Some(None);
    };
    if b == 0 || a % b != 0 {
        return Some(None);
    }
    Some(Some((a / b).to_string()))
}

/// Right or wrong: both sides canonical, compared as strings.
pub fn is_correct(extracted: Option<&str>, gold: &str) -> bool {
    extracted == Some(gold)
}

/// First line of a dump. `bin/gsm8kpair` refuses a file that does not start
/// with it.
pub const DUMP_TAG: &str = "llvq_gsm8k_dump";
pub const DUMP_VERSION: u64 = 1;

/// One scored problem, as a dump line.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Row {
    pub index: usize,
    /// `eval::token_fingerprint` of the prompt ids, hex. Proves question by
    /// question that two arms were asked the same thing in the same tokens.
    pub qhash: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    /// `eos` or `cap`.
    pub stop: String,
    pub gold: String,
    pub extracted: Option<String>,
    pub source: String,
    pub correct: bool,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub completion: String,
}

/// McNemar's exact test: the two-sided binomial p-value of `min(b, c)`
/// successes in `b + c` fair flips. Same computation as `bin/mmlupair`.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let mut ln_fact = vec![0.0f64; n + 1];
    for k in 1..=n {
        ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
    }
    let ln2 = std::f64::consts::LN_2;
    let tail: f64 = (0..=b.min(c))
        .map(|k| (ln_fact[n] - ln_fact[k] - ln_fact[n - k] - n as f64 * ln2).exp())
        .sum();
    (2.0 * tail).min(1.0)
}

/// Paired difference A − B over the same problems, as a rate, with its 95 %
/// normal interval and no finite-population correction: the split is read as
/// a draw from the problems the model could be asked, the reading every
/// census interval of this project takes.
///
/// `d[i]` is +1 when only A is right, −1 when only B is, 0 otherwise.
pub fn paired_interval(d: &[i8]) -> (f64, f64, f64) {
    let n = d.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = d.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, mean, mean);
    }
    let var = d
        .iter()
        .map(|&x| (x as f64 - mean) * (x as f64 - mean))
        .sum::<f64>()
        / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    (mean, mean - 1.96 * se, mean + 1.96 * se)
}

/// A dump read back: its header, its rows, the run fingerprint of its trailer.
#[derive(Debug)]
pub struct Dump {
    pub header: serde_json::Value,
    pub rows: Vec<Row>,
    pub fingerprint: String,
}

impl Dump {
    /// A header field as text, for printing. `?` when absent.
    pub fn field(&self, key: &str) -> String {
        match self.header.get(key) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => "?".to_string(),
        }
    }
}

/// Parse a dump. Refuses a file without its header tag, and a file without
/// its trailer: a job killed at a timeout leaves a dump that parses and is
/// short, and only the trailer tells the two apart.
pub fn parse_dump(text: &str, name: &str) -> anyhow::Result<Dump> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: serde_json::Value = serde_json::from_str(
        lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("{name} is empty"))?,
    )?;
    anyhow::ensure!(
        header.get(DUMP_TAG).and_then(|v| v.as_u64()) == Some(DUMP_VERSION),
        "{name} does not open with {{\"{DUMP_TAG}\": {DUMP_VERSION}, ...}}: not a GSM8K dump \
         of this version"
    );
    let mut rows = Vec::new();
    let mut fingerprint = None;
    for line in lines {
        let v: serde_json::Value = serde_json::from_str(line)?;
        if v.get("end").is_some() {
            anyhow::ensure!(fingerprint.is_none(), "{name} carries two trailers");
            let questions = v.get("questions").and_then(|q| q.as_u64());
            anyhow::ensure!(
                questions == Some(rows.len() as u64),
                "{name}: the trailer counts {questions:?} problems, the file holds {}",
                rows.len()
            );
            fingerprint = Some(
                v.get("fingerprint")
                    .and_then(|f| f.as_str())
                    .unwrap_or("?")
                    .to_string(),
            );
            continue;
        }
        anyhow::ensure!(fingerprint.is_none(), "{name}: a row after the trailer");
        rows.push(serde_json::from_value(v)?);
    }
    let fingerprint = fingerprint.ok_or_else(|| {
        anyhow::anyhow!("{name} has no trailer: the run did not finish, and a short dump is not a score")
    })?;
    Ok(Dump {
        header,
        rows,
        fingerprint,
    })
}

/// Two arms on the same problems.
#[derive(Debug, PartialEq)]
pub struct Pair {
    pub n: usize,
    pub right_a: usize,
    pub right_b: usize,
    /// Right in A only.
    pub only_a: usize,
    /// Right in B only.
    pub only_b: usize,
    /// Paired difference A − B as a rate, and its 95 % interval.
    pub delta: (f64, f64, f64),
    pub p_mcnemar: f64,
}

/// Grade a row again from its completion, with the rules of this module.
/// Returns whether anything changed.
///
/// `bin/gsm8kpair` calls it on every row of both dumps before pairing. A dump
/// written by `ops/gsm8k_vllm.py` arrives ungraded, and a dump written before
/// a rule changed arrives graded by the old rule: one grader then scores
/// every arm, whichever engine generated it.
pub fn regrade(row: &mut Row) -> anyhow::Result<bool> {
    let gold = canonical(&row.gold).ok_or_else(|| {
        anyhow::anyhow!("problem {}: gold {:?} is not a number", row.index, row.gold)
    })?;
    let (source, extracted) = extract(&row.completion);
    let correct = is_correct(extracted.as_deref(), &gold);
    let changed = row.gold != gold
        || row.extracted != extracted
        || row.source != source.name()
        || row.correct != correct;
    row.gold = gold;
    row.extracted = extracted;
    row.source = source.name().to_string();
    row.correct = correct;
    Ok(changed)
}

/// Pair two dumps problem by problem. Refuses two dumps that do not hold the
/// same problems in the same prompt tokens, or that grade a problem against
/// two different golds: the paired test is only valid on identical questions,
/// and `qhash` is what proves it, row by row.
pub fn pair(a: &Dump, b: &Dump) -> anyhow::Result<Pair> {
    use std::collections::BTreeMap;
    fn by_index(d: &Dump) -> anyhow::Result<BTreeMap<usize, &Row>> {
        let mut m = BTreeMap::new();
        for r in &d.rows {
            anyhow::ensure!(m.insert(r.index, r).is_none(), "problem {} appears twice", r.index);
        }
        Ok(m)
    }
    let (ma, mb) = (by_index(a)?, by_index(b)?);
    anyhow::ensure!(
        ma.keys().eq(mb.keys()),
        "the two dumps do not hold the same problems ({} and {}); pair runs of the same sample",
        ma.len(),
        mb.len()
    );
    let mut d: Vec<i8> = Vec::with_capacity(ma.len());
    let (mut right_a, mut right_b, mut only_a, mut only_b) = (0, 0, 0, 0);
    for (i, ra) in &ma {
        let rb = mb[i];
        anyhow::ensure!(
            ra.qhash == rb.qhash,
            "problem {i}: prompt hash {} against {}, the two arms were not asked the same tokens",
            ra.qhash,
            rb.qhash
        );
        anyhow::ensure!(
            canonical(&ra.gold) == canonical(&rb.gold),
            "problem {i}: gold {:?} against {:?}",
            ra.gold,
            rb.gold
        );
        right_a += usize::from(ra.correct);
        right_b += usize::from(rb.correct);
        let x = i8::from(ra.correct) - i8::from(rb.correct);
        only_a += usize::from(x > 0);
        only_b += usize::from(x < 0);
        d.push(x);
    }
    Ok(Pair {
        n: d.len(),
        right_a,
        right_b,
        only_a,
        only_b,
        delta: paired_interval(&d),
        p_mcnemar: mcnemar_exact(only_a, only_b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(index: usize, qhash: &str, correct: bool) -> Row {
        Row {
            index,
            qhash: qhash.into(),
            n_prompt: 10,
            n_gen: 5,
            stop: "eos".into(),
            gold: "1".into(),
            extracted: Some(if correct { "1" } else { "2" }.into()),
            source: "boxed".into(),
            correct,
            prefill_s: 0.0,
            decode_s: 0.0,
            completion: "\\boxed{1}".into(),
        }
    }

    fn dump_text(rows: &[Row], trailer: Option<usize>) -> String {
        let mut s = format!("{{\"{DUMP_TAG}\": {DUMP_VERSION}, \"model\": \"m\"}}\n");
        for r in rows {
            s.push_str(&serde_json::to_string(r).unwrap());
            s.push('\n');
        }
        if let Some(n) = trailer {
            s.push_str(&format!("{{\"end\": true, \"fingerprint\": \"ab\", \"questions\": {n}}}\n"));
        }
        s
    }

    #[test]
    fn a_dump_round_trips_and_a_short_one_is_refused() {
        let rows = vec![row(3, "aa", true), row(7, "bb", false)];
        let d = parse_dump(&dump_text(&rows, Some(2)), "t").unwrap();
        assert_eq!(d.rows, rows);
        assert_eq!(d.fingerprint, "ab");
        assert!(parse_dump(&dump_text(&rows, None), "t").is_err(), "no trailer");
        assert!(parse_dump(&dump_text(&rows, Some(3)), "t").is_err(), "wrong count");
        assert!(parse_dump("{\"other\": 1}\n", "t").is_err(), "no tag");
    }

    #[test]
    fn pairing_counts_the_discordant_problems() {
        let a = parse_dump(
            &dump_text(&[row(0, "a", true), row(1, "b", true), row(2, "c", false)], Some(3)),
            "a",
        )
        .unwrap();
        let b = parse_dump(
            &dump_text(&[row(0, "a", true), row(1, "b", false), row(2, "c", false)], Some(3)),
            "b",
        )
        .unwrap();
        let p = pair(&a, &b).unwrap();
        assert_eq!((p.n, p.right_a, p.right_b, p.only_a, p.only_b), (3, 2, 1, 1, 0));
        assert!((p.delta.0 - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn an_ungraded_row_is_graded_and_a_graded_one_is_left_alone() {
        let mut r = row(0, "a", false);
        r.extracted = None;
        r.source = "ungraded".into();
        r.gold = "1,000".into();
        r.completion = "so \\boxed{1000}".into();
        assert!(regrade(&mut r).unwrap(), "an ungraded row changes");
        assert_eq!((r.correct, r.gold.as_str(), r.source.as_str()), (true, "1000", "boxed"));
        assert!(!regrade(&mut r).unwrap(), "grading twice changes nothing");
        r.gold = "twelve".into();
        assert!(regrade(&mut r).is_err(), "a gold that is not a number is refused");
    }

    #[test]
    fn pairing_refuses_two_golds_for_one_problem() {
        let a = parse_dump(&dump_text(&[row(0, "a", true)], Some(1)), "a").unwrap();
        let mut other = row(0, "a", true);
        other.gold = "2".into();
        let b = parse_dump(&dump_text(&[other], Some(1)), "b").unwrap();
        assert!(pair(&a, &b).is_err());
    }

    #[test]
    fn pairing_refuses_different_prompts_or_problems() {
        let a = parse_dump(&dump_text(&[row(0, "a", true)], Some(1)), "a").unwrap();
        let other_tokens = parse_dump(&dump_text(&[row(0, "z", true)], Some(1)), "b").unwrap();
        let other_problem = parse_dump(&dump_text(&[row(1, "a", true)], Some(1)), "c").unwrap();
        assert!(pair(&a, &other_tokens).is_err());
        assert!(pair(&a, &other_problem).is_err());
    }

    #[test]
    fn the_gold_is_the_number_after_the_last_marker() {
        assert_eq!(gold("Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18").as_deref(), Some("18"));
        assert_eq!(gold("a #### 3\n#### 1,000").as_deref(), Some("1000"));
        assert_eq!(gold("#### -7").as_deref(), Some("-7"));
        assert_eq!(gold("no marker at all, 12"), None);
    }

    #[test]
    fn canonical_forms_agree_and_differ_where_they_should() {
        assert_eq!(canonical("18").as_deref(), Some("18"));
        assert_eq!(canonical("18.00").as_deref(), Some("18"));
        assert_eq!(canonical("018").as_deref(), Some("18"));
        assert_eq!(canonical("1,000.50").as_deref(), Some("1000.5"));
        assert_eq!(canonical("-0.0").as_deref(), Some("0"));
        assert_eq!(canonical("0.5").as_deref(), Some("0.5"));
        assert_eq!(canonical(".5").as_deref(), Some("0.5"));
        assert_ne!(canonical("18.5"), canonical("18"));
        assert_eq!(canonical("eighteen"), None);
        assert_eq!(canonical(""), None);
        assert_eq!(canonical("1.2.3"), None);
    }

    #[test]
    fn numbers_are_read_as_written() {
        assert_eq!(numbers("a 1,000 b"), vec!["1,000"]);
        assert_eq!(numbers("3, 4 and 5."), vec!["3", "4", "5"]);
        assert_eq!(numbers("10-2=8"), vec!["10", "-2", "8"]);
        assert_eq!(numbers("$18.50 each."), vec!["18.50"]);
        assert_eq!(numbers("no digits"), Vec::<&str>::new());
    }

    #[test]
    fn the_last_closed_box_wins() {
        assert_eq!(last_boxed("x \\boxed{3} then \\boxed{18}"), Some("18"));
        assert_eq!(last_boxed("\\boxed{\\frac{1}{2}}"), Some("\\frac{1}{2}"));
        assert_eq!(last_boxed("\\boxed{12"), None, "a box cut by the cap gives no answer");
        assert_eq!(last_boxed("no box"), None);
    }

    #[test]
    fn extraction_reads_the_box_first_and_the_text_second() {
        let (s, v) = extract("So she makes 9 * 2 = 18 dollars.\n\n\\boxed{18}");
        assert_eq!((s, v.as_deref()), (Source::Boxed, Some("18")));
        let (s, v) = extract("\\boxed{\\$1,080}");
        assert_eq!((s, v.as_deref()), (Source::Boxed, Some("1080")));
        let (s, v) = extract("\\boxed{18 \\text{ dollars}}");
        assert_eq!((s, v.as_deref()), (Source::Boxed, Some("18")));
        let (s, v) = extract("\\boxed{2 \\times 9 = 18}");
        assert_eq!((s, v.as_deref()), (Source::Boxed, Some("18")));
        let (s, v) = extract("The answer is 42.");
        assert_eq!((s, v.as_deref()), (Source::LastNumber, Some("42")));
        let (s, v) = extract("I cannot tell.");
        assert_eq!((s, v), (Source::None, None));
    }

    #[test]
    fn a_fraction_box_is_an_integer_or_nothing() {
        assert_eq!(extract("\\boxed{\\frac{36}{2}}").1.as_deref(), Some("18"));
        assert_eq!(extract("\\boxed{\\dfrac{3}{4}}"), (Source::Boxed, None));
        assert_eq!(extract("\\boxed{\\frac{3}{0}}"), (Source::Boxed, None));
    }

    #[test]
    fn a_truncated_box_falls_back_to_the_last_number() {
        // Cut inside the box: the box gives nothing and the fallback reads the
        // last number, the partial one. The dump carries `stop` and `source`,
        // so a strict re-grade that counts every `cap` as wrong needs no rerun.
        let (s, v) = extract("9 * 2 = 18, so \\boxed{1");
        assert_eq!((s, v.as_deref()), (Source::LastNumber, Some("1")));
    }

    #[test]
    fn grading_is_exact() {
        let g = gold("#### 18").unwrap();
        assert!(is_correct(extract("\\boxed{18.0}").1.as_deref(), &g));
        assert!(!is_correct(extract("\\boxed{18.5}").1.as_deref(), &g));
        assert!(!is_correct(None, &g));
    }

    #[test]
    fn the_sample_is_fixed_nested_and_sorted() {
        let a = select(1319, 50);
        assert_eq!(a.len(), 50);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(a, select(1319, 50), "same seed, same sample");
        let b = select(1319, 200);
        assert!(a.iter().all(|i| b.contains(i)), "a smaller sample nests in a larger one");
        assert_eq!(select(1319, 5000), (0..1319).collect::<Vec<_>>());
        assert_ne!(a, (0..50).collect::<Vec<_>>(), "a sample is not the head of the split");
    }

    #[test]
    fn mcnemar_matches_the_hand_computation() {
        // b = 0, c = 5: p = 2 · (1/2)^5 = 0.0625.
        assert!((mcnemar_exact(0, 5) - 0.0625).abs() < 1e-12);
        assert!((mcnemar_exact(3, 3) - 1.0).abs() < 1e-12);
        assert_eq!(mcnemar_exact(0, 0), 1.0);
    }

    #[test]
    fn the_paired_interval_is_centred_and_closes_on_agreement() {
        let (m, lo, hi) = paired_interval(&[0, 0, 0, 0]);
        assert_eq!((m, lo, hi), (0.0, 0.0, 0.0));
        let d: Vec<i8> = [vec![1i8; 30], vec![-1; 10], vec![0; 60]].concat();
        let (m, lo, hi) = paired_interval(&d);
        assert!((m - 0.2).abs() < 1e-12);
        assert!(lo < m && m < hi && lo > 0.0);
    }

    #[test]
    fn the_instruction_follows_the_problem_on_its_own_line() {
        assert_eq!(
            user_body("  How many?  "),
            format!("How many?\n{INSTRUCTION}")
        );
    }
}
