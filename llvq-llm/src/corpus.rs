//! Evaluation corpora.
//!
//! WikiText-2 (raw) is the perplexity reference every 2-bit quantization
//! paper reports, LLVQ included, so reproducing its exact preparation is not
//! optional: the standard is to join every row of the `test` split with a
//! blank line, tokenize the result once, and score **non-overlapping**
//! windows. Sliding windows or per-row scoring give systematically different
//! numbers and would make our figures incomparable to the paper's.

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

/// The revision every corpus here is read at.
///
/// All three call sites used to hard-code `main`, and `main` is a *branch*: a
/// Hugging Face dataset repo is mutable, so the same command two months later
/// can score other bytes and nothing in the log would say so. For a project
/// whose argument is reproducibility that is a hole, and it is a cheap one to
/// close — `LLVQ_DATASET_REV` pins a commit sha, a tag, any revision the Hub
/// resolves.
///
/// One variable for the three corpora, not one each: they are read by the same
/// harness within one run, and a per-dataset pin would let a figure be half
/// reproducible — worse than an unpinned one, because it looks pinned.
///
/// Unset or blank means `main`, i.e. exactly the previous behaviour. Nobody
/// has to change a command.
pub fn dataset_revision() -> String {
    revision_or_main(std::env::var("LLVQ_DATASET_REV").ok().as_deref())
}

/// The decision [`dataset_revision`] makes, separated from the environment so
/// it can be tested without mutating a process-wide variable.
fn revision_or_main(v: Option<&str>) -> String {
    match v {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => "main".to_string(),
    }
}

/// Fetch a split of a Hugging Face dataset stored as a single parquet shard
/// and return its text column, row by row.
pub fn hf_parquet_text(repo: &str, path: &str) -> anyhow::Result<Vec<String>> {
    let api = hf_hub::api::sync::Api::new()?;
    let file = api
        .repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Dataset,
            dataset_revision(),
        ))
        .get(path)?;
    let reader = SerializedFileReader::new(std::fs::File::open(file)?)?;
    let mut out = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        for (_, field) in row.get_column_iter() {
            if let Field::Str(s) = field {
                out.push(s.clone());
            }
        }
    }
    anyhow::ensure!(!out.is_empty(), "no text column found in {repo}/{path}");
    Ok(out)
}

/// One MMLU question: the stem, its four options, and the index of the right
/// one.
#[derive(Clone, Debug)]
pub struct MmluItem {
    pub subject: String,
    pub question: String,
    pub choices: [String; 4],
    /// 0..=3, matching `choices`.
    pub answer: usize,
}

/// A split of `cais/mmlu` (config `all`) — `test` for scoring, `dev` for the
/// five worked examples per subject that make the prompt 5-shot.
///
/// The column order is not assumed: parquet rows are read by name, because a
/// silent column swap between `question` and `subject` would still produce a
/// runnable prompt and a meaningless score.
pub fn mmlu_split(split: &str) -> anyhow::Result<Vec<MmluItem>> {
    let api = hf_hub::api::sync::Api::new()?;
    let file = api
        .repo(hf_hub::Repo::with_revision(
            "cais/mmlu".to_string(),
            hf_hub::RepoType::Dataset,
            dataset_revision(),
        ))
        .get(&format!("all/{split}-00000-of-00001.parquet"))?;
    let reader = SerializedFileReader::new(std::fs::File::open(file)?)?;
    let mut out = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let (mut subject, mut question, mut answer) = (None, None, None);
        let mut choices: Vec<String> = Vec::new();
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("question", Field::Str(s)) => question = Some(s.clone()),
                ("subject", Field::Str(s)) => subject = Some(s.clone()),
                // The label is an integer in `cais/mmlu`; some mirrors store
                // it as the letter instead, so both are accepted.
                ("answer", Field::Long(v)) => answer = Some(*v as usize),
                ("answer", Field::Int(v)) => answer = Some(*v as usize),
                ("answer", Field::Str(s)) => {
                    answer = "ABCD".find(s.trim());
                }
                ("choices", Field::ListInternal(list)) => {
                    for e in list.elements() {
                        if let Field::Str(s) = e {
                            choices.push(s.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        let (Some(subject), Some(question), Some(answer)) = (subject, question, answer) else {
            anyhow::bail!("mmlu/{split}: a row is missing question, subject or answer");
        };
        anyhow::ensure!(
            choices.len() == 4,
            "mmlu/{split}: {} options for {question:?}, expected 4",
            choices.len()
        );
        anyhow::ensure!(answer < 4, "mmlu/{split}: answer {answer} out of range");
        let choices: [String; 4] = choices.try_into().expect("checked length");
        out.push(MmluItem {
            subject,
            question,
            choices,
            answer,
        });
    }
    anyhow::ensure!(!out.is_empty(), "mmlu/{split} is empty");
    Ok(out)
}

/// The WikiText-2 raw test split, prepared the standard way.
pub fn wikitext2_test() -> anyhow::Result<String> {
    let rows = hf_parquet_text(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )?;
    Ok(rows.join("\n\n"))
}

/// What a slice of C4 is being asked for. The two roles must never read the
/// same text, which is why the shard is chosen here and nowhere else.
///
/// ## The trap this closes
///
/// `c4_validation` used to serve both. `bin/smoke` called it to build the
/// calibration set and `bin/ppl` called it — identical arguments — to build
/// the evaluation set, and the function is entirely deterministic: one
/// hard-coded shard, read from byte 0, documents in file order up to a
/// character budget. Both harnesses window from token 0 as well. So under
/// `LLVQ_CALIB=c4` with a C4 evaluation corpus, the first evaluation windows
/// **were** the calibration windows: GPTQ optimized the weights on the exact
/// text the perplexity then graded them on.
///
/// No published number was contaminated — `bin/smoke` always evaluates on
/// WikiText-2, and the C4 column of the 4B table belongs to a WikiText-calibrated
/// row — but the docstring on this function advertised it as *the
/// out-of-domain control*, the very property the configuration destroyed.
/// That is the kind of comment one trusts.
///
/// Disjointness is now structural rather than arithmetic: the roles read
/// different shards of a dataset whose shards partition it. An offset into one
/// shard would have worked too, and would have needed a reserve constant
/// larger than every future calibration budget — a promise nobody would
/// remember to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C4Role {
    /// Scoring. Keeps shard 0, so every C4 perplexity already published stays
    /// measured on exactly the same text.
    Evaluation,
    /// Building the Hessians.
    Calibration,
}

/// `allenai/c4`'s English validation split is published in 8 shards.
const C4_SHARDS: usize = 8;

/// The shard a role reads. Different by construction — see [`C4Role`].
pub fn c4_shard_path(role: C4Role) -> String {
    let i = match role {
        C4Role::Evaluation => 0,
        C4Role::Calibration => 1,
    };
    format!("en/c4-validation.{i:05}-of-{C4_SHARDS:05}.json.gz")
}

/// C4 (web crawl), validation split — the **out-of-domain** control.
///
/// Our calibration set is WikiText-2 *train* and our headline number is
/// measured on WikiText-2 *test*: same corpus, same style, same topics. GPTQ
/// therefore optimized the weights for text that looks like Wikipedia, and
/// then we graded it on Wikipedia. The paper avoids this by calibrating on
/// DCLM-edu.
///
/// Scoring the *same* quantized model on C4 says how much of the result is
/// the method and how much is the domain. If the degradation is comparable,
/// the number stands; if it jumps, the calibration set is doing the work.
pub fn c4_validation(max_chars: usize) -> anyhow::Result<String> {
    c4_text(C4Role::Evaluation, max_chars)
}

/// C4 text for building Hessians — a different shard from [`c4_validation`],
/// so calibrating and evaluating on "C4" can no longer be the same text.
pub fn c4_calibration(max_chars: usize) -> anyhow::Result<String> {
    c4_text(C4Role::Calibration, max_chars)
}

fn c4_text(role: C4Role, max_chars: usize) -> anyhow::Result<String> {
    use std::io::Read;
    let api = hf_hub::api::sync::Api::new()?;
    let file = api
        .repo(hf_hub::Repo::with_revision(
            "allenai/c4".to_string(),
            hf_hub::RepoType::Dataset,
            dataset_revision(),
        ))
        .get(&c4_shard_path(role))?;
    let mut gz = flate2::read::GzDecoder::new(std::fs::File::open(file)?);
    let mut raw = String::new();
    // The shard is far larger than we need; stop once we have enough text.
    let mut buf = vec![0u8; 1 << 20];
    while raw.len() < max_chars * 2 {
        let n = gz.read(&mut buf)?;
        if n == 0 {
            break;
        }
        raw.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    let mut out = String::new();
    for line in raw.lines() {
        // The last line of the buffer is very likely truncated.
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(t) = v["text"].as_str() {
            out.push_str(t);
            out.push_str("\n\n");
        }
        if out.len() >= max_chars {
            break;
        }
    }
    anyhow::ensure!(!out.is_empty(), "no usable text decoded from the C4 shard");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole split rests on. A refactor that collapses the
    /// two roles onto one shard — or a `c4_shard_path` that stops reading its
    /// argument — reinstates the self-grading configuration in silence.
    #[test]
    fn the_two_roles_read_different_shards() {
        assert_ne!(
            c4_shard_path(C4Role::Calibration),
            c4_shard_path(C4Role::Evaluation)
        );
    }

    /// Evaluation must stay on shard 0: every C4 perplexity in `CLAUDE.md` was
    /// measured there, and moving it would silently restate them.
    #[test]
    fn evaluation_keeps_the_published_shard() {
        assert_eq!(
            c4_shard_path(C4Role::Evaluation),
            "en/c4-validation.00000-of-00008.json.gz"
        );
    }

    /// The default has to stay `main`, byte for byte with the three hard-coded
    /// strings it replaced: every published perplexity was measured there, and
    /// a default that drifted would restate them without touching a number.
    #[test]
    fn an_unset_pin_means_main() {
        for v in [None, Some(""), Some("   ")] {
            assert_eq!(revision_or_main(v), "main", "{v:?}");
        }
    }

    /// And a set one must actually be carried — a pin that is read and dropped
    /// is worse than none, because the command line claims a revision.
    #[test]
    fn a_set_pin_is_carried() {
        assert_eq!(revision_or_main(Some("b3a7f01")), "b3a7f01");
        assert_eq!(revision_or_main(Some("  b3a7f01 ")), "b3a7f01");
    }

    /// Both paths have to be shards that exist. A formatting slip — a missing
    /// zero-pad, a wrong shard count — would only surface as a 404 hours into
    /// a run, on the machine of whoever tried to reproduce the number.
    #[test]
    fn both_paths_are_well_formed_shards() {
        for role in [C4Role::Evaluation, C4Role::Calibration] {
            let p = c4_shard_path(role);
            let i: usize = p["en/c4-validation.".len()..][..5]
                .parse()
                .unwrap_or_else(|_| panic!("{p} has no shard index"));
            assert!(i < C4_SHARDS, "{p} indexes past {C4_SHARDS} shards");
            assert!(p.ends_with(&format!("-of-{C4_SHARDS:05}.json.gz")), "{p}");
        }
    }
}
