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

/// Fetch a split of a Hugging Face dataset stored as a single parquet shard
/// and return its text column, row by row.
pub fn hf_parquet_text(repo: &str, path: &str) -> anyhow::Result<Vec<String>> {
    let api = hf_hub::api::sync::Api::new()?;
    let file = api
        .repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Dataset,
            "main".to_string(),
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
            "main".to_string(),
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
                    answer = "ABCD".find(s.trim()).map(|i| i);
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
    use std::io::Read;
    let api = hf_hub::api::sync::Api::new()?;
    let file = api
        .repo(hf_hub::Repo::with_revision(
            "allenai/c4".to_string(),
            hf_hub::RepoType::Dataset,
            "main".to_string(),
        ))
        .get("en/c4-validation.00000-of-00008.json.gz")?;
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
