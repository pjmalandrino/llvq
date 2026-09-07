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
use parquet::schema::types::Type as SchemaType;
use std::path::{Path, PathBuf};

/// The revision every corpus here is read at.
///
/// All three call sites used to hard-code `main`, and `main` is a *branch*: a
/// Hugging Face dataset repo is mutable, so the same command two months later
/// can score other bytes and nothing in the log would say so. For a project
/// whose argument is reproducibility that is a hole, and it is a cheap one to
/// close — `LLVQ_DATASET_REV` pins a commit sha, a tag, any revision the Hub
/// resolves.
///
/// One variable for every corpus here, not one each: they are read by the same
/// harness within one run, and a per-dataset pin would let a figure be half
/// reproducible, worse than an unpinned one because it looks pinned.
///
/// ## What that costs, now that four repositories share the variable
///
/// A sha is valid in exactly one repository. `bin/smoke` reads
/// `Salesforce/wikitext` for its reference perplexity before it touches the
/// calibration corpus, so a run set to a `HuggingFaceTB/dclm-edu` commit dies
/// on a 404 before the first Hessian. The only value that works across a run
/// is therefore `main`, a branch. `ARTIFACT-EVALUATION.md` states the same
/// limit for the reviewer.
///
/// Since the pin cannot be used, the revision actually read is recorded
/// instead: [`BoundedRead::revision`] carries the commit sha `hf_hub` resolved.
/// That does not make a figure reproducible, it makes it auditable.
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
///
/// Reads the whole column into memory, which is why it is only used on the
/// WikiText-2 splits, tens of megabytes each. A shard of a pretraining corpus
/// is three orders of magnitude larger; read those with
/// `parquet_text_bounded`.
pub fn hf_parquet_text(repo: &str, path: &str) -> anyhow::Result<Vec<String>> {
    let file = hf_dataset_file(repo, path)?;
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

/// `HuggingFaceTB/dclm-edu`, the corpus the LLVQ paper calibrates on.
///
/// Section 5 of the paper builds its layer-wise Hessians on 6,100 sequences of
/// DCLM-edu. We build ours on WikiText-2 or C4, and read 16.94 perplexity for
/// 55.59 MMLU where the paper reads 17.05 for 60.7: better on likelihood, 5.1
/// points worse on the exam. DCLM-edu is web text filtered for educational
/// content, which is the domain MMLU examines, so the calibration corpus is a
/// candidate for that dissociation.
///
/// Calibration only. Nothing is ever scored on DCLM-edu, so the shard split
/// [`C4Role`] enforces has no counterpart here.
pub const DCLM_EDU_REPO: &str = "HuggingFaceTB/dclm-edu";

/// The shard [`dclm_edu_calibration`] reads. The dataset is published in 1,771
/// parquet shards; this one holds 776,000 rows in a single row group, 4.68 G
/// characters of text, and 2 905 491 151 bytes on disk (*measured*,
/// 2026-09-07).
///
/// The largest volume this harness asks for is 25.2 M characters, so the shard
/// carries 186 times it. In tokens the factor is 246: the Qwen3-4B tokenizer
/// reads 4.5356 bytes per token over the 25 MB actually consumed (*measured*,
/// 2026-09-07), which puts the shard at about 1.03 G tokens for a need of
/// 4.19 M. The character figure is the one the code compares, and it needs no
/// tokenizer to be checked.
pub const DCLM_EDU_SHARD: &str = "data/000_00000.parquet";

/// Resolve one file of a Hugging Face dataset at the pinned revision.
///
/// Every corpus here goes through it, so [`dataset_revision`] cannot be
/// forgotten at a new call site.
fn hf_dataset_file(repo: &str, path: &str) -> anyhow::Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    Ok(api
        .repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Dataset,
            dataset_revision(),
        ))
        .get(path)?)
}

/// What a bounded read touched, against what the file holds.
///
/// Returned rather than printed: the caller writes it into the run journal,
/// and a test asserts on it. A bound nobody can read back is a bound nobody
/// trusts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedRead {
    /// Bytes of text returned.
    pub chars: usize,
    /// Bytes of text asked for.
    pub budget: usize,
    pub rows_read: usize,
    pub rows_total: usize,
    pub row_groups_read: usize,
    pub row_groups_total: usize,
    /// The commit the bytes came from, when the file is in the Hugging Face
    /// cache. See [`snapshot_revision`] and [`dataset_revision`].
    pub revision: Option<String>,
}

impl BoundedRead {
    /// One line for the run journal.
    pub fn report(&self) -> String {
        format!(
            "bounded read: {} chars for {} asked, {} of {} rows, {} of {} row groups, revision {}",
            self.chars,
            self.budget,
            self.rows_read,
            self.rows_total,
            self.row_groups_read,
            self.row_groups_total,
            self.revision.as_deref().unwrap_or("not from the Hub cache")
        )
    }
}

/// The commit a Hugging Face cache path was resolved to.
///
/// `hf_hub` lays a download down as `…/snapshots/<sha>/<path in repo>`, and
/// that sha is the commit the bytes really came from, whatever revision was
/// asked for. Since [`dataset_revision`] cannot pin a corpus once a run reads
/// two repositories, digging the sha back out of the path is what turns an
/// unpinned figure into an auditable one.
///
/// `None` for anything that is not under a `snapshots/<rev>` directory, which
/// covers every file the tests below write.
fn snapshot_revision(path: &Path) -> Option<String> {
    let parts: Vec<_> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let i = parts.iter().position(|p| p == "snapshots")?;
    parts.get(i + 1).cloned()
}

/// The `column` text column of a parquet file, up to `max_chars`, and not one
/// row further.
///
/// [`hf_parquet_text`] returns the whole column. On a DCLM-edu shard that is a
/// `Vec<String>` of 4.68 G characters for a need of 25.2 M, and the process
/// dies before the first Hessian. This one stops.
///
/// ## The bound is on memory and on decoding, not on bytes fetched
///
/// The whole shard still lands on disk. [`hf_dataset_file`] asks `hf_hub` for
/// a file, and `hf_hub` downloads files whole: 2.91 GB for DCLM-edu against
/// 40.7 MB for the C4 calibration shard, a factor of 71. A container running
/// this arm pays that download and needs the disk for it, on top of the
/// checkpoint and the artifact. What stays small is the resident text and the
/// decompression work, which is what killed a job here before.
///
/// The stop is per **row**, not per row group. The shard carries a single row
/// group of 776,000 rows and 4.83 GB uncompressed (*measured*, 2026-09-07), so
/// a loop testing its budget only at group boundaries would read all of it.
/// Row groups are still opened one at a time, so a file laid out in many
/// groups never has its later groups touched.
///
/// Two more things keep the read small. Only `column` is projected, so the
/// seven other columns of a DCLM-edu row are never decompressed. And the last
/// row is kept whole: the budget is a floor to cross, not a length to
/// truncate to, because a torn last document would land a broken sentence on a
/// window boundary for nothing.
///
/// `max_chars` counts UTF-8 bytes, as `c4_text` does, and rows are joined with
/// a blank line, as every other corpus here is.
fn parquet_text_bounded(
    path: &Path,
    column: &str,
    max_chars: usize,
) -> anyhow::Result<(String, BoundedRead)> {
    let reader = SerializedFileReader::new(std::fs::File::open(path)?)?;
    let meta = reader.metadata();
    let row_groups_total = meta.num_row_groups();
    let rows_total = meta.file_metadata().num_rows().max(0) as usize;
    let projection = one_column_projection(meta.file_metadata().schema(), column, path)?;

    let mut out = String::new();
    let (mut rows_read, mut row_groups_read) = (0usize, 0usize);
    for g in 0..row_groups_total {
        if out.len() >= max_chars {
            break;
        }
        row_groups_read += 1;
        let group = reader.get_row_group(g)?;
        for row in group.get_row_iter(Some(projection.clone()))? {
            let row = row?;
            rows_read += 1;
            for (_, field) in row.get_column_iter() {
                if let Field::Str(s) = field {
                    out.push_str(s);
                    out.push_str("\n\n");
                }
            }
            if out.len() >= max_chars {
                break;
            }
        }
    }

    let stats = BoundedRead {
        chars: out.len(),
        budget: max_chars,
        rows_read,
        rows_total,
        row_groups_read,
        row_groups_total,
        revision: snapshot_revision(path),
    };
    // A short file fails here rather than three hours later. `bin/smoke`
    // refuses to serve fewer calibration windows than were asked for; a corpus
    // that cannot fill the request must say so with the same voice.
    anyhow::ensure!(
        stats.chars >= max_chars,
        "{}: {} — the file is exhausted and the budget is not met",
        path.display(),
        stats.report()
    );
    Ok((out, stats))
}

/// A projection over one named column of `root`, so the row iterator decodes
/// nothing else.
fn one_column_projection(
    root: &SchemaType,
    column: &str,
    path: &Path,
) -> anyhow::Result<SchemaType> {
    let field = root
        .get_fields()
        .iter()
        .find(|f| f.name() == column)
        .ok_or_else(|| anyhow::anyhow!("no `{column}` column in {}", path.display()))?
        .clone();
    Ok(SchemaType::group_type_builder(root.name())
        .with_fields(vec![field])
        .build()?)
}

/// DCLM-edu text for building Hessians, read up to `max_chars` and no further.
///
/// The paper's calibration set (see [`DCLM_EDU_REPO`]). Returns the text and
/// what it cost to get it, the resolved commit included.
///
/// Two costs, and only one of them is bounded. Decoding stops at `max_chars`;
/// the download does not, so the caller pays 2.91 GB of disk for the shard
/// whatever it asks for.
pub fn dclm_edu_calibration(max_chars: usize) -> anyhow::Result<(String, BoundedRead)> {
    let file = hf_dataset_file(DCLM_EDU_REPO, DCLM_EDU_SHARD)?;
    parquet_text_bounded(&file, "text", max_chars)
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
    let file = hf_dataset_file("cais/mmlu", &format!("all/{split}-00000-of-00001.parquet"))?;
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
    let file = hf_dataset_file("allenai/c4", &c4_shard_path(role))?;
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

    /// A parquet file shaped like a DCLM-edu shard — a `text` column and one
    /// other, several row groups — small enough for the fast loop.
    ///
    /// Built rather than downloaded: the real shard is 2.9 GB, and a test that
    /// pulls it cannot run in the loop that would catch a regression here.
    fn write_tiny_parquet(
        path: &Path,
        groups: usize,
        rows_per_group: usize,
        row_chars: usize,
    ) -> anyhow::Result<()> {
        use parquet::data_type::{ByteArray, ByteArrayType};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use std::sync::Arc;

        let schema = Arc::new(parquet::schema::parser::parse_message_type(
            "message shard { REQUIRED BYTE_ARRAY text (UTF8); REQUIRED BYTE_ARRAY url (UTF8); }",
        )?);
        let props = Arc::new(WriterProperties::builder().build());
        let mut writer = SerializedFileWriter::new(std::fs::File::create(path)?, schema, props)?;
        for g in 0..groups {
            let mut group = writer.next_row_group()?;
            let texts: Vec<ByteArray> = (0..rows_per_group)
                .map(|r| {
                    let head = format!("g{g}r{r} ");
                    let mut s = head.repeat(row_chars / head.len() + 1);
                    s.truncate(row_chars);
                    ByteArray::from(s.as_str())
                })
                .collect();
            let urls: Vec<ByteArray> = (0..rows_per_group)
                .map(|r| ByteArray::from(format!("https://example.invalid/{g}/{r}").as_str()))
                .collect();
            for column in [texts, urls] {
                let mut w = group.next_column()?.expect("two columns in the schema");
                w.typed::<ByteArrayType>()
                    .write_batch(&column, None, None)?;
                w.close()?;
            }
            group.close()?;
        }
        writer.close()?;
        Ok(())
    }

    /// A path in the system temp directory, unique to this process and tag.
    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llvq-bounded-{}-{tag}.parquet", std::process::id()))
    }

    /// The property the whole variant rests on: the read stops. The DCLM-edu
    /// shard is 2.9 GB on disk for a need of ~17 MB, and reading it whole —
    /// what [`hf_parquet_text`] does — kills the process.
    ///
    /// Row groups are the coarse half of the bound and rows the fine half. The
    /// assertion covers both: one group out of eight, and a handful of rows
    /// out of that group's sixteen.
    #[test]
    fn a_bounded_read_stops_well_before_the_end() -> anyhow::Result<()> {
        let path = scratch("stops");
        write_tiny_parquet(&path, 8, 16, 1000)?;
        let (text, stats) = parquet_text_bounded(&path, "text", 3000)?;
        std::fs::remove_file(&path)?;

        assert_eq!(stats.row_groups_total, 8);
        assert_eq!(stats.rows_total, 8 * 16);
        assert_eq!(stats.row_groups_read, 1, "{}", stats.report());
        assert_eq!(stats.rows_read, 3, "{}", stats.report());
        assert!(stats.chars >= 3000, "{}", stats.report());
        assert_eq!(stats.chars, text.len());
        // Read from the front, in file order, and nothing from group 1.
        assert!(text.starts_with("g0r0 "), "{:?}", &text[..16]);
        assert!(!text.contains("g1r0 "), "{}", stats.report());
        Ok(())
    }

    /// The commit is dug out of the cache path and carried into the report.
    ///
    /// This is what stands in for a pin: `LLVQ_DATASET_REV` cannot pin one
    /// corpus of a run that reads four repositories, so a figure is auditable
    /// only if the run journal says which commit it read. A refactor that drops
    /// the field would take a published number's provenance with it.
    #[test]
    fn the_resolved_commit_is_recovered_from_the_cache_path() {
        let sha = "dbad8ad71224482740cd9c9d353591adbf62fe04";
        let hub = PathBuf::from("/home/u/.cache/huggingface/hub");
        let p = hub
            .join("datasets--HuggingFaceTB--dclm-edu/snapshots")
            .join(sha)
            .join("data/000_00000.parquet");
        assert_eq!(snapshot_revision(&p).as_deref(), Some(sha));
        // A branch name resolves to a directory too, and it is just as much
        // what was read.
        let p = hub.join("datasets--x--y/snapshots/main/data/a.parquet");
        assert_eq!(snapshot_revision(&p).as_deref(), Some("main"));
        // Anything outside the cache says so instead of inventing a commit.
        assert_eq!(snapshot_revision(Path::new("/tmp/a.parquet")), None);
        assert_eq!(snapshot_revision(Path::new("/tmp/snapshots")), None);
    }

    /// And the report prints it, in both cases. A field nobody prints is a
    /// field nobody reads back.
    #[test]
    fn the_report_names_the_revision() -> anyhow::Result<()> {
        let path = scratch("revision");
        write_tiny_parquet(&path, 1, 4, 500)?;
        let (_, stats) = parquet_text_bounded(&path, "text", 600)?;
        std::fs::remove_file(&path)?;
        assert_eq!(stats.revision, None);
        assert!(
            stats.report().contains("not from the Hub cache"),
            "{}",
            stats.report()
        );

        let pinned = BoundedRead {
            revision: Some("dbad8ad7".into()),
            ..stats
        };
        assert!(
            pinned.report().contains("revision dbad8ad7"),
            "{}",
            pinned.report()
        );
        Ok(())
    }

    /// Only the asked-for column is decoded. A projection dropped in a refactor
    /// would cost the seven other DCLM-edu columns on every row, and nothing
    /// downstream would fail.
    #[test]
    fn a_bounded_read_projects_one_column() -> anyhow::Result<()> {
        let path = scratch("project");
        write_tiny_parquet(&path, 2, 4, 500)?;
        let (text, _) = parquet_text_bounded(&path, "text", 600)?;
        std::fs::remove_file(&path)?;
        assert!(
            !text.contains("example.invalid"),
            "the url column leaked in"
        );
        Ok(())
    }

    /// A file that cannot fill the budget fails, and the error carries the
    /// count. `bin/smoke` refuses to serve fewer windows than were asked for;
    /// a corpus that comes up short must fail with the same voice, not hand
    /// back a shorter string.
    #[test]
    fn a_bounded_read_refuses_to_come_up_short() -> anyhow::Result<()> {
        let path = scratch("short");
        write_tiny_parquet(&path, 2, 2, 100)?;
        let err = parquet_text_bounded(&path, "text", 10_000).unwrap_err();
        std::fs::remove_file(&path)?;
        let msg = err.to_string();
        assert!(msg.contains("4 of 4 rows"), "{msg}");
        assert!(msg.contains("2 of 2 row groups"), "{msg}");
        Ok(())
    }

    /// A missing text column names itself. The alternative is an empty string
    /// and a perplexity nobody can explain.
    #[test]
    fn a_missing_column_is_named() -> anyhow::Result<()> {
        let path = scratch("column");
        write_tiny_parquet(&path, 1, 2, 100)?;
        let err = parquet_text_bounded(&path, "contents", 10).unwrap_err();
        std::fs::remove_file(&path)?;
        assert!(err.to_string().contains("`contents`"), "{err}");
        Ok(())
    }

    /// The same bound, on the real 2.91 GB shard. Gated to release like every
    /// other heavy test here: it needs the file in the Hugging Face cache, and
    /// downloads it whole if it is not there.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_dclm_shard_is_read_bounded() -> anyhow::Result<()> {
        let budget = 1 << 20;
        let (text, stats) = dclm_edu_calibration(budget)?;
        eprintln!("{}", stats.report());
        assert_eq!(text.len(), stats.chars);
        assert!(stats.chars >= budget, "{}", stats.report());
        // The shard is one row group of 776,000 rows. What proves the bound is
        // the row count: a read that walked the file would report all of them.
        assert!(
            stats.rows_read * 100 < stats.rows_total,
            "{}",
            stats.report()
        );
        // And the run must be able to say afterwards which commit it read,
        // since it cannot say beforehand which one it wants.
        let rev = stats.revision.clone().expect("a Hub cache path");
        assert!(!rev.is_empty(), "{}", stats.report());
        Ok(())
    }

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
