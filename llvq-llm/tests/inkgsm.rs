//! `bin/inkgsm` against its two references, on the checkpoint's files. Both carry an unconditional `#[ignore]`, as
//! the sealed-archive tests do, and fail naming the file they lack.
//!
//! - The prompt ids `inkgsm prompts` renders for the 200 problems equal, id for id, those of the checkpoint's own
//!   template rendered by `transformers` (`ops/inkling/gsm8k_prompt_ref.py`, written to
//!   `$INKLING_DATA/lot4-g8/gsm8k_prompt_ref_200.json`).
//! - `ops/data/spec-gsm8k-200.jsonl` is rows 0 to 199 of the GSM8K test parquet at revision 740312ad, built as
//!   `bin/gsm8k` builds a problem (`gsm8k::user_body`, `gsm8k::gold`); a copy with one gold changed is refused.

use std::path::{Path, PathBuf};
use std::process::Command;

fn data_dir() -> PathBuf {
    match std::env::var_os("INKLING_DATA") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("llvq-inkling-mini"),
    }
}

fn need(p: &Path) -> &Path {
    assert!(p.exists(), "missing {}: this test needs it", p.display());
    p
}

fn problems() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ops/data/spec-gsm8k-200.jsonl")
}

fn parquet() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(
        ".cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/\
         test-00000-of-00001.parquet",
    )
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("inkgsm-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn inkgsm(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_inkgsm")).args(args).output().expect("run inkgsm")
}

#[test]
#[ignore = "needs ~/llvq-inkling-mini/native/tokenizer.json and lot4-g8/gsm8k_prompt_ref_200.json"]
fn prompt_ids_equal_the_reference_template() {
    let d = data_dir();
    let tok = d.join("native/tokenizer.json");
    let refp = d.join("lot4-g8/gsm8k_prompt_ref_200.json");
    let out = scratch("ids.json");
    let o = inkgsm(&[
        "prompts",
        "--tokenizer",
        need(&tok).to_str().unwrap(),
        "--problems",
        problems().to_str().unwrap(),
        "--limit",
        "200",
        "--effort",
        "none",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "inkgsm prompts: {}", String::from_utf8_lossy(&o.stderr));
    let got: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let want: serde_json::Value = serde_json::from_slice(&std::fs::read(need(&refp)).unwrap()).unwrap();
    assert_eq!(want["meta"]["effort"], "none");
    assert_eq!(got["tokenizer_sha256"], want["meta"]["tokenizer_sha256"], "another tokenizer.json");
    let (g, w) = (got["cases"].as_array().unwrap(), want["cases"].as_array().unwrap());
    assert_eq!(g.len(), 200);
    assert_eq!(w.len(), 200);
    for (a, b) in g.iter().zip(w) {
        assert_eq!(a["index"], b["index"]);
        assert_eq!(a["content"], b["content"], "problem {}: another user turn", a["index"]);
        assert_eq!(a["ids"], b["ids"], "problem {}: prompt ids differ from the template's", a["index"]);
        assert_eq!(a["qhash"], b["qhash"], "problem {}: qhash", a["index"]);
    }
}

#[test]
#[ignore = "needs ~/llvq-inkling-mini/native/tokenizer.json and the GSM8K test parquet in the Hub cache"]
fn the_problems_file_is_the_head_of_the_test_split() {
    let tok = data_dir().join("native/tokenizer.json");
    let pq = parquet();
    let run = |file: &Path| {
        inkgsm(&[
            "prompts",
            "--tokenizer",
            need(&tok).to_str().unwrap(),
            "--problems",
            file.to_str().unwrap(),
            "--limit",
            "200",
            "--parquet",
            need(&pq).to_str().unwrap(),
            "--out",
            scratch("parquet-ids.json").to_str().unwrap(),
        ])
    };
    let o = run(&problems());
    assert!(o.status.success(), "the problems file against the parquet: {}", String::from_utf8_lossy(&o.stderr));

    // The same file with the gold of row 117 changed: refused, naming the row.
    let text = std::fs::read_to_string(problems()).unwrap();
    let mut changed = 0;
    let lines: Vec<String> = text
        .lines()
        .map(|l| {
            if l.contains("\"gsm8k-test-0117\"") {
                let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
                let g: i64 = v["gold"].as_str().unwrap().parse().unwrap();
                v["gold"] = serde_json::json!((g + 1).to_string());
                changed += 1;
                v.to_string()
            } else {
                l.to_string()
            }
        })
        .collect();
    assert_eq!(changed, 1);
    let bad = scratch("bad-problems.jsonl");
    std::fs::write(&bad, lines.join("\n")).unwrap();
    let o = run(&bad);
    assert!(!o.status.success(), "a changed gold must be refused");
    assert!(String::from_utf8_lossy(&o.stderr).contains("row 117"), "{}", String::from_utf8_lossy(&o.stderr));
}
