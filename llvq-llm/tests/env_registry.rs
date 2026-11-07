//! Every `LLVQ_*` name the Rust code quotes is documented in `docs/ENV.md`.
//!
//! The audit of 2026-11-07 found 49 variables read by the code and absent from
//! `CLAUDE.md`, two of them set by 35 and 10 jobs. This test makes the registry
//! the place a new variable has to be written before it ships: it collects
//! every quoted literal `"LLVQ_<NAME>"` in the workspace's `.rs` files and
//! fails, naming the file, for each one the page does not mention.
//!
//! It reads files only, so it runs on every platform and in the CPU CI job.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Directories that hold no source of this workspace, or frozen copies of it.
const SKIP: [&str; 4] = ["target", "docs", "llvq-tetra", "node_modules"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            // Hidden directories include `.git` and `.claude/worktrees`, whose
            // other checkouts must not be read as this one.
            if !name.starts_with('.') && !SKIP.contains(&name.as_str()) {
                rust_files(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// Every `"LLVQ_<NAME>"` literal in `text`, where `<NAME>` is `[A-Z0-9_]+`.
fn quoted_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("\"LLVQ_") {
        let after = &rest[i + 1..];
        let len = after.bytes().take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_').count();
        if len > "LLVQ_".len() && after.as_bytes().get(len) == Some(&b'"') {
            names.push(after[..len].to_string());
        }
        rest = &after[len.max(1)..];
    }
    names
}

#[test]
fn the_scanner_reads_quoted_names_only() {
    let text = r#"var("LLVQ_ONE") "LLVQ_TWO_2" "LLVQ_" "LLVQ_three" "LLVQ_FOUR is a sentence" LLVQ_FIVE"#;
    assert_eq!(quoted_names(text), ["LLVQ_ONE", "LLVQ_TWO_2"]);
}

#[test]
fn every_quoted_variable_is_in_the_registry() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let registry = std::fs::read_to_string(root.join("docs/ENV.md")).expect("docs/ENV.md");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    assert!(files.len() > 300, "found only {} .rs files under {}", files.len(), root.display());

    let mut missing: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut seen = 0usize;
    for f in &files {
        // This file quotes sample names to test the scanner.
        if f.ends_with("tests/env_registry.rs") {
            continue;
        }
        let text = std::fs::read_to_string(f).unwrap_or_default();
        for name in quoted_names(&text) {
            seen += 1;
            if !registry.contains(&format!("`{name}`")) {
                missing.entry(name).or_insert_with(|| f.strip_prefix(&root).unwrap_or(f).to_path_buf());
            }
        }
    }
    assert!(seen > 100, "only {seen} quoted LLVQ_ names: the scan is broken");
    assert!(
        missing.is_empty(),
        "documented nowhere in docs/ENV.md (write a row, in backquotes):\n{}",
        missing.iter().map(|(n, f)| format!("  {n}  (first seen in {})", f.display())).collect::<Vec<_>>().join("\n")
    );
}
