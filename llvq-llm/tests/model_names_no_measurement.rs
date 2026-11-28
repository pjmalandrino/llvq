//! The served forward pass, `llvq-llm/src/model/`, names no measurement module in its code.
//!
//! Batches 4 and 9 of the architecture audit took two measurement modes out of the forward pass,
//! and two more (`LLVQ_KV_LADDER`, `LLVQ_KV_CAPTURE`) went back in within three weeks. They now
//! reach it through hooks the model declares and the measurement modules implement (`Capture`,
//! `KvRoundTrip`). This test is what stops the next one: it fails, naming the file and the line,
//! when code under `src/model/` names a module of `src/` whose header declares it measurement only.
//!
//! The list of such modules is read from their headers, not written here: a module is measurement
//! only when its leading `//!` block carries one of [`MARKERS`]. Comments are not code, so a doc line
//! that names such a module (to say who implements a hook) passes. It reads files only, so it runs
//! on every platform and in the CPU CI job.

use std::path::{Path, PathBuf};

/// The phrases by which a module's header declares it off the served path.
const MARKERS: [&str; 3] = ["MEASUREMENT ONLY", "A MEASUREMENT MODE", "Measurement code only"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The leading `//!` block of a file.
fn header(text: &str) -> String {
    text.lines().take_while(|l| l.starts_with("//!")).collect::<Vec<_>>().join("\n")
}

/// The module a file under `src/` defines, by its path: `kvladder.rs` is `kvladder`,
/// `persist4e_cuda/mod.rs` is `persist4e_cuda`. Only modules of the crate root count; a submodule
/// is reached through its parent's name, which this test already watches.
fn root_module(src: &Path, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(src).ok()?;
    let parts: Vec<_> = rel.iter().map(|s| s.to_string_lossy().into_owned()).collect();
    match parts.as_slice() {
        [f] if f != "lib.rs" => Some(f.trim_end_matches(".rs").to_string()),
        [d, m] if m == "mod.rs" && d != "bin" => Some(d.clone()),
        _ => None,
    }
}

/// The crate's measurement-only modules, from their headers.
fn measurement_modules(src: &Path) -> Vec<String> {
    let mut files = Vec::new();
    rust_files(src, &mut files);
    let mut out: Vec<String> = files
        .iter()
        .filter_map(|f| {
            let name = root_module(src, f)?;
            let h = header(&std::fs::read_to_string(f).ok()?);
            MARKERS.iter().any(|m| h.contains(m)).then_some(name)
        })
        .collect();
    out.sort();
    out
}

/// Every line of `text` whose code (the part before any `//`) names `module` as a whole
/// identifier, with its 1-based number.
fn code_lines_naming(text: &str, module: &str) -> Vec<(usize, String)> {
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut hits = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let code = line.find("//").map_or(line, |i| &line[..i]);
        let bytes = code.as_bytes();
        let mut from = 0;
        while let Some(i) = code[from..].find(module) {
            let (s, e) = (from + i, from + i + module.len());
            let before = s == 0 || !is_ident(bytes[s - 1]);
            let after = e == bytes.len() || !is_ident(bytes[e]);
            if before && after {
                hits.push((n + 1, line.trim().to_string()));
                break;
            }
            from = e;
        }
    }
    hits
}

#[test]
fn the_scanner_reads_code_not_comments() {
    let text = "use crate::kvladder::LadderAux;\n\
                /// [`crate::kvladder`] implements it.\n\
                let x = 1; // kvladder\n\
                let kvladder_like = 2;\n\
                fn f(_: crate::kvladder::Ladder) {}\n";
    let hits: Vec<usize> = code_lines_naming(text, "kvladder").into_iter().map(|(n, _)| n).collect();
    assert_eq!(hits, [1, 5]);
    assert_eq!(header("//! a\n//! MEASUREMENT ONLY\nuse x;\n//! not header"), "//! a\n//! MEASUREMENT ONLY");
}

#[test]
fn the_served_forward_pass_names_no_measurement_module() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let modules = measurement_modules(&src);
    // The two the audit found, read back from their own headers: if either is missing the scan is
    // broken, or the module was renamed or deleted and this line must follow it.
    for known in ["kvcapture", "kvladder"] {
        assert!(modules.iter().any(|m| m == known), "{known} not found measurement only among {modules:?}");
    }

    let mut files = Vec::new();
    rust_files(&src.join("model"), &mut files);
    assert!(files.len() >= 8, "found only {} files under src/model", files.len());
    files.sort();

    let mut found = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).expect("reads");
        for m in &modules {
            for (n, line) in code_lines_naming(&text, m) {
                found.push(format!("{}:{n}: names `{m}`: {line}", f.strip_prefix(&src).unwrap_or(f).display()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "the served forward pass names a measurement-only module; give the mode a hook the model \
         declares (as `Capture` or `KvRoundTrip`) and implement it in the module's own file:\n{}",
        found.join("\n")
    );
}
