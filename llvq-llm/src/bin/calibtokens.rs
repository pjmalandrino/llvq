//! # `calibtokens`: the encoder's calibration token ids, and nothing else
//!
//! Writes the ids `bin/smoke` calibrates on, cut by the same function
//! (`llvq_llm::guided::calibration_ids`), as little-endian u32 in window order, with a JSON sidecar
//! carrying their FNV-1a 64 fingerprint and sha256. `ops/guided_weights.py` reads that file and
//! computes its per-token weights on exactly these ids; `smoke` refuses the weights unless their
//! manifest carries the fingerprint of its own windows.
//!
//! Reads the tokenizer and the corpus. It maps no weight and loads no model: the checkpoint is only
//! resolved for its `tokenizer.json`.
//!
//! ```text
//! LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=dclm-edu cargo run --release -p llvq-llm --bin calibtokens -- \
//!   64 2048 calib-tokens.u32
//! ```
//!
//! Environment: `LLVQ_MODEL` (default `Qwen/Qwen3-0.6B`, as `smoke`), `LLVQ_CALIB`,
//! `LLVQ_CALIB_SEED`, each read as `smoke` reads it.

use llvq_llm::corpus::CalibCorpus;
use llvq_llm::loader::Checkpoint;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        a.len() == 4,
        "usage: calibtokens <n_calib> <calib_len> <out.u32>   (LLVQ_MODEL, LLVQ_CALIB, LLVQ_CALIB_SEED)"
    );
    let n_calib: usize = a[1].parse().map_err(|e| anyhow::anyhow!("n_calib {:?}: {e}", a[1]))?;
    let calib_len: usize = a[2].parse().map_err(|e| anyhow::anyhow!("calib_len {:?}: {e}", a[2]))?;
    let out = std::path::PathBuf::from(&a[3]);
    anyhow::ensure!(n_calib > 0 && calib_len > 0, "n_calib and calib_len must be positive");

    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let calib_env = std::env::var("LLVQ_CALIB").ok();
    let calib = CalibCorpus::parse(calib_env.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    let calib_seed = match std::env::var("LLVQ_CALIB_SEED") {
        Ok(s) if !s.is_empty() => Some(
            s.parse::<u64>()
                .map_err(|e| anyhow::anyhow!("LLVQ_CALIB_SEED={s:?} is not an integer: {e}"))?,
        ),
        _ => None,
    };

    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let tok_sha = llvq_llm::digest::sha256_file(&ck.tokenizer)?;
    eprintln!("tokenizer {} sha256 {tok_sha}", ck.tokenizer.display());
    eprintln!("tokenizing calibration set ({})…", calib.name());
    let cal = llvq_llm::guided::calibration_ids(&tok, calib, n_calib, calib_len, calib_seed)?;
    let fp = cal.fingerprint();
    let sha = llvq_llm::guided::write_token_file(
        &out,
        &cal.ids,
        n_calib,
        calib_len,
        &[
            ("model", repo.clone()),
            ("calibration", calib.name().to_string()),
            (
                "calib_seed",
                match calib_seed {
                    Some(s) => s.to_string(),
                    None => "prefix".into(),
                },
            ),
            ("tokens_read", cal.read.to_string()),
            ("windows_available", cal.available.to_string()),
            ("tokenizer_sha256", tok_sha),
        ],
    )?;
    eprintln!(
        "  {n_calib} windows of {calib_len} = {} tokens ({} available)",
        cal.ids.len(),
        cal.available
    );
    println!("calibration token fingerprint {fp:016x} ({} ids)", cal.ids.len());
    println!("wrote {} sha256 {sha}", out.display());
    Ok(())
}
