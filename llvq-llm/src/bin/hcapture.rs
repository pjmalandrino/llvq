//! # `hcapture` — L36, the capture-only pass
//!
//! One pass over the calibration set with the **served** weights, keeping the
//! Hessian the encoder throws away. It writes no artifact, quantizes nothing
//! and yields zero MMLU points by construction; what it yields is the input of
//! `llvq-bench --example hstats`, which decides eight rows of
//! `docs/ROADMAP-QUALITY.md` for the price of one Mac hour.
//!
//! ## Why the served weights and not the checkpoint
//!
//! Every statistic downstream is about the model that is *shipped*: where its
//! reconstruction error lands, which directions of `H` carry it, what a
//! per-row bias would buy on it. An `H` captured on the f16 checkpoint answers
//! a different question — it is the `H` the encoder used to *make* the file,
//! not the one that describes it. So the model is built by
//! [`llvq_llm::sealed::load`], which is the served object itself: config,
//! tokenizer, q8 embedding, norms and quantized projections in one file.
//!
//! ## Why it is one pass and not two
//!
//! `quantize_model` costs two forwards per block because the weights change
//! underneath it. Here nothing is quantized, so the pass that accumulates `H`
//! is also the pass that produces the next block's input. On the 4B the
//! encoder's capture phase is 394.9 s against 6 h 58 for the whole run
//! (*measured*, `docs/fiche-4b.md` §3.4).
//!
//! ## Usage
//!
//! ```text
//! LLVQ_CALIB=c4 cargo run --release -p llvq-llm --features metal \
//!   --bin hcapture -- <sealed.bin> <n_calib> <calib_len> metal <out-dir>
//! ```
//!
//! Environment: `LLVQ_CALIB`, `LLVQ_CALIB_SEED`, `LLVQ_DTYPE`, `LLVQ_H_SHRINK`,
//! `LLVQ_ROT` (`rot` — the default — or `norot`), `LLVQ_H_DENSE` (`1` writes
//! the dense `n × n`; off by default, because the 4B is **35.86 GB** at f64).
//! `LLVQ_CONFIG` is refused by name: this is a measurement mode, and
//! `CLAUDE.md` fixes that such a mode is never a served config.
//!
//! ## What it writes
//!
//! `meta.csv`, one row per emission, carrying the run identity and the
//! reductions that are cheap; then `diag-<block>-<act>.f64`, `norms-<act>.f32`
//! and `mean-<block>-<act>.f64`, and under `LLVQ_H_DENSE=1` the full
//! `h-<block>-<act>.f64`. Flat little-endian, the convention `f1recdump`
//! already uses, so the bench side stays free of dependencies.

use candle_core::DType;
use llvq_llm::calib::{
    capture_model_hessians, CaptureConfig, CapturedHessian, HessianSink, ROTATION_SEED,
};
use llvq_llm::corpus::{calib_chars, window_starts, CalibCorpus};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Writes each emission's reductions, and its dense `H` only when asked.
struct DumpSink {
    dir: PathBuf,
    dense: bool,
    meta: std::fs::File,
    rows: usize,
    dense_bytes: u64,
}

/// Little-endian flat, no header: the header is `meta.csv`, and a reader that
/// had to parse two formats to find one number would be two chances to get the
/// basis wrong instead of one.
fn write_f64(path: &Path, v: &[f64]) -> anyhow::Result<u64> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok((v.len() * 8) as u64)
}

fn write_f32(path: &Path, v: &[f32]) -> anyhow::Result<u64> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok((v.len() * 4) as u64)
}

impl HessianSink for DumpSink {
    fn push_hessian(&mut self, c: CapturedHessian<'_>) -> anyhow::Result<()> {
        let n = c.n;
        let tag = format!("{}-{:?}-{}", c.block, c.act, c.basis.as_str());
        // The diagonal and the trace are what seven of the eight rows read;
        // the dense matrix is what two of them read. Writing the cheap ones
        // always costs 8·n bytes and removes the reason to keep 8·n².
        let diag: Vec<f64> = (0..n).map(|i| c.h[i * n + i]).collect();
        let trace: f64 = diag.iter().sum();
        // Off-diagonal mass, computed here while `H` is in hand: it is a
        // single scalar and it is the statistic that says whether a diagonal
        // approximation is defensible at all.
        let total: f64 = c.h.iter().map(|v| v * v).sum();
        let off = (total - diag.iter().map(|v| v * v).sum::<f64>()).max(0.0).sqrt();

        write_f64(&self.dir.join(format!("diag-{tag}.f64")), &diag)?;
        if let Some(m) = c.mean {
            write_f64(&self.dir.join(format!("mean-{tag}.f64")), m)?;
        }
        if let Some(t) = c.token_norms {
            write_f32(&self.dir.join(format!("norms-{tag}.f32")), t)?;
        }
        if self.dense {
            self.dense_bytes += write_f64(&self.dir.join(format!("h-{tag}.f64")), c.h)?;
        }
        writeln!(
            self.meta,
            "{},{:?},{},{},{},{:.10e},{:.10e},{},{},{}",
            c.block,
            c.act,
            n,
            c.basis.as_str(),
            c.rotation_seed.map(|s| s.to_string()).unwrap_or_default(),
            trace,
            off,
            c.mean.is_some(),
            c.token_norms.map(|t| t.len()).unwrap_or(0),
            self.dense
        )?;
        self.rows += 1;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    // A measurement mode is never a served config. `CLAUDE.md` fixes this for
    // every one of them, and the refusal is by name so the message says which.
    anyhow::ensure!(
        std::env::var("LLVQ_CONFIG").is_err(),
        "LLVQ_CONFIG is refused beside hcapture: this is a measurement mode, \
         not a served object. Unset it."
    );

    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() == 5,
        "usage: hcapture <sealed.bin> <n_calib> <calib_len> <device> <out-dir>\n\
         environment: LLVQ_CALIB, LLVQ_CALIB_SEED, LLVQ_DTYPE, LLVQ_H_SHRINK, \
         LLVQ_ROT, LLVQ_H_DENSE"
    );
    let (sealed, out) = (a[0].clone(), PathBuf::from(&a[4]));
    let n_calib: usize = a[1].parse().map_err(|e| anyhow::anyhow!("n_calib: {e}"))?;
    let calib_len: usize = a[2].parse().map_err(|e| anyhow::anyhow!("calib_len: {e}"))?;

    let corpus = CalibCorpus::parse(std::env::var("LLVQ_CALIB").ok().as_deref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let calib_seed = match std::env::var("LLVQ_CALIB_SEED") {
        Ok(s) => Some(
            s.parse::<u64>()
                .map_err(|e| anyhow::anyhow!("LLVQ_CALIB_SEED={s:?} is not an integer: {e}"))?,
        ),
        Err(_) => None,
    };
    let h_shrink: f64 = match std::env::var("LLVQ_H_SHRINK") {
        Ok(s) => s
            .parse()
            .map_err(|e| anyhow::anyhow!("LLVQ_H_SHRINK={s:?}: {e}"))?,
        Err(_) => 1.0,
    };
    // Same grammar as smoke's `rot` positional, and the same default: every
    // published encoding is rotated, so an unrotated capture is the exception
    // and has to be asked for.
    let rotation_seed = match std::env::var("LLVQ_ROT").ok().as_deref() {
        None | Some("") | Some("rot") => Some(ROTATION_SEED),
        Some("norot") => None,
        Some(o) => anyhow::bail!("LLVQ_ROT={o}: accepted values `rot` (default) and `norot`"),
    };
    let dense = matches!(std::env::var("LLVQ_H_DENSE").ok().as_deref(), Some("1"));
    // The repository's own parsers, not a second pair: `eval::dtype` refuses
    // an unknown `LLVQ_DTYPE` instead of falling back, and `eval::device`
    // refuses an unknown device. A typo silently scoring in the wrong
    // precision is the failure that module exists to close.
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    let device = llvq_llm::eval::device(&a[3])?;
    eprintln!(
        "hcapture · {sealed} · {n_calib}×{calib_len} = {} tokens · calib {} · \
         rot {} · ρ {h_shrink} · dense {dense}",
        n_calib * calib_len,
        corpus.name(),
        if rotation_seed.is_some() { "on" } else { "off" },
    );

    let t0 = std::time::Instant::now();
    let s = llvq_llm::sealed::load(&sealed, dtype, &device, llvq_llm::kvq::KvMode::F16)?;
    eprintln!(
        "  served object: {} matrices, {} quantized weights, {:.3} GB on disk, loaded in {:.1} s",
        s.matrices,
        s.quantized_weights,
        s.bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    // The same windows as the encoding, drawn by the same code — see the note
    // at the head of `corpus.rs`'s calibration section for why this is not a
    // copy.
    let text = corpus.text(calib_chars(n_calib, calib_len))?;
    let ids = s
        .tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let available = ids.len() / calib_len;
    anyhow::ensure!(
        available >= n_calib,
        "the calibration corpus only holds {available} windows of {calib_len} \
         ({} tokens read) — {n_calib} requested. This run fails here rather \
         than serving fewer in silence.",
        ids.len()
    );
    eprintln!(
        "  {n_calib} windows of {calib_len} ({available} available), {}",
        match calib_seed {
            Some(x) => format!("seeded offsets (seed {x})"),
            None => "contiguous prefix from token 0".into(),
        }
    );
    let mut hidden = Vec::with_capacity(n_calib);
    for &st in &window_starts(n_calib, ids.len(), calib_len, calib_seed) {
        let t = candle_core::Tensor::from_slice(&ids[st..st + calib_len], (1, calib_len), &device)?;
        hidden.push(s.model.embed_tokens(&t)?);
    }

    std::fs::create_dir_all(&out)?;
    let mut meta = std::fs::File::create(out.join("meta.csv"))?;
    writeln!(
        meta,
        "# hcapture · sealed={sealed} · calib={} · seed={} · n_calib={n_calib} \
         · calib_len={calib_len} · dtype={dtype:?} · rot={} · h_shrink={h_shrink} \
         · dataset_rev={}",
        corpus.name(),
        calib_seed.map(|s| s.to_string()).unwrap_or_else(|| "prefix".into()),
        rotation_seed.map(|s| format!("{s:#x}")).unwrap_or_else(|| "off".into()),
        llvq_llm::corpus::dataset_revision(),
    )?;
    writeln!(
        meta,
        "block,act,n,basis,rotation_seed,trace,offdiag_frobenius,has_mean,ntokens,dense"
    )?;

    let mut sink = DumpSink {
        dir: out.clone(),
        dense,
        meta,
        rows: 0,
        dense_bytes: 0,
    };
    let nb = s.model.blocks.len();
    let phases = capture_model_hessians(
        &s.model,
        &mut hidden,
        &CaptureConfig {
            h_shrink,
            rotation_seed,
            emit_natural: false,
        },
        &mut sink,
        |t, n| {
            if t % 4 == 0 || t + 1 == n {
                eprintln!("  block {}/{n}", t + 1);
            }
        },
    )?;

    eprintln!(
        "\n{} emissions over {nb} blocks · capture {:.1} s · reduce {:.1} s · \
         total {:.1} s · dense {:.2} GB",
        sink.rows,
        phases.capture,
        phases.factor,
        t0.elapsed().as_secs_f64(),
        sink.dense_bytes as f64 / 1e9
    );
    eprintln!("written to {}", out.display());
    Ok(())
}
