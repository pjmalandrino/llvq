//! What int4 would buy, matrix by matrix, in the metric the loop minimizes.
//!
//! ```text
//! LLVQ_CALIB=wikitext2 hratio <sealed.bin> <ckpt-dir> <n_calib> <calib_len> <device> <out.tsv>
//! ```
//!
//! ## The question
//!
//! An int4 allocation is a knapsack: each of the 216 `Tetra` matrices has a
//! cost in bits and a benefit in quality, and the budget is `b_max`. The costs
//! are arithmetic. The benefits are the problem: measuring them on MMLU is 216
//! arms, and the winner's curse at that width would eat the answer before it
//! arrived.
//!
//! This prices the benefit in the metric GPTQ itself minimizes:
//!
//! ```text
//!   ratio_i = tr(ΔW_tetra H ΔW_tetraᵀ) / tr(ΔW_int4 H ΔW_int4ᵀ)
//! ```
//!
//! ## Why a ratio and not a difference
//!
//! An earlier attempt ranked matrices by the absolute `tr(ΔW H ΔWᵀ)` of the
//! served reconstruction and produced a ranking correlated **0.988 with the
//! calibration samples per dimension** and −0.925 with the measured MMLU gain.
//! The reason is mechanical: a Hessian estimated from fewer samples than it has
//! dimensions has a null space, and every error component living in it costs
//! nothing. `down_proj`, at 1.68 samples per dimension, scored last on a
//! criterion it should have led.
//!
//! A ratio puts the same `H` above and below. Its scale cancels, and so does
//! its null space: both arms lose exactly the components `H` cannot see.
//!
//! ## Everything is in the natural basis, and that is not a shortcut
//!
//! `tr(ΔW H ΔWᵀ)` is invariant under an orthogonal change of basis applied to
//! both factors, since `Q Qᵀ = I`. So the rotated and natural bases give the
//! same number, and the natural one is the basis int4 g128 actually groups in.
//! [`llvq_artifact::llvq_artifact::decode_matrix`] un-rotates on its way out, the
//! checkpoint is natural, and the capture is asked for its natural emission.
//! Mixing the two bases would be the one way to get this wrong, and no code
//! path here can.
//!
//! ## What it is not
//!
//! A local metric over one projection. The repository holds six cases where a
//! better local proxy composed worse, two of them measured on 2026-09-18. So
//! the ratios below are a hypothesis until they reproduce the three measured
//! MMLU gains, and the tool prints what is needed to check that rather than a
//! recommendation.

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use llvq_llm::calib::{capture_model_hessians, CaptureConfig, CapturedHessian, HBasis, HessianSink};
use llvq_llm::corpus::{calib_chars, window_starts, CalibCorpus};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom, Write};

/// Offsets of every lattice record in the sealed file, by tensor name.
fn index_records(path: &str) -> anyhow::Result<(u32, HashMap<String, u64>)> {
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    let mut at = HashMap::new();
    for _ in 0..head.matrices {
        let off = r.stream_position()?;
        match llvq_artifact::read_record(&mut r, head.version)? {
            llvq_artifact::Record::Lattice(m) => {
                at.insert(m.name.clone(), off);
            }
            llvq_artifact::Record::Int4(_) => {}
        }
    }
    Ok((head.version, at))
}

fn decode_at(
    path: &str,
    version: u32,
    off: u64,
    cbs: &llvq_artifact::Codebooks,
) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(off))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    // `read_matrix_with` unpacks the indices through the record's own kind,
    // which is what makes a Ball and a Tetra file read by the same line.
    let m = llvq_artifact::read_matrix_with(&mut r, version, cbs)?;
    Ok((m.d_out, m.d_in, llvq_artifact::decode_matrix(&m)))
}

/// `tr(D H Dᵀ)` with `D` of shape `d_out × n`, on the device, in f32.
///
/// f32 and not f64: the quantity is a ranking input, the GEMM is
/// `2 d_out n²` and reaches 484 Gflop on `down_proj` alone, and candle's
/// device path is f32. The tool prints the two terms so a reader can see the
/// dynamic range it is asking of them.
fn weighted(d: &Tensor, h: &Tensor) -> anyhow::Result<f64> {
    let m = d.matmul(h)?;
    Ok(m.mul(d)?.sum_all()?.to_scalar::<f32>()? as f64)
}

struct Ratios<'a> {
    sealed: &'a str,
    version: u32,
    at: HashMap<String, u64>,
    cbs: llvq_artifact::Codebooks,
    ck: candle_core::safetensors::MmapedSafetensors,
    device: Device,
    out: File,
    rows: usize,
}

impl HessianSink for Ratios<'_> {
    fn push_hessian(&mut self, c: CapturedHessian<'_>) -> anyhow::Result<()> {
        // The rotated emission describes the same quadratic form; taking both
        // would double the work and change no number.
        if !matches!(c.basis, HBasis::Natural) {
            return Ok(());
        }
        let h = Tensor::from_iter(c.h.iter().map(|v| *v as f32), &self.device)?
            .reshape((c.n, c.n))?;
        for proj in c.act.consumers() {
            let name = format!("model.layers.{}.{}.weight", c.block, proj);
            let Some(&off) = self.at.get(&name) else {
                continue; // int4 in the served file, or absent: nothing to price
            };
            let (d_out, d_in, tetra) = decode_at(self.sealed, self.version, off, &self.cbs)?;
            anyhow::ensure!(d_in == c.n, "{name}: d_in {d_in} against H of {}", c.n);
            let w = self
                .ck
                .load(&name, &self.device)?
                .to_dtype(DType::F32)?
                .reshape((d_out, d_in))?;
            let q4 = llvq_llm::sealed::quantize_dequantize_q4(
                &w,
                &name,
                llvq_llm::sealed::Q4_GROUP,
                DType::F32,
            )?;
            let wt = Tensor::from_vec(tetra, (d_out, d_in), &self.device)?;
            let d_tetra = (&w - &wt)?;
            let d_q4 = (&w - &q4)?;
            let (a, b) = (weighted(&d_tetra, &h)?, weighted(&d_q4, &h)?);
            writeln!(
                self.out,
                "{}\t{}\t{d_out}\t{d_in}\t{a:.9e}\t{b:.9e}\t{:.6}",
                c.block,
                proj,
                if b > 0.0 { a / b } else { f64::INFINITY }
            )?;
            self.rows += 1;
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    anyhow::ensure!(
        std::env::var("LLVQ_CONFIG").is_err(),
        "LLVQ_CONFIG is refused beside hratio: this is a measurement mode."
    );
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() == 6,
        "usage: hratio <sealed.bin> <ckpt-dir> <n_calib> <calib_len> <device> <out.tsv>"
    );
    let n_calib: usize = a[2].parse().context("n_calib")?;
    let calib_len: usize = a[3].parse().context("calib_len")?;
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    let device = llvq_llm::eval::device(&a[4])?;
    let corpus = CalibCorpus::parse(std::env::var("LLVQ_CALIB").ok().as_deref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let (version, at) = index_records(&a[0])?;
    eprintln!("hratio · {} lattice records indexed · calib {} · {n_calib}x{calib_len}",
        at.len(), corpus.name());

    let s = llvq_llm::sealed::load(&a[0], dtype, &device, llvq_llm::kvq::KvMode::F16)?;
    let text = corpus.text(calib_chars(n_calib, calib_len))?;
    let ids = s.tokenizer.encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(ids.len() / calib_len >= n_calib, "corpus too short");
    let mut hidden = Vec::with_capacity(n_calib);
    for &st in &window_starts(n_calib, ids.len(), calib_len, None) {
        let t = Tensor::from_slice(&ids[st..st + calib_len], (1, calib_len), &device)?;
        hidden.push(s.model.embed_tokens(&t)?);
    }

    let ckpt = llvq_llm::loader::Checkpoint::from_dir(std::path::Path::new(&a[1]))?;
    // Safety: the checkpoint files are not modified while the mapping is
    // alive, the same contract `sealed::load` takes for the restore path.
    let ck = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&ckpt.weights)? };

    let mut out = File::create(&a[5])?;
    writeln!(out, "# hratio, natural basis, f32 device arithmetic")?;
    writeln!(out, "# sealed={} calib={} tokens={}", a[0], corpus.name(), n_calib * calib_len)?;
    writeln!(out, "block\tproj\td_out\td_in\ttetra_wH\tint4_wH\tratio")?;
    let mut sink = Ratios {
        sealed: &a[0],
        version,
        at,
        cbs: llvq_artifact::Codebooks::new(),
        ck,
        device: device.clone(),
        out,
        rows: 0,
    };
    let cfg = CaptureConfig { h_shrink: 1.0, rotation_seed: None, emit_natural: true };
    let t0 = std::time::Instant::now();
    capture_model_hessians(&s.model, &mut hidden, &cfg, &mut sink, |t, n| {
        if t % 6 == 0 {
            eprintln!("  block {t}/{n}, {:.0} s", t0.elapsed().as_secs_f64());
        }
    })?;
    eprintln!("{} matrices priced in {:.0} s", sink.rows, t0.elapsed().as_secs_f64());
    Ok(())
}
