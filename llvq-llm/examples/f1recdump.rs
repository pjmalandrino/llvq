//! Dump the blocks the served encoder actually sees — the compensated, rotated
//! 24-blocks of one transformer block of Qwen3-0.6B — so the F1 encoder
//! prototype can be timed on real GPTQ residues rather than on N(0,1) draws.
//!
//! `cargo run --release -p llvq-llm --features metal,fast-linalg --example f1recdump -- <out_dir> [n_sample] [n_calib] [calib_len] [device]`
//!
//! ## What it records
//!
//! The GPTQ loop (`llvq-quant/src/gptq.rs`) hands every block to
//! `BlockQuantizer::quantize` **after** the error feedback of the earlier
//! columns has rewritten it and **after** the incoherence rotation — that `v`
//! is the encoder's input in production, and it is what gets written here.
//! A recording quantizer wraps the served `LeechShapeGain` (leech1c12: 1 gain
//! bit, shell ≤ 12, 5 levels), copies each `v` and the row scale announced for
//! it, then delegates, so the loop's error feedback is exactly the served one.
//!
//! Every block of transformer block 0 is kept in memory (648,192 blocks on the
//! 0.6B, ~130 MB), then a uniform stride sample of `n_sample` blocks over the
//! seven matrices is written out. The whole block rather than the first
//! `n_sample` calls: the loop is column-major, so the first 20,000 calls would
//! all come from the first ten column positions of `q_proj`, where the
//! accumulated compensation is smallest — the easiest residues, not a sample.
//!
//! ## Determinism across threads
//!
//! Each encoder thread owns one wrapper and records its blocks in loop order
//! (column block outer, row inner) over its own row chunk; the chunks are
//! then ordered by a hash of their first block. Row chunks are independent in
//! the loop (`parallel_matches_serial_exactly`), so the values do not depend on
//! the thread count and the ordering does not depend on scheduling. The column
//! block of every record is recovered from its position in its chunk.
//!
//! ## Files (little-endian, flat)
//!
//! - `blocks-<tag>.f64`: `n_sample × 24` f64, the compensated rotated blocks.
//! - `rowscale-<tag>.f64`: `n_sample` f64, the row scale announced for each.
//! - `meta-<tag>.csv`: `matrix,colblock` per block, same order.
//!
//! Settings mirror the 0.6B gate runs (`docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt`):
//! wikitext-2 train, 64 × 2048 contiguous prefix, rotation on (seed
//! 0x110feed), nogs, damping 1e-2, f32. The model weights are never written
//! back: only block 0 is quantized, and nothing downstream reads it.

use candle_core::{DType, Tensor};
use llvq_core::DIM;
use llvq_llm::calib::{block_matrix_plan, effective_rotation_seed, Hessian};
use llvq_llm::corpus::hf_parquet_text;
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{Act, Capture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, BlockCode, BlockQuantizer, LeechShapeGain};
use llvq_quant::rotation::Rotation;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// The served codebook, `leech1c12` as `smoke` resolves it.
const GAIN_BITS: u32 = 1;
const SHELL_CAP: u32 = 12;

/// `smoke`'s rotation seed — a constant there too.
const ROTATION_SEED: u64 = 0x11_0FEED;

/// One thread's recording: the matrix it served and its blocks in loop order.
struct Chunk {
    matrix: usize,
    blocks: Vec<[f64; DIM]>,
    row_scales: Vec<f64>,
}

/// A [`BlockQuantizer`] that copies every block it is handed, then delegates.
struct Recorder {
    inner: LeechShapeGain,
    matrix: usize,
    row_scale: f64,
    blocks: Vec<[f64; DIM]>,
    row_scales: Vec<f64>,
    sink: Arc<Mutex<Vec<Chunk>>>,
}

impl BlockQuantizer for Recorder {
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    fn set_row_scale(&mut self, scale: f64) {
        self.row_scale = scale;
        self.inner.set_row_scale(scale);
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        let x: [f64; DIM] = v.try_into().expect("a Leech block is 24 weights");
        self.blocks.push(x);
        self.row_scales.push(self.row_scale);
        self.inner.quantize(v, out);
    }

    fn last_code(&self) -> Option<BlockCode> {
        self.inner.last_code()
    }

    fn retraction_target(&self, norm_before: f64) -> Option<f64> {
        self.inner.retraction_target(norm_before)
    }

    fn reproject(&self, code: &BlockCode, norm: f64, out: &mut [f64]) -> Option<BlockCode> {
        self.inner.reproject(code, norm, out)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let chunk = Chunk {
            matrix: self.matrix,
            blocks: std::mem::take(&mut self.blocks),
            row_scales: std::mem::take(&mut self.row_scales),
        };
        self.sink.lock().expect("recorder sink poisoned").push(chunk);
    }
}

/// Collects the four Hessians of one block — `calib.rs`'s private capture.
struct BlockCapture {
    target: usize,
    acc: HashMap<Act, Hessian>,
}

impl Capture for BlockCapture {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == self.target {
            if let Some(h) = self.acc.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

/// FNV-1a over the bit patterns of a block: a scheduling-independent key.
fn block_hash(x: &[f64; DIM]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in x {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn arg<T: std::str::FromStr>(a: &[String], i: usize, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match a.get(i) {
        None => default,
        Some(s) => s.parse().unwrap_or_else(|e| panic!("argument {i} = {s:?}: {e}")),
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = a.first().cloned().unwrap_or_else(|| ".".into());
    let n_sample: usize = arg(&a, 1, 20_000);
    let n_calib: usize = arg(&a, 2, 64);
    let calib_len: usize = arg(&a, 3, 2048);
    let device = llvq_llm::eval::device(a.get(4).map(String::as_str).unwrap_or("metal"))?;
    let threads = match std::env::var("LLVQ_THREADS") {
        Ok(s) if !s.is_empty() => s.parse::<usize>()?.max(1),
        _ => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    };
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let damping = 1e-2;
    let target_block = 0usize;
    let tag = "0.6b-b0";

    eprintln!("=== f1recdump ===");
    eprintln!("  model        {repo}, transformer block {target_block}");
    eprintln!("  codebook     leech1c12 ({GAIN_BITS} gain bit, shell ≤ {SHELL_CAP}, 5 levels)");
    eprintln!("  calibration  wikitext2 train, {n_calib} × {calib_len}, contiguous prefix");
    eprintln!("  rotation     on (seed {ROTATION_SEED:#x}), nogs, damping {damping:e}, f32");
    eprintln!("  device       {device:?}, {threads} encoder threads");
    eprintln!("  sample       {n_sample} blocks, uniform stride over the whole block");
    eprintln!("  out          {out_dir}/{{blocks,rowscale,meta}}-{tag}.*");

    let t_all = std::time::Instant::now();
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let vb = ck.var_builder(DType::F32, &device)?;
    let model = Qwen3::new(&ck.config, vb, llvq_llm::kvq::KvMode::F16)?;

    // ---- calibration windows: the prefix, as every gate run ----
    let train = hf_parquet_text(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    )?
    .join("\n\n");
    let train_ids = tok
        .encode(train.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let available = train_ids.len() / calib_len;
    anyhow::ensure!(
        available >= n_calib,
        "the corpus holds {available} windows of {calib_len}, {n_calib} requested"
    );
    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for w in 0..n_calib {
        let ids = &train_ids[w * calib_len..(w + 1) * calib_len];
        let t = Tensor::from_slice(ids, (1, calib_len), &device)?;
        hidden.push(model.embed_tokens(&t)?);
    }
    let total_rows: usize = hidden.iter().map(|h| h.dim(1).unwrap_or(0)).sum();
    eprintln!("loaded in {:.1} s; {total_rows} calibration rows", t_all.elapsed().as_secs_f64());

    // ---- pass 1: H per activation of block 0, with the original weights ----
    let t0 = std::time::Instant::now();
    let mut cap = BlockCapture {
        target: target_block,
        acc: HashMap::new(),
    };
    for act in Act::ALL {
        let w = act.width(model.config());
        cap.acc.insert(act, Hessian::new(w, &device, total_rows)?);
    }
    let mask = model.causal_mask_for(&hidden[0])?;
    for h in hidden.iter() {
        let _ = model.blocks[target_block].forward(h, model.rotary(), &mask, target_block, &mut cap)?;
    }
    eprintln!("Hessians captured in {:.1} s", t0.elapsed().as_secs_f64());

    // ---- factor per activation, in the rotated basis ----
    let t0 = std::time::Instant::now();
    let mut factors: HashMap<Act, (GptqFactor, Rotation)> = HashMap::new();
    for act in Act::ALL {
        let mut h = cap
            .acc
            .remove(&act)
            .expect("inserted above")
            .to_f64()?;
        let n = act.width(model.config());
        let rot = Rotation::new(n, effective_rotation_seed(ROTATION_SEED, target_block, act));
        rot.rotate_hessian(&mut h);
        let factor = GptqFactor::new(&h, n, damping)
            .map_err(|e| anyhow::anyhow!("block {target_block}, {act:?}: {e}"))?;
        factors.insert(act, (factor, rot));
    }
    eprintln!("factored in {:.1} s", t0.elapsed().as_secs_f64());

    // ---- quantize the seven matrices with the recording wrapper ----
    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    let sink: Arc<Mutex<Vec<Chunk>>> = Arc::new(Mutex::new(Vec::new()));
    let plan = block_matrix_plan();
    let mut expected: Vec<(String, usize, usize)> = Vec::with_capacity(plan.len());
    for (mi, (act, name)) in plan.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (factor, rot) = &factors[act];
        let lin = model.blocks[target_block].linear(name);
        let w = lin.weight();
        let (d_out, d_in) = w.dims2()?;
        let flat: Vec<f64> = w
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let mut weights = Weights::new(d_out, d_in, flat);
        rot.rotate_weight_rows(&mut weights.w, d_out);
        let gain = fit_gain_centroids(&weights.w, d_out, d_in, cfg.block, GAIN_BITS, 40);
        let sink_m = sink.clone();
        let make = move || -> Box<dyn BlockQuantizer> {
            Box::new(Recorder {
                inner: LeechShapeGain::with_caps(
                    gain.clone(),
                    SHELL_CAP,
                    llvq_search::generic::MAX_LEVELS_ANY,
                ),
                matrix: mi,
                row_scale: 1.0,
                blocks: Vec::new(),
                row_scales: Vec::new(),
                sink: sink_m.clone(),
            })
        };
        llvq_quant::gptq::quantize_layer_parallel_capturing(
            &mut weights,
            factor,
            None,
            &make,
            &cfg,
            threads,
            None,
        );
        let nb = d_in / cfg.block;
        expected.push((name.to_string(), d_out, nb));
        eprintln!(
            "  {name:<20} {d_out} × {d_in}: {} blocks in {:.1} s",
            d_out * nb,
            t0.elapsed().as_secs_f64()
        );
    }

    // ---- order the chunks, recover the column block, sample ----
    let mut chunks = std::mem::take(&mut *sink.lock().expect("sink poisoned"));
    chunks.retain(|c| !c.blocks.is_empty());
    chunks.sort_by_key(|c| (c.matrix, block_hash(&c.blocks[0])));
    // (matrix, column block, block, row scale), in a deterministic order.
    let mut all: Vec<(usize, usize, [f64; DIM], f64)> = Vec::new();
    let mut per_matrix = vec![0usize; plan.len()];
    for c in &chunks {
        let nb = expected[c.matrix].2;
        anyhow::ensure!(
            c.blocks.len().is_multiple_of(nb),
            "chunk of {} blocks for {} is not a multiple of {nb} column blocks",
            c.blocks.len(),
            expected[c.matrix].0
        );
        let rows = c.blocks.len() / nb;
        for (k, (b, rs)) in c.blocks.iter().zip(&c.row_scales).enumerate() {
            all.push((c.matrix, k / rows, *b, *rs));
        }
        per_matrix[c.matrix] += c.blocks.len();
    }
    for (mi, (name, d_out, nb)) in expected.iter().enumerate() {
        anyhow::ensure!(
            per_matrix[mi] == d_out * nb,
            "{name}: recorded {} blocks, the loop quantizes {}",
            per_matrix[mi],
            d_out * nb
        );
    }
    let total = all.len();
    anyhow::ensure!(total >= n_sample, "only {total} blocks recorded, {n_sample} asked");
    let zero = all.iter().filter(|(_, _, b, _)| b.iter().all(|&v| v == 0.0)).count();
    eprintln!("{total} blocks recorded over {} matrices, {zero} all-zero", plan.len());

    let mut blocks = std::fs::File::create(format!("{out_dir}/blocks-{tag}.f64"))?;
    let mut scales = std::fs::File::create(format!("{out_dir}/rowscale-{tag}.f64"))?;
    let mut meta = std::fs::File::create(format!("{out_dir}/meta-{tag}.csv"))?;
    writeln!(meta, "matrix,colblock")?;
    let mut in_sample = vec![0usize; plan.len()];
    for j in 0..n_sample {
        // Midpoints of n_sample equal bins over the whole recording.
        let i = ((2 * j + 1) * total) / (2 * n_sample);
        let (mi, s, b, rs) = &all[i];
        for v in b {
            blocks.write_all(&v.to_le_bytes())?;
        }
        scales.write_all(&rs.to_le_bytes())?;
        writeln!(meta, "{},{s}", expected[*mi].0)?;
        in_sample[*mi] += 1;
    }
    for (mi, (name, _, _)) in expected.iter().enumerate() {
        eprintln!("  sample: {name:<20} {:>6} of {:>7}", in_sample[mi], per_matrix[mi]);
    }
    eprintln!(
        "wrote {n_sample} blocks to {out_dir}/blocks-{tag}.f64 in {:.1} s total",
        t_all.elapsed().as_secs_f64()
    );
    Ok(())
}
