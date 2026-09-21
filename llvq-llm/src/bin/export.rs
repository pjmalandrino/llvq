//! Turn a sealed `.llvq` into a standard Hugging Face checkpoint.
//!
//! Usage: `cargo run --release -p llvq-llm --bin export -- model.llvq out_dir/`
//!
//! ## Why this exists
//!
//! Two reasons, and they are independent.
//!
//! **Evaluation.** What we want to measure is the *quantization*, not our
//! inference code. Writing our own MMLU harness would put a second unvalidated
//! thing between the model and the number, and MMLU is a family of protocols —
//! 0- or 5-shot, log-prob of the letter or of the answer text, normalized or
//! not — whose reasonable variants differ by several points. Dequantizing into
//! a checkpoint that `lm-evaluation-harness` already knows how to score makes
//! our numbers directly comparable to the paper's (Qwen3-4B: 70.2 baseline,
//! 60.7 for LLVQ without fine-tuning).
//!
//! That substitution is legitimate because the chain is already pinned: the
//! artifact decodes **bit for bit** to the weights that were evaluated, and the
//! fused GPU kernel is verified to 1e-8 against those same weights. The
//! dequantized checkpoint *is* the model — it is simply stored expensively.
//!
//! **Portability.** `.llvq` is neither GGUF nor safetensors, so nothing else
//! reads it. This is the bridge: the output loads in `transformers`, converts
//! with `mlx_lm.convert`, and can be re-quantized by anyone's toolchain.
//!
//! ⚠️ The output is **f16, full size** — ~8 GB for a 4B. It is a measurement
//! and interchange artifact, never a distribution format. Quoting its size as
//! a compression ratio would be exactly backwards.

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// One record, decoded to the `d_out × d_in` weights a checkpoint carries.
///
/// Extracted from the loop so the int4 arm is testable. A mutation that
/// exported int4 records as zeros survived the suite on 2026-09-19 because
/// nothing reached this code: a binary's `main` is not callable from a test.
fn record_to_weights(
    rec: llvq_artifact::Record,
    cbs: &llvq_artifact::Codebooks,
) -> anyhow::Result<(String, usize, usize, Vec<f32>)> {
    Ok(match rec {
        llvq_artifact::Record::Lattice(raw) => {
            let m = llvq_artifact::decode_raw(raw, cbs)?;
            let w = llvq_artifact::decode_matrix(&m);
            (m.name, m.d_out, m.d_in, w)
        }
        // The same `to_f32` `sealed::load` calls to run MMLU on a mixed file,
        // so the weights this writes for an int4 record are the weights every
        // published bar on that file was read through.
        llvq_artifact::Record::Int4(m) => {
            let w = m.to_f32();
            (m.name, m.d_out, m.d_in, w)
        }
    })
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give the path to a sealed .llvq model"))?;
    let out: PathBuf = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "llvq-export".into())
        .into();
    let device = Device::Cpu;

    let f = std::fs::File::open(&path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{path} is a projections-only artifact (format v{}); seal it first",
        head.version
    );
    // At the header, naming the set. `read_matrix_with` refuses an int4
    // record mid-file (`NotALatticeRecord`), which is correct and late: the
    // export would already have written half a directory of tensors, and the
    // half it wrote would look complete.
    // Int4G128 records are dequantized through `Int4Matrix::to_f32`, the same
    // path `sealed::load` takes to run MMLU on a mixed file. Before 2026-09-19
    // this was a refusal, because no path existed and a half-written directory
    // would have looked complete. The path exists now, and refusing would have
    // kept the served object out of every tool that reads a checkpoint.
    std::fs::create_dir_all(&out)?;
    eprintln!("reading {path} — {} quantized matrices", head.matrices);

    // ---- the quantized projections, decoded ----
    //
    // `decode_matrix` is the artifact's own decoder: it rebuilds in f64, undoes
    // the incoherence rotation, and only then narrows. Doing any of that in
    // f32 changes the last bits, and the whole claim of the format is that it
    // does not — so the export goes through the same path a reader would.
    // Read through the map each record names: a v5 Tetra file exports the same
    // way a v4 Ball one does, because `decode_matrix` is common to both.
    let cbs = llvq_artifact::Codebooks::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized = 0usize;
    let mut int4 = 0usize;
    for i in 0..head.matrices {
        let rec = llvq_artifact::read_record(&mut r, head.version)?;
        if matches!(rec, llvq_artifact::Record::Int4(_)) {
            int4 += 1;
        }
        let (name, d_out, d_in, w) = record_to_weights(rec, &cbs)?;
        quantized += d_out * d_in;
        let t = Tensor::from_vec(w, (d_out, d_in), &device)?.to_dtype(DType::F16)?;
        tensors.insert(name, t);
        if i % 36 == 0 {
            eprintln!("  {i:>3}/{}", head.matrices);
        }
    }

    // ---- everything the quantizer never touched, as stored ----
    let n_raw = read_u32(&mut r)?;
    let mut carried = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        carried += t.len();
        // Exported checkpoints are f16 whatever the stored encoding: a
        // quantized embedding is dequantized through the format's own decoder.
        let tensor = match &t.data {
            llvq_artifact::RawData::F16(d) => {
                let vals: Vec<half::f16> = d.iter().map(|b| half::f16::from_bits(*b)).collect();
                Tensor::from_vec(vals, t.dims.clone(), &device)?
            }
            llvq_artifact::RawData::Quant(_) => {
                Tensor::from_vec(t.to_f32(), t.dims.clone(), &device)?.to_dtype(DType::F16)?
            }
        };
        tensors.insert(t.name, tensor);
    }

    // ---- config and tokenizer, byte for byte from the sealed file ----
    let n_blob = read_u32(&mut r)?;
    let mut wrote_config = false;
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r)?;
        std::fs::write(out.join(&b.name), &b.bytes)?;
        wrote_config |= b.name == "config.json";
        eprintln!("  wrote {}", b.name);
    }
    anyhow::ensure!(wrote_config, "sealed file carries no config.json");

    // `transformers` reads generation defaults from the tokenizer config; the
    // sealed file does not carry one, and its absence makes some loaders warn
    // rather than fail. A minimal one keeps the export self-describing.
    let tok_cfg = out.join("tokenizer_config.json");
    if !tok_cfg.exists() {
        std::fs::write(
            &tok_cfg,
            br#"{"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 40960}"#,
        )?;
        eprintln!("  wrote tokenizer_config.json (minimal)");
    }

    let n_tensors = tensors.len();
    let model_path = out.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &model_path)?;
    let bytes = std::fs::metadata(&model_path)?.len();

    // ---- read back what we just wrote, and demand it bit for bit ----
    //
    // The export's failure mode is silent: a wrong name, a transposed shape or
    // a dtype narrowed twice all produce a file that loads and generates
    // plausible text. So the written file is re-read and compared against the
    // tensors in hand, element by element, as bit patterns rather than floats
    // — `==` on f16 would call two different NaNs equal and 0.0 == -0.0.
    eprintln!("verifying {} …", model_path.display());
    let back = candle_core::safetensors::load(&model_path, &device)?;
    anyhow::ensure!(
        back.len() == n_tensors,
        "wrote {n_tensors} tensors, read back {}",
        back.len()
    );
    for (name, want) in &tensors {
        let got = back
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name}: missing after write"))?;
        anyhow::ensure!(
            got.dims() == want.dims(),
            "{name}: shape {:?} written, {:?} read back",
            want.dims(),
            got.dims()
        );
        anyhow::ensure!(
            got.dtype() == want.dtype(),
            "{name}: dtype {:?} written, {:?} read back",
            want.dtype(),
            got.dtype()
        );
        let a: Vec<half::f16> = want.flatten_all()?.to_vec1()?;
        let b: Vec<half::f16> = got.flatten_all()?.to_vec1()?;
        let bad = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x.to_bits() != y.to_bits());
        anyhow::ensure!(bad.is_none(), "{name}: element {} differs", bad.unwrap());
    }
    eprintln!("  {n_tensors} tensors identical bit for bit\n");

    println!(
        "\n{} → {}\n  {n_tensors} tensors, {:.2} B weights dequantized ({int4} from Int4G128) + {:.0} M carried\n  \
         model.safetensors: {:.2} GB (f16)\n",
        path,
        out.display(),
        quantized as f64 / 1e9,
        carried as f64 / 1e6,
        bytes as f64 / 1e9
    );
    println!(
        "WARNING: this file is a MEASUREMENT artifact, not a distribution one. It is\n   \
         as large as the model in f16. The compression ratio is {path}.\n"
    );
    println!("Next:\n  mlx_lm.convert --hf-path {0} --mlx-path {0}-mlx\n  \
              mlx_lm.evaluate --model {0}-mlx --tasks mmlu --num-shots 5", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_llm::sealed::int4_record;

    fn dense(d_out: usize, d_in: usize) -> candle_core::Tensor {
        let v: Vec<f32> = (0..d_out * d_in)
            .map(|i| ((i % 97) as f32 - 48.0) / 32.0)
            .collect();
        candle_core::Tensor::from_vec(v, (d_out, d_in), &candle_core::Device::Cpu).unwrap()
    }

    #[test]
    fn an_int4_record_exports_the_weights_it_holds() {
        let t = dense(4, 256);
        let m = int4_record("model.layers.0.self_attn.v_proj.weight", &t).unwrap();
        let want = m.to_f32();
        let cbs = llvq_artifact::Codebooks::new();
        let (name, d_out, d_in, got) =
            record_to_weights(llvq_artifact::Record::Int4(m), &cbs).unwrap();
        assert_eq!(name, "model.layers.0.self_attn.v_proj.weight");
        assert_eq!((d_out, d_in), (4, 256));
        assert_eq!(got, want, "the int4 arm must export what the record holds");
    }

    #[test]
    fn an_int4_export_is_not_zero_and_tracks_the_dense_tensor() {
        // The mutation that survived was `vec![0.0; n]`. Shape and length are
        // not enough to catch it; the values have to be read.
        let t = dense(4, 256);
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let m = int4_record("model.layers.0.self_attn.v_proj.weight", &t).unwrap();
        let cbs = llvq_artifact::Codebooks::new();
        let (_, _, _, got) =
            record_to_weights(llvq_artifact::Record::Int4(m), &cbs).unwrap();
        assert!(got.iter().any(|&x| x != 0.0), "an int4 export of zeros");
        let err: f32 = got
            .iter()
            .zip(&flat)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / got.len() as f32;
        let scale: f32 = flat.iter().map(|x| x.abs()).sum::<f32>() / flat.len() as f32;
        assert!(
            err < 0.1 * scale,
            "int4 g128 should track the dense tensor: mean |error| {err} against scale {scale}"
        );
    }
}
