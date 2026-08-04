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
    std::fs::create_dir_all(&out)?;
    eprintln!("reading {path} — {} quantized matrices", head.matrices);

    // ---- the quantized projections, decoded ----
    //
    // `decode_matrix` is the artifact's own decoder: it rebuilds in f64, undoes
    // the incoherence rotation, and only then narrows. Doing any of that in
    // f32 changes the last bits, and the whole claim of the format is that it
    // does not — so the export goes through the same path a reader would.
    let ix = llvq_search::index::Indexer::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized = 0usize;
    for i in 0..head.matrices {
        let m = llvq_artifact::read_matrix(&mut r, &ix)?;
        quantized += m.d_out * m.d_in;
        let w = llvq_artifact::decode_matrix(&m);
        let t = Tensor::from_vec(w, (m.d_out, m.d_in), &device)?.to_dtype(DType::F16)?;
        tensors.insert(m.name, t);
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
        "\n{} → {}\n  {n_tensors} tensors, {:.2} Md de poids déquantifiés + {:.0} M portés\n  \
         model.safetensors : {:.2} Go (f16)\n",
        path,
        out.display(),
        quantized as f64 / 1e9,
        carried as f64 / 1e6,
        bytes as f64 / 1e9
    );
    println!(
        "⚠️  Ce fichier est un artefact de MESURE, pas de distribution : il pèse\n   \
         ce que pèse le modèle en f16. Le taux de compression, c'est {path}.\n"
    );
    println!("Ensuite :\n  mlx_lm.convert --hf-path {0} --mlx-path {0}-mlx\n  \
              mlx_lm.evaluate --model {0}-mlx --tasks mmlu --num-shots 5", out.display());
    Ok(())
}
