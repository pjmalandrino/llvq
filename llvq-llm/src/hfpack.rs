//! Rewriting a sealed `.llvq` as a Hugging Face directory, still compressed.
//!
//! [`crate::sealed`] loads a sealed file into our own model, and `bin/export`
//! dequantizes one into an f16 checkpoint, 8 GB for a 4B. Neither lets anything
//! outside this repository read the **compressed** object. This module writes
//! it as safetensors beside a `config.json` whose `quantization_config` block
//! says how to read it back, which is stage 0 of `docs/plan-transformers.md`.
//!
//! ## The three format decisions, and where they come from
//!
//! Taken by the operator on 2026-09-28 and recorded in
//! `proofs/preregistration-hf-safetensors-2026-09-28.md` §3.
//!
//! * **Naming is ours.** The artifact's `<prefix>.weight` becomes
//!   `<prefix>.<field>` ([`tensor_prefix`]). No in-tree quantization method has
//!   Tetra's shape: aqlm and vptq store codes against a global codebook, and a
//!   Tetra word indexes a 47-bit map with no dictionary at all.
//! * **The payload is the disk's.** `codes` holds the bytes the record holds,
//!   one 48-bit word per block, MSB-first, dense ([`llvq_artifact::code_stream`]).
//!   The served `tetra48` layout would save the kernel a transcode and cost
//!   2.148 against 2.000 b/weight on the codes, and would marry a distribution
//!   file to one kernel layout.
//! * **The rotation travels as its tables.** `signs` and `small`, once per
//!   distinct `(d_in, seed)`, deduplicated. The seed alone would need a
//!   bit-exact port of `SplitMix64`, of the Gaussian draw and of Gram-Schmidt
//!   into whatever language reads the file; a last-bit disagreement there
//!   changes the weights and breaks nothing visibly.
//!
//! ## Why a digest file
//!
//! The gate of stage 0 is an **independent** reader rebuilding every field bit
//! for bit. So the packer also writes [`DIGEST_FILE`], one SHA-256 per field,
//! computed from the fields as `llvq-artifact` reads them out of the `.llvq`.
//! `ops/llvq_hf_check.py` recomputes every one of them from the safetensors
//! bytes alone, including its own unpacking of the 48-bit words, and the two
//! paths share no code. A digest over the packed bytes alone would have proved
//! only that a byte array survived a copy.
//!
//! The convention is fixed here, in one place, because both sides implement it:
//! **SHA-256 over little-endian bit patterns, in the order the field stores
//! them**. `indices` and `gains` are hashed as the values the reader produced,
//! u64 and u32, which is what makes the bit order load-bearing on both sides.
//!
//! ## What it refuses
//!
//! A Ball record, by name, at the header. This packer has one map. A Ball index
//! is 47 or 48 bits depending on a shell cap, its gain field is up to 2 bits,
//! and its blocks are therefore not byte aligned; writing one here would ship a
//! file whose codes no reader of ours unpacks. The three served files carry no
//! Ball record.

use crate::digest::{
    sha256_f64, sha256_f64_as_f32, sha256_file, sha256_hex, sha256_u16, sha256_u32, sha256_u64,
    Sha256,
};
use candle_core::{DType, Device, Tensor};
use llvq_artifact::{CodeKind, RawData, Record};
use llvq_core::DIM;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Version of the safetensors layout this module writes, in
/// `quantization_config.llvq_layout_version`. Bumped when a tensor name or a
/// byte convention changes, never for a new field.
pub const LAYOUT_VERSION: u32 = 1;

/// `quantization_config.quant_method`, the name a `transformers` quantizer
/// registers under.
pub const QUANT_METHOD: &str = "llvq";

/// How `codes` is packed. The value a reader must check before it unpacks a
/// single word.
pub const CODE_ORDER: &str = "msb_first_dense";

pub const MODEL_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";
pub const DIGEST_FILE: &str = "llvq-digest.json";

/// The tensor-name stem of a record: the artifact's name minus a trailing
/// `.weight`.
///
/// `model.layers.0.self_attn.q_proj.weight` becomes
/// `model.layers.0.self_attn.q_proj`, so the fields land as `q_proj.codes` and
/// not `q_proj.weight.codes`. A state-dict key with `weight.codes` in it names
/// a submodule called `weight`, which is not what any loader expects.
pub fn tensor_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

/// Key of a rotation in `quantization_config.rotations`.
///
/// `(d_in, seed)` and not the seed alone: [`llvq_quant::rotation::Rotation`]
/// takes both, and the same seed at two widths is two different transforms.
pub fn rotation_key(d_in: usize, seed: u64) -> String {
    format!("{d_in}_{seed:016x}")
}

/// What one run wrote.
#[derive(Debug)]
pub struct Summary {
    pub records: u32,
    pub lattice: u32,
    pub int4: u32,
    pub rotations: usize,
    pub raw_tensors: u32,
    pub blobs: usize,
    pub tensors: usize,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    pub model_bytes: u64,
    pub config_bytes: u64,
    pub dir_bytes: u64,
}

fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Widen the f16 bit patterns a record stores into the tensor candle writes.
fn f16_tensor(bits: &[u16], dims: &[usize], device: &Device) -> anyhow::Result<Tensor> {
    let vals: Vec<half::f16> = bits.iter().map(|b| half::f16::from_bits(*b)).collect();
    Ok(Tensor::from_vec(vals, dims.to_vec(), device)?)
}

/// Write a sealed `.llvq` as a Hugging Face directory, compressed.
///
/// The whole file is read once, record by record: a 4B holds 119 M blocks and
/// nothing here may hold two copies of them.
pub fn pack(src: &Path, out: &Path) -> anyhow::Result<Summary> {
    let device = Device::Cpu;
    let src_sha = sha256_file(src)?;

    let f = std::fs::File::open(src)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{} is a projections-only artifact (format v{}); seal it first",
        src.display(),
        head.version
    );
    // At the header, before a record is read, which is the only place a
    // refusal is cheap. A Ball index has no fixed width and no byte alignment;
    // see the module header.
    anyhow::ensure!(
        !head.kinds().contains(CodeKind::Ball),
        "{} declares kinds {}: this packer writes Tetra and Int4G128 records only. \
         A Ball index is 47 or 48 bits wide depending on its shell cap, so its blocks \
         are not byte aligned, and no reader of the written file unpacks them",
        src.display(),
        head.kinds()
    );

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut records: Map<String, Value> = Map::new();
    let mut record_digests: Map<String, Value> = Map::new();
    let mut rotations: Map<String, Value> = Map::new();
    let mut rotation_digests: Map<String, Value> = Map::new();
    let mut lattice = 0u32;
    let mut int4 = 0u32;
    let mut quantized_weights = 0usize;

    for i in 0..head.matrices {
        match llvq_artifact::read_record(&mut r, head.version)? {
            Record::Lattice(raw) => {
                anyhow::ensure!(
                    raw.kind == CodeKind::Tetra,
                    "{}: record kind {} is not one this packer writes",
                    raw.name,
                    raw.kind
                );
                lattice += 1;
                quantized_weights += raw.d_out * raw.d_in;
                let (index_bits, gain_bits) = llvq_artifact::code_widths(
                    raw.kind,
                    &raw.name,
                    raw.shell_cap,
                    raw.centroids.len(),
                )?;
                let nblocks = raw.d_in / DIM;
                let tail_cols = raw.d_in % DIM;
                let prefix = tensor_prefix(&raw.name).to_string();

                // The bytes the record holds, from the format itself.
                let codes = llvq_artifact::code_stream(&raw)?;
                let mut digest = Map::new();
                digest.insert("codes".into(), json!(sha256_hex(&codes)));
                digest.insert("indices".into(), json!(sha256_u64(&raw.indices)));
                digest.insert("gains".into(), json!(sha256_u32(&raw.gains)));
                digest.insert("row_scales".into(), json!(sha256_f64(&raw.row_scales)));
                digest.insert("centroids".into(), json!(sha256_f64(&raw.centroids)));
                if tail_cols > 0 {
                    digest.insert("tail".into(), json!(sha256_f64_as_f32(&raw.tail)));
                }

                let code_bytes = codes.len();
                tensors.insert(
                    format!("{prefix}.codes"),
                    Tensor::from_vec(codes, code_bytes, &device)?,
                );
                tensors.insert(
                    format!("{prefix}.row_scales"),
                    Tensor::from_vec(raw.row_scales.clone(), raw.d_out, &device)?,
                );
                tensors.insert(
                    format!("{prefix}.centroids"),
                    Tensor::from_vec(raw.centroids.clone(), raw.centroids.len(), &device)?,
                );
                if tail_cols > 0 {
                    let tail: Vec<f32> = raw.tail.iter().map(|v| *v as f32).collect();
                    tensors.insert(
                        format!("{prefix}.tail"),
                        Tensor::from_vec(tail, (raw.d_out, tail_cols), &device)?,
                    );
                }

                // One rotation per distinct (d_in, seed), built once.
                let rotation = match raw.rotation_seed {
                    None => Value::Null,
                    Some(seed) => {
                        let key = rotation_key(raw.d_in, seed);
                        if !rotations.contains_key(&key) {
                            let rot = llvq_quant::rotation::Rotation::new(raw.d_in, seed);
                            let (signs, small) = (rot.signs(), rot.small());
                            let k = rot.odd();
                            rotations.insert(
                                key.clone(),
                                json!({
                                    "seed": seed.to_string(),
                                    "n": rot.dim(),
                                    "pow2": rot.pow2(),
                                    "odd": k,
                                }),
                            );
                            rotation_digests.insert(
                                key.clone(),
                                json!({
                                    "signs": sha256_f64(signs),
                                    "small": sha256_f64(small),
                                }),
                            );
                            tensors.insert(
                                format!("llvq.rotations.{key}.signs"),
                                Tensor::from_vec(signs.to_vec(), rot.dim(), &device)?,
                            );
                            tensors.insert(
                                format!("llvq.rotations.{key}.small"),
                                Tensor::from_vec(small.to_vec(), (k, k), &device)?,
                            );
                        }
                        json!(key)
                    }
                };

                records.insert(
                    raw.name.clone(),
                    json!({
                        "kind": "tetra",
                        "prefix": prefix,
                        "d_out": raw.d_out,
                        "d_in": raw.d_in,
                        "shell_cap": raw.shell_cap,
                        "n_centroids": raw.centroids.len(),
                        "index_bits": index_bits,
                        "gain_bits": gain_bits,
                        "nblocks": nblocks,
                        "tail_cols": tail_cols,
                        "code_bytes": code_bytes,
                        "rotation": rotation,
                    }),
                );
                record_digests.insert(raw.name.clone(), Value::Object(digest));
            }
            Record::Int4(m) => {
                int4 += 1;
                quantized_weights += m.d_out * m.d_in;
                anyhow::ensure!(
                    m.d_in % 2 == 0,
                    "{}: d_in {} is odd, so its nibbles do not split by row",
                    m.name,
                    m.d_in
                );
                let gpr = m.groups_per_row();
                let prefix = tensor_prefix(&m.name).to_string();
                record_digests.insert(
                    m.name.clone(),
                    json!({
                        "qweight": sha256_hex(&m.packed),
                        "scales": sha256_u16(&m.scales),
                        "biases": sha256_u16(&m.biases),
                    }),
                );
                records.insert(
                    m.name.clone(),
                    json!({
                        "kind": "int4g128",
                        "prefix": prefix,
                        "d_out": m.d_out,
                        "d_in": m.d_in,
                        "bits": m.bits,
                        "group": m.group,
                        "groups_per_row": gpr,
                    }),
                );
                tensors.insert(
                    format!("{prefix}.scales"),
                    f16_tensor(&m.scales, &[m.d_out, gpr], &device)?,
                );
                tensors.insert(
                    format!("{prefix}.biases"),
                    f16_tensor(&m.biases, &[m.d_out, gpr], &device)?,
                );
                tensors.insert(
                    format!("{prefix}.qweight"),
                    Tensor::from_vec(m.packed, (m.d_out, m.d_in / 2), &device)?,
                );
            }
        }
        if i % 36 == 0 {
            eprintln!("  record {i:>3}/{}", head.matrices);
        }
    }

    // ---- the tensors the quantizer never touched, as the file stores them ----
    let n_raw = read_u32(&mut r)?;
    let mut raw_desc: Map<String, Value> = Map::new();
    let mut raw_digests: Map<String, Value> = Map::new();
    let mut carried_weights = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        carried_weights += t.len();
        match &t.data {
            RawData::F16(bits) => {
                raw_desc.insert(
                    t.name.clone(),
                    json!({"encoding": "f16", "dims": t.dims, "tensor": t.name}),
                );
                raw_digests.insert(t.name.clone(), json!({"values": sha256_u16(bits)}));
                tensors.insert(t.name.clone(), f16_tensor(bits, &t.dims, &device)?);
            }
            RawData::Quant(q) => {
                let row_len = t.dims.last().copied().unwrap_or(1).max(1);
                let rows = t.len() / row_len;
                let gpr = row_len.div_ceil(q.group);
                anyhow::ensure!(
                    q.bits != 4 || row_len % 2 == 0,
                    "{}: row length {row_len} is odd at 4 bits, so its nibbles do not split by row",
                    t.name
                );
                let prefix = tensor_prefix(&t.name).to_string();
                let cols = if q.bits == 4 { row_len / 2 } else { row_len };
                raw_desc.insert(
                    t.name.clone(),
                    json!({
                        "encoding": "quant",
                        "dims": t.dims,
                        "prefix": prefix,
                        "bits": q.bits,
                        "group": q.group,
                        "rows": rows,
                        "groups_per_row": gpr,
                    }),
                );
                raw_digests.insert(
                    t.name.clone(),
                    json!({
                        "qweight": sha256_hex(&q.packed),
                        "scales": sha256_u16(&q.scales),
                        "biases": sha256_u16(&q.biases),
                    }),
                );
                tensors.insert(
                    format!("{prefix}.scales"),
                    f16_tensor(&q.scales, &[rows, gpr], &device)?,
                );
                tensors.insert(
                    format!("{prefix}.biases"),
                    f16_tensor(&q.biases, &[rows, gpr], &device)?,
                );
                tensors.insert(
                    format!("{prefix}.qweight"),
                    Tensor::from_vec(q.packed.clone(), (rows, cols), &device)?,
                );
            }
        }
    }

    // ---- config and tokenizer ----
    let n_blob = read_u32(&mut r)?;
    let blobs = (0..n_blob)
        .map(|_| llvq_artifact::read_blob(&mut r))
        .collect::<Result<Vec<_>, _>>()?;
    let cfg = blobs
        .iter()
        .find(|b| b.name == CONFIG_FILE)
        .ok_or_else(|| anyhow::anyhow!("sealed file carries no {CONFIG_FILE}"))?;
    let base: Value = serde_json::from_slice(&cfg.bytes)?;
    anyhow::ensure!(
        base.is_object(),
        "{CONFIG_FILE} of the sealed file is not a JSON object"
    );

    let quantization_config = json!({
        "quant_method": QUANT_METHOD,
        "llvq_layout_version": LAYOUT_VERSION,
        "artifact_version": head.version,
        "artifact_sha256": src_sha,
        "artifact_kinds": head.kinds().to_string(),
        "codebook_fingerprint": head.codebook.map(|v| format!("{v:016x}")),
        "tetra_fingerprint": head.tetra.map(|v| format!("{v:016x}")),
        "block_dim": DIM,
        "code_order": CODE_ORDER,
        "records": Value::Object(records.clone()),
        "rotations": Value::Object(rotations.clone()),
        "raw": Value::Object(raw_desc.clone()),
    });

    std::fs::create_dir_all(out)?;
    let mut written = base.clone();
    written
        .as_object_mut()
        .expect("checked above")
        .insert("quantization_config".into(), quantization_config);
    let config_text = serde_json::to_vec_pretty(&written)?;
    std::fs::write(out.join(CONFIG_FILE), &config_text)?;

    let mut blob_digests: Map<String, Value> = Map::new();
    for b in &blobs {
        // A blob name is a path this code joins to the output directory. The
        // sealed files are ours, and that is not a reason to let a name out of
        // the directory it is written into.
        anyhow::ensure!(
            !b.name.contains('/') && !b.name.contains('\\') && b.name != ".." && !b.name.is_empty(),
            "blob name {:?} is not a plain file name",
            b.name
        );
        let verbatim = b.name != CONFIG_FILE;
        if verbatim {
            std::fs::write(out.join(&b.name), &b.bytes)?;
        }
        blob_digests.insert(
            b.name.clone(),
            json!({"sha256": sha256_hex(&b.bytes), "written_verbatim": verbatim}),
        );
    }

    // `transformers` reads generation defaults from the tokenizer config; the
    // sealed file does not carry one, and its absence makes some loaders warn
    // rather than fail. The same minimal file `bin/export` writes.
    let tok_cfg = out.join("tokenizer_config.json");
    if !tok_cfg.exists() {
        std::fs::write(
            &tok_cfg,
            br#"{"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 40960}"#,
        )?;
    }

    let digest = json!({
        "artifact": {
            // The file name and not the path: this file is published beside the
            // model, and a home directory has no business travelling with it.
            "file": src.file_name().map(|n| n.to_string_lossy().to_string()),
            "sha256": src_sha,
            "version": head.version,
            "matrices": head.matrices,
            "kinds": head.kinds().to_string(),
        },
        "convention": "sha256 over little-endian bit patterns, in the order the field stores them",
        "counts": {
            "lattice": lattice,
            "int4": int4,
            "raw_tensors": n_raw,
            "rotations": rotations.len(),
            "quantized_weights": quantized_weights,
            "carried_weights": carried_weights,
            "tensors": tensors.len(),
        },
        "records": Value::Object(record_digests),
        "rotations": Value::Object(rotation_digests),
        "raw": Value::Object(raw_digests),
        "blobs": Value::Object(blob_digests),
        "config_base": base,
    });
    std::fs::write(out.join(DIGEST_FILE), serde_json::to_vec_pretty(&digest)?)?;

    // ---- write, then read back and demand every bit ----
    let n_tensors = tensors.len();
    let model_path = out.join(MODEL_FILE);
    candle_core::safetensors::save(&tensors, &model_path)?;
    let model_bytes = std::fs::metadata(&model_path)?.len();
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
            got.dims() == want.dims() && got.dtype() == want.dtype(),
            "{name}: {:?} {:?} written, {:?} {:?} read back",
            want.dims(),
            want.dtype(),
            got.dims(),
            got.dtype()
        );
        let (a, b) = (bit_digest(want)?, bit_digest(got)?);
        anyhow::ensure!(a == b, "{name}: differs after the round trip through disk");
    }
    drop(back);

    let mut dir_bytes = 0u64;
    for e in std::fs::read_dir(out)? {
        dir_bytes += e?.metadata()?.len();
    }
    Ok(Summary {
        records: head.matrices,
        lattice,
        int4,
        rotations: rotations.len(),
        raw_tensors: n_raw,
        blobs: blobs.len(),
        tensors: n_tensors,
        quantized_weights,
        carried_weights,
        model_bytes,
        config_bytes: config_text.len() as u64,
        dir_bytes,
    })
}

/// SHA-256 of a tensor's values as bit patterns, little-endian.
///
/// Bit patterns and not floats: `==` on f16 calls two different NaNs equal and
/// `0.0 == -0.0`, and the claim being checked is that the disk round trip moved
/// nothing at all.
pub fn bit_digest(t: &Tensor) -> anyhow::Result<String> {
    let flat = t.flatten_all()?;
    let mut h = Sha256::new();
    match t.dtype() {
        DType::U8 => h.update(&flat.to_vec1::<u8>()?),
        DType::F16 => {
            for v in flat.to_vec1::<half::f16>()? {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        DType::F32 => {
            for v in flat.to_vec1::<f32>()? {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        DType::F64 => {
            for v in flat.to_vec1::<f64>()? {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        d => anyhow::bail!("no bit convention for dtype {d:?}"),
    }
    Ok(h.finish())
}

// ---------------------------------------------------------------------------
// The dense reference of stage 1
// ---------------------------------------------------------------------------

/// Name of the file [`dense_digest`] writes.
pub const DENSE_DIGEST_FILE: &str = "llvq-dense-digest.json";

/// What one dense digest run covered.
#[derive(Debug)]
pub struct DenseSummary {
    pub records: u32,
    pub quantized_tensors: u32,
    pub weights: usize,
}

/// One SHA-256 per record of the **dequantized f32 weights**, in the natural
/// basis, plus one per group-affine raw tensor.
///
/// This is the reference gate A of stage 1 compares against
/// (`proofs/preregistration-hf-quantizer-2026-09-30.md` §5). It is
/// [`llvq_artifact::decode_matrix`] and nothing else: the artifact's own
/// decoder, which rebuilds in f64, restores the tail, un-rotates and only then
/// narrows. A Python reader that agrees with this file has rebuilt our weights,
/// and a reader that agrees only to a tolerance has not.
///
/// The f16 raw tensors are not digested here. Stage 0 already compared their
/// bit patterns, and widening f16 to f32 is exact.
pub fn dense_digest(src: &Path, out: &Path) -> anyhow::Result<DenseSummary> {
    let f = std::fs::File::open(src)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{} is a projections-only artifact",
        src.display()
    );
    let cbs = llvq_artifact::Codebooks::new();
    let mut records: Map<String, Value> = Map::new();
    let mut weights = 0usize;
    for i in 0..head.matrices {
        let (name, w) = match llvq_artifact::read_record(&mut r, head.version)? {
            Record::Lattice(raw) => {
                let m = llvq_artifact::decode_raw(raw, &cbs)?;
                let w = llvq_artifact::decode_matrix(&m);
                (m.name, w)
            }
            Record::Int4(m) => {
                let w = m.to_f32();
                (m.name, w)
            }
        };
        weights += w.len();
        records.insert(name, json!(sha256_f32(&w)));
        if i % 36 == 0 {
            eprintln!("  record {i:>3}/{}", head.matrices);
        }
    }

    let n_raw = read_u32(&mut r)?;
    let mut raw: Map<String, Value> = Map::new();
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        if matches!(t.data, RawData::Quant(_)) {
            let w = t.to_f32();
            weights += w.len();
            raw.insert(t.name.clone(), json!(sha256_f32(&w)));
        }
    }
    let quantized_tensors = raw.len() as u32;

    let digest = json!({
        "artifact": {
            "file": src.file_name().map(|n| n.to_string_lossy().to_string()),
            "sha256": sha256_file(src)?,
            "version": head.version,
        },
        "convention": "sha256 over the dequantized f32 values, little-endian, row-major",
        "reference": "llvq_artifact::decode_matrix, and RawTensor::to_f32 for a quantized raw tensor",
        "records": Value::Object(records.clone()),
        "raw": Value::Object(raw),
    });
    std::fs::write(out, serde_json::to_vec_pretty(&digest)?)?;
    Ok(DenseSummary {
        records: head.matrices,
        quantized_tensors,
        weights,
    })
}

// ---------------------------------------------------------------------------
// The universal decode tables
// ---------------------------------------------------------------------------

/// The Tetra map as data, for a reader that does not run Rust.
///
/// The map is a property of the codebook and not of a model, which is why the
/// `.llvq` header carries a fingerprint and no table
/// ([`llvq_artifact::tetra_fingerprint`]). So these tables ship with the Python
/// package rather than with every published model, and the package refuses a
/// file whose fingerprint is not the one they were dumped under.
///
/// Everything here comes from [`llvq_search::tetra`]'s own accessors. Nothing is
/// recomputed, so there is no second derivation to keep in step.
pub fn tetra_tables(out: &Path) -> anyhow::Result<(String, usize)> {
    use llvq_search::tetra::{self, Tetra};
    let t = Tetra::new();
    let device = Device::Cpu;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let order: Vec<u8> = t.order().iter().map(|&v| v as u8).collect();
    tensors.insert("order".into(), Tensor::from_vec(order, DIM, &device)?);

    let flat = |t: &[[u8; 2]; 64]| -> Vec<u8> { t.iter().flatten().copied().collect() };
    tensors.insert(
        "prefixes".into(),
        Tensor::from_vec(flat(t.prefixes()), (64, 2), &device)?,
    );
    tensors.insert(
        "suffixes".into(),
        Tensor::from_vec(flat(t.suffixes()), (64, 2), &device)?,
    );
    let (mut c2, mut s16) = (Vec::new(), Vec::new());
    for state in t.branches() {
        for &(byte, next) in state {
            c2.push(byte);
            s16.push(next);
        }
    }
    tensors.insert(
        "branch_c2".into(),
        Tensor::from_vec(c2.clone(), (64, tetra::BRANCHES), &device)?,
    );
    // The shader's own layout: one u16 per edge, `byte | s16 << 8`
    // (`F1rTables::branches` in `llvq-llm/kernels/llvq_tetra48.metal`). Written
    // here and not repacked in Python, so the layout has one owner.
    let branches_u16: Vec<u32> = c2
        .iter()
        .zip(&s16)
        .map(|(&byte, &next)| byte as u32 | ((next as u32) << 8))
        .collect();
    tensors.insert(
        "branches_u16".into(),
        Tensor::from_vec(branches_u16, (64, tetra::BRANCHES), &device)?,
    );
    tensors.insert(
        "branch_s16".into(),
        Tensor::from_vec(s16, (64, tetra::BRANCHES), &device)?,
    );
    tensors.insert(
        "rows".into(),
        Tensor::from_vec(t.rows().to_vec(), tetra::ROWS, &device)?,
    );
    // `val(o, rho)`: the rho-th value of o + 4Z, outward from zero.
    let values: Vec<i64> = (0..4)
        .flat_map(|o| (0..8).map(move |rho| tetra::val(o, rho) as i64))
        .collect();
    tensors.insert("values".into(), Tensor::from_vec(values, (4, 8), &device)?);

    // Flat, as the shader binds them: [s8][b1] and [s16][b3], 128 bytes each.
    tensors.insert(
        "prefixes_flat".into(),
        Tensor::from_vec(flat(t.prefixes()), 128, &device)?,
    );
    tensors.insert(
        "suffixes_flat".into(),
        Tensor::from_vec(flat(t.suffixes()), 128, &device)?,
    );

    let n = tensors.len();
    candle_core::safetensors::save(&tensors, out)?;

    // The served shader, copied beside the tables with its digest. One MSL
    // source in the repository, and a copy the package can ship after the
    // extraction of stage 5; `the_shipped_shader_is_the_repositorys` compares
    // them in the fast loop.
    let shader_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/llvq_tetra48.metal");
    let shader = std::fs::read(&shader_src)?;
    let shader_sha = sha256_hex(&shader);
    let shader_out = out.with_file_name("llvq_tetra48.metal");
    std::fs::write(&shader_out, &shader)?;

    // The constants a reader needs beside the arrays, and the fingerprint that
    // says which map they are.
    let fingerprint = format!("{:016x}", llvq_artifact::tetra_fingerprint());
    let meta = json!({
        "tetra_fingerprint": fingerprint,
        "shader": {
            "file": "llvq_tetra48.metal",
            "sha256": shader_sha,
            "entry": "tetra48_probe",
            "note": "one thread a block, no tile, no reduction: the decoder judged on its own",
        },
        "label_bits": tetra::LABEL_BITS,
        "word_bits": tetra::WORD_BITS,
        "class_rows": tetra::CLASS_ROWS,
        "n0_mixed": tetra::N0_MIXED,
        "section": tetra::SECTION,
        "dim": DIM,
        "fields": tetra::LAYOUT.iter().map(|(n, (lo, w))| json!({"name": n, "lo": lo, "width": w}))
            .collect::<Vec<_>>(),
        "note": "dumped by llvq-llm/src/bin/tetratables.rs from llvq_search::tetra; \
                 the .llvq header's tetra fingerprint must equal the one above",
    });
    let json_path = out.with_extension("json");
    std::fs::write(&json_path, serde_json::to_vec_pretty(&meta)?)?;
    Ok((fingerprint, n))
}

/// SHA-256 of a byte slice, lowercase hex.
///
/// Public for `the_shipped_shader_is_the_repositorys`, which compares the digest
/// recorded beside the tables against the shader's own bytes.
pub fn sha256_bytes(data: &[u8]) -> String {
    sha256_hex(data)
}

/// SHA-256 over f32 values, little-endian.
fn sha256_f32(v: &[f32]) -> String {
    let mut h = Sha256::new();
    for chunk in v.chunks(1 << 16) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for x in chunk {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        h.update(&bytes);
    }
    h.finish()
}
