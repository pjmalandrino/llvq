//! Local, checkpoint-prefix pilot for gains-only Tetra diagnostics.
//!
//! Capture never replaces model weights. Replay quantizes sampled rows with a
//! source-norm GPTQ witness. This is an optimizer diagnostic, not a sequential
//! full-model quantization or a quality evaluation of a served artifact.

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use llvq_core::{SplitMix64, DIM};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::fit_gain_centroids;
use llvq_quant::rotation::Rotation;
use llvq_quant::schur::{diagnose_row, GainComparison, RowDiagnostic};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::calib::{effective_rotation_seed, shrink_off_diagonal, Hessian};
use crate::loader::Checkpoint;
use crate::model::{Act, Capture, NoCapture, Qwen3};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u32,
    pub checkpoint: PathBuf,
    pub revision: String,
    pub calibration: PathBuf,
    pub validation: PathBuf,
    pub device: String,
    pub layers: Vec<usize>,
    pub projections: Vec<String>,
    pub rows_per_projection: usize,
    pub snapshots_per_row: usize,
    pub damping: f64,
    pub h_shrink: f64,
    pub rotation_seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Windows {
    pub source: PathBuf,
    pub source_fingerprint_fnv1a: String,
    pub seed: u64,
    pub offsets: Vec<usize>,
    pub ids: Vec<Vec<u32>>,
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(BufReader::new(
        File::open(path).with_context(|| path.display().to_string())?,
    ))
    .with_context(|| path.display().to_string())
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| path.display().to_string())?;
    let mut out = BufWriter::new(f);
    serde_json::to_writer(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn projection_act(name: &str) -> Result<Act> {
    Act::ALL
        .into_iter()
        .find(|a| a.consumers().contains(&name))
        .ok_or_else(|| anyhow::anyhow!("unknown projection {name}"))
}

/// Deterministic, distinct positions spanning the entire population.
/// Selection uses dimensions only, before any diagnostic outcomes exist.
pub fn positions(total: usize, count: usize) -> Result<Vec<usize>> {
    ensure!(
        count > 0 && count <= total,
        "requested {count} positions in {total}"
    );
    Ok(if count == 1 {
        vec![total / 2]
    } else {
        (0..count).map(|i| i * (total - 1) / (count - 1)).collect()
    })
}

pub fn validate_plan(plan: &Plan, ck: &Checkpoint) -> Result<()> {
    ensure!(plan.version == 1, "unsupported diagnostic plan version");
    ensure!(
        plan.revision.len() == 40 && plan.revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "revision must be a full checkpoint commit"
    );
    ensure!(
        plan.checkpoint.file_name().and_then(|s| s.to_str()) == Some(&plan.revision),
        "use the pinned local snapshot directory whose basename equals revision"
    );
    ensure!(
        matches!(plan.device.as_str(), "cpu" | "metal"),
        "pilot supports cpu or metal"
    );
    ensure!(
        !plan.layers.is_empty() && plan.layers.windows(2).all(|p| p[0] < p[1]),
        "layers must be sorted and unique"
    );
    ensure!(
        *plan.layers.last().unwrap() < ck.config.num_hidden_layers,
        "layer outside checkpoint"
    );
    ensure!(!plan.projections.is_empty(), "no projections");
    ensure!(
        plan.damping.is_finite() && plan.damping >= 0.0,
        "invalid damping"
    );
    ensure!((0.0..=1.0).contains(&plan.h_shrink), "invalid shrinkage");
    for (i, p) in plan.projections.iter().enumerate() {
        ensure!(
            !plan.projections[..i].contains(p),
            "duplicate projection {p}"
        );
        ensure!(
            p != "self_attn.v_proj",
            "v_proj is excluded from the mixed-recipe pilot"
        );
        let act = projection_act(p)?;
        positions(act.width(&ck.config) / DIM, plan.snapshots_per_row)?;
        let rows = match p.as_str() {
            "self_attn.q_proj" => ck.config.num_attention_heads * ck.config.head_dim,
            "self_attn.k_proj" => ck.config.num_key_value_heads * ck.config.head_dim,
            "mlp.gate_proj" | "mlp.up_proj" => ck.config.intermediate_size,
            _ => ck.config.hidden_size,
        };
        positions(rows, plan.rows_per_projection)?;
    }
    Ok(())
}

fn validate_windows(w: &Windows, vocab: usize, max_context: usize) -> Result<()> {
    ensure!(
        !w.ids.is_empty() && w.ids.len() == w.offsets.len(),
        "empty windows or missing offsets"
    );
    let len = w.ids[0].len();
    ensure!(
        len >= 8 && len <= max_context,
        "invalid window length {len}"
    );
    let mut intervals: Vec<_> = w
        .offsets
        .iter()
        .map(|&s| (s, s.saturating_add(len)))
        .collect();
    intervals.sort_unstable();
    ensure!(
        intervals.windows(2).all(|p| p[0].1 <= p[1].0),
        "overlapping token windows"
    );
    ensure!(
        w.ids
            .iter()
            .all(|v| v.len() == len && v.iter().all(|&id| (id as usize) < vocab)),
        "invalid token windows"
    );
    Ok(())
}

pub fn load_inputs(plan: &Plan, ck: &Checkpoint) -> Result<(Windows, Windows)> {
    let c: Windows = read_json(&plan.calibration)?;
    let v: Windows = read_json(&plan.validation)?;
    validate_windows(&c, ck.config.vocab_size, ck.config.max_position_embeddings)?;
    validate_windows(&v, ck.config.vocab_size, ck.config.max_position_embeddings)?;
    ensure!(
        c.source != v.source && c.source_fingerprint_fnv1a != v.source_fingerprint_fnv1a,
        "calibration and validation must use distinct corpus files"
    );
    ensure!(
        c.ids.iter().all(|a| !v.ids.contains(a)),
        "calibration window reused in validation"
    );
    Ok((c, v))
}

/// Read-only resource accounting. No tensors, model inference or factorization.
pub fn inspect(plan: &Plan) -> Result<Value> {
    let ck = Checkpoint::from_dir(&plan.checkpoint)?;
    validate_plan(plan, &ck)?;
    let (c, v) = load_inputs(plan, &ck)?;
    let mut calls = 0usize;
    let mut site_bytes = 0usize;
    for p in &plan.projections {
        let n = projection_act(p)?.width(&ck.config);
        let blocks = n / DIM;
        let selected = positions(blocks, plan.snapshots_per_row)?;
        calls += plan.rows_per_projection
            * (blocks + 2 * selected.iter().map(|b| blocks - b - 1).sum::<usize>());
        site_bytes += 8
            * (n * n
                + v.ids.iter().map(Vec::len).sum::<usize>() * n
                + plan.rows_per_projection * n);
    }
    Ok(
        json!({"label":"computed, no inference", "prefix":"checkpoint-f32",
        "projection_count":plan.layers.len()*plan.projections.len(),
        "row_count":plan.layers.len()*plan.projections.len()*plan.rows_per_projection,
        "encoder_calls_upper_bound":calls*plan.layers.len(),
        "bundle_array_bytes":site_bytes*plan.layers.len(),
        "calibration_tokens":c.ids.iter().map(Vec::len).sum::<usize>(),
        "validation_tokens":v.ids.iter().map(Vec::len).sum::<usize>(),
        "checkpoint_disk_bytes":ck.weights.iter().map(|p|fs::metadata(p).map(|m|m.len())).collect::<std::io::Result<Vec<_>>>()?.iter().sum::<u64>(),
        "device":plan.device, "dtype":"f32", "gain_policy":"source-norm witness",
        "tail":"KeepExact", "group_scales":false, "design_c":false,
        "wall_time":"unmeasured"}),
    )
}

fn fingerprint(path: &Path) -> Result<String> {
    let mut f = BufReader::new(File::open(path)?);
    let mut buf = [0u8; 65536];
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for b in &buf[..n] {
            hash = (hash ^ *b as u64).wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayFile {
    pub name: String,
    pub count: usize,
    pub fingerprint_fnv1a: String,
}

pub fn write_array(dir: &Path, name: &str, values: &[f64]) -> Result<ArrayFile> {
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite {name}");
    let path = dir.join(name);
    let mut f = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?,
    );
    for x in values {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok(ArrayFile {
        name: name.into(),
        count: values.len(),
        fingerprint_fnv1a: fingerprint(&path)?,
    })
}

pub fn read_array(dir: &Path, meta: &ArrayFile) -> Result<Vec<f64>> {
    ensure!(
        Path::new(&meta.name).components().count() == 1 && !meta.name.starts_with('.'),
        "array name must be a filename"
    );
    let path = dir.join(&meta.name);
    ensure!(
        fs::metadata(&path)?.len()
            == (meta.count as u64)
                .checked_mul(8)
                .context("array size overflow")?,
        "wrong byte length: {}",
        path.display()
    );
    ensure!(
        fingerprint(&path)? == meta.fingerprint_fnv1a,
        "fingerprint mismatch: {}",
        path.display()
    );
    let data = fs::read(&path)?;
    let out: Vec<_> = data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(out.iter().all(|v| v.is_finite()), "non-finite array");
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub version: u32,
    pub plan: Plan,
    pub provenance: Value,
    pub layer: usize,
    pub projection: String,
    pub width: usize,
    pub row_ids: Vec<usize>,
    pub snapshot_blocks: Vec<usize>,
    pub centroids: [f64; 2],
    /// Rotated, shrunk H before damping. Replay applies damping exactly once.
    pub hessian: ArrayFile,
    pub validation: ArrayFile,
    pub original_rows: ArrayFile,
}

struct SiteCapture {
    layer: usize,
    hessians: HashMap<Act, Hessian>,
    validation: HashMap<Act, Vec<f64>>,
    held_out: bool,
}
impl Capture for SiteCapture {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == self.layer {
            if self.held_out {
                if let Some(out) = self.validation.get_mut(&act) {
                    out.extend(
                        x.to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .into_iter()
                            .map(|v| v as f64),
                    );
                }
            } else if let Some(h) = self.hessians.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

fn embed(model: &Qwen3, windows: &Windows, device: &Device) -> Result<Vec<Tensor>> {
    windows
        .ids
        .iter()
        .map(|ids| Ok(model.embed_tokens(&Tensor::from_slice(ids, (1, ids.len()), device)?)?))
        .collect()
}

fn advance(
    model: &Qwen3,
    layer: usize,
    hidden: &mut [Tensor],
    capture: &mut dyn Capture,
) -> Result<()> {
    for h in hidden {
        let mask = model.causal_mask_for(h)?;
        *h = model.blocks[layer].forward(h, model.rotary(), &mask, layer, capture)?;
    }
    Ok(())
}

/// Capture the selected checkpoint projections. Oracle runs first on this backend.
/// A fresh output directory is required; no model download and no resume.
pub fn capture(plan: &Plan, out: &Path) -> Result<()> {
    let ck = Checkpoint::from_dir(&plan.checkpoint)?;
    validate_plan(plan, &ck)?;
    let (calib, validation) = load_inputs(plan, &ck)?;
    fs::create_dir(out)
        .with_context(|| format!("fresh output directory required: {}", out.display()))?;
    let begin = Instant::now();
    write_json(&out.join("plan.json"), plan)?;
    write_json(&out.join("calibration.json"), &calib)?;
    write_json(&out.join("validation.json"), &validation)?;
    let device = crate::eval::device(&plan.device)?;
    let vb = ck.var_builder(DType::F32, &device)?;
    let model = Qwen3::new(&ck.config, vb.clone(), crate::kvq::KvMode::F16)?;
    let oracle_ids = &calib.ids[0][..calib.ids[0].len().min(32)];
    let input = Tensor::from_slice(oracle_ids, (1, oracle_ids.len()), &device)?;
    let mine = model.hidden(&input, &mut NoCapture)?;
    let mut reference = candle_transformers::models::qwen3::Model::new(&ck.config, vb)?;
    let theirs = reference.forward(&input, 0)?;
    let delta: f32 = (&mine - &theirs)?.abs()?.max_all()?.to_scalar()?;
    let scale: f32 = theirs.abs()?.max_all()?.to_scalar()?;
    let relative = delta / scale.max(1e-6);
    write_json(
        &out.join("oracle.json"),
        &json!({"max_abs":delta,"relative":relative,"threshold":1e-4}),
    )?;
    ensure!(
        relative < 1e-4,
        "oracle divergence {relative}; capture refused"
    );
    drop(reference);
    eprintln!("oracle MATCH on {}, relative {relative:e}", plan.device);
    let mut files = Vec::new();
    for path in ck.weights.iter().chain([&ck.config_path, &ck.tokenizer]) {
        files.push(json!({"path":path,"bytes":fs::metadata(path)?.len(),"fingerprint_fnv1a":fingerprint(path)?}));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let git = std::process::Command::new("git")
        .current_dir(root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    ensure!(git.status.success(), "git revision unavailable");
    let diff = std::process::Command::new("git")
        .current_dir(root)
        .args(["diff", "HEAD", "--"])
        .output()?;
    ensure!(diff.status.success(), "git diff unavailable");
    fs::write(out.join("working-tree.patch"), &diff.stdout)?;
    // Include untracked diagnostic sources as well as the tracked diff.
    let mut sources = Vec::new();
    for name in [
        "llvq-quant/src/schur.rs",
        "llvq-quant/src/quantizer.rs",
        "llvq-llm/src/tetra_diag.rs",
        "llvq-llm/src/bin/tetra_schur.rs",
    ] {
        let bytes = fs::read(root.join(name))?;
        let dest = name.replace('/', "_");
        fs::write(out.join(&dest), bytes)?;
        sources.push(json!({"name":name,"fingerprint_fnv1a":fingerprint(&root.join(name))?}));
    }
    let executable = std::env::current_exe()?;
    let provenance = json!({"git_commit":String::from_utf8(git.stdout)?.trim(),
        "executable":executable,"executable_fingerprint_fnv1a":fingerprint(&executable)?,"source_files":sources,"checkpoint_files":files,"prefix":"checkpoint-f32",
        "dtype":"f32","tail":"KeepExact","continuation":"source-norm",
        "calibration_seed":calib.seed,"validation_seed":validation.seed,
        "calibration_tokens":calib.ids.iter().map(|v|format!("{:016x}",crate::eval::token_fingerprint(v))).collect::<Vec<_>>(),
        "validation_tokens":validation.ids.iter().map(|v|format!("{:016x}",crate::eval::token_fingerprint(v))).collect::<Vec<_>>(),
        "oracle_relative":relative,"started_unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()});
    write_json(&out.join("provenance.json"), &provenance)?;
    let mut hidden = embed(&model, &calib, &device)?;
    let mut reserved = embed(&model, &validation, &device)?;
    let total_rows = calib.ids.iter().map(Vec::len).sum();
    let acts: Vec<_> = Act::ALL
        .into_iter()
        .filter(|a| {
            plan.projections
                .iter()
                .any(|p| a.consumers().contains(&p.as_str()))
        })
        .collect();
    for layer in 0..=*plan.layers.last().unwrap() {
        if !plan.layers.contains(&layer) {
            advance(&model, layer, &mut hidden, &mut NoCapture)?;
            advance(&model, layer, &mut reserved, &mut NoCapture)?;
            continue;
        }
        let mut cap = SiteCapture {
            layer,
            hessians: HashMap::new(),
            validation: HashMap::new(),
            held_out: false,
        };
        for &act in &acts {
            cap.hessians.insert(
                act,
                Hessian::new(act.width(&ck.config), &device, total_rows)?,
            );
            cap.validation.insert(act, Vec::new());
        }
        advance(&model, layer, &mut hidden, &mut cap)?;
        cap.held_out = true;
        advance(&model, layer, &mut reserved, &mut cap)?;
        for &act in &acts {
            let n = act.width(&ck.config);
            let mut h = cap.hessians.remove(&act).unwrap().to_f64()?;
            shrink_off_diagonal(&mut h, n, plan.h_shrink);
            let rot = Rotation::new(n, effective_rotation_seed(plan.rotation_seed, layer, act));
            rot.rotate_hessian(&mut h);
            let mut validation = cap.validation.remove(&act).unwrap();
            let validation_count = validation.len() / n;
            rot.rotate_weight_rows(&mut validation, validation_count);
            for p in plan
                .projections
                .iter()
                .filter(|p| act.consumers().contains(&p.as_str()))
            {
                let weights = model.blocks[layer].linear(p).weight();
                let (rows, width) = weights.dims2()?;
                ensure!(width == n, "projection width mismatch");
                let mut full: Vec<_> = weights
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                rot.rotate_weight_rows(&mut full, rows);
                let centroids = fit_gain_centroids(&full, rows, n, DIM, 1, 40);
                let row_ids = positions(rows, plan.rows_per_projection)?;
                let selected: Vec<_> = row_ids
                    .iter()
                    .flat_map(|&r| full[r * n..(r + 1) * n].iter().copied())
                    .collect();
                let stem = format!("layer-{layer}-{}", p.replace('.', "-"));
                let bundle = Bundle {
                    version: 1,
                    plan: plan.clone(),
                    provenance: provenance.clone(),
                    layer,
                    projection: p.clone(),
                    width: n,
                    row_ids,
                    snapshot_blocks: positions(n / DIM, plan.snapshots_per_row)?,
                    centroids: centroids.try_into().unwrap(),
                    hessian: write_array(out, &format!("{stem}-h.f64le"), &h)?,
                    validation: write_array(out, &format!("{stem}-validation.f64le"), &validation)?,
                    original_rows: write_array(out, &format!("{stem}-rows.f64le"), &selected)?,
                };
                write_json(&out.join(format!("{stem}.json")), &bundle)?;
                eprintln!(
                    "captured layer {layer}, {p}, {} rows, elapsed {:.1}s",
                    plan.rows_per_projection,
                    begin.elapsed().as_secs_f64()
                );
            }
        }
    }
    write_json(
        &out.join("capture-complete.json"),
        &json!({"seconds":begin.elapsed().as_secs_f64(),"label":"measured capture wall time"}),
    )?;
    Ok(())
}

fn comparison_json(c: &GainComparison) -> Value {
    json!({"block":c.block,"source_norm":c.source_norm,"projected_gain":c.projected_gain,
        "amplitudes":c.amplitudes,"point":c.codes[0].point,"choices_ABC":c.choices,
        "euclidean":c.euclidean,"conditional":c.conditional,"prefix_loss":c.prefix_loss})
}

fn row_json(d: &RowDiagnostic) -> Value {
    let code_json = |c: &llvq_quant::quantizer::BlockCode| json!({"point":c.point,"gain":c.gain});
    json!({"original":d.original,"row_scale":d.row_scale,"encoder_calls_upper_bound":d.encoder_calls,
        "shadow":d.shadow.iter().map(comparison_json).collect::<Vec<_>>(),
        "witness":d.witness,"witness_codes":d.witness_codes.iter().map(code_json).collect::<Vec<_>>(),
        "snapshots":d.snapshots.iter().map(|s|json!({"working":s.working,
            "prefix_codes":s.prefix_codes.iter().map(code_json).collect::<Vec<_>>(),
            "comparison":comparison_json(&s.comparison),
            "branches":s.branches.iter().map(|b|json!({"gain":b.gain,"codes":b.codes.iter().map(code_json).collect::<Vec<_>>(),
                "reconstructed":b.reconstructed,"rollout_loss":b.rollout_loss,
                "continuous_lower_bound":b.continuous_lower_bound,"rollout_excess":b.rollout_excess,
                "validation_loss":b.validation_loss,"encoder_calls_upper_bound":b.encoder_calls})).collect::<Vec<_>>()
        })).collect::<Vec<_>>()})
}

/// Replay one retained projection without loading a model or changing the bundle.
pub fn replay(bundle_path: &Path, out: &Path) -> Result<()> {
    let bundle: Bundle = read_json(bundle_path)?;
    ensure!(bundle.version == 1, "unsupported bundle version");
    let n = bundle.width;
    ensure!(
        n >= DIM && bundle.hessian.count == n.checked_mul(n).context("dimension overflow")?,
        "invalid Hessian shape"
    );
    ensure!(
        !bundle.row_ids.is_empty() && bundle.row_ids.windows(2).all(|p| p[0] < p[1]),
        "invalid row IDs"
    );
    ensure!(
        bundle.original_rows.count
            == n.checked_mul(bundle.row_ids.len())
                .context("row count overflow")?,
        "invalid rows shape"
    );
    ensure!(
        bundle.validation.count > 0 && bundle.validation.count.is_multiple_of(n),
        "invalid validation shape"
    );
    ensure!(
        bundle.centroids.iter().all(|v| v.is_finite() && *v >= 0.0)
            && bundle.centroids[0] <= bundle.centroids[1],
        "invalid centroids"
    );
    ensure!(
        !bundle.snapshot_blocks.is_empty()
            && bundle.snapshot_blocks.iter().all(|&b| b < n / DIM)
            && bundle.snapshot_blocks.windows(2).all(|b| b[0] < b[1]),
        "invalid snapshot blocks"
    );
    ensure!(
        bundle.plan.damping.is_finite() && bundle.plan.damping >= 0.0,
        "invalid damping"
    );
    let dir = bundle_path.parent().context("bundle directory")?;
    let h = read_array(dir, &bundle.hessian)?;
    let validation = read_array(dir, &bundle.validation)?;
    let rows = read_array(dir, &bundle.original_rows)?;
    let begin = Instant::now();
    for i in 0..n {
        for j in 0..i {
            ensure!(
                (h[i * n + j] - h[j * n + i]).abs()
                    <= 1e-12 * h[i * n + i].abs().max(h[j * n + j].abs()).max(1.0),
                "asymmetric Hessian at ({i},{j})"
            );
        }
    }
    let factor = GptqFactor::new(&h, n, bundle.plan.damping)?;
    let factor_seconds = begin.elapsed().as_secs_f64();
    fs::create_dir(out)
        .with_context(|| format!("fresh output directory required: {}", out.display()))?;
    write_json(&out.join("bundle.json"), &bundle)?;
    let mut disagreements = [0usize; 3]; // A/B, A/C, B/C
    let mut regret = [0.0; 3];
    let mut validation_regret = [0.0; 3];
    let mut snapshots = 0;
    let mut shadow_count = 0;
    let mut calls = 0;
    for (&row_id, row) in bundle.row_ids.iter().zip(rows.chunks_exact(n)) {
        let start = Instant::now();
        let d = diagnose_row(
            row,
            &factor,
            bundle.centroids,
            &validation,
            &bundle.snapshot_blocks,
        );
        for c in &d.shadow {
            shadow_count += 1;
            for (i, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                disagreements[i] += usize::from(c.choices[a] != c.choices[b]);
            }
        }
        for s in &d.snapshots {
            snapshots += 1;
            let best = s
                .branches
                .iter()
                .map(|b| b.rollout_loss)
                .fold(f64::INFINITY, f64::min);
            let val_best = s
                .branches
                .iter()
                .map(|b| b.validation_loss)
                .fold(f64::INFINITY, f64::min);
            for arm in 0..3 {
                regret[arm] += s.branches[s.comparison.choices[arm]].rollout_loss - best;
                validation_regret[arm] +=
                    s.branches[s.comparison.choices[arm]].validation_loss - val_best;
            }
            for b in &s.branches {
                ensure!(
                    b.rollout_loss.is_finite() && b.validation_loss.is_finite(),
                    "non-finite rollout"
                );
                ensure!(
                    b.rollout_excess >= -2e-9 * b.rollout_loss.abs().max(1.0),
                    "rollout below continuous bound"
                );
            }
        }
        calls += d.encoder_calls;
        let seconds = start.elapsed().as_secs_f64();
        write_json(
            &out.join(format!("row-{row_id}.json")),
            &json!({"row":row_id,"seconds":seconds,"diagnostic":row_json(&d)}),
        )?;
        eprintln!("{} row {row_id}, {seconds:.2}s", bundle.projection);
    }
    write_json(
        &out.join("summary.json"),
        &json!({"layer":bundle.layer,"projection":bundle.projection,
        "calibration_seed":bundle.provenance["calibration_seed"],"row_count":bundle.row_ids.len(),
        "shadow_count":shadow_count,"disagreements_AB_AC_BC":disagreements,
        "snapshot_count":snapshots,"mean_rollout_regret_ABC":regret.map(|v|v/snapshots as f64),
        "mean_validation_regret_ABC":validation_regret.map(|v|v/snapshots as f64),
        "factor_seconds":factor_seconds,"seconds":begin.elapsed().as_secs_f64(),
        "encoder_calls_upper_bound":calls,"label":"measured diagnostic, not MMLU",
        "array_bytes_computed":8*(h.len()+validation.len()+rows.len()),
        "inference_unit":"projection x depth x seed; blocks are not independent replicates"}),
    )?;
    Ok(())
}

/// Prepare deterministic, non-overlapping windows from a LOCAL parquet text column.
/// Reads at most a fixed prefix of 2 million characters. No model inference.
pub fn prepare_tokens(
    checkpoint: &Path,
    source: &Path,
    out: &Path,
    count: usize,
    len: usize,
    seed: u64,
) -> Result<()> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;
    ensure!(count > 0 && len >= 8, "invalid window count or length");
    let ck = Checkpoint::from_dir(checkpoint)?;
    let tokenizer = ck.tokenizer()?;
    let reader = SerializedFileReader::new(File::open(source)?)?;
    let mut text = String::new();
    let mut chars = 0;
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let Some(col) = row.get_column_iter().position(|(name, _)| name == "text") else {
            bail!("parquet has no text column");
        };
        let s = row.get_string(col)?;
        let remaining = 2_000_000 - chars;
        let part: String = s.chars().take(remaining).collect();
        chars += part.chars().count();
        text.push_str(&part);
        if chars >= 2_000_000 {
            break;
        }
        text.push('\n');
        chars += 1;
    }
    let ids = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let available = ids.len() / len;
    ensure!(
        count <= available,
        "only {available} complete windows available"
    );
    let mut indices: Vec<_> = (0..available).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..indices.len()).rev() {
        let j = (rng.next() % (i + 1) as u64) as usize;
        indices.swap(i, j);
    }
    indices.truncate(count);
    let offsets: Vec<_> = indices.iter().map(|i| i * len).collect();
    let windows = Windows {
        source: fs::canonicalize(source)?,
        source_fingerprint_fnv1a: fingerprint(source)?,
        seed,
        ids: offsets.iter().map(|&i| ids[i..i + len].to_vec()).collect(),
        offsets,
    };
    write_json(out, &windows)
}
