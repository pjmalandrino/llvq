//! The Inkling 2 Mini artifact: one `.llvq` file in the repository's own
//! format, unchanged, plus a sidecar safetensors for what that format cannot
//! carry.
//!
//! Three tools of lot 1 share this module:
//!
//! * the seal (`bin/inkseal`, item 2): gathers the per-expert records that
//!   `bin/extq` left in `layers/Lxx/codes/eNNN.llvqrec`, any further record
//!   files (attention, `bin/inkwh`) and raw tables (`bin/inktable`) into one
//!   v5 file, by record passthrough ([`ArtifactWriter::push_record`], byte
//!   for byte what the record files hold);
//! * the generic `(W, H)` driver (`bin/inkwh`, item 1b): the sealed Tetra
//!   recipe of [`crate::extq`] or the int4 g128 record of
//!   [`crate::sealed::int4_record`] on any matrix given as raw files;
//! * the tables (`bin/inktable`, item 1c): [`crate::embedquant::quantize_affine`]
//!   at int4 g64 on a table read from a safetensors file.
//!
//! ## Names
//!
//! The native checkpoint names its text model `model.llm.*`. Every name this
//! module writes, in the sealed file and in the sidecar alike, goes through
//! [`rename`]: the prefix `model.llm.` becomes `model.`, and nothing else
//! changes. The vision tower keeps `model.visual.*`. The rule is a bijection
//! on the native names (no native name starts with `model.layers.`,
//! `model.embed` or `model.norm` before it is applied), so a reader can map
//! back. Under it the expert records are named
//! `model.layers.{L}.mlp.experts.{e}.{gate|up|down}_proj.weight`, which is
//! what `bin/extq` already writes, and
//! [`llvq_artifact::split_name`] reads them as block `L`, projection
//! `mlp.experts.{e}.gate_proj`: it takes everything between the block index
//! and the final `weight`, so the extra dotted level is kept, not refused.
//! Attention records are `model.layers.{L}.attn.{wq_du|...}.weight`, the
//! tables `model.embed.weight` and `model.unembed.weight`.
//!
//! ## What goes where
//!
//! Every tensor of the native checkpoint ends in exactly one place, and the
//! set is a complement, as in `bin/seal`: a native tensor is **covered** when
//! the sealed file carries its replacement (a record of the same renamed name,
//! a raw table of the same renamed name, or, for the expert tensors
//! `w13_weight*` and `w2_weight*` of a layer, the gate and up, or the down,
//! records of all its experts). An uncovered tensor in BF16 or F32 goes to the
//! sidecar as its native bytes, renamed; one in F16 goes to the sealed raw
//! section unchanged; any other dtype (the NVFP4 `U8` and `F8_E4M3` tensors of
//! an expert layer whose records are missing) is refused, unless the seal is
//! declared partial, in which case it is omitted and counted in the sidecar's
//! metadata and in the `inkling-seal.json` blob. `--text-only` omits
//! `model.visual.*` the same way, counted.
//!
//! No field, tag or record shape of the format moves: the file is a v5 file
//! with a header declaring the kinds its records use, records, raw tensors in
//! the existing two encodings, and blobs.

use crate::extq::{encode_matrix, factor_hessian, Settings};
use anyhow::{bail, ensure, Context, Result};
use llvq_artifact::{
    decode_matrix, read_header, read_raw, read_record, write_matrix_with, write_raw, ArtifactWriter,
    Blob, CodeKind, Codebooks, KindSet, QuantizedMatrix, RawData, RawMatrix, RawTensor, Record,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The format version every record of this module is read and written at.
pub const VERSION: u32 = llvq_artifact::VERSION;

/// The native text-model prefix, and what [`rename`] makes of it.
pub const NATIVE_PREFIX: &str = "model.llm.";
pub const SEALED_PREFIX: &str = "model.";

/// The name of the blob that describes the seal (sidecar, omissions).
pub const SEAL_BLOB: &str = "inkling-seal.json";

/// `model.llm.X` → `model.X`; any other name unchanged.
pub fn rename(native: &str) -> String {
    match native.strip_prefix(NATIVE_PREFIX) {
        Some(rest) => format!("{SEALED_PREFIX}{rest}"),
        None => native.to_string(),
    }
}

/// The record name `bin/extq` writes for one expert matrix.
pub fn expert_name(layer: u64, expert: u64, proj: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight")
}

/// The three matrices of an expert, in the order `bin/extq` writes them.
pub const EXPERT_PROJS: [&str; 3] = ["gate", "up", "down"];

// ---------------------------------------------------------------------------
// safetensors, read and written by hand: the header is JSON (serde_json is
// already a dependency), the data is copied as bytes. No new crate.
// ---------------------------------------------------------------------------

/// One tensor of a safetensors file: absolute byte range in the file.
#[derive(Clone, Debug)]
pub struct StTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl StTensor {
    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
    }
}

/// A safetensors file's header.
pub struct StFile {
    pub path: PathBuf,
    /// Sorted by name.
    pub tensors: Vec<StTensor>,
    pub metadata: BTreeMap<String, String>,
}

impl StFile {
    pub fn get(&self, name: &str) -> Option<&StTensor> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.tensors[i])
    }
}

/// Bytes per element of the dtypes this module meets.
pub fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "BOOL" => 1,
        _ => return None,
    })
}

pub fn read_safetensors(path: &Path) -> Result<StFile> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    let n = u64::from_le_bytes(b);
    ensure!(n < 1 << 30, "{}: header of {n} bytes", path.display());
    let mut h = vec![0u8; n as usize];
    f.read_exact(&mut h)?;
    let v: Value = serde_json::from_slice(&h).with_context(|| format!("{}: header", path.display()))?;
    let obj = v.as_object().context("safetensors header is not an object")?;
    let base = 8 + n;
    let file_len = f.metadata()?.len();
    let mut tensors = Vec::new();
    let mut metadata = BTreeMap::new();
    for (k, t) in obj {
        if k == "__metadata__" {
            if let Some(m) = t.as_object() {
                for (mk, mv) in m {
                    metadata.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                }
            }
            continue;
        }
        let dtype = t["dtype"].as_str().with_context(|| format!("{k}: dtype"))?.to_string();
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .with_context(|| format!("{k}: shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).context("shape entry"))
            .collect::<Result<_>>()?;
        let off = t["data_offsets"].as_array().with_context(|| format!("{k}: data_offsets"))?;
        ensure!(off.len() == 2, "{k}: data_offsets");
        let (s, e) = (off[0].as_u64().context("offset")?, off[1].as_u64().context("offset")?);
        let t = StTensor { name: k.clone(), dtype, shape, start: base + s, end: base + e };
        let per = dtype_bytes(&t.dtype).with_context(|| format!("{k}: dtype {}", t.dtype))?;
        ensure!(
            t.end - t.start == t.numel() * per && t.end <= file_len && t.start <= t.end,
            "{k}: byte range {}..{} does not hold {:?} {}",
            t.start,
            t.end,
            t.shape,
            t.dtype
        );
        tensors.push(t);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(StFile { path: path.to_path_buf(), tensors, metadata })
}

/// Read a tensor's bytes.
pub fn read_tensor_bytes(st: &StFile, t: &StTensor) -> Result<Vec<u8>> {
    let mut f = File::open(&st.path)?;
    f.seek(SeekFrom::Start(t.start))?;
    let mut buf = vec![0u8; (t.end - t.start) as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// One tensor the sidecar receives: its name there, and the native bytes.
pub struct SidecarEntry {
    pub name: String,
    pub src: StTensor,
}

/// Write a safetensors file whose tensors are byte ranges copied from `src`.
/// Returns the file length.
pub fn write_sidecar(
    out: &Path,
    src: &StFile,
    entries: &[SidecarEntry],
    metadata: &BTreeMap<String, String>,
) -> Result<u64> {
    let mut sorted: Vec<&SidecarEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        Value::Object(metadata.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()),
    );
    let mut at = 0u64;
    for e in &sorted {
        let len = e.src.end - e.src.start;
        header.insert(
            e.name.clone(),
            json!({"dtype": e.src.dtype, "shape": e.src.shape, "data_offsets": [at, at + len]}),
        );
        at += len;
    }
    let mut h = serde_json::to_vec(&Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let tmp = out.with_extension("safetensors.tmp");
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
        w.write_all(&(h.len() as u64).to_le_bytes())?;
        w.write_all(&h)?;
        let mut f = File::open(&src.path)?;
        let mut buf = vec![0u8; 1 << 24];
        for e in &sorted {
            f.seek(SeekFrom::Start(e.src.start))?;
            let mut left = e.src.end - e.src.start;
            while left > 0 {
                let k = left.min(buf.len() as u64) as usize;
                f.read_exact(&mut buf[..k])?;
                w.write_all(&buf[..k])?;
                left -= k as u64;
            }
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, out)?;
    Ok(std::fs::metadata(out)?.len())
}

// ---------------------------------------------------------------------------
// Expert records from a layer directory of `bin/extq`.
// ---------------------------------------------------------------------------

/// A finished layer directory: `VERIFIED` present, every expert done.
pub struct LayerDir {
    pub path: PathBuf,
    pub layer: u64,
    pub experts: usize,
}

pub fn open_layer_dir(path: &Path) -> Result<LayerDir> {
    ensure!(
        path.join("VERIFIED").is_file(),
        "{}: no VERIFIED file; a layer is read only once `extq verify` passed on it",
        path.display()
    );
    let enc: Value = serde_json::from_reader(File::open(path.join("encode.json")).context("encode.json")?)?;
    let layer = enc["layer"].as_u64().context("encode.json: layer")?;
    let total = enc["experts_total"].as_u64().context("encode.json: experts_total")? as usize;
    let done = enc["experts_done"].as_u64().context("encode.json: experts_done")? as usize;
    ensure!(done == total, "{}: {done} of {total} experts encoded", path.display());
    let m: Value = serde_json::from_reader(File::open(path.join("manifest.json")).context("manifest.json")?)?;
    if let Some(l) = m.get("layer").and_then(Value::as_u64) {
        ensure!(l == layer, "{}: manifest says layer {l}, encode.json {layer}", path.display());
    }
    for e in 0..total {
        let p = codes_path(path, e as u64);
        ensure!(p.is_file(), "{}: missing", p.display());
    }
    Ok(LayerDir { path: path.to_path_buf(), layer, experts: total })
}

pub fn codes_path(dir: &Path, expert: u64) -> PathBuf {
    dir.join("codes").join(format!("e{expert:03}.llvqrec"))
}

/// The three records of one expert, checked: names, kind, and nothing after.
pub fn read_expert_records(ld: &LayerDir, expert: u64) -> Result<Vec<Record>> {
    let p = codes_path(&ld.path, expert);
    let mut r = BufReader::new(File::open(&p)?);
    let mut out = Vec::with_capacity(3);
    for proj in EXPERT_PROJS {
        let rec = read_record(&mut r, VERSION).with_context(|| format!("{}: {proj}", p.display()))?;
        let want = expert_name(ld.layer, expert, proj);
        ensure!(rec.name() == want, "{}: record {:?}, expected {want:?}", p.display(), rec.name());
        ensure!(rec.kind() == CodeKind::Tetra, "{}: {want} is {}, not Tetra", p.display(), rec.kind());
        out.push(rec);
    }
    let mut rest = Vec::new();
    r.read_to_end(&mut rest)?;
    ensure!(rest.is_empty(), "{}: {} trailing bytes", p.display(), rest.len());
    Ok(out)
}

/// Every record of a record file (a file of back-to-back v5 records, as
/// `bin/extq` and `bin/inkwh` write them).
pub fn read_record_file(path: &Path) -> Result<Vec<Record>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut c = std::io::Cursor::new(&bytes[..]);
    let mut out = Vec::new();
    while (c.position() as usize) < bytes.len() {
        out.push(read_record(&mut c, VERSION).with_context(|| format!("{}", path.display()))?);
    }
    ensure!(!out.is_empty(), "{}: no record", path.display());
    Ok(out)
}

/// A raw-tensor file: one tensor in the sealed stream's tagged framing, as
/// `bin/inktable` writes it.
pub fn write_raw_file(path: &Path, t: &RawTensor) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        write_raw(&mut w, t)?;
        w.flush()?;
    }
    std::fs::rename(tmp, path)?;
    Ok(())
}

pub fn read_raw_file(path: &Path) -> Result<RawTensor> {
    let mut r = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let t = read_raw(&mut r, VERSION)?;
    let mut rest = Vec::new();
    r.read_to_end(&mut rest)?;
    ensure!(rest.is_empty(), "{}: {} trailing bytes", path.display(), rest.len());
    Ok(t)
}

// ---------------------------------------------------------------------------
// The seal.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct SealInputs {
    pub out: PathBuf,
    pub layer_dirs: Vec<PathBuf>,
    pub record_files: Vec<PathBuf>,
    /// `inkwh` output directories: every record their `inkwh.json` lists.
    pub wh_dirs: Vec<PathBuf>,
    pub raw_files: Vec<PathBuf>,
    /// The native checkpoint; without it no complement is computed and no
    /// sidecar is written.
    pub native: Option<PathBuf>,
    pub sidecar: Option<PathBuf>,
    /// `(blob name, path)`.
    pub blobs: Vec<(String, PathBuf)>,
    pub partial: bool,
    pub text_only: bool,
}

#[derive(Default, Debug)]
pub struct SealReport {
    pub records: usize,
    pub expert_records: usize,
    pub int4_records: usize,
    pub payload_bits: u64,
    pub raws: usize,
    pub sidecar_tensors: usize,
    pub sidecar_bytes: u64,
    pub omitted_tensors: Vec<String>,
    pub omitted_elements: u64,
    pub file_bytes: u64,
}

/// Which native expert tensor group a name belongs to: `(layer, "w13"|"w2")`.
fn native_expert_group(native: &str) -> Option<(u64, &'static str)> {
    let rest = native.strip_prefix("model.llm.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let l: u64 = l.parse().ok()?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    if rest.starts_with("w13_weight") {
        Some((l, "w13"))
    } else if rest.starts_with("w2_weight") {
        Some((l, "w2"))
    } else {
        None
    }
}

pub fn seal(inp: &SealInputs) -> Result<SealReport> {
    let mut rep = SealReport::default();
    // ---- pass 1: everything that will be written, checked, before a byte ----
    let mut layers: Vec<LayerDir> = inp.layer_dirs.iter().map(|p| open_layer_dir(p)).collect::<Result<_>>()?;
    layers.sort_by_key(|l| l.layer);
    for w in layers.windows(2) {
        ensure!(w[0].layer != w[1].layer, "layer {} given twice", w[0].layer);
    }
    // Record files: those named, then those of the inkwh directories in
    // (layer, name) order, so the file's order does not depend on how the
    // directories were listed.
    let mut wh_files: Vec<(u64, String, PathBuf)> = Vec::new();
    for d in &inp.wh_dirs {
        for e in wh_entries(d)? {
            let layer = e
                .name
                .strip_prefix("model.layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, _)| l.parse().ok())
                .unwrap_or(u64::MAX);
            wh_files.push((layer, e.name, e.record));
        }
    }
    wh_files.sort();
    let record_files: Vec<PathBuf> =
        inp.record_files.iter().cloned().chain(wh_files.into_iter().map(|(_, _, p)| p)).collect();
    let mut names: BTreeSet<String> = BTreeSet::new();
    let mut kinds = KindSet::empty();
    let mut n_records = 0usize;
    // Experts complete per (layer, group) — what covers a native tensor.
    let mut expert_cover: BTreeMap<(u64, &'static str), usize> = BTreeMap::new();
    for ld in &layers {
        for e in 0..ld.experts as u64 {
            for rec in read_expert_records(ld, e)? {
                ensure!(names.insert(rec.name().to_string()), "record {} given twice", rec.name());
                kinds.insert(rec.kind());
                n_records += 1;
            }
        }
        expert_cover.insert((ld.layer, "w13"), ld.experts);
        expert_cover.insert((ld.layer, "w2"), ld.experts);
        rep.expert_records += 3 * ld.experts;
    }
    for p in &record_files {
        for rec in read_record_file(p)? {
            ensure!(names.insert(rec.name().to_string()), "record {} given twice", rec.name());
            ensure!(
                !rec.name().contains(".mlp.experts."),
                "{}: expert record {} outside a layer directory",
                p.display(),
                rec.name()
            );
            if rec.kind() == CodeKind::Int4G128 {
                rep.int4_records += 1;
            }
            kinds.insert(rec.kind());
            n_records += 1;
        }
    }
    ensure!(n_records > 0, "nothing to seal: no layer directory and no record file");
    let mut raws: Vec<RawTensor> = inp.raw_files.iter().map(|p| read_raw_file(p)).collect::<Result<_>>()?;
    let mut raw_names: BTreeSet<String> = BTreeSet::new();
    for t in &raws {
        ensure!(!names.contains(&t.name), "{} is both a record and a raw tensor", t.name);
        ensure!(raw_names.insert(t.name.clone()), "raw tensor {} given twice", t.name);
    }

    // ---- the complement of the native checkpoint ----
    let mut sidecar: Vec<SidecarEntry> = Vec::new();
    let native = inp.native.as_deref().map(read_safetensors).transpose()?;
    if let Some(st) = &native {
        let mut renamed: BTreeSet<String> = BTreeSet::new();
        for t in &st.tensors {
            let new = rename(&t.name);
            ensure!(renamed.insert(new.clone()), "rename collision on {new}");
            let covered = if let Some((l, g)) = native_expert_group(&t.name) {
                let e = st
                    .get(&format!("model.llm.layers.{l}.mlp.experts.{g}_weight"))
                    .and_then(|w| w.shape.first().copied())
                    .unwrap_or(0);
                match expert_cover.get(&(l, g)) {
                    Some(&have) => {
                        ensure!(have == e, "layer {l}: {have} experts sealed, the checkpoint has {e}");
                        true
                    }
                    None => {
                        // An expert layer whose records are missing goes as a
                        // whole: its F32 scalars are no use without its
                        // NVFP4 bytes, so they are not put in the sidecar.
                        ensure!(
                            inp.partial,
                            "{} has no replacement in the seal: seal layer {l}'s experts, or declare \
                             the seal --partial",
                            t.name
                        );
                        rep.omitted_elements += t.numel();
                        rep.omitted_tensors.push(t.name.clone());
                        continue;
                    }
                }
            } else {
                names.contains(&new) || raw_names.contains(&new)
            };
            if covered {
                continue;
            }
            if inp.text_only && t.name.starts_with("model.visual.") {
                rep.omitted_elements += t.numel();
                rep.omitted_tensors.push(t.name.clone());
                continue;
            }
            match t.dtype.as_str() {
                "BF16" | "F32" => sidecar.push(SidecarEntry { name: new, src: t.clone() }),
                "F16" => {
                    let b = read_tensor_bytes(st, t)?;
                    raws.push(RawTensor {
                        name: new,
                        dims: t.shape.clone(),
                        data: RawData::F16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
                    });
                }
                other => {
                    ensure!(
                        inp.partial,
                        "{} ({other}) has no replacement in the seal and cannot travel as it is; \
                         seal its layer, or declare the seal --partial",
                        t.name
                    );
                    rep.omitted_elements += t.numel();
                    rep.omitted_tensors.push(t.name.clone());
                }
            }
        }
        // A record or raw table that replaces nothing native is a naming
        // error, not an addition.
        for n in names.iter().chain(&raw_names) {
            let native_name = match n.strip_prefix(SEALED_PREFIX) {
                Some(rest) if !rest.starts_with("visual.") => format!("{NATIVE_PREFIX}{rest}"),
                _ => n.clone(),
            };
            let is_expert = n.contains(".mlp.experts.");
            ensure!(is_expert || st.get(&native_name).is_some(), "{n} replaces no tensor of the checkpoint");
        }
    } else {
        ensure!(inp.sidecar.is_none(), "a sidecar needs the native checkpoint (--native)");
    }
    raws.sort_by(|a, b| a.name.cmp(&b.name));
    rep.raws = raws.len();
    if sidecar.is_empty() {
        ensure!(inp.sidecar.is_none() || native.is_none(), "no tensor left for the sidecar");
    } else {
        ensure!(inp.sidecar.is_some(), "{} tensors need the sidecar (--sidecar)", sidecar.len());
    }

    // ---- blobs, including the seal's own description ----
    let mut blobs: Vec<Blob> = Vec::new();
    for (name, p) in &inp.blobs {
        ensure!(name != SEAL_BLOB, "{SEAL_BLOB} is written by the seal");
        blobs.push(Blob { name: name.clone(), bytes: std::fs::read(p).with_context(|| format!("{}", p.display()))? });
    }
    let sidecar_name = inp
        .sidecar
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string());
    let sidecar_params: u64 = sidecar.iter().map(|e| e.src.numel()).sum();
    let desc = json!({
        "tool": "inkseal",
        "naming": "model.llm.* -> model.*, other names unchanged",
        "layers": layers.iter().map(|l| l.layer).collect::<Vec<_>>(),
        "records": n_records,
        "raw_tensors": raws.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        "native": inp.native.as_ref().map(|p| p.display().to_string()),
        "sidecar": sidecar_name,
        "sidecar_tensors": sidecar.len(),
        "sidecar_params": sidecar_params,
        "partial": inp.partial,
        "text_only": inp.text_only,
        "omitted_elements": rep.omitted_elements,
        "omitted_text_elements": native.as_ref().map(|st| {
            rep.omitted_tensors
                .iter()
                .filter(|n| !n.starts_with("model.visual."))
                .filter_map(|n| st.get(n).map(StTensor::numel))
                .sum::<u64>()
        }).unwrap_or(0),
        "omitted_tensors": rep.omitted_tensors,
    });
    blobs.push(Blob { name: SEAL_BLOB.into(), bytes: serde_json::to_vec_pretty(&desc)? });

    // ---- pass 2: write ----
    let tmp = inp.out.with_extension("llvq.tmp");
    {
        let out = BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
        let default = if kinds.contains(CodeKind::Tetra) { CodeKind::Tetra } else { CodeKind::Int4G128 };
        let mut w = ArtifactWriter::with_kinds(out, VERSION, n_records as u32, default, kinds)?;
        for ld in &layers {
            for e in 0..ld.experts as u64 {
                for rec in read_expert_records(ld, e)? {
                    w.push_record(&rec)?;
                }
            }
        }
        for p in &record_files {
            for rec in read_record_file(p)? {
                w.push_record(&rec)?;
            }
        }
        ensure!(w.matrices as usize == n_records, "{} records written, {n_records} counted", w.matrices);
        let (bits, _) = w.seal(&raws, &blobs)?;
        rep.payload_bits = bits;
    }
    let f = File::open(&tmp)?;
    f.sync_all()?;
    std::fs::rename(&tmp, &inp.out)?;
    rep.records = n_records;
    rep.file_bytes = std::fs::metadata(&inp.out)?.len();

    if let (Some(st), Some(sp)) = (&native, &inp.sidecar) {
        if !sidecar.is_empty() {
            let mut md = BTreeMap::new();
            md.insert("format".to_string(), "pt".to_string());
            md.insert("llvq_naming".to_string(), "model.llm.* -> model.*".to_string());
            md.insert(
                "llvq_sealed_file".to_string(),
                inp.out.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            );
            md.insert("llvq_partial".to_string(), inp.partial.to_string());
            md.insert("llvq_text_only".to_string(), inp.text_only.to_string());
            md.insert("llvq_omitted_elements".to_string(), rep.omitted_elements.to_string());
            md.insert("llvq_omitted_tensors".to_string(), rep.omitted_tensors.len().to_string());
            rep.sidecar_bytes = write_sidecar(sp, st, &sidecar, &md)?;
            rep.sidecar_tensors = sidecar.len();
        }
    }
    Ok(rep)
}

// ---------------------------------------------------------------------------
// Verification: every record of the sealed file decoded against its
// reference, every raw table against its file, every sidecar tensor against
// the native bytes, and every native tensor accounted for exactly once.
// ---------------------------------------------------------------------------

/// What `inkseal verify` checks against.
#[derive(Default)]
pub struct VerifyInputs {
    pub sealed: PathBuf,
    /// Expert layer directories: records against `w13_hat` / `w2_hat`.
    pub layer_dirs: Vec<PathBuf>,
    /// `inkwh` output directories: records against `<name>.llvqrec` (bytes)
    /// and `<name>.f32` (the decode).
    pub wh_dirs: Vec<PathBuf>,
    /// Raw-table files: the sealed raw tensor against the file's bytes.
    pub raw_files: Vec<PathBuf>,
    pub sidecar: Option<PathBuf>,
    pub native: Option<PathBuf>,
    pub threads: usize,
}

#[derive(Default, Debug)]
pub struct VerifyReport {
    pub records: usize,
    pub lattice_decoded: usize,
    pub decode_failures: Vec<String>,
    /// Experts compared against float32 hats, bit for bit.
    pub expert_f32_records: usize,
    pub expert_f32_weights: u64,
    pub expert_f32_differing: u64,
    /// Experts compared against float16 hats: the decoded f32 narrowed to
    /// f16 (round to nearest even, as `hats_f16.py`'s `astype(float16)`),
    /// then compared bit for bit with the stored f16.
    pub expert_f16_records: usize,
    pub expert_f16_weights: u64,
    pub expert_f16_differing: u64,
    /// `inkwh` records: record bytes equal to the file, decode equal to the
    /// reconstruction bit for bit.
    pub wh_records: usize,
    pub wh_weights: u64,
    pub wh_differing: u64,
    pub wh_bytes_differ: Vec<String>,
    pub raws: usize,
    pub raws_compared: usize,
    pub raws_differ: Vec<String>,
    pub blobs: usize,
    pub sidecar_tensors: usize,
    pub sidecar_bytes: u64,
    /// Native accounting, when the native file is given.
    pub native_tensors: usize,
    pub by_class: BTreeMap<&'static str, usize>,
    pub accounting_errors: Vec<String>,
    /// Native text-model tensors the seal omitted (a partial seal).
    pub text_omitted: usize,
    /// Per expert layer: which comparison was made.
    pub layer_modes: Vec<(u64, &'static str)>,
}

impl VerifyReport {
    /// Every check that can fail, as one list.
    pub fn failures(&self) -> Vec<String> {
        let mut f = Vec::new();
        if !self.decode_failures.is_empty() {
            f.push(format!("{} records do not decode", self.decode_failures.len()));
        }
        if self.expert_f32_differing > 0 {
            f.push(format!("{} expert weights differ from the f32 hats", self.expert_f32_differing));
        }
        if self.expert_f16_differing > 0 {
            f.push(format!("{} expert weights differ from the f16 hats after narrowing", self.expert_f16_differing));
        }
        if self.wh_differing > 0 {
            f.push(format!("{} attention weights differ from inkwh's reconstructions", self.wh_differing));
        }
        if !self.wh_bytes_differ.is_empty() {
            f.push(format!("records differ from inkwh's files: {:?}", self.wh_bytes_differ));
        }
        if !self.raws_differ.is_empty() {
            f.push(format!("raw tensors differ from their files: {:?}", self.raws_differ));
        }
        if !self.accounting_errors.is_empty() {
            f.push(format!("{} accounting errors, first: {}", self.accounting_errors.len(), self.accounting_errors[0]));
        }
        f
    }
}

enum HatKind {
    F32,
    F16,
}

struct Hats {
    layer: u64,
    experts: usize,
    w13: PathBuf,
    w2: PathBuf,
    kind: HatKind,
    seen: usize,
}

impl Hats {
    fn open(p: &Path) -> Result<Self> {
        let ld = open_layer_dir(p)?;
        let (w13, w2, kind) = if p.join("w13_hat.f32").is_file() {
            (p.join("w13_hat.f32"), p.join("w2_hat.f32"), HatKind::F32)
        } else if p.join("w13_hat.f16").is_file() {
            (p.join("w13_hat.f16"), p.join("w2_hat.f16"), HatKind::F16)
        } else {
            bail!("{}: neither w13_hat.f32 nor w13_hat.f16", p.display());
        };
        ensure!(w2.is_file(), "{}: missing", w2.display());
        Ok(Self { layer: ld.layer, experts: ld.experts, w13, w2, kind, seen: 0 })
    }
    fn width(&self) -> u64 {
        match self.kind {
            HatKind::F32 => 4,
            HatKind::F16 => 2,
        }
    }
}

fn read_at(path: &Path, offset: u64, bytes: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0u8; bytes];
    f.read_exact(&mut b)?;
    Ok(b)
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// `(layer, expert, proj)` of an expert record name.
pub fn parse_expert_name(name: &str) -> Option<(u64, u64, &'static str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e, rest) = rest.split_once('.')?;
    let proj = EXPERT_PROJS.into_iter().find(|p| rest == format!("{p}_proj.weight"))?;
    Some((l.parse().ok()?, e.parse().ok()?, proj))
}

/// Counts the weights of one decoded expert matrix that differ from the
/// hats. The w13 layout is `[expert, gate rows then up rows, d_in]`.
fn compare_expert(d: &[f32], rows: usize, cols: usize, hats: &Hats, expert: u64, proj: &str) -> Result<u64> {
    let n = rows * cols;
    let (path, at) = match proj {
        "gate" => (&hats.w13, expert as usize * 2 * n),
        "up" => (&hats.w13, expert as usize * 2 * n + n),
        _ => (&hats.w2, expert as usize * n),
    };
    let b = read_at(path, at as u64 * hats.width(), n * hats.width() as usize)?;
    Ok(match hats.kind {
        HatKind::F32 => d.iter().zip(f32s(&b)).filter(|(a, w)| a.to_bits() != w.to_bits()).count() as u64,
        HatKind::F16 => d
            .iter()
            .zip(b.chunks_exact(2))
            .filter(|(a, w)| half::f16::from_f32(**a).to_bits() != u16::from_le_bytes([w[0], w[1]]))
            .count() as u64,
    })
}

/// One matrix `inkwh` wrote: its record file and its reconstruction.
pub struct WhEntry {
    pub name: String,
    pub record: PathBuf,
    pub recon: PathBuf,
}

/// The matrices of an `inkwh` output directory, from its `inkwh.json`.
pub fn wh_entries(dir: &Path) -> Result<Vec<WhEntry>> {
    let s: Value = serde_json::from_reader(
        File::open(dir.join("inkwh.json")).with_context(|| format!("{}: inkwh.json", dir.display()))?,
    )?;
    let mut out = Vec::new();
    for m in s["matrices"].as_array().context("inkwh.json: matrices")? {
        let name = m["name"].as_str().context("inkwh.json: name")?.to_string();
        let record = dir.join(format!("{name}.llvqrec"));
        let recon = dir.join(format!("{name}.f32"));
        ensure!(record.is_file() && recon.is_file(), "{}: {name}'s files are missing", dir.display());
        out.push(WhEntry { name, record, recon });
    }
    Ok(out)
}

/// Every `inkwh` output directory under `root` (a directory holding
/// `inkwh.json`), sorted.
pub fn wh_dirs_under(root: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .with_context(|| format!("{}", root.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("inkwh.json").is_file())
        .collect();
    v.sort();
    Ok(v)
}

/// A decoded record waiting for its comparison.
enum Check {
    Expert { hi: usize, expert: u64, proj: &'static str },
    Wh { recon: PathBuf },
    None,
}

type Pending = (Vec<f32>, usize, usize, Check);

/// The seal's own description, read back from the sealed file's blob.
fn seal_blob(blobs: &[Blob]) -> Option<Value> {
    blobs.iter().find(|b| b.name == SEAL_BLOB).and_then(|b| serde_json::from_slice(&b.bytes).ok())
}

pub fn verify_all(inp: &VerifyInputs) -> Result<VerifyReport> {
    let mut hats: Vec<Hats> = inp.layer_dirs.iter().map(|p| Hats::open(p)).collect::<Result<_>>()?;
    let mut wh: BTreeMap<String, (WhEntry, bool)> = BTreeMap::new();
    for d in &inp.wh_dirs {
        for e in wh_entries(d)? {
            let name = e.name.clone();
            ensure!(wh.insert(name.clone(), (e, false)).is_none(), "{name} in two inkwh directories");
        }
    }
    let mut rep = VerifyReport::default();
    for hh in &hats {
        let mode = match hh.kind {
            HatKind::F32 => "f32 hats, bit for bit",
            HatKind::F16 => "f16 hats, decode narrowed to f16 then bit for bit",
        };
        rep.layer_modes.push((hh.layer, mode));
    }
    let mut r = BufReader::with_capacity(1 << 22, File::open(&inp.sealed)?);
    let h = read_header(&mut r)?;
    ensure!(h.version == VERSION, "{}: format v{}", inp.sealed.display(), h.version);
    let cbs = Codebooks::new();
    let threads = inp.threads.max(1);
    let mut names: BTreeSet<String> = BTreeSet::new();
    let mut batch: Vec<Pending> = Vec::new();
    // (differing expert f32, f16, wh) and (weights) accumulated by flush.
    let flush = |batch: &mut Vec<Pending>, hats: &[Hats], rep: &mut VerifyReport| -> Result<()> {
        let chunk = batch.len().div_ceil(threads).max(1);
        let results: Vec<Result<[u64; 9]>> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch
                .chunks(chunk)
                .map(|part| {
                    sc.spawn(move || -> Result<[u64; 9]> {
                        // [f32 recs, f32 w, f32 bad, f16 recs, f16 w, f16 bad, wh recs, wh w, wh bad]
                        let mut a = [0u64; 9];
                        for (d, rows, cols, c) in part {
                            let n = (rows * cols) as u64;
                            match c {
                                Check::Expert { hi, expert, proj } => {
                                    let bad = compare_expert(d, *rows, *cols, &hats[*hi], *expert, proj)?;
                                    let o = if matches!(hats[*hi].kind, HatKind::F32) { 0 } else { 3 };
                                    a[o] += 1;
                                    a[o + 1] += n;
                                    a[o + 2] += bad;
                                }
                                Check::Wh { recon } => {
                                    let b = std::fs::read(recon)?;
                                    ensure!(b.len() as u64 == n * 4, "{}: {} bytes for {n} weights", recon.display(), b.len());
                                    let bad = d.iter().zip(f32s(&b)).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                                    a[6] += 1;
                                    a[7] += n;
                                    a[8] += bad as u64;
                                }
                                Check::None => {}
                            }
                        }
                        Ok(a)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("verify thread")).collect()
        });
        for res in results {
            let a = res?;
            rep.expert_f32_records += a[0] as usize;
            rep.expert_f32_weights += a[1];
            rep.expert_f32_differing += a[2];
            rep.expert_f16_records += a[3] as usize;
            rep.expert_f16_weights += a[4];
            rep.expert_f16_differing += a[5];
            rep.wh_records += a[6] as usize;
            rep.wh_weights += a[7];
            rep.wh_differing += a[8];
        }
        batch.clear();
        Ok(())
    };
    for _ in 0..h.matrices {
        let rec = read_record(&mut r, h.version)?;
        rep.records += 1;
        let name = rec.name().to_string();
        ensure!(names.insert(name.clone()), "{name} appears twice");
        // An inkwh record: the sealed record re-serialized must be the file.
        let wh_check = if let Some((e, seen)) = wh.get_mut(&name) {
            *seen = true;
            let mut b = Vec::new();
            llvq_artifact::write_record(&mut b, h.version, &rec)?;
            if b != std::fs::read(&e.record)? {
                rep.wh_bytes_differ.push(name.clone());
            }
            Some(e.recon.clone())
        } else {
            None
        };
        let (d, rows, cols) = match rec {
            Record::Lattice(raw) => match llvq_artifact::decode_raw(raw, &cbs) {
                Ok(m) => {
                    rep.lattice_decoded += 1;
                    (decode_matrix(&m), m.d_out, m.d_in)
                }
                Err(e) => {
                    rep.decode_failures.push(format!("{name}: {e}"));
                    continue;
                }
            },
            Record::Int4(q) => (q.to_f32(), q.d_out, q.d_in),
        };
        let check = if let Some(recon) = wh_check {
            Check::Wh { recon }
        } else if let Some((hi, expert, proj)) = parse_expert_name(&name)
            .and_then(|(l, e, p)| hats.iter().position(|hh| hh.layer == l).map(|hi| (hi, e, p)))
        {
            let hh = &mut hats[hi];
            ensure!((expert as usize) < hh.experts, "{name}: expert {expert} of {}", hh.experts);
            let per = (rows * cols) as u64 * hh.width() * if proj == "down" { 1 } else { 2 };
            let len = std::fs::metadata(if proj == "down" { &hh.w2 } else { &hh.w13 })?.len();
            ensure!(len == per * hh.experts as u64, "{name}: {len}-byte hats for {} experts of {per}", hh.experts);
            hh.seen += 1;
            Check::Expert { hi, expert, proj }
        } else {
            Check::None
        };
        if !matches!(check, Check::None) {
            batch.push((d, rows, cols, check));
        }
        if batch.len() >= 4 * threads {
            flush(&mut batch, &hats, &mut rep)?;
        }
    }
    flush(&mut batch, &hats, &mut rep)?;
    for hh in &hats {
        ensure!(
            hh.seen == 3 * hh.experts,
            "layer {}: {} expert records in the sealed file, {} expected",
            hh.layer,
            hh.seen,
            3 * hh.experts
        );
    }
    let missing: Vec<&String> = wh.iter().filter(|(_, (_, s))| !s).map(|(n, _)| n).collect();
    ensure!(missing.is_empty(), "inkwh records absent from the sealed file: {missing:?}");

    // ---- raw tensors, blobs, end of file ----
    let raw_ref: BTreeMap<String, Vec<u8>> = inp
        .raw_files
        .iter()
        .map(|p| -> Result<(String, Vec<u8>)> { Ok((read_raw_file(p)?.name, std::fs::read(p)?)) })
        .collect::<Result<_>>()?;
    let mut raw_names: BTreeSet<String> = BTreeSet::new();
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    for _ in 0..u32::from_le_bytes(b4) {
        let t = read_raw(&mut r, h.version)?;
        rep.raws += 1;
        ensure!(raw_names.insert(t.name.clone()), "raw tensor {} appears twice", t.name);
        ensure!(!names.contains(&t.name), "{} is both a record and a raw tensor", t.name);
        if let Some(want) = raw_ref.get(&t.name) {
            let mut b = Vec::new();
            write_raw(&mut b, &t)?;
            rep.raws_compared += 1;
            if &b != want {
                rep.raws_differ.push(t.name.clone());
            }
        }
    }
    for n in raw_ref.keys() {
        ensure!(raw_names.contains(n), "raw tensor {n} absent from the sealed file");
    }
    let mut blobs = Vec::new();
    r.read_exact(&mut b4)?;
    for _ in 0..u32::from_le_bytes(b4) {
        blobs.push(llvq_artifact::read_blob(&mut r)?);
        rep.blobs += 1;
    }
    let mut rest = Vec::new();
    r.read_to_end(&mut rest)?;
    ensure!(rest.is_empty(), "{}: {} trailing bytes", inp.sealed.display(), rest.len());

    // ---- sidecar against native, and the native accounting ----
    if let (Some(sc), Some(nat)) = (&inp.sidecar, &inp.native) {
        let (k, b) = verify_sidecar(sc, nat)?;
        rep.sidecar_tensors = k;
        rep.sidecar_bytes = b;
    }
    if let Some(nat) = &inp.native {
        let st = read_safetensors(nat)?;
        let side: BTreeSet<String> = match &inp.sidecar {
            Some(p) => read_safetensors(p)?.tensors.into_iter().map(|t| t.name).collect(),
            None => BTreeSet::new(),
        };
        let omitted: BTreeSet<String> = seal_blob(&blobs)
            .and_then(|v| v["omitted_tensors"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        // Expert groups the records cover: (layer, w13|w2) -> experts seen.
        let mut groups: BTreeMap<(u64, &'static str), BTreeSet<u64>> = BTreeMap::new();
        for n in &names {
            if let Some((l, e, p)) = parse_expert_name(n) {
                let g = if p == "down" { "w2" } else { "w13" };
                groups.entry((l, g)).or_default().insert(e);
            }
        }
        let mut used_records: BTreeSet<&String> = BTreeSet::new();
        let mut used_raws: BTreeSet<&String> = BTreeSet::new();
        let mut used_side: BTreeSet<&String> = BTreeSet::new();
        let mut used_omit: BTreeSet<&String> = BTreeSet::new();
        for t in &st.tensors {
            rep.native_tensors += 1;
            let new = rename(&t.name);
            let mut hits: Vec<&'static str> = Vec::new();
            if let Some((l, g)) = native_expert_group(&t.name) {
                let e = st
                    .get(&format!("model.llm.layers.{l}.mlp.experts.{g}_weight"))
                    .and_then(|w| w.shape.first().copied())
                    .unwrap_or(0) as u64;
                if let Some(seen) = groups.get(&(l, g)) {
                    if seen.len() as u64 == e && seen.iter().all(|x| *x < e) {
                        hits.push("expert records");
                    } else {
                        rep.accounting_errors.push(format!("{}: {} of {e} experts sealed", t.name, seen.len()));
                    }
                }
            }
            if let Some(n) = names.get(&new) {
                hits.push("named record");
                used_records.insert(n);
            }
            if let Some(n) = raw_names.get(&new) {
                hits.push("raw tensor");
                used_raws.insert(n);
            }
            if let Some(n) = side.get(&new) {
                hits.push("sidecar");
                used_side.insert(n);
            }
            if let Some(n) = omitted.get(&t.name) {
                hits.push("omitted");
                used_omit.insert(n);
                if !t.name.starts_with("model.visual.") {
                    rep.text_omitted += 1;
                }
            }
            match hits.as_slice() {
                [one] => *rep.by_class.entry(one).or_default() += 1,
                [] => rep.accounting_errors.push(format!("{}: in no place", t.name)),
                many => rep.accounting_errors.push(format!("{}: in {many:?}", t.name)),
            }
        }
        for n in names.iter().filter(|n| parse_expert_name(n).is_none() && !used_records.contains(n)) {
            rep.accounting_errors.push(format!("record {n} replaces no native tensor"));
        }
        for (l, g) in groups.keys() {
            let nat_name = format!("model.llm.layers.{l}.mlp.experts.{g}_weight");
            if st.get(&nat_name).is_none() {
                rep.accounting_errors.push(format!("expert records of layer {l} {g}: no native {nat_name}"));
            }
        }
        for n in raw_names.iter().filter(|n| !used_raws.contains(n)) {
            rep.accounting_errors.push(format!("raw tensor {n} replaces no native tensor"));
        }
        for n in side.iter().filter(|n| !used_side.contains(n)) {
            rep.accounting_errors.push(format!("sidecar tensor {n} has no native tensor"));
        }
        for n in omitted.iter().filter(|n| !used_omit.contains(n)) {
            rep.accounting_errors.push(format!("omitted tensor {n} is not in the native file"));
        }
    }
    Ok(rep)
}

/// The lot-1 entry, kept for the tests: experts only.
pub fn verify(sealed: &Path, layer_dirs: &[PathBuf], threads: usize) -> Result<VerifyReport> {
    let rep = verify_all(&VerifyInputs {
        sealed: sealed.to_path_buf(),
        layer_dirs: layer_dirs.to_vec(),
        threads,
        ..Default::default()
    })?;
    Ok(rep)
}

impl VerifyReport {
    /// Expert records compared, both hat widths.
    pub fn compared_records(&self) -> usize {
        self.expert_f32_records + self.expert_f16_records
    }
    pub fn compared_weights(&self) -> u64 {
        self.expert_f32_weights + self.expert_f16_weights
    }
    pub fn differing_weights(&self) -> u64 {
        self.expert_f32_differing + self.expert_f16_differing + self.wh_differing
    }
}

/// Every sidecar tensor against the native bytes of the name it came from.
/// Returns `(tensors, bytes)` compared.
pub fn verify_sidecar(sidecar: &Path, native: &Path) -> Result<(usize, u64)> {
    let sc = read_safetensors(sidecar)?;
    let st = read_safetensors(native)?;
    let by_new: BTreeMap<String, &StTensor> = st.tensors.iter().map(|t| (rename(&t.name), t)).collect();
    let (mut n, mut bytes) = (0usize, 0u64);
    for t in &sc.tensors {
        let src = by_new.get(&t.name).with_context(|| format!("{}: no native tensor", t.name))?;
        ensure!(src.dtype == t.dtype && src.shape == t.shape, "{}: dtype or shape moved", t.name);
        let (a, b) = (read_tensor_bytes(&sc, t)?, read_tensor_bytes(&st, src)?);
        ensure!(a == b, "{}: bytes differ from {}", t.name, src.name);
        n += 1;
        bytes += a.len() as u64;
    }
    Ok((n, bytes))
}

/// A reader that counts the bytes it hands out.
struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

/// Copy `src` to `dst` with one bit flipped in the payload of one item:
/// record `index` when `raw` is false (a lattice record's code stream, or an
/// int4 record's packed nibbles), raw tensor `index` when `raw` is true (its
/// f16 values, or its packed integers). `bit` counts from the start of that
/// payload. Returns the byte offset flipped.
///
/// Where the payload sits is read off the framing: a lattice record ends
/// with its code stream; an int4 record ends with its packed bytes, a `u64`
/// group count, then the scales and the biases (2 bytes a group each); a
/// quantized raw tensor ends the same way, and an f16 one with its values.
pub fn flip_bit(src: &Path, dst: &Path, raw: bool, index: usize, bit: u64) -> Result<u64> {
    let mut r = Counting { inner: BufReader::with_capacity(1 << 22, File::open(src)?), pos: 0 };
    let h = read_header(&mut r)?;
    let (mut end, mut len, mut tail) = (0u64, 0u64, 0u64);
    if !raw {
        ensure!(index < h.matrices as usize, "record {index} of {}", h.matrices);
        for i in 0..=index {
            let rec = read_record(&mut r, h.version)?;
            if i == index {
                end = r.pos;
                match rec {
                    Record::Lattice(m) => {
                        let (ib, gb) = llvq_artifact::code_widths(m.kind, &m.name, m.shell_cap, m.centroids.len())?;
                        len = (m.indices.len() as u64 * (ib + gb) as u64).div_ceil(8);
                    }
                    Record::Int4(q) => {
                        len = q.packed.len() as u64;
                        tail = 8 + 4 * q.scales.len() as u64;
                    }
                }
            }
        }
    } else {
        for _ in 0..h.matrices {
            read_record(&mut r, h.version)?;
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let n = u32::from_le_bytes(b4) as usize;
        ensure!(index < n, "raw tensor {index} of {n}");
        for i in 0..=index {
            let t = read_raw(&mut r, h.version)?;
            if i == index {
                end = r.pos;
                match &t.data {
                    RawData::F16(d) => len = 2 * d.len() as u64,
                    RawData::Quant(q) => {
                        len = q.packed.len() as u64;
                        tail = 8 + 4 * q.scales.len() as u64;
                    }
                }
            }
        }
    }
    ensure!(bit / 8 < len, "bit {bit} outside a payload of {len} bytes");
    let offset = end - tail - len + bit / 8;
    drop(r);
    std::fs::copy(src, dst)?;
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(dst)?;
    let mut b = [0u8; 1];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(&mut b)?;
    b[0] ^= 1 << (7 - (bit % 8));
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(&b)?;
    Ok(offset)
}

/// [`flip_bit`] on a record (lot 1's entry).
pub fn flip_code_bit(src: &Path, dst: &Path, index: usize, bit: u64) -> Result<u64> {
    flip_bit(src, dst, false, index, bit)
}

// ---------------------------------------------------------------------------
// Tables (item 1c).
// ---------------------------------------------------------------------------

/// A table read from a safetensors file and narrowed to f16 exactly as
/// `bin/seal` narrows a carried tensor (candle's `to_dtype(F16)`:
/// `f16::from_f32(v.to_f32())` from BF16, `f16::from_f32` from F32).
pub fn table_f16(st: &StFile, native_name: &str, out_name: &str) -> Result<RawTensor> {
    let t = st.get(native_name).with_context(|| format!("{}: no tensor {native_name}", st.path.display()))?;
    let b = read_tensor_bytes(st, t)?;
    let data: Vec<u16> = match t.dtype.as_str() {
        "BF16" => b
            .chunks_exact(2)
            .map(|c| half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits())
            .collect(),
        "F16" => b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        "F32" => b
            .chunks_exact(4)
            .map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits())
            .collect(),
        other => bail!("{native_name}: dtype {other}, expected BF16, F16 or F32"),
    };
    Ok(RawTensor { name: out_name.to_string(), dims: t.shape.clone(), data: RawData::F16(data) })
}

/// The group the tables are stored at, `LLVQ_EMBED=q4`'s.
pub const TABLE_GROUP: usize = 64;

/// [`table_f16`], then [`crate::embedquant::quantize_affine`] at int4 g64.
pub fn table_q4(st: &StFile, native_name: &str, out_name: &str) -> Result<RawTensor> {
    let t = table_f16(st, native_name, out_name)?;
    crate::embedquant::quantize_affine(&t, 4, TABLE_GROUP)
}

// ---------------------------------------------------------------------------
// The generic (W, H) driver (item 1b).
// ---------------------------------------------------------------------------

/// How one matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhKind {
    /// The sealed Tetra recipe ([`Settings::sealed_recipe`]).
    Tetra,
    /// An int4 g128 record ([`crate::sealed::int4_record`]); no Hessian.
    Int4,
}

impl WhKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "tetra" => Self::Tetra,
            "int4" => Self::Int4,
            other => bail!("kind {other:?}: expected tetra or int4"),
        })
    }
}

/// One matrix of a group: all of them read the same input, so they share
/// one Hessian, one rotation and one factor, as the projections of one
/// activation do in the Qwen3 block loop.
pub struct WhMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// Row-major `d_out × d_in`, natural basis.
    pub w: Vec<f32>,
    pub kind: WhKind,
}

/// What one matrix becomes: its record bytes (one v5 record) and its dense
/// reconstruction, f32 row-major in the natural basis.
pub struct WhOut {
    pub name: String,
    pub kind: WhKind,
    pub record: Vec<u8>,
    pub recon: Vec<f32>,
    pub payload_bits: u64,
}

/// The rotation seed of an attention input stream. Shares the shape of
/// `bin/extq`'s seed (`ROTATION_SEED ^ layer << 32 ^ expert << 20 ^ group << 16`)
/// with expert 0 and groups 2 to 5, which no expert seed uses (an expert seed
/// has group 0 or 1), so no two matrices of the model share a seed by accident.
pub fn stream_seed(layer: u64, stream: &str) -> Result<u64> {
    let group: u64 = match stream {
        "q" => 2,
        "k" => 3,
        "v" => 4,
        "o" => 5,
        other => bail!("stream {other:?}: expected q, k, v or o"),
    };
    Ok(crate::calib::ROTATION_SEED ^ (layer << 32) ^ (group << 16))
}

/// The flags a group can raise; the recipe would abort on the second.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WhFlags {
    pub identity_h: bool,
    pub factor_failed: bool,
}

/// Encode one group. `h_sum` is the unnormalised `Σ w x xᵀ`, `n × n`
/// row-major, natural basis, with `weight_sum` its weight total; it is
/// divided by `weight_sum` (a positive scale) before the chain of
/// [`crate::extq`], as `bin/extq` does. With zero tokens or a non-positive
/// weight sum the group is encoded against `H = I` and flagged. Widths that
/// are not a multiple of 24 keep their last `d_in % 24` columns as the
/// format's f32 tail (`TailPolicy::KeepExact`), exactly as the block loop
/// does; an int4 record needs `d_in` a multiple of 128 and refuses otherwise.
pub fn encode_wh_group(
    h_sum: Option<Vec<f64>>,
    n: usize,
    weight_sum: f64,
    tokens: u64,
    seed: Option<u64>,
    mats: &[WhMatrix],
    s: &Settings,
) -> Result<(Vec<WhOut>, WhFlags)> {
    let mut flags = WhFlags::default();
    for m in mats {
        ensure!(m.d_in == n, "{}: width {} in a group of width {n}", m.name, m.d_in);
        ensure!(m.w.len() == m.d_out * m.d_in, "{}: {} values for {} × {}", m.name, m.w.len(), m.d_out, m.d_in);
        ensure!(m.w.iter().all(|v| v.is_finite()), "{}: W holds a non-finite value", m.name);
    }
    let needs_h = mats.iter().any(|m| m.kind == WhKind::Tetra);
    let factor = if needs_h {
        let seed = seed.context("a Tetra matrix needs a rotation seed")?;
        let identity = || {
            let mut h = vec![0.0f64; n * n];
            for i in 0..n {
                h[i * n + i] = 1.0;
            }
            h
        };
        let h = match h_sum {
            Some(h) if tokens > 0 && weight_sum > 0.0 => {
                ensure!(h.len() == n * n, "H holds {} values, expected {n} × {n}", h.len());
                ensure!(h.iter().all(|v| v.is_finite()), "H holds a non-finite value");
                let inv = 1.0 / weight_sum;
                h.into_iter().map(|v| v * inv).collect()
            }
            _ => {
                flags.identity_h = true;
                identity()
            }
        };
        Some(match factor_hessian(h, n, Some(seed), s.damping) {
            Ok(f) => f,
            Err(_) => {
                flags.factor_failed = true;
                factor_hessian(identity(), n, Some(seed), s.damping).map_err(|e| anyhow::anyhow!("{e}"))?
            }
        })
    } else {
        None
    };
    let enc = llvq_quant::quantizer::TetraShapeGain::encoder();
    let tb = llvq_artifact::Codebook::new(CodeKind::Tetra)?;
    let cbs = Codebooks::new();
    let mut out = Vec::with_capacity(mats.len());
    for m in mats {
        match m.kind {
            WhKind::Tetra => {
                let f = factor.as_ref().expect("factor built for a Tetra group");
                let e = encode_matrix(&m.w, m.d_out, m.d_in, f, s, &enc)?;
                let qm = QuantizedMatrix {
                    name: m.name.clone(),
                    d_out: m.d_out,
                    d_in: m.d_in,
                    codes: e.codes,
                    row_scales: e.row_scales,
                    centroids: e.centroids,
                    rotation_seed: f.rotation.as_ref().map(|_| seed.unwrap()),
                    shell_cap: llvq_artifact::TETRA_SHELL_CAP,
                    tail: e.tail,
                };
                let mut record = Vec::new();
                let bits = write_matrix_with(&mut record, VERSION, &tb, &qm)?;
                // Read back and decode: a record that exists is one that
                // decodes to the reconstruction, as in `bin/extq`.
                let back = llvq_artifact::read_matrix_with(&mut &record[..], VERSION, &cbs)?;
                let d = decode_matrix(&back);
                let bad = d.iter().zip(&e.recon).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                ensure!(bad == 0, "{}: {bad} weights decode differently from the reconstruction", m.name);
                out.push(WhOut { name: m.name.clone(), kind: m.kind, record, recon: e.recon, payload_bits: bits });
            }
            WhKind::Int4 => {
                let t = candle_core::Tensor::from_slice(&m.w, (m.d_out, m.d_in), &candle_core::Device::Cpu)?;
                let rec = crate::sealed::int4_record(&m.name, &t)?;
                let mut record = Vec::new();
                let bits = llvq_artifact::write_matrix_int4(&mut record, VERSION, &rec)?;
                let recon = rec.to_f32();
                out.push(WhOut { name: m.name.clone(), kind: m.kind, record, recon, payload_bits: bits });
            }
        }
    }
    Ok((out, flags))
}

fn read_le<const B: usize>(path: &Path, n: usize) -> Result<Vec<[u8; B]>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(b.len() == n * B, "{}: {} bytes, expected {}", path.display(), b.len(), n * B);
    Ok(b.chunks_exact(B).map(|c| c.try_into().unwrap()).collect())
}

/// Run the driver on a manifest:
///
/// ```json
/// {"n": 2304, "h": "h.f64", "weight_sum": 1234.5, "tokens": 4096,
///  "rotation_seed": 123,            // or "layer": 6, "stream": "q"
///  "matrices": [{"name": "model.layers.6.attn.wq_du.weight", "w": "wq.f32",
///                "d_out": 3072, "kind": "tetra"}]}
/// ```
///
/// Paths are relative to the manifest's directory. `h` may be absent when
/// every matrix is int4. Writes `<out>/<name>.llvqrec` (one record) and
/// `<out>/<name>.f32` (the reconstruction) per matrix, and `<out>/inkwh.json`.
pub fn run_wh(manifest: &Path, out_dir: &Path, threads: usize) -> Result<Value> {
    let m: Value = serde_json::from_reader(File::open(manifest).with_context(|| format!("{}", manifest.display()))?)?;
    let base = manifest.parent().unwrap_or(Path::new("."));
    let n = m["n"].as_u64().context("manifest: n")? as usize;
    let explicit = m.get("rotation_seed").and_then(Value::as_u64);
    let derived = match (m.get("layer").and_then(Value::as_u64), m.get("stream").and_then(Value::as_str)) {
        (Some(l), Some(s)) => Some(stream_seed(l, s)?),
        (None, None) => None,
        _ => bail!("manifest: `layer` and `stream` go together"),
    };
    let seed = match (explicit, derived) {
        (Some(a), Some(b)) => {
            ensure!(a == b, "manifest: rotation_seed {a} contradicts layer/stream ({b})");
            Some(a)
        }
        (a, b) => a.or(b),
    };
    let mut mats = Vec::new();
    for x in m["matrices"].as_array().context("manifest: matrices")? {
        let name = x["name"].as_str().context("matrix: name")?.to_string();
        let d_out = x["d_out"].as_u64().context("matrix: d_out")? as usize;
        let kind = WhKind::parse(x["kind"].as_str().context("matrix: kind")?)?;
        let w: Vec<f32> = read_le::<4>(&base.join(x["w"].as_str().context("matrix: w")?), d_out * n)?
            .into_iter()
            .map(f32::from_le_bytes)
            .collect();
        mats.push(WhMatrix { name, d_out, d_in: n, w, kind });
    }
    ensure!(!mats.is_empty(), "manifest: no matrix");
    let h: Option<Vec<f64>> = match m.get("h").and_then(Value::as_str) {
        Some(p) => Some(read_le::<8>(&base.join(p), n * n)?.into_iter().map(f64::from_le_bytes).collect()),
        None => None,
    };
    ensure!(h.is_some() || mats.iter().all(|x| x.kind == WhKind::Int4), "manifest: a Tetra matrix needs `h`");
    let weight_sum = m.get("weight_sum").and_then(Value::as_f64).unwrap_or(0.0);
    let tokens = m.get("tokens").and_then(Value::as_u64).unwrap_or(0);
    let s = Settings::sealed_recipe(threads);
    // The Hessian the proxy is read against: `H / weight_sum`, the matrix the
    // encoder sees (any positive scale cancels in the proxy's ratio).
    let h_proxy: Option<Vec<f64>> = match &h {
        Some(h) if tokens > 0 && weight_sum > 0.0 => Some(h.iter().map(|v| v / weight_sum).collect()),
        _ => None,
    };
    let t0 = std::time::Instant::now();
    let (outs, flags) = encode_wh_group(h, n, weight_sum, tokens, seed, &mats, &s)?;
    let encode_seconds = t0.elapsed().as_secs_f64();
    std::fs::create_dir_all(out_dir)?;
    let mut lines = Vec::new();
    for (o, x) in outs.iter().zip(&mats) {
        std::fs::write(out_dir.join(format!("{}.llvqrec", o.name)), &o.record)?;
        let mut b = Vec::with_capacity(o.recon.len() * 4);
        for v in &o.recon {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(out_dir.join(format!("{}.f32", o.name)), b)?;
        let (mut dw2, mut w2) = (0.0f64, 0.0f64);
        for (a, b) in x.w.iter().zip(&o.recon) {
            dw2 += ((*b - *a) as f64).powi(2);
            w2 += (*a as f64).powi(2);
        }
        lines.push(json!({
            "name": o.name, "kind": format!("{:?}", o.kind), "d_out": x.d_out, "d_in": n,
            "payload_bits": o.payload_bits,
            "archive_bpw": o.payload_bits as f64 / (x.d_out * n) as f64,
            "tail_weights": if o.kind == WhKind::Tetra { x.d_out * (n % 24) } else { 0 },
            "weight_rel_mse": dw2 / w2.max(f64::MIN_POSITIVE),
            // tr(ΔW H ΔWᵀ) / tr(W H Wᵀ): the Hessian-weighted relative error.
            "proxy_rel": h_proxy.as_ref().map(|hp| {
                let (num, den) = crate::extq::proxy_traces(&x.w, &o.recon, x.d_out, n, hp, threads);
                num / den
            }),
        }));
    }
    let summary = json!({
        "tool": "inkwh", "manifest": manifest.display().to_string(), "n": n,
        "rotation_seed": seed, "weight_sum": weight_sum, "tokens": tokens,
        "identity_h": flags.identity_h, "factor_failed": flags.factor_failed,
        "encode_seconds": encode_seconds, "threads": threads,
        "recipe": "Settings::sealed_recipe (tetra1, nogs, retract, KeepExact, damping 1e-2) / int4_record g128",
        "matrices": lines,
    });
    std::fs::write(out_dir.join("inkwh.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(summary)
}

/// The threads a tool of this module uses: `LLVQ_THREADS`, else 2.
pub fn threads() -> Result<usize> {
    match std::env::var("LLVQ_THREADS") {
        Ok(s) => s.parse::<usize>().ok().filter(|t| *t > 0).with_context(|| format!("LLVQ_THREADS={s:?}")),
        Err(_) => Ok(2),
    }
}

/// The payload bits of one record as stored, the accounting of `bin/extq`'s
/// `archive_bpw`: words, f64 row scales, f64 centroids, f32 tail; for int4,
/// nibbles plus an f16 scale and bias per group.
pub fn record_payload_bits(rec: &RawMatrix) -> Result<u64> {
    let (ib, gb) = llvq_artifact::code_widths(rec.kind, &rec.name, rec.shell_cap, rec.centroids.len())?;
    Ok(rec.indices.len() as u64 * (ib + gb) as u64
        + rec.row_scales.len() as u64 * 64
        + rec.centroids.len() as u64 * 64
        + rec.tail.len() as u64 * 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_strips_only_the_text_prefix() {
        assert_eq!(rename("model.llm.layers.6.attn.wq_du.weight"), "model.layers.6.attn.wq_du.weight");
        assert_eq!(rename("model.llm.embed.weight"), "model.embed.weight");
        assert_eq!(rename("model.visual.proj_in.weight"), "model.visual.proj_in.weight");
        assert_eq!(rename("model.layers.0.x"), "model.layers.0.x");
    }

    #[test]
    fn expert_names_parse_and_split() {
        let n = expert_name(6, 17, "up");
        assert_eq!(parse_expert_name(&n), Some((6, 17, "up")));
        let (b, p) = llvq_artifact::split_name(&n).unwrap();
        assert_eq!((b, p.as_str()), (6, "mlp.experts.17.up_proj"));
        assert_eq!(parse_expert_name("model.layers.6.attn.wq_du.weight"), None);
        assert_eq!(native_expert_group("model.llm.layers.12.mlp.experts.w13_weight_scale"), Some((12, "w13")));
        assert_eq!(native_expert_group("model.llm.layers.12.mlp.experts.w2_weight"), Some((12, "w2")));
        assert_eq!(native_expert_group("model.llm.layers.12.mlp.gate.weight"), None);
    }

    #[test]
    fn stream_seeds_are_distinct_from_expert_seeds() {
        let mut all = BTreeSet::new();
        for l in 0..24u64 {
            for s in ["q", "k", "v", "o"] {
                assert!(all.insert(stream_seed(l, s).unwrap()));
            }
            for e in 0..128u64 {
                for g in 0..2u64 {
                    let x = crate::calib::ROTATION_SEED ^ (l << 32) ^ (e << 20) ^ (g << 16);
                    assert!(all.insert(x), "layer {l} expert {e} group {g} collides");
                }
            }
        }
    }
}
