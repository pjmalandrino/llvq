//! `extq`: the sealed Tetra recipe applied to the experts of one external
//! MoE layer whose Hessians were captured outside this repository.
//!
//! ```text
//! extq <layer_dir> [--experts a..b] [--layer N]   encode (resumable)
//! extq fake <dir> [n_experts] [--layer N]          write a synthetic layer dir
//! extq verify <layer_dir>                          decode every codes file against the hats
//! ```
//!
//! Build with `--features fast-linalg` (the published factorization; the
//! binary refuses to run without it). Threads from `LLVQ_THREADS`.
//!
//! ## Inputs (raw little-endian, row-major, `[expert, out, in]`)
//!
//! `w13.f32` `[E, 2304, 2304]` (rows 0..1152 gate, 1152..2304 up),
//! `w2.f32` `[E, 2304, 1152]`, `h13.f64` `[E, 2304, 2304]`, `h2.f64`
//! `[E, 1152, 1152]` (unnormalised `Σ w² x xᵀ`), and `manifest.json`. The
//! manifest is read leniently: `layer` (optional, else `--layer`), and per
//! expert a token count and a weight sum, either as an array `experts` of
//! objects with `tokens` / `weight_sum` (aliases `token_count`, `n_tokens`;
//! `sum_weights`, `weights_sum`, `sum_w`), or as two top-level arrays of the
//! same names. `E` is read from `num_experts`/`n_experts` if present, else
//! from the size of `w13.f32`; the four files must agree with it.
//!
//! ## What it does per expert
//!
//! `H ← H / weight_sum` (a positive scale; the recipe's `H = AᵀA/N` is also
//! a mean), then the chain of [`llvq_llm::extq`]: gate and up are two
//! matrices sharing one Hessian, one rotation and one factor, as `gate_proj`
//! and `up_proj` share `Act::Mlp` in the Qwen3 loop; down is one matrix on
//! its own. An expert with zero tokens (or a zero weight sum) is encoded
//! against `H = I` and flagged `identity_h`. A Hessian whose damped
//! factorization still fails is also encoded against `I`, flagged
//! `factor_failed`; the recipe would abort there instead.
//!
//! ## Rotation seed
//!
//! `ROTATION_SEED ^ (layer << 32) ^ (expert << 20) ^ (group << 16)`, group 0
//! for the gate/up input, 1 for the down input: distinct for every (layer,
//! expert, group) with expert < 4096, and the same shape as
//! `calib::effective_rotation_seed`, which uses `act << 16`.
//!
//! ## Outputs, in the layer dir
//!
//! `w13_hat.f32`, `w2_hat.f32` (same shapes, the dense reconstructions),
//! `codes/eNNN.llvqrec` (per expert, three records gate, up, down back to back
//! in the repository's own record format, `llvq_artifact::write_matrix_with`
//! at format version `llvq_artifact::VERSION` = 5, Tetra kind: 48-bit words,
//! f64 row scales, two f64 centroids, f32 tail, the rotation seed, the name
//! `model.layers.{L}.mlp.experts.{e}.{gate|up|down}_proj.weight` and the
//! shape; read back with `read_matrix_with(r, 5, &Codebooks::new())` and
//! `decode_matrix`), `extq-records.jsonl` (one line per finished expert, appended after its
//! reconstructions are synced: the resume marker), and `encode.json`
//! (rewritten after every expert from the records file).

use anyhow::{bail, ensure, Context, Result};
use llvq_llm::calib::ROTATION_SEED;
use llvq_llm::extq::{encode_matrix, factor_hessian, proxy_traces, Settings};
use llvq_artifact::{decode_matrix, read_matrix_with, write_matrix_with, CodeKind, Codebooks, QuantizedMatrix};
use llvq_quant::quantizer::TetraShapeGain;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const D: usize = 2304;
const F: usize = 1152;
const BLOCK: usize = 24;
const WORD_BITS: u64 = 48;

fn seed(layer: u64, expert: u64, group: u64) -> u64 {
    ROTATION_SEED ^ (layer << 32) ^ (expert << 20) ^ (group << 16)
}

/// `LLVQ_THREADS`, else ncpu-4: this tool runs on machines someone works on.
fn threads() -> Result<usize> {
    llvq_llm::eval::env_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).saturating_sub(4).max(1))
}

fn read_at(f: &mut File, offset: u64, bytes: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bytes];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_f32(f: &mut File, offset: u64, n: usize) -> Result<Vec<f32>> {
    Ok(read_at(f, offset, n * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f64(f: &mut File, offset: u64, n: usize) -> Result<Vec<f64>> {
    Ok(read_at(f, offset, n * 8)?
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn write_f32_at(f: &mut File, offset: u64, v: &[f32]) -> Result<()> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(&buf)?;
    Ok(())
}

fn num(v: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| v.get(*k).and_then(Value::as_f64))
}

const TOK: &[&str] = &["tokens", "token_count", "n_tokens"];
const WSUM: &[&str] = &["weight_sum", "sum_weights", "weights_sum", "sum_w"];

/// `(layer, experts, per-expert (tokens, weight_sum))`.
/// Per expert: `(tokens, weight_sum)`.
type Stats = Vec<(u64, f64)>;

fn manifest(dir: &Path, layer_arg: Option<u64>) -> Result<(u64, usize, Stats)> {
    let m: Value = serde_json::from_reader(File::open(dir.join("manifest.json")).context("manifest.json")?)?;
    let layer = match (m.get("layer").and_then(Value::as_u64), layer_arg) {
        (Some(a), Some(b)) if a != b => bail!("manifest says layer {a}, --layer says {b}"),
        (Some(a), _) | (None, Some(a)) => a,
        (None, None) => bail!("no `layer` in manifest.json and no --layer"),
    };
    let w13_len = std::fs::metadata(dir.join("w13.f32"))?.len();
    ensure!(w13_len % (D * D * 4) as u64 == 0, "w13.f32 is {w13_len} bytes, not a multiple of one expert");
    let e_files = (w13_len / (D * D * 4) as u64) as usize;
    let e = match num(&m, &["num_experts", "n_experts"]) {
        Some(x) => x as usize,
        None => e_files,
    };
    ensure!(e == e_files, "manifest says {e} experts, w13.f32 holds {e_files}");
    for (name, per) in [("w2.f32", D * F * 4), ("h13.f64", D * D * 8), ("h2.f64", F * F * 8)] {
        let len = std::fs::metadata(dir.join(name)).with_context(|| name.to_string())?.len();
        ensure!(len == (e * per) as u64, "{name} is {len} bytes, expected {} for {e} experts", e * per);
    }
    let stats: Vec<(u64, f64)> = if let Some(arr) = m.get("experts").and_then(Value::as_array) {
        arr.iter()
            .enumerate()
            .map(|(i, x)| {
                Ok((
                    num(x, TOK).with_context(|| format!("expert {i}: no token count"))? as u64,
                    num(x, WSUM).with_context(|| format!("expert {i}: no weight sum"))?,
                ))
            })
            .collect::<Result<_>>()?
    } else {
        let toks = TOK.iter().find_map(|k| m.get(*k).and_then(Value::as_array)).context("no per-expert token counts")?;
        let ws = WSUM.iter().find_map(|k| m.get(*k).and_then(Value::as_array)).context("no per-expert weight sums")?;
        ensure!(toks.len() == ws.len(), "token counts and weight sums differ in length");
        toks.iter()
            .zip(ws)
            .map(|(t, w)| Ok((t.as_f64().context("token count")? as u64, w.as_f64().context("weight sum")?)))
            .collect::<Result<_>>()?
    };
    ensure!(stats.len() == e, "manifest has stats for {} experts, the files hold {e}", stats.len());
    Ok((layer, e, stats))
}

/// The rate of one matrix, in the accountings the repository uses.
fn bits(d_out: usize, d_in: usize) -> Value {
    let weights = (d_out * d_in) as u64;
    let blocks = (d_out * (d_in / BLOCK)) as u64;
    let tail = (d_out * (d_in % BLOCK)) as u64;
    let code = blocks * WORD_BITS;
    let rows = d_out as u64;
    let w = weights as f64;
    json!({
        "weights": weights,
        "blocks": blocks,
        "tail_weights": tail,
        "code_bits": code,
        // rtbits `V5Rate::kernel_bpw(true)`: 48-bit words + f32 row scale + f32 tail; centroids not charged.
        "kernel_bpw": (code + rows * 32 + tail * 32) as f64 / w,
        // calib `Report::bits_per_weight`: words + f16 row scale + f16 tail.
        "report_bpw": (code + rows * 16 + tail * 16) as f64 / w,
        // the archive record as written: f64 row scales, two f64 centroids, f32 tail.
        "archive_bpw": (code + rows * 64 + 2 * 64 + tail * 32) as f64 / w,
    })
}

fn sum_bits(mats: &[(usize, usize)]) -> Value {
    let (mut w, mut k, mut r, mut a) = (0u64, 0f64, 0f64, 0f64);
    for &(o, i) in mats {
        let b = bits(o, i);
        let ww = b["weights"].as_u64().unwrap();
        w += ww;
        k += b["kernel_bpw"].as_f64().unwrap() * ww as f64;
        r += b["report_bpw"].as_f64().unwrap() * ww as f64;
        a += b["archive_bpw"].as_f64().unwrap() * ww as f64;
    }
    json!({"weights": w, "kernel_bpw": k / w as f64, "report_bpw": r / w as f64, "archive_bpw": a / w as f64})
}

struct Group<'a> {
    name: &'static str,
    group: u64,
    n: usize,
    /// `(matrix name, row range in the expert's matrix)`.
    parts: &'a [(&'static str, usize, usize)],
}

#[allow(clippy::too_many_arguments)]
fn encode_group(
    g: &Group,
    w: &[f32],
    h_raw: Vec<f64>,
    tokens: u64,
    wsum: f64,
    layer: u64,
    expert: u64,
    s: &Settings,
    enc: &std::sync::Arc<llvq_search::tetra::Encoder>,
    flags: &mut Vec<String>,
) -> Result<(Vec<f32>, Vec<Value>, Vec<QuantizedMatrix>)> {
    let n = g.n;
    let identity = |n: usize| {
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        h
    };
    let t0 = Instant::now();
    let mut h = if tokens == 0 || wsum <= 0.0 {
        flags.push(format!("{}:identity_h", g.name));
        identity(n)
    } else {
        ensure!(h_raw.iter().all(|v| v.is_finite()), "expert {expert} {}: H holds a non-finite value", g.name);
        let inv = 1.0 / wsum;
        h_raw.into_iter().map(|v| v * inv).collect()
    };
    let sd = seed(layer, expert, g.group);
    let fac = match factor_hessian(h.clone(), n, Some(sd), s.damping) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("expert {expert} {}: factorization failed ({e}); falling back to H = I", g.name);
            flags.push(format!("{}:factor_failed", g.name));
            h = identity(n);
            factor_hessian(h.clone(), n, Some(sd), s.damping).map_err(|e| anyhow::anyhow!("{e}"))?
        }
    };
    let t_factor = t0.elapsed().as_secs_f64();
    let mut recon = vec![0.0f32; w.len()];
    let mut out = Vec::new();
    let mut qms = Vec::new();
    for &(name, r0, r1) in g.parts {
        let t = Instant::now();
        let sub = &w[r0 * n..r1 * n];
        let e = encode_matrix(sub, r1 - r0, n, &fac, s, enc)?;
        let secs = t.elapsed().as_secs_f64();
        let (num, den) = proxy_traces(sub, &e.recon, r1 - r0, n, &h, s.threads);
        let (mut dw2, mut w2) = (0.0f64, 0.0f64);
        for (a, b) in sub.iter().zip(&e.recon) {
            dw2 += ((*b - *a) as f64).powi(2);
            w2 += (*a as f64).powi(2);
        }
        recon[r0 * n..r1 * n].copy_from_slice(&e.recon);
        out.push(json!({
            "matrix": name,
            "d_out": r1 - r0,
            "d_in": n,
            "rotation_seed": sd,
            "proxy_rel": num / den,
            "weight_rel_mse": dw2 / w2,
            "centroids": e.centroids,
            "seconds_encode": secs,
            "seconds_factor_shared": t_factor,
            "bits": bits(r1 - r0, n),
        }));
        qms.push(QuantizedMatrix {
            name: format!("model.layers.{layer}.mlp.experts.{expert}.{name}_proj.weight"),
            d_out: r1 - r0,
            d_in: n,
            codes: e.codes,
            row_scales: e.row_scales,
            centroids: e.centroids,
            rotation_seed: Some(sd),
            shell_cap: llvq_artifact::TETRA_SHELL_CAP,
            tail: e.tail,
        });
    }
    Ok((recon, out, qms))
}

fn open_out(path: &Path, len: u64) -> Result<File> {
    let f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
    let cur = f.metadata()?.len();
    if cur != len {
        ensure!(cur == 0, "{} is {cur} bytes, expected 0 or {len}", path.display());
        f.set_len(len)?;
    }
    Ok(f)
}

fn done_experts(path: &Path) -> Result<BTreeMap<u64, Value>> {
    let mut m = BTreeMap::new();
    if path.exists() {
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&line).context("extq-records.jsonl: a torn line")?;
            m.insert(v["expert"].as_u64().context("record without expert")?, v);
        }
    }
    Ok(m)
}

fn write_summary(dir: &Path, layer: u64, e: usize, s: &Settings, recs: &BTreeMap<u64, Value>) -> Result<()> {
    let expert_bits = sum_bits(&[(F, D), (F, D), (D, F)]);
    let flagged: Vec<u64> = recs
        .values()
        .filter(|r| r["flags"].as_array().is_some_and(|a| !a.is_empty()))
        .map(|r| r["expert"].as_u64().unwrap())
        .collect();
    let summary = json!({
        "tool": "extq",
        "layer": layer,
        "experts_total": e,
        "experts_done": recs.len(),
        "flagged_experts": flagged,
        "recipe": {
            "codebook": "tetra1 (Codebook::Tetra { gain_bits: 1, post_shape_gain: false })",
            "mode": "nogs", "retract": s.gptq.retract, "tail": "KeepExact",
            "damping": s.damping, "h_shrink": 1.0, "gain_scale": s.gain_scale,
            "gain_fit": "Lloyd-Max, 2 levels, 40 iterations, on the rotated weights",
            "rotation": "ROTATION_SEED ^ (layer<<32) ^ (expert<<20) ^ (group<<16), group 0 = w13 input, 1 = w2 input",
            "h_normalisation": "H / weight_sum",
            "gate_up": "w13 split in rows 0..1152 (gate) and 1152..2304 (up), one shared H, rotation and factor",
            "threads": s.threads,
        },
        "expert_bits": expert_bits,
        "experts": recs.values().collect::<Vec<_>>(),
    });
    let tmp = dir.join("encode.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&summary)?)?;
    std::fs::rename(tmp, dir.join("encode.json"))?;
    Ok(())
}

const REC_VERSION: u32 = llvq_artifact::VERSION;

fn codes_path(dir: &Path, expert: u64) -> PathBuf {
    dir.join("codes").join(format!("e{expert:03}.llvqrec"))
}

/// Writes the expert's records, re-reads and decodes them, and checks every
/// decoded weight against `want` bit for bit and every record's payload bits
/// against [`bits`]'s archive accounting. Returns the payload bits per record.
fn write_codes<'a>(
    dir: &Path,
    expert: u64,
    qms: impl Iterator<Item = &'a QuantizedMatrix>,
    tb: &llvq_artifact::Codebook,
    want: &[&[f32]],
) -> Result<Vec<u64>> {
    let path = codes_path(dir, expert);
    let tmp = path.with_extension("tmp");
    let mut buf = Vec::new();
    let mut paid = Vec::new();
    for m in qms {
        let p = write_matrix_with(&mut buf, REC_VERSION, tb, m)?;
        let a = bits(m.d_out, m.d_in)["archive_bpw"].as_f64().unwrap() * (m.d_out * m.d_in) as f64;
        ensure!((p as f64 - a).abs() < 0.5, "{}: record pays {p} bits, the accounting says {a}", m.name);
        paid.push(p);
    }
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_data()?;
    }
    let bad = check_codes(&tmp, want)?;
    ensure!(bad == 0, "expert {expert}: {bad} weights decoded from the codes differ from the reconstruction");
    std::fs::rename(&tmp, &path)?;
    Ok(paid)
}

/// Decodes a codes file and counts the weights that differ, in any bit, from
/// `want` (gate, up, down).
fn check_codes(path: &Path, want: &[&[f32]]) -> Result<usize> {
    let cbs = Codebooks::new();
    let mut r = BufReader::new(File::open(path)?);
    let mut bad = 0;
    for w in want {
        let m = read_matrix_with(&mut r, REC_VERSION, &cbs)?;
        let d = decode_matrix(&m);
        ensure!(d.len() == w.len(), "{}: {} weights decoded, {} expected", m.name, d.len(), w.len());
        bad += d.iter().zip(w.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    }
    let mut rest = Vec::new();
    r.read_to_end(&mut rest)?;
    ensure!(rest.is_empty(), "{}: {} trailing bytes", path.display(), rest.len());
    Ok(bad)
}

/// Every finished expert: its codes file decoded against the hats on disk.
fn verify(dir: PathBuf) -> Result<()> {
    let recs = done_experts(&dir.join("extq-records.jsonl"))?;
    let mut o13 = File::open(dir.join("w13_hat.f32"))?;
    let mut o2 = File::open(dir.join("w2_hat.f32"))?;
    let (mut total, mut bytes) = (0usize, 0u64);
    for &x in recs.keys() {
        let xu = x as usize;
        let r13 = read_f32(&mut o13, (xu * D * D * 4) as u64, D * D)?;
        let r2 = read_f32(&mut o2, (xu * D * F * 4) as u64, D * F)?;
        let p = codes_path(&dir, x);
        bytes += std::fs::metadata(&p)?.len();
        let bad = check_codes(&p, &[&r13[..F * D], &r13[F * D..], &r2[..]])?;
        println!("expert {x}: {bad} of {} weights differ", 3 * F * D);
        total += bad;
    }
    println!("verify: {} experts, {total} differing weights, codes {bytes} bytes", recs.len());
    ensure!(total == 0, "the codes do not decode to the reconstructions");
    Ok(())
}

fn parse_range(s: &str, e: usize) -> Result<(usize, usize)> {
    let (a, b) = s.split_once("..").context("--experts takes a..b (half-open)")?;
    let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
    ensure!(a < b && b <= e, "--experts {a}..{b} outside 0..{e}");
    Ok((a, b))
}

fn encode(dir: PathBuf, experts: Option<String>, layer_arg: Option<u64>) -> Result<()> {
    if !cfg!(feature = "fast-linalg") {
        bail!("extq must be built with --features fast-linalg (the published factorization path)");
    }
    let (layer, e, stats) = manifest(&dir, layer_arg)?;
    let (a, b) = match experts {
        Some(s) => parse_range(&s, e)?,
        None => (0, e),
    };
    let s = Settings::sealed_recipe(threads()?);
    let enc = TetraShapeGain::encoder();
    let tb = llvq_artifact::Codebook::new(CodeKind::Tetra)?;
    std::fs::create_dir_all(dir.join("codes"))?;
    let mut w13 = File::open(dir.join("w13.f32"))?;
    let mut w2 = File::open(dir.join("w2.f32"))?;
    let mut h13 = File::open(dir.join("h13.f64"))?;
    let mut h2 = File::open(dir.join("h2.f64"))?;
    let mut o13 = open_out(&dir.join("w13_hat.f32"), (e * D * D * 4) as u64)?;
    let mut o2 = open_out(&dir.join("w2_hat.f32"), (e * D * F * 4) as u64)?;
    let rec_path = dir.join("extq-records.jsonl");
    let mut recs = done_experts(&rec_path)?;
    eprintln!(
        "extq: layer {layer}, experts {a}..{b} of {e}, {} already done, {} threads",
        (a..b).filter(|x| recs.contains_key(&(*x as u64))).count(),
        s.threads
    );
    let g13 = Group { name: "w13", group: 0, n: D, parts: &[("gate", 0, F), ("up", F, D)] };
    let g2 = Group { name: "w2", group: 1, n: F, parts: &[("down", 0, D)] };
    #[allow(clippy::needless_range_loop)]
    for x in a..b {
        let xe = x as u64;
        if recs.contains_key(&xe) {
            continue;
        }
        let t0 = Instant::now();
        let (tokens, wsum) = stats[x];
        let mut flags = Vec::new();
        let w = read_f32(&mut w13, (x * D * D * 4) as u64, D * D)?;
        let h = read_f64(&mut h13, (x * D * D * 8) as u64, D * D)?;
        let (r13, mut m13, q13) = encode_group(&g13, &w, h, tokens, wsum, layer, xe, &s, &enc, &mut flags)?;
        let w = read_f32(&mut w2, (x * D * F * 4) as u64, D * F)?;
        let h = read_f64(&mut h2, (x * F * F * 8) as u64, F * F)?;
        let (r2, mut m2, q2) = encode_group(&g2, &w, h, tokens, wsum, layer, xe, &s, &enc, &mut flags)?;
        // The codes, as the repository's own records (`write_matrix_with`,
        // format v5, Tetra kind): gate, up, down back to back. Written to a
        // temporary name, synced, read back and decoded against the
        // reconstructions before the rename, so a codes file that exists is
        // one that decodes to the hats.
        let paid = write_codes(&dir, xe, q13.iter().chain(&q2), &tb, &[&r13[..F * D], &r13[F * D..], &r2[..]])?;
        for (m, p) in m13.iter_mut().chain(m2.iter_mut()).zip(&paid) {
            m["payload_bits"] = json!(p);
        }
        write_f32_at(&mut o13, (x * D * D * 4) as u64, &r13)?;
        write_f32_at(&mut o2, (x * D * F * 4) as u64, &r2)?;
        o13.sync_data()?;
        o2.sync_data()?;
        let secs = t0.elapsed().as_secs_f64();
        let rec = json!({
            "expert": xe,
            "tokens": tokens,
            "weight_sum": wsum,
            "flags": flags,
            "seconds": secs,
            "matrices": m13.into_iter().chain(m2).collect::<Vec<_>>(),
        });
        let mut f = OpenOptions::new().create(true).append(true).open(&rec_path)?;
        writeln!(f, "{}", serde_json::to_string(&rec)?)?;
        f.sync_data()?;
        let pr: Vec<String> = rec["matrices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| format!("{} {:.5}", m["matrix"].as_str().unwrap(), m["proxy_rel"].as_f64().unwrap()))
            .collect();
        eprintln!("expert {x}: {secs:.1} s, tokens {tokens}, proxy_rel [{}] {:?}", pr.join(", "), rec["flags"]);
        recs.insert(xe, rec);
        write_summary(&dir, layer, e, &s, &recs)?;
    }
    write_summary(&dir, layer, e, &s, &recs)?;
    Ok(())
}

/// A synthetic layer dir at the real shapes: Gaussian weights, and per expert
/// a Hessian `Σ x xᵀ` from correlated Gaussian rows, unnormalised like the
/// Python side writes it. Expert 1 gets zero tokens when `n_experts >= 2`,
/// so the identity fallback is exercised.
fn fake(dir: PathBuf, n_experts: usize, layer: u64) -> Result<()> {
    std::fs::create_dir_all(&dir)?;
    let mut rng = 0x5EED_u64 ^ layer;
    let mut gauss = move || {
        // SplitMix64 then Box-Muller.
        let mut next = || {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let (u, v) = (next().max(1e-300), next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut f_w13 = File::create(dir.join("w13.f32"))?;
    let mut f_w2 = File::create(dir.join("w2.f32"))?;
    let mut f_h13 = File::create(dir.join("h13.f64"))?;
    let mut f_h2 = File::create(dir.join("h2.f64"))?;
    let mut experts = Vec::new();
    for x in 0..n_experts {
        let mut put32 = |f: &mut File, n: usize, scale: f64| -> Result<()> {
            let mut buf = Vec::with_capacity(n * 4);
            for _ in 0..n {
                buf.extend_from_slice(&((gauss() * scale) as f32).to_le_bytes());
            }
            f.write_all(&buf)?;
            Ok(())
        };
        put32(&mut f_w13, D * D, 0.02)?;
        put32(&mut f_w2, D * F, 0.02)?;
        let tokens: usize = if x == 1 { 0 } else { 3000 };
        let mut weights_sum = 0.0;
        // H = Σ w² x xᵀ with x = z + 0.5·shift, a few heavy channels; built
        // as Gram matrices of random rows (2·n rows, so it is full rank).
        for (f, n) in [(&mut f_h13, D), (&mut f_h2, F)] {
            let mut h = vec![0.0f64; n * n];
            if tokens > 0 {
                let rows = 2 * n;
                let mut a = vec![0.0f64; rows * n];
                for r in 0..rows {
                    for c in 0..n {
                        let heavy = if c % 97 == 0 { 8.0 } else { 1.0 };
                        a[r * n + c] = gauss() * heavy;
                    }
                }
                for i in 0..n {
                    for j in 0..=i {
                        let mut s = 0.0;
                        for r in 0..rows {
                            s += a[r * n + i] * a[r * n + j];
                        }
                        h[i * n + j] = s * 0.25;
                        h[j * n + i] = s * 0.25;
                    }
                }
                weights_sum = 0.25 * tokens as f64;
            }
            let mut buf = Vec::with_capacity(n * n * 8);
            for v in &h {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            f.write_all(&buf)?;
        }
        experts.push(json!({"tokens": tokens, "weight_sum": weights_sum}));
    }
    let m = json!({
        "layer": layer,
        "num_experts": n_experts,
        "shapes": {"w13": [n_experts, D, D], "w2": [n_experts, D, F], "h13": [n_experts, D, D], "h2": [n_experts, F, F]},
        "synthetic": true,
        "experts": experts,
    });
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m)?)?;
    eprintln!("fake layer dir written: {} ({n_experts} experts, layer {layer})", dir.display());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut experts, mut layer) = (None, None);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--experts" => experts = Some(it.next().context("--experts needs a..b")?),
            "--layer" => layer = Some(it.next().context("--layer needs N")?.parse::<u64>()?),
            _ if a.starts_with("--") => bail!("unknown flag {a}"),
            _ => pos.push(a),
        }
    }
    match pos.as_slice() {
        [cmd, dir, rest @ ..] if cmd == "fake" => {
            let n = rest.first().map(|s| s.parse()).transpose()?.unwrap_or(2);
            fake(PathBuf::from(dir), n, layer.unwrap_or(0))
        }
        [cmd, dir] if cmd == "verify" => verify(PathBuf::from(dir)),
        [dir] => encode(PathBuf::from(dir), experts, layer),
        _ => bail!("usage: extq <layer_dir> [--experts a..b] [--layer N] | extq fake <dir> [n] [--layer N]"),
    }
}
