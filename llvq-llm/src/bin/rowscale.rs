//! Folds trained per-row multipliers into an artifact's `row_scales`.
//!
//! ```text
//! cargo run --release -p llvq-llm --bin rowscale -- \
//!   in.bin out.bin sigma.json
//! ```
//!
//! ## Why this exists rather than a re-encode
//!
//! A Tetra block reconstructs as `centroids[g] * row_scale * u`, so
//! multiplying `row_scales[i]` by `s` multiplies row `i` and only row `i`.
//! `llvq-bench/examples/rhoapply.rs` proves that claim on a record taken from
//! the artifact itself, and its `rewrite` already takes a factor of
//! `(matrix, row)`. This binary is the same edit driven by a vector instead of
//! a constant.
//!
//! No code moves, no index byte changes, the rate is untouched and the decoder
//! stays byte-identical. That is why training row scales costs zero bits.
//!
//! ## What it refuses
//!
//! An `int4` record holds no `row_scales`, so there is nothing to fold into.
//! Those records are copied through by kind and counted, the same exclusion
//! `rhoapply` makes by construction. `v_proj` is served at int4 in
//! `configs/qwen3-4b-tetra-q5.json`, so it is never scaled.
//!
//! A `free_params` export is refused by name. It also carries trained tails,
//! and applying half of an export silently would write a file that matches
//! neither the artifact nor what was trained.
//!
//! ## `row_norms`: the norms too
//!
//! A `row_norms` export (`ops/llvqtune/llvqtune/trainables/row_norms.py`)
//! carries `tau` beside `sigma`: one multiplier per value of an RMSNorm
//! weight. Those weights are raw f16 tensors in the sealed section, so the
//! fold reads that section tensor by tensor, writes `w * tau` rounded to f16
//! once, and copies the blobs after it verbatim. A norm the export names and
//! the file does not hold is refused, as is one that is not stored in f16.
//! With `tau` all ones the section is re-serialized byte for byte.
//!
//! ## The idempotence control
//!
//! A multiplier of exactly 1.0 is skipped rather than multiplied. So a run
//! whose sigma is all ones writes a byte-identical file, and a diff of the two
//! files shows exactly the rows that moved and nothing else.

use anyhow::Context;
use llvq_artifact as format;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};

/// The artifact's own key for a record, through the artifact's own splitter.
///
/// `split_name` drops the last segment, so a name that already ends in
/// `.weight` and one that does not must be normalized first. The trainer
/// names a matrix `model.layers.0.self_attn.q_proj`; the artifact stores
/// `model.layers.0.self_attn.q_proj.weight`. Both have to land on the same
/// key or every lookup misses and the tool reports "untouched" while writing
/// a copy. That failure is not hypothetical: `artscale` shipped it once.
fn key_of(name: &str) -> anyhow::Result<(usize, String)> {
    let normalized = if name.ends_with(".weight") {
        name.to_string()
    } else {
        format!("{name}.weight")
    };
    format::split_name(&normalized).map_err(|e| anyhow::anyhow!("{name}: {e}"))
}

/// Row multipliers keyed like the artifact's records, and norm multipliers
/// keyed by the norm's tensor name without `.weight`. `tau` is empty for a
/// `row_scales` export.
type Export = (HashMap<(usize, String), Vec<f64>>, HashMap<String, Vec<f64>>);

/// Reads the trainer's export. Accepts the sink's wrapper or a bare payload.
fn export(path: &str) -> anyhow::Result<Export> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let doc: serde_json::Value = serde_json::from_str(&text).context("parsing the export")?;
    let payload = doc.get("result").unwrap_or(&doc);

    let kind = payload.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    anyhow::ensure!(
        kind == "row_scales" || kind == "row_norms",
        "export kind is {kind:?}; this tool folds row_scales and row_norms only, \
         and a free_params export also carries trained tails"
    );

    let table = payload
        .get("sigma")
        .and_then(|s| s.as_object())
        .context("the export has no `sigma` object")?;
    anyhow::ensure!(!table.is_empty(), "the export scales no matrix");

    let mut out = HashMap::with_capacity(table.len());
    for (name, values) in table {
        out.insert(key_of(name)?, multipliers(name, "sigma", values)?);
    }

    let mut tau = HashMap::new();
    if kind == "row_norms" {
        let norms = payload
            .get("tau")
            .and_then(|s| s.as_object())
            .context("a row_norms export has no `tau` object")?;
        anyhow::ensure!(!norms.is_empty(), "a row_norms export scales no norm");
        for (name, values) in norms {
            let bare = name.strip_suffix(".weight").unwrap_or(name);
            tau.insert(bare.to_string(), multipliers(name, "tau", values)?);
        }
    } else {
        anyhow::ensure!(
            payload.get("tau").is_none(),
            "a row_scales export carries `tau`; it would be dropped silently"
        );
    }
    Ok((out, tau))
}

/// One vector of positive finite multipliers, or a refusal naming the slot.
fn multipliers(name: &str, what: &str, values: &serde_json::Value) -> anyhow::Result<Vec<f64>> {
    let list = values
        .as_array()
        .with_context(|| format!("{name}: {what} is not an array"))?;
    let mut row = Vec::with_capacity(list.len());
    for (i, v) in list.iter().enumerate() {
        let s = v
            .as_f64()
            .with_context(|| format!("{name}: {what}[{i}] is not a number"))?;
        anyhow::ensure!(
            s.is_finite() && s > 0.0,
            "{name}: {what}[{i}] = {s} is not a positive finite number"
        );
        row.push(s);
    }
    anyhow::ensure!(!row.is_empty(), "{name}: {what} is empty");
    Ok(row)
}

/// What a fold did, so the caller can say it out loud.
#[derive(Default, Debug)]
struct Report {
    scaled: usize,
    untouched: usize,
    int4: usize,
    rows: usize,
    worst: f64,
    seen: Vec<(usize, String)>,
    norms_scaled: usize,
    norm_values: usize,
    norm_worst: f64,
}

/// Rewrite every record, multiplying the row scales the export names.
///
/// Takes readers and writers rather than paths so the controls can run on
/// memory buffers. The two that matter are idempotence, a sigma of all ones
/// must reproduce the input byte for byte, and surgery, one changed row must
/// move one f64 and nothing else.
fn fold(
    r: &mut impl std::io::Read,
    w: &mut impl std::io::Write,
    by_key: &HashMap<(usize, String), Vec<f64>>,
    tau: &HashMap<String, Vec<f64>>,
) -> anyhow::Result<Report> {
    let head = format::read_header(r)?;
    format::write_header_kinds(w, head.version, head.matrices, head.default_kind, head.kinds)?;
    let mut rep = Report::default();

    for _ in 0..head.matrices {
        let mut rec = format::read_record(r, head.version)?;
        match &mut rec {
            format::Record::Lattice(m) => {
                let key = key_of(&m.name)?;
                match by_key.get(&key) {
                    Some(sigma) => {
                        anyhow::ensure!(
                            sigma.len() == m.row_scales.len(),
                            "{}: {} multipliers for {} rows",
                            m.name,
                            sigma.len(),
                            m.row_scales.len()
                        );
                        let mut moved = false;
                        for (s, &f) in m.row_scales.iter_mut().zip(sigma) {
                            // 1.0 is skipped, not multiplied: an untouched row
                            // must come out byte for byte identical.
                            if f != 1.0 {
                                *s *= f;
                                moved = true;
                                rep.worst = rep.worst.max((f - 1.0).abs());
                            }
                        }
                        rep.rows += sigma.len();
                        if moved {
                            rep.scaled += 1;
                        } else {
                            rep.untouched += 1;
                        }
                        rep.seen.push(key);
                    }
                    None => rep.untouched += 1,
                }
            }
            // Counted rather than silently skipped: an int4 record has no
            // `row_scales`, and saying so out loud is cheaper than asking the
            // reader to trust a branch.
            format::Record::Int4(_) => rep.int4 += 1,
        }
        format::write_record(w, head.version, &rec)?;
    }

    // Everything after the last record, copied verbatim. On a sealed file that
    // is the raw tensors and the blobs. Copying the bytes rather than
    // re-serializing keeps the claim honest: this edits `row_scales` and
    // nothing else, including the parts of the format it does not model.
    if !tau.is_empty() {
        fold_norms(r, w, head.version, tau, &mut rep)?;
    }
    std::io::copy(r, w)?;
    Ok(rep)
}

/// Rewrite the raw-tensor section, multiplying the norms `tau` names.
///
/// Leaves `r` at the blob count, so the caller's verbatim copy takes the
/// blobs. Every tensor is re-serialized, which is byte-identical for an
/// untouched one: `write_raw` is the inverse of `read_raw` on `LVQ3` framing.
fn fold_norms(
    r: &mut impl std::io::Read,
    w: &mut impl std::io::Write,
    version: u32,
    tau: &HashMap<String, Vec<f64>>,
    rep: &mut Report,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        version >= 3,
        "version {version} carries untagged raw tensors; the norm fold writes LVQ3 framing"
    );
    let mut count = [0u8; 4];
    r.read_exact(&mut count).context("reading the raw tensor count")?;
    w.write_all(&count)?;
    let mut seen = Vec::new();
    for _ in 0..u32::from_le_bytes(count) {
        let mut t = format::read_raw(r, version)?;
        let bare = t.name.strip_suffix(".weight").unwrap_or(&t.name).to_string();
        if let Some(f) = tau.get(&bare) {
            let len = t.len();
            let format::RawData::F16(values) = &mut t.data else {
                anyhow::bail!("{}: not stored in f16, the norm fold refuses it", t.name);
            };
            anyhow::ensure!(
                f.len() == len,
                "{}: {} multipliers for {len} values",
                t.name,
                f.len()
            );
            let mut moved = false;
            for (v, &m) in values.iter_mut().zip(f) {
                // 1.0 is skipped, as for the row scales: an untouched value
                // must come out byte for byte identical.
                if m != 1.0 {
                    let x = half::f16::from_bits(*v).to_f64() * m;
                    *v = half::f16::from_f64(x).to_bits();
                    moved = true;
                    rep.norm_worst = rep.norm_worst.max((m - 1.0).abs());
                }
            }
            rep.norm_values += len;
            if moved {
                rep.norms_scaled += 1;
            }
            seen.push(bare);
        }
        format::write_raw(w, &t)?;
    }
    let missing: Vec<&String> = tau.keys().filter(|k| !seen.contains(k)).collect();
    anyhow::ensure!(
        missing.is_empty(),
        "the export names {} norms the artifact does not hold, first {:?}",
        missing.len(),
        missing.first()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() == 3,
        "usage: rowscale <in.bin|in.llvq> <out> <sigma.json>"
    );
    let (by_key, tau) = export(&a[2])?;

    let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(&a[0])?);
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(&a[1])?);
    let rep = fold(&mut r, &mut w, &by_key, &tau)?;
    use std::io::Write as _;
    w.flush()?;

    let missing: Vec<&(usize, String)> = by_key.keys().filter(|k| !rep.seen.contains(k)).collect();
    anyhow::ensure!(
        missing.is_empty(),
        "the export names {} matrices the artifact does not hold, first {:?}",
        missing.len(),
        missing.first()
    );

    println!(
        "{} scaled, {} untouched, {} int4 passed through",
        rep.scaled, rep.untouched, rep.int4
    );
    println!(
        "{} row scales read, largest multiplier deviation {:.6}",
        rep.rows, rep.worst
    );
    if !tau.is_empty() {
        println!(
            "{} norms scaled of {} named, {} norm values read, largest deviation {:.6}",
            rep.norms_scaled,
            tau.len(),
            rep.norm_values,
            rep.norm_worst
        );
    }
    if rep.scaled == 0 && rep.norms_scaled == 0 {
        println!("every multiplier was 1.0, so the output is byte-identical");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_artifact::{ArtifactWriter, CodeKind, QuantizedMatrix, TETRA_SHELL_CAP};
    use llvq_core::{SplitMix64, DIM};
    use llvq_quant::quantizer::BlockCode;
    use llvq_search::tetra::{Tetra, LABEL_MASK};
    use std::io::Write;

    fn tmp(name: &str, body: &str) -> String {
        let p = std::env::temp_dir().join(format!("rowscale-{name}.json"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p.to_string_lossy().into_owned()
    }

    /// A two-matrix artifact, in memory. `d_in` carries a tail so the record
    /// exercises the field that sits after `row_scales`.
    fn fixture() -> Vec<u8> {
        build(false)
    }

    /// The same two matrices, sealed with two f16 norms, one other f16
    /// tensor and a blob, so the fold has a raw section to walk and bytes
    /// after it to preserve.
    fn sealed_fixture() -> Vec<u8> {
        build(true)
    }

    fn build(sealed: bool) -> Vec<u8> {
        let tetra = Tetra::new();
        let mut rng = SplitMix64::new(0x0005_CA1E_5EED);
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = ArtifactWriter::with_kind(&mut buf, CodeKind::Tetra, 2).expect("header");
            for (name, d_out, d_in) in [
                ("model.layers.0.self_attn.q_proj.weight", 4usize, 2 * DIM + 8),
                ("model.layers.0.mlp.up_proj.weight", 3, 3 * DIM),
            ] {
                let codes: Vec<BlockCode> = (0..d_out * (d_in / DIM))
                    .map(|_| BlockCode {
                        point: tetra.decode(rng.next() & LABEL_MASK),
                        gain: (rng.next() & 1) as u32,
                    })
                    .collect();
                let m = QuantizedMatrix {
                    name: name.to_string(),
                    d_out,
                    d_in,
                    codes,
                    row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
                    centroids: vec![0.7, 1.1],
                    rotation_seed: Some(0xABCD),
                    shell_cap: TETRA_SHELL_CAP,
                    tail: (0..d_out * (d_in % DIM)).map(|_| rng.next_f64()).collect(),
                };
                w.push(&m).expect("write");
            }
            if sealed {
                let (raws, blobs) = sealed_section();
                w.seal(&raws, &blobs).expect("seal");
            } else {
                w.finish().expect("flush");
            }
        }
        buf
    }

    fn run(src: &[u8], by_key: &HashMap<(usize, String), Vec<f64>>) -> (Vec<u8>, Report) {
        let mut r = std::io::Cursor::new(src.to_vec());
        let mut out: Vec<u8> = Vec::new();
        let rep = fold(&mut r, &mut out, by_key, &HashMap::new()).expect("fold");
        (out, rep)
    }

    #[test]
    fn a_sigma_of_all_ones_reproduces_the_file_byte_for_byte() {
        let src = fixture();
        let mut by_key = HashMap::new();
        by_key.insert((0, "self_attn.q_proj".to_string()), vec![1.0; 4]);
        by_key.insert((0, "mlp.up_proj".to_string()), vec![1.0; 3]);
        let (out, rep) = run(&src, &by_key);
        assert_eq!(out, src, "an untouched fold must be byte-identical");
        assert_eq!(rep.scaled, 0);
        assert_eq!(rep.untouched, 2);
        assert_eq!(rep.rows, 7);
    }

    #[test]
    fn an_empty_export_leaves_every_record_alone() {
        let src = fixture();
        let (out, rep) = run(&src, &HashMap::new());
        assert_eq!(out, src);
        assert_eq!(rep.untouched, 2);
        assert_eq!(rep.rows, 0);
    }

    #[test]
    fn one_changed_row_moves_one_f64_and_nothing_else() {
        let src = fixture();
        let mut sigma = vec![1.0; 3];
        sigma[1] = 1.5;
        let mut by_key = HashMap::new();
        by_key.insert((0, "mlp.up_proj".to_string()), sigma);
        let (out, rep) = run(&src, &by_key);

        assert_eq!(out.len(), src.len(), "the width never moves");
        assert_eq!(rep.scaled, 1);
        assert_eq!(rep.worst, 0.5);

        let differing: Vec<usize> = (0..src.len()).filter(|&i| src[i] != out[i]).collect();
        assert!(!differing.is_empty(), "the fold changed nothing");
        let span = differing[differing.len() - 1] - differing[0] + 1;
        assert!(span <= 8, "{span} bytes moved, one f64 is 8");

        // The slot holds exactly 1.5 times what it held. Reading it back is
        // what proves the edit landed on the scale and not beside it.
        let start = differing[0] + 1 - span.min(differing[0] + 1);
        let mut found = false;
        for s in start..=differing[0] {
            if s + 8 > src.len() {
                break;
            }
            let a = f64::from_le_bytes(src[s..s + 8].try_into().unwrap());
            let b = f64::from_le_bytes(out[s..s + 8].try_into().unwrap());
            if a != 0.0 && (b / a - 1.5).abs() < 1e-12 {
                found = true;
                break;
            }
        }
        assert!(found, "no 8-byte slot around the diff reads exactly 1.5x");
    }

    #[test]
    fn a_sigma_of_the_wrong_length_is_refused() {
        let src = fixture();
        let mut by_key = HashMap::new();
        by_key.insert((0, "mlp.up_proj".to_string()), vec![1.0; 99]);
        let mut r = std::io::Cursor::new(src);
        let mut out: Vec<u8> = Vec::new();
        let e = fold(&mut r, &mut out, &by_key, &HashMap::new())
            .unwrap_err()
            .to_string();
        assert!(e.contains("99 multipliers for 3 rows"), "{e}");
    }

    fn sealed_section() -> (Vec<format::RawTensor>, Vec<format::Blob>) {
        let f16s = |v: &[f32]| -> Vec<u16> {
            v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
        };
        let raws = vec![
            format::RawTensor {
                name: "model.layers.0.input_layernorm.weight".into(),
                dims: vec![4],
                data: format::RawData::F16(f16s(&[0.5, 1.25, -0.75, 2.0])),
            },
            format::RawTensor {
                name: "model.embed_tokens.weight".into(),
                dims: vec![2, 3],
                data: format::RawData::F16(f16s(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
            },
            format::RawTensor {
                name: "model.norm.weight".into(),
                dims: vec![3],
                data: format::RawData::F16(f16s(&[1.0, 1.5, 0.25])),
            },
        ];
        let blobs = vec![format::Blob {
            name: "config.json".into(),
            bytes: b"{\"hidden_size\": 4}".to_vec(),
        }];
        (raws, blobs)
    }

    fn run_norms(
        src: &[u8],
        tau: &HashMap<String, Vec<f64>>,
    ) -> anyhow::Result<(Vec<u8>, Report)> {
        let mut r = std::io::Cursor::new(src.to_vec());
        let mut out: Vec<u8> = Vec::new();
        let rep = fold(&mut r, &mut out, &HashMap::new(), tau)?;
        Ok((out, rep))
    }

    #[test]
    fn a_tau_of_all_ones_reproduces_the_sealed_file_byte_for_byte() {
        let src = sealed_fixture();
        let mut tau = HashMap::new();
        tau.insert("model.layers.0.input_layernorm".to_string(), vec![1.0; 4]);
        tau.insert("model.norm".to_string(), vec![1.0; 3]);
        let (out, rep) = run_norms(&src, &tau).unwrap();
        assert_eq!(out, src, "re-serializing the raw section must be exact");
        assert_eq!(rep.norms_scaled, 0);
        assert_eq!(rep.norm_values, 7);
    }

    #[test]
    fn one_norm_value_doubled_moves_one_half_word_and_nothing_else() {
        let src = sealed_fixture();
        let mut tau = HashMap::new();
        tau.insert("model.norm".to_string(), vec![1.0, 2.0, 1.0]);
        let (out, rep) = run_norms(&src, &tau).unwrap();
        assert_eq!(out.len(), src.len(), "the width never moves");
        assert_eq!(rep.norms_scaled, 1);

        let differing: Vec<usize> = (0..src.len()).filter(|&i| src[i] != out[i]).collect();
        assert!(!differing.is_empty(), "the fold changed nothing");
        let span = differing[differing.len() - 1] - differing[0] + 1;
        assert!(span <= 2, "{span} bytes moved, one f16 is 2");

        // 1.5 doubled is 3.0, exact in binary16. Find the slot and read it.
        let old = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        let new = half::f16::from_f32(3.0).to_bits().to_le_bytes();
        let at = (0..src.len() - 1)
            .find(|&i| src[i..i + 2] == old && out[i..i + 2] == new)
            .expect("no half-word went from 1.5 to 3.0");
        assert!(differing.iter().all(|&i| i == at || i == at + 1));
    }

    #[test]
    fn a_norm_the_file_does_not_hold_is_refused() {
        let src = sealed_fixture();
        let mut tau = HashMap::new();
        tau.insert("model.layers.9.input_layernorm".to_string(), vec![1.1; 4]);
        let e = run_norms(&src, &tau).unwrap_err().to_string();
        assert!(e.contains("norms the artifact does not hold"), "{e}");
    }

    #[test]
    fn a_tau_of_the_wrong_length_is_refused() {
        let src = sealed_fixture();
        let mut tau = HashMap::new();
        tau.insert("model.norm".to_string(), vec![1.1; 5]);
        let e = run_norms(&src, &tau).unwrap_err().to_string();
        assert!(e.contains("5 multipliers for 3 values"), "{e}");
    }

    #[test]
    fn a_row_norms_export_carries_both_vectors() {
        let p = tmp(
            "rownorms",
            r#"{"result":{"kind":"row_norms",
               "sigma":{"model.layers.0.mlp.up_proj":[1.0,2.0]},
               "tau":{"model.norm":[1.5,1.0,0.5]}}}"#,
        );
        let (sigma, tau) = export(&p).unwrap();
        assert_eq!(sigma.len(), 1);
        assert_eq!(tau["model.norm"], vec![1.5, 1.0, 0.5]);
    }

    #[test]
    fn a_row_norms_export_without_tau_is_refused() {
        let p = tmp(
            "notau",
            r#"{"kind":"row_norms","sigma":{"model.layers.0.mlp.up_proj":[1.0]}}"#,
        );
        assert!(export(&p).unwrap_err().to_string().contains("no `tau`"));
    }

    #[test]
    fn a_row_scales_export_carrying_tau_is_refused() {
        let p = tmp(
            "stau",
            r#"{"kind":"row_scales","sigma":{"model.layers.0.mlp.up_proj":[1.0]},
               "tau":{"model.norm":[1.0]}}"#,
        );
        assert!(export(&p).unwrap_err().to_string().contains("dropped silently"));
    }

    #[test]
    fn a_trainer_name_and_an_artifact_name_land_on_one_key() {
        let a = key_of("model.layers.7.self_attn.q_proj").unwrap();
        let b = key_of("model.layers.7.self_attn.q_proj.weight").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, (7, "self_attn.q_proj".to_string()));
    }

    #[test]
    fn the_sink_wrapper_and_a_bare_payload_both_parse() {
        let bare = tmp(
            "bare",
            r#"{"kind":"row_scales","sigma":{"model.layers.0.mlp.up_proj":[1.0,2.0]}}"#,
        );
        let wrapped = tmp(
            "wrapped",
            r#"{"cost":{"is_free":true},"result":{"kind":"row_scales",
               "sigma":{"model.layers.0.mlp.up_proj":[1.0,2.0]}}}"#,
        );
        assert_eq!(export(&bare).unwrap().0, export(&wrapped).unwrap().0);
    }

    #[test]
    fn a_free_params_export_is_refused_by_name() {
        let p = tmp("free", r#"{"kind":"free_params","sigma":{"a":[1.0]}}"#);
        let e = export(&p).unwrap_err().to_string();
        assert!(e.contains("free_params"), "{e}");
    }

    #[test]
    fn a_non_positive_multiplier_is_refused() {
        let p = tmp(
            "neg",
            r#"{"kind":"row_scales","sigma":{"model.layers.0.mlp.up_proj":[1.0,-0.5]}}"#,
        );
        let e = export(&p).unwrap_err().to_string();
        assert!(e.contains("positive finite"), "{e}");
    }

    #[test]
    fn a_non_finite_multiplier_is_refused() {
        let p = tmp(
            "nan",
            r#"{"kind":"row_scales","sigma":{"model.layers.0.mlp.up_proj":[1e400]}}"#,
        );
        assert!(export(&p).is_err());
    }

    #[test]
    fn an_export_that_scales_nothing_is_refused() {
        let p = tmp("empty", r#"{"kind":"row_scales","sigma":{}}"#);
        let e = export(&p).unwrap_err().to_string();
        assert!(e.contains("scales no matrix"), "{e}");
    }

    #[test]
    fn an_empty_vector_is_refused() {
        let p = tmp(
            "emptyvec",
            r#"{"kind":"row_scales","sigma":{"model.layers.0.mlp.up_proj":[]}}"#,
        );
        assert!(export(&p).unwrap_err().to_string().contains("sigma is empty"));
    }
}
