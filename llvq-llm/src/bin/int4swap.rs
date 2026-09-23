//! Transplant int4 records into a sealed file, in place of lattice records.
//!
//! Usage:
//!   `LLVQ_MODEL=<checkpoint> cargo run --release -p llvq-llm --bin int4swap -- in.bin out.bin <types>`
//!
//! `<types>` takes the `LLVQ_RESTORE_Q4` grammar, windows included:
//! `o_proj,down_proj@12-23`. Every lattice record it covers is replaced by the
//! int4 g128 record [`llvq_llm::sealed::int4_record`] builds from the
//! checkpoint's tensor of the same name. Every other record, every raw tensor
//! and every blob is copied byte for byte.
//!
//! This is L37 of `docs/ROADMAP-QUALITY.md`: what `LLVQ_RESTORE_Q4` measured on
//! the dense path becomes a file a kernel can serve. The two are one
//! arithmetic. The restore path loads the checkpoint tensor, narrows it to f16
//! and calls `quantize_affine(4, 128)` (`sealed::quantize_dequantize_q4`); this
//! tool loads the same tensor, narrows it to f16 and makes the same call
//! (`sealed::int4_record`). `the_two_int4_paths_agree_bit_for_bit` pins the
//! pair, so the dense reconstruction of the output is the restored arm, value
//! for value, and the trained row scales of the replaced matrices are dropped
//! exactly as the restore dropped them.
//!
//! The header keeps the file's version and default kind, and declares
//! `Int4G128` if it did not already. A file below v5 is refused: its records
//! carry no kind tag, so an int4 record cannot be written into it.

use candle_core::{DType, Device, Tensor};
use llvq_artifact::{self as format, CodeKind, Record};
use llvq_llm::sealed::{int4_record, RestoreF16};

/// What a pass did, so the caller can say it out loud.
#[derive(Default, Debug)]
struct Report {
    version: u32,
    kinds_before: String,
    kinds_after: String,
    /// Lattice records copied through undecoded.
    lattice_kept: u32,
    /// Int4 records the input already carried, copied through.
    int4_kept: u32,
    /// `(name, weights, bytes before, bytes after)` of each record replaced.
    replaced: Vec<(String, usize, u64, u64)>,
}

impl Report {
    fn weights(&self) -> usize {
        self.replaced.iter().map(|r| r.1).sum()
    }
}

/// Copy `r` to `w`, replacing every lattice record `spec` covers by the int4
/// record of `fetch(name)`.
///
/// Readers, writers and a closure rather than paths and a checkpoint, so the
/// controls run on memory buffers and synthetic tensors. Every type named in
/// `spec` must replace at least one record, as `restore_projections` demands:
/// a transplant that silently replaced nothing would ship the file it was
/// asked to change.
fn transplant(
    r: &mut impl std::io::Read,
    w: &mut impl std::io::Write,
    spec: &RestoreF16,
    mut fetch: impl FnMut(&str) -> anyhow::Result<Tensor>,
) -> anyhow::Result<Report> {
    anyhow::ensure!(!spec.is_empty(), "no projection type to transplant");
    let head = format::read_header(r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "a projections-only artifact (format v{}), seal it first",
        head.version
    );
    anyhow::ensure!(
        head.version >= format::FIRST_KINDED_VERSION,
        "format v{}: its records carry no kind tag, so an int4 record cannot be written \
         into it (v{} and later)",
        head.version,
        format::FIRST_KINDED_VERSION
    );
    let kinds = head.kinds().with(CodeKind::Int4G128);
    format::write_header_kinds(w, head.version, head.matrices, head.default_kind(), kinds)?;
    let mut rep = Report {
        version: head.version,
        kinds_before: head.kinds().to_string(),
        kinds_after: kinds.to_string(),
        ..Report::default()
    };

    for _ in 0..head.matrices {
        let rec = format::read_record(r, head.version)?;
        let out = match rec {
            Record::Int4(m) => {
                anyhow::ensure!(
                    !spec.covers(&m.name),
                    "{}: already an int4 record, and a transplant never requantizes one",
                    m.name
                );
                rep.int4_kept += 1;
                Record::Int4(m)
            }
            Record::Lattice(m) if spec.covers(&m.name) => {
                let t = fetch(&m.name)?;
                anyhow::ensure!(
                    t.dims() == [m.d_out, m.d_in],
                    "{}: the file carries {}x{}, the checkpoint {:?}, not the same model",
                    m.name,
                    m.d_out,
                    m.d_in,
                    t.dims()
                );
                let q = int4_record(&m.name, &t)?;
                let before =
                    format::write_record(&mut std::io::sink(), head.version, &Record::Lattice(m))?;
                let rec = Record::Int4(q);
                let after = format::write_record(&mut std::io::sink(), head.version, &rec)?;
                let (d_out, d_in) = rec.dims();
                rep.replaced
                    .push((rec.name().to_string(), d_out * d_in, before / 8, after / 8));
                rec
            }
            Record::Lattice(m) => {
                rep.lattice_kept += 1;
                Record::Lattice(m)
            }
        };
        format::write_record(w, head.version, &out)?;
    }
    for t in spec.types() {
        anyhow::ensure!(
            rep.replaced
                .iter()
                .any(|r| r.0.ends_with(&format!(".{t}.weight"))),
            "{t}: no lattice record of that type in its window, nothing to transplant"
        );
    }

    // The raw tensors and the blobs are not touched, so they are copied as
    // bytes, the way `rowscale` and `embedq` copy what follows their records.
    std::io::copy(r, w)?;
    Ok(rep)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst, types) = match (a.first(), a.get(1), a.get(2)) {
        (Some(s), Some(d), Some(t)) => (s.clone(), d.clone(), t.clone()),
        _ => anyhow::bail!("usage: LLVQ_MODEL=<checkpoint> int4swap <in> <out> <types, e.g. o_proj,down_proj@12-23>"),
    };
    let spec = RestoreF16::parse(&types).map_err(anyhow::Error::msg)?;
    let repo = std::env::var("LLVQ_MODEL").map_err(|_| {
        anyhow::anyhow!(
            "int4swap requires LLVQ_MODEL=<checkpoint>: the int4 records come from there"
        )
    })?;
    let ck = llvq_llm::loader::Checkpoint::fetch(&repo)?;
    // Safety: the checkpoint files are not modified while the mapping is
    // alive, the contract `sealed::load_with_restored` states for the same
    // files.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&ck.weights)? };
    // Loaded, then narrowed to f16: the order the restore path follows at
    // dtype f16, which is the dtype every census arm ran at.
    let fetch = |name: &str| -> anyhow::Result<Tensor> {
        Ok(st.load(name, &Device::Cpu)?.to_dtype(DType::F16)?)
    };

    let mut r = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&src)?);
    let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&dst)?);
    let rep = match transplant(&mut r, &mut w, &spec, fetch) {
        Ok(rep) => rep,
        Err(e) => {
            drop(w);
            let _ = std::fs::remove_file(&dst);
            anyhow::bail!("{src}: {e}");
        }
    };
    use std::io::Write as _;
    w.flush()?;

    let source = match &ck.source {
        llvq_llm::loader::Source::Local(p) => p.display().to_string(),
        llvq_llm::loader::Source::Hub { repo, revision } => format!("{repo}@{revision}"),
    };
    eprintln!(
        "{src}: format v{}, kinds {} -> {}",
        rep.version, rep.kinds_before, rep.kinds_after
    );
    eprintln!("checkpoint {source}, shards:");
    for p in &ck.weights {
        eprintln!("  {}", p.display());
    }
    for (name, n, b0, b1) in &rep.replaced {
        eprintln!("  {name}: {n} weights, lattice {b0} B -> int4 g128 {b1} B");
    }
    eprintln!(
        "transplanted {} ({} matrices, {} weights); {} lattice + {} int4 records passed through undecoded",
        spec.describe(),
        rep.replaced.len(),
        rep.weights(),
        rep.lattice_kept,
        rep.int4_kept
    );
    let (src_b, dst_b) = (
        std::fs::metadata(&src)?.len(),
        std::fs::metadata(&dst)?.len(),
    );
    println!("\n── {dst}");
    println!(
        "   file       {:.3} GB -> {:.3} GB ({:+} B)",
        src_b as f64 / 1e9,
        dst_b as f64 / 1e9,
        dst_b as i64 - src_b as i64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_artifact::{Blob, KindSet, QuantizedMatrix, RawData, RawTensor, TETRA_SHELL_CAP};
    use llvq_core::{SplitMix64, DIM};
    use llvq_quant::quantizer::BlockCode;
    use llvq_search::tetra::{Tetra, LABEL_MASK};

    const D_OUT: usize = 4;
    const D_IN: usize = 3 * 128; // a multiple of both DIM (24) and the int4 group

    fn lattice(name: &str, rng: &mut SplitMix64, tetra: &Tetra) -> QuantizedMatrix {
        QuantizedMatrix {
            name: name.to_string(),
            d_out: D_OUT,
            d_in: D_IN,
            codes: (0..D_OUT * (D_IN / DIM))
                .map(|_| BlockCode {
                    point: tetra.decode(rng.next() & LABEL_MASK),
                    gain: (rng.next() & 1) as u32,
                })
                .collect(),
            row_scales: (0..D_OUT).map(|_| 1e-3 + rng.next_f64()).collect(),
            centroids: vec![0.7, 1.1],
            rotation_seed: Some(0xABCD),
            shell_cap: TETRA_SHELL_CAP,
            tail: Vec::new(),
        }
    }

    /// The checkpoint, as a pure function of the name: every call for a name
    /// returns the same tensor, and two names never share one.
    fn checkpoint(name: &str) -> anyhow::Result<Tensor> {
        let seed = name.bytes().fold(0x00C0_FFEEu64, |h, b| {
            h.wrapping_mul(131).wrapping_add(b as u64)
        });
        let mut rng = SplitMix64::new(seed);
        let v: Vec<f32> = (0..D_OUT * D_IN)
            .map(|_| (rng.next_f64() - 0.5) as f32)
            .collect();
        Ok(Tensor::from_vec(v, (D_OUT, D_IN), &Device::Cpu)?.to_dtype(DType::F16)?)
    }

    const NAMES: [&str; 5] = [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ];
    const V_PROJ: &str = "model.layers.0.self_attn.v_proj.weight";

    /// A served file in miniature: v5, Tetra by default, five lattice records
    /// and one int4 `v_proj` (declared when `with_int4`), an f16 embedding, an
    /// f16 norm and two blobs.
    fn fixture(with_int4: bool) -> Vec<u8> {
        let tetra = Tetra::new();
        let mut rng = SplitMix64::new(0x1234_5678);
        let mut kinds = KindSet::of(CodeKind::Tetra);
        if with_int4 {
            kinds.insert(CodeKind::Int4G128);
        }
        let n = (NAMES.len() + usize::from(with_int4)) as u32;
        let mut buf = Vec::new();
        let mut w =
            format::ArtifactWriter::with_kinds(&mut buf, 5, n, CodeKind::Tetra, kinds).unwrap();
        for name in NAMES {
            w.push(&lattice(name, &mut rng, &tetra)).unwrap();
        }
        if with_int4 {
            let v = int4_record(V_PROJ, &checkpoint(V_PROJ).unwrap()).unwrap();
            w.push_int4(&v).unwrap();
        }
        let f16 = |name: &str, n: usize, rng: &mut SplitMix64| RawTensor {
            name: name.into(),
            dims: vec![n / 128, 128],
            data: RawData::F16(
                (0..n)
                    .map(|_| half::f16::from_f64(rng.next_f64() - 0.5).to_bits())
                    .collect(),
            ),
        };
        let raws = [
            f16("model.embed_tokens.weight", 1024, &mut rng),
            f16("model.norm.weight", 128, &mut rng),
        ];
        let blobs = [
            Blob {
                name: "config.json".into(),
                bytes: b"{\"tie_word_embeddings\":true}".to_vec(),
            },
            Blob {
                name: "tokenizer.json".into(),
                bytes: vec![7u8; 33],
            },
        ];
        w.seal(&raws, &blobs).unwrap();
        buf
    }

    fn run(src: &[u8], spec: &str) -> anyhow::Result<(Vec<u8>, Report)> {
        let spec = RestoreF16::parse(spec).map_err(anyhow::Error::msg)?;
        let mut out = Vec::new();
        let rep = transplant(
            &mut std::io::Cursor::new(src.to_vec()),
            &mut out,
            &spec,
            checkpoint,
        )?;
        Ok((out, rep))
    }

    /// Every record in order, then the raw section and the blobs as bytes.
    fn walk(bytes: &[u8]) -> (format::Header, Vec<Record>, Vec<u8>) {
        let mut r = std::io::Cursor::new(bytes.to_vec());
        let head = format::read_header(&mut r).unwrap();
        let recs = (0..head.matrices)
            .map(|_| format::read_record(&mut r, head.version).unwrap())
            .collect();
        let at = r.position() as usize;
        (head, recs, bytes[at..].to_vec())
    }

    fn record_bytes(version: u32, rec: &Record) -> Vec<u8> {
        let mut b = Vec::new();
        format::write_record(&mut b, version, rec).unwrap();
        b
    }

    #[test]
    fn covered_records_become_the_restore_paths_int4() {
        let src = fixture(true);
        let (out, rep) = run(&src, "o_proj,down_proj@1-1").unwrap();
        let (_, before, _) = walk(&src);
        let (_, after, _) = walk(&out);
        assert_eq!(before.len(), after.len());
        let replaced: Vec<&str> = rep.replaced.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(
            replaced,
            [NAMES[0], NAMES[2], NAMES[3]],
            "both o_proj, and down_proj in layer 1 only"
        );
        for (b, a) in before.iter().zip(&after) {
            assert_eq!(b.name(), a.name(), "record order moved");
            if replaced.contains(&a.name()) {
                let Record::Int4(q) = a else {
                    panic!("{} is not int4", a.name())
                };
                // What the dense arm restored: the same quantizer on the same
                // tensor, decoded.
                let want = llvq_llm::sealed::quantize_dequantize_q4(
                    &checkpoint(a.name()).unwrap(),
                    a.name(),
                    128,
                    DType::F32,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                assert_eq!(q.to_f32(), want, "{}", a.name());
            } else {
                assert_eq!(record_bytes(5, a), record_bytes(5, b), "{} moved", a.name());
            }
        }
        assert_eq!((rep.lattice_kept, rep.int4_kept), (2, 1));
        assert_eq!(rep.weights(), 3 * D_OUT * D_IN);
    }

    #[test]
    fn raws_and_blobs_are_copied_byte_for_byte() {
        let src = fixture(true);
        let (out, _) = run(&src, "o_proj").unwrap();
        assert_eq!(walk(&out).2, walk(&src).2);
    }

    #[test]
    fn a_header_that_declares_int4_is_unchanged() {
        let src = fixture(true);
        let (out, _) = run(&src, "o_proj").unwrap();
        let mut a = std::io::Cursor::new(src.clone());
        let mut b = std::io::Cursor::new(out.clone());
        let (ha, hb) = (
            format::read_header(&mut a).unwrap(),
            format::read_header(&mut b).unwrap(),
        );
        assert_eq!(a.position(), b.position());
        assert_eq!(src[..a.position() as usize], out[..b.position() as usize]);
        assert_eq!(ha.kinds(), hb.kinds());
    }

    #[test]
    fn a_tetra_only_header_gains_int4() {
        let src = fixture(false);
        let (out, _) = run(&src, "o_proj").unwrap();
        let (head, recs, _) = walk(&out);
        assert!(head.kinds().contains(CodeKind::Int4G128));
        assert!(head.kinds().contains(CodeKind::Tetra));
        assert_eq!(head.default_kind(), CodeKind::Tetra);
        assert!(recs.iter().any(|r| matches!(r, Record::Int4(_))));
    }

    #[test]
    fn a_type_that_matches_nothing_is_refused() {
        let src = fixture(true);
        let e = run(&src, "o_proj,gate_proj").unwrap_err().to_string();
        assert!(e.contains("gate_proj"), "{e}");
        let e = run(&src, "down_proj@5-9").unwrap_err().to_string();
        assert!(e.contains("down_proj"), "{e}");
    }

    #[test]
    fn an_int4_record_is_never_requantized() {
        let src = fixture(true);
        let e = run(&src, "v_proj").unwrap_err().to_string();
        assert!(e.contains("already an int4 record"), "{e}");
    }

    #[test]
    fn a_shape_the_checkpoint_disagrees_on_is_refused() {
        let src = fixture(true);
        let spec = RestoreF16::parse("o_proj").unwrap();
        let wrong = |_: &str| -> anyhow::Result<Tensor> {
            Ok(Tensor::zeros((D_OUT, 256), DType::F16, &Device::Cpu)?)
        };
        let e = transplant(
            &mut std::io::Cursor::new(src),
            &mut Vec::new(),
            &spec,
            wrong,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("not the same model"), "{e}");
    }

    #[test]
    fn a_file_without_kind_tags_is_refused() {
        let mut rng = SplitMix64::new(9);
        let mut buf = Vec::new();
        let w = format::ArtifactWriter::new(&mut buf, 0).unwrap();
        let raws = [RawTensor {
            name: "model.embed_tokens.weight".into(),
            dims: vec![2, 128],
            data: RawData::F16(
                (0..256)
                    .map(|_| half::f16::from_f64(rng.next_f64()).to_bits())
                    .collect(),
            ),
        }];
        w.seal(&raws, &[]).unwrap();
        let e = run(&buf, "o_proj").unwrap_err().to_string();
        assert!(e.contains("no kind tag"), "{e}");
    }
}
