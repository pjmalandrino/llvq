//! Requantize the carried tensors of a sealed artifact — the embedding.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --bin embedq -- in.llvq out.llvq [q8|q4] [min_weights]`
//!
//! Reads a sealed file, copies every matrix record through **without decoding
//! it**, and rewrites every f16 raw tensor of at least `min_weights` weights
//! (default 1,000,000 — i.e. the embedding and nothing else; norms stay f16)
//! as group-affine int8 or int4, groups of 64, MLX's exact scheme.
//!
//! Needs no checkpoint, no cache, no network: it operates on the deliverable.
//! On Qwen3-4B, `q4` takes the embedding from 778 MB to 219 MB — the sealed
//! file from 1.77 GB to ~1.21 GB — with the quality change to be *measured*
//! (ppl + MMLU on the output file), not assumed.
//!
//! ## Any kind of file, by record
//!
//! The matrix section is walked with `read_record` and `write_record`, the
//! pair `rowscale` walks the served mixed file with, and the header is
//! rewritten from the file's own version, default kind and declared kinds. So
//! a v4 Ball file, a v5 Tetra file and the served v5 file whose 36 `v_proj`
//! records are int4 all come out with a header and a matrix section
//! byte-identical to the input's (`the_matrix_section_is_copied_byte_for_byte`).
//! Until 2026-09-23 this tool wrote a Ball header over whatever it read and
//! refused every kinded file by name, which is why the refusal could not
//! simply be lifted.

use llvq_artifact::{self as format, RawData};
use llvq_llm::embedquant::{bits_per_weight, quantize_affine};

const GROUP: usize = 64;

fn read_u32(r: &mut impl std::io::Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn put_u32(w: &mut impl std::io::Write, v: u32) -> anyhow::Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

/// What a pass did, so the caller can say it out loud.
#[derive(Default, Debug)]
struct Report {
    version: u32,
    kinds: String,
    lattice: u32,
    int4: u32,
    /// `(name, weights, bytes before, bytes after)` of each tensor rewritten.
    quantized: Vec<(String, usize, u64, u64)>,
    carried_before: u64,
    carried_after: u64,
}

/// Copy `r` to `w`, rewriting every f16 raw tensor of at least `min_weights`
/// weights at `bits`.
///
/// Readers and writers rather than paths, so the controls run on memory
/// buffers. The matrix records are never decoded: a lattice record goes back
/// out as the same codes, an int4 record as the same nibbles.
fn requantize(
    r: &mut impl std::io::Read,
    w: &mut impl std::io::Write,
    bits: u8,
    min_weights: usize,
) -> anyhow::Result<Report> {
    let head = format::read_header(r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "a projections-only artifact (format v{}) — seal it first",
        head.version
    );
    // The file's own version and kinds, not a writer default: a v5 record
    // carries a kind tag and a v4 one does not, so a passthrough that changed
    // either would have to rewrite every record it claims to copy.
    format::write_header_kinds(
        w,
        head.version,
        head.matrices,
        head.default_kind,
        head.kinds,
    )?;
    let mut rep = Report {
        version: head.version,
        kinds: head.kinds().to_string(),
        ..Report::default()
    };
    for _ in 0..head.matrices {
        let rec = format::read_record(r, head.version)?;
        match &rec {
            format::Record::Lattice(_) => rep.lattice += 1,
            format::Record::Int4(_) => rep.int4 += 1,
        }
        format::write_record(w, head.version, &rec)?;
    }

    let n_raw = read_u32(r)?;
    put_u32(w, n_raw)?;
    for _ in 0..n_raw {
        let t = format::read_raw(r, head.version)?;
        let before = t.bytes();
        let quantize = matches!(t.data, RawData::F16(_)) && t.len() >= min_weights;
        let t = if quantize {
            let q = quantize_affine(&t, bits, GROUP)?;
            rep.quantized
                .push((q.name.clone(), q.len(), before, q.bytes()));
            q
        } else {
            t
        };
        rep.carried_before += before;
        rep.carried_after += t.bytes();
        format::write_raw(w, &t)?;
    }

    // The blobs (config, tokenizer) are not modelled here, so they are copied
    // as bytes, like `rowscale` copies everything after its last record.
    std::io::copy(r, w)?;
    Ok(rep)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst) = match (a.first(), a.get(1)) {
        (Some(s), Some(d)) => (s.clone(), d.clone()),
        _ => anyhow::bail!("usage: embedq <in.llvq> <out.llvq> [q8|q4] [min_weights]"),
    };
    let bits: u8 = match a.get(2).map(String::as_str).unwrap_or("q8") {
        "q8" => 8,
        "q4" => 4,
        other => anyhow::bail!("unknown mode {other:?} — expected q8 or q4"),
    };
    let min_weights: usize = a
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1_000_000);

    let mut r = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&src)?);
    let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&dst)?);
    let rep =
        requantize(&mut r, &mut w, bits, min_weights).map_err(|e| anyhow::anyhow!("{src}: {e}"))?;
    use std::io::Write as _;
    w.flush()?;

    eprintln!(
        "{src}: format v{}, kinds {}, {} lattice + {} int4 records passed through undecoded",
        rep.version, rep.kinds, rep.lattice, rep.int4
    );
    for (name, n, b0, b1) in &rep.quantized {
        eprintln!(
            "  {name}: {n} weights, f16 → int{bits} g{GROUP} ({:.2} b/w), {:.1} → {:.1} MB",
            bits_per_weight(bits, GROUP),
            *b0 as f64 / 1e6,
            *b1 as f64 / 1e6,
        );
    }
    anyhow::ensure!(
        !rep.quantized.is_empty(),
        "{src}: no f16 tensor of at least {min_weights} weights — nothing was requantized \
         and {dst} is a copy"
    );

    let (src_b, dst_b) = (
        std::fs::metadata(&src)?.len(),
        std::fs::metadata(&dst)?.len(),
    );
    println!("\n── {dst}");
    println!(
        "   carried    {:.3} GB → {:.3} GB",
        rep.carried_before as f64 / 1e9,
        rep.carried_after as f64 / 1e9
    );
    println!(
        "   file       {:.3} GB → {:.3} GB",
        src_b as f64 / 1e9,
        dst_b as f64 / 1e9
    );
    println!("\n   score the OUTPUT file (ppl + mmlu) before believing anything.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_artifact::{
        Blob, CodeKind, Int4Matrix, KindSet, QuantizedMatrix, RawTensor, TETRA_SHELL_CAP,
    };
    use llvq_core::{SplitMix64, DIM};
    use llvq_quant::quantizer::BlockCode;
    use llvq_search::tetra::{Tetra, LABEL_MASK};

    const EMBED: &str = "model.embed_tokens.weight";
    const NORM: &str = "model.norm.weight";

    fn f16_tensor(name: &str, dims: Vec<usize>, rng: &mut SplitMix64) -> RawTensor {
        let n: usize = dims.iter().product();
        let data = (0..n)
            .map(|_| half::f16::from_f64(rng.next_f64() - 0.5).to_bits())
            .collect();
        RawTensor {
            name: name.to_string(),
            dims,
            data: RawData::F16(data),
        }
    }

    /// The served file in miniature: a v5 header declaring Tetra and int4, a
    /// Tetra record, an int4 record, an f16 "embedding" of 1,024 weights, an
    /// f16 "norm" of 64, and two blobs. Returns the whole file and the length
    /// of its header plus matrix section.
    fn mixed_fixture() -> (Vec<u8>, usize) {
        let tetra = Tetra::new();
        let mut rng = SplitMix64::new(0xE3BE_D05E_ED00);
        let mut kinds = KindSet::of(CodeKind::Tetra);
        kinds.insert(CodeKind::Int4G128);

        let (d_out, d_in) = (4usize, 2 * DIM);
        let lattice = QuantizedMatrix {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            d_out,
            d_in,
            codes: (0..d_out * (d_in / DIM))
                .map(|_| BlockCode {
                    point: tetra.decode(rng.next() & LABEL_MASK),
                    gain: (rng.next() & 1) as u32,
                })
                .collect(),
            row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
            centroids: vec![0.7, 1.1],
            rotation_seed: Some(0xABCD),
            shell_cap: TETRA_SHELL_CAP,
            tail: Vec::new(),
        };
        let (i_out, i_in) = (2usize, 256usize);
        let groups = i_out * i_in / 128;
        let int4 = Int4Matrix {
            name: "model.layers.0.self_attn.v_proj.weight".to_string(),
            d_out: i_out,
            d_in: i_in,
            bits: 4,
            group: 128,
            packed: (0..i_out * i_in / 2).map(|_| rng.next() as u8).collect(),
            scales: (0..groups)
                .map(|_| half::f16::from_f64(0.01 + rng.next_f64()).to_bits())
                .collect(),
            biases: (0..groups)
                .map(|_| half::f16::from_f64(rng.next_f64() - 0.5).to_bits())
                .collect(),
        };
        let raws = [
            f16_tensor(EMBED, vec![8, 128], &mut rng),
            f16_tensor(NORM, vec![64], &mut rng),
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

        let write = |seal: bool| {
            let mut buf: Vec<u8> = Vec::new();
            let mut w =
                format::ArtifactWriter::with_kinds(&mut buf, 5, 2, CodeKind::Tetra, kinds).unwrap();
            w.push(&lattice).unwrap();
            w.push_int4(&int4).unwrap();
            if seal {
                w.seal(&raws, &blobs).unwrap();
            } else {
                w.finish().unwrap();
            }
            buf
        };
        // A projections-only close adds two zero counts after the records.
        let matrix_end = write(false).len() - 8;
        (write(true), matrix_end)
    }

    fn run(src: &[u8], bits: u8, min_weights: usize) -> (Vec<u8>, Report) {
        let mut r = std::io::Cursor::new(src.to_vec());
        let mut out = Vec::new();
        let rep = requantize(&mut r, &mut out, bits, min_weights).expect("requantize");
        (out, rep)
    }

    /// Walk a file back: its header, its records, its raws, its blobs.
    fn read_back(bytes: &[u8]) -> (format::Header, Vec<RawTensor>, Vec<Blob>) {
        let mut r = std::io::Cursor::new(bytes.to_vec());
        let head = format::read_header(&mut r).unwrap();
        for _ in 0..head.matrices {
            format::read_record(&mut r, head.version).unwrap();
        }
        let raws = (0..read_u32(&mut r).unwrap())
            .map(|_| format::read_raw(&mut r, head.version).unwrap())
            .collect();
        let blobs = (0..read_u32(&mut r).unwrap())
            .map(|_| format::read_blob(&mut r).unwrap())
            .collect();
        assert_eq!(r.position() as usize, bytes.len(), "trailing bytes");
        (head, raws, blobs)
    }

    #[test]
    fn the_matrix_section_is_copied_byte_for_byte() {
        let (src, matrix_end) = mixed_fixture();
        let (out, rep) = run(&src, 4, 1000);
        assert_eq!(
            out[..matrix_end],
            src[..matrix_end],
            "header or a record moved"
        );
        assert_eq!((rep.lattice, rep.int4), (1, 1));
        assert_eq!(rep.version, 5);
    }

    #[test]
    fn the_mixed_header_keeps_its_kinds() {
        let (src, _) = mixed_fixture();
        let (out, _) = run(&src, 4, 1000);
        let (a, _, _) = read_back(&src);
        let (b, _, _) = read_back(&out);
        assert_eq!(b.version, a.version);
        assert_eq!(b.default_kind, CodeKind::Tetra);
        assert_eq!(b.kinds, a.kinds);
        assert!(b.kinds.contains(CodeKind::Int4G128));
    }

    #[test]
    fn the_embedding_goes_to_int4_g64_and_nothing_else_moves() {
        let (src, _) = mixed_fixture();
        let (out, rep) = run(&src, 4, 1000);
        let (_, raws_in, blobs_in) = read_back(&src);
        let (_, raws_out, blobs_out) = read_back(&out);

        assert_eq!(rep.quantized.len(), 1);
        assert_eq!(rep.quantized[0].0, EMBED);
        let e = raws_out.iter().find(|t| t.name == EMBED).unwrap();
        let RawData::Quant(q) = &e.data else {
            panic!("the embedding is still f16");
        };
        assert_eq!((q.bits, q.group), (4, GROUP));
        assert_eq!(e.dims, vec![8, 128]);
        // What the file stores is what the shipped quantizer returns, and
        // nothing of its own.
        let want = quantize_affine(&raws_in[0], 4, GROUP).unwrap();
        assert_eq!(e.to_f32(), want.to_f32());

        let n_in = raws_in.iter().find(|t| t.name == NORM).unwrap();
        let n_out = raws_out.iter().find(|t| t.name == NORM).unwrap();
        let (RawData::F16(a), RawData::F16(b)) = (&n_in.data, &n_out.data) else {
            panic!("the norm left f16");
        };
        assert_eq!(a, b);

        assert_eq!(blobs_in.len(), blobs_out.len());
        for (a, b) in blobs_in.iter().zip(&blobs_out) {
            assert_eq!((&a.name, &a.bytes), (&b.name, &b.bytes));
        }
    }

    #[test]
    fn q8_is_the_same_walk_at_eight_bits() {
        let (src, matrix_end) = mixed_fixture();
        let (out, _) = run(&src, 8, 1000);
        assert_eq!(out[..matrix_end], src[..matrix_end]);
        let (_, raws, _) = read_back(&out);
        let RawData::Quant(q) = &raws[0].data else {
            panic!("the embedding is still f16");
        };
        assert_eq!((q.bits, q.group), (8, GROUP));
    }

    #[test]
    fn a_threshold_above_every_tensor_reproduces_the_file() {
        let (src, _) = mixed_fixture();
        let (out, rep) = run(&src, 4, usize::MAX);
        assert_eq!(out, src, "nothing to requantize must mean a byte copy");
        assert!(rep.quantized.is_empty());
    }

    #[test]
    fn an_already_quantized_embedding_is_not_requantized() {
        let (src, _) = mixed_fixture();
        let (once, _) = run(&src, 4, 1000);
        let (twice, rep) = run(&once, 8, 1000);
        assert_eq!(twice, once, "a quantized tensor is carried, never stacked");
        assert!(rep.quantized.is_empty());
    }

    #[test]
    fn a_v4_ball_file_still_passes() {
        let mut rng = SplitMix64::new(0xBA11);
        let mut buf = Vec::new();
        {
            let w = format::ArtifactWriter::new(&mut buf, 0).unwrap();
            let raws = [f16_tensor(EMBED, vec![4, 128], &mut rng)];
            w.seal(&raws, &[]).unwrap();
        }
        let (out, rep) = run(&buf, 4, 100);
        assert_eq!(rep.version, 4);
        assert_eq!(rep.quantized.len(), 1);
        let (head, raws, _) = read_back(&out);
        assert_eq!(head.version, 4);
        assert!(head.is_ball_only());
        assert!(matches!(&raws[0].data, RawData::Quant(q) if q.bits == 4));
    }

    #[test]
    fn a_projections_only_file_is_refused() {
        let mut buf = Vec::new();
        format::write_header(&mut buf, 1, 0).unwrap();
        put_u32(&mut buf, 0).unwrap();
        put_u32(&mut buf, 0).unwrap();
        let mut out = Vec::new();
        let e = requantize(&mut std::io::Cursor::new(buf), &mut out, 4, 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("seal it first"), "{e}");
    }
}
