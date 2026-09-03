//! Requantize the carried tensors of a sealed artifact — the embedding.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --bin embedq -- in.llvq out.llvq [q8|q4] [min_weights]`
//!
//! Reads a sealed file, copies the quantized matrices through **without
//! decoding them** (`write_matrix_raw` — byte-identical, pinned by test), and
//! rewrites every f16 raw tensor of at least `min_weights` weights (default
//! 1,000,000 — i.e. the embedding and nothing else; norms stay f16) as
//! group-affine int8 or int4, groups of 64, MLX's exact scheme.
//!
//! Needs no checkpoint, no cache, no network: it operates on the deliverable.
//! On Qwen3-4B, `q4` takes the embedding from 778 MB to 219 MB — the sealed
//! file from 1.77 GB to ~1.21 GB — with the quality change to be *measured*
//! (ppl + MMLU on the output file), not assumed.

use llvq_artifact::{ArtifactWriter, RawData};
use llvq_llm::embedquant::{bits_per_weight, quantize_affine};

const GROUP: usize = 64;

fn read_u32(r: &mut impl std::io::Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
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

    let f = std::fs::File::open(&src)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{src} is a projections-only artifact — seal it first"
    );
    eprintln!(
        "{src}: format v{}, {} matrices — passing them through undecoded",
        head.version, head.matrices
    );

    let out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&dst)?);
    let mut w = ArtifactWriter::new(out, head.matrices)?;
    for i in 0..head.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r)?;
        w.push_raw(&m)?;
        if i % 36 == 0 {
            eprintln!("  {i:>3}/{}", head.matrices);
        }
    }

    let n_raw = read_u32(&mut r)?;
    let mut raws = Vec::with_capacity(n_raw as usize);
    let mut before = 0u64;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        before += t.bytes();
        let quantize = matches!(t.data, RawData::F16(_)) && t.len() >= min_weights;
        let t = if quantize {
            let q = quantize_affine(&t, bits, GROUP)?;
            eprintln!(
                "  {}: {} weights, f16 → int{bits} g{GROUP} ({:.2} b/w), {:.1} → {:.1} MB",
                q.name,
                q.len(),
                bits_per_weight(bits, GROUP),
                t.bytes() as f64 / 1e6,
                q.bytes() as f64 / 1e6,
            );
            q
        } else {
            t
        };
        raws.push(t);
    }
    let after: u64 = raws.iter().map(|t| t.bytes()).sum();

    let n_blob = read_u32(&mut r)?;
    let mut blobs = Vec::with_capacity(n_blob as usize);
    for _ in 0..n_blob {
        blobs.push(llvq_artifact::read_blob(&mut r)?);
    }
    w.seal(&raws, &blobs)?;

    let (src_b, dst_b) = (
        std::fs::metadata(&src)?.len(),
        std::fs::metadata(&dst)?.len(),
    );
    println!("\n── {dst}");
    println!(
        "   carried    {:.3} GB → {:.3} GB",
        before as f64 / 1e9,
        after as f64 / 1e9
    );
    println!(
        "   file       {:.3} GB → {:.3} GB",
        src_b as f64 / 1e9,
        dst_b as f64 / 1e9
    );
    println!("\n   score the OUTPUT file (ppl + mmlu) before believing anything.");
    Ok(())
}
