//! Rescales the fitted gain centroids of an artifact, matrix by matrix.
//!
//! ```text
//! cargo run --release -p llvq-llm --bin artscale -- \
//!   in.llvq out.llvq map.csv 0.04
//! ```
//!
//! ## Why this exists rather than a re-encode
//!
//! A Tetra block reconstructs as `centroids[g] · row_scale · u`, so multiplying
//! a matrix's centroids by `s` multiplies every block it holds by `s` — exactly
//! the post-hoc correction `errmap` measures, and nothing else. No code moves,
//! no row scale moves, no byte of the index changes, and the rate is untouched.
//!
//! That equivalence is why the correction costs no bit: the centroids are
//! already stored per matrix, so a corrected model is the same file with 2
//! numbers changed per matrix.
//!
//! Re-encoding with scaled centroids is a DIFFERENT operation — it changes
//! which level each block picks and what later columns are compensated against
//! — and it was measured to be worth something else entirely
//! (`docs/mesures/gain-scale-0.6b-2026-09-15.txt`). This binary does the first,
//! not the second.
//!
//! The map is an `errmap` CSV; the fourth argument is the trust half-width the
//! optimum is read at, and it must be the one the map was scored with.

use anyhow::Context;
use llvq_artifact as format;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};

/// The artifact's own key for a record, through the artifact's own splitter.
///
/// Reconstructing the name here instead cost a silent no-op on the first run:
/// records are `model.layers.<n>.<proj>.weight` and the guess was
/// `blocks.<n>.<proj>`, so every lookup missed and `artscale` reported "252
/// untouched" while writing a byte-identical copy. `split_name` is what the
/// reader uses, so it cannot drift from the file.
fn key_of(name: &str) -> anyhow::Result<(usize, String)> {
    let (layer, proj) = llvq_artifact::split_name(name)
        .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    Ok((layer, proj))
}

/// One scale per matrix, read from the map at the given trust half-width.
fn scales(path: &str, trust: f64) -> anyhow::Result<HashMap<(usize, String), f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().context("empty map")?;
    anyhow::ensure!(
        header.starts_with("layer,projection,gradient,curvature"),
        "{path}: not an errmap CSV"
    );
    let mut out = HashMap::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(f.len() >= 4, "{path}: line {} is short", n + 2);
        let layer: usize = f[0].parse()?;
        let (g, h): (f64, f64) = (f[2].parse()?, f[3].parse()?);
        // The same rule as `Sensitivity::optimum_within`, deliberately
        // duplicated in four lines rather than linked against: this binary
        // must keep reading old maps even as the library's default moves.
        let d = if h > 0.0 {
            (-g / h).clamp(-trust, trust)
        } else if g > 0.0 {
            -trust
        } else if g < 0.0 {
            trust
        } else {
            0.0
        };
        out.insert((layer, f[1].to_string()), 1.0 + d);
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() == 4,
        "usage: artscale <in.llvq> <out.llvq> <map.csv> <trust>"
    );
    let trust: f64 = a[3].parse().context("trust must be a number")?;
    anyhow::ensure!(
        trust.is_finite() && trust > 0.0 && trust <= 0.5,
        "trust {trust} is outside (0, 0.5]"
    );
    let by_name = scales(&a[2], trust)?;

    let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(&a[0])?);
    let head = format::read_header(&mut r)?;
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(&a[1])?);
    format::write_header_kinds(&mut w, head.version, head.matrices, head.default_kind, head.kinds)?;

    let (mut scaled, mut untouched, mut int4) = (0usize, 0usize, 0usize);
    let mut worst: f64 = 0.0;
    for _ in 0..head.matrices {
        let mut rec = format::read_record(&mut r, head.version)?;
        match &mut rec {
            format::Record::Lattice(m) => match by_name.get(&key_of(&m.name)?) {
                // 1.0 is skipped rather than multiplied: a matrix the map does
                // not move must come out byte for byte identical, so that a
                // diff of the two files shows exactly what was corrected.
                Some(&s) if s != 1.0 => {
                    for c in m.centroids.iter_mut() {
                        *c *= s;
                    }
                    worst = worst.max((s - 1.0).abs());
                    scaled += 1;
                }
                _ => untouched += 1,
            },
            format::Record::Int4(_) => int4 += 1,
        }
        format::write_record(&mut w, head.version, &rec)?;
    }
    println!("{} matrices: {scaled} rescaled, {untouched} untouched, {int4} int4", head.matrices);
    anyhow::ensure!(
        scaled > 0,
        "no matrix matched the map: {} entries were read and none of the {} records \
         share a (layer, projection) key with them",
        by_name.len(),
        head.matrices
    );
    println!("largest correction applied: {:.4} %", 100.0 * worst);
    println!("written to {}", a[1]);
    Ok(())
}
