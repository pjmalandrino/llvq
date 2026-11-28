//! The host half of a resident expert layer, on every platform: read the
//! three records of an expert from `layers/Lxx/codes/eNNN.llvqrec`, check them
//! against the slot `extq` writes, and transcode each to the served `Tetra48`
//! stream as a [`FusedMatrix`], the struct `fused_cuda` uploads.
//!
//! The checks are those of `resident::ResidentExperts::load` (Metal): the
//! `VERIFIED` file, three records and no trailing byte, the record name, the
//! Tetra kind, the shape, one gain bit, the rotation seed. The matrix is built
//! the way `crate::fused::load` builds one from a sealed file (`fused/load.rs`, the
//! `FusedMatrix { .. }` of its read loop): same `nblocks`, `tail_w`, rotation
//! key `(d_in, seed)`, gain centroids, row scales, binary16 tail and byte
//! count. It lives apart from `resident_cuda` so that it compiles, and is
//! tested, on a machine without a card.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use super::config::InklingConfig;
use crate::fused::{matrix_side_bytes, tail_f16_bits, FusedLayout, FusedMatrix, Transcoder};

/// Names of the three records of an expert, in file order.
pub const MATS: [&str; 3] = ["gate", "up", "down"];

/// The rotation seed `extq` gives an expert's input group: group 0 is the
/// input of the fused `w13` (gate and up), group 1 the input of `down`
/// (`W/llvq-llm/src/bin/extq.rs`, `fn seed`; the same formula as
/// `resident::rotation_seed`, which a test holds this one against).
pub fn rotation_seed(layer: usize, expert: usize, group: u64) -> u64 {
    crate::calib::ROTATION_SEED ^ ((layer as u64) << 32) ^ ((expert as u64) << 20) ^ (group << 16)
}

/// The record name `extq` writes.
pub fn record_name(layer: usize, expert: usize, mat: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{mat}_proj.weight")
}

/// The directory of layer `layer` under `layers_dir` (`D/layers`).
pub fn layer_dir(layers_dir: &Path, layer: usize) -> PathBuf {
    layers_dir.join(format!("L{layer:02}"))
}

/// The layers of `layers_dir` whose `Lxx/VERIFIED` exists, ascending.
pub fn verified_layers(layers_dir: &Path, n_layers: usize) -> Vec<usize> {
    (0..n_layers).filter(|&l| layer_dir(layers_dir, l).join("VERIFIED").exists()).collect()
}

/// The transcoder of the served layout for Tetra records.
pub fn tetra48_transcoder() -> Result<Transcoder> {
    Transcoder::for_kind(FusedLayout::Tetra48, llvq_artifact::CodeKind::Tetra).map_err(anyhow::Error::msg)
}

/// The three matrices of one expert, ready to upload, and what reading them cost.
pub struct ExpertMatrices {
    pub gate: FusedMatrix,
    pub up: FusedMatrix,
    pub down: FusedMatrix,
    /// Bytes of the codes file.
    pub file_bytes: u64,
    pub seconds_read: f64,
    pub seconds_transcode: f64,
}

/// `(d_out, d_in, rotation group)` of gate, up and down.
pub fn mat_shapes(cfg: &InklingConfig) -> [(usize, usize, u64); 3] {
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    [(i, h, 0), (i, h, 0), (h, i, 1)]
}

/// Read, check and transcode expert `e` of `layer` from `dir` (`D/layers/Lxx`).
///
/// The caller checks `VERIFIED` once per layer ([`check_layer`]); this reads
/// one file and refuses anything that is not the slot `extq` writes.
pub fn read_expert(tr: &Transcoder, dir: &Path, layer: usize, e: usize, cfg: &InklingConfig) -> Result<ExpertMatrices> {
    let path = dir.join("codes").join(format!("e{e:03}.llvqrec"));
    let t = std::time::Instant::now();
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let mut r = &bytes[..];
    let mut recs = Vec::with_capacity(3);
    for _ in MATS {
        let rec = llvq_artifact::read_record(&mut r, llvq_artifact::VERSION)
            .map_err(|err| anyhow::anyhow!("{}: {err}", path.display()))?;
        recs.push(rec);
    }
    ensure!(r.is_empty(), "{}: {} trailing bytes after three records", path.display(), r.len());
    let seconds_read = t.elapsed().as_secs_f64();

    let t = std::time::Instant::now();
    let mut out = Vec::with_capacity(3);
    for (rec, (mat, (d_out, d_in, group))) in recs.into_iter().zip(MATS.iter().zip(mat_shapes(cfg))) {
        let llvq_artifact::Record::Lattice(m) = rec else {
            bail!("{}: {mat} is not a lattice record", path.display());
        };
        let want = record_name(layer, e, mat);
        ensure!(m.name == want, "{}: record {:?} where {want:?} belongs", path.display(), m.name);
        ensure!(m.kind == llvq_artifact::CodeKind::Tetra, "{want}: kind {:?}, the tetra48 layout reads Tetra", m.kind);
        ensure!((m.d_out, m.d_in) == (d_out, d_in), "{want}: {}x{}, expected {d_out}x{d_in}", m.d_out, m.d_in);
        ensure!(m.centroids.len() == 2, "{want}: {} centroids, the kernel reads one gain bit", m.centroids.len());
        let seed = rotation_seed(layer, e, group);
        ensure!(m.rotation_seed == Some(seed), "{want}: rotation seed {:?}, extq gives {seed}", m.rotation_seed);
        let nblocks = m.d_in / llvq_core::DIM;
        let tail_w = m.d_in % llvq_core::DIM;
        let (stream, payload) =
            tr.stream(&m.indices, &m.gains, m.d_out, nblocks).map_err(|err| anyhow::anyhow!("{want}: {err}"))?;
        out.push(FusedMatrix {
            name: m.name,
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks,
            tail_w,
            stream,
            gscale: [m.centroids[0] as f32, m.centroids[1] as f32],
            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
            tail: tail_f16_bits(&m.tail),
            rotation: Some((m.d_in, seed)),
            bytes: payload + matrix_side_bytes(m.d_out, tail_w),
        });
    }
    let seconds_transcode = t.elapsed().as_secs_f64();
    let down = out.pop().expect("three records");
    let up = out.pop().expect("three records");
    let gate = out.pop().expect("three records");
    Ok(ExpertMatrices { gate, up, down, file_bytes: bytes.len() as u64, seconds_read, seconds_transcode })
}

/// Refuse a layer directory without its `VERIFIED` file.
pub fn check_layer(dir: &Path) -> Result<()> {
    ensure!(dir.join("VERIFIED").exists(), "{}: no VERIFIED file, the layer is not complete", dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::HostStream;

    fn data_dir() -> PathBuf {
        match std::env::var("INKLING_DATA") {
            Ok(d) => PathBuf::from(d),
            Err(_) => PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini"),
        }
    }

    /// The seed formula is the Metal module's, bit for bit, wherever both compile.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn seed_matches_the_metal_module() {
        for (l, e, g) in [(0, 0, 0), (6, 77, 1), (23, 127, 0), (17, 5, 1)] {
            assert_eq!(rotation_seed(l, e, g), super::super::resident::rotation_seed(l, e, g));
            assert_eq!(record_name(l, e, "down"), super::super::resident::record_name(l, e, "down"));
        }
    }

    /// One real expert of layer 6: three Tetra48 matrices of the shapes the
    /// config gives, keyed on the rotations `extq` wrote, and a wrong layer
    /// number refused by name. Needs `D/layers/L06` and `D/textonly/config.json`
    /// and fails naming them if they are missing.
    #[test]
    #[ignore]
    fn reads_one_real_expert_of_layer_6() -> Result<()> {
        let d = data_dir();
        let config = d.join("textonly/config.json");
        let dir = layer_dir(&d.join("layers"), 6);
        assert!(config.exists(), "missing {}", config.display());
        assert!(dir.join("VERIFIED").exists(), "missing {}", dir.join("VERIFIED").display());
        let cfg = InklingConfig::from_file(&config)?;
        let tr = tetra48_transcoder()?;
        check_layer(&dir)?;
        let em = read_expert(&tr, &dir, 6, 3, &cfg)?;
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        for (m, (d_out, d_in, g)) in [&em.gate, &em.up, &em.down].into_iter().zip(mat_shapes(&cfg)) {
            assert_eq!((m.d_out, m.d_in, m.nblocks, m.tail_w), (d_out, d_in, d_in / 24, 0), "{}", m.name);
            assert_eq!(m.rotation, Some((d_in, rotation_seed(6, 3, g))), "{}", m.name);
            assert_eq!(m.rscale.len(), d_out);
            let HostStream::Tetra48 { words, stride_u32 } = &m.stream else { panic!("{}: not Tetra48", m.name) };
            // the rows, plus the one guard word the transcoder appends for the last read window
            assert_eq!(words.len(), d_out * *stride_u32 as usize + 1, "{}", m.name);
            assert!(*stride_u32 as usize * 4 >= 6 * m.nblocks, "{}", m.name);
        }
        assert_eq!((em.gate.d_out, em.down.d_out), (i, h));
        // the same file read as layer 7's: the names and seeds are layer 6's, so refused
        let err = read_expert(&tr, &dir, 7, 3, &cfg).err().expect("layer 6's codes read as layer 7's");
        assert!(err.to_string().contains("where \"model.layers.7.mlp.experts.3.gate_proj.weight\" belongs"), "{err}");
        Ok(())
    }
}
