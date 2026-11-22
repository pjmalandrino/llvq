//! The two vocabulary tables of the sealed file, read by the engine.
//!
//! The sealed Inkling 2 Mini holds its embedding and output tables in int4
//! with groups of 64 (`bin/inktable` writes them, `bin/inkseal` seals them
//! and `inkseal verify` checks the sealed copies equal these files). Without
//! this module the engine reads both tables from the checkpoint, in BF16:
//! the experts are the file's and the tables are not.
//!
//! [`load`] decodes the two files through `RawTensor::to_f32`, the reader's
//! one entry point, and gives the engine the file's own values:
//!
//!   - [`EmbedTable`]: the embedding rows, f32 on the host; a token's row is
//!     copied to the device at each call, as the checkpoint's row is today.
//!   - [`DenseHead`]: the output table, f32 on the device, behind the
//!     [`OutputProjection`] seam.
//!
//! Both are dense: this is the path of a quality measurement, where the
//! values must be the file's and the memory is not the object. The int4 head
//! that keeps the table packed on the device is [`super::head`].

use std::path::Path;

use anyhow::{bail, ensure, Context};
use candle_core::{Device, Tensor};

use super::config::InklingConfig;
use super::head::OutputProjection;
use super::layers;
use super::loader::NativeCheckpoint;

/// Name of the embedding table in its raw file.
pub const EMBED_NAME: &str = "model.embed.weight";
/// Name of the output table in its raw file.
pub const UNEMBED_NAME: &str = "model.unembed.weight";
/// File names `bin/inktable` writes, looked up in the directory given to [`load`].
pub const EMBED_FILE: &str = "embed.q4g64.llvqraw";
pub const UNEMBED_FILE: &str = "unembed.q4g64.llvqraw";

/// Output rows of one product: the size of the chunks `Inkling::head` uses.
const HEAD_CHUNK: usize = 16384;

/// Decode the first `rows` rows of the int4 g64 table in `path`, row-major.
///
/// Refuses a file that is not the table named `name`, not int4 in groups of
/// 64, not `hidden` wide, or shorter than `rows`.
pub fn decode_table(path: &Path, name: &str, rows: usize, hidden: usize) -> anyhow::Result<Vec<f32>> {
    let t = crate::inkseal::read_raw_file(path)?;
    ensure!(t.name == name, "{}: holds {:?}, expected {name:?}", path.display(), t.name);
    let llvq_artifact::RawData::Quant(q) = &t.data else {
        bail!("{}: an f16 table, the sealed tables are int4 g64", path.display());
    };
    ensure!(q.bits == 4 && q.group == 64, "{}: int{} in groups of {}, expected int4 g64", path.display(), q.bits, q.group);
    ensure!(
        t.dims.len() == 2 && t.dims[1] == hidden && t.dims[0] >= rows && rows > 0,
        "{}: dims {:?}, expected at least [{rows}, {hidden}]",
        path.display(),
        t.dims
    );
    let mut v = t.to_f32();
    v.truncate(rows * hidden);
    Ok(v)
}

/// The embedding rows of the sealed file, decoded, on the host.
pub struct EmbedTable {
    rows: Vec<f32>,
    vocab: usize,
    hidden: usize,
}

impl EmbedTable {
    pub fn from_rows(rows: Vec<f32>, vocab: usize, hidden: usize) -> anyhow::Result<Self> {
        ensure!(rows.len() == vocab * hidden, "{} values for [{vocab}, {hidden}]", rows.len());
        Ok(Self { rows, vocab, hidden })
    }

    /// The rows of `ids`, `[T, hidden]` f32 on `dev`: the table-side twin of
    /// `NativeCheckpoint::embed_rows`.
    pub fn rows(&self, ids: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            if id >= self.vocab {
                candle_core::bail!("token {id} outside the vocabulary of {}", self.vocab);
            }
            out.extend_from_slice(&self.rows[id * self.hidden..(id + 1) * self.hidden]);
        }
        Tensor::from_vec(out, (ids.len(), self.hidden), dev)
    }

    /// `layers::embed` with this table's rows: the row, then the RMS norm with
    /// the checkpoint's weight.
    pub fn embed(&self, ck: &NativeCheckpoint, cfg: &InklingConfig, ids: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
        let rows = self.rows(ids, dev)?;
        let w = ck.f32("embed_norm.weight", dev)?;
        layers::rms_norm(&rows, &w, cfg.rms_norm_eps)
    }
}

/// The output table of the sealed file, decoded, f32 on the device.
pub struct DenseHead {
    w: Tensor,
    vocab: usize,
    hidden: usize,
}

impl DenseHead {
    pub fn from_rows(rows: Vec<f32>, vocab: usize, hidden: usize, dev: &Device) -> anyhow::Result<Self> {
        ensure!(rows.len() == vocab * hidden, "{} values for [{vocab}, {hidden}]", rows.len());
        Ok(Self { w: Tensor::from_vec(rows, (vocab, hidden), dev)?, vocab, hidden })
    }
}

impl OutputProjection for DenseHead {
    fn name(&self) -> &str {
        "sealed int4 g64 table, decoded to f32"
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    /// The products `Inkling::head` makes, a chunk of rows at a time, on rows
    /// that are already f32.
    fn project(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        let mut parts = vec![];
        let mut start = 0;
        while start < self.vocab {
            let n = HEAD_CHUNK.min(self.vocab - start);
            parts.push(layers::linear(h, &self.w.narrow(0, start, n)?)?);
            start += n;
        }
        Tensor::cat(&parts, 1)
    }

    fn device_bytes(&self) -> u64 {
        (self.vocab * self.hidden * 4) as u64
    }
}

/// What [`load`] returns: the two tables and the sha256 of their files.
pub struct SealedTables {
    pub embed: EmbedTable,
    pub head: DenseHead,
    pub embed_sha256: String,
    pub unembed_sha256: String,
}

/// Read `dir/embed.q4g64.llvqraw` and `dir/unembed.q4g64.llvqraw`.
///
/// The embedding keeps the padded vocabulary (`vocab_size` rows), the head the
/// unpadded one, as the checkpoint's tables do in `Inkling::load`.
pub fn load(dir: &Path, cfg: &InklingConfig, dev: &Device) -> anyhow::Result<SealedTables> {
    let (e, u) = (dir.join(EMBED_FILE), dir.join(UNEMBED_FILE));
    let hidden = cfg.hidden_size;
    let embed = decode_table(&e, EMBED_NAME, cfg.vocab_size, hidden)?;
    let unembed = decode_table(&u, UNEMBED_NAME, cfg.unpadded_vocab_size, hidden)?;
    Ok(SealedTables {
        embed: EmbedTable::from_rows(embed, cfg.vocab_size, hidden)?,
        head: DenseHead::from_rows(unembed, cfg.unpadded_vocab_size, hidden, dev)?,
        embed_sha256: crate::digest::sha256_file(&e).with_context(|| format!("sha256 of {}", e.display()))?,
        unembed_sha256: crate::digest::sha256_file(&u).with_context(|| format!("sha256 of {}", u.display()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The head's chunked products equal one plain product, and the rows of
    /// the embedding table are the rows asked for.
    #[test]
    fn dense_head_and_embed_rows() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (HEAD_CHUNK + 37, 5);
        let rows: Vec<f32> = (0..vocab * hidden).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
        let head = DenseHead::from_rows(rows.clone(), vocab, hidden, &dev)?;
        let h = Tensor::from_vec(vec![0.5f32, -1.0, 0.25, 2.0, -0.125, 1.0, 0.0, -2.0, 0.75, 0.5], (2, hidden), &dev)?;
        let got = head.project(&h)?.to_vec2::<f32>()?;
        let hv = h.to_vec2::<f32>()?;
        for (t, hrow) in hv.iter().enumerate() {
            for v in [0, 1, HEAD_CHUNK - 1, HEAD_CHUNK, vocab - 1] {
                let want: f32 = (0..hidden).map(|j| hrow[j] * rows[v * hidden + j]).sum();
                assert!((got[t][v] - want).abs() <= 1e-5 * want.abs().max(1.0), "row {t} vocab {v}: {} vs {want}", got[t][v]);
            }
        }
        assert_eq!(got[0].len(), vocab);
        assert_eq!(head.device_bytes(), (vocab * hidden * 4) as u64);

        let table = EmbedTable::from_rows(rows.clone(), vocab, hidden)?;
        let r = table.rows(&[3, 0, (vocab - 1) as u32], &dev)?.to_vec2::<f32>()?;
        assert_eq!(r[0], rows[3 * hidden..4 * hidden]);
        assert_eq!(r[1], rows[..hidden]);
        assert_eq!(r[2], rows[(vocab - 1) * hidden..]);
        assert!(table.rows(&[vocab as u32], &dev).is_err());
        assert!(EmbedTable::from_rows(rows, vocab, hidden + 1).is_err());
        Ok(())
    }
}
