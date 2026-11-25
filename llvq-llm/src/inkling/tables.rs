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
//!
//! [`load_embed`] reads the embedding alone, for a stack whose head is the
//! int4 one: dense rows for the GPU stacks, or the int4 table itself for the
//! CPU, each row decoded when a token asks for it. A decoded row is the value
//! `to_f32` gives, bit for bit.

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

/// How an [`EmbedTable`] holds its rows.
enum Rows {
    /// Decoded once at load: 4 bytes a value.
    Dense(Vec<f32>),
    /// The int4 table as the file stores it, a row decoded at each call.
    Q4(crate::tetra_cpu::Q4Matrix),
}

/// The embedding rows of the sealed file, on the host.
pub struct EmbedTable {
    rows: Rows,
    vocab: usize,
    hidden: usize,
}

impl EmbedTable {
    pub fn from_rows(rows: Vec<f32>, vocab: usize, hidden: usize) -> anyhow::Result<Self> {
        ensure!(rows.len() == vocab * hidden, "{} values for [{vocab}, {hidden}]", rows.len());
        Ok(Self { rows: Rows::Dense(rows), vocab, hidden })
    }

    /// The table kept in int4: `m` holds exactly `vocab` rows of `hidden`.
    pub fn from_q4(m: crate::tetra_cpu::Q4Matrix) -> anyhow::Result<Self> {
        m.check().map_err(anyhow::Error::msg)?;
        let (vocab, hidden) = (m.d_out, m.d_in);
        Ok(Self { rows: Rows::Q4(m), vocab, hidden })
    }

    /// Bytes the table holds on the host.
    pub fn bytes(&self) -> u64 {
        match &self.rows {
            Rows::Dense(v) => 4 * v.len() as u64,
            Rows::Q4(m) => m.bytes(),
        }
    }

    /// The rows of `ids`, `[T, hidden]` f32 on `dev`: the table-side twin of
    /// `NativeCheckpoint::embed_rows`.
    pub fn rows(&self, ids: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
        let mut out = vec![0f32; ids.len() * self.hidden];
        for (&id, o) in ids.iter().zip(out.chunks_mut(self.hidden)) {
            let id = id as usize;
            if id >= self.vocab {
                candle_core::bail!("token {id} outside the vocabulary of {}", self.vocab);
            }
            match &self.rows {
                Rows::Dense(v) => o.copy_from_slice(&v[id * self.hidden..(id + 1) * self.hidden]),
                Rows::Q4(m) => m.decode_row(id, o),
            }
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

/// The first `rows` rows of the int4 g64 table in `path`, kept in int4.
///
/// The same refusals as [`decode_table`].
pub fn q4_table(path: &Path, name: &str, rows: usize, hidden: usize) -> anyhow::Result<crate::tetra_cpu::Q4Matrix> {
    let t = crate::inkseal::read_raw_file(path)?;
    q4_rows(&t, name, rows, hidden).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// The first `rows` rows of the int4 g64 table `t`, sliced to exact sizes.
pub fn q4_rows(t: &llvq_artifact::RawTensor, name: &str, rows: usize, hidden: usize) -> anyhow::Result<crate::tetra_cpu::Q4Matrix> {
    ensure!(t.name == name, "holds {:?}, expected {name:?}", t.name);
    let llvq_artifact::RawData::Quant(q) = &t.data else {
        bail!("an f16 table, the sealed tables are int4 g64");
    };
    ensure!(q.bits == 4 && q.group == 64, "int{} in groups of {}, expected int4 g64", q.bits, q.group);
    ensure!(
        t.dims.len() == 2 && t.dims[1] == hidden && t.dims[0] >= rows && rows > 0 && hidden.is_multiple_of(64),
        "dims {:?}, expected at least [{rows}, {hidden}] with a row a multiple of 64",
        t.dims
    );
    let gpr = hidden / 64;
    let (nbytes, ngroups) = (rows * hidden / 2, rows * gpr);
    ensure!(
        q.packed.len() >= nbytes && q.scales.len() >= ngroups && q.biases.len() >= ngroups,
        "payload shorter than {rows} rows of {hidden}"
    );
    let m = crate::tetra_cpu::Q4Matrix {
        d_out: rows,
        d_in: hidden,
        group: 64,
        packed: q.packed[..nbytes].to_vec(),
        scales: q.scales[..ngroups].to_vec(),
        biases: q.biases[..ngroups].to_vec(),
    };
    m.check().map_err(anyhow::Error::msg)?;
    Ok(m)
}

/// What [`load_embed`] returns: the embedding and the sha256 of both files.
pub struct SealedEmbed {
    pub embed: EmbedTable,
    pub embed_sha256: String,
    pub unembed_sha256: String,
}

/// Read `dir/embed.q4g64.llvqraw` alone, for a stack whose head reads
/// `dir/unembed.q4g64.llvqraw` itself: no dense head is built. `packed`
/// keeps the table in int4 (the CPU stack, 0.56 bytes a value); otherwise
/// the rows are decoded once, as [`load`] decodes them. The unembedding
/// file is hashed for the load's report, not read into memory.
pub fn load_embed(dir: &Path, cfg: &InklingConfig, packed: bool) -> anyhow::Result<SealedEmbed> {
    let (e, u) = (dir.join(EMBED_FILE), dir.join(UNEMBED_FILE));
    let hidden = cfg.hidden_size;
    let embed = match packed {
        true => EmbedTable::from_q4(q4_table(&e, EMBED_NAME, cfg.vocab_size, hidden)?)?,
        false => EmbedTable::from_rows(decode_table(&e, EMBED_NAME, cfg.vocab_size, hidden)?, cfg.vocab_size, hidden)?,
    };
    Ok(SealedEmbed {
        embed,
        embed_sha256: crate::digest::sha256_file(&e).with_context(|| format!("sha256 of {}", e.display()))?,
        unembed_sha256: crate::digest::sha256_file(&u).with_context(|| format!("sha256 of {}", u.display()))?,
    })
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

    /// The int4 table kept packed hands out the rows `to_f32` decodes, bit
    /// for bit, and refuses what `decode_table` refuses.
    #[test]
    fn packed_embedding_rows_are_the_decoded_rows() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let (rows, hidden, vocab) = (40usize, 128usize, 37usize);
        let gpr = hidden / 64;
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let packed = (0..rows * hidden / 2).map(|_| next() as u8).collect();
        let scales = (0..rows * gpr).map(|_| half::f16::from_f32(0.01 + (next() % 100) as f32 * 1e-4).to_bits()).collect();
        let biases = (0..rows * gpr).map(|_| half::f16::from_f32((next() % 200) as f32 * 1e-3 - 0.1).to_bits()).collect();
        let t = llvq_artifact::RawTensor {
            name: EMBED_NAME.into(),
            dims: vec![rows, hidden],
            data: llvq_artifact::RawData::Quant(llvq_artifact::QuantData { bits: 4, group: 64, packed, scales, biases }),
        };
        let dense = t.to_f32();
        let q4 = EmbedTable::from_q4(q4_rows(&t, EMBED_NAME, vocab, hidden)?)?;
        let ids = [0u32, 5, 36, 17];
        let got = q4.rows(&ids, &dev)?.to_vec2::<f32>()?;
        for (r, &id) in got.iter().zip(&ids) {
            let want = &dense[id as usize * hidden..(id as usize + 1) * hidden];
            assert!(r.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()), "row {id}");
        }
        assert!(q4.rows(&[vocab as u32], &dev).is_err(), "a token past the unpadded rows");
        assert_eq!(q4.bytes(), (vocab * hidden / 2 + 4 * vocab * gpr) as u64);
        assert!(q4_rows(&t, UNEMBED_NAME, vocab, hidden).is_err(), "another table's name");
        assert!(q4_rows(&t, EMBED_NAME, rows + 1, hidden).is_err(), "more rows than the table");
        assert!(q4_rows(&t, EMBED_NAME, vocab, 64).is_err(), "another width");
        Ok(())
    }
}
