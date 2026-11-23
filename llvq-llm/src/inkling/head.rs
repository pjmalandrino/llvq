//! The output projection of Inkling 2 Mini, behind one seam (plan item D4).
//!
//! [`super::Inkling`] computes its logits through an [`OutputProjection`]
//! when one is set, and through its BF16 `unembed` rows otherwise. The
//! softcap stays in `Inkling::head`, after either.
//!
//! [`Q4MetalHead`] reads the int4 g64 table `bin/inktable` writes
//! (`unembed.q4g64.llvqraw`: one raw tensor in the sealed stream's framing,
//! `inkseal::read_raw_file`) and runs it through the int4 embedding kernel on
//! Metal, one SIMD-group a vocabulary row, f32 in and f32 out
//! (`tv_emb_q4_metal_xf`, `fused_metal::MetalRuntime::head_q4_f32`).
//!
//! This is not class A against the BF16 head: the table's values are the
//! int4 values the operator chose to ship, not the BF16 ones. Its gate is the
//! plain f32 product with the same table decoded.

use candle_core::{Result, Tensor};

/// Hidden states to logits, before the softcap.
pub trait OutputProjection {
    /// A name for reports.
    fn name(&self) -> &str;

    /// Rows of the output, the unpadded vocabulary.
    fn vocab(&self) -> usize;

    /// `h` `[T, H]` f32 to logits `[T, vocab]` f32 on the device of `h`.
    fn project(&self, h: &Tensor) -> Result<Tensor>;

    /// Bytes this projection holds on the device.
    fn device_bytes(&self) -> u64;
}

/// Rows the kernel computes for `vocab`: it carries no row guard and a
/// threadgroup covers 8 rows, so the table is uploaded to the next multiple
/// of 8 and the output narrowed back.
pub fn padded_rows(vocab: usize) -> usize {
    vocab.next_multiple_of(8)
}

/// The int4 g64 head on Metal.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct Q4MetalHead {
    rt: std::sync::Arc<crate::fused_metal::MetalRuntime>,
    table: std::sync::Arc<crate::fused_metal::MetalEmbedTable>,
    vocab: usize,
    bytes: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Q4MetalHead {
    /// Read `path` (`bin/inktable`'s output) and upload its first
    /// `padded_rows(vocab)` rows.
    pub fn load(
        rt: std::sync::Arc<crate::fused_metal::MetalRuntime>,
        path: &std::path::Path,
        vocab: usize,
        hidden: usize,
    ) -> Result<Self> {
        let t = crate::inkseal::read_raw_file(path)
            .map_err(|e| candle_core::Error::msg(format!("{}: {e:#}", path.display())))?;
        Self::from_raw(rt, &t, vocab, hidden)
    }

    /// Upload the first `padded_rows(vocab)` rows of an int4 g64 table.
    ///
    /// Refuses a table that is not int4 in groups of 64, whose row is not
    /// `hidden` wide, or that has fewer rows than the kernel computes.
    pub fn from_raw(
        rt: std::sync::Arc<crate::fused_metal::MetalRuntime>,
        t: &llvq_artifact::RawTensor,
        vocab: usize,
        hidden: usize,
    ) -> Result<Self> {
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{}: an f16 table, the int4 head reads int4 g64", t.name);
        };
        if q.bits != 4 || q.group != 64 {
            candle_core::bail!("{}: int{} in groups of {}, the int4 head reads int4 g64", t.name, q.bits, q.group);
        }
        if t.dims.len() != 2 || t.dims[1] != hidden {
            candle_core::bail!("{}: dims {:?}, expected [rows, {hidden}]", t.name, t.dims);
        }
        let rows = padded_rows(vocab);
        if vocab == 0 || t.dims[0] < rows {
            candle_core::bail!(
                "{}: {} rows for a vocabulary of {vocab}, the kernel computes {rows}",
                t.name,
                t.dims[0]
            );
        }
        let gpr = hidden.div_ceil(q.group);
        let nbytes = rows * hidden / 2;
        if q.packed.len() < nbytes || q.scales.len() < rows * gpr || q.biases.len() < rows * gpr {
            candle_core::bail!("{}: payload shorter than {rows} rows of {hidden}", t.name);
        }
        let wq: Vec<u32> =
            q.packed[..nbytes].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let table = rt.upload_embed(
            &t.name,
            4,
            rows,
            hidden,
            gpr,
            &wq,
            &q.scales[..rows * gpr],
            &q.biases[..rows * gpr],
        )?;
        let bytes = (nbytes + 4 * rows * gpr) as u64;
        Ok(Self { rt, table, vocab, bytes })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl OutputProjection for Q4MetalHead {
    fn name(&self) -> &str {
        &self.table.name
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let y = self.rt.head_q4_f32(&self.table, h)?;
        let rows = y.dim(y.rank() - 1)?;
        if rows == self.vocab {
            Ok(y)
        } else {
            y.narrow(y.rank() - 1, 0, self.vocab)?.contiguous()
        }
    }

    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}

/// The int4 g64 head on CUDA: [`Q4MetalHead`]'s twin, through
/// `fused_cuda::head_q4_f32` (`kernels/ink_head.cu`), f32 in and f32 out.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub struct Q4CudaHead {
    rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
    table: crate::fused_cuda::Q4HeadF32,
    vocab: usize,
    bytes: u64,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
impl Q4CudaHead {
    /// Read `path` (`bin/inktable`'s `unembed.q4g64.llvqraw`) and upload the
    /// rows the kernel computes: the vocabulary padded to a multiple of 8.
    pub fn load(
        rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
        path: &std::path::Path,
        vocab: usize,
        hidden: usize,
    ) -> Result<Self> {
        let t = crate::inkseal::read_raw_file(path)
            .map_err(|e| candle_core::Error::msg(format!("{}: {e:#}", path.display())))?;
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{}: an f16 table, the int4 head reads int4 g64", t.name);
        };
        if q.bits != 4 || q.group != 64 {
            candle_core::bail!("{}: int{} in groups of {}, the int4 head reads int4 g64", t.name, q.bits, q.group);
        }
        let rows = padded_rows(vocab);
        if t.dims.len() != 2 || t.dims[1] != hidden || vocab == 0 || t.dims[0] < rows {
            candle_core::bail!("{}: dims {:?} for a vocabulary of {vocab} over {hidden}", t.name, t.dims);
        }
        let gpr = hidden.div_ceil(q.group);
        let nbytes = rows * hidden / 2;
        if q.packed.len() < nbytes || q.scales.len() < rows * gpr || q.biases.len() < rows * gpr {
            candle_core::bail!("{}: payload shorter than {rows} rows of {hidden}", t.name);
        }
        let wq: Vec<u32> =
            q.packed[..nbytes].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let table =
            rt.upload_head_q4_f32(&t.name, rows, hidden, gpr, &wq, &q.scales[..rows * gpr], &q.biases[..rows * gpr])?;
        Ok(Self { rt, table, vocab, bytes: (nbytes + 4 * rows * gpr) as u64 })
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
impl OutputProjection for Q4CudaHead {
    fn name(&self) -> &str {
        &self.table.name
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let y = self.rt.head_q4_f32(&self.table, h)?;
        let rows = y.dim(y.rank() - 1)?;
        if rows == self.vocab {
            Ok(y)
        } else {
            y.narrow(y.rank() - 1, 0, self.vocab)?.contiguous()
        }
    }

    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}
