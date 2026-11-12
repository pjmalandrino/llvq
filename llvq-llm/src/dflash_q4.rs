//! The drafter's vocabulary through the target's own int4 g64 table on the
//! card, in place of a dense copy ([`crate::dflash::DenseVocab`], 0.78 GB in
//! f16 at the 4B). No kernel of its own: both ends reuse kernels the card
//! already compiles for the target.
//!
//! * `embed`: the target's gather, `emb_q4_gather` (`kernels/emb_q4.cu`):
//!   `f16(scale * q + bias)`, an f32 multiply then an f32 add, rounded once,
//!   the arithmetic of [`crate::dflash::dequant_q4_f16`]. The rows are the
//!   dense table's bit for bit.
//! * `logits`: the batched head, `tv_q4_mma_n8/n16/n32`
//!   (`kernels/tv_batch_mma.cu` through [`crate::batch_cuda::CudaBatch`]),
//!   which reads the table once for up to 32 rows. Per 32-column unit it
//!   forms `s * sum(q x) + b * sum(x)` with the nibbles and the f16 activation
//!   exact on the tensor cores, f32 accumulation in the tensor core's order,
//!   the units in column order, K folded in slice order, one f16 store. The
//!   dense path multiplies the f16-rounded weights `f16(s q + b)` instead, in
//!   cuBLAS's order: the two differ by the weights' f16 rounding and the
//!   order of the sums, not bit for bit (`tests/dflash_q4_head.rs` measures
//!   the gap on the fixture's own head inputs).

use std::sync::Arc;

use candle_core::{DType, Tensor};

use crate::batch::BatchDevice;
use crate::device::QuantEmbedTable;
use crate::dflash::TargetVocab;
use crate::model::{Embed, Head, Qwen3};

/// The target's int4 g64 tables (embedding and head, one table when tied)
/// and the batched kernels that read them.
pub struct Q4Vocab {
    embed: Arc<dyn QuantEmbedTable>,
    head: Arc<dyn QuantEmbedTable>,
    batch: crate::batch_cuda::CudaBatch,
    /// Device bytes of the head's table, already resident for the target.
    pub table_bytes: u64,
}

impl Q4Vocab {
    /// From a loaded model whose ends are the int4 g64 tables
    /// (`LLVQ_EMBED=q4`, the served configs). `groups`: query heads a KV
    /// head (4 at the 4B), which the batched module's attention entry needs.
    pub fn new(model: &Qwen3, groups: usize) -> anyhow::Result<Self> {
        let (_, e, h, _, _) = model.pers_4e_parts();
        let embed = match e {
            Embed::Q8(t) => t.clone(),
            Embed::Dense(_) => anyhow::bail!("q4 vocab: the embedding is dense; the served configs carry it int4 g64"),
        };
        let head = match h {
            Head::Q8(t) => t.clone(),
            Head::Dense(_) => anyhow::bail!("q4 vocab: the head is dense; the served configs carry it int4 g64"),
        };
        let view = |t: &Arc<dyn QuantEmbedTable>, what: &str| -> anyhow::Result<(crate::fused_cuda::EmbedView, Arc<crate::fused_cuda::FusedRuntime>)> {
            let c = t
                .as_any()
                .and_then(|a| a.downcast_ref::<crate::fused_cuda::CudaEmbed>())
                .ok_or_else(|| anyhow::anyhow!("q4 vocab: the {what} is not a CUDA table"))?;
            let v = c.q4_view().ok_or_else(|| anyhow::anyhow!("q4 vocab: the {what} is not the int4 g64 table"))?;
            Ok((v, c.runtime().clone()))
        };
        let (ve, _) = view(&embed, "embedding")?;
        let (vh, rt) = view(&head, "head")?;
        anyhow::ensure!(ve.d == vh.d && ve.vocab == vh.vocab, "q4 vocab: embedding {}x{} and head {}x{}", ve.vocab, ve.d, vh.vocab, vh.d);
        let table_bytes = (vh.vocab as u64 * vh.d as u64) / 2 + 2 * 2 * vh.vocab as u64 * vh.gpr as u64;
        let batch = crate::batch_cuda::CudaBatch::new(rt, groups)?;
        Ok(Self { embed, head, batch, table_bytes })
    }

    /// The batched module's provenance lines (unit sha256, registers).
    pub fn provenance(&self) -> &[String] {
        &self.batch.provenance
    }
}

impl TargetVocab for Q4Vocab {
    fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed.gather(&ids.to_dtype(DType::U32)?.contiguous()?)
    }
    fn logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let n = hidden.dim(0)?;
        self.batch.head_mma(self.head.as_ref(), &hidden.to_dtype(DType::F16)?.contiguous()?, n)?.to_dtype(DType::F32)
    }
}
