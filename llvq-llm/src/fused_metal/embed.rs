//! The quantized embedding table on the device, and the two ends that read it: the token gather and the
//! lm_head. One file because both ends read the same table and must agree on its layout.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::tuning::{group_exactly, group_within};
use super::{MetalRuntime, GROUP, LANES, THREADGROUP_LIMIT};

// ---------------------------------------------------------------------------
// The q8 embedding, and the lm_head that reads the same table.
//
// 413.3 MB of the served 4B's 1.39 GB, so it is not a detail. One table, two
// ends: `gather` for the token lookup and `project` for the logits.
// ---------------------------------------------------------------------------

/// A quantized g64 embedding table on the device, int8 or int4.
pub struct MetalEmbedTable {
    pub name: String,
    pub vocab: usize,
    pub d: usize,
    pub(super) gpr: usize,
    /// 8 or 4: which pair of kernels reads this table.
    pub(super) bits: u8,
    pub(super) wq: Arc<Buffer>,
    pub(super) scales: Arc<Buffer>,
    pub(super) biases: Arc<Buffer>,
}

impl MetalRuntime {
    /// Upload one table at `bits` (8 or 4). A tied model uploads ONE and
    /// points both ends at it.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_embed(
        &self,
        name: &str,
        bits: u8,
        vocab: usize,
        d: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<Arc<MetalEmbedTable>> {
        if bits != 8 && bits != 4 {
            candle_core::bail!("{name}: int{bits}, the embedding kernels read int8 or int4");
        }
        let per_word = 32 / bits as usize;
        if !d.is_multiple_of(per_word) {
            candle_core::bail!("{name}: d {d} is not a multiple of {per_word}, so a row would \
                                start inside a word");
        }
        if wq.len() != vocab * d / per_word {
            candle_core::bail!("{name}: {} words for {vocab}x{d} int{bits}", wq.len());
        }
        if scales.len() != vocab * gpr || biases.len() != vocab * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {vocab}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(Arc::new(MetalEmbedTable {
            name: name.to_string(),
            vocab,
            d,
            gpr,
            bits,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        }))
    }
}

/// The table plus the runtime that can launch it.
pub struct MetalEmbed {
    rt: Arc<MetalRuntime>,
    table: Arc<MetalEmbedTable>,
}

impl MetalEmbed {
    pub fn new(rt: Arc<MetalRuntime>, table: Arc<MetalEmbedTable>) -> Self {
        Self { rt, table }
    }

    /// The inner table, so the loader can compare two ends with `Arc::ptr_eq`.
    ///
    /// The tie is NOT observable on the `MetalEmbed`: a tied model wraps one
    /// table in two of them. It IS observable here, which is where
    /// `load_resolved` checks it against `tie_word_embeddings`, exactly as the
    /// CUDA loader does on its own buffers.
    pub fn table(&self) -> &Arc<MetalEmbedTable> {
        &self.table
    }
}

impl crate::device::QuantEmbedTable for MetalEmbed {
    fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::U32)?.contiguous()?;
        ids.apply_op1_no_bwd(&GatherOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let dims = h.dims();
        let d = *dims.last().expect("rank >= 1");
        if d != self.table.d {
            candle_core::bail!("{}: hidden of {d} for d={}", self.table.name, self.table.d);
        }
        let h = h.to_dtype(DType::F16)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }
}

struct GatherOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for GatherOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-gather"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 gather has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::U32 {
            candle_core::bail!("{}: ids must be u32, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous ids", t.name)))?;
        let n_ids = end - start;
        if n_ids == 0 {
            candle_core::bail!("{}: a gather of zero ids", t.name);
        }
        let kname = match t.bits {
            4 => "emb_q4_gather_metal",
            _ => "emb_q8_gather_metal",
        };
        let pipe = self.rt.pipeline(kname)?;
        // Not reducible: the kernel takes its token from the threadgroup's
        // position, so the grid is `n_ids` groups of exactly this many.
        let group = group_exactly(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(n_ids * t.d, DType::F32, "llvq-metal-q8-gather")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*t.wq, 0usize));
        set_param(&enc, 1, (&*t.scales, 0usize));
        set_param(&enc, 2, (&*t.biases, 0usize));
        set_param(&enc, 3, (storage.buffer(), 0usize));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, t.d as u32);
        set_param(&enc, 6, t.gpr as u32);
        // In ELEMENTS: the kernel reads `ids[ids_off + tok]`.
        set_param(&enc, 7, start as u32);
        for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: n_ids * group, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        dims.push(t.d);
        Ok((MetalStorage::new(out, dev, n_ids * t.d, DType::F32), Shape::from(dims)))
    }
}

struct HeadOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-head"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 head has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the head reads f16, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden", t.name)))?;
        let rows = (end - start) / t.d;
        if rows * t.d != end - start {
            candle_core::bail!("{}: {} values is not a whole number of rows of {}",
                               t.name, end - start, t.d);
        }
        let staged = t.d * DType::F32.size_in_bytes();
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!("{}: staging d={} wants {staged} B against {THREADGROUP_LIMIT}",
                               t.name, t.d);
        }
        let kname = match t.bits {
            4 => "tv_emb_q4_metal",
            _ => "tv_q8_metal",
        };
        let pipe = self.rt.pipeline(kname)?;
        let group = group_within(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(rows * t.vocab, DType::F32, "llvq-metal-q8-head")?;
        use candle_metal_kernels::utils::set_param;
        // One launch a row, which is what the CUDA host does: the kernel
        // computes one vocabulary row per SIMD-group and takes one activation.
        for r in 0..rows {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (&*t.wq, 0usize));
            set_param(&enc, 1, (&*t.scales, 0usize));
            set_param(&enc, 2, (&*t.biases, 0usize));
            set_param(&enc, 3, (storage.buffer(), 0usize));
            set_param(&enc, 4, (&*out, 0usize));
            set_param(&enc, 5, t.d as u32);
            set_param(&enc, 6, t.gpr as u32);
            set_param(&enc, 7, (start + r * t.d) as u32);
            set_param(&enc, 8, (r * t.vocab) as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: t.vocab * LANES, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
            );
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.vocab;
        Ok((MetalStorage::new(out, dev, rows * t.vocab, DType::F32), Shape::from(dims)))
    }
}
