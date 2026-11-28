//! Wave 1 of the Inkling 2 Mini throughput plan: resident buffers outside candle's pool, the f32 prefill rows
//! kernel and the f32-input int4 head. One file because the three items came in together, with their own MSL
//! text appended to shipped units, and touch no served path.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::tuning::TETRA_PREFILL_MAX;
use super::launch::{GROUP, LANES, THREADGROUP_LIMIT};
use super::types::{HostProj, MetalRuntime, MetalTetraProj};
use super::{MetalEmbedTable, MetalRotation};

// ---------------------------------------------------------------------------
// Wave 1 of the Inkling 2 Mini throughput plan (`docs/plan-debit-inkling2-mini.md`).
//
// Three additions, none of which touches a served path:
//   - K1: resident data allocated by the Metal device itself, outside
//     candle's buffer pool, with the pool's storage mode.
//   - P1: `tv_tetra48_metal_lutdrf`, the prefill rows kernel storing f32.
//   - D4: `tv_emb_q4_metal_xf`, the int4 g64 head reading an f32 activation.
//
// The two kernels are new text appended to a shipped unit and compiled as a
// library of their own (`source_of`), so the served pipelines keep their
// source bytes and their register allocation.
// ---------------------------------------------------------------------------

/// `tv_tetra48_metal_lutdr` storing f32: the same loop, the same per-row
/// arithmetic, and the served f32 epilogue `tetra48_rows_epilogue` instead of
/// the half one. Row r of a chunk is therefore the f32 the served matvec
/// returns for that row alone. Gated: `tests/inkling_wave1_kernels.rs`.
macro_rules! tetra_rows_f32_msl {
    () => {
        r#"
// ---------------------------------------------------------------------------
// tv_tetra48_metal_lutdrf: tv_tetra48_metal_lutdr with f32 rows (plan item P1).
// ---------------------------------------------------------------------------
kernel void tv_tetra48_metal_lutdrf(const device uint*   words          [[buffer(0)]],
                                    constant uint&       row_stride_u32 [[buffer(1)]],
                                    const device uint*   rows           [[buffer(2)]],
                                    const device uchar*  prefixes       [[buffer(3)]],
                                    const device ushort* branches       [[buffer(4)]],
                                    const device uchar*  suffixes       [[buffer(5)]],
                                    const device float*  gscale         [[buffer(6)]],
                                    const device float*  invnorm        [[buffer(7)]],
                                    const device float*  rscale         [[buffer(8)]],
                                    const device half*   tail           [[buffer(9)]],
                                    const device float*  x              [[buffer(10)]],
                                    device float*        y              [[buffer(11)]],
                                    constant uint&       nblocks        [[buffer(12)]],
                                    constant uint&       tail_w         [[buffer(13)]],
                                    constant uint&       nrows          [[buffer(15)]],
                                    constant uint&       d_out          [[buffer(16)]],
                                    uint gid  [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    uint d_in = nblocks * 24u + tail_w;
    float acc[TETRA48_PREFILL_MAX];
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
            if (r < nrows) {
                const device float* xb = x + r * d_in + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) {
                    a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                }
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue(acc[r], row, lane, rscale, tail, x + r * d_in, y + r * d_out,
                                  nblocks, tail_w);
        }
    }
}
"#
    };
}

/// `tv_emb_q4_metal` reading an f32 activation: the int4 g64 head of a model
/// that runs in f32 (Inkling 2 Mini), so the hidden state is not narrowed to
/// f16 on its way in. The staging, the lane order, the dequantization and the
/// butterfly are `tv_emb_q4_metal`'s text; only the type of `x` changes.
macro_rules! emb_q4_xf_msl {
    () => {
        r#"
// ---------------------------------------------------------------------------
// tv_emb_q4_metal_xf: tv_emb_q4_metal with an f32 activation (plan item D4).
// ---------------------------------------------------------------------------
kernel void tv_emb_q4_metal_xf(const device uint*  wq     [[buffer(0)]],
                               const device half*  scales [[buffer(1)]],
                               const device half*  biases [[buffer(2)]],
                               const device float* x      [[buffer(3)]],
                               device float*       y      [[buffer(4)]],
                               constant uint&      d      [[buffer(5)]],
                               constant uint&      gpr    [[buffer(6)]],
                               constant uint&      x_off  [[buffer(7)]],
                               constant uint&      y_off  [[buffer(8)]],
                               threadgroup float*  xs     [[threadgroup(0)]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint gid  [[thread_position_in_grid]],
                               uint tgs  [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    for (uint i = tid; i < d; i += tgs) {
        xs[i] = x[x_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nwords = d >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc = 0.0f;
    for (uint wi = lane; wi < nwords; wi += 32u) {
        uint p = wq[w0 + wi];
        uint c = wi << 3;
        half sb = scales[s0 + (c >> 6)];
        half bb = biases[s0 + (c >> 6)];
        for (uint k = 0; k < 8u; ++k) {
            acc = fma(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xs[c + k], acc);
        }
    }
    acc = e4_warp_sum(acc);
    if (lane == 0u) {
        y[y_off + row] = acc;
    }
}
"#
    };
}

pub(super) const SRC_TETRA_ROWS_F32: &str = concat!(
    include_str!("../../kernels/llvq_tetra48.metal"),
    include_str!("../../kernels/llvq_tetra48_rows.metal"),
    tetra_rows_f32_msl!()
);
pub(super) const SRC_E4_XF: &str = concat!(include_str!("../../kernels/emb_q4.metal"), emb_q4_xf_msl!());

impl MetalRuntime {
    /// A buffer holding `data`, made by the Metal device itself, in the
    /// storage mode candle's pool uses (`StorageModeShared`), and NOT
    /// registered in that pool (plan item K1).
    ///
    /// candle 0.9.2 registers every `new_buffer_with_data` in its pool, and
    /// walks the pool at every output allocation (`find_available_buffer`)
    /// and at every command-buffer flush (`drop_unused_buffers`). A resident
    /// Inkling layer puts 2,048 buffers there, 49,152 for 24 layers. A buffer
    /// made here is freed when its last `Arc` drops, and the allocator can
    /// never hand it out as scratch (trap 3 of this file's header).
    pub(super) fn buffer_outside_pool<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        if size == 0 {
            candle_core::bail!("Metal refuses a zero-length buffer");
        }
        let b = self
            .device
            .device()
            .new_buffer_with_data(data.as_ptr().cast(), size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .map_err(|e| candle_core::Error::msg(format!("a buffer of {size} B outside the pool: {e:?}")))?;
        Ok(Arc::new(b))
    }

    /// [`Self::upload_stream`] with every buffer outside candle's pool: the
    /// same checks, the same bytes, another allocator. The resident Inkling
    /// experts use it; no served path does.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_stream_unpooled(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u32],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        // The checks of `upload`, in its order and with its messages.
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}. The kernel \
                 carries no row guard, so a partial threadgroup would compute and STORE \
                 rows past the output"
            );
        }
        if nblocks * llvq_core::DIM > d_in {
            candle_core::bail!("{name}: {nblocks} blocks for d_in {d_in}");
        }
        let tail_w = d_in - nblocks * llvq_core::DIM;
        if rscale.len() != d_out {
            candle_core::bail!("{name}: {} row scales for {d_out} rows", rscale.len());
        }
        if tail.len() != d_out * tail_w {
            candle_core::bail!("{name}: {} tail values for {d_out}x{tail_w}", tail.len());
        }
        let want_stride = llvq_artifact::tetra48::stride_u32(nblocks);
        if row_stride_u32 != want_stride {
            candle_core::bail!(
                "{name}: row stride {row_stride_u32} for {nblocks} blocks, which wants \
                 {want_stride}"
            );
        }
        let want_words = d_out * row_stride_u32;
        if words.len() != want_words && words.len() != want_words + 1 {
            candle_core::bail!(
                "{name}: {} stream words for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_words} or {want_words} plus one guard word",
                words.len()
            );
        }
        let dummy = [0u16];
        let tail_src = if tail.is_empty() { &dummy[..] } else { tail };
        // The words as the reader hands them back: on this little-endian
        // machine the same bytes `upload_stream` builds by hand.
        Ok(MetalTetraProj {
            name: name.to_string(),
            d_out,
            d_in,
            nblocks,
            tail_w,
            row_stride_u32,
            words: self.buffer_outside_pool(words)?,
            gscale: self.buffer_outside_pool(gscale)?,
            rscale: self.buffer_outside_pool(rscale)?,
            tail: self.buffer_outside_pool(tail_src)?,
            host: self.tables.host.is_some().then(|| HostProj {
                words: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
                gscale: gscale.to_vec(),
                rscale: rscale.to_vec(),
                tail: tail.to_vec(),
            }),
        })
    }

    /// [`Self::upload_rotation`] with both tables outside candle's pool.
    pub fn upload_rotation_unpooled(&self, t: &crate::fused::RotationTables) -> Result<MetalRotation> {
        let dummy = [0f32];
        let small = if t.small.is_empty() { &dummy[..] } else { &t.small };
        Ok(MetalRotation {
            signbits: self.buffer_outside_pool(&t.signbits)?,
            small: self.buffer_outside_pool(small)?,
            n: t.n as u32,
            m: t.m as u32,
            k: t.k as u32,
            inv: t.inv,
            threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
        })
    }

    /// `Y = W X` for `n_rows` rotated rows `[n_rows, d_in]` in f32, one launch
    /// of `tv_tetra48_metal_lutdrf`, as f32 `[n_rows, d_out]`. Row r is the
    /// served matvec of row r, bit for bit (plan item P1). Same bounds as
    /// [`Self::matvec_rows`].
    pub fn matvec_rows_f32(&self, proj: &MetalTetraProj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill_rows.max(1) || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for a prefill launch of at most {} (kernel bound {TETRA_PREFILL_MAX})",
                proj.name,
                self.prefill_rows
            );
        }
        x.apply_op1_no_bwd(&TetraRowsF32 { rt: self, proj, n_rows })
    }

    /// Logits `h W^T` of an int4 g64 table for `h` `[rows, d]` in f32, as f32
    /// `[rows, vocab]`: one launch of `tv_emb_q4_metal_xf` a row, the f32
    /// activation staged as it arrives (plan item D4).
    pub fn head_q4_f32(&self, t: &MetalEmbedTable, h: &Tensor) -> Result<Tensor> {
        if t.bits != 4 {
            candle_core::bail!("{}: int{}, the f32 head reads int4", t.name, t.bits);
        }
        let d = *h.dims().last().expect("rank >= 1");
        if d != t.d {
            candle_core::bail!("{}: hidden of {d} for d={}", t.name, t.d);
        }
        let h = h.to_dtype(DType::F32)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadF32Op { rt: self, t })
    }
}

impl MetalTetraProj {
    /// Device bytes of the four buffers (words, gain scales, row scales,
    /// tail): what a report counts.
    pub fn device_bytes(&self) -> u64 {
        (self.words.length() + self.gscale.length() + self.rscale.length() + self.tail.length()) as u64
    }
}

/// One launch of `tv_tetra48_metal_lutdrf`, as a candle op.
struct TetraRowsF32<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
    n_rows: usize,
}

impl CustomOp1 for TetraRowsF32<'_> {
    fn name(&self) -> &'static str {
        "tetra48-rows-f32"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "{}: the f32 prefill kernel runs on Metal only; its reference is the matvec, row by row",
            self.proj.name
        )
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: prefill rows in f32 only, got {:?}", p.name, storage.dtype());
        }
        if !layout.is_contiguous() {
            candle_core::bail!("{}: prefill rows must be contiguous", p.name);
        }
        if layout.dims() != [self.n_rows, p.d_in] {
            candle_core::bail!(
                "{}: prefill activation {:?} for [{}, {}]",
                p.name,
                layout.dims(),
                self.n_rows,
                p.d_in
            );
        }
        if !p.d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("{}: d_out {} is not a multiple of {} rows", p.name, p.d_out, GROUP / LANES);
        }
        let dev = storage.device().clone();
        let pipe = self.rt.pipeline("tv_tetra48_metal_lutdrf")?;
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F32, "tetra48-rows-f32")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, p.row_stride_u32 as u32);
        set_param(&enc, 2, (&*self.rt.tables.rows, 0usize));
        set_param(&enc, 3, (&*self.rt.tables.prefixes, 0usize));
        set_param(&enc, 4, (&*self.rt.tables.branches, 0usize));
        set_param(&enc, 5, (&*self.rt.tables.suffixes, 0usize));
        set_param(&enc, 6, (&*p.gscale, 0usize));
        set_param(&enc, 7, (&*self.rt.tables.invnorm, 0usize));
        set_param(&enc, 8, (&*p.rscale, 0usize));
        set_param(&enc, 9, (&*p.tail, 0usize));
        set_param(&enc, 10, (storage.buffer(), x_off));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, p.nblocks as u32);
        set_param(&enc, 13, p.tail_w as u32);
        set_param(&enc, 15, self.n_rows as u32);
        set_param(&enc, 16, p.d_out as u32);
        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F32),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

/// `tv_emb_q4_metal_xf`, one launch a row, as a candle op.
struct HeadF32Op<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for HeadF32Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4-head-f32"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 f32 head has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: the f32 head reads f32, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden", t.name)))?;
        let rows = (end - start) / t.d;
        if rows == 0 || rows * t.d != end - start {
            candle_core::bail!("{}: {} values is not a whole number of rows of {}", t.name, end - start, t.d);
        }
        // The kernel's own limits: no row guard (8 rows a threadgroup), the
        // activation staged whole, every index a u32.
        if !t.vocab.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("{}: {} rows is not a multiple of {}", t.name, t.vocab, GROUP / LANES);
        }
        let staged = t.d * DType::F32.size_in_bytes();
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!("{}: staging d={} wants {staged} B against {THREADGROUP_LIMIT}", t.name, t.d);
        }
        let reach = [rows * t.vocab, t.vocab * (t.d / 8), t.vocab * t.gpr, start + rows * t.d];
        if reach.iter().any(|&v| v > u32::MAX as usize) || t.vocab * LANES > u32::MAX as usize {
            candle_core::bail!("{}: an index past u32 ({reach:?})", t.name);
        }
        let pipe = self.rt.pipeline("tv_emb_q4_metal_xf")?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(rows * t.vocab, DType::F32, "llvq-metal-q4-head-f32")?;
        use candle_metal_kernels::utils::set_param;
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
                objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
            );
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.vocab;
        Ok((MetalStorage::new(out, dev, rows * t.vocab, DType::F32), Shape::from(dims)))
    }
}
