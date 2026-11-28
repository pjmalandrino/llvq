//! The int4 arm: a projection served as affine int4 g128, its upload and its one-row and rows launches. One
//! file because it is the only record kind the lattice kernels do not read.

use super::*;

// ---------------------------------------------------------------------------
// The int4 arm: `v_proj` on the served mixed file, 36 of the 252 projections.
// ---------------------------------------------------------------------------

/// One projection served as affine int4 g128.
pub struct MetalInt4Proj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    gpr: usize,
    /// `None`: `tv_q4_metal` stages the whole activation.
    /// `Some(c)`: `tv_q4_metal_tiled` stages `c` columns a tile.
    tile_cols: Option<usize>,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalInt4Proj {
    /// The tile this record was routed with: `None` for the whole activation.
    pub fn tile_cols(&self) -> Option<usize> {
        self.tile_cols
    }
}

impl MetalRuntime {
    /// Upload one int4 projection, at this runtime's tile if it is past the
    /// wall.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_int4(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<MetalInt4Proj> {
        self.upload_int4_tiled(name, d_out, d_in, gpr, wq, scales, biases, self.q4_tile_cols)
    }

    /// [`Self::upload_int4`] at an explicit tile, which only a record past
    /// the wall uses. The primitive the binding test drives.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_int4_tiled(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
        tile: usize,
    ) -> Result<MetalInt4Proj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}, and the kernel \
                 carries no row guard"
            );
        }
        if !d_in.is_multiple_of(8) {
            candle_core::bail!("{name}: d_in {d_in} is not a multiple of 8, so a row would \
                                start at a shifted nibble");
        }
        // The whole activation when it fits, `tile` columns at a time past
        // the wall (`tv_q4_metal_tiled`). The tile is checked whether or not
        // this record needs it, so a bad one is refused at the first record.
        MetalTuning { q4_tile_cols: tile, ..Default::default() }.validate()?;
        let tile_cols = int4_tile_for(name, d_in, gpr, tile)?;
        if wq.len() != d_out * d_in / 8 {
            candle_core::bail!("{name}: {} words for {d_out}x{d_in} int4", wq.len());
        }
        if scales.len() != d_out * gpr || biases.len() != d_out * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {d_out}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(MetalInt4Proj {
            name: name.to_string(),
            d_out,
            d_in,
            gpr,
            tile_cols,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        })
    }

    /// `y = W x`, one vector, through `tv_q4_metal`.
    pub fn matvec_int4(&self, proj: &MetalInt4Proj, x: &Tensor) -> Result<Tensor> {
        // Widened here, as `FusedRuntime::forward_int4` does. This arm reads
        // the CALLER's activation, which arrives in the model's f16, while the
        // kernel stages f32. `contiguous` because a narrowed row is a view.
        //
        // The half-io twins read the f16 activation as it arrives and widen
        // it while staging, which is the same exact widening.
        let x = match (self.half_io, x.dtype()) {
            (true, DType::F16) => x.contiguous()?,
            _ => x.to_dtype(DType::F32)?.contiguous()?,
        };
        x.apply_op1_no_bwd(&Int4Op { rt: self, proj })
    }

    /// `Y = W X` for `n_rows` rows `[n_rows, d_in]`, one launch of
    /// `tv_q4_metal_rows`, as f16 `[n_rows, d_out]`. Row r is the int4 matvec
    /// of row r narrowed to f16, bit for bit. Widened to f32 first, as the
    /// served path widens: exact from f16.
    pub fn matvec_int4_rows(&self, proj: &MetalInt4Proj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if !self.q4_rows || n_rows == 0 || n_rows > self.prefill_rows || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for an int4 prefill launch of at most {} (q4 rows {})",
                proj.name,
                self.prefill_rows,
                self.q4_rows
            );
        }
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        x.apply_op1_no_bwd(&Int4Rows { rt: self, proj, n_rows })
    }
}

/// One launch of `tv_q4_metal_rows`, as a candle op.
struct Int4Rows<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalInt4Proj,
    n_rows: usize,
}

impl CustomOp1 for Int4Rows<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 prefill kernel has no CPU path", self.proj.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: int4 prefill rows in f32 only, got {:?}", p.name, storage.dtype());
        }
        if !layout.is_contiguous() || layout.dims() != [self.n_rows, p.d_in] {
            candle_core::bail!(
                "{}: int4 prefill activation {:?} for a contiguous [{}, {}]",
                p.name,
                layout.dims(),
                self.n_rows,
                p.d_in
            );
        }
        let pipe = self.rt.pipeline("tv_q4_metal_rows")?;
        let group = group_within(&pipe, "tv_q4_metal_rows", GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F16, "llvq-metal-q4-rows")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.wq, 0usize));
        set_param(&enc, 1, (&*p.scales, 0usize));
        set_param(&enc, 2, (&*p.biases, 0usize));
        set_param(&enc, 3, (storage.buffer(), layout.start_offset() * DType::F32.size_in_bytes()));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, p.d_in as u32);
        set_param(&enc, 6, p.gpr as u32);
        set_param(&enc, 7, self.n_rows as u32);
        set_param(&enc, 8, p.d_out as u32);
        for b in [&*p.wq, &*p.scales, &*p.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F16),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

struct Int4Op<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalInt4Proj,
}

impl CustomOp1 for Int4Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 kernel has no CPU path", self.proj.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        let half = match (storage.dtype(), self.rt.half_io) {
            (DType::F32, _) => false,
            (DType::F16, true) => true,
            (other, _) => candle_core::bail!(
                "{}: f32, or f16 on the half-io twins, got {other:?} with half-io {}",
                p.name,
                self.rt.half_io
            ),
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", p.name)))?;
        if end - start != p.d_in {
            candle_core::bail!("{}: activation of {} values for d_in={}", p.name, end - start, p.d_in);
        }
        let (kname, staged_cols) = match (p.tile_cols, self.rt.q4_prefetch, half) {
            (None, false, false) => ("tv_q4_metal", p.d_in),
            (None, true, false) => ("tv_q4_metal_p", p.d_in),
            (Some(c), false, false) => ("tv_q4_metal_tiled", c),
            (Some(c), true, false) => ("tv_q4_metal_tiled_p", c),
            (None, false, true) => ("tv_q4_metal_h", p.d_in),
            (Some(c), false, true) => ("tv_q4_metal_tiled_h", c),
            (_, true, true) => candle_core::bail!("{}: half-io and prefetch are two arms", p.name),
        };
        let io = if half { DType::F16 } else { DType::F32 };
        let pipe = self.rt.pipeline(kname)?;
        let group = group_within(&pipe, kname, GROUP)?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(p.d_out, io, "llvq-metal-q4")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.wq, 0usize));
        set_param(&enc, 1, (&*p.scales, 0usize));
        set_param(&enc, 2, (&*p.biases, 0usize));
        // Bound AT the offset, so the kernel's `x[i]` starts at the row.
        set_param(&enc, 3, (storage.buffer(), start * io.size_in_bytes()));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, p.d_in as u32);
        set_param(&enc, 6, p.gpr as u32);
        if let Some(c) = p.tile_cols {
            // Never left unbound: a zero here loops `c0 += 0` forever.
            set_param(&enc, 7, c as u32);
        }
        // The tile, never d_in, on the tiled path: d_in * 4 is above the
        // limit there. Omitted, the kernel would run and write zeros.
        enc.set_threadgroup_memory_length(0, staged_cols * DType::F32.size_in_bytes());
        for b in [&*p.wq, &*p.scales, &*p.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((MetalStorage::new(out, dev, p.d_out, io), Shape::from(dims)))
    }
}
