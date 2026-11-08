//! The forward entry points of [`FusedRuntime`] and the candle custom ops
//! they run.

use super::*;

impl FusedRuntime {
    /// One activation `[1, d_in]` in f16 → the same activation in the rotated
    /// basis, `[1, d_in]` in f32. Half of what `forward` used to do in one
    /// call; `crate::rotplan` says why it was split and why nothing is cached.
    /// The row loop lives in `model::group_forward` now, hence the one row.
    pub fn rotate(&self, proj: &FusedProj, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the projections of a group.",
                proj.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotOp { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for `n_rows` activations, ONE launch.
    ///
    /// `x` is `[n_rows, d_in]` f16 and contiguous — `model::row_block` narrows
    /// it out of one allocation, so its rows are `d_in` apart and no copy was
    /// made to put them there. The result is `[n_rows, d_in]` f32, contiguous,
    /// which is exactly what `forward_rotated_rows` reads.
    ///
    /// That last sentence is the whole point. Before this, a chunk was rotated
    /// a row at a time into `n_rows` separate allocations and then stacked with
    /// `Tensor::cat` — one device copy a row — to rebuild the very shape the
    /// matvec wanted. Now the shape comes out of the rotation.
    pub fn rotate_rows(
        &self,
        proj: &FusedProj,
        x: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{}: {rows} rows of activation for {n_rows}", proj.name);
        }
        if n_rows == 0 {
            candle_core::bail!("{}: a rotation of zero rows", proj.name);
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotRowsOp { rt: self, proj, n_rows };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one activation already in the rotated basis. `xr` is
    /// [`Self::rotate`]'s f32 output; `out_dims` is the *caller's* shape, so
    /// the result keeps the caller's rank, exactly as a `Linear` would.
    /// `y = W x` for an int4 projection, from the activation in its **natural**
    /// basis — there is no rotated form, and asking for one would be asking for
    /// a basis the weights were never quantized in.
    ///
    /// The kernel accumulates against `x` in f32, "as in every projection
    /// kernel of this family", while the model carries f16. An f16 activation
    /// goes to `tv_q4_hh`, which widens it while staging: exact, so the same
    /// floats as a cast, and one launch fewer a call (84 a token on the sealed
    /// 4B). Any other dtype is widened here and goes to `tv_q4_h`. Neither
    /// narrows the activation: the reference this path is checked against,
    /// `RawTensor::to_f32`, is f32 arithmetic.
    pub fn forward_int4(
        &self,
        proj: &FusedInt4Proj,
        x: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        // The kernel reads from the base pointer. The cast always handed it a
        // fresh tensor at offset zero; a row the prefill narrows out of a
        // block is still at an offset, so that case keeps the cast and
        // `tv_q4_h`, as before. A decode step's activation is fresh already.
        // (`Tensor::copy` would not do: it keeps the layout, offset included.)
        let c = x.contiguous()?;
        // A record staged in f16 (`tv_q4_hs`, past the card's ceiling for the
        // f32 staging) takes its f16 activation where it lies, offset
        // included. An f32 activation cannot reach it: narrowing would change
        // the arithmetic, and `tv_q4_h` does not fit this card at this width.
        if proj.staging == llvq_cuda::wide::Int4Staging::F16 {
            if c.dtype() != DType::F16 {
                candle_core::bail!(
                    "{}: an f32 activation at an int4 record staged in f16 (d_in {}). Narrowing \
                     it would change the arithmetic, and tv_q4_h's f32 staging does not fit \
                     this card at this width",
                    proj.name,
                    proj.d_in
                );
            }
            let op = FusedInt4Op { rt: self, proj, out_shape, path: Int4Path::Wide };
            return c.apply_op1_no_bwd(&op);
        }
        let (xs, half) = match (c.dtype(), c.layout().start_offset()) {
            (DType::F16, 0) => (c, true),
            _ => (c.to_dtype(DType::F32)?.contiguous()?, false),
        };
        let op = FusedInt4Op {
            rt: self,
            proj,
            out_shape,
            path: if half { Int4Path::Half } else { Int4Path::F32 },
        };
        xs.apply_op1_no_bwd(&op)
    }

    /// Whether this runtime's layout carries a kernel that takes several rows.
    pub fn has_rows_kernel(&self) -> bool {
        self.f_matvec_rows.is_some()
    }

    /// Rows this runtime's prefill kernel was COMPILED to take.
    ///
    /// `model::group_forward` chunks by it. Read from the runtime and not from
    /// `tile::PREFILL_ROWS`, because under `LLVQ_PREFILL` they differ — and a
    /// chunk of eight handed to a kernel compiled at four is exactly the
    /// silent corruption the pair exists to prevent.
    pub fn prefill_rows(&self) -> usize {
        self.prefill.rows
    }

    /// `y = W X` for `n_rows` rotated rows, one launch.
    ///
    /// `xr` is `[n_rows, d_in]` f32, contiguous, and based at offset zero —
    /// [`RotRowsOp`] allocates it, so all three hold by construction. It used
    /// to be built by `Tensor::cat` of `n_rows` single-row rotations, one
    /// device copy each; the rotation produces the shape now.
    pub fn forward_rotated_rows(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill.rows {
            candle_core::bail!(
                "{n_rows} rows for a kernel compiled at {}",
                self.prefill.rows
            );
        }
        let op = FusedRowsOp {
            rt: self,
            proj,
            n_rows,
            out_shape: Shape::from(vec![n_rows, proj.d_out]),
        };
        xr.apply_op1_no_bwd(&op)
    }

    pub fn forward_rotated(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        let op = FusedOp {
            rt: self,
            proj,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for a fused group — one `rot_apply`, one f32
    /// `[1, d_in]` result. The parts share one `d_in` and one rotation key by
    /// construction (`fused::segment_matrices` re-asserts both), so there is
    /// one rotation to do and no choice of which.
    pub fn rotate_group(
        &self,
        g: &FusedSegProj,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != g.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", g.name, g.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the parts of the group.",
                g.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotSegOp { rt: self, group: g };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one fused group. `out_dims` is the caller's shape; the
    /// last axis becomes the group's **total** width, which
    /// `model::group_forward` then narrows back into the parts.
    pub fn forward_rotated_seg(
        &self,
        g: &FusedSegProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        // The only place the absent `CudaFunction` can surface, so it surfaces
        // as a message naming the group rather than as an `unwrap` in the
        // middle of a billed job.
        let f = self.f_matvec_seg.as_ref().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "{}: fused group launched by a runtime built without tv_planes_seg_h \
                 (LLVQ_FUSE=0, or a layout that does not segment)",
                g.name
            ))
        })?;
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = g.d_out;
            Shape::from(d)
        };
        let op = FusedSegOp {
            rt: self,
            f,
            group: g,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// Upload a quantized g64 table, the embedding, for the runtime's two
    /// embedding kernels: int8 under [`EmbedMode::Q8`], int4 under
    /// [`EmbedMode::Q4`].
    ///
    /// Every assumption the kernels compile in is asserted here rather than
    /// trusted: the runtime's width, group 64, rows on word boundaries
    /// (`d % 4 == 0` at 8 bits, `d % 8 == 0` at 4), `vocab % 8 == 0` (whole
    /// warps, no bounds guard), and the staged activation within the card's
    /// shared memory.
    pub fn upload_embed(
        &self,
        t: &llvq_artifact::RawTensor,
    ) -> candle_core::Result<QuantEmbed> {
        let (Some(bits), Some((_, head))) = (self.emode.bits(), crate::fused::emb_kernel_names(self.emode))
        else {
            candle_core::bail!("runtime built without embedding kernels (LLVQ_EMBED=f16)");
        };
        if self.f_emb.is_none() {
            candle_core::bail!("runtime built without the {} kernels", self.emode.name());
        }
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{}: not a quantized tensor", t.name);
        };
        if q.bits != bits || q.group != EMBED_GROUP {
            candle_core::bail!(
                "{}: int{} g{}, but the {} kernels hardcode int{bits} g{EMBED_GROUP}",
                t.name, q.bits, q.group, self.emode.name()
            );
        }
        if t.dims.len() != 2 {
            candle_core::bail!("{}: dims {:?}, an embedding is 2-D", t.name, t.dims);
        }
        let (vocab, d) = (t.dims[0], t.dims[1]);
        let per_word = 32 / bits as usize;
        if !d.is_multiple_of(per_word) {
            candle_core::bail!("{}: d={d} is not a multiple of {per_word}", t.name);
        }
        if !vocab.is_multiple_of(8) {
            candle_core::bail!("{}: vocab={vocab} is not a multiple of 8", t.name);
        }
        if d * 4 > self.shared_limit {
            candle_core::bail!(
                "{}: {head} asks for {} B of shared memory, the card offers {}",
                t.name, d * 4, self.shared_limit
            );
        }
        let gpr = d.div_ceil(EMBED_GROUP);
        if q.packed.len() != vocab * d / per_word * 4
            || q.scales.len() != vocab * gpr
            || q.biases.len() != q.scales.len()
        {
            candle_core::bail!(
                "{}: inconsistent payload ({} bytes, {} scales, {} biases for {vocab}×{d})",
                t.name, q.packed.len(), q.scales.len(), q.biases.len()
            );
        }
        let words: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bytes =
            q.packed.len() as u64 + (q.scales.len() as u64 + q.biases.len() as u64) * 2;
        Ok(QuantEmbed {
            vocab,
            d,
            gpr: gpr as u32,
            words: self.cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
            scales: self.cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
            biases: self.cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
            bytes,
        })
    }

    /// Token ids `(.., l)` in u32 → embeddings `(.., l, d)` in f16, one
    /// gather launch for the whole call, rows dequantized on the device.
    pub fn embed(&self, q: &QuantEmbed, ids: &Tensor) -> candle_core::Result<Tensor> {
        let ids = ids.contiguous()?;
        let mut dims = ids.dims().to_vec();
        dims.push(q.d);
        let op = EmbedOp {
            rt: self,
            q,
            out_shape: Shape::from(dims),
        };
        ids.apply_op1_no_bwd(&op)
    }

    /// `logits = W_q8 · h` for hidden states `(.., d)` in f16 — the tied
    /// `lm_head`, one matvec launch per row, all rows into one buffer.
    pub fn lm_head(&self, q: &QuantEmbed, h: &Tensor) -> candle_core::Result<Tensor> {
        let dims = h.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != q.d {
            candle_core::bail!("lm_head q8 expects d={}, got {d_in}", q.d);
        }
        let h = h.contiguous()?.to_dtype(DType::F16)?;
        let out_shape = {
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = q.vocab;
            Shape::from(d)
        };
        let op = HeadOp {
            rt: self,
            q,
            out_shape,
        };
        h.apply_op1_no_bwd(&op)
    }
}

/// The gather: token ids in, f16 rows out.
struct EmbedOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for EmbedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-emb-q8-gather"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the q8 gather has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let ids = storage.as_cuda_slice::<u32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous ids"))?;
        let ntok = end - start;
        if ntok == 0 {
            candle_core::bail!("q8 gather: zero tokens");
        }
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(ntok * self.q.d)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc emb: {e}")))?;
        let (f_gather, _) = self.rt.f_emb.as_ref().expect("checked at upload");
        let (gather_name, _) = crate::fused::emb_kernel_names(self.rt.emode).expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (ntok as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (d, gpr, ids_off) = (self.q.d as u32, self.q.gpr, start as u32);
        let mut b = self.rt.cuda.stream().launch_builder(f_gather);
        b.arg(&self.q.words)
            .arg(&self.q.scales)
            .arg(&self.q.biases)
            .arg(ids)
            .arg(&mut y)
            .arg(&d)
            .arg(&gpr)
            .arg(&ids_off);
        unsafe { b.launch(cfg) }
            .map_err(|e| candle_core::Error::msg(format!("{gather_name}: {e}")))?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// The tied `lm_head`: f16 hidden states in, f16 logits out.
struct HeadOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-lmhead-q8"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the q8 lm_head has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let x = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len == 0 || !len.is_multiple_of(self.q.d) {
            candle_core::bail!("lm_head q8: {len} values for d={}", self.q.d);
        }
        let rows = len / self.q.d;
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(rows * self.q.vocab)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc logits: {e}")))?;
        let (_, f_head) = self.rt.f_emb.as_ref().expect("checked at upload");
        let (_, head_name) = crate::fused::emb_kernel_names(self.rt.emode).expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (self.q.vocab as u32 * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: (self.q.d * 4) as u32,
        };
        let (d, gpr) = (self.q.d as u32, self.q.gpr);
        for r in 0..rows {
            let x_off = (start + r * self.q.d) as u32;
            let y_off = (r * self.q.vocab) as u32;
            let mut b = self.rt.cuda.stream().launch_builder(f_head);
            b.arg(&self.q.words)
                .arg(&self.q.scales)
                .arg(&self.q.biases)
                .arg(x)
                .arg(&mut y)
                .arg(&d)
                .arg(&gpr)
                .arg(&x_off)
                .arg(&y_off);
            unsafe { b.launch(cfg) }
                .map_err(|e| candle_core::Error::msg(format!("{head_name}: {e}")))?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `rot_apply` as a candle op: f16 activation in, f32 rotated activation out.
///
/// The rotated activation used to be a scratch buffer keyed on `d_in` and held
/// between two launches under one lock; it is a tensor now, owned by the group
/// that shares it. `crate::rotplan` carries the argument. Output uninitialised
/// on purpose: `rot_apply` writes every coordinate of `[0, n)`.
struct RotOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
}

/// `rot_apply_rows` as a candle op: `[n_rows, d_in]` f16 in, the same shape in
/// f32 rotated out.
///
/// Deliberately a second op rather than a row count on [`RotOp`]: a chunk of
/// one takes the one-row path — `model::Proj::prepare_rows` short-circuits —
/// so every decode step stays on the kernel `bin/oracle` certifies, and this
/// one is reached only by a prefill.
struct RotRowsOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    n_rows: usize,
}

impl candle_core::CustomOp1 for RotRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply-rows"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation chunk"))?;
        let want = self.proj.d_in * self.n_rows;
        if end - start != want {
            candle_core::bail!(
                "activation of {} values for {} rows of d_in={}",
                end - start,
                self.n_rows,
                self.proj.d_in
            );
        }
        let rot = match self.proj.rotation {
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered, see fused_cuda.rs",
                self.proj.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(want) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot rows: {e}")))?;
        self.rt.launch_rotation_rows(
            rot,
            all,
            &mut xr,
            start as u32,
            // The input rows are `d_in` apart, which is `rot.n` today and
            // is not the same statement: `rot.n` describes the transform,
            // `d_in` describes the buffer.
            self.proj.d_in as u32,
            self.n_rows as u32,
        )?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![self.n_rows, self.proj.d_in]),
        ))
    }
}

impl candle_core::CustomOp1 for RotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        if end - start != self.proj.d_in {
            candle_core::bail!(
                "activation of {} values for d_in={}",
                end - start,
                self.proj.d_in
            );
        }
        let rot = match self.proj.rotation {
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered, see fused_cuda.rs",
                self.proj.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(self.proj.d_in) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot: {e}")))?;
        self.rt.launch_rotation(rot, all, &mut xr, start as u32)?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![1, self.proj.d_in]),
        ))
    }
}

/// The `CustomOp1` candle needs to let us at the tensor's storage.
///
/// There is no public way to reach a `Tensor`'s device pointer other than this
/// trait, which is the right design: it keeps the layout in the picture. We
/// require contiguity rather than honouring arbitrary strides — the kernel
/// stages 24 consecutive floats per block and a strided activation would be
/// silently wrong.
///
/// Since lot A4 it reads [`RotOp`]'s f32 output and launches nothing but the
/// matvec: it no longer rotates, and no longer decides anything.
struct FusedOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        // `(start, start + elem_count)` — a *range*, not `(offset, length)`.
        // Taking the second field for a length is silent at offset 0 and wrong
        // everywhere else, which is exactly how it survived the first run and
        // died on the second vector of the prompt.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.proj.d_in {
            candle_core::bail!(
                "rotated activation of {len} values for d_in={}",
                self.proj.d_in
            );
        }
        // The matvec kernels index `x` from the base pointer and take no
        // offset, so a nonzero start would read the wrong slice and return
        // finite, plausible, wrong numbers. Asserted, not assumed.
        if start != 0 {
            candle_core::bail!(
                "{}: rotated activation at offset {start}, while the matvec kernels \
                 index from the base",
                self.proj.name
            );
        }

        // f16, and uninitialised. Two things at once:
        //
        //  * `tv_slot_h` stores halves, so candle no longer needs a conversion
        //    kernel per projection — 252 launches a token, on a decode whose
        //    budget is half launch latency;
        //  * nothing zeroes it: the kernel writes `y[row]` for every row, the
        //    grid is exact and there is no bounds guard.
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.proj.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        let shared = self.rt.tile.shared_bytes();

        // One arm per layout. The Planes14 arm has no bases to pass — the
        // variant carries none — and the Slot32 arm is the exact call that
        // shipped before the switch existed.
        match &self.proj.stream {
            DeviceStream::Slot32 { words, bases } => self
                .rt
                .cuda
                .launch_slot_h(
                    &self.rt.f_matvec,
                    words,
                    bases,
                    &self.rt.tab,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?,
            DeviceStream::Tetra48 { words, stride_u32 } => {
                let t = self.rt.tetra_tabs.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "tetra48 stream without its constant tables: the runtime was built \
                         for another layout",
                    )
                })?;
                launch_tetra48_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    words,
                    *stride_u32,
                    t,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?
            }
            DeviceStream::Planes14 { words } => launch_planes_h(
                &self.rt.cuda,
                &self.rt.f_matvec,
                words,
                &self.rt.tab,
                &self.proj.gscale,
                &self.proj.rscale,
                &self.proj.tail,
                xr,
                &mut y,
                self.proj.nblocks,
                self.proj.tail_w,
                self.proj.d_out as u32,
                THREADS,
                shared,
            )
            .map_err(candle_core::Error::msg)?,
            DeviceStream::Planes12x {
                words,
                exc_idx,
                exc_words,
                row_exc,
            } => launch_planes12x_h(
                &self.rt.cuda,
                &self.rt.f_matvec,
                &[words, exc_idx, exc_words, row_exc],
                &self.rt.tab,
                &self.proj.gscale,
                &self.proj.rscale,
                &self.proj.tail,
                xr,
                &mut y,
                self.proj.nblocks,
                self.proj.tail_w,
                self.proj.d_out as u32,
                THREADS,
                shared,
            )
            .map_err(candle_core::Error::msg)?,
            DeviceStream::Golay70 {
                words,
                exc_idx,
                exc_words,
                row_exc,
            } => {
                let (cwtab, gtab) = self.rt.g70_tabs.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "Golay70 stream without constant tables, a runtime construction bug",
                    )
                })?;
                launch_golay70_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    &[words, exc_idx, exc_words, row_exc],
                    cwtab,
                    gtab,
                    &self.rt.tab,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `y = W x` for one int4 projection.
///
/// Its own type rather than an arm of [`FusedOp`], for the reason `RotSegOp`
/// is its own: the two carry different borrows and share no field. This one
/// has no rotation, no gain scale, no row scale and no tail to reach for.
struct FusedInt4Op<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedInt4Proj,
    out_shape: Shape,
    /// Which entry point, and so which activation dtype. Set by
    /// [`FusedRuntime::forward_int4`], which made it so.
    path: Int4Path,
}

/// The three int4 entry points, by what they read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Int4Path {
    /// `tv_q4_h`: an f32 activation at offset zero, staged in f32.
    F32,
    /// `tv_q4_hh`: an f16 activation at offset zero, widened into f32 staging.
    Half,
    /// `tv_q4_hs`: an f16 activation at any offset, staged as f16 bits. Only
    /// for records whose f32 staging passes the card's ceiling.
    Wide,
}

impl candle_core::CustomOp1 for FusedInt4Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-int4-matvec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ int4 kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        // A range, not an offset and a length — `FusedOp` records what taking
        // the second field for a length costs, and it costs the same here.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.proj.d_in {
            candle_core::bail!(
                "{}: activation of {len} values for d_in={}",
                self.proj.name,
                self.proj.d_in
            );
        }
        // `tv_q4_h` and `tv_q4_hh` index `x` from the base pointer and take
        // no offset, so a nonzero start would read the wrong slice and return
        // finite, plausible, wrong numbers. `tv_q4_hs` takes the offset.
        if start != 0 && self.path != Int4Path::Wide {
            candle_core::bail!(
                "{}: activation at offset {start}, and the kernel reads from the base",
                self.proj.name
            );
        }
        let missing = || {
            candle_core::Error::msg(
                "an int4 projection on a runtime built without `tv_q4_h`: its source \
                 is appended only when the file carried an int4 record",
            )
        };
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.proj.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        // `as_cuda_slice` refuses a storage of the other dtype, so a `half`
        // flag that lied would fail here by name rather than read wrong bytes.
        // A tiled record goes to the tiled pair and nowhere else: the
        // served pair on tiled words would return finite, wrong numbers.
        let tiled = match self.proj.layout {
            crate::fused::Int4Layout::Served => None,
            crate::fused::Int4Layout::Tiled => Some(self.rt.f_int4_tile.as_ref().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "{}: tiled words on a runtime that carries no tv_q4_ht / tv_q4_hht",
                    self.proj.name
                ))
            })?),
        };
        let launched = match self.path {
            Int4Path::Half => {
                let f = match tiled {
                    Some((_, hh)) => hh,
                    None => self.rt.f_int4_half.as_ref().ok_or_else(missing)?,
                };
                let x = storage.as_cuda_slice::<f16>()?;
                launch_q4_h(&self.rt.cuda, f, self.proj, x, &mut y, THREADS)
            }
            Int4Path::F32 => {
                let f = match tiled {
                    Some((h, _)) => h,
                    None => self.rt.f_int4.as_ref().ok_or_else(missing)?,
                };
                let x = storage.as_cuda_slice::<f32>()?;
                launch_q4_h(&self.rt.cuda, f, self.proj, x, &mut y, THREADS)
            }
            Int4Path::Wide if tiled.is_some() => {
                return Err(candle_core::Error::msg(format!(
                    "{}: an f16-staged launch on tiled words; upload_int4 refuses that pairing",
                    self.proj.name
                )));
            }
            Int4Path::Wide => {
                let f = self.rt.wide.as_ref().and_then(|w| w.int4.as_ref()).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "{}: staged in f16, and the runtime carries no tv_q4_hs",
                        self.proj.name
                    ))
                })?;
                let x = storage.as_cuda_slice::<f16>()?;
                launch_q4_hs(&self.rt.cuda, f, self.proj, x, &mut y, start as u32, THREADS)
            }
        };
        launched.map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// `y = W X` for up to `PREFILL_ROWS` rotated activation rows at once.
///
/// The prefill counterpart of [`FusedOp`]. It exists so a prompt is not
/// decoded once per token: the one-row kernel reads the whole weight stream
/// per row, and a 5-shot MMLU question is several hundred rows.
struct FusedRowsOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    n_rows: usize,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec-rows"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.n_rows * self.proj.d_in {
            candle_core::bail!(
                "{} rows of {} values arrived as {len}",
                self.n_rows,
                self.proj.d_in
            );
        }
        if start != 0 {
            candle_core::bail!("activation at offset {start}, and the kernel reads from the base");
        }
        let f = self.rt.f_matvec_rows.as_ref().ok_or_else(|| {
            candle_core::Error::msg(
                "a prefill launch on a runtime whose layout carries no rows kernel",
            )
        })?;
        let t = self.rt.tetra_tabs.as_ref().ok_or_else(|| {
            candle_core::Error::msg("the rows kernel without the Tetra constant tables")
        })?;
        let words = match &self.proj.stream {
            DeviceStream::Tetra48 { words, .. } => words,
            _ => candle_core::bail!("the rows kernel on a stream that is not Tetra48"),
        };
        let stride_u32 = match &self.proj.stream {
            DeviceStream::Tetra48 { stride_u32, .. } => *stride_u32,
            _ => unreachable!("checked above"),
        };
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(self.n_rows * self.proj.d_out)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        launch_tetra48_rows_h(
            &self.rt.cuda,
            f,
            words,
            stride_u32,
            t,
            &self.proj.gscale,
            &self.proj.rscale,
            &self.proj.tail,
            xr,
            &mut y,
            self.proj.nblocks,
            self.proj.tail_w,
            self.n_rows as u32,
            self.proj.d_in as u32,
            self.proj.d_out as u32,
            THREADS,
            self.rt.prefill_shared,
            // The same `Prefill` that produced the `#define` and the shared
            // bytes above. One value, three uses, no way for them to disagree.
            self.rt.prefill_rows(),
        )
        .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// [`RotOp`] for a fused group — the same launch, keyed on the group's shared
/// rotation instead of one matrix's.
///
/// A separate type rather than an `enum` field on `RotOp`: the two carry
/// different borrows and nothing else, and a shared struct with two `Option`s
/// would be one `unwrap` away from launching a group's rotation on a lone
/// projection's width.
struct RotSegOp<'a> {
    rt: &'a FusedRuntime,
    group: &'a FusedSegProj,
}

impl candle_core::CustomOp1 for RotSegOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply-seg"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        if end - start != self.group.d_in {
            candle_core::bail!(
                "activation of {} values for d_in={}",
                end - start,
                self.group.d_in
            );
        }
        let rot = match self.group.rotation {
            None => candle_core::bail!(
                "{}: group without rotation, path not covered, see fused_cuda.rs",
                self.group.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} missing")))?,
        };
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(self.group.d_in) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot: {e}")))?;
        self.rt.launch_rotation(rot, all, &mut xr, start as u32)?;
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![1, self.group.d_in]),
        ))
    }
}

/// [`FusedOp`] for a fused group: one launch over the row concatenation, with
/// the same three guards on the activation and the same uninitialised output.
///
/// It carries `f` rather than reading `rt.f_matvec_seg` here, so the absent
/// function is refused in [`FusedRuntime::forward_rotated_seg`] — before any
/// allocation — instead of inside a `CustomOp1` whose error surfaces two frames
/// away from the projection that caused it.
struct FusedSegOp<'a> {
    rt: &'a FusedRuntime,
    f: &'a CudaFunction,
    group: &'a FusedSegProj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedSegOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec-seg"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ fused kernel has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let xr = storage.as_cuda_slice::<f32>()?;
        // `(start, start + elem_count)` — a *range*, not `(offset, length)`,
        // exactly as in `FusedOp::cuda_fwd`. Taking the second field for a
        // length is silent at offset 0 and wrong everywhere else.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("non-contiguous activation"))?;
        let len = end - start;
        if len != self.group.d_in {
            candle_core::bail!(
                "rotated activation of {len} values for d_in={}",
                self.group.d_in
            );
        }
        // The matvec kernels index `x` from the base pointer and take no
        // offset, so a nonzero start would read the wrong slice and return
        // finite, plausible, wrong numbers. Asserted, not assumed.
        if start != 0 {
            candle_core::bail!(
                "{}: rotated activation at offset {start}, while the matvec kernels \
                 index from the base",
                self.group.name
            );
        }

        // f16, and uninitialised — the same argument as `FusedOp`, and it needs
        // restating because segmentation is exactly the shape in which it could
        // stop holding: a segmented matrix is a concatenation **by rows**, rows
        // partition the output, the grid is exact, and every row is *stored*
        // rather than accumulated into. No CTA outside a row's own warp writes
        // that row, so there is nothing to zero. ⚠️ If an exception region is
        // ever added to a segmented layout, this allocation becomes a memset in
        // the same commit.
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.group.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        let shared = self.rt.tile.shared_bytes();

        launch_planes_seg_h(
            &self.rt.cuda,
            self.f,
            &self.group.words,
            &self.rt.tab,
            &self.group.gscale,
            &self.group.gs_off,
            &self.group.rscale,
            &self.group.tail,
            xr,
            &mut y,
            self.group.nblocks,
            self.group.tail_w,
            self.group.d_out as u32,
            THREADS,
            shared,
        )
        .map_err(candle_core::Error::msg)?;

        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}
