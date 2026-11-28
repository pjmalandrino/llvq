//! The candle custom ops that launch the served kernels: embedding, head,
//! rotation, the lattice and int4 matvecs, the rows and the segmented forms.
//! One file because each is a thin adapter from a candle storage to a launch.

use super::*;

/// One quantized g64 embedding table (int8 or int4), resident on the device.
///
/// When the model ties its two ends (Qwen3-4B) a single instance serves both
/// the gather at the input and the `lm_head` at the output — which is the
/// point: the −365 MB lot B validated exist only if no f16 copy is ever
/// materialized beside this. When they are untied (Qwen3-8B) there are two
/// instances, one per table, and `EmbedTables::wiring` says which is which.
pub struct QuantEmbed {
    pub vocab: usize,
    pub d: usize,
    pub(super) gpr: u32,
    /// Packed rows as `u32` words, rows word-aligned (`upload_embed` asserts
    /// the width that makes them so).
    pub(super) words: CudaSlice<u32>,
    pub(super) scales: CudaSlice<u16>,
    pub(super) biases: CudaSlice<u16>,
    /// Device bytes: payload + scales + biases.
    pub bytes: u64,
}

/// The gather: token ids in, f16 rows out.
pub(super) struct EmbedOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) q: &'a QuantEmbed,
    pub(super) out_shape: Shape,
}

impl candle_core::CustomOp1 for EmbedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-emb-gather"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the {} gather has no CPU path", self.rt.emode.name())
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
            candle_core::bail!("{} gather: zero tokens", self.rt.emode.name());
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
pub(super) struct HeadOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) q: &'a QuantEmbed,
    pub(super) out_shape: Shape,
}

impl candle_core::CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-lmhead"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the {} lm_head has no CPU path", self.rt.emode.name())
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
            candle_core::bail!("lm_head {}: {len} values for d={}", self.rt.emode.name(), self.q.d);
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
pub(super) struct RotOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) proj: &'a FusedProj,
}

/// `rot_apply_rows` as a candle op: `[n_rows, d_in]` f16 in, the same shape in
/// f32 rotated out.
///
/// Deliberately a second op rather than a row count on [`RotOp`]: a chunk of
/// one takes the one-row path — `model::Proj::prepare_rows` short-circuits —
/// so every decode step stays on the kernel `bin/oracle` certifies, and this
/// one is reached only by a prefill.
pub(super) struct RotRowsOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) proj: &'a FusedProj,
    pub(super) n_rows: usize,
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
pub(super) struct FusedOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) proj: &'a FusedProj,
    pub(super) out_shape: Shape,
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
pub(super) struct FusedInt4Op<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) proj: &'a FusedInt4Proj,
    pub(super) out_shape: Shape,
    /// Which entry point, and so which activation dtype. Set by
    /// [`FusedRuntime::forward_int4`], which made it so.
    pub(super) path: Int4Path,
}

/// The three int4 entry points, by what they read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Int4Path {
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
pub(super) struct FusedRowsOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) proj: &'a FusedProj,
    pub(super) n_rows: usize,
    pub(super) out_shape: Shape,
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
pub(super) struct RotSegOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) group: &'a FusedSegProj,
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
pub(super) struct FusedSegOp<'a> {
    pub(super) rt: &'a FusedRuntime,
    pub(super) f: &'a CudaFunction,
    pub(super) group: &'a FusedSegProj,
    pub(super) out_shape: Shape,
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

