//! The runtime's core methods and the Tetra matvec itself: the decoder tables uploaded, the pipeline cache, the
//! one-row and prefill-rows launches as candle ops, and the host oracle that replays the shader's order. One
//! file because the oracle is only worth anything next to the launch it shadows.

use super::*;

impl MetalRuntime {
    /// Upload the decoder tables and take the device.
    ///
    /// `keep_host` keeps a host copy of everything, which is what lets
    /// [`TetraMatvec::cpu_fwd`] act as the oracle a test compares against. A
    /// served model passes `false` and the CPU arm then refuses by name
    /// instead of silently costing a model its memory twice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &candle_core::Device,
        rows: &[u32],
        prefixes: &[u8],
        branches: &[u16],
        suffixes: &[u8],
        invnorm: &[f32],
        tile: usize,
        keep_host: bool,
    ) -> Result<Self> {
        let dev = match device {
            candle_core::Device::Metal(d) => d.clone(),
            other => candle_core::bail!("the Metal runtime wants a Metal device, got {other:?}"),
        };
        check_threadgroup_memory(&dev)?;
        // 32..=256, not the CUDA contract's 32..=512. The staging is
        // `tile * 24 * 4` bytes against Apple's 32,768 B of threadgroup
        // memory, so 341 blocks is the ceiling and 256 the largest power of
        // two under it. 512 would ask for 49,152 B, pass a validator copied
        // from the wrong platform, and abort the process at dispatch.
        if !tile.is_power_of_two() || !(32..=256).contains(&tile) {
            candle_core::bail!(
                "tile {tile} is not a power of two in 32..=256. Apple stages \
                 {} B for it against a limit of 32768",
                tile * llvq_core::DIM * 4
            );
        }
        let tables = MetalTables {
            rows: dev.new_buffer_with_data(rows)?,
            prefixes: dev.new_buffer_with_data(prefixes)?,
            branches: dev.new_buffer_with_data(branches)?,
            suffixes: dev.new_buffer_with_data(suffixes)?,
            invnorm: dev.new_buffer_with_data(invnorm)?,
            c3_of: dev.new_buffer_with_data(&compose_c3(branches, suffixes))?,
            host: keep_host.then(|| HostTables {
                rows: rows.to_vec(),
                prefixes: prefixes.to_vec(),
                branches: branches.to_vec(),
                suffixes: suffixes.to_vec(),
                invnorm: invnorm.to_vec(),
            }),
        };
        Ok(Self {
            kernel: kernel_from_env()?,
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
            moe_down_sg: std::sync::atomic::AtomicBool::new(MOE_DOWN_SG_DEFAULT),
            device: dev,
            tables,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Upload one projection's stream and its scales.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u8],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}. The kernel \
                 carries no row guard, so a partial threadgroup would compute and STORE \
                 rows past the output"
            );
        }
        let tail_w = d_in - nblocks * llvq_core::DIM;
        if rscale.len() != d_out {
            candle_core::bail!("{name}: {} row scales for {d_out} rows", rscale.len());
        }
        if tail.len() != d_out * tail_w {
            candle_core::bail!("{name}: {} tail values for {d_out}x{tail_w}", tail.len());
        }
        // The stream, which the kernel indexes as `words + row * row_stride_u32`
        // with no bound of its own. `rscale` and `tail` were checked and this
        // was not, which is the asymmetry an audit of 2026-09-21 named.
        let want_stride = llvq_artifact::tetra48::stride_u32(nblocks);
        if row_stride_u32 != want_stride {
            candle_core::bail!(
                "{name}: row stride {row_stride_u32} for {nblocks} blocks, which wants \
                 {want_stride}"
            );
        }
        // The stream is `d_out * row_stride_u32` words, plus AT MOST one guard
        // word. `f1r_load` reads `row[w]` and `row[w + 1]`, so the last block
        // of the last row reaches one word past the table; the writer pads for
        // it. An equality here refused the served 4B by four bytes.
        let want_bytes = d_out * row_stride_u32 * 4;
        if words.len() != want_bytes && words.len() != want_bytes + 4 {
            candle_core::bail!(
                "{name}: {} stream bytes for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_bytes} or {want_bytes} plus one guard word",
                words.len()
            );
        }
        // Metal refuses a zero-length buffer and hands back a null pointer,
        // the same wall cudarc puts up. A `d_in` that is a multiple of 24 has
        // no tail, so it gets a one-element dummy the kernel never reads:
        // `tail_w == 0` makes the epilogue's loop empty.
        let dummy = [0u16];
        let tail_src = if tail.is_empty() { &dummy[..] } else { tail };
        Ok(MetalTetraProj {
            name: name.to_string(),
            d_out,
            d_in,
            nblocks,
            tail_w,
            row_stride_u32,
            words: self.device.new_buffer_with_data(words)?,
            gscale: self.device.new_buffer_with_data(gscale)?,
            rscale: self.device.new_buffer_with_data(rscale)?,
            tail: self.device.new_buffer_with_data(tail_src)?,
            host: self.tables.host.is_some().then(|| HostProj {
                words: words.to_vec(),
                gscale: gscale.to_vec(),
                rscale: rscale.to_vec(),
                tail: tail.to_vec(),
            }),
        })
    }

    /// The pipeline for `name` at this runtime's tile, compiled once.
    pub(super) fn pipeline(&self, name: &'static str) -> Result<ComputePipeline> {
        let key: Key = (name, self.tile);
        if let Some(p) = self.pipelines.read().expect("no panic held this lock").get(&key) {
            return Ok(p.clone());
        }
        let src = format!("#define LLVQ_TILE_BLOCKS {}u\n{}", self.tile, source_of(name)?);
        let dev = self.device.device().clone();
        // A bad source PANICS rather than returning Err: both
        // `new_library_with_source` and `new_compute_pipeline_state_with_function`
        // end in `.unwrap()`. The panic payload carries Metal's own diagnostics,
        // so it is re-raised as the message instead of being swallowed.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lib = dev.new_library_with_source(&src, None)?;
            let func = lib.get_function(name, None)?;
            dev.new_compute_pipeline_state_with_function(&func)
        }));
        let pipe = match built {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => candle_core::bail!("{name}: Metal refused the source: {e:?}"),
            Err(panic) => {
                let what = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "no diagnostics".into());
                candle_core::bail!("{name}: the MSL did not compile: {what}");
            }
        };
        // The matvec and head kernels give a row one SIMD-group of 32 lanes,
        // find it as `gid >> 5`, reduce across lanes `l ^ 16 .. l ^ 1` and
        // write from lane 0. A pipeline that runs another width would return
        // wrong rows with no error. Checked on every pipeline, before it is
        // cached: a device whose width is not 32 serves none of the path.
        {
            use objc2_metal::MTLComputePipelineState as _;
            let width = pipe.as_ref().threadExecutionWidth();
            if width != LANES {
                candle_core::bail!(
                    "{name}: the pipeline runs SIMD-groups of {width} threads, and the kernels of this \
                     file assume {LANES}"
                );
            }
        }
        self.pipelines
            .write()
            .expect("no panic held this lock")
            .insert(key, pipe.clone());
        Ok(pipe)
    }

    /// `y = W x` for one activation row, through the kernel.
    pub fn matvec(&self, proj: &MetalTetraProj, x: &Tensor) -> Result<Tensor> {
        let op = TetraMatvec { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }

    /// Prompt rows a Tetra prefill launch takes; 1 is no rows kernel.
    pub fn prefill_rows(&self) -> usize {
        self.prefill_rows
    }

    /// `Y = W X` for `n_rows` rotated rows `[n_rows, d_in]` in f32, one launch
    /// of `tv_tetra48_metal_lutdr`, as f16 `[n_rows, d_out]`. Row r is the
    /// matvec of row r narrowed to f16, bit for bit.
    pub fn matvec_rows(&self, proj: &MetalTetraProj, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill_rows.max(1) || n_rows > TETRA_PREFILL_MAX {
            candle_core::bail!(
                "{}: {n_rows} rows for a prefill launch of at most {} (kernel bound {TETRA_PREFILL_MAX})",
                proj.name,
                self.prefill_rows
            );
        }
        x.apply_op1_no_bwd(&TetraRows { rt: self, proj, n_rows })
    }
}

/// One launch of `tv_tetra48_metal_lutdr`, as a candle op.
struct TetraRows<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
    n_rows: usize,
}

impl CustomOp1 for TetraRows<'_> {
    fn name(&self) -> &'static str {
        "tetra48-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "{}: the Tetra prefill kernel runs on Metal only; its reference is the matvec, row by row",
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
        let pipe = self.rt.pipeline("tv_tetra48_metal_lutdr")?;
        let group = group_within(&pipe, "tv_tetra48_metal_lutdr", GROUP)?;
        let out = dev.new_buffer(self.n_rows * p.d_out, DType::F16, "tetra48-rows")?;
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
            objc2_metal::MTLSize { width: group, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.n_rows * p.d_out, DType::F16),
            Shape::from((self.n_rows, p.d_out)),
        ))
    }
}

/// One launch of `tv_tetra48_metal`, as a candle op.
struct TetraMatvec<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
}

impl TetraMatvec<'_> {
    /// What both arms owe the caller before either computes anything.
    ///
    /// Added after an audit of 2026-09-21 found `metal_fwd` checking dtype and
    /// contiguity and nothing else, while `cpu_fwd` checked only the dtype.
    /// The CUDA twin refuses all of this by name at `fused_cuda/`, with
    /// a comment saying the missing check already cost one run.
    fn check(&self, dtype: DType, layout: &Layout) -> Result<()> {
        let p = self.proj;
        if dtype != DType::F32 {
            candle_core::bail!("{}: f32 only, got {dtype:?}", p.name);
        }
        if !layout.is_contiguous() {
            candle_core::bail!(
                "{}: the activation must be contiguous. The kernel indexes it linearly \
                 from one base and cannot honour a stride",
                p.name
            );
        }
        let dims = layout.dims();
        let d_in = *dims.last().unwrap_or(&0);
        if d_in != p.d_in {
            candle_core::bail!(
                "{}: activation of {d_in} values for d_in={}. A short one reads past the \
                 buffer and returns finite, plausible, wrong numbers",
                p.name,
                p.d_in
            );
        }
        // One vector a launch, like the CUDA twin. The output is allocated at
        // `d_out` elements, so accepting several rows would hand back a tensor
        // whose shape claims more than its storage holds.
        let rows: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: {rows} vectors at once. This kernel is a matvec and takes one; the \
                 row loop belongs to the caller",
                p.name
            );
        }
        Ok(())
    }
}

impl CustomOp1 for TetraMatvec<'_> {
    fn name(&self) -> &'static str {
        "tetra48-matvec"
    }

    /// The oracle, not a stub.
    ///
    /// `CustomOp1` requires a CPU arm, and making it the reference rather than
    /// a `bail!` is what lets a test drive the SAME op on both devices and
    /// demand equality. It reproduces the kernel's summation order exactly:
    /// 32 lanes striding the blocks, the shuffle-xor butterfly, and the
    /// multiply-then-add tail. A plain loop would need a tolerance, and a
    /// tolerance is the size of a real defect here.
    ///
    /// Refuses when the runtime was built without host copies, which is the
    /// served case: a model that kept them would pay for its weights twice.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (ht, hp) = match (&self.rt.tables.host, &self.proj.host) {
            (Some(t), Some(p)) => (t, p),
            _ => candle_core::bail!(
                "{}: this runtime was built without host copies, so its CPU arm is not \
                 available. It is the oracle of a test, never a fallback for a served model",
                self.proj.name
            ),
        };
        self.check(storage.dtype(), layout)?;
        let x = match storage {
            CpuStorage::F32(v) => &v[layout.start_offset()..],
            other => candle_core::bail!("{}: f32 only, got {:?}", self.proj.name, other.dtype()),
        };
        let p = self.proj;
        let tile = self.rt.tile;
        let mut y = vec![0f32; p.d_out];
        let ntiles = p.nblocks.div_ceil(tile);

        for (row, out) in y.iter_mut().enumerate() {
            let mut lanes = [0f32; LANES];
            for t in 0..ntiles {
                let jlo = t * tile;
                let jhi = (jlo + tile).min(p.nblocks);
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut j = jlo + lane;
                    while j < jhi {
                        *acc += host_block(ht, hp, p, row, j, x);
                        j += LANES;
                    }
                }
            }
            for k in [16usize, 8, 4, 2, 1] {
                let mut next = [0f32; LANES];
                for (l, n) in next.iter_mut().enumerate() {
                    *n = lanes[l] + lanes[l ^ k];
                }
                lanes = next;
            }
            let mut tv = 0f32;
            let xt = &x[p.nblocks * llvq_core::DIM..];
            // `mul_add` at both sites: the CUDA twin contracts there under
            // NVRTC's `--fmad=true`, and the shader now spells it out.
            for (i, xi) in xt.iter().enumerate().take(p.tail_w) {
                tv = h2f(hp.tail[row * p.tail_w + i]).mul_add(*xi, tv);
            }
            *out = lanes[0].mul_add(hp.rscale[row], tv);
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((CpuStorage::F32(y), Shape::from(dims)))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.check(storage.dtype(), layout)?;
        let p = self.proj;
        let dev = storage.device().clone();
        // The ARITHMETIC variant, not the faithful port.
        //
        // `tv_tetra48_metal` mirrors the CUDA v3 decoder, which gathers its
        // coordinates with `prmt.b32`, one instruction. Metal has no PRMT, so
        // that gather costs a four-trip emulation called twelve times a block.
        // `tv_tetra48_metal_ar` computes the value instead, which is what the
        // SCALAR CUDA decoder always did, and measures 25 to 34 percent faster.
        // `_lut` then replaces the ten-operation computation with ONE indexed
        // load from 64 constant floats, and takes a further 9 to 16 percent. `_lutg`
        // moves those 64 floats from `constant` to `threadgroup`, worth 9 to
        // 18 more: `constant` is optimised for UNIFORM access and this index
        // varies per lane.
        // Cumulative against the faithful port: 19.85 ms a token to 12.66
        // (`llvq-metal/examples/metalsplit.rs`).
        //
        // Both are proven equal to the same host reference. The faithful port
        // is kept as the reference implementation and as the thing the
        // arithmetic one is diffed against.
        // `LLVQ_METAL_KERNEL` picks the matvec, for A/B only.
        //
        // A MEASUREMENT MODE, never a served setting. It exists because this
        // machine drifts: the same binary measured 49.7 tok/s in the morning
        // and 23.1 after three hours of compiling, so two numbers taken an
        // hour apart compare the thermals and not the kernels. Interleaving
        // two kernels in ONE process is the only honest A/B here.
        //
        // Unset, the served kernel is `_lut` and nothing changes.
        // The half-io twin stores f16 where the served kernel stores f32; the
        // adapter's narrowing cast then has nothing to do.
        let (kname, out_dtype) = match self.rt.half_io {
            false => (self.rt.kernel, DType::F32),
            true => (
                tetra_half_twin(self.rt.kernel).ok_or_else(|| {
                    candle_core::Error::msg(format!("half-io: {} has no half-store twin", self.rt.kernel))
                })?,
                DType::F16,
            ),
        };
        let pipe = self.rt.pipeline(kname)?;
        // The one-launch kernels' group; the row-looping ones size their own
        // below.
        let group = group_within(&pipe, kname, GROUP)?;

        let out = dev.new_buffer(p.d_out, out_dtype, "tetra48-matvec")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);

        // `start_offset()` is in ELEMENTS; the binding wants bytes.
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        let stride = p.row_stride_u32 as u32;
        let nb = p.nblocks as u32;
        let tw = p.tail_w as u32;
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, stride);
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
        set_param(&enc, 12, nb);
        set_param(&enc, 13, tw);
        set_param(&enc, 14, (&*self.rt.tables.c3_of, 0usize));

        // Not optional, and not an error if omitted: the kernel would run to
        // completion and write zeros. The tile in the length and the tile in
        // the source are the same number by construction, because both come
        // from `self.rt.tile`.
        if !tetra_reads_direct(kname) {
            let staged = self.rt.tile * llvq_core::DIM * DType::F32.size_in_bytes();
            enc.set_threadgroup_memory_length(0, staged);
        }

        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*self.rt.tables.c3_of,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);

        // The table-staging kernels: as many simdgroups a threadgroup as the
        // pipeline allows up to 32, each looping over rows with a guard, and
        // enough threadgroups for about TETRA_TG_LOOP_ROWS rows a simdgroup.
        if tetra_loops_rows(kname) {
            set_param(&enc, 16, p.d_out as u32);
            for (i, len) in TETRA_TG_TABLES.iter().enumerate() {
                enc.set_threadgroup_memory_length(i, *len);
            }
            let threads = (pipe.max_total_threads_per_threadgroup().min(1024) / LANES) * LANES;
            if threads == 0 {
                candle_core::bail!("{kname}: the pipeline allows fewer than {LANES} threads a threadgroup");
            }
            let nsg = threads / LANES;
            let ntg = p.d_out.div_ceil(nsg * TETRA_TG_LOOP_ROWS).max(1);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize { width: ntg, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
            );
            let mut dims = layout.dims().to_vec();
            *dims.last_mut().expect("rank >= 1") = p.d_out;
            return Ok((MetalStorage::new(out, dev, p.d_out, out_dtype), Shape::from(dims)));
        }

        // A multi-row twin covers R rows a simdgroup and carries no row guard,
        // like the served kernel: a d_out that eight simdgroups of R rows do
        // not tile would compute and store rows past the output.
        let rps = tetra_rows_per_simd(self.rt.kernel);
        if !p.d_out.is_multiple_of(rps * GROUP / LANES) {
            candle_core::bail!(
                "{}: d_out {} is not a multiple of {} rows, which {} covers a threadgroup",
                p.name,
                p.d_out,
                rps * GROUP / LANES,
                self.rt.kernel
            );
        }
        let grid = objc2_metal::MTLSize {
            width: p.d_out / rps * LANES,
            height: 1,
            depth: 1,
        };
        let group = objc2_metal::MTLSize {
            width: group,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, group);

        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((
            MetalStorage::new(out, dev, p.d_out, out_dtype),
            Shape::from(dims),
        ))
    }
}

/// One block's contribution, in the shader's order.
fn host_block(
    ht: &HostTables,
    hp: &HostProj,
    p: &MetalTetraProj,
    row: usize,
    j: usize,
    x: &[f32],
) -> f32 {
    let (lo, hi16) = host_load(&hp.words, p.row_stride_u32, row, j);
    let q = host_quads(ht, lo, hi16);
    let xb = &x[j * llvq_core::DIM..(j + 1) * llvq_core::DIM];
    let mut d = 0f32;
    for (i, quad) in q.iter().enumerate() {
        for k in 0..4 {
            let idx = ORDER[4 * i + k];
            let v = ((quad >> (8 * k)) & 0xff) as f32 - 128.0;
            d = v.mul_add(xb[idx], d);
        }
    }
    let n2: i32 = q
        .iter()
        .flat_map(|quad| (0..4).map(move |k| (((quad >> (8 * k)) & 0xff) as i32) - 128))
        .map(|v| v * v)
        .sum();
    let m = (((n2 as u32) >> 4) & 31) as usize;
    let g = ((hi16 >> 15) & 1) as usize;
    d * hp.gscale[g] * ht.invnorm[m]
}

/// `f1r_load`, on the host.
fn host_load(words: &[u8], row_stride_u32: usize, row: usize, j: usize) -> (u32, u32) {
    let base = row * row_stride_u32;
    let w = (3 * j) >> 1;
    let at = |i: usize| -> u32 {
        let o = (base + i) * 4;
        u32::from_le_bytes([words[o], words[o + 1], words[o + 2], words[o + 3]])
    };
    let w0 = at(w);
    let w1 = at(w + 1);
    match j & 1 {
        1 => ((w0 >> 16) | (w1 << 16), w1 >> 16),
        _ => (w0, w1 & 0xffff),
    }
}

/// CUDA's PRMT, default mode. See the shader's `prmt`.
fn prmt(a: u32, b: u32, s: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4 {
        let sel = (s >> (4 * i)) & 0xf;
        let idx = sel & 7;
        let src = if idx < 4 { a } else { b };
        let byte = (src >> (8 * (idx & 3))) & 0xff;
        let v = if sel & 8 != 0 {
            if byte & 0x80 != 0 {
                0xff
            } else {
                0
            }
        } else {
            byte
        };
        out |= v << (8 * i);
    }
    out
}

/// `f1r_v3_quads`, on the host.
fn host_quads(t: &HostTables, lo: u32, hi16: u32) -> [u32; 6] {
    const T: [[u32; 2]; 4] = [
        [0x887c8480, 0x90748c78],
        [0x79857d81, 0x718d7589],
        [0x7a867e82, 0x728e768a],
        [0x877b837f, 0x8f738b77],
    ];
    let p = lo & 1;
    let r = (lo >> 1) & 1;
    let s8 = (lo >> 2) & 63;
    let b1 = (lo >> 8) & 1;
    let i1 = (lo >> 9) & 0x7ff;
    let b2 = (lo >> 20) & 15;
    let i2 = ((lo >> 24) | ((hi16 & 7) << 8)) & 0x7ff;
    let b3 = (hi16 >> 3) & 1;
    let i3 = (hi16 >> 4) & 0x7ff;

    let c1 = t.prefixes[(2 * s8 + b1) as usize] as u32;
    let br = t.branches[(16 * s8 + b2) as usize] as u32;
    let c2 = br & 0xff;
    let s16 = (br >> 8) & 63;
    let c3 = t.suffixes[(2 * s16 + b3) as usize] as u32;

    let row1 = t.rows[(2048 * r + i1) as usize];
    let mid = i2 < 1240;
    let idx2 = if mid { i2 } else { (2048 - 1240) + i2 };
    let row2 = t.rows[idx2 as usize];
    let delta = u32::from(!mid);
    let r3 = (p ^ r ^ delta) & 1;
    let row3 = t.rows[(2048 * r3 + i3) as usize];

    let (c0lo, c0hi) = if p == 1 { (T[1][0], T[1][1]) } else { (T[0][0], T[0][1]) };
    let (c1lo, c1hi) = if p == 1 { (T[3][0], T[3][1]) } else { (T[2][0], T[2][1]) };
    let quad = |sel: u32, c4: u32| -> u32 {
        let m0 = prmt(c0lo, c0hi, sel);
        let m1 = prmt(c1lo, c1hi, sel);
        let k = ((c4.wrapping_mul(0x0020_4081)) & 0x0101_0101).wrapping_mul(0xff);
        (m0 & !k) | (m1 & k)
    };
    [
        quad(row1, c1 & 0xf),
        quad(row1 >> 16, c1 >> 4),
        quad(row2, c2 & 0xf),
        quad(row2 >> 16, c2 >> 4),
        quad(row3, c3 & 0xf),
        quad(row3 >> 16, c3 >> 4),
    ]
}
