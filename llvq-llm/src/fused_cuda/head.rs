//! The int4 g64 output head of an f32 model through `ink_head.cu`. One file
//! because it is one additive module with its own table and op.

use super::*;

// ---------------------------------------------------------------------------
// The int4 g64 output head of an f32 model (`ink_head.cu`), CUDA twin of
// `fused_metal::head_q4_f32`. Additive: a module of its own on the stream.
// ---------------------------------------------------------------------------

const INK_HEAD_CU_EMBED: &str = include_str!("../../kernels/ink_head.cu");

/// An int4 g64 table on the device with the kernel that projects through it.
pub struct Q4HeadF32 {
    pub name: String,
    pub rows: usize,
    pub d: usize,
    gpr: u32,
    wq: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    f: CudaFunction,
    _module: llvq_cuda::gpu::Cuda,
}

impl FusedRuntime {
    /// Upload an int4 g64 table of `rows` rows of `d` (`rows` a multiple of 8,
    /// `d` a multiple of 64) and compile its f32 head.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_head_q4_f32(
        &self,
        name: &str,
        rows: usize,
        d: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> candle_core::Result<Q4HeadF32> {
        if rows == 0 || !rows.is_multiple_of(8) || !d.is_multiple_of(64) || gpr != d / 64 {
            candle_core::bail!("{name}: {rows} rows of {d} with {gpr} groups a row: want rows % 8 == 0 and d % 64 == 0");
        }
        if wq.len() != rows * d / 8 || scales.len() != rows * gpr || biases.len() != rows * gpr {
            candle_core::bail!("{name}: payload of {} words, {} scales, {} biases for {rows} rows", wq.len(), scales.len(), biases.len());
        }
        let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu"]).map_err(candle_core::Error::msg)?;
        let (emb, _) = crate::fused::load_emb_q4_sources().map_err(candle_core::Error::msg)?;
        let defines = format!("{}{}", self.tile.define(), self.prefill.defines());
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(base.parts.iter().map(String::as_str))
            .chain([emb.as_str(), INK_HEAD_CU_EMBED])
            .collect();
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        println!("NVRTC head unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let module = llvq_cuda::gpu::Cuda::on_stream(self.device.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
        let f = module.func_dynamic_shared("tv_emb_q4_xf", (d * 4) as u32).map_err(candle_core::Error::msg)?;
        let up = |e: String| candle_core::Error::msg(format!("{name}: {e}"));
        Ok(Q4HeadF32 {
            name: name.to_string(),
            rows,
            d,
            gpr: gpr as u32,
            wq: module.up_u32(wq).map_err(up)?,
            scales: module.up_u16(scales).map_err(up)?,
            biases: module.up_u16(biases).map_err(up)?,
            f,
            _module: module,
        })
    }

    /// `h` `[.., d]` f32 through the table: f32 `[.., rows]`.
    pub fn head_q4_f32(&self, t: &Q4HeadF32, h: &Tensor) -> candle_core::Result<Tensor> {
        let d = *h.dims().last().expect("rank >= 1");
        if d != t.d {
            candle_core::bail!("{}: hidden of {d} for d={}", t.name, t.d);
        }
        let h = h.to_dtype(DType::F32)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadF32Op { rt: self, t })
    }
}

struct HeadF32Op<'a> {
    rt: &'a FusedRuntime,
    t: &'a Q4HeadF32,
}

impl candle_core::CustomOp1 for HeadF32Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-head-q4-f32"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the int4 f32 head has no CPU path")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let t = self.t;
        let x = storage.as_cuda_slice::<f32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden state", t.name)))?;
        let n = (end - start) / t.d;
        if n == 0 || (end - start) % t.d != 0 {
            candle_core::bail!("{}: {} values for rows of {}", t.name, end - start, t.d);
        }
        let stream = self.rt.device.cuda_stream();
        let mut y = unsafe { stream.alloc::<f32>(n * t.rows) }
            .map_err(|e| candle_core::Error::msg(format!("{}: alloc of the logits: {e}", t.name)))?;
        let cfg = LaunchConfig {
            grid_dim: ((t.rows as u32) * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: (t.d * 4) as u32,
        };
        let (d, gpr) = (t.d as u32, t.gpr);
        for r in 0..n {
            let (x_off, y_off) = ((start + r * t.d) as u32, (r * t.rows) as u32);
            let mut b = stream.launch_builder(&t.f);
            b.arg(&t.wq).arg(&t.scales).arg(&t.biases).arg(x).arg(&mut y).arg(&d).arg(&gpr).arg(&x_off).arg(&y_off);
            let probe = crate::inkling::probe_begin(|| stream.synchronize().is_ok());
                crate::inkling::count_launch();
                unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("tv_emb_q4_xf: {e}")))?;
                crate::inkling::probe_end(probe, "tv_emb_q4_xf", || stream.synchronize().is_ok());
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.rows;
        Ok((CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()), Shape::from(dims)))
    }
}
