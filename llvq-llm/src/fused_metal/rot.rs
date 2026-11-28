//! The rotation: its tables on the device and the Walsh-Hadamard launch as a candle op. One file because the
//! rotation is one kernel with one upload, shared by every projection that names its key.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::{MetalRuntime, LANES, THREADGROUP_LIMIT};

// ---------------------------------------------------------------------------
// The rotation.
//
// `prepare` owes this: the weights were quantized in a rotated basis, so the
// activation has to arrive in it. The kernel is a Walsh-Hadamard transform
// with a per-coordinate sign flip, then either a scale-out or a small mix.
//
// One thing differs from CUDA and it is not cosmetic. There the staging is a
// `__shared__` array, per block and free. Here it needs `n` floats a row, and
// `n` reaches 9,728 on the served 4B, which is 38,912 B against Apple's 32,768
// of threadgroup memory. So the scratch is a DEVICE buffer and every stage of
// the transform goes through global memory. Nothing about that cost is
// measured; it is the first thing to look at if the Metal throughput
// disappoints.
// ---------------------------------------------------------------------------

/// One rotation, uploaded. Shared by every projection that names its key.
pub struct MetalRotation {
    pub(super) signbits: Arc<Buffer>,
    pub(super) small: Arc<Buffer>,
    pub(super) n: u32,
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) inv: f32,
    /// Threads a threadgroup, the same rule the CUDA host uses.
    pub(super) threads: usize,
}

impl MetalRuntime {
    /// Upload one rotation's tables.
    pub fn upload_rotation(&self, t: &crate::fused::RotationTables) -> Result<MetalRotation> {
        // Metal refuses a zero-length buffer. `small` is empty when k == 1,
        // which is the common case: `rot_mix` is not reached then and the
        // kernel never reads it.
        let dummy = [0f32];
        let small = if t.small.is_empty() { &dummy[..] } else { &t.small };
        Ok(MetalRotation {
            signbits: self.device.new_buffer_with_data(&t.signbits)?,
            small: self.device.new_buffer_with_data(small)?,
            n: t.n as u32,
            m: t.m as u32,
            k: t.k as u32,
            inv: t.inv,
            threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
        })
    }

    /// `x` into the basis the weights were quantized in, one vector.
    pub fn rotate(&self, rot: &MetalRotation, name: &str, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{name}: rotation requested for {rows} vectors. The row loop belongs to \
                 the caller, which shares it across a group"
            );
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: 1 })
    }

    /// The same for a chunk of rows, one threadgroup each.
    pub fn rotate_rows(&self, rot: &MetalRotation, name: &str, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{name}: {rows} rows of activation for {n_rows}");
        }
        if n_rows == 0 {
            candle_core::bail!("{name}: a rotation of zero rows");
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: n_rows })
    }
}

struct RotOp<'a> {
    rt: &'a MetalRuntime,
    rot: &'a MetalRotation,
    name: String,
    rows: usize,
}

impl CustomOp1 for RotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-rot"
    }

    /// No CPU path, and deliberately none.
    ///
    /// Unlike the matvec, this op has no oracle to be: the rotation is judged
    /// by `llvq-metal/tests/rot_matches_host.rs` against the CUDA text itself,
    /// executed. A second Rust transcription here would be a third place for
    /// the same mistake to live.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the LLVQ rotation has no CPU path", self.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the rotation reads f16, got {:?}", self.name, storage.dtype());
        }
        let n = self.rot.n as usize;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", self.name)))?;
        if end - start != n * self.rows {
            candle_core::bail!(
                "{}: activation of {} values for {} rows of n={n}",
                self.name,
                end - start,
                self.rows
            );
        }
        let batched = self.rows > 1;
        let dev = storage.device().clone();
        let out = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot")?;
        use candle_metal_kernels::utils::set_param;

        // One row that fits: the vector staged in threadgroup memory, no
        // scratch, the same stages.
        let staged = n * DType::F32.size_in_bytes();
        if self.rt.rot_tg && !batched && staged <= THREADGROUP_LIMIT {
            let pipe = self.rt.pipeline("rot_apply_tg_metal")?;
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (storage.buffer(), 0usize));
            set_param(&enc, 1, (&*self.rot.signbits, 0usize));
            set_param(&enc, 2, (&*self.rot.small, 0usize));
            set_param(&enc, 3, (&*out, 0usize));
            set_param(&enc, 5, self.rot.n);
            set_param(&enc, 6, self.rot.m);
            set_param(&enc, 7, self.rot.k);
            set_param(&enc, 8, self.rot.inv);
            set_param(&enc, 9, start as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [storage.buffer(), &*self.rot.signbits, &*self.rot.small] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            // One threadgroup, never larger than the pipeline takes. The
            // kernel strides every loop by its thread count, so any count
            // gives the same bits.
            let threads = (self.rot.threads.min(pipe.max_total_threads_per_threadgroup()) / LANES) * LANES;
            if threads == 0 {
                candle_core::bail!("rot_apply_tg_metal: the pipeline allows fewer than {LANES} threads a threadgroup");
            }
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
            );
            return Ok((MetalStorage::new(out, dev, n, DType::F32), Shape::from(vec![1, n])));
        }

        let kname = if batched { "rot_apply_rows_metal" } else { "rot_apply_metal" };
        let pipe = self.rt.pipeline(kname)?;
        // The staging, in DEVICE memory. See the section header.
        let scratch = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot-scratch")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        // `x_off` is in ELEMENTS here, not bytes: the kernel indexes
        // `xin[x_off + i]` on a ushort pointer. The Tetra matvec takes a byte
        // offset instead, because it binds at the offset. Two kernels, two
        // conventions, each matched to its own source.
        set_param(&enc, 0, (storage.buffer(), 0usize));
        set_param(&enc, 1, (&*self.rot.signbits, 0usize));
        set_param(&enc, 2, (&*self.rot.small, 0usize));
        set_param(&enc, 3, (&*out, 0usize));
        set_param(&enc, 4, (&*scratch, 0usize));
        set_param(&enc, 5, self.rot.n);
        set_param(&enc, 6, self.rot.m);
        set_param(&enc, 7, self.rot.k);
        set_param(&enc, 8, self.rot.inv);
        set_param(&enc, 9, start as u32);
        if batched {
            set_param(&enc, 10, n as u32);
        }
        for b in [storage.buffer(), &*self.rot.signbits, &*self.rot.small] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        // One threadgroup a row: a group larger than the pipeline takes would
        // split a row over two groups on one scratch. The kernel strides every
        // loop by its thread count, so any count gives the same bits.
        let threads = (self.rot.threads.min(pipe.max_total_threads_per_threadgroup()) / LANES) * LANES;
        if threads == 0 {
            candle_core::bail!("{kname}: the pipeline allows fewer than {LANES} threads a threadgroup");
        }
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * self.rows, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.rows * n, DType::F32),
            Shape::from(vec![self.rows, n]),
        ))
    }
}
