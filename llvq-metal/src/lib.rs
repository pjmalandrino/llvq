//! Running a compute shader on Metal, and timing it honestly.
//!
//! The fused kernel's whole viability rests on one number: how many GPU
//! operations decoding a 24-weight block costs. Every estimate so far has been
//! an instruction count on paper, and the last one was wrong by 32× — I
//! counted simdgroup instructions against a budget expressed in lane-ops.
//! This crate exists to stop counting and start measuring.
//!
//! ## What "honestly" means here
//!
//! * **Warm-up runs are discarded.** The first dispatch pays shader
//!   compilation, buffer residency and clock ramp; including it would flatter
//!   or wreck the figure depending on the wind.
//! * **The result is read back and checked.** A kernel whose output nobody
//!   looks at is a kernel the compiler is free to delete, and a dead kernel
//!   benchmarks beautifully.
//! * **Submission overhead is measured and subtracted.** `metal-rs` 0.29 does
//!   not expose `GPUStartTime`, so timing is wall-clock around
//!   `commit()`/`wait_until_completed()` — which includes encoding, submission
//!   and synchronisation. [`Kernel::overhead`] measures that floor with an
//!   empty dispatch so it can be taken out, and every figure below is reported
//!   net of it. Dispatches are also sized to run for milliseconds, so the
//!   residue of that correction is noise rather than signal.

#[cfg(target_os = "macos")]
mod gpu {
    use metal::{
        CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;

    /// A compiled shader, ready to dispatch.
    pub struct Kernel {
        device: Device,
        queue: metal::CommandQueue,
        pipeline: ComputePipelineState,
    }

    /// What one timed dispatch cost.
    pub struct Timing {
        /// Wall-clock around commit + wait, seconds. Gross of submission
        /// overhead — see [`Kernel::overhead`].
        pub seconds: f64,
        /// Threads the dispatch actually ran.
        pub threads: u64,
    }

    impl Kernel {
        /// Compile `source` and take `name` out of it.
        pub fn new(source: &str, name: &str) -> Result<Self, String> {
            let device = Device::system_default().ok_or("no Metal device")?;
            let library = device
                .new_library_with_source(source, &CompileOptions::new())
                .map_err(|e| format!("shader compilation failed: {e}"))?;
            let function = library
                .get_function(name, None)
                .map_err(|e| format!("no function {name}: {e}"))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("pipeline creation failed: {e}"))?;
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                pipeline,
            })
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// Threads per threadgroup the hardware will actually accept.
        pub fn max_threads_per_group(&self) -> u64 {
            self.pipeline.max_total_threads_per_threadgroup() as u64
        }

        /// Width of a SIMD group — 32 on every Apple GPU, but read rather than
        /// assumed, because the whole lane-op accounting depends on it.
        pub fn simd_width(&self) -> u64 {
            self.pipeline.thread_execution_width() as u64
        }

        /// A shared-storage buffer holding `data`.
        pub fn buffer<T: Copy>(&self, data: &[T]) -> metal::Buffer {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(data) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// An uninitialised shared-storage buffer of `len` elements.
        pub fn empty<T>(&self, len: usize) -> metal::Buffer {
            self.device.new_buffer(
                (len * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// Read a buffer back as a slice.
        ///
        /// # Safety
        /// `buf` must hold at least `len` values of type `T`, written by a
        /// dispatch that has completed.
        pub unsafe fn read<T: Copy>(&self, buf: &metal::Buffer, len: usize) -> Vec<T> {
            std::slice::from_raw_parts(buf.contents() as *const T, len).to_vec()
        }

        /// Dispatch `threads` threads and return the GPU's own timing.
        ///
        /// `bind` receives the encoder so the caller can set buffers.
        pub fn dispatch(
            &self,
            threads: u64,
            group: u64,
            bind: impl Fn(&metal::ComputeCommandEncoderRef),
        ) -> Timing {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            bind(enc);
            enc.dispatch_threads(
                MTLSize::new(threads, 1, 1),
                MTLSize::new(group.min(self.max_threads_per_group()), 1, 1),
            );
            enc.end_encoding();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            Timing {
                seconds: t.elapsed().as_secs_f64(),
                threads,
            }
        }

        /// The floor: what a dispatch costs when it does nothing.
        ///
        /// Encoding, submission and synchronisation are paid whatever the
        /// shader does. Subtracting this is what turns a wall-clock number
        /// into something about the GPU rather than about the driver.
        pub fn overhead(&self, reps: usize) -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                best = best.min(self.dispatch(1, 1, |_| {}).seconds);
            }
            best
        }

        /// Dispatch `reps + warmup` times, discard the warm-ups, return the
        /// best timing.
        ///
        /// The **best** rather than the mean: a GPU shared with a compositor
        /// has a noise floor above it, never below, so the minimum is the
        /// closest thing to the machine's actual capability.
        pub fn time(
            &self,
            threads: u64,
            group: u64,
            warmup: usize,
            reps: usize,
            bind: impl Fn(&metal::ComputeCommandEncoderRef),
        ) -> Timing {
            for _ in 0..warmup {
                self.dispatch(threads, group, &bind);
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                best = best.min(self.dispatch(threads, group, &bind).seconds);
            }
            Timing {
                seconds: best,
                threads,
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{Kernel, Timing};

#[cfg(not(target_os = "macos"))]
compile_error!("llvq-metal targets Apple GPUs; there is nothing to measure elsewhere");
