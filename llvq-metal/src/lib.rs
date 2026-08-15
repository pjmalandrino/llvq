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
            self.pipeline.max_total_threads_per_threadgroup()
        }

        /// Width of a SIMD group — 32 on every Apple GPU, but read rather than
        /// assumed, because the whole lane-op accounting depends on it.
        pub fn simd_width(&self) -> u64 {
            self.pipeline.thread_execution_width()
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

        /// Encode `k` sequential dispatches of the same kernel in **one**
        /// command buffer and return the wall-clock of the whole buffer.
        ///
        /// This exists for kernels whose single run is the same order as the
        /// submission overhead (~0.15 ms): a matvec is ~150 µs, and
        /// subtracting a number from a number of its own size leaves noise.
        /// With k of them back to back the measurement is milliseconds and
        /// the commit overhead is subtracted once. The per-encoder cost
        /// stays in — real inference also pays one dispatch per layer op.
        ///
        /// ⚠️ What serializes the passes is **not** the encoder boundary —
        /// Apple GPUs overlap hazard-free passes of one command buffer.
        /// It is the write-write hazard on the *shared, tracked output
        /// buffer* every dispatch binds. A caller that gave each dispatch
        /// its own output (or moved buffers into an untracked heap) would
        /// let the passes overlap and measure something much prettier and
        /// entirely false. Keep one output buffer.
        ///
        /// `bind` receives the dispatch index so callers can rotate *input*
        /// buffers across dispatches — the cache-defeat protocol: replaying
        /// one resident weight buffer 32× measures the system-level cache,
        /// not the DRAM streaming regime real inference lives in.
        pub fn dispatch_many(
            &self,
            threads: u64,
            group: u64,
            k: usize,
            bind: impl Fn(&metal::ComputeCommandEncoderRef, usize),
        ) -> Timing {
            let cmd = self.queue.new_command_buffer();
            for i in 0..k {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline);
                bind(enc, i);
                enc.dispatch_threads(
                    MTLSize::new(threads, 1, 1),
                    MTLSize::new(group.min(self.max_threads_per_group()), 1, 1),
                );
                enc.end_encoding();
            }
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            Timing {
                seconds: t.elapsed().as_secs_f64(),
                threads,
            }
        }

        /// Encode `n` **different** dispatches — one per work item — into a
        /// single command buffer, and time the whole thing.
        ///
        /// `bind` receives the encoder and the item index, and returns that
        /// item's `(threads, threadgroup)`. This is the shape of a real
        /// decode step: many distinct matrices, each read once, one commit.
        /// Because every item has its own weights, the pass is cold by
        /// construction — no rotation trick needed.
        ///
        /// Serialization comes from the write-write hazard on the shared
        /// output buffer, as in [`Self::dispatch_many`]; give each dispatch
        /// its own output and the passes would overlap and lie.
        pub fn dispatch_batch(
            &self,
            bind: impl Fn(&metal::ComputeCommandEncoderRef, usize) -> (u64, u64),
            n: usize,
        ) -> Timing {
            let cmd = self.queue.new_command_buffer();
            let mut total = 0u64;
            for i in 0..n {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline);
                let (threads, group) = bind(enc, i);
                enc.dispatch_threads(
                    MTLSize::new(threads, 1, 1),
                    MTLSize::new(group.min(self.max_threads_per_group()), 1, 1),
                );
                enc.end_encoding();
                total += threads;
            }
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            Timing {
                seconds: t.elapsed().as_secs_f64(),
                threads: total,
            }
        }

        /// [`Self::dispatch_many`] with warm-up and best-of-`reps`, like
        /// [`Self::time`].
        pub fn time_many(
            &self,
            threads: u64,
            group: u64,
            k: usize,
            warmup: usize,
            reps: usize,
            bind: impl Fn(&metal::ComputeCommandEncoderRef, usize),
        ) -> Timing {
            for _ in 0..warmup {
                self.dispatch_many(threads, group, k, &bind);
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                best = best.min(self.dispatch_many(threads, group, k, &bind).seconds);
            }
            Timing {
                seconds: best,
                threads,
            }
        }

        /// Bytes of threadgroup memory handed to `[[threadgroup(i)]]`.
        pub fn set_threadgroup_memory(
            enc: &metal::ComputeCommandEncoderRef,
            index: u64,
            bytes: u64,
        ) {
            enc.set_threadgroup_memory_length(index, bytes);
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

/// Host tables for the two P1 arms — the `#[repr(C)]` mirrors and class
/// tables `shaders/cascade_uniform.metal` and `shaders/binomial_walk.metal`
/// assume. They live in this crate, beside [`PAYLOAD_MSL`] and
/// [`gpu_class_table`], so a mirror and the struct it mirrors cannot drift
/// apart in two crates.
pub mod p1host;

/// f32 → IEEE 754 binary16 bits, round to nearest even. No `half` crate —
/// this crate stays at two dependencies.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        // Inf / NaN.
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00; // overflow to inf
    }
    if e16 <= 0 {
        // Subnormal (or zero): shift the implicit-1 mantissa down.
        if e16 < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = man >> shift;
        // Round to nearest even.
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent — that is correct rounding
    }
    v as u16
}

pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f64;
    sign * match exp {
        0 => man * (-24f64).exp2(),
        31 => {
            if man == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + man / 1024.0) * ((e as f64) - 15.0).exp2(),
    }
}

/// GPU-side class record; must match [`PAYLOAD_MSL`]'s `ClassRec` byte for
/// byte — which is why it lives in the same crate as that source.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuClassRec {
    /// Level values divided by the shell norm — a block's weights are
    /// `vals[level] * gscale[gain]`, the row scale folded by the caller.
    pub vals: [f32; 5],
    pub counts: [u8; 4],
    pub len: u8,
    pub nz: u8,
    /// Index of the zero level, 255 if the class has none.
    pub zlev: u8,
    /// Sign base per level for the level-major sign order of `Flat32`:
    /// `sbase[k]` = Σ counts of nonzero levels before `k`.
    pub sbase: [u8; 5],
}

/// The 384-entry table of [`PAYLOAD_MSL`]'s decoders: entry 0 the origin,
/// entry `1 + ci` class `ci` — the id convention of
/// `llvq_artifact::runtime::ClassTable`.
pub fn gpu_class_table(fd: &llvq_search::fastdec::FastDecoder) -> Vec<GpuClassRec> {
    let mut recs = Vec::with_capacity(fd.n_classes() + 1);
    recs.push(GpuClassRec {
        vals: [0.0; 5],
        counts: [24, 0, 0, 0],
        len: 1,
        nz: 0,
        zlev: 0,
        sbase: [0; 5],
    });
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let mut vals = [0.0f32; 5];
        for (v, &raw) in vals.iter_mut().zip(&lv.values[..lv.len]) {
            *v = (raw as f64 / norm) as f32;
        }
        let zlev = lv.values[..lv.len]
            .iter()
            .position(|&v| v == 0)
            .map_or(255, |z| z as u8);
        let mut sbase = [0u8; 5];
        let mut acc = 0u8;
        for (k, sb) in sbase.iter_mut().enumerate().take(lv.len) {
            *sb = acc;
            if lv.values[k] != 0 {
                acc += lv.counts[k];
            }
        }
        recs.push(GpuClassRec {
            vals,
            counts: [lv.counts[0], lv.counts[1], lv.counts[2], lv.counts[3]],
            len: lv.len as u8,
            nz: lv.nonzero,
            zlev,
            sbase,
        });
    }
    recs
}

/// Metal source of the runtime-payload decoder, shared verbatim by every bin
/// that reads the frozen format (`decreal`, `matvec`). One copy, so a bench
/// and the kernel it justifies can never drift apart.
///
/// Contract with the Rust side: `ClassRec` mirrors `decreal`'s
/// `GpuClassRec` byte for byte, and the payload layout is
/// `[class 9][gain 1][signs nz][nested masks]`, LSB-first — the format
/// pinned by `llvq-artifact/tests/runtime_format.rs`.
pub const PAYLOAD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint DIM = 24;

// Mirror of the Rust-side record. 32 bytes; `vals` are the level values
// already divided by the shell norm, so a block's weights are
// vals[level] * gscale[gain] (row scale folded elsewhere).
struct ClassRec {
    float vals[5];
    uchar counts[4];
    uchar len;
    uchar nz;
    uchar zlev;      // index of the zero level, 255 if the class has none
    uchar sbase[5];  // level-major sign base per level (Flat32 payloads)
};

// ---------------------------------------------------------------------------
// A 96-bit cursor in registers. Fields are ≤ 32 bits wide; width 0 is legal
// (the origin's empty sign field) and must not shift by 64.
// ---------------------------------------------------------------------------
struct Cursor { ulong lo; uint hi; };

static inline uint take(thread Cursor &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (ulong(c.hi) << (64u - width));
    c.hi = c.hi >> width;
    return v;
}

// ---------------------------------------------------------------------------
// The payload decode both layouts share, from an aligned cursor.
//
// The level chain is the popcount form the synthetic bench measured: the
// mask of the block's *last* level is replaced by all-ones, so the chain
// needs no per-level count checks — it stops where the class says it stops.
// The sign index is a running nonzero counter, free in the serial walk.
// ---------------------------------------------------------------------------
static inline float decode_payload(Cursor c,
                                   constant ClassRec* tab,
                                   constant float* gscale,
                                   threadgroup const float* xs)
{
    uint id   = take(c, 9);
    uint gain = take(c, 1);            // the published 4B: 1 gain bit
    constant ClassRec &r = tab[id];
    uint len = r.len;
    uint signs = take(c, r.nz);

    uint left = DIM;
    uint m0 = 0xffffffu, m1 = 0xffffffu, m2 = 0xffffffu, m3 = 0xffffffu;
    if (len > 1) { m0 = take(c, left); left -= r.counts[0];
        if (len > 2) { m1 = take(c, left); left -= r.counts[1];
            if (len > 3) { m2 = take(c, left); left -= r.counts[2];
                if (len > 4) { m3 = take(c, left); }
                else m3 = 0xffffffu;
            } else { m2 = 0xffffffu; }
        } else { m1 = 0xffffffu; }
    } else { m0 = 0xffffffu; }

    float dot = 0.0f;
    uint nz = 0;
    for (uint i = 0; i < DIM; ++i) {
        uint b0 = 1u << i;
        float v;
        if (m0 & b0) {
            v = r.vals[0];
        } else {
            uint p1 = popcount(~m0 & (b0 - 1u));
            uint b1 = 1u << p1;
            if (m1 & b1) {
                v = r.vals[1];
            } else {
                uint p2 = popcount(~m1 & (b1 - 1u));
                uint b2 = 1u << p2;
                if (m2 & b2) {
                    v = r.vals[2];
                } else {
                    uint p3 = popcount(~m2 & (b2 - 1u));
                    v = (m3 & (1u << p3)) ? r.vals[3] : r.vals[4];
                }
            }
        }
        if (v != 0.0f) {
            if ((signs >> nz) & 1u) v = -v;
            ++nz;
        }
        dot += v * xs[i];
    }
    return dot * gscale[gain];
}

// ---------------------------------------------------------------------------
// 24 bits at an absolute offset of a 128-bit register pair. `off` here is
// always 24/48/72/96, so the funnel shifts never degenerate.
// ---------------------------------------------------------------------------
static inline uint ext24(ulong lo, ulong hi, uint off) {
    ulong v = (off < 64u) ? ((lo >> off) | (hi << (64u - off)))
                          : (hi >> (off - 64u));
    return uint(v) & 0xffffffu;
}

// ---------------------------------------------------------------------------
// Slot32 — the production decoder. Every field at a fixed offset:
// [class 9][gain 1][smask 24][m1..m4 @ 24]. A fixed 24-slot loop with no nz,
// no zero level, no per-level walk, no serial state: the level comes from
// four slot-mask tests (absent masks are zero, level 0 is the default), the
// sign from one bit. Zero divergence, four independent FMA chains.
//
// `xb` points at the block's 24 activations. The caller owns the staging;
// with a large d_in that must be tiled, since threadgroup memory is 32 KB.
// ---------------------------------------------------------------------------
static inline float slot_dot(device const uint* words,
                             device const uint* bases,
                             constant ClassRec* tab,
                             constant float* gscale,
                             uint b,
                             threadgroup const float* xb)
{
    uint g = b >> 5;
    uint base = bases[g];
    uint stride = (bases[g + 1] - base) >> 5;
    uint byte = base + (b & 31u) * stride;

    // Five words: the worst case is a 24-bit byte-alignment shift plus a
    // 130-bit payload = 154 bits.
    uint w = byte >> 2;
    uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
         w3 = words[w + 3], w4 = words[w + 4];
    uint sh = (byte & 3u) * 8u;
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    ulong hi = (ulong(w3) << 32) | ulong(w2);

    // The 10-bit header is read in place (sh + 10 <= 34 < 64); its shift
    // folds into the alignment shift, and the remaining <= 120 bits land in
    // two registers.
    uint hdr  = uint(lo >> sh) & 0x3ffu;
    uint id   = hdr & 0x1ffu;
    uint gain = hdr >> 9;
    uint fs = sh + 10u;
    ulong pay_lo = (lo >> fs) | (hi << (64u - fs));
    ulong pay_hi = (hi >> fs) | (ulong(w4) << (64u - fs));

    constant ClassRec &r = tab[id];
    uint smask = uint(pay_lo) & 0xffffffu;
    uint nlev = r.len;
    uint m1 = (nlev > 1) ? ext24(pay_lo, pay_hi, 24u) : 0u;
    uint m2 = (nlev > 2) ? ext24(pay_lo, pay_hi, 48u) : 0u;
    uint m3 = (nlev > 3) ? ext24(pay_lo, pay_hi, 72u) : 0u;
    uint m4 = (nlev > 4) ? ext24(pay_lo, pay_hi, 96u) : 0u;
    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2],
          v3 = r.vals[3], v4 = r.vals[4];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (uint i = 0; i < DIM; i += 4) {
        for (uint k = 0; k < 4; ++k) {
            uint j = i + k;
            uint bj = 1u << j;
            float v = (m1 & bj) ? v1 : (m2 & bj) ? v2 : (m3 & bj) ? v3
                    : (m4 & bj) ? v4 : v0;
            v = (smask & bj) ? -v : v;
            switch (k) {
                case 0: d0 = fma(v, xb[j], d0); break;
                case 1: d1 = fma(v, xb[j], d1); break;
                case 2: d2 = fma(v, xb[j], d2); break;
                default: d3 = fma(v, xb[j], d3); break;
            }
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[gain];
}

// ---------------------------------------------------------------------------
// Grouped32 addressing: the block's bytes start anywhere, so four aligned
// words are read and funnel-shifted down to an aligned cursor.
// ---------------------------------------------------------------------------
static inline Cursor cursor_g32(device const uint* words,
                                device const uint* bases,
                                uint b)
{
    uint g = b >> 5;
    uint base = bases[g];
    uint stride = (bases[g + 1] - base) >> 5;
    uint byte = base + (b & 31u) * stride;

    uint w = byte >> 2;
    uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2], w3 = words[w + 3];
    uint sh = (byte & 3u) * 8u;
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    uint  hi;
    if (sh == 0) {
        hi = w2;
    } else {
        lo = (lo >> sh) | (ulong(w2) << (64u - sh));
        hi = (w2 >> sh) | (w3 << (32u - sh));
    }
    return Cursor { lo, hi };
}

// ---------------------------------------------------------------------------
// Grouped32 — one block, block-local activations. Same shape as `slot_dot`,
// so a caller can swap layouts without touching its tiling: `xb` points at
// this block's 24 floats and nothing else is read from shared memory.
// ---------------------------------------------------------------------------
static inline float g32_dot(device const uint* words,
                            device const uint* bases,
                            constant ClassRec* tab,
                            constant float* gscale,
                            uint b,
                            threadgroup const float* xb)
{
    return decode_payload(cursor_g32(words, bases, b), tab, gscale, xb);
}

// ---------------------------------------------------------------------------
// A 160-bit cursor: the Flat32 worst case is a 130-bit payload plus a byte
// shift of up to 24.
// ---------------------------------------------------------------------------
struct Cur5 { ulong lo; ulong hi; uint xt; };

static inline uint take5(thread Cur5 &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (c.hi << (64u - width));
    c.hi = (c.hi >> width) | (ulong(c.xt) << (64u - width));
    c.xt = c.xt >> width;
    return v;
}

// One level's contribution: walk its set bits, shifting signs out as we go.
static inline float lrun(uint m, float v, uint s, threadgroup const float* xb) {
    float d = 0.0f;
    while (m) {
        uint i = ctz(m);
        m &= m - 1u;
        d = fma((s & 1u) ? -v : v, xb[i], d);
        s >>= 1u;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Flat32 — slot-space masks (level 0 implicit) with a level-major sign field.
// The nested payload pays a prefix popcount per slot; this one comes off its
// word with ctz, signs shift out sequentially, and zero slots are never
// touched. Block-local `xb`, like the two above.
// ---------------------------------------------------------------------------
static inline float flat_dot(device const uint* words,
                             device const uint* bases,
                             constant ClassRec* tab,
                             constant float* gscale,
                             uint b,
                             threadgroup const float* xb)
{
    uint g = b >> 5;
    uint base = bases[g];
    uint stride = (bases[g + 1] - base) >> 5;
    uint byte = base + (b & 31u) * stride;

    uint w = byte >> 2;
    uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
         w3 = words[w + 3], w4 = words[w + 4];
    uint sh = (byte & 3u) * 8u;
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    ulong hi = (ulong(w3) << 32) | ulong(w2);
    uint xt = w4;
    if (sh != 0) {
        lo = (lo >> sh) | (hi << (64u - sh));
        hi = (hi >> sh) | (ulong(xt) << (64u - sh));
        xt >>= sh;
    }
    Cur5 c = { lo, hi, xt };

    uint id   = take5(c, 9);
    uint gain = take5(c, 1);
    constant ClassRec &r = tab[id];
    uint nlev = r.len;
    uint signs = take5(c, r.nz);
    uint m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    if (nlev > 1) { m1 = take5(c, 24);
        if (nlev > 2) { m2 = take5(c, 24);
            if (nlev > 3) { m3 = take5(c, 24);
                if (nlev > 4) { m4 = take5(c, 24); } } } }
    uint m0 = ~(m1 | m2 | m3 | m4) & 0xffffffu;

    uint zl = r.zlev;
    float d = 0.0f;
    if (zl != 0u) d += lrun(m0, r.vals[0], signs >> r.sbase[0], xb);
    if (nlev > 1 && zl != 1u) d += lrun(m1, r.vals[1], signs >> r.sbase[1], xb);
    if (nlev > 2 && zl != 2u) d += lrun(m2, r.vals[2], signs >> r.sbase[2], xb);
    if (nlev > 3 && zl != 3u) d += lrun(m3, r.vals[3], signs >> r.sbase[3], xb);
    if (nlev > 4 && zl != 4u) d += lrun(m4, r.vals[4], signs >> r.sbase[4], xb);
    return d * gscale[gain];
}
"#;

#[cfg(not(target_os = "macos"))]
compile_error!("llvq-metal targets Apple GPUs; there is nothing to measure elsewhere");
