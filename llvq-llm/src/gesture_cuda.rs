//! The decode gestures on the card: the key argmax `tok_argmax_h` (gestures
//! a2 and a3), its load check against candle's own kernel, and the probes of
//! `LLVQ_GESTURE_AB`.
//!
//! Nothing here is on a served path. The key kernel compiles as its own NVRTC
//! module onto candle's stream, so the served translation unit and its
//! register allocation do not move, and every launch is ordered with candle's
//! by issue on that one stream.

use std::sync::Mutex;
use std::time::Instant;

use candle_core::cuda_backend::cudarc::driver::{
    sys::CUevent_flags, CudaEvent, CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg,
};
use candle_core::{CudaStorage, DType, Device, IndexOp, Layout, Storage, Tensor, D};
use half::f16;

use crate::argmax_key;
use crate::model::KeyKernel;

fn cu<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> candle_core::Error {
    move |e| candle_core::Error::msg(format!("{what}: {e}"))
}

/// `tok_argmax_h` on candle's stream, with its device state: the running
/// max (`slot`), the block counter (`ticket`) and the sticky NaN `flag`.
pub struct KeyArgmax {
    cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    /// Behind a mutex because [`KeyKernel`] takes `&self`; uncontended, one
    /// lock a token.
    state: Mutex<KeyState>,
    /// Of the text NVRTC compiled: the pristine kernel or a mutant.
    pub sha256: String,
}

struct KeyState {
    slot: CudaSlice<u64>,
    ticket: CudaSlice<u32>,
    flag: CudaSlice<u32>,
}

impl KeyArgmax {
    /// The embedded kernel. `LLVQ_KERNEL_DIR` is not consulted: the card
    /// mutants are applied to this text in memory, and the mode that loads
    /// it refuses that variable.
    pub fn new(device: &Device) -> candle_core::Result<Self> {
        Self::from_text(device, argmax_key::TOK_ARGMAX_CU)
    }

    /// Any text that defines `tok_argmax_h`: the pristine kernel, or one of
    /// [`argmax_key::KERNEL_MUTANTS`] applied to it.
    pub fn from_text(device: &Device, text: &str) -> candle_core::Result<Self> {
        let stream = device.as_cuda_device()?.cuda_stream();
        let src = llvq_cuda::gpu::KernelSource::new(&[text]);
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;
        let f = cuda.func("tok_argmax_h").map_err(candle_core::Error::msg)?;
        let st = cuda.stream();
        let state = KeyState {
            slot: st.alloc_zeros::<u64>(1).map_err(cu("key slot"))?,
            ticket: st.alloc_zeros::<u32>(1).map_err(cu("key ticket"))?,
            flag: st.alloc_zeros::<u32>(1).map_err(cu("key flag"))?,
        };
        Ok(Self { cuda, f, state: Mutex::new(state), sha256: src.sha256 })
    }

    fn lock(&self) -> candle_core::Result<std::sync::MutexGuard<'_, KeyState>> {
        self.state.lock().map_err(|_| candle_core::Error::msg("tok_argmax_h: state lock poisoned"))
    }

    /// One launch: the argmax of `x[x_off .. x_off + n]` into `ring[pos]`.
    fn launch(
        &self,
        x: &CudaSlice<f16>,
        x_off: usize,
        n: usize,
        ring: &mut CudaSlice<u32>,
        pos: usize,
    ) -> candle_core::Result<()> {
        argmax_key::key_preconditions(n, x_off).map_err(candle_core::Error::msg)?;
        if x_off + n > x.len() || pos >= ring.len() {
            candle_core::bail!(
                "tok_argmax_h: {n} logits at {x_off} in a buffer of {}, ring slot {pos} of {}",
                x.len(),
                ring.len()
            );
        }
        let stream = self.cuda.stream();
        {
            let (ptr, _record) = x.device_ptr(stream);
            if ptr % 16 != 0 {
                candle_core::bail!("tok_argmax_h: logits buffer at {ptr:#x}, not 16-byte aligned");
            }
        }
        let mut st = self.lock()?;
        let KeyState { slot, ticket, flag } = &mut *st;
        let cfg = LaunchConfig {
            grid_dim: (argmax_key::key_grid(n), 1, 1),
            block_dim: (argmax_key::KEY_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (x_off, n, pos) = (x_off as u32, n as u32, pos as u32);
        let mut b = stream.launch_builder(&self.f);
        b.arg(x).arg(&x_off).arg(&n).arg(slot).arg(ticket).arg(flag).arg(ring).arg(&pos);
        unsafe { b.launch(cfg) }.map_err(cu("tok_argmax_h"))?;
        Ok(())
    }

    /// The card check: [`argmax_key::card_sequence`] in one sequence through
    /// one slot, ticket and flag, each case against candle's own `argmax` on
    /// the same device tensor (`fast_argmax_f16`) and against the host replay.
    ///
    /// `Ok(Ok(line))` passes. `Ok(Err(why))` is a mismatch, what every card
    /// mutant must produce. `Err` is a CUDA error, which voids the run rather
    /// than counting as a kill.
    pub fn check(&self, device: &Device) -> candle_core::Result<Result<String, String>> {
        let cases = argmax_key::card_sequence();
        let ring = Tensor::from_vec(vec![u32::MAX; cases.len()], cases.len(), device)?;
        self.arm()?;
        let mut candle_ids = Vec::with_capacity(cases.len());
        let mut flag_before_nan = None;
        for (k, c) in cases.iter().enumerate() {
            let vals: Vec<f16> = c.bits.iter().map(|&b| f16::from_bits(b)).collect();
            let x = Tensor::from_vec(vals, (1, 1, c.bits.len()), device)?;
            if c.nan {
                flag_before_nan = Some(self.nan_seen()?);
            } else {
                candle_ids.push(x.i((0, 0))?.argmax_keepdim(D::Minus1)?);
            }
            self.write(&x, &ring, k)?;
        }
        let got = ring.to_vec1::<u32>()?;
        let flag = self.nan_seen()?;
        let candle = Tensor::cat(&candle_ids, 0)?.to_vec1::<u32>()?;
        self.arm()?;
        if flag_before_nan != Some(false) {
            return Ok(Err(format!("the NaN flag read {flag_before_nan:?} before the NaN case")));
        }
        let mut non_nan = 0;
        for (k, c) in cases.iter().enumerate() {
            if c.nan {
                if !flag || got[k] != 0 {
                    return Ok(Err(format!(
                        "case {} ({}): flag {flag}, token {}; want the flag raised and token 0",
                        k + 1,
                        c.name,
                        got[k]
                    )));
                }
                continue;
            }
            let on_card = candle[non_nan];
            non_nan += 1;
            let widened: Vec<f32> = c.bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
            let replay = argmax_key::candle_fast_argmax(&widened);
            if on_card != replay || c.want.is_some_and(|w| w != on_card) {
                return Ok(Err(format!(
                    "case {} ({}): candle's kernel {on_card}, the host replay {replay}, pinned {:?}. \
                     The REFERENCE disagrees with itself: the replay or the pin is wrong",
                    k + 1,
                    c.name,
                    c.want
                )));
            }
            if got[k] != on_card {
                return Ok(Err(format!(
                    "case {} ({}): tok_argmax_h {} against candle's {on_card}",
                    k + 1,
                    c.name,
                    got[k]
                )));
            }
        }
        Ok(Ok(format!(
            "tok_argmax: candle's fast_argmax_f16 index on {non_nan} cases in sequence, NaN flag raised"
        )))
    }
}

/// The key kernel as an in-place op on the ring: no allocation, no upload.
struct KeyOp<'a> {
    k: &'a KeyArgmax,
    pos: usize,
}

impl candle_core::InplaceOp2 for KeyOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-tok-argmax"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("tok_argmax_h has no CPU path")
    }

    fn cuda_fwd(&self, ring: &mut CudaStorage, rl: &Layout, x: &CudaStorage, xl: &Layout) -> candle_core::Result<()> {
        let (rs, re) = rl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("tok_argmax_h: non-contiguous ring"))?;
        if self.pos >= re - rs {
            candle_core::bail!("tok_argmax_h: slot {} of a ring of {}", self.pos, re - rs);
        }
        let (xs, xe) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("tok_argmax_h: non-contiguous logits"))?;
        let x = x.as_cuda_slice::<f16>()?;
        let ring = ring.as_cuda_slice_mut::<u32>()?;
        self.k.launch(x, xs, xe - xs, ring, rs + self.pos)
    }
}

impl KeyKernel for KeyArgmax {
    fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> candle_core::Result<()> {
        if logits.dtype() != DType::F16 || ring.dtype() != DType::U32 {
            candle_core::bail!(
                "tok_argmax_h reads f16 logits into a u32 ring, given {:?} and {:?}",
                logits.dtype(),
                ring.dtype()
            );
        }
        ring.inplace_op2(logits, &KeyOp { k: self, pos })
    }

    fn arm(&self) -> candle_core::Result<()> {
        let mut st = self.lock()?;
        self.cuda.stream().memset_zeros(&mut st.flag).map_err(cu("key flag"))
    }

    fn nan_seen(&self) -> candle_core::Result<bool> {
        let st = self.lock()?;
        let v = self.cuda.stream().clone_dtoh(&st.flag).map_err(cu("key flag"))?;
        Ok(v[0] != 0)
    }
}

/// P2: every card mutant, compiled as its own module, must fail the check.
/// Returns one line a mutant, or an error naming the survivor.
pub fn kill_card_mutants(device: &Device) -> candle_core::Result<Vec<String>> {
    let mut lines = Vec::new();
    for m in &argmax_key::KERNEL_MUTANTS {
        let text = argmax_key::apply_mutant(argmax_key::TOK_ARGMAX_CU, m).map_err(candle_core::Error::msg)?;
        let k = KeyArgmax::from_text(device, &text)?;
        match k.check(device)? {
            Ok(_) => candle_core::bail!("card mutant {} SURVIVED the 41-call check: the run is void", m.name),
            Err(why) => lines.push(format!(
                "  {}: {} edit(s), each found once; killed, {why} (expected: {})",
                m.name,
                m.edits.len(),
                m.killed_by
            )),
        }
    }
    Ok(lines)
}

/// The measurement kernels of `LLVQ_GESTURE_AB`: the spin, and timing events.
pub struct GestureProbe {
    cuda: llvq_cuda::gpu::Cuda,
    f_spin: CudaFunction,
}

const SPIN_CU: &str = include_str!("../kernels/gesture_spin.cu");

/// Per-launch device times of P3, microseconds, medians of 5 batches of 100.
pub struct KernelTimes {
    pub argmax_f16_direct: f64,
    pub argmax_f32_direct: f64,
    pub cast: f64,
    pub key: f64,
    pub candle_argmax_call: f64,
}

/// Host times of P4, milliseconds, medians of 5, each behind a 5 ms spin.
pub struct CopyTimes {
    pub from_slice: f64,
    pub candle_argmax: f64,
    pub to_scalar: f64,
}

pub const PROBE_REPS: usize = 5;
pub const PROBE_BATCH: usize = 100;
pub const P4_SPIN_NS: u64 = 5_000_000;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The f16 storage and offset of a contiguous tensor on the card.
fn with_f16<R>(t: &Tensor, f: impl FnOnce(&CudaSlice<f16>, usize, usize) -> candle_core::Result<R>) -> candle_core::Result<R> {
    let (st, l) = t.storage_and_layout();
    let (s, e) = l.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("non-contiguous"))?;
    match &*st {
        Storage::Cuda(c) => f(c.as_cuda_slice::<f16>()?, s, e - s),
        _ => candle_core::bail!("not on the card"),
    }
}

fn with_f32<R>(t: &Tensor, f: impl FnOnce(&CudaSlice<f32>, usize, usize) -> candle_core::Result<R>) -> candle_core::Result<R> {
    let (st, l) = t.storage_and_layout();
    let (s, e) = l.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("non-contiguous"))?;
    match &*st {
        Storage::Cuda(c) => f(c.as_cuda_slice::<f32>()?, s, e - s),
        _ => candle_core::bail!("not on the card"),
    }
}

impl GestureProbe {
    pub fn new(device: &Device) -> candle_core::Result<Self> {
        let stream = device.as_cuda_device()?.cuda_stream();
        let src = llvq_cuda::gpu::KernelSource::new(&[SPIN_CU]);
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;
        let f_spin = cuda.func("spin_ns").map_err(candle_core::Error::msg)?;
        Ok(Self { cuda, f_spin })
    }

    /// Hold candle's stream for `ns` nanoseconds.
    pub fn spin(&self, ns: u64) -> candle_core::Result<()> {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.cuda.stream().launch_builder(&self.f_spin);
        b.arg(&ns);
        unsafe { b.launch(cfg) }.map_err(cu("spin_ns"))?;
        Ok(())
    }

    /// A timing event (`CU_EVENT_DEFAULT`: `None` would disable timing).
    pub fn event(&self) -> candle_core::Result<CudaEvent> {
        self.cuda
            .stream()
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(cu("event"))
    }

    pub fn record(&self, ev: &CudaEvent) -> candle_core::Result<()> {
        ev.record(self.cuda.stream()).map_err(cu("event record"))
    }

    pub fn sync(&self) -> candle_core::Result<()> {
        self.cuda.stream().synchronize().map_err(cu("sync"))
    }

    /// Microseconds a launch: `PROBE_BATCH` back-to-back calls of `body`
    /// between two events, `PROBE_REPS` times, the median.
    fn per_launch(&self, mut body: impl FnMut() -> candle_core::Result<()>) -> candle_core::Result<f64> {
        let (e0, e1) = (self.event()?, self.event()?);
        let mut us = Vec::with_capacity(PROBE_REPS);
        for _ in 0..PROBE_REPS {
            self.sync()?;
            self.record(&e0)?;
            for _ in 0..PROBE_BATCH {
                body()?;
            }
            self.record(&e1)?;
            let ms = e0.elapsed_ms(&e1).map_err(cu("event elapsed"))?;
            us.push(ms as f64 * 1e3 / PROBE_BATCH as f64);
        }
        let m = median(&mut us);
        if !m.is_finite() {
            candle_core::bail!("P3: a non-finite time");
        }
        Ok(m)
    }

    /// P3, on the real `[1, 1, vocab]` f16 logits: candle's `fast_argmax_f16`
    /// and `fast_argmax_f32` launched directly with their info uploaded once,
    /// the f32 cast, `tok_argmax_h`, and candle's full `argmax` call.
    pub fn kernel_times(&self, device: &Device, key: &KeyArgmax, logits: &Tensor) -> candle_core::Result<KernelTimes> {
        let dev = device.as_cuda_device()?;
        let row = logits.i((0, 0))?.contiguous()?;
        let row32 = row.to_dtype(DType::F32)?;
        let n = row.elem_count();
        let stream = self.cuda.stream();
        let info: CudaSlice<usize> = stream.clone_htod(&[n, 1usize][..]).map_err(cu("argmax info"))?;
        let mut out: CudaSlice<u32> = stream.alloc_zeros(1).map_err(cu("argmax out"))?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (src_el, per_block, ndims) = (n, n, 1usize);
        let f16_fn = dev.get_or_load_func("fast_argmax_f16", &candle_core::cuda_backend::kernels::REDUCE)?;
        let f32_fn = dev.get_or_load_func("fast_argmax_f32", &candle_core::cuda_backend::kernels::REDUCE)?;
        let argmax_f16_direct = with_f16(&row, |x, s, e| {
            let x = x.slice(s..s + e);
            self.per_launch(|| {
                let mut b = f16_fn.builder();
                b.arg(&src_el).arg(&per_block).arg(&ndims).arg(&info).arg(&x).arg(&mut out);
                unsafe { b.launch(cfg) }.map_err(cu("fast_argmax_f16"))?;
                Ok(())
            })
        })?;
        let argmax_f32_direct = with_f32(&row32, |x, s, e| {
            let x = x.slice(s..s + e);
            self.per_launch(|| {
                let mut b = f32_fn.builder();
                b.arg(&src_el).arg(&per_block).arg(&ndims).arg(&info).arg(&x).arg(&mut out);
                unsafe { b.launch(cfg) }.map_err(cu("fast_argmax_f32"))?;
                Ok(())
            })
        })?;
        let cast = self.per_launch(|| logits.to_dtype(DType::F32).map(drop))?;
        let scratch = Tensor::zeros(1, DType::U32, device)?;
        let key_us = self.per_launch(|| key.write(logits, &scratch, 0))?;
        let candle_argmax_call = self.per_launch(|| row.argmax(D::Minus1).map(drop))?;
        // The direct launches and the full call must agree with the key on
        // these logits, or the times belong to different answers.
        let direct = stream.clone_dtoh(&out).map_err(cu("argmax out"))?[0];
        let via_key = scratch.i(0)?.to_scalar::<u32>()?;
        let via_candle = row.argmax(D::Minus1)?.to_scalar::<u32>()?;
        if direct != via_candle || via_key != via_candle {
            candle_core::bail!(
                "P3: candle direct {direct}, candle call {via_candle}, key {via_key}: the probe's \
                 launches do not compute the same token"
            );
        }
        Ok(KernelTimes { argmax_f16_direct, argmax_f32_direct, cast, key: key_us, candle_argmax_call })
    }

    /// P4: the host time of three calls, each behind a 5 ms spin on candle's
    /// stream. Microseconds mean the call did not wait for the stream; about
    /// 5 ms means it did. `to_scalar` is the control, which must wait.
    pub fn copy_times(&self, device: &Device, logits: &Tensor) -> candle_core::Result<CopyTimes> {
        let row = logits.i((0, 0))?;
        let one = Tensor::new(7u32, device)?;
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for _ in 0..PROBE_REPS {
            self.sync()?;
            self.spin(P4_SPIN_NS)?;
            let t = Instant::now();
            let up = Tensor::from_slice(&[7u32], (1, 1), device)?;
            a.push(t.elapsed().as_secs_f64() * 1e3);
            self.sync()?;
            drop(up);

            self.spin(P4_SPIN_NS)?;
            let t = Instant::now();
            let am = row.argmax(D::Minus1)?;
            b.push(t.elapsed().as_secs_f64() * 1e3);
            self.sync()?;
            drop(am);

            self.spin(P4_SPIN_NS)?;
            let t = Instant::now();
            let v = one.to_scalar::<u32>()?;
            c.push(t.elapsed().as_secs_f64() * 1e3);
            self.sync()?;
            if v != 7 {
                candle_core::bail!("P4: the control read {v}, not 7");
            }
        }
        Ok(CopyTimes { from_slice: median(&mut a), candle_argmax: median(&mut b), to_scalar: median(&mut c) })
    }
}

/// The spun clock's hook: after ln1 of each named layer it closes the open
/// segment (an event and a host time), launches the next spin, and opens the
/// next segment. Segment 0 opens at the step start, the last closes after
/// the key kernel; the driver does both ends.
pub struct SpinCapture<'a> {
    pub probe: &'a GestureProbe,
    pub ns: u64,
    pub at_layers: Vec<usize>,
    pub ev_start: Vec<CudaEvent>,
    pub ev_end: Vec<CudaEvent>,
    pub t_spin: Vec<Instant>,
    pub t_done: Vec<Instant>,
}

impl<'a> SpinCapture<'a> {
    pub fn new(probe: &'a GestureProbe, ns: u64, at_layers: Vec<usize>) -> candle_core::Result<Self> {
        let segs = at_layers.len() + 1;
        let now = Instant::now();
        Ok(Self {
            probe,
            ns,
            ev_start: (0..segs).map(|_| probe.event()).collect::<candle_core::Result<_>>()?,
            ev_end: (0..segs).map(|_| probe.event()).collect::<candle_core::Result<_>>()?,
            t_spin: vec![now; segs],
            t_done: vec![now; segs],
            at_layers,
        })
    }

    /// Open segment `k`: a spin, its host time, its start event.
    pub fn open(&mut self, k: usize) -> candle_core::Result<()> {
        self.probe.spin(self.ns)?;
        self.t_spin[k] = Instant::now();
        self.probe.record(&self.ev_start[k])
    }

    /// Close segment `k`: its end event and its host time.
    pub fn close(&mut self, k: usize) -> candle_core::Result<()> {
        self.probe.record(&self.ev_end[k])?;
        self.t_done[k] = Instant::now();
        Ok(())
    }

    /// After a synchronize: (host enqueue span H, device time D) in ms, and
    /// whether every segment was enqueued at least 1 ms before its spin
    /// could end, which is what makes D the device's work alone.
    pub fn read(&self) -> candle_core::Result<(f64, f64, bool)> {
        let (mut h, mut d, mut valid) = (0.0, 0.0, true);
        let margin = self.ns as f64 / 1e6 - 1.0;
        for k in 0..self.ev_start.len() {
            let hk = self.t_done[k].duration_since(self.t_spin[k]).as_secs_f64() * 1e3;
            valid &= hk < margin;
            h += hk;
            d += self.ev_start[k].elapsed_ms(&self.ev_end[k]).map_err(cu("event elapsed"))? as f64;
        }
        Ok((h, d, valid))
    }
}

impl crate::model::Capture for SpinCapture<'_> {
    fn on_activation(&mut self, layer: usize, act: crate::model::Act, _x: &Tensor) -> candle_core::Result<()> {
        if act == crate::model::Act::Attn {
            if let Some(k) = self.at_layers.iter().position(|&l| l == layer) {
                self.close(k)?;
                self.open(k + 1)?;
            }
        }
        Ok(())
    }
}
