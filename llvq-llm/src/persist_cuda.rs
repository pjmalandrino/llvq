//! Route P, step A, on CUDA: the MLP half of a one-row decode step in one
//! cooperative launch per layer (`llvq-cuda/kernels/pers_segment.cu`).
//!
//! Design: `docs/archive/debit-vague5-2026-11-02/persistent.md` §3.1, §3.5-3.7.
//! Host proof: `llvq-cuda/tests/pers_matches_served.rs` (the segment equals the
//! eager chain bit for bit on the Mac, its barrier and residual mutants killed).
//!
//! Measurement only. Nothing installs a segment unless `bin/fusedrun` is asked
//! for `LLVQ_PERS_SEGMENT=1`; every served path runs the eager ops.
//!
//! What install checks, layer by layer, before a segment may replace the eager
//! ops: the projections are the ones the kernel reads (Tetra48 gate and up under
//! one shared rotation; a Tetra48 or int4 down), the activation is SiLU, the
//! dtype is f16, the kernel neither spills nor fails its occupancy, and the
//! segment's output equals [`Block::mlp_eager`]'s bit for bit on several rows.
//! A layer that fails any of them keeps the eager ops, and the line printed
//! says which and why. SiLU itself is never reproduced: its 65,536 outputs are
//! filled by running the block's own activation, candle's kernel, on every
//! binary16 pattern.
//!
//! [`Block::mlp_eager`]: crate::model::Block::mlp_eager

use std::sync::{Arc, Mutex};

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Device, Layout, Shape, Storage, Tensor};
use half::f16;

use crate::fused_cuda::{CudaInt4, CudaLattice, TetraView};
use crate::model::{MlpSegment, Proj, Qwen3};
use llvq_cuda::pers;

/// Threads a CTA, the `__launch_bounds__` of the segment kernels.
const THREADS: u32 = 768;

struct Runtime {
    cuda: llvq_cuda::gpu::Cuda,
    f_tetra: CudaFunction,
    f_q4: CudaFunction,
    smem: u32,
    grid: u32,
    /// Held, read by the kernel only: its address is in every layer's SegArgs.
    #[allow(dead_code)]
    bar: CudaSlice<u32>,
    abort: CudaSlice<u32>,
    /// Arrivals on `bar` so far: the base of the next launch's target.
    bar_base: Mutex<u32>,
    _held: Vec<CudaSlice<u16>>,
    _held32: Vec<CudaSlice<u32>>,
}

/// One layer's segment.
pub struct Segment {
    rt: Arc<Runtime>,
    args: CudaSlice<u8>,
    tetra_down: bool,
    hidden: usize,
}

impl MlpSegment for Segment {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.contiguous()?.apply_op1_no_bwd(&SegOp { seg: self })
    }
}

struct SegOp<'a> {
    seg: &'a Segment,
}

impl candle_core::CustomOp1 for SegOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-pers-mlp-segment"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("route P's segment has no CPU path")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let seg = self.seg;
        let rt = &seg.rt;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("route P: non-contiguous residual"))?;
        if end - start != seg.hidden {
            candle_core::bail!("route P: a residual of {} values for hidden {}", end - start, seg.hidden);
        }
        let stream = rt.cuda.stream();
        let x = storage.as_cuda_slice::<f16>()?;
        let x_ptr = x.device_ptr(stream).0 + 2 * start as u64;
        // SAFETY: every value is written by the kernel's last phase (each row
        // of `hidden` belongs to exactly one CTA's slice) before it is read.
        let out = unsafe { stream.alloc::<f16>(seg.hidden) }.map_err(|e| candle_core::Error::msg(format!("alloc: {e}")))?;
        let out_ptr = out.device_ptr(stream).0;
        let bar_base = {
            let mut b = rt.bar_base.lock().map_err(|_| candle_core::Error::msg("route P: poisoned ledger"))?;
            let v = *b;
            *b = b.wrapping_add(rt.grid);
            v
        };
        let launch = pers::SegLaunch { x: x_ptr, out: out_ptr, bar_base, pad: 0 };
        let f = if seg.tetra_down { &rt.f_tetra } else { &rt.f_q4 };
        let cfg = LaunchConfig { grid_dim: (rt.grid, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: rt.smem };
        let mut b = stream.launch_builder(f);
        b.arg(&seg.args).arg(&launch);
        // SAFETY: the kernel's own signature (SegArgs*, SegLaunch), checked by
        // the layout test; one CTA a SM, which passed the occupancy check; a
        // cooperative launch refuses a grid that cannot be co-resident.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| candle_core::Error::msg(format!("pers_seg: {e}")))?;
        Ok((CudaStorage::wrap_cuda_slice(out, storage.device.clone()), layout.shape().clone()))
    }
}

fn tensor_addr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> candle_core::Result<u64> {
    let (st, lay) = t.storage_and_layout();
    match &*st {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<f16>()?.device_ptr(stream).0 + 2 * lay.start_offset() as u64),
        _ => candle_core::bail!("route P: a tensor that is not on the card"),
    }
}

fn tetra_of(p: &Proj) -> Option<(TetraView, &CudaLattice)> {
    let Proj::Lattice(d) = p else { return None };
    let c = d.as_any()?.downcast_ref::<CudaLattice>()?;
    Some((c.tetra_view()?, c))
}

fn int4_of(p: &Proj) -> Option<&CudaInt4> {
    let Proj::Int4(d) = p else { return None };
    d.as_any()?.downcast_ref::<CudaInt4>()
}

fn seg_tetra(v: &TetraView) -> pers::SegTetraHost {
    pers::SegTetraHost {
        words: v.words,
        rscale: v.rscale,
        tail: v.tail,
        gscale: v.gscale,
        stride_u32: v.stride_u32,
        nblocks: v.nblocks,
        tail_w: v.tail_w,
        d_out: v.d_out,
    }
}

/// What [`install`] built: one segment per layer (`None` = the eager ops),
/// the report lines, and the runtime whose abort flag says whether any launch
/// gave up on its watchdog.
pub struct Installed {
    pub segs: Vec<Option<Arc<dyn MlpSegment>>>,
    pub lines: Vec<String>,
    rt: Arc<Runtime>,
}

impl Installed {
    /// Put the checked segments on the model.
    pub fn set(&self, model: &mut Qwen3) {
        for (b, s) in model.blocks.iter_mut().zip(&self.segs) {
            b.set_pers_mlp(s.clone());
        }
    }

    /// Whether any segment launch aborted on its watchdog. Read after a run,
    /// never inside a timed one: it synchronises.
    pub fn aborted(&self) -> anyhow::Result<bool> {
        Ok(self.rt.cuda.down_u32(&self.rt.abort).map_err(anyhow::Error::msg)?[0] != 0)
    }
}

/// Build one segment per layer and check each against the eager ops. Nothing
/// is put on the model: [`Installed::set`] does that.
pub fn install(model: &Qwen3, device: &Device) -> anyhow::Result<Installed> {
    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let mut lines = Vec::new();
    let first = model.blocks.first().ok_or_else(|| anyhow::anyhow!("route P: a model with no block"))?;
    let (_, _, _, gate0, _, _) = first.mlp_parts();
    let (_, lat0) = tetra_of(gate0).ok_or_else(|| anyhow::anyhow!("route P: layer 0's gate_proj is not a Tetra48 stream on CUDA"))?;
    let frt = lat0.runtime().clone();
    let tables = frt.table_view().ok_or_else(|| anyhow::anyhow!("route P: the runtime carries no Tetra tables"))?;
    let (hidden, inter) = (lat0.tetra_view().map_or(0, |v| v.d_in), gate0.d_out()? as u32);

    let text = pers::pers_unit_text(&frt.tile_define());
    let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
    lines.push(format!("route P: persistent module {} bytes, sha256 {}, own module on candle's stream", src.text.len(), src.sha256));
    let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;
    let smem = (pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * hidden.max(inter)).div_ceil(16) * 16;
    let f_tetra = cuda.func_dynamic_shared("pers_seg_tetra_v0", smem).map_err(anyhow::Error::msg)?;
    let f_q4 = cuda.func_dynamic_shared("pers_seg_q4_v0", smem).map_err(anyhow::Error::msg)?;
    for name in ["pers_seg_tetra_v0", "pers_seg_q4_v0"] {
        let r = cuda.report(name).map_err(anyhow::Error::msg)?;
        lines.push(format!("route P: {name}: {} registers, {} B local, {smem} B shared", r.num_regs, r.local_bytes));
        if r.local_bytes != 0 {
            anyhow::bail!("route P: {name} spills {} B; the segment is not installed", r.local_bytes);
        }
    }
    let occ = f_tetra
        .occupancy_max_active_blocks_per_multiprocessor(THREADS, smem as usize, None)
        .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?
        .min(f_q4.occupancy_max_active_blocks_per_multiprocessor(THREADS, smem as usize, None).map_err(|e| anyhow::anyhow!("occupancy: {e}"))?);
    if occ < 1 {
        anyhow::bail!("route P: no CTA of {THREADS} threads with {smem} B fits a SM");
    }
    let grid = cuda.device().map_err(anyhow::Error::msg)?.sm_count as u32;

    // SiLU by exhaustion: the block's own activation, candle's kernel, on
    // every binary16 pattern.
    let (_, _, act0, _, _, _) = first.mlp_parts();
    if !matches!(act0, candle_nn::Activation::Silu) {
        anyhow::bail!("route P: the activation is {act0:?}, the segment carries SiLU");
    }
    let pats: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
    let silu_t = Tensor::from_vec(pats, 65536, device)?.apply(&act0)?;
    let silu_bits: Vec<u16> = silu_t.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
    let silu = cuda.up_u16(&silu_bits).map_err(anyhow::Error::msg)?;
    let act = cuda.zeros_u16(inter as usize).map_err(anyhow::Error::msg)?;
    let bar = cuda.up_u32(&[0]).map_err(anyhow::Error::msg)?;
    let abort = cuda.up_u32(&[0]).map_err(anyhow::Error::msg)?;
    let gu: Vec<u32> = pers::slices(inter, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let dn: Vec<u32> = pers::slices(hidden, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let d_gu = cuda.up_u32(&gu).map_err(anyhow::Error::msg)?;
    let d_dn = cuda.up_u32(&dn).map_err(anyhow::Error::msg)?;
    let addr32 = |s: &CudaSlice<u32>| s.device_ptr(&stream).0;
    let addr16 = |s: &CudaSlice<u16>| s.device_ptr(&stream).0;
    let base = pers::SegArgsHost {
        silu: addr16(&silu),
        act: addr16(&act),
        tables: [tables.rows, tables.pref, tables.br, tables.suf, tables.invnorm],
        slice_gu: addr32(&d_gu),
        slice_dn: addr32(&d_dn),
        bar: addr32(&bar),
        abort_flag: addr32(&abort),
        hidden,
        inter,
        ..Default::default()
    };
    let rt = Arc::new(Runtime {
        cuda,
        f_tetra,
        f_q4,
        smem,
        grid,
        bar,
        abort,
        bar_base: Mutex::new(0),
        _held: vec![silu, act],
        _held32: vec![d_gu, d_dn],
    });

    let mut segs: Vec<Option<Arc<dyn MlpSegment>>> = Vec::with_capacity(model.blocks.len());
    let mut refused = Vec::new();
    for (l, block) in model.blocks.iter().enumerate() {
        let built = (|| -> anyhow::Result<Segment> {
            let (ln2_w, eps, _, gate, up, down) = block.mlp_parts();
            if ln2_w.dtype() != DType::F16 {
                anyhow::bail!("ln2 is {:?}, the segment reads binary16", ln2_w.dtype());
            }
            let (g, _) = tetra_of(gate).ok_or_else(|| anyhow::anyhow!("gate_proj is not a Tetra48 stream"))?;
            let (u, _) = tetra_of(up).ok_or_else(|| anyhow::anyhow!("up_proj is not a Tetra48 stream"))?;
            let key = g.rotation.ok_or_else(|| anyhow::anyhow!("gate_proj carries no rotation"))?;
            if u.rotation != Some(key) {
                anyhow::bail!("gate and up are not under one rotation (LLVQ_ROT_SHARE=1 is the served form)");
            }
            let rh = frt.rot_view(&key).ok_or_else(|| anyhow::anyhow!("no rotation buffers for gate/up"))?;
            if pers::rot_mk(rh.n).map_err(anyhow::Error::msg)? != (rh.m, rh.k) || rh.n != hidden {
                anyhow::bail!("the gate/up rotation is {}x{} at width {}", rh.m, rh.k, rh.n);
            }
            let mut a = base.clone();
            a.alpha = tensor_addr(ln2_w, &stream)?;
            a.eps = eps as f32;
            a.signs_h = rh.signbits;
            a.small_h = rh.small;
            a.m_h = rh.m;
            a.k_h = rh.k;
            a.inv_h = rh.inv;
            a.gate = seg_tetra(&g);
            a.up = seg_tetra(&u);
            let tetra_down = if let Some((d, _)) = tetra_of(down) {
                let dk = d.rotation.ok_or_else(|| anyhow::anyhow!("down_proj carries no rotation"))?;
                let ri = frt.rot_view(&dk).ok_or_else(|| anyhow::anyhow!("no rotation buffers for down"))?;
                if ri.n != inter || pers::rot_mk(ri.n).map_err(anyhow::Error::msg)? != (ri.m, ri.k) {
                    anyhow::bail!("the down rotation is {}x{} at width {}", ri.m, ri.k, ri.n);
                }
                a.down = seg_tetra(&d);
                a.signs_i = ri.signbits;
                a.small_i = ri.small;
                a.m_i = ri.m;
                a.k_i = ri.k;
                a.inv_i = ri.inv;
                true
            } else if let Some(q) = int4_of(down) {
                let v = q.int4_view();
                if v.d_in != inter || v.d_out != hidden || v.gpr != inter.div_ceil(128) {
                    anyhow::bail!("the int4 down_proj is {}x{} with {} groups a row", v.d_out, v.d_in, v.gpr);
                }
                a.q4w = v.wq;
                a.q4s = v.scales;
                a.q4b = v.biases;
                a.gpr = v.gpr;
                false
            } else {
                anyhow::bail!("down_proj is neither a Tetra48 stream nor an int4 record");
            };
            let args = rt.cuda.stream().clone_htod(&a.bytes()).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            Ok(Segment { rt: rt.clone(), args, tetra_down, hidden: hidden as usize })
        })();
        let seg = match built {
            Ok(s) => s,
            Err(e) => {
                refused.push(format!("layer {l}: {e}"));
                segs.push(None);
                continue;
            }
        };
        // The load check: the segment against the eager ops, bit for bit,
        // on rows of plausible residual magnitude, several seeds.
        let mut same = true;
        for seed in 0..4u64 {
            let mut rng = llvq_core::SplitMix64::new(0x5e9_0000 + 97 * l as u64 + seed);
            let xs: Vec<f16> = (0..hidden).map(|_| f16::from_f64(rng.next_gaussian() * (1.0 + seed as f64))).collect();
            let x = Tensor::from_vec(xs, (1, 1, hidden as usize), device)?;
            let want: Vec<u16> = block.mlp_eager(&x)?.flatten_all()?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
            let got: Vec<u16> = seg.forward(&x)?.flatten_all()?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
            if rt.cuda.down_u32(&rt.abort).map_err(anyhow::Error::msg)?[0] != 0 {
                anyhow::bail!("route P: the segment of layer {l} aborted on its watchdog at install");
            }
            if got != want {
                let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                refused.push(format!("layer {l}: differs from the eager ops at row {i} (seed {seed}): {:#06x} against {:#06x}", got[i], want[i]));
                same = false;
                break;
            }
        }
        segs.push(same.then(|| Arc::new(seg) as Arc<dyn MlpSegment>));
    }
    let n = segs.iter().filter(|s| s.is_some()).count();
    lines.push(format!(
        "route P: {n} of {} layers equal the eager MLP bit for bit at install (4 rows each), grid {grid} x {THREADS}",
        segs.len()
    ));
    lines.extend(refused);
    Ok(Installed { segs, lines, rt })
}

/// Back to the eager ops on every layer.
pub fn clear(model: &mut Qwen3) {
    for b in model.blocks.iter_mut() {
        b.set_pers_mlp(None);
    }
}
