//! The batched step on a card: [`crate::batch::BatchDevice`] for CUDA.
//!
//! Two NVRTC modules of their own, compiled once, so neither the served
//! projection unit nor the served attention module changes a byte:
//!
//! * `tv_batch_mma.cu` after the decoder headers it reads, for the
//!   tensor-core projections (Tetra, int4 records, the int4 head);
//! * `tv_batch_gemm.cu`, the same headers, for the staged and folded GEMM of
//!   [`crate::batch::BatchProjs::Gemm`], in a module of its own so the first
//!   kernel's unit keeps its text;
//! * `attn_dec.cu` then `attn_dec_batch.cu`, for the attention of B
//!   sequences in one launch, each with the served kernel's bits.
//!
//! The device buffers of a projection are reached through the views 4E reads
//! (`tetra_view`, `int4_view`, `q4_view`), so nothing here owns weights.

use std::sync::{Arc, Mutex};

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::backend::BackendStorage;
use candle_core::{CudaStorage, Device, Layout, Shape, Storage, Tensor};
use half::f16;

use crate::fused_cuda::{CudaEmbed, CudaInt4, CudaLattice, FusedRuntime, TableView};

const MMA_CU: &str = include_str!("../kernels/tv_batch_mma.cu");
const GEMM_CU: &str = include_str!("../kernels/tv_batch_gemm.cu");
const ATTN_BATCH_CU: &str = include_str!("../../llvq-cuda/kernels/attn_dec_batch.cu");
/// Rows a CTA of the tensor-core kernels owns; `d_out` must be a multiple.
const BM_ROWS: usize = 128;
/// Activation rows the widest entry point of `tv_batch_mma.cu` takes.
pub const MAX_ROWS: usize = 32;
/// Activation rows the widest entry point of `tv_batch_gemm.cu` takes.
pub const GEMM_MAX_ROWS: usize = 128;
/// Rows a CTA of the GEMM owns, and projections a launch carries.
const BG_ROWS: usize = 128;
/// Threads of a GEMM CTA: 8 warps of 16 rows.
const BG_THREADS: u32 = 256;
const BG_MAXSEG: usize = 4;

/// The GEMM's entries: rows they carry and the K chunk each was compiled
/// with (`BG_ENTRY` lines of `tv_batch_gemm.cu`).
const BG_ENTRIES: [(&str, usize, usize); 5] =
    [("tv_bg_n8", 8, 192), ("tv_bg_n16", 16, 192), ("tv_bg_n32", 32, 192), ("tv_bg_n64", 64, 96), ("tv_bg_n128", 128, 96)];
/// The two-CTA entries of the measurement arm `gemm-occ`.
const BG_OCC_ENTRIES: [(&str, usize, usize); 5] =
    [("tv_bgo_n8", 8, 192), ("tv_bgo_n16", 16, 96), ("tv_bgo_n32", 32, 96), ("tv_bgo_n64", 64, 48), ("tv_bgo_n128", 128, 48)];

/// Shared bytes of a GEMM entry: `BG_SMEM_WORDS` of the kernel, transcribed.
fn bg_smem_bytes(n: usize, kc: usize) -> u32 {
    let stage = |kind: usize| {
        let ww = match kind {
            0 => (kc / 16 + 1) / 4 * 4 + 2,
            _ => kc / 8 + if kc > 48 { 2 } else { 0 },
        };
        n * (kc / 2 + 4) + BG_ROWS * ww
    };
    ((2 * stage(0).max(stage(1)) + 8 * 16 * 28) * 4) as u32
}

/// `BgSeg` of `tv_batch_gemm.cu`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct BgSeg {
    w: u64,
    a0: u64,
    a1: u64,
    tail: u64,
    x: u64,
    y: u64,
    part: u64,
    kind: u32,
    stride: u32,
    nblocks: u32,
    tail_w: u32,
    d_in: u32,
    d_out: u32,
    nk: u32,
    slice: u32,
    s: u32,
    cta0: u32,
    tk0: u32,
    tiled: u32,
}

/// `BgArgs` of `tv_batch_gemm.cu`, passed by value as the kernel's one
/// parameter.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct BgArgs {
    seg: [BgSeg; BG_MAXSEG],
    trows: u64,
    pref: u64,
    br: u64,
    suf: u64,
    invnorm: u64,
    ticket: u64,
    nseg: u32,
    n_rows: u32,
}

// SAFETY: plain old data, `#[repr(C)]`, the layout of the kernel's struct
// (the sizes are asserted in `tests`); the driver copies its bytes.
unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for BgArgs {}

fn addr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream, elem: u64) -> candle_core::Result<u64> {
    let (st, lay) = t.storage_and_layout();
    if !lay.is_contiguous() {
        candle_core::bail!("batched kernel: a non-contiguous tensor");
    }
    let base = match &*st {
        Storage::Cuda(cs) => match cs.dtype() {
            candle_core::DType::F16 => cs.as_cuda_slice::<f16>()?.device_ptr(stream).0,
            candle_core::DType::F32 => cs.as_cuda_slice::<f32>()?.device_ptr(stream).0,
            candle_core::DType::U32 => cs.as_cuda_slice::<u32>()?.device_ptr(stream).0,
            d => candle_core::bail!("batched kernel: a {d:?} tensor"),
        },
        _ => candle_core::bail!("batched kernel: a tensor that is not on the card"),
    };
    Ok(base + elem * lay.start_offset() as u64)
}

fn slice_addr<T>(s: &CudaSlice<T>, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> u64 {
    s.device_ptr(stream).0
}

/// One launch whose output is a fresh f16 tensor: a `CustomOp1` on an input
/// the launch reads, so candle allocates nothing it then overwrites.
struct Launch<'a> {
    name: &'static str,
    out_elems: usize,
    out_shape: Shape,
    run: &'a dyn Fn(u64) -> candle_core::Result<()>,
}

impl candle_core::CustomOp1 for Launch<'_> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("{}: no CPU path", self.name)
    }
    fn cuda_fwd(&self, s: &CudaStorage, _: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let stream = s.device.cuda_stream();
        let y = unsafe { stream.alloc::<f16>(self.out_elems) }
            .map_err(|e| candle_core::Error::msg(format!("{}: alloc: {e}", self.name)))?;
        (self.run)(slice_addr(&y, &stream))?;
        Ok((CudaStorage::wrap_cuda_slice(y, s.device.clone()), self.out_shape.clone()))
    }
}

struct Scratch {
    gemm_part: CudaSlice<f32>,
    gemm_ticket: CudaSlice<u32>,
    part: CudaSlice<f32>,
    ticket: CudaSlice<u32>,
    attn_part: CudaSlice<f32>,
    attn_ticket: CudaSlice<u32>,
}

pub struct CudaBatch {
    rt: Arc<FusedRuntime>,
    _mma: llvq_cuda::gpu::Cuda,
    gemm_mod: llvq_cuda::gpu::Cuda,
    attn: llvq_cuda::gpu::Cuda,
    tetra: [CudaFunction; 3],
    q4: [CudaFunction; 3],
    bg: Vec<(CudaFunction, usize, usize, u32)>,
    bgo: Vec<(CudaFunction, usize, usize, u32)>,
    attn_slot: CudaFunction,
    tables: TableView,
    sms: usize,
    groups: usize,
    scratch: Mutex<Scratch>,
    pub provenance: Vec<String>,
}

const PART_FLOATS: usize = 2 * 142 * BM_ROWS * MAX_ROWS * 2;
const TICKETS: usize = 4096;

impl CudaBatch {
    /// Compile both modules on the runtime's stream. `groups` is the number of
    /// query heads per KV head (4 for Qwen3-4B and 8B, 5 for 14B).
    pub fn new(rt: Arc<FusedRuntime>, groups: usize) -> candle_core::Result<Self> {
        let msg = |e: String| candle_core::Error::msg(e);
        let stream = rt.cuda_device().cuda_stream();
        let tables = rt.table_view().ok_or_else(|| candle_core::Error::msg("batched kernels: the runtime carries no Tetra tables"))?;
        let parts = [
            "#define TILE_BLOCKS 64u\n",
            llvq_cuda::SLOT_CUH,
            llvq_cuda::MATVEC_CU,
            llvq_cuda::F1RANK_CUH,
            llvq_cuda::F1RANK_V3_CUH,
            llvq_cuda::TETRA48_CUH,
            MMA_CU,
        ];
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        let mma = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(msg)?;
        let mut gparts = parts;
        gparts[parts.len() - 1] = GEMM_CU;
        let gsrc = llvq_cuda::gpu::KernelSource::new(&gparts);
        let gemm_mod = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &gsrc).map_err(msg)?;
        let asrc = llvq_cuda::gpu::KernelSource::new(&[llvq_cuda::ATTN_DEC_CU, ATTN_BATCH_CU]);
        let attn = llvq_cuda::gpu::Cuda::on_stream(stream, &asrc).map_err(msg)?;
        let mut provenance = vec![
            format!("batch mma unit: {} bytes, sha256 {}, target {}", src.text.len(), src.sha256, llvq_cuda::gpu::arch()),
            format!("batch gemm unit: {} bytes, sha256 {}", gsrc.text.len(), gsrc.sha256),
            format!("batch attention unit: {} bytes, sha256 {}", asrc.text.len(), asrc.sha256),
        ];
        let names_t = ["tv_tetra48_mma_n8", "tv_tetra48_mma_n16", "tv_tetra48_mma_n32"];
        let names_q = ["tv_q4_mma_n8", "tv_q4_mma_n16", "tv_q4_mma_n32"];
        let (attn_name, slot_name) = match groups {
            4 => ("attn_batch_g4_u64", "attn_slot_g4_u64"),
            5 => ("attn_batch_g5_u64", "attn_slot_g5_u64"),
            g => candle_core::bail!("batched attention: {g} query heads a KV head, compiled for 4 and 5"),
        };
        for n in names_t.iter().chain(names_q.iter()) {
            let r = mma.report(n).map_err(msg)?;
            provenance.push(format!("{n}: {} registers, {} B local", r.num_regs, r.local_bytes));
            if r.local_bytes != 0 {
                candle_core::bail!("{n} spills {} B to local memory", r.local_bytes);
            }
        }
        let mut sets = [Vec::with_capacity(5), Vec::with_capacity(5)];
        for (set, entries) in sets.iter_mut().zip([BG_ENTRIES, BG_OCC_ENTRIES]) {
            for (n, rows, kc) in entries {
                let bytes = bg_smem_bytes(rows, kc);
                let f = gemm_mod.func_dynamic_shared(n, bytes).map_err(msg)?;
                let r = gemm_mod.report(n).map_err(msg)?;
                provenance.push(format!("{n}: {} registers, {} B local, {bytes} B shared", r.num_regs, r.local_bytes));
                if r.local_bytes != 0 {
                    candle_core::bail!("{n} spills {} B to local memory", r.local_bytes);
                }
                set.push((f, rows, kc, bytes));
            }
        }
        let [bg, bgo] = sets;
        for n in [attn_name, slot_name] {
            let r = attn.report(n).map_err(msg)?;
            provenance.push(format!("{n}: {} registers, {} B local", r.num_regs, r.local_bytes));
        }
        let f = |c: &llvq_cuda::gpu::Cuda, n: &str| c.func(n).map_err(candle_core::Error::msg);
        let tetra = [f(&mma, names_t[0])?, f(&mma, names_t[1])?, f(&mma, names_t[2])?];
        let q4 = [f(&mma, names_q[0])?, f(&mma, names_q[1])?, f(&mma, names_q[2])?];
        let attn_slot = f(&attn, slot_name)?;
        let sms = mma.device().map_err(msg)?.sm_count as usize;
        let scratch = Scratch {
            gemm_part: gemm_mod.zeros_f32(1).map_err(msg)?,
            gemm_ticket: gemm_mod.zeros_u32(TICKETS).map_err(msg)?,
            part: mma.zeros_f32(PART_FLOATS).map_err(msg)?,
            ticket: mma.zeros_u32(TICKETS).map_err(msg)?,
            attn_part: attn.zeros_f32(1).map_err(msg)?,
            attn_ticket: attn.zeros_u32(1).map_err(msg)?,
        };
        Ok(Self {
            rt,
            _mma: mma,
            gemm_mod,
            attn,
            tetra,
            q4,
            bg,
            bgo,
            attn_slot,
            tables,
            sms,
            groups,
            scratch: Mutex::new(scratch),
            provenance,
        })
    }

    fn entry(n: usize) -> candle_core::Result<usize> {
        match n {
            1..=8 => Ok(0),
            9..=16 => Ok(1),
            17..=32 => Ok(2),
            _ => candle_core::bail!("batched projection: {n} rows, 1..={MAX_ROWS} allowed"),
        }
    }

    /// Slices of K so the grid holds about two CTAs a SM, and the slice length.
    fn split(&self, d_out: usize, units: usize, n: usize) -> (u32, u32) {
        let rg = d_out / BM_ROWS;
        let want = (2 * self.sms).div_ceil(rg).max(1);
        let s = want.min(units.div_ceil(4)).max(1);
        let slice = units.div_ceil(s);
        let s = units.div_ceil(slice);
        // The partials must fit; fewer slices otherwise.
        let s = s.min((PART_FLOATS / (n * d_out)).max(1));
        let slice = units.div_ceil(s);
        (units.div_ceil(slice) as u32, slice as u32)
    }

    fn smem(nt: usize) -> u32 {
        let n = 8 * nt;
        let xs = (16 / nt) * 16 + 4;
        ((2 * n * xs + 4 * 32 * 20 + 4 * 64) * 4) as u32
    }

    fn check_rows(name: &str, d_out: usize) -> candle_core::Result<()> {
        if !d_out.is_multiple_of(BM_ROWS) {
            candle_core::bail!("{name}: d_out {d_out} is not a multiple of {BM_ROWS}");
        }
        if d_out / BM_ROWS > TICKETS {
            candle_core::bail!("{name}: {} row groups, {TICKETS} tickets", d_out / BM_ROWS);
        }
        Ok(())
    }
}

impl crate::batch::BatchDevice for CudaBatch {
    fn name(&self) -> &str {
        "cuda"
    }

    fn attend(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        pos: &Tensor,
        slots: &Tensor,
        max_len: usize,
    ) -> candle_core::Result<Tensor> {
        use llvq_cuda::attn_ref::{HD, KMAX, PSTRIDE, SPLIT};
        let (b, nh, one, hd) = q.dims4()?;
        let (_, n_kv, cap, _) = kc.dims4()?;
        if slots.dims() != [b] || pos.dims() != [b] {
            candle_core::bail!("batched attention: {b} rows, positions {:?}, slots {:?}", pos.dims(), slots.dims());
        }
        if one != 1 || hd != HD || nh != n_kv * self.groups || !cap.is_multiple_of(SPLIT) || max_len > cap {
            candle_core::bail!(
                "batched attention: q {:?}, cache {:?}, length {max_len} (head dim {HD}, {} groups, cap a multiple of {SPLIT})",
                q.dims(),
                kc.dims(),
                self.groups
            );
        }
        let stream = self.rt.cuda_device().cuda_stream();
        // Bound here so the buffers outlive the launch that reads their addresses.
        let (q, k_new, v_new) = (q.contiguous()?, k_new.contiguous()?, v_new.contiguous()?);
        let (aq, ak, av, akc, avc, apos, aslot) = (
            addr(&q, &stream, 2)?,
            addr(&k_new, &stream, 2)?,
            addr(&v_new, &stream, 2)?,
            addr(kc, &stream, 2)?,
            addr(vc, &stream, 2)?,
            addr(pos, &stream, 4)?,
            addr(slots, &stream, 4)?,
        );
        let gx = max_len.div_ceil(SPLIT);
        let s_stride = gx.next_power_of_two();
        let mut sc = self.scratch.lock().map_err(|_| candle_core::Error::msg("batched scratch poisoned"))?;
        let need_p = b * n_kv * s_stride * KMAX * self.groups * PSTRIDE;
        if sc.attn_part.len() < need_p {
            sc.attn_part = self.attn.zeros_f32(need_p).map_err(candle_core::Error::msg)?;
        }
        if sc.attn_ticket.len() < b * n_kv {
            sc.attn_ticket = self.attn.zeros_u32(b * n_kv).map_err(candle_core::Error::msg)?;
        }
        let (part, ticket) = (slice_addr(&sc.attn_part, &stream), slice_addr(&sc.attn_ticket, &stream));
        let c = llvq_cuda::attn_ref::score_scale(HD);
        let (cap_i, ss) = (cap as i32, s_stride as i32);
        let f = &self.attn_slot;
        let run = |out: u64| -> candle_core::Result<()> {
            let mut l = stream.launch_builder(f);
            l.arg(&aq).arg(&ak).arg(&av).arg(&akc).arg(&avc).arg(&apos).arg(&aslot).arg(&cap_i).arg(&c).arg(&part).arg(&ticket).arg(&out).arg(&ss);
            let cfg = LaunchConfig { grid_dim: (gx as u32, n_kv as u32, b as u32), block_dim: (llvq_cuda::attn::THREADS, 1, 1), shared_mem_bytes: 0 };
            unsafe { l.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("attn_batch: {e}")))?;
            Ok(())
        };
        let op = Launch { name: "llvq-attn-batch", out_elems: b * nh * hd, out_shape: Shape::from((b, nh, 1, hd)), run: &run };
        let out = q.apply_op1_no_bwd(&op);
        drop(sc);
        out
    }

    fn lattice_mma(&self, p: &dyn crate::device::LatticeProj, xr: &Tensor, n: usize) -> candle_core::Result<Tensor> {
        let c = p
            .as_any()
            .and_then(|a| a.downcast_ref::<CudaLattice>())
            .ok_or_else(|| candle_core::Error::msg(format!("{}: not a CUDA lattice projection", p.name())))?;
        let v = c.tetra_view().ok_or_else(|| candle_core::Error::msg(format!("{}: not a Tetra48 stream", p.name())))?;
        let (d_out, d_in) = (v.d_out as usize, v.d_in as usize);
        Self::check_rows(p.name(), d_out)?;
        let e = Self::entry(n)?;
        if xr.dims() != [n, d_in] || xr.dtype() != candle_core::DType::F32 {
            candle_core::bail!("{}: rotated rows {:?} {:?}, expected [{n}, {d_in}] f32", p.name(), xr.dims(), xr.dtype());
        }
        let stream = self.rt.cuda_device().cuda_stream();
        let ax = addr(xr, &stream, 4)?;
        let (s, slice) = self.split(d_out, v.nblocks as usize, n);
        let sc = self.scratch.lock().map_err(|_| candle_core::Error::msg("batched scratch poisoned"))?;
        let (part, ticket) = (slice_addr(&sc.part, &stream), slice_addr(&sc.ticket, &stream));
        let t = self.tables;
        let (nr, di, dout) = (n as u32, d_in as u32, d_out as u32);
        let f = &self.tetra[e];
        let smem = Self::smem(1 << e);
        let run = |y: u64| -> candle_core::Result<()> {
            let mut l = stream.launch_builder(f);
            l.arg(&v.words).arg(&v.stride_u32).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&v.gscale)
                .arg(&t.invnorm).arg(&v.rscale).arg(&v.tail).arg(&ax).arg(&y).arg(&v.nblocks).arg(&v.tail_w)
                .arg(&nr).arg(&di).arg(&dout).arg(&slice).arg(&part).arg(&ticket);
            let cfg = LaunchConfig { grid_dim: (dout / BM_ROWS as u32, s, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem };
            unsafe { l.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("tv_tetra48_mma: {e}")))?;
            Ok(())
        };
        let op = Launch { name: "llvq-tetra48-mma", out_elems: n * d_out, out_shape: Shape::from((n, d_out)), run: &run };
        let out = xr.apply_op1_no_bwd(&op);
        drop(sc);
        out
    }

    fn int4_mma(&self, p: &dyn crate::device::Int4Proj, x: &Tensor, n: usize) -> candle_core::Result<Tensor> {
        let c = p
            .as_any()
            .and_then(|a| a.downcast_ref::<CudaInt4>())
            .ok_or_else(|| candle_core::Error::msg(format!("{}: not a CUDA int4 projection", p.name())))?;
        let v = c.int4_view();
        if v.layout != crate::fused::Int4Layout::Served {
            candle_core::bail!("{}: int4 words in 4E v9's tiled order; the batched kernel reads the file's order", p.name());
        }
        let (d_out, d_in) = (v.d_out as usize, v.d_in as usize);
        if v.gpr as usize * 128 != d_in {
            candle_core::bail!("{}: {} groups a row for d_in {d_in}, the kernel reads groups of 128", p.name(), v.gpr);
        }
        self.q4_launch(p.name(), v.wq, v.scales, v.biases, d_out, d_in, v.gpr, 7, x, n)
    }

    fn head_mma(&self, t: &dyn crate::device::QuantEmbedTable, h: &Tensor, n: usize) -> candle_core::Result<Tensor> {
        let c = t
            .as_any()
            .and_then(|a| a.downcast_ref::<CudaEmbed>())
            .ok_or_else(|| candle_core::Error::msg("head: not a CUDA embedding table"))?;
        let v = c.q4_view().ok_or_else(|| candle_core::Error::msg("head: the batched head reads the int4 g64 table only"))?;
        self.q4_launch("lm_head", v.words, v.scales, v.biases, v.vocab as usize, v.d as usize, v.gpr, 6, h, n)
    }

    fn max_rows(&self, projs: crate::batch::BatchProjs) -> usize {
        match projs {
            crate::batch::BatchProjs::Gemm | crate::batch::BatchProjs::GemmOcc => GEMM_MAX_ROWS,
            crate::batch::BatchProjs::Mma => MAX_ROWS,
            crate::batch::BatchProjs::Rows => usize::MAX,
        }
    }

    fn gemm(&self, items: &[crate::batch::GemmIn<'_>], n: usize, projs: crate::batch::BatchProjs) -> candle_core::Result<Vec<Tensor>> {
        let occ = match projs {
            crate::batch::BatchProjs::Gemm => false,
            crate::batch::BatchProjs::GemmOcc => true,
            p => candle_core::bail!("batched GEMM under the {} arm", p.name()),
        };
        let mut out = Vec::with_capacity(items.len());
        for group in items.chunks(BG_MAXSEG) {
            out.extend(self.gemm_launch(group, n, occ)?);
        }
        Ok(out)
    }
}

/// One projection of a GEMM launch, resolved to device addresses.
struct BgItem {
    seg: BgSeg,
    d_out: usize,
    tiles: usize,
}

impl CudaBatch {
    fn bg_item(&self, it: &crate::batch::GemmIn<'_>, n: usize, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> candle_core::Result<BgItem> {
        use crate::batch::GemmIn;
        let x_of = |name: &str, x: &Tensor, d_in: usize, dt: candle_core::DType, elem: u64| -> candle_core::Result<u64> {
            if x.dims() != [n, d_in] || x.dtype() != dt {
                candle_core::bail!("{name}: rows {:?} {:?}, expected [{n}, {d_in}] {dt:?}", x.dims(), x.dtype());
            }
            let a = addr(x, stream, elem)?;
            if !a.is_multiple_of(16) {
                candle_core::bail!("{name}: rows at {a:#x}, not 16-byte aligned");
            }
            Ok(a)
        };
        let int4 = |name: &str, wq: u64, scales: u64, biases: u64, d_out: usize, d_in: usize, gpr: u32, gshift: u32, x: u64| -> candle_core::Result<BgSeg> {
            if gpr as usize * (1usize << gshift) != d_in || !d_in.is_multiple_of(32) {
                candle_core::bail!("{name}: d_in {d_in}, {gpr} groups of {}", 1 << gshift);
            }
            Ok(BgSeg { w: wq, a0: scales, a1: biases, x, kind: 1, stride: gpr, nblocks: gshift, d_in: d_in as u32, d_out: d_out as u32, ..BgSeg::default() })
        };
        let seg = match *it {
            GemmIn::Lattice(p, x) => {
                let c = p
                    .as_any()
                    .and_then(|a| a.downcast_ref::<CudaLattice>())
                    .ok_or_else(|| candle_core::Error::msg(format!("{}: not a CUDA lattice projection", p.name())))?;
                let v = c.tetra_view().ok_or_else(|| candle_core::Error::msg(format!("{}: not a Tetra48 stream", p.name())))?;
                if !(v.d_in as usize).is_multiple_of(8) || v.stride_u32 as usize * 4 < 6 * v.nblocks as usize {
                    candle_core::bail!("{}: d_in {} or row stride {} out of the kernel's contract", p.name(), v.d_in, v.stride_u32);
                }
                let ax = x_of(p.name(), x, v.d_in as usize, candle_core::DType::F16, 2)?;
                BgSeg {
                    w: v.words,
                    a0: v.gscale,
                    a1: v.rscale,
                    tail: v.tail,
                    x: ax,
                    kind: 0,
                    stride: v.stride_u32,
                    nblocks: v.nblocks,
                    tail_w: v.tail_w,
                    d_in: v.d_in,
                    d_out: v.d_out,
                    ..BgSeg::default()
                }
            }
            GemmIn::Int4(p, x) => {
                let c = p
                    .as_any()
                    .and_then(|a| a.downcast_ref::<CudaInt4>())
                    .ok_or_else(|| candle_core::Error::msg(format!("{}: not a CUDA int4 projection", p.name())))?;
                let v = c.int4_view();
                let ax = x_of(p.name(), x, v.d_in as usize, candle_core::DType::F16, 2)?;
                let tiled = v.layout == crate::fused::Int4Layout::Tiled;
                if tiled && !crate::fused::int4_tileable(v.d_in as usize) {
                    candle_core::bail!("{}: tiled words on a width of {} the layout cannot hold", p.name(), v.d_in);
                }
                BgSeg { tiled: tiled as u32, ..int4(p.name(), v.wq, v.scales, v.biases, v.d_out as usize, v.d_in as usize, v.gpr, 7, ax)? }
            }
            GemmIn::Head(t, x) => {
                let c = t
                    .as_any()
                    .and_then(|a| a.downcast_ref::<CudaEmbed>())
                    .ok_or_else(|| candle_core::Error::msg("head: not a CUDA embedding table"))?;
                let v = c.q4_view().ok_or_else(|| candle_core::Error::msg("head: the batched head reads the int4 g64 table only"))?;
                let ax = x_of("lm_head", x, v.d as usize, candle_core::DType::F16, 2)?;
                int4("lm_head", v.words, v.scales, v.biases, v.vocab as usize, v.d as usize, v.gpr, 6, ax)?
            }
        };
        let d_out = seg.d_out as usize;
        if !d_out.is_multiple_of(BG_ROWS) {
            candle_core::bail!("batched GEMM: d_out {d_out} is not a multiple of {BG_ROWS}");
        }
        Ok(BgItem { seg, d_out, tiles: d_out / BG_ROWS })
    }

    /// Slices of K for every projection of a launch, so the grid holds about
    /// two CTAs a SM, each slice two chunks or more.
    fn bg_split(&self, items: &mut [BgItem], kc: usize) {
        let tiles: usize = items.iter().map(|i| i.tiles).sum();
        let want = (2 * self.sms).div_ceil(tiles.max(1)).max(1);
        for it in items.iter_mut() {
            let nk = (it.seg.d_in as usize).div_ceil(kc);
            let s = want.min(nk / 2).clamp(1, 32);
            let slice = nk.div_ceil(s);
            it.seg.nk = nk as u32;
            it.seg.slice = slice as u32;
            it.seg.s = nk.div_ceil(slice) as u32;
        }
    }

    fn gemm_launch(&self, group: &[crate::batch::GemmIn<'_>], n: usize, occ: bool) -> candle_core::Result<Vec<Tensor>> {
        if n == 0 || n > GEMM_MAX_ROWS || group.is_empty() || group.len() > BG_MAXSEG {
            candle_core::bail!("batched GEMM: {n} rows and {} projections; 1..={GEMM_MAX_ROWS} and 1..={BG_MAXSEG}", group.len());
        }
        let stream = self.rt.cuda_device().cuda_stream();
        let set = if occ { &self.bgo } else { &self.bg };
        let (f, _, kc, smem) = set.iter().find(|e| e.1 >= n).expect("an entry of 128 rows");
        let mut items = group.iter().map(|it| self.bg_item(it, n, &stream)).collect::<candle_core::Result<Vec<_>>>()?;
        self.bg_split(&mut items, *kc);
        let mut sc = self.scratch.lock().map_err(|_| candle_core::Error::msg("batched scratch poisoned"))?;
        let part_floats: usize = items.iter().filter(|i| i.seg.s > 1).map(|i| i.seg.s as usize * n * i.d_out).sum();
        if sc.gemm_part.len() < part_floats {
            sc.gemm_part = self.gemm_mod.zeros_f32(part_floats).map_err(candle_core::Error::msg)?;
        }
        let tickets: usize = items.iter().map(|i| i.tiles).sum();
        if sc.gemm_ticket.len() < tickets {
            sc.gemm_ticket = self.gemm_mod.zeros_u32(tickets).map_err(candle_core::Error::msg)?;
        }
        let (part, ticket) = (slice_addr(&sc.gemm_part, &stream), slice_addr(&sc.gemm_ticket, &stream));
        let t = self.tables;
        let mut args = BgArgs {
            trows: t.rows,
            pref: t.pref,
            br: t.br,
            suf: t.suf,
            invnorm: t.invnorm,
            ticket,
            nseg: items.len() as u32,
            n_rows: n as u32,
            ..BgArgs::default()
        };
        let (mut cta, mut tk, mut poff, mut yoff) = (0u32, 0u32, 0u64, Vec::with_capacity(items.len()));
        let mut total = 0usize;
        for (i, it) in items.iter().enumerate() {
            let mut sg = it.seg;
            sg.cta0 = cta;
            sg.tk0 = tk;
            if sg.s > 1 {
                sg.part = part + 4 * poff;
                poff += sg.s as u64 * (n * it.d_out) as u64;
            }
            cta += (it.tiles as u32) * sg.s;
            tk += it.tiles as u32;
            yoff.push(total);
            total += n * it.d_out;
            args.seg[i] = sg;
        }
        // The x the launch reads is the first projection's: any input works
        // as the custom op's anchor, the kernel reads every one by address.
        let anchor = match group[0] {
            crate::batch::GemmIn::Lattice(_, x) | crate::batch::GemmIn::Int4(_, x) | crate::batch::GemmIn::Head(_, x) => x,
        };
        let smem = *smem;
        let run = |y: u64| -> candle_core::Result<()> {
            let mut a = args;
            for (i, off) in yoff.iter().enumerate() {
                a.seg[i].y = y + 2 * *off as u64;
            }
            let mut l = stream.launch_builder(f);
            l.arg(&a);
            let cfg = LaunchConfig { grid_dim: (cta, 1, 1), block_dim: (BG_THREADS, 1, 1), shared_mem_bytes: smem };
            unsafe { l.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("tv_bg: {e}")))?;
            Ok(())
        };
        let op = Launch { name: "llvq-batch-gemm", out_elems: total, out_shape: Shape::from(total), run: &run };
        let flat = anchor.apply_op1_no_bwd(&op)?;
        drop(sc);
        let mut out = Vec::with_capacity(items.len());
        for (it, off) in items.iter().zip(yoff.iter()) {
            out.push(flat.narrow(0, *off, n * it.d_out)?.reshape((n, it.d_out))?);
        }
        Ok(out)
    }
}

impl CudaBatch {
    #[allow(clippy::too_many_arguments)]
    fn q4_launch(
        &self,
        name: &str,
        wq: u64,
        scales: u64,
        biases: u64,
        d_out: usize,
        d_in: usize,
        gpr: u32,
        gshift: u32,
        x: &Tensor,
        n: usize,
    ) -> candle_core::Result<Tensor> {
        Self::check_rows(name, d_out)?;
        if !d_in.is_multiple_of(32) {
            candle_core::bail!("{name}: d_in {d_in} is not a multiple of 32");
        }
        let e = Self::entry(n)?;
        let x = x.to_dtype(candle_core::DType::F16)?.contiguous()?;
        if x.dims() != [n, d_in] {
            candle_core::bail!("{name}: rows {:?}, expected [{n}, {d_in}]", x.dims());
        }
        let stream = self.rt.cuda_device().cuda_stream();
        let ax = addr(&x, &stream, 2)?;
        let (s, slice) = self.split(d_out, d_in / 32, n);
        let sc = self.scratch.lock().map_err(|_| candle_core::Error::msg("batched scratch poisoned"))?;
        let (part, ticket) = (slice_addr(&sc.part, &stream), slice_addr(&sc.ticket, &stream));
        let (nr, di, dout) = (n as u32, d_in as u32, d_out as u32);
        let f = &self.q4[e];
        let smem = Self::smem(1 << e);
        let run = |y: u64| -> candle_core::Result<()> {
            let mut l = stream.launch_builder(f);
            l.arg(&wq).arg(&scales).arg(&biases).arg(&ax).arg(&y).arg(&di).arg(&dout).arg(&gpr).arg(&gshift)
                .arg(&nr).arg(&slice).arg(&part).arg(&ticket);
            let cfg = LaunchConfig { grid_dim: (dout / BM_ROWS as u32, s, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem };
            unsafe { l.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("tv_q4_mma: {e}")))?;
            Ok(())
        };
        let op = Launch { name: "llvq-q4-mma", out_elems: n * d_out, out_shape: Shape::from((n, d_out)), run: &run };
        let out = x.apply_op1_no_bwd(&op);
        drop(sc);
        out
    }
}

/// The runtime a loaded model's projections live on, found through its first
/// Tetra projection.
pub fn runtime_of(m: &crate::model::Qwen3) -> candle_core::Result<Arc<FusedRuntime>> {
    for b in &m.blocks {
        for p in [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, &b.gate_proj, &b.up_proj, &b.down_proj] {
            if let crate::model::Proj::Lattice(d) = p {
                if let Some(c) = d.as_any().and_then(|a| a.downcast_ref::<CudaLattice>()) {
                    return Ok(c.runtime().clone());
                }
            }
        }
    }
    candle_core::bail!("no CUDA lattice projection in this model")
}

/// Whether `d` is a CUDA device (the batched step exists there only).
pub fn on_cuda(d: &Device) -> bool {
    matches!(d, Device::Cuda(_))
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_argument_structs_have_the_kernels_layout() {
        // 7 eight-byte fields and 12 four-byte ones; four of those, six
        // pointers and two counts.
        assert_eq!(std::mem::size_of::<super::BgSeg>(), 104);
        assert_eq!(std::mem::size_of::<super::BgArgs>(), 4 * 104 + 6 * 8 + 2 * 4);
    }

    #[test]
    fn the_shared_bytes_are_the_kernels() {
        // BG_SMEM_WORDS at (8, 192) and (128, 96), worked by hand: the wider
        // stage is int4's (words of 26 and 14 a row), the A tiles 8 x 16 x 28.
        assert_eq!(super::bg_smem_bytes(8, 192), (2 * (8 * 100 + 128 * 26) + 3584) * 4);
        assert_eq!(super::bg_smem_bytes(128, 96), (2 * (128 * 52 + 128 * 14) + 3584) * 4);
        assert_eq!(super::bg_smem_bytes(128, 48), (2 * (128 * 28 + 128 * 6) + 3584) * 4);
        // The measurement arm fits two CTAs a SM (100 KB, 1 KB reserved a CTA).
        for (_, n, kc) in super::BG_OCC_ENTRIES.iter() {
            assert!(super::bg_smem_bytes(*n, *kc) <= 49 * 1024, "{n} rows: {} B", super::bg_smem_bytes(*n, *kc));
        }
    }
}
