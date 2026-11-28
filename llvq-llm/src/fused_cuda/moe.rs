//! The routed experts of a token through `moe_tetra.cu`: the kernels, the
//! layer of expert references, and the custom ops that launch them. One file
//! because it is one additive module on the runtime's stream.

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, Layout, Shape, Tensor};
use half::f16;

use crate::fused::{load_planes_sources, FusedLayout};

use super::adapters::dev_addr;
use super::{DeviceStream, ExtRotation, FusedProj, FusedRuntime, THREADS};

// ---------------------------------------------------------------------------
// The routed experts of one decoded token in four launches (`moe_tetra.cu`),
// the CUDA twin of the Metal kernels of `fused_metal/`. Additive: a module
// of its own on the runtime's stream, built on first use; the served unit and
// its launches do not move. The expert records are the buffers
// `upload_matrix_ext` and `upload_rotation_ext` made, read through a table of
// their device addresses.
// ---------------------------------------------------------------------------

const MOE_TETRA_CU_EMBED: &str = include_str!("../../kernels/moe_tetra.cu");

/// Routed experts a call can name.
pub const MOE_MAX_SELECTED: usize = 32;
/// `u64` of one `MoePtr` of `moe_tetra.cu`.
const MOE_PTR_WORDS: usize = 16;

/// The three functions of `moe_tetra.cu`, compiled for one runtime.
pub struct MoeKernels {
    cuda: llvq_cuda::gpu::Cuda,
    f_rot: CudaFunction,
    f_gu: CudaFunction,
    f_down: CudaFunction,
    f_act: CudaFunction,
    /// `moe_wsum`, each token's weighted rows of a prompt's down output.
    f_wsum: CudaFunction,
    /// `moe_tetra_rows`, the prompt's launch; `None` on a unit whose prefill
    /// pair does not fit the default shared-memory allowance.
    f_rows: Option<CudaFunction>,
}

/// One expert of a layer, as its resident records: gate, up, down, and the two rotations.
pub struct MoeExpertRefs<'a> {
    pub gate: &'a FusedProj,
    pub up: &'a FusedProj,
    pub down: &'a FusedProj,
    pub rot_in: &'a ExtRotation,
    pub rot_down: &'a ExtRotation,
}

/// What the four launches read of one layer: the address table of its experts
/// and the shapes every expert shares.
///
/// SAFETY (open), the clause every `moe_*` launch below leans on: `ptr` holds
/// raw device addresses of buffers owned by the `FusedProj` and
/// `ExtRotation` values [`FusedRuntime::moe_layer`] borrowed, and nothing ties
/// their lifetime to this struct. `moe_layer` is a `pub` safe function, so
/// safe code can drop the experts and keep launching on this table, which
/// then reads freed device memory (the BUG-15 shape). The one caller,
/// `inkling::resident_cuda::ResidentExperts`, keeps both in private fields of
/// one struct that never replaces `experts`, so the served path holds. Second
/// gap: the kernels index the table by the expert ids in `sel` (or the
/// `chunks`/`side` tables of the prompt ops) with no bound; the ops check the
/// tables' lengths but not that each id is below `experts`, which the router
/// guarantees as far as this crate shows.
pub struct MoeLayer {
    ptr: CudaSlice<u64>,
    experts: usize,
    hidden: usize,
    inter: usize,
    /// (row_stride_u32, nblocks, tail_w) of gate and up, then of down.
    gu: (u32, u32, u32),
    down: (u32, u32, u32),
    /// (n, m, k, inv, threads) of the input rotation, then of the down one.
    rot_in: (u32, u32, u32, f32, u32),
    rot_down: (u32, u32, u32, f32, u32),
}

impl MoeLayer {
    pub fn experts(&self) -> usize {
        self.experts
    }
}

/// (row_stride_u32, nblocks, tail_w) of a Tetra48 projection.
fn tetra_shape(p: &FusedProj) -> candle_core::Result<(u32, u32, u32)> {
    let DeviceStream::Tetra48 { stride_u32, .. } = &p.stream else {
        candle_core::bail!("{}: not a tetra48 stream", p.name);
    };
    Ok((*stride_u32, p.nblocks, p.tail_w))
}

impl FusedRuntime {
    /// Compile `moe_tetra.cu` beside the served unit, on the same stream.
    pub fn moe_kernels(&self) -> candle_core::Result<&MoeKernels> {
        if let Some(k) = self.moe.get() {
            return Ok(k);
        }
        let k = self.compile_moe()?;
        Ok(self.moe.get_or_init(|| k))
    }

    fn compile_moe(&self) -> candle_core::Result<MoeKernels> {
        if self.tetra_tabs.is_none() {
            candle_core::bail!("moe_kernels needs a runtime built for tetra48");
        }
        let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_rot.cuh", "rotate.cu"])
            .map_err(candle_core::Error::msg)?;
        let (planes, _) = load_planes_sources(FusedLayout::Tetra48).map_err(candle_core::Error::msg)?;
        let defines = format!("{}{}", self.tile.define(), self.prefill.defines());
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(base.parts.iter().map(String::as_str))
            .chain(planes.iter().map(String::as_str))
            .chain(std::iter::once(MOE_TETRA_CU_EMBED))
            .collect();
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        println!("NVRTC moe unit: {} bytes, sha256 {} ({} parts)", src.text.len(), src.sha256, parts.len());
        let cuda = llvq_cuda::gpu::Cuda::on_stream(self.device.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
        let rot_bytes = self
            .rotations
            .keys()
            .map(|k| llvq_cuda::shared::rot_bytes(k.0))
            .max()
            .unwrap_or(0);
        let f_rot = cuda.func_dynamic_shared("moe_rot", rot_bytes as u32).map_err(candle_core::Error::msg)?;
        let f_gu = cuda.func("moe_tetra_gu").map_err(candle_core::Error::msg)?;
        let f_down = cuda.func("moe_tetra_down").map_err(candle_core::Error::msg)?;
        let f_act = cuda.func("moe_act").map_err(candle_core::Error::msg)?;
        let f_wsum = cuda.func("moe_wsum").map_err(candle_core::Error::msg)?;
        let f_rows = (self.prefill.shared_bytes() <= llvq_cuda::tile::SHARED_DEFAULT)
            .then(|| cuda.func("moe_tetra_rows"))
            .transpose()
            .map_err(candle_core::Error::msg)?;
        Ok(MoeKernels { cuda, f_rot, f_gu, f_down, f_act, f_wsum, f_rows })
    }

    /// The address table of a layer's experts. Refused unless every expert has
    /// the shapes of the first and both rotations stage in shared memory.
    pub fn moe_layer(&self, experts: &[MoeExpertRefs<'_>]) -> candle_core::Result<MoeLayer> {
        let first = experts.first().ok_or_else(|| candle_core::Error::msg("moe_layer: no expert"))?;
        let (hidden, inter) = (first.gate.d_in, first.gate.d_out);
        let shape = tetra_shape;
        let gu = shape(first.gate)?;
        let down = shape(first.down)?;
        let rot = |r: &ExtRotation| -> candle_core::Result<(u32, u32, u32, f32, u32)> {
            if !matches!(r.buf.kernel, llvq_cuda::wide::RotKernel::Shared(_)) {
                candle_core::bail!("moe_layer: a rotation of width {} staged on the device", r.buf.n);
            }
            Ok((r.buf.n, r.buf.m, r.buf.k, r.buf.inv, r.buf.threads))
        };
        let (rot_in, rot_down) = (rot(first.rot_in)?, rot(first.rot_down)?);
        let st = self.device.cuda_stream();
        let mut table: Vec<u64> = Vec::with_capacity(experts.len() * MOE_PTR_WORDS);
        for (e, x) in experts.iter().enumerate() {
            let same = x.gate.d_in == hidden
                && x.up.d_in == hidden
                && x.gate.d_out == inter
                && x.up.d_out == inter
                && x.down.d_in == inter
                && x.down.d_out == hidden
                && shape(x.gate)? == gu
                && shape(x.up)? == gu
                && shape(x.down)? == down
                && rot(x.rot_in)? == rot_in
                && rot(x.rot_down)? == rot_down
                && x.gate.rotation == Some(x.rot_in.key)
                && x.up.rotation == Some(x.rot_in.key)
                && x.down.rotation == Some(x.rot_down.key);
            if !same {
                candle_core::bail!("moe_layer: expert {e} does not have the shapes and rotations of expert 0");
            }
            let words = |p: &FusedProj| match &p.stream {
                DeviceStream::Tetra48 { words, .. } => dev_addr(words, &st),
                _ => unreachable!("checked by tetra_shape"),
            };
            let m = [x.gate, x.up, x.down];
            table.extend(m.iter().map(|p| words(p)));
            table.extend(m.iter().map(|p| dev_addr(&p.rscale, &st)));
            table.extend(m.iter().map(|p| dev_addr(&p.gscale, &st)));
            table.extend(m.iter().map(|p| dev_addr(&p.tail, &st)));
            table.push(dev_addr(&x.rot_in.buf.signbits, &st));
            table.push(dev_addr(&x.rot_in.buf.small, &st));
            table.push(dev_addr(&x.rot_down.buf.signbits, &st));
            table.push(dev_addr(&x.rot_down.buf.small, &st));
        }
        if inter % (THREADS as usize / 32) != 0 || hidden % (THREADS as usize / 32) != 0 {
            candle_core::bail!("moe_layer: rows {inter} and {hidden} do not fill whole blocks");
        }
        let ptr = self.cuda.up_u64(&table).map_err(candle_core::Error::msg)?;
        Ok(MoeLayer { ptr, experts: experts.len(), hidden, inter, gu, down, rot_in, rot_down })
    }

    /// Launch a or c: `x` f16 into the basis of each routed expert, f32
    /// `[nsel, n]`. `down` false: `x` is `[1, hidden]`, the same row for every
    /// expert. `down` true: `x` is `[nsel, inter]`, a row an expert.
    pub fn moe_rotate(&self, k: &MoeKernels, layer: &MoeLayer, down: bool, x: &Tensor, sel: &Tensor) -> candle_core::Result<Tensor> {
        x.apply_op2_no_bwd(sel, &MoeOp { rt: self, k, layer, step: if down { MoeStep::RotDown } else { MoeStep::RotIn } })
    }

    /// Launch b: gate and up of the routed experts on their rotated rows `xr`
    /// f32 `[nsel, hidden]`: f32 `[2, nsel, inter]`.
    pub fn moe_gate_up(&self, k: &MoeKernels, layer: &MoeLayer, xr: &Tensor, sel: &Tensor) -> candle_core::Result<Tensor> {
        xr.apply_op2_no_bwd(sel, &MoeOp { rt: self, k, layer, step: MoeStep::GateUp })
    }

    /// Launch d: down of the routed experts on their rotated activations `ar`
    /// f32 `[nsel, inter]`, weighted and summed: f32 `[1, hidden]`.
    pub fn moe_down_sum(&self, k: &MoeKernels, layer: &MoeLayer, ar: &Tensor, sel: &Tensor) -> candle_core::Result<Tensor> {
        ar.apply_op2_no_bwd(sel, &MoeOp { rt: self, k, layer, step: MoeStep::DownSum })
    }
}

impl FusedRuntime {
    /// The activation of the routed experts in one launch: `gu` f32
    /// `[2, nsel, inter]` (gate rows, then up rows) to f16 `[nsel, inter]`.
    pub fn moe_activation(&self, k: &MoeKernels, gu: &Tensor, cap_gate: f64, cap_up: f64) -> candle_core::Result<Tensor> {
        let (two, rows, inter) = gu.dims3()?;
        if two != 2 || rows * inter == 0 {
            candle_core::bail!("moe activation: gate and up as {:?}, want [2, rows, inter]", gu.dims());
        }
        gu.apply_op1_no_bwd(&MoeActOp { rt: self, k, rows, inter, caps: [cap_gate as f32, cap_up as f32, (1.0 / cap_gate) as f32, (1.0 / cap_up) as f32] })
    }
}

struct MoeActOp<'a> {
    rt: &'a FusedRuntime,
    k: &'a MoeKernels,
    rows: usize,
    inter: usize,
    /// cap_gate, cap_up, 1 / cap_gate, 1 / cap_up.
    caps: [f32; 4],
}

impl candle_core::CustomOp1 for MoeActOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-moe-act"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the routed-experts activation launch has no CPU path")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let x = storage.as_cuda_slice::<f32>()?;
        let n = self.rows * self.inter;
        match layout.contiguous_offsets() {
            Some((0, end)) if end == 2 * n => {}
            _ => candle_core::bail!("moe activation: gate and up must be contiguous at offset 0"),
        }
        let stream = self.rt.device.cuda_stream();
        // SAFETY: clause U of the launch contract (`fused_cuda` module doc):
        // `moe_act` runs `ceil(n / THREADS)` blocks, returns past `i >= n`
        // and stores `out[i]` for every `i < n`.
        let mut out = unsafe { stream.alloc::<f16>(n) }
            .map_err(|e| candle_core::Error::msg(format!("moe activation: alloc: {e}")))?;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        let n32 = n as u32;
        let [cap_g, cap_u, inv_g, inv_u] = self.caps;
        let mut b = self.k.cuda.stream().launch_builder(&self.k.f_act);
        b.arg(x).arg(&mut out).arg(&n32).arg(&inv_g).arg(&cap_g).arg(&inv_u).arg(&cap_u);
        let probe = crate::inkling::probe_begin(|| self.k.cuda.stream().synchronize().is_ok());
                crate::inkling::count_launch();
                // SAFETY: the launch contract (`fused_cuda` module doc). L1:
                // the seven arguments of `moe_act` (`kernels/moe_tetra.cu`).
                // L3: `x` holds `2n` f32 at offset 0 (checked above), `out`
                // `n` halves; the kernel guards `i >= n`, so the rounded-up
                // grid stays inside both. `self.k.cuda` is on candle's stream.
                unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_act: {e}")))?;
                crate::inkling::probe_end(probe, "moe_act", || self.k.cuda.stream().synchronize().is_ok());
        Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((self.rows, self.inter))))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum MoeStep {
    RotIn,
    RotDown,
    GateUp,
    DownSum,
}

struct MoeOp<'a> {
    rt: &'a FusedRuntime,
    k: &'a MoeKernels,
    layer: &'a MoeLayer,
    step: MoeStep,
}

impl candle_core::CustomOp2 for MoeOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-moe-tetra"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the routed-experts launches have no CPU path")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ss: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let l = self.layer;
        // The routing: ids ascending, then the weights' bits, `[2 * nsel]` u32.
        let sel = ss.as_cuda_slice::<u32>()?;
        let (s0, s1) = sl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe: non-contiguous routing"))?;
        let nsel = (s1 - s0) / 2;
        if s0 != 0 || (s1 - s0) % 2 != 0 || nsel == 0 || nsel > MOE_MAX_SELECTED {
            candle_core::bail!("moe: a routing of {} words at offset {s0}, which wants 2 x 1..={MOE_MAX_SELECTED} at 0", s1 - s0);
        }
        let (x0, x1) = xl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe: non-contiguous input"))?;
        let len = x1 - x0;
        let stream = self.rt.device.cuda_stream();
        let alloc = |n: usize| {
            // SAFETY: clause U of the launch contract (`fused_cuda` module
            // doc), for the three outputs of the match below: `moe_rot` writes
            // `[0, n)` of each of its `nsel` rows, and `moe_tetra_gu` and
            // `moe_tetra_down` store one value per output row from that row's
            // warp over an exact grid (`inter` and `hidden` multiples of 8,
            // `moe_layer`).
            unsafe { stream.alloc::<f32>(n) }.map_err(|e| candle_core::Error::msg(format!("moe: alloc of {n} floats: {e}")))
        };
        let tabs = self
            .rt
            .tetra_tabs
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("moe: a runtime without the tetra tables"))?;
        let shared = self.rt.tile.shared_bytes();
        let (nsel32, per_block) = (nsel as u32, THREADS / 32);
        match self.step {
            MoeStep::RotIn | MoeStep::RotDown => {
                let down = self.step == MoeStep::RotDown;
                let (n, m, k, inv, threads) = if down { l.rot_down } else { l.rot_in };
                let (rows, stride) = if down { (nsel, n) } else { (1, 0) };
                if len != rows * n as usize {
                    candle_core::bail!("moe rotation: {len} values for {rows} rows of {n}");
                }
                let x = xs.as_cuda_slice::<f16>()?;
                let mut out = alloc(nsel * n as usize)?;
                let cfg = LaunchConfig { grid_dim: (nsel32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: n * 4 };
                let (x_off, which) = (x0 as u32, down as u32);
                let mut b = self.k.cuda.stream().launch_builder(&self.k.f_rot);
                b.arg(x).arg(&l.ptr).arg(sel).arg(&mut out).arg(&n).arg(&m).arg(&k).arg(&inv).arg(&x_off).arg(&stride).arg(&which);
                let probe = crate::inkling::probe_begin(|| self.k.cuda.stream().synchronize().is_ok());
                crate::inkling::count_launch();
                // SAFETY (open): L1 holds (the eleven arguments of `moe_rot`,
                // `kernels/moe_tetra.cu`), and L3 for `x` (`rows * n` halves
                // from `x_off`, checked above), `out` (`nsel * n`) and `sel`
                // (`2 * nsel` words); the shared request `n * 4` is what the
                // kernel stages, below `rot_bytes` of the widest key the
                // function was loaded with. The rotation tables are read
                // through `l.ptr`: see the open note on `MoeLayer`.
                unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_rot: {e}")))?;
                crate::inkling::probe_end(probe, "moe_rot", || self.k.cuda.stream().synchronize().is_ok());
                Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((nsel, n as usize))))
            }
            MoeStep::GateUp => {
                if x0 != 0 || len != nsel * l.hidden {
                    candle_core::bail!("moe gate/up: {len} values at offset {x0} for {nsel} rows of {}", l.hidden);
                }
                let x = xs.as_cuda_slice::<f32>()?;
                let (stride, nblocks, tail_w) = l.gu;
                let rows = 2 * nsel32 * l.inter as u32;
                let mut out = alloc(rows as usize)?;
                let cfg = LaunchConfig { grid_dim: (rows / per_block, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared };
                let d_out = l.inter as u32;
                let mut b = self.k.cuda.stream().launch_builder(&self.k.f_gu);
                b.arg(&l.ptr).arg(sel).arg(&stride).arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
                    .arg(&tabs.invnorm).arg(x).arg(&mut out).arg(&nblocks).arg(&tail_w).arg(&d_out).arg(&nsel32);
                let probe = crate::inkling::probe_begin(|| self.k.cuda.stream().synchronize().is_ok());
                crate::inkling::count_launch();
                // SAFETY (open): L1 holds (the fourteen arguments of
                // `moe_tetra_gu`); L3 for `x` (`nsel * hidden` f32 at offset 0,
                // checked above) and `out` (`2 * nsel * inter`); L4: `inter`
                // a multiple of 8 (`moe_layer`). The weights are read through
                // `l.ptr`: see the open note on `MoeLayer`.
                unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_tetra_gu: {e}")))?;
                crate::inkling::probe_end(probe, "moe_tetra_gu", || self.k.cuda.stream().synchronize().is_ok());
                Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((2, nsel, l.inter))))
            }
            MoeStep::DownSum => {
                if x0 != 0 || len != nsel * l.inter {
                    candle_core::bail!("moe down: {len} values at offset {x0} for {nsel} rows of {}", l.inter);
                }
                let x = xs.as_cuda_slice::<f32>()?;
                let (stride, nblocks, tail_w) = l.down;
                let rows = l.hidden as u32;
                let mut out = alloc(rows as usize)?;
                let cfg = LaunchConfig { grid_dim: (rows / per_block, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared };
                let mut b = self.k.cuda.stream().launch_builder(&self.k.f_down);
                b.arg(&l.ptr).arg(sel).arg(&stride).arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
                    .arg(&tabs.invnorm).arg(x).arg(&mut out).arg(&nblocks).arg(&tail_w).arg(&nsel32);
                let probe = crate::inkling::probe_begin(|| self.k.cuda.stream().synchronize().is_ok());
                crate::inkling::count_launch();
                // SAFETY (open): L1 holds (the thirteen arguments of
                // `moe_tetra_down`); L3 for `x` (`nsel * inter` f32 at offset
                // 0, checked above) and `out` (`hidden`); L4: `hidden` a
                // multiple of 8 (`moe_layer`). The weights and the routing
                // weights in `sel[nsel..]` are read through `l.ptr` and `sel`:
                // see the open note on `MoeLayer`.
                unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_tetra_down: {e}")))?;
                crate::inkling::probe_end(probe, "moe_tetra_down", || self.k.cuda.stream().synchronize().is_ok());
                Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((1, l.hidden))))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A prompt's rows through the routed experts, one launch a step for every
// (token, expert) pair of a layer (`moe_tetra_rows` and `moe_rot`). The values
// are those of the path that takes one expert and one chunk at a time.
// ---------------------------------------------------------------------------

impl FusedRuntime {
    /// Whether the prompt's launches exist on this runtime.
    pub fn moe_has_rows(&self, k: &MoeKernels) -> bool {
        k.f_rows.is_some()
    }

    /// `x` f16 `[npairs, n]`, a row a pair, into the basis of each pair's
    /// expert (`pair_expert` u32 `[npairs]`): f32 `[npairs, n]`, one launch.
    /// `down` picks the rotation of `down` and its width.
    pub fn moe_rotate_pairs(&self, k: &MoeKernels, layer: &MoeLayer, down: bool, x: &Tensor, pair_expert: &Tensor) -> candle_core::Result<Tensor> {
        x.apply_op2_no_bwd(pair_expert, &MoeRowsOp { rt: self, k, layer, down, rotate: true })
    }

    /// Gate and up (`down` false, f16 `[2, npairs, inter]`) or down (`down`
    /// true, f16 `[1, npairs, hidden]`) of every pair on its rotated row `xr`
    /// f32 `[npairs, d_in]`, one launch. `chunks`: u32 `[3 * nchunks]`, the
    /// expert, the first pair and the count of each chunk; the chunks cover
    /// the pairs in order and none holds more than [`Self::prefill_rows`].
    pub fn moe_matrix_pairs(&self, k: &MoeKernels, layer: &MoeLayer, down: bool, xr: &Tensor, chunks: &Tensor) -> candle_core::Result<Tensor> {
        xr.apply_op2_no_bwd(chunks, &MoeRowsOp { rt: self, k, layer, down, rotate: false })
    }

    /// Each token's `k_sel` weighted rows of `y` summed, one launch. `y`: the
    /// down output of [`Self::moe_matrix_pairs`], f16 `[1, npairs, d]` (any
    /// contiguous shape whose last dim is `d`, at offset 0). `side`: u32
    /// `[2 * ntok * k_sel]`, for token t and rank r the pair of its r-th
    /// expert in ascending expert order at `t * k_sel + r`, then that pair's
    /// weight as f32 bits at `ntok * k_sel + t * k_sel + r`. Returns f32
    /// `[ntok, d]`: per token and column, `+0` then `+= f32(y) * w` rank by
    /// rank, each operation rounded once, as the host's multiply and adds did.
    pub fn moe_weighted_sum(&self, k: &MoeKernels, y: &Tensor, side: &Tensor, k_sel: usize) -> candle_core::Result<Tensor> {
        y.apply_op2_no_bwd(side, &MoeWsumOp { rt: self, k, k_sel })
    }
}

struct MoeWsumOp<'a> {
    rt: &'a FusedRuntime,
    k: &'a MoeKernels,
    k_sel: usize,
}

impl candle_core::CustomOp2 for MoeWsumOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-moe-wsum"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the routed-experts weighted sum has no CPU path")
    }

    fn cuda_fwd(
        &self,
        ys: &CudaStorage,
        yl: &Layout,
        ss: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let y = ys.as_cuda_slice::<f16>()?;
        let side = ss.as_cuda_slice::<u32>()?;
        let (y0, y1) = yl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe wsum: non-contiguous rows"))?;
        let (s0, s1) = sl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe wsum: non-contiguous table"))?;
        let d = yl.dims().last().copied().unwrap_or(0);
        let k_sel = self.k_sel;
        if y0 != 0 || s0 != 0 || d == 0 || k_sel == 0 || y1 % d != 0 || s1 == 0 || s1 % (2 * k_sel) != 0 {
            candle_core::bail!("moe wsum: rows of {y1} at {y0} and width {d}, table of {s1} at {s0}, k {k_sel}");
        }
        let ntok = s1 / (2 * k_sel);
        let total = ntok as u64 * d as u64;
        if total + THREADS as u64 > u32::MAX as u64 {
            candle_core::bail!("moe wsum: {ntok} tokens of {d} overflow the u32 index");
        }
        let stream = self.rt.device.cuda_stream();
        // SAFETY: clause U of the launch contract (`fused_cuda` module doc):
        // `moe_wsum` runs one thread per output, returns past
        // `i >= ntok * d`, and stores each `out[i]` from `+0`.
        let mut out = unsafe { stream.alloc::<f32>(total as usize) }
            .map_err(|e| candle_core::Error::msg(format!("moe wsum: alloc of {total} floats: {e}")))?;
        let cfg = LaunchConfig { grid_dim: ((total as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        let (ntok32, d32, k32) = (ntok as u32, d as u32, k_sel as u32);
        let mut b = self.k.cuda.stream().launch_builder(&self.k.f_wsum);
        b.arg(y).arg(side).arg(&mut out).arg(&ntok32).arg(&d32).arg(&k32);
        let probe = crate::inkling::probe_begin(|| self.k.cuda.stream().synchronize().is_ok());
        crate::inkling::count_launch();
        // SAFETY (open): L1 holds (the six arguments of `moe_wsum`); `out`
        // holds `ntok * d`, guarded in the kernel, and the index fits a `u32`
        // (checked above). Not established here: each pair index in
        // `side[.. ntok * k]` is below `y1 / d`, the rows `y` holds; the
        // kernel reads `y` at it unchecked, and only the caller's
        // construction of `side` bounds it.
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_wsum: {e}")))?;
        crate::inkling::probe_end(probe, "moe_wsum", || self.k.cuda.stream().synchronize().is_ok());
        Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((ntok, d))))
    }
}

struct MoeRowsOp<'a> {
    rt: &'a FusedRuntime,
    k: &'a MoeKernels,
    layer: &'a MoeLayer,
    down: bool,
    rotate: bool,
}

impl candle_core::CustomOp2 for MoeRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-moe-tetra-rows"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the routed-experts prompt launches have no CPU path")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ss: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let l = self.layer;
        let side = ss.as_cuda_slice::<u32>()?;
        let (s0, s1) = sl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe rows: non-contiguous table"))?;
        let (x0, x1) = xl.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe rows: non-contiguous input"))?;
        if s0 != 0 || x0 != 0 || s1 == 0 {
            candle_core::bail!("moe rows: input at offset {x0}, table of {} words at offset {s0}", s1 - s0);
        }
        let len = x1;
        let stream = self.rt.device.cuda_stream();
        if self.rotate {
            let (n, m, k, inv, threads) = if self.down { l.rot_down } else { l.rot_in };
            let npairs = s1;
            if len != npairs * n as usize || npairs > u32::MAX as usize {
                candle_core::bail!("moe rows rotation: {len} values for {npairs} rows of {n}");
            }
            let x = xs.as_cuda_slice::<f16>()?;
            // SAFETY: clause U of the launch contract (`fused_cuda` module
            // doc): `moe_rot` runs one block per pair (`npairs` rows of `n`,
            // `len == npairs * n` checked above) and writes `[0, n)` of each.
            let mut out = unsafe { stream.alloc::<f32>(len) }
                .map_err(|e| candle_core::Error::msg(format!("moe rows: alloc of {len} floats: {e}")))?;
            let cfg = LaunchConfig { grid_dim: (npairs as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: n * 4 };
            let (x_off, which) = (0u32, self.down as u32);
            let mut b = self.k.cuda.stream().launch_builder(&self.k.f_rot);
            b.arg(x).arg(&l.ptr).arg(side).arg(&mut out).arg(&n).arg(&m).arg(&k).arg(&inv).arg(&x_off).arg(&n).arg(&which);
            crate::inkling::count_launch();
            // SAFETY (open): L1 holds (`moe_rot`, eleven arguments, `side`
            // in the place of `sel` and row stride `n`); L3 for `x` and
            // `out` (`npairs * n` each). The expert of each pair is read from
            // `side` and the tables through `l.ptr`: see the open note on
            // `MoeLayer`.
            unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_rot (rows): {e}")))?;
            return Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((npairs, n as usize))));
        }
        let f = self.k.f_rows.as_ref().ok_or_else(|| candle_core::Error::msg("moe rows: this unit carries no moe_tetra_rows"))?;
        let tabs = self
            .rt
            .tetra_tabs
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("moe rows: a runtime without the tetra tables"))?;
        let ((stride, nblocks, tail_w), d_in, d_out, mat0, nmat) =
            if self.down { (l.down, l.inter, l.hidden, 2u32, 1usize) } else { (l.gu, l.hidden, l.inter, 0u32, 2usize) };
        let nchunks = s1 / 3;
        let npairs = len / d_in;
        let warps = nmat as u64 * nchunks as u64 * d_out as u64;
        if !s1.is_multiple_of(3) || nchunks == 0 || npairs * d_in != len || warps * 32 > u32::MAX as u64 {
            candle_core::bail!("moe rows: {len} values of width {d_in}, {} table words, {warps} warps", s1);
        }
        let x = xs.as_cuda_slice::<f32>()?;
        // SAFETY (open): clause U holds only if the `chunks` table covers
        // every pair once, as `moe_matrix_pairs`' doc asks of the caller:
        // `moe_tetra_rows` stores `y` for the rows of each chunk it is given
        // (`r < n_rows`) and nothing else, and this op does not read the
        // table, which lives on the device. A table that skips a pair leaves
        // its rows uninitialised.
        let mut out = unsafe { stream.alloc::<f16>(nmat * npairs * d_out) }
            .map_err(|e| candle_core::Error::msg(format!("moe rows: alloc of {} halves: {e}", nmat * npairs * d_out)))?;
        let per_block = THREADS / 32;
        let cfg = LaunchConfig {
            grid_dim: ((warps as u32) / per_block, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: self.rt.prefill_shared,
        };
        let (d_out32, nchunks32, npairs32) = (d_out as u32, nchunks as u32, npairs as u32);
        let mut b = self.k.cuda.stream().launch_builder(f);
        b.arg(&l.ptr).arg(side).arg(&stride).arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
            .arg(&tabs.invnorm).arg(x).arg(&mut out).arg(&nblocks).arg(&tail_w).arg(&d_out32).arg(&nchunks32).arg(&npairs32).arg(&mat0);
        crate::inkling::count_launch();
        // SAFETY (open): L1 holds (the sixteen arguments of `moe_tetra_rows`);
        // L3 for `x` (`npairs * d_in` f32 at offset 0) and `out`; L4: the
        // grid is `nmat * nchunks * d_out` warps, `d_out` a multiple of 8,
        // and the count fits a `u32` (checked above). Not established here:
        // each chunk's expert, first pair and count stay inside `experts`
        // and `npairs` and its count within `prefill_rows`; the kernel reads
        // them from the device table unchecked. See also the open note on
        // `MoeLayer`.
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("moe_tetra_rows: {e}")))?;
        Ok((CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()), Shape::from((nmat, npairs, d_out))))
    }
}
