//! 4E on CUDA: the whole decode token in one cooperative launch, N tokens a
//! launch (`llvq-cuda/kernels/pers_4e.cu`). Built by `bin/fusedrun`'s
//! `LLVQ_PERS_4E` (a measurement), and by a served config that names the
//! persistent decode (`crate::served_decode`, kernel form from position 0 on
//! the sealed 4B, operator T8 of 2026-11-05).
//!
//! Every phase of the kernel equals the served chain bit for bit on the host
//! (`llvq-cuda/tests/pers_matches_served.rs`), attention included as the
//! "kernel" form computes it (`attn_ref.rs`). So on the card, with every
//! decode row of the eager path sent to the attention kernel
//! (`Qwen3::set_kernel_from_for_measurement(0)`), 4E owes that path's tokens.
//!
//! The runtime refuses a launch that starts below its threshold. Installed,
//! the threshold is [`crate::kvq::KERNEL_FROM`]: an eager config that names
//! the kernel form (`configs/qwen3-4b-tetra-e4-kernel-from1280.json`) runs
//! candle's grouped arithmetic below 1,280 (cuBLAS matmuls, candle's
//! softmax), whose bits no hand kernel reproduces, so 4E owes that config's
//! tokens only from there. The served 4B config,
//! `configs/qwen3-4b-tetra-e4-4e-v4.json` (and its v6 twin,
//! `configs/qwen3-4b-tetra-e4-4e-v6.json`), names the persistent decode and
//! sends every decode row to the kernel from position 0, the form the
//! operator's T8 extension of 2026-11-05 licenses on the sealed 4B
//! (`docs/mesures/attn-t8-below-4b-2026-11-05.txt`). `Served::decoder` sets
//! the threshold to 0 for it (`Runtime4e::set_served_kernel_from`). A
//! measurement may lower it to 0 too
//! ([`Runtime4e::set_kernel_from_for_measurement`]), against an eager path
//! with every row on the kernel.
//!
//! What only the card can decide is RoPE: candle's `rope` is `__half`
//! arithmetic the driver's ptxas may fuse, so [`install`] runs candle's rope
//! on the card and keeps the form (pers_rope_pair) that matches it, or
//! refuses.
//!
//! This version reads the three sealed shapes: q and k Tetra under one
//! rotation, v int4, o int4 (4B, 14B) or Tetra under its own rotation (8B, a
//! `Pers4eOT` image a layer), a Tetra or int4 down per layer, and the head on
//! the embedding's int4 g64 table (4B, tied) or on its own (8B, 14B). Anything
//! else is refused at install by name.
//!
//! v9 (row form 7) reads the int4 words tiled. Under a v9 config the loader
//! wrote them so and the eager path reads the same buffers
//! (`fused_cuda::upload_int4`, `tv_q4_tile.cu`): the int4 bytes are on the
//! card once. On the bench door's file-order runtime v9 relays a copy.

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::{DType, Device, Tensor};
use half::f16;

use crate::model::{Embed, Head, KvCache, Proj, Qwen3};
use crate::persist_cuda::{int4_of, mlp_args, seg_tetra, tensor_addr, tetra_of};
use llvq_cuda::pers;

/// The positions 4E's attention scratch is sized for, at most. Defined in
/// `served_decode`, which every build compiles, so a harness checks its
/// prompts against it before a run on any host.
pub const MAX_CAP: usize = crate::served_decode::P4E_MAX_CAP;

pub struct Runtime4e {
    cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    /// The phase-stamp symbol of the same form and block size, when one
    /// exists (512 threads): [`Runtime4e::decode_stamped`].
    f_stamp: Option<CudaFunction>,
    pub threads: u32,
    /// The module's row form (`pers::P4E_ROWS_FORMS`).
    pub rows_form: u32,
    pub grid: u32,
    smem: u32,
    /// The launch asks for more shared bytes than the form needs.
    padded: bool,
    pub form: u32,
    layers_host: Vec<pers::P4eLayerHost>,
    args_host: pers::P4eArgsHost,
    layers: CudaSlice<u8>,
    args: CudaSlice<u8>,
    /// The barrier counter: its address is in the images; the host writes it
    /// back to 0 after a faulted launch ([`pers::p4e_settle`]).
    bar: CudaSlice<u32>,
    abort: CudaSlice<u32>,
    nan: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    max_tokens: u32,
    bar_base: u32,
    n_layers: u32,
    /// Buffers whose addresses live in the images: held, never read.
    _held16: Vec<CudaSlice<u16>>,
    _held32: Vec<CudaSlice<u32>>,
    _heldf: Vec<CudaSlice<f32>>,
    _held64: Vec<CudaSlice<u64>>,
    /// The Tetra o images (`Pers4eOT`), one a layer that has one.
    _held8: Vec<CudaSlice<u8>>,
    /// The (cap, K, V addresses) the uploaded images carry.
    kv_seen: Vec<(usize, u64, u64)>,
    /// The first position a launch may start at: the served threshold, or 0
    /// for a measurement against the kernel on every row.
    kernel_from: usize,
    pub lines: Vec<String>,
    /// The tile define of the model's runtime, for the verify module's text.
    tile_define: String,
    inter: u32,
    /// The verify pass, once [`Runtime4e::install_verify`] has run.
    verify: Option<Verify4e>,
    /// The tree verify pass, once [`Runtime4e::install_tree`] has run.
    tree: Option<Tree4e>,
}

/// The card's barrier counter and flags, as [`pers::p4e_settle`] reads and
/// clears them.
struct Counters4e<'a> {
    cuda: &'a llvq_cuda::gpu::Cuda,
    bar: &'a mut CudaSlice<u32>,
    abort: &'a mut CudaSlice<u32>,
    nan: &'a mut CudaSlice<u32>,
}

impl pers::P4eCounters for Counters4e<'_> {
    fn flags(&mut self) -> Result<(u32, u32), String> {
        Ok((self.cuda.down_u32(self.abort)?[0], self.cuda.down_u32(self.nan)?[0]))
    }

    fn zero(&mut self) -> Result<(), String> {
        let stream = self.cuda.stream();
        for d in [&mut *self.bar, &mut *self.abort, &mut *self.nan] {
            stream.memcpy_htod(&[0u32], d).map_err(|e| format!("H2D: {e}"))?;
        }
        stream.synchronize().map_err(|e| format!("sync: {e}"))
    }
}

/// 4E's verify pass (`pers_4ev.cu`): its own module (the v9 module and the
/// verify kernel), its block size and staging area, and its K-row buffers.
struct Verify4e {
    _cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    threads: u32,
    smem: u32,
    act_floats: u32,
    part_stride: u32,
    toks: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    _held16: Vec<CudaSlice<u16>>,
    held16_addr: [u64; 8],
    part: CudaSlice<f32>,
    tickets: CudaSlice<u32>,
    slots: CudaSlice<u64>,
}

/// The tree verify pass (`pers_4et.cu`, `docs/plan-verify-tree.md`): its own
/// module (the v9 module and the tree kernel), its area and the model's
/// slices, and buffers for `pers::P4T_RMAX` rows. Not a served path: S1
/// measures it (`bin/vtreebench`).
struct Tree4e {
    _cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    smem: u32,
    area: u32,
    /// The arms a launch picks from (job N1): each its slices and its form.
    arms: Vec<TreeArm>,
    /// The tiled module and the tree-tiled copies, when an arm is tiled.
    tiled: Option<TreeTiled>,
    part_stride: u32,
    ldx: u32,
    n_ot: u32,
    vocab: u32,
    toks: CudaSlice<u32>,
    depth: CudaSlice<u32>,
    path: CudaSlice<u8>,
    mv: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    logits: CudaSlice<u16>,
    /// x_a, x_b, xin, xn, q_raw, k_raw, v_raw, q_out, ctx, g_raw, u_raw, tk, tv
    _held16: Vec<CudaSlice<u16>>,
    held16_addr: [u64; 13],
    part: CudaSlice<f32>,
    fpart: CudaSlice<f32>,
    tickets: CudaSlice<u32>,
    ftickets: CudaSlice<u32>,
    slots: CudaSlice<u64>,
}

/// The kernel an arm of the tree pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeForm {
    /// `pers_4et_rows_*`: the words in their row layout (arms a0, a1).
    Rows,
    /// `pers_4et_tiled_*` (`P4T_TILED`): the tree-tiled copies staged by
    /// chunk (arm a2).
    Tiled,
    /// `pers_4et_stream_*` (`P4T_STREAM`): the tree-tiled copies streamed by
    /// each warp, the activation slice staged once a unit (arm a3).
    Stream,
    /// `pers_4et_a4_*` (`P4T_A4`): the stream unit with the fold's loads
    /// batched and the prologues on 16-byte vectors (arm a4).
    A4,
    /// `pers_4et_a5_*` (`P4T_A5`): a4 with the L2 prefetch of the next row
    /// phase's tile streams on 4E's schedule (arm a5).
    A5,
    /// `pers_4et_a6_*` (`P4T_A6`): a3 with the int4 down's input dealt over
    /// the grid (arm a6).
    A6,
    /// `pers_4et_a7_*` (`P4T_A7`): a6 with a5's prefetch points (arm a7).
    A7,
    /// `pers_4et_a3f_*` (`P4T_A3F`): a FLOOR arm, timing only, values void by
    /// design: a3 with no slice fold (slice 0 stored as if S were 1).
    A3f,
}

/// Where an arm's d_in slices come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreePlan {
    /// S1's `pers::p4t_splits`.
    S1,
    /// A proposal `pers::p4t_splits_balanced` keeps or replaces (job N1).
    Balanced([u32; 5]),
    /// `pers::p4t_splits_stream` at the stream unit's cap (job N2).
    Stream,
    /// Slices given by the bench (`arm@q.o.g.d.h`), refused when a slice
    /// exceeds the stream unit's cap or an S is outside 1..=32.
    Explicit([u32; 5]),
}

/// One arm of the tree pass: the d_in slices of q/k/v, o, gate/up, down and
/// the head, the plan they came from, and the kernel.
#[derive(Clone, Debug)]
pub struct TreeArm {
    pub name: String,
    pub splits: [u32; 5],
    pub plan: TreePlan,
    pub form: TreeForm,
    /// Why the arm cannot run (its module did not build), or None.
    pub void: Option<String>,
}

/// An arm asked of [`Runtime4e::install_tree`]: its name, its plan, its
/// kernel.
pub type TreeArmSpec<'a> = (&'a str, TreePlan, TreeForm);

/// The tree-tiled arms' modules and the tree-tiled copies of every
/// projection 4E reads, built once at install from the same words
/// (`p4t_tile_words`), and the images of `Pers4eLayer` and `Pers4eArgs` that
/// point at them, re-uploaded when the caches move. The tiled (a2) and the
/// stream (a3) kernels read the same copies.
struct TreeTiled {
    _cudas: Vec<llvq_cuda::gpu::Cuda>,
    /// The tiled symbol, when a tiled arm is installed.
    f: Option<CudaFunction>,
    /// The stream symbol, when a stream arm is installed.
    fs: Option<CudaFunction>,
    /// The a4 symbol, when an a4 arm is installed.
    f4: Option<CudaFunction>,
    /// The a5 symbol, when an a5 arm is installed.
    f5: Option<CudaFunction>,
    /// The a6, a7 and a3f symbols, when such an arm is installed.
    fx: [Option<CudaFunction>; 3],
    _held: Vec<CudaSlice<u32>>,
    /// Each layer's q, k, v, o, gate, up and down words in the copies.
    swaps: Vec<[u64; 7]>,
    head: u64,
    layers: CudaSlice<u8>,
    args: CudaSlice<u8>,
    seen: Vec<(usize, u64, u64)>,
}

/// What one tree launch returns: each row's argmax, its binary16 logits when
/// asked (`[rows][vocab]`), the phase clocks when stamped.
pub struct TreeOut {
    pub ring: Vec<u32>,
    pub logits: Vec<u16>,
    pub stamps: Vec<u64>,
}

fn addr16(s: &CudaSlice<u16>, st: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> u64 {
    s.device_ptr(st).0
}

/// Run candle's rope on the card and each of the three forms on the same
/// quads; the matching forms, in order.
fn rope_forms(cuda: &llvq_cuda::gpu::Cuda, cos: &Tensor, sin: &Tensor, device: &Device) -> anyhow::Result<Vec<u32>> {
    let positions: Vec<u32> = vec![0, 1, 2, 5, 31, 100, 257, 1000, 1279, 1280, 2047, 4095, 8191];
    let t = positions.len();
    let idx = Tensor::from_vec(positions, t, device)?;
    let (c, s) = (cos.index_select(&idx, 0)?.contiguous()?, sin.index_select(&idx, 0)?.contiguous()?);
    let mut rng = llvq_core::SplitMix64::new(0x40e5);
    let heads = 8usize;
    let q: Vec<f16> = (0..heads * t * 128).map(|_| f16::from_f64(rng.next_gaussian() * 3.0)).collect();
    let qt = Tensor::from_vec(q.clone(), (1, heads, t, 128), device)?;
    let y = candle_nn::rotary_emb::rope(&qt, &c, &s)?;
    let y: Vec<u16> = y.flatten_all()?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
    let cb: Vec<u16> = c.flatten_all()?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
    let sb: Vec<u16> = s.flatten_all()?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect();
    let (mut a, mut b, mut cc, mut ss, mut w1, mut w2) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for h in 0..heads {
        for p in 0..t {
            for i in 0..64 {
                let at = (h * t + p) * 128;
                a.push(q[at + i].to_bits());
                b.push(q[at + i + 64].to_bits());
                cc.push(cb[p * 64 + i]);
                ss.push(sb[p * 64 + i]);
                w1.push(y[at + i]);
                w2.push(y[at + i + 64]);
            }
        }
    }
    let n = a.len() as u32;
    let up = |v: &Vec<u16>| cuda.up_u16(v).map_err(anyhow::Error::msg);
    let (da, db, dc, ds) = (up(&a)?, up(&b)?, up(&cc)?, up(&ss)?);
    let mut found = Vec::new();
    for form in 0..3u32 {
        let f = cuda.func(&format!("pers_rope_probe_f{form}")).map_err(anyhow::Error::msg)?;
        let mut o1 = cuda.zeros_u16(n as usize).map_err(anyhow::Error::msg)?;
        let mut o2 = cuda.zeros_u16(n as usize).map_err(anyhow::Error::msg)?;
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = cuda.stream().launch_builder(&f);
        l.arg(&da).arg(&db).arg(&dc).arg(&ds).arg(&mut o1).arg(&mut o2).arg(&n);
        // SAFETY: the probe's own signature, n quads in every buffer.
        unsafe { l.launch(cfg) }.map_err(|e| anyhow::anyhow!("rope probe: {e}"))?;
        cuda.sync().map_err(anyhow::Error::msg)?;
        if cuda.down_u16(&o1).map_err(anyhow::Error::msg)? == w1 && cuda.down_u16(&o2).map_err(anyhow::Error::msg)? == w2 {
            found.push(form);
        }
    }
    Ok(found)
}

/// Build 4E for `model`, check RoPE against candle on the card, and pick the
/// block size whose kernel does not spill. `max_tokens` bounds one launch.
/// `threads`: `None` takes 768 where the card's module does not spill, else
/// 512; `Some(n)` takes n (512 or 768) or refuses, for a measurement that
/// times both geometries.
pub fn install(model: &Qwen3, device: &Device, max_tokens: u32, threads: Option<u32>) -> anyhow::Result<Runtime4e> {
    install_rows(model, device, max_tokens, threads, 1)
}

/// [`install`] with the module's row form (`pers::P4E_ROWS_FORMS`): 1 is v3;
/// 2 is v4, what the served config runs since 2026-11-05; 3 is v5; 4 is v6.
/// `LLVQ_PERS_4E` times `pers::P4E_BENCH_FORMS`.
pub fn install_rows(model: &Qwen3, device: &Device, max_tokens: u32, threads: Option<u32>, rows_form: u32) -> anyhow::Result<Runtime4e> {
    install_rows_as(model, device, max_tokens, threads, rows_form, None)
}

/// [`install_rows`], the launch asking for at least the dynamic shared bytes
/// of row form `smem_of` (a measurement: the same code and bits, with the
/// L1/shared carve-out a larger form forces). Its label carries an `s`.
pub fn install_rows_as(
    model: &Qwen3,
    device: &Device,
    max_tokens: u32,
    threads: Option<u32>,
    rows_form: u32,
    smem_of: Option<u32>,
) -> anyhow::Result<Runtime4e> {
    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let mut lines = Vec::new();
    let cfg = model.config();
    let (n_kv, groups, hd) = model.attn_geometry();
    if hd != 128 {
        anyhow::bail!("4E: head_dim {hd}, the kernel carries 128");
    }
    let n_heads = n_kv * groups;
    let first = model.blocks.first().ok_or_else(|| anyhow::anyhow!("4E: a model with no block"))?;
    let (_, _, _, q0, _, _, _) = first.attn_parts();
    let (qv0, lat0) = tetra_of(q0).ok_or_else(|| anyhow::anyhow!("4E: layer 0's q_proj is not a Tetra48 stream on CUDA"))?;
    let frt = lat0.runtime().clone();
    let tables = frt.table_view().ok_or_else(|| anyhow::anyhow!("4E: no Tetra tables"))?;
    let (hidden, inter) = (qv0.d_in, cfg.intermediate_size as u32);

    let text = format!("{}{}", pers::p4e_rows_define(rows_form).map_err(anyhow::Error::msg)?, pers::pers_unit_text(&frt.tile_define()));
    let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
    lines.push(format!(
        "4E: persistent module {} bytes, sha256 {}, row form {rows_form}, own module on candle's stream",
        src.text.len(),
        src.sha256
    ));
    let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;

    // RoPE: the card decides.
    let (norm_w, embed, head, cos, sin) = model.pers_4e_parts();
    let forms = rope_forms(&cuda, cos, sin, device)?;
    let form = *forms.first().ok_or_else(|| {
        anyhow::anyhow!("4E: none of the three RoPE forms equals candle's rope on this card; 4E stays off")
    })?;
    lines.push(format!("4E: RoPE forms equal to candle's rope on this card: {forms:?}; running form {form}"));

    // The block size: 768 where the card's module does not spill, else 512.
    let own = pers::p4e_smem_bytes_form(rows_form, hidden, inter, groups as u32) + pers::p4e_scratch_bytes(rows_form);
    let smem = smem_of.map_or(own, |o| own.max(pers::p4e_smem_bytes_form(o, hidden, inter, groups as u32) + pers::p4e_scratch_bytes(o)));
    if smem != own {
        lines.push(format!("4E: row form {rows_form} asks for {smem} B of dynamic shared memory, its own {own} B padded (measurement)"));
    }
    let mut chosen = None;
    let sizes: Vec<u32> = match threads {
        None => vec![768, 512],
        Some(n) => vec![n],
    };
    for threads in sizes {
        let sym = pers::p4e_symbol(form, threads).map_err(anyhow::Error::msg)?;
        let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
        let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
            .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
        lines.push(format!("4E: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes));
        if r.local_bytes == 0 && occ >= 1 {
            chosen = Some((f, threads));
            break;
        }
    }
    // The refusal names what each block size read (step2-4e did not print it).
    let (f, threads) = chosen.ok_or_else(|| anyhow::anyhow!("4E: every block size asked for spills or fails its occupancy: {}", lines.join("; ")))?;
    // The stamp symbol of the chosen geometry, kept where one CTA a SM fits.
    // It may hold local memory where the token symbol does not (768 threads
    // under the devel ptxas): its launch runs outside the timed rounds, and
    // its token time is printed beside the timed one.
    let f_stamp = match pers::p4e_stamp_symbol(form, threads) {
        Ok(sym) => {
            let fs = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
            let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
            let occ = fs
                .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
            lines.push(format!("4E: {sym}: {} registers, {} B local, {occ} resident a SM (phase stamps, outside the timed rounds)", r.num_regs, r.local_bytes));
            (occ >= 1).then_some(fs)
        }
        Err(e) => {
            lines.push(e);
            None
        }
    };
    let grid = cuda.device().map_err(anyhow::Error::msg)?.sm_count as u32;

    // The tied table.
    let emb = match embed {
        Embed::Q8(t) => t.as_any().and_then(|a| a.downcast_ref::<crate::fused_cuda::CudaEmbed>()).and_then(|c| c.q4_view()),
        Embed::Dense(_) => None,
    }
    .ok_or_else(|| anyhow::anyhow!("4E: the embedding is not the int4 g64 table (LLVQ_EMBED=q4)"))?;
    // The head: the embedding's own table when tied (the 4B), a table of its
    // own when not (the 8B and the 14B), of the same shape and group, read by
    // the head phase through its own pointers.
    let hv = match head {
        Head::Q8(t) => t.as_any().and_then(|a| a.downcast_ref::<crate::fused_cuda::CudaEmbed>()).and_then(|c| c.q4_view()),
        Head::Dense(_) => None,
    }
    .ok_or_else(|| anyhow::anyhow!("4E: the head is not an int4 g64 table (LLVQ_EMBED=q4)"))?;
    if emb.d != hidden || emb.vocab % 8 != 0 || (hv.vocab, hv.d, hv.gpr) != (emb.vocab, emb.d, emb.gpr) {
        anyhow::bail!(
            "4E: the head table ({} x {}, {} groups a row) is not the embedding's shape ({} x {}, {})",
            hv.vocab, hv.d, hv.gpr, emb.vocab, emb.d, emb.gpr
        );
    }
    let tied = hv.words == emb.words;
    lines.push(format!(
        "4E: head {} ({} rows)",
        if tied { "tied to the embedding's table" } else { "on a table of its own (untied)" },
        hv.vocab
    ));
    if cos.dtype() != DType::F16 || norm_w.dtype() != DType::F16 {
        anyhow::bail!("4E: the rotary tables and norms must be binary16");
    }

    // Scratch, counters, the SiLU table candle's own activation fills.
    let silu_bits: Vec<u16> = {
        let (_, _, act, _, _, _) = first.mlp_parts();
        if !matches!(act, candle_nn::Activation::Silu) {
            anyhow::bail!("4E: the activation is {act:?}, the segment carries SiLU");
        }
        let pats: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        Tensor::from_vec(pats, 65536, device)?.apply(&act)?.to_vec1::<f16>()?.iter().map(|v| v.to_bits()).collect()
    };
    let e = |r: Result<CudaSlice<u16>, String>| r.map_err(anyhow::Error::msg);
    let silu = e(cuda.up_u16(&silu_bits))?;
    let act = e(cuda.zeros_u16(inter as usize))?;
    let (q_raw, k_raw, v_raw) = (e(cuda.zeros_u16(n_heads * 128))?, e(cuda.zeros_u16(n_kv * 128))?, e(cuda.zeros_u16(n_kv * 128))?);
    let (q_out, ctx) = (e(cuda.zeros_u16(n_heads * 128))?, e(cuda.zeros_u16(n_heads * 128))?);
    let (x_a, x_b) = (e(cuda.zeros_u16(hidden as usize))?, e(cuda.zeros_u16(hidden as usize))?);
    // The split states: [n_kv][ceil(cap / 256)][groups][130] floats at most.
    let scratch = cuda.zeros_f32(n_heads * MAX_CAP.div_ceil(256) * 130).map_err(anyhow::Error::msg)?;
    let ticket = cuda.up_u32(&vec![0u32; n_kv]).map_err(anyhow::Error::msg)?;
    let bar = cuda.up_u32(&[0]).map_err(anyhow::Error::msg)?;
    let abort = cuda.up_u32(&[0]).map_err(anyhow::Error::msg)?;
    let nan = cuda.up_u32(&[0]).map_err(anyhow::Error::msg)?;
    let slots = cuda.up_u64(&[0, 0]).map_err(anyhow::Error::msg)?;
    let ring = cuda.up_u32(&vec![u32::MAX; max_tokens as usize]).map_err(anyhow::Error::msg)?;
    let gu: Vec<u32> = pers::slices(inter, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let dn: Vec<u32> = pers::slices(hidden, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let (d_gu, d_dn) = (cuda.up_u32(&gu).map_err(anyhow::Error::msg)?, cuda.up_u32(&dn).map_err(anyhow::Error::msg)?);
    let a32 = |s: &CudaSlice<u32>| s.device_ptr(&stream).0;
    let base = pers::SegArgsHost {
        silu: addr16(&silu, &stream),
        act: addr16(&act, &stream),
        tables: [tables.rows, tables.pref, tables.br, tables.suf, tables.invnorm],
        slice_gu: a32(&d_gu),
        slice_dn: a32(&d_dn),
        bar: a32(&bar),
        abort_flag: a32(&abort),
        hidden,
        inter,
        ..Default::default()
    };

    let mut layers_host = Vec::with_capacity(model.blocks.len());
    let mut int4_layout: Option<crate::fused::Int4Layout> = None;
    let mut held_ot: Vec<CudaSlice<u8>> = Vec::new();
    for (l, block) in model.blocks.iter().enumerate() {
        let (ln1, qn, kn, qp, kp, vp, op) = block.attn_parts();
        let ctxt = |m: &str| anyhow::anyhow!("4E: layer {l}: {m}");
        let (qv, _) = tetra_of(qp).ok_or_else(|| ctxt("q_proj is not a Tetra48 stream"))?;
        let (kv, _) = tetra_of(kp).ok_or_else(|| ctxt("k_proj is not a Tetra48 stream"))?;
        let key = qv.rotation.ok_or_else(|| ctxt("q_proj carries no rotation"))?;
        if kv.rotation != Some(key) {
            return Err(ctxt("q and k are not under one rotation (LLVQ_ROT_SHARE=1 is the served form)"));
        }
        let ra = frt.rot_view(&key).ok_or_else(|| ctxt("no rotation buffers for q/k"))?;
        if ra.n != hidden || pers::rot_mk(ra.n).map_err(anyhow::Error::msg)? != (ra.m, ra.k) {
            return Err(ctxt("the q/k rotation is not the hidden width's"));
        }
        let vq = int4_of(vp).ok_or_else(|| ctxt("v_proj is not an int4 record"))?.int4_view();
        if vq.d_in != hidden || vq.d_out != (n_kv * 128) as u32 {
            return Err(ctxt("v has an unexpected shape"));
        }
        // o: an int4 record (the 4B, the 14B), or a Tetra stream under its own
        // rotation at the context's width (the 8B), named by a Pers4eOT image.
        let d_ctx = (n_heads * 128) as u32;
        let (oq, ot_addr) = match (int4_of(op), tetra_of(op)) {
            (Some(q), _) => {
                let oq = q.int4_view();
                if oq.d_in != d_ctx || oq.d_out != hidden {
                    return Err(ctxt("o has an unexpected shape"));
                }
                (Some(oq), 0u64)
            }
            (None, Some((ov, _))) => {
                if ov.d_in != d_ctx || ov.d_out != hidden {
                    return Err(ctxt("the Tetra o has an unexpected shape"));
                }
                let okey = ov.rotation.ok_or_else(|| ctxt("the Tetra o_proj carries no rotation"))?;
                let ro = frt.rot_view(&okey).ok_or_else(|| ctxt("no rotation buffers for o"))?;
                if ro.n != d_ctx || pers::rot_mk(ro.n).map_err(anyhow::Error::msg)? != (ro.m, ro.k) {
                    return Err(ctxt("the o rotation is not the context's width"));
                }
                let img = pers::P4eOTHost { t: seg_tetra(&ov), signs: ro.signbits, small: ro.small, m: ro.m, k: ro.k, inv: ro.inv };
                let buf = stream.clone_htod(&img.bytes()).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
                let a = buf.device_ptr(&stream).0;
                held_ot.push(buf);
                (None, a)
            }
            (None, None) => return Err(ctxt("o_proj is neither an int4 record nor a Tetra48 stream")),
        };
        if qv.d_out != (n_heads * 128) as u32 || kv.d_out != (n_kv * 128) as u32 || qv.d_in != hidden || kv.d_in != hidden {
            return Err(ctxt("q or k has an unexpected shape"));
        }
        let (mlp, tetra_down, down_layout) = mlp_args(block, &frt, &base, &stream)?;
        // One word order for every int4 record 4E reads: the runtime chose it
        // for all of them at load.
        for (what, got) in [("v_proj", Some(vq.layout)), ("o_proj", oq.map(|o| o.layout)), ("down_proj", down_layout)] {
            let Some(got) = got else { continue };
            match int4_layout {
                None => int4_layout = Some(got),
                Some(want) if want != got => {
                    return Err(ctxt(&format!("{what} holds {} words, an earlier int4 record {}", got.name(), want.name())));
                }
                Some(_) => {}
            }
        }
        layers_host.push(pers::P4eLayerHost {
            ln1: tensor_addr(ln1, &stream)?,
            signs_a: ra.signbits,
            small_a: ra.small,
            m_a: ra.m,
            k_a: ra.k,
            inv_a: ra.inv,
            qkv: pers::P4eQkvHost {
                q: seg_tetra(&qv),
                k: seg_tetra(&kv),
                vw: vq.wq,
                vsc: vq.scales,
                vbi: vq.biases,
                v_dout: vq.d_out,
                v_gpr: vq.gpr,
                d_in: hidden,
                out_q: addr16(&q_raw, &stream),
                out_k: addr16(&k_raw, &stream),
                out_v: addr16(&v_raw, &stream),
            },
            q_norm: tensor_addr(qn, &stream)?,
            k_norm: tensor_addr(kn, &stream)?,
            kc: 0,
            vc: 0,
            ow: oq.map_or(0, |o| o.wq),
            osc: oq.map_or(0, |o| o.scales),
            obi: oq.map_or(0, |o| o.biases),
            o_gpr: oq.map_or(0, |o| o.gpr),
            tetra_down,
            mlp,
            ot: ot_addr,
        });
    }
    let args_host = pers::P4eArgsHost {
        ptrs: [
            0,
            emb.words,
            emb.scales,
            emb.biases,
            tensor_addr(norm_w, &stream)?,
            tensor_addr(cos, &stream)?,
            tensor_addr(sin, &stream)?,
            addr16(&x_a, &stream),
            addr16(&x_b, &stream),
            addr16(&q_out, &stream),
            addr16(&ctx, &stream),
            scratch.device_ptr(&stream).0,
            a32(&bar),
            a32(&abort),
            slots.device_ptr(&stream).0,
            a32(&nan),
            tables.rows,
            tables.pref,
            tables.br,
            tables.suf,
            tables.invnorm,
            a32(&ticket),
        ],
        n_layers: model.blocks.len() as u32,
        vocab: emb.vocab,
        hidden,
        emb_gpr: emb.gpr,
        n_heads: n_heads as u32,
        n_kv: n_kv as u32,
        groups: groups as u32,
        cap: 0,
        eps: cfg.rms_norm_eps as f32,
        c: llvq_cuda::attn_ref::score_scale(128),
        head: [hv.words, hv.scales, hv.biases],
    };
    let tetra_o = layers_host.iter().filter(|y| y.ot != 0).count();
    if tetra_o > 0 {
        lines.push(format!("4E: {tetra_o} Tetra o_proj read under their own rotation, {} int4", layers_host.len() - tetra_o));
    }
    if rows_form >= 4 {
        let mut addrs = vec![
            ("act".to_string(), base.act),
            ("x_a".to_string(), addr16(&x_a, &stream)),
            ("x_b".to_string(), addr16(&x_b, &stream)),
        ];
        for (l, y) in layers_host.iter().enumerate() {
            addrs.push((format!("layer {l} ln1"), y.ln1));
            addrs.push((format!("layer {l} ln2"), y.mlp.alpha));
        }
        // v7 also widens the o phase's input from vectors, and stages the
        // rotations' sign words in a 576-word area
        if rows_form >= 5 {
            addrs.push(("ctx".to_string(), addr16(&ctx, &stream)));
            if inter.div_ceil(32) > 576 || hidden.div_ceil(32) > 576 {
                anyhow::bail!("4E v7: the sign words of width {} do not fit the 576 of the v7 area", inter.max(hidden));
            }
        }
        pers::p4e_v6_inputs(&addrs, hidden, inter).map_err(anyhow::Error::msg)?;
        let label = pers::P4E_ROWS_FORMS.iter().find(|f| f.0 == rows_form).map_or("v?", |f| f.1);
        lines.push(format!("4E {label}: the {} prologue inputs are 16-byte aligned; the vector path runs on every layer", addrs.len()));
    }
    // The int4 words against the row form. v3 to v8 read the file's order
    // (pers_q4_row); a runtime that holds tiled words (a v9 config) is
    // refused by name, since those forms would read wrong words. v9 reads
    // tiled words: in place when the runtime holds them so (a v9 config, the
    // served form, the int4 bytes once on the card), else from a copy relaid
    // here (the bench door's measurement form, the bytes twice).
    let tiled_in_place = int4_layout == Some(crate::fused::Int4Layout::Tiled);
    if tiled_in_place && rows_form < 7 {
        let label = pers::P4E_ROWS_FORMS.iter().find(|f| f.0 == rows_form).map_or("v?", |f| f.1);
        anyhow::bail!(
            "4E {label}: the runtime holds its int4 words tiled (a config whose rows are \"v9\"), and row form \
             {rows_form} reads the file's order; load with that form's config, or serve v9"
        );
    }
    if tiled_in_place {
        lines.push(format!(
            "4E v9: the int4 records are read in place, tiled at load and held once on the card ({} layers)",
            layers_host.len()
        ));
    }
    // v9 (form 7) on a runtime that holds the file's order: the int4 records
    // 4E reads (v, o, the int4 downs), relaid once into buffers of the same
    // size (pers_q4_tile_index, llvq_pers.cuh); the served records stay as
    // they are for the prefill and the eager path, so this measurement form
    // holds the int4 bytes twice on the card.
    let mut tiled: Vec<CudaSlice<u32>> = Vec::new();
    if rows_form >= 7 && !tiled_in_place {
        let relay = cuda.func("pers_4e_q4_tile_relayout").map_err(anyhow::Error::msg)?;
        let mut bytes = 0u64;
        {
            let mut tile = |src: u64, d_out: u32, d_in: u32| -> anyhow::Result<u64> {
                if !d_in.is_multiple_of(32) || !src.is_multiple_of(16) {
                    anyhow::bail!("4E v9: an int4 record of d_in {d_in} at {src:#x} cannot be read by 16-byte tiles");
                }
                let nwords = d_in / 8;
                let buf = cuda.zeros_u32(d_out as usize * nwords as usize).map_err(anyhow::Error::msg)?;
                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = cuda.stream().launch_builder(&relay);
                b.arg(&src).arg(&buf).arg(&d_out).arg(&nwords);
                // SAFETY: src is a live int4 record of d_out x nwords words, buf
                // a fresh buffer of the same size; the kernel only moves words.
                unsafe { b.launch(cfg) }.map_err(|e| anyhow::anyhow!("pers_4e_q4_tile_relayout: {e}"))?;
                let p = buf.device_ptr(&stream).0;
                bytes += 4 * u64::from(d_out) * u64::from(nwords);
                tiled.push(buf);
                Ok(p)
            };
            for y in layers_host.iter_mut() {
                y.qkv.vw = tile(y.qkv.vw, y.qkv.v_dout, y.qkv.d_in)?;
                if y.ot == 0 {
                    y.ow = tile(y.ow, hidden, (n_heads * 128) as u32)?;
                }
                if !y.tetra_down {
                    y.mlp.q4w = tile(y.mlp.q4w, hidden, inter)?;
                }
            }
        }
        cuda.sync().map_err(anyhow::Error::msg)?;
        lines.push(format!(
            "4E v9: {} int4 records relaid into tiled words, {:.3} GB more on the card (measurement form)",
            tiled.len(),
            bytes as f64 / 1e9
        ));
    }
    let layers = stream.clone_htod(&vec![0u8; pers::P4E_LAYER_BYTES * layers_host.len()]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
    let args = stream.clone_htod(&vec![0u8; pers::P4E_ARGS_BYTES]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
    lines.push(format!("4E: {} layers, grid {grid} x {threads}, {} barriers a token", layers_host.len(), pers::n_bar_4e(layers_host.len() as u32)));
    let n_layers = layers_host.len() as u32;
    Ok(Runtime4e {
        cuda,
        f,
        f_stamp,
        threads,
        rows_form,
        grid,
        padded: smem != own,
        smem,
        form,
        layers_host,
        args_host,
        layers,
        args,
        bar,
        abort,
        nan,
        ring,
        max_tokens,
        bar_base: 0,
        n_layers,
        _held16: vec![silu, act, q_raw, k_raw, v_raw, q_out, ctx, x_a, x_b],
        _held32: {
            let mut v = vec![d_gu, d_dn, ticket];
            v.extend(tiled);
            v
        },
        _heldf: vec![scratch],
        _held64: vec![slots],
        _held8: held_ot,
        kv_seen: Vec::new(),
        kernel_from: crate::kvq::KERNEL_FROM,
        lines,
        tile_define: frt.tile_define(),
        inter,
        verify: None,
        tree: None,
    })
}

impl Runtime4e {
    /// Decode up to `n` tokens from `tok` at position `pos0`, the caches
    /// holding `pos0` positions, stopping after `eos`. Returns the tokens
    /// made; the caches hold `pos0 + made` positions after.
    pub fn decode(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32) -> anyhow::Result<Vec<u32>> {
        Ok(self.launch(caches, pos0, tok, n, eos, false)?.0)
    }

    /// The form and the geometry, as the measurement prints them: "v4 at 512
    /// threads".
    pub fn label(&self) -> String {
        let form = pers::P4E_ROWS_FORMS.iter().find(|f| f.0 == self.rows_form).map_or("v?", |f| f.1);
        let pad = if self.padded { "s" } else { "" };
        format!("{form}{pad} at {} threads", self.threads)
    }

    /// The first position a launch may start at.
    pub fn kernel_from(&self) -> usize {
        self.kernel_from
    }

    /// Lower (or restore) the threshold, for a measurement only: below
    /// [`crate::kvq::KERNEL_FROM`] 4E's tokens are the kernel form's, which
    /// an eager config does not run there. The eager side of such a
    /// measurement must run the same form
    /// (`Qwen3::set_kernel_from_for_measurement`). A served config that names
    /// the persistent decode sets 0 through `Served::decoder` instead.
    pub fn set_kernel_from_for_measurement(&mut self, from: usize) {
        self.kernel_from = from;
    }

    /// The served config's threshold (`Served::decoder` only): 0 under a
    /// config that names the persistent decode, the kernel form the
    /// operator's T8 extension of 2026-11-05 licenses on the sealed 4B at
    /// every position.
    pub(crate) fn set_served_kernel_from(&mut self, from: usize) {
        self.kernel_from = from;
    }

    /// Does this runtime carry the phase-stamp symbol?
    pub fn can_stamp(&self) -> bool {
        self.f_stamp.is_some()
    }

    /// [`Self::decode`] on the phase-stamp symbol (a measurement): the same
    /// tokens, and the stamps `[made][p4e_events][grid]` of the grid clock,
    /// in ns, for `pers::p4e_phase_report`.
    pub fn decode_stamped(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        self.launch(caches, pos0, tok, n, eos, true)
    }

    /// The layers and the grid a stamp buffer is laid out by.
    pub fn stamp_shape(&self) -> (u32, u32) {
        (self.n_layers, self.grid)
    }

    /// The two flags after a launch; on a fault, the counter, both flags and
    /// the barrier base back to 0, so one fault fails one request and not
    /// every request after it.
    fn settle(&mut self) -> anyhow::Result<Option<pers::P4eFault>> {
        let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
        pers::p4e_settle(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)
    }

    fn launch(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32, stamped: bool) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        if stamped && self.f_stamp.is_none() {
            anyhow::bail!("4E: no phase-stamp symbol at {} threads", self.threads);
        }
        if n == 0 || n > self.max_tokens {
            anyhow::bail!("4E: {n} tokens a launch, the runtime holds {}", self.max_tokens);
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!(
                "4E: a launch at position {pos0}, below the kernel threshold {}: the served config runs the grouped \
                 arithmetic there (cuBLAS), which 4E does not compute; decode those positions on the eager path",
                self.kernel_from
            );
        }
        self.prepare(caches, pos0, n as usize)?;
        let stream = self.cuda.stream().clone();
        stream
            .memcpy_htod(&vec![u32::MAX; self.max_tokens as usize], &mut self.ring)
            .map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let n_stamps = if stamped { n as usize * pers::p4e_events(self.n_layers) as usize * self.grid as usize } else { 0 };
        let stamps: Option<CudaSlice<u64>> = if stamped {
            Some(stream.alloc_zeros::<u64>(n_stamps).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        let launch = pers::P4eLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_tokens: n,
            tok0: tok,
            eos,
            pad: 0,
            ring: self.ring.device_ptr(&stream).0,
            stamps: stamps.as_ref().map_or(0, |b| b.device_ptr(&stream).0),
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (self.threads, 1, 1), shared_mem_bytes: self.smem };
        let f = if stamped { self.f_stamp.as_ref().expect("checked above") } else { &self.f };
        let mut b = stream.launch_builder(f);
        b.arg(&self.args).arg(&launch);
        // SAFETY: the kernel's own signature (Pers4eArgs*, Pers4eLaunch), the
        // layouts checked on the host; one CTA a SM, past the occupancy check.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E: {e}"))?;
        let ring = self.cuda.down_u32(&self.ring).map_err(anyhow::Error::msg)?;
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E: a NaN logit; the eager path is the one to read"),
            None => {}
        }
        let (made, bar_base) = pers::p4e_after_launch(&ring, n, self.bar_base, self.n_layers, self.grid);
        self.bar_base = bar_base;
        for c in caches.iter_mut() {
            c.set_len(pos0 + made.len())?;
        }
        let stamps = match stamps {
            Some(d) => {
                let all = stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?;
                all[..made.len() * pers::p4e_events(self.n_layers) as usize * self.grid as usize].to_vec()
            }
            None => Vec::new(),
        };
        Ok((made, stamps))
    }

    /// Install the verify pass ([`Self::verify`]): its module (row form 7
    /// and `pers_4ev.cu`, compiled beside the served module, whose text does
    /// not move), its block size (768 where the symbol does not spill, else
    /// 512), the largest staging area the card's shared memory allows
    /// (`pers::p4v_act_floats`), and buffers for `pers::P4V_KMAX` rows.
    /// Refused below row form 7: the pass reads the int4 rows tiled.
    pub fn install_verify(&mut self) -> anyhow::Result<()> {
        if self.rows_form < 7 {
            anyhow::bail!("4E verify: the pass reads the int4 rows on tiled words (row form 7, v9); this runtime is form {}", self.rows_form);
        }
        if self.verify.is_some() {
            return Ok(());
        }
        // The pass stages a rotation's k rows of `small` in P4V_SSM floats.
        let kmax = pers::P4V_SSM / llvq_cuda::pers::ROT_KMAX;
        if let Some((l, k)) = self
            .layers_host
            .iter()
            .enumerate()
            .flat_map(|(l, y)| [(l, y.k_a), (l, y.mlp.k_h), (l, y.mlp.k_i)])
            .find(|&(_, k)| k > kmax)
        {
            anyhow::bail!("4E verify: layer {l} has a rotation of odd factor {k}, the pass stages {kmax} rows of `small`");
        }
        let stream = self.cuda.stream().clone();
        let text = pers::p4v_unit_text(&self.tile_define);
        let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
        self.lines.push(format!("4E verify: module {} bytes, sha256 {}, row form 7 and pers_4ev.cu, its own module", src.text.len(), src.sha256));
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;
        let dev = cuda.device().map_err(anyhow::Error::msg)?;
        let (hidden, n_heads, n_kv, groups) = (self.args_host.hidden, self.args_host.n_heads, self.args_host.n_kv, self.args_host.groups);
        let d_ctx = n_heads * 128;
        let act_floats = pers::p4v_act_floats(hidden, self.inter, d_ctx, groups, dev.shared_per_block_optin as u32).map_err(anyhow::Error::msg)?;
        let smem = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * act_floats;
        let k = pers::P4V_KMAX;
        // the kernel's group size: the rows after the staged `small` and the head's maxima
        let group = |w: u32| ((act_floats - pers::P4V_SSM - 2 * k) / w).clamp(1, k);
        self.lines.push(format!(
            "4E verify: {act_floats} floats of staging, {smem} B shared; rows a group at most {k}: {} at hidden {hidden}, {} at the context {d_ctx}, {} at inter {}",
            group(hidden),
            group(d_ctx),
            group(self.inter),
            self.inter
        ));
        let mut chosen = None;
        let mut why = Vec::new();
        for threads in [768u32, 512] {
            let sym = pers::p4v_symbol(self.form, threads).map_err(anyhow::Error::msg)?;
            let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
            let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
            let occ = f
                .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
            let line = format!("4E verify: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes);
            self.lines.push(line.clone());
            why.push(line);
            if occ >= 1 && (r.local_bytes == 0 || threads == 512) {
                chosen = Some((f, threads));
                break;
            }
        }
        let (f, threads) = chosen.ok_or_else(|| anyhow::anyhow!("4E verify: no block size fits one CTA a SM: {}", why.join("; ")))?;
        let e = |r: Result<CudaSlice<u16>, String>| r.map_err(anyhow::Error::msg);
        let (h, dq, dkv, i) = ((k * hidden) as usize, (k * d_ctx) as usize, (k * n_kv * 128) as usize, (k * self.inter) as usize);
        let held = vec![e(cuda.zeros_u16(h))?, e(cuda.zeros_u16(h))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(dkv))?, e(cuda.zeros_u16(dkv))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(i))?];
        let mut held16_addr = [0u64; 8];
        for (a, b) in held16_addr.iter_mut().zip(&held) {
            *a = addr16(b, &stream);
        }
        let part_stride = n_kv * (MAX_CAP as u32).div_ceil(256) * groups * 130;
        let part = cuda.zeros_f32((k * part_stride) as usize).map_err(anyhow::Error::msg)?;
        let tickets = cuda.up_u32(&vec![0u32; (k * n_kv) as usize]).map_err(anyhow::Error::msg)?;
        let slots = cuda.up_u64(&vec![0u64; k as usize]).map_err(anyhow::Error::msg)?;
        let toks = cuda.up_u32(&vec![0u32; k as usize]).map_err(anyhow::Error::msg)?;
        let ring = cuda.up_u32(&vec![u32::MAX; k as usize]).map_err(anyhow::Error::msg)?;
        let bytes = 2 * (2 * h + 3 * dq + 2 * dkv + i) + 4 * (k * part_stride) as usize;
        self.lines.push(format!("4E verify: {threads} threads; {:.3} GB of row buffers on the card", bytes as f64 / 1e9));
        self.verify = Some(Verify4e { _cuda: cuda, f, threads, smem, act_floats, part_stride, toks, ring, _held16: held, held16_addr, part, tickets, slots });
        Ok(())
    }

    /// The verify pass's block size, once installed.
    pub fn verify_threads(&self) -> Option<u32> {
        self.verify.as_ref().map(|v| v.threads)
    }

    /// One verify launch: `toks.len()` rows (1 to `pers::P4V_KMAX`), row r
    /// the token `toks[r]` at position `pos0 + r`, the caches holding `pos0`
    /// positions. Returns each row's argmax, the target's token after it; the
    /// caches hold `pos0 + toks.len()` after (rewind them to what is kept).
    /// Each row is the 4E token at its position bit for bit
    /// (`tests/pers_matches_served.rs::the_4e_verify_rows_are_the_served_chain`).
    pub fn verify(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32]) -> anyhow::Result<Vec<u32>> {
        Ok(self.verify_inner(caches, pos0, toks, false)?.0)
    }

    /// [`Self::verify`] with its phases stamped: the same rows and tokens,
    /// and `[1 + n_bar_4e][grid]` clocks (`pers::p4v_phases` reads them).
    /// A measurement launch.
    pub fn verify_stamped(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32]) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        self.verify_inner(caches, pos0, toks, true)
    }

    /// The layers and the grid a verify stamp buffer is laid out by.
    pub fn verify_stamp_shape(&self) -> (u32, u32) {
        (self.n_layers, self.grid)
    }

    fn verify_inner(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32], stamped: bool) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        let n = toks.len();
        if n == 0 || n > pers::P4V_KMAX as usize {
            anyhow::bail!("4E verify: {n} rows, the pass carries 1 to {}", pers::P4V_KMAX);
        }
        if self.verify.is_none() {
            anyhow::bail!("4E verify: not installed (Runtime4e::install_verify)");
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!("4E verify: a launch at position {pos0}, below the kernel threshold {}", self.kernel_from);
        }
        self.prepare(caches, pos0, n)?;
        let stream = self.cuda.stream().clone();
        let v = self.verify.as_mut().expect("checked above");
        stream.memcpy_htod(toks, &mut v.toks).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(&vec![u32::MAX; pers::P4V_KMAX as usize], &mut v.ring).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let a = v.held16_addr;
        let n_stamps = if stamped { (1 + pers::n_bar_4e(self.n_layers)) as usize * self.grid as usize } else { 0 };
        let stamps: Option<CudaSlice<u64>> = if stamped {
            Some(stream.alloc_zeros::<u64>(n_stamps).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        let launch = pers::P4vLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_rows: n as u32,
            act_floats: v.act_floats,
            toks: v.toks.device_ptr(&stream).0,
            ring: v.ring.device_ptr(&stream).0,
            x_a: a[0],
            x_b: a[1],
            q_raw: a[2],
            k_raw: a[3],
            v_raw: a[4],
            q_out: a[5],
            ctx: a[6],
            act: a[7],
            part: v.part.device_ptr(&stream).0,
            tickets: v.tickets.device_ptr(&stream).0,
            slots: v.slots.device_ptr(&stream).0,
            part_stride: v.part_stride,
            pad: 0,
            stamps: stamps.as_ref().map_or(0, |b| b.device_ptr(&stream).0),
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (v.threads, 1, 1), shared_mem_bytes: v.smem };
        let mut b = stream.launch_builder(&v.f);
        b.arg(&self.args).arg(&launch);
        // SAFETY: the kernel's own signature (Pers4eArgs*, Pers4vLaunch), the
        // layouts checked on the host; one CTA a SM, past the occupancy check.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E verify: {e}"))?;
        let ring = stream.clone_dtoh(&v.ring).map_err(|e| anyhow::anyhow!("D2H: {e}"))?;
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E verify: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E verify: a NaN logit; the eager path is the one to read"),
            None => {}
        }
        let made: Vec<u32> = ring[..n].to_vec();
        if made.contains(&u32::MAX) {
            // The barriers this launch passed are unknown: start over.
            let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
            pers::p4e_reset(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)?;
            anyhow::bail!("4E verify: a row wrote no token ({made:?}); the counters are reset for the next launch");
        }
        self.bar_base = self.bar_base.wrapping_add(pers::n_bar_4e(self.n_layers).wrapping_mul(self.grid));
        for c in caches.iter_mut() {
            c.set_len(pos0 + n)?;
        }
        let stamps = match stamps {
            Some(d) => stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?,
            None => Vec::new(),
        };
        Ok((made, stamps))
    }

    /// Install the tree verify pass ([`Self::tree`]): its module (row form 7
    /// and `pers_4et.cu`, its own module, so the served module's text does
    /// not move), 512 threads, the area `pers::p4t_area_floats` gives for 32
    /// rows, the model's slices on this grid (`pers::p4t_splits`), and
    /// buffers for `pers::P4T_RMAX` rows. Refused below row form 7 and when
    /// one CTA a SM does not fit. `arms`: the slices and forms a launch picks
    /// from ([`Self::tree`]'s `arm`); a tiled arm installs the tiled module
    /// and the tree-tiled copies.
    pub fn install_tree(&mut self, arms: &[TreeArmSpec]) -> anyhow::Result<()> {
        if self.rows_form < 7 {
            anyhow::bail!("4E tree: the pass reads the int4 rows on tiled words (row form 7, v9); this runtime is form {}", self.rows_form);
        }
        if self.tree.is_some() {
            return Ok(());
        }
        let stream = self.cuda.stream().clone();
        let text = pers::p4t_unit_text(&self.tile_define);
        let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
        self.lines.push(format!("4E tree: module {} bytes, sha256 {}, row form 7 and pers_4et.cu, its own module", src.text.len(), src.sha256));
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;
        let dev = cuda.device().map_err(anyhow::Error::msg)?;
        let (hidden, n_heads, n_kv, groups, vocab) =
            (self.args_host.hidden, self.args_host.n_heads, self.args_host.n_kv, self.args_host.groups, self.args_host.vocab);
        let (d_ctx, inter, threads) = (n_heads * 128, self.inter, 512u32);
        let nw = threads / 32;
        let widest = hidden.max(inter).max(d_ctx);
        let area = pers::p4t_area_floats(nw, 32, widest, groups);
        let smem = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * area;
        if smem > dev.shared_per_block_optin as u32 {
            anyhow::bail!("4E tree: {smem} B of shared memory a CTA, the card allows {}", dev.shared_per_block_optin);
        }
        let sym = pers::p4t_symbol(self.form).map_err(anyhow::Error::msg)?;
        let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
        let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
            .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
        self.lines.push(format!("4E tree: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes));
        if occ < 1 {
            anyhow::bail!("4E tree: {sym} fits no CTA on a SM at {threads} threads and {smem} B");
        }
        let s1 = pers::p4t_splits(hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
        if arms.is_empty() {
            anyhow::bail!("4E tree: no arm asked");
        }
        let mut arm_v = Vec::with_capacity(arms.len());
        let cap = pers::p4t_stream_cap(area, nw);
        let d_ins = [hidden, d_ctx, hidden, inter, hidden];
        for &(name, prop, form) in arms {
            let splits = match prop {
                TreePlan::S1 => s1,
                TreePlan::Balanced(p) => {
                    let b = pers::p4t_splits_balanced(p, hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
                    if b != p {
                        self.lines.push(format!("4E tree: arm {name}: the proposed slices {p:?} fill under 90 % on q/k/v or gate/up; {b:?} instead"));
                    }
                    b
                }
                TreePlan::Stream => pers::p4t_splits_stream(hidden, inter, d_ctx, n_kv, vocab, nw, self.grid, cap).map_err(anyhow::Error::msg)?,
                TreePlan::Explicit(p) => {
                    if p.iter().any(|&x| x == 0 || x > 32) || d_ins.iter().zip(p).any(|(&d, x)| pers::p4t_slice_steps(d, x) > cap) {
                        anyhow::bail!("4E tree: arm {name}: the explicit plan {p:?} has an S outside 1..=32 or under the stream unit's floor ({cap} sub-steps a slice)");
                    }
                    p
                }
            };
            if !matches!(form, TreeForm::Rows | TreeForm::Tiled) && d_ins.iter().zip(splits).any(|(&d, s)| pers::p4t_slice_steps(d, s) > cap) {
                anyhow::bail!("4E tree: arm {name}: slices {splits:?} exceed the stream unit's {cap} sub-steps a slice");
            }
            let plan = pers::p4t_plan(splits, hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
            let names = ["q/k/v", "o", "gate/up", "down", "head"];
            let plan_name = match prop {
                TreePlan::S1 => "S1's plan",
                TreePlan::Balanced(_) => "the balanced plan",
                TreePlan::Stream => "the stream plan",
                TreePlan::Explicit(_) => "an explicit plan",
            };
            self.lines.push(format!(
                "4E tree: arm {name} ({}, {plan_name}): slices {splits:?}; S, units, waves, fill, sub-steps a slice at most: {}",
                match form {
                    TreeForm::Rows => "rows",
                    TreeForm::Tiled => "tiled",
                    TreeForm::Stream => "stream",
                    TreeForm::A4 => "a4",
                    TreeForm::A5 => "a5",
                    TreeForm::A6 => "a6",
                    TreeForm::A7 => "a7",
                    TreeForm::A3f => "a3f (floor, values void)",
                },
                names
                    .iter()
                    .zip(plan)
                    .zip(splits.iter().zip(d_ins))
                    .map(|((n, (u, w, f)), (&s, d))| format!("{n} {s} {u} {w} {:.1} % {}", 100.0 * f, pers::p4t_slice_steps(d, s)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            arm_v.push(TreeArm { name: name.to_string(), splits, plan: prop, form, void: None });
        }
        if arm_v.iter().any(|a| !matches!(a.form, TreeForm::Rows | TreeForm::Tiled)) {
            self.lines.push(format!("4E tree: the stream unit's slices hold at most {cap} sub-steps (area {area} floats, {nw} warps, 32 staged rows)"));
        }
        let n_ot = self.layers_host.iter().filter(|y| y.ot != 0).count() as u32;
        let ldx = widest.next_multiple_of(8);
        let rm = pers::P4T_RMAX;
        let dkv = n_kv * 128;
        let l = self.layers_host.len() as u32;
        let sizes = [rm * hidden, rm * hidden, rm * ldx, rm * hidden, rm * d_ctx, rm * dkv, rm * dkv, rm * d_ctx, rm * d_ctx, rm * inter, rm * inter, l * rm * dkv, l * rm * dkv];
        let mut held = Vec::with_capacity(sizes.len());
        let mut held16_addr = [0u64; 13];
        for (a, &n) in held16_addr.iter_mut().zip(&sizes) {
            let b = cuda.zeros_u16(n as usize).map_err(anyhow::Error::msg)?;
            *a = addr16(&b, &stream);
            held.push(b);
        }
        let part_stride = n_kv * (MAX_CAP as u32).div_ceil(256) * groups * 130;
        let fp = arm_v.iter().map(|a| pers::p4t_fpart_floats(a.splits, hidden, inter, d_ctx, n_kv, vocab)).max().unwrap_or(0).max(1) as usize;
        let nft = pers::p4t_ftickets(hidden, inter, d_ctx, n_kv, vocab) as usize;
        let e32 = |r: Result<CudaSlice<u32>, String>| r.map_err(anyhow::Error::msg);
        let part = cuda.zeros_f32((rm * part_stride) as usize).map_err(anyhow::Error::msg)?;
        let fpart = cuda.zeros_f32(fp).map_err(anyhow::Error::msg)?;
        let tickets = e32(cuda.up_u32(&vec![0u32; (rm * n_kv) as usize]))?;
        let ftickets = e32(cuda.up_u32(&vec![0u32; nft]))?;
        let slots = cuda.up_u64(&vec![0u64; rm as usize]).map_err(anyhow::Error::msg)?;
        let toks = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let depth = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let mv = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let ring = e32(cuda.up_u32(&vec![u32::MAX; rm as usize]))?;
        let path = stream.alloc_zeros::<u8>((rm * rm) as usize).map_err(|e| anyhow::anyhow!("alloc: {e}"))?;
        let logits = cuda.zeros_u16((rm * vocab) as usize).map_err(anyhow::Error::msg)?;
        let bytes = 2 * sizes.iter().map(|&n| n as u64).sum::<u64>() + 4 * (rm * part_stride) as u64 + 4 * fp as u64 + 2 * (rm * vocab) as u64;
        self.lines.push(format!("4E tree: {threads} threads, {area} floats of area; {:.3} GB of row buffers", bytes as f64 / 1e9));
        let want = [TreeForm::Tiled, TreeForm::Stream, TreeForm::A4, TreeForm::A5, TreeForm::A6, TreeForm::A7, TreeForm::A3f]
            .map(|f| arm_v.iter().any(|a| a.form == f));
        let tiled = if want.iter().any(|&w| w) { Some(self.install_tiled(smem, threads, want)?) } else { None };
        // an arm whose module did not build is void; the others run
        if let Some(t) = &tiled {
            for a in arm_v.iter_mut() {
                let built = match a.form {
                    TreeForm::Rows => true,
                    TreeForm::Tiled => t.f.is_some(),
                    TreeForm::Stream => t.fs.is_some(),
                    TreeForm::A4 => t.f4.is_some(),
                    TreeForm::A5 => t.f5.is_some(),
                    TreeForm::A6 => t.fx[0].is_some(),
                    TreeForm::A7 => t.fx[1].is_some(),
                    TreeForm::A3f => t.fx[2].is_some(),
                };
                if !built {
                    a.void = Some(format!("its {:?} module did not build", a.form));
                    self.lines.push(format!("4E tree: arm {} VOID: its {:?} module did not build", a.name, a.form));
                }
            }
        }
        self.tree = Some(Tree4e {
            _cuda: cuda, f, smem, area, arms: arm_v, tiled, part_stride, ldx, n_ot, vocab, toks, depth, path, mv, ring, logits,
            _held16: held, held16_addr, part, fpart, tickets, ftickets, slots,
        });
        Ok(())
    }

    /// The tree-tiled arms: the tiled module (`pers::p4t_tiled_unit_text`)
    /// when `want_tiled`, the stream module (`pers::p4t_stream_unit_text`)
    /// when `want_stream`, each symbol at the same shared bytes, and the
    /// tree-tiled copy of every projection the pass reads (q, k, v, o, gate,
    /// up, down, the head), built on the card by `p4t_tile_words` from the
    /// words 4E holds, once for both. A Tetra o is refused (no sealed file the
    /// job reads carries one).
    fn install_tiled(&mut self, smem: u32, threads: u32, want: [bool; 7]) -> anyhow::Result<TreeTiled> {
        if self.layers_host.iter().any(|y| y.ot != 0) {
            anyhow::bail!("4E tree: the tiled and stream arms do not copy a Tetra o_proj; this model carries one");
        }
        let stream = self.cuda.stream().clone();
        let mut cudas = Vec::new();
        let mut fns: [Option<CudaFunction>; 7] = [None, None, None, None, None, None, None];
        for (k, want) in want.into_iter().enumerate() {
            if !want {
                continue;
            }
            let (tag, text, sym, def) = match k {
                0 => ("tiled", pers::p4t_tiled_unit_text(&self.tile_define), pers::p4t_tiled_symbol(self.form), "P4T_TILED 1"),
                1 => ("stream", pers::p4t_stream_unit_text(&self.tile_define), pers::p4t_stream_symbol(self.form), "P4T_STREAM 1 and P4T_TILED 1"),
                2 => ("a4", pers::p4t_a4_unit_text(&self.tile_define, true, true), pers::p4t_a4_symbol(self.form), "P4T_A4 1 (F1, F2), P4T_STREAM 1 and P4T_TILED 1"),
                3 => ("a5", pers::p4t_a5_unit_text(&self.tile_define), pers::p4t_a5_symbol(self.form), "P4T_A5 1, P4T_A4 1 (F1, F2), P4T_STREAM 1 and P4T_TILED 1"),
                _ => {
                    let (arm, def) = match k {
                        4 => ("a6", "P4T_A6 1 after a3's text"),
                        5 => ("a7", "P4T_A7 1 after a5's text"),
                        _ => ("a3f", "P4T_A3F 1 after a3's text, a floor arm"),
                    };
                    let text = pers::p4t_a6_unit_text(arm, &self.tile_define).map_err(anyhow::Error::msg)?;
                    (arm, text, pers::p4t_a6_symbol(arm, self.form), def)
                }
            };
            let sym = sym.map_err(anyhow::Error::msg)?;
            let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
            self.lines.push(format!("4E tree {tag}: module {} bytes, sha256 {}, {def}, its own module", src.text.len(), src.sha256));
            // a module that fails to build or to fit voids its arm only
            let built = (|| -> anyhow::Result<(llvq_cuda::gpu::Cuda, CudaFunction)> {
                let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;
                let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
                let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
                let occ = f
                    .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                    .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
                self.lines.push(format!("4E tree {tag}: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes));
                if occ < 1 {
                    anyhow::bail!("{sym} fits no CTA on a SM at {threads} threads and {smem} B");
                }
                Ok((cuda, f))
            })();
            match built {
                Ok((cuda, f)) => {
                    fns[k] = Some(f);
                    cudas.push(cuda);
                }
                Err(e) => self.lines.push(format!("4E tree {tag}: module FAILED, its arms are void: {e}")),
            }
        }
        if cudas.is_empty() {
            anyhow::bail!("4E tree: no tree-tiled module built; see the lines above");
        }
        let tile_f = cudas[0].func("p4t_tile_words").map_err(anyhow::Error::msg)?;
        let mut held: Vec<CudaSlice<u32>> = Vec::new();
        let mut bytes = 0u64;
        let (swaps, head) = {
            let mut copy = |src: u64, int4: bool, d_out: u32, stride: u32, d_in: u32, served_tiled: bool| -> anyhow::Result<u64> {
                if !d_out.is_multiple_of(16) || src == 0 {
                    anyhow::bail!("4E tree tiled: a projection of {d_out} rows at {src:#x} cannot be tiled by 16 rows");
                }
                let n = pers::p4t_tiled_words(int4, d_out, d_in);
                let buf = cudas[0].zeros_u32(n as usize).map_err(anyhow::Error::msg)?;
                let (kind, nsub, served) = (u32::from(int4), d_in.div_ceil(48), u32::from(served_tiled));
                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = stream.launch_builder(&tile_f);
                b.arg(&src).arg(&buf).arg(&kind).arg(&stride).arg(&nsub).arg(&served).arg(&n);
                // SAFETY: src is a live record of d_out rows of `stride` words, buf a
                // fresh buffer of n words; the kernel reads src and writes buf only.
                unsafe { b.launch(cfg) }.map_err(|e| anyhow::anyhow!("p4t_tile_words: {e}"))?;
                let p = buf.device_ptr(&stream).0;
                bytes += 4 * n;
                held.push(buf);
                Ok(p)
            };
            let (hidden, d_ctx, inter) = (self.args_host.hidden, self.args_host.n_heads * 128, self.inter);
            let tet = |c: &mut dyn FnMut(u64, bool, u32, u32, u32, bool) -> anyhow::Result<u64>, t: &pers::SegTetraHost| {
                c(t.words, false, t.d_out, t.stride_u32, t.nblocks * 24 + t.tail_w, false)
            };
            let mut swaps = Vec::with_capacity(self.layers_host.len());
            for y in &self.layers_host {
                let q = &y.qkv;
                let down = if y.tetra_down { tet(&mut copy, &y.mlp.down)? } else { copy(y.mlp.q4w, true, hidden, inter / 8, inter, true)? };
                swaps.push([
                    tet(&mut copy, &q.q)?,
                    tet(&mut copy, &q.k)?,
                    copy(q.vw, true, q.v_dout, q.d_in / 8, q.d_in, true)?,
                    copy(y.ow, true, hidden, d_ctx / 8, d_ctx, true)?,
                    tet(&mut copy, &y.mlp.gate)?,
                    tet(&mut copy, &y.mlp.up)?,
                    down,
                ]);
            }
            let head = copy(self.args_host.head[0], true, self.args_host.vocab, hidden / 8, hidden, false)?;
            (swaps, head)
        };
        cudas[0].sync().map_err(anyhow::Error::msg)?;
        self.lines.push(format!(
            "4E tree tiled: {} tree-tiled copies (q, k, v, o, gate, up, down a layer, the head), {:.3} GB more on the card",
            held.len(),
            bytes as f64 / 1e9
        ));
        let layers = stream.clone_htod(&vec![0u8; pers::P4E_LAYER_BYTES * self.layers_host.len()]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let args = stream.clone_htod(&vec![0u8; pers::P4E_ARGS_BYTES]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let [f, fs, f4, f5, f6, f7, f3f] = fns;
        Ok(TreeTiled { _cudas: cudas, f, fs, f4, f5, fx: [f6, f7, f3f], _held: held, swaps, head, layers, args, seen: Vec::new() })
    }

    /// The arms installed, in the order [`Self::tree`] indexes them.
    pub fn tree_arms(&self) -> Vec<TreeArm> {
        self.tree.as_ref().map_or_else(Vec::new, |t| t.arms.clone())
    }

    /// The tiled images brought to the caches the 4E images carry.
    fn sync_tiled(&mut self) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        let Some(t) = self.tree.as_mut().and_then(|v| v.tiled.as_mut()) else {
            anyhow::bail!("4E tree: no tiled arm installed");
        };
        if t.seen == self.kv_seen {
            return Ok(());
        }
        let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * self.layers_host.len());
        for (h, w) in self.layers_host.iter().zip(&t.swaps) {
            let mut y = h.clone();
            y.qkv.q.words = w[0];
            y.qkv.k.words = w[1];
            y.qkv.vw = w[2];
            y.ow = w[3];
            y.mlp.gate.words = w[4];
            y.mlp.up.words = w[5];
            if y.tetra_down {
                y.mlp.down.words = w[6];
            } else {
                y.mlp.q4w = w[6];
            }
            img.extend(y.bytes());
        }
        stream.memcpy_htod(&img, &mut t.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let mut a = self.args_host.clone();
        a.ptrs[0] = t.layers.device_ptr(&stream).0;
        a.head[0] = t.head;
        stream.memcpy_htod(&a.bytes(), &mut t.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        t.seen = self.kv_seen.clone();
        Ok(())
    }

    /// One tree launch: row r the token `toks[r]` at position `pos0 +
    /// depth[r]`, `path[r][d]` its ancestor's row at depth d and
    /// `path[r][depth[r]] = r`, the caches holding `pos0` positions, which the
    /// launch reads and does not write (no move: the accepted rows stay in the
    /// tree buffer). Each row's argmax, its logits when `logits`, the phase
    /// clocks when `stamped` (`pers::p4t_phases`).
    #[allow(clippy::too_many_arguments)]
    pub fn tree(&mut self, arm: usize, caches: &mut [KvCache], pos0: usize, toks: &[u32], depth: &[u32], path: &[[u8; 32]], logits: bool, stamped: bool) -> anyhow::Result<TreeOut> {
        let n = toks.len();
        if n == 0 || n > pers::P4T_RMAX as usize || depth.len() != n || path.len() != n {
            anyhow::bail!("4E tree: {n} rows, {} depths, {} paths; the pass carries 1 to {}", depth.len(), path.len(), pers::P4T_RMAX);
        }
        for r in 0..n {
            let d = depth[r] as usize;
            let ok = d < 32
                && path[r][d] as usize == r
                && (0..=d).all(|e| (path[r][e] as usize) < n && depth[path[r][e] as usize] as usize == e && path[path[r][e] as usize][..=e] == path[r][..=e]);
            if !ok {
                anyhow::bail!("4E tree: row {r}'s path {:?} at depth {d} is not a path of this tree", &path[r][..=d.min(31)]);
            }
        }
        if self.tree.is_none() {
            anyhow::bail!("4E tree: not installed (Runtime4e::install_tree)");
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!("4E tree: a launch at position {pos0}, below the kernel threshold {}", self.kernel_from);
        }
        let span = 1 + *depth.iter().max().expect("n > 0") as usize;
        if pos0 + span > MAX_CAP {
            anyhow::bail!("4E tree: positions up to {}, the scratch holds {MAX_CAP}", pos0 + span);
        }
        let spec = match self.tree.as_ref().expect("checked above").arms.get(arm) {
            Some(a) => a.clone(),
            None => anyhow::bail!("4E tree: arm {arm} of {} installed", self.tree.as_ref().map_or(0, |t| t.arms.len())),
        };
        self.prepare(caches, pos0, span)?;
        if spec.form != TreeForm::Rows {
            self.sync_tiled()?;
        }
        let stream = self.cuda.stream().clone();
        let v = self.tree.as_mut().expect("checked above");
        stream.memcpy_htod(toks, &mut v.toks).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(depth, &mut v.depth).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let flat: Vec<u8> = path.iter().flat_map(|p| p.iter().copied()).collect();
        stream.memcpy_htod(&flat, &mut v.path).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(&vec![u32::MAX; pers::P4T_RMAX as usize], &mut v.ring).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let n_bar = pers::n_bar_4et(self.n_layers, v.n_ot);
        let stamps: Option<CudaSlice<u64>> = if stamped {
            // an a4 launch adds its F3 slots after the phase events: 20 a CTA
            let extra = if matches!(spec.form, TreeForm::A4 | TreeForm::A5) { 20 } else { 0 };
            Some(stream.alloc_zeros::<u64>((1 + 2 * n_bar + extra) as usize * self.grid as usize).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        let a = v.held16_addr;
        let launch = pers::P4tLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_rows: n as u32,
            area_floats: v.area,
            n_move: 0,
            move_pos0: 0,
            part_stride: v.part_stride,
            ldx: v.ldx,
            split_qkv: spec.splits[0],
            split_o: spec.splits[1],
            split_gu: spec.splits[2],
            split_dn: spec.splits[3],
            split_head: spec.splits[4],
            pad: 0,
            toks: v.toks.device_ptr(&stream).0,
            depth: v.depth.device_ptr(&stream).0,
            path: v.path.device_ptr(&stream).0,
            mv: v.mv.device_ptr(&stream).0,
            ring: v.ring.device_ptr(&stream).0,
            logits: if logits { v.logits.device_ptr(&stream).0 } else { 0 },
            x_a: a[0],
            x_b: a[1],
            xin: a[2],
            xn: a[3],
            q_raw: a[4],
            k_raw: a[5],
            v_raw: a[6],
            q_out: a[7],
            ctx: a[8],
            g_raw: a[9],
            u_raw: a[10],
            tk: a[11],
            tv: a[12],
            part: v.part.device_ptr(&stream).0,
            fpart: v.fpart.device_ptr(&stream).0,
            tickets: v.tickets.device_ptr(&stream).0,
            ftickets: v.ftickets.device_ptr(&stream).0,
            slots: v.slots.device_ptr(&stream).0,
            stamps: stamps.as_ref().map_or(0, |b| b.device_ptr(&stream).0),
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: v.smem };
        let (func, args) = match (spec.form, v.tiled.as_ref()) {
            (TreeForm::Rows, _) => (&v.f, &self.args),
            (TreeForm::Tiled, Some(TreeTiled { f: Some(f), args, .. })) => (f, args),
            (TreeForm::Stream, Some(TreeTiled { fs: Some(f), args, .. })) => (f, args),
            (TreeForm::A4, Some(TreeTiled { f4: Some(f), args, .. })) => (f, args),
            (TreeForm::A5, Some(TreeTiled { f5: Some(f), args, .. })) => (f, args),
            (TreeForm::A6, Some(TreeTiled { fx: [Some(f), _, _], args, .. })) => (f, args),
            (TreeForm::A7, Some(TreeTiled { fx: [_, Some(f), _], args, .. })) => (f, args),
            (TreeForm::A3f, Some(TreeTiled { fx: [_, _, Some(f)], args, .. })) => (f, args),
            (form, _) => anyhow::bail!("4E tree: arm {} is {form:?} and its module is not installed", spec.name),
        };
        let mut b = stream.launch_builder(func);
        b.arg(args).arg(&launch);
        // SAFETY: the kernel's own signature (Pers4eArgs*, Pers4tLaunch), the
        // layout checked on the host (tree_verify_host.rs); one CTA a SM, past
        // the occupancy check at install.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E tree: {e}"))?;
        let ring = stream.clone_dtoh(&v.ring).map_err(|e| anyhow::anyhow!("D2H: {e}"))?;
        let lg = if logits {
            let all = stream.clone_dtoh(&v.logits).map_err(|e| anyhow::anyhow!("D2H logits: {e}"))?;
            all[..n * v.vocab as usize].to_vec()
        } else {
            Vec::new()
        };
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E tree: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E tree: a NaN logit"),
            None => {}
        }
        let made: Vec<u32> = ring[..n].to_vec();
        if made.contains(&u32::MAX) {
            let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
            pers::p4e_reset(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)?;
            anyhow::bail!("4E tree: a row wrote no token ({made:?}); the counters are reset for the next launch");
        }
        self.bar_base = self.bar_base.wrapping_add(n_bar.wrapping_mul(self.grid));
        let stamps = match stamps {
            Some(d) => stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?,
            None => Vec::new(),
        };
        Ok(TreeOut { ring: made, logits: lg, stamps })
    }

    /// The caches grown to hold a launch of `n` positions from `pos0`, and
    /// the images re-uploaded when a buffer moved.
    fn prepare(&mut self, caches: &mut [KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        // Grow every cache to hold the launch; the images carry the
        // addresses, re-uploaded only when a buffer moved.
        let mut seen = Vec::with_capacity(caches.len());
        for c in caches.iter_mut() {
            if c.len() != pos0 {
                anyhow::bail!("4E: a cache holds {} positions, the launch starts at {pos0}", c.len());
            }
            c.reserve(pos0 + n)?;
            let (k, v) = c.grow_buffers().ok_or_else(|| anyhow::anyhow!("4E: a cache with no Grow buffer"))?;
            let cap = k.dim(2)?;
            if cap > MAX_CAP {
                anyhow::bail!("4E: a cache of {cap} positions, the attention scratch holds {MAX_CAP}");
            }
            seen.push((cap, tensor_addr(k, &stream)?, tensor_addr(v, &stream)?));
        }
        if seen != self.kv_seen {
            let caps: std::collections::BTreeSet<usize> = seen.iter().map(|s| s.0).collect();
            if caps.len() != 1 {
                anyhow::bail!("4E: the layers' caches have different capacities {caps:?}");
            }
            let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * seen.len());
            for (h, &(_, k, v)) in self.layers_host.iter_mut().zip(&seen) {
                h.kc = k;
                h.vc = v;
                img.extend(h.bytes());
            }
            stream.memcpy_htod(&img, &mut self.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.args_host.ptrs[0] = self.layers.device_ptr(&stream).0;
            self.args_host.cap = seen[0].0 as u32;
            stream.memcpy_htod(&self.args_host.bytes(), &mut self.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.kv_seen = seen;
        }
        Ok(())
    }

    /// Synchronise the stream (timing).
    pub fn sync(&self) -> anyhow::Result<()> {
        self.cuda.sync().map_err(anyhow::Error::msg)
    }
}

/// Is a projection the int4 one 4E reads (for the mode's refusals)?
pub fn is_int4(p: &Proj) -> bool {
    int4_of(p).is_some()
}
