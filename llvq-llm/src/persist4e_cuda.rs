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
//! The served 4B config (`configs/qwen3-4b-tetra-e4-kernel-from1280.json`)
//! sends a decode row to the kernel from position 1,280 only, and runs
//! candle's grouped arithmetic below it (cuBLAS matmuls, candle's softmax),
//! whose bits no hand kernel reproduces. So 4E keeps the served bits, inside
//! the operator's T8, only from position [`crate::kvq::KERNEL_FROM`]: the
//! runtime refuses a launch that starts below it. A measurement may lower the
//! threshold to 0 ([`Runtime4e::set_kernel_from_for_measurement`]) against an
//! eager path with every row on the kernel, the form the first readings ran:
//! that form has no T8 and is never served.
//!
//! What only the card can decide is RoPE: candle's `rope` is `__half`
//! arithmetic the driver's ptxas may fuse, so [`install`] runs candle's rope
//! on the card and keeps the form (pers_rope_pair) that matches it, or
//! refuses.
//!
//! This version reads the 4B and 14B shapes: q and k Tetra under one rotation,
//! v and o int4, a Tetra or int4 down per layer, a tied int4 g64 table.
//! Anything else is refused at install by name.

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::{DType, Device, Tensor};
use half::f16;

use crate::model::{Embed, Head, KvCache, Proj, Qwen3};
use crate::persist_cuda::{int4_of, mlp_args, seg_tetra, tensor_addr, tetra_of};
use llvq_cuda::pers;

/// The positions 4E's attention scratch is sized for, at most.
pub const MAX_CAP: usize = 8192 + 512;

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
    pub form: u32,
    layers_host: Vec<pers::P4eLayerHost>,
    args_host: pers::P4eArgsHost,
    layers: CudaSlice<u8>,
    args: CudaSlice<u8>,
    /// Held, read by the kernel only: its address is in the images.
    #[allow(dead_code)]
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
    /// The (cap, K, V addresses) the uploaded images carry.
    kv_seen: Vec<(usize, u64, u64)>,
    /// The first position a launch may start at: the served threshold, or 0
    /// for a measurement against the kernel on every row.
    kernel_from: usize,
    pub lines: Vec<String>,
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

/// [`install`] with the module's row form (`pers::P4E_ROWS_FORMS`): 1 is v3,
/// what the served config runs; 2 and 3 are measured by `LLVQ_PERS_4E`.
pub fn install_rows(model: &Qwen3, device: &Device, max_tokens: u32, threads: Option<u32>, rows_form: u32) -> anyhow::Result<Runtime4e> {
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
    let smem = pers::p4e_smem_bytes(hidden, inter, groups as u32) + pers::p4e_scratch_bytes(rows_form);
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
    let (f, threads) = chosen.ok_or_else(|| anyhow::anyhow!("4E: every block size asked for spills or fails its occupancy"))?;
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
    let tied = match head {
        Head::Q8(t) => t.as_any().and_then(|a| a.downcast_ref::<crate::fused_cuda::CudaEmbed>()).and_then(|c| c.q4_view()),
        Head::Dense(_) => None,
    };
    if tied.map(|h| h.words) != Some(emb.words) || emb.d != hidden || emb.vocab % 8 != 0 {
        anyhow::bail!("4E: the head is not the embedding's own int4 g64 table (the tied 4B and 14B)");
    }
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
        let oq = int4_of(op).ok_or_else(|| ctxt("o_proj is not an int4 record (an 8B Tetra o is not in this version)"))?.int4_view();
        if vq.d_in != hidden || vq.d_out != (n_kv * 128) as u32 || oq.d_in != (n_heads * 128) as u32 || oq.d_out != hidden {
            return Err(ctxt("v or o has an unexpected shape"));
        }
        if qv.d_out != (n_heads * 128) as u32 || kv.d_out != (n_kv * 128) as u32 || qv.d_in != hidden || kv.d_in != hidden {
            return Err(ctxt("q or k has an unexpected shape"));
        }
        let (mlp, tetra_down) = mlp_args(block, &frt, &base, &stream)?;
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
            ow: oq.wq,
            osc: oq.scales,
            obi: oq.biases,
            o_gpr: oq.gpr,
            tetra_down,
            mlp,
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
    };
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
        _held32: vec![d_gu, d_dn, ticket],
        _heldf: vec![scratch],
        _held64: vec![slots],
        kv_seen: Vec::new(),
        kernel_from: crate::kvq::KERNEL_FROM,
        lines,
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
        format!("{form} at {} threads", self.threads)
    }

    /// The first position a launch may start at.
    pub fn kernel_from(&self) -> usize {
        self.kernel_from
    }

    /// Lower (or restore) the threshold, for a measurement only: below
    /// [`crate::kvq::KERNEL_FROM`] 4E's tokens are the kernel form's, which
    /// the served config does not run there and no T8 covers. The eager side
    /// of such a measurement must run the same form
    /// (`Qwen3::set_kernel_from_for_measurement`).
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
        let stream = self.cuda.stream().clone();
        // Grow every cache to hold the launch; the images carry the
        // addresses, re-uploaded only when a buffer moved.
        let mut seen = Vec::with_capacity(caches.len());
        for c in caches.iter_mut() {
            if c.len() != pos0 {
                anyhow::bail!("4E: a cache holds {} positions, the launch starts at {pos0}", c.len());
            }
            c.reserve(pos0 + n as usize)?;
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
        if self.cuda.down_u32(&self.abort).map_err(anyhow::Error::msg)?[0] != 0 {
            anyhow::bail!("4E: a barrier aborted on its watchdog; the counters are poisoned");
        }
        if self.cuda.down_u32(&self.nan).map_err(anyhow::Error::msg)?[0] != 0 {
            anyhow::bail!("4E: a NaN logit; the eager path is the one to read");
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

    /// Synchronise the stream (timing).
    pub fn sync(&self) -> anyhow::Result<()> {
        self.cuda.sync().map_err(anyhow::Error::msg)
    }
}

/// Is a projection the int4 one 4E reads (for the mode's refusals)?
pub fn is_int4(p: &Proj) -> bool {
    int4_of(p).is_some()
}
