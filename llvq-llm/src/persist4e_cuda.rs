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
use candle_core::{DType, Device, Storage, Tensor};
use half::f16;

use crate::fused_cuda::{CudaInt4, CudaLattice, TetraView};
use crate::model::{Embed, Head, KvCache, Proj, Qwen3};
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
/// 2 is v4, 4 is v6 and 7 is v9, which the served configs run since
/// 2026-11-07. `LLVQ_PERS_4E` times `pers::P4E_BENCH_FORMS`.
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
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (v.threads, 1, 1), shared_mem_bytes: v.smem };
        let mut b = stream.launch_builder(&v.f);
        b.arg(&self.args).arg(&launch);
        // SAFETY: the kernel's own signature (Pers4eArgs*, Pers4vLaunch), the
        // layouts checked on the host; one CTA a SM, past the occupancy check.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E verify: {e}"))?;
        let ring = stream.clone_dtoh(&v.ring).map_err(|e| anyhow::anyhow!("D2H: {e}"))?;
        if self.cuda.down_u32(&self.abort).map_err(anyhow::Error::msg)?[0] != 0 {
            anyhow::bail!("4E verify: a barrier aborted on its watchdog; the counters are poisoned");
        }
        if self.cuda.down_u32(&self.nan).map_err(anyhow::Error::msg)?[0] != 0 {
            anyhow::bail!("4E verify: a NaN logit; the eager path is the one to read");
        }
        let made: Vec<u32> = ring[..n].to_vec();
        if made.contains(&u32::MAX) {
            anyhow::bail!("4E verify: a row wrote no token ({made:?})");
        }
        self.bar_base = self.bar_base.wrapping_add(pers::n_bar_4e(self.n_layers).wrapping_mul(self.grid));
        for c in caches.iter_mut() {
            c.set_len(pos0 + n)?;
        }
        Ok(made)
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

// ---- the device views 4E reads, moved here from route P's host (2026-11-08) ----

pub(crate) fn tensor_addr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> candle_core::Result<u64> {
    let (st, lay) = t.storage_and_layout();
    match &*st {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<f16>()?.device_ptr(stream).0 + 2 * lay.start_offset() as u64),
        _ => candle_core::bail!("4E: a tensor that is not on the card"),
    }
}

pub(crate) fn tetra_of(p: &Proj) -> Option<(TetraView, &CudaLattice)> {
    let Proj::Lattice(d) = p else { return None };
    let c = d.as_any()?.downcast_ref::<CudaLattice>()?;
    Some((c.tetra_view()?, c))
}

pub(crate) fn int4_of(p: &Proj) -> Option<&CudaInt4> {
    let Proj::Int4(d) = p else { return None };
    d.as_any()?.downcast_ref::<CudaInt4>()
}

pub(crate) fn seg_tetra(v: &TetraView) -> pers::SegTetraHost {
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

/// One layer's `SegArgs`, the constants of `base` (SiLU table, scratch,
/// tables, slices, counters, widths) completed with this block's ln2, its
/// gate/up rotation and Tetra streams, and its Tetra or int4 down. Returns the
/// image, whether the down is Tetra, and an int4 down's word order (`None`
/// for a Tetra down).
pub(crate) fn mlp_args(
    block: &crate::model::Block,
    frt: &crate::fused_cuda::FusedRuntime,
    base: &pers::SegArgsHost,
    stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> anyhow::Result<(pers::SegArgsHost, bool, Option<crate::fused::Int4Layout>)> {
    let (hidden, inter) = (base.hidden, base.inter);
    let mut q4_layout = None;
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
    a.alpha = tensor_addr(ln2_w, stream)?;
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
        q4_layout = Some(v.layout);
        false
    } else {
        anyhow::bail!("down_proj is neither a Tetra48 stream nor an int4 record");
    };
    Ok((a, tetra_down, q4_layout))
}
