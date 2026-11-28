//! Installing 4E: the RoPE probe against candle on the card, the public
//! `install*` doors, the e1 arm checks, and `install_inner`, which builds the
//! module, the layer images and the scratch. One file because every door ends
//! in `install_inner`.

use super::*;

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
/// 2 is v4; 3 is v5; 4 is v6; 5 is v7; 6 is v8; 7 is v9, what the served
/// configs run. `LLVQ_PERS_4E` times `pers::P4E_BENCH_FORMS`.
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
    install_inner(model, device, max_tokens, threads, rows_form, smem_of, None)
}

/// An E1 arm (`pers::P4E_E1_ARMS`: "v9", "v9np", "v9mv", ..., "v9k8"), a measurement:
/// row form 7's module with the arm's edits (`pers::p4e_e1_unit_text`; the
/// served v9 module's text does not move), at 512 threads, with its logits
/// symbol. Experiment E1 of `docs/mesures/large-4e-budget-2026-11-11.txt`.
/// The RoPE form 4E has run on the L40S in every reading (the card's load
/// check picked form 1 in large-4e-e1, large-attn-ft-4e and the jeton run):
/// the load check's verdict reads this form's token symbol, and prints the
/// two others beside it.
pub const CHECK_ROPE_FORM: u32 = 1;

/// The load check of E1 arms (`fusedrun LLVQ_PERS_4E=check:<arms>`): each
/// arm's module compiled as [`install_e1`] compiles it (the same text and
/// sha256, NVRTC for the same arch, the driver's JIT), with no model loaded
/// and nothing launched. The arms run one after the other in the order named,
/// and every line is printed as it is known and returned. For each arm, the token symbol
/// of each RoPE form at 512 threads, its stamp and logits symbols at
/// [`CHECK_ROPE_FORM`]: registers, local bytes, dynamic shared bytes, CTAs a
/// SM, at the sealed 4B's shape (hidden 2,560, inter 9,728, 4 groups, 85,952
/// B). An rc arm's layout is checked at the card's grid (`p4e_rc_layout`).
/// The verdict is [`install_e1`]'s rule on the token symbol of
/// [`CHECK_ROPE_FORM`]: no local memory and one CTA a SM, or refused. Lines
/// `CHECK ARM <arm> LOADS` or `CHECK ARM <arm> REFUSED: <why>` carry it.
pub fn check_e1_arms(device: &Device, arms: &[&'static str]) -> anyhow::Result<Vec<String>> {
    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let cc = llvq_cuda::gpu::compute_cap_of(stream.context()).map_err(anyhow::Error::msg)?;
    let tile = llvq_cuda::tile::resolve(cc);
    let td = tile.define();
    let (hidden, inter, groups, d_ctx) = (2560u32, 9728u32, 4u32, 4096u32);
    let smem = pers::p4e_smem_bytes_form(7, hidden, inter, groups) + pers::p4e_scratch_bytes(7);
    let mut lines = vec![format!(
        "4E check: {} arms, sm_{}{}, {} define, the sealed 4B's shape: {smem} B of dynamic shared memory",
        arms.len(),
        cc.0,
        cc.1,
        td.trim()
    )];
    let srcs: Vec<llvq_cuda::gpu::KernelSource> = arms
        .iter()
        .map(|a| pers::p4e_e1_unit_text(&td, a).map(|t| llvq_cuda::gpu::KernelSource::new(&[t.as_str()])))
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::msg)?;
    // One arm after the other, in the order named, every line printed and
    // flushed as it is known: a check killed by its timeout leaves what it
    // had read. The first card run (6af73aed) compiled seven arms on seven
    // threads, printed at the end, and left nothing.
    let say = |lines: &mut Vec<String>, l: String| {
        use std::io::Write;
        println!("{l}");
        let _ = std::io::stdout().flush();
        lines.push(l);
    };
    let head = lines.clone();
    lines.clear();
    for l in head {
        say(&mut lines, l);
    }
    let t0 = std::time::Instant::now();
    for (arm, src) in arms.iter().zip(&srcs) {
        say(&mut lines, format!("4E check {arm}: module {} bytes, sha256 {}", src.text.len(), src.sha256));
        let t = std::time::Instant::now();
        let mut l = Vec::new();
        let verdict = llvq_cuda::gpu::compile_ptx(src).and_then(|p| {
            say(&mut lines, format!("4E check {arm}: NVRTC {:.1} s", t.elapsed().as_secs_f64()));
            check_one_arm(&stream, p, arm, smem, (hidden, inter, groups, d_ctx), &mut l)
        });
        for x in l {
            say(&mut lines, x);
        }
        say(
            &mut lines,
            match verdict {
                Ok(()) => format!("CHECK ARM {arm} LOADS ({:.1} s)", t.elapsed().as_secs_f64()),
                Err(e) => format!("CHECK ARM {arm} REFUSED: {e} ({:.1} s)", t.elapsed().as_secs_f64()),
            },
        );
    }
    say(&mut lines, format!("4E check: {} arms one after the other, {:.1} s", arms.len(), t0.elapsed().as_secs_f64()));
    Ok(lines)
}

/// [`check_e1_arms`] for one arm, from its PTX: the JIT, the rc layout, the
/// symbols' attributes; `Err` names why the token symbol would be refused.
fn check_one_arm(
    stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ptx: candle_core::cuda_backend::cudarc::nvrtc::Ptx,
    arm: &str,
    smem: u32,
    (hidden, inter, groups, d_ctx): (u32, u32, u32, u32),
    lines: &mut Vec<String>,
) -> Result<(), String> {
    // this thread's current context: the function attributes below read it
    stream.context().bind_to_thread().map_err(|e| format!("bind the context: {e}"))?;
    let cuda = llvq_cuda::gpu::Cuda::on_stream_ptx(stream.clone(), ptx)?;
    let grid = cuda.device()?.sm_count as u32;
    if let Some(body) = pers::p4e_rc_body(arm) {
        let ph = pers::p4e_rc_layout(body, hidden, inter, d_ctx, false, true, groups, grid)?;
        let fmt: Vec<String> = ph.iter().map(|p| format!("{} {} B", p.phase, 4 * p.end)).collect();
        lines.push(format!("CHECK {arm} rc layout on {grid} CTAs: {} of {} B", fmt.join(", "), 4 * ph[0].limit));
    }
    let mut syms: Vec<(&str, bool)> =
        (0..3).map(|f| pers::p4e_symbol(f, 512).map(|s| (s, f == CHECK_ROPE_FORM))).collect::<Result<_, _>>()?;
    syms.push((pers::p4e_stamp_symbol(CHECK_ROPE_FORM, 512)?, false));
    syms.push((pers::p4e_logits_symbol(CHECK_ROPE_FORM)?, false));
    let mut decides = Err(format!("no token symbol of RoPE form {CHECK_ROPE_FORM}"));
    for (sym, decisive) in syms {
        let f = cuda.func_dynamic_shared(sym, smem)?;
        let r = cuda.report(sym)?;
        let occ = f.occupancy_max_active_blocks_per_multiprocessor(512, smem as usize, None).map_err(|e| format!("occupancy: {e}"))?;
        let ok = pers::p4e_token_symbol_loads(r.local_bytes, occ, false);
        let tail = match (sym.contains("_tokens_"), ok) {
            (true, true) => ": loads",
            (true, false) => ": refused",
            (false, _) => " (outside the timed rounds)",
        };
        lines.push(format!("CHECK {arm} {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM{tail}", r.num_regs, r.local_bytes));
        if decisive {
            decides = if ok { Ok(()) } else { Err(format!("{sym}: {} B local, {occ} resident a SM", r.local_bytes)) };
        }
    }
    // v9k8's fill, a plain launch outside the timed rounds: printed, never decisive
    if pers::p4e_e1_k8(arm) {
        let sym = pers::P4E_K8_FILL_SYMBOL;
        cuda.func(sym)?;
        let r = cuda.report(sym)?;
        lines.push(format!("CHECK {arm} {sym}: {} registers, {} B local (the fill, outside the timed rounds)", r.num_regs, r.local_bytes));
    }
    decides
}

pub fn install_e1(model: &Qwen3, device: &Device, max_tokens: u32, arm: &str) -> anyhow::Result<Runtime4e> {
    let arm = *pers::P4E_E1_ARMS
        .iter()
        .find(|a| **a == arm)
        .ok_or_else(|| anyhow::anyhow!("4E E1: no arm {arm:?}; the arms are {:?}", pers::P4E_E1_ARMS))?;
    install_inner(model, device, max_tokens, Some(512), 7, None, Some((arm, false)))
}

/// [`install_e1`] with the local-memory half of the load rule waived
/// (`pers::p4e_token_symbol_loads`, `spill`): MEASUREMENT ONLY, for an arm
/// `LLVQ_PERS_4E` names `<arm>+local`. The runtime carries the bytes
/// ([`Runtime4e::local_bytes`]) and its label prints them.
pub fn install_e1_local(model: &Qwen3, device: &Device, max_tokens: u32, arm: &str) -> anyhow::Result<Runtime4e> {
    let arm = *pers::P4E_E1_ARMS
        .iter()
        .find(|a| **a == arm)
        .ok_or_else(|| anyhow::anyhow!("4E E1: no arm {arm:?}; the arms are {:?}", pers::P4E_E1_ARMS))?;
    install_inner(model, device, max_tokens, Some(512), 7, None, Some((arm, true)))
}

fn install_inner(
    model: &Qwen3,
    device: &Device,
    max_tokens: u32,
    threads: Option<u32>,
    rows_form: u32,
    smem_of: Option<u32>,
    e1: Option<(&'static str, bool)>,
) -> anyhow::Result<Runtime4e> {
    // an E1 arm, and whether the local-memory half of the load rule is waived for it
    let (e1_arm, spill) = (e1.map(|e| e.0), e1.is_some_and(|e| e.1));
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

    let text = match e1_arm {
        None => format!("{}{}", pers::p4e_rows_define(rows_form).map_err(anyhow::Error::msg)?, pers::pers_unit_text(&frt.tile_define())),
        Some(arm) => pers::p4e_e1_unit_text(&frt.tile_define(), arm).map_err(anyhow::Error::msg)?,
    };
    let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
    lines.push(format!(
        "4E: persistent module {} bytes, sha256 {}, row form {rows_form}{}, own module on candle's stream",
        src.text.len(),
        src.sha256,
        e1_arm.map_or(String::new(), |a| format!(", E1 arm {a}"))
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
        if pers::p4e_token_symbol_loads(r.local_bytes, occ, spill) {
            if r.local_bytes != 0 {
                lines.push(format!(
                    "4E: MEASUREMENT ONLY: {sym} holds {} B of local memory and loads because its arm was named {}; \
                     the load rule refuses it on every other path",
                    r.local_bytes,
                    pers::P4E_LOCAL_SUFFIX
                ));
            }
            chosen = Some((f, threads, r.local_bytes));
            break;
        }
    }
    // The refusal names what each block size read (step2-4e did not print it).
    let (f, threads, local_bytes) = chosen.ok_or_else(|| anyhow::anyhow!("4E: every block size asked for spills or fails its occupancy: {}", lines.join("; ")))?;
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
    // An E1 module's logits symbol, at 512 threads only. Like the stamp
    // symbol it runs outside the timed rounds and may hold local memory.
    let f_logits = match e1_arm {
        None => None,
        Some(arm) => {
            if threads != 512 {
                anyhow::bail!("4E E1 {arm}: the logits symbol is at 512 threads, the runtime at {threads}");
            }
            let sym = pers::p4e_logits_symbol(form).map_err(anyhow::Error::msg)?;
            let fl = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
            let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
            let occ = fl
                .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
            lines.push(format!("4E: {sym}: {} registers, {} B local, {occ} resident a SM (logits, outside the timed rounds)", r.num_regs, r.local_bytes));
            (occ >= 1).then_some(fl)
        }
    };
    let grid = cuda.device().map_err(anyhow::Error::msg)?.sm_count as u32;
    // Arm v9k8: the fill symbol (outside the timed rounds, a launch of its own
    // before a launch that takes over the prefill's cache).
    let k8 = match e1_arm {
        Some(arm) if pers::p4e_e1_k8(arm) => {
            let fill = cuda.func(pers::P4E_K8_FILL_SYMBOL).map_err(anyhow::Error::msg)?;
            let r = cuda.report(pers::P4E_K8_FILL_SYMBOL).map_err(anyhow::Error::msg)?;
            let per_pos = 2 * llvq_cuda::kv8_ref::POS_BYTES * n_kv * model.blocks.len();
            lines.push(format!(
                "4E v9k8: int8 KV caches, {} B a position and head, {per_pos} B a position ({:.4} of binary16's {}), \
                 {:.1} MB at 8,192 positions; the fill {}: {} registers, {} B local",
                llvq_cuda::kv8_ref::POS_BYTES,
                per_pos as f64 / (512 * n_kv * model.blocks.len()) as f64,
                512 * n_kv * model.blocks.len(),
                (per_pos * 8192) as f64 / 1e6,
                pers::P4E_K8_FILL_SYMBOL,
                r.num_regs,
                r.local_bytes
            ));
            Some(Kv8 { fill, n_kv: n_kv as u32, bufs: Vec::new(), state: pers::K8State::default(), f16_seen: Vec::new(), release: false })
        }
        _ => None,
    };

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
    // The research row bodies (E1 arms v9ep and v9fl, kernels/pers_4e_rc.cuh):
    // their regions in 4E's buffer at this shape, refused by name where one
    // would pass the v7 area (the 14B); and how many Tetra matrices flat reads
    // by u32 loads because their words are not on 16 bytes (none expected).
    if let Some(body) = e1_arm.and_then(pers::p4e_rc_body) {
        let tetra_down = layers_host.iter().any(|y| y.tetra_down);
        let ph = pers::p4e_rc_layout(body, hidden, inter, (n_heads * 128) as u32, tetra_o > 0, tetra_down, groups as u32, grid)
            .map_err(anyhow::Error::msg)?;
        let fmt: Vec<String> = ph.iter().map(|p| format!("{} {} B", p.phase, 4 * p.end)).collect();
        let words: Vec<u64> = layers_host
            .iter()
            .flat_map(|y| [y.qkv.q.words, y.qkv.k.words, y.mlp.gate.words, y.mlp.up.words].into_iter().chain(y.tetra_down.then_some(y.mlp.down.words)))
            .collect();
        let off = words.iter().filter(|&&w| w % 16 != 0).count();
        lines.push(format!(
            "4E rc: body {body} ({}), row vector used to {} of the v7 area at {} B; {off} of {} Tetra matrices off 16 bytes",
            ["", "endpass", "flat", "ep2", "ep3", "fl2", "fl3"][body as usize],
            fmt.join(", "),
            4 * ph[0].limit,
            words.len()
        ));
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
        f_logits,
        e1_arm,
        local_bytes,
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
        k8,
        tree_modules: TreeModules::Ptx,
    })
}
