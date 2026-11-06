// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, section 5, the first paid
// step L1; prereg-persistent-0.md, process 2), revised the same day after an
// adversarial review. Sketch of a `LLVQ_BOUNDARY_AB` measurement mode for
// `bin/fusedrun`, on the `LLVQ_GQA_AB` template (fusedrun.rs:767-806) and the
// `run_trim_ab` divergence rule (fusedrun.rs:187-243).
//
// What it measures: the in-situ cost of one kernel boundary on the served
// object, by class (wave2/gap-2.md): launch, CTA dispatch, drain and ramp,
// underfill, candle host op. Every arm is bit-identical by construction, so a
// token that differs voids the run.
//
// What the review changed (finding 5): `fusedrun` without `LLVQ_CONFIG` loads
// through `load_with`, which passes `source: None` (fused_cuda.rs:2783), and
// the event trim runs only `if source.is_some()` (fused_cuda.rs:2832). So this
// mode turns cudarc's events off itself, after the single-stream check, and
// prints the line, exactly as LLVQ_GQA_AB does (fusedrun.rs:795-806). It sets
// the grouped form on Grow(256) itself and prints it from the model's getters
// (the line `fusedrun.rs:865` prints), since `GqaArm` names print
// "grow(256)+grouped" and nothing else would print the served form here.
//
// Where the code goes:
//   1. llvq-llm/src/fused_cuda.rs: a process-wide arm flag (the INT4_VIA_F32
//      pattern, fused_cuda.rs:2208-2214) read in FusedOp::cuda_fwd and
//      FusedInt4Op::cuda_fwd, plus a tiny NVRTC module loaded ON candle's
//      stream (Cuda::on_stream, gpu.rs:359). Never new_on_fresh_stream: a
//      second stream makes the disabled event tracking unsafe
//      (fused_cuda.rs:2821-2844), and the mode re-checks after loading it.
//   2. llvq-llm/src/bin/fusedrun.rs: the mode block below, refused beside
//      LLVQ_CONFIG (add it to the list at fusedrun.rs:677-679), beside the
//      other measurement modes, and beside LLVQ_KV_PREALLOC.
//   3. CLAUDE.md environment table: one line naming the mode as a
//      measurement mode, never a served config.

// ---- 1. fused_cuda.rs -------------------------------------------------------

/// Measurement only, 0 everywhere else.
static BOUNDARY_ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryArm {
    Served = 0,
    SplitWave = 1,  // gate_proj as rows [0, 6816) + [6816, 9728): 852 + 364 CTAs, 36 extra boundaries
    SplitHalf = 2,  // gate_proj as 4864 + 4864 rows: 608 + 608 CTAs
    SplitDown = 3,  // the 24 Tetra down_proj: 1280 + 1280 rows, 160 + 160 CTAs
    Empty1 = 4,     // one 1-CTA empty kernel after each down_proj: +36 launches
    Empty4 = 5,     // four: +144, for the slope and a linearity check
    EmptyGrid = 6,  // one 852-CTA empty grid after each down_proj: +36
    CandleTiny = 7, // one candle affine on a 1-element f16 tensor per layer (model.rs hook)
}

pub fn set_boundary_arm(a: BoundaryArm) {
    BOUNDARY_ARM.store(a as u8, std::sync::atomic::Ordering::Relaxed);
}

// Inside FusedOp::cuda_fwd, Tetra48 arm (today fused_cuda.rs:1592-1615):
//
//   let arm = BOUNDARY_ARM.load(Relaxed);
//   let split_at = match (arm, self.proj.name.as_str()) {
//       (1, n) if n.ends_with("gate_proj") => Some(6816u32),
//       (2, n) if n.ends_with("gate_proj") => Some(4864u32),
//       (3, n) if n.ends_with("down_proj") => Some(1280u32),
//       _ => None,
//   };
//   match split_at {
//       None => launch_tetra48_h(.., words, .., rscale, tail, xr, &mut y, .., d_out, ..)?,
//       Some(r1) => {
//           // Rows partition the output and the kernel derives `row` from the
//           // grid, so two launches over offset views compute the same rows
//           // with the same arithmetic: bit-identical by construction.
//           let s = *stride_u32 as usize;
//           let tw = self.proj.tail_w as usize;
//           launch_tetra48_h_views(.., &words.slice(..r1 as usize * s), .., &rscale.slice(..r1 as usize),
//                                  &tail.slice(..r1 as usize * tw), xr, &mut y.slice_mut(..r1 as usize), r1)?;
//           launch_tetra48_h_views(.., &words.slice(r1 as usize * s..), .., &rscale.slice(r1 as usize..),
//                                  &tail.slice(r1 as usize * tw..), xr, &mut y.slice_mut(r1 as usize..),
//                                  d_out - r1)?;
//       }
//   }
//   if self.proj.name.ends_with("down_proj") { self.rt.boundary_probe(arm)?; }
//
// and the same `boundary_probe` call after the int4 down_proj launch in
// FusedInt4Op::cuda_fwd, so all 36 layers get their probe.

pub const EMPTY_SRC: &str = r#"extern "C" __global__ void llvq_empty(unsigned int u) { (void)u; }"#;

// impl FusedRuntime {
//     fn boundary_probe(&self, arm: u8) -> candle_core::Result<()> {
//         let f = match &self.f_empty { Some(f) => f, None => return Ok(()) };
//         let (n, grid) = match arm { 4 => (1, 1), 5 => (4, 1), 6 => (1, 852), _ => return Ok(()) };
//         for _ in 0..n {
//             let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
//             let mut b = self.cuda.stream().launch_builder(f);
//             let z = 0u32;
//             b.arg(&z);
//             unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("llvq_empty: {e}")))?;
//         }
//         Ok(())
//     }
// }
// `f_empty` is loaded only when the mode asks for it, from EMPTY_SRC compiled
// as its own module on candle's stream, so the served NVRTC unit keeps its
// bytes and its sha256: the `NVRTC source:` line load_with prints must carry
// the digest the served door prints in the same job (gate 4 of
// prereg-persistent-0.md; 6193094310bc18ac... at HEAD 06e1602, reproduced
// offline from the working tree on 2026-11-02, persistent.md section 8).

// ---- 2. fusedrun.rs: the mode block, beside LLVQ_GQA_AB's -------------------
//
// if let Ok(v) = std::env::var("LLVQ_BOUNDARY_AB") {
//     if v != "1" {
//         anyhow::bail!("LLVQ_BOUNDARY_AB={v:?}: the value is 1, and unset is off");
//     }
//     for other in ["LLVQ_CONTEXTS", "LLVQ_TRIM_AB", "LLVQ_PREFILL_TOKENS", "LLVQ_GQA_AB", "LLVQ_GQA_PPL"] {
//         if std::env::var(other).is_ok() {
//             anyhow::bail!("LLVQ_BOUNDARY_AB runs alone, and {other} is set beside it");
//         }
//     }
//     if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
//         anyhow::bail!("LLVQ_BOUNDARY_AB sets the store itself: unset LLVQ_KV_PREALLOC");
//     }
//     let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
//     let t = Instant::now();
//     let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;   // prints `NVRTC source:` and the int4 parity line
//     f.runtime().load_boundary_probe()?;                                       // EMPTY_SRC on candle's stream
//     println!("loaded in {:.1} s. Boundary A/B: {} arms, one model, interleaved rounds",
//              t.elapsed().as_secs_f64(), BOUNDARY_ARMS.len());
//     println!("LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
//              f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches);
//     // The served trim, done here because load_with did not (source: None).
//     let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
//     if ctx.is_in_multi_stream_mode() {
//         anyhow::bail!("LLVQ_BOUNDARY_AB: the context holds several streams, and turning \
//                        cudarc's events off is only safe on one");
//     }
//     // Safe: one stream, checked above, orders every access by issue.
//     unsafe { ctx.disable_event_tracking(); }
//     println!("cudarc event tracking: off (one stream, ordered by issue)");
//     // The served attention form, set by the mode as LLVQ_GQA_AB's grouped arm does
//     // (GqaArm::Grouped.apply, fusedrun.rs:301-305), printed from the getters.
//     f.model.set_kv_store(llvq_llm::kvq::KvStore::Grow(llvq_llm::kvq::GROW_CHUNK));
//     f.model.set_attn_heads(llvq_llm::kvq::AttnHeads::Grouped);
//     println!("attention {} on {}", f.model.attn_heads().label(), f.model.kv_store().label());
//     let seed = f.tokenizer.encode(PROMPT, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
//     run_boundary_ab(&mut f.model, &seed, 256)?;
//     return Ok(());
// }

const BOUNDARY_ARMS: [(BoundaryArm, &str, u32); 8] = [
    (BoundaryArm::Served, "served", 0),
    (BoundaryArm::SplitWave, "split@wave", 36),
    (BoundaryArm::SplitHalf, "split@half", 36),
    (BoundaryArm::SplitDown, "split-down", 24),
    (BoundaryArm::Empty1, "empty x1", 36),
    (BoundaryArm::Empty4, "empty x4", 144),
    (BoundaryArm::EmptyGrid, "emptygrid x1", 36),
    (BoundaryArm::CandleTiny, "candle-tiny x1", 36),
];

/// One model, every arm in every round, same order, 2 rounds discarded then
/// ROUNDS_TIMED (5); per-arm time per decode token from 256-token generates;
/// the cost of one added boundary formed ROUND BY ROUND:
///   c = (t_arm - t_served) / boundaries_per_token,  t = elapsed / n_new.
/// Any divergence from the first arm's tokens, in any arm or round, voids the
/// run before a number is printed (the run_trim_ab rule).
fn run_boundary_ab(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    n_new: usize,
) -> anyhow::Result<()> {
    const WARMUP: usize = 2;
    let mut reference: Option<Vec<u32>> = None;
    let mut secs: Vec<Vec<f64>> = vec![Vec::new(); BOUNDARY_ARMS.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, (arm, name, _)) in BOUNDARY_ARMS.iter().enumerate() {
            llvq_llm::fused_cuda::set_boundary_arm(*arm);
            model.set_candle_probe(*arm == BoundaryArm::CandleTiny); // model.rs: one affine per layer in finish()
            let t = Instant::now();
            let out = model.generate(ids, n_new, &mut NoCapture)?;
            let s = t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!("round {round}, arm {name}: divergence at token {i}; the arms are \
                                       bit-identical by construction, the run is void");
                    }
                }
            }
            if round >= WARMUP {
                secs[k].push(s / n_new as f64);
            }
        }
    }
    llvq_llm::fused_cuda::set_boundary_arm(BoundaryArm::Served);
    model.set_candle_probe(false);
    println!("  {} tokens identical across {} arms and {} rounds",
             n_new, BOUNDARY_ARMS.len(), WARMUP + ROUNDS_TIMED);
    println!("  arm              tok/s [range]       us per added boundary, round by round [range]");
    for (k, (_, name, nb)) in BOUNDARY_ARMS.iter().enumerate() {
        let rates: Vec<f64> = secs[k].iter().map(|t| 1.0 / t).collect();
        let (m, lo, hi) = rate_stats(&rates);
        if *nb == 0 {
            println!("  {name:<16} {m:6.1} [{lo:.1}-{hi:.1}]");
            continue;
        }
        let per: Vec<f64> = secs[0].iter().zip(&secs[k]).map(|(a, b)| (b - a) / *nb as f64 * 1e6).collect();
        let (pm, pl, ph) = rate_stats(&per);
        println!("  {name:<16} {m:6.1} [{lo:.1}-{hi:.1}]   {pm:.2} [{pl:.2}-{ph:.2}]");
    }
    // Derived classes, formed round by round from the per-boundary series
    // (printed, never gated): drain+ramp = split@wave - empty x1;
    // underfill = split@half - split@wave; dispatch = emptygrid x1 - empty x1;
    // linearity = empty x4 against empty x1.
    Ok(())
}
