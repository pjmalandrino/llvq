//! The split-KV decode-attention kernel alone: identity, read rate, verify
//! cost, registers. The bench of `proofs/preregistration-attn-bench-*.md`.
//!
//! No artifact, no model: synthetic per-layer KV sets of the served geometry
//! (8 KV heads, head dim 128, f16), 36 sets at G = 4 (Qwen3-4B) and 40 at
//! G = 5 (Qwen3-14B), each its own buffer, so one pass of 36 or 40 launches
//! reads every byte once and reuses none: above the 96 MB L2 at every length
//! timed. Times come from CUDA events around each pass, on a fresh stream with
//! cudarc's event tracking off.
//!
//! What it prints, in order:
//!  1. the source provenance and the card;
//!  2. every entry point's driver report (registers, spill, shared);
//!  3. identity, every variant: the load-time check of the served path
//!     (`AttnDec::self_check`), then at each timed length and k in {1, 2, 4, 8}
//!     the card against the CPU replay, the k rows of a verify call against k
//!     decode calls, and a grid of every split the capacity holds against the
//!     eager grid, all bit for bit;
//!  4. timing: 2 discarded and 7 timed rounds; inside a round every (variant,
//!     G, L, k) runs once, in one fixed order. Medians with the range over
//!     rounds; eta_k and r_k formed round by round (house rule 7).
//!
//! eta_k(L) = sets x 8 x L x 128 x 2 x 2 B / (t x 864 GB/s), the fraction of
//! the L40S's 864 GB/s the pass reads its cache at (*computed*); r_k(L) =
//! t(k) / t(1), round by round. A bench: it informs the build, it gates
//! nothing (METHODE §1).
//!
//! Usage: attnbench [L,L,...]   (default 2048,4096,8192)

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("attnbench needs a CUDA card: Linux only");
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("attnbench: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use llvq_cuda::attn::{AttnDec, AttnModule};
    use llvq_cuda::attn_ref::{self, Kv, HD, SPLIT};

    /// The nameplate every eta of this repository is a fraction of (rule 5).
    const PEAK_GBS: f64 = 864.0;
    const N_KV: usize = 8;
    const KS: [usize; 4] = [1, 2, 4, 8];
    const DISCARD: usize = 2;
    const ROUNDS: usize = 7;

    let lengths: Vec<usize> = match std::env::args().nth(1) {
        None => vec![2048, 4096, 8192],
        Some(a) => a
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| format!("length {s:?}: {e}")))
            .collect::<Result<_, _>>()?,
    };
    if lengths.iter().any(|&l| l < 8) {
        return Err("every length must hold the 8 rows of a verify call".into());
    }
    let lmax = *lengths.iter().max().expect("at least one length");
    let cap = lmax.div_ceil(SPLIT) * SPLIT;

    let module = AttnModule::fresh()?;
    println!("{}", module.provenance());
    let cuda = module.cuda();
    let dev = cuda.device()?;
    println!(
        "card: {} sm_{}{}, {} SMs, L2 {} MB, nameplate {:.0} GB/s; eta is against {PEAK_GBS} GB/s",
        dev.name,
        dev.compute_cap.0,
        dev.compute_cap.1,
        dev.sm_count,
        dev.l2_bytes / (1 << 20),
        dev.nameplate_bandwidth_gbs()
    );

    // Variants in the fixed order every round runs them.
    let variants: Vec<(usize, usize)> = vec![(4, 64), (4, 32), (5, 64), (5, 32)];
    let mut decs = Vec::new();
    for &(g, unit) in &variants {
        let d = AttnDec::new(module.clone(), N_KV, g, unit)?;
        println!("report {}", d.report_line());
        decs.push(d);
    }

    // ---- identity ----
    let c = attn_ref::score_scale(HD);
    let mut failed: Vec<String> = Vec::new();
    for d in &decs {
        match d.self_check() {
            Ok(line) => println!("identity load-check {line}"),
            Err(e) => {
                println!("identity FAIL {}: {e}", d.name);
                failed.push(d.name.to_string());
                continue;
            }
        }
        let g = d.groups;
        let mut ok = true;
        for &l in &lengths {
            let cap1 = l.div_ceil(SPLIT) * SPLIT;
            let kc = attn_ref::fill(N_KV * cap1 * HD, l as u64 ^ 0xa, 1.0);
            let vc = attn_ref::fill(N_KV * cap1 * HD, l as u64 ^ 0xb, 1.0);
            let (dk, dv) = (cuda.up_u16(&kc)?, cuda.up_u16(&vc)?);
            for &k in &KS {
                let pos0 = l - k;
                let q = attn_ref::fill(N_KV * g * k * HD, (l * 16 + k) as u64, 4.0);
                let want = attn_ref::attend(&q, &Kv { k: &kc, v: &vc, cap: cap1 }, N_KV, g, pos0, k, c, d.unit);
                let dq = cuda.up_u16(&q)?;
                let mut out = cuda.zeros_u16(want.len())?;
                for grid in [None, Some(cap1 / SPLIT + 2)] {
                    d.attend(&dq.slice(..), &dk.slice(..), &dv.slice(..), &mut out, pos0, k, cap1, c, grid)?;
                    if cuda.down_u16(&out)? != want {
                        println!("identity FAIL {} L={l} k={k} grid={grid:?}: card differs from the replay", d.name);
                        ok = false;
                    }
                }
                if k == KS[KS.len() - 1] {
                    for r in 0..k {
                        let mut q1 = Vec::new();
                        for hq in 0..N_KV * g {
                            q1.extend_from_slice(&q[(hq * k + r) * HD..(hq * k + r + 1) * HD]);
                        }
                        let dq1 = cuda.up_u16(&q1)?;
                        let mut o1 = cuda.zeros_u16(q1.len())?;
                        d.attend(&dq1.slice(..), &dk.slice(..), &dv.slice(..), &mut o1, pos0 + r, 1, cap1, c, None)?;
                        let got = cuda.down_u16(&o1)?;
                        for hq in 0..N_KV * g {
                            if got[hq * HD..(hq + 1) * HD] != want[(hq * k + r) * HD..(hq * k + r + 1) * HD] {
                                println!("identity FAIL {} L={l}: verify row {r} differs from its decode call", d.name);
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
        if ok {
            println!(
                "identity {} pass: lengths {lengths:?}, k {KS:?}, eager and wide grids, verify rows alone, bit for bit",
                d.name
            );
        } else {
            failed.push(d.name.to_string());
        }
    }

    // ---- timing ----
    // One buffer pair per layer, each `[8][cap][128]`, filled from one host
    // layer: distinct buffers, so a pass reads each once.
    let host_k = attn_ref::fill(N_KV * cap * HD, 0x51, 1.0);
    let host_v = attn_ref::fill(N_KV * cap * HD, 0x52, 1.0);
    let sets_of = |g: usize| if g == 4 { 36 } else { 40 };
    let max_sets = 40;
    let mut kcs = Vec::with_capacity(max_sets);
    let mut vcs = Vec::with_capacity(max_sets);
    for _ in 0..max_sets {
        kcs.push(cuda.up_u16(&host_k)?);
        vcs.push(cuda.up_u16(&host_v)?);
    }
    let qbuf = cuda.up_u16(&attn_ref::fill(N_KV * 5 * 8 * HD, 0x53, 4.0))?;
    let mut out = cuda.zeros_u16(N_KV * 5 * 8 * HD)?;
    let (e0, e1) = (cuda.new_event()?, cuda.new_event()?);

    let mut configs = Vec::new();
    for (vi, _) in variants.iter().enumerate() {
        for &l in &lengths {
            for &k in &KS {
                configs.push((vi, l, k));
            }
        }
    }
    // times[config][round], ms a pass.
    let mut times = vec![Vec::with_capacity(ROUNDS); configs.len()];
    for round in 0..DISCARD + ROUNDS {
        for (ci, &(vi, l, k)) in configs.iter().enumerate() {
            let d = &decs[vi];
            if failed.iter().any(|f| f == d.name) {
                continue;
            }
            let sets = sets_of(d.groups);
            let nq = N_KV * d.groups * k * HD;
            e0.record(cuda.stream()).map_err(|e| format!("event: {e}"))?;
            for s in 0..sets {
                d.attend(
                    &qbuf.slice(..nq),
                    &kcs[s].slice(..),
                    &vcs[s].slice(..),
                    &mut out,
                    l - k,
                    k,
                    cap,
                    c,
                    None,
                )?;
            }
            e1.record(cuda.stream()).map_err(|e| format!("event: {e}"))?;
            cuda.sync()?;
            let ms = e0.elapsed_ms(&e1).map_err(|e| format!("elapsed: {e}"))? as f64;
            if round >= DISCARD {
                times[ci].push(ms);
            }
        }
    }

    let med = |v: &mut Vec<f64>| -> (f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    println!("\ntiming: {ROUNDS} rounds kept after {DISCARD}; median [min-max] over rounds (*measured*)");
    println!("variant              L      k   ms a pass                    eta_k                r_k = t(k)/t(1)");
    for (ci, &(vi, l, k)) in configs.iter().enumerate() {
        let d = &decs[vi];
        if times[ci].is_empty() {
            println!("{:<18} {l:>6} {k:>3}   (excluded: identity failed)", d.name);
            continue;
        }
        let sets = sets_of(d.groups) as f64;
        let bytes = sets * (N_KV * l * HD * 2 * 2) as f64;
        let mut t = times[ci].clone();
        let (tm, tlo, thi) = med(&mut t);
        let mut eta: Vec<f64> = times[ci].iter().map(|ms| bytes / (ms * 1e-3) / (PEAK_GBS * 1e9)).collect();
        let (em, elo, ehi) = med(&mut eta);
        // r_k round by round, against the k = 1 config of the same (variant, L).
        let c1 = configs.iter().position(|&(v2, l2, k2)| v2 == vi && l2 == l && k2 == 1).expect("k=1 config");
        let mut r: Vec<f64> = times[ci].iter().zip(&times[c1]).map(|(a, b)| a / b).collect();
        let (rm, rlo, rhi) = med(&mut r);
        println!(
            "{:<18} {l:>6} {k:>3}   {tm:.4} [{tlo:.4}-{thi:.4}]   {em:.3} [{elo:.3}-{ehi:.3}]   {rm:.3} [{rlo:.3}-{rhi:.3}]",
            d.name
        );
    }
    if !failed.is_empty() {
        println!("\nidentity failed for {failed:?}: excluded from every reading (prereg rule 1)");
    }
    Ok(())
}
