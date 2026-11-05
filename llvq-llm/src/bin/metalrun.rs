//! `metalrun`: the decode bench of a sealed file on the Mac.
//!
//! `fusedrun` is CUDA only and `chat` disclaims its own tok/s, so before this
//! binary no Metal number had a protocol. This one times arms of the SAME
//! sealed file in ONE process, interleaved, because this machine drifts with
//! its thermals (49.7 tok/s cold, 23.1 after three hours of compiling, commit
//! 5f439df): only a ratio formed round by round in one process survives that.
//!
//! ```text
//! metalrun time  <sealed> <config> <arms> [n_new] [timed_rounds]
//! metalrun gate  <sealed> <config> <arms>
//! metalrun phased <sealed> <config> <arm> [n_new]
//! metalrun dispatch
//! ```
//!
//! `<arms>` is a comma list. Each fused arm is a Tetra matvec, optionally
//! followed by modifiers:
//!
//! | spec | what it loads |
//! |---|---|
//! | `lut` | the served Metal kernels, the int4 records past the wall through `tv_q4_metal_tiled` at 2,048 columns |
//! | `lutp`, `lutd`, `lutdp` | the Tetra row-loop twins of `kernels/llvq_tetra48_rows.metal` |
//! | `+q4p` | the prefetching int4 twins of `kernels/tv_q4_pf.metal` |
//! | `+t8192`, `+t256` | another int4 tile (zero bits) |
//! | `+grp` | grouped attention: a TOLERANCE event, refused on the served Metal door, a measurement arm only |
//! | `D` | the dense reconstruction, f16, through candle (`sealed::load`) |
//!
//! The first arm is the reference: every other arm's tokens and every
//! projection's output are compared with it.
//!
//! `time` runs, in order: the bit gate (below), then two discarded and
//! `timed_rounds` timed rounds, every arm in the same order each round, each a
//! `generate_timed` of `n_new` tokens after the 5-token prompt of `fusedrun`.
//! It prints, per arm, the door rate (n_new over prompt plus decode, the
//! quantity of `docs/ETAT.md`'s table) and the decode rate ((n_new - 1) over
//! decode), as medians with the range over rounds, and each arm over the
//! reference formed round by round (rule 7).
//!
//! The bit gate: one decode step of the reference arm with every projection
//! input captured (q/k/v, o, gate/up, down of each layer), then every arm's
//! projection run on those same inputs and compared bit for bit with the
//! reference's, and the head likewise. That puts each kernel twin against
//! the served kernel on the 252 records of the real file, not only on the
//! synthetic fixtures of the tests.
//!
//! Every `LLVQ_*` and `CANDLE_*` variable is refused: the arms are named here,
//! and `CANDLE_METAL_COMPUTE_PER_BUFFER` alone moved a Metal step from 33.9
//! to 27.7 tok/s (41e6628).
//!
//! Never a served config: the served door is `served::load_resolved` with
//! `Served::apply`, which this binary calls for every fused arm before it
//! adds a measurement modifier.

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("metalrun requires macOS and the `metal` feature")
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> anyhow::Result<()> {
    mac::main()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod mac {
    use anyhow::Context;
    use candle_core::{DType, Device, Tensor, D};
    use llvq_llm::fused_metal::{load_resolved_tuned, MetalTuning, SERVED_TETRA};
    use llvq_llm::kvq::AttnHeads;
    use llvq_llm::model::{Act, Capture, NoCapture, Qwen3};
    use llvq_llm::served::Served;
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::Instant;

    const PROMPT: &str = "The capital of France is";
    const DISCARDED: usize = 2;
    const CHUNK: usize = 256;
    const REFUSED_PREFIXES: [&str; 2] = ["LLVQ_", "CANDLE_"];
    const SITES: [(&str, Act); 7] = [
        ("self_attn.q_proj", Act::Attn),
        ("self_attn.k_proj", Act::Attn),
        ("self_attn.v_proj", Act::Attn),
        ("self_attn.o_proj", Act::AttnOut),
        ("mlp.gate_proj", Act::Mlp),
        ("mlp.up_proj", Act::Mlp),
        ("mlp.down_proj", Act::MlpOut),
    ];

    pub fn main() -> anyhow::Result<()> {
        environment_check()?;
        let a: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| -> anyhow::Result<usize> {
            Ok(match a.get(i) {
                Some(s) => s.parse().with_context(|| format!("argument {i}: {s:?} is not a count"))?,
                None => d,
            })
        };
        match a.first().map(String::as_str) {
            Some("time") if a.len() >= 4 => time(&a[1], &a[2], &a[3], num(4, 128)?, num(5, 7)?),
            Some("gate") if a.len() == 4 => {
                let device = Device::new_metal(0)?;
                let cfg = header(&a[1], &a[2])?;
                let arms = load_arms(&a[1], &cfg, &device, &a[3])?;
                bit_gate(&arms)?;
                Ok(())
            }
            Some("phased") if a.len() >= 4 => phased(&a[1], &a[2], &a[3], num(4, 64)?),
            Some("dispatch") => dispatch(),
            _ => anyhow::bail!(
                "usage: metalrun time <sealed> <config> <arms> [n_new=128] [timed_rounds=7]\n\
                 \x20      metalrun gate <sealed> <config> <arms>\n\
                 \x20      metalrun phased <sealed> <config> <arm> [n_new=64]\n\
                 \x20      metalrun dispatch\n\
                 arms: comma list of lut|lutp|lutd|lutdp[+q4p][+t<cols>][+grp] or D; the first is the reference"
            ),
        }
    }

    // ---- the outside world ---------------------------------------------------

    fn environment_check() -> anyhow::Result<()> {
        let seen: Vec<String> = std::env::vars_os()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v))
            .filter(|(k, _)| REFUSED_PREFIXES.iter().any(|p| k.starts_with(p)))
            .map(|(k, v)| format!("{k}={v:?}"))
            .collect();
        anyhow::ensure!(
            seen.is_empty(),
            "metalrun names its arms itself and reads no knob; unset {}",
            seen.join(", ")
        );
        println!("environment: no LLVQ_* or CANDLE_* variable");
        Ok(())
    }

    fn command(prog: &str, args: &[&str]) -> String {
        std::process::Command::new(prog)
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|e| format!("{prog} failed: {e}"))
    }

    /// The other processes' summed %CPU, this one excluded.
    fn others_cpu() -> f64 {
        let me = std::process::id().to_string();
        command("ps", &["-A", "-o", "pid=,%cpu="])
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let (pid, cpu) = (it.next()?, it.next()?);
                (pid != me).then(|| cpu.parse::<f64>().ok()).flatten()
            })
            .sum()
    }

    /// The GPU's "Device Utilization %", no privilege needed.
    fn gpu_utilization() -> Option<u32> {
        let out = command("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"]);
        out.split("\"Device Utilization %\"=").nth(1)?.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    }

    fn therm() -> String {
        command("pmset", &["-g", "therm"]).replace('\n', " | ")
    }

    fn header(path: &str, cfg_path: &str) -> anyhow::Result<Served> {
        let exe = std::env::current_exe()?;
        println!("binary  {} sha256 {}", exe.display(), llvq_llm::digest::sha256_file(&exe)?);
        println!("object  {path} sha256 {}", llvq_llm::digest::sha256_file(Path::new(path))?);
        println!("config  {cfg_path} sha256 {}", llvq_llm::digest::sha256_file(Path::new(cfg_path))?);
        let cfg = Served::read(Path::new(cfg_path)).map_err(anyhow::Error::msg)?;
        println!("{}", cfg.provenance());
        println!("machine {} | {}", command("sysctl", &["-n", "machdep.cpu.brand_string"]), therm());
        println!(
            "load    GPU {:?} % busy, other processes {:.0} % CPU",
            gpu_utilization(),
            others_cpu()
        );
        Ok(cfg)
    }

    // ---- arms ----------------------------------------------------------------

    struct Arm {
        label: String,
        model: Qwen3,
        tokenizer: tokenizers::Tokenizer,
        fused: bool,
        /// Projections plus the head table: what a decode step reads.
        weights_a_token: f64,
        /// The loader's host count, projections and carried.
        host_bytes: f64,
    }

    #[derive(Debug)]
    struct Spec {
        tuning: MetalTuning,
        grouped: bool,
    }

    fn parse_spec(s: &str) -> anyhow::Result<Option<Spec>> {
        if s == "D" {
            return Ok(None);
        }
        let mut parts = s.split('+');
        let kernel = match parts.next().unwrap_or("") {
            "lut" => SERVED_TETRA,
            "lutp" => "tv_tetra48_metal_lutp",
            "lutd" => "tv_tetra48_metal_lutd",
            "lutdp" => "tv_tetra48_metal_lutdp",
            other => anyhow::bail!("arm {s:?}: {other:?} is not lut, lutp, lutd, lutdp or D"),
        };
        let mut spec = Spec { tuning: MetalTuning { tetra_kernel: kernel, ..Default::default() }, grouped: false };
        for m in parts {
            match m {
                "q4p" => spec.tuning.q4_prefetch = true,
                "grp" => spec.grouped = true,
                t if t.starts_with('t') => {
                    spec.tuning.q4_tile_cols = t[1..].parse().with_context(|| format!("arm {s:?}: tile {t:?}"))?
                }
                other => anyhow::bail!("arm {s:?}: unknown modifier {other:?}"),
            }
        }
        spec.tuning.validate()?;
        Ok(Some(spec))
    }

    fn head_bytes(c: &candle_transformers::models::qwen3::Config, bits: Option<u8>) -> f64 {
        let (v, d) = (c.vocab_size as f64, c.hidden_size as f64);
        match bits {
            None => v * d * 2.0,
            Some(b) => v * d * b as f64 / 8.0 + v * (d / 64.0) * 4.0,
        }
    }

    fn dense_weights_a_token(c: &candle_transformers::models::qwen3::Config) -> f64 {
        let (d, i, l) = (c.hidden_size, c.intermediate_size, c.num_hidden_layers);
        let (q, kv) = (c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim);
        2.0 * (l * (d * q + 2 * d * kv + q * d + 3 * d * i) + c.vocab_size * d) as f64
    }

    fn load_arm(path: &str, cfg: &Served, device: &Device, spec_s: &str) -> anyhow::Result<Arm> {
        let t0 = Instant::now();
        let arm = match parse_spec(spec_s)? {
            None => {
                let mut s = llvq_llm::sealed::load(path, DType::F16, device, cfg.kv)?;
                cfg.apply(&mut s.model, device).map_err(anyhow::Error::msg)?;
                Arm {
                    label: spec_s.to_string(),
                    weights_a_token: dense_weights_a_token(&s.config),
                    host_bytes: 2.0 * (s.quantized_weights + s.carried_weights) as f64,
                    model: s.model,
                    tokenizer: s.tokenizer,
                    fused: false,
                }
            }
            Some(spec) => {
                println!("arm {spec_s}: loading with {:?}", spec.tuning);
                let mut f = load_resolved_tuned(
                    path,
                    device,
                    DType::F16,
                    cfg.layout,
                    cfg.embed,
                    cfg.rot_share,
                    cfg.fuse,
                    cfg.kv,
                    Some("metalrun"),
                    Some(spec.tuning),
                )?;
                // The served door's own pair, store then form, first.
                cfg.apply(&mut f.model, device).map_err(anyhow::Error::msg)?;
                if spec.grouped {
                    // A measurement arm: Served::apply refuses this form on
                    // Metal by name, and this binary is not a served door.
                    f.model.set_kv_store(AttnHeads::Grouped.store());
                    f.model.set_attn_heads(AttnHeads::Grouped);
                }
                Arm {
                    label: spec_s.to_string(),
                    weights_a_token: f.runtime_bytes as f64 + head_bytes(&f.config, cfg.embed.bits()),
                    host_bytes: (f.runtime_bytes + f.carried_bytes) as f64,
                    model: f.model,
                    tokenizer: f.tokenizer,
                    fused: true,
                }
            }
        };
        let load_s = t0.elapsed().as_secs_f64();
        println!(
            "arm {:<12} loaded in {load_s:5.1} s | host count {:.3} GB | {:.3} GB of weights a token | attention {} on {}{}",
            spec_s,
            arm.host_bytes / 1e9,
            arm.weights_a_token / 1e9,
            arm.model.attn_heads().label(),
            arm.model.kv_store().label(),
            if arm.fused { "" } else { " | dense reconstruction" },
        );
        Ok(arm)
    }

    fn load_arms(path: &str, cfg: &Served, device: &Device, list: &str) -> anyhow::Result<Vec<Arm>> {
        let specs: Vec<&str> = list.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        anyhow::ensure!(!specs.is_empty(), "no arm named");
        let mut seen = std::collections::HashSet::new();
        for s in &specs {
            anyhow::ensure!(seen.insert(*s), "arm {s:?} named twice");
        }
        specs.iter().map(|s| load_arm(path, cfg, device, s)).collect()
    }

    // ---- the bit gate ----------------------------------------------------------

    /// Keeps each projection input of the step it watches.
    struct Keep(HashMap<(usize, u64), Tensor>);
    impl Capture for Keep {
        fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
            self.0.insert((layer, act.index()), x.clone());
            Ok(())
        }
    }

    fn bits16(t: &Tensor) -> anyhow::Result<Vec<u16>> {
        Ok(t.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>()?.iter().map(|v| v.to_bits()).collect())
    }

    /// Every fused arm's 252 projections and head against the reference arm's,
    /// on the reference's own decode-step inputs, bit for bit. Grouped arms
    /// are compared too: the gate is about the projections, which the form
    /// does not touch.
    fn bit_gate(arms: &[Arm]) -> anyhow::Result<()> {
        let r = &arms[0];
        anyhow::ensure!(r.fused, "the reference arm of the bit gate is a fused arm");
        let ids = r.tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let dev = r.model.device().clone();
        let mut caches = r.model.fresh_caches();
        let input = Tensor::from_slice(&ids, (1, ids.len()), &dev)?;
        r.model.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let mut keep = Keep(HashMap::new());
        let step = Tensor::from_slice(&[ids[ids.len() - 1]], (1, 1), &dev)?;
        let h = r.model.hidden_cached(&step, ids.len(), &mut caches, &mut keep)?;
        let want_head = bits16(&r.model.project_head(&h)?)?;
        let mut want: HashMap<(usize, &str), Vec<u16>> = HashMap::new();
        for (i, blk) in r.model.blocks.iter().enumerate() {
            for (name, act) in SITES {
                let x = keep.0.get(&(i, act.index())).with_context(|| format!("layer {i}: {act:?} not captured"))?;
                want.insert((i, name), bits16(&blk.proj(name).forward(x)?)?);
            }
        }
        for a in arms.iter().skip(1).filter(|a| a.fused) {
            let (mut equal, mut compared, mut first_bad) = (0usize, 0usize, None::<String>);
            for (i, blk) in a.model.blocks.iter().enumerate() {
                for (name, act) in SITES {
                    let x = &keep.0[&(i, act.index())];
                    let got = bits16(&blk.proj(name).forward(x)?)?;
                    compared += 1;
                    match got.iter().zip(&want[&(i, name)]).position(|(p, q)| p != q) {
                        None => equal += 1,
                        Some(row) => {
                            first_bad.get_or_insert(format!("layer {i} {name} row {row}"));
                        }
                    }
                }
            }
            let got_head = bits16(&a.model.project_head(&h)?)?;
            compared += 1;
            match got_head.iter().zip(&want_head).position(|(p, q)| p != q) {
                None => equal += 1,
                Some(row) => {
                    first_bad.get_or_insert(format!("head row {row}"));
                }
            }
            println!(
                "bit gate: arm {} against {}: {equal} of {compared} records equal bit for bit{}",
                a.label,
                r.label,
                first_bad.as_ref().map(|s| format!(", FIRST DIFFERENCE {s}")).unwrap_or_default()
            );
            anyhow::ensure!(equal == compared, "BIT GATE FAILS for arm {}", a.label);
        }
        Ok(())
    }

    // ---- time ----------------------------------------------------------------

    fn stats(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let n = s.len();
        let med = if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) };
        (med, s[0], s[n - 1])
    }

    fn time(path: &str, cfg_path: &str, list: &str, n_new: usize, timed: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n_new >= 2 && timed >= 1, "n_new >= 2 and at least one timed round");
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let arms = load_arms(path, &cfg, &device, list)?;
        if arms.iter().filter(|a| a.fused).count() > 1 {
            bit_gate(&arms)?;
        }
        let ids = arms[0].tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        println!(
            "prompt {PROMPT:?} = {ids:?}; {n_new} new tokens; {DISCARDED} discarded + {timed} timed rounds; order {}",
            arms.iter().map(|a| a.label.as_str()).collect::<Vec<_>>().join(" ")
        );
        let n = arms.len();
        let (mut door, mut dec) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        let mut texts: Vec<Option<Vec<u32>>> = vec![None; n];
        let mut cpu_max = 0f64;
        for r in 0..DISCARDED + timed {
            for (i, a) in arms.iter().enumerate() {
                let g = a.model.generate_timed(&ids, n_new, CHUNK)?;
                let cpu = others_cpu();
                cpu_max = cpu_max.max(cpu);
                match &texts[i] {
                    None => texts[i] = Some(g.tokens.clone()),
                    Some(first) => {
                        anyhow::ensure!(first == &g.tokens, "arm {} changed its tokens in round {r}", a.label)
                    }
                }
                let (rd, rc) = (n_new as f64 / (g.prompt_s + g.decode_s), (n_new - 1) as f64 / g.decode_s);
                if r >= DISCARDED {
                    door[i].push(rd);
                    dec[i].push(rc);
                }
                println!(
                    "round {r}{} arm {:<12} door {rd:7.2} tok/s | decode {rc:7.2} tok/s | first token {:6.1} ms | others {cpu:.0} % CPU",
                    if r < DISCARDED { " (discarded)" } else { "" },
                    a.label,
                    g.prompt_s * 1e3,
                );
            }
        }
        println!("--- medians over {timed} timed rounds [min-max], niced, measured on this Mac");
        for (i, a) in arms.iter().enumerate() {
            let (m, lo, hi) = stats(&door[i]);
            let (dm, dlo, dhi) = stats(&dec[i]);
            println!(
                "arm {:<12} door {m:7.2} tok/s [{lo:.2}-{hi:.2}] | decode {dm:7.2} tok/s [{dlo:.2}-{dhi:.2}] | {:.2} ms a decode token | {:.1} GB/s of weights (computed)",
                a.label,
                1e3 / dm,
                dm * a.weights_a_token / 1e9
            );
        }
        let reference = &arms[0].label;
        for (i, a) in arms.iter().enumerate().skip(1) {
            let ratio = |v: &[Vec<f64>]| -> (f64, f64, f64) {
                stats(&v[i].iter().zip(&v[0]).map(|(p, q)| p / q).collect::<Vec<_>>())
            };
            let (m, lo, hi) = ratio(&door);
            let (dm, dlo, dhi) = ratio(&dec);
            println!(
                "ratio {:<12} / {reference}: door x{m:.3} [{lo:.3}-{hi:.3}] | decode x{dm:.3} [{dlo:.3}-{dhi:.3}], round by round",
                a.label
            );
        }
        let t0 = texts[0].as_ref().expect("one round ran");
        for (i, a) in arms.iter().enumerate().skip(1) {
            let ti = texts[i].as_ref().expect("one round ran");
            let verdict = match ti.iter().zip(t0).position(|(p, q)| p != q) {
                None if ti.len() == t0.len() => format!("the same {} tokens", t0.len()),
                None => "different lengths".to_string(),
                Some(p) => format!("FIRST DIVERGENCE at token {}", p + 1),
            };
            println!("tokens {:<12} against {reference}: {verdict}", a.label);
        }
        let digest = |t: &[u32]| {
            let bytes: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
            llvq_llm::digest::sha256_hex(&bytes)
        };
        println!("tokens of {reference}: sha256 {} (u32 little-endian)", digest(t0));
        println!(
            "text of {reference}: {:?}",
            arms[0].tokenizer.decode(t0, true).map_err(anyhow::Error::msg)?
        );
        println!("other processes, worst sample between arms: {cpu_max:.0} % CPU");
        println!("end     {} | GPU {:?} % busy", therm(), gpu_utilization());
        Ok(())
    }

    // ---- phased: where a decode token goes, fenced, off protocol --------------

    fn phased(path: &str, cfg_path: &str, spec: &str, n_new: usize) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let a = load_arm(path, &cfg, &device, spec)?;
        let ids = a.tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let reference = a.model.generate(&ids, n_new, &mut NoCapture)?;
        let (toks, rep) = a.model.generate_phased(&ids, n_new)?;
        anyhow::ensure!(toks == reference, "phased tokens differ from generate");
        let med = |v: &[f64]| stats(v);
        let line = |name: &str, v: &[f64]| {
            let (m, lo, hi) = med(v);
            println!("phase {name:<7} {m:7.3} ms [{lo:.3}-{hi:.3}]");
        };
        println!("OFF PROTOCOL: every phase closed by a device synchronize; never a rate");
        line("embed", &rep.embed_ms);
        line("blocks", &rep.blocks_ms);
        line("head", &rep.head_ms);
        line("argmax", &rep.rest_ms);
        Ok(())
    }

    // ---- dispatch: what one candle launch costs on this Mac -------------------

    /// `n` dependent one-element candle ops, then one read back. Each op is
    /// one compute dispatch, so the per-op time is the floor a launch costs
    /// on this Mac through candle: encoding, command-buffer handling and the
    /// GPU's own dispatch latency, with a dependency chain like a decode
    /// step's.
    fn dispatch() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        println!("machine {} | {}", command("sysctl", &["-n", "machdep.cpu.brand_string"]), therm());
        let x0 = Tensor::from_slice(&[1.0f32], (1, 1), &device)?.to_dtype(DType::F16)?;
        for n in [1usize, 64, 1024, 4096] {
            let mut per = Vec::new();
            for _ in 0..9 {
                let t0 = Instant::now();
                let mut x = x0.clone();
                for _ in 0..n {
                    x = x.affine(1.0, 0.0)?;
                }
                let _ = x.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                per.push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            let (m, lo, hi) = stats(&per[2..]);
            println!("dispatch: {n:5} dependent affine ops then one read: {m:8.2} us an op [{lo:.2}-{hi:.2}], 7 rounds");
        }
        // The same count of launches on a row-sized activation, f16, the
        // shape of a decode step's elementwise ops.
        let v = Tensor::zeros((1, 2560), DType::F16, &device)?;
        let mut per = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            let mut x = v.clone();
            for _ in 0..1024 {
                x = (&x + &v)?;
            }
            let _ = x.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            per.push(t0.elapsed().as_secs_f64() * 1e6 / 1024.0);
        }
        let (m, lo, hi) = stats(&per[2..]);
        println!("dispatch: 1024 dependent [1, 2560] f16 adds then one read: {m:8.2} us an op [{lo:.2}-{hi:.2}]");
        let _ = D::Minus1;
        Ok(())
    }
}
