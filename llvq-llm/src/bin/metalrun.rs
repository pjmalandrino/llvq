//! `metalrun`: the decode bench of a sealed file on the Mac.
//!
//! `fusedrun` is CUDA only and `chat` disclaims its own tok/s, so before this
//! binary no Metal number had a protocol. This one times arms of the SAME
//! sealed file in ONE process, interleaved, because this machine drifts with
//! its thermals (49.7 tok/s cold, 23.1 after three hours of compiling, commit
//! 5f439df): only a ratio formed round by round in one process survives that.
//!
//! ```text
//! metalrun time  <sealed> <config> <arms> [n_new] [timed_rounds] [prompt_tokens]
//! metalrun gate  <sealed> <config> <arms>
//! metalrun phased <sealed> <config> <arm> [n_new]
//! metalrun dispatch
//! metalrun ppl <sealed> <config> <arm> [ctx] [windows] [grp|ker]
//! metalrun chat <sealed> <config> <arm> [temp] [max_new]
//! metalrun profile <sealed> <config> <arm> [timed_rounds]
//! ```
//!
//! `<arms>` is a comma list. Each fused arm is a Tetra matvec, optionally
//! followed by modifiers:
//!
//! | spec | what it loads |
//! |---|---|
//! | `lut` | the served Metal kernels, the int4 records past the wall through `tv_q4_metal_tiled` at 2,048 columns |
//! | `lutp`, `lutd`, `lutdp`, `lutd2`, `lutd4` | the Tetra row-loop twins of `kernels/llvq_tetra48_rows.metal` |
//! | `+q4p` | the prefetching int4 twins of `kernels/tv_q4_pf.metal` |
//! | `+h` | the half-io twins: the Tetra matvec stores f16, the int4 matvecs read and store f16, no adapter casts (`lut`, `lutd` only) |
//! | `+p` | the Tetra prefill kernel: a prompt chunk of 8 rows a launch, each block decoded once for the 8 (`tv_tetra48_metal_lutdr`); same bits a row |
//! | `+t8192`, `+t256` | another int4 tile (zero bits) |
//! | `+grp` | grouped attention, refused on the served Metal door: a measurement arm |
//! | `+ker` | the split-KV decode-attention kernel (`kernels/attn_dec.metal`) from position 0, as the 4B's served card config reads it; refused on the served Metal door: a measurement arm |
//! | `D` | the dense reconstruction, f16, through candle (`sealed::load`) |
//!
//! The first arm is the reference: every other arm's tokens and every
//! projection's output are compared with it.
//!
//! `time` runs, in order: the bit gate (below), then two discarded and
//! `timed_rounds` timed rounds, every arm in the same order each round, each a
//! `generate_timed` of `n_new` tokens after the 5-token prompt of `fusedrun`,
//! or, when `prompt_tokens` is not 0, after the first `prompt_tokens` tokens of
//! the WikiText-2 test split (natural text, prefilled in chunks of 256 rows).
//! It prints, per arm, the door rate (n_new over prompt plus decode, the
//! quantity of `docs/ETAT.md`'s table) and the decode rate ((n_new - 1) over
//! decode), as medians with the range over rounds, and each arm over the
//! reference formed round by round (rule 7).
//!
//! `chat` loads one arm and talks to it through `llvq_llm::repl`, the loop of
//! `bin/chat`: streamed, stopped on `<|im_end|>`, cache kept across turns.
//! `temp` 0 is greedy (default 0.7, top_p 0.9, seed 0). Its tok/s is a REPL's
//! rate, never a measurement: it exists so that a person can try an arm.
//!
//! `profile` splits the `blocks` phase of a decode token by op class. It
//! captures one decode step's projection inputs, then replays the per-layer
//! ops of the 36 blocks in model order (norms, rotations, every Tetra and int4
//! matvec, the activation, the residual adds) and times that sequence closed
//! by one device synchronize: with every class, with each class left out, and
//! with each class alone, interleaved round by round, each reading preceded
//! by an untimed full pass and averaged over three passes. A class's marginal cost
//! is the sequence with it minus the sequence without it, so every matrix is
//! read in its real order and the system cache sees the real footprint. What
//! the sequence leaves out (RoPE, q and k norms, the attention, the cache
//! append, reshapes) is the `blocks` phase minus the full sequence, both read
//! in the same process. Off protocol: closed by a synchronize, never a rate.
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
            Some("time") if a.len() >= 4 => time(&a[1], &a[2], &a[3], num(4, 128)?, num(5, 7)?, num(6, 0)?),
            Some("gate") if a.len() == 4 => {
                let device = Device::new_metal(0)?;
                let cfg = header(&a[1], &a[2])?;
                let arms = load_arms(&a[1], &cfg, &device, &a[3])?;
                bit_gate(&arms)?;
                Ok(())
            }
            Some("phased") if a.len() >= 4 => phased(&a[1], &a[2], &a[3], num(4, 64)?),
            Some("dispatch") => dispatch(),
            Some("ppl") if a.len() >= 4 => {
                ppl(&a[1], &a[2], &a[3], num(4, 4096)?, num(5, 4)?, a.get(6).map(String::as_str).unwrap_or("grp"))
            }
            Some("profile") if a.len() >= 4 => profile(&a[1], &a[2], &a[3], num(4, 11)?),
            Some("chat") if a.len() >= 4 => chat(&a[1], &a[2], &a[3], a.get(4).map(String::as_str), num(5, 512)?),
            _ => anyhow::bail!(
                "usage: metalrun time <sealed> <config> <arms> [n_new=128] [timed_rounds=7] [prompt_tokens=0]\n\
                 \x20      metalrun gate <sealed> <config> <arms>\n\
                 \x20      metalrun phased <sealed> <config> <arm> [n_new=64]\n\
                 \x20      metalrun dispatch\n\
                 \x20      metalrun ppl <sealed> <config> <arm> [ctx=4096] [windows=4] [grp|ker]\n\
                 \x20      metalrun chat <sealed> <config> <arm> [temp=0.7] [max_new=512]\n\
                 \x20      metalrun profile <sealed> <config> <arm> [timed_rounds=11]\n\
                 arms: comma list of lut|lutp|lutd|lutdp|lutd2|lutd4[+q4p][+h][+p][+t<cols>][+grp|+ker] or D; the first is the reference"
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
        kernel: bool,
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
            "lutd2" => "tv_tetra48_metal_lutd2",
            "lutd4" => "tv_tetra48_metal_lutd4",
            other => anyhow::bail!("arm {s:?}: {other:?} is not lut, lutp, lutd, lutdp, lutd2, lutd4 or D"),
        };
        let mut spec =
            Spec { tuning: MetalTuning { tetra_kernel: kernel, ..Default::default() }, grouped: false, kernel: false };
        for m in parts {
            match m {
                "q4p" => spec.tuning.q4_prefetch = true,
                "h" => spec.tuning.half_io = true,
                "p" => spec.tuning.prefill_rows = llvq_llm::fused_metal::TETRA_PREFILL_MAX,
                "grp" => spec.grouped = true,
                "ker" => spec.kernel = true,
                t if t.starts_with('t') => {
                    spec.tuning.q4_tile_cols = t[1..].parse().with_context(|| format!("arm {s:?}: tile {t:?}"))?
                }
                other => anyhow::bail!("arm {s:?}: unknown modifier {other:?}"),
            }
        }
        anyhow::ensure!(!(spec.grouped && spec.kernel), "arm {s:?}: +grp and +ker are two forms; name one");
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
                apply_config(cfg, &mut s.model, device)?;
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
                // The config's own pair, store then form, first.
                apply_config(cfg, &mut f.model, device)?;
                if spec.grouped {
                    // A measurement arm: Served::apply refuses this form on
                    // Metal by name, and this binary is not a served door.
                    f.model.set_kv_store(AttnHeads::Grouped.store());
                    f.model.set_attn_heads(AttnHeads::Grouped);
                }
                if spec.kernel {
                    let (n_kv, groups, hd) = f.model.attn_geometry();
                    let kern = llvq_llm::attn::kernel_for(device, n_kv, groups, hd).map_err(anyhow::Error::msg)?;
                    println!("arm {spec_s}: {} compiled and checked against the CPU replay, bit for bit", kern.name());
                    f.model.set_kv_store(AttnHeads::Kernel.store());
                    f.model.set_attn_heads(AttnHeads::Kernel);
                    f.model.set_attn_kernel(Some(kern));
                    f.model.set_kernel_from_for_measurement(0);
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

    /// The config's store and form. A form the Metal door refuses (grouped,
    /// kernel) goes through the measurement door, said so on one line: this
    /// binary times it, nothing serves it.
    fn apply_config(cfg: &Served, model: &mut Qwen3, device: &Device) -> anyhow::Result<()> {
        match cfg.check_device(true) {
            Ok(()) => cfg.apply(model, device).map_err(anyhow::Error::msg),
            Err(why) => {
                println!("NOT SERVED on Metal, measured through Served::apply_measurement: {why}");
                cfg.apply_measurement(model, device).map_err(anyhow::Error::msg)
            }
        }
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

    fn time(path: &str, cfg_path: &str, list: &str, n_new: usize, timed: usize, prompt_tokens: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n_new >= 2 && timed >= 1, "n_new >= 2 and at least one timed round");
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let arms = load_arms(path, &cfg, &device, list)?;
        if arms.iter().filter(|a| a.fused).count() > 1 {
            bit_gate(&arms)?;
        }
        let ids = match prompt_tokens {
            0 => arms[0].tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec(),
            n => {
                let text = llvq_llm::corpus::wikitext2_test()?;
                let head: String = text.chars().take(n * 12).collect();
                let all = arms[0].tokenizer.encode(head.as_str(), false).map_err(anyhow::Error::msg)?;
                anyhow::ensure!(all.get_ids().len() >= n, "{n} prompt tokens asked, {} found", all.get_ids().len());
                let ids = all.get_ids()[..n].to_vec();
                println!(
                    "prompt: the first {n} tokens of the WikiText-2 test split, sha256 {}",
                    llvq_llm::digest::sha256_u32(&ids)
                );
                ids
            }
        };
        let shown = if ids.len() <= 8 { format!("{PROMPT:?} = {ids:?}") } else { format!("{} tokens", ids.len()) };
        println!(
            "prompt {shown}; {n_new} new tokens; {DISCARDED} discarded + {timed} timed rounds; order {}",
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

    // ---- chat: one arm, for a person to try --------------------------------

    fn chat(path: &str, cfg_path: &str, spec: &str, temp: Option<&str>, max_new: usize) -> anyhow::Result<()> {
        let temp: f32 = match temp {
            None => 0.7,
            Some(t) => t.parse().with_context(|| format!("temperature {t:?} is not a number"))?,
        };
        anyhow::ensure!(temp >= 0.0 && max_new >= 1, "temperature >= 0 and max_new >= 1");
        anyhow::ensure!(!spec.contains(','), "chat talks to one arm, {spec:?} names several");
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let arm = load_arm(path, &cfg, &device, spec)?;
        let opts = llvq_llm::repl::ReplOptions {
            sampler: llvq_llm::repl::Sampler { temp, top_p: 0.9, state: 0x5EED_5EED_5EED_5EED },
            max_new,
            think: false,
            show_ids: false,
            system: None,
        };
        llvq_llm::repl::run(&arm.model, &arm.tokenizer, opts, &mut std::io::stdin().lock())
    }

    // ---- profile: the blocks phase by op class, off protocol -----------------

    const CLASSES: [&str; 12] =
        ["norm", "rot", "tetra.q", "tetra.k", "tetra.gate", "tetra.up", "tetra.down", "int4.v", "int4.o", "int4.down", "act", "resid"];

    /// One layer's captured inputs, rotated once, and the outputs the
    /// activation and residual classes consume.
    struct LayerIn {
        h_attn: Tensor,
        ctx: Tensor,
        h_mlp: Tensor,
        gated: Tensor,
        r_qk: llvq_llm::model::Rotated,
        r_gu: llvq_llm::model::Rotated,
        r_down: Option<llvq_llm::model::Rotated>,
        g_out: Tensor,
        u_out: Tensor,
        o_out: Tensor,
    }

    /// Device bytes of one projection, computed from its shape: Tetra48 is 6
    /// bytes a block of 24, 2 a tail weight, 8 a row (scale and tail side);
    /// int4 g128 is half a byte a weight plus a scale and a bias a group.
    fn proj_bytes(p: &llvq_llm::model::Proj) -> anyhow::Result<f64> {
        use llvq_llm::model::Proj;
        Ok(match p {
            Proj::Lattice(d) => (d.d_out() * ((d.d_in() / 24) * 6 + (d.d_in() % 24) * 2 + 8)) as f64,
            Proj::Int4(d) => (d.d_out() * d.d_in() / 2 + d.d_out() * (d.d_in() / 128) * 4) as f64,
            _ => anyhow::bail!("{}: profile reads Lattice and Int4 projections only", p.site_name()),
        })
    }

    fn profile(path: &str, cfg_path: &str, spec: &str, timed: usize) -> anyhow::Result<()> {
        use candle_core::Module;
        use llvq_llm::model::Proj;
        anyhow::ensure!(timed >= 3, "at least three timed rounds");
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let a = load_arm(path, &cfg, &device, spec)?;
        anyhow::ensure!(a.fused, "profile reads a fused arm");
        let m = &a.model;
        let ids = a.tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let reference = m.generate(&ids, 64, &mut NoCapture)?;
        let (toks, rep) = m.generate_phased(&ids, 64)?;
        anyhow::ensure!(toks == reference, "phased tokens differ from generate");

        // One decode step's projection inputs, as the bit gate takes them.
        let mut caches = m.fresh_caches();
        m.hidden_cached(&Tensor::from_slice(&ids, (1, ids.len()), &device)?, 0, &mut caches, &mut NoCapture)?;
        let mut keep = Keep(HashMap::new());
        let step = Tensor::from_slice(&[ids[ids.len() - 1]], (1, 1), &device)?;
        m.hidden_cached(&step, ids.len(), &mut caches, &mut keep)?;
        let mut layers = Vec::new();
        let mut bytes = [0f64; CLASSES.len()];
        for (i, blk) in m.blocks.iter().enumerate() {
            let get = |act: Act| keep.0.get(&(i, act.index())).cloned().with_context(|| format!("layer {i}: {act:?}"));
            let (h_attn, ctx, h_mlp, gated) = (get(Act::Attn)?, get(Act::AttnOut)?, get(Act::Mlp)?, get(Act::MlpOut)?);
            let (_, _, _, q, k, v, o) = blk.attn_parts();
            let (_, _, act, gate, up, down) = blk.mlp_parts();
            for (c, p) in [(2, q), (3, k), (4, gate), (5, up), (7, v), (8, o)] {
                bytes[c] += proj_bytes(p)?;
            }
            let r_down = match down {
                Proj::Lattice(_) => {
                    bytes[6] += proj_bytes(down)?;
                    Some(down.prepare(&gated)?)
                }
                _ => {
                    bytes[9] += proj_bytes(down)?;
                    None
                }
            };
            anyhow::ensure!(matches!(v, Proj::Int4(_)) && matches!(o, Proj::Int4(_)), "layer {i}: v and o are not int4");
            let (r_qk, r_gu) = (q.prepare(&h_attn)?, gate.prepare(&h_mlp)?);
            let g_out = gate.forward_with(&r_gu, &h_mlp)?;
            let u_out = up.forward_with(&r_gu, &h_mlp)?;
            let o_out = o.forward(&ctx)?;
            let _ = act;
            layers.push(LayerIn { h_attn, ctx, h_mlp, gated, r_qk, r_gu, r_down, g_out, u_out, o_out });
        }
        let eps = m.blocks[0].mlp_parts().1 as f32;

        // The per-layer ops of the 36 blocks, in model order, `on` naming
        // the classes that run; closed by one synchronize.
        let pass = |on: &[bool; CLASSES.len()]| -> anyhow::Result<()> {
            let mut sink = Vec::with_capacity(16);
            let mut rsink = Vec::with_capacity(8);
            for (blk, l) in m.blocks.iter().zip(&layers) {
                let (ln1_w, _, _, q, k, v, o) = blk.attn_parts();
                let (ln2_w, _, act, gate, up, down) = blk.mlp_parts();
                if on[0] {
                    sink.push(candle_nn::ops::rms_norm(&l.h_attn, ln1_w, eps)?);
                }
                if on[1] {
                    rsink.push(q.prepare(&l.h_attn)?);
                }
                if on[2] {
                    sink.push(q.forward_with(&l.r_qk, &l.h_attn)?);
                }
                if on[3] {
                    sink.push(k.forward_with(&l.r_qk, &l.h_attn)?);
                }
                if on[7] {
                    sink.push(v.forward(&l.h_attn)?);
                }
                if on[8] {
                    sink.push(o.forward(&l.ctx)?);
                }
                if on[11] {
                    sink.push((&l.h_attn + &l.o_out)?);
                }
                if on[0] {
                    sink.push(candle_nn::ops::rms_norm(&l.h_attn, ln2_w, eps)?);
                }
                if on[1] {
                    rsink.push(gate.prepare(&l.h_mlp)?);
                }
                if on[4] {
                    sink.push(gate.forward_with(&l.r_gu, &l.h_mlp)?);
                }
                if on[5] {
                    sink.push(up.forward_with(&l.r_gu, &l.h_mlp)?);
                }
                if on[10] {
                    sink.push(act.forward(&l.g_out)?.mul(&l.u_out)?);
                }
                match &l.r_down {
                    Some(rd) => {
                        if on[1] {
                            rsink.push(down.prepare(&l.gated)?);
                        }
                        if on[6] {
                            sink.push(down.forward_with(rd, &l.gated)?);
                        }
                    }
                    None => {
                        if on[9] {
                            sink.push(down.forward(&l.gated)?);
                        }
                    }
                }
                if on[11] {
                    sink.push((&l.h_attn + &l.o_out)?);
                }
                // Every config frees its outputs at the same point, the end
                // of each layer, so the buffer pool sees the same churn.
                sink.clear();
                rsink.clear();
            }
            drop((sink, rsink));
            Ok(())
        };
        // One timed reading: an untimed full pass first, so the GPU clock and
        // the buffer pool are the same whichever config ran before, then
        // REPS passes of the config, closed by one synchronize, per pass.
        const REPS: usize = 3;
        let full = [true; CLASSES.len()];
        let run = |on: &[bool; CLASSES.len()]| -> anyhow::Result<f64> {
            pass(&full)?;
            device.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..REPS {
                pass(on)?;
            }
            device.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64)
        };

        let n = CLASSES.len();
        let mut configs: Vec<(String, [bool; CLASSES.len()])> = vec![("all".into(), [true; CLASSES.len()]), ("none".into(), [false; CLASSES.len()])];
        for c in 0..n {
            let mut without = [true; CLASSES.len()];
            without[c] = false;
            configs.push((format!("-{}", CLASSES[c]), without));
            let mut only = [false; CLASSES.len()];
            only[c] = true;
            configs.push((format!("+{}", CLASSES[c]), only));
        }
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
        let nc = configs.len();
        for r in 0..DISCARDED + timed {
            // The order turns by one config a round, so no config always
            // follows the same one.
            for jj in 0..nc {
                let j = (jj + r) % nc;
                let on = &configs[j].1;
                let ms = run(on)?;
                if r >= DISCARDED {
                    t[j].push(ms);
                }
            }
        }
        let idx = |name: &str| configs.iter().position(|(c, _)| c == name).expect("a config");
        let diff = |a: usize, b: usize| -> (f64, f64, f64) {
            stats(&t[a].iter().zip(&t[b]).map(|(p, q)| p - q).collect::<Vec<_>>())
        };
        let (blocks, blo, bhi) = stats(&rep.blocks_ms);
        let (all, none) = (idx("all"), idx("none"));
        let (seq, slo, shi) = diff(all, none);
        println!("OFF PROTOCOL: each reading a warm full pass, then {REPS} passes closed by one synchronize; order turned each round; never a rate");
        println!("phase blocks (generate_phased, 64 tokens):    {blocks:7.3} ms [{blo:.3}-{bhi:.3}]");
        println!("sequence, every class, minus the empty one:   {seq:7.3} ms [{slo:.3}-{shi:.3}]");
        let (e, elo, ehi) = stats(&t[none]);
        println!("empty sequence (the synchronize alone):       {e:7.3} ms [{elo:.3}-{ehi:.3}]");
        println!("left out of the sequence (attention, RoPE, q/k norms, cache, reshapes): {:.3} ms (computed)", blocks - seq);
        println!("class        marginal ms [range]        alone ms [range]      share of blocks | bytes a token | GB/s alone (computed)");
        let mut marg_sum = 0.0;
        for (c, name) in CLASSES.iter().enumerate() {
            let (mm, mlo, mhi) = diff(all, idx(&format!("-{name}")));
            let (am, alo, ahi) = diff(idx(&format!("+{name}")), none);
            marg_sum += mm;
            let gbs = match bytes[c] {
                b if b > 0.0 => format!("{:9.1} MB | {:6.1}", b / 1e6, b / (am * 1e-3) / 1e9),
                _ => "".to_string(),
            };
            println!(
                "{name:<11} {mm:7.3} [{mlo:6.3}-{mhi:6.3}]   {am:7.3} [{alo:6.3}-{ahi:6.3}]   {:5.1} %   {gbs}",
                100.0 * mm / blocks
            );
        }
        println!("sum of marginals {marg_sum:.3} ms against the full sequence {seq:.3} ms");
        let tetra: f64 = (2..=6).map(|c| bytes[c]).sum();
        let int4: f64 = (7..=9).map(|c| bytes[c]).sum();
        println!("weights a token: Tetra {:.1} MB, int4 {:.1} MB (computed from the shapes)", tetra / 1e6, int4 / 1e6);
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

    // ---- ppl: the grouped form's tolerance on Metal, a reading ---------------

    /// WikiText-2 test windows of `ctx` tokens, each fed one true token at a
    /// time through two caches of the same model in lockstep, expanded and
    /// grouped (`Qwen3::decode_nll_paired`). The decode path only: Metal
    /// prefills 256 rows a call, so the prefill path of the L40S reading
    /// (`gqa-ppl-2026-11-01`) cannot score a window of 4,096 here.
    /// `grp`: grouped against expanded. `ker`: the Metal attention kernel from
    /// position 0 against the grouped form, the T8 pairing of the card.
    fn ppl(path: &str, cfg_path: &str, spec: &str, ctx: usize, windows: usize, pair: &str) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path)?;
        let mut a = load_arm(path, &cfg, &device, spec)?;
        anyhow::ensure!(a.fused && a.model.attn_heads() == AttnHeads::Expanded, "ppl takes a fused expanded arm");
        let (arms, names) = match pair {
            "grp" => (
                [(AttnHeads::Expanded.store(), AttnHeads::Expanded), (AttnHeads::Grouped.store(), AttnHeads::Grouped)],
                ["expanded", "grouped"],
            ),
            "ker" => {
                let (n_kv, groups, hd) = a.model.attn_geometry();
                let kern = llvq_llm::attn::kernel_for(&device, n_kv, groups, hd).map_err(anyhow::Error::msg)?;
                println!("{} compiled and checked against the CPU replay, bit for bit; kernel from position 0", kern.name());
                a.model.set_attn_kernel(Some(kern));
                a.model.set_kernel_from_for_measurement(0);
                (
                    [(AttnHeads::Grouped.store(), AttnHeads::Grouped), (AttnHeads::Kernel.store(), AttnHeads::Kernel)],
                    ["grouped", "kernel"],
                )
            }
            other => anyhow::bail!("ppl pair {other:?}: grp or ker"),
        };
        let text = llvq_llm::corpus::wikitext2_test()?;
        let ids = a.tokenizer.encode(text.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let nwin = (ids.len() / ctx).min(windows);
        anyhow::ensure!(nwin >= 2 && ctx >= 2, "two windows of two tokens at least: {} tokens", ids.len());
        let span = &ids[..nwin * ctx];
        println!(
            "ppl: wikitext2 test, {nwin} windows of {ctx} from the start ({} tokens in the split, span sha256 {}), \
             {} against {}, decode path in lockstep; a TOLERANCE reading, never a served decision",
            ids.len(),
            llvq_llm::digest::sha256_u32(span),
            names[1],
            names[0]
        );
        let (mut ratios, mut nll, mut count, mut agree, mut max_d) = (Vec::new(), [0f64; 2], 0usize, 0usize, 0f32);
        let t0 = Instant::now();
        for w in 0..nwin {
            let win = &span[w * ctx..(w + 1) * ctx];
            let d = a.model.decode_nll_paired(win, arms)?;
            let (pe, pg) = ((d.nll[0] / d.count as f64).exp(), (d.nll[1] / d.count as f64).exp());
            ratios.push(pg / pe);
            nll[0] += d.nll[0];
            nll[1] += d.nll[1];
            count += d.count;
            agree += d.top1_agree;
            max_d = max_d.max(d.max_dlogit);
            println!(
                "  window {:>2}: ppl {} {pe:.6} {} {pg:.6}, {} / {} - 1 = {:+.4} %, top-1 \
                 agreement {}/{}, max |dlogit| {:.3e}, {:.0} s elapsed",
                w + 1,
                names[0],
                names[1],
                names[1],
                names[0],
                (pg / pe - 1.0) * 100.0,
                d.top1_agree,
                d.count,
                d.max_dlogit,
                t0.elapsed().as_secs_f64()
            );
        }
        let (pe, pg) = ((nll[0] / count as f64).exp(), (nll[1] / count as f64).exp());
        let (m, lo, hi) = stats(&ratios);
        println!(
            "ppl over {nwin} windows: {} {pe:.6}, {} {pg:.6}, pooled {} / {} - 1 = {:+.4} %; \
             per window median {:+.4} % [{:+.4} ; {:+.4}]; top-1 agreement {agree}/{count} = {:.3} %; max |dlogit| {max_d:.3e}",
            names[0],
            names[1],
            names[1],
            names[0],
            (pg / pe - 1.0) * 100.0,
            (m - 1.0) * 100.0,
            (lo - 1.0) * 100.0,
            (hi - 1.0) * 100.0,
            100.0 * agree as f64 / count as f64
        );
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
