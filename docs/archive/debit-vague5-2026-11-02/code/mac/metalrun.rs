// DRAFT, never compiled or run. Wave 5 deep dive, lead "mac", 2026-11-02, revised the same day.
//
// Proposed as `llvq-llm/src/bin/metalrun.rs`: the Mac's decode bench, the
// harness of prereg-mac-1 and prereg-mac-2. `fusedrun` is CUDA-only
// (fusedrun.rs:649-653) and `chat` disclaims its own tok/s (chat.rs:36-38), so
// no Metal number has a protocol today. Written against the API proposed in
// code/mac/fused_metal_wall.rs (`load_resolved_tuned`, `MetalTuning`); every
// other call is an existing public item, with the two objc2 calls marked.
//
// Built once, then run by path, never through `cargo run`:
//   nice -n 15 cargo build --release -p llvq-llm --features metal --bin metalrun -j 4
//   (30 minutes, quiet machine)
//   nice -n 10 target/release/metalrun check <sealed> <config>
//   nice -n 10 target/release/metalrun time  <sealed> <config> 128 7
//   nice -n 10 target/release/metalrun null  <sealed> <config> 128 7      (off protocol)
//
// Protocol, the one the preregs stamp:
//   - the environment is read once, first: every LLVQ_*, CANDLE_*, MTL_* and
//     METAL_* variable is refused, the allowlist is empty, and the check is
//     printed. CANDLE_METAL_ENABLE_FAST_MATH changes every candle kernel and
//     CANDLE_METAL_COMPUTE_PER_BUFFER moved the q5 step 33.9 -> 27.7 tok/s;
//   - `check`: the record gate (every projection and the head against its host
//     oracle, bit for bit) and the logits gate G-L (F against D on a 256-token
//     WikiText-2 window);
//   - `time`: arms F (tile 2048), F' (tile 8192), Fh (f16 head), D (dense) in
//     ONE process, each through `Served::apply`; the window gate F = F'; two
//     discarded rounds then `timed`, the arms in the SAME order every round
//     (METHODE section 2); tok/s = n_new / t over a whole `generate`; ratios
//     round by round, raw F/D and same-head Fh/D together; then F's 256-token
//     line and the fenced phased pass;
//   - GB/s = weights a token (projections + head table) x tok/s, never the
//     device bytes;
//   - memory: the loader's host count AND MTLDevice.currentAllocatedSize;
//   - load samples outside every timed window: the GPU's "Device Utilization
//     %", the other processes' summed %CPU, `pmset -g therm`.
//
// It stays out of the CUDA image (no Dockerfile COPY, no UPLOAD_ALLOW entry),
// but it must still build as a stub on Linux, like fusedrun, so `cargo build`
// and the Linux type-check of METHODE section 6 stay green.

const PROMPT: &str = "The capital of France is";
const DISCARDED: usize = 2;
const WINDOW: usize = 256; // = model::MAX_ROWS, the per-row path's cap on Metal
const REFUSED_PREFIXES: [&str; 4] = ["LLVQ_", "CANDLE_", "MTL_", "METAL_"];
const ALLOWED: [&str; 0] = [];

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
    use super::*;
    use anyhow::Context;
    use candle_core::{DType, Device, IndexOp, Tensor, D};
    use llvq_llm::fused::EmbedMode;
    use llvq_llm::fused_metal::{load_resolved_tuned, MetalTuning, Q4_TILE_COLS};
    use llvq_llm::model::{Act, Capture, NoCapture, Proj, Qwen3};
    use llvq_llm::served::Served;
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::Instant;

    pub fn main() -> anyhow::Result<()> {
        // Before anything else reads a variable, the library included.
        environment_check()?;
        let a: Vec<String> = std::env::args().skip(1).collect();
        match (a.first().map(String::as_str), a.len()) {
            (Some("check"), 3) => check(&a[1], &a[2]),
            (Some("time"), 5) => time(&a[1], &a[2], a[3].parse()?, a[4].parse()?),
            (Some("null"), 5) => null(&a[1], &a[2], a[3].parse()?, a[4].parse()?),
            _ => anyhow::bail!(
                "usage: metalrun check <sealed> <config> | time <sealed> <config> <n_new> <timed_rounds> \
                 | null <sealed> <config> <n_new> <timed_rounds>"
            ),
        }
    }

    // ---- the outside world, read once ---------------------------------------

    fn environment_check() -> anyhow::Result<()> {
        let mut seen = Vec::new();
        for (k, v) in std::env::vars_os() {
            let k = k.to_string_lossy().into_owned();
            if REFUSED_PREFIXES.iter().any(|p| k.starts_with(p)) && !ALLOWED.contains(&k.as_str()) {
                seen.push(format!("{k}={:?}", v));
            }
        }
        anyhow::ensure!(
            seen.is_empty(),
            "metalrun names its arms itself and reads no knob; unset {}",
            seen.join(", ")
        );
        println!(
            "environment: no LLVQ_*, CANDLE_*, MTL_* or METAL_* variable (allowlist empty); RAYON_NUM_THREADS={:?}",
            std::env::var("RAYON_NUM_THREADS").ok()
        );
        Ok(())
    }

    fn header(path: &str, cfg_path: &str, device: &Device) -> anyhow::Result<Served> {
        let exe = std::env::current_exe()?;
        println!("binary  {} sha256 {}", exe.display(), llvq_llm::digest::sha256_file(&exe)?);
        println!("object  {path} sha256 {}", llvq_llm::digest::sha256_file(Path::new(path))?);
        println!("config  {cfg_path} sha256 {}", llvq_llm::digest::sha256_file(Path::new(cfg_path))?);
        let cfg = Served::read(Path::new(cfg_path)).map_err(anyhow::Error::msg)?;
        println!("{}", cfg.provenance());
        // The Metal door refuses `grouped` by name (served.rs:275-289); the Mac
        // runs the expanded config, the default of ROADMAP section 5.
        cfg.check_device(true).map_err(anyhow::Error::msg)?;
        println!("working set {:.2} GB (recommendedMaxWorkingSetSize)", working_set(device) as f64 / 1e9);
        println!("pmset -g therm: {}", command("pmset", &["-g", "therm"]).replace('\n', " | "));
        Ok(cfg)
    }

    fn command(prog: &str, args: &[&str]) -> String {
        std::process::Command::new(prog)
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|e| format!("{prog} failed: {e}"))
    }

    /// The other processes' summed %CPU (a decaying average in `ps`), this one excluded.
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

    /// The GPU's "Device Utilization %" from the IOAccelerator entry, no privilege needed.
    fn gpu_utilization() -> Option<u32> {
        let out = command("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"]);
        out.split("\"Device Utilization %\"=").nth(1)?.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    }

    // API TO CONFIRM AT COMPILE: candle's MetalDevice -> its wrapper Device ->
    // the objc2 protocol object, whose two getters are safe calls in
    // objc2-metal 0.3.2 (generated/MTLDevice.rs:722 and :832).
    fn mtl(device: &Device) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice> {
        match device {
            Device::Metal(m) => m.device().as_ref(),
            _ => unreachable!("Device::new_metal above"),
        }
    }
    fn allocated(device: &Device) -> u64 {
        use objc2_metal::MTLDevice;
        mtl(device).currentAllocatedSize() as u64
    }
    fn working_set(device: &Device) -> u64 {
        use objc2_metal::MTLDevice;
        mtl(device).recommendedMaxWorkingSetSize()
    }

    // ---- arms ----------------------------------------------------------------

    struct Arm {
        name: &'static str,
        model: Qwen3,
        tokenizer: tokenizers::Tokenizer,
        /// The loader's host count, the card's accounting (computed).
        host_bytes: f64,
        /// MTLDevice.currentAllocatedSize across this arm's load (measured).
        alloc_bytes: f64,
        /// Projections plus the head table: what a decode step reads.
        weights_a_token: f64,
        load_s: f64,
    }

    /// The head table's bytes: f16, or int bits plus an f16 scale and bias per 64.
    fn head_bytes(c: &candle_transformers::models::qwen3::Config, mode: EmbedMode) -> f64 {
        let (v, d) = (c.vocab_size as f64, c.hidden_size as f64);
        match mode.bits() {
            None => v * d * 2.0,
            Some(b) => v * d * b as f64 / 8.0 + v * (d / 64.0) * 4.0,
        }
    }

    /// Every weight of the dense model is read once a token; the head reads one table.
    fn dense_weights_a_token(c: &candle_transformers::models::qwen3::Config) -> f64 {
        let (d, i, l) = (c.hidden_size, c.intermediate_size, c.num_hidden_layers);
        let (q, kv) = (c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim);
        2.0 * (l * (d * q + 2 * d * kv + q * d + 3 * d * i) + c.vocab_size * d) as f64
    }

    fn fused_arm(
        name: &'static str,
        path: &str,
        cfg: &Served,
        device: &Device,
        embed: EmbedMode,
        tuning: MetalTuning,
    ) -> anyhow::Result<Arm> {
        let before = allocated(device);
        let t0 = Instant::now();
        let mut f = load_resolved_tuned(
            path, device, DType::F16, cfg.layout, embed, cfg.rot_share, cfg.fuse, cfg.kv, Some("metalrun"), tuning,
        )?;
        cfg.apply(&mut f.model, device).map_err(anyhow::Error::msg)?;
        let load_s = t0.elapsed().as_secs_f64();
        Ok(Arm {
            name,
            weights_a_token: f.runtime_bytes as f64 + head_bytes(&f.config, embed),
            host_bytes: (f.runtime_bytes + f.carried_bytes) as f64,
            alloc_bytes: allocated(device).saturating_sub(before) as f64,
            model: f.model,
            tokenizer: f.tokenizer,
            load_s,
        })
    }

    fn dense_arm(path: &str, cfg: &Served, device: &Device) -> anyhow::Result<Arm> {
        let before = allocated(device);
        let t0 = Instant::now();
        let mut s = llvq_llm::sealed::load(path, DType::F16, device, cfg.kv)?;
        // The same attention and store as the fused arms, so the arms differ
        // in their projections and head only.
        cfg.apply(&mut s.model, device).map_err(anyhow::Error::msg)?;
        let load_s = t0.elapsed().as_secs_f64();
        Ok(Arm {
            name: "D",
            weights_a_token: dense_weights_a_token(&s.config),
            host_bytes: 2.0 * (s.quantized_weights + s.carried_weights) as f64,
            alloc_bytes: allocated(device).saturating_sub(before) as f64,
            model: s.model,
            tokenizer: s.tokenizer,
            load_s,
        })
    }

    fn print_arm(a: &Arm, params: f64) {
        println!(
            "arm {:<3} loaded in {:5.1} s | host count {:.3} GB = {:.3} b/param (computed) | device +{:.3} GB = {:.3} b/param \
             (measured) | weights a token {:.3} GB | attention {} on {}",
            a.name, a.load_s, a.host_bytes / 1e9, a.host_bytes * 8.0 / params, a.alloc_bytes / 1e9,
            a.alloc_bytes * 8.0 / params, a.weights_a_token / 1e9,
            a.model.attn_heads().label(), a.model.kv_store().label()
        );
    }

    // ---- the window ----------------------------------------------------------

    /// The first WINDOW + 1 tokens of the WikiText-2 test split, no special
    /// tokens: WINDOW for the prefill, one for the teacher-forced decode step.
    fn window_ids(tok: &tokenizers::Tokenizer) -> anyhow::Result<Vec<u32>> {
        let text = llvq_llm::corpus::wikitext2_test()?;
        let head: String = text.chars().take(12_000).collect(); // far past 257 tokens
        let ids = tok.encode(head.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        anyhow::ensure!(ids.len() > WINDOW, "the window needs {} tokens, got {}", WINDOW + 1, ids.len());
        Ok(ids[..=WINDOW].to_vec())
    }

    /// Logits at every position of the window, f32, `[WINDOW][vocab]`.
    fn window_logits(m: &Qwen3, ids: &[u32]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut caches = m.fresh_caches();
        let input = Tensor::from_slice(&ids[..WINDOW], (1, WINDOW), m.device())?;
        let h = m.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        Ok(m.project_head(&h)?.to_dtype(DType::F32)?.i(0)?.to_vec2::<f32>()?)
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
    }

    fn nll(logits: &[Vec<f32>], ids: &[u32]) -> f64 {
        (0..WINDOW - 1)
            .map(|p| {
                let row = &logits[p];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = m + row.iter().map(|&x| ((x as f64) - m).exp()).sum::<f64>().ln();
                lse - row[ids[p + 1] as usize] as f64
            })
            .sum()
    }

    // ---- check: the record gate and G-L ---------------------------------------

    /// Keeps each projection input of the step it watches (model.rs:85-88).
    struct Keep(HashMap<(usize, Act), Tensor>);
    impl Capture for Keep {
        fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
            self.0.insert((layer, act), x.clone());
            Ok(())
        }
    }

    // copy from llvq-metal/tests/q4_matches_host.rs::reference (the lane-order
    // reference: 32 lanes striding words, mul_add, shuffle-xor butterfly).
    fn lane_order_reference(_m: &llvq_artifact::Int4Matrix, _x: &[f32]) -> Vec<f32> {
        unimplemented!("copy from q4_matches_host.rs::reference")
    }
    // copy from llvq-metal/tests/q4e_matches_host.rs::head_reference, fed the
    // int4 g64 head table that `fused::load_with` carries in the file's encoding.
    fn head_reference(_fm: &llvq_llm::fused::FusedModel, _x16: &[u16]) -> anyhow::Result<Vec<f32>> {
        unimplemented!("copy from q4e_matches_host.rs::head_reference")
    }

    fn f16_bits(t: &Tensor) -> anyhow::Result<Vec<u16>> {
        Ok(t.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>()?.iter().map(|v| v.to_bits()).collect())
    }

    fn check(path: &str, cfg_path: &str) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path, &device)?;
        let tuning = MetalTuning { q4_tile_cols: Q4_TILE_COLS, keep_host: true, null_kernels: false };
        let f = fused_arm("F", path, &cfg, &device, cfg.embed, tuning)?;
        let ids = window_ids(&f.tokenizer)?;

        // The record gate. Prefill the window, then one teacher-forced decode
        // step with the capture: the inputs every projection sees in decode.
        let mut caches = f.model.fresh_caches();
        let input = Tensor::from_slice(&ids[..WINDOW], (1, WINDOW), &device)?;
        f.model.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let mut keep = Keep(HashMap::new());
        let step = Tensor::from_slice(&ids[WINDOW..], (1, 1), &device)?;
        let h_last = f.model.hidden_cached(&step, WINDOW, &mut caches, &mut keep)?;

        let host = llvq_llm::fused::load_with(path, cfg.layout, cfg.fuse).map_err(anyhow::Error::msg)?;
        let int4: HashMap<&str, llvq_artifact::Int4Matrix> = host
            .int4
            .iter()
            .map(|q| {
                (q.name.as_str(), llvq_artifact::Int4Matrix {
                    name: q.name.clone(), d_out: q.d_out, d_in: q.d_in, bits: 4, group: q.group,
                    packed: q.packed.clone(), scales: q.scales.clone(), biases: q.biases.clone(),
                })
            })
            .collect();
        let (mut compared, mut equal, mut first_bad) = (0usize, 0usize, None::<String>);
        for (i, blk) in f.model.blocks.iter().enumerate() {
            let sites = [
                ("q_proj", &blk.q_proj, Act::Attn), ("k_proj", &blk.k_proj, Act::Attn),
                ("v_proj", &blk.v_proj, Act::Attn), ("o_proj", &blk.o_proj, Act::AttnOut),
                ("gate_proj", &blk.gate_proj, Act::Mlp), ("up_proj", &blk.up_proj, Act::Mlp),
                ("down_proj", &blk.down_proj, Act::MlpOut),
            ];
            for (pname, proj, act) in sites {
                let x = keep.0.get(&(i, act)).with_context(|| format!("layer {i}: no {act:?} captured"))?;
                let x = x.reshape((1, x.dim(D::Minus1)?))?;
                let (got, want) = match proj {
                    // Both arms of the same op: metal_fwd on the device, cpu_fwd on
                    // the host copies, each narrowed to f16 at the adapter
                    // (fused_metal.rs:1332-1343).
                    Proj::Lattice(p) => {
                        let xr = p.prepare(&x)?;
                        let dev = p.matvec(&xr, &[1, p.d_out()])?;
                        let cpu = p.matvec(&xr.to_device(&Device::Cpu)?, &[1, p.d_out()])?;
                        (f16_bits(&dev)?, f16_bits(&cpu)?)
                    }
                    Proj::Int4(p) => {
                        let dev = p.matvec(&x, &[1, p.d_out()])?;
                        let xf: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                        let m = int4.get(p.name()).with_context(|| format!("{} not in the file", p.name()))?;
                        let want = lane_order_reference(m, &xf).iter().map(|v| half::f16::from_f32(*v).to_bits()).collect();
                        (f16_bits(&dev)?, want)
                    }
                    _ => anyhow::bail!("layer {i} {pname}: neither lattice nor int4 under FUSE=0"),
                };
                compared += 1;
                match got.iter().zip(&want).position(|(a, b)| a != b) {
                    None => equal += 1,
                    Some(r) => {
                        first_bad.get_or_insert(format!("layer {i} {pname}, row {r}: {:#06x} against {:#06x}", got[r], want[r]));
                    }
                }
            }
        }
        // The head, at f32: the q4 kernel's logits against its host reference.
        let x16 = f16_bits(&h_last)?;
        let dev_logits: Vec<f32> = f.model.project_head(&h_last)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let want_logits = head_reference(&host, &x16)?;
        compared += 1;
        match dev_logits.iter().zip(&want_logits).position(|(a, b)| a.to_bits() != b.to_bits()) {
            None => equal += 1,
            Some(r) => {
                first_bad.get_or_insert(format!("head, row {r}: {} against {}", dev_logits[r], want_logits[r]));
            }
        }
        println!("record gate: {equal} of {compared} equal to their host oracles, bit for bit");
        anyhow::ensure!(equal == compared, "RECORD GATE FAILS at {}", first_bad.unwrap_or_default());

        // G-L: F against D on the window. D is loaded, used once, released.
        let lf = window_logits(&f.model, &ids)?;
        drop(caches);
        let d = dense_arm(path, &cfg, &device)?;
        let ld = window_logits(&d.model, &ids)?;
        let agree = (0..WINDOW).filter(|&p| argmax(&lf[p]) == argmax(&ld[p])).count();
        let worst = lf.iter().flatten().zip(ld.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let scale = ld.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        let (nf, nd) = (nll(&lf, &ids), nll(&ld, &ids));
        println!(
            "G-L: {agree} of {WINDOW} positions share the top token; max |dlogit| {worst:.3e} on a scale of {scale:.3e}; \
             NLL F {nf:.4} D {nd:.4}, F/D - 1 = {:+.3} %",
            (nf / nd - 1.0) * 100.0
        );
        anyhow::ensure!(agree >= 248, "G-L FAILS: {agree} of {WINDOW} under the 248 the prereg requires");
        Ok(())
    }

    // ---- time: the four arms ----------------------------------------------------

    fn time(path: &str, cfg_path: &str, n_new: usize, timed: usize) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path, &device)?;
        println!("load sample before the first load: GPU {:?} %, others {:.0} % CPU", gpu_utilization(), others_cpu());

        let tile = |c| MetalTuning { q4_tile_cols: c, ..Default::default() };
        let mut arms = vec![
            fused_arm("F", path, &cfg, &device, cfg.embed, tile(Q4_TILE_COLS))?,
            fused_arm("F'", path, &cfg, &device, cfg.embed, tile(8192))?,
            fused_arm("Fh", path, &cfg, &device, EmbedMode::F16, tile(Q4_TILE_COLS))?,
        ];
        arms.push(dense_arm(path, &cfg, &device)?);
        let params = count_params(&arms[3]); // 4,022,468,096 at 4B (docs/fiche-4b.md:66)
        for a in &arms {
            print_arm(a, params);
        }

        // Gate 7, first half: the two tiles give the same window logits.
        let tok = arms[0].tokenizer.clone();
        let win = window_ids(&tok)?;
        let (l0, l1) = (window_logits(&arms[0].model, &win)?, window_logits(&arms[1].model, &win)?);
        let same = l0.iter().flatten().zip(l1.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits());
        anyhow::ensure!(same, "GATE 7 FAILS: F and F' differ on the window. The tiled path is not exact as implemented");
        println!("gate 7: F and F' window logits equal, every f32 at {WINDOW} positions");

        let ids = tok.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        println!("prompt {PROMPT:?} = {ids:?}, {n_new} new tokens, {DISCARDED} discarded + {timed} timed rounds, order F F' Fh D");
        let n = arms.len();
        let mut rates = vec![Vec::new(); n];
        let mut texts: Vec<Option<Vec<u32>>> = vec![None; n];
        let mut cpu_max = 0f64;
        for r in 0..DISCARDED + timed {
            for i in 0..n {
                let t0 = Instant::now();
                let out = arms[i].model.generate(&ids, n_new, &mut NoCapture)?;
                let t = t0.elapsed().as_secs_f64();
                // Outside the timed window.
                let cpu = others_cpu();
                cpu_max = cpu_max.max(cpu);
                match &texts[i] {
                    None => texts[i] = Some(out),
                    Some(first) => anyhow::ensure!(first == &out, "GATE 8 FAILS: arm {} changed its text in round {r}", arms[i].name),
                }
                if r >= DISCARDED {
                    rates[i].push(n_new as f64 / t);
                }
                println!("round {r} arm {:<3} {:7.2} tok/s, others {:.0} % CPU", arms[i].name, n_new as f64 / t, cpu);
            }
            if r + 1 == DISCARDED {
                println!("device allocation after the discarded rounds: {:.3} GB (all arms)", allocated(&device) as f64 / 1e9);
            }
        }
        anyhow::ensure!(texts[0] == texts[1], "GATE 7 FAILS: F and F' generated different tokens");

        let stats = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            (s[s.len() / 2], s[0], s[s.len() - 1])
        };
        for (i, a) in arms.iter().enumerate() {
            let (m, lo, hi) = stats(&rates[i]);
            println!(
                "arm {:<3} {m:7.2} tok/s [{lo:.2}-{hi:.2}], niced on a quiet machine | {:.2} ms a token | {:.1} GB/s of weights (computed)",
                a.name, 1e3 / m, m * a.weights_a_token / 1e9
            );
        }
        for (label, x, y) in [("raw F/D", 0usize, 3usize), ("same-head Fh/D", 2, 3), ("tile F'/F", 1, 0)] {
            let ratios: Vec<f64> = rates[x].iter().zip(&rates[y]).map(|(p, q)| p / q).collect();
            let (m, lo, hi) = stats(&ratios);
            println!("{label:<16} x{m:.3} [{lo:.3}-{hi:.3}], formed round by round");
        }
        println!("others' CPU, worst sample between arms: {cpu_max:.0} %");
        let first_div = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
        let (tf, td, tfh) = (texts[0].as_ref().unwrap(), texts[3].as_ref().unwrap(), texts[2].as_ref().unwrap());
        for (label, other) in [("F vs D ", td), ("F vs Fh", tfh)] {
            println!("{label}: {}", first_div(tf, other).map_or("identical through the run".into(), |p| format!("first divergence at token {}", p + 1)));
        }
        println!("first round, F vs its last timed round: {:.2} -> {:.2} tok/s", rates[0][0], rates[0][timed - 1]);

        // Step 4, untimed: the line the card hashes (`fusedrun.rs:1781-1786` format).
        let t256 = arms[0].model.generate(&ids, 256, &mut NoCapture)?;
        println!("  fused F0: {}", tok.decode(&t256, true).map_err(anyhow::Error::msg)?);

        // Step 5, off protocol: fenced phases. Never set beside the rates above.
        for i in [0usize, 2, 3] {
            let (toks, rep) = arms[i].model.generate_phased(&ids, 32)?;
            anyhow::ensure!(Some(&toks[..]) == texts[i].as_ref().map(|t| &t[..32]), "phased tokens differ from generate");
            let med = |v: &[f64]| stats(v).0;
            println!(
                "phased {:<3} (fenced, never a rate): embed {:.3} | blocks {:.3} | head {:.3} | argmax {:.3} ms",
                arms[i].name, med(&rep.embed_ms), med(&rep.blocks_ms), med(&rep.head_ms), med(&rep.rest_ms)
            );
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
        println!("load sample after the run: GPU {:?} %, others {:.0} % CPU", gpu_utilization(), others_cpu());
        println!("pmset -g therm: {}", command("pmset", &["-g", "therm"]).replace('\n', " | "));
        Ok(())
    }

    /// The dense model's parameter count, from the host count of its f16 weights.
    fn count_params(d: &Arm) -> f64 {
        d.host_bytes / 2.0
    }

    // ---- null: the floor, off protocol ---------------------------------------

    fn null(path: &str, cfg_path: &str, n_new: usize, timed: usize) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let cfg = header(path, cfg_path, &device)?;
        println!("OFF PROTOCOL: arm N reads no weight byte and decodes token 0 at every step; its time is a floor, never a rate");
        let arms = [
            fused_arm("F", path, &cfg, &device, cfg.embed, MetalTuning::default())?,
            fused_arm("N", path, &cfg, &device, cfg.embed, MetalTuning { null_kernels: true, ..Default::default() })?,
        ];
        let ids = arms[0].tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let mut ms = [Vec::new(), Vec::new()];
        for r in 0..DISCARDED + timed {
            for (i, a) in arms.iter().enumerate() {
                let t0 = Instant::now();
                a.model.generate(&ids, n_new, &mut NoCapture)?;
                if r >= DISCARDED {
                    ms[i].push(t0.elapsed().as_secs_f64() * 1e3 / n_new as f64);
                }
            }
        }
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            s[s.len() / 2]
        };
        let (f, n) = (med(&ms[0]), med(&ms[1]));
        let share: Vec<f64> = ms[1].iter().zip(&ms[0]).map(|(n, f)| n / f).collect();
        println!("F {f:.2} ms a token, N {n:.2} ms; N/F {:.3} (median of per-round quotients); F - N {:.2} ms", med(&share), f - n);
        Ok(())
    }
}
