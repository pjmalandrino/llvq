//! A conversational REPL over a sealed `.llvq`, and nothing else on disk.
//!
//! ```text
//! cargo run --release -p llvq-llm --features cuda --bin chat -- model.llvq cuda
//! cargo run --release -p llvq-llm --features metal --bin chat -- model.llvq metal
//! ```
//!
//! ## Why this is not `bin/run` with a loop around it
//!
//! `run` proves the file is deployable: it opens a sealed artifact, rebuilds a
//! model and makes it answer, with no checkpoint and no network. It answers
//! ONE prompt, greedily, and returns the whole string at the end.
//!
//! A conversation needs four things `generate` deliberately does not do, and
//! `generate` is left byte-identical because every published token count runs
//! through it:
//!
//!   * **stop on a token.** `generate` stops on a count. A chat that cannot see
//!     `<|im_end|>` talks until its quota, which is the single most visible
//!     defect a reader would find.
//!   * **sample.** `generate` is a chain of argmaxes, which is what makes the
//!     fused-against-dense comparison lethal and what makes a chat repeat
//!     itself. Both are right, for different jobs.
//!   * **stream.** A `Vec<u32>` returned at the end is a four-second silence.
//!   * **keep the cache across turns.** `generate` calls `fresh_caches()`. Doing
//!     that per turn re-prefills the whole history every time, and a prompt
//!     token costs 3.929 ms through the served kernel (`f1e0-2026-09-10`), so a
//!     2,000-token history would pay eight seconds a turn.
//!
//! So this drives its own loop over the public pieces — `fresh_caches`,
//! `hidden_cached`, `logits_last` — and adds the four.
//!
//! ## What it does not claim
//!
//! Nothing here is measured. It is a demonstrator: the numbers this project
//! publishes come from `mmlu`, `ppl`, `planesbench` and `fusedrun`, all of
//! which hold their protocol. A tok/s printed by a REPL is a REPL's tok/s.
//!
//! The loop itself lives in `llvq_llm::repl` since 2026-11-06, shared with
//! `metalrun chat`; this binary is its loader and its environment.
//!
//! Under `LLVQ_CONFIG` it serves the config whole: the attention form, its
//! store and the decode engine, through `Served::serve`. `LLVQ_KV_PREALLOC`
//! beside it is refused, and so is a temperature above zero under the
//! persistent decode, which only continues a greedy chain.

use candle_core::DType;
use llvq_llm::repl::{ReplOptions, Sampler};
use std::io::Write;

fn env_f32(key: &str, default: f32) -> anyhow::Result<f32> {
    match std::env::var(key) {
        Err(_) => Ok(default),
        Ok(v) => v
            .parse()
            .map_err(|_| anyhow::anyhow!("{key}={v:?} is not a number")),
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        !a.is_empty(),
        "usage: chat <model.llvq> [device] \n\
         env: LLVQ_CHAT_TEMP (0 = greedy, default 0.7), LLVQ_CHAT_TOP_P (0.9), \
         LLVQ_CHAT_SEED, LLVQ_CHAT_MAX_NEW (512), LLVQ_CHAT_SYSTEM"
    );
    let dev_name = a.get(1).map(String::as_str).unwrap_or("cpu");
    let device = llvq_llm::eval::device(dev_name)?;
    // f16, the same choice `bin/run` makes and for the same reason: the model
    // ran at half precision anyway, and it is 7 GB of RAM against 15 on a 4B.
    let dtype = llvq_llm::eval::dtype(DType::F16)?;
    // Resolved here and carried by value: `model.rs` reads no environment
    // variable, and an unknown name is an error rather than a silent fallback.
    let kv_mode = llvq_llm::kvq::KvMode::from_env().map_err(anyhow::Error::msg)?;
    let kv_store = llvq_llm::kvq::KvStore::from_env().map_err(anyhow::Error::msg)?;
    let temp = env_f32("LLVQ_CHAT_TEMP", 0.7)?;
    // The served config, read and checked against the environment BEFORE the
    // load: a variable it refuses must cost a second, not eighty.
    let served = llvq_llm::served::Served::from_env().map_err(anyhow::Error::msg)?;
    if let Some(cfg) = &served {
        llvq_llm::repl::check_served_chat(cfg, temp, |v| std::env::var(v).ok()).map_err(anyhow::Error::msg)?;
    }

    // Announced BEFORE the load, not after. A sealed 4B takes about eighty
    // seconds to rebuild on Metal, and a prompt that appears out of eighty
    // seconds of silence reads as a hang.
    print!("loading {} on {dev_name}… ", a[0]);
    std::io::stdout().flush()?;
    let t0 = std::time::Instant::now();
    // `LLVQ_CONFIG` puts this REPL on the served kernel instead of the dense
    // reconstruction. Unset, nothing changes and the binary is the demo it
    // has always been.
    //
    // Chat is the one harness the fused path can carry today on Metal, and
    // the reason is the prefill: a decode step is one vector, while `mmlu`
    // hands 358 at once and `model::MAX_ROWS` refuses past 256. There is no
    // Metal prefill kernel, so a prompt is fanned out one token a launch.
    //
    // On the served arm the config decides the store, the attention form and
    // the decode engine (`Served::serve`, the door `bin/gsm8k` takes), and
    // refuses on Metal what it does not serve there. Before 2026-11-06 this
    // arm printed the config's provenance, never applied it, and set the
    // store from `LLVQ_KV_PREALLOC`: a grouped config ran the expanded form.
    let (qwen, tokenizer, mut dec) = match &served {
        None => {
            let mut s = llvq_llm::sealed::load(&a[0], dtype, &device, kv_mode)?;
            println!("{:.1} s", t0.elapsed().as_secs_f64());
            s.model.set_kv_store(kv_store);
            (s.model, s.tokenizer, llvq_llm::served_decode::Decoder::eager())
        }
        Some(cfg) => {
            println!();
            println!("{}", cfg.provenance());
            let mut f = llvq_llm::served::load_resolved(
                &a[0], &device, dtype, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv,
                Some("LLVQ_CONFIG"),
            )?;
            println!("{:.1} s", t0.elapsed().as_secs_f64());
            let dec = cfg.serve(&mut f.model, &device)?;
            for line in dec.lines() {
                eprintln!("{line}");
            }
            (f.model, f.tokenizer, dec)
        }
    };
    let opts = ReplOptions {
        sampler: Sampler {
            temp,
            top_p: env_f32("LLVQ_CHAT_TOP_P", 0.9)?,
            state: env_f32("LLVQ_CHAT_SEED", 0.0)? as u64 ^ 0x5EED_5EED_5EED_5EED,
        },
        max_new: env_f32("LLVQ_CHAT_MAX_NEW", 512.0)? as usize,
        think: env_f32("LLVQ_CHAT_THINK", 0.0)? != 0.0,
        // `LLVQ_CHAT_IDS=1` prints the raw ids of each answer.
        show_ids: env_f32("LLVQ_CHAT_IDS", 0.0)? != 0.0,
        system: std::env::var("LLVQ_CHAT_SYSTEM").ok(),
    };
    llvq_llm::repl::run_with(&qwen, &tokenizer, opts, &mut std::io::stdin().lock(), &mut dec)
}
