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

use anyhow::Context;
use candle_core::{DType, IndexOp, Tensor, D};
use llvq_llm::model::NoCapture;
use std::io::{BufRead, Write};

/// Qwen3's chat markers, resolved from the tokenizer the SEALED FILE carries
/// rather than hard-coded. A wrong id here does not crash: it produces a model
/// that never stops, which is the failure this whole binary exists to avoid.
struct Marks {
    im_start: u32,
    im_end: u32,
    eot: u32,
    nl: u32,
}

impl Marks {
    fn resolve(t: &tokenizers::Tokenizer) -> anyhow::Result<Self> {
        let id = |s: &str| -> anyhow::Result<u32> {
            t.token_to_id(s)
                .with_context(|| format!("the tokenizer has no {s:?}; is this a Qwen3 artifact?"))
        };
        let nl = t
            .encode("\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .first()
            .copied()
            .context("the tokenizer encodes a newline to nothing")?;
        Ok(Marks {
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            eot: id("<|endoftext|>")?,
            nl,
        })
    }

    fn stops(&self) -> [u32; 2] {
        [self.im_end, self.eot]
    }
}

/// `<|im_start|>{role}\n{body}<|im_end|>\n`, built from ids.
///
/// Built from ids and not from a formatted string: the string form would go
/// back through the tokenizer, and a BPE merge across a marker boundary would
/// silently produce a different prompt than the one intended.
fn turn(
    t: &tokenizers::Tokenizer,
    m: &Marks,
    role: &str,
    body: &str,
) -> anyhow::Result<Vec<u32>> {
    let mut v = vec![m.im_start];
    v.extend(
        t.encode(role, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.nl);
    v.extend(
        t.encode(body, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.im_end);
    v.push(m.nl);
    Ok(v)
}

/// The assistant's opening, which carries no body: the model writes it.
fn open_assistant(t: &tokenizers::Tokenizer, m: &Marks) -> anyhow::Result<Vec<u32>> {
    let mut v = vec![m.im_start];
    v.extend(
        t.encode("assistant", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.nl);
    Ok(v)
}

/// Temperature and nucleus, on the CPU, over one row of logits.
///
/// `temp == 0` is argmax and is exact: the same chain `generate` walks, so a
/// chat started at zero reproduces `run` token for token, which is the only
/// cheap way to tell a sampling bug from a model bug.
struct Sampler {
    temp: f32,
    top_p: f32,
    state: u64,
}

impl Sampler {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64, the generator the rest of the repository uses.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick(&mut self, logits: &Tensor) -> anyhow::Result<u32> {
        if self.temp <= 0.0 {
            return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
        }
        let v = logits.to_vec1::<f32>()?;
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<(u32, f32)> = v
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as u32, ((x - max) / self.temp).exp()))
            .collect();
        p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Nucleus: the shortest prefix whose mass reaches `top_p`. At least one
        // token always survives, so a degenerate `top_p` cannot empty the set.
        let total: f32 = p.iter().map(|x| x.1).sum();
        let mut acc = 0.0f32;
        let mut keep = p.len();
        for (i, (_, w)) in p.iter().enumerate() {
            acc += w / total;
            if acc >= self.top_p {
                keep = i + 1;
                break;
            }
        }
        p.truncate(keep.max(1));

        let mass: f32 = p.iter().map(|x| x.1).sum();
        let r = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32 * mass;
        let mut acc = 0.0f32;
        for (id, w) in &p {
            acc += w;
            if acc >= r {
                return Ok(*id);
            }
        }
        Ok(p.last().expect("the nucleus keeps at least one token").0)
    }
}

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

    let mut sealed = llvq_llm::sealed::load(&a[0], dtype, &device, kv_mode)?;
    sealed.model.set_kv_store(kv_store);
    let tok = &sealed.tokenizer;
    let marks = Marks::resolve(tok)?;
    let stops = marks.stops();

    let mut sampler = Sampler {
        temp: env_f32("LLVQ_CHAT_TEMP", 0.7)?,
        top_p: env_f32("LLVQ_CHAT_TOP_P", 0.9)?,
        state: env_f32("LLVQ_CHAT_SEED", 0.0)? as u64 ^ 0x5EED_5EED_5EED_5EED,
    };
    let max_new: usize = env_f32("LLVQ_CHAT_MAX_NEW", 512.0)? as usize;

    println!("{} on {dev_name}", a[0]);
    println!(
        "  temp {:.2}, top_p {:.2}, max {} new tokens a turn",
        sampler.temp, sampler.top_p, max_new
    );
    println!("  /reset drops the conversation, /bye leaves\n");

    // The cache and the position are the conversation. They are built once and
    // carried across turns: only the NEW tokens of a turn are ever fed, which
    // is the whole difference from calling `generate` in a loop.
    let mut caches = sealed.model.fresh_caches();
    let mut offset = 0usize;

    if let Ok(sys) = std::env::var("LLVQ_CHAT_SYSTEM") {
        let ids = turn(tok, &marks, "system", &sys)?;
        sealed
            .model
            .hidden_cached(
                &Tensor::from_slice(&ids, (1, ids.len()), sealed.model.device())?,
                offset,
                &mut caches,
                &mut NoCapture,
            )
            .context("the system turn")?;
        offset += ids.len();
    }

    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/bye" => return Ok(()),
            "/reset" => {
                caches = sealed.model.fresh_caches();
                offset = 0;
                println!("  conversation dropped, {} tokens released\n", offset);
                continue;
            }
            _ => {}
        }

        let mut ids = turn(tok, &marks, "user", line)?;
        ids.extend(open_assistant(tok, &marks)?);
        let dev = sealed.model.device();
        let mut h = sealed.model.hidden_cached(
            &Tensor::from_slice(&ids, (1, ids.len()), dev)?,
            offset,
            &mut caches,
            &mut NoCapture,
        )?;
        offset += ids.len();

        let mut n = 0usize;
        loop {
            let logits = sealed.model.logits_last(&h)?.i((0, 0))?;
            let next = sampler.pick(&logits)?;
            if stops.contains(&next) {
                break;
            }
            // Decoded one token at a time. A multi-byte character arrives in
            // pieces, and `decode` on a partial sequence yields the replacement
            // character rather than failing, which is the right behaviour for a
            // stream and the wrong one for a transcript.
            print!(
                "{}",
                tok.decode(&[next], false).map_err(|e| anyhow::anyhow!("{e}"))?
            );
            std::io::stdout().flush()?;

            n += 1;
            if n >= max_new {
                println!("\n  [stopped at {max_new} tokens]");
                break;
            }
            h = sealed.model.hidden_cached(
                &Tensor::from_slice(&[next], (1, 1), dev)?,
                offset,
                &mut caches,
                &mut NoCapture,
            )?;
            offset += 1;
        }
        // The assistant's own `<|im_end|>` goes into the cache, or the next
        // turn starts inside an unterminated message and the model keeps
        // writing the previous answer.
        let close = [marks.im_end, marks.nl];
        sealed.model.hidden_cached(
            &Tensor::from_slice(&close, (1, 2), dev)?,
            offset,
            &mut caches,
            &mut NoCapture,
        )?;
        offset += close.len();
        println!("\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(v: &[f32]) -> Tensor {
        Tensor::from_slice(v, v.len(), &candle_core::Device::Cpu).expect("a row")
    }

    #[test]
    fn temperature_zero_is_argmax_and_ignores_the_seed() {
        let l = logits(&[0.1, 5.0, 0.2, 4.9]);
        for seed in [0u64, 7, 12345] {
            let mut s = Sampler { temp: 0.0, top_p: 0.9, state: seed };
            assert_eq!(s.pick(&l).expect("a token"), 1);
        }
    }

    #[test]
    fn a_nucleus_of_one_always_returns_the_top_token() {
        let l = logits(&[0.1, 5.0, 0.2, 4.9]);
        let mut s = Sampler { temp: 1.0, top_p: 0.0, state: 3 };
        for _ in 0..32 {
            assert_eq!(s.pick(&l).expect("a token"), 1, "top_p = 0 must keep one token");
        }
    }

    #[test]
    fn the_nucleus_never_returns_a_token_it_cut() {
        // Two tokens carry essentially all the mass; the other two must never
        // come back, whatever the draw.
        let l = logits(&[-20.0, 5.0, -20.0, 4.9]);
        let mut s = Sampler { temp: 1.0, top_p: 0.95, state: 11 };
        for _ in 0..200 {
            let id = s.pick(&l).expect("a token");
            assert!(id == 1 || id == 3, "the nucleus returned {id}");
        }
    }

    #[test]
    fn sampling_is_reproducible_from_its_seed() {
        let l = logits(&[1.0, 1.1, 0.9, 1.05]);
        let draw = |seed| {
            let mut s = Sampler { temp: 1.0, top_p: 1.0, state: seed };
            (0..24).map(|_| s.pick(&l).expect("a token")).collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42), "the same seed must replay");
        assert_ne!(draw(42), draw(43), "two seeds that agree would be a dead RNG");
    }
}
