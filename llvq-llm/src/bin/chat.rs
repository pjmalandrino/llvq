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
    /// The id of `"\n\n"`, which is ONE token (271) and not two of `nl`.
    /// Qwen3's own template prefills the empty reasoning block with it, and a
    /// model trained on 271 does not read 198 twice as the same thing — which
    /// is exactly what the first attempt got wrong.
    nl2: u32,
    /// `<think>` and `</think>`, present on Qwen3 and absent on a model that
    /// does not reason out loud. `Option`, because this binary must not refuse
    /// an artifact for lacking them.
    think: Option<(u32, u32)>,
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
        let nl2 = t
            .encode("\n\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .first()
            .copied()
            .context("the tokenizer encodes a blank line to nothing")?;
        Ok(Marks {
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            eot: id("<|endoftext|>")?,
            nl,
            nl2,
            think: t
                .token_to_id("<think>")
                .zip(t.token_to_id("</think>")),
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
///
/// With `think` false the opening is PRE-FILLED with an empty reasoning block,
/// `<think>\n\n</think>\n\n`, which is how Qwen3 is told not to reason out
/// loud. It is not a stop token and not a filter: the block is closed before
/// the model writes a word, so there is nothing to strip afterwards and the
/// budget goes to the answer.
///
/// Off by default here. A 48-token budget on the first smoke went entirely into
/// the model thinking about the question, which is correct behaviour and a
/// useless chat.
fn open_assistant(
    t: &tokenizers::Tokenizer,
    m: &Marks,
    think: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut v = vec![m.im_start];
    v.extend(
        t.encode("assistant", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.nl);
    if !think {
        if let Some((open, close)) = m.think {
            v.push(open);
            v.push(m.nl2);
            v.push(close);
            v.push(m.nl2);
        }
    }
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

/// Positions this turn would need: the prompt, plus the worst case of the
/// answer, plus the two tokens that close the assistant's message.
///
/// RoPE's tables are built once to `max_position_embeddings` rows, so walking
/// past them is not a slow path, it is an opaque candle error thousands of
/// steps into a conversation. `generate`'s own header records the same edge at
/// `max_new = 0`. Checked BEFORE the prefill, because a refusal after it would
/// leave the cache holding half a turn.
fn turn_fits(offset: usize, prompt: usize, max_new: usize, limit: usize) -> bool {
    offset
        .checked_add(prompt)
        .and_then(|n| n.checked_add(max_new))
        .and_then(|n| n.checked_add(CLOSE_LEN))
        .is_some_and(|n| n <= limit)
}

/// `<|im_end|>` and the newline that follow every assistant turn.
const CLOSE_LEN: usize = 2;

/// The part of a decoded answer that is safe to print now.
///
/// A character outside the ASCII range spans several tokens, and decoding a
/// sequence that ends mid-character yields a TRAILING U+FFFD. Printing it puts
/// a replacement character on screen that the next token would have completed
/// — which is what the first two attempts did, once per token and then once
/// per character.
///
/// So the tail of replacements is held back. It costs nothing: the next token
/// either completes the character, and it appears whole, or the answer ends
/// and the character was genuinely broken.
fn printable(full: &str) -> &str {
    full.trim_end_matches('\u{FFFD}')
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
    let limit = sealed.model.config().max_position_embeddings;
    let tok = &sealed.tokenizer;
    let marks = Marks::resolve(tok)?;
    let stops = marks.stops();

    let mut sampler = Sampler {
        temp: env_f32("LLVQ_CHAT_TEMP", 0.7)?,
        top_p: env_f32("LLVQ_CHAT_TOP_P", 0.9)?,
        state: env_f32("LLVQ_CHAT_SEED", 0.0)? as u64 ^ 0x5EED_5EED_5EED_5EED,
    };
    let max_new: usize = env_f32("LLVQ_CHAT_MAX_NEW", 512.0)? as usize;
    let think = env_f32("LLVQ_CHAT_THINK", 0.0)? != 0.0;
    // `LLVQ_CHAT_IDS=1` prints the raw ids of each answer. Kept rather than
    // deleted after the hunt it was written for: a replacement character on
    // screen has three possible causes — a partial decode, a byte-fallback
    // token the model really emitted, and a terminal — and only the ids tell
    // them apart.
    let show_ids = env_f32("LLVQ_CHAT_IDS", 0.0)? != 0.0;

    println!("{} on {dev_name}", a[0]);
    println!(
        "  temp {:.2}, top_p {:.2}, max {} new tokens a turn",
        sampler.temp, sampler.top_p, max_new
    );
    println!(
        "  reasoning {}{}",
        match think {
            true => "on",
            false => "off",
        },
        match marks.think {
            None => " (this artifact has no think tokens)",
            Some(_) => "",
        }
    );
    println!("  context {limit} positions");
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
                // Read BEFORE the reset. Printing it after set `offset` to zero
                // and reported that every conversation released nothing.
                println!("  conversation dropped, {offset} positions released\n");
                caches = sealed.model.fresh_caches();
                offset = 0;
                continue;
            }
            _ => {}
        }

        let mut ids = turn(tok, &marks, "user", line)?;
        ids.extend(open_assistant(tok, &marks, think)?);
        if !turn_fits(offset, ids.len(), max_new, limit) {
            println!(
                "  this turn needs {} positions of the {limit} this model has, and {offset} \
                 are already held.\n  /reset to start over.\n",
                offset + ids.len() + max_new + CLOSE_LEN
            );
            continue;
        }
        let dev = sealed.model.device();
        let mut h = sealed.model.hidden_cached(
            &Tensor::from_slice(&ids, (1, ids.len()), dev)?,
            offset,
            &mut caches,
            &mut NoCapture,
        )?;
        offset += ids.len();

        let mut n = 0usize;
        // The answer so far, in ids and in the text already written. The two
        // are kept side by side because a token does not map to a character.
        let mut said: Vec<u32> = Vec::new();
        let mut shown = String::new();
        loop {
            let logits = sealed.model.logits_last(&h)?.i((0, 0))?;
            let next = sampler.pick(&logits)?;
            if stops.contains(&next) {
                break;
            }
            // NOT `decode(&[next])`. A character outside the ASCII range
            // arrives across several tokens, and decoding one in isolation
            // yields U+FFFD — which the first version printed, three of them,
            // in place of an emoji.
            //
            // So the whole answer is decoded each time and only the NEW SUFFIX
            // is written. A partial character simply does not extend the
            // string yet, and appears whole on the token that completes it.
            // Quadratic in `decode` calls, on hundreds of tokens, against a
            // forward pass a thousand times dearer.
            said.push(next);
            let full = tok.decode(&said, false).map_err(|e| anyhow::anyhow!("{e}"))?;
            let safe = printable(&full);
            if let Some(fresh) = safe.strip_prefix(shown.as_str()) {
                print!("{fresh}");
                std::io::stdout().flush()?;
                shown = safe.to_string();
            }

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
        if show_ids {
            println!("\n  ids: {said:?}");
        }
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
    fn a_trailing_replacement_is_held_back_and_a_settled_one_is_not() {
        assert_eq!(printable("Hello"), "Hello");
        assert_eq!(printable("Hello \u{FFFD}"), "Hello ");
        assert_eq!(printable("Hello \u{FFFD}\u{FFFD}"), "Hello ");
        // Only the TAIL is held: a replacement the model really produced in
        // the middle of settled text stays, or the stream would rewind.
        assert_eq!(printable("a\u{FFFD}b"), "a\u{FFFD}b");
        assert_eq!(printable("\u{FFFD}"), "");
    }

    #[test]
    fn the_stream_never_rewinds_what_it_printed() {
        // The invariant the loop rests on: what is safe to print only grows.
        let steps = ["", "H", "He", "He\u{FFFD}", "He\u{1F600}", "He\u{1F600}!"];
        let mut shown = String::new();
        for full in steps {
            let safe = printable(full);
            if let Some(fresh) = safe.strip_prefix(shown.as_str()) {
                shown.push_str(fresh);
            }
            assert!(safe.starts_with(shown.as_str()) || shown.starts_with(safe));
        }
        assert_eq!(shown, "He\u{1F600}!");
    }

    #[test]
    fn a_turn_that_would_overrun_the_context_is_refused() {
        // Exactly full fits; one more position does not.
        assert!(turn_fits(0, 10, 88, 100));
        assert!(!turn_fits(0, 10, 89, 100));
        assert!(turn_fits(90, 4, 4, 100));
        assert!(!turn_fits(91, 4, 4, 100));
    }

    #[test]
    fn the_fit_check_cannot_overflow_into_a_pass() {
        // A saturating add would wrap to a small number and report room.
        assert!(!turn_fits(usize::MAX - 1, 4, 4, 100));
        assert!(!turn_fits(0, usize::MAX, 4, usize::MAX));
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
