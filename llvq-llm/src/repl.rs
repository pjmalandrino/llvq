//! The conversation loop of `bin/chat`, shared with `bin/metalrun chat`.
//!
//! Moved out of `bin/chat` on 2026-11-06 so that the Mac bench can talk to the
//! same arms it times (`lut`, `lutd+grp`, `D`, ...) through the same loop the
//! REPL has always run. Two copies of a loop are two loops. `bin/chat` keeps
//! its loader and its environment; this module only takes a built model.
//!
//! A conversation needs four things `generate` deliberately does not do, and
//! `generate` is left byte-identical because every published token count runs
//! through it:
//!
//!   * **stop on a token.** `generate` stops on a count. A chat that cannot see
//!     `<|im_end|>` talks until its quota.
//!   * **sample.** `generate` is a chain of argmaxes, which is what makes the
//!     fused-against-dense comparison lethal and what makes a chat repeat
//!     itself.
//!   * **stream.** A `Vec<u32>` returned at the end is a four-second silence.
//!   * **keep the cache across turns.** `generate` calls `fresh_caches()`.
//!     Doing that per turn re-prefills the whole history every time.
//!
//! Nothing here is measured. A tok/s printed by a REPL is a REPL's tok/s: no
//! warmup, no rounds, no interleaving, and a CPU sampler in every step.

use crate::chatfmt::{open_assistant, turn, Marks};
use crate::model::{NoCapture, Qwen3};
use anyhow::Context;
use candle_core::{IndexOp, Tensor, D};
use std::io::{BufRead, Write};

/// Temperature and nucleus, on the CPU, over one row of logits.
///
/// `temp == 0` is argmax and is exact: the same chain `generate` walks, so a
/// chat started at zero reproduces `run` token for token, which is the only
/// cheap way to tell a sampling bug from a model bug.
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub state: u64,
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

    pub fn pick(&mut self, logits: &Tensor) -> anyhow::Result<u32> {
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
/// steps into a conversation. Checked BEFORE the prefill, because a refusal
/// after it would leave the cache holding half a turn.
pub fn turn_fits(offset: usize, prompt: usize, max_new: usize, limit: usize) -> bool {
    offset
        .checked_add(prompt)
        .and_then(|n| n.checked_add(max_new))
        .and_then(|n| n.checked_add(CLOSE_LEN))
        .is_some_and(|n| n <= limit)
}

/// `<|im_end|>` and the newline that follow every assistant turn.
pub const CLOSE_LEN: usize = 2;

/// The part of a decoded answer that is safe to print now.
///
/// A character outside the ASCII range spans several tokens, and decoding a
/// sequence that ends mid-character yields a TRAILING U+FFFD. So the tail of
/// replacements is held back: the next token either completes the character,
/// and it appears whole, or the answer ends and the character was genuinely
/// broken.
pub fn printable(full: &str) -> &str {
    full.trim_end_matches('\u{FFFD}')
}

/// What a conversation is run with.
pub struct ReplOptions {
    pub sampler: Sampler,
    pub max_new: usize,
    pub think: bool,
    /// Print the raw ids of each answer: a replacement character on screen has
    /// three possible causes, a partial decode, a byte-fallback token the
    /// model really emitted, and a terminal, and only the ids tell them apart.
    pub show_ids: bool,
    pub system: Option<String>,
}

/// Runs the conversation on `input` until end of input or `/bye`.
pub fn run(
    qwen: &Qwen3,
    tok: &tokenizers::Tokenizer,
    mut opts: ReplOptions,
    input: &mut dyn BufRead,
) -> anyhow::Result<()> {
    let limit = qwen.config().max_position_embeddings;
    let marks = Marks::resolve(tok)?;
    let stops = marks.stops();
    let (max_new, think) = (opts.max_new, opts.think);

    println!(
        "  temp {:.2}, top_p {:.2}, max {} new tokens a turn",
        opts.sampler.temp, opts.sampler.top_p, max_new
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
    let mut caches = qwen.fresh_caches();
    let mut offset = 0usize;

    if let Some(sys) = &opts.system {
        let ids = turn(tok, &marks, "system", sys)?;
        qwen.hidden_cached(
            &Tensor::from_slice(&ids, (1, ids.len()), qwen.device())?,
            offset,
            &mut caches,
            &mut NoCapture,
        )
        .context("the system turn")?;
        offset += ids.len();
    }

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
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
                caches = qwen.fresh_caches();
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
        let dev = qwen.device();
        let t_prompt = std::time::Instant::now();
        let mut h = qwen.hidden_cached(
            &Tensor::from_slice(&ids, (1, ids.len()), dev)?,
            offset,
            &mut caches,
            &mut NoCapture,
        )?;
        offset += ids.len();
        let prompt_s = t_prompt.elapsed().as_secs_f64();

        let t_turn = std::time::Instant::now();
        let mut n = 0usize;
        // The answer so far, in ids and in the text already written. The two
        // are kept side by side because a token does not map to a character.
        let mut said: Vec<u32> = Vec::new();
        let mut shown = String::new();
        loop {
            let logits = qwen.logits_last(&h)?.i((0, 0))?;
            let next = opts.sampler.pick(&logits)?;
            if stops.contains(&next) {
                break;
            }
            // NOT `decode(&[next])`: a character outside the ASCII range
            // arrives across several tokens. The whole answer is decoded each
            // time and only the NEW SUFFIX is written.
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
            h = qwen.hidden_cached(&Tensor::from_slice(&[next], (1, 1), dev)?, offset, &mut caches, &mut NoCapture)?;
            offset += 1;
        }
        // The assistant's own `<|im_end|>` goes into the cache, or the next
        // turn starts inside an unterminated message and the model keeps
        // writing the previous answer.
        let close = [marks.im_end, marks.nl];
        qwen.hidden_cached(&Tensor::from_slice(&close, (1, 2), dev)?, offset, &mut caches, &mut NoCapture)?;
        offset += close.len();
        if opts.show_ids {
            println!("\n  ids: {said:?}");
        }
        let secs = t_turn.elapsed().as_secs_f64();
        println!(
            "\n  {n} tokens in {secs:.1} s ({:.1} tok/s) after a {}-token prompt read in {:.1} s; \
             {offset} positions held. This REPL's rate, not a measurement\n",
            n as f64 / secs.max(1e-9),
            ids.len(),
            prompt_s
        );
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
