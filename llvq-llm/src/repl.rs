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
use crate::model::Qwen3;
use crate::served::{Decode, Served};
use crate::served_decode::Decoder;
use crate::session::{Finish, Flow, Generate, Session};
use anyhow::Context;
use candle_core::{Tensor, D};
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
        let p: Vec<(u32, f32)> = v
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as u32, ((x - max) / self.temp).exp()))
            .collect();
        let top_p = self.top_p;
        let p = nucleus_selected(&p, top_p).unwrap_or_else(|| nucleus_sorted(p, top_p));

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

/// The nucleus by a full sort: every weight in descending order, then the
/// shortest prefix whose mass reaches `top_p`. At least one token always
/// survives, so a degenerate `top_p` cannot empty the set.
fn nucleus_sorted(mut p: Vec<(u32, f32)>, top_p: f32) -> Vec<(u32, f32)> {
    p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let total: f32 = p.iter().map(|x| x.1).sum();
    let mut acc = 0.0f32;
    let mut keep = p.len();
    for (i, (_, w)) in p.iter().enumerate() {
        acc += w / total;
        if acc >= top_p {
            keep = i + 1;
            break;
        }
    }
    p.truncate(keep.max(1));
    p
}

/// The largest selection [`nucleus_selected`] tries: 64, 128, 256, 512 and
/// 1,024 weights, five selections at most.
const NUCLEUS_MAX_K: usize = 1024;

/// [`nucleus_sorted`] without sorting the whole row, or `None` when the
/// shortcut cannot be shown to return the same vector, bit for bit.
///
/// The `k + 1` heaviest weights are selected, then sorted: their values are
/// the first `k + 1` values of the full sort. Three conditions make the
/// result the full sort's nucleus.
///
/// * The total. The full sort sums in descending order. Once adding a weight
///   leaves the running sum unchanged, every later weight, smaller and not
///   negative, leaves it unchanged too, because rounding is monotone. That
///   weight must lie in the sorted part.
/// * The cut. The running share reaches `top_p` inside the sorted part, with
///   one weight to spare.
/// * The order. The kept weights and the first weight cut are strictly
///   decreasing. The unstable sort orders equal weights in a way a selection
///   does not reproduce, so a tie returns `None`.
///
/// A NaN, an infinite or a negative weight returns `None`, and so does a
/// `top_p` of one or more, whose nucleus is usually the whole row.
///
/// `k` starts at 64 and doubles up to [`NUCLEUS_MAX_K`], so a row whose total
/// has not settled within its 1,025 heaviest weights costs at most five
/// selections before the full sort. The bound only decides when to give up:
/// a `Some` is the full sort's vector at any bound.
fn nucleus_selected(p: &[(u32, f32)], top_p: f32) -> Option<Vec<(u32, f32)>> {
    if top_p.is_nan()
        || top_p >= 1.0
        || !p.iter().all(|x| x.1.is_finite() && x.1.is_sign_positive())
    {
        return None;
    }
    let desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
    let mut work = p.to_vec();
    let mut k = 64;
    while k < p.len() / 4 && k <= NUCLEUS_MAX_K {
        work.select_nth_unstable_by(k, desc);
        let mut c = work[..=k].to_vec();
        c.sort_unstable_by(desc);
        let mut total = 0.0f32;
        let settled = c.iter().any(|x| {
            let sum = total + x.1;
            let still = sum == total;
            total = sum;
            still
        });
        if settled {
            let mut acc = 0.0f32;
            let cut = c.iter().position(|x| {
                acc += x.1 / total;
                acc >= top_p
            });
            if let Some(i) = cut.filter(|&i| i < k) {
                let strict = c[..=i + 1].windows(2).all(|w| w[0].1 > w[1].1);
                c.truncate(i + 1);
                return strict.then_some(c);
            }
        }
        k *= 2;
    }
    None
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

/// A served config against the REPL's environment, before the load.
///
/// `bin/chat` calls it under `LLVQ_CONFIG`, with `lookup` the shell and
/// `temp` the resolved `LLVQ_CHAT_TEMP`, so a refusal costs a second and not
/// an eighty-second load. Two refusals:
///
/// * `LLVQ_KV_PREALLOC`: the config names the cache store
///   ([`crate::served::check_store_env`]);
/// * the persistent decode at a temperature above zero. 4E takes the argmax
///   on the card, token after token inside one launch, so it can only
///   continue a greedy chain; sampling would need the eager path, which is
///   not what the config names.
pub fn check_served_chat(
    cfg: &Served,
    temp: f32,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    crate::served::check_store_env(lookup)?;
    if cfg.decode == Decode::Persistent && temp > 0.0 {
        return Err(format!(
            "{}: decode \"persistent\" at LLVQ_CHAT_TEMP={temp}. 4E takes the argmax on the card and \
             cannot sample; set LLVQ_CHAT_TEMP=0 for a greedy chat on 4E, or serve this object with a \
             config that names the eager engine.",
            cfg.path.display()
        ));
    }
    Ok(())
}

/// Write the part of `said` not yet on screen; `shown` is what is.
fn stream(tok: &tokenizers::Tokenizer, said: &[u32], shown: &mut String) -> anyhow::Result<()> {
    let full = tok.decode(said, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let safe = printable(&full);
    if let Some(fresh) = safe.strip_prefix(shown.as_str()) {
        print!("{fresh}");
        std::io::stdout().flush()?;
        *shown = safe.to_string();
    }
    Ok(())
}

/// Runs the conversation on `input` until end of input or `/bye`, on the
/// eager decode.
pub fn run(
    qwen: &Qwen3,
    tok: &tokenizers::Tokenizer,
    opts: ReplOptions,
    input: &mut dyn BufRead,
) -> anyhow::Result<()> {
    run_with(qwen, tok, opts, input, &mut Decoder::eager())
}

/// [`run`] on the decode engine a served config installed
/// ([`Served::serve`]).
///
/// The loop is [`Session::answer`] since 2026-11-06, the one the runtime
/// serves: the same `hidden_cached` calls in the same order as the loop that
/// lived here, so the same ids. Eager: one step a token. Persistent: the
/// prefill is the eager path's, the first token is the sampler's (an argmax,
/// since [`check_served_chat`] holds the temperature at zero), and the rest
/// of the answer comes from 4E, one launch of at most
/// [`Decoder::launch_tokens`] tokens at a time, streamed after each launch.
/// The context is bounded by 4E's attention scratch as well as by RoPE.
pub fn run_with(
    qwen: &Qwen3,
    tok: &tokenizers::Tokenizer,
    mut opts: ReplOptions,
    input: &mut dyn BufRead,
    dec: &mut Decoder,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !dec.is_persistent() || opts.sampler.temp <= 0.0,
        "the persistent decode is greedy: a REPL on it runs at temperature 0, not {}",
        opts.sampler.temp
    );
    let rope = qwen.config().max_position_embeddings;
    let limit = dec.max_positions().map_or(rope, |e| e.min(rope));
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

    // The session is the conversation: its caches and the ids they hold,
    // carried across turns. Only the NEW tokens of a turn are ever fed, which
    // is the whole difference from calling `generate` in a loop.
    let mut session = Session::new(qwen);

    if let Some(sys) = &opts.system {
        let ids = turn(tok, &marks, "system", sys)?;
        session.feed(qwen, &ids).context("the system turn")?;
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
                // Read BEFORE the reset. Printing it after set the count to
                // zero and reported that every conversation released nothing.
                println!("  conversation dropped, {} positions released\n", session.len());
                session.reset(qwen);
                continue;
            }
            _ => {}
        }

        let mut ids = turn(tok, &marks, "user", line)?;
        ids.extend(open_assistant(tok, &marks, think)?);
        if !session.fits(ids.len(), max_new, limit) {
            println!(
                "  this turn needs {} positions of the {limit} this model has, and {} \
                 are already held.\n  /reset to start over.\n",
                session.len() + ids.len() + max_new + CLOSE_LEN,
                session.len()
            );
            continue;
        }
        let t_prompt = std::time::Instant::now();
        // The prefill the config names: the chain under a chain config.
        let fed = session.feed_with(qwen, dec, &ids)?;
        let prompt_s = t_prompt.elapsed().as_secs_f64();

        let t_turn = std::time::Instant::now();
        // The text already written. A token does not map to a character, so
        // the whole answer is decoded each time and only the NEW SUFFIX is
        // written (`stream`).
        let mut shown = String::new();
        let mut failed: Option<anyhow::Error> = None;
        let answer = session.answer_fed(
            qwen,
            dec,
            fed,
            Generate { sampler: &mut opts.sampler, max_new, stops: &stops },
            &mut |said| match stream(tok, said, &mut shown) {
                Ok(()) => Flow::Continue,
                Err(e) => {
                    failed = Some(e);
                    Flow::Cancel
                }
            },
        )?;
        if let Some(e) = failed {
            return Err(e);
        }
        if answer.finish == Finish::Length {
            println!("\n  [stopped at {max_new} tokens]");
        }
        // The assistant's own `<|im_end|>` goes into the cache, or the next
        // turn starts inside an unterminated message and the model keeps
        // writing the previous answer.
        session.feed(qwen, &[marks.im_end, marks.nl])?;
        if opts.show_ids {
            println!("\n  ids: {:?}", answer.said);
        }
        let n = answer.said.len();
        let secs = t_turn.elapsed().as_secs_f64();
        println!(
            "\n  {n} tokens in {secs:.1} s ({:.1} tok/s) after a {}-token prompt read in {:.1} s; \
             {} positions held. This REPL's rate, not a measurement\n",
            n as f64 / secs.max(1e-9),
            ids.len(),
            prompt_s,
            session.len()
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

    /// Uniform in `[0, 1)`, from the sampler's own generator.
    fn uniform(state: &mut u64) -> f32 {
        let mut s = Sampler { temp: 0.0, top_p: 0.0, state: *state };
        let x = s.next_u64();
        *state = s.state;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A row the way a model hands it over: `heads` tokens at `level` and
    /// above, over a bulk spread on `[-bulk, bulk]`.
    fn row(seed: u64, n: usize, heads: usize, level: f32, bulk: f32) -> Vec<f32> {
        let mut st = seed;
        let mut v: Vec<f32> = (0..n).map(|_| bulk * (2.0 * uniform(&mut st) - 1.0)).collect();
        for h in 0..heads {
            let at = (uniform(&mut st) * n as f32) as usize % n;
            v[at] = level + 8.0 * uniform(&mut st) + h as f32 * 0.25;
        }
        v
    }

    /// The weights `pick` builds.
    fn weights(v: &[f32], temp: f32) -> Vec<(u32, f32)> {
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        v.iter().enumerate().map(|(i, &x)| (i as u32, ((x - max) / temp).exp())).collect()
    }

    fn bits(p: &[(u32, f32)]) -> Vec<(u32, u32)> {
        p.iter().map(|&(i, w)| (i, w.to_bits())).collect()
    }

    /// The shares the full sort accumulates, with its own total.
    fn sorted_shares(p: &[(u32, f32)]) -> Vec<f32> {
        let mut s = p.to_vec();
        s.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let total: f32 = s.iter().map(|x| x.1).sum();
        let mut acc = 0.0f32;
        s.iter()
            .map(|x| {
                acc += x.1 / total;
                acc
            })
            .collect()
    }

    #[test]
    fn the_selected_nucleus_is_the_sorted_one_bit_for_bit() {
        // (heads, level, bulk): peaked rows, a wide head, a flat row.
        let shapes: [(usize, f32, f32); 4] =
            [(1, 24.0, 4.0), (6, 24.0, 4.0), (90, 12.0, 4.0), (0, 0.0, 1.0)];
        let (mut peaked, mut taken) = (0, 0);
        for seed in 0..16u64 {
            for (heads, level, bulk) in shapes {
                let v = row(seed, 4096, heads, level, bulk);
                for temp in [0.05f32, 0.7, 1.0, 2.0] {
                    let p = weights(&v, temp);
                    for top_p in [0.0f32, 0.3, 0.9, 0.99] {
                        let fast = nucleus_selected(&p, top_p);
                        if let Some(fast) = &fast {
                            assert_eq!(
                                bits(fast),
                                bits(&nucleus_sorted(p.clone(), top_p)),
                                "seed {seed}, heads {heads}, temp {temp}, top_p {top_p}"
                            );
                        }
                        if heads <= 6 && level > 0.0 && temp <= 1.0 {
                            peaked += 1;
                            taken += usize::from(fast.is_some());
                        }
                    }
                }
            }
        }
        // A peaked row at a usual temperature is what the shortcut is for: if
        // it falls back there, it is dead code that the identity check above
        // would never see.
        assert!(taken * 10 >= peaked * 9, "the shortcut ran on {taken} of {peaked} peaked rows");
    }

    #[test]
    fn a_top_p_on_the_edge_of_a_share_cuts_where_the_sort_cuts() {
        // `top_p` equal to a share the full sort computes, and one step either
        // side of it. A total summed in another order moves a share by an
        // ulp, and the cut with it.
        for seed in 0..16u64 {
            let v = row(seed, 4096, 6, 24.0, 4.0);
            for temp in [0.7f32, 1.0] {
                let p = weights(&v, temp);
                let shares = sorted_shares(&p);
                for &s in &shares[..6] {
                    let s_bits = s.to_bits();
                    for edge in [f32::from_bits(s_bits - 1), s, f32::from_bits(s_bits + 1)] {
                        if let Some(fast) = nucleus_selected(&p, edge) {
                            assert_eq!(
                                bits(&fast),
                                bits(&nucleus_sorted(p.clone(), edge)),
                                "seed {seed}, temp {temp}, top_p {edge}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_tie_in_the_nucleus_falls_back_to_the_sort() {
        let mut v = row(5, 4096, 1, 24.0, 4.0);
        // Two equal maxima: the unstable sort decides which comes first.
        v[10] = 40.0;
        v[20] = 40.0;
        let p = weights(&v, 0.7);
        assert!(nucleus_selected(&p, 0.9).is_none());
        // Without the tie, the same row takes the shortcut.
        v[20] = 39.0;
        assert!(nucleus_selected(&weights(&v, 0.7), 0.9).is_some());
        // A NaN anywhere, an infinite logit, a `top_p` of one and a short row
        // fall back too.
        let mut nan = row(5, 4096, 1, 24.0, 4.0);
        nan[7] = f32::NAN;
        assert!(nucleus_selected(&weights(&nan, 0.7), 0.9).is_none());
        let mut inf = row(5, 4096, 1, 24.0, 4.0);
        inf[7] = f32::INFINITY;
        assert!(nucleus_selected(&weights(&inf, 0.7), 0.9).is_none());
        assert!(nucleus_selected(&p, 1.0).is_none());
        assert!(nucleus_selected(&weights(&row(5, 200, 1, 24.0, 4.0), 0.7), 0.9).is_none());
    }

    /// `heavy` distinct weights near one over a row of zeros: the total
    /// settles at the first zero, so the selection that holds `heavy + 1`
    /// weights is the first that can answer.
    fn plateau(n: usize, heavy: usize) -> Vec<(u32, f32)> {
        (0..n)
            .map(|i| {
                let w = if i < heavy { 1.0 - i as f32 * (1.0 / (1 << 20) as f32) } else { 0.0 };
                (i as u32, w)
            })
            .collect()
    }

    #[test]
    fn the_selection_gives_up_past_its_cap() {
        // 500 heavy weights settle in the selection of 513, under the cap:
        // taken, and equal to the sort.
        let p = plateau(16_384, 500);
        let fast = nucleus_selected(&p, 0.5).expect("a total that settles under the cap is taken");
        assert_eq!(bits(&fast), bits(&nucleus_sorted(p.clone(), 0.5)));
        // 1,500 settle only in the selection of 2,049, past the cap: the
        // shortcut gives up, though 2,048 is still under a quarter of the row.
        let p = plateau(16_384, 1_500);
        assert!(nucleus_selected(&p, 0.5).is_none(), "a selection past {NUCLEUS_MAX_K} weights ran");
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
