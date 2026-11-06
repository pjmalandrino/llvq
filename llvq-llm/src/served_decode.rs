//! The decode engine a served config names ([`crate::served::Decode`]): the
//! eager path, or 4E (`crate::persist4e_cuda`).
//!
//! The prefill is the eager path's under both engines, in chunks of at most
//! [`crate::model::MAX_PREFILL_ROWS`] rows, as `Qwen3::generate_timed` sends
//! it. Under `Persistent` the decode then runs in 4E launches of at most
//! [`Decoder::launch_tokens`] tokens, each starting where the last stopped.
//! The config sends every decode row to the attention kernel from position 0,
//! on both engines ([`crate::served::Served::apply`]), so 4E owes the eager
//! path's tokens bit for bit: the identity the card checks on every run of the
//! served door.
//!
//! [`chunked_greedy`] is the launch loop on its own, pure, so its chunking and
//! its stop rule are tested on the host against the plain greedy loop.

use crate::model::{KvCache, Qwen3, TimedGeneration};
use crate::served::{Decode, Served};

/// The tokens of one 4E launch a served run asks for, at most. Large enough
/// that the per-launch cost (one host read of the token ring, a few uploads)
/// is spread over many tokens, small enough that a stop token wastes little:
/// 4E stops on one token, and a second stop token (GSM8K has two) is cut
/// after the launch.
pub const LAUNCH_TOKENS: u32 = 64;

/// The engine, built once per process from the served config.
pub struct Decoder {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    rt: Option<Box<crate::persist4e_cuda::Runtime4e>>,
    lines: Vec<String>,
}

impl Decoder {
    /// The eager path: what every binary did before the field existed.
    pub fn eager() -> Self {
        Decoder {
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            rt: None,
            lines: Vec::new(),
        }
    }

    pub fn is_persistent(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            self.rt.is_some()
        }
        #[cfg(not(all(target_os = "linux", feature = "cuda")))]
        {
            false
        }
    }

    /// The install lines (4E's module, its RoPE form, its registers), to print.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// The tokens one launch makes, at most.
    pub fn launch_tokens(&self) -> u32 {
        LAUNCH_TOKENS
    }

    /// Greedy continuation of `first`, already chosen and not yet in
    /// `caches`, which hold `offset` positions: up to `more` tokens after it,
    /// cut before the first token of `stops`. Returns the tokens and whether a
    /// stop token ended them. Eager: one step a token, as
    /// `Qwen3::decode_greedy` does. Persistent: 4E launches.
    pub fn continue_greedy(
        &mut self,
        model: &Qwen3,
        caches: &mut [KvCache],
        first: u32,
        offset: usize,
        more: usize,
        stops: &[u32],
    ) -> anyhow::Result<(Vec<u32>, bool)> {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let Some(rt) = self.rt.as_mut() {
            let eos = stops.first().copied().unwrap_or(u32::MAX);
            return chunked_greedy(first, offset, more, stops, LAUNCH_TOKENS, |pos, tok, k| {
                rt.decode(caches, pos, tok, k, eos)
            });
        }
        chunked_greedy(first, offset, more, stops, 1, |pos, tok, _| {
            let logits = model.step_logits(tok, pos, caches)?;
            Ok(vec![logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?])
        })
    }

    /// `Qwen3::generate_timed` through this engine: `max_new` tokens after
    /// `tokens`, the prompt in chunks of `chunk`. Eager: that function, call
    /// for call. Persistent: the same prefill, then 4E with no stop token.
    pub fn generate_timed(
        &mut self,
        model: &Qwen3,
        tokens: &[u32],
        max_new: usize,
        chunk: usize,
    ) -> anyhow::Result<TimedGeneration> {
        if !self.is_persistent() {
            return Ok(model.generate_timed(tokens, max_new, chunk)?);
        }
        anyhow::ensure!(!tokens.is_empty() && max_new > 0 && chunk > 0, "generate_timed: nothing to time");
        let mut caches = model.fresh_caches();
        let t0 = std::time::Instant::now();
        let first = model.prefill_next_in(tokens, chunk, &mut caches)?;
        let prompt_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let (rest, _) = self.continue_greedy(model, &mut caches, first, tokens.len(), max_new - 1, &[])?;
        let decode_s = t1.elapsed().as_secs_f64();
        let mut out = Vec::with_capacity(max_new);
        out.push(first);
        out.extend(rest);
        Ok(TimedGeneration { tokens: out, prompt_s, decode_s })
    }
}

impl Served {
    /// The engine this config names, for `model` on `device`. Call after
    /// [`Served::apply`], which puts the eager path on the same threshold.
    /// `Persistent` installs 4E at 512 threads, the geometry its card readings
    /// ran (the driver's ptxas spills the 768-thread symbol), and checks RoPE
    /// against candle on the card ([`crate::persist4e_cuda::install`]).
    pub fn decoder(&self, model: &Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
        match self.decode {
            Decode::Eager => Ok(Decoder::eager()),
            Decode::Persistent => {
                anyhow::ensure!(
                    model.kernel_from() == 0,
                    "{}: the persistent decode needs Served::apply first, which sends every decode row to the \
                     kernel; the model's threshold is {}",
                    self.path.display(),
                    model.kernel_from()
                );
                #[cfg(all(target_os = "linux", feature = "cuda"))]
                {
                    let mut rt = crate::persist4e_cuda::install_rows(
                        model,
                        device,
                        LAUNCH_TOKENS,
                        Some(512),
                        self.rows.form(),
                    )?;
                    rt.set_served_kernel_from(0);
                    let lines = std::mem::take(&mut rt.lines);
                    Ok(Decoder { rt: Some(Box::new(rt)), lines })
                }
                #[cfg(not(all(target_os = "linux", feature = "cuda")))]
                {
                    let _ = device;
                    anyhow::bail!(
                        "{}: decode \"persistent\" in a build with no CUDA backend; 4E is one cooperative CUDA launch",
                        self.path.display()
                    )
                }
            }
        }
    }
}

/// The launch loop of [`Decoder::continue_greedy`], pure. `launch(pos, tok,
/// k)` feeds `tok` at position `pos` and returns the next tokens it chose, at
/// most `k`, the caches then holding `pos` plus their count; it may stop
/// early on a stop token, which it returns last. At most `max_launch` tokens a
/// launch. The tokens are cut before the first stop token, wherever a launch
/// put it.
pub fn chunked_greedy(
    first: u32,
    offset: usize,
    more: usize,
    stops: &[u32],
    max_launch: u32,
    mut launch: impl FnMut(usize, u32, u32) -> anyhow::Result<Vec<u32>>,
) -> anyhow::Result<(Vec<u32>, bool)> {
    anyhow::ensure!(max_launch > 0, "chunked_greedy: launches of 0 tokens");
    let mut out = Vec::with_capacity(more);
    let (mut tok, mut pos) = (first, offset);
    while out.len() < more {
        let k = (more - out.len()).min(max_launch as usize) as u32;
        let made = launch(pos, tok, k)?;
        anyhow::ensure!(
            !made.is_empty() && made.len() <= k as usize,
            "a launch asked for {k} tokens made {}",
            made.len()
        );
        for &t in &made {
            if stops.contains(&t) {
                return Ok((out, true));
            }
            out.push(t);
        }
        anyhow::ensure!(made.len() == k as usize, "a launch stopped after {} of {k} tokens on no stop token", made.len());
        pos += made.len();
        tok = *made.last().expect("checked non-empty");
    }
    Ok((out, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy greedy model: the next token is a function of the token fed and
    /// its position, and a launch stops after `eos`, as 4E does.
    fn next(tok: u32, pos: usize) -> u32 {
        (tok.wrapping_mul(7).wrapping_add(pos as u32 * 3 + 1)) % 23
    }

    fn reference(first: u32, offset: usize, more: usize, stops: &[u32]) -> (Vec<u32>, bool) {
        let mut out = Vec::new();
        let (mut tok, mut pos) = (first, offset);
        while out.len() < more {
            let t = next(tok, pos);
            if stops.contains(&t) {
                return (out, true);
            }
            out.push(t);
            tok = t;
            pos += 1;
        }
        (out, false)
    }

    #[test]
    fn the_launch_loop_is_the_greedy_loop() {
        for stops in [vec![], vec![5u32], vec![5, 11], vec![11, 5], vec![99]] {
            for max_launch in [1u32, 2, 3, 7, 64] {
                for more in [0usize, 1, 2, 9, 40] {
                    for first in [0u32, 4, 17] {
                        let offset = 13usize;
                        let mut cache_len = offset;
                        let eos = stops.first().copied().unwrap_or(u32::MAX);
                        let got = chunked_greedy(first, offset, more, &stops, max_launch, |pos, tok, k| {
                            assert_eq!(pos, cache_len, "a launch starts where the caches end");
                            let (mut t, mut p, mut made) = (tok, pos, Vec::new());
                            while made.len() < k as usize {
                                t = next(t, p);
                                p += 1;
                                made.push(t);
                                if t == eos {
                                    break;
                                }
                            }
                            cache_len = pos + made.len();
                            Ok(made)
                        })
                        .unwrap();
                        assert_eq!(got, reference(first, offset, more, &stops), "stops {stops:?}, launches of {max_launch}, {more} more after {first}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_launch_loop_refuses_a_short_launch_with_no_stop() {
        let e = chunked_greedy(1, 0, 10, &[], 4, |_, _, k| Ok(vec![2; k as usize - 1])).unwrap_err();
        assert!(e.to_string().contains("on no stop token"), "{e}");
        let e = chunked_greedy(1, 0, 10, &[], 4, |_, _, _| Ok(Vec::new())).unwrap_err();
        assert!(e.to_string().contains("made 0"), "{e}");
    }
}
