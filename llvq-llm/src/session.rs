//! One conversation's cache, and the generation loop that runs on it.
//!
//! Moved out of `repl.rs` on 2026-11-06 for the runtime (`llvq-runtime`): the
//! REPL and the HTTP server run this one loop, so a token the server streams
//! is a token `bin/chat` would print. Two copies of a loop are two loops.
//!
//! A [`Session`] holds the caches and the ids they hold, position for
//! position. That second half is what the server needs and the REPL did not:
//! a stateless API sends the whole history every request, and
//! [`Session::reuse_prefix`] keeps the longest common prefix of what is cached
//! and what is asked, so only the new tokens are read.
//!
//! [`Session::answer`] is the REPL's two decode branches, unchanged in what
//! they feed and when: eager, one step a token; persistent (4E), the first
//! token from the sampler then launches of at most
//! [`Decoder::launch_tokens`]. The last token of an answer cut by its cap is
//! never fed, as in the REPL since 2026-09-26, so [`Session::ids`] lists it
//! out. A callback sees the answer grow and can cancel it between steps (eager)
//! or between launches (4E).

use crate::model::{KvCache, NoCapture, Qwen3};
use crate::repl::Sampler;
use crate::served_decode::Decoder;
use candle_core::{IndexOp, Tensor};

/// The caches of one conversation, and the ids they hold.
pub struct Session {
    caches: Vec<KvCache>,
    ids: Vec<u32>,
}

/// Why an answer ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model wrote a stop token. It is not in the answer and not fed.
    Stop,
    /// The answer reached its cap. Its last token is not fed.
    Length,
    /// The callback asked to stop.
    Cancelled,
}

/// An answer: every token said, and why it ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub said: Vec<u32>,
    pub finish: Finish,
}

/// What [`Session::answer`] runs with.
pub struct Generate<'a> {
    pub sampler: &'a mut Sampler,
    /// Tokens at most, counting the first. Zero answers nothing.
    pub max_new: usize,
    pub stops: &'a [u32],
}

/// What a prefill leaves for [`Session::answer_fed`]: the eager path's hidden
/// states, from which the first token is sampled, or the first token the
/// chain prefill already chose.
pub enum Fed {
    Hidden(Tensor),
    First(u32),
}

/// The callback's verdict after each step or launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Cancel,
}

/// The longest common prefix of two id lists.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The positions a cache holds, on any store: [`KvCache::len`] counts them
/// under `Grow` and `Prealloc` only, and under `Cat` the buffers are the
/// history.
fn positions_held(c: &KvCache) -> anyhow::Result<usize> {
    Ok(match c.store() {
        crate::kvq::KvStore::Cat => c.held()?.map_or(Ok(0), |(k, _)| k.dim(2))?,
        crate::kvq::KvStore::Grow(_) | crate::kvq::KvStore::Prealloc(_) => c.len(),
    })
}

impl Session {
    pub fn new(qwen: &Qwen3) -> Self {
        Session { caches: qwen.fresh_caches(), ids: Vec::new() }
    }

    /// A conversation from caches that already hold `ids`, position for
    /// position: a sequence taken back from a batch slot
    /// (`batch::Pool::caches_of`).
    pub fn from_caches(qwen: &Qwen3, caches: Vec<KvCache>, ids: Vec<u32>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            caches.len() == qwen.fresh_caches().len(),
            "Session from_caches: {} caches for {} layers",
            caches.len(),
            qwen.fresh_caches().len()
        );
        for c in &caches {
            let held = c.held()?.map_or(Ok(0), |(k, _)| k.dim(2))?;
            anyhow::ensure!(held == ids.len(), "Session from_caches: a cache holds {held} positions, {} ids", ids.len());
        }
        Ok(Session { caches, ids })
    }

    /// The ids the caches hold, in position order.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The caches, one a layer: what a batch pool copies into a slot when
    /// this conversation joins it (`batch::Pool::adopt`).
    pub fn caches(&self) -> &[KvCache] {
        &self.caches
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Drop the conversation: fresh caches, as `/reset` did in the REPL.
    pub fn reset(&mut self, qwen: &Qwen3) {
        self.caches = qwen.fresh_caches();
        self.ids.clear();
    }

    /// Keep the first `n` positions and forget the rest. Every cache is cut to
    /// `n`, even at `n == ids.len()`: a feed or a launch that failed part way
    /// leaves caches ahead of the ids, and this is where they come back.
    pub fn truncate(&mut self, n: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n <= self.ids.len(), "Session truncate: {n} positions asked, {} held", self.ids.len());
        for c in self.caches.iter_mut() {
            if positions_held(c)? != n {
                c.truncate(n)?;
            }
        }
        self.ids.truncate(n);
        Ok(())
    }

    /// Keep the longest prefix of the cache that `prompt` starts with, and
    /// return how many positions were kept. At most `prompt.len() - 1`: the
    /// last prompt token is always fed again, because the next token is read
    /// from its hidden state and the cache keeps no hidden state.
    pub fn reuse_prefix(&mut self, prompt: &[u32]) -> anyhow::Result<usize> {
        let keep = common_prefix(&self.ids, prompt).min(prompt.len().saturating_sub(1));
        self.truncate(keep)?;
        Ok(keep)
    }

    /// Feed `ids` after what is held, in chunks of at most
    /// [`Qwen3::prefill_rows`], and return the hidden states of the last
    /// chunk. One chunk is one `hidden_cached` call. The chunk is the
    /// model's, not [`crate::model::MAX_PREFILL_ROWS`]: the served Metal path
    /// takes 256 rows a call, and a turn of 257 tokens or more was refused
    /// there until 2026-11-07.
    pub fn feed(&mut self, qwen: &Qwen3, ids: &[u32]) -> anyhow::Result<Tensor> {
        anyhow::ensure!(!ids.is_empty(), "Session feed: nothing to feed");
        let dev = qwen.device();
        let mut h = None;
        for part in ids.chunks(qwen.prefill_rows()) {
            let x = Tensor::from_slice(part, (1, part.len()), dev)?;
            drop(h.take());
            h = Some(qwen.hidden_cached(&x, self.ids.len(), &mut self.caches, &mut NoCapture)?);
            self.ids.extend_from_slice(part);
        }
        Ok(h.expect("ids is not empty"))
    }

    /// Feed `ids` after what is held through the prefill `dec`'s config
    /// names: the chain prefill ([`Decoder::chain_prefill`]) returns the first
    /// answer token, the eager one ([`Self::feed`]) the hidden states it is
    /// read from. Either goes to [`Self::answer_fed`].
    pub fn feed_with(&mut self, qwen: &Qwen3, dec: &mut Decoder, ids: &[u32]) -> anyhow::Result<Fed> {
        if !dec.has_chain() {
            return Ok(Fed::Hidden(self.feed(qwen, ids)?));
        }
        anyhow::ensure!(!ids.is_empty(), "Session feed: nothing to feed");
        let first = dec.chain_prefill(qwen, &mut self.caches, self.ids.len(), ids)?;
        self.ids.extend_from_slice(ids);
        Ok(Fed::First(first))
    }

    /// The answer after a prefill whose hidden states are `h`.
    ///
    /// `on_tokens` receives the whole answer so far after each eager step and
    /// after each 4E launch, and may cancel. It is not called with an empty
    /// answer. A stop token is never in the answer.
    pub fn answer(
        &mut self,
        qwen: &Qwen3,
        dec: &mut Decoder,
        h: Tensor,
        gen: Generate<'_>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> anyhow::Result<Answer> {
        self.answer_fed(qwen, dec, Fed::Hidden(h), gen, on_tokens)
    }

    /// [`Self::answer`] after either prefill ([`Self::feed_with`]). A first
    /// token from the chain is the pass's greedy pick, so it is refused beside
    /// the eager decode and beside a sampler above temperature 0.
    pub fn answer_fed(
        &mut self,
        qwen: &Qwen3,
        dec: &mut Decoder,
        fed: Fed,
        gen: Generate<'_>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> anyhow::Result<Answer> {
        let Generate { sampler, max_new, stops } = gen;
        anyhow::ensure!(
            !dec.is_persistent() || sampler.temp <= 0.0,
            "the persistent decode is greedy: it runs at temperature 0, not {}",
            sampler.temp
        );
        anyhow::ensure!(
            matches!(fed, Fed::Hidden(_)) || (dec.is_persistent() && sampler.temp <= 0.0),
            "a first token from the chain prefill is greedy and hands over to 4E: this engine is the eager decode or \
             samples at temperature {}",
            sampler.temp
        );
        let mut said: Vec<u32> = Vec::new();
        if max_new == 0 {
            return Ok(Answer { said, finish: Finish::Length });
        }
        if dec.is_persistent() {
            let first = match fed {
                Fed::Hidden(h) => sampler.pick(&qwen.logits_last(&h)?.i((0, 0))?)?,
                Fed::First(t) => t,
            };
            let launch_tokens = dec.launch_tokens() as usize;
            return self.answer_launches(first, max_new, stops, launch_tokens, on_tokens, |caches, last, offset, k| {
                dec.continue_greedy(qwen, caches, last, offset, k, stops)
            });
        }
        let Fed::Hidden(mut h) = fed else {
            anyhow::bail!("a first token from the chain prefill beside the eager decode");
        };
        loop {
            let next = sampler.pick(&qwen.logits_last(&h)?.i((0, 0))?)?;
            if stops.contains(&next) {
                return Ok(Answer { said, finish: Finish::Stop });
            }
            said.push(next);
            if on_tokens(&said) == Flow::Cancel {
                return Ok(Answer { said, finish: Finish::Cancelled });
            }
            if said.len() >= max_new {
                return Ok(Answer { said, finish: Finish::Length });
            }
            h = self.feed(qwen, &[next])?;
        }
    }

    /// The persistent branch of [`Self::answer_fed`], from its first token.
    /// `launch(caches, last, offset, k)` is one 4E launch: it feeds `last` at
    /// `offset`, where the caches stop, writes its positions in place, and
    /// returns at most `k` tokens and whether a stop token ended them, as
    /// [`Decoder::continue_greedy`] does. This method keeps the ids and the
    /// cache lengths across launches. Public so a host test can drive it with
    /// a launch of its own.
    pub fn answer_launches(
        &mut self,
        first: u32,
        max_new: usize,
        stops: &[u32],
        launch_tokens: usize,
        on_tokens: &mut dyn FnMut(&[u32]) -> Flow,
        mut launch: impl FnMut(&mut [KvCache], u32, usize, usize) -> anyhow::Result<(Vec<u32>, bool)>,
    ) -> anyhow::Result<Answer> {
        let mut said: Vec<u32> = Vec::new();
        if max_new == 0 {
            return Ok(Answer { said, finish: Finish::Length });
        }
        if stops.contains(&first) {
            return Ok(Answer { said, finish: Finish::Stop });
        }
        said.push(first);
        if on_tokens(&said) == Flow::Cancel {
            return Ok(Answer { said, finish: Finish::Cancelled });
        }
        let mut last = first;
        while said.len() < max_new {
            let k = (max_new - said.len()).min(launch_tokens);
            let offset = self.ids.len();
            let (made, stopped) = launch(self.caches.as_mut_slice(), last, offset, k)?;
            // Fed: `last`, then every token made but the final one, or all
            // of them when a stop token ended the launch. 4E stops on the
            // first stop token only, so a second one leaves the cache past
            // the answer: cut it back, as the REPL did.
            self.ids.push(last);
            match stopped {
                true => self.ids.extend_from_slice(&made),
                false => self.ids.extend_from_slice(&made[..made.len().saturating_sub(1)]),
            }
            for c in self.caches.iter_mut() {
                c.set_len(self.ids.len())?;
            }
            said.extend_from_slice(&made);
            if stopped {
                if !made.is_empty() {
                    on_tokens(&said);
                }
                return Ok(Answer { said, finish: Finish::Stop });
            }
            last = match made.last() {
                Some(&t) => t,
                None => return Ok(Answer { said, finish: Finish::Stop }),
            };
            if on_tokens(&said) == Flow::Cancel {
                // The last token made is not fed: the cache stops where
                // the next launch would have started.
                return Ok(Answer { said, finish: Finish::Cancelled });
            }
        }
        Ok(Answer { said, finish: Finish::Length })
    }

    /// Positions the next feed of `n` tokens and an answer of `max_new` would
    /// end at, against `limit`. See [`crate::repl::turn_fits`].
    pub fn fits(&self, n: usize, max_new: usize, limit: usize) -> bool {
        crate::repl::turn_fits(self.ids.len(), n, max_new, limit)
    }
}
