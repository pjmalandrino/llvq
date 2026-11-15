//! The decode engine a served config names ([`crate::served::Decode`]): the
//! eager path, or 4E (`crate::persist4e_cuda`).
//!
//! The prefill is the eager path's under both engines, in chunks of at most
//! [`crate::model::MAX_PREFILL_ROWS`] rows, as `Qwen3::generate_timed` sends
//! it, unless the config names `"prefill": "chain"`: the prompt then goes
//! through 4E's tree verify pass ([`Decoder::chain_prefill`]), and the
//! decode that follows is 4E's unchanged. Under `Persistent` the decode then runs in 4E launches of at most
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

/// The positions 4E's attention scratch is sized for, at most: a cache whose
/// capacity passes it is refused at the launch (`persist4e_cuda::MAX_CAP` is
/// this constant). Here rather than in that module, which only a CUDA build
/// on Linux compiles, so a harness checks its prompts against it on any host.
/// A whole number of `kvq::GROW_CHUNK` chunks, so `n` positions fit iff
/// `n <= P4E_MAX_CAP`.
pub const P4E_MAX_CAP: usize = 8192 + 512;

/// The engine, built once per process from the served config.
pub struct Decoder {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    rt: Option<Box<crate::persist4e_cuda::Runtime4e>>,
    lines: Vec<String>,
    /// A position cap on the eager path, for the tests of a caller that must
    /// hand a sequence over before 4E's cap ([`Decoder::eager_capped`]).
    cap: Option<usize>,
    /// The chain prefill's rows a launch, when the config names
    /// `"prefill": "chain"`: the tree arm is the runtime's arm 0.
    chain: Option<usize>,
}

impl Decoder {
    /// The eager path: what every binary did before the field existed.
    pub fn eager() -> Self {
        Decoder {
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            rt: None,
            lines: Vec::new(),
            cap: None,
            chain: None,
        }
    }

    /// Whether a prompt is prefilled through the tree pass
    /// ([`crate::served::Prefill::Chain`]) instead of the eager path.
    pub fn has_chain(&self) -> bool {
        self.chain.is_some()
    }

    /// The chain prefill of `ids` after the `pos0` positions `caches` hold,
    /// then the hand-over: the caches hold `pos0 + ids.len()` positions and
    /// the first answer token is returned, not yet in them, as
    /// [`Self::continue_greedy`] takes it. A fresh cache gets its Grow buffers
    /// first (the tree pass and 4E write in place). Refused on a decoder whose
    /// config does not name the chain.
    pub fn chain_prefill(&mut self, model: &Qwen3, caches: &mut [KvCache], pos0: usize, ids: &[u32]) -> anyhow::Result<u32> {
        let rows = self.chain.ok_or_else(|| anyhow::anyhow!("chain prefill: this engine's config names the eager prefill"))?;
        anyhow::ensure!(!ids.is_empty(), "chain prefill: nothing to feed");
        make_grow_buffers(model, caches)?;
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let Some(rt) = self.rt.as_mut() {
            return Ok(rt.chain_prefill(0, caches, pos0, ids, rows)?.first);
        }
        let _ = (rows, pos0);
        anyhow::bail!("chain prefill: the tree pass is a cooperative CUDA launch, and this engine holds no 4E runtime")
    }

    /// The prompt through this engine's prefill into fresh caches: the chain
    /// when the config names it, else the eager path in chunks of `chunk`.
    /// Returns the caches, the first answer token and the prefill's seconds.
    pub fn prefill_fresh(&mut self, model: &Qwen3, tokens: &[u32], chunk: usize) -> anyhow::Result<(Vec<KvCache>, u32, f64)> {
        let mut caches = model.fresh_caches();
        let t0 = std::time::Instant::now();
        let first = match self.has_chain() {
            true => self.chain_prefill(model, &mut caches, 0, tokens)?,
            false => model.prefill_next_in(tokens, chunk, &mut caches)?,
        };
        // Both branches end on a host read of the token (the chain syncs the
        // stream after its hand-over), so the clock reads a finished prefill.
        Ok((caches, first, t0.elapsed().as_secs_f64()))
    }

    /// The reference of a chain config's door: this engine's prefill, then
    /// the EAGER decode on the caches it left, one step a token. 4E owes these
    /// tokens bit for bit: the decode after the chain is 4E v9 unchanged.
    pub fn generate_eager_decode(&mut self, model: &Qwen3, tokens: &[u32], max_new: usize, chunk: usize) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(!tokens.is_empty() && max_new > 0, "generate_eager_decode: nothing to generate");
        let (mut caches, first, _) = self.prefill_fresh(model, tokens, chunk)?;
        let (rest, _) = chunked_greedy(first, tokens.len(), max_new - 1, &[], 1, |pos, tok, _| {
            let logits = model.step_logits(tok, pos, &mut caches)?;
            Ok(vec![logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?])
        })?;
        let mut out = vec![first];
        out.extend(rest);
        Ok(out)
    }

    /// The eager path, reporting `positions` as its cap the way 4E reports
    /// [`P4E_MAX_CAP`]: a CPU test of a caller that must stop decoding a
    /// sequence on this path at the cap (the batch engine's hand-over).
    pub fn eager_capped(positions: usize) -> Self {
        Decoder { cap: Some(positions), ..Self::eager() }
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

    /// The positions this engine decodes to, at most: `None` for the eager
    /// path, [`P4E_MAX_CAP`] for 4E. See [`Decode::max_positions`].
    pub fn max_positions(&self) -> Option<usize> {
        let engine = match self.is_persistent() {
            true => Decode::Persistent,
            false => Decode::Eager,
        };
        engine.max_positions().or(self.cap)
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

    /// Install 4E's verify pass on this engine (`Runtime4e::install_verify`),
    /// so it can be the target of [`Self::continue_speculative`]. Refused on
    /// the eager engine and below row form 7. Its install lines are appended
    /// to [`Self::lines`].
    /// The 4E runtime, for a measurement binary that drives its launches
    /// directly (specrun's phase stamps). None on the eager path.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn runtime_mut(&mut self) -> Option<&mut crate::persist4e_cuda::Runtime4e> {
        self.rt.as_deref_mut()
    }

    pub fn install_verify(&mut self) -> anyhow::Result<()> {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let Some(rt) = self.rt.as_mut() {
            let before = rt.lines.len();
            rt.install_verify()?;
            self.lines.extend(rt.lines.drain(before..));
            return Ok(());
        }
        anyhow::bail!("the speculative decode's verify pass is a 4E launch: this engine is the eager path")
    }

    /// The greedy continuation of [`Self::continue_greedy`], with `draft`
    /// (the drafter's engine, its own model and caches) proposing up to `k`
    /// tokens a round and this engine checking them in one verify pass
    /// ([`crate::speculative::speculative_greedy`]). Both caches hold `offset`
    /// positions and neither holds `first`. The tokens are this engine's
    /// greedy tokens: each verify row is its 4E token bit for bit. `k + 1`
    /// rows must fit the verify pass (`pers::P4V_KMAX`).
    #[allow(clippy::too_many_arguments)]
    pub fn continue_speculative(
        &mut self,
        caches: &mut [KvCache],
        draft: &mut Decoder,
        draft_caches: &mut [KvCache],
        first: u32,
        offset: usize,
        more: usize,
        stops: &[u32],
        k: usize,
    ) -> anyhow::Result<(Vec<u32>, bool, crate::speculative::SpecStats)> {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let (Some(rt), Some(drt)) = (self.rt.as_mut(), draft.rt.as_mut()) {
            anyhow::ensure!(
                k < llvq_cuda::pers::P4V_KMAX as usize,
                "speculative decode: {k} drafts a round, the verify pass carries {} rows",
                llvq_cuda::pers::P4V_KMAX
            );
            let eos = stops.first().copied().unwrap_or(u32::MAX);
            let t = std::cell::RefCell::new((rt, caches));
            let d = std::cell::RefCell::new((drt, draft_caches));
            return crate::speculative::speculative_greedy(
                first,
                offset,
                more,
                stops,
                k,
                |pos, tok, n| {
                    let mut g = d.borrow_mut();
                    let (rt, c) = &mut *g;
                    rt.decode(c, pos, tok, n, eos)
                },
                |pos, rows| {
                    let mut g = t.borrow_mut();
                    let (rt, c) = &mut *g;
                    rt.verify(c, pos, rows)
                },
                |side, len| {
                    let set = |c: &mut [KvCache]| -> anyhow::Result<()> {
                        for x in c.iter_mut() {
                            x.set_len(len)?;
                        }
                        Ok(())
                    };
                    match side {
                        crate::speculative::Side::Target => set(t.borrow_mut().1),
                        crate::speculative::Side::Draft => set(d.borrow_mut().1),
                    }
                },
            );
        }
        let _ = (caches, draft, draft_caches, first, offset, more, stops, k);
        anyhow::bail!("the speculative decode needs 4E on both the target and the drafter")
    }

    /// The greedy continuation by prompt lookup (`speculative::lookup_greedy`):
    /// n-grams of `prompt` and of the text so far draft up to `k` tokens a
    /// round, 4E's verify pass checks them, and every token is the target's
    /// greedy token. No drafter, no memory. `caches` hold `prompt`.
    #[allow(clippy::too_many_arguments)]
    pub fn continue_lookup(
        &mut self,
        caches: &mut [KvCache],
        prompt: &[u32],
        first: u32,
        more: usize,
        stops: &[u32],
        k: usize,
        policy: crate::speculative::LookupPolicy,
    ) -> anyhow::Result<(Vec<u32>, bool, crate::speculative::SpecStats)> {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let Some(rt) = self.rt.as_mut() {
            anyhow::ensure!(
                k < llvq_cuda::pers::P4V_KMAX as usize,
                "prompt lookup: {k} drafts a round, the verify pass carries {} rows",
                llvq_cuda::pers::P4V_KMAX
            );
            let t = std::cell::RefCell::new((rt, caches));
            return crate::speculative::lookup_greedy(
                prompt,
                first,
                more,
                stops,
                k,
                policy,
                |pos, rows| {
                    let mut g = t.borrow_mut();
                    let (rt, c) = &mut *g;
                    // A round with no draft is one greedy step: 4E's token
                    // launch, the served symbol, not a verify pass of one row.
                    if rows.len() == 1 { rt.decode(c, pos, rows[0], 1, u32::MAX) } else { rt.verify(c, pos, rows) }
                },
                |len| {
                    for x in t.borrow_mut().1.iter_mut() {
                        x.set_len(len)?;
                    }
                    Ok(())
                },
            );
        }
        let _ = (caches, prompt, first, more, stops, k, policy);
        anyhow::bail!("prompt lookup needs 4E and its verify pass on the target")
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
        // The chain prefill when the config names it; the eager one otherwise,
        // the same call as before the field existed.
        let (mut caches, first, prompt_s) = self.prefill_fresh(model, tokens, chunk)?;
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
    /// The served door, whole: [`Served::apply`] (the device and object
    /// checks, the store, the attention form, its threshold and its kernel),
    /// then [`Served::decoder`]. What `bin/gsm8k` and `bin/chat` call on the
    /// model a served config loaded, so neither can print a config's
    /// provenance and run another form.
    pub fn serve(&self, model: &mut Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
        self.apply(model, device).map_err(anyhow::Error::msg)?;
        self.decoder(model, device)
    }

    /// The engine this config names, for `model` on `device`. Call after
    /// [`Served::apply`], which puts the eager path on the same threshold.
    /// `Persistent` installs 4E at 512 threads, the geometry its card readings
    /// ran (the driver's ptxas spills the 768-thread symbol), and checks RoPE
    /// against candle on the card ([`crate::persist4e_cuda::install`]).
    pub fn decoder(&self, model: &Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
        if self.decode_cache == crate::served::DecodeCache::Int8 {
            anyhow::bail!(
                "{}: decode_cache \"int8\" at a served door. Its token symbol (arm v9k8) holds 8 B of local memory \
                 at 128 registers on the L40S, which 4E's load rule refuses (pers::p4e_token_symbol_loads); whether a \
                 served module may hold local memory is the operator's decision. This config is NOT SERVED: \
                 bin/prefillbench and bin/gsm8k measure it through Served::decoder_measurement.",
                self.path.display()
            );
        }
        self.build_decoder(model, device)
    }

    /// [`Self::decoder`] for a measurement binary (`bin/prefillbench`,
    /// `bin/gsm8k`): the same engine, and under `decode_cache: int8` arm v9k8
    /// installed with the local-memory half of 4E's load rule waived
    /// (`persist4e_cuda::install_e1_local`, the `+local` measurement mode),
    /// the binary16 caches released at each hand-over. No served door calls
    /// it; its install lines say MEASUREMENT ONLY.
    pub fn decoder_measurement(&self, model: &Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
        self.build_decoder(model, device)
    }

    /// [`Self::serve`] through [`Self::decoder_measurement`].
    pub fn serve_measurement(&self, model: &mut Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
        self.apply(model, device).map_err(anyhow::Error::msg)?;
        self.decoder_measurement(model, device)
    }

    fn build_decoder(&self, model: &Qwen3, device: &candle_core::Device) -> anyhow::Result<Decoder> {
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
                    let mut rt = match self.decode_cache {
                        crate::served::DecodeCache::F16 => crate::persist4e_cuda::install_rows(
                            model,
                            device,
                            LAUNCH_TOKENS,
                            Some(512),
                            self.rows.form(),
                        )?,
                        crate::served::DecodeCache::Int8 => {
                            let mut rt = crate::persist4e_cuda::install_e1_local(model, device, LAUNCH_TOKENS, "v9k8")?;
                            rt.set_k8_release(true)?;
                            rt.lines.push(format!(
                                "MEASUREMENT ONLY ({}): decode cache int8 alone, arm v9k8 with {} B of local memory in its token \
                                 symbol, the binary16 caches released at each hand-over",
                                self.path.display(),
                                rt.local_bytes
                            ));
                            rt
                        }
                    };
                    rt.set_served_kernel_from(0);
                    let chain = match self.prefill_form {
                        crate::served::Prefill::Eager => None,
                        crate::served::Prefill::Chain => {
                            install_chain(&mut rt)?;
                            Some(crate::served::CHAIN_ROWS)
                        }
                    };
                    let lines = std::mem::take(&mut rt.lines);
                    Ok(Decoder { rt: Some(Box::new(rt)), lines, cap: None, chain })
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

/// The chain prefill's tree arm on a 4E runtime: the compiled-module cache
/// from `LLVQ_MODULE_CACHE` (`llvq_cuda::modcache::dir_from`), then
/// [`crate::served::CHAIN_ARM`] as the runtime's only tree arm, refused when
/// its module did not build.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn install_chain(rt: &mut crate::persist4e_cuda::Runtime4e) -> anyhow::Result<()> {
    use crate::persist4e_cuda::{TreeForm, TreePlan};
    let dir = llvq_cuda::modcache::dir_from(|v| std::env::var(v).ok()).map_err(anyhow::Error::msg)?;
    rt.lines.push(match &dir {
        Some(d) => format!("chain prefill: compiled-module cache {}", d.0.display()),
        None => format!("chain prefill: compiled-module cache off ({}=off), the cubin compiled at every start", llvq_cuda::modcache::ENV),
    });
    rt.set_tree_modules(crate::persist4e_cuda::TreeModules::Cubin(dir));
    rt.install_tree(&[(crate::served::CHAIN_ARM, TreePlan::Explicit(crate::served::CHAIN_PLAN), TreeForm::A12ipg32Ft)])?;
    let arms = rt.tree_arms();
    let a = arms.first().ok_or_else(|| anyhow::anyhow!("chain prefill: no tree arm installed"))?;
    anyhow::ensure!(a.void.is_none(), "chain prefill: arm {} is void: {:?}", a.name, a.void);
    anyhow::ensure!(
        a.form.rmax() as usize >= crate::served::CHAIN_ROWS,
        "chain prefill: arm {} carries {} rows a launch, the config's chain {}",
        a.name,
        a.form.rmax(),
        crate::served::CHAIN_ROWS
    );
    rt.lines.push(format!(
        "chain prefill: arm {} ({:?}, slices {:?}, {} threads, {} B shared), {} rows a launch",
        a.name,
        a.form,
        a.splits,
        a.threads,
        a.smem,
        crate::served::CHAIN_ROWS
    ));
    Ok(())
}

/// Grow buffers on every cache that has none (a fresh cache, `len` 0): one
/// zero position appended through the store, then the length set back to 0.
/// The tree pass and 4E write the buffers in place and cannot make them.
fn make_grow_buffers(model: &Qwen3, caches: &mut [KvCache]) -> anyhow::Result<()> {
    let cfg = model.config();
    for c in caches.iter_mut() {
        anyhow::ensure!(
            matches!(c.store(), crate::kvq::KvStore::Grow(_)),
            "chain prefill: a cache on the {} store, not Grow: the tree pass and 4E write the Grow buffers in place",
            c.store().label()
        );
        if c.grow_buffers().is_some() {
            continue;
        }
        anyhow::ensure!(c.is_empty(), "chain prefill: a cache holds {} positions and no Grow buffer", c.len());
        let z = candle_core::Tensor::zeros((1, cfg.num_key_value_heads, 1, cfg.head_dim), model.dtype(), model.device())?;
        c.restore(&z, &z)?;
        c.set_len(0)?;
    }
    Ok(())
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

    /// Every served door refuses the int8 decode cache by name, before it
    /// looks at the model or the build: the local-memory rule is the
    /// operator's. The binary16 config goes on to its own checks.
    #[test]
    fn the_served_door_refuses_the_int8_cache() {
        let m = crate::model::tests_support::tiny_model_kernel();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
        let read = |name: &str| {
            let path = dir.join(name);
            crate::served::Served::of(crate::served::Served::read_file(&path).unwrap(), path).unwrap()
        };
        let k = read("qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json");
        let e = k.decoder(&m, &candle_core::Device::Cpu).err().expect("refused at the door").to_string();
        assert!(e.contains("decode_cache \"int8\" at a served door") && e.contains("local memory"), "{e}");
        // the measurement door and the binary16 config pass the int8 check and stop
        // later, at the threshold (the toy was never applied)
        for (cfg, name) in [(&k, "measurement"), (&read("qwen3-4b-tetra-e4-4e-v9.json"), "v9")] {
            let r = match name {
                "measurement" => cfg.decoder_measurement(&m, &candle_core::Device::Cpu),
                _ => cfg.decoder(&m, &candle_core::Device::Cpu),
            };
            let e = r.err().expect("the toy is not applied").to_string();
            assert!(e.contains("Served::apply first"), "{name}: {e}");
        }
    }

    /// The chain prefill's caches: a fresh Grow cache gets its two buffers
    /// and holds 0 positions, a cache that has them is left alone, and a
    /// cache that holds positions with no Grow buffer is refused.
    #[test]
    fn fresh_caches_get_their_grow_buffers_and_hold_nothing() {
        let mut m = crate::model::tests_support::tiny_model_kernel();
        m.set_kv_store(crate::kvq::KvStore::Grow(256));
        let mut caches = m.fresh_caches();
        assert!(caches.iter().all(|c| c.grow_buffers().is_none()));
        make_grow_buffers(&m, &mut caches).unwrap();
        for c in &caches {
            let (k, v) = c.grow_buffers().expect("buffers made");
            assert_eq!(k.dims4().unwrap(), (1, 2, 256, 128));
            assert_eq!(v.dims4().unwrap(), (1, 2, 256, 128));
            assert_eq!(c.len(), 0, "the zero position is not held");
        }
        // a second call keeps the same buffers and the positions held
        caches[0].set_len(3).unwrap();
        let before = caches[0].grow_buffers().unwrap().0.dims4().unwrap();
        make_grow_buffers(&m, &mut caches).unwrap();
        assert_eq!((caches[0].len(), caches[0].grow_buffers().unwrap().0.dims4().unwrap()), (3, before));
        // a released cache (the int8 form's hand-over) holds nothing, keeps its
        // store, and takes new buffers for the next prompt
        caches[0].release();
        assert!(caches[0].grow_buffers().is_none() && caches[0].is_empty());
        assert_eq!(caches[0].store(), crate::kvq::KvStore::Grow(256));
        make_grow_buffers(&m, &mut caches).unwrap();
        assert_eq!((caches[0].len(), caches[0].grow_buffers().map(|b| b.0.dim(2).unwrap())), (0, Some(256)));
        // a Grow cache holding positions with no buffer cannot exist through the
        // store's own writes; the Cat store has no buffer to write in place
        let mut cat = vec![crate::model::KvCache::new(crate::kvq::KvMode::F16)];
        let e = make_grow_buffers(&m, &mut cat).unwrap_err();
        assert!(e.to_string().contains("not Grow"), "{e}");
    }
}
