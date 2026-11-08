//! Greedy speculative decoding: a drafter proposes up to `k` tokens, the
//! target checks them in one verify pass, and the output is the target's
//! greedy output token for token (`docs/plan-speculation.md`).
//!
//! The target's tokens do not depend on the drafter. A verify pass returns,
//! for each of its rows, the target's argmax after that row; row 0 is the
//! pending token, rows 1.. the drafts. The leading drafts that equal the
//! target's own choice are kept, then the target's token after the last kept
//! row: the greedy loop's next `a + 1` tokens. This holds when each verify row
//! is the target's single-token step bit for bit, which the 4E verify pass
//! owes (`llvq-cuda/tests/pers_matches_served.rs`), and when both caches hold
//! exactly the kept prefix before every round, which this loop keeps:
//!
//! - the target's caches hold `pos + rows` after a verify; they are rewound to
//!   `pos + a + 1`, the pending token and the `a` kept drafts;
//! - the drafter's caches hold `pos + drafts` after drafting (it fed the
//!   pending token and all drafts but the last). Rewound to `pos + a + 1` when
//!   a draft was refused; when every draft was kept it lacks the last one, which
//!   it is fed at the next round before drafting (a catch-up step whose output
//!   is not used).
//!
//! [`speculative_greedy`] is the loop alone, pure over three closures, so the
//! host test holds it against the plain greedy loop on toy models whose next
//! token depends on their whole cache: a stale position fails it.

/// What the loop did, for the report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Verify passes.
    pub rounds: usize,
    /// Tokens the drafter proposed.
    pub drafted: usize,
    /// Proposed tokens the target kept.
    pub accepted: usize,
    /// Drafter steps fed only to catch its cache up after a round that kept
    /// every draft.
    pub catchups: usize,
    /// Drafter launches that proposed tokens.
    pub draft_launches: usize,
    /// Wall time inside the drafter's launches (catch-ups included) and inside
    /// the verify passes, in ns. Each launch reads its result back, so each is
    /// synchronous and these add up to the decode less the host's own work.
    pub draft_ns: u64,
    pub verify_ns: u64,
}

impl SpecStats {
    /// Kept over proposed.
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// Tokens emitted a verify pass (kept drafts plus the target's own).
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.accepted + self.rounds) as f64 / self.rounds as f64
        }
    }
}

/// Whose caches a rewind sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Target,
    Draft,
}

/// The greedy continuation of `first` (chosen, in neither model's caches;
/// both hold `offset` positions): up to `more` tokens after it, cut before
/// the first token of `stops`. Returns the tokens, whether a stop token ended
/// them, and the loop's counts.
///
/// - `draft(pos, tok, n)` feeds the drafter `tok` at `pos` and returns the
///   next tokens it chooses greedily, 1 to `n` of them (fewer only after its
///   stop token); its caches then hold `pos` plus their count.
/// - `verify(pos, rows)` runs the target on `rows` at `pos..` in one pass and
///   returns its argmax after each row; its caches then hold `pos + rows.len()`.
/// - `rewind(side, len)` sets that model's caches to their first `len`
///   positions.
///
/// `k` is the most drafts a round, 0 for the plain greedy loop on the verify
/// pass; `k + 1` rows must fit the verify pass.
#[allow(clippy::too_many_arguments)]
pub fn speculative_greedy(
    first: u32,
    offset: usize,
    more: usize,
    stops: &[u32],
    k: usize,
    mut draft: impl FnMut(usize, u32, u32) -> anyhow::Result<Vec<u32>>,
    mut verify: impl FnMut(usize, &[u32]) -> anyhow::Result<Vec<u32>>,
    mut rewind: impl FnMut(Side, usize) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, bool, SpecStats)> {
    let mut st = SpecStats::default();
    let mut out = Vec::with_capacity(more);
    let (mut tok, mut pos) = (first, offset);
    // A token the drafter still has to be fed at a position, before it drafts.
    let mut lag: Option<(usize, u32)> = None;
    while out.len() < more {
        // Never draft past what may still be emitted: a round emits at most
        // its drafts plus one.
        let kk = k.min(more - out.len() - 1);
        let drafts = if kk > 0 {
            let t0 = std::time::Instant::now();
            if let Some((p, t)) = lag.take() {
                let _ = draft(p, t, 1)?;
                st.catchups += 1;
            }
            let d = draft(pos, tok, kk as u32)?;
            st.draft_ns += t0.elapsed().as_nanos() as u64;
            anyhow::ensure!(!d.is_empty() && d.len() <= kk, "a draft launch asked for {kk} tokens made {}", d.len());
            st.draft_launches += 1;
            d
        } else {
            Vec::new()
        };
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(tok);
        rows.extend_from_slice(&drafts);
        let t0 = std::time::Instant::now();
        let ys = verify(pos, &rows)?;
        st.verify_ns += t0.elapsed().as_nanos() as u64;
        anyhow::ensure!(ys.len() == rows.len(), "a verify pass of {} rows returned {} tokens", rows.len(), ys.len());
        st.rounds += 1;
        st.drafted += drafts.len();
        let a = drafts.iter().zip(&ys).take_while(|(d, y)| d == y).count();
        st.accepted += a;
        // ys[..a] are the kept drafts, ys[a] the target's next token.
        for &t in &ys[..=a] {
            if stops.contains(&t) {
                return Ok((out, true, st));
            }
            out.push(t);
            if out.len() == more {
                return Ok((out, false, st));
            }
        }
        rewind(Side::Target, pos + a + 1)?;
        if kk > 0 {
            if a < drafts.len() {
                rewind(Side::Draft, pos + a + 1)?;
            } else {
                // every draft kept: the drafter fed rows[..a], it lacks rows[a]
                lag = Some((pos + a, rows[a]));
            }
        } else if k > 0 {
            // A round with no draft emits one token, the last one asked for,
            // and has returned above; the drafter is never left behind.
            anyhow::bail!("speculative_greedy: a round with no draft continued (k = {k})");
        }
        tok = ys[a];
        pos += a + 1;
    }
    Ok((out, false, st))
}

/// Prompt lookup, the drafter with no model (Saxena 2023, "prompt lookup
/// decoding"): the tokens that followed the latest earlier occurrence of the
/// context's last n-gram, longest n first from `max_n` down to `min_n`, at
/// most `k` of them. Empty when no n-gram repeats. It costs no weight read and
/// no memory: what it buys depends only on how much the text repeats itself
/// (code edits, summaries, retrieval answers quote their input).
pub fn lookup_draft(ctx: &[u32], k: usize, min_n: usize, max_n: usize) -> Vec<u32> {
    let len = ctx.len();
    if k == 0 || min_n == 0 {
        return Vec::new();
    }
    for n in (min_n..=max_n.min(len.saturating_sub(1))).rev() {
        let tail = &ctx[len - n..];
        // the latest start i < len - n whose n tokens equal the tail
        for i in (0..len - n).rev() {
            if &ctx[i..i + n] == tail {
                let from = i + n;
                return ctx[from..(from + k).min(len)].to_vec();
            }
        }
    }
    Vec::new()
}

/// The n-gram lengths prompt lookup tries, longest first: 3 down to 1, the
/// usual setting of the method.
pub const LOOKUP_MIN_N: usize = 1;
pub const LOOKUP_MAX_N: usize = 3;

/// When prompt lookup drafts. `min_n`: the shortest n-gram that may draft
/// (from [`LOOKUP_MAX_N`] down to it). `probe`: a round drafts one token
/// until `probe` drafted tokens in a row were kept, then up to `k`; a round
/// that loses a draft goes back to one (0: always up to `k`). Neither moves a
/// token: every round's tokens are the target's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupPolicy {
    pub min_n: usize,
    pub probe: usize,
}

impl LookupPolicy {
    /// The method's usual setting: any n-gram from 3 down to 1, always `k`.
    pub const PLAIN: LookupPolicy = LookupPolicy { min_n: LOOKUP_MIN_N, probe: 0 };

    /// `lookup` (the plain setting) or `lookup-n<min_n>p<probe>`, as specrun
    /// names an arm.
    pub fn parse(word: &str) -> Option<LookupPolicy> {
        if word == "lookup" {
            return Some(Self::PLAIN);
        }
        let rest = word.strip_prefix("lookup-n")?;
        let (n, p) = rest.split_once('p')?;
        let (min_n, probe) = (n.parse().ok()?, p.parse().ok()?);
        (1..=LOOKUP_MAX_N).contains(&min_n).then_some(LookupPolicy { min_n, probe })
    }
}

/// [`speculative_greedy`] with prompt lookup as the drafter: the greedy
/// continuation of `first` after `prompt` (the target's caches hold
/// `prompt.len()` positions), up to `more` tokens, cut before a stop token.
/// A round drafts up to `k` tokens from the context (the prompt, `first` and
/// every token emitted), and a round with no match verifies the pending token
/// alone: one row, the greedy step. The drafter has no cache, so only the
/// target is rewound. `verify` and `rewind` are [`speculative_greedy`]'s.
/// `draft_ns` counts the host's lookups.
#[allow(clippy::too_many_arguments)]
pub fn lookup_greedy(
    prompt: &[u32],
    first: u32,
    more: usize,
    stops: &[u32],
    k: usize,
    policy: LookupPolicy,
    mut verify: impl FnMut(usize, &[u32]) -> anyhow::Result<Vec<u32>>,
    mut rewind: impl FnMut(usize) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, bool, SpecStats)> {
    let mut st = SpecStats::default();
    let mut out = Vec::with_capacity(more);
    let mut ctx = Vec::with_capacity(prompt.len() + more + 1);
    ctx.extend_from_slice(prompt);
    ctx.push(first);
    let (mut tok, mut pos) = (first, prompt.len());
    // drafted tokens kept in a row, for the probe
    let mut streak = 0usize;
    while out.len() < more {
        let mut kk = k.min(more - out.len() - 1);
        if policy.probe > 0 && streak < policy.probe {
            kk = kk.min(1);
        }
        let t0 = std::time::Instant::now();
        let drafts = lookup_draft(&ctx, kk, policy.min_n, LOOKUP_MAX_N);
        st.draft_ns += t0.elapsed().as_nanos() as u64;
        if !drafts.is_empty() {
            st.draft_launches += 1;
        }
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(tok);
        rows.extend_from_slice(&drafts);
        let t0 = std::time::Instant::now();
        let ys = verify(pos, &rows)?;
        st.verify_ns += t0.elapsed().as_nanos() as u64;
        anyhow::ensure!(ys.len() == rows.len(), "a verify pass of {} rows returned {} tokens", rows.len(), ys.len());
        st.rounds += 1;
        st.drafted += drafts.len();
        let a = drafts.iter().zip(&ys).take_while(|(d, y)| d == y).count();
        st.accepted += a;
        if !drafts.is_empty() {
            streak = if a == drafts.len() { streak + a } else { 0 };
        }
        for &t in &ys[..=a] {
            if stops.contains(&t) {
                return Ok((out, true, st));
            }
            out.push(t);
            ctx.push(t);
            if out.len() == more {
                return Ok((out, false, st));
            }
        }
        rewind(pos + a + 1)?;
        tok = ys[a];
        pos += a + 1;
    }
    Ok((out, false, st))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy model whose next token is a hash of its whole cache, the token
    /// fed and its position: a cache that keeps one wrong position changes
    /// every later token.
    #[derive(Clone)]
    struct Toy {
        cache: Vec<u32>,
        salt: u64,
        vocab: u64,
    }

    impl Toy {
        fn new(salt: u64, vocab: u64) -> Self {
            Toy { cache: Vec::new(), salt, vocab }
        }
        fn feed(&mut self, pos: usize, tok: u32) -> u32 {
            assert_eq!(self.cache.len(), pos, "fed at {pos} with {} cached", self.cache.len());
            self.cache.push(tok);
            let mut h = self.salt ^ 0x9e37_79b9_7f4a_7c15;
            for (i, &t) in self.cache.iter().enumerate() {
                h = (h ^ (t as u64 + 1)).wrapping_mul(0x100_0000_01b3).rotate_left(5) ^ i as u64;
            }
            // most of the mass on a function of the last token, so a drafter
            // with another salt agrees often but not always
            let base = (tok as u64 * 7 + pos as u64 % 3) % self.vocab;
            if h.is_multiple_of(4) { (h >> 8) as u32 % self.vocab as u32 } else { base as u32 }
        }
    }

    /// The target's greedy continuation, one token at a time.
    fn greedy(target: &Toy, first: u32, offset: usize, more: usize, stops: &[u32]) -> (Vec<u32>, bool) {
        let mut m = target.clone();
        let (mut tok, mut pos, mut out) = (first, offset, Vec::new());
        while out.len() < more {
            let t = m.feed(pos, tok);
            if stops.contains(&t) {
                return (out, true);
            }
            out.push(t);
            tok = t;
            pos += 1;
        }
        (out, false)
    }

    /// Both toys prefilled with the same prompt of `offset` tokens.
    fn prefilled(salt_t: u64, salt_d: u64, offset: usize) -> (Toy, Toy) {
        let (mut t, mut d) = (Toy::new(salt_t, 17), Toy::new(salt_d, 17));
        for p in 0..offset {
            let tok = (p as u32 * 5 + 3) % 17;
            t.feed(p, tok);
            d.feed(p, tok);
        }
        (t, d)
    }

    #[test]
    fn the_speculative_loop_is_the_greedy_loop() {
        for (salt_d, what) in [(1u64, "the same model as drafter"), (2, "another drafter"), (99, "a drafter that rarely agrees")] {
            for k in 0..=4usize {
                for more in [0usize, 1, 2, 3, 7, 30] {
                    for stops in [vec![], vec![5u32], vec![5, 11]] {
                        for first in [0u32, 4, 13] {
                            let offset = 6;
                            let (target, drafter) = prefilled(1, salt_d, offset);
                            let want = greedy(&target, first, offset, more, &stops);
                            let (t, d) = (std::cell::RefCell::new(target.clone()), std::cell::RefCell::new(drafter));
                            let eos = stops.first().copied();
                            let got = speculative_greedy(
                                first,
                                offset,
                                more,
                                &stops,
                                k,
                                |pos, tok, n| {
                                    let mut m = d.borrow_mut();
                                    let (mut tok, mut pos, mut made) = (tok, pos, Vec::new());
                                    while made.len() < n as usize {
                                        tok = m.feed(pos, tok);
                                        pos += 1;
                                        made.push(tok);
                                        if Some(tok) == eos {
                                            break;
                                        }
                                    }
                                    // 4E's drafter: the caches hold every token it was fed
                                    Ok(made)
                                },
                                |pos, rows| {
                                    assert!(rows.len() <= k + 1, "{} rows at k = {k}", rows.len());
                                    let mut m = t.borrow_mut();
                                    Ok(rows.iter().enumerate().map(|(i, &r)| m.feed(pos + i, r)).collect())
                                },
                                |side, len| {
                                    let mut m = match side {
                                        Side::Target => t.borrow_mut(),
                                        Side::Draft => d.borrow_mut(),
                                    };
                                    assert!(len <= m.cache.len(), "{side:?}: a rewind to {len} past {}", m.cache.len());
                                    m.cache.truncate(len);
                                    Ok(())
                                },
                            )
                            .unwrap();
                            assert_eq!((got.0.clone(), got.1), want, "{what}, k {k}, {more} more, stops {stops:?}, first {first}");
                            let s = got.2;
                            assert!(s.accepted <= s.drafted, "{s:?}");
                            if salt_d == 1 && k > 0 && more > 1 && !got.1 {
                                assert_eq!(s.accepted, s.drafted, "the target as its own drafter keeps every draft: {s:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_loop_counts_and_refusals() {
        // a verify pass that returns the wrong count is refused
        let e = speculative_greedy(1, 0, 5, &[], 2, |_, _, n| Ok(vec![3; n as usize]), |_, rows| Ok(vec![3; rows.len() + 1]), |_, _| Ok(()))
            .unwrap_err();
        assert!(e.to_string().contains("returned"), "{e}");
        // a drafter that proposes nothing is refused
        let e = speculative_greedy(1, 0, 5, &[], 2, |_, _, _| Ok(vec![]), |_, rows| Ok(vec![3; rows.len()]), |_, _| Ok(())).unwrap_err();
        assert!(e.to_string().contains("made 0"), "{e}");
        // every draft kept: k + 1 tokens a round, one catch-up a round after the first
        let (out, stopped, s) =
            speculative_greedy(1, 0, 9, &[], 2, |_, _, n| Ok(vec![3; n as usize]), |_, rows| Ok(vec![3; rows.len()]), |_, _| Ok(())).unwrap();
        assert_eq!((out.len(), stopped), (9, false));
        assert_eq!((s.rounds, s.drafted, s.accepted, s.catchups, s.draft_launches), (3, 6, 6, 2, 3));
        assert_eq!(s.tokens_per_round(), 3.0);
    }

    #[test]
    fn lookup_drafts_the_latest_continuation_of_the_longest_ngram() {
        // ... 1 2 3 9 ... 1 2 3 7 8 ... 2 3 | the tail "2 3" last followed by 7 8
        let ctx = [1, 2, 3, 9, 5, 1, 2, 3, 7, 8, 4, 2, 3];
        assert_eq!(lookup_draft(&ctx, 2, 1, 3), vec![7, 8]);
        assert_eq!(lookup_draft(&ctx, 5, 1, 3), vec![7, 8, 4, 2, 3]);
        // the trigram "4 2 3" never repeats, the bigram does
        assert_eq!(lookup_draft(&ctx, 1, 3, 3), Vec::<u32>::new());
        // nothing repeats
        assert!(lookup_draft(&[1, 2, 3, 4], 3, 1, 3).is_empty());
        // a match at the end of the context: what follows is cut by the context
        assert_eq!(lookup_draft(&[5, 6, 5], 4, 1, 3), vec![6, 5]);
        assert!(lookup_draft(&ctx, 0, 1, 3).is_empty());
        assert!(lookup_draft(&[], 3, 1, 3).is_empty());
        assert!(lookup_draft(&[7], 3, 1, 3).is_empty());
        // the longest n-gram wins over a later match of a shorter one: the
        // bigram "1 2" last followed by 9, the unigram "2" last by 7
        let ctx = [1, 2, 9, 3, 2, 7, 1, 2];
        assert_eq!(lookup_draft(&ctx, 1, 1, 3), vec![9]);
        assert_eq!(lookup_draft(&ctx, 1, 1, 1), vec![7]);
    }

    #[test]
    fn the_lookup_loop_is_the_greedy_loop() {
        let mut some_kept = false;
        for policy in [LookupPolicy::PLAIN, LookupPolicy { min_n: 2, probe: 1 }, LookupPolicy { min_n: 3, probe: 2 }] {
        for k in 0..=4usize {
            for more in [0usize, 1, 2, 3, 7, 30, 60] {
                for stops in [vec![], vec![5u32], vec![5, 11]] {
                    for first in [0u32, 4, 13] {
                        let offset = 6;
                        let (target, _) = prefilled(1, 1, offset);
                        let prompt: Vec<u32> = target.cache.clone();
                        let want = greedy(&target, first, offset, more, &stops);
                        let t = std::cell::RefCell::new(target.clone());
                        let got = lookup_greedy(
                            &prompt,
                            first,
                            more,
                            &stops,
                            k,
                            policy,
                            |pos, rows| {
                                assert!(rows.len() <= k + 1, "{} rows at k = {k}", rows.len());
                                let mut m = t.borrow_mut();
                                Ok(rows.iter().enumerate().map(|(i, &r)| m.feed(pos + i, r)).collect())
                            },
                            |len| {
                                let mut m = t.borrow_mut();
                                assert!(len <= m.cache.len(), "a rewind to {len} past {}", m.cache.len());
                                m.cache.truncate(len);
                                Ok(())
                            },
                        )
                        .unwrap();
                        assert_eq!((got.0.clone(), got.1), want, "k {k}, {more} more, stops {stops:?}, first {first}");
                        let s = got.2;
                        assert!(s.accepted <= s.drafted && (k > 0 || s.drafted == 0), "{s:?}");
                        // a round emits its kept drafts and one token, less what a stop or `more` cut
                        assert!(s.rounds + s.accepted >= got.0.len(), "{s:?}");
                        if !got.1 {
                            assert_eq!(s.rounds + s.accepted, got.0.len(), "{s:?}");
                        }
                        some_kept |= s.accepted > 0;
                    }
                }
            }
        }
        }
        // the toy repeats itself (its next token is mostly a function of the last), so lookup keeps drafts
        assert!(some_kept, "no draft kept on a text that repeats");
    }

    #[test]
    fn the_lookup_policy_parses_and_probes() {
        assert_eq!(LookupPolicy::parse("lookup"), Some(LookupPolicy::PLAIN));
        assert_eq!(LookupPolicy::parse("lookup-n3p1"), Some(LookupPolicy { min_n: 3, probe: 1 }));
        assert_eq!(LookupPolicy::parse("lookup-n4p1"), None);
        assert_eq!(LookupPolicy::parse("lookup-n0p1"), None);
        assert_eq!(LookupPolicy::parse("lookup-n2"), None);
        assert_eq!(LookupPolicy::parse("lookupx"), None);
        // a text that copies itself: the probe drafts one token until one
        // was kept, then three; a lost draft goes back to one
        let prompt: Vec<u32> = (0..40).map(|i| i % 10).collect();
        let next = |pos: usize, _t: u32| ((pos + 1) % 10) as u32;
        let sizes = std::cell::RefCell::new(Vec::new());
        let (out, _, st) = lookup_greedy(
            &prompt,
            next(39, 0),
            30,
            &[],
            3,
            LookupPolicy { min_n: 3, probe: 1 },
            |pos, rows| {
                sizes.borrow_mut().push(rows.len());
                Ok((0..rows.len()).map(|i| next(pos + i, rows[i])).collect())
            },
            |_| Ok(()),
        )
        .unwrap();
        let want: Vec<u32> = (41..71).map(|p| (p % 10) as u32).collect();
        assert_eq!(out, want);
        let sizes = sizes.into_inner();
        assert_eq!(sizes[0], 2, "the first round probes one draft: {sizes:?}");
        assert!(sizes[1..sizes.len() - 1].iter().all(|&n| n == 4), "then three drafts a round: {sizes:?}");
        assert_eq!(st.drafted, st.accepted, "{st:?}");
    }
}
