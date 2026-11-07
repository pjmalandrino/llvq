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
}
