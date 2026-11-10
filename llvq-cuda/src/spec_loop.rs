//! The speculative decode loop around the tree verify pass
//! (`kernels/pers_4et*.cu`), host side and portable: a [`Drafter`] proposes
//! up to [`MAX_DRAFT`] tokens as a tree, a [`Verifier`] runs one tree pass
//! over the root and the drafted nodes, and the loop accepts along the path
//! where the pass's argmax at a node is the drafted child, takes the pass's
//! own argmax at the first disagreement, and moves the accepted rows into the
//! cache in the next launch's first phase (`n_move`). The emitted tokens are
//! the greedy decode of the pass at R = 1, token for token, whatever the
//! drafter proposes (the pass's identity contract, plan section 6).
//!
//! The card runs it through `llvq_llm::persist4e_cuda` (its `Verifier`) and
//! `bin/specbench`; the Mac runs it through the host twin
//! (`tests/tree_verify_host.rs`).

use std::time::Instant;

/// The drafted nodes a round carries at most: the pass's 32 rows hold more,
/// the drafters this loop serves draft blocks of 16 (the root and 15).
pub const MAX_DRAFT: usize = 15;

/// What a drafter is given each round.
pub struct DraftInput<'a> {
    /// The positions whose K and V the target holds when the next verify pass
    /// runs, including the rows that pass moves in first: the root's position.
    pub ctx_len: usize,
    /// Every token so far, the prompt first; the last is the root of the next
    /// verify pass (accepted, its K and V not yet computed).
    pub tokens: &'a [u32],
    /// The target's hidden states at the positions committed since the
    /// previous call, oldest first, `[n][TAPS][hidden]` binary16 (the
    /// prompt's on the first call when the caller has them; empty when the
    /// verifier writes no taps). Position p's states are those the target
    /// computed for token `tokens[p]`.
    pub new_taps: &'a [u16],
    /// The width of one tap.
    pub hidden: usize,
}

/// The tap points a verify pass writes a row: the inputs of five layers
/// (DFlash's target layers 1, 9, 17, 25, 33 read HF `hidden_states[i + 1]`:
/// the inputs of layers 2, 10, 18, 26, 34).
pub const TAPS: usize = 5;

/// A drafter. Returns the drafted nodes as `(token, parent)` pairs, at most
/// [`MAX_DRAFT`]: parent 0 is the root (the last of `tokens`), parent `j` the
/// j-th returned node (1-based), every parent before its children. A chain
/// of n tokens is parents 0, 1, ..., n - 1; nothing drafted is R = 1 decoding.
pub trait Drafter {
    fn draft(&mut self, input: &DraftInput) -> Vec<(u32, u8)>;
}

/// One verify launch: row r is `toks[r]` at position `pos0 + depth[r]`,
/// `path[r][..=depth[r]]` its ancestors' rows, root first, ending at r; the
/// launch first moves the previous launch's rows `mv` into the cache at
/// positions `move_pos0`, `move_pos0 + 1`, ... (`move_pos0 + mv.len() ==
/// pos0` when `mv` is not empty).
pub struct VerifyLaunch<'a> {
    pub pos0: usize,
    pub toks: &'a [u32],
    pub depth: &'a [u32],
    pub path: &'a [[u8; 32]],
    pub mv: &'a [u32],
    pub move_pos0: usize,
}

/// What a verify launch returns: each row's argmax, and its taps
/// (`[R][TAPS][hidden]`, empty when the verifier writes none).
pub struct VerifyOut {
    pub ring: Vec<u32>,
    pub taps: Vec<u16>,
}

/// One tree verify pass, on the card or on the host twin.
pub trait Verifier {
    fn verify(&mut self, launch: &VerifyLaunch) -> Result<VerifyOut, String>;
}

/// The drafter that drafts nothing: the loop is then R = 1 decoding.
pub struct NullDrafter;

impl Drafter for NullDrafter {
    fn draft(&mut self, _: &DraftInput) -> Vec<(u32, u8)> {
        Vec::new()
    }
}

/// A stand-in that knows the greedy continuation: each round it drafts a
/// chain of `correct` true tokens then one wrong one (the true token + 1
/// modulo `vocab`), so every pass accepts `correct` drafted tokens and adds
/// the pass's own, until the continuation runs out.
pub struct OracleDrafter {
    /// The tokens before the first generated one (the prompt).
    pub prompt_len: usize,
    /// The greedy tokens from the first generated one on.
    pub truth: Vec<u32>,
    pub correct: usize,
    pub vocab: u32,
}

impl Drafter for OracleDrafter {
    fn draft(&mut self, input: &DraftInput) -> Vec<(u32, u8)> {
        // tokens = prompt, then the generated ones; the root is truth[g - 1]
        let g = input.tokens.len().saturating_sub(self.prompt_len);
        let mut out = Vec::new();
        for i in 0..self.correct.min(MAX_DRAFT - 1) {
            match self.truth.get(g + i) {
                Some(&t) => out.push((t, out.len() as u8)),
                None => return out,
            }
        }
        if let Some(&t) = self.truth.get(g + out.len()) {
            out.push(((t + 1) % self.vocab, out.len() as u8));
        }
        out
    }
}

/// A fault the loop's host gates inject (mutants of the acceptance and of
/// the move); `None` in every real run.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Accept the drafted child at the first disagreement too.
    AcceptOneMore,
    /// Move a rejected sibling's row in place of the last accepted one.
    MoveRejected,
}

/// What the loop leaves: the generated tokens after `first` (at most the
/// budget), and per round the accepted drafted tokens and the walls.
#[derive(Default, Debug)]
pub struct LoopOut {
    pub tokens: Vec<u32>,
    /// Drafted tokens accepted in each pass (the pass's own token not counted).
    pub accepted: Vec<usize>,
    /// Seconds in the verifier, the drafter, and the whole loop.
    pub t_verify: Vec<f64>,
    pub t_draft: Vec<f64>,
    pub t_total: f64,
    /// Whether the loop stopped on `eos`.
    pub eos: bool,
}

/// Run the loop: the prompt of `prompt_len` tokens is in the cache, `first`
/// (the prefill's next token) sits at position `prompt_len`, its K and V not
/// yet computed. Generates until `budget` tokens after `first` or `eos`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    ver: &mut dyn Verifier,
    drafter: &mut dyn Drafter,
    prompt: &[u32],
    first: u32,
    prompt_taps: &[u16],
    hidden: usize,
    budget: usize,
    eos: Option<u32>,
) -> Result<LoopOut, String> {
    run_with_fault(ver, drafter, prompt, first, prompt_taps, hidden, budget, eos, Fault::None)
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn run_with_fault(
    ver: &mut dyn Verifier,
    drafter: &mut dyn Drafter,
    prompt: &[u32],
    first: u32,
    prompt_taps: &[u16],
    hidden: usize,
    budget: usize,
    eos: Option<u32>,
    fault: Fault,
) -> Result<LoopOut, String> {
    let t0 = Instant::now();
    let mut out = LoopOut::default();
    let mut tokens: Vec<u32> = prompt.to_vec();
    tokens.push(first);
    let mut pos0 = prompt.len();
    let (mut mv, mut move_pos0): (Vec<u32>, usize) = (Vec::new(), 0);
    let mut new_taps: Vec<u16> = prompt_taps.to_vec();
    if eos == Some(first) {
        out.eos = true;
    }
    while out.tokens.len() < budget && !out.eos {
        // 1. draft
        let td = Instant::now();
        let nodes = drafter.draft(&DraftInput { ctx_len: pos0, tokens: &tokens, new_taps: &new_taps, hidden });
        out.t_draft.push(td.elapsed().as_secs_f64());
        if nodes.len() > MAX_DRAFT {
            return Err(format!("a drafter returned {} nodes, at most {MAX_DRAFT}", nodes.len()));
        }
        // 2. the launch: slot 0 the root, slot j the j-th node
        let n = nodes.len() + 1;
        let mut toks = vec![*tokens.last().expect("not empty")];
        let mut depth = vec![0u32];
        let mut path = vec![[0u8; 32]];
        for (j, &(tok, parent)) in nodes.iter().enumerate() {
            let slot = j + 1;
            let p = parent as usize;
            if p >= slot {
                return Err(format!("node {slot}'s parent {p} does not precede it"));
            }
            let d = depth[p] as usize + 1;
            let mut row = path[p];
            row[d] = slot as u8;
            toks.push(tok);
            depth.push(d as u32);
            path.push(row);
        }
        let tv = Instant::now();
        let res = ver.verify(&VerifyLaunch { pos0, toks: &toks, depth: &depth, path: &path, mv: &mv, move_pos0 })?;
        out.t_verify.push(tv.elapsed().as_secs_f64());
        if res.ring.len() != n {
            return Err(format!("the verifier returned {} argmaxes for {n} rows", res.ring.len()));
        }
        // 3. accept along the path while the pass's argmax is a drafted child
        let mut acc = vec![0usize];
        let mut cur = 0usize;
        loop {
            let want = res.ring[cur];
            let child = (1..n).find(|&c| path[c][..depth[c] as usize].last() == Some(&(cur as u8)) && depth[c] == depth[cur] + 1 && toks[c] == want);
            match child {
                Some(c) => {
                    acc.push(c);
                    cur = c;
                }
                None => {
                    if fault == Fault::AcceptOneMore {
                        if let Some(c) = (1..n).find(|&c| depth[c] == depth[cur] + 1 && path[c][depth[cur] as usize] == cur as u8) {
                            acc.push(c);
                            cur = c;
                        }
                    }
                    break;
                }
            }
        }
        let bonus = res.ring[cur];
        out.accepted.push(acc.len() - 1);
        // 4. emit the accepted nodes' tokens and the pass's own
        let mut emitted: Vec<u32> = acc[1..].iter().map(|&c| toks[c]).collect();
        emitted.push(bonus);
        for &t in &emitted {
            if out.tokens.len() >= budget {
                break;
            }
            out.tokens.push(t);
            tokens.push(t);
            if eos == Some(t) {
                out.eos = true;
                break;
            }
        }
        // 5. the next launch moves the accepted rows (root first) into the cache
        let mut moved: Vec<u32> = acc.iter().map(|&c| c as u32).collect();
        if fault == Fault::MoveRejected {
            if let Some(r) = (1..n).find(|c| !acc.contains(c)) {
                *moved.last_mut().expect("the root") = r as u32;
            }
        }
        new_taps = Vec::new();
        if !res.taps.is_empty() {
            let w = TAPS * hidden;
            for &c in &acc {
                new_taps.extend_from_slice(&res.taps[c * w..(c + 1) * w]);
            }
        }
        move_pos0 = pos0;
        pos0 += moved.len();
        mv = moved;
    }
    out.t_total = t0.elapsed().as_secs_f64();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verifier whose argmax at a row is a function of the row's token
    /// alone (the "greedy model" tok -> tok * 7 + 3 mod 1000): the loop must
    /// emit that chain whatever the drafter does, and move the accepted rows.
    struct Toy {
        moves: Vec<(usize, Vec<u32>, usize)>,
    }
    impl Verifier for Toy {
        fn verify(&mut self, l: &VerifyLaunch) -> Result<VerifyOut, String> {
            self.moves.push((l.move_pos0, l.mv.to_vec(), l.pos0));
            Ok(VerifyOut { ring: l.toks.iter().map(|&t| (t * 7 + 3) % 1000).collect(), taps: Vec::new() })
        }
    }

    fn greedy(first: u32, n: usize) -> Vec<u32> {
        let mut v = Vec::new();
        let mut t = first;
        for _ in 0..n {
            t = (t * 7 + 3) % 1000;
            v.push(t);
        }
        v
    }

    #[test]
    fn the_loop_emits_the_greedy_chain_with_any_drafter() {
        let prompt = [1u32, 2, 3];
        let truth = greedy(5, 40);
        for correct in [0usize, 1, 3, 14] {
            let mut ver = Toy { moves: Vec::new() };
            let mut d = OracleDrafter { prompt_len: prompt.len(), truth: [vec![5], truth.clone()].concat(), correct, vocab: 1000 };
            let o = run(&mut ver, &mut d, &prompt, 5, &[], 8, 30, None).unwrap();
            assert_eq!(o.tokens, truth[..30], "oracle at {correct}");
            // each full pass accepts `correct` and adds one: positions advance by correct + 1
            for w in ver.moves.windows(2) {
                let (_, mv, pos) = &w[1];
                assert_eq!(w[1].0, w[0].2, "the move starts at the previous root");
                assert_eq!(*pos, w[0].2 + mv.len());
            }
            assert!(o.accepted.iter().rev().skip(1).all(|&a| a == correct.min(MAX_DRAFT - 1)));
        }
        let mut ver = Toy { moves: Vec::new() };
        let o = run(&mut ver, &mut NullDrafter, &prompt, 5, &[], 8, 10, None).unwrap();
        assert_eq!(o.tokens, truth[..10]);
        assert_eq!(ver.moves.len(), 10, "one pass a token");
    }

    #[test]
    fn the_faults_break_the_chain() {
        let prompt = [1u32, 2, 3];
        let truth = greedy(5, 40);
        let mut ver = Toy { moves: Vec::new() };
        let mut d = OracleDrafter { prompt_len: 3, truth: [vec![5], truth.clone()].concat(), correct: 2, vocab: 1000 };
        let o = run_with_fault(&mut ver, &mut d, &prompt, 5, &[], 8, 12, None, Fault::AcceptOneMore).unwrap();
        assert_ne!(o.tokens, truth[..12]);
    }

    #[test]
    fn eos_stops_the_loop() {
        let truth = greedy(5, 40);
        let mut ver = Toy { moves: Vec::new() };
        let o = run(&mut ver, &mut NullDrafter, &[1], 5, &[], 8, 30, Some(truth[4])).unwrap();
        assert_eq!(o.tokens, truth[..5]);
        assert!(o.eos);
    }
}
