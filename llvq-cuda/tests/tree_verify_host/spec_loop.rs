//! The speculative loop (`llvq_cuda::spec_loop`) driven on the host twin, against
//! the greedy decode. One file for the loop's only host gate.

use super::*;

// ---- the speculative loop (llvq_cuda::spec_loop) on the host twin ----

/// The twin as a verifier: the tree buffer persists only within one command,
/// so each call replays every launch so far and returns the last one's rows.
struct TwinVerifier {
    f: Fix,
    history: Vec<Pass>,
}

impl llvq_cuda::spec_loop::Verifier for TwinVerifier {
    fn verify(&mut self, l: &llvq_cuda::spec_loop::VerifyLaunch) -> Result<llvq_cuda::spec_loop::VerifyOut, String> {
        self.history.push(Pass {
            pos0: l.pos0 as u32,
            seed: 0x7600 + self.history.len() as u64,
            toks: l.toks.to_vec(),
            depth: l.depth.to_vec(),
            path: l.path.iter().zip(l.depth).map(|(p, &d)| p[..=d as usize].to_vec()).collect(),
            move_pos0: l.move_pos0 as u32,
            mv: l.mv.to_vec(),
        });
        let o = exec(host(), &self.f, None, &self.history)?;
        let last = o.passes.last().expect("a pass");
        Ok(llvq_cuda::spec_loop::VerifyOut {
            ring: last.iter().map(|r| r.tok).collect(),
            taps: last.iter().flat_map(|r| r.taps.iter().copied()).collect(),
        })
    }
}

/// The loop on the twin (a12 at 64 threads, its taps on): the null drafter
/// gives the twin's greedy decode at R = 1 (one pass a token, each moving the
/// previous root); the oracle at 1, 3 and 7 correct tokens a pass gives the
/// same tokens in fewer passes, its accepted rows moved; the two faults
/// (accept one node too many, move a rejected row) change the tokens.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_loop_on_the_twin_is_the_greedy_decode() {
    use llvq_cuda::spec_loop::{run, run_with_fault, Fault, NullDrafter, OracleDrafter};
    let f = Fix { split: SPLIT_N2B, variant: A12, ..FIX };
    let prompt: Vec<u32> = (0..9).collect();
    let first = 517u32;
    let n = 8usize;
    let mut v = TwinVerifier { f, history: Vec::new() };
    let greedy = run(&mut v, &mut NullDrafter, &prompt, first, &[], H, n, None).unwrap();
    assert_eq!(greedy.tokens.len(), n);
    assert_eq!(v.history.len(), n, "one pass a token");
    assert!(v.history.iter().skip(1).all(|p| p.mv.len() == 1), "each pass moves the previous root");
    eprintln!("twin greedy: {:?}", greedy.tokens);
    let truth: Vec<u32> = std::iter::once(first).chain(greedy.tokens.iter().copied()).collect();
    for correct in [1usize, 3, 7] {
        let mut v = TwinVerifier { f, history: Vec::new() };
        let mut d = OracleDrafter { prompt_len: prompt.len(), truth: truth.clone(), correct, vocab: VOCAB };
        let o = run(&mut v, &mut d, &prompt, first, &[], H, n, None).unwrap();
        assert_eq!(o.tokens, greedy.tokens, "oracle at {correct}");
        assert!(v.history.len() < n, "oracle at {correct}: {} passes", v.history.len());
        for w in v.history.windows(2) {
            assert_eq!(w[1].move_pos0, w[0].pos0);
            assert_eq!(w[1].pos0, w[0].pos0 + w[1].mv.len() as u32, "the accepted rows moved in");
        }
        assert!(o.accepted[..o.accepted.len() - 1].iter().all(|&a| a == correct), "oracle at {correct}: {:?}", o.accepted);
        eprintln!("oracle at {correct}: {} passes, accepted {:?}", v.history.len(), o.accepted);
    }
    for fault in [Fault::AcceptOneMore, Fault::MoveRejected] {
        let mut v = TwinVerifier { f, history: Vec::new() };
        let mut d = OracleDrafter { prompt_len: prompt.len(), truth: truth.clone(), correct: 3, vocab: VOCAB };
        let o = run_with_fault(&mut v, &mut d, &prompt, first, &[], H, n, None, fault).unwrap();
        assert_ne!(o.tokens, greedy.tokens, "fault {fault:?} survived");
        eprintln!("fault {fault:?}: killed, tokens {:?}", o.tokens);
    }
}
