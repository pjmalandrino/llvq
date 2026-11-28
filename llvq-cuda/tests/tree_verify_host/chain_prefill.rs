//! The chain prefill: the prompt through the tree verify pass, against the
//! one-row path, and its mutants. One file for the chain's passes and gates.

use super::*;

// ---- the chain prefill (docs/mesures/large-prefill-contexte-2026-11-11.txt, first experiment) ----

/// The twin's variant bit: every pass but the command's last launched with
/// `split_head` 0 (no final norm, no head), as the card's chain prefill runs
/// the passes before the prompt's last.
const HEAD_LAST: u32 = 1073741824;

pub(super) fn pass_of(l: &llvq_cuda::spec_loop::ChainLaunch, seed: u64) -> Pass {
    Pass {
        pos0: l.pos0 as u32,
        seed,
        toks: l.toks.clone(),
        depth: l.depth.clone(),
        path: l.path.iter().zip(&l.depth).map(|(p, &d)| p[..=d as usize].to_vec()).collect(),
        move_pos0: l.move_pos0 as u32,
        mv: l.mv.clone(),
    }
}

/// The chain prefill of `toks` from position 0 at `r` rows a pass, on
/// scheduler seeds of its own, and with `first`, the hand-over pass (one row
/// at the prompt's end, moving the last chunk in first).
fn chain_passes(toks: &[u32], r: usize, first: Option<u32>, fault: llvq_cuda::spec_loop::ChainFault) -> Vec<Pass> {
    let (ls, mv, mp) = llvq_cuda::spec_loop::chain_launches(toks, 0, r, fault).unwrap();
    let mut v: Vec<Pass> = ls.iter().enumerate().map(|(i, l)| pass_of(l, 0x8a00 + 64 * r as u64 + i as u64)).collect();
    if let Some(t) = first {
        v.push(pass_of(&llvq_cuda::spec_loop::chain_handover(t, &mv, mp), 0x8bff));
    }
    v
}

/// Every row of every pass in position order (a chain: pass by pass, row by row).
fn rows_in_order(o: &Out) -> Vec<&RowOut> {
    o.passes.iter().flat_map(|p| p.iter()).collect()
}

/// Two chain prefills of the same prompt (with their hand-over rows) row
/// for row: argmax, logits, residual, K and V bit for bit.
pub(super) fn gate_chain_rows(a: &Out, b: &Out, what: &str) -> Result<usize, String> {
    let (ra, rb) = (rows_in_order(a), rows_in_order(b));
    if ra.len() != rb.len() {
        return Err(format!("{what}: {} rows against {}", ra.len(), rb.len()));
    }
    for (p, (x, y)) in ra.iter().zip(&rb).enumerate() {
        if let Some(d) = diff(x, y) {
            return Err(format!("{what}: position {p}: {d} differs"));
        }
    }
    Ok(ra.len())
}

/// The first experiment's host gate, on the 4B's shapes (two layers at its
/// widths, a Tetra then an int4 down, the tied head), arm a12ip, from an
/// empty cache: the chain prefill of a 37-token prompt at R = 16 and at
/// R = 5 equals the pass's one-row path (R = 1, each pass moving the
/// previous row) bit for bit on every row's argmax, logits, residual, K and
/// V, and on the hand-over row that reads the whole prompt from the cache;
/// at R = 16 with the head on the last pass only, every residual, K and V
/// is the same, the last row's logits too, and the passes before it write
/// no logit. The R = 1 pass at position 0 is within 1e-2 of 4E's logits.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn chain_prefill_equals_the_one_row_path() {
    use llvq_cuda::spec_loop::ChainFault;
    let mut g = Lcg(0x8a1);
    let toks: Vec<u32> = (0..37).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let first = 611u32;
    let f = Fix { split: SPLIT_N2B, variant: A12IP, ..FIX };
    let one = exec(host(), &f, None, &chain_passes(&toks, 1, Some(first), ChainFault::None)).unwrap();
    for r in [16usize, 5] {
        let o = exec(host(), &f, None, &chain_passes(&toks, r, Some(first), ChainFault::None)).unwrap();
        let n = gate_chain_rows(&o, &one, &format!("R = {r} against R = 1")).unwrap();
        eprintln!("chain R = {r}: {n} rows (37 prompt positions and the hand-over) equal to R = 1 bit for bit");
    }
    // the head on the last pass only, as the card runs it
    let full = exec(host(), &f, None, &chain_passes(&toks, 16, None, ChainFault::None)).unwrap();
    let last = exec(host(), &Fix { variant: A12IP | HEAD_LAST, ..f }, None, &chain_passes(&toks, 16, None, ChainFault::None)).unwrap();
    gate_head_last(&full, &last).unwrap();
    eprintln!("chain R = 16, head on the last pass only: residuals, K and V of the 37 rows and the last row's logits equal; no logit before it");
    let (e, t4e, tt) = gate_close(host(), &Fix { cap: 288, ..f }, toks[0], 0).unwrap();
    eprintln!("R = 1 at position 0: relative RMS {e:.3e} against 4E, argmax 4E {t4e}, pass {tt}");
}

/// [`chain_prefill_equals_the_one_row_path`]'s head check: `last` (the head
/// on the last pass only) against `full` (the head on every pass).
fn gate_head_last(full: &Out, last: &Out) -> Result<(), String> {
    let (rf, rl) = (rows_in_order(full), rows_in_order(last));
    if rf.len() != rl.len() {
        return Err(format!("{} rows against {}", rl.len(), rf.len()));
    }
    let n_last = last.passes.last().map_or(0, |p| p.len());
    for (p, (a, b)) in rf.iter().zip(&rl).enumerate() {
        if a.x != b.x || a.kv != b.kv {
            return Err(format!("position {p}: the residual or K/V moved with the head off"));
        }
        if p + n_last >= rl.len() {
            if a.logits != b.logits || a.tok != b.tok {
                return Err(format!("position {p}: the last pass's logits or argmax differ"));
            }
        } else if b.logits.iter().any(|&v| v != 0xFFFF) {
            return Err(format!("position {p}: a pass without the head wrote a logit"));
        }
    }
    Ok(())
}

/// The path off by one (each row's ancestors shifted to the slot before) is
/// killed by the R = 1 gate; a head guard that never skips is killed by the
/// head-last gate.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn chain_prefill_mutants_are_killed() {
    use llvq_cuda::spec_loop::ChainFault;
    let mut g = Lcg(0x8a2);
    let toks: Vec<u32> = (0..21).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let f = Fix { split: SPLIT_N2B, variant: A12IP, ..FIX };
    let one = exec(host(), &f, None, &chain_passes(&toks, 1, Some(5), ChainFault::None)).unwrap();
    let good = exec(host(), &f, None, &chain_passes(&toks, 16, Some(5), ChainFault::None)).unwrap();
    gate_chain_rows(&good, &one, "unmutated").expect("the unmutated chain passes its gate");
    match exec(host(), &f, None, &chain_passes(&toks, 16, Some(5), ChainFault::PathOffByOne)).and_then(|o| gate_chain_rows(&o, &one, "path off by one")) {
        Err(e) => eprintln!("path_off_by_one: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("chain mutant path_off_by_one survived its gate"),
    }
    let full = exec(host(), &f, None, &chain_passes(&toks, 16, None, ChainFault::None)).unwrap();
    let k = "llvq-cuda/kernels/pers_4et_a12.cu";
    let b = mutant("head_never_skipped", &[(k, "    const bool head_on = L.split_head != 0u;", "    const bool head_on = true;")]);
    let r = exec(&b, &Fix { variant: A12IP | HEAD_LAST, ..f }, None, &chain_passes(&toks, 16, None, ChainFault::None)).and_then(|o| gate_head_last(&full, &o));
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("head_never_skipped: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("chain mutant head_never_skipped survived its gate"),
    }
}
