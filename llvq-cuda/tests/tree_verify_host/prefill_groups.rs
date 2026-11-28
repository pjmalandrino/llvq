//! Arms a12ipg and a12ipg32: the cache read once for a group of rows, at 16 and
//! 32 rows a launch, for the large prefill. One file for the grouped read.

use super::*;

// ---- arm a12ipg: the cache read once for a group of rows (large prefill, D1) ----

/// A chain of `toks` at `pos0` (row r at depth r, its path 0..=r), nothing moved.
pub(super) fn chain_at(toks: &[u32], pos0: u32, seed: u64) -> Pass {
    let n = toks.len();
    Pass { pos0, seed, toks: toks.to_vec(), depth: (0..n as u32).collect(), path: (0..n).map(|r| (0..=r as u8).collect()).collect(), move_pos0: 0, mv: vec![] }
}

/// a12ipg (on `bin`) against a12ip (on the unmutated twin) on the same
/// passes: every row's argmax, logits, residual, K, V and taps bit for bit.
fn gate_ipg(bin: &Path, f: &Fix, passes: &[Pass]) -> Result<usize, String> {
    let u = exec(host(), &Fix { variant: f.variant | A12IP, ..*f }, None, passes)?;
    let w = exec(bin, &Fix { variant: f.variant | A12IPG, ..*f }, None, passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&w.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("a12ipg against a12ip, pass {pi} (pos0 {}, {} rows), slot {s}: {what}", passes[pi].pos0, a.len()));
            }
            if rb.taps != ra.taps {
                return Err(format!("a12ipg against a12ip, pass {pi}, slot {s}: taps"));
            }
        }
    }
    Ok(rows)
}

/// The D1 passes: chains of 1, 5 and 16 rows at a short cache (pos0 9, no
/// split below pos0, every job a12ip's) and at two long ones (pos0 500: one
/// shared split, the rows' lengths crossing 512 inside a group; pos0 1000:
/// three shared splits), and a branching tree of 16 at pos0 1000. The twin's
/// caches hold random values at every position.
fn ipg_passes(sizes: &[usize], pos0s: &[u32]) -> Vec<Pass> {
    let mut g = Lcg(0x8d1);
    let toks: Vec<u32> = (0..16).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let mut v = Vec::new();
    for (i, &p) in pos0s.iter().enumerate() {
        for (j, &r) in sizes.iter().enumerate() {
            v.push(chain_at(&toks[..r], p, 0x8d00 + 16 * i as u64 + j as u64));
        }
    }
    v
}

pub(super) const FIX_LONG: Fix = Fix { cap: 1040, split: SPLIT_N2B, ..FIX };

/// a12ipg equals a12ip bit for bit at R = 1, 5, 16, at a short and two long
/// caches, on chains and on a branching tree, with the 4B's shapes and with
/// the 8B's parts.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ipg_rows_equal_a12ip_rows_at_short_and_long_caches() {
    let t = tree(16, 5, 0x8d2);
    for ext in [0u32, 16] {
        let f = Fix { variant: ext, ..FIX_LONG };
        let mut passes = ipg_passes(&[1, 5, 16], &[9, 500, 1000]);
        passes.push(Pass::of(&t, &t.subset(16, 0x8d3), 1000, 0x8d4));
        let n = gate_ipg(host(), &f, &passes).unwrap();
        eprintln!("variant {ext}: {n} rows over {} passes (pos0 9, 500, 1000; R 1, 5, 16; a tree of 16), a12ipg equals a12ip bit for bit, taps included", passes.len());
    }
}

/// The chain prefill of `toks` from `pos0` at `r` rows a pass, then the
/// hand-over row.
pub(super) fn chain_passes_at(toks: &[u32], pos0: usize, r: usize, first: u32) -> Vec<Pass> {
    use llvq_cuda::spec_loop::{chain_handover, chain_launches, ChainFault};
    let (ls, mv, mp) = chain_launches(toks, pos0, r, ChainFault::None).unwrap();
    let mut v: Vec<Pass> = ls.iter().enumerate().map(|(i, l)| pass_of(l, 0x8e00 + 64 * r as u64 + i as u64)).collect();
    v.push(pass_of(&chain_handover(first, &mv, mp), 0x8eff));
    v
}

/// The chain prefill of 37 tokens after 600 cached positions: a12ipg at
/// R = 16 and R = 5 equals a12ip's one-row path (R = 1, each pass moving the
/// previous row) on every row and on the hand-over row.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ipg_chain_prefill_equals_the_one_row_path() {
    let mut g = Lcg(0x8e1);
    let toks: Vec<u32> = (0..37).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let f = FIX_LONG;
    let one = exec(host(), &Fix { variant: A12IP, ..f }, None, &chain_passes_at(&toks, 600, 1, 611)).unwrap();
    for r in [16usize, 5] {
        let o = exec(host(), &Fix { variant: A12IPG, ..f }, None, &chain_passes_at(&toks, 600, r, 611)).unwrap();
        let n = gate_chain_rows(&o, &one, &format!("a12ipg R = {r} against a12ip R = 1")).unwrap();
        eprintln!("a12ipg chain from 600, R = {r}: {n} rows equal to a12ip's R = 1 bit for bit");
    }
}

/// Two mutants of the grouped attention, each applied at exactly one site
/// (the builder refuses any other count): a row of a group scoring with its
/// neighbour's staged q, and one split too many read as shared (the split
/// that holds pos0, whose tree positions then come from the cache). Each is
/// killed by the a12ip gate.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ipg_mutants_are_killed() {
    let f = FIX_LONG;
    let passes = ipg_passes(&[5, 16], &[500, 1000]);
    gate_ipg(host(), &f, &passes).expect("the unmutated a12ipg passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a12g.cu";
    let muts: [(&str, &str, &str); 2] = [
        (
            "ipg_neighbour_q",
            "                const float* qrow = qs0 + ri * qs_row + ge * PERS_HD;",
            "                const float* qrow = qs0 + ((ri + 1u) % nr) * qs_row + ge * PERS_HD;",
        ),
        ("ipg_split_past_pos0", "    return pos0 / PERS_SPLIT;", "    return pos0 / PERS_SPLIT + (pos0 >= PERS_SPLIT ? 1u : 0u);"),
    ];
    for (tag, from, to) in muts {
        let text = std::fs::read_to_string(repo().join(k)).unwrap();
        eprintln!("{tag}: {} site(s) in {k}", text.matches(from).count());
        let b = mutant(tag, &[(k, from, to)]);
        let r = gate_ipg(&b, &f, &passes);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or("")),
            Ok(_) => panic!("a12ipg mutant {tag} survived its gate"),
        }
    }
}

// ---- arm a12ipg32: a12ipg at 32 rows a launch (large prefill, design A) ----

/// The prompts of the a12ipg32 gates. Short: 45 tokens from position 500,
/// across the split boundary at 512, so a pass of 32 starts at 500 (one
/// shared split) and the next at 532 (two), where the passes of 16 start at
/// 500, 516 and 532. Long: 300 tokens from 250, across 256 and 512, three
/// splits, the shared ones changing from pass to pass at 16 and at 32. Neither
/// length is a multiple of 32 or of 16.
pub(super) fn g32_prompts() -> (Vec<u32>, Vec<u32>) {
    let mut g = Lcg(0x8f1);
    let short: Vec<u32> = (0..45).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let long: Vec<u32> = (0..300).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    (short, long)
}

/// a12ipg32 at R = 32 equals a12ipg at R = 16 and a12ip's one-row path
/// (R = 1, each pass moving the previous row) on every row's argmax, logits,
/// residual, K and V and on the hand-over row: the short prompt against both,
/// with the 4B's shapes and with the 8B's parts against R = 16; the long one
/// against R = 16.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ipg32_chain_prefill_equals_r16_and_r1() {
    let (short, long) = g32_prompts();
    let f = FIX_LONG;
    let one = exec(host(), &Fix { variant: A12IP, ..f }, None, &chain_passes_at(&short, 500, 1, 611)).unwrap();
    for ext in [0u32, 16] {
        let r16 = exec(host(), &Fix { variant: ext | A12IPG, ..f }, None, &chain_passes_at(&short, 500, 16, 611)).unwrap();
        let r32 = exec(host(), &Fix { variant: ext | A12IPG32, ..f }, None, &chain_passes_at(&short, 500, 32, 611)).unwrap();
        let n = gate_chain_rows(&r32, &r16, &format!("variant {ext}: a12ipg32 R = 32 against a12ipg R = 16, from 500")).unwrap();
        eprintln!("variant {ext}: a12ipg32 chain of 45 from 500, R = 32: {n} rows equal to a12ipg's R = 16 bit for bit");
        if ext == 0 {
            let n = gate_chain_rows(&r32, &one, "a12ipg32 R = 32 against a12ip R = 1, from 500").unwrap();
            eprintln!("a12ipg32 chain of 45 from 500, R = 32: {n} rows equal to a12ip's R = 1 bit for bit");
        }
    }
    let r16 = exec(host(), &Fix { variant: A12IPG, ..f }, None, &chain_passes_at(&long, 250, 16, 611)).unwrap();
    let r32 = exec(host(), &Fix { variant: A12IPG32, ..f }, None, &chain_passes_at(&long, 250, 32, 611)).unwrap();
    let n = gate_chain_rows(&r32, &r16, "a12ipg32 R = 32 against a12ipg R = 16, from 250").unwrap();
    eprintln!("a12ipg32 chain of 300 from 250, R = 32: {n} rows equal to a12ipg's R = 16 bit for bit");
}

/// Three mutants of the 32-row pass, each at exactly one site (the builder
/// refuses any other count), each killed by the R = 32 against R = 16 gate on
/// the short prompt: n8 tiles 2 and 3 stored in tiles 0 and 1's columns;
/// their accumulators never zeroed; the arm compiled at RC = 2 (the row cap
/// left at 16) while the host sends 32 rows, which must abort the launch.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ipg32_mutants_are_killed() {
    let (short, _) = g32_prompts();
    let f = FIX_LONG;
    let r16 = exec(host(), &Fix { variant: A12IPG, ..f }, None, &chain_passes_at(&short, 500, 16, 611)).unwrap();
    let gate = |bin: &Path| -> Result<usize, String> {
        let o = exec(bin, &Fix { variant: A12IPG32, ..f }, None, &chain_passes_at(&short, 500, 32, 611))?;
        gate_chain_rows(&o, &r16, "a12ipg32 R = 32 against a12ipg R = 16")
    };
    gate(host()).expect("the unmutated a12ipg32 passes its gate");
    let rc = "llvq-cuda/kernels/pers_4et_a10rc.cu";
    let g32 = "llvq-cuda/kernels/pers_4et_a12g32.cu";
    let muts: [(&str, &str, &str, &str); 3] = [
        (
            "g32_tiles_2_3_in_0_1",
            rc,
            "        for (u32 e = 0; e < 4u; ++e) {\n            const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);",
            "        for (u32 e = 0; e < 4u; ++e) {\n            const u32 row = wrow + gid + (e >> 1) * 8u, n = (nt & 1u) * 8u + 2u * tig + (e & 1u);",
        ),
        ("g32_acc_not_zeroed", rc, "        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;", "        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = nt < 2u ? 0.0f : acc[nt][e];"),
        ("g32_row_cap_16", g32, "        pers_4et_a12g_body<FORM_, 4u, true>(A, L);", "        pers_4et_a12g_body<FORM_, 2u, true>(A, L);"),
    ];
    for (tag, k, from, to) in muts {
        let text = std::fs::read_to_string(repo().join(k)).unwrap();
        eprintln!("{tag}: {} site(s) in {k}", text.matches(from).count());
        let b = mutant(tag, &[(k, from, to)]);
        let r = gate(&b);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or("")),
            Ok(_) => panic!("a12ipg32 mutant {tag} survived its gate"),
        }
    }
}
