//! The full-tile arms: the attention tile with no guard between positions, held
//! to their guarded parents. One file for the ft build and its gates.

use super::*;

// ---- the full-tile arms: the attention tile with no guard between positions ----
// (docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt)

/// The twin built from the sources with the ft edit (`pers::pers_ft_edit`,
/// the edit `pers::p4t_ft_unit_text` and the 4E arm v9ft apply to the card's
/// modules); `twist` rewrites one site of the inserted text (a mutant of the
/// arm), which must hold it exactly once.
fn ft_build(tag: &str, twist: Option<(&str, &str)>) -> PathBuf {
    let (part, from, mut to) = pers::pers_ft_edit();
    if let Some((a, b)) = twist {
        let n = to.matches(a).count();
        eprintln!("{tag}: {n} site(s) in the ft text");
        assert_eq!(n, 1, "ft mutant {tag}: {n} sites for {a:?} in the ft text, expected exactly one");
        to = to.replacen(a, b, 1);
    }
    let file = format!("llvq-cuda/kernels/{part}");
    mutant(tag, &[(file.as_str(), from.as_str(), to.as_str())])
}

fn ft_host() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| ft_build("ft", None))
}

/// The ft passes: chains of each size in `sizes` at pos0 9, 250, 500 (the
/// rows' lengths crossing 512) and 1000 (three whole splits below pos0), the
/// a12ipg gates' long caches. A chain's rows end at every length from pos0 + 1
/// to pos0 + R, so its last tiles hold 1 to 8 positions. The twin's caches
/// hold random values at every position. (pos0 519 at 16 rows trips the twin's
/// lowered spin watchdog, PERS_SPIN_LIMIT 1 << 14, on a12ip itself.)
fn ft_passes(sizes: &[usize]) -> Vec<Pass> {
    let mut g = Lcg(0x8f7);
    let toks: Vec<u32> = (0..32).map(|_| (g.next() % VOCAB as u64) as u32).collect();
    let mut v = Vec::new();
    for (i, &p) in [9u32, 250, 500, 1000].iter().enumerate() {
        for (j, &r) in sizes.iter().enumerate() {
            v.push(chain_at(&toks[..r], p, 0x8f00 + 16 * i as u64 + j as u64));
        }
    }
    v
}

/// The arm on `bin` (the ft twin or a mutant of it) against its guarded
/// parent on the unmutated twin, the same variant and passes: every row's
/// argmax, logits, residual, K, V and taps bit for bit.
fn gate_ft(bin: &Path, f: &Fix, passes: &[Pass]) -> Result<usize, String> {
    let u = exec(host(), f, None, passes)?;
    let w = exec(bin, f, None, passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&w.passes).enumerate() {
        if a.len() != b.len() {
            return Err(format!("ft, pass {pi}: {} rows against {}", b.len(), a.len()));
        }
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("ft against its parent, variant {}, pass {pi} (pos0 {}, {} rows), slot {s}: {what}", f.variant, passes[pi].pos0, a.len()));
            }
            if rb.taps != ra.taps {
                return Err(format!("ft against its parent, pass {pi}, slot {s}: taps"));
            }
        }
    }
    Ok(rows)
}

/// a12ipft, a12ipgft and a12ipg32ft each equal their guarded parent bit for
/// bit at R = 1, 5, 16 (and 32 for a12ipg32ft) at four cache lengths
/// (`ft_passes`), with the 4B's shapes; and
/// on the chain prefill of 45 tokens from 500 (crossing the split at 512) at
/// R = 16 (a12ipgft) and R = 32 (a12ipg32ft), the hand-over row included.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ft_rows_equal_their_guarded_parents() {
    let f = FIX_LONG;
    for (name, variant, sizes) in [("a12ipft", A12IP, &[1usize, 5, 16][..]), ("a12ipgft", A12IPG, &[1, 5, 16]), ("a12ipg32ft", A12IPG32, &[1, 5, 16, 32])] {
        let passes = ft_passes(sizes);
        let n = gate_ft(ft_host(), &Fix { variant, ..f }, &passes).unwrap();
        eprintln!("{name}: {n} rows over {} passes (pos0 9, 250, 500, 1000; R {sizes:?}) equal to its parent's bit for bit, taps included", passes.len());
    }
    let (short, _) = g32_prompts();
    for (name, variant, r) in [("a12ipgft", A12IPG, 16usize), ("a12ipg32ft", A12IPG32, 32)] {
        let passes = chain_passes_at(&short, 500, r, 611);
        let n = gate_ft(ft_host(), &Fix { variant, ..f }, &passes).unwrap();
        eprintln!("{name}: chain of 45 from 500 at R = {r}: {n} rows equal to its parent's bit for bit");
    }
}

/// Two mutants of the ft text, one site each (the builder counts it), each
/// killed by a12ipg32ft's gate against a12ipg32 at R = 1, 5, 32: the
/// full-tile body taken on a tile of 7 (its eighth position read from the
/// stage's zero fill), and position 3's chain reading position 4's K.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ft_mutants_are_killed() {
    let f = Fix { variant: A12IPG32, ..FIX_LONG };
    let passes = ft_passes(&[1, 5, 32]);
    gate_ft(ft_host(), &f, &passes).expect("the unmutated a12ipg32ft passes its gate");
    let muts: [(&str, &str, &str); 2] = [
        ("ft_taken_at_7", "    if (nv == PERS_TT) {", "    if (nv + 1u >= PERS_TT) {"),
        ("ft_neighbour_k", "for (u32 j = 0; j < PERS_TT; ++j) kw[j] = ks[j * 16u + l16];", "for (u32 j = 0; j < PERS_TT; ++j) kw[j] = ks[(j == 3u ? 4u : j) * 16u + l16];"),
    ];
    for (tag, from, to) in muts {
        let b = ft_build(tag, Some((from, to)));
        let r = gate_ft(&b, &f, &passes);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or("")),
            Ok(_) => panic!("ft mutant {tag} survived its gate"),
        }
    }
}
