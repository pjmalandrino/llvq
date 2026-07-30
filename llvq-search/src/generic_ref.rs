//! Deliberately naive reference for the generic ball-13 engine.
//!
//! This exists to make the Phase 2c optimizations of [`crate::generic`]
//! *falsifiable*: it recomputes the same per-shell maxima with no bounds, no
//! pruning, no prefix sums and no incremental structure — full sorts and
//! plain dot products, one class at a time. It shares only the closed forms
//! themselves (validated independently by the G2/G2b suites: brute force on
//! shells 2–3, exhaustive DP on the even parity repair).
//!
//! One extra degree of freedom is left open on purpose: the even parity
//! repair here tries the sacrificed value at **every** support slot, not
//! just the minimal-|x| one. So this reference does not assume the
//! conclusion the fast path relies on — if that conclusion were wrong, the
//! reference would score strictly higher and the comparison test would fail.
//!
//! Slow by construction (~0.5 s per query). Tests only.

use crate::classes::{enumerate_classes, ClassSet, MAX_SHELL};
use crate::generic::NSHELLS;
use crate::Searcher;
use llvq_core::DIM;

pub struct RefBallSearcher {
    classes: ClassSet,
}

impl RefBallSearcher {
    pub fn new() -> Self {
        Self {
            classes: enumerate_classes(MAX_SHELL),
        }
    }

    /// `max_{v ∈ Shell(m)} ⟨x, v⟩` for m = 2..=13, the slow way.
    pub fn shell_dots(&self, s: &Searcher, x: &[f64; DIM]) -> [f64; NSHELLS] {
        let g = s.leech().golay();
        let mut best = [f64::NEG_INFINITY; NSHELLS];

        let mut abs_desc: [f64; DIM] = core::array::from_fn(|i| x[i].abs());
        abs_desc.sort_unstable_by(|a, b| b.total_cmp(a));

        for c in &self.classes.even {
            let si = (c.shell - 2) as usize;
            let word: Vec<f64> = expand(&c.word_vals);
            let free: Vec<f64> = expand(&c.free_vals);
            let w = word.len();
            debug_assert_eq!(w, c.w as usize);

            if w == 0 {
                let score: f64 = free.iter().zip(abs_desc.iter()).map(|(v, a)| v * a).sum();
                best[si] = best[si].max(score);
                continue;
            }

            for &cw in g.of_weight(w) {
                let mut sup: Vec<f64> = (0..DIM)
                    .filter(|&i| cw >> i & 1 == 1)
                    .map(|i| x[i].abs())
                    .collect();
                sup.sort_unstable_by(|a, b| b.total_cmp(a));
                let mut off: Vec<f64> = (0..DIM)
                    .filter(|&i| cw >> i & 1 == 0)
                    .map(|i| x[i].abs())
                    .collect();
                off.sort_unstable_by(|a, b| b.total_cmp(a));

                let off_score: f64 = free.iter().zip(off.iter()).map(|(v, a)| v * a).sum();
                let neg = (0..DIM).filter(|&i| cw >> i & 1 == 1 && x[i] < 0.0).count();

                let on_score = if (neg % 2) as u8 == c.p_req {
                    word.iter().zip(sup.iter()).map(|(v, a)| v * a).sum()
                } else {
                    // One value takes a mismatched sign. Try every value kind
                    // at every slot; the rest pack greedily on the remaining
                    // slots (rearrangement inequality, all sign-matched).
                    let mut bestrep = f64::NEG_INFINITY;
                    for k in 0..word.len() {
                        if k > 0 && word[k] == word[k - 1] {
                            continue; // same kind, same result
                        }
                        let mut rest = word.clone();
                        let u = rest.remove(k);
                        for j in 0..w {
                            let mut sc = -u * sup[j];
                            let mut it = rest.iter();
                            for (slot, &a) in sup.iter().enumerate() {
                                if slot == j {
                                    continue;
                                }
                                sc += it.next().expect("w−1 values for w−1 slots") * a;
                            }
                            bestrep = bestrep.max(sc);
                        }
                    }
                    bestrep
                };
                best[si] = best[si].max(on_score + off_score);
            }
        }

        for c in &self.classes.odd {
            let si = (c.shell - 2) as usize;
            let mut a: Vec<f64> = Vec::with_capacity(DIM);
            for &(v, n) in &c.vals {
                let av = if v % 4 == 3 { v as f64 } else { -(v as f64) };
                for _ in 0..n {
                    a.push(av);
                }
            }
            a.sort_unstable_by(f64::total_cmp);
            for &cw in g.codewords() {
                let mut y: [f64; DIM] =
                    core::array::from_fn(|i| if cw >> i & 1 == 1 { x[i] } else { -x[i] });
                y.sort_unstable_by(f64::total_cmp);
                let score: f64 = a.iter().zip(y.iter()).map(|(p, q)| p * q).sum();
                best[si] = best[si].max(score);
            }
        }

        best
    }
}

fn expand(vals: &[(u8, u8)]) -> Vec<f64> {
    let mut out = Vec::new();
    for &(v, n) in vals {
        for _ in 0..n {
            out.push(v as f64);
        }
    }
    out
}

impl Default for RefBallSearcher {
    fn default() -> Self {
        Self::new()
    }
}
