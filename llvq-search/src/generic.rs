//! Generic exact per-class nearest-neighbour engine for shells m ≤ 13
//! (Phase 2b — the 2 bit/dim regime of the paper).
//!
//! ## Per-class maximizers (both exact)
//!
//! **Odd classes.** Signs are forced by codeword membership, and the
//! contribution of placing |value| v at position i is `a(v)·yᵢ` with
//! `a(v) = +v` if v ≡ 3 (mod 4), `−v` otherwise, and
//! `yᵢ = (i ∈ c ? xᵢ : −xᵢ)`. With no residual constraint (the sum
//! condition is class-level, see `classes`), the maximum over arrangements
//! is the sorted pairing of the `a`-multiset with `y` — exact by the
//! rearrangement inequality.
//!
//! **Even classes.** Off-support free values pair greedily with the largest
//! off-support |x| (signs free, rearrangement inequality). On-support word
//! values pair greedily with support |x| sorted descending, signs matched;
//! if the matched minus-parity (= #{i ∈ c : xᵢ < 0}, arrangement-
//! independent) differs from the class requirement, exactly one value must
//! take a mismatched sign (three or more mismatches are dominated). The
//! exact repair: the sacrificed value v contributes `−v·|x|`, so it goes to
//! the **smallest** |x| of the support while the remaining values repack
//! greedily on the rest — computed for each distinct value kind via one
//! suffix scan, NOT by flipping in place (flipping v at its greedy slot is
//! suboptimal: sacrificing value u at slot w and promoting the values after
//! it beats it whenever the promotion gain exceeds the slot difference).

use crate::classes::{enumerate_classes, ClassSet, MAX_SHELL};
use crate::{Found, Metric, Searcher, Workspace};
use llvq_core::{Point, DIM};

/// Number of shells covered (m = 2..=13).
pub const NSHELLS: usize = (MAX_SHELL - 1) as usize;

/// Runtime form of an even class.
struct EvenRt {
    shell_idx: usize,
    /// Word values expanded descending, e.g. [6, 2, 2, …] (len = w).
    word: Vec<f64>,
    /// Distinct word kinds with the index range they occupy in `word`.
    kinds: Vec<(f64, usize, usize)>, // (value, first, last)
    /// Free values expanded descending.
    free: Vec<f64>,
    p_req: u32,
    /// Upper bound on the class dot given sorted |x| (computed per query).
    bound: f64,
}

/// Runtime form of an odd class.
struct OddRt {
    shell_idx: usize,
    /// Contribution multiset ascending: a(v) = +v (v ≡ 3 mod 4) else −v.
    a_asc: [f64; DIM],
    bound: f64,
}

/// Winner descriptor, enough to materialize the point once at the end.
#[derive(Clone, Copy)]
enum GBest {
    None,
    Odd {
        class: usize,
        c: u32,
    },
    Even {
        /// Group index in `even_by_w`, or `usize::MAX` for the w = 0 group.
        grp: usize,
        idx: usize,
        c: u32,
        /// Sacrificed word-kind value (goes to the min-|x| support slot
        /// with mismatched sign), if the parity repair was needed.
        sac: Option<f64>,
    },
}

/// The generic engine: class tables grouped for the codeword loops.
pub struct BallSearcher {
    even_by_w: Vec<(u32, Vec<EvenRt>)>,
    even_w0: Vec<EvenRt>,
    odd: Vec<OddRt>,
}

impl BallSearcher {
    pub fn new() -> Self {
        let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
        let mut even_by_w: Vec<(u32, Vec<EvenRt>)> = Vec::new();
        let mut even_w0 = Vec::new();
        for c in even {
            let mut word = Vec::new();
            let mut kinds = Vec::new();
            for &(v, n) in &c.word_vals {
                let first = word.len();
                for _ in 0..n {
                    word.push(v as f64);
                }
                kinds.push((v as f64, first, word.len() - 1));
            }
            let mut free = Vec::new();
            for &(v, n) in &c.free_vals {
                for _ in 0..n {
                    free.push(v as f64);
                }
            }
            let rt = EvenRt {
                shell_idx: (c.shell - 2) as usize,
                word,
                kinds,
                free,
                p_req: c.p_req as u32,
                bound: 0.0,
            };
            if c.w == 0 {
                even_w0.push(rt);
            } else {
                match even_by_w.iter_mut().find(|(w, _)| *w == c.w) {
                    Some((_, v)) => v.push(rt),
                    None => even_by_w.push((c.w, vec![rt])),
                }
            }
        }
        even_by_w.sort_unstable_by_key(|&(w, _)| w);

        let odd = odd
            .into_iter()
            .map(|c| {
                let mut a = Vec::with_capacity(DIM);
                for &(v, n) in &c.vals {
                    let av = if v % 4 == 3 { v as f64 } else { -(v as f64) };
                    for _ in 0..n {
                        a.push(av);
                    }
                }
                debug_assert_eq!(a.len(), DIM);
                a.sort_unstable_by(f64::total_cmp);
                OddRt {
                    shell_idx: (c.shell - 2) as usize,
                    a_asc: a.try_into().expect("24 odd values"),
                    bound: 0.0,
                }
            })
            .collect();

        Self {
            even_by_w,
            even_w0,
            odd,
        }
    }

    /// Exact `argmax ⟨x, v⟩` per shell m = 2..=13. `ws` must be reusable;
    /// the searcher provides the Golay tables.
    pub fn shell_bests(
        &mut self,
        s: &Searcher,
        ws: &mut Workspace,
        x: &[f64; DIM],
    ) -> [Found; NSHELLS] {
        ws.fill(x);
        let g = s.leech().golay();

        let mut best = [f64::NEG_INFINITY; NSHELLS];
        let mut best_at = [GBest::None; NSHELLS];

        // Global |x| order (descending), shared by all even classes.
        let mut order: [usize; DIM] = core::array::from_fn(|i| i);
        order.sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
        let abs_desc: [f64; DIM] = core::array::from_fn(|k| x[order[k]].abs());

        // Per-query class bounds (sorted-pairing upper bounds, support-free:
        // any placement satisfies Σ v·(±x) ≤ Σ sorted(values)·sorted(|x|)).
        for grp in self.even_by_w.iter_mut() {
            for cl in grp.1.iter_mut() {
                let mut vals: Vec<f64> =
                    cl.word.iter().chain(cl.free.iter()).copied().collect();
                vals.sort_unstable_by(|a, b| b.total_cmp(a));
                cl.bound = vals.iter().zip(abs_desc.iter()).map(|(v, a)| v * a).sum();
            }
        }
        for cl in self.even_w0.iter_mut() {
            cl.bound = cl.free.iter().zip(abs_desc.iter()).map(|(v, a)| v * a).sum();
        }
        for cl in self.odd.iter_mut() {
            let mut abs_a: [f64; DIM] = core::array::from_fn(|k| cl.a_asc[k].abs());
            abs_a.sort_unstable_by(|a, b| b.total_cmp(a));
            cl.bound = abs_a.iter().zip(abs_desc.iter()).map(|(v, a)| v * a).sum();
        }

        // w = 0 even classes: codeword 0, free values on the global top |x|.
        for (ci, cl) in self.even_w0.iter().enumerate() {
            let score: f64 = cl.free.iter().zip(abs_desc.iter()).map(|(v, a)| v * a).sum();
            if score > best[cl.shell_idx] {
                best[cl.shell_idx] = score;
                best_at[cl.shell_idx] = GBest::Even {
                    grp: usize::MAX,
                    idx: ci,
                    c: 0,
                    sac: None,
                };
            }
        }

        // Even classes, grouped by Golay weight.
        let mut sup = [0f64; DIM];
        for (gi, (w, group)) in self.even_by_w.iter().enumerate() {
            let w = *w as usize;
            for &c in g.of_weight(w) {
                // Support |x| sorted descending.
                let mut n = 0;
                for (i, xi) in x.iter().enumerate() {
                    if c >> i & 1 == 1 {
                        sup[n] = xi.abs();
                        n += 1;
                    }
                }
                debug_assert_eq!(n, w);
                sup[..w].sort_unstable_by(|a, b| b.total_cmp(a));
                let matched_parity = ws.neg_count(c) % 2;

                // Off-support |x| descending (probe the global order).
                let mut off = [0f64; DIM];
                let mut no = 0;
                let max_free = group.iter().map(|cl| cl.free.len()).max().unwrap_or(0);
                for &p in order.iter() {
                    if no == max_free {
                        break;
                    }
                    if c >> p & 1 == 0 {
                        off[no] = x[p].abs();
                        no += 1;
                    }
                }

                for (ci, cl) in group.iter().enumerate() {
                    if cl.bound <= best[cl.shell_idx] {
                        continue;
                    }
                    let off_score: f64 =
                        cl.free.iter().zip(off.iter()).map(|(v, a)| v * a).sum();
                    let base: f64 =
                        cl.word.iter().zip(sup.iter()).map(|(v, a)| v * a).sum();
                    let (on_score, sac) = if matched_parity == cl.p_req {
                        (base, None)
                    } else {
                        // Exact one-mismatch repair: sacrifice one value of
                        // kind u at the min-|x| slot, repack the rest.
                        // Removing the last occurrence (index j) of u shifts
                        // values j+1..w up one slot: gain D_j =
                        // Σ_{i>j} V_i·(A_{i−1} − A_i); score =
                        // base − u·A_j + D_j − u·A_{w−1}.
                        let mut bestrep = f64::NEG_INFINITY;
                        let mut bestu = 0.0;
                        // Suffix scan: D over j from w−1 down to 0.
                        let mut d = 0.0;
                        let mut k_iter = cl.kinds.len();
                        // kinds are stored desc by value; their `last`
                        // indices increase. Walk j from w−1 downward.
                        let mut j = w;
                        while j > 0 {
                            j -= 1;
                            // If j is the last occurrence of some kind,
                            // evaluate the sacrifice of that kind.
                            while k_iter > 0 && cl.kinds[k_iter - 1].2 == j {
                                let u = cl.kinds[k_iter - 1].0;
                                let cand = base - u * sup[j] + d - u * sup[w - 1];
                                if cand > bestrep {
                                    bestrep = cand;
                                    bestu = u;
                                }
                                k_iter -= 1;
                            }
                            if j > 0 {
                                d += cl.word[j] * (sup[j - 1] - sup[j]);
                            }
                        }
                        (bestrep, Some(bestu))
                    };
                    let score = on_score + off_score;
                    if score > best[cl.shell_idx] {
                        best[cl.shell_idx] = score;
                        best_at[cl.shell_idx] = GBest::Even {
                            grp: gi,
                            idx: ci,
                            c,
                            sac,
                        };
                    }
                }
            }
        }

        // Odd classes: shared y-sort per codeword.
        for &c in g.codewords() {
            let mut y: [f64; DIM] = core::array::from_fn(|i| {
                if c >> i & 1 == 1 {
                    x[i]
                } else {
                    -x[i]
                }
            });
            y.sort_unstable_by(f64::total_cmp);
            for (ci, cl) in self.odd.iter().enumerate() {
                if cl.bound <= best[cl.shell_idx] {
                    continue;
                }
                let score: f64 = cl.a_asc.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                if score > best[cl.shell_idx] {
                    best[cl.shell_idx] = score;
                    best_at[cl.shell_idx] = GBest::Odd { class: ci, c };
                }
            }
        }

        core::array::from_fn(|si| Found {
            point: self.materialize(x, &order, best_at[si]),
            shell: si as u32 + 2,
            dot: best[si],
        })
    }

    /// Exact NN over the ball Λ₂₄(13) ∪ {0} under `metric` at scale 1.
    pub fn nearest_ball13(
        &mut self,
        s: &Searcher,
        ws: &mut Workspace,
        x: &[f64; DIM],
        metric: Metric,
    ) -> Found {
        let bests = self.shell_bests(s, ws, x);
        let mut winner = bests[0];
        let mut key = f64::NEG_INFINITY;
        for f in bests {
            let k = match metric {
                Metric::Euclidean => 2.0 * f.dot - (16 * f.shell) as f64,
                Metric::Angular => f.dot / ((16 * f.shell) as f64).sqrt(),
            };
            if k > key {
                key = k;
                winner = f;
            }
        }
        winner
    }

    fn materialize(&self, x: &[f64; DIM], order: &[usize; DIM], b: GBest) -> Point {
        match b {
            GBest::None => unreachable!("every shell 2..=13 is non-empty"),
            GBest::Odd { class, c } => {
                let cl = &self.odd[class];
                let mut yi: Vec<(f64, usize)> = (0..DIM)
                    .map(|i| {
                        let y = if c >> i & 1 == 1 { x[i] } else { -x[i] };
                        (y, i)
                    })
                    .collect();
                yi.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                let mut p: Point = [0; DIM];
                for (k, &(_, i)) in yi.iter().enumerate() {
                    let a = cl.a_asc[k];
                    let coord = a * if c >> i & 1 == 1 { 1.0 } else { -1.0 };
                    p[i] = coord as i32;
                }
                p
            }
            GBest::Even { grp, idx, c, sac } => {
                let cl = if grp == usize::MAX {
                    &self.even_w0[idx]
                } else {
                    &self.even_by_w[grp].1[idx]
                };
                let mut p: Point = [0; DIM];
                // Support positions sorted by |x| descending.
                let mut supi: Vec<usize> =
                    (0..DIM).filter(|&i| c >> i & 1 == 1).collect();
                supi.sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
                let w = supi.len();
                debug_assert_eq!(w, cl.word.len());
                if w > 0 {
                    let mut vals = cl.word.clone();
                    if let Some(u) = sac {
                        // Remove the last occurrence of u; the sacrificed
                        // value goes to the min-|x| slot with mismatched
                        // sign, the rest pack greedily on the other slots.
                        let j = vals
                            .iter()
                            .rposition(|&v| v == u)
                            .expect("sacrificed kind present");
                        vals.remove(j);
                        for (k, &v) in vals.iter().enumerate() {
                            let i = supi[k];
                            p[i] = if x[i] < 0.0 { -(v as i32) } else { v as i32 };
                        }
                        let i = supi[w - 1];
                        // Mismatched sign on purpose.
                        p[i] = if x[i] < 0.0 { u as i32 } else { -(u as i32) };
                    } else {
                        for (k, &v) in vals.iter().enumerate() {
                            let i = supi[k];
                            p[i] = if x[i] < 0.0 { -(v as i32) } else { v as i32 };
                        }
                    }
                }
                // Free values on the largest off-support |x|.
                let mut fi = 0;
                for &i in order.iter() {
                    if fi == cl.free.len() {
                        break;
                    }
                    if c >> i & 1 == 0 {
                        let v = cl.free[fi];
                        p[i] = if x[i] < 0.0 { -(v as i32) } else { v as i32 };
                        fi += 1;
                    }
                }
                p
            }
        }
    }
}

impl Default for BallSearcher {
    fn default() -> Self {
        Self::new()
    }
}
