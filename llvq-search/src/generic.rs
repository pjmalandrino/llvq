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
//! exact repair is then a single flip of the **smallest** word value, which
//! the sorted pairing has already placed on the smallest support |x|; see
//! `even_dot` for why sacrificing a larger value and repacking the rest can
//! never beat it.
//!
//! ## Phase 2c — why the inner loops are cheap
//!
//! Every maximizer above is a **sorted pairing** `Σᵢ Vᵢ·Aᵢ` of a class value
//! vector `V` against a sorted query vector `A`. `V` is constant over long
//! runs (at most 3 word kinds, 2 free kinds, 5 odd kinds), so with the
//! prefix sums `P` of `A` the sum telescopes to
//!
//! ```text
//! Σᵢ Vᵢ·Aᵢ = Σ_runs (v_r − v_{r+1}) · P[end_r]      (v_{last+1} := 0)
//! ```
//!
//! — **one multiply-add per run**, no per-coordinate work at all
//! (`Lin`). The parity repair telescopes the same way. What is left per
//! codeword is building `P`, and that no longer needs a sort either: the
//! support / off-support orders fall out of one pass over the global |x|
//! order, and the odd `y` order is a two-pointer **merge** of two
//! already-sorted halves (24 comparisons, not a 24-element sort).
//!
//! The class tables are then stored **flat and per shell**, physically
//! reordered per query by decreasing upper bound, so each class loop is a
//! contiguous scan that `break`s — no index indirection and no pointer
//! chasing on the hot path. Data needed only to materialize the single
//! winner (expanded value vectors) lives in a separate cold side table.
//!
//! [`crate::generic_ref`] recomputes the same maxima the naive way, and the
//! `g2c_reference` suite requires exact agreement.

use crate::classes::{enumerate_classes, ClassSet, MAX_SHELL};

/// A cap that excludes nothing: no class of the m ≤ 13 ball has more than
/// five distinct magnitudes.
pub const MAX_LEVELS_ANY: usize = 5;
use crate::{Found, Metric, Searcher};
use llvq_core::{Point, DIM};

/// Number of shells covered (m = 2..=13).
pub const NSHELLS: usize = (MAX_SHELL - 1) as usize;

/// Padded length of the prefix / sorted scratch arrays. A power of two, so a
/// masked index provably stays in range and the bounds check disappears
/// without `unsafe` (the crate is `forbid(unsafe_code)`).
const PLEN: usize = 32;
const PMASK: usize = PLEN - 1;

/// A telescoped sorted pairing: `Σₖ coef[k]·P[pos[k]]`, where `P` holds the
/// prefix sums of the query vector. Unused slots carry `coef = 0`, so the
/// evaluation is branch-free and fully unrolled.
#[derive(Clone, Copy)]
struct Lin<const N: usize> {
    coef: [f64; N],
    pos: [u8; N],
}

impl<const N: usize> Lin<N> {
    /// Build from `(value, count)` runs laid out over consecutive slots from
    /// 0, the values monotone in the order given.
    fn new(vals: &[(u8, u8)]) -> Self {
        let v: Vec<(f64, u8)> = vals.iter().map(|&(v, n)| (v as f64, n)).collect();
        Self::from_values(&v)
    }

    fn from_values(vals: &[(f64, u8)]) -> Self {
        assert!(vals.len() <= N, "run count exceeds the compiled capacity");
        let mut coef = [0.0; N];
        let mut pos = [0u8; N];
        let mut end = 0usize;
        for (k, &(v, n)) in vals.iter().enumerate() {
            end += n as usize;
            // Telescoping coefficient: vₖ − vₖ₊₁, with vₗₐₛₜ₊₁ := 0.
            let next = vals.get(k + 1).map_or(0.0, |&(v, _)| v);
            coef[k] = v - next;
            pos[k] = end as u8;
        }
        Self { coef, pos }
    }

    #[inline]
    fn score(&self, p: &[f64; PLEN]) -> f64 {
        let mut acc = 0.0;
        for k in 0..N {
            acc += self.coef[k] * p[self.pos[k] as usize & PMASK];
        }
        acc
    }
}

/// Hot record of an odd class: bound, score form, and its side-table id.
///
/// `bound` holds the per-query upper bound **in the objective of the table
/// this record lives in** — the raw dot product in the per-shell tables of
/// [`BallSearcher::shell_bests`], the β-objective in the flat tables of
/// [`BallSearcher::nearest_scaled`]. Each pass recomputes it before use.
#[derive(Clone, Copy)]
struct OddHot {
    bound: f64,
    /// `a(v)` ascending, paired with `y` ascending.
    a: Lin<5>,
    /// `‖v‖² = 16m` and `1/‖v‖` — the shell radius, for the objectives.
    norm: f64,
    inv_r: f64,
    shell: u8,
    meta: u16,
}

/// Hot record of an even class within one Golay-weight group. `bound` has
/// the same table-relative meaning as in [`OddHot`].
#[derive(Clone, Copy)]
struct EvenHot {
    bound: f64,
    /// Word values descending, paired with support |x| descending.
    word: Lin<3>,
    /// Free values descending, paired with off-support |x| descending.
    free: Lin<2>,
    /// Smallest word value — the one the parity repair flips.
    w_min: f64,
    p_req: u8,
    /// `‖v‖² = 16m` and `1/‖v‖` — the shell radius, for the objectives.
    norm: f64,
    inv_r: f64,
    shell: u8,
    meta: u16,
}

/// Cold side table: the per-query bound form plus everything needed only to
/// materialize the one winner.
struct OddCold {
    /// `|a(v)|` descending, paired with global |x| descending — the bound.
    absl: Lin<5>,
    a_asc: [f64; DIM],
}

struct EvenCold {
    /// Word ∪ free values descending over all 24 slots — the bound.
    all: Lin<5>,
    word_flat: Vec<f64>,
    free_flat: Vec<f64>,
    w: usize,
    shell_idx: usize,
}

/// What the encoder is maximizing over the codebook.
///
/// Both objectives are **strictly increasing in the dot product** at fixed
/// shell, which is the only property the pruning needs: a class whose dot
/// bound is `B` has objective bound `key(B, shell)`, so one global threshold
/// orders every class of every shell.
#[derive(Clone, Copy)]
enum Obj {
    /// `2β⟨x,v⟩ − β²‖v‖²` — maximizing it minimizes `‖x − βv‖²`
    /// (spherical shaping, the ball `β·(Λ₂₄(13) ∪ {0})`).
    Euclidean { two_beta: f64, beta2: f64 },
    /// `⟨x, v⟩/‖v‖ = ⟨x, v̂⟩` — the projection onto the unit direction
    /// (shape–gain: the codebook is the *normalized* ball, and the magnitude
    /// is left to a separate gain quantizer or to a closed-form scale).
    Angular,
}

impl Obj {
    #[inline]
    fn key(&self, dot: f64, norm: f64, inv_r: f64) -> f64 {
        match *self {
            Obj::Euclidean { two_beta, beta2 } => two_beta * dot - beta2 * norm,
            Obj::Angular => dot * inv_r,
        }
    }
}

/// Winner descriptor, enough to materialize the point once at the end.
#[derive(Clone, Copy)]
enum GBest {
    None,
    Odd {
        meta: u16,
        c: u32,
    },
    Even {
        meta: u16,
        c: u32,
        /// Whether the parity repair fired — the smallest word value, on the
        /// smallest support |x|, then carries a mismatched sign.
        sac: bool,
    },
}

/// One Golay-weight group, in two layouts of the same classes: bucketed per
/// shell for [`BallSearcher::shell_bests`], and flat (shells mixed) for
/// [`BallSearcher::nearest_scaled`], whose single global threshold wants one
/// list to sort and one place to break. Both are a few hundred records; the
/// duplication buys each pass a contiguous, correctly ordered scan.
struct Group {
    w: usize,
    shells: [Vec<EvenHot>; NSHELLS],
    flat: Vec<EvenHot>,
}

/// The generic engine: flat per-shell class tables plus cold side tables.
pub struct BallSearcher {
    even_by_w: Vec<Group>,
    /// `w = 0` even classes (codeword 0 only, 17 of them): cold enough to
    /// keep as a plain list of side-table ids.
    even_w0: Vec<u16>,
    odd_by_shell: [Vec<OddHot>; NSHELLS],
    odd_flat: Vec<OddHot>,
    even_cold: Vec<EvenCold>,
    odd_cold: Vec<OddCold>,
    scratch: Scratch,
    /// Highest shell the search may return. Lowering it shrinks the codebook
    /// — `Λ₂₄(12)` needs 47 index bits instead of 48, which buys a gain bit
    /// at the same total rate (the paper's Table 8 best configuration).
    shell_cap: u32,
}

/// Per-query scratch buffers (kept in the searcher to stay allocation-free).
///
/// The per-codeword partitions below are written **branchlessly**: each
/// element is stored to both destination arrays and only the matching cursor
/// advances, so the wrong store is overwritten later. The membership bit is
/// a coin flip on 24 coordinates × 8191 codewords — a predicted branch there
/// mispredicts about half the time, which costs more than the redundant
/// stores by a wide margin.
struct Scratch {
    /// Bit i set iff `xᵢ < 0`. The even parity repair needs
    /// `#{i ∈ c : xᵢ < 0} mod 2`, which is one masked popcount — the reason
    /// this engine needs no [`Workspace`] subset tables at all.
    neg_mask: u32,
    /// |x| descending and its prefix sums.
    abs_desc: [f64; DIM],
    p_abs: [f64; PLEN],
    /// Positions by |x| descending (support split + materialization).
    order: [usize; DIM],
    /// Positions by xᵢ ascending, and the matching values (odd merge).
    asc_idx: [usize; DIM],
    asc_val: [f64; DIM],
    /// Support / off-support |x| descending, with their prefix sums.
    sup: [f64; PLEN],
    p_sup: [f64; PLEN],
    off: [f64; PLEN],
    p_off: [f64; PLEN],
    /// Odd merge halves, each ascending and `+∞`-terminated so the merge
    /// needs no exhaustion test; only the prefix sums of `y` are ever kept.
    in_asc: [f64; PLEN],
    out_asc: [f64; PLEN],
    p_y: [f64; PLEN],
}

impl Scratch {
    fn new() -> Self {
        Self {
            neg_mask: 0,
            abs_desc: [0.0; DIM],
            p_abs: [0.0; PLEN],
            order: [0; DIM],
            asc_idx: [0; DIM],
            asc_val: [0.0; DIM],
            sup: [0.0; PLEN],
            p_sup: [0.0; PLEN],
            off: [0.0; PLEN],
            p_off: [0.0; PLEN],
            in_asc: [f64::INFINITY; PLEN],
            out_asc: [f64::INFINITY; PLEN],
            p_y: [0.0; PLEN],
        }
    }

    /// Branchless split of the global |x| order along the support of `c`
    /// into `sup` / `off` (both descending), with their prefix sums.
    /// `w` must be `c.count_ones()`.
    #[inline]
    fn split_support(&mut self, c: u32, w: usize) {
        let (mut ns, mut no) = (0usize, 0usize);
        for k in 0..DIM {
            let a = self.abs_desc[k];
            let bit = (c >> self.order[k] & 1) as usize;
            self.sup[ns & PMASK] = a;
            self.off[no & PMASK] = a;
            ns += bit;
            no += 1 - bit;
        }
        debug_assert_eq!(ns, w);
        self.p_sup[0] = 0.0;
        for k in 0..w {
            self.p_sup[k + 1] = self.p_sup[k] + self.sup[k];
        }
        self.p_off[0] = 0.0;
        for k in 0..DIM - w {
            self.p_off[k + 1] = self.p_off[k] + self.off[k];
        }
    }

    /// Prefix sums of `yᵢ = (i ∈ c ? xᵢ : −xᵢ)` sorted ascending.
    ///
    /// The off-support half is filled from the back: `−x` runs descending
    /// along the ascending-`x` order, so writing it in reverse lands it
    /// ascending. Two sorted halves then merge in 24 comparisons, and `y`
    /// itself is never stored — it feeds straight into `p_y`.
    #[inline]
    fn merge_y(&mut self, c: u32) {
        let ni_tot = c.count_ones() as usize;
        let no_tot = DIM - ni_tot;
        let (mut ni, mut no) = (0usize, 0usize);
        for k in 0..DIM {
            let v = self.asc_val[k];
            let bit = (c >> self.asc_idx[k] & 1) as usize;
            self.in_asc[ni & PMASK] = v;
            // `no` can reach `no_tot` before the pass ends; the `+ PLEN`
            // keeps the index unsigned and the mask parks those spare
            // stores on the unused slot 31.
            self.out_asc[(no_tot + PLEN - 1 - no) & PMASK] = -v;
            ni += bit;
            no += 1 - bit;
        }
        debug_assert_eq!(ni, ni_tot);
        self.in_asc[ni_tot] = f64::INFINITY;
        self.out_asc[no_tot] = f64::INFINITY;
        let (mut ia, mut ib) = (0usize, 0usize);
        self.p_y[0] = 0.0;
        for k in 0..DIM {
            let vi = self.in_asc[ia & PMASK];
            let vo = self.out_asc[ib & PMASK];
            let take_in = vi <= vo;
            let y = if take_in { vi } else { vo };
            ia += take_in as usize;
            ib += !take_in as usize;
            self.p_y[k + 1] = self.p_y[k] + y;
        }
    }
}

/// `max ⟨x, v⟩` over the arrangements of one even class on one codeword, and
/// whether the parity repair fired.
///
/// The repair is a **single case**: flip the smallest word value, which the
/// sorted pairing already placed on the smallest support |x|. Sacrificing a
/// larger kind `u` at its last slot `j` and repacking the values after it
/// scores `base − u·A_j + D_j − u·A_{w−1}` with
/// `D_j = Σ_{i>j} Vᵢ(A_{i−1} − Aᵢ)`; expanding that telescoping sum gives
///
/// ```text
/// score(j) − score(w−1) = Σ_{t=j}^{w−2} (V_{t+1} − V_t)·A_t + (V_{w−1} − V_j)·A_{w−1}
/// ```
///
/// and every term is ≤ 0, since `V` is descending and `A ≥ 0`. So no repack
/// can ever beat the plain flip. Two tests hold this down:
/// `even_repair_matches_dp_reference` compares it to an exhaustive
/// assignment DP, and `generic_ref` — which still tries every kind at every
/// slot — pins it end to end.
///
/// `a_last` is the smallest support |x| and `matched_parity` the sign parity
/// of the codeword; both are shared by every class of the group.
#[inline]
fn even_dot(h: &EvenHot, sc: &Scratch, matched_parity: u8, a_last: f64) -> (f64, bool) {
    let base = h.word.score(&sc.p_sup);
    let repair = matched_parity != h.p_req;
    let on_score = if repair {
        base - 2.0 * h.w_min * a_last
    } else {
        base
    };
    (on_score + h.free.score(&sc.p_off), repair)
}

impl BallSearcher {
    pub fn new() -> Self {
        Self::with_level_cap(MAX_LEVELS_ANY)
    }

    /// Build a searcher restricted to classes with at most `cap` distinct
    /// magnitudes per block — **zero included**, so `{6, 4, 2, 0}` counts as
    /// four.
    ///
    /// This is a codebook restriction, not a search heuristic: the excluded
    /// points simply do not exist for this searcher, and the index built over
    /// the same filter stays bijective. It exists because the runtime layout's
    /// width is `34 + 24(L−1)` bits — the level count *is* the memory cost, so
    /// capping it is the one knob that trades distortion for RAM directly.
    ///
    /// Removing entries from tables that are sorted by bound keeps them
    /// sorted, so every prune and every `break` stays valid.
    pub fn with_level_cap(cap: usize) -> Self {
        let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
        // Level count of an even class: its distinct word/free values, plus
        // zero when any coordinate is off both.
        let even: Vec<_> = even.into_iter().filter(|c| c.n_levels() <= cap).collect();
        let odd: Vec<_> = odd.into_iter().filter(|c| c.n_levels() <= cap).collect();

        let mut even_by_w: Vec<Group> = Vec::new();
        let mut even_w0 = Vec::new();
        let mut even_cold: Vec<EvenCold> = Vec::new();
        for c in even {
            let w = c.w as usize;
            let shell_idx = (c.shell - 2) as usize;
            let meta = u16::try_from(even_cold.len()).expect("class count fits u16");
            let mut all_vals: Vec<(u8, u8)> = c
                .word_vals
                .iter()
                .chain(c.free_vals.iter())
                .copied()
                .collect();
            all_vals.sort_unstable_by_key(|v| core::cmp::Reverse(v.0));
            even_cold.push(EvenCold {
                all: Lin::new(&all_vals),
                word_flat: flatten(&c.word_vals),
                free_flat: flatten(&c.free_vals),
                w,
                shell_idx,
            });

            if w == 0 {
                even_w0.push(meta);
                continue;
            }
            let hot = EvenHot {
                bound: 0.0,
                word: Lin::new(&c.word_vals),
                free: Lin::new(&c.free_vals),
                w_min: c.word_vals.last().expect("w > 0 implies a word value").0 as f64,
                p_req: c.p_req,
                norm: (16 * c.shell) as f64,
                inv_r: 1.0 / ((16 * c.shell) as f64).sqrt(),
                shell: c.shell as u8,
                meta,
            };
            match even_by_w.iter_mut().find(|g| g.w == w) {
                Some(g) => {
                    g.shells[shell_idx].push(hot);
                    g.flat.push(hot);
                }
                None => {
                    let mut g = Group {
                        w,
                        shells: core::array::from_fn(|_| Vec::new()),
                        flat: vec![hot],
                    };
                    g.shells[shell_idx].push(hot);
                    even_by_w.push(g);
                }
            }
        }
        even_by_w.sort_unstable_by_key(|g| g.w);

        let mut odd_cold: Vec<OddCold> = Vec::new();
        let mut odd_by_shell: [Vec<OddHot>; NSHELLS] = core::array::from_fn(|_| Vec::new());
        let mut odd_flat: Vec<OddHot> = Vec::new();
        for c in odd {
            let shell_idx = (c.shell - 2) as usize;
            // a(v) = +v for v ≡ 3 (mod 4), −v otherwise. Ascending order:
            // the negatives by decreasing |v| first, then the positives.
            let mut asc: Vec<(f64, u8)> = Vec::with_capacity(c.vals.len());
            for &(v, n) in &c.vals {
                if v % 4 != 3 {
                    asc.push((-(v as f64), n));
                }
            }
            for &(v, n) in c.vals.iter().rev() {
                if v % 4 == 3 {
                    asc.push((v as f64, n));
                }
            }
            debug_assert!(asc.windows(2).all(|p| p[0].0 < p[1].0));
            let mut flat = Vec::with_capacity(DIM);
            for &(v, n) in &asc {
                for _ in 0..n {
                    flat.push(v);
                }
            }
            let meta = u16::try_from(odd_cold.len()).expect("class count fits u16");
            odd_cold.push(OddCold {
                // |a(v)| = v, and `c.vals` is already descending.
                absl: Lin::new(&c.vals),
                a_asc: flat.try_into().expect("24 odd values"),
            });
            let hot = OddHot {
                bound: 0.0,
                a: Lin::from_values(&asc),
                norm: (16 * c.shell) as f64,
                inv_r: 1.0 / ((16 * c.shell) as f64).sqrt(),
                shell: c.shell as u8,
                meta,
            };
            odd_by_shell[shell_idx].push(hot);
            odd_flat.push(hot);
        }

        Self {
            even_by_w,
            even_w0,
            odd_by_shell,
            odd_flat,
            even_cold,
            odd_cold,
            scratch: Scratch::new(),
            shell_cap: MAX_SHELL,
        }
    }

    /// Restrict the search to shells `2..=cap`.
    ///
    /// The class tables keep their bounds and their order, so the pruning
    /// stays valid: a skipped class could not have improved the incumbent
    /// anyway, and the `break` still fires on a bound that dominates every
    /// later entry.
    pub fn set_shell_cap(&mut self, cap: u32) {
        assert!((2..=MAX_SHELL).contains(&cap), "cap must be in 2..=13");
        self.shell_cap = cap;
    }

    pub fn shell_cap(&self) -> u32 {
        self.shell_cap
    }

    /// Exact `argmax ⟨x, v⟩` per shell m = 2..=13. `ws` must be reusable;
    /// the searcher provides the Golay tables.
    pub fn shell_bests(&mut self, s: &Searcher, x: &[f64; DIM]) -> [Found; NSHELLS] {
        let g = s.leech().golay();
        let sc = &mut self.scratch;

        let mut best = [f64::NEG_INFINITY; NSHELLS];
        let mut best_at = [GBest::None; NSHELLS];

        // Global orders: |x| descending (even classes) and x ascending (odd).
        sc.neg_mask = 0;
        for (i, xi) in x.iter().enumerate() {
            sc.neg_mask |= ((*xi < 0.0) as u32) << i;
        }
        for (i, sl) in sc.order.iter_mut().enumerate() {
            *sl = i;
        }
        sc.order
            .sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
        sc.p_abs[0] = 0.0;
        for k in 0..DIM {
            sc.abs_desc[k] = x[sc.order[k]].abs();
            sc.p_abs[k + 1] = sc.p_abs[k] + sc.abs_desc[k];
        }
        for (i, sl) in sc.asc_idx.iter_mut().enumerate() {
            *sl = i;
        }
        sc.asc_idx.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
        for k in 0..DIM {
            sc.asc_val[k] = x[sc.asc_idx[k]];
        }

        // Per-query bounds: pair the class values (descending) with |x|
        // (descending), ignoring the support split — no placement of the
        // class can score more (rearrangement inequality). Then sort each
        // per-shell table by decreasing bound so the class loops can break.
        for grp in self.even_by_w.iter_mut() {
            for shelf in grp.shells.iter_mut() {
                for h in shelf.iter_mut() {
                    h.bound = self.even_cold[h.meta as usize].all.score(&sc.p_abs);
                }
                shelf.sort_unstable_by(|a, b| b.bound.total_cmp(&a.bound));
            }
        }
        for shelf in self.odd_by_shell.iter_mut() {
            for h in shelf.iter_mut() {
                h.bound = self.odd_cold[h.meta as usize].absl.score(&sc.p_abs);
            }
            shelf.sort_unstable_by(|a, b| b.bound.total_cmp(&a.bound));
        }

        // w = 0 even classes: codeword 0, free values on the global top |x|
        // — the bound is attained, so it *is* the score.
        for &meta in self.even_w0.iter() {
            let cold = &self.even_cold[meta as usize];
            let score = cold.all.score(&sc.p_abs);
            if score > best[cold.shell_idx] {
                best[cold.shell_idx] = score;
                best_at[cold.shell_idx] = GBest::Even {
                    meta,
                    c: 0,
                    sac: false,
                };
            }
        }

        // Even classes, grouped by Golay weight.
        for grp in self.even_by_w.iter() {
            let w = grp.w;
            for &c in g.of_weight(w) {
                sc.split_support(c, w);
                let matched_parity = ((c & sc.neg_mask).count_ones() % 2) as u8;
                let a_last = sc.sup[(w - 1) & PMASK];

                for (si, shelf) in grp.shells.iter().enumerate() {
                    for h in shelf.iter() {
                        if h.bound <= best[si] {
                            break; // table is in decreasing bound order
                        }
                        let (score, sac) = even_dot(h, sc, matched_parity, a_last);
                        if score > best[si] {
                            best[si] = score;
                            best_at[si] = GBest::Even {
                                meta: h.meta,
                                c,
                                sac,
                            };
                        }
                    }
                }
            }
        }

        // Odd classes: `y` ascending per codeword, then run-wise scores.
        for &c in g.codewords() {
            // yᵢ = (i ∈ c ? xᵢ : −xᵢ). Split the global ascending order in
            // two: the in-support part stays ascending, the off-support part
            // negates to descending — so `y` is a two-pointer merge, not a
            // sort, and it is consumed straight into its prefix sums.
            sc.merge_y(c);

            for (si, shelf) in self.odd_by_shell.iter().enumerate() {
                for h in shelf.iter() {
                    if h.bound <= best[si] {
                        break; // table is in decreasing bound order
                    }
                    let score = h.a.score(&sc.p_y);
                    if score > best[si] {
                        best[si] = score;
                        best_at[si] = GBest::Odd { meta: h.meta, c };
                    }
                }
            }
        }

        let order = sc.order;
        core::array::from_fn(|si| Found {
            point: self.materialize(x, &order, best_at[si]),
            shell: si as u32 + 2,
            dot: best[si],
        })
    }

    /// Exact NN over the ball Λ₂₄(13) ∪ {0} under `metric` at scale 1.
    pub fn nearest_ball13(&mut self, s: &Searcher, x: &[f64; DIM], metric: Metric) -> Found {
        let bests = self.shell_bests(s, x);
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

    /// Exact nearest codeword of `β·(Λ₂₄(13) ∪ {0})` — **spherical
    /// shaping**, the production encoder path for a Euclidean quantizer.
    ///
    /// Returns the **unscaled** lattice point `v` (reconstruct with `β·v`);
    /// `shell = 0` with a zero point means the origin won.
    pub fn nearest_scaled(&mut self, s: &Searcher, x: &[f64; DIM], beta: f64) -> Found {
        assert!(beta > 0.0, "scale must be positive");
        // The origin is a codeword of the ball, and its key is exactly 0.
        self.nearest_by(
            s,
            x,
            Obj::Euclidean {
                two_beta: 2.0 * beta,
                beta2: beta * beta,
            },
            0.0,
        )
    }

    /// Exact `argmax ⟨x, v⟩/‖v‖` over `Λ₂₄(13)` — the **shape** quantizer of
    /// shape–gain, i.e. the nearest point of the *normalized* ball on the
    /// unit sphere.
    ///
    /// This is the direction quantizer `Q_dir` of the paper's Algorithm 1,
    /// and the configuration its Table 8 and Appendix I recommend: with a
    /// code of low angular distortion, capacity is better spent entirely on
    /// directions, magnitudes being carried by the norm-preserving GPTQ step
    /// and a closed-form scale afterwards.
    ///
    /// Returns the **unscaled** lattice point `v`; the direction is
    /// `v/‖v‖ = v/√(16·shell)` and `dot/‖v‖` is the projection `⟨x, v̂⟩` the
    /// gain quantizer has to encode. The origin is *not* a candidate — a
    /// direction is always required.
    pub fn nearest_angular(&mut self, s: &Searcher, x: &[f64; DIM]) -> Found {
        self.nearest_by(s, x, Obj::Angular, f64::NEG_INFINITY)
    }

    /// Shared encoder core: one global threshold over every class of every
    /// shell.
    ///
    /// [`Self::shell_bests`] answers twelve independent questions and so
    /// carries twelve independent pruning thresholds; a quantizer only ever
    /// asks one. Ranking every class by a single objective lets one
    /// threshold prune *across* shells. Two consequences the per-shell pass
    /// cannot have: sections (each Golay-weight group, and the whole odd
    /// coset) are visited in decreasing top bound, so the best incumbent is
    /// found first; and a section whose best class cannot reach that
    /// incumbent is skipped outright — 4096 codewords at a time.
    ///
    /// `best0` seeds the threshold with any codeword that needs no search
    /// (the origin, at key 0, for a ball; `−∞` when a direction is
    /// mandatory).
    fn nearest_by(&mut self, s: &Searcher, x: &[f64; DIM], obj: Obj, best0: f64) -> Found {
        let g = s.leech().golay();
        let cap = self.shell_cap as u8;
        let sc = &mut self.scratch;

        sc.neg_mask = 0;
        for (i, xi) in x.iter().enumerate() {
            sc.neg_mask |= ((*xi < 0.0) as u32) << i;
        }
        for (i, sl) in sc.order.iter_mut().enumerate() {
            *sl = i;
        }
        sc.order
            .sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
        sc.p_abs[0] = 0.0;
        for k in 0..DIM {
            sc.abs_desc[k] = x[sc.order[k]].abs();
            sc.p_abs[k + 1] = sc.p_abs[k] + sc.abs_desc[k];
        }
        for (i, sl) in sc.asc_idx.iter_mut().enumerate() {
            *sl = i;
        }
        sc.asc_idx.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
        for k in 0..DIM {
            sc.asc_val[k] = x[sc.asc_idx[k]];
        }

        // Per-query bounds, in the objective, and the resulting order.
        for grp in self.even_by_w.iter_mut() {
            for h in grp.flat.iter_mut() {
                let b = self.even_cold[h.meta as usize].all.score(&sc.p_abs);
                h.bound = obj.key(b, h.norm, h.inv_r);
            }
            grp.flat.sort_unstable_by(|a, b| b.bound.total_cmp(&a.bound));
        }
        for h in self.odd_flat.iter_mut() {
            let b = self.odd_cold[h.meta as usize].absl.score(&sc.p_abs);
            h.bound = obj.key(b, h.norm, h.inv_r);
        }
        self.odd_flat
            .sort_unstable_by(|a, b| b.bound.total_cmp(&a.bound));

        let mut best = best0;
        let mut best_dot = 0.0f64;
        let mut best_shell = 0u32;
        let mut best_at = GBest::None;

        // w = 0 even classes: codeword 0, free values on the global top |x|
        // — the bound is attained, so it *is* the score.
        for &meta in self.even_w0.iter() {
            let cold = &self.even_cold[meta as usize];
            let shell = cold.shell_idx as u32 + 2;
            if shell > cap as u32 {
                continue;
            }
            let dot = cold.all.score(&sc.p_abs);
            let norm = (16 * shell) as f64;
            let key = obj.key(dot, norm, 1.0 / norm.sqrt());
            if key > best {
                best = key;
                best_dot = dot;
                best_shell = shell;
                best_at = GBest::Even {
                    meta,
                    c: 0,
                    sac: false,
                };
            }
        }

        // Sections in decreasing top bound: `usize::MAX` marks the odd coset.
        let mut sections: Vec<(f64, usize)> = self
            .even_by_w
            .iter()
            .enumerate()
            .map(|(gi, grp)| (grp.flat[0].bound, gi))
            .collect();
        sections.push((self.odd_flat[0].bound, usize::MAX));
        sections.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        for &(top, sec) in sections.iter() {
            if top <= best {
                break; // sorted descending: no later section can win either
            }
            if sec == usize::MAX {
                for &c in g.codewords() {
                    if self.odd_flat[0].bound <= best {
                        break; // the whole coset is exhausted
                    }
                    sc.merge_y(c);
                    for h in self.odd_flat.iter() {
                        if h.bound <= best {
                            break; // table is in decreasing bound order
                        }
                        if h.shell > cap {
                            continue;
                        }
                        let dot = h.a.score(&sc.p_y);
                        let key = obj.key(dot, h.norm, h.inv_r);
                        if key > best {
                            best = key;
                            best_dot = dot;
                            best_shell = h.shell as u32;
                            best_at = GBest::Odd { meta: h.meta, c };
                        }
                    }
                }
            } else {
                let grp = &self.even_by_w[sec];
                let w = grp.w;
                for &c in g.of_weight(w) {
                    if grp.flat[0].bound <= best {
                        break; // the whole weight group is exhausted
                    }
                    sc.split_support(c, w);
                    let matched_parity = ((c & sc.neg_mask).count_ones() % 2) as u8;
                    let a_last = sc.sup[(w - 1) & PMASK];
                    for h in grp.flat.iter() {
                        if h.bound <= best {
                            break;
                        }
                        if h.shell > cap {
                            continue;
                        }
                        let (dot, sac) = even_dot(h, sc, matched_parity, a_last);
                        let key = obj.key(dot, h.norm, h.inv_r);
                        if key > best {
                            best = key;
                            best_dot = dot;
                            best_shell = h.shell as u32;
                            best_at = GBest::Even {
                                meta: h.meta,
                                c,
                                sac,
                            };
                        }
                    }
                }
            }
        }

        if matches!(best_at, GBest::None) {
            // Only reachable for a ball objective, where the origin wins.
            return Found {
                point: [0; DIM],
                shell: 0,
                dot: 0.0,
            };
        }
        let order = sc.order;
        Found {
            point: self.materialize(x, &order, best_at),
            shell: best_shell,
            dot: best_dot,
        }
    }

    fn materialize(&self, x: &[f64; DIM], order: &[usize; DIM], b: GBest) -> Point {
        match b {
            GBest::None => unreachable!("every shell 2..=13 is non-empty"),
            GBest::Odd { meta, c } => {
                let cold = &self.odd_cold[meta as usize];
                let mut yi: Vec<(f64, usize)> = (0..DIM)
                    .map(|i| {
                        let y = if c >> i & 1 == 1 { x[i] } else { -x[i] };
                        (y, i)
                    })
                    .collect();
                yi.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                let mut p: Point = [0; DIM];
                for (k, &(_, i)) in yi.iter().enumerate() {
                    let a = cold.a_asc[k];
                    let coord = a * if c >> i & 1 == 1 { 1.0 } else { -1.0 };
                    p[i] = coord as i32;
                }
                p
            }
            GBest::Even { meta, c, sac } => {
                let cold = &self.even_cold[meta as usize];
                let mut p: Point = [0; DIM];
                // Support positions sorted by |x| descending.
                let mut supi: Vec<usize> = (0..DIM).filter(|&i| c >> i & 1 == 1).collect();
                supi.sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
                let w = supi.len();
                debug_assert_eq!(w, cold.w);
                // Word values pair with the support |x| descending, signs
                // matched — except the last one, which the parity repair
                // flips on purpose.
                for (k, &v) in cold.word_flat.iter().enumerate() {
                    let i = supi[k];
                    let neg = (x[i] < 0.0) ^ (sac && k == w - 1);
                    p[i] = if neg { -(v as i32) } else { v as i32 };
                }
                // Free values on the largest off-support |x|.
                let mut fi = 0;
                for &i in order.iter() {
                    if fi == cold.free_flat.len() {
                        break;
                    }
                    if c >> i & 1 == 0 {
                        let v = cold.free_flat[fi];
                        p[i] = if x[i] < 0.0 { -(v as i32) } else { v as i32 };
                        fi += 1;
                    }
                }
                p
            }
        }
    }
}

fn flatten(vals: &[(u8, u8)]) -> Vec<f64> {
    let mut out = Vec::new();
    for &(v, n) in vals {
        for _ in 0..n {
            out.push(v as f64);
        }
    }
    out
}

impl Default for BallSearcher {
    fn default() -> Self {
        Self::new()
    }
}
