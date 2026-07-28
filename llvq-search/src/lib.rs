//! # llvq-search — Phase 2
//!
//! Exact nearest-neighbour search on unions of Leech lattice shells,
//! generalizing the single-shell algorithm of Adoul & Barth (1988) as
//! described in §3.1 of the LLVQ paper (arXiv:2603.11021).
//!
//! ## Principle
//!
//! Within one equivalence class (fixed absolute-value pattern + parity
//! type), the maximum of `⟨x, v⟩` has a **closed form**: assign the free
//! signs to match `sign(xᵢ)`, and when the class carries a minus-count
//! parity constraint, repair a violation by flipping the coordinate of
//! minimal `|xᵢ|` (any valid configuration differs from the sign-matched
//! one by an odd number of flips, each costing `2·value·|xᵢ|`, so a single
//! minimal flip is optimal). Codeword-indexed classes reduce to per-codeword
//! affine scores `S − 2·T_c + …` maximized over 24 positions. The search
//! therefore ranks 759–4096 closed-form class maxima instead of enumerating
//! up to 16.7 M lattice points — and it is *exact*, not approximate.
//!
//! ## Half-word tables
//!
//! Per query we build subset-sum DP tables over the two 12-coordinate
//! halves (4096 entries each): `Σxᵢ`, `Σ|xᵢ|`, negative-coordinate count,
//! and min/max trackers. Every per-codeword quantity then costs two table
//! lookups instead of a 24-iteration branchy loop, and an upper bound
//! (`base + 4·k·max|x|`) prunes codewords that cannot beat the running
//! best. This is what makes the search fast enough to be an encoder.
//!
//! On a single shell all candidates share a norm, so Euclidean and angular
//! rankings coincide; across shells they differ (§3.1 of the paper):
//! Euclidean minimizes `‖x‖² + ‖v‖² − 2⟨x,v⟩`, angular maximizes
//! `⟨x,v⟩ / ‖v‖`. Both are supported.
//!
//! Shells currently covered: m = 2 and m = 3 (hand-derived class
//! decompositions, cross-checked by the G1/G2 suites). The generic
//! leader-table engine for m ≤ 13 (2 bits/dim regime) is Phase 2b.

#![forbid(unsafe_code)]

pub mod classes;
pub mod generic;
pub mod index;

use llvq_core::{Leech, Point, DIM};

/// Ranking metric across shells (§3.1 of the paper).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Minimize `‖x − v‖²` (spherical-shaping regime).
    Euclidean,
    /// Maximize `⟨x, v⟩ / ‖v‖` (shape–gain regime).
    Angular,
}

/// A search result: the argmax lattice point and its raw dot product.
#[derive(Clone, Copy, Debug)]
pub struct Found {
    /// The nearest lattice point (integer embedding, `‖point‖² = 16·shell`).
    pub point: Point,
    /// Shell index m.
    pub shell: u32,
    /// `⟨x, point⟩`.
    pub dot: f64,
}

impl Found {
    /// Squared Euclidean distance to `x`.
    pub fn dist2(&self, x: &[f64; DIM]) -> f64 {
        let xx: f64 = x.iter().map(|v| v * v).sum();
        xx + (16 * self.shell) as f64 - 2.0 * self.dot
    }

    /// Cosine similarity with `x` (0 if `x = 0`).
    pub fn cosine(&self, x: &[f64; DIM]) -> f64 {
        let xx: f64 = x.iter().map(|v| v * v).sum();
        if xx == 0.0 {
            return 0.0;
        }
        self.dot / (xx.sqrt() * ((16 * self.shell) as f64).sqrt())
    }
}

const CHUNK: usize = 8;
const CMASK: u32 = 0xFF;
const NCHUNK: usize = 3;
const NONE: u8 = u8::MAX;

/// Subset DP tables over one 8-coordinate chunk (256 entries per quantity).
/// Three chunks cover the 24 coordinates; all tables together fit in L1,
/// and a per-query rebuild costs only 3·256 DP steps per quantity.
struct Chunk {
    /// `Σ xᵢ` over the mask.
    t: [f64; 256],
    /// `Σ |xᵢ|` over the mask.
    a: [f64; 256],
    /// Number of `xᵢ < 0` over the mask.
    neg: [u8; 256],
    /// `(min |xᵢ|, local index)` over the mask.
    min_abs: [(f64, u8); 256],
    /// `(max xᵢ, local index)` over the mask.
    max_in: [(f64, u8); 256],
    /// `(min xᵢ, local index)` over the mask.
    min_in: [(f64, u8); 256],
    /// Top-3 of `yᵢ = (i ∈ mask ? xᵢ : −xᵢ)` over the chunk's 8 positions,
    /// as (values desc, local indices). Any global top-3 element is in its
    /// chunk's top-3, so merging three tables is exact.
    top3: [([f64; 3], [u8; 3]); 256],
}

impl Chunk {
    fn new() -> Self {
        Self {
            t: [0.0; 256],
            a: [0.0; 256],
            neg: [0; 256],
            min_abs: [(f64::INFINITY, NONE); 256],
            max_in: [(f64::NEG_INFINITY, NONE); 256],
            min_in: [(f64::INFINITY, NONE); 256],
            top3: [([f64::NEG_INFINITY; 3], [NONE; 3]); 256],
        }
    }

    /// O(256) incremental fill: entry `s` extends `s` minus its lowest bit.
    fn fill(&mut self, xs: &[f64]) {
        debug_assert_eq!(xs.len(), CHUNK);
        for s in 1usize..256 {
            let i = s.trailing_zeros() as usize;
            let r = s & (s - 1);
            let v = xs[i];
            let av = v.abs();
            self.t[s] = self.t[r] + v;
            self.a[s] = self.a[r] + av;
            self.neg[s] = self.neg[r] + (v < 0.0) as u8;
            self.min_abs[s] = if av < self.min_abs[r].0 {
                (av, i as u8)
            } else {
                self.min_abs[r]
            };
            self.max_in[s] = if v > self.max_in[r].0 {
                (v, i as u8)
            } else {
                self.max_in[r]
            };
            self.min_in[s] = if v < self.min_in[r].0 {
                (v, i as u8)
            } else {
                self.min_in[r]
            };
        }
        // Top-3 signed values per mask (direct 8-element pass per entry).
        for s in 0usize..256 {
            let mut vals = [f64::NEG_INFINITY; 3];
            let mut idx = [NONE; 3];
            for (i, &v) in xs.iter().enumerate() {
                let y = if s >> i & 1 == 1 { v } else { -v };
                if y > vals[0] {
                    vals = [y, vals[0], vals[1]];
                    idx = [i as u8, idx[0], idx[1]];
                } else if y > vals[1] {
                    vals[2] = vals[1];
                    vals[1] = y;
                    idx[2] = idx[1];
                    idx[1] = i as u8;
                } else if y > vals[2] {
                    vals[2] = y;
                    idx[2] = i as u8;
                }
            }
            self.top3[s] = (vals, idx);
        }
    }
}

/// Reusable per-query scratch tables (three 8-bit chunk tables, ~50 KiB).
/// Reuse across queries in hot loops (`*_with` methods).
pub struct Workspace {
    chunks: [Chunk; NCHUNK],
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            chunks: [Chunk::new(), Chunk::new(), Chunk::new()],
        }
    }

    fn fill(&mut self, x: &[f64; DIM]) {
        for (k, ch) in self.chunks.iter_mut().enumerate() {
            ch.fill(&x[k * CHUNK..(k + 1) * CHUNK]);
        }
    }

    #[inline]
    fn part(&self, c: u32, k: usize) -> usize {
        (c >> (k * CHUNK) & CMASK) as usize
    }

    #[inline]
    fn t(&self, c: u32) -> f64 {
        self.chunks[0].t[self.part(c, 0)]
            + self.chunks[1].t[self.part(c, 1)]
            + self.chunks[2].t[self.part(c, 2)]
    }

    #[inline]
    fn sum_abs(&self, c: u32) -> f64 {
        self.chunks[0].a[self.part(c, 0)]
            + self.chunks[1].a[self.part(c, 1)]
            + self.chunks[2].a[self.part(c, 2)]
    }

    #[inline]
    fn neg_count(&self, c: u32) -> u32 {
        (self.chunks[0].neg[self.part(c, 0)]
            + self.chunks[1].neg[self.part(c, 1)]
            + self.chunks[2].neg[self.part(c, 2)]) as u32
    }

    /// `(min |xᵢ|, global index)` over the support of `c`.
    #[inline]
    fn min_abs(&self, c: u32) -> (f64, usize) {
        let mut best = (f64::INFINITY, NONE as usize);
        for k in 0..NCHUNK {
            let e = self.chunks[k].min_abs[self.part(c, k)];
            if e.0 < best.0 {
                best = (e.0, e.1 as usize + k * CHUNK);
            }
        }
        best
    }

    /// `(max xᵢ, global index)` over the support of `c`.
    #[inline]
    fn max_in(&self, c: u32) -> (f64, usize) {
        let mut best = (f64::NEG_INFINITY, NONE as usize);
        for k in 0..NCHUNK {
            let e = self.chunks[k].max_in[self.part(c, k)];
            if e.0 > best.0 {
                best = (e.0, e.1 as usize + k * CHUNK);
            }
        }
        best
    }

    /// `(min xᵢ, global index)` over the support of `c`.
    #[inline]
    fn min_in(&self, c: u32) -> (f64, usize) {
        let mut best = (f64::INFINITY, NONE as usize);
        for k in 0..NCHUNK {
            let e = self.chunks[k].min_in[self.part(c, k)];
            if e.0 < best.0 {
                best = (e.0, e.1 as usize + k * CHUNK);
            }
        }
        best
    }

    /// Exact top-3 of `yᵢ = (i ∈ c ? xᵢ : −xᵢ)` over all 24 positions:
    /// merge the three per-chunk top-3 lists (9 candidates).
    #[inline]
    fn top3_y(&self, c: u32) -> (f64, [usize; 3]) {
        let mut vals = [f64::NEG_INFINITY; 3];
        let mut idx = [NONE as usize; 3];
        for k in 0..NCHUNK {
            let (v, i) = self.chunks[k].top3[self.part(c, k)];
            for j in 0..3 {
                let y = v[j];
                let gi = i[j] as usize + k * CHUNK;
                if y > vals[0] {
                    vals = [y, vals[0], vals[1]];
                    idx = [gi, idx[0], idx[1]];
                } else if y > vals[1] {
                    vals[2] = vals[1];
                    vals[1] = y;
                    idx[2] = idx[1];
                    idx[1] = gi;
                } else if y > vals[2] {
                    vals[2] = y;
                    idx[2] = gi;
                }
            }
        }
        (vals[0] + vals[1] + vals[2], idx)
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Complement of a 24-bit support.
#[inline]
fn comp(c: u32) -> u32 {
    !c & 0xFF_FFFF
}

/// ±1 base vector of the odd coset: −1 on the codeword support, +1 off it.
fn ones_base(c: u32) -> Point {
    let mut x = [1i32; DIM];
    for (i, xi) in x.iter_mut().enumerate() {
        if c >> i & 1 == 1 {
            *xi = -1;
        }
    }
    x
}

/// Materialize a ±value-on-support vector with sign-matched entries and an
/// optional repair flip at `flip` (global index).
fn valued_point(x: &[f64; DIM], support: u32, value: i32, flip: Option<usize>) -> Point {
    let mut p: Point = [0; DIM];
    for (i, pi) in p.iter_mut().enumerate() {
        if support >> i & 1 == 1 {
            let mut neg = x[i] < 0.0;
            if flip == Some(i) {
                neg = !neg;
            }
            *pi = if neg { -value } else { value };
        }
    }
    p
}

/// Argmax bookkeeping for one shell: enough to materialize the point once.
enum Best {
    None,
    /// ±value on `support`, with optional repair flip index.
    TwoSupport {
        support: u32,
        value: i32,
        flip: Option<usize>,
    },
    /// ±4 at `p4` added on top of a two-support configuration.
    FourTwo {
        support: u32,
        flip: Option<usize>,
        p4: usize,
    },
    /// ±4² at two positions.
    FourFour { i1: usize, i2: usize },
    /// (±a, ±1²³): codeword + position + whether the a-coordinate sits on
    /// the support.
    AOnes { c: u32, p: usize, on: bool, a: i32 },
    /// (±3³, ±1²¹): codeword + the three positions.
    ThreeThrees { c: u32, t: [usize; 3] },
}

fn materialize(x: &[f64; DIM], b: &Best) -> Point {
    match *b {
        Best::None => unreachable!("search always finds a candidate"),
        Best::TwoSupport {
            support,
            value,
            flip,
        } => valued_point(x, support, value, flip),
        Best::FourTwo { support, flip, p4 } => {
            let mut p = valued_point(x, support, 2, flip);
            p[p4] = if x[p4] < 0.0 { -4 } else { 4 };
            p
        }
        Best::FourFour { i1, i2 } => {
            let mut p: Point = [0; DIM];
            p[i1] = if x[i1] < 0.0 { -4 } else { 4 };
            p[i2] = if x[i2] < 0.0 { -4 } else { 4 };
            p
        }
        Best::AOnes { c, p, on, a } => {
            let mut pt = ones_base(c);
            // On the support the ±1 baseline is −1 and the large value is
            // +a for a = 3, −a for a = 5; off the support the baseline is
            // +1 and the large value is −a for a = 3, +a for a = 5.
            pt[p] = match (on, a) {
                (true, 3) => 3,
                (false, 3) => -3,
                (true, _) => -a,
                (false, _) => a,
            };
            pt
        }
        Best::ThreeThrees { c, t } => {
            let mut pt = ones_base(c);
            for &i in &t {
                pt[i] = if c >> i & 1 == 1 { 3 } else { -3 };
            }
            pt
        }
    }
}

/// Exact nearest-neighbour searcher over Leech shells.
pub struct Searcher {
    leech: Leech,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            leech: Leech::new(),
        }
    }

    /// The underlying lattice (e.g. for membership checks in tests).
    pub fn leech(&self) -> &Leech {
        &self.leech
    }

    /// Exact `argmax_{v ∈ Shell(2)} ⟨x, v⟩` (allocates a fresh workspace;
    /// prefer [`Searcher::nearest_in_shell2_with`] in hot loops).
    pub fn nearest_in_shell2(&self, x: &[f64; DIM]) -> Found {
        self.nearest_in_shell2_with(&mut Workspace::new(), x)
    }

    /// Exact `argmax_{v ∈ Shell(3)} ⟨x, v⟩` (allocates a fresh workspace).
    pub fn nearest_in_shell3(&self, x: &[f64; DIM]) -> Found {
        self.nearest_in_shell3_with(&mut Workspace::new(), x)
    }

    /// Exact NN over `Shell(2) ∪ Shell(3)` (allocates a fresh workspace).
    pub fn nearest_m3(&self, x: &[f64; DIM], metric: Metric) -> Found {
        self.nearest_m3_with(&mut Workspace::new(), x, metric)
    }

    /// Exact `argmax_{v ∈ Shell(2)} ⟨x, v⟩`, reusing `ws`.
    pub fn nearest_in_shell2_with(&self, ws: &mut Workspace, x: &[f64; DIM]) -> Found {
        ws.fill(x);
        self.shell2_inner(ws, x)
    }

    /// Exact `argmax_{v ∈ Shell(3)} ⟨x, v⟩`, reusing `ws`.
    pub fn nearest_in_shell3_with(&self, ws: &mut Workspace, x: &[f64; DIM]) -> Found {
        ws.fill(x);
        self.shell3_inner(ws, x)
    }

    /// Exact NN over `Shell(2) ∪ Shell(3)` under `metric`, reusing `ws`.
    pub fn nearest_m3_with(&self, ws: &mut Workspace, x: &[f64; DIM], metric: Metric) -> Found {
        ws.fill(x);
        let f2 = self.shell2_inner(ws, x);
        let f3 = self.shell3_inner(ws, x);
        let better = match metric {
            // Minimizing ‖x−v‖² = ‖x‖² + 16m − 2⟨x,v⟩
            // ⇔ maximizing 2⟨x,v⟩ − 16m.
            Metric::Euclidean => 2.0 * f3.dot - 48.0 > 2.0 * f2.dot - 32.0,
            // Maximizing ⟨x,v⟩ / √(16m).
            Metric::Angular => f3.dot / 48f64.sqrt() > f2.dot / 32f64.sqrt(),
        };
        if better {
            f3
        } else {
            f2
        }
    }

    fn shell2_inner(&self, ws: &Workspace, x: &[f64; DIM]) -> Found {
        let g = self.leech.golay();
        let mut best = f64::NEG_INFINITY;
        let mut best_at = Best::None;

        // Class (±2⁸): per octad, sign-matched sum with even-minus repair.
        for &oct in g.of_weight(8) {
            let mut s = ws.sum_abs(oct);
            let mut flip = None;
            if ws.neg_count(oct) % 2 == 1 {
                let (ma, mi) = ws.min_abs(oct);
                s -= 2.0 * ma;
                flip = Some(mi);
            }
            if 2.0 * s > best {
                best = 2.0 * s;
                best_at = Best::TwoSupport {
                    support: oct,
                    value: 2,
                    flip,
                };
            }
        }

        // Class (±4²): the two largest |xᵢ|, all signs free.
        {
            let (mut i1, mut i2) = (0usize, 1usize);
            if x[i2].abs() > x[i1].abs() {
                (i1, i2) = (i2, i1);
            }
            for i in 2..DIM {
                let a = x[i].abs();
                if a > x[i1].abs() {
                    i2 = i1;
                    i1 = i;
                } else if a > x[i2].abs() {
                    i2 = i;
                }
            }
            let cand = 4.0 * (x[i1].abs() + x[i2].abs());
            if cand > best {
                best = cand;
                best_at = Best::FourFour { i1, i2 };
            }
        }

        // Class (±3, ±1²³): per codeword c,
        // score = S − 2·T_c + 4·max( max_{p∈c} x_p , max_{p∉c} −x_p ),
        // pruned by the bound base + 4·max|x|.
        let s_all: f64 = x.iter().sum();
        let a_max = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for &c in g.codewords() {
            let base = s_all - 2.0 * ws.t(c);
            if base + 4.0 * a_max <= best {
                continue;
            }
            let (vin, pin) = ws.max_in(c); // p ∈ c → +3
            let (vout_min, pout) = ws.min_in(comp(c)); // p ∉ c → −3
            let (gain, p, on) = if vin >= -vout_min {
                (vin, pin, true)
            } else {
                (-vout_min, pout, false)
            };
            let cand = base + 4.0 * gain;
            if cand > best {
                best = cand;
                best_at = Best::AOnes { c, p, on, a: 3 };
            }
        }

        Found {
            point: materialize(x, &best_at),
            shell: 2,
            dot: best,
        }
    }

    fn shell3_inner(&self, ws: &Workspace, x: &[f64; DIM]) -> Found {
        let g = self.leech.golay();
        let mut best = f64::NEG_INFINITY;
        let mut best_at = Best::None;

        // Class (±2¹²): per dodecad, even-minus repair.
        for &dodecad in g.of_weight(12) {
            let mut s = ws.sum_abs(dodecad);
            let mut flip = None;
            if ws.neg_count(dodecad) % 2 == 1 {
                let (ma, mi) = ws.min_abs(dodecad);
                s -= 2.0 * ma;
                flip = Some(mi);
            }
            if 2.0 * s > best {
                best = 2.0 * s;
                best_at = Best::TwoSupport {
                    support: dodecad,
                    value: 2,
                    flip,
                };
            }
        }

        // Class (±4, ±2⁸): per octad, odd-minus 2s + the best |x_p| outside
        // the octad (the ±4 sign is free: it shifts Σ by 8). The outside
        // argmax probes a precomputed descending-|x| order (≤ 9 probes).
        let mut order: [usize; DIM] = core::array::from_fn(|i| i);
        order.sort_unstable_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
        for &oct in g.of_weight(8) {
            let mut s2 = ws.sum_abs(oct);
            let mut flip = None;
            if ws.neg_count(oct).is_multiple_of(2) {
                let (ma, mi) = ws.min_abs(oct);
                s2 -= 2.0 * ma;
                flip = Some(mi);
            }
            let &p4 = order
                .iter()
                .find(|&&p| oct >> p & 1 == 0)
                .expect("octad has weight 8 < 24");
            let cand = 2.0 * s2 + 4.0 * x[p4].abs();
            if cand > best {
                best = cand;
                best_at = Best::FourTwo {
                    support: oct,
                    flip,
                    p4,
                };
            }
        }

        // Codeword classes (±5, ±1²³) and (±3³, ±1²¹), merged in one pruned
        // loop. (±5): +5 OFF the support, −5 ON it (roles swapped vs ±3).
        // Bounds: the (±3³) gain is at most 4·(A₁+A₂+A₃) (three largest
        // |x|), the (±5) gain at most 4·A₁ ≤ that, so one shared bound
        // prunes both classes.
        let s_all: f64 = x.iter().sum();
        let mut abs_sorted: [f64; DIM] = core::array::from_fn(|i| x[i].abs());
        abs_sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let a1 = abs_sorted[0];
        let a3sum = abs_sorted[0] + abs_sorted[1] + abs_sorted[2];
        for &c in g.codewords() {
            let base = s_all - 2.0 * ws.t(c);
            if base + 4.0 * a3sum <= best {
                continue; // neither class can beat the running best
            }

            // (±5, ±1²³)
            if base + 4.0 * a1 > best {
                let (von_min, pon) = ws.min_in(c); // p ∈ c → −5
                let (voff, poff) = ws.max_in(comp(c)); // p ∉ c → +5
                let (gain, p, on) = if -von_min >= voff {
                    (-von_min, pon, true)
                } else {
                    (voff, poff, false)
                };
                let cand5 = base + 4.0 * gain;
                if cand5 > best {
                    best = cand5;
                    best_at = Best::AOnes { c, p, on, a: 5 };
                }
            }

            // (±3³, ±1²¹): exact top-3 of yᵢ = (i ∈ c ? xᵢ : −xᵢ) via the
            // per-chunk top-3 tables (9-candidate merge, no 24-pass).
            let (y3, top_i) = ws.top3_y(c);
            let cand3 = base + 4.0 * y3;
            if cand3 > best {
                best = cand3;
                best_at = Best::ThreeThrees { c, t: top_i };
            }
        }

        Found {
            point: materialize(x, &best_at),
            shell: 3,
            dot: best,
        }
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(x: &[f64; DIM], v: &Point) -> f64 {
        x.iter().zip(v.iter()).map(|(a, &b)| a * b as f64).sum()
    }

    /// The reported dot must equal the recomputed dot of the returned point.
    #[test]
    fn reported_dot_is_consistent() {
        let s = Searcher::new();
        let mut ws = Workspace::new();
        let x: [f64; DIM] = core::array::from_fn(|i| (i as f64 * 0.37).sin());
        for f in [
            s.nearest_in_shell2_with(&mut ws, &x),
            s.nearest_in_shell3_with(&mut ws, &x),
            s.nearest_m3_with(&mut ws, &x, Metric::Euclidean),
            s.nearest_m3_with(&mut ws, &x, Metric::Angular),
        ] {
            assert!((f.dot - dot(&x, &f.point)).abs() < 1e-9);
            assert!(s.leech().contains(&f.point));
            assert_eq!(Leech::shell_index(&f.point), Some(f.shell as u64));
        }
    }
}
