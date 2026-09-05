//! Where the bench F1 encoder spends its 0.24 s per block — counted, not guessed.
//!
//! `cargo run --release -p llvq-bench --example f1encprof -- [n_blocks]`
//!
//! Single thread. Re-implements the rank-table encoder of `f1rankbench.rs`
//! (copied, not modified) with counters on every nearest call, every candidate
//! generated and every membership test, and a stopwatch on each phase of
//! `encode_at_scale`: middle solves, end solves, prefix solves + trellis join.
//! It also records, per section solve, whether the shrink-1.0 base candidate
//! was already a member (in which case it is the exact coset optimum and the
//! remaining 7 shrinks × 17 candidates cannot beat it), and at which shrink
//! the winner was found. Those two numbers are what a production encoder can
//! skip, and the design in the report is priced on them.
//!
//! Nothing here touches `f1.rs` or `f1rankbench.rs`.

use llvq_bench::f1::{SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use llvq_bench::gauss_block;
use llvq_core::SplitMix64;
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Counters (single-threaded, so plain thread-locals are enough)
// ---------------------------------------------------------------------------

thread_local! {
    static N_NEAREST: Cell<u64> = const { Cell::new(0) };
    static N_NEAREST_C: Cell<u64> = const { Cell::new(0) };
    static N_CAND: Cell<u64> = const { Cell::new(0) };
    static N_CONTAINS: Cell<u64> = const { Cell::new(0) };
    static N_MEMBER: Cell<u64> = const { Cell::new(0) };
    static BASE_IN: Cell<u64> = const { Cell::new(0) };
    static SOLVES: Cell<u64> = const { Cell::new(0) };
    static WIN_SHRINK: [Cell<u64>; 9] = const { [const { Cell::new(0) }; 9] };
    static T_MID: Cell<f64> = const { Cell::new(0.0) };
    static T_END: Cell<f64> = const { Cell::new(0.0) };
    static T_S1: Cell<f64> = const { Cell::new(0.0) };
    static T_JOIN: Cell<f64> = const { Cell::new(0.0) };
}

fn bump(c: &'static std::thread::LocalKey<Cell<u64>>, by: u64) {
    c.with(|v| v.set(v.get() + by));
}
fn addt(c: &'static std::thread::LocalKey<Cell<f64>>, by: f64) {
    c.with(|v| v.set(v.get() + by));
}

// ---------------------------------------------------------------------------
// The rank-space table — same construction as f1rankbench.rs
// ---------------------------------------------------------------------------

fn val(o: u32, rho: u32) -> i32 {
    match o {
        0 => {
            if rho == 0 {
                0
            } else {
                let m = 4 * rho.div_ceil(2) as i32;
                if rho.is_multiple_of(2) { -m } else { m }
            }
        }
        2 => {
            let m = (2 + 4 * (rho / 2)) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        1 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        3 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { -m } else { m }
        }
        _ => unreachable!(),
    }
}

fn rank_of(o: u32, y: i32) -> Option<u32> {
    if y.rem_euclid(4) != o as i32 {
        return None;
    }
    (0..8u32).find(|&r| val(o, r) == y)
}

fn rank_class(rho: &[u32; SECTION]) -> u32 {
    rho.iter().filter(|&&r| r == 1 || r == 2).count() as u32 & 1
}

fn pack(rho: &[u32; SECTION]) -> u32 {
    rho.iter().enumerate().fold(0u32, |a, (j, &r)| a | (r << (4 * j)))
}

pub struct RankTable {
    class: [HashSet<u32>; 2],
    mixed: HashSet<u32>,
}

fn rank_table() -> RankTable {
    fn walk(j: usize, acc: i64, cap: i64, rho: &mut [u32; SECTION], out: &mut Vec<([u32; SECTION], i64)>) {
        if j == SECTION {
            out.push((*rho, acc));
            return;
        }
        for r in 0..8u32 {
            let c = (2 * r as i64 + 1).pow(2);
            if acc + c > cap {
                break;
            }
            rho[j] = r;
            walk(j + 1, acc + c, cap, rho, out);
        }
    }
    let mut cap = 64i64;
    loop {
        let mut all = Vec::new();
        walk(0, 0, cap, &mut [0; SECTION], &mut all);
        all.sort_by_key(|&(r, c)| (c, r));
        let c0: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 0).take(2048).map(|&(r, _)| r).collect();
        let c1: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 1).take(2048).map(|&(r, _)| r).collect();
        if c0.len() == 2048 && c1.len() == 2048 && all.len() >= 4096 {
            let cost = |r: &[u32; SECTION]| r.iter().map(|&x| (2 * x as i64 + 1).pow(2)).sum::<i64>();
            let kept = all.iter().take(4096).map(|&(_, c)| c).max().unwrap();
            if kept < cap && cost(&c0[2047]) < cap && cost(&c1[2047]) < cap {
                let mixed: Vec<_> = all.iter().take(2048).map(|&(r, _)| r).collect();
                return RankTable {
                    class: [c0.iter().map(pack).collect(), c1.iter().map(pack).collect()],
                    mixed: mixed.iter().map(pack).collect(),
                };
            }
        }
        cap *= 2;
    }
}

#[derive(Clone, Copy)]
enum Mode {
    End(u32),
    Mid,
}

pub struct RankRegion {
    set: SectionSet,
    table: Arc<RankTable>,
    mode: Mode,
    fallback: [i32; SECTION],
}

impl RankRegion {
    fn new(set: SectionSet, table: Arc<RankTable>, mode: Mode) -> Self {
        let mut r = Self { set, table, mode, fallback: [0; SECTION] };
        let mut t = 8usize;
        loop {
            let pts = r.set.enumerate_below(t);
            if let Some(y) = pts
                .into_iter()
                .filter(|y| r.contains(y))
                .min_by_key(|y| (y.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>(), *y))
            {
                r.fallback = y;
                return r;
            }
            t *= 2;
            assert!(t < 4096, "no member found");
        }
    }

    fn rho_of(&self, y: &[i32; SECTION]) -> Option<[u32; SECTION]> {
        let p = self.set.p as i32;
        let mut rho = [0u32; SECTION];
        for j in 0..SECTION {
            let d = y[j] - p;
            if d.rem_euclid(2) != 0 {
                return None;
            }
            let cj = (d.div_euclid(2)).rem_euclid(2) as u32;
            let o = self.set.p + 2 * cj;
            rho[j] = rank_of(o, y[j])?;
        }
        Some(rho)
    }

    pub fn contains(&self, y: &[i32; SECTION]) -> bool {
        bump(&N_CONTAINS, 1);
        if !self.set.contains(y) {
            return false;
        }
        let Some(rho) = self.rho_of(y) else { return false };
        let key = pack(&rho);
        let m = match self.mode {
            Mode::End(r) => self.table.class[r as usize].contains(&key),
            Mode::Mid => self.table.mixed.contains(&key),
        };
        if m {
            bump(&N_MEMBER, 1);
        }
        m
    }

    fn k_parity_of(&self, pattern: u8, y: &[i32; SECTION]) -> u32 {
        (0..SECTION)
            .map(|j| {
                let base = self.set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                ((y[j] - base).div_euclid(4)).rem_euclid(2) as u32
            })
            .fold(0, |a, b| a ^ b)
    }

    fn candidates_with(&self, pattern: u8, target: &[f64; SECTION], want_parity: Option<u32>) -> Vec<[i32; SECTION]> {
        let base: Vec<i32> = (0..SECTION).map(|j| self.set.p as i32 + 2 * ((pattern >> j & 1) as i32)).collect();
        let z: Vec<f64> = (0..SECTION).map(|j| (target[j] - base[j] as f64) / 4.0).collect();
        let mut k: Vec<i32> = z.iter().map(|v| v.round() as i32).collect();
        let mut regret: Vec<(f64, usize)> = (0..SECTION).map(|j| ((z[j] - k[j] as f64).abs(), j)).collect();
        regret.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(r) = want_parity {
            let sum: i32 = k.iter().sum();
            if sum.rem_euclid(2) != r as i32 {
                let j = regret[0].1;
                k[j] += if z[j] > k[j] as f64 { 1 } else { -1 };
            }
        }
        let build = |k: &[i32]| {
            let mut y = [0i32; SECTION];
            for j in 0..SECTION {
                y[j] = base[j] + 4 * k[j];
            }
            y
        };
        let mut out = vec![build(&k)];
        for &(_, j) in &regret {
            for step in [-1i32, 1] {
                let mut k2 = k.clone();
                k2[j] += step;
                if want_parity.is_some() {
                    let j2 = regret.iter().map(|&(_, x)| x).find(|&x| x != j).expect("8 > 1");
                    k2[j2] += if z[j2] > k[j2] as f64 { 1 } else { -1 };
                }
                out.push(build(&k2));
            }
        }
        bump(&N_CAND, out.len() as u64);
        out
    }

    fn nearest(&self, target: &[f64; SECTION]) -> ([i32; SECTION], f64) {
        bump(&N_NEAREST, 1);
        bump(&SOLVES, 1);
        let dist = |y: &[i32; SECTION]| -> f64 { y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum() };
        let mut best = (self.fallback, dist(&self.fallback));
        let mut win = 8usize; // 8 = fallback won
        for (si, shrink) in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1].into_iter().enumerate() {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for (pi, &pattern) in self.set.patterns.iter().enumerate() {
                for (ci, cand) in self.candidates_with(pattern, &scaled, self.set.k_parity).into_iter().enumerate() {
                    let member = self.contains(&cand);
                    if si == 0 && ci == 0 && pi == 0 && member {
                        bump(&BASE_IN, 1);
                    }
                    if !member {
                        continue;
                    }
                    let d = dist(&cand);
                    if d < best.1 {
                        best = (cand, d);
                        win = si;
                    }
                }
            }
        }
        WIN_SHRINK.with(|w| w[win].set(w[win].get() + 1));
        best
    }

    fn nearest_constrained(&self, target: &[f64; SECTION], pattern: u8, kparity: u32) -> Option<([i32; SECTION], f64)> {
        bump(&N_NEAREST_C, 1);
        bump(&SOLVES, 1);
        let dist = |y: &[i32; SECTION]| -> f64 { y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum() };
        let mut best: Option<([i32; SECTION], f64)> = None;
        let mut win = 8usize;
        for (si, shrink) in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0].into_iter().enumerate() {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for (ci, cand) in self.candidates_with(pattern, &scaled, Some(kparity)).into_iter().enumerate() {
                let member = self.contains(&cand) && self.k_parity_of(pattern, &cand) == kparity;
                if si == 0 && ci == 0 && member {
                    bump(&BASE_IN, 1);
                }
                if !member {
                    continue;
                }
                let d = dist(&cand);
                if best.as_ref().is_none_or(|&(_, bd)| d < bd) {
                    best = Some((cand, d));
                    win = si;
                }
            }
        }
        WIN_SHRINK.with(|w| w[win].set(w[win].get() + 1));
        best
    }
}

// ---------------------------------------------------------------------------
// The codebook, as in f1rankbench.rs, with a stopwatch per phase
// ---------------------------------------------------------------------------

type MidPick = [Option<([i32; SECTION], f64)>; 2];

pub struct Codebook {
    pub trellis: Trellis,
    sec1: Vec<RankRegion>,
    sec2: Vec<RankRegion>,
    sec3: Vec<RankRegion>,
    mset_of: Vec<usize>,
    msets: Vec<Vec<u8>>,
}

fn msets_of(trellis: &Trellis) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut msets: Vec<Vec<u8>> = Vec::new();
    let mut mset_of = vec![0usize; GOLAY_STATES];
    for (s8, slot) in mset_of.iter_mut().enumerate() {
        let mut m: Vec<u8> = trellis.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        *slot = match msets.iter().position(|s| *s == m) {
            Some(i) => i,
            None => {
                msets.push(m);
                msets.len() - 1
            }
        };
    }
    assert_eq!(msets.len(), 8);
    (msets, mset_of)
}

impl Codebook {
    fn build(table: Arc<RankTable>) -> Self {
        let trellis = Trellis::new();
        let (msets, mset_of) = msets_of(&trellis);
        let mut sec1 = Vec::new();
        let mut sec3 = Vec::new();
        for p in 0..2u32 {
            for r in 0..2u32 {
                for s in 0..GOLAY_STATES {
                    sec1.push(RankRegion::new(trellis.section1(s, p, r), table.clone(), Mode::End(r)));
                    sec3.push(RankRegion::new(
                        SectionSet { patterns: trellis.suffixes[s].to_vec(), p, k_parity: Some(r) },
                        table.clone(),
                        Mode::End(r),
                    ));
                }
            }
        }
        let mut sec2 = Vec::new();
        for p in 0..2u32 {
            for m in &msets {
                sec2.push(RankRegion::new(SectionSet { patterns: m.clone(), p, k_parity: None }, table.clone(), Mode::Mid));
            }
        }
        Self { trellis, sec1, sec2, sec3, mset_of, msets }
    }

    fn s1(&self, p: u32, r: u32, s8: usize) -> &RankRegion {
        &self.sec1[((p * 2 + r) as usize) * GOLAY_STATES + s8]
    }
    fn s3(&self, p: u32, r_out: u32, s16: usize) -> &RankRegion {
        &self.sec3[((p * 2 + r_out) as usize) * GOLAY_STATES + s16]
    }
    fn s2(&self, p: u32, mset: usize) -> &RankRegion {
        &self.sec2[p as usize * 8 + mset]
    }

    pub fn encode_at_scale(&self, x: &[f64; 24], s: f64) -> [i32; 24] {
        let part = |lo: usize| -> [f64; SECTION] {
            let mut t = [0.0f64; SECTION];
            for (j, v) in t.iter_mut().enumerate() {
                *v = x[lo + j] / s;
            }
            t
        };
        let (t1, t2, t3) = (part(0), part(8), part(16));
        let mut best: Option<(f64, [i32; 24])> = None;
        for p in 0..2u32 {
            let t0 = Instant::now();
            let mut mid: Vec<MidPick> = Vec::new();
            for m in 0..8usize {
                for &b in &self.msets[m] {
                    mid.push([self.s2(p, m).nearest_constrained(&t2, b, 0), self.s2(p, m).nearest_constrained(&t2, b, 1)]);
                }
            }
            addt(&T_MID, t0.elapsed().as_secs_f64());
            let mid_at = |m: usize, b: u8| -> &MidPick {
                let k = self.msets[m].iter().position(|&x| x == b).expect("byte in its set");
                &mid[m * BRANCHES + k]
            };
            let t0 = Instant::now();
            let mut end: Vec<([i32; SECTION], f64)> = Vec::with_capacity(2 * GOLAY_STATES);
            for r_out in 0..2u32 {
                for s16 in 0..GOLAY_STATES {
                    end.push(self.s3(p, r_out, s16).nearest(&t3));
                }
            }
            addt(&T_END, t0.elapsed().as_secs_f64());
            for r in 0..2u32 {
                for s8 in 0..GOLAY_STATES {
                    let t0 = Instant::now();
                    let (y1, d1) = self.s1(p, r, s8).nearest(&t1);
                    addt(&T_S1, t0.elapsed().as_secs_f64());
                    let t0 = Instant::now();
                    let m = self.mset_of[s8];
                    for &(b, s16) in &self.trellis.branches[s8] {
                        for delta in 0..2u32 {
                            let Some((y2, d2)) = mid_at(m, b)[delta as usize] else { continue };
                            let r_out = (p ^ r ^ delta) & 1;
                            let (y3, d3) = end[r_out as usize * GOLAY_STATES + s16 as usize];
                            let cost = d1 + d2 + d3;
                            if best.as_ref().is_none_or(|&(bc, _)| cost < bc) {
                                let mut y = [0i32; 24];
                                y[..8].copy_from_slice(&y1);
                                y[8..16].copy_from_slice(&y2);
                                y[16..].copy_from_slice(&y3);
                                best = Some((cost, y));
                            }
                        }
                    }
                    addt(&T_JOIN, t0.elapsed().as_secs_f64());
                }
            }
        }
        best.expect("never empty").1
    }

    pub fn best_t(&self, x: &[f64; 24], scales: &[f64]) -> (f64, [i32; 24], usize) {
        let mut best = (f64::NEG_INFINITY, [0i32; 24], 0usize);
        for (i, &s) in scales.iter().enumerate() {
            let y = self.encode_at_scale(x, s);
            let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
            let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
            if nn > 0.0 {
                let t = dot / nn.sqrt();
                if t > best.0 {
                    best = (t, y, i);
                }
            }
        }
        best
    }
}

fn reset() {
    for c in [&N_NEAREST, &N_NEAREST_C, &N_CAND, &N_CONTAINS, &N_MEMBER, &BASE_IN, &SOLVES] {
        c.with(|v| v.set(0));
    }
    WIN_SHRINK.with(|w| w.iter().for_each(|c| c.set(0)));
    for c in [&T_MID, &T_END, &T_S1, &T_JOIN] {
        c.with(|v| v.set(0.0));
    }
}

fn report(label: &str, n_blocks: usize, n_scales: usize, wall: f64) {
    let per_bs = |c: &'static std::thread::LocalKey<Cell<u64>>| c.with(|v| v.get()) as f64 / (n_blocks * n_scales) as f64;
    let solves = SOLVES.with(|v| v.get()) as f64;
    println!("\n[{label}] {n_blocks} blocs × {n_scales} échelle(s) : {:.1} ms/bloc, {:.2} ms par (bloc, échelle)", 1e3 * wall / n_blocks as f64, 1e3 * wall / (n_blocks * n_scales) as f64);
    println!("  par (bloc, échelle) : {:.0} nearest + {:.0} nearest_constrained, {:.0} candidats, {:.0} tests d'appartenance, {:.0} membres",
        per_bs(&N_NEAREST), per_bs(&N_NEAREST_C), per_bs(&N_CAND), per_bs(&N_CONTAINS), per_bs(&N_MEMBER));
    println!("  base (rétrécissement 1,0, premier candidat) déjà membre : {:.1} % des {:.0} résolutions par (bloc, échelle)",
        100.0 * BASE_IN.with(|v| v.get()) as f64 / solves, per_bs(&SOLVES));
    let wins: Vec<u64> = WIN_SHRINK.with(|w| w.iter().map(|c| c.get()).collect());
    let row: Vec<String> = wins.iter().map(|&c| format!("{:.2}", 100.0 * c as f64 / solves)).collect();
    println!("  rétrécissement gagnant (1,0 / 0,85 / 0,7 / 0,55 / 0,4 / 0,25 / 0,1 / 0,0 / repli) en % des résolutions : [{}]", row.join(" "));
    let (tm, te, ts, tj) = (T_MID.with(|v| v.get()), T_END.with(|v| v.get()), T_S1.with(|v| v.get()), T_JOIN.with(|v| v.get()));
    let tot = tm + te + ts + tj;
    println!("  temps : milieu {:.1} %  fins {:.1} %  préfixes {:.1} %  jonction {:.1} %  (chronométré {:.1} % du mur)",
        100.0 * tm / tot, 100.0 * te / tot, 100.0 * ts / tot, 100.0 * tj / tot, 100.0 * tot / wall);
    println!("  jonction : {:.1} µs par (bloc, échelle) pour 8 192 sommes", 1e6 * tj / (n_blocks * n_scales) as f64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let table = Arc::new(rank_table());
    let t0 = Instant::now();
    let cb = Codebook::build(table);
    println!("codebook de rangs préparé en {:.2} s ; 8 msets × 16 = 128 octets de milieu, 128 préfixes, 128 suffixes", t0.elapsed().as_secs_f64());

    // The bench's blocks: seed and order of f1rankbench.rs (4,000 training blocks drawn first).
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    for _ in 0..4_000 {
        gauss_block(&mut rng);
    }
    let xs: Vec<[f64; 24]> = (0..n).map(|_| gauss_block(&mut rng)).collect();
    let scales: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();

    // Warm-up.
    cb.best_t(&xs[0], &scales);

    // (1) the full 18-scale sweep, as the bench runs it.
    reset();
    let t0 = Instant::now();
    let mut hist = vec![0usize; scales.len()];
    for x in &xs {
        let (_, _, i) = cb.best_t(x, &scales);
        hist[i] += 1;
    }
    let wall = t0.elapsed().as_secs_f64();
    report("balayage 18 échelles", n, scales.len(), wall);
    let row: Vec<String> = hist.iter().map(|c| c.to_string()).collect();
    println!("  échelle gagnante (indice 0..17) : [{}]", row.join(" "));

    // (2) one scale, the one the sweep most often picks, so the per-scale
    // numbers are read where a production encoder would sit.
    let best_i = hist.iter().enumerate().max_by_key(|&(_, c)| *c).map(|(i, _)| i).unwrap_or(9);
    let one = [scales[best_i]];
    reset();
    let t0 = Instant::now();
    for x in &xs {
        cb.best_t(x, &one);
    }
    report(&format!("une échelle s = {:.3}", one[0]), n, 1, t0.elapsed().as_secs_f64());

    // (3) the smallest scale of the grid, where targets sit far outside the
    // region and the shrinks do the work.
    let small = [scales[0]];
    reset();
    let t0 = Instant::now();
    for x in &xs {
        cb.best_t(x, &small);
    }
    report(&format!("échelle s = {:.3} (la plus petite de la grille)", small[0]), n, 1, t0.elapsed().as_secs_f64());
}
