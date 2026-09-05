//! F1 — routes that shrink what the decoder must keep resident. Exact counts, $0.
//!
//! `cargo run --release -p llvq-bench --example f1shrink`
//!
//! Audit input for the F1 table-floor verdict of 2026-09-05: the measured curve
//! says a table ≤ 8 KiB costs 0.344 ms, ≤ 16 KiB 0.663 ms, and anything from
//! 128 KiB to 16 MiB ≈ 4.5 ms. So only routes that bring the HOT part of the
//! decoder table under ~8–16 KiB matter. This computes, for the split 12/15/12:
//!
//! 1. per sign-orbit: max |coord|, share of regions, hot-set sizes at 6 B / 4 B;
//! 2. orbits under the full monomial group 2^8·8! (exact: descriptor BFS,
//!    checked against an invariant lower bound and against point sets);
//! 3. shaping cost of a PRODUCT region shared by all patterns — the task's
//!    "shared k-region" and a rank-space variant — by exact second moments;
//! 4. the six-section (4-coordinate) trellis: states at every cut, edges,
//!    the isotropic label split, table sizes, and the 4-ball shaping loss.
//!
//! Nothing here touches a served path. Bench-only, like `f1.rs`.

use llvq_bench::f1::{SectionSet, Trellis, GOLAY_STATES, SECTION};
use std::collections::{HashMap, HashSet, VecDeque};

const T_SIG: usize = 64;
const W_END: u32 = 12;
const W_MID: u32 = 15;

fn norm2(y: &[i32; SECTION]) -> i64 {
    y.iter().map(|&v| (v as i64) * (v as i64)).sum()
}

// ---------------------------------------------------------------------------
// The family of regions, as the trellis produces it
// ---------------------------------------------------------------------------

struct Reg {
    label: String,
    set: SectionSet,
}

fn family(t: &Trellis) -> (Vec<Reg>, Vec<Reg>) {
    let mut ends = Vec::new();
    for p in 0..2u32 {
        for r in 0..2u32 {
            for s in 0..GOLAY_STATES {
                ends.push(Reg { label: format!("s1 p{p} r{r} g{s}"), set: t.section1(s, p, r) });
            }
        }
    }
    for p in 0..2u32 {
        for r in 0..2u32 {
            for s in 0..GOLAY_STATES {
                ends.push(Reg {
                    label: format!("s3 p{p} r{r} g{s}"),
                    set: SectionSet { patterns: t.suffixes[s].to_vec(), p, k_parity: Some(r) },
                });
            }
        }
    }
    let mut msets: Vec<Vec<u8>> = Vec::new();
    for s8 in 0..GOLAY_STATES {
        let mut m: Vec<u8> = t.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        if !msets.contains(&m) {
            msets.push(m);
        }
    }
    let mut mids = Vec::new();
    for p in 0..2u32 {
        for (i, m) in msets.iter().enumerate() {
            mids.push(Reg {
                label: format!("s2 p{p} m{i}"),
                set: SectionSet { patterns: m.clone(), p, k_parity: None },
            });
        }
    }
    (ends, mids)
}

// ---------------------------------------------------------------------------
// Orbits under sign flips (f1table's method) and the monomial invariant
// ---------------------------------------------------------------------------

fn signature(set: &SectionSet) -> Vec<[i32; SECTION]> {
    let mut v = set.enumerate_below(T_SIG);
    v.sort_unstable();
    v
}

fn flip_point(y: &[i32; SECTION], eps: u32) -> [i32; SECTION] {
    let mut z = *y;
    for (j, v) in z.iter_mut().enumerate() {
        if eps >> j & 1 == 1 {
            *v = -*v;
        }
    }
    z
}

fn canonical_sign(sig: &[[i32; SECTION]]) -> Vec<[i32; SECTION]> {
    (0u32..256)
        .map(|eps| {
            let mut img: Vec<[i32; SECTION]> = sig.iter().map(|y| flip_point(y, eps)).collect();
            img.sort_unstable();
            img
        })
        .min()
        .expect("256 flips")
}

/// Invariant under every signed permutation: the multiset of sorted |coord|
/// vectors of the signature. Distinct values ⇒ distinct monomial orbits.
fn mono_invariant(sig: &[[i32; SECTION]]) -> Vec<[i32; SECTION]> {
    let mut v: Vec<[i32; SECTION]> = sig
        .iter()
        .map(|y| {
            let mut a = y.map(i32::abs);
            a.sort_unstable_by(|x, y| y.cmp(x));
            a
        })
        .collect();
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------------
// Exact monomial orbits by descriptor BFS
//
// A region is `{p·1 + 2c + 4k : c ∈ patterns, Σk ≡ r_c}` with a parity per
// pattern (2 = free). A sign flip on mask d maps it to another such set:
//   p = 0 : c unchanged,   r_c ← r_c ⊕ wt(c ∧ d)   (k_j ← −k_j − c_j)
//   p = 1 : c ← c ⊕ d,     r_c ← r_c ⊕ wt(d)       (k_j ← −k_j − 1)
// A permutation π moves pattern bit j to bit π(j). Both rules are CHECKED on
// point sets below before any orbit is counted with them.
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Desc {
    n: u32,
    p: u8,
    pats: Vec<(u32, u8)>,
}

fn desc_of(set: &SectionSet) -> Desc {
    let r = set.k_parity.map(|r| r as u8).unwrap_or(2);
    let mut pats: Vec<(u32, u8)> = set.patterns.iter().map(|&c| (c as u32, r)).collect();
    pats.sort_unstable();
    Desc { n: SECTION as u32, p: set.p as u8, pats }
}

fn desc_flip(d: &Desc, mask: u32) -> Desc {
    let mut pats: Vec<(u32, u8)> = d
        .pats
        .iter()
        .map(|&(c, r)| {
            let (c2, shift) = if d.p == 0 {
                (c, (c & mask).count_ones() & 1)
            } else {
                (c ^ mask, mask.count_ones() & 1)
            };
            let r2 = if r == 2 { 2 } else { r ^ shift as u8 };
            (c2, r2)
        })
        .collect();
    pats.sort_unstable();
    Desc { n: d.n, p: d.p, pats }
}

fn desc_perm(d: &Desc, pi: &[usize]) -> Desc {
    let mut pats: Vec<(u32, u8)> = d
        .pats
        .iter()
        .map(|&(c, r)| {
            let c2 = (0..d.n as usize).fold(0u32, |acc, j| acc | ((c >> j) & 1) << pi[j]);
            (c2, r)
        })
        .collect();
    pats.sort_unstable();
    Desc { n: d.n, p: d.p, pats }
}

/// Orbit id of every input descriptor under the group generated by the single
/// single-bit flips and the adjacent transpositions — i.e. the full monomial group.
/// Intermediate descriptors outside the family are followed too, so an orbit
/// that leaves the family and comes back is still one orbit.
fn orbits_mono(descs: &[Desc], flips: bool, perms: bool) -> Vec<usize> {
    let mut seen: HashMap<Desc, usize> = HashMap::new();
    let mut ids = vec![usize::MAX; descs.len()];
    let mut next = 0usize;
    for (i, d) in descs.iter().enumerate() {
        if let Some(&k) = seen.get(d) {
            ids[i] = k;
            continue;
        }
        let k = next;
        next += 1;
        let n = d.n as usize;
        let mut q = VecDeque::new();
        seen.insert(d.clone(), k);
        q.push_back(d.clone());
        while let Some(x) = q.pop_front() {
            let mut imgs = Vec::new();
            if flips {
                for j in 0..n {
                    imgs.push(desc_flip(&x, 1 << j));
                }
            }
            if perms {
                for j in 0..n - 1 {
                    let mut pi: Vec<usize> = (0..n).collect();
                    pi.swap(j, j + 1);
                    imgs.push(desc_perm(&x, &pi));
                }
            }
            for y in imgs {
                if !seen.contains_key(&y) {
                    seen.insert(y.clone(), k);
                    q.push_back(y);
                }
            }
        }
        ids[i] = k;
    }
    ids
}

/// Every point of a generalized descriptor with ‖y‖² ≤ t, sorted.
fn enum_desc(d: &Desc, t: usize) -> Vec<[i32; SECTION]> {
    assert_eq!(d.n as usize, SECTION);
    let mut out = Vec::new();
    for &(c, r) in &d.pats {
        let s = SectionSet {
            patterns: vec![c as u8],
            p: d.p as u32,
            k_parity: if r == 2 { None } else { Some(r as u32) },
        };
        out.extend(s.enumerate_below(t));
    }
    out.sort_unstable();
    out
}

/// The transform rules, checked on point sets: the image of the region under
/// (flip d, then π) must be exactly the region of the transformed descriptor.
fn check_desc_rules(regs: &[&Reg], rng: &mut llvq_core::SplitMix64) {
    const T: usize = 40;
    let mut checked = 0;
    for _ in 0..60 {
        let reg = regs[(rng.next() % regs.len() as u64) as usize];
        let d = desc_of(&reg.set);
        let mask = (rng.next() & 0xff) as u32;
        let mut pi: Vec<usize> = (0..SECTION).collect();
        for j in (1..SECTION).rev() {
            let k = (rng.next() % (j as u64 + 1)) as usize;
            pi.swap(j, k);
        }
        let d2 = desc_perm(&desc_flip(&d, mask), &pi);
        let mut img: Vec<[i32; SECTION]> = enum_desc(&d, T)
            .iter()
            .map(|y| {
                let f = flip_point(y, mask);
                let mut z = [0i32; SECTION];
                for j in 0..SECTION {
                    z[pi[j]] = f[j];
                }
                z
            })
            .collect();
        img.sort_unstable();
        let want = enum_desc(&d2, T);
        assert_eq!(img, want, "descriptor rule fails on {} mask {mask:#04x} pi {pi:?}", reg.label);
        checked += 1;
    }
    println!("  règles de transformation vérifiées sur {checked} tirages (ensembles de points, ‖y‖² ≤ {T})");
}

// ---------------------------------------------------------------------------
// Per-region statistics of the exact region
// ---------------------------------------------------------------------------

struct Stat {
    rho2: usize,
    maxabs: i32,
    s_true: f64,
}

fn stat(set: &SectionSet, w: u32) -> Stat {
    let r = set.region(w);
    let pts = set.region_points(w);
    assert_eq!(pts.len(), 1usize << w);
    let maxabs = pts.iter().map(|y| y.iter().map(|v| v.abs()).max().unwrap()).max().unwrap();
    let s_true = pts.iter().map(norm2).sum::<i64>() as f64 / pts.len() as f64;
    Stat { rho2: r.rho2, maxabs, s_true }
}

// ---------------------------------------------------------------------------
// Route 3a — the task's shared k-region: K = the 2^11 lowest AVERAGE-cost k
// over the two offsets a coordinate can take (o ∈ {p, p+2}, each in half the
// patterns of any family set), with the section's parity rule.
// ---------------------------------------------------------------------------

fn cost_k(p: u32, k: i32) -> i64 {
    let k = k as i64;
    if p == 0 {
        16 * k * k + 8 * k + 2 // ½[(4k)² + (2+4k)²]
    } else {
        16 * k * k + 16 * k + 5 // ½[(1+4k)² + (3+4k)²]
    }
}

fn enum_k(p: u32, cap: i64) -> Vec<([i32; SECTION], i64)> {
    fn walk(p: u32, j: usize, acc: i64, cap: i64, k: &mut [i32; SECTION], out: &mut Vec<([i32; SECTION], i64)>) {
        if j == SECTION {
            out.push((*k, acc));
            return;
        }
        for v in -6..=6i32 {
            let c = cost_k(p, v);
            if acc + c > cap {
                continue;
            }
            k[j] = v;
            walk(p, j + 1, acc + c, cap, k, out);
        }
    }
    let mut out = Vec::new();
    walk(p, 0, 0, cap, &mut [0; SECTION], &mut out);
    out.sort_by_key(|&(k, c)| (c, k));
    out
}

struct SharedK {
    mid: Vec<[i32; SECTION]>,
    end: [Vec<[i32; SECTION]>; 2],
}

fn shared_k(p: u32) -> SharedK {
    let mut cap = 64i64;
    loop {
        let all = enum_k(p, cap);
        let par = |k: &[i32; SECTION]| k.iter().map(|&v| v.rem_euclid(2)).sum::<i32>() % 2;
        let n0 = all.iter().filter(|(k, _)| par(k) == 0).count();
        let n1 = all.len() - n0;
        if all.len() >= 4096 && n0 >= 2048 && n1 >= 2048 {
            let mid: Vec<_> = all.iter().take(2048).map(|&(k, _)| k).collect();
            let end0: Vec<_> = all.iter().filter(|(k, _)| par(k) == 0).take(2048).map(|&(k, _)| k).collect();
            let end1: Vec<_> = all.iter().filter(|(k, _)| par(k) == 1).take(2048).map(|&(k, _)| k).collect();
            return SharedK { mid, end: [end0, end1] };
        }
        cap *= 2;
    }
}

fn s_shared_k(set: &SectionSet, ks: &[[i32; SECTION]]) -> (f64, i32) {
    let mut sum = 0i64;
    let mut maxabs = 0i32;
    for &c in &set.patterns {
        for k in ks {
            let y = set.point(c, k);
            sum += norm2(&y);
            maxabs = maxabs.max(y.iter().map(|v| v.abs()).max().unwrap());
        }
    }
    (sum as f64 / (set.patterns.len() * ks.len()) as f64, maxabs)
}

// ---------------------------------------------------------------------------
// Route 3b — rank space. Coordinate j has offset o = p + 2c_j ∈ {0,1,2,3}; its
// admissible values o + 4Z are listed outward from zero, positive first on
// ties, and a table entry stores the RANK ρ_j in that list. A rank vector then
// decodes under every pattern and both parities, through one arithmetic map.
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

/// The rank map is a bijection onto the progression, listed outward from 0.
fn check_val() {
    for o in 0..4u32 {
        let mut prog: Vec<i32> = (-40i32..=40).filter(|&v| v.rem_euclid(4) == o as i32).collect();
        prog.sort_by_key(|&v| (v.abs(), -v));
        for (rho, &v) in prog.iter().take(12).enumerate() {
            assert_eq!(val(o, rho as u32), v, "val({o}, {rho})");
        }
    }
}

fn rank_cost(rho: &[u32; SECTION]) -> i64 {
    rho.iter().map(|&r| (2 * r as i64 + 1).pow(2)).sum()
}

/// Parity of Σk for pattern c under parity p, of the point decoded from ρ.
fn rank_kparity(p: u32, c: u8, rho: &[u32; SECTION]) -> u32 {
    (0..SECTION)
        .map(|j| {
            let o = p + 2 * ((c >> j) & 1) as u32;
            let y = val(o, rho[j]);
            ((y - o as i32).div_euclid(4)).rem_euclid(2) as u32
        })
        .fold(0, |a, b| a ^ b)
}

/// The pattern-free class: Σ [ρ_j ∈ {1, 2}] mod 2.
fn rank_class(rho: &[u32; SECTION]) -> u32 {
    rho.iter().filter(|&&r| r == 1 || r == 2).count() as u32 & 1
}

struct RankTable {
    /// Per parity class, the 2048 lowest-cost rank vectors.
    class: [Vec<[u32; SECTION]>; 2],
    /// The 2048 lowest-cost overall (the middle's mixed region), and how many
    /// of them fall in class 0.
    mixed: Vec<[u32; SECTION]>,
    n0_mixed: usize,
    max_rank: u32,
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
            // The cap must exceed the largest cost kept, so no vector of that
            // cost was pruned before the sort.
            let kept = all.iter().take(4096).map(|&(_, c)| c).max().unwrap();
            let last0 = rank_cost(&c0[2047]);
            let last1 = rank_cost(&c1[2047]);
            if kept < cap && last0 < cap && last1 < cap {
                let mixed: Vec<_> = all.iter().take(2048).map(|&(r, _)| r).collect();
                let n0_mixed = mixed.iter().filter(|r| rank_class(r) == 0).count();
                let max_rank = c0.iter().chain(&c1).flat_map(|r| r.iter().copied()).max().unwrap();
                return RankTable { class: [c0, c1], mixed, n0_mixed, max_rank };
            }
        }
        cap *= 2;
    }
}

fn s_rank(set: &SectionSet, rows: &[[u32; SECTION]]) -> (f64, i32) {
    let mut sum = 0i64;
    let mut maxabs = 0i32;
    for &c in &set.patterns {
        for rho in rows {
            let mut y = [0i32; SECTION];
            for j in 0..SECTION {
                let o = set.p + 2 * ((c >> j) & 1) as u32;
                y[j] = val(o, rho[j]);
            }
            debug_assert!(set.contains(&y) || set.k_parity.is_some());
            sum += norm2(&y);
            maxabs = maxabs.max(y.iter().map(|v| v.abs()).max().unwrap());
        }
    }
    (sum as f64 / (set.patterns.len() * rows.len()) as f64, maxabs)
}

// ---------------------------------------------------------------------------
// Route 4 — the six-section trellis
// ---------------------------------------------------------------------------

fn profile(words: &[u32], i: usize) -> (u32, u32) {
    let low = if i >= 32 { u32::MAX } else { (1u32 << i) - 1 };
    let past = words.iter().filter(|&&c| c & !low == 0).count();
    let future = words.iter().filter(|&&c| c & low == 0).count();
    (past.trailing_zeros(), future.trailing_zeros())
}

fn subspace(words: &[u32], cut: usize) -> Vec<u32> {
    let low = (1u32 << cut) - 1;
    let mut v = vec![0u32];
    for &g in words.iter().filter(|&&c| c & !low == 0 || c & low == 0) {
        if !v.contains(&g) {
            let grown: Vec<u32> = v.iter().map(|&x| x ^ g).collect();
            v.extend(grown);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn reps(words: &[u32], cut: usize) -> (Vec<u32>, usize) {
    let v = subspace(words, cut);
    let r: Vec<u32> = words.iter().map(|&c| v.iter().map(|&x| c ^ x).min().unwrap()).collect();
    let mut d = r.clone();
    d.sort_unstable();
    d.dedup();
    (r, d.len())
}

fn ball_gain_db(n: u32) -> f64 {
    // G_n = Γ(n/2+1)^(2/n) / ((n+2)π), n even; gain vs the cube 1/12.
    let half = n / 2;
    let ln_gamma: f64 = (1..=half).map(|i| (i as f64).ln()).sum();
    let g = (2.0 * ln_gamma / n as f64).exp() / ((n as f64 + 2.0) * std::f64::consts::PI);
    10.0 * ((1.0 / 12.0) / g).log10()
}

fn main() {
    let t = Trellis::new();
    check_val();
    let (ends, mids) = family(&t);
    println!("F1 — pistes de réduction de la table du décodeur, découpe {W_END}/{W_MID}/{W_END}");
    println!("{} régions extrêmes (2 × 256), {} régions du milieu\n", ends.len(), mids.len());

    // ------------------------------------------------------------------ 1
    println!("== 1. Orbites sous les 256 changements de signe, max|coord| et parts ==");
    let end_stats: Vec<Stat> = ends.iter().map(|r| stat(&r.set, W_END)).collect();
    let mid_stats: Vec<Stat> = mids.iter().map(|r| stat(&r.set, W_MID)).collect();

    let orbit_table = |regs: &[Reg], stats: &[Stat], w: u32, name: &str| -> Vec<usize> {
        let mut orb: HashMap<Vec<[i32; SECTION]>, usize> = HashMap::new();
        let mut of = Vec::new();
        for r in regs {
            let sig = signature(&r.set);
            let n = orb.len();
            let id = *orb.entry(canonical_sign(&sig)).or_insert(n);
            of.push(id);
        }
        let n_orb = orb.len();
        let mut size = vec![0usize; n_orb];
        let mut maxabs = vec![0i32; n_orb];
        let mut p_of = vec![9u32; n_orb];
        let mut rho = vec![(usize::MAX, 0usize); n_orb];
        for (i, r) in regs.iter().enumerate() {
            let o = of[i];
            size[o] += 1;
            maxabs[o] = maxabs[o].max(stats[i].maxabs);
            p_of[o] = r.set.p;
            rho[o] = (rho[o].0.min(stats[i].rho2), rho[o].1.max(stats[i].rho2));
        }
        let mut order: Vec<usize> = (0..n_orb).collect();
        order.sort_by_key(|&o| (std::cmp::Reverse(size[o]), maxabs[o]));
        let entries = 1usize << w;
        println!("{name} : {n_orb} orbites, {} régions, 2^{w} entrées par orbite", regs.len());
        println!("  {:>6} {:>7} {:>8} {:>9} {:>10} {:>8} {:>8}", "orbite", "régions", "part %", "max|y|", "rho2", "6 o", "4 o");
        let mut shown = 0;
        let mut cum = 0usize;
        for &o in &order {
            cum += size[o];
            if shown < 6 || size[o] > 1 {
                println!(
                    "  {:>6} {:>7} {:>8.2} {:>9} {:>4}..{:<4} {:>6.0} K {:>6.0} K   p={}",
                    o,
                    size[o],
                    100.0 * size[o] as f64 / regs.len() as f64,
                    maxabs[o],
                    rho[o].0,
                    rho[o].1,
                    (entries * 6) as f64 / 1024.0,
                    (entries * 4) as f64 / 1024.0,
                    p_of[o]
                );
            }
            shown += 1;
            if shown == 6 && size[o] == 1 {
                let rest = n_orb - 6;
                println!("  … {rest} orbites de taille 1 ({:.1} % des régions)", 100.0 * (regs.len() - cum) as f64 / regs.len() as f64);
                break;
            }
        }
        let le7 = (0..n_orb).filter(|&o| maxabs[o] <= 7).map(|o| size[o]).sum::<usize>();
        let le8 = (0..n_orb).filter(|&o| maxabs[o] <= 8).map(|o| size[o]).sum::<usize>();
        println!(
            "  régions dont max|y| ≤ 7 : {le7}/{} ({:.1} %) ; ≤ 8 : {le8} ({:.1} %) ; max global {}",
            regs.len(),
            100.0 * le7 as f64 / regs.len() as f64,
            100.0 * le8 as f64 / regs.len() as f64,
            maxabs.iter().max().unwrap()
        );
        println!(
            "  ⚠️ toutes les coordonnées d'une section ont la parité p : l'entrée peut stocker n = (y − p)/2,\n     n ∈ [{}, {}] pour |y| ≤ {}, donc 4 bits signés suffisent QUELLE QUE SOIT l'orbite, sans condition ≤ 7.",
            (-(*maxabs.iter().max().unwrap()) - 1) / 2,
            (*maxabs.iter().max().unwrap()) / 2,
            maxabs.iter().max().unwrap()
        );
        of
    };
    let end_sorb = orbit_table(&ends, &end_stats, W_END, "sections extrêmes");
    println!();
    let mid_sorb = orbit_table(&mids, &mid_stats, W_MID, "section du milieu");

    let e6 = (1usize << W_END) * 6;
    let e4 = (1usize << W_END) * 4;
    let m6 = (1usize << W_MID) * 6;
    let m4 = (1usize << W_MID) * 4;
    println!("\njeu chaud (2 orbites extrêmes les plus consultées + 1 orbite du milieu) :");
    println!("  6 o/entrée : {} + {} = {:.0} Kio", 2 * e6 / 1024, m6 / 1024, (2 * e6 + m6) as f64 / 1024.0);
    println!("  4 o/entrée : {} + {} = {:.0} Kio", 2 * e4 / 1024, m4 / 1024, (2 * e4 + m4) as f64 / 1024.0);
    println!("  une seule orbite du milieu, seule : {} Kio à 6 o, {} Kio à 4 o — jamais ≤ 16 Kio", m6 / 1024, m4 / 1024);

    // ------------------------------------------------------------------ 2
    println!("\n== 2. Orbites sous les permutations signées 2^8·8! ==");
    let mut rng = llvq_core::SplitMix64::new(0xf15e_2026_0905);
    let all_regs: Vec<&Reg> = ends.iter().chain(mids.iter()).collect();
    check_desc_rules(&all_regs, &mut rng);

    for (name, regs, sorb, w) in [("extrêmes", &ends, &end_sorb, W_END), ("milieu", &mids, &mid_sorb, W_MID)] {
        let descs: Vec<Desc> = regs.iter().map(|r| desc_of(&r.set)).collect();
        // Sign flips alone through the descriptors must reproduce f1table's count.
        let ids_sign = orbits_mono(&descs, true, false);
        let n_sign = ids_sign.iter().collect::<HashSet<_>>().len();
        let n_sign_pts = sorb.iter().collect::<HashSet<_>>().len();
        assert_eq!(n_sign, n_sign_pts, "{name}: descriptor sign-orbits {n_sign} ≠ point-set sign-orbits {n_sign_pts}");
        let ids_perm = orbits_mono(&descs, false, true);
        let n_perm = ids_perm.iter().collect::<HashSet<_>>().len();
        let ids = orbits_mono(&descs, true, true);
        let n_mono = ids.iter().collect::<HashSet<_>>().len();
        let inv: HashSet<Vec<[i32; SECTION]>> = regs.iter().map(|r| mono_invariant(&signature(&r.set))).collect();
        let mut size: HashMap<usize, usize> = HashMap::new();
        for &i in &ids {
            *size.entry(i).or_insert(0) += 1;
        }
        let mut sizes: Vec<usize> = size.values().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        println!(
            "{name} : signes seuls {n_sign} (= f1table) · permutations seules {n_perm} · signes ET permutations {n_mono}\n  borne inférieure par l'invariant (multi-ensemble des |coord| triés) : {} classes → {}",
            inv.len(),
            if inv.len() == n_mono { "compte EXACT" } else { "PAS exact, à creuser" }
        );
        println!("  tailles d'orbite : {sizes:?}");
        // What each orbit is, in words.
        let mut seen = HashSet::new();
        for (i, r) in regs.iter().enumerate() {
            if seen.insert(ids[i]) {
                let c0 = r.set.patterns[0];
                let wt = c0.count_ones().min(8 - c0.count_ones());
                println!(
                    "    orbite {} : {} régions, p={}, motif de poids min {}, parité {:?}, ex. {}",
                    ids[i], size[&ids[i]], r.set.p, wt, r.set.k_parity, r.label
                );
            }
        }
        let n_e = if w == W_END { n_mono } else { 0 };
        let _ = n_e;
    }
    let n_end_mono = {
        let descs: Vec<Desc> = ends.iter().map(|r| desc_of(&r.set)).collect();
        orbits_mono(&descs, true, true).iter().collect::<HashSet<_>>().len()
    };
    let n_mid_mono = {
        let descs: Vec<Desc> = mids.iter().map(|r| desc_of(&r.set)).collect();
        orbits_mono(&descs, true, true).iter().collect::<HashSet<_>>().len()
    };
    for (bytes, name) in [(6usize, "6 o"), (4, "4 o")] {
        let tot = n_end_mono * (1 << W_END) * bytes + n_mid_mono * (1 << W_MID) * bytes;
        println!(
            "table complète sous permutations signées à {name}/entrée : {} × {} Kio + {} × {} Kio = {:.0} Kio",
            n_end_mono,
            (1 << W_END) * bytes / 1024,
            n_mid_mono,
            (1 << W_MID) * bytes / 1024,
            tot as f64 / 1024.0
        );
    }
    println!(
        "1 orbite extrême + 1 orbite du milieu : {} Kio à 6 o, {} Kio à 4 o. Le milieu seul est 2^15 entrées : ≥ {} Kio.\n→ Cette piste ne peut PAS atteindre la L1 : le milieu est hors gabarit à lui seul.",
        ((1 << W_END) * 6 + (1 << W_MID) * 6) / 1024,
        ((1 << W_END) * 4 + (1 << W_MID) * 4) / 1024,
        (1 << W_MID) * 4 / 1024
    );

    // ------------------------------------------------------------------ 3
    println!("\n== 3. Région produit partagée par tous les motifs : coût de mise en forme ==");
    let sk = [shared_k(0), shared_k(1)];
    let rt = rank_table();
    println!(
        "table de rangs : 2 × 2048 vecteurs (classe de parité 0/1), rang max {} → {} bits par coordonnée ; milieu mixte 2048 = {} de classe 0 + {} de classe 1",
        rt.max_rank,
        (rt.max_rank + 1).next_power_of_two().trailing_zeros(),
        rt.n0_mixed,
        2048 - rt.n0_mixed
    );
    // The class must be pattern-independent for every family pattern, else the
    // ends could not select a parity prefix without knowing the pattern.
    let mut checked = 0;
    for r in ends.iter().chain(mids.iter()) {
        for &c in &r.set.patterns {
            for rho in rt.class[0].iter().take(64).chain(rt.class[1].iter().take(64)) {
                assert_eq!(rank_kparity(r.set.p, c, rho), rank_class(rho), "class is pattern-dependent on {}", r.label);
                checked += 1;
            }
        }
    }
    println!("  classe de parité indépendante du motif : vérifiée sur {checked} (motif, rang)");

    struct Agg {
        s_true: f64,
        s_k: f64,
        s_rank: f64,
        s_rank_a: f64,
        n: usize,
        max_k: i32,
        max_r: i32,
    }
    let mut agg: HashMap<(u8, u32), Agg> = HashMap::new(); // (kind, p)
    println!("\n  {:<14} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}", "région", "rho2", "S exact", "S K-part", "S rangs", "ΔK dB", "Δrangs dB");
    let mut show = |label: &str, kind: u8, set: &SectionSet, st: &Stat, verbose: bool| {
        let p = set.p;
        let (s_k, max_k) = match set.k_parity {
            Some(r) => s_shared_k(set, &sk[p as usize].end[r as usize]),
            None => s_shared_k(set, &sk[p as usize].mid),
        };
        let (s_r, max_r, s_ra) = match set.k_parity {
            Some(r) => {
                let (s, m) = s_rank(set, &rt.class[r as usize]);
                (s, m, s)
            }
            None => {
                let (s, m) = s_rank(set, &rt.mixed);
                let (sa0, _) = s_rank(set, &rt.class[0][..1024]);
                let (sa1, _) = s_rank(set, &rt.class[1][..1024]);
                (s, m, 0.5 * (sa0 + sa1))
            }
        };
        if verbose {
            println!(
                "  {:<14} {:>8} {:>9.3} {:>9.3} {:>9.3} {:>+9.3} {:>+9.3}",
                label,
                st.rho2,
                st.s_true,
                s_k,
                s_r,
                10.0 * (s_k / st.s_true).log10(),
                10.0 * (s_r / st.s_true).log10()
            );
        }
        let a = agg.entry((kind, p)).or_insert(Agg { s_true: 0.0, s_k: 0.0, s_rank: 0.0, s_rank_a: 0.0, n: 0, max_k: 0, max_r: 0 });
        a.s_true += st.s_true;
        a.s_k += s_k;
        a.s_rank += s_r;
        a.s_rank_a += s_ra;
        a.n += 1;
        a.max_k = a.max_k.max(max_k);
        a.max_r = a.max_r.max(max_r);
    };
    // A few representative rows, then the averages over the whole family.
    let mut shown_orb = HashSet::new();
    for (i, r) in ends.iter().enumerate() {
        let verbose = shown_orb.insert(end_sorb[i]) && shown_orb.len() <= 8;
        show(&r.label, 0, &r.set, &end_stats[i], verbose);
    }
    for (i, r) in mids.iter().enumerate() {
        show(&r.label, 1, &r.set, &mid_stats[i], true);
    }
    println!("\n  moyennes sur la famille (S = E‖y‖² par section de 8, régions équiprobables) :");
    let mut block_true = 0.0;
    let mut block_k = 0.0;
    let mut block_r = 0.0;
    let mut block_ra = 0.0;
    for (kind, p) in [(0u8, 0u32), (0, 1), (1, 0), (1, 1)] {
        let a = &agg[&(kind, p)];
        let n = a.n as f64;
        let name = if kind == 0 { "extrêmes" } else { "milieu  " };
        println!(
            "  {name} p={p} ({:>3} régions) : exact {:>8.3}  K-partagé {:>8.3} ({:+.3} dB, max|y| {})  rangs {:>8.3} ({:+.3} dB, max|y| {}){}",
            a.n,
            a.s_true / n,
            a.s_k / n,
            10.0 * (a.s_k / a.s_true).log10(),
            a.max_k,
            a.s_rank / n,
            10.0 * (a.s_rank / a.s_true).log10(),
            a.max_r,
            if kind == 1 { format!("  [1024+1024 : {:.3}, {:+.3} dB]", a.s_rank_a / n, 10.0 * (a.s_rank_a / a.s_true).log10()) } else { String::new() }
        );
        // Block = two end sections + one middle, averaged over p (½ each).
        let wgt = if kind == 0 { 2.0 } else { 1.0 } * 0.5;
        block_true += wgt * a.s_true / n;
        block_k += wgt * a.s_k / n;
        block_r += wgt * a.s_rank / n;
        block_ra += wgt * if kind == 1 { a.s_rank_a / n } else { a.s_rank / n };
    }
    let mse_meas = 2f64.powf(-2.0 * 2.0 * 0.8955);
    let ret = |mse: f64| 100.0 * (-0.5 * mse.log2()) / 2.0;
    println!("\n  bloc de 24 (2 extrêmes + 1 milieu, p équiprobable) : E‖y‖² exact {block_true:.3}");
    for (name, s) in [("K partagé (piste 3 telle qu'énoncée)", block_k), ("rangs partagés, milieu mixte", block_r), ("rangs partagés, milieu 1024+1024", block_ra)] {
        let ratio = s / block_true;
        println!(
            "  {name:<40} E‖y‖² {s:.3}  rapport {ratio:.4}  {:+.3} dB  → rétention *estimée* {:.2} % (depuis 89,55 mesuré, MSE ∝ second moment)",
            10.0 * ratio.log10(),
            ret(mse_meas * ratio)
        );
    }
    println!(
        "  rappel : MSE mesurée de F1 12/15/12 = 2^(−4·0,8955) = {mse_meas:.5} ; 1 pp de rétention = {:.2} % de MSE",
        100.0 * (2f64.powf(0.04) - 1.0)
    );

    // ------------------------------------------------------------------ 4
    println!("\n== 4. Six sections de quatre coordonnées ==");
    let words = &t.code.words;
    println!("  profil du treillis sous l'ordre trio épinglé (états Golay ; Λ₂₄ = ×4) :");
    println!("  {:>4} {:>7} {:>8} {:>12} {:>12}", "coupe", "k_passé", "k_futur", "états Golay", "états Λ₂₄");
    for i in [0usize, 4, 8, 12, 16, 20, 24] {
        let (kp, kf) = profile(words, i);
        let bits = 12 - kp - kf;
        println!("  {i:>4} {kp:>7} {kf:>8} {:>12} {:>12}", 1u32 << bits, if i == 0 || i == 24 { 1 } else { 1u32 << (bits + 2) });
    }
    // Which half of each octad comes first is a free choice; the cut count at
    // 4, 12 and 20 depends on it. Search the 70 halves of each octad.
    let mut best_order: Vec<u32> = (0..24).collect();
    for (oct, cut) in [(0usize, 4usize), (1, 12), (2, 20)] {
        let base = oct * 8;
        let mut best = (u32::MAX, Vec::new());
        for mask in 0u32..256 {
            if mask.count_ones() != 4 {
                continue;
            }
            let mut order: Vec<u32> = (0..24).collect();
            let first: Vec<u32> = (0..8).filter(|j| mask >> j & 1 == 1).map(|j| (base + j) as u32).collect();
            let second: Vec<u32> = (0..8).filter(|j| mask >> j & 1 == 0).map(|j| (base + j) as u32).collect();
            for (k, &v) in first.iter().chain(second.iter()).enumerate() {
                order[base + k] = v;
            }
            let permuted: Vec<u32> = words.iter().map(|&w| (0..24).fold(0u32, |acc, j| acc | (w >> order[j] & 1) << j)).collect();
            let (kp, kf) = profile(&permuted, cut);
            let bits = 12 - kp - kf;
            if bits < best.0 {
                best = (bits, order.clone());
            }
        }
        best_order[base..base + 8].copy_from_slice(&best.1[base..base + 8]);
        println!("  coupe {cut} : meilleure moitié d'octade → 2^{} états Golay (2^{} Λ₂₄)", best.0, best.0 + 2);
    }
    let words6: Vec<u32> = words.iter().map(|&w| (0..24).fold(0u32, |acc, j| acc | (w >> best_order[j] & 1) << j)).collect();
    let cuts = [0usize, 4, 8, 12, 16, 20, 24];
    let mut states = Vec::new();
    let mut repv = Vec::new();
    for &c in &cuts {
        let (r, n) = reps(&words6, c);
        states.push(n);
        repv.push(r);
    }
    println!("\n  sous l'ordre optimisé : états Golay aux coupes {:?} = {:?}", cuts, states);
    println!("  {:>7} {:>7} {:>9} {:>10} {:>10} {:>9}", "section", "motifs", "arêtes", "sortants", "entrants", "V (libre)");
    let mut out_deg = Vec::new();
    let mut in_deg = Vec::new();
    for s in 0..6 {
        let a = cuts[s];
        let mut edges: Vec<(u32, u32, u32)> = words6.iter().enumerate().map(|(i, &c)| (repv[s][i], (c >> a) & 0xf, repv[s + 1][i])).collect();
        edges.sort_unstable();
        edges.dedup();
        let mut pats: Vec<u32> = edges.iter().map(|e| e.1).collect();
        pats.sort_unstable();
        pats.dedup();
        let od = edges.len() / states[s];
        let id = edges.len() / states[s + 1];
        // Every state must have the same out-degree, else the label is not a fixed width.
        let mut per: HashMap<u32, usize> = HashMap::new();
        for e in &edges {
            *per.entry(e.0).or_insert(0) += 1;
        }
        assert!(per.values().all(|&v| v == od), "section {s}: out-degree not uniform");
        out_deg.push(od);
        in_deg.push(id);
        println!("  {:>7} {:>7} {:>9} {:>10} {:>10} {:>9}", s + 1, pats.len(), edges.len(), od, id, 256 / od);
    }
    // Word structure: transmit the Λ₂₄ state at the cut with the fewest states,
    // decode backward before it and forward after it. Sections decoded forward
    // choose their branch by the out-degree, backward by the in-degree; the
    // section at either end of the block has its Σk parity forced.
    let (tcut, tstates) = (1..6).map(|i| (i, states[i])).min_by_key(|&(_, n)| n).unwrap();
    let state_bits = (tstates as f64).log2() as u32 + 2;
    println!("\n  état transmis à la coupe {} : {} états Golay × 4 = {} bits", cuts[tcut], tstates, state_bits);
    let mut sum_log_v = 0f64;
    let mut vs = Vec::new();
    for s in 0..6 {
        let forward = s >= tcut;
        let branches = if forward { out_deg[s] } else { in_deg[s] };
        let constrained = s == 0 || s == 5;
        let v = 256.0 * if constrained { 2.0 } else { 1.0 } / branches as f64;
        vs.push((branches, constrained, v));
        sum_log_v += v.log2();
    }
    let label_bits = 47 - state_bits;
    let c = (label_bits as f64 + sum_log_v) / 6.0;
    println!("  bits d'étiquette : 47 − {state_bits} = {label_bits} ; découpe isotrope w_i = C − log2 V_i, C = {c:.3}");
    let mut ws = Vec::new();
    for (s, &(br, con, v)) in vs.iter().enumerate() {
        let w = c - v.log2();
        ws.push(w);
        println!(
            "    section {} : {} branches{}, covolume {:.0}, w ≈ {:.2} bits (dont {:.0} codés) → 2^{} entrées × 4 coord",
            s + 1,
            br,
            if con { ", parité forcée" } else { "" },
            v,
            w,
            (br as f64).log2(),
            w.round() as u32
        );
    }
    let wsum: f64 = ws.iter().map(|w| w.round()).sum();
    println!("  somme des arrondis : {wsum:.0} bits (à ajuster à {label_bits} par ±1 sur une section)");
    // Sign-orbit / monomial-orbit count of the 4-dimensional section sets.
    let mut total_bytes = 0usize;
    for s in 0..6 {
        let (a, _) = (cuts[s], cuts[s + 1]);
        let forward = s >= tcut;
        let side = if forward { s } else { s + 1 };
        let mut by_state: HashMap<u32, Vec<u32>> = HashMap::new();
        for (i, &c) in words6.iter().enumerate() {
            by_state.entry(repv[side][i]).or_default().push((c >> a) & 0xf);
        }
        let mut descs = Vec::new();
        for pats in by_state.values() {
            let mut ps = pats.clone();
            ps.sort_unstable();
            ps.dedup();
            for p in 0..2u8 {
                let rs: Vec<u8> = if s == 0 || s == 5 { vec![0, 1] } else { vec![2] };
                for r in rs {
                    descs.push(Desc { n: 4, p, pats: ps.iter().map(|&c| (c, r)).collect() });
                }
            }
        }
        let n_sign = orbits_mono(&descs, true, false).iter().collect::<HashSet<_>>().len();
        let n_mono = orbits_mono(&descs, true, true).iter().collect::<HashSet<_>>().len();
        let w = ws[s].round() as u32;
        let bytes = n_sign * (1usize << w) * 2;
        total_bytes += bytes;
        println!(
            "    section {} : {} ensembles → {} orbites (signes) / {} (permutations signées) ; table {} × 2^{w} × 2 o = {:.1} Kio",
            s + 1,
            descs.len(),
            n_sign,
            n_mono,
            n_sign,
            bytes as f64 / 1024.0
        );
    }
    println!("  total à 2 o/entrée (4 nibbles), orbites de signe : {:.1} Kio ; 6 consultations par bloc", total_bytes as f64 / 1024.0);
    let g4 = ball_gain_db(4);
    let g8 = ball_gain_db(8);
    let g24 = ball_gain_db(24);
    println!(
        "\n  gain de forme de la boule : n=4 {g4:.4} dB · n=8 {g8:.4} dB · n=24 {g24:.4} dB ; 6×4 contre 3×8 : {:.4} dB de perte",
        g8 - g4
    );
    let loss = g8 - g4;
    let r_model = ret(mse_meas * 10f64.powf(loss / 10.0));
    // Calibration on the one split measured against the model: 13/13/13 lost
    // 89.55 − 85.96 = 3.59 pp where the product-ceiling model predicted
    // 0.7150 − 0.5175 = 0.1975 dB.
    let meas_db = 10.0 * (2f64.powf(2.0 * 2.0 * (0.8955 - 0.8596))).log10();
    let model_db = 0.7150 - 0.5175;
    let factor = meas_db / model_db;
    println!(
        "  rétention *estimée* 6×4 depuis 89,55 : {r_model:.2} % (modèle second moment, −{loss:.3} dB)\n  calibration : 13/13/13 a perdu {meas_db:.3} dB mesurés contre {model_db:.3} dB au modèle, facteur {factor:.2} → 6×4 calibré {:.2} %",
        ret(mse_meas * 10f64.powf(loss * factor / 10.0))
    );
}
