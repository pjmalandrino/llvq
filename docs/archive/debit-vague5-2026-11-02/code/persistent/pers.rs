//! DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
//! persistent-kernel lead (wave5-deep/persistent.md, sections 3 and 5), and
//! revised the same day after an adversarial review.
//!
//! Sketch of `llvq-cuda/src/pers.rs`: the portable half of the persistent
//! bench arms, on the model of `occ.rs` (A3). Portable so that `cargo test`
//! exercises it on the Mac: the arm table (one row per kernel symbol of
//! `tetra48_pers.cu`), the `LLVQ_PERS_ARMS` parser, the site table in served
//! order with its waits, signals and prologues, the static row slices, the
//! barrier counts the host needs to advance `bar_base`, the wrap-safe target
//! arithmetic, the staging maps and their bank maps, the byte ledger of the
//! bare file, the shared-memory budget, and a validator that refuses a
//! schedule that could deadlock or read stale data. No `unsafe`, no external
//! dependency: the crate keeps `forbid(unsafe_code)` for this file.
//!
//! What the review changed here: every arm is a row of one table that names
//! its symbol and its template parameters, so "every arm is its own symbol and
//! the unit never changes with the selection" holds by construction (finding
//! 13); signals are flagged on the 48 waited sites only and the two trailing
//! barriers are counted (finding 6); the h estimator's divisor is computed
//! here, 145, not written as 97 (finding 6); the rows series is gone from this
//! registry (finding 12).

/// One persistent arm: a kernel symbol and the template parameters behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersArm {
    pub name: &'static str,
    pub symbol: &'static str,
    /// 0 staging and reduction, 1 word stream, 2 decode with tables in L1,
    /// 3 decode with tables in shared memory.
    pub mode: u8,
    pub threads: u32,
    /// Floats between two staged blocks: 24 (as today), 25, 28.
    pub stride: u32,
    /// Trio-order staging, read as six float4 a block.
    pub trio: bool,
    /// Register prefetch of block j + 32's words.
    pub prefetch: bool,
    /// 0 none, 1 the served schedule, 2 the served schedule with every sync doubled.
    pub waits: u8,
    /// Per-CTA norm and rotation prologues (route P's), work only.
    pub prologue: bool,
    /// Rows compared with the served `tv_tetra48_h` bit for bit before timing.
    pub owes_identity: bool,
}

const fn arm(
    name: &'static str,
    symbol: &'static str,
    mode: u8,
    threads: u32,
    stride: u32,
    trio: bool,
    prefetch: bool,
    waits: u8,
    prologue: bool,
) -> PersArm {
    PersArm { name, symbol, mode, threads, stride, trio, prefetch, waits, prologue, owes_identity: mode >= 2 }
}

/// Registration order is dispatch order (the arms.rs rule: a selection skips
/// arms, it never moves them). The identity and time reference, `tetra48h`,
/// is not in this table: it is the served unit's own `tv_tetra48_h`, loaded
/// from the served unit's bytes in a module of its own, and it always runs.
pub const PERS_ARMS: [PersArm; N_PERS_ARMS] = [
    arm("pnull", "tv_pers_null", 0, 768, 28, true, false, 0, false),
    arm("pword", "tv_pers_word", 1, 768, 28, true, false, 0, false),
    arm("pl1", "tv_pers_l1", 2, 768, 24, false, false, 0, false),
    arm("psmem", "tv_pers_smem", 3, 768, 24, false, false, 0, false),
    arm("psmem_s25", "tv_pers_smem_s25", 3, 768, 25, false, false, 0, false),
    arm("psmem_t28", "tv_pers_smem_t28", 3, 768, 28, true, false, 0, false),
    arm("psmem_t28_u2", "tv_pers_smem_t28_u2", 3, 768, 28, true, true, 0, false),
    arm("pt28_512", "tv_pers_t28_512", 3, 512, 28, true, false, 0, false),
    arm("pt28_1024", "tv_pers_t28_1024", 3, 1024, 28, true, false, 0, false),
    arm("pchain_w", "tv_pers_chain_w", 3, 768, 28, true, false, 1, false),
    arm("pchain_w2", "tv_pers_chain_w2", 3, 768, 28, true, false, 2, false),
    arm("pchain_wp", "tv_pers_chain_wp", 3, 768, 28, true, false, 1, true),
];
pub const N_PERS_ARMS: usize = 12;

/// The base whose body `pchain_w`, `pchain_w2` and `pchain_wp` share, so that
/// their differences are the waits and the prologues and nothing else.
pub const CHAIN_BASE: &str = "psmem_t28";

pub fn parse_pers_arms(spec: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(spec) = spec else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if name == "tetra48h" {
            return Err("LLVQ_PERS_ARMS: tetra48h is the reference and always runs; do not name it".into());
        }
        let i = PERS_ARMS
            .iter()
            .position(|a| a.name == name)
            .ok_or_else(|| format!("LLVQ_PERS_ARMS: unknown arm {name:?}"))?;
        if out.contains(&i) {
            return Err(format!("LLVQ_PERS_ARMS: {name:?} named twice"));
        }
        out.push(i);
    }
    // A chained arm is read as a difference against its base: refuse it alone.
    let base = PERS_ARMS.iter().position(|a| a.name == CHAIN_BASE).expect("the base is registered");
    if out.iter().any(|&i| PERS_ARMS[i].waits > 0) && !out.contains(&base) {
        return Err(format!("LLVQ_PERS_ARMS: a pchain arm needs {CHAIN_BASE} in the same rounds"));
    }
    out.sort_unstable();
    Ok(out)
}

/// One projection of the bare file, in served order within a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

/// What a CTA waits for before a site, in the served 4B schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    None,
    /// Grid barrier with ordinal `n` in the site list (W1, W2, W3).
    All(u32),
    /// Every CTA has signalled site `s` (q/k/v -> o, and the int4-pattern
    /// down chunks, laid on Tetra sites here). In the bench every CTA signals,
    /// so this wait is a grid barrier too, and it is counted as one.
    Count(u32),
}

/// The prologue a site runs before its staging, under `pchain_wp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prologue {
    None,
    /// RMSNorm then rotation at n = 2,560 (W1 before q, W2 before gate).
    NormRot2560,
    /// Rotation at n = 9,728 (W3 before a Tetra down).
    Rot9728,
}

#[derive(Clone, Debug)]
pub struct Site {
    pub layer: u32,
    pub proj: Proj,
    pub d_out: u32,
    pub d_in: u32,
    pub wait: Wait,
    /// Every CTA signals this site's counter after its rows (its consumer waits on it).
    pub signal_after: bool,
    pub prologue: Prologue,
}

/// The served 4B wait pattern (wave-2 judge, re-derived by two verifiers):
/// W1 before each layer's q/k/v except layer 0 (35), W2 before gate/up (36),
/// W3 before a Tetra down (layers 0-11 and 24-35: 24). That is 95 barrier
/// ordinals in the site list. The o sites (36) and the int4-pattern downs
/// (layers 12-23: 12) wait on counters: 48. The final norm's W1 and the
/// argmax combine W4 follow the site list: 2 trailing barriers.
pub fn served_4b_schedule() -> Vec<Site> {
    let (h, i) = (2560u32, 9728u32);
    let tetra_down = |l: u32| !(12..24).contains(&l);
    let mut v = Vec::new();
    let mut bar = 0u32;
    for l in 0..36u32 {
        let w1 = if l == 0 {
            Wait::None
        } else {
            let w = Wait::All(bar);
            bar += 1;
            w
        };
        v.push(Site { layer: l, proj: Proj::Q, d_out: 4096, d_in: h, wait: w1, signal_after: false, prologue: Prologue::NormRot2560 });
        v.push(Site { layer: l, proj: Proj::K, d_out: 1024, d_in: h, wait: Wait::None, signal_after: false, prologue: Prologue::None });
        // V is signalled last by every CTA; Q and K precede it in program order.
        v.push(Site { layer: l, proj: Proj::V, d_out: 1024, d_in: h, wait: Wait::None, signal_after: true, prologue: Prologue::None });
        let v_idx = (v.len() - 1) as u32;
        v.push(Site { layer: l, proj: Proj::O, d_out: 2560, d_in: 4096, wait: Wait::Count(v_idx), signal_after: false, prologue: Prologue::None });
        let w2 = Wait::All(bar);
        bar += 1;
        v.push(Site { layer: l, proj: Proj::Gate, d_out: i, d_in: h, wait: w2, signal_after: false, prologue: Prologue::NormRot2560 });
        v.push(Site { layer: l, proj: Proj::Up, d_out: i, d_in: h, wait: Wait::None, signal_after: !tetra_down(l), prologue: Prologue::None });
        let up_idx = (v.len() - 1) as u32;
        let (wd, pd) = if tetra_down(l) {
            let w = Wait::All(bar);
            bar += 1;
            (w, Prologue::Rot9728)
        } else {
            (Wait::Count(up_idx), Prologue::None)
        };
        v.push(Site { layer: l, proj: Proj::Down, d_out: h, d_in: i, wait: wd, signal_after: false, prologue: pd });
    }
    v
}

/// The two barriers after the site list (final norm W1, argmax W4).
pub const N_TRAILING: u32 = 2;

/// Grid-wide syncs of one `pchain_w` launch: the h estimator's divisor.
pub fn n_grid_syncs(s: &[Site]) -> u32 {
    let all = s.iter().filter(|x| matches!(x.wait, Wait::All(_))).count() as u32;
    let cnt = s.iter().filter(|x| matches!(x.wait, Wait::Count(_))).count() as u32;
    all + cnt + N_TRAILING
}

/// Calls of the barrier counter in one launch of an arm, every CTA alike:
/// what the host adds to `bar_base` (times the grid) after the launch.
pub fn n_bar(s: &[Site], waits: u8) -> u32 {
    let all = s.iter().filter(|x| matches!(x.wait, Wait::All(_))).count() as u32;
    let cnt = s.iter().filter(|x| matches!(x.wait, Wait::Count(_))).count() as u32;
    match waits {
        0 => 0,
        1 => all + N_TRAILING,
        // each barrier doubled, and one barrier after each counter wait
        _ => 2 * (all + N_TRAILING) + cnt,
    }
}

/// Static contiguous row slices, balanced by rows within a site.
pub fn slices(d_out: u32, grid: u32) -> Vec<(u32, u32)> {
    (0..grid)
        .map(|c| {
            let lo = (d_out as u64 * c as u64 / grid as u64) as u32;
            let hi = (d_out as u64 * (c as u64 + 1) / grid as u64) as u32;
            (lo, hi)
        })
        .collect()
}

/// Wrap-safe "counter reached target", the device's `pers_reached`.
pub fn reached(v: u32, target: u32) -> bool {
    (v.wrapping_sub(target) as i32) >= 0
}

/// Target of barrier ordinal `n` in a launch whose arrivals start at `base`.
pub fn target(base: u32, n: u32, grid: u32) -> u32 {
    base.wrapping_add((n + 1).wrapping_mul(grid))
}

/// `TETRA48_ORDER` (llvq_tetra48.cuh:70-73): trio slot k holds natural coordinate ORDER[k].
pub const ORDER: [u8; 24] = [0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23];
/// Its inverse, `PERS_INV_ORDER` of tetra48_pers.cu: natural coordinate c lands in slot INV_ORDER[c].
pub const INV_ORDER: [u8; 24] = [0, 1, 2, 3, 4, 16, 8, 5, 17, 18, 6, 9, 7, 10, 11, 19, 12, 13, 14, 15, 20, 21, 22, 23];

/// Staged index of coordinate `i` of an activation of width `d_in`.
pub fn staged_index(i: u32, d_in: u32, stride: u32, trio: bool) -> u32 {
    let nblk = d_in / 24;
    let (b, k) = (i / 24, i % 24);
    if b >= nblk {
        return nblk * stride + (i - nblk * 24);
    }
    b * stride + if trio { INV_ORDER[k as usize] as u32 } else { k }
}

/// Bank-conflict degree of a warp-wide 32-bit load at offset `o` of each
/// lane's block, lanes 0..31 on consecutive blocks; worst over the 24 offsets.
pub fn scalar_conflict_ways(stride: u32) -> u32 {
    (0..24u32)
        .map(|o| {
            let mut count = [0u32; 32];
            for l in 0..32u32 {
                count[((stride * l + o) % 32) as usize] += 1;
            }
            *count.iter().max().expect("32 banks")
        })
        .max()
        .expect("24 offsets")
}

/// The same for 128-bit loads of aligned quads: served per quarter-warp,
/// eight lanes, conflict when two lanes' 16-byte words share a bank group.
pub fn quad_conflict_ways(stride: u32) -> u32 {
    assert_eq!(stride % 4, 0, "128-bit loads need a stride that is a multiple of 4 floats");
    let mut worst = 0;
    for q in 0..6u32 {
        for qw in 0..4u32 {
            let mut count = [0u32; 8];
            for l in qw * 8..qw * 8 + 8 {
                count[(((stride * l + 4 * q) % 32) / 4) as usize] += 1;
            }
            worst = worst.max(*count.iter().max().expect("8 groups"));
        }
    }
    worst
}

/// Bytes of one bench pass over the bare 252-Tetra file, from row formulas:
/// (words with the row-stride padding, f16-family total, f32-family total).
/// The bench prints 0.98 GB for its f32 arm (banc-252 sweep-64).
pub fn bare_file_bytes() -> (u64, u64, u64) {
    let layers = 36u64;
    let rows_2560 = 4096 + 1024 + 1024 + 9728 + 9728u64;
    let (rows_4096, rows_9728) = (2560u64, 2560u64);
    let stride = |nb: u64| (6 * nb).div_ceil(8) * 8;
    let words = layers * (rows_2560 * stride(106) + rows_4096 * stride(170) + rows_9728 * stride(405));
    let tail = layers * (rows_2560 * 16 + rows_4096 * 16 + rows_9728 * 8);
    let rscale = layers * (rows_2560 + rows_4096 + rows_9728) * 4;
    (words, words + 2 * tail + rscale, words + 4 * tail + rscale)
}

/// Dynamic shared bytes of an arm's launch: tables (MODE 3), then the widest
/// staging (d_in = 9,728: 405 blocks and a tail of 8).
pub fn shared_bytes(a: &PersArm) -> u32 {
    let tables = if a.mode == 3 { 16_384 + 128 + 2_048 + 128 + 128 } else { 0 };
    tables + (405 * a.stride + 8) * 4
}

/// Validator: refuses a schedule that could deadlock or read stale data.
/// (a) barrier ordinals are 0, 1, 2, ... in program order (every CTA runs the
///     same site list, so equal ordinals mean the same instance);
/// (b) a Count wait names an earlier site, and that site signals;
/// (c) every signalled site has a consumer (a signal nobody waits for would
///     be timed as part of h);
/// (d) a prologue sits only where the staged width changes, since the kernel
///     runs it on a restage only.
pub fn validate(s: &[Site]) -> Result<(), String> {
    let mut next_bar = 0u32;
    for (idx, site) in s.iter().enumerate() {
        match site.wait {
            Wait::All(n) => {
                if n != next_bar {
                    return Err(format!("site {idx}: barrier {n}, expected {next_bar}"));
                }
                next_bar += 1;
            }
            Wait::Count(p) => {
                if p as usize >= idx {
                    return Err(format!("site {idx}: waits on site {p}, not earlier"));
                }
                if !s[p as usize].signal_after {
                    return Err(format!("site {idx}: waits on site {p}, which never signals"));
                }
            }
            Wait::None => {}
        }
        if site.signal_after && !s.iter().any(|x| x.wait == Wait::Count(idx as u32)) {
            return Err(format!("site {idx}: signals, and no site waits on it"));
        }
        if site.prologue != Prologue::None && idx > 0 && s[idx - 1].d_in == site.d_in {
            return Err(format!("site {idx}: a prologue where the staged width does not change"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_has_the_served_counts() {
        let s = served_4b_schedule();
        assert_eq!(s.len(), 252);
        let all = s.iter().filter(|x| matches!(x.wait, Wait::All(_))).count();
        let cnt = s.iter().filter(|x| matches!(x.wait, Wait::Count(_))).count();
        assert_eq!(all, 35 + 36 + 24);
        assert_eq!(cnt, 36 + 12);
        assert_eq!(s.iter().filter(|x| x.signal_after).count(), 48);
        assert_eq!(s.iter().filter(|x| x.prologue == Prologue::NormRot2560).count(), 72);
        assert_eq!(s.iter().filter(|x| x.prologue == Prologue::Rot9728).count(), 24);
        // the 97 waits per token of the served design = 95 + the 2 trailing barriers
        assert_eq!(all as u32 + N_TRAILING, 97);
        assert_eq!(n_grid_syncs(&s), 145);
        assert_eq!(n_bar(&s, 1), 97);
        assert_eq!(n_bar(&s, 2), 242);
        validate(&s).expect("valid");
    }

    #[test]
    fn the_validator_refuses_what_it_should() {
        let mut s = served_4b_schedule();
        s[2].signal_after = false; // layer 0's V: o of layer 0 now waits on a silent site
        assert!(validate(&s).is_err());
        let mut s = served_4b_schedule();
        s[1].signal_after = true; // K signals, nobody waits
        assert!(validate(&s).is_err());
        let mut s = served_4b_schedule();
        s[1].prologue = Prologue::NormRot2560; // K restages nothing
        assert!(validate(&s).is_err());
        let mut s = served_4b_schedule();
        s[3].wait = Wait::Count(10); // a later producer
        assert!(validate(&s).is_err());
    }

    #[test]
    fn slices_partition_every_site() {
        for d in [1024u32, 2560, 4096, 9728] {
            let sl = slices(d, 142);
            assert_eq!(sl.first().unwrap().0, 0);
            assert_eq!(sl.last().unwrap().1, d);
            assert!(sl.windows(2).all(|w| w[0].1 == w[1].0));
        }
    }

    #[test]
    fn wrap_safe_targets() {
        assert!(reached(5, 3));
        assert!(!reached(3, 5));
        assert!(reached(2, u32::MAX - 1));
        assert_eq!(target(u32::MAX - 141, 0, 142), 0);
    }

    #[test]
    fn bank_maps() {
        assert_eq!(scalar_conflict_ways(24), 8);
        assert_eq!(scalar_conflict_ways(25), 1);
        assert_eq!(scalar_conflict_ways(28), 4);
        assert_eq!(quad_conflict_ways(24), 2);
        assert_eq!(quad_conflict_ways(28), 1);
    }

    #[test]
    fn inv_order_inverts_order() {
        for c in 0..24usize {
            assert_eq!(ORDER[INV_ORDER[c] as usize] as usize, c);
        }
    }

    #[test]
    fn staging_is_a_bijection() {
        for (stride, trio) in [(24, false), (25, false), (28, true)] {
            for d_in in [2560u32, 4096, 9728] {
                let mut seen = std::collections::HashSet::new();
                for i in 0..d_in {
                    assert!(seen.insert(staged_index(i, d_in, stride, trio)));
                }
            }
        }
    }

    #[test]
    fn bytes_match_the_bench_ledger() {
        let (w, f16, f32) = bare_file_bytes();
        assert_eq!(w, 908_328_960);
        // banc-252 sweep-64 prints 0.98 GB for the f32 arm
        assert!((f32 as f64 / 1e9 - 0.98).abs() < 0.005);
        assert!(f16 < f32 && w < f16);
    }

    #[test]
    fn shared_budget_fits_the_opt_in() {
        for a in PERS_ARMS.iter() {
            assert!(shared_bytes(a) + 132 <= 101_376, "{}", a.name);
        }
        let t28 = PERS_ARMS.iter().find(|a| a.name == "psmem_t28").unwrap();
        assert_eq!(shared_bytes(t28), 64_208);
    }

    #[test]
    fn chained_arms_share_the_base_body() {
        let base = PERS_ARMS.iter().find(|a| a.name == CHAIN_BASE).unwrap();
        for a in PERS_ARMS.iter().filter(|a| a.waits > 0) {
            assert_eq!(
                (a.mode, a.threads, a.stride, a.trio, a.prefetch),
                (base.mode, base.threads, base.stride, base.trio, base.prefetch),
                "{}",
                a.name
            );
        }
    }
}
