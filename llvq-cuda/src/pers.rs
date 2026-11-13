//! The persistent Tetra kernel, host half: what every launch of
//! `kernels/tetra48_pers.cu` and of route P's segment kernel depends on, kept
//! portable so `cargo test` checks it on the Mac.
//!
//! Design: `docs/archive/debit-vague5-2026-11-02/persistent.md` §3 and §5.
//! Preregs: `proofs/preregistration-pers-l1-*.md` (step L1) and
//! `proofs/preregistration-pers-l2-*.md` (step L2).
//!
//! What lives here, and why each piece cannot wait for a card:
//!
//! - the arm table of the L2 bench (one row per kernel symbol), and the
//!   `LLVQ_PERS_ARMS` parser;
//! - the site list of the served 4B pattern laid on the bare file, its waits,
//!   signals and prologues, and a validator that refuses a schedule that could
//!   deadlock or read stale data;
//! - the barrier arithmetic: how many arrivals one launch adds, the wrap-safe
//!   targets, and a ledger that keeps `bar_base` and the counter epochs right
//!   across arms, identity launches and aborts;
//! - the staging maps, their bank maps, the byte ledger and the shared budget;
//! - the row judge: identical, moved association, or defect;
//! - the route-P segment plan per model, and the 4E wait counts.
//!
//! A grid one barrier short deadlocks, and a counter target one epoch off
//! reads stale rows: both fail silently or hang on a rented card.

/// One persistent arm: a kernel symbol and the template parameters behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersArm {
    pub name: &'static str,
    pub symbol: &'static str,
    /// 0 staging and reduction, 1 word stream, 2 decode with the tables read
    /// in place through L1, 3 decode with the tables copied to shared memory,
    /// 4 and 5 the decode of 2 and 3 without the dot (values XOR-folded, no
    /// activation read, no FMA).
    pub mode: u8,
    pub threads: u32,
    /// Floats between two staged blocks: 24 (as today), 25, 28.
    pub stride: u32,
    /// Trio-order staging, read as six float4 a block.
    pub trio: bool,
    /// Register prefetch of block j + 32's words.
    pub prefetch: bool,
    /// 0 none, 1 the served schedule, 2 the served schedule with one more grid
    /// barrier after every sync.
    pub waits: u8,
    /// Per-CTA norm and rotation prologues (route P's), work only.
    pub prologue: bool,
}

impl PersArm {
    /// Decode arms owe the served kernel's rows bit for bit.
    pub const fn owes_identity(&self) -> bool {
        self.mode == 2 || self.mode == 3
    }

    /// The tables sit in shared memory, copied once a launch.
    pub const fn tables_in_shared(&self) -> bool {
        self.mode == 3 || self.mode == 5
    }
}

const fn arm(name: &'static str, symbol: &'static str, mode: u8, threads: u32, stride: u32, trio: bool) -> PersArm {
    PersArm { name, symbol, mode, threads, stride, trio, prefetch: false, waits: 0, prologue: false }
}

/// Registration order is dispatch order (the `arms.rs` rule: a selection
/// skips arms, it never moves them). The identity and time reference,
/// `tetra48h`, is not in this table: it is the served unit's own
/// `tv_tetra48_h`, compiled from the served unit's bytes, and it always runs.
pub const PERS_ARMS: [PersArm; N_PERS_ARMS] = [
    arm("pnull", "tv_pers_null", 0, 768, 28, true),
    arm("pword", "tv_pers_word", 1, 768, 28, true),
    arm("pdec_l1", "tv_pers_dec_l1", 4, 768, 28, true),
    arm("pdec_smem", "tv_pers_dec_smem", 5, 768, 28, true),
    arm("pl1", "tv_pers_l1", 2, 768, 24, false),
    arm("pl1_t28", "tv_pers_l1_t28", 2, 768, 28, true),
    arm("psmem", "tv_pers_smem", 3, 768, 24, false),
    arm("psmem_s25", "tv_pers_smem_s25", 3, 768, 25, false),
    arm("psmem_t28", "tv_pers_smem_t28", 3, 768, 28, true),
    PersArm { prefetch: true, ..arm("psmem_t28_u2", "tv_pers_smem_t28_u2", 3, 768, 28, true) },
    arm("pt28_512", "tv_pers_t28_512", 3, 512, 28, true),
    arm("pt28_1024", "tv_pers_t28_1024", 3, 1024, 28, true),
    PersArm { waits: 1, ..arm("pchain_w", "tv_pers_chain_w", 3, 768, 28, true) },
    PersArm { waits: 2, ..arm("pchain_w2", "tv_pers_chain_w2", 3, 768, 28, true) },
    PersArm { waits: 1, prologue: true, ..arm("pchain_wp", "tv_pers_chain_wp", 3, 768, 28, true) },
];
pub const N_PERS_ARMS: usize = 15;

/// The arm whose body `pchain_w`, `pchain_w2` and `pchain_wp` share, so their
/// differences are the waits and the prologues and nothing else.
pub const CHAIN_BASE: &str = "psmem_t28";

/// `LLVQ_PERS_ARMS`: a comma-separated list of arm names, or unset (no arm).
/// Unknown or repeated names are refused, the reference cannot be named, and
/// a chained arm needs its base in the same rounds. The result is in
/// registration order whatever order the variable lists.
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
    if out.is_empty() {
        return Err("LLVQ_PERS_ARMS is set and names no arm; unset it instead".into());
    }
    let base = PERS_ARMS.iter().position(|a| a.name == CHAIN_BASE).expect("the base is registered");
    if out.iter().any(|&i| PERS_ARMS[i].waits > 0) && !out.contains(&base) {
        return Err(format!("LLVQ_PERS_ARMS: a pchain arm is read against {CHAIN_BASE}, name it in the same rounds"));
    }
    out.sort_unstable();
    Ok(out)
}

/// One projection of a layer, in served order.
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

/// What a CTA waits for before a site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    None,
    /// Grid barrier with ordinal `n` in the launch.
    All(u32),
    /// Every contributor has signalled site `s`'s counter. In the bench every
    /// CTA computes rows of every site, so every CTA contributes and this wait
    /// is a grid-wide sync too: [`n_grid_syncs`] counts it as one.
    Count(u32),
}

/// The prologue a site runs before its staging, under `pchain_wp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prologue {
    None,
    /// RMSNorm then rotation at the hidden width (W1 before q, W2 before gate).
    NormRot,
    /// Rotation at the intermediate width (W3 before a Tetra down).
    Rot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub layer: u32,
    pub proj: Proj,
    pub d_out: u32,
    pub d_in: u32,
    pub wait: Wait,
    /// Every CTA signals this site's counter after its rows.
    pub signal_after: bool,
    pub prologue: Prologue,
}

/// The served 4B wait pattern laid on the bare 252-Tetra file (wave-2 judge,
/// re-derived by two verifiers): W1 before each layer's q/k/v except layer
/// 0 (35), W2 before gate/up (36), W3 before a Tetra down (layers 0-11 and
/// 24-35: 24). That is 95 barrier ordinals in the site list. The o sites (36)
/// and the int4-pattern downs (layers 12-23: 12) wait on counters: 48. The
/// final norm's W1 and the argmax combine W4 follow the site list:
/// [`N_TRAILING`] trailing barriers.
pub fn served_4b_schedule() -> Vec<Site> {
    schedule_for(36, 2560, 9728, |l| !(12..24).contains(&l))
}

/// The same pattern on `layers` layers of widths `h` and `i`: the host test
/// runs it on a short model, the bench on the 4B.
pub fn schedule_for(layers: u32, h: u32, i: u32, tetra_down: impl Fn(u32) -> bool) -> Vec<Site> {
    let mut v = Vec::new();
    let mut bar = 0u32;
    let mut next_bar = || {
        let w = Wait::All(bar);
        bar += 1;
        w
    };
    for l in 0..layers {
        let w1 = if l == 0 { Wait::None } else { next_bar() };
        v.push(Site { layer: l, proj: Proj::Q, d_out: 4096, d_in: h, wait: w1, signal_after: false, prologue: Prologue::NormRot });
        v.push(Site { layer: l, proj: Proj::K, d_out: 1024, d_in: h, wait: Wait::None, signal_after: false, prologue: Prologue::None });
        // V is signalled last by every CTA; Q and K precede it in program order.
        v.push(Site { layer: l, proj: Proj::V, d_out: 1024, d_in: h, wait: Wait::None, signal_after: true, prologue: Prologue::None });
        let v_idx = (v.len() - 1) as u32;
        v.push(Site { layer: l, proj: Proj::O, d_out: h, d_in: 4096, wait: Wait::Count(v_idx), signal_after: false, prologue: Prologue::None });
        let w2 = next_bar();
        v.push(Site { layer: l, proj: Proj::Gate, d_out: i, d_in: h, wait: w2, signal_after: false, prologue: Prologue::NormRot });
        let td = tetra_down(l);
        v.push(Site { layer: l, proj: Proj::Up, d_out: i, d_in: h, wait: Wait::None, signal_after: !td, prologue: Prologue::None });
        let up_idx = (v.len() - 1) as u32;
        let (wd, pd) = if td { (next_bar(), Prologue::Rot) } else { (Wait::Count(up_idx), Prologue::None) };
        v.push(Site { layer: l, proj: Proj::Down, d_out: h, d_in: i, wait: wd, signal_after: false, prologue: pd });
    }
    v
}

/// The two barriers after the site list (final norm W1, argmax W4).
pub const N_TRAILING: u32 = 2;

/// Grid-wide syncs of one `pchain_w` launch: the divisor of the h estimators.
///
/// Every counter wait of the bench is a grid-wide sync, because every CTA
/// signals every waited site; dividing the chained arm's extra time by the 97
/// barrier calls alone would overstate h by 145/97 = 1.49 (review finding 6).
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

/// Static contiguous row slices, balanced by rows within a site. `granule`
/// rounds every boundary to a multiple (8 for the gate/up pairs of a segment,
/// 1 for the bench), so a slice never splits what one warp must do together.
pub fn slices(d_out: u32, grid: u32, granule: u32) -> Vec<(u32, u32)> {
    assert!(granule > 0 && d_out.is_multiple_of(granule), "d_out {d_out} is not a multiple of {granule}");
    let units = (d_out / granule) as u64;
    (0..grid as u64)
        .map(|c| {
            let lo = (units * c / grid as u64) as u32 * granule;
            let hi = (units * (c + 1) / grid as u64) as u32 * granule;
            (lo, hi)
        })
        .collect()
}

/// The `[site][cta]` slice tables the device reads, `lo` then `hi`.
pub fn slice_tables(s: &[Site], grid: u32) -> (Vec<u32>, Vec<u32>) {
    let mut lo = Vec::with_capacity(s.len() * grid as usize);
    let mut hi = Vec::with_capacity(s.len() * grid as usize);
    for site in s {
        for (a, b) in slices(site.d_out, grid, 1) {
            lo.push(a);
            hi.push(b);
        }
    }
    (lo, hi)
}

/// Wrap-safe "counter reached target", the device's `pers_reached`.
pub fn reached(v: u32, target: u32) -> bool {
    (v.wrapping_sub(target) as i32) >= 0
}

/// Target of barrier ordinal `n` in a launch whose arrivals start at `base`.
pub fn target(base: u32, n: u32, grid: u32) -> u32 {
    base.wrapping_add((n + 1).wrapping_mul(grid))
}

/// Floats one counter region holds: one counter per site, rounded up so each
/// arm's region starts on its own 128-byte line.
pub fn counter_region(nsites: usize) -> usize {
    nsites.div_ceil(32) * 32
}

/// The barrier and counter bookkeeping of one runtime, across launches.
///
/// The device keeps two kinds of monotonic counters and never resets them
/// inside a launch: one barrier counter, and one counter per waited site. The
/// host passes, per launch, the barrier arrivals already made (`bar_base`)
/// and the number of earlier launches that signalled the same counters (the
/// counter epoch).
///
/// Defect of the 2026-11-02 draft fixed here: the draft passed each arm its
/// OWN launch count as the epoch while every chained arm signalled the SAME
/// counter array. After one `pchain_w` launch, `pchain_wp`'s first launch
/// targeted `1 × grid` on counters already at `1 × grid`: its counter waits
/// were satisfied before any of its own rows existed, and it read stale rows
/// without a hang to show it. Each arm now owns a counter region, and its
/// epoch is the number of its own launches that signalled.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
    bar_base: u32,
    epochs: Vec<u32>,
    poisoned: bool,
}

/// What one launch passes the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchCounters {
    pub bar_base: u32,
    /// Counter epoch of this arm: the target of a counter wait is
    /// `(epoch + 1) × contributors`.
    pub epoch: u32,
    /// First counter of this arm's region, in counters.
    pub cnt_offset: u32,
}

impl Ledger {
    pub fn new(n_arms: usize) -> Self {
        Ledger { bar_base: 0, epochs: vec![0; n_arms], poisoned: false }
    }

    /// Counters for the next launch of arm `a`, whose sites number `nsites`.
    pub fn before(&self, a: usize, nsites: usize) -> Result<LaunchCounters, String> {
        if self.poisoned {
            return Err("persistent counters: an earlier launch aborted; zero the counters and reset the ledger".into());
        }
        Ok(LaunchCounters { bar_base: self.bar_base, epoch: self.epochs[a], cnt_offset: (a * counter_region(nsites)) as u32 })
    }

    /// Record a completed launch of arm `a`: `n_bar` barrier calls by each of
    /// `grid` CTAs, and whether its sites signalled their counters.
    pub fn after(&mut self, a: usize, n_bar: u32, grid: u32, signalled: bool) {
        self.bar_base = self.bar_base.wrapping_add(n_bar.wrapping_mul(grid));
        if signalled {
            self.epochs[a] += 1;
        }
    }

    /// A launch that aborted left the counters at an unknown value.
    pub fn aborted(&mut self) {
        self.poisoned = true;
    }

    /// After the host zeroed every counter and the abort flag.
    pub fn reset(&mut self) {
        self.bar_base = 0;
        self.epochs.iter_mut().for_each(|e| *e = 0);
        self.poisoned = false;
    }
}

/// `TETRA48_ORDER` (llvq_tetra48.cuh): trio slot k holds natural coordinate ORDER[k].
pub const ORDER: [u8; 24] = [0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23];
/// Its inverse, `PERS_INV_ORDER` of `llvq_pers.cuh`: natural coordinate c
/// lands in slot INV_ORDER[c].
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

/// Bytes of the decoder tables a MODE 3 arm copies to shared memory, invnorm
/// included: 16,384 + 128 + 2,048 + 128 + 128.
pub const TABLE_BYTES: u32 = 18_816;
/// Fixed header of the dynamic shared buffer: claim cursor, wait flags, norm
/// partials (32 floats) and spare, kept 16-byte aligned. The persistent
/// kernels declare no static `__shared__`, so the host harness can give each
/// CTA its own copy.
pub const SMEM_HEADER: u32 = 256;

/// Dynamic shared bytes of an arm's launch: header, tables (MODE 3), then the
/// widest staging (d_in = 9,728: 405 blocks and a tail of 8), or the widest
/// prologue vector, whichever is larger.
pub fn shared_bytes(a: &PersArm) -> u32 {
    let tables = if a.tables_in_shared() { TABLE_BYTES } else { 0 };
    let staging = (405 * a.stride + 8) * 4;
    let prologue = if a.prologue { 9728 * 4 } else { 0 };
    (SMEM_HEADER + tables + staging.max(prologue)).div_ceil(16) * 16
}

/// The register-WHT rotation prologue's plan for a width: `m` the largest
/// power of two dividing `n`, `k = n / m`, as `llvq_quant::Rotation` and the
/// served `rot_apply` split it. The prologue runs the stages below 512 in
/// registers (one warp a 512-chunk, 16 values a lane) and the stages from 512
/// to `m/2` in shared memory, so it needs `m >= 512` and `k <= KMAX`.
pub fn rot_mk(n: u32) -> Result<(u32, u32), String> {
    if n == 0 {
        return Err("rotation of width 0".into());
    }
    let m = 1u32 << n.trailing_zeros();
    let k = n / m;
    if m < 512 {
        return Err(format!("width {n}: power-of-two factor {m} < 512, the register WHT needs whole 512-chunks"));
    }
    if k > ROT_KMAX {
        return Err(format!("width {n}: odd factor {k} > {ROT_KMAX}"));
    }
    Ok((m, k))
}

/// `LLVQ_ROT_KMAX` of `llvq_rot.cuh`.
pub const ROT_KMAX: u32 = 32;

/// How a row a persistent kernel computed compares with the served row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowVerdict {
    /// The binary16 output is the served one, bit for bit.
    Identical,
    /// Different bits, and the f32 value before `f2h` within 1e-5 of
    /// `Σ|w·x|` of the f64 reference: an association moved. Timed, reported.
    Moved { f32_err: f64, h_err: f64 },
    /// Outside that: the arm is invalidated.
    Defect { f32_err: f64, h_err: f64 },
    /// The row still holds the poison: it was never written.
    Unwritten,
}

/// The tolerance of an f32 value against its f64 reference, relative to the
/// row's `Σ|w·x|` (floor 1): the bench's convention for f32 outputs.
pub const F32_TOL: f64 = 1e-5;
/// The tolerance of a binary16 output against its f64 reference, same
/// normalisation: the bench's convention for binary16 outputs (`banc-252`
/// sweep-64, "AWQ, cublasf16: 1e-3, binary16 output"). A binary16 value
/// carries 11 significant bits, so judging it at 1e-5 would void every
/// correct arm (review finding 3).
pub const F16_TOL: f64 = 1e-3;
/// The binary16 pattern a row is poisoned with before an identity launch: a
/// NaN no kernel writes.
pub const POISON_H: u16 = 0xFFFF;

/// Judge one row. `got_f32` is the value before `f2h`, from the debug buffer.
pub fn judge_row(served_h: u16, got_h: u16, got_f32: f32, f64_ref: f64, sum_abs: f64) -> RowVerdict {
    if got_h == POISON_H && served_h != POISON_H {
        return RowVerdict::Unwritten;
    }
    if got_h == served_h {
        return RowVerdict::Identical;
    }
    let norm = sum_abs.max(1.0);
    let f32_err = (got_f32 as f64 - f64_ref).abs() / norm;
    let h_err = (f16_to_f64(got_h) - f64_ref).abs() / norm;
    if f32_err <= F32_TOL && h_err <= F16_TOL {
        RowVerdict::Moved { f32_err, h_err }
    } else {
        RowVerdict::Defect { f32_err, h_err }
    }
}

/// binary16 bits to f64, exact.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The device's site record, 8 u32 a site (`PersSite` in `llvq_pers.cuh`).
pub const SITE_U32: usize = 8;

/// One site's record. `words` is the device address of its row-strided word
/// stream.
pub fn site_record(site: &Site, words: u64, row0_in_y: u32, stride_u32: u32) -> [u32; SITE_U32] {
    let nblocks = site.d_in / 24;
    let tail_w = site.d_in % 24;
    let wait = match site.wait {
        Wait::None => 0,
        Wait::All(_) => 1,
        Wait::Count(p) => 2 | (p << 2),
    };
    let prol = match site.prologue {
        Prologue::None => 0u32,
        Prologue::NormRot => 1,
        Prologue::Rot => 2,
    };
    [
        words as u32,
        (words >> 32) as u32,
        site.d_out,
        nblocks | (tail_w << 16),
        row0_in_y,
        stride_u32,
        wait,
        u32::from(site.signal_after) | (prol << 1) | (site.layer << 8),
    ]
}

/// Validator: refuses a schedule that could deadlock or read stale data.
///
/// (a) barrier ordinals are 0, 1, 2, ... in program order (every CTA runs the
///     same site list, so equal ordinals mean the same instance);
/// (b) a counter wait names an earlier site, and that site signals;
/// (c) every signalled site has a consumer (a signal nobody waits for would be
///     timed as part of h);
/// (d) a prologue sits only where the staged width changes, since the kernel
///     runs it on a restage only;
/// (e) a site whose input is another site's output waits for it: the o rows
///     read what attention made of q/k/v, gate/up read the residual that o
///     wrote, and down reads what gate/up wrote. In the bench every input is
///     the same fixed `x`, so the rule checks the SHAPE of the served schedule,
///     which route P inherits.
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
        if site.prologue != Prologue::None && idx > 0 && s[idx - 1].d_in == site.d_in && s[idx - 1].layer == site.layer {
            return Err(format!("site {idx}: a prologue where the staged width does not change"));
        }
        let needs_wait = match site.proj {
            Proj::O | Proj::Down | Proj::Gate => true,
            Proj::Q => site.layer > 0,
            Proj::K | Proj::V | Proj::Up => false,
        };
        if needs_wait && site.wait == Wait::None {
            return Err(format!("site {idx} ({:?} of layer {}): reads a vector other CTAs write, and waits for nothing", site.proj, site.layer));
        }
    }
    Ok(())
}

/// Which projections of a served model are int4 records, everything else
/// being Tetra. Read from the served configs (`configs/*.json`).
#[derive(Clone, Copy, Debug)]
pub struct ModelShape {
    pub layers: u32,
    pub hidden: u32,
    pub inter: u32,
    pub v_int4: bool,
    pub o_int4: bool,
    /// Layers `[lo, hi)` whose down_proj is int4.
    pub down_int4: (u32, u32),
}

impl ModelShape {
    pub const Q4B: ModelShape = ModelShape { layers: 36, hidden: 2560, inter: 9728, v_int4: true, o_int4: true, down_int4: (12, 24) };
    pub const Q8B: ModelShape = ModelShape { layers: 36, hidden: 4096, inter: 12288, v_int4: true, o_int4: false, down_int4: (10, 27) };
    pub const Q14B: ModelShape = ModelShape { layers: 40, hidden: 5120, inter: 17408, v_int4: true, o_int4: true, down_int4: (10, 29) };

    pub fn tetra_down(&self, l: u32) -> bool {
        !(self.down_int4.0..self.down_int4.1).contains(&l)
    }
}

/// Grid-wide waits per token of 4E (`persistent.md` §3.9): W1 before every
/// layer's q/k/v but the first, plus the final norm's; W2 before every
/// gate/up; W_o before a Tetra o (its rotation reads the whole attention
/// output); W3 before a Tetra down; W4 for the argmax combine. An int4 o or
/// down waits on finer counters, not on a grid barrier.
pub fn waits_4e(m: &ModelShape) -> u32 {
    let w1 = m.layers; // L - 1 layer entries + the final norm
    let w2 = m.layers;
    let wo = if m.o_int4 { 0 } else { m.layers };
    let w3 = (0..m.layers).filter(|&l| m.tetra_down(l)).count() as u32;
    w1 + w2 + wo + w3 + 1
}

/// One phase of 4E's token, in launch order (`persistent.md` §3.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase4E {
    /// Every CTA dequantizes the token's embedding row (`emb_q4_gather`'s
    /// arithmetic), no wait: the row is a function of the token only.
    Embed,
    /// ln1, the q/k rotation, the q and k rows, the v rows (int4, unrotated).
    Qkv(u32),
    /// q/k-norm, RoPE, the KV append, attention: our split-KV kernel, under
    /// a T8 the operator has not granted. Waits on q/k/v by counters.
    Attention(u32),
    /// The o rows and the residual. A Tetra o rotates the whole attention
    /// output first (W_o); an int4 o waits on the attention counters.
    O(u32),
    /// ln2 to the residual after down: route P step A's segment body.
    Mlp(u32),
    /// The final norm, then the head rows and the argmax key, then W4.
    Head,
}

/// What a 4E phase waits for before it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait4E {
    None,
    /// A grid barrier: the phase reads a whole vector other CTAs wrote.
    Grid,
    /// Counters with fewer contributors (q/k/v to attention to an int4 o).
    Counter,
}

/// 4E's phases and their waits for one token. The grid barriers inside the
/// phases (W3 in an MLP segment with a Tetra down, W4 after the head) are
/// counted by [`grid_waits_4e`] beside the entry waits listed here.
pub fn schedule_4e(m: &ModelShape) -> Vec<(Phase4E, Wait4E)> {
    let mut v = vec![(Phase4E::Embed, Wait4E::None)];
    for l in 0..m.layers {
        // W1: ln1 reads the whole residual; layer 0 reads the embedding,
        // which every CTA computed itself.
        v.push((Phase4E::Qkv(l), if l == 0 { Wait4E::None } else { Wait4E::Grid }));
        v.push((Phase4E::Attention(l), Wait4E::Counter));
        v.push((Phase4E::O(l), if m.o_int4 { Wait4E::Counter } else { Wait4E::Grid }));
        // W2: ln2 reads the whole residual after o
        v.push((Phase4E::Mlp(l), Wait4E::Grid));
    }
    // the final W1: the final norm reads the whole residual
    v.push((Phase4E::Head, Wait4E::Grid));
    v
}

/// Grid-wide waits of one 4E token: the entry barriers of [`schedule_4e`],
/// plus W3 inside every MLP phase with a Tetra down, plus W4 after the head.
pub fn grid_waits_4e(m: &ModelShape) -> u32 {
    let entry = schedule_4e(m).iter().filter(|(_, w)| *w == Wait4E::Grid).count() as u32;
    let w3 = (0..m.layers).filter(|&l| m.tetra_down(l)).count() as u32;
    entry + w3 + 1
}

/// Grid barriers one token of `pers_4e.cu` passes: the embedding, six a
/// layer (q/k/v, norm/RoPE/append, after attention, o, the MLP's own, after
/// the MLP), the head's W4. Attention has none inside since version 2 (its
/// split fold is a per-KV-head ticket), so the count does not depend on the
/// position. The host advances `bar_base` by this times the grid for every
/// token the launch produced.
pub fn n_bar_4e(layers: u32) -> u32 {
    2 + 6 * layers
}

/// Shared bytes of a 4E CTA: the header, the decoder tables, then one region
/// that is the f32 vector of the row phases (`max(inter, 2 x hidden)` floats)
/// or the attention phase's stage, unit states and q (`PERS_ATTN_SMEM` of
/// llvq_pers.cuh), whichever is larger, rounded to 16 bytes.
/// 4E's row forms (`PERS_4E_ROWS` in `llvq_pers.cuh`): 1, the stream (4E v3,
/// what the served config runs); 2, v2's row loop on the same static rows
/// (4E v4); 3, form 2 with each CTA's last rows split over every warp (4E v5);
/// 4, form 2 with ln1, ln2 and the down's input in one round trip a thread
/// (4E v6, `llvq_pers.cuh` section 7b); 5, form 4 with the launch's
/// constants (argument structs, row slices, sign words, `small`) copied into
/// shared memory during the barrier before they are read, and the o phase's
/// input widened from vectors (4E v7, section 7c); 6, form 5 with the
/// attention phase's steps staged by `cp.async` into a ring of
/// [`p4e_attn_ring`] stages, several steps in flight (4E v8, K3a of
/// `docs/plan-300.md`).
pub const P4E_ROWS_FORMS: [(u32, &str); 7] = [(1, "v3"), (2, "v4"), (3, "v5"), (4, "v6"), (5, "v7"), (6, "v8"), (7, "v9")];

/// The forms `fusedrun LLVQ_PERS_4E` times, the first the reference of the
/// others' round-by-round ratios: v4, the served form since 2026-11-05; v6,
/// served on the operator's go of 2026-11-07; v7 (K2) and v8 (K2 and K3a),
/// step 2 of `docs/plan-300.md`; v9 (v8 with the int4 rows on tiled words,
/// K11), step 3. v3 and v5 were read by the shaker (job 6aec9a1d).
pub const P4E_BENCH_FORMS: [(u32, &str); 5] = [(2, "v4"), (4, "v6"), (5, "v7"), (6, "v8"), (7, "v9")];

/// The line a module of row form `form` starts with; form 1 is the
/// default and adds nothing, so its module keeps its text.
pub fn p4e_rows_define(form: u32) -> Result<String, String> {
    match form {
        1 => Ok(String::new()),
        2..=7 => Ok(format!("#define PERS_4E_ROWS {form}u\n")),
        _ => Err(format!("4E row form {form}: the forms are 1 (v3), 2 (v4), 3 (v5), 4 (v6), 5 (v7), 6 (v8) and 7 (v9)")),
    }
}

/// Form 4's conditions on the card (`pers_1rt_ok` in `llvq_pers.cuh`): every
/// input of a vector prologue 16-byte aligned, `hidden` and `inter` multiples
/// of 8. Unmet, the kernel takes the scalar path and would time as v4 under
/// v6's name, so the install refuses it, naming the first input that fails.
pub fn p4e_v6_inputs(addrs: &[(String, u64)], hidden: u32, inter: u32) -> Result<(), String> {
    if !hidden.is_multiple_of(8) || !inter.is_multiple_of(8) {
        return Err(format!("4E v6: hidden {hidden} and inter {inter} must be multiples of 8 for the vector prologues"));
    }
    match addrs.iter().find(|(_, a)| !a.is_multiple_of(16)) {
        Some((name, a)) => Err(format!("4E v6: {name} at {a:#x} is not 16-byte aligned; the vector prologues would not run")),
        None => Ok(()),
    }
}

/// Where word `wi` of an int4 row of `nwords` words sits in 4E v9's tiled
/// layout: `pers_q4_tile_index` of `llvq_pers.cuh`, which the device relayout
/// uses, transcribed. Inside a full tile of 128 words, word `128 t + k` goes
/// to `128 t + 4 (k mod 32) + k / 32`, so one 128-bit load gives lane l its
/// words `128 t + l + 32 s`, s = 0..3; the words past the last full tile keep
/// the served order. `tests/pers_matches_served.rs` compares it with the
/// device function on every word of the served widths.
pub fn q4_tile_index(wi: u32, nwords: u32) -> u32 {
    let nt = nwords >> 7;
    if wi >= nt << 7 {
        return wi;
    }
    let (t, k) = (wi >> 7, wi & 127);
    (t << 7) + 4 * (k & 31) + (k >> 5)
}

/// An int4 record's words, `d_out` rows of `nwords`, in the tiled layout:
/// the loader's form of `pers_4e_q4_tile_relayout`, run on the host before
/// the upload so the card never holds the served order beside it. A
/// permutation inside each row: the same words, the same count. Refuses a
/// width that is not a whole number of 16-byte loads.
pub fn q4_tile_words(words: &[u32], d_out: usize, nwords: usize) -> Result<Vec<u32>, String> {
    if !nwords.is_multiple_of(4) {
        return Err(format!("int4 tiles: rows of {nwords} words are not a whole number of 16-byte loads"));
    }
    if words.len() != d_out * nwords {
        return Err(format!("int4 tiles: {} words for {d_out} rows of {nwords}", words.len()));
    }
    let mut out = vec![0u32; words.len()];
    for (src, dst) in words.chunks_exact(nwords).zip(out.chunks_exact_mut(nwords)) {
        for (wi, &w) in src.iter().enumerate() {
            dst[q4_tile_index(wi as u32, nwords as u32) as usize] = w;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Experiment E1 (docs/mesures/large-4e-budget-2026-11-11.txt, cut 1): where
// 4E v9 asks L2 for later rows. Three bench modules, each 4E v9's text with
// edits applied as text, so the served v9 module (pinned below,
// `the_4e_v9_module_is_the_one_the_doors_gate_on`) does not move:
//   "v9"   v9 and a logits symbol (the reference the arms are compared with);
//   "v9np" "v9" without the attention phase's prefetch of the o and gate/up
//          rows: nothing else;
//   "v9mv" "v9np" with layer l + 1's q/k/v and o rows asked of L2 during
//          layer l's gate/up rows (after ln2), layer 0's before its ln1.
//   "v9ft" "v9" with the attention tile's full-tile path ([`pers_ft_edit`],
//          docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt).
//   "v9ep" "v9" with every Tetra row phase on rowbench's `endpass` body
//          ([`p4e_rc_edits`], kernels/pers_4e_rc.cuh): the bits.
//   "v9fl" "v9" with every Tetra row phase on rowbench's `flat` body: a
//          bound of 1e-5 on rows, NOT the bits ([`p4e_e1_owes_bits`]).
// ---------------------------------------------------------------------------

/// The E1 arms, the first the reference of the others.
pub const P4E_E1_ARMS: [&str; 10] = ["v9", "v9np", "v9mv", "v9ft", "v9ep", "v9fl", "v9ep2", "v9ep3", "v9fl2", "v9fl3"];

/// The research bodies' arms and their body codes (`PERS_RC_ARM`): 1 endpass,
/// 2 flat (pers_4e_rc.cuh); 3 ep2, 4 ep3, 5 fl2, 6 fl3 (pers_4e_rc2.cuh, the
/// lighter variants after the first card run refused 1 and 2 for 8 B of
/// local memory, docs/mesures/recherche-debit-jeton-2026-11-13.txt).
pub const P4E_RC_ARMS: [(&str, u32); 6] = [("v9ep", 1), ("v9fl", 2), ("v9ep2", 3), ("v9ep3", 4), ("v9fl2", 5), ("v9fl3", 6)];

/// The body code of a research arm, or none.
pub fn p4e_rc_body(arm: &str) -> Option<u32> {
    P4E_RC_ARMS.iter().find(|a| a.0 == arm).map(|a| a.1)
}

/// Whether a body is flat's family (owes the bound, not the bits).
pub fn rc_body_is_flat(body: u32) -> bool {
    matches!(body, 2 | 5 | 6)
}

/// Whether a body runs the CTA's slice in one chunk of slots (it traps
/// otherwise): ep3, fl2, fl3.
pub fn rc_body_one_chunk(body: u32) -> bool {
    matches!(body, 4..=6)
}

/// Whether an E1 arm owes v9's bits (its tokens and logits equal v9's). Every
/// arm does but flat's (`v9fl`, `v9fl2`, `v9fl3`), whose rows owe a bound of
/// 1e-5 (pers_4e_rc.cuh): their tokens and logits are compared with v9's and
/// reported, never assumed equal.
pub fn p4e_e1_owes_bits(arm: &str) -> bool {
    !p4e_rc_body(arm).is_some_and(rc_body_is_flat)
}

/// The research phase's row bodies as 4E row forms (step R6 of
/// docs/CERVEAU.md O11): inserted into llvq_pers.cuh by [`p4e_rc_edits`].
pub const PERS_RC_PART: (&str, &str) = ("pers_4e_rc.cuh", include_str!("../kernels/pers_4e_rc.cuh"));

/// The lighter variants (bodies 3 to 6), inserted after [`PERS_RC_PART`].
pub const PERS_RC2_PART: (&str, &str) = ("pers_4e_rc2.cuh", include_str!("../kernels/pers_4e_rc2.cuh"));

/// The f32 slots of a body (`PERS_RC_SLOTS` of pers_4e_rc.cuh).
pub const PERS_RC_SLOTS: u32 = 2048;

/// `pers_rc_padded_floats`: flat's padded staging of a `d_in` vector (block b
/// at 24 b + b / 8, the tail after), rounded to 16 bytes.
pub fn rc_padded_floats(d_in: u32) -> u32 {
    let (nb, tw) = (d_in / 24, d_in % 24);
    (24 * nb + nb / 8 + tw + 3) & !3
}

/// The edits that put a 4E v9 module's Tetra row phases on a research body:
/// `body` 1 endpass, 2 flat. The body's text and its define before `struct
/// Pers4eQkv`; then the four call sites (q/k, gate/up, the Tetra down, the
/// 8B's Tetra o), each from v9's warp lists (pers_4e_rows) to the CTA's whole
/// slice (pers_rc_rows). Every phase, its free region (pers_4e_rc.cuh, header):
///   q/k     after xr and xn, v + 2 hidden; flat's padded copy there, the slots after;
///   gate/up the same, then the gate and up values (binary16) of the CTA's pairs;
///   down, o in place: flat restages the activation where it lies, the slots after it.
/// The gate/up list is the CTA's gate rows THEN its up rows (rowbench's gate/up
/// form), where v9 interleaves them a warp at a time (gate r, up r) and folds
/// each product under the next pair; here the products follow a CTA barrier.
///
/// Bodies 3 to 6 (pers_4e_rc2.cuh): the same call sites, naming
/// `pers_rc2_rows` and `pers_rc2_work_floats`, and both texts inserted. The
/// texts of bodies 1 and 2 are unchanged by them (their modules keep their
/// sha256, `the_rc_modules_keep_their_bytes`).
pub fn p4e_rc_edits(body: u32) -> Result<Vec<(&'static str, String, String)>, String> {
    if !(1..=6).contains(&body) {
        return Err(format!("4E rc: body {body}, the bodies are 1 (endpass), 2 (flat), 3 to 6 (pers_4e_rc2.cuh)"));
    }
    let v = p4e_rc_edits_12(body)?;
    if body <= 2 {
        return Ok(v);
    }
    Ok(v.into_iter()
        .map(|(f, from, to)| {
            let to = to
                .replace(PERS_RC_PART.1, &format!("{}\n{}", PERS_RC_PART.1, PERS_RC2_PART.1))
                .replace("pers_rc_rows<PERS_RC_ARM>(", "pers_rc2_rows<PERS_RC_ARM>(")
                .replace("pers_rc_work_floats<PERS_RC_ARM>(", "pers_rc2_work_floats<PERS_RC_ARM>(");
            (f, from, to)
        })
        .collect())
}

/// The edits of bodies 1 and 2, and the template of the others ([`p4e_rc_edits`]).
fn p4e_rc_edits_12(body: u32) -> Result<Vec<(&'static str, String, String)>, String> {
    let (cuh, seg, e4) = ("llvq_pers.cuh", "pers_segment.cu", "pers_4e.cu");
    Ok(vec![
        (cuh, "struct Pers4eQkv {\n".into(), format!("#define PERS_RC_ARM {body}u\n{}\nstruct Pers4eQkv {{\n", PERS_RC_PART.1)),
        (
            cuh,
            "        (void)scratch;
        const auto row_of = [&](u32 k) { return qk_row(lo + warp + k * nw); };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) { qk_out(lo + warp + k * nw, y); };
        pers_4e_rows(pers_warp_rows(lo, thi, warp, nw), P.nblocks, P.tail_w, tab, invnorm, xr, lane, row_of, start,
                     epi);
"
            .into(),
            "        (void)scratch;  // rc: the CTA's q | k rows as one list; the free region after xr and xn
        const auto row_at = [&](u32 idx) { return qk_row(lo + idx); };
        const auto epi = [&](u32 idx, float y) { qk_out(lo + idx, y); };
        pers_rc_rows<PERS_RC_ARM>(thi - lo, P.nblocks, P.tail_w, tab, invnorm, const_cast<float*>(xr), row_at, epi,
                                  const_cast<float*>(xr) + 2u * P.d_in, false);
"
            .into(),
        ),
        (
            seg,
            "        const u32 warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u);
        const u32 npairs = pers_warp_rows(glo, ghi, warp, nw);
        unsigned short g = 0u, pend_sv = 0u, pend_u = 0u;
        u32 pend_r = 0xffffffffu;
        const auto row_of = [&](u32 k) {
            const u32 r = glo + warp + (k >> 1) * nw;
            const SegTetra& M = (k & 1u) ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) {
            if ((k & 1u) == 0u) {
                g = f2h(y);
                return;
            }
            if (pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
            pend_sv = A->silu[g];
            pend_u = f2h(y);
            pend_r = glo + warp + (k >> 1) * nw;
        };
        pers_4e_rows(2u * npairs, A->gate.nblocks, A->gate.tail_w, tab, t_inv, v, lane, row_of, start, epi);
        if (lane == 0u && pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
"
            .into(),
            "        // rc: the CTA's gate rows, then its up rows, as one list; their binary16 values after the body's
        // region, the products after the body's last barrier (exact_hmul(silu[g], u), the stream's own product)
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u), np = ghi - glo;
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        float* const work = v + 2u * H;
        float* const gu_vals = work + pers_rc_work_floats<PERS_RC_ARM>(nb, tw, false);
        pers_rc_guard(gu_vals + np);
        unsigned short* const sgu = reinterpret_cast<unsigned short*>(gu_vals);
        const auto row_at = [&](u32 idx) {
            const bool up = idx >= np;
            const u32 r = glo + (up ? idx - np : idx);
            const SegTetra& M = up ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto epi = [&](u32 idx, float y) { sgu[idx] = f2h(y); };
        pers_rc_rows<PERS_RC_ARM>(2u * np, nb, tw, tab, t_inv, v, row_at, epi, work, false);
        for (u32 p = threadIdx.x; p < np; p += blockDim.x) A->act[glo + p] = exact_hmul(A->silu[sgu[p]], sgu[np + p]);
"
            .into(),
        ),
        (
            seg,
            "            unsigned short xr = 0u;
            const auto row_of = [&](u32 k) {
                const u32 r = dlo + warp + k * nw;
                return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
            };
            const auto start = [&](u32 k) { xr = __ldcg(L.x + dlo + warp + k * nw); };
            const auto epi = [&](u32 k, float y) { L.out[dlo + warp + k * nw] = exact_hadd(xr, f2h(y)); };
            pers_4e_rows(pers_warp_rows(dlo, dhi, warp, nw), A->down.nblocks, A->down.tail_w, tab, t_inv, v, lane,
                         row_of, start, epi);
"
            .into(),
            "            // rc: the CTA's down rows as one list; flat restages the activation in place, the slots after it
            const u32 nb = A->down.nblocks, tw = A->down.tail_w;
            const auto row_at = [&](u32 idx) {
                const u32 r = dlo + idx;
                return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { L.out[dlo + idx] = exact_hadd(__ldcg(L.x + dlo + idx), f2h(y)); };
            pers_rc_rows<PERS_RC_ARM>(dhi - dlo, nb, tw, tab, t_inv, v, row_at, epi, v + pers_rc_padded_floats(nb, tw), true);
"
            .into(),
        ),
        (
            e4,
            "    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned short xr = 0u;
    const auto row_of = [&](u32 k) {
        const u32 r = lo + warp + k * nw;
        return PersTRow{O.t.words + r * O.t.stride_u32, O.t.gscale, O.t.rscale, O.t.tail, r};
    };
    const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw); };
    const auto epi = [&](u32 k, float y) { out[lo + warp + k * nw] = exact_hadd(xr, f2h(y)); };
    pers_4e_rows(pers_warp_rows(lo, hi, warp, nw), O.t.nblocks, O.t.tail_w, tab, invnorm, cs, lane, row_of, start, epi);
"
            .into(),
            "    // rc: the CTA's o rows as one list; flat restages the context in place, the slots after it
    const u32 nb = O.t.nblocks, tw = O.t.tail_w;
    const auto row_at = [&](u32 idx) {
        const u32 r = lo + idx;
        return PersTRow{O.t.words + r * O.t.stride_u32, O.t.gscale, O.t.rscale, O.t.tail, r};
    };
    const auto epi = [&](u32 idx, float y) { out[lo + idx] = exact_hadd(__ldcg(x + lo + idx), f2h(y)); };
    pers_rc_rows<PERS_RC_ARM>(hi - lo, nb, tw, tab, invnorm, cs, row_at, epi, cs + pers_rc_padded_floats(nb, tw), true);
"
            .into(),
        ),
    ])
}

/// One phase of a research body in 4E's buffer: its name, the last float it
/// uses (from 4E's row vector `v`), and the v7 area's offset it must not pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RcPhase {
    pub phase: &'static str,
    pub end: u32,
    pub limit: u32,
}

/// Where `body` (1 endpass, 2 flat) puts its region at every Tetra phase of a
/// 4E v9 shape, as pers_4e_rc.cuh and the edits place it: `hidden`, `inter`,
/// `d_ctx` (the attention output's width; `tetra_o`: an o_proj in Tetra, the
/// 8B's), `groups` and the grid (the CTA's gate/up pairs at most). Refuses, by
/// phase and numbers, a shape where a phase would pass the v7 area: no byte is
/// added to 4E's launch, so a shape that does not fit is not run.
#[allow(clippy::too_many_arguments)]
pub fn p4e_rc_layout(body: u32, hidden: u32, inter: u32, d_ctx: u32, tetra_o: bool, tetra_down: bool, groups: u32, grid: u32) -> Result<Vec<RcPhase>, String> {
    if !(1..=6).contains(&body) || grid == 0 || groups == 0 {
        return Err(format!("4E rc: body {body}, grid {grid}, {groups} groups"));
    }
    let flat = rc_body_is_flat(body);
    let attn = 16 * 1024 * p4e_attn_ring(7) + 4 * 4 * groups * 130 + 4 * groups * 128;
    let limit = inter.max(2 * hidden).max(attn / 4);
    let work = |d_in: u32, in_place: bool| if flat && !in_place { rc_padded_floats(d_in) } else { 0 } + PERS_RC_SLOTS;
    let pairs = inter.div_ceil(grid);
    let mut v = vec![
        RcPhase { phase: "q/k", end: 2 * hidden + work(hidden, false), limit },
        RcPhase { phase: "gate/up", end: 2 * hidden + work(hidden, false) + pairs, limit },
    ];
    if tetra_down {
        v.push(RcPhase { phase: "down", end: rc_padded_floats(inter) + work(inter, true), limit });
    }
    if tetra_o {
        v.push(RcPhase { phase: "o", end: rc_padded_floats(d_ctx) + work(d_ctx, true), limit });
    }
    let name = ["", "endpass", "flat", "ep2", "ep3", "fl2", "fl3"][body as usize];
    if let Some(p) = v.iter().find(|p| p.end > p.limit) {
        return Err(format!(
            "4E rc {name}: the {} phase needs floats up to {} of 4E's row vector, past the v7 area at {} (hidden {hidden}, \
             inter {inter}, d_ctx {d_ctx}, {groups} groups): refused, 4E's launch gets no byte more",
            p.phase, p.end, p.limit
        ));
    }
    if flat && inter.max(hidden).max(d_ctx).div_ceil(24).div_ceil(8) > PERS_RC_SLOTS {
        return Err(format!("4E rc {name}: a row of more than {PERS_RC_SLOTS} runs"));
    }
    // One chunk (ep3, fl2, fl3): the CTA's largest slice of every phase, its rows (endpass) or rows x runs (flat),
    // within the slots; the body traps otherwise.
    if rc_body_one_chunk(body) {
        let runs = |d_in: u32| if flat { (d_in / 24).div_ceil(8) } else { 1 };
        let mut need = vec![("q/k", (d_ctx + d_ctx / groups).div_ceil(grid) * runs(hidden)), ("gate/up", 2 * pairs * runs(hidden))];
        if tetra_down {
            need.push(("down", hidden.div_ceil(grid) * runs(inter)));
        }
        if tetra_o {
            need.push(("o", hidden.div_ceil(grid) * runs(d_ctx)));
        }
        if let Some((ph, n)) = need.iter().find(|p| p.1 > PERS_RC_SLOTS) {
            return Err(format!(
                "4E rc {name}: one chunk, and the {ph} phase's slice needs {n} slots of {PERS_RC_SLOTS} on {grid} CTAs: refused"
            ));
        }
    }
    Ok(v)
}

/// The mv arm's helpers (`pers_4e_e1.cuh`), inserted at the head of
/// `pers_segment.cu`'s text in that arm's module only.
pub const P4E_E1_PRE: (&str, &str) = ("pers_4e_e1.cuh", include_str!("../kernels/pers_4e_e1.cuh"));

/// The logits symbols every E1 module carries: `pers_4e_body` with a
/// fourth template parameter, false by default (the token and stamp symbols
/// keep their code), true in these symbols, which write the head's binary16
/// logits of token t at `stamps + t x vocab` (the launch's stamp pointer,
/// read as a buffer of `n_tokens x vocab` binary16).
const P4E_E1_LOGITS_SYMBOLS: &str = "// e1-logits: the head's binary16 logits of every token of the launch, written at
// L.stamps + t x vocab (a measurement symbol, outside the timed rounds)
#define PERS_4E_LOGITS(sym, FORM_)                                                                       \\
    extern \"C\" __global__ void __launch_bounds__(512, 1) sym(const Pers4eArgs* __restrict__ A,          \\
                                                             const Pers4eLaunch L)                        \\
    {                                                                                                    \\
        pers_4e_body<FORM_, false, true, true>(A, L);                                                    \\
    }

PERS_4E_LOGITS(pers_4e_logits_f0_t512, 0)
PERS_4E_LOGITS(pers_4e_logits_f1_t512, 1)
PERS_4E_LOGITS(pers_4e_logits_f2_t512, 2)
";

/// The edits of an E1 arm: (part of [`PERS_UNIT_PARTS`], text, replacement),
/// each text present exactly once in its part. Every arm carries the logits
/// edits; np and mv add theirs. `tests/pers_matches_served.rs` builds the
/// host twin of each arm from the same edits.
pub fn p4e_e1_edits(arm: &str) -> Result<Vec<(&'static str, String, String)>, String> {
    let e4 = "pers_4e.cu";
    let seg = "pers_segment.cu";
    let mut v: Vec<(&'static str, String, String)> = vec![
        (
            e4,
            "template <int FORM, bool STAMP, bool PF>\n__device__ __forceinline__ void pers_4e_body(".into(),
            "template <int FORM, bool STAMP, bool PF, bool LOGITS = false>\n__device__ __forceinline__ void pers_4e_body(".into(),
        ),
        (
            e4,
            "PERS_BAR(), A->abort_flag, nullptr);".into(),
            "PERS_BAR(), A->abort_flag,\n                                          \
             LOGITS ? reinterpret_cast<unsigned short*>(L.stamps) + (size_t)t * A->vocab : nullptr);"
                .into(),
        ),
        (
            e4,
            "PERS_4E(pers_4e_stamps_f2_t768, 2, 768, true, false)\n".into(),
            format!("PERS_4E(pers_4e_stamps_f2_t768, 2, 768, true, false)\n\n{P4E_E1_LOGITS_SYMBOLS}"),
        ),
    ];
    let np = [
        (
            e4,
            "            if (c0 >= attn_jobs) prefetch_next();\n".to_string(),
            "            (void)attn_jobs;  // e1-pf: no o or gate/up row is asked of L2 in this phase\n            \
             (void)prefetch_next;  // e1-pf\n"
                .to_string(),
        ),
        (e4, "            if (c0 < attn_jobs) prefetch_next();\n".to_string(), String::new()),
    ];
    match arm {
        "v9" => {}
        // the attention tile's full-tile path (pers_ft_edit), nothing else
        "v9ft" => v.push(pers_ft_edit()),
        // the research bodies on every Tetra row phase (pers_4e_rc.cuh), nothing else
        "v9ep" | "v9fl" | "v9ep2" | "v9ep3" | "v9fl2" | "v9fl3" => {
            v.extend(p4e_rc_edits(p4e_rc_body(arm).expect("a research arm has a body"))?)
        }
        "v9np" => v.extend(np),
        "v9mv" => {
            v.extend(np);
            v.extend([
                (seg, "struct SegTetra {\n".to_string(), format!("{}\nstruct SegTetra {{\n", P4E_E1_PRE.1)),
                (
                    seg,
                    "    PERS_SEG_SUB(0u);\n".to_string(),
                    "    PERS_SEG_SUB(0u);\n    \
                     if (IN4E) pers_e1_prefetch_staged();  // e1-pf: layer l + 1's q/k/v and o rows, during these gate/up rows\n"
                        .to_string(),
                ),
                (
                    e4,
                    "        pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);\n    };\n".to_string(),
                    "        pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);\n        \
                     pers_e1_stage_next(Ag->layers, l + 1u, Ag->n_layers, Ag->hidden, Ag->n_heads);  // e1-pf: layer l + 1's shares\n    };\n"
                        .to_string(),
                ),
                (
                    e4,
                    "            if (PF) {\n                const Pers4eQkv& N = Y.qkv;\n".to_string(),
                    "            if (l == 0u) {  // e1-pf: layer 0's q/k/v here, every later layer's during the gate/up rows before it\n            \
                     if (PF) {\n                const Pers4eQkv& N = Y.qkv;\n"
                        .to_string(),
                ),
                (
                    e4,
                    "            }\n            // ln1, the q/k rotation, the q/k/v rows\n".to_string(),
                    "            }\n            pers_e1_prefetch_o(Y, H, A->n_heads);  // e1-pf: and layer 0's o rows\n            \
                     }  // e1-pf\n            // ln1, the q/k rotation, the q/k/v rows\n"
                        .to_string(),
                ),
            ]);
        }
        _ => return Err(format!("4E E1: no arm {arm:?}; the arms are {P4E_E1_ARMS:?}")),
    }
    Ok(v)
}

/// The module text of an E1 arm at a tile: row form 7's define, then
/// [`PERS_UNIT_PARTS`] with the arm's edits ([`p4e_e1_edits`]), joined as
/// [`pers_unit_text`] joins them. Refuses an edit whose text is not in its
/// part exactly once.
pub fn p4e_e1_unit_text(tile_define: &str, arm: &str) -> Result<String, String> {
    let mut parts: Vec<(&str, String)> = PERS_UNIT_PARTS.iter().map(|p| (p.0, p.1.to_string())).collect();
    for (file, from, to) in p4e_e1_edits(arm)? {
        let part = parts.iter_mut().find(|p| p.0 == file).ok_or_else(|| format!("4E E1 {arm}: no part {file}"))?;
        let n = part.1.matches(from.as_str()).count();
        if n != 1 {
            return Err(format!("4E E1 {arm}: {n} sites in {file} for {from:?}, expected exactly one"));
        }
        part.1 = part.1.replacen(from.as_str(), &to, 1);
    }
    let body = std::iter::once(tile_define.to_string()).chain(parts.into_iter().map(|p| p.1)).collect::<Vec<_>>().join("\n");
    Ok(format!("{}{body}", p4e_rows_define(7)?))
}

/// The logits symbol of an E1 module for a RoPE form (512 threads).
pub fn p4e_logits_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4e_logits_f0_t512",
        1 => "pers_4e_logits_f1_t512",
        2 => "pers_4e_logits_f2_t512",
        _ => return Err(format!("4E E1: no logits symbol for RoPE form {form}")),
    })
}

/// A line E1 may add, drop or change: a prefetch statement or the code that
/// steers one. It names a prefetch, or carries the `e1-pf` tag; and it
/// assigns nothing (no ` = ` outside a declaration of a prefetch helper).
fn p4e_e1_prefetch_line(l: &str) -> bool {
    let t = l.trim();
    (t.contains("prefetch") || t.contains("e1-pf"))
        && (!t.split("//").next().unwrap_or("").contains(" = ") || t.starts_with("const auto prefetch_next = "))
}

/// The structural check of an E1 arm against its reference (`"v9"`'s text):
/// with every prefetch line ([`p4e_e1_prefetch_line`]) and the mv helpers'
/// block (`// e1-pf begin` .. `// e1-pf end`) taken out of both, the two
/// texts are the same lines in the same order. The block itself may hold no
/// kernel, no atomic, no barrier, no grid wait, and may store only to its
/// header slots (`s[i] = ...`).
pub fn p4e_e1_structure(reference: &str, arm: &str) -> Result<(), String> {
    fn strip(t: &str) -> Result<Vec<&str>, String> {
        let mut out = Vec::new();
        let mut inside = false;
        for l in t.lines() {
            let s = l.trim();
            if s == "// e1-pf begin" {
                inside = true;
                continue;
            }
            if s == "// e1-pf end" {
                inside = false;
                continue;
            }
            if inside {
                for bad in ["__global__", "atomic", "__syncthreads", "pers_grid_wait", "PERS_BAR", "A->", "L."] {
                    if s.contains(bad) {
                        return Err(format!("E1 helpers: {bad:?} in {s:?}"));
                    }
                }
                // every assignment's target: a local being declared, or a slot
                let code = s.split("//").next().unwrap_or("");
                for (i, _) in code.match_indices(" = ") {
                    let before = &code[..i];
                    let cut = before.rfind([';', ',', '{']).map_or(0, |k| k + 1);
                    let mut lhs = before[cut..].trim();
                    lhs = lhs.strip_prefix("for (").unwrap_or(lhs);
                    if let Some(k) = lhs.rfind(") ") {
                        lhs = lhs[k + 2..].trim();
                    }
                    if (lhs.contains('[') && !lhs.starts_with("s[")) || lhs.starts_with('*') || lhs.contains("->") || lhs.contains('.') {
                        return Err(format!("E1 helpers: a store that is not a slot: {s:?}"));
                    }
                }
                continue;
            }
            // comments and blank lines are not code
            if s.is_empty() || s.starts_with("//") {
                continue;
            }
            if !p4e_e1_prefetch_line(l) {
                out.push(l);
            }
        }
        if inside {
            return Err("E1 helpers: a block opened and never closed".into());
        }
        Ok(out)
    }
    let (a, b) = (strip(reference)?, strip(arm)?);
    if let Some(i) = (0..a.len().min(b.len())).find(|&i| a[i] != b[i]) {
        return Err(format!("E1: past the prefetch lines, line {i} differs: {:?} against {:?}", a[i], b[i]));
    }
    if a.len() != b.len() {
        return Err(format!("E1: past the prefetch lines, {} lines against {}", a.len(), b.len()));
    }
    Ok(())
}

/// The shared scratch after the activation staging (`PERS_4E_SCRATCH`):
/// form 3's split rows, form 4's staged rows of the rotations' `small`,
/// form 5's v7 area (`PERS_V7_FLOATS` floats: `small`, the argument structs,
/// the CTA's slices and up to 576 sign words).
pub fn p4e_scratch_bytes(form: u32) -> u32 {
    match form {
        3 => 16_384,
        4 => 4_096,
        5..=7 => 7_360,
        _ => 0,
    }
}

/// The attention phase's stages (`PERS_ATTN_RING` in `llvq_pers.cuh`): one
/// below form 6, staged through registers; three from form 6 on, staged by
/// `cp.async`, so three steps of 16 KB are in flight at once.
pub fn p4e_attn_ring(form: u32) -> u32 {
    if form >= 6 { 3 } else { 1 }
}

/// [`p4e_smem_bytes_form`] for the forms below 6.
pub fn p4e_smem_bytes(hidden: u32, inter: u32, groups: u32) -> u32 {
    p4e_smem_bytes_form(1, hidden, inter, groups)
}

/// Shared bytes of a 4E CTA of row form `form`, before the scratch.
pub fn p4e_smem_bytes_form(form: u32, hidden: u32, inter: u32, groups: u32) -> u32 {
    let vec = 4 * inter.max(2 * hidden);
    let attn = 16 * 1024 * p4e_attn_ring(form) + 4 * 4 * groups * 130 + 4 * groups * 128;
    (SMEM_HEADER + TABLE_BYTES + vec.max(attn)).div_ceil(16) * 16
}

/// After a 4E launch: the tokens it made (the ring up to its first unwritten
/// slot, at most `n`), and the barrier base of the next launch. A launch that
/// stopped on an end-of-sequence passed the barriers of the tokens it made
/// and no others.
pub fn p4e_after_launch(ring: &[u32], n: u32, bar_base: u32, layers: u32, grid: u32) -> (Vec<u32>, u32) {
    let made: Vec<u32> = ring.iter().copied().take(n as usize).take_while(|&t| t != u32::MAX).collect();
    let next = bar_base.wrapping_add((made.len() as u32).wrapping_mul(n_bar_4e(layers)).wrapping_mul(grid));
    (made, next)
}

/// Why a 4E launch (decode or verify) failed on the card: one of the two
/// flags the kernel raises and never lowers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P4eFault {
    /// A grid barrier passed `PERS_SPIN_LIMIT` and every CTA left.
    Abort,
    /// The head saw a NaN logit and every CTA left.
    Nan,
}

/// The device words a 4E launch leaves behind it and the host must keep in
/// step with: the barrier counter, the abort flag and the NaN flag.
pub trait P4eCounters {
    /// The abort flag and the NaN flag as the launch left them.
    fn flags(&mut self) -> Result<(u32, u32), String>;
    /// The barrier counter, the abort flag and the NaN flag back to 0.
    fn zero(&mut self) -> Result<(), String>;
}

/// After every 4E launch, before reading its tokens. A clean launch returns
/// `None`. A faulted one leaves the counter wherever its CTAs stopped and
/// both flags as they are, so every later launch would bail on the same
/// flag (the runtime answered 500 until a restart, audit 2026-11-08 §1.1):
/// here the device words go back to 0 and so does the host's barrier base,
/// and the next launch starts as the first one did.
pub fn p4e_settle(dev: &mut impl P4eCounters, bar_base: &mut u32) -> Result<Option<P4eFault>, String> {
    let (abort, nan) = dev.flags()?;
    let fault = if abort != 0 {
        P4eFault::Abort
    } else if nan != 0 {
        P4eFault::Nan
    } else {
        return Ok(None);
    };
    p4e_reset(dev, bar_base)?;
    Ok(Some(fault))
}

/// The device words and the host's barrier base back to 0: after a fault,
/// or after any launch whose barrier count the host cannot account for.
pub fn p4e_reset(dev: &mut impl P4eCounters, bar_base: &mut u32) -> Result<(), String> {
    dev.zero()?;
    *bar_base = 0;
    Ok(())
}

/// One route-P step-A launch: the MLP segment of one layer, from `ln2` to the
/// residual add after down_proj. Its grid-wide waits inside the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentPlan {
    /// Grid barriers inside the launch: one before the down rows (the
    /// rotation of a Tetra down, or the int4 down's full input), and one at
    /// the end so no CTA writes the next launch's input while another reads.
    pub barriers: u32,
    pub tetra_down: bool,
}

/// Step A's launches for a model: one segment per layer.
pub fn step_a_plan(m: &ModelShape) -> Vec<SegmentPlan> {
    (0..m.layers).map(|l| SegmentPlan { barriers: 1, tetra_down: m.tetra_down(l) }).collect()
}

/// Median and range of a series: the report's one statistic (rule 7: a
/// ratio is formed round by round first, then summarised).
pub fn med_range(v: &[f64]) -> (f64, f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) };
    (med, s[0], s[n - 1])
}

// ---------------------------------------------------------------------------
// The NVRTC units, embedded so the Mac can hash and compose what the card
// compiles. Parts are joined by "\n" exactly as `gpu::KernelSource::new` does.
// ---------------------------------------------------------------------------

/// The persistent module: L1's `tv_tetra48_persall` and microbenchmarks,
/// L2's fifteen arms, the L2 prologue reference, and route P's segment.
/// Its own module, so neither the bench unit nor the served unit moves.
pub const PERS_UNIT_PARTS: [(&str, &str); 13] = [
    ("llvq_slot.cuh", include_str!("../kernels/llvq_slot.cuh")),
    ("matvec.cu", include_str!("../kernels/matvec.cu")),
    ("llvq_f1rank.cuh", include_str!("../kernels/llvq_f1rank.cuh")),
    ("llvq_f1rank_v3.cuh", include_str!("../kernels/llvq_f1rank_v3.cuh")),
    ("llvq_tetra48.cuh", include_str!("../kernels/llvq_tetra48.cuh")),
    ("llvq_rot.cuh", include_str!("../kernels/llvq_rot.cuh")),
    ("rotate.cu", include_str!("../kernels/rotate.cu")),
    ("llvq_pers.cuh", include_str!("../kernels/llvq_pers.cuh")),
    ("tetra48_pers.cu", include_str!("../kernels/tetra48_pers.cu")),
    ("tetra48_persall.cu", include_str!("../kernels/tetra48_persall.cu")),
    ("pers_ref.cu", include_str!("../kernels/pers_ref.cu")),
    ("pers_segment.cu", include_str!("../kernels/pers_segment.cu")),
    ("pers_4e.cu", include_str!("../kernels/pers_4e.cu")),
];

/// The persistent module's text for a tile (`tile::Tile::define()`).
pub fn pers_unit_text(tile_define: &str) -> String {
    std::iter::once(tile_define).chain(PERS_UNIT_PARTS.iter().map(|p| p.1)).collect::<Vec<_>>().join("\n")
}

/// The served unit of the 4B, in the order `fused_cuda.rs` assembles it for
/// a Tetra48 file with int4 records and `LLVQ_EMBED=q4`: the defines, then
/// these ten parts. The L2 bench compiles it as its own module so its
/// `tv_tetra48_h` is the served kernel's text, and prints its digest; the job
/// compares that digest with the served door's.
pub const SERVED_UNIT_PARTS: [(&str, &str); 10] = [
    ("llvq_slot.cuh", include_str!("../kernels/llvq_slot.cuh")),
    ("matvec.cu", include_str!("../kernels/matvec.cu")),
    ("llvq_rot.cuh", include_str!("../kernels/llvq_rot.cuh")),
    ("rotate.cu", include_str!("../kernels/rotate.cu")),
    ("llvq_f1rank.cuh", include_str!("../kernels/llvq_f1rank.cuh")),
    ("llvq_f1rank_v3.cuh", include_str!("../kernels/llvq_f1rank_v3.cuh")),
    ("llvq_tetra48.cuh", include_str!("../kernels/llvq_tetra48.cuh")),
    ("tv_tetra48_h.cu", include_str!("../../llvq-llm/kernels/tv_tetra48_h.cu")),
    ("emb_q4.cu", include_str!("../../llvq-llm/kernels/emb_q4.cu")),
    ("tv_q4_h.cu", include_str!("../../llvq-llm/kernels/tv_q4_h.cu")),
];

/// The served unit's text for its defines (`tile.define()` then
/// `Prefill::defines()`).
pub fn served_unit_text(defines: &str) -> String {
    std::iter::once(defines).chain(SERVED_UNIT_PARTS.iter().map(|p| p.1)).collect::<Vec<_>>().join("\n")
}

/// The tiled int4 readers (`tv_q4_tile.cu`), the part `fused_cuda.rs`
/// appends after [`SERVED_UNIT_PARTS`] for a runtime that holds its int4
/// words tiled: a config whose rows are "v9".
pub const SERVED_UNIT_TILE_PART: (&str, &str) = ("tv_q4_tile.cu", include_str!("../../llvq-llm/kernels/tv_q4_tile.cu"));

/// The served unit under a v9 config: [`served_unit_text`], then the tiled
/// int4 readers. The other configs' unit does not move.
pub fn served_unit_text_tiled(defines: &str) -> String {
    format!("{}\n{}", served_unit_text(defines), SERVED_UNIT_TILE_PART.1)
}

/// The served unit's defines at the served tile of the L40S and the default
/// prefill: what every published 4B number since 2026-09-20 compiled.
pub const SERVED_DEFINES_SM89: &str = "#define TILE_BLOCKS 64u\n#define TETRA48_ROWS 4u\n#define TETRA48_TILE 128u\n";

// ---------------------------------------------------------------------------
// The device structs, mirrored byte for byte. `tests/pers_matches_served.rs`
// compiles the kernels' own struct definitions and checks every offset.
// ---------------------------------------------------------------------------

/// Pointers of `PersArgs`, in declaration order.
pub const PERS_ARGS_PTRS: [&str; 22] = [
    "sites", "rows", "pref", "br", "suf", "invnorm", "gscale", "rscale", "tail", "x", "y_all", "slice_lo",
    "slice_hi", "bar", "cnts", "abort_flag", "resid", "alpha", "signs_h", "signs_i", "small_h", "small_i",
];
/// `sizeof(PersArgs)`.
pub const PERS_ARGS_BYTES: usize = 232;

/// The host image of `PersArgs`.
#[derive(Clone, Debug, Default)]
pub struct PersArgsHost {
    pub ptrs: [u64; 22],
    pub nsites: u32,
    pub hidden: u32,
    pub inter: u32,
    pub m_h: u32,
    pub k_h: u32,
    pub m_i: u32,
    pub k_i: u32,
    pub inv_h: f32,
    pub inv_i: f32,
    pub eps: f32,
}

impl PersArgsHost {
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(PERS_ARGS_BYTES);
        for p in self.ptrs {
            b.extend(p.to_le_bytes());
        }
        for v in [self.nsites, self.hidden, self.inter, self.m_h, self.k_h, self.m_i, self.k_i, 0, 0, 0] {
            b.extend(v.to_le_bytes());
        }
        for v in [self.inv_h, self.inv_i, self.eps] {
            b.extend(v.to_le_bytes());
        }
        b.resize(PERS_ARGS_BYTES, 0);
        b
    }
}

/// `PersLaunch`, passed by value: 40 bytes, 8-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PersLaunch {
    pub bar_base: u32,
    pub epoch: u32,
    pub cnt_offset: u32,
    pub pad: u32,
    pub dbg: u64,
    pub dbg_norm: u64,
    pub dbg_rot: u64,
}

/// `sizeof(PersLaunch)`.
pub const PERS_LAUNCH_BYTES: usize = 40;

/// One Tetra projection of a segment, as `SegTetra` holds it: 48 bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegTetraHost {
    pub words: u64,
    pub rscale: u64,
    pub tail: u64,
    pub gscale: u64,
    pub stride_u32: u32,
    pub nblocks: u32,
    pub tail_w: u32,
    pub d_out: u32,
}

impl SegTetraHost {
    fn put(&self, b: &mut Vec<u8>) {
        for p in [self.words, self.rscale, self.tail, self.gscale] {
            b.extend(p.to_le_bytes());
        }
        for v in [self.stride_u32, self.nblocks, self.tail_w, self.d_out] {
            b.extend(v.to_le_bytes());
        }
    }
}

/// `sizeof(SegArgs)` and `sizeof(SegLaunch)`.
pub const SEG_ARGS_BYTES: usize = 336;
pub const SEG_LAUNCH_BYTES: usize = 24;

/// The host image of one layer's `SegArgs` (`kernels/pers_segment.cu`).
#[derive(Clone, Debug, Default)]
pub struct SegArgsHost {
    pub alpha: u64,
    pub signs_h: u64,
    pub small_h: u64,
    pub silu: u64,
    pub act: u64,
    pub gate: SegTetraHost,
    pub up: SegTetraHost,
    pub down: SegTetraHost,
    pub signs_i: u64,
    pub small_i: u64,
    pub q4w: u64,
    pub q4s: u64,
    pub q4b: u64,
    pub tables: [u64; 5],
    pub slice_gu: u64,
    pub slice_dn: u64,
    pub bar: u64,
    pub abort_flag: u64,
    pub hidden: u32,
    pub inter: u32,
    pub m_h: u32,
    pub k_h: u32,
    pub m_i: u32,
    pub k_i: u32,
    pub inv_h: f32,
    pub inv_i: f32,
    pub eps: f32,
    pub gpr: u32,
}

impl SegArgsHost {
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(SEG_ARGS_BYTES);
        for p in [self.alpha, self.signs_h, self.small_h, self.silu, self.act] {
            b.extend(p.to_le_bytes());
        }
        self.gate.put(&mut b);
        self.up.put(&mut b);
        self.down.put(&mut b);
        for p in [self.signs_i, self.small_i, self.q4w, self.q4s, self.q4b] {
            b.extend(p.to_le_bytes());
        }
        for p in self.tables {
            b.extend(p.to_le_bytes());
        }
        for p in [self.slice_gu, self.slice_dn, self.bar, self.abort_flag] {
            b.extend(p.to_le_bytes());
        }
        for v in [self.hidden, self.inter, self.m_h, self.k_h, self.m_i, self.k_i] {
            b.extend(v.to_le_bytes());
        }
        for v in [self.inv_h, self.inv_i, self.eps] {
            b.extend(v.to_le_bytes());
        }
        b.extend(self.gpr.to_le_bytes());
        b.resize(SEG_ARGS_BYTES, 0);
        b
    }
}

/// `SegLaunch`, passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegLaunch {
    pub x: u64,
    pub out: u64,
    pub bar_base: u32,
    pub pad: u32,
}

// ---- 4E (`kernels/pers_4e.cu`): Pers4eQkv, Pers4eLayer, Pers4eArgs, Pers4eLaunch ----

/// `sizeof(Pers4eQkv)`, `sizeof(Pers4eLayer)`, `sizeof(Pers4eArgs)`,
/// `sizeof(Pers4eLaunch)`; the offsets below are checked against the kernel's
/// own definitions by `tests/pers_matches_served.rs`.
pub const P4E_QKV_BYTES: usize = 152;
pub const P4E_LAYER_BYTES: usize = 600;
pub const P4E_ARGS_BYTES: usize = 240;
/// `sizeof(Pers4eOT)`: a Tetra o and its rotation.
pub const P4E_OT_BYTES: usize = 80;
/// Where `ot` sits in a `Pers4eLayer`, and the head pointers in `Pers4eArgs`.
pub const P4E_LAYER_OT: usize = 592;
pub const P4E_ARGS_HEAD: usize = 216;
pub const P4E_LAUNCH_BYTES: usize = 40;
/// Where `qkv` and `mlp` sit in a `Pers4eLayer`.
pub const P4E_LAYER_QKV: usize = 40;
pub const P4E_LAYER_MLP: usize = 256;

/// The q, k (Tetra) and v (int4) projections of one layer, and the scratch
/// they write.
#[derive(Clone, Copy, Debug, Default)]
pub struct P4eQkvHost {
    pub q: SegTetraHost,
    pub k: SegTetraHost,
    pub vw: u64,
    pub vsc: u64,
    pub vbi: u64,
    pub v_dout: u32,
    pub v_gpr: u32,
    pub d_in: u32,
    pub out_q: u64,
    pub out_k: u64,
    pub out_v: u64,
}

/// One layer of 4E.
#[derive(Clone, Debug, Default)]
pub struct P4eLayerHost {
    pub ln1: u64,
    pub signs_a: u64,
    pub small_a: u64,
    pub m_a: u32,
    pub k_a: u32,
    pub inv_a: f32,
    pub qkv: P4eQkvHost,
    pub q_norm: u64,
    pub k_norm: u64,
    pub kc: u64,
    pub vc: u64,
    pub ow: u64,
    pub osc: u64,
    pub obi: u64,
    pub o_gpr: u32,
    pub tetra_down: bool,
    pub mlp: SegArgsHost,
    /// The device address of this layer's `Pers4eOT` (a Tetra o), or 0 for
    /// the int4 o.
    pub ot: u64,
}

/// A Tetra o_proj and its own rotation (`Pers4eOT`), uploaded once a layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct P4eOTHost {
    pub t: SegTetraHost,
    pub signs: u64,
    pub small: u64,
    pub m: u32,
    pub k: u32,
    pub inv: f32,
}

impl P4eOTHost {
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(P4E_OT_BYTES);
        self.t.put(&mut b);
        b.extend(self.signs.to_le_bytes());
        b.extend(self.small.to_le_bytes());
        for v in [self.m, self.k, self.inv.to_bits(), 0] {
            b.extend(v.to_le_bytes());
        }
        debug_assert_eq!(b.len(), P4E_OT_BYTES);
        b
    }
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

impl P4eLayerHost {
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; P4E_LAYER_BYTES];
        put64(&mut b, 0, self.ln1);
        put64(&mut b, 8, self.signs_a);
        put64(&mut b, 16, self.small_a);
        put32(&mut b, 24, self.m_a);
        put32(&mut b, 28, self.k_a);
        put32(&mut b, 32, self.inv_a.to_bits());
        let q = P4E_LAYER_QKV;
        let t = &self.qkv;
        for (i, p) in [t.q.words, t.q.rscale, t.q.tail, t.q.gscale].into_iter().enumerate() {
            put64(&mut b, q + 8 * i, p);
        }
        put32(&mut b, q + 32, t.q.stride_u32);
        put32(&mut b, q + 36, t.q.d_out);
        for (i, p) in [t.k.words, t.k.rscale, t.k.tail, t.k.gscale].into_iter().enumerate() {
            put64(&mut b, q + 40 + 8 * i, p);
        }
        put32(&mut b, q + 72, t.k.stride_u32);
        put32(&mut b, q + 76, t.k.d_out);
        put64(&mut b, q + 80, t.vw);
        put64(&mut b, q + 88, t.vsc);
        put64(&mut b, q + 96, t.vbi);
        put32(&mut b, q + 104, t.v_dout);
        put32(&mut b, q + 108, t.v_gpr);
        put32(&mut b, q + 112, t.q.nblocks);
        put32(&mut b, q + 116, t.q.tail_w);
        put32(&mut b, q + 120, t.d_in);
        put64(&mut b, q + 128, t.out_q);
        put64(&mut b, q + 136, t.out_k);
        put64(&mut b, q + 144, t.out_v);
        let o = q + P4E_QKV_BYTES;
        for (i, p) in [self.q_norm, self.k_norm, self.kc, self.vc, self.ow, self.osc, self.obi].into_iter().enumerate() {
            put64(&mut b, o + 8 * i, p);
        }
        put32(&mut b, o + 56, self.o_gpr);
        put32(&mut b, o + 60, u32::from(self.tetra_down));
        b[P4E_LAYER_MLP..P4E_LAYER_MLP + SEG_ARGS_BYTES].copy_from_slice(&self.mlp.bytes());
        put64(&mut b, P4E_LAYER_OT, self.ot);
        b
    }
}

/// The launch-wide constants of 4E: 22 pointers, then 8 u32 and 2 f32, then
/// the head's three pointers.
#[derive(Clone, Debug, Default)]
pub struct P4eArgsHost {
    /// layers, emb_w, emb_s, emb_b, final_norm, cosr, sinr, x_a, x_b, q_out,
    /// ctx, attn_scratch, bar, abort_flag, slots, nan_flag, rows, pref, br,
    /// suf, invnorm, attn_ticket
    pub ptrs: [u64; 22],
    pub n_layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub emb_gpr: u32,
    pub n_heads: u32,
    pub n_kv: u32,
    pub groups: u32,
    pub cap: u32,
    pub eps: f32,
    pub c: f32,
    /// head_w, head_s, head_b: the embedding's own table when tied (the 4B),
    /// the head's table when not (the 8B and the 14B).
    pub head: [u64; 3],
}

impl P4eArgsHost {
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; P4E_ARGS_BYTES];
        for (i, p) in self.ptrs.into_iter().enumerate() {
            put64(&mut b, 8 * i, p);
        }
        for (i, v) in [self.n_layers, self.vocab, self.hidden, self.emb_gpr, self.n_heads, self.n_kv, self.groups, self.cap]
            .into_iter()
            .enumerate()
        {
            put32(&mut b, 176 + 4 * i, v);
        }
        put32(&mut b, 208, self.eps.to_bits());
        put32(&mut b, 212, self.c.to_bits());
        for (i, p) in self.head.into_iter().enumerate() {
            put64(&mut b, P4E_ARGS_HEAD + 8 * i, p);
        }
        b
    }
}

/// `Pers4eLaunch`, passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct P4eLaunch {
    pub bar_base: u32,
    pub pos0: u32,
    pub n_tokens: u32,
    pub tok0: u32,
    pub eos: u32,
    pub pad: u32,
    pub ring: u64,
    /// The phase stamps of a `pers_4e_stamps_*` launch, `[n][p4e_events][grid]`
    /// u64; 0 and unread on the token symbols.
    pub stamps: u64,
}

// ---- 4E's verify pass (`kernels/pers_4ev.cu`): n rows of the decode a launch ----

/// `P4V_KMAX` of `pers_4ev.cu`: the rows one verify launch carries, at most.
pub const P4V_KMAX: u32 = 4;

/// `Pers4vLaunch`, passed by value: 4 words, 13 device pointers, 2 words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct P4vLaunch {
    pub bar_base: u32,
    pub pos0: u32,
    pub n_rows: u32,
    /// Floats of the CTA's staging area after the tables ([`p4v_act_floats`]).
    pub act_floats: u32,
    pub toks: u64,
    pub ring: u64,
    pub x_a: u64,
    pub x_b: u64,
    pub q_raw: u64,
    pub k_raw: u64,
    pub v_raw: u64,
    pub q_out: u64,
    pub ctx: u64,
    pub act: u64,
    pub part: u64,
    pub tickets: u64,
    pub slots: u64,
    /// Floats of one row's split states: n_kv x ceil(cap / 256) x groups x 130.
    pub part_stride: u32,
    pub pad: u32,
    /// 0, or `[1 + n_bar_4e][grid]` u64 phase clocks ([`p4v_phases`]).
    pub stamps: u64,
}

/// `sizeof(Pers4vLaunch)`.
pub const P4V_LAUNCH_BYTES: usize = 136;

/// The verify pass's phases, in the order of its waits within a layer.
pub const P4V_PHASES: [&str; 8] = ["embed", "qkv", "rope", "attn", "o", "gateup", "down", "head"];

/// A stamped verify launch's time by phase, nanoseconds, summed over the
/// layers: `stamps` is `[1 + n_bar_4e(n_layers)][grid]` (event 0 each CTA's
/// start, event e after its e-th wait). Event e's time is the median over
/// the CTAs, and phase e lasts from event e - 1 to event e. Returns the
/// totals in the order of [`P4V_PHASES`], or an error when the buffer does
/// not have that shape or a CTA never wrote an event.
pub fn p4v_phases(stamps: &[u64], n_layers: u32, grid: u32) -> Result<[f64; 8], String> {
    let events = 1 + n_bar_4e(n_layers) as usize;
    let g = grid as usize;
    if g == 0 || stamps.len() != events * g {
        return Err(format!("verify stamps: {} values for {events} events of {g} CTAs", stamps.len()));
    }
    if stamps.contains(&0) {
        return Err("verify stamps: a CTA wrote no clock at some event".into());
    }
    let med = |e: usize| {
        let mut v: Vec<u64> = stamps[e * g..(e + 1) * g].to_vec();
        v.sort_unstable();
        v[g / 2] as f64
    };
    let t: Vec<f64> = (0..events).map(med).collect();
    let mut out = [0.0f64; 8];
    for e in 1..events {
        // wait 1 ends the embedding, waits 2..=1 + 6L the layers' six
        // phases, the last the head
        let kind = if e == 1 {
            0
        } else if e == events - 1 {
            7
        } else {
            1 + (e - 2) % 6
        };
        out[kind] += t[e] - t[e - 1];
    }
    Ok(out)
}

/// The verify kernel's symbol for a RoPE form and a block size.
pub fn p4v_symbol(form: u32, threads: u32) -> Result<&'static str, String> {
    Ok(match (form, threads) {
        (0, 768) => "pers_4ev_rows_f0",
        (1, 768) => "pers_4ev_rows_f1",
        (2, 768) => "pers_4ev_rows_f2",
        (0, 512) => "pers_4ev_rows_f0_t512",
        (1, 512) => "pers_4ev_rows_f1_t512",
        (2, 512) => "pers_4ev_rows_f2_t512",
        _ => return Err(format!("4E verify: no kernel for RoPE form {form} at {threads} threads")),
    })
}

/// `P4V_SSM` of `pers_4ev.cu`: the floats before the rows, a rotation's
/// staged rows of `small` (k x 32, k at most 20).
pub const P4V_SSM: u32 = 640;

/// The verify launch's staging area, in floats, under `smem_max` bytes of
/// dynamic shared memory a block: what every phase of `P4V_KMAX` rows needs
/// at once when it fits (the staged `small`, the widest of hidden, the
/// context and inter times the rows, the head's maxima), else as much as
/// fits, a multiple of 4. Refused when it cannot hold one row of each phase
/// or attention's stage ([`p4e_smem_bytes_form`]'s at form 7): then no group
/// size works.
pub fn p4v_act_floats(hidden: u32, inter: u32, d_ctx: u32, groups: u32, smem_max: u32) -> Result<u32, String> {
    let attn = (16 * 1024 * p4e_attn_ring(7) + 4 * 4 * groups * 130 + 4 * groups * 128).div_ceil(4);
    let widest = hidden.max(inter).max(d_ctx);
    let fixed = P4V_SSM + 2 * P4V_KMAX;
    let need = attn.max(widest + fixed);
    let want = (P4V_KMAX * widest + fixed).max(attn);
    let room = smem_max.saturating_sub(SMEM_HEADER + TABLE_BYTES) / 4;
    let got = want.min(room) & !3;
    if got < need {
        return Err(format!(
            "4E verify: {room} floats of staging under {smem_max} B of shared memory, one row of each phase needs {need}"
        ));
    }
    Ok(got)
}

/// The verify module's text: the 4E module of row form 7 (`p4e_rows_define`
/// then [`pers_unit_text`]) and `pers_4ev.cu`. Its own module, so the served
/// 4E module keeps its text.
pub fn p4v_unit_text(tile_define: &str) -> String {
    format!("{}{}\n{}", p4e_rows_define(7).expect("form 7 exists"), pers_unit_text(tile_define), P4V_PART.1)
}

/// `pers_4ev.cu`, embedded so the Mac hashes what the card compiles.
pub const P4V_PART: (&str, &str) = ("pers_4ev.cu", include_str!("../kernels/pers_4ev.cu"));

// ---- the tree verify pass (`kernels/pers_4et.cu`, docs/plan-verify-tree.md) ----

/// `P4T_RMAX` of `pers_4et.cu`: the rows one tree launch carries, at most.
pub const P4T_RMAX: u32 = 32;

/// `Pers4tLaunch`, passed by value: 14 words, 25 device pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct P4tLaunch {
    pub bar_base: u32,
    /// The root's position.
    pub pos0: u32,
    pub n_rows: u32,
    /// Floats of the area after the tables ([`p4t_area_floats`]), even.
    pub area_floats: u32,
    /// Rows of the previous launch moved into the cache, and the position of the first.
    pub n_move: u32,
    pub move_pos0: u32,
    /// Floats of one row's attention split states: n_kv x ceil(cap / 256) x groups x 130.
    pub part_stride: u32,
    /// `xin`'s row stride in halves: at least every projection's d_in, a multiple of 8.
    pub ldx: u32,
    /// The d_in slices of q/k/v, o, gate/up, down and the head ([`p4t_splits`]).
    pub split_qkv: u32,
    pub split_o: u32,
    pub split_gu: u32,
    pub split_dn: u32,
    pub split_head: u32,
    pub pad: u32,
    /// `[R]` tokens, `[R]` depths, `[R][P4T_RMAX]` u8 paths, `[n_move]` slots.
    pub toks: u64,
    pub depth: u64,
    pub path: u64,
    pub mv: u64,
    /// `[R]` each row's argmax.
    pub ring: u64,
    /// 0, or `[R][vocab]` binary16 logits (a check).
    pub logits: u64,
    pub x_a: u64,
    pub x_b: u64,
    pub xin: u64,
    pub xn: u64,
    pub q_raw: u64,
    pub k_raw: u64,
    pub v_raw: u64,
    pub q_out: u64,
    pub ctx: u64,
    pub g_raw: u64,
    pub u_raw: u64,
    /// `[layers][P4T_RMAX][n_kv x 128]` each slot's K and V.
    pub tk: u64,
    pub tv: u64,
    pub part: u64,
    /// The projections' slice partials ([`p4t_fpart_floats`]).
    pub fpart: u64,
    pub tickets: u64,
    /// The slice folds' counters ([`p4t_ftickets`]), zero at rest.
    pub ftickets: u64,
    pub slots: u64,
    /// 0, or `[1 + n_bar_4et][grid]` u64 phase clocks ([`p4t_phases`]).
    pub stamps: u64,
}

/// `sizeof(Pers4tLaunch)`.
pub const P4T_LAUNCH_BYTES: usize = 256;

/// The waits of one tree launch: the embedding and move, 9 a layer, 1 more
/// a layer with a Tetra o, the final norm, the head.
pub fn n_bar_4et(layers: u32, tetra_o_layers: u32) -> u32 {
    3 + 9 * layers + tetra_o_layers
}

/// `p4t_area_floats` of `pers_4et.cu`: the area at `nw` warps a CTA and `rs`
/// staged activation rows (a multiple of 8, up to 32), `widest` the widest
/// row a prologue holds, `groups` query heads a KV head (attention's share
/// at row form 7).
pub fn p4t_area_floats(nw: u32, rs: u32, widest: u32, groups: u32) -> u32 {
    let attn = (16 * 1024 * p4e_attn_ring(7) + 4 * 4 * groups * 130 + 4 * groups * 128).div_ceil(4);
    let mma = nw * 16 * 28 + 2 * (rs * 52 + 16 * nw * 14);
    widest.max(attn).max(mma) + 2 * P4T_RMAX
}

/// The d_in slices of one projection phase: about one unit a CTA. `tiles`
/// are its 16-row tiles summed over the phase's projections, `nw` the warps
/// a CTA, `d_in` the phase's input width. At least 1, at most half the
/// 48-column sub-steps (two sub-steps a slice at least). A property of the
/// model and the grid, never of the rows a launch carries: identity needs
/// it (plan section 6).
pub fn p4t_split(tiles: u32, nw: u32, d_in: u32, grid: u32) -> u32 {
    let blocks = tiles.div_ceil(nw).max(1);
    let s = (grid + blocks / 2) / blocks;
    s.clamp(1, (d_in.div_ceil(48) / 2).max(1))
}

/// The five phases' slices, q/k/v, o, gate/up, down, head, at the widths
/// of a model.
pub fn p4t_splits(hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32, nw: u32, grid: u32) -> [u32; 5] {
    let dkv = n_kv * 128;
    [
        p4t_split((d_ctx + 2 * dkv) / 16, nw, hidden, grid),
        p4t_split(hidden / 16, nw, d_ctx, grid),
        p4t_split(2 * inter / 16, nw, hidden, grid),
        p4t_split(hidden / 16, nw, inter, grid),
        p4t_split(vocab / 16, nw, hidden, grid),
    ]
}

/// Floats of the slice partials: the largest phase's, `P4T_RMAX` rows.
pub fn p4t_fpart_floats(splits: [u32; 5], hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32) -> u64 {
    let dkv = n_kv * 128;
    let outs = [d_ctx + 2 * dkv, hidden, 2 * inter, hidden, vocab];
    let rows = P4T_RMAX as u64;
    splits.iter().zip(outs).map(|(&s, d)| if s > 1 { s as u64 * rows * d as u64 } else { 0 }).max().unwrap_or(0)
}

/// The slice folds' counters: the largest phase's 16-row tiles.
pub fn p4t_ftickets(hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32) -> u32 {
    [d_ctx + 2 * n_kv * 128, hidden, 2 * inter, vocab].into_iter().max().unwrap_or(0) / 16
}

/// The four parts of a row phase an a4 launch times (F3), in slot order.
pub const P4T_PARTS: [&str; 4] = ["setup", "staging", "loops", "store"];

/// The row phases an a4 launch times (F3), in slot order.
pub const P4T_ROW_PHASES: [&str; 5] = ["qkv", "o", "gateup", "down", "head"];

/// An a4 stamped launch's F3 slots: `tail` is what follows the phase events,
/// `[4 phase + part][grid]` nanoseconds, each CTA's sum over the layers.
/// Per row phase and part, the median over the CTAs.
pub fn p4t_parts(tail: &[u64], grid: u32) -> Result<[[f64; 4]; 5], String> {
    let g = grid as usize;
    if g == 0 || tail.len() != 20 * g {
        return Err(format!("tree parts: {} values for 20 slots of {g} CTAs", tail.len()));
    }
    let mut out = [[0.0f64; 4]; 5];
    for (ph, o) in out.iter_mut().enumerate() {
        for (p, x) in o.iter_mut().enumerate() {
            let mut v: Vec<f64> = tail[(4 * ph + p) * g..(4 * ph + p + 1) * g].iter().map(|&t| t as f64).collect();
            v.sort_by(f64::total_cmp);
            *x = v[g / 2];
        }
    }
    Ok(out)
}

/// The tree pass's phases, in the order of its waits, for a model with no
/// Tetra o.
pub const P4T_PHASES: [&str; 12] =
    ["embed", "ln1", "qkv", "rope", "attn", "o", "ln2", "gateup", "dn_in", "down", "norm", "head"];

/// A stamped tree launch's time by phase, nanoseconds, summed over the
/// layers, in the order of [`P4T_PHASES`]. The stamps are `[1 + 2 n_bar][grid]`:
/// event 0 a CTA's start, event 2e - 1 when the CTA reached wait e (its work
/// of the phase done), event 2e when it left it. For each phase: `[0]` the
/// wall, from the CTAs' median leaving of the previous wait to their median
/// leaving of this one; `[1]`, `[2]`, `[3]` the minimum, median and maximum
/// over the CTAs of a CTA's busy time (its arrival at the wait minus its
/// leaving of the previous one), each CTA's summed over the layers first. A
/// phase whose units run in two waves shows its maximum near twice its
/// minimum. A model with a Tetra o is refused (its layers carry one more
/// wait).
pub fn p4t_phases(stamps: &[u64], n_layers: u32, grid: u32) -> Result<[[f64; 4]; 12], String> {
    let n_bar = n_bar_4et(n_layers, 0) as usize;
    let events = 1 + 2 * n_bar;
    let g = grid as usize;
    if g == 0 || stamps.len() != events * g {
        return Err(format!("tree stamps: {} values for {events} events of {g} CTAs", stamps.len()));
    }
    if stamps.contains(&0) {
        return Err("tree stamps: a CTA wrote no clock at some event".into());
    }
    let at = |e: usize, c: usize| stamps[e * g + c] as f64;
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let kind = |w: usize| {
        if w == 1 {
            0
        } else if w == n_bar - 1 {
            10
        } else if w == n_bar {
            11
        } else {
            1 + (w - 2) % 9
        }
    };
    let mut wall = [0.0f64; 12];
    let mut busy = vec![[0.0f64; 12]; g];
    for w in 1..=n_bar {
        let k = kind(w);
        let mut prev: Vec<f64> = (0..g).map(|c| at(2 * w - 2, c)).collect();
        let mut leave: Vec<f64> = (0..g).map(|c| at(2 * w, c)).collect();
        wall[k] += med(&mut leave) - med(&mut prev);
        for (c, b) in busy.iter_mut().enumerate() {
            b[k] += at(2 * w - 1, c) - at(2 * w - 2, c);
        }
    }
    let mut out = [[0.0f64; 4]; 12];
    for (k, (o, w)) in out.iter_mut().zip(wall).enumerate() {
        let mut v: Vec<f64> = busy.iter().map(|b| b[k]).collect();
        v.sort_by(f64::total_cmp);
        *o = [w, v[0], v[g / 2], v[g - 1]];
    }
    Ok(out)
}

/// The units of each projection phase (q/k/v, o, gate/up, down, head) under
/// `splits`, on `grid` CTAs of `nw` warps, as `pers_4et.cu`'s p4t_phase deals
/// them (each projection's blocks of nw tiles, times its slices; unit u on
/// CTA u mod grid): (units, waves, fill), fill = (units / grid) / waves, the
/// share of the CTAs' unit slots the phase uses.
#[allow(clippy::too_many_arguments)]
pub fn p4t_plan(splits: [u32; 5], hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32, nw: u32, grid: u32) -> [(u32, u32, f64); 5] {
    let b = |d_out: u32| (d_out / 16).div_ceil(nw);
    let dkv = n_kv * 128;
    let blocks = [b(d_ctx) + 2 * b(dkv), b(hidden), 2 * b(inter), b(hidden), b(vocab)];
    let mut out = [(0u32, 0u32, 0.0f64); 5];
    for ((o, b), s) in out.iter_mut().zip(blocks).zip(splits) {
        let u = b * s;
        let w = u.div_ceil(grid);
        *o = (u, w, u as f64 / grid as f64 / w as f64);
    }
    out
}

/// Job N1's balanced plan: `proposed` kept where q/k/v and gate/up fill at
/// least 90 % of the grid's slots ([`p4t_plan`]); otherwise that phase's S
/// in 1..=16 (at most half its sub-steps) with the largest fill, the
/// smallest S on a tie. The other phases keep the proposal.
#[allow(clippy::too_many_arguments)]
pub fn p4t_splits_balanced(proposed: [u32; 5], hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32, nw: u32, grid: u32) -> [u32; 5] {
    let mut s = proposed;
    for (i, d_in) in [(0usize, hidden), (2, hidden)] {
        let fill = |x: u32| {
            let mut t = s;
            t[i] = x;
            p4t_plan(t, hidden, inter, d_ctx, n_kv, vocab, nw, grid)[i].2
        };
        if fill(s[i]) < 0.9 {
            let cap = (d_in.div_ceil(48) / 2).clamp(1, 16);
            let mut best = 1;
            for x in 2..=cap {
                if fill(x) > fill(best) + 1e-12 {
                    best = x;
                }
            }
            s[i] = best;
        }
    }
    s
}

/// The tree kernel's tiled symbol for a RoPE form (a module of
/// [`p4t_tiled_unit_text`]).
pub fn p4t_tiled_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_tiled_f0_t512",
        1 => "pers_4et_tiled_f1_t512",
        2 => "pers_4et_tiled_f2_t512",
        _ => return Err(format!("tree verify: no tiled kernel for RoPE form {form}")),
    })
}

/// The tiled tree module's text: [`p4t_unit_text`] under `P4T_TILED 1`, which
/// emits the tiled symbols and `p4t_tile_words` instead of the untiled ones.
pub fn p4t_tiled_unit_text(tile_define: &str) -> String {
    format!("#define P4T_TILED 1\n{}", p4t_unit_text(tile_define))
}

/// u32 words of a projection's tree-tiled copy: its 16-row tiles times its
/// 48-column sub-steps times 48 (Tetra) or 96 (int4) words.
pub fn p4t_tiled_words(int4: bool, d_out: u32, d_in: u32) -> u64 {
    (d_out / 16) as u64 * d_in.div_ceil(48) as u64 * if int4 { 96 } else { 48 }
}

/// `p4t_stream_cap` of `pers_4et_stream.cu`: the 48-column sub-steps one
/// slice of the stream unit may hold, in an area of `area_floats` floats at
/// `nw` warps a CTA: the warps' A tiles, `P4T_RMAX` staged rows of 24 ns + 4
/// words, the argmax maxima. A function of the area and the warps, never of
/// the rows a launch carries.
pub fn p4t_stream_cap(area_floats: u32, nw: u32) -> u32 {
    let fixed = nw * 16 * 28 + 2 * P4T_RMAX;
    let per = area_floats.saturating_sub(fixed) / P4T_RMAX;
    per.saturating_sub(4) / 24
}

/// The largest slice of a projection of `d_in` columns cut in `s`: its
/// sub-steps, as `pers_4et.cu` cuts them (slice sl from sl nsub / S).
pub fn p4t_slice_steps(d_in: u32, s: u32) -> u32 {
    d_in.div_ceil(48).div_ceil(s)
}

/// Job N2's plan for the stream unit (arm a3, and a2 beside it). For each
/// phase (q/k/v, o, gate/up, down, head) the floor is the smallest S whose
/// slices fit `cap` sub-steps ([`p4t_stream_cap`]); from it, the smallest S
/// whose fill ([`p4t_plan`]) reaches 90 %, then S + 1 or else S + 2 when
/// that reaches 98 %. S at most 32 and at most the phase's sub-steps; when
/// no S reaches 90 %, the best fill, the smaller S on a tie. Every slice
/// costs a staged activation and a partial written and read back, so S
/// stays as small as a filled grid allows. Refused when no S up to 32 fits.
#[allow(clippy::too_many_arguments)]
pub fn p4t_splits_stream(hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, vocab: u32, nw: u32, grid: u32, cap: u32) -> Result<[u32; 5], String> {
    let d_ins = [hidden, d_ctx, hidden, inter, hidden];
    let mut s = [1u32; 5];
    for (i, &d_in) in d_ins.iter().enumerate() {
        let top = d_in.div_ceil(48).min(32);
        let lo = (1..=top).find(|&x| p4t_slice_steps(d_in, x) <= cap).ok_or_else(|| {
            format!("tree stream: phase {i} ({d_in} columns) needs slices of at most {cap} sub-steps, which no S up to {top} gives")
        })?;
        let fill = |x: u32| {
            let mut t = s;
            t[i] = x;
            p4t_plan(t, hidden, inter, d_ctx, n_kv, vocab, nw, grid)[i].2
        };
        let mut pick = match (lo..=top).find(|&x| fill(x) >= 0.9) {
            Some(x) => x,
            None => (lo..=top).fold(lo, |b, x| if fill(x) > fill(b) + 1e-12 { x } else { b }),
        };
        if let Some(x) = [pick + 1, pick + 2].into_iter().find(|&x| x <= top && fill(x) >= 0.98) {
            pick = x;
        }
        s[i] = pick;
    }
    Ok(s)
}

/// The stream tree module's text: [`p4t_tiled_unit_text`] (for
/// `p4t_tile_words`) under `P4T_STREAM 1`, then `pers_4et_stream.cu`, which
/// emits the `pers_4et_stream_*` symbols. `pers_4et.cu` keeps its bytes, so
/// the other arms' modules keep theirs.
pub fn p4t_stream_unit_text(tile_define: &str) -> String {
    format!("#define P4T_STREAM 1\n{}\n{}", p4t_tiled_unit_text(tile_define), P4T_STREAM_PART.1)
}

/// The tree kernel's stream symbol for a RoPE form (a module of
/// [`p4t_stream_unit_text`]).
pub fn p4t_stream_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_stream_f0_t512",
        1 => "pers_4et_stream_f1_t512",
        2 => "pers_4et_stream_f2_t512",
        _ => return Err(format!("tree verify: no stream kernel for RoPE form {form}")),
    })
}

/// The a4 tree module's text: [`p4t_stream_unit_text`] under `P4T_A4 1`,
/// then `pers_4et_a4.cu` (the stream unit with the batched fold, F1, and the
/// vector prologues, F2), which emits the `pers_4et_a4_*` symbols. `fold`
/// and `pro` false build a4 without F1 or F2 (`P4T_A4_FOLD 0`,
/// `P4T_A4_PRO 0`). `pers_4et.cu` and `pers_4et_stream.cu` keep their bytes.
pub fn p4t_a4_unit_text(tile_define: &str, fold: bool, pro: bool) -> String {
    format!(
        "#define P4T_A4 1\n#define P4T_A4_FOLD {}\n#define P4T_A4_PRO {}\n{}\n{}",
        u32::from(fold),
        u32::from(pro),
        p4t_stream_unit_text(tile_define),
        P4T_A4_PART.1
    )
}

/// The tree kernel's a4 symbol for a RoPE form (a module of
/// [`p4t_a4_unit_text`]).
pub fn p4t_a4_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a4_f0_t512",
        1 => "pers_4et_a4_f1_t512",
        2 => "pers_4et_a4_f2_t512",
        _ => return Err(format!("tree verify: no a4 kernel for RoPE form {form}")),
    })
}

/// The a5 tree module's text: [`p4t_a4_unit_text`] (F1 and F2 on) under
/// `P4T_A5 1`, then `pers_4et_a5.cu` (a4 with the L2 prefetch of the next
/// row phase's tile streams), which emits the `pers_4et_a5_*` symbols.
/// `pers_4et.cu`, `pers_4et_stream.cu` and `pers_4et_a4.cu` keep their bytes.
pub fn p4t_a5_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A5 1\n{}\n{}", p4t_a4_unit_text(tile_define, true, true), P4T_A5_PART.1)
}

/// The tree kernel's a5 symbol for a RoPE form (a module of
/// [`p4t_a5_unit_text`]).
pub fn p4t_a5_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a5_f0_t512",
        1 => "pers_4et_a5_f1_t512",
        2 => "pers_4et_a5_f2_t512",
        _ => return Err(format!("tree verify: no a5 kernel for RoPE form {form}")),
    })
}

/// The module text of arm a6, a7 or a3f (`pers_4et_a6.cu`): a6 and a3f after
/// [`p4t_stream_unit_text`] (a3's), a7 after [`p4t_a5_unit_text`] (for
/// a5's prefetch helpers). a3f is a floor arm, timing only: its values are
/// void by design. The earlier files keep their bytes.
pub fn p4t_a6_unit_text(arm: &str, tile_define: &str) -> Result<String, String> {
    Ok(match arm {
        "a6" => format!("#define P4T_A6 1\n{}\n{}", p4t_stream_unit_text(tile_define), P4T_A6_PART.1),
        "a7" => format!("#define P4T_A7 1\n{}\n{}", p4t_a5_unit_text(tile_define), P4T_A6_PART.1),
        "a3f" => format!("#define P4T_A3F 1\n{}\n{}", p4t_stream_unit_text(tile_define), P4T_A6_PART.1),
        _ => return Err(format!("tree verify: no module for arm {arm}")),
    })
}

/// The symbol of arm a6, a7 or a3f for a RoPE form.
pub fn p4t_a6_symbol(arm: &str, form: u32) -> Result<&'static str, String> {
    Ok(match (arm, form) {
        ("a6", 0) => "pers_4et_a6_f0_t512",
        ("a6", 1) => "pers_4et_a6_f1_t512",
        ("a6", 2) => "pers_4et_a6_f2_t512",
        ("a7", 0) => "pers_4et_a7_f0_t512",
        ("a7", 1) => "pers_4et_a7_f1_t512",
        ("a7", 2) => "pers_4et_a7_f2_t512",
        ("a3f", 0) => "pers_4et_a3f_f0_t512",
        ("a3f", 1) => "pers_4et_a3f_f1_t512",
        ("a3f", 2) => "pers_4et_a3f_f2_t512",
        _ => return Err(format!("tree verify: no kernel for arm {arm} at RoPE form {form}")),
    })
}

/// The module text of arm a3p, a3t or a8 (`pers_4et_a8.cu`), after
/// [`p4t_stream_unit_text`] (a3's). a3p and a3t are floor arms, timing only:
/// their values are void by design. The earlier files keep their bytes.
pub fn p4t_a8_unit_text(arm: &str, tile_define: &str) -> Result<String, String> {
    let def = match arm {
        "a3p" => "P4T_A3P",
        "a3t" => "P4T_A3T",
        "a8" => "P4T_A8",
        _ => return Err(format!("tree verify: no module for arm {arm}")),
    };
    Ok(format!("#define {def} 1\n{}\n{}", p4t_stream_unit_text(tile_define), P4T_A8_PART.1))
}

/// The symbol of arm a3p, a3t or a8 for a RoPE form.
pub fn p4t_a8_symbol(arm: &str, form: u32) -> Result<&'static str, String> {
    Ok(match (arm, form) {
        ("a3p", 0) => "pers_4et_a3p_f0_t512",
        ("a3p", 1) => "pers_4et_a3p_f1_t512",
        ("a3p", 2) => "pers_4et_a3p_f2_t512",
        ("a3t", 0) => "pers_4et_a3t_f0_t512",
        ("a3t", 1) => "pers_4et_a3t_f1_t512",
        ("a3t", 2) => "pers_4et_a3t_f2_t512",
        ("a8", 0) => "pers_4et_a8_f0_t512",
        ("a8", 1) => "pers_4et_a8_f1_t512",
        ("a8", 2) => "pers_4et_a8_f2_t512",
        _ => return Err(format!("tree verify: no kernel for arm {arm} at RoPE form {form}")),
    })
}

/// The module text of arm a9 (`pers_4et_a9.cu`, the fold after the phase's
/// wait), after [`p4t_stream_unit_text`] (a3's), which keeps its bytes.
pub fn p4t_a9_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A9 1\n{}\n{}", p4t_stream_unit_text(tile_define), P4T_A9_PART.1)
}

/// The a9 symbol for a RoPE form.
pub fn p4t_a9_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a9_f0_t512",
        1 => "pers_4et_a9_f1_t512",
        2 => "pers_4et_a9_f2_t512",
        _ => return Err(format!("tree verify: no a9 kernel for RoPE form {form}")),
    })
}

/// `n_bar_a9` of `pers_4et_a9.cu`: a9's waits, a fold step's after each of
/// the four row phases of a layer and after the head.
pub fn n_bar_a9(layers: u32, tetra_o_layers: u32) -> u32 {
    n_bar_4et(layers, tetra_o_layers) + 4 * layers + 1
}

/// An a9 launch's phases, in the order of its waits (no Tetra o).
pub const P4T_PHASES_A9: [&str; 17] = [
    "embed", "ln1", "qkv", "qkv fold", "rope", "attn", "o", "o fold", "ln2", "gateup", "gateup fold", "dn_in", "down",
    "down fold", "norm", "head", "head fold",
];

/// [`p4t_phases`] for an a9 launch: per phase of [`P4T_PHASES_A9`], the wall
/// and the spread of a CTA's busy time, summed over the layers.
pub fn p4t_phases_a9(stamps: &[u64], n_layers: u32, grid: u32) -> Result<Vec<[f64; 4]>, String> {
    let n_bar = n_bar_a9(n_layers, 0) as usize;
    let events = 1 + 2 * n_bar;
    let g = grid as usize;
    if g == 0 || stamps.len() != events * g {
        return Err(format!("tree stamps (a9): {} values for {events} events of {g} CTAs", stamps.len()));
    }
    if stamps.contains(&0) {
        return Err("tree stamps (a9): a CTA wrote no clock at some event".into());
    }
    let at = |e: usize, c: usize| stamps[e * g + c] as f64;
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let kind = |w: usize| {
        if w == 1 {
            0
        } else if w + 2 >= n_bar {
            17 - (n_bar + 1 - w)
        } else {
            1 + (w - 2) % 13
        }
    };
    let np = P4T_PHASES_A9.len();
    let mut wall = vec![0.0f64; np];
    let mut busy = vec![vec![0.0f64; np]; g];
    for w in 1..=n_bar {
        let k = kind(w);
        let mut prev: Vec<f64> = (0..g).map(|c| at(2 * w - 2, c)).collect();
        let mut leave: Vec<f64> = (0..g).map(|c| at(2 * w, c)).collect();
        wall[k] += med(&mut leave) - med(&mut prev);
        for (c, b) in busy.iter_mut().enumerate() {
            b[k] += at(2 * w - 1, c) - at(2 * w - 2, c);
        }
    }
    Ok((0..np)
        .map(|k| {
            let mut v: Vec<f64> = busy.iter().map(|b| b[k]).collect();
            v.sort_by(f64::total_cmp);
            [wall[k], v[0], v[g / 2], v[g - 1]]
        })
        .collect())
}

/// The module text of arm a10 (`pers_4et_a10.cu`, a9 with 4E's prologues),
/// after [`p4t_a9_unit_text`], which keeps its bytes. Same waits as a9.
pub fn p4t_a10_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A10 1\n{}\n{}", p4t_a9_unit_text(tile_define), P4T_A10_PART.1)
}

/// The a10 symbol for a RoPE form.
pub fn p4t_a10_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a10_f0_t512",
        1 => "pers_4et_a10_f1_t512",
        2 => "pers_4et_a10_f2_t512",
        _ => return Err(format!("tree verify: no a10 kernel for RoPE form {form}")),
    })
}

/// The module text of arm a11 (`pers_4et_a11.cu`, a10 with the L2 prefetch
/// in its wait gaps), after [`p4t_a10_unit_text`], which keeps its bytes.
pub fn p4t_a11_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A11 1\n{}\n{}", p4t_a10_unit_text(tile_define), P4T_A11_PART.1)
}

/// The a11 symbol for a RoPE form.
pub fn p4t_a11_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a11_f0_t512",
        1 => "pers_4et_a11_f1_t512",
        2 => "pers_4et_a11_f2_t512",
        _ => return Err(format!("tree verify: no a11 kernel for RoPE form {form}")),
    })
}

/// The module text of arm a11i (`pers_4et_a11i.cu`, a11 with fine stamps),
/// after [`p4t_a11_unit_text`], which keeps its bytes.
pub fn p4t_a11i_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A11I 1\n{}\n{}", p4t_a11_unit_text(tile_define), P4T_A11I_PART.1)
}

/// The a11i symbol for a RoPE form.
pub fn p4t_a11i_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a11i_f0_t512",
        1 => "pers_4et_a11i_f1_t512",
        2 => "pers_4et_a11i_f2_t512",
        _ => return Err(format!("tree verify: no a11i kernel for RoPE form {form}")),
    })
}

/// `pers_4et_a11i.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A11I_PART: (&str, &str) = ("pers_4et_a11i.cu", include_str!("../kernels/pers_4et_a11i.cu"));

/// a11i's unit parts, in slot order (`pers_4et_a11i.cu`).
pub const P4T_IPARTS: [&str; 6] = ["units", "staging", "loop fast", "loop slow", "close wait", "to arrival"];

/// An a11i stamped launch's slots after the phase events, `[5][8][grid]`:
/// per row phase ([`P4T_ROW_PHASES`]) and part ([`P4T_IPARTS`]), the
/// median, minimum and maximum over the CTAs (units as counts, the rest in
/// nanoseconds, each CTA's sum over the layers).
pub fn p4t_iparts(tail: &[u64], grid: u32) -> Result<[[[f64; 3]; 6]; 5], String> {
    let g = grid as usize;
    if g == 0 || tail.len() != 40 * g {
        return Err(format!("tree a11i slots: {} values for 40 slots of {g} CTAs", tail.len()));
    }
    let mut out = [[[0.0f64; 3]; 6]; 5];
    for (ph, o) in out.iter_mut().enumerate() {
        for (p, x) in o.iter_mut().enumerate() {
            let mut v: Vec<f64> = tail[(8 * ph + p) * g..(8 * ph + p + 1) * g].iter().map(|&t| t as f64).collect();
            v.sort_by(f64::total_cmp);
            *x = [v[g / 2], v[0], v[g - 1]];
        }
    }
    Ok(out)
}

/// From a stamped a9-family launch's phase events: per wait class (after a
/// prologue: ln1, ln2, dn_in, norm; after a row phase; after a fold step) a
/// CTA's arrival-to-release time summed over the layers, median and maximum
/// over the CTAs; and each prologue's busy time on the CTAs that own rows
/// 0 .. rows - 1 (CTA r mod grid), the maximum over them. Nanoseconds.
pub fn p4t_wait_classes(stamps: &[u64], n_layers: u32, grid: u32, rows: u32) -> Result<([[f64; 2]; 3], [f64; 4]), String> {
    let n_bar = n_bar_a9(n_layers, 0) as usize;
    let g = grid as usize;
    if g == 0 || stamps.len() != (1 + 2 * n_bar) * g {
        return Err(format!("tree stamps (a9): {} values for {} events of {g} CTAs", stamps.len(), 1 + 2 * n_bar));
    }
    let name = |w: usize| -> &str {
        let k = if w == 1 {
            0
        } else if w + 2 >= n_bar {
            17 - (n_bar + 1 - w)
        } else {
            1 + (w - 2) % 13
        };
        P4T_PHASES_A9[k]
    };
    let at = |e: usize, c: usize| stamps[e * g + c] as f64;
    let mut cls = vec![[0.0f64; 3]; g];
    let mut pro = [0.0f64; 4];
    let mut pro_c = vec![[0.0f64; 4]; g];
    for w in 1..=n_bar {
        let n = name(w);
        let k = if n.ends_with("fold") {
            Some(2)
        } else if ["qkv", "o", "gateup", "down", "head"].contains(&n) {
            Some(1)
        } else if ["ln1", "ln2", "dn_in", "norm"].contains(&n) {
            Some(0)
        } else {
            None
        };
        let pk = ["ln1", "ln2", "dn_in", "norm"].iter().position(|p| *p == n);
        for c in 0..g {
            if let Some(k) = k {
                cls[c][k] += at(2 * w, c) - at(2 * w - 1, c);
            }
            if let Some(p) = pk {
                pro_c[c][p] += at(2 * w - 1, c) - at(2 * w - 2, c);
            }
        }
    }
    for pc in pro_c.iter().take((rows as usize).min(g)) {
        for (x, &y) in pro.iter_mut().zip(pc) {
            *x = x.max(y);
        }
    }
    let mut out = [[0.0f64; 2]; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut v: Vec<f64> = cls.iter().map(|x| x[k]).collect();
        v.sort_by(f64::total_cmp);
        *o = [v[g / 2], v[g - 1]];
    }
    Ok((out, pro))
}

/// The module text of the row-cap arms a10r8 and a10r16
/// (`pers_4et_a10rc.cu`, a10 for at most 8 or 16 rows a launch), after
/// [`p4t_a10_unit_text`], which keeps its bytes. `rows` 8 or 16: the
/// module emits that arm's symbols only.
pub fn p4t_a10rc_unit_text(tile_define: &str, rows: u32) -> String {
    format!("#define P4T_A10RC {rows}\n{}\n{}", p4t_a10_unit_text(tile_define), P4T_A10RC_PART.1)
}

/// The row-cap arms' symbols: `rows` 8 or 16, a RoPE form.
pub fn p4t_a10rc_symbol(rows: u32, form: u32) -> Result<&'static str, String> {
    Ok(match (rows, form) {
        (8, 0) => "pers_4et_a10r8_f0_t512",
        (8, 1) => "pers_4et_a10r8_f1_t512",
        (8, 2) => "pers_4et_a10r8_f2_t512",
        (16, 0) => "pers_4et_a10r16_f0_t512",
        (16, 1) => "pers_4et_a10r16_f1_t512",
        (16, 2) => "pers_4et_a10r16_f2_t512",
        _ => return Err(format!("tree verify: no row-cap kernel for {rows} rows at RoPE form {form}")),
    })
}

/// `pers_4et_a10rc.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A10RC_PART: (&str, &str) = ("pers_4et_a10rc.cu", include_str!("../kernels/pers_4et_a10rc.cu"));

/// The module text of the thread-count arms a10t384 and a10t256
/// (`pers_4et_a10t.cu`: a10's pass under `__launch_bounds__(threads, 1)`),
/// after [`p4t_a10_unit_text`], which keeps its bytes. `threads` 384 or 256.
pub fn p4t_a10t_unit_text(tile_define: &str, threads: u32) -> String {
    format!("#define P4T_A10T {threads}\n{}\n{}", p4t_a10_unit_text(tile_define), P4T_A10T_PART.1)
}

/// The thread-count arms' symbols.
pub fn p4t_a10t_symbol(threads: u32, form: u32) -> Result<&'static str, String> {
    Ok(match (threads, form) {
        (384, 0) => "pers_4et_a10_f0_t384",
        (384, 1) => "pers_4et_a10_f1_t384",
        (384, 2) => "pers_4et_a10_f2_t384",
        (256, 0) => "pers_4et_a10_f0_t256",
        (256, 1) => "pers_4et_a10_f1_t256",
        (256, 2) => "pers_4et_a10_f2_t256",
        _ => return Err(format!("tree verify: no a10 kernel at {threads} threads and RoPE form {form}")),
    })
}

/// The module text of arm a10t448, a10t320 or a10t384r16
/// (`pers_4et_a10t2.cu`): after [`p4t_a10_unit_text`] for the first two,
/// after [`p4t_a10rc_unit_text`] at 16 rows for the third; those keep their
/// bytes.
pub fn p4t_a10t2_unit_text(tile_define: &str, arm: &str) -> Result<String, String> {
    Ok(match arm {
        "a10t448" => format!("#define P4T_A10T2 448\n{}\n{}", p4t_a10_unit_text(tile_define), P4T_A10T2_PART.1),
        "a10t320" => format!("#define P4T_A10T2 320\n{}\n{}", p4t_a10_unit_text(tile_define), P4T_A10T2_PART.1),
        "a10t384r16" => format!("#define P4T_A10T2 38416\n{}\n{}", p4t_a10rc_unit_text(tile_define, 16), P4T_A10T2_PART.1),
        _ => return Err(format!("tree verify: no module for arm {arm}")),
    })
}

/// The symbols of arm a10t448, a10t320 or a10t384r16.
pub fn p4t_a10t2_symbol(arm: &str, form: u32) -> Result<&'static str, String> {
    Ok(match (arm, form) {
        ("a10t448", 0) => "pers_4et_a10_f0_t448",
        ("a10t448", 1) => "pers_4et_a10_f1_t448",
        ("a10t448", 2) => "pers_4et_a10_f2_t448",
        ("a10t320", 0) => "pers_4et_a10_f0_t320",
        ("a10t320", 1) => "pers_4et_a10_f1_t320",
        ("a10t320", 2) => "pers_4et_a10_f2_t320",
        ("a10t384r16", 0) => "pers_4et_a10r16_f0_t384",
        ("a10t384r16", 1) => "pers_4et_a10r16_f1_t384",
        ("a10t384r16", 2) => "pers_4et_a10r16_f2_t384",
        _ => return Err(format!("tree verify: no kernel for arm {arm} at RoPE form {form}")),
    })
}

/// `pers_4et_a10t2.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A10T2_PART: (&str, &str) = ("pers_4et_a10t2.cu", include_str!("../kernels/pers_4et_a10t2.cu"));

/// The module text of arm a12 (`pers_4et_a12.cu`: a10t384r16 with the
/// drafter's taps), after [`p4t_a10rc_unit_text`] at 16 rows, which keeps
/// its bytes.
pub fn p4t_a12_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A12 1\n{}\n{}", p4t_a10rc_unit_text(tile_define, 16), P4T_A12_PART.1)
}

/// The a12 symbol for a RoPE form (384 threads).
pub fn p4t_a12_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a12_f0_t384",
        1 => "pers_4et_a12_f1_t384",
        2 => "pers_4et_a12_f2_t384",
        _ => return Err(format!("tree verify: no a12 kernel for RoPE form {form}")),
    })
}

/// The module text of arm a12ip (step M1 of the memory axis): a12's pass on
/// the words 4E holds, read in place, and the install's load check
/// `p4t_ip_check`; [`p4t_a12_unit_text`] under `P4T_A12IP 1`, which emits
/// those symbols instead of a12's.
pub fn p4t_a12ip_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A12IP 1\n{}", p4t_a12_unit_text(tile_define))
}

/// The a12ip symbol for a RoPE form (384 threads).
pub fn p4t_a12ip_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a12ip_f0_t384",
        1 => "pers_4et_a12ip_f1_t384",
        2 => "pers_4et_a12ip_f2_t384",
        _ => return Err(format!("tree verify: no a12ip kernel for RoPE form {form}")),
    })
}

/// The module text of arm a12ipg (D1 of
/// docs/mesures/large-prefill-contexte-2026-11-11.txt): a12ip's pass whose
/// attention reads each split that ends at or below pos0 once for a group of
/// [`P4G_RG`] rows (`pers_4et_a12g.cu`), appended after
/// [`p4t_a12ip_unit_text`], which keeps its bytes. The module also carries
/// a12ip's symbols and its load check `p4t_ip_check`.
pub fn p4t_a12ipg_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A12IPG 1\n{}\n{}", p4t_a12ip_unit_text(tile_define), P4T_A12G_PART.1)
}

/// The a12ipg symbol for a RoPE form (384 threads).
pub fn p4t_a12ipg_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a12ipg_f0_t384",
        1 => "pers_4et_a12ipg_f1_t384",
        2 => "pers_4et_a12ipg_f2_t384",
        _ => return Err(format!("tree verify: no a12ipg kernel for RoPE form {form}")),
    })
}

/// The module text of arm a12ipg32 (design A of
/// docs/mesures/large-prefill-rangees-etude-2026-11-11.txt): a12ipg's pass
/// compiled at RC = 4, 32 rows a launch (`pers_4et_a12g32.cu`), appended after
/// [`p4t_a12ipg_unit_text`], which keeps its bytes. The module also carries
/// a12ip's and a12ipg's symbols and the load check `p4t_ip_check`.
pub fn p4t_a12ipg32_unit_text(tile_define: &str) -> String {
    format!("#define P4T_A12IPG32 1\n{}\n{}", p4t_a12ipg_unit_text(tile_define), P4T_A12G32_PART.1)
}

/// The a12ipg32 symbol for a RoPE form (384 threads).
pub fn p4t_a12ipg32_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_a12ipg32_f0_t384",
        1 => "pers_4et_a12ipg32_f1_t384",
        2 => "pers_4et_a12ipg32_f2_t384",
        _ => return Err(format!("tree verify: no a12ipg32 kernel for RoPE form {form}")),
    })
}

/// `pers_4et_a12g32.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A12G32_PART: (&str, &str) = ("pers_4et_a12g32.cu", include_str!("../kernels/pers_4et_a12g32.cu"));

// ---------------------------------------------------------------------------
// The attention tile's full-tile path (ft arms,
// docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt): a module text with
// `pers_attn_full.cuh` inserted before `pers_attn_tile` and a branch at the
// head of `pers_attn_tile` that runs it when the tile holds PERS_TT
// positions. A text edit of the parent module, so a12ip, a12ipg, a12ipg32
// and 4E v9 keep their bytes.
// ---------------------------------------------------------------------------

/// `pers_attn_full.cuh`, inserted by [`pers_ft_edit`] in the ft arms only.
pub const PERS_FT_PART: (&str, &str) = ("pers_attn_full.cuh", include_str!("../kernels/pers_attn_full.cuh"));

/// `pers_attn_tile`'s head in `llvq_pers.cuh`, the ft edit's anchor.
const PERS_FT_ANCHOR: &str = "__device__ __forceinline__ void pers_attn_tile(const float* qrow, float* row, bool own, const uint4* ks,
                                               const uint4* vs, u32 nv, float c, u32 lane)
{
";

/// The ft edit: (part, text, replacement), the text present exactly once in
/// `llvq_pers.cuh` and in every module that carries it once.
pub fn pers_ft_edit() -> (&'static str, String, String) {
    (
        "llvq_pers.cuh",
        PERS_FT_ANCHOR.to_string(),
        format!(
            "{}\n{PERS_FT_ANCHOR}    if (nv == PERS_TT) {{  // ft: a full tile, its positions with no guard between them\n        \
             pers_attn_tile_full(qrow, row, own, ks, vs, c, lane);\n        return;\n    }}\n",
            PERS_FT_PART.1
        ),
    )
}

/// A module text with the ft edit applied; refuses a text that does not hold
/// `pers_attn_tile`'s head exactly once.
pub fn pers_ft_apply(text: &str) -> Result<String, String> {
    let (_, from, to) = pers_ft_edit();
    let n = text.matches(from.as_str()).count();
    if n != 1 {
        return Err(format!("ft: {n} sites of pers_attn_tile's head, expected exactly one"));
    }
    Ok(text.replacen(from.as_str(), &to, 1))
}

/// The module text of a full-tile tree arm: "a12ipft", "a12ipgft" or
/// "a12ipg32ft", its parent's text with the ft edit. Same symbols as the parent.
pub fn p4t_ft_unit_text(tile_define: &str, arm: &str) -> Result<String, String> {
    let parent = match arm {
        "a12ipft" => p4t_a12ip_unit_text(tile_define),
        "a12ipgft" => p4t_a12ipg_unit_text(tile_define),
        "a12ipg32ft" => p4t_a12ipg32_unit_text(tile_define),
        _ => return Err(format!("ft: no tree arm {arm:?}; the arms are a12ipft, a12ipgft, a12ipg32ft")),
    };
    pers_ft_apply(&parent)
}

/// Rows a shared attention job of a12ipg serves (`P4G_RG`).
pub const P4G_RG: u32 = 4;

/// `p4g_area_floats` of `pers_4et_a12g.cu`: a12ip's area, or the grouped
/// attention's (two stages, then each of [`P4G_RG`] rows' unit states and q)
/// with the maxima, whichever is larger.
pub fn p4g_area_floats(nw: u32, rs: u32, widest: u32, groups: u32) -> u32 {
    let g = (16 * 1024 * 2 + P4G_RG * (4 * 4 * groups * 130 + 4 * groups * 128)).div_ceil(4) + 2 * P4T_RMAX;
    p4t_area_floats(nw, rs, widest, groups).max(g)
}

/// `pers_4et_a12g.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A12G_PART: (&str, &str) = ("pers_4et_a12g.cu", include_str!("../kernels/pers_4et_a12g.cu"));

/// u32 words of a projection in the served order, as the tree pass reads it
/// in place: Tetra `stride_u32` a row, int4 d_in / 8. The bytes a12ip reads
/// where a12 reads [`p4t_tiled_words`].
pub fn p4t_served_words(int4: bool, d_out: u32, stride_u32: u32, d_in: u32) -> u64 {
    d_out as u64 * if int4 { (d_in / 8) as u64 } else { stride_u32 as u64 }
}

/// `pers_4et_a12.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A12_PART: (&str, &str) = ("pers_4et_a12.cu", include_str!("../kernels/pers_4et_a12.cu"));

/// `pers_4et_a10t.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A10T_PART: (&str, &str) = ("pers_4et_a10t.cu", include_str!("../kernels/pers_4et_a10t.cu"));

/// `pers_4et_a11.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A11_PART: (&str, &str) = ("pers_4et_a11.cu", include_str!("../kernels/pers_4et_a11.cu"));

/// What a11 prefetches into L2 for one layer, per row phase (q/k/v, o,
/// gate/up, down with `int4_down` or Tetra): the phase's bytes in all and the
/// most one CTA asks in its gap, the tree-tiled streams of the units
/// `pers_4et.cu`'s deal gives it (16-row tiles, 48-column sub-steps of 192
/// bytes Tetra and 384 int4; whole lines, a stream rounded up to 128 bytes
/// at each end). A model of the 4B's kinds: q, k, gate, up Tetra; v, o int4.
#[allow(clippy::too_many_arguments)]
pub fn p4t_prefetch_volume(splits: [u32; 5], hidden: u32, inter: u32, d_ctx: u32, n_kv: u32, int4_down: bool, nw: u32, grid: u32) -> [(u64, u64); 4] {
    // (d_out, d_in, int4) of each projection of a phase
    let dkv = n_kv * 128;
    let phases: [Vec<(u32, u32, bool)>; 4] = [
        vec![(d_ctx, hidden, false), (dkv, hidden, false), (dkv, hidden, true)],
        vec![(hidden, d_ctx, true)],
        vec![(inter, hidden, false), (inter, hidden, false)],
        vec![(hidden, inter, int4_down)],
    ];
    let mut out = [(0u64, 0u64); 4];
    for (i, segs) in phases.iter().enumerate() {
        let s = splits[[0usize, 1, 2, 3][i]];
        let mut per_cta = vec![0u64; grid as usize];
        let mut total = 0u64;
        let mut before = 0u32;
        for &(d_out, d_in, int4) in segs {
            let (ntiles, nsub, sw) = (d_out / 16, d_in.div_ceil(48), if int4 { 384u64 } else { 192 });
            let nb = ntiles.div_ceil(nw);
            for lu in 0..nb * s {
                let (sl, blk) = (lu % s, lu / s);
                let (lo, hi) = ((sl * nsub / s) as u64, ((sl + 1) * nsub / s) as u64);
                let cta = ((before + lu) % grid) as usize;
                for w in 0..nw {
                    let tile = (blk * nw + w) as u64;
                    if tile >= ntiles as u64 || hi == lo {
                        continue;
                    }
                    let a = (tile * nsub as u64 + lo) * sw;
                    let b = a + (hi - lo) * sw;
                    let lines = (b.div_ceil(128) - a / 128) * 128;
                    per_cta[cta] += lines;
                    total += lines;
                }
            }
            before += nb * s;
        }
        out[i] = (total, per_cta.into_iter().max().unwrap_or(0));
    }
    out
}

/// `pers_4et_a10.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A10_PART: (&str, &str) = ("pers_4et_a10.cu", include_str!("../kernels/pers_4et_a10.cu"));

/// `pers_4et_a9.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A9_PART: (&str, &str) = ("pers_4et_a9.cu", include_str!("../kernels/pers_4et_a9.cu"));

/// `pers_4et_a8.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A8_PART: (&str, &str) = ("pers_4et_a8.cu", include_str!("../kernels/pers_4et_a8.cu"));

/// `pers_4et_a6.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A6_PART: (&str, &str) = ("pers_4et_a6.cu", include_str!("../kernels/pers_4et_a6.cu"));

/// `pers_4et_a5.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A5_PART: (&str, &str) = ("pers_4et_a5.cu", include_str!("../kernels/pers_4et_a5.cu"));

/// `pers_4et_a4.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A4_PART: (&str, &str) = ("pers_4et_a4.cu", include_str!("../kernels/pers_4et_a4.cu"));

/// `pers_4et_stream.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_STREAM_PART: (&str, &str) = ("pers_4et_stream.cu", include_str!("../kernels/pers_4et_stream.cu"));

/// The tree kernel's symbol for a RoPE form (512 threads only).
pub fn p4t_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4et_rows_f0_t512",
        1 => "pers_4et_rows_f1_t512",
        2 => "pers_4et_rows_f2_t512",
        _ => return Err(format!("tree verify: no kernel for RoPE form {form}")),
    })
}

/// The tree module's text: the 4E module of row form 7 and `pers_4et.cu`.
/// Its own module, so the served 4E module keeps its text.
pub fn p4t_unit_text(tile_define: &str) -> String {
    format!("{}{}\n{}", p4e_rows_define(7).expect("form 7 exists"), pers_unit_text(tile_define), P4T_PART.1)
}

/// `pers_4et.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_PART: (&str, &str) = ("pers_4et.cu", include_str!("../kernels/pers_4et.cu"));

/// The token kernel's symbol for a RoPE form and a block size.
pub fn p4e_symbol(form: u32, threads: u32) -> Result<&'static str, String> {
    Ok(match (form, threads) {
        (0, 768) => "pers_4e_tokens_f0",
        (1, 768) => "pers_4e_tokens_f1",
        (2, 768) => "pers_4e_tokens_f2",
        (0, 512) => "pers_4e_tokens_f0_t512",
        (1, 512) => "pers_4e_tokens_f1_t512",
        (2, 512) => "pers_4e_tokens_f2_t512",
        _ => return Err(format!("4E: no kernel for RoPE form {form} at {threads} threads")),
    })
}

/// The phase-stamp symbol for a RoPE form and a block size: the same body
/// and the same rows' variant as the token symbol of that size.
pub fn p4e_stamp_symbol(form: u32, threads: u32) -> Result<&'static str, String> {
    Ok(match (form, threads) {
        (0, 512) => "pers_4e_stamps_f0_t512",
        (1, 512) => "pers_4e_stamps_f1_t512",
        (2, 512) => "pers_4e_stamps_f2_t512",
        (0, 768) => "pers_4e_stamps_f0_t768",
        (1, 768) => "pers_4e_stamps_f1_t768",
        (2, 768) => "pers_4e_stamps_f2_t768",
        _ => return Err(format!("4E stamps: no stamp symbol for RoPE form {form} at {threads} threads")),
    })
}

/// Whether the 4E symbols of a block size keep the rows' next block or word
/// in flight: 512 threads yes, 768 no (its registers would spill).
pub fn p4e_rows_prefetch(threads: u32) -> bool {
    threads == 512
}

/// Stamp events of one 4E token (`pers_4e.cu`, STAMP = true): the
/// embedding's arrival and release, fifteen a layer in the order they happen
/// ([`P4E_LAYER`]), the head's end.
pub fn p4e_events(layers: u32) -> u32 {
    3 + 15 * layers
}

/// A layer's stamp events, offsets from its first (`pers_4e.cu`): q/k/v's
/// ln1 and rotation done, then each grid wait's arrival and release, with the
/// MLP's four inner points before its own wait.
pub const P4E_LAYER: [&str; 15] = [
    "q/k/v ln1+rotation done", "q/k/v arrival", "q/k/v release", "norm arrival", "norm release",
    "attention arrival", "attention release", "o arrival", "o release", "MLP ln2+rotation done",
    "MLP gate/up done", "MLP inner release", "MLP down input ready", "MLP arrival", "MLP release",
];

/// The parts of q/k/v and MLP the inner stamps split them into, as
/// [`p4e_phase_report`] names them; each is CTA work, no grid wall.
pub const P4E_SUBS: [&str; 7] = [
    "q/k/v: ln1+rotation", "q/k/v: rows", "MLP: ln2+rotation", "MLP: gate/up rows", "MLP: inner wait",
    "MLP: down input", "MLP: down rows",
];

/// The phases of a 4E token, in order, as [`p4e_phase_report`] names them.
pub const P4E_PHASES: [&str; 7] = ["embedding", "q/k/v", "norm/RoPE/append", "attention", "o", "MLP", "head"];

/// One phase of [`P4eReport`], summed over the layers of a token. Each is
/// the median over the tokens read, with its range.
#[derive(Clone, Debug, PartialEq)]
pub struct P4ePhase {
    pub name: &'static str,
    /// How far the grid's last release moved: the phase's share of the token.
    pub wall: (f64, f64, f64),
    /// The slowest CTA's work, from its own release to its own arrival.
    pub crit: (f64, f64, f64),
    /// The mean CTA's work.
    pub mean: (f64, f64, f64),
    /// The CTA that was slowest most often in this phase, and how often.
    pub crit_cta: (u32, f64),
}

/// [`p4e_phase_report`]'s reading of a stamped launch.
#[derive(Clone, Debug, PartialEq)]
pub struct P4eReport {
    pub phases: Vec<P4ePhase>,
    /// [`P4E_SUBS`], from the inner stamps: their `wall` is unread (zeros).
    pub subs: Vec<P4ePhase>,
    /// Release to release, the head's end of one token to the next's.
    pub token: (f64, f64, f64),
    pub tokens_read: usize,
    /// Smallest positive step between two stamps: the clock's grain.
    pub grain: u64,
}

/// Time per phase from the stamps of one launch of `n` tokens,
/// `[n][p4e_events(layers)][grid]`, in the clock's unit (ns on a card).
///
/// Refused, by name, when the stamps contradict the barriers they bracket: a
/// release before an arrival of the same wait on any CTA, or a CTA's stamps
/// going back within the launch. Token 0 has no previous end and is not read,
/// so `n` must be at least 2. The phases of a token start at the previous
/// release (the head's end for the embedding) and end at the arrival; the
/// head, whose W4 is inside it, ends at its release.
pub fn p4e_phase_report(stamps: &[u64], n: u32, layers: u32, grid: u32) -> Result<P4eReport, String> {
    let (ev, g) = (p4e_events(layers) as usize, grid as usize);
    if n < 2 || stamps.len() != n as usize * ev * g {
        return Err(format!("4E stamps: {} stamps for {n} tokens x {ev} events x {grid} CTAs (n >= 2)", stamps.len()));
    }
    let at = |t: usize, e: usize, c: usize| stamps[(t * ev + e) * g + c];
    for c in 0..g {
        let mut prev = 0u64;
        for t in 0..n as usize {
            for e in 0..ev {
                let v = at(t, e, c);
                if v == 0 || v < prev {
                    return Err(format!("4E stamps: CTA {c}, token {t}, event {e}: {v} after {prev}"));
                }
                prev = v;
            }
        }
    }
    // Every wait: its last arrival before its first release (the MLP's inner
    // wait too: its gate/up-done stamp is each CTA's arrival).
    let mut waits = vec![0usize];
    for l in 0..layers as usize {
        let b = 2 + 15 * l;
        waits.extend([b + 1, b + 3, b + 5, b + 7, b + 10, b + 13]);
    }
    for t in 0..n as usize {
        for &a in &waits {
            let last_arrival = (0..g).map(|c| at(t, a, c)).max().unwrap();
            let first_release = (0..g).map(|c| at(t, a + 1, c)).min().unwrap();
            if first_release < last_arrival {
                return Err(format!("4E stamps: token {t}, event {a}: a release at {first_release} before an arrival at {last_arrival}"));
            }
        }
    }
    let mut sorted: Vec<u64> = stamps.to_vec();
    sorted.sort_unstable();
    let grain = sorted.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0).min().unwrap_or(0);
    let (k, ks) = (P4E_PHASES.len(), P4E_SUBS.len());
    let (mut wall, mut crit, mut mean) = (vec![Vec::new(); k + ks], vec![Vec::new(); k + ks], vec![Vec::new(); k + ks]);
    let mut crit_ctas = vec![vec![0u32; g]; k + ks];
    let mut token = Vec::new();
    let head_end = ev - 1;
    for t in 1..n as usize {
        let (mut w, mut cr, mut mn) = (vec![0.0; k + ks], vec![0.0; k + ks], vec![0.0; k + ks]);
        // (kind, start event or None for the previous token's end, end event,
        // whether the kind has a grid wall: the phases do, the parts do not)
        let mut segs: Vec<(usize, Option<usize>, usize, bool)> = vec![(0, None, 0, true)];
        let mut rel = 1usize;
        for l in 0..layers as usize {
            let b = 2 + 15 * l;
            segs.push((1, Some(rel), b + 1, true));
            segs.push((k, Some(rel), b, false));
            segs.push((k + 1, Some(b), b + 1, false));
            segs.push((2, Some(b + 2), b + 3, true));
            segs.push((3, Some(b + 4), b + 5, true));
            segs.push((4, Some(b + 6), b + 7, true));
            segs.push((5, Some(b + 8), b + 13, true));
            for (i, e) in (b + 8..b + 13).enumerate() {
                segs.push((k + 2 + i, Some(e), e + 1, false));
            }
            rel = b + 14;
        }
        segs.push((6, Some(rel), head_end, true));
        for (kind, start, end, walled) in segs {
            let s_of = |c: usize| match start {
                Some(e) => at(t, e, c),
                None => at(t - 1, head_end, c),
            };
            if walled {
                // the grid's release that ends this phase: its end event's release
                let end_rel = if end == head_end { end } else { end + 1 };
                let grid_from = (0..g).map(s_of).max().unwrap();
                let grid_to = (0..g).map(|c| at(t, end_rel, c)).max().unwrap();
                w[kind] += (grid_to - grid_from) as f64;
            }
            let works: Vec<u64> = (0..g).map(|c| at(t, end, c) - s_of(c)).collect();
            // the slowest CTA, the lowest index on a tie
            let (ci, cw) = works.iter().enumerate().fold((0usize, 0u64), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            cr[kind] += cw as f64;
            mn[kind] += works.iter().sum::<u64>() as f64 / g as f64;
            crit_ctas[kind][ci] += 1;
        }
        for i in 0..k + ks {
            wall[i].push(w[i]);
            crit[i].push(cr[i]);
            mean[i].push(mn[i]);
        }
        let t_end = (0..g).map(|c| at(t, head_end, c)).max().unwrap();
        let t_prev = (0..g).map(|c| at(t - 1, head_end, c)).max().unwrap();
        token.push((t_end - t_prev) as f64);
    }
    let mut all: Vec<P4ePhase> = (0..k + ks)
        .map(|i| {
            let total: u32 = crit_ctas[i].iter().sum();
            let (cta, hits) = crit_ctas[i].iter().enumerate().fold((0usize, 0u32), |b, (c, &h)| if h > b.1 { (c, h) } else { b });
            P4ePhase {
                name: if i < k { P4E_PHASES[i] } else { P4E_SUBS[i - k] },
                wall: med_range(&wall[i]),
                crit: med_range(&crit[i]),
                mean: med_range(&mean[i]),
                crit_cta: (cta as u32, hits as f64 / total.max(1) as f64),
            }
        })
        .collect();
    let subs = all.split_off(k);
    Ok(P4eReport { phases: all, subs, token: med_range(&token), tokens_read: n as usize - 1, grain })
}

/// [`P4eReport`] as printed lines, the clock in ns: µs a token per phase.
pub fn p4e_report_lines(r: &P4eReport) -> Vec<String> {
    let us = |v: (f64, f64, f64)| format!("{:8.1} [{:.1}-{:.1}]", v.0 / 1e3, v.1 / 1e3, v.2 / 1e3);
    let mut out = vec![format!(
        "  phase stamps: {} tokens read, clock grain {} ns; us a token, median [range] over tokens",
        r.tokens_read, r.grain
    )];
    out.push(format!("    {:<18} {:>24} {:>24} {:>24} {:>6} {:>12}", "phase", "wall", "slowest CTA's work", "mean CTA's work", "share", "slowest CTA"));
    for p in &r.phases {
        out.push(format!(
            "    {:<18} {} {} {} {:5.1}% {:>5} ({:.0}%)",
            p.name,
            us(p.wall),
            us(p.crit),
            us(p.mean),
            100.0 * p.wall.0 / r.token.0,
            p.crit_cta.0,
            100.0 * p.crit_cta.1
        ));
    }
    out.push(format!("    {:<18} {}", "token", us(r.token)));
    out.push("    inside q/k/v and MLP (CTA work, no wall)".to_string());
    for p in &r.subs {
        out.push(format!("      {:<22} {} {} {:>5} ({:.0}%)", p.name, us(p.crit), us(p.mean), p.crit_cta.0, 100.0 * p.crit_cta.1));
    }
    out
}

/// Rows a `tv_tetra48_persall` work group covers: one per warp of its
/// 256-thread CTA, as `tv_tetra48_seg` lays its rows.
pub const PERSALL_GROUP_ROWS: u32 = 8;

/// The one-launch Tetra pass and its variants (`tetra48_persall.cu`): name,
/// symbol, whether the decoder tables sit in shared memory after the tile
/// (18,816 bytes more). The first is L1's kernel, byte for byte; the others
/// add the rows' prefetch (`pf`), the tables in shared memory (`ts`) and the
/// L2 prefetch of the CTA's next group (`l2`). Every one owes the bits.
pub const PERSALL_VARIANTS: [(&str, &str, bool); 5] = [
    ("tpersall", "tv_tetra48_persall", false),
    ("tpersall_pf", "tv_tetra48_persall_pf", false),
    ("tpersall_ts", "tv_tetra48_persall_ts", true),
    ("tpersall_ts_pf", "tv_tetra48_persall_ts_pf", true),
    ("tpersall_ts_pf_l2", "tv_tetra48_persall_ts_pf_l2", true),
];

/// The first group of every site and the total, sites in launch order.
pub fn persall_groups(d_outs: &[u32]) -> (Vec<u32>, u32) {
    let mut g0 = Vec::with_capacity(d_outs.len());
    let mut at = 0u32;
    for &d in d_outs {
        assert!(d.is_multiple_of(PERSALL_GROUP_ROWS), "d_out {d} is not a whole number of 8-row groups");
        g0.push(at);
        at += d / PERSALL_GROUP_ROWS;
    }
    (g0, at)
}

/// One `tv_tetra48_persall` site, as the kernel unpacks it.
#[allow(clippy::too_many_arguments)]
pub fn persall_site_words(
    words: u64,
    gscale: u64,
    gs_off: u64,
    rscale: u64,
    tail: u64,
    y: u64,
    nblocks: u32,
    tail_w: u32,
    stride_u32: u32,
    group0: u32,
    ngroups: u32,
) -> [u64; 8] {
    assert!(nblocks < 1 << 16 && tail_w < 1 << 16);
    [
        words,
        gscale,
        gs_off,
        rscale,
        tail,
        y,
        u64::from(nblocks) | (u64::from(tail_w) << 16) | (u64::from(stride_u32) << 32),
        u64::from(group0) | (u64::from(ngroups) << 32),
    ]
}

/// The f64 value of one Tetra row against `x`, and its `Σ|w·x|`: the
/// reference a row that missed the owed equality is judged against.
/// `block(j)` returns the lattice point and the gain bit of block j.
pub fn tetra_row_f64(
    nblocks: usize,
    block: impl Fn(usize) -> ([i32; 24], u32),
    centroids: [f32; 2],
    rscale: f32,
    tail: &[f64],
    x: &[f32],
) -> (f64, f64) {
    let (mut acc, mut mag) = (0.0f64, 0.0f64);
    for j in 0..nblocks {
        let (y, g) = block(j);
        let n2: i64 = y.iter().map(|&v| (v as i64) * (v as i64)).sum();
        if n2 == 0 {
            continue;
        }
        let scale = centroids[g as usize] as f64 / ((n2 as f64).sqrt());
        for k in 0..24 {
            let w = y[k] as f64 * scale * rscale as f64;
            acc += w * x[24 * j + k] as f64;
            mag += (w * x[24 * j + k] as f64).abs();
        }
    }
    for (i, &t) in tail.iter().enumerate() {
        let v = t * x[24 * nblocks + i] as f64;
        acc += v;
        mag += v.abs();
    }
    (acc, mag)
}

/// The row-loop microbenchmark's module: the persistent unit, then
/// `pers_trows.cu`, then plan-300 step 1's int4 rows (`pers_q4rows.cu`) and
/// counters (`pers_ctr.cu`). Its own module, so 4E's keeps its bytes.
pub fn trows_unit_text(tile_define: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}",
        pers_unit_text(tile_define),
        include_str!("../kernels/pers_trows.cu"),
        include_str!("../kernels/pers_q4rows.cu"),
        include_str!("../kernels/pers_ctr.cu")
    )
}

/// The verify-row microbenchmark's module (`pers_vrows.cu`): the row-loop
/// unit, then the verify rows. Its own module, so neither 4E's nor the row
/// loops' change bytes.
pub fn vrows_unit_text(tile_define: &str) -> String {
    format!("{}\n{}", trows_unit_text(tile_define), include_str!("../kernels/pers_vrows.cu"))
}

/// What a verify-row arm owes (`pers_vrows.cu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VrowsOwes {
    /// The served rows on each of its activations, bit for bit.
    Served,
    /// Nothing: a diagnostic (`pword`).
    Nothing,
    /// The rows of `i16` (K = 1) on each activation, bit for bit, and those
    /// within [`VROWS_I16_TOL`] of the served rows.
    I16,
}

/// The verify-row arms: name, symbol, activations a pass (K), what they owe.
/// `v2` is the control every speed is formed against; `k2` replicates
/// shaker-4e's second-token cost in f32; `i16` to `i16k4` are B1's row with
/// one decode for K activations.
pub const VROWS_ARMS: [(&str, &str, u32, VrowsOwes); 9] = [
    ("v2", "tv_vrows_v2", 1, VrowsOwes::Served),
    ("k2", "tv_vrows_k2", 2, VrowsOwes::Served),
    ("pword", "tv_vrows_pword", 1, VrowsOwes::Nothing),
    ("lean", "tv_vrows_lean", 1, VrowsOwes::Served),
    ("rot", "tv_vrows_rot", 1, VrowsOwes::Served),
    ("i16", "tv_vrows_i16k1", 1, VrowsOwes::I16),
    ("i16k2", "tv_vrows_i16k2", 2, VrowsOwes::I16),
    ("i16k3", "tv_vrows_i16k3", 3, VrowsOwes::I16),
    ("i16k4", "tv_vrows_i16k4", 4, VrowsOwes::I16),
];

/// Whether an arm reads the relabeled words (`tv_vrows_lean_relayout`).
pub fn vrows_lean(name: &str) -> bool {
    name == "lean" || name.starts_with("i16")
}

/// The integer dot's tolerance against the served rows, on the binary16 rows
/// the card returns: a row passes when it is within one binary16 ulp of the
/// served row, or within this share of its site's rms (a row near zero, where
/// an ulp is finer than the rounding). The activation's rounding moves a
/// matvec output by about 2e-5 of its scale (docs/recherche-debit-maths.md
/// §4), under the binary16 ulp of 4.9e-4 to 9.8e-4.
pub const VROWS_I16_TOL: f64 = 1e-3;

/// What the integer dot's rows say against the served rows.
#[derive(Clone, Debug, PartialEq)]
pub struct I16Check {
    pub rows: usize,
    /// Rows equal to the served row, bit for bit.
    pub same: usize,
    /// Rows one binary16 ulp away.
    pub one_ulp: usize,
    /// Rows two ulps or more away, within the rms bound.
    pub within_rms: usize,
    /// Rows that pass neither the ulp nor the rms bound.
    pub out: usize,
    /// The largest |row - served row| over its site's rms.
    pub worst: f64,
}

/// Binary16 bits as an integer ordered like the values (+0 and -0 both 0).
fn f16_ordered(h: u16) -> i32 {
    let m = i32::from(h & 0x7fff);
    if h & 0x8000 != 0 { -m } else { m }
}

/// The check of [`VROWS_I16_TOL`], site by site (`d_outs`, in order).
pub fn vrows_i16_check(got: &[u16], want: &[u16], d_outs: &[u32]) -> I16Check {
    assert_eq!(got.len(), want.len());
    assert_eq!(d_outs.iter().map(|&d| d as usize).sum::<usize>(), want.len());
    let mut c = I16Check { rows: want.len(), same: 0, one_ulp: 0, within_rms: 0, out: 0, worst: 0.0 };
    let mut at = 0usize;
    for &d in d_outs {
        let r = at..at + d as usize;
        let rms = (want[r.clone()].iter().map(|&h| f64::from(crate::attn_ref::h2f(h)).powi(2)).sum::<f64>() / f64::from(d)).sqrt();
        for (&g, &w) in got[r.clone()].iter().zip(&want[r]) {
            let ulp = (f16_ordered(g) - f16_ordered(w)).unsigned_abs();
            let e = (f64::from(crate::attn_ref::h2f(g)) - f64::from(crate::attn_ref::h2f(w))).abs() / rms.max(f64::MIN_POSITIVE);
            c.worst = c.worst.max(e);
            match ulp {
                0 => c.same += 1,
                1 => c.one_ulp += 1,
                _ if e <= VROWS_I16_TOL => c.within_rms += 1,
                _ => c.out += 1,
            }
        }
        at += d as usize;
    }
    c
}

/// `VROWS_TABLE_BYTES` of the kernel: rows, prefixes, branches, suffixes, GA,
/// GB, invnorm.
pub const VROWS_TABLE_BYTES: u32 = 16_384 + 128 + 2_048 + 128 + 256 + 512 + 128;

/// `vrows_area_bytes`: one int16 activation of `nb` blocks, its bf16 scales,
/// padded to 16, and 64 bytes of f32 tail.
pub fn vrows_area_bytes(nb: u32) -> u32 {
    ((50 * nb + 15) & !15) + 64
}

/// Dynamic shared bytes of a `tv_vrows_*` launch at the widest d_in.
pub fn vrows_smem_bytes(name: &str, k: u32, max_d_in: u32) -> u32 {
    let act = if name.starts_with("i16") {
        k * vrows_area_bytes(max_d_in / 24)
    } else if name == "k2" {
        4 * (9728 + max_d_in)
    } else {
        4 * max_d_in
    };
    SMEM_HEADER + VROWS_TABLE_BYTES + act
}

/// One `tv_vrows_*` site (`VROWS_SITE_WORDS`): the rows on activation a at
/// `y_f16[a]` (0 past the arm's K), no f32 output on the card.
#[allow(clippy::too_many_arguments)]
pub fn vrows_site_words(
    words: u64,
    gscale: u64,
    rscale: u64,
    tail_f16: u64,
    y_f16: [u64; 4],
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
    stride_u32: u32,
) -> [u64; 14] {
    [
        words,
        gscale,
        rscale,
        tail_f16,
        y_f16[0],
        y_f16[1],
        y_f16[2],
        y_f16[3],
        u64::from(d_out) | (u64::from(nblocks) << 32),
        u64::from(tail_w) | (u64::from(stride_u32) << 32),
        0,
        0,
        0,
        0,
    ]
}

/// The verify gate's rule (proofs/preregistration-spec-micro-2026-11-08.md):
/// the cost of an extra row, f = (t(i16kK) - t(i16)) / ((K - 1) t(v2)), the
/// median over K of the medians over rounds. At or under PASS the Tetra part
/// of the extra row leaves the 1.15-token verify of the study's rank 1 in
/// reach; over FAIL it cannot.
pub const VROWS_PASS: f64 = 0.15;
pub const VROWS_FAIL: f64 = 0.26;

/// The levers' bars against v2 (docs/recherche-debit-maths.md §8): lean and
/// rot ×1.05 each, B1's row (i16) ×1.15.
pub fn vrows_lever_bar(name: &str) -> Option<f64> {
    match name {
        "lean" | "rot" => Some(1.05),
        "i16" => Some(1.15),
        _ => None,
    }
}

/// The row-loop arms (`pers_trows.cu`): name, symbol, whether every row owes
/// the served bits, CTAs a SM. The occupancy arm `v2_o2` runs V2 at two CTAs a
/// SM on the sites of d_in 4,096 or less, beside `v2_s`, the same sites at one.
/// Then K4's five arms of plan-300 step 1, each against `v2`, and the row
/// lead's three (2026-11-07): `gb` (v2 with a grid barrier after every site),
/// `gbc` (gb with the rows claimed across the grid) and `v2r64` (v2's loop at
/// 64 registers, one CTA a SM).
pub const TROWS_ARMS: [(&str, &str, bool, u32); 18] = [
    ("v2", "tv_trows_v2", true, 1),
    ("v3", "tv_trows_v3", true, 1),
    ("ilp2", "tv_trows_ilp2", true, 1),
    ("v2u", "tv_trows_v2u", true, 1),
    ("l1w", "tv_trows_l1w", false, 1),
    ("bcast", "tv_trows_bcast", false, 1),
    ("xbcast", "tv_trows_xbcast", false, 1),
    ("v2_s", "tv_trows_v2", true, 1),
    ("v2_o2", "tv_trows_v2_o2", true, 2),
    ("k2", "tv_trows_k2", true, 1),
    ("nc", "tv_trows_nc", true, 1),
    ("nc256", "tv_trows_nc256", true, 1),
    ("cpa", "tv_trows_cpa", true, 1),
    ("tile", "tv_trows_tile", true, 1),
    ("bal", "tv_trows_bal", true, 1),
    ("gb", "tv_trows_gb", true, 1),
    ("gbc", "tv_trows_gbc", true, 1),
    ("v2r64", "tv_trows_v2_o2", true, 1),
];

/// The row lead's bar (proofs/preregistration-rows-grid-2026-11-07.md): the
/// median of gbc's round-by-round speed over gb that licenses a 4E form with
/// rows claimed across the grid; under [`ROWS_GRID_DEAD`] the claim is dead.
pub const ROWS_GRID_LEVER: f64 = 1.05;
pub const ROWS_GRID_DEAD: f64 = 1.02;

/// Does arm `name` read the counters (`x2` as u32: barrier, abort, a cursor a
/// site) that the host zeroes before each launch?
pub fn trows_counters(name: &str) -> bool {
    name == "gb" || name == "gbc"
}

/// The CTAs' spread over a launch's sites: the sum over sites of the slowest
/// CTA's busy cycles over the sum of the mean CTA's (1.0 when every CTA takes
/// as long). Inside 4E a grid barrier ends each phase, so the slowest CTA of a
/// phase sets its wall.
pub fn trows_cta_spread(stamps: &[u32], nsites: usize, grid: u32, nw: u32) -> f64 {
    let (g, w1) = (grid as usize, (nw + 1) as usize);
    assert_eq!(stamps.len(), nsites * g * w1, "stamps of {nsites} sites x {grid} CTAs x {w1}");
    let (mut slow, mut mean) = (0f64, 0f64);
    for s in 0..nsites {
        let cta = |c: usize| f64::from(stamps[(s * g + c) * w1 + nw as usize]);
        slow += (0..g).map(cta).fold(0f64, f64::max);
        mean += (0..g).map(cta).sum::<f64>() / g as f64;
    }
    if mean > 0.0 { slow / mean } else { 0.0 }
}

/// K4's bar (plan-300 §4, step 1): the speed over `v2` alone, median of the
/// round-by-round ratios, that licenses a 4E port. Under [`K4_DEAD`] an arm is
/// dead; between the two it is reported and is not a lever.
pub const K4_LEVER: f64 = 1.15;
pub const K4_DEAD: f64 = 1.10;

/// CPA's ring: three slots of 192 bytes a warp (`TROWS_CPA_WARP`), after the
/// widest activation, where K2 stages its second one.
pub fn trows_cpa_smem_bytes(warps: u32) -> u32 {
    SMEM_HEADER + TABLE_BYTES + 4 * 9728 + warps * 3 * 192
}

/// Whether CPA can stage a row stride: its 16-byte chunks must tile the row.
pub fn trows_cpa_ok(stride_u32: u32) -> bool {
    stride_u32.is_multiple_of(4)
}

/// TILE's row stride in u32 for `nblocks` blocks (`trows_tile_stride`):
/// 192 u32 a full tile of 128 blocks, then 6 bytes a block left, padded to 16.
pub fn trows_tile_stride(nblocks: u32) -> u32 {
    let nt = nblocks >> 7;
    let rem = nblocks - (nt << 7);
    192 * nt + (6 * rem).div_ceil(16) * 4
}

/// The served stream's row stride in u32: 6 bytes a block, padded to 8.
pub fn tetra_stride(nblocks: u32) -> u32 {
    (6 * nblocks).div_ceil(8) * 2
}

/// The int4 arms of K11 (`pers_q4rows.cu`): name, symbol, whether every row
/// owes the served bits. The served `tv_q4_hh` (84 launches) is the
/// reference of rows and time, and runs beside them.
pub const Q4ROWS_ARMS: [(&str, &str, bool); 4] = [
    ("q4", "tv_q4rows_q4", true),
    ("q4nc", "tv_q4rows_nc", true),
    ("q4t", "tv_q4rows_tile", true),
    ("q4read", "tv_q4rows_read", false),
];

/// K11's bar: `q4nc` or `q4t` over `q4`, median of the round-by-round ratios.
pub const K11_LEVER: f64 = 1.10;

/// The 84 int4 records of the served 4B (`configs/qwen3-4b-tetra-e4-4e-v6.json`
/// reads v and o int4 in every layer, down in layers 12 to 23), in served
/// order: (d_out, d_in, layer, projection).
pub fn q4rows_sites() -> Vec<(u32, u32, u32, &'static str)> {
    let mut v = Vec::with_capacity(84);
    for l in 0..36u32 {
        v.push((1024, 2560, l, "v"));
        v.push((2560, 4096, l, "o"));
        if (12..24).contains(&l) {
            v.push((2560, 9728, l, "down"));
        }
    }
    v
}

/// Bytes one pass over int4 records reads: the words (4 bits a weight) and a
/// binary16 scale and bias a group of 128.
pub fn q4rows_bytes(sites: &[(u32, u32, u32, &str)]) -> u64 {
    sites.iter().map(|&(o, i, _, _)| u64::from(o) * (u64::from(i) / 2 + 4 * u64::from(i / 128))).sum()
}

/// One `tv_q4rows_*` site (`Q4ROWS_SITE_WORDS`); no f32 output on the card.
pub fn q4rows_site_words(words: u64, scales: u64, biases: u64, y_f16: u64, d_out: u32, d_in: u32) -> [u64; 6] {
    assert!(d_in.is_multiple_of(256), "a row is whole steps of 32 words: d_in {d_in}");
    [words, scales, biases, y_f16, u64::from(d_out) | (u64::from(d_in) << 32), 0]
}

/// Dynamic shared bytes of a `tv_q4rows_*` launch: header and the f32
/// staging of the widest activation.
pub fn q4rows_smem_bytes(max_d_in: u32) -> u32 {
    SMEM_HEADER + 4 * max_d_in
}

/// The counter kernels of K3b (`pers_ctr.cu`): name, symbol. `work` is the
/// control of `split`, not a hop.
pub const CTR_ARMS: [(&str, &str); 6] = [
    ("bar", "tv_ctr_bar"),
    ("tight", "tv_ctr_tight"),
    ("rel", "tv_ctr_rel"),
    ("ctr8", "tv_ctr_8"),
    ("split", "tv_ctr_split"),
    ("work", "tv_ctr_work"),
];

/// K3b's bar: a counter form under `bar` by this many microseconds a hop.
pub const K3B_GAIN_US: f64 = 0.3;

/// CTR8's group size and the counters it needs at a grid (one 32-u32 line
/// each, `CTR_LINE`).
pub const CTR_GROUP: u32 = 8;
pub fn ctr_groups(grid: u32) -> u32 {
    grid.div_ceil(CTR_GROUP)
}

/// What one launch of `n` hops leaves in the counters of `arm` from zero, a
/// line apart (`CTR_LINE` = 32 u32): the host's check that every CTA ran
/// every hop.
pub fn ctr_expected(arm: &str, grid: u32, n: u32) -> Vec<u32> {
    match arm {
        "ctr8" => (0..ctr_groups(grid)).map(|p| n * (grid - p * CTR_GROUP).min(CTR_GROUP)).collect(),
        "work" => vec![grid],
        _ => vec![n * grid],
    }
}

/// Whether an arm runs only the sites of d_in 4,096 or less.
pub fn trows_small_only(name: &str) -> bool {
    name == "v2_s" || name == "v2_o2"
}

/// One `tv_trows_*` site, as the kernel unpacks it (`TROWS_SITE_WORDS`); no
/// f32 output (words 7 and 9 are 0), as on every card launch. `y2_f16`: K2's
/// rows on the second activation, 0 for every other arm.
#[allow(clippy::too_many_arguments)]
pub fn trows_site_words(
    words: u64,
    gscale: u64,
    rscale: u64,
    tail_f16: u64,
    y_f16: u64,
    y2_f16: u64,
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
    stride_u32: u32,
) -> [u64; 10] {
    assert!(tail_w == 8 || tail_w == 16, "the stream reads the tail as one or two 16-byte chunks: tail_w {tail_w}");
    [
        words,
        gscale,
        rscale,
        tail_f16,
        y_f16,
        u64::from(d_out) | (u64::from(nblocks) << 32),
        u64::from(tail_w) | (u64::from(stride_u32) << 32),
        0,
        y2_f16,
        0,
    ]
}

/// Dynamic shared bytes of a `tv_trows_*` launch: 4E's header, tables and
/// natural staging of the widest activation. At d_in 9,728 it is 4E's 57,984.
pub fn trows_smem_bytes(max_d_in: u32) -> u32 {
    SMEM_HEADER + TABLE_BYTES + 4 * max_d_in
}

/// K2's dynamic shared bytes: two activations, the second at `TROWS_X2_AT` =
/// 9,728 floats after the first. 96,896 B, under the 101,376 a CTA may opt
/// into on sm_89.
pub fn trows_k2_smem_bytes() -> u32 {
    SMEM_HEADER + TABLE_BYTES + 2 * 4 * 9728
}

/// What the stamps of one `tv_trows_*` launch say about one class of sites
/// (one d_in), summed over its sites and CTAs.
#[derive(Clone, Debug, PartialEq)]
pub struct TrowsClass {
    pub d_in: u32,
    pub sites: usize,
    /// Warp-steps: a warp's rows times ceil(nblocks / 32), summed.
    pub steps: u64,
    /// Busy cycles of the warps, summed.
    pub warp_cycles: u64,
    /// Cycles of the CTAs (to the barrier after the slowest warp), summed.
    pub cta_cycles: u64,
    /// Warps with at least one row, summed over sites and CTAs.
    pub busy_warps: u64,
}

impl TrowsClass {
    /// Cycles a busy warp spends on one step (32 blocks of one row).
    pub fn cycles_per_step(&self) -> f64 {
        self.warp_cycles as f64 / self.steps as f64
    }
    /// The warps' mean busy time over the CTA's time: 1 when every warp of
    /// every CTA finishes together.
    pub fn balance(&self, nw: u32) -> f64 {
        self.warp_cycles as f64 / (self.cta_cycles as f64 * f64::from(nw))
    }
}

/// Rows a warp takes from a CTA's slice [lo, hi), `pers_warp_rows` on the host.
pub fn trows_warp_rows(lo: u32, hi: u32, warp: u32, nw: u32) -> u32 {
    if hi > lo + warp {
        (hi - lo - warp).div_ceil(nw)
    } else {
        0
    }
}

/// Fold the stamps of one launch, `stamps[(s x grid + c) x (nw + 1) + w]` (a
/// warp's cycles, then the CTA's at w = nw), over `sites` = (d_out, d_in), into
/// one class a d_in in first-seen order.
pub fn trows_classes(stamps: &[u32], sites: &[(u32, u32)], grid: u32, nw: u32) -> Vec<TrowsClass> {
    let mut out: Vec<TrowsClass> = Vec::new();
    let w1 = (nw + 1) as usize;
    assert_eq!(stamps.len(), sites.len() * grid as usize * w1, "stamps of {} sites x {grid} CTAs x {w1}", sites.len());
    for (s, &(d_out, d_in)) in sites.iter().enumerate() {
        let nit = u64::from((d_in / 24).div_ceil(32));
        let k = match out.iter().position(|c| c.d_in == d_in) {
            Some(k) => k,
            None => {
                out.push(TrowsClass { d_in, sites: 0, steps: 0, warp_cycles: 0, cta_cycles: 0, busy_warps: 0 });
                out.len() - 1
            }
        };
        let cl = &mut out[k];
        cl.sites += 1;
        for c in 0..grid {
            let lo = (u64::from(d_out) * u64::from(c) / u64::from(grid)) as u32;
            let hi = (u64::from(d_out) * u64::from(c + 1) / u64::from(grid)) as u32;
            let at = (s * grid as usize + c as usize) * w1;
            for w in 0..nw {
                let rows = trows_warp_rows(lo, hi, w, nw);
                if rows > 0 {
                    cl.steps += u64::from(rows) * nit;
                    cl.warp_cycles += u64::from(stamps[at + w as usize]);
                    cl.busy_warps += 1;
                }
            }
            cl.cta_cycles += u64::from(stamps[at + nw as usize]);
        }
    }
    out
}

/// The SM clock of a launch in MHz, from `ends[4c..4c + 4]` = (clock64 at
/// start, globaltimer at start, clock64 at end, globaltimer at end) of every
/// CTA: total cycles over total nanoseconds.
pub fn trows_clock_mhz(ends: &[u64]) -> f64 {
    let (mut cyc, mut ns) = (0u64, 0u64);
    for e in ends.chunks_exact(4) {
        cyc += e[2].wrapping_sub(e[0]);
        ns += e[3].wrapping_sub(e[1]);
    }
    1e3 * cyc as f64 / ns as f64
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_tree_pass_fits_the_l40s_at_the_three_sizes() {
        // 512 threads, 32 staged rows: under the L40S's 101,376 B with the
        // header and the tables, at the 4B, 8B and 14B widths
        for (hidden, inter, d_ctx, groups) in [(2560u32, 9728u32, 4096u32, 4u32), (4096, 12288, 4096, 4), (5120, 17408, 5120, 5)] {
            let area = p4t_area_floats(16, 32, hidden.max(inter).max(d_ctx), groups);
            assert!(SMEM_HEADER + TABLE_BYTES + 4 * area <= 101_376, "{hidden}: {area} floats");
        }
        assert_eq!(p4t_area_floats(16, 32, 9728, 4), 17_728);
    }

    #[test]
    fn the_tree_splits_are_one_unit_a_cta_and_never_depend_on_rows() {
        // the 4B on 142 SMs: q/k/v 24 blocks, o 10, gate/up 76, down 10, head 594
        assert_eq!(p4t_splits(2560, 9728, 4096, 8, 151_936, 16, 142), [6, 14, 2, 14, 1]);
        assert_eq!(p4t_split(160, 16, 96, 142), 1, "never past half the sub-steps");
        assert!(p4t_fpart_floats([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936) <= 2_000_000);
        assert_eq!(p4t_ftickets(2560, 9728, 4096, 8, 151_936), 9496);
        assert_eq!(n_bar_4et(36, 0), 327);
    }

    #[test]
    fn the_tree_phases_sum_the_layers() {
        let (layers, g) = (2u32, 3usize);
        let n_bar = n_bar_4et(layers, 0) as usize;
        let events = 1 + 2 * n_bar;
        // leaving wait w at 1000 (w + 1) + c, reaching it 10 (c + 1) earlier
        let st: Vec<u64> = (0..events)
            .flat_map(|e| {
                (0..g).map(move |c| {
                    let w = e.div_ceil(2) as u64;
                    if e % 2 == 1 { 1000 * (w + 1) + c as u64 - 10 * (c as u64 + 1) } else { 1000 * (w + 1) + c as u64 }
                })
            })
            .collect();
        let p = p4t_phases(&st, layers, g as u32).unwrap();
        assert_eq!(p[0][0], 1000.0);
        assert_eq!(p[1][0], 2000.0, "ln1 twice");
        assert_eq!(p[10][0], 1000.0);
        assert_eq!(p[11][0], 1000.0);
        assert_eq!(p.iter().map(|x| x[0]).sum::<f64>(), 1000.0 * n_bar as f64);
        // busy: 1000 - 10 (c + 1) a wait, CTA 0 the longest
        assert_eq!(p[0][3], 990.0);
        assert_eq!(p[0][1], 970.0);
        assert_eq!(p[1][2], 2.0 * 980.0);
        assert!(p4t_phases(&st[1..], layers, g as u32).is_err());
    }

    #[test]
    fn the_n1_plan_fills_the_grid_at_the_4b() {
        // S1's plan at the 4B on 142 SMs: q/k/v 144 units in two waves, gate/up 152
        let s1 = p4t_plan([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936, 16, 142);
        assert_eq!((s1[0].0, s1[0].1), (144, 2));
        assert_eq!((s1[2].0, s1[2].1), (152, 2));
        let a1 = [11, 14, 7, 14, 1];
        assert_eq!(p4t_splits_balanced(a1, 2560, 9728, 4096, 8, 151_936, 16, 142), a1, "a1 reads 90 % or more");
        let p = p4t_plan(a1, 2560, 9728, 4096, 8, 151_936, 16, 142);
        assert_eq!((p[0].0, p[0].1), (264, 2));
        assert_eq!((p[2].0, p[2].1), (532, 4));
        assert!(p[0].2 >= 0.9 && p[2].2 >= 0.9);
        // a poor proposal is replaced by the best fill
        assert_eq!(p4t_splits_balanced([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936, 16, 142)[0], 11);
        assert_eq!(p4t_tiled_words(false, 4096, 2560), 256 * 54 * 48);
        assert_eq!(p4t_tiled_words(true, 151_936, 2560), 9496 * 54 * 96);
    }

    #[test]
    fn the_n2_stream_plan_fits_the_area_and_fills_the_grid() {
        // the 4B on 142 SMs, 16 warps: a2's area, so a2's shared bytes
        let area = p4t_area_floats(16, 32, 9728, 4);
        let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
        assert_eq!((area, smem), (17_728, 89_984));
        assert!(smem <= 101_376);
        let cap = p4t_stream_cap(area, 16);
        assert_eq!(cap, 13, "32 rows of 24 x 13 + 4 words beside the A tiles");
        assert!(16 * 16 * 28 + 32 * (24 * cap + 4) + 2 * 32 <= area && 16 * 16 * 28 + 32 * (24 * (cap + 1) + 4) + 2 * 32 > area);
        let (h, i, d, kv, v) = (2560, 9728, 4096, 8, 151_936);
        let s = p4t_splits_stream(h, i, d, kv, v, 16, 142, cap).unwrap();
        let names = ["q/k/v", "o", "gate/up", "down", "head"];
        let d_ins = [h, d, h, i, h];
        let floor: Vec<u32> = d_ins.iter().map(|&x| (1..=32).find(|&s| p4t_slice_steps(x, s) <= cap).unwrap()).collect();
        assert_eq!(floor, [5, 7, 5, 16, 5]);
        let p = p4t_plan(s, h, i, d, kv, v, 16, 142);
        for k in 0..5 {
            eprintln!(
                "4B, 142 CTAs: {} S {} (floor {}), units {}, waves {}, fill {:.1} %, at most {} sub-steps a slice",
                names[k], s[k], floor[k], p[k].0, p[k].1, 100.0 * p[k].2, p4t_slice_steps(d_ins[k], s[k])
            );
            assert!(s[k] >= floor[k] && p4t_slice_steps(d_ins[k], s[k]) <= cap);
        }
        assert_eq!(s, [11, 14, 7, 28, 5]);
        let fp = p4t_fpart_floats(s, h, i, d, kv, v);
        eprintln!("slice partials {fp} floats, {:.1} MB", 4.0 * fp as f64 / 1e6);
        assert_eq!(fp, 5 * 32 * 151_936);
        // the host twin: 3 CTAs of 2 warps, its area the widest row's
        let ht = p4t_area_floats(2, 32, 9728, 4);
        let hcap = p4t_stream_cap(ht, 2);
        assert_eq!(hcap, 18);
        let hs = p4t_splits_stream(h, i, d, kv, 1024, 2, 3, hcap).unwrap();
        let hp = p4t_plan(hs, h, i, d, kv, 1024, 2, 3);
        for k in 0..5 {
            eprintln!(
                "host twin, 3 CTAs of 2 warps: {} S {}, units {}, waves {}, fill {:.1} %, at most {} sub-steps a slice",
                names[k], hs[k], hp[k].0, hp[k].1, 100.0 * hp[k].2, p4t_slice_steps(d_ins[k], hs[k])
            );
        }
        assert_eq!(hs, [4, 6, 4, 13, 4]);
        assert!(p4t_splits_stream(h, i, d, kv, v, 16, 142, 3).is_err(), "the down needs S > 32 at 3 sub-steps a slice");
    }

    #[test]
    fn the_stream_body_is_the_tree_body() {
        // pers_4et_stream.cu carries pers_4et.cu's pass with three
        // substitutions, and nothing else of it changed
        let src = P4T_PART.1;
        let a = src.find("template <int FORM, bool TILED>\n__device__ __forceinline__ void pers_4et_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let b = a + src[a..].find(end).unwrap() + end.len();
        let body = src[a..b]
            .replacen("template <int FORM, bool TILED>\n", "template <int FORM>\n", 1)
            .replacen("void pers_4et_body(", "void pers_4et_stream_body(", 1)
            .replace("p4t_phase<TILED>(", "p4t_phase_stream(");
        assert!(!body.contains("TILED"));
        assert_eq!(P4T_STREAM_PART.1.matches(body.as_str()).count(), 1, "the stream body drifted from pers_4et_body");
        let t = p4t_stream_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_tiled_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_stream_symbol(f).unwrap()).count(), 1);
            assert_eq!(p4t_unit_text(SERVED_DEFINES_SM89).matches(p4t_stream_symbol(f).unwrap()).count(), 0);
        }
        // the module texts of a0/a1 and a2, and a3's (the bench builds them from the runtime's tile define)
        for (n, x) in [("rows", p4t_unit_text(SERVED_DEFINES_SM89)), ("tiled", p4t_tiled_unit_text(SERVED_DEFINES_SM89)), ("stream", t)] {
            eprintln!("tree module {n} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", x.len(), crate::sha256::sha256_hex(x.as_bytes()));
            // LLVQ_P4T_DUMP=<dir>: the texts NVRTC compiles, for an offline nvcc SASS probe
            if let Ok(dir) = std::env::var("LLVQ_P4T_DUMP") {
                std::fs::write(std::path::Path::new(&dir).join(format!("tree_{n}.cu")), &x).unwrap();
            }
        }
    }

    #[test]
    fn the_a4_body_is_the_stream_body() {
        // pers_4et_a4.cu carries pers_4et_stream.cu's unit, deal and pass with
        // the substitutions below, and nothing else of them changed
        let src = P4T_STREAM_PART.1;
        let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + src[a..].find("// p4t_phase's deal, the stream unit.").unwrap();
        let unit = src[a..b]
            .replace("p4t_unit_stream(", "p4t_unit_a4(")
            .replace(
                "                                                u32* area, unsigned long long* cmax)\n",
                "                                                u32* area, unsigned long long* cmax, unsigned long long* tm)\n",
            )
            .replace("    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);\n", "    P4T_A4_MARK(0);\n    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);\n")
            .replace("    p4t_cp_wait<0u>();\n    __syncthreads();\n", "    p4t_cp_wait<0u>();\n    __syncthreads();\n    P4T_A4_MARK(1);\n")
            .replace(
                "    __syncthreads();\n    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n",
                "    __syncthreads();\n    P4T_A4_MARK(2);\n    if (has) p4t_store_a4(g, L, A, sl, acc, wrow, lane, cmax);\n    P4T_A4_MARK(3);\n",
            );
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let dn = "            for (u32 i = threadIdx.x; i < I; i += blockDim.x) {\n                const unsigned short a = exact_silu_mul(M.silu, __ldcg(gr + i), __ldcg(ur + i));\n                if (Y.tetra_down) {\n                    const float f = h2f(a);\n                    S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;\n                } else {\n                    dst[i] = a;\n                }\n            }\n";
        assert_eq!(src[c..d].matches(dn).count(), 1);
        let body = src[c..d]
            .replacen("void pers_4et_stream_body(", "void pers_4et_a4_body(", 1)
            .replace("p4t_phase_stream(", "p4t_phase_a4(")
            .replace("p4t_ln(", "p4t_ln_a4(")
            .replace("p4t_out(", "p4t_out_a4(")
            .replace(dn, "            p4t_dn_in_a4(M, Y.tetra_down, gr, ur, dst, S, I);\n");
        let both = format!("{unit}{body}");
        assert_eq!(P4T_A4_PART.1.matches(both.as_str()).count(), 1, "the a4 pass drifted from the stream pass");
        let t = p4t_a4_unit_text(SERVED_DEFINES_SM89, true, true);
        assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)), "a3's text inside a4's, unchanged");
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a4_symbol(f).unwrap()).count(), 1);
            assert_eq!(p4t_stream_unit_text(SERVED_DEFINES_SM89).matches(p4t_a4_symbol(f).unwrap()).count(), 0);
        }
        eprintln!("tree module a4 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }

    #[test]
    fn the_a5_body_is_the_a4_body() {
        // pers_4et_a5.cu carries pers_4et_a4.cu's pass with three prefetch
        // calls inserted, and nothing else of it changed
        let src = P4T_A4_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a4_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let mut body = src[c..d].replacen("void pers_4et_a4_body(", "void pers_4et_a5_body(", 1);
        for (x, y) in [
            ("        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;\n", "        p4t_pf_qkv(Y, L);\n"),
            ("        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));\n", "        p4t_pf_o_gu(Y, L, H, dq);\n"),
            ("            p4t_phase_a4(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);\n", "            p4t_pf_down(Y, L, H);\n"),
        ] {
            assert_eq!(body.matches(x).count(), 1, "{x}");
            body = body.replacen(x, &format!("{x}{y}"), 1);
        }
        assert_eq!(P4T_A5_PART.1.matches(body.as_str()).count(), 1, "the a5 pass drifted from the a4 pass");
        let t = p4t_a5_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_a4_unit_text(SERVED_DEFINES_SM89, true, true)), "a4's text inside a5's, unchanged");
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a5_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a5 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }

    #[test]
    fn the_a6_family_is_the_stream_body() {
        // pers_4et_a6.cu carries a3's pass three times with the substitutions below
        let src = P4T_STREAM_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let body = &src[c..d];
        let dn = "        for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw";
        assert_eq!(body.matches(dn).count(), 1);
        let b6 = body.replacen("void pers_4et_stream_body(", "void pers_4et_a6_body(", 1).replacen(
            dn,
            "        if (!Y.tetra_down && (I & 7u) == 0u) p4t_dn_in_grid(M, L, R, I);\n        else for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw",
            1,
        );
        let mut b7 = b6.replacen("void pers_4et_a6_body(", "void pers_4et_a7_body(", 1);
        for (x, y) in [
            ("        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;\n", "        p4t_pf_qkv(Y, L);\n"),
            ("        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));\n", "        p4t_pf_o_gu(Y, L, H, dq);\n"),
            ("            p4t_phase_stream(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);\n", "            p4t_pf_down(Y, L, H);\n"),
        ] {
            assert_eq!(b7.matches(x).count(), 1, "{x}");
            b7 = b7.replacen(x, &format!("{x}{y}"), 1);
        }
        let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + src[a..].find("// ---- the pass:").unwrap();
        let uf = src[a..b]
            .replace("p4t_unit_stream(", "p4t_unit_floor(")
            .replace("p4t_phase_stream(", "p4t_phase_floor(")
            .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_floor(g, L, A, sl, acc, wrow, lane, cmax);\n");
        let bf = body.replacen("void pers_4et_stream_body(", "void pers_4et_a3f_body(", 1).replace("p4t_phase_stream(", "p4t_phase_floor(");
        for (n, x) in [("a6", b6.clone()), ("a7", b7), ("a3f", format!("{uf}{bf}"))] {
            assert_eq!(P4T_A6_PART.1.matches(x.as_str()).count(), 1, "the {n} pass drifted from the stream pass");
            let t = p4t_a6_unit_text(n, SERVED_DEFINES_SM89).unwrap();
            for f in 0..3 {
                assert_eq!(t.matches(p4t_a6_symbol(n, f).unwrap()).count(), 1);
            }
            eprintln!("tree module {n} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        }
        assert!(p4t_a6_unit_text("a6", SERVED_DEFINES_SM89).unwrap().contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
    }

    #[test]
    fn the_a8_family_is_the_stream_body() {
        // pers_4et_a8.cu carries a3's unit, deal and pass three times, only the store changed
        let src = P4T_STREAM_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + src[a..].find("// ---- the pass:").unwrap();
        for (tag, store) in [("a3p", "p4t_store_proto<false>"), ("a3t", "p4t_store_proto<true>"), ("a8", "p4t_store_lean")] {
            let u = src[a..b]
                .replace("p4t_unit_stream(", &format!("p4t_unit_{tag}("))
                .replace("p4t_phase_stream(", &format!("p4t_phase_{tag}("))
                .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", &format!("    if (has) {store}(g, L, A, sl, acc, wrow, lane, cmax);\n"));
            let bd = src[c..d]
                .replacen("void pers_4et_stream_body(", &format!("void pers_4et_{tag}_body("), 1)
                .replace("p4t_phase_stream(", &format!("p4t_phase_{tag}("));
            assert_eq!(P4T_A8_PART.1.matches(format!("{u}{bd}").as_str()).count(), 1, "the {tag} pass drifted from the stream pass");
            let t = p4t_a8_unit_text(tag, SERVED_DEFINES_SM89).unwrap();
            assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
            for f in 0..3 {
                assert_eq!(t.matches(p4t_a8_symbol(tag, f).unwrap()).count(), 1);
            }
            eprintln!("tree module {tag} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        }
    }

    #[test]
    fn the_a9_body_is_the_stream_body() {
        // pers_4et_a9.cu carries a3's unit with a9's store and a3's pass with a
        // wait and a fold step after each row phase
        let src = P4T_STREAM_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + src[a..].find("// ---- the pass:").unwrap();
        let u = src[a..b]
            .replace("p4t_unit_stream(", "p4t_unit_a9(")
            .replace("p4t_phase_stream(", "p4t_phase_a9(")
            .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_a9(g, L, A, sl, acc, wrow, lane, cmax);\n");
        let mut bd = src[c..d].replacen("void pers_4et_stream_body(", "void pers_4et_a9_body(", 1);
        for segs in ["sq, sk, sv, 3u", "so, so, so, 1u", "sg, su, su, 2u", "sd, sd, sd, 1u"] {
            let x = format!("            p4t_phase_stream({segs}, L, A, tab, t_inv, area, cmax);\n        }}\n");
            assert_eq!(bd.matches(x.as_str()).count(), 1);
            bd = bd.replace(
                &x,
                &format!("            p4t_phase_a9({segs}, L, A, tab, t_inv, area, cmax);\n            P4T_WAIT();\n            p4t_fold_a9({segs}, L, A, cmax);\n        }}\n"),
            );
        }
        let fl = "        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);\n";
        let x = format!("        p4t_phase_stream(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        __syncthreads();\n{fl}");
        assert_eq!(bd.matches(x.as_str()).count(), 1);
        bd = bd.replace(
            &x,
            &format!("        p4t_phase_a9(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        __syncthreads();\n{fl}        P4T_WAIT();\n        p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n        __syncthreads();\n{fl}"),
        );
        assert_eq!(P4T_A9_PART.1.matches(format!("{u}{bd}").as_str()).count(), 1, "the a9 pass drifted from the stream pass");
        assert_eq!(bd.matches("P4T_WAIT();").count(), src[c..d].matches("P4T_WAIT();").count() + 5);
        let t = p4t_a9_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a9_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a9 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        // the waits: 13 a layer, 3 + 1 at the ends; the phases sum the layers
        assert_eq!(n_bar_a9(36, 0), 327 + 145);
        let (layers, g) = (2u32, 3usize);
        let n_bar = n_bar_a9(layers, 0) as usize;
        let st: Vec<u64> = (0..1 + 2 * n_bar).flat_map(|e| (0..g).map(move |c| 1000 * (e as u64 + 1) + c as u64)).collect();
        let p = p4t_phases_a9(&st, layers, g as u32).unwrap();
        assert_eq!(p.len(), 17);
        assert_eq!(p[3][0], 2.0 * 2000.0, "qkv fold twice");
        assert_eq!(p[16][0], 2000.0, "head fold once");
        assert_eq!(p[14][0], 2000.0, "norm once");
        assert_eq!(p.iter().map(|x| x[0]).sum::<f64>(), 2000.0 * n_bar as f64);
    }

    #[test]
    fn the_a10_body_is_the_a9_body() {
        // pers_4et_a10.cu carries a9's pass with the prologue and fold-step
        // substitutions; the waits are a9's (P4T_WAIT_GAP counts as one)
        let src = P4T_A9_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a9_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let a9 = &src[c..d];
        // the generator's substitutions, replayed (scratch gen_a10.py's pairs)
        let gap = r##"#define P4T_WAIT_GAP(...)                                                                            \
    do {                                                                                             \
        if (L.stamps != nullptr) {                                                                   \
            __syncthreads();                                                                         \
            if (threadIdx.x == 0u) L.stamps[(size_t)(2u * nb + 1u) * G + c0] = pers_now();           \
        }                                                                                            \
        if (!pers_grid_wait_gap(A->bar, L.bar_base + (++nb) * G, A->abort_flag, [&] { __VA_ARGS__; })) return; \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)(2u * nb) * G + c0] = pers_now(); \
    } while (0)
    // a10: the staged rotation area (null: the prologues keep a9's path)
    float* const v7a10 = p4t_v7_a10(S, L.area_floats, 2u * H > A->layers[0].mlp.inter ? 2u * H : A->layers[0].mlp.inter);
"##;
        let dn0 = a9.find("        for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw").unwrap();
        let dn_end = "            } else {\n                __syncthreads();\n            }\n        }\n";
        let dn1 = dn0 + a9[dn0..].find(dn_end).unwrap() + dn_end.len();
        let dn_old = &a9[dn0..dn1];
        let pairs: Vec<(String, String)> = vec![
            ("void pers_4et_a9_body(".into(), "void pers_4et_a10_body(".into()),
            ("    } while (0)\n".into(), format!("    }} while (0)\n{gap}")),
            (r##"                   H * (c0 + 1u) / G);
    P4T_WAIT();
    for (u32 l = 0; l < NL; ++l) {
"##.into(),
             r##"                   H * (c0 + 1u) / G);
    P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, A->layers[0].signs_a, H, A->layers[0].small_a, A->layers[0].k_a));
    for (u32 l = 0; l < NL; ++l) {
"##.into()),
            (r##"            p4t_ln(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, A->eps, H, S, L.xn + (size_t)r * H);
            pers_rot_finish(S, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
"##.into(),
             r##"            p4t_ln_rot_a10(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, Y.small_a, A->eps, H, Y.m_a, Y.k_a, Y.inv_a, S, v7a10,
                           L.xn + (size_t)r * H, L.xin + (size_t)r * L.ldx);
"##.into()),
            (r##"            p4t_fold_a9(so, so, so, 1u, L, A, cmax);
        }
        P4T_WAIT();
"##.into(),
             r##"            p4t_fold_a9(so, so, so, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, M.signs_h, H, M.small_h, M.k_h));
"##.into()),
            (r##"            p4t_ln(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.eps, H, S, nullptr);
            pers_rot_finish(S, M.small_h, H, M.m_h, M.k_h, M.inv_h);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
"##.into(),
             r##"            p4t_ln_rot_a10(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.small_h, M.eps, H, M.m_h, M.k_h, M.inv_h, S, v7a10,
                           nullptr, L.xin + (size_t)r * L.ldx);
"##.into()),
            (r##"            p4t_fold_a9(sg, su, su, 2u, L, A, cmax);
        }
        P4T_WAIT();
"##.into(),
             r##"            p4t_fold_gu_a10(sg, su, M, L);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr && Y.tetra_down) pers_v7_stage_rot(v7a10, M.signs_i, I, M.small_i, M.k_i));
"##.into()),
            (dn_old.to_string(), r##"        for (u32 r = c0; r < R; r += G) p4t_dn_in_a10(M, Y.tetra_down, L.xin + (size_t)r * L.ldx, S, I, v7a10);
"##.into()),
            (r##"            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT();
    }
"##.into(),
             r##"            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) {
            if (l + 1u < NL) pers_v7_stage_rot(v7a10, A->layers[l + 1u].signs_a, H, A->layers[l + 1u].small_a, A->layers[l + 1u].k_a);
            else p4t_v7_zero_signs_a10(v7a10, H);
        });
    }
"##.into()),
            (r##"        p4t_ln(L.x_a + (size_t)r * H, A->final_norm, nullptr, A->eps, H, S, L.xin + (size_t)r * L.ldx);
"##.into(),
             r##"        p4t_ln_final_a10(L.x_a + (size_t)r * H, A->final_norm, A->eps, H, S, v7a10, L.xin + (size_t)r * L.ldx);
"##.into()),
            ("#undef P4T_WAIT\n}\n".into(), "#undef P4T_WAIT_GAP\n#undef P4T_WAIT\n}\n".into()),
        ];
        let mut b10 = a9.to_string();
        for (x, y) in &pairs {
            assert_eq!(b10.matches(x.as_str()).count(), 1, "{}", &x[..x.len().min(60)]);
            b10 = b10.replacen(x.as_str(), y, 1);
        }
        assert_eq!(P4T_A10_PART.1.matches(b10.as_str()).count(), 1, "the a10 pass drifted from the a9 pass");
        let waits = |t: &str| t.matches("P4T_WAIT();").count() + t.matches("P4T_WAIT_GAP(if").count();
        assert_eq!(waits(a9), waits(&b10), "a10 keeps a9's waits");
        let t = p4t_a10_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_a9_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a10_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a10 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        // the staged area fits above the widest prologue row at the 4B and on the twin
        for (nw, widest) in [(16u32, 9728u32), (2, 9728)] {
            let area = p4t_area_floats(nw, 32, widest, 4);
            let off = (area - 2 * P4T_RMAX - 1840) & !3;
            eprintln!("area {area} floats at {nw} warps: the staged rotation at {off}, the prologue needs 9728");
            assert!(off >= 9728);
        }
    }

    #[test]
    fn the_a11_pass_is_a10s_with_prefetches_in_its_gaps() {
        // a11's pass is a10's with prefetch calls inside wait gaps only: the
        // same waits, and every a10 line outside the five rewritten sites kept
        let src = P4T_A10_PART.1;
        let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + src[c..].find(end).unwrap() + end.len();
        let a10 = &src[c..d];
        let a11s = P4T_A11_PART.1;
        let e = a11s.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a11_body(").unwrap();
        let f = e + a11s[e..].find(end).unwrap() + end.len();
        let a11 = &a11s[e..f];
        let waits = |t: &str| t.matches("P4T_WAIT();").count() + t.matches("P4T_WAIT_GAP(").count();
        assert_eq!(waits(a10), waits(a11), "a11 keeps a10's waits");
        assert_eq!(a11.matches("p4t_pf11_").count(), 5, "five prefetch calls");
        for line in a11.lines().filter(|l| l.contains("p4t_pf11_")) {
            assert!(!line.trim_start().starts_with("p4t_pf11_") || line.contains("));"), "a prefetch outside a gap: {line}");
        }
        let mut at = 0usize;
        for line in a10.lines().filter(|l| !l.contains("pers_4et_a10_body(") && !l.contains("P4T_WAIT") && !l.contains("p4t_attention(") && !l.contains("pers_v7_stage_rot(v7a10, M.signs_i") && !l.contains("p4t_v7_zero_signs_a10") && !l.trim().starts_with("});")) {
            let i = a11[at..].find(line).unwrap_or_else(|| panic!("a10's line {line:?} missing or out of order in a11's pass"));
            at += i + line.len();
        }
        let t = p4t_a11_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a11_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a11 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        // the volumes at the 4B, plan [5, 14, 5, 28, 5], 142 CTAs of 16 warps
        for int4_down in [false, true] {
            let v = p4t_prefetch_volume([5, 14, 5, 28, 5], 2560, 9728, 4096, 8, int4_down, 16, 142);
            let total: u64 = v.iter().map(|x| x.0).sum();
            eprintln!(
                "4B, {} down: q/k/v {:.2} MB (a CTA at most {:.1} KB), o {:.2} MB ({:.1} KB), gate/up {:.2} MB ({:.1} KB), down {:.2} MB ({:.1} KB); a layer {:.2} MB of the L40S's 96 MB L2",
                if int4_down { "int4" } else { "Tetra" },
                v[0].0 as f64 / 1e6, v[0].1 as f64 / 1e3, v[1].0 as f64 / 1e6, v[1].1 as f64 / 1e3,
                v[2].0 as f64 / 1e6, v[2].1 as f64 / 1e3, v[3].0 as f64 / 1e6, v[3].1 as f64 / 1e3, total as f64 / 1e6
            );
            assert!(total < 96_000_000);
        }
    }

    #[test]
    fn the_a11i_pass_is_a11s_with_marks() {
        // a11i's pass is a11's with its phase calls renamed; its unit is a9's
        // with the marks only (every a9 unit line kept, in order)
        let a10 = P4T_A11_PART.1;
        let c = a10.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a11_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + a10[c..].find(end).unwrap() + end.len();
        let body = a10[c..d].replacen("void pers_4et_a11_body(", "void pers_4et_a11i_body(", 1).replace("p4t_phase_a9(", "p4t_phase_a11i(");
        assert_eq!(P4T_A11I_PART.1.matches(body.as_str()).count(), 1, "the a11i pass drifted from a11's");
        let a9 = P4T_A9_PART.1;
        let a = a9.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + a9[a..].find("// p4t_phase's deal, the stream unit.").unwrap();
        let ui = P4T_A11I_PART.1;
        let mut at = ui.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        for line in a9[a..b].lines().filter(|l| !l.contains("p4t_unit_a9(") && !l.contains("u32* area, unsigned long long* cmax)")) {
            let i = ui[at..].find(line).unwrap_or_else(|| panic!("a9's unit line {line:?} missing or out of order in a11i's"));
            at += i + line.len();
        }
        let t = p4t_a11i_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_a11_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a11i_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a11i at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        // the wait classes: arrival 10 (c + 1) before leaving, busy 1000 - 10 (c + 1)
        let (layers, g) = (2u32, 3usize);
        let n_bar = n_bar_a9(layers, 0) as usize;
        let st: Vec<u64> = (0..1 + 2 * n_bar)
            .flat_map(|e| (0..g).map(move |c| { let w = e.div_ceil(2) as u64; if e % 2 == 1 { 1000 * (w + 1) - 10 * (c as u64 + 1) } else { 1000 * (w + 1) } }))
            .collect();
        let (cls, pro) = p4t_wait_classes(&st, layers, g as u32, 1).unwrap();
        // 3 prologue waits a layer + the norm (7), 4 row waits a layer + the head (9), 4 folds + 1 (9)
        assert_eq!(cls[0], [20.0 * 7.0, 30.0 * 7.0]);
        assert_eq!(cls[1], [20.0 * 9.0, 30.0 * 9.0]);
        assert_eq!(cls[2][1], 30.0 * 9.0);
        assert_eq!(pro[0], 2.0 * 990.0, "ln1 on CTA 0, twice");
    }

    #[test]
    fn the_row_cap_passes_are_a10s() {
        // pers_4et_a10rc.cu: a9's unit and deal over RC n8 tiles, a10's pass
        // with the row cap; nothing else of them changed
        let a9 = P4T_A9_PART.1;
        let a = a9.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
        let b = a + a9[a..].find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a9_body(").unwrap();
        let unit = a9[a..b]
            .replacen("__device__ __forceinline__ void p4t_unit_a9(", "template <u32 RC, bool IP = false>\n__device__ __forceinline__ void p4t_unit_rc(", 1)
            // a12ip (M1): the words read in place under IP, the copy otherwise
            .replacen(
                "// once and streamed by each warp alone.\n",
                "// once and streamed by each warp alone. IP: the words read in place, from\n// the served order (p4t_ip_load), instead of the tree-tiled copy.\n",
                1,
            )
            .replacen(
                "        const u32* wp = g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);\n        u32 c0, c1, c2;\n        p4t_stream_load(g.kind, wp, c0, c1, c2);\n",
                "        const u32* wp = IP ? p4t_ip_row(g, wrow + r)\n                           : g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);\n        u32 c0, c1, c2;\n        if (IP) p4t_ip_load(g, wp, sub_lo, h, c0, c1, c2);\n        else p4t_stream_load(g.kind, wp, c0, c1, c2);\n",
                1,
            )
            .replacen(
                "            if (s + 1u < ns) p4t_stream_load(g.kind, wp + SW, n0, n1, n2);\n",
                "            if (IP) {\n                if (s + 1u < ns) p4t_ip_load(g, wp, sub_lo + s + 1u, h, n0, n1, n2);\n            } else if (s + 1u < ns) {\n                p4t_stream_load(g.kind, wp + SW, n0, n1, n2);\n            }\n",
                1,
            )
            .replacen("            wp += SW;\n", "            if (!IP) wp += SW;\n", 1)
            .replacen("    float acc[4][4];\n#pragma unroll\n    for (u32 nt = 0; nt < 4u; ++nt)", "    float acc[RC][4];\n#pragma unroll\n    for (u32 nt = 0; nt < RC; ++nt)", 1)
            .replacen("                for (u32 nt = 0; nt < 4u; ++nt) {", "                for (u32 nt = 0; nt < RC; ++nt) {", 1)
            .replacen("    if (has) p4t_store_a9(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_rc<RC>(g, L, A, sl, acc, wrow, lane, cmax);\n", 1)
            .replacen("__device__ __forceinline__ void p4t_phase_a9(", "template <u32 RC, bool IP = false>\n__device__ __forceinline__ void p4t_phase_rc(", 1)
            .replacen("        p4t_unit_a9(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);", "        p4t_unit_rc<RC, IP>(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);", 1);
        assert!(!unit.contains("nt < 4u"), "every n8-tile loop of the unit is over RC");
        let a10 = P4T_A10_PART.1;
        let c = a10.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(").unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + a10[c..].find(end).unwrap() + end.len();
        assert_eq!(a10[c..d].matches("if (R == 0u || R > P4T_RMAX ||").count(), 1);
        let body = a10[c..d]
            .replacen("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(", "template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(", 1)
            .replace("p4t_phase_a9(", "p4t_phase_rc<RC>(")
            .replacen("if (R == 0u || R > P4T_RMAX ||", "if (R == 0u || R > 8u * RC ||", 1);
        assert_eq!(P4T_A10RC_PART.1.matches(format!("{unit}{body}").as_str()).count(), 1, "the row-cap pass drifted from a10's");
        for rows in [8, 16] {
            let t = p4t_a10rc_unit_text(SERVED_DEFINES_SM89, rows);
            assert!(t.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
            for f in 0..3 {
                assert_eq!(t.matches(p4t_a10rc_symbol(rows, f).unwrap()).count(), 1);
            }
            eprintln!("tree module a10r{rows} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        }
    }

    #[test]
    fn the_thread_count_arms_are_a10s_pass() {
        // pers_4et_a10t.cu adds symbols around a10's body and nothing else
        let t = P4T_A10T_PART.1;
        assert_eq!(t.matches("pers_4et_a10_body<FORM_>(A, L);").count(), 1);
        assert_eq!(t.matches("__device__").count(), 0, "no device function of its own");
        for threads in [384u32, 256] {
            let m = p4t_a10t_unit_text(SERVED_DEFINES_SM89, threads);
            assert!(m.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
            for f in 0..3 {
                assert_eq!(m.matches(p4t_a10t_symbol(threads, f).unwrap()).count(), 1);
            }
            // the 4B at this warp count: area, shared bytes, cap, the stream rule's plan
            let nw = threads / 32;
            let area = p4t_area_floats(nw, 32, 9728, 4);
            let cap = p4t_stream_cap(area, nw);
            let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
            let rule = p4t_splits_stream(2560, 9728, 4096, 8, 151_936, nw, 142, cap).unwrap();
            for (name, s) in [("[5, 14, 5, 28, 5]", [5u32, 14, 5, 28, 5]), ("the stream rule's", rule)] {
                let p = p4t_plan(s, 2560, 9728, 4096, 8, 151_936, nw, 142);
                let steps: Vec<u32> = s.iter().zip([2560u32, 4096, 2560, 9728, 2560]).map(|(&x, d)| p4t_slice_steps(d, x)).collect();
                eprintln!(
                    "4B at {threads} threads ({nw} warps): area {area} floats, {smem} B shared, cap {cap}; plan {name} {s:?}: units, waves, fill {:?}; sub-steps a slice {steps:?}",
                    p.iter().map(|x| (x.0, x.1, (x.2 * 1000.0).round() / 10.0)).collect::<Vec<_>>()
                );
                assert!(steps.iter().all(|&x| x <= cap));
            }
            eprintln!("tree module a10t{threads} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", m.len(), crate::sha256::sha256_hex(m.as_bytes()));
        }
    }

    #[test]
    fn the_second_thread_count_arms_are_a10s_passes() {
        // pers_4et_a10t2.cu adds symbols around a10's and the row cap's bodies, nothing else
        let t = P4T_A10T2_PART.1;
        assert_eq!(t.matches("pers_4et_a10_body<FORM_>(A, L);").count(), 1);
        assert_eq!(t.matches("pers_4et_rc_body<FORM_, 2u>(A, L);").count(), 1);
        assert_eq!(t.matches("__device__").count(), 0);
        for (arm, threads, base) in [("a10t448", 448u32, p4t_a10_unit_text(SERVED_DEFINES_SM89)), ("a10t320", 320, p4t_a10_unit_text(SERVED_DEFINES_SM89)), ("a10t384r16", 384, p4t_a10rc_unit_text(SERVED_DEFINES_SM89, 16))] {
            let m = p4t_a10t2_unit_text(SERVED_DEFINES_SM89, arm).unwrap();
            assert!(m.contains(&base));
            for f in 0..3 {
                assert_eq!(m.matches(p4t_a10t2_symbol(arm, f).unwrap()).count(), 1);
            }
            let nw = threads / 32;
            let area = p4t_area_floats(nw, 32, 9728, 4);
            let cap = p4t_stream_cap(area, nw);
            let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
            let rule = p4t_splits_stream(2560, 9728, 4096, 8, 151_936, nw, 142, cap).unwrap();
            for (name, s) in [("[5, 14, 5, 28, 5]", [5u32, 14, 5, 28, 5]), ("[8, 10, 8, 20, 6]", [8, 10, 8, 20, 6]), ("the stream rule's", rule)] {
                let p = p4t_plan(s, 2560, 9728, 4096, 8, 151_936, nw, 142);
                let steps: Vec<u32> = s.iter().zip([2560u32, 4096, 2560, 9728, 2560]).map(|(&x, d)| p4t_slice_steps(d, x)).collect();
                eprintln!(
                    "4B, {arm} ({nw} warps): area {area} floats, {smem} B shared, cap {cap}; plan {name} {s:?}: units, waves, fill {:?}; sub-steps a slice {steps:?}",
                    p.iter().map(|x| (x.0, x.1, (x.2 * 1000.0).round() / 10.0)).collect::<Vec<_>>()
                );
            }
            eprintln!("tree module {arm} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", m.len(), crate::sha256::sha256_hex(m.as_bytes()));
        }
    }

    #[test]
    fn the_a12_pass_is_the_row_cap_pass_with_taps() {
        // pers_4et_a12.cu: pers_4et_rc_body renamed, two tap calls added
        let rc = P4T_A10RC_PART.1;
        let c = rc.find("template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(").unwrap();
        // the a12ip edits (M1): IP threaded through the body's five phases, nothing else
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + rc[c..].find(end).unwrap() + end.len();
        let body = rc[c..d]
            .replacen("template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(", "template <int FORM, u32 RC, bool IP = false>\n__device__ __forceinline__ void pers_4et_a12_body(", 1)
            .replace("p4t_phase_rc<RC>(", "p4t_phase_rc<RC, IP>(")
            .replacen(
                "        // ln1: the natural values for the int4 v rows, the signed and\n",
                "        p4t_taps_a12(L, H, R, l, c0, G);\n        // ln1: the natural values for the int4 v rows, the signed and\n",
                1,
            )
            .replacen(
                "    // the final norm, natural: the head's input\n",
                "    p4t_taps_a12(L, H, R, NL, c0, G);\n    // A launch with L.split_head == 0 (the chain prefill's passes before the\n    // prompt's last, docs/mesures/large-prefill-contexte-2026-11-11.txt) skips\n    // the final norm and the head, every wait kept: no logits, no argmax (the\n    // ring reads u32 max), the rows' residuals, K, V and taps untouched.\n    const bool head_on = L.split_head != 0u;\n    // the final norm, natural: the head's input\n",
                1,
            )
            // the head switch (96c501a): the final norm, the head's phase and
            // its fold run only when split_head is not 0; every wait and
            // every slot write stay unconditional
            .replacen(
                "    for (u32 r = c0; r < R; r += G)\n        p4t_ln_final_a10(",
                "    if (head_on)\n        for (u32 r = c0; r < R; r += G)\n            p4t_ln_final_a10(",
                1,
            )
            .replacen(
                "        p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);\n        unsigned long long po = 0ull;\n        u32 to = 0u;\n        p4t_place(sh, L, po, to);\n        p4t_phase_rc<RC, IP>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n",
                "        if (head_on) {\n            p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);\n            unsigned long long po = 0ull;\n            u32 to = 0u;\n            p4t_place(sh, L, po, to);\n            p4t_phase_rc<RC, IP>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        }\n",
                1,
            )
            .replacen("        p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n", "        if (head_on) p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n", 1);
        assert_eq!(P4T_A12_PART.1.matches(body.as_str()).count(), 1, "the a12 pass drifted from the row-cap pass");
        let t = p4t_a12_unit_text(SERVED_DEFINES_SM89);
        assert!(t.contains(&p4t_a10rc_unit_text(SERVED_DEFINES_SM89, 16)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a12_symbol(f).unwrap()).count(), 1);
        }
        eprintln!("tree module a12 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
        // a12ip: a12's text under one more define; its symbols and the load
        // check, and not a12's
        let ip = p4t_a12ip_unit_text(SERVED_DEFINES_SM89);
        assert_eq!(ip, format!("#define P4T_A12IP 1\n{t}"));
        assert_eq!(P4T_A12_PART.1.matches("pers_4et_a12_body<FORM_, 2u, true>(A, L);").count(), 1);
        assert_eq!(P4T_A12_PART.1.matches("extern \"C\" __global__ void p4t_ip_check(").count(), 1);
        for f in 0..3 {
            assert_eq!(ip.matches(p4t_a12ip_symbol(f).unwrap()).count(), 1);
        }
        assert!(p4t_a12ip_symbol(3).is_err());
        eprintln!("tree module a12ip at SERVED_DEFINES_SM89: {} bytes, sha256 {}", ip.len(), crate::sha256::sha256_hex(ip.as_bytes()));
    }

    /// The card's tile define on the L40S (`TileConfig::define`, 64 blocks).
    const CARD_TILE_SM89: &str = "#define TILE_BLOCKS 64u\n";

    /// The module texts the card compiled in large-prefill D1, at their
    /// sha256 (d1.txt's loader lines): 4E's tree module, a12ip's, a12ipg's.
    /// An arm added beside them must leave all three where they are.
    #[test]
    fn the_d1_tree_modules_keep_their_bytes() {
        for (name, text, sha) in [
            ("4E tree", p4t_unit_text(CARD_TILE_SM89), "ce1c354d2583eb178ae8622ff99e10f9b457543972bb5039ce89ede9eec31e44"),
            ("a12ip", p4t_a12ip_unit_text(CARD_TILE_SM89), "63155df95ef832aabbff328bd92062b6981763ed57358e32dc118d150df7c6ad"),
            ("a12ipg", p4t_a12ipg_unit_text(CARD_TILE_SM89), "246e99b18895a7f9452411372b8ff4bbb686276408318e547656fb809b722da8"),
            // the 32-row arm (large-prefill-r32-mac) and the served 4E v9 module
            ("a12ipg32", p4t_a12ipg32_unit_text(CARD_TILE_SM89), "a35f2b83061dc8a22ee65e6805c9e73454a93ec773ec919a14afc80fcf60aa8f"),
            ("4E v9", format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(CARD_TILE_SM89)), "1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530"),
        ] {
            assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "{name}: {} bytes", text.len());
        }
    }

    /// Each full-tile module is its parent's text with the ft edit and nothing
    /// else: the anchor once in the parent, the inserted part before it and
    /// one branch after it; taking both out gives the parent back. v9ft is
    /// E1's v9 module with the same edit. The edit is refused on a text
    /// without the anchor or with two.
    #[test]
    fn the_ft_modules_are_their_parents_with_the_full_tile_branch() {
        let (_, from, to) = pers_ft_edit();
        assert!(to.starts_with(PERS_FT_PART.1) && to.ends_with("        return;\n    }\n"));
        assert_eq!(to.matches("pers_attn_tile_full(").count(), 2, "one definition, one call");
        assert_eq!(to.matches("if (nv == PERS_TT) {").count(), 1);
        for (arm, parent) in [
            ("a12ipft", p4t_a12ip_unit_text(CARD_TILE_SM89)),
            ("a12ipgft", p4t_a12ipg_unit_text(CARD_TILE_SM89)),
            ("a12ipg32ft", p4t_a12ipg32_unit_text(CARD_TILE_SM89)),
        ] {
            let ft = p4t_ft_unit_text(CARD_TILE_SM89, arm).unwrap();
            assert_eq!(parent.matches(from.as_str()).count(), 1, "{arm}");
            assert_eq!(ft.replacen(to.as_str(), &from, 1), parent, "{arm}: the parent and the ft edit, nothing else");
        }
        let v9 = p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap();
        let v9ft = p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap();
        assert_eq!(v9ft.replacen(to.as_str(), &from, 1), v9, "v9ft: E1's v9 and the ft edit, nothing else");
        assert!(p4t_ft_unit_text(CARD_TILE_SM89, "a12ft").is_err());
        assert!(pers_ft_apply("no tile here").is_err());
        assert!(pers_ft_apply(&format!("{from}{from}")).is_err());
    }

    /// The ft modules at the card's tile define, at the sha256 the job scripts
    /// check (ops/jobs/large-prefill-ft.sh, ops/jobs/large-attn-ft-4e.sh): an
    /// edit to pers_attn_full.cuh moves these and must move the scripts too.
    #[test]
    fn the_ft_modules_keep_their_bytes() {
        for (name, text, sha) in [
            ("a12ipft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipft").unwrap(), "544f1dd5ff2dd0b7eb52ecb60002af677748dbf402182ea3d526a0ccabd73132"),
            ("a12ipgft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipgft").unwrap(), "e0fd30be4318032e371854eb4f485c14b232292481603ee2810001c0415086f2"),
            ("a12ipg32ft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipg32ft").unwrap(), "19b734a3fb6029848d87339c7d505e9bcfc612f182256793026ea964d09853e2"),
            ("4E v9ft", p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap(), "22ff8320cb881b39690cbbfe6f1fd9ee8c1715142adfb4ee1ffbe932b061619a"),
            ("4E E1 v9", p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap(), "d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860"),
        ] {
            assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "{name}: {} bytes", text.len());
        }
    }

    /// The rc arms' modules at the card's tile define, at the sha256 the R6
    /// job script checks (ops/jobs/recherche-debit-jeton.sh, control 3): an
    /// edit to pers_4e_rc.cuh or to the rc edits moves these and must move
    /// the script and its prereg too.
    #[test]
    fn the_rc_modules_keep_their_bytes() {
        for (name, sha) in [
            ("v9ep", "f4166ccf9d857b30cae8f7691e9e4f83c2726218cb48c29dc512cb42b1a668dd"),
            ("v9fl", "9925b4ad769c791d2372d52a14b68e650a549ce15edd05eecae0dececfa4f03d"),
            // R6b's variants (ops/jobs/recherche-debit-charge.sh, recherche-debit-jeton2.sh)
            ("v9ep2", "a7d9a04d4bb963aac4ed831f1d7b41f6473e939dbc6c7c047d3b9e90b9801aa9"),
            ("v9ep3", "60dfa277ef82cf5c94b5a4b0b5745610324abac432c8ef4725ea96ba150714f7"),
            ("v9fl2", "d863288607ba961223b3ac8501b99f2e4c97dfc082cd237b80300df8b33dbd11"),
            ("v9fl3", "713ed5877aef63da831b8d149d31418889e42ac2580589294fa303e408ccf3fe"),
        ] {
            let text = p4e_e1_unit_text(CARD_TILE_SM89, name).unwrap();
            assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "4E {name}: {} bytes", text.len());
        }
    }

    /// Writes the tree modules at the card's tile define into
    /// `target/tree-modules/` for an offline compile (ops/tree-sass-probe.sh).
    #[test]
    #[ignore]
    fn dump_tree_modules() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/tree-modules");
        std::fs::create_dir_all(&dir).unwrap();
        let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(CARD_TILE_SM89));
        for (name, text) in [
            ("4e-v9", v9),
            ("a12ip", p4t_a12ip_unit_text(CARD_TILE_SM89)),
            ("a12ipg", p4t_a12ipg_unit_text(CARD_TILE_SM89)),
            ("a12ipg32", p4t_a12ipg32_unit_text(CARD_TILE_SM89)),
            ("a12ipft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipft").unwrap()),
            ("a12ipgft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipgft").unwrap()),
            ("a12ipg32ft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipg32ft").unwrap()),
            ("4e-v9ft", p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap()),
            ("4e-v9e1", p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap()),
            ("4e-v9ep", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep").unwrap()),
            ("4e-v9fl", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl").unwrap()),
            ("4e-v9ep2", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep2").unwrap()),
            ("4e-v9ep3", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep3").unwrap()),
            ("4e-v9fl2", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl2").unwrap()),
            ("4e-v9fl3", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl3").unwrap()),
        ] {
            std::fs::write(dir.join(format!("{name}.cu")), &text).unwrap();
            eprintln!("{name}: {} bytes, sha256 {}", text.len(), crate::sha256::sha256_hex(text.as_bytes()));
        }
    }

    #[test]
    fn the_a12ipg_pass_is_a12ip_with_the_grouped_attention() {
        // pers_4et_a12g.cu: a12's body renamed, its area check and its
        // attention call swapped for the grouped ones, nothing else
        let a12 = P4T_A12_PART.1;
        let head = "template <int FORM, u32 RC, bool IP = false>\n__device__ __forceinline__ void pers_4et_a12_body(";
        let c = a12.find(head).unwrap();
        let end = "#undef P4T_WAIT\n}\n";
        let d = c + a12[c..].find(end).unwrap() + end.len();
        let body = a12[c..d]
            .replacen(head, "template <int FORM, u32 RC, bool IP>\n__device__ __forceinline__ void pers_4et_a12g_body(", 1)
            .replacen("L.area_floats < p4t_area_floats(", "L.area_floats < p4g_area_floats(", 1)
            .replacen("        p4t_attention(A, L,", "        p4g_attention(A, L,", 1);
        assert_eq!(P4T_A12G_PART.1.matches(body.as_str()).count(), 1, "the a12ipg pass drifted from a12's");
        assert_eq!(P4T_A12G_PART.1.matches("pers_4et_a12g_body<FORM_, 2u, true>(A, L);").count(), 1);
        assert_eq!(P4T_A12G_PART.1.matches("#define P4G_RG 4u ").count(), 1, "P4G_RG mirrors the kernel's");
        let ip = p4t_a12ip_unit_text(SERVED_DEFINES_SM89);
        let g = p4t_a12ipg_unit_text(SERVED_DEFINES_SM89);
        assert_eq!(g, format!("#define P4T_A12IPG 1\n{ip}\n{}", P4T_A12G_PART.1));
        for f in 0..3 {
            assert_eq!(g.matches(p4t_a12ipg_symbol(f).unwrap()).count(), 1);
            assert_eq!(ip.matches(p4t_a12ipg_symbol(f).unwrap()).count(), 0, "a12ip's text carries no a12ipg symbol");
        }
        assert!(p4t_a12ipg_symbol(3).is_err());
        // the 4B at 12 warps: 18,624 floats, 93,568 B a CTA with the tables, under sm_89's 101,376
        let a = p4g_area_floats(12, 32, 9728, 4);
        assert_eq!(a, 18_624);
        assert_eq!(a, p4t_area_floats(12, 32, 9728, 4) + 3_680);
        assert!(SMEM_HEADER + TABLE_BYTES + 4 * a <= 101_376, "{} B", SMEM_HEADER + TABLE_BYTES + 4 * a);
        eprintln!("tree module a12ipg at SERVED_DEFINES_SM89: {} bytes, sha256 {}; area {a} floats, {} B shared", g.len(), crate::sha256::sha256_hex(g.as_bytes()), SMEM_HEADER + TABLE_BYTES + 4 * a);
    }

    #[test]
    fn the_a12ipg32_pass_is_a12ipg_at_four_tiles() {
        // pers_4et_a12g32.cu: a12ipg's body instantiated at RC = 4, nothing else
        let part = P4T_A12G32_PART.1;
        assert_eq!(part.matches("pers_4et_a12g_body<").count(), 1);
        assert_eq!(part.matches("pers_4et_a12g_body<FORM_, 4u, true>(A, L);").count(), 1);
        assert_eq!(part.matches("__global__").count(), 1, "one kernel macro, no other code");
        let g = p4t_a12ipg_unit_text(SERVED_DEFINES_SM89);
        let t = p4t_a12ipg32_unit_text(SERVED_DEFINES_SM89);
        assert_eq!(t, format!("#define P4T_A12IPG32 1\n{g}\n{part}"));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a12ipg32_symbol(f).unwrap()).count(), 1);
            assert_eq!(g.matches(p4t_a12ipg32_symbol(f).unwrap()).count(), 0, "a12ipg's text carries no a12ipg32 symbol");
        }
        assert!(p4t_a12ipg32_symbol(3).is_err());
        // the area is a12ipg's, which already stages 32 rows
        assert_eq!(p4g_area_floats(12, 32, 9728, 4), 18_624);
        eprintln!("tree module a12ipg32 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }

    use super::*;

    #[test]
    fn the_verify_phases_sum_their_events() {
        // two layers, three CTAs: 15 events; CTA 1 is the median at each
        let (layers, g) = (2u32, 3usize);
        let events = 1 + n_bar_4e(layers) as usize;
        let mut st = vec![0u64; events * g];
        let mut t = 100u64;
        for e in 0..events {
            t += 10 * (e as u64 + 1);
            st[e * g] = t - 3;
            st[e * g + 1] = t;
            st[e * g + 2] = t + 50;
        }
        let p = p4v_phases(&st, layers, g as u32).unwrap();
        // phase of event e lasts 10 (e + 1)
        assert_eq!(p[0], 20.0);
        assert_eq!(p[1], 30.0 + 90.0);
        assert_eq!(p[6], 80.0 + 140.0);
        assert_eq!(p[7], 150.0);
        assert_eq!(p.iter().sum::<f64>(), (st[(events - 1) * g + 1] - st[1]) as f64);
        assert!(p4v_phases(&st[1..], layers, g as u32).is_err());
        st[5] = 0;
        assert!(p4v_phases(&st, layers, g as u32).is_err());
    }

    #[test]
    fn the_trows_unit_carries_every_arm() {
        let t = trows_unit_text("#define TILE_BLOCKS 64u");
        assert!(t.starts_with(&pers_unit_text("#define TILE_BLOCKS 64u")), "4E's module bytes come first, unchanged");
        for (name, sym, _, minb) in TROWS_ARMS {
            if name == "v2_s" || name == "v2r64" {
                continue; // the symbols of v2 and v2_o2, on other sites or CTAs a SM
            }
            let arm = sym.trim_start_matches("tv_trows_").trim_end_matches("_o2").to_uppercase();
            assert!(t.contains(&format!("TROWS_KERNEL({sym}, TROWS_{arm}, {minb})")), "{sym}");
        }
        assert_eq!(trows_smem_bytes(9728), 57_984, "4E's own shared bytes at d_in 9,728");
        assert_eq!(trows_k2_smem_bytes(), 96_896);
        assert!(t.contains("#define TROWS_X2_AT 9728u"), "K2's second activation sits after the widest first one");
        assert!(trows_small_only("v2_o2") && trows_small_only("v2_s") && !trows_small_only("v2"));
        assert!(!trows_small_only("v2r64"), "v2r64 runs every site at one CTA a SM");
        assert!(trows_counters("gb") && trows_counters("gbc") && !trows_counters("v2") && !trows_counters("bal"));
        // two sites, two CTAs, one warp: CTA busy at w = 1
        let st = [0u32, 10, 0, 30, 0, 20, 0, 20];
        assert!((trows_cta_spread(&st, 2, 2, 1) - 50.0 / 40.0).abs() < 1e-12, "slowest 30 + 20 over mean 20 + 20");
        assert!(t.contains("if (!pers_grid_wait(ctr, (s + 1u) * g, ctr + 1)) return;"), "gb and gbc wait on the grid after a site");
        assert_eq!(trows_cpa_smem_bytes(16), 67_200, "CPA's ring after 4E's own bytes");
        for (name, sym, _) in Q4ROWS_ARMS {
            assert!(t.contains(&format!("extern \"C\" __global__ void __launch_bounds__(512, 1) {sym}(")) || t.contains(&format!("Q4ROWS_KERNEL({sym}, ")), "{name}");
        }
        for (name, sym) in CTR_ARMS {
            assert!(t.contains(&format!("extern \"C\" __global__ void {sym}(")), "{name}");
        }
    }

    #[test]
    fn the_tile_layout_keeps_the_served_bytes_at_the_4b_widths() {
        // 2,560, 4,096 and 9,728 columns: 106, 170 and 405 blocks
        for nb in [106u32, 170, 405] {
            assert_eq!(trows_tile_stride(nb), tetra_stride(nb), "nb {nb}");
            assert!(trows_cpa_ok(tetra_stride(nb)), "nb {nb}");
        }
        assert_eq!(trows_tile_stride(128), 192);
        assert_eq!(trows_tile_stride(5), 8);
        assert_eq!(trows_tile_stride(129), 196);
    }

    #[test]
    fn the_int4_sites_are_the_84_records_of_the_4b() {
        let s = q4rows_sites();
        assert_eq!(s.len(), 84);
        assert_eq!(s.iter().filter(|r| r.3 == "down").count(), 12);
        let b = q4rows_bytes(&s);
        assert_eq!(b, 409_436_160, "0.409 GB, plan-300 §4");
        assert!(s.iter().all(|r| r.1 % 256 == 0));
        assert_eq!(ctr_expected("ctr8", 142, 3), [vec![24u32; 17], vec![18]].concat());
        assert_eq!(ctr_expected("bar", 142, 3), vec![426]);
    }

    #[test]
    fn the_i16_check_reads_ulps_and_the_rms_bound() {
        // site 1: 1.0, 2.0, -1.0 ; site 2: 0.5, 1e-4 (a row near zero)
        let want = [0x3c00u16, 0x4000, 0xbc00, 0x3800, 0x068e];
        let same = vrows_i16_check(&want, &want, &[3, 2]);
        assert_eq!((same.same, same.one_ulp, same.within_rms, same.out), (5, 0, 0, 0));
        // one ulp up on 1.0, two ulps on 2.0 (out: 2 ulps of 2.0 is 3.9e-3 of
        // site 1's rms 1.41), a near-zero row moved by many ulps but 1e-4 of 0.35
        let got = [0x3c01u16, 0x4002, 0xbc00, 0x3800, 0x0600];
        let c = vrows_i16_check(&got, &want, &[3, 2]);
        assert_eq!((c.same, c.one_ulp, c.within_rms, c.out), (2, 1, 1, 1), "{c:?}");
        assert!(c.worst > 2e-3 && c.worst < 3e-3, "{c:?}");
        // the two zeros are one value
        assert_eq!(vrows_i16_check(&[0x8000], &[0x0000], &[1]).same, 1);
        assert_eq!(vrows_i16_check(&[0x8000], &[0x0000], &[1]).out, 0);
    }

    #[test]
    fn the_verify_row_shared_bytes_fit_one_cta() {
        for (name, _, k, _) in VROWS_ARMS {
            let b = vrows_smem_bytes(name, k, 9728);
            assert!(b <= 101_376, "{name}: {b} B over the 101,376 a CTA may opt into on sm_89");
        }
        assert_eq!(vrows_smem_bytes("i16k4", 4, 9728), 256 + 19_584 + 4 * 20_320);
    }

    #[test]
    fn trows_stamps_fold_by_class() {
        // 2 sites of d_in 2,560 (nit 4) and 1 of 9,728 (nit 13), grid 2, 4 warps.
        let sites = [(10u32, 2560u32), (6, 2560), (3, 9728)];
        let (grid, nw) = (2u32, 4u32);
        let mut st = vec![0u32; sites.len() * 2 * 5];
        for (s, &(d_out, _)) in sites.iter().enumerate() {
            for c in 0..grid {
                let lo = d_out * c / grid;
                let hi = d_out * (c + 1) / grid;
                let at = (s * 2 + c as usize) * 5;
                for w in 0..nw {
                    st[at + w as usize] = 100 * trows_warp_rows(lo, hi, w, nw);
                }
                st[at + 4] = 100 * (hi - lo).div_ceil(nw);
            }
        }
        let cl = trows_classes(&st, &sites, grid, nw);
        assert_eq!(cl.len(), 2);
        assert_eq!((cl[0].d_in, cl[0].sites, cl[1].d_in, cl[1].sites), (2560, 2, 9728, 1));
        // site 0: 5 rows a CTA over 4 warps (2,1,1,1); site 1: 3 rows (1,1,1,0)
        assert_eq!(cl[0].steps, 4 * (5 + 5 + 3 + 3));
        assert_eq!(cl[0].busy_warps, 4 + 4 + 3 + 3);
        assert_eq!(cl[0].cycles_per_step(), 25.0);
        assert_eq!(cl[0].cta_cycles, 200 + 200 + 100 + 100);
        assert!((cl[0].balance(nw) - 1600.0 / 2400.0).abs() < 1e-12);
        // site 2: CTA 0 rows [0, 1), CTA 1 rows [1, 3)
        assert_eq!(cl[1].steps, 13 * 3);
        assert_eq!(trows_clock_mhz(&[0, 0, 2_520, 1_000, 10, 5, 2_530, 1_005]), 2_520.0);
    }


    #[test]
    fn the_schedule_has_the_served_counts() {
        let s = served_4b_schedule();
        assert_eq!(s.len(), 252);
        let all = s.iter().filter(|x| matches!(x.wait, Wait::All(_))).count();
        let cnt = s.iter().filter(|x| matches!(x.wait, Wait::Count(_))).count();
        assert_eq!(all, 35 + 36 + 24);
        assert_eq!(cnt, 36 + 12);
        assert_eq!(s.iter().filter(|x| x.signal_after).count(), 48);
        assert_eq!(s.iter().filter(|x| x.prologue == Prologue::NormRot).count(), 72);
        assert_eq!(s.iter().filter(|x| x.prologue == Prologue::Rot).count(), 24);
        // the 97 waits per token of the served design = 95 + the 2 trailing barriers
        assert_eq!(all as u32 + N_TRAILING, 97);
        assert_eq!(n_grid_syncs(&s), 145);
        assert_eq!(n_bar(&s, 0), 0);
        assert_eq!(n_bar(&s, 1), 97);
        assert_eq!(n_bar(&s, 2), 242);
        validate(&s).expect("valid");
    }

    /// The mutants of `persistent.md` §6, step 1: each is a schedule the
    /// validator must refuse.
    #[test]
    fn the_validator_refuses_what_it_should() {
        let base = served_4b_schedule();
        let refuse = |f: &dyn Fn(&mut Vec<Site>), what: &str| {
            let mut s = base.clone();
            f(&mut s);
            assert!(validate(&s).is_err(), "the validator accepted: {what}");
        };
        refuse(&|s| s[2].signal_after = false, "a waited site that does not signal");
        refuse(&|s| s[1].signal_after = true, "a signal nobody waits for");
        refuse(&|s| s[1].prologue = Prologue::NormRot, "a prologue on a shared staging");
        refuse(&|s| s[3].wait = Wait::Count(10), "a counter wait on a later site");
        refuse(&|s| s[3].wait = Wait::Count(3), "a counter wait on itself");
        refuse(&|s| s[4].wait = Wait::None, "gate reading o's residual with no wait");
        refuse(&|s| s[3].wait = Wait::None, "o reading attention with no wait");
        refuse(
            &|s| {
                // An ordinal skipped: the CTA that skips arrives once fewer,
                // which is the arrival a CTA with no rows used to skip.
                if let Wait::All(n) = s[4].wait {
                    s[4].wait = Wait::All(n + 1);
                }
            },
            "a barrier ordinal skipped",
        );
        refuse(
            &|s| {
                // Two sites with the same ordinal: one barrier instance where
                // the kernel calls two.
                let w = s[4].wait;
                s[6].wait = w;
            },
            "a barrier ordinal repeated",
        );
    }

    #[test]
    fn short_schedules_validate_and_count() {
        for layers in 1..5u32 {
            let s = schedule_for(layers, 2560, 9728, |l| l % 2 == 0);
            validate(&s).expect("valid");
            assert_eq!(s.len() as u32, 7 * layers);
            let tetra = (0..layers).filter(|l| l % 2 == 0).count() as u32;
            assert_eq!(n_bar(&s, 1), (layers - 1) + layers + tetra + N_TRAILING);
        }
    }

    #[test]
    fn slices_partition_every_site() {
        for d in [16u32, 1024, 2560, 4096, 9728] {
            for grid in [1u32, 3, 4, 142, 284] {
                let sl = slices(d, grid, 1);
                assert_eq!(sl.len(), grid as usize);
                assert_eq!(sl.first().unwrap().0, 0);
                assert_eq!(sl.last().unwrap().1, d);
                assert!(sl.windows(2).all(|w| w[0].1 == w[1].0));
                let (mn, mx) = sl.iter().fold((u32::MAX, 0), |(a, b), &(l, h)| (a.min(h - l), b.max(h - l)));
                assert!(mx - mn <= 1, "rows balanced within one: {mn}..{mx}");
            }
        }
        let sl = slices(9728, 142, 8);
        assert!(sl.iter().all(|&(l, h)| l % 8 == 0 && h % 8 == 0));
        assert_eq!(sl.last().unwrap().1, 9728);
    }

    #[test]
    fn wrap_safe_targets() {
        assert!(reached(5, 3));
        assert!(!reached(3, 5));
        assert!(reached(2, u32::MAX - 1));
        assert!(!reached(u32::MAX - 1, 2));
        assert_eq!(target(u32::MAX - 141, 0, 142), 0);
        // Across the wrap: a base just below 2^32, 97 barriers of 142.
        let base = u32::MAX - 1000;
        for n in 0..97u32 {
            let t = target(base, n, 142);
            assert!(reached(t, t));
            assert!(!reached(t.wrapping_sub(1), t));
            assert!(reached(t.wrapping_add(142), t));
        }
    }

    /// The draft's shared counter array, replayed: the second chained arm's
    /// counter waits are met before it has run a row. The ledger gives each
    /// arm its own region and epoch.
    #[test]
    fn counter_epochs_are_per_arm() {
        let s = served_4b_schedule();
        let grid = 142u32;
        let w = PERS_ARMS.iter().position(|a| a.name == "pchain_w").unwrap();
        let wp = PERS_ARMS.iter().position(|a| a.name == "pchain_wp").unwrap();
        let mut led = Ledger::new(N_PERS_ARMS);
        // identity launch then two timed launches of pchain_w
        for _ in 0..3 {
            let c = led.before(w, s.len()).unwrap();
            led.after(w, n_bar(&s, 1), grid, true);
            assert_eq!(c.cnt_offset as usize, w * counter_region(s.len()));
        }
        let c = led.before(wp, s.len()).unwrap();
        assert_eq!(c.epoch, 0, "pchain_wp's first launch waits for its own first signals");
        assert_ne!(c.cnt_offset, led.before(w, s.len()).unwrap().cnt_offset);
        // the draft's scheme: one array, the arm's own launch count. The
        // counter of a waited site holds 3 x grid after pchain_w; the draft's
        // target for pchain_wp's first launch is 1 x grid: already met.
        let counter_after_w = 3 * grid;
        let draft_target = grid;
        assert!(reached(counter_after_w, draft_target), "the replay shows the draft's stale read");
        assert_eq!(led.before(w, s.len()).unwrap().bar_base, 3 * n_bar(&s, 1) * grid);
        led.aborted();
        assert!(led.before(w, s.len()).is_err());
        led.reset();
        assert_eq!(led.before(w, s.len()).unwrap(), LaunchCounters { bar_base: 0, epoch: 0, cnt_offset: (w * counter_region(s.len())) as u32 });
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

    /// The order the header carries, read out of the header, so a drift
    /// between the kernel and this table fails here.
    #[test]
    fn order_is_the_headers() {
        let src = include_str!("../kernels/llvq_tetra48.cuh");
        let body = src.split("TETRA48_ORDER[24] = {").nth(1).unwrap().split('}').next().unwrap();
        let hdr: Vec<u8> = body.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        assert_eq!(hdr, ORDER.to_vec());
        let src = include_str!("../kernels/llvq_pers.cuh");
        let body = src.split("PERS_INV_ORDER[24] = {").nth(1).unwrap().split('}').next().unwrap();
        let inv: Vec<u8> = body.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        assert_eq!(inv, INV_ORDER.to_vec());
    }

    #[test]
    fn staging_is_a_bijection() {
        for (stride, trio) in [(24, false), (25, false), (28, true)] {
            for d_in in [2560u32, 4096, 9728] {
                let mut seen = std::collections::HashSet::new();
                let bound = (d_in / 24) * stride + d_in % 24;
                for i in 0..d_in {
                    let j = staged_index(i, d_in, stride, trio);
                    assert!(j < bound);
                    assert!(seen.insert(j));
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
        assert!((f16 as f64 / 1e9 - 0.9467).abs() < 0.0005);
        assert!(f16 < f32 && w < f16);
    }

    #[test]
    fn shared_budget_fits_the_opt_in() {
        for a in PERS_ARMS.iter() {
            assert!(shared_bytes(a) <= 101_376, "{}", a.name);
            assert_eq!(shared_bytes(a) % 16, 0, "{}: a whole number of 16-B lines", a.name);
        }
        let t28 = PERS_ARMS.iter().find(|a| a.name == "psmem_t28").unwrap();
        assert_eq!(shared_bytes(t28), SMEM_HEADER + 18_816 + 45_392);
        assert_eq!(TABLE_BYTES % 16, 0);
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
        let names: std::collections::HashSet<_> = PERS_ARMS.iter().map(|a| a.symbol).collect();
        assert_eq!(names.len(), N_PERS_ARMS, "every arm is its own symbol");
    }

    /// Every symbol of the table is defined in the kernel text with the
    /// table's template arguments.
    #[test]
    fn the_table_is_the_kernel_text() {
        let src = include_str!("../kernels/tetra48_pers.cu");
        for a in PERS_ARMS.iter() {
            let line = src
                .lines()
                .find(|l| l.starts_with("PERS_ARM(") && l.contains(&format!("({},", a.symbol)))
                .unwrap_or_else(|| panic!("{} is not instantiated", a.symbol));
            let args: Vec<String> = line
                .trim_start_matches("PERS_ARM(")
                .trim_end_matches(')')
                .split(',')
                .map(|s| s.trim().trim_end_matches('u').to_string())
                .collect();
            let want = [
                a.symbol.to_string(),
                a.threads.to_string(),
                a.mode.to_string(),
                a.stride.to_string(),
                a.trio.to_string(),
                a.prefetch.to_string(),
                a.waits.to_string(),
                a.prologue.to_string(),
            ];
            assert_eq!(args, want, "{}", a.name);
        }
    }

    /// The L2 job script runs every registered arm, in an order the parser
    /// accepts: an arm added here and forgotten there would not run.
    #[test]
    fn the_l2_job_runs_every_arm() {
        let script = include_str!("../../ops/jobs/pers-l2.sh");
        let line = script.lines().find(|l| l.starts_with("ARMS=")).expect("an ARMS= line");
        let got = parse_pers_arms(Some(line.trim_start_matches("ARMS="))).unwrap();
        assert_eq!(got, (0..N_PERS_ARMS).collect::<Vec<_>>());
    }

    #[test]
    fn the_parser_keeps_its_contract() {
        assert_eq!(parse_pers_arms(None).unwrap(), Vec::<usize>::new());
        assert!(parse_pers_arms(Some("")).is_err());
        assert!(parse_pers_arms(Some("tetra48h")).is_err());
        assert!(parse_pers_arms(Some("pnull,pnull")).is_err());
        assert!(parse_pers_arms(Some("nope")).is_err());
        assert!(parse_pers_arms(Some("pchain_w")).is_err());
        assert_eq!(parse_pers_arms(Some("pchain_w, psmem_t28 ,pnull")).unwrap(), vec![0, 8, 12]);
    }

    #[test]
    fn the_judge() {
        // identical
        assert_eq!(judge_row(0x3c00, 0x3c00, 1.0, 1.0, 1.0), RowVerdict::Identical);
        // one binary16 ulp off, the f32 value within 1e-5: a moved association
        let v = judge_row(0x3c00, 0x3c01, 1.000_49, 1.000_488_3, 10.0);
        assert!(matches!(v, RowVerdict::Moved { .. }), "{v:?}");
        // the f32 value off by 1e-3 of the row: a defect
        let v = judge_row(0x3c00, 0x3c08, 1.01, 1.0, 10.0);
        assert!(matches!(v, RowVerdict::Defect { .. }), "{v:?}");
        // a binary16 output 1e-4 away is NOT a defect under the 1e-3 rule,
        // where the draft's 1e-5 would have voided it
        let v = judge_row(0x3c00, 0x3c01, 1.000_976_6, 1.000_976_6 - 1e-6, 1.0);
        assert!(matches!(v, RowVerdict::Moved { .. }), "{v:?}");
        assert_eq!(judge_row(0x3c00, POISON_H, 0.0, 1.0, 1.0), RowVerdict::Unwritten);
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0x0001), 2f64.powi(-24));
        assert_eq!(f16_to_f64(0xc000), -2.0);
    }

    #[test]
    fn site_records_encode() {
        let s = served_4b_schedule();
        let r = site_record(&s[3], 0x1234_5678_9abc, 77, 128);
        assert_eq!(r[0], 0x5678_9abc);
        assert_eq!(r[1], 0x1234);
        assert_eq!(r[2], 2560);
        assert_eq!(r[3] & 0xffff, 170);
        assert_eq!(r[3] >> 16, 16);
        assert_eq!(r[6], 2 | (2 << 2), "o waits on v's counter");
        let r = site_record(&s[2], 0, 0, 0);
        assert_eq!(r[7], 1, "v signals, no prologue");
        let r = site_record(&s[4], 0, 0, 0);
        assert_eq!((r[6], r[7]), (1, 2), "gate: a grid barrier and the norm prologue");
        let r = site_record(&s[6], 0, 0, 0);
        assert_eq!((r[6], r[7]), (1, 4), "a Tetra down: a barrier and the rotation");
        let r = site_record(&s[7 * 5 + 4], 0, 0, 0);
        assert_eq!(r[7] >> 8, 5, "the layer rides in the flags");
    }

    #[test]
    fn rotation_plans() {
        assert_eq!(rot_mk(2560).unwrap(), (512, 5));
        assert_eq!(rot_mk(9728).unwrap(), (512, 19));
        assert_eq!(rot_mk(4096).unwrap(), (4096, 1));
        assert_eq!(rot_mk(12288).unwrap(), (4096, 3));
        assert_eq!(rot_mk(5120).unwrap(), (1024, 5));
        assert_eq!(rot_mk(17408).unwrap(), (1024, 17));
        assert!(rot_mk(25600).is_ok());
        assert!(rot_mk(24 * 100).is_err());
    }

    #[test]
    fn the_4e_schedule_has_the_design_counts() {
        for m in [ModelShape::Q4B, ModelShape::Q8B, ModelShape::Q14B] {
            assert_eq!(grid_waits_4e(&m), waits_4e(&m), "{m:?}");
            let s = schedule_4e(&m);
            assert_eq!(s.len() as u32, 2 + 4 * m.layers);
            assert_eq!(s.first(), Some(&(Phase4E::Embed, Wait4E::None)));
            assert_eq!(s.last(), Some(&(Phase4E::Head, Wait4E::Grid)));
            // every phase but the embedding and layer 0's q/k/v reads what
            // other CTAs wrote, and waits for it
            for (k, (p, w)) in s.iter().enumerate() {
                let free = matches!(p, Phase4E::Embed | Phase4E::Qkv(0));
                assert_eq!(*w == Wait4E::None, free, "phase {k}: {p:?}");
            }
        }
    }

    /// The card's three words as a 4E launch leaves them, for
    /// [`p4e_settle`]: each barrier adds `grid` to the counter, a fault
    /// raises its flag and stops every CTA at that barrier.
    struct FakeCounters {
        bar: u32,
        abort: u32,
        nan: u32,
    }

    impl P4eCounters for FakeCounters {
        fn flags(&mut self) -> Result<(u32, u32), String> {
            Ok((self.abort, self.nan))
        }
        fn zero(&mut self) -> Result<(), String> {
            (self.bar, self.abort, self.nan) = (0, 0, 0);
            Ok(())
        }
    }

    impl FakeCounters {
        /// One launch of `barriers` barriers from the host's base. The
        /// kernel waits for `base + k * grid` at barrier k: a counter that
        /// does not start at the base either deadlocks (behind it, the
        /// watchdog aborts) or lets the barriers pass early (ahead of it).
        /// Returns the barriers it passed.
        fn launch(&mut self, base: u32, barriers: u32, grid: u32, fault: Option<(u32, P4eFault)>) -> u32 {
            if self.nan != 0 {
                return 0; // the head reads the flag and leaves at its first token
            }
            if self.bar != base {
                self.abort = 1; // early passes are a race; count them as the abort they risk
                return 0;
            }
            for k in 1..=barriers {
                if let Some((at, f)) = fault {
                    if k == at {
                        match f {
                            P4eFault::Abort => self.abort = 1,
                            P4eFault::Nan => self.nan = 1,
                        }
                        self.bar = self.bar.wrapping_add(grid / 2); // some CTAs arrived
                        return k - 1;
                    }
                }
                self.bar = self.bar.wrapping_add(grid);
            }
            barriers
        }
    }

    #[test]
    fn a_faulted_4e_launch_does_not_poison_the_next() {
        let (grid, per) = (142u32, n_bar_4e(36));
        for fault in [P4eFault::Abort, P4eFault::Nan] {
            let mut dev = FakeCounters { bar: 0, abort: 0, nan: 0 };
            let mut base = 0u32;
            // a clean launch of 3 tokens
            assert_eq!(dev.launch(base, 3 * per, grid, None), 3 * per);
            assert_eq!(p4e_settle(&mut dev, &mut base), Ok(None));
            let (_, next) = p4e_after_launch(&[1, 2, 3], 3, base, 36, grid);
            base = next;
            assert_eq!(dev.bar, base, "the host's base follows the counter");
            // a launch that faults in its second token
            dev.launch(base, 3 * per, grid, Some((per + 5, fault)));
            assert_eq!(p4e_settle(&mut dev, &mut base), Ok(Some(fault)), "{fault:?} is reported");
            assert_eq!((dev.bar, base), (0, 0), "{fault:?}: counter and base start over together");
            // the next launches run, and stay in step
            for _ in 0..2 {
                assert_eq!(dev.launch(base, 2 * per, grid, None), 2 * per, "{fault:?}: the next launch runs");
                assert_eq!(p4e_settle(&mut dev, &mut base), Ok(None), "{fault:?}: the next launch is clean");
                base = p4e_after_launch(&[4, 5], 2, base, 36, grid).1;
                assert_eq!(dev.bar, base);
            }
        }
        // the NaN flag wins nothing over the abort: both set reads as Abort
        let mut dev = FakeCounters { bar: 7, abort: 1, nan: 1 };
        let mut base = 3u32;
        assert_eq!(p4e_settle(&mut dev, &mut base), Ok(Some(P4eFault::Abort)));
        assert_eq!((dev.bar, dev.abort, dev.nan, base), (0, 0, 0, 0));
        // a reset outside a fault (a launch the host cannot account for)
        let mut dev = FakeCounters { bar: 99, abort: 0, nan: 0 };
        let mut base = 42u32;
        p4e_reset(&mut dev, &mut base).unwrap();
        assert_eq!((dev.bar, base), (0, 0));
    }

    #[test]
    fn the_4e_launch_bookkeeping() {
        assert_eq!(n_bar_4e(36), 218);
        let (made, next) = p4e_after_launch(&[5, 9, 2, u32::MAX], 4, 10, 36, 142);
        assert_eq!(made, vec![5, 9, 2]);
        assert_eq!(next, 10 + 3 * 218 * 142);
        // stopped on EOS after one token: one token's barriers
        let (made, next) = p4e_after_launch(&[7, u32::MAX, u32::MAX], 3, 0, 2, 3);
        assert_eq!((made, next), (vec![7], 14 * 3));
        // the shared bytes: the 4B's row vector (9,728 floats) is larger than
        // attention's 26,752 bytes; a narrow model takes attention's
        assert_eq!(p4e_smem_bytes(2560, 9728, 4), (SMEM_HEADER + TABLE_BYTES + 4 * 9728).div_ceil(16) * 16);
        let cuh = include_str!("../kernels/llvq_pers.cuh");
        assert!(cuh.contains("#if PERS_4E_ROWS == 3\n#define PERS_4E_SCRATCH 16384u"), "p4e_scratch_bytes mirrors the scratch");
        assert!(cuh.contains("#elif PERS_4E_ROWS == 4\n#define PERS_4E_SCRATCH 4096u"), "form 4's scratch too");
        assert!(cuh.contains("#elif PERS_4E_ROWS >= 5\n#define PERS_4E_SCRATCH 7360u"), "forms 5 and 6's scratch too");
        assert!(cuh.contains("#define PERS_V7_FLOATS 1840u"), "form 5's scratch is the v7 area");
        let scratch: Vec<u32> = (1..=7).map(p4e_scratch_bytes).collect();
        assert_eq!(scratch, vec![0, 0, 16_384, 4_096, 7_360, 7_360, 7_360]);
        // the v7 area, and v8's ring, at the 4B, the 8B (4,096 wide, inter
        // 12,288, 4 groups) and the 14B (5,120, 17,408, 5 groups) fit under
        // sm_89's 101,376-byte opt-in
        for form in [5u32, 6, 7] {
            for (h, i, g) in [(2560u32, 9728u32, 4u32), (4096, 12_288, 4), (5120, 17_408, 5)] {
                let b = p4e_smem_bytes_form(form, h, i, g) + p4e_scratch_bytes(form);
                assert!(b <= 101_376, "form {form} at {h}/{i}/{g}: {b} B");
            }
        }
        assert_eq!(p4e_smem_bytes_form(6, 2560, 9728, 4), (SMEM_HEADER + TABLE_BYTES + 3 * 16_384 + 8_320 + 2_048).div_ceil(16) * 16);
        assert_eq!(p4e_smem_bytes_form(5, 2560, 9728, 4), p4e_smem_bytes(2560, 9728, 4));
        assert!(cuh.contains("#if PERS_4E_ROWS >= 6\n#define PERS_ATTN_RING 3u\n#else\n#define PERS_ATTN_RING 1u"), "p4e_attn_ring mirrors the ring");
        let inter_14b: u32 = 17_408;
        assert!(inter_14b.div_ceil(32) <= 576, "the 14B's down signs fit PERS_V7_SIGNS_MAX");
        // form 4's scratch holds every row of a KMAX x KMAX `small`
        assert_eq!(p4e_scratch_bytes(4) as usize, 4 * 32 * 32);
        assert_eq!(p4e_rows_define(1).unwrap(), "");
        assert_eq!(p4e_rows_define(3).unwrap(), "#define PERS_4E_ROWS 3u\n");
        assert_eq!(p4e_rows_define(4).unwrap(), "#define PERS_4E_ROWS 4u\n");
        assert_eq!(p4e_rows_define(5).unwrap(), "#define PERS_4E_ROWS 5u\n");
        assert_eq!(p4e_rows_define(6).unwrap(), "#define PERS_4E_ROWS 6u\n");
        assert_eq!(p4e_rows_define(7).unwrap(), "#define PERS_4E_ROWS 7u\n");
        assert!(p4e_rows_define(8).is_err());
        assert_eq!(p4e_smem_bytes_form(7, 2560, 9728, 4), p4e_smem_bytes_form(6, 2560, 9728, 4));
        assert!(P4E_BENCH_FORMS.iter().all(|f| P4E_ROWS_FORMS.contains(f)), "the bench forms are row forms");
        let ok = vec![("act".to_string(), 0x7000_0000u64), ("layer 0 ln1".to_string(), 0x7000_0100)];
        assert!(p4e_v6_inputs(&ok, 2560, 9728).is_ok());
        let bad = vec![("act".to_string(), 0x7000_0000u64), ("layer 3 ln2".to_string(), 0x7000_0108)];
        assert!(p4e_v6_inputs(&bad, 2560, 9728).unwrap_err().contains("layer 3 ln2"));
        assert!(p4e_v6_inputs(&ok, 2564, 9728).is_err());
        assert!(cuh.contains("#ifndef PERS_4E_ROWS\n#define PERS_4E_ROWS 1u"), "form 1 is the default");
        assert_eq!(p4e_smem_bytes(256, 512, 5), (SMEM_HEADER + TABLE_BYTES + 16384 + 10400 + 2560).div_ceil(16) * 16);
        let cuh = include_str!("../kernels/llvq_pers.cuh");
        for line in [
            "#define PERS_STAGE_U4 (2u * PERS_STAGE * (PERS_HD / 8u))",
            "#define PERS_STAGE (PERS_UPS * PERS_TT)",
            "#define PERS_UPS 4u",
            "#define PERS_PSTRIDE 130u",
            "#define PERS_ATTN_SMEM(groups) (16u * PERS_STAGE_U4 * PERS_ATTN_RING + 4u * PERS_UPS * (groups) * PERS_PSTRIDE + 4u * (groups) * PERS_HD)",
        ] {
            assert!(cuh.contains(line), "p4e_smem_bytes mirrors {line:?}");
        }
        // a full ring past n is not read
        let (made, _) = p4e_after_launch(&[1, 2, 3], 2, 0, 1, 1);
        assert_eq!(made, vec![1, 2]);
        // across the wrap
        let (_, next) = p4e_after_launch(&[1], 1, u32::MAX - 1, 1, 1);
        assert_eq!(next, 6);
        assert_eq!(p4e_symbol(1, 512).unwrap(), "pers_4e_tokens_f1_t512");
        assert!(p4e_symbol(3, 768).is_err() && p4e_symbol(0, 1024).is_err());
        // the symbols the table names are the kernel's
        let cu = include_str!("../kernels/pers_4e.cu");
        for f in 0..3u32 {
            for t in [768u32, 512] {
                let pf = p4e_rows_prefetch(t);
                assert!(cu.contains(&format!("PERS_4E({}, {f}, {t}, false, {pf})", p4e_symbol(f, t).unwrap())), "{f} {t}");
                assert!(cu.contains(&format!("PERS_4E({}, {f}, {t}, true, {pf})", p4e_stamp_symbol(f, t).unwrap())), "stamps {f} {t}");
            }
            assert!(p4e_stamp_symbol(f, 1024).is_err());
        }
        assert_eq!(p4e_events(36), 543);
    }

    /// The phase report on stamps made by hand: two CTAs, one layer, two
    /// tokens; every phase's wall, slowest and mean work known in advance.
    #[test]
    fn the_4e_phase_report_reads_the_stamps() {
        let (layers, grid, n) = (1u32, 2usize, 2u32);
        let ev = p4e_events(layers) as usize; // 18
        // Per token, CTA 0 and CTA 1, in order: the embedding (a wait), then
        // the layer's 15 events. Each grid wait: CTA 0 works 10, CTA 1 works
        // 10 + 2k (k the wait's index), a release 1 after the last arrival.
        // The two inner stamps of q/k/v and MLP split their phase's work: q/k/v
        // 3 then the rest; MLP 2, then gate/up to the inner wait's arrival, the
        // inner wait, 1, the rest. The head then takes 7 on both.
        let mut st = vec![0u64; n as usize * ev * grid];
        let mut clock = 100u64;
        let put = |st: &mut Vec<u64>, t: usize, e: usize, v: [u64; 2]| {
            for c in 0..2 {
                st[(t * ev + e) * grid + c] = v[c];
            }
        };
        for t in 0..n as usize {
            let mut rel = [clock; 2];
            // embedding, q/k/v, norm, attention, o, MLP: arrival offsets
            let arrivals = [0usize, 3, 5, 7, 9, 15];
            for (k, &e) in arrivals.iter().enumerate() {
                let arr = [rel[0] + 10, rel[1] + 10 + 2 * k as u64];
                if k == 1 {
                    put(&mut st, t, 2, [rel[0] + 3, rel[1] + 3]);
                }
                if k == 5 {
                    put(&mut st, t, 11, [rel[0] + 2, rel[1] + 2]);
                    let gu = [rel[0] + 5, rel[1] + 6];
                    put(&mut st, t, 12, gu);
                    let inner = gu[0].max(gu[1]) + 1;
                    put(&mut st, t, 13, [inner, inner]);
                    put(&mut st, t, 14, [inner + 1, inner + 1]);
                }
                let arr = if k == 5 { [arr[0].max(st[(t * ev + 14) * grid] + 1), arr[1].max(st[(t * ev + 14) * grid + 1] + 1)] } else { arr };
                let r = arr[0].max(arr[1]) + 1;
                put(&mut st, t, e, arr);
                put(&mut st, t, e + 1, [r, r]);
                rel = [r, r];
            }
            put(&mut st, t, ev - 1, [rel[0] + 7, rel[1] + 7]);
            clock = rel[0] + 7;
        }
        let r = p4e_phase_report(&st, n, layers, grid as u32).unwrap();
        assert_eq!(r.tokens_read, 1);
        assert_eq!(r.grain, 1);
        for (k, p) in r.phases.iter().enumerate().take(5) {
            let w = 10.0 + 2.0 * k as f64;
            assert_eq!(p.crit.0, w, "{}", p.name);
            assert_eq!(p.mean.0, (10.0 + w) / 2.0, "{}", p.name);
            assert_eq!(p.wall.0, w + 1.0, "{}", p.name);
        }
        assert_eq!(r.phases[6].wall.0, 7.0);
        let walls: f64 = r.phases.iter().map(|p| p.wall.0).sum();
        assert_eq!(walls, r.token.0);
        // the parts add up to their phase's mean work, CTA by CTA
        let sum = |lo: usize, hi: usize| r.subs[lo..hi].iter().map(|p| p.mean.0).sum::<f64>();
        assert_eq!(sum(0, 2), r.phases[1].mean.0, "q/k/v parts");
        assert_eq!(sum(2, 7), r.phases[5].mean.0, "MLP parts");
        assert_eq!(r.subs[0].mean.0, 3.0, "ln1 + rotation");
        assert_eq!(r.subs[2].mean.0, 2.0, "ln2 + rotation");
        assert_eq!(r.subs[3].crit.0, 4.0, "gate/up: CTA 1's 6 - 2");
        assert_eq!(r.subs[4].mean.0, 1.5, "the inner wait: 2 on CTA 0, 1 on CTA 1");
        assert_eq!(r.subs[5].mean.0, 1.0, "the down's input");
        // a release before the last arrival of its wait is refused
        let mut bad = st.clone();
        bad[(ev + 4) * grid] = bad[(ev + 3) * grid + 1] - 1;
        assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
        // so is an inner release before a gate/up arrival
        let mut bad = st.clone();
        bad[(ev + 13) * grid] = bad[(ev + 12) * grid + 1] - 1;
        assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
        // an unwritten stamp is refused
        let mut bad = st.clone();
        bad[ev * grid + 5] = 0;
        assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
        assert!(p4e_phase_report(&st[..ev * grid], 1, layers, grid as u32).is_err(), "one token reads nothing");
    }

    #[test]
    fn the_4e_wait_counts() {
        assert_eq!(waits_4e(&ModelShape::Q4B), 97);
        assert_eq!(waits_4e(&ModelShape::Q8B), 128);
        assert_eq!(waits_4e(&ModelShape::Q14B), 102);
        assert_eq!(step_a_plan(&ModelShape::Q4B).iter().filter(|p| p.tetra_down).count(), 24);
        assert_eq!(step_a_plan(&ModelShape::Q8B).iter().filter(|p| p.tetra_down).count(), 19);
        assert_eq!(step_a_plan(&ModelShape::Q14B).iter().filter(|p| p.tetra_down).count(), 21);
    }

    #[test]
    fn the_served_unit_is_the_published_one() {
        let t = served_unit_text(SERVED_DEFINES_SM89);
        assert_eq!(t.len(), 101_625);
        assert_eq!(
            crate::sha256::sha256_hex(t.as_bytes()),
            "6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812",
            "the served unit moved: the L2 reference would not be the served kernel"
        );
    }

    /// 4E v9's dynamic shared bytes at the three sealed shapes, against the
    /// 101,376 B a CTA may ask on sm_89: the 14B's int4 down input (17,408
    /// floats) fits, which is why the 14B's refusal is its untied head and
    /// not its width. The v7 area holds every width's sign words (576).
    #[test]
    fn the_4e_v9_shared_bytes_at_the_three_sealed_shapes() {
        let at = |hidden: u32, inter: u32, groups: u32| p4e_smem_bytes_form(7, hidden, inter, groups) + p4e_scratch_bytes(7);
        assert_eq!(at(2560, 9728, 4), 85_952, "the 4B, as step3-4e printed");
        assert_eq!(at(4096, 12288, 4), 85_952);
        assert_eq!(at(5120, 17408, 5), 96_064);
        for (h, i) in [(2560u32, 9728u32), (4096, 12288), (5120, 17408)] {
            assert!(at(h, i, 5) <= 101_376);
            assert!(i.div_ceil(32) <= 576 && h.div_ceil(32) <= 576, "the v7 area at {h}/{i}");
        }
    }

    /// 4E v9's module at the L40S's tile, the one the served-v9 doors gate
    /// on: the step-3 module with the head's own pointers and the Tetra o
    /// phase added (step3-4e timed d9da09aa…, 250,057 B).
    #[test]
    fn the_4e_v9_module_is_the_one_the_doors_gate_on() {
        let t = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text("#define TILE_BLOCKS 64u\n"));
        assert_eq!(t.len(), 252_757);
        assert_eq!(
            crate::sha256::sha256_hex(t.as_bytes()),
            "1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530",
            "the 4E v9 module moved: the served-v9 doors gate on this digest"
        );
    }

    /// E1's reference module is the served v9 text with the logits edits
    /// only: the token and stamp symbols' text is v9's, the template
    /// parameter defaults to false, and three logits symbols follow.
    #[test]
    fn the_e1_reference_is_v9_and_its_logits_symbols() {
        let tile = "#define TILE_BLOCKS 64u\n";
        let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(tile));
        let r = p4e_e1_unit_text(tile, "v9").unwrap();
        assert_ne!(r, v9);
        // the reference minus the logits edits is v9 byte for byte
        let mut back = r.clone();
        for (_, from, to) in p4e_e1_edits("v9").unwrap() {
            assert_eq!(back.matches(to.as_str()).count(), 1, "{to:?}");
            back = back.replacen(to.as_str(), &from, 1);
        }
        assert_eq!(back, v9, "the E1 reference is v9 and the logits edits, nothing else");
        for f in 0..3 {
            assert!(r.contains(&format!("PERS_4E_LOGITS({}, {f})", p4e_logits_symbol(f).unwrap())));
        }
        assert!(p4e_logits_symbol(3).is_err());
        assert!(p4e_e1_unit_text(tile, "v9x").is_err());
    }

    /// v9ep and v9fl are E1's v9 reference with the rc edits and nothing
    /// else: undoing the edits gives the reference back byte for byte; the body
    /// is inserted once with its define; no call site of v9's warp lists is
    /// left on a Tetra row phase of the 512-thread path; the served v9 module
    /// and the served unit keep their pins (their own tests).
    #[test]
    fn the_rc_arms_are_v9_with_the_body_on_every_tetra_phase() {
        let tile = "#define TILE_BLOCKS 64u\n";
        let r = p4e_e1_unit_text(tile, "v9").unwrap();
        for (arm, body) in P4E_RC_ARMS {
            let t = p4e_e1_unit_text(tile, arm).unwrap();
            let mut back = t.clone();
            for (_, from, to) in p4e_rc_edits(body).unwrap() {
                assert_eq!(back.matches(to.as_str()).count(), 1, "{arm}: {to:?}");
                back = back.replacen(to.as_str(), &from, 1);
            }
            assert_eq!(back, r, "{arm}: E1's v9 and the rc edits, nothing else");
            assert_eq!(t.matches(&format!("#define PERS_RC_ARM {body}u\n")).count(), 1);
            assert_eq!(t.matches("#ifndef PERS_4E_RC_CUH").count(), 1);
            assert_eq!(t.matches("#ifndef PERS_4E_RC2_CUH").count(), usize::from(body >= 3), "{arm}");
            let call = if body >= 3 { "pers_rc2_rows<PERS_RC_ARM>(" } else { "pers_rc_rows<PERS_RC_ARM>(" };
            assert_eq!(t.matches(call).count(), 4, "{arm}: q/k, gate/up, down, o");
            assert_eq!(t.matches("_rows<PERS_RC_ARM>(").count(), 4, "{arm}: no call of the other family");
            // pers_4e_rows keeps its definition and no caller
            assert_eq!(t.matches("pers_4e_rows(").count(), 1, "{arm}");
            assert_eq!(p4e_e1_owes_bits(arm), !rc_body_is_flat(body), "{arm}");
            assert_eq!(p4e_rc_body(arm), Some(body));
        }
        assert!(p4e_rc_edits(0).is_err() && p4e_rc_edits(7).is_err());
        for arm in ["v9", "v9np", "v9mv", "v9ft", "v9ep", "v9ep2", "v9ep3"] {
            assert!(p4e_e1_owes_bits(arm), "{arm}");
        }
        for arm in ["v9fl", "v9fl2", "v9fl3"] {
            assert!(!p4e_e1_owes_bits(arm), "{arm}");
        }
        // every research arm is an E1 arm
        assert!(P4E_RC_ARMS.iter().all(|a| P4E_E1_ARMS.contains(&a.0)));
    }

    /// Where the rc bodies sit in 4E v9's buffer at the three sealed shapes
    /// (*computed*): the 4B and the 8B fit every phase in the attention
    /// phase's region, the 14B's down input fills its row vector to the v7
    /// area and is refused by name, for both bodies. The numbers are the
    /// shared-memory report of the R6 prereg.
    #[test]
    fn the_rc_layout_at_the_three_sealed_shapes() {
        assert_eq!(rc_padded_floats(2560), 2576);
        assert_eq!(rc_padded_floats(9728), 9780);
        assert_eq!(rc_padded_floats(4096), 4120);
        assert_eq!(rc_padded_floats(12288), 12352);
        let ends = |body: u32, h: u32, i: u32, ctx: u32, o: bool, g: u32| {
            p4e_rc_layout(body, h, i, ctx, o, true, g, 142).map(|v| v.iter().map(|p| (p.phase, p.end, p.limit)).collect::<Vec<_>>())
        };
        // the 4B: 142 CTAs, 69 gate/up pairs at most
        assert_eq!(ends(1, 2560, 9728, 4096, false, 4).unwrap(), vec![("q/k", 7168, 14880), ("gate/up", 7237, 14880), ("down", 11828, 14880)]);
        assert_eq!(ends(2, 2560, 9728, 4096, false, 4).unwrap(), vec![("q/k", 9744, 14880), ("gate/up", 9813, 14880), ("down", 11828, 14880)]);
        // the 8B: its Tetra o at the context's width
        assert_eq!(
            ends(2, 4096, 12288, 4096, true, 4).unwrap(),
            vec![("q/k", 14360, 14880), ("gate/up", 14447, 14880), ("down", 14400, 14880), ("o", 6168, 14880)]
        );
        assert!(ends(1, 4096, 12288, 4096, true, 4).is_ok());
        // the 14B: refused by name, both bodies; flat's q/k (17,436 floats) already passes the v7 area at 17,408
        for (body, phase) in [(1, "the down phase"), (2, "the q/k phase")] {
            let e = ends(body, 5120, 17408, 5120, false, 5).unwrap_err();
            assert!(e.contains(phase) && e.contains("refused"), "{e}");
        }
        assert!(p4e_rc_layout(7, 2560, 9728, 4096, false, true, 4, 142).is_err());
        // the variants: their families' regions; the one-chunk bodies fit the 4B on 142 CTAs (largest slice: flat's
        // gate/up, 138 rows x 14 runs = 1,932 slots), and fl2, fl3 are refused on the 8B (174 x 22 = 3,828)
        assert_eq!(ends(3, 2560, 9728, 4096, false, 4).unwrap(), ends(1, 2560, 9728, 4096, false, 4).unwrap());
        assert_eq!(ends(4, 2560, 9728, 4096, false, 4).unwrap(), ends(1, 2560, 9728, 4096, false, 4).unwrap());
        for body in [5, 6] {
            assert_eq!(ends(body, 2560, 9728, 4096, false, 4).unwrap(), ends(2, 2560, 9728, 4096, false, 4).unwrap());
            let e = ends(body, 4096, 12288, 4096, true, 4).unwrap_err();
            assert!(e.contains("one chunk") && e.contains("gate/up phase's slice needs 3828"), "{e}");
        }
        assert!(ends(4, 4096, 12288, 4096, true, 4).is_ok(), "ep3 on the 8B: 174 rows a slice");
        // the twin's grids of 3 to 5 CTAs do not fit one chunk: the one-chunk arms are twin-tested on 142
        assert!(p4e_rc_layout(5, 2560, 9728, 4096, false, true, 4, 3).is_err());
    }

    /// The arms differ from the reference in prefetch lines only
    /// ([`p4e_e1_structure`]), and each does what it says: np issues no
    /// prefetch in the attention phase, mv issues its own and keeps v9's
    /// down prefetch and layer 0's q/k/v prefetch.
    #[test]
    fn the_e1_arms_differ_from_v9_in_prefetch_lines_only() {
        let tile = "#define TILE_BLOCKS 64u\n";
        let r = p4e_e1_unit_text(tile, "v9").unwrap();
        let np = p4e_e1_unit_text(tile, "v9np").unwrap();
        let mv = p4e_e1_unit_text(tile, "v9mv").unwrap();
        p4e_e1_structure(&r, &r).unwrap();
        p4e_e1_structure(&r, &np).unwrap();
        p4e_e1_structure(&r, &mv).unwrap();
        assert_eq!(r.matches("prefetch_next();").count(), 2);
        for t in [&np, &mv] {
            assert_eq!(t.matches("prefetch_next();").count(), 0, "no o or gate/up prefetch in the attention phase");
            assert_eq!(t.matches("pers_l2_prefetch(A->down.words").count(), 1, "the down prefetch stays");
        }
        assert!(!np.contains("pers_e1_"));
        assert_eq!(mv.matches("if (IN4E) pers_e1_prefetch_staged();").count(), 1);
        assert_eq!(mv.matches("pers_e1_stage_next(Ag->layers, l + 1u,").count(), 1);
        assert_eq!(mv.matches("if (l == 0u) {  // e1-pf").count(), 1);
        // the served v9 module's digest is pinned by
        // the_4e_v9_module_is_the_one_the_doors_gate_on, which this does not touch
    }

    /// The structural check kills what it must: a line of arithmetic added
    /// untagged or tagged, a line dropped, a barrier or a foreign store in
    /// the helpers' block.
    #[test]
    fn the_e1_structure_check_kills_its_mutants() {
        let tile = "#define TILE_BLOCKS 64u\n";
        let r = p4e_e1_unit_text(tile, "v9").unwrap();
        let mv = p4e_e1_unit_text(tile, "v9mv").unwrap();
        let once = |t: &str, from: &str, to: &str| {
            assert_eq!(t.matches(from).count(), 1, "{from:?}");
            t.replacen(from, to, 1)
        };
        let anchor = "            PERS_WAIT(e0 + 5u);\n";
        for (tag, m) in [
            ("untagged", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;\n"))),
            ("tagged store", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;  // e1-pf\n"))),
            ("tagged prefetch store", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;  // prefetch\n"))),
            ("dropped line", once(&mv, anchor, "")),
            ("helper barrier", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    __syncthreads();\n")),
            ("helper store", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    layers[0].qkv.q[0] = 0u;\n")),
            ("helper deref store", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    *pers_hdr() = 0u;\n")),
            ("helper open", once(&mv, "// e1-pf end\n", "")),
        ] {
            assert!(p4e_e1_structure(&r, &m).is_err(), "structure mutant {tag} survived");
        }
    }

    /// The verify pass's module at the L40S's tile: the 4E v9 module the
    /// served doors gate on, unchanged, then `pers_4ev.cu`. Pinned at the
    /// module spec-final ran (288,366 B at d7686d86, with the phase stamps of
    /// 486fa45); spec-gate and spec-lookup ran the earlier 287,744 B at ab162e7f.
    #[test]
    fn the_verify_module_is_the_v9_module_and_the_verify_pass() {
        let tile = "#define TILE_BLOCKS 64u\n";
        let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(tile));
        let t = p4v_unit_text(tile);
        assert!(t.starts_with(&v9) && t[v9.len()..] == format!("\n{}", P4V_PART.1), "the verify module is the v9 module and pers_4ev.cu");
        assert_eq!(t.len(), 288_366);
        assert_eq!(
            crate::sha256::sha256_hex(t.as_bytes()),
            "d7686d86b990b079c47a754ccb5675e090c0b9d19e1b3710558967614c3944e5",
            "the verify module moved: the speculative gate gates on this digest"
        );
    }

    /// The unit a v9 config compiles: the served unit unchanged, then the
    /// tiled int4 readers. Pinned so the door job's digest gate is this text.
    #[test]
    fn the_tiled_served_unit_is_the_served_unit_and_the_tile_readers() {
        let (plain, tiled) = (served_unit_text(SERVED_DEFINES_SM89), served_unit_text_tiled(SERVED_DEFINES_SM89));
        assert!(tiled.starts_with(&plain), "the served unit's text moved under v9");
        for sym in ["tv_q4_ht(", "tv_q4_hht("] {
            assert_eq!(tiled.matches(sym).count(), 1, "{sym}");
            assert!(!plain.contains(sym), "{sym} leaked into the other configs' unit");
        }
        assert_eq!(tiled.len(), 106_947);
        assert_eq!(
            crate::sha256::sha256_hex(tiled.as_bytes()),
            "e269ea5d11b7e40ba0682f536f52ed9f09abfaa9fd6f3185eb306edca5c60e53",
            "the v9 served unit moved: the served-v9 door gates on this digest"
        );
    }

    #[test]
    fn the_pers_unit_carries_every_symbol() {
        let t = pers_unit_text("#define TILE_BLOCKS 64u\n");
        for a in PERS_ARMS.iter() {
            assert!(t.contains(&format!("PERS_ARM({},", a.symbol)), "{}", a.symbol);
        }
        for sym in ["tv_tetra48_persall", "tv_pers_bar", "tv_pers_ring", "tv_pers_empty", "pers_ref_rmsnorm", "pers_seg_tetra_v0", "pers_seg_q4_v0"] {
            assert!(t.contains(sym), "{sym}");
        }
        // the pass's variants: each symbol defined once, with the flags its
        // name says, and the tables-in-shared column agreeing with the name
        for (name, sym, ts) in PERSALL_VARIANTS.iter().skip(1) {
            let (pf, l2) = (name.contains("_pf"), name.contains("_l2"));
            assert_eq!(*ts, name.contains("_ts"), "{name}");
            assert!(t.contains(&format!("TOCC_PERSALL_V({sym}, {pf}, {ts}, {l2})")), "{sym}");
        }
        assert!(t.contains("extern \"C\" __global__ void tv_tetra48_persall(const unsigned long long* __restrict__ sites,"));
        // composition: every part's guard is defined by an earlier part
        let pos = |n: &str| PERS_UNIT_PARTS.iter().position(|p| p.0 == n).unwrap();
        assert!(pos("matvec.cu") < pos("llvq_pers.cuh"));
        assert!(pos("llvq_tetra48.cuh") < pos("tetra48_persall.cu"));
        assert!(pos("llvq_rot.cuh") < pos("llvq_pers.cuh"));
        assert!(pos("llvq_pers.cuh") < pos("pers_segment.cu"));
        assert!(pos("pers_segment.cu") < pos("pers_4e.cu"));
        for sym in ["pers_4e_tokens_f0", "pers_4e_tokens_f1", "pers_4e_tokens_f2"] {
            assert!(t.contains(sym), "{sym}");
        }
    }

    #[test]
    fn persall_sites_pack_as_the_kernel_unpacks() {
        let w = persall_site_words(1, 2, 3, 4, 5, 6, 405, 8, 1216, 1234, 320);
        assert_eq!(w[6] & 0xffff, 405);
        assert_eq!((w[6] >> 16) & 0xffff, 8);
        assert_eq!(w[6] >> 32, 1216);
        assert_eq!(w[7] & 0xffff_ffff, 1234);
        assert_eq!(w[7] >> 32, 320);
        let cu = include_str!("../kernels/tetra48_persall.cu");
        assert!(cu.contains("#define LLVQ_TOCC_SITE_WORDS 8u"));
        let (g0, total) = persall_groups(&[6144, 2560, 19456, 2560]);
        assert_eq!(g0, vec![0, 768, 1088, 3520]);
        assert_eq!(total, 3840);
    }

    #[test]
    fn args_images_have_the_struct_sizes() {
        assert_eq!(PersArgsHost::default().bytes().len(), PERS_ARGS_BYTES);
        assert_eq!(std::mem::size_of::<PersLaunch>(), PERS_LAUNCH_BYTES);
        // the offsets themselves: tests/pers_matches_served.rs compiles the
        // kernels' struct definitions and checks every one
    }

    #[test]
    fn med_range_is_the_middle() {
        assert_eq!(med_range(&[3.0, 1.0, 2.0]), (2.0, 1.0, 3.0));
        assert_eq!(med_range(&[4.0, 1.0, 2.0, 3.0]), (2.5, 1.0, 4.0));
        assert!(med_range(&[]).0.is_nan());
    }

    #[test]
    fn the_f64_row_reference() {
        // one block, y = e_0 * 4 (n2 = 16), gain 0 centroid 2: w_0 = 2 * 4 / 4 = 2
        let mut y = [0i32; 24];
        y[0] = 4;
        let x: Vec<f32> = (0..26).map(|i| i as f32 + 1.0).collect();
        let (v, m) = tetra_row_f64(1, |_| (y, 0), [2.0, 3.0], 0.5, &[0.25, -1.0], &x);
        assert_eq!(v, 2.0 * 0.5 * 1.0 + 0.25 * 25.0 - 26.0);
        assert_eq!(m, 1.0 + 6.25 + 26.0);
    }
}
