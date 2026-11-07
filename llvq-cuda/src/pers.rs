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
pub const P4E_ROWS_FORMS: [(u32, &str); 6] = [(1, "v3"), (2, "v4"), (3, "v5"), (4, "v6"), (5, "v7"), (6, "v8")];

/// The forms `fusedrun LLVQ_PERS_4E` times, the first the reference of the
/// others' round-by-round ratios: v4, the served form since 2026-11-05; v6,
/// served on the operator's go of 2026-11-07; v7 (K2) and v8 (K2 and K3a),
/// step 2 of `docs/plan-300.md`. v3 and v5 were read by the shaker (job
/// 6aec9a1d).
pub const P4E_BENCH_FORMS: [(u32, &str); 4] = [(2, "v4"), (4, "v6"), (5, "v7"), (6, "v8")];

/// The line a module of row form `form` starts with; form 1 is the
/// default and adds nothing, so its module keeps its text.
pub fn p4e_rows_define(form: u32) -> Result<String, String> {
    match form {
        1 => Ok(String::new()),
        2..=6 => Ok(format!("#define PERS_4E_ROWS {form}u\n")),
        _ => Err(format!("4E row form {form}: the forms are 1 (v3), 2 (v4), 3 (v5), 4 (v6), 5 (v7) and 6 (v8)")),
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

/// The shared scratch after the activation staging (`PERS_4E_SCRATCH`):
/// form 3's split rows, form 4's staged rows of the rotations' `small`,
/// form 5's v7 area (`PERS_V7_FLOATS` floats: `small`, the argument structs,
/// the CTA's slices and up to 576 sign words).
pub fn p4e_scratch_bytes(form: u32) -> u32 {
    match form {
        3 => 16_384,
        4 => 4_096,
        5 | 6 => 7_360,
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
pub const P4E_LAYER_BYTES: usize = 592;
pub const P4E_ARGS_BYTES: usize = 216;
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
        b
    }
}

/// The launch-wide constants of 4E: 22 pointers, then 8 u32 and 2 f32.
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

/// The row-loop arms (`pers_trows.cu`): name, symbol, whether every row owes
/// the served bits, CTAs a SM. The occupancy arm `v2_o2` runs V2 at two CTAs a
/// SM on the sites of d_in 4,096 or less, beside `v2_s`, the same sites at one.
/// The last five are K4's arms of plan-300 step 1, each against `v2`.
pub const TROWS_ARMS: [(&str, &str, bool, u32); 15] = [
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
];

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
    use super::*;

    #[test]
    fn the_trows_unit_carries_every_arm() {
        let t = trows_unit_text("#define TILE_BLOCKS 64u");
        assert!(t.starts_with(&pers_unit_text("#define TILE_BLOCKS 64u")), "4E's module bytes come first, unchanged");
        for (name, sym, _, minb) in TROWS_ARMS {
            if name == "v2_s" {
                continue; // the symbol of v2, on fewer sites
            }
            let arm = sym.trim_start_matches("tv_trows_").trim_end_matches("_o2").to_uppercase();
            assert!(t.contains(&format!("TROWS_KERNEL({sym}, TROWS_{arm}, {minb})")), "{sym}");
        }
        assert_eq!(trows_smem_bytes(9728), 57_984, "4E's own shared bytes at d_in 9,728");
        assert_eq!(trows_k2_smem_bytes(), 96_896);
        assert!(t.contains("#define TROWS_X2_AT 9728u"), "K2's second activation sits after the widest first one");
        assert!(trows_small_only("v2_o2") && trows_small_only("v2_s") && !trows_small_only("v2"));
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
        let scratch: Vec<u32> = (1..=6).map(p4e_scratch_bytes).collect();
        assert_eq!(scratch, vec![0, 0, 16_384, 4_096, 7_360, 7_360]);
        // the v7 area, and v8's ring, at the 4B, the 8B (4,096 wide, inter
        // 12,288, 4 groups) and the 14B (5,120, 17,408, 5 groups) fit under
        // sm_89's 101,376-byte opt-in
        for form in [5u32, 6] {
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
        assert!(p4e_rows_define(7).is_err());
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
