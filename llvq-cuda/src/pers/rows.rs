//! The row-loop microbenchmarks: the one-launch Tetra pass (persall), the
//! vrows, trows, q4rows and counter arms, their byte ledgers, levers and
//! stamp readers. One file for the row bench's host mirror.

use super::{pers_unit_text, SMEM_HEADER, TABLE_BYTES};

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
        include_str!("../../kernels/pers_trows.cu"),
        include_str!("../../kernels/pers_q4rows.cu"),
        include_str!("../../kernels/pers_ctr.cu")
    )
}

/// The verify-row microbenchmark's module (`pers_vrows.cu`): the row-loop
/// unit, then the verify rows. Its own module, so neither 4E's nor the row
/// loops' change bytes.
pub fn vrows_unit_text(tile_define: &str) -> String {
    format!("{}\n{}", trows_unit_text(tile_define), include_str!("../../kernels/pers_vrows.cu"))
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
const VROWS_TABLE_BYTES: u32 = 16_384 + 128 + 2_048 + 128 + 256 + 512 + 128;

/// `vrows_area_bytes`: one int16 activation of `nb` blocks, its bf16 scales,
/// padded to 16, and 64 bytes of f32 tail.
fn vrows_area_bytes(nb: u32) -> u32 {
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
const CTR_GROUP: u32 = 8;
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
pub(super) fn trows_warp_rows(lo: u32, hi: u32, warp: u32, nw: u32) -> u32 {
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
