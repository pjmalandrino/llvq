//! The persistent Tetra kernels, host half: what every launch of 4E
//! (`kernels/pers_4e.cu`), of its verify pass, of route P's segment kernel
//! and of the persistent bench sections depends on, kept portable so
//! `cargo test` checks it on the Mac.
//!
//! Design: `docs/archive/debit-vague5-2026-11-02/persistent.md` §3 and §5.
//! Preregs: `proofs/preregistration-pers-l1-*.md` (step L1) and the plan-300
//! ones (`-step1-micro-*`, `-rows-grid-*`).
//!
//! What lives here, and why each piece cannot wait for a card:
//!
//! - the unit texts the doors compile, and the tests that pin their digests;
//! - the device structs, mirrored byte for byte;
//! - row slices, the staging maps, their bank maps and the byte ledger;
//! - the route-P segment plan per model, and 4E's wait counts and barrier
//!   bookkeeping;
//! - the tables of the one-launch pass and of the row-loop microbenchmark.
//!
//! L2's bench arms (`LLVQ_PERS_ARMS`) and their host mirrors (arm table,
//! ledger, row judge, `PersArgs` image) were removed after their verdict.
//! Their kernels stay in `tetra48_pers.cu`, a part of [`PERS_UNIT_PARTS`], so
//! the 4E module keeps its bytes. The site schedule stays
//! ([`served_4b_schedule`]): `bin/rowbench` orders the bare file's rows by it.
//!
//! A grid one barrier short deadlocks, and a counter target one epoch off
//! reads stale rows: both fail silently or hang on a rented card.

mod e1;
mod launch;
mod units;
mod verify;
mod report;
mod rows;

pub use e1::*;
pub use launch::*;
pub use units::*;
pub use verify::*;
pub use report::*;
pub use rows::*;

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

/// Bytes of the decoder tables a persistent kernel copies to shared memory, invnorm
/// included: 16,384 + 128 + 2,048 + 128 + 128.
pub const TABLE_BYTES: u32 = 18_816;
/// Fixed header of the dynamic shared buffer: claim cursor, wait flags, norm
/// partials (32 floats) and spare, kept 16-byte aligned. The persistent
/// kernels declare no static `__shared__`, so the host harness can give each
/// CTA its own copy.
pub const SMEM_HEADER: u32 = 256;

/// The register-WHT rotation prologue's plan for a width: `m` the largest
/// power of two dividing `n`, `k = n / m`, as `llvq_search::rotation::Rotation` and the
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

/// The tolerance of an f32 value against its f64 reference, relative to the
/// row's `Σ|w·x|` (floor 1): the bench's convention for f32 outputs.
pub const F32_TOL: f64 = 1e-5;

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

/// 4E's row forms (`PERS_4E_ROWS` in `llvq_pers.cuh`): 1, the stream
/// (4E v3); 2, v2's row loop on the same static rows
/// (4E v4); 3, form 2 with each CTA's last rows split over every warp (4E v5);
/// 4, form 2 with ln1, ln2 and the down's input in one round trip a thread
/// (4E v6, `llvq_pers.cuh` section 7b); 5, form 4 with the launch's
/// constants (argument structs, row slices, sign words, `small`) copied into
/// shared memory during the barrier before they are read, and the o phase's
/// input widened from vectors (4E v7, section 7c); 6, form 5 with the
/// attention phase's steps staged by `cp.async` into a ring of
/// [`p4e_attn_ring`] stages, several steps in flight (4E v8, K3a of
/// `docs/plan-300.md`); 7, form 6 with the int4 rows read on tiled words
/// (4E v9, K11), the form the served configs run.
pub const P4E_ROWS_FORMS: [(u32, &str); 7] = [(1, "v3"), (2, "v4"), (3, "v5"), (4, "v6"), (5, "v7"), (6, "v8"), (7, "v9")];

/// The forms `fusedrun LLVQ_PERS_4E` times, the first the reference of the
/// others' round-by-round ratios: v4, the form served from 2026-11-05 until
/// v9; v6, the reference v9's door proves its tokens against; v9 (v8 with the
/// int4 rows on tiled words, K11), the form the served configs run. v3 and v5
/// were read by the shaker (job 6aec9a1d), v7 and v8 by plan-300's steps 2
/// and 3 (jobs 6aef3f69, 6aef51d3); `served.rs` refuses v5, v7 and v8 by
/// name, and their code stays in the kernel as v9's (forms 5 and 6 are
/// cumulative into 7).
pub const P4E_BENCH_FORMS: [(u32, &str); 3] = [(2, "v4"), (4, "v6"), (7, "v9")];

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

#[cfg(test)]
mod tests;
