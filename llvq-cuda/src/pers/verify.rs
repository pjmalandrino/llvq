//! The verify passes: 4E's n-row verify (`pers_4ev.cu`) and the tree verify
//! pass (`pers_4et*.cu`) with its arms, plans, phase readers and the
//! attention tile's full-tile path. One file for one family of modules.

use super::*;

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
pub const P4V_PART: (&str, &str) = ("pers_4ev.cu", include_str!("../../kernels/pers_4ev.cu"));

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
pub const P4T_A11I_PART: (&str, &str) = ("pers_4et_a11i.cu", include_str!("../../kernels/pers_4et_a11i.cu"));

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
pub const P4T_A10RC_PART: (&str, &str) = ("pers_4et_a10rc.cu", include_str!("../../kernels/pers_4et_a10rc.cu"));

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
pub const P4T_A10T2_PART: (&str, &str) = ("pers_4et_a10t2.cu", include_str!("../../kernels/pers_4et_a10t2.cu"));

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
pub const P4T_A12G32_PART: (&str, &str) = ("pers_4et_a12g32.cu", include_str!("../../kernels/pers_4et_a12g32.cu"));

// ---------------------------------------------------------------------------
// The attention tile's full-tile path (ft arms,
// docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt): a module text with
// `pers_attn_full.cuh` inserted before `pers_attn_tile` and a branch at the
// head of `pers_attn_tile` that runs it when the tile holds PERS_TT
// positions. A text edit of the parent module, so a12ip, a12ipg, a12ipg32
// and 4E v9 keep their bytes.
// ---------------------------------------------------------------------------

/// `pers_attn_full.cuh`, inserted by [`pers_ft_edit`] in the ft arms only.
pub const PERS_FT_PART: (&str, &str) = ("pers_attn_full.cuh", include_str!("../../kernels/pers_attn_full.cuh"));

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
pub const P4T_A12G_PART: (&str, &str) = ("pers_4et_a12g.cu", include_str!("../../kernels/pers_4et_a12g.cu"));

/// u32 words of a projection in the served order, as the tree pass reads it
/// in place: Tetra `stride_u32` a row, int4 d_in / 8. The bytes a12ip reads
/// where a12 reads [`p4t_tiled_words`].
pub fn p4t_served_words(int4: bool, d_out: u32, stride_u32: u32, d_in: u32) -> u64 {
    d_out as u64 * if int4 { (d_in / 8) as u64 } else { stride_u32 as u64 }
}

/// `pers_4et_a12.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A12_PART: (&str, &str) = ("pers_4et_a12.cu", include_str!("../../kernels/pers_4et_a12.cu"));

/// `pers_4et_a10t.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A10T_PART: (&str, &str) = ("pers_4et_a10t.cu", include_str!("../../kernels/pers_4et_a10t.cu"));

/// `pers_4et_a11.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A11_PART: (&str, &str) = ("pers_4et_a11.cu", include_str!("../../kernels/pers_4et_a11.cu"));

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
pub const P4T_A10_PART: (&str, &str) = ("pers_4et_a10.cu", include_str!("../../kernels/pers_4et_a10.cu"));

/// `pers_4et_a9.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A9_PART: (&str, &str) = ("pers_4et_a9.cu", include_str!("../../kernels/pers_4et_a9.cu"));

/// `pers_4et_a8.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A8_PART: (&str, &str) = ("pers_4et_a8.cu", include_str!("../../kernels/pers_4et_a8.cu"));

/// `pers_4et_a6.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A6_PART: (&str, &str) = ("pers_4et_a6.cu", include_str!("../../kernels/pers_4et_a6.cu"));

/// `pers_4et_a5.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A5_PART: (&str, &str) = ("pers_4et_a5.cu", include_str!("../../kernels/pers_4et_a5.cu"));

/// `pers_4et_a4.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_A4_PART: (&str, &str) = ("pers_4et_a4.cu", include_str!("../../kernels/pers_4et_a4.cu"));

/// `pers_4et_stream.cu`, embedded so the Mac hashes what the card compiles.
pub const P4T_STREAM_PART: (&str, &str) = ("pers_4et_stream.cu", include_str!("../../kernels/pers_4et_stream.cu"));

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
pub const P4T_PART: (&str, &str) = ("pers_4et.cu", include_str!("../../kernels/pers_4et.cu"));
