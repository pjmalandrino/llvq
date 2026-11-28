//! 4E's launch bookkeeping and the device structs mirrored byte for byte:
//! shared bytes, barrier counters, route P's segment plan and the argument
//! images the host writes. One file because each struct's offsets meet here.

use super::{n_bar_4e, ModelShape, SMEM_HEADER, TABLE_BYTES};

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

/// Shared bytes of a 4E CTA of row form `form`, before the scratch: the
/// header, the decoder tables, then one region that is the f32 vector of the
/// row phases (`max(inter, 2 x hidden)` floats) or the attention phase's
/// stage, unit states and q (`PERS_ATTN_SMEM` of llvq_pers.cuh), whichever is
/// larger, rounded to 16 bytes.
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
// The device structs, mirrored byte for byte. `tests/pers_matches_served/served_text.rs`
// compiles the kernels' own struct definitions and checks every offset.
// ---------------------------------------------------------------------------

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
/// own definitions by `tests/pers_matches_served/served_text.rs`.
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
