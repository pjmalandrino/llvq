//! The int8 KV cache of 4E's research arm `v9k8`, replayed on the CPU.
//!
//! The arm (`kernels/pers_4e_k8.cuh`, `pers::p4e_e1_edits("v9k8")`) stores K
//! and V as group-affine int8: groups of 64 along head_dim, one binary16 scale
//! and one binary16 bias a group, the scheme of `llvq_llm::kvq` (`LLVQ_KV=q8`,
//! the `k8v8` rung of the KV ladder). This file states three things once, so
//! the kernel text can be checked against them bit for bit on the Mac:
//!
//! - the layout ([`blk_offset`], [`BLK`]): per KV head, one block of 1,088 bytes
//!   for each 8 positions aligned to absolute positions (4E's attention tile),
//!   the 8 rows of 128 codes first (position j at `128 j`), then the 8 rows of
//!   scale and bias (position j at `1,024 + 8 j`: scale 0, bias 0, scale 1,
//!   bias 1, binary16 little-endian). 136 bytes a position and head, 0.531 of
//!   the 256 of binary16, and every block a whole number of 16-byte chunks;
//! - the quantizer ([`quant_row`]): the operations of `kvq::affine_round_trip`
//!   at (64, 255) in its order, each an f32 round to nearest even, its round
//!   half up on `t >= 0` (candle's `round`, half away from zero), so the codes
//!   are kvq's codes (`llvq-llm` checks that against candle itself). The bytes
//!   of a position are a function of its binary16 row alone: a position the
//!   prefill wrote and the same position appended by a decode step hold the
//!   same bytes;
//! - the value the arm reads ([`dequant`]): `fma(scale, code, bias)` in f32,
//!   scale and bias widened from binary16, NOT rounded back to binary16. kvq's
//!   round trip, whose quality the ladder measured, rounds `code x scale +
//!   bias` (two f32 roundings) to binary16; the arm's value differs from it by
//!   that rounding, so the arm's quality is read through the arm.
//!
//! The attention over such a cache is `attn_ref.rs`'s arithmetic on these
//! values ([`Cache8`] implements `attn_ref::KvRows`).

use crate::attn_ref::{f2h, h2f, KvRows, HD};

/// Values a group: kvq's `GROUP`.
pub const GROUP: usize = 64;
/// Positions a block: 4E's attention tile (`PERS_TT`).
pub const BLK_POS: usize = 8;
/// Bytes of a block's codes: 8 rows of 128.
pub const CODES: usize = BLK_POS * HD;
/// Bytes of a position's scales and biases: 2 groups, 2 binary16 each.
pub const SB_BYTES: usize = 8;
/// Bytes of one block: the codes, then the scales and biases.
pub const BLK: usize = CODES + BLK_POS * SB_BYTES;
/// Bytes of one position of one head.
pub const POS_BYTES: usize = BLK / BLK_POS;

/// `1 / 255` as candle's `Tensor / 255.0` multiplies by it: `T::from_f64(1. / 255.)`.
pub fn inv255() -> f32 {
    (1.0f64 / 255.0) as f32
}

/// The quantized form of one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group8 {
    pub q: [u8; GROUP],
    pub s: u16,
    pub b: u16,
}

/// One group of binary16 values, kvq's operations in order:
/// `mn`, `mx` the exact minimum and maximum (a zero read as +0);
/// `s = f2h((mx - mn) x inv255)`; `b = f2h(mn)` (exact); a group whose
/// scale is zero codes every value 0 (kvq's ungraded branch, where
/// `round(x - mn)` is 0 since `x - mn < 2^-17`); otherwise
/// `t = (x - mn) / s`, `q = floor(t) + (t - floor(t) >= 0.5)`, at most 255.
/// Finite inputs only: the cache of a run that reached a NaN is void.
pub fn quant_group(x: &[u16]) -> Group8 {
    assert_eq!(x.len(), GROUP, "a group is {GROUP} values");
    let v: Vec<f32> = x.iter().map(|&h| h2f(h)).collect();
    let mut mn = v.iter().copied().fold(f32::INFINITY, f32::min);
    let mut mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if mn == 0.0 {
        mn = 0.0;
    }
    if mx == 0.0 {
        mx = 0.0;
    }
    let s16 = f2h((mx - mn) * inv255());
    let b16 = f2h(mn);
    let s = h2f(s16);
    let mut q = [0u8; GROUP];
    if s != 0.0 {
        for (qi, &xi) in q.iter_mut().zip(&v) {
            *qi = code(xi, mn, s);
        }
    }
    Group8 { q, s: s16, b: b16 }
}

/// The code of one value: `t = (x - mn) / s`, round half up, at most 255.
pub fn code(x: f32, mn: f32, s: f32) -> u8 {
    let t = (x - mn) / s;
    let fl = t.floor();
    let q = fl as u32 + u32::from(t - fl >= 0.5);
    q.min(255) as u8
}

/// The value the arm reads for a code: `fma(scale, code, bias)` in f32.
pub fn dequant(s: u16, b: u16, q: u8) -> f32 {
    h2f(s).mul_add(q as f32, h2f(b))
}

/// One row of 128 binary16 values: its codes (128 bytes) and its scales and
/// biases (`[s0, b0, s1, b1]`).
pub fn quant_row(x: &[u16]) -> ([u8; HD], [u16; 4]) {
    assert_eq!(x.len(), HD, "a row is {HD} values");
    let g0 = quant_group(&x[..GROUP]);
    let g1 = quant_group(&x[GROUP..]);
    let mut q = [0u8; HD];
    q[..GROUP].copy_from_slice(&g0.q);
    q[GROUP..].copy_from_slice(&g1.q);
    (q, [g0.s, g0.b, g1.s, g1.b])
}

/// Byte offset of position `p`'s block in head `h` of a cache of `cap`
/// positions (`cap` a multiple of 8).
pub fn blk_offset(h: usize, p: usize, cap: usize) -> usize {
    assert!(cap.is_multiple_of(BLK_POS), "an int8 cache holds whole blocks of {BLK_POS}");
    (h * (cap / BLK_POS) + p / BLK_POS) * BLK
}

/// Bytes of one int8 cache (K or V) of `n_kv` heads and `cap` positions.
pub fn cache_bytes(n_kv: usize, cap: usize) -> usize {
    n_kv * cap.div_ceil(BLK_POS) * BLK
}

/// Write one quantized row at position `p` of head `h`.
pub fn put_row(buf: &mut [u8], h: usize, p: usize, cap: usize, x: &[u16]) {
    let (q, sb) = quant_row(x);
    let o = blk_offset(h, p, cap);
    let j = p % BLK_POS;
    buf[o + j * HD..o + (j + 1) * HD].copy_from_slice(&q);
    for (k, w) in sb.iter().enumerate() {
        let at = o + CODES + j * SB_BYTES + 2 * k;
        buf[at..at + 2].copy_from_slice(&w.to_le_bytes());
    }
}

/// The values the arm reads at position `p` of head `h`.
pub fn row_values(buf: &[u8], h: usize, p: usize, cap: usize) -> [f32; HD] {
    let o = blk_offset(h, p, cap);
    let j = p % BLK_POS;
    let sb = |k: usize| {
        let at = o + CODES + j * SB_BYTES + 2 * k;
        u16::from_le_bytes([buf[at], buf[at + 1]])
    };
    std::array::from_fn(|d| {
        let g = d / GROUP;
        dequant(sb(2 * g), sb(2 * g + 1), buf[o + j * HD + d])
    })
}

/// One layer's int8 cache: K and V, `[n_kv][cap / 8][BLK]` bytes each.
#[derive(Clone, Debug)]
pub struct Cache8 {
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub n_kv: usize,
    pub cap: usize,
}

impl Cache8 {
    /// An empty cache, every byte zero (as the card's buffers are allocated).
    pub fn new(n_kv: usize, cap: usize) -> Self {
        Cache8 { k: vec![0; cache_bytes(n_kv, cap)], v: vec![0; cache_bytes(n_kv, cap)], n_kv, cap }
    }

    /// Positions `lo..hi` of a binary16 cache `[n_kv][cap16][128]`, quantized
    /// position by position (the fill the arm runs after a prefill).
    pub fn fill(&mut self, k16: &[u16], v16: &[u16], cap16: usize, lo: usize, hi: usize) {
        for h in 0..self.n_kv {
            for p in lo..hi {
                let o = (h * cap16 + p) * HD;
                put_row(&mut self.k, h, p, self.cap, &k16[o..o + HD]);
                put_row(&mut self.v, h, p, self.cap, &v16[o..o + HD]);
            }
        }
    }

    /// One position appended (a decode step): its K and V rows of 128.
    pub fn append(&mut self, h: usize, p: usize, k: &[u16], v: &[u16]) {
        put_row(&mut self.k, h, p, self.cap, k);
        put_row(&mut self.v, h, p, self.cap, v);
    }
}

impl KvRows for Cache8 {
    fn cap(&self) -> usize {
        self.cap
    }
    fn k_row(&self, h: usize, p: usize) -> [f32; HD] {
        row_values(&self.k, h, p, self.cap)
    }
    fn v_row(&self, h: usize, p: usize) -> [f32; HD] {
        row_values(&self.v, h, p, self.cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_ref;

    /// candle's divisor, pinned: a constant that drifted would move every scale.
    #[test]
    fn the_scale_multiplies_by_candles_reciprocal() {
        assert_eq!(inv255().to_bits(), 0x3b80_8081);
        assert_eq!(BLK, 1088);
        assert_eq!(BLK % 16, 0, "a block is whole 16-byte chunks");
        assert_eq!(POS_BYTES, 136);
    }

    /// The bytes a position takes at the sealed 4B's geometry, and their
    /// share of binary16's: 36 layers x 8 KV heads x (K, V).
    #[test]
    fn a_position_of_the_4b_is_78336_bytes() {
        let per_pos = 36 * 8 * 2 * POS_BYTES;
        assert_eq!(per_pos, 78_336);
        let f16 = 36 * 8 * 2 * 256;
        assert_eq!(f16, 147_456);
        assert!((per_pos as f64 / f16 as f64 - 0.53125).abs() < 1e-12);
        // at 8,192 positions: 641.7 MB against 1,208.0 MB
        assert_eq!(per_pos * 8192, 641_728_512);
        assert_eq!(f16 * 8192, 1_207_959_552);
    }

    fn row_of(seed: u64, amp: f32) -> Vec<u16> {
        attn_ref::fill(HD, seed, amp)
    }

    /// The codes reach 0 and 255, every dequantized value lies within one
    /// scale of its input, and the extremes come back within half a scale.
    #[test]
    fn a_row_quantizes_within_its_scale() {
        for seed in 1..200u64 {
            let x = row_of(seed, 1.0 + (seed % 7) as f32);
            let (q, sb) = quant_row(&x);
            for g in 0..2 {
                let qs = &q[g * GROUP..(g + 1) * GROUP];
                assert_eq!(*qs.iter().min().unwrap(), 0);
                assert_eq!(*qs.iter().max().unwrap(), 255, "seed {seed} group {g}");
                let s = h2f(sb[2 * g]);
                for d in 0..GROUP {
                    let want = h2f(x[g * GROUP + d]);
                    let got = dequant(sb[2 * g], sb[2 * g + 1], qs[d]);
                    assert!((got - want).abs() <= s, "seed {seed} d {d}: {got} against {want}, scale {s}");
                }
            }
        }
    }

    /// A constant group: scale 0, every code 0, the bias carries the value,
    /// which comes back exactly; the other group still varies.
    #[test]
    fn a_constant_group_reconstructs_exactly() {
        let mut x = row_of(9, 2.0);
        for v in x.iter_mut().take(GROUP) {
            *v = f2h(0.375);
        }
        let (q, sb) = quant_row(&x);
        assert_eq!(sb[0], 0, "scale 0");
        assert!(q[..GROUP].iter().all(|&c| c == 0));
        for &c in &q[..GROUP] {
            assert_eq!(dequant(sb[0], sb[1], c), 0.375);
        }
        assert!(q[GROUP..].iter().any(|&c| c > 100));
    }

    /// A zero minimum is stored as +0, whatever the sign the input carried.
    #[test]
    fn a_negative_zero_minimum_is_stored_as_plus_zero() {
        let mut x: Vec<u16> = (0..GROUP).map(|i| f2h(i as f32 / 64.0)).collect();
        x[0] = 0x8000;
        let g = quant_group(&x);
        assert_eq!(g.b, 0x0000);
        let mut y = x.clone();
        y[0] = 0x0000;
        assert_eq!(quant_group(&y), g, "-0 and +0 give the same bytes");
    }

    /// Round half up: a value exactly half a scale above a code takes the
    /// upper code, and truncation (`as u8`) would not.
    #[test]
    fn the_round_is_half_up() {
        // mn 0, mx 255 x 2^-8: s = 2^-8 exactly in binary16, t = (k + 0.5) on the planted values
        let mut x: Vec<u16> = vec![f2h(0.0); GROUP];
        x[1] = f2h(255.0 / 256.0);
        x[2] = f2h(2.5 / 256.0);
        x[3] = f2h(7.5 / 256.0);
        x[4] = f2h(3.25 / 256.0);
        let g = quant_group(&x);
        assert_eq!(h2f(g.s), 1.0 / 256.0);
        assert_eq!((g.q[1], g.q[2], g.q[3], g.q[4]), (255, 3, 8, 3));
    }

    /// The clamp is reached: a subnormal scale below range / 255 puts t
    /// past 255 at the maximum, which codes 255.
    #[test]
    fn a_subnormal_scale_clamps_at_255() {
        // range 2^-24 x 255 x 1.4: s32 = 1.4 x 2^-24 rounds to the subnormal 2^-24, t at mx = 357
        let mut x: Vec<u16> = vec![0; GROUP];
        let mx = (255.0f64 * 1.4 * (-24f64).exp2()) as f32;
        x[5] = f2h(mx);
        let g = quant_group(&x);
        assert_eq!(g.s, 0x0001, "the least subnormal");
        let t = (h2f(x[5]) - 0.0) / h2f(g.s);
        assert!(t > 255.5, "t {t}");
        assert_eq!(g.q[5], 255);
    }

    /// The scale is candle's product by `1 / 255`, not a division: two
    /// groups where the two give binary16 scales one ulp apart.
    #[test]
    fn the_scale_is_the_product_and_not_the_division() {
        for (mn, mx, s) in [(0x94c6u16, 0x3f46u16, 0x1f4fu16), (0xd558, 0x204f, 0x355e)] {
            let mut x = vec![0u16; GROUP];
            x[0] = mn;
            x[1] = mx;
            let g = quant_group(&x);
            assert_eq!(g.s, s);
            assert_eq!(f2h((h2f(mx) - h2f(mn)) / 255.0), s - 1, "the division gives the scale below");
        }
    }

    /// kvq's tensor operations transcribed on one group (min, max, the
    /// product by candle's reciprocal, the casts through f16, the division,
    /// `round`, the clamp): the same codes as `quant_group`.
    #[test]
    fn the_codes_are_kvqs_transcribed() {
        for seed in 300..700u64 {
            let x = attn_ref::fill(GROUP, seed, 0.01 + (seed % 13) as f32);
            let v: Vec<f32> = x.iter().map(|&h| h2f(h)).collect();
            let mn = v.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s = h2f(f2h((mx - mn) * inv255()));
            let mn16 = h2f(f2h(mn));
            let safe = if s > 0.0 { s } else { 1.0 };
            let want: Vec<u8> = v.iter().map(|&xi| ((xi - mn16) / safe).round().clamp(0.0, 255.0) as u8).collect();
            assert_eq!(quant_group(&x).q.to_vec(), want, "seed {seed}");
        }
    }

    /// A position's bytes are a function of its row alone: one fill of a
    /// prefix and the same positions appended one at a time give the same
    /// bytes, and a position's block neighbours do not move it.
    #[test]
    fn a_fill_and_appends_give_the_same_bytes() {
        let (n_kv, cap, len) = (3usize, 40usize, 37usize);
        let k16 = attn_ref::fill(n_kv * cap * HD, 11, 3.0);
        let v16 = attn_ref::fill(n_kv * cap * HD, 12, 1.0);
        let mut a = Cache8::new(n_kv, cap);
        a.fill(&k16, &v16, cap, 0, len);
        let mut b = Cache8::new(n_kv, cap);
        for p in (0..len).rev() {
            for h in 0..n_kv {
                let o = (h * cap + p) * HD;
                b.append(h, p, &k16[o..o + HD], &v16[o..o + HD]);
            }
        }
        assert_eq!(a.k, b.k);
        assert_eq!(a.v, b.v);
    }

    /// The attention through `Cache8` is attn_ref's arithmetic on the
    /// dequantized values: equal to the f16 path when every dequantized value
    /// is a binary16 (a cache built so), and the trait reads the layout.
    #[test]
    fn the_attention_reads_the_dequantized_values() {
        let (n_kv, groups, cap) = (2usize, 4usize, 72usize);
        let mut c8 = Cache8::new(n_kv, cap);
        // codes and scales chosen so fma(s, q, b) is exact in binary16: s 2^-6, b a multiple of 2^-6
        let mut k16 = vec![0u16; n_kv * cap * HD];
        let mut v16 = vec![0u16; n_kv * cap * HD];
        let mut st = 7u64;
        for h in 0..n_kv {
            for p in 0..cap {
                let o = blk_offset(h, p, cap);
                let j = p % BLK_POS;
                for (buf, f16s, base) in [(&mut c8.k, &mut k16, -1.5f32), (&mut c8.v, &mut v16, -0.75)] {
                    for g in 0..2 {
                        let s = f2h(1.0 / 64.0);
                        let b = f2h(base + g as f32 / 8.0);
                        let at = o + CODES + j * SB_BYTES + 4 * g;
                        buf[at..at + 2].copy_from_slice(&s.to_le_bytes());
                        buf[at + 2..at + 4].copy_from_slice(&b.to_le_bytes());
                        for d in 0..GROUP {
                            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            let q = (st >> 56) as u8;
                            buf[o + j * HD + g * GROUP + d] = q;
                            let val = dequant(s, b, q);
                            assert_eq!(h2f(f2h(val)), val, "exact in binary16");
                            f16s[(h * cap + p) * HD + g * GROUP + d] = f2h(val);
                        }
                    }
                }
            }
        }
        let kv = attn_ref::Kv { k: &k16, v: &v16, cap };
        let q = attn_ref::fill(n_kv * groups * HD, 5, 2.0);
        let c = attn_ref::score_scale(128);
        for l in [1usize, 8, 9, 64, 65, 72] {
            let a = attn_ref::attend(&q, &kv, n_kv, groups, l - 1, 1, c, attn_ref::UNIT_SERVED);
            let b = attn_ref::attend(&q, &c8, n_kv, groups, l - 1, 1, c, attn_ref::UNIT_SERVED);
            assert_eq!(a, b, "l {l}");
        }
    }
}
