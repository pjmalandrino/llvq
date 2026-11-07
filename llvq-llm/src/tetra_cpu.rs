//! The served matvecs on the CPU: Tetra, int4 g128 and the int4 g64 embedding,
//! each decoding its codes on the fly.
//!
//! No candle here, only slices, so the arithmetic is testable on its own and
//! [`crate::fused_cpu`] is the thin adapter around it.
//!
//! ## The Tetra decode, as tables
//!
//! `llvq_search::tetra` states the decode: a coordinate of section `k` is
//! `val(p + 2·bit_j(c_k), nibble_j(row_k))`. For one rank row and one parity
//! that is two candidate bytes per coordinate, one for each value of the code
//! bit. [`Tables`] holds both candidates as two `u64` of eight `i8` lanes, for
//! every row of the 16 KiB table and both parities (128 KiB), and the code bits
//! choose between them through a 256-entry byte mask. A block is then three
//! lookups, three masked merges, and 24 `i8` in trio order. The activation is
//! permuted into trio order once per call ([`permute_blocks`]), so the dot
//! product reads the decoded lanes as they come.
//!
//! The tables are built from `llvq_search::tetra`'s public objects (`rows`,
//! `prefixes`, `branches`, `suffixes`, `val`, `order`) and not from the GPU
//! kernels' byte permutes, so the decode here is a second route to the same
//! points. `decode_matches_the_reference` pins it against `Tetra::decode`.
//!
//! ## Arithmetic
//!
//! Per block, `d = <y, x>` in f32 over the 24 integer coordinates, and the
//! row accumulates `d · gscale[g] / ‖y‖`. The row scale multiplies the sum
//! and the exact tail is added last, which is `decode_matrix`'s formula
//! distributed over the product. The reference reconstructs the weights in
//! f64; this path rounds in f32, so the two agree to rounding, not to the bit.
//!
//! ## SIMD and threads
//!
//! NEON on aarch64 (always present there), AVX2 + FMA on x86_64 when the CPU
//! has them, a scalar loop otherwise. Rows are split across a rayon pool of
//! `LLVQ_THREADS` threads (all cores when unset).

use llvq_core::DIM;
use llvq_search::tetra::{Tetra, CLASS_ROWS, N0_MIXED, ROWS};
use rayon::prelude::*;
use std::sync::OnceLock;

/// Coordinates per section of a Tetra word.
const SECTION: usize = 8;

/// Activations one batched call carries at most.
pub const MAX_BATCH: usize = 16;

/// The decode tables of every Tetra matrix, built once.
pub struct Tables {
    /// `lut[p · ROWS + row] = [candidates with code bit 0, with code bit 1]`,
    /// eight `i8` lanes each, lane `j` the coordinate `j` of the section.
    lut: Vec<[u64; 2]>,
    /// `(s8, b1, b2, b3)` → `c1 | c2 << 8 | c3 << 16`, the three code bytes.
    paths: Vec<u32>,
    /// Byte `j` set to `0xff` where bit `j` of the index is set.
    mask: [u64; 256],
    /// `1/√(16 m)` with entry 0 zero: the origin reconstructs to zero.
    invnorm: [f32; 32],
    /// `order[t]`: the natural coordinate at trio position `t`.
    order: [usize; DIM],
}

impl Default for Tables {
    fn default() -> Self {
        Self::new(&Tetra::new())
    }
}

impl Tables {
    pub fn new(t: &Tetra) -> Self {
        let rows = t.rows();
        let mut lut = vec![[0u64; 2]; 2 * ROWS];
        for p in 0..2u32 {
            for (ri, &row) in rows.iter().enumerate() {
                let mut e = [0u64; 2];
                for (b, out) in e.iter_mut().enumerate() {
                    for j in 0..SECTION {
                        let v = llvq_search::tetra::val(p + 2 * b as u32, (row >> (4 * j)) & 15);
                        let v = i8::try_from(v).expect("a Tetra coordinate fits an i8");
                        *out |= u64::from(v as u8) << (8 * j);
                    }
                }
                lut[p as usize * ROWS + ri] = e;
            }
        }
        let mut paths = vec![0u32; 1 << 12];
        for s8 in 0..64usize {
            for b1 in 0..2usize {
                for b2 in 0..16usize {
                    for b3 in 0..2usize {
                        let c1 = t.prefixes()[s8][b1] as u32;
                        let (c2, s16) = t.branches()[s8][b2];
                        let c3 = t.suffixes()[s16 as usize][b3] as u32;
                        paths[(s8 << 6) | (b1 << 5) | (b2 << 1) | b3] = c1 | (c2 as u32) << 8 | c3 << 16;
                    }
                }
            }
        }
        let mut mask = [0u64; 256];
        for (c, m) in mask.iter_mut().enumerate() {
            for j in 0..8 {
                if (c >> j) & 1 == 1 {
                    *m |= 0xff << (8 * j);
                }
            }
        }
        let mut invnorm = [0f32; 32];
        for (m, e) in invnorm.iter_mut().enumerate().skip(1) {
            *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
        }
        let mut order = [0usize; DIM];
        for (o, &n) in order.iter_mut().zip(t.order()) {
            *o = n as usize;
        }
        Self { lut, paths, mask, invnorm, order }
    }

    /// The three sections of a word, eight `i8` lanes each, and its gain bit.
    #[inline(always)]
    pub fn sections(&self, w: u64) -> ([u64; 3], usize) {
        let p = (w & 1) as usize;
        let r = ((w >> 1) & 1) as usize;
        let path_at = ((w >> 2) & 63) << 6 | ((w >> 8) & 1) << 5 | ((w >> 20) & 15) << 1 | ((w >> 35) & 1);
        let path = self.paths[path_at as usize];
        let i1 = ((w >> 9) & 0x7ff) as usize;
        let i2 = ((w >> 24) & 0x7ff) as usize;
        let i3 = ((w >> 36) & 0x7ff) as usize;
        let delta = usize::from(i2 >= N0_MIXED);
        let r1 = CLASS_ROWS * r + i1;
        let r2 = i2 + delta * (CLASS_ROWS - N0_MIXED);
        let r3 = CLASS_ROWS * (p ^ r ^ delta) + i3;
        let lut = &self.lut[p * ROWS..(p + 1) * ROWS];
        let pick = |ri: usize, c: u32| {
            let e = lut[ri];
            let k = self.mask[(c & 0xff) as usize];
            (e[0] & !k) | (e[1] & k)
        };
        ([pick(r1, path), pick(r2, path >> 8), pick(r3, path >> 16)], ((w >> 47) & 1) as usize)
    }

    /// The point of a word in trio order, and its gain bit.
    pub fn decode_trio(&self, w: u64) -> ([i32; DIM], usize) {
        let (s, g) = self.sections(w);
        let mut y = [0i32; DIM];
        for (k, sec) in s.iter().enumerate() {
            for j in 0..SECTION {
                y[SECTION * k + j] = i32::from((sec >> (8 * j)) as u8 as i8);
            }
        }
        (y, g)
    }

    /// `gscale[g] / ‖y‖` from `n2 = ‖y‖²`, through the kernels' table.
    #[inline(always)]
    fn scale(&self, gscale: &[f32; 2], g: usize, n2: u32) -> f32 {
        gscale[g] * self.invnorm[((n2 >> 4) & 31) as usize]
    }

    pub fn order(&self) -> &[usize; DIM] {
        &self.order
    }
}

/// `x` with each full block of 24 put in trio order, the tail left in place.
///
/// The decoded lanes come out in trio order, so permuting the activation once
/// per call is what lets the dot product read them without a shuffle.
pub fn permute_blocks(t: &Tables, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let nblocks = x.len() / DIM;
    for j in 0..nblocks {
        let (src, dst) = (&x[j * DIM..(j + 1) * DIM], &mut out[j * DIM..(j + 1) * DIM]);
        for (d, &o) in dst.iter_mut().zip(t.order.iter()) {
            *d = src[o];
        }
    }
    out[nblocks * DIM..].copy_from_slice(&x[nblocks * DIM..]);
}

/// One Tetra matrix, held as the served `tetra48` stream.
pub struct TetraMatrix {
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    /// u32 per row of `words`.
    pub stride: usize,
    /// Row-strided little-endian words, as `llvq_artifact::tetra48` writes them.
    pub words: Vec<u32>,
    pub gscale: [f32; 2],
    pub rscale: Vec<f32>,
    /// `d_out × tail_w`, binary16 bits.
    pub tail: Vec<u16>,
}

impl TetraMatrix {
    /// Refuses a stream whose shape does not cover what the loops read.
    pub fn check(&self) -> Result<(), String> {
        if self.nblocks * DIM + self.tail_w != self.d_in {
            return Err(format!("{} blocks and a tail of {} for d_in {}", self.nblocks, self.tail_w, self.d_in));
        }
        // The last block reads u32 `(3(n-1)) >> 1` and the one after it.
        if self.nblocks > 0 && ((3 * (self.nblocks - 1)) >> 1) + 2 > self.stride {
            return Err(format!("row stride {} too short for {} blocks", self.stride, self.nblocks));
        }
        // The loader's stream may carry a pad word past the last row.
        if self.words.len() < self.d_out * self.stride {
            return Err(format!("{} words for {} rows of {}", self.words.len(), self.d_out, self.stride));
        }
        if self.rscale.len() != self.d_out || self.tail.len() != self.d_out * self.tail_w {
            return Err(format!("{} row scales and {} tail weights for {} rows", self.rscale.len(), self.tail.len(), self.d_out));
        }
        Ok(())
    }

    /// Bytes this matrix holds in memory.
    pub fn bytes(&self) -> u64 {
        (self.words.len() * 4 + self.rscale.len() * 4 + self.tail.len() * 2) as u64
    }

    fn row(&self, r: usize) -> &[u32] {
        &self.words[r * self.stride..(r + 1) * self.stride]
    }

    /// The tail's contribution to row `r` and the row scale applied.
    #[inline(always)]
    fn finish(&self, r: usize, acc: f32, x: &[f32]) -> f32 {
        let mut s = acc * self.rscale[r];
        let tail = &self.tail[r * self.tail_w..(r + 1) * self.tail_w];
        for (&h, &v) in tail.iter().zip(&x[self.nblocks * DIM..]) {
            s = half::f16::from_bits(h).to_f32().mul_add(v, s);
        }
        s
    }
}

/// The 48-bit word of block `j` of a row, read as `f1r_load` reads it.
#[inline(always)]
fn word_at(row: &[u32], j: usize) -> u64 {
    let w = (3 * j) >> 1;
    let pair = u64::from(row[w]) | u64::from(row[w + 1]) << 32;
    (pair >> (16 * (j & 1))) & 0xffff_ffff_ffff
}

/// Which SIMD path this CPU takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Isa {
    /// The best path this CPU runs.
    pub fn detect() -> Self {
        static ISA: OnceLock<Isa> = OnceLock::new();
        *ISA.get_or_init(|| {
            #[cfg(target_arch = "aarch64")]
            {
                Isa::Neon
            }
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                    return Isa::Avx2;
                }
                Isa::Scalar
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                Isa::Scalar
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => "avx2+fma",
        }
    }
}

/// Every path this CPU can run, the best last: what the tests sweep.
pub fn available_isas() -> Vec<Isa> {
    let mut v = vec![Isa::Scalar];
    let best = Isa::detect();
    if best != Isa::Scalar {
        v.push(best);
    }
    v
}

// ---------------------------------------------------------------------------
// One row against one activation.
// ---------------------------------------------------------------------------

fn tetra_row_scalar(t: &Tables, row: &[u32], nblocks: usize, gscale: &[f32; 2], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for j in 0..nblocks {
        let (s, g) = t.sections(word_at(row, j));
        let xb = &x[j * DIM..(j + 1) * DIM];
        let (mut d, mut n2) = (0f32, 0u32);
        for (k, sec) in s.iter().enumerate() {
            for l in 0..SECTION {
                let v = i32::from((sec >> (8 * l)) as u8 as i8);
                n2 += (v * v) as u32;
                d = (v as f32).mul_add(xb[SECTION * k + l], d);
            }
        }
        acc = d.mul_add(t.scale(gscale, g, n2), acc);
    }
    acc
}

fn tetra_rows_scalar(
    t: &Tables,
    row: &[u32],
    nblocks: usize,
    gscale: &[f32; 2],
    xs: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    let n = out.len();
    let mut acc = [0f32; MAX_BATCH];
    for j in 0..nblocks {
        let (y, g) = t.decode_trio(word_at(row, j));
        let n2: i32 = y.iter().map(|v| v * v).sum();
        let sc = t.scale(gscale, g, n2 as u32);
        for (r, a) in acc.iter_mut().enumerate().take(n) {
            let xb = &xs[r * stride + j * DIM..r * stride + (j + 1) * DIM];
            let mut d = 0f32;
            for (v, &xv) in y.iter().zip(xb) {
                d = (*v as f32).mul_add(xv, d);
            }
            *a = d.mul_add(sc, *a);
        }
    }
    out.copy_from_slice(&acc[..n]);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{word_at, Tables, DIM, MAX_BATCH};
    use std::arch::aarch64::*;

    /// The 24 lanes of a block as three `int16x8`, and `‖y‖²`.
    #[inline(always)]
    unsafe fn widen(s: [u64; 3]) -> (int16x8_t, int16x8_t, int16x8_t, u32) {
        let lo = vcombine_s8(vcreate_s8(s[0]), vcreate_s8(s[1]));
        let a0 = vmovl_s8(vget_low_s8(lo));
        let a1 = vmovl_high_s8(lo);
        let a2 = vmovl_s8(vcreate_s8(s[2]));
        // Each lane holds three squares of at most 11², and the whole sum is
        // at most 416: both fit an i16.
        let sq = vmlaq_s16(vmlaq_s16(vmulq_s16(a0, a0), a1, a1), a2, a2);
        (a0, a1, a2, vaddvq_s16(sq) as u16 as u32)
    }

    #[inline(always)]
    unsafe fn lo(a: int16x8_t) -> float32x4_t {
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(a)))
    }

    #[inline(always)]
    unsafe fn hi(a: int16x8_t) -> float32x4_t {
        vcvtq_f32_s32(vmovl_high_s16(a))
    }

    /// `<y, x>` over one block, as four lanes still to be summed.
    #[inline(always)]
    unsafe fn dot(f: &[float32x4_t; 6], x: *const f32) -> float32x4_t {
        let mut d = vmulq_f32(f[0], vld1q_f32(x));
        d = vfmaq_f32(d, f[1], vld1q_f32(x.add(4)));
        d = vfmaq_f32(d, f[2], vld1q_f32(x.add(8)));
        d = vfmaq_f32(d, f[3], vld1q_f32(x.add(12)));
        d = vfmaq_f32(d, f[4], vld1q_f32(x.add(16)));
        vfmaq_f32(d, f[5], vld1q_f32(x.add(20)))
    }

    pub fn row(t: &Tables, row: &[u32], nblocks: usize, gscale: &[f32; 2], x: &[f32]) -> f32 {
        assert!(x.len() >= nblocks * DIM);
        // SAFETY: NEON is part of the aarch64 baseline, so the intrinsics are
        // always available; every load reads 24 floats of `x` at block `j`,
        // inside the length asserted above.
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for j in 0..nblocks {
                let (s, g) = t.sections(word_at(row, j));
                let (a0, a1, a2, n2) = widen(s);
                let f = [lo(a0), hi(a0), lo(a1), hi(a1), lo(a2), hi(a2)];
                let d = dot(&f, x.as_ptr().add(j * DIM));
                acc = vfmaq_n_f32(acc, d, t.scale(gscale, g, n2));
            }
            vaddvq_f32(acc)
        }
    }

    pub fn rows(
        t: &Tables,
        row: &[u32],
        nblocks: usize,
        gscale: &[f32; 2],
        xs: &[f32],
        stride: usize,
        out: &mut [f32],
    ) {
        let n = out.len();
        assert!(n <= MAX_BATCH && n > 0);
        assert!(stride >= nblocks * DIM && xs.len() >= (n - 1) * stride + nblocks * DIM);
        // SAFETY: as in `row`; activation `r` is read at `r * stride + 24 j`,
        // inside the length asserted above.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); MAX_BATCH];
            for j in 0..nblocks {
                let (s, g) = t.sections(word_at(row, j));
                let (a0, a1, a2, n2) = widen(s);
                let f = [lo(a0), hi(a0), lo(a1), hi(a1), lo(a2), hi(a2)];
                let sc = t.scale(gscale, g, n2);
                for (r, a) in acc.iter_mut().enumerate().take(n) {
                    let d = dot(&f, xs.as_ptr().add(r * stride + j * DIM));
                    *a = vfmaq_n_f32(*a, d, sc);
                }
            }
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = vaddvq_f32(*a);
            }
        }
    }

    /// `Σ q·x` over one int4 group, `x` deinterleaved: `xe` the even
    /// elements (low nibbles), `xo` the odd ones (high nibbles).
    pub fn q4_group(bytes: &[u8], xe: &[f32], xo: &[f32]) -> f32 {
        assert!(bytes.len().is_multiple_of(16) && xe.len() >= bytes.len() && xo.len() >= bytes.len());
        // SAFETY: NEON baseline; 16 bytes and 16 floats of each half per step,
        // inside the lengths asserted above.
        unsafe {
            let m = vdupq_n_u8(0x0f);
            let mut acc = vdupq_n_f32(0.0);
            for c in (0..bytes.len()).step_by(16) {
                let v = vld1q_u8(bytes.as_ptr().add(c));
                let l = vandq_u8(v, m);
                let h = vshrq_n_u8(v, 4);
                for (half, xp) in [(l, xe.as_ptr().add(c)), (h, xo.as_ptr().add(c))] {
                    let w0 = vmovl_u8(vget_low_u8(half));
                    let w1 = vmovl_high_u8(half);
                    acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0))), vld1q_f32(xp));
                    acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_high_u16(w0)), vld1q_f32(xp.add(4)));
                    acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1))), vld1q_f32(xp.add(8)));
                    acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_high_u16(w1)), vld1q_f32(xp.add(12)));
                }
            }
            vaddvq_f32(acc)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{word_at, Tables, DIM, MAX_BATCH};
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn lanes(s: u64) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(s as i64)))
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }

    /// The three lane vectors of a block and `‖y‖²`, exact in f32.
    #[inline(always)]
    unsafe fn block(s: [u64; 3]) -> ([__m256; 3], u32) {
        let f = [lanes(s[0]), lanes(s[1]), lanes(s[2])];
        let sq = _mm256_fmadd_ps(f[2], f[2], _mm256_fmadd_ps(f[1], f[1], _mm256_mul_ps(f[0], f[0])));
        (f, hsum(sq) as u32)
    }

    #[inline(always)]
    unsafe fn dot(f: &[__m256; 3], x: *const f32) -> __m256 {
        let d = _mm256_mul_ps(f[0], _mm256_loadu_ps(x));
        let d = _mm256_fmadd_ps(f[1], _mm256_loadu_ps(x.add(8)), d);
        _mm256_fmadd_ps(f[2], _mm256_loadu_ps(x.add(16)), d)
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn row(t: &Tables, row: &[u32], nblocks: usize, gscale: &[f32; 2], x: &[f32]) -> f32 {
        assert!(x.len() >= nblocks * DIM);
        let mut acc = _mm256_setzero_ps();
        for j in 0..nblocks {
            let (s, g) = t.sections(word_at(row, j));
            let (f, n2) = block(s);
            let d = dot(&f, x.as_ptr().add(j * DIM));
            acc = _mm256_fmadd_ps(d, _mm256_set1_ps(t.scale(gscale, g, n2)), acc);
        }
        hsum(acc)
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rows(
        t: &Tables,
        row: &[u32],
        nblocks: usize,
        gscale: &[f32; 2],
        xs: &[f32],
        stride: usize,
        out: &mut [f32],
    ) {
        let n = out.len();
        assert!(n <= MAX_BATCH && n > 0);
        assert!(stride >= nblocks * DIM && xs.len() >= (n - 1) * stride + nblocks * DIM);
        let mut acc = [_mm256_setzero_ps(); MAX_BATCH];
        for j in 0..nblocks {
            let (s, g) = t.sections(word_at(row, j));
            let (f, n2) = block(s);
            let sc = _mm256_set1_ps(t.scale(gscale, g, n2));
            for (r, a) in acc.iter_mut().enumerate().take(n) {
                let d = dot(&f, xs.as_ptr().add(r * stride + j * DIM));
                *a = _mm256_fmadd_ps(d, sc, *a);
            }
        }
        for (o, a) in out.iter_mut().zip(acc.iter()) {
            *o = hsum(*a);
        }
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q4_group(bytes: &[u8], xe: &[f32], xo: &[f32]) -> f32 {
        assert!(bytes.len().is_multiple_of(16) && xe.len() >= bytes.len() && xo.len() >= bytes.len());
        let m = _mm_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        for c in (0..bytes.len()).step_by(16) {
            let v = _mm_loadu_si128(bytes.as_ptr().add(c) as *const __m128i);
            let l = _mm_and_si128(v, m);
            let h = _mm_and_si128(_mm_srli_epi16(v, 4), m);
            for (half, xp) in [(l, xe.as_ptr().add(c)), (h, xo.as_ptr().add(c))] {
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(half));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(half, 8)));
                acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xp), acc);
                acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp.add(8)), acc);
            }
        }
        hsum(acc)
    }
}

fn tetra_row(isa: Isa, t: &Tables, row: &[u32], nblocks: usize, gscale: &[f32; 2], x: &[f32]) -> f32 {
    match isa {
        Isa::Scalar => tetra_row_scalar(t, row, nblocks, gscale, x),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::row(t, row, nblocks, gscale, x),
        // SAFETY: `Isa::Avx2` is only produced where the CPU reports both
        // features (`Isa::detect`).
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::row(t, row, nblocks, gscale, x) },
    }
}

#[allow(clippy::too_many_arguments)]
fn tetra_rows(
    isa: Isa,
    t: &Tables,
    row: &[u32],
    nblocks: usize,
    gscale: &[f32; 2],
    xs: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    match isa {
        Isa::Scalar => tetra_rows_scalar(t, row, nblocks, gscale, xs, stride, out),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::rows(t, row, nblocks, gscale, xs, stride, out),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::rows(t, row, nblocks, gscale, xs, stride, out) },
    }
}

fn q4_group(isa: Isa, bytes: &[u8], xe: &[f32], xo: &[f32]) -> f32 {
    if !bytes.len().is_multiple_of(16) {
        return q4_group_scalar(bytes, xe, xo);
    }
    match isa {
        Isa::Scalar => q4_group_scalar(bytes, xe, xo),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::q4_group(bytes, xe, xo),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::q4_group(bytes, xe, xo) },
    }
}

fn q4_group_scalar(bytes: &[u8], xe: &[f32], xo: &[f32]) -> f32 {
    let mut acc = 0f32;
    for ((&b, &e), &o) in bytes.iter().zip(xe).zip(xo) {
        acc = ((b & 15) as f32).mul_add(e, acc);
        acc = ((b >> 4) as f32).mul_add(o, acc);
    }
    acc
}

// ---------------------------------------------------------------------------
// The pool and the whole-matrix calls.
// ---------------------------------------------------------------------------

/// Threads the CPU backend uses: `LLVQ_THREADS`, or every core.
pub fn threads() -> Result<usize, String> {
    match std::env::var("LLVQ_THREADS") {
        Ok(s) => {
            let n: usize = s.trim().parse().map_err(|e| format!("LLVQ_THREADS={s:?} is not a number: {e}"))?;
            if n == 0 {
                return Err(format!("LLVQ_THREADS={s:?}: at least one is needed"));
            }
            Ok(n)
        }
        Err(_) => Ok(std::thread::available_parallelism().map_or(1, |n| n.get())),
    }
}

/// The pool every matvec of this module runs on, built at first use.
pub fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = threads().unwrap_or_else(|e| panic!("{e}"));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("llvq-cpu-{i}"))
            .build()
            .expect("the CPU pool starts")
    })
}

/// Rows a task takes: enough tasks to balance, few enough to stay cheap.
fn chunk_rows(d_out: usize) -> usize {
    let tasks = pool().current_num_threads() * 8;
    d_out.div_ceil(tasks).clamp(4, 512)
}

/// `y = W x` for one activation already in trio order ([`permute_blocks`]).
pub fn tetra_matvec(isa: Isa, t: &Tables, m: &TetraMatrix, xp: &[f32], y: &mut [f32]) {
    assert_eq!(xp.len(), m.d_in);
    assert_eq!(y.len(), m.d_out);
    let c = chunk_rows(m.d_out);
    pool().install(|| {
        y.par_chunks_mut(c).enumerate().for_each(|(ci, ys)| {
            for (k, yo) in ys.iter_mut().enumerate() {
                let r = ci * c + k;
                let acc = tetra_row(isa, t, m.row(r), m.nblocks, &m.gscale, xp);
                *yo = m.finish(r, acc, xp);
            }
        })
    });
}

/// `Y = W X` for `n` activations, `[n, d_in]` row-major and in trio order,
/// into `[n, d_out]`. Each block is decoded once for the whole batch.
pub fn tetra_matvec_rows(isa: Isa, t: &Tables, m: &TetraMatrix, xs: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(xs.len(), n * m.d_in);
    assert_eq!(y.len(), n * m.d_out);
    let c = chunk_rows(m.d_out);
    let mut yt = vec![0f32; n * m.d_out];
    pool().install(|| {
        yt.par_chunks_mut(c * n).enumerate().for_each(|(ci, ys)| {
            for (k, out) in ys.chunks_mut(n).enumerate() {
                let r = ci * c + k;
                let mut start = 0;
                while start < n {
                    let len = (n - start).min(MAX_BATCH);
                    let o = &mut out[start..start + len];
                    tetra_rows(isa, t, m.row(r), m.nblocks, &m.gscale, &xs[start * m.d_in..], m.d_in, o);
                    for (i, v) in o.iter_mut().enumerate() {
                        let x = &xs[(start + i) * m.d_in..(start + i + 1) * m.d_in];
                        *v = m.finish(r, *v, x);
                    }
                    start += len;
                }
            }
        })
    });
    // `yt` is `[d_out, n]`; the caller wants `[n, d_out]`.
    for r in 0..m.d_out {
        for i in 0..n {
            y[i * m.d_out + r] = yt[r * n + i];
        }
    }
}

/// One group-affine int4 matrix: `w = scale · q + bias`, low nibble first,
/// one binary16 scale and bias per `group` along each row.
pub struct Q4Matrix {
    pub d_out: usize,
    pub d_in: usize,
    pub group: usize,
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
}

impl Q4Matrix {
    pub fn check(&self) -> Result<(), String> {
        if self.group == 0 || !self.group.is_multiple_of(2) || !self.d_in.is_multiple_of(self.group) {
            return Err(format!("group {} for d_in {}", self.group, self.d_in));
        }
        let groups = self.d_out * (self.d_in / self.group);
        if self.packed.len() != self.d_out * self.d_in / 2 || self.scales.len() != groups || self.biases.len() != groups {
            return Err(format!(
                "{} bytes, {} scales, {} biases for {} x {} in groups of {}",
                self.packed.len(),
                self.scales.len(),
                self.biases.len(),
                self.d_out,
                self.d_in,
                self.group
            ));
        }
        Ok(())
    }

    pub fn bytes(&self) -> u64 {
        (self.packed.len() + 2 * self.scales.len() + 2 * self.biases.len()) as u64
    }

    /// Row `r` decoded, `scale · q + bias` in f32 as `RawTensor::to_f32` does.
    pub fn decode_row(&self, r: usize, out: &mut [f32]) {
        let gpr = self.d_in / self.group;
        for (c, o) in out.iter_mut().enumerate().take(self.d_in) {
            let i = r * self.d_in + c;
            let q = (self.packed[i / 2] >> (4 * (i % 2))) & 0xf;
            let g = r * gpr + c / self.group;
            let (s, b) = (half::f16::from_bits(self.scales[g]).to_f32(), half::f16::from_bits(self.biases[g]).to_f32());
            *o = s * q as f32 + b;
        }
    }
}

/// An activation laid out for the int4 rows: even and odd elements apart, and
/// the sum of each group, which multiplies the bias.
pub struct Q4Act {
    xe: Vec<f32>,
    xo: Vec<f32>,
    gsum: Vec<f32>,
}

impl Q4Act {
    pub fn new(x: &[f32], group: usize) -> Self {
        let xe = x.iter().step_by(2).copied().collect();
        let xo = x.iter().skip(1).step_by(2).copied().collect();
        let gsum = x.chunks(group).map(|g| g.iter().sum()).collect();
        Self { xe, xo, gsum }
    }
}

fn q4_row(isa: Isa, m: &Q4Matrix, r: usize, a: &Q4Act) -> f32 {
    let (half_row, half_g, gpr) = (m.d_in / 2, m.group / 2, m.d_in / m.group);
    let bytes = &m.packed[r * half_row..(r + 1) * half_row];
    let mut acc = 0f32;
    for g in 0..gpr {
        let at = g * half_g;
        let d = q4_group(isa, &bytes[at..at + half_g], &a.xe[at..at + half_g], &a.xo[at..at + half_g]);
        let s = half::f16::from_bits(m.scales[r * gpr + g]).to_f32();
        let b = half::f16::from_bits(m.biases[r * gpr + g]).to_f32();
        acc = s.mul_add(d, b.mul_add(a.gsum[g], acc));
    }
    acc
}

/// `y = W x` for one int4 matrix and one activation in the natural basis.
pub fn q4_matvec(isa: Isa, m: &Q4Matrix, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), m.d_in);
    assert_eq!(y.len(), m.d_out);
    let a = Q4Act::new(x, m.group);
    let c = chunk_rows(m.d_out);
    pool().install(|| {
        y.par_chunks_mut(c).enumerate().for_each(|(ci, ys)| {
            for (k, yo) in ys.iter_mut().enumerate() {
                *yo = q4_row(isa, m, ci * c + k, &a);
            }
        })
    });
}

/// Activations one pass of [`q4_matvec_rows`] holds: bounds its scratch to
/// `64 · d_out` floats when a 4,096-token window meets the 151,936-row head.
const Q4_PASS: usize = 64;

/// `Y = W X` for `n` activations `[n, d_in]` into `[n, d_out]`.
pub fn q4_matvec_rows(isa: Isa, m: &Q4Matrix, xs: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(xs.len(), n * m.d_in);
    assert_eq!(y.len(), n * m.d_out);
    let c = chunk_rows(m.d_out);
    for i0 in (0..n).step_by(Q4_PASS) {
        let k = Q4_PASS.min(n - i0);
        let acts: Vec<Q4Act> =
            xs[i0 * m.d_in..(i0 + k) * m.d_in].chunks(m.d_in).map(|x| Q4Act::new(x, m.group)).collect();
        let mut yt = vec![0f32; k * m.d_out];
        pool().install(|| {
            yt.par_chunks_mut(c * k).enumerate().for_each(|(ci, ys)| {
                for (r, out) in ys.chunks_mut(k).enumerate() {
                    for (o, a) in out.iter_mut().zip(&acts) {
                        *o = q4_row(isa, m, ci * c + r, a);
                    }
                }
            })
        });
        for r in 0..m.d_out {
            for i in 0..k {
                y[(i0 + i) * m.d_out + r] = yt[r * k + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_core::SplitMix64;

    #[test]
    fn decode_matches_the_reference() {
        let tetra = Tetra::new();
        let t = Tables::new(&tetra);
        let mut rng = SplitMix64::new(0x0c90_0001);
        let mut words: Vec<u64> = (0..200_000).map(|_| rng.next() & 0xffff_ffff_ffff).collect();
        // The origin, both gains, and every boundary of the mixed split.
        words.extend([0, 1 << 47, (N0_MIXED as u64 - 1) << 24, (N0_MIXED as u64) << 24, 0x7ffu64 << 24]);
        for w in words {
            let (trio, g) = t.decode_trio(w);
            assert_eq!(trio, tetra.decode_trio_order(w), "word {w:#014x}");
            assert_eq!(g as u64, w >> 47, "gain of {w:#014x}");
            let natural = tetra.decode(w);
            for (k, &n) in t.order().iter().enumerate() {
                assert_eq!(trio[k], natural[n], "word {w:#014x}, trio position {k}");
            }
        }
    }

    #[test]
    fn word_at_reads_the_served_stream() {
        let mut rng = SplitMix64::new(7);
        for nblocks in [1usize, 2, 3, 106, 107] {
            let idx: Vec<u64> = (0..2 * nblocks).map(|_| rng.next() & ((1 << 47) - 1)).collect();
            let gains: Vec<u32> = (0..2 * nblocks).map(|_| (rng.next() & 1) as u32).collect();
            let s = llvq_artifact::tetra48::transcode_tetra48(&idx, &gains, 2, nblocks).unwrap();
            let words: Vec<u32> = s.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            for r in 0..2 {
                let row = &words[r * s.stride_u32..(r + 1) * s.stride_u32];
                for j in 0..nblocks {
                    assert_eq!(word_at(row, j), s.word(r, j), "nblocks {nblocks} row {r} block {j}");
                }
            }
        }
    }
}
