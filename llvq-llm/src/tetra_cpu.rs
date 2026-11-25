//! The served matvecs on the CPU: Tetra, int4 g128 and the int4 g64 embedding,
//! each decoding its codes on the fly.
//!
//! Beside them, [`bf16_matvec`] reads a BF16 matrix as a checkpoint stores
//! it: the attention products of Inkling 2 Mini on the CPU path.
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
//! NEON on aarch64 (always present there), AVX2 + FMA + F16C on x86_64 when the CPU
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

/// Entries of the path table: `(s8, b1, b2, b3)` packed in 12 bits.
const PATHS: usize = 1 << 12;

// The row index is masked to `ROWS - 1` in `sections`, which is a no-op only
// while every index it builds stays below `ROWS`: `CLASS_ROWS · r + i` with
// `i < 2^11`, and the mixed split's `i2 + (CLASS_ROWS − N0_MIXED)`.
const _: () = assert!(ROWS.is_power_of_two() && ROWS == 2 * CLASS_ROWS && CLASS_ROWS == 1 << 11);
const _: () = assert!((1 << 11) - 1 + CLASS_ROWS - N0_MIXED < ROWS);

/// The candidate table, aligned to a cache line so that the 16-byte entry a
/// SIMD path loads whole never straddles two lines.
#[repr(C, align(64))]
struct Lut([[[u64; 2]; ROWS]; 2]);

impl Lut {
    /// Both parities as one table, entry `p · ROWS + row`: what `avx2`
    /// addresses with one byte offset.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn flat(&self) -> &[[u64; 2]; 2 * ROWS] {
        // SAFETY: `[[T; ROWS]; 2]` and `[T; 2 · ROWS]` have the same size,
        // alignment and element order.
        unsafe { &*self.0.as_ptr().cast::<[[u64; 2]; 2 * ROWS]>() }
    }
}

impl std::ops::Index<usize> for Lut {
    type Output = [[u64; 2]; ROWS];
    #[inline(always)]
    fn index(&self, p: usize) -> &Self::Output {
        &self.0[p]
    }
}

/// [`Tables::sel`], aligned like [`Lut`].
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
struct Sel([[u8; 16]; 256]);

/// The decode tables of every Tetra matrix, built once.
///
/// Fixed-size arrays, so the indices the decode builds (12 bits for a path,
/// masked to `ROWS - 1` for a row) need no bounds check in the inner loop.
pub struct Tables {
    /// `lut[p][row] = [candidates with code bit 0, with code bit 1]`, eight
    /// `i8` lanes each, lane `j` the coordinate `j` of the section.
    lut: Box<Lut>,
    /// `(s8, b1, b2, b3)` → `c1 | c2 << 8 | c3 << 16`, the three code bytes.
    paths: Box<[u32; PATHS]>,
    /// Byte `j` set to `0xff` where bit `j` of the index is set.
    mask: [u64; 256],
    /// The same choice as a byte shuffle of a `lut` entry (16 bytes, the
    /// bit-0 candidates then the bit-1 ones): byte `j < 8` is `j + 8·bit_j`,
    /// bytes 8 to 15 are `0x80`, which zeroes them. What `avx2` selects with.
    #[cfg(target_arch = "x86_64")]
    sel: Sel,
    /// `1/√(16 m)` with entry 0 zero: the origin reconstructs to zero.
    invnorm: [f32; 32],
    /// `nibval[p][n] = val(p + 2·(n >> 3), n & 7)` as an `i8`: the coordinate
    /// a nibble of [`TetraNib`] stands for in a block of parity `p`.
    nibval: [[u8; 16]; 2],
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
        assert_eq!(rows.len(), ROWS, "a Tetra table of {ROWS} rows");
        let mut lut = Box::new(Lut([[[0u64; 2]; ROWS]; 2]));
        for p in 0..2u32 {
            for (ri, &row) in rows.iter().enumerate() {
                let mut e = [0u64; 2];
                for (b, out) in e.iter_mut().enumerate() {
                    for j in 0..SECTION {
                        let v = llvq_search::tetra::val(p + 2 * b as u32, (row >> (4 * j)) & 15);
                        let v = i8::try_from(v).expect("a Tetra coordinate fits an i8");
                        // What `avx2::norm2` takes squares in bytes on.
                        assert!(v.unsigned_abs() <= 11, "a Tetra coordinate of {v} past the byte squares");
                        *out |= u64::from(v as u8) << (8 * j);
                    }
                }
                lut.0[p as usize][ri] = e;
            }
        }
        let mut paths: Box<[u32; PATHS]> = vec![0u32; PATHS].into_boxed_slice().try_into().expect("4096 paths");
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
        #[cfg(target_arch = "x86_64")]
        let sel = {
            let mut sel = [[0x80u8; 16]; 256];
            for (c, s) in sel.iter_mut().enumerate() {
                for (j, b) in s.iter_mut().take(SECTION).enumerate() {
                    *b = (j + SECTION * ((c >> j) & 1)) as u8;
                }
            }
            Sel(sel)
        };
        let mut invnorm = [0f32; 32];
        for (m, e) in invnorm.iter_mut().enumerate().skip(1) {
            *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
        }
        let mut nibval = [[0u8; 16]; 2];
        for (p, tab) in nibval.iter_mut().enumerate() {
            for (n, v) in tab.iter_mut().enumerate() {
                let y = llvq_search::tetra::val(p as u32 + 2 * (n as u32 >> 3), n as u32 & 7);
                *v = i8::try_from(y).expect("a Tetra coordinate fits an i8") as u8;
            }
        }
        let mut order = [0usize; DIM];
        for (o, &n) in order.iter_mut().zip(t.order()) {
            *o = n as usize;
        }
        Self {
            lut,
            paths,
            mask,
            #[cfg(target_arch = "x86_64")]
            sel,
            invnorm,
            nibval,
            order,
        }
    }

    /// Where a word's sections are: its parity, the three table rows, the
    /// three code bytes, and its gain bit.
    #[inline(always)]
    fn locate(&self, w: u64) -> (usize, [usize; 3], [usize; 3], usize) {
        let p = (w & 1) as usize;
        let r = ((w >> 1) & 1) as usize;
        let path_at = ((w >> 2) & 63) << 6 | ((w >> 8) & 1) << 5 | ((w >> 20) & 15) << 1 | ((w >> 35) & 1);
        let path = self.paths[path_at as usize] as usize;
        let i1 = ((w >> 9) & 0x7ff) as usize;
        let i2 = ((w >> 24) & 0x7ff) as usize;
        let i3 = ((w >> 36) & 0x7ff) as usize;
        let delta = usize::from(i2 >= N0_MIXED);
        let r1 = CLASS_ROWS * r + i1;
        let r2 = i2 + delta * (CLASS_ROWS - N0_MIXED);
        let r3 = CLASS_ROWS * (p ^ r ^ delta) + i3;
        // The mask is a no-op on these indices (the asserts by `PATHS`); it
        // lets the compiler drop the bounds check.
        let rows = [r1 & (ROWS - 1), r2 & (ROWS - 1), r3 & (ROWS - 1)];
        (p, rows, [path & 0xff, (path >> 8) & 0xff, (path >> 16) & 0xff], ((w >> 47) & 1) as usize)
    }

    /// The three sections of a word, eight `i8` lanes each, and its gain bit.
    #[inline(always)]
    pub fn sections(&self, w: u64) -> ([u64; 3], usize) {
        let (p, rows, codes, g) = self.locate(w);
        let lut = &self.lut[p];
        let pick = |k: usize| {
            let e = lut[rows[k]];
            let m = self.mask[codes[k]];
            (e[0] & !m) | (e[1] & m)
        };
        ([pick(0), pick(1), pick(2)], g)
    }

    /// `gscale[g] / ‖y‖` for every gain bit and shell, at `g << 5 | m`: the
    /// product [`Self::scale`] takes, formed once per matrix call.
    pub fn scale_table(&self, gscale: &[f32; 2]) -> [f32; 64] {
        let mut s = [0f32; 64];
        for (i, v) in s.iter_mut().enumerate() {
            *v = gscale[i >> 5] * self.invnorm[i & 31];
        }
        s
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

    /// `gscale[g] / ‖y‖` from `n2 = ‖y‖²`, through the kernels' norm table
    /// folded into [`Self::scale_table`]'s.
    #[inline(always)]
    fn scale(sc: &[f32; 64], g: usize, n2: u32) -> f32 {
        sc[((g & 1) << 5) | ((n2 >> 4) & 31) as usize]
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
    pub(crate) fn finish(&self, isa: Isa, r: usize, acc: f32, x: &[f32]) -> f32 {
        let tail = &self.tail[r * self.tail_w..(r + 1) * self.tail_w];
        let xt = &x[self.nblocks * DIM..];
        match isa {
            // SAFETY: as in `tetra_row`.
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe { avx2::finish(self.rscale[r], tail, acc, xt) },
            _ => {
                let mut s = acc * self.rscale[r];
                for (&h, &v) in tail.iter().zip(xt) {
                    s = half::f16::from_bits(h).to_f32().mul_add(v, s);
                }
                s
            }
        }
    }
}

/// The 48-bit word of block `j` of a row, read as `f1r_load` reads it.
#[inline(always)]
fn word_at(row: &[u32], j: usize) -> u64 {
    let w = (3 * j) >> 1;
    let pair = u64::from(row[w]) | u64::from(row[w + 1]) << 32;
    (pair >> (16 * (j & 1))) & 0xffff_ffff_ffff
}

/// The words of blocks `2i` and `2i + 1`, from the three u32 the two fill:
/// what [`word_at`] reads, with one load fewer and no bounds check.
#[inline(always)]
fn pair_words(w: &[u32; 3]) -> (u64, u64) {
    let (a, b, c) = (u64::from(w[0]), u64::from(w[1]), u64::from(w[2]));
    (a | (b & 0xffff) << 32, b >> 16 | c << 16)
}

/// `f(j, word)` for every block `j` of a row, in order, two at a time. A row
/// of `n` blocks holds at least `3⌊n/2⌋` u32 ([`TetraMatrix::check`]), and the
/// odd last block is read by [`word_at`].
///
/// The order of the blocks is the order of the sums: one accumulator per row,
/// as before, so one row and a batch round the same way.
#[inline(always)]
fn for_each_word(row: &[u32], nblocks: usize, mut f: impl FnMut(usize, u64)) {
    let pairs = nblocks / 2;
    for (i, w) in row[..3 * pairs].as_chunks::<3>().0.iter().enumerate() {
        let (w0, w1) = pair_words(w);
        f(2 * i, w0);
        f(2 * i + 1, w1);
    }
    if nblocks % 2 == 1 {
        f(nblocks - 1, word_at(row, nblocks - 1));
    }
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
                // F16C converts the binary16 scales and tails; every CPU with
                // AVX2 has it, the check costs nothing.
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                    && std::arch::is_x86_feature_detected!("f16c")
                {
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

fn tetra_row_scalar(t: &Tables, row: &[u32], nblocks: usize, sc: &[f32; 64], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for_each_word(row, nblocks, |j, w| {
        let (s, g) = t.sections(w);
        let xb = &x[j * DIM..(j + 1) * DIM];
        let (mut d, mut n2) = (0f32, 0u32);
        for (k, sec) in s.iter().enumerate() {
            for l in 0..SECTION {
                let v = i32::from((sec >> (8 * l)) as u8 as i8);
                n2 += (v * v) as u32;
                d = (v as f32).mul_add(xb[SECTION * k + l], d);
            }
        }
        acc = d.mul_add(Tables::scale(sc, g, n2), acc);
    });
    acc
}

fn tetra_rows_scalar(
    t: &Tables,
    row: &[u32],
    nblocks: usize,
    sc: &[f32; 64],
    xs: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    let n = out.len();
    let mut acc = [0f32; MAX_BATCH];
    for_each_word(row, nblocks, |j, w| {
        let (y, g) = t.decode_trio(w);
        let n2: i32 = y.iter().map(|v| v * v).sum();
        let s = Tables::scale(sc, g, n2 as u32);
        for (r, a) in acc.iter_mut().enumerate().take(n) {
            let xb = &xs[r * stride + j * DIM..r * stride + (j + 1) * DIM];
            let mut d = 0f32;
            for (v, &xv) in y.iter().zip(xb) {
                d = (*v as f32).mul_add(xv, d);
            }
            *a = d.mul_add(s, *a);
        }
    });
    out.copy_from_slice(&acc[..n]);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{for_each_word, Tables, DIM, MAX_BATCH};
    use std::arch::aarch64::*;

    /// [`Tables::sections`] with the candidate select done in vector
    /// registers: each table row loads as one `uint64x2` (both candidates),
    /// its mask as one `uint64x1`, and a bit select merges them. Same lanes,
    /// fewer scalar instructions and no general-to-vector move.
    #[inline(always)]
    unsafe fn sections(t: &Tables, w: u64) -> ([int8x8_t; 3], usize) {
        let (p, rows, codes, g) = t.locate(w);
        let lut = &t.lut[p];
        let pick = |k: usize| {
            let e = vld1q_u64(lut[rows[k]].as_ptr());
            let m = vld1_u64(&t.mask[codes[k]]);
            vreinterpret_s8_u64(vbsl_u64(m, vget_high_u64(e), vget_low_u64(e)))
        };
        ([pick(0), pick(1), pick(2)], g)
    }

    /// The 24 lanes of a block as three `int16x8`, and `‖y‖²`.
    #[inline(always)]
    unsafe fn widen(s: [int8x8_t; 3]) -> (int16x8_t, int16x8_t, int16x8_t, u32) {
        let a0 = vmovl_s8(s[0]);
        let a1 = vmovl_s8(s[1]);
        let a2 = vmovl_s8(s[2]);
        // Each lane holds three squares of at most 16², and the whole sum is
        // at most 8 · 768 = 6,144: both fit an i16.
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

    pub fn row(t: &Tables, row: &[u32], nblocks: usize, sc: &[f32; 64], x: &[f32]) -> f32 {
        assert!(x.len() >= nblocks * DIM);
        // SAFETY: NEON is part of the aarch64 baseline, so the intrinsics are
        // always available; every load reads 24 floats of `x` at block `j`,
        // inside the length asserted above.
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for_each_word(row, nblocks, |j, w| {
                let (s, g) = sections(t, w);
                let (a0, a1, a2, n2) = widen(s);
                let f = [lo(a0), hi(a0), lo(a1), hi(a1), lo(a2), hi(a2)];
                let d = dot(&f, x.as_ptr().add(j * DIM));
                acc = vfmaq_n_f32(acc, d, Tables::scale(sc, g, n2));
            });
            vaddvq_f32(acc)
        }
    }

    pub fn rows(
        t: &Tables,
        row: &[u32],
        nblocks: usize,
        sc: &[f32; 64],
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
            for_each_word(row, nblocks, |j, w| {
                let (s, g) = sections(t, w);
                let (a0, a1, a2, n2) = widen(s);
                let f = [lo(a0), hi(a0), lo(a1), hi(a1), lo(a2), hi(a2)];
                let s = Tables::scale(sc, g, n2);
                for (r, a) in acc.iter_mut().enumerate().take(n) {
                    let d = dot(&f, xs.as_ptr().add(r * stride + j * DIM));
                    *a = vfmaq_n_f32(*a, d, s);
                }
            });
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = vaddvq_f32(*a);
            }
        }
    }

    /// The 24 lanes of a block of [`super::TetraNib`]: twelve bytes split
    /// into nibbles, interleaved back into coordinate order, and looked up
    /// in the parity's sixteen values. The same floats as [`sections`],
    /// [`widen`], [`lo`] and [`hi`] give for the block's word.
    ///
    /// # Safety
    /// `nib` points at 16 readable bytes, the block's twelve first.
    #[inline(always)]
    unsafe fn nib_lanes(t: &Tables, nib: *const u8, h: u8) -> [float32x4_t; 6] {
        let v = vld1q_u8(nib);
        let (l, u) = (vandq_u8(v, vdupq_n_u8(0x0f)), vshrq_n_u8(v, 4));
        let tab = vreinterpretq_s8_u8(vld1q_u8(t.nibval[usize::from(h >> 6) & 1].as_ptr()));
        // Coordinates 0..16, then 16..24 in the low half.
        let (y0, y1) = (vqtbl1q_s8(tab, vzip1q_u8(l, u)), vqtbl1q_s8(tab, vzip2q_u8(l, u)));
        let (a0, a1, a2) = (vmovl_s8(vget_low_s8(y0)), vmovl_high_s8(y0), vmovl_s8(vget_low_s8(y1)));
        [lo(a0), hi(a0), lo(a1), hi(a1), lo(a2), hi(a2)]
    }

    pub fn nib_row(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], x: &[f32]) -> f32 {
        let n = hdr.len();
        assert!(x.len() >= n * DIM && nib.len() >= n * super::NIB_BYTES + super::NIB_PAD);
        // SAFETY: NEON baseline; block `j` reads 16 bytes at `12 j`, inside
        // the pad asserted above, and 24 floats of `x` at `24 j`.
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for (j, &h) in hdr.iter().enumerate() {
                let f = nib_lanes(t, nib.as_ptr().add(j * super::NIB_BYTES), h);
                let d = dot(&f, x.as_ptr().add(j * DIM));
                acc = vfmaq_n_f32(acc, d, sc[usize::from(h & 63)]);
            }
            vaddvq_f32(acc)
        }
    }

    pub fn nib_rows(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], xs: &[f32], stride: usize, out: &mut [f32]) {
        let (n, nb) = (out.len(), hdr.len());
        assert!(n <= MAX_BATCH && n > 0 && nib.len() >= nb * super::NIB_BYTES + super::NIB_PAD);
        assert!(stride >= nb * DIM && xs.len() >= (n - 1) * stride + nb * DIM);
        // SAFETY: as in `nib_row`; activation `r` is read at `r * stride + 24 j`.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); MAX_BATCH];
            for (j, &h) in hdr.iter().enumerate() {
                let f = nib_lanes(t, nib.as_ptr().add(j * super::NIB_BYTES), h);
                let s = sc[usize::from(h & 63)];
                for (r, a) in acc.iter_mut().enumerate().take(n) {
                    let d = dot(&f, xs.as_ptr().add(r * stride + j * DIM));
                    *a = vfmaq_n_f32(*a, d, s);
                }
            }
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = vaddvq_f32(*a);
            }
        }
    }

    /// `Σ q·x` over one int4 group of `half_g` bytes, `x` deinterleaved:
    /// `xe` the even elements (low nibbles), `xo` the odd ones (high
    /// nibbles). Four lanes still to be summed, from four accumulators.
    #[inline(always)]
    unsafe fn q4_group(bytes: *const u8, xe: *const f32, xo: *const f32, half_g: usize) -> float32x4_t {
        let m = vdupq_n_u8(0x0f);
        let mut s = [vdupq_n_f32(0.0); 4];
        for c in (0..half_g).step_by(16) {
            let v = vld1q_u8(bytes.add(c));
            let l = vandq_u8(v, m);
            let h = vshrq_n_u8(v, 4);
            for (half, xp) in [(l, xe.add(c)), (h, xo.add(c))] {
                let w0 = vmovl_u8(vget_low_u8(half));
                let w1 = vmovl_high_u8(half);
                s[0] = vfmaq_f32(s[0], vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0))), vld1q_f32(xp));
                s[1] = vfmaq_f32(s[1], vcvtq_f32_u32(vmovl_high_u16(w0)), vld1q_f32(xp.add(4)));
                s[2] = vfmaq_f32(s[2], vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1))), vld1q_f32(xp.add(8)));
                s[3] = vfmaq_f32(s[3], vcvtq_f32_u32(vmovl_high_u16(w1)), vld1q_f32(xp.add(12)));
            }
        }
        vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3]))
    }

    /// One int4 row: `Σ_g scale_g · Σ q·x + bias_g · Σ x`, the scaled group
    /// sums kept in lanes and summed once per row.
    pub fn q4_row(bytes: &[u8], scales: &[u16], biases: &[u16], half_g: usize, a: &super::Q4Act) -> f32 {
        let gpr = scales.len();
        assert!(half_g.is_multiple_of(16) && bytes.len() == gpr * half_g && biases.len() == gpr);
        assert!(a.xe.len() >= bytes.len() && a.xo.len() >= bytes.len() && a.gsum.len() >= gpr);
        // SAFETY: NEON baseline; group `g` reads `half_g` bytes and floats of
        // each half at `g · half_g`, inside the lengths asserted above.
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            let mut bias = 0f32;
            for g in 0..gpr {
                let at = g * half_g;
                let d = q4_group(bytes.as_ptr().add(at), a.xe.as_ptr().add(at), a.xo.as_ptr().add(at), half_g);
                acc = vfmaq_n_f32(acc, d, half::f16::from_bits(scales[g]).to_f32());
                bias = half::f16::from_bits(biases[g]).to_f32().mul_add(a.gsum[g], bias);
            }
            vaddvq_f32(acc) + bias
        }
    }

    /// `Σ w·x` over one BF16 row. A weight is widened to f32 by a zip with
    /// zeros, which puts its 16 bits in the high half of the lane; four
    /// accumulators of four lanes take 16 weights a step, so four FMA chains
    /// run side by side. The weights past the last multiple of 16 are added
    /// after, in order, as [`super::bf16_row_tail`] does.
    pub fn bf16_row(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        assert_eq!(x.len(), n, "a BF16 row of {n} against {} values", x.len());
        let main = n - n % 16;
        // SAFETY: NEON baseline; the step at `c` reads 16 u16 of `w` and 16
        // f32 of `x` from `c`, and `c + 16 <= main <= n`.
        let s = unsafe {
            let z = vdupq_n_u16(0);
            let mut a = [vdupq_n_f32(0.0); 4];
            let (wp, xp) = (w.as_ptr(), x.as_ptr());
            for c in (0..main).step_by(16) {
                let v0 = vld1q_u16(wp.add(c));
                let v1 = vld1q_u16(wp.add(c + 8));
                let f = [vzip1q_u16(z, v0), vzip2q_u16(z, v0), vzip1q_u16(z, v1), vzip2q_u16(z, v1)];
                for (k, fk) in f.into_iter().enumerate() {
                    a[k] = vfmaq_f32(a[k], vreinterpretq_f32_u16(fk), vld1q_f32(xp.add(c + 4 * k)));
                }
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])))
        };
        super::bf16_row_tail(&w[main..], &x[main..], s)
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{word_at, Tables, CLASS_ROWS, DIM, MAX_BATCH, N0_MIXED, PATHS, ROWS};
    use std::arch::x86_64::*;

    // The Tetra rows. Per block the scalar path spends most of its
    // instructions forming addresses (`Tables::locate`): about 45 integer
    // operations a block, on a core with four integer ALUs, ahead of a long
    // dependency chain. Here the addresses of eight blocks are formed at once
    // in vector registers and stored, a chunk ahead of the decode that reads
    // them back as plain loads. The decode then selects candidates with a
    // byte shuffle and takes the norm in integers. Every sum is formed as
    // before, in the same block order.

    /// Blocks whose addresses one pass forms before it decodes them. A
    /// multiple of eight; large enough that a block's fields were stored
    /// some hundred instructions before they are loaded, so the loads read
    /// the cache and do not wait on store forwarding.
    const CHUNK: usize = 64;

    /// Where each block of a chunk reads, one array of `CHUNK` per field so
    /// that eight blocks store as one vector: fields 0 to 2 the byte offsets
    /// of its three entries in [`super::Lut::flat`] (parity included),
    /// [`PATH`] its path index, [`GS`] `g << 5`, the gain's half of the
    /// scale table. Left uninitialised and handled through raw pointers:
    /// every entry the decode reads was written first, and zeroing 1.25 KiB
    /// a row would be a cost for nothing.
    #[repr(C, align(32))]
    struct Where([[u32; CHUNK]; 5]);
    const PATH: usize = 3;
    const GS: usize = 4;

    /// [`Where`] for blocks `b .. b + 8` of a chunk, from the twelve u32 at
    /// `u` that hold their words: the fields of [`Tables::locate`], formed on
    /// eight words at once in 32-bit lanes.
    ///
    /// # Safety
    /// AVX2; `u` points at twelve readable u32; `wh` at a [`Where`];
    /// `b + 8 ≤ CHUNK`.
    #[inline(always)]
    unsafe fn prep8(u: *const u32, wh: *mut u32, b: usize) {
        // Lane 0 of each vector holds pairs 0 and 1 (u32 0..6), lane 1 pairs
        // 2 and 3 (u32 6..12): x = u32 0..4 | 6..10, y = u32 2..6 | 8..12.
        let x = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(u.cast())), _mm_loadu_si128(u.add(6).cast()), 1);
        let y = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(u.add(2).cast())), _mm_loadu_si128(u.add(8).cast()), 1);
        let (xf, yf) = (_mm256_castsi256_ps(x), _mm256_castsi256_ps(y));
        // Word `i` of a lane starts in u32 `a_i` and ends in `c_i`: pair
        // (a, b, c) holds words a | b << 32 and b >> 16 | c << 16.
        let a = _mm256_castps_si256(_mm256_shuffle_ps(xf, yf, 0b10_01_01_00));
        let c = _mm256_castps_si256(_mm256_shuffle_ps(xf, yf, 0b11_10_10_01));
        let odd = _mm256_setr_epi32(0, 16, 0, 16, 0, 16, 0, 16);
        // Bits 0..32 of each word, and bits 32..48.
        let lo = _mm256_or_si256(
            _mm256_srlv_epi32(a, odd),
            _mm256_sllv_epi32(c, _mm256_setr_epi32(32, 16, 32, 16, 32, 16, 32, 16)),
        );
        let hi = _mm256_and_si256(_mm256_srlv_epi32(c, odd), _mm256_set1_epi32(0xffff));

        let k = |v: u32| _mm256_set1_epi32(v as i32);
        let (and, or) = (|a, b| _mm256_and_si256(a, b), |a, b| _mm256_or_si256(a, b));
        let p = and(lo, k(1));
        let r = and(lo, k(2));
        // p · ROWS entries of 16 bytes.
        let pb = _mm256_slli_epi32(p, 16);
        // i1 = bits 9..20, i2 = bits 24..35, i3 = bits 36..47, all << 4.
        let i1 = and(_mm256_srli_epi32(lo, 5), k(0x7ff << 4));
        let i2 = and(or(_mm256_srli_epi32(lo, 24), _mm256_slli_epi32(hi, 8)), k(0x7ff));
        let i3 = and(hi, k(0x7ff << 4));
        let delta = _mm256_cmpgt_epi32(i2, k(N0_MIXED as u32 - 1));
        // Row CLASS_ROWS · r + i1.
        let o1 = or(or(pb, _mm256_slli_epi32(r, 14)), i1);
        // Row i2 + δ · (CLASS_ROWS − N0_MIXED).
        let o2 = _mm256_add_epi32(
            _mm256_add_epi32(pb, _mm256_slli_epi32(i2, 4)),
            and(delta, k(((CLASS_ROWS - N0_MIXED) << 4) as u32)),
        );
        // Row CLASS_ROWS · (p ^ r ^ δ) + i3.
        let class3 = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_slli_epi32(p, 15), _mm256_slli_epi32(r, 14)),
            and(delta, k(1 << 15)),
        );
        let o3 = or(or(pb, class3), i3);
        // (s8, b1, b2, b3) = bits 2..8, 8, 20..24, 35, as `locate` packs them.
        let path = or(
            or(and(_mm256_slli_epi32(lo, 4), k(0xfc0)), and(_mm256_srli_epi32(lo, 3), k(0x20))),
            or(and(_mm256_srli_epi32(lo, 19), k(0x1e)), and(_mm256_srli_epi32(hi, 3), k(1))),
        );
        // The gain bit, bit 47, at bit 5.
        let gs = and(_mm256_srli_epi32(hi, 10), k(32));
        let at = |f: usize| wh.add(f * CHUNK + b).cast::<__m256i>();
        _mm256_storeu_si256(at(0), o1);
        _mm256_storeu_si256(at(1), o2);
        _mm256_storeu_si256(at(2), o3);
        _mm256_storeu_si256(at(PATH), path);
        _mm256_storeu_si256(at(GS), gs);
    }

    /// [`Where`] for block `b` of a chunk from its word alone, through
    /// [`Tables::locate`]: the route of a row's last `nblocks mod 8` blocks.
    ///
    /// # Safety
    /// `wh` points at a [`Where`]; `b < CHUNK`.
    #[inline(always)]
    unsafe fn prep1(t: &Tables, w: u64, wh: *mut u32, b: usize) {
        let (p, rows, _, g) = t.locate(w);
        for (f, row) in rows.into_iter().enumerate() {
            wh.add(f * CHUNK + b).write(((p * ROWS + row) << 4) as u32);
        }
        let path = ((w >> 2) & 63) << 6 | ((w >> 8) & 1) << 5 | ((w >> 20) & 15) << 1 | ((w >> 35) & 1);
        wh.add(PATH * CHUNK + b).write(path as u32);
        wh.add(GS * CHUNK + b).write((g << 5) as u32);
    }

    /// `‖y‖²` of a block from its selected bytes, in integers. Every lane has
    /// `|y| ≤ 11` ([`Tables::new`] asserts it), so a square is a byte looked
    /// up by `|y|`, two squares add in a byte without carry (at most 242),
    /// and `psadbw` sums each half. Exact, as the f32 sum it replaces was, so
    /// the scale it indexes is the same; it starts from the bytes, so it does
    /// not wait for the float conversion, and it needs no saturating
    /// float-to-int.
    #[inline(always)]
    unsafe fn norm2(s: &[__m128i; 3]) -> u32 {
        let sq = _mm_setr_epi8(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 0, 0, 0, 0);
        // Sections 0 and 1 side by side; section 2's high half is zero.
        let a = _mm_shuffle_epi8(sq, _mm_abs_epi8(_mm_unpacklo_epi64(s[0], s[1])));
        let b = _mm_shuffle_epi8(sq, _mm_abs_epi8(s[2]));
        let d = _mm_sad_epu8(_mm_add_epi8(a, b), _mm_setzero_si128());
        let d = _mm_add_epi32(d, _mm_unpackhi_epi64(d, d));
        _mm_cvtsi128_si32(d) as u32
    }

    /// The eight `i8` lanes of a section as f32, exactly.
    #[inline(always)]
    unsafe fn lanes(s: __m128i) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s))
    }

    /// Block `b` of a chunk: its three lane vectors, and its scale
    /// `gscale[g] / ‖y‖` in every lane.
    ///
    /// Each table entry loads as one 16-byte vector (both candidates), and
    /// `sel[code]` takes byte `j` or `j + 8` of it into lane `j`, the high
    /// half zero: [`Tables::sections`] without the scalar merge and the
    /// general-to-vector move.
    ///
    /// # Safety
    /// AVX2; entry `b` of the [`Where`] at `wh` written by [`prep8`] or
    /// [`prep1`]. Those store
    /// every offset as a masked field: a 16-byte multiple below
    /// `2 · ROWS · 16` (at most `65536 + 32768 + 32752`), a path index below
    /// `PATHS`, and `g << 5` in `{0, 32}`; so every unchecked read below is
    /// inside its table. Masks here would cost a constant and an AND per
    /// read, on the loop that is the token.
    #[inline(always)]
    unsafe fn decode(t: &Tables, wh: *const u32, b: usize, sc: &[f32; 64]) -> ([__m256; 3], __m256) {
        let lut = t.lut.flat().as_ptr().cast::<u8>();
        let (pi, gs) = (wh.add(PATH * CHUNK + b).read(), wh.add(GS * CHUNK + b).read());
        debug_assert!((pi as usize) < PATHS && gs <= 32);
        let path = *t.paths.get_unchecked(pi as usize);
        let pick = |k: usize, code: u32| {
            let o = wh.add(k * CHUNK + b).read() as usize;
            debug_assert!(o < 2 * ROWS * 16 && o.is_multiple_of(16));
            // Both tables are 64-byte aligned with 16-byte entries.
            let e = _mm_load_si128(lut.add(o).cast());
            _mm_shuffle_epi8(e, _mm_load_si128(t.sel.0[(code & 0xff) as usize].as_ptr().cast()))
        };
        let s = [pick(0, path), pick(1, path >> 8), pick(2, path >> 16)];
        let n2 = norm2(&s);
        let scale = *sc.get_unchecked((gs | ((n2 >> 4) & 31)) as usize);
        ([lanes(s[0]), lanes(s[1]), lanes(s[2])], _mm256_set1_ps(scale))
    }

    /// `f(j, lanes, scale)` for every block `j` of a row, in order: one call
    /// site, so that `f` inlines.
    ///
    /// # Safety
    /// AVX2.
    #[inline(always)]
    unsafe fn each_block(
        t: &Tables,
        row: &[u32],
        nblocks: usize,
        sc: &[f32; 64],
        mut f: impl FnMut(usize, &[__m256; 3], __m256),
    ) {
        // Blocks `8q .. 8q + 8` fill u32 `12q .. 12q + 12`.
        let grouped = nblocks / 8 * 8;
        assert!(3 * grouped / 2 <= row.len(), "a row of {} u32 for {nblocks} blocks", row.len());
        let mut wh = std::mem::MaybeUninit::<Where>::uninit();
        let wh = wh.as_mut_ptr().cast::<u32>();
        let mut start = 0;
        while start < nblocks {
            let len = (nblocks - start).min(CHUNK);
            // `start` and `grouped` are multiples of 8, so is `vec`.
            let vec = grouped.min(start + len) - start;
            for b in (0..vec).step_by(8) {
                prep8(row.as_ptr().add(3 * (start + b) / 2), wh, b);
            }
            for b in vec..len {
                prep1(t, word_at(row, start + b), wh, b);
            }
            for b in 0..len {
                let (lanes, scale) = decode(t, wh, b, sc);
                f(start + b, &lanes, scale);
            }
            start += len;
        }
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
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
    pub unsafe fn row(t: &Tables, row: &[u32], nblocks: usize, sc: &[f32; 64], x: &[f32]) -> f32 {
        assert!(x.len() >= nblocks * DIM);
        let mut acc = _mm256_setzero_ps();
        // `j < nblocks`, so the offsets below stay inside the lengths
        // asserted; `wrapping_mul` spares the overflow check a profile with
        // overflow checks would put on every block.
        each_block(t, row, nblocks, sc, |j, f, s| {
            let d = dot(f, x.as_ptr().add(j.wrapping_mul(DIM)));
            acc = _mm256_fmadd_ps(d, s, acc);
        });
        hsum(acc)
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rows(
        t: &Tables,
        row: &[u32],
        nblocks: usize,
        sc: &[f32; 64],
        xs: &[f32],
        stride: usize,
        out: &mut [f32],
    ) {
        let n = out.len();
        assert!(n <= MAX_BATCH && n > 0);
        assert!(stride >= nblocks * DIM && xs.len() >= (n - 1) * stride + nblocks * DIM);
        let mut acc = [_mm256_setzero_ps(); MAX_BATCH];
        // As in `row`: every offset is inside the lengths asserted above.
        each_block(t, row, nblocks, sc, |j, f, s| {
            let xb = xs.as_ptr().add(j.wrapping_mul(DIM));
            for (r, a) in acc.iter_mut().enumerate().take(n) {
                let d = dot(f, xb.add(r.wrapping_mul(stride)));
                *a = _mm256_fmadd_ps(d, s, *a);
            }
        });
        for (o, a) in out.iter_mut().zip(acc.iter()) {
            *o = hsum(*a);
        }
    }

    /// The 24 lanes of a block of [`super::TetraNib`], as `neon::nib_lanes`:
    /// the same floats [`decode`] gives for the block's word.
    ///
    /// # Safety
    /// AVX2; `nib` points at 16 readable bytes, the block's twelve first.
    #[inline(always)]
    unsafe fn nib_lanes(t: &Tables, nib: *const u8, h: u8) -> [__m256; 3] {
        let m = _mm_set1_epi8(0x0f);
        let v = _mm_loadu_si128(nib.cast());
        let (l, u) = (_mm_and_si128(v, m), _mm_and_si128(_mm_srli_epi16(v, 4), m));
        let tab = _mm_loadu_si128(t.nibval[usize::from(h >> 6) & 1].as_ptr().cast());
        // Coordinates 0..16, then 16..24 in the low half.
        let y0 = _mm_shuffle_epi8(tab, _mm_unpacklo_epi8(l, u));
        let y1 = _mm_shuffle_epi8(tab, _mm_unpackhi_epi8(l, u));
        [lanes(y0), lanes(_mm_srli_si128(y0, 8)), lanes(y1)]
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn nib_row(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], x: &[f32]) -> f32 {
        let n = hdr.len();
        assert!(x.len() >= n * DIM && nib.len() >= n * super::NIB_BYTES + super::NIB_PAD);
        let mut acc = _mm256_setzero_ps();
        // Block `j` reads 16 bytes at `12 j`, inside the pad asserted above,
        // and 24 floats of `x` at `24 j`.
        for (j, &h) in hdr.iter().enumerate() {
            let f = nib_lanes(t, nib.as_ptr().add(j.wrapping_mul(super::NIB_BYTES)), h);
            let d = dot(&f, x.as_ptr().add(j.wrapping_mul(DIM)));
            acc = _mm256_fmadd_ps(d, _mm256_set1_ps(sc[usize::from(h & 63)]), acc);
        }
        hsum(acc)
    }

    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn nib_rows(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], xs: &[f32], stride: usize, out: &mut [f32]) {
        let (n, nb) = (out.len(), hdr.len());
        assert!(n <= MAX_BATCH && n > 0 && nib.len() >= nb * super::NIB_BYTES + super::NIB_PAD);
        assert!(stride >= nb * DIM && xs.len() >= (n - 1) * stride + nb * DIM);
        let mut acc = [_mm256_setzero_ps(); MAX_BATCH];
        for (j, &h) in hdr.iter().enumerate() {
            let f = nib_lanes(t, nib.as_ptr().add(j.wrapping_mul(super::NIB_BYTES)), h);
            let s = _mm256_set1_ps(sc[usize::from(h & 63)]);
            let xb = xs.as_ptr().add(j.wrapping_mul(DIM));
            for (r, a) in acc.iter_mut().enumerate().take(n) {
                let d = dot(&f, xb.add(r.wrapping_mul(stride)));
                *a = _mm256_fmadd_ps(d, s, *a);
            }
        }
        for (o, a) in out.iter_mut().zip(acc.iter()) {
            *o = hsum(*a);
        }
    }

    /// `Σ q·x` over one int4 group of `half_g` bytes, as in `neon`: eight
    /// lanes still to be summed, from four accumulators.
    #[inline(always)]
    unsafe fn q4_group(bytes: *const u8, xe: *const f32, xo: *const f32, half_g: usize) -> __m256 {
        let m = _mm_set1_epi8(0x0f);
        let mut s = [_mm256_setzero_ps(); 4];
        for c in (0..half_g).step_by(16) {
            let v = _mm_loadu_si128(bytes.add(c) as *const __m128i);
            let l = _mm_and_si128(v, m);
            let h = _mm_and_si128(_mm_srli_epi16(v, 4), m);
            for (k, (half, xp)) in [(l, xe.add(c)), (h, xo.add(c))].into_iter().enumerate() {
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(half));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(half, 8)));
                s[2 * k] = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xp), s[2 * k]);
                s[2 * k + 1] = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp.add(8)), s[2 * k + 1]);
            }
        }
        _mm256_add_ps(_mm256_add_ps(s[0], s[1]), _mm256_add_ps(s[2], s[3]))
    }

    /// A binary16 value as f32, exactly, by `vcvtph2ps`. `half`'s own
    /// conversion is a runtime F16C check and an out-of-line call on x86_64,
    /// twice per int4 group; this is one instruction.
    #[inline(always)]
    unsafe fn h2f(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
    }

    /// One int4 row, as `neon::q4_row`.
    ///
    /// # Safety
    /// The CPU must have AVX2, FMA and F16C ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_row(bytes: &[u8], scales: &[u16], biases: &[u16], half_g: usize, a: &super::Q4Act) -> f32 {
        let gpr = scales.len();
        assert!(half_g.is_multiple_of(16) && bytes.len() == gpr * half_g && biases.len() == gpr);
        assert!(a.xe.len() >= bytes.len() && a.xo.len() >= bytes.len() && a.gsum.len() >= gpr);
        let mut acc = _mm256_setzero_ps();
        let mut bias = 0f32;
        for g in 0..gpr {
            let at = g * half_g;
            let d = q4_group(bytes.as_ptr().add(at), a.xe.as_ptr().add(at), a.xo.as_ptr().add(at), half_g);
            acc = _mm256_fmadd_ps(d, _mm256_set1_ps(h2f(scales[g])), acc);
            bias = h2f(biases[g]).mul_add(a.gsum[g], bias);
        }
        hsum(acc) + bias
    }

    /// [`super::TetraMatrix::finish`] for one row: `acc · rscale`, then the
    /// binary16 tail, one fused multiply-add per weight in order, as the
    /// generic loop does. There, on x86_64 without the `fma` feature,
    /// `mul_add` is a call to `fmaf` and each conversion a call too; here
    /// both are one instruction, with the same roundings.
    ///
    /// # Safety
    /// The CPU must have AVX2, FMA and F16C ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn finish(rscale: f32, tail: &[u16], acc: f32, xt: &[f32]) -> f32 {
        let mut s = acc * rscale;
        for (&h, &v) in tail.iter().zip(xt) {
            s = h2f(h).mul_add(v, s);
        }
        s
    }

    /// One BF16 row, as `neon::bf16_row`: a weight zero-extended to 32 bits
    /// and shifted left by 16 is its f32; four accumulators of eight lanes
    /// take 32 weights a step. The weights past the last multiple of 32 are
    /// added after, in order.
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA ([`super::Isa::detect`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn bf16_row(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        assert_eq!(x.len(), n, "a BF16 row of {n} against {} values", x.len());
        let main = n - n % 32;
        let mut a = [_mm256_setzero_ps(); 4];
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        // The step at `c` reads 32 u16 of `w` and 32 f32 of `x` from `c`, and `c + 32 <= main <= n`.
        for c in (0..main).step_by(32) {
            for (k, ak) in a.iter_mut().enumerate() {
                let v = _mm256_cvtepu16_epi32(_mm_loadu_si128(wp.add(c + 8 * k).cast()));
                let f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(v));
                *ak = _mm256_fmadd_ps(f, _mm256_loadu_ps(xp.add(c + 8 * k)), *ak);
            }
        }
        let s = hsum(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
        super::bf16_row_tail(&w[main..], &x[main..], s)
    }
}

fn tetra_row(isa: Isa, t: &Tables, row: &[u32], nblocks: usize, sc: &[f32; 64], x: &[f32]) -> f32 {
    match isa {
        Isa::Scalar => tetra_row_scalar(t, row, nblocks, sc, x),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::row(t, row, nblocks, sc, x),
        // SAFETY: `Isa::Avx2` is only produced where the CPU reports both
        // features (`Isa::detect`).
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::row(t, row, nblocks, sc, x) },
    }
}

#[allow(clippy::too_many_arguments)]
fn tetra_rows(
    isa: Isa,
    t: &Tables,
    row: &[u32],
    nblocks: usize,
    sc: &[f32; 64],
    xs: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    match isa {
        Isa::Scalar => tetra_rows_scalar(t, row, nblocks, sc, xs, stride, out),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::rows(t, row, nblocks, sc, xs, stride, out),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::rows(t, row, nblocks, sc, xs, stride, out) },
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
    let sc = t.scale_table(&m.gscale);
    let c = chunk_rows(m.d_out);
    pool().install(|| {
        y.par_chunks_mut(c).enumerate().for_each(|(ci, ys)| {
            for (k, yo) in ys.iter_mut().enumerate() {
                let r = ci * c + k;
                let acc = tetra_row(isa, t, m.row(r), m.nblocks, &sc, xp);
                *yo = m.finish(isa, r, acc, xp);
            }
        })
    });
}

/// `Y = W X` for `n` activations, `[n, d_in]` row-major and in trio order,
/// into `[n, d_out]`. Each block is decoded once for the whole batch.
pub fn tetra_matvec_rows(isa: Isa, t: &Tables, m: &TetraMatrix, xs: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(xs.len(), n * m.d_in);
    assert_eq!(y.len(), n * m.d_out);
    let sc = t.scale_table(&m.gscale);
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
                    tetra_rows(isa, t, m.row(r), m.nblocks, &sc, &xs[start * m.d_in..], m.d_in, o);
                    for (i, v) in o.iter_mut().enumerate() {
                        let x = &xs[(start + i) * m.d_in..(start + i + 1) * m.d_in];
                        *v = m.finish(isa, r, *v, x);
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

// ---------------------------------------------------------------------------
// The same points, decoded once: four bits a coordinate.
// ---------------------------------------------------------------------------

/// Bytes of nibbles a block holds: 24 coordinates, four bits each.
const NIB_BYTES: usize = DIM / 2;

/// Bytes past the last block of [`TetraNib::nib`]: the SIMD rows load 16
/// bytes at a block of 12.
const NIB_PAD: usize = 16 - NIB_BYTES;

/// The codes of a [`TetraMatrix`] decoded once into four bits a coordinate.
///
/// In a block of parity `p` a coordinate is `val(p + 2·bit, rank)` with a
/// rank below eight: sixteen values, a nibble. The block's parity, its gain
/// bit and its shell go in one header byte. That is 4.33 bits a weight where
/// the served words hold 2, and a row needs no table walk: twelve bytes, one
/// sixteen-entry lookup, the dot product. The lanes and the scale are the
/// ones [`Tables::sections`] and [`Tables::scale`] give, and every sum is
/// formed in the same order, so a matvec over nibbles equals the matvec over
/// words bit for bit on each SIMD path.
///
/// Row scales and tails stay in the [`TetraMatrix`] it was expanded from.
pub struct TetraNib {
    nblocks: usize,
    /// `d_out × nblocks × 12` bytes and [`NIB_PAD`]: coordinate `t` of a
    /// block, in trio order, is nibble `t & 1` of its byte `t / 2`.
    nib: Vec<u8>,
    /// `d_out × nblocks`: `p << 6 | g << 5 | shell`, the low six bits the
    /// index [`Tables::scale_table`] is read at.
    hdr: Vec<u8>,
}

impl TetraNib {
    /// Every word of `m` decoded, rows across the pool.
    pub fn expand(t: &Tables, m: &TetraMatrix) -> Self {
        let per = m.nblocks;
        let mut nib = vec![0u8; m.d_out * per * NIB_BYTES + NIB_PAD];
        let mut hdr = vec![0u8; m.d_out * per];
        if per == 0 {
            return Self { nblocks: 0, nib, hdr };
        }
        // The nibble of each value, per parity: the two progressions of a
        // parity are disjoint, so a value has one nibble.
        let mut inv = [[0xffu8; 64]; 2];
        for (p, row) in inv.iter_mut().enumerate() {
            for (n, &v) in t.nibval[p].iter().enumerate() {
                row[(i32::from(v as i8) + 32) as usize] = n as u8;
            }
        }
        pool().install(|| {
            nib[..m.d_out * per * NIB_BYTES].par_chunks_mut(per * NIB_BYTES).zip(hdr.par_chunks_mut(per)).enumerate().for_each(
                |(r, (nr, hr))| {
                    for_each_word(m.row(r), per, |j, w| {
                        let (y, g) = t.decode_trio(w);
                        let p = (w & 1) as usize;
                        let mut n2 = 0u32;
                        for (k, &v) in y.iter().enumerate() {
                            n2 += (v * v) as u32;
                            let n = inv[p][((v + 32) & 63) as usize];
                            assert!(n < 16, "word {w:#014x}: coordinate {v} is not a value of parity {p}");
                            nr[j * NIB_BYTES + k / 2] |= n << (4 * (k & 1));
                        }
                        hr[j] = (p as u8) << 6 | (g as u8 & 1) << 5 | ((n2 >> 4) & 31) as u8;
                    });
                },
            )
        });
        Self { nblocks: per, nib, hdr }
    }

    /// Bytes the nibbles and headers hold in memory.
    pub fn bytes(&self) -> u64 {
        (self.nib.len() + self.hdr.len()) as u64
    }

    /// The nibbles of row `r`, [`NIB_PAD`] readable bytes past them, and its
    /// headers.
    #[inline(always)]
    fn row(&self, r: usize) -> (&[u8], &[u8]) {
        let n = self.nblocks;
        (&self.nib[r * n * NIB_BYTES..(r + 1) * n * NIB_BYTES + NIB_PAD], &self.hdr[r * n..(r + 1) * n])
    }
}

/// Coordinate `k` of a block, from its twelve bytes and its value table.
#[inline(always)]
fn nib_coord(tab: &[u8; 16], b: &[u8], k: usize) -> f32 {
    f32::from(tab[((b[k / 2] >> (4 * (k & 1))) & 15) as usize] as i8)
}

fn nib_row_scalar(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (j, &h) in hdr.iter().enumerate() {
        let tab = &t.nibval[usize::from(h >> 6) & 1];
        let (b, xb) = (&nib[j * NIB_BYTES..(j + 1) * NIB_BYTES], &x[j * DIM..(j + 1) * DIM]);
        let mut d = 0f32;
        for (k, &xv) in xb.iter().enumerate() {
            d = nib_coord(tab, b, k).mul_add(xv, d);
        }
        acc = d.mul_add(sc[usize::from(h & 63)], acc);
    }
    acc
}

fn nib_rows_scalar(t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], xs: &[f32], stride: usize, out: &mut [f32]) {
    let n = out.len();
    let mut acc = [0f32; MAX_BATCH];
    for (j, &h) in hdr.iter().enumerate() {
        let tab = &t.nibval[usize::from(h >> 6) & 1];
        let b = &nib[j * NIB_BYTES..(j + 1) * NIB_BYTES];
        let y: [f32; DIM] = std::array::from_fn(|k| nib_coord(tab, b, k));
        let s = sc[usize::from(h & 63)];
        for (r, a) in acc.iter_mut().enumerate().take(n) {
            let xb = &xs[r * stride + j * DIM..r * stride + (j + 1) * DIM];
            let mut d = 0f32;
            for (v, &xv) in y.iter().zip(xb) {
                d = v.mul_add(xv, d);
            }
            *a = d.mul_add(s, *a);
        }
    }
    out.copy_from_slice(&acc[..n]);
}

fn nib_row(isa: Isa, t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], x: &[f32]) -> f32 {
    match isa {
        Isa::Scalar => nib_row_scalar(t, nib, hdr, sc, x),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::nib_row(t, nib, hdr, sc, x),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::nib_row(t, nib, hdr, sc, x) },
    }
}

#[allow(clippy::too_many_arguments)]
fn nib_rows(isa: Isa, t: &Tables, nib: &[u8], hdr: &[u8], sc: &[f32; 64], xs: &[f32], stride: usize, out: &mut [f32]) {
    match isa {
        Isa::Scalar => nib_rows_scalar(t, nib, hdr, sc, xs, stride, out),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::nib_rows(t, nib, hdr, sc, xs, stride, out),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::nib_rows(t, nib, hdr, sc, xs, stride, out) },
    }
}

/// [`tetra_matvec`] over the nibbles of `m`: the same `y`, bit for bit.
pub fn tetra_nib_matvec(isa: Isa, t: &Tables, m: &TetraMatrix, nb: &TetraNib, xp: &[f32], y: &mut [f32]) {
    assert_eq!(xp.len(), m.d_in);
    assert_eq!(y.len(), m.d_out);
    assert!(nb.nblocks == m.nblocks && nb.hdr.len() == m.d_out * m.nblocks, "nibbles of another matrix");
    let sc = t.scale_table(&m.gscale);
    let c = chunk_rows(m.d_out);
    pool().install(|| {
        y.par_chunks_mut(c).enumerate().for_each(|(ci, ys)| {
            for (k, yo) in ys.iter_mut().enumerate() {
                let r = ci * c + k;
                let (nib, hdr) = nb.row(r);
                *yo = m.finish(isa, r, nib_row(isa, t, nib, hdr, &sc, xp), xp);
            }
        })
    });
}

/// [`tetra_matvec_rows`] over the nibbles of `m`: the same `Y`, bit for bit.
pub fn tetra_nib_matvec_rows(isa: Isa, t: &Tables, m: &TetraMatrix, nb: &TetraNib, xs: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(xs.len(), n * m.d_in);
    assert_eq!(y.len(), n * m.d_out);
    assert!(nb.nblocks == m.nblocks && nb.hdr.len() == m.d_out * m.nblocks, "nibbles of another matrix");
    let sc = t.scale_table(&m.gscale);
    let c = chunk_rows(m.d_out);
    let mut yt = vec![0f32; n * m.d_out];
    pool().install(|| {
        yt.par_chunks_mut(c * n).enumerate().for_each(|(ci, ys)| {
            for (k, out) in ys.chunks_mut(n).enumerate() {
                let r = ci * c + k;
                let (nib, hdr) = nb.row(r);
                let mut start = 0;
                while start < n {
                    let len = (n - start).min(MAX_BATCH);
                    let o = &mut out[start..start + len];
                    nib_rows(isa, t, nib, hdr, &sc, &xs[start * m.d_in..], m.d_in, o);
                    for (i, v) in o.iter_mut().enumerate() {
                        let x = &xs[(start + i) * m.d_in..(start + i + 1) * m.d_in];
                        *v = m.finish(isa, r, *v, x);
                    }
                    start += len;
                }
            }
        })
    });
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

/// One int4 row, the reference order: per group, the bias term then the
/// scaled sum, into one scalar.
fn q4_row_scalar(bytes: &[u8], scales: &[u16], biases: &[u16], half_g: usize, a: &Q4Act) -> f32 {
    let mut acc = 0f32;
    for (g, (&s, &b)) in scales.iter().zip(biases).enumerate() {
        let at = g * half_g;
        let d = q4_group_scalar(&bytes[at..at + half_g], &a.xe[at..at + half_g], &a.xo[at..at + half_g]);
        let (s, b) = (half::f16::from_bits(s).to_f32(), half::f16::from_bits(b).to_f32());
        acc = s.mul_add(d, b.mul_add(a.gsum[g], acc));
    }
    acc
}

fn q4_row(isa: Isa, m: &Q4Matrix, r: usize, a: &Q4Act) -> f32 {
    let (half_row, half_g, gpr) = (m.d_in / 2, m.group / 2, m.d_in / m.group);
    let bytes = &m.packed[r * half_row..(r + 1) * half_row];
    let (scales, biases) = (&m.scales[r * gpr..(r + 1) * gpr], &m.biases[r * gpr..(r + 1) * gpr]);
    // The SIMD rows step 16 bytes at a time; a group of another width takes
    // the scalar row.
    if !half_g.is_multiple_of(16) {
        return q4_row_scalar(bytes, scales, biases, half_g, a);
    }
    match isa {
        Isa::Scalar => q4_row_scalar(bytes, scales, biases, half_g, a),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::q4_row(bytes, scales, biases, half_g, a),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::q4_row(bytes, scales, biases, half_g, a) },
    }
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

// ---------------------------------------------------------------------------
// BF16 matrices, as a checkpoint stores them.
// ---------------------------------------------------------------------------

/// A BF16 value as f32, exactly: its 16 bits are the high half of the f32.
#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `s + Σ w·x` over the weights a SIMD row leaves, one fused multiply-add
/// each, in order.
#[inline(always)]
fn bf16_row_tail(w: &[u16], x: &[f32], mut s: f32) -> f32 {
    for (&b, &v) in w.iter().zip(x) {
        s = bf16_to_f32(b).mul_add(v, s);
    }
    s
}

/// One BF16 row, the scalar path: eight accumulators, each weight `k` on
/// accumulator `k mod 8`, summed in order, then the tail.
fn bf16_row_scalar(w: &[u16], x: &[f32]) -> f32 {
    assert_eq!(x.len(), w.len(), "a BF16 row of {} against {} values", w.len(), x.len());
    let main = w.len() - w.len() % 8;
    let mut acc = [0f32; 8];
    for (wc, xc) in w[..main].chunks_exact(8).zip(x[..main].chunks_exact(8)) {
        for ((a, &b), &v) in acc.iter_mut().zip(wc).zip(xc) {
            *a = bf16_to_f32(b).mul_add(v, *a);
        }
    }
    bf16_row_tail(&w[main..], &x[main..], acc.iter().sum())
}

fn bf16_row(isa: Isa, w: &[u16], x: &[f32]) -> f32 {
    match isa {
        Isa::Scalar => bf16_row_scalar(w, x),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::bf16_row(w, x),
        // SAFETY: as in `tetra_row`.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::bf16_row(w, x) },
    }
}

/// `y = W x` for one BF16 matrix `[d_out, d_in]`, row-major, and one f32
/// activation. The rows are split across the pool as the other matvecs do.
pub fn bf16_matvec(isa: Isa, w: &[u16], d_out: usize, d_in: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(w.len(), d_out * d_in, "{} weights for {d_out} x {d_in}", w.len());
    assert_eq!(x.len(), d_in);
    assert_eq!(y.len(), d_out);
    let c = chunk_rows(d_out);
    pool().install(|| {
        y.par_chunks_mut(c).enumerate().for_each(|(ci, ys)| {
            for (k, yo) in ys.iter_mut().enumerate() {
                let r = ci * c + k;
                *yo = bf16_row(isa, &w[r * d_in..(r + 1) * d_in], x);
            }
        })
    });
}

/// `Y = W X` for `n` activations `[n, d_in]` into `[n, d_out]`. Each weight
/// row is read once from memory and stays in the first-level cache while the
/// `n` activations pass over it; each output is the row of [`bf16_matvec`],
/// so a row inside a batch equals the same row alone, bit for bit.
pub fn bf16_matvec_rows(isa: Isa, w: &[u16], d_out: usize, d_in: usize, xs: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(w.len(), d_out * d_in, "{} weights for {d_out} x {d_in}", w.len());
    assert_eq!(xs.len(), n * d_in);
    assert_eq!(y.len(), n * d_out);
    let c = chunk_rows(d_out);
    let mut yt = vec![0f32; n * d_out];
    pool().install(|| {
        yt.par_chunks_mut(c * n).enumerate().for_each(|(ci, ys)| {
            for (k, out) in ys.chunks_mut(n).enumerate() {
                let r = ci * c + k;
                let row = &w[r * d_in..(r + 1) * d_in];
                for (o, x) in out.iter_mut().zip(xs.chunks(d_in)) {
                    *o = bf16_row(isa, row, x);
                }
            }
        })
    });
    for r in 0..d_out {
        for i in 0..n {
            y[i * d_out + r] = yt[r * n + i];
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
