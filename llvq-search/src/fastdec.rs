//! Fast decoding of format v1 — no u128, no factorials, no allocations.
//!
//! Promoted from the `decfull` bench (243 ns/block, 3.3× `Indexer::decode`)
//! once the load-time transcoder needed it as a library: the runtime layout
//! is produced by decoding every archive block once, and 151 M blocks at
//! 869 ns is a minute of load time where 243 ns is seconds.
//!
//! Two things live here:
//!
//! * [`FastDecoder::decode`] — same bits, same points as [`Indexer::decode`],
//!   verified bit-for-bit on every class boundary plus 200 000 uniform draws
//!   (`tests/fastdec.rs`).
//! * [`ClassLevels`] — the *class-level* facts a runtime format needs. A class
//!   is a fixed multiset of |values|, so the number of magnitude levels, their
//!   counts, the zero count and the coset are all functions of the class
//!   alone, never of the block. Anything accounted per class is exact over
//!   every block of that class.
//!
//! ## The u64 trap
//!
//! Multinomial *values* fit u64 (they divide the 48-bit index budget), but
//! 21!..24! overflow it. The unrank below never touches a factorial: the
//! initial multinomial comes from the class table and is maintained by the
//! exact recurrence `M' = M·c_j/n`.
//!
//! [`Indexer::decode`]: crate::index::Indexer::decode

use crate::classes::{enumerate_classes, gamma, ClassSet, MAX_SHELL};
use crate::index::N13;
use llvq_core::{Golay, Point, DIM};

/// Most distinct |values| any class of the m ≤ 13 ball can hold.
///
/// ‖x‖² = 16m ≤ 208 caps it: five distinct magnitudes cost at least
/// 0²+2²+4²+6²+10² = 156 (even) or forbid a sixth (adding one more distinct
/// value always exceeds 208). Asserted over all 383 classes in
/// [`FastDecoder::new`], not just claimed here.
pub const MAX_LEVELS: usize = 5;

const MAXK: usize = 8;

/// The magnitude structure of one class, as a runtime format sees it.
///
/// Levels are ordered **descending count, ties by descending |value|** — the
/// canonical order shared by the transcoder, the runtime tables and any
/// kernel, chosen so nested-mask layouts spend their widest mask on the
/// most-populated level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassLevels {
    /// Distinct |values|, canonical order. Slots ≥ `len` are zero.
    pub values: [i32; MAX_LEVELS],
    /// Coordinate count per level; sums to 24 over `len` slots.
    pub counts: [u8; MAX_LEVELS],
    /// Number of levels actually used.
    pub len: usize,
    /// Nonzero coordinates — the ones that carry a sign.
    pub nonzero: u8,
    /// Odd coset (all coordinates odd) or even.
    pub odd: bool,
    /// Shell index m, ‖x‖² = 16m.
    pub shell: u32,
}

/// Flat, fixed-size mirror of one class of `Indexer`'s layout.
#[derive(Clone, Copy, Default)]
struct FastClass {
    offset: u64,
    odd: bool,
    // Even side.
    w: u32,
    p_req: u32,
    word_vals: [i32; 4],
    word_counts: [u8; 4],
    n_word: usize,
    free_vals: [i32; 4],
    off_counts: [u8; MAXK],
    n_off: usize,
    a_on: u64,
    a_off: u64,
    s_w: u64,
    s_f: u64,
    // Odd side.
    vals: [i32; MAXK],
    counts: [u8; MAXK],
    n_kinds: usize,
    m_arr: u64,
}

fn multinomial_u64(counts: &[u8]) -> u64 {
    let mut fact = [1u128; 25];
    for i in 1..25 {
        fact[i] = fact[i - 1] * i as u128;
    }
    let n: usize = counts.iter().map(|&c| c as usize).sum();
    let mut m = fact[n];
    for &c in counts {
        m /= fact[c as usize];
    }
    u64::try_from(m).expect("arrangement count fits u64")
}

/// One multiply and one divide per candidate — `M' = M·c_j/n`, exact.
#[inline]
fn unrank_fast(mut rank: u64, counts: &mut [u8; MAXK], k: usize, m0: u64, out: &mut [u8]) {
    let mut m = m0;
    let mut n = out.len() as u64;
    for slot in out.iter_mut() {
        #[allow(clippy::needless_range_loop)] // `j` is written into `*slot`
        for j in 0..k {
            let c = counts[j] as u64;
            if c == 0 {
                continue;
            }
            let mj = m * c / n;
            if rank < mj {
                *slot = j as u8;
                counts[j] -= 1;
                m = mj;
                break;
            }
            rank -= mj;
        }
        n -= 1;
    }
}

/// Decoder for the ball Λ₂₄(13) ∪ {0}, format v1.
pub struct FastDecoder {
    classes: Vec<FastClass>,
    /// `offset + cardinality` per class, for the binary search.
    ends: Vec<u64>,
    levels: Vec<ClassLevels>,
    golay: Golay,
}

impl FastDecoder {
    /// Mirror of `Indexer::new`'s layout: shells ascending, even classes then
    /// odd classes, enumeration order within each. Any divergence fails the
    /// exhaustive boundary check of `tests/fastdec.rs`.
    pub fn new() -> Self {
        let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
        let mut classes: Vec<FastClass> = Vec::new();
        let mut levels: Vec<ClassLevels> = Vec::new();
        let mut cards: Vec<u64> = Vec::new();
        for m in 2..=MAX_SHELL {
            for c in even.iter().filter(|c| c.shell == m) {
                let mut fc = FastClass {
                    w: c.w,
                    p_req: c.p_req as u32,
                    ..Default::default()
                };
                fc.n_word = c.word_vals.len();
                for (i, &(v, n)) in c.word_vals.iter().enumerate() {
                    fc.word_vals[i] = v as i32;
                    fc.word_counts[i] = n;
                }
                let free_n: u32 = c.free_vals.iter().map(|&(_, n)| n as u32).sum();
                for (i, &(v, n)) in c.free_vals.iter().enumerate() {
                    fc.free_vals[i] = v as i32;
                    fc.off_counts[i] = n;
                }
                let n0 = (24 - c.w - free_n) as u8;
                fc.off_counts[c.free_vals.len()] = n0;
                fc.n_off = c.free_vals.len() + 1;
                fc.a_on = multinomial_u64(&fc.word_counts[..fc.n_word]);
                fc.a_off = multinomial_u64(&fc.off_counts[..fc.n_off]);
                fc.s_w = if c.w == 0 { 1 } else { 1u64 << (c.w - 1) };
                fc.s_f = 1u64 << free_n;
                cards.push(gamma(c.w) * fc.a_on * fc.a_off * fc.s_w * fc.s_f);

                let mut lv: Vec<(i32, u8)> = c
                    .word_vals
                    .iter()
                    .chain(c.free_vals.iter())
                    .map(|&(v, n)| (v as i32, n))
                    .collect();
                if n0 > 0 {
                    lv.push((0, n0));
                }
                levels.push(build_levels(&lv, false, m));
                classes.push(fc);
            }
            for c in odd.iter().filter(|c| c.shell == m) {
                let mut fc = FastClass {
                    odd: true,
                    ..Default::default()
                };
                fc.n_kinds = c.vals.len();
                for (i, &(v, n)) in c.vals.iter().enumerate() {
                    fc.vals[i] = v as i32;
                    fc.counts[i] = n;
                }
                fc.m_arr = multinomial_u64(&fc.counts[..fc.n_kinds]);
                cards.push(4096 * fc.m_arr);

                let lv: Vec<(i32, u8)> =
                    c.vals.iter().map(|&(v, n)| (v as i32, n)).collect();
                levels.push(build_levels(&lv, true, m));
                classes.push(fc);
            }
        }
        let mut off = 0u64;
        let mut ends = Vec::with_capacity(classes.len());
        for (fc, card) in classes.iter_mut().zip(&cards) {
            fc.offset = off;
            off += card;
            ends.push(off);
        }
        assert_eq!(off, N13, "class layout must cover exactly N(13)");
        Self {
            classes,
            ends,
            levels,
            golay: Golay::new(),
        }
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Class holding index `idx`, or `None` for the origin (0) and anything
    /// past `N13`.
    pub fn class_of(&self, idx: u64) -> Option<usize> {
        if idx == 0 || idx > N13 {
            return None;
        }
        Some(self.ends.partition_point(|&e| e < idx))
    }

    /// First and last index of class `ci`, in `decode`'s convention.
    pub fn class_range(&self, ci: usize) -> (u64, u64) {
        (self.classes[ci].offset + 1, self.ends[ci])
    }

    /// Magnitude structure of class `ci`.
    pub fn levels(&self, ci: usize) -> &ClassLevels {
        &self.levels[ci]
    }

    /// Decode an index in `0 ..= N13` back to its codebook point.
    /// Returns `None` for out-of-range indices.
    pub fn decode(&self, idx: u64) -> Option<Point> {
        if idx == 0 {
            return Some([0; DIM]);
        }
        let i = idx - 1;
        if i >= N13 {
            return None;
        }
        let ci = self.ends.partition_point(|&e| e <= i);
        let fc = &self.classes[ci];
        let mut local = i - fc.offset;
        let mut p = [0i32; DIM];

        if fc.odd {
            let r_arr = local % fc.m_arr;
            let gi = (local / fc.m_arr) as usize;
            let c = self.golay.codewords()[gi];
            let mut counts = fc.counts;
            let mut kinds = [0u8; DIM];
            unrank_fast(r_arr, &mut counts, fc.n_kinds, fc.m_arr, &mut kinds);
            for (i, pi) in p.iter_mut().enumerate() {
                let v = fc.vals[kinds[i] as usize];
                // Forced signs: value ≡ 3 (mod 4) is positive on the support,
                // negative off it; ≡ 1 (mod 4) the opposite.
                let a = if v % 4 == 3 { v } else { -v };
                *pi = if c >> i & 1 == 1 { a } else { -a };
            }
        } else {
            let r_sf = local % fc.s_f;
            local /= fc.s_f;
            let r_sw = local % fc.s_w;
            local /= fc.s_w;
            let r_off = local % fc.a_off;
            local /= fc.a_off;
            let r_on = local % fc.a_on;
            let gi = (local / fc.a_on) as usize;
            let c = self.golay.of_weight(fc.w as usize)[gi];
            let w = fc.w as usize;

            let mut wc = [0u8; MAXK];
            wc[..4].copy_from_slice(&fc.word_counts);
            let mut on_kinds = [0u8; DIM];
            unrank_fast(r_on, &mut wc, fc.n_word, fc.a_on, &mut on_kinds[..w]);

            let mut oc = fc.off_counts;
            let mut off_kinds = [0u8; DIM];
            unrank_fast(r_off, &mut oc, fc.n_off, fc.a_off, &mut off_kinds[..DIM - w]);

            let mut sign_bits = [0u32; DIM];
            let mut par = 0u32;
            for (j, sb) in sign_bits.iter_mut().enumerate().take(w.saturating_sub(1)) {
                let b = (r_sw >> j & 1) as u32;
                *sb = b;
                par ^= b;
            }
            if w > 0 {
                sign_bits[w - 1] = par ^ fc.p_req;
            }

            let zero_kind = (fc.n_off - 1) as u8;
            let (mut s_on, mut s_off, mut fbit) = (0usize, 0usize, 0u32);
            for (i, pi) in p.iter_mut().enumerate() {
                if c >> i & 1 == 1 {
                    let v = fc.word_vals[on_kinds[s_on] as usize];
                    *pi = if sign_bits[s_on] == 1 { -v } else { v };
                    s_on += 1;
                } else {
                    let k = off_kinds[s_off];
                    if k != zero_kind {
                        let v = fc.free_vals[k as usize];
                        *pi = if r_sf >> fbit & 1 == 1 { -v } else { v };
                        fbit += 1;
                    }
                    s_off += 1;
                }
            }
        }
        Some(p)
    }
}

impl Default for FastDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical level order: descending count, ties by descending |value|.
fn build_levels(kinds: &[(i32, u8)], odd: bool, shell: u32) -> ClassLevels {
    let mut lv = kinds.to_vec();
    lv.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    assert!(
        lv.len() <= MAX_LEVELS,
        "class exceeds {MAX_LEVELS} magnitude levels: {lv:?}"
    );
    let mut out = ClassLevels {
        len: lv.len(),
        odd,
        shell,
        ..Default::default()
    };
    let mut total = 0u32;
    for (i, &(v, n)) in lv.iter().enumerate() {
        out.values[i] = v;
        out.counts[i] = n;
        total += n as u32;
        if v != 0 {
            out.nonzero += n;
        }
    }
    assert_eq!(total, DIM as u32, "level counts must cover the block");
    out
}
