//! Bijective indexing of the ball Λ₂₄(13) ∪ {0} — Phase 3, gate G3.
//!
//! Maps every codebook point to a unique integer `0 ..= N(13)`
//! (`N(13) + 1 = 280,974,212,784,721 ≤ 2⁴⁸` — a 48-bit code per 24-weight
//! block, i.e. exactly the paper's 2 bit-per-weight regime) **without
//! materializing the codebook**, following the hierarchy of §3.2 / App. A
//! of the paper: shells by increasing radius, classes in a fixed order
//! within each shell, and inside a class the local degrees of freedom
//! (codeword, value arrangement, sign patterns) linearized by standard
//! mixed-radix composition.
//!
//! ## Format v1 — stability contract
//!
//! The index format is determined by, and only by:
//! 1. the Golay generator `0xC75` and the codeword order of
//!    [`Golay::codewords`] (weight-major, ascending within a weight);
//! 2. the class enumeration order of [`enumerate_classes`] (deterministic
//!    nested loops), laid out shells ascending, even classes before odd
//!    classes within a shell;
//! 3. the composition orders documented on [`Indexer::encode`].
//!
//! Any change to these is a breaking format change.
//!
//! ## Local index composition
//!
//! * **Even class** (Golay weight w, γ codewords, word/free value kinds):
//!   `local = (((g·A_on + r_on)·A_off + r_off)·S_w + r_sw)·S_f + r_sf`
//!   with `g` the codeword rank within its weight bucket, `r_on`/`r_off`
//!   multiset-permutation ranks of the word values on the support and the
//!   free values + zeros off it (positions ascending), `r_sw` the word sign
//!   pattern (w−1 free bits, the last support position's sign fixed by the
//!   parity constraint), `r_sf` the free sign bits.
//! * **Odd class**: `local = g₄₀₉₆·M + r_arr` with `g₄₀₉₆` the flat Golay
//!   rank (all 4096 codewords are admissible), `r_arr` the
//!   multiset-permutation rank of the |values| over the 24 positions;
//!   signs carry no information (forced by membership).

use crate::classes::{enumerate_classes, multinomial, ClassSet, MAX_SHELL};
use llvq_core::{Golay, Leech, Point, DIM};

/// `N(13)` — codebook size excluding the origin (paper Table 1).
pub const N13: u64 = llvq_core::leech::N_SHELL_13_CUMULATIVE;

enum Kind {
    Even {
        /// Word value kinds `(value, count)`, descending values.
        word: Vec<(i32, u8)>,
        /// Free value kinds `(value, count)`, descending values.
        free: Vec<(i32, u8)>,
        w: u32,
        p_req: u32,
        /// Off-support kind counts: free kinds then the zero kind.
        off_counts: Vec<u8>,
        a_on: u128,
        a_off: u128,
        s_w: u128,
        s_f: u128,
    },
    Odd {
        /// Value kinds `(value, count)`, descending values; Σ counts = 24.
        vals: Vec<(i32, u8)>,
        m_arr: u128,
    },
}

struct ClassIx {
    shell: u32,
    offset: u64,
    card: u64,
    kind: Kind,
}

/// Encoder/decoder for the ball Λ₂₄(13) ∪ {0}.
pub struct Indexer {
    classes: Vec<ClassIx>,
    leech: Leech,
}

/// Multiset-permutation rank of `kind_at` (kind index per slot) under the
/// remaining-counts convention: kinds ordered by their index.
fn perm_rank(kind_at: &[u8], counts: &mut [u8]) -> u128 {
    let mut rank = 0u128;
    for &k in kind_at {
        for j in 0..k {
            if counts[j as usize] > 0 {
                counts[j as usize] -= 1;
                rank += multinomial(counts);
                counts[j as usize] += 1;
            }
        }
        counts[k as usize] -= 1;
    }
    rank
}

/// Inverse of [`perm_rank`]: writes the kind index per slot into `out`.
fn perm_unrank(mut rank: u128, counts: &mut [u8], out: &mut [u8]) {
    for slot in out.iter_mut() {
        let mut placed = false;
        for j in 0..counts.len() {
            if counts[j] == 0 {
                continue;
            }
            counts[j] -= 1;
            let m = multinomial(counts);
            if rank < m {
                *slot = j as u8;
                placed = true;
                break;
            }
            rank -= m;
            counts[j] += 1;
        }
        debug_assert!(placed, "rank out of range for the multiset");
    }
    debug_assert_eq!(rank, 0);
}

impl Indexer {
    pub fn new() -> Self {
        let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
        let mut classes = Vec::new();
        // Format order: shells ascending, even classes then odd classes,
        // each in enumeration order.
        for m in 2..=MAX_SHELL {
            for c in even.iter().filter(|c| c.shell == m) {
                let word: Vec<(i32, u8)> =
                    c.word_vals.iter().map(|&(v, n)| (v as i32, n)).collect();
                let free: Vec<(i32, u8)> =
                    c.free_vals.iter().map(|&(v, n)| (v as i32, n)).collect();
                let free_n: u32 = free.iter().map(|&(_, n)| n as u32).sum();
                let n0 = 24 - c.w - free_n;
                let mut off_counts: Vec<u8> = free.iter().map(|&(_, n)| n).collect();
                off_counts.push(n0 as u8);
                let word_counts: Vec<u8> = word.iter().map(|&(_, n)| n).collect();
                let a_on = multinomial(&word_counts);
                let a_off = multinomial(&off_counts);
                let s_w: u128 = if c.w == 0 { 1 } else { 1 << (c.w - 1) };
                let s_f: u128 = 1 << free_n;
                let card = crate::classes::gamma(c.w) as u128 * a_on * a_off * s_w * s_f;
                classes.push(ClassIx {
                    shell: m,
                    offset: 0,
                    card: u64::try_from(card).expect("class fits u64"),
                    kind: Kind::Even {
                        word,
                        free,
                        w: c.w,
                        p_req: c.p_req as u32,
                        off_counts,
                        a_on,
                        a_off,
                        s_w,
                        s_f,
                    },
                });
            }
            for c in odd.iter().filter(|c| c.shell == m) {
                let vals: Vec<(i32, u8)> =
                    c.vals.iter().map(|&(v, n)| (v as i32, n)).collect();
                let counts: Vec<u8> = vals.iter().map(|&(_, n)| n).collect();
                let m_arr = multinomial(&counts);
                classes.push(ClassIx {
                    shell: m,
                    offset: 0,
                    card: u64::try_from(4096 * m_arr).expect("class fits u64"),
                    kind: Kind::Odd { vals, m_arr },
                });
            }
        }
        let mut off = 0u64;
        for c in classes.iter_mut() {
            c.offset = off;
            off += c.card;
        }
        assert_eq!(off, N13, "class layout must cover exactly N(13) points");
        Self {
            classes,
            leech: Leech::new(),
        }
    }

    fn golay(&self) -> &Golay {
        self.leech.golay()
    }

    /// Encode a codebook point to its index in `0 ..= N13`
    /// (0 = origin). Returns `None` for any `x` outside the ball —
    /// including non-lattice vectors (full membership is verified).
    pub fn encode(&self, x: &Point) -> Option<u64> {
        if *x == [0; DIM] {
            return Some(0);
        }
        if !self.leech.contains(x) {
            return None;
        }
        let m = match Leech::shell_index(x) {
            Some(m) if (2..=MAX_SHELL as u64).contains(&m) => m as u32,
            _ => return None,
        };
        let parity = x[0].rem_euclid(2);

        // Codeword from the mod-4 pattern (as in the membership predicate).
        let mut c = 0u32;
        for (i, &v) in x.iter().enumerate() {
            if ((v - parity) >> 1).rem_euclid(2) == 1 {
                c |= 1 << i;
            }
        }

        // Class identification by |value| signature.
        let mut counts_by_val = [0u8; 16];
        for &v in x.iter() {
            counts_by_val[v.unsigned_abs() as usize] += 1;
        }

        if parity == 0 {
            let cls = self.classes.iter().find(|cl| {
                cl.shell == m
                    && match &cl.kind {
                        Kind::Even { word, free, .. } => {
                            let mut sig = [0u8; 16];
                            for &(v, n) in word.iter().chain(free.iter()) {
                                sig[v as usize] = n;
                            }
                            sig[0] = counts_by_val[0];
                            sig == counts_by_val
                        }
                        Kind::Odd { .. } => false,
                    }
            })?;
            let Kind::Even {
                word,
                free,
                w,
                p_req,
                a_on,
                a_off,
                s_w,
                s_f,
                ..
            } = &cls.kind
            else {
                unreachable!()
            };
            let g = self.golay().rank_in_weight(c)? as u128;

            // On-support arrangement + word signs (positions ascending).
            let mut on_kinds = Vec::with_capacity(*w as usize);
            let mut sign_bits = Vec::with_capacity(*w as usize);
            for (i, &xi) in x.iter().enumerate() {
                if c >> i & 1 == 1 {
                    let av = xi.unsigned_abs() as i32;
                    let k = word.iter().position(|&(v, _)| v == av)?;
                    on_kinds.push(k as u8);
                    sign_bits.push((xi < 0) as u32);
                }
            }
            debug_assert_eq!(
                sign_bits.iter().sum::<u32>() % 2,
                *p_req,
                "member must satisfy the word sign parity"
            );
            let mut wc: Vec<u8> = word.iter().map(|&(_, n)| n).collect();
            let r_on = perm_rank(&on_kinds, &mut wc);
            let r_sw: u128 = sign_bits
                .iter()
                .take(w.saturating_sub(1) as usize)
                .enumerate()
                .map(|(j, &b)| (b as u128) << j)
                .sum();

            // Off-support arrangement + free signs.
            let zero_kind = free.len() as u8;
            let mut off_kinds = Vec::with_capacity(DIM - *w as usize);
            let mut free_bits = Vec::new();
            for (i, &xi) in x.iter().enumerate() {
                if c >> i & 1 == 0 {
                    if xi == 0 {
                        off_kinds.push(zero_kind);
                    } else {
                        let av = xi.unsigned_abs() as i32;
                        let k = free.iter().position(|&(v, _)| v == av)?;
                        off_kinds.push(k as u8);
                        free_bits.push((xi < 0) as u128);
                    }
                }
            }
            let Kind::Even { off_counts, .. } = &cls.kind else {
                unreachable!()
            };
            let mut oc = off_counts.clone();
            let r_off = perm_rank(&off_kinds, &mut oc);
            let r_sf: u128 = free_bits
                .iter()
                .enumerate()
                .map(|(j, &b)| b << j)
                .sum();

            let local = (((g * a_on + r_on) * a_off + r_off) * s_w + r_sw) * s_f + r_sf;
            Some(1 + cls.offset + u64::try_from(local).expect("local fits u64"))
        } else {
            let cls = self.classes.iter().find(|cl| {
                cl.shell == m
                    && match &cl.kind {
                        Kind::Odd { vals, .. } => {
                            let mut sig = [0u8; 16];
                            for &(v, n) in vals.iter() {
                                sig[v as usize] = n;
                            }
                            sig == counts_by_val
                        }
                        Kind::Even { .. } => false,
                    }
            })?;
            let Kind::Odd { vals, m_arr } = &cls.kind else {
                unreachable!()
            };
            let g = self.golay().rank(c)? as u128;
            let mut kinds_at = [0u8; DIM];
            for (i, &v) in x.iter().enumerate() {
                let av = v.unsigned_abs() as i32;
                kinds_at[i] = vals.iter().position(|&(u, _)| u == av)? as u8;
            }
            let mut counts: Vec<u8> = vals.iter().map(|&(_, n)| n).collect();
            let r_arr = perm_rank(&kinds_at, &mut counts);
            let local = g * m_arr + r_arr;
            Some(1 + cls.offset + u64::try_from(local).expect("local fits u64"))
        }
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
        let ci = self
            .classes
            .partition_point(|c| c.offset + c.card <= i);
        let cls = &self.classes[ci];
        let mut local = (i - cls.offset) as u128;
        let mut p: Point = [0; DIM];

        match &cls.kind {
            Kind::Even {
                word,
                free,
                w,
                p_req,
                off_counts,
                a_on,
                a_off,
                s_w,
                s_f,
            } => {
                let r_sf = local % s_f;
                local /= s_f;
                let r_sw = local % s_w;
                local /= s_w;
                let r_off = local % a_off;
                local /= a_off;
                let r_on = local % a_on;
                let g = (local / a_on) as usize;
                let c = self.golay().of_weight(*w as usize)[g];

                // On-support values and signs.
                let mut wc: Vec<u8> = word.iter().map(|&(_, n)| n).collect();
                let mut on_kinds = vec![0u8; *w as usize];
                perm_unrank(r_on, &mut wc, &mut on_kinds);
                let mut sign_bits = vec![0u32; *w as usize];
                let mut par = 0u32;
                for (j, sb) in sign_bits
                    .iter_mut()
                    .enumerate()
                    .take(w.saturating_sub(1) as usize)
                {
                    let b = (r_sw >> j & 1) as u32;
                    *sb = b;
                    par ^= b;
                }
                if *w > 0 {
                    sign_bits[*w as usize - 1] = par ^ p_req;
                }
                let mut slot = 0;
                for (i, pi) in p.iter_mut().enumerate() {
                    if c >> i & 1 == 1 {
                        let v = word[on_kinds[slot] as usize].0;
                        *pi = if sign_bits[slot] == 1 { -v } else { v };
                        slot += 1;
                    }
                }

                // Off-support values and signs.
                let mut oc = off_counts.clone();
                let mut off_kinds = vec![0u8; DIM - *w as usize];
                perm_unrank(r_off, &mut oc, &mut off_kinds);
                let zero_kind = free.len() as u8;
                let mut slot = 0;
                let mut fbit = 0;
                for (i, pi) in p.iter_mut().enumerate() {
                    if c >> i & 1 == 0 {
                        let k = off_kinds[slot];
                        if k != zero_kind {
                            let v = free[k as usize].0;
                            *pi = if r_sf >> fbit & 1 == 1 { -v } else { v };
                            fbit += 1;
                        }
                        slot += 1;
                    }
                }
            }
            Kind::Odd { vals, m_arr } => {
                let r_arr = local % m_arr;
                let g = (local / m_arr) as usize;
                let c = self.golay().codewords()[g];
                let mut counts: Vec<u8> = vals.iter().map(|&(_, n)| n).collect();
                let mut kinds_at = [0u8; DIM];
                perm_unrank(r_arr, &mut counts, &mut kinds_at);
                for (i, pi) in p.iter_mut().enumerate() {
                    let v = vals[kinds_at[i] as usize].0;
                    // Forced signs: value ≡ 3 (mod 4) is positive on the
                    // support, negative off it; ≡ 1 (mod 4) the opposite.
                    let a = if v % 4 == 3 { v } else { -v };
                    *pi = if c >> i & 1 == 1 { a } else { -a };
                }
            }
        }
        debug_assert!(self.leech.contains(&p), "decoded point must be a member");
        Some(p)
    }
}

impl Default for Indexer {
    fn default() -> Self {
        Self::new()
    }
}
