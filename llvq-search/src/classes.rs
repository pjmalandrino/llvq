//! Programmatic enumeration of the equivalence classes of Leech shells
//! (paper §2.4–2.5), for shells m ≤ 13 — the 2 bit/dim regime.
//!
//! A class = (parity coset, multiset of absolute coordinate values). The
//! constraints, derived from the coset conditions (Eq. 5) and validated
//! against every hand-derived count of the G1 suite plus the theta series:
//!
//! **Even classes** (values in {0, 2, 4, …}): positions carrying values
//! ≡ 2 (mod 4) — the *word values* {2, 6, 10, 14} — must form a Golay
//! codeword, so their total count w must be a Golay weight; values
//! ≡ 0 (mod 4) — the *free values* {4, 8, 12} — sit off the support with
//! free signs (a flip shifts Σ by 0 mod 8). Word-value flips shift Σ by
//! 4 (mod 8), so the number of minus signs among word values has fixed
//! parity `p_req = (S₀/4) mod 2` (S₀ = sum of all values; S₀ ≡ 0 mod 4
//! always since w is even). A w = 0 class exists only if `p_req = 0`.
//!
//! **Odd classes** (values in {1, 3, 5, …}): signs are *forced* by the
//! codeword — positions in c carry signed values ≡ 3 (mod 4), positions
//! off c carry ≡ 1 (mod 4). The residual sum condition reduces (mod 2,
//! membership signs cancel) to the **class-level** condition
//! `#{values ≡ ±1 mod 8} = n₁+n₇+n₉ odd`; every codeword and every
//! arrangement is then valid.
//!
//! Cardinalities (validated: shells 2–5 against the theta series, and the
//! cumulative sum over m ≤ 13 against the paper's Table 1 `N(13)`):
//!
//! * even: `γ(w) · 2^(w−1 if w>0 else 0) · 2^(Σ free counts) ·
//!   w!/Π(word counts)! · (24−w)!/(n₀!·Π(free counts)!)`
//! * odd: `4096 · 24!/Π(counts)!`

/// Highest supported shell (‖x‖² = 16·13 = 208, exactly 2.000 bits/dim).
pub const MAX_SHELL: u32 = 13;

/// Golay codeword count per weight (0 for non-weights).
pub fn gamma(w: u32) -> u64 {
    match w {
        0 => 1,
        8 => 759,
        12 => 2576,
        16 => 759,
        24 => 1,
        _ => 0,
    }
}

/// An even-coset class.
#[derive(Clone, Debug)]
pub struct EvenClass {
    /// Shell index m (‖x‖² = 16m).
    pub shell: u32,
    /// Word values `(value, count)`, value ≡ 2 (mod 4), descending.
    pub word_vals: Vec<(u8, u8)>,
    /// Free values `(value, count)`, value ≡ 0 (mod 4) and > 0, descending.
    pub free_vals: Vec<(u8, u8)>,
    /// Golay weight `w = Σ word counts`.
    pub w: u32,
    /// Required minus-sign parity among word values.
    pub p_req: u8,
}

/// An odd-coset class (all 24 coordinates odd).
#[derive(Clone, Debug)]
pub struct OddClass {
    /// Shell index m.
    pub shell: u32,
    /// Absolute values `(value, count)`, descending; Σ counts = 24.
    pub vals: Vec<(u8, u8)>,
}

/// All classes of shells 2..=max_shell.
pub struct ClassSet {
    pub even: Vec<EvenClass>,
    pub odd: Vec<OddClass>,
}

pub(crate) const FACT: [u128; 25] = {
    let mut f = [1u128; 25];
    let mut i = 1;
    while i < 25 {
        f[i] = f[i - 1] * i as u128;
        i += 1;
    }
    f
};

/// `n! / Π counts[i]!` — the number of distinct arrangements of a multiset
/// with the given kind counts over `n = Σ counts` slots (n ≤ 24).
pub(crate) fn multinomial(counts: &[u8]) -> u128 {
    let n: usize = counts.iter().map(|&c| c as usize).sum();
    debug_assert!(n <= 24);
    let mut m = FACT[n];
    for &c in counts {
        m /= FACT[c as usize];
    }
    m
}

impl EvenClass {
    /// Distinct magnitudes a block of this class shows, **zero included** —
    /// `{6, 4, 2, 0}` is four. Zero counts because the runtime layout spends
    /// a mask on it like any other level, so this is the number its width
    /// `34 + 24(L−1)` is a function of.
    pub fn n_levels(&self) -> usize {
        let free_n: u32 = self.free_vals.iter().map(|&(_, n)| n as u32).sum();
        let n0 = 24 - self.w - free_n;
        self.word_vals.len() + self.free_vals.len() + usize::from(n0 > 0)
    }

    pub fn cardinality(&self) -> u64 {
        let free_n: u32 = self.free_vals.iter().map(|&(_, n)| n as u32).sum();
        let n0 = 24 - self.w - free_n;
        let sign_word: u128 = if self.w == 0 { 1 } else { 1 << (self.w - 1) };
        let sign_free: u128 = 1 << free_n;
        let mut arr_on = FACT[self.w as usize];
        for &(_, n) in &self.word_vals {
            arr_on /= FACT[n as usize];
        }
        let mut arr_off = FACT[(24 - self.w) as usize] / FACT[n0 as usize];
        for &(_, n) in &self.free_vals {
            arr_off /= FACT[n as usize];
        }
        let total = gamma(self.w) as u128 * sign_word * sign_free * arr_on * arr_off;
        u64::try_from(total).expect("class cardinality fits u64")
    }
}

impl OddClass {
    /// Distinct magnitudes; an odd class never contains a zero.
    pub fn n_levels(&self) -> usize {
        self.vals.len()
    }

    pub fn cardinality(&self) -> u64 {
        let mut arr = FACT[24];
        for &(_, n) in &self.vals {
            arr /= FACT[n as usize];
        }
        u64::try_from(4096 * arr).expect("class cardinality fits u64")
    }
}

/// Enumerate every class of shells `2..=max_shell` (m = 1 provably yields
/// none: the even candidates fail Golay/parity, the odd norm floor is 24).
pub fn enumerate_classes(max_shell: u32) -> ClassSet {
    assert!(max_shell <= MAX_SHELL);
    let max_norm = 16 * max_shell;
    let mut even = Vec::new();
    let mut odd = Vec::new();

    // Even: word values 14, 10, 6, 2 — free values 12, 8, 4.
    for n14 in 0..=(max_norm / 196) {
        for n10 in 0..=(max_norm / 100) {
            for n6 in 0..=(max_norm / 36) {
                for n2 in 0..=(max_norm / 4) {
                    let w = n14 + n10 + n6 + n2;
                    if w > 24 || gamma(w) == 0 {
                        continue;
                    }
                    let base = 196 * n14 + 100 * n10 + 36 * n6 + 4 * n2;
                    if base > max_norm {
                        continue;
                    }
                    for n12 in 0..=((max_norm - base) / 144) {
                        for n8 in 0..=((max_norm - base) / 64) {
                            for n4 in 0..=((max_norm - base) / 16) {
                                let norm = base + 144 * n12 + 64 * n8 + 16 * n4;
                                let total = w + n12 + n8 + n4;
                                if norm > max_norm || total > 24 || norm % 16 != 0 {
                                    continue;
                                }
                                let m = norm / 16;
                                if !(2..=max_shell).contains(&m) {
                                    continue;
                                }
                                let s0 = 2 * n2 + 6 * n6 + 10 * n10 + 14 * n14
                                    + 4 * n4
                                    + 8 * n8
                                    + 12 * n12;
                                debug_assert_eq!(s0 % 4, 0);
                                let p_req = ((s0 / 4) % 2) as u8;
                                if w == 0 && p_req != 0 {
                                    continue; // no word signs available to fix parity
                                }
                                let word_vals: Vec<(u8, u8)> = [(14, n14), (10, n10), (6, n6), (2, n2)]
                                    .into_iter()
                                    .filter(|&(_, n)| n > 0)
                                    .map(|(v, n)| (v as u8, n as u8))
                                    .collect();
                                let free_vals: Vec<(u8, u8)> = [(12, n12), (8, n8), (4, n4)]
                                    .into_iter()
                                    .filter(|&(_, n)| n > 0)
                                    .map(|(v, n)| (v as u8, n as u8))
                                    .collect();
                                even.push(EvenClass {
                                    shell: m,
                                    word_vals,
                                    free_vals,
                                    w,
                                    p_req,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Odd: values 13, 11, 9, 7, 5, 3, 1 with Σ counts = 24.
    for n13 in 0..=(max_norm / 169) {
        for n11 in 0..=(max_norm / 121) {
            for n9 in 0..=(max_norm / 81) {
                for n7 in 0..=(max_norm / 49) {
                    for n5 in 0..=(max_norm / 25) {
                        let heavy = n13 + n11 + n9 + n7 + n5;
                        if heavy > 24 {
                            continue;
                        }
                        let hnorm =
                            169 * n13 + 121 * n11 + 81 * n9 + 49 * n7 + 25 * n5;
                        if hnorm > max_norm {
                            continue;
                        }
                        for n3 in 0..=(24 - heavy) {
                            let n1 = 24 - heavy - n3;
                            let norm = hnorm + 9 * n3 + n1;
                            if norm > max_norm || norm % 16 != 0 {
                                continue;
                            }
                            let m = norm / 16;
                            if !(2..=max_shell).contains(&m) {
                                continue;
                            }
                            // Class-level sum condition: n₁+n₇+n₉ odd
                            // (values ≡ ±1 mod 8: 1, 7, 9).
                            if (n1 + n7 + n9) % 2 != 1 {
                                continue;
                            }
                            let vals: Vec<(u8, u8)> =
                                [(13, n13), (11, n11), (9, n9), (7, n7), (5, n5), (3, n3), (1, n1)]
                                    .into_iter()
                                    .filter(|&(_, n)| n > 0)
                                    .map(|(v, n)| (v as u8, n as u8))
                                    .collect();
                            odd.push(OddClass { shell: m, vals });
                        }
                    }
                }
            }
        }
    }

    ClassSet { even, odd }
}

impl ClassSet {
    /// Total point count of `Shell(m)` from the class cardinalities.
    pub fn shell_cardinality(&self, m: u32) -> u64 {
        self.even
            .iter()
            .filter(|c| c.shell == m)
            .map(EvenClass::cardinality)
            .sum::<u64>()
            + self
                .odd
                .iter()
                .filter(|c| c.shell == m)
                .map(OddClass::cardinality)
                .sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_core::leech::{N_SHELL_13_CUMULATIVE, THETA};

    /// The money test: class enumeration + cardinality formula must
    /// reproduce the theta series (shells 2–5, known constants) AND the
    /// cumulative count N(13) pinned by the paper's Table 1. A wrong
    /// constraint (Golay weight, sign parity, the odd n₁+n₇+n₉ rule…)
    /// cannot survive both checks.
    #[test]
    fn classes_reproduce_theta_series() {
        let cs = enumerate_classes(MAX_SHELL);
        for &(m, n) in THETA.iter() {
            assert_eq!(cs.shell_cardinality(m), n, "shell {m}");
        }
        let cumulative: u64 = (2..=13).map(|m| cs.shell_cardinality(m)).sum();
        assert_eq!(cumulative, N_SHELL_13_CUMULATIVE, "N(13), paper Table 1");
    }

    /// Shell 1 must be empty and known small classes must match G1 counts.
    #[test]
    fn spot_checks() {
        let cs = enumerate_classes(2);
        assert_eq!(cs.shell_cardinality(1), 0);
        assert_eq!(cs.shell_cardinality(2), 196_560);
        // Exactly three classes on shell 2: (2⁸), (4²), (3, 1²³).
        assert_eq!(
            cs.even.iter().filter(|c| c.shell == 2).count()
                + cs.odd.iter().filter(|c| c.shell == 2).count(),
            3
        );
    }
}
