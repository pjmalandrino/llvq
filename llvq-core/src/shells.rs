//! Exhaustive shell enumeration (streaming).
//!
//! Emits every point of `Shell(m)` exactly once, per equivalence class, for
//! the shells whose class decomposition is hand-derived (m = 2, 3). Each
//! emitter constructs only valid members (signs chosen to satisfy the sum
//! congruence directly); the G1 suite cross-checks the totals against the
//! theta series and the membership predicate, so a construction bug here
//! cannot survive testing.
//!
//! Class decompositions (all counts pinned in `tests/g1_invariants.rs`):
//!
//! * Shell(2) = (±2⁸ on octads, even −) ∪ (±4²) ∪ (±3, ±1²³)
//!   = 97,152 + 1,104 + 98,304 = 196,560
//! * Shell(3) = (±2¹² on dodecads, even −) ∪ (±4, ±2⁸ odd −) ∪
//!   (±5, ±1²³) ∪ (±3³, ±1²¹)
//!   = 5,275,648 + 3,108,864 + 98,304 + 8,290,304 = 16,773,120

use crate::golay::Golay;
use crate::leech::{Leech, Point, DIM};

impl Leech {
    /// Visit every point of `Shell(m)` exactly once. Returns `false` if the
    /// shell is not supported (currently m ∈ {2, 3}).
    pub fn for_each_shell_point(&self, m: u32, mut f: impl FnMut(&Point)) -> bool {
        match m {
            2 => {
                shell2(self.golay(), &mut f);
                true
            }
            3 => {
                shell3(self.golay(), &mut f);
                true
            }
            _ => false,
        }
    }
}

/// ±1 base vector of the odd coset: −1 on the codeword support, +1 off it.
fn ones_base(c: u32) -> Point {
    let mut x = [1i32; DIM];
    for (i, xi) in x.iter_mut().enumerate() {
        if c >> i & 1 == 1 {
            *xi = -1;
        }
    }
    x
}

/// Emit all ±2-on-support vectors with the required minus-sign parity.
fn emit_two_support(support: u32, want_odd_minus: bool, f: &mut impl FnMut(&Point)) {
    let pos: Vec<usize> = (0..DIM).filter(|&i| support >> i & 1 == 1).collect();
    let n = pos.len() as u32;
    for s in 0u32..(1 << n) {
        if (s.count_ones() % 2 == 1) != want_odd_minus {
            continue;
        }
        let mut x: Point = [0; DIM];
        for (j, &p) in pos.iter().enumerate() {
            x[p] = if s >> j & 1 == 1 { -2 } else { 2 };
        }
        f(&x);
    }
}

fn shell2(g: &Golay, f: &mut impl FnMut(&Point)) {
    // (±2⁸): octad support, even number of minus signs (Σ ≡ 0 mod 8).
    for &oct in g.of_weight(8) {
        emit_two_support(oct, false, f);
    }
    // (±4²): every placement and sign (each flip shifts Σ by 8).
    for i in 0..DIM {
        for j in (i + 1)..DIM {
            for s in 0u32..4 {
                let mut x: Point = [0; DIM];
                x[i] = if s & 1 == 1 { -4 } else { 4 };
                x[j] = if s & 2 == 2 { -4 } else { 4 };
                f(&x);
            }
        }
    }
    // (±3, ±1²³): one vector per (codeword, position): +3 on the support
    // (replacing a −1), −3 off it (replacing a +1). Both variants satisfy
    // Σ ≡ 4 (mod 8) for every codeword (28−2w ≡ 4 since 2w ≡ 0 mod 8).
    for &c in g.codewords() {
        for p in 0..DIM {
            let mut x = ones_base(c);
            x[p] = if c >> p & 1 == 1 { 3 } else { -3 };
            f(&x);
        }
    }
}

fn shell3(g: &Golay, f: &mut impl FnMut(&Point)) {
    // (±2¹²): dodecad support, even number of minus signs (Σ = 24−4k).
    for &dodecad in g.of_weight(12) {
        emit_two_support(dodecad, false, f);
    }
    // (±4, ±2⁸): octad support for the 2s (odd minus count), the ±4 on any
    // position outside the octad, both signs (±4 shifts Σ by 8).
    for &oct in g.of_weight(8) {
        let pos: Vec<usize> = (0..DIM).filter(|&i| oct >> i & 1 == 1).collect();
        for p4 in (0..DIM).filter(|&p| oct >> p & 1 == 0) {
            for v4 in [4i32, -4] {
                for s in 0u32..256 {
                    if s.count_ones() % 2 == 0 {
                        continue; // need odd minus count among the 2s
                    }
                    let mut x: Point = [0; DIM];
                    for (j, &p) in pos.iter().enumerate() {
                        x[p] = if s >> j & 1 == 1 { -2 } else { 2 };
                    }
                    x[p4] = v4;
                    f(&x);
                }
            }
        }
    }
    // (±5, ±1²³): one vector per (codeword, position): +5 off the support
    // (5 ≡ 1 mod 4), −5 on it (−5 ≡ 3 mod 4).
    for &c in g.codewords() {
        for p in 0..DIM {
            let mut x = ones_base(c);
            x[p] = if c >> p & 1 == 1 { -5 } else { 5 };
            f(&x);
        }
    }
    // (±3³, ±1²¹): one vector per (codeword, 3-subset T); signs forced:
    // +3 on T ∩ c, −3 on T ∖ c.
    for &c in g.codewords() {
        for i in 0..DIM {
            for j in (i + 1)..DIM {
                for k in (j + 1)..DIM {
                    let mut x = ones_base(c);
                    for &t in &[i, j, k] {
                        x[t] = if c >> t & 1 == 1 { 3 } else { -3 };
                    }
                    f(&x);
                }
            }
        }
    }
}
