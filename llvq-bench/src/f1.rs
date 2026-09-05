//! F1 — reading Λ₂₄ as a three-section E₈ coset code, for the Gaussian bench.
//!
//! Lead F1 proposes to stop unfolding the served index. Today a block of 24
//! weights is written as 48 bits on disk and served as 4.804 b/weight in VRAM,
//! because a 47-bit index into a ball of 1.1·10¹⁴ points has no table and must
//! be transcoded into bit planes at load. F1 words the same 24 weights as
//! `[state 8][s₁ w₁][s₂ w₂][s₃ w₃][gain 1]`, three sections of eight
//! coordinates chained by a state, each section decoded through a small table —
//! so the word read from VRAM is the word written on disk.
//!
//! **This module is bench-only.** Nothing here touches `llvq-core`,
//! `llvq-search`, `llvq-quant`, `llvq-artifact` or `llvq-llm`; no served path,
//! no format, no existing test, and `codebook_fingerprint` does not move. The
//! trio reordering lives in a private permuted copy of the Golay table built
//! below. See [`proofs/preregistration-f1b-2026-09-04.md`].
//!
//! ## What was established before this file existed
//!
//! Counted exactly in `bin/f1count.rs` and journalled in
//! `docs/mesures/f1a-comptes-2026-09-04.txt` (with the correction appended to
//! it the same evening):
//!
//! * under the repository's natural coordinate order the two section cuts carry
//!   2¹⁰ = 1024 states, so an 8-bit state field would not fit;
//! * under an ordering that splits the 24 coordinates into three disjoint
//!   octads — a **trio** — each cut carries 2⁸ = 256 states, and it does;
//! * the middle section has 1,024 edges over the 64 Golay states, i.e. **16
//!   branches per state**, four coded label bits.
//!
//! Λ₂₄ adds exactly two bits to the Golay trellis state and no more. A point of
//! the integer embedding is `xⱼ = p + 2cⱼ + 4kⱼ` with `p` the shared parity, `c`
//! a Golay codeword and `Σk ≡ p (mod 2)` — the third constraint of
//! `Leech::contains`, reduced in its own comment. A later section needs `p`, to
//! lay down its coordinates, and the running parity of `Σk`, to know what the
//! remainder owes. Two bits, four states, multiplying the code's own count.

use llvq_core::Golay;

/// The universal 16 KiB decoder table and the 48-bit word it decodes — the
/// reference the CUDA header `llvq_f1rank.cuh` is checked against.
pub mod rank;
pub mod rankbook;

/// The trio: three disjoint octads of the Golay code covering all 24
/// coordinates. Found by `bin/f1count.rs` as the first among the 759 octads;
/// pinned here rather than searched, so this module and that binary cannot
/// drift apart, and asserted against the code in [`TrioCode::new`].
pub const TRIO: [u32; 3] = [0x0000_149f, 0x000f_6840, 0x00f0_8320];

/// Number of Λ₂₄ trellis states at each section cut: 64 Golay states times the
/// four combinations of `(p, parity of Σk)`.
pub const STATES: usize = 256;

/// Coordinates per section.
pub const SECTION: usize = 8;

/// The Golay code in trio order, with the tables a sectioned reading needs.
///
/// Every field is derived from `llvq_core::Golay`, whose weight distribution
/// the G1 suite pins against the published constants — these counts inherit
/// that validation instead of asserting their own.
pub struct TrioCode {
    /// `order[j]` is the natural coordinate index that lands at trio position
    /// `j`. Octad 0 fills positions 0..8, octad 1 fills 8..16, octad 2 16..24.
    pub order: [u32; 24],
    /// The 4,096 codewords, relabelled into trio order.
    pub words: Vec<u32>,
}

impl TrioCode {
    /// Build the permuted table, and refuse to exist if the trio is not one.
    pub fn new() -> Self {
        let g = Golay::new();
        let [a, b, c] = TRIO;

        // The trio must be three disjoint octads of the code covering the 24
        // coordinates. Checked here rather than trusted: a wrong constant would
        // still produce a permutation, still produce numbers, and quantize a
        // different lattice.
        for w in TRIO {
            assert!(g.contains(w), "{w:#08x} is not a Golay codeword");
            assert_eq!(w.count_ones(), 8, "{w:#08x} is not an octad");
        }
        assert_eq!(a & b, 0, "octads 0 and 1 overlap");
        assert_eq!(a & c, 0, "octads 0 and 2 overlap");
        assert_eq!(b & c, 0, "octads 1 and 2 overlap");
        assert_eq!(a | b | c, 0x00ff_ffff, "the trio does not cover the 24");

        let mut order = [0u32; 24];
        let mut j = 0;
        for octad in TRIO {
            for i in 0..24u32 {
                if octad >> i & 1 == 1 {
                    order[j] = i;
                    j += 1;
                }
            }
        }
        assert_eq!(j, 24);

        let words: Vec<u32> = g.codewords().iter().map(|&w| permute(w, &order)).collect();
        Self { order, words }
    }

    /// Coordinate `j` of the trio order, as an index into a natural-order point.
    pub fn natural_index(&self, trio_position: usize) -> usize {
        self.order[trio_position] as usize
    }
}

impl Default for TrioCode {
    fn default() -> Self {
        Self::new()
    }
}

/// Relabel a codeword into trio order: bit `j` of the result is bit `order[j]`
/// of the input.
pub fn permute(w: u32, order: &[u32; 24]) -> u32 {
    (0..24).fold(0u32, |acc, j| acc | (w >> order[j] & 1) << j)
}

/// Undo [`permute`]: bit `order[j]` of the result is bit `j` of the input.
pub fn unpermute(w: u32, order: &[u32; 24]) -> u32 {
    (0..24).fold(0u32, |acc, j| acc | (w >> j & 1) << order[j])
}

/// Move a point from trio order back to the repository's natural order, which
/// is the only order `llvq_core::Leech::contains` understands.
pub fn point_to_natural(trio: &[i32; 24], order: &[u32; 24]) -> [i32; 24] {
    let mut out = [0i32; 24];
    for (j, &v) in trio.iter().enumerate() {
        out[order[j] as usize] = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three section supports are codewords of the permuted table — and
    /// they are not, under an ordering that is not a trio.
    ///
    /// Both halves are needed, and only the second has teeth. `permute(TRIO[0])
    /// == 0xff` is true **by construction**, whatever `TRIO` holds, because
    /// `order` lists that word's own set bits first; asserting it checks
    /// bookkeeping, not mathematics. What discriminates is that `0xff`,
    /// `0xff00` and `0xff0000` are members of the permuted CODE, which is a
    /// statement about the Golay code and fails under any other ordering.
    ///
    /// This replaces the control the F1b spec originally proposed. That one —
    /// reproduce 196,560 through the three-section construction — was
    /// implemented by an adversarial reviewer and run under the trio, the
    /// identity, and a random shuffle: all three returned 196,560. It cannot
    /// see what it is billed to catch. Measured here instead: the trio order
    /// puts 3 of 3 supports in the code, the identity 0 of 3, and twenty random
    /// shuffles 0 of 60.
    #[test]
    fn only_a_trio_ordering_makes_the_section_supports_codewords() {
        const SUPPORTS: [u32; 3] = [0x0000_00ff, 0x0000_ff00, 0x00ff_0000];
        let t = TrioCode::new();
        for w in SUPPORTS {
            assert!(t.words.contains(&w), "{w:#08x} missing from the trio-ordered table");
        }

        // The same table under orderings that isolate no octad. If any of these
        // passed, the check above would be measuring nothing.
        let g = Golay::new();
        let mut identity = [0u32; 24];
        for (i, v) in identity.iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut orders = vec![identity];
        // A deterministic shuffle: a fixed rotation by a stride coprime with 24
        // would map octads to octads too often, so walk a multiplier that is
        // not, and check the result really is a permutation.
        for stride in [5u32, 7, 11, 13, 17] {
            let mut o = [0u32; 24];
            for (i, v) in o.iter_mut().enumerate() {
                *v = (i as u32 * stride + 3) % 24;
            }
            let mut seen = o;
            seen.sort_unstable();
            assert!(seen.iter().copied().eq(0..24), "stride {stride} is not a permutation");
            orders.push(o);
        }
        for o in orders {
            let words: Vec<u32> = g.codewords().iter().map(|&w| permute(w, &o)).collect();
            let hits = SUPPORTS.iter().filter(|w| words.contains(w)).count();
            assert_eq!(hits, 0, "a non-trio ordering put {hits} supports in the code");
        }
    }

    #[test]
    fn the_permutation_is_an_involution_pair_and_preserves_the_code() {
        let t = TrioCode::new();
        let g = Golay::new();
        assert_eq!(t.words.len(), 4096);
        for (&nat, &tri) in g.codewords().iter().zip(&t.words) {
            assert_eq!(unpermute(tri, &t.order), nat, "round trip");
            assert_eq!(nat.count_ones(), tri.count_ones(), "weight is not preserved");
        }
        let mut sorted = t.words.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4096, "the permutation is not injective on the code");
    }

    /// A point permuted into trio order and back is the point it started as,
    /// which is what makes the `Leech::contains` check of §4.2 meaningful.
    #[test]
    fn a_point_survives_the_round_trip_to_natural_order() {
        let t = TrioCode::new();
        let mut x = [0i32; 24];
        for (i, v) in x.iter_mut().enumerate() {
            *v = i as i32 - 11;
        }
        let nat = point_to_natural(&x, &t.order);
        let mut back = [0i32; 24];
        for (j, b) in back.iter_mut().enumerate() {
            *b = nat[t.order[j] as usize];
        }
        assert_eq!(back, x);
    }
}

// ---------------------------------------------------------------------------
// The trellis
// ---------------------------------------------------------------------------

/// The three-section trellis of the trio-ordered Golay code.
///
/// A codeword is a path: a prefix byte on coordinates 0..8, a middle byte on
/// 8..16, a suffix byte on 16..24, with the two cuts carrying the state. Two
/// codewords share a state at a cut when their difference lies in
/// `L_past ⊕ L_future` — the subspace spanned by the codewords living entirely
/// on one side — so a canonical representative is the smallest word of `c ⊕ V`.
/// `V` has 64 elements at either cut, which makes that reduction exhaustive and
/// exact rather than a basis reduction that could be subtly wrong.
///
/// Every count this builds is asserted, because each is a number that will end
/// up in a journal and each has already been got wrong once today: the first
/// run of `bin/f1count.rs` divided 1,024 Golay edges by the 256 Λ₂₄ states and
/// published four branches where there are sixteen.
pub struct Trellis {
    pub code: TrioCode,
    /// The two prefix bytes of each of the 64 Golay states at cut 8.
    pub prefixes: Vec<[u8; 2]>,
    /// The two suffix bytes of each of the 64 Golay states at cut 16.
    pub suffixes: Vec<[u8; 2]>,
    /// `branches[s8]` — the 16 middle bytes leaving state `s8`, each with the
    /// state at cut 16 it lands in. Sorted by byte, so the label is an index.
    pub branches: Vec<Vec<(u8, u8)>>,
}

/// The subspace `L_past(cut) ⊕ L_future(cut)`, spanned by the codewords
/// supported entirely on one side of the cut.
fn subspace(words: &[u32], cut: usize) -> Vec<u32> {
    let low = (1u32 << cut) - 1;
    let mut v = vec![0u32];
    for &g in words.iter().filter(|&&c| c & !low == 0 || c & low == 0) {
        if !v.contains(&g) {
            let grown: Vec<u32> = v.iter().map(|&x| x ^ g).collect();
            v.extend(grown);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Canonical representative of `c`'s coset modulo `v`: the smallest member.
fn coset_rep(c: u32, v: &[u32]) -> u32 {
    v.iter().map(|&x| c ^ x).min().expect("the subspace contains 0")
}

/// Index the distinct values of `keys` in ascending order.
fn index_of(keys: &[u32]) -> std::collections::HashMap<u32, usize> {
    let mut d: Vec<u32> = keys.to_vec();
    d.sort_unstable();
    d.dedup();
    d.into_iter().enumerate().map(|(i, k)| (k, i)).collect()
}

/// Golay trellis states at each cut. Λ₂₄ carries four times this, the extra
/// two bits being `p` and the running parity of `Σk`.
pub const GOLAY_STATES: usize = 64;

/// Middle-section branches leaving one Golay state.
pub const BRANCHES: usize = 16;

impl Trellis {
    pub fn new() -> Self {
        let code = TrioCode::new();
        let (v8, v16) = (subspace(&code.words, 8), subspace(&code.words, 16));
        assert_eq!(v8.len(), 64, "|V₈| is not 64");
        assert_eq!(v16.len(), 64, "|V₁₆| is not 64");

        let rep8: Vec<u32> = code.words.iter().map(|&c| coset_rep(c, &v8)).collect();
        let rep16: Vec<u32> = code.words.iter().map(|&c| coset_rep(c, &v16)).collect();
        let ix8 = index_of(&rep8);
        let ix16 = index_of(&rep16);
        assert_eq!(ix8.len(), GOLAY_STATES, "cut 8 does not carry 64 states");
        assert_eq!(ix16.len(), GOLAY_STATES, "cut 16 does not carry 64 states");

        let mut pre: Vec<Vec<u8>> = vec![Vec::new(); GOLAY_STATES];
        let mut suf: Vec<Vec<u8>> = vec![Vec::new(); GOLAY_STATES];
        let mut edges: Vec<Vec<(u8, u8)>> = vec![Vec::new(); GOLAY_STATES];
        for (i, &w) in code.words.iter().enumerate() {
            let (s8, s16) = (ix8[&rep8[i]], ix16[&rep16[i]]);
            pre[s8].push((w & 0xff) as u8);
            suf[s16].push((w >> 16 & 0xff) as u8);
            edges[s8].push(((w >> 8 & 0xff) as u8, s16 as u8));
        }

        let dedup = |v: &mut Vec<u8>| {
            v.sort_unstable();
            v.dedup();
        };
        let mut prefixes = Vec::with_capacity(GOLAY_STATES);
        let mut suffixes = Vec::with_capacity(GOLAY_STATES);
        for s in 0..GOLAY_STATES {
            dedup(&mut pre[s]);
            dedup(&mut suf[s]);
            // Two per state, and the pair differs by the all-ones byte: the two
            // prefixes of a state are complementary, which is what makes the
            // section point set a coset of 4·E₈ rather than of 4·D₈.
            assert_eq!(pre[s].len(), 2, "state {s} has {} prefixes", pre[s].len());
            assert_eq!(suf[s].len(), 2, "state {s} has {} suffixes", suf[s].len());
            assert_eq!(pre[s][0] ^ pre[s][1], 0xff, "prefixes of state {s} are not complementary");
            assert_eq!(suf[s][0] ^ suf[s][1], 0xff, "suffixes of state {s} are not complementary");
            prefixes.push([pre[s][0], pre[s][1]]);
            suffixes.push([suf[s][0], suf[s][1]]);

            edges[s].sort_unstable();
            edges[s].dedup();
            assert_eq!(edges[s].len(), BRANCHES, "state {s} has {} branches", edges[s].len());
            // The transition must be single-valued: one middle byte out of one
            // state reaches exactly one state. If it were not, the label would
            // not determine the path and the word would not be a bijection.
            let mut bytes: Vec<u8> = edges[s].iter().map(|&(b, _)| b).collect();
            bytes.dedup();
            assert_eq!(bytes.len(), BRANCHES, "state {s}: a middle byte reaches two states");
        }

        let t = Self { code, prefixes, suffixes, branches: edges };
        t.assert_closes();
        t
    }

    /// The count that must close, and the one whose first version closed on the
    /// wrong number.
    ///
    /// 64 states × 2 prefixes × 16 branches × 2 suffixes = 4,096, the code's own
    /// word count. ⚠️ The journal of 2026-09-04 wrote this as 512 × 4 × 2, which
    /// is also 4,096 — the two errors cancelled exactly, so the check passed on
    /// a branch count that was four times too small. Written out factor by
    /// factor here, with each factor asserted separately above, so no product
    /// can compensate for a wrong term.
    fn assert_closes(&self) {
        let paths: usize = (0..GOLAY_STATES)
            .map(|s8| {
                self.prefixes[s8].len()
                    * self
                        .branches[s8]
                        .iter()
                        .map(|&(_, s16)| self.suffixes[s16 as usize].len())
                        .sum::<usize>()
            })
            .sum();
        assert_eq!(paths, 4096, "the trellis enumerates {paths} words, the code has 4096");
    }

    /// Every distinct middle byte, across all states.
    pub fn middle_bytes(&self) -> Vec<u8> {
        let mut v: Vec<u8> = self.branches.iter().flatten().map(|&(b, _)| b).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Every distinct edge `(state8, middle byte, state16)`.
    pub fn edge_count(&self) -> usize {
        self.branches.iter().map(Vec::len).sum()
    }
}

impl Default for Trellis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod trellis_tests {
    use super::*;

    /// Every count of `docs/mesures/f1a-comptes-2026-09-04.txt`, re-derived by a
    /// second implementation. `bin/f1count.rs` reaches them through Golay coset
    /// reduction on raw codewords; this reaches them through the built trellis.
    #[test]
    fn the_trellis_reproduces_the_counted_structure() {
        let t = Trellis::new();
        assert_eq!(t.edge_count(), 1024, "edges");
        assert_eq!(t.edge_count() / GOLAY_STATES, 16, "branches per Golay state");
        assert_eq!(t.middle_bytes().len(), 128, "distinct middle bytes");

        let prefixes: std::collections::HashSet<u8> =
            t.prefixes.iter().flatten().copied().collect();
        let suffixes: std::collections::HashSet<u8> =
            t.suffixes.iter().flatten().copied().collect();
        assert_eq!(prefixes.len(), 128, "distinct prefix bytes");
        assert_eq!(suffixes.len(), 128, "distinct suffix bytes");
    }

    /// Walking the trellis must rebuild the permuted code exactly, set for set:
    /// no path leaves the code, and no codeword is unreachable. This is the
    /// check that a wrong state definition fails and a wrong branch count
    /// cannot survive.
    #[test]
    fn walking_the_trellis_rebuilds_the_code() {
        let t = Trellis::new();
        let mut walked = std::collections::HashSet::new();
        for s8 in 0..GOLAY_STATES {
            for &p in &t.prefixes[s8] {
                for &(m, s16) in &t.branches[s8] {
                    for &s in &t.suffixes[s16 as usize] {
                        walked.insert(p as u32 | (m as u32) << 8 | (s as u32) << 16);
                    }
                }
            }
        }
        let code: std::collections::HashSet<u32> = t.code.words.iter().copied().collect();
        assert_eq!(walked.len(), 4096, "the walk yields {} words", walked.len());
        assert_eq!(walked, code, "the walk does not reproduce the permuted code");
    }
}

// ---------------------------------------------------------------------------
// Per-section point sets, and the counting DP the truncation needs
// ---------------------------------------------------------------------------

/// The admissible 8-dimensional points of one section, given the state.
///
/// A point of the integer embedding is `yⱼ = p + 2cⱼ + 4kⱼ`. Fixing the state
/// fixes `p`, fixes which byte patterns `c` the section may use, and — for the
/// two end sections — fixes what the parity of `Σk` must be. So a section set
/// is exactly a list of patterns, a parity `p`, and an optional constraint on
/// `Σk mod 2`.
///
/// Written in that literal form on purpose. An abstract `E8`/`D8` type would
/// read better and would be one step further from `Leech::contains`, which is
/// what actually decides whether a point is in the lattice.
///
/// Covolumes, which close the construction:
///
/// | section | patterns | parity | covolume |
/// |---|---|---|---|
/// | 1, 3 (ends) | 2, complementary | constrained | 2¹⁶ |
/// | 2 (middle)  | 16 | free | 2¹² |
///
/// `2¹⁶ · 2¹² · 2¹⁶ = 2⁴⁴` per state, over 256 states, is `2³⁶` — and
/// `det(√8·Λ₂₄) = (√8)²⁴ = 8¹² = 2³⁶`. The word is a bijection, and this is
/// where that is visible rather than asserted.
#[derive(Clone)]
pub struct SectionSet {
    /// The byte patterns this section may use; bit `j` is coordinate `j`.
    pub patterns: Vec<u8>,
    /// The shared parity of the whole block.
    pub p: u32,
    /// Required parity of `Σk` over this section's eight coordinates, or `None`
    /// when the section is free to choose (the middle one).
    pub k_parity: Option<u32>,
}

impl SectionSet {
    /// `y` for one pattern and one choice of `k` — the definition, written
    /// forwards. `pub` because the encoder will build points with it; today
    /// only the tests do, and they are the reason it exists: `contains` reads
    /// `k` back out of a point, so a bug shared between the two would cancel
    /// unless one of them is the plain forward map.
    pub fn point(&self, pattern: u8, k: &[i32; SECTION]) -> [i32; SECTION] {
        let mut y = [0i32; SECTION];
        for (j, out) in y.iter_mut().enumerate() {
            *out = self.p as i32 + 2 * ((pattern >> j & 1) as i32) + 4 * k[j];
        }
        y
    }

    /// Is `y` a member? Recovers `k` from `y` and checks every constraint, so
    /// this is the definition and not a re-derivation of it.
    pub fn contains(&self, y: &[i32; SECTION]) -> bool {
        let mut pattern = 0u8;
        let mut ksum = 0i64;
        for (j, &v) in y.iter().enumerate() {
            let d = v - self.p as i32;
            if d.rem_euclid(2) != 0 {
                return false;
            }
            let half = d.div_euclid(2);
            if half.rem_euclid(2) == 1 {
                pattern |= 1 << j;
            }
            ksum += ((v - self.p as i32 - 2 * (half.rem_euclid(2))) / 4) as i64;
        }
        if !self.patterns.contains(&pattern) {
            return false;
        }
        match self.k_parity {
            Some(r) => ksum.rem_euclid(2) == r as i64,
            None => true,
        }
    }

    /// `count[t]` = number of members with `‖y‖² == t`, for `t` in `0..=t_max`.
    ///
    /// Keyed **per pattern** and summed, never pooled. A per-position pooled
    /// walk would count vectors taking coordinate `j` from one pattern and
    /// `j+1` from another; the resulting mod-4 word is generally not a Golay
    /// codeword, and an adversarial review measured the inflation at 609,553
    /// against a true 2,401 on one real coset — a factor of 254. It is silent:
    /// the cardinality stays plausible and the failure surfaces later as a rank
    /// collision.
    pub fn norm_histogram(&self, t_max: usize) -> Vec<u64> {
        let mut total = vec![0u64; t_max + 1];
        for &pattern in &self.patterns {
            // dp[n][par] over the coordinates seen so far.
            let mut dp = vec![[0u64; 2]; t_max + 1];
            dp[0][0] = 1;
            for j in 0..SECTION {
                let base = self.p as i32 + 2 * ((pattern >> j & 1) as i32);
                let mut next = vec![[0u64; 2]; t_max + 1];
                // |y_j| ≤ √t_max bounds k_j on both sides.
                let bound = (t_max as f64).sqrt().ceil() as i32;
                let (lo, hi) = ((-bound - base).div_euclid(4), (bound - base).div_euclid(4) + 1);
                // The range must span every value that could contribute, which
                // is every `v ≡ base (mod 4)` with `v² ≤ t_max`. Stated as that
                // condition and not as a probe one step outside it: a mutation
                // test of 2026-09-04 shortened `bound` by four, and both the
                // brute-force comparison (small `t_max`) and the covolume ratio
                // (a thin tail inside tolerance) let it through — as did a first
                // version of this guard, which only looked one step out and
                // moved with the range it was checking.
                let m_max = (t_max as f64).sqrt().floor() as i32;
                assert!(
                    base + 4 * lo <= -m_max && base + 4 * hi >= m_max,
                    "the k range [{lo}, {hi}] misses values with v² ≤ {t_max}"
                );
                for k in lo..=hi {
                    let v = base + 4 * k;
                    let sq = (v * v) as usize;
                    if sq > t_max {
                        continue;
                    }
                    let kp = k.rem_euclid(2) as usize;
                    for n in 0..=(t_max - sq) {
                        for par in 0..2 {
                            let c = dp[n][par];
                            if c != 0 {
                                next[n + sq][par ^ kp] += c;
                            }
                        }
                    }
                }
                dp = next;
            }
            let want = self.k_parity.map(|r| r as usize);
            for (n, row) in dp.iter().enumerate() {
                total[n] += match want {
                    Some(r) => row[r],
                    None => row[0] + row[1],
                };
            }
        }
        total
    }
}

impl Trellis {
    /// Section 1 of a state: the prefixes, with `Σk` forced to `r`.
    pub fn section1(&self, s8: usize, p: u32, r: u32) -> SectionSet {
        SectionSet { patterns: self.prefixes[s8].to_vec(), p, k_parity: Some(r) }
    }

    /// Section 2 of a state: the 16 middle bytes, parity free — it is a
    /// transmitted degree of freedom, which is what makes this section's
    /// covolume 2¹² where the ends are 2¹⁶.
    pub fn section2(&self, s8: usize, p: u32) -> SectionSet {
        let patterns = self.branches[s8].iter().map(|&(b, _)| b).collect();
        SectionSet { patterns, p, k_parity: None }
    }

    /// Section 3 of a state at cut 16: the suffixes, with `Σk` forced so the
    /// block closes on `Σk ≡ p (mod 2)`.
    pub fn section3(&self, s16: usize, p: u32, r_in: u32) -> SectionSet {
        SectionSet { patterns: self.suffixes[s16].to_vec(), p, k_parity: Some((p ^ r_in) & 1) }
    }
}

#[cfg(test)]
mod section_tests {
    use super::*;

    /// The DP counts what the definition admits, checked by brute force.
    ///
    /// Enumerating every `k` in a box and testing `contains` is the slow,
    /// obviously-correct answer; the DP is the fast one that the truncation
    /// will rely on. They must agree exactly, histogram bin by histogram bin.
    #[test]
    fn the_counting_dp_agrees_with_brute_force() {
        let t = Trellis::new();
        const T_MAX: usize = 120;
        for (label, set) in [
            ("section 1", t.section1(0, 0, 0)),
            ("section 1, odd", t.section1(7, 1, 1)),
            ("section 2", t.section2(0, 0)),
            ("section 3", t.section3(3, 1, 0)),
        ] {
            let dp = set.norm_histogram(T_MAX);

            let mut brute = vec![0u64; T_MAX + 1];
            let bound = (T_MAX as f64).sqrt().ceil() as i32 / 4 + 2;
            let mut k = [0i32; SECTION];
            // Eight nested loops as one odometer over [-bound, bound]^8.
            let span = (2 * bound + 1) as i64;
            for code in 0..span.pow(SECTION as u32) {
                let mut c = code;
                for slot in k.iter_mut() {
                    *slot = (c % span) as i32 - bound;
                    c /= span;
                }
                for &pattern in &set.patterns {
                    let y = set.point(pattern, &k);
                    let n: i64 = y.iter().map(|&v| (v as i64) * (v as i64)).sum();
                    if n as usize <= T_MAX && set.contains(&y) {
                        brute[n as usize] += 1;
                    }
                }
            }
            assert_eq!(dp, brute, "{label}: the DP and brute force disagree");
        }
    }

    /// The covolume claim, checked by counting rather than asserted.
    ///
    /// A lattice coset of covolume `V` puts about `vol(ball) / V` points inside
    /// a large ball. The end sections claim 2¹⁶ and the middle 2¹², and their
    /// product over 256 states is `det(√8·Λ₂₄) = 2³⁶` — the bijection argument.
    /// At radius² 4000 in dimension 8 the count is in the thousands, so a
    /// wrong covolume by any factor of two is far outside the sampling wobble.
    #[test]
    fn the_section_covolumes_are_what_the_bijection_needs() {
        let t = Trellis::new();
        const T: usize = 4000;
        // vol(B_8(R)) = π⁴ R⁸ / 24
        let vol = std::f64::consts::PI.powi(4) * (T as f64).powi(4) / 24.0;
        for (label, set, covol) in [
            ("section 1", t.section1(0, 0, 0), 65536.0),
            ("section 2", t.section2(0, 0), 4096.0),
            ("section 3", t.section3(5, 1, 1), 65536.0),
        ] {
            let n: u64 = set.norm_histogram(T).iter().sum();
            let expected = vol / covol;
            let ratio = n as f64 / expected;
            assert!(
                (0.96..1.04).contains(&ratio),
                "{label}: {n} points against {expected:.0} expected for covolume {covol}, ratio {ratio:.4}"
            );
        }
    }

    /// A section point, embedded back into 24 coordinates alongside two others
    /// drawn from the same state, must land in Λ₂₄ — checked by `llvq-core`
    /// itself, in its own coordinate order. This is the check that a wrong
    /// state definition cannot survive.
    #[test]
    fn a_path_through_the_trellis_lands_in_the_leech_lattice() {
        let t = Trellis::new();
        let leech = llvq_core::Leech::new();
        let mut checked = 0;
        for p in 0..2u32 {
            for r in 0..2u32 {
                for s8 in [0usize, 7, 23, 61] {
                    let (mid, s16) = t.branches[s8][3];
                    let s1 = t.section1(s8, p, r);
                    let s2 = t.section2(s8, p);
                    let s3 = t.section3(s16 as usize, p, r);
                    // One representative per section: k = 0 everywhere, except
                    // that section 1 and 3 must meet their parity, and section
                    // 2 must contribute the parity the state assumed.
                    let mut k1 = [0i32; SECTION];
                    k1[0] = r as i32;
                    let mut k3 = [0i32; SECTION];
                    k3[0] = ((p ^ r) & 1) as i32;
                    let y1 = s1.point(s1.patterns[0], &k1);
                    let y2 = s2.point(mid, &[0i32; SECTION]);
                    let y3 = s3.point(s3.patterns[0], &k3);
                    assert!(s1.contains(&y1), "section 1 rejects its own point");
                    assert!(s2.contains(&y2), "section 2 rejects its own point");
                    assert!(s3.contains(&y3), "section 3 rejects its own point");

                    let mut trio = [0i32; 24];
                    trio[..8].copy_from_slice(&y1);
                    trio[8..16].copy_from_slice(&y2);
                    trio[16..].copy_from_slice(&y3);
                    let natural = point_to_natural(&trio, &t.code.order);
                    assert!(
                        leech.contains(&natural),
                        "p={p} r={r} s8={s8}: the path leaves Λ₂₄ — {natural:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 16);
    }
}

// ---------------------------------------------------------------------------
// Truncation: which 2^w points of a section the label field addresses
// ---------------------------------------------------------------------------

/// The truncated region of one section: exactly `2^w` points, so the label is a
/// bijection and the rate is `w` bits by construction.
///
/// **Not "the points inside a fixed radius".** The number of coset points in a
/// ball depends on the coset offset, so a fixed radius gives a different count
/// per state and the 47-bit word stops being one-to-one — after which the
/// claimed 2.000 b/dim is simply false. The rule is instead *the `2^w`
/// lowest-norm points, ties on the boundary shell broken by ascending
/// lexicographic order of the eight coordinates*.
///
/// The boundary shell is not optional in either direction. Dropping it entirely
/// leaves a section short of its label space; including it whole overshoots —
/// an adversarial review measured the full-shell codebook at 3.13× too large,
/// worth about +2.5 pp of retention, which is more than the whole question F1b
/// is asked to settle.
pub struct Region {
    /// Squared norm of the boundary shell.
    pub rho2: usize,
    /// Members strictly inside it.
    pub n_below: u64,
    /// Members taken from the boundary shell, in lexicographic order.
    pub n_tie: u64,
    /// `2^w`.
    pub size: u64,
}

impl SectionSet {
    /// Locate the boundary shell for a label field of `w` bits.
    pub fn region(&self, w: u32) -> Region {
        let size = 1u64 << w;
        // A radius wide enough to hold 2^w points, found by doubling rather
        // than guessed: an under-sized t_max would silently return the largest
        // shell it happened to see.
        let mut t_max = 64usize;
        let (hist, rho2, cum) = loop {
            let hist = self.norm_histogram(t_max);
            let mut cum = 0u64;
            let mut found = None;
            for (t, &c) in hist.iter().enumerate() {
                cum += c;
                if cum >= size {
                    found = Some((t, cum));
                    break;
                }
            }
            match found {
                Some((t, c)) => break (hist, t, c),
                None => t_max *= 2,
            }
            // No termination guard is needed: the set is an infinite lattice
            // coset, so the cumulative count diverges and the loop ends.
        };
        let n_below = cum - hist[rho2];
        Region { rho2, n_below, n_tie: size - n_below, size }
    }

    /// Every member with `‖y‖² ≤ t`, enumerated. Recursive with norm pruning,
    /// which keeps it linear in the answer rather than in the search box.
    pub fn enumerate_below(&self, t: usize) -> Vec<[i32; SECTION]> {
        let mut out = Vec::new();
        for &pattern in &self.patterns {
            let mut y = [0i32; SECTION];
            self.walk(pattern, 0, 0, 0, &mut y, t, &mut out);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        pattern: u8,
        j: usize,
        norm: usize,
        kpar: u32,
        y: &mut [i32; SECTION],
        t: usize,
        out: &mut Vec<[i32; SECTION]>,
    ) {
        if j == SECTION {
            if self.k_parity.is_none_or(|r| r == kpar) {
                out.push(*y);
            }
            return;
        }
        let base = self.p as i32 + 2 * ((pattern >> j & 1) as i32);
        let reach = ((t - norm) as f64).sqrt().floor() as i32;
        let lo = (-reach - base).div_euclid(4);
        let hi = (reach - base).div_euclid(4) + 1;
        for k in lo..=hi {
            let v = base + 4 * k;
            let sq = (v * v) as usize;
            if norm + sq > t {
                continue;
            }
            y[j] = v;
            self.walk(pattern, j + 1, norm + sq, kpar ^ (k.rem_euclid(2) as u32), y, t, out);
        }
    }

    /// The region's members, in the order the rule defines: by norm, then
    /// lexicographically. Brute force, for the checks and for small `w`.
    pub fn region_points(&self, w: u32) -> Vec<[i32; SECTION]> {
        let r = self.region(w);
        let mut pts = self.enumerate_below(r.rho2);
        pts.sort_by_key(|y| {
            let n: i64 = y.iter().map(|&v| (v as i64) * (v as i64)).sum();
            (n, *y)
        });
        pts.truncate(r.size as usize);
        pts
    }

    /// Is `y` inside the region, decided without enumerating it.
    ///
    /// ⚠️ The lexicographic rank is computed **per pattern and summed**, never
    /// pooled across the patterns of the set. A per-position pooled walk counts
    /// vectors taking coordinate `j` from one pattern and `j+1` from another;
    /// the resulting mod-4 word is generally not a Golay codeword. An
    /// adversarial review measured the inflation on one real section coset at
    /// 609,553 against a true 2,401 — a factor of 254 — and noted it is silent:
    /// the cardinality stays plausible and the failure surfaces much later, as
    /// a rank collision.
    pub fn in_region(&self, y: &[i32; SECTION], r: &Region) -> bool {
        if !self.contains(y) {
            return false;
        }
        let n: usize = y.iter().map(|&v| (v * v) as usize).sum();
        match n.cmp(&r.rho2) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => self.lex_rank_on_shell(y, r.rho2) < r.n_tie,
        }
    }

    /// How many members of the boundary shell are lexicographically before `y`.
    fn lex_rank_on_shell(&self, y: &[i32; SECTION], rho2: usize) -> u64 {
        let mut rank = 0u64;
        for &pattern in &self.patterns {
            // suffix[j][n][par]: ways to fill positions j..8 with exact norm n
            // and k-parity par, for THIS pattern.
            let mut suffix = vec![vec![[0u64; 2]; rho2 + 1]; SECTION + 1];
            suffix[SECTION][0][0] = 1;
            for j in (0..SECTION).rev() {
                let base = self.p as i32 + 2 * ((pattern >> j & 1) as i32);
                let reach = (rho2 as f64).sqrt().floor() as i32;
                let lo = (-reach - base).div_euclid(4);
                let hi = (reach - base).div_euclid(4) + 1;
                for k in lo..=hi {
                    let v = base + 4 * k;
                    let sq = (v * v) as usize;
                    if sq > rho2 {
                        continue;
                    }
                    let kp = k.rem_euclid(2) as usize;
                    for n in 0..=(rho2 - sq) {
                        for par in 0..2 {
                            let c = suffix[j + 1][n][par];
                            if c != 0 {
                                suffix[j][n + sq][par ^ kp] += c;
                            }
                        }
                    }
                }
            }

            // Walk y's own positions, counting completions under every smaller
            // admissible value at each one.
            let mut norm = 0usize;
            let mut kpar = 0u32;
            for j in 0..SECTION {
                let base = self.p as i32 + 2 * ((pattern >> j & 1) as i32);
                let reach = ((rho2 - norm) as f64).sqrt().floor() as i32;
                let lo = (-reach - base).div_euclid(4);
                let hi = (reach - base).div_euclid(4) + 1;
                for k in lo..=hi {
                    let v = base + 4 * k;
                    if v >= y[j] {
                        continue;
                    }
                    let sq = (v * v) as usize;
                    if norm + sq > rho2 {
                        continue;
                    }
                    let need_par = self.k_parity.map(|r| r ^ kpar ^ (k.rem_euclid(2) as u32));
                    let row = &suffix[j + 1][rho2 - norm - sq];
                    rank += match need_par {
                        Some(p) => row[p as usize],
                        None => row[0] + row[1],
                    };
                }
                // Continue down y's own branch, but only if y uses this pattern.
                let d = y[j] - base;
                if d.rem_euclid(4) != 0 {
                    break;
                }
                let k = d.div_euclid(4);
                norm += (y[j] * y[j]) as usize;
                kpar ^= k.rem_euclid(2) as u32;
                if norm > rho2 {
                    break;
                }
            }
        }
        rank
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    /// The region holds exactly `2^w` distinct points, all of them members.
    /// This is the bijection, at the level of one section.
    #[test]
    fn a_region_is_exactly_two_to_the_w_members() {
        let t = Trellis::new();
        for (label, set) in [
            ("section 1", t.section1(0, 0, 0)),
            ("section 2", t.section2(11, 1)),
            ("section 3", t.section3(40, 1, 0)),
        ] {
            for w in [8u32, 10, 12] {
                let pts = set.region_points(w);
                assert_eq!(pts.len(), 1 << w, "{label} w={w}: size");
                let mut d = pts.clone();
                d.sort_unstable();
                d.dedup();
                assert_eq!(d.len(), 1 << w, "{label} w={w}: duplicates");
                for y in &pts {
                    assert!(set.contains(y), "{label} w={w}: a region point is not a member");
                }
            }
        }
    }

    /// The fast membership test agrees with brute-force enumeration, on the
    /// points inside AND on the ones just outside.
    ///
    /// This is the test the whole truncation rests on. `in_region` decides by
    /// counting; `region_points` decides by enumerating and sorting. They share
    /// no code beyond `contains`, so a wrong rank cannot hide in both — and a
    /// rank pooled across patterns instead of summed per pattern is exactly the
    /// error that would pass a cardinality check and fail here.
    #[test]
    fn the_counted_membership_agrees_with_the_enumerated_region() {
        let t = Trellis::new();
        for (label, set) in [
            ("section 1", t.section1(3, 0, 1)),
            ("section 2", t.section2(3, 0)),
            ("section 3", t.section3(17, 1, 1)),
        ] {
            for w in [8u32, 10, 12] {
                let r = set.region(w);
                let inside: std::collections::HashSet<[i32; SECTION]> =
                    set.region_points(w).into_iter().collect();

                // Everything on or below the boundary shell, so the sample
                // straddles the tie-break rather than avoiding it.
                let all = set.enumerate_below(r.rho2);
                assert!(
                    all.len() as u64 > r.size,
                    "{label} w={w}: the shell adds nothing, the tie-break is untested"
                );
                let mut disagree = 0;
                for y in &all {
                    if set.in_region(y, &r) != inside.contains(y) {
                        disagree += 1;
                    }
                }
                assert_eq!(
                    disagree, 0,
                    "{label} w={w}: {disagree} of {} points are ranked differently by the two rules",
                    all.len()
                );
            }
        }
    }

    /// The boundary shell really is a boundary: `n_below` members strictly
    /// inside, `n_tie` taken from the shell, and the two sum to `2^w`.
    #[test]
    fn the_boundary_shell_accounting_closes() {
        let t = Trellis::new();
        for (label, set) in [("section 1", t.section1(9, 1, 0)), ("section 2", t.section2(9, 1))] {
            for w in [8u32, 12] {
                let r = set.region(w);
                assert_eq!(r.n_below + r.n_tie, r.size, "{label} w={w}: accounting");
                assert!(r.n_tie > 0, "{label} w={w}: the shell contributes nothing");
                let below = set.enumerate_below(r.rho2 - 1).len() as u64;
                assert_eq!(below, r.n_below, "{label} w={w}: n_below disagrees with enumeration");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The encoder: constrained nearest point, and a region test fast enough to use
// ---------------------------------------------------------------------------

/// A section set with everything the encoder needs precomputed.
///
/// [`SectionSet::in_region`] rebuilds a suffix table on every call, which is
/// fine for a test and hopeless for 20,000 blocks × a scale sweep × 256 states.
/// The tables depend only on the set and its boundary shell, so they are built
/// once here. Points strictly inside the shell then cost one comparison, and
/// only shell points pay the walk — and shell points are a thin minority.
pub struct Prepared {
    pub set: SectionSet,
    pub region: Region,
    /// `suffix[pattern index][j][n][parity]`: ways to fill positions `j..8`
    /// with exact norm `n` and k-parity `parity`, for that pattern alone.
    suffix: Vec<Vec<Vec<[u64; 2]>>>,
    /// A member of lowest norm, so [`Prepared::nearest`] can always answer.
    /// Without it the encoder returns nothing for a target far outside the
    /// region, and "no answer" is not a thing a quantizer may do.
    fallback: [i32; SECTION],
}

impl Prepared {
    pub fn new(set: SectionSet, w: u32) -> Self {
        let region = set.region(w);
        let rho2 = region.rho2;
        let suffix = set
            .patterns
            .iter()
            .map(|&pattern| {
                let mut s = vec![vec![[0u64; 2]; rho2 + 1]; SECTION + 1];
                s[SECTION][0][0] = 1;
                for j in (0..SECTION).rev() {
                    let base = set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                    let reach = (rho2 as f64).sqrt().floor() as i32;
                    let lo = (-reach - base).div_euclid(4);
                    let hi = (reach - base).div_euclid(4) + 1;
                    for k in lo..=hi {
                        let v = base + 4 * k;
                        let sq = (v * v) as usize;
                        if sq > rho2 {
                            continue;
                        }
                        let kp = k.rem_euclid(2) as usize;
                        for n in 0..=(rho2 - sq) {
                            for par in 0..2 {
                                let c = s[j + 1][n][par];
                                if c != 0 {
                                    s[j][n + sq][par ^ kp] += c;
                                }
                            }
                        }
                    }
                }
                s
            })
            .collect();
        // The lowest-norm member, found by growing a radius rather than
        // enumerating the whole region: at w = 15 that would be 32,768 points
        // per prepared section, and there are hundreds of them.
        let mut t = 8usize;
        let fallback = loop {
            let pts = set.enumerate_below(t.min(rho2));
            if let Some(y) = pts.into_iter().min_by_key(|y| {
                let n: i64 = y.iter().map(|&v| (v as i64) * (v as i64)).sum();
                (n, *y)
            }) {
                break y;
            }
            assert!(t <= rho2, "the region is empty up to its own boundary shell");
            t *= 2;
        };
        Self { set, region, suffix, fallback }
    }

    /// Membership, with the tables already built.
    pub fn contains(&self, y: &[i32; SECTION]) -> bool {
        if !self.set.contains(y) {
            return false;
        }
        let n: usize = y.iter().map(|&v| (v * v) as usize).sum();
        match n.cmp(&self.region.rho2) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => self.shell_rank(y) < self.region.n_tie,
        }
    }

    /// [`SectionSet::lex_rank_on_shell`], reading the precomputed tables.
    /// Kept a separate implementation on purpose: the two agree in
    /// `region_tests`, and a shared body would make that agreement vacuous.
    fn shell_rank(&self, y: &[i32; SECTION]) -> u64 {
        let rho2 = self.region.rho2;
        let mut rank = 0u64;
        for (pi, &pattern) in self.set.patterns.iter().enumerate() {
            let mut norm = 0usize;
            let mut kpar = 0u32;
            for (j, &yj) in y.iter().enumerate() {
                let base = self.set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                let reach = ((rho2 - norm) as f64).sqrt().floor() as i32;
                let lo = (-reach - base).div_euclid(4);
                let hi = (reach - base).div_euclid(4) + 1;
                for k in lo..=hi {
                    let v = base + 4 * k;
                    if v >= yj {
                        continue;
                    }
                    let sq = (v * v) as usize;
                    if norm + sq > rho2 {
                        continue;
                    }
                    let row = &self.suffix[pi][j + 1][rho2 - norm - sq];
                    rank += match self.set.k_parity {
                        Some(r) => row[(r ^ kpar ^ (k.rem_euclid(2) as u32)) as usize],
                        None => row[0] + row[1],
                    };
                }
                let d = yj - base;
                if d.rem_euclid(4) != 0 {
                    break;
                }
                norm += (yj * yj) as usize;
                kpar ^= d.div_euclid(4).rem_euclid(2) as u32;
                if norm > rho2 {
                    break;
                }
            }
        }
        rank
    }

    /// Nearest member of the region to `target`, or `None` when every candidate
    /// tried falls outside it.
    ///
    /// The unconstrained answer first: for one pattern the admissible set is
    /// `{p·1 + 2c + 4k}` with `Σk` free or fixed, so writing
    /// `z = (target − p·1 − 2c)/4` reduces it to "nearest integer vector to `z`,
    /// with a parity constraint" — round every coordinate, and if the parity is
    /// wrong, re-round the single coordinate whose rounding was least decided.
    /// That is the textbook D₈ decode, and it is exact.
    ///
    /// ⚠️ Exact for the **coset**, not for the truncated region. When the
    /// winner falls outside, the spec's own rule was to drop the branch, and an
    /// adversarial review named that the uncontrolled bias of the design: it
    /// bites hardest exactly at the boundary, which is where the whole shaping
    /// question lives. So the fallbacks below are tried before giving up —
    /// every single-coordinate re-rounding, which walks the answer back toward
    /// the origin one step at a time. The residual bias can only understate F1,
    /// and `region_tests` bounds it rather than assuming it small.
    pub fn nearest(&self, target: &[f64; SECTION]) -> ([i32; SECTION], f64) {
        let dist = |y: &[i32; SECTION]| -> f64 {
            y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum()
        };
        let mut best = (self.fallback, dist(&self.fallback));
        // Radial shrinks. A target well outside the region has no in-region
        // neighbour among the single re-roundings of its own nearest coset
        // point, so the search is repeated on the target walked back toward the
        // origin. Distances are always measured against the ORIGINAL target, so
        // a shrink can only be chosen when it genuinely wins.
        for shrink in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1] {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for &pattern in &self.set.patterns {
                for cand in self.candidates(pattern, &scaled) {
                    if !self.contains(&cand) {
                        continue;
                    }
                    let d = dist(&cand);
                    if d < best.1 {
                        best = (cand, d);
                    }
                }
            }
        }
        best
    }

    /// Nearest member of the region using **one** pattern and one k-parity,
    /// while membership is still decided against the whole region.
    ///
    /// Section 2 needs this: its region is defined on all sixteen middle bytes
    /// with the parity free, because that is what the label field addresses —
    /// but the encoder must know, for each byte and each outgoing parity,
    /// what the best point is, since those two choices are what the trellis
    /// branches on. Filtering the search while testing membership against the
    /// full region is the only way to get both right.
    pub fn nearest_constrained(
        &self,
        target: &[f64; SECTION],
        pattern: u8,
        kparity: u32,
    ) -> Option<([i32; SECTION], f64)> {
        let dist = |y: &[i32; SECTION]| -> f64 {
            y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum()
        };
        let mut best: Option<([i32; SECTION], f64)> = None;
        for shrink in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0] {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for cand in self.candidates_with(pattern, &scaled, Some(kparity)) {
                if !self.contains(&cand) || self.k_parity_of(pattern, &cand) != kparity {
                    continue;
                }
                let d = dist(&cand);
                if best.as_ref().is_none_or(|&(_, bd)| d < bd) {
                    best = Some((cand, d));
                }
            }
        }
        best
    }

    /// `Σk mod 2` of a point read under one pattern.
    fn k_parity_of(&self, pattern: u8, y: &[i32; SECTION]) -> u32 {
        (0..SECTION)
            .map(|j| {
                let base = self.set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                ((y[j] - base).div_euclid(4)).rem_euclid(2) as u32
            })
            .fold(0, |a, b| a ^ b)
    }

    /// The unconstrained winner for one pattern, then its single-coordinate
    /// re-roundings, nearest first.
    fn candidates(&self, pattern: u8, target: &[f64; SECTION]) -> Vec<[i32; SECTION]> {
        self.candidates_with(pattern, target, self.set.k_parity)
    }

    /// [`Prepared::candidates`] against an explicit parity target.
    fn candidates_with(
        &self,
        pattern: u8,
        target: &[f64; SECTION],
        want_parity: Option<u32>,
    ) -> Vec<[i32; SECTION]> {
        let base: Vec<i32> = (0..SECTION)
            .map(|j| self.set.p as i32 + 2 * ((pattern >> j & 1) as i32))
            .collect();
        let z: Vec<f64> = (0..SECTION).map(|j| (target[j] - base[j] as f64) / 4.0).collect();
        let mut k: Vec<i32> = z.iter().map(|v| v.round() as i32).collect();
        // Rounding regret per coordinate, and the direction that undoes it.
        let mut regret: Vec<(f64, usize)> =
            (0..SECTION).map(|j| ((z[j] - k[j] as f64).abs(), j)).collect();
        regret.sort_by(|a, b| b.0.total_cmp(&a.0));

        if let Some(r) = want_parity {
            let sum: i32 = k.iter().sum();
            if sum.rem_euclid(2) != r as i32 {
                // Flip the least decided coordinate: the exact D₈ repair.
                let j = regret[0].1;
                k[j] += if z[j] > k[j] as f64 { 1 } else { -1 };
            }
        }
        let build = |k: &[i32]| {
            let mut y = [0i32; SECTION];
            for j in 0..SECTION {
                y[j] = base[j] + 4 * k[j];
            }
            y
        };
        let mut out = vec![build(&k)];
        // Fallbacks, in order of how little they cost: re-round one coordinate
        // at a time, starting from the least decided. Parity is preserved by
        // moving in pairs when the section constrains it.
        for &(_, j) in &regret {
            for step in [-1i32, 1] {
                let mut k2 = k.clone();
                k2[j] += step;
                if want_parity.is_some() {
                    // A single step breaks the parity; fix it on the next least
                    // decided coordinate.
                    let j2 = regret.iter().map(|&(_, x)| x).find(|&x| x != j).expect("8 > 1");
                    k2[j2] += if z[j2] > k[j2] as f64 { 1 } else { -1 };
                }
                out.push(build(&k2));
            }
        }
        out
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use llvq_core::SplitMix64;

    fn target(rng: &mut SplitMix64, spread: f64) -> [f64; SECTION] {
        let mut t = [0.0f64; SECTION];
        for v in t.iter_mut() {
            // Two draws folded into one normal-ish value; the encoder only
            // needs a spread of targets, not a calibrated distribution.
            let u: f64 = rng.next_f64();
            let w: f64 = rng.next_f64();
            *v = spread * (-2.0f64 * (u + 1e-12).ln()).sqrt() * (std::f64::consts::TAU * w).cos();
        }
        t
    }

    /// The precomputed membership test and the one that rebuilds its tables
    /// must agree. They are separate implementations on purpose: if they shared
    /// a body, `region_tests` proving one correct would prove nothing about the
    /// one the encoder actually calls.
    #[test]
    fn the_prepared_membership_agrees_with_the_slow_one() {
        let t = Trellis::new();
        for (label, set) in
            [("section 1", t.section1(5, 0, 1)), ("section 2", t.section2(5, 0))]
        {
            for w in [8u32, 10] {
                let r = set.region(w);
                let pts = set.enumerate_below(r.rho2);
                let prep = Prepared::new(
                    SectionSet {
                        patterns: set.patterns.clone(),
                        p: set.p,
                        k_parity: set.k_parity,
                    },
                    w,
                );
                let mut bad = 0;
                for y in &pts {
                    if prep.contains(y) != set.in_region(y, &r) {
                        bad += 1;
                    }
                }
                assert_eq!(bad, 0, "{label} w={w}: {bad} of {} disagree", pts.len());
            }
        }
    }

    /// The encoder's answer is the true nearest member of the region — checked
    /// against exhaustive search over the whole region.
    ///
    /// This is the check that bounds the design's known bias. The candidate
    /// list is a heuristic: the unconstrained D₈ winner plus single-coordinate
    /// re-roundings. Where it is not exact, it can only return something
    /// farther away, so the measured MSE is an upper bound and the retention a
    /// lower one — the bias is one-sided and against F1. What is not acceptable
    /// is not knowing its size, so this measures it.
    #[test]
    fn the_candidate_search_is_exact_or_its_shortfall_is_bounded() {
        let t = Trellis::new();
        let mut rng = SplitMix64::new(0x5f1b_2026_0904);
        for (label, set, w) in [
            ("section 1", t.section1(2, 0, 0), 10u32),
            ("section 2", t.section2(2, 0), 10),
            ("section 3", t.section3(31, 1, 1), 10),
        ] {
            let region: Vec<[i32; SECTION]> = set.region_points(w);
            let prep = Prepared::new(
                SectionSet { patterns: set.patterns.clone(), p: set.p, k_parity: set.k_parity },
                w,
            );
            let (mut exact, mut total, mut worst) = (0usize, 0usize, 0.0f64);
            for _ in 0..200 {
                let x = target(&mut rng, 4.0);
                let truth = region
                    .iter()
                    .map(|y| {
                        y.iter().zip(&x).map(|(&v, &e)| (v as f64 - e).powi(2)).sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                let got = prep.nearest(&x).1;
                assert!(got >= truth - 1e-9, "{label}: found a point closer than the region's best");
                total += 1;
                if got <= truth + 1e-9 {
                    exact += 1;
                } else {
                    worst = worst.max(got / truth - 1.0);
                }
            }
            let rate = exact as f64 / total as f64;
            assert!(
                rate >= 0.90,
                "{label}: exact on only {:.1}% of targets, worst excess {:.3}",
                100.0 * rate,
                worst
            );
        }
    }

    /// Every point the encoder returns is a member of the region, and of Λ₂₄
    /// once the three sections are assembled and un-permuted.
    #[test]
    fn what_the_encoder_returns_is_in_the_lattice() {
        let t = Trellis::new();
        let leech = llvq_core::Leech::new();
        let mut rng = SplitMix64::new(0xf1b_0904);
        for s8 in [0usize, 19, 44] {
            for p in 0..2u32 {
                for r in 0..2u32 {
                    let (mid_byte, s16) = t.branches[s8][5];
                    let p1 = Prepared::new(t.section1(s8, p, r), 10);
                    let p2 = Prepared::new(
                        SectionSet { patterns: vec![mid_byte], p, k_parity: None },
                        10,
                    );
                    let x = target(&mut rng, 6.0);
                    let (y1, _) = p1.nearest(&x);
                    let (y2, _) = p2.nearest(&x);
                    // Section 2's k-parity is what section 3 must close against.
                    let delta: u32 = (0..SECTION)
                        .map(|j| {
                            let base = p as i32 + 2 * ((mid_byte >> j & 1) as i32);
                            ((y2[j] - base) / 4).rem_euclid(2) as u32
                        })
                        .fold(0, |a, b| a ^ b);
                    let p3 = Prepared::new(t.section3(s16 as usize, p, r ^ delta), 10);
                    let (y3, _) = p3.nearest(&x);

                    let mut trio = [0i32; 24];
                    trio[..8].copy_from_slice(&y1);
                    trio[8..16].copy_from_slice(&y2);
                    trio[16..].copy_from_slice(&y3);
                    let natural = point_to_natural(&trio, &t.code.order);
                    assert!(
                        leech.contains(&natural),
                        "s8={s8} p={p} r={r}: the encoded block is not in Λ₂₄"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The whole codebook: three sections, 256 states, one best path per block
// ---------------------------------------------------------------------------

/// One section's chosen point and its squared distance, when a choice exists.
pub type SectionPick = Option<([i32; SECTION], f64)>;

/// The F1 codebook, prepared once and encoded against many blocks.
///
/// ## Why a scale sweep and not a direct search
///
/// The bench scores shape-gain: `e² = ‖x‖² − 2·g·t + g²` with `t = ⟨x, y⟩/‖y‖`
/// and `g` fixed by `‖x‖` alone (`shape_gain_mse_shipped`). With `g` fixed,
/// minimizing `e²` is maximizing `t` — a purely angular objective, and one that
/// does **not** decompose across sections, because `‖y‖` couples them.
///
/// What does decompose is `‖x − s·y‖² = Σᵢ ‖xᵢ − s·yᵢ‖²` at a fixed `s`. So the
/// encoder sweeps `s`, solves each section independently at that scale, joins
/// them through the trellis, and then scores the winner on the true angular
/// objective. Every candidate is feasible by construction, so a coarse sweep
/// can only understate F1 — the bias is one-sided, and against it.
///
/// ## What the trellis actually branches on
///
/// Section 2 chooses two things at once: which of its sixteen middle bytes, and
/// which outgoing k-parity `δ`. The byte fixes the state at cut 16; `δ` fixes
/// what section 3's parity must close against, `p ⊕ r ⊕ δ`. Both are free
/// choices paid for by the label field, which is why the middle section carries
/// more bits than the ends.
pub struct Codebook {
    pub trellis: Trellis,
    pub w: [u32; 3],
    /// `[(p * 2 + r) * 64 + s8]`
    sec1: Vec<Prepared>,
    /// `[p * 8 + mset]`
    sec2: Vec<Prepared>,
    /// `[(p * 2 + r_out) * 64 + s16]`
    sec3: Vec<Prepared>,
    /// Which of the eight distinct middle-byte sets each state uses.
    mset_of: Vec<usize>,
    msets: Vec<Vec<u8>>,
}

impl Codebook {
    pub fn new(w: [u32; 3]) -> Self {
        let trellis = Trellis::new();
        // The 64 states share only eight distinct middle-byte sets, each a
        // coset of a 4-dimensional subspace (`examples/f1mids.rs`). Preparing
        // eight regions instead of sixty-four is what makes the sweep
        // affordable.
        let mut msets: Vec<Vec<u8>> = Vec::new();
        let mut mset_of = vec![0usize; GOLAY_STATES];
        for (s8, slot) in mset_of.iter_mut().enumerate() {
            let mut m: Vec<u8> = trellis.branches[s8].iter().map(|&(b, _)| b).collect();
            m.sort_unstable();
            *slot = match msets.iter().position(|s| *s == m) {
                Some(i) => i,
                None => {
                    msets.push(m);
                    msets.len() - 1
                }
            };
        }
        assert_eq!(msets.len(), 8, "the middle-byte sets do not collapse to eight");

        let mut sec1 = Vec::with_capacity(4 * GOLAY_STATES);
        let mut sec3 = Vec::with_capacity(4 * GOLAY_STATES);
        for p in 0..2u32 {
            for r in 0..2u32 {
                for s in 0..GOLAY_STATES {
                    sec1.push(Prepared::new(trellis.section1(s, p, r), w[0]));
                    sec3.push(Prepared::new(
                        SectionSet {
                            patterns: trellis.suffixes[s].to_vec(),
                            p,
                            k_parity: Some(r),
                        },
                        w[2],
                    ));
                }
            }
        }
        let mut sec2 = Vec::with_capacity(16);
        for p in 0..2u32 {
            for m in &msets {
                sec2.push(Prepared::new(
                    SectionSet { patterns: m.clone(), p, k_parity: None },
                    w[1],
                ));
            }
        }
        Self { trellis, w, sec1, sec2, sec3, mset_of, msets }
    }

    fn s1(&self, p: u32, r: u32, s8: usize) -> &Prepared {
        &self.sec1[((p * 2 + r) as usize) * GOLAY_STATES + s8]
    }
    fn s3(&self, p: u32, r_out: u32, s16: usize) -> &Prepared {
        &self.sec3[((p * 2 + r_out) as usize) * GOLAY_STATES + s16]
    }
    fn s2(&self, p: u32, mset: usize) -> &Prepared {
        &self.sec2[p as usize * 8 + mset]
    }

    /// Best codeword for `x` at one scale, as `(point in trio order, ‖x − s·y‖²)`.
    pub fn encode_at_scale(&self, x: &[f64; 24], s: f64) -> [i32; 24] {
        let part = |lo: usize| -> [f64; SECTION] {
            let mut t = [0.0f64; SECTION];
            for (j, v) in t.iter_mut().enumerate() {
                *v = x[lo + j] / s;
            }
            t
        };
        let (t1, t2, t3) = (part(0), part(8), part(16));

        let mut best: Option<(f64, [i32; 24])> = None;
        for p in 0..2u32 {
            // Section 2, per middle-byte set, per byte, per outgoing parity.
            let mut mid: Vec<[SectionPick; 2]> = Vec::new();
            for m in 0..8usize {
                for &b in &self.msets[m] {
                    mid.push([
                        self.s2(p, m).nearest_constrained(&t2, b, 0),
                        self.s2(p, m).nearest_constrained(&t2, b, 1),
                    ]);
                }
            }
            let mid_at = |m: usize, b: u8| -> &[SectionPick; 2] {
                let k = self.msets[m].iter().position(|&x| x == b).expect("byte in its set");
                &mid[m * 16 + k]
            };

            // Section 3 depends on the path only through `(r_out, s16)`, of
            // which there are 128 — not through the 2,048 (state, branch,
            // parity) triples that reach them. Hoisted out of the innermost
            // loop for that reason: leaving it inside recomputed the same
            // eight-dimensional search sixteen times over, and cost a factor of
            // nine on the whole measurement.
            let mut end: Vec<([i32; SECTION], f64)> = Vec::with_capacity(2 * GOLAY_STATES);
            for r_out in 0..2u32 {
                for s16 in 0..GOLAY_STATES {
                    end.push(self.s3(p, r_out, s16).nearest(&t3));
                }
            }

            for r in 0..2u32 {
                for s8 in 0..GOLAY_STATES {
                    let (y1, d1) = self.s1(p, r, s8).nearest(&t1);
                    let m = self.mset_of[s8];
                    for &(b, s16) in &self.trellis.branches[s8] {
                        for delta in 0..2u32 {
                            let Some((y2, d2)) = mid_at(m, b)[delta as usize] else {
                                continue;
                            };
                            let r_out = (p ^ r ^ delta) & 1;
                            let (y3, d3) = end[r_out as usize * GOLAY_STATES + s16 as usize];
                            let cost = d1 + d2 + d3;
                            if best.as_ref().is_none_or(|&(bc, _)| cost < bc) {
                                let mut y = [0i32; 24];
                                y[..8].copy_from_slice(&y1);
                                y[8..16].copy_from_slice(&y2);
                                y[16..].copy_from_slice(&y3);
                                best = Some((cost, y));
                            }
                        }
                    }
                }
            }
        }
        best.expect("the codebook is never empty").1
    }

    /// The block's best `t = ⟨x, y⟩/‖y‖` over a scale grid — the one number the
    /// bench's scoring rule consumes.
    pub fn best_t(&self, x: &[f64; 24], scales: &[f64]) -> (f64, [i32; 24]) {
        let mut best = (f64::NEG_INFINITY, [0i32; 24]);
        for &s in scales {
            let y = self.encode_at_scale(x, s);
            let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
            let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
            if nn > 0.0 {
                let t = dot / nn.sqrt();
                if t > best.0 {
                    best = (t, y);
                }
            }
        }
        best
    }
}
