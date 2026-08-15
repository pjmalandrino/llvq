//! The CNS — E1v's combinatorial numbering, and the width it actually costs.
//!
//! `proofs/preregistration-p5-2026-08-14.md`, amendment É0 (2026-08-15). P5's
//! C1 originally asked this width to equal `Widths::radix2` bit for bit; É0
//! established that no object can satisfy that *and* C3's "zero division",
//! because the two round at different granularities:
//!
//! * **per stage** — one `⌈log₂⌉` per composition radix (the codeword, the two
//!   arrangements, the two sign fields). That is `radix2`, and it is what the
//!   published 51,87 bits/block is. Its arrangement field is a **packed**
//!   multiset rank, and peeling it kind by kind is a mixed-radix peel, i.e. a
//!   **division** by the product of the remaining radices.
//! * **per kind** — one `⌈log₂⌉` per *kind* of each arrangement, so every
//!   radix is a power of two and the peel is a **shift**. That is what this
//!   module builds, what a binomial walk reads, and it costs **+0,3516
//!   bit/block** in file order (101 of 383 classes differ, up to +2 bits).
//!
//! É0 retained the second: P1 measured a binomial walk at 0,3101 ns/block, not
//! a packed peeler, and the number that opens the line has to describe the
//! object that was measured.
//!
//! ## What this module is not allowed to look at
//!
//! §4 of P5: *ni la CNS ni son test de largeur n'appellent `Widths`,
//! `widths_of_even`, `widths_of_odd` ni quoi que ce soit de `radixstudy`* —
//! otherwise C1 is the tautology §2.3 warns about. That separation is
//! structural here (`radixstudy` lives in `llvq-bench`, which this crate cannot
//! see) **and** pinned by [`tests::the_cns_names_nothing_from_the_width_study`],
//! which greps this very file. Structure can be undone by a refactor; the grep
//! notices.
//!
//! ## The record, fixed here
//!
//! ```text
//!   [header 10 = class 9 | gain 1]
//!   even : [golay ⌈log₂ γ(w)⌉] [support: one field per kind but the last]
//!          [off-support: idem] [word signs w−1] [free signs free_n]
//!   odd  : [golay 12]          [24 slots: one field per kind but the last]
//!   origin: the header alone, 10 bits, no rank field
//! ```
//!
//! The final kind of an arrangement carries **no field**: it takes the slots
//! left, so its radix is 1 and storing a rank for it would be storing a
//! constant. The two sign fields are already powers of two in the archive's own
//! composition (`s_w = 2^(w−1)`, `s_f = 2^free_n`), so per-stage and per-kind
//! agree on them — the whole difference lives in the two arrangements.
//!
//! ## What a width proves, and what it does not
//!
//! A width at or above `⌈log₂ |class|⌉` proves only that **no bijection is
//! impossible for want of room**. It is not a bijection. That is C2's sweep,
//! and this module's [`cns_cardinality`] is the precondition C2 needs: the
//! product of the CNS's own radices must equal the class's cardinality, or the
//! numbering counts a different set.

use crate::fastdec::{FastDecoder, MAX_KINDS};
use crate::rankdec::walk_radices;
use llvq_core::{Golay, DIM};

/// Class field (9 bits) plus gain (1 bit) — the same header every MSL of this
/// repo reads, and the whole record of an origin block.
pub const HEADER_BITS: u64 = 10;

/// `⌈log₂ x⌉`, zero for `x ≤ 1`.
#[inline]
pub fn lg_ceil(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        128 - (x - 1).leading_zeros()
    }
}

/// One class's CNS record layout: the radix of every field, in composition
/// order, and nothing derived from another crate's width table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CnsLayout {
    /// Number of Golay codewords the class may carry — `γ(w)` on the even
    /// coset, the whole table on the odd one.
    pub golay_radix: u64,
    /// Per-kind radices of the support arrangement (`w` slots, even coset) or
    /// of the single 24-slot arrangement (odd coset). The last kind's radix is
    /// 1 and carries no field.
    pub on_radices: [u64; MAX_KINDS],
    pub n_on: usize,
    /// Per-kind radices of the off-support arrangement. Empty on the odd coset.
    pub off_radices: [u64; MAX_KINDS],
    pub n_off: usize,
    /// Free sign patterns on the support — `2^(w−1)`, one sign being forced by
    /// the parity repair. `1` when `w == 0`.
    pub s_w: u64,
    /// Free sign patterns off the support — `2^free_n`. `1` on the odd coset,
    /// where membership forces every sign.
    pub s_f: u64,
    pub odd: bool,
    /// Support weight; `DIM` on the odd coset, whose single problem spans all
    /// 24 slots.
    pub w: u32,
}

impl CnsLayout {
    /// The record's width in bits: the header, plus one `⌈log₂ radix⌉` per
    /// field. **This is the per-kind accounting of É0**, not `radix2`.
    pub fn bits(&self) -> u64 {
        let mut b = HEADER_BITS + u64::from(lg_ceil(u128::from(self.golay_radix)));
        for &r in self.on_radices.iter().take(self.n_on) {
            b += u64::from(lg_ceil(u128::from(r)));
        }
        for &r in self.off_radices.iter().take(self.n_off) {
            b += u64::from(lg_ceil(u128::from(r)));
        }
        b + u64::from(lg_ceil(u128::from(self.s_w))) + u64::from(lg_ceil(u128::from(self.s_f)))
    }

    /// How many distinct records the layout admits — the product of every
    /// radix.
    ///
    /// C2's precondition: this must equal the class's cardinality, or the CNS
    /// numbers a different set and no bijection with the archive can exist.
    /// It is deliberately **not** the same quantity as [`Self::bits`]: a layout
    /// can have room to spare (rounding) and still count the right set.
    pub fn cardinality(&self) -> u128 {
        let mut p = u128::from(self.golay_radix);
        for &r in self.on_radices.iter().take(self.n_on) {
            p *= u128::from(r);
        }
        for &r in self.off_radices.iter().take(self.n_off) {
            p *= u128::from(r);
        }
        p * u128::from(self.s_w) * u128::from(self.s_f)
    }
}

/// The origin's record: the header alone.
///
/// P5 §2.2 decides this rather than leaving it to the chantier — the 4B carries
/// **zero** origin block, so no sweep will ever exercise it and the fixture
/// must. It decodes to the zero vector.
pub const ORIGIN_BITS: u64 = HEADER_BITS;

/// Build class `ci`'s layout, from the class table and a binomial table, and
/// from nothing else.
pub fn cns_layout(fd: &FastDecoder, golay: &Golay, ci: usize) -> CnsLayout {
    let c = fd.cascade_class(ci);
    let (mut on_radices, mut off_radices) = ([1u64; MAX_KINDS], [1u64; MAX_KINDS]);

    if c.odd {
        // One arrangement over the 24 coordinates. Signs are forced by
        // membership in the codeword, so they carry no bits — the fact
        // `generic.rs` states and `Golay70`'s kernel recomputes.
        let mut counts = [0u8; MAX_KINDS];
        counts[..c.n_kinds].copy_from_slice(&c.counts[..c.n_kinds]);
        on_radices = walk_radices(&counts, c.n_kinds, DIM);
        return CnsLayout {
            golay_radix: golay.codewords().len() as u64,
            on_radices,
            n_on: c.n_kinds,
            off_radices,
            n_off: 0,
            s_w: 1,
            s_f: 1,
            odd: true,
            w: DIM as u32,
        };
    }

    let w = c.w as usize;
    if w > 0 {
        let mut wc = [0u8; MAX_KINDS];
        wc[..c.n_word].copy_from_slice(&c.word_counts[..c.n_word]);
        on_radices = walk_radices(&wc, c.n_word, w);
    }
    if DIM - w > 0 {
        off_radices = walk_radices(&c.off_counts, c.n_off, DIM - w);
    }
    CnsLayout {
        golay_radix: golay.of_weight(w).len() as u64,
        on_radices,
        n_on: if w > 0 { c.n_word } else { 0 },
        off_radices,
        n_off: if DIM - w > 0 { c.n_off } else { 0 },
        s_w: c.s_w,
        s_f: c.s_f,
        odd: false,
        w: c.w,
    }
}

/// The whole codebook's layouts, class order.
pub fn cns_layouts(fd: &FastDecoder, golay: &Golay) -> Vec<CnsLayout> {
    (0..fd.n_classes())
        .map(|ci| cns_layout(fd, golay, ci))
        .collect()
}

/// How many records class `ci`'s layout admits — the product of its radices.
pub fn cns_cardinality(fd: &FastDecoder, golay: &Golay, ci: usize) -> u128 {
    cns_layout(fd, golay, ci).cardinality()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rankdec::{binom, walk_cardinality};

    /// The class's own cardinality, read from the index map rather than
    /// recomputed — `class_range` is `decode`'s convention, so this is the size
    /// of the set the archive numbers.
    fn archive_cardinality(fd: &FastDecoder, ci: usize) -> u128 {
        let (first, last) = fd.class_range(ci);
        u128::from(last - first + 1)
    }

    /// 🚨 **The load-bearing one.** The CNS's radices must multiply to exactly
    /// what the archive numbers. A layout that counted a different set could
    /// still be wide enough, still decode *something*, and still pass every
    /// width check — and no bijection with the archive would exist.
    ///
    /// This is the analogue of `the_exact_column_is_the_class_cardinality`
    /// (P5 §2.3), and it is here rather than in a bench because everything
    /// downstream is only as good as it.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "sweeps every class, run in release")]
    fn the_cns_radices_multiply_to_the_class_cardinality() {
        let fd = FastDecoder::new();
        let golay = Golay::new();
        for ci in 0..fd.n_classes() {
            assert_eq!(
                cns_cardinality(&fd, &golay, ci),
                archive_cardinality(&fd, ci),
                "classe {ci} : la CNS numérote un autre ensemble que l'archive"
            );
        }
    }

    /// A width below `⌈log₂ |classe|⌉` makes a bijection impossible for want of
    /// room. It proves nothing positive — that is C2 — but it fails loudly on a
    /// counting bug.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "sweeps every class, run in release")]
    fn the_cns_is_never_narrower_than_the_class_rank() {
        let fd = FastDecoder::new();
        let golay = Golay::new();
        for ci in 0..fd.n_classes() {
            let l = cns_layout(&fd, &golay, ci);
            let card = archive_cardinality(&fd, ci);
            assert!(
                l.bits() - HEADER_BITS >= u64::from(lg_ceil(card)),
                "classe {ci} : {} bits de champ pour une cardinalité de {card}",
                l.bits() - HEADER_BITS
            );
        }
    }

    /// Every arrangement's per-kind radices must multiply to that arrangement's
    /// own count — the property that lets the walk replace the archive's
    /// unranking at all. Checked against `walk_cardinality`, which is the
    /// walk's own product, and against the class record's `a_on`/`a_off`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "sweeps every class, run in release")]
    fn each_arrangement_spans_its_own_count() {
        let fd = FastDecoder::new();
        let golay = Golay::new();
        for ci in 0..fd.n_classes() {
            let c = fd.cascade_class(ci);
            let l = cns_layout(&fd, &golay, ci);
            let prod = |r: &[u64; MAX_KINDS], n: usize| -> u128 {
                r.iter().take(n).map(|&x| u128::from(x)).product()
            };
            if c.odd {
                let mut counts = [0u8; MAX_KINDS];
                counts[..c.n_kinds].copy_from_slice(&c.counts[..c.n_kinds]);
                assert_eq!(prod(&l.on_radices, l.n_on), u128::from(c.m_arr), "classe {ci}");
                assert_eq!(
                    walk_cardinality(&counts, c.n_kinds, DIM),
                    c.m_arr,
                    "classe {ci} : la marche ne couvre pas m_arr"
                );
            } else {
                assert_eq!(prod(&l.on_radices, l.n_on), u128::from(c.a_on), "classe {ci}");
                assert_eq!(prod(&l.off_radices, l.n_off), u128::from(c.a_off), "classe {ci}");
            }
        }
    }

    /// Every radix but the last of an arrangement must be a real choice, and the
    /// last must be exactly 1: a final kind takes what is left, so a field for
    /// it would store a constant — and a radix of 1 anywhere else would mean the
    /// walk had already run out of slots.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "sweeps every class, run in release")]
    fn the_final_kind_of_each_arrangement_carries_no_field() {
        let fd = FastDecoder::new();
        let golay = Golay::new();
        for ci in 0..fd.n_classes() {
            let l = cns_layout(&fd, &golay, ci);
            for (r, n) in [(&l.on_radices, l.n_on), (&l.off_radices, l.n_off)] {
                if n == 0 {
                    continue;
                }
                assert_eq!(r[n - 1], 1, "classe {ci} : le dernier genre porte un champ");
                assert_eq!(
                    lg_ceil(u128::from(r[n - 1])),
                    0,
                    "classe {ci} : et il coûte des bits"
                );
            }
        }
    }

    /// The origin's record is the header alone — P5 §2.2, decided rather than
    /// deferred, because the 4B carries no origin block and no sweep will ever
    /// exercise it.
    #[test]
    fn the_origin_record_is_the_header_alone() {
        assert_eq!(ORIGIN_BITS, HEADER_BITS);
        assert_eq!(ORIGIN_BITS, 10);
    }

    /// The binomial table this module reads is the crate's own, not a second
    /// derivation — `C(p, c)` zero above the diagonal, which is what lets a
    /// kind that must take every remaining slot be placed at all.
    #[test]
    fn the_binomials_are_the_crates_own() {
        for n in 0..=DIM {
            for k in 0..=DIM {
                let want: u64 = if k > n {
                    0
                } else {
                    (0..k).fold(1u64, |a, i| a * (n - i) as u64 / (i + 1) as u64)
                };
                assert_eq!(binom(n, k), want, "C({n},{k})");
            }
        }
    }

    /// 🚨 **C1's independence clause, enforced on the source itself.**
    ///
    /// P5 §4: *l'égalité doit être un résultat de deux chemins indépendants,
    /// jamais une lecture* — ni la CNS ni son test de largeur n'appellent
    /// `Widths`, `widths_of_even`, `widths_of_odd` ni quoi que ce soit de
    /// `radixstudy`. Today that separation is structural, `radixstudy` living in
    /// a crate this one cannot see; a refactor could undo it silently, and this
    /// notices.
    ///
    /// Two exclusions, and both are the difference between the rule and a
    /// caricature of it. **Comment lines are stripped**: the module header has
    /// to name `Widths` and the study to explain what É0 arbitrated, and a rule
    /// that forbade explaining itself would be obeyed by deleting the
    /// explanation. **This test's own body is cut off** at its `fn` line, since
    /// it has to spell the names to forbid them. What remains is the code, and
    /// the code is what C1 is about.
    #[test]
    fn the_cns_names_nothing_from_the_width_study() {
        let src = include_str!("cns.rs");
        let marker = "fn the_cns_names_nothing_from_the_width_study";
        let body = src.find(marker).expect("ce test est dans ce fichier");
        let head: String = src[..body]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for forbidden in ["radixstudy", "Widths", "widths_of_"] {
            assert!(
                !head.contains(forbidden),
                "la CNS nomme `{forbidden}` : C1 redeviendrait une lecture, pas une \
                 dérivation indépendante (P5 §4)"
            );
        }
    }
}
