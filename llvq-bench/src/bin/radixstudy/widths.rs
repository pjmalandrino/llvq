//! What one block costs under each runtime variant: the constants of the
//! accounting, the class key, the per-class widths of the even and odd cosets,
//! and the table of priced variants they fill.

use super::*;

/// Class id + gain, the header every runtime variant pays.
pub(super) const HEADER_BITS: u64 = 10;
/// Bits of a Golay codeword **rank** among all 4096 — `GOLAY70_RANK_BITS`.
/// Resolved through the 16 KiB codeword table `Golay70` already passes to the
/// device (`llvq-cuda/kernels/llvq_golay.cuh:14-18`).
pub(super) const GOLAY_RANK_BITS: u64 = 12;
/// Bits of a Golay **message**, the `[24, 12, 8]` code's dimension: the field
/// `golay_signs` stores in place of a 24-bit sign mask on the odd coset,
/// resolved by re-encoding (`m · g₂(x)` plus the parity extension —
/// `llvq-core/src/golay.rs:62-70`), i.e. 12 conditional XORs and no table.
///
/// Numerically equal to [`GOLAY_RANK_BITS`] because `|Golay| = 2¹²`, and
/// deliberately a separate constant: a rank costs a table lookup, a message
/// costs an encode, and the two are not interchangeable in a kernel. Pinned by
/// [`tests::a_golay_message_and_a_golay_rank_are_both_twelve_bits`].
pub(super) const GOLAY_MSG_BITS: u64 = 12;
/// Index bits of the published `leech1c12` file, plus its gain bit.
pub(super) const ARCHIVE_BITS: u64 = 47 + 1;
/// The spec's admission threshold for opening the E3 workstream, in the kernel
/// accounting. `S_spec` of the 2026-08-13 pre-registration, unchanged since
/// `spec-memoire-extreme-2026-08-12.md:185-188`.
pub(super) const E3_CRITERION: f64 = 2.6;
/// The **other half** of the same criterion, and the one this binary ignored
/// until 2026-08-13: `spec-memoire-extreme-2026-08-12.md:186-187` opens the E3
/// workstream on a conjunction: "≤ 2.6 b/weight projected thesis **AND** a
/// fixed-depth decoder **≤ 24 steps** with no inter-slot serial state".
///
/// A bit count that clears [`E3_CRITERION`] with a deeper decoder is a
/// **reopening** of this clause, not an admission under it, and
/// [`Admissibility`] is what keeps the two apart in the decision rather than
/// only in the prose.
pub(super) const MAX_DEPTH: u32 = 24;
/// `S_alt` — the same 32 GB rung re-derived at the (32 GB; 2 GB margin; KV q8;
/// 8k) triple, pre-registration §2.
///
/// ⚠️ A **candidate** threshold, not the project's: it is opposable only if the
/// product note of §5 keeps that triple, and that note does not exist. It
/// exists here so a result between the two can be *placed*, never so it can be
/// called a pass.
pub(super) const S_ALT: f64 = 3.09;
/// Lanes a variable-width stream is grouped over — `llvq_artifact::e1c::E1C_GROUP`.
pub(super) const GROUP: u64 = 32;
/// Bits of base offset a group pays, in **either** addressing mode: the word
/// that says where the group starts.
pub(super) const BASE_OFFSET_BITS: u64 = 32;
/// Machine word a warp-scan group's payload rounds up to.
pub(super) const WORD_BITS: u64 = 32;
/// The two thresholds as the pre-registration writes them in bits per block on
/// the 4B, kept verbatim so the run can show its own conversion against them.
/// They are *labels*, not the rule: the rule is stated in b/weight.
pub(super) const PREREG_BITS: (f64, f64) = (58.86, 70.66);

/// `⌈log₂ n⌉`, and 0 for `n ≤ 1` — a field with one possible value costs
/// nothing, which is the whole point of `golay_tight`.
pub(super) fn lg_ceil(n: u128) -> u64 {
    if n <= 1 {
        0
    } else {
        (128 - (n - 1).leading_zeros()) as u64
    }
}

/// Canonical key of a class: `(odd, [(|value|, count); 5])` in `ClassLevels`
/// order — descending count, ties by descending value.
///
/// Exists because the two sources of truth are indexed differently:
/// `enumerate_classes` knows the composition radices, `FastDecoder` knows
/// which class a block is in. Matching them by *content* rather than by
/// position means a reordering of either shows up as a lookup failure instead
/// of as silently mismatched radices.
pub(super) type Key = (bool, [(i32, u8); MAX_LEVELS]);

pub(super) fn canonical(mut kinds: Vec<(i32, u8)>, odd: bool) -> Key {
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut k = [(0i32, 0u8); MAX_LEVELS];
    for (i, &v) in kinds.iter().enumerate() {
        k[i] = v;
    }
    (odd, k)
}

pub(super) fn key_of_levels(lv: &ClassLevels) -> Key {
    canonical(
        (0..lv.len).map(|i| (lv.values[i], lv.counts[i])).collect(),
        lv.odd,
    )
}

/// What one class costs under each variant. All widths are **payload bits per
/// block**, header included.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Widths {
    /// Rounded mixed-radix composition, class stored in 9 bits.
    pub(super) radix2: u64,
    /// Same with a uniform 12-bit codeword field.
    radix2_g12: u64,
    /// Per-slot level fields — the `Planes` geometry.
    pub(super) perslot: u64,
    /// `Golay70`, flat.
    pub(super) golay70: u64,
    /// `Golay70` with the A and sign planes cut to this class's needs.
    pub(super) golay_tight: u64,
    /// T3: the `perslot` level planes with the sign field cut to what the
    /// class cardinality charges — [`even_sign_bits`] on the even coset, a
    /// 12-bit Golay message on the odd one.
    pub(super) golay_signs: u64,
    /// Sign bits alone of that variant, so the run can report the expected
    /// saving against the flat 24-bit mask the served layouts spend. Not a
    /// variant: a component, printed once.
    pub(super) sign_field: u64,
    /// `⌈log₂ |class|⌉` — the information floor of the class alone, printed
    /// to show how much of the archive's 47 bits is the *choice of class*
    /// rather than the point within it, and the width `e1v-packé` pays for the
    /// point (plus the header).
    ///
    /// It is the ceiling of the log of `|class|` itself, not of a model of it:
    /// [`tests::the_exact_column_is_the_class_cardinality`] checks it against
    /// `EvenClass::cardinality` / `OddClass::cardinality` for all 383 classes,
    /// which is what makes every "never narrower than the class rank"
    /// assertion below mean something.
    pub(super) exact: u64,
}

/// Sign bits a class **really** charges, read off the index's own two sign
/// radices: `S_w · S_f = 2^(w−1 if w>0 else 0) · 2^free_n`
/// (`llvq-search/src/index.rs:138-139`, `classes.rs:112-113`).
///
/// The subtlety, and the reason this is not `nonzero − 1`: the parity
/// constraint fixes the sign of the last slot **of the codeword support** —
/// the values ≡ 2 (mod 4) — not of the last nonzero slot in general. A class
/// with `w = 0` has no support, so nothing is fixed and every one of its
/// `free_n` nonzero slots pays a bit. `nonzero − 1` would charge `free_n − 1`
/// there, which no encoder could invert.
pub(super) fn even_sign_bits(c: &EvenClass) -> u64 {
    let free_n: u64 = c.free_vals.iter().map(|&(_, n)| u64::from(n)).sum();
    u64::from(c.w.saturating_sub(1)) + free_n
}

/// Bits an even class spends on the two planes `Golay70` carries flat.
///
/// * **A plane** — the codeword fixes each slot's residue mod 4 (word values
///   are `≡ 2`, free values and zero are `≡ 0`). Within a residue the class
///   may hold several magnitudes, and only then does a slot need a bit:
///   `⌈log₂ k⌉` per slot of that residue, where `k` is the count of distinct
///   magnitudes in it. `Golay70` charges 1 everywhere and refuses `k > 2`;
///   this charges what the class needs and allows `k > 2` at `⌈log₂ k⌉`.
/// * **sign plane** — one bit per nonzero slot, less the one the parity
///   constraint fixes (`S_w = 2^(w−1)`, not `2^w`, in the index's own
///   composition). Zero slots carry no sign.
pub(super) fn even_planes(c: &EvenClass) -> (u64, u64) {
    let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
    let n0 = 24 - c.w - free_n;
    // Residue 2: the word values, on the codeword support (w slots).
    // Residue 0: the free values and the zeros, off it.
    let a_word = u64::from(c.w) * lg_ceil(c.word_vals.len() as u128);
    let off_kinds = c.free_vals.len() + usize::from(n0 > 0);
    let a_free = u64::from(24 - c.w) * lg_ceil(off_kinds as u128);
    let nonzero = 24 - n0;
    (a_word + a_free, u64::from(nonzero.saturating_sub(1)))
}

/// Bits an odd class spends on the same two planes.
///
/// The sign plane is **zero**: on the odd coset the signs are forced by
/// membership (`generic.rs` / `index.rs` — "signs carry no information"), and
/// `Golay70`'s kernel recomputes them as `neg = c_i ^ flag`. The magnitude
/// plane, however, costs `⌈log₂ L⌉` per slot and **not less**: the codeword
/// says which slots carry `x_i ≡ 3 (mod 4)`, a property of the *signed* value,
/// so it constrains the sign-magnitude pair jointly and cannot be spent twice.
/// That is precisely the refutation of the 46-bit variant, and the reason this
/// function does not subtract anything for it.
pub(super) fn odd_planes(c: &OddClass) -> (u64, u64) {
    (24 * lg_ceil(c.vals.len() as u128), 0)
}

pub(super) fn widths_of_even(c: &EvenClass) -> Widths {
    let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
    let n0 = 24 - c.w - free_n;
    // The five radices of the even composition, in the order `Indexer::encode`
    // multiplies them: codeword, on-support arrangement, off-support
    // arrangement, word signs, free signs.
    let arr_on = multiset(u128::from(c.w), c.word_vals.iter().map(|&(_, n)| n));
    let arr_off = multiset(
        u128::from(24 - c.w),
        c.free_vals
            .iter()
            .map(|&(_, n)| n)
            .chain(std::iter::once(n0 as u8)),
    );
    let s_w = if c.w == 0 { 1u128 } else { 1u128 << (c.w - 1) };
    let s_f = 1u128 << free_n;
    let radices = [u128::from(gamma(c.w)), arr_on, arr_off, s_w, s_f];
    let rounded: u64 = radices.iter().map(|&r| lg_ceil(r)).sum();
    let exact = lg_ceil(radices.iter().product::<u128>());
    let (a, sg) = even_planes(c);
    let levels = c.n_levels();
    let planes = 24 * lg_ceil(levels as u128);
    let sign_field = even_sign_bits(c);
    Widths {
        // `e1v-séparé` is this same expression — one `lg_ceil` per stage of
        // `Indexer::encode`'s composition — read through the variant table
        // rather than recomputed. See the T2 section of the module header.
        radix2: HEADER_BITS + rounded,
        radix2_g12: HEADER_BITS + rounded - lg_ceil(u128::from(gamma(c.w))) + GOLAY_RANK_BITS,
        perslot: HEADER_BITS + planes + u64::from(24 - n0),
        golay70: HEADER_BITS + GOLAY_RANK_BITS + 24 + 24,
        golay_tight: HEADER_BITS + GOLAY_RANK_BITS + a + sg,
        golay_signs: HEADER_BITS + planes + sign_field,
        sign_field,
        exact,
    }
}

pub(super) fn widths_of_odd(c: &OddClass) -> Widths {
    let arr = multiset(24, c.vals.iter().map(|&(_, n)| n));
    let radices = [4096u128, arr];
    let rounded: u64 = radices.iter().map(|&r| lg_ceil(r)).sum();
    let (a, sg) = odd_planes(c);
    let levels = c.n_levels();
    let planes = 24 * lg_ceil(levels as u128);
    Widths {
        radix2: HEADER_BITS + rounded,
        // The odd codeword field is already the flat 4096 rank: 12 bits, the
        // same field `golay70` carries, so the two variants coincide here.
        radix2_g12: HEADER_BITS + rounded,
        // The odd coset has no zeros, so every slot carries a sign — except
        // that on this coset signs are forced, hence none are stored.
        //
        // ⚠️ Forced *given the codeword*, which the level planes do not carry:
        // this column is 12 bits below anything a decoder could invert. Frozen
        // as published and pinned by
        // `golay_signs_is_the_first_perslot_variant_decodable_on_the_odd_coset`;
        // `golay_signs` below is the sound counterpart.
        perslot: HEADER_BITS + planes,
        golay70: HEADER_BITS + GOLAY_RANK_BITS + 24 + 24,
        golay_tight: HEADER_BITS + GOLAY_RANK_BITS + a + sg,
        // The 12 bits `perslot` is missing, stored as the Golay *message*:
        // the kernel re-encodes it to `c` and reads the signs off `c ^ f`.
        golay_signs: HEADER_BITS + planes + GOLAY_MSG_BITS,
        sign_field: GOLAY_MSG_BITS,
        exact: lg_ceil(radices.iter().product::<u128>()),
    }
}

/// `n! / Π cᵢ!` — the multiset permutation count, in `u128` because 24! does
/// not fit `u64`.
pub(super) fn multiset(n: u128, counts: impl Iterator<Item = u8>) -> u128 {
    let fact = |k: u128| (1..=k).product::<u128>();
    let mut m = fact(n);
    for c in counts {
        m /= fact(u128::from(c));
    }
    m
}

/// One variant's identity: how it decodes, and whether that decode is
/// admissible for a fused kernel at all.
pub(super) struct Variant {
    pub(super) name: &'static str,
    /// Bits per block of this class.
    pub(super) get: fn(&Widths) -> u64,
    /// Can every field be extracted **and interpreted** without a
    /// data-dependent loop?
    pub(super) shift_only: bool,
    /// Bounded decode depth, in dependent steps. `None` = unbounded/serial.
    pub(super) depth: Option<u32>,
    pub(super) note: &'static str,
}

/// Which of the spec's **two** clauses a variant's decoder satisfies. Not a
/// label: this is what the verdict branches on, so that a bit count alone can
/// never carry a point into the admitted set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Admissibility {
    /// Shift-only **and** fixed depth ≤ [`MAX_DEPTH`] — the conjunction the
    /// spec asks for. Only these may be measured against [`E3_CRITERION`] as a
    /// pass.
    Admissible,
    /// Shift-only, but the decode is deeper than the spec's clause. A bit count
    /// under the criterion here is a **reopening of the depth clause**, and has
    /// to be published as one.
    DepthReopening,
    /// Not shift-only: a data-dependent serial chain, out of the question for a
    /// fused kernel whatever it costs in bits.
    Serial,
}

impl Variant {
    pub(super) fn admissibility(&self) -> Admissibility {
        match (self.shift_only, self.depth) {
            (false, _) => Admissibility::Serial,
            (true, Some(d)) if d <= MAX_DEPTH => Admissibility::Admissible,
            // `None` is an unbounded depth, which is *worse* than 96 and must
            // not fall through to `Admissible` on the strength of `shift_only`.
            (true, _) => Admissibility::DepthReopening,
        }
    }
}

pub(super) const VARIANTS: &[Variant] = &[
    Variant {
        name: "archive (le fichier)",
        get: |_| ARCHIVE_BITS,
        shift_only: false,
        depth: None,
        note: "~509 serial ops, 8.27 ns/block measured, 75× a mask layout",
    },
    Variant {
        name: "radix2",
        get: |w| w.radix2,
        shift_only: false,
        depth: None,
        note: "fields extracted by shift, multiset ranks still serial",
    },
    Variant {
        name: "radix2 + golay 12 b",
        get: |w| w.radix2_g12,
        shift_only: false,
        depth: None,
        note: "same, uniform codeword field (offset independent of the class)",
    },
    Variant {
        name: "golay_tight",
        get: |w| w.golay_tight,
        shift_only: true,
        depth: Some(24),
        note: "A and sign planes cut to what the class needs, variable width",
    },
    Variant {
        name: "golay70 (mesuré, écarté)",
        get: |w| w.golay70,
        shift_only: true,
        depth: Some(24),
        note: "3.589 b/weight and 1.31× vs FP16, under the 1.6× criterion",
    },
    Variant {
        // ⚠️ NOT `perslot (= Planes)`, which is how this row was labelled until
        // 2026-08-13: the served `Planes14` is a flat 112 b/block = 4.804
        // b/weight, and this row is neither that width nor that price.
        name: "perslot (PAS Planes14)",
        get: |w| w.perslot,
        shift_only: true,
        depth: Some(24),
        note: "plane GEOMETRY, NOT the served layout: Planes14 is the flat 112 b/block \
               (4.804 b/weight); WARNING: perslot undercharges 12 b on the 157 odd classes",
    },
    // ---- T2 (prereg §3): the exact rank per class. Appended, never
    // inserted: the six rows above are indexed positionally by the
    // non-regression control of the 2026-08-12 log.
    Variant {
        name: "e1v-packé",
        get: |w| w.exact + HEADER_BITS,
        shift_only: false,
        depth: Some(96),
        note: "one field only: extracting the sub-ranks needs divmods by magic constants",
    },
    Variant {
        name: "e1v-séparé",
        // Literally `radix2`: one ⌈log₂⌉ per composition stage IS what that
        // field computes. Read, not recomputed — the identity is the T2 result
        // and it is pinned by `e1v_split_is_radix2_bit_for_bit`. What differs
        // is the decode assumed on top of it, i.e. the three fields below.
        get: |w| w.radix2,
        shift_only: true,
        depth: Some(96),
        note: "width = radix2 to the bit; fixed-count binomial walk, 48-96 steps. \
               WARNING: REOPENING of the spec's ≤ 24 clause, not a variant that satisfies it",
    },
    // ---- T3 (prereg §3): the signs rebuilt from the Golay codeword.
    Variant {
        name: "golay_signs",
        get: |w| w.golay_signs,
        shift_only: true,
        depth: Some(24),
        note: "perslot planes + signs at the class tariff: 12 b Golay message (odd), \
               2^(w−1)·2^free_n (even)",
    },
];

/// Number of priced variants — the width of every per-block width vector.
pub(super) const NVAR: usize = VARIANTS.len();
