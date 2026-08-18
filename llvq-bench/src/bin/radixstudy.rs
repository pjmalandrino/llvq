//! **X4 — the E3 study, on paper.** What an index designed *for the decoder*
//! would cost, priced exactly on the 384 classes and, when an artifact is
//! given, weighted by the real blocks of the published 4B.
//!
//! `docs/archive/spec-memoire-extreme-2026-08-12.md` opens the E3 chantier only if a
//! decomposition exists that is **both**
//!
//! * ≤ **2.6 b/weight** in the kernel accounting, and
//! * **shift-only at bounded depth** — every field extracted by shift/mask and
//!   interpreted without a data-dependent loop, ≤ 24 steps.
//!
//! This bench answers that question with counts rather than opinion, for 0 $
//! and no card. It cannot answer whether a shift-only decoder is *fast* — the
//! dossier's own rule is that no conclusion about speed survives without SASS
//! — but it can prove that no admissible decomposition reaches the bit target,
//! which is the cheap half of the verdict and the one that closes the chantier.
//!
//! ## Why the archive itself is not the answer
//!
//! The file is 48 bits per block and would be 2.15 b/weight in the kernel
//! accounting — under the criterion, and it is *already written*. It is
//! disqualified by the other half: unranking a mixed-radix multiset rank is a
//! serial, data-dependent chain of ~509 operations per block (`decfast`),
//! measured on GPU at **8.27 ns/block against 0.11** for a mask layout — 75×.
//! Every variant below is an attempt to buy that chain back with bits.
//!
//! ## The menu
//!
//! | variant | what changes | shift-only |
//! |---|---|---|
//! | `archive` | nothing — the file's own index | ❌ ~509 serial ops |
//! | `radix2` | every composition radix rounded up to a power of two | ❌ fields extract by shift, multiset ranks still unrank serially |
//! | `radix2+g12` | same, codeword as a uniform 12-bit rank | ❌ same |
//! | `perslot` | the arrangement replaced by 24 per-slot level fields | ✅ the **geometry** of `Planes14`/`Planes12x`, **not** their price — see below |
//! | `golay70` | the measured E2 layout | ✅ depth 24, **1.31× — écarté** |
//! | `golay_tight` | `golay70` with the A plane and the sign plane cut to what each class actually needs | ✅ depth 24, variable width |
//! | `e1v-packé` | the exact class rank in one field (T2) | ❌ divmod by magic constants to split the sub-ranks |
//! | `e1v-séparé` | one `⌈log₂⌉` per composition stage (T2) | ✅ **depth 48–96**, a reopening — see below |
//! | `golay_signs` | the `perslot` planes with the sign field cut to what the class really charges (T3) | ✅ depth 24 + a 12-XOR Golay encode |
//!
//! `golay_tight` was the only genuinely new point of the X4 lot, and it is the
//! one that lot turned on: `Golay70` spends a flat 24 bits on its A plane and
//! 24 on its sign plane, while a class whose residues each hold a single
//! magnitude needs **no** A bit at all, and the odd coset needs no sign bit
//! ever. Those are per-class counts, so they are exact here.
//!
//! ## T2 — `e1v`, and why `e1v-séparé` is `radix2` wearing a different decoder
//!
//! `pre-registration §3 T2` asks for the class rank at its **exact** width
//! (`⌈log₂|class|⌉`, the `exact` column) plus the 10-bit header, in two
//! sub-variants. The second one turns out to be a width this bench already
//! computed:
//!
//! * `e1v-packé` = `exact + 10`. One field at the width of the *product* of
//!   the composition radices. Splitting it back into `(g, r_on, r_off, r_sw,
//!   r_sf)` is a chain of divisions by radices that are **not** powers of two
//!   (`γ(w) = 2576`, `a_on`, `a_off`), so a kernel would need magic-constant
//!   divmods. Not shift-only, and this bench says so rather than rounding the
//!   inconvenience away.
//! * `e1v-séparé` = one `⌈log₂⌉` **per composition stage**, which is exactly
//!   what [`Widths::radix2`] already was — see [`widths_of_even`], where
//!   `rounded` sums `lg_ceil` over the five radices of `Indexer::encode`
//!   (`llvq-search/src/index.rs:300`), and [`widths_of_odd`] over its two
//!   (`index.rs:327`). **Same width, to the bit, class by class**, pinned by
//!   [`tests::e1v_split_is_radix2_bit_for_bit`]; so it is priced by *reading
//!   the same field*, not by a second copy of the arithmetic.
//!
//! What differs is the **interpretation**, and it is the whole of T2:
//!
//! * `radix2` keeps the archive's multiset-permutation rank, whose inverse
//!   (`perm_unrank`, `index.rs:97`) walks slots and kinds serially with a
//!   data-dependent inner loop — the ~509-op chain. Hence `shift_only: false`.
//! * `e1v` assumes the transcoder **re-bijects** that rank into a
//!   combinatorial number system, whose inverse is a fixed-count binomial
//!   walk: no division, no data-dependent branch, but 48–96 dependent steps
//!   (two arrangements of ≤ 24 slots, plus the sign and codeword fields).
//!
//! ⚠️ **48–96 steps is beyond the spec's own `≤ 24` clause**, and this bench
//! does not disguise it: `depth: Some(96)` is declared as written, and the
//! variant's note names it a **reopening** of the depth bound rather than a
//! variant that satisfies it. Whether that reopening is granted is a decision
//! for the passation, not for a counter of bits.
//!
//! **And the verdict enforces that, rather than merely printing it.** The
//! spec's criterion is a **conjunction** — ≤ 2.6 b/weight **and** a fixed-depth
//! decoder of ≤ 24 steps — so a point that clears the bit half while blowing
//! the depth half is classified [`Admissibility::DepthReopening`] and never
//! [`Admissibility::Admissible`]. The sentence [`OPEN_AT_SPEC`] is reserved for
//! the conjunction and cannot be reached otherwise;
//! [`tests::the_verdict_needs_both_clauses_not_shift_only_alone`] and
//! [`tests::a_depth_reopening_never_prints_the_spec_admission_sentence`] fail
//! if anyone restores a decision taken on [`Variant::shift_only`] alone — which
//! is exactly what this binary did until 2026-08-13, and it published
//! "E3 est ouvert au sens de la spec" on a `depth: Some(96)` point.
//!
//! ## T3 — `golay_signs`, and the sign bits a class really charges
//!
//! The served layouts spend a flat **24-bit sign mask**. On the odd coset
//! those 24 bits hold 12 bits of information: signs are forced by membership,
//! `neg = c ^ f` with `f` the mod-4 flag of the magnitude the level planes
//! already give (`llvq-cuda/kernels/llvq_golay.cuh:33` and `:178`,
//! `d.nw = odd ? (cw ^ fw) : f.bm`; the rule itself is
//! `llvq-search/src/index.rs:426-430`). Since the mask determines `c = neg ^ f`
//! and `c` ranges over the 4096 codewords, storing the **12-bit Golay message**
//! and re-encoding it (12 conditional XORs of generator constants, no table)
//! carries the same information in half the bits.
//!
//! On the even coset the parity constraint frees exactly one sign — and **not
//! the last nonzero slot**. `Indexer` charges `S_w · S_f = 2^(w−1) · 2^free_n`
//! (`index.rs:138-139`, mirrored by `EvenClass::cardinality`,
//! `llvq-search/src/classes.rs:112-113`): the freed sign is on the **codeword
//! support**, so a class with `w = 0` — all its nonzeros are free values — pays
//! `free_n` bits, not `free_n − 1`. [`even_sign_bits`] is that exponent, and
//! [`tests::even_sign_bits_is_the_exponent_of_the_two_sign_radices`] pins it
//! against `S_w · S_f` class by class.
//!
//! ## ⚠️ Two optimistic slacks in the frozen variants, pinned rather than fixed
//!
//! Deriving [`even_sign_bits`] from the cardinality exposed two places where
//! the variants frozen by the 2026-08-12 X4 log charge **less** than a
//! decodable layout would. Neither is repaired here: the pre-registration makes
//! reproducing that log (`3.0444` on `golay_tight`, `4.804` on `Planes14`) the
//! condition for any column of this lot being readable, and a variant that no
//! longer reproduces it would invalidate the lot to fix a number that is
//! already red. Both are therefore **asserted as they stand**, with their sign:
//!
//! 1. `golay_tight` (via [`even_planes`]) charges `nonzero − 1` sign bits,
//!    which is right whenever `w > 0` and **one bit short** on the `w = 0`
//!    even classes — 17 of the 226. Bounded to exactly one bit, on exactly
//!    those classes —
//!    [`tests::the_frozen_even_sign_plane_undercharges_w0_classes_by_one_bit`].
//! 2. `perslot` charges **zero** sign bits on the odd coset. The signs are
//!    forced *given the codeword*, but the codeword is 12 bits the level
//!    planes do not carry, so that column is 12 bits below what any decoder
//!    needs — on **all 157** odd classes.
//!    ⚠️ A width bound is necessary, not sufficient: it catches only the 10
//!    classes where the missing 12 bits push the row under `⌈log₂|class|⌉`,
//!    the other 147 having enough slack in `24·⌈log₂L⌉` to stay above a floor
//!    they still cannot invert. Counting bits cannot see the rest, which is
//!    why the mutation check inside
//!    [`tests::golay_signs_matches_the_class_cardinality`] exists at all —
//!    and [`tests::golay_signs_is_the_first_perslot_variant_decodable_on_the_odd_coset`]
//!    pins both numbers.
//!    `golay_signs` is the sound counterpart: the 12 bits it adds there are
//!    precisely the ones `perslot` omits.
//!
//! Both slacks make their variant look **cheaper** than it is, and both
//! variants are already over every threshold, so no verdict of this lot can be
//! reversed by them. That is why they can be pinned and reported instead of
//! silently corrected.
//!
//! ## ⚠️ `perslot` is the served layouts' **geometry**, never their **price**
//!
//! The row was labelled `perslot (= Planes)` until 2026-08-13, which invites
//! reading its 4.15 b/weight as "`Planes14` costs 4.15". It does not:
//! `Planes14` is a **flat 112 bits per block**, i.e. `4.804` b/weight, and the
//! whole gap is the two things this row does *not* pay for — a per-class field
//! width instead of a uniform stride, and the 12 bits of codeword it omits on
//! all 157 odd classes (slack 2 above). Every class is strictly narrower than
//! 112 bits here, so no row of this column is ever a served layout's cost;
//! [`tests::perslot_is_never_the_served_planes14_price`] pins both halves and
//! the label.
//!
//! ## ⚠️ The origin tariff is wrong for fixed-width variants, and asserted
//!
//! An out-of-ball block — the origin — is charged
//! `(v.get)(&Widths::default()).max(HEADER_BITS)`, which gives every
//! **fixed-width** variant a 10-bit origin: `golay70` is a flat 70-bit format
//! and would be charged 10. Both sealed artifacts contain **zero** origin
//! blocks (asserted, not assumed — [`tests::the_sealed_4b_reproduces_the_published_class_major_verdict`]),
//! so no published number depends on it, and repairing a tariff no file
//! exercises would be a change nothing could check. Instead [`main`] refuses to
//! print a table for a file that has any, by name and with the fix to make —
//! [`tests::the_origin_tariff_is_wrong_for_fixed_width_variants`] pins the
//! defect and its direction.
//!
//! ## Two addressing modes, and why a width alone is not a cost
//!
//! A variable-width stream does not have one price: what the kernel pays
//! depends on how a lane finds its own block. `proofs/preregistration-2026-08-13.md`
//! §1 freezes the two modes this bench prices, and neither is a free
//! parameter:
//!
//! * [`Mode::Grp32Max`] — `32 · ⌈max_of_group/8⌉ · 8 + 32`. Uniform stride at
//!   the group's widest block, so a lane's offset is `lane · stride`: one
//!   multiply, no cooperation. This is the only mode the dossier had ever
//!   priced.
//! * [`Mode::WarpScan`] — `Σ widths of the group`, rounded up to a 32-bit
//!   word, `+ 32` bits of base offset. Every block pays exactly its own width
//!   and the intra-group offsets come out of a warp prefix sum. Never priced
//!   before, for any layout.
//!
//! The `+ 32` and the word rounding commute: `⌈Σ/32⌉·32 + 32` is
//! `⌈(Σ+32)/32⌉·32`, because the base offset is itself one whole word — so
//! the pre-registration's phrasing has only one reading. Pinned by
//! [`tests::warp_scan_is_the_word_rounded_sum_plus_a_base_word`].
//!
//! ## Two scan orders, and why the old one was a control and not an answer
//!
//! [`Order::ClassMajor`] walks the class histogram: 32 blocks of a group are
//! then almost always the *same* class, so the group's maximum is its own
//! width and the stride is as tight as it can ever be. That is the column the
//! X4 log of 2026-08-12 published, and it is **optimistic by construction** —
//! the binary said so, and the pre-registration forbids citing it in a
//! verdict. It is kept here, unchanged to the last bit, as the non-regression
//! control.
//!
//! [`Order::FileOrder`] forms groups from 32 blocks **consecutive in the
//! file**, which is what a kernel actually reads, and which mixes the 301
//! classes inside every group.
//!
//! ## The geometry of a "group of 32", verified rather than assumed
//!
//! Read off the layout that defines it rather than supposed:
//!
//! * a group is `E1C_GROUP = 32` blocks and block `b` sits at lane `b % 32` of
//!   group `b / 32` — `llvq-artifact/src/e1c.rs:204-205` (`transpose`);
//! * `b` indexes the block stream in the order the transcoder consumes it,
//!   `indices[b]` — `llvq-artifact/src/runtime.rs:491` (`transcode_planes14`);
//! * and that order is `RawMatrix::indices`, **row-major `d_out × (d_in/24)`**
//!   — `llvq-artifact/src/format.rs:199-201`.
//!
//! So a group is 32 **consecutive entries of `m.indices`**, and two things
//! follow that this sweep respects:
//!
//! 1. **A group never spans two matrices.** The repack is handed one matrix's
//!    stream at a time (`llvq-artifact/tests/e1c_format.rs:210-231` sweeps
//!    matrix by matrix in chunks of whole groups; `bin/rtbits:249-262` charges
//!    `blocks.div_ceil(32)` groups). The sweep therefore closes the open group
//!    at every matrix boundary and **counts** how many that truncates, so a
//!    file that did pay for padding lanes could not hide it.
//! 2. **A group does span two rows**, whenever `(d_in/24) % 32 ≠ 0`. Nothing
//!    in the layout aligns a group to a row, and this bench does not pretend
//!    otherwise; it prints the count of matrices where it happens.
//!
//! ## What is deliberately NOT in the menu
//!
//! * **Entropy coding of the rank.** Closed by measurement: the index's
//!   entropy is 46.6536 bits against 47 paid (`archive/verdicts-lot-b-2026-08-06.md`
//!   §B5). There is nothing there.
//! * **The 46-bit odd-coset variant.** Proposed and *refuted*: the codeword
//!   bit gives the residue of the **signed** value, not of `|x|`, so it cannot
//!   stand in for a magnitude bit. `golay_tight` is guarded against
//!   re-deriving it — see `the_refuted_46_bit_odd_variant_is_not_reachable`.
//!
//! ## Run
//!
//! `cargo run --release -p llvq-bench --bin radixstudy [path/to/model.llvq]`
//!
//! Without a path: every class weighted equally, plus the worst case — which
//! is what a fixed-stride format actually pays, and **no** file order, so no
//! verdict. With one: the real block histogram *and* the real block order of
//! that artifact, which is what decides.

use llvq_core::DIM;
use llvq_search::classes::{enumerate_classes, gamma, EvenClass, OddClass};
use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// Class id + gain, the header every runtime variant pays.
const HEADER_BITS: u64 = 10;
/// Bits of a Golay codeword **rank** among all 4096 — `GOLAY70_RANK_BITS`.
/// Resolved through the 16 KiB codeword table `Golay70` already passes to the
/// device (`llvq-cuda/kernels/llvq_golay.cuh:14-18`).
const GOLAY_RANK_BITS: u64 = 12;
/// Bits of a Golay **message**, the `[24, 12, 8]` code's dimension: the field
/// `golay_signs` stores in place of a 24-bit sign mask on the odd coset,
/// resolved by re-encoding (`m · g₂(x)` plus the parity extension —
/// `llvq-core/src/golay.rs:62-70`), i.e. 12 conditional XORs and no table.
///
/// Numerically equal to [`GOLAY_RANK_BITS`] because `|Golay| = 2¹²`, and
/// deliberately a separate constant: a rank costs a table lookup, a message
/// costs an encode, and the two are not interchangeable in a kernel. Pinned by
/// [`tests::a_golay_message_and_a_golay_rank_are_both_twelve_bits`].
const GOLAY_MSG_BITS: u64 = 12;
/// Index bits of the published `leech1c12` file, plus its gain bit.
const ARCHIVE_BITS: u64 = 47 + 1;
/// The spec's admission threshold for opening the E3 chantier, in the kernel
/// accounting. `S_spec` of the 2026-08-13 pre-registration, unchanged since
/// `spec-memoire-extreme-2026-08-12.md:185-188`.
const E3_CRITERION: f64 = 2.6;
/// The **other half** of the same criterion, and the one this binary ignored
/// until 2026-08-13: `spec-memoire-extreme-2026-08-12.md:186-187` opens the E3
/// chantier on a conjunction — "≤ 2,6 b/poids thesis projeté **ET** un décodeur
/// à profondeur fixe **≤ 24 étapes** sans état sériel inter-slot".
///
/// A bit count that clears [`E3_CRITERION`] with a deeper decoder is a
/// **reopening** of this clause, not an admission under it, and
/// [`Admissibility`] is what keeps the two apart in the decision rather than
/// only in the prose.
const MAX_DEPTH: u32 = 24;
/// `S_alt` — the same 32 GB rung re-derived at the (32 GB; 2 GB margin; KV q8;
/// 8k) triple, pre-registration §2.
///
/// ⚠️ A **candidate** threshold, not the project's: it is opposable only if the
/// product note of §5 keeps that triple, and that note does not exist. It
/// exists here so a result between the two can be *placed*, never so it can be
/// called a pass.
const S_ALT: f64 = 3.09;
/// Lanes a variable-width stream is grouped over — `llvq_artifact::e1c::E1C_GROUP`.
const GROUP: u64 = 32;
/// Bits of base offset a group pays, in **either** addressing mode: the word
/// that says where the group starts.
const BASE_OFFSET_BITS: u64 = 32;
/// Machine word a warp-scan group's payload rounds up to.
const WORD_BITS: u64 = 32;
/// The two thresholds as the pre-registration writes them in bits per block on
/// the 4B, kept verbatim so the run can show its own conversion against them.
/// They are *labels*, not the rule: the rule is stated in b/weight.
const PREREG_BITS: (f64, f64) = (58.86, 70.66);

/// `⌈log₂ n⌉`, and 0 for `n ≤ 1` — a field with one possible value costs
/// nothing, which is the whole point of `golay_tight`.
fn lg_ceil(n: u128) -> u64 {
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
type Key = (bool, [(i32, u8); MAX_LEVELS]);

fn canonical(mut kinds: Vec<(i32, u8)>, odd: bool) -> Key {
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut k = [(0i32, 0u8); MAX_LEVELS];
    for (i, &v) in kinds.iter().enumerate() {
        k[i] = v;
    }
    (odd, k)
}

fn key_of_levels(lv: &ClassLevels) -> Key {
    canonical(
        (0..lv.len).map(|i| (lv.values[i], lv.counts[i])).collect(),
        lv.odd,
    )
}

/// What one class costs under each variant. All widths are **payload bits per
/// block**, header included.
#[derive(Clone, Copy, Debug, Default)]
struct Widths {
    /// Rounded mixed-radix composition, class stored in 9 bits.
    radix2: u64,
    /// Same with a uniform 12-bit codeword field.
    radix2_g12: u64,
    /// Per-slot level fields — the `Planes` geometry.
    perslot: u64,
    /// `Golay70`, flat.
    golay70: u64,
    /// `Golay70` with the A and sign planes cut to this class's needs.
    golay_tight: u64,
    /// T3: the `perslot` level planes with the sign field cut to what the
    /// class cardinality charges — [`even_sign_bits`] on the even coset, a
    /// 12-bit Golay message on the odd one.
    golay_signs: u64,
    /// Sign bits alone of that variant, so the run can report the expected
    /// saving against the flat 24-bit mask the served layouts spend. Not a
    /// variant: a component, printed once.
    sign_field: u64,
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
    exact: u64,
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
fn even_sign_bits(c: &EvenClass) -> u64 {
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
fn even_planes(c: &EvenClass) -> (u64, u64) {
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
fn odd_planes(c: &OddClass) -> (u64, u64) {
    (24 * lg_ceil(c.vals.len() as u128), 0)
}

fn widths_of_even(c: &EvenClass) -> Widths {
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

fn widths_of_odd(c: &OddClass) -> Widths {
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
fn multiset(n: u128, counts: impl Iterator<Item = u8>) -> u128 {
    let fact = |k: u128| (1..=k).product::<u128>();
    let mut m = fact(n);
    for c in counts {
        m /= fact(u128::from(c));
    }
    m
}

/// One variant's identity: how it decodes, and whether that decode is
/// admissible for a fused kernel at all.
struct Variant {
    name: &'static str,
    /// Bits per block of this class.
    get: fn(&Widths) -> u64,
    /// Can every field be extracted **and interpreted** without a
    /// data-dependent loop?
    shift_only: bool,
    /// Bounded decode depth, in dependent steps. `None` = unbounded/serial.
    depth: Option<u32>,
    note: &'static str,
}

/// Which of the spec's **two** clauses a variant's decoder satisfies. Not a
/// label: this is what the verdict branches on, so that a bit count alone can
/// never carry a point into the admitted set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Admissibility {
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
    fn admissibility(&self) -> Admissibility {
        match (self.shift_only, self.depth) {
            (false, _) => Admissibility::Serial,
            (true, Some(d)) if d <= MAX_DEPTH => Admissibility::Admissible,
            // `None` is an unbounded depth, which is *worse* than 96 and must
            // not fall through to `Admissible` on the strength of `shift_only`.
            (true, _) => Admissibility::DepthReopening,
        }
    }
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "archive (le fichier)",
        get: |_| ARCHIVE_BITS,
        shift_only: false,
        depth: None,
        note: "~509 ops sérielles, 8,27 ns/bloc mesuré — 75× un layout à masques",
    },
    Variant {
        name: "radix2",
        get: |w| w.radix2,
        shift_only: false,
        depth: None,
        note: "champs extraits par shift, rangs de multiensemble encore sériels",
    },
    Variant {
        name: "radix2 + golay 12 b",
        get: |w| w.radix2_g12,
        shift_only: false,
        depth: None,
        note: "idem, champ de codeword uniforme (offset indépendant de la classe)",
    },
    Variant {
        name: "golay_tight",
        get: |w| w.golay_tight,
        shift_only: true,
        depth: Some(24),
        note: "plans A et signes réduits à ce que la classe exige — largeur variable",
    },
    Variant {
        name: "golay70 (mesuré, écarté)",
        get: |w| w.golay70,
        shift_only: true,
        depth: Some(24),
        note: "3,589 b/poids et 1,31× vs FP16, sous le critère de 1,6×",
    },
    Variant {
        // ⚠️ NOT `perslot (= Planes)`, which is how this row was labelled until
        // 2026-08-13: the served `Planes14` is a flat 112 b/block = 4,804
        // b/weight, and this row is neither that width nor that price.
        name: "perslot (PAS Planes14)",
        get: |w| w.perslot,
        shift_only: true,
        depth: Some(24),
        note: "GÉOMÉTRIE des plans, PAS le layout servi : Planes14 est le 112 b/bloc plat \
               (4,804 b/poids) ⚠️ et perslot sous-facture 12 b sur les 157 classes impaires",
    },
    // ---- T2 (pré-enr. §3) : le rang exact par classe. Appended, never
    // inserted: the six rows above are indexed positionally by the
    // non-regression control of the 2026-08-12 log.
    Variant {
        name: "e1v-packé",
        get: |w| w.exact + HEADER_BITS,
        shift_only: false,
        depth: Some(96),
        note: "un seul champ : extraire les sous-rangs exige des divmod par constantes magiques",
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
        note: "largeur = radix2 au bit près ; marche binomiale à compte fixe, 48-96 pas \
               — ⚠️ RÉOUVERTURE de la clause ≤ 24 du spec, pas une variante qui la respecte",
    },
    // ---- T3 (pré-enr. §3) : les signes reconstruits par le mot de Golay.
    Variant {
        name: "golay_signs",
        get: |w| w.golay_signs,
        shift_only: true,
        depth: Some(24),
        note: "plans perslot + signes au tarif de la classe : message Golay 12 b (impair), \
               2^(w−1)·2^free_n (pair)",
    },
];

/// Number of priced variants — the width of every per-block width vector.
const NVAR: usize = VARIANTS.len();

// ---------------------------------------------------------------------------
// Addressing: the two ways a lane finds its block, priced side by side
// ---------------------------------------------------------------------------

/// One variant's group accumulator, carrying **both** addressing modes at
/// once: they see the same block sequence and differ only in what a closed
/// group costs, so pricing them separately would be two sweeps of the file for
/// no reason.
///
/// A partial group — one closed early by a matrix boundary — is charged the
/// way each mode really behaves, and the two disagree:
///
/// * `Grp32Max` pays **32 full lanes** regardless. The stride is uniform and
///   the padding lanes are addressable whether or not they hold a block; this
///   is also what `E1c` does, padding its trailing group with origin blocks
///   and charging them (`bin/rtbits:245-252`).
/// * `WarpScan` pays only the widths it actually holds, rounded to a word.
///   Nothing is bit-sliced across lanes here, so an absent block occupies no
///   bits.
///
/// The distinction is inert on both sealed artifacts as they stand — **measured**
/// by a header scan of each: every matrix's block count is `d_out · (d_in/24)`
/// with `d_out` a multiple of 32 (1024, 2560, 4096, 9728, 12288), so no group
/// is ever partial. That is a property of those two files, not of the format,
/// which is exactly why the sweep *counts* partial groups instead of assuming
/// there are none. Pinned for the 4B by
/// [`tests::the_sealed_4b_reproduces_the_published_class_major_verdict`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Group {
    /// Lanes filled in the group being built.
    filled: u64,
    /// Widest block of the group being built.
    gmax: u64,
    /// Sum of widths of the group being built.
    gsum: u64,
    /// Total bits over closed groups, `Grp32Max`.
    max_bits: u64,
    /// Total bits over closed groups, `WarpScan`.
    scan_bits: u64,
    /// Groups closed, partial ones included.
    groups: u64,
    /// Groups closed with fewer than 32 lanes.
    partial: u64,
}

impl Group {
    fn push(&mut self, w: u64) {
        self.gmax = self.gmax.max(w);
        self.gsum += w;
        self.filled += 1;
        if self.filled == GROUP {
            self.close();
        }
    }

    /// Close the group being built, if any. Idempotent on an empty one, so a
    /// matrix boundary that happens to fall on a group boundary costs nothing
    /// and counts nothing.
    fn close(&mut self) {
        if self.filled == 0 {
            return;
        }
        self.max_bits += GROUP * self.gmax.div_ceil(8) * 8 + BASE_OFFSET_BITS;
        self.scan_bits += self.gsum.div_ceil(WORD_BITS) * WORD_BITS + BASE_OFFSET_BITS;
        self.groups += 1;
        self.partial += u64::from(self.filled < GROUP);
        self.filled = 0;
        self.gmax = 0;
        self.gsum = 0;
    }

    fn bits(&self, mode: Mode) -> u64 {
        match mode {
            Mode::Grp32Max => self.max_bits,
            Mode::WarpScan => self.scan_bits,
        }
    }
}

/// How a lane finds its own block inside a group of 32.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Uniform byte stride at the group's widest block.
    Grp32Max,
    /// Exact widths, offsets by warp prefix sum.
    WarpScan,
}

impl Mode {
    const ALL: [Mode; 2] = [Mode::Grp32Max, Mode::WarpScan];
    fn tag(self) -> &'static str {
        match self {
            Mode::Grp32Max => "max",
            Mode::WarpScan => "scan",
        }
    }
}

/// In which order groups of 32 are formed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    /// Class histogram, ascending class id — the 2026-08-12 column. Groups are
    /// class-homogeneous almost everywhere, so the stride is as tight as it
    /// can ever be. **Optimistic by construction**; kept as the control.
    ClassMajor,
    /// 32 consecutive blocks of the file, groups closed at every matrix
    /// boundary — what a kernel reads.
    FileOrder,
}

impl Order {
    fn tag(self) -> &'static str {
        match self {
            Order::ClassMajor => "CM",
            Order::FileOrder => "FO",
        }
    }
}

/// Column label of one (order, mode) pair — the four combinations, named the
/// same way everywhere they are printed.
fn col(o: Order, m: Mode) -> String {
    format!("{}/{}", o.tag(), m.tag())
}

// ---------------------------------------------------------------------------
// The kernel accounting
// ---------------------------------------------------------------------------

/// The shapes the kernel accounting divides by. Mirrors `bin/rtbits`' `Shapes`
/// and `planesbench`' `Mat`: `weights` is **every** weight including the tail
/// columns, `tail_weights` the `KeepExact` columns kept as `f32` on the
/// device, `rows` the `f32` row scales.
///
/// Read from the file when there is one, so the 8B arm of T2bis is priced on
/// the 8B rather than on remembered 4B constants — which is a real trap: the
/// same stream is 3.05 b/weight on one and something else on the other.
/// [`tests::the_sealed_4b_reproduces_the_published_class_major_verdict`] pins
/// that the 4B file reproduces the constants below exactly.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Shapes {
    weights: u64,
    tail_weights: u64,
    rows: u64,
    blocks: u64,
}

impl Shapes {
    /// Qwen3-4B, the published `leech1c12` file. Same constants as `rtbits`'
    /// acceptance test.
    const fn qwen3_4b() -> Self {
        Self {
            weights: 3_633_315_840,
            tail_weights: 16_957_440,
            rows: 1_105_920,
            blocks: 150_681_600,
        }
    }

    fn push(&mut self, d_out: usize, d_in: usize) {
        self.weights += (d_out * d_in) as u64;
        self.tail_weights += (d_out * (d_in % DIM)) as u64;
        self.rows += d_out as u64;
        self.blocks += (d_out * (d_in / DIM)) as u64;
    }

    /// Kernel-accounting b/weight of a stream costing `bits_per_block`: the
    /// `f32` tail and row scales join the numerator, and the denominator is
    /// every weight. The one grandeur the 2.6 criterion is stated in.
    fn kernel_bpw(&self, bits_per_block: f64) -> f64 {
        let side = (self.tail_weights + self.rows) * 32;
        (bits_per_block * self.blocks as f64 + side as f64) / self.weights as f64
    }

    /// The inverse: what a b/weight threshold is worth in bits per block on
    /// **these** shapes. Printed next to the pre-registration's own figures so
    /// a rounding cannot pass for a disagreement.
    fn bits_for(&self, bpw: f64) -> f64 {
        let side = (self.tail_weights + self.rows) * 32;
        (bpw * self.weights as f64 - side as f64) / self.blocks as f64
    }
}

/// Kernel b/weight on the 4B shapes — the fixed conversion the criterion was
/// written against, kept as a free function because the tests pin it.
fn kernel_bpw(bits_per_block: f64) -> f64 {
    Shapes::qwen3_4b().kernel_bpw(bits_per_block)
}

// ---------------------------------------------------------------------------
// The sweeps
// ---------------------------------------------------------------------------

/// Per-class width vectors, flattened `class-major × NVAR`, plus the width an
/// **origin** block costs each variant.
///
/// Flat rather than `Vec<Vec<_>>` on purpose: the file sweep touches this once
/// per block, 150.7 M times, and a slice of `NVAR` contiguous `u64` is one
/// cache line rather than a pointer chase.
struct WidthTable {
    flat: Vec<u64>,
    origin: [u64; NVAR],
    per_class: Vec<Widths>,
}

impl WidthTable {
    fn build(fd: &FastDecoder) -> Self {
        let cs = enumerate_classes(13);
        let mut by_key: HashMap<Key, Widths> = HashMap::new();
        for c in &cs.even {
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
            let n0 = 24 - c.w - free_n;
            let mut kinds: Vec<(i32, u8)> = c
                .word_vals
                .iter()
                .chain(&c.free_vals)
                .map(|&(v, n)| (i32::from(v), n))
                .collect();
            if n0 > 0 {
                kinds.push((0, n0 as u8));
            }
            by_key.insert(canonical(kinds, false), widths_of_even(c));
        }
        for c in &cs.odd {
            let kinds = c.vals.iter().map(|&(v, n)| (i32::from(v), n)).collect();
            by_key.insert(canonical(kinds, true), widths_of_odd(c));
        }

        // Per-class widths in FastDecoder order, so the sweep is a direct index.
        let per_class: Vec<Widths> = (0..fd.n_classes())
            .map(|ci| {
                let lv = fd.levels(ci);
                *by_key
                    .get(&key_of_levels(lv))
                    .unwrap_or_else(|| panic!("classe {ci} absente de enumerate_classes : {lv:?}"))
            })
            .collect();
        let mut flat = Vec::with_capacity(per_class.len() * NVAR);
        for w in &per_class {
            for v in VARIANTS {
                flat.push((v.get)(w));
            }
        }
        // An origin block carries a class id, a gain bit and nothing else in
        // every *variable-width* runtime variant; in the archive it is still a
        // full index.
        //
        // ⚠️ And that is wrong for a **fixed-width** variant: `golay70` is a
        // flat 70-bit record whose origin would still be 70, not 10. Both
        // sealed files hold zero origin blocks, so nothing published depends on
        // this; `main` refuses by name to price a file that has any, rather
        // than repair a tariff no artifact exercises. Pinned with its direction
        // by `the_origin_tariff_is_wrong_for_fixed_width_variants`.
        let mut origin = [0u64; NVAR];
        for (k, v) in VARIANTS.iter().enumerate() {
            origin[k] = (v.get)(&Widths::default()).max(HEADER_BITS);
        }
        Self { flat, origin, per_class }
    }

    fn row(&self, ci: usize) -> &[u64] {
        &self.flat[ci * NVAR..(ci + 1) * NVAR]
    }
}

/// The class-major sweep, from the histogram alone.
///
/// **Bit-for-bit the 2026-08-12 loop**, and deliberately so: it is the control
/// that makes every other column readable. Classes ascending, `n == 0`
/// skipped, one continuous run of lanes across class boundaries (the group
/// being built is *not* closed between classes), one close at the end.
///
/// The one deviation, inert on both sealed files: the `origins` run is
/// appended to the sweep instead of being left out of the grouping entirely.
/// The published file has **zero** origin blocks, so the control reproduces to
/// the last bit; a file that had them would otherwise have been priced with
/// its origins in the mean and absent from the stride, which is not a stream
/// anybody could read.
fn sweep_class_major(t: &WidthTable, count: &[u64], origins: u64) -> Vec<Group> {
    let mut acc = vec![Group::default(); NVAR];
    for (ci, &n) in count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let ws = t.row(ci);
        for _ in 0..n {
            for (g, &w) in acc.iter_mut().zip(ws) {
                g.push(w);
            }
        }
    }
    for _ in 0..origins {
        for (g, &w) in acc.iter_mut().zip(&t.origin) {
            g.push(w);
        }
    }
    for g in &mut acc {
        g.close();
    }
    acc
}

/// What one pass over an artifact collects.
struct Sweep {
    /// Blocks per class — the histogram the means and the `exact` column need.
    count: Vec<u64>,
    /// Blocks whose index is outside the ball: the origin.
    origins: u64,
    /// File-order groups, one accumulator per variant.
    file: Vec<Group>,
    shapes: Shapes,
    matrices: u64,
    /// Matrices whose blocks-per-row is not a multiple of 32, i.e. where a
    /// group straddles two rows.
    row_straddling: u64,
}

impl Sweep {
    fn total(&self) -> u64 {
        self.count.iter().sum::<u64>() + self.origins
    }

    /// Fold one matrix into the histogram and into the file-order groups, then
    /// close the open group.
    ///
    /// Split out of [`sweep_file`] so the streaming glue can be exercised on
    /// **one** real matrix rather than only on the whole 981 MB archive: the
    /// per-block lookup, the group boundary and the matrix boundary all live
    /// here, and a sweep is the only other way to reach them.
    fn push_matrix(&mut self, fd: &FastDecoder, t: &WidthTable, m: &llvq_artifact::RawMatrix) {
        self.shapes.push(m.d_out, m.d_in);
        self.matrices += 1;
        self.row_straddling += u64::from(!((m.d_in / DIM) as u64).is_multiple_of(GROUP));
        for &idx in &m.indices {
            let ws = match fd.class_of(idx) {
                Some(ci) => {
                    self.count[ci] += 1;
                    t.row(ci)
                }
                None => {
                    self.origins += 1;
                    &t.origin[..]
                }
            };
            for (g, &w) in self.file.iter_mut().zip(ws) {
                g.push(w);
            }
        }
        // A group never spans two matrices — see the module header. Closing
        // here is what makes that true, and `Group::partial` is what says how
        // much it cost.
        for g in &mut self.file {
            g.close();
        }
    }

    fn empty(n_classes: usize) -> Self {
        Self {
            count: vec![0u64; n_classes],
            origins: 0,
            file: vec![Group::default(); NVAR],
            shapes: Shapes::default(),
            matrices: 0,
            row_straddling: 0,
        }
    }
}

/// One streaming pass over a sealed artifact: histogram **and** file-order
/// groups for every variant at once.
///
/// No `Vec` of 150.7 M widths anywhere — the widths are looked up per block
/// and folded straight into the accumulators, so the peak allocation is one
/// matrix's index vector, which `read_matrix_raw` allocates anyway.
fn sweep_file(fd: &FastDecoder, t: &WidthTable, path: &str) -> Sweep {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut r = BufReader::with_capacity(1 << 20, f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    let mut s = Sweep::empty(fd.n_classes());
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        s.push_matrix(fd, t, &m);
    }
    assert_eq!(s.matrices, u64::from(h.matrices), "matrices lues ≠ annoncées");
    s
}

/// Mean bits per block of one variant, from the histogram — the same order of
/// accumulation as the published run (classes ascending, then the origins).
fn mean_bits(t: &WidthTable, count: &[u64], origins: u64, k: usize) -> f64 {
    let mut sum = 0f64;
    for (ci, &n) in count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        sum += t.row(ci)[k] as f64 * n as f64;
    }
    sum += origins as f64 * t.origin[k] as f64;
    sum / (count.iter().sum::<u64>() + origins) as f64
}

fn worst_bits(t: &WidthTable, count: &[u64], origins: u64, k: usize) -> u64 {
    let mut worst = if origins > 0 { t.origin[k] } else { 0 };
    for (ci, &n) in count.iter().enumerate() {
        if n > 0 {
            worst = worst.max(t.row(ci)[k]);
        }
    }
    worst
}

fn main() {
    let fd = FastDecoder::new();
    let t = WidthTable::build(&fd);

    // ---- weights: real blocks if we have a file, else one per class ----
    let (s, source) = match std::env::args().nth(1) {
        Some(path) => {
            let s = sweep_file(&fd, &t, &path);
            let src = format!("{path} — {} matrices, blocs réels", s.matrices);
            (s, src)
        }
        None => {
            let s = Sweep {
                count: vec![1u64; fd.n_classes()],
                origins: 0,
                // No file, no file order: the columns print `—` rather than a
                // number that would be the class-major one wearing a costume.
                file: Vec::new(),
                // Nothing to read shapes from, so the criterion's own 4B
                // accounting — which is what the equal-weight arm has always
                // been priced in.
                shapes: Shapes::qwen3_4b(),
                matrices: 0,
                row_straddling: 0,
            };
            (s, "les 383 classes à poids égal (⚠️ PAS la distribution réelle)".to_string())
        }
    };
    let have_file = !s.file.is_empty();
    let total = s.total();
    let cm = sweep_class_major(&t, &s.count, s.origins);
    // The file's own shapes when there is one, the 4B constants when there is
    // not — set once, at construction, so the two can never drift apart.
    let shapes = s.shapes;

    println!("source : {source}");
    println!("{total} blocs, {} classes", fd.n_classes());
    // The origin tariff is `Widths::default()`, i.e. 10 bits, which is right
    // for a variable-width variant and **wrong** for a fixed-width one: an
    // origin in `golay70` still occupies its flat 70-bit record. No sealed
    // artifact has ever contained one, so rather than repair a tariff nothing
    // can check, refuse to print a table that would silently under-charge —
    // by name, and saying what to fix. See the module header.
    assert_eq!(
        s.origins,
        0,
        "{} blocs origine dans ce fichier, et le tarif origine de ce banc est FAUX pour les \
         variantes à LARGEUR FIXE : WidthTable::build lit (v.get)(&Widths::default()).max(10), \
         ce qui facture {} bits à « golay70 (mesuré, écarté) » au lieu de ses 70 bits plats. \
         Corriger le tarif origine variante par variante AVANT de publier une colonne sur ce \
         fichier — aucune ligne de ce run ne serait lisible",
        s.origins,
        t.origin[VARIANTS
            .iter()
            .position(|v| v.name.starts_with("golay70"))
            .expect("golay70 au menu")]
    );
    if have_file {
        let g = s.file[0];
        println!(
            "{} groupes de 32 en ordre-fichier, dont {} partiels (fermés en fin de matrice)",
            g.groups, g.partial
        );
        println!(
            "{}/{} matrices ont blocs/ligne non multiple de 32 : un groupe y enjambe deux lignes",
            s.row_straddling, s.matrices
        );
        println!(
            "comptabilité noyau LUE DANS LE FICHIER : {} poids, {} de queue, {} lignes, {} blocs",
            shapes.weights, shapes.tail_weights, shapes.rows, shapes.blocks
        );
        if shapes != Shapes::qwen3_4b() {
            // Two accountings in one dossier is exactly how a b/weight figure
            // goes wrong (CLAUDE.md §7, règle 1). Name the gap rather than let
            // a reader put this table beside a 4B one.
            println!(
                "⚠️ CE N'EST PAS la comptabilité du 4B : 112 b/bloc valent {:.4} ici contre {:.4}\n\
                 sur le 4B publié. Aucune ligne de ce run ne se compare à un chiffre 4B.",
                shapes.kernel_bpw(112.0),
                kernel_bpw(112.0)
            );
        }
        assert_eq!(
            shapes.blocks, total,
            "les blocs comptés ({total}) ne sont pas ceux des formes ({})",
            shapes.blocks
        );
    } else {
        println!("pas de fichier : aucun ORDRE réel, donc pas de colonne ordre-fichier ni de verdict");
    }

    // ---- geometry of the group, stated rather than assumed ----
    println!(
        "\n  géométrie du groupe (vérifiée, pas supposée) : 32 blocs CONSÉCUTIFS de m.indices,\n  \
         row-major d_out × (d_in/24) — e1c.rs:204, runtime.rs:491, format.rs:199. Un groupe ne\n  \
         franchit JAMAIS une matrice ; il franchit les lignes dès que (d_in/24) mod 32 ≠ 0."
    );

    // ---- the menu, priced: bits per block ----
    println!("\n  bits par bloc — les 4 combinaisons (ordre de balayage × adressage)");
    println!("  {}", "-".repeat(98));
    println!(
        "  {:<24}{:>7}{:>6}{:>9}{:>9}{:>9}{:>9}{:>7}  prof.",
        "variante",
        "moy",
        "pire",
        col(Order::ClassMajor, Mode::Grp32Max),
        col(Order::ClassMajor, Mode::WarpScan),
        col(Order::FileOrder, Mode::Grp32Max),
        col(Order::FileOrder, Mode::WarpScan),
        "shift"
    );
    for (k, v) in VARIANTS.iter().enumerate() {
        let per_block = |g: &Group, m: Mode| g.bits(m) as f64 / total as f64;
        let fo = |m: Mode| match have_file {
            true => format!("{:>9.2}", per_block(&s.file[k], m)),
            false => format!("{:>9}", "—"),
        };
        println!(
            "  {:<24}{:>7.2}{:>6}{:>9.2}{:>9.2}{}{}{:>7}  {}",
            v.name,
            mean_bits(&t, &s.count, s.origins, k),
            worst_bits(&t, &s.count, s.origins, k),
            per_block(&cm[k], Mode::Grp32Max),
            per_block(&cm[k], Mode::WarpScan),
            fo(Mode::Grp32Max),
            fo(Mode::WarpScan),
            if v.shift_only { "oui" } else { "NON" },
            v.depth.map_or("sér.".to_string(), |d| format!("{d}")),
        );
    }
    println!("  {}", "-".repeat(98));
    println!(
        "  CM = classe-majeur — CONTRÔLE de non-régression, optimiste par construction :\n  \
         un groupe y est presque toujours d'une seule classe. Interdit de citation (pré-enr. §1).\n  \
         FO = ordre du fichier — ce qu'un noyau lit. max = stride uniforme au plus large du\n  \
         groupe ; scan = largeurs exactes, offsets par somme préfixe de warp."
    );

    // ---- the same bits per block at 3 decimals, because the decision sections
    // quote 3 and the table above prints 2. A number inverted out of a
    // 4-decimal b/weight column is a transcription, not an output: this block
    // exists so every figure a verdict cites is something a command printed.
    if have_file {
        println!("\n  les MÊMES bits/bloc à 3 décimales — la table ci-dessus arrondit à 2, et un");
        println!("  chiffre à 3 décimales ne s'en déduit pas. Δ = le coût de l'ordre réel.");
        println!("  {}", "-".repeat(98));
        println!(
            "  {:<24}{:>10}{:>10}{:>10}{:>10}{:>14}",
            "variante",
            col(Order::ClassMajor, Mode::Grp32Max),
            col(Order::ClassMajor, Mode::WarpScan),
            col(Order::FileOrder, Mode::Grp32Max),
            col(Order::FileOrder, Mode::WarpScan),
            "Δ FO−CM max",
        );
        for (k, v) in VARIANTS.iter().enumerate() {
            let per_block = |g: &Group, m: Mode| g.bits(m) as f64 / total as f64;
            println!(
                "  {:<24}{:>10.3}{:>10.3}{:>10.3}{:>10.3}{:>+14.3}",
                v.name,
                per_block(&cm[k], Mode::Grp32Max),
                per_block(&cm[k], Mode::WarpScan),
                per_block(&s.file[k], Mode::Grp32Max),
                per_block(&s.file[k], Mode::WarpScan),
                per_block(&s.file[k], Mode::Grp32Max) - per_block(&cm[k], Mode::Grp32Max),
            );
        }
        println!("  {}", "-".repeat(98));
    }

    // ---- the same, in the grandeur the criterion is stated in ----
    let (sp, sa) = (shapes.bits_for(E3_CRITERION), shapes.bits_for(S_ALT));
    println!("\n  b/poids NOYAU — mêmes 4 combinaisons, et les deux seuils du pré-enregistrement");
    println!(
        "  S_spec = {E3_CRITERION:.2} b/poids ⇔ {sp:.2} bits/bloc (pré-enr. : {:.2})   ·   \
         S_alt = {S_ALT:.2} ⇔ {sa:.2} (pré-enr. : {:.2})",
        PREREG_BITS.0, PREREG_BITS.1
    );
    println!("  {}", "-".repeat(98));
    println!(
        "  {:<24}{:>9}{:>9}{:>9}{:>9}   {:<14}FO vs S_alt",
        "variante",
        col(Order::ClassMajor, Mode::Grp32Max),
        col(Order::ClassMajor, Mode::WarpScan),
        col(Order::FileOrder, Mode::Grp32Max),
        col(Order::FileOrder, Mode::WarpScan),
        "FO vs S_spec",
    );
    for (k, v) in VARIANTS.iter().enumerate() {
        let bpw = |g: &Group, m: Mode| shapes.kernel_bpw(g.bits(m) as f64 / total as f64);
        let fo = |m: Mode| match have_file {
            true => format!("{:>9.4}", bpw(&s.file[k], m)),
            false => format!("{:>9}", "—"),
        };
        let spec_col = || match have_file {
            false => "—         ".to_string(),
            true => format!(
                "{} max  {} scan",
                mark(bpw(&s.file[k], Mode::Grp32Max) <= E3_CRITERION),
                mark(bpw(&s.file[k], Mode::WarpScan) <= E3_CRITERION)
            ),
        };
        let alt_col = || match have_file {
            false => "—         ".to_string(),
            true => format!(
                "{} max  {} scan",
                mark_alt(bpw(&s.file[k], Mode::Grp32Max)),
                mark_alt(bpw(&s.file[k], Mode::WarpScan))
            ),
        };
        println!(
            "  {:<24}{:>9.4}{:>9.4}{}{}   {:<14}{}",
            v.name,
            bpw(&cm[k], Mode::Grp32Max),
            bpw(&cm[k], Mode::WarpScan),
            fo(Mode::Grp32Max),
            fo(Mode::WarpScan),
            spec_col(),
            alt_col(),
        );
    }
    println!("  {}", "-".repeat(98));
    for v in VARIANTS {
        // 26, not 24: two of the names are exactly 24 characters and would
        // otherwise run into their own note.
        println!("  {:<26}{}", v.name, v.note);
    }
    println!(
        "\n  LÉGENDE DES MARQUEURS — et le ⊘ est la moitié qui compte :\n  \
         ✅ = sous S_spec ({E3_CRITERION:.2}), le seul seuil du projet. C'est le seul marqueur\n  \
            qui veuille dire « ça passe », et il ne s'imprime que là.\n  \
         ⊘ = sous S_alt SEULEMENT, donc dans la bande ]{E3_CRITERION:.2} ; {S_ALT:.2}]. NE SE CITE\n  \
            JAMAIS SEUL : S_alt n'est pas le seuil du projet, il n'existe qu'à contexte 8k figé\n  \
            et ne deviendrait opposable que si la note produit du pré-enregistrement §5 retenait\n  \
            le triplet (32 Go ; 2 Go ; KV q8 ; 8k). CETTE NOTE N'EXISTE PAS. Un ⊘ se publie\n  \
            « ça passerait si et seulement si ce triplet était retenu », jamais « ça passe ».\n  \
         ❌ = au-dessus des deux."
    );

    // ---- where the archive's 47 bits actually go ----
    let exact_mean: f64 = s
        .count
        .iter()
        .enumerate()
        .map(|(ci, &n)| t.per_class[ci].exact as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    println!("\n  où passent les 47 bits de l'archive");
    println!("  {}", "-".repeat(98));
    println!("  point DANS sa classe, moyenne de ⌈log₂ |classe|⌉ : {exact_mean:.2} bits");
    println!(
        "  choix de la classe : 47 − {exact_mean:.2} ≈ {:.2} bits, que toute variante à champ\n  \
         de classe explicite repaie en {HEADER_BITS} bits d'en-tête",
        47.0 - exact_mean
    );
    println!("  e1v-packé paie donc {exact_mean:.2} + {HEADER_BITS} bits par bloc, hors adressage");

    // ---- T3: what the sign field costs, against the flat mask it replaces ----
    let sign_mean: f64 = s
        .count
        .iter()
        .enumerate()
        .map(|(ci, &n)| t.per_class[ci].sign_field as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    println!("\n  ce que le champ de signes de golay_signs coûte");
    println!("  {}", "-".repeat(98));
    println!(
        "  moyenne du champ : {sign_mean:.2} bits contre les 24 du masque plat que les layouts\n  \
         servis dépensent, soit {:.2} bits/bloc d'espérance gagnée (prédiction du pré-enr. §3 T3 : −9,85).\n  \
         ⚠️ Espérance ≠ ce qu'un flux paie : les colonnes max ci-dessus facturent le plus large\n  \
         du groupe, et c'est là que se juge T3.",
        24.0 - sign_mean
    );

    // ---- the verdict, taken on the file order only ----
    println!("\n  VERDICT X4");
    println!("  {}", "-".repeat(98));
    if !have_file {
        println!(
            "  ⏸️  Aucun verdict sans fichier. La seule colonne disponible est le contrôle\n  \
             classe-majeur, que le pré-enregistrement §1 interdit de citer dans un verdict."
        );
        return;
    }
    // Two buckets, because the spec asks a **conjunction** and a bit count
    // alone answers only half of it. Points that clear `E3_CRITERION` with a
    // decoder deeper than `MAX_DEPTH` go to `reopening`, never to `admissible`.
    let mut admissible: Vec<Point> = Vec::new();
    let mut reopening: Vec<Point> = Vec::new();
    let mut best = f64::INFINITY;
    for (k, v) in VARIANTS.iter().enumerate() {
        let cls = v.admissibility();
        if cls == Admissibility::Serial {
            continue;
        }
        for m in Mode::ALL {
            let bpw = shapes.kernel_bpw(s.file[k].bits(m) as f64 / total as f64);
            // "Best" is the best point that satisfies the WHOLE criterion —
            // taking the minimum over reopenings too would quote a number the
            // spec does not admit as if it were the state of the art.
            if cls == Admissibility::Admissible {
                best = best.min(bpw);
            }
            if bpw <= E3_CRITERION {
                let p = Point { name: v.name, mode: m, bpw, depth: v.depth };
                match cls {
                    Admissibility::Admissible => admissible.push(p),
                    _ => reopening.push(p),
                }
            }
        }
    }
    print!("{}", verdict_text(&admissible, &reopening, best));
}

/// One priced point of the verdict: a variant, an addressing mode, its cost and
/// the decode depth that decides which clause it answers.
struct Point {
    name: &'static str,
    mode: Mode,
    bpw: f64,
    depth: Option<u32>,
}

/// The one sentence that asserts the spec's **conjunction** is satisfied.
///
/// A constant so that [`tests::a_depth_reopening_never_prints_the_spec_admission_sentence`]
/// can require its absence: printing it for a point that only cleared the bit
/// half is precisely the over-claim of the 2026-08-13 journals, and it is now a
/// test failure rather than a proofreading duty.
const OPEN_AT_SPEC: &str = "Le chantier E3 est ouvert AU SENS DE LA SPEC";

/// The verdict, as text, so it can be asserted on rather than eyeballed.
///
/// `admissible` holds only points satisfying **both** clauses; `reopening`
/// holds points under the bit criterion whose decoder is deeper than
/// [`MAX_DEPTH`]. The two are never merged, and the second can never reach
/// [`OPEN_AT_SPEC`].
fn verdict_text(admissible: &[Point], reopening: &[Point], best: f64) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    if admissible.is_empty() {
        let _ = writeln!(
            o,
            "  ❌ AUCUNE décomposition ADMISSIBLE — shift-only ET profondeur fixe ≤ {MAX_DEPTH} \
             étapes,\n     la CONJONCTION que le spec exige — ne passe sous {E3_CRITERION} \
             b/poids noyau EN ORDRE-FICHIER,\n     ni en grp32-max ni en warp-scan."
        );
        if best.is_finite() {
            let _ = writeln!(
                o,
                "  Le meilleur point ADMISSIBLE vaut {best:.4} b/poids, soit {:.0} % au-dessus du seuil.",
                100.0 * (best / E3_CRITERION - 1.0)
            );
        }
        let _ = writeln!(
            o,
            "  Le critère d'ouverture du chantier E3 n'est PAS atteint : E3 s'enterre sur papier,\n  \
             pour 0 $, comme E2 s'est enterré au banc. La marche suivante de l'échelle mémoire\n  \
             reste E1c (X1/X2), et le plancher pratique du projet est le layout mesuré."
        );
    } else {
        let _ = writeln!(
            o,
            "  ✅ décomposition(s) ADMISSIBLES (shift-only ET profondeur ≤ {MAX_DEPTH}) sous le \
             critère,\n     en ordre-fichier :"
        );
        for p in admissible {
            let _ = writeln!(
                o,
                "     {:<24}{:<6}{:.4} b/poids noyau   (profondeur {})",
                p.name,
                p.mode.tag(),
                p.bpw,
                depth_tag(p.depth)
            );
        }
        let _ = writeln!(
            o,
            "  {OPEN_AT_SPEC}. ⚠️ Ce verdict porte sur les BITS\n  \
             seuls : la vitesse d'un tel décodeur reste non mesurée, et Golay70 rappelle qu'un\n  \
             format juste et compact peut mourir en ALU (195 Go/s contre 425)."
        );
    }
    if !reopening.is_empty() {
        let _ = writeln!(
            o,
            "\n  ⚠️ SOUS LE CRITÈRE DE BITS, MAIS HORS DE LA CLAUSE DE PROFONDEUR ≤ {MAX_DEPTH} :"
        );
        for p in reopening {
            let _ = writeln!(
                o,
                "     {:<24}{:<6}{:.4} b/poids noyau   (profondeur {})",
                p.name,
                p.mode.tag(),
                p.bpw,
                depth_tag(p.depth)
            );
        }
        let _ = writeln!(
            o,
            "  Ces points ne sont PAS admis : le critère d'ouverture d'E3\n  \
             (spec-memoire-extreme-2026-08-12.md:186-187) est une CONJONCTION — ≤ {E3_CRITERION} \
             b/poids\n  \
             ET un décodeur à profondeur fixe ≤ {MAX_DEPTH} étapes sans état sériel inter-slot.\n  \
             Ils constituent une RÉOUVERTURE DE LA CLAUSE DE PROFONDEUR, qui est une décision de\n  \
             passation et non un résultat de ce banc. Les publier comme une admission serait la\n  \
             sur-revendication que ce bloc existe pour empêcher."
        );
    }
    o
}

fn depth_tag(d: Option<u32>) -> String {
    d.map_or("sérielle".to_string(), |d| format!("{d}"))
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "✅"
    } else {
        "❌"
    }
}

/// Marker of the `S_alt` column, which has **three** states and not two.
///
/// The pre-registration §2 forbids publishing a result in `]S_spec ; S_alt]` as
/// a pass, but a bare ✅ in that band cites itself — it was printed there until
/// 2026-08-13. `⊘` is deliberately not a verdict glyph: it cannot be quoted
/// without its legend, which is the point.
fn mark_alt(bpw: f64) -> &'static str {
    if bpw <= E3_CRITERION {
        "✅"
    } else if bpw <= S_ALT {
        "⊘"
    } else {
        "❌"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path of the sealed 4B, overridable so a machine that keeps it elsewhere
    /// can still run the file tests. A variable may *move* the search, never
    /// satisfy it: a missing file is a failure, never a skip.
    fn sealed_4b() -> String {
        std::env::var("LLVQ_Q4B").unwrap_or_else(|_| {
            format!("{}/llvq-q4b.llvq", std::env::var("HOME").expect("HOME"))
        })
    }

    #[test]
    fn lg_ceil_is_the_field_width() {
        assert_eq!(lg_ceil(0), 0);
        assert_eq!(lg_ceil(1), 0, "un seul choix ne coûte aucun bit");
        assert_eq!(lg_ceil(2), 1);
        assert_eq!(lg_ceil(3), 2);
        assert_eq!(lg_ceil(4), 2);
        assert_eq!(lg_ceil(5), 3);
        assert_eq!(lg_ceil(4096), 12, "le rang de codeword tient en 12 bits");
        assert_eq!(lg_ceil(4097), 13);
    }

    /// Rounding radices to powers of two can only **cost** bits, never save
    /// any, and the waste is bounded by one bit per radix. Both halves matter:
    /// a variant that came out cheaper than the exact composition would be
    /// arithmetically impossible and must fail loudly.
    #[test]
    fn rounding_costs_between_zero_and_one_bit_per_radix() {
        let cs = enumerate_classes(13);
        for c in &cs.even {
            let w = widths_of_even(c);
            let rounded = w.radix2 - HEADER_BITS;
            assert!(rounded >= w.exact, "classe paire : arrondi sous l'exact");
            assert!(rounded <= w.exact + 5, "5 radices, 5 bits de gaspillage max");
        }
        for c in &cs.odd {
            let w = widths_of_odd(c);
            let rounded = w.radix2 - HEADER_BITS;
            assert!(rounded >= w.exact);
            assert!(rounded <= w.exact + 2, "2 radices côté impair");
        }
    }

    /// `golay_tight` may never beat `golay70` by more than the two planes it
    /// trims, and may never exceed it except where a class genuinely needs
    /// more than one bit per slot — the case `Golay70` refuses outright and
    /// pays for with an exception.
    #[test]
    fn golay_tight_only_trims_what_a_class_does_not_use() {
        let cs = enumerate_classes(13);
        let mut trimmed = 0;
        for c in &cs.even {
            let w = widths_of_even(c);
            assert!(w.golay_tight >= HEADER_BITS + GOLAY_RANK_BITS);
            if w.golay_tight < w.golay70 {
                trimmed += 1;
            }
        }
        assert!(trimmed > 0, "aucune classe paire ne profite du serrage");
    }

    /// **The refutation, encoded.** On the odd coset the codeword constrains
    /// the *signed* residue, so it cannot pay for a magnitude bit as well: a
    /// two-magnitude odd class must still spend 24 bits, giving 46 only if
    /// one also stopped storing something else. The dossier proposed 46 bits
    /// there and refuted it; this pins the refutation so no future edit can
    /// quietly re-derive it.
    #[test]
    fn the_refuted_46_bit_odd_variant_is_not_reachable() {
        let cs = enumerate_classes(13);
        for c in &cs.odd {
            let w = widths_of_odd(c);
            let (a, sg) = odd_planes(c);
            assert_eq!(sg, 0, "les signes impairs sont calculés, jamais stockés");
            assert_eq!(a, 24 * lg_ceil(c.vals.len() as u128));
            assert!(
                w.golay_tight >= 46,
                "classe impaire à {} bits — la variante 46 b réfutée est de retour",
                w.golay_tight
            );
        }
    }

    /// Index of a variant by name, so a test never hard-codes a position in
    /// [`VARIANTS`] — the six published rows are positional, everything after
    /// them is not.
    fn variant(name: &str) -> usize {
        VARIANTS
            .iter()
            .position(|v| v.name == name)
            .unwrap_or_else(|| panic!("variante « {name} » absente du menu"))
    }

    /// The `exact` column must be `⌈log₂ |class|⌉` of the **cardinality
    /// formula**, not of a model of it. Everything that follows — both
    /// "never narrower than the class rank" tests, and the 41.50 the E3
    /// verdict is argued from — is only as good as this.
    #[test]
    fn the_exact_column_is_the_class_cardinality() {
        let cs = enumerate_classes(13);
        let mut want_sum = 0u64;
        for c in &cs.even {
            let e = widths_of_even(c).exact;
            assert_eq!(e, lg_ceil(u128::from(c.cardinality())), "classe paire {c:?}");
            want_sum += e;
        }
        for c in &cs.odd {
            let e = widths_of_odd(c).exact;
            assert_eq!(e, lg_ceil(u128::from(c.cardinality())), "classe impaire {c:?}");
            want_sum += e;
        }
        // And the same column as the sweep reads it, through `FastDecoder`'s
        // class order: `every_fastdecoder_class_has_its_radices` proves the
        // keys are a bijection, so the two totals must coincide.
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let got: u64 = t.per_class.iter().map(|w| w.exact).sum();
        assert_eq!(got, want_sum, "l'ordre FastDecoder ne voit pas les mêmes classes");
    }

    /// [`even_sign_bits`] must be the exponent of the index's own two sign
    /// radices `S_w · S_f` (`index.rs:138-139`, `classes.rs:112-113`) — the
    /// only definition under which a sign field is invertible.
    #[test]
    fn even_sign_bits_is_the_exponent_of_the_two_sign_radices() {
        let cs = enumerate_classes(13);
        for c in &cs.even {
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
            let s_w: u128 = if c.w == 0 { 1 } else { 1 << (c.w - 1) };
            let s_f: u128 = 1 << free_n;
            assert_eq!(
                1u128 << even_sign_bits(c),
                s_w * s_f,
                "classe paire w={} free_n={free_n} : {} bits contre S_w·S_f",
                c.w,
                even_sign_bits(c)
            );
        }
    }

    /// A Golay message and a Golay rank are both 12 bits because `|Golay| =
    /// 2¹²`, and the bench keeps them as two constants because a rank costs a
    /// 16 KiB table and a message costs an encode.
    #[test]
    fn a_golay_message_and_a_golay_rank_are_both_twelve_bits() {
        let g = llvq_core::Golay::new();
        assert_eq!(g.codewords().len(), 4096);
        assert_eq!(GOLAY_RANK_BITS, lg_ceil(g.codewords().len() as u128));
        assert_eq!(GOLAY_MSG_BITS, GOLAY_RANK_BITS, "|Golay| = 2¹² : même largeur");
    }

    /// **T2's lethal test, written before reading any result**
    /// (pre-registration §4): an `e1v` narrower than `⌈log₂ |class|⌉` would be
    /// an impossible bijection — a counting bug, never a discovery.
    ///
    /// Checked on **both** sub-variants, against **both** class sources: the
    /// widths as `enumerate_classes` builds them, and the row the sweep
    /// actually reads through `FastDecoder`'s class order.
    #[test]
    fn e1v_is_never_narrower_than_the_class_rank() {
        let cs = enumerate_classes(13);
        for c in &cs.even {
            let w = widths_of_even(c);
            let floor = lg_ceil(u128::from(c.cardinality()));
            assert!(w.exact >= floor, "e1v-packé paire : {} < {floor}", w.exact);
            assert!(
                w.radix2 - HEADER_BITS >= floor,
                "e1v-séparé paire : {} < {floor}",
                w.radix2 - HEADER_BITS
            );
        }
        for c in &cs.odd {
            let w = widths_of_odd(c);
            let floor = lg_ceil(u128::from(c.cardinality()));
            assert!(w.exact >= floor, "e1v-packé impaire : {} < {floor}", w.exact);
            assert!(
                w.radix2 - HEADER_BITS >= floor,
                "e1v-séparé impaire : {} < {floor}",
                w.radix2 - HEADER_BITS
            );
        }

        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let (packed, split) = (variant("e1v-packé"), variant("e1v-séparé"));
        for ci in 0..fd.n_classes() {
            let (row, floor) = (t.row(ci), t.per_class[ci].exact);
            for (k, tag) in [(packed, "packé"), (split, "séparé")] {
                assert!(
                    row[k] >= floor + HEADER_BITS,
                    "classe {ci} : e1v-{tag} {} bits, en-tête compris, sous {floor} + {HEADER_BITS}",
                    row[k]
                );
            }
            // The packed variant is the floor itself, so any gap it shows is
            // the header and nothing else — the assertion above would still
            // pass if it silently grew, this one would not.
            assert_eq!(row[packed], floor + HEADER_BITS, "classe {ci}");
        }
        // The origin is not a class and is charged separately; it must still
        // be at least a header.
        assert!(t.origin[packed] >= HEADER_BITS && t.origin[split] >= HEADER_BITS);
    }

    /// **T2's verdict on the second sub-variant**: `e1v-séparé` is not a new
    /// width, it is [`Widths::radix2`] — one `⌈log₂⌉` per stage of
    /// `Indexer::encode`'s composition, which is what that field always
    /// computed. What T2 adds is a different *decode* on the same bits, so the
    /// difference lives entirely in the three declaration fields checked here.
    #[test]
    fn e1v_split_is_radix2_bit_for_bit() {
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let (split, r2) = (variant("e1v-séparé"), variant("radix2"));
        for ci in 0..fd.n_classes() {
            let row = t.row(ci);
            assert_eq!(row[split], row[r2], "classe {ci} : les deux largeurs ont divergé");
        }
        assert_eq!(t.origin[split], t.origin[r2], "bloc origine");
        // The interpretation, which is the whole of the difference.
        assert!(!VARIANTS[r2].shift_only, "radix2 déclaré shift-only");
        assert!(VARIANTS[split].shift_only, "e1v-séparé n'est plus shift-only");
        assert_eq!(VARIANTS[r2].depth, None, "le rang de multiensemble est sériel");
        assert_eq!(
            VARIANTS[split].depth,
            Some(96),
            "la marche binomiale est à 48-96 pas, et la profondeur doit le DIRE"
        );
        assert!(
            VARIANTS[split].depth.unwrap() > 24,
            "une profondeur ≤ 24 déguiserait la réouverture de la clause du spec"
        );
    }

    /// **T3's lethal test** (same rule as T2's): a `golay_signs` narrower than
    /// `⌈log₂ |class|⌉` would be an impossible bijection.
    ///
    /// The last block is the one that gives the test teeth: it re-prices the
    /// odd coset **without** the 12-bit Golay message and requires that at
    /// least one class then fall under its own rank. Without it, the variant
    /// could drop the codeword entirely and this test would still be green on
    /// the classes whose arrangements happen to be small — which is exactly
    /// how `perslot` came to charge zero there.
    #[test]
    fn golay_signs_matches_the_class_cardinality() {
        let cs = enumerate_classes(13);
        for c in &cs.even {
            let w = widths_of_even(c);
            assert_eq!(w.sign_field, even_sign_bits(c), "champ de signes pair");
            assert!(
                w.golay_signs - HEADER_BITS >= w.exact,
                "classe paire : golay_signs {} sous le rang {}",
                w.golay_signs - HEADER_BITS,
                w.exact
            );
        }
        for c in &cs.odd {
            let w = widths_of_odd(c);
            assert_eq!(w.sign_field, GOLAY_MSG_BITS, "champ de signes impair");
            assert!(
                w.golay_signs - HEADER_BITS >= w.exact,
                "classe impaire : golay_signs {} sous le rang {}",
                w.golay_signs - HEADER_BITS,
                w.exact
            );
        }
        // Through FastDecoder's order, i.e. the row the sweep reads.
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let k = variant("golay_signs");
        for ci in 0..fd.n_classes() {
            assert!(
                t.row(ci)[k] - HEADER_BITS >= t.per_class[ci].exact,
                "classe {ci} : golay_signs sous son rang"
            );
        }
        // Lethality: the 12-bit message is load-bearing, not padding.
        let broken = cs
            .odd
            .iter()
            .filter(|c| {
                let w = widths_of_odd(c);
                w.golay_signs - HEADER_BITS - GOLAY_MSG_BITS < w.exact
            })
            .count();
        assert!(
            broken > 0,
            "retirer le message Golay ne casse aucune classe : le test ne prouve rien"
        );
    }

    /// The odd coset's signs are forced **given the codeword**, and the
    /// `perslot` level planes do not carry it: that column is 12 bits below
    /// anything invertible, on all 157 odd classes.
    ///
    /// ⚠️ Only **10** of them are visible to a width bound — on the other 147,
    /// `24·⌈log₂L⌉` keeps the row above `⌈log₂|class|⌉` while still not being
    /// an invertible encoding. Both counts are pinned, because the gap between
    /// them *is* the limit of what a bit-counter can prove and the reason the
    /// lethal test elsewhere needs a mutation check rather than a bound.
    ///
    /// Frozen as published (module header, slack 2) and pinned here with its
    /// direction. `golay_signs` is the sound counterpart, and this test is
    /// what says so in machine-checkable terms rather than in prose.
    #[test]
    fn golay_signs_is_the_first_perslot_variant_decodable_on_the_odd_coset() {
        let cs = enumerate_classes(13);
        let mut short = 0;
        for c in &cs.odd {
            let w = widths_of_odd(c);
            assert_eq!(
                w.golay_signs,
                w.perslot + GOLAY_MSG_BITS,
                "golay_signs impair n'est plus perslot + le message"
            );
            if w.perslot - HEADER_BITS < w.exact {
                short += 1;
            }
        }
        assert_eq!(cs.odd.len(), 157, "l'énumération des classes impaires a bougé");
        assert_eq!(
            short, 10,
            "10 des 157 classes impaires tombent sous leur rang en perslot, pas {short}"
        );
        // The even coset is sound in `perslot` — `nonzero ≥ (w−1) + free_n` —
        // so the defect is odd-only, and naming it that way is part of the
        // finding.
        for c in &cs.even {
            let w = widths_of_even(c);
            assert!(w.perslot - HEADER_BITS >= w.exact, "classe paire perslot sous son rang");
            // …and there `golay_signs` is at most one bit cheaper, because
            // `perslot` already charges per nonzero slot rather than a flat 24.
            assert!(w.perslot >= w.golay_signs && w.perslot - w.golay_signs <= 1);
        }
    }

    /// **Slack 1 of the module header, pinned rather than fixed.** The frozen
    /// [`even_planes`] charges `nonzero − 1` sign bits; the parity constraint
    /// frees a sign on the codeword **support**, so that is right whenever
    /// `w > 0` and one bit short when `w = 0`.
    ///
    /// The three things this has to establish, because the number is published
    /// and not being corrected: the defect exists, it is bounded to one bit on
    /// exactly the `w = 0` classes, and it cannot make `golay_tight` claim an
    /// impossible bijection (its flat 12-bit codeword field has the slack to
    /// absorb it).
    #[test]
    fn the_frozen_even_sign_plane_undercharges_w0_classes_by_one_bit() {
        let cs = enumerate_classes(13);
        let mut w0 = 0;
        for c in &cs.even {
            let (_, frozen) = even_planes(c);
            let truth = even_sign_bits(c);
            let free_n: u64 = c.free_vals.iter().map(|&(_, n)| u64::from(n)).sum();
            if c.w == 0 {
                // A `w = 0` class is made of free values only, and cannot be
                // empty (the origin is not a class), so `nonzero = free_n ≥ 1`
                // and `nonzero − 1` loses exactly the bit nothing fixed.
                assert!(free_n > 0, "classe paire vide");
                assert_eq!(truth, free_n);
                assert_eq!(truth, frozen + 1, "w = 0 : le manque n'est pas d'un bit");
                w0 += 1;
            } else {
                assert_eq!(frozen, truth, "w > 0 : nonzero−1 EST (w−1) + free_n");
            }
            // Optimistic, bounded, and harmless to the bijection.
            assert!(truth >= frozen && truth - frozen <= 1);
            let w = widths_of_even(c);
            assert!(
                w.golay_tight - HEADER_BITS >= w.exact,
                "golay_tight sous son rang : le serrage a cessé d'être valide"
            );
        }
        assert_eq!(cs.even.len(), 226, "l'énumération des classes paires a bougé");
        assert_eq!(w0, 17, "17 des 226 classes paires sont à w = 0, pas {w0}");
    }

    /// The two class sources must cover each other exactly, or the sweep
    /// would price blocks against another class's radices.
    #[test]
    fn every_fastdecoder_class_has_its_radices() {
        let fd = FastDecoder::new();
        let cs = enumerate_classes(13);
        let n = cs.even.len() + cs.odd.len();
        assert_eq!(n, fd.n_classes(), "{n} classes énumérées, {} au décodeur", fd.n_classes());
        let mut keys: Vec<Key> = Vec::new();
        for c in &cs.even {
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
            let n0 = 24 - c.w - free_n;
            let mut kinds: Vec<(i32, u8)> = c
                .word_vals
                .iter()
                .chain(&c.free_vals)
                .map(|&(v, n)| (i32::from(v), n))
                .collect();
            if n0 > 0 {
                kinds.push((0, n0 as u8));
            }
            keys.push(canonical(kinds, false));
        }
        for c in &cs.odd {
            keys.push(canonical(
                c.vals.iter().map(|&(v, n)| (i32::from(v), n)).collect(),
                true,
            ));
        }
        let set: HashMap<Key, ()> = keys.iter().map(|&k| (k, ())).collect();
        assert_eq!(set.len(), keys.len(), "deux classes partagent une clé canonique");
        for ci in 0..fd.n_classes() {
            let k = key_of_levels(fd.levels(ci));
            assert!(set.contains_key(&k), "classe {ci} sans radices : {:?}", fd.levels(ci));
        }
    }

    /// `multiset` must agree with the cardinalities `classes.rs` computes, or
    /// every radix below it is wrong.
    #[test]
    fn multiset_matches_the_class_cardinalities() {
        let cs = enumerate_classes(13);
        for c in &cs.odd {
            let arr = multiset(24, c.vals.iter().map(|&(_, n)| n));
            assert_eq!(
                u64::try_from(4096 * arr).unwrap(),
                c.cardinality(),
                "cardinalité impaire"
            );
        }
    }

    /// The criterion is stated in the kernel accounting, so the conversion
    /// must be the same one `rtbits` and the CUDA bench use: `Planes14`'s 112
    /// bits per block are 4.804 b/weight on the 4B.
    #[test]
    fn kernel_conversion_reproduces_planes14() {
        let p14 = kernel_bpw(112.0);
        assert!((p14 - 4.804).abs() < 5e-4, "Planes14 : {p14:.4}");
        // And the archive's own 48 bits, the floor E3 is chasing.
        let arch = kernel_bpw(ARCHIVE_BITS as f64);
        assert!(arch < E3_CRITERION, "le fichier lui-même vaut {arch:.4}");
    }

    /// The pre-registration states each threshold twice — in b/weight and in
    /// bits per block — and this bench decides on the first. The two must be
    /// the same statement, so the conversion is checked in both directions and
    /// the residual rounding is **named** rather than left to be discovered in
    /// a rapport.
    #[test]
    fn the_preregistered_thresholds_are_one_statement_in_two_units() {
        let s = Shapes::qwen3_4b();
        // S_spec: 2.60 ⇔ 58.86 to a hundredth of a bit, both ways.
        assert!((s.bits_for(E3_CRITERION) - PREREG_BITS.0).abs() < 5e-3);
        assert!((s.kernel_bpw(PREREG_BITS.0) - E3_CRITERION).abs() < 5e-4);
        // S_alt: the pre-registration's 70.66 is the ⇔ of 3.0895, not of
        // 3.0900 — a 0.012-bit rounding, inert at this distance from every
        // prediction, and pinned here so nobody re-derives it as a
        // disagreement.
        assert!((s.bits_for(S_ALT) - 70.6716).abs() < 1e-3, "{}", s.bits_for(S_ALT));
        assert!((s.kernel_bpw(PREREG_BITS.1) - 3.0895).abs() < 1e-3);
        assert!(s.bits_for(S_ALT) - PREREG_BITS.1 < 0.02);
    }

    /// The shapes read from a file and the pinned 4B constants must be the
    /// same object, or `kernel_bpw` would silently price the 8B on the 4B.
    /// Checked here on the shapes alone — the file test below checks that the
    /// file really produces them.
    #[test]
    fn the_pinned_shapes_are_the_published_4b() {
        let s = Shapes::qwen3_4b();
        assert_eq!(s.blocks * DIM as u64 + s.tail_weights, s.weights);
        assert!((s.kernel_bpw(112.0) - 4.804).abs() < 5e-4);
    }

    // ---- addressing, on synthetic widths ----

    fn group_of(ws: &[u64]) -> Group {
        let mut g = Group::default();
        for &w in ws {
            g.push(w);
        }
        g.close();
        g
    }

    /// `Grp32Max` charges every lane the group's widest block, rounded up to a
    /// byte, plus one base word — and a single wide block in the group is
    /// enough to make all 32 lanes pay for it. That asymmetry is the whole
    /// reason the file order matters.
    #[test]
    fn grp32max_charges_the_group_maximum_to_every_lane() {
        let flat = group_of(&[47u64; 32]);
        assert_eq!(flat.max_bits, 32 * 48 + 32, "47 arrondi à 48 par lane");
        let mut ws = [47u64; 32];
        ws[13] = 94;
        let spiked = group_of(&ws);
        assert_eq!(spiked.max_bits, 32 * 96 + 32, "un bloc à 94 fait payer 96 aux 32");
        // 47 → 94 on ONE lane costs 32·(96−48) = 1536 bits of stride.
        assert_eq!(spiked.max_bits - flat.max_bits, 32 * (96 - 48));
        // The scan mode pays that one block and the word it rounds into:
        // Σ goes 32·47 = 1504 (already a whole word) to 31·47 + 94 = 1551,
        // which rounds to 1568 — 64 bits, not the 1536 of the stride.
        assert_eq!(flat.scan_bits, 1504 + 32);
        assert_eq!(spiked.scan_bits, 1568 + 32);
        assert_eq!(spiked.scan_bits - flat.scan_bits, 64);
    }

    /// `WarpScan` is `⌈Σ/32⌉·32 + 32`, and the pre-registration's phrasing
    /// ("Σ + 32 d'offset de base, arrondi au mot de 32 bits") has only one
    /// reading because the base offset is itself a whole word: rounding before
    /// or after adding it gives the same number.
    #[test]
    fn warp_scan_is_the_word_rounded_sum_plus_a_base_word() {
        for &w in &[10u64, 47, 48, 63, 70, 94, 101] {
            let g = group_of(&[w; 32]);
            let sum = 32 * w;
            assert_eq!(g.scan_bits, sum.div_ceil(WORD_BITS) * WORD_BITS + BASE_OFFSET_BITS);
            assert_eq!(g.scan_bits, (sum + BASE_OFFSET_BITS).div_ceil(WORD_BITS) * WORD_BITS);
            // And it can never exceed the uniform stride, which pays the same
            // max to every lane after rounding it up to a byte.
            assert!(g.scan_bits <= g.max_bits, "scan {} > max {}", g.scan_bits, g.max_bits);
        }
    }

    /// A partial group is where the two modes genuinely disagree, and where an
    /// off-by-one would live: `Grp32Max` still uploads 32 lanes, `WarpScan`
    /// uploads what it holds. Both are counted as one group, and as partial.
    #[test]
    fn a_partial_group_pays_full_lanes_under_max_and_only_its_blocks_under_scan() {
        let g = group_of(&[48u64; 5]);
        assert_eq!((g.groups, g.partial), (1, 1));
        assert_eq!(g.max_bits, 32 * 48 + 32, "les 32 voies sont adressables");
        // 5·48 = 240 bits, which is 7.5 words: the group rounds to 8 (256) and
        // then pays its base word. Charging 240 + 32 would be the off-by-one
        // this assertion exists to catch.
        assert_eq!(g.scan_bits, 256 + 32, "5 blocs arrondis au mot, pas 32 voies");
        assert!(g.scan_bits < g.max_bits);
        // A full group is not partial, and closing an empty accumulator is a
        // no-op — which is what makes the matrix-boundary close free when the
        // boundary falls on a group boundary.
        let full = group_of(&[48u64; 32]);
        assert_eq!((full.groups, full.partial), (1, 0));
        let mut e = Group::default();
        e.close();
        e.close();
        assert_eq!(e, Group::default());
    }

    /// Groups must close at a matrix boundary, and this has to be proved
    /// **through [`Sweep::push_matrix`]** — the only code path that does it.
    ///
    /// 🕳️ The version of this test before 2026-08-13 called [`Group::close`] by
    /// hand and never touched `push_matrix`, so deleting the closing loop at
    /// the end of `push_matrix` left the whole suite green. It could not have
    /// been caught by the file tests either: on both sealed artifacts every
    /// `d_out` is a multiple of 32, so that loop is a **no-op** there and the
    /// two sweeps agree whether or not it exists. That is exactly the pattern
    /// CLAUDE.md §5 names — an assertion that does not exercise the parameter
    /// it is supposed to cover tests nothing.
    ///
    /// The fix is a matrix whose block count is **not** a multiple of 32:
    /// 3 rows × 17 blocks = 51 = one full group plus a partial of 19. Two such
    /// matrices must make **4** groups, **2** of them partial. Without the
    /// close, the two streams merge into 102 consecutive blocks — 3 full
    /// groups, 0 partial, 6 blocks never even accounted — and every assertion
    /// below fails.
    #[test]
    fn a_matrix_boundary_closes_the_group() {
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        // 17 blocks per row, 3 rows: 51 blocks, deliberately not 32 | 51.
        let (rows, per_row) = (3usize, 17usize);
        let blocks = rows * per_row;
        assert!(!(blocks as u64).is_multiple_of(GROUP), "51 doit couper un groupe");
        let matrix = |name: &str| llvq_artifact::RawMatrix {
            name: name.to_string(),
            d_out: rows,
            d_in: per_row * DIM,
            // Real in-ball indices, so no block takes the origin tariff.
            indices: (0..blocks).map(|i| fd.class_range(i % fd.n_classes()).0).collect(),
            gains: vec![0; blocks],
            row_scales: Vec::new(),
            centroids: Vec::new(),
            rotation_seed: None,
            shell_cap: 12,
            tail: Vec::new(),
        };

        let mut s = Sweep::empty(fd.n_classes());
        s.push_matrix(&fd, &t, &matrix("A"));
        // One matrix alone already shows it: 51 blocks are 1 full group and a
        // partial of 19, and the partial must be **closed** by the boundary.
        assert_eq!(s.origins, 0, "les indices de test doivent tous être dans la boule");
        let single = s.file.clone();
        for (k, v) in VARIANTS.iter().enumerate() {
            assert_eq!(
                (single[k].groups, single[k].partial),
                (2, 1),
                "{} : la fin de matrice n'a pas fermé le groupe partiel",
                v.name
            );
            assert_eq!(single[k].filled, 0, "{} : 19 voies restées ouvertes", v.name);
        }

        s.push_matrix(&fd, &t, &matrix("B"));
        for (k, v) in VARIANTS.iter().enumerate() {
            let g = s.file[k];
            assert_eq!(
                (g.groups, g.partial),
                (4, 2),
                "{} : les deux matrices se sont fondues en un seul flux",
                v.name
            );
            // The two matrices are identical, so a boundary that really closes
            // makes the second cost exactly what the first did — in both modes.
            assert_eq!(g.max_bits, 2 * single[k].max_bits, "{}", v.name);
            assert_eq!(g.scan_bits, 2 * single[k].scan_bits, "{}", v.name);

            // And the mutation this test exists for: the same 102 blocks read
            // as ONE continuous stream make 3 full groups and leave 6 blocks
            // dangling in an accumulator nobody closes. Different count,
            // different bits — so the closing loop is load-bearing, not decor.
            let ws: Vec<u64> = (0..blocks).map(|i| t.row(i % fd.n_classes())[k]).collect();
            let mut joined = Group::default();
            for w in ws.iter().chain(ws.iter()) {
                joined.push(*w);
            }
            assert_eq!(joined.groups, 3, "le flux continu ne ferait que 3 groupes pleins");
            assert_eq!(joined.filled, 102 - 3 * GROUP, "6 blocs restent en suspens");
            assert_ne!(
                (g.groups, g.partial, g.max_bits, g.scan_bits),
                (joined.groups, joined.partial, joined.max_bits, joined.scan_bits),
                "{} : fermer à la frontière ne change rien, donc rien ne prouve qu'on ferme",
                v.name
            );
        }
    }

    /// The `Group` accumulator's own boundary arithmetic, kept next to the
    /// sweep-level test above: two closed halves cost two full strides where a
    /// single continuous group costs one.
    #[test]
    fn a_closed_group_is_paid_whole_in_max_and_by_the_word_in_scan() {
        let mut split = Group::default();
        for _ in 0..2 {
            for _ in 0..16 {
                split.push(48);
            }
            split.close();
        }
        assert_eq!((split.groups, split.partial), (2, 2));
        let joined = group_of(&[48u64; 32]);
        assert_eq!((joined.groups, joined.partial), (1, 0));
        assert_eq!(split.max_bits, 2 * joined.max_bits, "deux groupes payés plein");
        assert_eq!(split.scan_bits, joined.scan_bits + BASE_OFFSET_BITS);
    }

    // ---- the verdict's own clauses ----

    /// **D1, pinned.** The spec's criterion is a **conjunction** — ≤ 2.6
    /// b/weight *and* a fixed decode depth ≤ 24 steps
    /// (`spec-memoire-extreme-2026-08-12.md:186-187`, module header). Until
    /// 2026-08-13 the verdict branched on [`Variant::shift_only`] alone and
    /// [`Variant::depth`] entered no decision at all, so `e1v-séparé` — a
    /// declared `depth: Some(96)` — was published as an admission.
    ///
    /// This test is the mutation: restore the one-clause filter and it fails,
    /// because `e1v-séparé` would then classify as [`Admissibility::Admissible`].
    #[test]
    fn the_verdict_needs_both_clauses_not_shift_only_alone() {
        // The clause exists in the code as a number, not only in the prose.
        assert_eq!(MAX_DEPTH, 24, "la clause du spec est ≤ 24 étapes");

        let mut deep = 0;
        for v in VARIANTS {
            let cls = v.admissibility();
            match (v.shift_only, v.depth) {
                (false, _) => assert_eq!(cls, Admissibility::Serial, "{}", v.name),
                (true, Some(d)) if d <= MAX_DEPTH => {
                    assert_eq!(cls, Admissibility::Admissible, "{}", v.name);
                }
                (true, _) => {
                    deep += 1;
                    assert_eq!(
                        cls,
                        Admissibility::DepthReopening,
                        "{} est shift-only à profondeur {:?} : la CONJONCTION du spec exige \
                         qu'il soit une RÉOUVERTURE, jamais une admission",
                        v.name,
                        v.depth
                    );
                    assert_ne!(cls, Admissibility::Admissible, "{}", v.name);
                }
            }
        }
        // Without such a variant the test proves nothing — it would be green on
        // a menu where the two clauses happen to coincide.
        assert!(
            deep > 0,
            "aucune variante shift-only de profondeur > {MAX_DEPTH} au menu : ce test ne \
             distingue plus la conjonction du seul shift_only"
        );
        // And the one that made the defect matter, by name.
        let split = &VARIANTS[variant("e1v-séparé")];
        assert!(split.shift_only, "e1v-séparé n'est plus shift-only");
        assert_eq!(split.depth, Some(96));
        assert_eq!(split.admissibility(), Admissibility::DepthReopening);
    }

    /// **D1, the printed half.** The sentence that claims the spec's
    /// conjunction is met must be unreachable for a point that only cleared the
    /// bit criterion — the exact over-claim the 2026-08-13 journals carried.
    #[test]
    fn a_depth_reopening_never_prints_the_spec_admission_sentence() {
        let deep = Point {
            name: "e1v-séparé",
            mode: Mode::WarpScan,
            bpw: 2.3709,
            depth: Some(96),
        };
        let shallow = Point {
            name: "hypothétique",
            mode: Mode::WarpScan,
            bpw: 2.3709,
            depth: Some(24),
        };

        // A reopening alone: no admission sentence, and the reopening named.
        let t = verdict_text(&[], std::slice::from_ref(&deep), f64::INFINITY);
        assert!(!t.contains(OPEN_AT_SPEC), "réouverture publiée comme admission :\n{t}");
        assert!(t.contains("AUCUNE décomposition ADMISSIBLE"), "{t}");
        assert!(t.contains("HORS DE LA CLAUSE DE PROFONDEUR"), "{t}");
        assert!(t.contains("RÉOUVERTURE"), "{t}");
        assert!(t.contains("2.3709"), "le point doit rester chiffré :\n{t}");

        // The same point, plus a genuinely admissible one: the sentence comes
        // back, and the reopening is still reported separately rather than
        // folded into the admitted list.
        let t2 = verdict_text(std::slice::from_ref(&shallow), std::slice::from_ref(&deep), 2.3709);
        assert!(t2.contains(OPEN_AT_SPEC), "{t2}");
        assert!(t2.contains("HORS DE LA CLAUSE DE PROFONDEUR"), "{t2}");
        assert!(t2.contains("hypothétique"), "{t2}");

        // And nothing at all under the criterion: the plain red verdict, with
        // the best ADMISSIBLE point — never the best shift-only one.
        let t3 = verdict_text(&[], &[], 2.9650);
        assert!(!t3.contains(OPEN_AT_SPEC), "{t3}");
        assert!(t3.contains("Le meilleur point ADMISSIBLE vaut 2.9650"), "{t3}");
        assert!(!t3.contains("RÉOUVERTURE"), "{t3}");
    }

    /// **D5a, pinned rather than fixed.** The origin tariff is
    /// `Widths::default()` capped at a header, which is right for a
    /// variable-width variant and **wrong** for a fixed-width one: `golay70` is
    /// a flat 70-bit record and is charged 10.
    ///
    /// Not repaired, because no sealed artifact contains an origin block, so
    /// the repair would be unverifiable and the published columns would not
    /// move. [`main`] asserts instead — this pins the defect, its direction and
    /// its size, so the assertion's message stays true.
    #[test]
    fn the_origin_tariff_is_wrong_for_fixed_width_variants() {
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let g70 = variant("golay70 (mesuré, écarté)");
        // The flat width every real `golay70` record has, class or not.
        let flat = HEADER_BITS + GOLAY_RANK_BITS + 24 + 24;
        assert_eq!(flat, 70);
        for ci in 0..fd.n_classes() {
            assert_eq!(t.row(ci)[g70], flat, "golay70 n'est plus à largeur fixe");
        }
        assert_eq!(
            t.origin[g70], HEADER_BITS,
            "le tarif origine de golay70 a changé : le message d'assertion de main() ment"
        );
        assert!(t.origin[g70] < flat, "le défaut est une SOUS-facturation, et il faut le dire");
        // The variable-width variants are the ones the tariff is right for.
        for name in ["perslot (PAS Planes14)", "golay_tight", "golay_signs"] {
            assert_eq!(t.origin[variant(name)], HEADER_BITS, "{name}");
        }
    }

    /// **D5b.** `perslot` is the served layouts' per-slot **geometry** priced
    /// at each class's own field widths — it is not `Planes14`, which is a flat
    /// **112** bits per block, i.e. 4.804 b/weight. The old label
    /// `perslot (= Planes)` invited reading this row's 4.15 as the served
    /// layout's price; both halves of why that is false are pinned here.
    #[test]
    fn perslot_is_never_the_served_planes14_price() {
        const PLANES14_BITS: u64 = 112;
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let k = variant("perslot (PAS Planes14)");
        assert!(
            !VARIANTS[k].name.contains("= Planes"),
            "l'étiquette réintroduit l'égalité fausse avec le layout servi"
        );
        for ci in 0..fd.n_classes() {
            assert!(
                t.row(ci)[k] < PLANES14_BITS,
                "classe {ci} : perslot {} atteint les {PLANES14_BITS} b/bloc plats de Planes14",
                t.row(ci)[k]
            );
        }
        // The served layout's own price, which this column is never allowed to
        // be read as.
        assert!((kernel_bpw(PLANES14_BITS as f64) - 4.804).abs() < 5e-4);
        // And the second reason the two are not comparable at all: on the odd
        // coset this row is 12 bits below anything invertible.
        for c in &enumerate_classes(13).odd {
            assert_eq!(
                widths_of_odd(c).perslot + GOLAY_MSG_BITS,
                widths_of_odd(c).golay_signs,
                "perslot impair n'est plus 12 bits sous le décodable"
            );
        }
    }

    /// **Mission C.** Every variant of the menu — the six frozen ones and the
    /// three added for T2/T3 — must be priced in all four `order × addressing`
    /// combinations, not just the ones the printer happens to reach.
    ///
    /// Structural rather than eyeballed: the accumulators are `NVAR` wide by
    /// construction, so what has to be checked is that a real sweep fills
    /// every slot of both orders and that both modes come out of it. The
    /// synthetic matrix carries **one block of every class plus one origin**,
    /// so no class is silently unpriced either.
    #[test]
    fn every_variant_is_priced_in_all_four_combinations() {
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        // 383 classes + 1 origin = 384 blocks = 8 rows of 48, and 384 is a
        // whole number of groups, so neither order pays for padding.
        let n = fd.n_classes();
        assert_eq!(n, 383, "le menu synthétique suppose les 383 classes");
        let mut indices: Vec<u64> = (0..n).map(|ci| fd.class_range(ci).0).collect();
        indices.push(0); // the origin
        let blocks = indices.len();
        let m = llvq_artifact::RawMatrix {
            name: "synthétique".to_string(),
            d_out: 8,
            d_in: 48 * DIM,
            indices,
            gains: vec![0; blocks],
            row_scales: Vec::new(),
            centroids: Vec::new(),
            rotation_seed: None,
            shell_cap: 12,
            tail: Vec::new(),
        };
        let mut s = Sweep::empty(n);
        s.push_matrix(&fd, &t, &m);
        assert_eq!(s.total(), blocks as u64);
        assert_eq!(s.origins, 1, "l'origine doit passer par le tarif origine");
        assert!(s.count.iter().all(|&c| c == 1), "une classe non vue par le balayage");

        let cm = sweep_class_major(&t, &s.count, s.origins);
        assert_eq!(cm.len(), NVAR, "ordre classe-majeur : {} accumulateurs", cm.len());
        assert_eq!(s.file.len(), NVAR, "ordre-fichier : {} accumulateurs", s.file.len());
        assert_eq!(NVAR, VARIANTS.len());
        for (k, v) in VARIANTS.iter().enumerate() {
            for (g, tag) in [(cm[k], "CM"), (s.file[k], "FO")] {
                for mode in Mode::ALL {
                    assert!(
                        g.bits(mode) > 0,
                        "{} non prixé en {tag}/{}",
                        v.name,
                        mode.tag()
                    );
                }
                assert!(
                    g.bits(Mode::WarpScan) <= g.bits(Mode::Grp32Max),
                    "{} : {tag}/scan au-dessus de {tag}/max",
                    v.name
                );
                assert_eq!(g.groups, blocks as u64 / GROUP, "{} en {tag}", v.name);
                assert_eq!(g.partial, 0, "{} en {tag}", v.name);
            }
        }
    }

    /// **T1's non-regression control.** In `(ClassMajor, Grp32Max)` this
    /// binary must still say exactly what the 2026-08-12 X4 log says, or no
    /// column of the lot is readable (pre-registration §3, §4).
    ///
    /// This half needs no artifact: it is the published arm 1, the 383 classes
    /// at equal weight, reproduced to the last printed digit. The arm that
    /// carries the 3.0444 needs the sealed file and lives in the `#[ignore]`d
    /// test below — an archive sweep costs minutes even in release, which is
    /// the repo's stated reason for the second `ignore` tier.
    #[test]
    fn class_major_grp32max_reproduces_the_published_verdict() {
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let count = vec![1u64; fd.n_classes()];
        let cm = sweep_class_major(&t, &count, 0);
        let total = count.iter().sum::<u64>();
        let shapes = Shapes::qwen3_4b();

        // radixstudy-x4-2026-08-12.txt, bras 1 : moy · pire · grp32 · b/p moy · b/p GRP
        let want: [(&str, f64, u64, f64, f64, f64); 6] = [
            ("archive (le fichier)", 48.00, 48, 49.13, 2.1498, 2.1965),
            ("radix2", 43.68, 56, 55.81, 1.9708, 2.4737),
            ("radix2 + golay 12 b", 45.08, 56, 56.48, 2.0288, 2.5015),
            ("golay_tight", 62.03, 94, 81.21, 2.7317, 3.5271),
            ("golay70 (mesuré, écarté)", 70.00, 70, 73.19, 3.0621, 3.1945),
            // Renamed 2026-08-13 (`perslot (= Planes)` → `perslot (PAS
            // Planes14)`): a label, never a number. The five figures on this
            // row are the published ones, unchanged.
            ("perslot (PAS Planes14)", 65.80, 102, 95.25, 2.8878, 4.1092),
        ];
        // The published six must stay a **prefix** of the menu: variants added
        // later (T2, T3) are appended, never inserted, or this control would
        // be comparing a row against another row's number.
        assert!(
            VARIANTS.len() >= want.len(),
            "le menu a rétréci : le contrôle publié n'est plus reproductible"
        );
        for (k, (name, moy, pire, grp, bpw_moy, bpw_grp)) in want.iter().enumerate() {
            assert_eq!(VARIANTS[k].name, *name, "l'ordre du menu a changé");
            let mean = mean_bits(&t, &count, 0, k);
            let grouped = cm[k].bits(Mode::Grp32Max) as f64 / total as f64;
            assert!((mean - moy).abs() < 5e-3, "{name} moy : {mean:.2} contre {moy}");
            assert_eq!(worst_bits(&t, &count, 0, k), *pire, "{name} pire");
            assert!((grouped - grp).abs() < 5e-3, "{name} grp32 : {grouped:.2} contre {grp}");
            let (a, b) = (shapes.kernel_bpw(mean), shapes.kernel_bpw(grouped));
            assert!((a - bpw_moy).abs() < 5e-5, "{name} b/p moy : {a:.4}");
            assert!((b - bpw_grp).abs() < 5e-5, "{name} b/p GRP : {b:.4}");
        }
        // The `exact` column of the same arm, and the Planes14 control that
        // ties this table to the 4.804 of the production layout.
        let exact: f64 = (0..fd.n_classes())
            .map(|ci| t.per_class[ci].exact as f64)
            .sum::<f64>()
            / total as f64;
        assert!((exact - 33.40).abs() < 5e-3, "⌈log₂|classe|⌉ : {exact:.2}");
        assert!((kernel_bpw(112.0) - 4.804).abs() < 5e-4);
    }

    /// The streaming glue, on **real** blocks: header, one matrix, the
    /// per-block class lookup, the group boundary and the matrix boundary.
    ///
    /// One matrix of the sealed 4B is 108 544 blocks and a couple of megabytes
    /// — a second, against the minutes of a full sweep — so this runs where the
    /// full control cannot. It asserts geometry and **direction**, never a
    /// rate: no number printed by this test is a result about a layout.
    ///
    /// `#[ignore]` on the same rule as the sweep below: it needs the archive,
    /// and a missing archive is a failure, not a skip.
    #[test]
    #[ignore = "reads the first matrix of the sealed 4B: needs ~/llvq-q4b.llvq"]
    fn the_file_order_glue_runs_on_a_real_matrix() {
        let path = sealed_4b();
        assert!(
            std::path::Path::new(&path).exists(),
            "archive scellée absente : {path} (LLVQ_Q4B déplace la recherche, ne la satisfait pas)"
        );
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let f = File::open(&path).expect("open");
        let mut r = BufReader::with_capacity(1 << 20, f);
        let h = llvq_artifact::read_header(&mut r).expect("valid header");
        assert_eq!(h.version, 1, "le 4B publié est un LVQ1 « projections seules »");
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(m.name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!((m.d_out, m.d_in), (4096, 2560));

        let mut s = Sweep::empty(fd.n_classes());
        s.push_matrix(&fd, &t, &m);
        let blocks = 4096 * (2560 / DIM) as u64;
        assert_eq!(s.total(), blocks, "434 176 blocs : 4096 lignes de 106");
        // 106 blocks per row is not a multiple of 32: groups straddle rows.
        assert_eq!(s.row_straddling, 1);
        // 434 176 blocks is 13 568 whole groups — nothing partial, so closing
        // at the matrix boundary costs this matrix nothing.
        assert!(blocks.is_multiple_of(GROUP));
        assert_eq!((s.file[0].groups, s.file[0].partial), (blocks / GROUP, 0));

        // Direction, which is the whole point of re-ordering: on the same
        // blocks, the real order pays a wider stride than the class-major
        // control, and warp-scan pays less than the stride it replaces.
        let k = VARIANTS.iter().position(|v| v.name == "golay_tight").expect("golay_tight");
        let cm = sweep_class_major(&t, &s.count, s.origins);
        assert!(
            s.file[k].max_bits > cm[k].max_bits,
            "ordre-fichier {} ≤ contrôle classe-majeur {} : le contrôle n'est plus optimiste",
            s.file[k].max_bits,
            cm[k].max_bits
        );
        assert!(s.file[k].scan_bits < s.file[k].max_bits);
        // Both orders see exactly the same blocks, so their means coincide —
        // only the grouping differs. A divergence here would mean the sweep
        // dropped or double-counted a block.
        assert_eq!(cm[k].groups * GROUP, s.file[k].groups * GROUP);
    }

    /// The same control on the object that decides: the sealed 4B, whose
    /// class-major grp32-max column is the published **3.0444** on
    /// `golay_tight` and whose `exact` mean is the **41.50** the whole E3
    /// verdict is argued from.
    ///
    /// `#[ignore]` because it reads a 981 MB archive — the repo's second
    /// `ignore` tier (CLAUDE.md §7), not a skip: invoked explicitly with the
    /// file absent, it **fails**, and says which file.
    #[test]
    #[ignore = "walks the sealed 4B artifact: minutes, and needs ~/llvq-q4b.llvq"]
    fn the_sealed_4b_reproduces_the_published_class_major_verdict() {
        let path = sealed_4b();
        assert!(
            std::path::Path::new(&path).exists(),
            "archive scellée absente : {path} (LLVQ_Q4B déplace la recherche, ne la satisfait pas)"
        );
        let fd = FastDecoder::new();
        let t = WidthTable::build(&fd);
        let s = sweep_file(&fd, &t, &path);
        let total = s.total();
        assert_eq!(total, 150_681_600, "ce n'est pas le 4B publié");
        assert_eq!(s.matrices, 252);
        assert_eq!(s.origins, 0, "le 4B publié n'a aucun bloc origine");
        // The shapes read from the file ARE the pinned constants — the whole
        // licence for reading them from the file instead of remembering them.
        assert_eq!(s.shapes, Shapes::qwen3_4b(), "formes lues ≠ formes épinglées");

        let cm = sweep_class_major(&t, &s.count, s.origins);
        // radixstudy-x4-2026-08-12.txt, bras 2 : moy · pire · grp32 · b/p GRP
        let want: [(&str, f64, u64, f64, f64); 6] = [
            ("archive (le fichier)", 48.00, 48, 49.00, 2.1912),
            ("radix2", 51.87, 55, 56.61, 2.5066),
            ("radix2 + golay 12 b", 52.24, 56, 56.69, 2.5101),
            ("golay_tight", 66.17, 94, 69.57, 3.0444),
            ("golay70 (mesuré, écarté)", 70.00, 70, 73.00, 3.1866),
            ("perslot (PAS Planes14)", 67.43, 101, 73.47, 3.2060),
        ];
        for (k, (name, moy, pire, grp, bpw_grp)) in want.iter().enumerate() {
            assert_eq!(VARIANTS[k].name, *name);
            let mean = mean_bits(&t, &s.count, s.origins, k);
            let grouped = cm[k].bits(Mode::Grp32Max) as f64 / total as f64;
            assert!((mean - moy).abs() < 5e-3, "{name} moy : {mean:.2} contre {moy}");
            assert_eq!(worst_bits(&t, &s.count, s.origins, k), *pire, "{name} pire");
            assert!((grouped - grp).abs() < 5e-3, "{name} grp32 : {grouped:.2} contre {grp}");
            let b = s.shapes.kernel_bpw(grouped);
            assert!((b - bpw_grp).abs() < 5e-5, "{name} b/p GRP : {b:.4} contre {bpw_grp}");
        }
        let exact: f64 = s
            .count
            .iter()
            .enumerate()
            .map(|(ci, &n)| t.per_class[ci].exact as f64 * n as f64)
            .sum::<f64>()
            / total as f64;
        assert!((exact - 41.50).abs() < 5e-3, "⌈log₂|classe|⌉ : {exact:.2}");

        // And the geometry the module header claims, measured rather than
        // assumed: every matrix's block count is a multiple of 32, so closing
        // at the boundary truncates nothing.
        assert_eq!(s.file[0].groups, 150_681_600 / 32);
        assert_eq!(s.file[0].partial, 0, "groupes partiels sur le 4B");
        assert_eq!(s.row_straddling, 252, "toutes les matrices enjambent des lignes");
    }
}
