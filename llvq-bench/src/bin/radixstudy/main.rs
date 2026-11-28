//! **X4 — the E3 study, on paper.** What an index designed *for the decoder*
//! would cost, priced exactly on the 384 classes and, when an artifact is
//! given, weighted by the real blocks of the published 4B.
//!
//! `docs/archive/spec-memoire-extreme-2026-08-12.md` opens the E3 workstream only if a
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
//! which is the cheap half of the verdict and the one that closes the workstream.
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
//! | `golay70` | the measured E2 layout | ✅ depth 24, **1.31×, dropped** |
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
//! for the handover, not for a counter of bits.
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
//! "E3 is open in the sense of the spec" on a `depth: Some(96)` point.
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

mod addressing;
mod sweeps;
mod verdict;
mod widths;

use addressing::*;
use sweeps::*;
use verdict::*;
use widths::*;

fn main() {
    let fd = FastDecoder::new();
    let t = WidthTable::build(&fd);

    // ---- weights: real blocks if we have a file, else one per class ----
    let (s, source) = match std::env::args().nth(1) {
        Some(path) => {
            let s = sweep_file(&fd, &t, &path);
            let src = format!("{path} — {} matrices, real blocks", s.matrices);
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
            (s, "the 383 classes at equal weight (WARNING: NOT the real distribution)".to_string())
        }
    };
    let have_file = !s.file.is_empty();
    let total = s.total();
    let cm = sweep_class_major(&t, &s.count, s.origins);
    // The file's own shapes when there is one, the 4B constants when there is
    // not — set once, at construction, so the two can never drift apart.
    let shapes = s.shapes;

    println!("source: {source}");
    println!("{total} blocks, {} classes", fd.n_classes());
    // The origin tariff is `Widths::default()`, i.e. 10 bits, which is right
    // for a variable-width variant and **wrong** for a fixed-width one: an
    // origin in `golay70` still occupies its flat 70-bit record. No sealed
    // artifact has ever contained one, so rather than repair a tariff nothing
    // can check, refuse to print a table that would silently under-charge —
    // by name, and saying what to fix. See the module header.
    assert_eq!(
        s.origins,
        0,
        "{} origin blocks in this file, and this bench's origin tariff is WRONG for the \
         FIXED-WIDTH variants: WidthTable::build reads (v.get)(&Widths::default()).max(10), \
         which charges {} bits to \"golay70 (mesuré, écarté)\" instead of its flat 70 bits. \
         Fix the origin tariff variant by variant BEFORE publishing a column on this \
         file: no row of this run would be readable",
        s.origins,
        t.origin[VARIANTS
            .iter()
            .position(|v| v.name.starts_with("golay70"))
            .expect("golay70 on the menu")]
    );
    if have_file {
        let g = s.file[0];
        println!(
            "{} groups of 32 in file order, {} of them partial (closed at a matrix end)",
            g.groups, g.partial
        );
        println!(
            "{}/{} matrices have blocks/row not a multiple of 32: a group there straddles two rows",
            s.row_straddling, s.matrices
        );
        println!(
            "kernel accounting READ FROM THE FILE: {} weights, {} of tail, {} rows, {} blocks",
            shapes.weights, shapes.tail_weights, shapes.rows, shapes.blocks
        );
        if shapes != Shapes::qwen3_4b() {
            // Two accountings in one dossier is exactly how a b/weight figure
            // goes wrong (CLAUDE.md §7, rule 1). Name the gap rather than let
            // a reader put this table beside a 4B one.
            println!(
                "WARNING: THIS IS NOT the 4B accounting: 112 b/block are worth {:.4} here,\n\
                 against {:.4} on the published 4B. No row of this run compares to a 4B figure.",
                shapes.kernel_bpw(112.0),
                kernel_bpw(112.0)
            );
        }
        assert_eq!(
            shapes.blocks, total,
            "the blocks counted ({total}) are not the ones the shapes give ({})",
            shapes.blocks
        );
    } else {
        println!("no file: no real ORDER, hence no file-order column and no verdict");
    }

    // ---- geometry of the group, stated rather than assumed ----
    println!(
        "\n  group geometry (verified, not assumed): 32 CONSECUTIVE blocks of m.indices,\n  \
         row-major d_out × (d_in/24) — e1c.rs:204, runtime.rs:491, format.rs:199. A group NEVER\n  \
         crosses a matrix; it crosses rows as soon as (d_in/24) mod 32 ≠ 0."
    );

    // ---- the menu, priced: bits per block ----
    println!("\n  bits per block — the 4 combinations (scan order × addressing)");
    println!("  {}", "-".repeat(98));
    println!(
        "  {:<24}{:>7}{:>6}{:>9}{:>9}{:>9}{:>9}{:>7}  depth",
        "variant",
        "mean",
        "worst",
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
            if v.shift_only { "yes" } else { "NO" },
            v.depth.map_or("ser.".to_string(), |d| format!("{d}")),
        );
    }
    println!("  {}", "-".repeat(98));
    println!(
        "  CM = class-major — non-regression CONTROL, optimistic by construction:\n  \
         a group there is almost always of one single class. Not citable (prereg §1).\n  \
         FO = file order — what a kernel reads. max = uniform stride at the widest block of\n  \
         the group; scan = exact widths, offsets by warp prefix sum."
    );

    // ---- the same bits per block at 3 decimals, because the decision sections
    // quote 3 and the table above prints 2. A number inverted out of a
    // 4-decimal b/weight column is a transcription, not an output: this block
    // exists so every figure a verdict cites is something a command printed.
    if have_file {
        println!("\n  the SAME bits/block at 3 decimals — the table above rounds to 2, and a");
        println!("  3-decimal figure does not follow from it. Δ = the cost of the real order.");
        println!("  {}", "-".repeat(98));
        println!(
            "  {:<24}{:>10}{:>10}{:>10}{:>10}{:>14}",
            "variant",
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
    println!("\n  KERNEL b/weight — same 4 combinations, and the two preregistered thresholds");
    println!(
        "  S_spec = {E3_CRITERION:.2} b/weight ⇔ {sp:.2} bits/block (prereg: {:.2})   ·   \
         S_alt = {S_ALT:.2} ⇔ {sa:.2} (prereg: {:.2})",
        PREREG_BITS.0, PREREG_BITS.1
    );
    println!("  {}", "-".repeat(98));
    println!(
        "  {:<24}{:>9}{:>9}{:>9}{:>9}   {:<14}FO vs S_alt",
        "variant",
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
        "\n  MARKER LEGEND — and the ⊘ is the half that counts:\n  \
         ✅ = under S_spec ({E3_CRITERION:.2}), the project's only threshold. It is the only\n  \
            marker that means \"it passes\", and it prints nowhere else.\n  \
         ⊘ = under S_alt ONLY, so in the band ]{E3_CRITERION:.2} ; {S_ALT:.2}]. NEVER CITED\n  \
            ALONE: S_alt is not the project's threshold, it exists only at a frozen 8k context\n  \
            and would become opposable only if the product note of prereg §5 kept the\n  \
            triple (32 GB; 2 GB; KV q8; 8k). THAT NOTE DOES NOT EXIST. A ⊘ is published as\n  \
            \"it would pass if and only if that triple were kept\", never as \"it passes\".\n  \
         ❌ = above both."
    );

    // ---- where the archive's 47 bits actually go ----
    let exact_mean: f64 = s
        .count
        .iter()
        .enumerate()
        .map(|(ci, &n)| t.per_class[ci].exact as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    println!("\n  where the archive's 47 bits go");
    println!("  {}", "-".repeat(98));
    println!("  point WITHIN its class, mean of ⌈log₂ |class|⌉: {exact_mean:.2} bits");
    println!(
        "  choice of the class: 47 − {exact_mean:.2} ≈ {:.2} bits, which every variant with an\n  \
         explicit class field pays back as {HEADER_BITS} header bits",
        47.0 - exact_mean
    );
    println!("  e1v-packé pays {exact_mean:.2} + {HEADER_BITS} bits per block, addressing aside");

    // ---- T3: what the sign field costs, against the flat mask it replaces ----
    let sign_mean: f64 = s
        .count
        .iter()
        .enumerate()
        .map(|(ci, &n)| t.per_class[ci].sign_field as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    println!("\n  what the golay_signs sign field costs");
    println!("  {}", "-".repeat(98));
    println!(
        "  mean of the field: {sign_mean:.2} bits against the 24 of the flat mask the served\n  \
         layouts spend, so {:.2} bits/block of expected saving (prereg §3 T3 predicts −9.85).\n  \
         WARNING: an expectation ≠ what a stream pays. The max columns above charge the widest\n  \
         block of the group, and that is where T3 is judged.",
        24.0 - sign_mean
    );

    // ---- the verdict, taken on the file order only ----
    println!("\n  VERDICT X4");
    println!("  {}", "-".repeat(98));
    if !have_file {
        println!(
            "  PAUSED: no verdict without a file. The only column available is the\n  \
             class-major control, which prereg §1 forbids citing in a verdict."
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
#[cfg(test)]
mod tests;
