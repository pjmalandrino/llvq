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
    assert_eq!(lg_ceil(1), 0, "a single choice costs no bit");
    assert_eq!(lg_ceil(2), 1);
    assert_eq!(lg_ceil(3), 2);
    assert_eq!(lg_ceil(4), 2);
    assert_eq!(lg_ceil(5), 3);
    assert_eq!(lg_ceil(4096), 12, "the codeword rank fits in 12 bits");
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
        assert!(rounded >= w.exact, "even class: rounding below the exact width");
        assert!(rounded <= w.exact + 5, "5 radices, 5 bits of waste at most");
    }
    for c in &cs.odd {
        let w = widths_of_odd(c);
        let rounded = w.radix2 - HEADER_BITS;
        assert!(rounded >= w.exact);
        assert!(rounded <= w.exact + 2, "2 radices on the odd side");
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
    assert!(trimmed > 0, "no even class benefits from the tightening");
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
        assert_eq!(sg, 0, "odd signs are computed, never stored");
        assert_eq!(a, 24 * lg_ceil(c.vals.len() as u128));
        assert!(
            w.golay_tight >= 46,
            "odd class at {} bits: the refuted 46 b variant is back",
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
        .unwrap_or_else(|| panic!("variant \"{name}\" missing from the menu"))
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
        assert_eq!(e, lg_ceil(u128::from(c.cardinality())), "even class {c:?}");
        want_sum += e;
    }
    for c in &cs.odd {
        let e = widths_of_odd(c).exact;
        assert_eq!(e, lg_ceil(u128::from(c.cardinality())), "odd class {c:?}");
        want_sum += e;
    }
    // And the same column as the sweep reads it, through `FastDecoder`'s
    // class order: `every_fastdecoder_class_has_its_radices` proves the
    // keys are a bijection, so the two totals must coincide.
    let fd = FastDecoder::new();
    let t = WidthTable::build(&fd);
    let got: u64 = t.per_class.iter().map(|w| w.exact).sum();
    assert_eq!(got, want_sum, "the FastDecoder order does not see the same classes");
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
            "even class w={} free_n={free_n}: {} bits against S_w·S_f",
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
    assert_eq!(GOLAY_MSG_BITS, GOLAY_RANK_BITS, "|Golay| = 2¹²: same width");
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
        assert!(w.exact >= floor, "e1v-packé even: {} < {floor}", w.exact);
        assert!(
            w.radix2 - HEADER_BITS >= floor,
            "e1v-séparé even: {} < {floor}",
            w.radix2 - HEADER_BITS
        );
    }
    for c in &cs.odd {
        let w = widths_of_odd(c);
        let floor = lg_ceil(u128::from(c.cardinality()));
        assert!(w.exact >= floor, "e1v-packé odd: {} < {floor}", w.exact);
        assert!(
            w.radix2 - HEADER_BITS >= floor,
            "e1v-séparé odd: {} < {floor}",
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
                "class {ci}: e1v-{tag} {} bits, header included, under {floor} + {HEADER_BITS}",
                row[k]
            );
        }
        // The packed variant is the floor itself, so any gap it shows is
        // the header and nothing else — the assertion above would still
        // pass if it silently grew, this one would not.
        assert_eq!(row[packed], floor + HEADER_BITS, "class {ci}");
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
        assert_eq!(row[split], row[r2], "class {ci}: the two widths have diverged");
    }
    assert_eq!(t.origin[split], t.origin[r2], "origin block");
    // The interpretation, which is the whole of the difference.
    assert!(!VARIANTS[r2].shift_only, "radix2 declared shift-only");
    assert!(VARIANTS[split].shift_only, "e1v-séparé is no longer shift-only");
    assert_eq!(VARIANTS[r2].depth, None, "the multiset rank is serial");
    assert_eq!(
        VARIANTS[split].depth,
        Some(96),
        "the binomial walk is 48-96 steps, and the depth must SAY so"
    );
    assert!(
        VARIANTS[split].depth.unwrap() > 24,
        "a depth ≤ 24 would disguise the reopening of the spec clause"
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
        assert_eq!(w.sign_field, even_sign_bits(c), "even sign field");
        assert!(
            w.golay_signs - HEADER_BITS >= w.exact,
            "even class: golay_signs {} under the rank {}",
            w.golay_signs - HEADER_BITS,
            w.exact
        );
    }
    for c in &cs.odd {
        let w = widths_of_odd(c);
        assert_eq!(w.sign_field, GOLAY_MSG_BITS, "odd sign field");
        assert!(
            w.golay_signs - HEADER_BITS >= w.exact,
            "odd class: golay_signs {} under the rank {}",
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
            "class {ci}: golay_signs under its own rank"
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
        "dropping the Golay message breaks no class: the test proves nothing"
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
            "odd golay_signs is no longer perslot + the message"
        );
        if w.perslot - HEADER_BITS < w.exact {
            short += 1;
        }
    }
    assert_eq!(cs.odd.len(), 157, "the odd class enumeration has moved");
    assert_eq!(
        short, 10,
        "10 of the 157 odd classes fall under their rank in perslot, not {short}"
    );
    // The even coset is sound in `perslot` — `nonzero ≥ (w−1) + free_n` —
    // so the defect is odd-only, and naming it that way is part of the
    // finding.
    for c in &cs.even {
        let w = widths_of_even(c);
        assert!(w.perslot - HEADER_BITS >= w.exact, "even class perslot under its own rank");
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
            assert!(free_n > 0, "empty even class");
            assert_eq!(truth, free_n);
            assert_eq!(truth, frozen + 1, "w = 0: the shortfall is not one bit");
            w0 += 1;
        } else {
            assert_eq!(frozen, truth, "w > 0: nonzero−1 IS (w−1) + free_n");
        }
        // Optimistic, bounded, and harmless to the bijection.
        assert!(truth >= frozen && truth - frozen <= 1);
        let w = widths_of_even(c);
        assert!(
            w.golay_tight - HEADER_BITS >= w.exact,
            "golay_tight under its own rank: the tightening is no longer valid"
        );
    }
    assert_eq!(cs.even.len(), 226, "the even class enumeration has moved");
    assert_eq!(w0, 17, "17 of the 226 even classes are at w = 0, not {w0}");
}

/// The two class sources must cover each other exactly, or the sweep
/// would price blocks against another class's radices.
#[test]
fn every_fastdecoder_class_has_its_radices() {
    let fd = FastDecoder::new();
    let cs = enumerate_classes(13);
    let n = cs.even.len() + cs.odd.len();
    assert_eq!(n, fd.n_classes(), "{n} classes enumerated, {} at the decoder", fd.n_classes());
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
    assert_eq!(set.len(), keys.len(), "two classes share one canonical key");
    for ci in 0..fd.n_classes() {
        let k = key_of_levels(fd.levels(ci));
        assert!(set.contains_key(&k), "class {ci} with no radices: {:?}", fd.levels(ci));
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
            "odd cardinality"
        );
    }
}

/// The criterion is stated in the kernel accounting, so the conversion
/// must be the same one `rtbits` and the CUDA bench use: `Planes14`'s 112
/// bits per block are 4.804 b/weight on the 4B.
#[test]
fn kernel_conversion_reproduces_planes14() {
    let p14 = kernel_bpw(112.0);
    assert!((p14 - 4.804).abs() < 5e-4, "Planes14: {p14:.4}");
    // And the archive's own 48 bits, the floor E3 is chasing.
    let arch = kernel_bpw(ARCHIVE_BITS as f64);
    assert!(arch < E3_CRITERION, "the file itself is worth {arch:.4}");
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
    assert_eq!(flat.max_bits, 32 * 48 + 32, "47 rounded to 48 per lane");
    let mut ws = [47u64; 32];
    ws[13] = 94;
    let spiked = group_of(&ws);
    assert_eq!(spiked.max_bits, 32 * 96 + 32, "one block at 94 makes all 32 pay 96");
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
/// ("Σ + 32 of base offset, rounded to the 32-bit word") has only one
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
    assert_eq!(g.max_bits, 32 * 48 + 32, "the 32 lanes are addressable");
    // 5·48 = 240 bits, which is 7.5 words: the group rounds to 8 (256) and
    // then pays its base word. Charging 240 + 32 would be the off-by-one
    // this assertion exists to catch.
    assert_eq!(g.scan_bits, 256 + 32, "5 blocks rounded to the word, not 32 lanes");
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
    assert!(!(blocks as u64).is_multiple_of(GROUP), "51 must cut a group");
    let matrix = |name: &str| llvq_artifact::RawMatrix {
        name: name.to_string(),
        d_out: rows,
        d_in: per_row * DIM,
        kind: llvq_artifact::CodeKind::Ball,
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
    assert_eq!(s.origins, 0, "every test index must be inside the ball");
    let single = s.file.clone();
    for (k, v) in VARIANTS.iter().enumerate() {
        assert_eq!(
            (single[k].groups, single[k].partial),
            (2, 1),
            "{}: the matrix end did not close the partial group",
            v.name
        );
        assert_eq!(single[k].filled, 0, "{}: 19 lanes left open", v.name);
    }

    s.push_matrix(&fd, &t, &matrix("B"));
    for (k, v) in VARIANTS.iter().enumerate() {
        let g = s.file[k];
        assert_eq!(
            (g.groups, g.partial),
            (4, 2),
            "{}: the two matrices merged into a single stream",
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
        assert_eq!(joined.groups, 3, "the continuous stream would make only 3 full groups");
        assert_eq!(joined.filled, 102 - 3 * GROUP, "6 blocks are left dangling");
        assert_ne!(
            (g.groups, g.partial, g.max_bits, g.scan_bits),
            (joined.groups, joined.partial, joined.max_bits, joined.scan_bits),
            "{}: closing at the boundary changes nothing, so nothing proves we close",
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
    assert_eq!(split.max_bits, 2 * joined.max_bits, "two groups paid in full");
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
    assert_eq!(MAX_DEPTH, 24, "the spec clause is ≤ 24 steps");

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
                    "{} is shift-only at depth {:?}: the spec CONJUNCTION requires it to be \
                     a REOPENING, never an admission",
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
        "no shift-only variant of depth > {MAX_DEPTH} on the menu: this test no longer \
         tells the conjunction apart from shift_only alone"
    );
    // And the one that made the defect matter, by name.
    let split = &VARIANTS[variant("e1v-séparé")];
    assert!(split.shift_only, "e1v-séparé is no longer shift-only");
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
        name: "hypothetical",
        mode: Mode::WarpScan,
        bpw: 2.3709,
        depth: Some(24),
    };

    // A reopening alone: no admission sentence, and the reopening named.
    let t = verdict_text(&[], std::slice::from_ref(&deep), f64::INFINITY);
    assert!(!t.contains(OPEN_AT_SPEC), "reopening published as an admission:\n{t}");
    assert!(t.contains("NO ADMISSIBLE decomposition"), "{t}");
    assert!(t.contains("OUTSIDE THE DEPTH CLAUSE"), "{t}");
    assert!(t.contains("REOPENING"), "{t}");
    assert!(t.contains("2.3709"), "the point must stay a figure:\n{t}");

    // The same point, plus a genuinely admissible one: the sentence comes
    // back, and the reopening is still reported separately rather than
    // folded into the admitted list.
    let t2 = verdict_text(std::slice::from_ref(&shallow), std::slice::from_ref(&deep), 2.3709);
    assert!(t2.contains(OPEN_AT_SPEC), "{t2}");
    assert!(t2.contains("OUTSIDE THE DEPTH CLAUSE"), "{t2}");
    assert!(t2.contains("hypothetical"), "{t2}");

    // And nothing at all under the criterion: the plain red verdict, with
    // the best ADMISSIBLE point — never the best shift-only one.
    let t3 = verdict_text(&[], &[], 2.9650);
    assert!(!t3.contains(OPEN_AT_SPEC), "{t3}");
    assert!(t3.contains("The best ADMISSIBLE point is 2.9650"), "{t3}");
    assert!(!t3.contains("REOPENING"), "{t3}");
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
        assert_eq!(t.row(ci)[g70], flat, "golay70 is no longer fixed-width");
    }
    assert_eq!(
        t.origin[g70], HEADER_BITS,
        "the golay70 origin tariff has changed: the assertion message in main() lies"
    );
    assert!(t.origin[g70] < flat, "the defect is an UNDER-charge, and that has to be said");
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
        "the label reintroduces the false equality with the served layout"
    );
    for ci in 0..fd.n_classes() {
        assert!(
            t.row(ci)[k] < PLANES14_BITS,
            "class {ci}: perslot {} reaches the flat {PLANES14_BITS} b/block of Planes14",
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
            "odd perslot is no longer 12 bits under the decodable width"
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
    assert_eq!(n, 383, "the synthetic menu assumes the 383 classes");
    let mut indices: Vec<u64> = (0..n).map(|ci| fd.class_range(ci).0).collect();
    indices.push(0); // the origin
    let blocks = indices.len();
    let m = llvq_artifact::RawMatrix {
        name: "synthetic".to_string(),
        d_out: 8,
        d_in: 48 * DIM,
        kind: llvq_artifact::CodeKind::Ball,
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
    assert_eq!(s.origins, 1, "the origin must go through the origin tariff");
    assert!(s.count.iter().all(|&c| c == 1), "a class the sweep did not see");

    let cm = sweep_class_major(&t, &s.count, s.origins);
    assert_eq!(cm.len(), NVAR, "class-major order: {} accumulators", cm.len());
    assert_eq!(s.file.len(), NVAR, "file order: {} accumulators", s.file.len());
    assert_eq!(NVAR, VARIANTS.len());
    for (k, v) in VARIANTS.iter().enumerate() {
        for (g, tag) in [(cm[k], "CM"), (s.file[k], "FO")] {
            for mode in Mode::ALL {
                assert!(
                    g.bits(mode) > 0,
                    "{} not priced in {tag}/{}",
                    v.name,
                    mode.tag()
                );
            }
            assert!(
                g.bits(Mode::WarpScan) <= g.bits(Mode::Grp32Max),
                "{}: {tag}/scan above {tag}/max",
                v.name
            );
            assert_eq!(g.groups, blocks as u64 / GROUP, "{} in {tag}", v.name);
            assert_eq!(g.partial, 0, "{} in {tag}", v.name);
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

    // radixstudy-x4-2026-08-12.txt, arm 1: mean · worst · grp32 · b/w mean · b/w GRP
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
        "the menu has shrunk: the published control is no longer reproducible"
    );
    for (k, (name, moy, pire, grp, bpw_moy, bpw_grp)) in want.iter().enumerate() {
        assert_eq!(VARIANTS[k].name, *name, "the menu order has changed");
        let mean = mean_bits(&t, &count, 0, k);
        let grouped = cm[k].bits(Mode::Grp32Max) as f64 / total as f64;
        assert!((mean - moy).abs() < 5e-3, "{name} mean: {mean:.2} against {moy}");
        assert_eq!(worst_bits(&t, &count, 0, k), *pire, "{name} worst");
        assert!((grouped - grp).abs() < 5e-3, "{name} grp32: {grouped:.2} against {grp}");
        let (a, b) = (shapes.kernel_bpw(mean), shapes.kernel_bpw(grouped));
        assert!((a - bpw_moy).abs() < 5e-5, "{name} b/w mean: {a:.4}");
        assert!((b - bpw_grp).abs() < 5e-5, "{name} b/w GRP: {b:.4}");
    }
    // The `exact` column of the same arm, and the Planes14 control that
    // ties this table to the 4.804 of the production layout.
    let exact: f64 = (0..fd.n_classes())
        .map(|ci| t.per_class[ci].exact as f64)
        .sum::<f64>()
        / total as f64;
    assert!((exact - 33.40).abs() < 5e-3, "⌈log₂|class|⌉: {exact:.2}");
    assert!((kernel_bpw(112.0) - 4.804).abs() < 5e-4);
}

/// The streaming glue, on **real** blocks: header, one matrix, the
/// per-block class lookup, the group boundary and the matrix boundary.
///
/// One matrix of the sealed 4B is 108,544 blocks and a couple of megabytes
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
        "sealed archive missing: {path} (LLVQ_Q4B moves the search, it does not satisfy it)"
    );
    let fd = FastDecoder::new();
    let t = WidthTable::build(&fd);
    let f = File::open(&path).expect("open");
    let mut r = BufReader::with_capacity(1 << 20, f);
    let h = llvq_artifact::read_header(&mut r).expect("valid header");
    assert_eq!(h.version, 1, "the published 4B is an LVQ1, \"projections only\"");
    let m = llvq_artifact::read_matrix_raw(&mut r, h.version).expect("valid matrix");
    assert_eq!(m.name, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!((m.d_out, m.d_in), (4096, 2560));

    let mut s = Sweep::empty(fd.n_classes());
    s.push_matrix(&fd, &t, &m);
    let blocks = 4096 * (2560 / DIM) as u64;
    assert_eq!(s.total(), blocks, "434,176 blocks: 4096 rows of 106");
    // 106 blocks per row is not a multiple of 32: groups straddle rows.
    assert_eq!(s.row_straddling, 1);
    // 434,176 blocks is 13,568 whole groups — nothing partial, so closing
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
        "file order {} ≤ class-major control {}: the control is no longer optimistic",
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
        "sealed archive missing: {path} (LLVQ_Q4B moves the search, it does not satisfy it)"
    );
    let fd = FastDecoder::new();
    let t = WidthTable::build(&fd);
    let s = sweep_file(&fd, &t, &path);
    let total = s.total();
    assert_eq!(total, 150_681_600, "this is not the published 4B");
    assert_eq!(s.matrices, 252);
    assert_eq!(s.origins, 0, "the published 4B has no origin block");
    // The shapes read from the file ARE the pinned constants — the whole
    // licence for reading them from the file instead of remembering them.
    assert_eq!(s.shapes, Shapes::qwen3_4b(), "shapes read ≠ shapes pinned");

    let cm = sweep_class_major(&t, &s.count, s.origins);
    // radixstudy-x4-2026-08-12.txt, arm 2: mean · worst · grp32 · b/w GRP
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
        assert!((mean - moy).abs() < 5e-3, "{name} mean: {mean:.2} against {moy}");
        assert_eq!(worst_bits(&t, &s.count, s.origins, k), *pire, "{name} worst");
        assert!((grouped - grp).abs() < 5e-3, "{name} grp32: {grouped:.2} against {grp}");
        let b = s.shapes.kernel_bpw(grouped);
        assert!((b - bpw_grp).abs() < 5e-5, "{name} b/w GRP: {b:.4} against {bpw_grp}");
    }
    let exact: f64 = s
        .count
        .iter()
        .enumerate()
        .map(|(ci, &n)| t.per_class[ci].exact as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    assert!((exact - 41.50).abs() < 5e-3, "⌈log₂|class|⌉: {exact:.2}");

    // And the geometry the module header claims, measured rather than
    // assumed: every matrix's block count is a multiple of 32, so closing
    // at the boundary truncates nothing.
    assert_eq!(s.file[0].groups, 150_681_600 / 32);
    assert_eq!(s.file[0].partial, 0, "partial groups on the 4B");
    assert_eq!(s.row_straddling, 252, "every matrix straddles rows");
}
