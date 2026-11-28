//! Unit tests of `smoke`: the strict parsers of the positionals and the
//! calibration windows. A file of their own because they outweigh the code.

use super::*;
use super::codebook::class_count;

const N: usize = 8;
const LEN: usize = 2048;
const TOKENS: usize = 4_000_000;

fn shape_gain(spec: &str) -> (u32, u32, bool, usize) {
    match parse_codebook(spec).unwrap_or_else(|e| panic!("{spec}: {e}")) {
        Codebook::ShapeGain {
            gain_bits,
            max_shell,
            free_magnitude,
            level_cap,
        } => (gain_bits, max_shell, free_magnitude, level_cap),
        other => panic!("{spec} resolved to {other:?}, not a shape–gain"),
    }
}

/// Every spelling this repo has ever launched has to keep meaning exactly
/// what it meant, or the strict parser silently reinterprets the published
/// runs instead of protecting them.
#[test]
fn the_documented_spellings_still_resolve() {
    // `leech` is the default when the positional is absent, and has always
    // meant one gain bit over the full ball.
    assert_eq!(shape_gain("leech"), (1, 13, false, 5));
    assert_eq!(shape_gain("leech1"), (1, 13, false, 5));
    // The sealed 4B artifact.
    assert_eq!(shape_gain("leech1c12"), (1, 12, false, 5));
    // The 8B run.
    assert_eq!(shape_gain("leech1c12L3"), (1, 12, false, 3));
    // The free-magnitude arm of docs/archive/retraction-et-gain.md.
    assert_eq!(shape_gain("leech1c12f"), (1, 12, true, 5));
    assert_eq!(shape_gain("leech0"), (0, 13, false, 5));
    assert_eq!(shape_gain("leech2c12"), (2, 12, false, 5));
    assert!(matches!(
        parse_codebook("identity").unwrap(),
        Codebook::Identity
    ));
    assert!(matches!(
        parse_codebook("grid").unwrap(),
        Codebook::Grid { .. }
    ));
    assert!(matches!(
        parse_codebook("direction").unwrap(),
        Codebook::Direction
    ));
}

/// The Tetra arm parses, at one gain bit and nothing else.
///
/// `tetra` and `tetra1` are the same object on purpose — the bare token is
/// what an operator types, and it must not mean "some default gain width"
/// the way `leech` had to be pinned to mean one bit.
#[test]
fn the_tetra_arm_parses_and_never_falls_back() {
    assert!(matches!(
        parse_codebook("tetra").unwrap(),
        Codebook::Tetra {
            gain_bits: 1,
            post_shape_gain: false
        }
    ));
    assert!(matches!(
        parse_codebook("tetra1").unwrap(),
        Codebook::Tetra {
            gain_bits: 1,
            post_shape_gain: false
        }
    ));
    assert!(matches!(
        parse_codebook("tetrapost").unwrap(),
        Codebook::Tetra {
            gain_bits: 1,
            post_shape_gain: true
        }
    ));
    assert!(matches!(
        parse_codebook("tetra1post").unwrap(),
        Codebook::Tetra {
            gain_bits: 1,
            post_shape_gain: true
        }
    ));
    for spec in [
        "tetra0",     // no gain bit: the word has one whether it is spent or not
        "tetra2",     // two: the word has room for one, at bit 47
        "tetraf",     // free magnitude: `leech1c12f`'s arm, at a rate this map cannot have
        "tetra1f",    // …spelled the other way
        "tetrac12",   // a shell cap: Tetra's label set is one fixed trellis
        "tetra1c12",  // …with the gain in front of it
        "tetra1L3",   // a level cap, same reason
        "tetrao",     // trailing junk
        "tetra ",     // the trailing space, the classic shell accident
        "TRIO",      // the prefix is case-sensitive, as `leech` and `int` are
        "Tetra",
        "tetra-1",
    ] {
        assert!(
            parse_codebook(spec).is_err(),
            "{spec:?} was accepted instead of refused"
        );
    }
}

/// 🚨 The two 48-bit arms must report the **same** rate and the same block
/// length, or the A/B that reads them moves two things at once.
///
/// This is the equality `bin/smoke`'s own step (c) checks to the fourth
/// decimal on a real run: 47 index bits over `Λ₂₄(12)` plus one gain bit
/// on one side, 47 bits of Tetra label plus one gain bit on the other.
#[test]
fn the_tetra_arm_spends_the_same_forty_eight_bits_as_the_ball() {
    let t = parse_codebook("tetra").unwrap();
    let b = parse_codebook("leech1c12").unwrap();
    assert_eq!(t.block_bits(), 48.0);
    assert_eq!(t.block_bits(), b.block_bits());
    assert_eq!(t.block_len(), 24);
    assert_eq!(t.block_len(), b.block_len());
    // …and they are not the same file: the kind is what says so, and it is
    // the only thing that does.
    assert_eq!(t.code_kind(), llvq_artifact::CodeKind::Tetra);
    assert_eq!(b.code_kind(), llvq_artifact::CodeKind::Ball);
    assert_eq!(
        parse_codebook("tetrapost").unwrap().block_bits(),
        48.0
    );
}

/// The journal line has to name the map, not only the rate — the two arms
/// spend the same 48 bits, so a line that printed the rate alone would
/// describe both files identically.
#[test]
fn the_journal_line_names_the_tetra_map() {
    let t = codebook_line("tetra", &parse_codebook("tetra").unwrap());
    assert!(t.contains("Tetra"), "{t}");
    assert!(t.contains("47 bits of label"), "{t}");
    assert!(t.contains("1 gain bit,"), "{t}");
    assert!(t.contains("48 bits/block"), "{t}");
    // Nothing of the ball's vocabulary: no shell, no class count.
    assert!(!t.contains("shell"), "{t}");
    assert!(!t.contains("classes"), "{t}");
    assert!(t.contains("from source norm"), "{t}");
    let post = codebook_line("tetrapost", &parse_codebook("tetrapost").unwrap());
    assert!(post.contains("after direction"), "{post}");
    assert_ne!(
        t, post,
        "the experimental arm must be explicit in the journal"
    );
    let b = codebook_line("leech1c12", &parse_codebook("leech1c12").unwrap());
    assert_ne!(t, b, "the two 48-bit arms must not print the same line");
}

/// **The $12.61 test.** One letter off used to resolve to a *different*
/// codebook at a different rate, and the log printed the string that was
/// asked for, so nothing looked wrong.
///
/// Each case below is a way the old parser fell back, not a made-up typo:
/// a bad prefix (`unwrap_or("1")`), an unreadable shell (`unwrap_or(13)`),
/// the wrong case (`split_once('c')` is case-sensitive), an unreadable
/// level cap (`unwrap_or(5)`), and fields out of order (`L` was split
/// before `c`, so the tail was swallowed whole).
#[test]
fn a_misspelled_codebook_is_refused() {
    for spec in [
        "leach1c12",  // the audit's typo → used to be leech1, full ball
        "Leech1c12",  // capitalized → used to be leech1
        "leech1C12",  // capital C → used to be the full ball
        "leech1cx",   // unreadable shell → used to be the full ball
        "leech1c",    // bare tag → used to be the full ball
        "leech1c12L", // bare tag → used to be 5 levels
        "leech1c12Lx",
        "leech1L3c12", // fields swapped → used to drop *both*
        "leech1c12x",  // trailing junk → used to be ignored
        "leech1c12 ",  // trailing space, the classic shell accident
        "leechx",
        "",
        "leech-1",
        "identity2",
        "gridd",
    ] {
        assert!(
            parse_codebook(spec).is_err(),
            "{spec:?} was accepted instead of refused"
        );
    }
}

/// Ranges the encoder asserts on hours into a run, asserted here in
/// microseconds instead. `set_shell_cap` panics outside `2..=13`, and a
/// level cap of 0 leaves the searcher with no classes at all.
#[test]
fn out_of_range_caps_are_refused() {
    for spec in [
        "leech1c14",
        "leech1c1",
        "leech1c0",
        "leech1c99",
        "leech9",
        "leech64",
        "leech1c12L0",
        "leech1c12L6",
        "leech1c12L99",
    ] {
        assert!(
            parse_codebook(spec).is_err(),
            "{spec:?} was accepted instead of refused"
        );
    }
    // …and the edges of those ranges stay legal, or the guard is a wall
    // rather than a range.
    assert_eq!(shape_gain("leech1c2").1, 2);
    assert_eq!(shape_gain("leech1c13").1, 13);
    assert_eq!(shape_gain("leech1c12L5").3, 5);
    assert_eq!(shape_gain("leech8").0, 8);
}

/// A codebook can be **empty** while every individual range holds, and
/// that combination has to be refused on its own.
///
/// This test exists because the guard survived mutation without it: both
/// caps were in range on every case the other tests tried, so deleting the
/// emptiness check changed nothing they could see. The measured grid says
/// otherwise — `class_count(m, 1)` is **0** for every shell up to 5, and
/// only reaches 1 at shell 6 — so `leech1c5L1` passes `2..=13` and
/// `1..=5` and leaves the searcher with nothing to search. Same lesson as
/// CLAUDE.md §5: an assertion that never exercises the parameter it covers
/// tests nothing.
#[test]
fn an_empty_codebook_is_refused() {
    for spec in ["leech1c2L1", "leech1c3L1", "leech1c4L1", "leech1c5L1"] {
        assert_eq!(class_count_of(spec), 0, "{spec} is not empty after all");
        assert!(
            parse_codebook(spec).is_err(),
            "{spec:?} is an empty codebook and was accepted"
        );
    }
    // Shell 6 is where a single class appears — degenerate but searchable,
    // so it stays legal. The guard is an emptiness check, not a taste one.
    assert_eq!(class_count(6, 1), 1);
    assert_eq!(shape_gain("leech1c6L1"), (1, 6, false, 1));
}

/// The class count the parser would compute for `spec`, ignoring whether
/// it accepts it — so the test above can assert emptiness *and* refusal
/// rather than assuming they coincide.
fn class_count_of(spec: &str) -> usize {
    let (shell, levels) = spec
        .trim_start_matches("leech1c")
        .split_once('L')
        .expect("the probes above are all `leech1c<m>L<l>`");
    class_count(shell.parse().unwrap(), levels.parse().unwrap())
}

/// The class count is the one number that separates two codebooks at a
/// glance in an archived log, so it has to be the real one: 301 classes
/// for Λ₂₄(12) — 180 even and 121 odd — against 383 for Λ₂₄(13)
/// (CLAUDE.md, §une seule coquille). A count computed on the wrong shell
/// would make the journal line agree with itself and with nothing else.
#[test]
fn the_class_count_is_the_documented_one() {
    assert_eq!(class_count(12, llvq_search::generic::MAX_LEVELS_ANY), 301);
    assert_eq!(class_count(13, llvq_search::generic::MAX_LEVELS_ANY), 383);
    // The level cap has to actually cut, or `L3` would be decoration.
    assert!(class_count(12, 3) < 301);
    assert!(class_count(12, 3) > 0);
}

/// The defect this file existed with on all three published runs: a
/// hard-coded "0 gain bits" on a configuration that spends one. The line
/// has to report the resolved value, and it has to carry enough to name
/// the object — shell cap, level cap, class count, bits per block.
#[test]
fn the_journal_line_reports_the_real_gain_bits() {
    let line = codebook_line("leech1c12", &parse_codebook("leech1c12").unwrap());
    // The trailing comma is load-bearing: without it `"1 gain bit"` is a
    // substring of `"1 gain bits"`, and a frozen plural survives the test.
    assert!(line.contains("1 gain bit,"), "{line}");
    assert!(!line.contains("0 gain"), "{line}");
    assert!(line.contains("shell ≤ 12"), "{line}");
    assert!(line.contains("301 classes"), "{line}");
    // 47 index bits over Λ₂₄(12) plus one gain bit — the sealed artifact's
    // rate, and the number the whole G5 section is written around.
    assert!(line.contains("48 bits/block"), "{line}");

    let zero = codebook_line("leech0c12", &parse_codebook("leech0c12").unwrap());
    assert!(zero.contains("0 gain bits"), "{zero}");
    assert!(zero.contains("47 bits/block"), "{zero}");

    // And the two must not print the same thing, which is what a literal
    // did.
    assert_ne!(line, zero);
}

/// A free magnitude is not a 2-bit code, and the line has to say so:
/// mistaking one for the other is the accounting error of 2026-07-31,
/// 2.1117 announced for 2.7338 real.
#[test]
fn the_journal_line_separates_a_free_magnitude() {
    let f = codebook_line("leech1c12f", &parse_codebook("leech1c12f").unwrap());
    assert!(f.contains("free f16"), "{f}");
    assert!(f.contains("63 bits/block"), "{f}");
    let g = codebook_line("leech1c12", &parse_codebook("leech1c12").unwrap());
    assert_ne!(f, g);
}

/// **The $56 test.** `blocks` mistyped used to become `usize::MAX`, i.e.
/// the full run — the de-risking pass turning into the thing it exists to
/// de-risk.
#[test]
fn a_malformed_positional_is_refused_not_defaulted() {
    let a: Vec<String> = [
        "64",
        "2048",
        "12",
        "4096",
        "metal",
        "nogs",
        "leech1c12",
        "4x",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let e = arg::<usize>(&a, 7, "blocks", usize::MAX).unwrap_err();
    assert!(format!("{e}").contains("blocks"), "{e}");
    // An empty positional is a shell accident (`$UNSET` expanded), not a
    // request for the default.
    let empty = vec![String::new()];
    assert!(arg::<usize>(&empty, 0, "n_calib", 16).is_err());
    // Absent stays the default, or every short invocation in the docs
    // stops working.
    assert_eq!(arg::<usize>(&a, 9, "blocks", 7).unwrap(), 7);
    assert_eq!(arg::<usize>(&a, 7, "blocks", usize::MAX).ok(), None);
    assert_eq!(arg::<usize>(&a, 0, "n_calib", 16).unwrap(), 64);
}

/// `gs` / `nogs` / `dc` / `sph`, and nothing that merely looks like them.
#[test]
fn a_misspelled_mode_is_refused() {
    assert_eq!(Mode::parse(None).unwrap(), Mode::NoGs);
    assert_eq!(Mode::parse(Some("nogs")).unwrap(), Mode::NoGs);
    assert_eq!(Mode::parse(Some("gs")).unwrap(), Mode::Gs);
    assert_eq!(Mode::parse(Some("dc")).unwrap(), Mode::DesignC);
    assert_eq!(Mode::parse(Some("sph")).unwrap(), Mode::Spherical);
    for bad in ["", "GS", "gsp", "no-gs", "designc", "off", "spherical", "SPH", "sph "] {
        assert!(Mode::parse(Some(bad)).is_err(), "{bad:?} was accepted");
    }
}

/// `ops/run.py` passes `norot` when the rotation is off and the docs pass
/// `rot`; anything else used to silently mean off — and the rotation is
/// the single largest quality lever measured in this project (×2.290 →
/// ×1.811).
#[test]
fn a_misspelled_rotation_is_refused() {
    assert_eq!(parse_rotation(None).unwrap(), None);
    assert_eq!(parse_rotation(Some("norot")).unwrap(), None);
    assert_eq!(parse_rotation(Some("rot")).unwrap(), Some(ROTATION_SEED));
    for bad in ["", "ROT", "rott", "on", "true", "1"] {
        assert!(parse_rotation(Some(bad)).is_err(), "{bad:?} was accepted");
    }
}

/// The affine scalar arm parses, and **nothing about it falls back**.
///
/// The group is not a cosmetic field: it *is* `GptqConfig::block` (see
/// `Codebook::block_len`), so a defaulted one would change the run's
/// error-feedback granularity and not merely its rate. That is why
/// `int3` alone is an error rather than "group 128".
#[test]
fn the_scalar_arm_parses_and_never_falls_back() {
    assert!(matches!(
        parse_codebook("int3g24").unwrap(),
        Codebook::ScalarGroup { bits: 3, group: 24 }
    ));
    assert!(matches!(
        parse_codebook("int4g128").unwrap(),
        Codebook::ScalarGroup {
            bits: 4,
            group: 128
        }
    ));
    for spec in [
        "int3",      // no group — would have to be invented
        "intg24",    // no bits
        "int3g",     // tag with no digits behind it
        "int0g24",   // zero bits
        "int9g24",   // past the 8-bit bound
        "int3g4",    // group below the f16 overhead bound
        "int3g2048", // group past the narrowest d_in this harness loads
        "int3g24x",  // trailing junk
        "int3G24",   // the tag is case-sensitive, as `c` is for leech
    ] {
        assert!(
            parse_codebook(spec).is_err(),
            "{spec:?} should be refused, not resolved"
        );
    }
}

/// 🚨 The block length follows the codebook, and the rate follows the
/// block length.
///
/// While every codebook was 24 wide, `Report::bits_per_weight` could
/// divide by `llvq_core::DIM` and be right by accident. The first one
/// that is not would have made it report a rate no file has, and nothing
/// downstream would have noticed — the failure `Codebook::block_len`
/// exists to make unwriteable.
#[test]
fn the_block_length_follows_the_codebook_and_the_rate_follows_it() {
    assert_eq!(parse_codebook("leech1c12").unwrap().block_len(), 24);
    assert_eq!(parse_codebook("identity").unwrap().block_len(), 24);
    assert_eq!(parse_codebook("int3g24").unwrap().block_len(), 24);
    assert_eq!(parse_codebook("int3g128").unwrap().block_len(), 128);

    // `bits + (16 + bits)/group`: an f16 scale and a zero point packed
    // at `bits` width, which is what a deployed INT-k file stores.
    //
    // 🕳️ This read `bits + 32/group` until the upstream source was
    // fetched — charging the zero point a second f16 it does not cost.
    // That overstated `int3g24` by 0.54 b/weight, on the arm whose whole
    // job is to be the field's own quantizer.
    for (spec, bits, group) in [
        ("int2g24", 2.0, 24.0),
        ("int3g24", 3.0, 24.0),
        ("int3g128", 3.0, 128.0),
        ("int4g128", 4.0, 128.0),
    ] {
        let c = parse_codebook(spec).unwrap();
        let want = group * bits + 16.0 + bits;
        assert!(
            (c.block_bits() - want).abs() < 1e-9,
            "{spec}: block_bits {} != {want}",
            c.block_bits()
        );
        let per_weight = c.block_bits() / c.block_len() as f64;
        assert!(
            (per_weight - (bits + (16.0 + bits) / group)).abs() < 1e-9,
            "{spec}: {per_weight} b/weight"
        );
    }
}

/// An unknown `LLVQ_CALIB` used to fall through to wikitext-2 train while
/// the log printed the string that was asked for — the codebook defect,
/// on the other input of the same run.
#[test]
fn an_unknown_calibration_corpus_is_refused() {
    assert_eq!(
        CalibCorpus::parse(None).unwrap(),
        CalibCorpus::Wikitext2Train
    );
    assert_eq!(
        CalibCorpus::parse(Some("")).unwrap(),
        CalibCorpus::Wikitext2Train
    );
    assert_eq!(CalibCorpus::parse(Some("c4")).unwrap(), CalibCorpus::C4);
    assert_eq!(
        CalibCorpus::parse(Some("dclm-edu")).unwrap(),
        CalibCorpus::DclmEdu
    );
    assert_eq!(
        CalibCorpus::parse(Some("wikitext2-test")).unwrap(),
        CalibCorpus::Wikitext2Test
    );
    for bad in ["c44", "wikitext", "wikitext2-train", "dclm", "DCLM-edu"] {
        assert!(
            CalibCorpus::parse(Some(bad)).is_err(),
            "{bad:?} was accepted"
        );
    }
}

/// The refusal must name every accepted value, the new one included: the
/// message is the only documentation the operator reads at 2 a.m.
#[test]
fn the_refusal_names_every_accepted_value() {
    let msg = CalibCorpus::parse(Some("dclmedu")).unwrap_err();
    for c in [
        CalibCorpus::Wikitext2Train,
        CalibCorpus::C4,
        CalibCorpus::DclmEdu,
        CalibCorpus::Wikitext2Test,
    ] {
        assert!(msg.contains(c.name()), "{msg:?} does not name {}", c.name());
    }
}

/// The budget follows the request. A literal here is what served ~13× the
/// asked volume in silence before lot B.
#[test]
fn the_calibration_budget_follows_the_request() {
    assert_eq!(calib_chars(64, 2048), 64 * 2048 * 6);
    assert!(calib_chars(2048, 2048) > calib_chars(64, 2048));
    assert_eq!(calib_chars(usize::MAX, 2), usize::MAX);
}

/// Names printed in the journal have to parse back, or a log cannot be
/// replayed from what it says — the property `eval::printed_names_parse_back`
/// already fixes for the dtype.
#[test]
fn printed_names_parse_back() {
    for m in [Mode::NoGs, Mode::Gs, Mode::DesignC, Mode::Spherical] {
        assert_eq!(Mode::parse(Some(m.name())).unwrap(), m);
    }
    for c in [
        CalibCorpus::Wikitext2Train,
        CalibCorpus::C4,
        CalibCorpus::DclmEdu,
        CalibCorpus::Wikitext2Test,
    ] {
        assert_eq!(CalibCorpus::parse(Some(c.name())).unwrap(), c);
    }
}

/// No seed must reproduce the historical prefix exactly. Every published
/// number was measured on it, so a change here reinterprets them.
#[test]
fn no_seed_is_the_contiguous_prefix() {
    let s = window_starts(N, TOKENS, LEN, None);
    assert_eq!(s, (0..N).map(|w| w * LEN).collect::<Vec<_>>());
}

/// Every window must fit. An off-by-one in the span would slice past the
/// end of the token vector and panic three hours into a run.
#[test]
fn every_window_fits_in_the_corpus() {
    // Deliberately tight: the last legal start is exactly `ntokens - len`.
    for tokens in [TOKENS, N * LEN, N * LEN + 1] {
        for &s in &window_starts(N, tokens, LEN, Some(7)) {
            assert!(s + LEN <= tokens, "window at {s} overruns {tokens}");
        }
    }
}

/// A seed has to be reproducible, and two seeds have to disagree —
/// otherwise the three runs that are supposed to produce an error bar
/// would produce the same number three times and report a spread of zero.
#[test]
fn seeds_are_reproducible_and_distinct() {
    assert_eq!(
        window_starts(N, TOKENS, LEN, Some(1)),
        window_starts(N, TOKENS, LEN, Some(1))
    );
    assert_ne!(
        window_starts(N, TOKENS, LEN, Some(1)),
        window_starts(N, TOKENS, LEN, Some(2))
    );
}

/// The point of seeding is to leave the head of the corpus, not to shuffle
/// within it. Drawing from `0..n·len` would satisfy every test above and
/// sample exactly the same tokens as the prefix.
#[test]
fn seeded_windows_leave_the_prefix() {
    let s = window_starts(N, TOKENS, LEN, Some(3));
    assert!(
        s.iter().any(|&x| x > N * LEN),
        "every window landed inside the prefix: {s:?}"
    );
}

/// Repeated windows would weight their tokens twice in the Hessian.
#[test]
fn windows_are_distinct() {
    let s = window_starts(64, TOKENS, LEN, Some(5));
    let uniq: std::collections::HashSet<_> = s.iter().copied().collect();
    assert_eq!(uniq.len(), s.len());
}
