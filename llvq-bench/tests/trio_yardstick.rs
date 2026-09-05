//! The bench is the yardstick: `llvq_search::trio` must decode every word to
//! the point `llvq_bench::f1::rank::decode_word` decodes it to.
//!
//! The bench module was written first, measured (−0.6 pp of retention
//! against exact F1 on the same blocks, 2026-09-05) and reproduced on the
//! card by `llvq-cuda/tests/f1rank_matches_rust.rs`. The production module
//! re-derives the trellis, the table and the word from `llvq_core::Golay`
//! with its own construction — echelon reduction where the bench enumerates
//! cosets, a cost cap where it doubles, a listing rule where it has four
//! formulas — so the two share no code, and agreement here is evidence.
//! `llvq-bench` depends on `llvq-search`, never the reverse: this is the one
//! place both are visible.

use llvq_bench::f1::rank::{decode_word, RankTable, LINEAR_COLUMNS as BENCH_COLUMNS};
use llvq_bench::f1::{point_to_natural, Trellis, BRANCHES, GOLAY_STATES, TRIO as BENCH_TRIO};
use llvq_core::SplitMix64;
use llvq_search::trio::{Trio, LINEAR_COLUMNS, TRIO, WORD_BITS};

/// (1) A hundred thousand random words, in trio order and in natural order.
#[test]
fn the_two_decoders_agree_on_a_hundred_thousand_words() {
    let trio = Trio::new();
    let (table, trellis) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x7210_2026_0905_0001);
    for _ in 0..100_000 {
        let w = rng.next() & ((1u64 << WORD_BITS) - 1);
        let bench = decode_word(w, &table, &trellis);
        assert_eq!(trio.decode_trio_order(w), bench, "{w:#014x} in trio order");
        assert_eq!(trio.decode(w), point_to_natural(&bench, &trellis.code.order), "{w:#014x} in natural order");
    }
}

/// The objects behind the decoders are the same bytes: the trio, the order,
/// the 4,096 rows, the three trellis tables, the twelve columns.
#[test]
fn the_tables_are_the_same_bytes() {
    let trio = Trio::new();
    let (table, trellis) = (RankTable::build(), Trellis::new());
    assert_eq!(TRIO, BENCH_TRIO);
    assert_eq!(*trio.order(), trellis.code.order);
    assert_eq!(&trio.rows()[..], &table.rows[..]);
    assert_eq!(LINEAR_COLUMNS, BENCH_COLUMNS);
    for s in 0..GOLAY_STATES {
        assert_eq!(trio.prefixes()[s], trellis.prefixes[s], "prefixes of state {s}");
        assert_eq!(trio.suffixes()[s], trellis.suffixes[s], "suffixes of state {s}");
        for b in 0..BRANCHES {
            assert_eq!(trio.branches()[s][b], trellis.branches[s][b], "branch {b} of state {s}");
        }
    }
}
