//! The fast v1 decoder must be indistinguishable from `Indexer::decode` —
//! same bits in, same point out, over the whole index space.
//!
//! The boundary sweep is the load-bearing part: `FastDecoder` rebuilds the
//! class layout independently (fixed arrays, u64 arithmetic), so the first
//! and last index of *every* class each pin one end of one class against the
//! reference. An off-by-one in any cardinality, offset or radix order fails
//! here before any random draw would find it.

use llvq_core::{Leech, SplitMix64};
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};
use llvq_search::index::{Indexer, N13};

#[test]
fn origin_and_range_edges() {
    let fd = FastDecoder::new();
    assert_eq!(fd.decode(0), Some([0; 24]));
    assert!(fd.decode(N13).is_some());
    assert_eq!(fd.decode(N13 + 1), None);
    assert_eq!(fd.class_of(0), None);
    assert_eq!(fd.class_of(N13 + 1), None);
    assert_eq!(fd.class_of(1), Some(0));
    assert_eq!(fd.class_of(N13), Some(fd.n_classes() - 1));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "exhaustive boundary sweep, run in release")]
fn fast_decoder_matches_indexer_exactly() {
    let ix = Indexer::new();
    let fd = FastDecoder::new();

    // Every class boundary: 383 × 2 points.
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        for idx in [first, last] {
            assert_eq!(fd.class_of(idx), Some(ci), "index {idx} lands in class {ci}");
            let a = ix.decode(idx).expect("valid");
            let b = fd.decode(idx).expect("valid");
            assert_eq!(a, b, "boundary index {idx} disagrees");
        }
    }

    // 200 000 uniform draws over the ball.
    let mut rng = SplitMix64::new(0x6_F0FF);
    for _ in 0..200_000 {
        let idx = 1 + rng.next() % N13;
        let a = ix.decode(idx).expect("valid");
        let b = fd.decode(idx).expect("valid");
        assert_eq!(a, b, "index {idx} disagrees");
    }
}

/// The class-level metadata is what the runtime format accounting stands on:
/// every decoded point of a class must realize exactly the levels the class
/// declares — same distinct |values|, same counts, same coset, same shell.
#[test]
#[cfg_attr(debug_assertions, ignore = "decodes two points per class, run in release")]
fn class_levels_match_decoded_points() {
    let fd = FastDecoder::new();
    for ci in 0..fd.n_classes() {
        let lv = *fd.levels(ci);
        assert!(lv.len <= MAX_LEVELS);
        assert!(
            (1..lv.len).all(|i| {
                let (c0, c1) = (lv.counts[i - 1], lv.counts[i]);
                c0 > c1 || (c0 == c1 && lv.values[i - 1] > lv.values[i])
            }),
            "class {ci}: levels must be in canonical order"
        );

        let (first, last) = fd.class_range(ci);
        for idx in [first, last] {
            let p = fd.decode(idx).expect("valid");
            assert_eq!(
                Leech::shell_index(&p),
                Some(lv.shell as u64),
                "class {ci}: shell mismatch"
            );
            assert_eq!(
                p.iter().all(|v| v % 2 != 0),
                lv.odd,
                "class {ci}: coset mismatch"
            );
            assert_eq!(
                p.iter().filter(|&&v| v != 0).count(),
                lv.nonzero as usize,
                "class {ci}: nonzero count mismatch"
            );
            // The multiset of |values| must match the declared levels.
            let mut counts = std::collections::BTreeMap::new();
            for &v in &p {
                *counts.entry(v.abs()).or_insert(0u8) += 1;
            }
            let mut declared = std::collections::BTreeMap::new();
            for i in 0..lv.len {
                declared.insert(lv.values[i], lv.counts[i]);
            }
            assert_eq!(counts, declared, "class {ci}: |value| multiset mismatch");
        }
    }
}
