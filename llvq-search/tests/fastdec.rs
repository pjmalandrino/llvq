//! The fast v1 decoder must be indistinguishable from `Indexer::decode` —
//! same bits in, same point out, over the whole index space.
//!
//! The boundary sweep is the load-bearing part: `FastDecoder` rebuilds the
//! class layout independently (fixed arrays, u64 arithmetic), so the first
//! and last index of *every* class each pin one end of one class against the
//! reference. An off-by-one in any cardinality, offset or radix order fails
//! here before any random draw would find it.

use llvq_core::{Golay, Leech, SplitMix64, DIM};
use llvq_search::fastdec::{unrank_multiset, FastDecoder, MAX_KINDS, MAX_LEVELS};
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

    // 200,000 uniform draws over the ball.
    let mut rng = SplitMix64::new(0x6_F0FF);
    for _ in 0..200_000 {
        let idx = 1 + rng.next() % N13;
        let a = ix.decode(idx).expect("valid");
        let b = fd.decode(idx).expect("valid");
        assert_eq!(a, b, "index {idx} disagrees");
    }
}

/// Run the archive cascade from the **public** table alone, and require that it
/// reproduces `decode`.
///
/// This is what makes `cascade_class` + `unrank_multiset` load-bearing rather
/// than decorative. The walk below never calls `decode`: it reads
/// `CascadeClass`, drives `unrank_multiset`, and rebuilds the point from
/// scratch. If the exported record were missing anything `decode` reaches for
/// privately — a count, a radix, a parity — this fails, and it fails on the
/// class that needs the missing field rather than in some kernel three crates
/// away.
///
/// It is also the shape a GPU cascade arm has to take: one class table, one
/// unranking primitive, no access to the decoder.
fn cascade_from_public_table(fd: &FastDecoder, golay: &Golay, idx: u64) -> [i32; DIM] {
    let ci = fd.class_of(idx).expect("in range");
    let fc = fd.cascade_class(ci);
    let mut local = idx - 1 - fc.offset;
    let mut p = [0i32; DIM];

    if fc.odd {
        let r_arr = local % fc.m_arr;
        let gi = (local / fc.m_arr) as usize;
        let c = golay.codewords()[gi];
        let mut counts = fc.counts;
        let mut kinds = [0u8; DIM];
        unrank_multiset(r_arr, &mut counts, fc.n_kinds, fc.m_arr, &mut kinds);
        for (i, pi) in p.iter_mut().enumerate() {
            let v = fc.vals[kinds[i] as usize];
            let a = if v % 4 == 3 { v } else { -v };
            *pi = if c >> i & 1 == 1 { a } else { -a };
        }
        return p;
    }

    let r_sf = local % fc.s_f;
    local /= fc.s_f;
    let r_sw = local % fc.s_w;
    local /= fc.s_w;
    let r_off = local % fc.a_off;
    local /= fc.a_off;
    let r_on = local % fc.a_on;
    let gi = (local / fc.a_on) as usize;
    let c = golay.of_weight(fc.w as usize)[gi];
    let w = fc.w as usize;

    let mut wc = [0u8; MAX_KINDS];
    wc[..4].copy_from_slice(&fc.word_counts);
    let mut on_kinds = [0u8; DIM];
    unrank_multiset(r_on, &mut wc, fc.n_word, fc.a_on, &mut on_kinds[..w]);

    let mut oc = fc.off_counts;
    let mut off_kinds = [0u8; DIM];
    unrank_multiset(r_off, &mut oc, fc.n_off, fc.a_off, &mut off_kinds[..DIM - w]);

    let mut sign_bits = [0u32; DIM];
    let mut par = 0u32;
    for (j, sb) in sign_bits.iter_mut().enumerate().take(w.saturating_sub(1)) {
        let b = (r_sw >> j & 1) as u32;
        *sb = b;
        par ^= b;
    }
    if w > 0 {
        sign_bits[w - 1] = par ^ fc.p_req;
    }

    let zero_kind = (fc.n_off - 1) as u8;
    let (mut s_on, mut s_off, mut fbit) = (0usize, 0usize, 0u32);
    for (i, pi) in p.iter_mut().enumerate() {
        if c >> i & 1 == 1 {
            let v = fc.word_vals[on_kinds[s_on] as usize];
            *pi = if sign_bits[s_on] == 1 { -v } else { v };
            s_on += 1;
        } else {
            let k = off_kinds[s_off];
            if k != zero_kind {
                let v = fc.free_vals[k as usize];
                *pi = if r_sf >> fbit & 1 == 1 { -v } else { v };
                fbit += 1;
            }
            s_off += 1;
        }
    }
    p
}

#[test]
#[cfg_attr(debug_assertions, ignore = "boundary sweep plus draws, run in release")]
fn the_public_cascade_table_is_complete() {
    let fd = FastDecoder::new();
    let golay = Golay::new();

    // Both ends of every class: an incomplete record fails on the class that
    // needs the missing field.
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        for idx in [first, last] {
            assert_eq!(
                cascade_from_public_table(&fd, &golay, idx),
                fd.decode(idx).expect("valid"),
                "class {ci}, index {idx}: the public table does not reproduce decode"
            );
        }
    }

    // Interiors too — boundaries exercise the extreme ranks of each radix, the
    // draws exercise the mixed ones.
    let mut rng = SplitMix64::new(0x0_CA5CADE);
    for _ in 0..50_000 {
        let idx = 1 + rng.next() % N13;
        assert_eq!(
            cascade_from_public_table(&fd, &golay, idx),
            fd.decode(idx).expect("valid"),
            "index {idx}: the public table does not reproduce decode"
        );
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
