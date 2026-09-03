//! The fused loader's host-side layout plumbing, pinned without a card.
//!
//! `llvq-artifact` already proves `transcode_planes14` a bit-exact bijection
//! of the Slot32 content (150.7 M blocks, `planes14_format.rs`). What that
//! sweep cannot see is what `llvq-llm` does on top: the packing of the byte
//! stream into the `u32` words the kernel indexes, the end-of-stream padding
//! that keeps the four-word window in bounds, the layout switch, and the
//! bases-free accounting of the planes path. Each of those is a way to ship
//! a correct stream and still crash — or lie — on the card, so each is
//! pinned here, on the machine that has no card.

use llvq_artifact::runtime::{transcode, ClassTable, Layout, PLANES14_BYTES};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::{FusedLayout, HostStream, Transcoder};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;

fn synthetic(n: usize, seed: u64) -> (Vec<u64>, Vec<u32>) {
    let mut rng = SplitMix64::new(seed);
    (
        (0..n).map(|_| 1 + rng.next() % N13).collect(),
        (0..n).map(|_| (rng.next() & 1) as u32).collect(),
    )
}

/// The kernel's four-word window, mirrored in Rust — same word indices, same
/// shifts, same field extraction as `planes_fields` in `llvq_planes.cuh`.
/// Indexing `words` directly is the point: a missing end pad is an
/// out-of-bounds panic here, exactly where it is an illegal address there.
fn planes_fields(words: &[u32], b: usize) -> (usize, u32, u32, [u32; 3]) {
    let byte = PLANES14_BYTES * b;
    let w = byte >> 2;
    let (w0, w1, w2, w3) = (
        words[w] as u64,
        words[w + 1] as u64,
        words[w + 2] as u64,
        words[w + 3] as u64,
    );
    let sh = ((byte & 3) * 8) as u32; // 0 or 16: 14·b mod 4 ∈ {0, 2}
    let lo = w1 << 32 | w0;
    let hi = w3 << 32 | w2;
    let hdr = ((lo >> sh) & 0x3ff) as u32;
    let fs = sh + 10;
    let pay_lo = (lo >> fs) | (hi << (64 - fs));
    let pay_hi = hi >> fs;
    let ext24 = |off: u32| -> u32 {
        if off < 64 {
            (((pay_lo >> off) | (pay_hi << (64 - off))) & 0xffffff) as u32
        } else {
            ((pay_hi >> (off - 64)) & 0xffffff) as u32
        }
    };
    let smask = (pay_lo & 0xffffff) as u32;
    (
        (hdr & 0x1ff) as usize,
        hdr >> 9,
        smask,
        [ext24(24), ext24(48), ext24(72)],
    )
}

/// Every block of the packed word stream, read through the kernel's own
/// window, decodes to exactly the point and gain the Slot32 stream carries.
///
/// Odd and even block counts on purpose: the two parities of `14·n mod 4`
/// are the two padding cases, and the odd one is where a truncated word
/// buffer loses the last block's tail.
#[test]
fn planes_words_decode_to_the_slot_content() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    for &(n, seed) in &[(33usize, 0xA1u64), (40, 0xB2)] {
        let (idx, gains) = synthetic(n, seed);
        let (stream, _) = tr.stream(&idx, &gains, 1, n).expect("transcode planes14");
        let words = match &stream {
            HostStream::Planes14 { words } => words,
            _ => panic!("planes14 layout asked, another stream returned"),
        };
        // The last block's window must sit inside the buffer — the invariant
        // the end-of-stream padding exists to provide.
        assert!(
            PLANES14_BYTES * (n - 1) / 4 + 4 <= words.len(),
            "n={n}: the 4-word window of the last block falls outside the buffer"
        );

        let slot = transcode(&fd, &table, &idx, &gains, Layout::Slot32).expect("transcode slot32");
        for b in 0..n {
            let (want_pt, want_gain) = slot.decode_block(&table, b);
            let (id, gain, smask, p) = planes_fields(words, b);
            let rec = table.record(id);
            let mut got = [0i32; DIM];
            if id != 0 {
                for (i, g) in got.iter_mut().enumerate() {
                    let lvl = ((p[0] >> i) & 1 | ((p[1] >> i) & 1) << 1 | ((p[2] >> i) & 1) << 2)
                        as usize;
                    assert!(lvl < rec.len as usize, "block {b}, slot {i}: level {lvl}");
                    let v = rec.values[lvl];
                    *g = if (smask >> i) & 1 == 1 { -v } else { v };
                }
            }
            assert_eq!(gain, want_gain, "block {b}: gain");
            assert_eq!(got, want_pt, "block {b}: point");
        }
    }
}

/// The switch resolves what it is told, defaults to Planes14, and refuses a
/// typo instead of silently running the wrong arm of an A/B.
#[test]
fn the_layout_switch_is_honoured() {
    assert_eq!(FusedLayout::parse(None).unwrap(), FusedLayout::Planes14);
    assert_eq!(FusedLayout::parse(Some("")).unwrap(), FusedLayout::Planes14);
    assert_eq!(
        FusedLayout::parse(Some("planes14")).unwrap(),
        FusedLayout::Planes14
    );
    assert_eq!(
        FusedLayout::parse(Some("planes12x")).unwrap(),
        FusedLayout::Planes12x
    );
    assert_eq!(
        FusedLayout::parse(Some("slot32")).unwrap(),
        FusedLayout::Slot32
    );
    assert_eq!(
        FusedLayout::parse(Some("golay70")).unwrap(),
        FusedLayout::Golay70
    );
    assert!(FusedLayout::parse(Some("Slot32")).is_err());
    assert!(FusedLayout::parse(Some("grouped32")).is_err());
    // Near-misses: two spellings of layouts that exist, and one bench-only
    // name. Refusing beats falling back — an A/B whose arm silently reverts
    // to the default reports the default's numbers under the other arm's
    // name. (Until 2026-08-11 `golay70` itself sat in this list; it is a
    // layout of the switch now — step 5 of the v2 campaign — and the near
    // misses keep guarding the same property.)
    assert!(FusedLayout::parse(Some("planes12")).is_err());
    assert!(FusedLayout::parse(Some("Planes12x")).is_err());
    assert!(FusedLayout::parse(Some("Golay70")).is_err());
    assert!(FusedLayout::parse(Some("golay70v2")).is_err());
    // And the refusal must *name* what it will accept, or the operator's next
    // move is a source dive.
    let msg = FusedLayout::parse(Some("planes12")).unwrap_err();
    for expected in ["planes14", "planes12x", "slot32", "golay70"] {
        assert!(msg.contains(expected), "the refusal does not name \"{expected}\": {msg}");
    }
}

/// The payload accounting follows the layout: Planes14 is exactly 14 bytes a
/// block and nothing else — no bases, by type — while Slot32 still counts its
/// stream plus a `u32` base per group, unchanged.
#[test]
fn payload_accounting_matches_the_layout() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let n = 50usize;
    let (idx, gains) = synthetic(n, 0xC3);

    let tr = Transcoder::new(FusedLayout::Planes14).unwrap();
    let (s, bytes) = tr.stream(&idx, &gains, 1, n).unwrap();
    match s {
        HostStream::Planes14 { .. } => {}
        _ => panic!("bases on the planes14 path"),
    }
    assert_eq!(bytes, (n * PLANES14_BYTES) as u64);

    let tr = Transcoder::new(FusedLayout::Slot32).unwrap();
    let (s, bytes) = tr.stream(&idx, &gains, 1, n).unwrap();
    let rt = transcode(&fd, &table, &idx, &gains, Layout::Slot32).unwrap();
    match s {
        HostStream::Slot32 { words, bases } => {
            assert_eq!(bases, rt.bases, "the bases must be the transcoder's");
            assert_eq!(bytes, rt.data.len() as u64 + rt.bases.len() as u64 * 4);
            // The five-word window's 20-byte pad, still there.
            assert!(words.len() * 4 >= rt.data.len() + 20);
        }
        _ => panic!("slot32 layout asked, another stream returned"),
    }
}

/// The row structure is part of the call, and a mismatch is refused.
///
/// `Planes12x` is the first layout whose kernel needs to know where an output
/// row ends; the shape check runs on every arm so that a caller that gets
/// `d_out` or `nblocks` wrong fails on the layout it happens to be testing,
/// not only on the one that reads them.
#[test]
fn a_mismatched_row_shape_is_refused() {
    let (idx, gains) = synthetic(48, 0xD4);
    for layout in [FusedLayout::Planes14, FusedLayout::Slot32] {
        let tr = Transcoder::new(layout).unwrap();
        assert!(tr.stream(&idx, &gains, 6, 8).is_ok(), "{layout:?}: 6 × 8 = 48");
        assert!(
            tr.stream(&idx, &gains, 6, 9).is_err(),
            "{layout:?}: 6 × 9 ≠ 48, and the stream must not build"
        );
    }
}
