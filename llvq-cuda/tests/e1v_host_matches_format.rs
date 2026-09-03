//! `src/e1v_host.rs` builds what the format wrote, and counts what it built.
//!
//! The host file is `include!`d by `bin/planesbench.rs`, which is entirely
//! under `cfg(target_os = "linux")` and therefore not compiled on the
//! development machine at all. Without this test nothing here would type-check
//! it, let alone run it — and it is the piece that decides what bytes reach the
//! card.
//!
//! Three statements:
//!
//! 1. **The words are the format's bytes**, little-endian, and the landing pad
//!    is exactly two words of zero past them.
//! 2. **The byte count excludes the pad** and includes the base table — the
//!    bench's accounting, not a convenient one.
//! 3. **`row_blocks` is `d_in / 24`**, the same number the kernel receives as
//!    `nblocks`. Two values would move every group boundary.

include!("../src/e1v_host.rs");

/// Every class at both ends and in the middle, plus the origin.
fn pool(fd: &llvq_search::fastdec::FastDecoder) -> Vec<u64> {
    let mut v = vec![0u64];
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
        v.push(first + (last - first) / 2);
    }
    v
}

#[test]
#[cfg_attr(debug_assertions, ignore = "builds the whole class table, run in release")]
fn the_host_builds_the_stream_the_format_wrote() {
    let fd = llvq_search::fastdec::FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let p = pool(&fd);

    for (d_in, rows) in [(2544usize, 4usize), (4080, 4), (9720, 3), (168, 8)] {
        let row_blocks = d_in / llvq_core::DIM;
        let n = rows * row_blocks;
        let idxs: Vec<u64> = (0..n).map(|i| p[i % p.len()]).collect();
        let gains: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();

        let m = e1v_mat(&fd, &golay, &idxs, &gains, d_in).expect("the host builds");
        let s = llvq_artifact::e1v::transcode_e1v_rows(&fd, &golay, &idxs, &gains, row_blocks)
            .expect("the format writes");

        // 3 — one number, not two.
        assert_eq!(m.row_blocks, row_blocks, "d_in {d_in}: row_blocks");
        assert_eq!(m.bases, s.bases, "d_in {d_in}: the base table");

        // 1 — the words ARE the bytes.
        assert_eq!(
            m.data.len(),
            s.data.len() / 4 + 2,
            "d_in {d_in}: two padding words, no more, no less"
        );
        for (w, c) in m.data.iter().zip(s.data.chunks_exact(4)) {
            assert_eq!(*w, u32::from_le_bytes([c[0], c[1], c[2], c[3]]), "d_in {d_in}");
        }
        assert_eq!(&m.data[m.data.len() - 2..], &[0, 0], "d_in {d_in}: the padding is zero");

        // 2 — the count is the stream and its bases, and the pad is not in it.
        assert_eq!(
            m.bytes,
            s.data.len() as u64 + 4 * s.bases.len() as u64,
            "d_in {d_in}: the byte accounting"
        );
        assert!(
            m.bytes < m.data.len() as u64 * 4 + 4 * m.bases.len() as u64,
            "d_in {d_in}: the padding would be counted"
        );
    }
}

/// The tables the kernel reads beside the stream, and the one identity that
/// makes the reinterpretation safe to write.
#[test]
#[cfg_attr(debug_assertions, ignore = "builds the whole class table, run in release")]
fn the_tables_are_the_records_reinterpreted() {
    let fd = llvq_search::fastdec::FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let (words, pay, binom) = e1v_tables(&fd, &golay);
    let recs = llvq_artifact::blockrec::block_records(&fd, &golay);

    assert_eq!(std::mem::size_of::<llvq_artifact::blockrec::GpuBlockRec>(), 108);
    assert_eq!(words.len(), recs.len() * 27, "108 bytes = 27 words per record");
    assert_eq!(pay.len(), llvq_artifact::blockrec::E1V_PAY_ENTRIES);
    assert_eq!(binom.len(), 25 * 25);

    // The first record is the origin, and the kernel reads its `k_on`/`geom`
    // out of the third-from-last word. Rather than decode the layout here —
    // which would be a second layout — check the property that matters: the
    // words round-trip to the same bytes the records occupy.
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    // SAFETY: same reinterpretation as `e1v_tables`, in the other direction,
    // over a slice whose size and alignment are asserted above.
    let raw = unsafe {
        std::slice::from_raw_parts(recs.as_ptr() as *const u8, std::mem::size_of_val(&recs[..]))
    };
    assert_eq!(bytes, raw, "the words are not the records");
}
