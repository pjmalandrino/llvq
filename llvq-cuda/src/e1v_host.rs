// Host-side E1v: one matrix's row-aligned stream, and the tables the kernel
// reads beside it.
//
// ⚠️ This file is `include!`d — by `bin/planesbench.rs` and by
// `tests/e1v_host_matches_format.rs` — rather than being a module of `lib.rs`,
// following `planes14_host.rs`: `llvq-artifact` is a **dev-dependency** of this
// crate, so a real module could not name it. It therefore declares no `use`
// items; every path is fully qualified, so it can land in any module without
// colliding with the host's imports.
//
// ## What it does, and the one thing it must not get wrong
//
// The stream is produced by `llvq_artifact::e1v::transcode_e1v_rows` and by
// nothing else: this file invents no field, no width and no order. What it adds
// is the shape the device wants — words instead of bytes, two words of landing
// pad for the three-word window, and the byte count in the bench's accounting.
//
// 🚨 **`row_blocks` and the kernel's `nblocks` are the same number.** The stream
// is *cut* on it (a group never straddles a row) and the kernel *derives* every
// group boundary from it. Two values would move every boundary and decode
// garbage from the second row on, so they come from one place — `d_in / 24` —
// and [`E1vMat::row_blocks`] carries it to the launch rather than letting the
// caller recompute it.

/// One matrix's E1v arm: the stream, the group bases, and what it costs.
pub struct E1vMat {
    /// The stream as `u32`, **plus two words of landing pad** so the kernel's
    /// three-word window cannot run off the end of the last group.
    pub data: Vec<u32>,
    /// Word offset of each group's start.
    pub bases: Vec<u32>,
    /// Blocks per row — the kernel's `nblocks`, and the cut's own parameter.
    pub row_blocks: usize,
    /// Bytes the arm reads per matrix, in the bench's accounting: the stream
    /// and its base table. **The landing pad is not counted** — the repo's
    /// convention for padding that exists so a windowed read cannot fault
    /// (`thesis.rs:731`).
    pub bytes: u64,
}

/// Build one matrix's arm from its indices and gains, row-major.
pub fn e1v_mat(
    fd: &llvq_search::fastdec::FastDecoder,
    golay: &llvq_core::Golay,
    indices: &[u64],
    gains: &[u32],
    d_in: usize,
) -> Result<E1vMat, String> {
    let row_blocks = d_in / llvq_core::DIM;
    let s = llvq_artifact::e1v::transcode_e1v_rows(fd, golay, indices, gains, row_blocks)
        .map_err(|e| format!("transcode E1v: {e}"))?;
    let stream_bytes = s.data.len() as u64 + 4 * s.bases.len() as u64;
    let mut data: Vec<u32> = s
        .data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    data.push(0);
    data.push(0);
    Ok(E1vMat {
        data,
        bases: s.bases,
        row_blocks,
        bytes: stream_bytes,
    })
}

/// The per-run tables: the block records as words, the payload widths, and the
/// transposed binomials.
///
/// The records are handed to the device as `u32` because that is what the
/// upload path takes; the kernel reads them back as `E1vRec`, whose layout is
/// asserted at build time on the Rust side and mirrored field for field in
/// `kernels/llvq_e1v.cuh`.
pub fn e1v_tables(
    fd: &llvq_search::fastdec::FastDecoder,
    golay: &llvq_core::Golay,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let recs = llvq_artifact::blockrec::block_records(fd, golay);
    let pay = llvq_artifact::blockrec::e1v_payload_bits(fd, golay, &recs);
    let binom = llvq_artifact::blockrec::binom_table();

    let n_words = std::mem::size_of_val(&recs[..]) / 4;
    // SAFETY: `GpuBlockRec` is `#[repr(C)]`, `Copy`, 108 bytes with alignment 4
    // (both asserted at build time), so the slice is exactly `n_words` aligned
    // `u32`. This is the reinterpretation the upload performs; doing it field by
    // field would be a second layout to keep in step with the kernel's.
    let words = unsafe { std::slice::from_raw_parts(recs.as_ptr() as *const u32, n_words) };
    (words.to_vec(), pay, binom)
}
