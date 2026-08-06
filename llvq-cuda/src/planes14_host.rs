// Host-side Planes14: the transcoder from a Slot32 stream, and the bit-exact
// CPU decoder the GPU kernel is checked against.
//
// ⚠️ This file is `include!`d — by `bin/planesbench.rs` and by
// `tests/planes_decoder_matches_rust.rs` — rather than being a module of
// `lib.rs`, so that adding the Planes14 layout touches **no existing file**
// (lot A owns `lib.rs` at the time of writing). It therefore declares no
// `use` items: every path is fully qualified, so it can land in any module
// without colliding with the host's imports.
//
// ## The single point to re-plug
//
// [`planes14_from_slot32`] is a stopgap: the artifact-side transcoder
// (`Layout::Planes14` in `llvq-artifact`) is being written in parallel
// against the same frozen spec. When it exists, callers replace this function
// with `transcode(fd, table, indices, gains, Layout::Planes14)` and delete
// this file; until then it is the reference the bench and the host test run
// on, and it is deliberately a **pure bit-level bijection** of the Slot32
// stream — no re-encoding, no lattice decode — so it cannot disagree with the
// content Slot32 carries.
//
// Record, from bit 0, LSB-first, 14 bytes per block at offset 14·b:
//
//     [class : 9][gain : 1][smask : 24][plane0 : 24][plane1 : 24][plane2 : 24][0 : 6]
//
// Level of slot j = plane0[j] | plane1[j]<<1 | plane2[j]<<2, indexing the
// class's canonical values; smask has Slot32's exact semantics (sign of slot
// j, zero slots written 0). A record is 112 bits = exactly 14 bytes, so every
// record is byte-aligned and a `u128` in little-endian order *is* the record.

/// Bytes per Planes14 record — 112 bits, 4.6667 b/weight of payload.
pub const PLANES14_STRIDE: usize = 14;

/// Read `nbits <= 24` starting at absolute bit `at`, LSB-first — the same
/// convention `runtime.rs` writes and the GPU reads.
fn p14_read_bits(data: &[u8], at: u64, nbits: u32) -> u32 {
    debug_assert!(nbits <= 24);
    let byte = (at >> 3) as usize;
    let sh = (at & 7) as u32;
    let mut v = 0u64;
    for (k, &b) in data[byte..].iter().take(8).enumerate() {
        v |= u64::from(b) << (8 * k as u32);
    }
    ((v >> sh) & ((1u64 << nbits) - 1)) as u32
}

/// One Planes14 record, read straight off the Slot32 bit stream.
///
/// The origin (`id == 0`) has **no** sign-mask field in Slot32 — its record
/// is the 10-bit header alone — so nothing past the header is read: smask and
/// the three planes are zero, which is the canonical origin record.
///
/// The mask-disjointness assert is not paranoia: Slot32's decoder resolves an
/// overlap as "last mask wins" while the plane encoding would OR the level
/// indices — the two would silently diverge. An overlapping stream is a
/// transcoder bug upstream and must die here, at the source.
fn planes14_record_from_slot32(
    rt: &llvq_artifact::runtime::RuntimeBlocks,
    table: &llvq_artifact::runtime::ClassTable,
    b: usize,
) -> u128 {
    let group = llvq_artifact::runtime::GROUP;
    let g = b / group;
    let stride = u64::from(rt.bases[g + 1] - rt.bases[g]) / group as u64;
    let bit0 = (u64::from(rt.bases[g]) + (b % group) as u64 * stride) * 8;

    let id = p14_read_bits(&rt.data, bit0, 9);
    let gain = p14_read_bits(&rt.data, bit0 + 9, 1);
    let mut rec = u128::from(id) | u128::from(gain) << 9;
    if id != 0 {
        let smask = p14_read_bits(&rt.data, bit0 + 10, 24);
        let len = u32::from(table.record(id as usize).len);
        let (mut p0, mut p1, mut p2) = (0u32, 0u32, 0u32);
        let mut union = 0u32;
        for k in 1..len {
            let mk = p14_read_bits(&rt.data, bit0 + 34 + 24 * u64::from(k - 1), 24);
            assert_eq!(union & mk, 0, "block {b}: Slot32 level masks must be disjoint");
            union |= mk;
            if k & 1 != 0 {
                p0 |= mk;
            }
            if k & 2 != 0 {
                p1 |= mk;
            }
            if k & 4 != 0 {
                p2 |= mk;
            }
        }
        rec |= u128::from(smask) << 10
            | u128::from(p0) << 34
            | u128::from(p1) << 58
            | u128::from(p2) << 82;
    }
    rec
}

/// Transcode a whole Slot32 stream into Planes14, block for block.
///
/// Returns exactly `14 * n_blocks` bytes — the accounted payload. The caller
/// pads for the GPU's four-word window (the last block's window reaches up to
/// 2 bytes past the stream) before packing into `u32` words.
pub fn planes14_from_slot32(
    rt: &llvq_artifact::runtime::RuntimeBlocks,
    table: &llvq_artifact::runtime::ClassTable,
) -> Vec<u8> {
    assert_eq!(
        rt.layout,
        llvq_artifact::runtime::Layout::Slot32,
        "Planes14 transcodes from Slot32 only"
    );
    // Same hard-coded width as every GPU decoder (`hdr >> 9`): a different
    // gain width would shift every field and decode garbage.
    assert_eq!(rt.gain_bits, 1, "Planes14 hard-codes one gain bit");
    let n = rt.n_blocks;
    let mut out = vec![0u8; n * PLANES14_STRIDE];
    let nthreads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let chunk = n.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, dst) in out.chunks_mut(chunk * PLANES14_STRIDE).enumerate() {
            sc.spawn(move || {
                for (l, rec) in dst.chunks_mut(PLANES14_STRIDE).enumerate() {
                    let r = planes14_record_from_slot32(rt, table, t * chunk + l);
                    rec.copy_from_slice(&r.to_le_bytes()[..PLANES14_STRIDE]);
                }
            });
        }
    });
    out
}

/// Decode Planes14 block `b` back to its lattice point and gain rank — the
/// reference the GPU kernel is checked against, mirroring
/// `RuntimeBlocks::decode_block` for the other layouts.
///
/// Lethal on purpose: a level index at or past the class's `len`, or a
/// nonzero bit among the record's six trailing zeros, is a corrupt stream and
/// panics rather than decoding something plausible.
pub fn planes14_decode_block(
    data: &[u8],
    table: &llvq_artifact::runtime::ClassTable,
    b: usize,
) -> (llvq_core::Point, u32) {
    let mut buf = [0u8; 16];
    buf[..PLANES14_STRIDE].copy_from_slice(&data[b * PLANES14_STRIDE..(b + 1) * PLANES14_STRIDE]);
    let r = u128::from_le_bytes(buf);
    assert_eq!(r >> 106, 0, "block {b}: bits 106..111 must be zero");
    let id = (r & 0x1ff) as usize;
    let gain = ((r >> 9) & 1) as u32;
    let rec = table.record(id);
    let smask = ((r >> 10) & 0xff_ffff) as u32;
    let p0 = ((r >> 34) & 0xff_ffff) as u32;
    let p1 = ((r >> 58) & 0xff_ffff) as u32;
    let p2 = ((r >> 82) & 0xff_ffff) as u32;

    let mut p = [0i32; llvq_core::DIM];
    for (j, pj) in p.iter_mut().enumerate() {
        let lev = (p0 >> j & 1) | (p1 >> j & 1) << 1 | (p2 >> j & 1) << 2;
        assert!(
            (lev as usize) < rec.len as usize,
            "block {b} slot {j}: level {lev} out of the class's {} levels",
            rec.len
        );
        let v = rec.values[lev as usize];
        *pj = if smask >> j & 1 == 1 { -v } else { v };
    }
    (p, gain)
}
