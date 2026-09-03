//! **The E1c CUDA decoder's arithmetic, checked on a machine with no CUDA.**
//!
//! `llvq-cuda/kernels/llvq_e1c.cuh` is arm `e1c14` / `e1c12` of
//! `proofs/preregistration-p4-2026-08-14.md` §2.5. It cannot be compiled here:
//! the development machine is a Mac, `cargo check -p llvq-cuda` finishes in
//! 0.4 s because the whole crate is behind `cfg(target_os = "linux")`, and
//! neither `nvcc` nor NVRTC exists. Everything written for a card is therefore
//! written blind — the established workflow of this crate, which `llvq-cuda`'s
//! own module header explains.
//!
//! This file shrinks that blind spot by the half that can be shrunk.
//!
//! ## What it checks, and what it cannot
//!
//! ✅ **The word map and the bit arithmetic.** `e1c_fields`'s indexing —
//! the group base, the ten header words, the straddling header window, the
//! slot-major stride of `1 + planes`, the lane bit `(w >> l) & 1` — is
//! transcribed below into Rust and run against `E1c14Blocks::decode_block` /
//! `E1c12Blocks::decode_approx_block` on a fixture touching every class plus
//! the origin. **This is the class of defect that cost two runs on
//! 2026-08-15** (a shader that rebuilt an index from counts, and a parity that
//! accumulated on the wrong slots): both were arithmetic, both would have been
//! caught here.
//!
//! ❌ **Nothing else.** Not a syntax error, not a launch configuration, not a
//! register spill, not a race, not the `#pragma unroll` folding away the
//! `PLANES > 1` branches. Those surface at job start on a rented card, and
//! `preflight` exists for them.
//!
//! ⚠️ **And this is a TRANSCRIPTION, not a shared implementation.** The `.cuh`
//! and the Rust below are two texts that must say the same thing, and nothing
//! detects a drift between them mechanically. What the test does do is read
//! the `.cuh` and fail if the constants it mirrors have moved — a weak lock,
//! honestly weak, and better than the alternative of asserting nothing.

use llvq_artifact::e1c::{
    group_words, header_words, E1c12Blocks, E1c14Blocks, E1C12_PLANES, E1C14_PLANES, E1C_GROUP,
};
use llvq_artifact::runtime::{transcode_planes12x, transcode_planes14, ClassTable};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::Searcher;

/// The `.cuh`'s own source, so the constants below can be checked against it.
const CUH: &str = include_str!("../../llvq-cuda/kernels/llvq_e1c.cuh");

/// **`e1c_fields`, transcribed.** Group base, header window, slot-major
/// fields, lane bit — the four things the kernel does that nothing else in
/// this repository does.
///
/// Deliberately shaped like the CUDA, not like `untranspose`: same variable
/// names, same order, same `w + 1` read whether or not the window straddles.
/// A version rewritten idiomatically would be easier to read and would stop
/// being a transcription.
fn e1c_fields(words: &[u32], planes: usize, b: usize) -> (u32, u32, u32, [u32; 3]) {
    let (g, l) = (b / E1C_GROUP, b % E1C_GROUP);
    let grp = g * (header_words() + DIM * (1 + planes));

    let bit = 10 * l;
    let w = bit >> 5;
    let sh = bit & 31;
    // The kernel reads `grp[w + 1]` unconditionally. At l = 31 that is bit 310,
    // word 9, shift 22 — and 22 + 10 = 32 exactly, so the window never runs
    // past word 10, which is the first slot word and always in range.
    let win = u64::from(words[grp + w]) | (u64::from(words[grp + w + 1]) << 32);
    let hdr = ((win >> sh) & 0x3ff) as u32;

    let (mut smask, mut p) = (0u32, [0u32; 3]);
    let slots = grp + header_words();
    for j in 0..DIM {
        let fw = slots + j * (1 + planes);
        smask |= ((words[fw] >> l) & 1) << j;
        p[0] |= ((words[fw + 1] >> l) & 1) << j;
        if planes > 1 {
            p[1] |= ((words[fw + 2] >> l) & 1) << j;
        }
        if planes > 2 {
            p[2] |= ((words[fw + 3] >> l) & 1) << j;
        }
    }
    (hdr & 0x1ff, hdr >> 9, smask, p)
}

/// Bytes to words, little-endian — what `const u32*` gives the kernel.
fn as_words(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every class at both ends, the origin, and a bounded draw — the fixture
/// shape this repo uses everywhere for "touches the whole table".
fn fixture(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut v = vec![0u64];
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
        v.push(first + rng.next() % (last - first + 1));
    }
    while !v.len().is_multiple_of(E1C_GROUP) {
        v.push(0);
    }
    v
}

#[test]
fn the_cuda_word_map_rebuilds_every_field() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x0000_E1C0_CDA0);
    let idxs = fixture(&fd, &mut rng);
    let gains: Vec<u32> = (0..idxs.len()).map(|i| (i % 2) as u32).collect();

    let p14 = transcode_planes14(&fd, &table, &idxs, &gains).expect("transcodes");
    let e14 = E1c14Blocks::from_planes14(&p14);
    let p12 = transcode_planes12x(&fd, &table, &s, &idxs, &gains).expect("transcodes");
    let e12 = E1c12Blocks::from_planes12x(&p12);

    let w14 = as_words(&e14.data);
    let w12 = as_words(&e12.data);
    assert_eq!(w14.len(), idxs.len() / E1C_GROUP * group_words(E1C14_PLANES));
    assert_eq!(w12.len(), idxs.len() / E1C_GROUP * group_words(E1C12_PLANES));

    for (b, &gain_b) in gains.iter().enumerate() {
        // The kernel's fields must be the module's fields. `decode_block` is
        // the proven side — swept over the 150,681,600 blocks of the sealed 4B
        // on 2026-08-12 — so a disagreement here is the kernel's.
        let (id, gain, smask, p) = e1c_fields(&w14, E1C14_PLANES, b);
        let (want, wgain) = e14.decode_block(&table, b);
        assert_eq!(gain, gain_b, "block {b}: gain, E1c14");
        assert_eq!(wgain, gain_b, "block {b}: the module's gain");
        assert_eq!(point_of(&table, id, smask, &p), want, "block {b}: E1c14");

        let (id, gain, smask, p) = e1c_fields(&w12, E1C12_PLANES, b);
        assert_eq!(gain, gain_b, "block {b}: gain, E1c12");
        assert_eq!(p[2], 0, "block {b}: E1c12 has only two planes");
        assert_eq!(
            point_of(&table, id, smask, &p),
            e12.decode_approx_block(&table, b).0,
            "block {b}: E1c12"
        );
    }
}

/// The value-selection tree of `e1c_dot`, transcribed: level from the plane
/// bits, value from `ClassRec.vals`, sign from `smask`.
fn point_of(table: &ClassTable, id: u32, smask: u32, p: &[u32; 3]) -> [i32; DIM] {
    let mut out = [0i32; DIM];
    if id == 0 {
        return out;
    }
    let rec = table.record(id as usize);
    for (j, o) in out.iter_mut().enumerate() {
        let lvl = ((p[0] >> j) & 1) | (((p[1] >> j) & 1) << 1) | (((p[2] >> j) & 1) << 2);
        let v = rec.values[lvl as usize];
        *o = if (smask >> j) & 1 == 1 { -v } else { v };
    }
    out
}

/// The weak lock: the constants this file mirrors must still be the `.cuh`'s.
///
/// It cannot see a changed expression, only a changed constant — and it says
/// so. A shared implementation would be better; it is not available across a
/// crate that does not build on this machine.
#[test]
fn the_mirrored_constants_are_still_the_kernels() {
    for (name, expect) in [
        ("#define LLVQ_E1C_GROUP", "32u"),
        ("#define LLVQ_E1C_HDR_BITS", "10u"),
    ] {
        let line = CUH
            .lines()
            .find(|l| l.trim_start().starts_with(name))
            .unwrap_or_else(|| panic!("{name} has disappeared from the .cuh"));
        assert!(
            line.contains(expect),
            "{name} is no longer {expect} in the .cuh: \"{line}\" — this file's mirror \
             no longer describes the kernel"
        );
    }
    assert_eq!(E1C_GROUP, 32);
    assert_eq!(header_words(), 10);
    assert_eq!(E1C14_PLANES, 3);
    assert_eq!(E1C12_PLANES, 2);
}
