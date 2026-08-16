//! **The CUDA E1v decoder, transcribed into Rust and run against the format's
//! own reader.**
//!
//! `llvq-cuda/kernels/llvq_e1v.cuh` cannot be executed on the development
//! machine — no nvcc, no NVRTC, no card — and `bin/cuhcheck` only establishes
//! that it *parses*. This file is the other half: every line of arithmetic that
//! file performs, rewritten here, decoding the same bytes, checked against
//! [`E1vBlocks::decode_block`] slot by slot.
//!
//! It is the pattern `e1c_cuda_mirror.rs` established, and it catches the class
//! of defect that costs a rented run: a wrong word map, a wrong shift, a lane
//! reading its neighbour's header, a prefix sum off by one field.
//!
//! ## What it caught while being written, which is the argument for writing it
//!
//! The kernel's `e1v_peek` is transcribed from the sealed Metal arm's `peek`,
//! whose own comment says *"`off` is always at least 10 — the header precedes
//! every field — so the `64 - off` shift below is always defined"*. **That
//! guarantee is false for E1v**, whose header lives in the group prefix rather
//! than in the record: its first payload field sits at offset zero, and
//! `hi << (64 - 0)` is a shift by 64 — undefined in C++, answered by NVIDIA
//! hardware as a shift by nothing. The Golay index of nearly every block would
//! have come back with garbage in its low bits.
//!
//! Carrying a body across a language boundary carries its guards; it does not
//! carry the *reasons* the guards were enough. That is the whole case for a
//! mirror over a transcription.
//!
//! ## What it cannot say
//!
//! Nothing about registers, spills, occupancy, launch configuration or races,
//! and nothing about `__shfl_up_sync` beyond the value a correct scan must
//! produce — the mirror computes the prefix sum serially, because what is under
//! test is the **addressing it feeds**, not the shuffle. A warp scan that
//! disagreed with a serial one on a real card is a hardware-level failure this
//! file is structurally unable to see, and `preflight` is where that surfaces.

// 🚨 Two lints are silenced, and for one reason: **this file is a
// transcription**. `needless_range_loop` would have it iterate where the kernel
// indexes, and `too_many_arguments` would have it bundle what the kernel passes
// flat. Either rewrite makes the mirror stop looking like the thing it mirrors,
// and looking like it is the only property that makes it worth having — a
// reader must be able to put the two side by side and see one text.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use llvq_artifact::blockrec::{binom_table, block_records, e1v_payload_bits, GpuBlockRec, BINOM_STRIDE};
use llvq_artifact::e1v::{transcode_e1v_rows, E1vBlocks, E1V_GROUP, E1V_ORIGIN_ID};
use llvq_core::{Golay, DIM};
use llvq_search::cns::{cns_layout, CnsLayout};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};

/// `LLVQ_E1V_HDR_BITS`.
const HDR_BITS: u64 = 10;

// ---------------------------------------------------------------------------
// The kernel's readers, transcribed
// ---------------------------------------------------------------------------

/// `e1v_read_small` — two words, LSB-first.
fn read_small(w: &[u32], bit: u64, width: u32) -> u32 {
    let i = (bit >> 5) as usize;
    let sh = (bit & 31) as u32;
    let two = u64::from(w[i]) | (u64::from(w[i + 1]) << 32);
    ((two >> sh) & ((1u64 << width) - 1)) as u32
}

/// `e1v_window` — three words, shifted so `bit` becomes bit 0 of `(lo, hi)`.
fn window(w: &[u32], bit: u64) -> (u64, u32) {
    let i = (bit >> 5) as usize;
    let sh = (bit & 31) as u32;
    let mut lo = u64::from(w[i]) | (u64::from(w[i + 1]) << 32);
    let mut hi = w[i + 2];
    if sh != 0 {
        lo = (lo >> sh) | (u64::from(hi) << (64 - sh));
        hi >>= sh;
    }
    (lo, hi)
}

/// `e1v_peek` — including the `off == 0` branch the Metal original does not
/// need and this one does.
fn peek(lo: u64, hi: u32, off: u32, width: u32) -> u32 {
    if width == 0 {
        return 0;
    }
    let v = if off == 0 {
        lo
    } else if off >= 64 {
        u64::from(hi) >> (off - 64)
    } else {
        (lo >> off) | (u64::from(hi) << (64 - off))
    };
    (v & ((1u64 << width) - 1)) as u32
}

/// `e1v_walk_kind` — colex unranking over the free positions, in slot space.
fn walk_kind(mut rank: u32, mut cnt: u32, freem: u32, binom: &[u32]) -> u32 {
    let mut row = (cnt as usize) * BINOM_STRIDE;
    let mut taken = 0u32;
    let mut p = freem.count_ones();
    let mut m = freem;
    while m != 0 {
        let s = 31 - m.leading_zeros();
        m ^= 1 << s;
        p -= 1;
        let b = binom[row + p as usize];
        let hit = rank >= b && cnt != 0;
        if hit {
            rank -= b;
            row -= BINOM_STRIDE;
            cnt -= 1;
            taken |= 1 << s;
        }
    }
    taken
}

// ---------------------------------------------------------------------------
// `e1v_dot`, emitting the per-slot values rather than their sum
// ---------------------------------------------------------------------------
//
// A dot product is a sum, and a sum is not injective: two different
// arrangements can share one. The kernel returns a float; the mirror returns
// the 24 values that float is built from, which is the stronger statement and
// costs nothing here.

/// The whole addressing, for one group: every lane's header, the exclusive
/// prefix sum over the group's payload widths, and each lane's payload start.
///
/// The scan is serial on purpose — see the module header.
fn group_offsets(words: &[u32], gbit: u64, len: usize, pay: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::with_capacity(len);
    let mut acc = 0u32;
    for lane in 0..len {
        let hdr = read_small(words, gbit + HDR_BITS * lane as u64, HDR_BITS as u32);
        let id = hdr & 0x1ff;
        let gain = hdr >> 9;
        out.push((id, gain, acc));
        acc += pay[id as usize];
    }
    out
}

/// One block's 24 slot values, exactly as `e1v_dot` would multiply them.
fn slots(
    words: &[u32],
    gbit: u64,
    len: usize,
    lane: usize,
    off: u32,
    id: u32,
    tab: &[GpuBlockRec],
    binom: &[u32],
    golay: &Golay,
) -> [f32; DIM] {
    let _ = lane;
    let (lo, hi) = window(words, gbit + HDR_BITS * len as u64 + u64::from(off));
    let r = &tab[if u64::from(id) == E1V_ORIGIN_ID { 0 } else { id as usize + 1 }];
    let w = u32::from(r.w);
    let odd = (r.geom >> 1) & 1 != 0;
    let p_req = u32::from(r.geom & 1);

    let mut pre = u32::from(r.wb_golay);
    for k in 0..MAX_KINDS {
        pre += u32::from(r.wb_on[k]) + u32::from(r.wb_off[k]);
    }
    let r_sw = peek(lo, hi, pre, u32::from(r.wb_sw));
    let r_sf = peek(lo, hi, pre + u32::from(r.wb_sw), u32::from(r.wb_sf));
    let par = r_sw.count_ones() & 1;

    let mut cur = 0u32;
    let gi = peek(lo, hi, cur, u32::from(r.wb_golay));
    cur += u32::from(r.wb_golay);
    let cw = golay.codewords()[r.golay_base as usize + gi as usize] & 0xff_ffff;

    let mut out = [0.0f32; DIM];

    // ---- the support -----------------------------------------------------
    let mut freem = if odd { 0xff_ffff } else { cw };
    for k in 0..MAX_KINDS {
        if k >= r.k_on as usize {
            break;
        }
        let taken;
        if k + 1 == r.k_on as usize {
            taken = freem;
            freem = 0;
        } else {
            let rank = peek(lo, hi, cur, u32::from(r.wb_on[k]));
            cur += u32::from(r.wb_on[k]);
            let c = u32::from(r.cnt_on[k]);
            if c == 0 {
                continue;
            }
            taken = walk_kind(rank, c, freem, binom);
            freem ^= taken;
        }
        let v = r.vals_on[k];
        let mag = v.abs();
        let flag = v.to_bits() >> 31;
        let mut m = taken;
        while m != 0 {
            let s = m.trailing_zeros();
            m &= m - 1;
            let neg = if odd {
                ((cw >> s) & 1) ^ flag
            } else {
                let s_on = (cw & ((1 << s) - 1)).count_ones();
                if s_on + 1 == w {
                    par ^ p_req
                } else {
                    (r_sw >> s_on) & 1
                }
            };
            out[s as usize] = f32::from_bits(mag.to_bits() ^ (neg << 31));
        }
    }

    // ---- the off-support, even coset only --------------------------------
    if !odd {
        let mut tk = [0u32; MAX_KINDS];
        let mut nz = 0u32;
        let mut freem = (!cw) & 0xff_ffff;
        for k in 0..MAX_KINDS {
            if k >= r.k_off as usize {
                break;
            }
            let taken;
            if k + 1 == r.k_off as usize {
                taken = freem;
                freem = 0;
            } else {
                let rank = peek(lo, hi, cur, u32::from(r.wb_off[k]));
                cur += u32::from(r.wb_off[k]);
                let c = u32::from(r.cnt_off[k]);
                if c == 0 {
                    continue;
                }
                taken = walk_kind(rank, c, freem, binom);
                freem ^= taken;
            }
            tk[k] = taken;
            if r.vals_off[k] != 0.0 {
                nz |= taken;
            }
        }
        for k in 0..MAX_KINDS {
            if k >= r.k_off as usize {
                break;
            }
            let v = r.vals_off[k];
            if v == 0.0 {
                continue;
            }
            let mut m = tk[k];
            while m != 0 {
                let s = m.trailing_zeros();
                m &= m - 1;
                let fbit = (nz & ((1 << s) - 1)).count_ones();
                let neg = (r_sf >> fbit) & 1;
                out[s as usize] = f32::from_bits(v.to_bits() ^ (neg << 31));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

fn layouts(fd: &FastDecoder, golay: &Golay) -> Vec<CnsLayout> {
    (0..fd.n_classes()).map(|ci| cns_layout(fd, golay, ci)).collect()
}

/// The stream as the host uploads it: words, plus two of padding so the
/// three-word window of the last lane cannot run off the end.
fn as_words(s: &E1vBlocks) -> Vec<u32> {
    let mut w: Vec<u32> = s
        .data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    w.push(0);
    w.push(0);
    w
}

/// Every class at both ends and in the middle, plus the origin.
fn pool(fd: &FastDecoder) -> Vec<u64> {
    let mut v = vec![0u64];
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
        v.push(first + (last - first) / 2);
    }
    v
}

/// 🚨 **The mirror.** The CUDA arithmetic, on the row-aligned stream, against
/// the format's own reader — slot by slot, in exact f32 equality.
///
/// Exact and not approximate: the mirror deposits `±vals[kind]`, and the
/// reference's integer coordinate divided by the class norm is the *same* f32
/// rounding of the same rational. Anything else is a decode divergence, not a
/// floating-point one. (A sign on a zero value gives `-0.0`, which compares
/// equal to `0.0` — the kernel's own comment says so, and IEEE agrees.)
///
/// Five shapes, including a row narrower than a group and one that divides
/// exactly: the partial group is where this cut differs from the other, and a
/// mirror that only ever saw full groups would prove the wrong thing.
#[test]
#[cfg_attr(debug_assertions, ignore = "builds the whole class table, run in release")]
fn the_cuda_mirror_decodes_what_the_format_reads() {
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays = layouts(&fd, &golay);
    let tab = block_records(&fd, &golay);
    let pay = e1v_payload_bits(&fd, &golay, &tab);
    let binom = binom_table();
    let pool = pool(&fd);

    let mut checked = 0usize;
    let mut origins = 0usize;
    let mut seen: std::collections::BTreeSet<Option<usize>> = std::collections::BTreeSet::new();
    for row_blocks in [106usize, 170, 405, 7, 64] {
        let rows = 4usize;
        let n = rows * row_blocks;
        // Walking the pool in order rather than by a stride: the third shape
        // alone is wider than the pool, so the union of the five covers every
        // class — and the coverage is ASSERTED below rather than hoped for.
        let idxs: Vec<u64> = (0..n).map(|i| pool[i % pool.len()]).collect();
        let gains: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let s = transcode_e1v_rows(&fd, &golay, &idxs, &gains, row_blocks).expect("transcodes");
        let words = as_words(&s);
        let per_row = row_blocks.div_ceil(E1V_GROUP);

        for row in 0..rows {
            for gl in 0..per_row {
                let g = row * per_row + gl;
                let len = E1V_GROUP.min(row_blocks - gl * E1V_GROUP);
                let gbit = u64::from(s.bases[g]) * 32;
                let heads = group_offsets(&words, gbit, len, &pay);

                for (lane, &(id, gain, off)) in heads.iter().enumerate() {
                    let b = row * row_blocks + gl * E1V_GROUP + lane;

                    // The addressing, against the format's own derivation.
                    let (rg, rl, rlen) = s.locate(b);
                    assert_eq!((rg, rl, rlen), (g, lane, len), "bloc {b} : adressage");

                    // The reference: the format's reader, and the archive's own
                    // decoder behind it.
                    let (point, ref_gain) = s.decode_block(&fd, &golay, &lays, b);
                    assert_eq!(gain, u32::from(ref_gain), "bloc {b} : gain");
                    assert_eq!(
                        id,
                        fd.class_of(idxs[b]).map_or(E1V_ORIGIN_ID as u32, |ci| ci as u32),
                        "bloc {b} : classe"
                    );

                    seen.insert(fd.class_of(idxs[b]));
                    let got = slots(&words, gbit, len, lane, off, id, &tab, &binom, &golay);
                    match fd.class_of(idxs[b]) {
                        None => {
                            origins += 1;
                            assert_eq!(got, [0.0f32; DIM], "bloc {b} : l'origine n'est pas nulle");
                        }
                        Some(ci) => {
                            let norm = ((16 * fd.levels(ci).shell) as f64).sqrt();
                            for (sl, (&g_v, &p_v)) in got.iter().zip(&point).enumerate() {
                                let want = (f64::from(p_v) / norm) as f32;
                                assert_eq!(
                                    g_v, want,
                                    "bloc {b} (forme {row_blocks}, classe {ci}), créneau {sl}"
                                );
                            }
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    // The §5 discipline: a fixture that quietly missed a class prints exactly
    // the same green line as one that did not.
    assert!(
        seen.contains(&None),
        "aucun bloc origine dans le miroir — l'id 511 est le cas où la charge utile est vide, \
         l'entrée de table est 0 et le scan ne reçoit rien"
    );
    for ci in 0..fd.n_classes() {
        assert!(seen.contains(&Some(ci)), "classe {ci} absente du miroir");
    }
    eprintln!(
        "miroir CUDA E1v — {checked} blocs sur 5 formes, dont {origins} origines,\n  \
         les {} classes toutes couvertes, créneau par créneau en égalité f32 EXACTE\n  \
         contre E1vBlocks::decode_block.",
        fd.n_classes()
    );
}
