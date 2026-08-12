//! What each candidate **runtime** layout costs, in bits, on real blocks —
//! the measurement the passation said to take before writing any kernel.
//!
//! The archive rank is optimal in bits and unusable in a kernel (8.27 ns/block
//! on GPU, 106× the floor); a mask layout decodes at 0.11 ns/block but pays
//! for it in RAM. This bench prices that exchange **completely**: not the
//! arrangement alone (`arrbits` did that, information-theoretically), but
//! everything a runtime decoder actually reads per block — level-to-slot
//! assignment, signs, class id, gain bit, and the addressing overhead a
//! variable-width stream cannot avoid.
//!
//! ## The accounting, per block
//!
//! Everything below is a function of the block's **class** (a class is a
//! fixed multiset of |values|), so the sweep over a real artifact is exact
//! and needs no lattice decoding — a binary search per index, nothing more.
//!
//! * class id: 9 bits (383 classes + origin), gain: as many bits as the file
//!   says (1 on the published 4B). Both variants pay them.
//! * signs: one bit per **nonzero** coordinate. The archive gets signs partly
//!   for free (forced by the codeword on the odd coset); a runtime layout
//!   that drops the codeword must spell them out. Zero slots carry no sign —
//!   the decoder walks slots in order and indexes signs by a running nonzero
//!   count it maintains anyway.
//! * assignment, three candidate encodings:
//!   - **positionnel** — ⌈log₂ L⌉ bits per slot, L the block's level count.
//!     Decode is one shift+mask+table lookup per slot, no popcount chain.
//!   - **masques imbriqués** — level 0's mask over 24 slots, level 1's over
//!     what level 0 left free, … Compact, but decode chains popcounts.
//!   - **champ fixe 128** — 24 sign+3-bit-level nibbles + class + gain in a
//!     uint4. One aligned load, zero divergence, 5.33 b/w flat.
//!   - **Slot32** — the layout the fused kernel actually reads: one 24-bit
//!     mask per level *in slot space*, level 0 implicit, signs indexed by
//!     slot rather than by nonzero rank. Costs `9 + gain + 24·L` bits, so it
//!     spends `24 − nonzero` bits more than the flat layout to buy a fixed
//!     field offset. Nothing priced it here before; the two figures the
//!     dossier publishes for it come from two other benches.
//!   - **Planes14** — today's production layout ([`PlanesBlocks`]): the same
//!     decoded content as `Slot32`, re-arranged as *bit-planes*. The level of
//!     a slot stops being a one-hot over `L` masks and becomes an explicit
//!     3-bit index spread over three 24-bit planes, so the record no longer
//!     grows with `L`: `9 + 1 + 24 + 3·24 = 106` bits of payload in a frozen
//!     [`PLANES14_BYTES`]-byte stride, padding included. Flat, class-blind,
//!     **and with no base table at all** — the addressing line below simply
//!     does not apply to it.
//!   - **Planes12x** — [`Planes12xBlocks`], `Planes14` minus its third plane:
//!     two planes address levels `0..4`, so the main record is
//!     `9 + 1 + 24 + 2·24 = 82` bits in a frozen [`PLANES12X_BYTES`]-byte
//!     stride. A block whose class carries five magnitudes cannot be said in
//!     two planes; the main stream takes its best `L ≤ 4` neighbour and the
//!     block is repeated **exactly** in an exception table, at
//!     [`PLANES12X_EXC_BYTES`] bytes each (a `u32` block index plus a full
//!     `Planes14` record). Cost is therefore `96·n + 144·n_exc` bits, and
//!     `n_exc` is a *count of five-level classes* — which this bench knows
//!     per block without decoding anything.
//! * addressing: a variable-width stream needs to say where block `i` starts.
//!   Charged two ways: +16 bits/block (a u16 offset each), and **grouped-32**
//!   (all 32 lanes of a SIMD group read the group's max width, one u32 base
//!   per group — padding traded against offsets). The two `Planes*` layouts
//!   have a constant stride and pay nothing here.
//!
//! ## ⚠️ Two accountings, and they are not interchangeable
//!
//! The dossier has been bitten three times by mixing byte accountings in one
//! list, so this bench states both and never blends them.
//!
//! * **payload** (`b/poids payload`, the historical column of this bench):
//!   numerator = the per-block stream and its base table, nothing else;
//!   denominator = **quantized weights only**, `24 × blocks`. Tails and row
//!   scales are excluded from both ends. This is the number that answers
//!   "what does one block cost", and it is what `Slot32 = 5.3756` means here.
//! * **kernel** (`b/poids noyau`), the accounting of the CUDA bench
//!   (`llvq-cuda/src/bin/planesbench.rs`, `docs/mesures/e2-golay70-*.txt`):
//!   numerator = the same stream **plus** the `f32` tail block and the `f32`
//!   row-scale vector every arm uploads; denominator = **every weight of the
//!   matrix**, `d_out · d_in`, tail columns included. This is what a matvec
//!   actually streams per weight, and it is what `Slot32 = 5.510`,
//!   `Planes14 = 4.804`, `Planes12x = 4.342` mean. Reproducing those three
//!   numbers from real blocks, on the CPU, is this bench's acceptance test —
//!   see `published_4b_kernel_rates_are_reproduced`.
//!
//! Neither is `b/param modèle entier`: that third grandeur divides by the
//! whole model, embedding included, and is the only one comparable to an
//! AWQ figure. It is printed only when the file is self-contained, so the
//! carried parameters are **counted from the file** rather than assumed.
//!
//! ## Run
//!
//! `cargo run --release -p llvq-bench --bin rtbits [path/to/model.llvq]`
//!
//! Without a path: 20 000 gaussian blocks through `nearest_angular`, the
//! same source as `classprofile`/`arrbits`. With one: every block of the
//! artifact, exhaustively.
//!
//! [`PlanesBlocks`]: llvq_artifact::runtime::PlanesBlocks
//! [`Planes12xBlocks`]: llvq_artifact::runtime::Planes12xBlocks

use llvq_artifact::e1c::{group_words, E1C12_PLANES, E1C14_PLANES, E1C_GROUP};
use llvq_artifact::runtime::{
    PLANES12X_BYTES, PLANES12X_EXC_BYTES, PLANES12X_LEVEL_CAP, PLANES12X_PLANE_BIT, PLANES14_BYTES,
};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;
use std::fs::File;
use std::io::{BufReader, Read};

/// Bits for the class id: 383 classes plus the origin fit in 9.
const CLASS_BITS: u64 = 9;
/// SIMD group width the grouped layout is sized for.
const GROUP: usize = 32;
/// Bits an `f32` costs. The kernel accounting stores the tail block and the
/// row-scale vector in single precision — not `f16`, not the `f64` the
/// archive keeps them in — because that is what every arm of the CUDA bench
/// uploads. Naming it once keeps the three layouts on one convention.
const F32_BITS: u64 = 32;
/// Values per scale/bias pair on the `q8` embedding path
/// (`llvq_llm::fused::EMBED_GROUP`), and the 32 bits that pair costs.
const EMBED_GROUP: f64 = 64.0;
const EMBED_GROUP_BITS: f64 = 32.0;

/// ⌈log₂ L⌉ — bits per slot of the positional encoding.
fn lg_ceil(l: usize) -> u64 {
    match l {
        0 | 1 => 0,
        _ => (usize::BITS - (l - 1).leading_zeros()) as u64,
    }
}

/// Nested-mask bits: canonical order, one mask per level over the slots the
/// previous levels left free; the last level is whatever remains.
fn mask_bits(lv: &ClassLevels) -> u64 {
    let mut left = DIM as u64;
    let mut bits = 0u64;
    for i in 0..lv.len.saturating_sub(1) {
        bits += left;
        left -= lv.counts[i] as u64;
    }
    bits
}

/// Per-block widths of the variable candidates, everything included
/// except addressing.
#[derive(Clone, Copy)]
struct Widths {
    mask: u64,
    pos: u64,
    /// `Layout::Slot32`, the production layout — mirrors `width_slot` in
    /// `llvq_artifact::runtime::ClassTable::new`. The sign field there is
    /// one bit per **slot**, not per nonzero, so `nonzero` cancels out of
    /// `common` and must not appear.
    slot: u64,
    levels: usize,
    /// The all-zero block, whose record stops after the header and so does
    /// not follow the `9 + gain + 24·L` rule.
    origin: bool,
}

fn widths(lv: &ClassLevels, gain_bits: u64) -> Widths {
    let common = lv.nonzero as u64 + CLASS_BITS + gain_bits;
    Widths {
        mask: mask_bits(lv) + common,
        pos: lg_ceil(lv.len) * DIM as u64 + common,
        slot: CLASS_BITS + gain_bits + DIM as u64 * lv.len as u64,
        levels: lv.len,
        origin: false,
    }
}

/// `Slot32` stride, in bits, of a group whose widest block has `levels`
/// levels — the byte-rounded field every lane of the group then reads.
fn slot_stride_bits(levels: usize, gain_bits: u64) -> u64 {
    (CLASS_BITS + gain_bits + DIM as u64 * levels as u64).div_ceil(8) * 8
}

// ---------------------------------------------------------------------------
// The two bit-plane layouts, priced post by post
// ---------------------------------------------------------------------------
//
// Both are **frozen strides**, not computed ones: `transcode_planes14` and
// `transcode_planes12x` write a constant number of bytes per block whatever
// the class, so the constants come from `llvq_artifact::runtime` rather than
// from a local literal. If the format moves, this bench moves with it or
// fails to compile — the drift that a hard-coded `14` would hide.

/// `Planes14` block stream, in bits. Post by post, per block:
///
/// * class 9 + gain 1 — the header, same two fields every layout pays;
/// * sign mask 24 — one bit per **slot**, not per nonzero: zero slots carry
///   a bit written 0, which is what buys the fixed field offsets;
/// * three 24-bit planes — the slot's level as an explicit 3-bit index
///   (`len <= 5`, so 3 planes always suffice), instead of `L−1` one-hot
///   masks. This is the whole trade: `Slot32` pays `24·L` and needs a
///   per-group stride, `Planes14` pays a flat `72` and needs none;
/// * 6 padding bits to close the [`PLANES14_BYTES`]-byte record.
///
/// Sums to `9 + 1 + 24 + 72 + 6 = 112` bits — and the origin block, whose
/// planes and sign mask are all zero, pays the same 112: the stride is the
/// format. **No base table**, so there is no addressing term to add.
fn planes14_bits(blocks: u64) -> u64 {
    blocks * PLANES14_BYTES as u64 * 8
}

/// `Planes12x` main stream **plus** its exception table, in bits.
///
/// Main record: class 9 + gain 1 + sign mask 24 + **two** 24-bit planes + 14
/// padding bits = [`PLANES12X_BYTES`] bytes. Two planes address levels
/// `0..4`, so this record is exact for every block whose class holds at most
/// [`PLANES12X_LEVEL_CAP`] magnitudes.
///
/// A five-level block cannot be said in two planes. The transcoder puts its
/// best `L ≤ 4` neighbour in the main stream and repeats the block exactly in
/// the exception table, at [`PLANES12X_EXC_BYTES`] bytes: a `u32` block index
/// (4 o, so the correction pass can find it) plus a full [`PLANES14_BYTES`]
/// record of the exact block. The exception is an **addition**, never a
/// replacement — the main record is written either way, which is why the
/// two terms add rather than trade.
///
/// So the count of exceptions is a count of five-level classes, and nothing
/// else: no lattice search, no decoding, no probability. `n_exc` here is the
/// same number `transcode_planes12x` will push onto `exc_idx`.
fn planes12x_bits(blocks: u64, exceptions: u64) -> u64 {
    blocks * PLANES12X_BYTES as u64 * 8 + exceptions * PLANES12X_EXC_BYTES as u64 * 8
}

// ---------------------------------------------------------------------------
// E1c — the same two streams, transposed over the warp (item X1/X2 of
// docs/spec-memoire-extreme-2026-08-12.md)
// ---------------------------------------------------------------------------
//
// `Planes14` and `Planes12x` each close their record on a byte boundary and
// pay for it: 6 padding bits per block and 14 respectively. Transposing the
// per-slot bits over a group of [`E1C_GROUP`] blocks removes that rounding
// entirely — a group is a whole number of 32-bit words, because the group *is*
// the word width — so E1c costs exactly the source's **payload** bits and
// nothing more.
//
// This is a pricing arm, not a claim about speed. The layout trades 82 or 106
// broadcast loads per group against 3 or 4 per lane, and which side wins is
// the question `bin/planesbench` exists to answer (spec X3: ≥ 1.9× to replace
// `Planes12x`, ≥ 2.05× to replace `Planes14`). What this bench can settle, and
// what it settles exactly, is the byte count.

/// `E1c14` stream, in bits: whole groups of [`E1C_GROUP`] blocks, each
/// [`group_words`]`(3)` words. The trailing group is padded with origin
/// blocks and **is charged** — a matrix whose block count is not a multiple
/// of 32 really does upload those words.
fn e1c14_bits(blocks: u64) -> u64 {
    blocks.div_ceil(E1C_GROUP as u64) * group_words(E1C14_PLANES) as u64 * 32
}

/// `E1c12` main stream plus the **same** exception table as `Planes12x` —
/// same predicate, same records, carried verbatim by
/// [`E1c12Blocks::from_planes12x`]. The two layouts therefore differ by
/// exactly the main stream's padding, which is what makes the comparison
/// below a single-variable one.
///
/// [`E1c12Blocks::from_planes12x`]: llvq_artifact::e1c::E1c12Blocks::from_planes12x
fn e1c12_bits(blocks: u64, exceptions: u64) -> u64 {
    blocks.div_ceil(E1C_GROUP as u64) * group_words(E1C12_PLANES) as u64 * 32
        + exceptions * PLANES12X_EXC_BYTES as u64 * 8
}

/// A block is an exception of `Planes12x` exactly when its class carries more
/// than [`PLANES12X_LEVEL_CAP`] magnitude levels — the predicate
/// `planes12x_chunk` tests, written once so a test can mutate it.
///
/// The origin (one level, every coordinate zero) is never an exception.
fn is_planes12x_exception(levels: usize) -> bool {
    levels > PLANES12X_LEVEL_CAP
}

/// The same predicate read off the **record geometry** instead of the named
/// cap: `k` bit-planes address `2^k` levels, and a `Planes12x` main record
/// holds [`PLANES12X_PLANE_BIT`]`.len()` of them.
///
/// Two routes to one number, on purpose. A cross-check whose two sides call
/// the same function is a tautology and proves nothing; this one would
/// survive only if the cap constant and the number of planes moved together,
/// which is the only way they *may* move.
fn exceeds_planes12x_geometry(levels: usize) -> bool {
    levels > 1 << PLANES12X_PLANE_BIT.len()
}

/// The shape counts one matrix contributes to the **kernel** accounting.
///
/// Named for shapes, not for the carried tensors: `carried_params` below is
/// a different thing entirely (the embedding), and confusing the two is how
/// a b/param figure goes wrong.
///
/// Mirrors `Mat`'s three counts in `llvq-cuda/src/bin/planesbench.rs`:
/// `nblocks = d_out · (d_in / 24)`, `tail_w = d_in % 24` columns kept in
/// full precision on every row, and `d_out` row scales.
#[derive(Default, Clone, Copy)]
struct Shapes {
    /// `Σ d_out · d_in` — every weight, tail columns included. **This** is
    /// the kernel accounting's denominator, and it is larger than
    /// `24 · blocks`; swapping the two is the mistake that turns 4.804 into
    /// 4.827.
    weights: u64,
    /// `Σ d_out · (d_in mod 24)` — the `TailPolicy::KeepExact` columns, one
    /// `f32` each on the device.
    tail_weights: u64,
    /// `Σ d_out` — one `f32` row scale each.
    rows: u64,
}

impl Shapes {
    fn push(&mut self, d_out: usize, d_in: usize) {
        self.weights += (d_out * d_in) as u64;
        self.tail_weights += (d_out * (d_in % DIM)) as u64;
        self.rows += d_out as u64;
    }

    /// Bits the kernel accounting adds to a block stream: the `f32` tail and
    /// the `f32` row scales, uploaded identically by every LLVQ arm.
    fn side_bits(&self) -> u64 {
        (self.tail_weights + self.rows) * F32_BITS
    }

    /// Bits per weight in the kernel accounting: stream + side data over
    /// **all** weights.
    fn kernel_bpw(&self, stream_bits: u64) -> f64 {
        (stream_bits + self.side_bits()) as f64 / self.weights as f64
    }
}

/// Bits per parameter of the **whole model**: a weighted mean of `(count,
/// bits per parameter)` groups over their total count.
///
/// The only grandeur comparable to an AWQ figure, and the one the dossier
/// got wrong twice by quoting a payload rate against a whole-model one. It
/// takes a list rather than two arguments because a real model has three
/// groups, not two — quantized linears, embedding tables, and the norms that
/// stay `f16` whatever is done to the embedding. Every count is printed
/// beside the result so the division is auditable.
fn model_bpw(groups: &[(u64, f64)]) -> f64 {
    let (bits, n) = groups
        .iter()
        .fold((0.0, 0u64), |(b, n), &(c, r)| (b + c as f64 * r, n + c));
    bits / n as f64
}

#[derive(Default)]
struct Acc {
    blocks: u64,
    mask: u64,
    pos: u64,
    levels: [u64; MAX_LEVELS + 1],
    worst_mask: u64,
    worst_pos: u64,
}

impl Acc {
    fn push(&mut self, w: &Widths) {
        self.blocks += 1;
        self.mask += w.mask;
        self.pos += w.pos;
        self.levels[w.levels] += 1;
        self.worst_mask = self.worst_mask.max(w.mask);
        self.worst_pos = self.worst_pos.max(w.pos);
    }
}

/// Tracks the grouped-32 layout: every lane of a group reads the group's max
/// width, one u32 base pointer per group.
#[derive(Default)]
struct Grouped {
    filled: usize,
    max_mask: u64,
    max_pos: u64,
    max_slot: u64,
    max_levels: usize,
    /// Does this group hold a block of exactly 4 levels? Under an `L ≤ 4`
    /// cap such a group's stride is *exactly* 14 bytes — see `close`.
    has_four: bool,
    /// Origin blocks seen in the group. When every lane is the origin the
    /// group's max width is the 10-bit header, which the general
    /// `9 + gain + 24·L` rule does not describe.
    origins: usize,
    sum_levels: u64,
    /// Totals over closed groups.
    mask_bits: u64,
    pos_bits: u64,
    /// Same stride, rounded up to a whole byte — what lanes can actually
    /// address.
    mask_bits_byte: u64,
    /// `Slot32`, byte-rounded stride: the production format's real cost.
    slot_bits_byte: u64,
    max_levels_sum: u64,
    /// Groups by their widest block's level count — the distribution the
    /// mean of maxima hides, and what decides the capped rate.
    maxl_hist: [u64; MAX_LEVELS + 1],
    groups_with_four: u64,
    /// Groups in which every lane is the origin, excluded from `maxl_hist`
    /// so the histogram cross-check stays a closed identity.
    origin_only_groups: u64,
    groups: u64,
}

impl Grouped {
    fn push(&mut self, w: &Widths) {
        self.filled += 1;
        self.max_mask = self.max_mask.max(w.mask);
        self.max_pos = self.max_pos.max(w.pos);
        self.max_slot = self.max_slot.max(w.slot);
        self.max_levels = self.max_levels.max(w.levels);
        self.has_four |= w.levels == 4;
        self.origins += usize::from(w.origin);
        self.sum_levels += w.levels as u64;
        if self.filled == GROUP {
            self.close();
        }
    }

    fn close(&mut self) {
        if self.filled == 0 {
            return;
        }
        // A partial trailing group still pays full-width lanes.
        self.mask_bits += GROUP as u64 * self.max_mask + 32;
        self.pos_bits += GROUP as u64 * self.max_pos + 32;
        self.mask_bits_byte += GROUP as u64 * self.max_mask.div_ceil(8) * 8 + 32;
        self.slot_bits_byte += GROUP as u64 * self.max_slot.div_ceil(8) * 8 + 32;
        self.max_levels_sum += self.max_levels as u64;
        if self.origins == self.filled {
            self.origin_only_groups += 1;
        } else {
            self.maxl_hist[self.max_levels] += 1;
        }
        self.groups_with_four += u64::from(self.has_four);
        self.groups += 1;
        self.filled = 0;
        self.max_mask = 0;
        self.max_pos = 0;
        self.max_slot = 0;
        self.max_levels = 0;
        self.has_four = false;
        self.origins = 0;
    }
}

/// Levels of a raw lattice point, canonical order — for the gaussian mode,
/// where blocks come from the searcher rather than from an artifact.
fn levels_of_point(p: &[i32; DIM]) -> ClassLevels {
    let mut kinds: Vec<(i32, u8)> = Vec::new();
    for &v in p {
        let a = v.abs();
        match kinds.iter_mut().find(|(u, _)| *u == a) {
            Some((_, n)) => *n += 1,
            None => kinds.push((a, 1)),
        }
    }
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut lv = ClassLevels {
        len: kinds.len(),
        odd: p.iter().all(|v| v % 2 != 0),
        shell: 0, // unused here
        ..Default::default()
    };
    for (i, &(v, n)) in kinds.iter().enumerate() {
        lv.values[i] = v;
        lv.counts[i] = n;
        if v != 0 {
            lv.nonzero += n;
        }
    }
    lv
}

fn main() {
    let path = std::env::args().nth(1);
    let fd = FastDecoder::new();

    let mut odd = Acc::default();
    let mut even = Acc::default();
    let mut grouped = Grouped::default();
    let mut origin_blocks = 0u64;
    let mut gain_bits = 1u64;
    // Archive rate anchor: index + gain bits per block, from the file when
    // there is one (the published 4B is cap 12: 47 + 1).
    let mut arch_bits = 49u64;
    let source: String;

    // ---- kernel-accounting bookkeeping (file mode only) ----
    // Shapes, so the tail and row-scale terms are the file's own and not a
    // remembered constant; and the exception count of `Planes12x`, which is
    // a plain count of five-level classes.
    let mut carried = Shapes::default();
    let mut exceptions = 0u64;
    // Parameters the file carries unquantized (embedding, lm_head). Counted
    // from the raw-tensor section when the file is self-contained, left at 0
    // otherwise — a projections-only file cannot know them, and guessing
    // them is exactly how a b/param figure goes wrong.
    let mut carried_params = 0u64;
    // Of those, the ones the `LLVQ_EMBED=q8` path actually replaces. The
    // names are `llvq_llm::fused::{EMBED_NAME, HEAD_NAME}`, repeated here
    // rather than depended on: `llvq-bench` stays dependency-free, and a
    // 690-crate tree to learn two strings is not a trade. Everything else
    // carried (the norms) stays f16 on both lines — charging norms at 8.5
    // would be inventing a quantizer that does not exist.
    let mut carried_embed = 0u64;
    // And what those tensors actually weigh in the file, encoding included.
    // The `f16 = 16` / `q8 = 8.5` rows below are a *model* of the carried
    // side; this is the file's own answer, and printing both is how the
    // model gets audited instead of trusted.
    let mut carried_bits = 0u64;

    // Per-class cross-check bookkeeping: the first block seen of each class
    // is decoded and its point-derived levels compared to the class table.
    let mut verified = vec![false; fd.n_classes()];
    let mut verified_classes = 0usize;

    if let Some(path) = path {
        // ---- every block of a real artifact ----
        let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut r = BufReader::new(f);
        let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
        source = format!("{path} — {} matrices", h.matrices);
        for _ in 0..h.matrices {
            let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
            // The class layout runs shells ascending, so the cap-c ball is a
            // strict prefix of the cap-13 layout: a table built for 13
            // decodes any cap ≤ 13 file identically.
            assert!(
                m.shell_cap <= 13,
                "{}: shell cap {} exceeds the class table's 13",
                m.name, m.shell_cap
            );
            gain_bits =
                m.centroids.len().next_power_of_two().trailing_zeros() as u64;
            arch_bits =
                llvq_quant::quantizer::index_bits(m.shell_cap) as u64 + gain_bits;
            // The two bit-plane layouts freeze the gain field at bit 9, one
            // bit wide (`transcode_planes14` refuses anything else). Pricing
            // them on a file they could not encode would be a number about
            // nothing, so refuse here rather than print it.
            assert_eq!(
                gain_bits, 1,
                "{}: {} gain bits — Planes14/Planes12x freeze the field at 1",
                m.name,
                gain_bits
            );
            carried.push(m.d_out, m.d_in);
            for &idx in &m.indices {
                let Some(ci) = fd.class_of(idx) else {
                    assert_eq!(idx, 0, "{}: index {idx} outside the ball", m.name);
                    origin_blocks += 1;
                    // The origin still occupies a lane: one level, no signs,
                    // no masks — only the class id and the gain bit.
                    let w = Widths {
                        mask: CLASS_BITS + gain_bits,
                        pos: CLASS_BITS + gain_bits,
                        // Mirrors the origin record of `ClassTable::new`:
                        // no sign field and no masks, so the general
                        // `9 + gain + 24·L` does not apply.
                        slot: CLASS_BITS + gain_bits,
                        levels: 1,
                        origin: true,
                    };
                    even.push(&w);
                    grouped.push(&w);
                    continue;
                };
                let lv = fd.levels(ci);
                if !verified[ci] {
                    let p = fd.decode(idx).expect("valid index");
                    assert_eq!(
                        levels_of_point(&p),
                        ClassLevels { shell: 0, ..*lv },
                        "class {ci}: table disagrees with a decoded point"
                    );
                    verified[ci] = true;
                    verified_classes += 1;
                }
                exceptions += u64::from(is_planes12x_exception(lv.len));
                let w = widths(lv, gain_bits);
                if lv.odd { &mut odd } else { &mut even }.push(&w);
                grouped.push(&w);
            }
        }
        // ---- carried tensors, counted rather than assumed ----
        // A self-contained file lists its raw tensors right after the
        // matrices: a u32 count, then one framed tensor each. Reading them
        // costs one transient allocation per tensor and turns "b/param
        // modèle entier" from a figure with a remembered vocabulary size
        // into one the file itself answers for.
        if h.is_self_contained() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).expect("raw-tensor count");
            for _ in 0..u32::from_le_bytes(b) {
                let t = llvq_artifact::read_raw(&mut r, h.version).expect("valid raw tensor");
                carried_params += t.len() as u64;
                carried_bits += t.bytes() * 8;
                if matches!(
                    t.name.as_str(),
                    "model.embed_tokens.weight" | "lm_head.weight"
                ) {
                    carried_embed += t.len() as u64;
                }
            }
        }
    } else {
        // ---- gaussian source, same harness as classprofile/arrbits ----
        const N: usize = 20_000;
        source = format!("{N} blocs gaussiens, nearest_angular");
        let mut rng = SplitMix64::new(0x6_A88B);
        let s = Searcher::new();
        let mut ball = BallSearcher::new();
        for _ in 0..N {
            let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
            let f = ball.nearest_angular(&s, &x);
            let lv = levels_of_point(&f.point);
            exceptions += u64::from(is_planes12x_exception(lv.len));
            let w = widths(&lv, gain_bits);
            if lv.odd { &mut odd } else { &mut even }.push(&w);
            grouped.push(&w);
        }
    }
    grouped.close();

    let total = odd.blocks + even.blocks;
    let pct = |n: u64| 100.0 * n as f64 / total as f64;
    let bpw = |bits: u64, blocks: u64| bits as f64 / (blocks * DIM as u64) as f64;

    println!("source : {source}");
    println!("{total} blocs — impair {:.1} %, pair {:.1} %", pct(odd.blocks), pct(even.blocks));
    if origin_blocks > 0 {
        println!("dont {origin_blocks} blocs origine (tout zéro)");
    }
    if verified_classes > 0 {
        println!("{verified_classes} classes recoupées point décodé ↔ table");
    }

    println!("\n  niveaux de magnitude par bloc");
    println!("  {}", "-".repeat(56));
    for l in 1..=MAX_LEVELS {
        let n = odd.levels[l] + even.levels[l];
        if n > 0 {
            println!("  {l} niveaux{:>16} blocs{:>8.2} %", n, pct(n));
        }
    }

    // ---- the table the decision is made from ----
    let all_mask = odd.mask + even.mask;
    let all_pos = odd.pos + even.pos;
    let worst_mask = odd.worst_mask.max(even.worst_mask);
    let worst_pos = odd.worst_pos.max(even.worst_pos);
    // Pad the worst case to the next u32 so every lane load is aligned.
    let fixed_mask = worst_mask.div_ceil(32) * 32;
    let fixed_pos = worst_pos.div_ceil(32) * 32;

    println!("\n  bits/poids en RAM, comptabilité complète");
    println!("  (assignation + signes + classe {CLASS_BITS} b + gain {gain_bits} b + adressage)");
    println!("  {}", "-".repeat(72));
    println!(
        "  {:<44}{:>10}{:>10}",
        "layout", "masques", "positionnel"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable, adressage gratuit (borne inf.)",
        bpw(all_mask, total),
        bpw(all_pos, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable + offset u16 par bloc",
        bpw(all_mask + 16 * total, total),
        bpw(all_pos + 16 * total, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "groupé 32 (stride = max du groupe + base u32)",
        bpw(grouped.mask_bits, total),
        bpw(grouped.pos_bits, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10}",
        "groupé 32, stride arrondi à l'octet",
        bpw(grouped.mask_bits_byte, total),
        "—"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        format!("champ fixe au pire cas ({worst_mask}/{worst_pos} b → u32)"),
        bpw(fixed_mask * total, total),
        bpw(fixed_pos * total, total)
    );
    println!(
        "  {:<44}{:>10}{:>10.4}",
        "champ fixe 128 (uint4, nibbles)", "—",
        128.0 / DIM as f64
    );

    println!("\n  par coset (variable, adressage gratuit)");
    println!("  {}", "-".repeat(72));
    for (name, a) in [("impair", &odd), ("pair", &even)] {
        if a.blocks > 0 {
            println!(
                "  {:<10}{:>9.1} %   masques {:.4}   positionnel {:.4}   pire {}/{} b",
                name,
                pct(a.blocks),
                bpw(a.mask, a.blocks),
                bpw(a.pos, a.blocks),
                a.worst_mask,
                a.worst_pos
            );
        }
    }

    // ---- divergence proxy ----
    if grouped.groups > 0 {
        let mean_l = grouped.sum_levels as f64 / total as f64;
        let mean_max_l = grouped.max_levels_sum as f64 / grouped.groups as f64;
        println!("\n  divergence sur groupes de {GROUP} lanes consécutives");
        println!("  {}", "-".repeat(72));
        println!(
            "  niveaux : moyenne {mean_l:.3}, moyenne des max par groupe {mean_max_l:.3} \
             (×{:.3})",
            mean_max_l / mean_l
        );
        println!(
            "  padding groupé 32 : masques ×{:.3}, positionnel ×{:.3} vs adressage gratuit",
            grouped.mask_bits as f64 / all_mask as f64,
            grouped.pos_bits as f64 / all_pos as f64
        );
    }

    // ---- Slot32: the layout the fused kernel reads, and what a level cap
    // would do to it ----
    //
    // `Slot32` is a third assignment encoding, not a variant of the two
    // above: one 24-bit mask per level in **slot** space, level 0 implicit,
    // signs indexed by slot. It buys fixed field offsets — the whole reason
    // the fused kernel has no popcount chain and no serial state — and pays
    // `24 − nonzero` bits per block for them.
    if grouped.groups > 0 {
        // Same total by a second, independent route: from the histogram of
        // per-group maxima, through the closed-form stride. If the running
        // accumulator and this disagree, one of the two is wrong and no
        // figure below can be believed.
        let from_hist: u64 = (1..=MAX_LEVELS)
            .map(|l| grouped.maxl_hist[l] * GROUP as u64 * slot_stride_bits(l, gain_bits))
            .sum::<u64>()
            + grouped.origin_only_groups
                * GROUP as u64
                * (CLASS_BITS + gain_bits).div_ceil(8)
                * 8
            + 32 * grouped.groups;
        assert_eq!(
            from_hist, grouped.slot_bits_byte,
            "l'histogramme des max et l'accumulateur de strides divergent"
        );

        println!("\n  Slot32 — le layout que le noyau fusé lit réellement");
        println!("  {}", "-".repeat(72));
        println!(
            "  groupé 32, stride octet, base u32 : {:.4} b/poids",
            bpw(grouped.slot_bits_byte, total)
        );
        println!("  {} groupes, max de niveaux :", grouped.groups);
        for l in 1..=MAX_LEVELS {
            if grouped.maxl_hist[l] > 0 {
                println!(
                    "    L = {l}{:>14} groupes{:>9.4} %   stride {} o",
                    grouped.maxl_hist[l],
                    100.0 * grouped.maxl_hist[l] as f64 / grouped.groups as f64,
                    slot_stride_bits(l, gain_bits) / 8
                );
            }
        }
        if grouped.origin_only_groups > 0 {
            println!(
                "    origine seule{:>13} groupes",
                grouped.origin_only_groups
            );
        }

        // What an L ≤ 4 encoder cap would cost. This is a **bound**, not a
        // simulation: capping re-quantizes, so the classes of the L = 5
        // blocks would change and cannot be predicted here. What can be
        // stated exactly:
        //
        //  * `L ≤ 4 ⇒ width_slot ≤ 9 + gain + 96 = 106 b ⇒ stride ≤ 14 o`,
        //    so 14 bytes is an unconditional majorant of every group;
        //  * a block already at `L ≤ 4` keeps its class under the cap — its
        //    codeword is the argmin over the full ball and stays the argmin
        //    over a subset that still contains it — so a group holding an
        //    L = 4 block reaches exactly 14 o, majorant attained.
        let capped_bits = grouped.groups * (GROUP as u64 * slot_stride_bits(4, gain_bits) + 32);
        println!(
            "  sous plafond L ≤ 4 : ≤ {:.4} b/poids ({} o de stride, majorant inconditionnel)",
            bpw(capped_bits, total),
            slot_stride_bits(4, gain_bits) / 8
        );
        println!(
            "  majorant ATTEINT sur {} groupes sur {} ({:.4} %) : ceux qui portent déjà un\n  \
             bloc L = 4, dont la classe est inchangée par le plafond. Gain {:.3} b/poids ({:.1} %).",
            grouped.groups_with_four,
            grouped.groups,
            100.0 * grouped.groups_with_four as f64 / grouped.groups as f64,
            bpw(grouped.slot_bits_byte, total) - bpw(capped_bits, total),
            100.0 * (1.0 - bpw(capped_bits, total) / bpw(grouped.slot_bits_byte, total))
        );
    }

    // ---- Planes14 / Planes12x: the bit-plane layouts ----
    //
    // Neither is grouped and neither has a base table, so both are priced by
    // a stride times a block count — plus, for Planes12x, a table whose size
    // is a *count of five-level classes*. Payload accounting: the stream
    // alone, over `24 · blocks`, exactly like every column above.
    let p14_bits = planes14_bits(total);
    let p12_bits = planes12x_bits(total, exceptions);
    let exc_pct = 100.0 * exceptions as f64 / total as f64;
    println!("\n  Planes14 et Planes12x — les plans binaires (b/poids payload)");
    println!("  {}", "-".repeat(72));
    println!(
        "  Planes14   stride {PLANES14_BYTES} o fixe, sans base ni exception : {:.4} b/poids",
        bpw(p14_bits, total)
    );
    println!(
        "  Planes12x  stride {PLANES12X_BYTES} o + {} exceptions à {PLANES12X_EXC_BYTES} o \
         ({exc_pct:.4} % des blocs)",
        exceptions
    );
    println!(
        "             = {:.4} b/poids, dont {:.4} de stream et {:.4} d'exceptions",
        bpw(p12_bits, total),
        bpw(total * PLANES12X_BYTES as u64 * 8, total),
        bpw(exceptions * PLANES12X_EXC_BYTES as u64 * 8, total)
    );
    // The exception count is a histogram identity, not a second measurement:
    // a block is an exception iff its class has more than PLANES12X_LEVEL_CAP
    // levels. Stating it as an assert makes a drift between the predicate and
    // the level histogram impossible to publish.
    let from_hist: u64 = (1..=MAX_LEVELS)
        .filter(|&l| exceeds_planes12x_geometry(l))
        .map(|l| odd.levels[l] + even.levels[l])
        .sum();
    assert_eq!(
        from_hist, exceptions,
        "le compte d'exceptions et l'histogramme de niveaux divergent"
    );
    println!(
        "  seuil : L > {PLANES12X_LEVEL_CAP} — {} blocs à {} niveaux, recoupés sur l'histogramme",
        exceptions,
        MAX_LEVELS
    );

    // ---- E1c: the same two streams with the byte rounding removed ----
    let e14_bits = e1c14_bits(total);
    let e12_bits = e1c12_bits(total, exceptions);
    println!("\n  E1c — les mêmes flux transposés sur le warp (b/poids payload)");
    println!("  {}", "-".repeat(72));
    println!(
        "  E1c14      {} mots par groupe de {E1C_GROUP}, sans bourrage : {:.4} b/poids \
         ({:+.4} vs Planes14)",
        group_words(E1C14_PLANES),
        bpw(e14_bits, total),
        bpw(e14_bits, total) - bpw(p14_bits, total)
    );
    println!(
        "  E1c12      {} mots par groupe + les MÊMES {} exceptions : {:.4} b/poids \
         ({:+.4} vs Planes12x)",
        group_words(E1C12_PLANES),
        exceptions,
        bpw(e12_bits, total),
        bpw(e12_bits, total) - bpw(p12_bits, total)
    );
    println!(
        "             dont {:.4} de stream et {:.4} d'exceptions — le terme d'exceptions\n  \
         est identique à celui de Planes12x, donc l'écart entre les deux layouts EST le bourrage",
        bpw(e12_bits - exceptions * PLANES12X_EXC_BYTES as u64 * 8, total),
        bpw(exceptions * PLANES12X_EXC_BYTES as u64 * 8, total)
    );

    // ---- the kernel accounting, the one the CUDA bench publishes ----
    //
    // Same streams, two changes and only two: the `f32` tail block and the
    // `f32` row-scale vector join the numerator, and the denominator becomes
    // every weight of every matrix instead of `24 · blocks`. Both changes are
    // *inflationary on the numerator* and *inflationary on the denominator*,
    // and they do not cancel — which is precisely why the two columns must
    // never be quoted in one list.
    if carried.weights > 0 {
        let tail_w = carried.weights - total * DIM as u64;
        println!("\n  b/poids NOYAU — comptabilité du banc CUDA (planesbench.rs)");
        println!("  {}", "-".repeat(72));
        println!(
            "  numérateur += queue f32 ({tail_w} poids) + échelles de ligne f32 ({} lignes)",
            carried.rows
        );
        println!(
            "  dénominateur = {} poids (Σ d_out·d_in, queue comprise), et non {} quantifiés",
            carried.weights,
            total * DIM as u64
        );
        assert_eq!(
            tail_w, carried.tail_weights,
            "les poids de queue comptés par forme et par différence divergent"
        );
        for (name, bits) in [
            ("Slot32", grouped.slot_bits_byte),
            ("Planes14", p14_bits),
            ("Planes12x", p12_bits),
            ("E1c14", e14_bits),
            ("E1c12", e12_bits),
        ] {
            println!(
                "  {name:<12}{:>10.4} b/poids noyau  ({:.4} payload, {:.2} Go de stream)",
                carried.kernel_bpw(bits),
                bpw(bits, total),
                (bits + carried.side_bits()) as f64 / 8.0 / 1e9
            );
        }
    }

    // ---- b/param modèle entier: the only figure comparable to an AWQ one ----
    //
    // Printed only when the file carries its embedding, so both counts of the
    // division come from the same bytes. `q8` is the `LLVQ_EMBED=q8` path:
    // int8 plus one f16 scale and one f16 bias per group of 64, i.e. exactly
    // 8 + 32/64 = 8.5 bits per parameter when the row length is a multiple of
    // 64 (2560 and 4096 both are).
    if carried_params > 0 && carried.weights > 0 {
        let embed_q8 = 8.0 + EMBED_GROUP_BITS / EMBED_GROUP;
        let other = carried_params - carried_embed;
        println!("\n  b/param MODÈLE ENTIER — la seule grandeur comparable à un chiffre AWQ");
        println!("  {}", "-".repeat(72));
        println!(
            "  {} poids quantifiés + {} d'embedding + {other} portés en f16 = {} paramètres",
            carried.weights,
            carried_embed,
            carried.weights + carried_params
        );
        println!(
            "  embedding f16 = 16,000 b/param ; q8 g{:.0} = {embed_q8:.3} b/param \
             (int8 + f16 échelle et biais par groupe)",
            EMBED_GROUP
        );
        // The third column is not a model: it is the carried section's own
        // byte count, so the row that matches this file's encoding must
        // reproduce one of the two modelled columns. If it does not, the
        // model is wrong and the modelled columns are not publishable.
        let carried_measured = carried_bits as f64 / carried_params as f64;
        println!(
            "  porté MESURÉ dans le fichier : {carried_measured:.3} b/param \
             ({:.2} Go sur {carried_params} paramètres)",
            carried_bits as f64 / 8.0 / 1e9
        );
        println!(
            "  {:<12}{:>14}{:>14}{:>14}",
            "layout", "embed f16", "embed q8", "porté mesuré"
        );
        for (name, bits) in [
            ("Slot32", grouped.slot_bits_byte),
            ("Planes14", p14_bits),
            ("Planes12x", p12_bits),
            ("E1c14", e14_bits),
            ("E1c12", e12_bits),
        ] {
            let lin = (carried.weights, carried.kernel_bpw(bits));
            println!(
                "  {name:<12}{:>14.3}{:>14.3}{:>14.3}",
                model_bpw(&[lin, (carried_embed, 16.0), (other, 16.0)]),
                model_bpw(&[lin, (carried_embed, embed_q8), (other, 16.0)]),
                model_bpw(&[lin, (carried_params, carried_measured)])
            );
        }
    }

    // ---- what each rate means for one token ----
    //
    // Both terms used to be Qwen3-4B constants. They are now taken from the
    // file when there is one — a 4B file reproduces the old numbers exactly
    // (3 633 315 840 weights, 0.778 GB of tied embedding), and an 8B file no
    // longer gets a 4B's head silently charged to it.
    let (linear_w, lm_head_gb, head_src) = match (carried.weights, carried_params) {
        (0, _) => (3_633_315_840f64, 0.778, "constantes Qwen3-4B"),
        (w, 0) => (w as f64, 0.778, "⚠️ lm_head SUPPOSÉ (fichier projections seules)"),
        (w, c) => (w as f64, c as f64 * 2.0 / 1e9, "mesurés sur le fichier"),
    };
    println!(
        "\n  trafic par token — linéaires + embedding f16 {lm_head_gb:.3} Go ({head_src})"
    );
    println!("  {}", "-".repeat(72));
    let show = |name: &str, b: f64| {
        let gb = linear_w * b / 8.0 / 1e9 + lm_head_gb;
        println!(
            "  {name:<44}{b:>7.3} b/p{:>9.2} Go{:>9.0} tok/s",
            gb,
            400.0 / gb
        );
    };
    show(
        &format!("archive v1, {arch_bits} b/bloc (indécodable en fusé)"),
        arch_bits as f64 / 24.0,
    );
    show("variable masques + u16", bpw(all_mask + 16 * total, total));
    show("variable positionnel + u16", bpw(all_pos + 16 * total, total));
    show("groupé 32, masques", bpw(grouped.mask_bits, total));
    show("groupé 32, masques, stride octet", bpw(grouped.mask_bits_byte, total));
    show("groupé 32, positionnel", bpw(grouped.pos_bits, total));
    show("champ fixe au pire cas (masques, u32)", bpw(fixed_mask * total, total));
    show("champ fixe au pire cas (pos., u32)", bpw(fixed_pos * total, total));
    show("champ fixe 128", 128.0 / 24.0);
    if grouped.groups > 0 {
        show(
            "groupé 32, Slot32",
            bpw(grouped.slot_bits_byte, total),
        );
        show(
            "groupé 32, Slot32, plafond L ≤ 4 (majorant)",
            bpw(
                grouped.groups * (GROUP as u64 * slot_stride_bits(4, gain_bits) + 32),
                total,
            ),
        );
    }
    show("Planes14 — CE QUI TOURNE AUJOURD'HUI", bpw(p14_bits, total));
    show(
        &format!("Planes12x ({exc_pct:.2} % d'exceptions)"),
        bpw(p12_bits, total),
    );
    show("E1c14 — transposé, sans bourrage", bpw(e14_bits, total));
    show("E1c12 — transposé + overlay", bpw(e12_bits, total));
    let fp16_gb = linear_w * 2.0 / 1e9 + lm_head_gb;
    println!(
        "\n  plafond FP16 : ~{:.0} tok/s ; plafond embedding seul : ~{:.0} tok/s",
        400.0 / fp16_gb,
        400.0 / lm_head_gb
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    // The frozen bit offsets of the two records, walked field by field
    // below. Test-only: the accounting itself needs the strides, not the
    // geometry — which is exactly why the geometry has to be pinned here.
    use llvq_artifact::runtime::{
        PLANES12X_PAD_BIT, PLANES12X_SMASK_BIT, PLANES14_PAD_BIT, PLANES14_PLANE_BIT,
        PLANES14_SMASK_BIT,
    };

    /// Qwen3-4B, the published `leech1c12` file — the shape the CUDA bench
    /// measured on an L40S. Every number here is a *count*, taken either from
    /// the architecture (36 layers × 7 projections) or from this bench's own
    /// earlier run, never from a rate.
    mod qwen3_4b {
        /// `Σ d_out · d_in` over the 252 linear matrices.
        pub const WEIGHTS: u64 = 3_633_315_840;
        /// `Σ d_out · (d_in / 24)`.
        pub const BLOCKS: u64 = 150_681_600;
        /// `Σ d_out · (d_in mod 24)` — 16 columns on the 2560/4096 inputs,
        /// 8 on the 9728 one.
        pub const TAIL_WEIGHTS: u64 = 16_957_440;
        /// `Σ d_out`, i.e. the 1 105 920 rows the CUDA bench verifies.
        pub const ROWS: u64 = 1_105_920;
        /// Five-level blocks, from `docs/mesures/k1c-rtbits-2026-08-05.txt`
        /// and reproduced by the CUDA bench as its exception count.
        pub const EXCEPTIONS: u64 = 5_096_688;
    }

    /// The `q8` embedding rate, derived the way the binary derives it.
    ///
    /// It exists because a first version of these tests hard-coded `8.5` on
    /// both sides of the 8B check, so zeroing the group-scale term left the
    /// suite green — the §5 pattern exactly: an assertion that never
    /// exercises the parameter it is meant to cover.
    fn embed_q8_bpw() -> f64 {
        8.0 + EMBED_GROUP_BITS / EMBED_GROUP
    }

    /// `q8 g64` is int8 **plus** one f16 scale and one f16 bias per group of
    /// 64 values — 32 bits per 64 weights, i.e. half a bit each. Forgetting
    /// that half-bit is a 6 % error on the carried tables and 1 % on the 8B
    /// model figure, which is more than the margin the AWQ comparison has.
    #[test]
    fn q8_embedding_is_eight_bits_plus_its_group_scales() {
        assert_eq!(EMBED_GROUP, 64.0, "llvq_llm::fused::EMBED_GROUP");
        assert_eq!(EMBED_GROUP_BITS, 32.0, "f16 échelle + f16 biais");
        assert!(
            (embed_q8_bpw() - 8.5).abs() < 1e-12,
            "q8 g64 vaut 8,5 b/param, pas {}",
            embed_q8_bpw()
        );
        // The 4B's tied table: 388 956 160 values, 40 groups per row of
        // 2560, 4 bytes per group — the 413.3 MB `q8_device_bytes` announces.
        let (packed, scales) = (388_956_160u64, 151_936u64 * 40 * 4);
        assert!(
            ((packed + scales) as f64 * 8.0 / packed as f64 - embed_q8_bpw()).abs() < 1e-9,
            "le taux et les octets annoncés par le chemin q8 divergent"
        );
    }

    fn carried_4b() -> Shapes {
        Shapes {
            weights: qwen3_4b::WEIGHTS,
            tail_weights: qwen3_4b::TAIL_WEIGHTS,
            rows: qwen3_4b::ROWS,
        }
    }

    /// Walk one record's fields at the offsets the format froze, and check
    /// that the posts add up to the stride this bench bills.
    ///
    /// Asserting `112` alone would survive a format that moved a field; this
    /// walks `PLANES14_SMASK_BIT`, `PLANES14_PLANE_BIT` and
    /// `PLANES14_PAD_BIT` from `llvq_artifact::runtime` and demands they be
    /// consistent with `class 9 | gain 1 | signes 24 | 3 × plan 24 | 6 de
    /// bourrage`. Any field that moves, widens or disappears lands here.
    #[test]
    fn one_planes14_record_is_112_bits_field_by_field() {
        // Header: class then a one-bit gain — the field the two Planes
        // layouts freeze, and the reason they refuse a file with more.
        assert_eq!(PLANES14_SMASK_BIT, CLASS_BITS + 1, "classe 9 b + gain 1 b");
        // Sign mask in slot space: 24 bits, not `nonzero`.
        assert_eq!(PLANES14_PLANE_BIT[0], PLANES14_SMASK_BIT + DIM as u64);
        // Three contiguous 24-bit planes.
        assert_eq!(PLANES14_PLANE_BIT.len(), 3);
        for w in PLANES14_PLANE_BIT.windows(2) {
            assert_eq!(w[1] - w[0], DIM as u64, "un plan fait 24 bits");
        }
        let last = PLANES14_PLANE_BIT[PLANES14_PLANE_BIT.len() - 1];
        assert_eq!(PLANES14_PAD_BIT, last + DIM as u64);
        assert_eq!(PLANES14_PAD_BIT, 106, "9 + 1 + 24 + 72");
        // And the stride closes the record to a whole number of bytes.
        let stride = PLANES14_BYTES as u64 * 8;
        assert_eq!(stride - PLANES14_PAD_BIT, 6, "6 bits de bourrage");
        assert_eq!(planes14_bits(1), stride);
        // Class-blind by construction: a thousand blocks cost a thousand
        // strides whatever they hold, origin included.
        assert_eq!(planes14_bits(1_000), 112_000);
    }

    /// Same walk for `Planes12x`: `9 + 1 + 24 + 2·24 = 82` payload bits, 14
    /// of padding, `12·8 = 96`; one exception adds `4 + 14 = 18` bytes.
    #[test]
    fn one_planes12x_record_is_96_bits_and_one_exception_144_field_by_field() {
        assert_eq!(PLANES12X_SMASK_BIT, CLASS_BITS + 1, "classe 9 b + gain 1 b");
        assert_eq!(PLANES12X_PLANE_BIT[0], PLANES12X_SMASK_BIT + DIM as u64);
        // **Two** planes, not three — the whole difference with Planes14.
        assert_eq!(PLANES12X_PLANE_BIT.len(), PLANES14_PLANE_BIT.len() - 1);
        for w in PLANES12X_PLANE_BIT.windows(2) {
            assert_eq!(w[1] - w[0], DIM as u64);
        }
        let last = PLANES12X_PLANE_BIT[PLANES12X_PLANE_BIT.len() - 1];
        assert_eq!(PLANES12X_PAD_BIT, last + DIM as u64);
        assert_eq!(PLANES12X_PAD_BIT, 82, "9 + 1 + 24 + 48");
        let stride = PLANES12X_BYTES as u64 * 8;
        assert_eq!(stride - PLANES12X_PAD_BIT, 14, "14 bits de bourrage");
        // An exception is a u32 block index plus a full Planes14 record.
        assert_eq!(PLANES12X_EXC_BYTES as u64 * 8, 32 + PLANES14_BYTES as u64 * 8);
        assert_eq!(planes12x_bits(1, 0), 96);
        // The exception is an ADDITION: the main record is written anyway,
        // with a swapped L ≤ 4 direction in it.
        assert_eq!(planes12x_bits(1, 1), 96 + 144);
        assert_eq!(planes12x_bits(1_000, 10), 96_000 + 1_440);
    }

    /// The exception predicate, at its two boundaries, by both routes.
    /// `PLANES12X_LEVEL_CAP` is the last level two bit-planes can address, so
    /// 4 is in and 5 is out; the origin (one level) is never an exception.
    #[test]
    fn only_five_level_classes_are_exceptions() {
        assert!(!is_planes12x_exception(1), "l'origine n'est pas une exception");
        for l in 2..=PLANES12X_LEVEL_CAP {
            assert!(!is_planes12x_exception(l), "L = {l} tient en deux plans");
        }
        assert!(is_planes12x_exception(PLANES12X_LEVEL_CAP + 1));
        assert!(is_planes12x_exception(MAX_LEVELS));
        // The cap is not a free parameter: `k` planes address `2^k` levels.
        // The bench's run-time cross-check leans on this, so it is pinned
        // here rather than assumed there.
        assert_eq!(PLANES12X_LEVEL_CAP, 1 << PLANES12X_PLANE_BIT.len());
        for l in 1..=MAX_LEVELS {
            assert_eq!(
                is_planes12x_exception(l),
                exceeds_planes12x_geometry(l),
                "les deux routes divergent à L = {l}"
            );
        }
    }

    /// **The acceptance test.** Two instruments, one object, one number.
    ///
    /// The CUDA bench on an L40S
    /// (`docs/mesures/e2-golay70-bench-2026-08-07.txt`) reports 5.510 /
    /// 4.804 / 4.342 b/weight for `Slot32` / `Planes14` / `Planes12x` on the
    /// published Qwen3-4B. Those figures are *measured GPU uploads*; what
    /// follows rebuilds them from block counts alone. If this test ever
    /// fails, one of the two instruments has drifted and neither figure may
    /// be published until it is known which.
    #[test]
    fn published_4b_kernel_rates_are_reproduced() {
        let c = carried_4b();
        // The shape counts must close on themselves first: quantized weights
        // plus tail weights are all the weights there are.
        assert_eq!(qwen3_4b::BLOCKS * DIM as u64 + qwen3_4b::TAIL_WEIGHTS, c.weights);

        let p14 = c.kernel_bpw(planes14_bits(qwen3_4b::BLOCKS));
        let p12 = c.kernel_bpw(planes12x_bits(qwen3_4b::BLOCKS, qwen3_4b::EXCEPTIONS));
        assert!(
            (p14 - 4.804).abs() < 5e-4,
            "Planes14 : {p14:.4} contre 4.804 au banc CUDA"
        );
        assert!(
            (p12 - 4.342).abs() < 5e-4,
            "Planes12x : {p12:.4} contre 4.342 au banc CUDA"
        );
        // And the exception rate the arbitration rests on.
        let rate = 100.0 * qwen3_4b::EXCEPTIONS as f64 / qwen3_4b::BLOCKS as f64;
        assert!((rate - 3.3824).abs() < 5e-4, "taux d'exceptions {rate:.4} %");
    }

    /// **The E1c acceptance test.** Both variants priced from block counts
    /// alone, against the values `docs/spec-memoire-extreme-2026-08-12.md`
    /// predicted before a line of the layout existed (4.56 and 3.76 in the
    /// kernel accounting).
    ///
    /// The spec derived them by converting a payload rate; this rebuilds them
    /// from the real shapes of the published 4B. Two routes to one number is
    /// the point — the conversion `thesis = payload × 0.99534 + 0.1589` was
    /// exact to the thousandth on `Planes14` and wrong by 0.5 b/weight on
    /// `Golay70`, so it is a heuristic, not an instrument.
    #[test]
    fn e1c_rates_on_the_published_4b_match_the_spec() {
        let c = carried_4b();
        let e14 = c.kernel_bpw(e1c14_bits(qwen3_4b::BLOCKS));
        let e12 = c.kernel_bpw(e1c12_bits(qwen3_4b::BLOCKS, qwen3_4b::EXCEPTIONS));
        assert!((e14 - 4.5551).abs() < 5e-4, "E1c14 : {e14:.4} contre 4.5551");
        assert!((e12 - 3.7618).abs() < 5e-4, "E1c12 : {e12:.4} contre 3.7618");
        // And what each buys over the layout it replaces — the figure the
        // bench's admission criteria are stated against.
        let p14 = c.kernel_bpw(planes14_bits(qwen3_4b::BLOCKS));
        let p12 = c.kernel_bpw(planes12x_bits(qwen3_4b::BLOCKS, qwen3_4b::EXCEPTIONS));
        assert!((p14 - e14 - 0.2489).abs() < 1e-3, "E1c14 rend {:.4}", p14 - e14);
        assert!((p12 - e12 - 0.5806).abs() < 1e-3, "E1c12 rend {:.4}", p12 - e12);
    }

    /// The saving is **the padding, and only the padding** — stated as an
    /// identity rather than as a measured difference.
    ///
    /// A group of 32 blocks costs `group_words · 32` bits where the source
    /// costs `stride · 8 · 32`; the ratio of the two rates is therefore the
    /// ratio of payload bits to stride bits, exactly. If that ever stops
    /// holding, the transposition has started costing or saving something
    /// else and the arm is no longer a single-variable comparison.
    #[test]
    fn e1c_costs_exactly_the_payload_and_nothing_else() {
        const N: u64 = 32 * 1_000;
        // Planes14: 112 bits of stride, 106 of payload.
        assert_eq!(e1c14_bits(N) * (PLANES14_BYTES as u64 * 8), planes14_bits(N) * 106);
        // Planes12x main stream: 96 bits of stride, 82 of payload. The
        // exception term is identical on both sides, so it is excluded here
        // and checked separately below.
        let e12_main = e1c12_bits(N, 0);
        assert_eq!(e12_main * (PLANES12X_BYTES as u64 * 8), planes12x_bits(N, 0) * 82);
        // Exceptions: the same records, carried verbatim — so the difference
        // between the two layouts does not move when exceptions appear.
        let d0 = planes12x_bits(N, 0) - e1c12_bits(N, 0);
        let d9 = planes12x_bits(N, 9_999) - e1c12_bits(N, 9_999);
        assert_eq!(d0, d9, "le terme d'exceptions n'est pas identique des deux côtés");
    }

    /// A block count that is not a multiple of 32 pays a whole trailing
    /// group, and the bench must charge it — otherwise a small matrix would
    /// be priced below what it uploads.
    ///
    /// The published 4B happens to divide exactly (150 681 600 = 32 ×
    /// 4 708 800), which is precisely why this needs its own test: the
    /// acceptance test above would never exercise the rounding.
    #[test]
    fn a_partial_trailing_group_is_charged() {
        assert_eq!(qwen3_4b::BLOCKS % E1C_GROUP as u64, 0, "le 4B tombe juste");
        let full = e1c14_bits(32);
        assert_eq!(e1c14_bits(1), full, "un bloc paie un groupe entier");
        assert_eq!(e1c14_bits(33), 2 * full);
        assert_eq!(e1c14_bits(64), 2 * full);
        assert_eq!(e1c14_bits(65), 3 * full);
    }

    /// The two accountings must **not** agree — and the gap must be the two
    /// terms that separate them, nothing else. A test that only checked the
    /// kernel figure would pass on a build that quietly used one denominator
    /// for both.
    #[test]
    fn payload_and_kernel_accountings_differ_by_exactly_the_side_data() {
        let c = carried_4b();
        let bits = planes14_bits(qwen3_4b::BLOCKS);
        let payload = bits as f64 / (qwen3_4b::BLOCKS * DIM as u64) as f64;
        assert!(
            (payload - 14.0 * 8.0 / 24.0).abs() < 1e-12,
            "le payload de Planes14 est 112/24 exactement, pas {payload}"
        );
        // Same numerator, whole-matrix denominator: strictly smaller.
        assert!(bits as f64 / (c.weights as f64) < payload);
        // Side data restores it and overshoots — the 4.804 above.
        assert_eq!(
            c.side_bits(),
            (qwen3_4b::TAIL_WEIGHTS + qwen3_4b::ROWS) * 32,
            "queue f32 + échelles de ligne f32, rien d'autre"
        );
        assert!(c.kernel_bpw(bits) > payload);
    }

    /// `Shapes` must derive its three counts the way
    /// `planesbench` does, or every kernel figure is off by the tail.
    #[test]
    fn carried_counts_shapes_like_the_cuda_bench() {
        let mut c = Shapes::default();
        // Qwen3-4B `down_proj`: 9728 = 24·405 + 8.
        c.push(2560, 9728);
        assert_eq!(c.weights, 2560 * 9728);
        assert_eq!(c.tail_weights, 2560 * 8);
        assert_eq!(c.rows, 2560);
        // Qwen3-8B `down_proj`: 12288 = 24·512, no tail at all — the reason
        // the 8B's kernel rate sits below the 4B's at identical payload.
        let mut d = Shapes::default();
        d.push(4096, 12288);
        assert_eq!(d.tail_weights, 0);
        assert_eq!(d.side_bits(), 4096 * 32);
    }

    /// The whole-model conversion, pinned on the two published Qwen3-8B
    /// cells (`docs/archive/tableau-8b-2026-08-07.md` §2–3): 6.461 b/param with an
    /// f16 embedding and 5.323 with `q8`, both measured in the engine.
    ///
    /// This is the step the dossier got wrong twice, so it is checked
    /// against numbers produced by a third instrument entirely — the model
    /// loader's own VRAM report, not this bench and not the CUDA bench.
    #[test]
    fn whole_model_conversion_matches_the_published_8b_cells() {
        // Qwen3-8B: 36 layers, hidden 4096, 8 kv heads, intermediate 12288;
        // `tie_word_embeddings = false`, so two 151936×4096 tables.
        const LIN: u64 = 6_945_767_424;
        const EMBED: u64 = 2 * 151_936 * 4_096;
        // Planes14 on the 8B shapes: 288 571 392 blocks, 20 054 016 tail
        // weights, 1 400 832 rows.
        let c = Shapes {
            weights: LIN,
            tail_weights: 20_054_016,
            rows: 1_400_832,
        };
        assert_eq!((LIN - c.tail_weights) % DIM as u64, 0);
        let lin = c.kernel_bpw(planes14_bits((LIN - c.tail_weights) / DIM as u64));

        let f16 = model_bpw(&[(LIN, lin), (EMBED, 16.0)]);
        let q8 = model_bpw(&[(LIN, lin), (EMBED, embed_q8_bpw())]);
        assert!((f16 - 6.461).abs() < 1e-3, "embed f16 : {f16:.3} contre 6.461");
        assert!((q8 - 5.323).abs() < 2e-3, "embed q8 : {q8:.3} contre 5.323");
        // Sanity on the direction: carrying 15.2 % of the model in f16 costs
        // more per parameter than the linears do.
        assert!(f16 > lin && q8 > lin);
    }
}
