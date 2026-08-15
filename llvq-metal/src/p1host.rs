//! Host side of the two P1 arms: the tables `cascade_uniform.metal` and
//! `binomial_walk.metal` assume, and which nothing built until now.
//!
//! Both shaders are written against a `#[repr(C)]` mirror declared *in their
//! own comments*, and both list the invariants the host must assert because
//! MSL cannot (`cascade_uniform.metal` §5, `binomial_walk.metal` §3). This
//! module is the other half of those two contracts and nothing else — it
//! builds tables and it asserts. It dispatches nothing and it times nothing.
//!
//! ## Where every number comes from
//!
//! Nothing here is a second derivation. The cascade records are read straight
//! out of [`FastDecoder::cascade_class`], i.e. the very data
//! [`FastDecoder::decode`] runs on; the walk records are read out of
//! `FastDecoder::levels`, the same source [`crate::gpu_class_table`] uses; the
//! binomials come from [`llvq_search::rankdec::binom`], narrowed to `u32` with
//! a checked cast rather than re-derived. The only arithmetic invented here is
//! the reciprocal tables, and they are pinned against real division by the
//! tests at the bottom of this file.
//!
//! ## The two normalisations, which are NOT the same
//!
//! * [`GpuCascadeRec::inv_norm`] is `1/√(16m)` kept **apart** from the values:
//!   the cascade reads the class's *integer* |values| as bytes and applies the
//!   scale once, at the end. Feeding it pre-divided values would be wrong by
//!   `1/√(16m)` (`cascade_uniform.metal` §4).
//! * [`GpuWalkRec::vals`] are **already divided** by `√(16m)`, exactly as
//!   `gpu_class_table` does. `decode_walk` never divides by a norm
//!   (`binomial_walk.metal` §8).
//!
//! A reference that mixed the two conventions would read as a decode
//! divergence rather than as the scale bug it is, which is why they are stated
//! here and asserted where they can be.

use llvq_core::{Golay, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::{binom, walk_radices};

/// Classes of the cap-13 ball — `FastDecoder::n_classes()`, restated so a
/// codebook that changed size fails here rather than inside a kernel.
pub const N_CLASSES: usize = 383;

/// Padded length of the cascade's `ends` table. The shader's class search is
/// nine unconditional steps over it (`CLASS_TOP = 256`), so every probe must
/// be in range and every pad entry must compare "not below" — hence
/// [`u64::MAX`] rather than a repeat of the last end.
pub const CLASS_PAD: usize = 512;

/// Entries of the walk table: the origin, then one per class — the id
/// convention of `llvq_artifact::runtime::ClassTable` and of
/// [`crate::gpu_class_table`].
pub const WALK_ENTRIES: usize = N_CLASSES + 1;

/// Row stride of the binomial table the walk reads.
pub const BINOM_STRIDE: usize = DIM + 1;

/// Bits the walk's record spends before the first rank field:
/// `[class 9][gain 1][smask 24]`.
pub const WALK_HEADER_BITS: u32 = 9 + 1 + DIM as u32;

/// Total width of the walk's record. Three aligned `u32`, fixed stride — the
/// addressing of `decreal`'s `sol` and `Fixed96` anchors, deliberately, so the
/// arm cannot be priced against them while paying a different addressing bill.
pub const WALK_RECORD_BITS: u32 = 96;

// ---------------------------------------------------------------------------
// Records — byte-for-byte mirrors of the shaders' structs
// ---------------------------------------------------------------------------

/// Mirror of `cascade_uniform.metal`'s `struct CascadeRec`, 88 bytes.
///
/// **A class is either odd or even and the two sides give the same fields
/// different meanings**, which is why the names are neutral: `d_on` is `a_on`
/// on the even side and `m_arr` on the odd one, `d_off` is `a_off` on the even
/// side and `1` on the odd one. Each is simultaneously a radix of the
/// mixed-radix peel and the `m0` of an arrangement problem — the same number,
/// so neither is stored twice.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuCascadeRec {
    /// `local rank = (idx - 1) - offset`.
    pub offset: u64,
    /// `a_on` (even) / `m_arr` (odd) — problem A's `m0`, and the last radix.
    pub d_on: u64,
    /// `a_off` (even) / `1` (odd) — problem B's `m0`, and the first radix.
    pub d_off: u64,
    /// Granlund–Montgomery multiplier for `d_on`; see [`div_magic`].
    pub magic_on: u64,
    /// Ditto for `d_off`.
    pub magic_off: u64,
    /// 8 kinds × 8 bits: `word_counts` (even) / `counts` (odd).
    pub counts_a: u64,
    /// 8 kinds × 8 bits: `off_counts`, zero run included (even); 0 (odd).
    pub counts_b: u64,
    /// 8 kinds × 8 bits: |value| per kind, 0 for unused kinds.
    pub vals_a: u64,
    /// Ditto for problem B; the zero run's byte is 0, which is what marks it.
    pub vals_b: u64,
    /// `1 / sqrt(16 m)` — applied **once**, at the end, with the gain centroid.
    pub inv_norm: f32,
    /// Offset into the 4096-word Golay table: 0 (odd), `offsets[w]` (even).
    pub golay_base: u32,
    /// `w | p_req<<8 | sf_shift<<16 | sw_shift<<24`.
    pub geom: u32,
    /// `shift_off | shift_on<<8 | one_off<<16 | one_on<<17 | odd<<18`.
    pub dflags: u32,
}

/// Mirror of `binomial_walk.metal`'s `struct WalkRec`, 52 bytes — the shape
/// that file spells out in its own comment (§3), followed here rather than
/// re-invented.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuWalkRec {
    /// Level value per kind, **already divided by** `sqrt(16·m)`.
    pub vals: [f32; MAX_KINDS],
    /// Slots this kind takes; sums to 24 over `k`.
    pub counts: [u8; MAX_KINDS],
    /// `ceil(log2 radix_j)`; 0 for the final kind, whose radix is 1.
    pub wbits: [u8; MAX_KINDS],
    /// Kinds in use, ≥ 1.
    pub k: u8,
    /// Explicit padding — the shader declares it, so the mirror declares it.
    pub pad: [u8; 3],
}

/// Mirror of `cascade_uniform.metal`'s `struct DivTab`, 600 bytes.
///
/// Indexed by the **stage** — the number of slots left, 24 down to 1 — never
/// by the class. The cascade's divisor at a given step is that count whatever
/// the class, so a per-(class, stage) table would hold 383 identical copies of
/// the same 24 entries (`rankdec.rs:52-55`). Three of the four arrays serve
/// three *different* division paths, selected at compile time by
/// `LLVQ_CASCADE_DIV`, and only the first is the pre-registered arm:
///
/// * `magic64` — the default: `mulhi64(x, magic64[n])` is `x / n` for every
///   `x` the cascade can hold. Wrong nowhere.
/// * `modinv` + `tz` — exact division by a modular inverse, valid only because
///   `n` divides `m·c`. Cheaper, and silently wrong the day that breaks.
/// * `magic32` — the q/r split, which moves the 64-bit divide out of the
///   candidate loop.
///
/// All four are built here, where the identities are unit-testable, so that no
/// constant in the shader was written by hand.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuDivTab {
    /// `floor(2^64 / n) + 1` for `n >= 2`; entries 0 and 1 are zero.
    pub magic64: [u64; BINOM_STRIDE],
    /// `(n >> ctz(n))^-1 mod 2^64` — the odd part's inverse.
    pub modinv: [u64; BINOM_STRIDE],
    /// `floor(2^32 / n) + 1` for `n >= 2`; entries 0 and 1 are zero.
    pub magic32: [u32; BINOM_STRIDE],
    /// `ctz(n)`; entry 0 is zero, `ctz(0)` having no value.
    pub tz: [u32; BINOM_STRIDE],
}

// A mirror that drifted from the shader must not compile. `size_of` is a const
// fn and `assert!` works in const context, so these are checked at build time,
// on every build, without a dependency.
const _: () = assert!(std::mem::size_of::<GpuCascadeRec>() == 88);
const _: () = assert!(std::mem::align_of::<GpuCascadeRec>() == 8);
const _: () = assert!(std::mem::size_of::<GpuWalkRec>() == 52);
const _: () = assert!(std::mem::align_of::<GpuWalkRec>() == 4);
const _: () = assert!(std::mem::size_of::<GpuDivTab>() == 600);
const _: () = assert!(std::mem::align_of::<GpuDivTab>() == 8);
// The shader dimensions its kind fields on eight bytes of one `ulong`; a
// wider `MAX_KINDS` would silently drop kinds off the top.
const _: () = assert!(MAX_KINDS == 8);
const _: () = assert!(DIM == 24);

// ---------------------------------------------------------------------------
// Reciprocals
// ---------------------------------------------------------------------------

/// `ceil(log2 x)` for `x >= 1`.
#[inline]
fn ceil_log2(x: u64) -> u32 {
    assert!(x >= 1, "log2 of zero");
    if x == 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}

/// The per-class reciprocal `div_big` consumes: `(magic, shift, one)` with
/// `magic = floor(2^(64+b)/d) + 1 - 2^64`, `b = ceil(log2 d)`, `shift = b - 1`.
///
/// `d == 1` has no `b`, so the shader takes the quotient straight from `x`
/// when `one` is set; the other two fields are then meaningless and are zero
/// here rather than left to a shift by `-1`.
///
/// This is the **only** reciprocal in the cascade that has to be per class:
/// the peel's radices are the class's arrangement counts.
pub fn div_magic(d: u64) -> (u64, u32, bool) {
    assert!(d != 0, "a radix of zero is not a divisor");
    if d == 1 {
        return (0, 0, true);
    }
    let b = ceil_log2(d);
    assert!(b <= 63, "divisor {d} is too wide for a 64-bit magic");
    let m = ((1u128 << (64 + b)) / d as u128) + 1 - (1u128 << 64);
    (m as u64, b - 1, false)
}

/// Inverse of an odd `a` modulo `2^64`, by Newton iteration.
///
/// `x0 = a` is already correct to three bits (`a·a ≡ 1 mod 8` for odd `a`) and
/// each step doubles that, so five suffice for 64; six are cheap and leave no
/// arithmetic to argue about.
fn inv_odd(a: u64) -> u64 {
    assert!(a & 1 == 1, "only an odd number is invertible mod 2^64");
    let mut x = a;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
    }
    x
}

/// The stage-indexed reciprocal table, all four arrays.
pub fn div_table() -> GpuDivTab {
    let mut t = GpuDivTab {
        magic64: [0; BINOM_STRIDE],
        modinv: [0; BINOM_STRIDE],
        magic32: [0; BINOM_STRIDE],
        tz: [0; BINOM_STRIDE],
    };
    for n in 1..BINOM_STRIDE {
        let nn = n as u64;
        if n >= 2 {
            t.magic64[n] = (((1u128 << 64) / nn as u128) + 1) as u64;
            t.magic32[n] = (((1u64 << 32) / nn) + 1) as u32;
        }
        t.tz[n] = nn.trailing_zeros();
        t.modinv[n] = inv_odd(nn >> nn.trailing_zeros());
    }
    t
}

// ---------------------------------------------------------------------------
// The cascade's tables
// ---------------------------------------------------------------------------

/// Cumulative class ends, padded to [`CLASS_PAD`] with [`u64::MAX`].
///
/// The shader's search reads `ends[0..CLASS_PAD-1]` unconditionally, so the
/// padding is not decoration: an entry that could compare "at or below" would
/// walk the class index past the table.
pub fn cascade_ends(fd: &FastDecoder) -> Vec<u64> {
    assert_eq!(
        fd.n_classes(),
        N_CLASSES,
        "the cascade shader is dimensioned on {N_CLASSES} classes"
    );
    let mut ends = vec![u64::MAX; CLASS_PAD];
    for (ci, e) in ends.iter_mut().enumerate().take(N_CLASSES) {
        *e = fd.class_range(ci).1;
    }

    // 🚨 `ends` and `CascadeRec::offset` come from two DIFFERENT accessors —
    // `class_range` and `cascade_class` — and the shader's class search only
    // decodes correctly if they agree. Strict growth and padding, which the
    // table test already checks, are true of any shifted table: a mutation
    // adding one to every end survives that test AND V0, and silently sends
    // every block near a class boundary to its neighbour.
    //
    // These three identities tie the two together, and they cost nothing.
    for ci in 0..N_CLASSES {
        let rec_off = fd.cascade_class(ci).offset;
        let (first, last) = fd.class_range(ci);
        assert_eq!(
            first - 1,
            rec_off,
            "class {ci}: class_range and cascade_class disagree on where the \
             class starts — the shader would decode across a boundary"
        );
        if ci > 0 {
            assert_eq!(
                ends[ci - 1],
                rec_off,
                "class {ci}: its offset is not where the previous class ends"
            );
        }
        assert_eq!(ends[ci], last, "class {ci}: end mismatch");
    }
    assert_eq!(
        ends[N_CLASSES - 1],
        llvq_search::index::N13,
        "the last class must end on N13, or the table covers the wrong ball"
    );
    ends
}

/// Start of each Hamming weight's bucket in [`Golay::codewords`].
///
/// `of_weight(w)` is a slice of that flat table; the shader is handed the flat
/// table and an offset, so the offset is recovered here rather than assumed.
fn golay_offsets(golay: &Golay) -> [u32; 25] {
    let flat = golay.codewords();
    let mut off = [0u32; 25];
    for (w, o) in off.iter_mut().enumerate() {
        *o = flat
            .iter()
            .position(|c| c.count_ones() as usize == w)
            .map_or(0, |p| p as u32);
    }
    off
}

/// Pack up to eight bytes, LSB-first, the way the shader reads them back with
/// `(word >> (8*j)) & 0xff`.
fn pack_bytes(src: &[u8]) -> u64 {
    assert!(src.len() <= MAX_KINDS);
    let mut w = 0u64;
    for (j, &b) in src.iter().enumerate() {
        w |= (b as u64) << (8 * j);
    }
    w
}

/// A class's |values| as bytes, zero beyond `n`.
///
/// `‖x‖² = 16m ≤ 208` caps every coordinate at 14, so a byte is not a gamble;
/// the assertion is here because that is a fact about the cap-13 ball and not
/// about this function.
fn pack_vals(vals: &[i32], n: usize) -> u64 {
    let mut w = 0u64;
    for (j, &v) in vals.iter().enumerate().take(n) {
        assert!((0..=255).contains(&v), "|value| {v} does not fit a byte");
        w |= (v as u64) << (8 * j);
    }
    w
}

/// The 383 cascade records, in class order — one per entry of the shader's
/// `classes` buffer.
///
/// Everything is read from [`FastDecoder::cascade_class`], so there is no
/// second derivation of the layout to drift from.
pub fn cascade_records(fd: &FastDecoder, golay: &Golay) -> Vec<GpuCascadeRec> {
    assert_eq!(fd.n_classes(), N_CLASSES);
    let goff = golay_offsets(golay);
    let mut out = Vec::with_capacity(N_CLASSES);

    for ci in 0..N_CLASSES {
        let c = fd.cascade_class(ci);
        let shell = fd.levels(ci).shell;
        assert!(shell > 0, "class {ci} has no shell");
        let inv_norm = (1.0 / ((16 * shell) as f64).sqrt()) as f32;

        let (d_on, d_off, counts_a, counts_b, vals_a, vals_b, golay_base, geom);
        if c.odd {
            // One 24-slot problem. `d_off = 1` makes the first peel a no-op,
            // so `r_on = local % m_arr` and `gi = local / m_arr` fall out
            // without a branch.
            d_on = c.m_arr;
            d_off = 1;
            counts_a = pack_bytes(&c.counts[..c.n_kinds]);
            counts_b = 0;
            vals_a = pack_vals(&c.vals, c.n_kinds);
            vals_b = 0;
            golay_base = 0;
            // w = 24 so the shader's state swap at `s == w` never fires.
            geom = DIM as u32;
        } else {
            // Two problems: the support (w slots) and its complement.
            d_on = c.a_on;
            d_off = c.a_off;
            counts_a = pack_bytes(&c.word_counts[..c.n_word]);
            counts_b = pack_bytes(&c.off_counts[..c.n_off]);
            vals_a = pack_vals(&c.word_vals, c.n_word);
            // The off-support kinds are `free_vals`, then the zero run — whose
            // value byte is 0, and that zero is what marks it.
            vals_b = pack_vals(&c.free_vals, c.n_off.saturating_sub(1));
            golay_base = goff[c.w as usize];
            assert!(c.s_w.is_power_of_two() && c.s_f.is_power_of_two());
            let sf_shift = c.s_f.trailing_zeros();
            let sw_shift = c.s_w.trailing_zeros();
            assert!(c.p_req <= 1);
            geom = c.w | (c.p_req << 8) | (sf_shift << 16) | (sw_shift << 24);
        }

        // Unused kinds must carry a zero count AND a zero value byte
        // (`cascade_uniform.metal` §5.4). `pack_bytes`/`pack_vals` only ever
        // write the kinds in use, so this is a check on the *source* record —
        // a class whose tables held a stale byte past `n_kinds` would place a
        // phantom slot, and the kernel has no way to notice.
        let kinds_a = if c.odd { c.n_kinds } else { c.n_word };
        let unused_a = 8 * kinds_a as u32;
        assert!(
            unused_a >= 64 || (counts_a >> unused_a == 0 && vals_a >> unused_a == 0),
            "class {ci}: a kind past {kinds_a} is not empty"
        );

        let (magic_on, shift_on, one_on) = div_magic(d_on);
        let (magic_off, shift_off, one_off) = div_magic(d_off);
        let dflags = shift_off
            | (shift_on << 8)
            | ((one_off as u32) << 16)
            | ((one_on as u32) << 17)
            | ((c.odd as u32) << 18);

        out.push(GpuCascadeRec {
            offset: c.offset,
            d_on,
            d_off,
            magic_on,
            magic_off,
            counts_a,
            counts_b,
            vals_a,
            vals_b,
            inv_norm,
            golay_base,
            geom,
            dflags,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// The walk's tables
// ---------------------------------------------------------------------------

/// `C(p, c)` **transposed**: `bt[c * 25 + p]`, zero when `c > p`.
///
/// The transpose is what lets the shader hold `c` fixed and walk `p` downward
/// at a stride of one word, decrementing the row pointer by 25 when a unit is
/// placed. The zeros for `c > p` are load-bearing, not a guard: they are how a
/// kind that must take every remaining position gets placed.
///
/// Narrowed from the crate's one `u64` table with a checked cast rather than
/// re-derived — the largest entry is `C(24,12) = 2 704 156`.
pub fn binom_table() -> Vec<u32> {
    let mut t = vec![0u32; BINOM_STRIDE * BINOM_STRIDE];
    for c in 0..BINOM_STRIDE {
        for p in 0..BINOM_STRIDE {
            t[c * BINOM_STRIDE + p] =
                u32::try_from(binom(p, c)).expect("C(p,c) fits u32 for p <= 24");
        }
    }
    t
}

/// The 384 walk records: entry 0 the origin, entry `1 + ci` class `ci`.
///
/// One 24-slot walk per class, over the class's magnitude levels — which is
/// what `binomial_walk` is, and what the shader decodes. How an *even* class
/// composes in the archive (two unrankings against a Golay codeword plus a
/// parity repair) is a different question, and `binomial_walk.metal` §9 marks
/// it ABSENT rather than pretending this table answers it.
pub fn walk_records(fd: &FastDecoder) -> Vec<GpuWalkRec> {
    assert_eq!(fd.n_classes(), N_CLASSES);
    let mut out = Vec::with_capacity(WALK_ENTRIES);

    // The origin: one kind, 24 slots, value zero. The outer loop is empty and
    // the deposit walk multiplies 24 zeros — a dot of 0, which is what the
    // origin means. No real block reaches it (the published 4B holds none),
    // which is exactly why it is in the table and not left to chance.
    out.push(GpuWalkRec {
        vals: [0.0; MAX_KINDS],
        counts: {
            let mut c = [0u8; MAX_KINDS];
            c[0] = DIM as u8;
            c
        },
        wbits: [0; MAX_KINDS],
        k: 1,
        pad: [0; 3],
    });

    for ci in 0..N_CLASSES {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let k = lv.len;
        assert!((1..=MAX_KINDS).contains(&k), "class {ci} has {k} levels");

        let mut counts = [0u8; MAX_KINDS];
        let mut vals = [0.0f32; MAX_KINDS];
        for j in 0..k {
            counts[j] = lv.counts[j];
            vals[j] = (lv.values[j] as f64 / norm) as f32;
        }
        assert_eq!(
            counts.iter().map(|&c| c as usize).sum::<usize>(),
            DIM,
            "class {ci}: the counts must fill exactly the 24 slots"
        );

        let radices = walk_radices(&counts, k, DIM);
        assert_eq!(
            radices[k - 1],
            1,
            "class {ci}: the last kind must have radix 1"
        );
        let mut wbits = [0u8; MAX_KINDS];
        let mut spent = WALK_HEADER_BITS;
        for j in 0..k {
            let w = ceil_log2(radices[j]);
            assert!(w <= 32, "class {ci}: rank field {j} is {w} bits wide");
            wbits[j] = w as u8;
            spent += w;
        }
        // The contract `binomial_walk.metal` §3 cannot check for itself, and
        // which it warns fits with ZERO SLACK: a second gain bit would
        // overflow the record by exactly one bit on the worst class.
        assert!(
            spent <= WALK_RECORD_BITS,
            "class {ci}: the record needs {spent} bits, the shader reads {WALK_RECORD_BITS}"
        );

        out.push(GpuWalkRec {
            vals,
            counts,
            wbits,
            k: k as u8,
            pad: [0; 3],
        });
    }
    out
}

/// Radices of every walk record, in the same order — what a feed draws its
/// ranks from.
pub fn walk_radix_table(recs: &[GpuWalkRec]) -> Vec<[u64; MAX_KINDS]> {
    recs.iter()
        .map(|r| walk_radices(&r.counts, r.k as usize, DIM))
        .collect()
}

/// Pack one walk record's 96 bits: `[class 9][gain 1][smask 24][ranks…]`,
/// LSB-first, into the three little-endian words the shader loads at
/// `words[3*gid ..]`.
///
/// `ranks[j]` must be below `radices[j]`; a rank outside its radix is
/// malformed input, and the shader would decode a wrong answer rather than
/// fault.
pub fn pack_walk_block(
    rec: &GpuWalkRec,
    id: u32,
    gain: u32,
    smask: u32,
    ranks: &[u64; MAX_KINDS],
) -> [u32; 3] {
    assert!(id < 512, "the class field is 9 bits");
    assert!(
        gain < 2,
        "one gain bit is hardcoded in every MSL of this repo"
    );
    assert!(smask < (1 << DIM), "the sign mask is 24 bits");

    let mut acc: u128 = 0;
    let mut pos: u32 = 0;
    let mut push = |v: u64, w: u32| {
        assert!(w == 0 || v < (1u64 << w), "a field overflows its width");
        acc |= (v as u128) << pos;
        pos += w;
    };
    push(id as u64, 9);
    push(gain as u64, 1);
    push(smask as u64, DIM as u32);
    // The final kind reads no rank at all: its radix is 1, so storing one
    // would be storing a constant.
    for (j, &rank) in ranks
        .iter()
        .enumerate()
        .take((rec.k as usize).saturating_sub(1))
    {
        push(rank, rec.wbits[j] as u32);
    }
    assert!(pos <= WALK_RECORD_BITS, "the record overflows 96 bits");
    [acc as u32, (acc >> 32) as u32, (acc >> 64) as u32]
}

// ---------------------------------------------------------------------------
// V0's etalons — shared by `bin/p1v0` and `bin/rankbench`
// ---------------------------------------------------------------------------
//
// These live in the library and not in a bin for one reason: the bench has to
// verify every arm before timing it (pre-registration §3, §7), and a bench that
// carried its own copy of the etalons would be checking its copy. Two bins, one
// yardstick.

/// Counts, level values in f64, and kind count — one entry per walk-table id,
/// the origin first. Built by [`walk_levels`] from [`FastDecoder::levels`],
/// **never** read back out of [`GpuWalkRec`].
pub type Levels = Vec<([u8; MAX_KINDS], [f64; MAX_KINDS], usize)>;

/// The etalon's own reading of the codebook.
///
/// 🚨 **This is not redundancy, it is the point.** The first version of `p1v0`
/// built the walk's etalon from `rec.counts` and `rec.vals`, i.e. from the very
/// table it was supposed to check, and a mutation that gave every shell-13
/// class the norm of shell 12 **survived**: the shader read the wrong value, the
/// CPU reference read the same wrong value, and the two agreed to nine
/// decimals. That is the "malentendu partagé" §3.1 of the pre-registration asks
/// the reference to be written independently of the shader to avoid.
///
/// Entry 0 is the origin — one kind, 24 slots, value zero — matching
/// [`walk_records`]'s convention, restated rather than imported.
pub fn walk_levels(fd: &FastDecoder) -> Levels {
    let mut out = Vec::with_capacity(fd.n_classes() + 1);
    let mut origin = [0u8; MAX_KINDS];
    origin[0] = DIM as u8;
    out.push((origin, [0.0f64; MAX_KINDS], 1));
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let (mut counts, mut vals) = ([0u8; MAX_KINDS], [0.0f64; MAX_KINDS]);
        for j in 0..lv.len {
            counts[j] = lv.counts[j];
            vals[j] = lv.values[j] as f64 / norm;
        }
        out.push((counts, vals, lv.len));
    }
    out
}

/// What the host packed into the walk's records — and therefore exactly what
/// the GPU has to give back.
#[derive(Default)]
pub struct WalkFeed {
    pub ids: Vec<u32>,
    pub gains: Vec<u8>,
    pub ranks: Vec<[u64; MAX_KINDS]>,
    pub smasks: Vec<u32>,
}

impl WalkFeed {
    pub fn push(&mut self, id: u32, gain: u8, ranks: [u64; MAX_KINDS], smask: u32) {
        self.ids.push(id);
        self.gains.push(gain);
        self.ranks.push(ranks);
        self.smasks.push(smask);
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Pad with the origin at rank zero, to a whole number of `group` blocks.
    ///
    /// The origin is the one entry whose dot is zero whatever the sign mask, so
    /// the padding cannot flatter the worst error of a real entry.
    pub fn pad(&mut self, group: usize) {
        while !self.len().is_multiple_of(group) {
            self.push(0, 0, [0u64; MAX_KINDS], 0);
        }
    }
}

/// `(Σ wᵢxᵢ, Σ|wᵢxᵢ|)` for one arrangement — the dot, and the magnitude the
/// accumulation ran at, which is what a tolerance on a dot product has to be
/// relative to.
pub fn dot_of(
    kinds: &[u8; DIM],
    vals: &[f64; MAX_KINDS],
    smask: u32,
    gain: f64,
    x: &[f32],
) -> (f64, f64) {
    let (mut dot, mut mag) = (0.0f64, 0.0f64);
    for (i, &kd) in kinds.iter().enumerate() {
        let v = vals[kd as usize];
        let v = if (smask >> i) & 1 == 1 { -v } else { v };
        let t = v * gain * x[i] as f64;
        dot += t;
        mag += t.abs();
    }
    (dot, mag)
}

/// Pack the walk's feed and compute its etalon in one pass — the CPU walk on
/// the very ranks the GPU is handed (amendment É2).
///
/// Deliberately one function: a feed built in one place and an etalon built in
/// another can drift, and the drift would read as a decode divergence. The
/// *packing* reads `walk` — it is the record's own field widths that are under
/// test — while the *etalon* reads `lev`, which owes it nothing.
pub fn walk_feed(
    walk: &[GpuWalkRec],
    lev: &Levels,
    feed: &WalkFeed,
    gscale: &[f32; 2],
    x: &[f32],
) -> (Vec<u32>, Vec<(f64, f64)>) {
    assert_eq!(walk.len(), lev.len());
    let n = feed.len();
    let mut words = Vec::with_capacity(3 * n);
    let mut want = Vec::with_capacity(n);
    let mut kinds = [0u8; DIM];
    for b in 0..n {
        let id = feed.ids[b] as usize;
        words.extend_from_slice(&pack_walk_block(
            &walk[id],
            feed.ids[b],
            feed.gains[b] as u32,
            feed.smasks[b],
            &feed.ranks[b],
        ));
        let (counts, vals, k) = &lev[id];
        llvq_search::rankdec::binomial_walk(&feed.ranks[b], counts, *k, &mut kinds);
        want.push(dot_of(
            &kinds,
            vals,
            feed.smasks[b],
            gscale[feed.gains[b] as usize] as f64,
            x,
        ));
    }
    (words, want)
}

/// The archive decoder's own point, dotted in f64 — the etalon of **both**
/// cascade arms, which decode the archive's order and therefore owe it exact
/// equality.
///
/// [`llvq_core::Leech::shell_index`] recomputes ‖p‖²/16 from the decoded point,
/// so a wrong `inv_norm` in the host table cannot cancel against itself here.
/// The origin has no shell, and its dot is zero rather than an error:
/// `decode(0)` is a legitimate point of the codebook and the §1.6 fixture feeds
/// it on purpose.
pub fn etalon_cascade(
    fd: &FastDecoder,
    idx: &[u64],
    gains: &[u8],
    gscale: &[f32; 2],
    x: &[f32],
) -> Result<Vec<(f64, f64)>, String> {
    let mut want = Vec::with_capacity(idx.len());
    for (b, &i) in idx.iter().enumerate() {
        let p = fd
            .decode(i)
            .ok_or_else(|| format!("index {i} hors de la boule"))?;
        let (mut dot, mut mag) = (0.0f64, 0.0f64);
        if let Some(m) = llvq_core::Leech::shell_index(&p).filter(|&m| m > 0) {
            let s = gscale[gains[b] as usize] as f64 / ((16 * m) as f64).sqrt();
            for (&v, &xi) in p.iter().zip(x) {
                let t = v as f64 * s * xi as f64;
                dot += t;
                mag += t.abs();
            }
        }
        want.push((dot, mag));
    }
    Ok(want)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_records_are_the_size_the_shaders_declare() {
        assert_eq!(std::mem::size_of::<GpuCascadeRec>(), 88);
        assert_eq!(std::mem::size_of::<GpuWalkRec>(), 52);
        assert_eq!(std::mem::size_of::<GpuDivTab>(), 600);
    }

    /// `div_big`'s arithmetic, run against real division on the divisors the
    /// cascade can actually receive — every class's `d_on` and `d_off`, plus
    /// the edges around each.
    #[test]
    fn the_per_class_magic_divides() {
        fn quotient(x: u64, magic: u64, shift: u32, one: bool) -> u64 {
            if one {
                return x;
            }
            let q = ((x as u128 * magic as u128) >> 64) as u64;
            let t = ((x - q) >> 1) + q;
            t >> shift
        }
        let fd = FastDecoder::new();
        let mut rng = llvq_core::SplitMix64::new(0x0_D1_D0);
        for ci in 0..fd.n_classes() {
            let c = fd.cascade_class(ci);
            let ds = if c.odd {
                [c.m_arr, 1]
            } else {
                [c.a_on, c.a_off]
            };
            for d in ds {
                let (magic, shift, one) = div_magic(d);
                let mut probes = vec![0u64, 1, d - 1, d, d + 1, d * 2 - 1, u64::MAX >> 16];
                probes.extend((0..64).map(|_| rng.next() >> 16));
                for x in probes {
                    assert_eq!(quotient(x, magic, shift, one), x / d, "x={x} d={d}");
                }
            }
        }
    }

    /// The stage table's three paths, on the range the cascade uses them over:
    /// `n` in 1..=24 and `x = m·c` bounded by the largest arrangement count.
    #[test]
    fn the_stage_table_divides() {
        let t = div_table();
        let mut rng = llvq_core::SplitMix64::new(0x0_57A6E);
        for n in 1..=DIM {
            let nn = n as u64;
            // path 0 — the round-up magic, the pre-registered arm
            for _ in 0..20_000 {
                let x = rng.next() >> 24; // < 2^40, well inside m·c < 2^37
                let q = if n == 1 {
                    x
                } else {
                    ((x as u128 * t.magic64[n] as u128) >> 64) as u64
                };
                assert_eq!(q, x / nn, "magic64 n={n} x={x}");
            }
            // path 1 — the modular inverse, exact only where n divides x
            for _ in 0..20_000 {
                let q0 = rng.next() >> 30;
                let x = q0 * nn;
                let q = (x >> t.tz[n]).wrapping_mul(t.modinv[n]);
                assert_eq!(q, x / nn, "modinv n={n} x={x}");
            }
            // path 2 — the 32-bit magic, used on r·c < 24·24 = 576
            for x in 0..1024u64 {
                let q = if n == 1 {
                    x
                } else {
                    (x * t.magic32[n] as u64) >> 32
                };
                assert_eq!(q, x / nn, "magic32 n={n} x={x}");
            }
        }
    }

    /// The binomial table is the crate's own, transposed — not a second
    /// derivation.
    #[test]
    fn the_binomial_table_is_the_transpose() {
        let t = binom_table();
        for c in 0..BINOM_STRIDE {
            for p in 0..BINOM_STRIDE {
                assert_eq!(t[c * BINOM_STRIDE + p] as u64, binom(p, c), "c={c} p={p}");
            }
        }
    }

    /// Every invariant the two shaders say the host must assert, on the real
    /// codebook rather than on a fixture.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "builds both full tables, run in release")]
    fn the_tables_satisfy_the_shaders_contracts() {
        let fd = FastDecoder::new();
        let golay = Golay::new();

        let ends = cascade_ends(&fd);
        assert_eq!(ends.len(), CLASS_PAD);
        assert!(
            ends[..N_CLASSES].windows(2).all(|w| w[0] < w[1]),
            "ends must be strictly increasing"
        );
        assert!(
            ends[N_CLASSES..].iter().all(|&e| e == u64::MAX),
            "the pad must never compare below an index"
        );

        let recs = cascade_records(&fd, &golay);
        assert_eq!(recs.len(), N_CLASSES);
        let flat = golay.codewords();
        for (ci, r) in recs.iter().enumerate() {
            let c = fd.cascade_class(ci);
            assert_eq!(r.offset, c.offset);
            let odd = (r.dflags >> 18) & 1 == 1;
            assert_eq!(odd, c.odd);
            let w = (r.geom & 0xff) as usize;
            assert_eq!(w, if c.odd { DIM } else { c.w as usize });
            // The Golay base must land on the first codeword of that weight.
            if !c.odd {
                assert_eq!(
                    flat[r.golay_base as usize],
                    golay.of_weight(c.w as usize)[0]
                );
            } else {
                assert_eq!(r.golay_base, 0);
            }
            // Unused kinds carry a zero count AND a zero value.
            let kinds_a = if c.odd { c.n_kinds } else { c.n_word };
            for j in kinds_a..MAX_KINDS {
                assert_eq!((r.counts_a >> (8 * j)) & 0xff, 0, "class {ci} kind {j}");
                assert_eq!((r.vals_a >> (8 * j)) & 0xff, 0, "class {ci} kind {j}");
            }
            let kinds_b = if c.odd { 0 } else { c.n_off };
            for j in kinds_b..MAX_KINDS {
                assert_eq!((r.counts_b >> (8 * j)) & 0xff, 0, "class {ci} kind {j}");
            }
            // The zero run's value byte is what marks it.
            if !c.odd {
                assert_eq!((r.vals_b >> (8 * (c.n_off - 1))) & 0xff, 0, "class {ci}");
            }
        }

        let walk = walk_records(&fd);
        assert_eq!(walk.len(), WALK_ENTRIES);
        for (id, r) in walk.iter().enumerate() {
            assert_eq!(
                r.counts.iter().map(|&c| c as usize).sum::<usize>(),
                DIM,
                "walk entry {id}"
            );
            assert_eq!(
                r.wbits[r.k as usize - 1],
                0,
                "walk entry {id}: the last kind reads no rank"
            );
            let spent: u32 = WALK_HEADER_BITS + r.wbits.iter().map(|&b| b as u32).sum::<u32>();
            assert!(spent <= WALK_RECORD_BITS, "walk entry {id}: {spent} bits");
        }
    }
}
