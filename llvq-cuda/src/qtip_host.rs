//! Host-side codec for QTIP's 2-bit trellis format — the comparison arm.
//!
//! A reviewer asked what the Leech kernel is worth against the 2-bit codec it
//! actually competes with, rather than against FP16. QTIP (Cornell-RelaxML) is
//! that codec. This module is the host half of the arm: it reads a QTIP 2-bit
//! weight buffer and produces the same f32 matrix their kernel would, so a
//! bench can check a device decoder against something, and so the bit
//! accounting can be stated rather than assumed.
//!
//! ## The licence boundary, and why this file exists at all
//!
//! **No QTIP code is committed to this repository, and none may be.** Their
//! kernel is GPL v3; this workspace is MIT OR Apache-2.0, and vendoring the
//! one into the other is not a thing a licence audit forgives. Their sources
//! are fetched inside the job that runs the bench and never leave it.
//!
//! What is committed is this file: a clean-room implementation written from a
//! *specification of the buffer format*, established on 2026-08-20 by reading
//! their kernel and then confirmed mechanically — a transcription of their
//! packer was run against the formula below on 200 random tiles, 200/200
//! agreeing. A file format is not their expression; this code is ours.
//!
//! ## The format, in four rules
//!
//! Config constants of the 2-bit setting: window `L = 16` bits, `S = 9` bits
//! of codebook index, `R = 2` bits per weight, `V = 2` weights per state, and
//! `K·V = 4` bits consumed per state.
//!
//! 1. **Tile grid.** An `d_out × d_in` matrix is cut into 16×16 tiles, and the
//!    tiles are stored **row-major on the grid**: tile `(ti, tj)` owns the 32
//!    `u16` starting at `(ti * (d_in/16) + tj) * 32`. 32 `u16` = 512 bits =
//!    256 weights at 2 bits = exactly 2.0000 b/weight, with no header, no
//!    scale and no exception stream. See [`bits_per_weight`].
//!
//! 2. **Inside a tile: one circular shift register.** Concatenate the 32
//!    `u16` into 512 bits, each word contributing its bits **MSB first**. The
//!    state `s ∈ [0, 128)` is the 16-bit window starting at bit `4s`, read
//!    **modulo 512**, first bit read being the state's most significant:
//!
//!    ```text
//!    state(s) = Σ_{j=0..15} bit[(4s + j) mod 512] << (15 - j)
//!    ```
//!
//!    The `mod` is not decoration: states 125, 126 and 127 read 4, 8 and 12
//!    bits of the tile's *first* word. [`tile_state`] is that formula, and
//!    `the_wrap_is_load_bearing` is the test that a dropped modulo fails.
//!
//! 3. **A state decodes to two weights** (their `quantlut_sym`): hash the
//!    state, take 9 bits of the hash as a codebook index, and take the hash's
//!    top bit as a sign that negates **the first element only**. See
//!    [`decode_state`].
//!
//! 4. **Where the 256 weights land.** The 128 states give 256 weights in
//!    order; flat index `f` sits at row `f / 16`, column `f % 16` of the tile.
//!    So `f = i * 16 + j` and the weight is element `f % 2` of state `f / 2`.
//!
//! What is deliberately **not** modelled here is the kernel's lane/warp read
//! pattern — `weight_idx`, the shuffles, the byte permutes. That is how their
//! kernel *reaches* this buffer, not a property of the buffer. Rules 1-4
//! define the format and nothing else does.
//!
//! ## Every bit pattern is a valid buffer
//!
//! This is a **fixed-rate** code: 2 bits per weight, always, whatever the
//! weights. There is no length field, no escape, no reserved pattern and no
//! checksum, so any sequence of `u16` decodes — to *some* matrix. That is why
//! [`pseudo_random_buffer`] can hand a bench a buffer without a checkpoint:
//! the shape, the traffic and the decode cost are exactly those of a real
//! one. It is also why nothing here can validate a buffer, and why a wrong
//! `d_in` produces a plausible matrix rather than an error — the only guard
//! is that the shapes are multiples of 16, which [`buffer_u16_len`] asserts.
//!
//! ⚠️ A pseudo-random buffer is a **shape and traffic** stand-in only. Any
//! quality number taken from one would be a number about our RNG.
//!
//! ## Shapes
//!
//! Qwen3-4B's projections take `d_in`/`d_out` from {1024, 2560, 4096, 9728},
//! and all four are multiples of 16 (64, 160, 256 and 608 tiles), so the
//! format covers the deliverable with no padding — pinned by
//! `the_qwen3_4b_shapes_are_all_tileable`.
//!
//! ## What holds this file
//!
//! Thirty-one mutants, twenty-nine killed (2026-08-20): the wrap dropped,
//! clamped or made constant; the window read LSB-first, backwards, or one bit
//! wide; the
//! step at 2 and at 8; the hash without its `+1` and without its truncation;
//! the codebook shift moved either way and its mask removed; the sign taken
//! from bit 14, never applied, applied to the second element and applied to
//! both; the pair emitted reversed; row and column transposed inside the tile
//! and again in the matrix walk; the tile grid stored column-major; a tile
//! accounted as 8-bit words; and both shape guards removed. Two mutants are
//! **equivalent and unkillable by construction**, which is the honest way to
//! report them: `div_ceil` for `/` in [`buffer_u16_len`] cannot differ once
//! its own assert has passed, and `% QTIP_U16_PER_TILE` for `% 32` is the
//! same expression written twice.
//!
//! ## Why a real module and not an `include!`
//!
//! The other host-side codecs here (`planes14_host.rs`, `golay70_host.rs`,
//! `e1v_host.rs`, `seg_host.rs`) are `include!`d because they name
//! `llvq-artifact`, which is a *dev*-dependency of this crate and therefore
//! unnameable from `lib.rs`. This file names nothing at all — no crate, not
//! even `llvq-core` — so it is a plain module, and its tests run under
//! `cargo test -p llvq-cuda --lib` on the development Mac, where there is no
//! CUDA and where tests are free.

/// Tile side, in weights. The format's only tiling.
pub const QTIP_TILE: usize = 16;

/// `u16` words per tile: 512 bits.
pub const QTIP_U16_PER_TILE: usize = 32;

/// Shift-register states per tile.
pub const QTIP_STATES_PER_TILE: usize = 128;

/// Weights per tile — `QTIP_TILE²`, and two per state.
pub const QTIP_WEIGHTS_PER_TILE: usize = QTIP_TILE * QTIP_TILE;

/// Bits per tile, i.e. the modulus of the circular read in rule 2.
pub const QTIP_TILE_BITS: u32 = (QTIP_U16_PER_TILE * 16) as u32;

/// Bits the register advances per state — `K·V` in their notation.
pub const QTIP_STEP_BITS: u32 = 4;

/// Width of the shift-register window — `L`.
pub const QTIP_WINDOW_BITS: u32 = 16;

/// Codebook entries — `2^S` with `S = tlut_bits = 9`.
pub const QTIP_TLUT_LEN: usize = 512;

/// Where the codebook index sits in the 16-bit hash: `16 − S − 1`, the one
/// bit above it being the sign. Their CUDA writes it as `(h & 0x7FC0) >> 6`.
const QTIP_TLUT_SHIFT: u32 = 16 - 9 - 1;

/// The codebook: 512 pairs, one per index. Their `tlut`.
///
/// A type alias rather than a newtype on purpose — the caller owns this, it
/// comes out of their checkpoint, and this module only ever indexes it.
pub type Tlut = [(f32, f32); QTIP_TLUT_LEN];

/// Bits per weight this format costs. Exactly 2.
///
/// Derived from the two constants that produce it rather than written as a
/// literal, so a mutation of either is a failing test and not a comment that
/// disagrees with the code.
///
/// This is the **payload**, which for this format is also the whole story:
/// there is no per-row scale, no tail and no side table, so unlike our own
/// layouts the payload and kernel accountings coincide. Stating that is the
/// point — repo rule n°3 is that a figure carries the accounting it was
/// measured in.
pub fn bits_per_weight() -> f64 {
    (QTIP_U16_PER_TILE * 16) as f64 / QTIP_WEIGHTS_PER_TILE as f64
}

/// `u16` a `d_out × d_in` weight buffer occupies.
///
/// # Panics
///
/// If either extent is zero or not a multiple of [`QTIP_TILE`]. There is no
/// padding convention in this format: a non-tileable shape has no encoding,
/// and returning a rounded length would silently hand the caller a buffer
/// whose tiles do not line up with its matrix.
pub fn buffer_u16_len(d_out: usize, d_in: usize) -> usize {
    assert!(
        d_out > 0 && d_in > 0,
        "QTIP: empty shape {d_out}x{d_in} has no tiling"
    );
    assert!(
        d_out.is_multiple_of(QTIP_TILE) && d_in.is_multiple_of(QTIP_TILE),
        "QTIP: {d_out}x{d_in} is not a whole number of {QTIP_TILE}x{QTIP_TILE} tiles"
    );
    (d_out / QTIP_TILE) * (d_in / QTIP_TILE) * QTIP_U16_PER_TILE
}

/// Base of the 128-word GROUP holding a tile, and the tile's slot inside it.
///
/// 🚨 **Rewritten 2026-08-20, after job `6a879628` refuted the previous model
/// on the card** at `1.60e-1` of the error budget — structural, not rounding.
/// The old model said a tile was 32 CONTIGUOUS words at
/// `(ti * tiles_k + tj) * 32`; neither half of that was right.
///
/// What the kernel does, derived by replaying its own arithmetic (`weight_idx`,
/// the `__byte_perm` selection in `load_reg_cs`, the 4-bit window shift, and
/// the `mma.m16n8k16` A-fragment layout) and checked against that replay on
/// 23.6 M `(row, col)` pairs over three real shapes:
///
/// * words come in **128-word groups**, each holding **four** tiles — two
///   adjacent K-tiles by two adjacent M-tiles;
/// * inside a group the four tiles are **interleaved word by word, stride 4**:
///   tile `t` owns words `base + 4*n + t`, `n = 0..31`.
///
/// That interleave is why the old model failed: a lane reads four consecutive
/// words belonging to four DIFFERENT tiles, one per `reg_cs` component.
fn group_of(d_in: usize, ti: usize, tj: usize) -> (usize, usize) {
    let tiles_k = d_in / QTIP_TILE;
    // A group holds two K-tiles by two M-tiles, so an odd tile count in either
    // direction has a tile with no partner and no place to live. The kernel
    // asserts the same thing more strongly (`tileCountM % 2 == 0`, and
    // `(tileCountK % 128) % 4 == 0`); this is the weakest form that makes the
    // addressing total, and it is stated here rather than discovered as an
    // out-of-bounds read.
    assert!(tiles_k.is_multiple_of(2), "QTIP: d_in={d_in} gives {tiles_k} K-tiles, which must be even");
    let (m_pair, submi) = (ti / 2, ti % 2);
    let subki = tj % 2;
    (64 * (m_pair * tiles_k + (tj - subki)), 2 * subki + submi)
}

/// State `s` of a tile — rule 2, the circular 16-bit window at bit `4s`.
///
/// Implemented as two shifts rather than as a bit vector because that is what
/// a decoder does; the bit-vector reading is written independently inside
/// `states_are_a_circular_window`, so the formula is checked by two paths and
/// not by one path agreeing with itself.
///
/// The wrap is the `% QTIP_U16_PER_TILE`: for `s ≥ 125` the window's tail
/// comes from word 0. `o` is always one of {0, 4, 8, 12}, so the second shift
/// is between 4 and 16 and the arithmetic stays inside `u32` — a `u16` shift
/// by 16 would be undefined-by-panic in debug and a no-op in release, which
/// is exactly the class of defect that cost this project a kernel (§5, the
/// fourth catch: a `hi << (64 - 0)` ported with its original's assumptions).
///
/// # Panics
///
/// If `s >= QTIP_STATES_PER_TILE`.
#[inline]
pub fn tile_state(tile_u16: &[u16; QTIP_U16_PER_TILE], s: usize) -> u16 {
    assert!(
        s < QTIP_STATES_PER_TILE,
        "QTIP: state {s} past the {QTIP_STATES_PER_TILE} a tile holds"
    );
    let base = QTIP_STEP_BITS as usize * s;
    let w = base / 16;
    let o = (base % 16) as u32;
    let hi = u32::from(tile_u16[w]);
    let lo = u32::from(tile_u16[(w + 1) % QTIP_U16_PER_TILE]);
    (((hi << o) | (lo >> (QTIP_WINDOW_BITS - o))) & 0xFFFF) as u16
}

/// All 128 states of a tile, in order — rule 2 applied 128 times.
pub fn tile_states(tile_u16: &[u16; QTIP_U16_PER_TILE]) -> [u16; QTIP_STATES_PER_TILE] {
    let mut out = [0u16; QTIP_STATES_PER_TILE];
    for (s, st) in out.iter_mut().enumerate() {
        *st = tile_state(tile_u16, s);
    }
    out
}

/// The 16-bit hash a state is decoded through: `((s + 1) · s) mod 2¹⁶`.
///
/// Exposed because the two things rule 3 derives from it — the index and the
/// sign — are tested separately, and a test that recomputed the hash itself
/// would only be testing its own copy.
#[inline]
pub fn state_hash(state: u16) -> u16 {
    let s = u32::from(state);
    (s.wrapping_add(1).wrapping_mul(s) & 0xFFFF) as u16
}

/// Codebook index of a state: 9 bits of the hash, just below the sign bit.
#[inline]
pub fn tlut_index(state: u16) -> usize {
    ((state_hash(state) >> QTIP_TLUT_SHIFT) as usize) & (QTIP_TLUT_LEN - 1)
}

/// Does this state negate its **first** weight? The hash's top bit.
#[inline]
pub fn state_is_negated(state: u16) -> bool {
    state_hash(state) & 0x8000 != 0
}

/// Decode a state into its two weights — rule 3.
///
/// The sign touches element `.0` and nothing else. In their kernel this is an
/// XOR of `0x8000` over the low half of the `uint32` holding the `half2`,
/// i.e. over `.x`, the first of the pair; negating `.1` as well, or instead,
/// would be a decoder that reproduces roughly half the matrix.
///
/// Negation is `-a`, a sign-bit flip, not a multiply: it matches the XOR
/// exactly, including on a zero entry, where `0.0` becomes `-0.0` and every
/// downstream sum is unaffected.
#[inline]
pub fn decode_state(state: u16, tlut: &Tlut) -> (f32, f32) {
    let (a, b) = tlut[tlut_index(state)];
    if state_is_negated(state) {
        (-a, b)
    } else {
        (a, b)
    }
}

/// Decode a whole tile to its 256 weights, **row-major within the tile**.
///
/// Rule 4: index `f` of the returned array is row `f / 16`, column `f % 16`,
/// and comes from element `f % 2` of state `f / 2`.
pub fn decode_tile(
    tile_u16: &[u16; QTIP_U16_PER_TILE],
    tlut: &Tlut,
) -> [f32; QTIP_WEIGHTS_PER_TILE] {
    let mut out = [0f32; QTIP_WEIGHTS_PER_TILE];
    for s in 0..QTIP_STATES_PER_TILE {
        let (a, b) = decode_state(tile_state(tile_u16, s), tlut);
        out[2 * s] = a;
        out[2 * s + 1] = b;
    }
    out
}
/// The 32 words of one tile, in the order the trellis unrolls them.
///
/// Gathered with stride 4 from the tile's group ([`group_of`]); they are not
/// contiguous in memory, and what comes back is the LOGICAL sequence the shift
/// register runs over.
///
/// # Panics
///
/// If the tile is not wholly inside `buf` — which means the buffer and the
/// `d_in` it is read with disagree, and a fixed-rate code cannot say so on its
/// own (see the module note).
pub fn tile_words(buf: &[u16], d_in: usize, ti: usize, tj: usize) -> [u16; QTIP_U16_PER_TILE] {
    let (base, t) = group_of(d_in, ti, tj);
    let end = base + 4 * (QTIP_U16_PER_TILE - 1) + t + 1;
    assert!(
        end <= buf.len(),
        "QTIP: tile ({ti},{tj}) reaches word {end} in {} u16 — buffer and d_in={d_in} disagree",
        buf.len()
    );
    let mut w = [0u16; QTIP_U16_PER_TILE];
    for (n, slot) in w.iter_mut().enumerate() {
        *slot = buf[base + 4 * n + t];
    }
    w
}

/// Which lane, which of the four MMA registers, and which half of it, holds
/// the weight at `(row, col)` of a tile.
///
/// The `mma.sync.aligned.m16n8k16.row.col` A-fragment layout read backwards:
/// register `j` covers rows `lane/4 (+8 if j is odd)` and columns
/// `(lane%4)*2 (+8 if j >= 2)`, so the inverse is exact and total.
fn fragment_slot(row: usize, col: usize) -> (usize, usize, usize) {
    let j = usize::from(row >= 8) + 2 * usize::from(col >= 8);
    let lane = (row % 8) * 4 + (col % 8) / 2;
    (lane, j, col % 2)
}

/// The trellis state a `(lane, register)` pair reads.
///
/// The window slides 4 bits per step across `s_lane | s_{lane+1} << 16`, so
/// register `j` sees state `4*lane + 4 - j`, and `j = 0` reaches entirely into
/// the next lane's word. Modulo 128 because `__shfl_sync(.., lane + 1)` wraps
/// at the warp edge — which is exactly the circular wrap the packer writes.
fn state_index(lane: usize, j: usize) -> usize {
    (4 * lane + 4 - j) % QTIP_STATES_PER_TILE
}

/// One weight of a `d_in`-wide buffer, at `(row, col)` of the matrix.
pub fn weight_at(buf: &[u16], d_in: usize, row: usize, col: usize, tlut: &Tlut) -> f32 {
    assert!(col < d_in, "QTIP: column {col} past width {d_in}");
    let words = tile_words(buf, d_in, row / QTIP_TILE, col / QTIP_TILE);
    let (lane, j, half) = fragment_slot(row % QTIP_TILE, col % QTIP_TILE);
    let (a, b) = decode_state(tile_state(&words, state_index(lane, j)), tlut);
    if half == 0 { a } else { b }
}

/// Decode a whole matrix, row-major, `d_out * d_in` values.
///
/// The reference a device decoder is diffed against. One tile at a time, each
/// weight placed through the fragment map, so the grid order and the in-tile
/// order stay visibly separate instead of fusing into one index expression
/// nobody can read.
///
/// # Panics
///
/// If the shape is not tileable, or `buf` is not [`buffer_u16_len`] long.
pub fn decode_matrix(buf: &[u16], d_out: usize, d_in: usize, tlut: &Tlut) -> Vec<f32> {
    let want = buffer_u16_len(d_out, d_in);
    assert_eq!(buf.len(), want, "QTIP: {d_out}x{d_in} needs {want} u16, got {}", buf.len());
    let mut out = vec![0f32; d_out * d_in];
    for ti in 0..d_out / QTIP_TILE {
        for tj in 0..d_in / QTIP_TILE {
            let words = tile_words(buf, d_in, ti, tj);
            for row in 0..QTIP_TILE {
                for col in 0..QTIP_TILE {
                    let (lane, j, half) = fragment_slot(row, col);
                    let (a, b) = decode_state(tile_state(&words, state_index(lane, j)), tlut);
                    out[(ti * QTIP_TILE + row) * d_in + tj * QTIP_TILE + col] =
                        if half == 0 { a } else { b };
                }
            }
        }
    }
    out
}

/// SplitMix64, so this module keeps its zero dependencies.
///
/// A copy of `llvq_core::rng::SplitMix64`, same constants and same stream —
/// duplicated rather than imported because `llvq-core` is a Linux-only and
/// dev-only dependency of this crate, and this module has to build on the
/// development Mac in a non-test profile. Twelve lines against a `cfg` that
/// would take the whole file off the Mac is the cheaper trade.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Fill a buffer with a deterministic pseudo-random bit pattern.
///
/// Valid by construction: the format is fixed-rate, so **every** bit pattern
/// decodes (module note). What this produces is a stand-in with a real
/// buffer's size and access pattern, for shape checks and for traffic — never
/// for a quality figure.
pub fn fill_pseudo_random(buf: &mut [u16], seed: u64) {
    let mut rng = SplitMix64(seed);
    for chunk in buf.chunks_mut(4) {
        let r = rng.next_u64();
        for (k, w) in chunk.iter_mut().enumerate() {
            *w = (r >> (16 * k)) as u16;
        }
    }
}

/// A whole `d_out × d_in` buffer of deterministic pseudo-random bits.
///
/// See [`fill_pseudo_random`] for what this is and is not good for.
pub fn pseudo_random_buffer(d_out: usize, d_in: usize, seed: u64) -> Vec<u16> {
    let mut buf = vec![0u16; buffer_u16_len(d_out, d_in)];
    fill_pseudo_random(&mut buf, seed);
    buf
}

/// A deterministic stand-in codebook, values in `[-1, 1)`.
///
/// ⚠️ **Not** their codebook. Theirs is trained and ships in their checkpoint;
/// this one exists so that shape, traffic and decode-path tests need no
/// download and no GPL file. Any number produced with this codebook describes
/// this function, not QTIP.
pub fn pseudo_random_tlut(seed: u64) -> Tlut {
    let mut rng = SplitMix64(seed);
    let mut tlut = [(0f32, 0f32); QTIP_TLUT_LEN];
    for e in tlut.iter_mut() {
        let a = (rng.next_u64() >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0;
        let b = (rng.next_u64() >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0;
        // 🚨 Rounded to binary16, and this is a correctness requirement, not a
        // cosmetic one. The kernel reads its codebook as `half2`, so the value
        // it decodes for a state is the binary16 rounding of the entry — not
        // the entry. A reference built on the unrounded f32 would differ from
        // the device by ~2^-11 relative on EVERY weight, which is a thousand
        // times the 1e-5 threshold this arm is held to, and the job would fail
        // its verification for a reason that has nothing to do with the
        // format. Rounding here makes the two agree by construction: the
        // stored value IS a binary16 value, so uploading it loses nothing.
        *e = (round_to_binary16(a), round_to_binary16(b));
    }
    tlut
}

/// Round an `f32` to the nearest `binary16` value and widen it back.
///
/// Written out rather than pulled in, for the reason the module header gives:
/// this file names no dependency, not even `llvq-core`. Ties-to-even, the IEEE
/// default and what a device `__float2half` does.
///
/// Only the range this module produces is exercised — finite values well
/// inside binary16's normal range. Subnormals round correctly by the same
/// arithmetic; infinities and NaN are out of scope and are asserted against
/// rather than silently mishandled.
pub fn round_to_binary16(x: f32) -> f32 {
    assert!(x.is_finite(), "binary16 rounding is only defined here for finite values");
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x007F_FFFF;
    // Beyond binary16's normal range in either direction, the caller is
    // outside what this module generates; refuse rather than clamp silently.
    assert!(exp <= 15, "value too large for binary16: {x}");
    let f = |m: u32, e: i32| -> f32 {
        f32::from_bits(sign | (((e + 127) as u32) << 23) | m)
    };
    if exp < -14 {
        // Subnormal in binary16: the step is 2^-24 whatever the exponent.
        let step = 2f32.powi(-24);
        let mag = f32::from_bits(bits & 0x7FFF_FFFF);
        let q = (mag / step).round_ties_even() * step;
        return f32::from_bits(sign | q.to_bits());
    }
    // Normal: binary16 keeps 10 mantissa bits of the 23, so drop 13 with
    // round-half-to-even.
    let keep = mant >> 13;
    let rest = mant & 0x1FFF;
    let half = 0x1000;
    let up = rest > half || (rest == half && (keep & 1) == 1);
    let keep = keep + u32::from(up);
    // A carry out of the 10 kept bits bumps the exponent, which is correct and
    // exactly representable.
    if keep == 0x400 {
        f(0, exp + 1)
    } else {
        f(keep << 13, exp)
    }
}

/// Bytes the QTIP arm streams for one matrix, in **the bench's accounting**:
/// the payload and nothing else.
///
/// Exactly `d_out * d_in / 4`: two bits per weight, with no tail columns, no
/// per-row scales and no side channel. That is a real asymmetry against our own
/// layouts, which bill an f32 tail and f32 row scales, and it runs **in QTIP's
/// favour** — so it is declared here rather than left for a reader to discover,
/// the same way `awq_bytes` declares its own.
///
/// The 512-entry codebook (4 KiB) is excluded on the same rule that excludes our
/// per-class tables: a resident constant of a few kilobytes against gigabytes of
/// stream.
pub fn qtip_bytes(d_out: usize, d_in: usize) -> u64 {
    assert!(
        d_out.is_multiple_of(QTIP_TILE) && d_in.is_multiple_of(QTIP_TILE),
        "shape must tile 16x16"
    );
    // And both tile counts must be EVEN: a group pairs two K-tiles by two
    // M-tiles, so an odd count leaves a tile with no partner and no address.
    // The five served shapes give 256/64/160/608 M-tiles and 160/256/608
    // K-tiles, all even; the kernel asserts the M half itself
    // (`static_assert(tileCountM % 2 == 0)`).
    assert!(
        (d_out / QTIP_TILE).is_multiple_of(2) && (d_in / QTIP_TILE).is_multiple_of(2),
        "QTIP: {d_out}x{d_in} gives {}x{} tiles; both counts must be even",
        d_out / QTIP_TILE,
        d_in / QTIP_TILE
    );
    (buffer_u16_len(d_out, d_in) * 2) as u64
}

/// The f64 reference for one output row: what the kernel will compute, exactly.
///
/// "Exactly" is meant literally, and it is why this arm can be held to our own
/// `1e-5` threshold rather than the looser one AWQ needs. We know the compressed
/// bits, so we know the state, so we know the codebook entry and the sign — the
/// decoded weight is not an approximation of anything. The kernel accumulates in
/// f32 (its output is `float`, not `half`), so the only difference left is
/// accumulation order.
///
/// ⚠️ `xf` must be the activation **rounded to binary16 and widened back**: the
/// kernel reads `half2`, and feeding the f32 activation here would inflate the
/// measured error by a gap that is not this arm's.
pub fn reference_row(buf: &[u16], d_in: usize, row: usize, xf: &[f64], tlut: &Tlut) -> f64 {
    assert_eq!(xf.len(), d_in, "activation length must be d_in");
    let mut acc = 0.0f64;
    for (col, &x) in xf.iter().enumerate() {
        acc += weight_at(buf, d_in, row, col, tlut) as f64 * x;
    }
    acc
}

/// The five distinct projection shapes of Qwen3-4B, and the kernel name each
/// one resolves to in `kernels/qtip_glue.cu`.
///
/// The model has seven projections; `k_proj`/`v_proj` share a shape and so do
/// `gate_proj`/`up_proj`, so five shims cover all seven. The kernel is templated
/// on `M` and `K`, which is why the name carries them.
pub const QTIP_SHAPES: [(usize, usize); 5] =
    [(4096, 2560), (1024, 2560), (2560, 4096), (9728, 2560), (2560, 9728)];

/// Name of the `extern "C"` shim for a shape, or `None` if no shim exists.
///
/// Returning `None` rather than formatting a name unconditionally is the point:
/// an unlisted shape must fail by name here, at build time, not as a missing
/// symbol lookup after the buffers are on the card.
pub fn kernel_name(d_out: usize, d_in: usize) -> Option<String> {
    QTIP_SHAPES
        .contains(&(d_out, d_in))
        .then(|| format!("qtip_mv_{d_out}x{d_in}"))
}

/// Dynamic shared memory the kernel needs: `1 << (S + 5 + V + 1)` with `S = 9`
/// and `V = 1`, i.e. the 512-entry codebook replicated 32 times to spread it
/// across banks.
///
/// 🚨 64 KiB is **above** the 48 KiB per-block default, so the launcher must opt
/// in (`Cuda::func_dynamic_shared`) exactly as the rotation kernel does. A shim
/// compiles without the opt-in and fails at launch.
pub const QTIP_SHARED_BYTES: u32 = 1 << (9 + 5 + 1 + 1);

/// The per-block default this exceeds. Kept as a named constant so the relation
/// below is a statement rather than a magic number in a comment.
pub const CUDA_DEFAULT_SHARED_BYTES: u32 = 49_152;

// Compile-time, because the whole point is that this can never stop being true
// without the launcher's opt-in becoming wrong at the same moment.
const _: () = assert!(QTIP_SHARED_BYTES > CUDA_DEFAULT_SHARED_BYTES);

#[cfg(test)]
mod tests {
    use super::*;

    /// Rule 2, spelled out the slow way: materialise the 512 bits, then read a
    /// circular window per state.
    ///
    /// This exists to be a **second path**, so it deliberately shares no
    /// arithmetic with [`tile_state`]: no packed-word shifts, no `hi`/`lo`
    /// pair, no modulus on word indices. If both agree, the two ways of
    /// spelling MSB-first-then-wrap agree, which is the only thing that could
    /// plausibly be wrong here.
    fn states_by_bit_vector(tile: &[u16; QTIP_U16_PER_TILE]) -> [u16; QTIP_STATES_PER_TILE] {
        let mut bits = [0u8; 512];
        for (w, &word) in tile.iter().enumerate() {
            for b in 0..16 {
                bits[w * 16 + b] = ((word >> (15 - b)) & 1) as u8;
            }
        }
        let mut out = [0u16; QTIP_STATES_PER_TILE];
        for (s, st) in out.iter_mut().enumerate() {
            let mut v = 0u16;
            for j in 0..16 {
                v |= u16::from(bits[(4 * s + j) % 512]) << (15 - j);
            }
            *st = v;
        }
        out
    }

    /// A codebook of exactly representable, strictly positive, pairwise
    /// distinct values — so an equality assertion is exact and a sign is
    /// visible. Every numerator is an integer below 1024 over 512, hence a
    /// binary32 with no rounding anywhere in these tests.
    fn analytic_tlut() -> Tlut {
        let mut t = [(0f32, 0f32); QTIP_TLUT_LEN];
        for (i, e) in t.iter_mut().enumerate() {
            *e = ((i as f32 + 1.0) / 512.0, -(i as f32 + 3.0) / 512.0);
        }
        t
    }

    /// Two independent readings of rule 2 agree — on random tiles, on the two
    /// saturated tiles, and on **every one of the 512 single-bit tiles**.
    ///
    /// The single-bit sweep is what makes this lethal rather than reassuring:
    /// it pins the contribution of each bit position separately, so a
    /// bit-order flip (LSB-first within a word), an off-by-one in the window,
    /// or a step of 2 or 8 instead of 4 moves at least one of the 512 cases.
    #[test]
    fn states_are_a_circular_window() {
        let mut rng = SplitMix64(0xC0FF_EE12_3456_789A);
        for _ in 0..64 {
            let mut tile = [0u16; QTIP_U16_PER_TILE];
            for w in tile.iter_mut() {
                *w = rng.next_u64() as u16;
            }
            assert_eq!(
                tile_states(&tile),
                states_by_bit_vector(&tile),
                "tile {tile:04x?}"
            );
        }
        for fill in [0x0000u16, 0xFFFF] {
            let tile = [fill; QTIP_U16_PER_TILE];
            assert_eq!(tile_states(&tile), states_by_bit_vector(&tile));
        }
        for p in 0..512usize {
            let mut tile = [0u16; QTIP_U16_PER_TILE];
            tile[p / 16] = 1u16 << (15 - p % 16);
            let a = tile_states(&tile);
            assert_eq!(
                a,
                states_by_bit_vector(&tile),
                "single bit at stream position {p}"
            );
            // Non-vacuity: a lone bit is seen by exactly four windows, since
            // the window is 16 bits and the step is 4. A formula that read
            // nothing, or read everything, would still pass an `assert_eq!`
            // of two broken paths only if both broke identically — this
            // catches the case where both return zero.
            assert_eq!(
                a.iter().filter(|&&s| s != 0).count(),
                4,
                "bit {p} must be visible to exactly four states"
            );
        }
    }

    /// States 125-127 read the tile's **first** word. Drop the modulus and
    /// this test dies: a decoder that indexes word 32 panics, one that clamps
    /// or masks to zero returns 0 where the format says 0x0ABC.
    ///
    /// The tile has a single non-zero word, so nothing else can supply these
    /// bits and the assertion is about the wrap and about nothing else.
    #[test]
    fn the_wrap_is_load_bearing() {
        let mut tile = [0u16; QTIP_U16_PER_TILE];
        tile[0] = 0xABCD;
        let st = tile_states(&tile);

        // The window at bit 0 is word 0 itself, and the next one has already
        // slid four bits: the register is a register.
        assert_eq!(st[0], 0xABCD);
        assert_eq!(st[1], 0xBCD0);

        // State 124 starts at bit 496 — the last word, entirely zero, and no
        // wrap yet. It is the control: it proves the three below are not
        // simply "every late state sees word 0".
        assert_eq!(st[124], 0x0000);

        // 125, 126, 127 start at bits 500, 504, 508 and take 4, 8 and 12 bits
        // from word 0 across the wrap.
        assert_eq!(st[125], 0x000A, "state 125 must wrap 4 bits into word 0");
        assert_eq!(st[126], 0x00AB, "state 126 must wrap 8 bits into word 0");
        assert_eq!(st[127], 0x0ABC, "state 127 must wrap 12 bits into word 0");

        // And the same three, read the other way, in case the constants above
        // were copied from the implementation rather than from the format.
        assert_eq!(&st[124..], &states_by_bit_vector(&tile)[124..]);
    }

    /// Rule 3's sign negates `.0` and leaves `.1` untouched — checked on all
    /// 65 536 states, with a codebook whose entries are all positive and
    /// pairwise distinct, so "unchanged" and "negated" are distinguishable
    /// values and not two spellings of zero.
    ///
    /// Kills: sign applied to `.1`, to both, to neither, or always.
    #[test]
    fn sign_flip_hits_only_the_first_element() {
        let tlut = analytic_tlut();
        let mut negated = 0usize;
        let mut plain = 0usize;
        for s in 0..=u16::MAX {
            let idx = tlut_index(s);
            let (want_a, want_b) = tlut[idx];
            let (a, b) = decode_state(s, &tlut);
            assert_eq!(b, want_b, "state {s:#06x}: second element must never flip");
            if state_is_negated(s) {
                assert_eq!(a, -want_a, "state {s:#06x}: first element must flip");
                assert!(a < 0.0);
                negated += 1;
            } else {
                assert_eq!(a, want_a, "state {s:#06x}: first element must not flip");
                assert!(a > 0.0);
                plain += 1;
            }
        }
        // Non-vacuity, and a fact about the hash worth pinning: the top bit is
        // set for exactly half the state space.
        assert_eq!((negated, plain), (32_768, 32_768));
    }

    /// The hash, its index field and its sign bit, pinned on the edges.
    ///
    /// `0xFFFF` is the one that matters: `(s + 1) · s` overflows 16 bits
    /// there, and the format keeps the low 16. A `u16` multiply would panic
    /// in debug and silently agree in release — the exact shape of defect §5
    /// calls the fourth catch.
    #[test]
    fn the_hash_is_the_published_one() {
        let cases = [
            //  state,  hash,   index, negated
            (0x0000u16, 0x0000u16, 0usize, false),
            (0x0001, 0x0002, 0, false),
            (0x00FF, 0xFF00, 508, true),
            (0x0100, 0x0100, 4, false),
            (0x7FFF, 0x8000, 0, true),
            (0x8000, 0x8000, 0, true),
            (0xFFFF, 0x0000, 0, false),
            (0xA4D3, 0xAABC, 170, true),
            (0x4D3C, 0x734C, 461, false),
        ];
        for (s, h, idx, neg) in cases {
            assert_eq!(state_hash(s), h, "hash of {s:#06x}");
            assert_eq!(tlut_index(s), idx, "index of {s:#06x}");
            assert_eq!(state_is_negated(s), neg, "sign of {s:#06x}");
        }
        // The whole codebook is reachable: a shift or mask that lost a bit
        // would leave half of it dead, which no per-state assertion notices.
        let seen: std::collections::HashSet<usize> = (0..=u16::MAX).map(tlut_index).collect();
        assert_eq!(seen.len(), QTIP_TLUT_LEN);
    }

    /// The rate is exactly two bits per weight, on the shapes the deliverable
    /// has and on a square one for good measure.
    ///
    /// Exact float equality is deliberate: 2.0 is a binary32 and a binary64,
    /// and a format that missed by a padding word would not land on it.
    #[test]
    fn rate_is_exactly_two_bits() {
        for (d_out, d_in) in [
            (2560usize, 2560usize),
            (2560, 9728),
            (9728, 2560),
            (4096, 4096),
            (1024, 2560),
            (2560, 1024),
            (16, 16),
        ] {
            let words = buffer_u16_len(d_out, d_in);
            assert_eq!(words * 16, d_out * d_in * 2, "{d_out}x{d_in}: bits");
            assert_eq!(
                (words * 16) as f64 / (d_out * d_in) as f64,
                2.0,
                "{d_out}x{d_in}: b/weight"
            );
        }
        assert_eq!(bits_per_weight(), 2.0);
    }

    /// Every projection width of Qwen3-4B is a whole number of tiles, so the
    /// format covers the deliverable with no padding and no exception.
    ///
    /// If a future model brings a width that is not a multiple of 16, this
    /// test is where that has to be noticed — the answer would be a padding
    /// convention, which this format does not have.
    #[test]
    fn the_qwen3_4b_shapes_are_all_tileable() {
        for d in [1024usize, 2560, 4096, 9728] {
            assert_eq!(d % QTIP_TILE, 0, "width {d}");
        }
        assert_eq!(
            [1024, 2560, 4096, 9728].map(|d: usize| d / QTIP_TILE),
            [64, 160, 256, 608]
        );
    }

    #[test]
    #[should_panic(expected = "not a whole number of")]
    fn a_non_tileable_shape_is_refused() {
        // 2552 = 16·159 + 8. There is no encoding for it, and rounding up
        // would hand back a buffer whose tiles do not line up with the matrix.
        buffer_u16_len(2560, 2552);
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn a_buffer_of_the_wrong_length_is_refused() {
        let tlut = analytic_tlut();
        // Packed for one shape, decoded as another. Both are tileable and
        // both have even tile counts, so only the LENGTH check can catch it —
        // which is the point: a fixed-rate code carries no shape of its own.
        let buf = pseudo_random_buffer(32, 64, 1);
        decode_matrix(&buf, 32, 96, &tlut);
    }

    /// The four tiles of a group are **interleaved word by word, stride 4** —
    /// the property that job `6a879628` proved and the previous model denied.
    ///
    /// 🚨 This test replaces `the_tile_grid_is_row_major`, which pinned the
    /// refuted model: it asserted a tile was 32 contiguous words at
    /// `(ti·tiles_k + tj)·32`. It passed, and it was wrong — a test can only
    /// pin what its author believes, which is why the card is the arbiter.
    /// Replay the KERNEL's own index arithmetic and confront the closed form
    /// with it — the test that would have caught the model job `6a879628`
    /// refuted, and the reason it did not exist is that the old model was
    /// derived by reading rather than by replaying.
    ///
    /// The replay walks blocks, warps, the `ki` loop and the `(submi, subki)`
    /// pair exactly as `kernel_decompress_matvec` does, and derives from
    /// `weight_idx` which group each `(M-tile, K-tile)` lands in. Two things
    /// are asserted: the map is a **bijection** onto every `(row, col)` of the
    /// matrix, and it **agrees with `group_of`** everywhere.
    ///
    /// Bijection is the load-bearing half. A wrong-but-consistent map would
    /// still agree with itself; only covering each weight exactly once, with
    /// no collision, pins the addressing.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "replays 128 blocks x 32 warps; release only")]
    fn the_closed_form_replays_the_kernel_indexing() {
        // A real served shape, small enough to enumerate: 1024x2560.
        const M: usize = 1024;
        const K: usize = 2560;
        let (tiles_m, tiles_k) = (M / QTIP_TILE, K / QTIP_TILE);
        let m_per_block = tiles_m.div_ceil(2 * 128);
        let k_per_block = tiles_k / (32 * 4) * 2;

        // One entry per (M-tile, K-tile); `usize::MAX` means "not yet seen".
        let mut seen = vec![usize::MAX; tiles_m * tiles_k];
        for block in 0..128usize {
            // Written as the kernel writes it: `tileIdM` advances at the
            // bottom of the loop body, and the guard is the kernel's own
            // early return. Clippy suggests zipping a range, which would
            // stop being a transcription.
            #[allow(clippy::mut_range_bound)]
            for mi in 0..m_per_block {
                let tile_id_m = m_per_block * block + mi;
                if tile_id_m * 2 >= tiles_m {
                    break;
                }
                for warp in 0..32usize {
                    let this_warp_k = if warp < (tiles_k % 128) / 4 {
                        k_per_block + 2
                    } else {
                        k_per_block
                    };
                    for ki in 0..this_warp_k {
                        // The base of the 128-word group `reg_cs` refers to.
                        let base = tile_id_m * (tiles_k * QTIP_U16_PER_TILE * 2)
                            + (ki / 2) * (32 * 128 * 2)
                            + warp * 256
                            + (ki % 2) * 128;
                        let kt0 = 128 * (ki / 2) + 4 * warp + 2 * (ki % 2);
                        for subki in 0..2 {
                            for submi in 0..2 {
                                let (mt, kt) = (2 * tile_id_m + submi, kt0 + subki);
                                let slot = 2 * subki + submi;
                                // The closed form must produce the same group
                                // and the same slot.
                                assert_eq!(
                                    group_of(K, mt, kt),
                                    (base, slot),
                                    "tile ({mt},{kt}) block={block} warp={warp} ki={ki}"
                                );
                                let cell = &mut seen[mt * tiles_k + kt];
                                assert_eq!(*cell, usize::MAX, "tile ({mt},{kt}) dispatched twice");
                                *cell = base;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&v| v != usize::MAX),
            "the kernel never dispatches some tile of a {M}x{K} matrix"
        );

        // And the fragment map is a bijection inside a tile: 32 lanes x 4
        // registers x 2 halves must hit each of the 256 positions once.
        let mut hit = [0u8; QTIP_WEIGHTS_PER_TILE];
        for row in 0..QTIP_TILE {
            for col in 0..QTIP_TILE {
                let (lane, j, half) = fragment_slot(row, col);
                assert!(lane < 32 && j < 4 && half < 2);
                hit[state_index(lane, j) * 2 + half] += 1;
            }
        }
        assert!(hit.iter().all(|&c| c == 1), "the fragment map is not a bijection");
    }

    #[test]
    fn the_four_tiles_of_a_group_are_interleaved() {
        let d_in = 64usize; // 4 K-tiles
        // Tiles (0,0), (1,0), (0,1), (1,1) share ONE group and differ only by
        // their slot; (0,2) starts the next group.
        assert_eq!(group_of(d_in, 0, 0), (0, 0));
        assert_eq!(group_of(d_in, 1, 0), (0, 1));
        assert_eq!(group_of(d_in, 0, 1), (0, 2));
        assert_eq!(group_of(d_in, 1, 1), (0, 3));
        assert_eq!(group_of(d_in, 0, 2), (128, 0));
        // A second M-pair is one full grid row of groups further on.
        assert_eq!(group_of(d_in, 2, 0), (64 * 4, 0));

        // And the gather really is strided: word n of tile t sits at 4n + t.
        let buf: Vec<u16> = (0..(64 / 16) * (32 / 16) * QTIP_U16_PER_TILE)
            .map(|i| i as u16)
            .collect();
        let w = tile_words(&buf, d_in, 1, 1);
        assert_eq!(w[0], 3, "tile slot 3 starts at word 3");
        assert_eq!(w[1], 7, "then 4 words later");
        assert_eq!(w[31], 4 * 31 + 3);
    }

    /// Rule 1: groups are row-major on the grid — checked on a **non-square**
    /// shape, the only kind that can tell one row stride from the other.
    #[test]
    fn the_group_grid_is_row_major() {
        let (d_out, d_in) = (32usize, 64usize); // 2 x 4 tiles, non-square and both even
        let tlut = analytic_tlut();
        let buf = pseudo_random_buffer(d_out, d_in, 0x9E37);
        assert_eq!(buf.len(), 8 * QTIP_U16_PER_TILE);

        // The whole matrix decoded two ways: tile by tile, and weight by
        // weight. They can only agree if the grid order and the fragment map
        // agree, which is what this shape is here to separate.
        let whole = decode_matrix(&buf, d_out, d_in, &tlut);
        for row in 0..d_out {
            for col in 0..d_in {
                assert_eq!(
                    whole[row * d_in + col],
                    weight_at(&buf, d_in, row, col, &tlut),
                    "({row},{col})"
                );
            }
        }

        // Non-vacuity: a transposed grid would place tile (1,0) where (0,1)
        // is, so the two must not decode alike. Without this the loop above
        // would also pass on a buffer whose tiles happened to be equal.
        assert_ne!(
            tile_words(&buf, d_in, 0, 1).to_vec(),
            tile_words(&buf, d_in, 1, 0).to_vec(),
            "a transposed grid would make these the same words"
        );
    }

    /// [`weight_at`] and [`decode_matrix`] are the same map, on both
    /// orientations of a non-square shape.
    ///
    /// They share `decode_state` and `tile_state` but not the index
    /// arithmetic of rules 1 and 4, which is the half that has two spellings
    /// in this file and could drift.
    #[test]
    fn weight_at_agrees_with_decode_matrix() {
        let tlut = pseudo_random_tlut(7);
        // Both orientations, and both tile counts even — a group pairs two
        // K-tiles by two M-tiles, so an odd count has no representation.
        for (d_out, d_in) in [(32usize, 64usize), (64, 32), (32, 96)] {
            let buf = pseudo_random_buffer(d_out, d_in, 0xBEEF);
            let m = decode_matrix(&buf, d_out, d_in, &tlut);
            assert_eq!(m.len(), d_out * d_in);
            for row in 0..d_out {
                for col in 0..d_in {
                    assert_eq!(
                        m[row * d_in + col],
                        weight_at(&buf, d_in, row, col, &tlut),
                        "{d_out}x{d_in} at ({row},{col})"
                    );
                }
            }
        }
    }

    /// A literal lock on the whole format: one fixed tile in, 128 states and
    /// the decoded weights out.
    ///
    /// The expected values were produced **outside this crate**, by a third
    /// implementation written straight from the specification in python, so
    /// they are not this code agreeing with itself. Any change to bit order,
    /// step, wrap, hash, index field, sign placement or in-tile layout moves
    /// at least one of these numbers.
    ///
    /// The shape of the state list is the format's own signature and can be
    /// read by eye: consecutive states slide by one hex digit — `0x4d3c`,
    /// `0xd3c1`, `0x3c12`, `0xc128` — and the last, `0xa4d3`, is the last
    /// word's low nibble glued to the first word's top three.
    #[test]
    fn known_vector() {
        // Laid out eight to a line, and kept that way: the sliding-nibble
        // signature described above is only visible in this shape.
        #[rustfmt::skip]
        const TILE: [u16; QTIP_U16_PER_TILE] = [
            0x4d3c, 0x1284, 0xeb3e, 0xcd58, 0x2f6c, 0xc5cd, 0xe5a7, 0x1ce0,
            0x6380, 0x3560, 0xf79f, 0x2eec, 0xd650, 0xbf3e, 0x8881, 0xce7a,
            0x3083, 0x5df1, 0x1295, 0xe1d3, 0x31dc, 0x3627, 0x58ec, 0x6dc8,
            0x6b47, 0xeac2, 0x87b3, 0xb240, 0xb9fa, 0x2072, 0x0047, 0x536a,
        ];
        // Laid out eight to a line, and kept that way: the sliding-nibble
        // signature described above is only visible in this shape.
        #[rustfmt::skip]
        const STATES: [u16; QTIP_STATES_PER_TILE] = [
            0x4d3c, 0xd3c1, 0x3c12, 0xc128, 0x1284, 0x284e, 0x84eb, 0x4eb3,
            0xeb3e, 0xb3ec, 0x3ecd, 0xecd5, 0xcd58, 0xd582, 0x582f, 0x82f6,
            0x2f6c, 0xf6cc, 0x6cc5, 0xcc5c, 0xc5cd, 0x5cde, 0xcde5, 0xde5a,
            0xe5a7, 0x5a71, 0xa71c, 0x71ce, 0x1ce0, 0xce06, 0xe063, 0x0638,
            0x6380, 0x3803, 0x8035, 0x0356, 0x3560, 0x560f, 0x60f7, 0x0f79,
            0xf79f, 0x79f2, 0x9f2e, 0xf2ee, 0x2eec, 0xeecd, 0xecd6, 0xcd65,
            0xd650, 0x650b, 0x50bf, 0x0bf3, 0xbf3e, 0xf3e8, 0x3e88, 0xe888,
            0x8881, 0x881c, 0x81ce, 0x1ce7, 0xce7a, 0xe7a3, 0x7a30, 0xa308,
            0x3083, 0x0835, 0x835d, 0x35df, 0x5df1, 0xdf11, 0xf112, 0x1129,
            0x1295, 0x295e, 0x95e1, 0x5e1d, 0xe1d3, 0x1d33, 0xd331, 0x331d,
            0x31dc, 0x1dc3, 0xdc36, 0xc362, 0x3627, 0x6275, 0x2758, 0x758e,
            0x58ec, 0x8ec6, 0xec6d, 0xc6dc, 0x6dc8, 0xdc86, 0xc86b, 0x86b4,
            0x6b47, 0xb47e, 0x47ea, 0x7eac, 0xeac2, 0xac28, 0xc287, 0x287b,
            0x87b3, 0x7b3b, 0xb3b2, 0x3b24, 0xb240, 0x240b, 0x40b9, 0x0b9f,
            0xb9fa, 0x9fa2, 0xfa20, 0xa207, 0x2072, 0x0720, 0x7200, 0x2004,
            0x0047, 0x0475, 0x4753, 0x7536, 0x536a, 0x36a4, 0x6a4d, 0xa4d3,
        ];
        assert_eq!(tile_states(&TILE), STATES);

        let tlut = analytic_tlut();
        let w = decode_tile(&TILE, &tlut);

        // Weights 0..8 — tile row 0, columns 0..8, i.e. states 0..4.
        //
        // Written as the exact rationals `analytic_tlut` holds, `k + 1` and
        // `−(k + 3)` over 512, so the codebook index each state selects is
        // visible in the fixture instead of being buried in a decimal. Every
        // one is a binary32 to the last bit, so `assert_eq!` is exact.
        // Note the negated *first* elements at f = 2 and f = 4, and the
        // untouched second elements beside them.
        assert_eq!(
            w[..8],
            [
                462.0f32 / 512.0, // state 0x4d3c -> tlut[461], sign kept
                -464.0 / 512.0,
                -46.0 / 512.0, // state 0xd3c1 -> tlut[45], negated
                -48.0 / 512.0,
                -182.0 / 512.0, // state 0x3c12 -> tlut[181], negated
                -184.0 / 512.0,
                94.0 / 512.0, // state 0xc128 -> tlut[93], sign kept
                -96.0 / 512.0,
            ]
        );
        // …and the last four, which are the two states that wrap.
        assert_eq!(
            w[QTIP_WEIGHTS_PER_TILE - 4..],
            [
                278.0f32 / 512.0, // state 0x6a4d -> tlut[277], sign kept
                -280.0 / 512.0,
                -171.0 / 512.0, // state 0xa4d3 -> tlut[170], negated
                -173.0 / 512.0,
            ]
        );
        // 59 of this tile's 128 states negate. Pinned because the count is a
        // property of the hash applied to *these* states: a decoder that
        // never negates, or always does, cannot reach 59.
        assert_eq!(STATES.iter().filter(|&&s| state_is_negated(s)).count(), 59);
    }

    /// The two guards that turn a confusing panic into a named one.
    ///
    /// Both were mutation survivors until this test existed: removing either
    /// assert left every other test green, because the slice index panics on
    /// its own a line later — with a message that says nothing about *why*.
    /// For a fixed-rate format that is the difference that matters: a buffer
    /// read with the wrong `d_in` is the one mistake nothing else can detect
    /// (module note), so the message has to name it.
    #[test]
    #[should_panic(expected = "buffer and d_in=64 disagree")]
    fn a_tile_past_the_buffer_names_the_disagreement() {
        let buf = pseudo_random_buffer(32, 64, 1);
        // Row 32 does not exist in a 32-row matrix; with a fixed-rate code
        // nothing in the buffer says so, and the tile index simply runs off
        // the end.
        let _ = tile_words(&buf, 64, 2, 0);
    }

    #[test]
    #[should_panic(expected = "state 128 past the 128")]
    fn a_state_past_the_tile_is_refused() {
        let tile = [0u16; QTIP_U16_PER_TILE];
        let _ = tile_state(&tile, QTIP_STATES_PER_TILE);
    }

    /// The stand-in generators are deterministic and are what they claim.
    ///
    /// Not decoration: `pseudo_random_buffer` is what a bench will hand a
    /// device decoder when no checkpoint is mounted, so a seed that did not
    /// reproduce would make two runs incomparable and nothing would say so.
    #[test]
    fn binary16_rounding_is_idempotent_and_ties_to_even() {
        // Idempotence on a wide sweep: rounding an already-rounded value must
        // be a no-op, which is the property the reference depends on.
        let mut x = -3.0f32;
        while x < 3.0 {
            let r = round_to_binary16(x);
            assert_eq!(round_to_binary16(r), r, "not idempotent at {x}");
            x += 0.0007;
        }
        // Exactly representable values survive untouched.
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.25, 2.0, 1024.0, -1024.0] {
            assert_eq!(round_to_binary16(v), v, "{v} is a binary16 value");
        }
        // The step just above 1.0 is 2^-10: 1 + 2^-11 is a tie and must go to
        // the even neighbour, which is 1.0 itself.
        let step = 2f32.powi(-10);
        assert_eq!(round_to_binary16(1.0 + step / 2.0), 1.0);
        // And 1 + 3·2^-12 is past the tie, so it goes up.
        assert_eq!(round_to_binary16(1.0 + step * 0.75), 1.0 + step);
        // Rounding never moves a value by more than half a step.
        let mut y = -2.0f32;
        while y < 2.0 {
            assert!((round_to_binary16(y) - y).abs() <= 2f32.powi(-11), "moved too far at {y}");
            y += 0.0013;
        }
    }

    #[test]
    fn the_stand_in_generators_are_deterministic() {
        assert_eq!(
            pseudo_random_buffer(32, 64, 42),
            pseudo_random_buffer(32, 64, 42)
        );
        assert_ne!(
            pseudo_random_buffer(32, 64, 42),
            pseudo_random_buffer(32, 64, 43)
        );
        let t = pseudo_random_tlut(5);
        assert_eq!(t.to_vec(), pseudo_random_tlut(5).to_vec());
        // Inclusive on purpose: binary16 rounding can carry a value just under
        // 1.0 to exactly 1.0, and the half-open range this test used before
        // 2026-08-20 excluded it. The generator's intent is "magnitudes about
        // ±1", not "strictly inside".
        assert!(t
            .iter()
            .all(|&(a, b)| (-1.0..=1.0).contains(&a) && (-1.0..=1.0).contains(&b)));
        // The invariant that makes the whole arm verifiable: every entry is
        // exactly representable in binary16, so the value the device decodes
        // and the value this reference decodes are the SAME number, not two
        // numbers 2^-11 apart.
        assert!(
            t.iter().all(|&(a, b)| round_to_binary16(a) == a && round_to_binary16(b) == b),
            "a tlut entry is not binary16-exact; the device would decode a different weight"
        );
        // A generator that returned a constant would satisfy everything above.
        assert!(t.iter().any(|&(a, _)| a != t[0].0));
    }
}

#[cfg(test)]
mod wiring_tests {
    use super::*;

    #[test]
    fn the_rate_the_bench_bills_is_two_bits() {
        for (d_out, d_in) in QTIP_SHAPES {
            let bytes = qtip_bytes(d_out, d_in);
            assert_eq!(bytes * 8, (d_out * d_in * 2) as u64, "{d_out}x{d_in}");
        }
    }

    #[test]
    fn every_shape_has_a_shim_and_nothing_else_does() {
        for (d_out, d_in) in QTIP_SHAPES {
            assert_eq!(kernel_name(d_out, d_in).unwrap(), format!("qtip_mv_{d_out}x{d_in}"));
        }
        // A shape the glue does not instantiate must be refused, not formatted:
        // 2048x2048 tiles perfectly well and would pass every upstream
        // static_assert, so only the shim list can reject it.
        assert!(kernel_name(2048, 2048).is_none());
        assert!(kernel_name(2560, 2560).is_none());
    }

    #[test]
    fn the_shared_request_is_the_upstream_formula() {
        // 1 << (S + 5 + V + 1), S = 9, V = 1 — and above the 48 KiB default,
        // which is the whole reason the launcher must opt in.
        assert_eq!(QTIP_SHARED_BYTES, 65536);
    }

    #[test]
    fn the_reference_row_is_the_decoded_matrix() {
        // Two paths to the same number: row-at-a-time against the whole
        // decoded matrix. A packing bug that shifted rows would pass one and
        // not the other.
        let (d_out, d_in) = (32, 64);
        let buf = pseudo_random_buffer(d_out, d_in, 0xF2_0000);
        let tlut = pseudo_random_tlut(0xF2_0001);
        let w = decode_matrix(&buf, d_out, d_in, &tlut);
        let xf: Vec<f64> = (0..d_in).map(|c| ((c % 7) as f64 - 3.0) / 4.0).collect();
        for row in 0..d_out {
            let got = reference_row(&buf, d_in, row, &xf, &tlut);
            let want: f64 =
                (0..d_in).map(|c| w[row * d_in + c] as f64 * xf[c]).sum();
            assert_eq!(got, want, "row {row}");
        }
    }

    #[test]
    #[should_panic(expected = "activation length must be d_in")]
    fn a_mismatched_activation_is_refused() {
        let buf = pseudo_random_buffer(16, 16, 1);
        let tlut = pseudo_random_tlut(2);
        reference_row(&buf, 16, 0, &[0.0; 8], &tlut);
    }
}
