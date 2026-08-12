//! The thesis bench: **one token's worth of linear algebra**, for the whole
//! model, every runtime layout against FP16 on the same machine.
//!
//! Every earlier number was one layer. This is the claim itself — a 2-bit
//! model is smaller *and* faster than the FP16 it replaces — measured on all
//! 252 projection matrices of the published Qwen3-4B, in model order, in one
//! command buffer per format.
//!
//! ## Why one command buffer, and why that is the honest shape
//!
//! * **Naturally cold.** A pass touches 2.4 GB (LLVQ) or 7.3 GB (FP16) of
//!   distinct weights — nothing is re-read, so no matrix can hide in the
//!   48 MB system cache. The `matvec` bench had to force this with rotating
//!   buffer copies; here the workload does it by construction. The lightest
//!   arm still reads 1.5 GB, so the argument survives the extra layouts.
//! * **Submission overhead amortized honestly.** 252 dispatches share one
//!   commit, exactly as a real decode step would.
//! * **Serialized by the real dependency.** Every dispatch writes the same
//!   output buffer, so the write-write hazard drains between them — the same
//!   mechanism that serializes a transformer's actually-dependent layers.
//!
//! What it deliberately leaves out: attention, norms, activations, the
//! rotation applied to `x`, and the tied `lm_head` (unquantized in this
//! artifact, and identical for every arm). It measures the part quantization
//! changes, and reports what the rest costs on top.
//!
//! ## Why the arms are interleaved, and why the spread is printed
//!
//! Three consecutive runs of the *unmodified* two-arm bench returned 2.029×,
//! 2.050× and 2.080× — monotonically rising, both arms accelerating together
//! (`docs/mesures/thesis-temoin-2026-08-04.txt`). That is a warm-up living
//! **between processes** — file cache over a 981 MB artifact, GPU clock
//! state, buffer residency — which the in-process warm-up cannot reach. Two
//! consequences, both structural:
//!
//! * arms are dispatched **round by round in a fixed order**, never all of
//!   one then all of the next, so a monotone drift charges every arm alike;
//! * the ratio is formed **within each round** and then summarised — median
//!   and range. Dividing one arm's best time by another's divides two rounds
//!   that never coexisted. A ratio quoted as a point value hides a ±1.3 %
//!   band, and the experiments this bench exists to arbitrate produce
//!   effects of exactly that size.
//!
//! What this does **not** fix: the absolute milliseconds still drift between
//! runs, so they belong in `docs/mesures/`, not in prose. The bytes and the
//! bits per weight, by contrast, are exact and reproduce to the digit.
//!
//! ## Verification
//!
//! Every matrix's output — all 252, every row, every arm — is checked
//! against an f64 CPU reference before any timing is believed. The three
//! LLVQ layouts are transcoded independently and their decoded blocks are
//! required to agree **exactly**, block for block, so a single reference
//! serves all three. Errors are reported relative to Σ|wᵢ·xᵢ|, the only
//! scale a float dot product is accountable to.
//!
//! Run: `cargo run --release -p llvq-metal --bin thesis [model.llvq]`

use llvq_artifact::runtime::{transcode, ClassTable, Layout, RuntimeBlocks};
use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

/// Blocks of activation staged per tile — 3072 columns, 12 KB, under the
/// 32 KB threadgroup budget with room for occupancy.
const TILE_BLOCKS: usize = 128;

/// Padded floats per staged block in the bank-conflict variant.
///
/// 24 is production. 28 is the smallest larger multiple of 4 — so `float4`
/// loads stay 16-byte aligned — and under a 32-bank/4-byte model it spreads
/// 32 lanes across all 32 banks instead of four. **Apple's bank geometry is
/// documented nowhere**; this constant is the experiment, not a fix.
const TILE_STRIDE: usize = 28;

/// Relative error a kernel may show against the f64 reference.
///
/// The observed worst is 3.4e-8, so this leaves ~300× of headroom. The old
/// 1e-3 could not catch a flipped sign bit: one wrong weight out of `d_in`
/// moves the sum by ~2/d_in of its scale, which is 2.1e-4 on `down_proj` —
/// under the threshold. A fire alarm is not a guard.
const TOL: f64 = 1e-5;

/// Timed rounds, and how many are discarded at the front.
const ROUNDS: usize = 7;
const WARMUP_ROUNDS: usize = 2;

const SRC: &str = r#"
struct Params { uint d_in; uint d_out; uint nblocks; uint tail_w; };

constant uint TILE_COLS = TILE_BLOCKS * DIM;

// ---------------------------------------------------------------------------
// FP16 baseline, tiled. One SIMD group per row, half4 loads.
// ---------------------------------------------------------------------------
kernel void tv_f16(device const half*   w   [[buffer(0)]],
                   device const float*  x   [[buffer(1)]],
                   device float*        y   [[buffer(2)]],
                   constant Params&     P   [[buffer(3)]],
                   threadgroup float*   xs  [[threadgroup(0)]],
                   uint gid  [[thread_position_in_grid]],
                   uint tid  [[thread_index_in_threadgroup]],
                   uint tgs  [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    float acc = 0.0f;
    uint ntiles = (P.d_in + TILE_COLS - 1u) / TILE_COLS;
    for (uint t = 0; t < ntiles; ++t) {
        uint c0 = t * TILE_COLS;
        uint n = min(TILE_COLS, P.d_in - c0);
        // Two barriers: the second orders the fill against the readers, the
        // first stops the next fill from racing the previous tile's readers.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) xs[i] = x[c0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half4* w4 =
            (device const half4*)(w + row * P.d_in + c0);
        for (uint j = lane; j < (n >> 2); j += 32) {
            float4 wv = float4(w4[j]);
            uint c = j << 2;
            acc += wv.x * xs[c] + wv.y * xs[c + 1]
                 + wv.z * xs[c + 2] + wv.w * xs[c + 3];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// The three LLVQ layouts. Identical tiling, identical barriers, identical
// block-local pointer — only the decoder call differs, which is the whole
// point: the bits↔speed curve must come from one protocol on one object.
//
// The tail epilogue reads `x` from **device** memory, not from `xs`: under
// tiling the tail columns sit past the last tile, so `xs[nblocks*DIM + i]`
// would be out of the staged window. `matvec`, which stages the whole row,
// reads them from `xs`; that form must not be copied here.
// ---------------------------------------------------------------------------
#define TV_LLVQ_BODY(DECODE)                                                  \
    uint row = gid >> 5;                                                      \
    uint b0r = row * P.nblocks;                                               \
    float acc = 0.0f;                                                         \
    uint ntiles = (P.nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;               \
    for (uint t = 0; t < ntiles; ++t) {                                       \
        uint jlo = t * TILE_BLOCKS;                                           \
        uint jhi = min(jlo + TILE_BLOCKS, P.nblocks);                         \
        uint c0 = jlo * DIM;                                                  \
        uint n = (jhi - jlo) * DIM;                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint i = tid; i < n; i += tgs) xs[i] = x[c0 + i];                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint j = jlo + lane; j < jhi; j += 32) {                         \
            acc += DECODE(words, bases, tab, gscale, b0r + j,                 \
                          xs + (j - jlo) * DIM);                              \
        }                                                                     \
    }                                                                         \
    acc = simd_sum(acc);                                                      \
    if (lane == 0) {                                                          \
        float tv = 0.0f;                                                      \
        uint tc0 = P.nblocks * DIM;                                           \
        for (uint i = 0; i < P.tail_w; ++i) {                                 \
            tv += tail[row * P.tail_w + i] * x[tc0 + i];                      \
        }                                                                     \
        y[row] = acc * rscale[row] + tv;                                      \
    }

#define TV_LLVQ_ARGS                                                          \
    device const uint*   words  [[buffer(0)]],                                \
    device const uint*   bases  [[buffer(1)]],                                \
    constant ClassRec*   tab    [[buffer(2)]],                                \
    constant float*      gscale [[buffer(3)]],                                \
    device const float*  rscale [[buffer(4)]],                                \
    device const float*  tail   [[buffer(5)]],                                \
    device const float*  x      [[buffer(6)]],                                \
    device float*        y      [[buffer(7)]],                                \
    constant Params&     P      [[buffer(8)]],                                \
    threadgroup float*   xs     [[threadgroup(0)]],                           \
    uint gid  [[thread_position_in_grid]],                                    \
    uint tid  [[thread_index_in_threadgroup]],                                \
    uint tgs  [[threads_per_threadgroup]],                                    \
    uint lane [[thread_index_in_simdgroup]]

kernel void tv_slot(TV_LLVQ_ARGS) { TV_LLVQ_BODY(slot_dot) }
kernel void tv_flat(TV_LLVQ_ARGS) { TV_LLVQ_BODY(flat_dot) }
kernel void tv_g32 (TV_LLVQ_ARGS) { TV_LLVQ_BODY(g32_dot)  }

// ===========================================================================
// The bank-conflict experiment. Everything below exists to answer one
// question and must not leak into the production path until it is answered.
//
// The claim under test (docs/archive/portage-noyau-cuda.md §3.2) is NVIDIA
// arithmetic transposed to Apple: with 32 banks of 4 bytes, lane L reading
// `xs[24·L + j]` hits bank `(24L + j) mod 32`, and `24L mod 32` takes only
// four distinct values over 32 lanes — an 8-way conflict on each of the 24
// reads. **Apple's threadgroup bank geometry is documented nowhere in this
// repository**, and the one hardware fact the code does read rather than
// assume is the SIMD width (`lib.rs`, "read rather than assumed"). So this
// is a hypothesis, and the three kernels below are the instrument.
//
// Three points, not two. Going straight from scalar@24 to float4@28 moves
// the access width AND the stride at once, and the repository's own record
// of A/B discipline says what that costs.
//
//   tv_slot        scalar loads, 24-float blocks   (production)
//   tv_slot_v4     float4 loads, 24-float blocks   (width only)
//   tv_slot_v4p    float4 loads, 28-float blocks   (width + stride)
//
// The FP16 arm gets the same treatment and for a sharper reason: its four
// reads `xs[c..c+3]` are already contiguous and 16-byte aligned, so the
// compiler *may* already fuse them. If it does, FP16 will not move under
// `tv_f16_v4` while LLVQ does, and a ratio computed from that would rise for
// a reason that is not a property of the format. Both arms or neither.
//
// Correctness: none of this changes a value, an order, or a parenthesis.
// The variants must be **bit-identical** to their base, not merely within
// tolerance — and the host asserts exactly that.
// ===========================================================================

constant uint BLOCK_VEC = DIM / 4;          // 6 float4 per 24-weight block

// One slot's signed value. Lifted out of the loop body as a function rather
// than a macro: a statement-expression macro is a GNU extension MSL is not
// obliged to accept, and this has to compile the same way for years.
static inline float slot_v(uint j, uint m1, uint m2, uint m3, uint m4,
                           uint smask, float v0, float v1, float v2,
                           float v3, float v4)
{
    uint bj = 1u << j;
    float v = (m1 & bj) ? v1 : (m2 & bj) ? v2 : (m3 & bj) ? v3
            : (m4 & bj) ? v4 : v0;
    return (smask & bj) ? -v : v;
}

// Slot32 decode reading its activations as float4. The header and mask
// extraction is `slot_dot` verbatim; only the inner loop differs. The four
// accumulators keep their slot assignment (d0 takes slots 0,4,8,…) and the
// final parenthesisation is unchanged, which is what makes bit-identity a
// requirement rather than a hope.
//
// It lives here and not in PAYLOAD_MSL on purpose: `slot_dot` is shared
// verbatim with `bin/matvec`, and touching it would move the single-layer
// number and the whole-model number in the same edit. If the experiment
// wins, promotion is a separate, measurable step.
static inline float slot_dot4(device const uint* words,
                              device const uint* bases,
                              constant ClassRec* tab,
                              constant float* gscale,
                              uint b,
                              threadgroup const float4* xb)
{
    uint g = b >> 5;
    uint base = bases[g];
    uint stride = (bases[g + 1] - base) >> 5;
    uint byte = base + (b & 31u) * stride;

    uint w = byte >> 2;
    uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
         w3 = words[w + 3], w4 = words[w + 4];
    uint sh = (byte & 3u) * 8u;
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    ulong hi = (ulong(w3) << 32) | ulong(w2);

    uint hdr  = uint(lo >> sh) & 0x3ffu;
    uint id   = hdr & 0x1ffu;
    uint gain = hdr >> 9;
    uint fs = sh + 10u;
    ulong pay_lo = (lo >> fs) | (hi << (64u - fs));
    ulong pay_hi = (hi >> fs) | (ulong(w4) << (64u - fs));

    constant ClassRec &r = tab[id];
    uint smask = uint(pay_lo) & 0xffffffu;
    uint nlev = r.len;
    uint m1 = (nlev > 1) ? ext24(pay_lo, pay_hi, 24u) : 0u;
    uint m2 = (nlev > 2) ? ext24(pay_lo, pay_hi, 48u) : 0u;
    uint m3 = (nlev > 3) ? ext24(pay_lo, pay_hi, 72u) : 0u;
    uint m4 = (nlev > 4) ? ext24(pay_lo, pay_hi, 96u) : 0u;
    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2],
          v3 = r.vals[3], v4 = r.vals[4];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (uint q = 0; q < BLOCK_VEC; ++q) {
        float4 xv = xb[q];
        uint j = q << 2;
        d0 = fma(slot_v(j + 0, m1,m2,m3,m4, smask, v0,v1,v2,v3,v4), xv.x, d0);
        d1 = fma(slot_v(j + 1, m1,m2,m3,m4, smask, v0,v1,v2,v3,v4), xv.y, d1);
        d2 = fma(slot_v(j + 2, m1,m2,m3,m4, smask, v0,v1,v2,v3,v4), xv.z, d2);
        d3 = fma(slot_v(j + 3, m1,m2,m3,m4, smask, v0,v1,v2,v3,v4), xv.w, d3);
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[gain];
}

// `TV` is the block pitch in float4 units: BLOCK_VEC for the dense tile,
// TILE_VEC_PAD for the padded one. The fill becomes a scatter; the global
// read stays contiguous across lanes, so it stays coalesced.
#define TV_SLOT4_BODY(TV)                                                     \
    uint row = gid >> 5;                                                      \
    uint b0r = row * P.nblocks;                                               \
    float acc = 0.0f;                                                         \
    uint ntiles = (P.nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;               \
    for (uint t = 0; t < ntiles; ++t) {                                       \
        uint jlo = t * TILE_BLOCKS;                                           \
        uint jhi = min(jlo + TILE_BLOCKS, P.nblocks);                         \
        uint nvec = (jhi - jlo) * BLOCK_VEC;                                  \
        device const float4* x4 =                                             \
            (device const float4*)(x + jlo * DIM);                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint q = tid; q < nvec; q += tgs) {                              \
            uint blk = q / BLOCK_VEC;                                         \
            xs[blk * (TV) + (q - blk * BLOCK_VEC)] = x4[q];                   \
        }                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint j = jlo + lane; j < jhi; j += 32) {                         \
            acc += slot_dot4(words, bases, tab, gscale, b0r + j,              \
                             xs + (j - jlo) * (TV));                          \
        }                                                                     \
    }                                                                         \
    acc = simd_sum(acc);                                                      \
    if (lane == 0) {                                                          \
        float tv = 0.0f;                                                      \
        uint tc0 = P.nblocks * DIM;                                           \
        for (uint i = 0; i < P.tail_w; ++i) {                                 \
            tv += tail[row * P.tail_w + i] * x[tc0 + i];                      \
        }                                                                     \
        y[row] = acc * rscale[row] + tv;                                      \
    }

// Same nine bindings as TV_LLVQ_ARGS, but `xs` is typed float4 rather than
// cast from float4*: the 16-byte alignment becomes a property of the type
// instead of an assumption nothing in the repository writes down.
#define TV_SLOT4_ARGS                                                         \
    device const uint*   words  [[buffer(0)]],                                \
    device const uint*   bases  [[buffer(1)]],                                \
    constant ClassRec*   tab    [[buffer(2)]],                                \
    constant float*      gscale [[buffer(3)]],                                \
    device const float*  rscale [[buffer(4)]],                                \
    device const float*  tail   [[buffer(5)]],                                \
    device const float*  x      [[buffer(6)]],                                \
    device float*        y      [[buffer(7)]],                                \
    constant Params&     P      [[buffer(8)]],                                \
    threadgroup float4*  xs     [[threadgroup(0)]],                           \
    uint gid  [[thread_position_in_grid]],                                    \
    uint tid  [[thread_index_in_threadgroup]],                                \
    uint tgs  [[threads_per_threadgroup]],                                    \
    uint lane [[thread_index_in_simdgroup]]

kernel void tv_slot_v4 (TV_SLOT4_ARGS) { TV_SLOT4_BODY(BLOCK_VEC)    }
kernel void tv_slot_v4p(TV_SLOT4_ARGS) { TV_SLOT4_BODY(TILE_VEC_PAD) }

// FP16 with the same load width, and no padding: its addresses are `4L`, so
// the stride is already 4 floats and only the access width is in question.
kernel void tv_f16_v4(device const half*   w   [[buffer(0)]],
                      device const float*  x   [[buffer(1)]],
                      device float*        y   [[buffer(2)]],
                      constant Params&     P   [[buffer(3)]],
                      threadgroup float4*  xs  [[threadgroup(0)]],
                      uint gid  [[thread_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint tgs  [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    float acc = 0.0f;
    uint ntiles = (P.d_in + TILE_COLS - 1u) / TILE_COLS;
    for (uint t = 0; t < ntiles; ++t) {
        uint c0 = t * TILE_COLS;
        uint n = min(TILE_COLS, P.d_in - c0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float4* x4 = (device const float4*)(x + c0);
        for (uint q = tid; q < (n >> 2); q += tgs) xs[q] = x4[q];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half4* w4 =
            (device const half4*)(w + row * P.d_in + c0);
        for (uint j = lane; j < (n >> 2); j += 32) {
            float4 wv = float4(w4[j]);
            float4 xv = xs[j];
            acc += wv.x * xv.x + wv.y * xv.y
                 + wv.z * xv.z + wv.w * xv.w;
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    d_in: u32,
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
}

/// One transcoded layout of one matrix, resident on the GPU.
struct Lay {
    words: metal::Buffer,
    bases: metal::Buffer,
    /// Everything a decode of this matrix reads: payload, base table, row
    /// scales, tail. The same formula for every layout — that is what makes
    /// the three rows of the table comparable.
    bytes: u64,
}

/// Everything one matrix needs on the GPU, plus what it costs.
struct Mat {
    name: String,
    d_out: usize,
    params: metal::Buffer,
    // Shared by the LLVQ arms.
    gscale: metal::Buffer,
    rscale: metal::Buffer,
    tail: metal::Buffer,
    slot: Lay,
    flat: Lay,
    g32: Lay,
    // FP16 side.
    w16: metal::Buffer,
    f16_bytes: u64,
    // Verification.
    y_ref: Vec<f64>,
    y16_ref: Vec<f64>,
    scale: Vec<f64>,
}

/// Which kernel, reading which bytes, against which reference.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    F16,
    Slot,
    Flat,
    G32,
    /// Bank-conflict variants. Same values, same order, same parentheses as
    /// their base — only the staging geometry moves.
    F16V4,
    SlotV4,
    SlotV4P,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::F16 => "FP16 (half4, scalaire)",
            Arm::Slot => "LLVQ Slot32 (scalaire@24)",
            Arm::Flat => "LLVQ Flat32",
            Arm::G32 => "LLVQ Grouped32",
            Arm::F16V4 => "FP16 (half4, float4)",
            Arm::SlotV4 => "LLVQ Slot32 (float4@24)",
            Arm::SlotV4P => "LLVQ Slot32 (float4@28)",
        }
    }

    fn kernel_name(self) -> &'static str {
        match self {
            Arm::F16 => "tv_f16",
            Arm::Slot => "tv_slot",
            Arm::Flat => "tv_flat",
            Arm::G32 => "tv_g32",
            Arm::F16V4 => "tv_f16_v4",
            Arm::SlotV4 => "tv_slot_v4",
            Arm::SlotV4P => "tv_slot_v4p",
        }
    }

    fn lay(self, m: &Mat) -> Option<&Lay> {
        match self {
            Arm::F16 | Arm::F16V4 => None,
            Arm::Slot | Arm::SlotV4 | Arm::SlotV4P => Some(&m.slot),
            Arm::Flat => Some(&m.flat),
            Arm::G32 => Some(&m.g32),
        }
    }

    fn bytes(self, m: &Mat) -> u64 {
        self.lay(m).map_or(m.f16_bytes, |l| l.bytes)
    }

    fn reference(self, m: &Mat) -> &[f64] {
        match self {
            Arm::F16 | Arm::F16V4 => &m.y16_ref,
            _ => &m.y_ref,
        }
    }

    /// Threadgroup bytes this arm's staging needs.
    fn tgmem(self) -> u64 {
        let floats = match self {
            Arm::SlotV4P => TILE_BLOCKS * TILE_STRIDE,
            _ => TILE_BLOCKS * DIM,
        };
        (floats * 4) as u64
    }

    /// The arm this one is a staging variant of, and whether it must agree
    /// **bit for bit**.
    ///
    /// The strict case is the assertion the experiment rests on: a variant
    /// that merely lands under the tolerance could be reading the wrong
    /// slots and compensating, and one that is accidentally *the same code* —
    /// a padded stride silently still 24 — would pass a tolerance check
    /// trivially and make the bench measure one kernel twice.
    ///
    /// `SlotV4`/`SlotV4P` are strict because `slot_dot4` writes its inner
    /// loop as explicit `fma` into four named accumulators with an explicit
    /// final parenthesisation: the expression is pinned, so only the staging
    /// geometry can move. `F16V4` is **not**, and that is a measurement, not
    /// an excuse: `tv_f16` writes `wv.x*xs[c] + wv.y*xs[c+1] + …`, four
    /// products joined by `+`, which the compiler may contract into FMA as
    /// it pleases — and it demonstrably contracts the vector form
    /// differently from the scalar one. The disagreement is printed in ulps
    /// so the size of the confound is on the table rather than assumed away.
    fn twin(self) -> Option<(Arm, bool)> {
        match self {
            Arm::F16V4 => Some((Arm::F16, false)),
            Arm::SlotV4 | Arm::SlotV4P => Some((Arm::Slot, true)),
            _ => None,
        }
    }
}

/// `max |got − want| / Σ|wᵢxᵢ|` over the rows.
fn worst_error(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .zip(scale)
        .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.max(1e-12))
        .fold(0.0, f64::max)
}

/// Min, median and max of a set of timings — the spread the point value hides.
fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap()));
    if !std::path::Path::new(&path).exists() {
        return Err(format!(
            "{path} : introuvable.\n  \
             Le fichier publié est Pier-Jean/Qwen3-4B-LLVQ-2bit sur Hugging Face ; \
             passer son chemin en argument."
        ));
    }

    // The tile constants are emitted by the host, never written twice. Two
    // independent copies of `TILE_BLOCKS` — one Rust, one MSL — were what
    // let the staging size and the kernel's tiling drift apart unnoticed.
    const {
        assert!(TILE_STRIDE.is_multiple_of(4), "float4 staging needs a multiple of 4");
        // Strictly greater, not `>=`. At 24 the padded arm becomes the dense
        // arm, both stay bit-identical to `tv_slot`, every assertion passes,
        // and the bench silently measures one kernel twice while reporting
        // two rows. That is the failure mode this repository has documented
        // three times: a parameter tested only at its neutral value.
        assert!(TILE_STRIDE > DIM, "the padded tile must actually pad");
    }
    let src = format!(
        "{}\nconstant uint TILE_BLOCKS = {TILE_BLOCKS}u;\n\
         constant uint TILE_VEC_PAD = {}u;\n{}",
        llvq_metal::PAYLOAD_MSL,
        TILE_STRIDE / 4,
        SRC
    );
    let arms = [
        Arm::F16,
        Arm::Slot,
        Arm::Flat,
        Arm::G32,
        Arm::F16V4,
        Arm::SlotV4,
        Arm::SlotV4P,
    ];
    for a in arms {
        assert!(
            a.tgmem() <= 32 * 1024,
            "{}: {} B of threadgroup memory exceeds Metal's 32 KB",
            a.name(),
            a.tgmem()
        );
        assert_eq!(a.tgmem() % 16, 0, "{}: float4 staging wants 16-byte", a.name());
    }
    let kernels: Vec<llvq_metal::Kernel> = arms
        .iter()
        .map(|a| llvq_metal::Kernel::new(&src, a.kernel_name()))
        .collect::<Result<_, _>>()?;
    println!("GPU : {}\n", kernels[0].device_name());

    let fd = FastDecoder::new();
    let recs = llvq_metal::gpu_class_table(&fd);
    let btab = kernels[0].buffer(&recs);

    // One activation, long enough for the widest layer. Which vector it is
    // does not change the cost; that it is the *same* one for every arm does
    // matter, and it is.
    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(0x6_7451);
    let xmax: Vec<f32> = (0..16384).map(|_| rng.next_gaussian() as f32).collect();
    let bx = kernels[0].buffer(&xmax);
    let mut by_len = 0usize;

    println!("Chargement et transcodage des {} matrices…", h.matrices);
    let t0 = Instant::now();
    let mut mats: Vec<Mat> = Vec::with_capacity(h.matrices as usize);
    let mut n_weights = 0u64;

    for mi in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        let (d_out, d_in) = (m.d_out, m.d_in);
        let nblocks = d_in / DIM;
        let tail_w = d_in % DIM;
        assert_eq!(d_in % 4, 0, "{}: the FP16 kernel loads half4", m.name);
        assert_eq!(d_out % 8, 0, "{}: rows must fill whole threadgroups", m.name);
        assert!(d_in <= xmax.len(), "{}: d_in {d_in} exceeds the activation", m.name);
        // Every decoder hard-codes one gain bit (`hdr >> 9`, `fs = sh + 10`).
        // A file with a different gain width would transcode into a coherent
        // stream and decode into garbage, and the tolerance above is not a
        // guard against that. Fail here, before 120 s of prologue.
        assert_eq!(
            m.centroids.len(),
            2,
            "{}: every shader hard-codes a 1-bit gain, this matrix has {} centroids",
            m.name,
            m.centroids.len()
        );
        let gain_bits = m.centroids.len().next_power_of_two().trailing_zeros();
        let table = ClassTable::new(&fd, gain_bits);
        // The five-word read of `slot_dot`/`flat_dot` starts at `byte >> 2`
        // and every group base is a multiple of 32, so the byte shift is at
        // most 24. This is the inequality that window rests on.
        assert!(
            24 + table.worst_width_slot() <= 160,
            "{}: the class table overflows the kernel's five-word window",
            m.name
        );

        let tr = |lay| {
            transcode(&fd, &table, &m.indices, &m.gains, lay).map_err(|e| e.to_string())
        };
        let rt_slot = tr(Layout::Slot32)?;
        let rt_flat = tr(Layout::Flat32)?;
        let rt_g32 = tr(Layout::Grouped32)?;
        n_weights += (d_out * d_in) as u64;
        by_len = by_len.max(d_out);

        // Rows are independent: decode, cross-check the three layouts,
        // round to f16 and build the reference in one parallel pass.
        let mut w16 = vec![0u16; d_out * d_in];
        let mut y_ref = vec![0.0f64; d_out];
        let mut y16_ref = vec![0.0f64; d_out];
        let mut scale = vec![0.0f64; d_out];
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let chunk = d_out.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, (((w16c, yc), y16c), sc)) in w16
                .chunks_mut(chunk * d_in)
                .zip(y_ref.chunks_mut(chunk))
                .zip(y16_ref.chunks_mut(chunk))
                .zip(scale.chunks_mut(chunk))
                .enumerate()
            {
                let (m, table, xmax) = (&m, &table, &xmax);
                let (rt_slot, rt_flat, rt_g32) = (&rt_slot, &rt_flat, &rt_g32);
                s.spawn(move || {
                    let mut wrow = vec![0.0f64; d_in];
                    for lr in 0..yc.len() {
                        let row = ci * chunk + lr;
                        wrow.fill(0.0);
                        for p in 0..nblocks {
                            let b = row * nblocks + p;
                            let (pt, gain) = rt_slot.decode_block(table, b);
                            // The three layouts are three arrangements of
                            // the same bits. Nothing pinned them against
                            // each other on real blocks until now; the
                            // synthetic round-trips in `llvq-artifact` walk
                            // indices the sealed cap-12 file never contains.
                            debug_assert_layouts_agree(
                                &m.name, b, (pt, gain),
                                rt_flat.decode_block(table, b),
                                rt_g32.decode_block(table, b),
                            );
                            if let Some(shell) = Leech::shell_index(&pt).filter(|&s| s > 0) {
                                let sc = m.centroids[gain as usize] * m.row_scales[row]
                                    / ((16 * shell) as f64).sqrt();
                                for (i, &v) in pt.iter().enumerate() {
                                    wrow[p * DIM + i] = v as f64 * sc;
                                }
                            }
                        }
                        for t in 0..tail_w {
                            wrow[nblocks * DIM + t] = m.tail[row * tail_w + t];
                        }
                        let (mut a, mut bb, mut ss) = (0.0, 0.0, 0.0);
                        for c in 0..d_in {
                            let xv = xmax[c] as f64;
                            let wv = wrow[c];
                            let hb = llvq_metal::f16_bits(wv as f32);
                            w16c[lr * d_in + c] = hb;
                            a += wv * xv;
                            bb += llvq_metal::f16_to_f64(hb) * xv;
                            ss += (wv * xv).abs();
                        }
                        yc[lr] = a;
                        y16c[lr] = bb;
                        sc[lr] = ss;
                    }
                });
            }
        });

        let k = &kernels[0];
        let params = [Params {
            d_in: d_in as u32,
            d_out: d_out as u32,
            nblocks: nblocks as u32,
            tail_w: tail_w as u32,
        }];
        let gscale: Vec<f32> = m.centroids.iter().map(|&c| c as f32).collect();
        let rscale: Vec<f32> = m.row_scales.iter().map(|&s| s as f32).collect();
        let tailf: Vec<f32> = m.tail.iter().map(|&t| t as f32).collect();
        let upload = |rt: &RuntimeBlocks| {
            let mut words = rt.data.clone();
            // The five-word read of the last block runs past the stream.
            // Benign on Metal; on CUDA the same read is a fault that kills
            // the context. The padding is not counted as bytes read.
            words.extend_from_slice(&[0u8; 20]);
            Lay {
                words: k.buffer(&words),
                bases: k.buffer(&rt.bases),
                bytes: rt.data.len() as u64
                    + rt.bases.len() as u64 * 4
                    + (d_out * tail_w) as u64 * 4
                    + d_out as u64 * 4,
            }
        };

        mats.push(Mat {
            name: m.name.clone(),
            d_out,
            params: k.buffer(&params),
            gscale: k.buffer(&gscale),
            rscale: k.buffer(&rscale),
            tail: k.buffer(if tailf.is_empty() { &[0.0f32] } else { &tailf }),
            slot: upload(&rt_slot),
            flat: upload(&rt_flat),
            g32: upload(&rt_g32),
            w16: k.buffer(&w16),
            f16_bytes: (d_out * d_in * 2) as u64,
            y_ref,
            y16_ref,
            scale,
        });
        if mi % 36 == 0 {
            println!(
                "  {mi:>3}/{}  {} ({d_out}×{d_in})",
                h.matrices,
                mats[mats.len() - 1].name
            );
        }
    }
    let load_s = t0.elapsed().as_secs_f64();
    let by = kernels[0].empty::<f32>(by_len);
    println!(
        "  {} matrices, {:.2} Md de poids, transcodées (3 layouts) en {load_s:.0} s\n",
        mats.len(),
        n_weights as f64 / 1e9
    );

    let tg = 256u64;
    let bind = |enc: &metal::ComputeCommandEncoderRef, m: &Mat, arm: Arm| {
        match arm.lay(m) {
            None => {
                enc.set_buffer(0, Some(&m.w16), 0);
                enc.set_buffer(1, Some(&bx), 0);
                enc.set_buffer(2, Some(&by), 0);
                enc.set_buffer(3, Some(&m.params), 0);
            }
            Some(l) => {
                enc.set_buffer(0, Some(&l.words), 0);
                enc.set_buffer(1, Some(&l.bases), 0);
                enc.set_buffer(2, Some(&btab), 0);
                enc.set_buffer(3, Some(&m.gscale), 0);
                enc.set_buffer(4, Some(&m.rscale), 0);
                enc.set_buffer(5, Some(&m.tail), 0);
                enc.set_buffer(6, Some(&bx), 0);
                enc.set_buffer(7, Some(&by), 0);
                enc.set_buffer(8, Some(&m.params), 0);
            }
        }
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, arm.tgmem());
    };

    // ---- verify every matrix, every row, every arm, before any timing ----
    println!("Vérification des {} matrices contre la référence CPU f64…", mats.len());
    let mut worst = vec![(0.0f64, String::new()); arms.len()];
    let mut twin_err = vec![0.0f64; arms.len()];
    let idx = |a: Arm| arms.iter().position(|&x| x == a).unwrap();
    for m in &mats {
        let threads = (m.d_out * 32) as u64;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(arms.len());
        for (ai, (&arm, k)) in arms.iter().zip(&kernels).enumerate() {
            k.dispatch(threads, tg, |e: &metal::ComputeCommandEncoderRef| bind(e, m, arm));
            let got: Vec<f32> = unsafe { k.read(&by, m.d_out) };
            let e = worst_error(&got, arm.reference(m), &m.scale);
            assert!(
                e < TOL,
                "{} / {}: off by {e:.2e}·Σ|w·x|",
                m.name,
                arm.name()
            );
            if e > worst[ai].0 {
                worst[ai] = (e, m.name.clone());
            }
            if let Some((t, strict)) = arm.twin() {
                let base = &out[idx(t)];
                // Reported on the same scale as every other error here,
                // Σ|wᵢxᵢ|. Subtracting bit patterns would be meaningless:
                // rows whose dot product sits near zero can land on either
                // side of it, and two adjacent floats across zero differ by
                // 2^31 in bits while being numerically identical.
                let d = got
                    .iter()
                    .zip(base)
                    .zip(&m.scale)
                    .map(|((a, b), &s)| (*a as f64 - *b as f64).abs() / s.max(1e-12))
                    .fold(0.0f64, f64::max);
                if strict {
                    let exact = got
                        .iter()
                        .zip(base)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        exact,
                        "{} / {}: differs from {} (up to {d:.2e}·Σ|w·x|) — this variant \
                         changes no value, no order and no parenthesis, so it must not",
                        m.name,
                        arm.name(),
                        t.name()
                    );
                }
                twin_err[ai] = twin_err[ai].max(d);
            }
            out.push(got);
        }
    }
    let rows: usize = mats.iter().map(|m| m.d_out).sum();
    println!("  {rows} lignes sur {} matrices, seuil {TOL:.0e} :", mats.len());
    for ((&arm, (e, name)), &terr) in arms.iter().zip(&worst).zip(&twin_err) {
        let twin = match arm.twin() {
            None => String::new(),
            Some((t, true)) => format!("  — identique à {} au bit près", t.name()),
            Some((t, false)) => {
                format!("  — vs {} : {terr:.1e}·Σ|w·x| (contraction FMA, non épinglée)", t.name())
            }
        };
        println!(
            "    {:<26} pire erreur {e:.1e}·Σ|w·x|  ({name}){twin}",
            arm.name()
        );
    }

    // ---- rounds of one command buffer each, arms interleaved ----
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for rep in 0..ROUNDS {
        for (ai, (&arm, k)) in arms.iter().zip(&kernels).enumerate() {
            let t = k.dispatch_batch(
                |enc, i| {
                    let m = &mats[i];
                    bind(enc, m, arm);
                    ((m.d_out * 32) as u64, tg)
                },
                mats.len(),
            );
            if rep >= WARMUP_ROUNDS {
                times[ai].push(t.seconds);
            }
        }
    }
    // The ratio is formed **round by round**, then summarised. Dividing one
    // arm's minimum by another's mixes two rounds that never coexisted, and
    // the resulting spread describes nothing that happened.
    let ratios: Vec<(f64, f64, f64)> = (0..arms.len())
        .map(|ai| {
            spread(
                times[0]
                    .iter()
                    .zip(&times[ai])
                    .map(|(a, b)| a / b)
                    .collect(),
            )
        })
        .collect();
    let stats: Vec<(f64, f64, f64)> = times.into_iter().map(spread).collect();

    println!("\nUN TOKEN — les 252 projections, un command buffer par bras, mémoire froide");
    println!(
        "  {} rounds, {WARMUP_ROUNDS} jetés, TOUS les bras à chaque round, même ordre",
        ROUNDS
    );
    println!("  {}", "-".repeat(89));
    println!(
        "  {:<27}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>23}",
        "format", "min ms", "méd ms", "max ms", "Go lus", "b/poids", "Go/s",
        "vs FP16 méd [min–max]"
    );
    for ((&arm, &(mn, md, mx)), &(rlo, rmd, rhi)) in
        arms.iter().zip(&stats).zip(&ratios)
    {
        let bytes: u64 = mats.iter().map(|m| arm.bytes(m)).sum();
        let is_f16 = arm.lay(&mats[0]).is_none();
        println!(
            "  {:<27}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9}{:>9.0}{:>23}",
            arm.name(),
            mn * 1e3,
            md * 1e3,
            mx * 1e3,
            bytes as f64 / 1e9,
            if is_f16 {
                "16.000".to_string()
            } else {
                format!("{:.3}", bytes as f64 * 8.0 / n_weights as f64)
            },
            bytes as f64 / mn / 1e9,
            format!("{rmd:.2}× [{rlo:.2}–{rhi:.2}]")
        );
    }
    println!("  {}", "-".repeat(89));
    println!(
        "  Le rapport est formé round par round puis résumé : médiane, puis plage\n  \
         sur les {} rounds gardés. Un min divisé par un min mêlerait deux rounds\n  \
         qui n'ont jamais coexisté.",
        ROUNDS - WARMUP_ROUNDS
    );

    // The tied lm_head is unquantized in this artifact and read once per
    // token by every arm — the same constant added to each, and what caps
    // the end-to-end ratio.
    let head_bytes = 389_070_848f64 * 2.0;
    let f16_bytes: u64 = mats.iter().map(|m| m.f16_bytes).sum();
    let bw = f16_bytes as f64 / stats[0].0; // this machine's measured streaming rate
    let head_s = head_bytes / bw;
    println!("\n  Débit d'un pas de décodage (projections + lm_head f16 non quantifié) :");
    println!("  {}", "-".repeat(89));
    for (&arm, &(mn, _, _)) in arms.iter().zip(&stats) {
        let total = mn + head_s;
        println!(
            "  {:<27}{:>8.2} ms{:>10.1} tok/s{:>14}",
            arm.name(),
            total * 1e3,
            1.0 / total,
            format!("(dont lm_head {:.2} ms)", head_s * 1e3)
        );
    }
    println!(
        "\n  Le lm_head lié ({:.0} M poids, f16) n'est pas quantifié dans cet artefact\n  \
         et coûte la même chose à tous les bras : c'est lui qui plafonne le rapport\n  \
         de bout en bout. Attention, normes et activations ne sont pas mesurées ici.",
        389_070_848f64 / 1e6
    );
    Ok(())
}

/// The three layouts must decode to the same point and the same gain.
///
/// Not a `debug_assert!` despite the name — it runs in release, because
/// release is the only configuration this bench is ever built in and the
/// property is the one that lets a single f64 reference serve three kernels.
#[allow(clippy::type_complexity)]
fn debug_assert_layouts_agree(
    name: &str,
    b: usize,
    slot: ([i32; DIM], u32),
    flat: ([i32; DIM], u32),
    g32: ([i32; DIM], u32),
) {
    assert!(
        slot == flat && slot == g32,
        "{name}: block {b} decodes differently across layouts\n  \
         slot {:?}\n  flat {:?}\n  g32  {:?}",
        slot,
        flat,
        g32
    );
}
