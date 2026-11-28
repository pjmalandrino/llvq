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
    /// `Σ d_out`, i.e. the 1,105,920 rows the CUDA bench verifies.
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
    assert_eq!(EMBED_GROUP_BITS, 32.0, "f16 scale + f16 bias");
    assert!(
        (embed_q8_bpw() - 8.5).abs() < 1e-12,
        "q8 g64 is 8.5 b/param, not {}",
        embed_q8_bpw()
    );
    // The 4B's tied table: 388,956,160 values, 40 groups per row of
    // 2560, 4 bytes per group — the 413.3 MB `q8_device_bytes` announces.
    let (packed, scales) = (388_956_160u64, 151_936u64 * 40 * 4);
    assert!(
        ((packed + scales) as f64 * 8.0 / packed as f64 - embed_q8_bpw()).abs() < 1e-9,
        "the rate and the bytes the q8 path announces disagree"
    );
}

fn carried_4b() -> Shapes {
    Shapes {
        weights: qwen3_4b::WEIGHTS,
        tail_weights: qwen3_4b::TAIL_WEIGHTS,
        rows: qwen3_4b::ROWS,
        // These fixtures pin the KERNEL accounting, which does not read
        // this field. An explicit zero rather than a
        // `..Default::default()`, so a future test that used it sees it
        // wrong.
        padded_blocks: 0,
    }
}

/// Walk one record's fields at the offsets the format froze, and check
/// that the posts add up to the stride this bench bills.
///
/// Asserting `112` alone would survive a format that moved a field; this
/// walks `PLANES14_SMASK_BIT`, `PLANES14_PLANE_BIT` and
/// `PLANES14_PAD_BIT` from `llvq_artifact::runtime` and demands they be
/// consistent with `class 9 | gain 1 | signs 24 | 3 × plane 24 | 6 of
/// padding`. Any field that moves, widens or disappears lands here.
#[test]
fn one_planes14_record_is_112_bits_field_by_field() {
    // Header: class then a one-bit gain — the field the two Planes
    // layouts freeze, and the reason they refuse a file with more.
    assert_eq!(PLANES14_SMASK_BIT, CLASS_BITS + 1, "class 9 b + gain 1 b");
    // Sign mask in slot space: 24 bits, not `nonzero`.
    assert_eq!(PLANES14_PLANE_BIT[0], PLANES14_SMASK_BIT + DIM as u64);
    // Three contiguous 24-bit planes.
    assert_eq!(PLANES14_PLANE_BIT.len(), 3);
    for w in PLANES14_PLANE_BIT.windows(2) {
        assert_eq!(w[1] - w[0], DIM as u64, "a plane is 24 bits");
    }
    let last = PLANES14_PLANE_BIT[PLANES14_PLANE_BIT.len() - 1];
    assert_eq!(PLANES14_PAD_BIT, last + DIM as u64);
    assert_eq!(PLANES14_PAD_BIT, 106, "9 + 1 + 24 + 72");
    // And the stride closes the record to a whole number of bytes.
    let stride = PLANES14_BYTES as u64 * 8;
    assert_eq!(stride - PLANES14_PAD_BIT, 6, "6 padding bits");
    assert_eq!(planes14_bits(1), stride);
    // Class-blind by construction: a thousand blocks cost a thousand
    // strides whatever they hold, origin included.
    assert_eq!(planes14_bits(1_000), 112_000);
}

/// Same walk for `Planes12x`: `9 + 1 + 24 + 2·24 = 82` payload bits, 14
/// of padding, `12·8 = 96`; one exception adds `4 + 14 = 18` bytes.
#[test]
fn one_planes12x_record_is_96_bits_and_one_exception_144_field_by_field() {
    assert_eq!(PLANES12X_SMASK_BIT, CLASS_BITS + 1, "class 9 b + gain 1 b");
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
    assert_eq!(stride - PLANES12X_PAD_BIT, 14, "14 padding bits");
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
    assert!(!is_planes12x_exception(1), "the origin is not an exception");
    for l in 2..=PLANES12X_LEVEL_CAP {
        assert!(!is_planes12x_exception(l), "L = {l} fits in two planes");
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
            "the two routes disagree at L = {l}"
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
        "Planes14: {p14:.4} against 4.804 at the CUDA bench"
    );
    assert!(
        (p12 - 4.342).abs() < 5e-4,
        "Planes12x: {p12:.4} against 4.342 at the CUDA bench"
    );
    // And the exception rate the arbitration rests on.
    let rate = 100.0 * qwen3_4b::EXCEPTIONS as f64 / qwen3_4b::BLOCKS as f64;
    assert!((rate - 3.3824).abs() < 5e-4, "exception rate {rate:.4} %");
}

/// **The E1c acceptance test.** Both variants priced from block counts
/// alone, against the values `docs/archive/spec-memoire-extreme-2026-08-12.md`
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
    assert!((e14 - 4.5551).abs() < 5e-4, "E1c14: {e14:.4} against 4.5551");
    assert!((e12 - 3.7618).abs() < 5e-4, "E1c12: {e12:.4} against 3.7618");
    // And what each buys over the layout it replaces — the figure the
    // bench's admission criteria are stated against.
    let p14 = c.kernel_bpw(planes14_bits(qwen3_4b::BLOCKS));
    let p12 = c.kernel_bpw(planes12x_bits(qwen3_4b::BLOCKS, qwen3_4b::EXCEPTIONS));
    assert!((p14 - e14 - 0.2489).abs() < 1e-3, "E1c14 saves {:.4}", p14 - e14);
    assert!((p12 - e12 - 0.5806).abs() < 1e-3, "E1c12 saves {:.4}", p12 - e12);
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
    assert_eq!(d0, d9, "the exception term is not identical on both sides");
}

/// A block count that is not a multiple of 32 pays a whole trailing
/// group, and the bench must charge it — otherwise a small matrix would
/// be priced below what it uploads.
///
/// The published 4B happens to divide exactly (150,681,600 = 32 ×
/// 4,708,800), which is precisely why this needs its own test: the
/// acceptance test above would never exercise the rounding.
#[test]
fn a_partial_trailing_group_is_charged() {
    assert_eq!(qwen3_4b::BLOCKS % E1C_GROUP as u64, 0, "the 4B divides exactly");
    let full = e1c14_bits(32);
    assert_eq!(e1c14_bits(1), full, "one block pays a whole group");
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
        "the Planes14 payload is exactly 112/24, not {payload}"
    );
    // Same numerator, whole-matrix denominator: strictly smaller.
    assert!(bits as f64 / (c.weights as f64) < payload);
    // Side data restores it and overshoots — the 4.804 above.
    assert_eq!(
        c.side_bits(),
        (qwen3_4b::TAIL_WEIGHTS + qwen3_4b::ROWS) * 32,
        "f32 tail + f32 row scales, nothing else"
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
    // Planes14 on the 8B shapes: 288,571,392 blocks, 20,054,016 tail
    // weights, 1,400,832 rows.
    let c = Shapes {
        weights: LIN,
        tail_weights: 20_054_016,
        rows: 1_400_832,
        padded_blocks: 0, // not read by this test, see `carried_4b`
    };
    assert_eq!((LIN - c.tail_weights) % DIM as u64, 0);
    let lin = c.kernel_bpw(planes14_bits((LIN - c.tail_weights) / DIM as u64));

    let f16 = model_bpw(&[(LIN, lin), (EMBED, 16.0)]);
    let q8 = model_bpw(&[(LIN, lin), (EMBED, embed_q8_bpw())]);
    assert!((f16 - 6.461).abs() < 1e-3, "embed f16: {f16:.3} against 6.461");
    assert!((q8 - 5.323).abs() < 2e-3, "embed q8: {q8:.3} against 5.323");
    // Sanity on the direction: carrying 15.2 % of the model in f16 costs
    // more per parameter than the linears do.
    assert!(f16 > lin && q8 > lin);
}

/// The pad between the file's Tetra stream and the card's, on the Tetra
/// records of the sealed 4B: q, k, gate and up at `d_in = 2560` in all 36
/// layers, down at `d_in = 9728` in 24 of them. The shapes are pinned by
/// the 946,176 rows and 14,647,296 tail weights `rtbits` printed for that
/// file (`docs/mesures/sealed-4b-2026-09-23-brut/rtbits-sealed.txt`). A
/// row of 106 blocks is 636 bytes of words in a 640-byte stride, a row of
/// 405 is 2,430 in 2,432: 884,736 rows × 32 bits + 61,440 × 16 bits =
/// 29,294,592 bits, computed by hand.
#[test]
fn the_card_stream_adds_the_tetra48_row_pad_to_the_file_stream() {
    let word = llvq_search::tetra::WORD_BITS as u64;
    let file_bits = |d_out: usize, d_in: usize| (d_out * (d_in / DIM)) as u64 * word;
    // One row at each width: 4 bytes of pad, then 2.
    assert_eq!(tetra_card_stream_bits(1, 2560) - file_bits(1, 2560), 32);
    assert_eq!(tetra_card_stream_bits(1, 9728) - file_bits(1, 9728), 16);

    // (layers, d_out, d_in) of every Tetra record of the sealed 4B.
    let records = [
        (36usize, 4096usize, 2560usize), // q_proj
        (36, 1024, 2560),                // k_proj
        (36, 9728, 2560),                // gate_proj
        (36, 9728, 2560),                // up_proj
        (24, 2560, 9728),                // down_proj
    ];
    let mut shapes = Shapes::default();
    let (mut rows_2560, mut rows_9728) = (0u64, 0u64);
    let (mut file, mut card) = (0u64, 0u64);
    for (layers, d_out, d_in) in records {
        for _ in 0..layers {
            shapes.push(d_out, d_in);
        }
        let rows = (layers * d_out) as u64;
        match d_in {
            2560 => rows_2560 += rows,
            _ => rows_9728 += rows,
        }
        file += layers as u64 * file_bits(d_out, d_in);
        card += layers as u64 * tetra_card_stream_bits(d_out, d_in);
    }
    assert_eq!(shapes.rows, 946_176, "rows of the sealed 4B's Tetra records");
    assert_eq!(shapes.tail_weights, 14_647_296, "tail of the sealed 4B's Tetra records");
    assert_eq!((rows_2560, rows_9728), (884_736, 61_440));
    assert_eq!(card - file, 29_294_592);

    // Carried into the kernel accounting with the file's 84 int4 records
    // (770,703,360 weights at 4.25 b/weight): the file count keeps the
    // 2.5420 `rtbits` printed, and the card count lands on the 2.550 the
    // served door prints for the same file
    // (`docs/mesures/served-v9-2026-11-07-brut/door.txt`).
    let int4_weights = 770_703_360u64;
    let rate = V5Rate {
        lattice: shapes,
        lattice_stream_bits: file,
        lattice_card_stream_bits: card,
        lattice_matrices: 168,
        int4_weights,
        int4_bits: int4_weights * 17 / 4,
        int4_matrices: 84,
    };
    let (at_file, at_card) = (rate.kernel_bpw(false), rate.card_kernel_bpw());
    assert!((at_file - 2.5420).abs() < 1e-4, "file stream: {at_file:.4} against 2.5420");
    assert!((at_card - 2.5501).abs() < 1e-4, "card stream: {at_card:.4} against 2.5501");
}
