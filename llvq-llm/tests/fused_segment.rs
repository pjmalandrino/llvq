//! The host half of the segmented fused path, proved on the development
//! machine — the only half that can be.
//!
//! ## What this file proves, and what it deliberately does not
//!
//! `kernels/tv_planes_seg_h.cu` cannot be validated here. A single-threaded
//! driver reproduces neither `__syncthreads()` nor a warp shuffle, so
//! `tests/host_planes_seg_h.cpp` compiles it and stops there; the kernel is an
//! open claim until a job compares its greedy tokens against the unfused arm.
//!
//! What *is* provable here is the host side, and it is where a wrong answer
//! would actually hide. Fusing q+k+v does not change one line of arithmetic —
//! the same blocks, in the same order, with the same centroids — so every risk
//! is bookkeeping, and every one of the three produces **finite, plausible,
//! wrong numbers** rather than a crash:
//!
//!  * **the pad in the middle of the stream.** `pack_plane_bytes` appends four
//!    bytes to every matrix; concatenating two packed streams verbatim buries
//!    that word inside the flow and reads every block of the second segment
//!    four bytes early, on the wrong shift parity. Nothing crashes: the buffer
//!    is *longer* than the last block's window needs;
//!  * **a wrong `gs_off`**, which hands some rows their neighbour's gain
//!    centroids — a factor of about 2 on those rows and nothing at all on the
//!    others, which any global tolerance waves through;
//!  * **a row order that drifts at a segment boundary**, which a `tail` or an
//!    `rscale` shifted by a constant reproduces exactly.
//!
//! So the assertions below are equalities — field by field, and bit for bit in
//! f64 — never tolerances. This is the `llvq-llm` twin of
//! `llvq-cuda/tests/planes_segment_matches_unfused.rs`, which proves the same
//! properties for the **bench's** concatenation; the two objects are different
//! (raw indices and an f32 tail there, packed `u32` words and a binary16 tail
//! here), and the packing and its padding — the risk this file exists for — are
//! covered by neither `llvq-cuda` nor `llvq-artifact`.

use llvq_artifact::runtime::PLANES14_BYTES;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::{
    check_seg_spans, matrix_side_bytes, planes_payload_words, segment_matrices, splice_planes14,
    FusedGroup, FusedLayout, FusedMatrix, HostStream, Transcoder,
};
use llvq_search::index::N13;

/// The kernel's four-word window, mirrored in Rust — same word indices, same
/// shifts, same field extraction as `planes_fields` in `llvq_planes.cuh`.
///
/// A verbatim twin of `tests/fused_layout.rs`'s mirror, kept as a copy rather
/// than shared: that file pins the packing of one matrix, this one pins the
/// splice of several, and a helper the two of them share would mean a single
/// edit could weaken both at once. Indexing `words` directly is the point — a
/// missing end pad is an out-of-bounds panic here, exactly where it is an
/// illegal address there.
fn planes_fields(words: &[u32], b: usize) -> (usize, u32, u32, [u32; 3]) {
    let byte = PLANES14_BYTES * b;
    let w = byte >> 2;
    let (w0, w1, w2, w3) = (
        words[w] as u64,
        words[w + 1] as u64,
        words[w + 2] as u64,
        words[w + 3] as u64,
    );
    let sh = ((byte & 3) * 8) as u32; // 0 or 16: 14·b mod 4 ∈ {0, 2}
    let lo = w1 << 32 | w0;
    let hi = w3 << 32 | w2;
    let hdr = ((lo >> sh) & 0x3ff) as u32;
    let fs = sh + 10;
    let pay_lo = (lo >> fs) | (hi << (64 - fs));
    let pay_hi = hi >> fs;
    let ext24 = |off: u32| -> u32 {
        if off < 64 {
            (((pay_lo >> off) | (pay_hi << (64 - off))) & 0xffffff) as u32
        } else {
            ((pay_hi >> (off - 64)) & 0xffffff) as u32
        }
    };
    let smask = (pay_lo & 0xffffff) as u32;
    (
        (hdr & 0x1ff) as usize,
        hdr >> 9,
        smask,
        [ext24(24), ext24(48), ext24(72)],
    )
}

/// One projection of the fixture, owned.
struct Part {
    suffix: &'static str,
    d_out: usize,
    indices: Vec<u64>,
    gains: Vec<u32>,
    gscale: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<u16>,
}

/// The shared activation width of the fixture: `128 = 24·5 + 8`, so
/// `nblocks = 5` and `TailPolicy::KeepExact` is exercised with `tail_w = 8`.
const D_IN: usize = 128;
const NBLOCKS: usize = D_IN / DIM;
const TAIL_W: usize = D_IN % DIM;

/// A plausible q/k/v group: 32:8:8 rows, one shared `d_in`, and three
/// **distinct** centroid pairs — without which every `gs_off` would be as good
/// as every other and half of this file would prove nothing.
fn qkv_fixture(seed: u64) -> Vec<Part> {
    let mut rng = SplitMix64::new(seed);
    let shapes = [
        ("self_attn.q_proj", 32usize, [0.625f32, 1.375]),
        ("self_attn.k_proj", 8, [0.5, 2.0]),
        ("self_attn.v_proj", 8, [0.75, 1.125]),
    ];
    shapes
        .iter()
        .map(|&(suffix, d_out, gscale)| Part {
            suffix,
            d_out,
            // A sixth of the blocks are the origin: `id == 0` is the one class
            // whose record is the header alone, so it is the case a stride or
            // an offset error trips over first.
            indices: (0..d_out * NBLOCKS)
                .map(|_| {
                    if rng.next().is_multiple_of(6) {
                        0
                    } else {
                        1 + rng.next() % N13
                    }
                })
                .collect(),
            gains: (0..d_out * NBLOCKS).map(|_| (rng.next() & 1) as u32).collect(),
            gscale,
            rscale: (0..d_out)
                .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
                .collect(),
            tail: (0..d_out * TAIL_W)
                .map(|_| half::f16::from_f32(rng.next_gaussian() as f32).to_bits())
                .collect(),
        })
        .collect()
}

/// The `FusedMatrix` `fused::load` would have built for this part — real
/// stream, real byte accounting, so `segment_matrices`' cross-check against
/// `matrix_side_bytes` is exercised rather than fed a made-up number.
fn to_matrix(tr: &Transcoder, layer: usize, p: &Part, seed: u64) -> FusedMatrix {
    let (stream, payload) = tr
        .stream(&p.indices, &p.gains, p.d_out, NBLOCKS)
        .expect("planes14 transcode");
    FusedMatrix {
        name: format!("model.layers.{layer}.{}.weight", p.suffix),
        d_out: p.d_out,
        d_in: D_IN,
        nblocks: NBLOCKS,
        tail_w: TAIL_W,
        stream,
        gscale: p.gscale,
        rscale: p.rscale.clone(),
        tail: p.tail.clone(),
        rotation: Some((D_IN, seed)),
        bytes: payload + matrix_side_bytes(p.d_out, TAIL_W),
    }
}

/// The message of a refusal `segment_matrices` owes. `expect_err` is not
/// available: neither `FusedMatrix` nor `FusedGroup` is `Debug`, and giving
/// them a derive so a test can print them would put 1.37 GB of payload one
/// `{:?}` away from a log.
fn refusal(r: Result<(Vec<FusedMatrix>, Vec<FusedGroup>), String>, why: &str) -> String {
    match r {
        Ok(_) => panic!("{why}"),
        Err(e) => e,
    }
}

fn words_of(s: &HostStream) -> &[u32] {
    match s {
        HostStream::Planes14 { words } => words,
        _ => panic!("planes14 layout asked, another stream returned"),
    }
}

// ---------------------------------------------------------------------------
// 1. The splice — the stream, the weights, the tail, the bytes
// ---------------------------------------------------------------------------

/// Every block, every row scale and every gain centroid of the fused group is
/// the one the unfused matrix carried, and the fused stream costs the same
/// bytes.
#[test]
fn the_spliced_stream_decodes_to_the_unfused_projections() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let parts = qkv_fixture(0x5_E6A4);
    // The reference streams, transcoded apart, before `segment_matrices` eats
    // the matrices it is handed.
    let unfused: Vec<HostStream> = parts
        .iter()
        .map(|p| tr.stream(&p.indices, &p.gains, p.d_out, NBLOCKS).expect("transcode").0)
        .collect();

    let matrices: Vec<FusedMatrix> = parts
        .iter()
        .map(|p| to_matrix(&tr, 0, p, 0x11))
        .collect();
    let (singles, groups) = segment_matrices(matrices).expect("q/k/v fuse");
    assert!(singles.is_empty(), "the fixture carries only fusable projections");
    assert_eq!(groups.len(), 1);
    let g = &groups[0];

    assert_eq!(g.key, "000.Attn");
    assert_eq!(g.d_out, 48, "the fixture stacks 32 + 8 + 8 rows");
    assert_eq!((g.nblocks, g.tail_w, g.d_in), (NBLOCKS, TAIL_W, D_IN));
    assert_eq!(
        g.parts.iter().map(|p| p.row0).collect::<Vec<_>>(),
        [0, 32, 40],
        "the rows are stacked in rank order"
    );
    assert_eq!(
        g.parts.iter().map(|p| p.rank).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    // Post-conditions of the concatenation, restated where a reader can see
    // them: one offset per ROW, two centroids per PART.
    assert_eq!(g.gs_off.len(), g.d_out);
    assert_eq!(g.gscale.len(), 2 * g.parts.len());
    assert_eq!(g.rscale.len(), g.d_out);
    assert_eq!(g.tail.len(), g.d_out * TAIL_W);

    let fused = words_of(&g.stream);
    for (s, p) in parts.iter().enumerate() {
        let part_words = words_of(&unfused[s]);
        let row0 = g.parts[s].row0;
        assert_eq!(g.parts[s].d_out, p.d_out);

        for r in 0..p.d_out {
            let grow = row0 + r;
            // The gain centroids are the only thing that does not concatenate,
            // so this indirection is the whole point of the segmented kernel.
            let gs = g.gs_off[grow] as usize;
            assert!(
                gs + 1 < g.gscale.len(),
                "row {grow}: gs_off {gs} past the table"
            );
            assert_eq!(gs, 2 * s, "row {grow}: segment {s}'s pair starts at 2·{s}");

            for b in 0..NBLOCKS {
                // (a) The stream, field by field — id, gain, sign mask and the
                // three bit-planes, as a tuple. Not a tolerance: the pad-in-the
                // -middle defect returns a *valid* class from a shifted field.
                let want = planes_fields(part_words, r * NBLOCKS + b);
                let got = planes_fields(fused, grow * NBLOCKS + b);
                assert_eq!(
                    got, want,
                    "segment {s}, row {r} (fused row {grow}), block {b}: the Planes14 \
                     content moved"
                );

                // (b) The scalar every weight of this block is multiplied by,
                // in exact f64. This is the assertion a wrong `gs_off` dies on:
                // swapping two segments' pairs moves some rows by ~2× and
                // leaves the rest untouched, and a global tolerance accepts it
                // on the rows where the two pairs happen to be close.
                let gain = got.1 as usize;
                let got_k = g.gscale[gs + gain] as f64 * g.rscale[grow] as f64;
                let want_k = p.gscale[gain] as f64 * p.rscale[r] as f64;
                assert_eq!(
                    got_k, want_k,
                    "segment {s}, row {r} (fused row {grow}), block {b}: the gain × row \
                     scale factor moved"
                );
            }

            // (c) The tail, in binary16 **bits** — the field that does not
            // transfer from `seg_host.rs` (an `f32` slice there) and that no
            // other test on this side covers.
            let (a, b) = (grow * TAIL_W, r * TAIL_W);
            assert_eq!(
                &g.tail[a..a + TAIL_W],
                &p.tail[b..b + TAIL_W],
                "segment {s}, row {r}: the tail moved"
            );
        }
    }

    // (d) Byte invariance. Structural for Planes14, not measured: a uniform
    // 14-byte stride and no base table means the stream is 14 bytes a block
    // whatever the grouping — unlike Slot32, whose per-group stride is the
    // widest record among 32 blocks and can move when a concatenation regroups
    // across a segment boundary.
    let total_blocks = g.d_out * NBLOCKS;
    let payload_words = planes_payload_words(total_blocks).expect("even block count");
    assert_eq!(
        fused.len(),
        payload_words + 1,
        "payload 14·N/4 plus ONE end-of-stream padding word"
    );
    let unfused_payload: usize = unfused
        .iter()
        .zip(&parts)
        .map(|(s, p)| {
            let n = p.d_out * NBLOCKS;
            assert_eq!(words_of(s).len(), planes_payload_words(n).unwrap() + 1);
            planes_payload_words(n).unwrap()
        })
        .sum();
    assert_eq!(
        payload_words, unfused_payload,
        "the Planes14 fusion changed the byte total: the uniform stride is gone"
    );
    assert_eq!(payload_words * 4, total_blocks * PLANES14_BYTES);
}

/// The last block's four-word window sits inside the buffer, and a buffer one
/// word too short panics there.
///
/// 🕳️ **The truncation has to remove a payload word, not the pad, and that is
/// not a detail.** For an even block count `pack_plane_bytes`'s four spare
/// bytes are an *equivalent mutant*: the word-alignment step alone already
/// supplies exactly what the window needs (`fused.rs` proves the identity).
/// Dropping only the pad would therefore pass, and a test written that way
/// would be green for the wrong reason. What must hold — and what is asserted
/// — is that the window's far end is exactly the payload's end.
#[test]
fn the_last_block_window_is_in_bounds_and_one_word_less_is_not() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let parts = qkv_fixture(0xB0_07);
    let matrices: Vec<FusedMatrix> = parts.iter().map(|p| to_matrix(&tr, 0, p, 0x11)).collect();
    let (_, groups) = segment_matrices(matrices).expect("q/k/v fuse");
    let g = &groups[0];
    let words = words_of(&g.stream);

    let last = g.d_out * NBLOCKS - 1;
    let far = (PLANES14_BYTES * last) / 4 + 4;
    assert!(far <= words.len(), "the last block window falls outside the buffer");
    let full = planes_fields(words, last);

    // One word short of what the window needs, so `words[w + 3]` is the first
    // index out of bounds — a panic here, an illegal address on the card.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let short = &words[..far - 1];
    let r = std::panic::catch_unwind(|| planes_fields(short, last));
    std::panic::set_hook(prev);
    assert!(
        r.is_err(),
        "a buffer that is one word short must panic on the last block window, \
         not return {full:?}"
    );
}

/// **The mutant, asserted to fail.** The naive concatenation — every part's
/// `words` verbatim, pads included — must decode *differently* from the splice
/// on the second segment.
///
/// This is what proves the file sees the defect it exists to catch. It also
/// records the shape of that defect: nothing panics, no index leaves the
/// buffer (the naive stream is *longer* than the spliced one), and every field
/// read out of the shifted bit fields is a valid class and a point in the ball.
#[test]
fn the_naive_word_concatenation_decodes_differently() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let parts = qkv_fixture(0x00DE_AD10);
    let unfused: Vec<HostStream> = parts
        .iter()
        .map(|p| tr.stream(&p.indices, &p.gains, p.d_out, NBLOCKS).expect("transcode").0)
        .collect();
    let matrices: Vec<FusedMatrix> = parts.iter().map(|p| to_matrix(&tr, 0, p, 0x11)).collect();
    let (_, groups) = segment_matrices(matrices).expect("q/k/v fuse");
    let g = &groups[0];
    let fused = words_of(&g.stream);

    // The defect, written out: `.concat()` on the packed words.
    let naive: Vec<u32> = unfused.iter().flat_map(|s| words_of(s).iter().copied()).collect();
    assert!(
        naive.len() > fused.len(),
        "the naive concatenation is LONGER, which is why it does not crash"
    );

    let row0 = g.parts[1].row0;
    let mut differ = 0usize;
    for r in 0..g.parts[1].d_out {
        for b in 0..NBLOCKS {
            let blk = (row0 + r) * NBLOCKS + b;
            if planes_fields(&naive, blk) != planes_fields(fused, blk) {
                differ += 1;
            }
        }
    }
    assert!(
        differ > 0,
        "the naive concatenation decodes like the splice, so this file would not see \
         the defect it exists to catch"
    );
    // And the first segment is untouched — the pad only shifts what follows it,
    // which is why the failure is partial and therefore plausible.
    for b in 0..g.parts[0].d_out * NBLOCKS {
        assert_eq!(
            planes_fields(&naive, b),
            planes_fields(fused, b),
            "the first segment cannot move, nothing precedes it"
        );
    }
}

/// The splice refuses a stream whose word count is not `payload + 1`, and an
/// odd block count, rather than trusting the caller's arithmetic.
#[test]
fn the_splice_refuses_a_stream_of_the_wrong_length() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let (idx, gains): (Vec<u64>, Vec<u32>) = {
        let mut rng = SplitMix64::new(0xF00D);
        (
            (0..40).map(|_| 1 + rng.next() % N13).collect(),
            (0..40).map(|_| (rng.next() & 1) as u32).collect(),
        )
    };
    let (ok, _) = tr.stream(&idx, &gains, 8, 5).expect("transcode");
    // The honest call: 40 blocks, and the stream really carries 40.
    assert_eq!(splice_planes14(&[(&ok, 40)]).expect("a single segment").len(), 141);

    // A block count that does not match the stream — the mutant "I took
    // `words.len() - 1` without knowing why" dies here.
    let e = splice_planes14(&[(&ok, 38)]).expect_err("38 blocks for a stream of 40");
    assert!(e.contains("words"), "{e}");
    // An odd block count: `14·n` is then not a multiple of 4, so the segment
    // boundary would not fall on a word at all.
    let e = splice_planes14(&[(&ok, 39)]).expect_err("odd count");
    assert!(e.contains("odd"), "{e}");
    // And a stream that is not Planes14 at all.
    let slot = Transcoder::new(FusedLayout::Slot32).expect("slot32 transcoder");
    let (s32, _) = slot.stream(&idx, &gains, 8, 5).expect("transcode");
    let e = splice_planes14(&[(&s32, 40)]).expect_err("slot32 stream");
    assert!(e.contains("Planes14"), "{e}");
}

// ---------------------------------------------------------------------------
// 2. The grouping — the key, the rank, and what stays alone
// ---------------------------------------------------------------------------

/// A two-layer model, seven projections each, in the order the artifact writes
/// them — with layer 1's q/k/v shuffled so that rank, not arrival, decides.
fn two_layer_model(tr: &Transcoder) -> Vec<FusedMatrix> {
    let mut out = Vec::new();
    let lone = |suffix: &'static str, d_out: usize| Part {
        suffix,
        d_out,
        indices: vec![1; d_out * NBLOCKS],
        gains: vec![0; d_out * NBLOCKS],
        gscale: [1.0, 2.0],
        rscale: vec![1.0; d_out],
        tail: vec![0; d_out * TAIL_W],
    };
    for layer in 0..2usize {
        let qkv = qkv_fixture(0x1234 + layer as u64);
        let gate_up: Vec<Part> = ["mlp.gate_proj", "mlp.up_proj"]
            .iter()
            .map(|&s| lone(s, 16))
            .collect();
        // Layer 1 hands q/k/v over as v, q, k: the row order is the contract,
        // and it must come from the rank rather than from arrival.
        let order: Vec<usize> = if layer == 0 { vec![0, 1, 2] } else { vec![2, 0, 1] };
        for i in order {
            out.push(to_matrix(tr, layer, &qkv[i], 0x100 + layer as u64));
        }
        out.push(to_matrix(tr, layer, &lone("self_attn.o_proj", 8), 0x200 + layer as u64));
        for p in &gate_up {
            out.push(to_matrix(tr, layer, p, 0x300 + layer as u64));
        }
        out.push(to_matrix(tr, layer, &lone("mlp.down_proj", 8), 0x400 + layer as u64));
    }
    out
}

/// The grouping keys on `(layer, Act)`, orders by rank, and leaves `o_proj` and
/// `down_proj` alone.
///
/// The layer is part of the key on purpose: q/k/v of two different layers share
/// a `d_in` and would splice perfectly well into nonsense — layer 0's
/// activation is not layer 1's. Dropping the layer from the key is the mutation
/// this test exists to kill.
#[test]
fn the_grouping_keys_on_the_layer_and_orders_by_rank() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let (singles, groups) = segment_matrices(two_layer_model(&tr)).expect("fusable model");

    assert_eq!(
        groups.iter().map(|g| g.key.as_str()).collect::<Vec<_>>(),
        ["000.Attn", "000.Mlp", "001.Attn", "001.Mlp"],
        "one group per (layer, activation), in first-appearance order"
    );
    // `o_proj` and `down_proj` consume activations nothing else does.
    assert_eq!(
        singles.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
    );

    for g in &groups {
        // Ranks are 0..n in row order, and the parts tile the group.
        let spans: Vec<(usize, usize, usize)> =
            g.parts.iter().map(|p| (p.rank, p.row0, p.d_out)).collect();
        check_seg_spans(&g.key, &spans, g.d_out).expect("the parts tile the group");
        assert!(g.d_out.is_multiple_of(8), "{}: total d_out", g.key);
        for p in &g.parts {
            assert!(p.d_out.is_multiple_of(8), "{}: {} d_out", g.key, p.name);
        }
        assert_eq!(g.gs_off.len(), g.d_out);
        assert_eq!(g.gscale.len(), 2 * g.parts.len());
        assert!(g.rotation.is_some(), "{}: the group carries a rotation", g.key);
    }

    // Layer 1's three arrived as v, q, k and must come back q, k, v.
    let l1 = &groups[2];
    assert_eq!(
        l1.parts.iter().map(|p| p.proj.as_str()).collect::<Vec<_>>(),
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "rank orders the rows, not arrival"
    );
    assert_eq!(groups[1].parts.iter().map(|p| p.proj.as_str()).collect::<Vec<_>>(), [
        "mlp.gate_proj",
        "mlp.up_proj"
    ]);
    // Two layers, four groups, four lone projections: 14 projections become 8
    // matvec launches — the 252 → 144 of the published 4B, in miniature.
    assert_eq!(
        singles.len() + groups.len(),
        8,
        "4 groups + 4 lone for 14 projections"
    );
}

/// An incomplete group is a broken file, not a degraded case.
#[test]
fn a_group_missing_a_consumer_is_refused() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let mut m = two_layer_model(&tr);
    // Drop layer 0's `v_proj`. The two survivors would splice into a matrix of
    // 40 rows that the model would then read as if it had 48.
    let i = m
        .iter()
        .position(|x| x.name == "model.layers.0.self_attn.v_proj.weight")
        .expect("the fixture carries it");
    m.remove(i);
    let e = refusal(segment_matrices(m), "an incomplete group must be refused");
    assert!(e.contains("000.Attn") && e.contains("v_proj"), "{e}");
}

/// Two parts of one group carrying two rotations is refused, naming them —
/// even though `check_rotation_partition` already ran at load. The splice owns
/// the row order and must not inherit a premise it cannot see.
#[test]
fn a_group_whose_parts_disagree_on_the_rotation_is_refused() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let mut m = two_layer_model(&tr);
    let i = m
        .iter()
        .position(|x| x.name == "model.layers.0.self_attn.k_proj.weight")
        .expect("the fixture carries it");
    m[i].rotation = Some((D_IN, 0xBAD));
    let e = refusal(segment_matrices(m), "two rotations in one group: refusal expected");
    assert!(e.contains("k_proj"), "{e}");
}

// ---------------------------------------------------------------------------
// 3. The byte accounting — what fusion costs, exactly
// ---------------------------------------------------------------------------

/// Fusion costs exactly `4 · d_out` bytes a group: the payload does not move
/// (14 bytes a block, grouping or not) and neither does the tail-and-scales
/// term (additive in `d_out` at a shared `tail_w`). What is added is `gs_off`,
/// one `u32` a fused row.
///
/// Measured here on the fixture, then *calculated* on the published 4B's real
/// shapes — the two together are what licenses the number the lot announces.
#[test]
fn fusion_costs_exactly_four_bytes_a_fused_row() {
    let tr = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");
    let model = two_layer_model(&tr);
    let unfused_total: u64 = model.iter().map(|m| m.bytes).sum();
    let (singles, groups) = segment_matrices(model).expect("fusable model");
    let fused_total: u64 =
        singles.iter().map(|m| m.bytes).sum::<u64>() + groups.iter().map(|g| g.bytes).sum::<u64>();
    let fused_rows: usize = groups.iter().map(|g| g.d_out).sum();
    assert_eq!(
        fused_total - unfused_total,
        fused_rows as u64 * 4,
        "fusion adds only gs_off"
    );

    // The published Qwen3-4B, *calculated* on its shapes: 36 layers, and per
    // layer q+k+v = 4096 + 1024 + 1024 = 6144 fused rows plus gate+up =
    // 9728 + 9728 = 19456, i.e. 25,600. `o_proj` and `down_proj` stay alone and
    // pay nothing.
    let per_layer_4b = (4096 + 1024 + 1024) + (9728 + 9728);
    assert_eq!(per_layer_4b, 25_600);
    assert_eq!(36 * per_layer_4b * 4, 3_686_400);
    // Which is +0.0081 b/weight over the 3,633,315,840 weights of the 4B's
    // projections — the term the runtime line must not drop.
    let b_per_weight: f64 = 3_686_400.0 * 8.0 / 3_633_315_840.0;
    assert!(
        (b_per_weight - 0.008_117).abs() < 1e-6,
        "b/weight added by gs_off: {b_per_weight}"
    );
}

// ---------------------------------------------------------------------------
// 4. The row order, restated where the model checks it
// ---------------------------------------------------------------------------

/// `check_seg_spans` accepts a group that tiles and refuses every way of not
/// tiling — this is the half of risk R3 a machine without a card can reach.
///
/// The loader assigns the row order and `model::SegPlan::of` re-derives it from
/// the projections it was handed; two places have to agree and neither may
/// assume. A group read in the order k,q,v returns finite, plausible, wrong
/// numbers with no assertion anywhere else.
#[test]
fn the_spans_must_tile_the_group_in_rank_order() {
    check_seg_spans("g", &[(0, 0, 4096), (1, 4096, 1024), (2, 5120, 1024)], 6144)
        .expect("q, k, v of the 4B");
    check_seg_spans("g", &[(0, 0, 8)], 8).expect("a one-part group tiles");

    // Out of order: the ranks are 0,1,2 in *some* order, the rows still tile,
    // and the result is a group read k,q,v.
    let e = check_seg_spans("g", &[(1, 0, 1024), (0, 1024, 4096), (2, 5120, 1024)], 6144)
        .expect_err("ranks out of order");
    assert!(e.contains("rank"), "{e}");
    // A hole between two parts.
    let e = check_seg_spans("g", &[(0, 0, 4096), (1, 5120, 1024)], 5144)
        .expect_err("hole between two parts");
    assert!(e.contains("starts"), "{e}");
    // Partial: the parts are contiguous and in order, but they do not cover the
    // group — the tail rows of the fused launch would be dropped in silence.
    let e = check_seg_spans("g", &[(0, 0, 4096), (1, 4096, 1024)], 6144)
        .expect_err("partial group");
    assert!(e.contains("6144"), "{e}");
    // And a launch with no part at all.
    assert!(check_seg_spans("g", &[], 6144).is_err());
}
