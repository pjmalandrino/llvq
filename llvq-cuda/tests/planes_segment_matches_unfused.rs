//! The row-concatenation behind `tv_planes_seg`, proved on the development
//! machine — the only half of item A4 that can be.
//!
//! ## What this file proves, and what it deliberately does not
//!
//! `kernels/planes_seg.cu` cannot be validated here. A single-threaded driver
//! reproduces neither `__syncthreads()` nor a warp shuffle, so `host_planes.cpp`
//! compiles it and stops there; the kernel is proved on the card, by the bench's
//! bit-exact comparison against `tv_planes`.
//!
//! What *is* provable here is the host side of the fusion, and it is where a
//! wrong answer would actually hide. Fusing q+k+v does not change one line of
//! arithmetic — the same blocks, in the same order, with the same centroids —
//! so every risk is bookkeeping: a row order that drifts at a segment boundary,
//! a `gs_off` that hands a row its neighbour's gain centroids, a `tail` or
//! `rscale` shifted by a constant past the join. Each of those produces
//! plausible numbers, which is exactly why the card-side test demands bit
//! equality rather than a tolerance. This file demands the same thing one level
//! up, in Rust, in two seconds, on a machine with no GPU:
//!
//! 1. **the block stream is invariant.** Every block of the concatenated
//!    transcode decodes — Slot32 *and* Planes14 — to the point and gain the
//!    corresponding block of the unfused matrix carries;
//! 2. **the weights are invariant.** Reconstructing a row's 24 weights per
//!    block through the fused `gscale`/`gs_off`/`rscale` gives **bit-identical**
//!    f64 to reconstructing them through the unfused matrix's own arrays. This
//!    is the assertion a wrong `gs_off` dies on: swapping two segments' pairs
//!    moves some rows by ~2× and leaves the rest untouched;
//! 3. **Planes14 fusion costs no bytes, structurally.** Uniform 14-byte stride
//!    and no base table means the stream is 14 bytes per block whatever the
//!    grouping — unlike Slot32, whose per-group stride is the widest record
//!    among 32 blocks and can therefore move when the concatenation regroups
//!    across a segment boundary. The Slot32 job had to *measure* that
//!    confounder (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`, −0.00 %); here
//!    it is asserted;
//! 4. **the grouping keys on the layer.** q/k/v of two different layers share a
//!    `d_in` and would concatenate perfectly well into nonsense.
//!
//! The transcoding path is the bench's, `planes14_from_slot32`, not
//! `llvq_artifact::runtime::transcode_planes14` — so this proves the object the
//! bench builds rather than a second implementation of it.

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;

include!("../src/planes14_host.rs");
include!("../src/seg_host.rs");

/// A part's own arrays, owned — `SegPart` borrows, and the fixture has to own
/// them somewhere.
struct Part {
    d_out: usize,
    indices: Vec<u64>,
    gains: Vec<u32>,
    centroids: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<f32>,
}

/// Three segments of a plausible q/k/v group: 4:1:1 rows, one shared `d_in`
/// with a tail (`128 = 24·5 + 8`, so `TailPolicy::KeepExact` is exercised),
/// and three **distinct** centroid pairs — without which every `gs_off` would
/// be as good as every other and assertion 2 would prove nothing.
fn fixture(rng: &mut SplitMix64) -> (Vec<Part>, usize) {
    let d_in = 128;
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    let centroids = [[0.625f32, 1.375], [0.5, 2.0], [0.75, 1.125]];
    let parts = [32usize, 8, 8]
        .iter()
        .zip(centroids)
        .map(|(&d_out, centroids)| Part {
            d_out,
            // A sixth of the blocks are the origin: `id == 0` is the one class
            // whose Slot32 record is the header alone, so it is the case a
            // stride or offset error trips over first.
            indices: (0..d_out * nblocks)
                .map(|_| if rng.next().is_multiple_of(6) { 0 } else { 1 + rng.next() % N13 })
                .collect(),
            gains: (0..d_out * nblocks).map(|_| (rng.next() & 1) as u32).collect(),
            centroids,
            rscale: (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect(),
            tail: (0..d_out * tail_w).map(|_| rng.next_gaussian() as f32).collect(),
        })
        .collect();
    (parts, d_in)
}

fn borrow(parts: &[Part], d_in: usize) -> Vec<SegPart<'_>> {
    parts
        .iter()
        .map(|p| SegPart {
            d_out: p.d_out,
            d_in,
            indices: &p.indices,
            gains: &p.gains,
            centroids: p.centroids,
            rscale: &p.rscale,
            tail: &p.tail,
        })
        .collect()
}

/// The 24 weights a block contributes to its row, exactly as every f64
/// reference in this repository forms them: the class's value, normalised by
/// its shell radius, times the gain centroid, times the row scale. The origin
/// contributes zeros — it has no shell.
fn block_weights(pt: &llvq_core::Point, centroid: f32, rscale: f32) -> [f64; DIM] {
    let mut w = [0.0f64; DIM];
    if let Some(shell) = llvq_core::Leech::shell_index(pt).filter(|&s| s > 0) {
        let k = centroid as f64 * rscale as f64 / ((16 * shell) as f64).sqrt();
        for (wi, &v) in w.iter_mut().zip(pt.iter()) {
            *wi = v as f64 * k;
        }
    }
    w
}

#[test]
fn the_fused_stream_decodes_to_the_unfused_weights() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x5_E6A4);
    let (parts, d_in) = fixture(&mut rng);
    let refs = borrow(&parts, d_in);
    let seg = seg_concat(&refs);

    assert_eq!(seg.d_out, 48, "the fixture stacks 32 + 8 + 8 rows");
    assert_eq!(seg.nblocks, 5);
    assert_eq!(seg.tail_w, 8);
    assert_eq!(seg.row_offsets, vec![0, 32, 40, 48]);

    let fused_slot =
        transcode(&fd, &table, &seg.indices, &seg.gains, Layout::Slot32).expect("transcodes");
    let fused_planes = planes14_from_slot32(&fused_slot, &table);

    let mut unfused_planes_bytes = 0usize;
    for (s, p) in parts.iter().enumerate() {
        let rt = transcode(&fd, &table, &p.indices, &p.gains, Layout::Slot32).expect("transcodes");
        let planes = planes14_from_slot32(&rt, &table);
        unfused_planes_bytes += planes.len();

        for r in 0..p.d_out {
            let grow = seg.row_offsets[s] + r;
            // The gain centroids are the only thing that does not concatenate,
            // so this indirection is the whole point of the segmented kernel.
            let gs = seg.gs_off[grow] as usize;
            assert!(gs + 1 < seg.centroids.len(), "row {grow}: gs_off {gs} off the table");

            for b in 0..seg.nblocks {
                let (want_pt, want_gain) = rt.decode_block(&table, r * seg.nblocks + b);
                let (got_pt, got_gain) =
                    fused_slot.decode_block(&table, grow * seg.nblocks + b);
                assert_eq!(
                    (got_pt, got_gain),
                    (want_pt, want_gain),
                    "segment {s}, row {r} (fused row {grow}), block {b}: Slot32 content moved"
                );

                let want_p = planes14_decode_block(&planes, &table, r * seg.nblocks + b);
                let got_p =
                    planes14_decode_block(&fused_planes, &table, grow * seg.nblocks + b);
                assert_eq!(
                    got_p, want_p,
                    "segment {s}, row {r} (fused row {grow}), block {b}: Planes14 content moved"
                );

                // The assertion a wrong `gs_off` dies on. Same arithmetic on
                // both sides, so the equality is exact — a tolerance here would
                // accept a swap of two segments' centroids on the rows where
                // the two pairs happen to be close.
                let want_w = block_weights(&want_pt, p.centroids[want_gain as usize], p.rscale[r]);
                let got_w = block_weights(
                    &got_pt,
                    seg.centroids[gs + got_gain as usize],
                    seg.rscale[grow],
                );
                assert_eq!(
                    got_w, want_w,
                    "segment {s}, row {r} (fused row {grow}), block {b}: weights moved"
                );
            }

            let (a, b) = (grow * seg.tail_w, r * seg.tail_w);
            assert_eq!(
                &seg.tail[a..a + seg.tail_w],
                &p.tail[b..b + seg.tail_w],
                "segment {s}, row {r}: the tail moved"
            );
        }
    }

    // Assertion 3: structural, not measured. `planes14_from_slot32` emits
    // `PLANES14_STRIDE` bytes per block and nothing else — no base table, no
    // per-group stride — so a concatenation of matrices sharing `d_in` reads
    // exactly as many bytes as the matrices did apart. Slot32 carries no such
    // guarantee, which is why its fusion job had to print the delta.
    assert_eq!(
        fused_planes.len(),
        unfused_planes_bytes,
        "Planes14 fusion changed the byte total: the uniform stride is gone"
    );
    assert_eq!(
        fused_planes.len(),
        seg.d_out * seg.nblocks * PLANES14_STRIDE,
        "Planes14 is 14 bytes per block, always"
    );
}

#[test]
fn the_grouping_keys_on_the_layer_and_orders_by_rank() {
    let names: Vec<&str> = vec![
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        // Same shapes, same d_in, different activation. Fusing across layers
        // would transcode and launch perfectly well, and compute nonsense.
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
    ];
    let groups = seg_groups(&names).expect("the artifact's own name grammar");

    assert_eq!(
        groups.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["000.qkv", "000.gateup", "001.qkv"],
        "one group per (layer, family), in first-appearance order"
    );
    assert_eq!(groups[0].1, vec![0, 1, 2], "q, then k, then v");
    assert_eq!(groups[1].1, vec![4, 5], "gate, then up");
    // Layer 1's three appear as v, q, k and must come back q, k, v: the row
    // order is the contract the per-part comparison lines up on.
    assert_eq!(groups[2].1, vec![8, 9, 7], "rank orders the rows, not arrival");
    // o_proj and down_proj consume activations nothing else does.
    assert!(seg_family("model.layers.0.self_attn.o_proj.weight").is_none());
    assert!(seg_family("model.layers.0.mlp.down_proj.weight").is_none());
}
