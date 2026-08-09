// Host-side row-concatenation of the projections that share an activation —
// the host half of `tv_planes_seg` (kernels/planes_seg.cu) and of the
// `tv_slot_seg` arm `bin/matvec.rs` already measures. Item A4.
//
// ⚠️ This file is `include!`d — by `bin/planesbench.rs` and by
// `tests/planes_segment_matches_unfused.rs` — rather than being a module of
// `lib.rs`, for the reason `planes14_host.rs` gives: `lib.rs` belongs to
// another lot, and this code has to be reachable from a bench and from a test
// that runs on the development Mac. It therefore declares no `use` items:
// every path is fully qualified, so it can land in any module without
// colliding with the host's imports.
//
// ## What a segment is
//
// q, k and v consume the same activation; so do gate and up. Stacking their
// rows gives one matrix with the same `d_in` — hence the same `nblocks` and
// the same `tail_w` — and one launch where there were three. Everything
// indexed by row concatenates with no special case: `rscale`, `tail`, and the
// block stream itself, whose blocks are numbered `row * nblocks + p` on both
// sides.
//
// The gain centroids are the exception: they belong to a *matrix*, and the
// decoder picks one of the pair with the block's gain bit, so they cannot be
// folded into the per-row `rscale`. `gs_off[row]` names where that row's pair
// starts in the concatenated `gscale`, and the kernel reads it once per row.
//
// ## Why this lives in one place instead of two
//
// The concatenation is the part a test on this machine can actually prove —
// row order, `gs_off`, the block stream — and the part a wrong answer would
// hide in: swapping two segments' centroids moves some rows by ~2× and leaves
// the rest untouched, which a global tolerance waves through. Sharing the code
// between the bench and `tests/planes_segment_matches_unfused.rs` is what makes
// that proof about the bench rather than about a re-implementation of it.

/// One projection's contribution to a segmented matrix.
///
/// Borrowed rather than owned: the caller already holds these (they are the
/// artifact's own arrays), and a segmented build of Qwen3-4B would otherwise
/// copy 1.2 GB of indices twice over.
pub struct SegPart<'a> {
    pub d_out: usize,
    pub d_in: usize,
    pub indices: &'a [u64],
    pub gains: &'a [u32],
    pub centroids: [f32; 2],
    pub rscale: &'a [f32],
    pub tail: &'a [f32],
}

/// The concatenation: exactly what one segmented launch needs.
///
/// `centroids` is the concatenated `gscale` — two floats per segment, in
/// segment order — and `gs_off[row]` is the index of the first of the pair
/// that row's segment owns. Every other field is what the equivalent unfused
/// matrix would have had, with its rows stacked.
pub struct Segmented {
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    pub indices: Vec<u64>,
    pub gains: Vec<u32>,
    pub centroids: Vec<f32>,
    pub gs_off: Vec<u32>,
    pub rscale: Vec<f32>,
    pub tail: Vec<f32>,
    /// First row of each segment in the concatenation, `parts.len() + 1`
    /// entries — the map a per-part comparison needs to line up.
    pub row_offsets: Vec<usize>,
}

/// Stack `parts` by rows, in the order given.
///
/// The order *is* the contract: it fixes which rows of the fused output
/// correspond to which matrix, so callers must sort by
/// [`seg_family`]'s rank before calling — [`seg_groups`] does.
///
/// Everything asserted here is a silent-wrong-answer guard rather than a
/// defensive habit. A mismatched `d_in` would give the fused matrix a block
/// count no segment agrees with; a short `rscale` or `tail` would shift every
/// row past the boundary by a constant and still produce plausible numbers.
pub fn seg_concat(parts: &[SegPart]) -> Segmented {
    assert!(!parts.is_empty(), "a segmented matrix needs at least one part");
    let d_in = parts[0].d_in;
    let nblocks = d_in / llvq_core::DIM;
    let tail_w = d_in % llvq_core::DIM;

    let d_out: usize = parts.iter().map(|p| p.d_out).sum();
    // The kernels launch whole 256-thread blocks of 8 rows and carry no bounds
    // guard (a `return` before `__syncthreads()` deadlocks). Asserting the
    // total alone would let a group of segments whose sizes are individually
    // ragged pass on a lucky sum, and then the *unfused* control could not be
    // launched at all — so both are checked.
    assert_eq!(d_out % 8, 0, "fused d_out {d_out} is not a whole number of blocks");

    let mut out = Segmented {
        d_out,
        d_in,
        nblocks,
        tail_w,
        indices: Vec::with_capacity(d_out * nblocks),
        gains: Vec::with_capacity(d_out * nblocks),
        centroids: Vec::with_capacity(2 * parts.len()),
        gs_off: Vec::with_capacity(d_out),
        rscale: Vec::with_capacity(d_out),
        tail: Vec::with_capacity(d_out * tail_w),
        row_offsets: Vec::with_capacity(parts.len() + 1),
    };

    let mut at = 0usize;
    for (seg, p) in parts.iter().enumerate() {
        assert_eq!(p.d_in, d_in, "segment {seg}: d_in {} against {d_in}", p.d_in);
        assert_eq!(p.d_out % 8, 0, "segment {seg}: d_out {} is not whole blocks", p.d_out);
        assert_eq!(
            p.indices.len(),
            p.d_out * nblocks,
            "segment {seg}: {} indices for {} rows of {nblocks} blocks",
            p.indices.len(),
            p.d_out
        );
        assert_eq!(p.gains.len(), p.indices.len(), "segment {seg}: one gain per block");
        assert_eq!(p.rscale.len(), p.d_out, "segment {seg}: one row scale per row");
        assert_eq!(
            p.tail.len(),
            p.d_out * tail_w,
            "segment {seg}: {} tail values for {} rows of {tail_w}",
            p.tail.len(),
            p.d_out
        );

        out.row_offsets.push(at);
        out.indices.extend_from_slice(p.indices);
        out.gains.extend_from_slice(p.gains);
        out.rscale.extend_from_slice(p.rscale);
        out.tail.extend_from_slice(p.tail);
        out.centroids.extend_from_slice(&p.centroids);
        // Two centroids per segment, so segment `seg`'s pair starts at `2·seg`.
        // One entry per row, because the kernel reads it as `gs_off[row]`.
        out.gs_off
            .extend(std::iter::repeat_n(2 * seg as u32, p.d_out));
        at += p.d_out;
    }
    out.row_offsets.push(at);
    out
}

/// Which fusion family a projection belongs to, and its rank inside it.
///
/// `None` for anything that is not fusible — `o_proj` and `down_proj` consume
/// activations nothing else does, so they stay one launch each.
///
/// The rank is load-bearing: it fixes the row order inside a group, so the
/// concatenation is deterministic and a per-part comparison against the
/// unfused kernels lines up. Matching on `contains` rather than on equality
/// because the artifact's names are fully qualified
/// (`model.layers.7.self_attn.q_proj.weight`).
pub fn seg_family(name: &str) -> Option<(&'static str, usize)> {
    for (i, p) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
        if name.contains(p) {
            return Some(("qkv", i));
        }
    }
    for (i, p) in ["gate_proj", "up_proj"].iter().enumerate() {
        if name.contains(p) {
            return Some(("gateup", i));
        }
    }
    None
}

/// Group projection names into fusible families, keyed by **layer and family**.
///
/// The layer is part of the key on purpose: q/k/v of two different layers share
/// a `d_in` and would concatenate perfectly well, and the result would be
/// nonsense — layer 7's activation is not layer 8's. Dropping the layer from
/// the key is the mutation this function's test exists to kill.
///
/// Returns the groups in first-appearance order, each holding indices into
/// `names`, sorted by [`seg_family`] rank. Groups of one are kept rather than
/// filtered: `bin/matvec.rs`'s Slot32 arm keeps them, and the two arms have to
/// fuse the same set for their deltas to be comparable.
pub fn seg_groups(names: &[&str]) -> Result<Vec<(String, Vec<usize>)>, String> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let Some((fam, _)) = seg_family(n) else { continue };
        let (layer, _) = llvq_artifact::split_name(n).map_err(|e| e.to_string())?;
        let key = format!("{layer:03}.{fam}");
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    for (_, v) in groups.iter_mut() {
        v.sort_by_key(|&i| seg_family(names[i]).map_or(usize::MAX, |(_, r)| r));
    }
    Ok(groups)
}
