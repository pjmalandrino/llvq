//! Row concatenation for the fused launches ([`FusedGroup`], [`SegPart`]) and the
//! model those groups end up in ([`FusedModel`]). One file because a group is
//! built only from a model's matrices and the model only holds what it built.

use std::collections::HashMap;

use llvq_artifact::runtime::PLANES14_BYTES;

use super::{matrix_side_bytes, FusedInt4, FusedLayout, FusedMatrix, HostStream, RotKey, RotationTables};

// ---------------------------------------------------------------------------
// Segmentation — the row-concatenation one fused launch reads
// ---------------------------------------------------------------------------
//
// The twin of `llvq-cuda/src/seg_host.rs`, which does the same job for the
// bench. The two are **deliberately not shared**: `seg_host.rs` concatenates
// raw indices *before* transcoding, carries an `f32` tail, and targets the f32
// bench kernel, while the served path holds streams already packed into `u32`
// words and a binary16 tail. Making one file serve both would mean reworking
// the object that pins the bench, inside a lot whose subject is not the bench.
// The price is paid in tests: `llvq-llm/tests/fused_segment.rs` has to be as
// lethal as `llvq-cuda/tests/planes_segment_matches_unfused.rs`, not lighter.

/// `u32` words a `Planes14` payload of `n_blocks` occupies, **pad excluded**.
///
/// `14·n ≡ 2n (mod 4)`, so this is a whole number of words exactly when `n` is
/// even — which the served path guarantees, since `upload_matrix` refuses
/// `d_out % 8 != 0` and `n = d_out · nblocks`. Returned as an `Option` rather
/// than assumed: the day a ragged `d_out` reaches here, this is where it stops.
pub fn planes_payload_words(n_blocks: usize) -> Option<usize> {
    n_blocks
        .is_multiple_of(2)
        .then(|| PLANES14_BYTES * n_blocks / 4)
}

/// Splice the packed `Planes14` streams of a group into the single stream the
/// segmented kernel indexes. `parts` is `(stream, n_blocks)` **in row order**.
///
/// ## Why this cannot be a `concat()`
///
/// [`pack_plane_bytes`] appends four bytes and then word-aligns, so every
/// `HostStream::Planes14` carries **one trailing pad word**. Concatenating two
/// of them buries that word in the middle of the stream: the kernel computes
/// `byte = 14·b` from the base of the buffer, so every block of the second
/// segment would be read four bytes early, with `sh = (byte & 3)·8` on the
/// wrong parity. The class id, the gain, the sign mask and the three planes all
/// come out of a shifted bit field — a valid class, a point in the ball, a
/// model that runs.
///
/// And nothing crashes: with two pads the buffer is *longer* than the last
/// block's four-word window needs, so there is no illegal address to catch it
/// in the middle of a billed job. Finite, plausible, wrong.
///
/// ## Why the concatenation is correct at all
///
/// `transcode_planes14` opens a fresh `BitSink` at `bit0 = b·14·8` for every
/// block and carries **no inter-block state**: record `b` depends only on
/// `indices[b]`, `gains[b]` and the `ClassTable`. So the transcode of a
/// concatenation is byte for byte the concatenation of the transcodes, and the
/// block of local row `r` of segment `s` carries the global number
/// `(row0_s + r)·nblocks + p` — exactly what the kernel computes.
///
/// The length check below is therefore the whole safety of this function, and
/// it is written as an equality rather than as `len() - 1`: it also catches the
/// day a stream arrives whose block count is odd, where `pack_plane_bytes`
/// emits a different number of pad bytes.
pub fn splice_planes14(parts: &[(&HostStream, usize)]) -> Result<Vec<u32>, String> {
    let total: usize = parts.iter().map(|&(_, n)| n).sum();
    let want = planes_payload_words(total)
        .ok_or_else(|| format!("{total} blocks in total, an odd count"))?;
    let mut out = Vec::with_capacity(want + 1);
    for (i, &(s, n)) in parts.iter().enumerate() {
        let HostStream::Planes14 { words } = s else {
            return Err(format!("segment {i}: stream that is not Planes14"));
        };
        let pw = planes_payload_words(n).ok_or_else(|| {
            format!(
                "segment {i}: {n} blocks, an odd count, so 14·n is not a multiple of 4 and \
                 the boundary does not fall on a word"
            )
        })?;
        if words.len() != pw + 1 {
            return Err(format!(
                "segment {i}: {} words for {n} blocks, {} expected (payload {pw} + one \
                 end-of-stream padding word)",
                words.len(),
                pw + 1
            ));
        }
        // Payload ONLY, for every segment including the last.
        out.extend_from_slice(&words[..pw]);
    }
    // One single pad, for the four-word window of the LAST block — exactly what
    // `pack_plane_bytes` would have emitted over the concatenated payload.
    out.push(0);
    debug_assert_eq!(out.len(), want + 1);
    Ok(out)
}

/// One projection's identity and place inside a fused group.
pub struct SegPart {
    /// The artifact's fully qualified name — what an error message must name.
    pub name: String,
    pub layer: usize,
    /// Checkpoint suffix, e.g. `self_attn.q_proj` — the key `Block::proj` uses.
    pub proj: String,
    pub d_out: usize,
    /// First row of this projection inside the group.
    pub row0: usize,
    /// Rank inside the family, i.e. its index in [`crate::model::Act::consumers`].
    /// This *is* the row order, and `model::group_forward` re-checks it
    /// structurally rather than trusting it: two places have to agree, so
    /// neither may assume.
    pub rank: usize,
}

/// A row-concatenation of the projections that share one activation — what one
/// segmented launch needs, and nothing else.
///
/// `stream` is a `HostStream::Planes14` by construction ([`splice_planes14`]
/// refuses anything else), so no code path can hand the segmented kernel a
/// bases array or an exception table.
pub struct FusedGroup {
    /// `"{layer:03}.{act:?}"` — the grouping key, for logs and errors.
    pub key: String,
    /// The **total**: Σ of the parts.
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    pub stream: HostStream,
    /// The concatenated `gscale`: two floats per part, in row order.
    pub gscale: Vec<f32>,
    /// `d_out` entries. `gs_off[row] = 2 · rank`, the index of the first of the
    /// pair that row's part owns — the one thing that does **not** concatenate,
    /// because a gain centroid belongs to a matrix while the gain bit belongs
    /// to a block.
    pub gs_off: Vec<u32>,
    pub rscale: Vec<f32>,
    pub tail: Vec<u16>,
    pub rotation: Option<RotKey>,
    pub parts: Vec<SegPart>,
    /// Device bytes: the payload, [`matrix_side_bytes`] for the tail and the
    /// row scales, **and `gs_off`** — 4 bytes a fused row, which no other
    /// layout pays. On the published 4B that last term is the whole difference
    /// between the two arms: 36 layers × 25,600 fused rows × 4 = 3,686,400
    /// bytes, i.e. +0.0081 b/weight. Counted here rather than dropped, because
    /// `runtime_bytes` is what `fusedrun` prints as "GB on the card".
    pub bytes: u64,
}

/// Check that the spans handed to one segmented launch are the whole group, in
/// row order. `spans` is `(rank, row0, d_out)` **in the caller's order**.
///
/// Lives here, beside the loader that assigns the row order, rather than inside
/// `model::group_forward` where it is called from: that function's segmented
/// branch exists on no target this workspace's development machine builds, and
/// this is the half of risk R3 a test can reach. Two places have to agree on
/// the row order and neither may assume — a group read in the order k,q,v would
/// otherwise return finite, plausible, wrong numbers with no assertion
/// anywhere.
pub fn check_seg_spans(
    key: &str,
    spans: &[(usize, usize, usize)],
    group_d_out: usize,
) -> Result<(), String> {
    if spans.is_empty() {
        return Err(format!("{key}: segmented group with no part at all"));
    }
    let mut at = 0usize;
    for (i, &(rank, row0, d_out)) in spans.iter().enumerate() {
        if rank != i {
            return Err(format!(
                "{key}: the projection at position {i} carries rank {rank}. The row order \
                 of the group is the one of Act::consumers(), and the split of the output \
                 depends on it."
            ));
        }
        if row0 != at {
            return Err(format!(
                "{key}: part {i} starts at row {row0}, {at} expected. The parts do not \
                 tile the group."
            ));
        }
        at += d_out;
    }
    if at != group_d_out {
        return Err(format!(
            "{key}: the parts cover {at} rows of the fused group's {group_d_out}"
        ));
    }
    Ok(())
}

/// One projection on its way into a group: its rank in `Act::consumers()`, the
/// checkpoint suffix that gave the rank, and the matrix itself.
type PendingPart = (usize, String, FusedMatrix);

/// Split a loaded model's matrices into fused groups and the projections that
/// stay alone (`o_proj`, `down_proj`).
///
/// **Consumes** the matrices: a group's parts are spliced into one stream, and
/// keeping the unfused copies alive would double the host's peak — 1.37 GB on
/// the published 4B — for nothing.
///
/// ## The grouping key is `(layer, Act)`, and never the name
///
/// `llvq-cuda/src/seg_host.rs::seg_family` matches `name.contains("q_proj")`
/// and is right to — its fixtures carry no rotation, so the name is the only
/// structure available. Here the artifact carries rotation keys, and
/// [`crate::rotplan::act_of_suffix`] states the rule outright: the name is a
/// **parallel channel** that can disagree with the file. So the family comes
/// from that function (which reads `Act::consumers()`, the model's own table)
/// and the rank is the position in that same slice — q,k,v then gate,up, the
/// order `seg_family` also produces, from one derivation instead of two. The
/// layer is part of the key because q/k/v of two different layers share a
/// `d_in` and would concatenate perfectly well into nonsense.
///
/// The rotation keys are re-asserted equal across the group even though
/// [`crate::rotplan::check_rotation_partition`] already ran, because this
/// function owns the row order and must not inherit a premise it cannot see.
pub fn segment_matrices(
    matrices: Vec<FusedMatrix>,
) -> Result<(Vec<FusedMatrix>, Vec<FusedGroup>), String> {
    use crate::model::Act;

    let mut singles: Vec<FusedMatrix> = Vec::new();
    // First-appearance order, so logs and errors follow the file rather than a
    // hash seed — `rotation_sites` does the same, for the same reason.
    let mut order: Vec<(usize, Act)> = Vec::new();
    let mut buckets: HashMap<(usize, Act), Vec<PendingPart>> = HashMap::new();

    for m in matrices {
        let (layer, suffix) = llvq_artifact::split_name(&m.name).map_err(|e| e.to_string())?;
        let act = crate::rotplan::act_of_suffix(&suffix).ok_or_else(|| {
            format!(
                "{}: \"{suffix}\" consumes none of the four activations of a Qwen3 block",
                m.name
            )
        })?;
        let consumers = act.consumers();
        if consumers.len() < 2 {
            // `o_proj` and `down_proj`: nothing else consumes their activation,
            // so there is no group to be part of.
            singles.push(m);
            continue;
        }
        let rank = consumers
            .iter()
            .position(|&c| c == suffix)
            .expect("act_of_suffix found this suffix in this very slice");
        let key = (layer, act);
        if !buckets.contains_key(&key) {
            order.push(key);
        }
        buckets.entry(key).or_default().push((rank, suffix, m));
    }

    let mut groups = Vec::with_capacity(order.len());
    for (layer, act) in order {
        let mut parts = buckets
            .remove(&(layer, act))
            .expect("every key pushed on `order` was inserted in the same breath");
        parts.sort_by_key(|&(rank, ..)| rank);
        groups.push(build_group(layer, act, parts)?);
    }
    Ok((singles, groups))
}

/// One group, from its parts already sorted by rank.
fn build_group(
    layer: usize,
    act: crate::model::Act,
    parts: Vec<PendingPart>,
) -> Result<FusedGroup, String> {
    let key = format!("{layer:03}.{act:?}");
    let want = act.consumers().len();
    if parts.len() != want {
        // An incomplete group is a broken file, never a degraded case: the
        // missing rows would silently shrink the fused matrix.
        return Err(format!(
            "{key}: {} projections out of the {want} this activation feeds ({})",
            parts.len(),
            act.consumers().join(", ")
        ));
    }
    let (d_in, nblocks, tail_w, rotation) = {
        let m = &parts[0].2;
        (m.d_in, m.nblocks, m.tail_w, m.rotation)
    };
    if rotation.is_none() {
        return Err(format!(
            "{key}: {} has no rotation. The fused path cannot read a matrix quantized in \
             the natural basis.",
            parts[0].2.name
        ));
    }

    // The spliced stream is built first, from borrows, so the parts can then be
    // consumed field by field without cloning 1.37 GB of payload.
    let words = {
        let streams: Vec<(&HostStream, usize)> = parts
            .iter()
            .map(|(_, _, m)| (&m.stream, m.d_out * nblocks))
            .collect();
        splice_planes14(&streams).map_err(|e| format!("{key}: {e}"))?
    };

    let mut d_out = 0usize;
    let mut payload = 0u64;
    let mut gscale: Vec<f32> = Vec::with_capacity(2 * want);
    let mut gs_off: Vec<u32> = Vec::new();
    let mut rscale: Vec<f32> = Vec::new();
    let mut tail: Vec<u16> = Vec::new();
    let mut seg_parts: Vec<SegPart> = Vec::with_capacity(want);

    for (i, (rank, suffix, m)) in parts.into_iter().enumerate() {
        if rank != i {
            return Err(format!(
                "{key}: {} carries rank {rank} at position {i}, so two projections share a \
                 rank, or a consumer is duplicated",
                m.name
            ));
        }
        if m.d_in != d_in {
            return Err(format!(
                "{key}: {} has d_in={}, the group has {d_in}. A group shares its \
                 activation, hence its block count and its tail.",
                m.name, m.d_in
            ));
        }
        if m.nblocks != nblocks || m.tail_w != tail_w {
            return Err(format!(
                "{key}: {} splits into {} blocks and a tail of {}, against {nblocks} and \
                 {tail_w} for the group",
                m.name, m.nblocks, m.tail_w
            ));
        }
        if m.rotation != rotation {
            return Err(format!(
                "{key}: {} carries rotation {:?}, {rotation:?} for the group. Hoisting \
                 would give one the basis of the other.",
                m.name, m.rotation
            ));
        }
        // Per part **and** on the total below. The kernels launch whole
        // 256-thread blocks of 8 rows and carry no bounds guard; asserting the
        // total alone would let an individually ragged part through on a lucky
        // sum, and then the *unfused* control arm could not be launched at all.
        if !m.d_out.is_multiple_of(8) {
            return Err(format!(
                "{key}: {} has d_out={}, which is not a multiple of 8",
                m.name, m.d_out
            ));
        }
        if m.rscale.len() != m.d_out || m.tail.len() != m.d_out * tail_w {
            return Err(format!(
                "{key}: {} carries {} row scales and {} tail values for {} \
                 rows of {tail_w}",
                m.name,
                m.rscale.len(),
                m.tail.len(),
                m.d_out
            ));
        }
        // The payload is 14 bytes a block whatever the grouping — structural
        // for `Planes14`, uniform stride and no bases. Cross-checked against
        // what `load` billed for this very matrix, so a drift between the two
        // accountings stops here rather than in a published b/weight.
        let part_payload = (PLANES14_BYTES * m.d_out * nblocks) as u64;
        if m.bytes != part_payload + matrix_side_bytes(m.d_out, tail_w) {
            return Err(format!(
                "{key}: {} bills {} bytes, {} expected (payload {part_payload} + \
                 tail and scales)",
                m.name,
                m.bytes,
                part_payload + matrix_side_bytes(m.d_out, tail_w)
            ));
        }

        seg_parts.push(SegPart {
            name: m.name,
            layer,
            proj: suffix,
            d_out: m.d_out,
            row0: d_out,
            rank,
        });
        gscale.extend_from_slice(&m.gscale);
        // Two centroids per part, so part `rank`'s pair starts at `2·rank`; one
        // entry per ROW, because the kernel reads it as `gs_off[row]`.
        gs_off.extend(std::iter::repeat_n(2 * rank as u32, m.d_out));
        rscale.extend_from_slice(&m.rscale);
        tail.extend_from_slice(&m.tail);
        payload += part_payload;
        d_out += m.d_out;
    }

    if !d_out.is_multiple_of(8) {
        return Err(format!(
            "{key}: d_out={d_out} in total, which is not a multiple of 8"
        ));
    }
    if gs_off.len() != d_out
        || rscale.len() != d_out
        || tail.len() != d_out * tail_w
        || gscale.len() != 2 * seg_parts.len()
    {
        return Err(format!(
            "{key}: inconsistent concatenation, {} gs_off, {} scales, {} tail values, \
             {} centroids for {d_out} rows of {tail_w} and {} parts",
            gs_off.len(),
            rscale.len(),
            tail.len(),
            gscale.len(),
            seg_parts.len()
        ));
    }

    Ok(FusedGroup {
        key,
        d_out,
        d_in,
        nblocks,
        tail_w,
        stream: HostStream::Planes14 { words },
        gscale,
        gs_off,
        rscale,
        tail,
        rotation,
        parts: seg_parts,
        // The payload does not move — 14 bytes a block, grouping or not — and
        // neither does `matrix_side_bytes`, which is additive in `d_out` at a
        // shared `tail_w`. What fusion adds is exactly `4 · d_out`.
        bytes: payload + matrix_side_bytes(d_out, tail_w) + d_out as u64 * 4,
    })
}

/// Everything a fused runtime needs, still on the host.
pub struct FusedModel {
    /// The runtime layout every matrix was transcoded to.
    pub layout: FusedLayout,
    /// The projections launched one at a time. **All** of them under
    /// [`FuseMode::Off`](crate::fused::FuseMode::Off); only `o_proj` and `down_proj` under `On`, the rest
    /// having moved into [`Self::groups`].
    pub matrices: Vec<FusedMatrix>,
    /// The row-concatenated groups, empty under [`FuseMode::Off`](crate::fused::FuseMode::Off) — where this
    /// struct is then byte-identical to what shipped before this lot.
    pub groups: Vec<FusedGroup>,
    /// The `Int4G128` projections, empty for every file written before
    /// `LLVQ_INT4_TYPES` existed — which is every published artifact. They sit
    /// beside the lattice matrices rather than among them because they share
    /// no code path with them: a different kernel, a different basis, a
    /// different packing order.
    pub int4: Vec<FusedInt4>,
    pub rotations: HashMap<RotKey, RotationTables>,
    /// Embedding and norms, carried verbatim, **still in the file's own
    /// encoding** — f16 bits, or int8 g64 for an `embedq` output. Decoding is
    /// the consumer's decision: the f16 path widens, the q8 path keeps the
    /// bytes. Holding the encoded form also halves what this struct weighs
    /// (the 778 MB embedding would be 1.5 GB as f32).
    pub raw: Vec<llvq_artifact::RawTensor>,
    pub config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy at runtime, in the chosen layout.
    pub runtime_bytes: u64,
}

impl FusedModel {
    /// Bits per weight the runtime actually spends on quantized projections.
    ///
    /// Not the same accounting as the file's: the runtime layout pays its
    /// addressing where the file packs 48 bits per block exactly. On the
    /// published 4B: `Slot32` (byte-rounded group strides plus a `u32` base
    /// per group) measured 5.510 b/weight, `Planes14` (uniform 14-byte
    /// records, no bases) 4.804, `Planes12x` (12-byte records plus the
    /// exception table) 4.342 at the bench, against 2.0702 effective in the
    /// file.
    ///
    /// 🚨 **This function no longer reproduces those three bench numbers, and
    /// the gap is a real difference of objects, not a drift.** Since lot A7a
    /// the inference path holds the `KeepExact` tail as binary16 while every
    /// bench arm (`planesbench`, and `rtbits`'s "b/weight kernel" column which
    /// models it) still uploads `f32`. The whole difference is
    /// `16 · tail_weights / weights`: **−0.0747 b/weight on the published 4B**
    /// (16,957,440 tail weights of 3,633,315,840) and **−0.0462 on the 8B**
    /// (20,054,016 of 6,945,767,424), so `Planes14` reads 4.729 here against
    /// the bench's 4.804 at 4B, and 4.706 against 4.752 at 8B. Two
    /// accountings of two different residencies — never subtract one from the
    /// other and call it a measurement.
    ///
    /// ⚠️ This crate's `Planes12x` figure additionally sits marginally *above*
    /// the bench's for an unrelated reason: it also bills the `d_out + 1`
    /// row-offset table the inference kernel reads, which the bench arm has no
    /// equivalent of (32/`d_in` b/weight — 0.0125 at `d_in = 2560`).
    /// Under-reporting a device array is exactly the accounting sin the K-1
    /// lot spent a dossier removing.
    pub fn runtime_bits_per_weight(&self) -> f64 {
        self.runtime_bytes as f64 * 8.0 / self.quantized_weights as f64
    }
}
