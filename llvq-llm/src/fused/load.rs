//! Reading a sealed file into a [`FusedModel`] without decoding it ([`load`],
//! [`load_with`]), and [`FusedSealed`], what a backend's loader hands back.

use std::collections::HashMap;
use std::io::Read;

use super::host::runtime_bytes_of;
use super::{
    check_kinds, lattice_kind, matrix_side_bytes, seg_kernel_name, segment_matrices, serves_kind, tail_f16_bits,
    EmbedMode, FuseMode, FusedInt4, FusedLayout, FusedMatrix, FusedModel, RotKey, RotationTables, Transcoder,
};

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

/// Load a sealed artifact in encoded form, transcoded to `layout`.
///
/// Costs one transcode of every matrix. Under `Planes14` that took 142 s for
/// the 4B on eight cores, measured 2026-08-05, against `sealed::load`'s full
/// decode, which took 208.7 s on the same file and produced eight gigabytes
/// of f16. Under `Tetra48`, the served layout, the transcode only copies six
/// bytes a block into the row layout (`llvq_artifact::tetra48`).
///
/// ⚠️ `Planes12x` is a different order of magnitude and it is not a defect:
/// it runs one exact lattice search per 5-level block on top of the same
/// work. Measured on this machine (M3 Max, 12 performance cores) by
/// `transcode_of_the_sealed_model_matches_planes14`, a sweep deleted on
/// 2026-10-07 together with the August archive it read.
pub fn load(path: &str, layout: FusedLayout) -> Result<FusedModel, String> {
    load_with(path, layout, FuseMode::Off)
}

/// The most elements a reader reserves up front for a count it read from the
/// file, the bound `llvq_artifact` keeps private for its own readers. A torn
/// or hostile header can announce four billion records; reserving that many
/// would abort the process before the first short read could return an error.
/// A valid file only grows the vector past it, element by element.
pub const PREALLOC_CAP: usize = 1 << 20;

/// [`load`] with the fusion mode named by the caller rather than read from the
/// environment.
///
/// Under [`FuseMode::Off`] this is `load` as it always was: [`FusedModel::
/// groups`] is empty, `matrices` carries all 252 projections of the published
/// 4B, and `runtime_bytes` is the same sum it always was. Under `On` the
/// projections that share an activation are spliced into [`FusedGroup`]s — 72
/// groups plus 72 lone projections, so 144 matvec launches a token.
pub fn load_with(path: &str, layout: FusedLayout, fuse: FuseMode) -> Result<FusedModel, String> {
    // Refused here as well as in `fused_cuda::load_with`, and on purpose: this
    // is the side of the boundary a test on a machine without a card reaches.
    // Without it `segment_matrices` would be handed exception-carrying streams
    // whose only symptom is a wrong number.
    if fuse == FuseMode::On && seg_kernel_name(layout).is_none() {
        return Err(format!(
            "LLVQ_FUSE=1 with LLVQ_FUSED_LAYOUT={}: only planes14 segments (see \
             seg_kernel_name)",
            layout.name()
        ));
    }
    let file_bytes = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    if !head.is_self_contained() {
        return Err(format!(
            "{path} is only a projections artifact (format v{}), it cannot run on its \
             own.",
            head.version
        ));
    }

    // The header's whole set of kinds, not its default: one Tetra matrix among
    // four hundred Ball ones is a file this path cannot serve, and the refusal
    // has to land here, before any record is read through `read_matrix_raw` as
    // a Ball record.
    check_kinds(layout, head.kinds())?;
    let tr = Transcoder::for_kind(layout, lattice_kind(layout))?;
    let mut matrices = Vec::with_capacity((head.matrices as usize).min(PREALLOC_CAP));
    let mut int4: Vec<FusedInt4> = Vec::new();
    let mut rotations: HashMap<RotKey, RotationTables> = HashMap::new();
    let mut quantized_weights = 0usize;

    for _ in 0..head.matrices {
        // `read_record` and not `read_matrix_raw`: the latter reads every
        // record as a lattice body whatever its kind tag says, which for an
        // int4 record is not a refusal but a misparse — it would consume the
        // wrong number of bytes and desynchronise every record after it.
        let rec = llvq_artifact::read_record(&mut r, head.version).map_err(|e| e.to_string())?;
        // The record's own kind, not only the header's set: the set is a
        // declaration, and a file whose header under-reports it would reach
        // the transcoder here with the wrong map in hand.
        if !serves_kind(layout, rec.kind()) {
            return Err(format!(
                "{}: a {:?} record in a file served as {}",
                rec.name(),
                rec.kind(),
                layout.name()
            ));
        }
        let m = match rec {
            llvq_artifact::Record::Int4(q) => {
                // Nothing to transcode: the kernel reads these bytes. The
                // weights count toward the model's quantized total exactly as
                // a lattice matrix's do — they are quantized, just not by us.
                quantized_weights += q.d_out * q.d_in;
                let bytes = (q.packed.len() + 2 * q.scales.len() + 2 * q.biases.len()) as u64;
                int4.push(FusedInt4 {
                    name: q.name,
                    d_out: q.d_out,
                    d_in: q.d_in,
                    group: q.group,
                    packed: q.packed,
                    scales: q.scales,
                    biases: q.biases,
                    bytes,
                });
                continue;
            }
            llvq_artifact::Record::Lattice(m) => m,
        };
        // Every lattice decoder hard-codes one gain bit (`hdr >> 9`). A file
        // with a different gain width would transcode into a coherent stream
        // and decode into garbage — silently. Lattice-only since 2026-09-09:
        // an int4 record carries no centroid at all, so this check used to
        // fire on exactly the record a mixed file most needs.
        if m.centroids.len() != 2 {
            return Err(format!(
                "{}: {} centroids, but the kernels hardcode 1 gain bit",
                m.name,
                m.centroids.len()
            ));
        }
        let nblocks = m.d_in / llvq_core::DIM;
        let tail_w = m.d_in % llvq_core::DIM;
        quantized_weights += m.d_out * m.d_in;

        let rotation = match m.rotation_seed {
            None => None,
            Some(seed) => {
                let key = (m.d_in, seed);
                // `entry` rather than contains/insert, but not for the reason
                // clippy gives: `RotationTables::build` runs a Gram–Schmidt and
                // allocates, and the naive form would do that work on every
                // matrix sharing a rotation — 252 builds where 144 are owed.
                if let std::collections::hash_map::Entry::Vacant(e) = rotations.entry(key) {
                    e.insert(RotationTables::build(m.d_in, seed)?);
                }
                Some(key)
            }
        };

        let (stream, payload) = tr
            .stream(&m.indices, &m.gains, m.d_out, nblocks)
            .map_err(|e| format!("{} : {e}", m.name))?;
        let bytes = payload + matrix_side_bytes(m.d_out, tail_w);

        matrices.push(FusedMatrix {
            name: m.name,
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks,
            tail_w,
            stream,
            gscale: [m.centroids[0] as f32, m.centroids[1] as f32],
            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
            tail: tail_f16_bits(&m.tail),
            rotation,
            bytes,
        });
    }

    // The rotation keys must partition the activation sites *before* anything
    // is uploaded: lot A4 hoists one rotation per site, and a file where two
    // consumers of one activation carry different rotations — or where two
    // activations share one — would make that hoist hand a projection the
    // wrong basis. Refused by name here rather than degraded silently, and
    // checked on every load so the `LLVQ_ROT_SHARE=0` arm rests on the same
    // premise as the `=1` arm.
    crate::rotplan::check_rotation_partition(&matrices)?;

    let (matrices, groups) = match fuse {
        FuseMode::Off => (matrices, Vec::new()),
        FuseMode::On => segment_matrices(matrices)?,
    };
    // Re-derived rather than patched: the splice moves payload between the two
    // vectors and adds `gs_off`, so an accumulator carried through the read
    // loop would have to be corrected twice. One sum over what the model
    // actually holds cannot drift from what it holds.
    //
    // 🕳️ The int4 term was missing, and it was missing on ONE side of a
    // division. `quantized_weights` counts an int4 record's weights — it says
    // so where it does it, "they are quantized, just not by us" — so
    // `runtime_bits_per_weight` divided a numerator without those bytes by a
    // denominator with those weights. On the served mixed file that prints
    // **2.030 b/weight where the truth is 2.141**, and a card footprint 50 MB
    // light, on the line `fusedrun` publishes. Nothing failed; the number was
    // simply wrong, and the test below now divides the two the same way.
    let runtime_bytes = runtime_bytes_of(&matrices, &groups, &int4);

    let n_raw = read_u32(&mut r)?;
    let mut raw = Vec::with_capacity((n_raw as usize).min(PREALLOC_CAP));
    let mut carried_weights = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version).map_err(|e| e.to_string())?;
        carried_weights += t.len();
        raw.push(t);
    }

    let n_blob = read_u32(&mut r)?;
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r).map_err(|e| e.to_string())?;
        blobs.insert(b.name, b.bytes);
    }

    Ok(FusedModel {
        int4,
        layout,
        matrices,
        groups,
        rotations,
        raw,
        config_json: blobs
            .remove("config.json")
            .ok_or_else(|| format!("{path} carries no config.json"))?,
        tokenizer_json: blobs
            .remove("tokenizer.json")
            .ok_or_else(|| format!("{path} carries no tokenizer.json"))?,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
    })
}

// ---------------------------------------------------------------------------
// What a backend's loader hands back.
//
// It lived in `fused_cuda/` until the device port, where it was reachable
// only on a CUDA build. A second loader then had no return type, so "Metal
// writes one new file" was false. Every field but one was already portable.
// ---------------------------------------------------------------------------

/// A model rebuilt from a sealed artifact **with its projections still
/// encoded**, plus what it took to do so.
pub struct FusedSealed {
    pub model: crate::model::Qwen3,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: candle_transformers::models::qwen3::Config,
    /// The runtime layout the projections were transcoded to.
    pub layout: FusedLayout,
    /// The `(rows, tile)` the PREFILL kernel was compiled at. Carried out of
    /// the runtime because a binary that prints "ceil(N/4) launches" while its
    /// unit was compiled at eight is describing a run that did not happen.
    ///
    /// [`crate::device::Prefill`] and not `llvq_cuda::tile::Prefill`, which
    /// holds the same three fields. `llvq-cuda` is target-gated on Linux and
    /// pulls `cudarc` there, so naming it from this module would drag the
    /// driver into a CPU-only build. The adapter converts at its boundary.
    pub prefill: crate::device::Prefill,
    /// How the embedding and tied `lm_head` sit on the device.
    pub embed_mode: EmbedMode,
    /// Whether a shared activation is rotated once per group (`LLVQ_ROT_SHARE`).
    pub rot_share: crate::rotplan::RotShare,
    /// `rot_apply` launches one decode token costs. Printed on both arms: a
    /// gate showing identical tokens at 252 launches each proves nothing.
    pub rot_launches: usize,
    /// Whether the projections that share an activation were row-concatenated
    /// into one launch (`LLVQ_FUSE`).
    pub fuse: FuseMode,
    /// Matvec launches one decode token costs — 252 unfused on the published
    /// 4B, 144 fused. Printed on the arm line for the same reason
    /// [`Self::rot_launches`] is: a gate showing identical tokens while both
    /// arms issued 252 matvecs proves the tokens and nothing about the lot.
    pub matvec_launches: usize,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy on the device — the number that decides
    /// whether a model fits, and the one a disk figure must never stand in for.
    pub runtime_bytes: u64,
    /// Bytes the carried tensors occupy on the device: `2 · carried_weights`
    /// under `LLVQ_EMBED=f16`, the int8 payload of **every** embedding table
    /// plus the f16 norms under `q8` — one table when the model ties its two
    /// ends, two when it unties them. `carried_weights · 2` must no longer
    /// stand in for this: that identity is exactly what q8 breaks, by −365 MB
    /// on the tied 4B and −1.17 GB on the untied 8B.
    pub carried_bytes: u64,
}
