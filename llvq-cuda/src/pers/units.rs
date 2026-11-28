//! The NVRTC units the doors compile, as ordered lists of embedded parts.
//! One file so the parts lists the pinned digests hash stay together, each
//! list one item in its original order.

// ---------------------------------------------------------------------------
// The NVRTC units, embedded so the Mac can hash and compose what the card
// compiles. Parts are joined by "\n" exactly as `gpu::KernelSource::new` does.
// ---------------------------------------------------------------------------

/// The persistent module: L1's `tv_tetra48_persall` and microbenchmarks,
/// L2's fifteen arms, the L2 prologue reference, and route P's segment.
/// Its own module, so neither the bench unit nor the served unit moves.
pub const PERS_UNIT_PARTS: [(&str, &str); 13] = [
    ("llvq_slot.cuh", include_str!("../../kernels/llvq_slot.cuh")),
    ("matvec.cu", include_str!("../../kernels/matvec.cu")),
    ("llvq_f1rank.cuh", include_str!("../../kernels/llvq_f1rank.cuh")),
    ("llvq_f1rank_v3.cuh", include_str!("../../kernels/llvq_f1rank_v3.cuh")),
    ("llvq_tetra48.cuh", include_str!("../../kernels/llvq_tetra48.cuh")),
    ("llvq_rot.cuh", include_str!("../../kernels/llvq_rot.cuh")),
    ("rotate.cu", include_str!("../../kernels/rotate.cu")),
    ("llvq_pers.cuh", include_str!("../../kernels/llvq_pers.cuh")),
    ("tetra48_pers.cu", include_str!("../../kernels/tetra48_pers.cu")),
    ("tetra48_persall.cu", include_str!("../../kernels/tetra48_persall.cu")),
    ("pers_ref.cu", include_str!("../../kernels/pers_ref.cu")),
    ("pers_segment.cu", include_str!("../../kernels/pers_segment.cu")),
    ("pers_4e.cu", include_str!("../../kernels/pers_4e.cu")),
];

/// The persistent module's text for a tile (`tile::Tile::define()`).
pub fn pers_unit_text(tile_define: &str) -> String {
    std::iter::once(tile_define).chain(PERS_UNIT_PARTS.iter().map(|p| p.1)).collect::<Vec<_>>().join("\n")
}

/// The served unit of the 4B, in the order `fused_cuda.rs` assembles it for
/// a Tetra48 file with int4 records and `LLVQ_EMBED=q4`: the defines, then
/// these ten parts. The L2 bench compiles it as its own module so its
/// `tv_tetra48_h` is the served kernel's text, and prints its digest; the job
/// compares that digest with the served door's.
pub const SERVED_UNIT_PARTS: [(&str, &str); 10] = [
    ("llvq_slot.cuh", include_str!("../../kernels/llvq_slot.cuh")),
    ("matvec.cu", include_str!("../../kernels/matvec.cu")),
    ("llvq_rot.cuh", include_str!("../../kernels/llvq_rot.cuh")),
    ("rotate.cu", include_str!("../../kernels/rotate.cu")),
    ("llvq_f1rank.cuh", include_str!("../../kernels/llvq_f1rank.cuh")),
    ("llvq_f1rank_v3.cuh", include_str!("../../kernels/llvq_f1rank_v3.cuh")),
    ("llvq_tetra48.cuh", include_str!("../../kernels/llvq_tetra48.cuh")),
    ("tv_tetra48_h.cu", include_str!("../../../llvq-llm/kernels/tv_tetra48_h.cu")),
    ("emb_q4.cu", include_str!("../../../llvq-llm/kernels/emb_q4.cu")),
    ("tv_q4_h.cu", include_str!("../../../llvq-llm/kernels/tv_q4_h.cu")),
];

/// The served unit's text for its defines (`tile.define()` then
/// `Prefill::defines()`).
pub fn served_unit_text(defines: &str) -> String {
    std::iter::once(defines).chain(SERVED_UNIT_PARTS.iter().map(|p| p.1)).collect::<Vec<_>>().join("\n")
}

/// The tiled int4 readers (`tv_q4_tile.cu`), the part `fused_cuda.rs`
/// appends after [`SERVED_UNIT_PARTS`] for a runtime that holds its int4
/// words tiled: a config whose rows are "v9".
pub const SERVED_UNIT_TILE_PART: (&str, &str) = ("tv_q4_tile.cu", include_str!("../../../llvq-llm/kernels/tv_q4_tile.cu"));

/// The served unit under a v9 config: [`served_unit_text`], then the tiled
/// int4 readers. The other configs' unit does not move.
pub fn served_unit_text_tiled(defines: &str) -> String {
    format!("{}\n{}", served_unit_text(defines), SERVED_UNIT_TILE_PART.1)
}

/// The served unit's defines at the served tile of the L40S and the default
/// prefill: what every published 4B number since 2026-09-20 compiled.
pub const SERVED_DEFINES_SM89: &str = "#define TILE_BLOCKS 64u\n#define TETRA48_ROWS 4u\n#define TETRA48_TILE 128u\n";
