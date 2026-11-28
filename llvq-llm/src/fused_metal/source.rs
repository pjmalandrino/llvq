//! The kernel texts and [`source_of`], the table of which text carries which entry point. One file because the
//! table has to name every unit, including the ones `moe`, `moe_rows` and `wave1` assemble.

use candle_core::Result;

use super::moe::{SRC_MOE_ROT, SRC_MOE_TETRA};
use super::moe_rows::SRC_MOE_ROWS;
use super::wave1::{SRC_E4_XF, SRC_TETRA_ROWS_F32};

const SRC_TETRA: &str = include_str!("../../kernels/llvq_tetra48.metal");
const SRC_ROT: &str = include_str!("../../kernels/llvq_rot.metal");
const SRC_Q4: &str = include_str!("../../kernels/tv_q4_h.metal");
const SRC_Q8: &str = include_str!("../../kernels/emb_q8.metal");
const SRC_E4: &str = include_str!("../../kernels/emb_q4.metal");
/// The row-loop twins of the Tetra matvec: the served unit's text, then the
/// new kernels. Compiled as its own library, so the served pipelines, which
/// compile from `SRC_TETRA` alone, keep their register allocation.
const SRC_TETRA_ROWS: &str =
    concat!(include_str!("../../kernels/llvq_tetra48.metal"), include_str!("../../kernels/llvq_tetra48_rows.metal"));
/// The prefetching twins of the int4 matvecs, on the same rule.
const SRC_Q4_PF: &str = concat!(include_str!("../../kernels/tv_q4_h.metal"), include_str!("../../kernels/tv_q4_pf.metal"));

/// Which file carries which entry point.
///
/// Four translation units, not one, for the reason `tetra48_seg.cu` gives on
/// the CUDA side: appending to a shipped unit changes its bytes and can move
/// the register allocation of a kernel no correctness test can see move.
pub(super) fn source_of(name: &str) -> Result<&'static str> {
    match name {
        "tetra48_probe" | "tv_tetra48_metal" | "tv_tetra48_metal_ar" | "tv_tetra48_metal_arh" | "tv_tetra48_metal_lut" | "tv_tetra48_metal_lutg" | "tv_tetra48_metal_ilp" | "tv_tetra48_metal_u2" | "tv_tetra48_metal_c3" => Ok(SRC_TETRA),
        "rot_apply_metal" | "rot_apply_rows_metal" | "rot_apply_tg_metal" => Ok(SRC_ROT),
        "tv_q4_metal" | "tv_q4_metal_tiled" => Ok(SRC_Q4),
        "tv_tetra48_metal_lutp" | "tv_tetra48_metal_lutd" | "tv_tetra48_metal_lutdp" | "tv_tetra48_metal_lutd2"
        | "tv_tetra48_metal_lutd4" | "tv_tetra48_metal_luth" | "tv_tetra48_metal_lutdh"
        | "tv_tetra48_metal_lutdr" | "tv_tetra48_metal_lutdt" | "tv_tetra48_metal_lutdth" => Ok(SRC_TETRA_ROWS),
        "tv_q4_metal_p" | "tv_q4_metal_tiled_p" | "tv_q4_metal_h" | "tv_q4_metal_tiled_h"
        | "tv_q4_metal_rows" => Ok(SRC_Q4_PF),
        "emb_q8_gather_metal" | "tv_q8_metal" => Ok(SRC_Q8),
        "emb_q4_gather_metal" | "tv_emb_q4_metal" => Ok(SRC_E4),
        "tv_tetra48_metal_lutdrf" => Ok(SRC_TETRA_ROWS_F32),
        "tv_emb_q4_metal_xf" => Ok(SRC_E4_XF),
        "moe_rot_metal" => Ok(SRC_MOE_ROT),
        "moe_tetra_gu_metal" | "moe_tetra_down_metal" | "moe_tetra_down_sg_metal" => Ok(SRC_MOE_TETRA),
        "moe_rows_lutdrf" | "moe_pairs_sum" => Ok(SRC_MOE_ROWS),
        other => candle_core::bail!("no Metal source carries {other}"),
    }
}
