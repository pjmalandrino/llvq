// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §3.6-3.7), revised the same day after
// review. Not part of the repository.
//
// The kernel choice for a rotation width and for an int4 activation width,
// and the dynamic shared size of each int4 entry point, as pure functions next
// to `llvq_cuda::shared::plan` (llvq-cuda/src/shared.rs), so the decision is
// tested on the Mac where tests are free (the reason shared.rs gives at its
// lines 20-28). The guiding rule: a width keeps today's kernel whenever
// today's kernel can run it, so no served object of the 4B, 8B or 14B changes
// path; the new kernels serve only what is refused today.
//
// Integration (not shown):
//  * Rotation. `RotBuffers` (llvq-llm/src/fused_cuda.rs:78, built in
//    `upload_rotation` at :2047-2061) gains `kernel: RotKernel`; the load path
//    (:756-783) loads the wide unit only when some width maps to `Device`,
//    allocates one scratch of max over those widths of
//    (PREFILL_ROWS * n + (KMAX - k) * m) floats, and adds the entry points to
//    `spill_checked` (:571); the launch sites (:1423, :1493, :1940) branch on
//    `kernel`. The opt-in on `rot_apply` stays posed for the widest SHARED
//    rotation only, so a 32B's 5,120 and 8,192 sites run without one.
//  * Int4. Today the load path poses ONE size, `int4_shared_bytes(all
//    records)` = max d_in * 4, on BOTH `tv_q4_h` and `tv_q4_hh`
//    (fused_cuda.rs:641-660), and `upload_int4` checks and stores
//    `shared = d_in * 4` per record (:2162-2173, :2198). A 32B with one int4
//    down_proj at 25,600 therefore asks 102,400 B and fails at load, new kernel
//    or not. The change: `int4_entry_bytes` below gives each entry point its
//    own size, each through `func_dynamic_shared` (which poses the opt-in when
//    the size passes the default, so `tv_q4_hs` gets its own at 51,200 B);
//    `upload_int4` takes the record's `Int4Staging` and stores and checks the
//    STAGED size; the launch reads `p.shared` as today (:2285).

use crate::shared::{ceiling, plan, rot_bytes, Fit};

/// Which rotation kernel a width gets on this card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotKernel {
    /// `rot_apply` (or, after its gates, `rot_apply_r`): the whole vector in
    /// shared memory, under the default or after the opt-in.
    Shared(Fit),
    /// `rot_apply_dev` (or, after its gates, `rot_apply_rw`): staged in device
    /// memory. Chosen only when `Shared` is refused.
    Device,
}

pub fn rot_kernel(n: usize, default_limit: usize, optin_limit: usize) -> RotKernel {
    match plan(rot_bytes(n), default_limit, optin_limit) {
        Some(fit) => RotKernel::Shared(fit),
        None => RotKernel::Device,
    }
}

/// Which staging the int4 matvec gets for an activation of `d_in`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int4Staging {
    /// `tv_q4_h` / `tv_q4_hh`, f32 staging, today's kernels.
    F32,
    /// `tv_q4_hs`, f16 staging, half the bytes, same bits. Takes an f16
    /// activation only: an f32 activation at such a record goes to an f32-input
    /// tiled kernel with the same chain, or is refused by name.
    F16,
}

impl Int4Staging {
    pub fn bytes(self, d_in: usize) -> usize {
        match self {
            Int4Staging::F32 => d_in * 4,
            Int4Staging::F16 => d_in * 2,
        }
    }
}

pub fn int4_staging(d_in: usize, default_limit: usize, optin_limit: usize) -> Option<Int4Staging> {
    let top = ceiling(default_limit, optin_limit);
    if d_in * 4 <= top {
        Some(Int4Staging::F32)
    } else if d_in * 2 <= top {
        Some(Int4Staging::F16)
    } else {
        None
    }
}

/// The dynamic shared size each int4 entry point is loaded with: `f32` for
/// `tv_q4_h` and `tv_q4_hh`, over the F32-staged records only; `f16` for
/// `tv_q4_hs`, over the F16-staged ones. `None` means that entry point is not
/// loaded at all. Refuses a model with a record no staging fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int4EntryBytes {
    pub f32: Option<usize>,
    pub f16: Option<usize>,
}

pub fn int4_entry_bytes(
    d_ins: impl IntoIterator<Item = usize>,
    default_limit: usize,
    optin_limit: usize,
) -> Result<Int4EntryBytes, String> {
    let (mut f32, mut f16) = (None::<usize>, None::<usize>);
    for d in d_ins {
        match int4_staging(d, default_limit, optin_limit) {
            Some(Int4Staging::F32) => f32 = Some(f32.map_or(d * 4, |b| b.max(d * 4))),
            Some(Int4Staging::F16) => f16 = Some(f16.map_or(d * 2, |b| b.max(d * 2))),
            None => {
                return Err(format!(
                    "an int4 record of d_in {d} fits neither staging: {} B in f32, {} B in f16, \
                     against {} B on this card",
                    d * 4,
                    d * 2,
                    ceiling(default_limit, optin_limit)
                ))
            }
        }
    }
    Ok(Int4EntryBytes { f32, f16 })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixtures: L40S measured (fusedrun-14b-2026-08-17); A100 measured
    // (f4-a100-2026-08-18.txt:95); H200 from the CUDA guide's 227 KB for
    // compute capability 9.0 (never read off a card in this repository).
    const L40S: (usize, usize) = (49_152, 101_376);
    const A100: (usize, usize) = (49_152, 166_912);
    const H200: (usize, usize) = (49_152, 232_448);

    #[test]
    fn served_widths_keep_todays_kernel_and_only_the_32b_moves() {
        let (d, o) = L40S;
        assert_eq!(rot_kernel(2_560, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(9_728, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(12_288, d, o), RotKernel::Shared(Fit::Default)); // on the bound
        assert_eq!(rot_kernel(17_408, d, o), RotKernel::Shared(Fit::OptIn));
        assert_eq!(rot_kernel(25_600, d, o), RotKernel::Device);
        // the widest odd factor under KMAX: 31 * 1,024
        assert_eq!(rot_kernel(31_744, d, o), RotKernel::Device);
    }

    #[test]
    fn the_32b_needs_no_lift_on_cards_with_a_larger_opt_in() {
        for (d, o) in [A100, H200] {
            assert_eq!(rot_kernel(25_600, d, o), RotKernel::Shared(Fit::OptIn));
            assert_eq!(int4_staging(25_600, d, o), Some(Int4Staging::F32));
        }
    }

    #[test]
    fn int4_moves_to_f16_staging_only_past_the_ceiling() {
        let (d, o) = L40S;
        for d_in in [2_560, 4_096, 5_120, 8_192, 9_728, 12_288, 17_408, 25_344] {
            assert_eq!(int4_staging(d_in, d, o), Some(Int4Staging::F32), "{d_in}");
        }
        assert_eq!(int4_staging(25_600, d, o), Some(Int4Staging::F16)); // 51,200 B
        assert_eq!(int4_staging(60_000, d, o), None);                   // 120,000 B
    }

    /// The pattern of `the_int4_staging_fits_the_card_at_every_sealed_size`
    /// (llvq-llm/src/fused.rs:2484), per entry point, with a 32B shape added.
    #[test]
    fn each_int4_entry_point_gets_its_own_size_at_every_shape() {
        let (d, o) = L40S;
        // (hidden, o_proj d_in, intermediate): Qwen3-4B, 8B, 14B: unchanged,
        // one F32 size, nothing staged in f16.
        let sealed = [((2560, 4096, 9728), 38_912), ((4096, 4096, 12288), 49_152), ((5120, 5120, 17408), 69_632)];
        for ((h, op, i), want) in sealed {
            let e = int4_entry_bytes([h, op, i], d, o).unwrap();
            assert_eq!(e, Int4EntryBytes { f32: Some(want), f16: None }, "{h}/{op}/{i}");
        }
        // Qwen3-32B on the L40S: v at 5,120, o at 8,192 stay F32 (32,768 B,
        // under the default); down at 25,600 goes F16 (51,200 B, past the
        // default, so `tv_q4_hs` takes its own opt-in).
        let e = int4_entry_bytes([5_120, 8_192, 25_600], d, o).unwrap();
        assert_eq!(e, Int4EntryBytes { f32: Some(32_768), f16: Some(51_200) });
        assert_eq!(plan(51_200, d, o), Some(Fit::OptIn));
        assert_eq!(plan(32_768, d, o), Some(Fit::Default));
        // Qwen3-32B on the A100: everything F32, one opt-in at 102,400 B.
        let (da, oa) = A100;
        let e = int4_entry_bytes([5_120, 8_192, 25_600], da, oa).unwrap();
        assert_eq!(e, Int4EntryBytes { f32: Some(102_400), f16: None });
        // No record at all: nothing loaded.
        assert_eq!(int4_entry_bytes([], d, o).unwrap(), Int4EntryBytes { f32: None, f16: None });
        // A record no staging fits is refused, by its width.
        assert!(int4_entry_bytes([60_000], d, o).unwrap_err().contains("60000"));
    }

    // Mutants this file must kill: `<` for `<=` in either branch (12,288 and
    // 25,344 sit on bounds); F16 returned before F32 (moves every served int4
    // record); Device returned for OptIn (moves the 14B's down rotation); the
    // F16 record's bytes counted as d_in * 4 (the 32B fails at load again);
    // the f32 entry size taken over ALL records (the bug of today's path).
}
