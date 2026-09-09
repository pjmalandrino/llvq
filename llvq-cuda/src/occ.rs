//! A3 — the occupancy arms of `planesbench`'s fusion section: their
//! registry, the `LLVQ_SEG_ARMS` selector, and every piece of launch
//! arithmetic the kernels in `kernels/planes_occ.cu` depend on.
//!
//! Portable on purpose, for the reason `arms.rs` and `shared.rs` give: the
//! bench is entirely `cfg(target_os = "linux")`, so nothing here would ever
//! be exercised on the development Mac if it lived there — and a grid that
//! is one CTA short, or a slice table that leaves a block out, corrupts on a
//! rented card without a log line. The two helpers the kernel text carries
//! (`occ_xs_index`, `occ_slice`) are mirrored here and diffed against the
//! compiled kernel text by `tests/planes_occ_matches_rust.rs`.
//!
//! ## The contract of `LLVQ_SEG_ARMS`
//!
//! A comma-separated list of arm names, or unset. Unset means **the four
//! historic arms only** — a bare `planesbench` prints exactly the fusion
//! block it printed before A3 existed. The arms named here are appended
//! AFTER those four, in registration order whatever order the variable
//! lists them (the `arms.rs` rule: a selection skips arms, it never moves
//! them). An unknown or duplicated name is a hard error, never a silent
//! fallback — the `LLVQ_FUSED_LAYOUT` rule.
//!
//! Prereg: `proofs/preregistration-a2-a3-geometrie-2026-08-31.md` §5.

use crate::TILE_BLOCKS;

/// Floats per staged block under the padded stride — `LLVQ_XS_PAD` in the
/// kernel, pinned by `the_padded_stride_matches_the_kernel`.
pub const XS_PAD: usize = 28;
/// Floats per staged block under the reference stride: one block.
/// A literal, not `llvq_core::DIM` — that crate is a Linux-only dependency
/// of this one — and pinned to it by a test where the dev-dependency exists.
pub const XS_DIM: usize = 24;
/// `u64` words per site descriptor of the persistent-global arm —
/// `LLVQ_OCC_SITE_WORDS` in the kernel.
pub const SITE_WORDS: usize = 8;
/// Rows per CTA at 256 threads: one warp per row.
pub const ROWS_PER_CTA: u32 = 8;

/// Registration order — the dispatch order after the four historic arms.
pub const SEG_ARM_NAMES: [&str; N_SEG_ARMS] =
    ["pad", "mr2", "mr4", "mr2p", "pers", "sk1", "sk2", "persall"];
pub const N_SEG_ARMS: usize = 8;

pub const PAD: usize = 0;
pub const MR2: usize = 1;
pub const MR4: usize = 2;
pub const MR2P: usize = 3;
pub const PERS: usize = 4;
pub const SK1: usize = 5;
pub const SK2: usize = 6;
pub const PERSALL: usize = 7;

/// The kernel each arm dispatches. `sk1` and `sk2` share one kernel and
/// differ by `nsplit` (see [`sk_nsplit`]).
pub const SEG_KERNEL: [&str; N_SEG_ARMS] = [
    "tv_planes_pad",
    "tv_planes_mr2",
    "tv_planes_mr4",
    "tv_planes_mr2p",
    "tv_planes_pers",
    "tv_planes_sk",
    "tv_planes_sk",
    "tv_planes_persall",
];

/// The row each arm prints in the fusion block.
pub const SEG_DISPLAY: [&str; N_SEG_ARMS] = [
    "Planes14 fused, staging at 28 (pad)",
    "Planes14 fused, 2 rows/warp (mr2)",
    "Planes14 fused, 4 rows/warp (mr4)",
    "Planes14 fused, mr2 + pad (mr2p)",
    "Planes14 fused, persistent/site (pers)",
    "Planes14 fused, split-K ×tiles (sk1)",
    "Planes14 fused, split-K ×2 tiles (sk2)",
    "Planes14 fused, 1 launch/round (persall)",
];

/// Rows per warp — the grid divisor of the multi-row arms.
pub const ROWS_PER_WARP: [u32; N_SEG_ARMS] = [1, 2, 4, 2, 1, 1, 1, 1];

/// Floats per staged block.
pub const XS_STRIDE: [usize; N_SEG_ARMS] = [XS_PAD, XS_DIM, XS_DIM, XS_PAD, XS_DIM, XS_DIM, XS_DIM, XS_DIM];

/// `nsplit = factor × tiles`; 0 for the arms that do not split K.
pub const SK_FACTOR: [u32; N_SEG_ARMS] = [0, 0, 0, 0, 0, 1, 2, 0];

/// Whether the arm's output must equal `tv_planes_seg`'s BIT FOR BIT. True
/// wherever the per-row lane order is the reference's; the split-K arms
/// re-associate wherever `nsplit > 1` and are held to the f64 reference
/// there instead (`sk_site_bit_exact` says which sites).
pub const BIT_EXACT: [bool; N_SEG_ARMS] = [true, true, true, true, true, false, false, true];

/// Parse `LLVQ_SEG_ARMS`. `None` (unset) is the empty selection: the four
/// historic arms and nothing else.
pub fn parse_seg_arms(spec: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(spec) = spec else {
        return Ok(Vec::new());
    };
    let mut seen = [false; N_SEG_ARMS];
    let mut named_any = false;
    for raw in spec.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            return Err(format!("LLVQ_SEG_ARMS: empty name in \"{}\"", spec.trim()));
        }
        named_any = true;
        let Some(arm) = SEG_ARM_NAMES.iter().position(|&n| n == name) else {
            return Err(format!(
                "LLVQ_SEG_ARMS: unknown arm \"{name}\". Valid: {}",
                SEG_ARM_NAMES.join(", ")
            ));
        };
        if seen[arm] {
            return Err(format!("LLVQ_SEG_ARMS: \"{name}\" named twice"));
        }
        seen[arm] = true;
    }
    if !named_any {
        return Err("LLVQ_SEG_ARMS: empty selection (leave the variable unset for the four historic arms alone)".to_string());
    }
    Ok((0..N_SEG_ARMS).filter(|&a| seen[a]).collect())
}

/// Where staged float `i` of a tile lands under stride `xs` — the mirror of
/// the kernel's `occ_xs_index`.
pub fn xs_index(i: usize, xs: usize) -> usize {
    (i / XS_DIM) * xs + (i % XS_DIM)
}

/// The K slice `[klo, khi)` of split `s` among `nsplit`, in blocks — the
/// mirror of the kernel's `occ_slice`.
pub fn slice(nblocks: u32, nsplit: u32, s: u32) -> (u32, u32) {
    let per = nblocks.div_ceil(nsplit);
    let lo = (s * per).min(nblocks);
    let hi = (lo + per).min(nblocks);
    (lo, hi)
}

/// Refuse the A3 section at any tile but the served one.
///
/// The A3 arms read [`crate::TILE_BLOCKS`] twice over, and neither reading can
/// see a resolved [`crate::tile::Tile`]:
///
///  * [`shared_bytes`] clamps at the const, and it is the **live** dynamic
///    shared argument of every A3 launch (`planesbench` :3310, :3318, :3328).
///    Compile the kernel at 32 and it would still be handed 128 tiles' worth
///    of staging, or the reverse — silently, since a launch never reports the
///    tile it was compiled for;
///  * [`tiles`], and through it [`sk_nsplit`] and [`sk_site_bit_exact`], set
///    the split-K split count from the same const.
///
/// And resizing them would not be enough. `kernels/planes_occ.cu` branches on
/// `nblocks <= TILE_BLOCKS` at `occ_rows_tiles` (:259) and `occ_pers` (:416):
/// the 2560-wide sites carry 106 blocks, so at the served 128 they take the
/// single-stage branch the kernel's own comment calls *"the whole persistence
/// dividend"*, and at 64 or 32 they do not. That is not a tile-sized arm, it
/// is **another algorithm under the same arm name and the same table row** —
/// the one defect a benchmark must never ship.
///
/// The section is opt-in and empty by default ([`parse_seg_arms`] of `None`),
/// and F1d does not run it. So it is refused by name rather than rebuilt: a
/// refusal costs nothing anyone is using, and threading a tile through four
/// functions would still leave the branch.
pub fn refuse_off_served_tile(seg_arms: &[usize], tile_is_served: bool) -> Result<(), String> {
    if seg_arms.is_empty() || tile_is_served {
        return Ok(());
    }
    Err(format!(
        "LLVQ_SEG_ARMS={} with a tile other than the served {}: refused. The A3 arms \
         size their launches from the constant, not from the resolved tile, and \
         `occ_pers` changes branch at `nblocks <= TILE_BLOCKS` — below the tile the \
         2560-wide sites lose the single-stage path, which is another algorithm under \
         the same name. Drop LLVQ_TILE_BLOCKS, or drop LLVQ_SEG_ARMS.",
        seg_arms
            .iter()
            .map(|&a| SEG_ARM_NAMES[a])
            .collect::<Vec<_>>()
            .join(","),
        crate::TILE_BLOCKS
    ))
}

/// Whether the single-stage branch of `occ_rows_tiles`/`occ_pers` applies —
/// `planes_occ.cu` :259 and :416, the persistence dividend.
///
/// Exposed so the refusal above can be justified by arithmetic rather than by
/// a comment: at the served 128 a 2560-wide site (106 blocks) takes it and at
/// 64 it does not.
pub fn pers_single_stage(nblocks: u32, tile: usize) -> bool {
    nblocks <= tile as u32
}

/// Tiles of `TILE_BLOCKS` a row of `nblocks` blocks stages.
pub fn tiles(nblocks: u32) -> u32 {
    nblocks.div_ceil(TILE_BLOCKS as u32)
}

/// The split count of a split-K arm on a site: `factor × tiles`, so `sk1`
/// splits exactly at the tile boundaries the reference kernel already has,
/// and `sk2` halves them.
pub fn sk_nsplit(nblocks: u32, factor: u32) -> u32 {
    factor * tiles(nblocks)
}

/// Whether a split-K arm is bit-identical to the reference on a site: only
/// where it does not split at all.
pub fn sk_site_bit_exact(nblocks: u32, factor: u32) -> bool {
    sk_nsplit(nblocks, factor) == 1
}

/// Dynamic shared bytes a launch stages: one tile, or one slice when the
/// slice is narrower than a tile, at `xs` floats per block.
pub fn shared_bytes(nblocks: u32, nsplit: u32, xs: usize) -> u32 {
    let per = nblocks.div_ceil(nsplit).min(TILE_BLOCKS as u32);
    per * xs as u32 * 4
}

/// The grid of a row-per-warp kernel with `r` rows per warp: exact, whole
/// CTAs only, refused otherwise (the kernels carry no bounds guard).
pub fn mr_grid(d_out: u32, threads: u32, r: u32) -> Result<u32, String> {
    let rows_per_cta = (threads / 32) * r;
    if !d_out.is_multiple_of(rows_per_cta) {
        return Err(format!(
            "d_out {d_out} is not a multiple of {rows_per_cta} rows per CTA ({r} rows per warp)"
        ));
    }
    Ok(d_out / rows_per_cta)
}

/// CTAs of `threads` threads, `regs` registers a thread and `shared` bytes of
/// dynamic shared memory that one SM holds at once — the smallest of the
/// three limits, every one of them read off the card and the loaded function.
///
/// A model, not a measurement: the hardware allocates registers by warp in
/// granules and shared memory in 128-byte units, both of which round UP the
/// real footprint. The number is therefore an upper bound on residency, and
/// it is printed next to the grid it sizes so a reader can see it.
pub fn residency(
    regs: u32,
    threads: u32,
    shared: u32,
    max_threads_sm: u32,
    regs_sm: u32,
    shared_sm: u32,
) -> u32 {
    let by_threads = max_threads_sm / threads;
    let by_regs = regs_sm / (regs.max(1) * threads);
    let by_shared = shared_sm.checked_div(shared).unwrap_or(u32::MAX);
    by_threads.min(by_regs).min(by_shared)
}

/// The persistent grid: one CTA per resident slot, and never more CTAs than
/// there are groups to walk.
pub fn pers_grid(ngroups: u32, resident_ctas: u32) -> u32 {
    ngroups.min(resident_ctas).max(1)
}

/// Waves of `ctas` over `slots` resident CTAs, and the fill of the last one
/// in percent — what the design note calls `remplissage de grille`.
pub fn waves(ctas: u32, slots: u32) -> (f64, f64) {
    let w = ctas as f64 / slots as f64;
    let full = ctas / slots;
    let rest = ctas - full * slots;
    let last = if rest == 0 { 100.0 } else { 100.0 * rest as f64 / slots as f64 };
    (w, last)
}

/// One site descriptor of the persistent-global arm, laid out as the kernel
/// reads it (`LLVQ_OCC_SITE_WORDS`): six device pointers, then two packed
/// pairs of `u32`.
#[allow(clippy::too_many_arguments)]
pub fn site_words(
    words: u64,
    gscale: u64,
    gs_off: u64,
    rscale: u64,
    tail: u64,
    y: u64,
    nblocks: u32,
    tail_w: u32,
    group0: u32,
    ngroups: u32,
) -> [u64; SITE_WORDS] {
    [
        words,
        gscale,
        gs_off,
        rscale,
        tail,
        y,
        u64::from(nblocks) | (u64::from(tail_w) << 32),
        u64::from(group0) | (u64::from(ngroups) << 32),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four fused shapes of Qwen3-4B — `(d_out, d_in)` after the
    /// q+k+v and gate+up concatenation of D1.
    const FUSED: [(u32, u32); 4] = [(6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728)];

    fn nblocks(d_in: u32) -> u32 {
        d_in / XS_DIM as u32
    }

    #[test]
    fn the_block_width_is_the_lattice_dimension() {
        assert_eq!(XS_DIM, llvq_core::DIM);
    }

    #[test]
    fn the_registry_is_consistent() {
        assert_eq!(SEG_ARM_NAMES.len(), N_SEG_ARMS);
        assert_eq!(SEG_KERNEL.len(), N_SEG_ARMS);
        assert_eq!(SEG_DISPLAY.len(), N_SEG_ARMS);
        let mut sorted = SEG_ARM_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), N_SEG_ARMS, "an arm name is registered twice");
        for a in 0..N_SEG_ARMS {
            assert!(SEG_KERNEL[a].starts_with("tv_planes_"), "{}", SEG_ARM_NAMES[a]);
            assert!(!SEG_DISPLAY[a].is_empty());
            // A split-K arm has a factor and is never bit-exact; the others
            // have no factor and are.
            assert_eq!(SK_FACTOR[a] > 0, !BIT_EXACT[a], "{}", SEG_ARM_NAMES[a]);
            assert!(ROWS_PER_WARP[a] == 1 || SK_FACTOR[a] == 0, "{}", SEG_ARM_NAMES[a]);
            assert!(XS_STRIDE[a] == XS_DIM || XS_STRIDE[a] == XS_PAD);
        }
        assert_eq!(SEG_KERNEL[SK1], SEG_KERNEL[SK2], "sk1 and sk2 share the kernel");
        assert_eq!(SEG_ARM_NAMES[PERSALL], "persall");
    }

    /// `LLVQ_XS_PAD 28u` in the kernel text, verbatim — the constant is
    /// duplicated across the language boundary, so it is pinned here.
    #[test]
    fn the_padded_stride_matches_the_kernel() {
        let cu = include_str!("../kernels/planes_occ.cu");
        assert!(cu.contains(&format!("#define LLVQ_XS_PAD {XS_PAD}u")), "LLVQ_XS_PAD ≠ {XS_PAD}");
        assert!(cu.contains(&format!("#define LLVQ_OCC_SITE_WORDS {SITE_WORDS}u")));
        for k in SEG_KERNEL {
            assert!(cu.contains(&format!("__global__ void {k}(")), "{k} absent from planes_occ.cu");
        }
    }

    #[test]
    fn unset_is_the_historic_four_only() {
        assert_eq!(parse_seg_arms(None).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn names_come_back_in_registration_order() {
        let got = parse_seg_arms(Some("sk2, pad ,persall,mr2")).unwrap();
        assert_eq!(got, vec![PAD, MR2, SK2, PERSALL]);
    }

    #[test]
    fn unknown_duplicate_and_empty_are_refused() {
        let e = parse_seg_arms(Some("pad,mr3")).unwrap_err();
        assert!(e.contains("mr3") && e.contains("persall"), "{e}");
        let e = parse_seg_arms(Some("pad,pad")).unwrap_err();
        assert!(e.contains("twice"), "{e}");
        assert!(parse_seg_arms(Some("pad,,mr2")).is_err());
        assert!(parse_seg_arms(Some("")).is_err());
        assert!(parse_seg_arms(Some("  ")).is_err());
    }

    #[test]
    fn every_arm_parses_alone_and_all_together() {
        for (a, name) in SEG_ARM_NAMES.iter().enumerate() {
            assert_eq!(parse_seg_arms(Some(name)).unwrap(), vec![a]);
        }
        let all = SEG_ARM_NAMES.join(",");
        assert_eq!(parse_seg_arms(Some(&all)).unwrap(), (0..N_SEG_ARMS).collect::<Vec<_>>());
    }

    /// The padded index leaves 4 dead floats after every 24: block b's slot
    /// s lands at 28b + s, and no two staged floats collide.
    #[test]
    fn the_padded_index_is_a_block_stride_of_28() {
        let n = TILE_BLOCKS * XS_DIM;
        let mut seen = vec![false; TILE_BLOCKS * XS_PAD];
        for i in 0..n {
            assert_eq!(xs_index(i, XS_DIM), i);
            let p = xs_index(i, XS_PAD);
            assert_eq!(p, (i / 24) * 28 + i % 24);
            assert!(!seen[p], "collision at {i}");
            seen[p] = true;
        }
        assert_eq!(xs_index(24, XS_PAD), 28);
        assert_eq!(xs_index(23, XS_PAD), 23);
    }

    /// Slices partition `[0, nblocks)` in order, each at most `per` wide, and
    /// any empty slice sits at the end — for every width the 4B has and every
    /// split up to 2× its tiles, plus a few adversarial widths.
    #[test]
    fn slices_partition_the_row() {
        let mut widths: Vec<u32> = FUSED.iter().map(|&(_, d_in)| nblocks(d_in)).collect();
        widths.extend([1, 2, 7, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1000]);
        for &nb in &widths {
            for factor in 1..=2 {
                let ns = sk_nsplit(nb, factor);
                assert!(ns >= 1);
                let per = nb.div_ceil(ns);
                let mut at = 0;
                let mut empty_seen = false;
                for s in 0..ns {
                    let (lo, hi) = slice(nb, ns, s);
                    assert_eq!(lo, at, "nb {nb} ns {ns} s {s}");
                    assert!(hi >= lo && hi - lo <= per);
                    if hi == lo {
                        empty_seen = true;
                    } else {
                        assert!(!empty_seen, "a non-empty slice after an empty one: nb {nb} ns {ns}");
                    }
                    at = hi;
                }
                assert_eq!(at, nb, "slices do not cover the row: nb {nb} ns {ns}");
            }
            // And a split wider than the row: every block still lands once.
            let ns = nb + 3;
            let mut at = 0;
            for s in 0..ns {
                let (lo, hi) = slice(nb, ns, s);
                assert_eq!(lo, at);
                at = hi;
            }
            assert_eq!(at, nb);
        }
    }

    /// The 4B's four fused widths: 106 blocks is one tile, 170 two, 405 four.
    #[test]
    fn the_split_counts_of_the_four_shapes() {
        assert_eq!(sk_nsplit(nblocks(2560), 1), 1);
        assert_eq!(sk_nsplit(nblocks(4096), 1), 2);
        assert_eq!(sk_nsplit(nblocks(9728), 1), 4);
        assert_eq!(sk_nsplit(nblocks(2560), 2), 2);
        assert_eq!(sk_nsplit(nblocks(4096), 2), 4);
        assert_eq!(sk_nsplit(nblocks(9728), 2), 8);
        assert!(sk_site_bit_exact(nblocks(2560), 1));
        assert!(!sk_site_bit_exact(nblocks(4096), 1));
        assert!(!sk_site_bit_exact(nblocks(2560), 2));
        // The design note's figures: o/down go from 320 CTAs to 640/1280
        // under sk1 (*computed*).
        assert_eq!(320 * sk_nsplit(nblocks(4096), 1), 640);
        assert_eq!(320 * sk_nsplit(nblocks(9728), 1), 1280);
    }

    /// Shared bytes follow the narrower of a tile and a slice: 12,288 bytes for
    /// the reference geometry, 14,336 padded, and less when a slice is short.
    #[test]
    fn shared_bytes_follow_the_slice() {
        assert_eq!(shared_bytes(106, 1, XS_DIM), 106 * 24 * 4);
        assert_eq!(shared_bytes(405, 1, XS_DIM), 128 * 24 * 4);
        assert_eq!(shared_bytes(405, 1, XS_PAD), 128 * 28 * 4);
        assert_eq!(shared_bytes(405, 8, XS_DIM), 51 * 24 * 4);
        assert_eq!(shared_bytes(106, 2, XS_DIM), 53 * 24 * 4);
        assert!(shared_bytes(128, 1, XS_PAD) <= 49_152, "the padded tile must fit the default allowance");
    }

    /// The four fused shapes launch whole CTAs at 1, 2 and 4 rows per warp,
    /// and a ragged shape is refused rather than launched short.
    #[test]
    fn the_multi_row_grids_are_exact_on_the_fused_shapes() {
        for &(d_out, _) in &FUSED {
            for r in [1, 2, 4] {
                let g = mr_grid(d_out, 256, r).unwrap();
                assert_eq!(g * 8 * r, d_out);
            }
        }
        assert_eq!(mr_grid(6144, 256, 1).unwrap(), 768);
        assert_eq!(mr_grid(2560, 256, 2).unwrap(), 160);
        assert_eq!(mr_grid(19456, 256, 4).unwrap(), 608);
        assert!(mr_grid(1028, 256, 1).is_err());
        assert_eq!(mr_grid(1032, 256, 1).unwrap(), 129, "1032 = 8 × 129: whole CTAs");
        assert!(mr_grid(2568, 256, 2).is_err());
    }

    /// The A3 section is refused off the served tile, and the arithmetic that
    /// justifies the refusal is checked rather than asserted in prose.
    #[test]
    fn the_a3_arms_are_refused_off_the_served_tile() {
        // Nothing selected: nothing to refuse, whatever the tile.
        assert!(refuse_off_served_tile(&[], true).is_ok());
        assert!(refuse_off_served_tile(&[], false).is_ok());
        // The served tile: the section runs as it always has.
        assert!(refuse_off_served_tile(&[PERS], true).is_ok());
        // Any other tile, and it is refused by name — both variables, so the
        // operator knows which of the two to drop.
        let e = refuse_off_served_tile(&[PERS, SK1], false).expect_err("must refuse");
        assert!(e.contains("LLVQ_SEG_ARMS"), "{e}");
        assert!(e.contains("LLVQ_TILE_BLOCKS"), "{e}");
        assert!(e.contains("pers") && e.contains("sk1"), "{e}");

        // Why it is a refusal and not a resize: a 2560-wide site carries
        // 2560/24 = 106 blocks, which is under the served 128 and over both
        // measured rows. The arm changes branch, not size.
        let nb = 2560 / XS_DIM as u32;
        assert_eq!(nb, 106);
        assert!(pers_single_stage(nb, crate::TILE_BLOCKS));
        assert!(!pers_single_stage(nb, 64));
        assert!(!pers_single_stage(nb, 32));
        // A 4096-wide site is over every tile: that one never had the
        // dividend, so it is not what the refusal protects.
        assert!(!pers_single_stage(4096 / XS_DIM as u32, crate::TILE_BLOCKS));
        // The branch is `nblocks <= TILE_BLOCKS` in `planes_occ.cu` (:259,
        // :416), inclusive. A site of exactly one tile takes the single-stage
        // path; a strict `<` here would put it on the other branch and the
        // refusal would be justified by an arithmetic the kernel does not run.
        assert!(pers_single_stage(crate::TILE_BLOCKS as u32, crate::TILE_BLOCKS));
        assert!(!pers_single_stage(crate::TILE_BLOCKS as u32 + 1, crate::TILE_BLOCKS));
    }

    /// The padded arms do not stage at 24 floats, which is why
    /// `tile::Tile::shared_bytes` cannot speak for them.
    #[test]
    fn two_a3_arms_stage_at_the_padded_stride() {
        assert_eq!(XS_STRIDE[PAD], XS_PAD);
        assert_eq!(XS_STRIDE[MR2P], XS_PAD);
        assert_ne!(XS_PAD, XS_DIM);
        // At the tile module's ceiling the padded stride asks for more than
        // the default per-block allowance every card here reports (49,152 B),
        // which is the second reason those arms are refused rather than
        // resized. Written as a subtraction so the margin is the number, and
        // so a ceiling that grows cannot make the check vacuous.
        let over = crate::tile::TILE_MAX * XS_PAD * 4;
        assert_eq!(over, 57_344);
        assert_eq!(over - 49_152, 8_192, "the padded stride overruns by 8 KiB at TILE_MAX");
    }

    /// The L40S numbers of the design note: 40 registers, 256 threads,
    /// 12,288 bytes → 6 CTAs an SM, 852 on 142 SM — the `852` every comment
    /// quotes. Padded staging (14,336 bytes) still allows 7 by shared memory, so
    /// threads keep the limit at 6.
    #[test]
    fn residency_reproduces_the_l40s_figure() {
        let (thr, regs_sm, sh_sm) = (1536, 65_536, 102_400);
        assert_eq!(residency(40, 256, 12_288, thr, regs_sm, sh_sm), 6);
        assert_eq!(residency(40, 256, 14_336, thr, regs_sm, sh_sm), 6);
        assert_eq!(6 * 142, 852);
        // Registers become the limit past 42 a thread.
        assert_eq!(residency(48, 256, 12_288, thr, regs_sm, sh_sm), 5);
        assert_eq!(residency(64, 256, 12_288, thr, regs_sm, sh_sm), 4);
        // Shared memory becomes the limit for a whole-row staging.
        assert_eq!(residency(40, 256, 38_912, thr, regs_sm, sh_sm), 2);
        assert_eq!(residency(40, 256, 0, thr, regs_sm, sh_sm), 6);
        assert_eq!(pers_grid(320, 852), 320);
        assert_eq!(pers_grid(2432, 852), 852);
        assert_eq!(pers_grid(0, 852), 1);
    }

    /// The design note's fill figures (*computed*): qkv 90.1 % of 852, o and
    /// down 37.6 %, gate_up 2.85 waves.
    #[test]
    fn waves_reproduce_the_design_note() {
        let (w, last) = waves(768, 852);
        assert!((w - 0.901).abs() < 1e-3 && (last - 90.1).abs() < 0.1);
        let (w, last) = waves(320, 852);
        assert!((w - 0.3756).abs() < 1e-3 && (last - 37.6).abs() < 0.1);
        let (w, last) = waves(2432, 852);
        assert!((w - 2.854).abs() < 1e-3 && (last - 85.4).abs() < 0.1);
        assert_eq!(waves(852, 852), (1.0, 100.0));
        assert_eq!(waves(1704, 852), (2.0, 100.0));
    }

    /// The descriptor packs as the kernel unpacks it: pointers first, then
    /// `nblocks | tail_w << 32` and `group0 | ngroups << 32`.
    #[test]
    fn site_words_pack_as_the_kernel_unpacks() {
        let w = site_words(1, 2, 3, 4, 5, 6, 405, 8, 1234, 320);
        assert_eq!(&w[..6], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(w[6] & 0xffff_ffff, 405);
        assert_eq!(w[6] >> 32, 8);
        assert_eq!(w[7] & 0xffff_ffff, 1234);
        assert_eq!(w[7] >> 32, 320);
        assert_eq!(w.len(), SITE_WORDS);
    }
}
