//! Unit tests of the fused host side, one file because they reach the private
//! helpers of every sibling through `super`.

use llvq_artifact::CodeKind;
use llvq_quant::rotation::Rotation;

use super::host::{runtime_bytes_of, ROT_KMAX};
use super::kernels::{EMB_Q4_CU_EMBED, Q4_H_CU_EMBED};
use super::*;

/// `LLVQ_FUSE` refuses anything that is not a mode — the [`FusedLayout::
/// parse`] contract, restated beside its siblings.
#[test]
fn fuse_mode_parse_refuses_anything_else() {
    assert_eq!(FuseMode::parse(None), Ok(FuseMode::Off));
    assert_eq!(FuseMode::parse(Some("")), Ok(FuseMode::Off));
    assert_eq!(FuseMode::parse(Some("0")), Ok(FuseMode::Off));
    assert_eq!(FuseMode::parse(Some("1")), Ok(FuseMode::On));
    for bad in ["on", "off", "true", "2", "1 ", "01", "yes", "planes14"] {
        let e = FuseMode::parse(Some(bad)).expect_err("must be refused");
        assert!(e.contains(bad), "the message must cite the value: {e}");
    }
}

/// The int4 kernel's staging, at the three sealed objects' widest int4
/// activation, against the L40S's bounds (49,152 by default, 101,376 on
/// request, `fusedrun-14b-2026-08-17.txt`): the 4B and the 8B fit the
/// default, the 14B needs the opt-in, and nothing needs more.
#[test]
// `llvq-cuda` is a test dependency on macOS only (llvq-llm/Cargo.toml).
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn the_int4_staging_fits_the_card_at_every_sealed_size() {
    use super::kernels::int4_shared_bytes;
    use llvq_cuda::shared::{plan, Fit};
    let (def, optin) = (49_152, 101_376);
    // (hidden, o_proj d_in, intermediate): Qwen3-4B, 8B, 14B.
    let sizes = [(2560, 4096, 9728), (4096, 4096, 12288), (5120, 5120, 17408)];
    let want = [(38_912, Fit::Default), (49_152, Fit::Default), (69_632, Fit::OptIn)];
    for ((h, o, i), (bytes, fit)) in sizes.into_iter().zip(want) {
        let got = int4_shared_bytes([h, o, i]);
        assert_eq!(got, bytes, "the widest of v, o and down, in f32");
        assert_eq!(plan(got, def, optin), Some(fit), "{bytes} B");
    }
    assert_eq!(int4_shared_bytes([]), 0, "no int4 record, no request");
}

/// The served 4B, 8B and 14B pick exactly today's kernels and today's
/// sizes, on every card this repository has served from, after the 32B
/// wall lift (`llvq_cuda::wide`).
///
/// Per model: every rotation width (hidden, `o_proj` input, intermediate)
/// stays on `rot_apply`, and the opt-in posed on it, the widest SHARED
/// rotation, is the widest rotation, as before. The int4 entry points get
/// one F32 size equal to `int4_shared_bytes` over all records, the old
/// size, and nothing is staged in f16, so the wide unit is never
/// compiled. The 32B moves on 99 KiB cards only, and only its 25,600.
#[test]
// `llvq-cuda` is a test dependency on macOS only (llvq-llm/Cargo.toml).
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn the_served_models_keep_todays_kernels_after_the_wall_lift() {
    use super::kernels::int4_shared_bytes;
    use llvq_cuda::shared::rot_bytes;
    use llvq_cuda::wide::{int4_entry_bytes, rot_kernel, Int4EntryBytes, RotKernel};
    // (default, opt-in): L40S and RTX 5090 / RTX PRO 6000 (99 KiB, the
    // L40S figure measured), A100 measured, H200 from the CUDA guide.
    let cards = [(49_152, 101_376), (49_152, 166_912), (49_152, 232_448)];
    // (hidden, o_proj d_in, intermediate) of Qwen3-4B, 8B, 14B; the int4
    // sets of the e4 configs are v, o and down, so the same three widths.
    let served = [(2_560, 4_096, 9_728), (4_096, 4_096, 12_288), (5_120, 5_120, 17_408)];
    for (def, optin) in cards {
        for (h, o, i) in served {
            let widths = [h, o, i];
            for n in widths {
                assert!(
                    matches!(rot_kernel(n, def, optin), RotKernel::Shared(_)),
                    "{n} left rot_apply on ({def}, {optin})"
                );
            }
            let shared_widest = widths
                .iter()
                .filter(|&&n| rot_kernel(n, def, optin) != RotKernel::Device)
                .map(|&n| rot_bytes(n))
                .max();
            assert_eq!(shared_widest, widths.iter().map(|&n| rot_bytes(n)).max());
            let e = int4_entry_bytes(widths, def, optin).unwrap();
            assert_eq!(
                e,
                Int4EntryBytes { f32: Some(int4_shared_bytes(widths)), f16: None },
                "{h}/{o}/{i} on ({def}, {optin})"
            );
        }
    }
    // The 32B: 25,600 moves on the 99 KiB card, and nothing else does.
    let (def, optin) = cards[0];
    assert_eq!(rot_kernel(25_600, def, optin), RotKernel::Device);
    assert!(matches!(rot_kernel(5_120, def, optin), RotKernel::Shared(_)));
    assert!(matches!(rot_kernel(8_192, def, optin), RotKernel::Shared(_)));
    let e = int4_entry_bytes([5_120, 8_192, 25_600], def, optin).unwrap();
    assert_eq!(e, Int4EntryBytes { f32: Some(32_768), f16: Some(51_200) });
    // On the larger cards the 32B needs no lift at all.
    for &(d, o) in &cards[1..] {
        assert!(matches!(rot_kernel(25_600, d, o), RotKernel::Shared(_)));
        assert_eq!(int4_entry_bytes([5_120, 8_192, 25_600], d, o).unwrap().f16, None);
    }
}

/// The synthetic int4 record is shaped as the uploader expects: `d_in`
/// a multiple of 8, `d_out` of 8, and the three arrays at their sizes.
#[test]
fn the_synthetic_int4_record_has_the_record_shape() {
    for (d_out, d_in, seed, _) in INT4_WIDE_CARD_CASES {
        let (r, x) = int4_synthetic(d_out, d_in, seed);
        assert_eq!(r.group, 128);
        assert_eq!(r.packed.len(), d_out * d_in / 2);
        assert_eq!(r.scales.len(), d_out * d_in.div_ceil(128));
        assert_eq!(r.biases.len(), r.scales.len());
        assert_eq!(x.len(), d_in);
        assert!(d_in.is_multiple_of(8) && d_out.is_multiple_of(8));
    }
    let m = parse_int4_wide_fixture(INT4_WIDE_FIXTURE).expect("the compiled-in fixture parses");
    assert_eq!(m.len(), INT4_WIDE_CARD_CASES.len());
    let h = "a".repeat(64);
    assert!(parse_int4_wide_fixture("256 25600 0x42 0\n").is_err());
    assert!(parse_int4_wide_fixture(&format!("256 25600 0x42 0 {h}\n256 25600 0x42 0 {h}\n")).is_err());
    assert!(parse_int4_wide_fixture("256 25600 0x42 0 xyz\n").is_err());
}

/// The int4 bytes are counted, and they are counted where the weights are.
///
/// 🕳️ The regression this pins is not hypothetical: `runtime_bytes`
/// summed the lattice matrices and the fused groups and stopped, while
/// `quantized_weights` counted an int4 record's weights — "they are
/// quantized, just not by us". A numerator missing a term whose weights
/// are in the denominator understates every b/weight the runtime prints,
/// and nothing fails.
#[test]
fn the_int4_bytes_reach_the_footprint_the_runtime_prints() {
    let q = FusedInt4 {
        name: "model.layers.0.self_attn.v_proj.weight".into(),
        d_out: 1024,
        d_in: 2560,
        group: 128,
        packed: Vec::new(),
        scales: Vec::new(),
        biases: Vec::new(),
        // What one served `v_proj` actually costs: 1024·2560/2 packed,
        // plus 1024·20 groups of one binary16 scale and one binary16 bias.
        bytes: 1024 * 2560 / 2 + 1024 * 20 * 2 * 2,
    };
    assert_eq!(q.bytes, 1_392_640);
    // The term that was missing, alone, so its absence cannot hide behind
    // the other two.
    assert_eq!(runtime_bytes_of(&[], &[], std::slice::from_ref(&q)), q.bytes);
    assert_eq!(runtime_bytes_of(&[], &[], &[]), 0);
    // Over the served file's 36 of them, against the hand arithmetic: the
    // whole term is 50.1 MB, which is what the printed line was light by.
    let all: Vec<FusedInt4> = (0..36)
        .map(|l| FusedInt4 { name: format!("model.layers.{l}.self_attn.v_proj.weight"), ..q.clone() })
        .collect();
    assert_eq!(runtime_bytes_of(&[], &[], &all), 50_135_040);
}

/// The name the host hands the driver is the name the embedded source
/// defines, and the group it hard-codes is the format's.
///
/// `fused_cuda/` looks `INT4_KERNEL_NAME` up on a card and nowhere else,
/// so a typo in it is a `named symbol not found` at the end of a load —
/// which is exactly how `tv_planes_seg_h` was found, on a rented card.
#[test]
fn the_int4_kernel_names_and_constants_match_its_source() {
    let src = Q4_H_CU_EMBED;
    assert!(
        src.contains(&format!("__global__ void {INT4_KERNEL_NAME}(")),
        "`{INT4_KERNEL_NAME}` is not defined by the source the host appends"
    );
    // The group is the format's constant on both sides of the boundary.
    assert!(
        src.contains(&format!("#define LLVQ_Q4_GROUP {}u", llvq_artifact::INT4G128_GROUP)),
        "the kernel's group and `INT4G128_GROUP` disagree"
    );
    // It needs `matvec.cu` (h2f, f2h, warp_sum) and nothing else, and that
    // file is the second part of EVERY unit — so appending this source
    // last is always legal, whatever the layout.
    assert!(src.contains("#ifndef TILE_COLS"), "the composition guard moved");
}

/// The q4 embedding pair the host looks up is the pair `emb_q4.cu`
/// defines, and neither name collides with a kernel of another source that
/// shares its translation unit: the served mixed file appends both
/// `tv_q4_h.cu` and `emb_q4.cu`.
#[test]
fn the_q4_embedding_kernel_names_match_its_source() {
    let (gather, head) = emb_kernel_names(EmbedMode::Q4).expect("q4 launches two kernels");
    for name in [gather, head] {
        assert!(
            EMB_Q4_CU_EMBED.contains(&format!("__global__ void {name}(")),
            "`{name}` is not defined by emb_q4.cu"
        );
        assert!(
            !Q4_H_CU_EMBED.contains(&format!(" {name}(")),
            "`{name}` is also a name in tv_q4_h.cu"
        );
    }
    assert_ne!(head, INT4_KERNEL_NAME, "the embedding head and the int4 projection share a name");
    // Its dequant helper must not collide with tv_q4_h.cu's either.
    assert!(!Q4_H_CU_EMBED.contains("e4_deq"));
    assert!(!EMB_Q4_CU_EMBED.contains("q4_deq("));
    assert!(EMB_Q4_CU_EMBED.contains("#ifndef TILE_COLS"), "the composition guard moved");
    assert_eq!(emb_kernel_names(EmbedMode::Q8), Some(("emb_q8_gather", "tv_q8_h")));
    assert_eq!(emb_kernel_names(EmbedMode::F16), None);
}

/// Every kernel a runtime looks up has its SOURCE in that runtime's unit.
///
/// 🕳️ The one invariant `fused_cuda/` cannot check for itself: that module
/// does not compile on this machine — it needs nvcc and a Linux cross
/// toolchain — so a name it asks the driver for is only ever tested on a
/// rented card, at the end of a load. On 2026-09-10 it asked for
/// `tv_planes_seg_h` on the `Tetra48` build, whose list shares nothing
/// with the ball layouts, and the run died after compiling, loading and
/// reporting 216 projections. Two of the three names it looks up are
/// pinned here instead, where the check is free.
///
/// `Slot32` is the exception and stays one: its kernel lives in the base
/// list every layout gets (`llvq_slot.cuh`, `matvec.cu`), not in the
/// per-layout list, which is precisely what keeps its translation unit
/// bit-identical to what shipped before any layout switch existed.
#[test]
fn every_layout_only_names_kernels_its_own_unit_carries() {
    for layout in [
        FusedLayout::Slot32,
        FusedLayout::Planes14,
        FusedLayout::Planes12x,
        FusedLayout::Golay70,
        FusedLayout::Tetra48,
    ] {
        let srcs = planes_source_names(layout);
        // The segmented kernel, ONE WAY ONLY — and the asymmetry is the
        // point. `Planes12x` and `Golay70` carry `tv_planes_seg_h.cu` and
        // cannot launch it: it is compiled so the register report stays a
        // drift detector on those builds, which the list's own header
        // argues at length. So *compiled* is a superset of *launchable*,
        // and only the implication holds — a layout that can launch it
        // must have it.
        //
        // Which is why `fused_cuda`'s register report keys on the SOURCE
        // LIST and not on either the layout or `seg_kernel_name`: on
        // `Tetra48` all three answers differ, and it picked the wrong one.
        if seg_kernel_name(layout).is_some() {
            assert!(
                srcs.contains(&"tv_planes_seg_h.cu"),
                "{}: launches `tv_planes_seg_h` and its unit does not carry it: {srcs:?}",
                layout.name()
            );
        }
        // Tetra48 has neither, and that is the case that crashed: it is
        // not Slot32, so "not Slot32" reported a kernel it never had.
        if layout == FusedLayout::Tetra48 {
            assert!(seg_kernel_name(layout).is_none());
            assert!(!srcs.contains(&"tv_planes_seg_h.cu"));
        }
        // The prefill entry point, same rule: named only where the unit
        // carries the source that defines it. `tests/served_unit.rs`
        // checks the other half — that the source really defines it.
        if let Some(n) = rows_kernel_name(layout) {
            assert_eq!(n, "tv_tetra48_rows_h");
            assert!(
                srcs.contains(&"tv_tetra48_h.cu"),
                "{}: names `{n}` and carries no source that defines it",
                layout.name()
            );
        }
        // The matvec kernel: its own `.cu` is in the list, except Slot32's
        // which is in the base list every unit already carries.
        let m = matvec_kernel_name(layout);
        if layout != FusedLayout::Slot32 {
            assert!(
                srcs.contains(&format!("{m}.cu").as_str()),
                "{}: looks up `{m}` and carries no `{m}.cu`: {srcs:?}",
                layout.name()
            );
        } else {
            assert!(srcs.is_empty(), "Slot32's unit is the base list and nothing else");
        }
    }
}

/// A Tetra header has no runtime layout **among the ball ones**: each of
/// the four refuses it with the artifact crate's words, and the Ball arm
/// is `new` — same tables, same layout.
///
/// 🕳️ This test swept the four ball layouts and stopped there, so
/// `Tetra48` never passed through `for_kind` on this machine at all. The
/// layout that the whole of step 6 exists to serve was the one layout the
/// portable test did not build — and it died on a card, at $0.02, on the
/// first run of the served path, before reading a single block. A sweep
/// that omits the subject is not a sweep. `Tetra48` is a POSITIVE case
/// below, which is what makes the omission impossible to repeat.
#[test]
fn a_tetra_file_has_no_runtime_layout_among_the_ball_ones() {
    for layout in [
        FusedLayout::Planes14,
        FusedLayout::Planes12x,
        FusedLayout::Slot32,
        FusedLayout::Golay70,
    ] {
        let e = Transcoder::for_kind(layout, CodeKind::Tetra)
            .err()
            .unwrap_or_else(|| panic!("{}: a Tetra header must be refused", layout.name()));
        assert!(
            e.contains("no runtime layout for Tetra: this reads the v1 ball's classes, and a Tetra word names none"),
            "{}: the refusal must say why: {e}",
            layout.name()
        );
        assert!(e.contains(layout.name()), "{}: the refusal must name the layout: {e}", layout.name());
        let ball = Transcoder::for_kind(layout, CodeKind::Ball).expect("a Ball header builds");
        assert_eq!(ball.layout(), layout);
        assert_eq!(
            ball.class_table().expect("a ball layout has a class table").n_entries(),
            Transcoder::new(layout)
                .expect("new")
                .class_table()
                .expect("a ball layout has a class table")
                .n_entries()
        );
    }

    // And the layout that serves Tetra BUILDS on a Tetra header, with no
    // class table at all — a Tetra word names no class, so asking for the
    // ball's 383 of them is what refused a file this layout serves.
    let t = Transcoder::for_kind(FusedLayout::Tetra48, CodeKind::Tetra)
        .expect("tetra48 must build on a Tetra header");
    assert_eq!(t.layout(), FusedLayout::Tetra48);
    assert!(t.class_table().is_none(), "tetra48 must hold no class table");

    // A BALL file under `tetra48` is still refused — not here, where the
    // layout needs no table, but at the record gate. What this asserts is
    // that dropping the table did not drop that: the two refusals are
    // different, and only one of them was ever this function's business.
    assert!(
        Transcoder::for_kind(FusedLayout::Tetra48, CodeKind::Ball).is_ok(),
        "the table is not where a Ball file under tetra48 is refused"
    );
    assert!(!serves_kind(FusedLayout::Tetra48, CodeKind::Ball));
    assert!(serves_kind(FusedLayout::Tetra48, CodeKind::Tetra));
}

/// The default is `Off`, and it *has* to be: [`check_fuse`] refuses `On`
/// beside `RotShare::Off`, whose own default is `Off`, so a default of `On`
/// here would make every `fused_cuda::load` fail with no environment set.
/// The two defaults are pinned together so neither can move alone.
#[test]
fn the_two_defaults_are_compatible() {
    let (fuse, share) = (
        FuseMode::parse(None).expect("default"),
        crate::rotplan::RotShare::parse(None).expect("default"),
    );
    assert_eq!(fuse, FuseMode::Off);
    assert_eq!(share, crate::rotplan::RotShare::Off);
    check_fuse(FusedLayout::Planes14, share, fuse).expect("the two defaults hold together");
    // And the pair this is guarding against, in both of its forms.
    let e = check_fuse(FusedLayout::Planes14, crate::rotplan::RotShare::Off, FuseMode::On)
        .expect_err("On + RotShare::Off must be refused");
    assert!(e.contains("LLVQ_ROT_SHARE"), "{e}");
    for layout in [FusedLayout::Planes12x, FusedLayout::Slot32, FusedLayout::Golay70] {
        let e = check_fuse(layout, crate::rotplan::RotShare::On, FuseMode::On)
            .expect_err("only planes14 segments");
        assert!(e.contains(layout.name()), "the refusal must name the layout: {e}");
        // …and `Off` stays launchable on every layout, since it is the
        // control arm.
        check_fuse(layout, crate::rotplan::RotShare::On, FuseMode::Off).expect("control");
    }
    check_fuse(FusedLayout::Planes14, crate::rotplan::RotShare::On, FuseMode::On)
        .expect("the only fusable pair");
}

/// The three tables must describe the transform `Rotation` computes.
///
/// Not a restatement of `RotationTables::build`: it rebuilds `Q x` from
/// the *tables alone*, the way a kernel would — sign bitmap, Walsh–
/// Hadamard per group, dense block across groups — and requires the
/// result to match. A packing bug (bit order, row-major vs column-major
/// `small`, a wrong `inv`) shows up here and nowhere else on this machine.
#[test]
fn the_tables_describe_the_rotation() {
    for (n, seed) in [(240usize, 1u64), (2560, 2), (9728, 3), (4096, 4), (96, 5)] {
        let t = RotationTables::build(n, seed).expect("within KMAX");
        let rot = Rotation::new(n, seed);
        let mut rng = llvq_core::SplitMix64::new(seed ^ 0xf00d);
        let x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

        let mut want = x.clone();
        rot.apply(&mut want);

        // From the tables only.
        let mut s: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if (t.signbits[i >> 5] >> (i & 31)) & 1 == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        let mut len = 1usize;
        while len < t.m {
            for p in 0..n / 2 {
                let j = (p / len) * (len << 1) + (p % len);
                let (a, b) = (s[j], s[j + len]);
                s[j] = a + b;
                s[j + len] = a - b;
            }
            len <<= 1;
        }
        let inv = t.inv as f64;
        let got: Vec<f64> = if t.k == 1 {
            s.iter().map(|v| v * inv).collect()
        } else {
            let mut out = vec![0.0f64; n];
            for j in 0..t.m {
                for g in 0..t.k {
                    let mut acc = 0.0;
                    for c in 0..t.k {
                        acc += t.small[g * ROT_KMAX + c] as f64 * s[c * t.m + j] * inv;
                    }
                    out[g * t.m + j] = acc;
                }
            }
            out
        };

        let nrm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
        let dev = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(dev / nrm < 1e-6, "n={n}: relative deviation {:.3e}", dev / nrm);
    }
}

/// `inv` is `1/√m`, not `1/√n`. On a width with an odd factor the two
/// differ by `√k` — a factor of 4.4 at `k = 19` — and every output would
/// be uniformly wrong, which is exactly the kind of error a norm check
/// catches and an eyeball does not.
#[test]
fn the_scale_is_over_the_power_of_two_factor() {
    let t = RotationTables::build(9728, 7).expect("within KMAX");
    assert_eq!((t.m, t.k), (512, 19));
    assert!((t.inv as f64 - 1.0 / 512f64.sqrt()).abs() < 1e-9);
    assert!((t.inv as f64 - 1.0 / 9728f64.sqrt()).abs() > 1e-3);
}

/// A header that announces four billion records with nothing behind it is
/// a short read and an `Err`, not a reservation the allocator aborts the
/// process on. The same for the count of carried tensors.
#[test]
fn a_huge_count_in_a_short_file_is_an_error_not_an_abort() {
    let path = std::env::temp_dir().join(format!("llvq-fused-count-{}.llvq", std::process::id()));
    let path_s = path.to_str().expect("a UTF-8 temp path").to_string();

    let mut bytes = Vec::new();
    llvq_artifact::write_header_kind(&mut bytes, 5, u32::MAX, CodeKind::Tetra).expect("a header");
    std::fs::write(&path, &bytes).expect("write the header");
    assert!(load(&path_s, FusedLayout::Tetra48).is_err(), "u32::MAX records, none written");

    let mut bytes = Vec::new();
    llvq_artifact::write_header_kind(&mut bytes, 5, 0, CodeKind::Tetra).expect("a header");
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write the header and the tensor count");
    assert!(load(&path_s, FusedLayout::Tetra48).is_err(), "u32::MAX carried tensors, none written");

    let _ = std::fs::remove_file(&path);
}
