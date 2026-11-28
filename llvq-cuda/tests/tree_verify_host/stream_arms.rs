//! Jobs N1 and N2 and arms a4 to a7: the tiled weights, the stream unit, the
//! batched fold, the prefetch and the int4 down's input over the grid. One file
//! because each arm is held to the one before it, row for row.

use super::*;

// ---- job N1: the tree-tiled weights (arm a2) ---------------------------------------

/// Passes of `sizes` rows (ancestor-closed subsets in a random slot order),
/// run by the untiled symbols on `untiled` and by the tiled symbols (variant
/// 32) on `bin`, at the same slices: every row's argmax, logits, residual, K
/// and V equal, bit for bit.
fn gate_tiled(bin: &Path, untiled: &Path, f: &Fix, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x5d00 + i as u64), pos0, 0x7d00 + i as u64)).collect();
    let u = exec(untiled, &Fix { variant: f.variant & !32, ..*f }, None, &passes)?;
    let v = exec(bin, &Fix { variant: f.variant | 32, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&v.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("tiled against untiled, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// The tiled pass equals the untiled pass bit for bit at R = 1, 2, 8, 16, 32,
/// under S0's slices and under a second plan; and on the 8B's parts (a
/// Tetra o, an untied head) at RoPE form 2, R = 1 and 5.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tiled_rows_equal_untiled_rows_under_both_plans() {
    let t = tree(32, 7, 0x3e5);
    for split in [SPLIT, SPLIT_N1] {
        let n = gate_tiled(host(), host(), &Fix { split, ..FIX }, &t, &[1, 2, 8, 16, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows of 5 passes, tiled equals untiled bit for bit");
    }
    let t = tree(12, 4, 0x3e6);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N1, ..FIX };
    let n = gate_tiled(host(), host(), &f, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, tiled equals untiled bit for bit");
}

/// Each guard of the tiling has a mutant, killed by the tiled gate against
/// the unmutated untiled pass.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tiled_mutants_are_killed() {
    let t = tree(12, 4, 0x3e7);
    let f = Fix { split: SPLIT_N1, ..FIX };
    gate_tiled(host(), host(), &f, &t, &[1, 9], 9).expect("the unmutated tiled pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et.cu";
    let muts: &[(&str, &[Edit])] = &[
        // the copy's row: each lane's row taken from its neighbour's
        ("tile_row", &[(k, "const u32 row = 16u * T + r;", "const u32 row = 16u * T + (r ^ 1u);")]),
        // the reader's Tetra row
        ("tile_read_row", &[(k, "wsw + P4T_TSW * ks + 3u * r :", "wsw + P4T_TSW * ks + 3u * (r ^ 1u) :")]),
        // the int4 halves swapped
        ("tile_q4_halves", &[(k, "wsw[P4T_TSQ * ks + 6u * r + 3u * h + k]", "wsw[P4T_TSQ * ks + 6u * r + 3u * (h ^ 1u) + k]")]),
        // a tile's stride one sub-step short
        ("tile_stride", &[(k, "* tsub + (ok ? sub0 + ks : 0u)) * SW", "* (tsub - 1u) + (ok ? sub0 + ks : 0u)) * SW")]),
        // one int4 word offset
        ("tile_q4_word", &[(k, "wsw[P4T_TSQ * ks + 6u * r + 3u * h + k]", "wsw[P4T_TSQ * ks + 6u * r + 3u * h + (k + 1u) % 3u]")]),
    ];
    let mut killed = 0;
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gate_tiled(&b, host(), &f, &t, &[1, 9], 9);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => {
                killed += 1;
                eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or(""))
            }
            Ok(_) => panic!("tiled mutant {tag} survived its gate"),
        }
    }
    eprintln!("tiled mutants: {killed} of {} killed", muts.len());
}

// ---- job N2: the stream unit (arm a3) ----------------------------------------------

/// The host twin's stream plan: `pers::p4t_splits_stream` at 3 CTAs of 2
/// warps, whose area (attention's share) holds slices of 18 sub-steps; the
/// head in 4.
pub(super) const SPLIT_N2: [u32; 5] = [4, 6, 4, 13, 4];
/// A second plan within the cap, every phase cut otherwise, the head in 5.
pub(super) const SPLIT_N2B: [u32; 5] = [5, 8, 4, 19, 5];

/// Passes of `sizes` rows run by the tiled symbols (variant 32, arm a2) on
/// `tiled` and by the stream symbols (variant 128, arm a3) on `bin`, at the
/// same slices: every row's argmax, logits, residual, K and V equal.
fn gate_stream(bin: &Path, tiled: &Path, f: &Fix, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x5e00 + i as u64), pos0, 0x7c00 + i as u64)).collect();
    let u = exec(tiled, &Fix { variant: (f.variant & !128) | 32, ..*f }, None, &passes)?;
    let v = exec(bin, &Fix { variant: (f.variant & !32) | 128, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&v.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("stream against tiled, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// The stream pass equals the tiled pass bit for bit at R = 1, 2, 8, 16, 32
/// under the twin's stream plan and a second plan, the head folded from 4
/// and 5 slices; and on the 8B's parts (a Tetra o, an untied head) at RoPE
/// form 2, R = 1 and 5.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn stream_rows_equal_tiled_rows_under_both_plans() {
    let cap = pers::p4t_stream_cap(pers::p4t_area_floats(2, 32, 9728, 4), 2);
    assert_eq!(pers::p4t_splits_stream(2560, 9728, 4096, 8, VOCAB, 2, 3, cap).unwrap(), SPLIT_N2);
    assert!(SPLIT_N2B.iter().zip([2560u32, 4096, 2560, 9728, 2560]).all(|(&s, d)| pers::p4t_slice_steps(d, s) <= cap));
    let t = tree(32, 7, 0x3e8);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_stream(host(), host(), &Fix { split, ..FIX }, &t, &[1, 2, 8, 16, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows of 5 passes, stream equals tiled bit for bit");
    }
    let t = tree(12, 4, 0x3e9);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_stream(host(), host(), &f, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, stream equals tiled bit for bit");
}

/// A plan whose q/k/v slice holds 27 sub-steps, past the twin's 18: the
/// stream pass refuses it through the abort flag.
#[test]
fn a_stream_slice_past_the_cap_is_refused() {
    let t = tree(4, 2, 0x3ea);
    let f = Fix { split: [2, 6, 3, 12, 3], variant: 128, ..FIX };
    let e = exec(host(), &f, None, &[Pass::of(&t, &[0], 9, 0x7c80)]).err().expect("a slice past the cap runs");
    assert!(e.contains("aborted"), "{e}");
}

/// Each guard of the stream unit has a mutant, killed by the stream gate
/// against the unmutated tiled pass.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn stream_mutants_are_killed() {
    let t = tree(12, 4, 0x3eb);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_stream(host(), host(), &f, &t, &[1, 9], 9).expect("the unmutated stream pass passes its gate");
    let s = "llvq-cuda/kernels/pers_4et_stream.cu";
    let muts: &[(&str, &[Edit])] = &[
        // the slice's columns staged one sub-step late
        ("slice_column", &[(s, "col = sub_lo * P4T_SUB + 8u * p;", "col = (sub_lo + 1u) * P4T_SUB + 8u * p;")]),
        // no restaging: the stage still holds the previous unit's slice
        ("stale_stage", &[(s, "    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);\n", "    (void)RS;\n")]),
        // a B fragment read from the next activation row
        ("b_row", &[(s, "const u32* xr = xs + (nt * 8u + gid) * XS + kw;", "const u32* xr = xs + ((nt * 8u + gid + 1u) % RS) * XS + kw;")]),
        // the tile stream advanced by half a sub-step
        ("stream_stride", &[(s, "            wp += SW;\n", "            wp += SW / 2u;\n")]),
        // the read-ahead takes the current sub-step's words for the next's
        ("read_ahead", &[(s, "p4t_stream_load(g.kind, wp + SW, n0, n1, n2);", "p4t_stream_load(g.kind, wp, n0, n1, n2);")]),
        // the head's partials and fold skipped
        ("head_fold", &[(s, "    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has && (g.epi != P4T_HEAD || g.S == 1u)) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n")]),
    ];
    let mut killed = 0;
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gate_stream(&b, host(), &f, &t, &[1, 9], 9);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => {
                killed += 1;
                eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or(""))
            }
            Ok(_) => panic!("stream mutant {tag} survived its gate"),
        }
    }
    eprintln!("stream mutants: {killed} of {} killed", muts.len());
}

// ---- arm a4: the stream unit, the fold batched (F1), the prologues on vectors (F2) ----

/// Passes of `sizes` rows run by the stream symbols (variant 128, arm a3)
/// on `stream` and by the a4 symbols (variant 256) on `bin`, at the same
/// slices: every row's argmax, logits, residual, K and V equal.
fn gate_a4(bin: &Path, stream: &Path, f: &Fix, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x5f00 + i as u64), pos0, 0x7b00 + i as u64)).collect();
    let u = exec(stream, &Fix { variant: (f.variant & !(32 | 256)) | 128, ..*f }, None, &passes)?;
    let v = exec(bin, &Fix { variant: (f.variant & !(32 | 128)) | 256, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&v.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("a4 against stream, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// a4 equals a3 bit for bit at R = 1, 2, 8, 16, 32 under both stream plans,
/// and on the 8B's parts at RoPE form 2, R = 1 and 5.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a4_rows_equal_stream_rows_under_both_plans() {
    let t = tree(32, 7, 0x3ec);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_a4(host(), host(), &Fix { split, ..FIX }, &t, &[1, 2, 8, 16, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows of 5 passes, a4 equals stream bit for bit");
    }
    let t = tree(12, 4, 0x3ed);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_a4(host(), host(), &f, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, a4 equals stream bit for bit");
}

/// Each guard of F1 and F2 has a mutant, killed by the a4 gate against the
/// unmutated stream pass. A mutant of the vector path survives if the path
/// never runs (a row or weight not 16-byte aligned), so their kills also
/// show the twin takes it.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a4_mutants_are_killed() {
    let t = tree(12, 4, 0x3ee);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_a4(host(), host(), &f, &t, &[1, 9], 9).expect("the unmutated a4 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a4.cu";
    let muts: &[(&str, &[Edit])] = &[
        // F1: a batch's adds in the reverse slice order
        ("fold_order", &[(k, "                    const u32 j = jj;\n", "                    const u32 j = 7u - jj;\n")]),
        // F1: the next batch starts one slice late, skipping slice 8 (the down in 19)
        ("fold_skip", &[(k, "for (u32 k0 = 0; k0 < g.S; k0 += 8u) {", "for (u32 k0 = 0; k0 < g.S; k0 += 9u) {")]),
        // F2: a loaded row vector widened one element off
        ("vec_load", &[(k, "for (u32 e = 0; e < 8u; ++e) S[8u * q + e] = xf[e];", "for (u32 e = 0; e < 8u; ++e) S[8u * q + e] = xf[(e + 1u) & 7u];")]),
        // F2: a stored vector one element off
        ("vec_store", &[(k, "for (u32 e = 0; e < 8u; ++e) ho[e] = f2h(S[8u * q + e]);", "for (u32 e = 0; e < 8u; ++e) ho[e] = f2h(S[8u * q + ((e + 1u) & 7u)]);")]),
        // F2: the down's input pairs the gate with the next up vector
        ("dn_pair", &[(k, "ub[u] = __ldcg(uv + q);", "ub[u] = __ldcg(uv + (q ^ 1u));")]),
    ];
    let mut killed = 0;
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gate_a4(&b, host(), &f, &t, &[1, 9], 9);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => {
                killed += 1;
                eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or(""))
            }
            Ok(_) => panic!("a4 mutant {tag} survived its gate"),
        }
    }
    eprintln!("a4 mutants: {killed} of {} killed", muts.len());
}

/// F3 is stamps only: a4 stamped (variant 64, its parts written after the
/// phase events) leaves every row's bits as the unstamped a4 pass.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a4_stamps_change_no_bit() {
    let t = tree(12, 4, 0x3ef);
    let f = Fix { split: SPLIT_N2B, variant: 256, ..FIX };
    let passes: Vec<Pass> = [1usize, 9].iter().enumerate().map(|(i, &s)| Pass::of(&t, &t.subset(s, 0x5f80 + i as u64), 9, 0x7b80 + i as u64)).collect();
    let u = exec(host(), &f, None, &passes).unwrap();
    let v = exec(host(), &Fix { variant: 256 | 64, ..f }, None, &passes).unwrap();
    for (pi, (a, b)) in u.passes.iter().zip(&v.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            assert!(diff(rb, ra).is_none(), "pass {pi}, slot {s}: the stamped a4 pass moved a bit");
        }
    }
}

// ---- arm a5: a4 with the L2 prefetch of the next row phase's tile streams ----

/// a5 (variant 512) against a4 (variant 256) on `a4b`, at the same slices:
/// every row equal. The prefetch writes nothing; on the twin it reads each
/// range's first and last words, so a range outside the copies faults.
fn gate_a5(bin: &Path, a4b: &Path, f: &Fix, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x6000 + i as u64), pos0, 0x7a80 + i as u64)).collect();
    let u = exec(a4b, &Fix { variant: (f.variant & !(32 | 128 | 512)) | 256, ..*f }, None, &passes)?;
    let v = exec(bin, &Fix { variant: (f.variant & !(32 | 128 | 256)) | 512, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&v.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("a5 against a4, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// a5 equals a4 bit for bit at R = 1, 8, 32 under both stream plans, and on
/// the 8B's parts (a Tetra o, whose stream a5 also prefetches) at form 2.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a5_rows_equal_a4_rows() {
    let t = tree(32, 7, 0x3f0);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_a5(host(), host(), &Fix { split, ..FIX }, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows of 3 passes, a5 equals a4 bit for bit");
    }
    let t = tree(12, 4, 0x3f1);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_a5(host(), host(), &f, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, a5 equals a4 bit for bit");
}

/// The prefetch is compiled in the twin and its ranges are read there: a
/// tile offset scaled far past the copies faults.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a5_prefetch_mutant_is_killed() {
    let t = tree(12, 4, 0x3f2);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_a5(host(), host(), &f, &t, &[1, 9], 9).expect("the unmutated a5 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a5.cu";
    let b = mutant(
        "pf_range",
        &[(k, "p4t_pf_lines(g.w + ((unsigned long long)tile * nsub + sub_lo) * SW,", "p4t_pf_lines(g.w + ((unsigned long long)tile * nsub * 1048576ull + sub_lo) * SW,")],
    );
    let r = gate_a5(&b, host(), &f, &t, &[1, 9], 9);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("pf_range: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a5 mutant pf_range survived its gate"),
    }
}

// ---- arms a6, a7 (a3 with the int4 down's input over the grid, a7 also prefetching) ----

/// Arm `v` (variant bit) on `bin` against arm `r` on `refb`, at the same
/// slices: every row's argmax, logits, residual, K and V equal.
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_pair(bin: &Path, refb: &Path, f: &Fix, v: u32, r: u32, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let arms = 32 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 | 524288 | 1048576 | 2097152 | 4194304 | A12 | A12IP;
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x6100 + i as u64), pos0, 0x7a00 + i as u64)).collect();
    let u = exec(refb, &Fix { variant: (f.variant & !arms) | r, ..*f }, None, &passes)?;
    let w = exec(bin, &Fix { variant: (f.variant & !arms) | v, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&w.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("variant {v} against {r}, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// a6 equals a3 and a7 equals a6, bit for bit, at R = 1, 8, 32 under both
/// stream plans (the twin's layer 1 has an int4 down, layer 0 a Tetra one).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a6_and_a7_rows_equal_their_guards() {
    let t = tree(32, 7, 0x3f3);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let f = Fix { split, ..FIX };
        let n = gate_pair(host(), host(), &f, 1024, 128, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a6 equals a3 bit for bit");
        let n = gate_pair(host(), host(), &f, 2048, 1024, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a7 equals a6 bit for bit");
    }
}

/// The grid deal of the down's input: a deal that skips the last piece is
/// killed by a6's gate against a3.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a6_mutant_is_killed() {
    let t = tree(12, 4, 0x3f4);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_pair(host(), host(), &f, 1024, 128, &t, &[1, 9], 9).expect("the unmutated a6 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a6.cu";
    let b = mutant("dn_skip_last", &[(k, "p < total; p += T) {", "p + 1u < total; p += T) {")]);
    let r = gate_pair(&b, host(), &f, 1024, 128, &t, &[1, 9], 9);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("dn_skip_last: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a6 mutant dn_skip_last survived its gate"),
    }
}

/// The floor arm a3f runs to its end on the twin (no abort, no ticket left,
/// every row an argmax); its values are void by design and not compared.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_floor_arm_runs() {
    let t = tree(12, 4, 0x3f5);
    let f = Fix { split: SPLIT_N2B, variant: 4096, ..FIX };
    let passes = [Pass::of(&t, &t.subset(9, 0x6180), 9, 0x7a80)];
    let o = exec(host(), &f, None, &passes).unwrap();
    assert!(o.passes[0].iter().all(|r| r.tok < VOCAB), "a floor row wrote no argmax");
}
