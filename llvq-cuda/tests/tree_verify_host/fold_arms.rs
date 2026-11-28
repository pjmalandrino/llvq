//! Arms a8 to a12ip: the lean fold, the fold after the wait, 4E's prologue path,
//! the row caps and warp counts, the drafter's taps, the words read in place.
//! One file because each is held to a10 or its parent, bit for bit.

use super::*;

// ---- arms a8 (a3 with the lean fold), a3p and a3t (floor arms) ----

/// a8 equals a3 bit for bit at R = 1, 8, 32 under both stream plans.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a8_rows_equal_stream_rows() {
    let t = tree(32, 7, 0x3f6);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_pair(host(), host(), &Fix { split, ..FIX }, 32768, 128, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a8 equals a3 bit for bit");
    }
}

/// A quarter read one element off is killed by a8's gate against a3.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a8_mutant_is_killed() {
    let t = tree(12, 4, 0x3f7);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_pair(host(), host(), &f, 32768, 128, &t, &[1, 9], 9).expect("the unmutated a8 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a8.cu";
    let b = mutant("quarter_off", &[(k, "p4t_epi(g, L, A, n, wrow + 4u * q + e, vq[e], cmax);", "p4t_epi(g, L, A, n, wrow + 4u * q + e, vq[(e + 1u) & 3u], cmax);")]);
    let r = gate_pair(&b, host(), &f, 32768, 128, &t, &[1, 9], 9);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("quarter_off: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a8 mutant quarter_off survived its gate"),
    }
}

/// The floor arms a3p and a3t run to their end on the twin (no abort, no
/// ticket left, every row an argmax); their values are void and not compared.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_protocol_floor_arms_run() {
    let t = tree(12, 4, 0x3f8);
    for v in [8192u32, 16384] {
        let f = Fix { split: SPLIT_N2B, variant: v, ..FIX };
        let o = exec(host(), &f, None, &[Pass::of(&t, &t.subset(9, 0x6190), 9, 0x7a90)]).unwrap();
        assert!(o.passes[0].iter().all(|r| r.tok < VOCAB), "variant {v}: a floor row wrote no argmax");
    }
}

// ---- arm a9: a3 with the fold after the phase's grid wait ----

/// a9 equals a3 bit for bit at R = 1, 2, 8, 16, 32 under both stream plans
/// (the head folded from 4 and 5 slices), and on the 8B's parts at form 2.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a9_rows_equal_stream_rows() {
    let t = tree(32, 7, 0x3f9);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_pair(host(), host(), &Fix { split, ..FIX }, 65536, 128, &t, &[1, 2, 8, 16, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a9 equals a3 bit for bit");
    }
    let t = tree(12, 4, 0x3fa);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_pair(host(), host(), &f, 65536, 128, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, a9 equals a3 bit for bit");
}

/// Each guard of the fold step has a mutant, killed by a9's gate against a3.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a9_mutants_are_killed() {
    let t = tree(12, 4, 0x3fb);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_pair(host(), host(), &f, 65536, 128, &t, &[1, 9], 9).expect("the unmutated a9 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a9.cu";
    let muts: &[(&str, &[Edit])] = &[
        // the slices added from the last down
        ("fold_order", &[(k, "                const u32 ks = k;\n", "                const u32 ks = g.S - k;\n")]),
        // slice 1 skipped
        ("fold_skip", &[(k, "for (u32 k = 1u; k < g.S; ++k) {\n                const u32 ks = k;", "for (u32 k = 2u; k < g.S; ++k) {\n                const u32 ks = k;")]),
        // the last pair of each projection dealt to nobody
        ("pair_nobody", &[(k, "p < np; p += T) {", "p + 1u < np; p += T) {")]),
        // a pair's residual read from the neighbour row
        ("resid_row", &[(k, "g.res + (unsigned long long)n * g.ldy + row);", "g.res + (unsigned long long)n * g.ldy + (row ^ 1u));")]),
        // the q/k/v fold step's wait removed
        ("wait_removed", &[(k, "            P4T_WAIT();\n            p4t_fold_a9(sq, sk, sv", "            p4t_fold_a9(sq, sk, sv")]),
    ];
    let mut killed = 0;
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gate_pair(&b, host(), &f, 65536, 128, &t, &[1, 9], 9);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => {
                killed += 1;
                eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or(""))
            }
            Ok(_) => panic!("a9 mutant {tag} survived its gate"),
        }
    }
    eprintln!("a9 mutants: {killed} of {} killed", muts.len());
}

// ---- arm a10: a9 with 4E's prologue path ----

/// a10 equals a9 bit for bit at R = 1, 2, 8, 16, 32 under both stream plans,
/// and on the 8B's parts at form 2.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a10_rows_equal_a9_rows() {
    let t = tree(32, 7, 0x3fc);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_pair(host(), host(), &Fix { split, ..FIX }, 131072, 65536, &t, &[1, 2, 8, 16, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a10 equals a9 bit for bit");
    }
    let t = tree(12, 4, 0x3fd);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_pair(host(), host(), &f, 131072, 65536, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, a10 equals a9 bit for bit");
}

/// Each guard of the staged prologues and of the silu in the fold step has a
/// mutant, killed by a10's gate against a9. The twin's layers share their
/// rotations, so "a stale small" is ln2 staged with ln1's rows.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a10_mutants_are_killed() {
    let t = tree(12, 4, 0x3fe);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_pair(host(), host(), &f, 131072, 65536, &t, &[1, 9], 9).expect("the unmutated a10 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a10.cu";
    let muts: &[(&str, &[Edit])] = &[
        ("stale_small", &[(k, "pers_v7_stage_rot(v7a10, M.signs_h, H, M.small_h, M.k_h)", "pers_v7_stage_rot(v7a10, M.signs_h, H, Y.small_a, M.k_h)")]),
        ("sign_word", &[(k, "pers_v7_stage_rot(v7a10, M.signs_h, H, M.small_h, M.k_h)", "pers_v7_stage_rot(v7a10, M.signs_h + 1, H, M.small_h, M.k_h)")]),
        ("silu_pairing", &[(k, "su.part + (unsigned long long)n * su.d_out + row0));", "su.part + (unsigned long long)n * su.d_out + (row0 ^ 4u)));")]),
        ("rows_short", &[(k, "M.small_i, M.k_i));", "M.small_i, M.k_i - 1u));")]),
    ];
    let mut killed = 0;
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gate_pair(&b, host(), &f, 131072, 65536, &t, &[1, 9], 9);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => {
                killed += 1;
                eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or(""))
            }
            Ok(_) => panic!("a10 mutant {tag} survived its gate"),
        }
    }
    eprintln!("a10 mutants: {killed} of {} killed", muts.len());
}

// ---- arm a11: a10 with the L2 prefetch in its wait gaps ----

/// a11 equals a10 bit for bit at R = 1, 8, 32 under both stream plans, and
/// on the 8B's parts (its Tetra o's stream prefetched too).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a11_rows_equal_a10_rows() {
    let t = tree(32, 7, 0x401);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let n = gate_pair(host(), host(), &Fix { split, ..FIX }, 524288, 131072, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a11 equals a10 bit for bit");
    }
    let t = tree(12, 4, 0x402);
    let f = Fix { form: 2, variant: 16, cap: 288, split: SPLIT_N2B, ..FIX };
    let n = gate_pair(host(), host(), &f, 524288, 131072, &t, &[1, 5], 254).unwrap();
    eprintln!("8B's parts, form 2: {n} rows, a11 equals a10 bit for bit");
}

/// The prefetch is compiled in the twin and its ranges read there: a tile
/// offset scaled far past the copies faults.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a11_prefetch_mutant_is_killed() {
    let t = tree(12, 4, 0x403);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_pair(host(), host(), &f, 524288, 131072, &t, &[1, 9], 9).expect("the unmutated a11 pass passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a11.cu";
    let b = mutant("pf11_range", &[(k, "const u32* p = g.w + ((unsigned long long)tile * nsub + sub_lo) * SW;", "const u32* p = g.w + ((unsigned long long)tile * nsub * 1048576ull + sub_lo) * SW;")]);
    let r = gate_pair(&b, host(), &f, 524288, 131072, &t, &[1, 9], 9);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("pf11_range: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a11 mutant pf11_range survived its gate"),
    }
}

// ---- arm a11i: a11 with fine stamps ----

/// a11i equals a11 bit for bit at R = 1, 8, 32 under both stream plans,
/// unstamped and stamped (variant 64: its slots written after the events).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a11i_rows_equal_a11_rows_stamped_or_not() {
    let t = tree(32, 7, 0x3ff);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let f = Fix { split, ..FIX };
        let n = gate_pair(host(), host(), &f, 1048576, 524288, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a11i equals a11 bit for bit");
        let n = gate_pair(host(), host(), &Fix { variant: 64, ..f }, 1048576 | 64, 524288, &t, &[1, 8, 32], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a11i stamped equals a11 bit for bit");
    }
}

// ---- arms a10r8 and a10r16: a10 for at most 8 and 16 rows a launch ----

/// a10r8 at R = 1 and 8, a10r16 at R = 1, 8 and 16: bit for bit with a10 at
/// the same rows, under both stream plans.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn row_cap_rows_equal_a10_rows() {
    let t = tree(16, 5, 0x404);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let f = Fix { split, ..FIX };
        let n = gate_pair(host(), host(), &f, 2097152, 131072, &t, &[1, 8], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a10r8 equals a10 bit for bit");
        let n = gate_pair(host(), host(), &f, 4194304, 131072, &t, &[1, 8, 16], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a10r16 equals a10 bit for bit");
    }
}

/// Above its cap a row-cap arm refuses the launch (the abort flag).
#[test]
fn a_row_cap_arm_refuses_rows_past_its_cap() {
    let t = tree(17, 5, 0x405);
    for (v, r) in [(2097152u32, 9usize), (4194304, 17)] {
        let f = Fix { split: SPLIT_N2B, variant: v, ..FIX };
        let e = exec(host(), &f, None, &[Pass::of(&t, &t.subset(r, 0x6200), 9, 0x7a40)]).err().expect("a launch past the cap runs");
        assert!(e.contains("aborted"), "variant {v}, {r} rows: {e}");
    }
}

// ---- arms a10t384 and a10t256: a10 at 12 and 8 warps a CTA ----

/// a10 at `threads` (variant `v`) against a10 at 64 threads (variant 131072),
/// the same slices: every row equal.
fn gate_threads(f: &Fix, threads: u32, v: u32, t: &Tree, sizes: &[usize], pos0: u32) -> Result<usize, String> {
    let passes: Vec<Pass> =
        sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x6300 + i as u64), pos0, 0x7900 + i as u64)).collect();
    let u = exec(host(), &Fix { variant: 131072, ..*f }, None, &passes)?;
    let w = exec(host(), &Fix { variant: v, threads, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&w.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("{threads} threads against 64, split {:?}, pass {pi} ({} rows), slot {s}: {what}", f.split, a.len()));
            }
        }
    }
    Ok(rows)
}

/// a10t384 and a10t256 equal a10 bit for bit at R = 1, 8, 32 under two
/// plans whose slices fit the stream cap at 12, 8 and 2 warps.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn thread_count_rows_equal_a10_rows() {
    let t = tree(32, 7, 0x406);
    let plans = [[5u32, 8, 5, 19, 5], [6, 9, 6, 17, 6]];
    for (threads, v) in [(384u32, 8388608u32), (256, 16777216)] {
        for nw in [threads / 32, 2] {
            let cap = pers::p4t_stream_cap(pers::p4t_area_floats(nw, 32, 9728, 4), nw);
            for split in plans {
                assert!(split.iter().zip([2560u32, 4096, 2560, 9728, 2560]).all(|(&s, d)| pers::p4t_slice_steps(d, s) <= cap), "{split:?} at {nw} warps");
            }
        }
        for split in plans {
            let n = gate_threads(&Fix { split, ..FIX }, threads, v, &t, &[1, 8, 32], 9).unwrap();
            eprintln!("{threads} threads, split {split:?}: {n} rows, equal to a10 at 64 threads bit for bit");
        }
    }
}

// ---- arms a10t448, a10t320, a10t384r16 ----

/// a10t448 and a10t320 equal a10 bit for bit at R = 1, 8, 32, a10t384r16 at
/// R = 1, 8, 16, under two plans whose slices fit the cap at their warps.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn second_thread_count_rows_equal_a10_rows() {
    let t = tree(32, 7, 0x407);
    let plans = [[5u32, 8, 5, 19, 5], [6, 9, 6, 17, 6]];
    for (threads, v, sizes) in [(448u32, 33554432u32, &[1usize, 8, 32][..]), (320, 67108864, &[1, 8, 32][..]), (384, 134217728, &[1, 8, 16][..])] {
        let nw = threads / 32;
        let cap = pers::p4t_stream_cap(pers::p4t_area_floats(nw, 32, 9728, 4), nw);
        for split in plans {
            assert!(split.iter().zip([2560u32, 4096, 2560, 9728, 2560]).all(|(&s, d)| pers::p4t_slice_steps(d, s) <= cap), "{split:?} at {nw} warps");
            let n = gate_threads(&Fix { split, ..FIX }, threads, v, &t, sizes, 9).unwrap();
            eprintln!("variant {v} at {threads} threads, split {split:?}: {n} rows, equal to a10 at 64 threads bit for bit");
        }
    }
}

/// The 384-thread 16-row arm refuses 17 rows.
#[test]
fn the_t384r16_arm_refuses_17_rows() {
    let t = tree(17, 5, 0x408);
    let f = Fix { split: SPLIT_N2B, variant: 134217728, threads: 384, ..FIX };
    let e = exec(host(), &f, None, &[Pass::of(&t, &t.subset(17, 0x6400), 9, 0x7800)]).err().expect("17 rows run");
    assert!(e.contains("aborted"), "{e}");
}

// ---- arm a12: a10t384r16 with the drafter's taps ----

/// a12 equals a10 bit for bit at R = 1, 8, 16 under both stream plans; its
/// taps (layers 0, 1 and n_layers = 2 on the twin, first 0 and stride 1)
/// hold, for every row, the final residual in the slot of layer 2 bit for
/// bit, and the slots past the layers stay zero.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12_rows_equal_a10_rows_and_its_taps_are_the_residual() {
    let t = tree(16, 5, 0x409);
    for split in [SPLIT_N2, SPLIT_N2B] {
        let f = Fix { split, ..FIX };
        let n = gate_pair(host(), host(), &f, A12, 131072, &t, &[1, 8, 16], 9).unwrap();
        eprintln!("split {split:?}: {n} rows, a12 equals a10 bit for bit");
        let n = gate_taps(host(), &f, &t, &[1, 8, 16]).unwrap();
        eprintln!("split {split:?}: {n} rows, the layer-2 tap is the final residual, slots 3 and 4 zero");
    }
}

/// The tap checks of [`a12_rows_equal_a10_rows_and_its_taps_are_the_residual`].
fn gate_taps(bin: &Path, f: &Fix, t: &Tree, sizes: &[usize]) -> Result<usize, String> {
    let passes: Vec<Pass> = sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x6500 + i as u64), 9, 0x7700 + i as u64)).collect();
    let o = exec(bin, &Fix { variant: (f.variant & !131072) | A12, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, pass) in o.passes.iter().enumerate() {
        for (s, row) in pass.iter().enumerate() {
            rows += 1;
            if row.taps.len() != 5 * H {
                return Err(format!("pass {pi}, slot {s}: {} tap values", row.taps.len()));
            }
            if row.taps[2 * H..3 * H] != row.x[..] {
                return Err(format!("pass {pi}, slot {s}: the layer-2 tap is not the final residual"));
            }
            if row.taps[3 * H..].iter().any(|&v| v != 0) || row.taps[..2 * H].iter().all(|&v| v == 0) {
                return Err(format!("pass {pi}, slot {s}: a slot past the layers written, or the layer 0 and 1 slots empty"));
            }
        }
    }
    Ok(rows)
}

/// A tap taken one layer late is killed by the tap gate.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12_tap_mutant_is_killed() {
    let t = tree(12, 4, 0x40a);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_taps(host(), &f, &t, &[1, 9]).expect("the unmutated taps pass their gate");
    let k = "llvq-cuda/kernels/pers_4et_a12.cu";
    let b = mutant("tap_layer_off", &[(k, "    const u32 j = p4t_tap_slot_a12(L.pad, l);", "    const u32 j = p4t_tap_slot_a12(L.pad, l + 1u);")]);
    let r = gate_taps(&b, &f, &t, &[1, 9]);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("tap_layer_off: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a12 mutant tap_layer_off survived its gate"),
    }
}

// ---- arm a12ip: a12 on the served words, read in place (memory axis, M1) ----

/// a12ip against a12 on the same passes: every row's argmax, logits,
/// residual, K, V and taps bit for bit. a12 reads the tree-tiled copies,
/// a12ip the words 4E reads; the twin runs the install's load check
/// (p4t_ip_check on every matrix against its copy) before the passes.
fn gate_ip(bin: &Path, f: &Fix, t: &Tree, sizes: &[usize]) -> Result<usize, String> {
    let passes: Vec<Pass> = sizes.iter().enumerate().map(|(i, &s)| Pass::of(t, &t.subset(s, 0x6600 + i as u64), 9, 0x7600 + i as u64)).collect();
    let u = exec(host(), &Fix { variant: f.variant | A12, ..*f }, None, &passes)?;
    let w = exec(bin, &Fix { variant: f.variant | A12IP, ..*f }, None, &passes)?;
    let mut rows = 0;
    for (pi, (a, b)) in u.passes.iter().zip(&w.passes).enumerate() {
        for (s, (ra, rb)) in a.iter().zip(b).enumerate() {
            rows += 1;
            if let Some(what) = diff(rb, ra) {
                return Err(format!("a12ip against a12, split {:?}, variant {}, pass {pi} ({} rows), slot {s}: {what}", f.split, f.variant, a.len()));
            }
            if rb.taps != ra.taps {
                return Err(format!("a12ip against a12, split {:?}, variant {}, pass {pi}, slot {s}: taps", f.split, f.variant));
            }
        }
    }
    Ok(rows)
}

/// a12ip equals a12 bit for bit at R = 1, 8, 16 under both stream plans,
/// on the 4B's shapes (layer 0 a Tetra down, layer 1 an int4 one, the tied
/// head) and with the 8B's parts (a Tetra o, an untied head), and its load
/// check passes on every matrix.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ip_rows_equal_a12_rows_on_the_served_words() {
    let t = tree(16, 5, 0x40b);
    for split in [SPLIT_N2, SPLIT_N2B] {
        for ext in [0u32, 16] {
            let f = Fix { split, variant: ext, ..FIX };
            let n = gate_ip(host(), &f, &t, &[1, 8, 16]).unwrap();
            eprintln!("split {split:?}, variant {ext}: {n} rows, a12ip equals a12 bit for bit, taps included");
        }
    }
}

/// A read in place that drops 4E v9's int4 order (the permutation inside
/// each row) is killed: by the load check, or by the rows.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a12ip_permutation_mutant_is_killed() {
    let t = tree(12, 4, 0x40c);
    let f = Fix { split: SPLIT_N2B, ..FIX };
    gate_ip(host(), &f, &t, &[1, 9]).expect("the unmutated a12ip passes its gate");
    let k = "llvq-cuda/kernels/pers_4et_a10rc.cu";
    let b = mutant(
        "ip_no_v9_order",
        &[(k, "    return rowp[(g.kind != 0u && g.tiled != 0u) ? pers_q4_tile_index(wi, sw) : wi];", "    return rowp[(g.kind != 0u && g.tiled == 2u) ? pers_q4_tile_index(wi, sw) : wi];")],
    );
    let r = gate_ip(&b, &f, &t, &[1, 9]);
    let _ = std::fs::remove_file(&b);
    match r {
        Err(e) => eprintln!("ip_no_v9_order: killed by {}", e.lines().next().unwrap_or("")),
        Ok(_) => panic!("a12ip mutant ip_no_v9_order survived its gate"),
    }
}
