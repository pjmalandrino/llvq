//! R6: the research row bodies as 4E row forms (`pers.rs::p4e_rc_edits`), held
//! near the served chain by a distance, not bit for bit. One file for that measure.

use super::*;

// ---- R6: the research row bodies as 4E row forms (pers.rs::p4e_rc_edits) ---------
//
// v9ep (endpass) owes the bits: it is in `the_4e_e1_arms_are_the_served_chain`
// above, on every command there. v9fl (flat) owes a bound of 1e-5 on rows
// (tests/rowcand_gate.rs, `the_rc_bodies_keep_what_they_owe`): its chain is
// compared here with the served chain and the distance printed, never assumed
// zero. The commands below run the 512-thread symbols (variants 1 and 2),
// whose rows are the rc bodies'; variant 0 runs v9's 768-thread rows, untouched.
// They take the smooth SiLU table (+ 256, host_pers_4e.inc): under the random
// table a one-ULP change of a gate value maps to an unrelated activation, and
// the chain's distance would measure the table, not the rows.

/// The token chain of the flat arm against the served chain, on one command:
/// (tokens equal, tokens, binary16 state values that differ, state values,
/// largest |difference| over the state's rms, both in f32).
fn flat_distance(got: &[u8], want: &[u8], n_tokens: usize) -> (usize, usize, usize, usize, f64) {
    let (gr, wr) = (u32s(&got[..4 * n_tokens]), u32s(&want[..4 * n_tokens]));
    let equal = gr.iter().zip(&wr).filter(|(a, b)| a == b).count();
    let end = want.len() - 8;
    let (gs, ws) = (u16s(&got[4 * n_tokens..end]), u16s(&want[4 * n_tokens..end]));
    assert_eq!(gs.len(), ws.len(), "the state's length");
    let f = |h: u16| f64::from(half_to_f32(h));
    let rms = (ws.iter().map(|&h| f(h) * f(h)).sum::<f64>() / ws.len() as f64).sqrt();
    let differ = gs.iter().zip(&ws).filter(|(a, b)| a != b).count();
    let max = gs.iter().zip(&ws).map(|(&a, &b)| (f(a) - f(b)).abs()).fold(0.0f64, f64::max) / rms;
    (equal, n_tokens, differ, ws.len(), max)
}

fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => f32::NAN,
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The distance the flat arm's chain may keep from the served one on the
/// commands below, as the largest |difference| over the state's rms: twice
/// the largest the unmutated arm read on the Mac when this test was written
/// (5.0e-2, the run from position 270; 1.1e-2 to 1.2e-2 on the others). A
/// one-step change of a binary16 value of 8 x rms is 0.8e-2 of rms; a wrong
/// phase moves the state by order 1 (the call-site mutants).
const FLAT_RMS: f64 = 0.1;

/// The flat arm's chain: the same commands as the E1 arms' (512-thread
/// symbols, positions past 256, two launches through one state, the 8B's
/// Tetra o), each against the served chain. Printed: tokens equal, the binary16
/// state values (residual and both layers' caches) that differ, and the
/// largest difference over the state's rms. Asserted: the chain runs, makes
/// every token, the barrier count and abort word are the served ones, and the
/// state stays within [`FLAT_RMS`] of its rms.
///
/// R6d's flat arms on the same commands and under the same bound: v9fn (flat on
/// q/k, down and o, v9's gate/up) and v9fg (flat everywhere, gate/up in two
/// calls), whose chain must also equal v9fl's byte for byte (the same rows,
/// the same product).
#[test]
fn the_4e_rc_flat_arm_stays_near_the_served_chain() {
    for arm in ["v9fl", "v9fn", "v9fg"] {
        flat_arm_near_the_served_chain(arm);
    }
}

fn flat_arm_near_the_served_chain(arm: &str) {
    let b = persistent_e1(arm, &format!("rc_{arm}"), None);
    let fl = (arm == "v9fg").then(|| persistent_e1("v9fl", "rc_v9fl_for_v9fg", None));
    let cmds: Vec<(String, Vec<u8>, usize, u32)> = vec![
        ("grid 4x96, RoPE 2".into(), g_cmd_variant(2, 4, 96, &[(5, 3, 7)], 16, u32::MAX, 0x9e42, 2 | 256), 3, 4),
        ("grid 5x64, RoPE 1".into(), g_cmd_variant(1, 5, 64, &[(5, 3, 7)], 16, u32::MAX, 0x9e41, 2 | 256), 3, 5),
        ("grid 3x512, RoPE 1".into(), g_cmd_variant(1, 3, 512, &[(5, 3, 7)], 16, u32::MAX, 0x9e41, 2 | 256), 3, 3),
        ("from 270, two splits".into(), g_cmd_variant(1, 3, 128, &[(270, 3, 7)], 288, u32::MAX, 0x9e52, 2 | 256), 3, 3),
        ("two launches".into(), g_cmd_variant(1, 4, 96, &[(5, 3, 7), (8, 2, 9)], 16, u32::MAX, 0x9e21, 2 | 256), 5, 4),
        ("the 8B's Tetra o".into(), g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e43, 2 | 16 | 256), 3, 3),
    ];
    let (mut tok, mut toks, mut dif, mut vals, mut worst) = (0, 0, 0, 0, 0.0f64);
    for (label, cmd, n, grid) in &cmds {
        let (want, got) = (run(served(), cmd), run(&b, cmd));
        assert_eq!(got.len(), want.len(), "{label}: the output's length");
        let bars = *n as u32 * pers::n_bar_4e(2) * grid;
        assert_eq!(u32s(&got[got.len() - 8..]), vec![bars, 0], "{label}: barrier count, abort and tickets");
        assert!(u32s(&got[..4 * n]).iter().all(|&t| t < 1024), "{label}: every token made");
        let (e, t, d, v, m) = flat_distance(&got, &want, *n);
        println!("{arm} {label:<22} tokens equal {e} of {t}; state values differing {d} of {v}; max |diff| / rms {m:.3e}");
        assert!(m <= FLAT_RMS, "{arm} {label}: the flat chain is {m:.3e} of the state's rms from the served chain");
        if let Some(fl) = &fl {
            assert!(got == run(fl, cmd), "{arm} {label}: not v9fl's chain byte for byte");
        }
        (tok, toks, dif, vals, worst) = (tok + e, toks + t, dif + d, vals + v, worst.max(m));
    }
    println!("{arm} over {} commands: tokens equal {tok} of {toks}; state values differing {dif} of {vals}; max |diff| / rms {worst:.3e}", cmds.len());
    if fl.is_some() {
        println!("{arm}: v9fl's chain byte for byte on the {} commands", cmds.len());
    }
    // the stamp symbol runs the same rows: its outputs equal the token symbol's
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(&b, &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 2 | 256));
    let stamped = run(&b, &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 1 | 256));
    assert!(stamped[..plain.len()] == plain[..], "{arm}: the stamped launch moved an output");
}

/// The lighter variants on the card's grid (142 CTAs, the L40S's SMs; one warp a CTA, so the twin stays small: 4,544
/// fibers of 64 KB of stack, about 0.3 GB; the grid cannot shrink, flat's gate/up slice fits one chunk from 134 CTAs
/// on, and 142 gives the card's own largest slice, 138 rows x 14 runs = 1,932 slots):
/// the one-chunk bodies need a CTA's slice to fit 2,048 slots, which the twin's usual grids of 3 to 5 CTAs do not
/// give. ep2 and ep3 equal the served chain byte for byte (tokens, residual, both layers' caches); fl2 and fl3 equal
/// v9fl's chain byte for byte (the same rows, `rowcand_gate::the_flat_variants_are_flat_bit_for_bit`), on a short
/// run and one from position 270, and on the stamped launch.
#[test]
fn the_4e_rc_one_chunk_arms_on_the_cards_grid() {
    let cmds = [
        ("from 5", g_cmd_variant(1, 142, 32, &[(5, 2, 7)], 16, u32::MAX, 0x9e90, 2 | 256)),
        ("from 270", g_cmd_variant(1, 142, 32, &[(270, 2, 7)], 288, u32::MAX, 0x9e91, 2 | 256)),
    ];
    let flat = persistent_e1("v9fl", "rc1_v9fl", None);
    for (label, cmd) in &cmds {
        let want = run(served(), cmd);
        let wflat = run(&flat, cmd);
        for arm in ["v9ep2", "v9ep3"] {
            let got = run(&persistent_e1(arm, &format!("rc1_{arm}"), None), cmd);
            assert!(got[..got.len() - 8] == want[..want.len() - 8], "{arm} {label}: not the served chain");
        }
        for arm in ["v9fl2", "v9fl3"] {
            let got = run(&persistent_e1(arm, &format!("rc1_{arm}"), None), cmd);
            assert!(got == wflat, "{arm} {label}: not v9fl's chain");
        }
        let (e, t, d, v, m) = flat_distance(&wflat, &want, 2);
        println!("142 CTAs {label}: v9ep2, v9ep3 the served chain; v9fl2, v9fl3 v9fl's; v9fl against served: tokens {e} of {t}, {d} of {v} values differ, max |diff| / rms {m:.3e}");
    }
    let stamp = g_cmd_variant(1, 142, 32, &[(5, 2, 7)], 16, u32::MAX, 0x9e90, 1 | 256);
    for arm in ["v9ep3", "v9fl3"] {
        let b = persistent_e1(arm, &format!("rc1_{arm}"), None);
        let plain = run(&b, &cmds[0].1);
        let stamped = run(&b, &stamp);
        assert!(stamped[..plain.len()] == plain[..], "{arm}: the stamped launch moved an output");
    }
}

/// Mutants of the rc call sites, each one site of an rc edit's replacement
/// text (`persistent_e1` counts it). The call sites are the same text in both
/// arms (p4e_rc_edits differs by the define only), so every mutant must be
/// killed on v9ep, against the served chain bit for bit. On v9fl the gate is
/// the chain distance of [`FLAT_RMS`], or the process failing (the
/// shared-memory guard traps): the mutants of what flat places differently
/// (its padded copy and its slots) must be killed there too; the others are
/// run and their verdict printed, a survivor named with its distance.
#[test]
fn the_4e_rc_call_site_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [
        g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e80, 2 | 256),
        g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e81, 2 | 16 | 256),
    ]
    .into_iter()
    .map(|c| {
        let want = run(served(), &c);
        (c, want)
    })
    .collect();
    // (tag, from, to, arms the mutant must be killed on, arms it is run and reported on)
    type Case<'a> = (&'a str, &'a str, &'a str, &'a [&'a str], &'a [&'a str]);
    let cases: [Case; 7] = [
        // gate/up: the product's operands swapped
        ("gu_product", "exact_hmul(A->silu[sgu[p]], sgu[np + p])", "exact_hmul(A->silu[sgu[np + p]], sgu[p])", &["v9ep"], &["v9fl"]),
        // gate/up: the up rows read from the gate matrix
        ("gu_matrix", "const SegTetra& M = up ? A->up : A->gate;", "const SegTetra& M = up ? A->gate : A->gate;", &["v9ep"], &["v9fl"]),
        // down: the residual of the CTA's first row on every row
        ("down_residual", "L.out[dlo + idx] = exact_hadd(__ldcg(L.x + dlo + idx), f2h(y));", "L.out[dlo + idx] = exact_hadd(__ldcg(L.x + dlo), f2h(y));", &["v9ep"], &["v9fl"]),
        // q/k: the region over xn, which the int4 v rows read after the q/k rows
        ("qk_region_on_xn", "const_cast<float*>(xr) + 2u * P.d_in, false);", "const_cast<float*>(xr) + P.d_in, false);", &["v9ep", "v9fl"], &[]),
        // down: flat's padded copy put after the natural one instead of in place; its slots then pass the v7 area
        // and the guard traps
        ("down_not_in_place", "v + pers_rc_padded_floats(nb, tw), true);", "v + pers_rc_padded_floats(nb, tw), false);", &["v9fl"], &[]),
        // o (the 8B's Tetra o): the residual of the CTA's first row on every row
        ("o_residual", "out[lo + idx] = exact_hadd(__ldcg(x + lo + idx), f2h(y));", "out[lo + idx] = exact_hadd(__ldcg(x + lo), f2h(y));", &["v9ep"], &["v9fl"]),
        // gate/up: the CTA's last pair dropped from the list
        ("gu_last_pair", "pers_rc_rows<PERS_RC_ARM>(2u * np, nb, tw,", "pers_rc_rows<PERS_RC_ARM>(2u * np - 2u, nb, tw,", &["v9ep"], &["v9fl"]),
    ];
    for (tag, from, to, must, report) in cases {
        for (arm, required) in must.iter().map(|a| (a, true)).chain(report.iter().map(|a| (a, false))) {
            let b = persistent_e1(arm, &format!("rc_{tag}_{arm}"), Some((from, to)));
            let (mut verdict, mut survived) = (None, String::new());
            for (c, w) in &cmds {
                match run_mutant(&b, c) {
                    None => verdict = Some("killed: the process failed".to_string()),
                    Some(o) if o.len() != w.len() => verdict = Some("killed: the output's length".to_string()),
                    Some(o) if *arm == "v9ep" => {
                        if o[..o.len() - 8] != w[..w.len() - 8] {
                            verdict = Some("killed: the chain moved".to_string());
                        }
                    }
                    Some(o) => {
                        let (e, t, _, _, m) = flat_distance(&o, w, 3);
                        if m > FLAT_RMS || e < t {
                            verdict = Some(format!("killed: {e} of {t} tokens equal, max |diff| / rms {m:.3e}"));
                        } else {
                            survived = format!("{e} of {t} tokens equal, max |diff| / rms {m:.3e}");
                        }
                    }
                }
                if verdict.is_some() {
                    break;
                }
            }
            match verdict {
                Some(v) => println!("rc mutant {tag:<18} 1 site, arm {arm}: {v}"),
                None if required => panic!("rc mutant {tag} survived on {arm}"),
                None => println!("rc mutant {tag:<18} 1 site, arm {arm}: SURVIVES the chain distance ({survived}); killed on v9ep, same text"),
            }
        }
    }
}

/// Mutants of R6d's two-call gate/up site (pers.rs::P4E_RC_GU_TWO_CALLS), on
/// v9eg against the served chain bit for bit. Each twist is one site of the
/// arm's replacement texts (`persistent_e1_twists` asserts it); a mutant of
/// two twists names two sites. Commands: the rc mutants' two (smooth SiLU
/// table, and the 8B's Tetra o) and one under the random table.
#[test]
fn the_4e_rc_two_call_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [
        g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e80, 2 | 256),
        g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e81, 2 | 16 | 256),
        g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x9e82, 2),
    ]
    .into_iter()
    .map(|c| {
        let want = run(served(), &c);
        (c, want)
    })
    .collect();
    let up_row = "return PersTRow{A->up.words + r * A->up.stride_u32, A->up.gscale, A->up.rscale, A->up.tail, r};";
    let gate_row = "return PersTRow{A->gate.words + r * A->gate.stride_u32, A->gate.gscale, A->gate.rscale, A->gate.tail, r};";
    let cases: [(&str, Vec<(&str, &str)>); 5] = [
        // silu applied on the up call's value instead of the gate call's
        (
            "gu2_silu_on_up",
            vec![
                ("A->act[glo + idx] = A->silu[f2h(y)];", "A->act[glo + idx] = f2h(y);"),
                ("exact_hmul(A->act[glo + idx], f2h(y))", "exact_hmul(A->act[glo + idx], A->silu[f2h(y)])"),
            ],
        ),
        // the second call reads the gate matrix
        ("gu2_up_reads_gate", vec![(up_row, gate_row)]),
        // the gate call one row short
        (
            "gu2_gate_short",
            vec![(
                "A->silu[f2h(y)]; };\n            pers_rc_rows<PERS_RC_ARM>(ghi - glo,",
                "A->silu[f2h(y)]; };\n            pers_rc_rows<PERS_RC_ARM>(ghi - glo - 1u,",
            )],
        ),
        // the up call one row short
        (
            "gu2_up_short",
            vec![(
                "f2h(y)); };\n            pers_rc_rows<PERS_RC_ARM>(ghi - glo,",
                "f2h(y)); };\n            pers_rc_rows<PERS_RC_ARM>(ghi - glo - 1u,",
            )],
        ),
        // the product replaced by the up value alone
        ("gu2_no_product", vec![("exact_hmul(A->act[glo + idx], f2h(y))", "f2h(y)")]),
    ];
    for (tag, twists) in &cases {
        let b = persistent_e1_twists("v9eg", &format!("rc_{tag}_v9eg"), twists);
        let mut verdict = None;
        for (c, w) in &cmds {
            verdict = match run_mutant(&b, c) {
                None => Some("killed: the process failed".to_string()),
                Some(o) if o.len() != w.len() => Some("killed: the output's length".to_string()),
                Some(o) if o[..o.len() - 8] != w[..w.len() - 8] => Some("killed: the chain moved".to_string()),
                Some(_) => None,
            };
            if verdict.is_some() {
                break;
            }
        }
        match verdict {
            Some(v) => println!("rc two-call mutant {tag:<18} {} site(s), arm v9eg: {v}", twists.len()),
            None => panic!("rc two-call mutant {tag} survived on v9eg"),
        }
    }
}
