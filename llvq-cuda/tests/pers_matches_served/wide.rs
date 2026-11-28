//! 4E at the 8B and the 14B: a Tetra o and an untied head, against the served
//! chain. One file for the two shapes the 4B does not have.

use super::*;

// ---- 4E at the 8B and the 14B: a Tetra o, an untied head --------------------

/// The 8B's two differences in the token loop (variant + 16): layer 1's
/// o_proj in Tetra under its own rotation at width 4,096, and a head table of
/// its own. Every form (v4, v6, v8, v9), both row variants, consecutive
/// launches, positions past 256, and the stamp symbol, against the served
/// chain (rot_apply then tv_tetra48_h for that o, tv_emb_q4_h on that table).
#[test]
fn the_4e_tetra_o_and_untied_head_are_the_served_chain() {
    type Side = fn() -> &'static Path;
    let forms: [(&str, Side); 4] =
        [("v4", persistent_v4), ("v6", persistent_v6), ("v8", persistent_v8), ("v9", persistent_v9)];
    for (label, side) in forms {
        for (form, grid, threads, variant) in [(1u32, 3u32, 128u32, 2u32), (0, 4, 96, 0), (2, 2, 512, 2)] {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x8b00 + form as u64, variant | 16);
            let (want, got) = (run(served(), &cmd), run(side(), &cmd));
            assert!(got[..got.len() - 8] == want[..want.len() - 8], "{label}, form {form}, {grid}x{threads}, variant {variant}: Tetra o, untied head");
        }
    }
    let runs = [(270u32, 3u32, 7u32), (5, 2, 9)];
    let cmd = g_cmd_variant(1, 3, 128, &runs, 288, u32::MAX, 0x8b10, 2 | 16);
    let (want, plain) = (run(served(), &cmd), run(persistent_v9(), &cmd));
    assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "v9: past 256 and a rewound second launch");
    let stamped = run(persistent_v9(), &g_cmd_variant(1, 3, 128, &runs, 288, u32::MAX, 0x8b10, 1 | 16));
    assert!(stamped[..plain.len()] == plain[..], "v9: the stamped launch moved an output");
    // the extension is not vacuous: its tokens or state differ from the 4B
    // shape's on the same seed
    let base = run(served(), &g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x8b01, 2));
    let ext = run(served(), &g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x8b01, 2 | 16));
    assert!(base != ext, "the Tetra o and the untied head change nothing in the served chain");
}

/// Each new guard has a mutant, killed by the 8B commands: the head reading
/// the embedding's table, the Tetra o phase skipped, its rotation dropped,
/// its residual read from the wrong row, its rows starting one too far.
#[test]
fn the_4e_tetra_o_and_untied_head_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(1u32, 3u32, 128u32, 2u32, 0x8b20u64), (1, 4, 96, 0, 0x8b21)]
        .iter()
        .map(|&(form, grid, threads, variant, seed)| {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, seed, variant | 16);
            let want = run(persistent_v9(), &cmd);
            (cmd, want)
        })
        .collect();
    let f = "llvq-cuda/kernels/pers_4e.cu";
    for (tag, from, to) in [
        ("head_tied", "pers_head_argmax<PF>(A->head_w, A->head_s, A->head_b,", "pers_head_argmax<PF>(A->emb_w, A->emb_s, A->emb_b,"),
        ("o_tetra_skipped", "            if (Y.ot != nullptr)\n                pers_4e_o_tetra(", "            if (false)\n                pers_4e_o_tetra("),
        ("o_tetra_no_rot", "    pers_rot_finish(cs, O.small, d_ctx, O.m, O.k, O.inv);\n    const u32 lane", "    __syncthreads();\n    const u32 lane"),
        ("o_tetra_resid", "const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw); };\n    const auto epi = [&](u32 k, float y) { out[lo", "const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw + 1u); };\n    const auto epi = [&](u32 k, float y) { out[lo"),
        ("o_tetra_rows", "    pers_4e_rows(pers_warp_rows(lo, hi, warp, nw), O.t.nblocks", "    pers_4e_rows(pers_warp_rows(lo + 1u, hi, warp, nw), O.t.nblocks"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "8B mutant {tag} survived");
    }
}
