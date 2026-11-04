// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §5, steps V0c and V3). Not part of the
// repository. Sketch of two measurement modes of llvq-llm/src/bin/fusedrun.rs,
// both built on the house pattern of LLVQ_TRIM_AB (`run_trim_ab`,
// fusedrun.rs:178-247): one model, every arm in every round, same order, two
// rounds discarded then ROUNDS_TIMED (5), each ratio formed round by round.
//
// Both are MEASUREMENT MODES, never a served config:
//   * refused by name beside LLVQ_CONFIG: add them to the list at
//     fusedrun.rs:677-679, which already refuses LLVQ_TRIM_AB, LLVQ_GQA_AB...;
//   * each runs alone: refused beside LLVQ_GQA_AB, LLVQ_TRIM_AB, LLVQ_GRAPH_AB,
//     LLVQ_CONTEXTS, LLVQ_PREFILL_TOKENS and each other;
//   * run at the bench door with the env of the e4 configs:
//     LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0,
//     KV f16 and the expanded attention (both defaults). The grouped form is
//     reachable only through a config's `attention` field or LLVQ_GQA_AB, so
//     these modes measure the expanded form; serving them under the grouped
//     config would need an operator ruling admitting them beside LLVQ_CONFIG,
//     as LLVQ_CONTEXTS is admitted.
//
// The runtime side: FusedCuda (fused_cuda.rs) keeps `f_rot` and `f_rot_rows`
// as today and gains, only under one of these modes, a second pair loaded from
// a unit of its own (rot_null.cu, or rotate_wide.cu for R1) with the SAME
// dynamic-shared attribute as the first pair (func_dynamic_shared with the
// model's widest rotation bytes, fused_cuda.rs:768-783), plus
// `set_rot_arm(RotArm)` that swaps which pair the three launch sites
// (:1423, :1493, :1940) use. Nothing else moves between arms.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RotArm {
    /// `rot_apply` / `rot_apply_rows`, the served kernels.
    K0,
    /// LLVQ_ROT_AB: `rot_apply_r` / `rot_apply_rows_r`, bit-exact by design.
    R1,
    /// LLVQ_ROT_NULL_AB: `rot_null` / `rot_null_rows`, same launch, no work.
    Null,
}

// ---- LLVQ_ROT_AB=1 (V3): K0 against R1, bit for bit ----------------------
//
// Arms [K0, R1]. Rule of run_trim_ab: any token that differs between two arms
// or two rounds stops the run before a number is printed, since R1 claims bit
// identity and a speed measured on arms that disagree is not a speed of the
// same thing. Printed: tok/s per arm, median [min-max]; K0-rate / R1-rate is
// NOT the ratio reported, R1-rate / K0-rate is (>1 means R1 faster), round by
// round, median [min-max]. Also printed: the sha256 of the decoded line, which
// at 4B must equal 8784ffc4..., the paper-table reference line of the bench
// door (ops/jobs/served-grouped.sh, REF_LINE_SHA), for a 5-token seed prompt
// and 256 new tokens.
//
// ---- LLVQ_ROT_NULL_AB=1 (V0c): the ceiling of a faster rotation ----------
//
// Arms [K0, Null]. The K0 arm's tokens must be identical in every round (and
// at 4B hash to 8784ffc4...); the Null arm's tokens are garbage by design,
// printed with that label, and never compared. Printed: tok/s per arm and
// Null-rate / K0-rate round by round, median [min-max]: the most any rotation
// that keeps its launch can buy on this object and this door.
//
// fn run_rot_ab(model, ids, n_new, arms: &[RotArm], compare_all: bool,
//               set: impl FnMut(&mut Qwen3, RotArm) -> anyhow::Result<()>)
//     -> anyhow::Result<()>
// {
//     const WARMUP: usize = 2;
//     let mut reference: Option<Vec<u32>> = None;      // K0's tokens
//     let mut rates = vec![Vec::new(); arms.len()];
//     for round in 0..WARMUP + ROUNDS_TIMED {
//         for (k, &arm) in arms.iter().enumerate() {
//             set(model, arm)?;
//             let t = Instant::now();
//             let out = model.generate(ids, n_new, &mut NoCapture)?;
//             let rate = n_new as f64 / t.elapsed().as_secs_f64();
//             let checked = arm == RotArm::K0 || compare_all;   // Null never compared
//             if checked {
//                 match &reference {
//                     None => reference = Some(out),
//                     Some(r) => if let Some(i) = first_divergence(r, &out)? {
//                         anyhow::bail!("round {round}, arm {arm:?}: divergence at token {i}; void");
//                     },
//                 }
//             }
//             if round >= WARMUP { rates[k].push(rate); }
//         }
//     }
//     set(model, RotArm::K0)?;                          // leave the served state
//     // print per-arm rate_stats and arm/K0 ratios round by round (rule 7)
// }
//
// Callers: LLVQ_ROT_AB -> run_rot_ab(.., &[K0, R1], true, ..);
//          LLVQ_ROT_NULL_AB -> run_rot_ab(.., &[K0, Null], false, ..).
//
// Mutants the portable tests must kill (the mode's argument logic is
// testable on the Mac; the cfg-gated body is not): the Null arm compared
// (the null mode then always dies); the R1 arm NOT compared (a wrong R1
// would print a speed); either mode accepted beside LLVQ_CONFIG.
