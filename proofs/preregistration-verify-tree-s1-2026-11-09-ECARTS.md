# Deviations from proofs/preregistration-verify-tree-s1-2026-11-09.md

The prereg is stamped and is not edited. Its departures are written here.

## Deviation 1, 2026-11-09: the bench stopped after the chat context

The job ran (6af20dabab34849891a881f8, 4 min billed, $0.12) and `vtreebench` exited 1 after the chat context's timed
rounds, at its fifth step, the stamped 4E token. That step stamps a launch of one token and passes it to
`p4e_phase_report`, which refuses fewer than two tokens by design (`llvq-cuda/src/pers.rs`: token 0 has no previous
end, so `n >= 2`). The error line: `4E stamps: 77106 stamps for 1 tokens x 543 events x 142 CTAs (n >= 2)`. The rag
context, both contexts' phase tables and the final `c_16` line were therefore not produced. The defect is in the bench
(`llvq-llm/src/bin/vtreebench.rs`, step 5), not in the measured passes; the host check never ran that step.

What is read: both controls (they hold), and at the chat context the identity, the `vt1` check and every c_R. The
decision rule reads the larger of the two contexts' c_16 and c_8. The chat values, 3.092 and 2.922, already pass both
FAIL bounds, and a second context can only raise the larger, so the outcome FAIL does not depend on the missing
context. The identity and `vt1` checks at the rag context are not read.

Not rerun. A rerun needs the bench fixed (stamp two tokens, or print 4E's stamps without the phase split) and the
operator's go. Its value would be the phase table, which locates the pass's fixed cost.
