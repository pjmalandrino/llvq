# Departures from the prereg chain-runtime-door (2026-11-15)

The prereg `preregistration-chain-runtime-door-2026-11-15.md` (sha256 07cf3e6d, stamped before the run) is not
edited. This file records what was done that its decision rule does not provide for.

## E1. The merge went ahead on a verdict that reads FAILS

The rule's row: "a control fails, or the way back does not load v9, or a profile is not refused: FAILS: fix before
the merge". Control 4, texts identical across rounds within each server, failed
(`docs/mesures/chain-runtime-door-2026-11-15.txt`).

What failed is the eager server, on the long prompt: one text with 0 prompt tokens cached, another with 3. The chain
held the control on both prompts, and its text is the eager server's of the three timed rounds. The row was written
for a failure of the path under test and fired on the path it replaces, which is already served on main.

The lead reported the verdict to the letter and did not merge. The operator decided on 2026-11-15, with the reading
in hand, to merge the chain prefill as the 4B's default on CUDA. Nothing was fixed before the merge. The eager
prefill's dependence on the conversation cache is entered in `docs/ETAT.md` section 5 as an open item.

No figure of the journal changes. No reading was rerun.
