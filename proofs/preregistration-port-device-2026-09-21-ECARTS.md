# Deviations from the port prereg of 2026-09-21

Beside the stamped file, never into it.

## É1. The identity gate ran 32 tokens, not the 256 the prereg named

§4 of the prereg reads: "**256 tokens identical to the dense arm, in the same
process, on both arms.**" The run did **32**, on both arms, which is
`fusedrun`'s default.

The fault is in `ops/jobs/port-device-fusedrun.sh`: it sets the served flags
and nothing else, so nothing asked for 256. The prereg promised a number the
script did not request, and neither the script nor the prereg was read against
the other before the launch.

What this costs. The identity claim for this object stands at **32 tokens**,
which is where it already stood after `dclm-ft-fusedrun-2026-09-20`. The run
therefore did not close the open item that `docs/plan-papier-2-2026-09-20.md`
lists as "256 tokens on the final object, $0.30". That item is still open and
still costs $0.30.

What it does not cost. The gate's PURPOSE was to refuse a dispatch refactor
that changes the arithmetic, and 32 tokens of greedy decode through 252 fused
matvecs and 144 rotations would show a changed matvec. The claim is narrower
than promised, not unsupported.

## É2. The cost came in under the announced figure

Announced about $0.30, billed about **$0.12** for 233 s of l40sx1. Two arms of
one process cost less than the one-arm reference did, because the model loads
once per arm and the dense arm dominates the wall clock.

Recorded because the announcement is the rule, and an announcement that is
only ever too high is still an announcement that is wrong.
