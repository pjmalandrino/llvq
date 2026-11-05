# Deviations of proofs/preregistration-pers-4e-2026-11-04.md

## E1, written before the retry: the first run was void, and the retry runs fixed code

The first run (job 6aebcb970e0338d1bb804b35, image 9d092a0f, `docs/mesures/pers-4e-2026-11-05.txt`) was void by the
prereg's identity rule: the first decode at a 5-token context gave 64 of 64 tokens equal to the eager path's, then the
first timing round moved its tokens. The cause, found on the host: the rewound decode read a stale argmax key (the
argmax uses two 64-bit slots, and the slot picked after the rewind still held the previous decode's key). Fixed in
commit 3e186a0 on `claude/serene-bohr-1edf69`, merged into `integration/lot1` at e403f4a; a host test that runs
consecutive decodes fails before the fix and passes after, and a mutant that skips the reset is caught.

The retry runs under this prereg unchanged, with `RETRY=2`, output `/out/pers-4e-2026-11-04-r2`, on the image rebuilt
from e403f4a. Same gates, same rules, same ceiling (40 min, $1.20).

## E2, written before the third run (R3): the 8,192 prefill is chunked, and the run prints a phase table

The retry (job 6aebd50b0e0338d1bb804dae, `docs/mesures/pers-4e-2026-11-05-r2.txt`) read the 5-token context: same
tokens as the eager path, 4E against eager x0.993. Its 8,192 context failed before any 4E timing, on the prefill guard
("8192 vectors at once, more than 4096") of two unchunked calls. Commit c4f4c51 on `claude/serene-bohr-1edf69` sends
both in chunks of 4,096, as the context sweep already does; nothing changes at or under 4,096. Commit 7fa4ff2 adds,
after the timed rounds, one stamped decode a context on 512-thread variants of the kernel that record a timestamp at
every barrier for every CTA; the timed kernel compiles to identical SASS. Both are merged into `integration/lot1`.

The third run keeps this prereg's gates, identity rule and timing rules unchanged, with `RETRY=3`, output
`/out/pers-4e-2026-11-04-r3`, on the image rebuilt from `integration/lot1` at the merge. Ceiling unchanged (40 min,
$1.20); expected about 3 minutes and $0.10 (*estimated*).

Added reading of the phase table, fixed here before the run (from section 4 of
`docs/mesures/pers-4e-loop-2026-11-05.txt`). The stamped decode must give the eager tokens, or its stamps are void and
none of the three readings below is made.
- Cause 1 (row split) holds if the slowest CTA's work over the mean CTA's is 1.3 or more on o, down and q/k/v, with
  q/k/v's slowest CTA at index 118 or above.
- Cause 2 (iteration rate) holds if the mean CTA's work over the rows' phases streams the 1.377 GB at 350 GB/s or
  less.
- The fixed work and barriers are read as the walls of norm/RoPE/append and attention, plus each phase's wall minus
  its slowest CTA's work.
Whichever holds names the first change to build, at $0, before any further card time. The table is a diagnosis: it
changes nothing served.
