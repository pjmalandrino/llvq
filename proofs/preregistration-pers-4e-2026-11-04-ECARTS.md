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
