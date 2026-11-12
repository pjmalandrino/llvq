# Deviations from prereg SP1 (`preregistration-spec-loop-sp1-2026-11-10.md`)

The prereg is not edited.

## Deviation 1: SP1 was not submitted; superseded by SP2

Written on 2026-11-10 at about 19:45 CET. SP1's image (Space sha a0cbaec0, published from 16bbb92 at 19:13 CET) was still
building at 19:39; the launch chain was stopped before any submission, on the orchestrating session's instruction, once
the real drafter was merged on the branch (merge d555263, integration 9e00978). No SP1 job exists; nothing was measured or
billed.

SP1 is superseded by SP2 (`preregistration-spec-loop-sp2-2026-11-10.md`, measured code 9e00978), whose first steps run
SP1's content (`vtreebench --res a12`; `specbench` with the null drafter and the oracles at K = 3 and 14) under SP1's
predictions, copied there. The oracles at K = 1 and 7 are not rerun.
