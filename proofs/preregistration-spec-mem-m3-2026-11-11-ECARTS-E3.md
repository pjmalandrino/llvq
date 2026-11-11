# Prereg M3: deviation E3

Beside `preregistration-spec-mem-m3-2026-11-11.md` (stamped, commit cf903db) and its first deviations file
`preregistration-spec-mem-m3-2026-11-11-ECARTS.md` (E1 and E2, stamped, commit 5b6c84a). Neither is edited: this one
is a file of its own so that both stamps still verify. Written on 2026-11-11 before the job was submitted, and stamped
before submission.

## E3. The weightless configuration: the tree pass without any drafter model

Operator ruling, relayed by the orchestrating session: the whole-model b/param must stay at the served 2.73, so the
target configuration is now the tree pass without any drafter model.

Added to `ops/jobs/spec-mem-m3.sh`, after the two house runs, in its own process beside its own 100 ms nvidia-smi
sampler (tag `house-null-a12ip`):
`LLVQ_CONFIG=<v9> specbench --prompts <house 44> <sealed> a12ip@8.10.8.20.6 null,oracle:3 256`.

No DFlash weights and no drafter vocabulary are loaded. This was checked in the code, and no code changed:
- `specbench.rs:189` sets `want_dflash` only for a `dflash:*` drafter;
- the drafter directory, its weights and the q4 or dense vocabulary are read only under `if want_dflash` (`:247`);
- `--vocab` and `--readout` are not passed.
The second `card memory:` line, after the drafter's load, is therefore not printed in this run. The first one is
printed, after the verify pass's install.

The summary prints this run's peak and tok/s beside the base-short pair, for `null` and for `oracle:3`. For `null` it
also prints the verify pass time, as 1000 / tok/s: the null drafter is one token a pass.

Dropped: nothing. There was no a12 run on GSM8K. a12 runs on the house set only, as the reference for the 1.308 GB drop.

Signed prediction, added by the orchestrating session, verbatim: peak at a short context 2.3 GB [2.1, 2.6], that is
base +0.25 GB [0.1, 0.5].

## Cost

This step adds about 8 min: a load, then the house set through 4E's greedy, the pass and two drafters at about 139
tok/s for the null drafter (spec-loop-sp2). New estimate: about 66 min, **$1.98** (*estimated*). Timeout 86 min:
**$2.58** at worst, under $2.60. $10.91 of $15 at worst after it.
