# Prereg SP2: the speculative loop end to end on the card, stand-in drafters and DFlash, sealed 4B

Written on 2026-11-10 at about 19:55 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp2-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-10 evening, for the end-to-end series ("a real test on the card"), relayed by the orchestrating
session; the night's go of 2026-11-09 about 23:00 CET set the cap. Cap: **$10**, $4.34 spent before this job. Registry
before this job: **$311.59**. Cost: about **$0.90** on l40sx1 (*estimated*, "Cost"), timeout 45 min, **$1.35** at
worst. Measured code: commit `9e00978` ("DFlash behind the spec loop: the shared contract, the card's table, specbench
drafters"), after the merge d555263 of the drafter branch; the image is published from a clean detached checkout of the
commit that holds this file's stamp, whose kernel, bench and runtime sources are 9e00978's (only the job scripts, the
CUDA image's two binary lists, which gain `dflashcheck`, and the proofs moved after it).

SP2 supersedes SP1 (`preregistration-spec-loop-sp1-2026-11-10.md`), which was never submitted (its ECARTS, deviation 1).
SP2's first steps run SP1's content under SP1's predictions, copied below.

## Question

Does the host-side speculative loop around the tree verify pass emit 4E's greedy tokens, with stand-in drafters and with
the real drafter, and what does it cost end to end? The real drafter is DFlash (`z-lab/Qwen3-4B-DFlash-b16` at revision
b74e3a32, MIT, about 1.0 GB), its vocabulary built from the sealed file. Its proposals only propose: equality with 4E's
greedy tokens must hold whatever it proposes.

## Setup

One image, one job, `ops/jobs/spec-loop-sp2.sh`. Inputs: the fixture `llvq-dflash-fixture-2026-11-10.safetensors`
(8,562,156 bytes, sha256 636f664d) from the jobs bucket, checked at its sha256 in the job; the drafter's `config.json`
(sha256 2de97cdf) and `model.safetensors` (sha256 a21d0ffe) fetched at the pinned revision into a job-local directory and
checked at their sha256. Controls first, then each step to its own file, non-fatal for the next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `vtreebench --res a12`, 4 min: a12's registers, local and shared bytes, no model loaded.
3. (b) `dflashcheck <fixture> cuda <sealed> <drafter> f32`, then the same with `f16`, 5 min together: the drafter on
   CUDA against the fixture written by its Mac reference (chain, tree, logits over 6 rounds), then the time of a round
   at contexts of 128, 512 and 2,048. The binary runs the equivalence in f32 whatever its last argument; the argument
   sets the dtype of the timed rounds only.
4. (c) `specbench <sealed> a12@8.10.8.20.6 null,oracle:3,oracle:14 64`, 12 min: SP1's stand-ins on four prompts (chat,
   rag, code, math), 64 tokens each.
5. (d) `specbench <sealed> a12@8.10.8.20.6 dflash:chain,dflash:tree 64 <drafter> f16`, 12 min, with `f32` instead if the
   f16 run of (b) failed or did not print PASS (the summary says which).

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture and drafter files at their sha256; no `LLVQ_*` in the job
   environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), in both specbench runs.
3. If the image fails to build, nothing is launched.

Token equality with 4E's greedy tokens, 64 of 64 on every prompt for every drafter, is the hard gate: anything else voids
the job's speed lines for that drafter, and a failed equality stops the series until it is explained.

## What gets published, and what does not get compared

Published: dflashcheck's lines; the equality lines; per prompt and drafter the accepted tokens a pass, passes, the verify,
drafter (split as the bench prints it) and other walls, tok/s end to end and without the drafter, 4E's greedy tok/s; the
oracle check lines; a12's registers, local bytes and c_R. Not established by this job: any served rate; acceptance on
standard benchmarks; anything at 8B or 14B; long context.

## Decision rule

| result | outcome |
|---|---|
| equality holds everywhere, and the stand-ins' oracle ratios sit inside their intervals | the loop is sound and its overhead is known |
| equality holds for DFlash | the DFlash lines are read against their predictions below |
| any equality fails | the journal names it; the series stops until it is explained |

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at the plan [8, 10, 8, 20, 6]. Signed by the orchestrating session.

Stand-ins (SP1's, copied):

| quantity | point | interval |
|---|---|---|
| token equality with 4E's greedy, every prompt, null and oracles | 64 of 64 | |
| a12 registers, local memory | 168, 736 B | |
| a12 c_1 | 1.56 | [1.52, 1.66] |
| a12 c_16 | 1.75 | [1.70, 1.86] |
| null drafter, tok/s end to end over 4E's greedy tok/s | 0.60 | [0.48, 0.66] |
| oracle:3, measured tok/s over its check line greedy x (K+1) / c_(K+2) | 0.90 | [0.72, 1.00] |
| oracle:14, measured tok/s over its check line | 0.88 | [0.68, 1.00] |
| oracle:14 on the chat prompt, tok/s | 1,650 | [1,250, 1,950] |
| host time outside the verify launches, null drafter, share of the loop's wall | under 12 % | |

The real drafter:

| quantity | point | interval |
|---|---|---|
| `dflashcheck` on CUDA, f32: same chain and same tree as the fixture | 6 of 6 rounds | |
| `dflashcheck` on CUDA, f16: chains equal; logits relative | at least 5 of 6; within 2e-2 | |
| token equality with 4E's greedy, every prompt, dflash:chain and dflash:tree (hard gate) | 64 of 64 | |
| skipped rounds | 0 | |
| accepted tokens a pass (emitted tokens over passes), chain, chat | 2.2 | [1.5, 3.2] |
| same, code | 4.3 | [2.8, 6.0] |
| same, math | 5.0 | [3.2, 7.0] |
| same, rag | 3.0 | [1.8, 5.0] |
| tree, accepted tokens a pass | at least the chain's on every prompt, within noise | |
| one drafter round on the card (context extension + block + head + readout) | 6.5 ms | [3.0, 14.0], the head more than a third of it |
| end-to-end tok/s over 4E's greedy, chain, chat | x0.75 | [0.45, 1.15] |
| same, code | x1.35 | [0.80, 2.10] |
| same, math | x1.55 | [0.90, 2.40] |
| outcome | code and math above greedy, chat below, with an unoptimised drafter | |

The f16 equivalence line cannot be read from `dflashcheck` as written: its equivalence runs in f32 whatever its last
argument (step 3). The f16 lines it gives are the timed rounds; whether f16 changes the drafter's proposals shows only
through (d)'s accepted tokens, never through equality, which holds whatever the drafter proposes.

## What this will not establish

Any served rate; acceptance on standard benchmarks; anything at 8B or 14B; long context. The oracle at K = 1 and 7
(SP1's) is not rerun.

## Cost

About 30 min, $0.90 (*estimated*): the drafter download (about 1.0 GB), four compile and load steps, two specbench runs.
Timeout 45 min: $1.35 at worst, under the cap. On l40sx1; if it is not RUNNING after 20 minutes, a twin on l40sx4 (one
device visible) is added under a deviation written beside this file before it is submitted, and the first to run is the
measurement; the other is cancelled by hand once one copy is seen RUNNING or COMPLETED, and the cancellation is confirmed.
