# Prereg SP7: a per-operation profile of the DFlash drafter round, sealed 4B

Written on 2026-11-11 at about 04:40 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp7-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-10 evening ("a real test on the card"); cap $15 with leave to launch from about 21:45 CET;
relayed by the orchestrating session. Cap: **$15**, $7.28 spent before this job. Registry before this job: **$314.53**.
Cost: about **$0.15** on l40sx1 (*estimated*, "Cost"), timeout 14 min, **$0.42** at worst. Measured code: merge commit
`b8a6fc3`, which brings in f3eb2df ("dflashcheck profile: every operation of a DFlash round timed alone"): a `profile`
mode in `llvq-llm/src/bin/dflashcheck.rs`, the module `llvq-llm/src/dflash_profile.rs`, and its declaration in
`llvq-llm/src/dflash.rs`. The image is published from a clean detached checkout of the commit that holds this file's
stamp, whose kernel, bench and runtime sources are b8a6fc3's (only the job script and the proofs moved after it).

## What this job is

A measurement. It changes nothing and tests no repair. It times each device operation of the drafter's round, with a
synchronisation after each:
- the context extension;
- the block;
- the head;
- both readouts, on the device and chunked.

It also times three candidate readouts:
- (a) max and argmax only;
- (b) the top 4 by argmax-and-mask on the f16 logits, with a logsumexp;
- (c) the readout of commit 1db1acb as it was.

## Question

Which device operations make up the drafter's round on this card, and what does each cost alone? What would the three
candidate readouts cost?

## Setup

One image, one job, `ops/jobs/spec-loop-sp7.sh`, on l40sx1 only (no twin). Inputs as SP6 for the drafter and the sealed
file: the fixture (sha256 636f664d), the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (`config.json`
2de97cdf, `model.safetensors` a21d0ffe), the sealed 4B at sha256 886391a8. Each step to its own file, non-fatal for the
next:

1. `oracle` on Qwen3-0.6B; the sealed file, fixture and drafter files at their sha256.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16 profile`, 5 min, into `profile-f16.txt`.
3. (b) the same in f32, 5 min, into `profile-f32.txt`.

The profile uses a context of 512 positions built from the fixture's taps, and an extension of 4 positions.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the sealed file, fixture and drafter files at their sha256; no `LLVQ_*` in the job
   environment.
2. If the image fails to build, nothing is launched; the build error goes back to the orchestrating session with its
   text.

There is no specbench run, so there is no 4E module control.

## What gets published, and what does not get compared

Published: both profile files in full, every line; the stage sums; the whole-stage times; f16 then f32. Not established:
anything about the loop's speed; any repair.

## Decision rule

| result | outcome |
|---|---|
| both profile runs print their tables | VALID: the tables are read; the one signed line is read against its prediction |
| a profile run fails | that dtype's tables are missing; the journal says so with the error |

## Signed prediction

Signed by the orchestrating session. One thing only: the stages run whole will reproduce SP6's split within 15 %.

| stage, run whole (f16) | SP6's value, reproduced within 15 % |
|---|---|
| extension at 4 positions | 0.35 to 0.47 ms |
| block | 1.9 to 2.0 ms |
| dense head | 1.12 ms |
| device readout | 2.0 ms |
| chunked readout | 1.9 ms |

No value is signed for any single operation or for the candidates. Three estimates from operation counts have failed on
this card, and finding the costly operation is the purpose of the job.

## Cost

About 5 min with the queue excluded, $0.15 (*estimated*). Timeout 14 min: $0.42 at worst, under the cap. On l40sx1
only: no twin.
