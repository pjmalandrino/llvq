# Prereg SP1: the speculative loop end to end on the card, with stand-in drafters, sealed 4B

Written on 2026-11-10 in the evening on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp1-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-10 evening, for the end-to-end series ("a real test on the card"), relayed by the orchestrating
session; the night's go of 2026-11-09 about 23:00 CET set the cap. Cap: **$10**, $4.34 spent before this job. Registry
before this job: **$311.59**. Cost: about **$0.48** on l40sx1 (*estimated*, "Cost"), timeout 28 min, **$0.84** at
worst. Measured code: commit `5e65b12` ("Tree verify: a12 (taps), the speculative loop, and specbench"); the image is
published from a clean detached checkout of the commit that holds this file's stamp, whose kernel, bench and runtime
sources are 5e65b12's (only the job script and the proofs moved after it).

## Question

The tree verify pass has been measured launch by launch (N1 to N11). Does the host-side speculative loop around it
(`llvq-cuda/src/spec_loop.rs`) emit 4E's greedy tokens, and what does it cost end to end? No real drafter exists yet. Two
stand-ins: a null drafter (one token a pass, R = 1) and an oracle that replays 4E's greedy continuation with K correct
tokens then a wrong one (a pass of K + 2 rows accepting K). This is not a served number: the oracle knows the answer and
costs nothing; it bounds what a perfect drafter of that acceptance would give, and it prices the loop's host overhead.

## Setup

One image, one job, `ops/jobs/spec-loop-sp1.sh`:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench --res a12`, its own timeout 4 min: a12's module through NVRTC, its registers, local and shared bytes, no
   model loaded. Its failure voids nothing (specbench's install prints the same numbers).
4. `LLVQ_CONFIG=<v9 config> specbench <sealed> a12@8.10.8.20.6 null,oracle:1,oracle:3,oracle:7,oracle:14 64`, its own
   timeout 22 min. a12 is a10t384r16 (384 threads, at most 16 rows) plus taps of the residual at the inputs of layers 2,
   10, 18, 26 and 34, at the plan [8, 10, 8, 20, 6]. On four prompts (chat, rag, code, math), 64 tokens each, 1
   discarded and 3 timed runs per prompt and drafter: the loop's tokens against 4E's greedy tokens, accepted tokens a
   pass, passes, the walls of the verify launches, of the drafter calls and of the rest of the host, tok/s end to end
   against 4E's greedy tok/s on the same prompt; for oracle:K the check line greedy x (K + 1) / c_(K+2), c_R a tree
   launch of R chained rows against one 4E token, timed in the same process.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. If the image fails to build, nothing is launched.

Token equality with 4E's greedy tokens, 64 of 64 on every prompt for every drafter, is read first: anything else voids
the job's speed lines, and a failed equality stops the series until it is explained.

## What gets published, and what does not get compared

Published: the equality lines; per prompt and drafter, accepted tokens a pass, passes, the verify, drafter and other
walls, tok/s end to end, 4E's greedy tok/s, the check line and the ratio; a12's registers, local and shared bytes and its
c_R; the `--res` lines. Not compared: any tok/s as a served rate (the oracle is not a drafter); the 8B and 14B; another
card.

## Decision rule

| result | outcome |
|---|---|
| equality holds everywhere, and the oracle ratios (measured tok/s over the check line) sit inside their intervals | PASS: the loop is sound and its overhead is known |
| otherwise | the journal names the line that failed; a failed equality stops the series until it is explained |

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at the plan [8, 10, 8, 20, 6]. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| token equality with 4E's greedy, every prompt, every drafter | 64 of 64 | |
| a12 registers, local memory | 168, 736 B | |
| a12 c_1 | 1.56 | [1.52, 1.66] |
| a12 c_16 | 1.75 | [1.70, 1.86] |
| null drafter, tok/s end to end over 4E's greedy tok/s | 0.60 | [0.48, 0.66] |
| oracle:K, tokens emitted a pass | exactly K + 1, the last pass possibly short | |
| oracle:K, measured tok/s over the check line, K = 1, 3, 7 | 0.90 | [0.72, 1.00] |
| oracle:14, measured tok/s over the check line | 0.88 | [0.68, 1.00] |
| oracle:14 on the chat prompt, tok/s | 1,650 | [1,250, 1,950] |
| host time outside the verify launches, null drafter, share of the loop's wall | under 12 % | |
| outcome | PASS | |

The c_R lines of a12 are read as the bench prints them (a12's taps add a copy a layer). The oracle:14 rate is a ceiling
for a perfect drafter at 15 tokens a pass, not a served rate.

## What this will not establish

Any real drafter's acceptance or cost. A served rate. The 8B and 14B. Any card but the L40S. Contexts beyond the four
prompts.

## Cost

N11 ran in 21 min billed, $0.63 (*measured*), six modules. This job compiles a12's module (and the guard-free arm the
bench needs) and runs four prompts by five drafters: about 16 min, $0.48 (*estimated*). Timeout 28 min: $0.84 at worst,
under the cap. On l40sx1; if it is not RUNNING after 20 minutes, a twin on l40sx4 (one device visible) is added under a
deviation written beside this file before it is submitted, and the first to run is the measurement; the other is
cancelled by hand once one copy is seen RUNNING or COMPLETED, and the cancellation is confirmed.
