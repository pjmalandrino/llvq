# Prereg SP8: the drafter's readout without the softmax, sealed 4B

Written on 2026-11-11 at about 05:35 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp8-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-10 evening ("a real test on the card"); cap $15 with leave to launch from about 21:45 CET;
relayed by the orchestrating session. Cap: **$15**, $7.34 spent before this job. Registry before this job: **$314.59**.
Cost: about **$1.86** on l40sx1 (*estimated*, "Cost"), timeout 86 min, **$2.58** at worst. Measured code: merge commit
`1d9743e`, which brings in 2fe5691 ("DFlash readout without the softmax"). That commit makes `dflash::LseReadout` the
drafter's default readout:
- for a tree: 4 argmax with in-place masks, one gather, and a logsumexp from sub, exp and sum, for 15 launches and one
  transfer;
- for a chain: one argmax.
The softmax and chunked readouts stay behind switches. `profile` and plain `dflashcheck` print the new stages. The image
is published from a clean detached checkout of the commit that holds this file's stamp, whose kernel, bench and runtime
sources are 1d9743e's (only the job script and the proofs moved after it).

Naming caveat: `specbench --readout device` now selects this new default. Its config line may still print "readout
device".

The gate is SP5's: self-consistency, every drafter's tokens equal to the loop's own R = 1 greedy. Agreement with 4E is
reported, not gated.

## Question

Without the full-vocabulary softmax, which SP7 measured at 1.485 of the 2.0 ms readout, what does the drafter round
cost in the loop? What do the tree and the chain give on the 44 house prompts, and the tree on 200 GSM8K problems?

## Setup

One image, one job, `ops/jobs/spec-loop-sp8.sh`, on l40sx1 only (no twin). Inputs and controls as SP6: the fixture
(sha256 636f664d), the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (`config.json` 2de97cdf,
`model.safetensors` a21d0ffe), the house set (sha256 f749eb35) and the 200-row GSM8K set (sha256 9e778f58), both in the
image. Thinking off. Each step to its own files, non-fatal for the next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16`, 6 min: the four readouts alone and the loop rounds with
   each. If it fails, (b) and (c) run the drafter in f32 and the summary says so.
3. (b) `specbench --prompts <house 44> --vocab q4 --readout device <sealed> a12@8.10.8.20.6 dflash:tree,dflash:chain
   256 <drafter> f16`, 24 min.
4. (c) `specbench --prompts <gsm8k 200> --vocab q4 --readout device --dump <O>/gsm200.jsonl <sealed> a12@8.10.8.20.6
   dflash:tree 768 <drafter> f16`, 50 min.

After the job, on the Mac, no model loaded: `uv run ops/spec_grade.py <O>/gsm200.jsonl --prompts
ops/data/spec-gsm8k-200.jsonl --pair 4e pass`, and again with `--pair pass dflash:tree`.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture, drafter files and both prompt sets at their sha256; no `LLVQ_*` in
   the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), in both specbench runs.
3. If the image fails to build, nothing is launched; the build error goes back to the orchestrating session with its
   text.

## What gets published, and what does not get compared

Published:
- (a)'s lines;
- for (b) and (c): the gates, agreement counts, aggregates and the drafter's stage split a round;
- the grader's output and the dump.

Not established: a served rate; a standard benchmark for chat and code; 8B, 14B; thinking mode; long context.

## Decision rule

| result | outcome |
|---|---|
| self-consistency holds on (b) (tree and chain, 44 of 44 each) and (c) (200 of 200) | VALID: the aggregates are read against their predictions |
| any self-consistency failure | NOT VALID: the journal leads with it; the series stops until it is explained |

## Signed prediction

Signed by the orchestrating session. Conditions: sealed 4B, one L40S, the v9 config, a12 at [8, 10, 8, 20, 6], the
drafter in f16 with `fc` in f32, the int4 vocabulary.

These values come from SP7's measured lines (`docs/mesures/spec-loop-sp7-2026-11-11.txt`), not from operation counts.

| quantity | point | interval |
|---|---|---|
| (a) readout alone, no-softmax tree | 0.62 ms | [0.50, 0.85] |
| (a) readout alone, no-softmax chain | 0.11 ms | [0.09, 0.20] |
| (a) readout alone, softmax | 2.0 ms | [1.9, 2.1] |
| (a) readout alone, chunked | 1.9 ms | [1.8, 2.0] |
| (a) loop round, tree, n = 4, one synchronisation | 4.05 ms | [3.75, 4.40] |
| (b) self-consistency, tree and chain | 44 of 44 each | |
| (b) accepted a pass, tree | 3.39 | [3.32, 3.46] |
| (b) accepted a pass, chain | 2.57 | [2.48, 2.66] |
| (b) drafter round in the loop, tree | 4.15 ms | [3.85, 4.50] |
| (b) drafter round in the loop, chain | 3.60 ms | [3.30, 3.95] |
| (b) tree, total tokens over total wall against 4E's greedy, all | x1.51 | [1.45, 1.58] |
| (b) tree, math | x2.05 | [1.97, 2.15] |
| (b) tree, code | x1.45 | [1.38, 1.52] |
| (b) tree, chat | x1.04 | [0.98, 1.10] |
| (b) tree, rag | x1.22 | [1.16, 1.30] |
| (b) chain, all | x1.28 | [1.20, 1.36] |
| (c) self-consistency | 200 of 200 | |
| (c) accepted a pass | 5.41 | [5.33, 5.49] |
| (c) tree tok/s | 494 | [472, 516] |
| (c) tree against 4E's greedy | x2.26 | [2.16, 2.36] |
| (c) token lists identical to 4E's | 177 of 200 (the pass decides alone) | |
| (c) graded: the tree correct | 167 of 200 | |
| (c) graded: 4E correct | 166 of 200 | |
| (c) differing answers | the same six as SP6 | |

## Cost

SP6 ran in 32 min billed, $0.96 (*measured*). This job adds the chain to the house run: about 62 min, $1.86
(*estimated*). Timeout 86 min: $2.58 at worst, under the cap. On l40sx1 only: no twin.
