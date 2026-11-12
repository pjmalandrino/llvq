# Prereg SP6: the drafter's readout on the device, and 200 GSM8K problems graded, sealed 4B

Written on 2026-11-11 at about 03:40 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp6-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-10 evening ("a real test on the card"); cap $15 with leave to launch from about 21:45 CET;
relayed by the orchestrating session. Cap: **$15**, $6.32 spent before this job. Registry before this job: **$313.57**.
Cost: about **$1.80** on l40sx1 (*estimated*, "Cost"), timeout 84 min, **$2.52** at worst. Measured code: commit
`7429fc7` ("specbench --readout device|chunked"), which merges the drafter's readout on the device (077aa5b: 16
launches and one small transfer at the end of a round; the previous readout behind `--readout chunked`) and adds
`ops/data/spec-gsm8k-200.jsonl` (GSM8K test rows 0 to 199 with gold, sha256 9e778f58) and the offline grader
`ops/spec_grade.py`. The image is published from a clean detached checkout of the commit that holds this file's stamp,
whose kernel, bench and runtime sources are 7429fc7's (only the image's file lists, the job script and the proofs moved
after it).

The gate is SP5's: self-consistency, every drafter's tokens equal to the loop's own R = 1 greedy. Agreement with 4E is
reported, not gated.

## Question

With the drafter's readout on the device, what does one drafter round cost in the loop, and what does the tree give on
the 44 house prompts? On 200 GSM8K problems at up to 768 tokens, does the loop stay self-consistent, how often does it
agree with 4E, and does the graded accuracy of the pass differ from 4E's?

## Setup

One image, one job, `ops/jobs/spec-loop-sp6.sh`, on l40sx1 only (no twin). Inputs and controls as SP5: the fixture
(sha256 636f664d), the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (`config.json` 2de97cdf,
`model.safetensors` a21d0ffe), the house set `ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35), and the 200-row set
`ops/data/spec-gsm8k-200.jsonl` (sha256 9e778f58), both in the image. Thinking off. Each step to its own files,
non-fatal for the next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16`, 6 min: both readouts alone, the loop rounds with each, the
   extension's upload share, the launch counts. If it fails, (b) and (c) run the drafter in f32 and the summary says so.
3. (b) `specbench --prompts <house 44> --vocab q4 --readout device <sealed> a12@8.10.8.20.6 dflash:tree 256 <drafter>
   f16`, 22 min.
4. (c) `specbench --prompts <gsm8k 200> --vocab q4 --readout device --dump <O>/gsm200.jsonl <sealed> a12@8.10.8.20.6
   dflash:tree 768 <drafter> f16`, 50 min.

After the job, on the Mac, no model loaded: `uv run ops/spec_grade.py <O>/gsm200.jsonl --prompts
ops/data/spec-gsm8k-200.jsonl --pair 4e pass`, and again with `--pair pass dflash:tree`; the output kept with the raw
files.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture, drafter files and both prompt sets at their sha256; no `LLVQ_*` in
   the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), in both specbench runs.
3. If the image fails to build, nothing is launched; the build error goes back to the orchestrating session with its
   text.

## What gets published, and what does not get compared

Published: dflashcheck's timing lines in full; (b)'s gate and aggregates with the drafter's stage split a round; (c)'s
gate, agreement counts and aggregates; the grader's full output; the dump. Not established: quality beyond these 200
problems and this instruction (thinking off, 768-token cap); a served rate; 8B, 14B.

## Decision rule

| result | outcome |
|---|---|
| self-consistency holds on (b) and (c) (44 of 44, 200 of 200) | VALID: the aggregates are read against their predictions |
| any self-consistency failure | NOT VALID: the journal leads with it; the series stops until it is explained |

The quality line reads "no measured difference" if the exact McNemar p is above 0.05, and states the discordant counts
either way.

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at [8, 10, 8, 20, 6], the drafter in f16 with `fc` in f32, the tree of 15, the
int4 vocabulary. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| (a) readout alone, device | 0.35 ms | [0.15, 0.90] |
| (a) readout alone, chunked | 1.95 ms | [1.70, 2.20] |
| (a) loop round at n = 4, device readout | 4.2 ms | [3.5, 5.0] (5.4 to 6.0 with chunked, as SP5) |
| (b) self-consistency | 44 of 44 | |
| (b) drafted tokens accepted a pass | 3.39 | [3.30, 3.48] |
| (b) one drafter round as the loop pays it | 3.9 ms | [3.2, 4.8] |
| (b) total tokens over total wall against 4E's greedy, all | x1.60 | [1.47, 1.72] |
| (b) same, math | x2.15 | [1.95, 2.35] |
| (b) same, code | x1.55 | [1.40, 1.70] |
| (b) same, chat | x1.09 | [0.98, 1.20] |
| (b) same, rag | x1.25 | [1.12, 1.40] |
| (c) self-consistency | 200 of 200 | |
| (c) token lists identical to 4E's | 172 of 200 | [140, 190] (about one near-tie flip in 2,650 tokens) |
| (c) tree against 4E's greedy on these prompts | x2.10 | [1.90, 2.35] |
| (c) 4E correct (grader, `--pair 4e pass`) | 165 of 200 | [148, 180] |
| (c) the pass correct | within 6 of 4E's count | |
| (c) discordant pairs | at most 14 | |
| (c) exact McNemar p | above 0.05 | |
| (c) the tree's answers equal the pass's | 200 of 200 (its tokens do) | |

## Cost

SP5 ran in 21 min billed, $0.62 (*measured*). This job: dflashcheck, one house run of one drafter, one 200-prompt run
at up to 768 tokens: about 60 min, $1.80 (*estimated*). Timeout 84 min: $2.52 at worst, under the cap. On l40sx1 only:
no twin.
