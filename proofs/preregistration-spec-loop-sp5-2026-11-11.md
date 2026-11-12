# Prereg SP5: the lighter drafter, its int4 vocabulary, and the self-consistency gate, sealed 4B

Written on 2026-11-11 at about 02:30 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp5-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-10 evening ("a real test on the card"); cap $15 with leave to launch from about 21:45 CET;
relayed by the orchestrating session. Cap: **$15**, $5.70 spent before this job. Registry before this job: **$312.95**.
Cost: about **$1.20** on l40sx1 (*estimated*, "Cost"), timeout 54 min, **$1.62** at worst. Measured code: commit
`f838b8b` ("specbench gates on the pass's own greedy and reports agreement with 4E"), which holds the merged lighter
drafter round (c617bc9, about half the device operations a round) and the int4 vocabulary for the drafter's embedding and
head through the target's own table and existing kernels (080088e, `--vocab q4`; the default stays dense); the image is
published from a clean detached checkout of the commit that holds this file's stamp, whose kernel, bench and runtime
sources are f838b8b's (only the job script and the proofs moved after it).

The gate, restated after SP4 (SP4's ECARTS, deviation 1): equality with 4E's greedy tokens is reported, not gated; the
hard gate is self-consistency, every drafter's tokens equal to the loop's with the null drafter, the pass's own R = 1
greedy decode.

## Question

With the lighter drafter round and, separately, the int4 vocabulary, what does the loop give on the 44 house prompts,
and does it stay self-consistent? Where does it agree with 4E?

## Setup

One image, one job, `ops/jobs/spec-loop-sp5.sh`, on l40sx1 only (no twin). Inputs and controls as SP4: the fixture
(sha256 636f664d), the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (`config.json` 2de97cdf,
`model.safetensors` a21d0ffe), the prompt set `ops/data/spec-prompts-v1.jsonl` in the image (sha256 f749eb35: 20 GSM8K
test rows, 10 code, 10 chat, 4 long-context; thinking off). Each step to its own files, non-fatal for the next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16`, 4 min, with its loop-like timing lines (extension of n = 1,
   4, 8, then one draft, one sync). If it fails, (b) and (c) run the drafter in f32 and the summary says so.
3. (b) `specbench --prompts <set> --vocab dense --dump <O>/dense.jsonl <sealed> a12@8.10.8.20.6 null,dflash:tree 256
   <drafter> f16`, 22 min.
4. (c) `specbench --prompts <set> --vocab q4 --dump <O>/q4.jsonl <sealed> a12@8.10.8.20.6 dflash:tree 256 <drafter>
   f16`, 22 min.

Each specbench run: up to 256 tokens a prompt, stopping at `<|im_end|>`; the null drafter as the pass's own greedy
reference; the self-consistency gate; the agreement report against 4E's greedy (chained launches of 64); the dumps of
every decoder's token ids and texts.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture, drafter files and prompt set at their sha256; no `LLVQ_*` in the
   job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), in both specbench runs.
3. If the image fails to build, nothing is launched; the build error goes back to the orchestrating session with its
   text.

## What gets published, and what does not get compared

Published: dflashcheck's timing lines; the self-consistency counts and any failure in full; the 4E agreement report; the
aggregates per kind and overall for null, tree dense and tree q4 (tokens, accepted a pass, tok/s, greedy, x, per-prompt
median and range, verify and drafter shares, the drafter's stage split a round); VRAM as each process reports it, if it
does; both dumps. Not established: the quality of the pass's text against 4E's beyond the agreement counts (no grading
here; the dumps allow it afterwards); a served rate; a standard benchmark; 8B, 14B; thinking mode.

## Decision rule

| result | outcome |
|---|---|
| the self-consistency gate holds on every run (null, tree dense, tree q4; 44 of 44 each) | VALID: the aggregates are read against their predictions |
| any self-consistency failure | NOT VALID: the journal leads with it; the series stops until it is explained |

The agreement with 4E is reported and read against its prediction; it does not gate.

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at [8, 10, 8, 20, 6], the drafter in f16 with `fc` in f32, the tree of 15, 44
house prompts, up to 256 tokens. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| self-consistency (hard gate): timed null, tree dense, tree q4 | 44 of 44 each | |
| agreement with 4E | 40 of 44, first divergences at gsm8k-test-0004 token 251, chat-02 token 84, chat-03 token 107, chat-07 token 71, in both runs | |
| one drafter round as the loop pays it (drafter wall over passes), dense | 3.4 ms | [2.8, 4.3] |
| same, q4 | 2.7 ms | [2.1, 3.6] |
| the head stage a round, dense | 1.1 ms | [0.9, 1.3] |
| same, q4 | 0.45 ms | [0.20, 0.80] |
| drafted tokens accepted a pass, all prompts, dense | 3.39 | [3.20, 3.55] |
| same, q4 | 3.35 | [3.05, 3.55] |
| total tokens over total wall against 4E's greedy, null | x0.62 | [0.58, 0.66] |
| tree dense, all prompts | x1.63 | [1.50, 1.76] |
| tree q4, all prompts | x1.72 | [1.55, 1.90] |
| tree q4, math | x2.30 | [2.05, 2.55] |
| tree q4, code | x1.68 | [1.48, 1.88] |
| tree q4, chat | x1.17 | [1.04, 1.30] |
| tree q4, rag | x1.30 | [1.12, 1.52] |
| VRAM, drafter side, q4 against dense | about 0.78 GB less (as each process reports it, if it does) | |

## Cost

SP4 ran in 11 min billed, $0.32 (*measured*), one specbench run of two drafters. This job runs two specbench processes
(three decoders and the reference each time): about 40 min, $1.20 (*estimated*). Timeout 54 min: $1.62 at worst, under
the cap. On l40sx1 only: no twin.
