# Prereg M3: the verify pass read in place, peak VRAM and tokens, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to `preregistration-spec-mem-m3-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11, for this one job; relayed by the orchestrating session. Cap: **$15**, $8.33 spent before
this job. Registry before this job: **$315.58**. Cost: about **$1.44** on l40sx1 (*estimated*, "Cost"), timeout 75 min,
**$2.25** at worst.

Measured code: the commit that holds this file. Its kernel and runtime sources are ff3bb32's (arm `a12ip`, step M1 of
`docs/mesures/spec-axe-memoire-2026-11-11.txt`). One line was added to `specbench` after ff3bb32: a second
`card memory:` line by cuMemGetInfo after the drafter's load, before the first prompt. The first one, after the verify
pass's install and before the drafter, was already in ff3bb32. The image is published from a clean detached checkout
of the commit that holds this file's stamp.

## Question

Does `a12ip` give the same tokens as `a12` on the card, and how much peak VRAM does it save, at a short context and at
8,192 tokens? What does reading the served order cost the verify pass?

## Setup

One image, one job, `ops/jobs/spec-mem-m3.sh`, on l40sx1 only (no twin, no placeholder job). Output
`spec-mem-m3-2026-11-11/` in `hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs as SP8: the sealed 4B (sha256 886391a8),
`configs/qwen3-4b-tetra-e4-4e-v9.json`, the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (`config.json`
2de97cdf, `model.safetensors` a21d0ffe) in f16, the house set (sha256 f749eb35) and the 200-row GSM8K set (sha256
9e778f58). New input: `ops/data/spec-long-8k-v1.jsonl` (sha256 da4edbd6), one prompt, the opening of WikiText-2 raw test
cut at a word boundary, 8,191 tokens with the chat template and thinking off (counted on the Mac with the Qwen3-4B
tokenizer; the job prints its own count). Thinking off. No dflashcheck step: SP8's f16 gate is not rerun.

Every specbench call is its own process, `--vocab q4 --readout device`, drafter `dflash:tree`, f16, beside a
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100` sampler written to its own file
(`vram-<tag>.csv`, the duel's sampler of `ops/jobs/duel-iq2.sh`); idle and peak go to `vram.txt`. Each step is non-fatal
for the next.

1. Controls: `oracle` on Qwen3-0.6B; the inputs at their sha256; no `LLVQ_*` in the job environment.
2. (a) House set, 44 prompts, 256 tokens: once with `a12@8.10.8.20.6`, once with `a12ip@8.10.8.20.6`.
3. (b) GSM8K-200, 768 tokens, `a12ip@8.10.8.20.6`, `--dump gsm200-ip.jsonl`. The job compares the dump with SP8's
   (`spec-loop-sp8-2026-11-11/gsm200.jsonl` in the bucket, checked by sha256 against
   `docs/mesures/spec-loop-sp8-2026-11-11-brut/gsm200.jsonl`): `cmp`, then per decoder (4e, pass, dflash:tree) the
   count of prompts whose line is identical, and the count of prompts identical on every decoder.
4. (c) The 8,191-token prompt, 256 tokens, `a12ip@8.10.8.20.6`. specbench takes it with no code change: N stays
   256 (cap 1,024), the prompt plus the generated tokens and a 16-row tree stay under the tree pass's 8,704 positions
   (`P4E_MAX_CAP`), the drafter's context grows by doubling, and the house set's RAG prompts already run the same path
   at about 1,500 tokens. It has never run at 8k: a refusal or failure there is a result, not a void.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, drafter files, the three prompt sets and SP8's dump at their sha256; no
   `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62) in every specbench run.
3. If the image fails to build, nothing is launched.

## Signed prediction

Signed by the orchestrating session, verbatim:

- NVRTC compiles a12ip; the load check passes; the loader prints 0 tree-tiled copies for a12ip and 1.308 GB for a12.
- Tokens: a12ip equals SP8's dump on 200 of 200 prompts; self-consistency 44/44 and 200/200. Any difference makes the step INVALID for M1.
- Peak VRAM, short context (house run), nvidia-smi: a12 4.9 GB [4.5, 5.4]; a12ip 3.6 GB [3.2, 4.1]; difference 1.31 GB [1.0, 1.5].
- Speed: reading the served order gives back the tiling's gain. Verify pass 9.8 ms [8.6, 10.6] against 8.53; GSM8K-200 459 tok/s [435, 500] against 502.1; house set all 318 tok/s [300, 343] against 343.1. Accepted a pass unchanged, 5.41.
- 8k, if run: peak 7.4 GB [6.9, 8.2].

Units: GB = 1e9 B; nvidia-smi reports MiB, converted as MiB x 1,048,576 / 1e9.

## Decision rule

| result | outcome |
|---|---|
| a12ip's tokens identical (SP8's dump on 200 of 200, self-consistency 44/44 and 200/200) and the house peak of a12ip at least 1.0 GB under a12's | M1 PASSES; the speed loss is reported with it and the operator decides |
| tokens identical, peak drop under 1.0 GB | M1 FAILS its memory gate |
| any token difference | INVALID for M1; the journal leads with it |

## What gets published, and what does not get compared

Published: the loader's lines (tree-tiled copies, `a12ip` load check, both `card memory:` lines), the sampler files and
peaks, the aggregates and stage splits, the dump comparison.

Not established: a served rate; 8B, 14B; the drafter in int4 (M2 ran on the Mac only); thinking mode.

## Cost

SP8 ran house (tree and chain) and GSM8K-200 plus dflashcheck in 33 min billed, $0.99 (*measured*); a specbench load
took 426 s cold and 244 s warm. This job has four specbench processes (house twice with the tree only, GSM8K once, 8k
once), so four loads: about 48 min, **$1.44** (*estimated*). Timeout 75 min: **$2.25** at worst; $10.58 of $15 at
worst after it. l40sx1 at $1.80/h.
