# Prereg SP4: the speculative loop with DFlash on a house set of 44 prompts, up to 256 tokens, sealed 4B

Written on 2026-11-11 at about 01:20 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp4-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-10 evening, for the end-to-end series ("a real test on the card"); the cap raised to $15 with
leave to launch about 21:45 CET; relayed by the orchestrating session. Cap: **$15**, $5.38 spent before this job.
Registry before this job: **$312.63**. Cost: about **$0.72** on l40sx1 (*estimated*, "Cost"), timeout 36 min, **$1.08**
at worst. Measured code: commit `202c2d7` ("specbench runs a prompt file to 1,024 tokens against a chained 4E
reference"; no kernel file changed); the image is published from a clean detached checkout of the commit that holds
this file's stamp, whose kernel, bench and runtime sources are 202c2d7's (only the job script, the CUDA image's runtime
COPY and `UPLOAD_ALLOW`, which now ship `ops/data/spec-prompts-v1.jsonl`, and the proofs moved after it).

## Question

SP3 (`docs/mesures/spec-loop-sp3-2026-11-10.txt`) put DFlash above 4E's greedy decode on four house prompts of 64
tokens, with token equality on all of them. Does equality hold at scale, over every generated token of 44 prompts of up
to 256 tokens, and what are the acceptance and the speed per kind of prompt? This is a validity test: the hard gate is
equality on all 88 runs.

## Setup

One image, one job, `ops/jobs/spec-loop-sp4.sh`, on l40sx1 only (no twin, whatever the queue). Inputs as SP3: the
fixture from the jobs bucket at sha256 636f664d; the drafter `z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (MIT),
`config.json` at sha256 2de97cdf and `model.safetensors` at sha256 a21d0ffe; and the house prompt set
`ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35), shipped in the image and checked at its sha256 in the job: 20 GSM8K
test rows, 10 code, 10 chat, 4 long-context (rag), through the same chat template, thinking off. Controls first, then
each step to its own file, non-fatal for the next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16`, 4 min; if it fails, (b) runs the drafter in f32 and the
   summary says so.
3. (b) `specbench --prompts <the set> <sealed> a12@8.10.8.20.6 dflash:chain,dflash:tree 256 <drafter> f16`, 24 min, its
   output flushed as it goes. Up to 256 tokens a prompt, stopping at `<|im_end|>`; 4E's greedy reference chained in
   launches of 64; one timed run a prompt. A control line on the first prompt compares one 64-token 4E launch with two
   chained 32-token launches. Per prompt and drafter: equality over every generated token (a mismatch prints its
   position, both tokens and the pass's logits for them), accepted tokens a pass, passes, the walls, tok/s against 4E's
   greedy tok/s; per drafter, aggregates per kind and overall.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture, drafter files and prompt set at their sha256; no `LLVQ_*` in the
   job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. If the image fails to build, nothing is launched; the build error goes back to the orchestrating session with its
   text.
4. The control line: one 64-token 4E launch equals two chained 32-token launches.

## What gets published, and what does not get compared

Published: the control line; the equality count and any mismatch in full; the aggregates per kind and overall for both
drafters (tokens, accepted a pass with its histogram, tok/s, greedy tok/s, x, median x over prompts with its range,
verify and drafter shares); the five slowest and five fastest prompts by x. Not established: a served rate; a standard
benchmark (this is a house set); 8B, 14B; contexts past about 2,000 tokens; thinking mode (off here).

## Decision rule

| result | outcome |
|---|---|
| equality holds on all 88 runs (44 prompts, 2 drafters) | VALID: the aggregates are read against their predictions below |
| any mismatch | NOT VALID: the journal leads with it (prompt, position, tokens, margin); the series stops until it is explained |

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at the plan [8, 10, 8, 20, 6], the drafter in f16, up to 256 tokens a prompt,
one timed run a prompt. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| control line: one 64-token 4E launch equals two chained 32-token launches | equal | |
| token equality with 4E's greedy over every generated token (hard gate) | 44 of 44, chain and tree | |
| aggregate drafted tokens accepted a pass, tree, math | 5.0 | [3.5, 6.8] |
| same, code | 4.0 | [2.8, 5.5] |
| same, chat | 2.4 | [1.5, 3.5] |
| same, rag | 5.0 | [2.5, 8.0] |
| aggregate tok/s over 4E's greedy (total tokens over total wall, both sides), tree, math | x1.90 | [1.50, 2.40] |
| same, code | x1.70 | [1.30, 2.20] |
| same, chat | x1.15 | [0.90, 1.50] |
| same, rag (prompts of 1,200 to 1,600 tokens; every verify row reads the whole cache, the pass's cost there unknown) | x1.30 | [0.70, 2.20] |
| same, all prompts | x1.60 | [1.25, 2.10] |
| tree against chain, aggregates | tree at least the chain on math, code and chat; nothing signed for rag | |
| the drafter's share of the wall | | [34 %, 42 %] |

## What this will not establish

A served rate; a standard benchmark (this is a house set); 8B, 14B; contexts past about 2,000 tokens; thinking mode.

## Cost

SP3's rerun ran in 8 min billed, $0.25 (*measured*), four prompts of 64 tokens. This job runs 44 prompts of up to 256
tokens through two drafters and the chained reference: about 24 min, $0.72 (*estimated*). Timeout 36 min: $1.08 at
worst, under the cap. On l40sx1 only: no twin, however long the queue.
