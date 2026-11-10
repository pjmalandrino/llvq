# Prereg SP3: the speculative loop with DFlash after the f16 fix, sealed 4B

Written on 2026-11-10 at about 23:45 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-loop-sp3-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-10 evening, for the end-to-end series ("a real test on the card"), and about 21:45 CET the cap
raised to $15 with leave to launch, relayed by the orchestrating session. Cap: **$15**, $4.89 spent before this job.
Registry before this job: **$312.14**. Cost: about **$0.66** on l40sx1 (*estimated*, "Cost"), timeout 32 min, **$0.96**
at worst. Measured code: commit `ce8fb7f` ("DFlash: fc and hidden_norm in f32; the f16 drafter overflowed and drafted NaN
(SP2)"); the image is published from a clean detached checkout of the commit that holds this file's stamp, whose kernel,
bench and runtime sources are ce8fb7f's (only the job script and the proofs moved after it).

## Question

SP2 (`docs/mesures/spec-loop-sp2-2026-11-10.txt`) found the loop sound and DFlash's proposals almost never accepted. The
cause, found at $0: the drafter's first layer `fc` overflowed binary16 (the layer-34 taps reach 15,352 and `fc` sums
12,800 of them), `hidden_norm` turned those rows into NaN, and the f16 drafter proposed token 151551 fifteen times a round;
`dflashcheck` checked only f32. ce8fb7f keeps `fc` and `hidden_norm` in f32 whatever the drafter's dtype, and
`dflashcheck` now gates the timed dtype; a new host test (`tests/dflash_prefill.rs`) gives 15 of 15 against the fixture in
f32 and f16 from the real `prefill_taps` taps. With the fix, what does DFlash accept and what does the loop give end to
end?

## Setup

One image, one job, `ops/jobs/spec-loop-sp3.sh`, on l40sx1 only (no twin, whatever the queue). Inputs as SP2: the fixture
`llvq-dflash-fixture-2026-11-10.safetensors` from the jobs bucket at sha256 636f664d; the drafter
`z-lab/Qwen3-4B-DFlash-b16` at revision b74e3a32 (MIT) fetched into a job-local directory, `config.json` at sha256
2de97cdf and `model.safetensors` at sha256 a21d0ffe. Controls first, then each step to its own file, non-fatal for the
next:

1. `oracle` on Qwen3-0.6B; the sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
2. (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16`, 4 min: the equivalence in f32, then the new gate on the timed
   dtype (logits within 5e-2 relative, at least 3 of 6 chains equal), then the timed rounds. If it fails, (b) runs the
   drafter in f32 and the summary says so.
3. (b) `specbench <sealed> a12@8.10.8.20.6 null,oracle:3,dflash:chain,dflash:tree 128 <drafter> f16`, 20 min: on four
   prompts (chat, rag, code, math), 128 tokens each, 1 discarded and 3 timed runs.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, config, fixture and drafter files at their sha256; no `LLVQ_*` in the job
   environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. If the image fails to build, nothing is launched.

Token equality with 4E's greedy tokens, 128 of 128 on every prompt for every drafter, is the hard gate: anything else voids
the speed lines of that drafter and stops the series until it is explained.

## What gets published, and what does not get compared

Published: dflashcheck's lines; the equality lines; per prompt and drafter the drafted tokens accepted a pass, passes, the
verify, drafter (split as the bench prints it) and other walls, tok/s end to end and without the drafter, 4E's greedy
tok/s and the ratio. Not established: a served rate; acceptance on standard benchmarks (four house prompts of 128
tokens); 8B, 14B; long context; whether the pass's own taps equal the prefill's beyond what the acceptance shows.

## Decision rule

| result | outcome |
|---|---|
| equality holds everywhere | the DFlash lines are read against their predictions below |
| any equality fails | the journal names it; the series stops until it is explained |

## Signed prediction

Sealed 4B, one L40S, the v9 config, a12 at the plan [8, 10, 8, 20, 6], 128 tokens a prompt, the drafter in f16 with `fc`
in f32. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| `dflashcheck` f16: passes its new gate (logits within 5e-2 relative, at least 3 of 6 chains equal) | 6 of 6 chains | |
| token equality with 4E's greedy, every prompt, every drafter (hard gate) | 128 of 128 | |
| skipped rounds | 0 | |
| rounds where the drafter proposes the same token 15 times (no NaN) | at most 5 % | |
| drafted tokens accepted a pass, chain, chat | 2.2 | [1.2, 3.4] |
| same, code | 4.0 | [2.3, 6.2] |
| same, math | 4.8 | [2.8, 7.2] |
| same, rag | 3.0 | [1.4, 5.2] |
| tree, drafted tokens accepted a pass | at least the chain's minus 0.1, every prompt | |
| one drafter round | 4.8 ms | [4.2, 6.2] |
| end to end over 4E's greedy, chain, chat | x1.10 | [0.78, 1.50] |
| same, code | x1.70 | [1.10, 2.45] |
| same, math | x1.95 | [1.25, 2.75] |
| same, rag | x1.30 | [0.80, 2.00] |
| outcome | code and math above greedy; chat near greedy, either side | |

## What this will not establish

A served rate; acceptance on standard benchmarks; 8B, 14B; long context; whether the pass's own taps equal the prefill's
beyond what the acceptance shows.

## Cost

SP2 ran in 18 min billed, $0.55 (*measured*). This job fetches the drafter, runs one dflashcheck and one specbench of four
drafters over four prompts of 128 tokens: about 22 min, $0.66 (*estimated*). Timeout 32 min: $0.96 at worst, under the
cap. On l40sx1 only: no twin, however long the queue.
