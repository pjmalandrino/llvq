# Prereg inkling-gsm8k-card: GSM8K on 200 problems for Inkling 2 Mini, our sealed code through our engine (2026-11-22)

Status: written and signed by the coordinator on 2026-11-22, timestamped by the `.ots` beside it before the run.
Cost: about $2.40 (*estimated*, §Cost); hard cap 285 min on `l4x1` at $0.80/h, $3.80. Card spend on this model so
far: $0.04 of the $25 the operator set on 2026-11-22. The operator's go covers this arm only.
Measured code: commit `f05dc6eb934129692b61e54c44139e2e38f7ced2`, the image of the first card run
(`proofs/preregistration-inkling-first-card-2026-11-22.md`, verdict PASSES). Job script
`ops/jobs/inkling-gsm8k-card.sh`; runner `llvq-llm/src/bin/inkgsm.rs`.

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

How many of the first 200 GSM8K test problems does Inkling 2 Mini solve when its 24 layers of experts are read from
our lattice codes, through this engine, on a card?
The whole-model reading of 2026-11-22 (`docs/mesures/inkling2-mini-whole-model-2026-11-22.txt`) measures fidelity
to the original on one text. It does not measure a task. This run is the first task reading of the object.

## What this arm is, and is not

The experts of the 24 layers are resident in lattice codes, 2.037 bits a weight. Every other tensor is the
original's, widened to f32: the attention, the router, the norms, and the two tables. The sealed file also holds the
two tables in int4 g64; the engine does not load that encoding yet. So this arm is the `experts24` arm of the
whole-model journal (KL 0.0834), not `experts24_tables` (KL 0.0880). The report says so wherever the figure is
given, and the figure is not presented as the sealed file's.

## The contract every arm of the comparison keeps

- Problems: GSM8K test rows 0 to 199 in parquet order, revision `740312ad`, `ops/data/spec-gsm8k-200.jsonl`, sha256
  `9e778f58f8a9e6c093aad13abac87ff279c967ab3eb39549910eb47b343c6a48`.
- Prompt: one user message, the checkpoint's chat template at reasoning effort `none`. The 200 rendered prompts are
  equal id for id to those of `transformers` 046bdb8cc2 (`llvq-llm/tests/inkgsm.rs`); prompt-stream fingerprint
  `5fb41b61271213dd`. Every arm takes these ids.
- Decoding: greedy, batch 1, stop on id 200006, not counted; at most 1,536 new tokens.
- Grading: `llvq_llm::gsm8k`, unchanged, then the cap rule of `inkgsm score`: a completion cut by the cap counts
  wrong.

## Setup

One `l4x1`, image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` at Space sha
`b959fa385a7713baba9befc7571e6454d4deb8cc`. Inputs: those of the first card run, at the same pins, plus the
checkpoint's `tokenizer.json`, sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`.
`inkgsm run --device cuda --require-layers 24`, deadline 15,304 s, job timeout 285 min. The dump is copied to the
bucket every 120 s. A run stopped by its deadline keeps every finished problem and is continued by `RESUME_FROM`;
a continuation is the same arm and needs the operator's go for its cost.

## Controls

1. `oracle` MATCH on this backend.
2. Every input at its pinned sha256, the problems file of the image included.
3. `inkgsm` reports 24 resident layers and no native layer.
4. The dump's header carries this protocol (200, 1,536, `none`) and the fingerprint above; 200 rows, in order.
5. `inkgsm score` and `gsm8kpair` both read the dump.

## What gets published

Published (*measured*): the count right of 200 under the cap rule, with its exact 95 % binomial interval; the count
of completions cut by the cap; the mean and the range of generated tokens; the median step seconds; the billed
time. The raw dump goes to the repository.
Not published as a comparison: this count against any figure from another harness, another prompt, another subset
or another card. In particular not against a GSM8K figure of a model card, ours or anyone's.

## The comparison this run prepares, and its rule

Three more arms are planned on the same contract, each under its own go and its own controls: unsloth's
`UD-IQ2_XXS` and `UD-Q2_K_XL` files, and the original model through unsloth's `UD-Q8_K_XL` file as its proxy, all
in llama.cpp. If they are run, the rule is fixed here, before any arm is known:

- Each pair is read by problem, with the exact two-sided McNemar test. At 200 problems a pair separates only a gap
  of about 7 points or more; a smaller gap is reported as not separated, never as a tie or a win.
- Ours against `UD-IQ2_XXS`, the file of the nearest size: HOLDS if ours is higher, or lower with p at least 0.05;
  BEHIND if ours is lower with p under 0.05.
- Ours against `UD-Q2_K_XL` and against the proxy of the original: the gap and its p are reported; no verdict.
- Each such pair crosses two engines, ours and llama.cpp, and the report says it beside the figure.
If the other arms are not run, this run's count stands alone and nothing is concluded about rivals.

## Decision rule for this run

| reading | verdict |
|---|---|
| controls hold; 200 rows | READ: the count is published as above |
| controls hold; the deadline stops the run short of 200 rows | PARTIAL: the rows are kept, nothing is published as a count of 200; a continuation is the operator's decision |
| a control fails, or `inkgsm` errors | VOID: the cause is read in the logs; a rerun is the operator's decision |

## Signed prediction

Signed by the coordinator before the run.

- The run reaches 200 rows before its deadline on the first launch: probability 0.65.
- Count right of 200: 164 to 186 (82 % to 93 %). The two problems run on the Mac were both right.
- Completions cut by the cap: at most 10 of 200.
- Mean generated tokens a problem: 200 to 350. Median step: 0.16 to 0.20 s.
- Billed: 2.0 to 3.5 hours.
- If the rival arms are run: ours against `UD-IQ2_XXS` reads HOLDS with probability 0.7.

## Reading table (filled after the run, in the journal, not here)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost (*computed* from the first card run's step, 0.1656 s)

A problem costs about 2 s of prompt reading plus its generated tokens times the step. 200 problems at a mean of 200,
300 or 450 tokens: 2.0, 3.0 or 4.3 hours, $1.63, $2.36 or $3.47 on `l4x1`. The step grows with the context, about
9 % over the range these problems reach (*measured* on the Mac), which the figures above do not include. Timeout
285 min, $3.80.
