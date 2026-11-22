# Prereg inkling-gsm8k-sealed: GSM8K on 200 problems for Inkling 2 Mini, the sealed file's values through our engine (2026-11-22)

Status: written and signed by the coordinator on 2026-11-22, timestamped by the `.ots` beside it before the run.
It replaces the run of `proofs/preregistration-inkling-gsm8k-card-2026-11-22.md`, which the operator stopped
(that file's `-ECARTS.md`, E1): that arm read the vocabulary tables from the checkpoint.
Cost: about $2.20 (*estimated*, §Cost); hard cap 285 min on `l4x1` at $0.80/h, $3.80. Card spend on this model so
far: $0.12 of the $25 the operator set on 2026-11-22. The operator's go of 2026-11-22 covers this run: same
contract, same price, the object corrected.
Measured code: commit `d4c60cad97cc`, image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` at Space sha
`ee94f304677295370cb0dbe24bc45672e75048ef`, if its build passes; a rebuild at another sha is a deviation written
beside this file before the run. Job script `ops/jobs/inkling-gsm8k-card.sh` with `TABLES=sealed`; runner
`llvq-llm/src/bin/inkgsm.rs` with `--tables`.

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

How many of the first 200 GSM8K test problems does Inkling 2 Mini solve when the engine computes with the sealed
file's values, on a card?

## The object

- The experts of the 24 layers, resident on the card in lattice codes, 2.037 bits a weight.
- The embedding and output tables with the sealed file's int4 g64 values, decoded at load by `RawTensor::to_f32`
  (`llvq-llm/src/inkling/tables.rs`); sha256 of the two table files
  `b7e9979f589a8a845215eeb8fecf5ed5a22e7685072b7bbdbddd45fee0bc027e` (embedding) and
  `cb26ce0c6145d35db52886b378185910b4adaf3de8024cb28e1e6f0062abcbb7` (output).
- Every other tensor as the sealed file's sidecar holds it, which is the checkpoint's, widened to f32.

These are the values of `inkling2-mini.llvq` and its sidecar (sha256 `deedd24d...5519` and `85d64d5c...1217`,
`docs/mesures/inkling2-mini-whole-model-2026-11-22.txt`): the arm `experts24_tables` of that journal, KL 0.0880.
Two precisions, given wherever the figure is given:
- The engine reads the expert code files and the two table files the seal was built from. `inkseal verify` found
  the sealed container equal to them, record for record. The engine does not open the `.llvq` container itself.
- The tables are held decoded, the output table dense in f32 on the card (1.84 GB). The memory of this run is not
  the sealed object's; its values are.

## The contract every arm of the comparison keeps

Unchanged from the first prereg.
- Problems: GSM8K test rows 0 to 199 in parquet order, revision `740312ad`, `ops/data/spec-gsm8k-200.jsonl`, sha256
  `9e778f58f8a9e6c093aad13abac87ff279c967ab3eb39549910eb47b343c6a48`.
- Prompt: one user message, the checkpoint's chat template at reasoning effort `none`; prompt-stream fingerprint
  `5fb41b61271213dd`. Every arm takes these ids.
- Decoding: greedy, batch 1, stop on id 200006, not counted; at most 1,536 new tokens.
- Grading: `llvq_llm::gsm8k`, unchanged, then the cap rule of `inkgsm score`: a completion cut by the cap counts
  wrong.

## Setup

One `l4x1`. Inputs: those of the first card run at the same pins, the tokenizer
(`7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`) and the two tables above.
`inkgsm run --device cuda --require-layers 24 --tables <dir>`, deadline 15,304 s, job timeout 285 min. The dump is
copied to the bucket every 120 s. A run stopped by its deadline keeps every finished problem and is continued by
`RESUME_FROM`; a continuation is the same arm and needs the operator's go for its cost.

## Controls

1. `oracle` MATCH on this backend.
2. Every input at its pinned sha256: the first card run's, the tokenizer, the problems file of the image, the two
   tables.
3. On this image and this card, `inkrun` generates the 32 tokens of the Mac's resident run, 32 of 32
   (`docs/mesures/inkling-first-card-2026-11-22-brut/mac-reference/`), before any problem. The image carries engine
   changes the first card run did not have (the cache written in place, the one-row visit, the chunked prompt),
   gated bit for bit on the CPU and on Metal and never run on CUDA. If the tokens differ, the job stops there.
4. `inkgsm` reports 24 resident layers and no native layer, and the dump's header carries the two tables' sha256.
5. The dump's header carries this protocol (200, 1,536, `none`) and the fingerprint above; 200 rows, in order.
6. `inkgsm score` and `gsm8kpair` both read the dump.

## What gets published

Published (*measured*): the count right of 200 under the cap rule, with its exact 95 % binomial interval; the count
of completions cut by the cap; the mean and the range of generated tokens; the median step seconds; the billed
time. The raw dump goes to the repository.
Not published as a comparison: this count against any figure from another harness, another prompt, another subset
or another card. In particular not against a GSM8K figure of a model card, ours or anyone's.

## The comparison this run prepares, and its rule

Unchanged from the first prereg, fixed before any arm is known. Three more arms are planned on the same contract,
each under its own go and its own controls: unsloth's `UD-IQ2_XXS` and `UD-Q2_K_XL` files, and the original model
through unsloth's `UD-Q8_K_XL` file as its proxy, all in llama.cpp.
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
| control 3 fails | VOID, with a finding: the engine changes do not hold on CUDA; no rerun before the cause is found on the Mac |
| another control fails, or `inkgsm` errors | VOID: the cause is read in the logs; a rerun is the operator's decision |

## Signed prediction

Signed by the coordinator before the run.

- Control 3 passes, 32 of 32: probability 0.8.
- The run reaches 200 rows before its deadline on the first launch, given control 3: probability 0.75.
- Count right of 200: 164 to 186 (82 % to 93 %).
- Completions cut by the cap: at most 10 of 200.
- Mean generated tokens a problem: 220 to 320. Median step: 0.16 to 0.19 s.
- Billed: 2.3 to 3.3 hours.
- If the rival arms are run: ours against `UD-IQ2_XXS` reads HOLDS with probability 0.7.

What the stopped run adds to these, as information and not as a result: its 4 problems read 259 generated tokens
on average at 0.170 to 0.175 s a step, 4 of 4 right, on the other arm.

## Reading table (filled after the run, in the journal, not here)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost (*computed* from the stopped run's step and token counts)

A problem costs about 2 s of prompt reading plus its generated tokens times the step. 200 problems at a mean of 220,
260 or 320 tokens and 0.172 s a step: 2.3, 2.7 or 3.2 hours with the setup, $1.85, $2.15 or $2.60 on `l4x1`.
Control 3 adds about a minute. Timeout 285 min, $3.80.
