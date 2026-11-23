# Prereg: Inkling 2 Mini on CUDA, the decode step after a long prompt (2026-11-23)

Written before the runs, and before the result of the events run was read. Not edited after its timestamp;
deviations go in `-ECARTS.md` beside it.

## Question

Every CUDA figure so far is 32 tokens after a 32-token prompt. What are the decode step, the time to read the
prompt and the device memory after 600 and 2,000 tokens of prompt?

## Object and method

The object of the stack prereg, arm `act`. Prompts: the 32-token prompt, then the first 600 and 2,000 tokens of
WikiText-2 test window 0. 32 greedy tokens after each, then the 2,000-token arm again with `--phases 12`. Job
`ops/jobs/inkling-cuda-context.sh` on `l40sx1` and `l4x1`. `EVENTS` is `off` if the events run passed its rule,
`on` otherwise; the journal says which.

## Decision rule

No verdict on speed: a measurement. A run that does not end at 2,000 tokens (memory, time) is a result, and the
journal names the cause. The prefill path at this length has never run on CUDA.

## Signed prediction (the coordinator, before the runs)

| quantity | predicted |
|---|---|
| step at 2,000 tokens against the step at 32, L40S | +3 % to +15 % |
| step at 2,000 tokens against the step at 32, L4 | +3 % to +20 % |
| prompt of 2,000 tokens read, L40S | 6 to 20 s |
| device memory at the end, 2,000 tokens | 9.5 to 14 GB |
| every arm ends on both cards | probability 0.7 |

The Mac moved from 0.0433 s at 32 tokens to 0.0450 s at 2,000 (`inkling-debit-boucle-2026-11-22.txt`).

## Cost (*estimated*)

About 9 minutes a card: $0.27 on `l40sx1`, $0.12 on `l4x1`.
