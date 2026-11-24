# Prereg: the server on a card with the long-prompt settings (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

`llvq serve --local` answered tool calls on the real model and reused its prefix (job `6b0564281460ffe1d5e77db0`),
with a first token after 61.37 s for a prompt of 8,791 tokens and 17.4 GB on the card after it (`nvidia-smi`). The
server now reads a prompt as `inkrun` measured it today: pieces of 512 tokens, scores in chunks of 128 rows, sliding
slots bounded with a slack of 1,024, the bias on the card. Does the served path get what `inkrun` got (23.85 s and
12.0 GB at the peak for 9,000 tokens), and does the prefix reuse survive the bounded slots?

## Object and method

Commit of this prereg. Job `ops/jobs/inkling-serve-card.sh` with `RETRY=levers`, an `l40sx1`, the checks of the
first run. Two caps of 32 tokens that the model spent reasoning are raised to 256 (`b-chat`, `e3-followup`).
A rollback a bounded slot refuses makes the server read the prompt again (`a_refused_rollback_reads_the_prompt_again`
tests it on a scripted model); on the card, an answer under 1,024 tokens must still be reused.

## Decision rule

Every correctness check PASS. The seconds and the memory are reported as read.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| every correctness check PASS | probability 0.55 |
| (e1) seconds to the first token, 8,791 tokens | 21 to 30 |
| (e3) tokens read on the follow-up, prompt minus cached | under 300, probability 0.7 |
| `nvidia-smi` after the long prompt | 11 to 14.5 GB |
| (d1) a call to `get_weather` with `city` Paris | probability 0.85 |

## Cost (*estimated*)

About 6 minutes on an `l40sx1`, $0.18. 30-minute timeout.
