# Prereg: Inkling 2 Mini on CUDA, the attention bias on the card and the experts' sum in one launch (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

After the batched read, 2,000 tokens take 6.87 s and 9,000 take 64.12 s on an L40S (*measured*); the scores of a
prompt hold 1.79 s of a profiled 9.03 s at 2,000 tokens and grow with the square of the length. What do three
changes meant to keep every bit bring, at 2,000 and at 9,000 tokens: the bias of the scores written on the card
(`inkrun --attn-bias device`), a prompt's routing rows sorted in parallel, and the experts' batched path in blocks of
1,024 tokens with each token's weighted sum in one launch (`moe_wsum`)?

## Object and method

Commit of this prereg. Job `ops/jobs/inkling-cuda-ttft.sh` with
`ARMS="chunk batch8 d2000 phd2000 t9000 d9000 phd9000" RETRY=bias`, an `l40sx1`. `chunk`: the one-expert path, the
reference of 2,000 tokens. `batch8`: the batched read with the blocks and `moe_wsum`, the bias on the host. `d2000`:
the same with the bias on the card; `phd2000` with the parts. `t9000`, `d9000`, `phd9000`: the same three at 9,000
tokens. The parallel sort is in every arm.

## Decision rule

IDENTITY: at 2,000 tokens every arm's logits have the sha256 `9c1624ad...b014e`; at 9,000 tokens, `a38aca4e...950c`,
the bytes of job `6b055b55ca8b51043c3b6713`. An arm that differs is a fault to find, not a speed result.
The seconds and the GB are reported as read. No threshold.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| the two new kernels compile and every arm ends | probability 0.65 |
| every arm identical, given it ran | probability 0.75 |
| `batch8`, seconds of the read | 6.2 to 7.0 |
| `d2000` | 5.4 to 6.6 |
| `t9000` | 54 to 64 |
| `d9000` | 38 to 56 |
| `phd9000`, share of the attention scores | 25 to 50 % |
| `t9000`, GB at the end | 14 to 19 (20.506 before the blocks) |

## Cost (*estimated*)

About 16 minutes on an `l40sx1`, $0.48. 30-minute timeout.
