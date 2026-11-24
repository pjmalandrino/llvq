# Prereg: Inkling 2 Mini on CUDA, three levers on the memory of a long prompt (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

A 9,000-token prompt peaks at 19.6 GB on an L40S and 19.0 GB on an L4, 20.2 to 20.5 GB at the end of a run, for
8.1 to 8.4 GB after the load (*measured*, jobs `6b055b55ca8b51043c3b6713` and `6b055c5fca8b51043c3b68fa`). What do
these bring, alone and together: the prompt read in pieces of 512 tokens (`--prefill-chunk 512`), the sliding slots
bounded to their window plus a slack (`--kv-window bounded`), and scores in chunks of 128 query rows
(`--attn-chunk 128`)? The exact reservation of the cache and the experts' blocks of 1,024 tokens are in every arm.

## Object and method

Commit of this prereg. Job `ops/jobs/inkling-cuda-ttft.sh` with
`ARMS="r2000 c2000 k2000 r9000 c9000 k9000 a9000 ta9000" RETRY=levers`, an `l40sx1`, an `l4x1` if none comes.
Every arm: the batched read at 8 rows, the bias on the card, the whole fused stack. `r`: the reference. `c`: pieces of
512. `k`: bounded slots. `a9000`: the three together, with the probe; `ta9000`: the three, without the probe, for the
seconds. All but `ta9000` carry the peak probe.

## Decision rule

`k` against `r`: the logits bytes are expected equal (one call reads the same keys, and the one-row kernel reads the
same visible keys); if they differ, the largest gap is reported and the lever owes the perplexity band.
`c` and `a`: not expected equal; the 32 tokens must be the reference's and the largest logit gap is reported. Equal
tokens here are not a quality gate: a lever that changes values owes the +-0.12 % perplexity band before it becomes a
default, and that run is not this one.
Memory: the probe's peak during the prompt and the driver's figure at the end. No threshold.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| every arm ends | probability 0.6 |
| `k2000` and `k9000` bytes equal to `r` | probability 0.6 |
| `c` and `a` arms: the 32 tokens of `r` | probability 0.8 |
| `r9000`, peak of the probe, GB | 17.5 to 19.6 |
| `c9000`, peak | 13.5 to 17.5 |
| `k9000`, peak | 15.5 to 18.5 |
| `a9000`, peak | 10.5 to 14 |
| `ta9000`, seconds of the read, against the bias arm of the same day | within 15 % |

## Cost (*estimated*)

About 18 minutes on an `l40sx1`, $0.54. 35-minute timeout.
