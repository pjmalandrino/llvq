# Prereg: Inkling 2 Mini on CUDA, the perplexity of a prompt read in pieces (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

Read in pieces of 512 tokens with the sliding slots bounded and the scores in chunks of 128 rows, a 9,000-token
prompt peaks at 12.0 GB instead of 19.5 and is read in 23.85 s instead of 55.89 (*measured*, L40S, job
`6b05708eaba0b0fd660a5494`). The 32 tokens after it are the same, but the logits are not: the largest gap is 8.69 at
9,000 tokens and 1.77 at 2,000, where the CPU model showed 2e-5. Is that the price of a few routing choices falling
the other way at a near tie, which costs nothing on average, or a fault of the pieces path?

## Object and method

Job `ops/jobs/inkling-cuda-pieces-ppl.sh`, an `l40sx1`, the image of the levers run. `inkrun --score-prompt on`
scores every position of a text through the prompt path. Texts: WikiText-2 test windows 0, 1 and 2 (2,047 positions
each) and windows 0 and 1 joined (4,095 positions, the only text longer than the bounded window). Arms: `ref` (one
call), `c` (pieces of 512), `all` (pieces of 512, bounded sliding slots, scores in chunks of 128). Paired by position
against `ref`.

## Decision rule

The pooled ratio over the three windows (6,141 positions), `exp(mean difference of NLL) - 1`, with its 95 % interval.
PASS: the point is inside +-0.12 %. A point outside is a FAIL: the pieces path is then a fault to find or a lever that
costs quality, and it does not become a default. The joined text is read apart, with the same band, and it is the one
that exercises the bounded slots; a FAIL there alone fails `all`.
A position-wise look is owed either way: if the differences are a fault they will sit at the piece boundaries
(positions 512, 1,024, 1,536), and the mean difference by position modulo 512 is reported after the run from the
dumps.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| `c`, pooled point inside the band | probability 0.6 |
| `all`, pooled point inside the band | probability 0.55 |
| `all`, joined text inside the band | probability 0.5 |
| the differences cluster at piece boundaries (a fault) | probability 0.3 |

## Cost (*estimated*)

About 14 minutes on an `l40sx1`, $0.42. 35-minute timeout.
