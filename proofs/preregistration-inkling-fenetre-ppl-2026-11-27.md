# Prereg: the perplexity of Inkling 2 Mini's window arm on the Mac (2026-11-27)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it. Operator's order: autonomy for the night of 2026-11-26, local work at 0 $. Cost: 0 $, this Mac only
(M3 Max, 64 GB). Predictions signed by the main session. The arm stays off by default whatever this lot reads:
turning it on is the operator's decision.

## Question

Part D of `inkling-fenetre-2026-11-27` kept the arm `inkrun --attn-window narrow` in the code, off: a 9,000-token
prompt is read x0.855 [0.854, 0.856] faster, with the same argmax on 32 rows and the same 32 tokens, and logits that
differ by up to 6.41 against a largest logit of 29.49 (*measured*,
[inkling-mac-fenetre](../docs/mesures/inkling-mac-fenetre-2026-11-27.txt)). Its journal says what it does not read:
equal greedy tokens over 32 steps are not a perplexity band.

The prompt read in pieces moved the logits by as much (8.69 at 9,000 tokens on an L40S) and its perplexity read
-0.042 % [-0.127, +0.044] against one call (`inkling-cuda-ttft-2026-11-24`). Is the window arm the same kind of
change, a few routing choices falling the other way at a near tie, or does it cost quality?

## Object and method

`inkrun --score-prompt on --nll-out` scores every position of a text through the prompt path, on Metal, with the
long-prompt settings of part D's reference (arm B of `inkling-mac-prompt-2026-11-26`). Two arms that differ by
`--attn-window` alone: `full` (the reference) and `narrow`. Paired by position. The binary is part D's
(`target/lotd/inkrun`, its sha256 in the raw files).

Texts: the three WikiText-2 test windows of 2,048 tokens this Mac already holds (`wikitext_w0_2048.json`,
`wikitext_w1_2048.json`, `wikitext_w2_2048.json`, 2,047 positions each), and the three joined in one text of 6,144
tokens (6,143 positions). The joined text is the one where the sliding slots hold more than the window for most rows.

More text, fixed before any run: if the script that made those windows and its corpus are on this Mac, windows 3, 4
and 5 are made the same way before the first run, read alone, and joined in a second text of 6,144 tokens. If they
cannot be made in 20 minutes, the lot runs on the three windows it has and the journal says so.

Each run under the guard, stdin closed, a hard time limit. Reading order: `full` then `narrow`, text by text.

## Decision rule

The ratio is `exp(mean difference of NLL, narrow minus full) - 1`, with the 95 % interval of the mean paired
difference, as `ops/jobs/inkling-cuda-pieces-ppl.sh` computes it.

- Primary: the joined texts pooled (6,143 positions, or 12,286 with the second text). PASS: the point is inside
  +-0.12 %.
- Secondary: the single windows pooled (6,141 positions, or 12,282). PASS: the point is inside +-0.12 %.
- The arm PASSES when both pass. The journal says whether each interval is inside the band too; an interval that
  leaves the band is reported and does not fail the arm, as for the pieces.
- A position-wise look is owed either way: the mean difference by position modulo 128 (the score chunk) and modulo
  512 (the piece), from the dumps. A fault of the arm would sit at chunk or piece boundaries.
- Nothing changes in the code on either verdict. On PASS the journal says the arm carries the band the pieces
  carry, on this Mac, and that serving it needs the operator's word and a reading on the other backends. On FAIL the
  journal says the arm costs quality and is not proposed.

## Signed predictions (the main session, before the run)

| quantity | predicted |
|---|---|
| primary, point inside +-0.12 % | probability 0.75 |
| secondary, point inside +-0.12 % | probability 0.80 |
| primary, the whole interval inside +-0.12 % | probability 0.35 with one joined text, 0.55 with two |
| the differences cluster at chunk or piece boundaries (a fault) | probability 0.10 |
| largest per-position difference of NLL | above 1.0 on at least one position, probability 0.6 |

Basis: the keys the arm drops have probability exactly zero, so each row is the same in exact arithmetic. The sums
run over fewer terms, the bits move, and a routing choice at a near tie can fall the other way; that moves single
positions in both directions and the mean by little.

## What this lot does not do

No card, no paid run. No default changes and the runtime is not touched. No perplexity on the CPU or on CUDA. No
GSM8K. One machine and at most 12,286 positions: a cost under about 0.1 % is not resolved.
