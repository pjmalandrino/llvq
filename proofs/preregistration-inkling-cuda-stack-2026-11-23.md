# Prereg: Inkling 2 Mini on CUDA, the score kernel and the one-launch activation (2026-11-23)

Written before the runs. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

On top of the stack of the head run (experts in four launches, fused glue, dense attention products, int4 head), what
do the score kernel (`kernels/ink_scores.cu`) and the activation in one launch (`moe_act`) change in a decode step,
and where does the step go afterwards?

## Object and method

The object of the head prereg, arm `head`. Arms `base`, `scores`, `act`, then `act` with `--phases 12`. Job
`ops/jobs/inkling-cuda-stack.sh`, on `l4x1` and, if it passes there, on `l40sx1`.

## Decision rule

PASS, per card: every arm ends and generates the 32 tokens of the Mac's run; a difference at a top-2 margin under
0.05 is read against that margin. The `act` arm is not gated bit for bit against candle's activation: its `tanhf`
and `expf` are NVRTC's. The phases are a measurement that orders work, with no verdict. No quality figure: the
perplexity band is owed before any of these switches is served.

## Signed prediction (the coordinator, before the runs; the head run's result was not yet known)

| quantity | predicted |
|---|---|
| NVRTC accepts both new kernels at the first try | probability 0.6 |
| tokens kept by every arm | probability 0.75 |
| step saved by the score kernel, L4 | 3 to 6 ms |
| step saved by the activation launch, L4 | 2 to 6 ms |

## Cost (*estimated*)

About 8 minutes a card: $0.11 on `l4x1`, $0.24 on `l40sx1`.
