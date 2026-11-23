# Prereg: Inkling 2 Mini on CUDA, the fused glue of a decoded row (2026-11-23)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

On an L4, with the experts in four launches in both arms, does `inkrun --glue fused` (`llvq-llm/kernels/ink_glue.cu`,
the routing handed to the experts on the card) generate the tokens of `inkrun --glue candle`, and how long is a
decode step on each?

## Object

The object of `preregistration-inkling-cuda-moe4-2026-11-23.md`. Job `ops/jobs/inkling-cuda-glue.sh`, `l4x1`.

## Decision rule

The fused glue is class B (its sums run in another order), so the logits are not expected equal.
- PASS: both arms generate the 32 tokens of the Mac's resident run, the job log of the `fused` arm prints
  `NVRTC glue unit`, and its report says `provenance.glue = fused`.
- FAIL: the `fused` arm's tokens differ from the `candle` arm's, or the kernels do not compile.
The largest logit difference between the arms is reported, not judged. The perplexity band of the Mac's class B
gates is not read here; it is owed before the fused glue is served on CUDA.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| NVRTC accepts `ink_glue.cu` at the first try | probability 0.5 |
| tokens equal, given it compiles | probability 0.8 |
| median step, `candle` | 0.100 to 0.115 s (0.1076 on the profile run) |
| median step, `fused` | 0.060 to 0.085 s |

## Cost (*estimated*)

`l4x1` at $0.80/h, about 5 minutes: $0.07, $0.40 at the timeout. Card spend before the run: $2.91 of $25.
