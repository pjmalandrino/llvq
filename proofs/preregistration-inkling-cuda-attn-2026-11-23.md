# Prereg: Inkling 2 Mini on CUDA, the dense attention kernels, and where the glue's logit drift comes from (2026-11-23)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Questions

1. The glue run (`inkling-cuda-glue-2026-11-23`, job 6b04360ca02d1cf5a7b7e3e7) passed its rule: same 32 tokens,
   0.1103 to 0.0658 s a step. It also read a largest logit difference of 1.02 between the candle and fused glue,
   where the Mac reads 0.006 on the same prompt. Is that a routing flip at a near-tie, as class B allows, or a
   defect of `ink_glue.cu`?
2. With `--attn fused` on top (`kernels/ink_attn.cu`), are the tokens kept, and how long is a step?

## Object and method

The object of the moe4 prereg. Three arms (candle; fused glue; fused glue and attention), each run once timed with
its logits written and once with the routing of every decode step recorded. Job `ops/jobs/inkling-cuda-attn.sh`.

## Reading rule

- Question 1. A flip explains the drift if the rows before the first visit that routes other experts differ from
  the candle arm by less than 0.05, and that visit's margin in the candle arm is under 1e-3. A drift above 0.05
  with no flip before it is a defect: the fused glue is then not used on CUDA until it is found.
- Question 2. PASS if the `attn` arm generates the 32 tokens of the Mac's run and its log prints `NVRTC attn unit`.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| the glue's drift is a routing flip | probability 0.55 |
| NVRTC accepts `ink_attn.cu` at the first try | probability 0.6 |
| tokens kept by the `attn` arm, given it compiles | probability 0.75 |
| median step, `attn` | 0.035 to 0.050 s |

## Cost (*estimated*)

`l4x1`, about 9 minutes: $0.12, $0.40 at the timeout. Card spend before the run: $2.96 of $25.
