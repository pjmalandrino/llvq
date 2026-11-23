# Prereg: Inkling 2 Mini on CUDA, the routed experts of a decoded token in four launches (2026-11-23)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

On an L4, does `inkrun --moe four` (`llvq-llm/kernels/moe_tetra.cu`: one rotation launch, one gate and up launch,
candle's activation once for the eight experts, one rotation launch, one down launch with the weighted sum) give the
logits of `inkrun --moe one` (one expert at a time, the path of every earlier card run) byte for byte, and how long
is a decode step on each?

## Object

The object of `preregistration-inkling-first-card-2026-11-22.md`: the text model in this engine, the experts of the
24 layers resident in lattice codes, every other tensor from the text-only checkpoint (BF16 tables), eager path,
attention and glue through candle. Not the sealed file's tables, not a served speed. 32-token prompt, 32 greedy
tokens. Job `ops/jobs/inkling-cuda-moe4.sh`, flavor `l4x1`, same inputs at the same pins.

## Controls

1. `oracle` MATCH on CUDA.
2. Every input at its pinned sha256.
3. The job log of the `four` arm prints `NVRTC moe unit` and never `keeps the one-expert path`; its report says
   `provenance.moe = four` and 4 lattice launches a visit (192 a step). The `one` arm reports 1,920 a step.
4. 24 resident layers in both arms.

## Decision rule

- PASS: controls hold and the two logits files (33 rows of 200,058 f32 each) are identical byte for byte.
- CLASS B: the files differ and the 32 tokens of both arms equal the Mac's reference. The four launches are then not
  the one-expert path's bits; the difference is located before any further use.
- FAIL: a control fails, or the tokens of the `four` arm differ from the `one` arm's.
The kernels have never met NVRTC: a compile refusal is control 3 failing, the `one` arm still reads, and the run is
not repeated before the source is fixed.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| NVRTC accepts `moe_tetra.cu` at the first try | probability 0.6 |
| logits identical byte for byte, given it compiles | probability 0.7 |
| median step, `one` | 0.155 to 0.170 s (0.1656 on 2026-11-22) |
| median step, `four` | 0.055 to 0.095 s |
| billed | 0.2 to 0.35 h |

## Cost (*estimated*)

`l4x1` at $0.80/h, about 15 minutes: $0.20, $0.40 at the 30-minute timeout. Card spend on this model before the
run: $2.82 of $25.

## What this does not read

A served speed, the sealed tables, a quality figure, any rival. The step time of the `four` arm is one stage of the
CUDA port: the attention and the glue are still candle's.
