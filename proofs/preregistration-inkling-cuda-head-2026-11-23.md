# Prereg: Inkling 2 Mini on CUDA, the int4 output head, on an L4 then on an L40S (2026-11-23)

Written before the runs. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

With the experts in four launches, the fused glue and the dense attention kernels, how long is a decode step with
the checkpoint's BF16 head through candle (arm `attn`) and with the sealed file's int4 g64 output table through
`kernels/ink_head.cu` (arm `head`), on an L4 and on an L40S?

## Object

The object of the moe4 prereg, eager path, 32-token prompt, 32 greedy tokens. In the `head` arm the output table is
the sealed file's (sha256 cb26ce0c...), packed on the card; the embedding stays the checkpoint's. Not the sealed
object in full, no quality figure, not a served speed. Job `ops/jobs/inkling-cuda-head.sh`, once with
`FLAVOR=l4x1`, once with `FLAVOR=l40sx1` if the first passes.

## Decision rule

PASS, per card: both arms end, the `head` arm's log prints `NVRTC head unit` and its report says `head int4`, and
its 32 tokens are the Mac's (the Mac's int4 head gives them on this prompt). A token difference at a top-2 margin
under 0.05 is read against that margin, not as a failure of the kernel. The two cards are read side by side and
never divided one by the other (hard rule 5).

## Signed prediction (the coordinator, before the runs)

| quantity | predicted |
|---|---|
| NVRTC accepts `ink_head.cu` at the first try | probability 0.7 |
| tokens of the `head` arm equal to the Mac's | probability 0.8 |
| median step, `attn`, L4 | 0.044 to 0.050 s (0.0466 on the attention run) |
| median step, `head`, L4 | 0.030 to 0.038 s |
| median step, `head`, L40S | 0.018 to 0.028 s |

## Cost (*estimated*)

About 6 minutes a card: $0.08 on `l4x1`, $0.18 on `l40sx1`. Card spend before the runs: $3.05 of $25.
