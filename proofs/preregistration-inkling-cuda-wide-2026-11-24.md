# Prereg: Inkling 2 Mini on CUDA, the dense attention products with 128-bit weight loads (2026-11-24)

Written before the runs. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The kernel-times run on the L4 (job `6b04d7c0c97b8ed4fcd6cc95`) charges 7.97 ms a step to `ad_products` and 3.17 ms
to `ad_out`, *measured* with a drained stream: 11.1 ms of the 30.06 ms probed, the first and third kernels. Both read
their BF16 weights as four 32-bit loads a lane and an iteration. Does one 128-bit load in their place, the form of
`tv_q4_tile.cu`, shorten the step?

## Object and method

The object of the stack prereg, arm `act`, on the commit that changes `kernels/ink_attn.cu` only: `dot8` takes a
`uint4`. Same values, same order of additions. Job `ops/jobs/inkling-cuda-ktimes.sh` with `RETRY=wide`, on `l4x1` and
`l40sx1`: its `plain` arm is the step, its `ktimes` arm the time of each kernel.

The baseline on the L4 is job `6b04d7c0c97b8ed4fcd6cc95` (plain 0.0273 s). The kernel-times job never got an L40S
(job `6b04c852ca8b51043c3a99fb`, canceled in the queue, $0), so the L40S baseline is the plain step of the events and
context runs, 0.0138 to 0.0139 s, and their `attention_products` phase, 5.17 ms.

## Decision rule

The change is kept if the 32 tokens are those of the earlier runs on the same card and the plain step is not longer.
It is a gain if the plain step moves by more than its round-to-round spread (0.0001 s on the L40S, 0.0003 s on the
L4). Different tokens: the change is reverted.

## Signed prediction (the coordinator, before the runs)

| quantity | predicted |
|---|---|
| `ad_products` + `ad_out`, L4, probed | 8 to 10 ms a step (11.1 before) |
| plain step, L4 | 0.0250 to 0.0268 s (0.0273 before) |
| plain step, L40S | 0.0128 to 0.0137 s (0.0138 before) |
| tokens | identical |

## Cost (*estimated*)

About 7 minutes a card: $0.21 on `l40sx1`, $0.09 on `l4x1`.
