# Prereg: Inkling 2 Mini on CUDA, cudarc's event tracking off (2026-11-23)

Written before the runs. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The stack run left a decode step of 0.0139 s on an L40S and 0.0275 s on an L4, and a profile where no part
dominates on the L40S. The Inkling path leaves cudarc's event tracking on: two events an allocation and one a use.
The served door of the Qwen3 path turns it off. What does `--events off` change in a step, and how many kernel
launches does a step hold?

## Object and method

The object of the stack prereg, arm `act`. Arms `on`, `off`, then `off` with `--phases 12`. Job
`ops/jobs/inkling-cuda-events.sh`, on `l40sx1` and on `l4x1`. Nothing on the device changes with the switch: the
kernels, their inputs and their order are the same.

## Decision rule

PASS, per card: both arms end and generate the same 32 tokens as each other, logits included where the report
compares them. A difference between the arms is a defect, not a tolerance: the switch removes host work only.
The count of launches is a measurement with no verdict.

## Signed prediction (the coordinator, before the runs)

| quantity | predicted |
|---|---|
| step saved on the L40S | 2 to 5 ms, so 85 to 115 tok/s |
| step saved on the L4 | 2 to 5 ms, so 39 to 44 tok/s |
| kernel launches a step | 700 to 1,100 |
| tokens equal between the arms | probability 0.95 |

## Cost (*estimated*)

About 8 minutes a card: $0.24 on `l40sx1`, $0.11 on `l4x1`.
