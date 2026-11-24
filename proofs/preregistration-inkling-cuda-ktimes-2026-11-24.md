# Prereg: Inkling 2 Mini on CUDA, every kernel of a step timed on its own (2026-11-24)

Written before the runs. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The events run on the L40S showed that turning cudarc's events off changes nothing (0.0139 s against 0.0138 s, 889
kernel launches a step): the signed prediction of 2 to 5 ms was wrong, and the step is not host time of that kind.
Which kernels hold the 13.8 ms?

## Object and method

The object of the stack prereg, arm `act`. `inkrun --kernel-times on` drains the stream before and after each
launch and charges the time between to the kernel's name. Job `ops/jobs/inkling-cuda-ktimes.sh` on `l40sx1` and
`l4x1`. A drained launch pays a floor the pipelined one does not: the sum over a step is expected above the plain
step, and the reading is the order of the kernels and their time each, not their sum.

## Decision rule

A measurement that orders work, with no verdict. The `plain` arm must give the tokens and, within its range, the
step of the stack run on the same card; otherwise the image is in question before the probe is read.

## Signed prediction (the coordinator, before the runs)

| quantity | predicted |
|---|---|
| the kernel with the most time a step, L40S | `ad_products` or `moe_tetra_gu` |
| glue kernels (`glue-*`), each | 8 to 20 us |
| sum of the probed times a step, L40S | 15 to 25 ms |

## Cost (*estimated*)

About 7 minutes a card: $0.21 on `l40sx1`, $0.09 on `l4x1`.
