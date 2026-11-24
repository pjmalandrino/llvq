# Deviations from the prereg of the CUDA head run

> The prereg
> [`preregistration-inkling-cuda-head-2026-11-23.md`](preregistration-inkling-cuda-head-2026-11-23.md)
> is stamped. It is not edited.

## E1: the head job never ran, and the stack job carries its two arms

**What the prereg asks**: `ops/jobs/inkling-cuda-head.sh` on an `l4x1`, then on an `l40sx1`.

**What happened**: job `6b0498c2c97b8ed4fcd63b1c` was created at 19:49 UTC on 2026-11-23 and stayed in `SCHEDULING`
("Waiting for requested hardware to become available") for 50 minutes. It was canceled before a card was rented.
Cost: $0, *measured* (no card, no bill).

**What replaces it**: the `base` arm of `ops/jobs/inkling-cuda-stack.sh`
([its prereg](preregistration-inkling-cuda-stack-2026-11-23.md)) is the `head` arm of this prereg, same flags, on an
image one commit later (d2b163d instead of bddb382). The predictions of this prereg for the head step, 0.030 to 0.038 s
on the L4 and 0.018 to 0.028 s on the L40S, are read against that arm. The `attn` arm, the stack without the int4
head, is not rerun on the L40S: no L40S figure without the head exists.

**Order of the cards**: the stack job goes to the L40S first and to the L4 second, the L4 queue being the one that
stalled. A launch that waits 25 minutes for a card is canceled.
