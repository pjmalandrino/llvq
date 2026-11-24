# Deviations from the prereg of the memory levers run

> The prereg
> [`preregistration-inkling-cuda-memory-levers-2026-11-24.md`](preregistration-inkling-cuda-memory-levers-2026-11-24.md)
> is stamped. It is not edited.

## E1: the job runs on both cards at once

**What the prereg asks**: an `l40sx1`, an `l4x1` if none comes.

**What is done**: both, launched together on one image. The operator named his target card after the prereg was
stamped: an RTX 4060 Ti with 16 GB. The L4 is the nearest card on hand (same generation, 24 GB), so the peak is read
there against 16 GB. The prereg's predictions are for the L40S; the peaks of the L4 are expected within 1 GB of them,
as they were in the run before (19.0 against 19.6 GB). Cost of the twin, *estimated*: 30 minutes, $0.40.
