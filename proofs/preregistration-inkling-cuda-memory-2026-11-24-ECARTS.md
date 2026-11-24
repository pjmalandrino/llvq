# Deviations from the prereg of the memory run

> The prereg
> [`preregistration-inkling-cuda-memory-2026-11-24.md`](preregistration-inkling-cuda-memory-2026-11-24.md)
> is stamped. It is not edited.

## E1: a twin of the job on an `l4x1`, launched while the `l40sx1` one waited for its card

**What the prereg asks**: the job on an `l40sx1`.

**What is added**: the same four arms on an `l4x1`, same image, `RETRY=memory`, launched at 10:45 CET while job
`6b055b55ca8b51043c3b6713` was in `SCHEDULING`. No long prompt has been read on an L4 before. The prereg's predictions
are for the L40S and are not read against this twin. The only expectation written before it: the 9,000-token arms may
run out of the L4's 23 GB or of their 6-minute limit, and that is a result. Cost, *estimated*: $0.25.
