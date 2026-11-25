# Deviations from preregistration-llvq-release-v0.1.2-2026-11-25.md

## E1, 2026-11-25: the release is published as latest without the three trials

Operator decision, before any trial was launched: the pre-release and its three trials are too heavy for this
release. v0.1.2 is published directly as latest. None of the three criteria was read; nothing is claimed from them.

What stands in their place: the CPU archive built from an earlier commit of the same branch served the 4B and the 8B
as a stranger gets it (job 6b063b3cd0582cfbbefd6251), and CI is green on main at `0a4ef2c`. The CUDA and Metal
archives of 0.1.2 have not served a model. If one fails, v0.1.1 is marked latest again.
