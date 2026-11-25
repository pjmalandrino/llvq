# Prereg: release v0.1.2, its three kinds of archive tried before it is marked latest (2026-11-25)

## Object

The five archives the workflow `llvq archives` builds from main at `0a4ef2c` (run 45586203132): Linux x86_64 CUDA,
macOS arm64 Metal, Linux x86_64 CPU, Linux arm64 CPU, macOS x86_64 CPU. They are published as a GitHub
**pre-release** `v0.1.2`, which `releases/latest` does not follow, and tried from there:

1. `ops/jobs/inkling-install-card.sh` with `REL=v0.1.2`, one job on an L4, image
   `nvidia/cuda:12.4.1-base-ubuntu22.04`: the same readings as `preregistration-inkling-install-card-v0.1.1-2026-11-24.md`.
2. `ops/jobs/llvq-cpu-install.sh` on `cpu-upgrade`, the released CPU archive and `install.sh` copied to the bucket:
   the readings of `preregistration-llvq-cpu-install-2026-11-25.md`.
3. On the operator's Mac (M3 Max), by hand: the Metal archive unpacked in a scratch directory, `llvq -v`,
   `llvq serve` on a port of its own, `llvq run qwen3:4b` on one prompt. One model of 1.4 GB, no other load.

Since v0.1.1 the tree gained the parallel pull, the `dist` build profile, the engine that frees its weights on a
panic, the CPU flavors, and the CPU kernels. Nothing in the CUDA or Metal decode path was changed on purpose; no
0.1.2 archive of those two has served a model.

## Criterion

- Card: PASS as in the v0.1.1 prereg (install, pull, run, every correctness check).
- CPU: PASS as in its prereg (flavor picked, Inkling refused by name, three right answers on the 4B).
- Mac: the version reads 0.1.2, the server starts, the answer is readable English on the Metal backend.

The release is marked latest only if the three pass. A FAIL names the step, the pre-release stays a pre-release,
and v0.1.1 stays latest.

## Signed predictions

- The three pass: 0.8.
- The pull of Inkling 2 Mini on the card: 15 to 45 s, faster than the 43 s of v0.1.1, four files at a time.
- nvidia-smi after the load: within 100 MiB of 7,786 MiB.
- The 4B on the CPU job: 3 to 7 tok/s.

## Cost

About 7 minutes of an L4, $0.08 to $0.15, and 15 minutes of `cpu-upgrade`, $0.01, *estimated*. Card spend on
Inkling before it: $8.88 of $25.
