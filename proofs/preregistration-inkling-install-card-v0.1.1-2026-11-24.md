# Prereg: release v0.1.1 installed and Inkling 2 Mini pulled by name, on one card (2026-11-24)

## Object

`ops/jobs/inkling-install-card.sh` with `REL=v0.1.1`, one job on an L4, image
`nvidia/cuda:12.4.1-base-ubuntu22.04`: no cuBLAS, no NVRTC, no compiler, none of our binaries. The archive is the
Linux one of release v0.1.1, built by the workflow `llvq archives` from main after pull requests 46 and 47. Its
library pins `Pier-Jean/Inkling-2-Mini-LLVQ-Tetra` at revision `697661e3c825a282a1d7ad4be758ef674d003e0b`, the file
set of 8,145,313,108 bytes.

## What is read

The same five readings as `preregistration-inkling-install-card-2026-11-24.md`: the install ends in `llvq -v`
(0.1.1), the anonymous pull and its seconds, `llvq run inklingmini-2-tetra`, the server checks (a) to (f),
nvidia-smi after the load and after the 8,791-token prompt. One more: the banner of `docs/brand` at the head of
the server log.

## Criterion

PASS: the install, the pull and the run hold and every correctness check passes. Anything else is a FAIL and names
the step. The release is marked latest only on a PASS.

## Signed predictions

- The whole of it passes: probability 0.85. The same code passed from our own image two hours ago; what is new is
  the archive build of this tree, never run.
- The pull: 45 to 90 s. nvidia-smi after the load: within 100 MiB of 7,786 MiB.

## Cost

About 7 minutes of an L4, $0.08 to $0.15, *estimated*. Card spend on this model before it: $8.80 of $25.
