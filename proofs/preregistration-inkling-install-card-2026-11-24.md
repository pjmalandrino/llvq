# Prereg: the released archive installed and Inkling 2 Mini pulled by name, on one card (2026-11-24)

## Object

`ops/jobs/inkling-install-card.sh`, one job on an L4 (Ada, sm_89, the generation of the RTX 4060 Ti; 24 GB where
that card has 16). Image `nvidia/cuda:12.4.1-base-ubuntu22.04`: no cuBLAS, no NVRTC, no compiler, none of our
binaries. The archive is the Linux one of the GitHub prerelease built from the commit that fills the `inkling2:mini`
entry of `llvq-runtime/library.json` with `Pier-Jean/Inkling-2-Mini-LLVQ-Tetra` at revision
`a7d3b5cf1d38be294ec544df2060fea10f180bf3`, index sha256 `9a4052a9b4847a2baf659f88d1de9607a3b8297004faad8585b243a158857efd`.

## What is read

1. The install script ends in `llvq -v`, on a machine whose `ldconfig` lists no cuBLAS and no NVRTC.
2. `llvq pull inkling2:mini` ends without a token being needed, and its seconds.
3. `llvq run inklingmini-2-tetra "<prompt>"` prints an answer.
4. The checks (a) to (f) of `ops/inkling/serve_checks.py` on the pulled model, as in the server jobs of today.
5. nvidia-smi after the load and after the 8,791-token prompt.

## Criterion

PASS: 1, 2 and 3 hold and every correctness check of 4 passes. Anything else is a FAIL and names the step.

## Signed predictions

- The whole of it passes on the first run: probability 0.35. The likeliest failure is the binary not finding a
  library or the kernels not compiling outside our image (0.3), then the pulled model not reaching the fused
  CUDA stack the `--local` server set (0.2).
- nvidia-smi after the long prompt, if reached: 10.5 to 12.5 GB as nvidia-smi prints MiB (11,082 MiB this
  morning with the same settings).
- First token of the 8,791-token prompt, if reached: 70 to 95 s (81.23 s this morning).
- The pull: 2 to 8 minutes.

## Cost

About 15 to 25 minutes of an L4, $0.20 to $0.35, *estimated*. Card spend on this model before it: $8.41 of $25.
