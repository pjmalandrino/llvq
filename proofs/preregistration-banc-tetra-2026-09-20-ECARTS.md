# Deviations from the bench prereg (2026-09-20)

The prereg is timestamped and is not edited.

## E1. The first launch was refused before it measured anything

Job 6aaf7b6852d0dbd7f1d72c28, l40sx1, ERROR after about two minutes, roughly $0.10.

    Error: "tetra48 is selected and no Tetra file was given. Pass the ball file
    as the first argument and the Tetra file as the SECOND — the two formats do
    not share a file. Or drop tetra48 from LLVQ_BENCH_ARMS."

`planesbench` takes **two** files. `planesbench.rs:1242` reads the Tetra path from
`std::env::args().nth(2)`, and the ball arms read the first. The job passed one.

The refusal landed at arm selection, after the 19 NVRTC kernels compiled and before any
timing, so **no number was produced and none could be wrong**. Section 4's signed prediction
is untouched and is scored against the relaunch.

Worth recording for the method rather than for the arithmetic: the bench refused by name, with
the fix in the message, and `run.py bench`'s own doc says why that matters — "without
`set -euo pipefail` a bench that dies leaves the rest of the line running and the job finishes
COMPLETED with no result, which reads as a pass".

## E2. A second file had to be uploaded, and it is the published ball object

`/out/ball-ref-2026-09-20/qwen3-4b-llvq.bin`, 1,770,527,533 bytes, the published Planes14
object. `rtbits` reads it as 252 matrices with 286 classes cross-checked against the decode
table, and the sealed-archive test of `llvq-artifact` identifies it as a legacy version, which
is what the ball arms expect.

So the run compares, in one process:

  ball arms   read the 2026-08 published object
  tetra48     reads the 2026-09-20 fine-tuned object

Two different models of the same architecture. For a **kernel** bench that is the intended
setup, since what is timed is the decode and the traffic at fixed shapes, not the weights. It
would not be acceptable for a quality bench, and no quality claim is taken from this run.

## E3. Relaunch

Job 6aaf7c1052d0dbd7f1d72c3e, same prereg, same phases, same card, both files.
