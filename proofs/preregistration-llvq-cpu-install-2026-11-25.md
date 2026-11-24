# Prereg: the CPU archive of llvq installed on a plain x86_64 machine, Qwen3 decoded on the CPU (2026-11-25)

## Object

`ops/jobs/llvq-cpu-install.sh`, one job on the Hugging Face flavor `cpu-upgrade` (8 vCPU, 32 GB, no GPU), image
`ubuntu:22.04`: no compiler, none of our binaries. The archive is `llvq-0.1.2-linux-x86_64-cpu.tar.gz`, built by
the workflow `llvq archives` from the branch `nuit/runtime-cpu-2026-11-25` (pull request 51), with the `dist`
profile. It is read from the job bucket, since no release carries it yet. The CPU kernel is the one of
`docs/mesures/cpu-tetra-speed-2026-11-25.txt`. Its AVX2 path has passed tests in CI and has never been timed; no
x86_64 machine has ever generated a token through it.

## What is read

1. `install.sh` picks `linux-x86_64-cpu` by itself and ends in `llvq -v` (0.1.2).
2. `llvq pull inkling2:mini` with no server, 8,145,313,108 bytes, with `LLVQ_PULL_JOBS=4` then `1`: seconds each.
3. `llvq run inkling2:mini` on this CPU build: the refusal and its text.
4. `llvq run qwen3:4b --verbose` on three prompts: the answer and its tok/s; the SIMD path and the thread count the
   server prints; the server's peak resident memory.
5. `llvq run qwen3:8b --verbose` on one prompt: the same. The 8B has never run on any CPU.

## Criterion

PASS: 1 holds, 3 is a refusal that names Inkling 2 and the CPU, and 4 gives three answers in readable English with a
rate. 2 and 5 are readings, not gates: a pull that is no faster at four files than at one is a result, and so is an
8B that fails. The job is one run on one shared virtual machine: its rates are a single reading with no interval.

## Signed predictions

- PASS: probability 0.7. The archive build scripts have never run, and the binary has never started on x86_64.
- The 4B on 8 vCPU with AVX2: 3 to 7 tok/s, against 8.65 on 12 cores of an M3 Max before the kernel change.
- The server's peak resident memory on the 4B: 1.8 to 2.4 GB.
- The pull at four files against one: between 1.0 and 1.5 times faster. In a datacentre one connection already
  moved 8.15 GB in 43 s, so the gain there is expected to be small.
- The 8B answers: probability 0.75, at 0.45 to 0.6 of the 4B's rate.

## Cost

About 25 minutes of `cpu-upgrade` at $0.03 an hour: under $0.05, *estimated*. Night ceiling for runs: $5.
