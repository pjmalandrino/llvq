# Prereg: the CPU matvec microbenchmark on x86_64, AVX2, before and after the kernel change (2026-11-25)

## Why

The job `llvq-cpu-install` (6b0615aad0582cfbbefd2fae) served the 4B on 8 vCPU of an AMD EPYC 7R13 at 0.5 to 0.6
tok/s, against a signed prediction of 3 to 7. The answers were right and the server printed `avx2+fma SIMD on 8
threads`. Either the AVX2 matvecs are an order of magnitude slower than the NEON ones, or the time is elsewhere.

## Object

`ops/jobs/cpu-matvec-x86.sh`, one job on `cpu-upgrade`, image `rust:1.95.0-bookworm`. The ignored test
`bench_cpu_matvecs_at_the_4b_shapes` of `llvq-llm/tests/cpu_tetra.rs`, random codes, no model: on the branch of pull
request 51 at 1, 4 and 8 threads, and on main 4071f5d with the bench commit picked on it at 1 and 8 threads. The six
correctness tests of `cpu_tetra` run first on this CPU.

## What is read

The per-token sum of the matvecs, *computed* from the bench rows as in `docs/mesures/cpu-tetra-speed-2026-11-25.txt`
(q, k, gate, up on 36 layers and down on 24 in Tetra; v, o on 36 and down on 12 in int4 g128; the head once).

## Reading

- If the 8-thread sum is above 1.2 s, the matvecs explain the 1.75 s a token: the AVX2 kernel is the problem.
- If it is under 0.5 s, the time is outside the matvecs (attention, candle's CPU ops, the thread pools).
- Between the two, both.

## Signed predictions

- The six correctness tests pass: 0.95.
- The 8-thread sum after the change is above 1.2 s: 0.55.
- After against before at one thread: 1.0 to 1.3 times faster, since AVX2 took only part of the change.

## Cost

About 40 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent tonight before it: $0.01.
