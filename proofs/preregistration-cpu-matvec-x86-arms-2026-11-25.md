# Prereg: three arms of the AVX2 Tetra kernel on x86_64, same job (2026-11-25)

## Object

`ops/jobs/cpu-matvec-x86-arms.sh`, one job on `cpu-upgrade` (the flavor of job 6b06256d1460ffe1d5e9592d: 8 vCPU of
an AMD EPYC 7R13 then, not guaranteed now), image `rust:1.95.0-bookworm`. The bench
`bench_cpu_matvecs_at_the_4b_shapes`, random codes, no model, at 1, 4 and 8 threads on three commits:

- `f8f27a0`: the kernel job 6b06256d timed, 3,082 ms a 4B token at one thread.
- `d47526c`: candidates picked by a byte shuffle, the norm in integers, binary16 through `vcvtph2ps`.
- `b4a6684`: on top of it, the addresses of eight blocks formed at once in vector registers.

All three give the same AVX2 outputs bit for bit on a fingerprint the agent that wrote them took under Rosetta, and
pass the six tests of `cpu_tetra` there. Under Rosetta `b4a6684` is 1.75 times slower than `d47526c`; Rosetta is not
a CPU and has no 256-bit registers, so that is not a reading of Zen 3.

## What is read and the rule

The per-token sum the bench computes, per arm and thread count. The first arm runs again at the end at one thread:
its drift bounds what a difference between arms can mean.

- The branch keeps the faster of `d47526c` and `b4a6684` at one thread, if it beats `f8f27a0` by more than the
  drift. If `b4a6684` is not faster than `d47526c` by more than the drift, it is reverted: it is the more complex one.
- If neither beats `f8f27a0`, both are dropped.

## Signed predictions

- The tests pass on all three arms: 0.9.
- `d47526c` against `f8f27a0`, one thread, token sum: 1.2 to 1.6 times faster.
- `b4a6684` is faster than `d47526c` on this CPU: 0.5.
- The best arm at 8 threads stays above 1.0 s a token on this machine: 0.7.

## Cost

About 45 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent tonight before it: $0.03.
