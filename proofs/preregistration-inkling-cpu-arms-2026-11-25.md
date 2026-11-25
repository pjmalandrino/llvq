# Prereg: three changes to Inkling 2 Mini's CPU token, one arm a commit (2026-11-25)

## Object

`ops/jobs/inkling-cpu-arms.sh`, one job on `cpu-upgrade` (8 vCPU of an AMD EPYC 7R13 in the jobs before it, not
guaranteed), image `rust:1.95.0-bookworm`. `inkcpu <file set> 32`, codes packed, on three commits of
`cpu/inkling-2026-11-25`, in order, then the first again:

- `86361cf`: the path job 6b06bb72 timed at 943.7 ms a token, with counters that split its `rest`.
- `72b16c5`: the attention's seven products held in BF16 and applied by a CPU kernel, in place of candle's f32.
- `2e3c3dc`: the routed experts of a visit run side by side on the pool, one fork a visit.

All three pass their synthetic gates on NEON and on AVX2 under Rosetta; the second and third have never run on the
real model.

## Rule

- The ids generated must be the same on `2e3c3dc` as on `72b16c5` (the change is bit-identical by construction). On
  `72b16c5` against `86361cf` the attention's sums change order: the text must stay readable English on the
  subject, and the ids are compared and reported, not required equal.
- A change is kept if its token is faster than the arm before it by more than the drift between the two runs of
  the first arm. A change that is not is reverted.

## Signed predictions

- The job ends on DONE with three readable texts: 0.7.
- `86361cf`: `attn proj` is above 400 ms of the token: 0.7.
- `72b16c5` against `86361cf`: the token 1.3 to 2.2 times faster.
- `2e3c3dc` against `72b16c5`: the token 1.1 to 1.5 times faster.
- The last arm decodes at 1.8 to 3.2 tok/s.
- Peak resident memory on the last arm: 10.5 to 11.5 GB.

## Cost

About 40 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent on CPU jobs before it: about $0.08.
