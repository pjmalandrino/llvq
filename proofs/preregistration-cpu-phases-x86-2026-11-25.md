# Prereg: where a CPU token's time goes on x86_64, on the real Qwen3-4B (2026-11-25)

## Object

`ops/jobs/cpu-phases-x86.sh`, one job on `cpu-upgrade` (8 vCPU of an AMD EPYC 7R13 in the four jobs of the night
before, not guaranteed), image `rust:1.95.0-bookworm`, the branch of pull request 51 at the commit the job prints.
The sealed Qwen3-4B pulled by name at the library's pin, served config `configs/qwen3-4b-tetra-e4.json`.

Job 6b063689 read 182 ms a token for the matvecs of a 4B token at 8 threads on random codes. Job 6b063b3c, another
instance, served the 4B at 2.1 to 3.0 tok/s, which is 330 to 475 ms a token. Between 150 and 290 ms a token were
not attributed, or the second instance was slower. This job reads both on one instance.

## What is read

1. The bench `bench_cpu_matvecs_at_the_4b_shapes` at 8 and 1 threads, with a new row: the gate/up shape with 16
   distinct labels, whose decode tables stay in the first-level cache.
2. `cpurun <sealed> phases 48`: four prompts, 47 greedy decode steps each, the wall time of the decode split by the
   counters of `fused_cpu::phases` (prepare, tetra, int4, head, gather) and the rest, which is the model around the
   projections. Three arms: default threads, `RAYON_NUM_THREADS=1` (candle's own pool held to one thread),
   `LLVQ_THREADS=4`.
3. `llvq run qwen3:4b` through `llvq serve`, the same build profile (`ci`), two answers of about 150 words.
4. The default phases arm and the 8-thread bench again at the end: the drift of the instance.

## Rule

- If the two kernels and the head are 85 % of the token or more in the default arm, the kernel is the token on this
  machine and the attribution closes there.
- Otherwise the largest other part above 15 % of the token is the next thing to change: `rest` points at candle's
  model code, `prepare` at the rotation, the gap between `tetra` and `tetra kernel` at the tensor copies.
- If `RAYON_NUM_THREADS=1` moves the token by more than the drift, the two thread pools contend and the runtime pins
  candle's.
- If the 16-label row beats the random row by more than 1.3 times at one thread, the Tetra kernel is bound by its
  table misses on this CPU, and the next kernel arm decodes from the 16 KiB row table. Below 1.15 times it is bound
  by its instructions and that arm is not written.
- The runtime's rate against `cpurun`'s at the same threads reads what the server and the sampler add.

## Signed predictions

- The job ends on DONE: 0.75. `phases` has never run on a real model.
- The kernels' time in `phases` is within 1.3 times the bench's sum on this instance: 0.6.
- `rest` is above 30 % of the token: 0.5.
- `RAYON_NUM_THREADS=1` moves the token by more than 10 %: 0.3.
- The 16-label row is more than 1.3 times faster than the random one at one thread: 0.4.
- The runtime's rate is within 15 % of `cpurun`'s: 0.6.

## Cost

About 40 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent on CPU jobs before it: about $0.05.
