# Prereg: Inkling 2 Mini on a CPU, first run on the real file set (2026-11-25)

## Object

`ops/jobs/inkling-cpu-x86.sh`, one job on `cpu-upgrade` (8 vCPU of an AMD EPYC 7R13 and 32 GB in the seven jobs
before it, not guaranteed), image `rust:1.95.0-bookworm`, branch `cpu/inkling-2026-11-25` at the commit the job
prints. The file set of 8,145,313,108 bytes pulled by name with the `llvq` of that commit.

The CPU path is new (`inkling/experts_cpu.rs`, `head::Q4CpuHead`, the runtime's CPU branch). It has passed nine
synthetic tests on NEON and five mutants; it has never read the real codes and never run on x86_64. The attention
and the glue are candle's on the CPU in f32. Computed from the config: 1,152 Tetra matvecs a token, about 3.06 G
weights; the attention reads about 5.2 GB of f32 a token.

## What is read

1. `inkcpu <file set> 32`, codes packed, under `/usr/bin/time -v`: load seconds, prompt seconds, decode rate, the
   token split by part (`moe`, `tetra kernel`, `rotate kernel`, `head`, `rest`), the text, peak resident memory.
2. The same with `LLVQ_CPU_WEIGHTS=nibbles`.
3. `llvq run inkling2:mini` through `llvq serve`: one answer of about 100 words, the client's rate.

## Criterion

PASS: the three steps end without an error, the text of each is readable English on the subject asked, and the
ids `inkcpu` generates are the same packed and in nibbles. Anything else is a FAIL and names the step.
The rate is read, not judged: there is no threshold. The split says what the next change is: the largest part.

## Signed predictions

- PASS: 0.5. First contact with the real codes, on an architecture the path was never compiled for locally.
- Decode, packed: 1.0 to 3.0 tok/s. In nibbles: 1.2 to 4.0 tok/s.
- `rest` (attention included) is the largest part of the token in nibbles: 0.6.
- Peak resident memory: packed 10 to 14 GB, nibbles 17 to 22 GB.
- The load takes 30 to 120 s packed.

## Cost

About 40 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent on CPU jobs before it: about $0.07.
