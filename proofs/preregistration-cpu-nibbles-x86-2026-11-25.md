# Prereg: the Tetra codes decoded once into nibbles on the CPU, against the served words (2026-11-25)

## Object

`ops/jobs/cpu-nibbles-x86.sh`, one job on `cpu-upgrade` (8 vCPU of an AMD EPYC 7R13 in the five jobs before it, not
guaranteed), image `rust:1.95.0-bookworm`, the branch of pull request 51 at the commit the job prints. The sealed
Qwen3-4B pulled by name, served config `configs/qwen3-4b-tetra-e4.json`.

`LLVQ_CPU_WEIGHTS=nibbles` decodes every Tetra word once at load into four bits a coordinate and one header byte a
block of 24 (`tetra_cpu::TetraNib`, 4.33 bits a weight for 2), and frees the words. The matvec then reads twelve
bytes a block and one sixteen-entry lookup. `nibbles_give_the_words_outputs_bit_for_bit` holds on NEON, on the
scalar path, and on AVX2 under Rosetta. Job 6b069ca1 read the Tetra kernel at 66 % of a 4B token on this CPU.

The target the operator named is a laptop of the same class: 4 cores and 8 threads, 32 GB, no usable GPU.

## What is read

1. The bench at 8 and 1 threads: each Tetra site as words and as nibbles, and the token's sum both ways.
2. `cpurun <sealed> phases 48` packed, nibbles, packed again, each under `/usr/bin/time -v`: ms a token by part,
   the rate, the peak resident memory.
3. `cpurun <sealed> tetra 48` in both modes with a dump: the greedy tokens compared.
4. `llvq run qwen3:4b` through `llvq serve` in both modes: the client's rate and the server's peak memory.

## Rule

- Tokens DIFFERENT in step 3: the mode is wrong, it is not kept, whatever its rate.
- Tokens identical and the nibble token faster than both packed runs by more than their spread: the mode is kept in
  the tree, default `packed`, and the rate and the memory go to the operator, who decides the CPU default.
- Not faster by more than the spread: the mode is removed.

## Signed predictions

- The job ends on DONE: 0.75.
- Tokens identical: 0.9.
- The Tetra kernel in `phases`, nibbles against packed: 1.8 to 3.0 times faster.
- The token, nibbles against packed: 1.5 to 2.0 times faster, which is 5.5 to 7.5 tok/s on this machine.
- Peak resident memory of `cpurun`, nibbles: 2.5 to 3.0 GB. The load passes through words and nibbles of one
  matrix at a time, not of the whole model.
- The load takes at most 10 s longer in nibbles: 0.7.

## Cost

About 30 minutes of `cpu-upgrade` at $0.03 an hour: $0.02, *estimated*. Spent on CPU jobs before it: about $0.06.
