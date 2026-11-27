Raw files of docs/mesures/qwen-cpu-groupee-2026-11-27.txt (lot G: the grouped attention on the CPU backend).

Scripts (each run under ~/.llvq-guard/heavy.sh, LLVQ_THREADS=6, RAYON_NUM_THREADS=6):
  tlim.sh            hard time limit: stdin closed, own process group, killed after N seconds
  waitload.sh        waits while the 1-minute load is above 6 (capped), prints load and memory
  job1-identity.sh   release build of cpurun, G-i lockstep, G-ii token runs; MODEL/TAG/EXP/GRP override the 4B
  job2-speed.sh      G-iii, rounds 0..ROUNDS, arms alternated, one process per arm and prompt
  job3-checks.sh     cargo clippy --all-targets, cargo test -p llvq-llm (debug), the prereg's mutant
  job4-8b.sh         G-iv: job1 (no build) then job2 on the 8B, ROUNDS=2
  analyse.py         prints every computed number of the journal from the files below:
                     python3 -I analyse.py 4b 3 ; python3 -I analyse.py 8b 2

Outputs:
  job1-identity.out, job2-speed.out, job3-checks.out   the jobs' own output, with the load before each run
  binary-4b.sha256                                     sha256 of target/release/cpurun used by every 4B run
  gi-lockstep-4b.log        G-i: one line per step (dlogit, both argmaxes, both max |logit|, bits same|differ),
                            the summary line, /usr/bin/time -l
  gii-4b-<arm>-<short|2048>.log/.json     G-ii: cpurun timed, the ids dumped by LLVQ_RUN_DUMP
  giii-4b-r<round>-<arm>-<short|2048>.log/.json   G-iii: same, round 0 discarded; the .log starts with the load
  job4-8b.out, binary-8b.sha256, gi-lockstep-8b.log, gii-8b-*, giii-8b-*   the same for the 8B (G-iv)
  clippy.log, test-llvq-llm.log, mutant.log       job3

arm exp = configs/qwen3-4b-tetra-e4.json (expanded, cat); arm grp = configs/qwen3-4b-tetra-e4-grouped.json
(grouped, grow(256)). Sealed file ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin, sha256 886391a8...71b8.
8B: ~/q8b-14b-sealed-2026-09-23/qwen3-8b-sealed-B.bin, sha256 7bdb9a55...5c13, arms
configs/qwen3-8b-tetra-e4.json and configs/qwen3-8b-tetra-e4-grouped.json.
No large binary file is kept here.
