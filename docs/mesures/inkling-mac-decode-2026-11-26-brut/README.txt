Raw files of prereg inkling-mac-decode-2026-11-26 (journal: ../inkling-mac-decode-2026-11-26.txt).
The *.logits.f32 files stay out of git; logits.sha256 pins them (two digests: 0140733a... for the 9,000-token
prompt, 1edb8d95... for the 32-token prompt and 128 steps).
run_quiet.sh starts one inkrun arm only with load < 6, GPU < 10 % and the guard free; <arm>.machine is the state at
its start. step0.sh: step 0 (s0-*). decode_arm.sh, rounds.sh: the decode arms and the alternated rounds (d3-*, k2-*);
summarize.py reads a set of rounds. k2_screen.sh: K2's screening (k2s-*). final.sh (stopped, E6) and final2.sh: the
end of the lot (fin-r*, serve/). k2prompt.sh: the prompt read under K2 (kp-*). <arm>.arm: the flags of a decode arm.
twins-*.log, mutants-1.log, tests-final.log: the twin tests and the mutants. kbench-1.log: the kernel bench.
clippy.log, linux-check.log: the lints. serve-stopped/: the partial server reading of E6.
