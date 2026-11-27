Raw files of prereg inkling-fenetre-2026-11-27 (journal: ../inkling-mac-fenetre-2026-11-27.txt).
The *.logits.f32 files (25.6 MB each) stay out of git, as in the folders before; logits.sha256 pins them.
run_arm.sh and arms.sh are those of ../inkling-mac-prompt-2026-11-26-brut/ with the changes of -ECARTS.md F1
(binary copied under target/lotd, hard limit, stdin closed, /usr/bin/time -l). <arm>.json, .log, .machine: inkrun's
report, its stderr followed by time -l, the machine before the run. compare.py: gate D-ii, one logits file against
another (max |dlogit|, largest |logit|, argmax of every row, the 32 tokens).
d-i.log: gate D-i (tests/inkling_window.rs). clippy*.log, linux-check.log: the lints. mutants.log: the three mutants
of the prereg seen red, sites counted. build.log: the release build of inkrun whose copy every arm ran.
