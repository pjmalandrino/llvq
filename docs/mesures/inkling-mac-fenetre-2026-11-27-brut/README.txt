Raw files of prereg inkling-fenetre-2026-11-27 (journal: ../inkling-mac-fenetre-2026-11-27.txt).
The *.logits.f32 files (25.6 MB each) stay out of git, as in the folders before; logits.sha256 pins them.
run_arm.sh and arms.sh are those of ../inkling-mac-prompt-2026-11-26-brut/ with the changes of -ECARTS.md F1
(binary copied under target/lotd, hard limit, stdin closed, /usr/bin/time -l). <arm>.json, .log, .machine: inkrun's
report, its stderr followed by time -l, the machine before the run. compare.py: gate D-ii, one logits file against
another (max |dlogit|, largest |logit|, argmax of every row, the 32 tokens).
all.sh and all.log: the guarded job of D-ii and D-iii (tree state first, then every arm). inkrun.sha256: the copy
every arm ran; build.log: its release build. dii-9000.txt, dii-2000.txt: gate D-ii. summary.py and d-iii.txt: gate
D-iii, pairs, medians, peak footprints. logits.sha256: every logits file of the lot.
d-i.log: gate D-i (tests/inkling_window.rs). mutants.sh and mutants.log: the three mutants of the prereg seen red,
sites counted, then the lints on the committed tree (clippy-final.log, linux-check-final.log). clippy.log and
linux-check.log: the first lints. tests-debug.log: `cargo test -p llvq-llm --features metal` in debug.
