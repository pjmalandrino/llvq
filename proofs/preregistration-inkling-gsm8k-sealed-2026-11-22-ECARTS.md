# Deviations from prereg inkling-gsm8k-sealed (2026-11-22)

The prereg is timestamped and is not edited. Deviations are written here.

## E1: the prompt-stream fingerprint printed in the prereg is the two-problem pilot's (found on 2026-11-23, after the run)

The prereg's contract and its control 5 name the prompt-stream fingerprint `5fb41b61271213dd`. That value is the
fingerprint of the stream of problems 0 and 1, the two-problem run on the Mac that tested the runner. The stream of
the 200 problems is `6d4d8fa2207c0146`. The coordinator copied the pilot's value into the prereg. The error is in
the prereg, not in the run. The first prereg (`preregistration-inkling-gsm8k-card-2026-11-22.md`) carries the same
wrong value.

What shows it:
- The run's dump carries `6d4d8fa2207c0146` in its header and its trailer
  (`docs/mesures/inkling-gsm8k-sealed-2026-11-22-brut/inkgsm.jsonl`).
- A Python FNV-1a over the reference ids (`transformers` 046bdb8cc2, the file the test `llvq-llm/tests/inkgsm.rs`
  compares the 200 rendered prompts with, id for id) gives `5fb41b61271213dd` for problems 0 and 1 together and
  `6d4d8fa2207c0146` for the 200 (*computed*).
- Each of the 200 rows of the dump carries the reference's own hash and prompt length for its problem, 200 of 200
  (*computed*).
- The dump of the run the operator stopped, on the same 200 problems, already carried `6d4d8fa2207c0146`. It was
  committed (`d4c60ca`) before this prereg's anchor (`76f1694`).

What it changes. Control 5 fails as written, on the printed value. What the control is there to guard holds: the
run read the 200 reference prompts, and nothing else. The coordinator reads the run as READ with this deviation and
says so wherever the count is given. Whether that reading stands is the operator's decision.

For every later arm of the comparison, the fingerprint to match is `6d4d8fa2207c0146`.
