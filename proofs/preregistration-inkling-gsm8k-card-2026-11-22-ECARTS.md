# Deviations from preregistration-inkling-gsm8k-card-2026-11-22.md

The prereg is timestamped and is not edited. This file records what departed from it.

## E1, 2026-11-22: the run was stopped by the operator after about 6 minutes; verdict VOID

Job `6b033fab130a00b5aa80e8db` started at 19:17:04 UTC and was cancelled at about 19:23 UTC, on the operator's
order. Its controls had passed (oracle MATCH, every input at its pin) and it had written 4 problems.

Reason. The prereg says what this arm is: the experts in lattice codes, every other tensor the original's, the two
vocabulary tables included. The operator read that after the launch and asked for the run on the sealed file's
own values instead: a count that cannot be published as the file's is not worth its cost. The coordinator had
announced the run as "our model" before the launch and had given this precision only after it. The precision
belonged before the go.

What replaces it. A run of the same contract in which the engine reads the two int4 g64 tables of the sealed file
(`llvq-llm/src/inkling/tables.rs`, `inkgsm --tables`), under its own prereg:
`proofs/preregistration-inkling-gsm8k-sealed-2026-11-22.md`.

What the 4 written problems read, as information only, not as a result: 4 of 4 right, 294, 183, 353 and 206
generated tokens, median step 0.170 to 0.175 s (*measured*, `docs/mesures/inkling-gsm8k-card-2026-11-22-brut/
stopped/inkgsm.jsonl`). They are another arm than the replacement and are not merged with it.

Cost. `ops/run.py monitor` reads 0.00 h billed for the cancelled job. About 6 minutes ran, $0.08 at $0.80/h
(*computed*); the registry carries that figure.
