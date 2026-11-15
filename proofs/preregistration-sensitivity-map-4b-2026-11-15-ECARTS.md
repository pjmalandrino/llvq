# Departures from the prereg sensitivity-map-4b (2026-11-15)

The prereg `preregistration-sensitivity-map-4b-2026-11-15.md` (sha256 1453e867, stamped before the first launch) is
not edited. This file is written and stamped before the relaunch it allows.

## E1. The first launch is void, and the `fisher_mmlu` questions are chosen by another rule

Job 6af9be7023985fb9cd8368be stopped after 79 s, before any model loaded, $0.04. Its own guard refused it: "the
Fisher questions share 0 questions and 2 token hashes with the scored ones". MMLU repeats a few prompts, and two of
round-robin rows 570 to 1139 have the same token hash as a scored prompt. Nothing was read. Raw files:
`docs/mesures/sensitivity-map-4b-2026-11-15-brut/void-1/`.

The prereg's choice 3 took round-robin rows 570 to 1139. The set is now the round robin walked from row 570,
skipping every row whose (subject, index) or token hash is in the scored 570 or already taken, until 570 rows are
held. The job prints the skipped rows. The guard is unchanged and must pass.

Nothing else in the measurement changes: the bases, the arms, the gates, the readings, the rule, the caps and the
cost are as stamped. The signed predictions stand as written. The relaunch writes to
`sensitivity-map-4b-2026-11-15-r2/`, beside the void run's folder.

The scripts of the relaunch, which the job accepts through this file:

- `ops/sensitivity_map.py` sha256 `127159b27121a1d71fa6cef4c3d2a2f15ca5e74b8568ca686dc37768b612a25a`
- `ops/jobs/sensitivity-map-4b.sh` sha256 `52e1f3466b38e9150c0b6f40e17364eb52196219c3bb9784115537c7edb96447`

`ops/patch_map.py`, `ops/contrast_map.py` and `ops/layer_vs_output.py` are unchanged, at the sha256 the prereg names.

The relaunch runs under the operator's go of 2026-11-15 for this reading: the void run and the relaunch together stay
under the ceiling he was given, $1.35.
