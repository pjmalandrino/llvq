# Deviations of proofs/preregistration-gestures-2026-11-04.md

## E1, written after the run: the script's check of gate 6 was narrower than the gate

Gate 6 asks for "the arm table and five ratio lines" (section 4). The job printed the five ratio lines
(`docs/mesures/gesture-ab-4b-2026-11-04-brut/gesture-ab-256.txt`): `f16 / base`, `key / f16`, `ondev-key / key`,
`ondev-candle / f16`, `ondev-key / base`. The script `ops/jobs/gesture-ab.sh` counted them with the pattern
`^ +[a-z-]+ / [a-z0-9]+ +[0-9.]+ \[`, whose numerator class has no digit, so it missed `f16 / base`, counted 4, and the
part exited 1 (`docs/mesures/lot1-4b-2026-11-04-brut/part-gestures.log`). Every other gate passed, in the job and in
this reading: oracle MATCH, object and config hashes, the parity and tile lines, gate 4 on 8784ffc4..., gate 5 on
0597cee8..., and every line of gate 6 in its order, with 256 identical tokens in every arm and round and 9 of 9 card
mutants killed.

The gate as the prereg writes it holds, so the run is read under the prereg's rules; the failed check is reported
here, not hidden. The pattern is corrected in the script (numerator `[a-z0-9-]+`), and the correction is checked against
the raw file: 5 lines.
