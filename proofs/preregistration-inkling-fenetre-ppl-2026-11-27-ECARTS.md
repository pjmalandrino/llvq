# Deviations from prereg inkling-fenetre-ppl-2026-11-27

P1. The prereg's conditional took the larger branch. The windows' maker is on this Mac (`ops/inkling/data.py`,
`eval_windows`, the WikiText-2 test parquet in the local Hub cache, the tokenizer of `~/llvq-inkling-mini/native`),
read only and offline. Its windows 0, 1 and 2 equal the files the Mac held, id for id, before anything was written
(`windows.py`, `windows.log`). Windows 3, 4 and 5 and the two joined texts were made before the first run, in a few
minutes. The lot therefore reads six single windows (12,282 positions) and two joined texts (12,286 positions).

P2. The scoring runs carry neither `--n 32` nor `--logits-out`: `inkrun`'s scoring mode returns before the greedy
steps and writes no logits file. Every other flag is part D's reference arm.

P3. The peak footprint of each run is `/usr/bin/time -l`'s, read on every run, as in part D.
