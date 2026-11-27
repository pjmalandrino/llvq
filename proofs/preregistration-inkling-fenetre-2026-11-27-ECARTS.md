# Deviations from prereg inkling-fenetre-2026-11-27

F1. The arm's scripts are those of `inkling-mac-prompt-2026-11-26` with three changes: `inkrun` is a copy of the
release binary kept under `target/lotd/` (lot A builds in the same worktree, so `target/release/inkrun` could change
between two arms), each run has a hard limit (perl `alarm`) and a closed stdin, and `/usr/bin/time -l` wraps every
run. The peak footprint of each arm is therefore read on every timed run, not on one process an arm.

F2. Gate D-ii compares the arm's logits files to the reference digests of 2026-11-26; the reference `.f32` files of
that lot were still on disk and served for the comparisons. The reference arms of D-iii, run from this lot's binary,
also give their digests, so the build is checked against the morning's.

F3. Gate D-i also runs on the CPU under the candle attention, where the steps go through the narrowed path too. The
prereg names the tiny model only; this is one more device, not a change to the gate.

F4. Gate D-ii ran the arm once at each length before D-iii, in the same guarded job; D-iii then ran its own pairs.
The arm's digests of D-ii and of the five D-iii arms at each length are the same.

F5. Part E was not run: the decode's score kernel reads the window only (`attn_scores.rs` line 193), so E is void,
as the prereg foresees. The two E predictions are not measured.
