# Deviations from preregistration-pers-l1-2026-11-04.md

Written on 2026-11-04, after the stamp (sha256 `9c2353466c41…`, `.ots` in commit 4039b5e) and before any measurement.
The stamped file is not edited.

## E1. Four table arms instead of five, timeout 30 min instead of 90 (before the run)

- **What changes.** The bench runs `LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48`: `nullk` leaves. The job's timeout
  goes from 90 to 30 min (`ops/jobs/pers-l1.sh`).
- **Why.** The operator's lot of 2026-11-04 asks every job's worst case under $1. METHODE §6's 90-minute rule applies to
  a `planesbench` with five arms or more, because ten-arm transcodes took about 1,470 s (*measured*, E2). With four
  arms the same bench ran its whole job in 270 s on 2026-11-04 (*measured*, `seg-tetra-2026-11-04.txt`), so 30 min
  keeps a margin above twice the expected 12 min.
- **What it costs the reading.** `nullk` was a coherence control of the table only (§5: "the table, as a coherence check
  only"). No rule of §7 reads it, and no prediction of §6 that a rule reads involves it. The table's control line
  becomes FP16 / Planes14 / Tetra / Slot32, each against `banc-252` sweep-64 (Slot32 against `banc-t64`). The
  one-launch pass, A4, the microbenchmarks, the cubins, the boundary A/B and rule H are untouched.
- **Cost.** Expected about 12 min, **$0.36** (*estimated*, unchanged). Worst **$0.90** (30 min at $1.80/h, *computed*),
  against $2.70 in §8.
- **Gates.** Unchanged. Gate 3's `tetra48: 252 of 252 matrices matched by name` and `tile 64` are printed with four
  arms as with five (`seg-tetra-2026-11-04` printed both).
