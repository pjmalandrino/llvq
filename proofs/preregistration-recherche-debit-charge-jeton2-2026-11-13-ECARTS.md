# Deviations from the prereg of R6b (charge, then jeton2)

Beside `preregistration-recherche-debit-charge-jeton2-2026-11-13.md` (sha256 5b42fb9d, stamped, not edited). Written
on 2026-11-14, before the second attempt of job 1, and stamped before it.

## 1. Why a second attempt

Job 1 ran once (6af73aed, 328 s, $0.16) and is void: the check printed nothing and was killed by its 5 min timeout
(`docs/mesures/recherche-debit-charge-2026-11-13.txt`). It compiled seven modules on seven threads and printed only
after the last thread. No arm has a verdict. The question of the prereg is unchanged and unanswered.

## 2. What changes in job 1

| prereg | second attempt |
|---|---|
| §2.2: one thread an arm, lines printed at the end | the arms one after the other on one thread, every line printed and flushed as it is known (`persist4e_cuda::check_e1_arms`) |
| §4: `check:v9,v9ep,v9fl,v9ep2,v9ep3,v9fl2,v9fl3` | `check:v9,v9ep3,v9fl2,v9ep2,v9fl3,v9ep,v9fl`: the control first, the four variants next, the two arms already refused last |
| §4: output `/out/recherche-debit-charge-2026-11-13` | `/out/recherche-debit-charge2-2026-11-14` |
| §7: check 2 to 4 min, job about 6 min, $0.18, cap 7 min, $0.21 | check about 11 min if a module takes 90 s, job about 13 min, $0.39 (*estimated*), cap 17 min, $0.51 (*computed*); the check's own timeout is 14 min |

A check killed by its timeout now leaves the verdict of every arm it finished and the NVRTC time of the arm in
flight.

## 3. What does not change

The seven module texts and their sha256. The verdict's rule (RoPE form 1's token symbol, no local memory, one CTA a
SM). The gates of §4: the seven digests, and v9 LOADS with v9ep and v9fl REFUSED for local memory. The decision rule
of §5. Job 2's steps, score and rule; its own load check runs v9 and at most two arms, and its timeout goes from 5
to 6 min inside the same 28 min cap.

One reading is added. If the check is cut before v9ep or v9fl, gate 3 is not read in full. The verdicts of the arms
that finished then describe the check and not yet the runtime: job 2's own load check, which stops the job, is what
protects it.

## 4. Signed predictions (main session)

Predictions 1, 3, 4 and 5 of the prereg's §8 stand as written. Prediction 2 is replaced:

2b. One module takes 90 s [50, 150] in the check, NVRTC and JIT together. The job runs 780 s [480, 1,000]. It is cut
    by a timeout before the seventh arm at 25 %.

## 5. Spend

Before this attempt: $0.92 of the phase's $5, $20.14 of the $25 cap, registry $327.39.
