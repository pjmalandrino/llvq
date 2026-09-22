# Deviations. Preregistration served-8b-2026-09-21

The prereg is stamped (sha256 `f34595208f4c18f1...`) and not edited.

## E1. The q8 arm ran 0.6 tok/s over its interval

Predicted 80 tok/s [70, 90]; measured 90.6 [90.1–90.8]. The point was set near the `Planes14`
8B's 75.5; `planesbench` in the same night put Tetra at 1.62× `Planes14` over its own passes, so
the decode had more room than the prediction gave it. No row of the decision rule depends on it.

## E2. The first bench launch was refused on a false positive

`bench-8b.sh` refused because `hf buckets ls` prints `(empty)` for a directory that does not
exist, and the test was `grep -q .`. Fixed in four launchers (`grep -v '^(empty)$'`) before any
job ran; nothing was billed.

## E3. The training launcher's timeout had to be written in one unit

`run.py` refuses `2h15m`; the launcher was changed to `135m`, the same ceiling. This belongs to
the row-scales prereg and is recorded here because it was found on the same pass.

## E4. The decode ratios are not formed round by round

The prereg asks for the decode arms' ratios "round by round". `fusedrun` loads one arm at a time,
so its rounds never coexist and cannot be paired (`fusedrun.rs:1086-1099`). The ×3.42 and ×1.47
are quotients of medians over five rounds an arm. Their envelope runs from fused lo over dense hi
to fused hi over dense lo. Each fused arm is divided by the dense arm of its own process: 26.5
for q8, 26.4 for f16.

## E5. Control 5 holds on three arms of four

The prereg asks the f64 row check to pass "on every arm". It passes on FP16, `Planes14` and Tetra
(worst 2.8e-8, FP16). The floor arm `nullk` computes no product and is never compared; the log
prints `-inf` for it by design (`planesbench.rs:3173-3180`). Tetra's check covers its 216
matrices. The 36 int4 `v_proj` are neither compared nor timed in the bench.
