# Deviations from prereg inkling-mac-decode-2026-11-26

E1. Step 2, the rotations in threadgroup memory, stopped at the kernel bench, as the prereg allows: 12.14 against
11.75 us on the input rotation, 9.80 against 9.83 on the down rotation, 225.57 against 225.51 us on the whole fused
call (median of 7 rounds of 400 chained launches, `kbench-1.log`). The kernel was bit for bit and was removed; it was
never wired and never timed in the model.

E2. Step 3 was timed before step 1. Its 5 rounds ran against the default of the time (K2 unset, `down` serial); K2's
screening then ran on that same default. K2's 5 rounds ran on the new default (`down` one simdgroup a slot), which is
what a K2 default would be served with.

E3. K2's screening is flat: every value from 5 to 1,000 reads 34.40 to 36.85 ms a step against 35.14 and 35.22 for
the default, inside the +-7 % of one run. The prereg sends the best value and every value within 2 % of it to the 5
rounds, which here is 10, 20 and 30. Only 30, the best, was timed in 5 rounds, for want of time in the session's
window; 10 and 20 read 34.94 and 34.72 ms, between 30 and the default.

E4. The kernel bench of `kbench-1.log` started with the GPU at 21 % (the prereg's bound of 10 % is written for timed
decode runs). Its 7 rounds are tight (87.53 us served, range 86.81 to 88.14), so the reading stands.

E5. Step 0 of the session, outside this prereg: arm B of prereg `inkling-mac-prompt-2026-11-26` against arm B with
the experts' rows batched and the bias on the device, three alternated rounds at 9,000 tokens. Its figures are in this
lot's journal and raw folder (`s0-*`), and its rule was fixed by the coordinating session before the runs.

E6. A first server reading (`ops/inkling/serve-mac.sh`) was started on the code with step 3 as default and stopped
by this session after its first requests, once K2 had passed its rounds: it would have checked a default about to
change again. Its partial files are kept in `serve-stopped/`. The reading that counts ran once, on both defaults.

E7. K2 is served through `inkling::new_metal_device`, which sets `CANDLE_METAL_COMPUTE_PER_BUFFER=30` only while the
Metal device of `inkrun` and of the runtime's Inkling load is made, and unsets it after, so other models of the same
process keep candle's 50. An explicit value in the environment still wins.

E8. The prereg reads D4 and D5 at the end of the lot from one final default run; three were run (`fin-r1` to
`fin-r3`), and their median is reported.
