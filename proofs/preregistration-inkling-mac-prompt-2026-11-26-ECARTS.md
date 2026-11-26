# Deviations from prereg inkling-mac-prompt-2026-11-26

E1. No step was kept. The two steps the prereg names (the experts' rows in a few launches, the score bias on the
device) keep every bit and move the 9,000-token read by less than the run-to-run spread, so neither became a default
and neither was wired into the runtime. The final reading of step 3 is therefore the default path, unchanged: its
figures are the reference arms run beside each step (four 9,000-token runs, three 2,000-token runs), plus one run of
each arm with `--peak-probe on`.

E2. The server reading of step 3 (`ops/inkling/serve-mac.sh`, the 13 checks and the first token of the 8,791-token
prompt) was not run. The runtime's code did not change in this lot, so the reading would have repeated the morning's
on the same code, at the cost of a quarter of an hour of the operator's GPU.

E3. Three kernel variants were timed that the prereg does not name, as kernel-only benches on synthetic codes: the
rows kernel with its activation staged in threadgroup memory (8, 16 and 32 rows a threadgroup), the rows kernel with
two output rows a simdgroup, and candle's f32 product at the experts' shapes (the price of plan item P3). They are
reported as benches; the first two were removed from the code, the third is an ignored test.

E4. Three timed arms started with the GPU at 55 to 60 % from a process outside the guard (`s1-ref2000` 58 %,
`s1-bias9000` 55 %, `s2-ref9000` 60 %), under the 60 % bound, so not marked busy. One reference arm read far slower
than its neighbours with load and GPU idle at its start (`s3-ref2000`, 42.83 s against 23 to 24 s); it is reported
as read, and no conclusion rests on it.

E5. Four launches under the guard started with the one-minute load between 6.4 and 7.2: each came right after one
of this lot's own release builds, whose load the reading still carried (the rule counts load outside this lot's
jobs). None was a timed `inkrun` arm.
