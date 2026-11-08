# Deviations from proofs/preregistration-spec-lookup-2026-11-08.md

1. **The 4B alone, under a cap of $0.75 (2026-11-08, after the stamp, before any measurement).** The operator chose
   "4B seul : 0,75 $" over the three sizes at $3.30. The job runs `SIZES=4b`, with the job's timeout at 25 min
   ($0.75 at worst, *computed* at $1.80/h) and specrun's at 22 min instead of 25, so the oracle and the file checks fit
   under the cap. Expected about 19 min, $0.57 (*estimated*). Gates, prompts, arms and predictions are unchanged; the
   8B and 14B rows of §3 and §6 are not run and need a new go. If the 22 min cut falls inside the GSM8K arms, the door
   prompts are reported and the GSM8K identity gate is read as not reached, not as failed.
