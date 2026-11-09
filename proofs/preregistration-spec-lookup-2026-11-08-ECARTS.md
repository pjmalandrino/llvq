# Deviations from proofs/preregistration-spec-lookup-2026-11-08.md

1. **The 4B alone, under a cap of $0.75 (2026-11-08, after the stamp, before any measurement).** The operator chose
   "4B seul : 0,75 $" over the three sizes at $3.30. The job runs `SIZES=4b`, with the job's timeout at 25 min
   ($0.75 at worst, *computed* at $1.80/h) and specrun's at 22 min instead of 25, so the oracle and the file checks fit
   under the cap. Expected about 19 min, $0.57 (*estimated*). Gates, prompts, arms and predictions are unchanged; the
   8B and 14B rows of §3 and §6 are not run and need a new go. If the 22 min cut falls inside the GSM8K arms, the door
   prompts are reported and the GSM8K identity gate is read as not reached, not as failed.

2. **The 8B and 14B, under a cap of $2.60 (2026-11-08, after the 4B run, before their measurement).** The operator's
   go: "Alors lance sur 8 et 14 pour test", cap $2.60 at worst for both. A second job, `SIZES="8b 14b"`, output
   `spec-lookup-2026-11-08-r1/`, job timeout 86 min ($2.58 at worst, *computed* at $1.80/h), specrun's timeouts 33
   min at 8B (35 in §7) and 50 min at 14B. Expected about 57 min, $1.70 (*estimated* from the 4B's 924 s, of which 212
   s of load, scaled by the greedy decode rates). Gates, prompts, arms and predictions are §3 and §6 unchanged.
