# Deviations from prereg SP3 (`preregistration-spec-loop-sp3-2026-11-10.md`)

The prereg is not edited.

## Deviation 1: step (b) failed on its token count; rerun at 64 tokens a prompt

Written on 2026-11-11 at about 00:50 CET, before any loop measurement. Job `6af3a6799737d9a9e091f846` ran step (a)
(dflashcheck f16, its gate passed) and stopped step (b) before any generation: "Error: 4E: 128 tokens a launch, the
runtime holds 64". specbench decodes 4E's greedy reference in one launch of N tokens and the 4E runtime holds 64; the
prereg asked for 128. Nothing of (b) was measured; 8 min billed, $0.24.

The job is rerun once, on the orchestrating session's instruction: same measured code ce8fb7f, same image (Space sha
28e2bcc1, published from 5c0b0ea), the same steps with step (a) kept as it was, on l40sx1 only, timeout 28 min, its own
output directory (`RETRY=1`: `spec-loop-sp3-2026-11-10-r1/`), with 64 tokens a prompt in step (b). The predictions and
the decision rule are unchanged, except that every "128" reads "64" (token equality: 64 of 64). Cost of the rerun about
18 min, $0.54 (*estimated*).
