# Deviations from prereg SP4 (`preregistration-spec-loop-sp4-2026-11-11.md`)

The prereg is not edited.

## Deviation 1: the gate is restated after SP4

Written on 2026-11-11 at about 02:20 CET, after the job. SP4's rule made equality with 4E's greedy tokens the hard gate;
the job found 4 of 44 prompts diverging at near-ties of the pass's binary16 logits, for both drafters
(`docs/mesures/spec-loop-sp4-2026-11-11.txt`). The tree verify pass has never been bit-identical to 4E (S0: its logits
within 3.5e-3 relative RMS of 4E's): the pass decides every token, and at a near-tie it may rank two candidates
otherwise than 4E does.

After SP4, the orchestrating session restated the gate for the series: equality with 4E is reported (a non-fatal
agreement report), and the hard gate is self-consistency (every drafter's tokens equal the loop's with the null drafter,
the pass's own R = 1 greedy decode). SP4's outcome under its own rule stays NOT VALID; its numbers are not re-read
under the restated gate.
