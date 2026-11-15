# Preregistration. Quality research phase 3 (CERVEAU O15): what the dumps on disk say of the depth arms, at $0

Written on 2026-11-15 on branch `recherche/qualite-n6-n7-2026-11-15`, before any of these readings was computed.
Timestamped before the first one, then never edited; a departure goes to
`preregistration-recherche-qualite3-dumps-2026-11-15-ECARTS.md`.

Cost: $0. No model is loaded, no job is launched, nothing is downloaded. A script reads committed files on the Mac.
The frame is `docs/mesures/recherche-qualite3-r1-cadrage-2026-11-15.txt`.

Seen before this prereg, and nothing else of these files' content: the opening characters of problem 0 in the FP16
and control GSM8K dumps, which differ from the first word, read while checking the dumps' fields.

## 1. The questions

1. **N7.** Do the late blocks cause more than their share of the token flips of a worked answer? The dull reading
   (H7-A) says GSM8K loses per weight about evenly over depth. The late window holds 0.306 of the restored weights
   (1,110,179,840 of 3,633,315,840, `depth-restore-4b` lines 27-28).
2. **N6.** Does restoring the late blocks, or the early ones, move the contrast of the MMLU answer logits apart from
   their noise? Q25 says the fall of the output slope after depth 25 is a rescale.
3. Two weak kills from the R0 note: H7-B (the late gain grows with the length of the chain) and H7-D (the late
   blocks carry the arithmetic).

## 2. The files

- MMLU, full split, four option logits a question, under `docs/data/mmlu-dumps/`: `mmlu-4b-f16-FULL.csv` (FP16),
  `mmlu-4b-abl-control-FULL.csv` (sealed control), `mmlu-4b-depth-late-FULL.csv`, `mmlu-4b-depth-early-FULL.csv`.
  For the September slices: `mmlu-q5-shipped-FULL.csv`, `mmlu-q5-down-0-11.csv`, `mmlu-q5-down-12-23.csv`,
  `mmlu-q5-down-24-35.csv`.
- GSM8K, 1,319 greedy completions an arm: `docs/mesures/gsm8k-depth-restore-4b-2026-11-14-brut/`
  `gsm8k-4b-depth-{control,late,early}.jsonl`, and FP16 in
  `docs/mesures/gsm8k-wave1-2026-09-26-brut/run5-vllm/gsm8k-4b-f16-vllm.jsonl`.

## 3. The readings

**G1, the flip hazard (N7).** For each arm X in {control, late, early} and each problem, d_X is the number of
leading characters X's completion shares with FP16's. A problem is censored for X when the two completions are
equal, or one is a prefix of the other. The hazard is h_X = (uncensored problems) / (sum of d_X over all problems),
flips a character. The cut by a restore is r_late = 1 − h_late / h_control and r_early = 1 − h_early / h_control.
Intervals: percentile bootstrap over problems, 10,000 draws, seed 20261115, the same draw for the three arms.
Beside it, with no rule: the median d_X, and the share of problems with d_X at least 20, 100 and 400.

**M1, slope and noise (N6).** The fit of `ops/logit_snr.fidelity` as
`docs/mesures/recherche-qualite2-r0-contraste-2026-11-14-brut/beta_table.py` runs it: the four logits of each
question centred, pooled, y = beta x + e, sd_res the residual sd. On the four MMLU dumps against FP16, and on the
four September dumps on their common questions. Intervals on beta, sd_res and beta / sd_res by bootstrap over
questions, same seed.

**M2, the flips of the late arm.** Of the questions whose pick the late arm changes against the control, the share
won and lost, and FP16's margin (top logit minus second) on those questions against all questions.

**G2, the chain's length (H7-B).** Problems cut in three equal groups by FP16's `n_gen`. In each group the paired
gain late minus control, in points. S = gain(long) − gain(short), bootstrap interval as G1.

**G3, false arithmetic (H7-D).** A checker reads each completion for steps of the form "a op b = c" with op in
{+, −, ×, ÷, *, /} and plain numbers (commas and currency signs stripped), and marks a step false when |a op b − c|
exceeds 0.005 of max(1, |c|). A completion is "false" when it holds at least one false step. Coverage is the share
of the control's completions with at least one parsed step. Read: the share of false completions among the control's
completions of the problems the late arm repairs (wrong in control, right in late), against the same share among the
problems the early arm repairs; problems both repair count in both.

**G4.** How many problems the late and the early arm both repair, of 108 and 143.

## 4. Gates

Each voids the reading it guards, and the journal says so.

- GSM8K grading: the script's grader gives 1,082, 1,139, 1,177 and 1,215 right of 1,319 for control, late, early and
  FP16, and 108 against 51 and 143 against 48 discordant, the counts of `gsm8k-depth-restore-4b-2026-11-14.txt` and
  its raw pair files. If a count differs, G2, G3 and G4 are void. G1 does not use the grades.
- MMLU: the script reproduces the control's and FP16's micro accuracy, 63.37 and 70.14, the paired gains +0.43 and
  +6.12, and the control's fit, beta 0.5817 and sd_res 1.4205 (`recherche-qualite2-r0-contraste` line 30).
- The four GSM8K dumps hold the same 1,319 problems in the same order, by `qhash`.
- G1: if fewer than 200 problems are uncensored for an arm, its hazard is void.
- G3: if coverage is under 50 %, G3 is void.
- A self-check of the script on built inputs: two completion sets with a planted flip rate give that rate back; a
  planted slope and noise give themselves back; three planted false steps are found.

## 5. The rules, fixed before the readings

| reading | rule | what it means |
|---|---|---|
| G1 | the interval of r_late lies above 0.306 | the late blocks cause more than their weight share of the flips: N7 holds a question about generation; the lens on generated tokens is the next reading |
| G1 | the interval of r_late contains 0.306 | the even reading is not rejected on flips either: N7 closes as explained, unless the patch map says otherwise |
| G1 | the interval of r_late lies below 0.306 | the late blocks flip fewer tokens than their share and still cost GSM8K at least their share: the loss is not in the count of flips |
| G2 | the interval of S lies above 0 | H7-B stands |
| G2 | the interval of S lies below +3 points | H7-B dead on this file |
| G2 | otherwise | not decided |
| G3 | the interval of the difference of shares, late-repaired minus early-repaired, lies above 0 | H7-D stands |
| G3 | it lies below +10 points | H7-D dead on this file |
| G3 | otherwise | not decided |

M1, M2 and G4 carry no rule: they describe. M1 is read against the predictions of §6.

A flip is not an error: G1 counts where a text leaves FP16's, not whether the answer is wrong. r_late and r_early
need not sum to 1.

## 6. Signed predictions (main session)

1. The grading gate passes: 90 %.
2. G1. r_late = 0.40 [0.25, 0.55] and r_early = 0.60 [0.45, 0.75]. The interval of r_late contains 0.306: 60 %; lies
   above it: 35 %.
3. G1. The control leaves FP16's text early: its median d is under 150 characters.
4. M1. The late arm reads beta 0.78 [0.70, 0.88] and sd_res 1.85 [1.60, 2.10]: its beta / sd_res is within 8 % of
   the control's 0.4095. The early arm reads beta 0.85 [0.72, 0.98] and sd_res 0.70 [0.45, 0.95].
5. M1, September slices: the 24-35 slice raises beta more than the 0-11 slice.
6. G2: not decided, 75 %.
7. G3: coverage at least 50 %, 70 %; if not void, not decided, 60 %.
8. G4: the two arms both repair 50 [35, 70] problems.

## 7. Not established by these readings

- Any cause. A hazard and a slope describe arms that already ran.
- Anything on the 8B, the 14B, a second calibration draw, or a task but MMLU and GSM8K.
- That a flip costs a point.
- Any mechanism. No encoder is changed.
