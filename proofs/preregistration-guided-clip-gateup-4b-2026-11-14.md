# Preregistration. The guided gate/up encode of the 4B, retried with clipped weights

**Written on 2026-11-14, BEFORE any clipped encode or score. NOT YET STAMPED.** The signed predictions (§8) are
filled by the main session, then the file is stamped (`ots stamp`) and committed before
`ops/jobs/guided-clip-encode-card-4b.sh` launches. Once stamped, this file is not edited; a deviation goes into
`-ECARTS.md` beside it.

Job scripts at writing: `ops/jobs/guided-clip-encode-card-4b.sh` sha256 `ad670322dd41d4b806db85ad775c3c0ab994b578205c4759dfdc72034e87bfa2`,
`ops/jobs/guided-clip-mmlu-4b.sh` sha256 `d1ac853fd1f9a0c7baeb93fb524de056ac771f1ec3c73453333ca637672da12f`. Encoder: `llvq-llm/src/guided.rs` and
`llvq-llm/src/bin/smoke.rs` at commit `cd70933` (the image is published from `8107e19`, which adds the job scripts and a `jobs.csv` fix), built into the private Space image
`Pier-Jean/llvq-runner-cuda-qualite` at sha `c0fb594fe50371bc0171de58167dfe7e1ed6f9e6`.

Cost: **$4.38** (*estimated*, §7), ceilings **$4.81** for the encode and **$1.02** for the MMLU arm, $5.83 in all,
inside the $5.89 left of the $15 of O12 ($9.11 spent before it).

## 1. Where this comes from, and the forking path it is

The first attempt (`preregistration-guided-gateup-4b-2026-11-13.md`, journal
`docs/mesures/guided-gateup-4b-2026-11-13.txt`) ended in LOSS: gate_proj and up_proj under gradient-weighted input
Hessians, G = 4, read 57.11 on the full MMLU split against 58.62 for the baseline, paired −1.51 [−2.30, −0.70].

After that result, and not preregistered, the weights were read: the effective sample size of a row group's weights is
1.58 % of the 131,072 calibration tokens on average, about 2,070 tokens, under the 2,560 inputs of the matrix; the top
1 % of tokens carry 46 % of the mass (journal, POST HOC DIAGNOSTIC).

This retry is chosen after seeing a result. What limits the damage:
- One value of one knob, fixed here before any clipped encode exists: the clip is 10 times the group mean. It was
  read from the weight files alone (`weights-diagnostic/`, and the table below), never from a score.
- No sweep. No second value runs under this prereg, whatever the outcome.
- The baseline is the 2026-11-13 one, already scored. Its dump does not move.

Clip levels, *computed* on the 72 weight files at G = 4 (one pass, threshold = c × the mean before clipping):

| clip | effective share, mean | worst matrix | worst group | mass kept, mean |
|---|---|---|---|---|
| none | 1.58 % | 0.16 % | 0.01 %, 13 tokens | 100 % |
| 50 × | 7.2 % | 2.4 % | 1.5 %, 1,958 tokens | 78 % |
| 20 × | 12.0 % | 4.2 % | 2.6 %, 3,426 tokens | 70 % |
| **10 ×** | **17.3 %** | **6.3 %** | **3.9 %, 5,130 tokens** | **63 %** |
| 5 × | 24.4 % | 9.4 % | 5.7 %, 7,474 tokens | 56 % |
| 3 × | 30.8 % | 12.3 % | 7.3 %, 9,520 tokens | 49 % |

10 × is the loosest level in this table at which every group's effective sample is at least twice the 2,560 inputs.

## 2. The question

With each group's weights clipped at 10 times their mean, does the guided encode of gate/up move MMLU against the
2026-11-13 baseline? And does it move against the 2026-11-13 unclipped candidate, as the diagnostic's conjecture
predicts?

## 3. The objects

- Weights: the 72 files of `guided-weights-4b-2026-11-13/` in the bucket, manifest sha256 `e63dad83…`, unchanged. No
  statistic job runs.
- Candidate arm (`candidate-clip10`): the 2026-11-13 candidate's command and environment, on the same flavor
  (`rtx-pro-6000`), plus `LLVQ_GUIDED_CLIP=10`. In `guided::group_weights`, for each group, `w_t` is the f64 sum of
  the group's columns, `m` its mean over all tokens, and the weight becomes `min(w_t, 10·m)` before the mean-1
  normalisation. Its `.llvq.state` carries the 2026-11-13 candidate's `guided` line plus ` clip=10`.
- Baseline arm: `guided-encode-4b-2026-11-13/baseline/baseline.bin`, sha256 `1f4c38d8ae32…`, encoded on 2026-11-13 by
  the image at Space sha `37a8e933`. Not re-encoded. Its dump is
  `docs/data/mmlu-dumps/mmlu-guided-baseline-FULL.csv` (58.62, 8,232 of 14,042).
- Unclipped candidate, for the secondary reading: `docs/data/mmlu-dumps/mmlu-guided-candidate-FULL.csv` (57.11).

## 4. Controls

1. Code. With `LLVQ_GUIDED_CLIP` unset the encoder is byte-identical to the 2026-11-13 one: the stored digest test of
   the unset encode and the unchanged-`describe()` test pass, and the mutants of the clip are caught (agent report,
   quoted in §8's disclosure before the stamp).
2. Same calibration tokens: the encode prints fingerprint `45fb781f65574d4d`.
3. One variable against the baseline: the `.llvq.state` equals the baseline's but for the `guided` line, and that
   line equals the 2026-11-13 candidate's plus ` clip=10` (the script's `compare`). Same GPU line as the 2026-11-13
   encodes (NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB); another card model is recorded and the result
   is still read, since the encode is CPU and f64 on the host for everything the code depends on, but it is said.
4. Artifact: verified bit for bit by `smoke`, sealed to 1,794,564,765 B, sha256 different from both 2026-11-13 files.
5. Dump: `dtype=f16`, `limit=census`, flat allocation, `config=none`, dense reconstruction, `kv=f16`, end fingerprint
   `a74a6d6213602979` over 14,042 questions; `oracle` MATCH first on the card.

If controls 2 to 4 fail, the MMLU arm is not paid for. If control 5 fails, no result is read.

A limit that no control removes: the two arms are not encoded by the same image, and their MMLU arms are not scored
by the same image. The image differs by the clip code alone, which the unset path does not reach. The repository has
reproduced a full-split dump pick for pick across jobs before (ablation-qualite-4b, 14,042 of 14,042).
Re-scoring the baseline under the new image would cost $0.70 and does not fit the budget.

## 5. The statistic

Primary: the paired MMLU difference, candidate-clip10 minus baseline, micro on the full split, by `mmlupair --no-fpc`,
candidate first, with its paired-bootstrap 95 % interval. Secondary, descriptive: candidate-clip10 minus the
2026-11-13 unclipped candidate, same tool; and the WikiText-2 perplexity `smoke` prints.

## 6. Decision rules, written before the data

Primary, against the baseline:
- **GAIN** if the paired 95 % interval lies above 0.
- **LOSS** if it lies below 0.
- **NOT RESOLVED** otherwise.

Secondary, against the unclipped candidate: the concentration conjecture is **supported** if the interval lies above
0, **not supported** otherwise.

What follows each outcome, for this phase:
- GAIN opens a second calibration draw before anything else; it is not a recipe, and it needs budget this phase no
  longer has.
- NOT RESOLVED or LOSS closes the gradient-weighted input-Hessian form for this phase. A two-sided form (YAQA) is
  another idea, another cost note and another go.

In every case one calibration draw cannot attribute a difference to the objective (first prereg, §6), and nothing
here reaches the sealed object, which adds row scales and the q4 embedding.

## 7. Cost

- Encode: 80 min billed on `rtx-pro-6000` (*measured* once for the unclipped candidate, 4,812 s), $3.68 at $2.75/h
  (*computed*); timeout 105 min, ceiling $4.81.
- MMLU: one arm, 22.3 min of scoring (*measured* twice on 2026-11-13) plus setup, 23 to 24 min, $0.70 (*estimated*);
  timeout 34 min, ceiling $1.02.
- Total $4.38 estimated, ceilings $5.83. Queue unbilled. Spend of the phase before: $9.11 of $15.

## 8. Dated disclosure, and the signed predictions

What has run, as of writing, all on the Mac through `~/.llvq-guard/heavy.sh`, debug builds:
- The clip code (`cd70933`): `guided::group_weights` takes an optional clip; `LLVQ_GUIDED_CLIP` is read by `smoke`
  and recorded as ` clip=<c>` on the run-state `guided` line, which a resume compares as a string.
- Targeted tests: 4 `test result:` lines, 57 passed, 0 failed. `cargo test -p llvq-llm`: 88 lines, 560 passed, 0
  failed, 22 ignored. `cargo clippy --all-targets`: 0 warnings. The stored digest test of the unset encode
  (`8b107e9f…`) passes untouched, and the unclipped path equals, bit for bit at every G, a verbatim copy of the
  2026-11-13 function.
- Seven mutants of the clip, each at one counted site, each caught, each restored (sha256 checked): clip ignored in
  `group_weights`; `load` passing no clip; cap at the max instead of `c·m`; `min` turned into `max`; clipped but not
  renormalised (caught by the unit test only, since GPTQ is invariant to the scale of `H`); the state line dropping
  the clip; the parser accepting `c` under 1.
- `bash -n` and `DRY_RUN=1` of the two job scripts. `compare` and `pair` run offline against the 2026-11-13 raw files
  with a stand-in arm: the happy paths pass, a wrong clip value and a sealed file equal to the old candidate's are
  refused.
- The image: published to the private Space on 2026-11-14 at 09:27 UTC, building at the time of writing. The encode
  job refuses unless its `smoke` binary carries the string `LLVQ_GUIDED_CLIP`.
- The clip table of §1, from the weight files (`weights-diagnostic/clip.py`, added with this file).
Not run: any clipped encode of a real model, any score of one. No clipped file exists.

### Signed predictions (main session)

Signed by the main session on 2026-11-14, before the stamp and before the clipped encode. The lead's view: clipping
should undo most of the damage, because the damage looks like a thin estimate; whether what is left of the signal
beats a uniform Hessian is the open part, and the lead leans slightly against. The unclipped candidate's loss also
carries draw-like noise this pair cannot separate.

| quantity | prediction | interval or alternatives |
|---|---|---|
| paired MMLU difference, clip10 minus baseline (pp) | −0.3 | [−2.0, +1.5] |
| primary outcome | NOT RESOLVED | about 50 %; LOSS about 30 %, GAIN about 20 % |
| clip10 MMLU micro, full split | 58.3 | [56.6, 60.1] |
| paired MMLU difference, clip10 minus unclipped candidate (pp) | +1.2 | [−0.5, +3.0] |
| secondary outcome | supported | about 65 % |
| WikiText-2 perplexity, clip10 (`exact-ppl`, 12 × 4,096) | 16.9 | [16.3, 17.7] |
| billed encode time (min) | 80 | [74, 92] |
