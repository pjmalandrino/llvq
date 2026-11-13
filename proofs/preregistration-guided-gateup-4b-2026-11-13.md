# Preregistration. A first output-aware encode on the 4B: gate/up against gradient-weighted Hessians

**Written on 2026-11-13, BEFORE any statistic, encode or score. NOT YET STAMPED.** The signed predictions (§9) are
filled by the main session, then the file is stamped (`ots stamp`) and committed before
`ops/jobs/guided-weights-4b.sh` launches. Once stamped, this file is not edited; a deviation goes into `-ECARTS.md`
beside it.

Statistic script: `ops/guided_weights.py`, sha256 `73e98d165cf2b8783786c2a47e67c2aaf604eae344d419679f023b32f9bccd8f`
at writing; it imports `ops/layer_vs_output.py`, sha256
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12` (committed in 0079d04, unchanged). Read on
2026-11-13 after the pre-launch runs of §9. Job scripts at writing: `ops/jobs/guided-weights-4b.sh`
`69b809deb11c33c664f980842f089e16495cba4be1533b6c756b7fa76ced241c`, `ops/jobs/guided-encode-4b.sh`
`2fafe2a8850171f3ec2ec619abbb3449e05496782c66408b3fbf40b519f47b8a`, `ops/jobs/guided-mmlu-4b.sh`
`2708e4f1dd1b491577b94e1670ada0a2a59bef13fdace5b9ff32694d439688fb`. What has run is stated in §9. If any of these
files changes before the stamp, these lines are updated before the stamp; the statistic job refuses to launch unless
this file names the sha256 of both Python scripts.
Encoder: `llvq-llm/src/guided.rs`, `llvq-llm/src/calib.rs` (`quantize_model_guided`), `llvq-llm/src/bin/smoke.rs`,
`llvq-llm/src/bin/calibtokens.rs`, at the commit that carries this file.

Cost: **$1.65 to $1.89** in all (*estimated*, §8): statistic job ceiling **$0.90**, MMLU pair ceiling **$1.92**, the
two encodes on the Mac at $0. Inside the $10 of O12 ($0.44 spent before it).

## 1. The question

Does encoding `gate_proj` and `up_proj` of the 4B against Hessians weighted by the final-output loss move MMLU,
against the same encode made the usual way, on the same calibration tokens, the same machine and the same code?

Today every matrix of an activation is encoded against `H = Σ_t x_t x_tᵀ / N`. The candidate encodes gate/up against
`G = 4` Hessians each, one per group of consecutive output rows, `H_g = Σ_t w̃_tg x_t x_tᵀ / N`, where `w_tg` is the
sum over the rows of group `g` of the squared gradient of the next-token cross-entropy with respect to that row's
output at token `t`, on the FP16 checkpoint, and `w̃_tg = w_tg / mean_t(w_tg)`. This is GuidedQuant's objective
(arXiv 2505.07004, Eq. 4, §3.2, `g = 4` for 7B and 13B). Every other matrix, knob and byte of the format is unchanged.

Why gate/up: on the sealed 4B their restore recovers +3.38 MMLU of 6.77 (CERVEAU Q16), the final-output judge ranks
them first and the layer judge last (Q23). Why groups and not a weight per row: the encoder treats every output row
independently, so a per-row weight changes no code (cost note §1, gptq.rs 580-587).

## 2. The objects

- Checkpoint: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, shard sha256 as pinned in
  `ops/layer_vs_output.py`.
- Calibration tokens: DCLM-edu, 64 windows of 2,048 from the prefix, 131,072 ids, cut by
  `llvq_llm::guided::calibration_ids` (the function `smoke` itself now calls) and written by `bin/calibtokens` with
  their FNV-1a 64 fingerprint. The same file feeds the statistic job; both encodes print the fingerprint of the
  windows they calibrate on.
- Recipe, both arms: the 2026-09-18 DCLM base's (`proofs/preregistration-dclm-4b-2026-09-18.md` §1): `tetra1`,
  rotation `0x110feed`, `nogs`, `h_shrink` 1, `gain_scale` 1, damping 1e-2, `LLVQ_INT4_TYPES=v_proj`, f32, Metal,
  `--features metal,fast-linalg`. Command line (`ops/jobs/guided-encode-4b.sh`, reconstructed from that prereg and
  `ops/jobs/encode-14b.sh` lines 355-357; the 2026-09-18 line itself is not journalled):
  `LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS=12 LLVQ_ARTIFACT=<arm>.llvq
  smoke 64 2048 12 4096 metal nogs tetra1 999 rot`, then `seal`.
- Baseline arm: that command, guided variables unset.
- Candidate arm: the same, plus `LLVQ_GUIDED_WEIGHTS=<weights dir> LLVQ_GUIDED_GROUPS=4`. Its `.llvq.state` carries
  a `guided = gate_proj,up_proj G=4 manifest_sha256=…` line; the baseline's carries none.
- Order: baseline, then candidate, one at a time on the Mac under `~/.llvq-guard/heavy.sh`, never beside another
  heavy job. Both before row scales: this attempt scores the encoder, as the 2026-09-18 base was scored (57.95).

## 3. The statistic job

`ops/jobs/guided-weights-4b.sh`, `l40sx1`, image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, timeout 30 min,
output `guided-weights-4b-2026-11-13/`. Per window: one forward and one backward of the checkpoint in fp32 (weights
cast up, TF32 off), each layer under activation checkpointing, loss the summed next-token cross-entropy on the
window's own tokens. For each layer and each of gate/up, the gradient at the output is reduced to 16 columns of
squared-gradient sums over 608 consecutive rows; one `[131,072, 16]` little-endian f32 file per (layer, type), 72
files, and a manifest naming the token fingerprint, the shard sha256, the loss per window and every file's sha256.
The job runs the script's selftest first and stops unless it reads GREEN.

## 4. Controls and rule D

1. The statistic job: selftest GREEN; the token file's sha256 and fingerprint match its sidecar; the tokenizer sha256
   the token file names is the pinned one; 72 files; the zero weights are exactly the last position of each window
   (73,728 = 64 × 36 × 2 × 16); the mean loss per token is reported (no rule on it).
2. Both encodes print the same calibration token fingerprint, the one in the weights manifest (the candidate refuses
   otherwise, `GuidedWeights::open`).
3. The candidate differs from the baseline only by the variable: the two `.llvq.state` files are equal but for the
   `guided` line (the encode script prints the diff and stops otherwise, before any upload), and both logs pass the recipe check of `guided-encode-4b.sh` (the `encode-14b.sh` lines).
4. Both arms verify their artifact bit for bit (`smoke`'s `weights identical, bit for bit`), seal to the same byte
   count, and differ in sha256.
5. Both dumps carry `dtype=f16`, `limit=census`, flat allocation, `config=none`, `arithmetic=dense reconstruction`,
   `kv=f16` and the plan fingerprint `a74a6d6213602979` over 14,042 questions; `oracle` MATCH first on the card.
6. Rule D, descriptive only: the baseline's MMLU is set beside the 2026-09-18 base's 57.95. It may differ, and a
   difference decides nothing, for three reasons: model.rs and the encode's forward have moved since (about 30
   commits); the 2026-09-18 command line is reconstructed, not read; and the repository once failed to reproduce its
   own file (HISTORIQUE line 392). A difference beyond the encode-to-encode noise (SE 0.45 a pair) is recorded as a
   reproducibility finding, not as a failure of this attempt. Before the candidate runs, the encode script also prints
   whether the baseline's sealed file is EQUAL or DIFFERENT to the 2026-09-18 base (1,794,564,765 B, sha256 prefix
   `471f39883b0baabc42b90c83`, the only digits the repository records), for information only.

If controls 1 to 5 fail, nothing more is paid for and no result is read.

## 5. The statistic

Primary: the paired MMLU difference, candidate minus baseline, micro on the full split, by `mmlupair --no-fpc` on the
two dumps (candidate given first, so its Δ is candidate minus baseline), with its paired-bootstrap 95 % interval and
McNemar's test. Secondary and not part of this prereg's cost: KL to FP16 by `ops/layer_vs_output.py`-style scoring of
the two exports, the answer-logit fit (`logit_snr.py`) on the two dumps, and the two WikiText-2 perplexities `smoke`
prints (`exact-ppl`, 12 windows of 4,096).

## 6. Decision rules, written before the data

- **GAIN** if the paired 95 % interval of candidate minus baseline lies above 0.
- **LOSS** if it lies below 0.
- **NOT RESOLVED** otherwise.

In every case: one calibration draw cannot attribute the difference to the objective. Two encodes on one draw differ
by the objective and by how it meets this draw (draw sd 2.92 on 2,280 q before row scales, CERVEAU Q2). A GAIN says
these two files differ in that direction on this draw; it opens a second draw, not a recipe. Nothing here reaches the
sealed object: the sealed 4B adds row scales, the int4 swap and the q4 embedding on top of its base.

## 7. What does not compare

- The candidate against the sealed 4B (63.37) or its 2026-09-18 base (57.95): other code, other runs.
- Any perplexity or KL move against MMLU: perplexity has moved against a resolved MMLU pair here (Q5), and YAQA's own
  4-bit row improved perplexity while MMLU fell (literature note line 280).
- `G = 4` against other `G`: one arm only. The weights serve every `G`; another arm is another encode and another
  prereg.

## 8. Cost

- Statistic: 9 to 15 min billed on `l40sx1` (setup 6 to 10 min, 64 windows at 2 to 4 s), $0.27 to $0.45
  (*estimated*, the cost note's range); ceiling $0.90 (timeout 30 min at $1.80/h).
- Encodes: two Mac runs of 1 h 47 to 2 h each (*estimated* from dclm-4b line 4), $0.
- MMLU: two arms of 23 to 24 min on `l40sx1` (jobs.csv lines 159, 172, 197), $1.38 to $1.44 (*estimated*); ceiling
  $1.92 (timeout 64 min).
- Total $1.65 to $1.89 estimated, ceilings $2.82. Queue unbilled.

## 9. Dated disclosure, and the signed predictions

What has run, as of writing (logs in `docs/mesures/guided-gateup-4b-2026-11-13-brut/pre-launch/`), all on the Mac
through `~/.llvq-guard/heavy.sh`, debug builds with `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`:
- The unset-path digest re-measured at 8d6c664 from a `git archive` export: `8b107e9f…`, equal to the stored one.
- Targeted tests (llvq-llm lib, `guided`, `served_config`, `env_registry`): the two-block routing test failed on its
  own reference (a run without a sink narrows the tail differently, by one ulp); the reference now writes an
  artifact as the test run does; then 257 passed, 0 failed, 2 ignored.
- Rust mutants M1 to M7, each at one counted site, each caught, each restored (sha256 checked).
- Workspace `cargo test`: 223 `test result:` lines, all ok, 1,321 passed, 0 failed, 142 ignored.
- `cargo clippy --all-targets`: 7 warnings in this attempt's code (four `is_multiple_of`, two doc-list indentations,
  one collapsible `if`), fixed without a change of behaviour; then 0 warnings. `cargo test -p llvq-llm` after the
  fixes: 88 `test result:` lines, 553 passed, 0 failed, 22 ignored; mutants M1 to M7 rerun, all caught.
- `uv run ops/guided_weights.py --selftest`: first RED (the writer needed numpy, absent from the pinned environment;
  the finite-difference check drowned in the f32 RMSNorm); the writer now packs with `struct` and the check runs with
  a float64 RMSNorm; then GREEN, 19 checks. Mutants P1 to P3 each RED, file restored, sha256 checked.
- Release build of `smoke`, `seal`, `calibtokens`, `mmlupair` with `--features metal,fast-linalg`.
- The token file: `~/llvq-guided-2026-11-13/calib-tokens-4b.u32`, 524,288 B, sha256 `6abdaeee4dbe1a0c…`,
  fingerprint `45fb781f65574d4d`, 131,072 ids, tokenizer sha256 `aeb13307…` (the pinned one), DCLM-edu revision
  `dbad8ad7`.
- `DRY_RUN=1` of the three job scripts: the statistic and MMLU job scripts parse (the MMLU one with placeholder
  arms); the encode script passed its checkpoint, corpus and binary checks and refused on the absent weights
  directory, as it must before the statistic job.
Not run: the statistic job, any upload, the two encodes, any score. No statistic, encode or score exists.

Known before the data, from the cost note `docs/mesures/recherche-qualite-r2-chiffrage-encodeur-2026-11-13.txt`:
- One encode-to-encode MMLU pair on the full split carries an SE of 0.45 (dclm-4b line 17): about 0.88 points is the
  smallest difference a pair resolves (*computed*).
- Published evidence against expecting much: a third-party reading of YAQA on Qwen3-8B 2-bit puts it at 72.14 against
  QTIP's 72.04; at 4 bits YAQA's perplexity improved and MMLU fell; the repository already trains row scales on a
  final-output KL (+3.15 at 4B), which this attempt does not apply.
- GuidedQuant reports its gains on non-uniform scalar quantization of Llama-2 at 2 bits (MMLU +4.77 on 7B), with
  `g = 1` to `4` nearly equal in perplexity (its Table 13); none of it is on a lattice code or on Qwen3.
- The share of gate/up blocks whose code changes is not logged by the encoder; it is computable offline from the two
  `.llvq` files, by a reader not written at this date.

### Signed predictions (main session)

Signed by the main session on 2026-11-13, before the stamp, before the statistic job and before either encode. The
lead's view, stated so that it can be scored: the direction is expected positive, with little confidence in its size.
Changing gate/up's Hessians moves the input of every later layer, so the two encodes differ everywhere, and part of
their difference behaves like a new calibration draw. That part alone can exceed the 0.88 pp this pair resolves, in
either direction. The weights come from 64 windows only, where GuidedQuant used 1,024 sequences.

| quantity | prediction | interval or alternatives |
|---|---|---|
| paired MMLU difference, candidate minus baseline (pp) | +0.5 | [−2.5, +3.5] |
| outcome among GAIN, LOSS, NOT RESOLVED | GAIN | about 40 %; NOT RESOLVED about 33 %, LOSS about 27 % |
| baseline MMLU micro, full split | 57.9 | [56.0, 59.5] |
| candidate MMLU micro, full split | 58.4 | [55.5, 61.5] |
| share of gate/up blocks whose code changes (if computed offline) | 85 % | [60, 99] |
| WikiText-2 perplexity, baseline (`exact-ppl`, 12 × 4,096) | 16.25 | [15.8, 16.8] |
| WikiText-2 perplexity, candidate | 16.0 | [15.0, 17.2] |
| mean next-token loss of the FP16 checkpoint on the 64 windows (statistic job) | 2.35 nats a token | [2.0, 2.7] |

A prediction is "in" when the measured value falls inside its interval, or equals the named outcome. The count goes
into the journal.
