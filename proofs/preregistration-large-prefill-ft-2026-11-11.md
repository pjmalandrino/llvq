# Prereg: a12ipg32ft, the attention tile's full-tile path on the chain prefill, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-prefill-2026-11-11`, from commit e66e2e0, once the code it describes
existed. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-large-prefill-ft-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11 evening ("ok pour ton job": the two jobs of the full attention tile), card spending
inside a **$25** cap; **$12.66** spent before these two jobs (relayed by the orchestrating session). Cost: about
**$0.84** on l40sx1 (*estimated*, "Cost"), timeout 30 min, **$0.90** at worst. With the 4E job
(`preregistration-large-attn-ft-4e-2026-11-11.md`, $0.75 at worst): **$14.31** at worst after both.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp and the 4E prereg's; one image serves both jobs. The build and its Mac gates:
`docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt`. The lead: `docs/mesures/large-prefill-r32-2026-11-11.txt`
(a12ipg32 at 32 rows PASSES, its 8k pass 76.0 % attention) and section 2 of
`docs/mesures/large-prefill-r32-mac-2026-11-11.txt` (the tile serialised by its nv guards).

## The arm

`a12ipg32ft` (`TreeForm::A12ipg32Ft`, selected only by its name): a12ipg32's module text with one edit
(`pers.rs::pers_ft_edit`). `llvq-cuda/kernels/pers_attn_full.cuh` is inserted before `pers_attn_tile`, and at its
head `if (nv == PERS_TT) { pers_attn_tile_full(...); return; }`. On a full tile of 8 positions the body has no guard
between positions; partial tiles keep the guarded loop. Each score, the tile max, each ex2, ts and each PV sum keep
their operands and their order. Everything but the text is a12ipg32's: R = 32, the plan's slices, the area, the row
buffers. No buffer grows; b/param of the file (2.73) does not move.

Module text at the card's tile define: 457,150 bytes, sha256 19b734a3. The modules of record keep their bytes:
4E tree ce1c354d, a12ip 63155df9, a12ipg32 a35f2b83 (pinned by `pers::tests::the_d1_tree_modules_keep_their_bytes`
and `the_ft_modules_keep_their_bytes`).

## Mac gates (before the stamp)

- SASS, devel ptxas 12.4 sm_89: the full-tile path is 552 instructions with 1 branch region, against 731 and 34 for
  a12ipg32's guarded loop; the 8 score chains are interleaved; the ex2 calls are branch-free but serial.
- Host twin: a12ipft, a12ipgft and a12ipg32ft equal their guarded parents bit for bit (argmax, logits, residual, every
  layer's K and V, the taps); two mutants killed on the tree side. The 15 tree twin tests pass.
- `the_ft_modules_are_their_parents_with_the_full_tile_branch`: each ft text with the edit reverted is its parent's.
- Clippy at zero new warnings, Mac and Linux.

## Setup

One image, one job, `ops/jobs/large-prefill-ft.sh` in its default form, on l40sx1 only. Output
`large-prefill-ft-2026-11-11/` in `hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs as R32: the sealed 4B (sha256
886391a8), `configs/qwen3-4b-tetra-e4-4e-v9.json`, `ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35, 44 prompts),
`ops/data/spec-long-8k-v1.jsonl` (sha256 da4edbd6, 8,191 tokens), cut to 1,024 and 4,096 the same way.

One `prefillbench` process, `--form chain --arm a12ipg32ft@8.10.8.20.6 --r 32 --against a12ip@8.10.8.20.6/16
--against a12ipg32@8.10.8.20.6/32`, the three arms installed beside 4E (the same words read in place, no copy), under
R32's two samplers. In order:

1. Timing at 1,024, 4,096 and 8,191 tokens: 1 discarded + 3 timed rounds, each round a12ip at 16, a12ipg32 at 32,
   a12ipg32ft at 32, prefill then 256 tokens of 4E greedy with no stop, tokens identical across rounds or refused.
   Time to the first token, prefill to the hand-over, decode tok/s after the first token, median [range] per arm;
   the ratios round by round.
2. Identity at the three lengths of a12ipg32 and of a12ipg32ft against a12ip at R = 16: argmax, last row's logits bit
   for bit, every layer's K and V over the prompt bit for bit (FNV-1a a layer).
3. Phase split of each arm: the 8,191 prompt prefilled once more at its own R, the last three passes stamped
   (`pers::p4t_phases_a9`). Pass = the sum of the phases, attention = phase `attn`.
4. The 47 texts (the three long prompts' first rounds, the 44 house prompts) through the three arms, each paired with
   a12ip's in the process.

Dropped to fit 30 minutes: a12ipg at 16 (D1's arm), the eager arm and the perplexity. Every arm is held bit for bit
to a12ip, which D1 gated against the eager prefill and scored on WikiText-2.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, the config and both prompt sets at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. The loader's module lines: 4E tree 306,060 bytes ce1c354d, a12ip 422,018 bytes 63155df9, a12ipg32 451,917 bytes
   a35f2b83, a12ipg32ft 457,150 bytes 19b734a3.
4. If the image fails to build, nothing is launched. If the a12ipg32ft module fails to build or to fit a CTA on a SM,
   the process refuses (arm void) and the journal says so.

## Signed prediction

The operator's session's signed predictions, verbatim (sealed 4B, l40sx1, chain prefill; a12ipg32ft at R = 32 against
a12ipg32 at R = 32 and the reference a12ip at R = 16):

1. Identity: a12ipg32ft and a12ipg32 each equal a12ip R = 16 bit for bit at 1,024, 4,096 and 8,191 (argmax, last row's logits, K and V of 36 layers); the 256 decoded tokens identical; texts 47 of 47.
2. a12ipg32 first token within 1.5 % of job large-prefill-r32's medians 0.448 / 2.681 / 7.800 s.
3. a12ipg32ft first token at 1,024: 0.42 s [0.39, 0.45].
4. At 4,096: 2.25 s [2.00, 2.55].
5. At 8,191: 6.0 s [5.0, 7.1].
6. Attention of a 32-row a12ipg32ft pass at 8k (median of the stamped 32-row passes, summed phases): 22 ms [15, 29].
7. The same pass outside attention: 10.8 ms [10.4, 11.4].
8. Phase peaks 2.318 / 2.855 / 3.560 GB, each within +0.05 GB.
9. Decode after the hand-over within 1 % of the other arms'.

Rule: PASSES for speed if the identity holds at the three lengths AND a12ipg32ft's first token at 8,191 is at least x1.15 sooner than a12ipg32's in this job, round by round, AND it is not more than 1 % later at 1,024. Identity failing voids the arm.

Operational reading, fixed here: first tokens are each arm's medians of the three timed rounds. "x1.15 sooner, round
by round" is the median over the three timed rounds of a12ipg32's first token over a12ipg32ft's, in the same round,
formed from the per-round first tokens the process prints for each arm (it prints the ratio only against a12ip);
"not more than 1 % later at 1,024" is the same median ratio at 1,024, at least 0.99. "A 32-row pass at 8k" is one of
the three stamped passes of step 3 that carries 32 rows (the last pass of the 8,191 prompt carries 31 and is read
beside them, not in the median); "outside attention" is the sum of its phases less phase `attn`, "attention" is phase
`attn`, each the median over those full passes. "Phase peak at a length" is the phase peak of a12ipg32ft's timing tag
at that length. "Decode" is the decode tok/s median after the first token.

## Decision rule

| result | outcome |
|---|---|
| identity holds at the three lengths, a12ipg32 / a12ipg32ft first token at least x1.15 at 8,191 and at least x0.99 at 1,024, round by round | PASSES for speed |
| identity holds, either bound missed | FAILS for speed; the phase split says which part moved |
| identity fails at any length for a12ipg32ft | VOID for speed, the arm is out |

Nothing is served by this job: no engine or chat door calls the chain prefill.

## Known limits

- The ptxas of the card's driver can differ from the devel image's (552 instructions and 1 branch region there); the
  loader prints the card's registers and spills.
- The latencies behind the estimates (2,900 to 1,117 cycles a tile call) are assumed, not measured.
- The full-tile path has never been timed on a card.
- a12ipg32 against R32's medians is a cross-job comparison (prediction 2) and is labelled so; the rule reads only
  this job's rounds.
- Not established: a served path; 8B, 14B; any card but the L40S.

## Cost

Four NVRTC modules at load: 4E tree about 165 s, a12ip about 254 s, a12ipg32 about 370 s, a12ipg32ft about 380 s (its
Mac compile 183 s against 177); load about 1,170 s (*estimated*, from D1's 758.6 s for its three and the Mac compile
ratios; R32 measured 789.3 s for the first three). The rest about 470 s for three arms (from D1's 383 s for two).
About 28 min, **$0.84** at $1.80 an hour (*estimated*). Timeout 30 min: **$0.90** at worst. Spent before both jobs
$12.66; this job $0.90 and the 4E job $0.75 at worst: **$14.31** of $25 at worst after both.
