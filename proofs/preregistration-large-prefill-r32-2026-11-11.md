# Prereg: a12ipg32, the chain prefill at 32 known tokens a pass, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-r32-2026-11-11`, from commit 34d5a83, once the code it describes existed.
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-large-prefill-r32-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11, card spending inside a **$25** cap; **$12.08** spent before this job (relayed by the
orchestrating session). Cost: about **$0.63** on l40sx1 (*estimated*, "Cost"), timeout 30 min, **$0.90** at worst,
**$12.98** at worst after it.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp. Design A of `docs/mesures/large-prefill-rangees-etude-2026-11-11.txt` (branch
`claude/arbre-nuit-2026-11-10`, 2881ea3): a12ipg's body compiled at RC = 4, so a pass carries 32 known tokens where
a12ipg and a12ip carry 16. The Mac's gates and the attention tile's SASS: `docs/mesures/large-prefill-r32-mac-2026-11-11.txt`.

## The arm

`a12ipg32` (`llvq-cuda/kernels/pers_4et_a12g32.cu`, `TreeForm::A12ipg32`, selected only by its name): three symbols,
each `pers_4et_a12g_body<FORM, 4u, true>`, appended after a12ipg's module text, which keeps its bytes. RC sets the n8
tiles of each lane's accumulators and the row guard (R > 8 RC aborts). Nothing else changes:

- Row n of a projection is column n of the MMA's B and D; no other row enters it. The slices are the plan's
  [8, 10, 8, 20, 6], never R's, and each tile folds its slices in order.
- The attention is a12ipg's: a split that ends at or below pos0 is read once for a group of 4 rows, the others once a
  row. D1 measured that a group job gives a row job's bits.
- Every row buffer on the card is already sized for 32 rows (0.153 GB); the area is a12ipg's, 93,568 B a CTA. No
  buffer grows; b/param of the file (2.73) does not move.

Module text at the card's tile define: 451,917 bytes, sha256 a35f2b83. The modules of record keep their bytes:
4E tree ce1c354d, a12ip 63155df9, a12ipg 246e99b1 (pinned by `pers::tests::the_d1_tree_modules_keep_their_bytes`).

## Mac gates (before the stamp)

- Compile probe, sm_89, devel ptxas 12.4: 168 registers, 728 B stack (8 B spill stores, 48 B spill loads), 93,568 B
  shared, under 170 registers, 744 B and 101,376 B. No fallback to 512 threads.
- Host twin (`llvq-cuda/tests/tree_verify_host.rs`, the 4B's widths, two layers, random caches, release):
  `a12ipg32_chain_prefill_equals_r16_and_r1`: 45 tokens from 500 at R = 32 equal a12ip's R = 1 path (46 rows) and
  a12ipg's R = 16 (46 rows, with the 4B's parts and the 8B's); 300 tokens from 250, across 256 and 512, at R = 32
  equal a12ipg's R = 16 (301 rows). Argmax, logits, residual, every layer's K and V, the hand-over row included.
- `a12ipg32_mutants_are_killed`: tiles 2 and 3 stored in tiles 0 and 1's columns, their accumulators not zeroed, the
  body at RC = 2 while 32 rows are sent; one site each, each killed.
- The earlier a12ip, a12ipg, chain and refusal twin gates, the 159 `pers.rs` tests, clippy at zero new warnings.

## Setup

One image, one job, `ops/jobs/large-prefill-r32.sh` in its default form (two arms), on l40sx1 only. Output
`large-prefill-r32-2026-11-11/` in `hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs as D1: the sealed 4B (sha256
886391a8), `configs/qwen3-4b-tetra-e4-4e-v9.json`, `ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35, 44 prompts),
`ops/data/spec-long-8k-v1.jsonl` (sha256 da4edbd6, 8,191 tokens), cut to 1,024 and 4,096 the same way.

One `prefillbench` process, `--form chain --arm a12ipg32@8.10.8.20.6 --r 32 --against a12ip@8.10.8.20.6/16`, both arms
installed beside 4E (the same words read in place, no copy), under D1's two samplers. The phase sampler's tags carry
the arm and the length (`time-<l>-<arm>`, one window a round; `phases-<arm>`). In order:

1. Timing at 1,024, 4,096 and 8,191 tokens: 1 discarded + 3 timed rounds, each round a12ip at 16 then a12ipg32 at 32,
   prefill then 256 tokens of 4E greedy with no stop, tokens identical across rounds or refused. Time to the first
   token, prefill to the hand-over, decode tok/s after the first token, median [range] per arm; the ratio a12ip over
   a12ipg32 round by round; whether the two arms' 256 tokens are identical.
2. Identity at the three lengths, a12ipg32 at R = 32 against a12ip at R = 16: argmax, last row's logits bit for bit,
   every layer's K and V over the prompt bit for bit (FNV-1a a layer).
3. Phase split of each arm: the 8,191 prompt prefilled once more at its own R, the last three passes stamped
   (`pers::p4t_phases_a9`). Pass = the sum of the phases, attention = phase `attn`.
4. The 47 texts (the three long prompts' first rounds, the 44 house prompts) through both arms, paired in the process.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, the config and both prompt sets at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. The loader's module lines: 4E tree 306,060 bytes ce1c354d, a12ip 422,018 bytes 63155df9, a12ipg32 451,917 bytes
   a35f2b83.
4. If the image fails to build, nothing is launched. If the a12ipg32 module fails to build or to fit a CTA on a SM,
   the process refuses (arm void) and the journal says so.

## Signed prediction

The operator's session's signed predictions, verbatim (every figure is for the sealed 4B on l40sx1, chain prefill,
a12ipg32 at R = 32 against the reference a12ip at R = 16):

1. Identity: a12ipg32 R = 32 equals a12ip R = 16 bit for bit at 1,024, 4,096 and 8,191 tokens (argmax, last row's logits, K and V of 36 layers); the 256 decoded tokens identical; texts 47 of 47.
2. Reference a12ip R = 16 first token: 0.70 / 4.31 / 13.36 s, each within 1.5 % of D1's medians.
3. a12ipg32 first token at 1,024: 0.50 s [0.44, 0.60].
4. a12ipg32 first token at 4,096: 2.95 s [2.65, 3.40].
5. a12ipg32 first token at 8,191: 8.3 s [7.5, 9.3].
6. A 32-row pass at 8k outside attention (summed phases, median of the stamped full passes): 11.4 ms [10.5, 13.5].
7. Attention of a 32-row pass at 8k (same median): 33.5 ms [29, 37].
8. Phase peaks at 1,024 / 4,096 / 8,191: 2.318 / 2.855 / 3.560 GB, each within +0.05 GB.
9. Decode after the hand-over: within 1 % of the reference arm's at each length.

Rule: the arm PASSES for speed if the identity holds at the three lengths AND the first token at 1,024 is at most 0.60 s AND the first token at 8,191 is at most 9.3 s. If the identity fails at any length the job is VOID for speed and the arm is out. If the speed rule fails, read the phase split: outside attention above 13.5 ms means rows 17 to 32 cost more than the study's slope; attention above 37 ms means the 32-row step does not hide latency.

Operational reading, fixed here: first tokens are each arm's medians of the three timed rounds. "A 32-row pass at 8k"
is one of the three stamped passes of step 3 that carries 32 rows (the last pass of the 8,191 prompt carries 31 and is
read beside them, not in the median); "outside attention" is the sum of its phases less phase `attn`, "attention" is
phase `attn`, each the median over those full passes. D1's medians for prediction 2 are 0.702 / 4.306 / 13.361 s.
"Phase peak at a length" is the phase peak of a12ipg32's timing tag at that length. "Decode" is the decode tok/s
median after the first token.

## Decision rule

| result | outcome |
|---|---|
| identity holds at the three lengths, first token at most 0.60 s at 1,024 and at most 9.3 s at 8,191 | PASSES |
| identity holds, either first-token bound missed | FAILS for speed; the phase split says which part (outside attention above 13.5 ms, attention above 37 ms) |
| identity fails at any length | VOID for speed, the arm is out |

## Known limits

- a12ipg at R = 16 is not rerun. D1's 0.731 / 3.904 / 10.262 s stand as the comparison of record for the grouped
  read at 16 rows. Any reading of a12ipg32 against them is a cross-job comparison and is labelled so.
- Rows 17 to 32 have never been timed on a card.
- The ptxas of the card's driver can differ from the devel image's (168 registers, 728 B there); the loader prints the
  card's figures.
- Not established: a served path (no engine or chat door calls the chain prefill); R above 32; the full-tile attention
  path the SASS points at; 8B, 14B; any card but the L40S.

## Cost

D1 ran 1,142 s with three tree modules (*measured*). This job compiles three: the 4E tree module, a12ip, a12ipg32. The
first job loaded the first two in 419.5 s; a12ipg32 about 370 s more (*estimated*: its Mac nvcc compile 177 s against
138 s for a12ip and 158 s for a12ipg, whose card loads took about 254 and 339 s). Load about 13 min, timing about
2.5 min, identity about 0.7 min, phases about 0.5 min, 94 texts about 2 min, oracle, controls and container start
about 2 min. About 21 min, **$0.63** at $1.80 an hour (*estimated*). Timeout 30 min: **$0.90** at worst; **$12.98**
of $25 at worst after it.
