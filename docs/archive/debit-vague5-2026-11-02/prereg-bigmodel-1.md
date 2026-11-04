# DRAFT, not stamped, not committed

# Preregistration. Gate K32: our sealed 14B against the best published 2-bit rung, then a 2-bit 32B against a 4-bit 14B

**Written on 2026-11-02 by a read-only session, for the operator's review; revised the same day after an adversarial
review (findings in `wave5-deep/bigmodel.md` §10). To be committed and TIMESTAMPED BEFORE the first question is
scored.** No go given. This is a **local measurement run** on the operator's Mac: it needs an explicit go (CLAUDE.md
rule 1). It also reopens rival GGUF arms, which the operator dropped on 2026-09-19 after two failed card attempts
(`docs/mesures/iq2m-2026-09-19.txt`), so it needs that ruling too. Cost: **$0**. Mac GPU time, *estimated* (§8):
stages P0 and A about 4-7 h, stage B 3-5 h more and only on its condition. Downloads 43.0 GB, plus 11.6 GB for stage B.
Spent to date: $248.84 over 237 rows of the registry (*computed*, `docs/data/jobs.csv`).

## 1. Why this run exists

ROADMAP §4 holds the 32B behind "a gate on the drop in the 14B-to-32B gap". That gate was never written, and the 32B
chain costs about $70-190 (*estimated*, `wave5-deep/bigmodel.md` §4). The operator's ML audit of 2026-11-02 asks for
rivals at the same rate at 14B before any 32B or MoE encode.

Two questions, in this order, because the second is worth asking only if the first comes out one way:

- **D_ours**: our sealed 14B (2.73 b/param) against the best published 2-bit rung of the same model, unsloth UD-IQ2_M
  (2.939 b/param). Same model, same questions, paired. The plan resolves it to about ±1.8 pp at 95 % (§7).
- **P32 against AWQ 14B + 1.0**: a 2-bit 32B is a product point only if it beats what the same memory buys at 4 bits,
  a 14B in AWQ (9.97 GB, 5.40 b/param) or Q4_K_M (9.00 GB, 4.876 b/param) (*computed*). P32 = R32 + D_ours projects
  our 32B from the rung's 32B score R32. On the plan this comparison resolves only to about ±3.8-4.0 pp at 95 %
  (*estimated*, §7), against a predicted margin of 1.7 pp: **the plan most likely cannot separate A from B**. The rule
  of §7 says so in advance and names the way out.

Our side needs no run. On the 2,280-question plan, the dumps already in the repository give 14B f16 78.97, AWQ 78.21
and sealed 75.85 micro (*computed* on 2026-11-02, §2).

## 2. What the repository already establishes

- `ops/gguf_mmlu.py` scores a GGUF through `llama-server` with the decision rule of `bin/mmlu`: the argmax of the four
  letter logprobs. It takes its prompts from `ops/vllm_score.py`, checks every qhash, and refuses a score if one
  question lacks a letter in its top 100. It sends the prompt as text, so `llama-server` tokenizes it.
- **Engine gate at 4B**: llama.cpp f16 70.36 against our 70.32 micro (*measured*,
  `docs/mesures/m3-iq2-metal-2026-08-30.txt`). Recomputed from `docs/data/mmlu-dumps/`: paired +0.04 pp, SE 0.08, with
  9 of 2,280 picks different (*computed*, house formula below). That run used a local F16 conversion of
  `Qwen/Qwen3-4B@1cfa9a72`, which is no longer on disk; the checkpoint is still in the local HF cache.
- **The plan.** `mmlu-4b-f16.csv` and `mmlu-14b-f16.csv` share their 2,280 keys and qhashes, and both end on
  `# end fingerprint=65dcd53655e8bfa5 questions=2280`. All 2,280 appear with the same qhash in the sealed 14B's
  full-split dump (`docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv`) and in the AWQ
  full-split dump (`docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-awq-FULL.csv`), both census runs with
  fingerprint `a74a6d6213602979`.
- **References on the plan**, micro with the house stratified SE (`bin/mmlu` `micro_stderr`: p(1−p)/(n−1) with the
  finite population correction 1 − n/N), *computed* by `wave5-deep/calc/bigmodel/k32_numbers.py`: f16 78.97 (1.19),
  AWQ 78.21 (1.20), sealed 75.85 (1.23); paired sealed − AWQ −2.36 (0.92), 247 discordant. The full split reads 78.88,
  78.12 and 75.66 (*measured*).
- **What `mmlupair` needs.** Every dump it reads must end on `# end fingerprint=… questions=…`
  (`llvq-llm/src/bin/mmlupair.rs:185-196`). `ops/vllm_score.py` `emit_dump` (:375-412), which `gguf_mmlu.py` calls,
  writes no such line, so today every GGUF dump is refused, `mmlu-4b-gguf-f16.csv` included. A census dump against a
  plan dump needs `--intersect` (:326-331), which waives the run fingerprint and never the per-question qhash.
- **The 2026-09-19 failure.** The card job scored for 9 minutes against a server whose health probe answered before
  the model was loaded. Here, readiness requires one real completion carrying `top_logprobs` (§3).

## 3. What the commit carrying this file must hold

Otherwise no run.

1. `ops/vllm_score.py` `emit_dump` writes the trailer `# end fingerprint=<fp:016x> questions=<n>`, where fp is
   `token_fingerprint` over the concatenated token ids of the prompts in plan order, exactly what `bin/mmlu` hashes
   (`mmlu.rs:785, 854`). `gguf_mmlu.py` computes fp before scoring and refuses to start when the reference dump carries a
   trailer for the same question count and the two fingerprints differ. Draft: `wave5-deep/code/bigmodel/emit_dump_trailer.py`.
2. `ops/k32_project.py` (draft `wave5-deep/code/bigmodel/k32_project.py`), checked before use: on the existing dumps it
   must reproduce `mmlupair --intersect <awq census> docs/data/mmlu-dumps/mmlu-14b-f16.csv` (estimate and SE, finite
   population correction on) within 0.05 pp.
3. The driver script (draft `wave5-deep/code/bigmodel/k32.sh`) and the pilot helper (draft `k32_pilot.py`).

**Fallback, on the operator's word only.** If `ops/vllm_score.py` is not to change, `k32_project.py` becomes
authoritative for every quantity, after the check of item 2, and `mmlupair` is not used.

## 4. What runs

On the M3 Max (64 GiB, 378 GiB free, llama.cpp 10050 `b15ca938a` at `/opt/homebrew/bin`, read on 2026-11-02), one arm
at a time, the machine otherwise idle, under `nice -n 10`.

Per arm:

1. Download the file into its own new directory, at the revision recorded below. Check its size and its sha256 against
   the table **before** the server starts. A file whose sha256 differs is not scored, whatever the revision says.
2. `llama-server -m <file> -ngl 99 -np 4 -c 8192 --port 8080`, the settings of the 2026-08-30 engine gate, where
   `-np 4` needed a total context of 8,192 for few-shot prompts.
3. Ready when `/health` answers and a probe completion returns `top_logprobs`.
4. `uv run ops/gguf_mmlu.py --arm <arm> --label <repo@rev/file> --reference-dump <plan> --out <dump>`. Its
   dependencies go through `uv run`, never the global conda environment.
5. Stop the server. Keep the dump, the server log, the wall time and the provenance file.

| arm | file | bytes | sha256 (LFS) | b/param | role |
|---|---|---|---|---|---|
| P0 | `unsloth/Qwen3-4B-GGUF` `Qwen3-4B-BF16.gguf` @`22c9fc8a` | 8,051,285,536 | `f59dbdf0a013…` | 16.013 | pilot: the instrument on today's build, and the time anchor |
| E14 | `unsloth/Qwen3-14B-GGUF` `Qwen3-14B-BF16.gguf` @`a04a82c4` | 29,543,424,160 | `1d12adb9b56b…` | 16.004 | the engine at 14B, against our f16 |
| R14 | `unsloth/Qwen3-14B-GGUF` `Qwen3-14B-UD-IQ2_M.gguf` @`a04a82c4` | 5,425,956,224 | `3c4aa2f82f9f…` | 2.939 | the rival rung at 14B |
| R32 | `unsloth/Qwen3-32B-GGUF` `Qwen3-32B-UD-IQ2_M.gguf` @`931c8406` | 11,553,366,688 | `6d78bba94fa7…` | 2.821 | the rival rung at 32B, stage B only |

Sizes, LFS sha256 and revisions: *measured* from the Hub API on 2026-11-02 (full digests in
`wave5-deep/downloads/bigmodel-hub-2026-11-02/`). b/param is file bytes over 4,022,468,096, 14,768,307,200 or
32,762,123,264 parameters, tables included (*computed*, rule 6). E14 is the BF16 file, not Q8_0: Q8_0 would add its
own answer flips to the engine check and make gate 3 a test of two things at once. Optional arms, on the operator's
word only: R14b `Qwen3-14B-UD-IQ3_XXS.gguf` (6,013,732,224 B, `346f4dcf…`, 3.258) and R32b `Qwen3-32B-Q4_K_M.gguf`
(19,762,150,048 B, `8df67573…`, 4.826, the 32B's own 4-bit level).

**Stages.**

- **P0, pilot.** 100 questions: every 23rd row of `mmlu-4b-f16.csv`, rows 0 to 2,277, which covers all 57 subjects.
  The 100-row reference file is written outside the repository by `k32_pilot.py`.
- **A.** E14, then R14, each on the 2,280-question plan (`--reference-dump docs/data/mmlu-dumps/mmlu-4b-f16.csv`).
- **B.** R32 on the plan, **only if** gate 3 passed and the lower 95 % bound of D_ours is above 0. Cases C, D and V
  (§7) do not depend on R32, so the run stops after stage A in those cases.

References, no run: ours-14B and awq-14B are the census dumps above, subset to the plan by (subject, index) with the
qhash checked; f16-14B is `docs/data/mmlu-dumps/mmlu-14b-f16.csv`.

## 5. Gates

- **P0 (instrument).** Four letters in the top 100 for 100 of 100, and at most 3 of the 100 picks differ from
  `docs/data/mmlu-dumps/mmlu-4b-gguf-f16.csv` (llama.cpp, 2026-08-30). Expected about 0.4-1 difference: the 4B gate saw
  9 in 2,280 between engines (*computed*); at a rate of 1 in 100, three or fewer differ with probability 0.98
  (*computed*, Poisson). If P0 fails, nothing else runs; the cause goes into `-ECARTS.md`.
- **Gate 1, each arm.** `gguf_mmlu.py` prints qhash 2,280/2,280 and four letters in the top 100 for 2,280/2,280.
- **Gate 2, each arm.** Revision, bytes and sha256 are in the provenance file before the first question, and the
  bytes and sha256 equal the table's.
- **Gate 3, engine at 14B.** Both: |E14 − f16-14B| ≤ 0.6 pp, paired micro (`mmlupair`); and at most 46 of the 2,280
  picks differ (2 %, against 9 at 4B). If either fails, D_ours and every other comparison between engines is void.
- **Gate 4, each arm on the plan.** The dump's trailer fingerprint equals `65dcd53655e8bfa5`, the plan's.

Gates 1, 2 and 4 void the arm. Gate 3 voids the cross-engine statements.

## 6. Outcomes reported, never gated

- Micro and macro of every arm, with the house stratified SE. Wall time per arm, and seconds a question.
- **D_ours** = ours-14B − R14, paired micro: `mmlupair --intersect <ours census> <R14 dump>` is authoritative, and
  `k32_project.py` cross-checks it.
- **Size** = R32 − awq-14B and R32 − R14: two models each, so SEs in quadrature (METHODE §3, `mmlupair` does not pair
  two sizes).
- **P32** = R32 + D_ours, and **Δ = P32 − (awq-14B + 1.0)** with SE(Δ) = √(SE(R32)² + SE(awq-14B)² + SE(D_ours)²).
  P32 assumes our edge over the rung carries from 14B to 32B at the same rate offset. That is a projection: it informs,
  never pronounces.
- Any llama.cpp speed only in GB/s, never as a ratio to our engine (rule 5).

## 7. Decision rules

They inform the operator. A kill is the operator's. [lo, hi] is the 95 % interval of D_ours from `mmlupair`.

After stage A:

- **V (gate 3 fails):** no statement about our format; stage B does not run. The engine gap is written up first. The
  operator may still order R32 alone for the llama.cpp-only size question R32 − R14.
- **D:** hi < 0. The rival rung beats our sealed 14B. Reported as such; stage B does not run; the quality axis comes
  before any size.
- **C:** lo ≤ 0 ≤ hi. Our sealed 14B does not separate from the rival rung. Stage B does not run; quality at 14B first.
- **Stage B runs:** lo > 0.

After stage B, with [Δlo, Δhi] the 95 % interval of Δ:

- **A:** Δlo > 0. Propose writing the 14B→32B gap gate and pricing the 32B chain (encode, row scales, censuses, served
  speed), each line against `jobs.csv`.
- **B:** Δhi < 0. Our edge holds at 14B, but the projected 32B misses the claim plan D set (AWQ 14B + 1.0). No 32B
  encode on this evidence. The 27B (gate K of ROADMAP-27B) or the 14B carries the quality axis.
- **U, unresolved on the plan:** otherwise. The choice is the operator's, among three:
  1. R32 on the full split (14,042 questions, about 6.2 times its plan time). Δ then keeps only D_ours's sampling
     error, about ±1.8 pp, since the census references and a census R32 carry none under the house convention (a
     census interval closes to a point, `mmlupair.rs:531-536`).
  2. R14 and R32 on the full split, about 6.2 times both plan times. Δ is then exact on the split, and only the
     transfer assumption remains.
  3. Stop here and record U.

**Otherwise**, an arm voided by gate 1, 2 or 4: rerun once after the cause is written beside this file in
`-ECARTS.md`. No decision is drawn from a partial set.

**Resolution, stated before the run.** D_ours: about ±1.8 pp at 95 % (paired SE 0.92 for sealed − AWQ on this plan,
*computed*; a rung with more discordance could reach ±2.2). Δ: about ±3.8-4.0 pp (*estimated*: SE(R32) taken as the
f16 14B's 1.19, SE(awq-14B) 1.20, SE(D_ours) 0.92-1.15). A needs P32 above about 83.0, B below about 75.4, against a
predicted 80.9. Holding the threshold fixed instead of measured would give ±2.9-3.2; METHODE §3 counts it.

## 8. Cost

$0. Mac GPU time, *estimated*, no journal supports it yet, and P0 is what anchors it: P0 10-20 min; E14 2-4 h; R14
1.5-2.5 h; R32 3-5 h. Stages P0 and A: about 4-7 h with downloads of 8.05 + 29.54 + 5.43 = 43.0 GB. Stage B: 3-5 h
and 11.6 GB more. The largest file, 29.5 GB, sits inside 64 GiB of unified memory with an 8,192-token cache of
1.3 GB (*computed*: 40 layers × 8 KV heads × 128 × 2 × 2 B a position). The GPU is busy for the duration, so stages
run overnight; disk use peaks at 43 GB if no file is deleted between arms. Optional arms add 25.8 GB and 4-7 h.

## 9. Signed predictions

All *estimated*.

| quantity | point | interval |
|---|---|---|
| P0 picks different from the 2026-08-30 llama.cpp dump | **1** | [0, 3] |
| E14 micro | **78.9** | [78.4, 79.5] |
| E14 picks different from f16-14B | **20** | [9, 46] |
| R14 micro | **73.0** | [68.5, 76.5] |
| D_ours | **+2.9**, lo > 0 | [−0.7, +7.4] |
| R32 micro | **78.0** | [75.0, 80.5] |
| R32 − AWQ 14B | **−0.2** | [−3.2, +2.3] |
| P32 | **80.9** | [77.0, 84.0] |
| case after stage B | **U**, unresolved on the plan | A or U |

Reasoning. BF16 holds the checkpoint's own weights, so E14 should sit within a few tenths of our f16; the 4B gate saw
9 picks move between engines, and a second build and dtype add a few. Our sealed 14B loses 3.12 pp to f16 on this
plan. The unsloth IQ2_M rung carries 0.21 b/param more than ours. At 4B our bare Tetra beat unsloth's UD-IQ2_XXS by
15.77 points while carrying 0.27 b/param more (`iq2m-2026-09-19.txt`), and 2-bit losses shrink with size, hence a
positive D_ours with a wide interval. For R32: the Qwen3 report's base models gain 2.56 MMLU points from 14B to 32B, which on our 78.97 gives a
32B f16 near 81.5 on this plan, minus a 2-bit loss of 2-6 points. With Δ predicted at +1.7 and a half-width near 3.9,
U is the most likely case.

## 10. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 11. What this run does not establish

- Our 32B's MMLU. P32 is a projection resting on D_ours transferring from 14B to 32B.
- A versus B, unless Δ is far from its prediction: the plan resolves Δ to about ±3.9 pp.
- Anything about the 27B: that is gate K of ROADMAP-27B, which needs the Qwen3.5 letter ids first.
- GSM8K, long context, speed, and any card.
- The full split, unless the operator orders it after U.
