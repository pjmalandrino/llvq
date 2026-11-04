# Preregistration. The 4B under the grouped config: GSM8K through the kernel, paired with the expanded census, and the 57-question MMLU smoke under both configs

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04 for the first wave of the throughput validation roadmap, under a $10 HF cap for that wave. Cost: about **$2.1** on l40sx1, *estimated* in §8; timeout 150 min, so **$4.50** at worst (*computed*). Cap for this wave: **$10** (operator, 2026-11-04), shared with the other jobs of the wave, whose own preregs carry their costs. Spent before it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus $3.19 for five billed jobs the registry misses (*computed* on their measured durations at $1.80/h: `6aaf9e00`, `6aafac81` and `6aafc951` of 2026-09-20, `6aae7666` of 2026-09-19, `6ab087f2` of 2026-09-21), so $252.03 billed before this run (*computed*).

## 1. Why this run exists

The 4B has served the grouped attention since 2026-11-02, and no task score has run under it. The published 63.37 MMLU and 82.49 GSM8K are the expanded form's (ETAT §5): 82.49 ran through the served kernel, 63.37 on the dense reconstruction of the same file with f16 tables (ETAT §2). ETAT §5 says: "This is the debt the switch took on, and it is the first thing to settle on this axis."

- GSM8K, expanded config, through the served kernel: 1,088 of 1,319, 82.49 %, 8 stops at the cap, 316.1 tokens on average (*measured*, `docs/mesures/gsm8k-wave1-2026-09-26.txt`, run 1, dump `docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/gsm8k-4b-served.jsonl`).
- Run 1 ran on the image of commit `5333ac8` (Space sha `97a2b62a`). Since then the GSM8K path gained 1,228 lines and lost 84 (`git diff --numstat 5333ac8 06e1602` over `model.rs`, `kvq.rs`, `fused_cuda.rs`, `fused.rs`, `served.rs`, `rotplan.rs`, `bin/gsm8k.rs`), the three trims of 2026-10-30 among them. The trims are exact by construction and were checked on 256 tokens after a 5-token prompt (*measured*, `trims-contexts-2026-10-30.txt`, gates 4 and 5). Nothing has checked them at GSM8K lengths, so this run replays 50 problems under the expanded config before it pairs anything with run 1 (§3 step 2).
- Run 0 and run 1 of 2026-09-26 wrote the same 50 completions, field for field once the timings are removed (*computed* from the two dumps). The served decode is deterministic from one job to the next on one image.
- The grouped form is not bit for bit on the decode path: its largest logit difference over 49,140 teacher-forced steps was 13.13 (*measured*, `gqa-ppl-2026-11-01.txt`). A greedy generation of about 316 tokens can part from the expanded one, so GSM8K can move.
- On the prefill path the grouped form read +0.0000 % in perplexity, bit for bit on 12 windows of 4,096 (*measured*, same journal). MMLU scores by the prefill logits of the answer letter. Its served arm runs the grouped branch too: `model.hidden` builds a fresh cache and goes through the same attention code (`bin/mmlu.rs:800`, `model.rs:1575-1585`, `model.rs:1648-1658`). A 5-shot prompt holds up to 3,096 tokens (`model.rs:723-724`), and the identity was measured at 4,096 only. The smoke checks it on MMLU's own prompts.
- A GSM8K loss here would argue against extending the form to 8B and 14B, which is why `prereg-gestures-2.md` §7 suspends its R1 proposals on this run.

## 2. What the commit holds

No engine code. `bin/gsm8k` and `bin/mmlu` read `LLVQ_CONFIG` and apply the config's store and attention form through `Served::apply` (`bin/gsm8k.rs:117`, `:186-189`; `bin/mmlu.rs:531`, `:579`; `served.rs:297-307`). Neither refuses the dump variables, `LLVQ_DATASET_REV` or `LLVQ_MMLU_ALLOC` beside it: `served.rs:156-176` refuses `LLVQ_KERNEL_DIR`, `LLVQ_TILE_BLOCKS` and a contradicting served variable, `bin/gsm8k.rs:123-131` adds `LLVQ_KV_PREALLOC`, and `bin/mmlu.rs:538-547` the restorations. The binaries the job calls, `oracle`, `gsm8k` and `mmlu`, are in the build list and in the runtime `COPY` of `ops/Dockerfile.cuda` (`:52-54`, `:131-153`), and the job adds no `COPY`.

The job is a new script, `ops/jobs/gsm8k-grouped.sh`, in the shape of `ops/jobs/served-grouped.sh`. `ops/jobs/gsm8k-served.sh` keeps its bytes, so the three arms of the 2026-09-26 campaign stay what they ran. What the new script does differently from `gsm8k-served.sh`:

- The sealed 4B as in `gsm8k-served.sh` `4b/served`, `TIMEOUT=150m`, and `PREREG` this file.
- Both configs are read from `configs/`, each checked to end in exactly one newline. Then comes the python check of `served-grouped.sh:51-57`: the five served fields equal in both, `attention` absent from `qwen3-4b-tetra-e4.json` and `"grouped"` in `qwen3-4b-tetra-e4-grouped.json`. Both go into `PRE` with their sha256.
- Output `/out/gsm8k-4b-grouped-2026-11-04`. The old script fixes the date at 2026-09-26 (`gsm8k-served.sh:44`).
- Both configs are written on the mount under their own names and checked by sha256. The old script writes its one config as `qwen3-$SIZE-tetra-e4.json` (`:57`). A grouped config written under that name would mislabel the dump, whose header names its config by path and carries no attention key (`bin/gsm8k.rs:250-257`).
- Dumps and logs: `gsm8k-4b-expanded-n50.jsonl` for the replay, `gsm8k-4b-grouped.jsonl` for the census, `mmlu-4b-expanded-door.csv` and `mmlu-4b-grouped-door.csv`, each with its `.txt` log. The old script fixes the dump name at `gsm8k-$SIZE-served.jsonl` (`:109`).
- The replay digest of §4 gate 4 goes into `PRE`. The launch script recomputes it from `docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl` and refuses to launch when it differs, as `served-grouped.sh:58-60` does for its reference line. `python3` is in the runtime image (`ops/Dockerfile.cuda:102-104`).
- The GSM8K commands take `LLVQ_CONFIG`, `LLVQ_GSM8K_DUMP` and `LLVQ_DATASET_REV`, as `gsm8k-served.sh:109` does. The MMLU commands take `LLVQ_CONFIG`, `LLVQ_MMLU_ALLOC=flat` and `LLVQ_MMLU_DUMP`, as `served-8b.sh:171-172` does, and not `LLVQ_DATASET_REV`. That one variable pins every corpus, and the GSM8K sha does not exist in `cais/mmlu` (`corpus.rs:28-35`).
- `mmlu <file> cuda 1` scores one question a subject, 57 in all. The limit is the third argument (`bin/mmlu.rs:495`), and the flat plan takes `min(population, limit)` a subject (`bin/mmlu.rs:216-218`). The 8B door printed "allocation: flat, 1 per subject, 57 questions" (*measured*, `docs/mesures/served-8b-2026-09-21-brut/mmlu-door.txt`), and `configs/README.md:115-117` calls it the served door.
- `DRY_RUN=1 bash ops/jobs/gsm8k-grouped.sh` parses on the Mac before the launch, and recomputes the replay digest of gate 4 from the committed run-0 dump.
- The image. Space sha `f77f803e`, built from `f808856`, compiles the same `bin/gsm8k.rs`, `bin/mmlu.rs`, `model.rs`, `kvq.rs`, `served.rs` and `fused_cuda.rs` as HEAD `06e1602`. Inside the upload perimeter (`ops/run.py:747-758`), only `bin/fusedrun.rs` and `configs/README.md` changed since. The job runs on the image `IMAGE_SHA` names, and the journal records it.

## 3. What runs

One job, l40sx1, the image named above, the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), configs `qwen3-4b-tetra-e4.json` (`098fe49c…`) and `qwen3-4b-tetra-e4-grouped.json` (`2455fe85…`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **Expanded replay**: the 50 problems of the pilot's sample under the expanded config, `gsm8k <file> cuda 50` (`gsm8k-served.sh:10-12`). It checks that today's image writes run 0's 50 completions. If one differs, the job stops here (§4 gate 4).
3. **GSM8K, grouped config**, all 1,319 problems, zero-shot, no reasoning block, cap 1,024 new tokens, dataset revision `740312ad…`. Every setting is run 1's but the config and the image, and step 2 is what ties the two images.
4. **MMLU smoke**, 57 questions under the expanded config, then the same 57 under the grouped config, one dump each.
5. **After the job, on the Mac, $0:** `gsm8kpair <grouped dump> <run-1 dump>`, so that A − B reads grouped − expanded. Then `mmlupair` of the two MMLU dumps, and a byte comparison of their data rows, the lines that do not start with `#`. Then the median character of the first difference between paired completions, by a few lines of python over the two GSM8K dumps.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The object and both configs at their bytes and sha256 on the mount.
3. The `served config:` line, which both binaries print (`served.rs:243-260`; `bin/gsm8k.rs:141`, `bin/mmlu.rs:548`), reads `attention expanded on cat` in the replay and in the expanded MMLU log. It reads `attention grouped on grow(256)` in the GSM8K census and in the grouped MMLU log. `bin/gsm8k` prints the same pair from the model's own getters on its `device:` line (`bin/gsm8k.rs:195-201`). The MMLU dumps carry `# attention=expanded` and `# attention=grouped` (`bin/mmlu.rs:741`). All four logs print `tile 64 (served: measured optimum for sm_89)` and `int4: tv_q4_hh equals tv_q4_h bit for bit`, as every card job since 2026-10-30 gates (`served-grouped.sh:126-127`).
4. The replay dump ends on `"end":true` and on the pilot's fingerprint `94c7b98876288901`. Its rows hash to `8300f75e771ef30d4c2ee4a10968261cfaa99b1ba90f8fa072e9f36446940805`, the digest of run 0's 50 rows by the same rule (*computed*). The rule: drop the header line, drop `prefill_s` and `decode_s` from each row, write each row and the trailer with `json.dumps(..., sort_keys=True, ensure_ascii=False)`, join them by newlines, and take the sha256 of the UTF-8 bytes.
5. The census dump ends on `"end":true` and on the census fingerprint `bfa9135ce4b55c75` of run 1 (*measured*, its trailer). The old script checks the first only (`gsm8k-served.sh:112`).
6. Both MMLU dumps end on `# end fingerprint=<f> questions=57`, with the same `<f>` (`bin/mmlu.rs:481-483`).
7. `gsm8kpair` re-grades 0 rows of either Rust dump, as on 2026-09-26. The grader has not changed since run 1: `llvq-llm/src/gsm8k.rs` was last touched in `685b32a`.
8. The job's environment holds no `LLVQ_*` variable at its start (`gsm8k-served.sh:91`). Each command sets its own on its command line, never exported.

## 5. Outcomes reported, never gated

- Accuracy, stops at the cap, mean tokens, decode ms a token, of the grouped arm.
- The replay's decode ms a token: the expanded form's on today's image, at GSM8K lengths. It is read beside the grouped arm's and never divided, because the two arms ran different problem sets and `gsm8k` timings price a campaign (`bin/gsm8k.rs:427`).
- The paired difference grouped − expanded in points, its 95 % interval, McNemar discordant counts and exact p, from `gsm8kpair`.
- Where two completions part: the median character of the first difference.
- MMLU: the 57 picks under each config, the count of identical picks, and whether the dumps' data rows are identical bytes.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| expanded replay: completions identical to run 0 | **50 of 50** | |
| GSM8K accuracy, grouped | **82.5 %** | [81.0, 84.0] |
| paired difference grouped − expanded | **+0.0 pp** | [−1.0, +1.0] |
| discordant problems, both directions | **25** | [8, 70] |
| McNemar exact p | **≥ 0.05** | |
| stops at the cap | **8** | [3, 15] |
| decode ms a token at GSM8K lengths | **7.7** | [7.3, 8.3] |
| MMLU smoke: identical picks | **57 of 57** | [56, 57] |
| MMLU smoke: dump rows identical bytes | **no**, *estimated* at even odds | |

Reasoning, *estimated*. The grouped decode moves rounding only, so flips should balance. Its teacher-forced top-1 disagreed on 104 of 49,140 steps (*measured*, `gqa-ppl-2026-11-01.txt`), so roughly half of 316-token completions should part somewhere (*computed*: 1 − (1 − 0.0021)^316 ≈ 0.49), yet a parted completion usually keeps its answer: two paths of the same model that should agree, our FP16 dense path and vLLM's, disagreed on 20 problems of 1,319 (*measured*, engine gate of 2026-09-26), and the Metal dense dump of the sealed file on 0 of 50 while 12 of 50 completions parted (*measured*, same journal). The 4B's 256-token reference text was the same under both configs (*measured*, `served-grouped-2026-11-02.txt`). The decode step under the grouped config: 7.43 ms at 5 positions and 8.78 ms at 2,048 (*measured*, the same journal's context sweep), so about 7.6 ms at the mean decode position of a GSM8K problem, about 250 (*computed* by linear interpolation). MMLU: the prefill path was bit for bit at 4,096 positions only. The same journal family measured grouped against expanded prompt logits at 3.711e-2 at 2,048 positions and 3.418e-2 at 8,192 (*measured*, `gqa-ab-2026-11-01.txt`), and a five-shot MMLU prompt is under 4,096 tokens, so the logits in the dump rows probably move by a few hundredths while the picks hold. The replay: the trims are exact by construction, and run 0 and run 1 already wrote these 50 completions on one image.

## 7. Decision rules

The criterion is the one the engine gate of 2026-09-26 used for two paths that should agree (`gsm8k-wave1-2026-09-26.txt`, "|delta| <= 1.5 pp and p >= 0.05"), applied to Δ = grouped − expanded. The G rules read Δ only when gate 4 passed, because that gate is what makes run 1 the expanded arm of today's image. G1, G2 and G3 cover every Δ: Δ = 0 means as many discordant problems each way, so p = 1 and G1 holds.

- **G1.** |Δ| ≤ 1.5 pp and McNemar exact p ≥ 0.05. The grouped config's GSM8K is settled at 4B. The grouped arm's score is proposed for ETAT §2 beside the expanded one, and the operator decides how the cards word it. The R1 proposals of `prereg-gestures-2.md` may go forward.
- **G2.** Δ < 0 and (|Δ| > 1.5 pp or p < 0.05). The grouped form loses on reasoning at 4B. The R1 proposals at 8B and 14B are suspended, and the operator decides whether the 4B stays grouped (the T8 grant of 2026-11-01 is revisited, the operator's ruling).
- **G3.** Δ > 0 and (|Δ| > 1.5 pp or p < 0.05). Reported as a difference in the form's favour; it suspends nothing, and the operator decides how it is worded.
- **M1.** The 57 MMLU picks are identical under both configs. The grouped form moves no pick on these prompts; whether the rows are identical bytes, and the largest logit difference, are reported. The switch adds nothing to what 63.37 measures. 63.37 stays a dense-reconstruction score with f16 tables (ETAT §2). The census through the kernel stays the open decision of ETAT §5 (about $20, *estimated*), for the operator to order.
- **M2.** Any pick differs. The grouped form moves MMLU picks on MMLU's prompts. No MMLU figure is quoted for the grouped config before a census through the kernel under it (about $20, ETAT §5).
- Otherwise, including any failed gate, the run is void, and a retry goes under this prereg with its deviation written beside it. When gate 4 fails, the job stops before the census, and the operator decides whether an expanded census on today's image runs beside a grouped one (about $2.1 each, *estimated*). A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Run 1 of 2026-09-26 billed 4,251 s, $2.13 (*measured*, `docs/data/jobs.csv`). Its loop took 4,206 s: 3,749 s of decode at 8.991 ms a token, and 456 s of prefill at 3.912 ms a prompt token over 88.3 prompt tokens a problem (*measured*, run-1 log; the split *computed*). Oracle, checks and load took the other 45 s (*computed*). The smoke of the same day read 8.961 ms a token on its 50 problems (*measured*).

- Grouped census: 1,319 × 316.1 tokens at 7.7 ms take about 53.5 min of decode (*computed* on an *estimated* step), plus 7.6 min of prefill at run 1's rate.
- Expanded replay: about 2.7 min, run 0's 160 s of decode and prefill (*measured*, its dump).
- MMLU smoke: two loads and 114 five-shot prompts, about 6 min. The 8B door took 184 s for its 57 (*measured*, `served-8b-2026-09-21-brut/mmlu-door.txt`), and a 4B prompt token costs less.
- Oracle, checks and loads: about 1 min.

In all about 71 min, **$2.1**, and 68 to 75 min over the predicted interval of the decode step (*computed* on *estimated* inputs). Timeout 150 min, **$4.50** (*computed*).

## 9. What this run does not establish

- MMLU as a score under the grouped config: the smoke is 57 questions, the census is not run.
- MMLU through the kernel against the dense reconstruction: both smoke arms run the kernel, so they say nothing of 63.37's dense arithmetic or of its f16 tables.
- The expanded form's GSM8K on today's image beyond the 50 replayed problems.
- Any size but 4B, any context beyond GSM8K's lengths, any card but the L40S.
- Why a problem flips, if some do: the comparison is paired, not explained.

## 10. Deviations

Any departure from this file goes in ``proofs/preregistration-gsm8k-grouped-2026-11-04.md-ECARTS.md``, beside it, never inside it.
