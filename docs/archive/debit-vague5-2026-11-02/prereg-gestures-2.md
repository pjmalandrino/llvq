DRAFT, not stamped, not committed. Wave-5 deep dive "gestures", 2026-11-02, revised after review. The job script it names is `code/gestures/gqa-size.sh`, rebuilt line for line from `ops/jobs/gqa-ab.sh` and `ops/jobs/gqa-ppl.sh`; the "to fill" items come from the commit that carries it, before the stamp. Proposed path: `proofs/preregistration-gqa-size-YYYY-MM-DD.md`.

# Preregistration. The grouped attention at 8B and 14B: speed from 5 to 8,192 tokens of prompt, and the paired perplexity of the tolerance gate

**Written on YYYY-MM-DD, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: none yet, one go a size, and the 14B only after the 8B landed in R1 or R2 (§3). Cost: about **$1.38** for the 8B and **$2.31** for the 14B on l40sx1, *estimated* below (§8); timeouts 90 and 120 min, so $2.70 and $3.60 at worst. Cap for this wave: **to fill before the stamp**, the operator's. METHODE §6 puts a cap in every wave's prereg, none is in force, and ETAT §5 says one is owed before the next paid job; $6.30 covers both worst cases of this prereg. Spent before it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus $3.19 over five billed jobs of 2026-09-19 to 09-21 that the registry misses (*computed* on their measured durations at $1.80/h, `hf jobs ps -a`), plus the run of `prereg-gestures-0.md` if it goes first, as asked in §1.

## 1. Why this run exists

At 4B the grouped attention decodes at 77.5 tok/s at 8,192 positions against 33.1 for the expanded form, ×2.34, and ×1.10 after a 5-token prompt (*measured*, `docs/mesures/gqa-ab-2026-11-01.txt`). Its paired WikiText-2 perplexity read +0.0000 % by prefill and +0.0095 % [−0.0048; +0.0238] by decode, inside the ±0.12 % band (*measured*, `gqa-ppl-2026-11-01.txt`), and the 4B has served it since 2026-11-02 (ETAT §2). The 8B and the 14B have never run it.

- The tolerance gate was granted for the 4B only, on 2026-11-01. Each size needs its own control, which this run is, and its own ruling.
- At 14B the expanded form is expected near 21.5 tok/s at 8,192 positions (*estimated*, `gestures.md` §3.4), close to the product's 20 tok/s chat floor. That floor is a batch-1 chat floor defined for the target card, a 5090, not the L40S. The L40S figure is a marker beside it, never a ratio to it (rule 5; `docs/archive/note-produit-2026-08-13.md` A5 quotes L40S figures the same way, and its batch of 8 belongs to the other floor, 60 s a document in extraction).
- `Grow` and `Prealloc` truncate by moving a length, and `Cat` copies the kept prefix (`model.rs:421-446`); of the two O(1) stores, only `Grow` reserves no window (`Prealloc` holds the 32 expanded heads, 4.83 GB at 8k at 4B, *computed* in the handover of 2026-11-01, so 9.61 b/param over the whole model, *computed* on its 4,022,468,096 parameters, `docs/fiche-4b.md`). That makes the grouped form on `Grow` the precondition of an O(1) speculative rollback, noted here and not used.
- **Order.** ETAT §5 ranks the 4B's task debt under the grouped form first on this axis: no task score has run under it. The operator is asked to run `prereg-gestures-0.md` (GSM8K through the kernel under the grouped config, about $2.0, *estimated* on the expanded arm's $2.13, *measured*) before this run, and §7 suspends this run's R1 proposals on its outcome.
- No engine code is needed: `LLVQ_GQA_AB` and `LLVQ_GQA_PPL` read the file, the tokenizer and wikitext-2, and nothing in them is 4B-specific (`fusedrun.rs:337-436`, `:535-620`); `grouped_attention` takes any divisor of the head counts (`model.rs:509-517`, `:571-593`): groups of 4 at 8B, 5 at 14B.

## 2. What the commit holds

- `ops/jobs/gqa-size.sh`, `SIZE=8b|14b`: the bodies of `gqa-ab.sh` and `gqa-ppl.sh` kept line for line in one job, with the refusal block, the `PRE` printf carrying every variable the body reads, the config written by the job and checked by sha256, a `DRY_RUN` branch that runs `bash -n`, and the `run.py bench` call. Only the size case is new, in the shape of `paper-served.sh`. `SIZE=14b` at launch also asks for `GQA_8B_VERDICT=R1|R2`.
- In `model.rs`, `attention_forms_agree` gains a head layout of 10 over 2, a group of 5, the 14B's shape, which no test runs today.
- The image is built from a commit whose engine code, tests aside, is the served one: `f808856` (the served image `f77f803e`) or HEAD `06e1602`, which differ by one refusal message on a path this job does not take. The group-of-5 test is a host test and changes no binary. An image that carries the audit's fusions (norm and rotation, `tv_tetra48_seg`) is another object, and gate 4 would void the run at step 2.

*Measured* on the Mac (to fill): `SIZE=8b DRY_RUN=1` and `SIZE=14b DRY_RUN=1` both parse; the group of 5 passes under the three forms and the stores, f32 and f16, within the bound the test already holds for `Grouped`; one mutant (the 10/2 layout fed as 8/2) killed. The wave-6 check of 2026-11-02 already ran both `DRY_RUN` paths of the draft read-only against HEAD `06e1602`: both job scripts parse, `SIZE=4b` is refused, and four in-memory mutants (a reference hash, two dense lines, an object path) are refused before any spend (*measured* on 2026-11-02 by the wave-6 check, outside the repository; the commit's own run replaces it).

## 3. What runs

Two jobs from one image, one a size, l40sx1.
- 8B: `sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin`, config `qwen3-8b-tetra-e4.json`.
- 14B: `sealed-14b-2026-09-23/qwen3-14b-sealed.bin`, config `qwen3-14b-tetra-e4.json`, launched only after the 8B landed in R1 or R2. Under R3 or a void 8B, the 14B waits for the operator, who may still order it: the 8B is the cheaper probe, $1.38 against $2.31 (*estimated*).
- Each object at its bytes and its sha256 on the mount, against committed measurements: 2,815,098,745 and 5,087,000,541 B (*measured*, `sealed-8b-27-2026-09-24.txt:12`, `sealed-8b-14b-2026-09-23.txt:12`, and again on the mount in `ppl-scelles-2026-10-04.txt:23`), sha256 `7bdb9a55…` and `61db37fe…` as paper-table runs 2 and 3 read them on the mount (their committed `files.sha256`). The Mac no longer holds either file, so nothing is hashed locally, as in `ops/jobs/ppl-scelles.sh` of 2026-10-04.
- Each config without an `attention` field, written by the job and checked by sha256.

Each job:
1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed.** `fusedrun <file> 256` with the served flags spelled out (`tetra48`, `ROT_SHARE=1`, `FUSE=0`, `KV=f16`, `EMBED=q4`), against the dense arm, as runs 2 and 3 of the paper table.
3. **The speed A/B.** The same flags plus `LLVQ_GQA_AB=5,2048,4096,8192`, 256 new tokens, cudarc's events off.
   - Three arms in one model: `cat+expanded` (published), `grow(256)+shared`, `grow(256)+grouped`.
   - For each length: one prefill an arm in chunks of 4,096, outside every clock; the logits of the prompt's last position and of the first decode step, compared bit for bit with the published arm's.
   - Then 1 discarded and 5 timed decodes of 256 tokens an arm, every arm in every round, each cache truncated back to the prompt before each decode. Speed is `255 / t`.
4. **The control.** The same flags plus `LLVQ_GQA_PPL=4096,12`. Twelve wikitext-2 windows of 4,096, the published arm against `grow(256)+grouped`, by the prefill path and by a teacher-forced decode path in lockstep, paired window by window, 95 % t intervals.

The timed rounds of the `Grow` arms reuse buffers their first decode grew. In service, one chunk reallocation copies the cache once every 256 tokens, about 0.1 % of a decode token at 8,192 (*computed* at 4B). At 14B the prefill's score tensors peak near 8.05 GB at 8k (*computed* in the operator's ML audit of 2026-11-02), beside 5.04 GB of weights and the three arms' caches, which step 3 keeps resident together, about 4.2 GB at 8,448 positions (*computed*: 163,840 B a position). That is about 17.3 GB, inside the L40S's 46 GB (*computed*).

**Option, the operator's before the stamp.** The 4B's isolated logit moves of up to 13.13 have no reference: nobody has measured how far the published path's own decode drifts from its own prefill. Half a day of engine code in `run_gqa_ppl` (keep the published arm's prefill logits of the window on the card, 2.5 GB in f32 at 4,096 × 151,936, and compare each decode step against its row) would print that reference here at no extra run. Default: not taken, and this run then needs no engine code.

## 4. Gates, each one voids that size's run

The script enforces every one of them (`code/gestures/gqa-size.sh`).

1. The image builds, and `oracle` prints MATCH.
2. The object at its bytes and sha256 on the mount; the config written from the launch command and its sha256 equal to the Mac file's.
3. Steps 2, 3 and 4 each print `tile 64 (served: measured optimum for sm_89)` and `int4: tv_q4_hh equals tv_q4_h bit for bit`; no `LLVQ_*` variable is in the job environment.
4. Step 2 prints the embedding line with `2 tables (model.embed_tokens.weight + lm_head.weight)` and exactly one `  fused F0:  ` line. 8B: `fused F0: 256 tokens identical to the dense arm`, and the line hashes to `323cabeb8520cba9cede5ea6c1a611a1ceb5322c0aae473fac63ed911d61e498`. 14B: `WARNING: fused F0: divergence at token 78 out of 256`, and the line hashes to `186974aa69fef4b5611d3fd7598b3f517bb1e35ae78142da836ee75db3cfeabe`. Both are the lines of paper-table runs 2 and 3 (*computed* on 2026-11-02 from their brut files; the launcher re-derives them before it spends).
5. Step 3 prints `cudarc event tracking: off (one stream, ordered by issue)`, four `prompt N` headers, four `cat+expanded` speed lines, and for each of `shared` and `grouped` four logit lines and four speed lines carrying `tokens identical` or `differs from token`; `run_gqa_ab` refuses to print if an arm returns a different text in two rounds.
6. Step 4 prints the event-tracking line, the token fingerprint `(tokens 3f1baca9033bf251)` of the 4B's scored stream, 12 window lines, both summary lines, and a harness ratio, decode over prefill on the published arm, inside [0.99, 1.01]. The fingerprint carries: `ppl` printed it for all three sealed files, each read with the tokenizer it carries (*measured*, `ppl-scelles-2026-10-04.txt:27`), and `fusedrun` reads that same tokenizer (`fused_cuda.rs:2903`).

## 5. Outcomes reported, never gated

- For each unexpanded arm and length: its logits bit for bit or the largest difference, prompt then first decode step; its text identical, or the first token that differs.
- Decode tok/s for every arm and length, median and range; arm / published, round by round.
- Perplexity of both arms by both paths, the paired intervals, the decode top-1 agreement, and the largest logit difference over every step.
- The published arm's numbers are the first measurement of the 8B and 14B after the trims.
- Step 2's speed against the dense arm, which `fusedrun` prints, is not reported: this job has no same-head arm, and rule 4 forbids the raw ratio alone.

## 6. Signed predictions

All *estimated*: the 4B's measured per-length increments for the expanded and grouped forms, identical attention shapes at 8B, and at 14B scaled by bytes a position (×1.35 expanded, ×1.11-1.25 grouped, whose floor is 40 layers over 36 on the same 8 key heads); the short step from the served speeds (95.0, 57.2, *measured*) minus the trims and launch removals scaled by layer, int4 and launch counts. Step 2 sits 1.3 % under the A/B's published arm at 5 tokens, as at 4B (120.3 against 121.9, *measured*, `gqa-ab-2026-11-01.txt`).

| quantity, 8B | point | interval |
|---|---|---|
| step 2, fused tok/s | **98** | [92, 104] |
| published decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **99 / 71 / 55 / 31** | [93, 105] / [64, 78] / [49, 61] / [28, 35] |
| grouped decode tok/s at the same four | **107 / 94 / 82 / 68** | [100, 114] / [85, 102] / [74, 90] / [60, 75] |
| grouped / published at 5 / 2,048 / 4,096 / 8,192 | **1.08 / 1.32 / 1.51 / 2.17** | [1.03, 1.14] / [1.15, 1.50] / [1.30, 1.75] / [1.70, 2.60] |
| published prefill ppl through the kernel | **9.71** | [9.60, 9.82] |

| quantity, 14B | point | interval |
|---|---|---|
| step 2, fused tok/s | **58.5** | [54.5, 62.5] |
| published decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **59 / 45 / 36 / 21.5** | [55, 63] / [40, 50] / [31, 41] / [18, 25] |
| grouped decode tok/s at the same four | **63 / 57 / 52 / 45** | [59, 67] / [51, 62] / [46, 58] / [39, 51] |
| grouped / published at 5 / 2,048 / 4,096 / 8,192 | **1.06 / 1.27 / 1.46 / 2.09** | [1.02, 1.12] / [1.10, 1.45] / [1.25, 1.70] / [1.60, 2.55] |
| published prefill ppl through the kernel | **8.44** | [8.34, 8.55] |

| quantity, both sizes | point | interval |
|---|---|---|
| grouped / published − 1, prefill | **0.000 %** | [−0.02, +0.02] |
| grouped / published − 1, decode | **+0.01 %** | [−0.04, +0.05] |
| both intervals inside ±0.12 % | **yes** | |
| decode top-1 agreement | **99.7 %** | [99.0, 99.95] |
| largest logit difference over every step | **10** | [1, 40] |
| grouped text at 5 tokens | **identical** | |
| token fingerprint | **3f1baca9033bf251** | |

The dense references are 9.7062 at 8B and 8.4440 at 14B (*measured*, `ppl-scelles-2026-10-04.txt`); the 4B read 12.5822 through the kernel against 12.5834 dense. The 4B's largest logit difference, 13.13, missed its own prediction by a factor of a hundred; it is reported, not gated.

## 7. Decision rules, size by size

- **Precondition, from `prereg-gestures-0.md`.** R1's proposal below goes to the operator only once the 4B's GSM8K under the grouped config has run and landed in G1 or G3. Under G2 (a loss outside the engine-gate criterion), every R1 proposal of this run is suspended, and the operator decides. If that run has not happened, R1 is recorded and proposed when it lands.
- **R1.** Both paired intervals lie inside [−0.12 %, +0.12 %], and every grouped / published range lies above 1.000 at all four lengths. The extension of the tolerance gate to this size is proposed to the operator. On a yes, a follow-up commit writes `configs/qwen3-<size>-tetra-e4-grouped.json`, and a served-door sweep under its own prereg measures the served object.
- **R2.** Both intervals lie inside the band, but at some length the grouped / published range does not lie entirely above 1.000. The form is not served at that size, the run is a curve point, and a kernel timeline comes before any further step on this path at that size.
- **R3.** At least one interval is not entirely inside the band, by either path. That size stays expanded, and the operator decides whether fresh windows are worth a second control. An interval that contains zero says the control lacked power; one that does not says the form moves quality at that size.
- **The 14B job** launches after an 8B R1 or R2 only; under R3 or a void 8B it waits for the operator.
- **Task debt.** Under R1 and a yes, GSM8K through the kernel at that size (about $2.2 at 8B and $3.9 at 14B, *estimated* on $2.37 and $4.24, `jobs.csv`, under the rules of `prereg-gestures-0.md`) is owed before the config is quoted with a task score.
- R1, R2 and R3 cover every outcome of a run whose gates passed. Otherwise, that is when a gate of §4 failed, that size's run is void, and a retry goes under this prereg with its deviation written beside it.
- A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. The same arithmetic gives about 32 min for the 4B's `gqa-ppl` job, which took 27.4 min (1,643 s, *measured*): it runs about 15 % high, so the figures below lean high.

- 8B: container, pull and oracle 5 min. Step 2 about 3.5 min (dense load 121 s, *measured* in paper-table run 2). Step 3 about 8.5 min: prefills of 0, 11, 22 and 45 s an arm, then 18 decodes of 255 tokens a length. Step 4 about 29 min: prefill path 12 × 2 × 4,096 rows at about 5.4 ms, decode path 49,140 steps × 25 ms for both arms. In all about 46 min, **$1.38**; timeout 90 min, **$2.70**.
- 14B: 5 min, then step 2 about 6.7 min (dense load 264 s, *measured* in paper-table run 3). Step 3 about 14 min: prefills to 8,192 at about 9.5 ms a token, three arms. Step 4 about 50 min: prefill path at about 9.5 ms a row, decode path 49,140 × 42 ms. In all about 77 min, **$2.31**; timeout 120 min, **$3.60**.

Anchors: `gqa-ab-4b`, 575 s and $0.29; `gqa-ppl-4b-r2`, 1,643 s and $0.82 (*measured*, `docs/data/jobs.csv`); `paper-served-14b-r2`, 933 s and $0.47 (*measured*, `paper-table-2026-09-25.txt:14`; the registry holds 16 billed min). At 4B the A/B step took 420 s and the perplexity step 1,488 s, 124 s a window (*measured*, the `date -u` lines of `gqa-ab-2026-11-01-brut/job.log` and `gqa-ppl-2026-11-01-brut/job.log`). Scaled by the GSM8K prefill and decode costs a token (*measured*, `jobs.csv`), they give about 8.7 and 14.8 min for step 3 and 31 and 53 min for step 4 (*estimated*), inside both timeouts.

## 9. What this run does not establish

- Task accuracy under the grouped form at either size.
- Quality past 4,096 positions: the windows stop at the fused prefill's rows.
- The served door at either size, which waits for R1 and a config file.
- Metal, which refuses the form by name, and any other card.
- The size of rounding spikes inside the published path itself, unless the option of §3 is taken.

## 10. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.
