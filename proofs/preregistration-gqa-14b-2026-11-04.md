# Preregistration. The grouped attention on the sealed 14B: paired perplexity against the expanded form, and the speed of both forms against the context

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04 ("on envoie le plan"), for the 14B twin of the 8B control. The 8B control read rule R1 (`docs/mesures/gqa-8b-2026-11-04.txt`). The operator then granted the T8 at 8B on the condition of a light MMLU check, which passed, 285 of 285 picks identical, rule M1 (`served-grouped-8b-2026-11-04.txt`). Cap: the $10 HF cap of the first wave, of which $4.85 is billed, so $5.15 remains (*computed* from the registry). Cost: about **$2.15** on l40sx1, *estimated* in §8; timeout 130 min, so **$3.90** at worst (*computed*). Registry before it: $256.88 over 248 rows (*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The grouped attention passed its control at 4B and at 8B. At 8B it reads +0.0000 % by prefill and −0.0048 % [−0.0117 ; +0.0021] by decode, inside the ±0.12 % band of `docs/ETAT.md` §4. It decodes ×1.076 faster at 5 tokens and ×2.156 at 8,192 (*measured*, `gqa-8b-2026-11-04.txt`). The T8 grant covers the 4B and the 8B only.

- The 14B has never run the grouped form. This run is its control, plus the speed curve that says whether the form is worth serving at 14B.
- The attention shapes change for the first time. Qwen3-14B has 40 layers and 40 query heads over 8 key heads of 128, a group of 5. The 4B and the 8B have 36 layers and a group of 4. Source: the Hub configs as the paper-2 audit read them (`docs/mesures/paper2-audit-2026-09-28-brut/sources-findings.json:708`).
- So the grouped matmul has a new shape: 5 query rows a key head, `[1, 8, 5·l, 128]` against `[1, 40, l, 128]` expanded. Whether cuBLAS rounds it as the expanded form does is a measurement, as it was at group 4.
- No engine code is needed. `group_view` takes any group that divides the head count (`llvq-llm/src/model.rs:509-517`), and `grouped_attention` reshapes by that group (`model.rs:571-593`). The host test `attention_forms_agree` covers groups of 2 and 4 only (`model.rs:2787-2795`). No host test has run a group of 5.

## 2. What the commit holds

- `ops/jobs/gqa-14b.sh`. The body of `ops/jobs/gqa-8b.sh`, with the 14B's object, reference line, divergence position and timeout in place of the 8B's. Output `/out/gqa-14b-2026-11-04`.
- `DRY_RUN=1 bash ops/jobs/gqa-14b.sh` parses on the Mac (*measured*, 2026-11-04). Twelve mutants of the launcher's refusals were refused before any spend, on a scratch copy (*measured*). They are: `LLVQ_EMBED=q8` and `LLVQ_ROT_SHARE=0` in the flags; a wrong reference-line hash, and the 8B's; a wrong object hash; the 8B config, the 8B grouped config, the 4B grouped config; one table in place of two; the divergence at 77 in place of 78; the 8B reference directory; the 8B object directory.
- The body's gates were run against measured files (*measured*). Step 2's pass on the raw file of paper-table run 3, all but the int4 parity line, which that image did not print yet. They refuse the 8B's step-2 file, a divergence moved to 79, and an edited decoded line. Steps 3 and 4 pass on the 8B's raw files of today.
- The sealed 14B on the Mac: `~/q8b-14b-sealed-2026-09-23/qwen3-14b-sealed.bin`, downloaded from the bucket on 2026-11-04, 5,087,000,541 B, sha256 `61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0` (*measured*). That is the digest paper-table run 3 read on the mount (`docs/mesures/paper-table-2026-09-25-brut/paper-served-14b-2026-09-25-r2/files.sha256`), and the one paper 2 publishes. The launcher refuses to launch unless the local copy, that digest and the bucket's byte count agree.

**The served fields, spelled out.** `LLVQ_GQA_AB` and `LLVQ_GQA_PPL` are measurement modes, refused by name beside `LLVQ_CONFIG`. The job sets the 14B's served fields through the bench door's own variables, as the 8B job did. The launcher derives them from `configs/qwen3-14b-tetra-e4.json` and refuses any mismatch. Field by field:

| config field | value | bench variable | equivalent? |
|---|---|---|---|
| `layout` | `tetra48` | `LLVQ_FUSED_LAYOUT=tetra48` | yes, the same `FusedLayout::parse` |
| `embed` | `q4` | `LLVQ_EMBED=q4` | yes, the same `EmbedMode::parse`; the file stores both tables int4 g64 |
| `rot_share` | `1` | `LLVQ_ROT_SHARE=1` | yes, the same `RotShare::parse` |
| `fuse` | `0` | `LLVQ_FUSE=0` | yes, the same `FuseMode::parse` |
| `kv` | `f16` | `LLVQ_KV=f16` | yes. `load_with` hard-codes `KvMode::F16` on the bench door, so the variable cannot move it |
| `attention` | absent | none | yes. Absent means `expanded` on the `Cat` store, the arm `cat+expanded` of both modes |
| tile | not a field | `LLVQ_TILE_BLOCKS` unset | yes. Unset reads 64 on sm_89 on both doors, and every step gates on the line that says so |

The config file is byte for byte the one paper-table run 3 wrote on the mount (sha256 `833aec4c…`, its `files.sha256`). The launcher checks that too. Paper-table run 3 set the same four flags plus `LLVQ_EMBED=q4` (`ops/jobs/paper-served.sh:99,102`).

**The image.** Space `Pier-Jean/llvq-runner-cuda` at sha `ef658736a8eb3e9e6fa0a4cf22ecbc486d8bc78f`, the image of the 8B control. It was built from commit `49ec154` of branch `validation/vague1`. Its `llvq-llm` crate and every core crate are those of `06e1602`. Nothing in the engine crates has changed since `49ec154` (`git diff --stat 49ec154 HEAD` is empty over them). The launcher refuses to launch unless the Space is at the sha `IMAGE_SHA` names.

## 3. What runs

One job, l40sx1, the sealed 14B, `sealed-14b-2026-09-23/qwen3-14b-sealed.bin`, config `qwen3-14b-tetra-e4.json`. Every step but the first runs `fusedrun` with the five variables of §2.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed.** `fusedrun <file> 256` against the dense arm in the same process, as paper-table run 3. That run diverged from the dense arm at token 78, a tie-break published with its position (`paper-table-2026-09-25.txt`, "THE DECISION RULES"). The replay must diverge at the same token, and its decoded line must hash to that run's. This is the first 14B check that the trims of 2026-10-30 left the default path unmoved.
3. **The speed A/B.** `LLVQ_GQA_AB=5,2048,4096,8192`, 256 new tokens, cudarc's events off. Three arms in one model: `cat+expanded` (published), `grow(256)+shared`, `grow(256)+grouped`. For each length, one prefill an arm outside every clock, and the logits of the prompt's last position and of the first decode step compared bit for bit with the published arm's. Then 1 discarded and 5 timed decodes of 256 tokens an arm, every arm in every round. Speed is 255 / t, ratios formed round by round.
4. **The control.** `LLVQ_GQA_PPL=4096,12`. Twelve WikiText-2 test windows of 4,096, the published arm against `grow(256)+grouped`, by the prefill path and by a teacher-forced decode path in lockstep, paired window by window, 95 % t intervals.

**Memory at 8,192.** It fits on the 46,068 MiB card with room to spare (*computed*). The model takes 5.04 GB (*measured*, paper-table run 3). Three arms hold 8,448 positions each at 163,840 B a position, 4.15 GB. The score transient of an 8k prefill chunk at 40 heads is 8.05 GB (`docs/archive/debit-vague5-2026-11-02/prefill.md`, peak table). That is about 17 GB before the other activations. Step 2 holds both arms in one process, 34.6 GB, as paper-table run 3 did. No 14B run has reached 8,192 positions on this card.

## 4. Gates, each one voids the run

The script enforces every one of them.

1. The image builds, and `oracle` prints MATCH.
2. The object at 5,087,000,541 B and sha256 `61db37fe…` on the mount. The config written from the launch command, at its sha256.
3. Steps 2, 3 and 4 each print `tile 64 (served: measured optimum for sm_89)`, `int4: tv_q4_hh equals tv_q4_h bit for bit`, and the embedding line with `2 tables (model.embed_tokens.weight + lm_head.weight)`. No `LLVQ_*` variable is in the job's environment.
4. Step 2 prints exactly one line `  WARNING: fused F0: divergence at token 78 out of 256`, no `tokens identical to the dense arm`, and exactly one `  fused F0:  ` line. That line hashes to `186974aa69fef4b5611d3fd7598b3f517bb1e35ae78142da836ee75db3cfeabe`, the line of paper-table run 3 (*computed* from its raw file, as `323cabeb…` was for the 8B). The job stops here on a miss, about 8 min in.
5. Step 3 prints the event-tracking line, four `prompt N` headers, four `cat+expanded` speed lines, and for each of `shared` and `grouped` four logit lines and four speed lines.
6. Step 4 prints the event-tracking line, the token fingerprint `(tokens 3f1baca9033bf251)`, 12 window lines and both summary lines. The harness ratio, decode over prefill on the published arm, lies inside [0.99, 1.01]. `ppl` printed that fingerprint for all three sealed files, each with the tokenizer it carries (*measured*, `ppl-scelles-2026-10-04.txt`).

## 5. Outcomes reported, never gated

- Decode tok/s for every arm and length, median and range, and arm over published, round by round.
- For each unexpanded arm and length: its logits bit for bit or the largest difference, prompt then first decode step; its text identical, or the first token that differs.
- Perplexity of both arms by both paths, the paired intervals, the decode top-1 agreement, and the largest logit difference over every step.
- Step 2's dense line, against `c71ee7fc…`, the dense line of paper-table run 3. Only the divergence position is gated.
- The published arm's numbers are the first 14B measurement after the trims of 2026-10-30.
- Step 2's speed against the dense arm is printed and not reported. This job has no same-head arm, and rule 4 forbids the raw ratio alone.

## 6. Signed predictions

All *estimated*. The 8B's per-length increments in ms a decode token (*computed* from `gqa-8b-2026-11-04.txt`) are scaled by the bytes attention reads a position. The expanded form scales by ×1.35: 40 layers over 36, and 40 expanded heads written and read over 32. The grouped form scales by ×1.18 [1.11, 1.25]: the same 8 key heads over 40 layers, read once. The shared form scales by ×1.3.

The 5-token step is the 14B's served 17.48 ms (57.2 tok/s, *measured*, paper-table run 3) minus 0.38 ms of trims. The trims saved 0.47 ms at 4B and 0.33 ms at 8B (*measured*, step 2 of each control). The A/B's published arm sits 1 % above step 2, as at 4B and 8B. The ratios fall below the 8B's at short context because the same saving sits on a longer step.

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s | **58.5** | [56.5, 60.5] |
| step 2, first divergence from the dense arm | **token 78** | gated |
| published decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **59 / 44.4 / 35.2 / 21.4** | [56, 62] / [41, 48] / [32, 38.5] / [19.5, 23.5] |
| grouped decode tok/s at the same four | **62 / 56.7 / 51.4 / 44.1** | [59, 65] / [53, 60.5] / [47.5, 55.5] / [40, 48.5] |
| grouped / published at 5 / 2,048 / 4,096 / 8,192 | **1.05 / 1.28 / 1.46 / 2.06** | [1.02, 1.08] / [1.18, 1.38] / [1.32, 1.60] / [1.80, 2.35] |
| shared / published at the same four | **0.98 / 1.17 / 1.31 / 1.70** | [0.95, 1.01] / [1.06, 1.27] / [1.17, 1.46] / [1.45, 1.95] |
| shared prompt logits bit for bit | **at 5** | |
| grouped prompt logits at 5 | **bit for bit** | |
| grouped first decode step, max abs logit difference, worst length | **0.1** | [0.02, 0.4] |
| grouped text identical to published | **at all four lengths** | |
| published prefill ppl through the kernel | **8.444** | [8.43, 8.46] |
| published decode over prefill ppl | **1.000** | [0.998, 1.002] |
| grouped / published − 1, prefill | **+0.0000 %, bit for bit on all 12 windows** | [−0.02, +0.02] % |
| grouped / published − 1, decode | **0.00 %** | [−0.04, +0.04] % |
| both paired intervals inside ±0.12 % | **yes** | |
| decode top-1 agreement | **99.8 %** | [99.0, 99.95] |
| largest logit difference over every decode step | **8** | [1, 40] |

The dense reference is 8.4440 (*measured*, `ppl-scelles-2026-10-04.txt`). The kernel read 9.7066 at 8B against 9.7062 dense. The 8B's largest logit difference was 5.42 and the 4B's 13.13; it is reported here, never gated. The prefill prediction is the least sure: at group 5 the prefill matmul is `[1, 8, 20,480, 128]`, a shape no control has run.

## 7. Decision rules

R1, R2 and R3 cover every run whose gates passed. The band is the 4B's and the 8B's: each paired 95 % interval, grouped over published minus one, inside [−0.12 %, +0.12 %].

- **R1.** Both intervals lie inside the band, by prefill and by decode, and the grouped / published range lies entirely above 1.000 at all four lengths. The extension of the T8 grant to the 14B is proposed to the operator, who decides. By the 8B's precedent, a light MMLU under both configs follows before the grant stands. On a yes, a follow-up commit writes `configs/qwen3-14b-tetra-e4-grouped.json`, and that check runs under its own prereg. GSM8K through the kernel under that config is owed before the config is quoted beside a task score.
- **R2.** Both intervals lie inside the band, but at some length the grouped / published range does not lie entirely above 1.000. The form is not proposed at 14B. The run is a curve point, and a kernel timeline comes before any further step on this path at 14B.
- **R3.** At least one interval is not entirely inside the band, by either path. The 14B stays expanded. The operator decides whether fresh windows are worth a second control. An interval that contains zero says the control lacked power; one that does not says the form moves quality at 14B.
- Otherwise, that is when a gate of §4 failed, the run is void, and a retry goes under this prereg with its deviation written beside it. A step-2 miss is reported as such: it says the default path moved at 14B, which outranks the grouped question.
- A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from the 8B control's steps scaled to the 14B. At 8B, step 2 took 203 s, the A/B 522 s and the control 1,852 s, plus about 45 s of oracle and checks, 2,622 s in all (*measured*, the `date -u` lines of `hf jobs logs 6aeb209ec78863f9c6327d2b`).

- Oracle, object hash and config: about 1 min. The hash reads 5.09 GB on the mount instead of 2.82.
- Step 2: about 6.7 min. Dense load 264 s and 16.9 tok/s, fused load 19 s and 57.2 tok/s, six decodes of 256 tokens an arm (*measured*, paper-table run 3). The same sum gives 202 s at 8B against 203 measured.
- Step 3: about 14 min, 13 to 15.5 over the ranges. The decodes take about 424 s at the speeds of §6. The 8B's prefills and overhead took 244 s. A fit on the 4B and the 8B, linear work times parameters plus attention, puts the 14B's at 370 to 440 s.
- Step 4: about 50 min, 46 to 54. A window took 153.6 s at 8B. At 14B its two lockstep decodes take about 166 s at the speeds of §6, and its two prefills and logit reads 78 to 85 s by the same fit. Twelve windows.

In all about 71 min, **$2.13**, and 66 to 77 min over the ranges, $1.97 to $2.31 (*computed* on *estimated* inputs). The plain scaling of 2,622 s by the served step ratio, 95.0 over 57.2, gives 73 min, $2.18. Timeout 130 min, **$3.90** (*computed*).

The wave: $4.85 billed so far (*measured*, the six wave rows of `jobs.csv`). With this run's $3.90 at worst, the wave reaches $8.75, under the $10 cap (*computed*). The light MMLU that R1 calls for cost $1.25 at 8B. At 14B it would cost about $2.1 (*estimated*), which does not fit the $1.25 left after this run's worst case. It needs the operator's cap.

## 9. What this run does not establish

- Task accuracy under the grouped form at 14B.
- Quality past 4,096 positions: the windows stop at the fused prefill's rows.
- The served door at 14B, which waits for R1 and a config file.
- Metal, which refuses the form by name, and any card but the L40S.
- Any model above 14B.

## 10. Deviations

Any departure from this file goes in `proofs/preregistration-gqa-14b-2026-11-04-ECARTS.md`, beside it, never inside it.
