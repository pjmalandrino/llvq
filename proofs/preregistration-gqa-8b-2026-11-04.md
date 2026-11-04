# Preregistration. The grouped attention on the sealed 8B: paired perplexity against the expanded form, and the speed of both forms against the context

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04 for the first wave of the throughput validation roadmap, under a $10 HF cap for that wave. This run is conditional: it launches only after the 4B GSM8K under the grouped config (`proofs/preregistration-gsm8k-grouped-2026-11-04.md`) landed in its rule G1 or G3, no loss. Cost: about **$1.41** on l40sx1, *estimated* in §8; timeout 85 min, so **$2.55** at worst (*computed*). Spent before the wave: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus $3.19 for five billed jobs it missed, now written in it (*computed* on their measured durations at $1.80/h), so $252.03 (*computed*).

## 1. Why this run exists

The grouped attention decodes the 4B at 77.5 tok/s at 8,192 positions against 33.1 for the expanded form, and at 134.4 against 121.9 after a 5-token prompt (*measured*, `docs/mesures/gqa-ab-2026-11-01.txt`). Its paired WikiText-2 perplexity read +0.0000 % by prefill and +0.0095 % [−0.0048 ; +0.0238] by decode, inside the ±0.12 % band of `docs/ETAT.md` §4 (*measured*, `gqa-ppl-2026-11-01.txt`). On that control the operator granted the tolerance (decision T8, 2026-11-01), for the 4B only.

- The 8B has never run the grouped form. Each size needs its own control before the grant can extend to it. This run is that control, plus the speed curve that says whether the form is worth serving at 8B.
- The attention shapes are the 4B's: 36 layers, 32 query heads, 8 key heads of 128, a group of 4 (Qwen3-8B and Qwen3-4B configs). The matmuls whose rounding the 4B measured run at the same shapes here. What changes is the hidden size, 4,096 against 2,560, so the weights, the untied head and every hidden state.
- No engine code is needed. `LLVQ_GQA_AB` and `LLVQ_GQA_PPL` read the file, its tokenizer and WikiText-2, and nothing in them is 4B-specific (`bin/fusedrun.rs`, `run_gqa_ab`, `run_gqa_ppl`).

## 2. What the commit holds

- `ops/jobs/gqa-8b.sh`. The bodies of `ops/jobs/gqa-ab.sh` and `ops/jobs/gqa-ppl.sh` in one job, the 8B's object, reference line and tables in place of the 4B's. Output `/out/gqa-8b-2026-11-04`.
- `DRY_RUN=1 bash ops/jobs/gqa-8b.sh` parses on the Mac (*measured*, 2026-11-04). Eight mutants of the launcher's refusals were refused before any spend: `LLVQ_EMBED=q8` and `LLVQ_ROT_SHARE=0` in the flags, a wrong reference-line hash, a wrong object hash, the 4B config, the 4B grouped config, one table in place of two, the 14B reference directory (*measured*, on a scratch copy). The body's gates of steps 3 and 4 pass on the 4B's raw files of 2026-11-01 (*measured*). The launch path ran on a scratch copy with the `run.py` call stubbed out: local hash, local bytes, bucket bytes, absent output directory and Space sha all passed, and a wrong `IMAGE_SHA` was refused (*measured*, 2026-11-04).
- The sealed 8B on the Mac: `~/q8b-14b-sealed-2026-09-23/qwen3-8b-sealed-B.bin`, downloaded from the bucket on 2026-11-04, 2,815,098,745 B, sha256 `7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13` (*measured*). That is the digest paper-table run 2 read on the mount (`docs/mesures/paper-table-2026-09-25-brut/paper-served-8b-2026-09-25-r2/files.sha256`). The launcher refuses to launch unless the local copy, that digest and the bucket's byte count agree.

**The served fields, spelled out.** `LLVQ_GQA_AB` and `LLVQ_GQA_PPL` are measurement modes, refused by name beside `LLVQ_CONFIG`. The job sets the 8B's served fields through the bench door's own variables. The launcher derives them from `configs/qwen3-8b-tetra-e4.json` and refuses any mismatch. Field by field:

| config field | value | bench variable | equivalent? |
|---|---|---|---|
| `layout` | `tetra48` | `LLVQ_FUSED_LAYOUT=tetra48` | yes, the same `FusedLayout::parse` |
| `embed` | `q4` | `LLVQ_EMBED=q4` | yes, the same `EmbedMode::parse`; the file stores both tables int4 g64 |
| `rot_share` | `1` | `LLVQ_ROT_SHARE=1` | yes, the same `RotShare::parse` |
| `fuse` | `0` | `LLVQ_FUSE=0` | yes, the same `FuseMode::parse` |
| `kv` | `f16` | `LLVQ_KV=f16` | yes. `load_with` hard-codes `KvMode::F16` on the bench door (`fused_cuda.rs`), so the variable cannot move it |
| `attention` | absent | none | yes. Absent means `expanded` on the `Cat` store (`AttnHeads::store`), which is the arm `cat+expanded` of both modes. Both modes refuse `LLVQ_KV_PREALLOC` |
| tile | not a field | `LLVQ_TILE_BLOCKS` unset | yes. Unset reads 64 on sm_89 on both doors, and every step gates on the line that says so |

The config file is byte for byte the one paper-table run 2 wrote on the mount (sha256 `6f0d489f…`, its `files.sha256`). The launcher checks that too.

**The image.** Space `Pier-Jean/llvq-runner-cuda` at sha `ef658736a8eb3e9e6fa0a4cf22ecbc486d8bc78f`, built from commit `49ec154` of branch `validation/vague1`. Its `llvq-llm` crate and every core crate are those of `06e1602` (`git diff --stat 06e1602..49ec154` is empty over them). The only engine change since is in `llvq-cuda` (`planesbench`, `seg_host.rs`, a test), which `fusedrun` does not use. The launcher refuses to launch unless the Space is at the sha `IMAGE_SHA` names.

## 3. What runs

One job, l40sx1, the sealed 8B, `sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin`, config `qwen3-8b-tetra-e4.json`. Every step but the first runs `fusedrun` with the five variables of §2.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed.** `fusedrun <file> 256` against the dense arm in the same process, as paper-table run 2. The decoded line must hash to that run's.
3. **The speed A/B.** `LLVQ_GQA_AB=5,2048,4096,8192`, 256 new tokens, cudarc's events off. Three arms in one model: `cat+expanded` (published), `grow(256)+shared`, `grow(256)+grouped`. For each length, one prefill an arm outside every clock, and the logits of the prompt's last position and of the first decode step compared bit for bit with the published arm's. Then 1 discarded and 5 timed decodes of 256 tokens an arm, every arm in every round. Speed is 255 / t, ratios formed round by round.
4. **The control.** `LLVQ_GQA_PPL=4096,12`. Twelve WikiText-2 test windows of 4,096, the published arm against `grow(256)+grouped`, by the prefill path and by a teacher-forced decode path in lockstep, paired window by window, 95 % t intervals.

## 4. Gates, each one voids the run

The script enforces every one of them.

1. The image builds, and `oracle` prints MATCH.
2. The object at 2,815,098,745 B and sha256 `7bdb9a55…` on the mount. The config written from the launch command, at its sha256.
3. Steps 2, 3 and 4 each print `tile 64 (served: measured optimum for sm_89)`, `int4: tv_q4_hh equals tv_q4_h bit for bit`, and the embedding line with `2 tables (model.embed_tokens.weight + lm_head.weight)`. No `LLVQ_*` variable is in the job's environment.
4. Step 2 prints `fused F0: 256 tokens identical to the dense arm` and exactly one `  fused F0:  ` line. That line hashes to `323cabeb8520cba9cede5ea6c1a611a1ceb5322c0aae473fac63ed911d61e498`, the line of paper-table run 2 (*computed* from its raw file). The job stops here on a miss, about 5 min in.
5. Step 3 prints the event-tracking line, four `prompt N` headers, four `cat+expanded` speed lines, and for each of `shared` and `grouped` four logit lines and four speed lines.
6. Step 4 prints the event-tracking line, the token fingerprint `(tokens 3f1baca9033bf251)`, 12 window lines and both summary lines. The harness ratio, decode over prefill on the published arm, lies inside [0.99, 1.01]. `ppl` printed that fingerprint for all three sealed files, each with the tokenizer it carries (*measured*, `ppl-scelles-2026-10-04.txt`).

## 5. Outcomes reported, never gated

- Decode tok/s for every arm and length, median and range, and arm over published, round by round.
- For each unexpanded arm and length: its logits bit for bit or the largest difference, prompt then first decode step; its text identical, or the first token that differs.
- Perplexity of both arms by both paths, the paired intervals, the decode top-1 agreement, and the largest logit difference over every step.
- The published arm's numbers are the first 8B measurement after the trims of 2026-10-30.
- Step 2's speed against the dense arm is printed and not reported. This job has no same-head arm, and rule 4 forbids the raw ratio alone.

## 6. Signed predictions

All *estimated*. The 4B's per-length increments in ms a decode token for each arm (*computed* in `gqa-ab-2026-11-01.txt`) carry over unchanged, because the attention shapes are the same. The 5-token step is the 8B's served 10.53 ms (95.0 tok/s, *measured*, paper-table run 2) minus the 0.47 ms the trims saved at 4B (113.8 to 120.3 tok/s in step 2, *measured*), the same 252 projections and 36 layers. The A/B's published arm sits 1.3 % above step 2, as at 4B. The arm ratios fall below the 4B's because the same saving in ms sits on a longer step.

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s | **99** | [93, 105] |
| published decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **101 / 72 / 55 / 31** | [95, 107] / [66, 78] / [50, 60] / [28.5, 34] |
| grouped decode tok/s at the same four | **109 / 95 / 83 / 68** | [102, 116] / [87, 103] / [75, 91] / [61, 75] |
| grouped / published at 5 / 2,048 / 4,096 / 8,192 | **1.08 / 1.32 / 1.51 / 2.18** | [1.03, 1.14] / [1.18, 1.45] / [1.33, 1.70] / [1.80, 2.55] |
| shared / published at the same four | **0.98 / 1.18 / 1.34 / 1.78** | [0.92, 1.02] / [1.03, 1.32] / [1.15, 1.55] / [1.45, 2.20] |
| shared logits bit for bit, prompt and first step | **at 5 only** | |
| grouped prompt logits at 5 | **bit for bit** | |
| grouped first decode step, max abs logit difference, worst length | **0.05** | [0.005, 0.3] |
| grouped text identical to published | **at all four lengths** | |
| published prefill ppl through the kernel | **9.705** | [9.67, 9.74] |
| published decode over prefill ppl | **1.000** | [0.998, 1.002] |
| grouped / published − 1, prefill | **+0.0000 %, bit for bit on all 12 windows** | [−0.02, +0.02] % |
| grouped / published − 1, decode | **+0.01 %** | [−0.04, +0.05] % |
| both paired intervals inside ±0.12 % | **yes** | |
| decode top-1 agreement | **99.75 %** | [99.0, 99.95] |
| largest logit difference over every decode step | **10** | [1, 40] |

The dense reference is 9.7062 (*measured*, `ppl-scelles-2026-10-04.txt`); the 4B read 12.5822 through the kernel against 12.5834 dense. The 4B's largest logit difference, 13.13, missed its own prediction by a factor of a hundred. It is reported here, never gated.

## 7. Decision rules

R1, R2 and R3 cover every run whose gates passed. The band is the 4B's: each paired 95 % interval, grouped over published minus one, inside [−0.12 %, +0.12 %].

- **R1.** Both intervals lie inside the band, by prefill and by decode, and the grouped / published range lies entirely above 1.000 at all four lengths. The extension of the T8 grant to the 8B is proposed to the operator, who decides. On a yes, a follow-up commit writes `configs/qwen3-8b-tetra-e4-grouped.json`, and a served-door sweep under its own prereg measures the served object. GSM8K through the kernel under that config (about $2.2, *estimated* on the expanded census's $2.37, *measured*, `jobs.csv`) is owed before the config is quoted beside a task score.
- **R2.** Both intervals lie inside the band, but at some length the grouped / published range does not lie entirely above 1.000. The form is not proposed at 8B. The run is a curve point, and a kernel timeline comes before any further step on this path at 8B.
- **R3.** At least one interval is not entirely inside the band, by either path. The 8B stays expanded. The operator decides whether fresh windows are worth a second control. An interval that contains zero says the control lacked power; one that does not says the form moves quality at 8B.
- **The 14B** is a later step under its own prereg, whatever the outcome here.
- Otherwise, that is when a gate of §4 failed, the run is void, and a retry goes under this prereg with its deviation written beside it.
- If the 4B GSM8K grouped run lands in G2, or is void, this run does not launch. The launcher refuses without `GSM8K_4B_VERDICT=G1` or `G3`, and the operator decides.
- A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from the 4B's two jobs scaled to the 8B. At 4B, step 2 took 128 s, the A/B 420 s and the control 1,488 s, with about 27 s of oracle and checks (*measured*, the `date -u` lines of `gqa-ab-2026-11-01-brut/job.log` and `gqa-ppl-2026-11-01-brut/job.log`).

- Oracle, object hash and config: about 1 min.
- Step 2: about 3.5 min. Dense load 121 s and 26.4 tok/s, fused load 8 s and 95.0 tok/s, six decodes of 256 tokens an arm (*measured*, paper-table run 2).
- Step 3: about 10 min. The decode adds 1.74 ms a token over the 4B's at every length (*computed*: 10.53 against 8.79 ms, the served 8B and 4B), over about 18,400 timed tokens. The prefills, 168 s at 4B, cost 1.5 to 2 times more at 8B.
- Step 4: about 32 min. The same 1.74 ms over 98,280 decode steps, and the 4B's 24 prefills of 4,096, about 360 s, at 1.5 to 2 times.

In all about 47 min, **$1.41**, and 43 to 52 min over the prefill range (*computed* on *estimated* inputs). Timeout 85 min, **$2.55** (*computed*).

The wave: $0.13 spent so far (*measured*, `tk-sweep-4b` and `fi-decode-anchor`, `jobs.csv`). The other preregs of the wave carry their own worst cases: $4.50 for the 4B GSM8K grouped run, which goes before this one, and $1.35 for `seg-tetra`. With this run's $2.55, the worst case of the wave is $8.53, under the $10 cap (*computed*).

## 9. What this run does not establish

- Task accuracy under the grouped form at 8B.
- Quality past 4,096 positions: the windows stop at the fused prefill's rows.
- The served door at 8B, which waits for R1 and a config file.
- The 14B, Metal, which refuses the form by name, and any card but the L40S.
- The size of rounding spikes inside the published path itself, the reference the 4B's 13.13 still lacks.

## 10. Deviations

Any departure from this file goes in `proofs/preregistration-gqa-8b-2026-11-04-ECARTS.md`, beside it, never inside it.
