# Preregistration. The 14B under the grouped config: the served door, and a light MMLU under both configs (2026-11-04)

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04. The
control `gqa-14b-2026-11-04` read rule R1 the same day. By the habit the operator set at 8B ("ok mais on refera un MMLU
light pour check alors"), the T8 at 14B waits on a light MMLU check; the operator approved this run ("Oui, lance-le"). Cap:
the $10 HF cap of the first wave: $7.05 billed in this session's jobs, and the attention-kernel bench of another
session ($0.63 estimated, $0.99 worst) is running. Cost: about **$1.68** on l40sx1 (*estimated*, §7); timeout 65 min,
so **$1.95** at worst (*computed*), which keeps the wave at or under $9.99 at worst. Registry before it: $259.08 over
249 rows (*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The grouped form kept WikiText-2 perplexity inside ±0.12 % at 14B, +0.0000 % by prefill and +0.0023 %
[−0.0062 ; +0.0109] by decode, and decoded x2.140 at 8,192 positions (*measured*, `gqa-14b-2026-11-04.txt`).
Perplexity is not a task score. At 8B the same check moved no MMLU pick of 285 while 4 rows differed in their logits
(*measured*, `served-grouped-8b-2026-11-04.txt`). The 14B groups 5 query heads a key head, a shape no MMLU has run.

## 2. What the commit holds

`configs/qwen3-14b-tetra-e4-grouped.json`: the 14B's published config (`qwen3-14b-tetra-e4.json`, sha256 `833aec4c…`)
with `"attention": "grouped"` and a note; no other field differs, and the launcher checks it.
`ops/jobs/served-grouped-14b.sh`, the 8B job of the same day without its context sweep. No engine code.

## 3. What runs

One job, l40sx1, Space sha `ef658736…` (llvq-llm sources of `06e1602`), the sealed 14B
`sealed-14b-2026-09-23/qwen3-14b-sealed.bin` (5,087,000,541 B, sha256 `61db37fe…`).

1. Oracle on Qwen3-0.6B.
2. The served door, `LLVQ_CONFIG=<config> fusedrun <file> 256`, under the expanded then the grouped config.
3. MMLU five-shot, `LLVQ_MMLU_ALLOC=flat`, `mmlu <file> cuda 5`: 5 questions a subject, 285 in all, under each config,
   one dump each.
4. After the job, on the Mac, $0: `mmlupair <grouped> <expanded>`, and a byte comparison of the data rows.

## 4. Gates, each one voids the run

1. Oracle MATCH; object and both configs at their sha256 on the mount.
2. Every log's `served config:` line names its form (`attention expanded on cat`, `attention grouped on grow(256)`),
   and prints `tile 64 (served: measured optimum for sm_89)` and the int4 parity line. The served-door logs print the
   q4 g64 embedding on 2 tables.
3. Both MMLU dumps carry their `# attention=` line and end on the same `# end fingerprint=<f> questions=285`.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| served door, grouped, tok/s after 5 tokens | **62** | [57, 66] |
| served door, expanded, tok/s after 5 tokens | **58** | [54, 61] |
| served texts equal under both configs | **yes** | |
| MMLU picks identical under both configs | **283 of 285** | [278, 285] |
| MMLU paired Δ grouped − expanded | **+0.0 pp** | [−0.7, +0.7] |
| MMLU rows identical bytes | **no** | |

Reasoning, *estimated*. The bench door read 63.1 and 59.2 tok/s at 5 tokens (*measured*, `gqa-14b-2026-11-04`), and
the published served door read 57.2 (paper table). At 8B no pick moved and 4 rows of 285 differed in their logits.

## 6. Decision rules

Δ = grouped − expanded, from `mmlupair`, with its McNemar exact p. The rules read the MMLU only; the speeds are
reported.

- **M1.** The 285 picks are identical. The grouped form moves no pick on this sample. The T8 at 14B is proposed as
  confirmed, and the grouped config as the 14B's served config; the operator decides.
- **M2.** Some picks differ, |Δ| ≤ 1.5 pp and p ≥ 0.05 (the criterion of the GSM8K engine gate). As M1, with the
  discordant questions reported.
- **M3.** |Δ| > 1.5 pp or p < 0.05. The grouped form moves MMLU at 14B. The T8 at 14B is not proposed, and the 14B
  stays served expanded.
- Otherwise, including any failed gate or the timeout, the run is void, and a retry goes under this prereg with its
  deviation beside it.

M1, M2 and M3 cover every Δ: identical picks give Δ = 0 and p = 1.

## 7. Cost

*Estimated* on l40sx1 at $1.80/h. The 8B's MMLU arms took 15.4 min each on 2026-11-04 (*measured*, job log); a 14B
WikiText window took 1.60 times an 8B one the same day (*computed* from the two `gqa-ppl` logs), so about 25 min an
arm, 50 min for two. The two doors, about 2 min; oracle and loads, about 4 min. In all about 56 min, **$1.68**.
Timeout 65 min, **$1.95**.

## 8. What this run does not establish

- MMLU at 14B under the grouped form beyond 285 questions; the 14B's GSM8K under it.
- The 14B's served speed against context under the config: the bench sweep of `gqa-14b-2026-11-04` stands for it.
- Metal, which refuses the form; any card but the L40S.

## 9. Deviations

Any departure goes in `proofs/preregistration-served-grouped-14b-2026-11-04-ECARTS.md`, beside this file.
