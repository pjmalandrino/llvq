# Preregistration. The 8B under the grouped config: the served door, and a light MMLU under both configs (2026-11-04)

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04. The
operator granted the T8 of the grouped form at 8B the same day, after the control `gqa-8b-2026-11-04` read rule R1, on
the condition of a light MMLU check ("ok mais on refera un MMLU light pour check alors"). This run is that check. Cap:
the $10 HF cap of the first wave, of which $3.60 is billed (*computed* from the registry). Cost: about **$1.35** on
l40sx1 (*estimated*, §7); timeout 90 min, so **$2.70** at worst (*computed*). Registry before it: $255.63 over 247 rows
(*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The grouped form kept WikiText-2 perplexity inside ±0.12 % at 8B, +0.0000 % by prefill and −0.0048 %
[−0.0117 ; +0.0021] by decode, and decoded x2.156 at 8,192 positions (*measured*, `gqa-8b-2026-11-04.txt`).
Perplexity is not a task score. At 4B, the grouped form wrote byte-identical MMLU rows on the 57-question door
(*measured*, `gsm8k-grouped-2026-11-04.txt`). At 8B the grouped prefill is not bit for bit at every length: at 2,048
positions its prompt logits differed by up to 2.891e-1 from the expanded form's (*measured*, `gqa-8b-2026-11-04-brut/
gqa-ab-256.txt`). A five-shot MMLU prompt is a prefill of a few hundred to about 3,000 tokens, so MMLU picks can move.

## 2. What the commit holds

`configs/qwen3-8b-tetra-e4-grouped.json`: the 8B's published config (`qwen3-8b-tetra-e4.json`, sha256 `6f0d489f…`)
with `"attention": "grouped"` and a note; no other field differs, and the launcher checks it. `ops/jobs/served-grouped-8b.sh`,
in the shape of `ops/jobs/served-grouped.sh` and `ops/jobs/gsm8k-grouped.sh`. No engine code.

## 3. What runs

One job, l40sx1, Space sha `ef658736…` (llvq-llm sources of `06e1602`), the sealed 8B
`sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin` (2,815,098,745 B, sha256 `7bdb9a55…`).

1. Oracle on Qwen3-0.6B.
2. The served door, `LLVQ_CONFIG=<config> fusedrun <file> 256`, under the expanded then the grouped config.
3. `LLVQ_CONTEXTS=5,2048,4096,8192` beside the grouped config, 256 new tokens.
4. MMLU five-shot, `LLVQ_MMLU_ALLOC=flat`, `mmlu <file> cuda 5`: 5 questions a subject, 285 in all, under each config,
   one dump each.
5. After the job, on the Mac, $0: `mmlupair <grouped> <expanded>`, and a byte comparison of the data rows.

## 4. Gates, each one voids the run

1. Oracle MATCH; object and both configs at their sha256 on the mount.
2. Every log's `served config:` line names its form (`attention expanded on cat`, `attention grouped on grow(256)`),
   and prints `tile 64 (served: measured optimum for sm_89)` and the int4 parity line. The served-door logs print the
   q4 g64 embedding on 2 tables.
3. The sweep prints four context lines.
4. Both MMLU dumps carry their `# attention=` line and end on the same `# end fingerprint=<f> questions=285`.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| served door, grouped, tok/s after 5 tokens | **104** | [98, 110] |
| sweep, grouped, tok/s at 8,192 | **66** | [60, 70] |
| served texts equal under both configs | **yes** | |
| MMLU picks identical under both configs | **283 of 285** | [278, 285] |
| MMLU paired Δ grouped − expanded | **+0.0 pp** | [−0.7, +0.7] |
| MMLU rows identical bytes | **no** | |

Reasoning, *estimated*. The bench door read 106.4 tok/s at 5 and 66.9 at 8,192 (*measured*, `gqa-8b-2026-11-04`); the
served door adds the q4 tables of two tables. Prompt logits moved by up to 0.29 at 2,048, so a few close questions
can flip either way; 4B moved none of 57.

## 6. Decision rules

Δ = grouped − expanded, from `mmlupair`, with its McNemar exact p. The rules read the MMLU only; the speeds are
reported.

- **M1.** The 285 picks are identical. The grouped form moves no pick on this sample. The T8 at 8B stands, the grouped
  config becomes the 8B's served config, and its served door and sweep are published beside the 4B's.
- **M2.** Some picks differ, |Δ| ≤ 1.5 pp and p ≥ 0.05 (the criterion of the GSM8K engine gate, `gsm8k-wave1-2026-09-26.txt`).
  The T8 at 8B stands as for M1, and the discordant questions are reported.
- **M3.** |Δ| > 1.5 pp or p < 0.05. The grouped form moves MMLU at 8B. The T8 at 8B is suspended, the 8B stays served
  expanded, and the operator decides on a census.
- Otherwise, including any failed gate, the run is void, and a retry goes under this prereg with its deviation beside it.

M1, M2 and M3 cover every Δ: identical picks give Δ = 0 and p = 1.

## 7. Cost

*Estimated* on l40sx1 at $1.80/h. The 8B MMLU door took 184 s for 57 questions with its load (*measured*,
`served-8b-2026-09-21-brut/mmlu-door.txt`), so about 13 min an arm for 285, 26 min for two. The served door twice,
about 4 min; the sweep, about 8 min (the 4B's two sweeps and three doors took 18 min, `served-grouped-2026-11-02`);
oracle and loads, about 3 min. In all about 45 min, **$1.35**. Timeout 90 min, **$2.70**.

## 8. What this run does not establish

- MMLU at 8B under the grouped form beyond 285 questions; the 8B's GSM8K under it (about $2.2, owed before the config
  is quoted beside a GSM8K score).
- The 14B; Metal, which refuses the form; any card but the L40S.

## 9. Deviations

Any departure goes in `proofs/preregistration-served-grouped-8b-2026-11-04-ECARTS.md`, beside this file.
