# Preregistration. The 8B and the 14B under the grouped config: GSM8K through the kernel, paired with each size's expanded census

**Written on 2026-11-04. NOT STAMPED, NOT LAUNCHED.** It is stamped with OpenTimestamps before the first measurement,
and renamed to the day of the stamp if that day is later. Operator go: the one go of lot 1 (`ops/jobs/lot1.sh`), to be
given in chat. Cost: about **$2.25** at 8B and **$4.11** at 14B on l40sx1 (*estimated*, §8); timeouts 100 and 170 min,
so **$3.00** and **$5.10** at worst (*computed*). Registry before the lot: $261.35 over 251 rows (*measured*,
`docs/data/jobs.csv`, 2026-11-04).

**Base commit:** `02ec809323eb` (branch `integration/lot1`); this file is committed on top of it. **Space sha:**
`142216fc72326a98b94316a7f6cd2e989f0efc50`, built from `02ec809`. The job script is `ops/jobs/gsm8k-grouped-size.sh`,
`SIZE=8b` or `SIZE=14b`, one job a size; it refuses to launch without this file and its `.ots`.

## 1. Why this run exists

- The 8B and the 14B are served grouped since 2026-11-04 (`configs/README.md`, commit `81ff209`). Their licence rests
  on paired perplexity inside ±0.12 % (8B −0.0048 % [−0.0117 ; +0.0021], 14B +0.0023 % [−0.0062 ; +0.0109], decode
  path) and on a light MMLU of 285 questions with the same picks under both configs (*measured*,
  `gqa-8b-2026-11-04.txt`, `gqa-14b-2026-11-04.txt`, `served-grouped-8b-2026-11-04.txt`,
  `served-grouped-14b-2026-11-04.txt`).
- No task with long greedy generations has run under the grouped form at either size. Their published GSM8K, 88.63 %
  at 8B and 92.04 % at 14B, are the expanded form's (*measured*, `gsm8k-wave2-2026-09-26.txt`).
- At 4B the same question read Δ = −0.53 pp [−1.07 ; +0.00], p = 0.092, 3 against 10 discordant: rule G1
  (*measured*, `gsm8k-grouped-2026-11-04.txt`). This run asks it at the two other sizes, with the same rules.

## 2. What the commit holds

No engine code. `bin/gsm8k` reads `LLVQ_CONFIG` and applies the config's store and attention form through
`Served::apply`, as the 4B run used it. `oracle` and `gsm8k` are in the build list and in the runtime `COPY` of
`ops/Dockerfile.cuda`. The new script `ops/jobs/gsm8k-grouped-size.sh` is `ops/jobs/gsm8k-grouped.sh` with these
changes:

- `SIZE=8b`: `sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin`, sha256 `7bdb9a55…`, configs `qwen3-8b-tetra-e4.json`
  (`6f0d489f…`) and `qwen3-8b-tetra-e4-grouped.json` (`418d2d00…`), reference census
  `gsm8k-wave2-2026-09-26-brut/run2-served-8b/gsm8k-8b-served.jsonl` (`23bbac8b…`), timeout 100 min.
- `SIZE=14b`: `sealed-14b-2026-09-23/qwen3-14b-sealed.bin`, sha256 `61db37fe…`, configs `qwen3-14b-tetra-e4.json`
  (`833aec4c…`) and `qwen3-14b-tetra-e4-grouped.json` (`d2353c78…`), reference census
  `gsm8k-wave2-2026-09-26-brut/run3-served-14b/gsm8k-14b-served.jsonl` (`05acf953…`), timeout 170 min.
- The launch script refuses a local object whose sha256 is not the census's, an expanded config that differs by a byte
  from the copy the census wrote, and a reference census whose sha256 or trailer moved.
- No MMLU step: the light MMLU of both sizes ran on 2026-11-04.
- Gate 4 has no 50-problem run of 2026-09-26 to replay at 8B or 14B: the campaign ran its smoke at 4B only. The replay
  takes the pilot's 50 problems (`gsm8k <file> cuda 50`, `gsm8k::select` with seed `0x65736d386b`, the sample the 4B
  gate used, not the first 50 of the split), and its rows are held to the census rows at the same indices. That this
  holds is *computed* at 4B: run 0's 50 rows and run 1's census rows at the same indices digest alike,
  `1de9a40d…`. The digests at 8B and 14B, computed from the census dumps by the same rule, go into the job.

## 3. What runs

One job a size, l40sx1, the image named above.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **Expanded replay**: the pilot's 50 problems under the expanded config. If one row differs from the census, the job
   stops here (§4 gate 4).
3. **GSM8K, grouped config**, all 1,319 problems, zero-shot, no reasoning block, cap 1,024 new tokens, dataset revision
   `740312ad…`: every setting of the census of 2026-09-26 but the config and the image; step 2 ties the two images.
4. **After the job, on the Mac, $0**: `gsm8kpair <grouped dump> <census of 2026-09-26>`, so that A − B reads grouped −
   expanded; the median character of the first difference between paired completions.

## 4. Gates, each one voids that size's run

1. The image builds, and `oracle` prints MATCH.
2. The object and both configs at their bytes and sha256 on the mount.
3. The `served config:` line reads `attention expanded on cat` in the replay and `attention grouped on grow(256)` in the
   census; both logs print `tile 64 (served: measured optimum for sm_89)` and `int4: tv_q4_hh equals tv_q4_h bit for
   bit`.
4. The replay ends on `"end":true` and on the pilot's fingerprint `94c7b98876288901`, and its 50 rows digest to the
   census rows at the same indices (*computed*): 8B `3e5666620e204b3aff244c6398963771dcae18f67e8a9fd87c7737d49b766ecd`,
   14B `14631e2b231f47cb2131e7d310e56f65c4e8207d35e648d45a463b9c46ad1f72`. The rule: drop the header and the trailer,
   drop `prefill_s` and `decode_s` from each row, write each row with sorted keys, join by newlines, sha256 of the UTF-8
   bytes.
5. The census dump ends on `"end":true` and on the census fingerprint `bfa9135ce4b55c75`.
6. `gsm8kpair` re-grades 0 rows of either dump.
7. No `LLVQ_*` variable in the job environment; each command sets its own.

## 5. Outcomes reported, never gated

- Per size: accuracy, stops at the cap, mean tokens, decode ms a token of the grouped arm; the replay's decode ms a
  token, read beside it and never divided.
- Per size: Δ = grouped − expanded in points, its 95 % interval, discordant counts both ways, McNemar's exact p.
- Where two completions part: the median character of the first difference.

## 6. Signed predictions

| quantity | 8B | 14B |
|---|---|---|
| replay: rows identical to the census | **50 of 50** | **50 of 50** |
| GSM8K accuracy, grouped | **88.6 %** [87.1, 90.1] | **92.0 %** [90.7, 93.3] |
| Δ grouped − expanded | **+0.0 pp** [−1.0, +1.0] | **+0.0 pp** [−1.0, +1.0] |
| discordant problems, both directions | **13** [5, 40] | **12** [4, 40] |
| McNemar exact p | **≥ 0.05** | **≥ 0.05** |
| stops at the cap | **9** [3, 16] | **5** [1, 12] |
| decode ms a token at GSM8K lengths | **9.5** [9.1, 10.2] | **16.2** [15.5, 17.0] |

Reasoning, *estimated*. The grouped form moves rounding only. Its teacher-forced top-1 agreement is 99.817 % at 8B and
99.792 % at 14B, against 99.788 % at 4B (*measured*), where 13 problems parted. The decode step: through the served
door the grouped form reads 105.8 tok/s at 8B and 62.5 at 14B, 9.45 and 16.0 ms (*measured*,
`served-grouped-8b-2026-11-04.txt`, `served-grouped-14b-2026-11-04.txt`), and at 4B the GSM8K step equalled the door's
within 1 % (7.46 against 7.50 ms, *measured*). The expanded census steps, 10.910 and 17.936 ms (*measured*), bound them
from above.

## 7. Decision rules

The same rules as the 4B run (`preregistration-gsm8k-grouped-2026-11-04.md` §7), read per size, on Δ = grouped −
expanded, and only when that size's gate 4 passed.

- **G1.** |Δ| ≤ 1.5 pp and McNemar exact p ≥ 0.05. The grouped config's GSM8K is settled at that size. Its score is
  proposed for ETAT §2 beside the expanded one; the operator decides how the cards word it.
- **G2.** Δ < 0 and (|Δ| > 1.5 pp or p < 0.05). The grouped form loses on reasoning at that size. The operator decides
  whether that size stays grouped (the T8 grant of 2026-11-04 is revisited, the operator's ruling).
- **G3.** Δ > 0 and (|Δ| > 1.5 pp or p < 0.05). Reported as a difference in the form's favour; it suspends nothing, and
  the operator decides how it is worded.
- Otherwise, including any failed gate, that size's run is void, and a retry goes under this prereg with its deviation
  written beside it, after a new go. When gate 4 fails, the job stops before the census, and the operator decides
  whether an expanded census on today's image runs beside a grouped one. A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from the censuses of 2026-09-26 (*measured*, `docs/data/jobs.csv` and the run logs):
8B 4,739 s running, 282.3 tokens a problem, decode 10.910 ms a token, prefill 5.344 ms a prompt token; 14B 8,480 s,
306.5 tokens, 17.936 and 9.880 ms; 88.3 prompt tokens a problem.

- 8B: decode 1,319 × 282.3 × 9.45 ms = 59 min, prefill 10.4 min, replay 2.8 min, start, oracle and loads about 3 min:
  about **75 min, $2.25**. With the expanded step and 5 % more tokens, 87 min. Timeout 100 min, **$3.00**.
- 14B: decode 1,319 × 306.5 × 16.2 ms = 109 min, prefill 19.2 min, replay 5 min, start, oracle and loads about 4 min:
  about **137 min, $4.11**. With the expanded step and 5 % more tokens, 155 min. Timeout 170 min, **$5.10**.

A job killed at its timeout keeps its dump, flushed a problem at a time, and is void by gate 5.

## 9. What this run does not establish

- MMLU beyond the light check of 2026-11-04.
- The expanded form's GSM8K on today's image beyond the 50 replayed problems.
- Any context beyond GSM8K's lengths, any card but the L40S, Metal.
- Why a problem flips, if some do.

## 10. Deviations

Any departure from this file goes in `preregistration-gsm8k-grouped-size-2026-11-04-ECARTS.md`, beside it, never inside
it.
