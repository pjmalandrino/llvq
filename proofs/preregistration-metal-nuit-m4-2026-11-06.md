# Preregistration. M4: the grouped attention form on Metal at 2,048 and 4,096 tokens of context

Written on 2026-11-06 at 00:35 on branch `claude/metal-nuit-3c4257`, after M1, M2 and M3 of
`preregistration-metal-nuit-2026-11-06.md` (journal `docs/mesures/metal-nuit-2026-11-06.txt`). To be timestamped before
its first measurement, then never edited. Cost **$0**, local, under the operator's go for local measurements this night.

## 1. Why

M3 read the grouped form's logits equal to the expanded form's, bit for bit, at all 16,380 decode steps of four
WikiText-2 windows of 4,096 on this Mac. M1 read no speed difference at a 5-token prompt (x0.999 [0.887-1.095]), as
predicted: the form removes the per-step copy of the cache (`Cat` concatenation plus `repeat_kv` to 32 heads), which is
small when the cache is 5 positions. Its value lies at long contexts. On the L40S it took 8,192 from 33.2 to 77.5 tok/s.
No Metal number exists past 5 tokens. This run reads one, with the token identity, on the after arm of M2.

## 2. What runs

Binary `metalrun` sha256 `a05c8ba7c16a3603fc564fce8438508b63339d9a92a0c8b4ccffcb978c0600fc`, built from the commit that
carries this file: the M1-M3 binary plus a `prompt_tokens` argument to `time` (the first N tokens of the WikiText-2
test split, prefilled in chunks of 256 rows). Run by path under nice 10, at least 15 minutes after the last build.

1. M4a: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd,lutd+grp 64 5 2048`.
2. M4b: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd,lutd+grp 64 5 4096`.

Each: the bit gate (253 of 253 records), then 2 discarded + 5 timed rounds, `lutd` then `lutd+grp` each round. The
decode rate is 63 tokens over the decode clock, after the prefill. The prefill clock is printed as "first token".

## 3. Gates

1. The bit gate holds (253 of 253).
2. Each arm returns the same tokens every round.

A failed gate voids that context's reading.

## 4. Reading, not a decision

This run decides nothing. The grouped form is refused by name on the Metal door (`served.rs` `check_device`); lifting
that refusal is the operator's decision. The journal reports, per context:
- the decode-rate ratio `lutd+grp / lutd`, median and range formed round by round;
- whether the two arms' 64 tokens are equal;
- the prefill time of each arm.

## 5. Predictions, signed

*Estimated*. The expanded form moves the cache about three times a step (concatenation, `repeat_kv` to four times the
heads, the attention reads), 302 MB of f16 cache at 2,048 positions and 604 MB at 4,096 on the 4B; the grouped form
reads it once in place.

| quantity | prediction |
|---|---|
| decode `lutd+grp / lutd` at 2,048 | x1.20 to x1.60 |
| decode `lutd+grp / lutd` at 4,096 | x1.40 to x2.20 |
| `lutd` decode at 2,048 | 30 to 42 tok/s |
| tokens, both contexts | equal (M3 read bit identity on the decode path) |
| prefill ratio | x0.95 to x1.10 (the prefill attends within 256-row chunks on both forms) |

## 6. Journal

Appended to `docs/mesures/metal-nuit-2026-11-06.txt` under "M4".
