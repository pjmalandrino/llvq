# Preregistration. The 4B's sealed composition at 8B and 14B: q4 tables pay for int4 tables

**Written on 2026-09-23, BEFORE the runs, timestamped before the first job.**
Operator go, 2026-09-23: "on reprend la composition avec embed en q4 et on rajoute l'int4 pour
combler ce qu'on a gagné sur le 8B et le 14B, ensuite on rejoue les MMLU", both cards in
parallel. Announced cost: 8B census ~$0.98 (timeout 60 min, ceiling $1.80), 14B census ~$1.95
(timeout 90 min, ceiling $2.70): **~$2.93**, priced on `dclm-8b-ft-mmlu` (1,966 s) and
`dclm-14b-ft-mmlu` (3,898 s). Project total before: $220.63.

## 1. The claim on trial

At 4B the package (q4 embedding, `o_proj` and `down_proj@12-23` as int4 g128) read **+2.26 pp**
over its base on the census (`embed-q4-swap-2026-09-23.txt`), at a lower served b/param. The
same rule at 8B and 14B, where the head is untied and two tables go to q4:

| | 8B | 14B |
|---|---|---|
| base | `qwen3-8b-dclm-ft.bin`, 68.16, sha256 `783cef67…` | `qwen3-14b-dclm-ft.bin`, 74.20, sha256 `c825cb92…` |
| tables q8 → q4 (freed, served accounting) | 2 × 622,329,856 weights, **622.33 MB** | 2 × 777,912,320 weights, **777.91 MB** |
| int4 bought | `o_proj` (36), `down_proj@2-33` (32 layers) | `o_proj` (40), `down_proj@10-28` (19 layers) |
| int4 cost (`int4swap --price`, exact) | 159.84 + 32 × 14.123 = **611.78 MB** | 287.13 + 19 × 24.873 = **759.72 MB** |
| served b/param, base → package | 3.0683 → **3.0637** (`rtbits` on the file) | 2.7371 → ~2.727 (*computed*; `rtbits` on the file before launch) |
| kernel b/weight, tail f16 | **2.8058** | ~2.53 (*computed*) |

The window rule, fixed before any file was built: `o_proj` whole, then the largest window of
`down_proj` centred in the stack whose int4 cost fits in what the two tables free. No
per-layer measurement exists at either size to do better; the 4B window (12-23) is the middle
third, the same idea.

The files: `int4swap` from the checkpoint at the revision the census references used (8B
`b968826d`, 14B `40c06982`), then `embedq q4` on both tables. Every other record and tensor
is the base's, byte for byte. 8B: `qwen3-8b-sealed.bin`, 3,186,786,577 B, sha256
`91903db9…`.

## 2. Signed predictions

| quantity | 8B | 14B |
|---|---|---|
| paired gain over the base | **+1.8 pp** [+0.6, +3.0] | **+1.0 pp** [0.0, +2.0] |
| micro, full split | 70.0 [68.8, 71.2] | 75.2 [74.2, 76.2] |

Reasoning. The 4B package upgraded 19 % of the projection weights and gave +2.26 with one q4
table. The 8B upgrades 32 %, the 14B 21 %. Two things pull down: every lever so far shrank
with size (row scales +3.3 at 4B and 8B, +1.67 at 14B; the base gap to f16 closes, 9.0 → 6.9
→ 4.7 pp), and a q4 **untied** head has never been measured. The 8B's v+o+down restore on the
older untrained base read +4.18 (`vod-8b-2026-09-18.txt`), an upper bound this trained base
will not reach. Named against me: a loss at either size would mean the untied q4 head costs
more than the int4 tables bring.

## 3. Decision rule, per size

Census on both sides, constant codes, the 0.43 pp bar. Δ is the paired micro difference,
package minus base (`docs/data/mmlu-dumps/mmlu-{8b,14b}-dclm-ft-FULL.csv`), p the exact
McNemar.

| outcome | reading |
|---|---|
| Δ ≥ +0.43 and p < 0.05 | the package is that size's reference object |
| \|Δ\| < 0.43 or p ≥ 0.05 | nothing measurable; the FT file stays the reference |
| Δ ≤ −0.43 and p < 0.05 | not kept; the q4 head is the first suspect |

The pair measures f16 tables → q4 tables plus the int4, as at 4B: the base censuses scored
the file's f16 embedding and head (config none, dense reconstruction).

## 4. Controls

1. `oracle` MATCH on each job (hard rule 10).
2. The sealed file's bytes and sha256 on the mount equal to the Mac's.
3. `int4swap` reports 68 matrices (8B) and 59 (14B) transplanted; `embedq` requantizes
   exactly `model.embed_tokens.weight` and `lm_head.weight`.
4. 14,042 questions, fingerprint `a74a6d6213602979`, headers dtype f16, limit census, alloc
   flat, config none, dense reconstruction, kv f16.
5. Harness across images: the 8B base census ran on `a963a020`, the 14B's on `af907416`,
   the image these jobs use. `census-14b-base-2026-09-22.txt` control 4 found the two images
   identical on 2,280 questions, picks and logits.

## 5. What it will not establish

- The share of the q4 head against the int4 tables: one arm a size.
- Anything served: dense reconstruction only.
- Whether another window would do better: one window a size, fixed by rule.
