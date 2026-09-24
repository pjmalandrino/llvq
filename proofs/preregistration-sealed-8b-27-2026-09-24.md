# Preregistration. The sealed 8B brought to 2.7 b/param: `o_proj` or `down_proj`?

**Written on 2026-09-24, BEFORE the runs, timestamped before the first job.**
Operator, 2026-09-24: the 8B cannot stay above 3 b/param, bring it to ~2.7 like the 4B and the
14B; go for arms A and B. Announced cost: two 8B censuses, ~$0.98 each (timeout 60 min,
ceiling $1.80 each): **~$1.96**. Project total before: $223.19.

## 1. The claim on trial

The sealed 8B reads 70.08 at 3.0637 b/param (`sealed-8b-14b-2026-09-23.txt`). Its untied head
makes two tables, 15 % of the parameters; at q4 they alone weigh 0.68 b/param, and the FT file
with q4 tables and no new int4 is at 2.4662 (*computed*). At 2.70 there is room for about
239 MB of int4 instead of 612. Two ways to spend it, same base (`qwen3-8b-dclm-ft.bin`, 68.16),
same q4 tables, built by `int4swap` then `embedq q4`:

| arm | int4 bought | int4 bytes (exact) | b/param served (`rtbits`) | file |
|---|---|---|---|---|
| **A** | `o_proj` (36) + `down_proj@15-20` (6) | 244,580,688 | **2.7047** | 2,819,588,161 B, sha256 `0c6b08b3…` |
| **B** | `down_proj@10-26` (17) | 240,091,272 | **2.6953** | 2,815,098,745 B, sha256 `7bdb9a55…` |

A keeps the recipe of the three sealed objects, shortened; B asks which type is worth more
per byte at 8B. The windows are centred; B's is the one announced to the operator (10 to 26).

## 2. Signed predictions

| quantity | A | B |
|---|---|---|
| paired gain over the FT base (68.16) | **+0.9 pp** [−0.1, +1.9] | **+0.7 pp** [−0.3, +1.7] |
| paired difference to the 3.06 sealed file (70.08) | −1.0 [−1.8, −0.2] | −1.2 [−2.0, −0.4] |
| A minus B | **+0.2** [−0.6, +1.0] | |

Reasoning. The full package bought 612 MB of int4 and read +1.92, net of whatever the untied
q4 head costs (never measured alone). About 40 % of that int4 is left, and the gain of
quantization upgrades is usually concave in the bytes spent, so a little more than 40 % of the
int4 share. `o_proj` read more per byte than `down_proj` on the 4B sample
(`q5-alloc-int4-2026-09-16.txt`, exploration only), hence A a little above B.

## 3. Decision rule

Census on both arms and on the references, constant codes, the 0.43 pp bar, exact McNemar.

| outcome | action |
|---|---|
| the better arm beats the FT base by ≥ +0.43, p < 0.05 | it is the 8B reference at ~2.7 b/param; the 3.06 file stays as the high-memory variant |
| neither beats the FT base | the 8B at ~2.7 is the FT file with q4 tables and no new int4 (2.4662), to be measured before it is claimed |
| \|A − B\| < 0.43 | the tie goes to A (the recipe of the other sizes) |

## 4. Controls

1. `oracle` MATCH on each job.
2. Each file's bytes and sha256 on the mount equal to the Mac's.
3. `int4swap`: 42 matrices for A, 17 for B; `embedq` requantizes exactly the two tables.
4. 14,042 questions, fingerprint `a74a6d6213602979`, headers as in the 8B/14B prereg.
5. Same image as the 70.08 census (`af907416`).

## 5. What it will not establish

- The cost of the q4 head alone (arm D, not run).
- Whether other windows at the same bytes do better: one window an arm.
- Anything served.
