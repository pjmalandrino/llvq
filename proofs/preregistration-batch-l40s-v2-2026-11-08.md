# Preregistration. The batched GEMM's second pass: f16 activation and 128-row tiles, the gate again, the step's attribution

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-l40s-v2-2026-11-08-ECARTS.md`.

Operator envelope for the batching branch: $5 (relayed on 2026-11-08); spent $1.48 (batch-l40s-2026-11-08). The
operator asked for this second pass ("Oui, ici"). Cost: **$1.35 expected, $2.25 at worst** (*estimated*, section 7);
the branch stays under $5 at worst ($3.73).

**Base commit:** the commit of `claude/batch-l40s` that carries this file and `ops/jobs/batch-l40s-v2.sh`.
**Image:** the Space sha built from it, passed as `IMAGE_SHA`; the launcher refuses another.

## 1. Why this run exists

- batch-l40s-2026-11-08 measured the GEMM about two times below AWQ in aggregate at every size and B, and read (by
  arithmetic, not profile) that each CTA of 64 rows re-read its K slice of the activation in f32: at 128 rows about 16
  bytes of L2 a cycle a SM to feed the tensor cores, more than the L2 delivers.
- This pass: the rotated rows of a Tetra projection are converted to f16 once a group (one more launch a group), the
  kernel stages the activation in f16 for both kinds, and a CTA owns 128 rows (8 warps). The activation re-read drops
  four times. Host-proved, 19 of 19 mutants killed; NVRTC 12.4, sm_89 and sm_80, no local memory.
- The arithmetic moved (the Tetra activation now rounds to f16 before the MMA; the K chunks changed), so the gate is
  read again.

## 2. What runs

One job, l40sx1, `ops/jobs/batch-l40s-v2.sh`, timeout 75 min. Oracle; the three sealed files and six configs at their
bytes and sha256. Then for each size, 4B, 8B, 14B, under its batch config:

1. (a) `batchrun kernels 1,8,32,128 gemm 10`: the projections of one step alone (every layer's q,k,v; o; gate,up;
   down, and the head), random rows, rotation and conversion included, ms a step per group;
2. (b) `batchrun check gemm`, as in batch-l40s;
3. (c) `batchrun door 1,8,16,32,64,128 gemm 256 3`, `LLVQ_BATCH_REF` the size's served fingerprint from batch-l40s
   (4B 61d0654532d35120, 8B c817aeaf2f5d1116, 14B dc4f9ab9aa24c4d3);
4. (d) `batchrun ppl 2048 16 gemm`; at 4B also `ppl 256 128 gemm`;
5. (e) at 4B, `batchrun gsm8k 50 gemm 50` against the served pilot dump (sha256 `a7203fa6…`, 40/50).

## 3. Gates that void the run

1. The image builds; `oracle` prints MATCH; no `LLVQ_*` in the job environment.
2. The objects and the configs at their bytes and sha256.
3. The GSM8K dump ends on fingerprint `94c7b98876288901`.
4. Every kernel loads with 0 B of local memory.

## 4. The verdicts this run reads

- **The gate (4B), all three required, as in batch-l40s:** check under 1e-2; pooled paired perplexity of gemm against
  rows inside +-0.12 % at 16 x 2,048 and 128 x 256; GSM8K correct count not below 40/50 by more than 2.
- **8B and 14B:** check under 1e-2 and the band at 16 x 2,048 keep their batch configs licensed.
- **The pass against the first GEMM:** aggregate tok/s at B = 32 and 128 at each size against batch-l40s's (same card
  type, same image recipe, another job): reported with both ranges; the pass is kept if the 4B at B = 128 gains and
  no B loses more than 5 %.
- **Attribution:** projections alone (a) against the door's step time (c), per B: the share of the step that is not
  the projections.

## 5. Outcomes reported, never gated

Per size and B: aggregate tok/s, range, tok/s a sequence, door rows equal to the fingerprint; (a) per group; check per
projection; perplexity per window; GSM8K discordant pairs. GB/s = weights x tok/s a sequence, *computed*; against
AWQ from batch-l40s-2026-11-08 (no x across stacks, rule 5).

## 6. Signed predictions

*Estimated*. The B-dependent part of the 4B step (about 22 ms at B = 128 in batch-l40s) shrinks by a half to two
thirds; the extra conversion costs about 0.5 ms a step at every B.

| quantity | point | interval |
|---|---|---|
| 4B aggregate tok/s at B = 1 / 8 / 32 / 128 | **102 / 790 / 2,500 / 6,500** | [90, 115] / [680, 900] / [2,000, 3,300] / [4,200, 9,000] |
| 8B at B = 128 | **4,200** | [2,700, 6,000] |
| 14B at B = 128 | **2,500** | [1,600, 3,700] |
| 4B projections alone at B = 128, ms a step | **8** | [4, 16] |
| 4B projections alone at B = 1, ms a step | **3.5** | [2, 6] |
| check, worst relative RMS | **6e-4** | [2e-4, 3e-3] |
| pooled ppl, gemm against rows, every width and size | **within +-0.02 %** | [-0.08, +0.08] |
| GSM8K: correct | **40/50** | [37, 43] |

## 7. Cost

Image pull and 13 model loads about 15 min; the 4B's parts about 12 min, the 8B's about 9, the 14B's about 12 (the
multi-row arm of the perplexity is most of it). About 45 min at $1.80/h: **$1.35**; timeout 75 min, **$2.25** at
worst (*estimated*).
