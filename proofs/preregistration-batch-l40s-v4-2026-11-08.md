# Preregistration. Batch v4: what slowed batch-l40s-v3, the int4 word order or the two-CTA entries (4B)

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-l40s-v4-2026-11-08-ECARTS.md`.

Operator envelope for the branch: $5; spent $3.18. Cost: **$0.51 expected, $0.90 at worst** (*estimated*); the branch
stays under $5 at worst ($4.08).

**Base commit:** the commit of `claude/batch-l40s` that carries this file and `ops/jobs/batch-l40s-v4.sh`.
**Image:** the Space sha built from it, passed as `IMAGE_SHA`.

## 1. Why this run exists

batch-l40s-v3 changed two things at once against batch-l40s-v2, the int4 words in 4E v9's tiled order (staged a word a
copy) and every entry at two CTAs a SM, and read the door 7 to 26 % slower at B >= 32. The served entries are back to
v2's (`gemm`); the two-CTA ones are the measurement arm `gemm-occ` (host-proved, `the_two_cta_entries_match_the_reference_too`).

## 2. What runs

One job, l40sx1, `ops/jobs/batch-l40s-v4.sh`, timeout 30 min, the sealed 4B only:

1. (a) `batchrun check gemm` under `qwen3-4b-tetra-e4-4e-v9-batch.json` (tiled words, served entries);
2. (b) `batchrun door 1,32,64,128 gemm,gemm-occ 256 3` under `qwen3-4b-tetra-e4-batch.json` (file order) and under
   `qwen3-4b-tetra-e4-4e-v9-batch.json` (tiled): two arms by two orders, in one job;
3. (c) `llvq serve` under `LLVQ_BATCH=128:1024` with the v9-batch config, `batch_serve_bench.py --batches 1,8,32,128
   --rounds 2`.

## 3. Gates that void the run

The image builds; oracle MATCH; no `LLVQ_*` in the job environment; the object, the two configs and the bench at their
sha256; the check log names the int4 layout tiled; every kernel at 0 B local; every request answers 256 tokens.

## 4. The verdicts this run reads

- **Order:** gemm under tiled against gemm under file order, at B = 32, 64, 128. The tiled order costs if its median is
  below by more than the two ranges.
- **Occupancy:** gemm-occ against gemm, under each order, same rule.
- **The served path:** the v9-batch config serves the faster of the two orders' measured costs only if the operator
  says so; this run reports, it does not switch a config.
- check under 1e-2; client B = 1 within 10 % of 223.4 tok/s a request.

## 5. Signed predictions

*Estimated*.

| quantity | point | interval |
|---|---|---|
| gemm, file order, at 128 (batch-l40s-v2 read 4,493.6) | **4,490** | [4,350, 4,650] |
| gemm, tiled order, at 128 | **4,300** | [3,900, 4,600] |
| gemm-occ against gemm, file order, at 128 | **-8 %** | [-15, +3] |
| client B = 1, tok/s a request | **224** | [215, 232] |
| client B = 128, aggregate | **3,100** | [2,700, 3,500] |
