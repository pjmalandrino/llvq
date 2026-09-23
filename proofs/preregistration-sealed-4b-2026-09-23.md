# Preregistration. The sealed 4B through the served kernel

**Written on 2026-09-23, BEFORE the run, to be timestamped before launch.**
Operator go for the object ("tu me fais le 4B propre et scellé", 2026-09-23); the
launch waits for its own go on the cost below.

## 1. The object

`qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8c03f66dc…`: the 61.11
object with `o_proj` and `down_proj@12-23` transplanted to int4 g128 records
(`int4swap`, 48 matrices, 676,331,520 weights) and its embedding stored int4
g64 (`embedq q4`). 168 `Tetra` + 84 int4 records. 2.7320 b/param served,
2.5420 kernel b/weight (*computed*, `rtbits`,
`preregistration-embed-q4-swap-2026-09-23-ECARTS.md` É1). Served config
`configs/qwen3-4b-tetra-e4.json`: `tetra48`, embedding `q4`, `rot_share` 1,
`fuse` 0, `kv` f16.

Its dense reconstruction is the restore arm that scored 63.37 by construction:
`int4swap` and `LLVQ_RESTORE_Q4` call the same quantizer on the same f16
bytes, and `embedq` wrote the same embedding in both. The Mac checks that
before stamping (§5, control 1).

## 2. The job

One `l40sx1` job, image republished with this branch's code (the q4 embedding
kernels exist in no image yet). Stages, cheapest first:

1. `oracle`, then bytes and sha256 of both files on the mount.
2. Prefill gate at 203 tokens through `LLVQ_CONFIG`.
3. Arm S: the sealed file, served flags with `LLVQ_EMBED=q4`, 256 tokens
   against its dense arm in the same process.
4. Arm R: the 61.11 file, served flags with `LLVQ_EMBED=q8`, 256 tokens, the
   same way. It is the same-card, same-image, same-tile control that the
   2026-09-20 journal lacked.
5. MMLU dense, full split, on the sealed file.
6. MMLU through the kernel (`LLVQ_CONFIG`), 2,280 questions, on the sealed file.

Cost: ~2 h 10 on l40sx1, **~$3.90**, timeout 2 h 45, ceiling $4.95
(*estimated*: census 24 min as `6aaf099f`; the kernel pass at 2,280 questions
1.51 h as measured at 4B; each fusedrun arm ~5 min with load).

## 3. Signed predictions

| quantity | point | interval |
|---|---|---|
| dense census of the sealed file against the restore-arm dump | identical | 14,042 of 14,042 picks and logits |
| MMLU micro, dense census | 63.37 | exact |
| arm S, 256 tokens against its dense arm | identical | 256 of 256 |
| arm S tok/s over arm R tok/s | 1.00 | [0.90, 1.08] |
| arm S GB on the card | 1.37 | [1.34, 1.40] |
| kernel minus dense on the same 2,280 questions | 0.0 pp | [−0.5, +0.5], at most 10 discordant |

Reasoning for the speed. Per token the sealed object reads about 1.15 GB of
projections against 0.97, and a 0.22 GB head against 0.41: nearly the same
bytes. The int4 kernel is a plain matvec, cheaper to decode than `Tetra`, but
the 4B decode is dominated by launches (48 % of a token outside the matmuls,
2026-08 attribution). Named against me: arm S slower than 0.90 of arm R would
mean `tv_q4_h` at `down_proj`'s width (d_in 9,728, 38.9 KB staged) is a
bottleneck.

## 4. What decides

| outcome | reading |
|---|---|
| census identical, 256 tokens identical, kernel within [−0.5, +0.5] | **The sealed file is the 63.37 object and it serves.** It becomes the 4B served object at 2.7320 b/param |
| census not identical | The file is not the measured object; stop and find the byte that differs |
| tokens differ, or kernel outside the interval | A served path defect at the new shapes; the file stands, the serving does not |

## 5. Controls

1. On the Mac, before stamping: the dense reconstruction of the sealed file and
   of the restore arm (`qwen3-4b-dclm-ft-e4.bin` + `LLVQ_RESTORE_Q4`) give
   identical logits on 57 questions, on Metal.
2. `oracle` MATCH first (hard rule 10).
3. The prefill gate names `168 lone + 84 int4` and the served tile.
4. The census dump: dtype f16, limit census, alloc flat, config none, dense
   reconstruction, kv f16, fingerprint `a74a6d6213602979`.
5. The kernel dump: `# arithmetic=served kernel`, `# config=` the served file.

Control 1 was run before stamping (*measured*, Metal, 57 questions at limit 1,
`~/q4b-sealed-2026-09-23/metal-smoke/`): the sealed file's dense reconstruction
and the restore arm (`qwen3-4b-dclm-ft-e4.bin` with
`LLVQ_RESTORE_Q4=o_proj,down_proj@12-23` from `Qwen/Qwen3-4B` at `1cfa9a72`)
give 57 of 57 identical picks and all four logits identical, joined on subject,
index and qhash.

## 6. What it will not establish

- A full-split score through the kernel: 2,280 questions only. The full split
  through the kernel is ~11 h at 4B, not in this job.
- Anything about the 8B or the 14B.
- A perplexity: `ppl` does not read `LLVQ_CONFIG`.
