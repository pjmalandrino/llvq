# Preregistration. The paper table: served speed of the three sealed objects, the 4B competitors on the census, AWQ speed at 8B and 14B

**Written on 2026-09-25, BEFORE the runs, timestamped before the first job.**
Operator, 2026-09-25: a small paper with the data it needs and nothing more. The criteria are
MMLU (census), b/param over the whole model, and served speed (tok/s, GB on the card). At 4B
the objects face every competitor; at 8B and 14B, f16 and AWQ only. Go for the seven runs
below, in parallel. Announced: **~3.25 $**, 1 h 50 of card. Project total before: 225.19 $.

## 1. What is already in hand, and what is missing

| size | f16 | AWQ w4 g128 | IQ2_XXS | ours (sealed) |
|---|---|---|---|---|
| 4B | 70.14 census | 70.04 **sample only** | 38.87 **sample only** | 63.37 census, 2.7320; **speed missing** |
| 8B | 75.05 census | 73.79 census; **speed missing** | — | 69.58 census, 2.6953; **speed missing** |
| 14B | 78.88 census | 78.12 census; **speed missing** | — | 75.66 census, 2.7305; **speed missing** |

QTIP and the paper's LLVQ enter at 4B by citation of their Table 6; nothing is run for them.

## 2. The seven runs

| # | run | image | object | est. card time | est. cost | ceiling |
|---|---|---|---|---|---|---|
| 1 | served 4B sealed | llvq-runner-cuda, rebuilt with the q4 embedding | `sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…` | ~10 min | ~0.30 $ | 30 min, 0.90 $ |
| 2 | served 8B sealed | same | `sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin`, 2,815,098,745 B, `7bdb9a55…` | ~13 min | ~0.40 $ | 30 min, 0.90 $ |
| 3 | served 14B sealed | same | `sealed-14b-2026-09-23/qwen3-14b-sealed.bin`, 5,087,000,541 B, `61db37fe…` | ~30 min | ~0.90 $ | 45 min, 1.35 $ |
| 4 | AWQ 4B census | same | `Pier-Jean/qwen3-4b-awq-deq` @ `2c78de3b` (from `Qwen/Qwen3-4B-AWQ` @ `74d4bd2b`) | ~23 min | ~0.70 $ | 45 min, 1.35 $ |
| 5 | IQ2_XXS 4B census | `ghcr.io/ggml-org/llama.cpp:full-cuda` | `m4-iq2-cuda-1b57b7d3/qwen3-4b-iq2xxs.gguf`, sha256 `19a8ed49…` | ~15 min | ~0.45 $ | 40 min, 1.20 $ |
| 6 | AWQ speed 8B, vLLM | `vllm/vllm-openai:v0.26.0` @ `sha256:ffb2d59b…` | `Qwen/Qwen3-8B-AWQ` @ `4da05a8e`, f16 witness `Qwen/Qwen3-8B` @ `b968826d` | ~9 min | ~0.27 $ | 30 min, 0.90 $ |
| 7 | AWQ speed 14B, vLLM | same | `Qwen/Qwen3-14B-AWQ` @ `31c69efc`, f16 witness `Qwen/Qwen3-14B` @ `40c06982` | ~15 min | ~0.45 $ | 40 min, 1.20 $ |

Total ~3.47 $, above the 3.25 announced by 0.22 $ because the vLLM runs download 22 and 40 GB
(re-costed while writing this); worst case at every ceiling 7.80 $.

**Runs 1–3**, `ops/jobs/paper-served.sh SIZE=…`: oracle, bytes and sha256, the config written
from the command line (`configs/qwen3-{4b,8b,14b}-tetra-e4.json`), the prefill gate at 203
through `LLVQ_CONFIG`, then `fusedrun` 256 tokens at the served flags spelled out
(`tetra48`, `ROT_SHARE=1`, `FUSE=0`, `KV=f16`, **`EMBED=q4`**) against the dense arm of the same
process, then the same at `EMBED=f16`, the same-head arm hard rule 4 wants. Five rounds an arm,
median with its range. No MMLU door: the census numbers stay the dense reconstruction's.

**Run 4**, `ops/jobs/paper-awq4b-census.sh`: `mmlu` on the dequantized AWQ 4B, flat census,
the six header checks and the fingerprint `a74a6d6213602979` of every census.

**Run 5**, `ops/jobs/paper-iq2-census.sh`: `llama-server` on the GGUF, `ops/gguf_mmlu_thin.py`
on `mmlu-prompts-FULL.jsonl` (44,851,904 B, sha256 `9bd7a441…`), built by
`ops/mmlu_prompts.py` from `mmlu-4b-f16-FULL.csv` with **14,042 / 14,042 qhash** checked. The
same tool rebuilds the 2,280-question file of 2026-08-30 byte for byte.

**Runs 6–7**, `ops/jobs/paper-awq-speed.sh SIZE=…`: `ops/awq_speed.py`, the protocol of
2026-08-17 (`preregistration-awq-vllm-2026-08-17.md`): raw prompt ids, 128 tokens greedy,
2 discarded + 5 timed rounds, f16 witness in the same engine. The 8B is interleaved
(`f16` 0.55 + `awq_marlin` 0.25 of the card); the 14B's two engines do not fit together
(29.5 + 10 GB), so it runs one arm per process and `--merge`, labelled "rounds not interleaved".
Only `awq_marlin`: at 4B the `awq` arm routed to Marlin anyway.

**The 8B AWQ pin.** `awq_speed.py` refused the 8B (`pinned=False`) because its revision never
passed `awq_dequant check`. The dequantized checkpoint our 8B census scored,
`Pier-Jean/qwen3-8b-awq-deq`, records `revision: main` in its `RECONSTRUCTION.json`, and the
Hub's `Qwen/Qwen3-8B-AWQ` has not moved since 2025-05-21 (`4da05a8e`, *measured*
2026-09-25). So `main` then was `4da05a8e`, and the script now pins it with that reason written
beside it.

## 3. Signed predictions

Served (runs 1–3), q4 arm; the references are *measured* at the served tile 64 except the 4B's
98.3, which was measured at tile 128 before the 2026-09-20 policy:

| | reference | tok/s predicted | GB on the card predicted |
|---|---|---|---|
| 4B | FT 98.3 tok/s, 1.39 GB (q8 table, tile 128) | **110** [95, 125] | **1.38** [1.30, 1.50] |
| 8B | FT 90.6 tok/s, 3.15 GB (q8 tables) | **100** [88, 112] | **2.77** [2.62, 2.92] |
| 14B | base 55.4 tok/s, 5.06 GB (q8 tables) | **58** [50, 66] | **5.04** [4.85, 5.25] |

Reasoning. The q4 tables halve the head's read (0.41 → 0.22 GB at 4B, 0.66 → 0.35 at 8B,
0.83 → 0.44 at 14B, *computed*), and the int4 records that replace `Tetra` matrices are read by
a kernel that is not decode-bound. Bytes per token move little; the int4 share moves time the
right way. GB on the card: the table saving (−0.19, −0.62, −0.78 GB) against the int4 added
(+0.18, +0.24, +0.76 GB, the `--price` figures).

Identity: **256 tokens identical to the dense arm** at each size, q4 arm. The f16 arm is read,
not predicted.

Census (runs 4–5): AWQ 4B **69.9** [69.2, 70.6]; IQ2_XXS 4B **39.0** [37.5, 40.5]. At 4B the
census read 0.18 under the sample for f16 and 0.85 over it for the served object, so the
sample's figure is the point and its bar the width.

vLLM (runs 6–7), `awq_marlin` absolute: 8B **90 tok/s** [75, 110], 14B **55** [45, 70]
(the 4B's 200.5 tok/s for 2.67 GB scaled by bytes, *estimated*). AWQ/f16 inside vLLM: 8B
×2.4 [2.0, 2.8], 14B ×2.6 [2.1, 3.1].

## 4. Decision rules

| outcome | action |
|---|---|
| a served arm diverges from the dense arm **before token 32** | the speed of that size is not published until the divergence is explained |
| divergence at token 32 or later | the position is published beside the speed, as for every served object since 2026-09-10 (a tie-break, not a defect) |
| prefill gate refuses (different argmax) | that size's run is a failure, its speed is not published |
| a census header or the fingerprint differs from §2 | the dump is not paired and its number not published |
| IQ2_XXS refuses (a letter missing from the top-100) | the 4B keeps its sample figure, labelled so |
| vLLM refuses or violates §7 of its script | the AWQ speed cell stays empty at that size |

Nothing here edits a published MMLU: the table's quality numbers stay the dense census ones.

## 5. Controls

1. `oracle` MATCH on runs 1–4.
2. Each sealed file's bytes and sha256 on the mount equal to the Mac's.
3. Runs 1–3: the log's tile line says 64 served for sm_89; the embedding line says `q4 g64`;
   the fused/int4 split is read from the log (*computed* before: 4B 168 + 84, 8B 199 + 53,
   14B 181 + 99).
4. Runs 1–4 on the rebuilt image, whose Space sha is recorded at launch and must not move
   while any of them is queued.
5. Run 5: the GGUF's sha256 on the mount, `four letters in the top-100` on 14,042 / 14,042.
6. Runs 6–7: the image tag and digest, resolved quantization (`awq_marlin`), prefix caching
   read back as off, 128 tokens every round.

## 6. What this will not establish

- MMLU through the served kernel: the census numbers are the dense reconstruction's, and the
  256-token identity is the link between the two, as for every object before.
- A ratio across stacks: AWQ tok/s is vLLM's, ours is `fusedrun`'s (hard rule 5).
- Anything at batch > 1 or long context.
