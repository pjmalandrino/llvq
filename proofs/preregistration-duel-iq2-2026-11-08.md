# Preregistration. The llama.cpp duel: IQ2_XXS and the rung at our bits against the served Tetra files, 4B, 8B, 14B

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-duel-iq2-2026-11-08-ECARTS.md`.

Cost: **$13.80 expected, $19.80 at worst** on l40sx1 (*estimated*, §8). No envelope is set for this work; nothing
launches before the operator's go. Registry total before this run: $280.04 over 278 rows (*computed*,
`docs/data/jobs.csv`).

**Base commit:** the commit of `claude/duel-iq2-xxs-c5d375` that carries this file, `ops/jobs/duel-iq2.sh`,
`ops/jobs/duel-tetra.sh` and `ops/duel/`.
**llama.cpp:** release v0.9.0, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` (2026-11-04), built in each job with
`GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=89`, in `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
**Our image:** Space `Pier-Jean/llvq-runner-cuda` at `c16c4c73` (built from `f63b590`), passed as `IMAGE_SHA`.

## 1. Question

At the same bits per parameter, on the same card, does a reader get more from llama.cpp's 2-bit GGUF or from our
served Tetra files, in decode speed, card memory and quality, at 4B, 8B and 14B?

Today one line answers it, at 4B only: IQ2_XXS at 2.48 b/param, MMLU 39.78, 312.9 tok/s, 1.25 GB (`docs/ETAT.md` §3).
Its GGUF was calibrated on C4, made by an August build, timed with `llama-bench` tg128 on another protocol, and has
no GSM8K, no perplexity on our windows and no peak memory. Nothing exists at 8B and 14B. It is the first question a
technical reader asks before we communicate.

## 2. The arms

Per size, three llama.cpp arms and our served file. Base checkpoints at the revisions the sealed files were encoded
from: `Qwen/Qwen3-4B@1cfa9a72`, `Qwen/Qwen3-8B@b968826d`, `Qwen/Qwen3-14B@40c06982`.

| arm | what it is |
|---|---|
| `own-iq2_xxs` | `llama-quantize --imatrix <ours> f16.gguf IQ2_XXS`, made in the job |
| `own-<r*>` | the rung at our bits, same imatrix, chosen in the job by the rule below |
| `unsloth-ud-iq2_xxs` | `unsloth/Qwen3-{4B,8B,14B}-GGUF`, file `*-UD-IQ2_XXS.gguf`, repo shas `22c9fc8a`, `a6adef13`, `a04a82c4`, sizes 1,255,345,952 / 2,605,221,696 / 4,470,113,664 B, checked |
| Tetra | the sealed file under `configs/qwen3-<size>-tetra-e4-4e-v9.json`, the served door |

**The imatrix.** `llama-imatrix -c 2048` over our calibration text: the contiguous prefix of 64 x 2,048 = 131,072
tokens of `HuggingFaceTB/dclm-edu` shard `000_00000` at `dbad8ad7`, the set every sealed file was encoded on. The bounded
read reproduces the sealed encodes' line (825,408 chars, 127 rows); the text is the decode of those 131,072 ids and
re-encodes to them all (`ops/duel/inputs.py`, *measured* on the Mac). Same corpus and same volume as our calibration.

**The rung at our bits (r\*).** Our file b/param = file bytes x 8 / parameters, the accounting a GGUF reads:
1,418,224,685 / 2,815,098,745 / 5,087,000,541 B over 4,022,468,096 / 8,190,735,360 / 14,768,307,200 parameters, so
**2.8206 / 2.7495 / 2.7556** (*computed*). `rtbits` reads 2.73 / 2.70 / 2.73 for the same files under its own
accounting (`docs/ETAT.md` §2); the duel uses the file accounting on both sides, which costs us 0.05 to 0.09. In the job, `llama-quantize --dry-run` gives the tensor bytes of IQ2_XS, IQ2_S, IQ2_M and IQ3_XXS;
r\* is the one nearest ours in b/param, the brief's "closest level". Run on the Mac before this file was stamped
(*computed*, the f16 GGUF of the same revisions, no imatrix needed for a dry run):

| rung | 4B b/param | 8B b/param |
|---|---|---|
| IQ2_XXS | 2.4675 | 2.4263 |
| IQ2_XS | 2.6812 | 2.6275 |
| IQ2_S | 2.8069 | 2.7922 |
| IQ2_M | 2.9972 | 2.9750 |
| IQ3_XXS | 3.4198 | 3.3944 |

These count tensor bytes only; the GGUF's tokenizer and metadata add about 0.01 b/param. The rule therefore picks
IQ2_S at 4B (2.81 against our 2.82, within 0.01 once the metadata is counted) and at 8B (2.79 against 2.75, llama.cpp 0.04 above us). A rule
"at or above ours" would have picked IQ2_M at 4B, 0.18 above us; it was set aside for that reason.

## 3. What runs

`ops/jobs/duel-iq2.sh` (one job a size, l40sx1) and `ops/jobs/duel-tetra.sh` (one job, three sizes, our image).

1. **Inputs**, built on the Mac by `ops/duel/inputs.py` and shipped by sha256:
   `ppl-tokens.u32` (the 12 x 4,096 WikiText-2 test windows, fingerprint `3f1baca9033bf251`, the one
   `ppl-scelles-2026-10-04` prints), `gsm8k-prompts.jsonl` (1,319 prompts as ids, qhash 1,319/1,319 and run fingerprint
   `bfa9135ce4b55c75` equal to our 4B census dump), `mmlu-prompts-ids.jsonl` (14,042 prompts as ids, qhash
   14,042/14,042, run fingerprint `a74a6d6213602979` equal to our census dumps), `calib-dclm.txt`.
2. **Perplexity**: `ops/duel/ppl_windows.cpp`, linked to the pinned libllama, scores every next-token position of
   each window (4,095 a window, no BOS), as `bin/ppl` does. `llama-perplexity` is not used: it tokenizes on its own
   and scores only the second half of each window. Checked on the Mac: the f16 GGUF of Qwen3-0.6B reads 18.9312
   against 18.9339 for our candle f16, -0.014 % (*measured*).
3. **Speed**: `llama-bench -ngl 99 -fa on -p 0 -n 256 -d <5|8192> -r 5 -o json`, one process per depth, median and
   range of the five `samples_ts`. Ours: `fusedrun <sealed> 256` under `LLVQ_CONFIG` and `LLVQ_CONTEXTS=<5|8192>`, one
   process per length, 1 discarded + 5 rounds, the decode rate it prints. Both are decode-only rates after the prompt.
4. **Card memory**: `nvidia-smi memory.used` every 100 ms around each speed process; peak minus the idle reading, in
   GB (MiB x 1,048,576 / 1e9). The same instrument on both sides. File size in bytes, b/param as in §2.
5. **Batch, 4B only** (ours exists at 4B only, `batch-tetra-2026-11-07`): `llama-batched-bench -npp 5 -ntg 256
   -npl 1,8,32 -fa on`, aggregate = B x 256 / (t_pp + t_tg), for `own-iq2_xxs` and `own-<r*>`.
6. **MMLU census**: llama-server, 16 slots, the rule of `ops/gguf_mmlu_thin.py` (top-100 log-probs of ' A' ' B' ' C' '
   D', argmax) on the shipped ids, by `ops/duel/mmlu_llamacpp.py`; dump read by `mmlupair` against our census dumps
   (`embed-q4-swap-2026-09-23-brut/mmlu-4b-embed-q4-swap-FULL.csv`, `sealed-8b-27-2026-09-24-brut/mmlu-B/...`,
   `sealed-8b-14b-2026-09-23-brut/mmlu-14b/...`).
7. **GSM8K census**: llama-server, 16 slots, the shipped ids, greedy, at most 1,024 new tokens, stop on
   `<|im_end|>`/`<|endoftext|>`, by `ops/duel/gsm8k_llamacpp.py`; `gsm8kpair` re-grades and pairs against our census
   dumps (`gsm8k-wave1-2026-09-26-brut/run1-served/...`, `gsm8k-wave2-2026-09-26-brut/run2-served-8b/...`,
   `.../run3-served-14b/...`).

Our perplexity, MMLU and GSM8K are not re-measured: 12.58 / 9.71 / 8.44, 63.37 / 69.58 / 75.66, 82.49 / 88.63 / 92.04
(`docs/ETAT.md` §2, the forms named there).

## 4. Gates that void a number

1. Inputs and scripts at their sha256 in the job; the llama.cpp checkout at `cd959bae`; `llama-bench` reports build
   `cd959ba` and flash attention on.
2. Every perplexity run prints fingerprint `3f1baca9033bf251`; every MMLU dump ends on `a74a6d6213602979` with 14,042
   questions and 14,042 four-letter answers in the top 100; every GSM8K dump ends on `bfa9135ce4b55c75` with 1,319 rows.
3. **f16 gate, per size**: the f16 GGUF on our windows within +-0.5 % of our f16 reading of the same checkpoint
   (12.2361, 8.9899, 7.9820). Outside: that size's llama.cpp perplexities are reported, not compared.
4. The unsloth file at its bytes; each own GGUF's sha256 recorded.
5. Ours: oracle MATCH, the sealed file at its sha256, the config at its sha256, the served-config line, and 4E equal
   to the eager path, 256 of 256, at each length.

## 5. What gets published, and what does not get compared

Published per size and arm: b/param (file), file GB, peak card GB at 5 and at 8,192, tok/s at 5 and at 8,192 (median
[range]), perplexity, MMLU micro, GSM8K; GB/s = file bytes x tok/s, *computed*. Paired gaps with their 95 % interval:
MMLU (`mmlupair`, McNemar in micro), GSM8K (`gsm8kpair`), perplexity (paired t over the 12 window NLLs).

Not formed: any x between llama.cpp and our stack (rule 5). Speeds are compared in levels and GB/s. Our peak memory
includes the eager check run in the same process. Our quality scores come from the forms `docs/ETAT.md` §2 names
(MMLU and perplexity on the dense reconstruction, GSM8K through the kernel), theirs from the served engine;
GSM8K in llama-server runs 16 slots at a time, as the vLLM arms ran batched.

## 6. Decision rule

Per size, per metric, Tetra against each llama.cpp arm:

| metric | Tetra wins | llama.cpp wins | otherwise |
|---|---|---|---|
| MMLU, GSM8K | paired 95 % interval of (Tetra - arm) above 0 | interval below 0 | not separated |
| perplexity | paired 95 % interval of the window NLL gap below 0 | above 0 | not separated |
| decode tok/s, at 5 and at 8,192 | Tetra's range above the arm's | below | ranges overlap: not separated |
| peak card GB, file GB, b/param | lower by more than 0.05 GB / 0.01 b/param | higher by more | tie |

The headline is the duel at our bits, Tetra against `own-<r*>`. A sentence for the outside names the arm, the level
of both sides and the gap, never a ratio. If the f16 gate fails at a size, its perplexity row reads "not compared".
Otherwise, any outcome the table does not cover: reported as is, operator decision.

## 7. Signed predictions

*Estimated*. Bits from llama.cpp's per-tensor type rules (`src/llama-quant.cpp` at the pinned commit) and the 4B
IQ2_XXS of August (2.479 b/param); speed from the August 312.9 tok/s scaled by bytes; quality from the August 4B line
and the usual shrink of the 2-bit gap with size. The known flaw: the dclm imatrix and the newer build may move every
quality number of llama.cpp by several points either way.

| quantity | 4B | 8B | 14B |
|---|---|---|---|
| r\* | **IQ2_S** (computed) | **IQ2_S** (computed) | **IQ2_S** |
| own-iq2_xxs b/param (file) | **2.48** [2.47, 2.49] | **2.44** [2.43, 2.45] | **2.38** [2.30, 2.45] |
| own-r\* b/param (file) | **2.82** [2.81, 2.83] | **2.80** [2.79, 2.81] | **2.78** [2.70, 2.85] |
| own-iq2_xxs tok/s at 5 | **315** [290, 345] | **185** [165, 210] | **115** [100, 130] |
| own-iq2_xxs tok/s at 8,192 | **250** [210, 290] | **160** [135, 185] | **100** [85, 115] |
| Tetra tok/s at 5 / 8,192 | **226 / 161** [220-232 / 155-167] | **144 / 110** [139-150 / 105-115] | **87 / 71** [84-90 / 68-75] |
| own-iq2_xxs peak GB at 8,192 | **3.2** [2.6, 4.0] | **4.6** [3.9, 5.5] | **6.8** [6.0, 7.8] |
| Tetra peak GB at 8,192 | **3.6** [3.0, 4.5] | **4.9** [4.2, 5.8] | **7.3** [6.5, 8.3] |
| own-iq2_xxs ppl | **30** [24, 38] | **16** [12, 22] | **12** [10, 15] |
| own-iq2_xxs MMLU | **40** [36, 45] | **52** [45, 60] | **62** [55, 68] |
| own-iq2_xxs GSM8K | **40** [20, 60] | **60** [40, 75] | **75** [60, 85] |
| own-r\* ppl | **18** [15, 24] | **12** [10.5, 14] | **9.8** [9.0, 11.0] |
| own-r\* MMLU | **50** [44, 56] | **60** [54, 66] | **68** [63, 72] |
| own-r\* GSM8K | **60** [45, 72] | **75** [62, 85] | **85** [77, 90] |
| unsloth vs own-iq2_xxs, MMLU | within 3 pp | within 3 pp | within 3 pp |

The signed verdict: llama.cpp wins decode speed at every size and both lengths, and its IQ2_XXS wins memory; Tetra
wins MMLU, GSM8K and perplexity against every llama.cpp arm at every size, including r\* at our bits; at 8,192 the
speed gap narrows.

## 8. Cost

Per llama.cpp job: apt and build ~12 min, pip and download 3-6 min, conversion 2-8 min, imatrix 1-3 min, quantize
two rungs 8-30 min on 8 vCPU, per arm perplexity and speed 2-4 min, MMLU census 13 / 25 / 45 min, GSM8K census 5 / 9
/ 15 min with 16 slots. **4B ~80 min, $2.40, timeout 2 h ($3.60); 8B ~130 min, $3.90, timeout 3 h ($5.40); 14B ~220
min, $6.60, timeout 5 h ($9.00).** Ours: ~30 min, $0.90, timeout 60 min ($1.80). Total **$13.80 expected, $19.80 at
worst** at $1.80/h (*estimated*). Checked against the registry after the jobs.
