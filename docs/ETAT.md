# Project state as of 2026-11-07

## 1. The project

LLVQ quantizes LLM weights on the Leech lattice, in Rust, to fit larger models on local hardware. The repository
holds the quantizer, the file format, fused CUDA and Metal kernels, a runtime and a Python reader. How it works is in
[`how-it-works.md`](how-it-works.md).

Three Qwen3 files are sealed at about 2.7 bits per parameter: 4B, 8B and 14B. They are public on the Hugging Face Hub
as `Pier-Jean/Qwen3-{4B,8B,14B}-LLVQ-Tetra-sealed`. Paper 1 is on Zenodo
([10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)). Paper 2, *Tetra: Serving Leech-Lattice
Quantized LLMs at 2.7 Bits per Parameter*, is in [`paper2/`](../paper2/README.md); its version 2 is attached to
release `v0.0.2` of 2026-10-06.

## 2. Served configuration

All three sizes share one setting: layout `tetra48`, `LLVQ_EMBED=q4`, `LLVQ_ROT_SHARE=1`, `LLVQ_FUSE=0`,
`LLVQ_KV=f16`. The config files are in [`configs/`](../configs/README.md), which is the authority on them.

- **CUDA, 4B, 8B and 14B**: `configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json`, served since 2026-11-07. Attention
  `kernel`, decode `persistent` (4E, one launch a token), rows `v9` (int4 words tiled once at load).
- **Mac, 4B**: the MLX door, `ops/llvq-mlx` under `configs/mlx/qwen3-4b-tetra-e4-mlx.json`, served since 2026-11-07.
- **Runtime**: `llvq` (crate `llvq-runtime`) pulls the three files and serves them with Ollama's and OpenAI's APIs. On
  CUDA it loads the v9 configs; on Metal it runs the expanded config through the Rust engine, not MLX.
- **Without a GPU**: the runtime's CPU backend since 2026-11-07, `fused_cpu` on the expanded e4 configs, at f32. It
  reads the Tetra codes at every matvec: 8.6 tok/s in 1.86 GB on the 4B at 12 threads of an M3 Max, against 0.71
  tok/s in 10.6 GB for the dense reconstruction it replaced; perplexity -0.0021 % of the dense f32 over 12 windows
  (*measured*, [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt)). Never run on x86.

| size | sealed file | b/param | MMLU | GSM8K | ppl | tok/s served | GB on card |
|---|---|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 82.49 | 12.58 | 221.5 | 1.38 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 9.71 | 142.5 | 2.76 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 8.44 | 86.3 | 5.04 |

- b/param: *computed* by `rtbits`, whole model, int4 tables included.
- MMLU: *measured*, all 14,042 test questions, 5-shot, micro average
  ([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt), [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt),
  [sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)).
- GSM8K: *measured* through the kernel, all 1,319 test problems, zero-shot, greedy
  ([gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)).
- Perplexity: *measured*, raw WikiText-2 test split, 12 windows of 4,096, f16
  ([ppl-scelles](mesures/ppl-scelles-2026-10-04.txt)).
- Speed and GB: *measured* on one L40S at each config's door, batch 1, 256 greedy tokens after a 5-token prompt,
  median of five rounds ([served-v9](mesures/served-v9-2026-11-07.txt),
  [-8b](mesures/served-v9-8b-2026-11-07.txt), [-14b](mesures/served-v9-14b-2026-11-07.txt)).

**The scores and the speeds come from different forms.** MMLU, GSM8K and perplexity ran under the `expanded`
attention form of paper 2, and MMLU and perplexity on the dense reconstruction. The served form adds the `kernel`
attention and 4E. It returns the eager path's tokens, and its attention is served under a perplexity tolerance (§4).
Its only task reading is a 50-problem GSM8K pilot per size: 40, 46 and 46 right at 4B, 8B and 14B, the same
scores as the earlier configs on those problems. Paper 2 and its
cited Hub cards report 113.8, 95.0 and 57.2 tok/s under the expanded form
([paper-table](mesures/paper-table-2026-09-25.txt)); they do not move.

The Hub's moving repository `Qwen3-4B-LLVQ-Tetra-served` still carries the grouped config of 2026-11-02 at 133.3
tok/s ([served-grouped](mesures/served-grouped-2026-11-02.txt)). The repositories paper 2 cites, and the safetensors
one, are frozen.

## 3. Against the baselines

| size | arm | b/param | MMLU | GSM8K | ppl | tok/s | GB |
|---|---|---|---|---|---|---|---|
| 4B | FP16, vLLM | 16.00 | 70.14 | 92.12 | 12.24 | 83.1 | 8.04 |
| 4B | AWQ w4g128, vLLM | 5.30 | 68.14 | 89.01 | 13.52 | 200.5 | 2.67 |
| 4B | IQ2_XXS, llama.cpp | 2.48 | 39.78 | | | 312.9 | 1.25 |
| 8B | FP16, vLLM | 16.00 | 75.05 | 93.25 | 8.99 | 46.3 | 16.38 |
| 8B | AWQ w4g128, vLLM | 5.96 | 73.79 | 92.95 | 9.42 | 123.2 | 6.10 |
| 14B | FP16, vLLM | 16.00 | 78.88 | 95.30 | 7.98 | 25.8 | 29.54 |
| 14B | AWQ w4g128, vLLM | 5.40 | 78.12 | 95.38 | 8.29 | 77.8 | 9.98 |

All *measured* on the L40S; each cell's journal is in `docs/data/paper2-results.csv` and `docs/data/paper2-ppl.csv`.
Speeds from different engines are never divided (rule 5).

Paired gaps, our file below the arm (MMLU *computed*, `docs/data/paper2-gaps.csv`; GSM8K *measured* by `gsm8kpair`):

| size | MMLU vs FP16 | MMLU vs AWQ | GSM8K vs FP16 | GSM8K vs AWQ |
|---|---|---|---|---|
| 4B | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |
| 8B | 5.48 [4.83, 6.10] | 4.21 [3.54, 4.86] | 4.62 [3.00, 6.25] | 4.32 [2.83, 5.81] |
| 14B | 3.22 [2.69, 3.75] | 2.46 [1.92, 3.00] | 3.26 [1.93, 4.59] | 3.34 [2.12, 4.55] |

We lose quality and win memory, at 45 to 52 % of AWQ's bits per parameter, and both gaps shrink with size. At 4B we
score 23.6 points above IQ2_XXS for 0.25 more bits per parameter. Perplexity ranks the files differently: our 4B reads
6.9 % below AWQ, our 8B and 14B 3.0 and 1.9 % above it.

One size up, at about the same memory, our file beats AWQ on MMLU (*measured*, not preregistered,
[sizeup](mesures/sizeup-2026-09-28.txt)): our 8B against AWQ's 4B, +1.44 [+0.71, +2.19] in 2.76 against 2.67 GB; our
14B against AWQ's 8B, +1.87 [+1.20, +2.57] in 5.04 against 6.10 GB.

## 4. Structural facts

**Format.** `Tetra` reads 2.150 kernel bits a weight in VRAM against 4.804 for `Planes14`, for the same 48 bits a
block (*computed*, `docs/data/echelle-formats.csv`). The product triplet (8k context, 5 GB margin, 32 GiB unit) sets
b_max = 3.00 kernel b/weight, and Tetra is the first layout under it, which admits models of 81 to 101 billion
parameters (*computed*).

**Kernel.** On the 4B's 252 projections one pass takes 3.734 ms, against 3.263 for AWQ and 11.022 for FP16 on the
L40S (*measured*, [banc-252](mesures/banc-252-2026-10-06.txt)). On an A100, Tetra beats our FP16 kernel by 1.11× but
not cuBLAS, and the files decode slower than vLLM's FP16 (*measured*, [a100](mesures/a100-2026-10-06.txt)). The speed
gain is an L40S result; no consumer card is measured.

**4E.** At the 4B's door v9 reads 221.5 tok/s against 129.0 for the eager engine, 1.717× round by round, 256 of 256
tokens equal; the 8B and 14B read 1.384× and 1.375× (*measured*, served-v9). The int4 words are held once: 1.980 GB on
the card under v9 and v6. 4E decodes batch 1 and greedy only, on the three sealed shapes, with no Metal kernel.

**Context.** The served doors decode 226.2 / 205.8 / 194.2 / 161.5 tok/s at 4B after prompts of 5 / 2,048 / 4,096 /
8,192 tokens, 144.5 / 135.2 / 129.3 / 110.0 at 8B and 87.3 / 83.7 / 80.9 / 71.5 at 14B (*measured*, served-v9). The
4B's first token at 8,192 takes 35.05 s.

**Attention tolerance.** The `grouped` and `kernel` forms are not bit for bit with `expanded`, so they are served
under a gate: paired WikiText-2 perplexity within ±0.12 %. Grouped against expanded reads +0.0095 %, -0.0048 % and
+0.0023 % by decode at 4B, 8B and 14B ([gqa-ppl](mesures/gqa-ppl-2026-11-01.txt),
[gqa-8b](mesures/gqa-8b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). Kernel from position 0 against
grouped reads +0.0018 % at 4B over 12 windows of 4,096 ([attn-t8-below](mesures/attn-t8-below-4b-2026-11-05.txt)),
+0.0043 % at 8B and -0.0108 % at 14B over 6 windows (served-v9-8b, -14b). All intervals sit inside the band. The
operator took this gate on 2026-11-01.

**Mac.** The MLX door is not bit for bit with the Rust engine. Against the Rust dense reconstruction in f16 on
WikiText-2 windows 13 to 24 it reads -0.0103 % [-0.0847, +0.0642] by decode, inside ±0.12 % (*measured*,
[mlx-pooled](mesures/mlx-pooled-2026-11-07.txt)). Its prototype decoded 93.9 tok/s on an M3 Max, where mlx-lm's
4-bit Qwen3-4B decodes 129.5 on another stack (*measured*, [metal-nuit](mesures/metal-nuit-2026-11-06.txt) M16). The
served door's own speed is not measured. The Rust Metal path runs the sealed 4B fused at 52.1 tok/s, and `served.rs`
refuses its grouped and kernel forms until the operator decides.

**Above 14B.** No model above 14B is served, and the 32B has never been encoded. The `rot_apply` shared-memory wall of
[`format-noyau.md`](format-noyau.md) §8 is lifted on the card up to width 25,600 (*measured*,
[lot1-4b](mesures/lot1-4b-2026-11-04.txt)).

**Row scales.** Trained per-row scales gain 3.15, 3.29 and 1.67 MMLU points for zero bits at 4B, 8B and 14B
(*measured*, [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt),
[-8b](mesures/dclm-8b-rowscales-2026-09-21.txt), [-14b](mesures/dclm-14b-rowscales-2026-09-22.txt)).

**Batch.** `Qwen3::batch_step` decodes B sequences a step (`llvq-llm/src/batch.rs`, `bin/batchrun`). Its
tensor-core kernel `tv_batch_mma.cu` reads 63.9 / 240.9 / 449.7 / 788.4 / 1,256.5 tok/s aggregate on the 4B at B = 1 /
4 / 8 / 16 / 32, against 203.0 / 778.7 / 1,524.6 / 2,883.2 / 5,128.7 for AWQ in vLLM (*measured*,
[batch-tetra](mesures/batch-tetra-2026-11-07.txt), [batch-awq](mesures/batch-awq-2026-11-07.txt)): 54.2 GB/s against
427.9 at B = 32 (*computed*). Every sequence decodes the batch-1 door tokens. Nothing batched is served.

**Noise.** The calibration draw moves MMLU by 2.92 pp and perplexity by 5.2 % (one standard deviation, 4B, three full
runs, *measured*, [bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). Every size is calibrated on one draw.
For an A/B on one file the bar is the paired interval, 0.43 pp of MMLU and 0.12 % of perplexity (*measured*,
[kvq8-4b](mesures/kvq8-4b-2026-08-15.txt)).

## 5. Open decisions

- **Throughput past v9.** At the bit the 4B tops out near 235 to 258 tok/s at the door (*estimated*,
  [`plan-300.md`](plan-300.md)). Next: speculative decoding with a 0.6B draft, and batching. Both are being worked on
  branches, not on main.
- **Scores through the served kernel.** MMLU and perplexity are scored on the dense reconstruction with f16 tables;
  the b/param count int4 tables. About $20 by the mixed route, $70 for the full census at three sizes (*estimated*,
  [`ROADMAP.md`](ROADMAP.md)). `ppl` does not read `LLVQ_CONFIG` yet.
- **A full GSM8K census under 4E.** Each size has a 50-problem pilot only. The 4B's grouped census cost $2.02.
- **The v9 config on the Hub's moving repository.** Publishing needs the operator.
- **A consumer GPU.** Speed is measured on an L40S and an A100 only.
- **The Rust Metal door.** Grouped and kernel forms read within tolerance on Metal; `served.rs` refuses them.
- **A model above 14B.** The 32B has to be encoded first.
- **`down_proj` instead of `o_proj` in int4 at 4B and 14B.** At 8B it scored 0.77 points better [0.26, 1.28].
- **A second calibration draw per size.**
- **An int4 KV cache.** It touches quality, so it is outside any "negligible" go.
- **The next venue for paper 2.** No archive holds version 2 yet. Default if silent: preprint only.
- **Upstream `transformers`.** `llvq-tetra` 0.1.0 is on PyPI and loads the 4B safetensors; the upstream issue and PR
  wait for the operator ([`plan-transformers.md`](plan-transformers.md)).
- **Spend.** $277.11 over 269 priced rows of 274 in `docs/data/jobs.csv` (*computed*). No cap is in force; one is owed
  before the next paid job.

## 6. Closed absent a new idea

| lead | why | source |
|---|---|---|
| E1v on the served path | 0.25× f16 | [e1v-cuda](mesures/e1v-cuda-2026-08-16.txt) |
| `Golay70` | 1.77× against a stamped threshold of 2.0× | [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt) |
| E3 | 3.0444 kernel b/weight against a criterion of 2.60 | [radixstudy](mesures/radixstudy-x4-2026-08-12.txt) |
| Design C | ×1.99 of perplexity at 28 blocks | [verdicts-nuit](archive/verdicts-nuit-2026-08-07.md) |
| `group_scales` | 44.66 to 53.60 of perplexity at 28 blocks | smoke of 2026-07-28, no journal |
| Spherical GPTQ feedback | +31.89 % of perplexity at the 0.6B | [sph-ppl](mesures/sph-ppl-0.6b-2026-09-22.txt) |
| A2, CUDA Graphs, served | +12.6 % of throughput for +47 % of VRAM at 4B | [ECARTS](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7 |
| Euclidean gain rule | dead end to end | [HISTORIQUE](HISTORIQUE.md), 2026-09-15 |
| 4E v5 | ×1.001 over v4; refused by name | [shaker-4e](mesures/shaker-4e-2026-11-05.txt) |
| 4E row-loop arms (K4) | best arm ×1.002 over v2, under the ×1.15 bar | [step1-micro](mesures/step1-micro-2026-11-07.txt) |
| Speculative decoding, the 0.6B drafting, 4E verifying | tokens identical to greedy at 4B, 8B and 14B, but decode ×0.595, ×0.629, ×0.625 of greedy at the best k (k = 1). Reopen when a 2-row verify pass costs under 1.4 greedy tokens and a draft token under 1 ms | [spec-gate](mesures/spec-gate-2026-11-07.txt) |
| A mixed Tetra plus int4 bench arm | the bench compares formats; served files are timed end to end | operator, 2026-10-06 |

Reopening any of them needs the condition in its row, not a new argument. A2 stays out of the core.
