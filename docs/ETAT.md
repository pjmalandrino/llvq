# Project state as of 2026-11-27

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
- **CUDA, the 4B's prefill**: since 2026-11-15 the 4B prefills by the chain,
  `configs/qwen3-4b-tetra-e4-4e-v9-chain.json`: the v9 config with `"prefill": "chain"`, the prompt fed through the
  tree verify pass, 4E v9 after it. The 8B and the 14B prefill eagerly: the pass does not carry them.
- **Mac, 4B**: the MLX door, `ops/llvq-mlx` under `configs/mlx/qwen3-4b-tetra-e4-mlx.json`, served since 2026-11-07.
- **Runtime**: `llvq` (crate `llvq-runtime`) pulls the three files and serves them with Ollama's and OpenAI's APIs. On
  CUDA it loads the chain config for the 4B and the v9 configs for the 8B and 14B; `LLVQ_PROFILE=eager` takes the
  4B back to its v9 config. Under `LLVQ_BATCH` it loads the batch configs, whose pool prefills eagerly. On Metal it
  runs the Rust engine, not MLX, with `configs/qwen3-{4b,8b,14b}-tetra-e4-metal.json` since 2026-11-26: the grouped
  attention and the Metal kernels `lutd+h+pq`, bit for bit with the expanded configs (matrix gate 253 of 253 at 4B
  and 8B, 281 of 281 at 14B; grouped against expanded max |dlogit| 0; the same tokens). Against the expanded
  config, round by round on an M3 Max, decode after a 5-token prompt is x1.142 / x1.097 / x1.077 at 4B / 8B / 14B,
  decode after a long prompt x3.201 at 4,096 tokens, x1.689 and x1.525 at 2,048, and the first token at the long
  prompt comes after 35.17 s instead of 67.87 at 4B, 28.44 instead of 55.45 at 8B and 52.59 instead of 91.92 at 14B
  (*measured*, [qwen-metal-servi](mesures/qwen-metal-servi-2026-11-26.txt)). The 4B's peak footprint at the long
  prompt falls from 4.98 to 3.55 GB and the 8B's and 14B's do not move. The ratios are the reading: that night's
  absolute rates came from a machine other jobs were loading. The CPU keeps the `*-e4.json` configs.
- **Without a GPU**: the runtime's CPU backend since 2026-11-07, `fused_cpu` on the expanded e4 configs, at f32. It
  reads the Tetra codes at every matvec: 8.6 tok/s in 1.86 GB on the 4B at 12 threads of an M3 Max, against 0.71
  tok/s in 10.6 GB for the dense reconstruction it replaced; perplexity -0.0021 % of the dense f32 over 12 windows
  (*measured*, [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt)). On 8 vCPU of an AMD EPYC 7R13 with its AVX2 kernel the
  4B decodes 5.80 and 6.53 tok/s through `cpurun` and 5.4 through `llvq run`, in 1.81 GB resident (*measured*, one
  instance, one run an arm, job 7 of [llvq-cpu-x86](mesures/llvq-cpu-x86-2026-11-25.txt); 3.64 and 3.90 on the build
  of job 5, [cpu-tetra-speed](mesures/cpu-tetra-speed-2026-11-25.txt)).
- **Inkling 2 Mini, prefix cache**: on by default on Metal and on the CPU, off on CUDA, where it has not run. On
  this Mac's CPU (6 threads) a restarted server gives the first token of a 2,316-token agent prompt after 20.33 s
  [20.31, 20.39] against 140.99 s cold, with the same answer as a server with the cache off; the file is 0.81 GB.
  A request whose whole prompt is held is answered from the logits kept at the end of its first read: on Metal the
  same 8,791-token prompt sent again gets its first token after 0.159 s [0.157, 0.165] against 2.687 s before
  (*measured*, [inkling-cpu-prefix-held](mesures/inkling-cpu-prefix-held-2026-11-26.txt)). Nothing is measured on
  x86 for the cache.

| size | sealed file | b/param | MMLU | GSM8K | ppl | tok/s served | GB weights on card | GB card used after load |
|---|---|---|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 82.49 | 12.58 | 221.5 | 1.38 | 1.98 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 9.71 | 142.5 | 2.76 | 3.36 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 8.44 | 86.3 | 5.04 | 5.64 |

- b/param: *computed* by `rtbits`, whole model, int4 tables included.
- MMLU: *measured*, all 14,042 test questions, 5-shot, micro average
  ([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt), [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt),
  [sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)).
- GSM8K: *measured* through the kernel, all 1,319 test problems, zero-shot, greedy
  ([gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)).
- Perplexity: *measured*, raw WikiText-2 test split, 12 windows of 4,096, f16
  ([ppl-scelles](mesures/ppl-scelles-2026-10-04.txt)).
- Speed: *measured* on one L40S at each config's door, batch 1, 256 greedy tokens after a 5-token prompt,
  median of five rounds ([served-v9](mesures/served-v9-2026-11-07.txt),
  [-8b](mesures/served-v9-8b-2026-11-07.txt), [-14b](mesures/served-v9-14b-2026-11-07.txt)).
- GB weights on card: *computed* by the door from the tensors it loads, projections plus the carried tensors
  (`total expected on the card` in the same jobs' raw `door.txt`).
- GB card used after load: *measured* by `cuMemGetInfo` after the load and 4E's install, in the same jobs. It adds
  the CUDA context, 4E's buffers and the allocator's slack to the weights. The peak after a long prompt is higher:
  the KV cache and the prefill's activations come on top.

**The scores and the speeds come from different forms.** MMLU, GSM8K and perplexity ran under the `expanded`
attention form of paper 2, and MMLU and perplexity on the dense reconstruction. The served form adds the `kernel`
attention and 4E. It returns the eager path's tokens, and its attention is served under a perplexity tolerance (§4).
Its only task reading is a 50-problem GSM8K pilot per size: 40, 46 and 46 right at 4B, 8B and 14B, the same
scores as the earlier configs on those problems. The Hub cards
paper 2 cites report 113.8, 95.0 and 57.2 tok/s under the expanded form
([paper-table](mesures/paper-table-2026-09-25.txt)); they do not move. Paper 2's source reports the served speeds
since 2026-11-17 and keeps those three as its per-matrix path.

The Hub's moving repository `Qwen3-4B-LLVQ-Tetra-served` still carries the grouped config of 2026-11-02 at 133.3
tok/s ([served-grouped](mesures/served-grouped-2026-11-02.txt)). The repositories paper 2 cites, and the safetensors
one, are frozen: their files, their digests and their figures do not move. The three sealed cards took one metadata
commit on 2026-11-20, a `pipeline_tag` and a link to the paper's page, and `docs/hub/` holds them byte for byte. The safetensors
card and the served card took the same link on 2026-11-23, nothing else.

## 3. Against the baselines

| size | arm | b/param | MMLU | GSM8K | ppl | tok/s | GB |
|---|---|---|---|---|---|---|---|
| 4B | FP16, vLLM | 16.00 | 70.14 | 92.12 | 12.24 | 83.1 | 8.04 |
| 4B | AWQ w4g128, vLLM | 5.30 | 68.14 | 89.01 | 13.52 | 200.5 | 2.67 |
| 4B | IQ2_XXS, llama.cpp | 2.48 | 39.78 | | | 312.9 | 1.25 |
| 4B | IQ2_S, llama.cpp | 2.82 | 54.29 | 68.23 | 16.78 | 287.5 | 1.42 |
| 8B | FP16, vLLM | 16.00 | 75.05 | 93.25 | 8.99 | 46.3 | 16.38 |
| 8B | AWQ w4g128, vLLM | 5.96 | 73.79 | 92.95 | 9.42 | 123.2 | 6.10 |
| 8B | IQ2_S, llama.cpp | 2.80 | 65.45 | 85.60 | 10.75 | 208.4 | 2.86 |
| 14B | FP16, vLLM | 16.00 | 78.88 | 95.30 | 7.98 | 25.8 | 29.54 |
| 14B | AWQ w4g128, vLLM | 5.40 | 78.12 | 95.38 | 8.29 | 77.8 | 9.98 |
| 14B | IQ2_S, llama.cpp | 2.69 | 72.77 | 91.89 | 9.47 | 123.5 | 4.96 |

All *measured* on the L40S; each cell's journal is in `docs/data/paper2-results.csv` and `docs/data/paper2-ppl.csv`,
the IQ2_S rows in [duel-iq2](mesures/duel-iq2-2026-11-08.txt) (llama.cpp cd959bae9, imatrix on dclm-edu, the nearest
rung to our bits). Speeds from different engines are never divided (rule 5). The IQ2_S rows keep the duel's accounting:
b/param from file bytes, tok/s of llama-bench decode after 5 tokens, GB of file. In that accounting our files read
2.82 / 2.75 / 2.76 b/param, 227.4 / 144.5 / 90.0 tok/s of decode and 1.42 / 2.82 / 5.09 GB (*measured* and *computed*,
same journal): at 4B Tetra and IQ2_S tie on b/param (2.8206 against 2.8188), where the `rtbits` 2.73 of §2 would put
us 0.09 below. Compare §2's door figures with the AWQ and FP16 rows, and these with the IQ2 rows.

Paired gaps, our file below the arm (MMLU *computed*, `docs/data/paper2-gaps.csv`; GSM8K *measured* by `gsm8kpair`):

| size | MMLU vs FP16 | MMLU vs AWQ | GSM8K vs FP16 | GSM8K vs AWQ |
|---|---|---|---|---|
| 4B | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |
| 8B | 5.48 [4.83, 6.10] | 4.21 [3.54, 4.86] | 4.62 [3.00, 6.25] | 4.32 [2.83, 5.81] |
| 14B | 3.22 [2.69, 3.75] | 2.46 [1.92, 3.00] | 3.26 [1.93, 4.59] | 3.34 [2.12, 4.55] |

We lose quality and win memory, at 45 to 52 % of AWQ's bits per parameter, and both gaps shrink with size. At 4B we
score 23.6 points above IQ2_XXS for 0.25 more bits per parameter. Against llama.cpp's IQ2_S at our bits, the same day on
the same card, we score +9.08, +4.13 and +2.89 MMLU points higher and read 12.58, 9.71 and 8.44 perplexity against
16.78, 10.75 and 9.47; GSM8K
+14.25 and +3.03 at 4B and 8B, not separated at 14B. IQ2_S decodes 60.1, 63.9 and 33.5 tok/s faster than our 4E v9
after 5 tokens (*measured*, [duel-iq2](mesures/duel-iq2-2026-11-08.txt)). Perplexity ranks the files differently: our
4B reads 6.9 % below AWQ, our 8B and 14B 3.0 and 1.9 % above it.

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
4B's first token at 8,192 takes 6.74 s under the served chain prefill, against 37.22 s for the eager prefill of the v9
config in the same job (*measured*, [chain-prefill-served](mesures/chain-prefill-served-2026-11-15.txt)).

**Attention tolerance.** The `grouped` and `kernel` forms are not bit for bit with `expanded`, so they are served
under a gate: paired WikiText-2 perplexity within ±0.12 %. Grouped against expanded reads +0.0095 %, -0.0048 % and
+0.0023 % by decode at 4B, 8B and 14B ([gqa-ppl](mesures/gqa-ppl-2026-11-01.txt),
[gqa-8b](mesures/gqa-8b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). Kernel from position 0 against
grouped reads +0.0018 % at 4B over 12 windows of 4,096 ([attn-t8-below](mesures/attn-t8-below-4b-2026-11-05.txt)),
+0.0043 % at 8B and -0.0108 % at 14B over 6 windows (served-v9-8b, -14b). All intervals sit inside the band. The
operator took this gate on 2026-11-01. Since 2026-11-15 the chain prefill is served under the same band: its 16-row
form reads +0.0007 % of paired perplexity against the eager prefill (*measured*,
[large-prefill](mesures/large-prefill-2026-11-11.txt)). Its tree modules hold 744 B of local memory; the operator
accepted that for them alone, and 4E's token symbol keeps its rule of none.

**Mac.** The MLX door is not bit for bit with the Rust engine. Against the Rust dense reconstruction in f16 on
WikiText-2 windows 13 to 24 it reads -0.0103 % [-0.0847, +0.0642] by decode, inside ±0.12 % (*measured*,
[mlx-pooled](mesures/mlx-pooled-2026-11-07.txt)). Its prototype decoded 93.9 tok/s on an M3 Max, where mlx-lm's
4-bit Qwen3-4B decodes 129.5 on another stack (*measured*, [metal-nuit](mesures/metal-nuit-2026-11-06.txt) M16). The
served door's own speed is not measured. The Rust Metal path runs the sealed 4B fused at 52.1 tok/s, and `served.rs`
accepts the grouped form on Metal only from the `-metal.json` configs of §2 and refuses the kernel form.

**Above 14B.** No model above 14B is served, and the 32B has never been encoded. The `rot_apply` shared-memory wall of
[`format-noyau.md`](format-noyau.md) §8 is lifted on the card up to width 25,600 (*measured*,
[lot1-4b](mesures/lot1-4b-2026-11-04.txt)).

**Row scales.** Trained per-row scales gain 3.15, 3.29 and 1.67 MMLU points for zero bits at 4B, 8B and 14B
(*measured*, [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt),
[-8b](mesures/dclm-8b-rowscales-2026-09-21.txt), [-14b](mesures/dclm-14b-rowscales-2026-09-22.txt)).

**Batch.** `llvq serve` batches continuously under `LLVQ_BATCH` through `configs/qwen3-*-tetra-e4-4e-v9-batch.json`,
the served v9 configs with a batch field: a lone request decodes on 4E v9, the others step together through the GEMM
`tv_batch_gemm.cu` (`llvq-llm/src/batch.rs`, `bin/batchrun`). Under concurrent requests the 4B reads 221.9 / 757.8 /
1,976.5 / 3,067.5 tok/s aggregate at B = 1 / 8 / 32 / 128, and 225.4 tok/s a request alone (*measured*,
[batch-l40s-v4](mesures/batch-l40s-v4-2026-11-08.txt)). The kernel alone, the prompt fed through the batch: 4B 105.5
/ 800.0 / 2,468.0 / 4,493.6, 8B 74.1 / 568.3 / 1,826.6 / 3,552.3, 14B 43.6 / 335.0 / 1,102.2 / 2,183.8 at B = 1 / 8 / 32
/ 128 (*measured*, [batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt)); AWQ in vLLM 203.0 / 1,524.8 / 5,133.8 /
10,146.2 on the 4B ([batch-l40s](mesures/batch-l40s-2026-11-08.txt)). That is 106.4 GB/s against 428.3 on the 4B at
B = 32 (*computed*). The projections are three quarters of the step and more; at one row the GEMM's Tetra decode moves
about 200 GB/s. The gate passed at every pass: perplexity within +-0.01 % of the multi-row path, GSM8K pilot 40/50 as
served, every door row equal to batch 1. Nothing batched is served by default.
The pool's KV caches are allocated at load, beside the weights: 64 slots of 2,048 positions weigh 19.3 / 19.3 / 21.5 GB
at 4B / 8B / 14B (*computed*). The load refuses a pool the card's free memory does not hold, and `/api/ps` counts it in
`size_vram`. A memory figure of the batch counts the pool (rule 6). A request left alone in the pool goes back to 4E, and a
lone request past 4E's 8,704 positions finishes in the pool (audit 2026-11-08, sections 1.2, 1.7 and 1.8; CPU tests in
`llvq-runtime/tests/batch_server.rs`, not run on the card).

**Noise.** The calibration draw moves MMLU by 2.92 pp and perplexity by 5.2 % (one standard deviation, 4B, three full
runs, *measured*, [bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). Every size is calibrated on one draw.
For an A/B on one file the bar is the paired interval, 0.43 pp of MMLU and 0.12 % of perplexity (*measured*,
[kvq8-4b](mesures/kvq8-4b-2026-08-15.txt)).

**Where the 4B loses quality.** The MLP carries most of the sealed 4B's 6.77-point MMLU gap to FP16 (*measured*,
[ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt)). Restored to f16, gate/up recovers +3.38
[+2.75, +4.02], down +2.01 [+1.46, +2.54] and the whole MLP +4.86 [+4.16, +5.57]. Attention recovers +0.73
[+0.16, +1.29]: qkv +0.55 [+0.01, +1.11], o -0.13 [-0.46, +0.19]. All 252 projections restored sit +0.04
[-0.23, +0.31] under FP16. So the int4 g64 embedding and tied head cost nothing resolved, and neither do the int4
records of `o`. The groups are super-additive: all exceeds the MLP plus attention by +1.14 [+0.43, +1.85]
(*computed*, paired per question). The scope is one calibration draw, MMLU only, on the dense f16 reconstruction.

## 5. Open decisions

- **Throughput past v9.** At the bit the 4B tops out near 235 to 258 tok/s at the door (*estimated*,
  [`plan-300.md`](plan-300.md)). Speculation is no longer closed on the verify cost: a tree
  pass verifies 16 rows for 1.73 tokens of 4E, an extra row costing 0.0114 to 0.0196 of a token (*measured*,
  [verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)), against 0.74 of a token a row on 2026-11-08
  ([spec-final](mesures/spec-final-2026-11-08.txt)). With the public DFlash drafter the loop decodes GSM8K-200 at
  502 tok/s (x2.30), the house set at 343 (x1.56) and chat at 242 (x1.09), GSM8K 167 correct against 166 (*measured*,
  [spec-loop-sp8](mesures/spec-loop-sp8-2026-11-11.txt)). It pays in memory: 5.41 GB against 2.01 and 4.87 b/param
  with the f16 drafter (5.41 *measured*, [spec-mem-m3](mesures/spec-mem-m3-2026-11-11.txt); 4.87 *computed*, [spec-memoire](mesures/spec-memoire-2026-11-11.txt)). Operator, 2026-11-11: the
  default stays at 2.73 b/param, and the fast profile is set aside, built when wanted. Asked for on 2026-11-15, it
  was stopped before any code: its tokens are the tree pass's greedy, 177 of 200 GSM8K prompts identical to 4E's
  (*measured*, spec-loop-sp8), where the request asked for 4E's, and the drafter's licence is recorded nowhere.
  Three bit-identical cuts of 4E did not pass and are in §6.
- **Card memory after an 8,192-token prompt.** The served prefill keeps one score tensor instead of three, bit for
  bit: 5.95 / 7.39 / 10.54 GB at 4B / 8B / 14B, from 11.48 / 12.66 / 17.09 in the duel, against 3.52 / 4.71 / 6.90 for
  llama.cpp IQ2_S (*measured*, L40S, [pic-vram-8k](mesures/pic-vram-8k-2026-11-08.txt)). The NOT SERVED
  `prefill_attention: kernel` measured 4.77 / 7.15 / 9.30, about 1 to 2 GB over its computed peak, and 1.14 to 1.21
  times slower to the first token: not proposed. Open: the excess it holds layer by layer is not traced.
- **The chain prefill, served on the 4B since 2026-11-15.** At the door the 4B's first token comes after 0.45 / 2.47 /
  6.74 s at 1,024 / 4,096 / 8,192 tokens against 3.96 / 17.79 / 37.22 s for the v9 config, with the same 256 tokens
  at each length. The phase peak at 8,192 is 3.633 GB against 6.047. GSM8K-200 reads 165 right in both, no discordant
  pair (*measured*, L40S, [chain-prefill-served](mesures/chain-prefill-served-2026-11-15.txt)). Its prereg read NOT SETTLED on one bound: the decode after the chain
  is 2.9 and 3.2 % faster than v9's at 4,096 and 8,192, outside a two-sided band. The operator served it on that
  reading, with the local memory of its tree modules and the ±0.12 % band (§4). `llvq`'s own door then read it on a card
  ([chain-runtime-door](mesures/chain-runtime-door-2026-11-15.txt), *measured*): with no variable set it loads the
  chain config, the first content of an 8,191-token prompt comes after 6.448 s against 34.537 s under
  `LLVQ_PROFILE=eager`, and of a 92-token prompt after 0.065 s against 0.410 s; the way back loads v9 and unknown
  profiles are refused by name. `llvq` loads the chain in 93.6 s from the cache against 54.9 s for the eager
  prefill; the 38 s are not explained. Still owed: MMLU and the near-tie texts under the chain; the 8B and the 14B,
  which the pass does not carry; a cold start of `llvq` on a card. A first start compiles the two tree modules:
  509.2 s to load cold in `fusedrun`, 55.2 s from the cache.
- **The eager prefill's text depends on the conversation cache.** In the same job the eager server gave one text for
  an 8,191-token prompt with none of it cached and another with the chat template's 3 tokens cached; the two part at
  a near tie, at character 130. The chain gave the same text in both states, the eager server's second. The eager
  prefill runs in chunks of 4,096 positions under the ±0.12 % band (§4), and 3 cached tokens move every chunk
  boundary. It serves the 8B and the 14B, and the 4B under `LLVQ_PROFILE=eager`. How often it happens is not read
  (*measured* once, [chain-runtime-door](mesures/chain-runtime-door-2026-11-15.txt)). That check's prereg read
  FAILS on this control; the operator merged the chain on 2026-11-15 with the reading in hand.
- **Scores through the served kernel.** MMLU and perplexity are scored on the dense reconstruction with f16 tables;
  the b/param count int4 tables. About $20 by the mixed route, $70 for the full census at three sizes (*estimated*,
  [`ROADMAP.md`](ROADMAP.md)). `ppl` does not read `LLVQ_CONFIG` yet.
- **A full GSM8K census under 4E.** Each size has a 50-problem pilot only. The 4B's grouped census cost $2.02.
- **The v9 config on the Hub's moving repository.** Publishing needs the operator.
- **A consumer GPU.** Speed is measured on an L40S and an A100 only.
- **The Rust Metal door, kernel form.** The grouped form is served on Metal since 2026-11-26; `served.rs` still refuses
  the kernel form there, which read within tolerance on the 4B only.
- **A model above 14B.** The 32B has to be encoded first.
- **`down_proj` instead of `o_proj` in int4 at 4B and 14B.** At 8B it scored 0.77 points better [0.26, 1.28]. Set
  aside by the operator's ruling of 2026-11-12 that per-model, per-type changes are not the project's direction
  ([`MODES.md`](MODES.md)); it reopens only on his word.
- **A second calibration draw per size.**
- **The next quality or speed work.** It opens as a phase under [`MODES.md`](MODES.md), on the operator's mandate.
- **An int8 KV cache, as a memory lead.** It holds the ±0.12 % tolerance at 8,192 on the 4B, +0.0126 %
  [−0.0485, +0.0737] against f16, and int4 keys fail it (*measured*,
  [kv-ladder](mesures/recherche-debit-kv-ladder-2026-11-14.txt)). It would take the cache from 1.21 to 0.64 GB at
  8,192 positions (*computed*). It buys no speed (§6). Kept by the operator on 2026-11-15. The form that frees the
  f16 cache at the hand-over is written, `configs/qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json`, refused at every
  served door and loaded by a measurement door only; it is not run on a card. Owed before serving: its card run
  (`proofs/preregistration-kv8-memory-2026-11-15-DRAFT.md`), the near-tie texts, GSM8K-200, the 8B and the 14B, and
  the operator's ruling on the 8 B of local memory in its token symbol, which the ruling of 2026-11-15 does not
  cover.
- **The next venue for paper 2.** No archive holds version 2 yet. Default if silent: preprint only.
- **Upstream `transformers`.** `llvq-tetra` 0.1.0 is on PyPI and loads the 4B safetensors; the upstream issue and PR
  wait for the operator ([`plan-transformers.md`](plan-transformers.md)).
- **Spend.** $307.13 on 2026-11-09 (*computed*): $306.88 over 297 priced rows of 302 in `docs/data/jobs.csv`, and
  `rows-grid` $0.25, which no CSV on main carries yet ([audit](archive/audit-bugs-2026-11-08.md) §3). The night of
  2026-11-08 added `pic-vram-8k` $2.04 and `spec-final` $1.04. No cap is in force; one is owed before the next paid
  job.

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
| Speculative decoding, the 0.6B drafting, 4E verifying | tokens identical to greedy at 4B, 8B and 14B, but decode ×0.595, ×0.629, ×0.625 of greedy at the best k (k = 1). Reopen when a 2-row verify pass costs under 1.4 greedy tokens and a draft token under 1 ms. With B1's integer dot, the Tetra part of an extra row costs 0.187 of a row alone, between the 0.15 and 0.26 gates (2026-11-08), not yet a verify pass | [spec-gate](mesures/spec-gate-2026-11-07.txt), [spec-micro](mesures/spec-micro-2026-11-08.txt) |
| Prompt lookup, 4E verifying | tokens identical to greedy at three sizes; best policy (n-grams 3, probe 1) decodes ×0.92 to ×0.96 of greedy on chat, code and summary at 4B, ×1.108 on retrieval, under its rule B. Reopen with the same condition as the 0.6B row | [spec-final](mesures/spec-final-2026-11-08.txt) |
| EAGLE-3 head on the card | acceptance measured on Tetra at $0 (α 0.43 to 0.61 on free text), projection ×0.72 on chat at today's verify, ×1.04 at an f32 floor. Reopen with a verify row under 0.2 of a 4E token: met on 2026-11-10 by the tree pass ([verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)), the head itself not built | [eagle-alpha](mesures/eagle-alpha-2026-11-08.txt) |
| 4E without its L2 prefetch in attention, or with it moved to gate/up (E1) | bit for bit, x1.003 and x0.996 of v9, x0.978 moved | [large-4e-e1](mesures/large-4e-e1-2026-11-11.txt) |
| K/V cache read once for a group of 4 rows in the chain prefill (D1), as a speed rule | bit for bit, x1.302 at 8,191 tokens under its x1.5 rule; the arm is kept inside the 32-row pass | [large-prefill-d1](mesures/large-prefill-d1-2026-11-11.txt) |
| 4E v9ft, the guard-free full attention tile in decode | bit for bit, x1.009 / x1.027 / x0.997 / x1.004 of v9 at 5 / 1,024 / 4,096 / 8,192 tokens, under its x1.05 rule; v9 stays served | [large-attn-ft-4e](mesures/large-attn-ft-4e-2026-11-11.txt) |
| A mixed Tetra plus int4 bench arm | the bench compares formats; served files are timed end to end | operator, 2026-10-06 |
| Batched GEMM at two CTAs a SM (K chunk halved, registers capped at 128) | 5.7 to 17.4 % below one CTA a SM at B = 32 to 128 on the 4B, both int4 orders | [batch-l40s-v4](mesures/batch-l40s-v4-2026-11-08.txt) |
| An int8 KV cache read by 4E, as a speed lever (arm v9k8) | x1.0109 [1.0108-1.0113] of v9 at 8,192 and x0.9668 [0.9664-0.9670] at 5 tokens, with 8 B of local memory in the token symbol; the attention phase loses 1.8 % for 46.9 % fewer bytes. Kept as a memory lead (§5). Reopen with a measured cause of the attention time at 8,192 that a smaller cache removes | [recherche-debit-kv8-local](mesures/recherche-debit-kv8-local-2026-11-15.txt) |
| A reshaped Tetra codebook at 48 bits: width profiles under a prefix code, tetrad pairings, one word for two blocks, sparse corrections | on Gaussian blocks the best reads −0.21 dB against Tetra, where the 4-profile selector reads −0.18 and the ball −0.375 (*computed*, a model calibrated on those two *measured* points); the excess is spread over blocks. Closed by the operator on 2026-11-26. Reopen with a per-section saturation rate on the encoded blocks of a sealed file above the Gaussian one | [geometrie-tetra-proxy](mesures/geometrie-tetra-proxy-2026-11-26.txt) |

Reopening any of them needs the condition in its row, not a new argument. A2 stays out of the core.
