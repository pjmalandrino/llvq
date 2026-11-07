# Project state as of 2026-11-07

## 1. The project

LLVQ quantizes the weights of an LLM to about 2 bits on the Leech lattice, in Rust. The goal is to fit larger models
on local hardware. The repository carries the quantizer, the file format, and a fused CUDA kernel that decodes and
multiplies without going back through f16.

Three Qwen3 models are sealed and served at about 2.7 bits per parameter: 4B, 8B, 14B. The three files are public on
the Hugging Face Hub as `Pier-Jean/Qwen3-{4B,8B,14B}-LLVQ-Tetra-sealed`. Paper 1 is on Zenodo (DOI
[10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)) and went to arXiv as sources on 2026-09-02. Paper 2,
*Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*, is in [`paper2/`](../paper2/README.md), and
its version 2 is attached to release `v0.0.2` of 2026-10-06.

## 2. Served configuration

One layout, one setting, at all three sizes: `tetra48`, `LLVQ_EMBED=q4`, `LLVQ_ROT_SHARE=1`, `LLVQ_FUSE=0`,
`LLVQ_KV=f16`. The files that carry it are in [`configs/`](../configs/README.md), which is the authority on them. The
sizes differ in three optional fields: the attention form, the decode engine and 4E's row loop.

- **4B**: `qwen3-4b-tetra-e4-4e-v4.json` since 2026-11-05, the operator's decision. Attention `kernel`: every decode
  row goes to the split-KV attention kernel from position 0. Decode `persistent`: 4E runs the whole token in one
  cooperative CUDA launch. Rows `v4`: 4E's second row loop. `qwen3-4b-tetra-e4-4e-v6.json`, the same with rows
  `v6`, is written and loads everywhere v4 loads; it is not served before its door reads (§5).
- **8B and 14B**: `qwen3-8b-tetra-e4-grouped.json` and `qwen3-14b-tetra-e4-grouped.json` since 2026-11-04, the
  operator's decision after a light MMLU check passed at each size.

An absent `attention` field means `expanded`, the form paper 2 ran and the form of every MMLU and perplexity below.
Neither served form is bit for bit with it; both are served under the ±0.12 % tolerance of §4.

The Hub's moving repository [Qwen3-4B-LLVQ-Tetra-served](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-served)
still carries the grouped config of 2026-11-02 (commit `159afed4`, file digest `886391a8` read back from the Hub), at
133.3 tok/s (*measured*, [served-grouped](mesures/served-grouped-2026-11-02.txt)). Moving it to the v4 config is an
open decision of §5. The three repositories paper 2 cites, and the safetensors one, are frozen.

| size | sealed file | b/param | MMLU | GSM8K expanded | GSM8K grouped | ppl | tok/s served | GB on card |
|---|---|---|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 82.49 | 81.96 | 12.58 | 178.0 | 1.38 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 88.55 | 9.71 | 105.8 | 2.76 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 91.96 | 8.44 | 62.5 | 5.04 |

b/param: *computed* by `rtbits`, whole model, int4 tables included. MMLU: *measured* on all 14,042 test questions,
5-shot, micro average, fingerprint `a74a6d62` ([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt),
[sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt), [sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)). GSM8K
expanded: *measured* through the kernel on all 1,319 test problems, zero-shot, greedy, fingerprint `bfa9135c`
([gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)). GSM8K grouped:
*measured* the same way under the grouped config. Paired against the expanded census it reads -0.53 pp [-1.07, +0.00]
at 4B, -0.08 [-0.61, +0.46] at 8B and -0.08 [-0.22, +0.07] at 14B, McNemar p = 0.092, 1.000 and 1.000
([gsm8k-grouped](mesures/gsm8k-grouped-2026-11-04.txt), [-8b](mesures/gsm8k-grouped-8b-2026-11-04.txt),
[-14b](mesures/gsm8k-grouped-14b-2026-11-04.txt)). Perplexity: *measured* on the raw WikiText-2 test split, 12
windows of 4,096, f16 ([ppl-scelles](mesures/ppl-scelles-2026-10-04.txt)). Speed and GB: *measured* on one L40S
through the served door of each size's config, batch 1, 256 greedy tokens after a 5-token prompt, prompt and decode,
median of five rounds ([served-v4](mesures/served-v4-2026-11-05.txt),
[served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt),
[served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt)). Paper 2 and its cited cards report 113.8, 95.0
and 57.2 tok/s under the expanded form ([paper-table](mesures/paper-table-2026-09-25.txt)), and they do not move.

**The score columns and the speed column come from different forms.** MMLU and perplexity were measured under the
expanded form only. Under the grouped form the 285-question MMLU plan picks the same answer on every question at 8B
and 14B ([served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt),
[served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt)), and so does the 57-question door at 4B
([gsm8k-grouped](mesures/gsm8k-grouped-2026-11-04.txt)). The 4B's served form adds the kernel and 4E to the grouped
form. Its only task reading is a 50-problem GSM8K pilot: 40 of 50, with the same 50 completions as the v3 config
([served-v4](mesures/served-v4-2026-11-05.txt)). The v3 config read 40 of 50 against 40 of 50 for the grouped census,
McNemar p = 1 ([served-4e](mesures/served-4e-2026-11-05.txt)). No full task score has run under the kernel form or 4E.

**MMLU and perplexity are scored on the dense reconstruction, not through the kernel.** Per-row checks against f64 and
a 256-token comparison tie the two: identical tokens at 4B and 8B, first difference at token 78 at 14B. One 4B census
on 2,280 questions (object of 2026-09-09) read 55.66 through the kernel and 55.52 dense, McNemar p = 0.25 (*measured*,
[f1e-census](mesures/f1e-census-2026-09-11.txt)). The MMLU above used f16 tables while the b/param count int4 tables,
which is the first open decision of section 5.

## 3. Against the baselines

| size | arm | b/param | MMLU | GSM8K | ppl | tok/s | GB |
|---|---|---|---|---|---|---|---|
| 4B | FP16, vLLM | 16.00 | 70.14 | 92.12 | 12.24 | 83.1 | 8.04 |
| 4B | AWQ w4g128, vLLM | 5.30 | 68.14 | 89.01 | 13.52 | 200.5 | 2.67 |
| 4B | IQ2_XXS, llama.cpp | 2.48 | 39.78 | not scored | not scored | 312.9 | 1.25 |
| 8B | FP16, vLLM | 16.00 | 75.05 | 93.25 | 8.99 | 46.3 | 16.38 |
| 8B | AWQ w4g128, vLLM | 5.96 | 73.79 | 92.95 | 9.42 | 123.2 | 6.10 |
| 14B | FP16, vLLM | 16.00 | 78.88 | 95.30 | 7.98 | 25.8 | 29.54 |
| 14B | AWQ w4g128, vLLM | 5.40 | 78.12 | 95.38 | 8.29 | 77.8 | 9.98 |

All *measured* on the L40S. The journal of each cell is in the source column of `docs/data/paper2-results.csv` and
`docs/data/paper2-ppl.csv`. Speeds come from different engines and are never divided across them (rule 5).

Paired gaps, our file against the arm (MMLU *computed*, `docs/data/paper2-gaps.csv`; GSM8K *measured* by `gsm8kpair`):

| size | MMLU below FP16 | MMLU below AWQ | GSM8K below FP16 | GSM8K below AWQ |
|---|---|---|---|---|
| 4B | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |
| 8B | 5.48 [4.83, 6.10] | 4.21 [3.54, 4.86] | 4.62 [3.00, 6.25] | 4.32 [2.83, 5.81] |
| 14B | 3.22 [2.69, 3.75] | 2.46 [1.92, 3.00] | 3.26 [1.93, 4.59] | 3.34 [2.12, 4.55] |

We lose quality and win memory, at 45 to 52% of AWQ's bits per parameter, and both gaps shrink with size. At 4B we
score 23.6 points above IQ2_XXS for 0.25 more bits per parameter.

Perplexity ranks the files differently. Our 4B reads 6.9% below AWQ, our 8B and 14B 3.0 and 1.9% above it, and all six
gaps to FP16 and AWQ are separated from zero (*computed*, paired t over 12 windows, `docs/data/paper2-ppl.csv`). The
last build step, int4 projections and 4-bit tables, gains MMLU at every size but raises perplexity by 2.1% [1.5, 2.6]
at 4B and 1.4% [1.1, 1.7] at 8B, not separated at 14B (*measured*,
[ppl-bases-carte](mesures/ppl-bases-carte-2026-10-05.txt)). Which half of the step costs it is not measured.

On GSM8K the loss exceeds the MMLU loss at 4B only. At 8B and 14B the GSM8K interval contains the MMLU gap. The engine
gate held at 4B: FP16 reads 91.51 in our dense path against 92.12 in vLLM, −0.61 [−1.27, +0.06].

One size up, at about the same memory, our file beats AWQ on MMLU and is not separated from it on GSM8K (*measured*,
not preregistered, [sizeup](mesures/sizeup-2026-09-28.txt)): our 8B against AWQ's 4B, +1.44 [+0.71, +2.19] MMLU in 2.76
against 2.67 GB; our 14B against AWQ's 8B, +1.87 [+1.20, +2.57] in 5.04 against 6.10 GB. Weights plus an 8k f16 KV
cache put us one size above AWQ from 3.97 to 11.32 GB, or from 3.97 to 5.52 and 6.39 to 9.08 GB if AWQ stored its
tables in 4 bits too (*computed*).

## 4. Structural facts

`Tetra` reads **2.150 kernel b/weight** in VRAM against 4.804 for `Planes14`, for the same 48 bits a block, over the
252 projections of the 4B (*computed*, `docs/data/echelle-formats.csv`). On those 252 matrices one pass takes 3.734 ms,
against 3.263 for AWQ and 11.022 for FP16 on the L40S (*measured*, [banc-252](mesures/banc-252-2026-10-06.txt)). Half
of AWQ's bytes is not half its time. The product triplet in force since 2026-08-16 (8k context, 5 GB margin, 32 GiB
unit, offload as reference only) sets b_max = 3.00 kernel b/weight. `Tetra` is the first layout under it, 28% clear,
which moves the admissible class from 43.3 to 81 to 101 billion parameters (*computed*).

**The speed gain is an L40S result.** On one A100, over the same 252 matrices, Tetra beats our FP16 kernel (1.11×) but
not cuBLAS (1.14×). There the files under their expanded configs decode at 0.92, 0.91 and 0.95 times our dense
path with f16 tables, and at 74.2, 57.9 and 39.0 tok/s where vLLM decodes FP16 at 145.6, 91.8 and 52.7 (*measured*,
[a100](mesures/a100-2026-10-06.txt)).

**No model above 14B is served, and the 32B has never been encoded.** The `rot_apply` wall of
[format-noyau](format-noyau.md) §8 is lifted on the card: the device-staged rotation and the f16-staged int4 kernel
return the host's bytes at every width up to 25,600, Qwen3-32B's `down_proj` (*measured*,
[lot1-4b](mesures/lot1-4b-2026-11-04.txt)). The rotation takes 45.43 µs at 25,600. Nothing past 14B has run end to end.

Trained row scales gain **+3.15, +3.29 and +1.67 MMLU points for zero bits** at 4B, 8B and 14B, halving with size
(*measured*, [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt),
[dclm-8b-rowscales](mesures/dclm-8b-rowscales-2026-09-21.txt),
[dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt)).

`LLVQ_TILE_BLOCKS` unset reads the row for the card: 64 on sm_89, 32 on sm_120, and 128 elsewhere, best of three on
the A100. Tile 128 to 64 on sm_89 gives Tetra **+17.6%** for zero bits, while `Planes14` moves 1.1% over the sweep:
the tile steals L1 from the decoder table (*measured*, [banc-252](mesures/banc-252-2026-10-06.txt)). Every figure
published before 2026-09-20 was measured at tile 128.

The `nullk` floor belongs to our launch geometry, not to the card: 2.306 ms for 252 projections without reading a
weight, where QTIP finishes the same projections in 2.246 ms (*measured*,
[f2-p3-qtip-banc](mesures/f2-p3-qtip-banc-2026-08-21.txt)). The comparable quantity across stacks is GB/s, never ×.

**Throughput falls with the context, much less under the served forms.** The served 4B decodes at 180.8 tok/s after a
5-token prompt, 169.5 at 2,048, 162.1 at 4,096 and 139.5 at 8,192, the product's context, where the first token takes
35.04 s (*measured*, [served-v4](mesures/served-v4-2026-11-05.txt)). The served 8B reads 106.9 / 93.1 / 82.1 / 67.1
at the same lengths through its door ([served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt)). The 14B reads
63.1 / 57.7 / 52.6 / 45.3 at the bench door (*measured*, [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). The expanded
form read 122.5, 82.7, 61.4 and 33.2 for the 4B (*measured*, [trims-contexts](mesures/trims-contexts-2026-10-30.txt)).
It concatenates the whole history at every step and copies each key and value head to its query heads, about 17 GB of
cache a token at 8,192 against 1.38 GB of weights (*computed*).

**Both served attention forms are licensed by a tolerance.** The grouped form stacks the query heads
of one key head as rows of one matmul and reads each cached byte once. It is not bit for bit on the card past 5
tokens: cuBLAS rounds a batch of 8 key heads differently from 32 expanded ones, up to 6.6e-2 on a logit. Paired
WikiText-2 perplexity through the kernel against the expanded form reads +0.0000 % by prefill at every size, and by
decode +0.0095 % [-0.0048 ; +0.0238] at 4B, -0.0048 % [-0.0117 ; +0.0021] at 8B and +0.0023 % [-0.0062 ; +0.0109]
at 14B, inside the 0.12 % band below (*measured*, [gqa-ppl](mesures/gqa-ppl-2026-11-01.txt),
[gqa-8b](mesures/gqa-8b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). A few isolated 4B decode steps
move one logit by up to 13.13, which no reading explains yet and which was not a criterion. The operator took this
change of gate on 2026-11-01.

The kernel form sends decode rows to the split-KV kernel `attn_dec.cu`. Against the grouped form it reads -0.0013 %
[-0.0074 ; +0.0047] over 12 windows of 4,096 and +0.0001 % [-0.0076 ; +0.0077] over 6 windows of 8,192 on the
positions from 1,280 (*measured*, [attn-t8-r2](mesures/attn-t8-r2-2026-11-04.txt)). With every row on the kernel from
position 0 it reads +0.0318 % [-0.0151 ; +0.0786] over 32 windows of 1,280 and +0.0018 % [-0.0053 ; +0.0088] over 12
windows of 4,096 (*measured*, [attn-t8-below](mesures/attn-t8-below-4b-2026-11-05.txt)). The operator granted the form
below 1,280 on the 4B only, on 2026-11-05.

**4E decodes a token in one launch and returns the eager path's tokens.** Under the served 4B config the door reads
178.0 tok/s [178.0-178.3] against 126.8 [126.7-126.9] for the eager engine, x1.404 [1.404-1.408] round by round, with
256 of 256 tokens equal (*measured*, [served-v4](mesures/served-v4-2026-11-05.txt)). The v4 row loop is x1.060
[1.059-1.062] over the v3 loop after a 5-token prompt, same tokens (*computed* from the round-by-round ratios,
[shaker-4e](mesures/shaker-4e-2026-11-05.txt)). The prefill is untouched. 4E runs on the Qwen3-4B's shape only, and
has no Metal kernel.

**The sealed 4B runs fused on Metal since 2026-11-06, and nothing is served there.** The int4 wall at d_in 8,192 is
lifted: the Metal fused door now loads the sealed 4B instead of refusing it. Its 12 int4 `down_proj` records go
through `tv_q4_metal_tiled`, bit-identical to the host reference (*measured* on an M3 Max, $0,
[metal-nuit](mesures/metal-nuit-2026-11-06.txt)). The `lutd` Tetra kernel decodes 52.07 tok/s [46.27-53.12] after a
5-token prompt, the served Metal kernel `lut` 50.85 and the dense reconstruction 7.81, with the same 256 tokens. That
ratio is raw: the dense arm copies its f16 head table every token, and no same-head arm exists on Metal (rule 4). The
grouped form returned the expanded form's logits bit for bit over 49,140 decode steps, and decodes x2.380
[2.362-2.396] faster at 4,096 positions. `attn_dec.metal` from position 0 reads +0.0051 % of perplexity against the
grouped form, inside the band. `served.rs` still refuses both forms on Metal; the two `*-metal-*-NOT-SERVED.json`
configs are measurement doors for `bin/metalrun`. An MLX prototype, `ops/mlx_tetra/`, decodes 93.92 tok/s
[88.38-94.66] on the same Mac (M16, same journal), not bit for bit with the Rust engine, perplexity not measured.
mlx-lm's 4-bit Qwen3-4B decodes 129.45 there, on another stack, so no ratio is formed.

**Noise.** The calibration draw carries a standard deviation of 2.92 pp of MMLU and 5.2% of perplexity at 4B over
three full runs (*measured*, [bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt),
[f5-graines](mesures/f5-graines-4b-2026-08-19.txt)). Every size is calibrated on one draw. For an A/B at constant file
the bar is the paired interval, 0.43 pp in MMLU and 0.12% in perplexity (*measured*,
[kvq8-4b](mesures/kvq8-4b-2026-08-15.txt)).

## 5. Open decisions

- **MMLU and perplexity through the served kernel**, with the int4 tables the b/param count. About $20 by the mixed
  route of [`ROADMAP.md`](ROADMAP.md) §2.1, about $70 for the full census at three sizes (*estimated*). `ppl` does not
  read `LLVQ_CONFIG` yet, which is code before it is a run.
- **`down_proj` instead of `o_proj` in int4 at 4B and 14B.** At 8B it scored 0.77 points better [0.26, 1.28] with fewer
  bytes, against our own prediction. Untested at the other two sizes.
- **A second calibration draw per size.** Each level is one draw, and the draw moves more than several gains.
- **A consumer GPU.** Speed is measured on an L40S and an A100, both data-center cards, and the served path loses on
  the A100. A consumer NVIDIA card, the target of the project, is unmeasured. The Mac runs the fused 4B and serves
  nothing (§4).
- **A model above 14B.** The `rot_apply` wall is lifted on the card, and the device rotation is proposed for a 32B's
  first light; the operator confirms. The 32B has to be encoded first.
- **Serving 4E v6 at 4B.** The operator gave the go to implement v6 on the 4B on 2026-11-07. The config
  `qwen3-4b-tetra-e4-4e-v6.json` is written and `served.rs` accepts `"rows": "v6"`. In the measurement form v6 reads
  x1.118 [1.117-1.118] over v4 after a 5-token prompt and x1.094 at 8,192, same tokens (*measured*,
  [pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)). At the served door that makes about 199 tok/s, and 152.6 at
  8,192 (*estimated*, served-v4 times those ratios). The door job `ops/jobs/served-v6.sh` costs about $0.42 on
  l40sx1 (*estimated*). It needs the stamp of `proofs/preregistration-served-v6-2026-11-07.md`, a Space rebuilt from
  the branch, and the operator's go. Serving v6 is the operator's call on that reading.
- **The v4 config on the Hub.** The moving repository still serves the grouped config at 133.3 tok/s (§2). Publishing
  needs the operator.
- **A served Metal config.** The grouped form is bit for bit with the expanded one on Metal (§4), and the two
  `NOT-SERVED` configs are ready. `served.rs` refuses both forms on Metal until the operator decides.
- **A full GSM8K census under 4E.** The 4B's served form has a 50-problem pilot only (§2). The grouped census of the
  4B cost $2.02 (*computed*, [gsm8k-grouped](mesures/gsm8k-grouped-2026-11-04.txt)).
- **An int4 KV cache.** It is the next throughput lever after 4E, and it touches quality, so it is outside any
  "negligible" go.
- **4E at 8B and 14B.** 4E runs on the 4B's shape only, and the kernel form has no T8 below 1,280 at those sizes.
- **The next venue for paper 2.** TACO desk-rejected paper 1 on 2026-08-27 on scope. Version 2 sits in release
  `v0.0.2`, and no archive (arXiv, Zenodo) holds it yet. Default if silent: preprint only.
- **A harder reasoning test**, Qwen3's reasoning mode or GSM-Symbolic, about 3 to 5 $ at 4B (*estimated*).
- **Loading in `transformers`.** The 4B is also published as safetensors that stay compressed,
  [Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra), read by the Python package `llvq-tetra/`.
  It gives the same 256 greedy tokens as our engine on a CPU, on Metal and on an L4 (*measured*,
  [plan-transformers](plan-transformers.md)). `llvq-tetra` 0.1.0 is on PyPI since 2026-10-07, the bytes that passed
  TestPyPI ([hf-pypi](mesures/hf-pypi-0.1.0-2026-10-07.txt)), so `pip install llvq-tetra` loads it. Next, on the
  operator's go: the upstream issue and PR, together.
- **Spend.** $274.05 over 270 rows of this branch's registry, 265 of them priced (*computed*, `docs/data/jobs.csv`);
  $274.46 with the served-v6 door's $0.41, recorded on `claude/serve-4e-v6-6e15c0`. The jobs
  of 2026-11-04 and 11-05 spent $21.27 over 25 rows (*computed*). No cap is in force; one is owed before the next
  paid job.

## 6. Closed absent a new idea

| lead | why | source |
|---|---|---|
| E1v on the served path | 0.25× f16 | [e1v-cuda](mesures/e1v-cuda-2026-08-16.txt) |
| `Golay70` | 1.77× against a stamped threshold of 2.0× | [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt) |
| E3 | 3.0444 kernel b/weight against a criterion of 2.60 | [radixstudy](mesures/radixstudy-x4-2026-08-12.txt) |
| Design C | ×1.99 of perplexity at 28 blocks | [verdicts-nuit](archive/verdicts-nuit-2026-08-07.md) |
| `group_scales` | 44.66 to 53.60 of perplexity at 28 blocks | smoke of 2026-07-28, no journal |
| Spherical GPTQ feedback | +31.89% of perplexity at the 0.6B, fifth case of a local reading composing worse | [sph-ppl](mesures/sph-ppl-0.6b-2026-09-22.txt) |
| A2, CUDA Graphs, served | +12.6% of throughput for +47% of VRAM at 4B | [ECARTS](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7 |
| Euclidean gain rule | dead end to end | [HISTORIQUE](HISTORIQUE.md), 2026-09-15 |
| 4E v5, the last rows spread over the warps | x1.001 over v4 after a 5-token prompt, its range touching x1.00; `"rows": "v5"` is refused by name | [shaker-4e](mesures/shaker-4e-2026-11-05.txt) |
| A mixed `Tetra` plus int4 bench arm | the bench compares formats on the same 252 matrices, and the served files are timed end to end | operator, 2026-10-06 |

Reopening any of them needs the condition written in its own row, not a new argument. A2 stays out of the core.
