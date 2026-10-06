# Project state as of 2026-10-06

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
`LLVQ_KV=f16`. The files that carry it are in [`configs/`](../configs/README.md), one per size.

| size | sealed file | b/param | MMLU | GSM8K | ppl | tok/s | GB on card |
|---|---|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 82.49 | 12.58 | 113.8 | 1.38 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 9.71 | 95.0 | 2.76 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 8.44 | 57.2 | 5.04 |

b/param: *computed* by `rtbits`, whole model, int4 tables included. MMLU: *measured* on all 14,042 test questions,
5-shot, micro average, fingerprint `a74a6d62` ([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt),
[sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt), [sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)). GSM8K:
*measured* through the served kernel on all 1,319 test problems, zero-shot, greedy, fingerprint `bfa9135c`
([gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)). Perplexity:
*measured* on the raw WikiText-2 test split, 12 windows of 4,096, f16
([ppl-scelles](mesures/ppl-scelles-2026-10-04.txt)). Speed and GB: *measured* on one L40S, batch 1, 256 greedy
tokens, median of five rounds ([paper-table](mesures/paper-table-2026-09-25.txt)).

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
not cuBLAS (1.14×). There the served files decode at 0.92, 0.91 and 0.95 times our dense path with f16 tables, and at
74.2, 57.9 and 39.0 tok/s where vLLM decodes FP16 at 145.6, 91.8 and 52.7 (*measured*,
[a100](mesures/a100-2026-10-06.txt)).

**No model above 14B is served.** The `rot_apply` wall of [format-noyau](format-noyau.md) §8 closes that path whatever
the format, and the 32B point has never been encoded.

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
  the A100. Local hardware, the target of the project, is unmeasured.
- **A model above 14B**, which needs the `rot_apply` wall lifted first.
- **The next venue for paper 2.** TACO desk-rejected paper 1 on 2026-08-27 on scope. Version 2 sits in release
  `v0.0.2`, and no archive (arXiv, Zenodo) holds it yet. Default if silent: preprint only.
- **A harder reasoning test**, Qwen3's reasoning mode or GSM-Symbolic, about 3 to 5 $ at 4B (*estimated*).
- **Spend.** $246.45 over 221 priced jobs (*measured*, `docs/data/jobs.csv`). The review campaign of 2026-10-04 to
  10-06 spent $4.57 under a $9 cap. No cap is in force; one is owed before the next paid job.

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
| A mixed `Tetra` plus int4 bench arm | the bench compares formats on the same 252 matrices, and the served files are timed end to end | operator, 2026-10-06 |

Reopening any of them needs the condition written in its own row, not a new argument. A2 stays out of the core.
