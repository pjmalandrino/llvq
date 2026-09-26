# Project state as of 2026-09-26

## 1. The project

LLVQ quantizes the weights of an LLM to about 2 bits on the Leech lattice, in Rust. The goal is to fit larger models
on local hardware. The repository carries the quantizer, the file format, and a fused CUDA kernel that decodes and
multiplies without going back through f16.

Three Qwen3 models are sealed and served at about 2.7 bits per parameter: 4B, 8B, 14B. The first manuscript is on
Zenodo (DOI [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)) and was submitted to arXiv as sources
on 2026-09-02. The second, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*, is written and
compiles, in [`paper2/`](../paper2/README.md).

## 2. Served configuration

One layout, one setting, at all three sizes: `tetra48`, `LLVQ_EMBED=q4`, `LLVQ_ROT_SHARE=1`, `LLVQ_FUSE=0`,
`LLVQ_KV=f16`. The files that carry it are in [`configs/`](../configs/README.md), one per size.

| size | sealed file | b/param | MMLU | tok/s | GB on card |
|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 113.8 | 1.38 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 95.0 | 2.76 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 57.2 | 5.04 |

b/param: *computed* by `rtbits` on the file's bytes, whole model, int4 tables included. MMLU: *measured* on all 14,042
test questions, 5-shot, micro average, question fingerprint `a74a6d62`. Speed and GB: *measured* on one L40S, batch 1,
256 greedy tokens, median of five rounds
([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt), [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt),
[sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt), [paper-table](mesures/paper-table-2026-09-25.txt)).

**MMLU is scored on the dense reconstruction of the sealed file, not through the kernel.** The per-row checks against
f64 and a 256-token comparison tie the two: identical tokens at 4B and 8B, first difference at token 78 at 14B. One
census has been read both ways, on 2,280 questions at 4B on the 2026-09-09 object: the kernel scored 55.66 against the
dense path's 55.52, three discordant questions, McNemar p = 0.25 (*measured*,
[f1e-census](mesures/f1e-census-2026-09-11.txt)). The b/param above count the int4 tables the kernel reads; the MMLU
above was read with those tables in f16. Closing that is the first open decision of section 5.

## 3. Against the baselines

| size | arm | b/param | MMLU | tok/s | GB |
|---|---|---|---|---|---|
| 4B | FP16, vLLM | 16.00 | 70.14 | 83.1 | 8.04 |
| 4B | AWQ w4g128, vLLM | 5.30 | 68.14 | 200.5 | 2.67 |
| 4B | IQ2_XXS, llama.cpp | 2.48 | 39.78 | 312.9 | 1.25 |
| 8B | FP16, vLLM | 16.00 | 75.05 | 46.3 | 16.38 |
| 8B | AWQ w4g128, vLLM | 5.96 | 73.79 | 123.2 | 6.10 |
| 14B | FP16, vLLM | 16.00 | 78.88 | 25.8 | 29.54 |
| 14B | AWQ w4g128, vLLM | 5.40 | 78.12 | 77.8 | 9.98 |

All *measured*, same journals as section 2, plus [census-8b](mesures/census-8b-2026-09-21.txt),
[census-14b-ref](mesures/census-14b-ref-2026-09-22.txt), [f16-full](mesures/f16-full-2026-09-18.txt),
[m4-iq2-cuda](mesures/m4-iq2-cuda-2026-08-30.txt). Speeds come from different engines and are never divided across
them (rule 5).

Paired gaps on the same questions, our file against the arm (*computed*, `docs/data/paper2-gaps.csv`):

| size | below FP16 | below AWQ |
|---|---|---|
| 4B | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] |
| 8B | 5.48 [4.83, 6.10] | 4.21 [3.54, 4.86] |
| 14B | 3.22 [2.69, 3.75] | 2.46 [1.92, 3.00] |

We lose quality and win memory: 45 to 52% of AWQ's bits per parameter. Both gaps shrink as the model grows. At 4B we
score 23.6 points above IQ2_XXS for 0.25 more bits per parameter.

## 4. Structural facts

`Tetra` reads **2.148 kernel b/weight** in VRAM against 4.804 for `Planes14`, for the same 48 bits a block (*computed*,
`docs/data/echelle-formats.csv`). The product triplet in force since 2026-08-16 (8k context, 5 GB margin, 32 GiB unit,
offload as reference only) sets b_max = 3.00 kernel b/weight. `Tetra` is the first layout under it, 28% clear, which
moves the admissible class from 43.3 to 81 to 101 billion parameters (*computed*).

**No model above 14B is served.** The `rot_apply` wall of [format-noyau](format-noyau.md) §8 closes that path whatever
the format, and the 32B point has never been encoded.

Trained row scales gain **+3.15, +3.29 and +1.67 MMLU points for zero bits** at 4B, 8B and 14B (*measured*,
[dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt),
[dclm-8b-rowscales](mesures/dclm-8b-rowscales-2026-09-21.txt),
[dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt)). The gain halves from the 4B to the 14B.

`LLVQ_TILE_BLOCKS` unset reads the measured row for the card since 2026-09-20: 64 on sm_89, 32 on sm_120. Going from
tile 128 to 64 on sm_89 is worth **+16.1%** to Tetra, for zero bits and a bit-identical output. Across the whole sweep
the Tetra arm swings 19.1% where `Planes14` swings 1.5%, which is what makes this a measurement of the mechanism: the
tile steals L1 from the decoder table (*measured*, [tuile-l40s](mesures/tuile-l40s-2026-09-20.txt)). Every figure
published before that date was measured at tile 128.

The `nullk` floor belongs to our launch geometry, not to the card: 2.306 ms for 252 projections without reading a
weight, where QTIP finishes the same projections in 2.246 ms (*measured*,
[f2-p3-qtip-banc](mesures/f2-p3-qtip-banc-2026-08-21.txt)). The comparable quantity across stacks is GB/s, never ×.

**Noise, and what a bar must clear.** The calibration draw carries 2.92 pp of MMLU and 5.2% of perplexity at 4B over
three full runs (*measured*, [bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt),
[f5-graines](mesures/f5-graines-4b-2026-08-19.txt)). Every size is calibrated on one draw. For an A/B at constant file
the bar is the paired interval, 0.43 pp in MMLU and 0.12% in perplexity (*measured*,
[kvq8-4b](mesures/kvq8-4b-2026-08-15.txt)).

The repository no longer reproduces the artifact published in 2026-08: one block re-encoded on 2026-09-06 differs from
the published file on the tail, the gains and 87% of the indices, most likely from commit `4a3e5f0` of 2026-08-26
changing the calibration volume (*measured*, `llvq-bench/examples/driftcheck.rs`, *not proved*). Consequence: the
`Planes14` figures of 2026-08 compare to nothing encoded after that date. The three sealed files of section 2 were all
encoded after it, and the evaluation harness is intact.

## 5. Open decisions

- **MMLU and perplexity through the served kernel**, with the int4 tables the b/param count. This is the paper's own
  first limitation. About $20 by the mixed route (kernel on 2,280 questions plus a full dense run with the int4 tables
  written in the file), about $70 for the full census through the kernel at three sizes (*estimated*). `ppl` does not
  read `LLVQ_CONFIG` yet, which is code before it is a run.
- **The kernel bench does not time equal work**: 216 matrices for `Tetra` against 252 for every other arm, because the
  int4 `v_proj` are counted and not timed. Any × formed on those passes is unequal.
- **`down_proj` instead of `o_proj` in int4 at 4B and 14B.** At 8B it scored 0.77 points better [0.26, 1.28] with fewer
  bytes, against our own prediction. Untested at the other two sizes.
- **A second calibration draw per size.** Each absolute level is one draw, and the draw carries more than several of
  the gains we report.
- **A second kind of GPU.** Every number is on one L40S. On an A100 none of our earlier lattice kernels beat FP16.
- **A model above 14B**, which needs the `rot_apply` wall lifted first.
- **Publishing the sealed files.** The paper gives their SHA-256 and nothing hosts them, so nobody outside can replay
  an MMLU.
- **The next venue for paper 2.** TACO desk-rejected paper 1 on 2026-08-27 on scope. Default if silent: preprint only.
- **Spend.** $227.62 over 204 priced jobs (*measured*, `docs/data/jobs.csv`). No cap is in force; one is owed before
  the next paid job.

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

Reopening any of them needs the condition written in its own row, not a new argument. A2 stays out of the core.
