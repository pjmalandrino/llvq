# Project state as of 2026-09-28

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

| size | sealed file | b/param | MMLU | GSM8K | tok/s | GB on card |
|---|---|---|---|---|---|---|
| 4B | `qwen3-4b-sealed.bin` | 2.73 | 63.37 | 82.49 | 113.8 | 1.38 |
| 8B | `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 95.0 | 2.76 |
| 14B | `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 57.2 | 5.04 |

b/param: *computed* by `rtbits` on the file's bytes, whole model, int4 tables included. MMLU: *measured* on all 14,042
test questions, 5-shot, micro average, question fingerprint `a74a6d62`. GSM8K: *measured* through the served
kernel on all 1,319 test problems, zero-shot, reasoning block empty, greedy, prompt fingerprint `bfa9135c`
([gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)). Speed and GB: *measured* on one L40S, batch 1,
256 greedy tokens, median of five rounds
([embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt), [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt),
[sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt), [paper-table](mesures/paper-table-2026-09-25.txt)).

**MMLU is scored on the dense reconstruction of the sealed file, not through the kernel.** The per-row checks against
f64 and a 256-token comparison tie the two: identical tokens at 4B and 8B, first difference at token 78 at 14B. One
census has been read both ways, on 2,280 questions at 4B on the 2026-09-09 object: the kernel scored 55.66 against the
dense path's 55.52, three discordant questions, McNemar p = 0.25 (*measured*,
[f1e-census](mesures/f1e-census-2026-09-11.txt)). The b/param above count the int4 tables the kernel reads; the MMLU
above was read with those tables in f16. Closing that is the first open decision of section 5. GSM8K is scored
through the kernel, and on 50 problems the kernel and the dense reconstruction give the same 50 answers.

## 3. Against the baselines

| size | arm | b/param | MMLU | GSM8K | tok/s | GB |
|---|---|---|---|---|---|---|
| 4B | FP16, vLLM | 16.00 | 70.14 | 92.12 | 83.1 | 8.04 |
| 4B | AWQ w4g128, vLLM | 5.30 | 68.14 | 89.01 | 200.5 | 2.67 |
| 4B | IQ2_XXS, llama.cpp | 2.48 | 39.78 | not scored | 312.9 | 1.25 |
| 8B | FP16, vLLM | 16.00 | 75.05 | 93.25 | 46.3 | 16.38 |
| 8B | AWQ w4g128, vLLM | 5.96 | 73.79 | 92.95 | 123.2 | 6.10 |
| 14B | FP16, vLLM | 16.00 | 78.88 | 95.30 | 25.8 | 29.54 |
| 14B | AWQ w4g128, vLLM | 5.40 | 78.12 | 95.38 | 77.8 | 9.98 |

All *measured*, same journals as section 2, plus [census-8b](mesures/census-8b-2026-09-21.txt),
[census-14b-ref](mesures/census-14b-ref-2026-09-22.txt), [f16-full](mesures/f16-full-2026-09-18.txt),
[m4-iq2-cuda](mesures/m4-iq2-cuda-2026-08-30.txt). Speeds come from different engines and are never divided across
them (rule 5).

Paired gaps on the same questions, our file against the arm. MMLU *computed*, `docs/data/paper2-gaps.csv`; GSM8K
*measured* by `gsm8kpair`, same journals as section 2:

| size | MMLU below FP16 | MMLU below AWQ | GSM8K below FP16 | GSM8K below AWQ |
|---|---|---|---|---|
| 4B | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |
| 8B | 5.48 [4.83, 6.10] | 4.21 [3.54, 4.86] | 4.62 [3.00, 6.25] | 4.32 [2.83, 5.81] |
| 14B | 3.22 [2.69, 3.75] | 2.46 [1.92, 3.00] | 3.26 [1.93, 4.59] | 3.34 [2.12, 4.55] |

We lose quality and win memory: 45 to 52% of AWQ's bits per parameter. Both gaps shrink as the model grows. At 4B we
score 23.6 points above IQ2_XXS for 0.25 more bits per parameter.

On GSM8K the loss exceeds the MMLU loss at 4B only. At 8B and 14B the GSM8K interval contains the MMLU gap. The GSM8K
references run in vLLM and our files in our kernel; the engine gate held at 4B, FP16 reading 91.51 in our dense path
against 92.12 in vLLM, −0.61 [−1.27, +0.06]. GSM8K saturates, FP16 reading 92 to 95%, and says nothing of Qwen3's
reasoning mode.

One size up, at about the same memory, our file beats AWQ on MMLU and ties it on GSM8K (*measured*, paired over the
committed dumps, not preregistered, [sizeup-2026-09-28](mesures/sizeup-2026-09-28.txt)). Our 8B against AWQ's 4B:
+1.44 [+0.71, +2.19] MMLU, −0.38 [−2.14, +1.39] GSM8K, 2.76 against 2.67 GB. Our 14B against AWQ's 8B: +1.87
[+1.20, +2.57] MMLU, −0.91 [−2.35, +0.53] GSM8K, 5.04 against 6.10 GB. Weights plus an 8k f16 KV cache put us one size
above AWQ from 3.97 to 11.32 GB (*computed*); if AWQ stored its tables in 4 bits too, from 3.97 to 5.52 and from 6.39
to 9.08 GB.

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
- **Publishing the sealed files.** The 4B is published since 2026-10-02, in two public repositories:
  [Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra), safetensors that stay compressed and
  that `transformers` reads, and
  [Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed), the single file the Rust
  engine reads. The Hub computed the sealed file's SHA-256 itself and it is the paper's,
  `886391a8c03f66dc`, so a reader verifies that digest without trusting us. **The 8B and the 14B are still
  unhosted, and their files no longer exist**: sealed locally on 2026-09-23 and deleted in a disk cleanup. Their
  inputs survive in the bucket and the chain is in the journals, so re-sealing is local and free, and it would say
  whether the paper's other two digests are reproducible.
  [plan-transformers](plan-transformers.md) has the remaining stages: PyPI is not done, and no precompiled kernel is
  published, so the fused arms compile at import.
- **The next venue for paper 2.** TACO desk-rejected paper 1 on 2026-08-27 on scope. Default if silent: preprint only.
- **A harder reasoning test.** Qwen3's reasoning mode, or GSM-Symbolic's unseen variants, about 3 to 5 $ at 4B
  (*estimated*). GSM8K cannot separate the 8B and 14B losses from their MMLU losses.
- **The model card on the Hub.** Settled for the 4B. `docs/hf-model-card.md` IS the sealed repository's card, byte
  for byte below its STATUS comment, because two cards drifted apart between 2026-09-27 and 2026-10-02; edit there,
  then re-upload. `Pier-Jean/Qwen3-4B-LLVQ-2bit` is untouched and still holds the August `Planes14` objects at zero
  downloads, with [fiche-4b](fiche-4b.md) as their provenance register.
- **Spend.** $242.18 over 220 priced jobs (*measured*, `docs/data/jobs.csv`). The two GSM8K waves spent 7.13 $ each
  under caps of 15.75 and 18.90 $. The operator set a $5 cap on the `transformers` wave on 2026-09-30, of which
  stage 4 spent $0.30 over seven launches. Six of those rows live only on the branch `hf-safetensors`, which is not
  to be merged, so the registry and `main` disagree until that is settled.

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
