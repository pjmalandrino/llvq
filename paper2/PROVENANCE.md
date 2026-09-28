# Paper 2 provenance

Each number in paper 2 is *measured* (read off a run), *computed* (arithmetic on measured quantities) or *cited*.
The table below gives, for each claim, its nature and the log or file that holds it.

The logs are files under `docs/mesures/`, all dated 2026, and the table shortens their names:
`f1a-comptes-09-04` is `docs/mesures/f1a-comptes-2026-09-04.txt`. "Raw output" is the `-brut/` directory beside a
log. The CSVs are in `docs/data/`. `paper2/scripts/make_figures.py` draws every data figure from them, and
`paper2/scripts/check_tables.py` stops the build if a table no longer matches its CSV.

| claim | nature | source |
|---|---|---|
| Planes14 reads 4.804 b/weight | measured | echelle-formats.csv; e2-golay70-bench-08-07 |
| ball holds 1.1·10¹⁴ points | computed | shells 2 to 12 of the class enumeration; theta-series test in `llvq-search` |
| 64 states a cut, 1,024 edges, N₀ = 1240 | computed | f1a-comptes-09-04 (states, edges); f1-rang-plancher-09-05 (N₀); all three asserted when `llvq-search` builds the tables |
| tables, 18,816 B | computed | sum of the five sizes printed at every launch; banc-t64-09-20 (raw output only) |
| m ≤ 26, max \|y_j\| = 10 | computed | `llvq-bench/examples/tetrashell.rs`, exhaustive; paper2-audit-09-28 |
| about 150 instructions a block, 20 more for gain and norm; about 380 plus 24 conversions for a decoder that converts | computed | `llvq-cuda/kernels/llvq_f1rank_v3.cuh` (default mask; its header counts 141 for the one-`prmt` mask), `llvq-cuda/kernels/llvq_tetra48.cuh`; recount in paper2-audit-09-28 |
| retention, 88.80 against 91.98, and 3.2 points on real blocks | measured | f1-encodeur-blocs-reels-09-05 |
| 0.6 points of retention from the cost order | measured | `llvq-bench/examples/f1rankbench.rs`; `docs/HISTORIQUE.md`, 2026-09-05 |
| shaping, 0.7292 against 1.0958 dB | computed, idealized | `ops/f1a_shaping.py` |
| G(E₈)/G(Λ₂₄) = 1.0899 | cited, computed | Conway and Sloane 1999, Table 2.3 |
| ten-arm benchmark: the kernel benchmark table of section 5.2, and Appendix A | measured | echelle-formats.csv; banc-t64-09-20 (raw output only); QTIP row: f2-p3-qtip-banc-08-21 |
| tile sweep, section 5.2 (the activation tile) | measured | tuile-l40s.csv; tuile-l40s-09-20 |
| sealed files, table of section 4: bytes, bits per parameter | measured, computed | `rtbits`; sealed-4b-09-23 (raw output), sealed-rownorms-09-23, sealed-8b-27-09-24, sealed-8b-14b-09-23 |
| MMLU of the sealed files; at 4B, of the same weights with the int4 matrices rebuilt at load | measured | embed-q4-swap-09-23, sealed-8b-27-09-24, sealed-8b-14b-09-23; question fingerprint `a74a6d62`; the 4B file gives the same answers and logits on 57 of 57 questions, sealed-4b-09-23 (raw output, `metal-smoke/`), control 1 of its preregistration |
| steps of the build, table of section 6.5 | measured, computed | paper2-chain.csv; dclm-rowscales-09-20 (4B interval: paper2-audit-09-28, raw output), dclm-8b-rowscales-09-21, dclm-14b-rowscales-09-22, census-8b-09-21, census-14b-base-09-22, embed-q4-swap-09-23, sealed-8b-27-09-24, sealed-8b-14b-09-23 |
| int4 `v_proj` at 4B, +1.73 on the full test set, for 0.05 bits per parameter | measured, computed | tetra-nu-full-09-18; references-comptabilite-09-18 |
| FP16 and AWQ MMLU, 8B and 14B | measured | census-8b-09-21; census-14b-ref-09-22 |
| FP16 MMLU 4B | measured | f16-full-09-18 |
| AWQ and IQ2_XXS MMLU 4B | measured | paper-table-09-25 |
| paired MMLU gaps, section 6.2 | measured | paper2-gaps.csv; the logs of the MMLU rows above |
| FP16 and AWQ tok/s in vLLM | measured | awq-vllm-4b-08-17; paper-table-09-25 |
| IQ2_XXS tok/s | measured | m4-iq2-cuda-08-30; the `llama-bench` flags, `-p 0 -n 128 -r 5`, read back by `hf jobs inspect` in paper2-audit-09-28 (raw output) |
| IQ2_XXS bits per parameter and GB | computed | 1,246,620,832 B in m3-iq2-metal-08-30; the same GGUF, sha256 `19a8ed49`, in m4-iq2-cuda-08-30 and paper-table-09-25 |
| f16 dense path tok/s, our engine | measured | paper-table-09-25, dense arm of each served run |
| served tok/s and GB of the sealed files | measured | paper-table-09-25 |
| 14B tokens against the dense reconstruction: first divergence at 78, at 137 with f16 tables; prefill gate, largest logit difference 0.095, 1.35 and 0.79 for logits up to 30.4, 36.6 and 27.5 | measured | paper-table-09-25 and its raw output (`prefill-203.txt`, `fused-q4-256.txt`, `fused-f16-256.txt`); the gate is in `llvq-llm/src/bin/fusedrun.rs` |
| 8B choice, `o_proj` against `down_proj`, section 6.5 | measured, computed | sealed-8b-27-09-24 |
| selection on the MMLU test set, section 6.5: the 8B better of two files; the 4B window 12 to 23, best of three on the full test set; the int4 types ranked on 2,280-question samples; the 14B window rule | measured, cited from the preregistration | sealed-8b-27-09-24; downproj-slices-09-18; m2-attribution-4b-09-02, q5-alloc-int4-09-16; `proofs/preregistration-sealed-8b-14b-2026-09-23.md` |
| `o_proj` gain +3.12 on the 2,280 questions that selected it, +1.55 on the other 11,762 | measured | oproj-int4-full-09-17 |
| second training (row scales and RMSNorm), 4B, in the predictions table of Appendix B | measured | sealed-rownorms-09-23 |
| three calibration draws, Planes14 4B, 2,280 questions: MMLU range 5.83, s.d. 2.92 | computed | bruit-mmlu-graines-4b-08-25 |
| GSM8K of the sealed files, prompt fingerprint `bfa9135c` | measured | gsm8k-wave1-09-26, gsm8k-wave2-09-26 |
| GSM8K of FP16 and AWQ, the engine check, the same-engine gap of 9.02 | measured | paper2-gsm8k.csv, paper2-gsm8k-gaps.csv; the logs of the row above |
| GSM8K paired gaps, section 6.3 | measured | paper2-gsm8k-gaps.csv; the same logs |
| gaps one size apart, section 6.4 | measured | paper2-sizeup.csv; sizeup-09-28; not preregistered |
| KV cache and memory budgets, section 6.4 | computed | paper2-sizeup-budget.csv; sizeup-09-28 |
| our files at 64 to 65 % of AWQ's bits per parameter if AWQ stored its tables in 4 bits, section 6.2 | computed | card bytes and AWQ bytes with 4.5-bit tables in sizeup-09-28 |
| bare Tetra against Planes14, and the encoder drift | measured | tetra-4b-09-06, and deviation É3 of its preregistration in `proofs/`; paired interval in paper2-audit-09-28 |
| sm_120 tile sweep, 0.82 to 1.03 times Planes14, in Limitations | measured | f1d-09-10 |
| load times 4.1 and 72.9 s, 96 rotation launches a token | measured | paper-table-09-25 |
| row padding, 4.1 MB | measured | banc-t64-09-20 (raw output only) |
| AWQ padding, 10.1 MB, and w4g128 at 4.156 b/weight without it | computed | `awq_strides` in `planesbench`, the 4B shapes |
| tokenizer and config in the file, 11.4 MB | measured | `docs/fiche-4b.md`, on the Planes14 file; the sealed 4B file ends with the same `tokenizer.json`, 11,422,654 B, sha256 `aeb13307`, which awq-vllm-4b-08-17 finds in every Qwen3 checkpoint |
| z = 1.1 for the fall of the AWQ gap from 4B to 8B | computed | standard errors of the paired MMLU gaps, from the logs of the paired-gaps row |
| AWQ bits per parameter | computed | echelle-4b-8b.csv; rtbits-planes-8b-08-09, rtbits-14b-08-17 |
| MMLU of LLVQ, QTIP and QuIP# | cited | van der Ouderaa et al. 2026, Table 6 |
