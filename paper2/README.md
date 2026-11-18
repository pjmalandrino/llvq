# Paper 2: Tetra

`main.tex` is the paper: *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7
Bits per Parameter*. Single column, standard arXiv structure: introduction,
background, the Tetra codebook, from codebook to served file, kernel,
experiments, limitations, conclusion, then two appendices (the ten-arm
kernel benchmark and the record of predictions). `PROVENANCE.md` gives the
nature and the source of every claim, and the commands that rerun the main
measurements are under [Reproduce](#reproduce). The published PDF, of 2026-10-06, is attached to release
[`v0.0.2`](https://github.com/pjmalandrino/llvq/releases/tag/v0.0.2). The source in this directory is ahead of
that PDF since 2026-11-17 and is not released.

## Layout and art direction

| | paper 1 | paper 2 |
|---|---|---|
| class | `acmsmall`, `nonacm=true` | `article` 10 pt, text block 5.5 in x 9 in |
| body font | Linux Libertine (from acmart) | Linux Libertine, loaded explicitly |
| figures with numbers | `scripts/make_figures.py`, matplotlib to PDF, drawn at 5.5 in | same, printed at 1:1 |
| palette | Okabe-Ito, `INK`/`GRAY`/`LIGHT` | same constants; Tetra is blue, the earlier Planes14 sky blue |
| rcParams | serif 8 pt, grey axes, no top/right spine, `#DDDDDD` grid | copied verbatim |
| markers | circles = ours, squares = deployed kernels | same |
| schematics | TikZ, `box`/`mem`/`flow`/`panel` styles | same styles, in `main.tex` |
| guard | `require_plotted` + `check_tables.py` | same |
| bibliography | ACM-Reference-Format | `plainnat`, numeric |

Data in `docs/data/`: `paper2-results.csv` (main table), `paper2-gaps.csv`
(paired MMLU gaps), `paper2-gsm8k.csv` and `paper2-gsm8k-gaps.csv` (GSM8K
scores and paired gaps, Table 6), `paper2-sizeup.csv` and `paper2-sizeup-budget.csv`
(the comparison one size up, Table 10, and the memory budgets), `paper2-chain.csv` (the steps and the sealed files),
`tuile-l40s.csv` (tile sweep), and paper 1's `echelle-formats.csv` and
`echelle-4b-8b.csv`. The revision of 2026-11-17 adds `paper2-duel.csv` and `paper2-duel-gaps.csv`
(llama.cpp IQ2 against our files, Tables 7 and 8), `paper2-context.csv` (decode speed
and first token against prompt length), `paper2-peak.csv` (peak card memory),
`paper2-batch.csv` (several requests) and `paper2-ablation.csv` (where the 4B
loses its points). Seven data figures are drawn from these files.

```bash
make                 # figures, then check, then latexmk
RELEASE=1 make check # refuses the build while a \pend cell is left
make arxiv           # arxiv-tetra.tar.gz: sources + figures + main.bbl
make clean
```

`\pdfoutput=1` must stay inside the first five lines of `main.tex`.

## Reproduce

The three sealed files, with the first eight hex digits of their SHA-256:

- `qwen3-4b-sealed.bin`, `886391a8`
- `qwen3-8b-sealed-B.bin`, `7bdb9a55`
- `qwen3-14b-sealed.bin`, `61db37fe`

Each file has a served configuration in `configs/`, named after its size and
ending in `-tetra-e4-4e-v9.json`; the 4B also has `-4e-v9-chain.json`, its
chain prefill. The files ending in `-tetra-e4.json` are the per-matrix
configurations with the expanded attention, which every score ran under. On an NVIDIA GPU:

```bash
# the served door: tok/s of the persistent kernel, 256 tokens after 5
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-4e-v9.json cargo run --release \
  -p llvq-llm --features cuda --bin fusedrun -- <file> 256
# the same after longer prompts: first token and decode speed
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-4e-v9.json LLVQ_CONTEXTS=5,2048,4096,8192 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun -- <file> 256
```

The llama.cpp side of section 6.4 is `ops/jobs/duel-iq2.sh`, and its tables
are printed by `python3 ops/duel/analyze.py docs/mesures/duel-iq2-2026-11-08-brut`.
The per-matrix measurements:

```bash
# tok/s and GB at the served flags, 256 tokens, against the dense arm
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 \
  LLVQ_EMBED=q4 cargo run --release -p llvq-llm --features cuda \
  --bin fusedrun -- <file> 256
# MMLU on the full test set, then a paired comparison of two dumps
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=<dump> cargo run --release \
  -p llvq-llm --features cuda --bin mmlu -- <file> cuda
cargo run --release -p llvq-llm --bin mmlupair -- <A> <B> --no-fpc
# GSM8K through the served kernel, then a paired comparison
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json LLVQ_GSM8K_DUMP=<dump> \
  cargo run --release -p llvq-llm --features cuda --bin gsm8k -- <file> cuda
cargo run --release -p llvq-llm --bin gsm8kpair -- <A> <B>
# bits per parameter over the whole model
cargo run --release -p llvq-bench --bin rtbits -- <file>
```

`mmlu` and `gsm8k` print a fingerprint of the prompts they scored. Two scores
compare only when their fingerprints match, and the two pairing programs refuse
dumps whose fingerprints differ.

## State (2026-11-17)

23 pages, no error, no overfull box, no undefined reference, and
`RELEASE=1 make check` passes. The paper has one published form, the PDF of
release `v0.0.2`; this source is its revision and says nothing of versions.
It adds the persistent kernel (section 5.3), the served speeds under it, the
comparison with llama.cpp's IQ2_S at three sizes (section 6.4), long prompts
and several requests (section 6.5) and the 4B ablation (section 6.8). No
score moved. The earlier speeds stay in the text as the per-matrix path with
the expanded attention. Its results section follows the rules of `docs/STYLE.md` of
2026-11-17: one message a paragraph, the numbers in the tables (Tables 5, 8
and 9 were added for that), captions above tables. Sections 2 to 4 were only
scanned for the listed tells. Every new number is copied from its journal in
`docs/mesures/` and listed in `PROVENANCE.md`; the audit of 2026-09-28
([paper2-audit-2026-09-28](../docs/mesures/paper2-audit-2026-09-28.txt)) does
not cover them.

Writing rule for every revision: short factual sentences, no em dash.
