# Paper 2: Tetra

`main.tex` is the paper: *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7
Bits per Parameter*. Single column, standard arXiv structure: introduction,
background, the Tetra codebook, from codebook to served file, kernel,
experiments, limitations, conclusion, then two appendices (the ten-arm
kernel benchmark and the record of predictions). `PROVENANCE.md` gives the
nature and the source of every claim, and the commands that rerun the main
measurements are under [Reproduce](#reproduce).

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
scores and paired gaps, Table 4), `paper2-sizeup.csv` and `paper2-sizeup-budget.csv`
(the comparison one size up, Table 5, and the memory budgets), `paper2-chain.csv` (the steps and the sealed files),
`tuile-l40s.csv` (tile sweep), and paper 1's `echelle-formats.csv` and
`echelle-4b-8b.csv`.

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
ending in `-tetra-e4.json`; the commands below use the 4B one. On an NVIDIA GPU:

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

## State (2026-09-28)

Complete: 15 pages, no error, no overfull or underfull box, no undefined
reference, and `RELEASE=1 make check` passes (no pending cell). It was cut from
20 pages on 2026-09-28 without changing a number: the provenance table moved to
`PROVENANCE.md`, the commands to Reproduce above, the tile and scale figures
were dropped with their values kept in the text, and repeated caveats are
stated once. The four review points of the same day ("not separated" for ties,
the 14B divergence, selection on the MMLU test set, the embedding tables in the
memory comparison) brought it back to 15. The served speeds
of the three sealed files come from paper-table-2026-09-25. GSM8K, scored through
the served kernel, comes from gsm8k-wave1-2026-09-26 and gsm8k-wave2-2026-09-26
(§6.3). The comparison one size up comes from sizeup-2026-09-28 (§6.4). Every
number and every cited source was audited on 2026-09-28
([paper2-audit-2026-09-28](../docs/mesures/paper2-audit-2026-09-28.txt)).

Writing rule for every revision: short factual sentences, no em dash.
