# Paper 2: Tetra

`main.tex` is the paper: *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7
Bits per Parameter*. Single column, standard arXiv structure: introduction,
background, the Tetra codebook, from codebook to served file, kernel,
experiments, limitations, conclusion, then three appendices (the ten-arm
kernel benchmark, the record of predictions, defects and commands, the
provenance table).

## Layout and art direction

| | paper 1 | paper 2 |
|---|---|---|
| class | `acmsmall`, `nonacm=true` | `article` 10 pt, text block 5.5 in x 9 in |
| body font | Linux Libertine (from acmart) | Linux Libertine, loaded explicitly |
| figures with numbers | `scripts/make_figures.py`, matplotlib to PDF, drawn at 5.5 in | same, printed at 1:1 |
| palette | Okabe-Ito, `INK`/`GRAY`/`LIGHT` | same constants; Tetra is blue, the earlier Planes14 sky blue |
| rcParams | serif 8 pt, grey axes, no top/right spine, `#DDDDDD` grid | copied verbatim |
| markers | circles = ours, squares = deployed kernels | same |
| scale figure | `fig_scale`, three panels against model size | same styles, panels: MMLU gaps, b/param, tok/s |
| schematics | TikZ, `box`/`mem`/`flow`/`panel` styles | same styles, in `main.tex` |
| guard | `require_plotted` + `check_tables.py` | same |
| bibliography | ACM-Reference-Format | `plainnat`, numeric |

Data in `docs/data/`: `paper2-results.csv` (main table), `paper2-gaps.csv`
(paired MMLU gaps), `paper2-gsm8k.csv` and `paper2-gsm8k-gaps.csv` (GSM8K
scores and paired gaps, Table 4), `paper2-sizeup.csv` and `paper2-sizeup-budget.csv`
(one size up at the same memory, Table 5, and the memory budgets), `paper2-chain.csv` (the steps and the sealed files),
`tuile-l40s.csv` (tile sweep), and paper 1's `echelle-formats.csv` and
`echelle-4b-8b.csv`.

```bash
make                 # figures, then check, then latexmk
RELEASE=1 make check # refuses the build while a \pend cell is left
make arxiv           # arxiv-tetra.tar.gz: sources + figures + main.bbl
make clean
```

`\pdfoutput=1` must stay inside the first five lines of `main.tex`.

## State (2026-09-28)

Complete: 20 pages, no error, no overfull or underfull box, no undefined
reference, and `RELEASE=1 make check` passes (no pending cell). The served speeds
of the three sealed files come from paper-table-2026-09-25. GSM8K, scored through
the served kernel, comes from gsm8k-wave1-2026-09-26 and gsm8k-wave2-2026-09-26
(§6.3). The comparison one size up comes from sizeup-2026-09-28 (§6.4). Every
number and every cited source was audited on 2026-09-28
([paper2-audit-2026-09-28](../docs/mesures/paper2-audit-2026-09-28.txt)).

Writing rule for every revision: short factual sentences, no em dash.
