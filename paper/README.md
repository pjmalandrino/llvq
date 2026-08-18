# The paper

An arXiv-style systems paper: *Decoding the Leech Lattice at Matvec Speed:
Fused Kernels and VRAM Layouts for 2-Bit LLM Inference*.

## Build

```bash
make          # figures from ../docs/data/*.csv, then latexmk → main.pdf
make figures  # regenerate figures only
make clean    # remove build artifacts and generated figures
```

Requires MacTeX (`latexmk`, `pdflatex`) and Python 3 with matplotlib —
both already on the dev machine.

## The contract

The paper rebuilds from the measurements, like everything else in the repo:

- **No number in a figure is typed by hand.** `scripts/make_figures.py`
  reads `docs/data/*.csv` — the same CSVs behind
  `docs/archive/publication-2026-08-07.md` — and every cell there traces to a
  dated, costed GPU job (`docs/data/jobs.csv`).
- **Every claim in the text carries its provenance** either inline
  (`docs/mesures/...` paths in table captions) or through the publication
  dossier. When editing a section, check the claim against the measurement
  file before changing a number.
- Reporting rules that must survive any edit (from the dossier's §7):
  AWQ speed is never compared *across stacks* — since 2026-08-17 the two
  within-stack ratios (×2.413 in vLLM, ×1.12 in ours) may sit side by side
  but never divide; the ×2.03 always carries its double formulation (~×1.4
  vs the corrected engine); speedups are medians of per-round ratios with
  ranges, never a third decimal; memory comparisons are whole-model b/param,
  embedding included.

## Layout

```
main.tex            preamble, title, abstract
sections/*.tex      one file per section
refs.bib            bibliography (provenance header inside: every field
                    checked 2026-08-08, six entries were wrong, two soft
                    spots named rather than hidden)
scripts/make_figures.py
figures/            generated — not committed
```
