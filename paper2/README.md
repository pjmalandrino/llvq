# Paper 2 — Tetra

`main.tex` is the paper: *Reading a 24-Dimensional Lattice Code Without
Unfolding It*. Five sections — problem, construction, kernel, results,
limitations — then three appendices: the ten-arm benchmark, the record of
trials and commands, the provenance table.

## Same build as paper 1, deliberately

| | paper 1 | paper 2 |
|---|---|---|
| class | `acmsmall`, `nonacm=true` | same |
| figures with numbers | `scripts/make_figures.py`, matplotlib → PDF | same script conventions |
| palette | Okabe-Ito, `INK`/`GRAY`/`LIGHT` | same constants, Tetra takes `PURPLE` |
| rcParams | serif 8 pt, grey axes, no top/right spine, `#DDDDDD` grid | copied verbatim |
| markers | circles = ours, squares = deployed kernels | same |
| schematics | TikZ, `box`/`mem`/`flow`/`panel` styles | same styles, in `main.tex` |
| guard | `require_plotted` + `check_tables.py` | same |
| data | `docs/data/*.csv` | same directory, three new CSVs |

The three new CSVs are `tuile-l40s.csv`, `tetra-rowscales.csv` and
`tetra-formats.csv`; the kernel figures read paper 1's `echelle-formats.csv`,
which already carries the `Tetra48` row.

```bash
make            # figures, then check, then latexmk
make arxiv      # arxiv-tetra.tar.gz: sources + figures + main.bbl
make clean
```

`\pdfoutput=1` must stay inside the first five lines of `main.tex`.

## State

**Not compiled.** The session that wrote this revision could not run commands
(a safety classifier blocked Bash for the whole session), so neither
`make_figures.py` nor `check_tables.py` nor `latexmk` has been executed against
the current source. `main.pdf` and `arxiv-tetra.tar.gz` in this directory are
an older two-column build and do not match it. First thing to do:

```bash
cd paper2 && make
```

then read the figure sizes at `acmsmall`'s 5.5 in text width, and the page
count: the target is 5–6 pages before references and appendices.

Leftovers to delete: `.write-test`, and every file in `sections/` whose first
line starts with `% Superseded` (they are not input by `main.tex`).

## Open before submitting

1. **The same-support kernel timing.** Table 1 deliberately orders bytes per
   weight and not milliseconds, because the Tetra arm covers 216 of the 252
   matrices. Running AWQ over the same 216, or Tetra over the 252-record
   pure-lattice file, is one job and closes it.
2. **Three bibliography entries are unverified** against a publisher record:
   `forney1988`, `vardy1991`, `dclmedu`. The comment block above them in
   `refs.bib` says so.
3. **`llvq1preprint`** points at the Zenodo DOI; add paper 1's arXiv
   identifier if it is known.
