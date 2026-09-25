# The paper rebuilds from the measurements, like paper 1: figures are
# generated from docs/data/*.csv, the hand-typed tables are checked against
# the same CSVs, then latexmk compiles. `make` does all three, and `check` is
# not optional — a table that drifts from its CSV must fail the build, not
# the review.
.PHONY: all figures check pdf arxiv clean

all: figures check pdf

figures:
	python3 scripts/make_figures.py

check:
	python3 scripts/check_tables.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# The tarball arXiv wants: sources, the generated figures, the compiled
# bibliography, no auxiliary files. arXiv runs latex itself but not bibtex,
# and not our figure script.
arxiv: all
	@rm -rf arxiv && mkdir -p arxiv/sections arxiv/figures
	cp main.tex arxiv/
	cp sections/*.tex arxiv/sections/
	cp figures/*.pdf arxiv/figures/
	cp main.bbl arxiv/
	cd arxiv && tar czf ../arxiv-tetra.tar.gz .
	@echo "wrote arxiv-tetra.tar.gz"

clean:
	latexmk -C main.tex
	rm -rf arxiv arxiv-tetra.tar.gz figures
