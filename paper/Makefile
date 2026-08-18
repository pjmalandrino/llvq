# The paper rebuilds from the measurements: figures are generated from
# docs/data/*.csv, the hand-typed tables are checked against the same CSVs,
# then latexmk compiles. `make` does all three, and `check` is not optional
# — a table that drifts from its CSV must fail the build, not the review.
.PHONY: all figures check pdf clean

all: figures check pdf

figures:
	python3 scripts/make_figures.py

check:
	python3 scripts/check_tables.py

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex
	rm -rf figures
