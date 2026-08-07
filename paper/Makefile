# The paper rebuilds from the measurements: figures are generated from
# docs/data/*.csv, then latexmk compiles. `make` does both.
.PHONY: all figures pdf clean

all: figures pdf

figures:
	python3 scripts/make_figures.py

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex
	rm -rf figures
