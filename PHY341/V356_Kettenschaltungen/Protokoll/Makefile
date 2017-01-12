all: build/main.pdf

build/main.pdf: versuchsaufbau.tex	theorie.tex header.tex main.tex diskussion.tex auswertung.tex lit.bib data.tex| build
		lualatex --output-directory=build main.tex
		biber build/main.bcf
		lualatex --output-directory=build main.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
