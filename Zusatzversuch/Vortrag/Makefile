all: build/presentation.pdf

build/presentation.pdf: presentation.tex header.tex vierpunkt.tex theorie.tex durchfuehrung.tex versuchsziel.tex| build
		lualatex --output-directory=build presentation.tex

full: presentation.tex header.tex theorie.tex versuchsziel.tex| build
		lualatex --output-directory=build presentation.tex
		biber build/presentation.bcf
		lualatex --output-directory=build presentation.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
