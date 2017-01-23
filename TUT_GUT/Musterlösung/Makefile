all: build/main.pdf

build/main.pdf: main.tex header.tex aufgabe1muster.tex aufgabe2muster.tex aufgabe3muster.tex| build
		latexmk \
			--lualatex
			--output-directory=build \
			--interaction=nonstopmode \
			--halt-on-error \
		main.tex
build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
