all: build/main.pdf

build/main.pdf: main.tex header.tex aufgabe1muster.tex aufgabe2muster.tex aufgabe3muster.tex| build
		lualatex --output-directory=build main.tex
build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
