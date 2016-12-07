all: build/main.pdf

build/main.pdf: main.tex header.tex kurzfragen.tex aufgabe1.tex aufgabe2.tex| build
		lualatex --output-directory=build main.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
