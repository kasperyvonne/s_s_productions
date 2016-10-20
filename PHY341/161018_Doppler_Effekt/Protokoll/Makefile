all: build/main.pdf

build/main.pdf: versuchsaufbau.tex	theorie.tex header_1.tex main.tex | build
		lualatex --output-directory=build main.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
