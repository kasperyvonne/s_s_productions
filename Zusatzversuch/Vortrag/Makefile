all: build/presentation.pdf

build/presentation.pdf: presentation.tex | build
		lualatex --output-directory=build presentation.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
