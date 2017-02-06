all: build/main.pdf

build/main.pdf: main.tex | build
		lualatex --output-directory=build main.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all