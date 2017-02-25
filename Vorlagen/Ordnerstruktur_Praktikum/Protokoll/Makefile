all: build/main.pdf

debug: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib  precom test| build
	 lualatex --output-directory=build --draft -shell-escape "&./build/header main.tex"

build/main.pdf: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib  precom | build
			lualatex --output-directory=build  -shell-escape "&./build/header main.tex"

full: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib  precom| build
		lualatex --output-directory=build  -shell-escape "&./build/header main.tex"
		biber build/main.bcf
		lualatex --output-directory=build  -shell-escape "&./build/header main.tex"

build/header.fmt: ./header.tex ./data.tex
		@cd build; lualatex -ini -job-name="main" "&lualatex ../header.tex\dump"
		@echo Preamble ist gebaut und ready

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
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
