CFLAGS?=-Wall -O2 -lm

default: all

# modify to generate parallel and optimized versions!
all: build build/nbody-seq

report: report.pdf

build:
	mkdir build

build/nbody-seq: src/nbody-seq.c
	$(CC) $(CFLAGS) -o build/nbody-seq $<

report.pdf: report/report.tex
	cd report && pdflatex report.tex && pdflatex report.tex
	mv report/report.pdf report.pdf

clean:
	rm -r bin
