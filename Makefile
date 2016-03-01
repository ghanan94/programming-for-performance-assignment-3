RM = rm -f

CC = gcc
CFLAGS = -O2 -lm

CXX = g++
CXXFLAGS = -std=c++0x -U__STRICT_ANSI__ -O2 -lOpenCL

default: all

all: bin nbody-seq nbody report

bin:
	mkdir bin

nbody-seq: src/nbody-seq.c
	$(CXX) $< $(CXXFLAGS) -o bin/nbody-seq

nbody: src/nbody.cpp
	$(CXX) $< $(CXXFLAGS) -o bin/nbody

report: report.pdf

report.pdf: report/report.tex
	cd report && pdflatex report.tex && pdflatex report.tex
	mv report/report.pdf report.pdf

clean:
	$(RM) nbody-seq nbody
	$(RM) report/*.aux report/*.log

.PHONY: all report clean
