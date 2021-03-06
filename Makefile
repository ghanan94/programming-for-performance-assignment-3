RM = rm -f

CC = gcc
CFLAGS = -O2 -lm

CXX = g++
CXXFLAGS = -std=c++0x -U__STRICT_ANSI__ -O2 -lOpenCL

default: all

all: bin nbody-seq nbody-opt-seq nbody nbody-opt report

bin:
	mkdir bin

nbody-opt-seq: src/nbody-opt-seq.c
	$(CXX) $< $(CXXFLAGS) -o bin/nbody-opt-seq

nbody-seq: src/nbody-seq.c
	$(CXX) $< $(CXXFLAGS) -o bin/nbody-seq

nbody: src/nbody.cpp
	$(CXX) $< $(CXXFLAGS) -o bin/nbody

nbody-opt: src/nbody-opt.cpp
	$(CXX) $< $(CXXFLAGS) -o bin/nbody-opt

report: report.pdf

report.pdf: report/report.tex
	cd report && pdflatex report.tex && pdflatex report.tex
	mv report/report.pdf report.pdf

clean:
	$(RM) bin/nbody bin/nbody-seq bin/nbody-opt bin/nbody-opt-seq
	$(RM) report/*.aux report/*.log

.PHONY: all report clean
