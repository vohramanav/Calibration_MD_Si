#!/bin/sh

pdflatex md_si.tex
pdflatex md_si.tex
rm *.aux *.log *.out *.toc *.nav *.snm
