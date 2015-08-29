PDFFILES=thesis.pdf
TEMPLATE=thesis.tex
BIB=references.bib
.PHONY: all clean
all:
	make $(PDFFILES) clean

# This target typesets the a PDF file.
$(PDFFILES): $(TEMPLATE) $(BIB)
	pdflatex -shell-escape $(basename $@).tex
	bibtex $(basename $@)
	pdflatex -shell-escape $(basename $@).tex
	pdflatex -shell-escape $(basename $@).tex

# This target removes any auxiliary files.
clean:
	rm -f *.aux *.log *.out *.toc *.lot *.lof *.bbl *.blg *.bcf *-blx.bib

# This target removes any auxiliary files
# and the output PDF files.
implode: clean
	rm -f $(PDFFILES)
