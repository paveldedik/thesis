TEMPLATE=thesis.tex
BIB=references.bib
.PHONY: all clean thesis printable
all:
	make thesis clean

# This target typesets the PDF file.
thesis: $(TEMPLATE) $(BIB)
	pdflatex -shell-escape $(basename $@).tex
	bibtex $(basename $@)
	pdflatex -shell-escape $(basename $@).tex
	pdflatex -shell-escape $(basename $@).tex

# This creates printable version of the thesis.
printable: thesis
	pdftk A=thesis.pdf B=description.pdf C=declaration.pdf cat A1-2 C1 A4 B1 A4-end output thesis-printable.pdf

# This target removes any auxiliary files.
clean:
	rm -f *.aux *.log *.out *.toc *.lot *.lof *.bbl *.blg *.bcf *-blx.bib


# This target removes any auxiliary files
# and the output PDF files.
implode: clean
	rm -f $(PDFFILES)
