# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
#SPHINXBUILD   = "python3 $(which sphinx-build)" #sphinx-build
SPHINXBUILD   = python3 -m sphinx
SPHINXPROJ    = pyActigraphy
SOURCEDIR     = source
BUILDDIR      = ../../pyActigraphy-docs
PDFBUILDDIR   = /tmp
PDF           = ../manual.pdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

latexpdf:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) $(PDFBUILDDIR)/latex
	make -C $(PDFBUILDDIR)/latex all-pdf
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)
	@echo "pdflatex finished; see $(PDF)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
