# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build
APIDOCDIR     = ../disk_analyzer
APIDOCOUT     = .

# Помощь
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help clean Makefile apidoc

apidoc:
	@echo "Generating API documentation with sphinx-apidoc..."
	@$(SPHINXAPIDOC) -o $(APIDOCOUT) $(APIDOCDIR) --force

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)

%: Makefile
	@$(MAKE) apidoc
	@echo "Building documentation target: $@"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
