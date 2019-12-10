# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = nucleus7
SOURCEDIR     = source
BUILDDIR      = build
DOCSDIR       = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ "$@" = "html" ]; then\
        # need to replace links to md files to generated html files \
        echo "Replace wrong links";\
        sed -i "s/\"[A-Za-z1-7_/.]*INSTALL.md\"/\"Installation.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*CONTRIBUTING.md\"/\"Contribution.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*Development.md\"/\"Development.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*ProjectConfigs.md\"/\"Project.html#configs\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*ProjectStructure.md\"/\"Project.html#structure\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*ProjectExecution.md\"/\"Project.html#execution\"/g" build/html/*.html ; \
        \
        sed -i "s/\"[A-Za-z1-7_/.]*data\/README.md\"/\"DataHandling.html#feeding-the-data\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*kpi\/README.md\"/\"KPI.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*coordinator\/README.md\"/\"Coordination.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*model\/README.md\"/\"Model.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*core\/README.md\"/\"Core.html\"/g" build/html/*.html ; \
        sed -i "s/\"[A-Za-z1-7_/.]*optimization\/README.md\"/\"Optimization.html\"/g" build/html/*.html ; \
        \
        echo "Copy requirements*.txt for working links"; \
        cp ../requirements*.txt build/html/ ;\
    fi && \
    cp -rT $(BUILDDIR)/html $(DOCSDIR) && \
	rm $(DOCSDIR)/.buildinfo
