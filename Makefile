# simple makefile 

build-docs:
	@echo "Building documentation"
	make -C docs/ clean
	make -C docs/ html-noplot
	cd docs/ && make view
