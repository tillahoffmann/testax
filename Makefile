.PHONY : all lint dist docs doctests tests

all : tests lint docs doctests dist

dist :
	python -m build
	twine check dist/*

docs :
	rm -rf docs/_build
	sphinx-build -nW . docs/_build

doctests:
	sphinx-build -nW -b doctest . docs/_build

tests :
	pytest

lint :
	black --check .
