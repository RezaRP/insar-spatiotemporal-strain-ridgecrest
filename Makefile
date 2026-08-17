.PHONY: help env install test lint manifests figures clean

help:
	@echo "env        Create the conda environment"
	@echo "install    Install the package in editable mode with test extras"
	@echo "test       Run the unit test suite"
	@echo "lint       Run ruff over src, tests and scripts"
	@echo "manifests  Rebuild data manifests from data/"
	@echo "figures    Regenerate manuscript figures into results/figures"
	@echo "clean      Remove caches and build artefacts"

env:
	conda env create -f environment.yml

install:
	python -m pip install -e ".[analysis,notebooks,test]"

test:
	pytest -q

lint:
	ruff check src tests scripts

manifests:
	python scripts/build_manifests.py --data-dir data --output-dir data/manifests

figures:
	python scripts/make_manuscript_figures.py --output-dir results/figures

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
