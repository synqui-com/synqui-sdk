.PHONY: help install install-dev test test-cov lint format type-check clean build upload upload-test

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install .

install-dev:  ## Install development dependencies
	pip install -e ".[dev,monitoring]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=cognitionflow --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 cognitionflow tests
	isort --check-only cognitionflow tests
	black --check cognitionflow tests

format:  ## Format code
	isort cognitionflow tests
	black cognitionflow tests

type-check:  ## Run type checking
	mypy cognitionflow

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

upload-test:  ## Upload to test PyPI
	python -m twine upload --repository testpypi dist/*

upload:  ## Upload to PyPI
	python -m twine upload dist/*