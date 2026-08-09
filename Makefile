.PHONY: help lint format check-docs test docs clean

help:
	@echo "Available commands:"
	@echo "  make lint        - Run ruff + pydocstyle + darglint"
	@echo "  make format      - Format and autofix with ruff"
	@echo "  make check-docs  - Check docstring style (pydocstyle + darglint)"
	@echo "  make test        - Run the test suite"
	@echo "  make docs        - Build the Sphinx documentation"
	@echo "  make clean       - Remove caches and build artifacts"

lint:
	@echo "ruff (lint)..."
	ruff check src/micropurc tests
	@echo "pydocstyle (Google docstrings)..."
	pydocstyle src/micropurc
	@echo "darglint (docstring/signature consistency)..."
	darglint --docstring-style google --strictness long src/micropurc || true

format:
	ruff format src/micropurc tests
	ruff check --fix src/micropurc tests

check-docs:
	pydocstyle src/micropurc
	darglint --docstring-style google --strictness long src/micropurc || true

test:
	pytest tests/ -v

docs:
	cd docs && $(MAKE) html

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf docs/_build
