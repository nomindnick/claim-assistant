# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `poetry install`
- Run CLI: `python -m claimctl.cli` or `./run.sh`

## Test Commands
- All tests: `pytest`
- Single test: `pytest tests/test_retrieval.py::test_recall`
- Verbose: `pytest -v`

## Lint/Format Commands
- Format code: `black .`
- Check format: `black --check .`
- Sort imports: `isort .`
- Type check: `mypy .`
- Lint: `ruff check .`

## Code Style
- Formatting: Black with 88 character line length
- Imports: Use isort with standard grouping (stdlib, third-party, local)
- Types: Strict typing with Pydantic for data validation
- Naming: snake_case for functions/variables, PascalCase for classes
- Docstrings: Triple-quotes for all modules, classes, and functions
- Error handling: Use specific exceptions with meaningful error messages
- CLI: Built with Typer and Rich for formatted output