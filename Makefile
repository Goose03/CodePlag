.RECIPEPREFIX = >
.PHONY: install test lint format build run dev clean

install:
> python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

test:
> pytest

lint:
> ruff check src tests && mypy src

format:
> ruff format src tests

build:
> docker compose build codeplag

run:
> docker compose run --rm codeplag --help

dev:
> docker compose --profile dev run --rm dev

clean:
> rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage
