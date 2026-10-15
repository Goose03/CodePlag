# AGENTS.md

Instructions for AI coding agents working in this repository. Read this first.

## Project summary

**codeplag** detects likely code plagiarism between two source files by combining:

1. **Semantic/AST analysis**: parse code with `tree-sitter`, normalize the tree (strip comments, rename identifiers, canonicalize literals), then compare trees with tree edit distance (`zss`).
2. **Neural similarity**: a siamese model on top of a pretrained code encoder (default `microsoft/unixcoder-base`) that compares code embeddings.
3. **Combiner**: merges both scores into one similarity verdict plus a human-readable report.

Entry point: `codeplag` CLI (Typer), defined at `codeplag.cli.main:app`.

> **Status:** the repo is scaffolded. Many modules are empty placeholders. Check whether a file has content before assuming it is implemented, and fill placeholders rather than creating parallel files.

## Environment

- Python **3.12** (matches the container). Do not use syntax or deps that require a different version.
- Virtualenv lives in `.venv/` at the repo root. Never commit it.
- Package uses the **src layout**: all importable code is under `src/codeplag/`.

```bash
# Activate the venv (from repo root)
source .venv/bin/activate

# Create it if missing (Arch: `uv` provides Python 3.12)
uv venv --python 3.12 .venv && source .venv/bin/activate
python -m ensurepip --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Always run commands from the repo root with the venv active.

## Repository layout

```
.
├── AGENTS.md                  # this file
├── README.md                  # human-facing overview
├── pyproject.toml             # deps, entry points, ruff/mypy/pytest config (single source of truth)
├── requirements.lock.txt      # pip freeze snapshot; regenerate after dependency changes
├── Makefile                   # shortcuts: install, test, lint, format, build, run, dev, clean
├── docker-compose.yml         # services: `codeplag` (runtime), `dev` (profile: dev)
├── .env.example / .env        # runtime settings (.env is git-ignored)
├── configs/
│   └── default.yaml           # default runtime configuration
├── docker/
│   ├── Dockerfile             # multi-stage, non-root runtime image
│   └── Dockerfile.dev         # dev image with dev extras
├── src/codeplag/
│   ├── __init__.py            # defines __version__
│   ├── core/                  # config loading (config.py), logging setup (logging.py)
│   ├── ast_analysis/          # parser.py, normalizer.py, tree_distance.py
│   ├── neural/                # model.py, dataset.py, train.py, infer.py
│   ├── similarity/            # combiner.py (score fusion), report.py (output formatting)
│   ├── data/                  # dataset loading / preprocessing helpers
│   └── cli/
│       └── main.py            # Typer `app`; thin layer only
├── tests/
│   ├── unit/                  # fast, isolated tests (test_ast.py, test_neural.py)
│   ├── integration/           # end-to-end pipeline tests (test_pipeline.py)
│   └── fixtures/              # small sample code pairs used by tests
├── data/
│   ├── raw/                   # untouched source datasets (git-ignored)
│   ├── processed/             # derived datasets (git-ignored)
│   └── samples/               # small example inputs safe to commit
├── models/checkpoints/        # trained weights (git-ignored)
├── scripts/                   # train.sh, evaluate.sh: thin wrappers around CLI/modules
├── docs/                      # architecture/, usage/, api/
├── notebooks/                 # exploration only; no production logic
└── .github/workflows/         # CI
```

## Where things go

| If you are adding...                              | Put it in                                         |
| ------------------------------------------------- | ------------------------------------------------- |
| Code parsing, AST normalization, tree comparison  | `src/codeplag/ast_analysis/`                      |
| Model definition, training loop, inference        | `src/codeplag/neural/`                            |
| Score fusion, thresholds, report rendering        | `src/codeplag/similarity/`                        |
| Settings, logging, shared utilities               | `src/codeplag/core/`                              |
| Dataset loading / preprocessing                   | `src/codeplag/data/` (and `neural/dataset.py` for torch `Dataset` classes) |
| New CLI command                                   | `src/codeplag/cli/main.py` (logic stays in library modules) |
| Tunable parameters                                | `configs/default.yaml`, surfaced via `core/config.py` |
| Test for module `X`                               | `tests/unit/test_<x>.py`                          |
| Multi-module behavior test                        | `tests/integration/`                              |
| Sample code files for tests                       | `tests/fixtures/`                                 |
| Docs                                              | `docs/architecture/`, `docs/usage/`, `docs/api/`  |
| New dependency                                    | `pyproject.toml` (then refresh `requirements.lock.txt`) |

## How to run things

```bash
# CLI (venv)
codeplag --help
codeplag version

# Tests (coverage is enabled by default via pyproject addopts)
pytest
pytest tests/unit/test_ast.py -k name_of_test

# Lint, format, type-check
ruff check src tests
ruff format src tests
mypy src

# Pre-commit on everything
pre-commit run --all-files

# Make shortcuts
make test | make lint | make format | make build | make run | make dev

# Docker (runtime image)
docker compose build codeplag
docker compose run --rm codeplag --help
docker compose run --rm codeplag version

# Docker (dev shell with source mounted)
docker compose --profile dev run --rm dev
```

Training and evaluation entry points live in `scripts/train.sh` and `scripts/evaluate.sh`; they should call into `codeplag.neural.train` and the CLI rather than contain logic.

## Configuration

- Environment variables (see `.env.example`): `CODEPLAG_DEVICE` (`cpu`/`cuda`), `CODEPLAG_MODEL_NAME` (HF model id), `CODEPLAG_LOG_LEVEL`.
- YAML defaults in `configs/default.yaml`. Environment variables override YAML.
- Never hardcode paths, model names, or thresholds in modules; read them from config.

## Conventions

- **Imports:** absolute, rooted at `codeplag` (e.g., `from codeplag.ast_analysis.parser import parse_source`). No `sys.path` hacks.
- **Style:** ruff, line length 100, rules `E,F,I,B,UP`. Type hints on all public functions; code must pass `mypy src`.
- **Docstrings:** Google style on public functions and classes.
- **Logging:** use the logger from `codeplag.core.logging`; no bare `print` in library code (CLI output goes through Typer's `typer.echo`).
- **Separation:** `cli/` only parses arguments and formats output. All logic lives in library modules so it is testable.
- **Languages:** parsing supports Python, Java, and JavaScript via tree-sitter. Add new languages by adding the `tree-sitter-<lang>` dependency and a language entry in `ast_analysis/parser.py`.
- **Determinism:** seed anything stochastic (`torch`, `numpy`, `random`) from config so tests are reproducible.
- **Scores:** every similarity score is a float in `[0.0, 1.0]`, where higher means more similar.

## Rules for agents

**Do**
- Run `ruff check`, `mypy src`, and `pytest` before declaring a task done.
- Add or update tests alongside every behavior change.
- Keep changes minimal and within the module that owns the concern.
- Update `docs/` and `CHANGELOG.md` when behavior or CLI surface changes.
- Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).

**Do not**
- Commit anything in `.venv/`, `.env`, `data/raw/`, `data/processed/`, or `models/checkpoints/`.
- Add large files (>1 MB) to git. Use `data/samples/` only for tiny fixtures.
- Put business logic in notebooks, scripts, or `cli/main.py`.
- Edit `requirements.lock.txt` by hand: regenerate it with `pip freeze > requirements.lock.txt`.
- Change the Python version, package name, or the `codeplag` entry point without being asked.
- Download models or datasets inside tests; use small fixtures or mocks.

## Planned pipeline (for orientation)

```
file A ─┐                                          ┌─> AST score ──┐
        ├─> parser ─> normalizer ─> tree_distance ─┤               ├─> combiner ─> report
file B ─┘                                          │               │
        └──────────> neural/infer (embeddings) ────┴─> NN score ───┘
```

Implementation order: `ast_analysis` first, then `neural`, then `similarity/combiner`, then wire the CLI.
