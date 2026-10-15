# codeplag

Detects likely plagiarism between two source code files by combining a **semantic tree (AST) analyzer** with a **neural similarity model**.

Supported languages: Python, Java, JavaScript.

## How it works

```
file A ─┐                                          ┌─> AST score ──┐
        ├─> parser ─> normalizer ─> tree_distance ─┤               ├─> combiner ─> report
file B ─┘                                          │               │
        └──────────> neural/infer (embeddings) ────┴─> NN score ───┘
```

1. **AST analyzer:** parses both files with `tree-sitter`, normalizes the trees (strips comments, renames identifiers, canonicalizes literals), and compares them with tree edit distance (`zss`).
2. **Neural model:** a siamese network on a pretrained code encoder (`microsoft/unixcoder-base`) that scores the similarity of two code embeddings.
3. **Combiner:** merges both scores into a final value in `[0.0, 1.0]` (higher means more similar) and produces a report with the matching regions.

Normalizing identifiers and comments lets the AST analyzer catch renaming and formatting changes, while the neural model catches rewrites that keep the same logic.

## Status

Early development. The repository structure, tooling, and containers are in place; modules are being implemented in this order: `ast_analysis` → `neural` → `similarity` → CLI. Check the [CHANGELOG](CHANGELOG.md) for what is currently usable.

## Requirements

- Linux or macOS
- Python **3.12**
- Git
- Docker with the Compose plugin (optional, for containerized runs)

## Installation

```bash
git clone https://github.com/Goose03/CodePlag codeplag
cd codeplag

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch (smaller). For GPU, install the CUDA build from pytorch.org instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
pre-commit install
cp .env.example .env
```

<details>
<summary>Arch Linux: getting Python 3.12</summary>

Arch ships a newer default Python, and some wheels lag behind it. Use `uv` to get 3.12:

```bash
sudo pacman -S --needed uv
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m ensurepip --upgrade
```

Then continue with the `pip install` lines above.
</details>

## Usage

```bash
# Compare two files
codeplag compare path/to/a.py path/to/b.py

# Machine-readable output
codeplag compare a.py b.py --json

# Scan a directory (pairwise)
codeplag scan path/to/submissions/

# Train and evaluate the neural model
codeplag train
codeplag evaluate

# Show version
codeplag version
```

Run `codeplag --help` for all options. The `compare`, `scan`, `train`, and `evaluate` commands are planned and land as their modules are completed; `version` works today.

## Running and testing

Run everything from the repo root with the virtual environment active (`source .venv/bin/activate`).

**1. Verify the install**

```bash
python -c "import torch, tree_sitter, transformers; print('OK', torch.__version__)"
codeplag --help
```

**2. Run the tests**

```bash
pytest                                   # all tests, with coverage
pytest tests/unit                        # fast unit tests only
pytest tests/integration                 # end-to-end pipeline tests
pytest tests/unit/test_ast.py -k rename  # a single test by name
```

Tests run offline: they use the fixtures in `tests/fixtures/` and never download models.

**3. Lint, format, type-check**

```bash
ruff check src tests
ruff format src tests
mypy src
pre-commit run --all-files
```

`make test`, `make lint`, and `make format` run the same commands.

**4. Run in Docker**

```bash
docker compose build codeplag
docker compose run --rm codeplag --help
docker compose run --rm codeplag version

# Compare files from the mounted ./data directory
docker compose run --rm codeplag compare data/samples/a.py data/samples/b.py

# Development shell with the source mounted
docker compose --profile dev run --rm dev
```

If Docker reports it cannot connect to `~/.docker/desktop/docker.sock`, run `docker context use default` and make sure the daemon is running (`sudo systemctl enable --now docker`).

**5. Prepare data and train**

```bash
bash scripts/prepare_data.sh   # downloads and processes the dataset into data/processed/
bash scripts/train.sh          # trains the neural model, saves checkpoints to models/checkpoints/
bash scripts/evaluate.sh       # evaluates on the held-out test split, writes to docs/results/
```

Training is configured to be feasible on CPU. Set `CODEPLAG_DEVICE=cuda` in `.env` to use a GPU.

## Configuration

Defaults live in `configs/default.yaml`. Environment variables (from `.env`) override them.

| Variable               | Default                     | Description                      |
| ---------------------- | --------------------------- | -------------------------------- |
| `CODEPLAG_DEVICE`      | `cpu`                       | `cpu` or `cuda`                  |
| `CODEPLAG_MODEL_NAME`  | `microsoft/unixcoder-base`  | Hugging Face model id            |
| `CODEPLAG_LOG_LEVEL`   | `INFO`                      | Python logging level             |

## Project structure

```
src/codeplag/
├── ast_analysis/   parser, normalizer, tree edit distance
├── neural/         siamese model, dataset, training, inference
├── similarity/     score combiner and report generation
├── core/           config and logging
├── data/           dataset loading and pair generation
└── cli/            Typer command-line interface
tests/              unit/, integration/, fixtures/
configs/            default.yaml
docker/             Dockerfile (runtime), Dockerfile.dev
docs/               architecture, usage, results, report
data/               raw/ and processed/ (git-ignored), samples/ (committed)
models/checkpoints/ trained weights (git-ignored)
scripts/            data prep, train, and evaluate wrappers
```

## Dataset

Training and evaluation use labeled code pairs: positives are solutions to the same problem or synthetically plagiarized variants (renaming, reordering, dead-code insertion); negatives are solutions to different problems. Splits are made by problem to avoid leakage. Dataset source, size, and license are documented in [docs/architecture/overview.md](docs/architecture/overview.md).

## Limitations

- A high score is evidence for human review, not proof of plagiarism. Boilerplate, template code, and common textbook solutions can score high.
- Similarity between independently written but algorithmically obvious solutions can produce false positives.
- Languages other than Python, Java, and JavaScript are not supported.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), and make sure `ruff check`, `mypy src`, and `pytest` pass before opening a pull request. AI coding agents should read [AGENTS.md](AGENTS.md) first.

## License

See [LICENSE](LICENSE).
