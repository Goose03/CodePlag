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

## Requirements

- Linux, macOS, or Windows
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

On Windows PowerShell, activate the environment with `. .\.venv\Scripts\Activate.ps1` after
creating it with `py -3.12 -m venv .venv`. Use `Copy-Item .env.example .env` for Docker Compose.

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

Run `codeplag --help` for all options. Without a local checkpoint, `compare` and `scan` use the
AST detector. After training and evaluation, they load the trained checkpoint and the fitted
combiner from `models/checkpoints/`.

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
codeplag prepare-data
codeplag train --device cuda
codeplag evaluate --device cpu --workers 4
```

For a shorter CPU experiment, use `codeplag train --device cpu --max-pairs 4000 --epochs 2`
and `codeplag evaluate --device cpu --workers 2`. A shorter training run produces its own
metrics and checkpoint. See the [quickstart](docs/usage/quickstart.md) for details.

For optional OpenVINO acceleration, install `pip install -e ".[openvino]"` and set
`CODEPLAG_OPENVINO_DEVICE=GPU` for a supported Intel GPU or `CPU` for CPU inference.
Training still uses PyTorch.

## Configuration

Defaults live in `configs/default.yaml`. Exported environment variables override them.
Docker Compose reads `.env` and passes its variables to the container.

| Variable               | Default                     | Description                      |
| ---------------------- | --------------------------- | -------------------------------- |
| `CODEPLAG_DEVICE`      | `cpu`                       | `cpu` or `cuda`                  |
| `CODEPLAG_MODEL_NAME`  | `microsoft/unixcoder-base`  | Hugging Face model id            |
| `CODEPLAG_LOG_LEVEL`   | `INFO`                      | Python logging level             |
| `CODEPLAG_OPENVINO_DEVICE` | unset                   | Optional FP32 encoder on `CPU` or Intel `GPU` |

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

## Evaluation

On 4,000 held-out Python pairs, the combined detector achieved F1 **0.9536**, recall
**95.65%**, and a **4.95%** false-positive rate at its validation-selected threshold.
Its false-positive rate rose to **77.6%** on the shared-template stress test.

See the [Spanish report](docs/report/informe.md) for the held-out comparison, robustness results,
and benchmark measurements. The [evaluation artifacts](docs/results/) include JSON, CSV,
and plots. Run `codeplag plots` to regenerate the figures.

## Limitations

- A high score is evidence for human review, not proof of plagiarism. Boilerplate, template code, and common textbook solutions can score high.
- Similarity between independently written but algorithmically obvious solutions can produce false positives.
- Languages other than Python, Java, and JavaScript are not supported.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), and make sure `ruff check`, `mypy src`, and `pytest` pass before opening a pull request. AI coding agents should read [AGENTS.md](AGENTS.md) first.

## License

See [LICENSE](LICENSE).
