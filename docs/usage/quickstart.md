# Quickstart

All commands run from the repository root with the virtual environment active.

## 1. Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
cp .env.example .env
```

## 2. Compare two files (works without a trained model)

```bash
codeplag compare data/samples/a.py data/samples/b.py
codeplag compare data/samples/a.py data/samples/c.py --json
codeplag compare data/samples/Original.java data/samples/Renamed.java --no-neural
```

Without `models/checkpoints/best.pt` the report uses AST features only and says so in its
warnings. With a checkpoint and `models/checkpoints/combiner.json` it uses the fitted
`combined` detector.

Example output (abridged):

```
codeplag report: data/samples/a.py  vs  data/samples/b.py
========================================================================
Verdict      : LIKELY PLAGIARISM  (score 0.99, threshold 0.6, detector combined)
AST score    : 1.000  ted=1.000  subtree=1.000  ngram=1.000
Neural score : 0.98

Largest matching regions (normalized structure):
  1. A lines 4-16  <->  B lines 2-14  (function_definition, 77 nodes)
```

## 3. Scan a directory

```bash
codeplag scan path/to/submissions --top 20
codeplag scan path/to/submissions --prefilter none     # exhaustive pairwise
codeplag scan path/to/submissions --json > scan.json
```

## 4. Reproduce training and evaluation

```bash
bash scripts/prepare_data.sh    # ~5 min: download 45 MB, clean, split, build pairs
bash scripts/train.sh           # CODEPLAG_DEVICE=cuda recommended; see below for CPU
bash scripts/evaluate.sh        # fit combiner on val, evaluate, benchmark, plot
```

CPU-only machines: `bash scripts/train.sh --max-pairs 4000 --epochs 2` keeps training
practical (fewer pairs; lower layers are already frozen).

Results land in `docs/results/` (`metrics.json`, `robustness.json`, `efficiency.json`, CSVs,
and `plots/`). `codeplag plots` regenerates the figures from those files.

## 5. Docker

```bash
docker compose build codeplag
docker compose run --rm codeplag compare data/samples/a.py data/samples/b.py
docker compose run --rm codeplag scan data/samples
```

`./data` and `./models` are mounted into the container, so a checkpoint trained on the host
is used automatically. The image is CPU-only.
