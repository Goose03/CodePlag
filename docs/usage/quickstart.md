# Quickstart

All commands run from the repository root with the virtual environment active. On Windows,
activate it in PowerShell with `. .\.venv\Scripts\Activate.ps1`.

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

Output for the sample pair with the reference combiner (abridged).

```
codeplag report: data/samples/a.py  vs  data/samples/b.py
========================================================================
Verdict      : LIKELY PLAGIARISM  (score 1.000, threshold 0.177, detector combined)
AST score    : 1.000  ted=1.000  subtree=1.000  ngram=1.000
Neural score : 1.000

Largest matching regions (normalized structure):
  1. A lines 1-26  <->  B lines 1-23  (module, 108 nodes)
```

## 3. Scan a directory

```bash
codeplag scan path/to/submissions --top 20
codeplag scan path/to/submissions --prefilter none     # exhaustive pairwise
codeplag scan path/to/submissions --json > scan.json
```

## 4. Reproduce training and evaluation

```bash
codeplag prepare-data
codeplag train --device cuda
codeplag evaluate --device cpu --workers 4
```

To train on CPU, use `--device cpu` for training too. For a shorter run add
`--max-pairs 4000 --epochs 2` to the training command. The reference run used all
24,000 training pairs and four epochs. Both commands require local access to Hugging
Face model files. `evaluate --skip-efficiency` runs the accuracy and robustness
evaluation without the timing benchmarks. To accelerate FP32 inference with OpenVINO,
install `pip install -e ".[openvino]"`. Set `CODEPLAG_OPENVINO_DEVICE=GPU` for a supported
Intel GPU or `CPU` for the CPU runtime. The reference evaluation used the Intel GPU, with
`OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` for the PyTorch scoring head.

Results land in `docs/results/` (`metrics.json`, `robustness.json`, `efficiency.json`, CSVs,
and `plots/`). `codeplag plots` regenerates the figures from those files. The efficiency suite
compares exhaustive and MinHash scans at 24 and 48 files and uses MinHash at 96 and 192.

## 5. Docker

```bash
cp .env.example .env   # PowerShell: Copy-Item .env.example .env
docker compose build codeplag
docker compose run --rm codeplag compare data/samples/a.py data/samples/b.py --no-neural
docker compose run --rm codeplag scan data/samples
```

`./data` and `./models` are mounted into the container, so a checkpoint trained on the host
is used automatically. The image is CPU-only.
