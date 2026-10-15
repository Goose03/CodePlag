# Architecture overview

codeplag scores how likely it is that two source files are plagiarized, using two independent
detectors and a calibrated combiner.

## Pipeline

```
                       ┌──────────────────────────── ast_analysis ─────────────────────────────┐
 file A ──┐            │                                                                         │
          ├─► parser ──┼─► normalizer ──► tree_distance ──► {ted, subtree, ngram} ─────────────┐ │
 file B ──┘ (tree-sitter)  (comments,      (Zhang-Shasha TED,                                  │ │
                       │    identifiers,    subtree hashes,                                    │ │
                       │    literals,       label n-grams,                                     │ │
                       │    loops ...)      matching regions)                                  │ │
                       └───────────────────────────────────────────────────────────────────────┼─┘
                                                                                               │
          ┌──────────────────────────────── neural ───────────────────────────────┐            │
          │ tokenizer ─► UniXcoder (shared) ─► mean pool ─► projection ─► u, v ─┐ │            │
          │                                      MLP on [u, v, |u-v|, u*v] ◄───┘ │─► neural ──┤
          └───────────────────────────────────────────────────────────────────────┘            │
                                                                                               ▼
                                       similarity/combiner: logistic regression (fit on val)
                                                         │  threshold chosen for FPR ≤ 5% on val
                                                         ▼
                                       similarity/report: score, verdict, per-detector scores,
                                                          matching line ranges, warnings
```

`scan DIR` adds a candidate prefilter in front of the pairwise comparison:
MinHash-LSH over normalized-tree label shingles (and, optionally, embedding nearest neighbors),
then compares only the candidate pairs, in parallel worker processes.

## Modules

| Module | Responsibility |
| --- | --- |
| `ast_analysis/parser.py` | Extension-based language detection; tree-sitter parsing for Python, Java, JavaScript; counts `ERROR`/`MISSING` nodes, optional strict mode. |
| `ast_analysis/normalizer.py` | Converts the syntax tree into a compact `NormNode` tree; every rule is documented in the module docstring. |
| `ast_analysis/tree_distance.py` | Tree edit distance similarity with a size guard, subtree-hash overlap, label n-gram overlap, matching regions. |
| `data/codenet.py` | Reads CodeNet archives directly from `.tar.gz`, cleans files, splits problems. |
| `data/transforms.py` | Seeded, behaviour-preserving plagiarism transformations. |
| `data/pairs.py` | Pair generation (balanced splits, hard suites, unseen language) and `prepare-data`. |
| `neural/model.py` | Siamese network: shared encoder, mean pooling, projection, symmetric MLP or cosine head. |
| `neural/dataset.py` | `PairDataset` and a batch tokenizing collator. |
| `neural/train.py` | BCE training with AdamW, warmup, AMP on CUDA, early stopping on validation ROC-AUC. |
| `neural/infer.py` | Checkpoint loading and embedding cache (memory and optional disk). |
| `similarity/combiner.py` | Logistic or weighted fusion; FPR-constrained threshold; JSON persistence. |
| `similarity/report.py` | Text and JSON reports. |
| `similarity/prefilter.py` | MinHash signatures, LSH banding, embedding neighbors. |
| `similarity/detector.py` | `PlagiarismDetector`: `compare`, `compare_files`, `scan`. Falls back to AST-only when no checkpoint exists. |
| `evaluation/` | Feature extraction, metrics, robustness suites, efficiency benchmarks, plots. |
| `cli/main.py` | Typer commands; no business logic. |

## Design decisions

* **Normalization before comparison.** Renaming identifiers, editing comments or reformatting
  does not change the normalized tree, so the AST detector is insensitive to the cheapest
  forms of plagiarism by construction.
* **Three structural features.** Tree edit distance captures global shape but is
  order-sensitive; subtree-hash overlap survives function and statement reordering; label
  n-grams are a soft, linear-time signal. The combiner learns how much to trust each one.
* **Fast Zhang-Shasha.** `zss` 1.2 copies an edit-operation list into every dynamic-programming
  cell even when the operations are not requested, which made one comparison take seconds.
  `tree_edit_distance` reuses `zss.AnnotatedTree` and the same recurrence without that
  bookkeeping; a unit test checks it returns exactly the same distance as
  `zss.simple_distance`.
* **Size guard.** Trees above `ast.max_nodes` (default 250 normalized nodes, above the 90th
  percentile of the test files) are depth-pruned before tree edit distance and the result is
  flagged approximate. On 100 test pairs this changed the TED similarity by 0.005 on average
  (0.097 at most) while cutting the mean time per pair by about a quarter.
* **Neural model.** UniXcoder (`microsoft/unixcoder-base`, RoBERTa architecture, 125M
  parameters) is pretrained on code in several languages. The embeddings and lower 6 of 12
  layers are frozen so training fits a laptop GPU (or a CPU with fewer pairs); the head is
  symmetric so `score(a, b) == score(b, a)`.
* **Calibrated fusion with a conservative threshold.** A false accusation is costlier than a
  missed case, so the decision threshold is the smallest one whose false-positive rate on
  the validation split is at most `combiner.max_fpr` (5%).
* **No pickles.** Combiners are stored as JSON; checkpoints are PyTorch state dicts.

## Dataset

See `docs/report/informe.md` (Spanish) for the full justification. Summary:

* **Primary:** IBM Project CodeNet, benchmark subset **Python800** (800 problems × 300 accepted
  Python solutions = 240,000 files). Source: <https://github.com/IBM/Project_CodeNet>; the
  original IBM download links are offline, so `prepare-data` downloads the archives from the
  Hugging Face mirror
  [`qiankunmu/Project_CodeNet_Python800_and_Java250`](https://huggingface.co/datasets/qiankunmu/Project_CodeNet_Python800_and_Java250).
  The CodeNet repository is released under Apache-2.0; the mirror declares Apache-2.0.
* **Unseen language:** CodeNet **Java250** (250 problems × 300 Java solutions = 75,000 files).
* **Fields per pair** (`data/processed/*.jsonl`): `id`, `split`, `label` (1 = plagiarism),
  `kind` (`same_problem`, `synthetic`, `different_problem`, `hard_*`), `lang_a`, `lang_b`,
  `file_a`, `file_b` (`<problem>/<submission>`), `code_a`, `code_b`, `transforms`.
* **Cleaning** (Python800): 240,000 files → 223,803 kept (11,677 with fewer than 3 non-blank
  lines, 4,051 not parseable as Python 3, 393 exact duplicates and 76 larger than 6,000 bytes
  removed); 40 files sampled per problem.
* **Splits by problem** (no file or problem is shared across splits): 560 / 120 / 120 problems;
  24,000 / 4,000 / 4,000 pairs, each split 50% positive (half same-problem, half synthetic)
  and 50% negative.
