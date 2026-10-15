# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `ast_analysis`: tree-sitter parsing (Python, Java, JavaScript) with syntax-error accounting;
  rename-invariant normalization (comments, docstrings, identifiers, literals, loops,
  compound assignments, parentheses); Zhang-Shasha tree edit distance with a size guard,
  subtree-hash overlap, label n-gram overlap and matching-region extraction.
- `data`: Project CodeNet Python800/Java250 loader that reads the archives without
  extracting them, cleaning, problem-level splits, six seeded behaviour-preserving
  plagiarism transformations for Python (three for Java), balanced pair generation, hard
  suites and an unseen-language set. `codeplag prepare-data`.
- `neural`: siamese network over `microsoft/unixcoder-base` with mean pooling, projection and
  a symmetric MLP (or cosine) head; BCE training with early stopping, AMP and checkpoints;
  inference with an embedding cache. `codeplag train`.
- `similarity`: logistic combiner fitted on validation with an FPR-constrained threshold,
  JSON persistence, text/JSON reports, MinHash-LSH and embedding prefilters,
  `PlagiarismDetector` with AST-only fallback, parallel `scan`.
- `evaluation`: test metrics for AST-only, neural-only and combined detectors, threshold
  analysis, per-transformation and per-kind results, hard suites, unseen-language set,
  latency/memory/throughput benchmarks and plots. `codeplag evaluate`, `codeplag plots`.
- CLI: `compare`, `scan`, `prepare-data`, `train`, `evaluate`, `plots`, `version`.
- Scripts `scripts/prepare_data.sh`, `scripts/train.sh`, `scripts/evaluate.sh`.
- Documentation: architecture overview, quickstart, Spanish report `docs/report/informe.md`.
- Tests: unit tests for every module (mocked encoder, no downloads) and an end-to-end
  integration test; fixtures in `tests/fixtures/`, samples in `data/samples/`.

### Changed

- `matplotlib` and `psutil` are runtime dependencies (used by `evaluate`).
- The runtime Docker image creates the Hugging Face cache directory owned by the non-root
  user so the named volume is writable.
