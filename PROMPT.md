# Prompt: Build the initial version of `codeplag`

You are a senior ML engineer and technical writer. Build the first working version of **codeplag**, a source-code plagiarism detector, and produce the academic report that goes with it. Work inside the existing repository described below. Read `AGENTS.md` first and follow it.

## 1. Goal

Given two source code files, output a similarity verdict (is this plausibly plagiarism?) using **two independent detectors plus a combiner**:

1. **Semantic tree analyzer (no ML):** parse both files into ASTs with `tree-sitter`, normalize them, and compare them with tree edit distance (`zss`) plus structural features (subtree hashing, node-type n-gram overlap).
2. **Neural model:** a siamese network over a pretrained code encoder (default `microsoft/unixcoder-base`, loaded via `transformers`) that outputs a similarity score from two code embeddings. Fine-tune it on labeled clone/plagiarism pairs.
3. **Combiner:** fuses both scores (start with a logistic regression or weighted average, calibrated on a validation split) into a final score in `[0.0, 1.0]` and a human-readable report that points to the matching code regions.

The project also serves as a university assignment, so the code must be reproducible and every number in the report must come from a real run.

## 2. Existing repository and constraints

- Python **3.12**, venv at `.venv/`, `src/` layout, package name `codeplag`.
- Already scaffolded (many files are empty placeholders, fill them in rather than creating parallel files):

```
src/codeplag/{core,ast_analysis,neural,similarity,data,cli}/
  ast_analysis/{parser,normalizer,tree_distance}.py
  neural/{model,dataset,train,infer}.py
  similarity/{combiner,report}.py
  core/{config,logging}.py
  cli/main.py                      # Typer app, entry point `codeplag`
tests/{unit,integration,fixtures}/
configs/default.yaml   docs/{architecture,usage,api}/   scripts/{train,evaluate}.sh
docker/{Dockerfile,Dockerfile.dev}   docker-compose.yml   pyproject.toml   Makefile
data/{raw,processed,samples}/   models/checkpoints/
```

- Dependencies are managed in `pyproject.toml` (torch CPU build, transformers, tree-sitter + python/java/javascript grammars, zss, scikit-learn, networkx, pydantic, pyyaml, typer, tqdm). Add new ones there only if justified.
- Tooling: ruff (line length 100), mypy, pytest with coverage, pre-commit. All must pass.
- Hardware assumption: **CPU-only laptop** unless I say otherwise. Training must be feasible on CPU (small subset, few epochs, frozen lower layers or a distilled/smaller encoder if needed). Make device configurable (`CODEPLAG_DEVICE`).
- Do not commit data, checkpoints, `.env`, or `.venv`.

## 3. Dataset

Select **one primary dataset** and justify the choice. Evaluate these candidates and pick the best fit, or propose a better one with the same evidence:

- **IBM Project CodeNet** (many solutions per problem, multiple languages, GitHub)
- **POJ-104** (104 problems, ~500 C/C++ programs each; standard for code clone/semantic similarity)
- **BigCloneBench** (Java clone pairs with labels)
- **SOCO / source code reuse corpora**

Requirements:

- Build pairs: **positives** = two solutions to the same problem (semantic clones) **and** synthetically plagiarized variants of one file; **negatives** = solutions to different problems. Balance classes.
- Generate plagiarism transformations in `src/codeplag/data/` with a documented, seeded generator: identifier renaming, comment/whitespace changes, statement reordering where safe, dead-code insertion, loop-type swaps (for/while), expression rewrites. Record which transformation produced each positive pair.
- **Avoid leakage:** split train/val/test **by problem or source file**, never by pair.
- Ship a tiny curated sample in `data/samples/` and fixtures in `tests/fixtures/` so tests run offline. A script `scripts/prepare_data.sh` (or a CLI command) downloads and processes the real dataset into `data/processed/`.
- Document: source and URL, license, number of records and variables, column/field descriptions, class balance, cleaning steps, and why it fits this problem. Verify all figures from the downloaded data instead of quoting them from memory.

## 4. Implementation requirements

**ast_analysis**
- `parser.py`: language detection by extension, tree-sitter parsing for Python/Java/JavaScript, graceful error on syntax errors.
- `normalizer.py`: strip comments and docstrings, canonicalize identifiers and literals (alpha-renaming by scope order), optionally canonicalize equivalent constructs. Document each rule.
- `tree_distance.py`: normalized tree edit distance converted to a similarity in `[0,1]`, plus subtree-hash overlap. Include a size guard or approximation for large trees so it does not blow up (document complexity).

**neural**
- `model.py`: siamese architecture (shared encoder, pooling, projection, cosine similarity or MLP head on `[u, v, |u-v|, u*v]`).
- `dataset.py`: PyTorch `Dataset` over the pair files, with tokenization and max-length truncation.
- `train.py`: contrastive or binary cross-entropy training, seeded, early stopping, checkpoints to `models/checkpoints/`, metrics logged per epoch, config from `configs/default.yaml`.
- `infer.py`: load checkpoint, embed files (with caching), return score.

**similarity**
- `combiner.py`: fuse AST and neural scores; fit and persist the combiner on the validation split; expose a configurable decision threshold.
- `report.py`: output text/JSON with the final score, per-detector scores, verdict, and top matching subtrees or line ranges.

**cli** (thin layer, logic stays in the library):
- `codeplag compare A B [--json]`
- `codeplag scan DIR` (pairwise comparison of a directory; add a cheap prefilter such as embedding nearest neighbors or subtree-hash LSH so it is not purely O(n²))
- `codeplag train`, `codeplag evaluate`, `codeplag version`

**Quality**
- Type hints, Google-style docstrings, logging via `codeplag.core.logging`.
- Unit tests for each module (including tricky cases: renamed variables, reordered functions, syntax errors, empty files, different languages), one integration test for the full pipeline using fixtures. Mock the HF model in unit tests; never download models in tests.
- Docker: make sure `docker compose build codeplag` and `docker compose run --rm codeplag compare a.py b.py` work.

## 5. Evaluation (real numbers only)

Run and record, on the held-out test split:

- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, for: AST-only, neural-only, and combined.
- Threshold analysis, with the false-positive rate highlighted (a false accusation of plagiarism is costly).
- **Robustness:** performance per transformation type; performance on a deliberately harder set (heavier obfuscation, different but equivalent algorithms, same-template boilerplate code); performance on a language or problem set not seen in training.
- **Overfitting check:** train vs validation vs test curves.
- **Efficiency:** latency per pair, memory, throughput for `scan` on N files for several N, on the stated hardware.
- Save results to `docs/results/` as JSON/CSV and generate plots with a reproducible script.

If any result is poor, report it honestly and explain why. Do not tune on the test set. Do not fabricate or estimate metrics.

## 6. Report deliverable (write in Spanish)

Create `docs/report/informe.md` following exactly this structure and answering every bullet:

**Parte 1: Presentación del Dataset y del Modelo Desarrollado**
1. Selección del Dataset
   - Descripción breve: fuente, tamaño (registros y variables), características principales (columnas clave), propósito respecto al problema.
   - Razones de la selección: relevancia, calidad y limpieza, aplicabilidad al modelo.
2. Modelo Desarrollado y Resultados
   - Descripción del modelo: tipo de modelo (red neuronal siamesa + analizador de árbol semántico + combinador), herramientas y lenguajes (Python, PyTorch, transformers, tree-sitter, zss, Docker), pasos clave (limpieza, generación de pares, selección de características, entrenamiento, validación, evaluación) y resultados con tablas y gráficas.

**Parte 3: Evaluación del Modelo y Viabilidad Comercial**
1. Revisión del Modelo Desarrollado
   - Robustez y confiabilidad: desempeño ante datos nuevos y métricas obtenidas.
   - Limitaciones: sesgos en los datos, límites técnicos, sobreajuste, falsos positivos (código boilerplate, soluciones algorítmicamente obvias, código generado por IA).
   - Áreas de mejora: estrategias concretas y priorizadas para futuras iteraciones.
2. Viabilidad Comercial del Modelo
   - Desempeño: ¿es suficientemente preciso y confiable para uso comercial? Justifícalo con las métricas.
   - Escalabilidad: volumen de datos, costo computacional, estrategia de prefiltrado, despliegue con contenedores.
   - Aspectos éticos y legales: privacidad del código de estudiantes como dato personal, GDPR, Ley Federal de Protección de Datos Personales en Posesión de los Particulares (México), licencias de los datasets y del código, sesgo, transparencia y explicabilidad, y necesidad de revisión humana antes de cualquier sanción.
   - Fundamenta con fortalezas y debilidades. No des asesoría legal definitiva: indica qué debe validar un abogado.

Write code comments, docstrings, and `README.md` in English; write only the report in Spanish. Also update `README.md` (install, usage, Docker, results summary), `docs/architecture/overview.md` (with the pipeline diagram), `docs/usage/quickstart.md`, and `CHANGELOG.md`.

## 7. How to work

1. Start by printing a short plan and the dataset decision with evidence. Wait for no confirmation; proceed unless something blocks you.
2. Implement in this order, committing after each step with Conventional Commit messages: `ast_analysis` → data pipeline → `neural` → `combiner`/`report` → CLI → Docker verification → evaluation → report.
3. After each step run `ruff check`, `mypy src`, and `pytest`, and fix failures before moving on.
4. If a requirement is ambiguous or infeasible (CPU limits, dataset access), choose the simplest reasonable option, state the assumption in the report, and continue.
5. Finish with a summary listing: what works, exact commands to reproduce training/evaluation, measured results, known limitations, and next steps.

## 8. Definition of done

- `pip install -e ".[dev]"` then `pytest`, `ruff check`, `mypy src` all pass.
- `codeplag compare a.py b.py` returns a score and report, locally and via `docker compose run --rm codeplag ...`.
- `codeplag train` and `codeplag evaluate` run end to end from a clean clone following the README.
- `docs/results/` contains the real metrics and plots; `docs/report/informe.md` is complete and cites those numbers.
