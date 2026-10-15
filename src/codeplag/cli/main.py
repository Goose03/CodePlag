"""Command-line interface. Thin layer: parse arguments, call the library, format output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from codeplag import __version__
from codeplag.core.config import Settings, load_settings
from codeplag.core.logging import setup_logging

app = typer.Typer(
    help="codeplag: source-code plagiarism detection (AST analysis + neural similarity).",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (default: configs/default.yaml)."),
]


def _settings(config: Path | None, **overrides: Any) -> Settings:
    settings = load_settings(config)
    setup_logging(settings.log_level)
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, key = dotted.split("__")
        setattr(getattr(settings, section), key, value)
    return settings


@app.command()
def version() -> None:
    """Print the codeplag version."""
    typer.echo(f"codeplag {__version__}")


@app.command()
def compare(
    file_a: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="First file.")],
    file_b: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Second file.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print a JSON report.")] = False,
    no_neural: Annotated[bool, typer.Option("--no-neural", help="Use AST features only.")] = False,
    config: ConfigOption = None,
) -> None:
    """Compare two source files and print a similarity report."""
    from codeplag.ast_analysis.parser import UnsupportedLanguageError
    from codeplag.similarity.detector import PlagiarismDetector

    settings = _settings(config)
    detector = PlagiarismDetector(settings, use_neural=not no_neural)
    try:
        report, a, b = detector.compare_files(file_a, file_b)
    except UnsupportedLanguageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(report.to_json() if as_json else report.to_text(a.code, b.code))


@app.command()
def scan(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Directory.")],
    top: Annotated[int, typer.Option(help="Number of most similar pairs to show.")] = 10,
    prefilter: Annotated[
        str | None, typer.Option(help="minhash | embedding | none (default from config).")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    no_neural: Annotated[bool, typer.Option("--no-neural", help="Use AST features only.")] = False,
    config: ConfigOption = None,
) -> None:
    """Compare all supported files in a directory (with a candidate prefilter)."""
    from codeplag.similarity.detector import PlagiarismDetector

    settings = _settings(config, scan__prefilter=prefilter)
    result = PlagiarismDetector(settings, use_neural=not no_neural).scan(directory)
    shown = result.reports[:top]
    if as_json:
        payload = {
            "files": result.files,
            "total_pairs": result.total_pairs,
            "candidate_pairs": result.candidate_pairs,
            "prefilter": result.prefilter,
            "seconds": result.seconds,
            "flagged": len(result.flagged),
            "pairs": [r.to_dict() for r in shown],
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"{result.files} files, {result.candidate_pairs}/{result.total_pairs} pairs compared "
        f"(prefilter: {result.prefilter}) in {result.seconds:.2f}s; "
        f"{len(result.flagged)} flagged."
    )
    for r in shown:
        mark = "FLAG" if r.verdict == "likely_plagiarism" else "    "
        typer.echo(f"{mark} {r.score:.3f}  {r.file_a}  <->  {r.file_b}")


@app.command("prepare-data")
def prepare_data(
    no_download: Annotated[
        bool, typer.Option("--no-download", help="Fail if archives are missing.")
    ] = False,
    config: ConfigOption = None,
) -> None:
    """Download CodeNet subsets, clean them, split by problem and build labeled pairs."""
    from codeplag.data.pairs import prepare_dataset

    settings = _settings(config)
    stats = prepare_dataset(settings, download=not no_download)
    typer.echo(json.dumps(stats.get("pairs", {}), indent=2))


@app.command()
def train(
    epochs: Annotated[int | None, typer.Option(help="Override train.epochs.")] = None,
    max_pairs: Annotated[int | None, typer.Option(help="Override train.max_train_pairs.")] = None,
    device: Annotated[
        str | None, typer.Option(help="cpu or cuda (overrides CODEPLAG_DEVICE).")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Fine-tune the siamese neural model on the prepared pairs."""
    from codeplag.neural.train import train_model

    settings = _settings(
        config, train__epochs=epochs, train__max_train_pairs=max_pairs, neural__device=device
    )
    result = train_model(settings)
    val = result.best_val
    typer.echo(
        f"Best epoch {result.best_epoch}: val ROC-AUC {val.get('roc_auc', float('nan')):.4f}, "
        f"F1 {val.get('f1', float('nan')):.4f}; checkpoint {result.checkpoint} "
        f"({result.seconds / 60:.1f} min)"
    )


@app.command()
def evaluate(
    device: Annotated[str | None, typer.Option(help="cpu or cuda.")] = None,
    skip_efficiency: Annotated[
        bool, typer.Option(help="Skip latency/throughput benchmarks.")
    ] = False,
    no_neural: Annotated[
        bool, typer.Option("--no-neural", help="Evaluate AST features only.")
    ] = False,
    workers: Annotated[int, typer.Option(help="Processes for AST feature extraction.")] = 0,
    config: ConfigOption = None,
) -> None:
    """Fit the combiner on validation and evaluate everything on held-out data."""
    from codeplag.evaluation.evaluate import run_evaluation

    settings = _settings(config, neural__device=device)
    summary = run_evaluation(
        settings, use_neural=not no_neural, workers=workers, efficiency=not skip_efficiency
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def plots(config: ConfigOption = None) -> None:
    """Regenerate plots in docs/results/plots from the saved result files."""
    from codeplag.evaluation.plots import make_all_plots

    settings = _settings(config)
    for path in make_all_plots(settings.paths.results_dir):
        typer.echo(str(path))


if __name__ == "__main__":  # pragma: no cover
    app()
