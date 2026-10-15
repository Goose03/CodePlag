"""Tests for configuration loading, logging and the CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from codeplag import __version__
from codeplag.cli.main import app
from codeplag.core.config import load_settings
from codeplag.core.logging import get_logger, setup_logging
from tests.conftest import FIXTURES

runner = CliRunner()


def _config(tmp_path: Path) -> Path:
    cfg = {
        "paths": {
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "combiner_path": str(tmp_path / "ckpt" / "combiner.json"),
            "results_dir": str(tmp_path / "results"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "neural": {"device": "cpu"},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_settings_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setenv("CODEPLAG_DEVICE", "cuda")
    monkeypatch.setenv("CODEPLAG_MODEL_NAME", "some/model")
    monkeypatch.setenv("CODEPLAG_LOG_LEVEL", "DEBUG")
    s = load_settings(path)
    assert s.paths.results_dir == tmp_path / "results"
    assert s.neural.device == "cuda" and s.neural.model_name == "some/model"
    assert s.log_level == "DEBUG"


def test_load_settings_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")


def test_load_settings_from_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEPLAG_CONFIG", str(_config(tmp_path)))
    assert load_settings().paths.processed_dir == tmp_path / "processed"


def test_logging_helpers() -> None:
    setup_logging("WARNING")
    setup_logging("not-a-level")
    logger = get_logger("tests.x")
    assert logger.name == "codeplag.tests.x"
    assert logging.getLogger("codeplag").level == logging.INFO


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_compare_text_and_json(tmp_path: Path) -> None:
    cfg = str(_config(tmp_path))
    a, b = str(FIXTURES / "python" / "original.py"), str(FIXTURES / "python" / "renamed.py")
    text = runner.invoke(app, ["compare", a, b, "--no-neural", "--config", cfg])
    assert text.exit_code == 0, text.output
    assert "LIKELY PLAGIARISM" in text.stdout
    js = runner.invoke(app, ["compare", a, b, "--json", "--config", cfg])
    assert js.exit_code == 0, js.output
    data = json.loads(js.stdout)
    assert data["verdict"] == "likely_plagiarism" and data["neural_score"] is None


def test_compare_unsupported_language(tmp_path: Path) -> None:
    bad = tmp_path / "x.cpp"
    bad.write_text("int main() {}")
    result = runner.invoke(app, ["compare", str(bad), str(bad), "--config", str(_config(tmp_path))])
    assert result.exit_code == 2


def test_scan_text_and_json(tmp_path: Path) -> None:
    cfg = str(_config(tmp_path))
    directory = str(FIXTURES / "python")
    text = runner.invoke(app, ["scan", directory, "--no-neural", "--config", cfg, "--top", "3"])
    assert text.exit_code == 0, text.output
    assert "pairs compared" in text.stdout and "FLAG" in text.stdout
    js = runner.invoke(app, ["scan", directory, "--json", "--no-neural", "--config", cfg])
    data = json.loads(js.stdout)
    assert data["files"] == 6 and data["pairs"]


def test_train_without_data_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--config", str(_config(tmp_path)), "--epochs", "1"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def test_plots_without_results(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plots", "--config", str(_config(tmp_path))])
    assert result.exit_code == 0
