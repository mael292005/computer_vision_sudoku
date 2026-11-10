"""Tests de fumée : le package s'installe, s'importe et expose sa CLI."""

from typer.testing import CliRunner

import sudoku_vision
from sudoku_vision.cli import app
from sudoku_vision.config import BOX_SIZE, DEFAULT_VISION, GRID_SIZE

runner = CliRunner()


def test_version_is_exposed() -> None:
    assert isinstance(sudoku_vision.__version__, str)
    assert sudoku_vision.__version__


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert sudoku_vision.__version__ in result.stdout


def test_cli_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout


def test_grid_constants_are_consistent() -> None:
    assert BOX_SIZE * BOX_SIZE == GRID_SIZE


def test_default_vision_config_is_sane() -> None:
    assert DEFAULT_VISION.warped_size % GRID_SIZE == 0, "les cases doivent avoir une taille entière"
    assert 0 < DEFAULT_VISION.cell_margin_ratio < 0.5
    assert 0 < DEFAULT_VISION.empty_cell_ink_ratio < 1
