"""Tests de fumée : le package s'installe, s'importe et expose sa CLI."""

from pathlib import Path

from typer.testing import CliRunner

import sudoku_vision
from sudoku_vision.cli import app
from sudoku_vision.config import BOX_SIZE, DEFAULT_VISION, GRID_SIZE
from tests.samples import DATA_DIR

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


def test_cli_solve_prints_solution() -> None:
    grid = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    result = runner.invoke(app, ["solve", "--grid", grid])
    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == "5 3 4 | 6 7 8 | 9 1 2"


def test_cli_solve_reports_invalid_grid() -> None:
    result = runner.invoke(app, ["solve", "--grid", "123"])
    assert result.exit_code == 1
    assert "Erreur" in result.output


def test_cli_solve_needs_exactly_one_source() -> None:
    assert runner.invoke(app, ["solve"]).exit_code == 1
    both = runner.invoke(app, ["solve", "--grid", "1", "--image", "x.png"])
    assert both.exit_code == 1


def test_cli_solve_image_with_debug(tmp_path: Path) -> None:
    image = DATA_DIR / "websudoku_easy.png"
    result = runner.invoke(
        app, ["solve", "--image", str(image), "--model", "template", "--debug", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Solution" in result.stdout
    assert "Temps" in result.stdout
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_cli_solve_image_reports_missing_file() -> None:
    result = runner.invoke(app, ["solve", "--image", "absent.png"])
    assert result.exit_code == 1
    assert "Erreur" in result.output


def test_cli_benchmark_prints_report() -> None:
    args = ["benchmark", "--data", str(DATA_DIR), "-m", "template", "--site", "websudoku.com"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Chiffres lus         : 115/115" in result.stdout
    assert "websudoku.com" in result.stdout


def test_cli_benchmark_accepts_degradation_options() -> None:
    result = runner.invoke(
        app,
        ["benchmark", "--data", str(DATA_DIR), "-m", "cnn", "--scale", "0.5", "--blur", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "échelle 0.5, flou 1" in result.stdout


def test_cli_benchmark_reports_missing_data(tmp_path: Path) -> None:
    result = runner.invoke(app, ["benchmark", "--data", str(tmp_path)])
    assert result.exit_code == 1


def test_grid_constants_are_consistent() -> None:
    assert BOX_SIZE * BOX_SIZE == GRID_SIZE


def test_default_vision_config_is_sane() -> None:
    assert DEFAULT_VISION.warped_size % GRID_SIZE == 0, "les cases doivent avoir une taille entière"
    assert 0 < DEFAULT_VISION.cell_margin_ratio < 0.5
    assert 0 < DEFAULT_VISION.empty_cell_ink_ratio < 1
