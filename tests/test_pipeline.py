"""Tests du pipeline complet, de la saisie automatique et de l'évaluation."""

from pathlib import Path

import numpy as np
import pytest

from sudoku_vision.actuator import fill_grid
from sudoku_vision.capture import ScreenShot
from sudoku_vision.digits import CnnClassifier, Prediction
from sudoku_vision.evaluate import degrade, evaluate, load_annotations
from sudoku_vision.grid import cell_centers, load_image
from sudoku_vision.pipeline import ReadingError, run, save_debug
from sudoku_vision.solver import solve
from tests.samples import DATA_DIR, Sample


class FakeController:
    """Enregistre les actions au lieu de bouger la vraie souris."""

    def __init__(self) -> None:
        self.actions: list[tuple[str, object]] = []

    def click(self, x: float, y: float) -> None:
        self.actions.append(("click", (x, y)))

    def type_digit(self, digit: int) -> None:
        self.actions.append(("type", digit))


class AlwaysOne:
    """Classifieur volontairement faux : lit des 1 partout."""

    name = "toujours-1"

    def predict(self, digits: list[np.ndarray]) -> list[Prediction]:
        return [Prediction(1, 1.0) for _ in digits]


def test_pipeline_solves_every_screenshot(sample: Sample) -> None:
    result = run(load_image(sample.path), CnnClassifier())
    assert result.puzzle == sample.grid
    assert result.solution == solve(sample.grid)
    assert set(result.timings) == {"grille", "cases", "chiffres", "solveur"}


def test_misread_digits_raise_reading_error(sample: Sample) -> None:
    with pytest.raises(ReadingError):
        run(load_image(sample.path), AlwaysOne())


def test_save_debug_writes_four_images(sample: Sample, tmp_path: Path) -> None:
    image = load_image(sample.path)
    written = save_debug(run(image, CnnClassifier()), image, tmp_path)
    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_cell_centers_fall_inside_their_cells() -> None:
    corners = np.array([[100, 50], [550, 50], [550, 500], [100, 500]], dtype=np.float32)
    centers = cell_centers(corners)
    assert centers.shape == (9, 9, 2)
    np.testing.assert_allclose(centers[0, 0], [125, 75], atol=0.01)
    np.testing.assert_allclose(centers[8, 8], [525, 475], atol=0.01)


def test_fill_grid_clicks_only_empty_cells(sample: Sample) -> None:
    solution = solve(sample.grid)
    centers = np.zeros((9, 9, 2), dtype=np.float32)
    controller = FakeController()
    count = fill_grid(sample.grid, solution, centers, controller)
    empty = sum(value == 0 for row in sample.grid for value in row)
    assert count == empty
    assert len(controller.actions) == 2 * empty
    assert controller.actions[0][0] == "click"
    assert controller.actions[1][0] == "type"


def test_fill_grid_types_the_solution_at_the_right_place(sample: Sample) -> None:
    result = run(load_image(sample.path), CnnClassifier())
    controller = FakeController()
    fill_grid(result.puzzle, result.solution, result.centers, controller)
    _, (x, y) = controller.actions[0]
    first = next((r, c) for r in range(9) for c in range(9) if sample.grid[r][c] == 0)
    np.testing.assert_allclose((x, y), result.centers[first], atol=0.01)
    assert controller.actions[1] == ("type", result.solution[first[0]][first[1]])


def test_fill_grid_rejects_wrong_centers_shape() -> None:
    grid = [[0] * 9 for _ in range(9)]
    with pytest.raises(ValueError):
        fill_grid(grid, grid, np.zeros((3, 2), dtype=np.float32), FakeController())


def test_screenshot_converts_to_mouse_coordinates() -> None:
    shot = ScreenShot(np.zeros((10, 10, 3), np.uint8), left=100, top=20, scale=0.5)
    points = np.array([[200.0, 40.0]], dtype=np.float32)
    np.testing.assert_allclose(shot.to_screen(points), [[150.0, 30.0]])


def test_evaluate_reports_perfect_scores() -> None:
    samples = load_annotations(DATA_DIR)
    report = evaluate(samples, CnnClassifier())
    assert report.images == report.grids_found == report.solved == len(samples)
    assert report.cells_correct == report.cells_total == 81 * len(samples)
    assert report.digit_accuracy == 1.0
    assert report.mean_ms > 0
    assert len(report.lines()) == 6


def test_evaluate_counts_misses(tmp_path: Path) -> None:
    blank = tmp_path / "vide.png"
    import cv2

    cv2.imwrite(str(blank), np.full((200, 200, 3), 255, np.uint8))
    (tmp_path / "grids.json").write_text(
        '{"vide.png": {"level": "Vide", "grid": ["' + "." * 81 + '"]}}', encoding="utf-8"
    )
    report = evaluate(load_annotations(tmp_path), AlwaysOne())
    assert report.images == 1
    assert report.grids_found == 0
    assert report.digit_accuracy == 0.0
    assert report.mean_ms == 0.0


def test_load_annotations_can_filter_by_site() -> None:
    every = load_annotations(DATA_DIR)
    websudoku = load_annotations(DATA_DIR, site="websudoku.com")
    assert 0 < len(websudoku) < len(every)
    assert {sample.site for sample in websudoku} == {"websudoku.com"}
    assert load_annotations(DATA_DIR, site="inconnu") == []


def test_degrade_resizes_and_blurs() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[40:60, 90:110] = 255
    small = degrade(image, scale=0.5)
    assert small.shape == (50, 100, 3)
    blurred = degrade(image, blur=2.0)
    assert blurred.shape == image.shape
    assert 0 < blurred[50, 89, 0] < 255
    assert degrade(image) is image or np.array_equal(degrade(image), image)


def test_cnn_survives_strong_degradation() -> None:
    """Le CNN reste fiable sur une grille 4 fois plus petite et floue."""
    websudoku = load_annotations(DATA_DIR, site="websudoku.com")
    report = evaluate(websudoku, CnnClassifier(), scale=0.25, blur=1.0)
    assert report.degraded
    assert "échelle 0.25" in report.lines()[0]
    assert report.digit_accuracy == 1.0
