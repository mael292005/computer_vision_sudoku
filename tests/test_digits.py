"""Tests de l'extraction et de la reconnaissance des chiffres."""

import numpy as np
import pytest

from sudoku_vision.cells import split_cells
from sudoku_vision.digits import (
    CNN_PATH,
    CnnClassifier,
    TemplateClassifier,
    extract_digit,
    load_classifier,
    read_digits,
)
from sudoku_vision.grid import detect_grid, load_image
from tests.samples import Sample

CLASSIFIERS = ["template", "cnn"]


def _blank(size: int = 40) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def test_extract_digit_returns_none_for_empty_cell() -> None:
    assert extract_digit(_blank()) is None


def test_extract_digit_ignores_grid_lines_and_specks() -> None:
    cell = _blank()
    cell[0, :] = 0  # ligne de grille sur le bord
    cell[20, 20] = 0  # un pixel isolé
    assert extract_digit(cell) is None


def test_extract_digit_is_centered_mnist_format() -> None:
    cell = _blank()
    cell[5:15, 25:35] = 0  # tache décentrée en haut à droite
    digit = extract_digit(cell)
    assert digit is not None
    assert digit.shape == (28, 28)
    ys, xs = np.nonzero(digit)
    assert abs(xs.mean() - 13.5) < 1.5
    assert abs(ys.mean() - 13.5) < 1.5
    assert max(np.ptp(xs), np.ptp(ys)) + 1 <= 20


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_classifier_reads_every_screenshot(name: str, sample: Sample) -> None:
    cells = split_cells(detect_grid(load_image(sample.path)).warped)
    reading = read_digits(cells, load_classifier(name))
    assert reading.grid == sample.grid


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_classifier_handles_empty_batch(name: str) -> None:
    assert load_classifier(name).predict([]) == []


def test_confidence_is_one_for_empty_cells(sample: Sample) -> None:
    cells = split_cells(detect_grid(load_image(sample.path)).warped)
    reading = read_digits(cells, TemplateClassifier.load())
    empty = np.array(sample.grid) == 0
    assert np.all(reading.confidence[empty] == 1.0)
    assert np.all((reading.confidence > 0) & (reading.confidence <= 1))


def test_unknown_classifier_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_classifier("magie")


def test_template_classifier_needs_nine_templates() -> None:
    with pytest.raises(ValueError):
        TemplateClassifier(np.zeros((3, 28, 28), dtype=np.float32))


def test_missing_cnn_file_is_reported(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        CnnClassifier(tmp_path / "absent.onnx")


def test_cnn_model_is_shipped_with_the_package() -> None:
    assert CNN_PATH.is_file()
