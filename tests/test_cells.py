"""Tests de la découpe des cases et de la détection des cases vides."""

import numpy as np
import pytest

from sudoku_vision.cells import filled_mask, ink_ratio, is_empty, split_cells
from sudoku_vision.debug import crop_around, draw_corners, draw_filled_mask
from sudoku_vision.grid import detect_grid, load_image
from tests.samples import Sample


def test_split_gives_81_cells_of_equal_size() -> None:
    cells = split_cells(np.full((450, 450), 255, dtype=np.uint8))
    assert len(cells) == 9
    assert all(len(row) == 9 for row in cells)
    assert len({cell.shape for row in cells for cell in row}) == 1


def test_split_rejects_non_square_image() -> None:
    with pytest.raises(ValueError):
        split_cells(np.zeros((100, 200), dtype=np.uint8))


def test_blank_cell_is_empty() -> None:
    assert is_empty(np.full((30, 30), 255, dtype=np.uint8))


def test_cell_with_a_stroke_is_not_empty() -> None:
    cell = np.full((30, 30), 255, dtype=np.uint8)
    cell[5:25, 14:17] = 0  # un trait vertical, comme un « 1 »
    assert not is_empty(cell)


def test_ink_ratio_of_empty_array_is_zero() -> None:
    assert ink_ratio(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_filled_cells_match_ground_truth(sample: Sample) -> None:
    cells = split_cells(detect_grid(load_image(sample.path)).warped)
    expected = np.array([[value != 0 for value in row] for row in sample.grid])
    np.testing.assert_array_equal(filled_mask(cells), expected)


def test_debug_images_have_expected_shapes(sample: Sample) -> None:
    image = load_image(sample.path)
    detection = detect_grid(image)
    mask = filled_mask(split_cells(detection.warped))
    assert draw_corners(image, detection.corners).shape == image.shape
    assert draw_corners(detection.warped, detection.corners).ndim == 3
    assert draw_filled_mask(detection.warped, mask).shape == (*detection.warped.shape, 3)
    assert crop_around(image, detection.corners).size > 0
