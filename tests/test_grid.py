"""Tests de la détection et du redressement de la grille."""

import numpy as np
import pytest

from sudoku_vision.config import DEFAULT_VISION
from sudoku_vision.grid import (
    GridNotFoundError,
    detect_grid,
    find_grid_corners,
    load_image,
    order_corners,
    threshold_image,
    to_gray,
    warp_grid,
)
from tests.samples import Sample


def test_grid_is_found_on_every_screenshot(sample: Sample) -> None:
    detection = detect_grid(load_image(sample.path))
    size = DEFAULT_VISION.warped_size
    assert detection.warped.shape == (size, size)


def test_detected_grid_is_square_and_large(sample: Sample) -> None:
    image = load_image(sample.path)
    corners = detect_grid(image).corners
    width = corners[1, 0] - corners[0, 0]
    height = corners[3, 1] - corners[0, 1]
    assert abs(width - height) / width < 0.02
    assert width > 0.2 * image.shape[1], "le logo WEB SUDOKU ne doit pas être pris pour la grille"


def test_order_corners_handles_any_input_order() -> None:
    shuffled = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    expected = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    np.testing.assert_array_equal(order_corners(shuffled), expected)


def test_synthetic_grid_is_detected() -> None:
    image = np.full((400, 600), 255, dtype=np.uint8)
    image[50:350, 150:450] = 0
    image[55:345, 155:445] = 255
    corners = find_grid_corners(threshold_image(image))
    assert np.allclose(corners[0], [150, 50], atol=8)
    assert np.allclose(corners[2], [449, 349], atol=8)


def test_blank_image_raises() -> None:
    blank = np.full((300, 300), 255, dtype=np.uint8)
    with pytest.raises(GridNotFoundError):
        find_grid_corners(threshold_image(blank))


def test_non_square_shape_is_rejected() -> None:
    image = np.full((400, 800), 255, dtype=np.uint8)
    image[100:300, 50:750] = 0
    image[105:295, 55:745] = 255
    with pytest.raises(GridNotFoundError):
        find_grid_corners(threshold_image(image))


def test_warp_produces_requested_size() -> None:
    image = np.zeros((200, 300), dtype=np.uint8)
    corners = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    assert warp_grid(image, corners, 90).shape == (90, 90)


def test_to_gray_keeps_gray_images() -> None:
    gray = np.zeros((10, 10), dtype=np.uint8)
    assert to_gray(gray) is gray


def test_missing_file_raises(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError):
        load_image("n_existe_pas.png")
