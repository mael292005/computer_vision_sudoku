"""Tests de la génération de chiffres synthétiques."""

import numpy as np
import pytest

from sudoku_vision.synthetic import (
    TEMPLATE_FONT_CANDIDATES,
    RenderParams,
    available_fonts,
    is_plausible_digit,
    make_dataset,
    make_templates,
    random_params,
    render_cell,
)

FONTS = available_fonts()
needs_fonts = pytest.mark.skipif(not FONTS, reason="aucune police installée")


@needs_fonts
def test_render_cell_draws_dark_digit_on_white() -> None:
    cell = render_cell(RenderParams(FONTS[0], 7))
    assert cell.dtype == np.uint8
    assert cell.max() == 255
    assert cell.min() < 100


@needs_fonts
def test_augmentations_produce_valid_cells() -> None:
    rng = np.random.default_rng(0)
    for digit in range(1, 10):
        params = random_params(rng, FONTS, digit)
        cell = render_cell(
            RenderParams(params.font, digit, angle=5, thickness=1, blur=1.0, noise=10)
        )
        assert cell.shape == (64, 64)
    thin = render_cell(RenderParams(FONTS[0], 8, thickness=-1))
    assert thin.min() < 128


@needs_fonts
def test_make_dataset_is_balanced_and_reproducible() -> None:
    x1, y1 = make_dataset(5, FONTS, seed=3)
    x2, y2 = make_dataset(5, FONTS, seed=3)
    assert x1.shape[1:] == (28, 28)
    assert np.bincount(y1).tolist() == [5] * 9
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_make_dataset_needs_a_font() -> None:
    with pytest.raises(ValueError):
        make_dataset(1, [])


@pytest.mark.skipif(not available_fonts(TEMPLATE_FONT_CANDIDATES), reason="police Times absente")
def test_templates_have_one_image_per_digit() -> None:
    templates = make_templates(available_fonts(TEMPLATE_FONT_CANDIDATES)[0])
    assert templates.shape == (9, 28, 28)
    assert templates.dtype == np.float32


def test_flat_shapes_are_not_plausible_digits() -> None:
    flat = np.zeros((28, 28), dtype=np.uint8)
    flat[12:16, 4:24] = 255
    tall = np.zeros((28, 28), dtype=np.uint8)
    tall[4:24, 12:16] = 255
    assert not is_plausible_digit(flat)
    assert not is_plausible_digit(np.zeros((28, 28), dtype=np.uint8))
    assert is_plausible_digit(tall)
