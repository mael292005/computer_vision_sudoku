"""Génère les figures du README à partir du jeu de test.

Usage :
    python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sudoku_vision.cells import filled_mask, split_cells
from sudoku_vision.debug import crop_around, draw_corners, draw_filled_mask, draw_solution
from sudoku_vision.digits import TEMPLATES_PATH, extract_digit
from sudoku_vision.evaluate import load_annotations
from sudoku_vision.grid import detect_grid, load_image
from sudoku_vision.solver import solve
from sudoku_vision.synthetic import available_fonts, make_dataset

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
SOURCE = ROOT / "tests" / "data" / "websudoku_evil.png"
SAMPLES = load_annotations(ROOT / "tests" / "data")
PANEL_HEIGHT = 420
TITLE_HEIGHT = 50
GAP = 24
WHITE = 255


def _panel(image: np.ndarray, title: str, height: int = PANEL_HEIGHT) -> np.ndarray:
    """Met une image à la hauteur voulue et ajoute un titre au-dessus."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = height / image.shape[0]
    resized = cv2.resize(
        image, (round(image.shape[1] * scale), height), interpolation=cv2.INTER_AREA
    )
    header = np.full((TITLE_HEIGHT, resized.shape[1], 3), WHITE, dtype=np.uint8)
    cv2.putText(header, title, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    return np.vstack([header, resized])


def _hstack(panels: list[np.ndarray]) -> np.ndarray:
    spacer = np.full((panels[0].shape[0], GAP, 3), WHITE, dtype=np.uint8)
    figure = panels[0]
    for panel in panels[1:]:
        figure = np.hstack([figure, spacer, panel])
    return figure


def _digit_row(digits: list[np.ndarray]) -> np.ndarray:
    """Aligne 9 chiffres 28x28 (blanc sur noir) en une bande, agrandis."""
    tiles = [cv2.copyMakeBorder(d, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=60) for d in digits]
    row = np.hstack(tiles)
    return cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def vision_steps() -> None:
    image = load_image(SOURCE)
    detection = detect_grid(image)
    mask = filled_mask(split_cells(detection.warped))
    outlined = crop_around(draw_corners(image, detection.corners), detection.corners)
    figure = _hstack(
        [
            _panel(outlined, "1. Grille detectee"),
            _panel(255 - crop_around(detection.threshold, detection.corners), "2. Seuillage"),
            _panel(draw_filled_mask(detection.warped, mask), "3. Redressee + cases pleines"),
        ]
    )
    cv2.imwrite(str(IMAGES / "pipeline_vision.png"), figure)


def digits_comparison() -> None:
    """Modèles du template matching, vrais chiffres extraits, exemples synthétiques."""
    with np.load(TEMPLATES_PATH) as data:
        model_digits = [np.clip(t, 0, 255).astype(np.uint8) for t in data["templates"]]

    real: dict[int, np.ndarray] = {}
    for sample in SAMPLES:
        cells = split_cells(detect_grid(load_image(sample.path)).warped)
        for r in range(9):
            for c in range(9):
                value = sample.grid[r][c]
                if value and value not in real:
                    digit = extract_digit(cells[r][c])
                    if digit is not None:
                        real[value] = digit

    x, y = make_dataset(3, available_fonts(), seed=7)
    synthetic = [x[int(np.flatnonzero(y == d)[2])] for d in range(9)]

    rows = [
        ("Modeles (police Liberation Serif)", model_digits),
        ("Vrais chiffres extraits des captures", [real[d] for d in range(1, 10)]),
        ("Exemples synthetiques pour le CNN", synthetic),
    ]
    blocks = []
    for title, digits in rows:
        strip = cv2.cvtColor(_digit_row(digits), cv2.COLOR_GRAY2BGR)
        header = np.full((40, strip.shape[1], 3), WHITE, dtype=np.uint8)
        cv2.putText(header, title, (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        blocks += [header, strip, np.full((12, strip.shape[1], 3), WHITE, dtype=np.uint8)]
    cv2.imwrite(str(IMAGES / "digits.png"), np.vstack(blocks))


def solution_figure() -> None:
    sample = next(s for s in SAMPLES if s.path.name == SOURCE.name)
    detection = detect_grid(load_image(SOURCE))
    solved = draw_solution(detection.warped, sample.grid, solve(sample.grid))
    cv2.imwrite(str(IMAGES / "solution.png"), solved)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    vision_steps()
    digits_comparison()
    solution_figure()
    print(f"Figures écrites dans {IMAGES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
