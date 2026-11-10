"""Génère les figures du README à partir du jeu de test.

Usage :
    python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sudoku_vision.cells import filled_mask, split_cells
from sudoku_vision.debug import crop_around, draw_corners, draw_filled_mask
from sudoku_vision.grid import detect_grid, load_image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "data" / "websudoku_evil.png"
OUTPUT = ROOT / "docs" / "images" / "pipeline_vision.png"
PANEL_HEIGHT = 420
TITLE_HEIGHT = 50
GAP = 24


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    """Met une image à la hauteur commune et ajoute un titre au-dessus."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = PANEL_HEIGHT / image.shape[0]
    resized = cv2.resize(image, (round(image.shape[1] * scale), PANEL_HEIGHT))
    header = np.full((TITLE_HEIGHT, resized.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    return np.vstack([header, resized])


def main() -> None:
    image = load_image(SOURCE)
    detection = detect_grid(image)
    mask = filled_mask(split_cells(detection.warped))

    outlined = crop_around(draw_corners(image, detection.corners), detection.corners)
    panels = [
        _panel(outlined, "1. Grille detectee"),
        _panel(255 - crop_around(detection.threshold, detection.corners), "2. Seuillage adaptatif"),
        _panel(draw_filled_mask(detection.warped, mask), "3. Redressee + cases pleines"),
    ]
    spacer = np.full((PANEL_HEIGHT + TITLE_HEIGHT, GAP, 3), 255, dtype=np.uint8)
    figure = panels[0]
    for panel in panels[1:]:
        figure = np.hstack([figure, spacer, panel])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), figure)
    print(f"Figure écrite : {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
