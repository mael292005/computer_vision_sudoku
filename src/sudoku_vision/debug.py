"""Images de debug : montrer visuellement ce que voit chaque étape du pipeline.

Ces images servent à comprendre et régler le pipeline (option ``--debug``)
et à illustrer le README.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from sudoku_vision.config import GRID_SIZE
from sudoku_vision.grid import Corners, Image

GREEN = (60, 170, 60)
RED = (40, 40, 220)


def draw_corners(image: Image, corners: Corners) -> Image:
    """Dessine le contour détecté de la grille sur une copie de l'image."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=GREEN, thickness=6)
    for x, y in corners.astype(int):
        cv2.circle(out, (int(x), int(y)), 14, RED, thickness=-1)
    return np.asarray(out, dtype=np.uint8)


def draw_filled_mask(warped: Image, mask: npt.NDArray[np.bool_]) -> Image:
    """Colore en vert les cases pleines et en rouge les cases vides de la grille redressée."""
    out = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped.ndim == 2 else warped.copy()
    overlay = out.copy()
    step = out.shape[0] / GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            p0 = (int(c * step) + 2, int(r * step) + 2)
            p1 = (int((c + 1) * step) - 2, int((r + 1) * step) - 2)
            cv2.rectangle(overlay, p0, p1, GREEN if mask[r, c] else RED, thickness=-1)
    blended = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    return np.asarray(blended, dtype=np.uint8)


def draw_solution(
    warped: Image,
    puzzle: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
) -> Image:
    """Écrit en vert, sur la grille redressée, les chiffres trouvés par le solveur."""
    out = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped.ndim == 2 else warped.copy()
    step = out.shape[0] / GRID_SIZE
    scale = step / 40
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if puzzle[r][c] == 0:
                text = str(solution[r][c])
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                x = int(c * step + (step - w) / 2)
                y = int(r * step + (step + h) / 2)
                cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, GREEN, 2)
    return np.asarray(out, dtype=np.uint8)


def crop_around(image: Image, corners: Corners, padding: int = 60) -> Image:
    """Recadre l'image autour de la grille, avec une marge."""
    x0, y0 = np.maximum(corners.min(axis=0).astype(int) - padding, 0)
    x1, y1 = corners.max(axis=0).astype(int) + padding
    return image[y0:y1, x0:x1]
