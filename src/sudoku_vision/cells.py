"""Découpe de la grille redressée en 81 cases et détection des cases vides.

Une fois la grille redressée en carré, chaque case occupe exactement
``taille / 9`` pixels de côté. On rogne une marge sur chaque bord pour
retirer les lignes de la grille, qui fausseraient la suite.

Pour savoir si une case est vide, on compte la proportion de pixels sombres
(« encre ») : une case vide est presque entièrement claire, une case avec un
chiffre en contient nettement plus.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sudoku_vision.config import DEFAULT_VISION, GRID_SIZE, VisionConfig
from sudoku_vision.grid import Image, to_gray

INK_LEVEL = 128
"""En dessous de ce niveau de gris (0 = noir, 255 = blanc), un pixel est de l'encre."""


def split_cells(warped: Image, config: VisionConfig = DEFAULT_VISION) -> list[list[Image]]:
    """Découpe la grille redressée en 9x9 cases, marges rognées.

    Returns:
        Une liste de 9 lignes de 9 images de cases, en niveaux de gris.

    Raises:
        ValueError: si l'image n'est pas carrée.
    """
    gray = to_gray(warped)
    height, width = gray.shape[:2]
    if height != width:
        raise ValueError(f"la grille redressée doit être carrée, reçu {width}x{height}")

    step = height / GRID_SIZE
    margin = round(step * config.cell_margin_ratio)
    cells: list[list[Image]] = []
    for r in range(GRID_SIZE):
        row: list[Image] = []
        for c in range(GRID_SIZE):
            y0, y1 = round(r * step), round((r + 1) * step)
            x0, x1 = round(c * step), round((c + 1) * step)
            row.append(gray[y0 + margin : y1 - margin, x0 + margin : x1 - margin])
        cells.append(row)
    return cells


def ink_ratio(cell: Image) -> float:
    """Proportion de pixels « encre » dans une case (entre 0 et 1)."""
    if cell.size == 0:
        return 0.0
    return float(np.count_nonzero(to_gray(cell) < INK_LEVEL)) / cell.size


def is_empty(cell: Image, config: VisionConfig = DEFAULT_VISION) -> bool:
    """Indique si une case ne contient pas de chiffre."""
    return ink_ratio(cell) < config.empty_cell_ink_ratio


def filled_mask(
    cells: list[list[Image]], config: VisionConfig = DEFAULT_VISION
) -> npt.NDArray[np.bool_]:
    """Tableau 9x9 de booléens : ``True`` là où la case contient un chiffre."""
    return np.array([[not is_empty(cell, config) for cell in row] for row in cells], dtype=bool)
