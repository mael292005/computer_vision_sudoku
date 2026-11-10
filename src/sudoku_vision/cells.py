"""Découpe de la grille redressée en 81 cases et détection des cases vides.

Une fois la grille redressée en carré, chaque case occupe exactement
``taille / 9`` pixels de côté. On rogne une marge sur chaque bord pour
retirer les lignes de la grille, qui fausseraient la suite.

Pour savoir si une case est vide, on compte la proportion de pixels « encre ».
Le seuil qui sépare l'encre du fond est calculé pour chaque case, à mi-chemin
entre son pixel le plus sombre et son pixel le plus clair : un chiffre flou ou
gris clair reste ainsi détecté. Une case sans contraste est vide d'office.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sudoku_vision.config import DEFAULT_VISION, GRID_SIZE, VisionConfig
from sudoku_vision.grid import Image, to_gray

MIN_CONTRAST = 60
"""Écart minimal entre le pixel le plus sombre et le plus clair pour qu'il y ait de l'encre."""


def ink_mask(cell: Image) -> npt.NDArray[np.bool_]:
    """Masque des pixels « encre » d'une case, avec un seuil propre à la case.

    Le seuil est placé à mi-chemin entre le pixel le plus sombre et le plus
    clair. Si la case n'a presque pas de contraste, elle n'a pas d'encre.
    """
    gray = to_gray(cell)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    darkest, lightest = int(np.min(gray)), int(np.max(gray))
    if lightest - darkest < MIN_CONTRAST:
        return np.zeros(gray.shape, dtype=bool)
    return np.asarray(gray < (darkest + lightest) / 2, dtype=bool)


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
    return float(np.count_nonzero(ink_mask(cell))) / cell.size


def is_empty(cell: Image, config: VisionConfig = DEFAULT_VISION) -> bool:
    """Indique si une case ne contient pas de chiffre."""
    return ink_ratio(cell) < config.empty_cell_ink_ratio


def filled_mask(
    cells: list[list[Image]], config: VisionConfig = DEFAULT_VISION
) -> npt.NDArray[np.bool_]:
    """Tableau 9x9 de booléens : ``True`` là où la case contient un chiffre."""
    return np.array([[not is_empty(cell, config) for cell in row] for row in cells], dtype=bool)
