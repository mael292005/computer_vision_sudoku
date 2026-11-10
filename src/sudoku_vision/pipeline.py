"""Enchaînement complet : image -> grille -> cases -> chiffres -> solution.

Chaque étape est chronométrée, et le résultat garde toutes les images
intermédiaires pour pouvoir les exporter en mode debug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from sudoku_vision import solver
from sudoku_vision.cells import filled_mask, split_cells
from sudoku_vision.config import DEFAULT_VISION, VisionConfig
from sudoku_vision.debug import crop_around, draw_corners, draw_filled_mask, draw_solution
from sudoku_vision.digits import DigitClassifier, GridReading, read_digits
from sudoku_vision.grid import GridDetection, Image, cell_centers, detect_grid


class ReadingError(RuntimeError):
    """La grille lue n'est pas un Sudoku valide : un chiffre a probablement été mal lu."""


@dataclass
class PipelineResult:
    """Tout ce que le pipeline a produit, étape par étape."""

    detection: GridDetection
    mask: npt.NDArray[np.bool_]
    reading: GridReading
    solution: solver.Grid
    timings: dict[str, float] = field(default_factory=dict)
    """Durée de chaque étape, en millisecondes."""

    @property
    def puzzle(self) -> solver.Grid:
        """Grille de départ lue sur l'image."""
        return self.reading.grid

    @property
    def centers(self) -> npt.NDArray[np.float32]:
        """Centre de chaque case dans l'image d'origine, tableau (9, 9, 2)."""
        return cell_centers(self.detection.corners)


def run(
    image: Image,
    classifier: DigitClassifier,
    config: VisionConfig = DEFAULT_VISION,
) -> PipelineResult:
    """Exécute tout le pipeline sur une image.

    Raises:
        GridNotFoundError: si la grille n'est pas trouvée.
        ReadingError: si les chiffres lus forment une grille invalide ou insoluble.
    """
    timings: dict[str, float] = {}

    def timed(name: str, start: float) -> None:
        timings[name] = (time.perf_counter() - start) * 1000

    t = time.perf_counter()
    detection = detect_grid(image, config)
    timed("grille", t)

    t = time.perf_counter()
    cells = split_cells(detection.warped, config)
    mask = filled_mask(cells, config)
    timed("cases", t)

    t = time.perf_counter()
    reading = read_digits(cells, classifier, config)
    timed("chiffres", t)

    t = time.perf_counter()
    try:
        solution = solver.solve(reading.grid)
    except (solver.InvalidGridError, solver.UnsolvableGridError) as error:
        raise ReadingError(
            f"grille lue invalide ({error}). Grille lue :\n{solver.format_grid(reading.grid)}"
        ) from error
    timed("solveur", t)

    return PipelineResult(detection, mask, reading, solution, timings)


def save_debug(result: PipelineResult, image: Image, directory: Path) -> list[Path]:
    """Enregistre les images de chaque étape dans ``directory``.

    Returns:
        La liste des fichiers écrits.
    """
    directory.mkdir(parents=True, exist_ok=True)
    corners = result.detection.corners
    images = {
        "1_grille_detectee.png": crop_around(draw_corners(image, corners), corners),
        "2_seuillage.png": 255 - crop_around(result.detection.threshold, corners),
        "3_cases_pleines.png": draw_filled_mask(result.detection.warped, result.mask),
        "4_solution.png": draw_solution(result.detection.warped, result.puzzle, result.solution),
    }
    written = []
    for name, picture in images.items():
        path = directory / name
        cv2.imwrite(str(path), picture)
        written.append(path)
    return written
