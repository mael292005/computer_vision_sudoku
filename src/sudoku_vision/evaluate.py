"""Évaluation du pipeline sur un jeu de test annoté.

Le jeu de test est un dossier qui contient des images et un fichier
``grids.json`` avec, pour chaque image, la grille attendue (voir
``tests/data/grids.json``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sudoku_vision.cells import filled_mask, split_cells
from sudoku_vision.digits import DigitClassifier, read_digits
from sudoku_vision.grid import GridNotFoundError, Image, detect_grid, load_image
from sudoku_vision.solver import Grid, InvalidGridError, UnsolvableGridError, parse_grid, solve


@dataclass(frozen=True)
class Annotated:
    """Une image du jeu de test et sa grille attendue."""

    path: Path
    name: str
    grid: Grid


def load_annotations(directory: Path) -> list[Annotated]:
    """Lit ``grids.json`` dans ``directory``."""
    raw = json.loads((directory / "grids.json").read_text(encoding="utf-8"))
    return [
        Annotated(directory / name, entry.get("level", name), parse_grid("".join(entry["grid"])))
        for name, entry in raw.items()
        if not name.startswith("_")
    ]


def degrade(image: Image, scale: float = 1.0, blur: float = 0.0) -> Image:
    """Dégrade une image pour tester la robustesse.

    Args:
        scale: facteur de réduction, pour simuler un navigateur moins zoomé
            (0.5 = grille deux fois plus petite à l'écran).
        blur: écart-type d'un flou gaussien, en pixels (0 = aucun).
    """
    out = image
    if scale != 1.0:
        out = np.asarray(
            cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), np.uint8
        )
    if blur > 0:
        out = np.asarray(cv2.GaussianBlur(out, (0, 0), blur), np.uint8)
    return out


@dataclass
class Report:
    """Résultats agrégés sur tout le jeu de test."""

    classifier: str
    scale: float = 1.0
    blur: float = 0.0
    images: int = 0
    grids_found: int = 0
    cells_total: int = 0
    cells_correct: int = 0
    digits_total: int = 0
    digits_correct: int = 0
    solved: int = 0
    total_ms: float = 0.0

    @property
    def digit_accuracy(self) -> float:
        return self.digits_correct / self.digits_total if self.digits_total else 0.0

    @property
    def degraded(self) -> bool:
        return self.scale != 1.0 or self.blur > 0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.images if self.images else 0.0

    def lines(self) -> list[str]:
        """Résumé lisible du rapport."""
        return [
            f"Classifieur          : {self.classifier}"
            + (f" (échelle {self.scale:g}, flou {self.blur:g})" if self.degraded else ""),
            f"Grilles détectées    : {self.grids_found}/{self.images}",
            f"Cases vides/pleines  : {self.cells_correct}/{self.cells_total}",
            f"Chiffres lus         : {self.digits_correct}/{self.digits_total} "
            f"({self.digit_accuracy:.1%})",
            f"Grilles résolues     : {self.solved}/{self.images}",
            f"Temps moyen          : {self.mean_ms:.0f} ms par image",
        ]


def evaluate(
    samples: list[Annotated],
    classifier: DigitClassifier,
    scale: float = 1.0,
    blur: float = 0.0,
) -> Report:
    """Fait passer chaque image dans le pipeline et compare à la vérité terrain.

    ``scale`` et ``blur`` dégradent les images avant l'évaluation (voir :func:`degrade`).
    """
    report = Report(classifier=classifier.name, scale=scale, blur=blur)
    for sample in samples:
        report.images += 1
        expected = np.array(sample.grid)
        start = time.perf_counter()
        try:
            detection = detect_grid(degrade(load_image(sample.path), scale, blur))
        except GridNotFoundError:
            continue
        report.grids_found += 1
        cells = split_cells(detection.warped)
        mask = filled_mask(cells)
        reading = read_digits(cells, classifier)
        try:
            solve(reading.grid)
            solved = True
        except (InvalidGridError, UnsolvableGridError):
            solved = False
        report.total_ms += (time.perf_counter() - start) * 1000

        read = np.array(reading.grid)
        has_digit = expected > 0
        report.cells_total += expected.size
        report.cells_correct += int((mask == has_digit).sum())
        report.digits_total += int(has_digit.sum())
        report.digits_correct += int((read[has_digit] == expected[has_digit]).sum())
        report.solved += int(solved and bool((read == expected).all()))
    return report
