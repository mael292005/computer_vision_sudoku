"""Reconnaissance des chiffres dans les cases de la grille.

Deux classifieurs sont disponibles, avec la même interface :

- :class:`TemplateClassifier` : **template matching**. On compare le chiffre à
  un modèle de référence de chaque chiffre (1 à 9) et on garde le plus
  ressemblant. Simple, rapide, sans entraînement, mais sensible à la police.
- :class:`CnnClassifier` : un petit **réseau de neurones convolutif**,
  entraîné avec PyTorch sur des chiffres synthétiques (plusieurs polices,
  déformations aléatoires), puis exporté au format ONNX. L'inférence passe
  par le module ``cv2.dnn`` d'OpenCV : PyTorch n'est pas nécessaire pour
  utiliser le bot, seulement pour réentraîner le modèle.

Les deux travaillent sur la même entrée normalisée, au format MNIST : une
image 28x28, chiffre blanc sur fond noir, centré dans une boîte de 20x20.
Aucun des deux n'a vu les captures de websudoku.com pendant sa construction :
le jeu de test reste une évaluation honnête.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from sudoku_vision.cells import ink_mask
from sudoku_vision.config import DEFAULT_VISION, GRID_SIZE, VisionConfig
from sudoku_vision.grid import Image

DIGIT_BOX = 20
"""Côté de la boîte dans laquelle le chiffre est mis à l'échelle (comme MNIST)."""

FloatArray = npt.NDArray[np.float32]


def _asset_path(name: str) -> Path:
    return Path(str(resources.files("sudoku_vision") / "assets" / name))


TEMPLATES_PATH = _asset_path("digit_templates.npz")
CNN_PATH = _asset_path("digits_cnn.onnx")


def extract_digit(cell: Image, size: int = DEFAULT_VISION.digit_size) -> Image | None:
    """Isole le chiffre d'une case et le normalise au format MNIST.

    Étapes : binarisation avec un seuil propre à la case (encre en blanc),
    suppression des petits débris et des traits collés aux bords, recadrage
    serré sur le chiffre, mise à l'échelle dans une boîte de 20x20 en gardant
    les proportions, puis centrage dans une image ``size`` x ``size``.

    Returns:
        L'image normalisée, ou ``None`` si la case ne contient pas de chiffre.
    """
    binary = np.where(ink_mask(cell), 255, 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    keep = np.zeros_like(binary)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        is_line = touches_border and (w > 0.8 * width or h > 0.8 * height)
        if area < 0.002 * binary.size or is_line:
            continue
        keep[labels == label] = 255

    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return None
    y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
    if (y1 - y0) / height < DEFAULT_VISION.min_digit_height_ratio:
        return None  # une poussière ou un reflet, pas un chiffre
    x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
    digit = keep[y0:y1, x0:x1]

    box = round(DIGIT_BOX * size / 28)
    scale = box / max(digit.shape)
    new_w = max(1, round(digit.shape[1] * scale))
    new_h = max(1, round(digit.shape[0] * scale))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


@dataclass(frozen=True)
class Prediction:
    """Chiffre reconnu et confiance associée (entre 0 et 1)."""

    digit: int
    confidence: float


class DigitClassifier(Protocol):
    """Interface commune des classifieurs de chiffres."""

    name: str

    def predict(self, digits: list[Image]) -> list[Prediction]:
        """Reconnaît une liste d'images normalisées (sortie de :func:`extract_digit`)."""
        ...


def _softmax(scores: FloatArray) -> FloatArray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float32)


class TemplateClassifier:
    """Template matching : corrélation normalisée avec un modèle par chiffre."""

    name = "template"

    def __init__(self, templates: FloatArray) -> None:
        """
        Args:
            templates: tableau (9, 28, 28) des modèles des chiffres 1 à 9.
        """
        if templates.shape[0] != GRID_SIZE:
            raise ValueError("il faut exactement un modèle par chiffre de 1 à 9")
        flat = templates.reshape(GRID_SIZE, -1).astype(np.float32)
        self._templates = self._normalize(flat)

    @classmethod
    def load(cls, path: Path = TEMPLATES_PATH) -> TemplateClassifier:
        """Charge les modèles fournis avec le package."""
        with np.load(path) as data:
            return cls(np.asarray(data["templates"], dtype=np.float32))

    @staticmethod
    def _normalize(vectors: FloatArray) -> FloatArray:
        centered = vectors - vectors.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return np.asarray(centered / np.maximum(norms, 1e-6), dtype=np.float32)

    def predict(self, digits: list[Image]) -> list[Prediction]:
        if not digits:
            return []
        vectors = self._normalize(np.stack(digits).reshape(len(digits), -1).astype(np.float32))
        scores = vectors @ self._templates.T  # corrélation entre -1 et 1
        best = scores.argmax(axis=1)
        return [
            Prediction(digit=int(i) + 1, confidence=float(max(scores[k, i], 0.0)))
            for k, i in enumerate(best)
        ]


class CnnClassifier:
    """Petit CNN exporté en ONNX, exécuté avec ``cv2.dnn`` (sans PyTorch)."""

    name = "cnn"

    def __init__(self, path: Path = CNN_PATH) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"modèle introuvable : {path}")
        self._net = cv2.dnn.readNetFromONNX(str(path))

    def predict(self, digits: list[Image]) -> list[Prediction]:
        if not digits:
            return []
        batch = (np.stack(digits).astype(np.float32) / 255.0)[:, None, :, :]
        self._net.setInput(batch)
        logits = np.asarray(self._net.forward(), dtype=np.float32).reshape(len(digits), -1)
        probs = _softmax(logits)
        best = probs.argmax(axis=1)
        return [
            Prediction(digit=int(i) + 1, confidence=float(probs[k, i])) for k, i in enumerate(best)
        ]


def load_classifier(name: str) -> DigitClassifier:
    """Charge un classifieur par son nom : ``"cnn"`` ou ``"template"``."""
    if name == "cnn":
        return CnnClassifier()
    if name == "template":
        return TemplateClassifier.load()
    raise ValueError(f"classifieur inconnu : {name!r} (choix : cnn, template)")


@dataclass(frozen=True)
class GridReading:
    """Résultat de la lecture des chiffres d'une grille."""

    grid: list[list[int]]
    """Grille 9x9 lue, 0 pour les cases vides."""

    confidence: npt.NDArray[np.float32]
    """Confiance 9x9 de chaque lecture (1.0 pour les cases vides)."""


def read_digits(
    cells: list[list[Image]],
    classifier: DigitClassifier,
    config: VisionConfig = DEFAULT_VISION,
) -> GridReading:
    """Lit les 81 cases : 0 pour une case vide, sinon le chiffre reconnu."""
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    confidence = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    positions: list[tuple[int, int]] = []
    images: list[Image] = []
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            digit = extract_digit(cell, config.digit_size)
            if digit is not None:
                positions.append((r, c))
                images.append(digit)

    for (r, c), prediction in zip(positions, classifier.predict(images), strict=True):
        grid[r][c] = prediction.digit
        confidence[r, c] = prediction.confidence
    return GridReading(grid=grid, confidence=confidence)
