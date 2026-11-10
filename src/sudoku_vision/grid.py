"""Détection de la grille de Sudoku dans une image, puis redressement.

Principe, en quatre étapes :

1. **Niveaux de gris + flou** : on ne garde que la luminosité et on lisse le
   bruit pour que les contours soient plus stables.
2. **Seuillage adaptatif** : chaque pixel est comparé à la moyenne de son
   voisinage, ce qui donne une image noir et blanc robuste aux variations
   d'éclairage (utile plus tard pour les photos).
3. **Contours** : parmi les formes trouvées, on garde la plus grande qui
   ressemble à un carré à 4 coins. C'est le cadre extérieur de la grille.
4. **Transformation de perspective** : on « déplie » ce quadrilatère en un
   carré parfait vu de face, de taille fixe. Les cases deviennent alors des
   carrés réguliers, faciles à découper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from sudoku_vision.config import DEFAULT_VISION, VisionConfig

Image = npt.NDArray[np.uint8]
Corners = npt.NDArray[np.float32]
"""Tableau (4, 2) des coins de la grille : haut-gauche, haut-droit, bas-droit, bas-gauche."""


class GridNotFoundError(RuntimeError):
    """Aucune forme ressemblant à une grille de Sudoku n'a été trouvée."""


@dataclass(frozen=True)
class GridDetection:
    """Résultat de la détection de la grille."""

    corners: Corners
    """Coins de la grille dans l'image d'origine (sert à convertir en coordonnées écran)."""

    warped: Image
    """Image de la grille redressée, carrée, en niveaux de gris."""

    threshold: Image
    """Image binaire utilisée pour trouver les contours (pour le debug)."""


def load_image(path: str | Path) -> Image:
    """Charge une image couleur (BGR) depuis le disque.

    Raises:
        FileNotFoundError: si le fichier n'existe pas ou n'est pas une image lisible.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"impossible de lire l'image : {path}")
    return np.asarray(image, dtype=np.uint8)


def to_gray(image: Image) -> Image:
    """Convertit une image couleur en niveaux de gris (ne change rien si déjà en gris)."""
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def threshold_image(gray: Image) -> Image:
    """Seuillage adaptatif inversé : les traits et chiffres deviennent blancs sur fond noir."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2,
    )
    return np.asarray(binary, dtype=np.uint8)


def order_corners(points: npt.ArrayLike) -> Corners:
    """Range 4 points dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche.

    Astuce classique : le coin haut-gauche a la plus petite somme x + y, le
    bas-droit la plus grande ; le haut-droit a la plus petite différence y - x,
    le bas-gauche la plus grande.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def find_grid_corners(
    binary: Image,
    min_area_ratio: float = 0.01,
    max_aspect_error: float = 0.2,
) -> Corners:
    """Trouve les 4 coins de la grille dans une image binaire.

    On parcourt les contours externes du plus grand au plus petit et on garde
    le premier qui, une fois simplifié, a 4 coins et une forme presque carrée.

    Args:
        binary: image issue de :func:`threshold_image`.
        min_area_ratio: surface minimale de la grille, en part de l'image.
        max_aspect_error: écart toléré entre largeur et hauteur (0.2 = 20 %).

    Raises:
        GridNotFoundError: si aucun contour ne convient.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * binary.shape[0] * binary.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        _, _, width, height = cv2.boundingRect(approx)
        if abs(width - height) / max(width, height) > max_aspect_error:
            continue
        return order_corners(approx)

    raise GridNotFoundError("aucune grille de Sudoku trouvée dans l'image")


def warp_grid(image: Image, corners: Corners, size: int) -> Image:
    """Redresse la zone délimitée par ``corners`` en un carré de ``size`` pixels."""
    target = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    return np.asarray(warped, dtype=np.uint8)


def detect_grid(image: Image, config: VisionConfig = DEFAULT_VISION) -> GridDetection:
    """Détecte la grille dans une image et renvoie sa version redressée.

    Raises:
        GridNotFoundError: si aucune grille n'est trouvée.
    """
    gray = to_gray(image)
    binary = threshold_image(gray)
    corners = find_grid_corners(binary)
    warped = warp_grid(gray, corners, config.warped_size)
    return GridDetection(corners=corners, warped=warped, threshold=binary)
